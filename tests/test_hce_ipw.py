import random
import statistics
import pytest
from hce.contract import parse
from hce.ipw import estimate, hard_regressions
from hce.selectors import SelectionRequest, build

TASKS = [f"t{i:02d}" for i in range(42)]


def _world(seed=0, drift=0.0):
    """A synthetic training set with a known true mean@k after the change.

    The per-task effect is deliberately heterogeneous. A uniform effect makes
    every sample of a stratum sum to the same value, so the estimator returns
    the true mean exactly and with zero variance -- correct, but it exercises
    nothing. `drift` moves the non-activated tasks, which is what an audit
    stratum exists to detect and what its absence biases away.
    """
    rng = random.Random(seed)
    p_hist = {t: rng.choice([0.0, 1 / 3, 2 / 3, 1.0]) for t in TASKS}
    mechs = {t: {"max_command_output_tokens": 3000 if i < 14 else 500,
                 "has_python_traceback": i % 7 == 0} for i, t in enumerate(TASKS)}
    effects = [0.0, 1 / 3, 2 / 3, 1 / 3, 0.0, 1.0]
    truth = {}
    for i, t in enumerate(TASKS):
        if i < 14:
            truth[t] = max(0.0, min(1.0, p_hist[t] + effects[i % len(effects)]))
        else:
            truth[t] = max(0.0, min(1.0, p_hist[t] + drift))
    return p_hist, mechs, truth


def _contract():
    return parse({"iteration": 2, "changes": [{
        "id": "chg-1", "files": ["m.py"], "evaluation_contract": {
            "scope": "conditional", "expected_effect": "improve",
            "activation_predicate": {"max_command_output_tokens": ">2000"}}}]},
        iteration=2)


def _run_once(selector, p_hist, mechs, truth, seed, *, min_audit=3, budget=17,
              split=None, audit_enabled=True):
    req = SelectionRequest(
        iteration=2, all_tasks=TASKS, p_hist=p_hist, mechanisms=mechs,
        variance={t: p_hist[t] * (1 - p_hist[t]) for t in TASKS},
        contract=_contract(), budget_tasks=budget, k=3,
        rng=random.Random(seed), min_audit=min_audit,
        split=split or {"improvement": 0.19, "regression": 0.41, "audit": 0.40},
        audit_enabled=audit_enabled)
    res = build(selector).select(req)
    scores = {t: truth[t] for t in res.selected}
    return estimate(all_tasks=TASKS, p_hist=p_hist, scores=scores,
                    strata=res.strata, pi=res.pi)


def test_full_set_reduces_to_plain_mean():
    p_hist, mechs, truth = _world()
    est = _run_once("full", p_hist, mechs, truth, 1)
    assert est.s_hat == pytest.approx(sum(truth.values()) / len(TASKS), abs=1e-12)
    assert est.se == 0.0                     # census: nothing was sampled away


def test_hce_is_unbiased_over_repeated_draws():
    p_hist, mechs, truth = _world()
    true_mean = sum(truth.values()) / len(TASKS)
    draws = [_run_once("hce", p_hist, mechs, truth, s).s_hat for s in range(4000)]
    mc_se = statistics.stdev(draws) / len(draws) ** 0.5
    assert abs(statistics.mean(draws) - true_mean) < 3 * mc_se, (
        f"mean={statistics.mean(draws):.4f} true={true_mean:.4f} mc_se={mc_se:.5f}")


def test_removing_the_audit_stratum_introduces_bias():
    """The experiment that justifies the audit stratum existing at all.

    With min_audit=0 the non-activated tasks become unreachable. They contribute
    only their history, so any change in them is invisible and the estimate is
    biased by exactly their unobserved delta.
    """
    # The non-activated tasks regress: only the audit stratum can see it.
    p_hist, mechs, truth = _world(drift=-1 / 3)
    true_mean = sum(truth.values()) / len(TASKS)
    with_audit = statistics.mean(
        _run_once("hce", p_hist, mechs, truth, s, min_audit=3).s_hat for s in range(1500))
    without = statistics.mean(
        _run_once("hce", p_hist, mechs, truth, s, audit_enabled=False).s_hat
        for s in range(1500))
    assert abs(with_audit - true_mean) < 0.02, (with_audit, true_mean)
    # The regression in the non-activated tasks is invisible, so the estimate
    # stays near the (higher) historical baseline for that whole segment.
    assert without - true_mean > 0.05, (without, true_mean)


def test_infra_loss_shrinks_the_realised_sample_not_the_pool():
    p_hist, mechs, truth = _world()
    req = SelectionRequest(
        iteration=2, all_tasks=TASKS, p_hist=p_hist, mechanisms=mechs,
        variance={t: 0.2 for t in TASKS}, contract=_contract(), budget_tasks=17,
        k=3, rng=random.Random(3), min_audit=3)
    res = build("hce").select(req)
    lost = res.selected[0]
    scores = {t: truth[t] for t in res.selected if t != lost}
    est = estimate(all_tasks=TASKS, p_hist=p_hist, scores=scores,
                   strata=res.strata, pi=res.pi)
    assert est.n_dropped_infra == 1 and est.dropped_tasks == [lost]
    assert est.n_measured == len(res.selected) - 1


def test_single_observation_stratum_reports_conservative_se_not_zero():
    p_hist, mechs, truth = _world()
    req = SelectionRequest(
        iteration=2, all_tasks=TASKS, p_hist=p_hist, mechanisms=mechs,
        variance={t: 0.2 for t in TASKS}, contract=_contract(), budget_tasks=17,
        k=3, rng=random.Random(5), min_audit=1)
    res = build("hce").select(req)
    est = estimate(all_tasks=TASKS, p_hist=p_hist,
                   scores={t: truth[t] for t in res.selected},
                   strata=res.strata, pi=res.pi)
    if any(len(s.taken) == 1 and s.pi is not None and s.pi < 1 for s in res.strata.values()):
        assert est.se_is_conservative and est.se > 0


def test_hard_regressions_only_count_census_tasks():
    p_hist = {"a": 1.0, "b": 1.0, "c": 0.5}
    scores = {"a": 0.0, "b": 0.0, "c": 0.0}
    pi = {"a": 1.0, "b": 0.3, "c": 1.0}
    # b is sampled, so its verdict carries a weight and cannot be an observation;
    # c was never reliably passing, so dropping to zero is not a regression.
    assert hard_regressions(scores=scores, p_hist=p_hist, pi=pi) == ["a"]

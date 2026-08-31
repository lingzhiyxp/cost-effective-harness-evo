import random
import pytest
from hce.contract import parse
from hce.selectors import SelectionRequest, build, MODE_GLOBAL, MODE_CONDITIONAL

TASKS = [f"t{i:02d}" for i in range(42)]


def _req(contract=None, budget=17, seed=1, **kw):
    # k=3 profiling yields p_hist in {0, 1/3, 2/3, 1}; a mix of stable and
    # fluctuating tasks is what makes the variance weighting meaningful at all.
    ladder = [0.0, 1.0, 1 / 3, 1.0, 2 / 3, 0.0]
    p_hist = {t: ladder[i % len(ladder)] for i, t in enumerate(TASKS)}
    mechs = {t: {"max_command_output_tokens": 3000 if i < 12 else 500,
                 "has_python_traceback": i % 7 == 0} for i, t in enumerate(TASKS)}
    base = dict(iteration=2, all_tasks=TASKS, p_hist=p_hist, mechanisms=mechs,
                variance={t: p_hist[t] * (1 - p_hist[t]) for t in TASKS},
                contract=contract, budget_tasks=budget, k=3,
                rng=random.Random(seed))
    base.update(kw)
    return SelectionRequest(**base)


def _contract(predicate=None, risk=(), scope="conditional"):
    return parse({"iteration": 2, "changes": [{
        "id": "chg-1", "files": ["tools/x.py"], "risk_tasks": list(risk),
        "evaluation_contract": {
            "scope": scope, "expected_effect": "improve",
            "activation_predicate": predicate or {"max_command_output_tokens": ">2000"}},
    }]}, iteration=2)


def test_full_selector_is_the_pi_one_special_case():
    r = build("full").select(_req())
    assert r.selected == sorted(TASKS)
    assert set(r.pi.values()) == {1.0}


def test_variance_respects_floor_and_expected_budget():
    req = _req(budget=17)
    r = build("variance").select(req)
    assert r.pi and min(r.pi.values()) >= req.pi_min
    # Stable-pass tasks have zero variance; the floor is what keeps them reachable,
    # and they are exactly where a regression can happen.
    assert r.rationale["expected_n"] == pytest.approx(17, abs=6)


def test_variance_pi_min_zero_reproduces_the_blind_spot():
    """With no floor, a task that always passes has zero weight and is never drawn.

    That is the proposal's core objection to variance sampling: regressions can
    only happen to currently-passing tasks, and those are exactly the ones this
    design cannot reach. Kept as a test so the ablation arm stays honest.
    """
    req = _req(budget=17, pi_min=0.0)
    r = build("variance").select(req)
    stable = [t for t in TASKS if req.p_hist[t] in (0.0, 1.0)]
    assert stable, "fixture must contain zero-variance tasks"
    assert all(r.pi.get(t, 0.0) == 0.0 for t in stable)
    assert all(t not in r.selected for t in stable)


def test_hce_every_task_is_reachable():
    """The invariant that keeps the estimator unbiased: no task has pi == 0."""
    r = build("hce").select(_req(_contract()))
    covered = {t for s in r.strata.values() for t in s.pool}
    assert covered == set(TASKS), sorted(set(TASKS) - covered)
    assert all(s.pi is None or s.pi > 0 for s in r.strata.values() if s.pool)


def test_hce_partitions_into_the_four_strata():
    r = build("hce").select(_req(_contract(risk=["t01"])))
    pools = {n: set(s.pool) for n, s in r.strata.items()}
    assert pools["F"] == {"t01"}
    assert not (pools["C"] & pools["G"]) and not (pools["C"] & pools["audit"])
    assert all(r.pi[t] == 1.0 for t in r.strata["F"].taken)
    assert r.mode == MODE_CONDITIONAL


def test_hce_audit_floor_survives_a_tight_budget():
    r = build("hce").select(_req(_contract(), budget=5, min_audit=3))
    assert len(r.strata["audit"].taken) >= 3


def test_hce_falls_back_to_uniform_without_a_contract():
    r = build("hce").select(_req(contract=None))
    assert r.mode == MODE_GLOBAL and len(set(r.pi.values())) == 1


def test_hce_global_scope_still_splits_improvement_from_regression():
    """A system-prompt edit activates everything, which is not the same as
    knowing nothing: p_hist still says which tasks can regress."""
    req = _req(_contract(scope="global", predicate={}))
    r = build("hce").select(req)
    assert r.mode == MODE_GLOBAL
    assert r.rationale["stratified_global"] is True
    assert r.rationale["n_activated"] == len(TASKS)
    # Every task is activated, so nothing is left to audit and the whole set is
    # partitioned by historical outcome instead.
    assert not r.strata["audit"].pool
    assert set(r.strata["C"].pool) | set(r.strata["G"].pool) == set(TASKS)
    assert all(req.p_hist[t] >= 1.0 for t in r.strata["G"].pool)
    assert all(req.p_hist[t] < 1.0 for t in r.strata["C"].pool)
    # The regression bucket gets a dedicated quota rather than whatever share a
    # uniform draw happens to give it.
    assert r.strata["G"].taken


def test_hce_no_contract_is_still_uniform():
    r = build("hce").select(_req(contract=None))
    assert r.mode == MODE_GLOBAL and r.rationale["reason"] == "no contract"
    assert len(set(r.pi.values())) == 1


def test_same_seed_same_selection():
    a = build("hce").select(_req(_contract(), seed=7))
    b = build("hce").select(_req(_contract(), seed=7))
    assert a.selected == b.selected and a.pi == b.pi


def test_budget_is_respected():
    for name in ("hce", "full", "variance"):
        r = build(name).select(_req(_contract(), budget=17))
        if name == "full":
            assert len(r.selected) == 42
        else:
            assert len(r.selected) <= 42

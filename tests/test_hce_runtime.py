"""End-to-end exercise of the loop's HCE path, with no harbor and no API calls."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from hce import runtime
from hce.profiles import TaskProfileDB

CORPUS = Path("/fs/nexus-projects/MeMas/projects/Harness-Evo/reset-free-coding-agent-harness/"
              "experiments/2026-08-29__20-58-50__tb2-train45-ahe-mini-medium/runs/"
              "iteration_001/input/benchmark/2026-08-29__20-59-16")
DATASET = Path("/fs/nexus-projects/MeMas/projects/Harness-Evo/"
               "reset-free-coding-agent-harness/dataset/tb2-42-train")

pytestmark = pytest.mark.skipif(not CORPUS.is_dir(), reason="verification corpus absent")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    measured = runtime.measure(CORPUS, max_iterations=300, max_context_tokens=200000)
    d = TaskProfileDB(tmp_path_factory.mktemp("p") / "task_profiles.json")
    runtime.commit_measurements(db=d, measured=measured, iteration=1,
                                fingerprint="seed", accepted=True)
    return d


def _manifest(predicate, predicted=(), files=("m.py",)):
    return {"iteration": 1, "changes": [{
        "id": "chg-1", "files": list(files), "predicted_fixes": list(predicted),
        "evaluation_contract": {"scope": "conditional", "expected_effect": "improve",
                                "activation_predicate": predicate}}]}


def _plan(tmp_path, db, manifest, **cfg):
    if manifest is not None:
        (tmp_path / "change_manifest.json").write_text(json.dumps(manifest))
    config = {"harbor": {"k": 3}, "hce": {"enabled": True, **cfg}}
    return runtime.plan_iteration(config=config, exp_dir=tmp_path, project_dir=tmp_path,
                                  iteration=2, db=db, all_tasks=sorted(db.tasks))


def test_measure_reproduces_the_audited_numbers(db):
    scores = [t["outcome"]["p_hist"] for t in db.tasks.values()]
    assert len(scores) == 45
    assert abs(sum(scores) / len(scores) - 0.5333) < 1e-3


def test_plan_selects_a_subset_and_covers_every_task(tmp_path, db):
    plan = _plan(tmp_path, db, _manifest({"has_python_traceback": True}))
    r = plan.record
    assert r["mode"] == "conditional"
    assert 0 < len(plan.task_names) < r["n_tasks_total"]
    covered = {t for s in r["strata"].values() for t in s["pool"]}
    assert covered == set(plan.all_tasks)          # every pi > 0
    assert set(r["p_hist_snapshot"]) == set(plan.all_tasks)


def test_impossible_claim_is_rejected_before_any_rollout(tmp_path, db):
    # Claims to fix tasks that its own predicate cannot reach.
    tiny = sorted(db.tasks)[:3]
    plan = _plan(tmp_path, db, _manifest({"max_command_output_tokens": ">100000"}, tiny))
    assert plan.pre_rejected
    assert "activation_predicate" in plan.falsification.reason


def test_global_file_forces_the_wider_budget(tmp_path, db):
    narrow = _plan(tmp_path, db, _manifest({"has_python_traceback": True}))
    wide = _plan(tmp_path, db, _manifest({"has_python_traceback": True},
                                         files=["systemprompt.md"]))
    assert wide.record["mode"] == "global"
    assert wide.record["budget_frac"] > narrow.record["budget_frac"]
    assert len(wide.task_names) > len(narrow.task_names)


def test_unparseable_contract_falls_back_without_crashing(tmp_path, db):
    plan = _plan(tmp_path, db, _manifest({"not_a_real_feature": ">1"}))
    assert plan.record["contract_error"] and "legal keys" in plan.record["contract_error"]
    assert plan.task_names                      # the iteration still runs


def test_gate_accepts_an_improvement_and_rejects_a_regression(tmp_path, db):
    plan = _plan(tmp_path, db, _manifest({"has_python_traceback": True}))
    better = {t: {"score": min(1.0, plan.p_hist[t] + 0.5), "n_pass": 3, "n_fail": 0,
                  "n_infra": 0, "agg": {}, "per_rollout": [], "mean_usd": None}
              for t in plan.task_names}
    est, decision = runtime.score_iteration(plan=plan, measured=better, s_ref=0.5333,
                                            config={"hce": {"enabled": True}})
    assert decision.accepted and est.s_hat > 0.5333

    worse = {t: {"score": 0.0, "n_pass": 0, "n_fail": 3, "n_infra": 0,
                 "agg": {}, "per_rollout": [], "mean_usd": None}
             for t in plan.task_names}
    _, bad = runtime.score_iteration(plan=plan, measured=worse, s_ref=0.5333,
                                     config={"hce": {"enabled": True}})
    assert not bad.accepted and bad.rule in ("G1_hard_regression", "G2_estimate")


def test_three_arms_are_interchangeable(tmp_path, db):
    manifest = _manifest({"has_python_traceback": True})
    for name in ("full", "variance", "hce"):
        plan = _plan(tmp_path, db, manifest, selector=name)
        measured = {t: {"score": plan.p_hist[t], "n_pass": 0, "n_fail": 0, "n_infra": 0,
                        "agg": {}, "per_rollout": [], "mean_usd": None}
                    for t in plan.task_names}
        est, _ = runtime.score_iteration(plan=plan, measured=measured, s_ref=None,
                                         config={"hce": {"enabled": True}})
        # Measuring exactly the history must reproduce the historical mean under
        # every design: the correction term is zero by construction.
        assert abs(est.s_hat - est.baseline_mean) < 1e-9, (name, est.s_hat)


def test_full_selector_estimate_equals_the_plain_mean(tmp_path, db):
    plan = _plan(tmp_path, db, _manifest({"has_python_traceback": True}), selector="full")
    truth = {t: (1.0 if i % 3 else 0.0) for i, t in enumerate(plan.task_names)}
    measured = {t: {"score": v, "n_pass": 0, "n_fail": 0, "n_infra": 0,
                    "agg": {}, "per_rollout": [], "mean_usd": None} for t, v in truth.items()}
    est, _ = runtime.score_iteration(plan=plan, measured=measured, s_ref=None,
                                     config={"hce": {"enabled": True}})
    assert abs(est.s_hat - sum(truth.values()) / len(truth)) < 1e-12


def test_profiling_iteration_is_a_census(tmp_path, db):
    """Iteration 1 must measure everything: p_hist defines the C/G partition,
    the estimator's baseline and the regression test, and an unmeasured task
    silently defaults to 0.0."""
    config = {"harbor": {"k": 3}, "hce": {"enabled": True}}
    plan = runtime.plan_iteration(config=config, exp_dir=tmp_path,
                                  project_dir=tmp_path, iteration=1, db=db,
                                  all_tasks=sorted(db.tasks))
    assert plan.record["is_profiling"] is True
    assert plan.record["budget_frac"] == 1.0
    assert len(plan.task_names) == len(db.tasks)
    assert set(plan.selection.pi.values()) == {1.0}


def test_iteration_after_profiling_samples(tmp_path, db):
    (tmp_path / "change_manifest.json").write_text(json.dumps(
        _manifest({"has_python_traceback": True})))
    config = {"harbor": {"k": 3}, "hce": {"enabled": True}}
    plan = runtime.plan_iteration(config=config, exp_dir=tmp_path,
                                  project_dir=tmp_path, iteration=2, db=db,
                                  all_tasks=sorted(db.tasks))
    assert plan.record["is_profiling"] is False
    assert len(plan.task_names) < len(db.tasks)


def test_profiling_metadata_is_stamped(tmp_path):
    """The profile database has to say which run built it, at what k."""
    fresh = TaskProfileDB(tmp_path / "p.json")
    runtime.commit_measurements(
        db=fresh, measured={"a": {"score": 1.0, "agg": {}, "per_rollout": [],
                                  "n_pass": 3, "n_fail": 0, "n_infra": 0, "mean_usd": 0.1}},
        iteration=1, fingerprint="fp", accepted=True, is_profiling=True, k=3,
        job_dir="runs/iteration_001/x")
    assert fresh.profiling == {"iteration": 1, "k": 3, "fingerprint": "fp",
                               "job_dir": "runs/iteration_001/x"}
    assert fresh.token_estimator == "chars_div_4"
    # A later, non-profiling iteration must not overwrite it.
    runtime.commit_measurements(
        db=fresh, measured={"a": {"score": 0.5, "agg": {}, "per_rollout": [],
                                  "n_pass": 1, "n_fail": 1, "n_infra": 0, "mean_usd": 0.1}},
        iteration=2, fingerprint="fp2", accepted=True, k=3)
    assert fresh.profiling["iteration"] == 1

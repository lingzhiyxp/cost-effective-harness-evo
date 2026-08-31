import subprocess
import pytest
from hce.contract import parse
from hce.falsify import check
from hce.gate import ACCEPT, REJECT, decide, revert_to
from hce.ipw import Estimate


def _est(s_hat, se=0.01):
    return Estimate(s_hat=s_hat, s_hajek=s_hat, se=se, se_is_conservative=False,
                    n_total=42, n_selected=17, n_measured=17, n_dropped_infra=0)


def test_hard_regression_vetoes_regardless_of_the_estimate():
    scores = {f"t{i}": 0.0 for i in range(3)}
    p_hist = {f"t{i}": 1.0 for i in range(3)}
    pi = {f"t{i}": 1.0 for i in range(3)}
    d = decide(estimate=_est(0.99), scores=scores, p_hist=p_hist, pi=pi,
               s_ref=0.50, max_hard_regressions=2)
    assert d.verdict == REJECT and d.rule == "G1_hard_regression"
    assert len(d.hard_regression_tasks) == 3


def test_sampled_tasks_do_not_count_as_hard_regressions():
    # pi < 1 means the verdict carries a sampling weight, so it is an estimate.
    d = decide(estimate=_est(0.6), scores={"a": 0.0, "b": 0.0, "c": 0.0},
               p_hist={"a": 1.0, "b": 1.0, "c": 1.0},
               pi={"a": 0.3, "b": 0.3, "c": 0.3}, s_ref=0.5, max_hard_regressions=0)
    assert d.verdict == ACCEPT


def test_estimate_veto_uses_a_band_of_max_tolerance_se():
    # A drop smaller than the measurement's own resolution is not a rejection.
    assert decide(estimate=_est(0.48, se=0.05), scores={}, p_hist={}, pi={},
                  s_ref=0.50, tolerance=0.02).verdict == ACCEPT
    assert decide(estimate=_est(0.40, se=0.01), scores={}, p_hist={}, pi={},
                  s_ref=0.50, tolerance=0.02).rule == "G2_estimate"


def test_no_reference_accepts():
    assert decide(estimate=_est(0.1), scores={}, p_hist={}, pi={},
                  s_ref=None).verdict == ACCEPT


def _contract(predicate, predicted, risk=()):
    return parse({"iteration": 2, "changes": [{
        "id": "chg-1", "files": ["m.py"], "predicted_fixes": list(predicted),
        "risk_tasks": list(risk),
        "evaluation_contract": {"scope": "conditional", "expected_effect": "improve",
                                "activation_predicate": predicate}}]}, iteration=2)


MECHS = {"big": {"max_command_output_tokens": 9000},
         "small": {"max_command_output_tokens": 2000},
         "tiny": {"max_command_output_tokens": 100}}


def test_falsification_rejects_a_claim_its_own_predicate_forbids():
    # Claims to fix `tiny`, but `tiny` never produced output over 8000 tokens.
    f = check(_contract({"max_command_output_tokens": ">8000"}, ["tiny", "small"]), MECHS)
    assert f.rejected and "activation_predicate" in f.reason
    assert "tiny" in f.reason


def test_falsification_passes_a_consistent_claim():
    assert not check(_contract({"max_command_output_tokens": ">8000"}, ["big"]),
                     MECHS).rejected


def test_falsification_rejects_an_unverifiable_predicate():
    f = check(_contract({"max_command_output_tokens": ">999999"}, []), MECHS)
    assert f.rejected and "cannot be verified" in f.reason


def test_revert_restores_the_tree_and_keeps_history(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    g = lambda *a: subprocess.run(["git", *a], cwd=ws, check=True,
                                  capture_output=True, text=True).stdout.strip()
    g("init", "-q")
    g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (ws / "keep.py").write_text("original\n")
    g("add", "-A"); g("commit", "-qm", "baseline"); g("tag", "iteration_1_before")

    # The agent edits a file, adds one, and deletes another.
    (ws / "keep.py").write_text("modified\n")
    (ws / "added.py").write_text("new\n")
    g("add", "-A"); g("commit", "-qm", "chg-1"); g("tag", "iteration_1")
    (ws / "untracked.txt").write_text("scratch\n")

    before_tree = g("rev-parse", "iteration_1_before^{tree}")
    sha = revert_to(ws, "iteration_1_before", reason="G1 hard regression", iteration=1)

    assert g("rev-parse", "HEAD^{tree}") == before_tree
    assert (ws / "keep.py").read_text() == "original\n"
    assert not (ws / "added.py").exists()
    assert not (ws / "untracked.txt").exists()
    assert g("status", "--porcelain") == ""
    # The tags that the attribution report and the snapshots key on still resolve.
    assert g("rev-parse", "iteration_1") and g("rev-parse", "iteration_1_before")
    assert g("rev-parse", "iteration_1_rejected") == sha
    assert "chg-1" in g("log", "--oneline")     # the rejection did not erase history

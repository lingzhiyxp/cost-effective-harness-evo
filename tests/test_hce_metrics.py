import json
import pytest
from hce.metrics import PASS, FAIL, INFRA, classify_trial, task_score, mean_at_k


def _trial(tmp_path, name, *, reward=None, exc_type=None, exception_txt=None):
    d = tmp_path / name
    (d / "verifier").mkdir(parents=True)
    if reward is not None:
        (d / "verifier" / "reward.txt").write_text(str(reward))
    (d / "result.json").write_text(json.dumps(
        {"exception_info": {"exception_type": exc_type} if exc_type else None}))
    if exception_txt:
        (d / "exception.txt").write_text(exception_txt)
    return d


def test_reward_decides_when_present(tmp_path):
    assert classify_trial(_trial(tmp_path, "a", reward=1.0))[0] == PASS
    assert classify_trial(_trial(tmp_path, "b", reward=0.0))[0] == FAIL
    # A partial reward is not a pass: the threshold is >= 1.0.
    assert classify_trial(_trial(tmp_path, "c", reward=0.9))[0] == FAIL


def test_outcome_errors_are_failures_not_infra(tmp_path):
    for exc in ("AgentTimeoutError", "VerifierTimeoutError", "RewardFileNotFoundError"):
        verdict, seen = classify_trial(_trial(tmp_path, f"t-{exc}", exc_type=exc))
        assert verdict == FAIL, exc
        assert seen == exc


def test_infrastructure_errors_are_excluded(tmp_path):
    verdict, seen = classify_trial(_trial(tmp_path, "e2b", exc_type="E2BSandboxError"))
    assert verdict == INFRA and seen == "E2BSandboxError"


def test_result_json_wins_over_exception_txt(tmp_path):
    # result.json is canonical; exception.txt is only the fallback.
    d = _trial(tmp_path, "both", exc_type="E2BSandboxError",
               exception_txt="AgentTimeoutError: ran out of time")
    assert classify_trial(d)[0] == INFRA
    d2 = _trial(tmp_path, "txtonly", exception_txt="E2BSandboxError: no sandbox")
    assert classify_trial(d2)[0] == INFRA


@pytest.mark.parametrize("verdicts,expected", [
    ([PASS, PASS, PASS], 1.0),
    ([FAIL, FAIL, FAIL], 0.0),
    ([PASS, FAIL, FAIL], 1 / 3),
    ([PASS, FAIL, INFRA], 0.5),      # infra leaves the denominator
    ([PASS, INFRA, INFRA], 1.0),
    ([INFRA, INFRA, INFRA], None),   # unmeasured, not zero
])
def test_task_score_denominator(verdicts, expected):
    assert task_score(verdicts) == expected


def test_mean_at_k_skips_unmeasured_tasks():
    out = mean_at_k({"a": [PASS, PASS], "b": [FAIL, FAIL], "c": [INFRA, INFRA]})
    assert out["mean_at_k"] == 0.5           # (1.0 + 0.0) / 2, c excluded
    assert out["n_measured"] == 2
    assert out["infra_only_tasks"] == ["c"]
    assert "c" not in out["per_task_score"]


def test_mean_at_k_all_infra_is_zero_not_crash():
    out = mean_at_k({"a": [INFRA]})
    assert out["mean_at_k"] == 0.0 and out["n_measured"] == 0


def test_ask_mode_hint_targets_the_models_that_need_it():
    """adb dispatches on two payload schemas and gpt-5.4-mini returns the wrong
    one for an ask request, losing that task's analysis entirely."""
    import evolve
    assert evolve._needs_ask_mode_hint({"llm": {"model": "gpt-5.4-mini"}})
    assert not evolve._needs_ask_mode_hint({"llm": {"model": "gpt-5.4"}})
    assert evolve._needs_ask_mode_hint({"llm": {"model": "gpt-5.4"}, "ask_mode_hint": "always"})
    assert not evolve._needs_ask_mode_hint({"llm": {"model": "gpt-5.4-mini"},
                                            "ask_mode_hint": "never"})
    assert evolve._needs_ask_mode_hint({"llm": {"model": "some-new-model"},
                                        "ask_hint_model_patterns": ["some-new"]})

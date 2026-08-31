"""Per-task scoring with an exception convention AHE's own arithmetic lacks.

`compute_stats` counts every exception in the denominator (`n = n_pass + n_fail +
n_exception`), while the evolve prompt tells the agent to ignore infrastructure
failures. Both cannot be right. The split below follows harbor's own
`--retry-exclude` default list, and the reasoning is in
`reset-free-coding-agent-harness/scripts/ahe/tb2_heldout_report.py`:

  * An agent or verifier that exhausted its own budget produced a *result*. The
    task was attempted and not solved. It belongs in the denominator at reward 0.
  * A sandbox that never started produced *no* result. Counting it as a failure
    charges the harness for the infrastructure's behaviour, and on a subset of a
    dozen tasks that noise is large enough to swamp the signal being measured.

Under subset evaluation the second case matters more than it does at full-set
scale: one lost sandbox out of twelve tasks moves the measured score by eight
points, and the IPW estimator would then weight that artefact by 1/pi.
"""

from __future__ import annotations

import json
from pathlib import Path

# Errors that mean "the attempt ran and did not succeed" -- a real outcome.
OUTCOME_ERRORS = frozenset({
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
})

PASS = "pass"
FAIL = "fail"
INFRA = "infra"


def classify_trial(trial_dir: Path) -> tuple[str, str]:
    """Return (verdict, exception_type) for one rollout directory.

    verdict is PASS / FAIL / INFRA. `result.json`'s `exception_info` is the
    canonical source -- it is present on every trial harbor writes -- and
    `exception.txt` is only the fallback for trials that predate it or that
    died before the result was written.
    """
    exc_type = ""
    result_src = trial_dir / "result.json"
    if result_src.exists():
        try:
            info = (json.loads(result_src.read_text(encoding="utf-8", errors="replace"))
                    or {}).get("exception_info")
            if isinstance(info, dict):
                exc_type = str(info.get("exception_type") or "")
        except (json.JSONDecodeError, OSError):
            pass

    if not exc_type:
        exception_src = trial_dir / "exception.txt"
        if exception_src.exists():
            text = exception_src.read_text(errors="replace").strip()
            first = text.splitlines()[-1] if text else ""
            exc_type = first.split(":", 1)[0].strip() or "Unknown"

    reward_src = trial_dir / "verifier" / "reward.txt"
    if reward_src.exists():
        try:
            return (PASS if float(reward_src.read_text().strip()) >= 1.0 else FAIL), exc_type
        except ValueError:
            return FAIL, exc_type

    if exc_type and exc_type not in OUTCOME_ERRORS:
        return INFRA, exc_type
    # No reward file and either no exception type or an outcome-class one:
    # the attempt ran and produced nothing gradeable, which is a failure.
    return FAIL, (exc_type or "Unknown")


def task_score(verdicts: list[str]) -> float | None:
    """mean@k for one task, with infrastructure failures left out entirely.

    Returns None when every rollout was an infrastructure failure: that task was
    not measured this iteration, which is different from having scored zero. The
    estimator treats it as unobserved and falls back to the task's history.
    """
    counted = [v for v in verdicts if v != INFRA]
    if not counted:
        return None
    return sum(1 for v in counted if v == PASS) / len(counted)


def mean_at_k(per_task_verdicts: dict[str, list[str]]) -> dict:
    """Macro-average of per-task mean@k over the tasks that were measured."""
    scores = {t: task_score(v) for t, v in per_task_verdicts.items()}
    measured = {t: s for t, s in scores.items() if s is not None}
    infra_only = sorted(t for t, s in scores.items() if s is None)
    return {
        "per_task_score": measured,
        "mean_at_k": (sum(measured.values()) / len(measured)) if measured else 0.0,
        "n_measured": len(measured),
        "n_infra_only": len(infra_only),
        "infra_only_tasks": infra_only,
    }

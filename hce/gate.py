"""Accepting or reverting a change, and the git surgery for the revert.

AHE leaves this to the evolve agent -- evolve.py's own comment says "rollback
decided by evolve agent". That is fine when every iteration is measured on the
whole training set and the agent can see everything. Under subset evaluation the
decision has to be made from an estimate with a known sampling design, which is
not something to delegate to a model reading a markdown table.

Two independent grounds for rejection, checked in this order:

  G1 counts hard regressions among the census tasks only. Those have pi = 1, so
     the count is an observation and not an estimate; it can veto on its own
     without any distributional argument. This is the signal that stays reliable
     when the estimate is noisy.
  G2 vetoes on the estimate itself, against a band of max(tolerance, se) so that
     a change is never rejected for a move smaller than the measurement's own
     resolution.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from hce.ipw import hard_regressions

ACCEPT = "accept"
REJECT = "reject"


@dataclass
class GateDecision:
    verdict: str
    rule: str
    reason: str
    s_hat: float | None = None
    s_ref: float | None = None
    se: float | None = None
    hard_regression_tasks: list[str] = field(default_factory=list)
    reverted_to: str = ""
    revert_commit: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict == ACCEPT

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "rule": self.rule, "reason": self.reason,
            "s_hat": self.s_hat, "s_ref": self.s_ref, "se": self.se,
            "hard_regression_tasks": self.hard_regression_tasks,
            "reverted_to": self.reverted_to, "revert_commit": self.revert_commit,
        }


def decide(*, estimate, scores: dict[str, float], p_hist: dict[str, float],
           pi: dict[str, float], s_ref: float | None,
           tolerance: float = 0.02, max_hard_regressions: int = 2,
           hard_floor: float = 0.0) -> GateDecision:
    regressions = hard_regressions(scores=scores, p_hist=p_hist, pi=pi,
                                   hard_floor=hard_floor)
    if len(regressions) > max_hard_regressions:
        return GateDecision(
            REJECT, "G1_hard_regression",
            f"{len(regressions)} reliably-passing tasks scored at the floor "
            f"(limit {max_hard_regressions}): {', '.join(regressions)}",
            s_hat=estimate.s_hat, s_ref=s_ref, se=estimate.se,
            hard_regression_tasks=regressions)

    if s_ref is not None:
        band = max(tolerance, estimate.se)
        if estimate.s_hat < s_ref - band:
            return GateDecision(
                REJECT, "G2_estimate",
                f"estimated mean@k {estimate.s_hat:.4f} is below the reference "
                f"{s_ref:.4f} by more than the band {band:.4f} "
                f"(tolerance {tolerance:.4f}, se {estimate.se:.4f})",
                s_hat=estimate.s_hat, s_ref=s_ref, se=estimate.se,
                hard_regression_tasks=regressions)

    return GateDecision(
        ACCEPT, "G3_accept",
        f"estimated mean@k {estimate.s_hat:.4f}"
        + (f" against reference {s_ref:.4f}" if s_ref is not None else " (no reference yet)")
        + f", {len(regressions)} hard regression(s)",
        s_hat=estimate.s_hat, s_ref=s_ref, se=estimate.se,
        hard_regression_tasks=regressions)


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workspace, check=True,
                          capture_output=True, text=True).stdout.strip()


def revert_to(workspace: Path, tag: str, *, reason: str, iteration: int) -> str:
    """Restore the workspace to `tag` with a new commit on top.

    Deliberately not `git reset --hard`. The evolve agent reads `git log` as part
    of its context, and the iteration tags are how the attribution report and the
    workspace snapshots line up; resetting would orphan those tags and erase the
    fact that a rejection happened. A forward commit whose tree equals the
    pre-change tree keeps the history honest and still puts the right files on
    disk. `git clean` is needed too -- restore does not remove files the agent
    added.
    """
    _git(workspace, "restore", f"--source={tag}", "--staged", "--worktree", "--", ".")
    _git(workspace, "clean", "-fd")
    _git(workspace, "add", "-A")
    if _git(workspace, "diff", "--cached", "--stat"):
        _git(workspace, "commit", "-m", f"hce: revert iteration {iteration} -- {reason}"[:200])
    try:
        _git(workspace, "tag", f"iteration_{iteration}_rejected")
    except subprocess.CalledProcessError:
        pass                       # a re-run of the same iteration; the tag stands
    return _git(workspace, "rev-parse", "HEAD")

"""Checks that can reject a change before a single rollout is paid for.

The contract gives the agent control over how much of the training set its own
change is measured on, which is an obvious incentive to write a narrow trigger.
These checks make that unprofitable: a predicate narrow enough to dodge
evaluation is also narrow enough to contradict the change's own claims, and
that contradiction is visible in data already on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hce.predicate import matches


@dataclass
class Falsification:
    verdict: str                     # "pass" | "reject"
    reason: str = ""
    checks: list[dict] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return self.verdict == "reject"


def check(contract, mechanisms: dict[str, dict], *,
          min_predicted_match_frac: float = 0.5,
          min_activation: int = 1) -> Falsification:
    """Test a contract against the profiles that predate the change."""
    checks: list[dict] = []

    for change in contract.changes:
        if change.scope == "global":
            checks.append({"change_id": change.change_id, "check": "scope",
                           "result": "skipped", "detail": "global scope, no predicate"})
            continue

        # F1: does the change's own trigger admit the tasks it claims to fix?
        claimed = change.predicted_fixes
        if claimed:
            hits = [t for t in claimed
                    if t in mechanisms and matches(change.activation_predicate,
                                                   mechanisms[t])[0]]
            unknown = [t for t in claimed if t not in mechanisms]
            checkable = [t for t in claimed if t in mechanisms]
            frac = (len(hits) / len(checkable)) if checkable else 1.0
            checks.append({
                "change_id": change.change_id, "check": "predicted_fixes_match",
                "result": "pass" if frac >= min_predicted_match_frac else "fail",
                "matched": hits, "claimed": claimed, "unknown": unknown,
                "fraction": round(frac, 3)})
            if checkable and frac < min_predicted_match_frac:
                misses = sorted(set(checkable) - set(hits))
                return Falsification("reject", checks=checks, reason=(
                    f"{change.change_id}: only {len(hits)}/{len(checkable)} of the tasks it "
                    f"claims to fix satisfy its own activation_predicate "
                    f"{change.activation_predicate}. Tasks that do not: {', '.join(misses)}. "
                    f"Either the trigger condition or the predicted fixes are wrong; "
                    f"the change cannot do what it says by the mechanism it states."))

    # F2: a predicate matching nothing describes a change nothing can verify.
    activated, _ = contract.activation(mechanisms)
    checks.append({"check": "activation_size", "n_activated": len(activated),
                   "n_tasks": len(mechanisms),
                   "result": "pass" if len(activated) >= min_activation else "fail"})
    if len(activated) < min_activation:
        return Falsification("reject", checks=checks, reason=(
            f"the activation predicate matches {len(activated)} of {len(mechanisms)} "
            f"profiled tasks, so this change cannot be verified on any of them. "
            f"Widen the trigger condition, or state the change as global in scope."))

    return Falsification("pass", checks=checks)

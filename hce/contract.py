"""Parsing and validating the Evaluation Contract the evolve agent must submit.

The contract rides inside the existing change_manifest.json rather than beside
it: `evaluate_changes` already reads `id`, `files`, `predicted_fixes` and
`risk_tasks` from each change, and those keep working untouched. Only the new
`evaluation_contract` object is ours.

Two things are decided here rather than trusted from the agent:

  * Scope. A change touching a globally visible file is global no matter what it
    declared. The agent has an incentive to under-declare -- a conditional scope
    is cheaper to verify -- and it also simply gets this wrong, so the file list
    decides and the declaration is only recorded.
  * Legality. An unparseable predicate is an error with a message, returned to
    the agent for one rewrite, rather than a silent fallback to full-set
    evaluation that would quietly erase the method being tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hce.mechanism import FEATURE_SCHEMA
from hce.predicate import PredicateError, activation_set, validate

SCOPE_CONDITIONAL = "conditional"
SCOPE_GLOBAL = "global"
DEFAULT_GLOBAL_FILES = ("systemprompt.md", "code_agent.yaml")


class ContractError(ValueError):
    """A contract that cannot drive selection. The message goes back to the agent."""


@dataclass
class ChangeContract:
    change_id: str
    scope: str
    mechanism: str
    activation_predicate: dict
    expected_effect: str
    files: list[str] = field(default_factory=list)
    predicted_fixes: list[str] = field(default_factory=list)
    risk_tasks: list[str] = field(default_factory=list)
    forced_global_reason: str = ""


@dataclass
class IterationContract:
    iteration: int
    changes: list[ChangeContract]

    @property
    def is_global(self) -> bool:
        return any(c.scope == SCOPE_GLOBAL for c in self.changes)

    @property
    def predicted_fixes(self) -> list[str]:
        return sorted({t for c in self.changes for t in c.predicted_fixes})

    @property
    def risk_tasks(self) -> list[str]:
        return sorted({t for c in self.changes for t in c.risk_tasks})

    def activation(self, mechanisms: dict[str, dict]) -> tuple[list[str], dict]:
        """Union of the per-change activation sets.

        A union, not an intersection: the iteration ships every change together,
        so a task any one of them can reach is a task this iteration can break.
        """
        if self.is_global:
            return sorted(mechanisms), {"mode": SCOPE_GLOBAL, "per_change": []}
        matched: set[str] = set()
        detail = []
        for change in self.changes:
            tasks, unknown = activation_set(change.activation_predicate, mechanisms)
            matched.update(tasks)
            detail.append({
                "change_id": change.change_id,
                "predicate": change.activation_predicate,
                "matched": tasks,
                "unknown_feature": sorted(unknown),
            })
        return sorted(matched), {"mode": SCOPE_CONDITIONAL, "per_change": detail}


def _global_by_files(files: list[str], global_files: tuple[str, ...]) -> str:
    for path in files:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        for marker in global_files:
            if name == marker or str(path).endswith(marker):
                return str(path)
    return ""


def parse(manifest: dict, *, iteration: int,
          global_files: tuple[str, ...] = DEFAULT_GLOBAL_FILES) -> IterationContract:
    """Build an IterationContract, or raise ContractError with a fixable message."""
    if not isinstance(manifest, dict):
        raise ContractError("change_manifest.json is not an object")
    changes = manifest.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ContractError("change_manifest.json has no `changes` array")

    parsed: list[ChangeContract] = []
    for index, raw in enumerate(changes):
        if not isinstance(raw, dict):
            raise ContractError(f"changes[{index}] is not an object")
        change_id = str(raw.get("id") or f"chg-{index + 1}")
        contract = raw.get("evaluation_contract")
        if not isinstance(contract, dict):
            raise ContractError(
                f"{change_id}: missing `evaluation_contract`. Every change must declare "
                f"how it is triggered so its blast radius can be estimated.")

        files = [str(f) for f in (raw.get("files") or [])]
        declared = str(contract.get("scope") or SCOPE_CONDITIONAL).strip().lower()
        if declared not in (SCOPE_CONDITIONAL, SCOPE_GLOBAL):
            raise ContractError(
                f"{change_id}: scope must be {SCOPE_CONDITIONAL!r} or {SCOPE_GLOBAL!r}")

        # The file list overrides the declaration, in that direction only.
        hit = _global_by_files(files, global_files)
        forced = ""
        scope = declared
        if hit and declared != SCOPE_GLOBAL:
            scope, forced = SCOPE_GLOBAL, hit

        predicate = contract.get("activation_predicate") or {}
        if scope == SCOPE_CONDITIONAL:
            try:
                validate(predicate, FEATURE_SCHEMA)
            except PredicateError as exc:
                raise ContractError(f"{change_id}: {exc}") from exc
        effect = str(contract.get("expected_effect") or "improve").strip().lower()
        if effect not in ("improve", "neutral", "mixed"):
            raise ContractError(
                f"{change_id}: expected_effect must be improve, neutral or mixed")

        parsed.append(ChangeContract(
            change_id=change_id, scope=scope,
            mechanism=str(contract.get("mechanism") or ""),
            activation_predicate=predicate if scope == SCOPE_CONDITIONAL else {},
            expected_effect=effect, files=files,
            predicted_fixes=[str(t) for t in (raw.get("predicted_fixes") or [])],
            risk_tasks=[str(t) for t in (raw.get("risk_tasks") or [])],
            forced_global_reason=forced,
        ))

    return IterationContract(iteration=int(iteration), changes=parsed)

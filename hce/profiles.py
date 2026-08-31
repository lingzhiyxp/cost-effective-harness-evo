"""The task profile database: what each task has historically done, and cost.

Two properties carry the design.

`p_hist` is last-observation-carried-forward over *accepted* records only. When
the gate rejects a change and reverts, the rollouts just paid for describe a
harness that no longer exists on disk; letting them become the baseline would
compare the next iteration against a tree nobody is standing on. The rejected
measurements are still written to `history` -- they are evidence about the
change -- they just do not move the baseline.

The mechanism aggregate answers "can this task trigger the mechanism", not "does
it on average", so maxima and ORs rather than means. One rollout out of three
reaching 12k tokens makes the task a candidate for a truncation change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class TaskProfileDB:
    def __init__(self, path: Path, dataset: str = "", token_estimator: str = ""):
        self.path = Path(path)
        self.dataset = dataset
        self.token_estimator = token_estimator
        self.tasks: dict[str, dict] = {}
        self.profiling: dict[str, Any] = {}

    # ---------------------------------------------------------------- io

    @classmethod
    def load(cls, path: Path) -> "TaskProfileDB":
        db = cls(path)
        if not db.path.exists():
            return db
        raw = json.loads(db.path.read_text(encoding="utf-8"))
        db.dataset = raw.get("dataset", "")
        db.token_estimator = raw.get("token_estimator", "")
        db.profiling = raw.get("profiling", {})
        db.tasks = raw.get("tasks", {})
        return db

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "token_estimator": self.token_estimator,
            "profiling": self.profiling,
            "tasks": self.tasks,
        }, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------ writes

    def record_outcome(self, task: str, *, iteration: int, fingerprint: str,
                       accepted: bool, score: float | None,
                       n_pass: int = 0, n_fail: int = 0, n_infra: int = 0) -> None:
        """Append one iteration's measurement of one task.

        `score is None` means every rollout was an infrastructure failure: the
        task was not measured, which is recorded but never treated as a zero.
        """
        entry = self.tasks.setdefault(task, {})
        outcome = entry.setdefault("outcome", {"history": []})
        outcome["history"].append({
            "iteration": int(iteration),
            "fingerprint": str(fingerprint),
            "accepted": bool(accepted),
            "score": score,
            "n_pass": int(n_pass), "n_fail": int(n_fail), "n_infra": int(n_infra),
        })
        self._refresh(task)

    def record_mechanism(self, task: str, *, iteration: int, agg: dict,
                         per_rollout: list[dict] | None = None,
                         mean_usd: float | None = None,
                         mean_wall_s: float | None = None) -> None:
        entry = self.tasks.setdefault(task, {})
        entry["mechanism"] = {
            "as_of_iteration": int(iteration),
            "n_rollouts": len(per_rollout or []),
            "agg": agg,
            "per_rollout": per_rollout or [],
        }
        entry["cost"] = {
            "mean_usd_per_rollout": mean_usd,
            "mean_wall_s": mean_wall_s,
            "is_lower_bound": mean_usd is None,
        }

    def set_static(self, task: str, **fields) -> None:
        self.tasks.setdefault(task, {}).update(
            {k: v for k, v in fields.items() if v is not None})

    def _refresh(self, task: str) -> None:
        outcome = self.tasks[task]["outcome"]
        accepted = [h for h in outcome["history"]
                    if h["accepted"] and h["score"] is not None]
        if accepted:
            latest = accepted[-1]
            outcome["p_hist"] = latest["score"]
            outcome["p_hist_source_iteration"] = latest["iteration"]
            scores = [h["score"] for h in accepted]
            mean = sum(scores) / len(scores)
            outcome["variance"] = round(mean * (1.0 - mean), 6)
        else:
            outcome["p_hist"] = None
            outcome["p_hist_source_iteration"] = None
            outcome["variance"] = None
        outcome["n_accepted_observations"] = len(accepted)

    # ------------------------------------------------------------- reads

    def p_hist(self, tasks: list[str] | None = None, default: float = 0.0) -> dict[str, float]:
        """Snapshot of the historical score, for the estimator's baseline.

        Taken before an iteration's own results exist and persisted alongside
        the selection; the estimator reads it from there and never from the live
        database, so the baseline cannot drift under it mid-iteration.
        """
        names = tasks if tasks is not None else sorted(self.tasks)
        out = {}
        for name in names:
            got = (self.tasks.get(name, {}).get("outcome") or {}).get("p_hist")
            out[name] = default if got is None else float(got)
        return out

    def variance(self, tasks: list[str] | None = None) -> dict[str, float]:
        names = tasks if tasks is not None else sorted(self.tasks)
        out = {}
        for name in names:
            got = (self.tasks.get(name, {}).get("outcome") or {}).get("variance")
            p = (self.tasks.get(name, {}).get("outcome") or {}).get("p_hist")
            out[name] = float(got) if got is not None else (
                float(p) * (1.0 - float(p)) if p is not None else 0.0)
        return out

    def mechanisms(self, tasks: list[str] | None = None) -> dict[str, dict]:
        names = tasks if tasks is not None else sorted(self.tasks)
        return {n: ((self.tasks.get(n, {}).get("mechanism") or {}).get("agg") or {})
                for n in names}

    def cost(self, tasks: list[str] | None = None) -> dict[str, float | None]:
        names = tasks if tasks is not None else sorted(self.tasks)
        return {n: (self.tasks.get(n, {}).get("cost") or {}).get("mean_usd_per_rollout")
                for n in names}

    def difficulty(self, tasks: list[str] | None = None) -> dict[str, str]:
        names = tasks if tasks is not None else sorted(self.tasks)
        return {n: self.tasks.get(n, {}).get("difficulty", "unknown") for n in names}

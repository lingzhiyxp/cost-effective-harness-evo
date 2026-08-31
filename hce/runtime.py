"""The glue between evolve.py's loop and the HCE modules.

Kept out of evolve.py so the loop gains call sites rather than algorithm, and so
every piece below stays unit-testable without a harbor job or an experiment
directory.

Two ordering constraints are enforced here rather than left to the caller:

  * The historical baseline is snapshotted at selection time and persisted into
    selection.json. `update_task_history` runs long before the gate, so reading
    p_hist from the live database at gate time would compare against a baseline
    that this iteration's own results have already moved.
  * A rejected iteration's measurements are recorded with accepted=False, so
    they stay as evidence about the change without becoming the baseline for a
    tree that was just reverted.
"""

from __future__ import annotations

import collections
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from hce import falsify, gate, ipw
from hce.contract import ContractError, parse
from hce.mechanism import (TOKEN_ESTIMATOR, aggregate, extract_rollout,
                           schema_reference_table)
from hce.metrics import INFRA, PASS, classify_trial, task_score
from hce.profiles import TaskProfileDB
from hce.selectors import SelectionRequest, build

_TRIAL_SUFFIX = re.compile(r"__[A-Za-z0-9]{6,}$")

DEFAULTS = {
    "enabled": False,
    "selector": "hce",
    "seed": 42,
    "budget_frac": 0.40,
    "global_budget_frac": 0.70,
    "split": {"improvement": 0.19, "regression": 0.41, "audit": 0.40},
    "min_audit": 3,
    "pi_min": 0.02,
    "audit_enabled": True,
    "global_files": ["systemprompt.md", "code_agent.yaml"],
    "falsification": {"enabled": True, "min_predicted_match_frac": 0.5, "min_activation": 1},
    "gate": {"enabled": True, "tolerance": 0.02, "max_hard_regressions": 2,
             "on_contract_invalid": "global_fallback"},
    "profiling": {"k": 3, "from_iteration": 1},
}


def settings(config: dict) -> dict:
    """HCE config with defaults filled in, one level deep on the nested blocks."""
    got = dict(DEFAULTS)
    got.update(config.get("hce") or {})
    for key in ("split", "falsification", "gate", "profiling"):
        merged = dict(DEFAULTS[key])
        merged.update((config.get("hce") or {}).get(key) or {})
        got[key] = merged
    return got


def dataset_tasks(config: dict, project_dir: Path) -> list[str]:
    """The full training set, read from the dataset directory.

    Deliberately not derived from the previous job directory: under subset
    evaluation that would shrink to whatever was measured last time, and N is
    the denominator of the estimator.
    """
    path = config.get("path")
    if not path:
        return []
    root = Path(path)
    if not root.is_absolute():
        root = (project_dir / root).resolve()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "task.toml").exists())


def group_trials(job_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = collections.defaultdict(list)
    for trial in sorted(job_dir.iterdir()):
        if trial.is_dir() and (trial / "result.json").exists():
            grouped[_TRIAL_SUFFIX.sub("", trial.name)].append(trial)
    return dict(grouped)


def measure(job_dir: Path, *, max_iterations: int, max_context_tokens: int,
            dataset_dir: Path | None = None) -> dict[str, dict]:
    """Per-task scores and mechanism profiles from one harbor job directory."""
    out = {}
    for task, trials in group_trials(job_dir).items():
        timeout = None
        if dataset_dir:
            toml_src = dataset_dir / task / "task.toml"
            if toml_src.exists():
                import tomllib
                meta = tomllib.loads(toml_src.read_text(encoding="utf-8"))
                timeout = (meta.get("agent") or {}).get("timeout_sec")
        verdicts = [classify_trial(t)[0] for t in trials]
        rollouts = [extract_rollout(t, max_iterations=max_iterations,
                                    max_context_tokens=max_context_tokens,
                                    agent_timeout_sec=timeout) for t in trials]
        costs = [r["rollout_usd"] for r in rollouts if r.get("rollout_usd") is not None]
        out[task] = {
            "score": task_score(verdicts), "agg": aggregate(rollouts),
            "per_rollout": rollouts, "n_pass": verdicts.count(PASS),
            "n_infra": verdicts.count(INFRA),
            "n_fail": len(verdicts) - verdicts.count(PASS) - verdicts.count(INFRA),
            "mean_usd": (sum(costs) / len(costs)) if costs else None,
        }
    return out


@dataclass
class Plan:
    """What this iteration will evaluate, and why."""
    selection: object
    p_hist: dict[str, float]
    contract: object | None
    falsification: object | None
    task_names: list[str]
    all_tasks: list[str]
    record: dict

    @property
    def pre_rejected(self) -> bool:
        return self.falsification is not None and self.falsification.rejected


def plan_iteration(*, config: dict, exp_dir: Path, project_dir: Path,
                   iteration: int, db: TaskProfileDB,
                   all_tasks: list[str]) -> Plan:
    """Choose this iteration's evaluation subset."""
    cfg = settings(config)
    k = int(config.get("harbor", {}).get("k", 1))
    p_hist = db.p_hist(all_tasks)
    mechanisms = db.mechanisms(all_tasks)

    contract, contract_error = None, ""
    manifest = None
    manifest_src = exp_dir / "change_manifest.json"
    if iteration > 1 and manifest_src.exists():
        try:
            manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            contract_error = f"change_manifest.json is not valid JSON: {exc}"
    if manifest is not None:
        try:
            contract = parse(manifest, iteration=iteration - 1,
                             global_files=tuple(cfg["global_files"]))
        except ContractError as exc:
            contract_error = str(exc)

    falsification = None
    if contract is not None and cfg["falsification"]["enabled"] and mechanisms:
        falsification = falsify.check(
            contract, mechanisms,
            min_predicted_match_frac=cfg["falsification"]["min_predicted_match_frac"],
            min_activation=cfg["falsification"]["min_activation"])

    # The profiling iteration is a census, not a sample. Everything downstream
    # is defined relative to p_hist: the C/G partition, the estimator's
    # baseline, the hard-regression test and the variance weights. A task never
    # measured falls back to a default of 0.0, which would put it in the
    # improvement bucket and give the estimator a baseline it never observed.
    # So iteration `profiling.from_iteration` evaluates the whole training set,
    # and its cost is the fixed price of having a profile database at all.
    profiling_iteration = int(cfg["profiling"].get("from_iteration") or 1)
    is_profiling = iteration <= profiling_iteration
    if is_profiling:
        frac = 1.0
        budget_tasks = len(all_tasks)
    else:
        frac = (cfg["global_budget_frac"] if (contract is not None and contract.is_global)
                else cfg["budget_frac"])
        budget_tasks = max(1, round(len(all_tasks) * float(frac)))

    req = SelectionRequest(
        iteration=iteration, all_tasks=all_tasks, p_hist=p_hist,
        mechanisms=mechanisms, variance=db.variance(all_tasks),
        contract=contract, budget_tasks=budget_tasks, k=k,
        rng=random.Random(int(cfg["seed"]) * 1000 + iteration),
        split=cfg["split"], min_audit=int(cfg["min_audit"]),
        pi_min=float(cfg["pi_min"]), audit_enabled=bool(cfg["audit_enabled"]))
    selection = build(cfg["selector"]).select(req)

    record = {
        "iteration": iteration, "selector": selection.selector, "mode": selection.mode,
        "seed": int(cfg["seed"]) * 1000 + iteration, "k": k,
        "budget_frac": frac, "budget_tasks": budget_tasks,
        "is_profiling": is_profiling,
        "budget_rollouts": len(selection.selected) * k,
        "n_tasks_total": len(all_tasks),
        "token_estimator": TOKEN_ESTIMATOR,
        # Authoritative baseline for this iteration's estimator. Read from here,
        # never from the live database, which this iteration will itself update.
        "p_hist_snapshot": {t: round(v, 6) for t, v in p_hist.items()},
        "strata": {name: {"pool": s.pool, "taken": s.taken, "design": s.design,
                          "pi": s.pi} for name, s in selection.strata.items()},
        "selected": selection.selected,
        "pi": {t: round(v, 6) for t, v in selection.pi.items()},
        "rationale": selection.rationale,
        "contract_error": contract_error,
        "falsification": ({"verdict": falsification.verdict,
                           "reason": falsification.reason,
                           "checks": falsification.checks}
                          if falsification else None),
    }
    return Plan(selection=selection, p_hist=p_hist, contract=contract,
                falsification=falsification, task_names=list(selection.selected),
                all_tasks=all_tasks, record=record)


def score_iteration(*, plan: Plan, measured: dict[str, dict], s_ref: float | None,
                    config: dict) -> tuple[object, object]:
    """Estimate the global score from the subset, then decide accept or reject."""
    cfg = settings(config)
    scores = {t: m["score"] for t, m in measured.items() if m["score"] is not None}
    est = ipw.estimate(all_tasks=plan.all_tasks, p_hist=plan.p_hist, scores=scores,
                       strata=plan.selection.strata, pi=plan.selection.pi)
    decision = gate.decide(
        estimate=est, scores=scores, p_hist=plan.p_hist, pi=plan.selection.pi,
        s_ref=s_ref, tolerance=float(cfg["gate"]["tolerance"]),
        max_hard_regressions=int(cfg["gate"]["max_hard_regressions"]))
    return est, decision


def commit_measurements(*, db: TaskProfileDB, measured: dict[str, dict],
                        iteration: int, fingerprint: str, accepted: bool,
                        is_profiling: bool = False, k: int | None = None,
                        job_dir: str = "") -> None:
    if is_profiling and not db.profiling:
        # Stamped once, on the census iteration: which run built this database,
        # under which harness, at what k. Without it the file cannot be audited
        # later against the run that produced it.
        db.profiling = {"iteration": int(iteration), "k": k,
                        "fingerprint": fingerprint, "job_dir": job_dir}
        db.token_estimator = db.token_estimator or TOKEN_ESTIMATOR
    for task, m in measured.items():
        db.record_outcome(task, iteration=iteration, fingerprint=fingerprint,
                          accepted=accepted, score=m["score"], n_pass=m["n_pass"],
                          n_fail=m["n_fail"], n_infra=m["n_infra"])
        db.record_mechanism(task, iteration=iteration, agg=m["agg"],
                            per_rollout=m["per_rollout"], mean_usd=m["mean_usd"])
    db.save()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def prompt_section(db: TaskProfileDB, tasks: list[str]) -> str:
    """The feature reference the evolve agent needs to write a usable predicate.

    Key names alone are not enough. The proposal's own worked example is
    `max_command_output_tokens > 8000`; on the measured Terminal-Bench profiles
    the largest observation is around 5,000, so that threshold selects nothing
    and the change is rejected before it runs. Showing the realised quartiles is
    what lets a threshold actually partition the task set.
    """
    return schema_reference_table(db.mechanisms(tasks))

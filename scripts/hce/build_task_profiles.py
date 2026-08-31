#!/usr/bin/env python3
"""Build (or inspect) a task profile database from a harbor job directory.

Runs entirely offline against trial artefacts already on disk, so the extractor
and the predicate keys can be developed and checked without spending anything.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hce.mechanism import (FEATURE_SCHEMA, TOKEN_ESTIMATOR, aggregate,  # noqa: E402
                           extract_rollout, schema_reference_table)
from hce.metrics import PASS, INFRA, classify_trial, task_score  # noqa: E402
from hce.profiles import TaskProfileDB  # noqa: E402

_TRIAL_SUFFIX = re.compile(r"__[A-Za-z0-9]{6,}$")


def collect(job_dir: Path, *, max_iterations: int, max_context_tokens: int,
            dataset_dir: Path | None) -> dict[str, dict]:
    grouped: dict[str, list[Path]] = collections.defaultdict(list)
    for trial in sorted(job_dir.iterdir()):
        if trial.is_dir() and (trial / "result.json").exists():
            grouped[_TRIAL_SUFFIX.sub("", trial.name)].append(trial)

    out = {}
    for task, trials in sorted(grouped.items()):
        difficulty = category = None
        timeout = None
        if dataset_dir:
            toml_src = dataset_dir / task / "task.toml"
            if toml_src.exists():
                import tomllib
                meta = tomllib.loads(toml_src.read_text(encoding="utf-8"))
                difficulty = (meta.get("metadata") or {}).get("difficulty")
                category = (meta.get("metadata") or {}).get("category")
                timeout = (meta.get("agent") or {}).get("timeout_sec")
        verdicts, rollouts, costs = [], [], []
        for trial in trials:
            verdicts.append(classify_trial(trial)[0])
            feat = extract_rollout(trial, max_iterations=max_iterations,
                                   max_context_tokens=max_context_tokens,
                                   agent_timeout_sec=timeout)
            rollouts.append(feat)
            if feat.get("rollout_usd") is not None:
                costs.append(feat["rollout_usd"])
        out[task] = {
            "verdicts": verdicts, "per_rollout": rollouts,
            "agg": aggregate(rollouts), "score": task_score(verdicts),
            "n_pass": verdicts.count(PASS), "n_infra": verdicts.count(INFRA),
            "n_fail": len(verdicts) - verdicts.count(PASS) - verdicts.count(INFRA),
            "mean_usd": (sum(costs) / len(costs)) if costs else None,
            "difficulty": difficulty, "category": category,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write task_profiles.json here")
    ap.add_argument("--dataset-dir", type=Path, help="for difficulty/category/timeout")
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--fingerprint", default="")
    ap.add_argument("--max-iterations", type=int, default=300)
    ap.add_argument("--max-context-tokens", type=int, default=200000)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--show-schema", action="store_true")
    args = ap.parse_args()

    data = collect(args.job_dir, max_iterations=args.max_iterations,
                   max_context_tokens=args.max_context_tokens,
                   dataset_dir=args.dataset_dir)
    if not data:
        print("no trials found", file=sys.stderr)
        return 1

    measured = [d for d in data.values() if d["score"] is not None]
    print(f"tasks={len(data)}  measured={len(measured)}  "
          f"mean@k={(sum(d['score'] for d in measured) / len(measured)):.4f}")
    priced = [d["mean_usd"] for d in data.values() if d["mean_usd"] is not None]
    if priced:
        print(f"cost: {len(priced)}/{len(data)} tasks priced, "
              f"total=${sum(priced):.4f}  mean=${sum(priced)/len(priced):.4f}/rollout")

    if args.show_schema:
        print("\n" + schema_reference_table({t: d["agg"] for t, d in data.items()}))

    if args.dry_run or not args.out:
        return 0

    db = TaskProfileDB(args.out, dataset=str(args.dataset_dir or ""),
                       token_estimator=TOKEN_ESTIMATOR)
    db.profiling = {"iteration": args.iteration, "k": max(len(d["verdicts"]) for d in data.values()),
                    "fingerprint": args.fingerprint, "job_dir": str(args.job_dir)}
    for task, d in data.items():
        db.set_static(task, difficulty=d["difficulty"], category=d["category"])
        db.record_outcome(task, iteration=args.iteration, fingerprint=args.fingerprint,
                          accepted=True, score=d["score"], n_pass=d["n_pass"],
                          n_fail=d["n_fail"], n_infra=d["n_infra"])
        db.record_mechanism(task, iteration=args.iteration, agg=d["agg"],
                            per_rollout=d["per_rollout"], mean_usd=d["mean_usd"])
    db.save()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Carry a completed profiling iteration into a fresh experiment directory.

The census iteration is the expensive one and it depends on nothing but the seed
harness -- not on the analysis stage, not on the evolve agent. When a run has to
be restarted for a reason downstream of it, re-running it buys nothing. This
copies the profile database and the iteration-1 artefacts across and marks the
new run to begin at iteration 2.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="experiment holding the profiling run")
    ap.add_argument("--dest", required=True, type=Path, help="new experiment directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profiles = args.source / "task_profiles.json"
    if not profiles.is_file():
        print(f"no task_profiles.json under {args.source}", file=sys.stderr)
        return 1
    db = json.loads(profiles.read_text(encoding="utf-8"))
    tasks = db.get("tasks") or {}
    measured = [t for t, v in tasks.items()
                if (v.get("outcome") or {}).get("p_hist") is not None]
    print(f"source: {args.source.name}")
    print(f"  tasks with a baseline: {len(measured)}/{len(tasks)}")
    print(f"  profiling metadata: {db.get('profiling') or '(absent)'}")
    if len(measured) != len(tasks):
        print("  WARNING: the profiling iteration did not measure every task; "
              "unmeasured tasks would default to p_hist=0.0", file=sys.stderr)

    carry = ["task_profiles.json", "harness_index.json", "iteration_scores.yaml",
             "task_history.json", "best_ever.json", "evolution_history.md",
             "config_snapshot.yaml"]
    run_dir = args.source / "runs" / "iteration_001"
    print(f"\nwould copy into {args.dest.name}:")
    for name in carry:
        src = args.source / name
        print(f"  {name:<26}{'ok' if src.exists() else 'absent'}")
    print(f"  runs/iteration_001/       {'ok' if run_dir.is_dir() else 'absent'}")
    print(f"  workspace/                {'ok' if (args.source / 'workspace').is_dir() else 'absent'}")

    if args.dry_run:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    for name in carry:
        src = args.source / name
        if src.exists():
            shutil.copy2(src, args.dest / name)
    for name in ("workspace", "evolve_agent"):
        src = args.source / name
        if src.is_dir():
            shutil.copytree(src, args.dest / name, dirs_exist_ok=True)
    if run_dir.is_dir():
        dest_run = args.dest / "runs" / "iteration_001"
        dest_run.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, dest_run, dirs_exist_ok=True)
    print(f"\nseeded {args.dest}")
    print(f"resume with: --experiment {args.dest.name} --start-iteration 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

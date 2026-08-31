#!/usr/bin/env python3
"""Split tb2-84 into difficulty-stratified train/test halves.

easy divides evenly at 2/2; medium (53) and hard (27) are both odd, so exactly
one of each spills. Two fills are therefore feasible -- {2,27,13} and {2,26,14}
-- and the tie is broken on secondary objectives rather than arbitrarily:
category balance first, then total agent timeout, which is the closest available
proxy for wall-clock balance between the halves.

Deliberately not reusing the existing tb2-train45/tb2-test44 split: it was drawn
from the 89-task set, which includes five tasks that never passed in any prior
run and cost 10-31 minutes each. They are pure burn for an evolution loop.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import shutil
import sys
import tomllib
from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard")


def read_tasks(dataset: Path) -> dict[str, dict]:
    out = {}
    for task_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        toml_src = task_dir / "task.toml"
        if not toml_src.exists():
            continue
        meta = tomllib.loads(toml_src.read_text(encoding="utf-8"))
        out[task_dir.name] = {
            "difficulty": (meta.get("metadata") or {}).get("difficulty", "unknown"),
            "category": (meta.get("metadata") or {}).get("category", "unknown"),
            "agent_timeout_sec": float((meta.get("agent") or {}).get("timeout_sec") or 0),
        }
    return out


def _cost(train: list[str], test: list[str], tasks: dict[str, dict]) -> tuple[float, float]:
    cats = sorted({t["category"] for t in tasks.values()})
    ctr = lambda names: collections.Counter(tasks[n]["category"] for n in names)
    a, b = ctr(train), ctr(test)
    l1 = sum(abs(a[c] - b[c]) for c in cats)
    secs = abs(sum(tasks[n]["agent_timeout_sec"] for n in train)
               - sum(tasks[n]["agent_timeout_sec"] for n in test))
    return l1, secs


def split(tasks: dict[str, dict], seed: int, restarts: int = 4000) -> tuple[list[str], list[str], dict]:
    """Difficulty-stratified halving, with the secondary objectives searched.

    Placing only the two spill tasks leaves the category balance entirely to
    whichever shuffle happened first, which on this data gives a category L1 of
    around 26. Re-drawing the within-stratum shuffle and keeping the best cost
    is cheap -- the whole search is a few thousand halvings of 84 names -- and
    roughly halves that imbalance. The seed still makes it reproducible.
    """
    by_diff = collections.defaultdict(list)
    for name, meta in tasks.items():
        by_diff[meta["difficulty"]].append(name)
    for names in by_diff.values():
        names.sort()

    best = None
    for restart in range(restarts):
        rng = random.Random(seed * 10_000 + restart)
        fixed_train, fixed_test, spill = [], [], []
        for diff in DIFFICULTIES:
            names = list(by_diff.get(diff, []))
            rng.shuffle(names)
            half = len(names) // 2
            fixed_train += names[:half]
            fixed_test += names[half:2 * half]
            if len(names) % 2:
                spill.append(names[-1])
        for assignment in itertools.product((0, 1), repeat=len(spill)):
            train = fixed_train + [n for n, side in zip(spill, assignment) if side == 0]
            test = fixed_test + [n for n, side in zip(spill, assignment) if side == 1]
            if len(train) != len(test):
                continue                       # exact halves only
            key = _cost(train, test, tasks)
            if best is None or key < best[0]:
                best = (key, sorted(train), sorted(test), spill, list(assignment), restart)
    if best is None:
        raise SystemExit("no exact-half assignment exists")
    key, train, test, spill, assignment, restart = best
    return train, test, {"category_l1": key[0], "timeout_delta_sec": round(key[1]),
                         "spill_tasks": spill, "spill_assignment": assignment,
                         "restarts_searched": restarts, "best_restart": restart}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, help="where the split dirs go (default: source's parent)")
    ap.add_argument("--train-name", default="tb2-42-train")
    ap.add_argument("--test-name", default="tb2-42-test")
    ap.add_argument("--smoke-name", default="tb2-smoke6")
    ap.add_argument("--smoke-size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--build", action="store_true", help="copy the task dirs, not just the manifest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = read_tasks(args.source)
    if not tasks:
        print(f"no tasks under {args.source}", file=sys.stderr)
        return 1
    train, test, tie = split(tasks, args.seed)

    # The smoke subset is drawn from train only, so it can never leak a held-out
    # task, and it is selected by *runtime* rather than by difficulty share. Its
    # job is to exercise the pipeline end to end, not to be representative, and
    # Terminal-Bench task timeouts span 600 to 12000 seconds -- a proportional
    # draw would land on the slowest tasks and make the smoke run useless.
    smoke = sorted(sorted(train, key=lambda n: (tasks[n]["agent_timeout_sec"], n))
                   [:args.smoke_size])

    counts = lambda names: {d: sum(1 for n in names if tasks[n]["difficulty"] == d)
                            for d in DIFFICULTIES}
    print(f"{'split':<12}{'n':>4}  " + "  ".join(f"{d:>7}" for d in DIFFICULTIES))
    for label, names in (("source", sorted(tasks)), ("train", train),
                         ("test", test), ("smoke", smoke)):
        c = counts(names)
        print(f"{label:<12}{len(names):>4}  " + "  ".join(f"{c[d]:>7}" for d in DIFFICULTIES))
    print(f"\ntie-break: category L1={tie['category_l1']}  "
          f"timeout delta={tie['timeout_delta_sec']:.0f}s  spill={tie['spill_tasks']}")

    assert not (set(train) & set(test)), "train and test overlap"
    assert set(train) | set(test) == set(tasks), "split does not cover the source"
    assert len(train) == len(test), "halves are uneven"
    assert set(smoke) <= set(train), "smoke subset must come from train"
    print("checks passed: disjoint, exhaustive, even, smoke drawn from train")

    if args.dry_run:
        return 0

    root = args.out_root or args.source.parent
    manifest = {
        "source": str(args.source), "seed": args.seed,
        "stratified_on": "task.toml [metadata].difficulty",
        "tie_break": ["category histogram L1", "sum of agent_timeout_sec"],
        "tie_break_result": tie,
        "n_train": len(train), "n_test": len(test),
        "counts": {"train": counts(train), "test": counts(test), "smoke": counts(smoke)},
        "train": train, "test": test, "smoke": smoke,
        "note": ("Drawn from tb2-84, not the 89-task flat set: the five dropped tasks "
                 "never passed in any prior run and cost 10-31 minutes each."),
    }
    (root / "tb2-42-split.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {root / 'tb2-42-split.manifest.json'}")

    if args.build:
        for name, names in ((args.train_name, train), (args.test_name, test),
                            (args.smoke_name, smoke)):
            dest = root / name
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for task in names:
                shutil.copytree(args.source / task, dest / task)
            print(f"built {dest} ({len(names)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

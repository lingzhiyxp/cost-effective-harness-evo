#!/usr/bin/env python3
"""Dry-run any selector against a saved profile database and a hand-written contract.

No API calls, no containers. This is the tool that answers the question the whole
method rests on -- does the predicate actually partition the task set -- before
any budget is committed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hce.contract import ContractError, parse  # noqa: E402
from hce.profiles import TaskProfileDB  # noqa: E402
from hce.selectors import SelectionRequest, build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--contract", type=Path, help="a change_manifest.json")
    ap.add_argument("--selector", default="hce", choices=("hce", "full", "variance"))
    ap.add_argument("--budget-frac", type=float, default=0.40)
    ap.add_argument("--min-audit", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iteration", type=int, default=2)
    args = ap.parse_args()

    db = TaskProfileDB.load(args.profiles)
    tasks = sorted(db.tasks)
    contract = None
    if args.contract:
        try:
            contract = parse(json.loads(args.contract.read_text()), iteration=args.iteration)
        except ContractError as exc:
            print(f"contract rejected: {exc}")
            return 2

    req = SelectionRequest(
        iteration=args.iteration, all_tasks=tasks, p_hist=db.p_hist(tasks),
        mechanisms=db.mechanisms(tasks), variance=db.variance(tasks),
        contract=contract, budget_tasks=max(1, round(len(tasks) * args.budget_frac)),
        k=args.k, rng=random.Random(args.seed * 1000 + args.iteration),
        min_audit=args.min_audit)
    result = build(args.selector).select(req)

    print(f"selector={result.selector}  mode={result.mode}  "
          f"N={len(tasks)}  budget={req.budget_tasks} tasks x k={args.k} "
          f"= {len(result.selected) * args.k} rollouts")
    if "n_activated" in result.rationale:
        print(f"activation set: {result.rationale['n_activated']}/{len(tasks)}")
    print(f"\n{'stratum':<9}{'pool':>6}{'taken':>7}{'design':>10}{'pi':>9}{'weight':>9}")
    for name, s in result.strata.items():
        pi = "per-task" if s.pi is None else f"{s.pi:.3f}"
        w = "-" if s.pi is None else f"{1 / s.pi:.2f}x" if s.pi else "inf"
        print(f"{name:<9}{len(s.pool):>6}{len(s.taken):>7}{s.design:>10}{pi:>9}{w:>9}")
    print(f"\nselected {len(result.selected)}: {' '.join(result.selected[:12])}"
          f"{' ...' if len(result.selected) > 12 else ''}")

    covered = {t for s in result.strata.values() for t in s.pool}
    missing = sorted(set(tasks) - covered)
    print(f"\nreachability: {len(covered)}/{len(tasks)} tasks have pi > 0"
          f"{'  MISSING: ' + ' '.join(missing) if missing else '  (estimator is valid)'}")
    est = db.cost(result.selected)
    spend = sum(v for v in est.values() if v is not None) * args.k
    print(f"estimated spend this iteration: ${spend:.2f} "
          f"(full set would be ${sum(v for v in db.cost(tasks).values() if v) * args.k:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

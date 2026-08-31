#!/usr/bin/env python3
"""Summarise an HCE run: what was selected, what it estimated, what it cost.

The three numbers the method stands or falls on:
  * the share of iterations that ran in conditional mode -- global-scope changes
    cannot be mechanism-targeted, so this bounds how much of the saving the
    method's distinctive part is responsible for
  * realised cost against the full-set counterfactual
  * gate decisions, and whether the estimate or the hard-regression count drove
    them
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--rollout-usd", type=float, default=0.155,
                    help="fallback unit cost when a run has no audited figure")
    args = ap.parse_args()
    exp = args.exp_dir

    scores = (yaml.safe_load((exp / "iteration_scores.yaml").read_text(encoding="utf-8"))
              or {}).get("scores", [])
    if not scores:
        print("no iteration_scores.yaml", file=sys.stderr)
        return 1

    rows = []
    for entry in scores:
        it = entry["iteration"]
        hce = entry.get("hce") or {}
        sel = exp / "runs" / f"iteration_{it:03d}" / "hce" / "selection.json"
        gate = exp / "runs" / f"iteration_{it:03d}" / "hce" / "gate.json"
        s = json.loads(sel.read_text()) if sel.exists() else {}
        g = json.loads(gate.read_text()) if gate.exists() else {}
        rows.append({
            "it": it, "mode": hce.get("mode") or s.get("mode", "-"),
            "profiling": s.get("is_profiling", False),
            "n": hce.get("n_evaluated", entry.get("n_total", 0)),
            "rollouts": hce.get("budget_rollouts", 0),
            "subset": hce.get("subset_pass_rate"), "est": hce.get("global_estimate"),
            "se": hce.get("se"), "verdict": g.get("verdict", "-"),
            "rule": g.get("rule", "-"),
            "usd": (entry.get("cost") or {}).get("total_usd"),
            "activated": (s.get("rationale") or {}).get("n_activated"),
        })

    n_total = rows[0]["n"] if rows else 0
    k = scores[0].get("k", 1)
    print(f"{'it':>3} {'mode':<12}{'act':>5}{'tasks':>6}{'roll':>6}"
          f"{'subset':>8}{'est':>8}{'se':>8}  {'gate':<8}{'rule':<20}{'usd':>7}")
    print("-" * 96)
    for r in rows:
        fmt = lambda v, w=8, p=4: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(f"{r['it']:>3} {r['mode'] + ('*' if r['profiling'] else ''):<12}"
              f"{str(r['activated'] or '-'):>5}{r['n']:>6}{r['rollouts']:>6}"
              f"{fmt(r['subset'])}{fmt(r['est'])}{fmt(r['se'])}  "
              f"{r['verdict']:<8}{r['rule']:<20}{fmt(r['usd'], 7, 2)}")
    print("  * = profiling iteration (full set by design)")

    post = [r for r in rows if not r["profiling"]]
    if post:
        cond = sum(1 for r in post if r["mode"] == "conditional")
        print(f"\nmode after profiling: conditional {cond}/{len(post)} "
              f"({cond / len(post):.0%}), global {len(post) - cond}/{len(post)}")
        print("  conditional is where mechanism targeting does any work; a global "
              "change is sampled, not targeted.")

    spent = sum(r["rollouts"] for r in rows)
    counterfactual = len(rows) * n_total * k
    if counterfactual:
        print(f"\nrollouts: {spent} vs {counterfactual} for full-set every iteration "
              f"= {1 - spent / counterfactual:.1%} fewer")
    audited = [r["usd"] for r in rows if r["usd"] is not None]
    if audited:
        print(f"audited spend: ${sum(audited):.2f} over {len(audited)} iteration(s)")
    summary_src = exp / "cost_summary.json"
    if summary_src.exists():
        c = json.loads(summary_src.read_text())
        parts = " ".join(f"{r}=${b['cost_usd']:.2f}" for r, b in sorted(c["by_role"].items()))
        print(f"by role: {parts}  total=${c['total_usd']:.2f}"
              + ("  (lower bound)" if c["is_lower_bound"] else ""))

    verdicts = [r["verdict"] for r in rows if r["verdict"] != "-"]
    if verdicts:
        acc = verdicts.count("accept")
        print(f"\ngate: {acc} accepted, {len(verdicts) - acc} rejected")
        for rule in sorted({r["rule"] for r in rows if r["verdict"] == "reject"}):
            print(f"  rejected by {rule}: "
                  f"{sum(1 for r in rows if r['rule'] == rule)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

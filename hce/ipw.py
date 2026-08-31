"""Extrapolating a subset measurement back to the whole training set.

The estimator is a stratified Horvitz-Thompson difference estimator. Each task
contributes its historical score unconditionally; each *measured* task
additionally contributes its change from history, weighted by 1/pi. Strata taken
by census have pi = 1 and therefore contribute no variance at all -- which is
the point of making the regression bucket a census whenever it fits.

Two quantities are reported, not one:

  * HT is unbiased but its variance is driven by the smallest pi in play. At a
    forty-task training set with an audit stratum of seven drawn from twenty-two,
    one audit task flipping moves the estimate by about eight points, which is
    larger than the effect sizes being looked for.
  * Hajek divides by the realised sum of weights instead of N. It is slightly
    biased and is often steadier, though not always: on the synthetic fixture in
    tests/test_hce_ipw.py HT has the lower spread (sd 0.039 against 0.068),
    because the difference form already removes most of the between-task
    variation that Hajek's ratio has to absorb. Both are reported so the choice
    can be made on measured behaviour rather than on which one reads better.

Removing the audit stratum does not merely add variance, it removes the
estimator's ability to notice. In the same fixture, a regression confined to the
non-activated tasks leaves the audit-free estimate 16 points high while its
reported standard error is exactly zero -- confidently wrong, which is worse
than noisy.

The historical baseline is passed in, never read from a live database: it has to
be the snapshot taken before this iteration produced any results, or the
comparison is against a moving target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INFRA_DROP_WARN_FRAC = 0.10


@dataclass
class Estimate:
    s_hat: float                 # Horvitz-Thompson, unbiased
    s_hajek: float               # ratio form, lower variance
    se: float
    se_is_conservative: bool
    n_total: int
    n_selected: int
    n_measured: int
    n_dropped_infra: int
    dropped_tasks: list[str] = field(default_factory=list)
    per_stratum: dict = field(default_factory=dict)
    baseline_mean: float = 0.0

    def to_dict(self) -> dict:
        return {
            "s_hat": round(self.s_hat, 6),
            "s_hajek": round(self.s_hajek, 6),
            "se": round(self.se, 6),
            "se_is_conservative": self.se_is_conservative,
            "baseline_mean": round(self.baseline_mean, 6),
            "n_total": self.n_total, "n_selected": self.n_selected,
            "n_measured": self.n_measured, "n_dropped_infra": self.n_dropped_infra,
            "dropped_tasks": self.dropped_tasks,
            "per_stratum": self.per_stratum,
        }


def estimate(*, all_tasks: list[str], p_hist: dict[str, float],
             scores: dict[str, float], strata: dict, pi: dict[str, float]) -> Estimate:
    """Global mean@k estimated from the measured subset.

    `scores` holds only the tasks that produced a gradeable result. A selected
    task absent from it was lost to infrastructure: it is removed from its
    stratum's realised sample while the stratum's *pool* stays the same size, so
    pi is recomputed as n_effective/N_h. That treats the loss as independent of
    the harness change, which is the assumption being made either way -- it is
    recorded and warned about rather than left implicit.
    """
    n = len(all_tasks)
    if n == 0:
        return Estimate(0.0, 0.0, 0.0, False, 0, 0, 0, 0)

    baseline_sum = sum(p_hist.get(t, 0.0) for t in all_tasks)
    selected = sorted(pi)
    dropped = [t for t in selected if t not in scores]

    correction = 0.0
    variance = 0.0
    conservative = False
    hajek_num = hajek_den = 0.0
    detail: dict = {}

    for name, stratum in strata.items():
        pool = list(getattr(stratum, "pool", []) or [])
        taken = [t for t in getattr(stratum, "taken", []) or [] if t in scores]
        n_h, big_n_h = len(taken), len(pool)
        if not taken or big_n_h == 0:
            detail[name] = {"pool": big_n_h, "measured": 0, "pi": None,
                            "mean_delta": None, "contribution": 0.0}
            continue

        stratum_pi = getattr(stratum, "pi", None)
        per_task = stratum_pi is None       # poisson: pi differs per task
        deltas = [scores[t] - p_hist.get(t, 0.0) for t in taken]

        if per_task:
            contribution = sum(d / pi[t] for d, t in zip(deltas, taken))
            for t in taken:
                hajek_num += scores[t] / pi[t]
                hajek_den += 1.0 / pi[t]
            # Poisson sampling: units are independent, so no FPC term.
            var_h = sum((d / pi[t]) ** 2 * (1.0 - pi[t]) for d, t in zip(deltas, taken))
            effective_pi = None
        else:
            # Recompute from what was actually measured, so an infra loss shows
            # up as a smaller realised sample rather than a silent bias.
            effective_pi = n_h / big_n_h
            contribution = sum(deltas) / effective_pi
            for t in taken:
                hajek_num += scores[t] / effective_pi
                hajek_den += 1.0 / effective_pi
            if effective_pi >= 1.0:
                var_h = 0.0                 # census stratum: nothing sampled away
            elif n_h == 1:
                # One observation gives no variance estimate. Substitute the
                # worst case for a delta in [-1, 1] rather than reporting zero.
                var_h = big_n_h ** 2 * (1.0 - n_h / big_n_h) * 1.0 / n_h
                conservative = True
            else:
                mean_d = sum(deltas) / n_h
                s2 = sum((d - mean_d) ** 2 for d in deltas) / (n_h - 1)
                var_h = big_n_h ** 2 * (1.0 - n_h / big_n_h) * s2 / n_h

        correction += contribution
        variance += var_h
        detail[name] = {
            "pool": big_n_h, "measured": n_h,
            "pi": (round(effective_pi, 6) if effective_pi is not None else "per-task"),
            "mean_delta": round(sum(deltas) / n_h, 6),
            "contribution": round(contribution / n, 6),
        }

    s_hat = (baseline_sum + correction) / n
    s_hajek = (hajek_num / hajek_den) if hajek_den else 0.0
    return Estimate(
        s_hat=s_hat, s_hajek=s_hajek,
        se=math.sqrt(variance) / n, se_is_conservative=conservative,
        n_total=n, n_selected=len(selected), n_measured=len(selected) - len(dropped),
        n_dropped_infra=len(dropped), dropped_tasks=dropped, per_stratum=detail,
        baseline_mean=baseline_sum / n)


def hard_regressions(*, scores: dict[str, float], p_hist: dict[str, float],
                     pi: dict[str, float], hard_floor: float = 0.0) -> list[str]:
    """Tasks that were reliably passing and scored at the floor this iteration.

    Restricted to pi == 1: this is meant to be an observation, not an estimate.
    A census task carries no sampling weight, so the count means exactly what it
    says and can veto a change on its own without any distributional argument.
    """
    return sorted(
        t for t, score in scores.items()
        if pi.get(t, 0.0) >= 1.0
        and p_hist.get(t, 0.0) >= 1.0
        and score <= hard_floor)

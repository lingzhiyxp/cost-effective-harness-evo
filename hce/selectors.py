"""Subset selection, with the inclusion probability that makes it estimable.

The contract every selector honours: for each selected task return a pi in
(0, 1], and declare the design that produced it. hce/ipw.py consumes nothing
else, which is what makes the three arms genuinely comparable -- the full-set
arm is the special case pi == 1 everywhere, not a separate code path.

On determinism: Horvitz-Thompson needs pi known and strictly positive, not
random. A task taken by census has pi = 1 and contributes no variance, which is
fine. The failure is pi = 0 -- a task the design *cannot* reach contributes only
its history, and biases the estimate by exactly its unobserved delta. That is
why randomisation appears only where a quota is smaller than its pool, and why
the audit stratum is mandatory rather than a nicety: without it the estimator
stops being a statistic and becomes the assumption that the predicate is
complete.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

MODE_FULL = "full"
MODE_VARIANCE = "variance"
MODE_CONDITIONAL = "conditional"
MODE_GLOBAL = "global"


@dataclass
class Stratum:
    name: str
    pool: list[str]
    taken: list[str]
    design: str          # "census" | "srswor" | "poisson" | "forced"
    pi: float | None     # None for poisson, where pi is per-task


@dataclass
class SelectionRequest:
    iteration: int
    all_tasks: list[str]
    p_hist: dict[str, float]
    mechanisms: dict[str, dict]
    variance: dict[str, float]
    contract: object | None          # IterationContract, or None on iteration 1
    budget_tasks: int
    k: int
    rng: random.Random
    split: dict = field(default_factory=lambda: {"improvement": 0.19,
                                                 "regression": 0.41,
                                                 "audit": 0.40})
    min_audit: int = 3
    pi_min: float = 0.02
    # Production always audits: leftover budget spills there, which widens
    # coverage and lowers the weight. Turning it off makes the estimator assume
    # the predicate is complete, and exists only for the ablation that shows
    # what that assumption costs.
    audit_enabled: bool = True
    already_measured: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionResult:
    selector: str
    mode: str
    selected: list[str]
    pi: dict[str, float]
    strata: dict[str, Stratum]
    k_per_task: dict[str, int]
    rationale: dict = field(default_factory=dict)


class Selector(Protocol):
    name: str
    def select(self, req: SelectionRequest) -> SelectionResult: ...


def _srswor(pool: list[str], quota: int, rng: random.Random) -> tuple[list[str], float, str]:
    """Take a whole stratum, or a uniform sample of it. Returns (taken, pi, design).

    Uniform, never a top-k by any score: an ordering would make pi undefined and
    silently break the estimator downstream. This is also why the proposal's
    "dispatch the most cost-effective tasks first" is deliberately absent.
    """
    if not pool:
        return [], 1.0, "census"
    if quota >= len(pool):
        return sorted(pool), 1.0, "census"
    taken = rng.sample(sorted(pool), quota)
    return sorted(taken), quota / len(pool), "srswor"


class FullSetSelector:
    name = MODE_FULL

    def select(self, req: SelectionRequest) -> SelectionResult:
        tasks = sorted(req.all_tasks)
        stratum = Stratum("all", tasks, tasks, "census", 1.0)
        return SelectionResult(
            selector=self.name, mode=MODE_FULL, selected=tasks,
            pi={t: 1.0 for t in tasks}, strata={"all": stratum},
            k_per_task={t: req.k for t in tasks},
            rationale={"note": "every task, pi=1; the estimator reduces to mean@k"})


class VarianceSelector:
    """Task-CoEvolve's baseline: weight by historical Bernoulli variance.

    Poisson sampling, not weighted sampling without replacement. Under the
    latter, a task's inclusion probability is not its weight and has no closed
    form, so the estimator would be running on a pi it cannot actually compute.
    Poisson gives each task an exact, independent pi; the sample size is random
    but the budget holds in expectation.

    pi_min is the floor that keeps a task with p_hist == 1 reachable. It is
    precisely what the proposal argues variance sampling lacks -- a stable-pass
    task has zero variance and thus zero weight, yet stable-pass tasks are the
    only place a regression can occur. Setting pi_min = 0 reproduces that
    failure mode for the ablation.
    """
    name = MODE_VARIANCE

    def select(self, req: SelectionRequest) -> SelectionResult:
        tasks = sorted(req.all_tasks)
        weights = {t: max(req.variance.get(t, 0.0), 0.0) for t in tasks}
        total = sum(weights.values())
        m = max(1, min(req.budget_tasks, len(tasks)))
        if total <= 0:
            pi = {t: m / len(tasks) for t in tasks}
        else:
            pi = {t: min(1.0, max(req.pi_min, m * w / total)) for t, w in weights.items()}
        taken = sorted(t for t in tasks if req.rng.random() < pi[t])
        if not taken:  # a degenerate draw would leave nothing to estimate from
            taken = sorted(req.rng.sample(tasks, min(m, len(tasks))))
        stratum = Stratum("poisson", tasks, taken, "poisson", None)
        return SelectionResult(
            selector=self.name, mode=MODE_VARIANCE, selected=taken,
            pi={t: pi[t] for t in taken}, strata={"poisson": stratum},
            k_per_task={t: req.k for t in taken},
            rationale={"pi_min": req.pi_min, "expected_n": round(sum(pi.values()), 2)})


class HCESelector:
    name = "hce"

    def select(self, req: SelectionRequest) -> SelectionResult:
        tasks = sorted(req.all_tasks)
        budget = max(1, min(req.budget_tasks, len(tasks)))

        # Iteration one, or any iteration whose manifest could not be read:
        # there is no behavioural information to condition on, so one uniform
        # stratum is all that is available.
        if req.contract is None:
            taken, pi, design = _srswor(tasks, budget, req.rng)
            stratum = Stratum("global", tasks, taken, design, pi)
            return SelectionResult(
                selector=self.name, mode=MODE_GLOBAL, selected=taken,
                pi={t: pi for t in taken}, strata={"global": stratum},
                k_per_task={t: req.k for t in taken},
                rationale={"reason": "no contract"})

        # A global-scope change still splits into improvement and regression.
        # "Global" means every task is activated, not that nothing is known: the
        # C/G partition is defined by p_hist, which is available regardless of
        # any predicate. Collapsing to a uniform sample here would drop the
        # regression-defence quota in the one case that most needs it -- a
        # system-prompt edit can break anything, and under uniform sampling the
        # currently-passing tasks, which are the only ones that can regress, get
        # sampled merely in proportion to how many of them there are. The audit
        # pool comes out empty because nothing is unactivated, and its quota
        # cascades into C and G.
        activated, detail = req.contract.activation(req.mechanisms)
        forced = [t for t in req.contract.risk_tasks if t in set(tasks)]
        forced_set = set(forced)

        pool_c = sorted(t for t in activated
                        if t not in forced_set and req.p_hist.get(t, 0.0) < 1.0)
        pool_g = sorted(t for t in activated
                        if t not in forced_set and req.p_hist.get(t, 0.0) >= 1.0)
        claimed = forced_set | set(pool_c) | set(pool_g)
        pool_a = sorted(t for t in tasks if t not in claimed)

        # The audit floor is taken first: it is the stratum that makes every
        # task reachable, so it must not be what gets squeezed out by rounding.
        remaining = max(0, budget - len(forced))
        if not req.audit_enabled:
            pool_a = []
        m_a = min(len(pool_a), max(req.min_audit if pool_a else 0,
                                   int(math.floor(remaining * req.split["audit"]))))
        left = max(0, remaining - m_a)
        denom = req.split["improvement"] + req.split["regression"]
        m_c = int(math.floor(left * req.split["improvement"] / denom)) if denom else 0
        m_g = left - m_c
        # Spill an unused quota to the other buckets rather than losing budget.
        if m_c > len(pool_c):
            m_g += m_c - len(pool_c); m_c = len(pool_c)
        if m_g > len(pool_g):
            spare = m_g - len(pool_g); m_g = len(pool_g)
            m_c = min(len(pool_c), m_c + spare)
            spare = max(0, budget - len(forced) - m_a - m_c - m_g)
            m_a = min(len(pool_a), m_a + spare)

        strata: dict[str, Stratum] = {}
        pi: dict[str, float] = {}
        selected: list[str] = []
        if forced:
            strata["F"] = Stratum("F", forced, forced, "forced", 1.0)
            pi.update({t: 1.0 for t in forced}); selected += forced
        for name, pool, quota in (("C", pool_c, m_c), ("G", pool_g, m_g),
                                  ("audit", pool_a, m_a)):
            taken, p, design = _srswor(pool, quota, req.rng)
            strata[name] = Stratum(name, pool, taken, design, p if pool else None)
            pi.update({t: p for t in taken}); selected += taken

        mode = MODE_GLOBAL if req.contract.is_global else MODE_CONDITIONAL
        return SelectionResult(
            selector=self.name, mode=mode, selected=sorted(selected),
            pi=pi, strata=strata, k_per_task={t: req.k for t in selected},
            rationale={"activation": detail, "n_activated": len(activated),
                       "quotas": {"F": len(forced), "C": m_c, "G": m_g, "audit": m_a},
                       "stratified_global": req.contract.is_global})


SELECTORS: dict[str, type] = {
    MODE_FULL: FullSetSelector,
    MODE_VARIANCE: VarianceSelector,
    "hce": HCESelector,
}


def build(name: str) -> Selector:
    try:
        return SELECTORS[name]()
    except KeyError:
        raise ValueError(f"unknown selector {name!r}; choose from {sorted(SELECTORS)}") from None

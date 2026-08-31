# Hypothesis-Conditioned Evaluation for Cost-Effective Harness Evolution

**Working title (method)**: HCE — Hypothesis-Conditioned Evaluation
**Working title (project)**: CEHE — Cost-Effective Harness Evolution
**Status**: proposal draft, 2026-08-23
**Code base**: fork of `agentic-harness-engineering` (AHE) at `cost-effective-harness-evo/`

---

## 0. Summary

Current harness-optimization loops (AHE, Meta-Harness, HarnessBank) re-evaluate every candidate
harness on the **entire** training set every round. This proposal replaces the uniform full-set
evaluation with an evaluation set that is **conditioned on the specific edit under test**.

Each round, the evolve agent emits a machine-checkable *evaluation contract* describing the
mechanism of its edit, the runtime condition under which that mechanism can fire, the tasks it
expects to fix, and the tasks it believes are at risk. A *task dossier* — built once from traces
the loop already pays for — records each training task's outcome history, structural fingerprint,
mechanism profile, and per-rollout cost. Contract and dossier together partition the training set
into three sets with three distinct statistical roles: a **confirmation set** (does the intended
effect appear?), a **guard set** (is aggregate harm bounded?), and a **deferred set** (imputed from
history, audited on rotation). Rollouts are allocated to the first two in waves under an
anytime-valid stopping rule, and the round's cross-harness score is recovered with a
history-anchored, inclusion-probability-corrected estimator.

The target claim is not that each round becomes statistically certifiable — the effect sizes are
smaller than the per-round noise floor at any realistic budget. The target claim is that the **same
end-of-campaign held-out pass@1 is reachable at 3–5× lower offline cost**, and that at a matched
dollar budget the loop reaches a **higher** held-out pass@1 than full-set evaluation, because the
budget buys more rounds and more repeats per informative task instead of re-confirming outcomes that
were already determined.

---

## 1. Motivation

### 1.1 The offline cost of a harness-evolution campaign is dominated by evaluation, not by optimization

AHE's reference campaign on Terminal-Bench 2 runs 89 tasks × k=2 rollouts × 10 iterations ≈ 1,780
rollouts, each with a 3,600 s per-task timeout, finishing in roughly 32 wall-clock hours at
concurrency 96 (AHE, arXiv:2604.25850, §4.1 and App. A). Nothing in that budget is spent on the
optimizer; it is spent on measuring.

A fully-instrumented replication on this project's own infrastructure priced every LLM call in a
5-iteration AHE run over 40 SWE-bench tasks at k=1 (`reset-free-coding-agent-harness/docs/
benchmark40-ahe-cost-audit-report.md`):

| Role | Calls | Cost | Share |
|---|---:|---:|---:|
| `code_agent` (rollouts) | 3,307 | \$37.61 | 67% |
| `agent_debugger` (trace distillation) | 1,380 | \$15.28 | 27% |
| `explore_agent` | 24 | \$1.57 | 3% |
| `evolve_agent` (the optimizer) | 81 | \$1.53 | 3% |
| **Total** | | **\$55.99** | |

**94% of the bill is evaluation and the analysis of what evaluation produced. The component that
performs the actual harness engineering is 3%.** Making the optimizer cheaper or smarter changes
almost nothing about the cost of the loop; the only lever with leverage is *how much evaluation each
round buys*.

Two secondary cost properties compound this:

- **Per-round cost grows with the harness.** In the audited run, cost per iteration rose 44% from
  iteration 1 to iteration 5 on an unchanged task set, because `systemprompt.md` grew from 548 B to
  2,465 B and a skill package was added, so every rollout carries a longer prompt. An evolving
  harness has a running cost that the pass-rate curve does not show.
- **Analysis cost is proportional to evaluation cost.** `agent_debugger` reads the traces evaluation
  produced. Cutting the number of evaluated tasks cuts 94% of the bill proportionally, not 67%.

### 1.2 A full evaluation round answers a question the loop did not ask

What the loop needs from round *t* is a decision — keep or revert the edits in Δ*t* — plus evidence
for the next proposal. A full pass over *N* tasks instead produces a high-precision estimate of
aggregate pass@1. Most of that precision is bought from tasks whose outcome was already determined.

Measured on the archived 40×5 outcome matrix of the audited AHE run
(`docs/results/benchmark40-ahe-run2-per-task.json`):

| Quantity | Value |
|---|---|
| Tasks with **zero** state transitions across all 5 iterations | **29 / 40 (72.5%)** |
| Tasks that change state at least once during the whole campaign | 11 / 40 (27.5%) |
| Task-rounds carrying a state change | 23 / 160 (**14.4%**) |
| Per-round tasks changing state | 5, 7, 5, 6 out of 40 |

An oracle that knew in advance which 11 tasks would ever move could have reproduced **every state
transition observed in the entire campaign at 27.5% of the rollout cost**. The remaining 72.5% of
the evaluation returned the same bit five times.

AHE's own Terminal-Bench 2 campaign shows the same density: across 9 evaluation rounds there were 45
actual regressions and a comparable number of actual fixes over 89 tasks, i.e. roughly 12–14 flips
per round out of 89 — again ≈14% of the evaluated set (AHE §4.4.2, App. D).

### 1.3 Most training tasks are deterministic under a fixed harness — but that does not make them safe to skip

A three-run repeat of one identical harness on 100 stratified SWE-bench-Verified tasks at k=2
(6 rollouts per task, `docs/results/benchmark100-baseline-per-task.json`) gives the per-task outcome
distribution:

| Passes out of 6 identical rollouts | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tasks | 26 | 1 | 5 | 2 | 4 | 8 | 54 |

**80 of 100 tasks return a constant outcome across six independent rollouts of the same harness.**
Under a fixed harness they carry zero within-harness variance. The only reason to run them is to
detect a *change* caused by an edit — which is precisely a question about the edit's mechanism, not
about the task's historical variance.

That determinism does not make a task safe to skip, and it separates two selection criteria that are
routinely conflated.

**A pass→fail transition can only occur on a task that is currently passing.** This is definitional,
not empirical: the regression surface of any harness edit is exactly the set of currently-passing
tasks. In the repeat above, 54 of those tasks pass 6/6, so their empirical pass rate satisfies
$\bar p_i = 1$ and their Bernoulli variance $\bar p_i(1-\bar p_i)$ is exactly $0$; a
variance-weighted sampler therefore assigns them weight $\approx \lambda/\sqrt{n_i}$, near zero.
**A selection rule built on outcome variance systematically under-samples the entire surface on which
regressions can appear.**

The audited AHE campaign shows that surface is where the loop's losses accumulate. Counting
transitions in the 40×5 matrix:

| Transition | fail→pass | pass→fail |
|---|---:|---:|
| iter1 → iter2 | 2 | 3 |
| iter2 → iter3 | 4 | 3 |
| iter3 → iter4 | 3 | 2 |
| iter4 → iter5 | 0 | 6 |
| **Total** | **9** | **14** |

Regressions outnumbered fixes 14 to 9, and the audit report records that across the five iterations
no task failing at baseline was passing at the end. The confirmation side of a round is therefore a
low-yield measurement, while the guard side carries the decision.

Together with §1.4 this fixes the shape of the design: the regression side dominates the round's
decision, it cannot be sampled by outcome variance, and §1.4 shows it cannot be named by the agent
either. It has to be derived from the mechanism of the edit.

### 1.4 The loop's self-declared predictions are informative about fixes and near-useless about regressions

AHE measures the evolve agent's own predictions against next-round ground truth over 9 rounds
(§4.4.2, App. D):

| Prediction type | Precision | Recall | Random baseline (P / R) |
|---|---:|---:|---|
| Fixes | 33.7% | 51.4% | 6.5% / 10.6% |
| Regressions | 11.8% | 11.1% | 5.6% / 5.4% |

Cumulatively, the agent issued 43 regression predictions of which 5 landed, while 40 regressions it
never named occurred. The local replication is consistent: 6 real flips out of 32 predicted fixes
across 4 manifests.

This is decisive for the design. A cost-effective loop that selects its evaluation set from the
agent's declared `predicted_fixes` and `risk_tasks` would inherit a regression recall of ~11% and
would simply become a cheaper way to accumulate regressions. **Any selection policy must derive the
guard set from something other than the agent's own guess.**

### 1.5 Why cost-effectiveness has become a correctness requirement, not an engineering nicety

"Rethinking the Evaluation of Harness Evolution for Agents" (arXiv:2607.12227) reports that automatic
harness evolution does not consistently outperform simple test-time-scaling baselines **under matched
feedback and inference budgets**, and shows limited generalization to held-out tasks. Once the field
requires budget-matched comparisons, the offline cost of a round stops being an implementation detail
and becomes the term that decides whether harness evolution wins the comparison at all. A method that
reaches the same harness quality for a third of the rollouts changes the outcome of exactly that
comparison.

### 1.6 Problem statement

> Given a harness-evolution loop with a fixed base model, a training set of $N$ tasks, and a total
> offline budget $B$ measured in dollars rather than in rounds, choose per round which tasks to
> evaluate and how many rollouts to spend on each, so as to maximize the held-out pass@1 of the
> harness the loop returns, subject to a bounded probability of accepting a net-harmful edit.

Writing $S_t$ for the set evaluated at round $t$, $k_i$ for the rollouts spent on task $i$, $c_i$ for
its unit cost, and $H^\star$ for the harness the loop returns:

$$\max_{\{S_t,\,k_i\}}\;\text{pass@1}\bigl(H^\star;\,D_{\text{test}}\bigr)
\quad\text{s.t.}\quad
\sum_{t=1}^{T}\sum_{i\in S_t}k_i\,c_i\;\le\;B,
\qquad
\Pr\bigl[\text{accept a net-harmful } \Delta_t\bigr]\;\le\;\alpha .$$

---

## 2. Related Work

### 2.1 Harness-optimization loops with full-set evaluation

**AHE** (arXiv:2604.25850) evolves seven file-level harness component types through
`evaluate → analyze → improve`, pairing every edit with a self-declared prediction verified against
the next round's task-level deltas. Evaluation is a full pass over the benchmark every round.
**Meta-Harness** (arXiv:2603.28052) searches over harness *code* with an agentic proposer that reads
the source, scores, and traces of all prior candidates through a filesystem; the search set is fixed.
**RHO** (arXiv:2606.05922) removes labels entirely, learning from unlabeled deployment trajectories
via self-validation, self-consistency, and pairwise self-preference. **TTHE** (arXiv:2607.08124)
moves adaptation to test time on unlabeled batches. **HarnessOpt-Bench** (arXiv:2608.06301)
benchmarks LLMs at the harness-optimization task itself.

*Relation.* HCE is orthogonal to what any of these optimizes and to what it edits. It changes only
the evaluation step, and can be dropped into AHE, Meta-Harness, or HarnessBank without touching the
proposer. AHE is the natural host because its change manifest already carries `predicted_fixes` and
`risk_tasks` and `compute_task_stability()` already derives stability classes — the substrate the
contract and dossier need is largely in place.

### 2.2 Candidate screening cascades

**AlphaEvolve** (arXiv:2506.13131) filters candidates through an evaluation cascade: cheap
syntax/sanity stages first, expensive full evaluation only for survivors. **HarnessBank**
(arXiv:2607.13683) applies the same idea to harness evolution with four sequential gates on a
*sampled* task subset — validity, activation (did the modification actually execute?), significance
(paired z ≥ 1.96), and gain — promoting only survivors to full training-set evaluation. **HARBOR**
(arXiv:2604.20938) uses multi-fidelity task subsets, escalating a candidate to a larger subset when
posterior-variance reduction justifies it.

*Relation.* These are *candidate-level* allocators: given several candidates, decide which deserves
more evaluation. HCE is an *instance-level* allocator: given one candidate, decide which tasks are
worth running at all. The two compose — a cascade decides *whether* to keep evaluating, HCE decides
*what to evaluate with the tokens the cascade releases*. HarnessBank's **activation gate** is the
closest existing primitive to the activation predicate of §3.2, but it is used post-hoc as a validity
check rather than a priori as a selection rule.

### 2.3 Adaptive validation-task selection — the closest prior work

**Task-CoEvolve** (arXiv:2608.20169) is the direct predecessor: it reduces per-candidate evaluation
cost in a Meta-Harness-style loop by sampling validation tasks each iteration with weight

$$w_i \;=\; \max\bigl(\bar p_i(1-\bar p_i),\; \ell_i\bigr) \;+\; \frac{\lambda}{\sqrt{n_i}},$$

then recovering comparable full-set scores with Hájek and anchored-difference estimators under
inclusion probabilities estimated by Monte Carlo. At a 20%
evaluation budget it matches full search on Terminal-Bench 2.1 (51.7% vs 52.8%), cuts input tokens
67–80%, and halves wall-clock; a 20% *uniform random* subset reaches only 48.4%.

*Relation — three concrete deltas.*

1. **Conditioning.** Task-CoEvolve's weights depend only on a task's *outcome history*; they are
   identical for every candidate edit at a given round. HCE conditions on the *mechanism of the
   specific edit*: two candidate edits at the same round get different evaluation sets. A change to
   shell-output truncation and a change to the finish-hook have disjoint blast radii, and no
   history-only weighting can express that.
2. **Objective asymmetry.** Bernoulli-variance weighting maximizes the information a task carries
   about a *ranking*. It assigns near-zero weight to tasks with $\bar p_i\approx 1$, which by definition is the entire
   surface on which a regression can occur (§1.3), and §1.4 shows the loop is already blind there.
   HCE separates a *power* objective (detect the intended gain) from a *one-sided risk* objective
   (bound harm) and samples each with its own rule.
3. **Adaptivity.** Task-CoEvolve fixes $m=\lceil\rho N\rceil$ before observing any result; the authors name
   "deciding this number during evaluation" as the natural extension. HCE allocates in waves with an
   anytime-valid stopping rule (§3.4), which is that extension.

**RoboPhD** (arXiv:2604.04347) takes the opposite route: abolish the validation set, evaluate three
agents per iteration on 20 randomly sampled training examples, and let Elo accumulate weak signals
across ~21 iterations under a fixed 1,500-evaluation budget. This establishes that many noisy rounds
can beat few clean ones. It is a strong baseline and a design warning: if random-shallow-many wins,
targeted selection must justify its overhead.

### 2.4 Evaluation-efficient benchmarking

**tinyBenchmarks** (arXiv:2402.14992) uses Item Response Theory to estimate a model's ability from
~100 curated items per scenario, with IRT-based anchor selection outperforming stratified sampling
and correctness clustering. **Active Testing** (arXiv:2103.05331) selects which test points to label
using acquisition functions, removing the resulting bias with the LURE estimator while reducing
variance. **Active Evaluation Acquisition** (arXiv:2410.05952) and **Neyman-allocation active
testing of LLMs** (arXiv:2605.10075) extend this to LLM benchmarking.

*Relation.* This literature estimates a *scalar ability of a single system* from few items, and its
selection criterion is item discrimination — again a property of the item, not of a proposed change.
HCE borrows the estimator machinery (inclusion-probability correction, control-variate anchoring) but
not the selection criterion. IRT discrimination is, however, a natural ingredient for the dossier's
prior (§3.1) and a natural ablation arm.

### 2.5 Budget allocation, racing, and best-arm identification

**F-Race / irace** (López-Ibáñez et al., *Operations Research Perspectives*, 2016) evaluates
configurations across instances and eliminates them by Friedman rank test as soon as evidence
permits. **Successive Halving / Hyperband** (arXiv:1603.06560) allocate a fixed budget across arms
under increasing fidelity. **ProTeGi / APO** (arXiv:2305.03495) selects prompt candidates with UCB,
Successive Rejects, and Successive Halving over minibatches. **GEPA** (arXiv:2507.19457) evaluates
mutations on small minibatches and keeps a *per-instance* Pareto frontier, promoting only survivors
to the anchor set. **Cost-Aware Multi-Objective Bandits** (arXiv:2608.04333) allocates a budget
across LLM configurations with a hypervolume-per-cost index and per-arm cost bounds, with regret and
Pareto-identification guarantees — but explicitly holds the instance set fixed at 1,000 items per
configuration.

*Relation.* All of these treat instances as interchangeable draws from a distribution, and allocate
*across arms*. HCE treats instances as non-interchangeable objects with individual, edit-dependent
informativeness, and allocates *within an arm*. The racing literature supplies the stopping-rule
formalism used in §3.4; GEPA's per-instance Pareto frontier is the closest existing acknowledgement
that instances are not interchangeable.

### 2.6 Regression test selection — the closest conceptual analogue outside ML

Selecting which tests to re-run after a code change is a forty-year-old problem in software
engineering. **Safe RTS** (Rothermel & Harrold, *TOSEM* 1997) guarantees that no test whose behavior
could differ is omitted, by static/dynamic change-impact analysis over the program's dependency
graph. **Ekstazi** (Gligoric et al., *ISSTA* 2015) makes this practical at file granularity.
**Yoo & Harman** (*STVR* 2012) survey minimization, selection, and prioritization.

*Relation.* The activation predicate of §3.2 is the analogue of change-impact analysis, and the
guard/confirmation split is the analogue of the safe-RTS guarantee versus prioritization for early
fault detection. The essential difference is that a harness edit has **no statically computable
dependency graph**: the "program" whose behavior changes is a stochastic policy, and the affected
"tests" can only be identified from observed execution. HCE therefore approximates the dependency
relation with a *mechanism profile* mined from traces (§3.1c) — an empirical, probabilistic
substitute for a safe over-approximation. The safety guarantee accordingly weakens from "no affected
test is omitted" to "the probability of omitting an affected task is bounded by the audit rate",
which is what §3.5 formalizes.

### 2.7 Cheap proxies for re-execution, and why they are not the answer

**Causal Agent Replay** (arXiv:2606.08275) models an agent run as a structural causal model and
re-executes forward under `do`-interventions to attribute failures. **The Replay Gap**
(arXiv:2608.08239) measures what happens when one instead *substitutes* recorded outputs into logged
trajectories: swapping models at early fork points causes 74–77% of branches to diverge at the very
first subsequent action, leaving 3–8% of replayed states valid; predicted patches show 0.00–0.11
similarity to reality; every success-relevant outcome flip is invisible to replay.

*Relation.* This closes off the most tempting shortcut. One cannot predict whether a harness edit
flips a task by replaying the old trajectory under the new harness — in a closed loop, the edit
changes the observations. **The rollouts have to be real.** That is precisely why the contribution
has to be *which* tasks to run, not *how* to avoid running them.

---

## 3. Technical Pipeline

### 3.0 Formulation

Fixed base model $M$. Training set $D=\{d_1,\dots,d_N\}$; task $d_i$ has rollout cost $c_i$ (dollars)
and outcome $x_i(H)\in\{0,1\}$ under harness $H$, or a pass rate over $k$ rollouts. The held-out set
$D_{\text{test}}$ is never touched by the loop. Round $t$ takes $H_{t-1}$ and produces an edit set
$\Delta_t$ and a candidate $H_t=\operatorname{apply}(\Delta_t,H_{t-1})$.

| Symbol | Meaning |
|---|---|
| $D_{\mathrm{pass}},\ D_{\mathrm{fail}},\ D_{\mathrm{unst}}$ | stability classes, dossier (a) in §3.1 |
| $A_t$ | activation set of $\Delta_t$, §3.2 |
| $T_t,\ R_t$ | tasks the contract declares as predicted fixes / at risk |
| $C_t,\ G_t,\ \mathcal{A}_t$ | confirmation set, guard set, audit sample, §3.3 |
| $S_t=C_t\cup G_t\cup\mathcal{A}_t$ | the round's evaluation set |
| $\pi_i=\Pr[\,i\in S_t\,]$ | inclusion probability of task $i$ |
| $\hat p_i^{\,\mathrm{hist}}$ | historical anchor for task $i$, §3.5 |
| $\rho,\ \tau,\ B$ | round budget fraction, audit period, total budget |

Per round the loop must output (a) a keep/revert decision for every edit in $\Delta_t$; (b) an
estimate $\hat S(H_t)$ of full-set performance on a scale comparable across rounds, so that
best-so-far selection remains meaningful; (c) trace evidence for round $t+1$.

The baseline cost of a full round is

$$C_{\text{full}}\;=\;k\sum_{i=1}^{N}c_i ,$$

and the target is a selection $S_t\subseteq D$ with per-task rollout counts $k_i$ satisfying

$$\sum_{i\in S_t}k_i\,c_i\;\approx\;\rho\,C_{\text{full}},\qquad \rho\in[0.15,\,0.4],$$

with (a) and (b) preserved.

### 3.1 Task dossier (built once, amortized, refreshed on anchor rounds)

A per-task record assembled from artifacts the loop already pays for. Marginal cost is one LLM judge
call per task, two orders of magnitude below one rollout.

**(a) Outcome history.** The per-round pass vector under every harness evaluated so far; derived
stability class (stable-pass / stable-fail / unstable / infra-only) and Bernoulli variance. AHE's
`compute_task_stability()` already computes this; it needs only to be persisted per task rather than
aggregated.

**(b) Structural fingerprint.** A codebase-agnostic description of failure mode, source of
difficulty, and technical scope, produced by an LLM judge from the task statement plus one observed
trajectory, then embedded. RHO's `difficulty_selector.py` judge prompt is a ready-made instantiation:
it returns `{difficulty ∈ [0,10], abstract_fingerprint}` with an explicit instruction to abstract away
repository, file, and library names, and RHO's `trajectory_digest.py` bounds the digest at ~10k
tokens with task-content redaction. The fingerprint supports two operations HCE needs: *nearest
neighbours of an affected task* (to extend a guard set beyond the agent's declared risk set) and
*coverage* of the pool (so the audit sample is stratified rather than uniform).

**(c) Mechanism profile — the new component.** A structured, machine-readable summary of which
harness mechanisms the task's trajectories actually exercise, extracted from the trace by
deterministic parsing rather than by an LLM where possible:
turn count and token trajectory; tool-call histogram; whether context compaction triggered; whether
any shell command exceeded the output or time threshold; whether a middleware hook fired, and which;
whether a skill was loaded; terminal state (verified success / early finalization / timeout /
infra exception); error-type histogram; number of distinct files edited; whether a test command was
run and whether it was filtered.

This is the field the activation predicate is matched against, and it is what makes selection
edit-conditional rather than history-conditional. It also matches this project's existing
failure-pattern taxonomy (`harness-failure-patterns-task-mapping.md`), which is already organized as
`process-friction / integrity / direction-correctness / capability-ceiling` with per-task evidence.

**(d) Cost profile.** Mean tokens, mean wall-clock, and mean dollars per rollout, plus their spread.
Needed because selection must maximize information *per dollar*, not per task: on Terminal-Bench 2
and LHTB, per-task cost spans more than an order of magnitude, and Task-CoEvolve reports that the
discriminative tasks are disproportionately the long multi-turn ones — the tension has to be priced,
not assumed away.

### 3.2 The evaluation contract

AHE already requires each edit to declare failure evidence, root cause, targeted fix, and predicted
impact, as free text plus two task-ID lists in `change_manifest.json`. HCE tightens this into a
schema whose fields are *checkable against traces*:

```jsonc
{
  "change_id": "c3",
  "component": "tools/shell_tools/run_shell_command.py",
  "scope": "local",                       // "local" | "global" — see §3.3
  "mechanism": "truncate stdout beyond 8k tokens and append a contract hint drawn from files adjacent to the command's cwd",
  "activation_predicate": {               // matched against dossier field (c)
    "any_command_output_tokens": ">8000"
  },
  "predicted_fixes":  ["task_a", "task_b"],
  "risk_tasks":       ["task_c"],
  "expected_direction": "increase verified-completion rate on long-output tasks"
}
```

Two properties matter.

**Activation is a selection rule *and* a falsification channel.** Matching the predicate against each
task's mechanism profile yields the **activation set** $A_t\subseteq D$, the tasks on which the edit
can fire at all. Tasks in $D\setminus A_t$ are, up to distribution shift and sampling noise,
unreachable by the edit.
After the round, the same predicate is checked against the *new* traces: an edit that was predicted
to fire on a task and did not fire is falsified without any outcome being needed, at zero additional
cost. This generalizes HarnessBank's activation gate from a post-hoc validity check into the primary
allocator.

**The agent writes the contract that decides how it is tested**, so the contract cannot be trusted on
its own. §4.4 details the countermeasures: activation is *verified*, never asserted; the guard set is
extended by fingerprint neighbourhood beyond the declared risk set; and an audit sample outside the
agent's control is always drawn.

### 3.3 The round partition: confirmation, guard, deferred

**Confirmation set $C_t$** — *does the intended effect appear?*

$$C_t\;\subseteq\;T_t\;\cup\;\bigl(A_t\cap(D_{\mathrm{fail}}\cup D_{\mathrm{unst}})\bigr),$$

ranked by expected transition probability per dollar. Its role is attribution and evidence
generation, not acceptance. §1.3 records 9 fail→pass events against 14 pass→fail across the audited
campaign, so the confirmation side is the low-yield half of the round: $C_t$ should be **small** —
enough to attribute the edit and to generate failure traces for the next round, not enough to certify
a gain.

**Guard set $G_t$** — *is aggregate harm bounded?* Three targeted sources plus an audit sample:

$$G_t\;=\;\underbrace{\bigl(A_t\cap D_{\mathrm{pass}}\bigr)}_{\text{(1) reachable and passing}}\;\cup\;\underbrace{R_t}_{\text{(2) declared}}\;\cup\;\underbrace{\mathcal{N}_\kappa(A_t)\cap D_{\mathrm{pass}}}_{\text{(3) fingerprint neighbourhood}}$$

$$\mathcal{A}_t\;\sim\;\operatorname{Strat}\bigl(D\setminus(C_t\cup G_t\cup A_t)\bigr)\quad\text{(4) random audit}$$

1. $A_t\cap D_{\mathrm{pass}}$ — tasks the edit can reach that currently pass deterministically. By
   definition this set contains every task on which a regression is possible (§1.3).
2. The declared $R_t$ (cheap to include, ~12% precision, so never the primary source).
3. $\mathcal{N}_\kappa(A_t)$, the $\kappa$ nearest tasks to $A_t$ in fingerprint space, restricted to
   $D_{\mathrm{pass}}$, covering the affected region the agent's predicate under-specified. This is
   the mechanism that repairs the 11% regression recall of §1.4 without relying on the agent's
   foresight.
4. $\mathcal{A}_t$, stratified by fingerprint cluster and by cost decile. Not optional: it keeps
   every inclusion probability $\pi_i>0$, which is what makes the §3.5 estimator unbiased, and it is
   the only channel that can catch a mechanism the activation predicate missed entirely.

**Scope handling.** Some edits have no narrow affected region. A longer system prompt changes the
context of *every* call, so the activation predicate holds everywhere and $A_t=D$. The contract's
`scope` field declares this, and the allocator must respond by degenerating to broad stratified
sampling with a larger audit fraction. A framework that quietly treats a global edit as local is
worse than no selection at all, so `scope: global` is enforced by a static rule (edits to
`systemprompt.md`, `code_agent.yaml`, or any always-on middleware are global by construction,
regardless of what the agent writes).

**Budget split.** §1.3 and §1.4 together imply that **the guard side should receive the majority of
the round's budget**: it carries more of the observed transitions, it is invisible to variance
weighting, and it is invisible to the agent's own predictions. The proposed default is

$$b_C : b_G : b_{\mathcal{A}}\;=\;25 : 55 : 20 .$$

This ratio is a hypothesis, not a result — Phase 0 (§5.1) measures it directly on the archived
campaign matrix, and it is an ablation arm in §5.4.

### 3.4 Wave allocation and anytime-valid stopping

Task-CoEvolve fixes m before seeing results. HCE runs the round in waves:

- **Wave 0 — activation check.** Cheapest possible evidence. For edits whose predicate is decidable
  from the harness code plus dossier alone (e.g. "fires only when a middleware hook is registered"),
  this costs nothing. Otherwise it is a handful of rollouts on the highest-activation tasks. An edit
  that fails Wave 0 is reverted without a single guard rollout.
- **Waves $1\dots W$.** Wave $w$ extends the evaluated set $S^{(w-1)}$ by drawing greedily on
  *expected decision-information per dollar*, with dossier (d) supplying the denominator:

$$i^\star\;=\;\operatorname*{arg\,max}_{i\,\in\,(C_t\cup G_t)\setminus S^{(w-1)}}\;\frac{\mathbb{E}\bigl[\,\mathcal{I}(i)\,\bigr]}{k_i\,c_i},$$

  where $\mathcal{I}(i)$ is the reduction in decision uncertainty contributed by task $i$. Within a
  wave, all rollouts run concurrently.
- **After each wave**, update two anytime-valid confidence sequences: $[L_w^{g},U_w^{g}]$ on the
  targeted-gain estimate over $C_t$, and $[\,\cdot\,,U_w^{h}]$ on the one-sided harm estimate over
  $G_t$. Stop when either the accept region or the revert region is entered, or the round budget is
  exhausted.

A confidence sequence for a parameter $\theta$ is a sequence of intervals satisfying

$$\Pr\bigl[\;\exists\,w\ge 1:\ \theta\notin[L_w,U_w]\;\bigr]\;\le\;\alpha ,$$

i.e. coverage holds *simultaneously* over all wave counts, so the interval may be inspected after
every wave and stopped on. Anytime-valid sequences (Howard et al.; see arXiv:2302.10108 for a
production account) are required rather than fixed-horizon tests because the stopping decision is
data-dependent — peeking after every wave with a fixed-horizon test inflates the type-I error
precisely in the regime where the effects are small. The cost is wider intervals; §4.5 is explicit that at realistic budgets this
machinery mostly serves as a *stopping* rule (stop early when the answer is obvious) rather than a
*certification* rule.

**Pairing.** All arms compared within a round must be evaluated on the *same* subset, which converts
the comparison from unpaired aggregates to paired per-task differences and removes between-task
variance for free. This project's own analyses already use McNemar's exact test on paired 500-task
comparisons; the same estimator applies to a 100-task subset with the same pairing. Subsets rotate
*across* rounds (for coverage) but never *within* a round (for pairing).

### 3.5 Cross-round score estimation and anchoring

Selection breaks the cross-round comparability the loop needs for best-so-far selection, so the
round's score is recovered with a history-anchored, inclusion-probability-corrected estimator:

$$\hat S(H_t)\;=\;\frac{1}{N}\sum_{i=1}^{N}\left[\;\hat p_i^{\,\mathrm{hist}}\;+\;\frac{\mathbf{1}\{i\in S_t\}}{\pi_i}\Bigl(x_i(H_t)-\hat p_i^{\,\mathrm{hist}}\Bigr)\right]$$

with $\hat p_i^{\,\mathrm{hist}}$ the task's anchor from the dossier and $\pi_i$ its inclusion
probability under the round's partition (closed-form for the deterministic components of the
partition, Monte Carlo for the randomized ones, following Task-CoEvolve). The bracketed term is a
control variate: the anchor carries the level and only the residual $x_i(H_t)-\hat p_i^{\,\mathrm{hist}}$
is paid for by sampling, so $\hat S$ is unbiased whenever $\pi_i>0$ for all $i$. Two HCE-specific
refinements:

- **Activation-conditioned anchors.** For $i\notin A_t$ the edit cannot fire, so
  $\mathbb{E}[x_i(H_t)]=\mathbb{E}[x_i(H_{t-1})]$, the anchor is unbiased and the residual is near
  zero. For $i\in A_t\setminus S_t$ (reachable but skipped) the anchor is biased in an unknown
  direction and must carry a widened interval. Reporting these two
  populations separately keeps the estimate honest about which part of the score is measured and
  which is imputed.
- **Rolling audit rather than periodic full passes.** Each round's audit sample rotates through the
  deferred set so that every task is re-measured at least once every $\tau$ rounds. This bounds anchor
  drift without the cost spike of a full pass, and it spreads the anchoring cost evenly. One genuine
  full pass is run at the *end* of a campaign, and that number — not any estimate — is what gets
  reported for the final harness.

**Anchor drift is the failure mode to monitor.** After many rounds of never re-measuring a task, its
anchor reflects a harness that no longer exists. At each rolling audit,

$$\mathrm{drift}_t\;=\;\bigl|\,\hat S(H_t)-S(H_t)\,\bigr|$$

is a first-class diagnostic, reported per round in §5.3.

### 3.6 Decision rule

An edit is kept when it activated where predicted, the guard-side harm estimate stays below
tolerance under the round's confidence sequence, and the *estimated full-set* score does not
decrease:

$$\operatorname{keep}(\Delta_t)\iff\underbrace{\operatorname{act}(\Delta_t)=\textsf{verified}}_{\text{(i)}}\ \wedge\ \underbrace{U_W^{h}\le h_{\max}}_{\text{(ii)}}\ \wedge\ \underbrace{\hat S(H_t)\ \ge\ \hat S(H_{t-1})}_{\text{(iii)}}$$

It is reverted when $U_W^{h}>h_{\max}$, or when Wave 0 shows it never activated.

Criterion (iii) is deliberately about the estimated full-set score and not the confirmation-set
score. The audited AHE run supplies the cautionary case: iteration 3's change was graded HARMFUL by
`evaluate_changes()` because its *named predictions* did not come true, yet the harness carrying it
scored the best result of the run (22/40), and acting on the HARMFUL verdict produced the run's worst
result (16/40). A verdict that grades predictions rather than outcomes will discard good edits. HCE
uses the contract to decide *where to look*, never to decide *what counts as success*.

### 3.7 Cost model and accounting

Every reported figure is a dollar figure, itemized by role, produced by the existing per-call
`UsageTracer` + `price_run.py` path. The cost of the loop's own machinery is charged to the method:

$$C_{\text{method}}\;=\;\underbrace{\sum_{t=1}^{T}\sum_{i\in S_t}k_i\,c_i}_{\text{rollouts}}\;+\;\underbrace{\sum_{t=1}^{T}C_{\mathrm{adb}}(S_t)}_{\text{trace analysis}}\;+\;\underbrace{C_{\mathrm{dossier}}+C_{\mathrm{contract}}+C_{\mathrm{est}}}_{\text{selection machinery}}$$

A method that removes 70% of the rollouts and spends 20% of the original budget on selection has
saved 50%. Savings are reported as the ratio $C_{\text{method}}/C_{\text{full}}$ against the
full-evaluation loop, never as a rollout count.

### 3.8 Algorithm

```
Input: seed harness H0, pool D, held-out D_test, budget B, per-round fraction ρ, audit period τ
Bootstrap: full evaluation of H0 on D at k rollouts  → dossier(a,b,c,d), anchors p̂^hist
for t = 1 .. T while budget remains:
    (Δ_t, contracts) ← Evolve(H_{t-1}, evidence corpus, prior verdicts)
    H_t             ← apply(Δ_t, H_{t-1})
    A_t             ← match(contract.activation_predicate, dossier.mechanism_profile)
    if any contract.scope == "global": widen A_t to D, raise audit fraction
    C_t, G_t, audit ← partition(A_t, stability classes, fingerprints, cost profile, ρ·B_t)
    for wave w = 0, 1, 2, ...:
        run batch(w) ⊆ C_t ∪ G_t ∪ audit ; update confidence sequences
        if accept-region or revert-region entered or round budget spent: break
    verify activation on new traces ; verdict ← DecisionRule(...)            # §3.6
    H_t ← rollback(H_t, verdict)  ;  Ŝ(H_t) ← anchored estimator             # §3.5
    if t mod τ == 0: refresh anchors from rolling audit ; report anchor_drift
    AgentDebugger over the *evaluated* traces only → evidence corpus
Final: one full evaluation of the best harness on D and on D_test           # reported number
```

### 3.9 Implementation path on the existing code

The fork already contains everything the loop needs except the selector:

| Needed | Existing hook |
|---|---|
| Stability classes | `evolve.py: compute_task_stability()` (line 783) |
| Per-round flip/regress sets | `evolve.py: compute_iteration_diff()` |
| Contract storage + attribution | `change_manifest.json`, `evolve.py: evaluate_changes()` (line 2239) |
| Per-call cost by role | `usage_tracer.py`, `scripts/price_run.py` |
| Fingerprint judge + digest | `retro-harness/src/rho/selection/{difficulty_selector,trajectory_digest,embedder}.py` |
| Subset dispatch | harbor task-list argument in the evaluation phase |

The concrete change is: a `selection/` module producing $S_t$ and $\pi$, a schema upgrade to
`change_manifest.json`, a `mechanism_profile` extractor over `nexau_in_memory_tracer.cleaned.json`,
and a wave loop wrapping the harbor dispatch. The evolve agent, the debugger, and the harness
substrate are untouched.

---

## 4. Challenges

### 4.1 Regression blindness is the binding constraint, and one regression class is undetectable by any targeted selection

Measured regression recall is 11.1% (§1.4). The mitigations in §3.3 — activation set, fingerprint
neighbourhood, stratified audit — address regressions that have a *mechanistic* trace. One class does
not: a longer system prompt or a larger always-on middleware degrades tasks through context length
and attention dilution, uniformly and without any distinguishing runtime signature. The audited run
shows the mechanism at work (prompt 548 B → 2,465 B, per-round cost +44%). For that class the
activation set is the whole pool and no targeting is possible; only the `scope: global` rule and the
random audit can catch it. **This is a hard limit of the approach and will be stated as such**, not
hidden. Quantifying what fraction of observed regressions falls in each class is Phase 0's second
deliverable.

### 4.2 Effect sizes are smaller than the per-round noise floor

The audited 40-task run has a ~10 pp noise floor and 22% unstable tasks at k=1; the 100-task repeat
finds 20% stochastic tasks at k=2 (§1.3). Round-to-round harness effects are 2–7 pp. **No selection
policy can make an individual round statistically certifiable at a research budget.** Two
consequences the proposal accepts explicitly:

- The claim is about *end-of-campaign held-out pass@1 per dollar*, not per-round significance.
- Allocation is two-dimensional: (how many tasks) × (how many rollouts each). Fewer tasks at higher k
  may dominate more tasks at k=1 at equal cost, because paired per-task comparison removes
  between-task variance while repeats remove within-task variance. The (N, k) frontier at fixed
  dollars is an experiment (§5.4), not an assumption.

### 4.3 Cutting evaluation cuts evidence

The evolve agent's next proposal is formed from traces the evaluation produced. Running 25% of the
pool yields 25% of the fresh failure evidence, which may reduce improvement-per-round even as it
improves improvement-per-dollar. Mitigations: the evidence corpus is cumulative, so older traces for
deferred tasks remain readable; the confirmation set is deliberately biased toward still-failing
tasks, which is where new evidence is generated anyway; and `agent_debugger` cost scales down with
the number of evaluated tasks, freeing budget for more rounds. Whether the net effect is positive is
the primary question of Phase 1 and will be reported as two separate curves (per round, per dollar)
so that a negative result is visible rather than averaged away.

### 4.4 The agent writes the contract that decides how it is tested

Contract gaming is a real incentive: a narrow activation predicate makes an edit cheap to validate
and hard to falsify. Countermeasures, in decreasing order of strength: (i) activation is *verified*
against post-round traces, never accepted as asserted, and a predicate that under-predicts activation
is itself a falsification event; (ii) the audit sample is drawn outside the agent's control and its
membership is never shown to the agent; (iii) `scope: global` is assigned by a static rule for
always-on components regardless of what the agent declares; (iv) the guard set is extended by
fingerprint neighbourhood, which the agent does not choose. A dedicated adversarial ablation — run
the loop with a contract-writing agent explicitly prompted to minimize its own evaluation set — is
included in §5.4.

### 4.5 Selection-induced overfitting and Goodhart pressure

If the loop preferentially evaluates where its own predictions say to look, it optimizes its own
predictions. §3.6's decision rule is the structural defence: the contract chooses *where to look*, the
estimated full-set score decides *what counts as success*, and the held-out set is never used inside
the loop. Independently, "Rethinking the Evaluation of Harness Evolution" (arXiv:2607.12227) shows
that same-set evolution overstates gains generally; this project's own AHE runs evolve on the tasks
they are scored on and flag it as a limitation. Every arm in §5 uses a strict train/test split.

### 4.6 Non-additive component interactions break per-edit attribution

AHE's component ablation finds the three positive single-component gains sum to +11.1 pp while full
AHE gains +7.3 pp, and that the memory-only variant beats full AHE on the Hard tier — components
push toward redundant verification and interfere. Per-edit contracts validated on disjoint subsets
therefore do not compose. Mitigation: one edit per round by default (which the audited run already
does in practice — one change per iteration); when multiple edits ship, their activation sets must
be checked for overlap, and overlapping edits are evaluated jointly rather than attributed
separately. A periodic *interaction audit* round evaluates the accumulated harness against the seed
on a full pass.

### 4.7 Cost heterogeneity works against selection

Task-CoEvolve reports that variance weighting concentrates budget on long-running multi-turn tasks —
the expensive ones. If the informative tasks are systematically the costly ones, selection by task
count saves less than it appears to. Dossier (d) and per-dollar ranking address this directly, and
the proposal will report savings in dollars and wall-clock, never in task counts.

### 4.8 Cold start and infrastructure noise

Round 1 has no anchors and no stability classes, so one bootstrap full pass is mandatory and is
charged to the method's budget. Separately, infra exceptions must not be silently absorbed: the
audited run had zero exceptions, but the earlier run lost an entire iteration to a middleware import
error that scored all 40 tasks 0. A selection policy that reads such a round as "the edit is
catastrophic" would revert a fine edit; validity gating (HarnessBank's first gate) is a prerequisite,
not an optional refinement.

---

## 5. Experiment Plan

### 5.1 Phase 0 — offline policy replay on archived outcome matrices (cost ≈ \$0, start immediately)

Existing archived artifacts make it possible to evaluate *selection policies* without spending a
single rollout:

| Artifact | Shape | Use |
|---|---|---|
| `benchmark40-ahe-run2-per-task.json` | 40 tasks × 5 iterations, k=1 | replay per-round decisions |
| `benchmark40-ahe-run2-manifest-{1..5}.json` | declared fixes and risks | contract fidelity |
| `benchmark40-ahe-run2-evaluation-{2..5}.json` | realized verdicts | ground-truth decisions |
| `benchmark100-baseline-per-task.json` | 100 tasks × 6 rollouts, one harness | per-task noise floor, stability classes |
| `swebench500-terra-{high,medium}-compare.json` | 500 tasks × 3 arms × 2 effort tiers, with `cost_usd` per task | per-task cost distribution; false-discovery rate of selection policies under a null effect |

Deliverables:

- **D0.1 — the decision-fidelity curve.** For each policy in {uniform random, Bernoulli-variance
  (Task-CoEvolve), stability-stratified, contract-driven (oracle activation), contract-driven
  (agent-declared)} and each $\rho\in\{0.1,\dots,1.0\}$: the probability that the round's keep/revert decision
  matches the full-evaluation decision, and the sign-error rate of the anchored estimator.
- **D0.2 — the regression taxonomy.** Classify every observed regression in the archives as
  mechanistically-traceable or global-degradation, giving the fraction of regressions that targeted
  selection can reach even in principle (§4.1).
- **D0.3 — the guard/confirmation yield ratio.** From the 40×5 matrix, the share of each round's
  decision-relevant transitions originating in the currently-passing versus currently-failing class,
  which sets the §3.3 budget split on measurement rather than on argument.
- **D0.4 — the false-discovery rate of each policy under a null effect.** The 500-task terra arms
  differ by less than the run-to-run standard deviation (0.76–1.04 pp), so they function as a null
  panel: how often does each subset policy report a difference where the full set reports none?
- **D0.5 — the per-task cost distribution.** `cost_usd` is recorded per task in the 500-task compare
  files, which gives the dossier's cost profile (§3.1d) and quantifies the §4.7 tension between
  informative tasks and cheap tasks before any live run.

*Stated limitation.* This replay assumes the campaign trajectory is unchanged by a different
decision at round t, which is false: a different decision at round 3 changes rounds 4–10. It bounds
*per-round* decision fidelity only. That is exactly the caution of "The Replay Gap"
(arXiv:2608.08239), and it is why Phase 0 is used to *design* the policy and never to claim an
end-to-end result.

### 5.2 Phase 1 — live campaign at small scale (SWE-bench-40, 5 rounds)

Reproduces the audited setting exactly so results are comparable to a known cost baseline.

| Arm | Per round | Expected cost / replicate |
|---|---|---|
| A. Full-set AHE (reference) | 40 tasks | ~\$56 (measured) |
| B. Uniform random at matched $\rho$ | $\rho\cdot 40$ tasks | ~\$25 |
| C. Bernoulli-variance (Task-CoEvolve reimplementation) | $\rho\cdot 40$ | ~\$25 |
| D. **HCE** | contract-driven, $\rho\cdot 40$ | ~\$25 + selection overhead |

Three seeds per arm, $\rho=0.3$, $k=2$ (the audited run's $k=1$ is a known handicap: at $k=1$ the
"partial pass" signal the evolve query calls its highest-leverage target does not exist at all).
Strict split: evolve on 40, score on a disjoint 60-task held-out set. Total ≈ \$1,200 including
held-out passes.

Primary readout: held-out pass@1 as a function of **dollars spent**, not rounds elapsed.

### 5.3 Phase 2 — live campaign at scale

| Setting | Pool | Rounds | Arms | Note |
|---|---|---|---|---|
| SWE-bench Verified | 300 train / 200 held-out | 8 | A–D | primary result |
| Terminal-Bench 2 | 89, AHE's split protocol | 10 | A, D | comparability with AHE's published campaign |

Metrics reported per arm:

1. **Held-out pass@1 per dollar** (primary) and at matched total dollars.
2. **Decision fidelity** against a full-evaluation oracle on a subsample of rounds where the full
   pass is also run (the price of knowing the answer, paid on a few rounds only).
3. **Regression escape rate** — the fraction of accepted edits that a later full pass shows to be net
   harmful.
4. **Informative-rollout rate** — the fraction of the round's rollouts landing on tasks that changed
   state. Full-set AHE's own value is 14.4% (§1.2); this is the number the method exists to raise.
5. **Anchor drift** at each rolling audit (§3.5).
6. **Wall-clock per round**, since sequential waves reduce parallelism and can cost time even while
   saving dollars — a real trade-off that must be reported, not omitted.
7. **Budget-matched test-time-scaling baseline** (best-of-n on the seed harness at equal dollars), as
   arXiv:2607.12227 requires.

### 5.4 Ablations

| Ablation | Question |
|---|---|
| − activation predicate (history-only weighting) | Is edit-conditioning worth its overhead over Task-CoEvolve? |
| − guard set (variance-weighted only) | Does variance weighting miss regressions at scale, as §1.3 predicts? |
| − fingerprint neighbourhood (declared risks only) | Can the 11% regression recall be repaired by structure? |
| − audit sample | How fast does anchor drift accumulate without it? |
| − wave stopping (fixed m) | Value of the adaptivity Task-CoEvolve leaves open |
| Budget split ∈ {50/30/20, 25/55/20, 10/70/20} | Confirm or refute the guard-heavy default of §3.3 |
| $\rho\in\{0.1,0.2,0.3,0.5\}$ | Where does decision fidelity break? |
| Adversarial contract writer | Contract gaming (§4.4) |
| Global-scope edits only | Graceful degradation (§4.1) |

### 5.5 What would falsify the proposal

Stated in advance:

- **Random-$\rho$ matches HCE at equal dollars.** RoboPhD's result (random shallow-many beats
  careful-few under a fixed evaluation budget) makes this a live possibility. If arm B ties arm D on
  held-out pass@1 per dollar across both scales, the contribution reduces to "subset evaluation
  works", which Task-CoEvolve already established.
- **Regressions turn out to be as detectable by variance weighting as by mechanism.** If at scale
  the tasks that regress are predominantly the unstable ones rather than the deterministically
  passing ones, the guard-heavy budget split loses its basis and variance weighting is sufficient.
- **Evidence starvation dominates.** If improvement-per-round degrades faster than cost-per-round
  falls (§4.3), the loop reaches a worse harness for the same money, and the honest conclusion is
  that full-set evaluation is buying evidence, not just precision.

### 5.6 Sequencing

| Phase | Depends on | Output |
|---|---|---|
| 0 — offline replay | archived matrices only | D0.1–D0.3; the policy that Phase 1 runs |
| 0.5 — dossier tooling | RHO judge port, mechanism extractor | dossier over SWE-bench-40 and -100 |
| 1 — small live campaign | Phase 0.5 | held-out pass@1 per dollar, 4 arms × 3 seeds |
| 2 — scaled campaign | Phase 1 | primary result, ablations |
| 3 — transfer | Phase 2 | frozen-harness transfer to a second benchmark and base model |

Phase 0 requires no API budget and no sandbox capacity, and is the correct first step: it can
falsify or sharpen the central design decision (§3.3's partition) before any rollout is spent.

---

## 6. Open design decisions

1. **Name.** HCE / CEHE are placeholders.
2. **Host loop.** AHE fork (proposed, because the manifest and stability machinery already exist) vs.
   Meta-Harness (proposed by Task-CoEvolve's setting, which would make the head-to-head comparison
   exact).
3. **Mechanism profile extraction** — deterministic trace parsing (cheap, brittle, needs one
   extractor per harness framework) vs. an LLM extractor (general, ~\$0.01/task, less reproducible).
   Recommendation: deterministic for the fields that can be parsed, LLM only for the residual.
4. **Whether Phase 2 runs Terminal-Bench 2 or SWE-bench Verified as the primary.** TB2 is where AHE
   and Task-CoEvolve both publish, which makes comparison direct; SWE-bench Verified is where this
   project already has 500-task per-task matrices at two effort tiers, which makes Phase 0 far
   stronger.

---

## References

Harness optimization
- Agentic Harness Engineering — arXiv:2604.25850
- Meta-Harness: End-to-End Optimization of Model Harnesses — arXiv:2603.28052
- Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference — arXiv:2606.05922
- HarnessBank: Semantic Gene-Bank Search with Gated Verification — arXiv:2607.13683
- HARBOR: Automated Harness Optimization — arXiv:2604.20938
- TTHE: Test-Time Harness Evolution — arXiv:2607.08124
- HarnessOpt-Bench: Evaluating LLMs at Harness Optimization — arXiv:2608.06301
- Rethinking the Evaluation of Harness Evolution for Agents — arXiv:2607.12227
- Training-Free Group Relative Policy Optimization — arXiv:2510.08191

Evaluation-budget allocation
- Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection — arXiv:2608.20169
- RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets — arXiv:2604.04347
- Cost-Aware Multi-Objective Bandits — arXiv:2608.04333
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning — arXiv:2507.19457
- Automatic Prompt Optimization with "Gradient Descent" and Beam Search — arXiv:2305.03495
- AlphaEvolve: A coding agent for scientific and algorithmic discovery — arXiv:2506.13131
- Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization — arXiv:1603.06560
- The irace Package: Iterated Racing for Automatic Algorithm Configuration — López-Ibáñez et al., *Operations Research Perspectives*, 2016

Evaluation-efficient benchmarking and estimation
- tinyBenchmarks: evaluating LLMs with fewer examples — arXiv:2402.14992
- Active Testing: Sample-Efficient Model Evaluation — arXiv:2103.05331
- Active Evaluation Acquisition for Efficient LLM Benchmarking — arXiv:2410.05952
- Active Testing of LLMs via Approximate Neyman Allocation — arXiv:2605.10075
- Anytime-Valid Confidence Sequences in an Enterprise A/B Testing Platform — arXiv:2302.10108

Regression test selection
- Rothermel & Harrold, A Safe, Efficient Regression Test Selection Technique — *ACM TOSEM*, 1997
- Gligoric, Eloussi & Marinov, Practical Regression Test Selection with Dynamic File Dependencies (Ekstazi) — *ISSTA*, 2015
- Yoo & Harman, Regression testing minimization, selection and prioritization: a survey — *STVR*, 2012

Replay and attribution caveats
- The Replay Gap: Static Evaluation of Model Switching in LLM Agents Scores the Wrong World — arXiv:2608.08239
- Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures — arXiv:2606.08275

Internal measurements cited in §1
- `reset-free-coding-agent-harness/docs/benchmark40-ahe-cost-audit-report.md`
- `reset-free-coding-agent-harness/docs/results/benchmark40-ahe-run2-per-task.json`
- `reset-free-coding-agent-harness/docs/results/benchmark100-baseline-per-task.json`
- `reset-free-coding-agent-harness/docs/results/swebench500-terra-{high,medium}-compare.json`
- `reset-free-coding-agent-harness/docs/harness-failure-patterns-task-mapping.md`

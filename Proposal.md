# TL;DR
Current automatic harness optimization pipelines (like AHE or Meta-Harness) are prohibitively expensive because they re-evaluate the entire training set for every single proposed modification.

This proposal replaces the full-set evaluation with a hypothesis-conditioned subset. Whenever the agent modifies the harness, it must output a machine-readable "Evaluation Contract" specifying the exact mechanism and trigger conditions of the change. By matching these conditions against historical task profiles, we can surgically select only the tasks that will actually be affected. By partitioning this subset into Improvement Validation, Regression Defense, and a Stratified Audit, and coupling it with early-stopping statistical checks, our goal is to reduce offline evaluation costs by 70% to 80% without sacrificing final Pass@1 performance on the hold-out set.

# 1 Motivation
## 1.1 The Evaluation Bill is Too High
In a baseline run of Agentic Harness Engineering (AHE) on 40 tasks over 5 iterations, 94% of the API costs were spent on executing evaluations and analyzing trajectories. Only 3% to 6% of the budget actually went to the LLM agent thinking about how to improve the harness. Furthermore, as the harness grows (e.g., longer system prompts), single-task costs scale up. To make automated harness evolution practical, we must reduce the number of tasks evaluated per iteration.

## 1.2 Full-Set Evaluations Have Low Information Density
We found that across 5 iterations of full-set evaluations, 72.5% of the tasks never changed their outcomes (they always passed or always failed). Only about 14.4% of the task-iteration pairs contained actual state transitions (Pass $\to$ Fail or Fail $\to$ Pass). If we could precisely target the tasks likely to flip, we could gather the exact same decision-making signal for a fraction of the cost.

## 1.3 Regressions (Pass -> Fail) Only Happen to Passing Tasks
A crucial intuition: A task can only "regress" if it is currently passing. If we use standard variance-based sampling (which favors tasks that historically flip-flop), we assign near-zero weight to tasks that have a 100% pass rate. However, these consistently passing tasks are exactly where regressions occur! Therefore, we cannot rely solely on historical variance to catch regressions; we must select tasks based on the mechanism of the new code change.

# 2 Related Works
- Full-Set Harness Optimizers (AHE, Meta-Harness): https://arxiv.org/abs/2604.25850, https://www.alphaxiv.org/abs/2603.28052
These frameworks search and optimize harness code iteratively but evaluate candidates on the entire dataset. HCE can act as a drop-in replacement for their evaluation step.

- Task-CoEvolve (Variance-based Sampling): https://arxiv.org/pdf/2608.20169
This is our most direct baseline. It selects subsets by weighting tasks based on their historical Bernoulli variance (tasks that oscillate get sampled more). HCE improves upon this by realizing that variance sampling misses regressions on stable tasks. HCE samples based on mechanism activation rather than just historical variance, and separates the budget aggressively to prioritize regression defense. 

- Verify Smarter, Evolve Further: Efficient Harness Evolution through Behavior-Aware Verificatio: https://arxiv.org/pdf/2608.27311 (/fs/nexus-projects/MeMas/projects/Harness-Evo/HarnessLens)
Another direct baseline. It firstly analyze each task and trajectory in the benchmark, then selects a subset and only evaluate on this subset. Although this project doens't claim any cost reduction. Instead, it only focus on better performance.

- Regression Test Selection in Software Engineering (e.g., Ekstazi): Traditional SE uses static code analysis to find which tests are affected by a code diff. Since LLM agents are stochastic, we cannot build static dependency graphs. HCE adapts this idea by using empirical "Mechanism Profiles" (runtime trajectory logs) as a probabilistic substitute for dependency graphs.

3 Technical Pipeline
## Formal Problem Statement
Given a fixed base model, a training set $D$ with $N$ tasks, and a total offline budget $B$ (measured in dollars, not iterations), we need to select an evaluation subset $S_t$ and the number of rollouts $k_i$ per task $i$ with unit cost $c_i$ for each iteration $t$. Our goal is to maximize the final harness $H^\star$'s Pass@1 on a unseen hold-out set $D_{\text{test}}$, while bounding the probability of accepting a harmful modification:

$$\max_{\{S_t,\,k_i\}}\;\text{pass@1}\bigl(H^\star;\,D_{\text{test}}\bigr) \quad\text{s.t.}\quad \sum_{t=1}^{T}\sum_{i\in S_t}k_i\,c_i\;\le\;B, \qquad \Pr\bigl[\,\mathrm{harm}(\Delta_t)\,\bigr]\;\le\;\alpha$$

## Step 1: Task Profiling (Building the Database) 
Before we start optimizing, we run the initial seed harness on the full training set for k times (for example, k could be 3). This generates a database of Task Profiles. For each task, we log:
- Historical Outcome: Does it consistently pass, consistently fail, or fluctuate?
- Structural Fingerprint: An LLM-generated vector embedding describing the task's difficulty and characteristics.
- Mechanism Profile (The Key Addition): What actually happens when this task runs? Does it output more than 8,000 tokens? Does it trigger a specific middleware hook? Does it encounter Python syntax errors?
- Cost Profile: How much does this specific task cost to run on average?

## tep 2: The Evaluation Contract
When the Evolve Agent proposes a code modification (e.g., adding a truncation feature for long outputs), it is forced to submit a structured JSON contract. The most important field is the activation_predicate (the trigger condition). For example: "activation_predicate": {"any_command_output_tokens": ">8000"}. We use this predicate to query our Task Profiles. If a task has never produced an output $>8000$ tokens in the past, it is highly unlikely to be affected by this change, so we can safely skip evaluating it. We call the set of tasks that do match the condition the Activation Set ($A_t$).

## Step 3: Budget Partitioning ($C_t, G_t, \mathcal{A}_t$)
Suppose we only have the budget to evaluate 30% of the tasks. We divide this limited budget into three distinct buckets, heavily prioritizing defense.
- Improvement Validation ($C_t$): To see if the change actually helps. We select tasks from the Activation Set ($A_t$) that have historically failed. We allocate a small budget here (e.g., 25%)—just enough to verify efficacy.
- Regression Defense ($G_t$): To ensure we didn't break anything. We select tasks from the Activation Set ($A_t$) that currently pass. Because the agent might write a loophole in its contract, we also use KNN on the structural fingerprints to pull in structurally similar tasks just in case. We allocate the majority of the budget here (e.g., 55%).

$$G_t\;=\;\underbrace{\bigl(A_t\cap D_{\mathrm{pass}}\bigr)}_{\text{Activated \& Passing}}\;\cup\;\underbrace{R_t}_{\text{Agent's Guesses}}\;\cup\;\underbrace{\mathcal{N}_\kappa(A_t)\cap D_{\mathrm{pass}}}_{\text{KNN Neighbors (Safety Net)}}$$

- Stratified Audit ($\mathcal{A}_t$): To catch unknown unknowns and keep our math rigorous. We randomly sample a few tasks from the remaining pool (tasks that didn't trigger the condition). This guarantees every task in the dataset has a non-zero probability ($\pi_i > 0$) of being tested. We allocate the rest of the budget here (e.g., 20%).

## Step 4: Wave Execution and Early Stopping
We don't run the selected tasks all at once. We dispatch them in "waves" (e.g., 5 tasks at a time), picking the most cost-effective tasks first.

1. Dynamic Rollout Allocation ($k_i \ge 1$): During wave execution, the rollout count $k_i$ is dynamic. To confirm a fix on a highly unstable task in $C_t$, we might dynamically assign $k_i = 3$. To quickly verify that a historically stable, simple task in $G_t$ hasn't broken, $k_i = 1$ is sufficient. Tasks are prioritized greedily by their expected information gain per dollar:

2. The Peeking Problem & AVCS: 
Normally, checking the results after every wave to decide if we should stop testing is a statistical cardinal sin (it drastically inflates the false positive rate). To legally "peek" at the data and stop early to save money, we use Anytime-Valid Confidence Sequences (AVCS).
Let $\Delta_i = x_i(H_t) - \hat{p}_i^{\text{hist}}$ be the paired score difference for task $i$. After testing $n$ tasks, we calculate the sample mean $\hat{\mu}_n$ and variance $\hat{\sigma}_n^2$. The AVCS provides a dynamic upper bound ($U_n$) and lower bound ($L_n$) that are guaranteed to contain the true mean across all possible sample sizes:
$$L_n = \hat{\mu}_n - 1.7 \sqrt{\frac{\hat{\sigma}_n^2}{n} \log\left(\frac{\log(n) + 1}{\alpha}\right)}$$
$$U_n = \hat{\mu}_n + 1.7 \sqrt{\frac{\hat{\sigma}_n^2}{n} \log\left(\frac{\log(n) + 1}{\alpha}\right)}$$

3. Early Stopping Logic: After every wave, we check the AVCS bounds:If $U_n < 0$: Even in the best-case statistical scenario, the new harness performs worse. We confirm a regression. We trigger an early stop, rollback the code change immediately, and save the budget for the remaining waves.If $L_n > 0$: Even in the worst-case scenario, the new harness is an improvement. We can safely stop early and keep the change.

## Step 5: Scoring with Inverse Probability Weighting (IPW)
Since we only evaluated a subset of tasks, how do we compare this iteration's performance to the all-time best score? We use a statistically unbiased estimator.
$$\hat{S}(H_t)\;=\;\frac{1}{N}\sum_{i=1}^{N}\left[\;\hat{p}_i^{\,\mathrm{hist}}\;+\;\frac{\mathbf{1}\{i\in S_t\}}{\pi_i}\Bigl(x_i(H_t)-\hat{p}_i^{\,\mathrm{hist}}\Bigr)\right]$$

What this means:
- For tasks we didn't test, we just use their historical score ($\hat{p}_i^{\,\mathrm{hist}}$). For tasks we did test ($i \in S_t$), we look at how their score changed. We then divide that change by the probability that the task was selected ($\pi_i$).
- Why? Because if a task only had a 10% chance of being tested ($\pi_i = 0.1$), but it failed, it mathematically "represents" 10 similar tasks that probably also failed. Dividing by $\pi_i$ amplifies the signal of sampled tasks to accurately estimate the global score.

If this estimated global score $\hat{S}(H_t)$ improves, and no regressions were detected, we keep the code change.

# 4 Challenges
- Agent Gaming the Contract: The agent might write an overly narrow trigger condition to bypass evaluation.
Mitigation: We run a "Zero-Cost Falsification" check. If the agent claims a code change fixes Task A when output $>8k$, but the actual execution log shows Task A only reached 2k tokens, we automatically reject the change without even looking at the Pass/Fail score. We also use KNN (Step 3) to force-expand the blast radius beyond the agent's control.
- Global Scope Modifications: If the agent changes the core System Prompt, every task is affected. The activation condition becomes the whole dataset.
Mitigation: This is a hard boundary of the method. When a global file is edited, the system bypasses the contract and falls back to a large-scale stratified random sample. We accept that global changes simply cost more to verify.
- Low Signal-to-Noise Ratio: Because we test fewer tasks, a genuine 2% improvement might be drowned out by infrastructure noise or LLM variance.

Mitigation: We do not demand strict statistical significance to accept a change (which is impossible under tight budgets). We use confidence intervals strictly as a veto mechanism (to reject obvious regressions). As long as the estimated score $\hat{S}(H_t)$ does not drop, we keep the change and let the hold-out set serve as the final judge at the very end of the experiment.

# Notes
- Codebase we could refer: https://github.com/china-qijizhifeng/agentic-harness-engineering
- An important thing to note is that in AHE’s original code, the metrics it uses is different from the common Pass@k or mean@k. We may need to change that metrics to mean@k to fit our framework.
- The “KNN neighbor” part might be over-engineering. We could remove it first to quickly validate our idea.

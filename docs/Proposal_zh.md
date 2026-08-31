# 面向成本有效的 harness 演化：以假设为条件的评测选择

**方法名（暂定）**：HCE — Hypothesis-Conditioned Evaluation，以假设为条件的评测
**项目名（暂定）**：CEHE — Cost-Effective Harness Evolution
**状态**：提案草稿，2026-08-23
**代码基线**：`cost-effective-harness-evo/`，由 `agentic-harness-engineering`（AHE）复制而来

本文为 [Proposal.md](Proposal.md) 的中文版本，章节编号一一对应。

---

## 0. 概述

现有的 harness 优化流程（AHE、Meta-Harness、HarnessBank）在每一轮都把候选 harness 放到**完整**训练集合上重新评测。本提案把这一步替换为**以当轮修改为条件**的评测集合。

每一轮，evolve agent 需要输出一份可机器校验的**评测契约**，其中写明本次修改的作用机制、该机制能够触发的运行时条件、预期修复的任务、以及预期存在退化风险的任务。**任务档案**由流程已经支付过的轨迹构建，记录每个训练任务的历史结果、结构指纹、机制画像与单次 rollout 成本。契约与档案共同把训练集合划分为三个承担不同统计作用的部分：**改进验证集合**（预期效应是否出现）、**退化防护集合**（总体损害是否受控）、**延迟集合**（由历史结果插补，按轮换审计）。rollout 按波次分配给前两者，停止规则采用随时有效置信序列；跨轮可比的分数由锚定并按纳入概率校正的估计量还原。

目标主张有两条：**在留出集合 pass@1 达到同等水平的前提下，离线成本降低到原来的 1/3 到 1/5**；在相同金额预算下，该流程得到的留出集合 pass@1 高于全量评测，因为同样的预算被用于更多轮次与更多重复采样。单轮统计显著性不在主张范围内：在任何现实预算下，单轮效应量都小于噪声下限。

---

## 1. 动机

### 1.1 离线成本的构成

AHE 在 Terminal-Bench 2 上的参考实验为 89 题 × k=2 rollout × 10 轮，约 1780 条 rollout，单题超时上限 3600 秒，并发 96，总墙钟约 32 小时（AHE，arXiv:2604.25850，§4.1 与附录 A）。该预算中没有一部分用于优化器本身。

本项目在自有基础设施上对一次 5 轮、40 题、k=1 的 AHE 运行做了全量计价（`reset-free-coding-agent-harness/docs/benchmark40-ahe-cost-audit-report.md`）：

| 角色 | 调用次数 | 成本 | 占比 |
|---|---:|---:|---:|
| `code_agent`（rollout） | 3307 | \$37.61 | 67% |
| `agent_debugger`（轨迹蒸馏） | 1380 | \$15.28 | 27% |
| `explore_agent` | 24 | \$1.57 | 3% |
| `evolve_agent`（优化器） | 81 | \$1.53 | 3% |
| **合计** | | **\$55.99** | |

**账单的 94% 用于评测以及对评测产物的分析，执行 harness 工程的组件占 3%。** 优化器变得更便宜或更强对总成本几乎没有影响；唯一有效的变量是每轮购买多少评测。

另有两项成本性质叠加在上面：

- **单轮成本随 harness 增长。** 在上述运行中，任务集合不变，单轮成本从第 1 轮到第 5 轮上升 44%，因为 `systemprompt.md` 从 548 B 增长到 2465 B 并新增一个 skill 包，每条 rollout 都携带更长的 prompt。pass 率曲线不显示这一项。
- **分析成本与评测成本成比例。** `agent_debugger` 读取的是评测产生的轨迹。减少被评测的任务数，按比例下降的是全部 94% 的账单。

### 1.2 全量评测的信息密度

第 $t$ 轮需要从评测中得到的是一个决策——保留还是回退 $\Delta_t$ 中的修改——以及供下一轮使用的证据。全量评测提供的是总体 pass@1 的高精度估计，其中大部分精度来自结果已经确定的任务。

对已归档的 40×5 结果矩阵（`docs/results/benchmark40-ahe-run2-per-task.json`）统计：

| 统计量 | 数值 |
|---|---|
| 五轮内未发生状态转移的任务 | **29 / 40（72.5%）** |
| 整个实验期间至少变化一次的任务 | 11 / 40（27.5%） |
| 携带状态转移的任务-轮次 | 23 / 160（**14.4%**） |
| 单轮发生状态转移的任务数 | 40 题中的 5、7、5、6 |

若事先已知哪 11 个任务会发生状态转移，则以 27.5% 的 rollout 成本可以复现整个实验期间观测到的全部状态转移。其余 72.5% 的评测在五轮中返回同一个值。

AHE 在 Terminal-Bench 2 上的密度与此接近：9 个评测轮次中实际退化 45 次，实际修复次数与之相当，89 题中单轮状态转移约 12 至 14 次，同样约为 14%（AHE §4.4.2、附录 D）。

### 1.3 结果确定性与退化可能发生的范围

对同一 harness 在 100 道分层抽样的 SWE-bench Verified 任务上运行三次、每次 k=2（每题 6 条 rollout，`docs/results/benchmark100-baseline-per-task.json`），得到每题的通过次数分布：

| 6 条相同 rollout 中的通过次数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 任务数 | 26 | 1 | 5 | 2 | 4 | 8 | 54 |

**100 题中有 80 题在六次独立 rollout 下返回恒定结果。** 在固定 harness 下这些任务的组内方差为 0。运行它们的唯一理由是检测某次修改引入的变化，而这是一个关于修改作用机制的问题。

由此可以区分两条常被混用的选择准则。

**pass→fail 的转移只能发生在当前通过的任务上。** 这是定义性质：任何 harness 修改可能引发退化的范围，恰好等于当前通过的任务集合。在上述重复实验中，54 个任务 6/6 通过，其经验通过率 $\bar p_i = 1$，伯努利方差 $\bar p_i(1-\bar p_i)$ 恰为 $0$，按方差加权的采样器给它们的权重约为 $\lambda/\sqrt{n_i}$，接近 0。**以结果方差为准则的选择规则，会系统性地少采样全部可能出现退化的范围。**

在已归档的 AHE 实验中，退化在该范围内累积。对 40×5 矩阵统计相邻轮次的转移：

| 轮次 | fail→pass | pass→fail |
|---|---:|---:|
| iter1 → iter2 | 2 | 3 |
| iter2 → iter3 | 4 | 3 |
| iter3 → iter4 | 3 | 2 |
| iter4 → iter5 | 0 | 6 |
| **合计** | **9** | **14** |

退化次数 14 次，修复次数 9 次；成本审计报告同时记录，五轮结束时没有任何一个在基线下失败的任务变为通过。因此单轮中改进验证一侧的产出率较低，决策由退化防护一侧承担。

与 §1.4 合并后，可以确定三项设计约束：退化一侧主导单轮决策；该侧无法由结果方差采样；§1.4 说明该侧也无法由 agent 自行指名。该侧只能由修改的作用机制导出。

### 1.4 自我预测的精确率与召回率

AHE 用 9 轮的下一轮真值检验 evolve agent 自身的预测（§4.4.2、附录 D）：

| 预测类型 | 精确率 | 召回率 | 随机基线（精确率 / 召回率） |
|---|---:|---:|---|
| 修复 | 33.7% | 51.4% | 6.5% / 10.6% |
| 退化 | 11.8% | 11.1% | 5.6% / 5.4% |

累计而言，agent 提出 43 次退化预测，其中 5 次成立；同期发生 40 次未被预测的退化。本项目的复现结果与之一致：4 份 manifest 共 32 次修复预测，实际发生 6 次。

这一项对设计具有决定作用。若成本有效的流程直接依据 agent 声明的 `predicted_fixes` 与 `risk_tasks` 选取评测集合，则会继承约 11% 的退化召回率，结果是以更低成本累积退化。**任何选择策略都必须用 agent 自身猜测以外的来源导出退化防护集合。**

### 1.5 预算对齐比较对离线成本的要求

《Rethinking the Evaluation of Harness Evolution for Agents》（arXiv:2607.12227）报告：在**反馈预算与推理预算对齐**的条件下，自动 harness 演化并不稳定地优于简单的 test-time scaling 基线，且在留出任务上的泛化有限。一旦领域要求预算对齐的比较，单轮离线成本就从实现细节变为决定比较结果的量。以三分之一的 rollout 达到同等 harness 质量的方法，直接改变该比较的结论。

### 1.6 问题陈述

> 给定基座模型固定的 harness 演化流程、含 $N$ 个任务的训练集合、以及以金额而非轮次计量的总离线预算 $B$，逐轮选择评测哪些任务、每个任务分配多少条 rollout，使流程返回的 harness 在留出集合上的 pass@1 最大化，同时把接受净负面修改的概率控制在给定上界以内。

记 $S_t$ 为第 $t$ 轮被评测的集合，$k_i$ 为任务 $i$ 分配的 rollout 数，$c_i$ 为其单位成本，$H^\star$ 为流程返回的 harness：

$$\max_{\{S_t,\,k_i\}}\;\text{pass@1}\bigl(H^\star;\,D_{\text{test}}\bigr)
\quad\text{s.t.}\quad
\sum_{t=1}^{T}\sum_{i\in S_t}k_i\,c_i\;\le\;B,
\qquad
\Pr\bigl[\,\mathrm{harm}(\Delta_t)\,\bigr]\;\le\;\alpha .$$

其中 $\mathrm{harm}(\Delta_t)$ 表示第 $t$ 轮接受了一项净负面的修改。

---

## 2. 相关工作

### 2.1 采用全量评测的 harness 优化流程

**AHE**（arXiv:2604.25850）以 `评测 → 分析 → 改进` 演化七类文件级 harness 组件，每次修改附带自我声明的预测，由下一轮的任务级差分校验；评测为每轮一次全量。**Meta-Harness**（arXiv:2603.28052）搜索 harness 代码，proposer agent 通过文件系统读取全部历史候选的源码、分数与轨迹，搜索集合固定。**RHO**（arXiv:2606.05922）不使用标签，从未标注的部署轨迹中通过自验证、自一致性与成对自偏好学习。**TTHE**（arXiv:2607.08124）把适配移到测试时的未标注批次上。**HarnessOpt-Bench**（arXiv:2608.06301）对 harness 优化任务本身建立基准。

*关系*。HCE 与上述工作优化的对象与编辑的对象均正交，只改变评测这一步，可以嵌入 AHE、Meta-Harness 或 HarnessBank 而不改动 proposer。以 AHE 为宿主最直接：其 change manifest 已经携带 `predicted_fixes` 与 `risk_tasks`，`compute_task_stability()` 已经给出稳定性分类，契约与档案所需的基础设施大部分已经存在。

### 2.2 候选筛选的级联评测

**AlphaEvolve**（arXiv:2506.13131）用级联评测筛候选：先做低成本的语法与约束检查，只有通过者进入完整评测。**HarnessBank**（arXiv:2607.13683）把同一思路用于 harness 演化，在**随机抽样**的任务子集上设四道门——有效性、激活（该修改是否真的被执行）、显著性（配对 z ≥ 1.96）、增益——通过者才进入完整训练集合评测。**HARBOR**（arXiv:2604.20938）使用多保真度任务子集，当后验方差下降足以支撑时把候选提升到更大的子集。

*关系*。这些是**候选级**分配器：在多个候选之间决定谁值得更多评测。HCE 是**样本级**分配器：对一个候选决定哪些任务值得运行。两者可组合——级联决定是否继续评测，HCE 决定释放出来的 token 用在哪些任务上。HarnessBank 的**激活门**是与 §3.2 激活条件最接近的已有构件，但它被用作事后的有效性检查，而非事前的选择规则。

### 2.3 自适应验证任务选择

**Task-CoEvolve**（arXiv:2608.20169）是直接的前序工作：在 Meta-Harness 形式的流程中，每轮按下式采样验证任务

$$w_i \;=\; \max\bigl(\bar p_i(1-\bar p_i),\; \ell_i\bigr) \;+\; \frac{\lambda}{\sqrt{n_i}},$$

再用 Hájek 与锚定差分估计量还原可比的全集分数，纳入概率由 Monte Carlo 估计。在 20% 评测预算下，Terminal-Bench 2.1 上达到 51.7%（全量搜索 52.8%），输入 token 减少 67–80%，墙钟减半；同预算的均匀随机子集只达到 48.4%。

*关系——三点具体差异*。

1. **条件对象**。Task-CoEvolve 的权重只依赖任务的历史结果，在给定轮次对所有候选修改取值相同。HCE 以**具体修改的作用机制**为条件：同一轮的两个候选修改得到不同的评测集合。对 shell 输出截断的修改与对 finish hook 的修改，影响范围不相交，仅依赖历史的加权无法表达这一点。
2. **目标不对称**。伯努利方差加权最大化的是任务对**排序**的信息量，它给 $\bar p_i \approx 1$ 的任务赋予接近 0 的权重，而按定义该集合是全部可能出现退化的范围（§1.3），§1.4 说明流程在该范围的召回率已经很低。HCE 把**功效目标**（检出既定增益）与**单侧风险目标**（约束损害）分开，各用一条采样规则。
3. **自适应性**。Task-CoEvolve 在观察任何结果之前固定 $m=\lceil\rho N\rceil$，作者把「在评测过程中决定该数量」列为自然的扩展方向。HCE 按波次分配并采用随时有效的停止规则（§3.4），即为该扩展。

**RoboPhD**（arXiv:2604.04347）采用相反的设定：取消验证集合，每轮在 20 个随机抽取的训练样本上评测三个 agent，用 Elo 在约 21 轮中累积弱信号，总预算固定为 1500 次评测。该结果表明多轮次的高噪声评测可以优于少轮次的低噪声评测。它既是强基线，也是设计上的约束条件：若随机浅层多轮胜出，定向选择必须能覆盖自身的额外开销。

### 2.4 评测样本效率

**tinyBenchmarks**（arXiv:2402.14992）用项目反应理论从每个场景约 100 个精选样本估计模型能力，基于 IRT 的锚点选择优于分层抽样与按正确性聚类。**Active Testing**（arXiv:2103.05331）用获取函数选择需要标注的测试点，并用 LURE 估计量消除由此引入的偏差同时降低方差。**Active Evaluation Acquisition**（arXiv:2410.05952）与**基于近似 Neyman 分配的 LLM 主动测试**（arXiv:2605.10075）把该框架扩展到 LLM 评测。

*关系*。该方向估计的是**单一系统的标量能力**，选择准则是样本的区分度，同样是样本自身的性质而非某项修改的性质。HCE 沿用其估计量机制（纳入概率校正、控制变量锚定），不沿用其选择准则。IRT 区分度可以作为任务档案先验的成分（§3.1），并作为一条消融分支。

### 2.5 预算分配、racing 与最优臂识别

**F-Race / irace**（López-Ibáñez 等，*Operations Research Perspectives*，2016）在多个实例上评测配置，一旦 Friedman 秩检验给出证据即淘汰劣势配置。**Successive Halving / Hyperband**（arXiv:1603.06560）在固定预算下按递增保真度分配。**ProTeGi / APO**（arXiv:2305.03495）在小批量上用 UCB、Successive Rejects 与 Successive Halving 选择 prompt 候选。**GEPA**（arXiv:2507.19457）在小批量上评测变异，维护**逐样本**的 Pareto 前沿，只有存活者进入 anchor 集合。**Cost-Aware Multi-Objective Bandits**（arXiv:2608.04333）以「单位成本超体积」指标在多个 LLM 配置之间分配预算，给出遗憾界与 Pareto 识别界，但明确把样本集合固定为每个配置 1000 个样本。

*关系*。上述工作都把样本视为同分布的可互换抽样，并在**臂之间**分配。HCE 把样本视为不可互换的个体，其信息量依赖于具体修改，并在**单个臂内部**分配。racing 方向提供了 §3.4 所用的停止规则形式；GEPA 的逐样本 Pareto 前沿是已有工作中最接近「样本不可互换」这一认识的构造。

### 2.6 回归测试选择

代码变更后选择重跑哪些测试，是软件工程中已有四十年历史的问题。**安全的回归测试选择**（Rothermel & Harrold，*ACM TOSEM*，1997）通过在程序依赖图上做静态或动态的变更影响分析，保证不遗漏任何行为可能改变的测试。**Ekstazi**（Gligoric 等，*ISSTA*，2015）在文件粒度上使其可用于工程实践。**Yoo & Harman**（*STVR*，2012）综述了最小化、选择与优先级排序三类技术。

*关系*。§3.2 的激活条件对应变更影响分析；改进验证与退化防护的划分对应「安全性保证」与「为尽早发现缺陷而排序」的区分。本质差异在于 harness 修改**没有可静态计算的依赖图**：行为发生改变的「程序」是一个随机策略，受影响的「测试」只能从已观测的执行中识别。因此 HCE 用从轨迹中提取的**机制画像**（§3.1c）近似依赖关系，这是一个经验性、概率性的替代物。相应地，安全性保证从「不遗漏任何受影响的测试」减弱为「遗漏受影响任务的概率由审计比例给出上界」，形式化见 §3.5。

### 2.7 重新执行的低成本替代方案

**Causal Agent Replay**（arXiv:2606.08275）把一次 agent 运行建模为结构因果模型，在某一步施加 `do` 干预后按同一策略向前重新执行，以归因失败。**The Replay Gap**（arXiv:2608.08239）测量另一种做法——把记录下来的输出替换进已有轨迹：在早期分叉点更换模型后，74–77% 的分支在紧随其后的第一个动作即发散，可用于评测的重放状态只剩 3–8%；预测补丁与实际补丁的相似度为 0.00–0.11；全部与成功相关的结果转移对重放不可见。

*关系*。这一结果排除了最直接的降低成本路径。在闭环中，修改会改变后续观测，因此无法通过在新 harness 下重放旧轨迹来预测某次修改是否会改变某个任务的结果。**rollout 必须真实执行。** 本提案的贡献因此落在「运行哪些任务」这一问题上。

---

## 3. 技术流程

### 3.0 形式化

基座模型 $M$ 固定。训练集合 $D=\{d_1,\dots,d_N\}$，任务 $d_i$ 的单次 rollout 成本为 $c_i$（美元），在 harness $H$ 下的结果为 $x_i(H)\in\{0,1\}$，或 $k$ 条 rollout 上的通过率。留出集合 $D_{\text{test}}$ 在流程内部不被访问。第 $t$ 轮以 $H_{t-1}$ 为输入，产生修改集合 $\Delta_t$ 与候选 $H_t=\operatorname{apply}(\Delta_t,H_{t-1})$。

| 符号 | 含义 |
|---|---|
| $D_{\mathrm{pass}},\ D_{\mathrm{fail}},\ D_{\mathrm{unst}}$ | 稳定性分类，见 §3.1 档案（a） |
| $A_t$ | $\Delta_t$ 的激活集合，见 §3.2 |
| $T_t,\ R_t$ | 契约声明的预期修复任务 / 风险任务 |
| $C_t,\ G_t,\ \mathcal{A}_t$ | 改进验证集合、退化防护集合、审计样本，见 §3.3 |
| $S_t=C_t\cup G_t\cup\mathcal{A}_t$ | 本轮的评测集合 |
| $\pi_i=\Pr[\,i\in S_t\,]$ | 任务 $i$ 的纳入概率 |
| $\hat p_i^{\,\mathrm{hist}}$ | 任务 $i$ 的历史锚点，见 §3.5 |
| $\rho,\ \tau,\ B$ | 单轮预算比例、审计周期、总预算 |

每轮必须输出：（a）$\Delta_t$ 中每项修改的保留或回退决策；（b）全集表现的估计 $\hat S(H_t)$，其量纲跨轮可比，使「历史最优」的选择保持有效；（c）供第 $t+1$ 轮使用的轨迹证据。

全量单轮的基线成本为

$$C_{\text{full}}\;=\;k\sum_{i=1}^{N}c_i ,$$

目标是找到子集 $S_t\subseteq D$ 与逐任务的 rollout 数 $k_i$，使

$$\sum_{i\in S_t}k_i\,c_i\;\approx\;\rho\,C_{\text{full}},\qquad \rho\in[0.15,\,0.4],$$

同时保持（a）与（b）。

### 3.1 任务档案（一次构建，成本摊销，锚定轮次刷新）

由流程已经支付过的产物组装的逐任务记录。边际成本为每个任务一次 LLM 判分调用，比一条 rollout 低两个数量级。

**（a）历史结果。** 在已评测过的每个 harness 下的通过向量；由此导出稳定性分类（恒通过 / 恒失败 / 有波动 / 仅基础设施异常）与伯努利方差。AHE 的 `compute_task_stability()` 已经计算该分类，只需按任务持久化而非聚合。

**（b）结构指纹。** 与具体代码库无关的失败模式、难度来源与技术范围描述，由 LLM 判分器从任务描述加一条已观测轨迹生成，随后做向量化。RHO 的 `difficulty_selector.py` 判分 prompt 是现成实现：返回 `{difficulty ∈ [0,10], abstract_fingerprint}`，并明确要求剥离仓库名、文件路径与库名；`trajectory_digest.py` 把摘要限制在约 10k token 并对任务内容做屏蔽。指纹支持 HCE 需要的两项操作：**检索受影响任务的最近邻**（把退化防护集合扩展到 agent 声明的风险任务之外）与**覆盖度计算**（使审计样本按分层抽取而非均匀抽取）。

**（c）机制画像——新增成分。** 对任务轨迹实际触发了哪些 harness 机制的结构化摘要，尽可能用确定性解析而非 LLM 提取：轮数与 token 轨迹；工具调用直方图；是否触发上下文压缩；是否有 shell 命令超过输出上限或时间上限；是否有 middleware hook 触发以及是哪一个；是否加载了 skill；终止状态（已验证成功 / 提前终止 / 超时 / 基础设施异常）；错误类型直方图；编辑的文件数；是否运行了测试命令以及是否使用了过滤参数。

激活条件与该字段匹配，这使选择以修改为条件而非以历史为条件。该字段也与本项目已有的失败模式分类（`harness-failure-patterns-task-mapping.md`）对应，后者已经按 `process-friction / integrity / direction-correctness / capability-ceiling` 组织并附逐任务证据。

**（d）成本画像。** 单次 rollout 的平均 token、平均墙钟、平均金额及其离散程度。该字段是必需的，因为选择需要最大化**单位金额的信息量**而非单位任务的信息量：在 Terminal-Bench 2 与 LHTB 上，单题成本跨越一个数量级以上，而 Task-CoEvolve 报告区分度高的任务不成比例地集中在长时多轮任务上，即信息量高的任务同时成本高。该关系按成本计价处理。

### 3.2 评测契约

AHE 已要求每项修改声明失败证据、根因、针对性修复与预期影响，形式为自由文本加 `change_manifest.json` 中的两个任务 ID 列表。HCE 把它收紧为字段**可对照轨迹校验**的结构：

```jsonc
{
  "change_id": "c3",
  "component": "tools/shell_tools/run_shell_command.py",
  "scope": "local",                       // "local" | "global"，见 §3.3
  "mechanism": "当 stdout 超过 8k token 时截断，并追加取自命令 cwd 邻近文件的契约提示",
  "activation_predicate": {               // 与档案字段 (c) 匹配
    "any_command_output_tokens": ">8000"
  },
  "predicted_fixes":  ["task_a", "task_b"],
  "risk_tasks":       ["task_c"],
  "expected_direction": "提高长输出任务上的已验证完成率"
}
```

两项性质是关键。

**激活既是选择规则，也是证伪通道。** 把条件与每个任务的机制画像匹配，得到**激活集合** $A_t\subseteq D$，即该修改可能触发的任务集合。在分布不变与采样噪声的范围内，$D\setminus A_t$ 中的任务不受该修改影响。本轮结束后，用同一条件对照**新的**轨迹检查：某项修改被预测会在某任务上触发而实际未触发，则无需任何结果即被证伪，额外成本为零。这把 HarnessBank 的激活门从事后有效性检查推广为主要的分配依据。

**契约由 agent 撰写，而契约决定 agent 如何被检验**，因此契约本身不可被单独信任。§4.4 给出对应措施：激活状态一律**校验**而不采信声明；退化防护集合由指纹邻域扩展到声明的风险任务之外；审计样本始终在 agent 的控制之外抽取。

### 3.3 单轮划分：改进验证、退化防护、延迟

**改进验证集合 $C_t$**——预期效应是否出现。

$$C_t\;\subseteq\;T_t\;\cup\;\bigl(A_t\cap(D_{\mathrm{fail}}\cup D_{\mathrm{unst}})\bigr),$$

按单位金额的期望状态转移概率排序。其作用是归因与生成证据，不是接受判定。§1.3 记录整个实验期间 fail→pass 9 次、pass→fail 14 次，因此改进验证是单轮中产出率较低的一侧：$C_t$ 的规模应当**较小**，以能够完成归因并为下一轮生成失败轨迹为准，不以证明增益为准。

**退化防护集合 $G_t$**——总体损害是否受控。三个定向来源加一个审计样本：

$$G_t\;=\;\underbrace{\bigl(A_t\cap D_{\mathrm{pass}}\bigr)}_{(1)}\;\cup\;\underbrace{R_t}_{(2)}\;\cup\;\underbrace{\mathcal{N}_\kappa(A_t)\cap D_{\mathrm{pass}}}_{(3)},
\qquad
\underbrace{\mathcal{A}_t\;\sim\;\operatorname{Strat}\bigl(D\setminus(C_t\cup G_t\cup A_t)\bigr)}_{(4)}$$

1. $A_t\cap D_{\mathrm{pass}}$——该修改可以触发、且当前恒定通过的任务。按定义，该集合包含全部可能发生退化的任务（§1.3）。
2. 声明的 $R_t$——纳入成本低，精确率约 12%，因此不作为主要来源。
3. $\mathcal{N}_\kappa(A_t)$：在指纹空间中与 $A_t$ 最近的 $\kappa$ 个任务，取其落在 $D_{\mathrm{pass}}$ 中的部分，用于覆盖激活条件未充分刻画的影响范围。这是在不依赖 agent 预测能力的前提下修正 §1.4 中 11% 退化召回率的构件。
4. $\mathcal{A}_t$：按指纹簇与成本分位分层抽取。该项不可省略：它保证每个任务的纳入概率 $\pi_i>0$，这是 §3.5 估计量无偏的前提；它也是唯一能够发现激活条件完全遗漏的机制的通道。

**scope 的处理。** 部分修改没有狭窄的影响范围。更长的 system prompt 改变每一次调用的上下文，激活条件在全集上取真，即 $A_t=D$。契约的 `scope` 字段声明这一情况，分配器相应退化为更大审计比例的分层随机抽样。把全局修改当作局部修改处理，比不做选择更差，因此 `scope: global` 由静态规则强制：对 `systemprompt.md`、`code_agent.yaml` 或任何常驻 middleware 的修改按构造判定为全局，与 agent 的声明无关。

**预算划分。** §1.3 与 §1.4 合并后表明**退化防护一侧应当获得单轮预算的多数**：该侧承担更多已观测的状态转移，方差加权无法覆盖它，agent 自身的预测也无法覆盖它。建议默认值为

$$b_C : b_G : b_{\mathcal{A}}\;=\;25 : 55 : 20 .$$

该比例是待检验的假设：Phase 0（§5.1）在已归档的实验矩阵上直接测量它，§5.4 将其列为消融分支。

### 3.4 分波分配与随时有效的停止规则

Task-CoEvolve 在观察结果前固定 m。HCE 把单轮拆为若干波次：

- **第 0 波——激活检查。** 成本最低的证据。若条件可由 harness 代码加任务档案直接判定（例如「仅当注册了某 middleware hook 时触发」），则成本为零；否则为若干条落在激活度最高任务上的 rollout。未通过第 0 波的修改在不消耗任何防护 rollout 的情况下回退。
- **第 $1$ 至 $W$ 波。** 第 $w$ 波在已评测集合 $S^{(w-1)}$ 的基础上，按**单位金额的期望决策信息量**贪心扩展，分母取自档案（d）：

$$i^\star\;=\;\operatorname*{arg\,max}_{i\,\in\,(C_t\cup G_t)\setminus S^{(w-1)}}\;\frac{\mathbb{E}\bigl[\,\mathcal{I}(i)\,\bigr]}{k_i\,c_i},$$

  其中 $\mathcal{I}(i)$ 为任务 $i$ 贡献的决策不确定性下降量。波内的 rollout 并发执行。
- **每波结束后**更新两条随时有效置信序列：$C_t$ 上定向增益的 $[L_w^{g},U_w^{g}]$，与 $G_t$ 上单侧损害的 $[\,\cdot\,,U_w^{h}]$。当进入接受域或回退域、或本轮预算耗尽时停止。

参数 $\theta$ 的置信序列是一列区间，满足

$$\Pr\bigl[\;\exists\,w\ge 1:\ \theta\notin[L_w,U_w]\;\bigr]\;\le\;\alpha ,$$

即覆盖率对所有波次**同时**成立，因此可以在每一波之后查看区间并据此停止。此处需要随时有效置信序列（Howard 等；工程实践见 arXiv:2302.10108）而非固定样本量检验，因为停止决策依赖数据：在每波之后用固定样本量检验做判断会抬高第一类错误率，而当前正处于效应量很小的区间。代价是区间更宽；§4.5 明确说明在现实预算下该机制的作用为**停止**规则（结论已明确时提前结束），认证作用不在其范围内。

**配对。** 单轮内被比较的各个方案必须在**同一**子集上评测，这把比较从非配对的总体差值转为配对的逐任务差值，无需额外成本即可消去任务间方差。本项目已有的分析对 500 题配对比较使用 McNemar 精确检验，同一估计量在 100 题子集上同样适用。子集在**轮次之间**轮换以获得覆盖度，在**轮次内部**不轮换以保持配对。

### 3.5 跨轮分数估计与锚定

选择破坏了流程做「历史最优」判断所需的跨轮可比性，因此单轮分数由锚定并按纳入概率校正的估计量还原：

$$\hat S(H_t)\;=\;\frac{1}{N}\sum_{i=1}^{N}\left[\;\hat p_i^{\,\mathrm{hist}}\;+\;\frac{\mathbf{1}\{i\in S_t\}}{\pi_i}\Bigl(x_i(H_t)-\hat p_i^{\,\mathrm{hist}}\Bigr)\right]$$

其中 $\hat p_i^{\,\mathrm{hist}}$ 为档案给出的任务锚点，$\pi_i$ 为该任务在本轮划分下的纳入概率（划分中的确定性部分有闭式解，随机部分按 Task-CoEvolve 的做法用 Monte Carlo 估计）。方括号内为控制变量形式：锚点承担水平项，只有残差 $x_i(H_t)-\hat p_i^{\,\mathrm{hist}}$ 需要由采样支付，因此只要对所有 $i$ 有 $\pi_i>0$，$\hat S$ 即为无偏。HCE 增加两项细化：

- **以激活状态为条件的锚点。** 对 $i\notin A_t$，该修改无法触发，$\mathbb{E}[x_i(H_t)]=\mathbb{E}[x_i(H_{t-1})]$，锚点无偏，残差项接近 0。对 $i\in A_t\setminus S_t$（可触发但被跳过）的任务，锚点存在方向未知的偏差，必须给出更宽的区间。分开报告这两类任务，使估计值明确区分哪一部分是测得的、哪一部分是插补的。
- **轮换审计取代周期性全量评测。** 每轮的审计样本在延迟集合中轮换，使每个任务至少每 $\tau$ 轮被重新测量一次。这在不产生成本尖峰的前提下约束锚点漂移，并把锚定成本均摊。实验**结束时**执行一次真实的全量评测，最终 harness 报告的数值取自该次评测而非任何估计。

**锚点漂移是需要监控的失效形式。** 若某任务长期未被重新测量，其锚点对应的 harness 已经不存在。在每次轮换审计时，

$$\mathrm{drift}_t\;=\;\bigl|\,\hat S(H_t)-S(H_t)\,\bigr|$$

作为一级诊断量，按轮记录，见 §5.3。

### 3.6 决策规则

某项修改在预测的任务上发生了激活、退化一侧的损害估计在本轮置信序列下低于容许阈值、且估计的全集分数未下降时予以保留：

$$\operatorname{keep}(\Delta_t)\iff\underbrace{\operatorname{act}(\Delta_t)=\textsf{verified}}_{\text{(i)}}\ \wedge\ \underbrace{U_W^{h}\le h_{\max}}_{\text{(ii)}}\ \wedge\ \underbrace{\hat S(H_t)\ \ge\ \hat S(H_{t-1})}_{\text{(iii)}}$$

当 $U_W^{h}>h_{\max}$、或第 0 波显示该修改从未激活时予以回退。

条件（iii）刻意采用估计的全集分数而非改进验证集合上的分数。已归档的 AHE 运行提供了对应案例：第 3 轮的修改被 `evaluate_changes()` 判为 HARMFUL，因为其**声明的预测**没有成立；而携带该修改的 harness 取得了整次运行的最好结果（22/40），依据该 HARMFUL 判定执行回退后，下一次评测降至 16/40。以预测而非以结果评分的判定会丢弃有效的修改。在 HCE 中，契约决定**在哪里观测**，成功的定义由估计的全集分数给出。

### 3.7 成本模型与记账

全部报告数值为按角色分项的金额，由已有的逐调用 `UsageTracer` 加 `price_run.py` 路径产生。流程自身机制的成本计入本方法：

$$C_{\text{method}}\;=\;\sum_{t=1}^{T}\sum_{i\in S_t}k_i\,c_i\;+\;\sum_{t=1}^{T}C_{\mathrm{adb}}(S_t)\;+\;\bigl(C_{\mathrm{dossier}}+C_{\mathrm{contract}}+C_{\mathrm{est}}\bigr)$$

三项依次为 rollout 成本、轨迹分析成本、选择机制成本。

若某方法减少 70% 的 rollout 而把原预算的 20% 用于选择，则实际节省为 50%。节省量以相对全量评测流程的比值 $C_{\text{method}}/C_{\text{full}}$ 报告，不以 rollout 条数报告。

### 3.8 算法

```
输入：种子 harness H0，训练集合 D，留出集合 D_test，预算 B，单轮比例 ρ，审计周期 τ
初始化：在 D 上以 k 条 rollout 全量评测 H0 → 档案(a,b,c,d)，锚点 p̂^hist
for t = 1 .. T，预算未耗尽时：
    (Δ_t, contracts) ← Evolve(H_{t-1}, 证据语料, 历史判定)
    H_t             ← apply(Δ_t, H_{t-1})
    A_t             ← match(contract.activation_predicate, 档案.机制画像)
    若任一 contract.scope == "global"：把 A_t 扩展到 D，提高审计比例
    C_t, G_t, audit ← partition(A_t, 稳定性分类, 指纹, 成本画像, ρ·B_t)
    for 波次 w = 0, 1, 2, ...：
        运行 batch(w) ⊆ C_t ∪ G_t ∪ audit；更新置信序列
        若进入接受域或回退域、或本轮预算耗尽：break
    在新轨迹上校验激活状态；verdict ← 决策规则(...)               # §3.6
    H_t ← rollback(H_t, verdict)；Ŝ(H_t) ← 锚定估计量              # §3.5
    若 t mod τ == 0：由轮换审计刷新锚点；记录 anchor_drift
    AgentDebugger 仅处理*已评测*的轨迹 → 证据语料
结束：对最优 harness 在 D 与 D_test 上各做一次全量评测           # 报告数值
```

### 3.9 在现有代码上的实现路径

除选择器外，所需构件在当前分支中均已存在：

| 所需 | 已有接口 |
|---|---|
| 稳定性分类 | `evolve.py: compute_task_stability()`（第 783 行） |
| 单轮状态转移与退化集合 | `evolve.py: compute_iteration_diff()` |
| 契约存储与归因 | `change_manifest.json`、`evolve.py: evaluate_changes()`（第 2239 行） |
| 按角色的逐调用成本 | `usage_tracer.py`、`scripts/price_run.py` |
| 指纹判分与轨迹摘要 | `retro-harness/src/rho/selection/{difficulty_selector,trajectory_digest,embedder}.py` |
| 子集派发 | 评测阶段 harbor 的任务列表参数 |

具体改动为：新增产生 $S_t$ 与 $\pi$ 的 `selection/` 模块；升级 `change_manifest.json` 的 schema；在 `nexau_in_memory_tracer.cleaned.json` 上新增机制画像提取器；在 harbor 派发外层包一个波次循环。evolve agent、agent debugger 与 harness 组件本身不变。

---

## 4. 挑战

### 4.1 退化预测召回率与不可覆盖的退化类型

实测退化召回率为 11.1%（§1.4）。§3.3 中的措施——激活集合、指纹邻域、分层审计——针对的是具有**机制痕迹**的退化。有一类不具备：更长的 system prompt 或体量更大的常驻 middleware 通过上下文长度与注意力分散影响任务，作用是均匀的且没有可区分的运行时特征。已归档运行中可以观察到该机制的条件（prompt 548 B → 2465 B，单轮成本 +44%）。对这一类，激活集合等于全集，定向选择不可行，只有 `scope: global` 规则与随机审计可以覆盖。**这是本方法的硬性边界，将按此表述，不做弱化。** 各类退化各占多少比例，由 Phase 0 的第二项交付物给出。

### 4.2 效应量与噪声下限

已归档的 40 题运行噪声下限约 10 pp，k=1 时 22% 的任务不稳定；100 题重复实验在 k=2 下测得 20% 的任务有波动（§1.3）。轮间 harness 效应为 2 至 7 pp。**在研究预算下，任何选择策略都无法使单轮达到统计可认证。** 提案明确接受两点后果：

- 主张的对象为**实验结束时、单位金额的留出集合 pass@1**；单轮显著性不在主张范围内。
- 分配是二维的：任务数量 × 每个任务的 rollout 数量。在同等金额下，任务更少而 k 更高的配置可能占优，因为配对的逐任务比较消去任务间方差，重复采样消去任务内方差。固定金额下的 (N, k) 前沿是一项实验（§5.4），不是假设。

### 4.3 评测量与证据量的耦合

evolve agent 的下一次提案由评测产生的轨迹形成。运行 25% 的训练集合得到 25% 的新增失败证据，这可能在提高单位金额改进量的同时降低单轮改进量。缓解措施：证据语料按累积方式维护，延迟任务的历史轨迹仍可读取；改进验证集合刻意偏向仍在失败的任务，本身即为新证据的来源；`agent_debugger` 成本随被评测任务数下降，释放的预算可用于更多轮次。净效应的方向由 Phase 1 回答，并以「每轮改进」与「每美元改进」两条曲线分别报告，使负面结果可见而不被平均掉。

### 4.4 契约撰写方与被检验方相同

存在收窄契约以规避检验的动机：范围狭窄的激活条件使修改易于验证且难以证伪。对应措施按强度递减排列：（i）激活状态始终对照本轮结束后的轨迹**校验**，不采信声明，激活范围被低估本身即构成一次证伪事件；（ii）审计样本在 agent 控制之外抽取，其成员不向 agent 展示；（iii）`scope: global` 对常驻组件由静态规则指派，与 agent 的声明无关；（iv）退化防护集合由指纹邻域扩展，该部分不由 agent 选择。§5.4 包含一条专门的对抗性消融：以「尽量缩小自身评测集合」为指令运行契约撰写方。

### 4.5 选择导致的过拟合

若流程优先评测自身预测指向的任务，则它优化的是自身的预测。§3.6 的决策规则是结构性防护：契约决定**在哪里观测**，估计的全集分数决定**什么算成功**，留出集合在流程内部不被使用。另有独立证据表明同集合演化会高估增益（arXiv:2607.12227）；本项目已有的 AHE 运行在被评分的同一批任务上演化，并已在报告中标注该限制。§5 中的每个方案都使用严格的训练与测试划分。

### 4.6 组件间相互作用与逐项归因

AHE 的组件消融显示三个单组件的正向增益合计 +11.1 pp，而完整 AHE 为 +7.3 pp；在 Hard 档，仅换入 memory 的变体高于完整 AHE。组件都指向重复的验证行为，在长时任务预算内相互干扰。因此在不相交子集上验证的逐项契约不可叠加。缓解措施：默认每轮一项修改（已归档运行在实践中即为每轮一项）；当同轮包含多项修改时，须检查各自激活集合的重叠，重叠的修改联合评测而不分别归因。周期性的**相互作用审计轮次**在全量评测下比较累积 harness 与种子 harness。

### 4.7 成本异质性

Task-CoEvolve 报告方差加权会把预算集中到长时多轮任务，即成本高的任务。若信息量高的任务系统性地也是成本高的任务，按任务计数的节省会高于按金额计的节省。档案（d）与按单位金额排序直接针对这一项，报告一律以金额与墙钟计量，不以任务数量计量。

### 4.8 冷启动与基础设施噪声

第 1 轮没有锚点也没有稳定性分类，因此必须执行一次自举全量评测，其成本计入本方法预算。另有一项：基础设施异常不可被静默吸收。已归档运行中异常次数为 0，但更早的一次运行因 middleware 引用了不存在的模块导致配置加载失败，整轮 40 题全部计 0 分。把这样一轮判定为该修改造成严重损害的选择策略会回退一项有效的修改，因此有效性门（HarnessBank 的第一道门）是前置条件而非可选项。

---

## 5. 实验计划

### 5.1 Phase 0——在已归档结果矩阵上做离线策略回放（成本约 \$0，可立即开始）

已有的归档产物使得在不消耗任何 rollout 的前提下评估**选择策略**成为可能：

| 产物 | 规模 | 用途 |
|---|---|---|
| `benchmark40-ahe-run2-per-task.json` | 40 题 × 5 轮，k=1 | 回放逐轮决策 |
| `benchmark40-ahe-run2-manifest-{1..5}.json` | 声明的修复与风险 | 契约保真度 |
| `benchmark40-ahe-run2-evaluation-{2..5}.json` | 实际判定 | 决策真值 |
| `benchmark100-baseline-per-task.json` | 100 题 × 6 条 rollout，单一 harness | 逐任务噪声下限、稳定性分类 |
| `swebench500-terra-{high,medium}-compare.json` | 500 题 × 3 个方案 × 2 档 effort，含逐任务 `cost_usd` | 逐任务成本分布；零效应条件下各选择策略的误检率 |

交付物：

- **D0.1——决策保真度曲线。** 对 {均匀随机、伯努利方差（Task-CoEvolve）、按稳定性分层、契约驱动（理想激活）、契约驱动（agent 声明）} 各策略与 $\rho\in\{0.1,\dots,1.0\}$：本轮保留或回退决策与全量评测决策一致的概率，以及锚定估计量的符号错误率。
- **D0.2——退化类型分布。** 把归档中每一次已观测的退化归入「有机制痕迹」或「全局性下降」，给出定向选择在原理上可覆盖的退化比例（§4.1）。
- **D0.3——防护侧与验证侧的产出比。** 由 40×5 矩阵给出每轮与决策相关的状态转移中，源自当前通过类与当前失败类的比例，使 §3.3 的预算划分建立在测量之上。
- **D0.4——零效应条件下各策略的误检率。** 500 题 terra 各方案之间的差值小于其运行间标准差（0.76–1.04 pp），因此可作为零效应面板：在全量评测未给出差异时，各子集策略报告出差异的频率是多少。
- **D0.5——逐任务成本分布。** 500 题 compare 文件按任务记录 `cost_usd`，可直接给出档案的成本画像（§3.1d），并在任何实机运行之前量化 §4.7 中信息量与成本的相关关系。

*限制说明。* 该回放假设不同的第 t 轮决策不改变后续实验轨迹，而这一假设不成立：第 3 轮的不同决策会改变第 4 至 10 轮。因此它只给出**单轮**决策保真度的上界。这与 The Replay Gap（arXiv:2608.08239）的结论一致，也是 Phase 0 只用于**设计**策略、不用于给出端到端结论的原因。

### 5.2 Phase 1——小规模实机实验（SWE-bench 40 题，5 轮）

复现已归档运行的设置，使结果与已知的成本基线可比。

| 方案 | 每轮 | 单次复制的预期成本 |
|---|---|---|
| A. 全量 AHE（参照） | 40 题 | 约 \$56（实测） |
| B. 同 $\rho$ 的均匀随机 | $\rho\cdot 40$ 题 | 约 \$25 |
| C. 伯努利方差（Task-CoEvolve 复现） | $\rho\cdot 40$ | 约 \$25 |
| D. **HCE** | 契约驱动，$\rho\cdot 40$ | 约 \$25 加选择开销 |

每个方案 3 个种子，$\rho=0.3$，$k=2$（已归档运行的 $k=1$ 是已知的不利条件：$k=1$ 时 evolve 查询称为最高杠杆目标的「部分通过」信号完全不存在）。严格划分：在 40 题上演化，在不相交的 60 题留出集合上评分。含留出评测合计约 \$1200。

主要读数：留出集合 pass@1 对**已花费金额**的曲线。

### 5.3 Phase 2——规模化实机实验

| 设置 | 训练/留出划分 | 轮次 | 方案 | 说明 |
|---|---|---|---|---|
| SWE-bench Verified | 300 训练 / 200 留出 | 8 | A–D | 主结果 |
| Terminal-Bench 2 | 89 题，沿用 AHE 的划分方式 | 10 | A、D | 与 AHE 已发表实验可比 |

每个方案报告的指标：

1. **单位金额的留出集合 pass@1**（主指标），以及总金额对齐时的取值。
2. **决策保真度**：在若干同时执行全量评测的轮次上，与全量评测判定的一致率。仅在少数轮次支付该成本。
3. **退化漏检率**：被接受的修改中，经后续全量评测显示为净负面的比例。
4. **有效 rollout 比例**：本轮 rollout 中落在发生状态转移的任务上的比例。全量 AHE 的该值为 14.4%（§1.2），该指标即本方法要提高的量。
5. **锚点漂移**：每次轮换审计时的取值（§3.5）。
6. **单轮墙钟**：顺序波次降低并发度，可能在节省金额的同时延长时间，该取舍需要报告。
7. **预算对齐的 test-time scaling 基线**：在同等金额下对种子 harness 做 best-of-n，按 arXiv:2607.12227 的要求执行。

### 5.4 消融

| 消融项 | 问题 |
|---|---|
| 去除激活条件（仅按历史加权） | 以修改为条件是否值得其开销 |
| 去除退化防护集合（仅方差加权） | 方差加权在规模化条件下是否如 §1.3 预测那样漏掉退化 |
| 去除指纹邻域（仅用声明的风险任务） | 11% 的退化召回率能否由结构信息修正 |
| 去除审计样本 | 锚点漂移的累积速度 |
| 去除波次停止（固定 m） | Task-CoEvolve 留出的自适应性的价值 |
| 预算划分 ∈ {50/30/20, 25/55/20, 10/70/20} | 确认或否定 §3.3 的防护侧偏重默认值 |
| $\rho\in\{0.1,0.2,0.3,0.5\}$ | 决策保真度在何处失效 |
| 对抗性契约撰写方 | 契约被规避的程度（§4.4） |
| 仅含全局 scope 的修改 | 退化到分层随机抽样时的表现（§4.1） |

### 5.5 否定条件

预先声明：

- **同等金额下均匀随机与 HCE 持平。** RoboPhD 的结果（固定评测预算下，随机浅层多轮优于少量精细轮次）使这一情况具有现实可能。若方案 B 与方案 D 在两个规模上的单位金额留出 pass@1 均无差异，则贡献退化为「子集评测可行」，而该结论 Task-CoEvolve 已经给出。
- **退化在方差加权下与在机制条件下同样可检出。** 若在规模化条件下发生退化的任务主要是有波动的任务而非恒定通过的任务，则防护侧偏重的预算划分失去依据，方差加权即为充分。
- **证据不足占主导。** 若单轮改进量下降的速度快于单轮成本下降的速度（§4.3），则同等金额下流程得到更差的 harness，对应的结论是全量评测购买的是证据而不只是精度。

### 5.6 顺序

| 阶段 | 依赖 | 产出 |
|---|---|---|
| 0——离线回放 | 仅需已归档矩阵 | D0.1–D0.5；Phase 1 采用的策略 |
| 0.5——任务档案工具 | RHO 判分器移植、机制画像提取器 | SWE-bench 40 题与 100 题上的档案 |
| 1——小规模实机实验 | Phase 0.5 | 单位金额留出 pass@1，4 方案 × 3 种子 |
| 2——规模化实验 | Phase 1 | 主结果与消融 |
| 3——迁移 | Phase 2 | 冻结 harness 向第二个基准与第二个基座模型的迁移 |

Phase 0 不需要 API 预算，也不需要沙箱容量，且是正确的第一步：它可以在消耗任何 rollout 之前否定或修正核心设计决策（§3.3 的划分）。

---

## 6. 待定的设计选择

1. **命名。** HCE / CEHE 为占位名称。
2. **宿主流程。** AHE 分支（本文建议，理由是 manifest 与稳定性机制已经存在）或 Meta-Harness（Task-CoEvolve 所在的设置，选它可使正面比较严格对齐）。
3. **机制画像的提取方式。** 确定性轨迹解析（成本低、脆弱、每个 harness 框架需要一个提取器）或 LLM 提取（通用、每题约 \$0.01、可复现性较低）。建议：可解析的字段用确定性解析，剩余部分用 LLM。
4. **Phase 2 的主基准取 Terminal-Bench 2 还是 SWE-bench Verified。** AHE 与 Task-CoEvolve 都在 Terminal-Bench 2 上发表，选它可使比较直接；本项目已有 SWE-bench Verified 500 题、两档 effort 的逐任务矩阵，选它可使 Phase 0 更充分。

---

## 参考文献

harness 优化
- Agentic Harness Engineering — arXiv:2604.25850
- Meta-Harness: End-to-End Optimization of Model Harnesses — arXiv:2603.28052
- Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference — arXiv:2606.05922
- HarnessBank: Semantic Gene-Bank Search with Gated Verification — arXiv:2607.13683
- HARBOR: Automated Harness Optimization — arXiv:2604.20938
- TTHE: Test-Time Harness Evolution — arXiv:2607.08124
- HarnessOpt-Bench: Evaluating LLMs at Harness Optimization — arXiv:2608.06301
- Rethinking the Evaluation of Harness Evolution for Agents — arXiv:2607.12227
- Training-Free Group Relative Policy Optimization — arXiv:2510.08191

评测预算分配
- Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection — arXiv:2608.20169
- RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets — arXiv:2604.04347
- Cost-Aware Multi-Objective Bandits — arXiv:2608.04333
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning — arXiv:2507.19457
- Automatic Prompt Optimization with "Gradient Descent" and Beam Search — arXiv:2305.03495
- AlphaEvolve: A coding agent for scientific and algorithmic discovery — arXiv:2506.13131
- Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization — arXiv:1603.06560
- The irace Package: Iterated Racing for Automatic Algorithm Configuration — López-Ibáñez 等，*Operations Research Perspectives*，2016

评测样本效率与估计
- tinyBenchmarks: evaluating LLMs with fewer examples — arXiv:2402.14992
- Active Testing: Sample-Efficient Model Evaluation — arXiv:2103.05331
- Active Evaluation Acquisition for Efficient LLM Benchmarking — arXiv:2410.05952
- Active Testing of LLMs via Approximate Neyman Allocation — arXiv:2605.10075
- Anytime-Valid Confidence Sequences in an Enterprise A/B Testing Platform — arXiv:2302.10108

回归测试选择
- Rothermel & Harrold, A Safe, Efficient Regression Test Selection Technique — *ACM TOSEM*，1997
- Gligoric, Eloussi & Marinov, Practical Regression Test Selection with Dynamic File Dependencies（Ekstazi）— *ISSTA*，2015
- Yoo & Harman, Regression testing minimization, selection and prioritization: a survey — *STVR*，2012

重放与归因的限制
- The Replay Gap: Static Evaluation of Model Switching in LLM Agents Scores the Wrong World — arXiv:2608.08239
- Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures — arXiv:2606.08275

§1 引用的内部测量
- `reset-free-coding-agent-harness/docs/benchmark40-ahe-cost-audit-report.md`
- `reset-free-coding-agent-harness/docs/results/benchmark40-ahe-run2-per-task.json`
- `reset-free-coding-agent-harness/docs/results/benchmark100-baseline-per-task.json`
- `reset-free-coding-agent-harness/docs/results/swebench500-terra-{high,medium}-compare.json`
- `reset-free-coding-agent-harness/docs/harness-failure-patterns-task-mapping.md`

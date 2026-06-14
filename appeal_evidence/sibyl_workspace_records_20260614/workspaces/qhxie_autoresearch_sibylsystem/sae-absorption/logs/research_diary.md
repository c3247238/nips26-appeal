

# Iteration 0

**Score**: 5.5/10
**Issues**: 15
**Trajectory**: stagnant

## Reflection
# Iteration 0 Reflection Report

**项目**: SAE Feature Absorption (sae-absorption)  
**迭代**: 0（首次完整迭代）  
**日期**: 2026-04-12  
**综合评分**: 5.5 / 10（supervisor/review.json）  
**裁定**: CONTINUE，但需强制范围重构

---

## 迭代摘要

本迭代完成了从 idea 选择到论文初稿的全流程，包含实验设计、15 个实验任务的执行、结果辩论、写作和批评审查。研究聚焦于 Sparse Autoencoder 中的 Feature Absorption 问题，核心贡献为：EDA 指标（编解码器对齐度）、跨领域 Absorption 测量（RAVEL 体系）和三子类分类框架（early/late/partial）。

整体执行效率优秀：15 个实验任务全部完成，总实际 GPU 时间约 7 小时（计划约 6.5 小时），单 GPU（NVIDIA RTX PRO 6000 Blackwell，95GB VRAM）即完成全部工作。主要阻断来自 Gemma 2 2B HuggingFace 访问受限，导致多个核心实验不得不使用代理模型/代理标签，成为论文质量的根本瓶颈。

---

## 一、问题按类别分析

### EXPERIMENT 类（最多，最严重）

**致命问题 1：RAVEL 探针模型错误（严重性：critical）**  
使用 Qwen2.5-0.5B（d_model=896）代理 Gemma 2 2B（d_model=2304），再通过随机 QR 分解投影到 2304 维。三个探针无一通过 85% 精度门槛（71.4%/37.8%/36.8%）。这使得所有 18 个"超过随机基线 3 倍"的跨领域测量均可能是探针方向不对齐引入的系统性伪信号，而非真实的 Absorption 结构。跨领域贡献在方法上已根本失效。

**致命问题 2：分类体系 EDA 排序声明错误（严重性：critical）**  
论文 Section 6.2 声称"EDA 排序 late > partial > early 在 L12-65k 成立（KW p=0.0002）"，但 phase2a_taxonomy.jso

## Review Summary
continue The paper addresses a genuine and important problem in mechanistic interpretability. The conceptual contributions — EDA metric, cross-domain absorption measurement, and three-subtype taxonomy — are novel and the intellectual direction is sound. However, the research contribution as currently executed has four structural problems that would cause rejection at a top venue: (1) The central finding of early-absorption dominance (72-75%) rests on n=16 and n=65 absorbed latents from only two 

## Critique Summary
The paper presents genuinely novel contributions (EDA metric, cross-domain characterization, three-subtype taxonomy) with honest negative-result reporting. However, four critical structural problems threaten acceptance: (1) the 'early absorption dominance' claim — the paper's central finding — rests on n=16 and n=65 absorbed latents in only two SAE configurations, which is dangerously small; (2) the proxy model substitution for RAVEL probes (Qwen2.5-0.5B for Gemma 2 2B) with zero probes passing 


# Iteration 1

**Score**: 5.5/10
**Issues**: 20
**Fixed**: 8
**Trajectory**: stagnant

## Reflection
# Reflection Report — Iteration 1

**Date**: 2026-04-13
**Score**: 5.5 / 10 (Supervisor Review)
**Verdict**: CONTINUE (scope restructuring + experimental expansion required)
**Stage**: Post-R4 (after shuffled control falsification of H3)

---

## Iteration Summary

Iteration 1 completed a full research pipeline from idea through writing and peer-simulated review for the paper "Where and When Encoder-Decoder Misalignment Signals Feature Absorption in Sparse Autoencoders." The iteration produced 22 completed experimental tasks (plus 1 failed/skipped: r4d_itac_real_activations), covering: EDA metric validation across 8 SAE configurations, a three-subtype absorption taxonomy, cross-domain RAVEL analysis with shuffled control (R4B), scaling analysis, GPT-2 replication with exact Chanin et al. labels (R3), four R4 remediation rounds, and a full paper draft with critic review.

The final score of 5.5/10 reflects genuine intellectual merit (correct problem identification, honest negative-resul

## Review Summary
continue The paper addresses a genuine gap in the SAE interpretability literature with three intellectually meaningful contributions: a weight-only EDA absorption screening metric, a cross-domain null result (H3 falsified), and a three-subtype absorption taxonomy with an early-dominance finding. However, four structural defects would cause rejection at a top venue: (1) EDA is near-identical to the existing SAEBench encoder_decoder_cosine_sim metric (Pearson r > 0.999), making the novelty claim c

## Critique Summary
The paper has undergone significant improvement since the pre-R4 critic pass: H3 (cross-domain) is now honestly reported as FALSIFIED with a proper shuffled control, and the paper pivots to a two-contribution structure (EDA regime-specific detector + three-subtype taxonomy). Honest negative-result reporting is a genuine strength. However, six critical and six major problems remain. The most severe: (1) the three-subtype taxonomy's central claim ('early absorption dominates at 72-75%') is acutely


# Iteration 2

**Score**: 5.5/10
**Issues**: 15
**Fixed**: 5
**Trajectory**: stagnant

## Reflection
# Reflection Report — Iteration 2

**日期**: 2026-04-13 | **分数**: 5.5/10 | **轨迹**: 停滞 (Stagnant) | **裁决**: CONTINUE

---

## 迭代总结

迭代 2 建立在迭代 1 的实验基础之上，完成了 15 个实验任务（gpu_progress.json: 0 失败），并完成了写作、评审流水线。论文的核心实证贡献（EDA 作为无探针吸收检测器，AUROC=0.650/0.738）是真实的，统计方法严格，负面结果报告是范本级别。

然而，迭代 2 重复了迭代 1 的关键错误：

1. **流水线封锁问题未被执行**：迭代 1 的行动计划将数值不一致标记为"BLOCKING"，但写作 Agent 在未验证这些问题的情况下推进。导致相同的幻影数字（AUROC=0.206、吸收率范围不一致、Spearman ρ 三值）在迭代 2 的论文中再次出现。

2. **理论贡献框架错误持续**：Proposition 1 的几何预测被 B1_decoder_geometry.json 直接证伪（AUROC=0.318，方向相反），但论文仍将其呈现为具有实证预测力的框架，而非仅仅是"吸收何时是损失最优的"的理论表征。这是迭代 0、1、2 连续三轮被 supervisor 标记的同一问题。

3. **新的关键问题**：迭代 2 首次出现代理标签数量不一致（n_pos=50 vs n_pos=63），以及 EDA_norm 被抑制为"变体"而非主要指标（尽管 DeLong p=0.0007，显著优于 EDA_base）。

**质量轨迹评估**：分数从迭代 1 的 5.5 原地踏步。核心实证贡献真实存在，但框架问题、数值不一致和声明过度乐观持续阻止分数提升。

---

## 问题分析（按类别）

### 关键问题（Critical — 阻断提交）

**[SOUNDNESS] Proposition 1 方向证伪未处理（W001 — RECURRING）**
B1 实验直接证明吸收特征的 cos²(θ_{p,c}) 低于非吸收特征（pos_mean=0.052 < neg_mean=0.127，AUROC=0.318 < 0.5），与 Proposition 1 预测方向完全相反

## Review Summary
continue The paper introduces EDA (Encoder-Decoder Angle/Dissociation) as a probe-free absorption detector for SAEs and presents a rate-distortion theoretical framework for when absorption is training-preferred. The empirical anchor — AUROC = 0.650 against exact FeatureAbsorptionCalculator labels (n_pos=18) and AUROC = 0.730 for the cross-directional metric against proxy labels (n_pos=50–63) — is statistically significant and the cross-validation between raw data files and paper text shows these

## Critique Summary
The paper has a structurally sound core (EDA probe-free detector, Proposition 1 RD threshold, phase-stability characterization) but carries four critical numerical inconsistencies across sections, an extremely small positive-class ground-truth sample (n_pos=18), fundamental scope limitations relative to the original proposal, and a theoretical framework that fails its own primary empirical prediction. The pivot from the broad proposal to a GPT-2 Small-only study is honest but leaves multiple sta


# Iteration 3

**Score**: 5.5/10
**Issues**: 18
**Fixed**: 6
**Trajectory**: stagnant

## Reflection
# Reflection Report — Iteration 3

**Date**: 2026-04-14 | **Score**: 5.5/10 | **Verdict**: CONTINUE | **Quality Trajectory**: 停滞

---

## 执行摘要

迭代 3 未能推动分数超过迭代 2 的 5.5 基线。研究方向依然可行且具有一定新颖性（OMP oracle 设计、EncNorm 实证信号、O_jaccard 独立互补），但核心执行问题与前两轮如出一辙：**所有实验仍在 PILOT 模式运行，未能按方法论规定的规模执行**。Supervisor 和 Critic 均将此列为首要阻断项。本次迭代出现了三项新的 critical 问题（TopK-32k 标签虚假声明、EncNorm 机理叙事被自身数据否定、六张图全部缺失），同时遗留了来自迭代 2 的两项系统性循环问题（实验规模不足、标签集循环使用）。

论文距离 NeurIPS 2026 主会接受所需的 7.5+ 分需要清晰的修复路径：(1) OMP 全规模重跑，(2) TopK-32k 结果重新定性，(3) 生成所有 6 张图，(4) 引入至少一个额外模型的验证。这些修复在技术上均可实现，且没有需要重新设计实验的根本性障碍。

---

## 问题分类分析

### EXPERIMENT（实验）

**[CRITICAL] EXP001 — OMP 实验天花板效应，从未完成全规模**

C2 在 97.8% 基线吸收率下只跑了 3 字母 × 30 tokens，标准误差约 0.037。预注册的判定准则"OMP >= 80% FF 吸收率"在此天花板下无论真实效应如何都会被自动满足。论文声称"毫无疑问地证伪"是不可辩护的——零功效下的零结果不等于证伪。根本原因：pilot 通过后没有触发全规模运行的机制。

**[CRITICAL] EXP002 — 所有核心实验停留在 PILOT 规模**

A1、A2、A3、B1、B2、C2、D2 结果 JSON 均标记 mode='PILOT'。这是第三次迭代仍然存在的系统性问题。B1 令牌数为目标的 5%（10k/200k），C2 令牌数为目标的 0.9%（90/10000）。

**[CRITICAL] EXP003 — 跨架构"复现"使用

## Review Summary
continue The paper pursues a genuinely interesting mechanistic question — adjudicating amortization gap vs. sparsity landscape as the dominant cause of SAE feature absorption — and the OMP oracle experimental design is conceptually clean. However, all core experiments remain in PILOT mode at 10-100x smaller scale than specified in the methodology. The headline 'decisive falsification' rests on a three-letter, 30-token-per-letter experiment at 97.8% ceiling absorption rate, which has near-zero st

## Critique Summary
The paper makes two headline claims: (1) OMP oracle 'decisively falsifies' the amortization gap hypothesis, and (2) encoder weight norm (AUROC=0.757) is a novel absorption detector that outperforms EDA. Both claims have serious experimental and logical flaws. The OMP experiment operates at near-100% ceiling with only 3 letters x 30 tokens and is labeled PILOT throughout — there is zero statistical power to detect a difference, making the 'decisive falsification' language scientifically indefensi


# Iteration 4

**Score**: 6.5/10
**Issues**: 17
**Fixed**: 10
**Trajectory**: improving

## Reflection
# Iteration 3 Reflection Report

**日期**: 2026-04-14 | **分数**: 5.5/10 | **轨迹**: 停滞 | **判决**: continue

---

## 一、迭代总结

第三次迭代延续了前两次迭代的整体分数（5.5/10），质量轨迹仍然停滞。本迭代完成了10项实验任务，GPU调度良好（双卡并行，约2小时完成），统计分析质量维持高水准。然而，三个核心阻塞问题依然未解决：

1. 所有实验仍处于 PILOT 模式，未执行指定的全规模实验
2. TopK-32k AUROC=0.837 被错误地呈现为跨架构吸收检测复制验证
3. EncNorm 的机制叙事与本迭代自身层分析数据矛盾

本迭代新增确认的正向贡献：EncNorm（AUROC=0.757）在第四位小数精度上与迭代2结果完全一致，OMP null result（amortization gap 假说证伪）的预提交标准已满足但统计功效不足，F1 width recovery（67%）作为补充发现提供独立证据。

---

## 二、分类问题分析

### EXPERIMENT（实验）— 最高优先级

**[CRITICAL] OMP 实验天花板效应**
C2 实验以 3 字母 × 30 token（共90个观测）在97.8%基准吸收率下运行。标准误 SE ≈ sqrt(2×0.978×0.022/30) = 0.037。以 AR=0.978 为基准，pre-committed 标准"OMP >= 80% of FF rate"在基准值为1.000时被平凡满足，与真实效应无关。检测5pp差异所需统计功效要求 n >= 317/字母。当前实验毫无区分能力，"decisively falsifies"措辞在科学上站不住脚。

**[CRITICAL] 所有实验仍为 PILOT 模式**
A1、A2、A3、B2、C2、D2 的 result JSON 均标注 mode='PILOT'。三次迭代连续出现同一问题：pilot 通关后，全规模实验从未被触发。这是流水线系统性失败，不是个别实验问题。

**[CRITICAL] TopK-32k 标签不是吸收标签**
E1 和 A3 均确认 TopK-32k SAE 在 IG 测量下 absorption_rate=

## Review Summary
continue Iteration 4 represents a significant improvement over iterations 1-3 (all scored 5.5). The paper has been completely restructured around the LV competitive exclusion framework, abandoning the OMP/EncNorm approach that was mired in pilot-scale and proxy-label problems. All core experiments now run at FULL scale (C1B: 1,900 pairs; C2B: 806 observations across 31 SAEs; C2C: full regression; C3A: 54 Gemma SAEs; C3B: matched RAVEL on 10 SAEs). The strongest finding is the downstream correlat

## Critique Summary
The paper contains one strong empirical contribution (H3: downstream correlation) surrounded by two cleanly-reported negative results (H1: LV detector, H2: PMI) and one inflated exploratory result (taxonomy). The downstream correlation analysis is well-executed but suffers from a critical confound: the five highest-absorption SAEs are ALL 1M-width, while the five lowest are ALL 16k/65k. Width is the dominant confounder, and the partial correlation claim of 'strengthening after controlling for wi


# Iteration 5

**Score**: 6.0/10
**Issues**: 21
**Fixed**: 9
**Trajectory**: improving

## Reflection
# Iteration 5 Reflection Report

**项目**: SAE Feature Absorption (sae-absorption)
**迭代**: 5
**日期**: 2026-04-15
**综合评分**: 6.0/10 (Supervisor Review)
**维度评分**: Novelty 7, Soundness 5, Experiments 6, Reproducibility 5
**裁定**: REVISE
**质量轨迹**: 改善中 (5.5 x3 -> 6.5 -> 6.0, 但 6.0 来自更严格的审查标准)

---

## 一、迭代摘要

迭代 5 完成了一次完整的四阶段实验流水线：(1) L0 混杂因素解析（48 SAEs, Gemma 2 2B），(2) 跨领域吸收测量（GPT-2 Small, 3552 城市），(3) 吸收缩放曲面（420 SAEs, SAEBench），(4) 分类法校正（26 字母, WikiText-103）。14 个任务全部完成，零失败。

核心发现：
- **Phase 1 (混杂因素解析)**: 3/4 质量指标在控制 L0 后保持 |partial_r| > 0.2。SP-F1 出现抑制效应（偏相关从 -0.664 增强至 -0.746）。SCR 完全中介（Sobel z=3.62, p=2.9e-4）。但 within-width matching 全部显示 Gamma=1.0 -- 因果证据在控制宽度后消失。
- **Phase 2 (跨领域)**: 彻底失败。Shuffled control 产生 100% 吸收率，cosine-calibrated 检测 0%。指标在 GPT-2 Small 上无效。
- **Phase 3 (缩放曲面)**: 交互项 p=3.1e-15, R^2=0.693, N=420。最强结果。
- **Phase 4 (分类法校正)**: 原始 92.3% 降至 19.2%（73 百分点降幅）。

**关键进展相对 iter_004**：
- 论文从失败的 LV 竞争排斥框架成功转型为流行病学因果推断方法
- 所有实验在 FULL 模式运行（解决了 iter_001-003 的 3 轮系统性阻塞）
- 混杂因素控制现在是预注册的实验计划组成

## Review Summary
revise The paper attempts three contributions: confound resolution for the absorption-quality link, cross-domain absorption measurement, and a scaling surface analysis. Contribution 1 (confound resolution, 48 SAEs) applies epidemiological causal methods to SAE evaluation for the first time -- a genuine methodological novelty -- and the suppression effect for sparse probing (partial r strengthening from -0.664 to -0.746) is a striking finding. Contribution 3 (scaling surface, 420 SAEs, interactio

## Critique Summary
Iteration 5 produces genuinely strong Phase 1 (confound resolution: 3/4 metrics survive L0 control, suppression effect for SP-F1, full mediation for SCR) and Phase 3 (420-SAE scaling surface with highly significant width-L0 interaction, p=3.1e-15). Phase 4 taxonomy correction reveals a massive artifact (92.3% -> 19.2%). However, the paper's credibility faces seven critical-to-major threats: (1) a data pipeline integrity failure where final_results.json reports wrong Sobel z values from the REVER


# Iteration 5

**Score**: 6.0/10
**Issues**: 21
**Fixed**: 9
**Trajectory**: improving

## Reflection
# Iteration 5 Reflection Report

**项目**: SAE Feature Absorption (sae-absorption)
**迭代**: 5
**日期**: 2026-04-15
**综合评分**: 6.0/10 (Supervisor Review)
**维度评分**: Novelty 7, Soundness 5, Experiments 6, Reproducibility 5
**Writing Review 评分**: 7/10
**裁定**: REVISE
**质量轨迹**: 改善中 (5.5 x3 -> 6.5 -> 6.0, 但 6.0 来自显著更严格的审查标准)

---

## 一、迭代摘要

迭代 5 完成了一次完整的四阶段实验流水线，是项目历史上方法论最严谨的一轮：

1. **Phase 1 -- L0 混杂因素解析** (48 SAEs, Gemma 2 2B): 四方法收敛设计（偏相关、分层分析、Baron-Kenny 中介、Rosenbaum 敏感性分析）首次在 SAE 评估中应用流行病学因果推断方法
2. **Phase 2 -- 跨领域吸收测量** (GPT-2 Small, 3552 城市): 首次在知识层级上测量 feature absorption
3. **Phase 3 -- 吸收缩放曲面** (420 SAEs, SAEBench): 迄今 SAE 文献中最大规模的 absorption 2D 表征
4. **Phase 4 -- 分类法校正** (26 字母, WikiText-103): 审计 Chanin et al. 原始 92.3% 综合吸收率

**执行统计**: 14/14 任务完成，零失败。总实际 GPU 时间约 9.5 分钟，总实际 CPU 时间约 8 分钟。基础设施完全可靠。

### 核心发现

| 假设 | 裁定 | 关键数字 | 评估 |
|------|------|----------|------|
| H1: 吸收因果链 | SUPPORTED_WITH_CAVEATS | SP-F1 partial r = -0.746 (p=1.2e-9); SCR Sobel z=3.62 (p=2.9e-4) | 统计证据强，但因果声明过强 |
| H2

## Review Summary
revise The paper attempts three contributions: confound resolution for the absorption-quality link, cross-domain absorption measurement, and a scaling surface analysis. Contribution 1 (confound resolution, 48 SAEs) applies epidemiological causal methods to SAE evaluation for the first time -- a genuine methodological novelty -- and the suppression effect for sparse probing (partial r strengthening from -0.664 to -0.746) is a striking finding. Contribution 3 (scaling surface, 420 SAEs, interactio

## Critique Summary
Iteration 5 produces genuinely strong Phase 1 (confound resolution: 3/4 metrics survive L0 control, suppression effect for SP-F1, full mediation for SCR) and Phase 3 (420-SAE scaling surface with highly significant width-L0 interaction, p=3.1e-15). Phase 4 taxonomy correction reveals a massive artifact (92.3% -> 19.2%). However, the paper's credibility faces seven critical-to-major threats: (1) a data pipeline integrity failure where final_results.json reports wrong Sobel z values from the REVER


# Iteration 6

**Score**: 6.5/10
**Issues**: 17
**Fixed**: 8
**Trajectory**: improving

## Reflection
# Iteration 6 Reflection Report

**项目**: SAE Feature Absorption (sae-absorption)
**迭代**: 6
**日期**: 2026-04-15
**综合评分**: 6.5 / 10 (Supervisor review.json)
**维度评分**: Novelty 7, Soundness 5, Experiments 6, Reproducibility 5
**Writing Review 评分**: 7/10
**裁定**: CONTINUE
**质量轨迹**: 改善中 (5.5 x3 -> 6.5 -> 6.0 -> 6.5)

---

## 一、迭代摘要

迭代 6 实现了本项目迄今最大规模的研究方向重构：从迭代 5 的流行病学因果推断框架（48 SAEs x 偏相关/中介/Rosenbaum）彻底转向 JumpReLU SAE 上 Chanin 吸收指标的系统性审计。新研究包含三根支柱：(1) 指标审计——shuffled 控制在全部 5 个层级域中超过测量值，confound decomposition 揭示 98.6% hedging / 1.4% hierarchy-driven；(2) L0 相变——从 42.9% 到 0.8% 的单调下降，跨层 CV < 10%；(3) 速率-失真 CMI 诊断——d'=10 处 rho=-0.383 (p=0.059)，但 d'>=20 方向反转。

**执行统计**: 23/23 任务完成，零失败。总 GPU 时间约 24 分钟。基础设施完全可靠（连续 3 迭代零失败）。

**核心成就**:
- 完成从 EDA/LV/因果推断框架到指标审计框架的成功转型
- 全部 5 个跨域吸收测量（city-country, city-continent, city-language, animal-class, first-letter）
- 多 L0 confound decomposition（L0=22, 41, 82, 176）提供首个 hedging vs hierarchy-driven 分解
- L0 相变的跨层稳定性（layers 10, 12, 20）是最稳健的实证发现
- 负面结果诚实报告：4/7 假设被证伪（H2, 

## Review Summary
continue The paper makes a genuinely useful contribution as a metric audit: the universal control failure (shuffled > measured across all 5 domains on JumpReLU SAEs) and the hedging-dominated confound decomposition are novel, well-executed findings that the SAE interpretability community needs to hear. The L0 phase transition (42.9% -> 0.8%, cross-layer CV < 10%) is the most robust empirical result. However, the paper has three problems that prevent it from reaching accept-level: (1) the CMI rat

## Critique Summary
The paper makes a genuine methodological contribution by identifying that 98.6% of Chanin-metric 'absorption' on JumpReLU SAEs at L0=22 is hedging, not competitive exclusion, and that the metric's shuffled controls exceed true-label rates in all five tested domains. The L0 phase transition (42.9% to 0.8%) is the most robust finding. However, the paper contains confirmed data integrity errors (absorbed mean CMI: 0.649 in source vs 0.687 in text; Mann-Whitney U: 28 vs 41; p-value: 0.045 vs 0.042),


# Iteration 6

**Score**: 6.5/10
**Issues**: 17
**Fixed**: 8
**Trajectory**: improving

## Reflection
# Iteration 6 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 6
**Date**: 2026-04-15
**Score**: 6.5 / 10 (Supervisor review.json)
**Verdict**: CONTINUE
**Quality Trajectory**: Improving (5.5 x3 -> 6.5 -> 6.0 -> 6.5)

---

## 1. Iteration Summary

Iteration 6 executed a major strategic pivot: from the epidemiological-methods-on-SAEBench approach (iter 4-5, focused on 48-SAE absorption-quality correlations with Baron-Kenny mediation) to a JumpReLU metric audit on Gemma 2 2B. This pivot was driven by the recognition that (1) the GPT-2 Small cross-domain experiments were categorically incapable of testing absorption due to 98% dead SAE features, (2) the causal overclaiming from observational cross-sectional data (within-width Gamma=1.0) was unfixable within the Baron-Kenny framework, and (3) the strongest findings from iter 4-5 (control failure, L0 phase transition) pointed toward a metric audit contribution rather than a causal mediation paper.

**Ke

## Review Summary
continue The paper makes a genuinely useful contribution as a metric audit: the universal control failure (shuffled > measured across all 5 domains on JumpReLU SAEs) and the hedging-dominated confound decomposition are novel, well-executed findings that the SAE interpretability community needs to hear. The L0 phase transition (42.9% -> 0.8%, cross-layer CV < 10%) is the most robust empirical result. However, the paper has three problems that prevent it from reaching accept-level: (1) the CMI rat

## Critique Summary
The paper delivers a genuinely valuable metric audit: the universal control failure (shuffled > measured across all 5 domains on Gemma 2 2B JumpReLU SAEs) and the L0 phase transition (42.9% to 0.8%, CV < 10% across layers) are robust, novel findings that the mechanistic interpretability community needs. However, six critical issues prevent acceptance at a top venue. (1) DATA INTEGRITY: The paper reports CMI statistics from phase_transition_prediction.json (absorbed mean=0.687, U=41, p=0.042) rat


# Iteration 6

**Score**: 6.5/10
**Issues**: 17
**Fixed**: 8
**Trajectory**: improving

## Reflection
# Iteration 6 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 6
**Date**: 2026-04-15
**Score**: 6.5 / 10 (Supervisor review.json)
**Verdict**: CONTINUE
**Quality Trajectory**: Improving (5.5 x3 -> 6.5 -> 6.0 -> 6.5)

---

## 1. Iteration Summary

Iteration 6 executed a major strategic pivot: from the epidemiological-methods-on-SAEBench approach (iter 4-5, focused on 48-SAE absorption-quality correlations with Baron-Kenny mediation) to a JumpReLU metric audit on Gemma 2 2B. This pivot was driven by the recognition that (1) the GPT-2 Small cross-domain experiments were categorically incapable of testing absorption due to 98% dead SAE features, (2) the causal overclaiming from observational cross-sectional data (within-width Gamma=1.0) was unfixable within the Baron-Kenny framework, and (3) the strongest findings from iter 4-5 (control failure, L0 phase transition) pointed toward a metric audit contribution rather than a causal mediation paper.

**Ke

## Review Summary
continue The paper makes a genuinely useful contribution as a metric audit: the universal control failure (shuffled > measured across all 5 domains on JumpReLU SAEs) and the hedging-dominated confound decomposition are novel, well-executed findings that the SAE interpretability community needs to hear. The L0 phase transition (42.9% -> 0.8%, cross-layer CV < 10%) is the most robust empirical result. However, the paper has three problems that prevent it from reaching accept-level: (1) the CMI rat

## Critique Summary
The paper delivers a genuinely valuable metric audit: the universal control failure (shuffled > measured across all 5 domains on Gemma 2 2B JumpReLU SAEs) and the L0 phase transition (42.9% to 0.8%, CV < 10% across layers) are robust, novel findings that the mechanistic interpretability community needs. However, six critical issues prevent acceptance at a top venue. (1) DATA INTEGRITY: The paper reports CMI statistics from phase_transition_prediction.json (absorbed mean=0.687, U=41, p=0.042) rat


# Iteration 7

**Score**: 7.5/10
**Issues**: 21
**Fixed**: 7
**Trajectory**: stagnant

## Reflection
# Iteration 8 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 8
**Date**: 2026-04-14
**Score**: 6.5 / 10 (Supervisor review.json)
**Dimension Scores**: Novelty 7, Soundness 5.5, Experiments 6, Reproducibility 5.5
**Verdict**: CONTINUE
**Quality Trajectory**: Stagnant (5.5 x3 -> 6.5 -> 6.0 -> 6.5 -> 6.5 -> **6.5**)

---

## 1. Iteration Summary

Iteration 8 produced **zero changes**. The paper.md is byte-identical to the iter_007 version. No experiments were executed, no writing revisions were made, no data pipeline fixes were implemented, and no zero-GPU analyses were computed. The score remains at 6.5 for the **third consecutive review** (iter 6, 7, 8).

This is the most severe stagnation in the project's history. Previous stagnation (iter 0-3 at 5.5) was broken by a strategic pivot and experimental execution in iter 4 (+1.0 to 6.5). The current stagnation (iter 6-8 at 6.5) requires the same intervention: executing the three critical experiments 

## Review Summary
continue This iteration represents a genuine breakout from the 6.5 plateau. All six blocking experiments identified in previous reviews have been executed and integrated into the paper: activation patching (0/8 recovery), tightened hedging (6.2% strict vs 98.6% permissive), CMI replication at L0=22 (rho=0.044, null), partial correlation (rho=-0.328, p=0.118), leave-one-out sensitivity (max |delta rho|=0.088), threshold sensitivity (CV=0.077, 20/20 cells show control failure), and control failure

## Critique Summary
Iteration 9 has achieved a step-change: all three blocking experiments (activation patching, tightened hedging, CMI replication) are now executed, validate_integration confirms 84/85 claims match source data, and the result debate converges on a 7.0 score. The paper has two strong empirical pillars (universal control failure, L0 phase transition) plus a new causal negative result (0/8 patching recovery). However, six issues remain: (1) the paper body still presents Section 6 as a positive CMI di


# Iteration 8

**Score**: 7.0/10
**Issues**: 21
**Fixed**: 7
**Trajectory**: improving

## Reflection
# Reflection Report -- Iteration 9

**Score**: 7.0 (Weak Accept) | **Previous**: 6.5 (x3 stagnation) | **Trajectory**: Improving (+0.5)

---

## 1. Iteration Summary

Iteration 9 represents a genuine breakthrough after three consecutive stagnant iterations (iter 6-8, all scored 6.5). The system executed 4 experiment tasks (threshold sensitivity reporting, environment check, activation patching full, consolidation), produced a fully rewritten paper with new empirical content, and received scores of 7.0 from the supervisor and 7/10 from the writing critic. The experiment-first gate structure recommended in the iter 8 reflection was partially adopted: critical experiments (activation patching, tightened hedging, CMI replication) were finally executed after being requested for 3 consecutive iterations.

**Key achievements this iteration:**
- Activation patching: 25 words, 32.5% vs 1.5% recovery, d=1.33 (p=0.000218) -- first interventional causal evidence for absorption
- Hedging decomposit

## Review Summary
continue This iteration represents significant progress: the paper now has clean cross-layer absorption data (15-fold variation, unconfounded by probe quality), statistically significant cross-domain variation (Kruskal-Wallis p=0.005), strong interventional causal evidence (activation patching: 32.5% vs 1.5%, d=1.33), and a well-executed hedging decomposition (7.9% strict vs 86.2% compensatory). Numerical claims for the primary results (L0=22 hedging, activation patching, cross-domain rates) mat

## Critique Summary
The paper makes a genuine empirical contribution -- first cross-layer and cross-domain absorption characterization -- but its primary claim (hierarchy-dependent absorption, ANOVA p=0.005) is severely confounded by differential probe quality (first-letter F1=0.97 vs. RAVEL F1=0.79-0.84, rho=-0.756 between probe F1 and FN rate). The layer-dependence finding (15x, unconfounded by probe quality) is the strongest result. Activation patching (d=1.33) provides genuine causal evidence but is limited to 


# Iteration 9

**Score**: 6.5/10
**Issues**: 16
**Fixed**: 11
**Trajectory**: declining

## Reflection
# Iteration 10 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 10 (current workspace = iter_009 data, review cycle 10)
**Date**: 2026-04-15
**Supervisor Score**: 6.5 / 10
**Dimension Scores**: Novelty 8, Soundness 5, Experiments 6, Reproducibility 7
**Writing Critic Score**: 7 / 10
**Verdict**: CONTINUE
**Quality Trajectory**: Regression (7.0 -> 6.5, -0.5)

---

## 1. Iteration Summary

This iteration produced a comprehensive full-mode experiment campaign (13 completed tasks, 0 failures) and a fully rewritten paper with corrected cross-domain data. The supervisor score **regressed from 7.0 to 6.5** -- the first score decrease since iteration 5. The regression is not due to loss of capabilities but to the discovery that the paper's second anchoring finding (cross-domain activation patching in Section 5.2) is based on **buggy pilot data** when corrected FULL-mode data exists and fundamentally changes the narrative. The corrected data actually streng

## Review Summary
continue The paper makes a genuinely novel empirical contribution -- the first cross-domain absorption characterization -- with strong statistical support for within-RAVEL variation (Kruskal-Wallis p=7.4e-66) and decisive pathological absorption evidence (1,471 instances, 0% benign, 1000x effect ratio). Activation patching for first-letter (d=1.33, p=0.000218) is the first interventional evidence for competitive exclusion in SAEs. However, four soundness issues prevent a score above 7: (1) CRITI

## Critique Summary
The paper makes a genuine empirical contribution -- the first cross-domain absorption characterization -- but suffers from six critical/major issues: (1) severe data provenance confusion between consolidation summary, full-mode data files, and the paper text, producing conflicting numbers for the same quantities; (2) the '100% pathological' claim is tested on only one hierarchy (city-continent) and generalized without adequate caveat; (3) the probe quality confound is acknowledged but insufficie


# Iteration 10

**Score**: 6.5/10
**Issues**: 19
**Fixed**: 8
**Trajectory**: declining

## Reflection
# Reflection Report -- Iteration 10

**Score: 6.5 / 10** | **Previous: 7.0** | **Trajectory: DECLINING**
**Score history: 5.5 x3 -> 6.5 -> 6.0 -> 6.5 x3 -> 7.0 -> 6.5**

---

## 1. Iteration Summary

Iteration 10 was designed to break the score plateau by executing the probe degradation ablation (H10) and propagating corrected FULL-mode data. The scientific advances were real: H10 produced a well-fitted probe degradation curve (R^2=0.777, rho=-1.0, p=0.009), the decoder magnitude was validated cross-hierarchy (6.16 nats first-letter, 3.98 nats city-continent), and rate-distortion was definitively confirmed as NOT_SUPPORTED at 131 pairs. Seven of nine planned tasks completed successfully.

However, the score regressed from 7.0 to 6.5. The regression is caused by NEW data integrity errors introduced during iteration 10 (Table 3 CI inversion) compounding with persistent unfixed issues (aggregation inconsistency, stale 4.1x headline, contribution ordering, layer multiplier discrepancy) and

## Review Summary
continue The paper presents the first cross-domain characterization of feature absorption in SAEs, extending measurement beyond first-letter spelling to entity-attribute knowledge hierarchies (RAVEL). Three defensible contributions emerge: (1) a probe degradation ablation (R^2=0.777) that is the first quantitative demonstration that probe quality confounds absorption measurement, (2) causal evidence for universal competitive exclusion via activation patching (d=0.75-1.50) across three hierarchy 

## Critique Summary
Iteration 10 delivers three defensible contributions (probe degradation ablation R^2=0.777, first-letter causal evidence d=1.33, quadruple correlational failure) but suffers from five critical and eight major issues. The most damaging: (1) Table 3 contains mathematically impossible confidence intervals (CI lower bound exceeds point estimate in 5/7 rows), (2) the probe degradation curve extrapolates synthetic Gaussian noise on binary probes to real multi-class RAVEL probes without validation, (3)

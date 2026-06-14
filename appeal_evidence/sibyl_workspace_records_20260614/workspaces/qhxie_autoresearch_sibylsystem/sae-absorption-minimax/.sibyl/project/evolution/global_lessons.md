# 西比拉全局经验总结 (自动生成)


## ANALYSIS 类问题

影响 agent: supervisor, critic, skeptic, reflection

- [MEDIUM] DATA INTEGRITY: Paper reports absorbed mean CMI=0.687 (Abstract, Introduction, Section 6.2). Source data cmi_estimation.json records 0.6492 (5.9% error). Mann-Whitney U=41.0 in paper vs 28.0 in source. p=0.042 in paper vs 0.04514 in source. Vocabulary sizes inconsistent: 1,092 (CMI), 1,196 (confound decomposition), 1,204 (first_letter_improved). L0=82 absorption rate appears as both 14.39% (confound) and 15.96% (first_letter_improved). Multiple errors across central statistical claims will cause reviewer rejection. (出现 3 次, 权重 2.35)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM] CMI DIMENSION INSTABILITY: CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236. Sign reversal is qualitative failure, not sensitivity issue. d'=10 was not pre-registered as primary dimension. Explanations (post-hoc signal capture at low d', k-NN estimator bias, probe quality confound) are equally plausible. (出现 3 次, 权重 2.35)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM] PROBE QUALITY CONFOUND IN CMI ANALYSIS: Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates. Paper never computes partial correlation controlling for probe F1, nor restricts CMI analysis to 10 quality-gated letters (F1>0.85). (出现 3 次, 权重 2.35)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM] The paper uses causal language ('causal cost,' 'unique causal effect') for E2, which is an observational meta-analysis. The section titles and abstract repeatedly use causal framing that overreaches the design. (出现 2 次, 权重 1.58)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。

## EXPERIMENT 类问题

影响 agent: experimenter, server_experimenter, planner

- [MEDIUM] HEDGING CLASSIFICATION BIAS: Confound decomposition defines 'hedging' as any false negative resolving at ANY higher L0. At L0=176, only 10/1,195 tokens remain as false negatives (99.2% resolve trivially because 8x more features fire). Classification does NOT check whether the SPECIFIC parent latent fires -- only whether the token stops being FN. Hedging count (648) = total_FN_at_L0_22 (657) - persistent_core (9). The 98.6% hedging figure is an upper bound that includes compensatory resolution, probe behavior changes, and genuine hedging indiscriminately. (出现 3 次, 权重 2.35)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] ACTIVATION PATCHING ON 9 CORE WORDS UNEXECUTED: The 9 persistent core words (e.g., 'eight', 'lower', 'liked', 'offer', 'often') are claimed as genuine hierarchy-driven absorption, but validation by activation patching (zero child feature -> check parent recovery) was never performed. Without this, the claim rests on observational cross-L0 persistence classification with known bias. This is also the ONLY experiment that can distinguish 'metric miscalibrated' from 'JumpReLU genuinely has minimal absorption.' Estimated 0.5-1 GPU-hour. (出现 3 次, 权重 2.35)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] CONTROL FAILURE UNDIAGNOSED: Paper identifies central finding (shuffled > measured across all 5 domains, ratios 2.7x to infinity) but never explains WHY. Without mechanistic diagnosis, cannot recommend recalibration, distinguish miscalibration from structural difference, or predict transferability to other metrics/architectures. Threshold sensitivity results already computed (141KB ablation_threshold_sensitivity.json) but not reported. (出现 3 次, 权重 2.35)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] All hierarchy probes achieve AUROC=1.0 exactly (80/80 perfect scores). This means the absorption formula's numerator (acc_resid - acc_sae) is bounded by (1.0 - acc_sae), minimizing absolute absorption scores. The paper mentions this as a limitation but does not address whether the ceiling effect itself invalidates the metric adaptation. (出现 2 次, 权重 1.94)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] The first-letter absorption proxy is degenerate: it returns exactly 0.0 on 26 of 27 GPT-2 checkpoints in E1 and 9 of 10 checkpoints in E3. A metric with near-zero variance cannot support claims about architectural tradeoffs or Pareto dominance. (出现 2 次, 权重 1.58)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] H3 is not merely unsupported—the task-agnostic metric is NEGATIVELY correlated with the first-letter benchmark (r = -0.592, p = 0.12). This suggests the two metrics measure different phenomena, not the same underlying construct. (出现 2 次, 权重 1.58)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] E1's architecture comparison conflates families. The summary table compares only 'Standard' (n=23) vs. 'feature_splitting' (n=4), but the 27-checkpoint corpus includes TopK, TopK_MLP, TopK_Attn, and multiple hook-point variants lumped into 'Standard'. (出现 2 次, 权重 1.58)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] The pilot_summary.md explicitly rates metric quality as NO-GO for publication-ready numbers because the simplified absorption/hedging proxies are too crude and dead-neuron estimates are unreliable at 2k tokens. Yet the main paper presents these exact numbers as primary evidence. (出现 2 次, 权重 1.58)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] The Gemma-2-2B experiment (e1_full_gemma) failed due to gated HuggingFace access. The paper mentions this as a limitation but still frames the work as spanning 'Gemma-2-2B and Pythia-160M' in the abstract, giving readers the impression that modern models were directly evaluated in the controlled Pareto analysis. (出现 2 次, 权重 1.58)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] The E3 pilot uses only 10 checkpoints and one hierarchy domain (geography). The paper acknowledges this but does not pre-register a replication protocol for expanding to 20-50 checkpoints and multiple domains. (出现 2 次, 权重 1.58)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## WRITING 类问题

影响 agent: sequential_writer, section_writer, editor, codex_writer

- [MEDIUM] THEORETICAL OVERCLAIMING: Section 6 title 'CMI Predicts Absorption Susceptibility' -- p=0.059/0.236 does not constitute prediction. Abstract: 'predicts absorption susceptibility.' Introduction: 'first information-theoretic criterion.' Phase transition prediction L0_crit=24.7 vs empirical 22.4 is partly circular (lambda fit from data); non-trivial prediction (rank ordering) has rho=+0.333 (p=0.103), non-significant. Systematic overclaiming in theoretical sections contrasts with excellent honesty in negative results. (出现 3 次, 权重 2.35)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] STRUCTURAL REDUNDANCY: Control failure finding stated 4 times (Introduction para 2, Section 4.1 opening, post-Table-2 summary, Section 4.4 opening). Near-verbatim sentence about 'higher absorption scores to randomized labels' appears in both Introduction and Section 4.1. Section 5.3 (~200 words) presents fully confounded cross-model comparison (JumpReLU on Gemma 2B vs L1 on GPT-2 Small: different architectures, parameters, training data). (出现 3 次, 权重 2.35)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] TITLE MISLEADING: 'Beyond Competitive Exclusion' implies competitive exclusion is disproven. Paper shows the Chanin METRIC conflates hedging with competitive exclusion on JumpReLU SAEs; whether competitive exclusion genuinely exists is left unanswered (pending activation patching on 9 core words). (出现 3 次, 权重 2.35)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] CROSS-DOMAIN NOVELTY SELF-CONTRADICTION: Paper simultaneously claims 'first cross-domain absorption characterization' (Section 4.4 novelty) and admits 'all rates fall below shuffled controls, so absolute rates cannot be interpreted as genuine absorption.' Cannot claim credit for first measurements and admit they are meaningless. (出现 3 次, 权重 2.35)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] Figure 4 is introduced for the first time in the Discussion (Section 7.1) with no forward reference in any prior section. This breaks the paper's own convention and will confuse readers. (出现 2 次, 权重 1.58)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] Tables 1-3 are rendered as inline markdown tables without LaTeX labels or captions. The outline explicitly planned labeled LaTeX tables for NeurIPS submission. (出现 2 次, 权重 1.58)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] Terminology is inconsistent: 'feature-splitting' vs. 'feature_splitting', 'CE recovered' vs. 'CE loss recovered' vs. '$\text{CE}_{\text{recovered}}$', 'JumpReLU' vs. 'JumpRelu'. TopK_MLP and TopK_Attn appear in E3 but are not defined in the glossary. (出现 2 次, 权重 1.58)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。

## 成功模式 (继续保持)

- STRATEGIC PIVOT: Iter_006 executed a major strategic pivot from epidemiological-methods-on-SAEBench (iter 4-5) to JumpReLU metric audit on Gemma 2 2B. This pivot was CORRECT: the universal control failure and hedging decomposition are genuinely novel findings the SAE community needs. Pivoting away from the failing GPT-2 Small cross-domain experiments was the right decision. (出现 3 次)
- COMPREHENSIVE CONTROL SUITE: Four controls (random probe, shuffled labels, dense probe ceiling, untrained SAE) across five domains. Most thorough control suite in any absorption study. Universal failure across all domains is devastating and well-presented. (出现 3 次)
- L0 PHASE TRANSITION (MOST ROBUST FINDING): 42.9% -> 0.8% across four L0 values with cross-layer stability (CV < 10%) and bootstrap CIs. Monotonic, cross-layer stable, directly actionable. No reviews questioned this finding. (出现 3 次)
- CONFOUND DECOMPOSITION CONCEPTUAL INNOVATION: Cross-L0 persistence criterion for classifying hedging vs hierarchy-driven absorption is novel and intuitive, even if the permissive definition needs tightening. The 98.6%/1.4% split and the identification of 9 persistent core words are compelling. (出现 3 次)
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- ZERO EXPERIMENT FAILURES: 23/23 tasks completed successfully in iter_006. Infrastructure fully reliable for 3 consecutive iterations (iter 4: 13/13, iter 5: 14/14, iter 6: 23/23). (出现 3 次)
- EXEMPLARY PER-LETTER TRACKING: Per-letter probe F1, per-letter absorption rate, per-letter CMI estimate. This granularity enables the probe quality confound identification (rho=-0.67), which is itself an important methodological observation. (出现 3 次)
- CROSS-LAYER VALIDATION: L0=82 tested at layers 10, 12, 20 with CV < 10% confirms the L0 phase transition is not layer-specific. Clean experimental design. (出现 3 次)
- E2 meta-analysis uses a large N=314 sample with appropriate statistical methods (partial correlation, OLS with cluster-robust SEs), providing the strongest empirical signal in the paper. (出现 2 次)
- Honest reporting of negative results: E3's unsupported H3 is reported without spin, framed as a valuable negative result that raises questions about benchmark validity. (出现 2 次)

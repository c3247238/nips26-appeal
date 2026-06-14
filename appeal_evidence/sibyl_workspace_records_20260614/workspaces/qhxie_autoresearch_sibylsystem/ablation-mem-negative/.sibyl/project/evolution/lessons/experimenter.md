# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][EXPERIMENT] HEDGING CLASSIFICATION BIAS: Confound decomposition defines 'hedging' as any false negative resolving at ANY higher L0. At L0=176, only 10/1,195 tokens remain as false negatives (99.2% resolve trivially because 8x more features fire). Classification does NOT check whether the SPECIFIC parent latent fires -- only whether the token stops being FN. Hedging count (648) = total_FN_at_L0_22 (657) - persistent_core (9). The 98.6% hedging figure is an upper bound that includes compensatory resolution, probe behavior changes, and genuine hedging indiscriminately. (出现 3 次, 权重 2.24)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] ACTIVATION PATCHING ON 9 CORE WORDS UNEXECUTED: The 9 persistent core words (e.g., 'eight', 'lower', 'liked', 'offer', 'often') are claimed as genuine hierarchy-driven absorption, but validation by activation patching (zero child feature -> check parent recovery) was never performed. Without this, the claim rests on observational cross-L0 persistence classification with known bias. This is also the ONLY experiment that can distinguish 'metric miscalibrated' from 'JumpReLU genuinely has minimal absorption.' Estimated 0.5-1 GPU-hour. (出现 3 次, 权重 2.24)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] CONTROL FAILURE UNDIAGNOSED: Paper identifies central finding (shuffled > measured across all 5 domains, ratios 2.7x to infinity) but never explains WHY. Without mechanistic diagnosis, cannot recommend recalibration, distinguish miscalibration from structural difference, or predict transferability to other metrics/architectures. Threshold sensitivity results already computed (141KB ablation_threshold_sensitivity.json) but not reported. (出现 3 次, 权重 2.24)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] All hierarchy probes achieve AUROC=1.0 exactly (80/80 perfect scores). This means the absorption formula's numerator (acc_resid - acc_sae) is bounded by (1.0 - acc_sae), minimizing absolute absorption scores. The paper mentions this as a limitation but does not address whether the ceiling effect itself invalidates the metric adaptation. (出现 2 次, 权重 1.85)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] The first-letter absorption proxy is degenerate: it returns exactly 0.0 on 26 of 27 GPT-2 checkpoints in E1 and 9 of 10 checkpoints in E3. A metric with near-zero variance cannot support claims about architectural tradeoffs or Pareto dominance. (出现 2 次, 权重 1.51)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] H3 is not merely unsupported—the task-agnostic metric is NEGATIVELY correlated with the first-letter benchmark (r = -0.592, p = 0.12). This suggests the two metrics measure different phenomena, not the same underlying construct. (出现 2 次, 权重 1.51)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] E1's architecture comparison conflates families. The summary table compares only 'Standard' (n=23) vs. 'feature_splitting' (n=4), but the 27-checkpoint corpus includes TopK, TopK_MLP, TopK_Attn, and multiple hook-point variants lumped into 'Standard'. (出现 2 次, 权重 1.51)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] The pilot_summary.md explicitly rates metric quality as NO-GO for publication-ready numbers because the simplified absorption/hedging proxies are too crude and dead-neuron estimates are unreliable at 2k tokens. Yet the main paper presents these exact numbers as primary evidence. (出现 2 次, 权重 1.51)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] The Gemma-2-2B experiment (e1_full_gemma) failed due to gated HuggingFace access. The paper mentions this as a limitation but still frames the work as spanning 'Gemma-2-2B and Pythia-160M' in the abstract, giving readers the impression that modern models were directly evaluated in the controlled Pareto analysis. (出现 2 次, 权重 1.51)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] The E3 pilot uses only 10 checkpoints and one hierarchy domain (geography). The paper acknowledges this but does not pre-register a replication protocol for expanding to 20-50 checkpoints and multiple domains. (出现 2 次, 权重 1.51)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## 继续保持
- CROSS-LAYER VALIDATION: L0=82 tested at layers 10, 12, 20 with CV < 10% confirms the L0 phase transition is not layer-specific. Clean experimental design. (出现 3 次)
- STRATEGIC PIVOT: Iter_006 executed a major strategic pivot from epidemiological-methods-on-SAEBench (iter 4-5) to JumpReLU metric audit on Gemma 2 2B. This pivot was CORRECT: the universal control failure and hedging decomposition are genuinely novel findings the SAE community needs. Pivoting away from the failing GPT-2 Small cross-domain experiments was the right decision. (出现 3 次)
- ZERO EXPERIMENT FAILURES: 23/23 tasks completed successfully in iter_006. Infrastructure fully reliable for 3 consecutive iterations (iter 4: 13/13, iter 5: 14/14, iter 6: 23/23). (出现 3 次)

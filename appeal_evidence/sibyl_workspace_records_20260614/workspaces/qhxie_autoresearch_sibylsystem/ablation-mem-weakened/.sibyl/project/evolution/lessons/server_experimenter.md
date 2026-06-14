# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][EXPERIMENT] HEDGING CLASSIFICATION BIAS: Confound decomposition defines 'hedging' as any false negative resolving at ANY higher L0. At L0=176, only 10/1,195 tokens remain as false negatives (99.2% resolve trivially because 8x more features fire). Classification does NOT check whether the SPECIFIC parent latent fires -- only whether the token stops being FN. Hedging count (648) = total_FN_at_L0_22 (657) - persistent_core (9). The 98.6% hedging figure is an upper bound that includes compensatory resolution, probe behavior changes, and genuine hedging indiscriminately. (出现 3 次, 权重 2.18)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] ACTIVATION PATCHING ON 9 CORE WORDS UNEXECUTED: The 9 persistent core words (e.g., 'eight', 'lower', 'liked', 'offer', 'often') are claimed as genuine hierarchy-driven absorption, but validation by activation patching (zero child feature -> check parent recovery) was never performed. Without this, the claim rests on observational cross-L0 persistence classification with known bias. This is also the ONLY experiment that can distinguish 'metric miscalibrated' from 'JumpReLU genuinely has minimal absorption.' Estimated 0.5-1 GPU-hour. (出现 3 次, 权重 2.18)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] CONTROL FAILURE UNDIAGNOSED: Paper identifies central finding (shuffled > measured across all 5 domains, ratios 2.7x to infinity) but never explains WHY. Without mechanistic diagnosis, cannot recommend recalibration, distinguish miscalibration from structural difference, or predict transferability to other metrics/architectures. Threshold sensitivity results already computed (141KB ablation_threshold_sensitivity.json) but not reported. (出现 3 次, 权重 2.18)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] E1 architecture comparison is massively confounded: JumpReLU is pretrained on Gemma data with d_SAE=24,576 while TopK is trained from scratch on OpenWebText with d_SAE=16,384. The 4x collision rate difference (15.4% vs 3.8%) cannot be attributed to architecture alone---it confounds architecture, training data, dictionary size, and training procedure. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] Dead feature ratio of 89-99% across all trained SAEs indicates catastrophic training failure. With 96-99% dead features, the SAE is effectively operating with only 160-640 active features out of 16,384. This likely invalidates all trained-SAE results. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] The collision rate metric itself is problematic. It measures whether multiple concepts activate the same feature, but this is not the same as Chanin et al.'s absorption (parent suppressing child under co-occurrence). Without parent-child hierarchy testing, calling this 'absorption' is misleading. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] E2 (causal impact) uses only 5 k-values as the independent variable, but these 5 points are not independent---they come from the same training run with different k. The 'causal' claim is weak because k directly controls both sparsity and reconstruction quality, creating a structural confound. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] UAD and DFDA are severely underpowered proof-of-concepts framed as contributions. UAD validated on only 25 first-letter collisions. DFDA evaluated on only 4 pairs, all sharing the same feature (18486), with one pair showing degradation (-21.4%). (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] Single-seed experiments (seed=42) with no replication. All correlation p-values hover around 0.87, suspiciously uniform. The pilot showed 30.8% collision for TopK while the full experiment shows 3.8%---an 8x discrepancy never explained. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] The core novelty claim is weakened by the confounded comparison. If the 4x difference is driven by training data/width rather than architecture, the 'cross-architecture benchmark' contribution evaporates. The actual contribution may be limited to a narrow negative result. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## 继续保持
- CROSS-LAYER VALIDATION: L0=82 tested at layers 10, 12, 20 with CV < 10% confirms the L0 phase transition is not layer-specific. Clean experimental design. (出现 3 次)
- STRATEGIC PIVOT: Iter_006 executed a major strategic pivot from epidemiological-methods-on-SAEBench (iter 4-5) to JumpReLU metric audit on Gemma 2 2B. This pivot was CORRECT: the universal control failure and hedging decomposition are genuinely novel findings the SAE community needs. Pivoting away from the failing GPT-2 Small cross-domain experiments was the right decision. (出现 3 次)
- ZERO EXPERIMENT FAILURES: 23/23 tasks completed successfully in iter_006. Infrastructure fully reliable for 3 consecutive iterations (iter 4: 13/13, iter 5: 14/14, iter 6: 23/23). (出现 3 次)

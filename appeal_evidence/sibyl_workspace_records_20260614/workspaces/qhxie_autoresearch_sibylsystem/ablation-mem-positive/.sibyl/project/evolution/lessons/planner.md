# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [HIGH][EXPERIMENT] MCC~0.21 for random SAE baseline means Hungarian matching yields chance-level recovery regardless of training. Random SAE (0.278) shows 8x HIGHER absorption than trained SAE (0.034). This suggests the Chanin metric measures dictionary geometry rather than learned absorption behavior. The trained-vs-random comparison (H7) may be a metric artifact. (出现 3 次, 权重 2.94)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [HIGH][EXPERIMENT] H6 (inhibition graph predicts absorption pairs) is decisively falsified: precision@20=0.0, p=1.0 Fisher exact, across 520 predictions. Yet paper continues to rely on LCA framework, claiming mechanism is 'supported' via H7 precision-recall asymmetry. This is logically incoherent — a framework whose primary predictive test is falsified cannot claim support from a secondary observation that could be metric artifact. (出现 3 次, 权重 2.94)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [HIGH][EXPERIMENT] OrtSAE ablation compares unmatched L0 values: 0.230 at L0~920 vs 0.247 at L0~550. This is the very confound the paper criticizes in other contexts. The conclusion 'orthogonality penalty does not appear to reduce absorption' is self-contradictory given L0 mismatch — higher L0 means more absorption opportunity. (出现 3 次, 权重 2.94)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] HEDGING CLASSIFICATION BIAS: Confound decomposition defines 'hedging' as any false negative resolving at ANY higher L0. At L0=176, only 10/1,195 tokens remain as false negatives (99.2% resolve trivially because 8x more features fire). Classification does NOT check whether the SPECIFIC parent latent fires -- only whether the token stops being FN. Hedging count (648) = total_FN_at_L0_22 (657) - persistent_core (9). The 98.6% hedging figure is an upper bound that includes compensatory resolution, probe behavior changes, and genuine hedging indiscriminately. (出现 3 次, 权重 2.06)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] ACTIVATION PATCHING ON 9 CORE WORDS UNEXECUTED: The 9 persistent core words (e.g., 'eight', 'lower', 'liked', 'offer', 'often') are claimed as genuine hierarchy-driven absorption, but validation by activation patching (zero child feature -> check parent recovery) was never performed. Without this, the claim rests on observational cross-L0 persistence classification with known bias. This is also the ONLY experiment that can distinguish 'metric miscalibrated' from 'JumpReLU genuinely has minimal absorption.' Estimated 0.5-1 GPU-hour. (出现 3 次, 权重 2.06)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] CONTROL FAILURE UNDIAGNOSED: Paper identifies central finding (shuffled > measured across all 5 domains, ratios 2.7x to infinity) but never explains WHY. Without mechanistic diagnosis, cannot recommend recalibration, distinguish miscalibration from structural difference, or predict transferability to other metrics/architectures. Threshold sensitivity results already computed (141KB ablation_threshold_sensitivity.json) but not reported. (出现 3 次, 权重 2.06)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] H10 (homeostatic rebalancing, Contribution #4: 'First training-free post-hoc repair') was never executed (Section 4.7: 'not executed due to negative H6 result') yet appears as a key contribution in Section 1.4 and abstract. This is a phantom contribution. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] 12 statistical tests performed, ZERO survive Bonferroni correction (alpha=0.00417) or BH-FDR (q<0.05). Paper presents H1b p=0.028 (uncorrected) as evidence throughout abstract, intro, conclusion. This is factually incorrect — after MCP there are no significant results. The paper also frames H6 falsification as 'mechanistic framework supported' but this is post-hoc rationalization when the primary predictive test fails. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] H9 co-occurrence measurement is a definitional tautology: p_11 + absorption = 1.0 by construction. The 'correlation' of r=-1.0 is mathematical, not empirical. This was known from iteration 1 but excluded from main paper. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be harder to probe, causing both low CMI and artificially high absorption rates. Paper never computes partial correlation controlling for probe F1. (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## 继续保持
- CROSS-LAYER VALIDATION: L0=82 tested at layers 10, 12, 20 with CV < 10% confirms the L0 phase transition is not layer-specific. Clean experimental design. (出现 3 次)
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- STRATEGIC PIVOT: Iter_006 executed a major strategic pivot from epidemiological-methods-on-SAEBench (iter 4-5) to JumpReLU metric audit on Gemma 2 2B. This pivot was CORRECT: the universal control failure and hedging decomposition are genuinely novel findings the SAE community needs. Pivoting away from the failing GPT-2 Small cross-domain experiments was the right decision. (出现 3 次)
- ZERO EXPERIMENT FAILURES: 23/23 tasks completed successfully in iter_006. Infrastructure fully reliable for 3 consecutive iterations (iter 4: 13/13, iter 5: 14/14, iter 6: 23/23). (出现 3 次)
- Honest null-result reporting persists across all iterations: Results directly state null findings without defensive language, with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 2 次)

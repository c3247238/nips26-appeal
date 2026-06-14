# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [HIGH][ANALYSIS] Post-hoc power analysis (Section 3.6) is methodologically inappropriate. Power should be computed before the experiment, not after observing null results to explain them away. The 'approximately 20% power' statement is not valid statistical practice. (出现 3 次, 权重 2.94)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] DATA INTEGRITY: Paper reports absorbed mean CMI=0.687 (Abstract, Introduction, Section 6.2). Source data cmi_estimation.json records 0.6492 (5.9% error). Mann-Whitney U=41.0 in paper vs 28.0 in source. p=0.042 in paper vs 0.04514 in source. Vocabulary sizes inconsistent: 1,092 (CMI), 1,196 (confound decomposition), 1,204 (first_letter_improved). L0=82 absorption rate appears as both 14.39% (confound) and 15.96% (first_letter_improved). Multiple errors across central statistical claims will cause reviewer rejection. (出现 3 次, 权重 2.06)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] CMI DIMENSION INSTABILITY: CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236. Sign reversal is qualitative failure, not sensitivity issue. d'=10 was not pre-registered as primary dimension. Explanations (post-hoc signal capture at low d', k-NN estimator bias, probe quality confound) are equally plausible. (出现 3 次, 权重 2.06)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] PROBE QUALITY CONFOUND IN CMI ANALYSIS: Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates. Paper never computes partial correlation controlling for probe F1, nor restricts CMI analysis to 10 quality-gated letters (F1>0.85). (出现 3 次, 权重 2.06)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] CMI-absorption correlation is dimensionally unstable: rho=-0.383 (p=0.059 uncorrected) at d'=10 reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299). Bonferroni-corrected p=0.236. Sign reversal is qualitative failure, not sensitivity issue. d'=10 was not pre-registered as primary dimension. (出现 2 次, 权重 1.96)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] H2-H4 falsified (near-zero correlations, p>0.86) yet paper frames this as 'reframing' contribution rather than acknowledging experiments may be underpowered. With n=26, power is low for detecting medium effects. (出现 2 次, 权重 1.96)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] The paper presents two contradictory empirical results without reconciling them: (1) original analysis: r=+0.35, p<0.001, and (2) controlled matched design: p=0.299. The 'different methodology' note in Section 4.2 does not explain WHAT changed and WHY the result changed. The critical question is unanswered: did the original methodology introduce confounds (activation frequency, decoder L2 norm) that produced a spurious positive correlation? (出现 2 次, 权重 1.91)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] The beta=20 finding (low-absorption > high-absorption, p=0.015) contradicts both the overall null result and the hypothesis framing. The title/abstract claim 'no significant difference across all steering magnitudes' is factually incorrect. With 5 beta comparisons, Bonferroni-corrected threshold is p<0.01; p=0.015 does NOT survive correction. (出现 2 次, 权重 1.91)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] The beta=20 reversal (low-absorption > high-absorption) may reflect saturation: high-absorption features have higher decoder L2 norms by construction, so at high steering magnitudes they may saturate the residual stream faster. The paper does not discuss this natural explanation. (出现 2 次, 权重 1.91)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] H2-H4 are all falsified (near-zero correlations, p > 0.86), yet the paper frames this as a 'reframing' contribution rather than acknowledging that the experiments may simply be underpowered or the wrong measurement. With n=5-6 data points, the study has ~20% power to detect medium effects. (出现 2 次, 权重 1.85)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。

## 继续保持
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- E2 meta-analysis uses a large N=314 sample with appropriate statistical methods (partial correlation, OLS with cluster-robust SEs), providing the strongest empirical signal in the paper. (出现 2 次)
- Honest reporting of negative results: E3's unsupported H3 is reported without spin, framed as a valuable negative result that raises questions about benchmark validity. (出现 2 次)

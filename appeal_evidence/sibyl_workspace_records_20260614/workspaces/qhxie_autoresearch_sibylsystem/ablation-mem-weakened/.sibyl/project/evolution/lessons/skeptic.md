# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][ANALYSIS] DATA INTEGRITY: Paper reports absorbed mean CMI=0.687 (Abstract, Introduction, Section 6.2). Source data cmi_estimation.json records 0.6492 (5.9% error). Mann-Whitney U=41.0 in paper vs 28.0 in source. p=0.042 in paper vs 0.04514 in source. Vocabulary sizes inconsistent: 1,092 (CMI), 1,196 (confound decomposition), 1,204 (first_letter_improved). L0=82 absorption rate appears as both 14.39% (confound) and 15.96% (first_letter_improved). Multiple errors across central statistical claims will cause reviewer rejection. (出现 3 次, 权重 2.18)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] CMI DIMENSION INSTABILITY: CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236. Sign reversal is qualitative failure, not sensitivity issue. d'=10 was not pre-registered as primary dimension. Explanations (post-hoc signal capture at low d', k-NN estimator bias, probe quality confound) are equally plausible. (出现 3 次, 权重 2.18)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] PROBE QUALITY CONFOUND IN CMI ANALYSIS: Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates. Paper never computes partial correlation controlling for probe F1, nor restricts CMI analysis to 10 quality-gated letters (F1>0.85). (出现 3 次, 权重 2.18)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] H2-H4 are all falsified (near-zero correlations, p > 0.86), yet the paper frames this as a 'reframing' contribution rather than acknowledging that the experiments may simply be underpowered or the wrong measurement. With n=5-6 data points, the study has ~20% power to detect medium effects. (出现 2 次, 权重 1.96)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] Section 3.6 claims 'approximately 20% power to detect a medium effect size (r = 0.5).' This is a post-hoc power analysis, which is methodologically questionable---power should be computed before the experiment. (出现 2 次, 权重 1.96)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] Multiple comparisons problem: H1b at layer 8 (Pearson p=0.028, Spearman p=0.009) does NOT survive Bonferroni correction (alpha=0.0042 for 12 tests) or Benjamini-Hochberg FDR (q=0.05 rejects all 12 hypotheses). The paper has ZERO statistically significant results after proper correction. The abstract, introduction, and conclusion all present H1b as evidence of a real effect, which is incorrect. (出现 2 次, 权重 1.96)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] H3 (cross-layer consistency) tested with only two layers (4 and 8). With n=2 slope pairs, the CV has no statistical meaning. Layer 0 and 10 absorption data were collected but never used. (出现 2 次, 权重 1.96)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] H3 (cross-layer consistency) is tested with only two layers (4 and 8), making the 'inconsistency' claim statistically weak. With only two data points for slope comparison, opposite signs are sufficient to reject consistency but the CV computation (CV=0.932 for H2) is based on just two slopes. The paper presents this as strong evidence of inconsistency, but with n=2 slope pairs, the CV has no statistical meaning. (出现 2 次, 权重 1.95)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] The paper uses causal language ('causal cost,' 'unique causal effect') for E2, which is an observational meta-analysis. The section titles and abstract repeatedly use causal framing that overreaches the design. (出现 2 次, 权重 1.46)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。

## 继续保持
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- E2 meta-analysis uses a large N=314 sample with appropriate statistical methods (partial correlation, OLS with cluster-robust SEs), providing the strongest empirical signal in the paper. (出现 2 次)
- Honest reporting of negative results: E3's unsupported H3 is reported without spin, framed as a valuable negative result that raises questions about benchmark validity. (出现 2 次)

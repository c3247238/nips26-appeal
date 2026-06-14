# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][ANALYSIS] DATA INTEGRITY: Paper reports absorbed mean CMI=0.687 (Abstract, Introduction, Section 6.2). Source data cmi_estimation.json records 0.6492 (5.9% error). Mann-Whitney U=41.0 in paper vs 28.0 in source. p=0.042 in paper vs 0.04514 in source. Vocabulary sizes inconsistent: 1,092 (CMI), 1,196 (confound decomposition), 1,204 (first_letter_improved). L0=82 absorption rate appears as both 14.39% (confound) and 15.96% (first_letter_improved). Multiple errors across central statistical claims will cause reviewer rejection. (出现 3 次, 权重 2.29)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] CMI DIMENSION INSTABILITY: CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236. Sign reversal is qualitative failure, not sensitivity issue. d'=10 was not pre-registered as primary dimension. Explanations (post-hoc signal capture at low d', k-NN estimator bias, probe quality confound) are equally plausible. (出现 3 次, 权重 2.29)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] PROBE QUALITY CONFOUND IN CMI ANALYSIS: Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates. Paper never computes partial correlation controlling for probe F1, nor restricts CMI analysis to 10 quality-gated letters (F1>0.85). (出现 3 次, 权重 2.29)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] The paper uses causal language ('causal cost,' 'unique causal effect') for E2, which is an observational meta-analysis. The section titles and abstract repeatedly use causal framing that overreaches the design. (出现 2 次, 权重 1.54)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。

## 继续保持
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- E2 meta-analysis uses a large N=314 sample with appropriate statistical methods (partial correlation, OLS with cluster-robust SEs), providing the strongest empirical signal in the paper. (出现 2 次)
- Honest reporting of negative results: E3's unsupported H3 is reported without spin, framed as a valuable negative result that raises questions about benchmark validity. (出现 2 次)

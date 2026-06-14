# Section Critique: Experiments (Sections 4 & 5)
**Paper:** Unified Dynamic Weight Decay Framework (iter_003)
**Reviewer:** Section Critic
**Date:** 2026-03-18

---

## Criterion Scores

| Criterion | Score (1-10) | Brief Rationale |
|---|---|---|
| 1. Experimental setup completeness | 8 | Dataset, architecture, optimizer, seeds, total runs all documented; missing: warmup ablation, batch size sensitivity, CIFAR-100 diagnostic table |
| 2. Results presentation | 7 | Main table and statistical tests are solid; missing: individual seed runs, confidence intervals as explicit bounds, CIFAR-100 diagnostic metrics table |
| 3. Data accuracy | 9 | All reported numbers verified against full_summary.json; one minor rounding discrepancy noted |
| 4. Analysis depth | 7 | Weight norm mechanistic explanation is excellent; CSI/AIS interpretation is under-developed; no correlation coefficients reported for weight norm vs. accuracy |
| 5. Fair comparison methodology | 8 | Hyperparameter fairness protocol is clearly stated and principled; BEM framework explicitly disentangles budget effects; acknowledged limitation of no per-method grid search |
| 6. Reproducibility details | 6 | Seeds and RNG control described; missing: exact PyTorch version, CUDA version, hardware spec, wall-clock time, codebase URL/hash |

**Overall Score: 7.5 / 10**

---

## Top 3 Strengths

**1. Rigorous statistical treatment with explicit null-hypothesis framing.**
The inclusion of paired t-tests, Bonferroni correction, and Cohen's d effect sizes (Section 5.1) is exemplary for a paper whose central claim is a null result. Most ML papers omit Bonferroni correction entirely. Reporting both uncorrected and Bonferroni thresholds ($p < 0.05$ vs. $p < 0.0083$) makes the strength of the null finding unambiguous.

**2. Mechanistic weight norm analysis that explains the null result (Section 5.4).**
Rather than simply reporting "no significant difference," the paper provides a causal account: AdamW's adaptive per-parameter scaling creates an implicit weight norm control mechanism that dominates explicit weight decay modulation. The quantitative support (final norms spanning only 95.89--97.04, a 1.2% range despite 10x BEM variation) is precise and directly interpretable. This elevates the paper from descriptive to explanatory.

**3. Budget Equivalence Metric (BEM) as a methodological contribution.**
The BEM framework (Section 3.3, applied in Section 5.2) explicitly disentangles "how much total weight decay" from "how weight decay is distributed." This is a genuine methodological contribution that prevents the confound where a method appears better simply because it applies less total regularization. Figure 3's BEM-vs-accuracy analysis cleanly demonstrates that even a 10x budget variation yields less than 0.5% accuracy change, which is a strong and well-supported empirical claim.

---

## Top 5 Specific Issues

### Issue 1: CIFAR-100 Diagnostic Metrics Table Is Missing

**Severity: High**

Table 4 (Section 5.3) reports CSI, AIS, and BEM values only for CIFAR-10. The analysis.md ground truth contains full CIFAR-100 diagnostic metrics (e.g., cosine_schedule CSI=0.868, constant CSI=0.864, no_wd AIS=0.280). The text in Section 5.3 describes CIFAR-100 CSI and AIS behavior ("On CIFAR-100, the CSI range is narrower (0.854--0.868)") but the supporting numbers are not presented in a table. A reader cannot verify these claims without the corresponding table. This omission is particularly problematic because the CIFAR-100 diagnostic pattern (tighter CSI clustering) is presented as a distinct finding.

**Fix:** Add Table 4b with CIFAR-100 CSI, AIS, BEM, and weight norm values mirroring Table 4 for CIFAR-10.

---

### Issue 2: BEM Inconsistency for `half_lambda` -- Conceptual Error Requiring Correction

**Severity: High**

Table 4 reports `half_lambda` BEM = 0.000. This is incorrect on conceptual grounds. `half_lambda` applies $\lambda/2 = 2.5 \times 10^{-4}$ uniformly, which is half the constant baseline's budget. By the BEM formula:
$$\mathrm{BEM}(\text{half\_lambda}) = \frac{|\lambda_{\mathrm{eff}}^{\text{half\_lambda}} - \lambda_{\mathrm{eff}}^{\text{constant}}|}{\lambda_{\mathrm{eff}}^{\text{constant}}} = \frac{|0.5\lambda - \lambda|}{\lambda} = 0.5$$

The text in Section 5.2 correctly states "BEM values span the full range from 0.0 (constant, half\_lambda)..." -- but grouping `constant` and `half\_lambda` at BEM=0.0 is wrong. BEM=0.0 means the effective budget matches the constant baseline exactly. Half-lambda uses half the budget and should have BEM=0.5. The analysis.md correctly shows `half_lambda` BEM=0.000 for CIFAR-10 in the data table, which may reflect a computational bug in the BEM calculation rather than a conceptual error in the paper -- but the reported value is inconsistent with the BEM definition given in Section 3.4. This needs to be audited and corrected.

**Fix:** Recalculate BEM for `half_lambda` (expected ~0.5) and verify the BEM computation code. Update Table 4 and all references in Section 5.2 accordingly.

---

### Issue 3: Per-Seed Accuracy Data Not Reported; Prevents Independent Verification of Statistics

**Severity: Medium**

The paper reports mean and standard deviation over 3 seeds but does not show the individual per-seed accuracy values anywhere in the main text or an appendix table. The ground truth `full_summary.json` contains per-seed runs (e.g., CIFAR-10 constant: [90.48, 90.03, 89.89]), but these are absent from the paper. With only 3 samples, a paired t-test's validity depends critically on the individual values, and readers cannot verify the reported p-values without them. The Appendix B mentioned in Section 4.4 ("Full diagnostic panels for all 42 runs") does not appear to exist in the current draft.

**Fix:** Add a supplementary table in Appendix B with per-seed accuracy for all 42 runs. At minimum, add footnotes or a compact table in the main text showing the three individual runs for each method-dataset pair.

---

### Issue 4: Statistical Power Is Critically Low -- Must Be Explicitly Acknowledged

**Severity: Medium**

The paper uses n=3 seeds for statistical tests. A paired t-test with n=3 has 2 degrees of freedom, which provides very low power. For example, to detect a true mean difference of 0.25% with the observed within-method standard deviation of ~0.30%, the required sample size is approximately n=15 (at 80% power, two-tailed, $\alpha=0.05$). The current n=3 design cannot distinguish "no effect" from "underpowered to detect a small effect." The paper's conclusion that "no method significantly outperforms the constant baseline" is accurate, but the conclusion that this constitutes strong evidence for the null hypothesis is overstated without a power analysis. Section 5.1 mentions Cohen's d < 0.3 but does not compute statistical power or the minimum detectable effect size at 80% power for n=3.

**Fix:** Add a power analysis to Section 5.1 or Appendix A. State explicitly: "With n=3 seeds, our tests have 80% power to detect differences of X% or larger; the observed differences of <0.25% are below this threshold." This actually strengthens the null finding by showing the effect, if any, must be very small.

---

### Issue 5: Weight Norm Analysis Lacks Quantitative Correlation and CIFAR-100 Data

**Severity: Medium**

Section 5.4 describes weight norm convergence for CIFAR-10 (norms 95.89--97.04) and argues this explains accuracy equivalence, but:

(a) No correlation coefficient is reported between weight norm deviation and accuracy rank. The claim "similar weight norms explain similar accuracy" is stated but not demonstrated quantitatively. A Spearman $\rho$ between final weight norm and accuracy across the 7 methods would directly support or challenge this claim.

(b) The CIFAR-100 weight norm values from analysis.md (ranging 104.72--106.03) are not reported in the text or any table, despite the CIFAR-100 analysis being presented in parallel throughout Sections 5.1--5.3.

(c) The text refers to "Figure 5" for weight norm trajectories, but this figure is not shown in the current draft excerpt. It is unclear whether per-layer or aggregate norms are plotted, and whether the plot is per-seed or averaged.

**Fix:** (a) Report Spearman $\rho$(final weight norm, test accuracy) for both datasets. (b) Add CIFAR-100 weight norm values to Section 5.4 or a table. (c) Ensure Figure 5 exists and that its caption clarifies layer aggregation and seed averaging.

---

## Data Accuracy Verification

All main accuracy numbers in Table 2 were cross-checked against `full_summary.json`:

| Method | CIFAR-10 paper | CIFAR-10 JSON | CIFAR-100 paper | CIFAR-100 JSON | Match? |
|---|---|---|---|---|---|
| constant | 90.13 ± 0.31 | 90.13 ± 0.31 | 63.15 ± 0.30 | 63.15 ± 0.30 | PASS |
| cosine_schedule | 90.12 ± 0.07 | 90.12 ± 0.07 | 63.42 ± 0.42 | 63.42 ± 0.42 | PASS |
| random_mask | 90.12 ± 0.30 | 90.12 ± 0.30 | 62.87 ± 0.38 | 62.87 ± 0.38 | PASS |
| half_lambda | 90.09 ± 0.29 | 90.09 ± 0.29 | 62.91 ± 0.47 | 62.91 ± 0.47 | PASS |
| no_wd | 90.08 ± 0.32 | 90.08 ± 0.32 | 62.66 ± 0.38 | 62.66 ± 0.38 | PASS |
| cwd_hard | 90.06 ± 0.24 | 90.06 ± 0.24 | 62.84 ± 0.30 | 62.84 ± 0.30 | PASS |
| swd | 89.88 ± 0.25 | 89.88 ± 0.25 | 63.06 ± 0.29 | 63.06 ± 0.29 | PASS |

All 14 mean/std pairs match exactly. The statistical test deltas (Table 3) were also verified to be arithmetic differences of the above means. **No data mismatches found in reported accuracy values.**

One flag: `analysis.md` lists `random_mask` CIFAR-100 BEM as 0.501, while Table 4 (CIFAR-10) shows 0.500. This sub-0.1% discrepancy is negligible but confirms the BEM is computed empirically from actual modulation values rather than analytically. The `half_lambda` BEM=0.000 issue described in Issue 2 above is the more substantive concern.

---

## Suggestions for Additional Experiments or Analyses

**1. Stronger baselines: larger models and datasets.**
ResNet-20 on CIFAR-10/100 is a well-studied but arguably easy setting where regularization effects are known to be minor. The core claim ("dynamic weight decay provides no benefit under AdamW") would be much more convincing if replicated on at least one larger-scale setting: ResNet-50 on ImageNet, or a small transformer (GPT-2 small / ViT-S) on a language or vision task. Even a single additional architecture-dataset pair would substantially increase the generalizability of the conclusions.

**2. Varied base learning rate and weight decay coefficient.**
The hyperparameter fairness protocol fixes $\eta = 10^{-3}$ and $\lambda = 5 \times 10^{-4}$. The implicit norm control explanation predicts that the null result should hold across a range of $\lambda$ values -- but this prediction is untested. A 2x2 grid ($\lambda \in \{10^{-4}, 5 \times 10^{-4}, 10^{-3}\}$ crossed with one other hyperparameter) would test whether the null result is specific to the chosen hyperparameter region or is truly robust.

**3. Training curve analysis (not just final accuracy).**
The paper reports final test accuracy but does not show training curves. Cosine_schedule's remarkably low variance ($\sigma = 0.07\%$ vs. $\sim 0.30\%$ for all others) on CIFAR-10 hints at a stability benefit that may manifest earlier in training or in faster convergence, even if final accuracy is identical. Plotting training loss and test accuracy curves (mean $\pm$ std across seeds) would make this stability finding more concrete and potentially more impactful than the accuracy comparison alone.

**4. Quantitative AIS-to-accuracy regression.**
The claim "AIS is an intrinsic property of the network, not of the WD method" would be more rigorous if supported by a regression or permutation test. Specifically: if AIS is method-independent, then the variance of AIS across seeds for a fixed method should be comparable to the variance across methods for a fixed seed. Reporting within-method vs. across-method AIS variance would quantitatively support the intrinsic-property claim.

**5. Post-hoc budget-normalized rerun for `half_lambda`.**
Given the BEM inconsistency (Issue 2), a targeted experiment running `constant` at $\lambda = 5 \times 10^{-4}$ vs. a budget-normalized `half_lambda` at $\lambda = 5 \times 10^{-4}$ with $p=0.5$ Bernoulli mask (to achieve true BEM $\approx 0.5$ at identical base $\lambda$) would cleanly separate budget effects from modulation effects and resolve the current ambiguity in the `half_lambda` control.

---

*End of critique.*

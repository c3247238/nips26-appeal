# Skeptic Analysis: Iteration 5 Results

**Role**: Skeptical statistician. Maximum scrutiny on significance, confounds, proxy metric gaming, and alternative explanations.

---

## 1. Statistical Risk Inventory (Top 3)

### Risk 1: Within-Width Matching Null Destroys the Causal Narrative

The Rosenbaum analysis reveals a devastating split: cross-width matching (Mahalanobis, 17 pairs) finds Gamma=2.65, but **all three within-width strategies fail** (median split: 23 pairs, p>0.12; tertile: 16 pairs, p>0.23; exact width + NN L0: 4 pairs, insufficient). The within-width strategies have MORE pairs than propensity score matching (6 pairs, which does find significance), so the null is not simply a power issue for the median-split case.

**Why this is unreliable**: The entire causal chain claim (L0 -> Absorption -> Quality) rests on absorption having an independent effect beyond width. Yet when you hold width constant and compare high-vs-low absorption SAEs, there is zero detectable quality difference. The Mahalanobis Gamma=2.65 is inflated because those 17 pairs are NOT width-balanced -- they compare across width strata, which is exactly the confound the analysis was supposed to eliminate.

**Severity**: This is the single most important result in the paper and it directly contradicts the headline claim. The summary labels H1 as "SUPPORTED_WITH_CAVEATS" but the caveats are load-bearing.

### Risk 2: Proportion Mediated for Sparse Probing is 4.785 (Mathematically Incoherent)

The mediation analysis for sparse probing F1 reports proportion_mediated = 4.785 with a 95% CI of [-4.15, 4.83]. A proportion mediated exceeding 1.0 is a well-known red flag in mediation analysis (MacKinnon et al., 2007). It occurs when the indirect effect (a*b) and total effect (c) have opposite signs or when the total effect is near zero.

**Specific numbers**: The total effect of L0 on sparse probing F1 is c = 0.00319 (p=0.448, non-significant). The direct effect is c' = -0.01207 (p=0.001). The indirect effect is a*b = 0.01526. The proportion mediated = indirect/total = 0.01526/0.00319 = 4.785.

**Why this is unreliable**: Baron & Kenny Step 3 fails (total effect is non-significant, p=0.448). The mediation framework requires a significant total effect to be interpretable. Without it, the indirect/total ratio is undefined noise. The synthesis summary reports "proportion mediated = 4.785" without flagging that this number is meaningless. The summary also incorrectly claims this metric shows "mediation_type: none" in one place and counts it as having "significant indirect effect" in another -- these are inconsistent.

### Risk 3: N=48 SAEs with Massive Multiple Testing

Phase 1 runs partial correlations, width-stratified correlations (3 strata x 4 metrics = 12 tests), mediation analysis (4 metrics x 4 tests each = 16 tests), five matching strategies x 4 metrics = 20 tests, SCR suppression diagnosis, clustered regression, hurdle model, beta regression, and Bradford Hill scoring. This is conservatively 50+ statistical tests on 48 observations.

**Specific concern**: No multiple comparison correction is applied anywhere. The TPP partial correlation (r=-0.331, p=0.022) would not survive Bonferroni correction over even 4 tests (threshold 0.0125). The Sobel test for TPP (z=2.08, p=0.037) is marginal even uncorrected. SCR's mediation Sobel p=0.00029 survives, but this is only 1 metric out of 4.

---

## 2. Alternative Explanations

### For H1 (Absorption-Quality Causal Chain)

**Alternative 1: Width is the true driver, absorption is epiphenomenal.**
The 1M width stratum has absorption range [0.072, 0.896] while 16k has [0.004, 0.352]. The partial correlation analysis claims to "control for width" but with only 3 discrete width values (16k, 65k, 1M), log(width) is nearly a 3-level factor, not a continuous variable. A three-level covariate in a linear model can fail to fully adjust for nonlinear width effects. The within-width matching null (23 pairs, all non-significant) is consistent with width being the true driver and absorption being a downstream consequence of width that has no independent causal effect on quality.

**Alternative 2: Layer confounds the suppression effect.**
The pilot summary notes that "layer_only" is the primary suppressor for SCR (bivariate r=-0.449 becomes partial r=-0.570 after layer control). The "suppression effect" for sparse probing (r strengthens from -0.664 to -0.746 with L0) may similarly reflect an algebraic artifact of covariate adjustment rather than a genuine unmasking of a hidden effect. Suppression variables are notoriously fragile to model specification in small samples.

### For H2 (Cross-Domain Absorption)

**Alternative 1: The dominance-based metric measures SAE sparsity, not absorption.**
The 100% shuffled control rate proves this beyond doubt. When random label permutations produce the same "absorption" rate as real labels, the metric is not detecting a label-dependent phenomenon. The paper itself acknowledges this. But then claiming the result as "PARTIALLY_SUPPORTED" is misleading -- the honest verdict is that H2 is untested because the metric is invalid for this domain.

**Alternative 2: GPT-2 Small lacks the knowledge representations needed.**
With 98% dead SAE features, the SAE barely functions as a dictionary. Claiming anything about "absorption of knowledge features" in this setting is like measuring absorption in white noise. The model choice (GPT-2 Small instead of the originally proposed Gemma 2 2B) fundamentally undermines the contribution.

### For H3 (Scaling Surface)

**Alternative 1: Architecture confound in the 420-SAE dataset.**
The dataset contains 360 "standard" SAEs, 54 "jumprelu" SAEs, and 6 "unknown." The GAM does not include architecture as a covariate. If jumprelu SAEs cluster in specific width/L0 regions (which they do -- the 54 gemma-scope-2b-pt-res SAEs are the Gemma Scope canonical set with specific width/L0 combinations), the interaction term could reflect architecture differences, not a genuine width-L0 interaction on absorption.

**Alternative 2: The absorption metric itself is width-dependent by construction.**
Wider SAEs have more features competing for activation budget, mechanically increasing the probability that a "more specific" feature dominates. This is not "absorption" in any meaningful sense -- it is a statistical property of overcomplete dictionaries. The interaction term may reflect the metric's sensitivity to dictionary size rather than a substantive property of how SAEs represent hierarchical features.

---

## 3. Proxy Metric Audit

### Metric: Partial correlation (r=-0.746 for sparse probing)
**What we measure**: Linear association between absorption rate and sparse probing F1 score, after partialling out log(width) and layer.
**What we claim**: "Absorption causally mediates the effect of L0 on downstream SAE quality."
**Gap**: Partial correlation measures association, not causation. The word "causal" in the hypothesis requires experimental manipulation (intervene on absorption, observe quality change) or a valid instrumental variable. Mediation analysis assumes correct causal ordering and no unmeasured confounders. Both assumptions are unverifiable with observational data from 48 SAEs.

### Metric: Rosenbaum Gamma (2.65)
**What we measure**: The magnitude of hidden bias needed to explain away the matched-pair quality difference.
**What we claim**: "Strong robustness to unmeasured confounders."
**Gap**: Gamma=2.65 is computed on Mahalanobis-matched pairs that are NOT width-balanced. An unmeasured confounder perfectly correlated with width (e.g., training dynamics at 1M scale) would produce exactly this pattern. The within-width Gamma=1.00 (non-significant) already demonstrates that width IS such a confounder. Reporting Gamma=2.65 without prominently noting that within-width Gamma=1.00 is cherry-picking the favorable number.

### Metric: GAM interaction p-value (3.11e-15)
**What we measure**: Whether a tensor product smooth of log(width) and log(L0) improves GAM fit over additive smooths.
**What we claim**: "Absorption cannot be predicted from width or L0 independently" (scaling surface has nonlinear interaction structure).
**Gap**: With N=420 and a flexible nonparametric model, detecting a significant interaction is almost guaranteed. The relevant question is effect size, not significance. The R-squared improvement from additive (0.620) to interaction (0.693) is 7.3 percentage points -- meaningful but modest. The paper should report this delta, not the p-value. More importantly, the "phase boundary" claim is undermined by `phase_boundary_detected: false` in the final_results.json while the P3 JSON says `phase_boundary_detected: true` -- these are contradictory.

### Metric: Dominance-based absorption rate (51-85% on knowledge hierarchies)
**What we measure**: Fraction of probe-misclassified tokens where the top SAE feature is "dominant" (activation magnitude exceeds a threshold relative to others).
**What we claim**: "Cross-domain absorption exists."
**Gap**: The 100% shuffled control rate proves the metric does not measure absorption. This metric measures the background rate of SAE feature dominance, which is a property of the SAE's activation distribution, not of the relationship between SAE features and probe directions. The claim "partially supported" is therefore unfounded.

### Metric: Taxonomy correction (92.3% -> 92.3%)
**What we measure**: Whether magnitude-ratio-based Type II classification changes with corrected baselines.
**What we claim**: "High Type II rate reflects genuine feature selectivity, not artifact."
**Gap**: The corrected taxonomy using Strategy A (non-letter context comparison) drops to **19.2%**, not 92.3%. The final_results_summary.md reports "Corrected rate: 92.3% (delta = 0.0%)" but the P4_taxonomy_correction.json shows corrected_comprehensive_rate = 0.192. This is a CRITICAL DATA REPORTING ERROR. The summary is using the wrong number. The Chanin-based absorption rate (73.1%) provides an independent estimate that is consistent with neither 92.3% nor 19.2%. The paper cannot claim the taxonomy correction is "minimal" when the actual corrected rate is 19.2%.

---

## 4. Severity Classification

### Fatal Flaws

**F1: Taxonomy correction data reporting error.**
The final_results_summary.md reports "corrected rate: 92.3% (delta = 0.0%)" but P4_taxonomy_correction.json shows corrected_comprehensive_rate = 0.192 (19.2%) with 19 letters changed. This is a straightforward data reporting bug that inverts the conclusion. The original 92.3% WAS an artifact, exactly as the proposal predicted. The corrected rate is 19.2%, a 73% absolute drop. This finding needs to be prominently reported as a major correction, not swept under the rug with "delta = 0.0%."

**F2: Cross-domain metric invalidated by its own controls.**
The dominance-based metric produces 100% on shuffled controls. This means every cross-domain absorption rate reported (51-85%) is uninterpretable. H2 is untested, not "partially supported." The cosine-calibrated metric at 0% suggests no absorption at all but may be too strict. The honest conclusion is that no valid cross-domain absorption measurement was achieved in this iteration.

### Serious Concerns

**S1: Within-width matching null contradicts causal narrative.**
All three within-width matching strategies fail to detect any absorption-quality association. This directly challenges H1's causal claim. The paper should present the within-width null as the primary result and the cross-width finding as a secondary observation, not the reverse. The headline "SUPPORTED_WITH_CAVEATS" should be downgraded to "INCONCLUSIVE" -- the evidence is ambiguous, with cross-width and within-width analyses giving opposite answers.

**S2: Proportion mediated > 1.0 for sparse probing (4.785).**
The mediation analysis for the paper's strongest metric (sparse probing F1) produces a mathematically incoherent proportion mediated because the total effect is non-significant. This metric cannot be used to support the mediation claim. Only SCR and TPP have interpretable mediation results. The synthesis summary's count of "3/4 metrics show significant indirect effect" is technically correct but misleading when the strongest metric's mediation is incoherent.

**S3: Architecture confound in 420-SAE scaling surface.**
The GAM does not control for SAE architecture (standard vs. jumprelu). The 54 jumprelu SAEs from Gemma Scope may have systematically different absorption properties due to their activation function, not due to width-L0 interaction. This confound should be tested.

**S4: Model downgrade from Gemma 2B to GPT-2 Small undermines Contribution 2.**
The proposal specified Gemma 2 2B for cross-domain experiments. The actual experiments used GPT-2 Small with 98% dead SAE features. This is such a fundamental deviation from the experimental plan that the Phase 2 results cannot support the original RQ2.

### Minor Caveats

**M1: Multiple testing without correction.**
50+ tests on 48 SAEs with no FDR or Bonferroni correction. The TPP results (p=0.022, Sobel p=0.037) are marginal.

**M2: Inconsistent phase_boundary_detected flag.**
P3_scaling_surface.json reports `phase_boundary_detected: true` (with 443 ridge points) while final_results.json reports `phase_boundary_detected: false`. One of these is wrong.

**M3: Mediation type inconsistency across summaries.**
The P1_synthesis_summary.md says "2/4 metrics show full Baron & Kenny mediation" while final_results.json says "n_full_mediations: 0." The raw data shows SCR and TPP have mediation_type: "full." The final_results.json `n_full_mediations: 0` is an aggregation bug.

---

## 5. Concrete Remediation

### For F1 (Taxonomy data reporting error)
**Action**: Fix the final_results_summary.md to report the correct corrected rate (19.2%, not 92.3%). Re-run the integration script that produced final_results.json. Verify the corrected_taxonomy section of P4_taxonomy_correction.json is being read correctly. This is a critical data integrity issue that can be fixed in < 1 hour.

**Expected outcome**: The corrected rate should be reported as 19.2% (from Strategy A/B correction) alongside the Chanin rate of 73.1%, with honest discussion of why these differ (Type II magnitude ratio measures something different from Chanin false-negative rate).

### For F2 (Cross-domain metric invalidation)
**Action**: The only valid path forward is to implement the Chanin integrated-gradients ablation method for knowledge hierarchies, as originally specified in Step 2.2 of the proposal ("Integrated-gradients ablation with KnowledgeGrader"). The dominance-based metric is a failed shortcut. Alternatively, obtain Gemma 2B HF access and re-run on a model where knowledge probes achieve >85% accuracy and SAE feature utilization exceeds 10%.

**Expected outcome**: If absorption genuinely exists on knowledge hierarchies, the IG-based method should detect it at >3x shuffled baseline. If it does not, H2 is honestly falsified.

### For S1 (Within-width matching null)
**Action**: Report the within-width null as a primary finding, not a caveat. Add a table comparing within-width vs. cross-width matching results side-by-side. Compute the minimum detectable effect size for within-width matching at 80% power given n=15-18 per stratum and report it. If the MDE is larger than the observed cross-width effect, the within-width null is uninformative (power too low). If the MDE is smaller, the within-width null is substantive evidence against the causal claim.

**Expected outcome**: Power analysis will likely show that n=15-18 per stratum has low power to detect moderate effects (r=0.3-0.5), making the within-width null somewhat uninformative. But this should be honestly quantified rather than assumed.

### For S2 (Proportion mediated > 1.0)
**Action**: Drop sparse probing F1 from the mediation narrative. Report that mediation analysis for sparse probing is uninterpretable because the total effect (L0 -> sparse probing) is non-significant (p=0.448). The causal chain claim should rest on SCR (full mediation, proportion=1.13, CI [0.43, 2.42]) and TPP (full mediation, proportion=0.54, CI [-0.01, 2.77]) only.

**Expected outcome**: The paper reports mediation results for 2 metrics (SCR, TPP) with full transparency about why sparse probing mediation fails.

### For S3 (Architecture confound)
**Action**: Re-fit the GAM with architecture (standard vs. jumprelu) as a factor covariate: `absorption ~ s(log_width) + s(log_l0) + ti(log_width, log_l0) + architecture`. If the interaction term remains significant after architecture control, the finding is robust. If not, report it.

**Expected outcome**: Given N=420 and 360 standard SAEs, the interaction likely remains significant even with architecture control. But this should be verified, not assumed. Runtime: < 5 minutes.

### For S4 (Model downgrade)
**Action**: Acknowledge in the paper that Phase 2 was conducted on GPT-2 Small rather than Gemma 2 2B due to access restrictions. Frame the Phase 2 results as a methodological finding (metric invalidation) rather than a substantive finding about cross-domain absorption. Defer the actual cross-domain contribution to a Gemma 2B experiment or a different model with adequate knowledge representations.

**Expected outcome**: The paper's Contribution 2 becomes "We demonstrate that dominance-based absorption metrics do not transfer to knowledge hierarchies on small models" rather than "We measure cross-domain absorption rates." This is still publishable but less impactful.

---

## Summary Verdict

The iteration 5 results contain one genuinely strong finding (H3: scaling surface interaction, p=3.1e-15, N=420), one ambiguous finding (H1: causal chain supported cross-width but not within-width), one failed finding (H2: metric invalidated by controls), and one finding that contradicts its own reported number (taxonomy correction: actual corrected rate 19.2%, reported as 92.3%).

**Two fatal flaws must be fixed before any writing proceeds**: (1) the taxonomy data reporting error, and (2) honest acknowledgment that the cross-domain metric is invalid. **Two serious concerns must be addressed in the paper narrative**: (1) the within-width matching null as evidence against strong causal claims, and (2) the incoherent sparse probing mediation.

The strongest path to a publishable paper is: (a) the 420-SAE scaling surface as the headline contribution, (b) the confound resolution as a methodological contribution with honest presentation of both the supporting (cross-width) and contradicting (within-width) evidence, and (c) the cross-domain metric failure as a cautionary tale. Attempting to sell absorption-quality causation as the headline claim would invite immediate rejection from any skeptical reviewer who notices the within-width null.

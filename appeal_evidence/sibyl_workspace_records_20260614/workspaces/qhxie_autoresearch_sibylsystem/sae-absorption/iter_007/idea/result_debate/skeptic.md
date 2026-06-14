# Skeptic Analysis: SAE Feature Absorption Audit (Iteration 9)

## 1. Statistical Risk Inventory

### Risk 1: The CMI-Absorption Correlation Is a Probe Quality Artifact

The raw Spearman rho(CMI, absorption) = -0.383 (p=0.059, uncorrected) already fails at alpha=0.05. After controlling for probe F1, the partial rho drops to -0.328 (permutation p=0.118; Bonferroni p=0.472). On the restricted set of 10 letters with F1>0.85, the correlation collapses to rho=-0.113 (p=0.757), with a CI spanning [-0.789, 0.808]. The "CMI diagnostic" contributes zero explanatory power once the probe quality confound is removed. The entire theoretical pillar (Q3) rests on a correlation that is statistically indistinguishable from zero after the most basic confound control.

Specific concern: rho(absorption, probe_F1) = -0.692 (p=0.0001). This is a far stronger relationship than any CMI association. Letters with bad probes (low F1) produce more false negatives, which the metric counts as "absorption." CMI's apparent association with absorption is likely driven entirely by this shared variance with probe quality.

### Risk 2: n=8 Persistent Core Words With Zero Recovery Is Not Interpretable

The activation patching experiment found 8 (not the expected 9) persistent core words and tested them. Recovery rate: 0/8 across three methods (primary, residual, all-children-zeroed). Control rate: 0/65. The proposal interprets this as "all-hedging narrative strengthened."

But consider: these 8 words have parent features with ZERO activation in the original encoding (confirmed: `parent_all_zero_confirmed: true` for all 8). If the parent features never fire at L0=82 regardless of context, then zeroing child features cannot make them fire, because the JumpReLU threshold simply prevents subthreshold activations from appearing. The patching experiment is testing whether removing one active feature makes inactive features cross their hard threshold -- but the parent features may be globally dead at this sparsity level, not locally suppressed by competition.

This is the critical distinction: **competitive exclusion** means the child's presence actively suppresses the parent. **Global sparsity** means the parent never fires at this L0 regardless. The patching experiment cannot distinguish between these two scenarios when parent activation is exactly 0.0 for all words. The experiment would need to test at an L0 where parents marginally fire (e.g., L0=41 where they fire for some words) to detect competitive exclusion.

### Risk 3: Tightened Hedging Redefines the Phenomenon Without Explaining It

The headline revision is dramatic: "98.6% hedging" becomes "6.2% strict hedging." But 615/656 FN tokens at L0=22 are NOT false negatives at L0=176, yet NONE of their 5 parent-associated latents fire at L0=176. The paper presents this as "compensatory features" resolving the FN. But this raises the question: if the letter information is captured by entirely different features at L0=176, what exactly is being "hedged"? The tightened analysis shows that the original 98.6% claim was essentially: "at higher L0, FNs resolve (trivially true because more features fire)." The strict 6.2% shows that specific parent-latent recovery is rare.

The real problem: the paper never defines what a "satisfactory" hedging rate would be. Is 6.2% strict hedging high or low? Compared to the shuffled control (3.4%), it is statistically significant (p=0.0004). But the absolute difference is 2.8 percentage points. This is a tiny effect, not the "dominant mechanism" the paper needs for its hedging narrative.

## 2. Alternative Explanations

### For the control failure (shuffled > measured in all 5 domains):
The paper claims this reveals a "structural" flaw in the Chanin metric. But the mechanistic explanation provided (`control_failure_diagnosis.json`) shows why: at cosine threshold 0.025 in R^2304, a random direction identifies 23% of features as candidates. With L0=82, at least one candidate fires with probability 1.0. The metric thus reduces to `absorption_rate ~ false_negative_rate`. Since shuffled probes have higher FN rates (worse classification), they produce higher "absorption." This is not a flaw in the metric -- it is a known property of high-dimensional spaces combined with a loose threshold. The paper could simply raise the cosine threshold to 0.05 (where only 1.6% of features are candidates) or 0.1 (where 0% are). The control failure may be entirely an artifact of the chosen threshold, not a structural property of the metric.

### For the L0 phase transition:
Absorption declining from 42.85% at L0=22 to 0.84% at L0=176 could simply reflect that higher-L0 SAEs have more features active, making the probe more accurate (higher F1) and reducing false negatives. If absorption_rate approximately equals false_negative_rate (as the control failure diagnosis claims), then the "L0 phase transition" is just the trivially expected improvement in probe accuracy at higher L0. The paper does not disentangle "fewer genuine absorption events" from "fewer probe errors at higher sparsity."

### For the activation patching null result:
JumpReLU SAEs have hard thresholds. A parent feature at subthreshold activation will read as exactly 0.0 regardless of whether a child is present. Zeroing the child adds its decoder contribution back to the residual stream, but this contribution may not push any parent feature past its JumpReLU threshold. This is an inherent limitation of patching experiments in JumpReLU architectures, not evidence about the mechanism of absorption. In L1-ReLU SAEs (the original Chanin architecture), parent features have continuous activations, so even partial suppression would be detectable through patching. The paper tests on JumpReLU only and draws conclusions about absorption in general.

## 3. Proxy Metric Audit

### What is measured vs. what is claimed:

| Claimed | Actually Measured | Gap |
|---------|------------------|-----|
| "Absorption metric does not transfer to JumpReLU" | Shuffled controls produce higher metric values than true labels | Demonstrates loose threshold in high-d space, not metric non-transfer. The metric may still correctly rank SAEs by absorption severity even if absolute values are miscalibrated. |
| "98.6% hedging (revised to 6.2%)" | FN tokens at L0=22 are not FN at L0=176 | Conflates "resolving at higher L0 via any mechanism" with "information was hedged across features." The operational definition of hedging is underspecified. |
| "L0 phase transition in absorption" | Absorption rate declines monotonically with L0 | Could be entirely explained by probe accuracy improvement. No separation of genuine absorption reduction from FN reduction. |
| "CMI predicts absorption susceptibility" | Marginal negative correlation at d'=10, non-significant after Bonferroni, collapses after probe F1 control | The CMI-absorption relationship does not survive basic confound analysis. Reporting it as "partially supported" is generous. |
| "Activation patching confirms all-hedging" | 0/8 parent recovery when parents have zero activation | Tests whether JumpReLU threshold can be crossed by removing features, not whether competitive exclusion exists. Confounded by architecture-specific threshold behavior. |

## 4. Severity Classification

### Fatal Flaws

**F1: CMI diagnostic is not a valid theoretical contribution.** Partial rho=-0.328 (Bonferroni p=0.472), restricted rho=-0.113 (p=0.757). The entire Q3 research question and H4 hypothesis are unsupported. The paper cannot present CMI as "partially supported" when the association vanishes after controlling for the most obvious confound (probe quality). Every sentence claiming CMI has diagnostic value must be removed or relegated to a "negative result" appendix.

**F2: The paper cannot distinguish genuine absorption from probe measurement error.** The control failure diagnosis (Section 5.1 presumably) demonstrates that `absorption_rate ~ false_negative_rate` at standard thresholds. If this is true, then the entire L0-absorption curve (H3) is measuring probe accuracy, not absorption. The paper's central claim -- that it audits absorption -- may be auditing probe quality instead. This undermines the entire contribution unless the paper can show that genuine absorption occurs above and beyond probe errors.

### Serious Concerns

**S1: Activation patching at L0=82 is confounded by JumpReLU architecture.** Testing competitive exclusion requires either (a) an architecture where parent activations are continuous (L1-ReLU), or (b) testing at an L0 where parent features marginally fire. The current 0/8 result is uninterpretable without this control. The paper should test at L0=41 (where 488 FNs remain but fewer than at L0=22) to find words where parents are near-threshold.

**S2: The "tightened hedging" experiment creates a narrative vacuum.** The paper demolished its own 98.6% headline but replaced it with 6.2%, which is only 2.8pp above shuffled controls. Neither the "all hedging" nor the "competitive exclusion" interpretation is supported. The paper needs to state clearly: "we cannot determine the mechanism for 93.8% of false negatives." This honest uncertainty would actually strengthen the contribution but requires significant narrative restructuring.

**S3: Cross-domain results are entirely negative but framed as a contribution.** "First cross-domain absorption measurements" (Contribution 4) produced zero interpretable results because controls fail universally. This is not a scientific contribution -- it is a failed experiment. Reporting it as a contribution invites reviewer pushback. It should be framed as: "we attempted cross-domain measurement and discovered that the metric is not suitable for this purpose."

**S4: Threshold sensitivity analysis undermines the "metric non-transfer" claim.** The threshold sensitivity analysis shows absorption rate is STABLE across threshold choices (CV=0.077, range 0.118-0.151). If the metric is "structurally broken," it should produce unstable results. Instead, it produces consistent results that the paper interprets as wrong because controls are also high. But consistent results with consistent control failure suggests the problem is CALIBRATION (threshold too loose for this dimensionality), not invalidity. A recalibrated metric (e.g., cosine >= 0.10, where zero random candidates appear) might work perfectly.

### Minor Caveats

**M1: 8 vs 9 persistent core words.** The proposal expected 9 but found 8. One word likely dropped due to edge-case filtering. This is minor but should be reported.

**M2: Letter G anomaly in tightened hedging.** G shows 90.5% strict hedging (19/21 FNs), while all other letters are 0-22%. This extreme outlier is driven by a single probe latent and should be investigated for data quality issues.

**M3: The leave-one-out CMI analysis finds no influential letters** (max delta_rho=0.088, below threshold), which is consistent with a weak, diffuse association rather than driven by outliers. This actually makes the story worse: the CMI correlation is not caused by outliers -- it is just weak overall.

## 5. Concrete Remediation

### For F1 (CMI diagnostic):
- **Action:** Demote CMI from "partially supported hypothesis" to "exploratory negative result."
- **Specific analysis:** Report that partial rho controlling for probe_F1 yields p=0.472 (Bonferroni). Report that restricted analysis (F1>0.85, n=10) yields rho=-0.113 (p=0.757).
- **Expected outcome:** The paper becomes stronger by not overclaiming. Reviewers will punish marginal significance presented as support but reward honest reporting.

### For F2 (Absorption vs. probe error confound):
- **Specific experiment:** At L0=82, compute absorption rate using ONLY the 10 letters with probe F1=1.0 (the tightened hedging experiment confirms all probes are F1=1.0 at L0=22 and L0=176). If absorption rate among perfect-probe letters is similar to the overall rate, then genuine absorption exists beyond probe error. If it drops to near zero, the entire phenomenon is probe error.
- **Expected outcome:** At L0=82 all probes are F1=1.0, so the existing absorption rate of 14.6% at L0=82 should already be free of probe quality confound. The paper should explicitly state this: "at L0=82 where all probes achieve F1=1.0, absorption rate is 14.6%, confirming that some absorption exists beyond probe error."
- **Wait** -- I note that tightened_hedging.json shows `probe_f1_l0_176: mean: 1.0` and `parent_latent_map_l0_22` all have `probe_f1: 1.0`. But the partial_correlation_cmi.json (which uses L0=82 SAE) shows per-letter F1 values ranging from 0.56 to 0.96. There is a discrepancy in which L0 point has perfect probes. The paper should clarify: probe F1 depends on L0, and the "perfect probes at L0=22" (from tightened hedging) are trained on L0=22 SAE features, while absorption rates from the main analysis use L0=82 SAE where probes are not perfect. This SAE-specific probe quality variation is a critical methodological detail.

### For S1 (Activation patching architecture confound):
- **Specific experiment:** Repeat activation patching at L0=41 (488 FNs), selecting words where at least one parent feature has non-zero activation (i.e., near-threshold). If such words exist and zeroing the child causes the parent to cross threshold, competitive exclusion is confirmed for those specific cases.
- **Alternative:** Test on the L1-ReLU SAEs from the Chanin et al. (2024) reproduction to compare patching results across architectures.
- **Expected outcome:** If 0 recovery at L0=41 as well, the case for hedging is very strong. If >0 recovery, the architecture-specific confound is confirmed.

### For S2 (Narrative vacuum):
- **Specific writing revision:** Replace the binary "hedging vs. competitive exclusion" framing with a three-category decomposition: (a) strict hedging (6.2%), (b) mechanism unknown -- resolved by compensatory features at higher L0 (93.8% - 6.2% = 87.6%), (c) not applicable (0% still-FN at L0=176 among non-hedging group). Explicitly state that the dominant mechanism is unresolved.

### For S4 (Recalibration rather than invalidity):
- **Specific analysis:** Re-run the absorption measurement using cosine threshold 0.10, where the control_failure_diagnosis shows zero random candidates on average. Report whether shuffled controls still exceed measured at this threshold. If shuffled <= measured at cosine >= 0.10, the metric is not "broken" -- it was miscalibrated.
- **Expected outcome:** At cosine >= 0.10, candidates drop to effectively zero per random direction. This may make the metric too stringent (no absorbing features detected), but it would resolve the control failure and demonstrate that the problem is threshold calibration for high-dimensional spaces, not fundamental metric invalidity. This distinction matters because "recalibrate the threshold" is a much weaker finding than "the metric is fundamentally flawed."

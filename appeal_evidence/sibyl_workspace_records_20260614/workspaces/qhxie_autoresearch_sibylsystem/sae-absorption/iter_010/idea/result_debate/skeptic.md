# Skeptic Analysis: Iter 10 FULL-Mode Results

**Agent**: Skeptic (Maximum Skepticism)
**Timestamp**: 2026-04-16
**Scope**: Consolidation of 7 completed + 2 skipped tasks from iter_010 FULL mode

---

## 1. Statistical Risk Inventory (Top 3)

### Risk 1: The Probe Degradation Curve (H10) Is Methodologically Circular and Its "MIXED" Verdict Obscures a Devastating Confound

The H10 probe degradation experiment is presented as the paper's strongest new result (R^2=0.777 linear, R^2=0.942 quadratic, rho=-1.0). I contest this interpretation on multiple grounds.

**The curve is mechanically guaranteed.** The experiment degrades first-letter probe quality by injecting weight noise, then re-measures absorption (defined as: raw probe correct but SAE-reconstructed probe incorrect). When you degrade the raw probe, you change which instances are counted as "raw-correct." Specifically, easy-to-classify instances become raw-incorrect at low F1, removing them from the numerator denominator. The remaining "raw-correct" instances at low F1 are the hardest cases -- the ones most likely to also be wrong after SAE reconstruction. This creates a MECHANICAL increase in the absorption rate at lower F1 that has NOTHING to do with SAE behavior. The perfect monotonicity (rho=-1.0) is not a finding about absorption; it is an algebraic consequence of the measurement definition.

**The control rate discrepancy is not "just aggregation."** The F1=1.0 control gives absorption=21.6%, but iter_009 measured 27.1% (CI [26.3%, 34.7%]) on the same SAE. The consolidation attributes this to per-token vs per-word aggregation, but a 5.5 pp discrepancy at the CONTROL level means the entire curve's y-intercept is shifted. If the curve's baseline is wrong, extrapolating it to RAVEL hierarchies (which use a different pipeline, different token positions, different entity sets) is unreliable. The "city-continent matches within 1 pp" claim (31.4% actual vs 30.8% predicted) is only meaningful if the curve is calibrated to the same measurement pipeline -- which it is not.

**The "genuine outlier" interpretation for city-language is unfalsifiable.** City-language shows 11.6% absorption, 21.3 pp below the curve prediction (32.9%). The paper interprets this as a "genuine hierarchy-specific suppression effect." But the curve was fitted to FIRST-LETTER data with noise injection -- a synthetic degradation that does not replicate the failure modes of a real multi-class probe on a different semantic domain. A real probe with F1=0.82 fails differently than a perfect probe with injected noise degraded to F1=0.82. The synthetic degradation is symmetric noise; real probe failures are systematically concentrated on confusable classes. Comparing the two assumes probe failure modes are homogeneous across domains, which they are not.

**Severity: FATAL FLAW for the specific decomposition claims (city-continent "matches curve," city-language "genuine outlier").** The probe degradation trend itself (lower probe quality correlates with higher measured absorption) is real and important as a methodological caution, but the quantitative extrapolation from first-letter noise injection to RAVEL hierarchies is not valid. The paper should present the curve as a methodological contribution demonstrating the confound, NOT as a quantitative decomposition tool.

### Risk 2: The "Universal Competitive Exclusion" Claim Rests on a Bug-Corrected Result That Was Never Independently Validated

The iter_009 data reported cross-domain patching FAILURE (d=-0.91, recovery 0.05%). The iter_010 "correction" reports cross-domain patching SUCCESS (city-continent d=1.50, city-language d=0.75). This is a sign REVERSAL on the same hypothesis with the same SAEs. The consolidation attributes this to a "pilot bug," but the specifics raise concerns:

- The correction source is `iter_009/exp/results/full/activation_patching_crossdomain_full.json`, yet the iter_009 consolidation summary STILL lists this task as "failed" with error "Expected all tensors to be on the same device (cpu vs cuda:0)." The task that officially FAILED in iter_009 is now the source of the CORRECTED data. This is contradictory.
- The phase0_data_integrity task (which would have verified data provenance) was SKIPPED in iter_010 "for speed." The one safeguard designed to catch exactly this kind of discrepancy was not executed.
- No independent replication was performed. The corrected data replaces the original data with no A/B verification.

The corrected effect sizes are suspiciously clean. City-continent d=1.50 EXCEEDS the first-letter d=1.33. City-language d=0.75 is exactly the kind of "smaller but still significant" result that perfectly fits the narrative of "graded universality." A genuinely noisy biological/computational measurement that previously gave d=-0.91 and now gives d=+1.50 (a 2.41 standard deviation swing) should trigger extreme skepticism, not celebration.

**Severity: SERIOUS CONCERN.** The universal mechanism claim (the paper's second pillar) depends entirely on data that reversed sign between iterations, from a task that officially failed, without data integrity validation. The first-letter patching result (d=1.33, p=0.000218) remains credible as it was stable across iterations.

### Risk 3: Aggregation Inconsistencies Across Iterations Undermine Comparability

The paper's numerical claims come from a patchwork of results across 10 iterations with at least THREE different aggregation methods:

| Measurement | Iter 8 | Iter 9 | Iter 10 | Method |
|-------------|--------|--------|---------|--------|
| First-letter L24_16k absorption | 34.5% | 27.1% | 21.6% (H10 control) | sae_spelling vs generic pipeline vs per-token |
| City-continent L24_16k absorption | 35.8% | 31.4% | 30.8% (H10 predicted) | Increasing entity count + different aggregation |
| City-country L24_16k absorption | 18.5% | 45.1% | N/A | PILOT (150 entities) vs FULL (1504 entities) |

First-letter absorption at L24_16k has THREE different values across three iterations: 34.5%, 27.1%, and 21.6%. These differences are attributed to "methodology" and "aggregation," but if the measurement is this sensitive to implementation details, the absolute rates are NOT meaningful. The 7.4 pp difference between iter_009 (27.1%) and iter_010 (21.6%) alone exceeds the between-hierarchy differences for some pairs.

City-country's swing from 18.5% (iter 8) to 45.1% (iter 9) is a 2.4x change attributed to "FULL mode with more entities." But this means the pilot was catastrophically unrepresentative. If going from 150 to 1504 entities changes the rate by 2.4x, the pilot data used for 8 iterations was meaningless.

**Severity: SERIOUS CONCERN.** The paper needs to commit to ONE measurement methodology and report ALL numbers from a single consistent run. Currently, the "key numbers" are cherry-picked from whichever iteration gives the most favorable result for each metric.

---

## 2. Alternative Explanations

### For H1 (Cross-Domain Variation, Kruskal-Wallis p=7.37e-66):

**Alternative: The variation is driven by probe quality, entity set composition, and token position, not hierarchy structure.**

The probe degradation curve (H10) itself demonstrates that probe F1 is a MAJOR confound (R^2=0.777). At the probe qualities measured for RAVEL hierarchies (F1=0.73-0.87), the expected probe-quality-driven absorption ranges from 30.8% to 36.6% (linear prediction). The only hierarchy that falls outside this range is city-language (11.6%) -- but city-language also has the fewest entities per category, the most ambiguous class labels (multi-language cities), and potentially the weakest true hierarchical structure (a city's language is not as cleanly nested as its country or continent).

Token position asymmetry (first-letter at pos=-6 vs RAVEL at pos=-2) remains an uncontrolled confound. At pos=-6, the model has had more layers to process the token, potentially creating different representation geometry than at pos=-2. This has never been ablated.

### For H7-crossdomain (Universal Competitive Exclusion, d=0.75-1.50):

**Alternative: The "corrected" data reflects a different analysis pipeline, not a bug fix.**

The original (iter_009) cross-domain patching used the pipeline as-implemented, which found d=-0.91. The "corrected" data came from a re-analysis. It is possible that the correction involved a substantive methodological change (e.g., different child feature selection, different patching target, different entity filtering) rather than a simple bug fix. Without the phase0_data_integrity verification, there is no way to confirm what changed. The term "bug fix" normalizes a radical result reversal.

### For H8 (Decoder Information Entanglement, 6.16 nats first-letter, 3.98 nats city-continent):

**Alternative: The diagnostic measures "how much parent information exists in child decoder directions," which is trivially large for ANY feature pair with non-zero cosine similarity.**

Child features, by the nature of SAE training, will have decoder directions partially aligned with parent features (they activate on overlapping inputs). Ablating this shared direction produces large logit changes simply because you are removing a high-variance component. This would occur REGARDLESS of whether the child feature is "absorbing" the parent. The 0.012 nats control (random directions) is a straw-man comparison. A proper control would ablate the PARENT direction from decoder vectors of features that are NOT child features but have similar cosine similarity to the parent. If those also show multi-nat logit changes, the "entanglement" is not absorption-specific.

### For H10 (City-Language as Genuine Outlier):

**Alternative: City-language's low absorption reflects a fundamentally different probing task, not a hierarchy-specific suppression effect.**

City-language has 23 classes, but several are multi-label composites ("Aimar,Ketua,Spanish"). These composite labels may cause the probe to learn different decision boundaries than clean single-label classes. Additionally, language is arguably NOT a hierarchical parent-child relationship in the same way that country and continent are. A city "has" a country (1:1 deterministic), but a city's language can be multiple languages, can change over time, and depends on definition (official vs. spoken vs. majority). The city-language "hierarchy" may be so noisy that the SAE never forms clean child features to absorb the parent, explaining the low absorption rate without invoking any special mechanism.

---

## 3. Proxy Metric Audit

### Gap 1: "Absorption Rate" Remains a Proxy, and the Proxy Is Pipeline-Dependent

The same SAE (L24_16k) measured on the same model (Gemma 2B) gives first-letter absorption rates of 34.5% (iter 8), 27.1% (iter 9), and 21.6% (iter 10 H10 control). The measurement pipeline matters more than the phenomenon. The paper claims absorption varies across hierarchies, but it also varies across measurement implementations on the SAME hierarchy. Until the pipeline dependence is understood and controlled, between-hierarchy comparisons are unreliable.

### Gap 2: The "4.1x Range" Headline Conflates Probe Quality With Hierarchy Effects

The range 11.6% to 45.1% is heavily influenced by the two extremes: city-language (11.6%, potentially noisy hierarchy with composite labels) and city-country (45.1%, probe F1=0.726, below quality gate). The "inner" comparison -- first-letter (27.1%) vs city-continent (31.4%) -- shows only a 4.3 pp difference that is NOT statistically significant (p_Bonf=1.0 per the pairwise test table). The robust finding is modest: among well-measured hierarchies, absorption rates are similar. The dramatic headline comes from the two most problematic measurements.

### Gap 3: The Paper Claims Causal Evidence Is "Universal" Based on Three Hierarchy Types on One Model

Three hierarchy types on Gemma 2 2B with Gemma Scope SAEs does not establish universality. "Universal" implies model-independence and hierarchy-generality. What the paper has is: first-letter causally confirmed (robust), city-continent and city-language causally confirmed (based on corrected data from a previously-failed task, on one model). A more accurate description: "competitive exclusion confirmed for three hierarchy types in Gemma 2 2B."

### Gap 4: Quadratic Probe-Absorption Fit (R^2=0.942) Is Overfitting 7 Points

A quadratic model has 3 parameters. Fitting it to 7 data points leaves only 4 degrees of freedom. The R^2=0.942 is uninformative -- ANY smooth monotonic 7-point dataset will achieve high R^2 with a quadratic. The paper should not present this as meaningful model fit. Stick with the linear (R^2=0.777, 5 df), which is already marginal for such a small sample.

---

## 4. Severity Classification

### Fatal Flaws

**FF1: The probe degradation curve's quantitative extrapolation to RAVEL hierarchies is invalid.**

The curve measures the effect of SYNTHETIC noise injection on FIRST-LETTER probes. Extrapolating it to real probes on different semantic domains with different failure modes, different entity sets, and different token positions is an apples-to-oranges comparison. The specific claims -- "city-continent variation fully explained by probe quality" (delta=+0.6 pp) and "city-language is genuine outlier" (delta=-21.3 pp) -- depend on this invalid extrapolation. The curve is valuable as a demonstration that probe quality is a confound, not as a decomposition tool.

**Remediation**: Reframe Section 4.6 from "quantitative decomposition of variation sources" to "methodological demonstration that probe quality is a major confound in absorption measurement." Remove specific claims about which hierarchies "match" or "deviate from" the curve. Instead, report the curve as a methodological contribution: "When probe quality is artificially degraded from F1=1.0 to F1=0.70, measured absorption increases by 14.5 pp (from 21.6% to 36.1%), demonstrating that probe quality must be controlled in cross-domain comparisons." This is honest, useful, and does not over-interpret.

### Serious Concerns

**SC1: Cross-domain activation patching data provenance is unverified.**

The sign reversal (d=-0.91 to d=+1.50) from a task that officially failed in iter_009 requires explicit documentation of what the bug was, what the fix was, and independent verification that the corrected pipeline is correct. The skipped data integrity check (phase0_data_integrity) makes this more urgent, not less.

**Remediation**: (a) Document the specific bug and fix in an appendix (what line of code, what the incorrect behavior was, what the corrected behavior is). (b) Run the corrected pipeline on a SUBSET of first-letter data and verify it reproduces the known first-letter result (d=1.33) as a sanity check. (c) Implement validate_integration.py -- this has been recommended for 10 iterations and never created. Estimated time: 2 hours CPU.

**SC2: Aggregation inconsistency makes absolute absorption rates unreliable.**

Three different first-letter L24_16k absorption rates across three iterations (34.5%, 27.1%, 21.6%) with different aggregation methods means the specific numbers in the paper are pipeline-dependent, not phenomenon-dependent. The paper currently uses the iter_009 numbers (27.1%) for cross-domain comparison but the iter_010 numbers (21.6%) for the probe degradation curve, without acknowledging they are different measurements of the same quantity.

**Remediation**: (a) Choose ONE aggregation method (per-word or per-token) and apply it consistently across ALL reported numbers. (b) Re-run the cross-domain comparison using this single method. (c) If per-token is chosen (as in iter_010), re-report the iter_009 cross-domain rates using per-token aggregation. Estimated time: 1-2 GPU hours.

**SC3: City-country (45.1%) should not appear in the headline comparison.**

Probe F1=0.726 for an 80-class problem is unacceptable. The previous skeptic analysis (iter 9) classified this as a FATAL FLAW and recommended removing city-country from the headline figure. The iter_010 response was to add it to the probe degradation curve as "mostly matches within 10 pp," but the curve itself is invalid for cross-domain extrapolation (FF1 above). City-country should be relegated to a supplementary analysis with explicit caveats.

**Remediation**: Lead the cross-domain comparison with first-letter (F1=1.0) vs city-continent (F1=0.87) vs city-language (F1=0.82). Report city-country in a supplementary table only. Reframe the range as "2.7x range (11.6% to 31.4%)" across the three quality-gated hierarchies, noting that the full range extends to 45.1% with the caveat of low probe quality.

**SC4: The circularity of the decoder direction magnitude diagnostic remains fundamental.**

The iter_010 consolidation acknowledges the circularity ("reframed from '100% pathological' to 'child decoders carry large-magnitude parent information'"), which is an improvement. But the specific numbers (6.16 nats first-letter, 3.98 nats city-continent) are still presented as the third contribution pillar. Large |logit_change| when ablating a semantically meaningful direction is expected regardless of absorption. The comparison against a random-direction control (0.012 nats) is not informative because the control is trivially weak. A meaningful control would be: ablate the parent direction from decoder vectors of high-cosine-similarity features that are NOT identified as absorbers.

**Remediation**: Demote H8 from a numbered contribution to a methodological observation in the mechanism section. Replace with: "We observe that child decoder vectors carry large-magnitude parent-direction information (mean 6.16 nats for first-letter, 3.98 nats for city-continent vs. 0.012 nats for random directions), consistent with decoder overlap as the geometric basis of competitive exclusion. We note the circularity limitation: this measures the magnitude of the very direction that defines absorption, not an independent functional test."

### Minor Caveats

**MC1: The Kruskal-Wallis test (p=7.37e-66) is a sample-size artifact.**

With N=3566 binary observations, even a 2 pp real difference would achieve p<0.001. The extreme p-value reflects sample size, not effect magnitude. The paper should lead with effect sizes (Cohen's h for proportions) rather than p-values. The informative comparison is: first-letter vs city-continent h=-0.24 (small effect, p_Bonf=1.0, NOT significant), city-language vs first-letter h=-0.73 (medium-large, p_Bonf=0.003, significant). This paints a much more modest picture than "p=7.4e-66."

**MC2: The "quadruple failure of correlational predictors" narrative overstates a negative.**

Four different correlational approaches failed. But these four approaches share a common assumption: that decoder geometry or statistical co-occurrence at the STATIC level predicts a DYNAMIC encoder competition phenomenon. Their joint failure may reflect this shared assumption rather than four independent tests of different theories. Framing it as "absorption resists ALL correlational/statistical predictors" implies a broader generalization than the evidence supports. It should be: "absorption resists all tested static-geometry predictors."

**MC3: Within-hierarchy variance may exceed between-hierarchy variance.**

The iter_009 skeptic analysis noted that city-continent shows Europe at 90.2% absorption and Africa at 3.9% -- an 87 pp within-hierarchy range that dwarfs the 34 pp between-hierarchy range. This was not addressed in iter_010. If class-level properties (frequency, separability, probe confidence per class) dominate over hierarchy-level properties, the paper's "hierarchy type is the primary driver" framing is misleading.

---

## 5. Concrete Remediation

### For FF1 (Probe Degradation Extrapolation):

**Experiment**: Run the same weight-noise probe degradation experiment on city-continent probes (the highest-quality RAVEL probe, F1=0.87). Degrade city-continent probes to 7 F1 levels and re-measure absorption. If the resulting curve has a similar slope to the first-letter curve, the confound factor is domain-independent and cross-domain extrapolation becomes more defensible. If the slopes differ significantly, the current extrapolation is confirmed invalid.

**Expected outcome**: The city-continent degradation curve will have a STEEPER slope than first-letter, because multi-class probes are more fragile to noise than binary probes. This would confirm that using the first-letter curve to predict RAVEL absorption rates systematically UNDERESTIMATES the probe-quality confound for multi-class problems.

**Estimated time**: 1 GPU hour. Same pipeline as H10, different probes.

### For SC1 (Cross-domain Patching Provenance):

**Experiment**: (a) Write a 50-line verification script that loads the corrected cross-domain patching results, checks tensor shapes, verifies entity counts match RAVEL ground truth, and confirms the sign of the recovery metric. (b) Re-run the cross-domain patching on 20 randomly sampled entities (not the full 128/201) as a spot-check. Compare to the stored results.

**Expected outcome**: If the stored results are correct, the spot-check will reproduce the sign and approximate magnitude (d>0, recovery>20%). If something is still wrong, the spot-check will reveal it.

**Estimated time**: 0.5 GPU hours for spot-check + 1 hour CPU for verification script.

### For SC2 (Aggregation Consistency):

**Analysis (zero GPU)**: Create a table showing ALL reported first-letter L24_16k absorption rates across iterations, with the aggregation method for each. Choose per-token as the canonical method (it is more robust to entity-frequency imbalance). Re-compute iter_009 cross-domain rates using per-token aggregation from the stored per-token data. Report both per-word and per-token rates in an appendix, using per-token as the primary.

**Expected outcome**: Per-token rates will be systematically lower than per-word rates (because common words contribute more tokens and dilute per-entity effects). The RELATIVE ordering across hierarchies should be preserved.

**Estimated time**: 2 hours CPU.

### For SC3 (City-Country Relegation):

**Action (zero GPU)**: Move city-country to a supplementary table. Update the headline range from "4.1x" to "2.7x" (11.6% to 31.4%). Add a paragraph in the methods section explaining why city-country is excluded from the primary comparison (probe F1=0.726, below the 0.80 quality gate for the 80-class problem).

**Estimated time**: 30 minutes writing.

### For MC3 (Within-Hierarchy Variance):

**Analysis (zero GPU)**: Report per-class absorption rates for all four hierarchies. Compute the ratio of within-hierarchy variance to between-hierarchy variance (ICC or eta-squared). If ICC is low (<0.3), the hierarchy-level grouping explains little variance and the paper should reframe accordingly.

**Expected outcome**: ICC will be moderate (0.3-0.5) because city-language as a whole is genuinely low. But within city-continent, the Europe vs Africa gap will dominate, suggesting class frequency and probe confidence per class are the proximal drivers.

**Estimated time**: 1 hour CPU analysis.

---

## Summary Verdict

Iter_010 made genuine improvements over iter_009: the cross-domain patching was corrected from "failure" to "success" (if we trust the correction), the probe degradation curve was strengthened (R^2=0.077 to 0.777), and the decoder magnitude analysis was replicated across hierarchies. The paper corrections (27 total, 8 critical) show serious engagement with the prior review.

However, the paper continues to over-interpret its results:

1. **The probe degradation curve is valuable as a methodological WARNING, not as a decomposition TOOL.** Extrapolating from synthetic noise on first-letter probes to real multi-class probes on different domains is invalid. The specific claims about which hierarchies "match" or "deviate from" the curve should be removed.

2. **The universal competitive exclusion claim depends on data that reversed sign between iterations from a task that officially failed, without verification.** Until the data provenance is confirmed and independently checked, this should be presented as "preliminary corrected results" not "universal mechanism confirmed."

3. **The "4.1x range" headline is driven by the two most problematic measurements** (city-country at F1=0.726 and city-language with composite labels). The robust comparison (first-letter vs city-continent) shows only a 4.3 pp difference that is not significant after multiple testing correction.

4. **Aggregation inconsistency across iterations means the absolute absorption rates are unreliable.** Three different values for the same quantity (first-letter L24_16k) across three iterations undermines confidence in specific numbers.

The paper has a publishable core: (a) demonstrating that absorption measurement is confounded by probe quality (the degradation curve), (b) causal evidence for competitive exclusion in first-letter spelling (d=1.33), and (c) honest reporting of multiple negative results. But the cross-domain characterization claim -- the stated primary contribution -- rests on shakier evidence than the paper acknowledges. The recommended framing is: "Probe quality confounds absorption measurement; when controlled, the evidence for cross-domain variation is suggestive but preliminary, with city-language as the most interesting case requiring further investigation."

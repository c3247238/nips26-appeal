# Revisionist Analysis: Iteration 5 Results

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| **H1: Absorption-Quality Causal Chain** | Confirmed with caveats | 3/4 metrics retain \|partial_r\| > 0.2 after L0 control; sparse probing *strengthened* from -0.664 to -0.746; Rosenbaum Gamma=2.65; 3/4 metrics significant mediation (Sobel p < 0.05) | **70%** -- the within-width matching null (0/2 within-width strategies significant) means we cannot fully disentangle absorption from width |
| **H2: Cross-Domain Absorption** | Refuted (as operationalized) | Shuffled controls show 100% absorption rate; cosine-calibrated metric shows 0%; dominance-based metric does not discriminate real vs. random hierarchies | **85%** -- the metric failure is unambiguous; the underlying question (does absorption generalize?) remains open |
| **H3: Scaling Surface Interaction** | Confirmed | GAM interaction p=3.11e-15 on N=420 SAEs; R-squared: 0.488 (linear) -> 0.620 (additive) -> 0.693 (interaction); phase boundary at log2(L0) in [2.7, 3.8] | **90%** -- large sample, highly significant, replicated across layers |
| **H4: Early Absorption Dominance** | Not tested | GPT-2 Small experiments did not include the decoder-geometry analysis needed for Type I/II classification on knowledge features | N/A |
| **H5: Taxonomy Correction** | Confirmed (but final summary has an error) | Corrected comprehensive rate: **19.2%** (down from 92.3%); Type II drops from 88.5% to 15.4%; Chanin-based rate: 73.1%. The final_results.json incorrectly reports "delta=0.0%" -- this contradicts P4_taxonomy_correction.json which shows delta=-73.1% | **90%** -- the correction methodology is sound; the final integration script has a reporting bug |

## 2. Surprise Analysis

### Surprise 1 (MAJOR): Sparse probing correlation STRENGTHENED after L0 control

**Expected**: We assumed L0 was a confounder inflating the absorption-quality correlation. The proposal estimated partial r would "drop from |0.595| to the range |0.3-0.45|."

**Actual**: Sparse probing partial r *increased* from -0.664 to -0.746 (delta = -0.082), a suppression effect.

**Wrong assumption**: We assumed L0 and absorption both independently lower quality, creating an upward-biased bivariate correlation. Instead, L0 partially *masks* absorption's true effect on sparse probing quality. The suppression variable mechanism means L0 and absorption have opposite partial effects on this metric -- higher L0 actually helps sparse probing quality through a path not involving absorption, and removing this confounding path reveals a stronger absorption signal.

**Magnitude of surprise**: The direction of the delta flipped relative to expectation. This is not a 20% deviation -- it is a qualitative reversal of the expected confounding structure.

### Surprise 2 (MAJOR): Cross-domain absorption metric completely fails to discriminate

**Expected**: We predicted cross-domain absorption rates of 10-35% exceeding 3x the shuffled baseline. The proposal anticipated knowledge hierarchies would behave similarly to first-letter spelling because "India is always in Asia" mirrors the deterministic co-occurrence of the spelling task.

**Actual**: The dominance-based metric shows 51-85% absorption on real labels AND 100% on shuffled labels. The cosine-calibrated metric shows 0% on both. The metric captures background SAE feature concentration (one "super-absorber" feature 8213 dominates nearly every token), not probe-direction-specific absorption.

**Wrong assumption**: We assumed that the dominance-based absorption metric (adapted from Chanin et al.) would transfer to knowledge hierarchies on GPT-2 Small. Two flawed premises:
1. We assumed GPT-2 Small SAEs (24k features, 98% dead) would have enough alive features to produce meaningful feature splits for geographic knowledge. With only ~300 alive features and ~75 active per token, a polysemantic "super-absorber" dominates by default.
2. We assumed the dominance threshold alone (activation_top / activation_second > threshold) would distinguish genuine absorption from background feature concentration. It does not. The metric needs an additional cosine-alignment requirement against the probe direction, but when we add that (cosine-calibrated), absorption drops to 0% -- suggesting the issue is not metric design but model capacity.

**Magnitude of surprise**: The 100% shuffled rate is devastating for the metric. The entire cross-domain contribution as operationalized is invalidated. This is >100% deviation from the 10% expected absorption rate -- the signal is pure noise.

### Surprise 3 (MODERATE): Taxonomy correction is massive, not minimal

**Expected**: We hypothesized the 92.3% combined rate would "drop below 80%" (H5b target <80%).

**Actual**: The corrected comprehensive rate is **19.2%** (95% CI: [3.8%, 34.6%]) -- a 73.1 percentage point drop. This far exceeds our pre-registered expectation. However, the Chanin false-negative-based metric still detects absorption in 73.1% of letters, validating that absorption exists but the magnitude-ratio-based taxonomy was severely inflated.

**Wrong assumption**: We assumed the global mean-when-active fallback would produce moderate inflation. The actual mechanism is more extreme: for 21/26 letters with n_comparison_tokens=0, the global baseline is systematically biased upward because it includes high-activation contexts unrelated to letter identity. When using proper non-letter-context comparison tokens (Strategy A), parent features fire *more strongly* on letter tokens, flipping the Type II classification entirely.

**CRITICAL NOTE**: The final_results.json and final_results_summary.md both report "corrected_rate: 0.923, delta: 0.0" -- this contradicts the actual P4_taxonomy_correction.json which shows corrected_rate: 0.192 and delta: -0.731. This appears to be a bug in the final integration script. The revisionist analysis uses the P4 task data as ground truth.

### Surprise 4 (MODERATE): Within-width matching produces null results

**Expected**: We predicted H1b would succeed for "at least 2 of 3 width strata."

**Actual**: Zero of five within-width matching strategies produced significant results. Exact-width NN L0 matching (4 pairs), median-split within width (23 pairs), and tertile within width (16 pairs) all yield p > 0.10 for every quality metric. Only cross-width strategies (Mahalanobis: 17 pairs, propensity: 6 pairs) detect significant effects.

**Wrong assumption**: We assumed there was sufficient L0 variation within each width stratum to create meaningful high/low absorption contrasts. In practice, within a given width (e.g., all 16k SAEs), the L0 range is narrow and absorption variation is small. The absorption-quality signal is dominated by cross-width differences, which may reflect a genuine width-driven mechanism rather than pure absorption.

### Surprise 5 (MINOR): PMI predicts absorption in hurdle model but not OLS

**Expected**: PMI (pointwise mutual information between letters and tokens) was presumed non-significant based on iter_004.

**Actual**: OLS with clustered SE: PMI p=0.667 (non-significant). Hurdle model logistic part: PMI p=0.006 (significant). Beta regression: PMI p=0.005 (significant). Higher PMI letters are less likely to show *any* absorption, but once absorption occurs, PMI does not predict its magnitude.

**Wrong assumption**: We treated the PMI-absorption relationship as a simple linear question, but 58.6% of observations are exactly zero. The distributional structure matters: PMI affects the probability of absorption onset, not its severity.

## 3. Mental Model Revision

**Original mental model**: Absorption is a causal quality-degrading mechanism that operates independently of SAE hyperparameters (width, L0). Cross-domain generalization is expected because the deterministic co-occurrence structure that drives absorption in first-letter spelling is also present in knowledge hierarchies. The 92.3% taxonomy rate reflects pervasive absorption.

**Revised mental model**: Absorption's relationship to quality is real and robust to L0 confounding for 3/4 metrics, but it is primarily a *cross-width* phenomenon -- it cannot be cleanly separated from width effects within a given width stratum. The evidence is strongest for the L0 -> Absorption -> SCR/TPP mediation pathway, not a universal law. Cross-domain generalization of absorption remains completely untested: the metric failure on GPT-2 Small with 98% dead features tells us about model/SAE capacity limitations, not about whether absorption generalizes in larger models. The true prevalence of first-letter absorption is approximately 19.2% (magnitude-based) or 73.1% (false-negative-based), depending on which definition one adopts -- the 92.3% figure was an artifact.

**Key update**: We must separate three questions that we had conflated: (1) Does absorption correlate with quality? (Yes, robustly.) (2) Does absorption *cause* quality degradation independent of width? (Uncertain -- the within-width null means cross-width confounding remains possible.) (3) Does absorption generalize beyond spelling? (Unknown -- the experiment failed to test this due to metric and model limitations.)

## 4. Reframing Test

**Would we frame the research question the same way?** No. Three revisions are needed:

1. **The confound resolution framing should emphasize the suppression effect, not just "surviving" L0 control.** The original framing asked "does the correlation survive?" The more interesting story is that the correlation *strengthens*, revealing a suppression structure. The paper should lead with: "L0 was masking, not driving, the absorption-quality relationship for sparse probing."

2. **The cross-domain contribution should be reframed as a negative/methodological result.** Instead of "first measurement of absorption on knowledge hierarchies," the finding is: "the standard absorption metric does not generalize to small models with high SAE sparsity, and a model-capacity prerequisite must be established before cross-domain claims can be made." This is a different but valuable contribution -- it identifies the boundary conditions for absorption measurement.

3. **The taxonomy correction should be promoted from a minor bookkeeping fix to a headline finding.** A 73.1 percentage point reduction in the reported absorption rate (from 92.3% to 19.2%) is arguably the most consequential result: it changes the field's quantitative understanding of absorption prevalence by a factor of 4.8x. The Chanin-based 73.1% rate should be the canonical number, with the corrected 19.2% as the conservative bound.

**Revised research question**: "Feature absorption in SAEs: how robust is the causal chain, how inflated is the prevalence, and what are the boundary conditions for cross-domain measurement?"

## 5. New Hypothesis Generation

### New H1: Model-Scale Threshold for Cross-Domain Absorption

**Statement**: Feature absorption on knowledge hierarchies requires a model-SAE configuration where (a) the SAE has >5% alive features (not 2% as in GPT-2 Small) AND (b) the model demonstrably encodes the target knowledge as a linear direction recoverable by probes with >85% accuracy.

**Falsifiable test**: Replicate Phase 2 on Gemma 2 2B with Gemma Scope SAEs (which have >50% alive features at 16k width). If absorption remains 0% on the cosine-calibrated metric despite meeting both conditions, then absorption is genuinely specific to syntactic features like first-letter identity and does not generalize to knowledge hierarchies.

**Rationale**: The GPT-2 Small experiment conflated two possible explanations for the null: (1) absorption doesn't generalize, or (2) the model/SAE was too small to represent knowledge features with enough dedicated SAE latents. Hypothesis New H1 resolves this ambiguity.

### New H2: Suppression-Variable Structure Is General

**Statement**: The L0 suppression effect on sparse probing (adding L0 as a covariate strengthens the absorption-quality correlation) will replicate across at least 2 additional quality metrics if the analysis is extended to the 420-SAE SAEBench dataset with richer L0/width variation.

**Falsifiable test**: Compute partial correlations between absorption and all available quality metrics on the 420-SAE dataset, with and without L0 as covariate. If the suppression pattern (partial r *increasing* after L0 control) appears for 0/3 additional metrics, the effect is specific to sparse probing and not a general property of the absorption-quality relationship.

**Rationale**: The suppression effect was the biggest surprise in Phase 1. If it generalizes beyond sparse probing, it fundamentally changes how SAE evaluators should interpret absorption: not as a competing explanation with L0, but as a complementary and partially masked signal.

### New H3: Absorption Prevalence Has a Bimodal Distribution

**Statement**: The corrected taxonomy reveals a bimodal pattern: letters either show clear absorption (Chanin partial/high, ~73%) or no absorption at all (~27%), with very few in the "low but non-zero" range. This bimodality reflects whether the SAE has *any* dedicated feature for that letter class (binary outcome, captured by the hurdle model) rather than a continuous absorption gradient.

**Falsifiable test**: Fit a mixture model (two-component beta mixture) to the per-letter absorption rates. If the BIC of the mixture model significantly beats the single-component model (delta BIC > 10), bimodality is confirmed. The hurdle model's logistic component should classify 80%+ of letters into the correct mixture component.

**Rationale**: The 58.6% zero-absorption rate and the PMI hurdle finding both point toward a binary mechanism: either an SAE has a dedicated feature for a letter class (no absorption) or it does not (absorption occurs). This binary view is more parsimonious than a continuous absorption gradient and directly actionable: practitioners should check for dedicated feature coverage rather than measuring absorption rates.

---

## Summary of Belief Updates

| Belief | Prior (pre-experiment) | Posterior (post-experiment) | Update Magnitude |
|---|---|---|---|
| Absorption causes quality degradation independent of L0 | 65% confident | 70% confident (stronger for sparse probing, weaker for within-width) | Small increase with major caveats |
| Absorption generalizes to knowledge hierarchies | 75% confident | 20% confident (metric failed; question unresolved, not answered) | **Large decrease** |
| 92.3% of letters show absorption | 50% confident (acknowledged as inflated) | 5% confident -- corrected rate is 19.2% (magnitude-based) or 73.1% (Chanin FN-based) | **Massive decrease** |
| Scaling surface has interaction structure | 60% confident | 95% confident (p=3.1e-15, N=420) | Large increase |
| L0 is a confound inflating absorption-quality correlation | 70% confident | 20% confident -- L0 is a **suppressor**, not a confound for sparse probing | **Direction reversal** |

## Critical Data Integrity Issue

The final integration (final_results.json and final_results_summary.md) reports H5 taxonomy correction as "delta = 0.0%, corrected_rate = 92.3%". This directly contradicts P4_taxonomy_correction.json which shows corrected_rate = 19.2% and delta = -73.1%. This is not a judgment call -- it is a reporting error in the integration script. The next iteration must fix this before any paper writing, as it would misrepresent the most consequential quantitative finding. The actual P4 result (92.3% -> 19.2%) transforms the paper's narrative from "absorption is nearly universal" to "absorption affects roughly one-fifth of letter features by the magnitude criterion."

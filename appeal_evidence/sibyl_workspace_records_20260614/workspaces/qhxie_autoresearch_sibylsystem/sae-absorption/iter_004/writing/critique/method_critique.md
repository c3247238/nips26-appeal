# Critique: Method (Section 3)

## Summary Assessment
The Method section defines three new quantities---the LV competition coefficient, the Distributed Absorption Score (DAS), and a three-tier taxonomy---with clear mathematical notation and a useful ecological analogy. The section reads well structurally, moving from a single-pair metric to a multi-child metric to a full taxonomy. However, several claims are stated without evidence or with misleading framing given that the detector ultimately fails (test F1 = 0.128), and the Type II definition has a known validity problem that the section underplays. The taxonomy table is strong, but the section over-promises on the sharpness prediction and under-specifies several methodological choices.

## Score: 6/10
**Justification**: Clean formal definitions and a sensible organizational structure earn the section solid ground. It loses points for (1) making strong theoretical predictions (sharp sigmoid at alpha ~ 1) in the method section that are decisively refuted in Section 5, creating a credibility problem for the attentive reader; (2) under-specifying the DAS estimation procedure; (3) a Type II definition with an acknowledged caveat that borders on invalidity. Reaching a 7 would require tightening the DAS specification, softening the sharpness prediction to a testable hypothesis rather than a confident claim, and providing a clearer justification for the Type II threshold.

## Critical Issues

### Issue 1: Type II definition relies on a measurement procedure the paper itself admits is invalid
- **Location**: Section 3.3, Table 1, Type II row
- **Quote**: "Parent activation magnitude < 50% of expected"
- **Problem**: The experiments section (5.3) and discussion (7.4) both disclose that parent features were identified via a "selectivity heuristic" rather than `sae-spelling` ground truth, and that the comparison token count is 0 for most letters, forcing a fallback to global statistics. This means the "expected magnitude" denominator is unreliable. Despite this, the Method section presents the Type II definition without any qualification. The caveat only appears 5 pages later in Section 3.3's final paragraph and is easy to miss relative to the prominent table. A reader encountering the taxonomy table will assume it is well-operationalized; the 92.3% comprehensive rate (which is dominated by Type II at 88.5%) is then built on a foundation the paper acknowledges is weak. For a NeurIPS submission, defining a core metric whose ground truth the paper itself questions is a critical problem---reviewers will flag this as inflated measurement.
- **Fix**: (a) Add an explicit caveat directly in Table 1 (e.g., a footnote: "Type II measurement requires validated parent feature IDs; see Section 7.4 for limitations of the heuristic used here"). (b) Define what "expected magnitude" means concretely (expected on what token set? how many tokens? what fallback if the set is empty?). (c) Consider reporting the taxonomy both with and without Type II to show robustness.

### Issue 2: The sharpness prediction is stated as a consequence of the framework but is decisively refuted
- **Location**: Section 3.1, paragraph 3
- **Quote**: "The LV competitive exclusion principle predicts that when alpha_ij > 1, the rarer species---here, the parent feature i---is excluded (absorbed) by the more frequent child j. The threshold at alpha_ij ~ 1 should produce a sharp sigmoid transition in absorption probability, distinguishing this framework from a generic monotone relationship between co-activation and absorption."
- **Problem**: The word "predicts" and "should produce" present the sharp-threshold property as a theoretical consequence of the framework. In Section 5.1, the sharpness test finds no sigmoid transition (linear AIC = -61.05 vs sigmoid AIC = -60.95; sigmoid center diverges to x_0 = 10.0). Presenting a confident prediction in the method section that is refuted in the results creates an uncomfortable reading experience: either the analogy is weaker than advertised, or the method section is overselling. For a paper that reports H1 as a negative result, the method section should frame the threshold prediction as a *testable hypothesis* (to be evaluated in Section 5), not as a deductive consequence of the framework.
- **Fix**: Rewrite as: "Under the LV competitive exclusion principle, alpha_ij > 1 *would* predict exclusion of the rarer species. If this analogy holds precisely, absorption probability should show a sharp sigmoid transition near alpha_ij ~ 1, distinguishing the LV framework from a generic monotone relationship. We test this prediction in Section 5.1." This framing signals to the reader that the prediction is being set up for empirical evaluation, making the negative result feel like rigorous science rather than a failed claim.

## Major Issues

### Issue 3: DAS estimation procedure is under-specified
- **Location**: Section 3.2, "Estimation procedure" steps 1--4
- **Quote**: "Fit a logistic regression predicting 1[a_P > 0] from (a_{C_1}, ..., a_{C_k})."
- **Problem**: Several critical details are missing: (a) Is the logistic regression fitted with a regularization penalty? For k=3, overfitting is unlikely, but k=1 (a single predictor) could suffer from perfect separation if the child always co-activates with the parent. (b) How is McFadden pseudo-R^2 used to approximate conditional entropy reduction? The paper states "DAS(P, k) = 1 - H(a_P | C_1,...,C_k) / H(a_P), approximated via the McFadden pseudo-R^2," but McFadden pseudo-R^2 is defined as 1 - log(L_full) / log(L_null), which is *not* algebraically equal to conditional entropy reduction. It is a rough proxy. The section should clarify this approximation and note its limitations. (c) Are activations binarized (a_P > 0) for both the dependent and independent variables, or are the children's raw activation magnitudes used as predictors?
- **Fix**: Add a brief clarification: "We use the McFadden pseudo-R^2 as an approximation to the conditional entropy ratio (Hagle & Mitchell, 1992). Raw activation magnitudes (a_{C_1}, ..., a_{C_k}) serve as continuous predictors; the dependent variable is binarized (1[a_P > 0]). No regularization is applied; for k=3 features on n=10,000 tokens, overfitting is negligible." Also note the approximation quality limitation.

### Issue 4: DAS(k=1) equivalence to Chanin metric is claimed but not demonstrated
- **Location**: Section 3.2, final paragraph of subsection
- **Quote**: "DAS(P, k=1) reduces to single-child absorption and is directly comparable to the Chanin metric."
- **Problem**: This equivalence is asserted without proof or empirical validation. The Chanin metric uses probe-directed comparison with ground-truth letter directions, while DAS(k=1) uses a logistic regression of parent activation on the top-alpha child's activation. These are structurally different computations. "Directly comparable" could mean (a) they measure the same underlying quantity, (b) they produce similar numerical values, or (c) they rank features similarly. The results never actually show a calibration between DAS(k=1) and the Chanin metric. If they diverge substantially, the entire DAS framework loses its grounding in the established literature.
- **Fix**: Either (a) provide a formal argument for why DAS(k=1) approximates the Chanin metric, (b) show empirical Pearson r between DAS(k=1) and Chanin metric across the 26 letters, or (c) weaken the claim to "DAS(P, k=1) captures single-child absorption and is conceptually analogous to, though computationally distinct from, the Chanin metric."

### Issue 5: Computational pre-filter coverage is poor, undermining the detector
- **Location**: Section 3.1, "Computational pre-filter" paragraph
- **Quote**: "their decoder columns have cosine similarity cos(d_i, d_j) > 0.15"
- **Problem**: Section 6.2 reveals that this filter retains only 34.0% of truly absorbed pairs. A pre-filter that discards 66% of positive cases before the detector even runs imposes a hard ceiling on recall of 0.34. This is a fundamental design limitation that should be flagged in the method section, not buried in an ablation. The 68x reduction in candidate pairs sounds impressive, but the reader needs to know the precision-coverage tradeoff upfront.
- **Fix**: Add one sentence: "We evaluate the coverage-precision tradeoff of this filter in Section 6.2; at the default threshold, 34% of absorbed pairs are retained, imposing a recall ceiling that limits overall detector sensitivity."

### Issue 6: Threshold selection for the Type II magnitude ratio is unjustified
- **Location**: Section 3.3, Table 1
- **Quote**: "Parent activation magnitude < 50% of expected"
- **Problem**: Why 50%? This is the central threshold for the largest absorption category (23/26 letters = 88.5%). No sensitivity analysis is offered. Would 40% or 60% change the results materially? The choice appears arbitrary, and for a metric that drives the headline "92.3% comprehensive absorption rate," the threshold should either be derived from a principled criterion or shown to be robust across a range.
- **Fix**: Either (a) justify the 50% threshold from prior work or a calibration analysis, (b) report the Type II rate at 30%, 40%, 50%, 60%, and 70% thresholds to show robustness, or (c) acknowledge the arbitrariness explicitly and show the sensitivity in the ablations.

## Minor Issues

- **Section 3.1, paragraph 1**: "Feature absorption admits a natural ecological analogy" uses "admits" in a way that implies the analogy is formally valid, which the paper later refutes. Consider "Feature absorption suggests an ecological analogy" to avoid implying formal equivalence.
- **Section 3.1, equation for sigma_ij**: The definition uses min(f_i, f_j) in the denominator. For the parent-child case where f_j >> f_i (as described), min is always f_i. This should be stated explicitly, since the asymmetry matters for interpretation: sigma_ij then equals P(both active) / f_i = P(j active | i active), the conditional probability that the child fires given the parent fires.
- **Section 3.1, "Detection rule" paragraph**: "calibrated threshold tau" is forward-referenced to Section 4.2, but Section 4.2 is labeled "Ground Truth and Data Split," not "Threshold Calibration." The cross-reference should point to Section 4.3 (Evaluation Protocol, H1) where the calibration procedure is described.
- **Section 3.2**: "n = 10,000 tokens" --- is this per parent or total? If per parent, the total computation may be significant for wide SAEs. If total, 10,000 tokens may under-represent rare parents. Clarify.
- **Table 1, Type III row**: The threshold "DAS(k=3) > 0.6 AND Type I not triggered" uses 0.6, but the proposal (proposal.md) uses "> 80% of parent's firing deficit." These are different criteria. Verify which was actually implemented and ensure consistency.
- **Section 3.3, final paragraph**: "Type II classification relies on comparing the parent latent's activation magnitude on letter-specific tokens against a baseline (global mean activation when the parent fires)" --- this parenthetical "(global mean activation when the parent fires)" is the first time the reader learns what "expected magnitude" means. This definition should appear in the Type II row of Table 1 itself.
- **Figure 1 placement**: The figure is referenced at the end of the introductory paragraph ("Figure 1 illustrates the conceptual mapping") but the actual figure image tag appears after Section 3.1. Per the paper's outline, Figure 1 should appear at the top of the method section or immediately after first reference. Verify placement in the final LaTeX.
- **Notation**: The section uses both "latent $i$" and "feature $i$" interchangeably. The glossary specifies "latent" for formal contexts. Standardize to "latent" throughout Section 3.

## Visual Element Assessment
- [x] Figures/tables match outline plan (Figure 1 and Table 1 both present as planned)
- [x] All visuals referenced before appearance (Figure 1 referenced in intro paragraph; Table 1 referenced implicitly by the taxonomy definition)
- [ ] Captions are self-explanatory --- Figure 1 caption is detailed and good; Table 1 caption could specify what "Measurement" column means (measurement method or measurement tool?)
- [ ] No text-heavy sections that need visual support --- the DAS estimation procedure (4 steps) would benefit from a small flow diagram or pseudocode block showing the logistic regression pipeline

## What Works Well

1. **The two-factor decomposition of alpha_ij is well-motivated**: Paragraph 4 of Section 3.1, which explains niche overlap (sigma_ij) and frequency imbalance (f_j / f_i) as distinct absorption risk factors, is the strongest explanatory passage in the section. It connects the mathematical definition to intuitive absorption mechanisms.

2. **The taxonomy table is a clear contribution format**: Table 1 with its four-column structure (Type, Definition, Threshold, Measurement) gives reviewers a concrete artifact to evaluate. The sequential evaluation protocol (I then II then III then None) ensuring mutual exclusivity is a clean design choice.

3. **The computational pre-filter paragraph is honest about engineering tradeoffs**: Reporting the exact pair-count reduction (3e8 to 4.4e6, 68x) and runtime (under 5 minutes on one GPU) gives readers a practical sense of feasibility. This is the kind of implementation detail that makes reproducibility easier.

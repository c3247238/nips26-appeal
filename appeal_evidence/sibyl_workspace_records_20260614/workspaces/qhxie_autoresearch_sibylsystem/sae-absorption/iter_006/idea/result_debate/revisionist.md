# Revisionist Analysis: Iteration 7 Full Experiment Results

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| **H1**: Cross-domain absorption >= 5% in 2+ knowledge hierarchies | **INCONCLUSIVE, leaning REFUTED** | City-continent 6.5%, city-language 6.6% at aggregate level superficially pass the 5% threshold, but ALL 5 domains fail control checks. Shuffled controls range from 10.3% (city-country) to 59.6% (first-letter), random controls from 9.2% (first-letter) to 34.3% (animal-class). Net signal is negative against shuffled baselines in every single domain. Absorption is concentrated in 1-2 parent categories (Asia: 21.6% for continent; English: 25.5% for language), not distributed. Without architecture-specific metric recalibration, the claimed cross-domain rates are indistinguishable from noise. | 0.20 |
| **H2**: Hierarchy-driven absorption > 80% at optimal L0 (L0~22) | **REFUTED** | Multi-L0 decomposition (n=1,196, all probe F1=1.0): at L0=22, hierarchy-driven = **1.4%**, hedging = 98.6%, reconstruction error = 0%. At L0=41: 1.8%. At L0=82: 4.9%. Only at L0=176 (with just 10 total FNs) does hierarchy-driven reach 90.0%. The pilot estimate of 96.9% was based on single-L0 testing with ~25 words/letter, where hedging classification was structurally impossible. Exactly 9 words persist as hierarchy-driven across all 4 L0 values. | 0.95 |
| **H3**: CMI negatively correlated with absorption (Spearman rho < -0.3) | **CONFIRMED at d'=10 only; dimension-dependent** | rho = -0.383 (p=0.059 marginal), Cohen's d = -0.924 (large), Mann-Whitney one-sided p = 0.023 at d'=10. Absorbed letters: mean CMI 0.649 vs non-absorbed: 0.861. **Correlation reverses at d'=20 (rho=+0.048), d'=30 (rho=+0.299), d'=50 (rho=+0.197).** Phase transition prediction: L0_crit ~ 24.7 vs empirical ~ 22.4 (10.2% error), but lambda is fit from empirical data so the scale match is partially circular. The rank-order prediction is the non-trivial test: rho = +0.333 (p=0.103, not significant). | 0.45 |
| **H4**: Unsupervised detection rho > 0.3 | **FALSIFIED** | Only 6/25 letters have any matching candidate pairs. ITAC: candidate median 1.35 vs random 1.14, Mann-Whitney not significant. Best Spearman rho = -0.104 (wrong direction). Best AUROC = 0.468 (below chance). Precision@50 = 0.0 across all pipeline variants. Pre-registered decision: report as negative result. | 0.98 |
| **H5**: Width-L0 interaction with phase transition at L0 ~ 7-14 | **PARTIALLY SUPPORTED with caveats** | Absorption vs L0 shows perfect monotonic decline: 42.9% (L0=22) -> 37.5% (L0=41) -> 14.4% (L0=82) -> 0.8% (L0=176). Spearman rho = -1.0. OLS width-L0 interaction p = 1.24e-06 (significant), but GAM interaction p = 1.0 with flexible smooths, suggesting the effect may be purely linear rather than capturing a nonlinear phase transition. Observed transition is near L0 ~ 29, not L0 ~ 7-14 as predicted. | 0.55 |
| **H6**: Hierarchy property predicts absorption (|rho| > 0.3, Bonferroni p < 0.05) | **INCONCLUSIVE (fatally underpowered)** | n=5 domains. Co-occurrence ratio rho=0.4 (p_bonf=1.0). Fan-out rho=0.2 (p_bonf=1.0). Depth rho=-0.577 (p_bonf=1.0). Bootstrap CIs span [-1.0, 1.0] for all predictors. Partial correlations controlling F1 show co-occurrence rho=1.0 and depth rho=-0.9 (p=0.037 uncorrected), but with n=5 these are statistically meaningless. | 0.10 |
| **H7**: JumpReLU bimodal vs L1 continuous (lateral inhibition bifurcation) | **REFUTED as originally stated; reframed finding** | Both architectures show bimodal distributions (JumpReLU mean BC=0.685, L1 BC=0.646, both above 0.555 threshold). KS D=0.607 (p near 0) confirms distributions differ, but in **level** not **shape**. L1 uniformly high (61-67% across layers). JumpReLU shows dramatic L0-dependent phase transition (64.3% at L0=22/65k to 1.1% at L0=176). Cross-model confound (Gemma 2B vs GPT-2 Small) cannot be separated from architecture effect. | 0.50 |

## 2. Surprise Analysis

### Surprise 1: Confound Decomposition Reversal (~97 percentage point deviation from pilot)

**Expected**: Based on pilot (iteration 6 Phase 1), ~96.9% of false negatives at L0=22 would be hierarchy-driven.

**Actual**: 1.4% hierarchy-driven, 98.6% hedging at L0=22 with 1,196 words.

**The assumption that was wrong**: The pilot's hedging classification requires checking whether the same word fails at multiple L0 values. The pilot tested only one L0 (L0=22), so the multi-L0 consistency check that defines hedging was structurally impossible. Additionally, with only 25 words per letter (total ~625), the absolute number of false negatives was small (~32), making it easy for most to pass the "hierarchy-driven" default classification. The full experiment reveals the 657 FNs at L0=22 are overwhelmingly words that also fail at L0=41 (the hedging signature). Only 9 words -- "eight," "lower," "liked," "offer," "often," and 4 others -- are hierarchy-driven FNs that persist even at L0=176 where total FNs drop to 10.

**Implication**: This is the single most consequential finding. It means the Chanin metric at low L0 is measuring the SAE's inability to fully represent any concept with limited active features, not specifically hierarchy-driven feature competition. The published 15-35% absorption rates likely conflate two distinct phenomena: hedging (~98% at low L0) and genuine absorption (~0.75%).

### Surprise 2: Universal Control Failure Across All Domains

**Expected**: Random probe control < 2%, shuffled control < 5%, per assumptions from Chanin et al. benchmarks.

**Actual**: Shuffled controls: first-letter 59.6% (improved run: 74.6%), city-continent 45.2%, animal-class 39.3%, city-language 18.0%, city-country 10.3%. Random controls: 9.2% to 34.3%.

**The assumption that was wrong**: The Chanin metric's thresholds (cosine > 0.025, magnitude gap >= 1.0) were calibrated for GPT-2 Small with L1+ReLU SAEs. Gemma Scope JumpReLU SAEs have fundamentally different activation statistics: 88.2% dead features, binary activation patterns due to hard thresholds, and unit-normalized decoder weights. In this regime, random probe directions can incidentally find "aligned" child features at rates far exceeding the expected noise floor. The improved first-letter experiment with 1,204 words and better probes (mean F1=0.817) actually shows WORSE controls: shuffled = 74.6%. This is not a bug -- the controls reveal a genuine portability problem of the metric.

**Implication**: Every absolute absorption rate measured in this study must be interpreted relative to its control baseline, and for most domains the measured rate is below or within the control range. The absorption metric as currently defined does not transfer to JumpReLU architectures without threshold recalibration.

### Surprise 3: Cross-Domain Absorption Is Specific, Not General

**Expected**: Absorption would be distributed across parent categories within each domain, with aggregate rates reflecting a general phenomenon proportional to hierarchy properties.

**Actual**: In city-language, ALL 12 absorbed instances come from English (25.5% absorption) while 17 other languages show exactly 0%. In city-continent, 8/12 come from Asia (21.6%), predominantly Chinese cities (Beijing, Shanghai, Shenzhen, Chengdu, Hangzhou), with only 4 from Europe (4.2%) and zero from 4 other continents. City-country shows 0% absorption across all 28 countries. Animal-class shows 1.4% overall, concentrated in 1 of 4 gated classes.

**The assumption that was wrong**: We assumed absorption is a general property of hierarchies that would distribute proportionally to structural features like fan-out or co-occurrence ratio. Instead, absorption is driven by specific dominant child subgroups: "Chinese cities" absorb "Asia," "English cities" absorb "English-speaking." These are cases where a very strong, high-frequency child cluster dominates the parent's activation space. This is actually consistent with the rate-distortion theory (low CMI for these pairs), but the phenomenon is far more selective than "cross-domain generalization" implies.

### Surprise 4: CMI Prediction Is Dimension-Dependent

**Expected**: CMI-absorption correlation would be robust across subspace dimensions, with higher dimensions providing more accurate estimates.

**Actual**: rho = -0.383 at d'=10, but +0.048 at d'=20, +0.299 at d'=30, +0.197 at d'=50. The predicted negative correlation holds only in the lowest-dimensional subspace.

**The assumption that was wrong**: We assumed the k-NN CMI estimator would improve with more dimensions (capturing more variance). Instead, higher dimensions introduce noise from absorption-irrelevant directions that overwhelm the signal. This could mean: (1) the genuine CMI-absorption relationship exists only in a narrow decoder-direction subspace (the "absorption-relevant manifold"), or (2) the d'=10 result is a statistical artifact of aggressively projecting onto the most correlated directions. Without a theory-driven dimension selection criterion, we cannot distinguish these interpretations.

### Surprise 5: Geometric Constant Degeneracy

**Expected**: The geometric constant c(w_P, w_C) from the rate-distortion framework would add predictive power beyond raw CMI, capturing decoder geometry effects.

**Actual**: c = sin^2(angle) with mean 0.960, std 0.021, CV 2.16%. All parent-child decoder pairs are near-orthogonal (mean cosine 0.17). CMI/c provides -2.2% improvement over raw CMI. Bootstrap 95% CI for improvement includes zero.

**The assumption that was wrong**: Modern SAE training (Gemma Scope included) unit-normalizes decoder columns and encourages decorrelation. This eliminates both sources of variability in c: the norm ||w_P||^2 is identically 1.0, and cos^2(w_P, w_C) is near 0 (so sin^2 near 1) for all pairs. The rate-distortion threshold simplifies from lambda / (CMI * c) to approximately lambda / CMI. This is actually a positive simplification for the theory, but it means the geometric term in the formal statement is vacuous for practical SAEs.

## 3. Mental Model Revision

**We assumed absorption is a prevalent, hierarchy-driven failure mode that the Chanin metric reliably measures. The data forces three fundamental revisions to this mental model.**

First, **what the metric calls "absorption" at low L0 is overwhelmingly hedging**, not competitive exclusion. When an SAE has only ~22 active features per token, it spreads activation thinly across many directions. Probes detect numerous "false negatives" where the predicted feature fires weakly or not at all, but these are not cases where a child feature has displaced the parent. They are cases where the entire representation is too compressed to represent any single concept strongly. Only 9/1,195 tested words (0.75%) exhibit genuine hierarchy-driven absorption that persists across all sparsity levels. The field's understanding of absorption as a 15-35% phenomenon may be an artifact of measuring at low L0 without multi-L0 confound decomposition.

Second, **the Chanin metric does not transfer to JumpReLU architectures without recalibration**. The metric's cosine and magnitude gap thresholds were implicitly calibrated for the activation statistics of GPT-2 Small + L1+ReLU SAEs. On Gemma Scope JumpReLU SAEs with unit-normalized decoders and 88% dead features, the noise floor (measured by shuffled/random controls) exceeds or matches the signal across all tested domains. This is not an implementation error -- it reveals that the metric contains hidden assumptions about activation distribution shape and feature density.

Third, **where genuine absorption occurs, it is driven by specific dominant child subgroups, not by general hierarchy structure**. Chinese cities absorb "Asia." English cities absorb "English-speaking." These are cases where a high-frequency, tightly clustered set of children has such strong child features that the parent feature's unique information (its CMI) is genuinely low. The rate-distortion theory is directionally correct in explaining this: absorption is information-theoretically cheap when the parent adds little beyond the child. But this makes absorption a feature-pair-specific phenomenon, not a domain-level one.

## 4. Reframing Test

**If we knew these results before designing the study, would we frame the research question the same way?**

No. The original framing -- "Does feature absorption generalize across knowledge hierarchy domains?" -- presupposes that (1) the absorption metric reliably measures what it claims, and (2) cross-domain generalization is the interesting axis of variation. Both presuppositions fail.

**Revised research question**: "How much of what the standard absorption metric measures is genuine hierarchy-driven feature competition versus low-sparsity hedging artifact, and what determines the metric's reliability across SAE architectures?"

This reframing is stronger for several reasons:

1. **The confound decomposition is the genuinely novel contribution.** Nobody has decomposed measured absorption into hedging vs hierarchy-driven components across L0 values. The finding that 98.6% is hedging at low L0 is a critical methodological contribution that will change how future work interprets absorption measurements.

2. **The control failure is a primary finding, not an embarrassment.** The metric's non-portability to JumpReLU SAEs is important for the field to know. Framing it as "metric validation" makes negative controls a positive result.

3. **The 9 genuinely absorbed words become ground truth.** Rather than reporting noisy aggregate rates, these 9 words (identified by multi-L0 consistency) provide a gold-standard dataset for future absorption research. The absorbing features are identified at each L0, enabling activation patching validation.

4. **Rate-distortion theory remains viable** but scoped to the specific question: "Among the feature pairs where genuine absorption occurs, does lower CMI predict it?" The d'=10 result (rho = -0.383) is a starting point, but the dimension-dependence must be transparently reported as a caveat.

The cross-domain question could become a secondary contribution: "Absorption in knowledge hierarchies is concentrated in specific dominant child subgroups (Chinese cities in Asia, English cities in language), not distributed across the domain." This is itself an interesting empirical finding about the structure of feature competition.

## 5. New Hypothesis Generation

### NH1: Hedging Fraction Is a Predictable Function of L0

**Statement**: The fraction of false negatives classified as hedging follows pct_hedging ~ 1 - k/L0 (or a power law variant), reaching near-zero only when L0 exceeds the number of semantically independent features in the input (approximately the number of "concepts" the probe measures).

**Falsification**: R^2 < 0.85 for the best-fit parametric model across 6+ L0 values, or the hedging fraction at intermediate L0 (L0=60) deviates from the prediction by more than 15 percentage points.

**Concrete experiment**: Run confound decomposition at L0 = {10, 15, 22, 41, 60, 82, 120, 176} with the 1,196-word vocabulary and F1=1.0 probes. Fit both linear-in-1/L0 and power law models. Estimate the L0 at which hedging drops below 50% (the "hedging crossover"), and compare with the number of first-letter features that show non-zero activation at each L0.

### NH2: Activation Patching Confirms the 9 Genuinely Absorbed Words

**Statement**: For the 9 words identified as hierarchy-driven FNs across all 4 L0 values (eight, lower, liked, offer, often, plus wrong, avoid, agree, and adopt), zeroing the identified absorbing child features causes the parent "starts with X" feature to recover (fire above the JumpReLU threshold) in at least 7/9 cases.

**Falsification**: Parent recovery occurs in fewer than 5/9 cases, suggesting the "hierarchy-driven" classification is itself unreliable and these are more complex failure modes.

**Concrete experiment**: For each of the 9 words at L0=82 (where the absorbing features are identified in the decomposition data), run a forward pass, zero the absorbing child feature activations in the SAE encoding, re-encode, and check whether the parent "starts with [letter]" feature now fires. This is the definitive causal test of absorption.

### NH3: The Chanin Metric's Noise Floor Scales with Dead Feature Fraction

**Statement**: The shuffled-label control rate (which measures the metric's false positive rate) is positively correlated with the SAE's dead feature fraction across architectures, because higher dead feature density concentrates activation on fewer features, increasing the probability that any active feature incidentally satisfies the cosine/magnitude criteria for a random probe direction. Spearman rho > 0.5 across 8+ SAE configurations spanning JumpReLU (Gemma Scope) and L1+ReLU (SAEBench GPT-2).

**Falsification**: rho < 0.3 between dead feature fraction and shuffled-control absorption rate, or L1+ReLU SAEs with similar dead feature fractions show comparable shuffled-control rates.

**Concrete experiment**: Compute shuffled-control absorption rates on 4 Gemma Scope JumpReLU configs (L0=22, 41, 82, 176) and 4 SAEBench GPT-2 L1+ReLU configs at comparable effective L0 values. Measure dead feature fraction for each. Correlate.

## Summary of Belief Updates

| Prior Belief | Updated Belief | Update Magnitude |
|---|---|---|
| Absorption is a 15-35% phenomenon on any SAE | Genuine hierarchy-driven absorption is ~0.75% (9/1,195); the rest is hedging at low L0 | **Massive** |
| Confound decomposition shows ~97% hierarchy-driven at L0=22 | 1.4% hierarchy-driven; pilot was structurally incapable of detecting hedging (single L0, small n) | **Massive** |
| Cross-domain absorption generalizes proportionally to hierarchy structure | Absorption is concentrated in specific dominant child subgroups (Chinese cities/English cities), not distributed | **Large** |
| The Chanin metric reliably transfers across SAE architectures | Metric noise floor on JumpReLU SAEs exceeds signal in all tested domains; shuffled controls 10-75% | **Large** |
| CMI robustly predicts absorption across dimensions | CMI predicts absorption only at d'=10; relationship reverses or disappears at d'>=20 | **Moderate** |
| Geometric constant c modulates absorption threshold | c is degenerate (near-constant 0.96) for normalized SAEs; rate-distortion simplifies to lambda/CMI | **Moderate** |
| JumpReLU produces bimodal, L1 produces continuous absorption distributions | Both are bimodal; JumpReLU shows dramatic L0-dependent phase transition in severity, L1 is uniformly high | **Moderate** |
| Unsupervised detection might work with refined pipeline | Confirmed negative result: AUROC below chance, precision@50 = 0, best rho = -0.104 (wrong direction) | **Small** (pre-registered as at-risk) |
| Hierarchy properties (fan-out, co-occurrence) predict absorption across domains | Cannot be tested meaningfully with n=5 domains; bootstrap CIs span full range | **Small** (design failure, not finding) |

## Core Recommendation

The strongest intellectual contribution of this work is the **confound decomposition and metric validation**, not cross-domain generalization. The paper should pivot to:

1. **Primary claim**: The standard absorption metric conflates hedging with genuine hierarchy-driven absorption. At low L0, ~98.6% of detected "absorption" is hedging artifact.
2. **Primary method**: Multi-L0 confound decomposition as a diagnostic tool, with the 9 genuinely absorbed words as validated ground truth.
3. **Theoretical contribution**: Rate-distortion framework correctly predicts the direction of genuine absorption (lower CMI -> more absorption), but only in a restricted subspace, and the geometric term degenerates for normalized SAEs.
4. **Negative results**: (a) The metric does not port to JumpReLU SAEs without recalibration; (b) unsupervised detection fails; (c) cross-domain claims are unsubstantiated when controls are applied.
5. **Next critical experiment**: Activation patching on the 9 absorbed words to provide causal validation before writing claims about "genuine absorption."

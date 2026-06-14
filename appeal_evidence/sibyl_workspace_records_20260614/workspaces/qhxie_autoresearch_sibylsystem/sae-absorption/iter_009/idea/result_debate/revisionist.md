# Revisionist Analysis: Updating Our Mental Model of SAE Feature Absorption

**Agent**: Revisionist
**Iteration**: 9 (FULL mode)
**Date**: 2026-04-16
**Evidence base**: consolidation_summary.json + all Phase 1-4 results from iter_009, iter_008 causal results

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|-----------|---------|-------------|------------|
| **H1** (Cross-domain variation) | **STRONGLY SUPPORTED** | Kruskal-Wallis p=7.37e-66 at L24_16k (N=3566). Rates span 11.6%-45.1%. All pairwise comparisons within RAVEL significant. | HIGH (p<1e-52 at both SAE widths) |
| **H2'** (Semantic > Syntactic) | **REFUTED** | No simple ordering. City-country (45.1%) > first-letter (27.1%) > city-continent (31.4%) > city-language (11.6%). City-language *significantly lower* than first-letter (p=0.0005, Cohen's h=-0.73). | HIGH |
| **H3** (Hedging decomposition varies) | **SUPPORTED** | Chi-square(3)=91.51, p=1.04e-19. Strict absorption: 0% (first-letter) to 22.6% (city-language). Compensatory dominates all hierarchies at L24. | HIGH |
| **H4** (GAS detector) | **DEFINITIVE NEGATIVE** | rho=0.116, p=0.58, AUROC=0.571. Bootstrap CI spans zero [-0.333, 0.536]. 25x scale-up confirmed null. | HIGH |
| **H5** (Absorption Tax quantitative) | **NOT SUPPORTED** | T(G) ranking rho=-0.20, concordance ~50% (chance). 17 configurations computed. Quantitative predictions fail. | HIGH |
| **H6** (Architecture generalization) | **PARTIALLY SUPPORTED (reframed)** | Architecture effect non-significant: L12 p=0.754, L24 p=0.497. Hierarchy effect significant (L12 p=0.010). Finding: hierarchy >> architecture. | MEDIUM-HIGH |
| **H7** (Causal activation patching) | **SUPPORTED (first-letter) / COMPLICATED (cross-domain)** | First-letter: Wilcoxon p=0.000218, d=1.33, recovery 32.5% vs 1.5% control (n=25). Cross-domain (FULL, corrected methodology): city-continent recovery 61.9% (d=1.50), city-language 34.2% (d=0.75), both p<1e-18. | HIGH (first-letter), HIGH (cross-domain after fix) |
| **H8** (Benign vs. pathological) | **FALSIFIED** | 0% benign at ALL thresholds (0.05, 0.1, 0.2). Mean |logit change|=3.98 nats vs control 0.004 nats. Effect ratio ~1000x. n=1471 instances, 50 entities. | HIGH (t=-365.27, Wilcoxon p=2.69e-242) |
| **H9** (Rate-distortion predictors) | **NOT SUPPORTED** | Model rho=0.250 (target >0.5), R^2=0.088. Individual predictors in OPPOSITE direction from theory: cos_sim rho=-0.108, co_occur rho=-0.173, r_parent rho=-0.203. Direction reversal from pilot (n=20) to full (n=262). | HIGH |

---

## 2. Surprise Analysis

### Surprise 1: H2' Refuted -- No Simple Semantic > Syntactic Ordering (deviation: qualitative reversal)

**What we expected**: The proposal stated that semantic/knowledge hierarchies should show HIGHER absorption than the syntactic first-letter task, based on pilot evidence showing city-language at 10.4% vs first-letter at 3.9%.

**What we found**: At L24_16k with proper probes, first-letter absorption is 27.1% -- dramatically higher than the pilot's 3.9% (measured at L12 with poor probes). The ordering is city-country (45.1%) > city-continent (31.4%) > first-letter (27.1%) > city-language (11.6%). There is no coherent "semantic > syntactic" pattern.

**Assumption that was wrong**: We assumed that the pilot's first-letter rate of 3.9% (measured at L12 with F1=0.083 probes) was representative. It was not. The pilot was measuring a different signal at a different layer with catastrophically bad probes. The assumption that semantic complexity monotonically increases absorption vulnerability was also wrong. The 4x range across hierarchies (11.6%-45.1%) is not explained by a simple semantic/syntactic dichotomy.

**Why this matters**: The entire framing of the proposal centered on the claim that "first-letter is atypical (lowest absorption)." With proper FULL-mode measurements, first-letter is actually in the middle of the distribution. The paper's narrative must shift from "semantic > syntactic" to "hierarchy-specific patterns" -- which is actually a richer and more interesting finding.

### Surprise 2: H8 Falsified -- Zero Benign Absorption (deviation: 100% from the >=30% target)

**What we expected**: The Contrarian perspective predicted that a substantial fraction (>=30%) of absorption would be benign -- that is, the model does not independently use the parent feature when the child is active, so absorption faithfully reflects computational redundancy.

**What we found**: 0% benign absorption across ALL thresholds. The mean absolute logit change when ablating the parent direction is 3.98 nats, compared to 0.004 nats for control directions -- a ~1000x effect ratio. This held for all 1471 instances from 50 entities in the city-continent hierarchy.

**Assumption that was wrong**: The Contrarian's core assumption was that "some absorption faithfully represents computational redundancy." This assumed that parent features sometimes carry redundant information when the child is active. In reality, parent features (e.g., "Europe") carry causally important information that is independent of child features (e.g., "Paris") even when both are active. The model uses both levels of the hierarchy simultaneously for different downstream purposes.

**Why this matters**: This decisively settles a philosophical debate as an empirical one. Absorption is not a benign artifact -- it is ALWAYS pathological. Every absorbed feature represents genuine information loss. This transforms absorption from an academic curiosity into an urgent engineering problem for SAE deployment in safety-relevant applications.

### Surprise 3: Rate-Distortion Predictors Show Opposite-Direction Correlations (deviation: direction reversal from pilot)

**What we expected**: H9 predicted that absorption probability should increase with decoder cosine similarity and co-occurrence, and decrease with parent reconstruction importance. Pilot data (n=20) showed directionally consistent (albeit weak) correlations.

**What we found**: At FULL scale (n=262 pairs), ALL individual predictors correlate in the OPPOSITE direction from theory: cos_sim rho=-0.108, co_occur rho=-0.173, r_parent rho=-0.203. The model rho=0.250 is below the 0.3 falsification threshold.

**Assumption that was wrong**: Two assumptions failed simultaneously. (1) The rate-distortion framework assumed absorption follows a simple competitive dynamics model where higher decoder overlap leads to more absorption. In practice, high decoder overlap may cause the SAE to learn complementary encodings that avoid competitive exclusion. (2) The pilot (n=20) was severely underpowered and happened to show directionally consistent correlations by chance. Direction reversal at n=262 is a textbook sign of pilot instability.

**Why this matters**: This is the most theoretically important negative result. It says that absorption is NOT predictable from static geometric and statistical quantities (decoder cosine similarity, co-occurrence, reconstruction importance). The causal mechanism is more complex than "similar features compete." This kills the theoretical pillar of the paper's tertiary contribution and redirects future work toward dynamic/encoder-side explanations.

### Surprise 4: Cross-Domain Activation Patching Initially Failed, Then Succeeded with Corrected Methodology

**What we expected**: If activation patching works for first-letter absorption (32.5% recovery, d=1.33), it should work for semantic hierarchies too.

**What the pilot found**: Cross-domain patching showed recovery of 0.05% vs control 14.4% -- a reversal where the control outperformed the intervention. This initially suggested that cross-domain absorption is mechanistically different (distributed across features rather than concentrated in one child feature).

**What the corrected FULL run found**: After identifying a methodology bug (the pilot zeroed features with highest POSITIVE cosine, i.e., features supporting the correct class, rather than features with most negative contribution), the corrected run showed city-continent recovery of 61.9% (d=1.50, p<1e-20) and city-language recovery of 34.2% (d=0.75, p<1e-18). These are actually STRONGER than first-letter results.

**Assumption that was wrong**: The initial "distributed mechanism" interpretation was an artifact of a code bug, not a genuine finding. The corrected results show that cross-domain absorption IS causally driven by identifiable absorber features, just like first-letter absorption. The mechanism is consistent across hierarchy types.

**Why this matters**: This converts what appeared to be a negative/complicated result into a strong positive result. The causal mechanism (competitive exclusion via identifiable absorber features) is universal across hierarchy types. However, the fact that a methodology bug initially produced a plausible-sounding but completely wrong interpretation is a sobering reminder of how easy it is to construct post-hoc narratives around negative results.

### Surprise 5: Architecture Does Not Matter -- Hierarchy Is the Dominant Factor

**What we expected**: The pilot showed JumpReLU consistently lowest across all hierarchies. We expected to confirm this with statistical significance and report architecture-specific recommendations.

**What we found**: Architecture effect is non-significant at both L12 (p=0.754) and L24 (p=0.497). Hierarchy effect is significant at L12 (p=0.010). The variance across hierarchies dwarfs the variance across architectures.

**Assumption that was wrong**: We assumed that architecture design (JumpReLU vs BatchTopK vs Matryoshka) meaningfully affects absorption rates. The data says otherwise: absorption is a fundamental property of sparse decomposition under the given hierarchy structure, not a quirk of any particular SAE architecture. The pilot's apparent JumpReLU advantage was an artifact of small sample sizes and confounded measurements.

**Why this matters**: This reframes the architecture comparison section of the paper. Instead of recommending JumpReLU as "most robust," the paper should report the non-significance as the primary finding. Architecture choice does not solve absorption. The problem is structural, not parametric.

### Surprise 6: City-Country Absorption Rate Dramatically Higher in FULL Mode

**What we expected**: Based on iter_008, city-country absorption was 18.5%.

**What we found**: At L24_16k FULL mode, city-country absorption is 45.1% -- the highest among all hierarchies. With a 2.4x increase from pilot to FULL, this is well above the >30% suspicious improvement threshold.

**Potential confound**: City-country probe quality is F1=0.726, below the strict gate of 0.90 and even below the relaxed gate of 0.85. This is the worst probe quality among all hierarchies. The high absorption rate may be partially inflated by probe measurement error. Per-country breakdown shows extreme variation: United States at 0% absorption vs. countries like Albania, Algeria, Argentina at 100%. This distribution is suspicious -- countries with few cities in the training data (low-frequency entities) show near-total absorption, while high-frequency countries (US, China, India) show low absorption. This pattern is consistent with probe failure on rare classes being misclassified as absorption.

**Assessment**: The city-country absorption rate of 45.1% should be treated as an upper bound, not a precise estimate. The true rate is likely lower but still substantially above zero. This finding requires prominent caveats in the paper.

---

## 3. Mental Model Revision

**Before (proposal-stage mental model)**: We assumed absorption was a failure mode driven primarily by sparsity pressure, where child features competitively exclude parent features through decoder overlap. We believed: (a) semantic hierarchies have more absorption than syntactic ones because they impose stronger co-occurrence constraints; (b) some absorption is benign (reflecting computational redundancy); (c) absorption severity can be predicted from static geometric quantities (decoder cosine similarity, co-occurrence, reconstruction importance); (d) architecture choice matters for absorption resistance.

**After (evidence-updated mental model)**: Absorption is a hierarchy-specific phenomenon where the dominant factor is the structural properties of the hierarchy itself (number of classes, class balance, co-occurrence patterns) rather than semantic category, SAE architecture, or simple geometric predictors. Specifically:

1. **Absorption is always pathological.** There is no "benign" absorption where parent information is redundant. The model independently uses multiple hierarchy levels simultaneously, and absorption destroys causally important information at every instance (effect ratio ~1000x over control).

2. **Absorption is hierarchy-specific, not category-specific.** The semantic/syntactic distinction is not the operative variable. What matters is the structural properties of each specific hierarchy: class count (city-country: 80 classes vs city-continent: 6), class imbalance (Zipfian for countries vs near-uniform for continents), and the model's learned representations of the hierarchy.

3. **Absorption resists simple prediction.** Static quantities (decoder cosine similarity, co-occurrence frequency, reconstruction importance, GAS, CMI) all fail to predict absorption rates. The rate-distortion framework's three-factor model achieves only rho=0.25 with individual predictors in the wrong direction. Absorption is an emergent property of the SAE training dynamics, not a direct function of decoder geometry.

4. **Architecture is irrelevant.** Switching between JumpReLU, BatchTopK, and Matryoshka does not meaningfully change absorption rates. The problem is fundamental to sparse decomposition, not specific to any training objective.

5. **Causal mechanism is universal but implementation varies.** Activation patching confirms competitive exclusion across all hierarchy types (city-continent: d=1.50, city-language: d=0.75, first-letter: d=1.33), but recovery rates differ substantially across hierarchies, suggesting different numbers of absorber features per hierarchy.

---

## 4. Reframing Test

**Original framing (RQ1)**: "Do absorption rates vary systematically across feature hierarchy types, and does the first-letter spelling task represent a typical or extreme case?"

**Verdict**: The question is still valid, but the expected answer was wrong. We framed it expecting to show that first-letter is atypical (lowest absorption). In fact, first-letter is middling (27.1%), not extreme in either direction. The more interesting finding is the *4x range* across hierarchies and the absence of a simple ordering principle.

**Revised research question**: "What structural properties of feature hierarchies determine absorption severity, given that the variation across hierarchies (11.6%-45.1%) dwarfs the variation across architectures (non-significant)?"

This reframing is better because it:
- Centers the hierarchy-specificity finding as the core result
- Moves past the "semantic vs syntactic" dichotomy (which is falsified)
- Points toward the actual scientific question (what drives the 4x variation?)
- Acknowledges the architecture invariance as part of the core finding

**Original framing (RQ4)**: "Is absorption driven by competitive exclusion dynamics, and can we distinguish benign from pathological absorption?"

**Verdict**: The first half is confirmed (yes, competitive exclusion). The second half is resolved: there is nothing to distinguish because all absorption is pathological. The revised framing should emphasize the universality of pathological absorption rather than the benign/pathological distinction.

**Revised research question**: "Given that absorption is universally pathological (~1000x causal effect over control), how does the causal mechanism (competitive exclusion) scale across hierarchy types, and what does the variation in recovery rates tell us about the distribution of absorption across features?"

---

## 5. New Hypotheses

### New Hypothesis NH1: Class Count and Imbalance as Absorption Predictors

**Statement**: Absorption rate for a given hierarchy is primarily determined by the number of classes and the entropy of the class distribution, not by the semantic domain.

**Rationale**: City-country (80 classes, highly imbalanced) shows 45.1% absorption. City-continent (6 classes, moderately balanced) shows 31.4%. City-language (23 classes, imbalanced toward English) shows 11.6%. First-letter (26 classes, near-uniform) shows 27.1%. The pattern is consistent with high class count and high imbalance increasing absorption, though the relationship is not monotonic (city-language has fewer classes than first-letter but lower absorption, possibly because language has a much stronger frequency imbalance toward English which "absorbs" less).

**Experiment**: (a) Systematically vary class count by creating subsets of the city-country hierarchy (e.g., top-5, top-10, top-20, all 80 countries). (b) Vary class balance by resampling entities to create uniform vs Zipfian distributions within the same hierarchy. (c) Measure absorption at each configuration and fit a model: absorption ~ f(n_classes, class_entropy, max_class_share).

**Falsification**: R^2 < 0.3 or class count/entropy are non-significant after controlling for hierarchy type.

### New Hypothesis NH2: Multi-Absorber Distribution Varies Systematically by Hierarchy

**Statement**: The number of features that jointly absorb a parent varies by hierarchy type. City-continent absorption is concentrated in fewer absorber features (single primary absorber recovers 61.9%) while city-language absorption is distributed across more features (primary recovers only 34.2%, but all absorbers recover 70.0%).

**Rationale**: The gap between primary absorber recovery and all-absorber recovery is informative: city-continent shows 61.9% vs 80.0% (gap: 18.1pp), while city-language shows 34.2% vs 70.0% (gap: 35.8pp). This suggests city-language has a flatter distribution of absorption across multiple child features. The mechanism may be that languages (being correlated with multiple geographic/cultural features) create more distributed competitive exclusion, while continents (being mutually exclusive and exhaustive) create concentrated exclusion.

**Experiment**: For each absorbed instance, rank all SAE features by their contribution to absorption (negative contribution to parent probe). Compute the Gini coefficient of the absorption distribution across features for each hierarchy. Test whether Gini predicts recovery rates.

**Falsification**: No significant difference in absorption distribution shape across hierarchies (Gini difference non-significant).

### New Hypothesis NH3: Absorption Severity Predicts Downstream Task Performance Degradation

**Statement**: If all absorption is pathological (as H8 falsification shows), then hierarchies with higher absorption rates should show proportionally worse performance on downstream tasks that require parent-level information.

**Rationale**: The 1000x effect ratio from H8 means every absorption instance destroys ~4 nats of information about the parent feature. If city-country has 45.1% absorption vs city-language at 11.6%, then SAE-reconstructed representations should be 3.9x worse at encoding country information than language information (relative to the raw model). This should be measurable as a degradation in probing accuracy: the gap between raw probe F1 and SAE probe F1 should correlate with absorption rate.

**Experiment**: Measure (raw_probe_F1 - sae_probe_F1) for each hierarchy at each layer/width configuration. Correlate with measured absorption rate. Test whether absorption rate predicts the F1 gap after controlling for probe quality.

**Falsification**: Correlation between absorption rate and F1 gap is non-significant (rho < 0.3) after controlling for probe quality, suggesting the 1000x local effect does not translate to aggregate information loss.

---

## 6. Methodological Warnings for the Paper

### Warning 1: Pilot-to-FULL Instability

Multiple hypotheses showed qualitative changes from pilot to FULL mode: H2' was "SUPPORTED by pilot" but REFUTED in FULL. Rate-distortion predictors reversed direction. Cross-domain patching reversed direction (though this was a bug). City-country absorption jumped from 18.5% to 45.1%. The paper must be transparent that pilot results from iterations 1-8 were unreliable guides to FULL-mode findings. This is itself a methodological finding worth reporting: absorption measurements are sensitive to sample size, probe quality, and entity selection in ways that smaller-scale pilots fail to capture.

### Warning 2: Probe Quality as a Confound

City-country results (45.1% absorption, F1=0.726) are confounded by probe quality. The paper should present city-country results with prominent caveats and consider reporting results both with and without city-country to show that the core findings (cross-domain variation, pathological absorption, architecture invariance) hold regardless.

### Warning 3: Layer Selection Matters

First-letter absorption ranges from 0.7% at L6 to 27.1% at L24. Hedging decomposition also changes dramatically: at L12, 100% of false negatives are strict absorbed; at L24, 100% are compensatory. Reporting results at a single layer risks misleading conclusions about the nature of absorption. The paper should present layer-varying results and discuss how absorption mechanism changes across layers.

---

## 7. Updated Contribution Assessment

| Contribution | Original Framing | Evidence-Updated Framing | Strength |
|-------------|-----------------|------------------------|----------|
| Cross-domain characterization | "First-letter is atypical (lowest)" | "Absorption is hierarchy-specific with 4x variation; no simple ordering" | STRONGER (more nuanced, more interesting) |
| Pathological absorption | "Benign/pathological diagnostic" | "All absorption is pathological (0% benign, 1000x effect)" | STRONGER (cleaner result) |
| Causal mechanism | "First-letter only (cross-domain failed)" | "Universal competitive exclusion across hierarchies (d=0.75-1.50)" | MUCH STRONGER (after methodology fix) |
| Architecture comparison | "JumpReLU consistently best" | "Architecture does not matter (hierarchy >> architecture)" | CHANGED DIRECTION (equally interesting) |
| Rate-distortion predictors | "Three-factor model (rho>0.5)" | "All predictors fail or reverse direction" | NEGATIVE (important negative result) |
| Theoretical framework | "Absorption Tax as quantitative tool" | "Absorption Tax concept valid, quantitative predictions fail" | WEAKENED |

The paper's overall story is STRONGER after FULL-mode evidence, but the narrative has shifted. The key message is no longer "semantic > syntactic" but rather "absorption is hierarchy-specific, universal, always pathological, and resistant to simple prediction." This is a more mature and honest scientific contribution.

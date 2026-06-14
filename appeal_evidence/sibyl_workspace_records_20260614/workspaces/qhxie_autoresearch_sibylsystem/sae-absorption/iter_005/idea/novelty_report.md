# Novelty Report: Iteration 5 Candidates

## Date: 2026-04-14
## Assessed by: sibyl-novelty-checker

---

## Executive Summary

The front-runner candidate (`cand_causal_chain`) and its integrated contributions have **high overall novelty** with some important qualifications. No paper has performed formal mediation analysis, Rosenbaum sensitivity bounds, or epidemiological causal methods on the absorption-quality relationship. No paper has measured feature absorption on RAVEL knowledge hierarchies. No paper has constructed a formal 2D absorption scaling surface with GAM interaction testing. However, the rapidly evolving landscape (Matryoshka SAEs at ICML 2025, OrtSAE, KronSAE, Feature Hedging) means the window for some contributions is narrowing.

**Overall novelty assessment: HIGH** (all candidates with "proceed" recommendation score >= 7)

---

## Candidate 1: `cand_causal_chain` (Front-Runner)

**Title:** Absorption-Quality Causal Chain via Epidemiological Methods

**Novelty Score: 8/10**

### Core Contribution Claims Checked

#### Claim 1: First rigorous confound control (L0 as covariate) for absorption-quality correlation

**Search queries:** "feature absorption partial correlation controlling L0 sparsity width confound causal", "feature absorption confound L0 width correlation quality metric SAE"

**Prior art found:**
- **Chanin et al. (2024)** [arXiv:2409.14507] show absorption varies with width and L0 but do NOT perform formal confound analysis or partial correlations controlling for L0. They show figures (e.g., Fig 7b) plotting absorption rate vs L0 at different widths, but these are descriptive, not causal.
- **SAEBench (Karvonen et al., 2025)** [arXiv:2503.09532] benchmarks absorption across width/L0 combinations (3 widths x 6 sparsities) but presents descriptive comparisons, not partial correlations or mediation analysis.
- **Feature Hedging (Chanin et al., 2025)** [arXiv:2505.11756] analyzes the interaction between width and absorption via a different mechanism (hedging) but does not perform causal confound control.
- **Gao et al. (2024, ICLR 2025)** [arXiv:2406.04093] establish scaling laws for SAE quality vs width and L0 but do not analyze absorption specifically as a mediator.

**Classification:** No exact match. No partial overlap on formal confound control methodology.
**Severity:** related_work -- these papers provide the empirical data that motivates the confound analysis but none performs it.

#### Claim 2: Mediation analysis (L0 -> Absorption -> Quality causal chain)

**Search queries:** "SAE feature absorption mediation analysis causal inference quality", "mediation analysis Rosenbaum sensitivity bounds SAE sparse autoencoder"

**Prior art found:**
- Causal mediation analysis is widely used in epidemiology (Tchetgen Tchetgen & Shpitser, 2012) and has been applied to mechanistic interpretability in a different sense: causal mediation analysis for identifying circuit components (Vig et al., 2020; Meng et al., 2022). However, these analyze which MODEL components mediate predictions, not which SAE HYPERPARAMETERS mediate quality.
- **CircuitLasso** and **Sparse Feature Circuits (Marks et al., 2024)** use causal mediation to study feature circuits, not SAE evaluation.

**Classification:** No exact match. The term "mediation analysis" appears in the SAE literature but refers to an entirely different application (circuit analysis vs. hyperparameter confound control).
**Severity:** related_work -- the epidemiological mediation framework applied to SAE evaluation hyperparameters is genuinely novel.

#### Claim 3: Rosenbaum sensitivity bounds for SAE evaluation

**Search queries:** "epidemiological methods mechanistic interpretability propensity matching Bradford Hill SAE", "Rosenbaum sensitivity bounds SAE evaluation"

**Prior art found:**
- Zero relevant results combining Rosenbaum bounds with SAE evaluation or any mechanistic interpretability topic.
- Rosenbaum bounds and propensity matching are standard in epidemiology (Rosenbaum & Rubin, 1983) but have never been applied to SAE quality evaluation.

**Classification:** No match.
**Severity:** N/A -- genuinely novel methodological transfer.

#### Claim 4: Epidemiological causal methods (Bradford Hill criteria, etc.) for SAE evaluation

**Search queries:** "epidemiological methods mechanistic interpretability propensity matching Bradford Hill SAE"

**Prior art found:**
- Zero results. Bradford Hill criteria have been updated for modern molecular epidemiology (Fedak et al., 2015) but never applied to ML model evaluation, let alone SAE quality assessment.

**Classification:** No match.
**Severity:** N/A -- genuinely novel.

### Collisions Summary
| Prior Work | Overlap | Severity |
|---|---|---|
| Chanin et al. (2024) NeurIPS | Foundational absorption measurement; descriptive width/L0 dependence | related_work |
| SAEBench (Karvonen et al., 2025) | Absorption metric across width/L0 grid | related_work |
| Gao et al. (2024) ICLR 2025 | SAE scaling laws (width, L0) | related_work |
| Feature Hedging (Chanin et al., 2025) | Width-absorption interaction via hedging mechanism | related_work |
| Matryoshka SAEs (Bussmann et al., 2025) ICML 2025 | Architecture that reduces absorption | related_work |

### Recommendation: **PROCEED**

**Differentiation notes:** The key novelty is methodological: transplanting epidemiological causal inference (mediation analysis, Rosenbaum bounds, propensity matching, Bradford Hill criteria) to SAE evaluation. This is a genuinely untouched methodological space. The empirical question (does absorption causally mediate quality or is it a confound with L0?) has been raised informally (e.g., SAEBench observes inverse scaling of absorption with width) but never formally tested with appropriate statistical tools.

**Risk:** The methodological contribution is strong but depends on the 54-SAE dataset having sufficient statistical power for stratified and mediation analyses.

---

## Candidate 2: `cand_phase_diagram` (Backup A)

**Title:** Absorption Phase Diagram: Unified Scaling Laws for Feature Recovery Under Hierarchy

**Novelty Score: 7/10**

### Core Contribution Claims Checked

#### Claim 1: First 2D absorption phase surface in (log(width), log(L0)) space

**Search queries:** "sparse autoencoder absorption scaling surface GAM interaction width sparsity", "SAE absorption rate phase transition scaling law width sparsity empirical map"

**Prior art found:**
- **Chanin et al. (2024)** Figures 7b/9b/9c show absorption rate vs L0 at different widths, providing scattered 2D views, but NOT a formal surface fit, NOT a GAM with interaction testing, and NOT a contour plot.
- **SAEBench (2025)** evaluates 3 widths x 6 sparsities = 18 grid points per architecture. The data EXISTS for partial surface construction but no formal 2D surface analysis has been published.
- **Understanding SAE Scaling with Feature Manifolds (2025)** [arXiv:2509.02565] studies SAE scaling regimes but focuses on reconstruction, not absorption specifically.
- **Universal Scaling Laws of Absorbing Phase Transitions (Tamai et al., 2025, PRR)** -- this is about signal propagation phase transitions in neural network depth, NOT about SAE feature absorption. Completely different topic despite superficial keyword overlap.

**Classification:** partial_overlap with SAEBench (the data grid exists but no formal surface analysis) and Chanin et al. (descriptive plots exist but no formal regression/interaction testing).
**Severity:** partial_overlap

#### Claim 2: GAM interaction term testing for joint width-L0 effect on absorption

**Search queries:** "sparse autoencoder absorption scaling surface GAM generalized additive model interaction"

**Prior art found:**
- Zero results combining GAMs with SAE absorption analysis.

**Classification:** No match.

#### Claim 3: Three-regime framework (hedging, absorption, recovery)

**Prior art found:**
- **Feature Hedging (Chanin et al., 2025)** identifies the hedging regime explicitly. The hedging-vs-absorption tradeoff is described: narrow SAEs suffer hedging, wide SAEs suffer absorption.
- The "recovery" regime at high width + high L0 is implicit in SAEBench results (Matryoshka SAEs improve at higher L0) but not formally named or characterized.

**Classification:** partial_overlap -- the hedging regime is already named and characterized.
**Severity:** partial_overlap

### Collisions Summary
| Prior Work | Overlap | Severity |
|---|---|---|
| Chanin et al. (2024) Figs 7b/9b/9c | Descriptive 2D plots (not formal surface) | partial_overlap |
| SAEBench (2025) 3x6 grid | Data exists for surface, no formal analysis | partial_overlap |
| Feature Hedging (Chanin et al., 2025) | Hedging regime already named | partial_overlap |
| SAE Scaling with Manifolds (2025) | Different scaling focus (reconstruction, not absorption) | related_work |

### Recommendation: **PROCEED with differentiation**

**Differentiation notes:** The formal GAM analysis with interaction term testing goes well beyond the existing descriptive plots. The novelty rests on: (a) formal statistical surface fitting rather than descriptive scatter plots, (b) interaction term test for nonlinearity, (c) gradient analysis for phase boundary detection, (d) scaling to 200+ SAEs (vs SAEBench's 18 grid points per architecture). The three-regime naming should acknowledge Feature Hedging explicitly for the hedging regime.

---

## Candidate 3: `cand_cross_domain` (Backup B)

**Title:** Cross-Domain Feature Absorption: From Spelling to Knowledge Hierarchies

**Novelty Score: 8/10**

### Core Contribution Claims Checked

#### Claim 1: First measurement of absorption on knowledge hierarchies (city-country, city-continent, city-language)

**Search queries:** "feature absorption cross-domain beyond spelling knowledge hierarchy sparse autoencoder", "SAE absorption measurement beyond first letter task other domains entity features", "feature absorption knowledge hierarchies RAVEL cross-domain sparse autoencoder"

**Prior art found:**
- **Chanin et al. (2024)** explicitly call out single-task limitation: "we use the first-letter identification task as a case study." They do NOT measure absorption on any other task.
- **SAEBench (2025)** implements the absorption metric ONLY on the first-letter task (extended to more layers than Chanin et al.).
- **SynthSAEBench (Chanin, 2026)** [arXiv:2602.14687] uses synthetic data with ground-truth features, NOT natural language knowledge hierarchies.
- **Chaudhary & Geiger (2024)** [arXiv:2409.04478] evaluate SAEs on RAVEL for DISENTANGLEMENT (not absorption). They measure whether SAE features can be used to intervene on country vs continent attributes, but do NOT measure whether features fail to fire (the absorption phenomenon).
- **OrtSAE (Korznikov et al., 2025)** [arXiv:2509.22033] evaluates absorption on SAEBench (first-letter only).
- **KronSAE (Kurochkin et al., 2025)** [arXiv:2505.22255] evaluates absorption but again on the standard first-letter task.
- **Matryoshka SAEs (Bussmann et al., 2025)** evaluate absorption on the SAEBench first-letter metric.
- At least 5 papers explicitly note the single-task limitation and call for cross-domain validation.

**Classification:** No exact match. No paper has measured absorption (as defined by Chanin et al.) on knowledge hierarchy features.
**Severity:** The gap between RAVEL-based disentanglement evaluation (Chaudhary & Geiger) and absorption measurement on RAVEL features is the key novelty space. Disentanglement != absorption.

#### Claim 2: Hierarchy sharpness correlation with absorption severity

**Prior art found:**
- No paper has correlated absorption rate with mutual information between parent and child features.

**Classification:** No match.

### Collisions Summary
| Prior Work | Overlap | Severity |
|---|---|---|
| Chanin et al. (2024) | Absorption metric (same methodology, different domain) | related_work |
| Chaudhary & Geiger (2024) | RAVEL + SAE evaluation, but DISENTANGLEMENT not absorption | partial_overlap |
| SAEBench (2025) | Absorption metric implementation (first-letter only) | related_work |
| OrtSAE, KronSAE, Matryoshka | All test absorption only on first-letter | related_work |

### Recommendation: **PROCEED**

**Differentiation notes:** This fills the most explicitly and repeatedly called-for gap in the literature. The critical distinction from Chaudhary & Geiger (2024) must be made clear: they measure disentanglement (can you intervene on one attribute without affecting others?), while this proposal measures absorption (do SAE latents fail to fire on inputs they should represent because a co-occurring child feature has subsumed them?). These are fundamentally different phenomena. The shuffled-hierarchy control is essential for establishing specificity.

---

## Candidate 4: `cand_immunodominance` (Backup C)

**Title:** Feature Immunodominance: Predicting Absorption from Ecological Competition Theory

**Novelty Score: 9/10**

### Core Contribution Claims Checked

#### Claim 1: Lotka-Volterra / immunodominance framework for SAE feature absorption

**Search queries:** "Lotka-Volterra immunodominance competitive exclusion sparse autoencoder features"

**Prior art found:**
- Zero results combining immunodominance or Lotka-Volterra with SAE features.
- Lotka-Volterra models have been applied to many domains (ecology, tumor modeling, even Hebbian learning in random systems [arXiv:2301.11703]) but NEVER to SAE feature dynamics.
- There is work on competitive exclusion in random Lotka-Volterra systems (Barbier et al., 2023) that is conceptually related but in a completely different domain.

**Classification:** No match.

#### Claim 2: R* rule for predicting absorption direction

**Prior art found:**
- The R* rule (Tilman's resource competition theory) has never been applied to predict which SAE features absorb which.

**Classification:** No match.

#### Claim 3: Active suppression signature in encoder weights

**Prior art found:**
- No paper has tested for inhibitory patterns in SAE encoder weights as evidence of active suppression between competing features.

**Classification:** No match.

### Collisions Summary
| Prior Work | Overlap | Severity |
|---|---|---|
| Chanin et al. (2024) | Feature absorption phenomenon (this proposes a theoretical framework for it) | related_work |
| Feature Hedging (Chanin et al., 2025) | Related failure mode (hedging as competition outcome) | related_work |
| Lotka-Volterra + Hebbian couplings (2023) | Competitive exclusion in random systems (different domain) | related_work |

### Recommendation: **PROCEED (with caution)**

**Differentiation notes:** This is the most genuinely novel candidate -- zero prior art connects immunodominance/competitive exclusion theory to SAE features. The risk is that the analogy may be decorative rather than load-bearing (as noted in the proposal's own weakness assessment). The R* rule prediction and active suppression signature provide concrete, testable claims that would validate the framework beyond mere analogy.

**Caution:** The proposal correctly deprioritizes this candidate because it does not address the blocking confound or cross-domain issues. As a standalone paper, it would need strong empirical validation of the R* predictions to avoid the "just an analogy" criticism.

---

## Candidate 5: `cand_honest_audit` (Dropped)

**Title:** The Honest Audit: Absorption Is Less Severe, Less General, and Less Causal Than Believed

**Novelty Score: 7/10**

### Core Contribution Claims Checked

#### Claim 1: Systematic deflationary audit of absorption literature

**Prior art found:**
- No paper has systematically challenged the absorption literature's claims.
- However, elements of this critique exist in scattered form:
  - Feature Hedging (Chanin et al., 2025) shows that narrow SAEs suffer from hedging, not absorption, suggesting some "absorption" may be misclassified.
  - "Sparse but Wrong" (2025) shows incorrect L0 leads to incorrect features, suggesting SAE pathologies may be hyperparameter-driven.
  - SAEBench notes inverse scaling of absorption with width but does not frame this as deflating absorption's importance.

**Classification:** partial_overlap -- the ingredients for a deflationary narrative exist scattered across multiple papers.
**Severity:** partial_overlap

#### Claim 2: Taxonomy inflation correction (92.3% -> corrected estimate)

**Prior art found:**
- No paper has specifically challenged the taxonomy classification methodology or the n_comparison_tokens=0 fallback.

**Classification:** No match (this is internal to the project's own prior iterations).

### Collisions Summary
| Prior Work | Overlap | Severity |
|---|---|---|
| Feature Hedging (2025) | Partial challenge to absorption narrative | partial_overlap |
| "Sparse but Wrong" (2025) | L0-driven pathologies, not inherent absorption | partial_overlap |

### Recommendation: **PROCEED only as fallback (as designed)**

**Differentiation notes:** The honest audit framing is novel as a unified narrative but draws on existing scattered evidence. It should only be activated if H1 and H2 are both falsified, as designed.

---

## Cross-Candidate Risk: Rapidly Evolving Field

Several recent publications (2025-2026) are rapidly closing the gaps this proposal targets:

1. **Matryoshka SAEs (ICML 2025)** -- propose an architectural solution to absorption. If the community adopts Matryoshka SAEs widely, the motivation for characterizing absorption in vanilla SAEs weakens (though characterization remains valuable for understanding WHY Matryoshka works).

2. **OrtSAE (Sep 2025)** -- reduces absorption by 65% via orthogonality penalty. Evaluates on SAEBench first-letter metric.

3. **KronSAE (May 2025)** -- reduces absorption via Kronecker factorization. Also evaluates on first-letter only.

4. **Feature Hedging (May 2025)** -- identifies a new failure mode (hedging) that interacts with absorption, complicating the simple "absorption is bad" narrative.

5. **SynthSAEBench (Feb 2026)** -- provides ground-truth synthetic benchmark, enabling precise absorption diagnosis.

6. **ATM (2025)** -- Adaptive Temporal Masking achieves absorption score of 0.0068 vs 0.1402 for TopK.

**Implication:** The field is moving fast toward MITIGATING absorption architecturally. This proposal's contribution is CHARACTERIZING absorption empirically (confound control, cross-domain, scaling surface), which is complementary to mitigation work. The characterization value actually INCREASES as mitigation approaches proliferate -- practitioners need to understand WHAT absorption is and WHEN it matters to evaluate whether mitigation is necessary.

---

## Literature Not Found (Absence of Evidence)

The following specific combinations were searched and returned ZERO relevant results:

1. Mediation analysis applied to SAE evaluation hyperparameters
2. Rosenbaum sensitivity bounds for any ML interpretability question
3. Bradford Hill criteria applied to ML model evaluation
4. Feature absorption measured on RAVEL knowledge hierarchies
5. Feature absorption measured on ANY non-spelling task in natural language
6. 2D absorption phase surface with formal statistical fitting (GAM or otherwise)
7. Lotka-Volterra/immunodominance framework for SAE feature dynamics
8. Propensity score matching for SAE architecture comparison

These absences provide strong evidence for the novelty of the proposal's core contributions.

---

## Summary Table

| Candidate | Score | Recommendation | Key Risk |
|---|---|---|---|
| cand_causal_chain | 8 | PROCEED | 54-SAE sample size for stratified analysis |
| cand_phase_diagram | 7 | PROCEED (differentiate from SAEBench grid) | Descriptive plots already exist; need formal analysis |
| cand_cross_domain | 8 | PROCEED | Must distinguish from Chaudhary & Geiger disentanglement |
| cand_immunodominance | 9 | PROCEED with caution (not blocking-priority) | "Just an analogy" risk |
| cand_honest_audit | 7 | PROCEED only as fallback | Ingredients scattered across recent papers |

**Overall assessment:** The front-runner candidate's integrated proposal (Contributions 1-3) has strong novelty across all three contributions. No single paper or combination of papers replicates any of the three core claims. The methodological novelty (epidemiological causal methods for SAE evaluation) is particularly strong and has zero prior art.

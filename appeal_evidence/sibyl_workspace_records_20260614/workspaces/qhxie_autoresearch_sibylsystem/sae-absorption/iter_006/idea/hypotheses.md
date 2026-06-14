# Testable Hypotheses with Expected Outcomes

## Primary Hypotheses

### H1: Cross-Domain Generalization (Revised)
**Statement**: Feature absorption occurs at rates >= 5% in at least two knowledge hierarchy domains (city-continent, city-language, or animal-class) on Gemma 2 2B Gemma Scope SAEs (layer 12, 16k width), after probe quality gating (F1 > 0.85) and control calibration.

**Expected outcome**: Based on pilot data, city-continent (6.5%) and city-language (6.6%) are near threshold. With improved probe quality and calibrated controls, these may solidify or weaken. City-country likely remains near 0%. The key finding is that absorption rates are domain-dependent, substantially lower for knowledge hierarchies than for first-letter.

**Falsification**: Absorption rate < 3% across ALL knowledge hierarchy domains after probe gating, frequency matching, and control calibration.

**Evidence basis**: Pilot data (city-continent 6.5%, city-language 6.6%, city-country 0%, animal-class 1.4%). Chanin et al. first-letter baseline (15-35%). Sparsity incentive is general but hierarchy structure varies.

**Revision from prior round**: Threshold lowered from >= 10% to >= 5% based on pilot evidence. The 0% city-country rate and domain variability are treated as findings, not failures.

---

### H2: Confound Decomposition
**Statement**: On Gemma Scope SAEs at optimal L0 (L0 ~ 22 where probes achieve F1 = 1.0), hierarchy-driven absorption accounts for > 80% of total false negatives, and this fraction varies systematically with L0.

**Expected outcome**: Pilot shows 96.9% hierarchy-driven at L0=22. At suboptimal L0 values, hedging should increase (too-low L0) and reconstruction error should increase (too-high L0). The decomposition profile across L0 is the novel contribution.

**Falsification**: Hierarchy-driven fraction < 50% at all tested L0 values.

**Evidence basis**: Pilot confound decomposition (96.9% hierarchy-driven, 3.1% hedging, 0% reconstruction error at L0=22).

---

### H3: Rate-Distortion Diagnostic (CMI Predicts Absorption)
**Statement**: The conditional mutual information I(X; f_parent | f_child), estimated in the decoder-direction subspace, is negatively correlated with absorption rate across 25 first-letter features: Spearman rho < -0.3.

**Expected outcome**: Letters with low CMI (parent carries little unique information beyond child) should be preferentially absorbed, because absorption is rate-distortion optimal for them. Letters with high CMI should resist absorption. This is the most novel prediction.

**Falsification**: Spearman rho > -0.2 (no directional predictive power).

**Evidence basis**: Rate-distortion theory (Equitz & Cover, 1991); Chanin et al.'s informal sparsity-gain argument formalized as a CMI threshold; pilot CMI estimates available but not yet validated against absorption rates with adequate probe quality.

---

### H4: Unsupervised Detection (Secondary)
**Statement**: The refined unsupervised absorption score correlates with probe-based absorption rate on first-letter at Spearman rho > 0.3.

**Expected outcome**: Pilot evidence is discouraging (ITAC no clear separation, few matching pairs). Full experiments with larger corpus and refined filtering may improve, but expectations are modest.

**Falsification**: Spearman rho < 0.15 on validation task.

**Pre-registered decision**: If rho < 0.3, report as negative result. Do not deploy cross-domain. This is explicitly the most at-risk hypothesis.

**Evidence basis**: Pilot unsupervised validation showed ITAC median 1.35 vs random 1.14 (not significant). Conditional cosine pipeline identified very few matching pairs per letter. The LessWrong negative result for naive approaches is confirmed; the question is whether our innovations suffice.

---

### H5: Scaling Law
**Statement**: Absorption rate exhibits a significant width-L0 interaction (p < 0.05 in GAM with architecture covariate) with a phase transition in the L0 ~ 7-14 range.

**Expected outcome**: Iteration 5 found p = 3.1e-15 on 420 SAEs without architecture covariate. With architecture as covariate, the interaction should remain significant within-architecture if it reflects genuine physics.

**Falsification**: Width-L0 interaction non-significant (p > 0.05) within any single architecture.

**Evidence basis**: Iteration 5 Phase 3 (420 SAEs, R^2 = 0.693).

---

### H6: Hierarchy Predictors
**Statement**: At least one hierarchy property (co-occurrence ratio, fan-out, depth) correlates with absorption rate at Spearman |rho| > 0.3 across 4+ independently measured domains with Bonferroni-corrected p < 0.05.

**Expected outcome**: Pilot data suggests fan-out may matter (city-continent: fan-out=30.8, rate=6.5%; first-letter: fan-out=23.0, rate=13.4%; city-country: fan-out=6.6, rate=0%). Co-occurrence ratio varies (first-letter: 1.0; city-continent: 0.08), potentially confounded with domain.

**Falsification**: No property achieves |rho| > 0.2 with Bonferroni-corrected p < 0.05.

**Evidence basis**: Chanin et al.'s sparsity gain argument predicts co-occurrence ratio matters. Pilot cross-domain rate variation suggests domain structure matters.

---

## Theoretical Hypothesis

### H7: Lateral Inhibition Bifurcation
**Statement**: JumpReLU SAEs exhibit more binary (all-or-nothing) absorption patterns than L1 SAEs at matched L0, as predicted by the LCA bifurcation analysis (transcritical bifurcation for hard thresholds vs continuous transition for soft thresholds).

**Expected outcome**: The distribution of per-parent absorption rates for JumpReLU should be bimodal (features either fully absorbed or not absorbed), while L1 should show a continuous distribution. This is directly testable if L1-trained SAEs are available via SAELens.

**Falsification**: Absorption rate distributions for JumpReLU and L1 are statistically indistinguishable (KS test p > 0.1).

**Evidence basis**: LCA dynamics (Rozell et al., 2008) predict hard thresholding produces discontinuous bifurcation. JumpReLU is a hard threshold; L1+ReLU is a soft threshold. SAEBench data shows JumpReLU has worse absorption, consistent with more aggressive competitive exclusion.

---

## Decision Tree

```
Exp 1a (improved first-letter) --> Controls calibrated (shuffled < 20%)?
  NO --> Investigate metric implementation; consider Alternative 1 (decomposition focus)
  YES --> Continue to cross-domain

Exp 1b-1e (cross-domain) --> At least 2 domains show rate >= 5%?
  NO --> H1 weakened. Domain-dependent absorption is the finding. Consider Alternative 1.
  YES --> Continue with full analysis

Exp 2a (confound decomposition) --> Hierarchy-driven > 80% at optimal L0?
  YES --> H2 supported. Strong case for genuine absorption.
  NO --> Absorption partially artifactual. Alternative 1 gains strength.

Exp 3a (CMI-absorption correlation) --> rho < -0.3?
  YES --> H3 supported. Rate-distortion theory validated. Strong paper.
  NO --> De-prioritize theory. Consider Alternative 1.

Exp 6 (unsupervised validation) --> rho > 0.3?
  YES --> H4 supported. Deploy cross-domain.
  NO --> H4 FALSIFIED. Report negative result. Drop unsupervised claims.

Overall scoring:
  H1 + H2 + H3 supported --> Strong paper (rate-distortion + cross-domain + decomposition)
  H1 + H2 only --> Moderate paper (cross-domain characterization + decomposition)
  H3 only --> Pivot to Alternative 2 (theory-first paper)
  None --> Pivot to Alternative 1 (metric validation / negative results paper)
```

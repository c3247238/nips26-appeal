# Supervisor Review -- Iteration 10

**Score: 6.5 / 10**
**Verdict: CONTINUE**
**Calibration: Borderline Reject at a top ML venue. Addressable within one iteration.**

---

## Executive Summary

The paper presents the first cross-domain characterization of feature absorption in SAEs, extending measurement from first-letter spelling to entity-attribute knowledge hierarchies (RAVEL) on Gemma 2 2B. Three contributions are defensible: (1) a probe degradation ablation demonstrating that probe quality is a major confound in absorption measurement (R^2=0.777, rho=-1.0, p=0.009), (2) causal evidence for universal competitive exclusion via activation patching (d=0.75-1.50 across three hierarchy types), and (3) a quadruple failure of correlational predictors (GAS, CMI, Absorption Tax, rate-distortion) establishing that absorption requires interventional methods.

However, five critical data integrity failures prevent a score above 6.5. The paper is approximately 75% submission-ready; the remaining 25% consists of fixable but unfixed problems that a careful NeurIPS reviewer would identify within the first 30 minutes of reading.

---

## Dimension Scores

### 1. Novelty & Significance: 7/10

The paper addresses a genuine gap: all published absorption measurements use first-letter spelling, and the field needs cross-domain characterization. The probe degradation ablation is a genuinely novel methodological contribution -- no prior work quantifies how probe quality confounds absorption measurement. The quadruple negative for correlational predictors, while not individually surprising, is collectively valuable as a field-guiding boundary.

However, the novelty faces two challenges. First, SAEBench already contains both absorption (first-letter) and RAVEL (city hierarchies) as independent metrics in the same benchmark. The conceptual leap from "two metrics in one benchmark" to "measure absorption on RAVEL hierarchies" is moderate. Second, the paper's own probe degradation result shows that 2 of 3 RAVEL hierarchies' absorption rates are explained by probe quality, meaning the "cross-domain characterization" headline is weaker than the paper claims. The truly novel finding is the city-language anomaly (-21.3 pp below the curve), not the full 4.1x range.

### 2. Technical Soundness: 6/10

The methodology is well-designed: quality gates with strict/relaxed thresholds, bootstrap CIs, Bonferroni-corrected pairwise tests, activation patching with magnitude-matched controls, and the probe degradation ablation protocol are all sound in principle. The statistical analysis is generally appropriate.

Five specific soundness failures reduce this score:

(a) **Table 3 CI inversion** -- 5 of 7 confidence intervals do not contain their point estimates. This is a fundamental statistical error. Source: per-token point estimates mixed with per-word bootstrap CIs.

(b) **Unverified patching sign reversal** -- The cross-domain patching data corrected from d=-0.91 to d=+1.50 without independent verification. The data integrity check was "skipped for speed."

(c) **Aggregation inconsistency** -- Three different first-letter absorption rates (21.6%, 27.1%, 34.5%) for the same experiment using different aggregation methods, never reconciled into one authoritative number.

(d) **Probe degradation extrapolation** -- Gaussian noise on binary probes extrapolated to multi-class probes without cross-domain validation. The quantitative decomposition claims rest on untested domain-transferability.

(e) **Layer multiplier contradiction** -- 26x, 15x, and 11x all claimed for the same L6-to-L24 first-letter ratio in different parts of the paper.

### 3. Experimental Rigor: 6/10

The experimental design is comprehensive: 4 hierarchies, 4 layers, 2 SAE widths, 7 probe degradation levels, activation patching with controls, decoder entanglement with random-direction controls, and 4 correlational approaches. The hypothesis testing framework (11 hypotheses with honest verdicts) is exemplary.

Specific rigor gaps:

(a) **City-country excluded from patching** -- The highest-absorption hierarchy is absent from the causal mechanism test, without explanation.

(b) **Figure 4 layer mismatch** -- First-letter patching at L12 compared visually with RAVEL patching at L24 in the same figure.

(c) **Architecture comparison underpowered** -- 12 observations across 4 architectures with a width mismatch confound. The paper correctly flags this but still includes the section.

(d) **Probe degradation sample size** -- R^2=0.777 on 7 points (5 df) and R^2=0.942 on 7 points (4 df) are overfitting-prone for quantitative extrapolation.

(e) **No cross-domain probe degradation** -- The degradation curve is trained on first-letter only. Whether it transfers to RAVEL domains is assumed.

### 4. Reproducibility: 5/10

The methodology section (Section 3) is comprehensive and describes all procedures before results appear. Quality gates, aggregation methods, threshold validation, and circularity caveats are documented proactively.

However:

(a) **No code availability** -- The custom pipeline spanning 4 libraries is not released.

(b) **No automated data verification** -- validate_integration.py has been recommended for 10 iterations and never implemented. At least 5 numerical inconsistencies exist.

(c) **Aggregation ambiguity** -- A reader cannot determine which aggregation method to use. Per-token, per-word, and per-letter all produce different rates.

(d) **Library versions** are documented in methodology notes but not in the paper.

(e) **Probe degradation calibration** -- The epsilon-to-F1 mapping procedure is described generically ("calibrating epsilon") without specifying grid search, bisection, or exact procedure.

---

## What Works Well

1. **Honest negative result reporting** -- The most consistently praised aspect across 10 review iterations. 4 supported hypotheses, 3 definitive negatives, 1 refuted, 1 mixed, 1 partially supported, 1 underpowered. Table 5 with verdict summary is exceptional.

2. **Probe degradation ablation** (Section 4.6) -- Genuinely creative experimental design that directly addresses the most obvious reviewer objection. The three-way decomposition (city-continent explained, city-country mostly explained, city-language genuine outlier) is the paper's strongest section.

3. **Statistical rigor** -- Bootstrap CIs (10k resamples), Mann-Whitney U, Kruskal-Wallis, Bonferroni correction, Cohen's d, DeLong tests. This level of statistical reporting is consistently appropriate.

4. **First-letter activation patching** -- d=1.33, p=0.000218, 16/19 words with positive recovery. This is the paper's cleanest causal result, stable across iterations, and the first interventional evidence for competitive exclusion in SAEs.

5. **Methodology section** -- Section 3 is preemptive: quality gates, aggregation methods, cosine threshold validation, circularity caveats, and activation patching controls documented before any results. This prevents many reviewer complaints.

---

## What Would Raise the Score

The score would move from 6.5 to 7.5+ with the following actions (estimated 2.5 GPU-hours + 8 CPU-hours):

1. **Fix Table 3 CI inversion** -- Use consistent per-token aggregation for both point estimates and CIs. [0 GPU, 2h CPU]

2. **Run 20-entity patching spot-check** -- Verify the cross-domain sign reversal correction. [0.5 GPU-hr]

3. **Choose one canonical aggregation method** -- Recompute all rates consistently. Reconcile Tables 2 and 3. [0 GPU, 2h CPU]

4. **Restructure contributions** -- Lead with probe degradation methodology, not cross-domain characterization. Replace 4.1x with 2.7x quality-gated. [0 GPU, 2h CPU]

5. **Run city-continent probe degradation** -- Validate cross-domain extrapolation. [1 GPU-hr]

6. **Fix Figure 4 layer mismatch** -- Re-run first-letter patching at L24 or add prominent caption note. [0-0.5 GPU-hr]

7. **Reconcile layer multiplier** -- One consistent number throughout. [0 GPU, 30 min]

The score could reach 8.0 (Accept-level) if items 1-4 resolve cleanly AND the cross-domain patching spot-check confirms the corrected data. The paper has genuine scientific contributions; the gap to acceptance is primarily data hygiene and framing, not fundamental experimental weakness.

---

## Comparison with Prior Review

The prior review (iteration 9) scored the paper at 7.0. This review scores 6.5. The regression reflects:

- **Table 3 CI inversion was introduced in iteration 10** (not present in iteration 9) and is a new critical error.
- **The probe degradation result** (new in iteration 10) is scientifically strong but undermines the paper's own headline contribution (cross-domain characterization), which was not reframed.
- **Cross-domain patching sign reversal** was flagged in iteration 9 but the data integrity verification was skipped in iteration 10.

The positive improvements (probe degradation R^2 from 0.077 to 0.777, decoder magnitude cross-hierarchy consistency, rate-distortion confirmation at scale) are real advances. The score regression is due to new errors introduced alongside these advances, not to the advances being weak.

---

## Bottom Line

This paper has a publishable core: probe degradation methodology, first-letter causal evidence, transparent negative results. The cross-domain extension and universal mechanism claims are promising but require verification. The critical path to submission quality is: fix the data integrity issues (Table 3, aggregation, layer multiplier), verify the cross-domain patching, and reframe contributions to match evidence strength. Total investment: approximately 2.5 GPU-hours + 8 CPU-hours. No structural redesign needed.

**CONTINUE with mandatory data integrity fixes and targeted verification.**

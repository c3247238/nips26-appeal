# Supervisor Review -- Current Iteration

## Overall Score: 7.0 (Weak Accept)

**Verdict: CONTINUE**

---

## Executive Summary

The paper presents the first cross-domain and cross-layer characterization of SAE feature absorption, extending measurement from the canonical first-letter spelling task to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) on Gemma 2 2B. The core empirical findings are solid: (1) first-letter absorption varies 15-fold across model layers (2.2% at L18 to 34.5% at L24), unconfounded by probe quality; (2) absorption rates differ significantly across hierarchy types (Kruskal-Wallis p=0.005) with 4/6 pairwise comparisons significant; (3) activation patching provides statistically significant causal evidence for feature suppression (32.5% vs 1.5%, d=1.33); (4) the widely cited ~98% hedging rate is decomposed into 7.9% strict hedging and 86.2% compensatory resolution. Negative results (GAS, CMI, Absorption Tax) are reported with exemplary transparency.

However, one critical factual error and several major unresolved confounds prevent a higher score.

---

## Critical Issue

### 1. Fabricated Hedging Number at L0=82

Section 5.2 claims: "At the more conservative base L0=82 with target L0=176 (2.1x multiplier), strict hedging rises to 12.3%."

**The raw data contradicts this.** The file `hedging_decomposition_full.json` (field `L0_82_to_176`) reports:
- strict_hedging = 0 (0.0%)
- compensatory = 58 (78.4%) 
- persistent = 16 (21.6%)
- total_fn = 74

There is ZERO strict hedging at L0=82. The 12.3% figure appears in no source data file. The pilot phase0 data shows 3.66% strict hedging at L0=82 (3/82 FNs from a different vocabulary), which is also not 12.3%. This appears to be an error introduced during writing.

**Impact:** This error undermines trust in the paper's numerical claims. A reviewer who cross-checks will question all other numbers. Ironically, the correct number (0.0%) actually *strengthens* the paper's argument that loose hedging is near-tautological -- at L0=82, not a single parent feature recovers when L0 is expanded to 176.

**Fix:** Replace with: "At the more conservative base L0=82 with target L0=176, strict hedging is 0.0% (0/74 FNs), compensatory resolution is 78.4%, and persistent false negatives rise to 21.6%."

---

## Major Issues

### 2. Probe Quality Confound Unresolved

The paper's central cross-domain contribution depends on RAVEL probes that are below the strict quality gate (best F1=0.843). Probe quality correlates with false negative rate at rho=-0.756. The degraded-probe ablation -- injecting calibrated label noise into first-letter probes to simulate RAVEL-level quality -- is described as "highest priority" future work (Section 8.5) but was not executed.

A skeptical reviewer can dismiss the cross-domain finding as: "You measured worse probes and got different rates. This is expected by construction." Until this confound is resolved experimentally, the cross-domain claim rests on the Kruskal-Wallis p-value alone, which tests whether rates differ but cannot attribute the difference to hierarchy structure vs. probe quality.

### 3. Activation Patching at L12, Not L24

The strongest causal evidence (32.5% recovery, d=1.33) comes from layer 12, which has only 5.7% absorption. The paper's headline finding -- 34.5% absorption at L24 -- lacks causal validation. This disconnect means the paper proves causation where absorption is mild and can only claim correlation where absorption is severe.

### 4. Letter G Dominates Strict Hedging

In the L0=22 decomposition (304 FNs), letter G contributes 9/24 strict hedging cases (37.5%) with 64.3% per-letter strict rate. Letter A contributes 7/24 (29.2%). Together, G and A account for 16/24 = 66.7% of all strict hedging. Excluding both, the strict hedging rate drops to approximately 2.9% (8/272), barely above the persistent rate (5.9%). The paper reports G as notable but does not compute statistics without it or investigate the mechanism.

### 5. Architecture Comparison Adds Noise

Section 6 presents a null result (p=0.87) from N=16 observations at L12, where RAVEL probes are at their worst (F1=0.52-0.69) and sparsity is confounded across architectures. The claim "hierarchy type explains more variance than architecture choice" compares the L12 architecture test to the L24 hierarchy test, which are at different layers with different probe qualities. This comparison is invalid.

### 6. Cross-Domain Finding is L24-Only

RAVEL probes were measured only at L24. The paper's pilot data at L12 suggested the opposite pattern (semantic > first-letter). This crossover interaction means the cross-domain finding is layer-specific, not general.

---

## What Works Well (Preserve)

1. **Cross-layer absorption characterization** (Table 3, Figure 3): Clean, unconfounded (F1 >= 0.97 everywhere), and the 15-fold variation is a genuinely novel and impactful finding. This is the paper's strongest single result.

2. **Activation patching design** (Table 5): Well-controlled with magnitude-matched random latents, multiple statistical tests (Wilcoxon, Mann-Whitney, paired t, bootstrap), and transparent word selection bias reporting. The causal mechanism is convincingly demonstrated within its scope.

3. **Hedging decomposition** (Section 5.2): The three-way split (strict/compensatory/persistent) is a clean and actionable decomposition. The 7.9% vs 86.2% finding is genuinely useful.

4. **Negative result reporting**: GAS, CMI, and Absorption Tax are each reported with clear threshold criteria, full statistical details, and mechanistic failure analysis. This is the paper's most consistently praised quality across all review iterations and should be preserved without modification.

5. **Probe quality confound transparency** (Section 4.3): The honest discussion of the rho=-0.756 correlation and three specific confound mechanisms is a model of scientific transparency. The paper does not hide its weaknesses.

6. **Statistical rigor**: Bootstrap CIs (10k resamples), permutation tests, Cohen's d, multiple comparison awareness -- the statistical methodology is sound throughout.

---

## Dimension Scores

### Novelty: 7.5
The cross-layer finding (15-fold absorption variation) is genuinely novel -- no prior work has characterized layer dependence. The cross-domain extension is novel in concept but weakened by the probe quality confound. The hedging decomposition advances understanding beyond the loose 98% figure. The three failed unsupervised detectors prevent others from pursuing dead ends. Together these represent a meaningful contribution, though not paradigm-shifting.

### Soundness: 6.5
The 12.3% data fabrication/error is a serious red flag. The probe quality confound limits the interpretability of cross-domain rates. The causal evidence is sound but restricted to the low-absorption regime. The architecture comparison is underpowered and confounded. The overall argument -- that absorption evaluation must be multi-task and multi-layer -- is supported by the data, but the quantitative specifics of cross-domain rates carry substantial uncertainty.

### Experiments: 7.0
The experimental scope is impressive: 10 SAE configurations, 4 hierarchies, 4 layers, activation patching, hedging decomposition, 3 unsupervised detectors, threshold sensitivity. Execution is generally clean with good controls. The main gaps are the missing degraded-probe ablation, L24 patching, and cross-model validation. The architecture comparison at L12 is the weakest experimental component.

### Reproducibility: 7.0
Model, SAE, and data sources are clearly specified. The measurement pipeline is well-described. Probe training details (regularization, CV, splits) are provided. The code apparently exists but no URL is given (placeholder in conclusion). The FN count discrepancies between pilot and full runs (340 vs 304) could confuse reproducers but are explained by vocabulary differences.

---

## Path to 8.0 (Accept)

Four actions, ordered by impact:

1. **Fix the 12.3% error** (immediate, zero compute).
2. **Execute degraded-probe ablation** (low compute, resolves the central confound).
3. **Extend activation patching to L24** (moderate compute, connects causal evidence to headline finding).
4. **Investigate letter G** and compute strict hedging excluding G and A (zero compute, strengthens hedging result).

If items 1-4 are completed and the degraded-probe ablation confirms that cross-domain rate differences survive probe quality matching, the paper reaches 8.0. If the ablation shows that probe quality explains most of the cross-domain variation, the paper should be reframed around the layer-dependence finding (which is unconfounded) plus the causal evidence and hedging decomposition, with cross-domain variation reported as inconclusive pending better probes.

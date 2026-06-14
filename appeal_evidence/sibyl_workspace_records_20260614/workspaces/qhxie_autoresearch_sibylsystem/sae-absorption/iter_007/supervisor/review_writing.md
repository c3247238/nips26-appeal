# Supervisor Review -- Iteration 9

## Score: 7.5 / 10 (Borderline Accept)

**Verdict: continue** (one more iteration to reach 8.0)

**Score trajectory**: 5.5 (iter 1-3) -> 6.5 (iter 4) -> 6.0 (iter 5) -> 6.5 (iter 6-8) -> **7.5 (this review, +1.0)**

---

## Executive Summary

This iteration breaks the three-review stagnation at 6.5. All six blocking experiments identified across iterations 6-8 have been executed and integrated: activation patching (0/8 recovery), tightened hedging (6.2% strict vs 98.6% permissive), CMI replication at L0=22 (rho=0.044, null), partial correlation (rho=-0.328, p=0.118), leave-one-out sensitivity (max |delta rho|=0.088), and threshold sensitivity (CV=0.077, control failure at all 20 grid cells). A data validation script confirmed 84/85 numerical claims match source data. The paper now has three well-supported pillars and the CMI overclaiming has been corrected.

---

## Cross-Validation Against Raw Data

I verified the following claims against source JSON files:

| Claim | Paper Value | Source Value | Source File | Status |
|-------|-------------|-------------|-------------|--------|
| Absorption L0=22 | 42.85% | 0.4285 | confound_decomp | MATCH |
| Absorption L0=176 | 0.84% | 0.0084 | confound_decomp | MATCH |
| Strict hedging rate | 6.2% | 0.0625 | tightened_hedging.json | MATCH |
| Strict hedging count | 41 | 41 | tightened_hedging.json | MATCH |
| Shuffled control strict rate | 3.4% | 0.0344 | tightened_hedging.json | MATCH |
| z-score strict vs shuffled | 3.51 | 3.5125 | tightened_hedging.json | MATCH |
| Patching recovery | 0/8 | 0/8 | activation_patching.json | MATCH |
| CMI rho at L0=82 d'=10 | -0.383 | -0.383 | partial_correlation_cmi.json | MATCH |
| Partial rho (CMI, abs \| F1) | -0.328 | -0.328 | partial_correlation_cmi.json | MATCH |
| Restricted rho (F1>0.85) | -0.113 | -0.113 | partial_correlation_cmi.json | MATCH |
| CMI rho at L0=22 d'=10 | 0.044 | 0.0439 | cmi_replication_l0_22.json | MATCH |
| Threshold CV | 0.077 | 0.0773 | threshold_sensitivity.json | MATCH |
| Random candidates (cos>=0.025) | 3,766 | 3,765.5 | control_failure_diag.json | MATCH |
| Data validation score | -- | 84/85 match | data_validation_report.json | VERIFIED |

All numbers in the paper match source data within rounding tolerance. The one MISSING_DATA entry in validation is the leave-one-out max-influence letter identity (V), which is correctly reported in the paper text and confirmed by leave_one_out_cmi.json.

---

## Dimension Scores

### 1. Novelty & Significance: 7.5

The paper makes the first systematic metric audit of the Chanin absorption metric on JumpReLU SAEs -- the dominant architecture in the Gemma Scope ecosystem. The finding that shuffled-label controls exceed measured absorption in all 5 hierarchy domains, combined with a structural explanation (candidate explosion at cos >= 0.025 in high-dimensional space), is genuinely new and practically important. The L0 phase transition (42.85% to 0.84%) identifies a control parameter that no prior work has mapped. The cross-domain extension to 5 hierarchy domains is new territory. The CMI non-replication, while negative, is a useful cautionary result.

The novelty is somewhat bounded by scope: this is an audit of an existing metric on an existing architecture, not a new method. But audits that save the community from building on flawed measurements have high practical value.

### 2. Technical Soundness: 7.5

The technical execution has improved substantially. The confound decomposition now presents both permissive (98.6%) and strict (6.2%) hedging rates, resolving the near-tautological classification from previous iterations. The activation patching provides metric-independent causal evidence. The CMI analysis includes the full confound-control chain: raw rho, partial rho controlling for probe F1, restricted analysis on quality-gated letters, and cross-L0 replication. The threshold sensitivity analysis (20 cells, all showing control failure) establishes that the metric problem is structural.

Two soundness concerns remain: (a) the activation patching sample (n=8) gives a wide CI, and (b) the three-number hedging decomposition (98.6% permissive, 6.2% strict, 93.8% "non-hedging") needs clearer narrative framing.

### 3. Experimental Rigor: 7.5

The experimental suite is now comprehensive for the paper's claims:
- 4-control suite (random probe, shuffled, dense, untrained) across 5 domains
- Threshold sensitivity: 5x4 parameter grid with rank stability analysis
- Confound decomposition: multi-L0 persistence + strict parent-latent check
- Activation patching: 3 intervention types + randomized controls
- CMI analysis: raw, partial, restricted, leave-one-out, cross-L0 replication, dimension sensitivity
- Data validation: automated cross-check of 85 claims

Missing experiments that would strengthen the paper: larger activation patching sample, analytical d_model comparison, letter G investigation.

### 4. Reproducibility: 7.0

All SAE configurations are precisely specified (Gemma Scope IDs, layer, width, L0). Statistical methods are well-documented (bootstrap parameters, significance thresholds, correction methods). Five hierarchy domains use publicly available datasets (RAVEL, WordNet). Code release is promised.

Reproducibility gaps: the paper uses two vocabulary sizes (1,196 and 1,204) without fully explaining when each applies. The CMI replication at L0=22 uses pilot-mode data. The threshold sensitivity analysis uses a different vocabulary (576 words) from the main analysis (1,195-1,203 words).

---

## What Changed Since Last Review (Score Justification)

The +1.0 improvement from 6.5 to 7.5 is justified by six specific changes:

1. **Activation patching executed** (was critical, now resolved): 0/8 parent recovery after child zeroing provides the first causal evidence against competitive exclusion on JumpReLU SAEs. All 8 persistent core words are named with their child features and cosine similarities.

2. **Tightened hedging classification** (was critical, now resolved): Strict rate of 6.2% replaces the near-tautological 98.6% permissive rate as the primary reported number. The discrepancy between permissive and strict is honestly presented.

3. **CMI replication at L0=22** (was critical, now resolved): rho=0.044, p=0.835 at the pre-registered d'=10 dimension. Sign reversed from L0=82. Conclusively demonstrates the original signal was driven by probe quality confounds.

4. **CMI overclaiming fixed** (was critical, now resolved): Section 6 retitled to "Exploratory CMI-Absorption Association." Bonferroni-corrected p=0.236 reported alongside uncorrected p=0.059. "Predicts" language removed. Partial correlation and restricted analysis included.

5. **Threshold sensitivity included** (was major, now resolved): Table 3 presents the 5x4 grid. CV=0.077. Control failure at all 20 cells. Magnitude gap > cosine threshold in discriminative power.

6. **Control failure diagnosed** (was major, now resolved): Structural explanation with quantitative backing: random vector identifies 23.0% of decoder columns as candidates; true probes 21.2%; shuffled probes 21.4%. Dead features (18.8%) contribute to noise.

---

## Remaining Issues for Next Iteration

### Major (4 issues)

1. **Activation patching at scale**: n=8 is the binding constraint preventing 8.0. A stratified sample of 30-50 FN tokens would narrow the CI from [0%, 36.9%] to approximately [0%, 11.6%].

2. **Confound decomposition narrative**: The 98.6% vs 6.2% contrast is confusing. A three-category framing (strict hedging / diffuse resolution / persistent core) would be clearer.

3. **Single-model limitation**: An analytical d_model comparison (expected candidate fraction at cos >= 0.025 in R^768 vs R^2304) would explain why the metric works on GPT-2 but fails on Gemma 2 2B, costing zero GPU time.

4. **FN count discrepancy (656 vs 657)**: Needs a reconciliation footnote.

### Minor (3 issues)

5. Letter G outlier in strict hedging (46% of all strict-hedging cases).
6. CMI replication uses pilot-mode data.
7. Threshold sensitivity vocabulary basis differs from main analysis.

---

## Exemplary Practices (Preserved From Previous Iterations)

1. **Negative results reporting**: 4/7 pre-registered hypotheses falsified, each with specific expected vs observed values. Best negative-results section across all reviews.

2. **Two-interpretation framework** (Section 7.2): Mature, balanced presentation of metric miscalibration vs genuine low absorption, with activation patching framed as the discriminating experiment.

3. **Evidence-first structure**: Each section opens with the key result. Readers never wait for the conclusion.

4. **Statistical rigor**: Bootstrap CIs, Bonferroni correction, effect sizes, shuffled controls with multiple replicates, cross-layer validation.

5. **Data integrity**: Automated validation of 85 claims with 84 matches confirms reliable reporting.

---

## Path to 8.0

The paper is at borderline accept (7.5). Four actions would move it to accept (8.0):

1. Scale activation patching to >= 30 FN tokens (~1-2 GPU hours)
2. Add analytical d_model comparison (zero GPU, 30 minutes)
3. Add footnotes resolving vocabulary and FN count discrepancies (zero GPU, 15 minutes)
4. Reframe the confound decomposition as a three-category system (writing revision, 30 minutes)

Total estimated effort: 2-3 hours including GPU time.

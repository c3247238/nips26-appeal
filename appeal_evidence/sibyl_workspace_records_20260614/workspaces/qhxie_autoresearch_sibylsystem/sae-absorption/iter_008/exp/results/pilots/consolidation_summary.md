# Phase 4.2: Final Consolidation Summary (v3 -- FULL MODE)

Timestamp: 2026-04-15T23:59:08.631152
Mode: **FULL** | Iteration: 9 | Elapsed: 0.0s
Tasks tracked: 4 completed, 0 failed

## Writing Gate: **GO_WRITE**

Primary contribution (cross-domain absorption differences) has statistical evidence (4/6 comparisons significant, ANOVA p=0.005). Causal evidence (activation patching) is statistically significant (Wilcoxon p=0.000218, d=1.33). Probe quality is acceptable with documented caveats (best RAVEL F1=0.84, first-letter F1=0.97).

### Paper Framing Update
- **Original:** Semantic hierarchies show MORE absorption than first-letter (pilot at L12)
- **Revised:** Absorption is LAYER-DEPENDENT and HIERARCHY-DEPENDENT. At L24, first-letter shows HIGHEST absorption (34.5%), while cross-domain hierarchies show lower rates (13-36%). This represents significant cross-domain variation (ANOVA p=0.005).
- **Implication:** Paper must be reframed: primary contribution is demonstrating significant cross-domain variation with the surprising layer-hierarchy interaction.

### Required Caveats
- RAVEL probes below strict quality gate (F1 0.79-0.84); absolute rates have uncertainty
- GAS, CMI, Absorption Tax are negative results -- appendix only
- Architecture comparison at layer 12 only; cross-domain probes below gate at that layer
- Activation patching probe F1=0.883 (below strict 0.90 gate)

## Hypothesis Verdicts

| Hypothesis | Name | Verdict | Confidence | Section |
|-----------|------|---------|------------|---------|
| H1 | Cross-Domain Variation | **SUPPORTED** | HIGH | Section 4 |
| H2' | Semantic > First-Letter | **REFUTED** | HIGH | Section 4 |
| H3 | Absorption-Hedging Decomposition | **PARTIALLY_SUPPORTED** | MEDIUM | Section 5 |
| H4 | GAS Detector | **REFUTED** | HIGH | Appendix |
| H5 | Absorption Tax Predictions | **NOT_SUPPORTED** | HIGH | Appendix |
| H6 | Architecture Generalization | **PARTIALLY_SUPPORTED** | LOW | Section 6 |
| H7 | Causal Absorption | **SUPPORTED** | HIGH | Section 5 |

## Key Numbers (Updated from Canonical Files)

- **Activation patching:** 32.5% vs 1.5%, diff=0.310, Wilcoxon p=0.000218, d=1.33
- **Tightened hedging (L0=22->176):** strict 7.9%, compensatory 86.2%, persistent 5.9%
- **First-letter absorption range:** 2.2% (L18_16k) to 34.5% (L24_16k)
- **GAS:** rho=0.116 (NEGATIVE)
- **Tax ranking:** rho=-0.20 (NOT SUPPORTED)
- **CMI:** rho=0.044 (NOT SUPPORTED)
- **Threshold sensitivity:** STRUCTURAL, CV=0.077
- **Architecture ANOVA:** arch p=0.87 (NS), hierarchy p=0.0050 (sig)

## Positive Results

- **Causal absorption confirmed via activation patching**: 32.5% recovery vs 1.5% control, Wilcoxon p=0.000218, d=1.33
- **Cross-domain absorption variation significant**: ANOVA p=0.005; 4/6 pairwise comparisons p<0.05
- **Tightened hedging reveals near-tautology**: Strict 7.9% vs loose 94.1%, 86.2% compensatory
- **Layer-dependent absorption (novel finding)**: First-letter: 2.2% (L18_16k) to 34.5% (L24_16k) -- 15x variation
- **Threshold sensitivity is structural**: FN constant (n=87) across 5x4 grid, CV=0.077
- **Probe quality strongly predicts false negative rate**: rho=-0.756, p<0.001

## Negative Results (Honestly Reported)

| Result | Metric | Section |
|--------|--------|---------|
| GAS fails as absorption detector | rho=0.116, AUROC=0.571 | Appendix B |
| CMI does not predict absorption | rho=0.044, p=0.83 | Appendix C |
| Absorption Tax ranking fails | ranking rho=-0.20, concordance ~50% | Appendix D |
| H2' refuted: semantic NOT > first-letter at L24 | first-letter 34.5% is highest at L24 | Section 4 (positive finding: cross-domain variation confirmed) |
| Architecture effect not significant | ANOVA p=0.87 | Section 6 |
| RAVEL probes below strict quality gate | best F1=0.84 (city-continent at L24) | Section 3 (documented caveat) |

## Probe Quality (Best per Hierarchy)

| Hierarchy | Best Layer | F1 | Gate |
|-----------|-----------|-----|------|
| city-continent | L24 | 0.8428 | Acceptable |
| city-country | L24 | 0.7895 | Below |
| city-language | L24 | 0.8234 | Acceptable |
| first-letter | L24 | 0.9711 | PASS (strict) |

## Architecture Comparison (Layer 12)

| Hierarchy | JumpReLU_16k | JumpReLU_65k | BatchTopK_16k | Matryoshka |
|-----------|----|----|----|----|
| first-letter | 0.7% | 1.3% | 3.4% | 1.4% |
| city-continent | 17.3% | 23.1% | 13.5% | 19.2% |
| city-language | 41.2% | 38.2% | 61.8% | 35.3% |
| city-country | 47.1% | 47.1% | 52.9% | 35.3% |

## Cross-Domain vs First-Letter at L24

| Hierarchy | SAE | FL Rate | CD Rate | Diff | p-value | Sig |
|-----------|-----|---------|---------|------|---------|-----|
| city-continent | L24_16k | 34.5% | 35.8% | +1.4% | 0.8292 | No |
| city-country | L24_16k | 34.5% | 18.5% | -16.0% | 0.0043 | **Yes** |
| city-language | L24_16k | 34.5% | 13.6% | -20.8% | 0.0001 | **Yes** |
| city-continent | L24_65k | 25.5% | 26.0% | +0.5% | 0.9319 | No |
| city-country | L24_65k | 25.5% | 12.7% | -12.8% | 0.0081 | **Yes** |
| city-language | L24_65k | 25.5% | 13.6% | -11.9% | 0.0149 | **Yes** |
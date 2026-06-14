# Critique of Experiment: Feature Absorption as Optimal Compression

## Overview

The experimental program is methodologically sound in its design (random baselines, MCP, cross-layer validation) but has critical gaps: insufficient power, confounded comparisons, and a null result framing problem.

## Critical Issues

### 1. Zero Significant Results After MCP (CRITICAL)

**Problem**: After Bonferroni correction for 12 tests (alpha=0.00417) and BH-FDR (q<0.05), the paper has ZERO statistically significant results. The sole uncorrected significant result (H1b L8: p=0.028) did not survive MCP.

**Why this matters**:
- H1-H4 (primary hypotheses): all null, none significant
- H1b (post-hoc): p=0.028 uncorrected, but this is a cherry-pick from H1
- H6 (primary): decisively falsified (precision@20=0.0)
- H8: r=+0.12, p=0.55 — not significant

**Statistical reality**: With n=26 features and 12 tests, the study has insufficient power to detect medium effects (r~0.5 requires n>26 for alpha=0.05, power=0.80). The "20% power" claim in Section 3.6 is post-hoc rationalization, not pre-registered analysis.

**Verdict**: The paper must reframe as a null-result study. Do not claim statistical significance that does not exist.

### 2. OrtSAE Ablation Comparison at Unmatched L0 (CRITICAL)

**Problem**: The paper compares OrtSAE without penalty (L0~920) vs. OrtSAE with penalty (L0~550) to conclude "orthogonality penalty does not appear to reduce absorption." This comparison is at unmatched L0 — the very confound the paper criticizes in other work.

**The confound**: L0 (number of active latents) directly affects absorption metrics. Higher L0 means more latents fire, which means more competition, which could increase absorption. Comparing absorption at different L0 values is comparing different sparsity regimes.

**Verdict**: Either match L0 values for a valid comparison, or explicitly acknowledge this confound invalidates the ablation conclusion.

### 3. Probe Quality Confound in CMI-Absorption Analysis (CRITICAL)

**Problem**: Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). CMI-absorption analysis uses rates from L0=82 where mean probe F1=0.817. Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates.

**Implication**: The CMI-absorption correlation (rho=-0.383, p=0.059 uncorrected at d'=10) may be driven by probe quality, not absorption. The sign reversal at d'>=20 (rho=+0.048 at d'=20) compounds the problem — the correlation is not even stable across dimensions.

**Verdict**: Compute partial correlation controlling for probe F1. Alternatively, restrict CMI analysis to quality-gated letters (F1>0.85).

### 4. MCC~0.21 Across ALL Variants Including Random Control (CRITICAL)

**Problem**: Hungarian matching yields MCC~0.21 at layer 8 for BOTH trained SAE and Random SAE. This suggests the matching procedure is at chance level regardless of training. If Random SAEs (which should have no meaningful structure) achieve MCC~0.21, the matching itself is not capturing genuine hierarchical relationships.

**Implication**: The absorption "reduction" from random to trained SAEs may be an artifact of the matching procedure, not a genuine reduction in hierarchical feature organization.

**Verdict**: Investigate whether Hungarian matching is valid before claiming any comparisons. If MCC is at chance level for Random SAEs, the entire matching pipeline is suspect.

### 5. CMI-Absorption Dimensional Instability (MAJOR)

**Problem**: The CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) reverses sign at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236 at d'=10.

**Sign reversal is qualitative failure, not sensitivity issue**. A correlation that changes sign across dimensions cannot be used to support any directional claim.

**Verdict**: Acknowledge dimension instability. Report results across all dimensions. Do not cherry-pick d'=10 as primary.

### 6. Post-Hoc Power Analysis (MAJOR)

**Problem**: Section 3.6 claims "approximately 20% power to detect a medium effect size (r=0.5)." This is post-hoc power analysis, which is methodologically questionable — power should be computed before the experiment, not after observing null results.

**Verdict**: Remove post-hoc power analysis. Acknowledge insufficient power as a design limitation: "n=26 features provides insufficient power to detect medium effects; null results are inconclusive rather than evidence of no effect."

## What Works

1. **Random baseline comparison (H7)**: Trained SAE (mean=0.034) vs. Random SAE (mean=0.278) is the paper's strongest empirical contribution. This is methodologically sound.

2. **MCP application**: Bonferroni and BH-FDR correction applied to all 12 tests. This is correct methodology.

3. **Cross-layer validation**: L4 and L8 tested independently. The finding that slopes have opposite signs (failing H3) is honestly reported.

4. **Cross-model pilot**: Pythia-70M pilot attempted. Results inconclusive (limited feature overlap) but the attempt is methodologically sound.

5. **Precision-recall decomposition (H5)**: k-sparse probing at k=1, 5, 10, 20 with precision-recall analysis. This is the one robust replicable finding (precision=1.0 universally at k>=5, recall varies).

## Experiment-Specific Recommendations

1. **Acknowledge power limitation upfront** — state in methodology: "n=26 features provides limited power to detect medium effects; detectable r>0.5 only"

2. **Match L0 in OrtSAE comparison** — or remove the ablation conclusion entirely

3. **Control for probe F1 in CMI analysis** — compute partial correlation or restrict to quality-gated letters (F1>0.85)

4. **Investigate MCC pipeline** — if Random SAEs yield MCC~0.21 (chance level), the matching procedure is invalid

5. **Report CMI results across all dimensions** — do not cherry-pick d'=10

6. **Execute H10 or remove it** — deferred is not a valid experimental status

7. **Report Feature U as n=1 observation** — not as evidence for general absorption benignity
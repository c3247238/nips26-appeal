# Critique: Ideation

## Overview

The research proposal is well-structured with clear hypotheses, strong motivation from the SAE Sanity Checks critique, and a well-identified prior art gap. The central H3 finding (absorption as steering signature, not silencing signal) is genuinely novel and compelling. However, there are significant issues with hypothesis design, experimental planning, and how the proposal handles contradictory results.

---

## Critical Issues

### 1. H2 Pilot/Full Contradiction Should Have Been Resolved Before Writing

The proposal (and paper) presents the pilot H2 results as confirmed findings: "TopK achieves 70.9% absorption reduction at 8x reconstruction cost." But the full H2 experiment (h2_mitigation_benchmark.json) shows the OPPOSITE: TopK increases absorption by 4.5x over vanilla SAE.

A good research proposal should have identified this contradiction as a critical unresolved issue before writing the paper. The fact that the paper presents pilot and full-scale results as consistent is a significant methodological failure.

### 2. H5 Pass Criterion Was Not Pre-Registered

The proposal claims "H5: DIRECTIONAL CONFIRMATION (marginal failure)" with a 5% threshold. But the full_h5 experiment was designed with a >8% threshold for causal task delta. The 5% threshold appears nowhere in the experimental design.

A properly pre-registered proposal would have specified the pass/fail criteria BEFORE running the experiment. Creating thresholds post-hoc and then reporting "marginal failure" against a non-existent criterion is p-hacking by another name.

### 3. H1 Instability Was Predictable

The proposal acknowledges two contradictory pilot runs for H1 (layer 4 vs layer 8 ordering reverses between runs) but proceeds to full H1 experiments anyway. The full H1 GPT-2 atlas (full_h1_gpt2_summary.md) shows yet another ordering: monotonic DECREASE from layer 2 (0.307) to layer 10 (0.167).

A well-designed research plan would have:
1. Identified the root cause of instability (token sampling? SAE seed? feature selection?)
2. Designed a controlled experiment to isolate the variable
3. Only proceeded to full-scale after stability was established

Running full experiments before resolving fundamental instability wastes resources and produces uninterpretable results.

---

## Major Issues

### 4. Absorption Metric Was Not Standardized

The proposal mentions multiple absorption metrics:
- Chanin first-letter probe absorption
- Gini absorption (in full_h1_gpt2)
- UAS (proposed unsupervised score)

These produce fundamentally different values for the same SAE:
- First-letter probe: ~0.225 (vanilla SAE, pilot)
- Gini absorption: ~0.015 (same vanilla SAE, full H2)

A research proposal should have standardized on ONE absorption metric BEFORE running experiments. Mixing metrics across experiments makes all cross-experiment comparisons invalid.

### 5. Feature Selection Threshold Mismatch

The proposal states high-absorption features: "UAS > 1.0" and low-absorption: "UAS < 0.3."

The actual full_h3.json data shows:
- High-absorption features have UAS ~0.57 (not > 1.0)
- Features were selected based on Gini absorption, not UAS

The feature selection criterion in the proposal does not match what was actually executed. This is a serious discrepancy that should have been caught before writing.

### 6. Pilot H3 vs Full H3 Contradiction Was Not Adequately Explained

Pilot (N=20): Spearman r = -0.31 (negative correlation)
Full (N=100): Spearman r = +0.35 (positive correlation)

The proposal frames this as "the larger sample size and broader alpha range reveal the true pattern." But:
1. The alpha range was the same [1, 3, 5, 10, 20] in both
2. The sample size increase (20→100) should reduce noise, not reverse the sign
3. No analysis of WHY the sign reversed was conducted

This is a classic sign of an unstable finding. The positive correlation at N=100 could reverse again at N=500.

### 7. Null Controls Were Not Pre-Specified

The proposal mentions null controls as a follow-up experiment ("Add two control conditions: shuffled feature directions (null) and random directions (baseline)") but these were not pre-specified as part of the main H3 experimental design.

As executed, the null controls at alpha=5 show NO difference between high and low absorption features (p=0.94), which contradicts the main 18% difference finding. A properly designed experiment would have specified the null control protocol alongside the main experiment protocol.

---

## Strengths of Ideation

1. **Strong motivation**: The SAE Sanity Checks critique is a genuine existential threat to the field. Addressing it directly is important.

2. **Clear prior art identification**: The proposal correctly identifies that no prior work has measured absorption vs. steering sensitivity directly.

3. **UAS metric is genuinely useful**: The unsupervised absorption score addresses a real practical need (training-time monitoring without labeled probes).

4. **Good alternative candidates**: The alternatives section (cand_b, cand_c, cand_d) shows the team thought carefully about pivot strategies.

5. **Honest negative results**: H3 reversal is reported honestly. The proposal's treatment of negative results is consistently strong across iterations.

---

## Recommendations for Ideation

1. **Standardize absorption metric before running experiments**: Pick one metric and use it consistently. Document WHY that metric was chosen.

2. **Pre-register pass/fail criteria**: Define thresholds BEFORE running experiments, not after.

3. **Resolve instabilities before scaling**: If pilot runs contradict each other, design controlled experiments to isolate the cause before running full-scale.

4. **Pre-specify null controls**: Design null control protocol alongside main experiment protocol.

5. **Investigate pilot/full contradictions**: The sign reversal in H3 (negative to positive) and the metric mismatch in H2 (0.225 to 0.015) both needed root-cause analysis before writing.

6. **Feature selection should match execution**: If UAS > 1.0 threshold was not actually used, update the proposal to match what was done.

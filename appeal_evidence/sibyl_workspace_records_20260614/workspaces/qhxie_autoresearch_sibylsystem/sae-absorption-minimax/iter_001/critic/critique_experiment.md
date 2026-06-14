# Critique: Experimental Design and Execution

## Overview

The experimental program is ambitious (5 hypotheses, 2 models, multiple layers, mitigation methods) but the execution has critical problems: wrong absorption metrics used across experiments, contradictory pilot vs. full-scale results presented as consistent, and H2 pass/fail assessment logic is incorrect. The experiments produced valuable data but the analysis framework has significant gaps.

---

## Critical Issues

### 1. H2 Absorption Metric Inconsistency (Pilot vs. Full)

**Pilot H2** (pilot_h2_summary.md):
- Absorption method: Chanin first-letter probe
- Vanilla SAE: absorption=0.2253, MSE=13.53
- TopK SAE: absorption=0.066, MSE=110.23

**Full H2** (full/h2_mitigation_benchmark.json):
- Absorption method: Gini absorption
- Vanilla (pre-trained): absorption=0.015, MSE=1.48
- TopK SAE: absorption=0.068, MSE=108.07

The vanilla SAE absorption differs by **15x** (0.225 vs 0.015) — this is not random variation, this is different metrics. Comparing TopK's 0.066 to the pilot's 0.225 baseline produces the "70.9% reduction" claim. But comparing the same TopK to the full-scale vanilla baseline (0.015) produces a **4.5x increase**.

The paper uses pilot numbers as its headline H2 results while full-scale results contradict them. This is a fundamental methodological failure.

### 2. H2 Pass/Fail Assessment Logic Is Wrong

The full_h2 assessment code sets `h2_pass: true` when any variant has `absorption > baseline`. But H2's hypothesis states: "Both OrtSAE and ATM **reduce** absorption by >40% relative to vanilla SAE."

The correct condition should be `absorption < baseline * 0.6` (40% reduction). The assessment logic is checking the wrong direction. As a result, the experiment flags H2 as "PASS" when all mitigation variants actually increase absorption.

### 3. H5 Pass Criterion Was Not Honored

**Planned H5**: Simple task delta < 2%, Causal task delta > 10%
**Full H5 actual criteria** (full/full_h5_downstream_tasks.json):
- gpt2_simple_delta_lte_3pct: true
- gpt2_causal_delta_gt_8pct: **false**
- l6_simple_delta_lte_3pct: true
- l6_causal_delta_gt_8pct: **false**
- **h5_confirmed: false**

The causal task criterion was >8% delta, achieved 2.51%. This is a clear failure, not a "marginal fail at the 5% threshold." The paper invents the 5% threshold.

---

## Major Issues

### 4. H3 Feature Selection Threshold Mismatch

**Proposal states**: High-absorption: UAS > 1.0, Low-absorption: UAS < 0.3
**Actual data** (full_h3.json): High-absorption features have UAS ~0.57

The paper claims a UAS > 1.0 threshold but the selected features don't match. This means either:
- The threshold was changed without updating the paper
- The feature selection was done on Gini absorption, not UAS
- The UAS formula was different from what was specified

This discrepancy should be documented and corrected.

### 5. Null Controls Do Not Match Main Experiment Protocol

**Main experiment**: alpha in [1, 3, 5, 10, 20], mean across all alphas
**Null controls**: alpha=5 only

At alpha=5, high and low absorption features show identical steering effects (0.7485 vs 0.7543, p=0.94). The main experiment's 18% difference is driven by alpha=10 and alpha=20 conditions. The null controls tested a condition where the main effect was absent.

This is not a valid null control for an experiment where the main finding is alpha-dependent.

### 6. H1 Layer-Wise Pattern Is Unstable Across All Runs

Four independent measurements of the layer-ordering:
| Run | Layer 4 | Layer 8 | Higher |
|-----|--------|---------|--------|
| Pilot H1+H4 | 0.0363 | 0.0402 | L8 (+10.6%) |
| Pilot H1 pilot (2026-04-26T02:19) | 0.0363 | 0.0402 | L8 (+10.6%) |
| Pilot H1 pilot (2026-04-26T18:01) | 0.0684 | 0.0527 | L4 (-22.9%) |
| Full H1 GPT-2 atlas | 0.267 | 0.200 | L4 (-25.1%) |

Three contradictory orderings across four runs. No stable layer-wise pattern exists in the data. The proposal acknowledges two contradictory pilots but runs more experiments without resolving the instability first.

### 7. H4 Full-Scale Regression

**Pilot H4**: r=0.8147 (layer 4), r=0.7603 (layer 8)
**Full H4 (pilot mode)**: r=0.587 (layer 4), r=0.706 (layer 8)

The correlation gets weaker (not stronger) with more data. The "full" H4 was actually run in "PILOT mode" with only 3 pairs, not full scale.

### 8. Gemma-2B Experiments Skipped

The full_h3_gemma.json shows: `{"experiment": "full_h3_gemma", "status": "skipped", "reason": "Gemma-2B requires HuggingFace authentication"}`

The paper title and outline promise multi-model scope. Without Gemma-2B replication, the H3 finding (r=+0.35) is GPT-2 Small-only. The proposal should have resolved the authentication issue before planning Gemma-2B experiments.

### 9. H5 Simple Task Performance Is Poor

Both absorption bins show low AUC on the causal task (0.547 and 0.522), near the 0.5 random baseline. This suggests the synthetic counterfactual pairs do not reliably engage GPT-2's causal reasoning. The H5 causal task may not be a valid measure of causal discriminability.

---

## Strengths of Experimental Design

1. **Large N for H3**: N=100 features is a significant improvement over the pilot (N=20). This gives adequate statistical power for the Spearman correlation.

2. **Multiple alpha values**: Testing [1, 3, 5, 10, 20] reveals alpha-dependence that would be missed with a single alpha.

3. **Honest negative result reporting**: The H3 pilot/full contradiction, H1 instability, and H5 failure are all reported honestly.

4. **Null controls**: Including random and shuffled direction baselines provides useful context even if the null control protocol has issues.

---

## Recommendations

1. **Standardize on one absorption metric**: Pick Chanin first-letter probe OR Gini absorption and use it consistently across ALL experiments (pilot and full-scale).

2. **Fix H2 assessment logic**: Change the pass condition from `absorption > baseline` to `absorption < baseline * 0.6` (40% reduction).

3. **Pre-register thresholds before running experiments**: Define pass/fail criteria for H5 before running H5.

4. **Run null controls at multiple alpha values**: Test null controls at alpha=5, alpha=10, and alpha=20 to match the main experiment.

5. **Resolve H1 instability**: Audit the experimental pipeline to find why layer ordering is unstable across runs.

6. **Document feature selection threshold**: Verify what threshold was actually used and update the paper to match.

7. **Replicate on Gemma-2B**: Either resolve HF authentication or remove multi-model scope claims from the paper.

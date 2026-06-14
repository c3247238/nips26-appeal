# Critique: Writing Quality

## Overview

The paper is well-motivated and the central H3 finding (r=+0.35, high-absorption features are more steerable) is a genuinely surprising and interesting empirical result. However, the writing has critical accuracy problems: quantitative claims are inconsistent between sections, key results are misrepresented, and important experimental details are omitted or misstated.

---

## Critical Issues

### 1. H2 Mitigation Numbers Are From Pilot, Not Full-Scale

**Location**: Section 4.4 and Table 2

The paper claims "TopK achieves 70.9% absorption reduction at 8x reconstruction cost" — these numbers are from the **pilot** experiment. The **full-scale H2** (h2_mitigation_benchmark.json, `full/` directory) produces:

| Variant | Full-Scale Absorption | vs Vanilla |
|---------|----------------------|-----------|
| Vanilla (pre-trained) | 0.015 | baseline |
| TopK SAE | 0.068 | **+4.5x increase** |
| JumpReLU SAE | 0.627 | +41x increase |
| OrtSAE | 0.629 | +42x increase |
| ATM SAE | 0.574 | +38x increase |

Every mitigation variant has **higher** absorption than vanilla in full-scale. The pilot claims (70.9% reduction) are the **opposite** of full-scale reality (4.5x increase).

The paper presents these as consistent results without noting the contradiction. A reader comparing Table 2 with the full experiment data would find the paper fundamentally misleading.

### 2. H5 Threshold Is Fabricated

**Location**: Section 4.5

The paper states: "This marginal degradation (4.95% vs 5% threshold) is not statistically significant (p=0.42)."

The actual experiment (full/full_h5_downstream_tasks.json) shows:
- Simple task delta: 7.45%
- Causal task delta: **2.51%**
- Actual pass criterion for causal: **>8%**

The paper invented a 5% threshold that does not exist in the experimental design. The correct framing is: **H5 FAILED**. The causal task delta was 5.49 percentage points below the required threshold.

### 3. Scope Mismatch

**Location**: Title, Abstract, Introduction

The title mentions "Empirical Study" and the abstract promises absorption quantification across multiple models and layers. In reality:
- Gemma-2B experiments: **skipped** (HF authentication required)
- H1 layer-wise atlas: **inconclusive** (contradictory across runs)
- Experiments conducted: GPT-2 Small, layer 8 only

The paper title and abstract create expectations the experiments do not fulfill.

---

## Major Issues

### 4. Effect Size Inconsistency

**Location**: Section 4.2 vs Abstract

Section 4.2: "~15% higher"
Abstract: "18%"
Correct calculation: (0.1035/0.0874 - 1) = 18.4%

The text understates the effect by 3.4 percentage points.

### 5. Abstract Overclaims Causality

**Location**: Abstract

"absorption may indicate features at high leverage points in the residual stream" and "causally important features"

The paper measures **steering sensitivity** (how much model outputs change when a feature direction is added). It does NOT measure **causal importance** (what happens when a feature is ablated). These are different things. Steering a feature direction is not the same as removing a feature from computation.

### 6. Missing Figures for Central Finding

**Location**: Section 4.2 (central contribution)

The paper's most important result (Section 4.2, H3 reversed) has NO figures despite the outline planning 5 figures. The outline specifically plans:
- Scatter plot (UAS vs. steering sensitivity)
- Bar chart (high vs. low absorption means)

Without these figures, readers cannot visually verify the correlation or group differences. This is especially critical because the 18% effect size is modest and requires visual support.

### 7. Null Controls Methodology Mismatch

**Location**: Section 4.3

Main experiment: alpha values [1, 3, 5, 10, 20], mean effect computed across all alphas.
Null controls: alpha=5 only, mean effect computed only at alpha=5.

At alpha=5, high and low absorption features show nearly identical effects (0.7485 vs 0.7543, p=0.94). The main experiment's 18% difference is driven by high-alpha conditions (10, 20). The null controls do not replicate the main experiment protocol, making the comparison incomplete.

### 8. Chanin Absorption Score Undefined

**Location**: Section 4.4

The paper references "Chanin absorption score A(f)" but never defines what it measures or its range. Section 4.4 mitigation table uses "Absorption" values (0.2253, 0.066) that are from the pilot's first-letter probe method, not from the full-scale Gini absorption. The metric confusion propagates through the entire H2 section.

### 9. Pilot vs Full-Scale Results Not Distinguished

**Location**: Throughout Results (Section 4)

H2 pilot results (Table 2): absorption=0.225, MSE=13.53 (vanilla)
H4 pilot results: Spearman r=0.8147 at layer 4, 0.7603 at layer 8
H4 full-scale (pilot mode): Spearman r=0.587 at layer 4, 0.706 at layer 8

The paper presents all results without labeling which are pilot and which are full-scale. Readers cannot assess the evidence strength.

---

## What Works Well

1. **H3 finding is genuinely surprising and important**: The positive correlation between absorption and steering sensitivity is counter-intuitive and worth reporting. The honest reporting of the pilot/full contradiction is a strength.

2. **UAS validation is solid**: The correlation results (r=0.65-0.79) are clearly presented. The practical utility of UAS as a training-time monitor is a genuine contribution.

3. **Entanglement hypothesis is a compelling framing**: Presenting the H3 reversal as supporting a named hypothesis elevates the paper's narrative.

4. **Limitations section is honest**: The paper acknowledges single-model scope, layer 8 only, and modest effect size.

---

## Recommendations

1. **Fix or remove H2 claims**: Either (a) remove the pilot H2 numbers entirely, or (b) add explicit comparison between pilot and full H2 showing the contradiction, with analysis of why they differ.

2. **Correct H5 framing**: State clearly that H5 failed on the causal task criterion (2.51% vs >8% required).

3. **Add figures to Section 4.2**: At minimum, add a scatter plot of UAS vs. steering effect and a bar chart of high vs. low absorption mean effects.

4. **Clarify scope**: Change title or add explicit scope statement. "Experiments on GPT-2 Small layer 8" if multi-model results are not available.

5. **Define absorption metric consistently**: Use one absorption metric throughout, or clearly label which metric is used in each comparison.

6. **Add visual showing null controls at multiple alpha values**: The alpha-dependence of the absorption-sensitivity relationship needs empirical support, not post-hoc explanation.

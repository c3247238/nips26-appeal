# Paper Outline

## Title

**Beyond Competitive Exclusion: Hedging Dominance, the L0 Phase Transition, and the Limits of Rate-Distortion Diagnostics in SAE Feature Absorption**

---

## Abstract (~200 words)

- Audit the Chanin absorption metric on Gemma 2 2B JumpReLU SAEs across five hierarchy domains
- Finding 1: Metric does not transfer---shuffled controls exceed true labels in all 5 domains; confound decomposition reveals strict hedging rate of 6.2% (not 98.6%); activation patching on 8 core words yields 0/8 parent recovery, ruling out competitive exclusion
- Finding 2: L0 phase transition---absorption declines monotonically from 42.85% (L0=22) to 0.84% (L0=176), with sharp transition at L0~40--80, stable across three layers (CV<10%)
- Finding 3: CMI-absorption association does not replicate---at L0=82 (rho=-0.383, p=0.059) but at L0=22 with perfect probes (rho=0.044, p=0.835); original correlation driven by probe quality confound
- Recommendation: validate absorption metrics on new SAE architectures before building mitigations

---

## 1. Introduction

### Key arguments
- SAEs are the primary unsupervised tool for mechanistic interpretability; feature absorption undermines their reliability by creating invisible recall holes in parent features
- The absorption finding (Chanin et al., 2024; NeurIPS 2025 Oral) triggered an architectural mitigation wave (Matryoshka SAE, OrtSAE, ATM-SAE, masked regularization), all assuming the diagnostic measures competitive exclusion
- No study has validated the Chanin metric on JumpReLU SAEs (which dominate Gemma Scope) with appropriate controls

### Research questions
- Q1 (Metric validity): Does the metric transfer from L1-ReLU/GPT-2 to JumpReLU/Gemma 2 2B? What fraction is genuine competitive exclusion vs. hedging vs. artifact?
- Q2 (Sparsity dynamics): How does absorption scale with L0? Is there a phase transition?
- Q3 (Theoretical criterion): Can CMI predict absorption susceptibility?

### Three headline findings (1 paragraph each)
1. Metric does not transfer: universal control failure, strict hedging 6.2%, activation patching 0/8 recovery
2. L0 phase transition: monotonic decline, sharp transition at L0~40--80, cross-layer stable
3. CMI does not replicate with perfect probes: probe quality confound is the driver

### Transition
- Sections 2--3 background/methodology; 4--5 metric audit + L0 phase transition; 6 CMI analysis; 7 discussion

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders in Mechanistic Interpretability
- SAE encoding: x -> sparse z via encoder W_e, reconstruction x_hat = W_d z with unit-normalized decoder columns
- L1-ReLU SAEs: continuous L1 penalty, graded suppression
- JumpReLU SAEs: hard per-latent threshold theta_j, binary activation regime (fire or zero)
- L0 operating point: configured number of active latents per forward pass
- Gemma Scope: 400+ open JumpReLU SAEs on Gemma 2 2B/9B/27B
- Theoretical limitations: features neither canonical nor atomic (Leask et al., ICLR 2025); recovery fails unless extremely sparse (Cui et al., 2025)

### 2.2 Feature Absorption
- Chanin et al. (2024) definition: parent z_p = 0 when child z_c > 0 on parent-positive inputs
- Measurement: k-sparse probes + cosine alignment + false-negative rate
- SAEBench rates: 15--35% on first-letter spelling, GPT-2 Small, L1-ReLU only
- Theoretical accounts: partial minima in biconvex optimization (Tang et al., 2025); amortization gap (O'Neill et al., 2024)
- Feature hedging: incorrect L0 merges correlated features, producing absorption-mimicking false negatives (Chanin & Garriga-Alonso, 2025)

### 2.3 Architectural Mitigations
- Matryoshka SAE (Bussmann et al., ICML 2025): nested dictionaries, absorption ~0.03
- OrtSAE (Korznikov et al., 2025): orthogonality penalty, 65% reduction
- ATM-SAE (Li et al., 2025): adaptive temporal masking, absorption 0.007
- Masked regularization (Narayanaswamy et al., 2026): disrupts co-occurrence patterns
- All assume the Chanin metric measures competitive exclusion; none validates on JumpReLU

### 2.4 Rate-Distortion Theory and Successive Refinement
- Successive refinement theorem (Equitz & Cover, 1991): lossless staged encoding iff descriptions form Markov chain
- If X -> f_child -> f_parent is Markov, CMI = 0, absorption is information-theoretically lossless
- When CMI > 0, absorption destroys unique parent information
- No prior work applies this to SAE absorption

---

## 3. Methodology

### 3.1 Models and SAE Configurations
- Primary: Gemma 2 2B (d_model=2304) + Gemma Scope JumpReLU SAEs
  - L12-16k at L0 = {22, 41, 82, 176}; L12-65k; L10-16k; L20-16k
- Secondary: GPT-2 Small (d_model=768) + L1-ReLU SAEs (d_SAE=24,576) at layers 8/10/11
- *Reference: Table 1*

### 3.2 Absorption Measurement Protocol
- Vocabulary: 1,204 single-token words (25 tested letters; X excluded), 1,196 for confound decomposition
- Probe training: k=5 sparse logistic regression on SAE latent activations
- Quality gate: F1 > 0.85 per parent class (10/25 pass at L0=82; 25/25 at L0=22)
- Candidate identification: cos(d_j, v_p) >= 0.025
- Absorption criterion: all k probe latents inactive + correct probe classification
- Bootstrap CIs: 10,000 resamples, seed=42

### 3.3 Four-Control Suite (C1--C4)
- C1 Random probe: random direction, 9.2% absorption (near-zero baseline)
- C2 Shuffled labels: permuted class assignments, 59.5% (invalid baseline---higher than measured)
- C3 Dense probe: all features, F1=0.929 (demonstrates signal exists in activations)
- C4 Untrained SAE: pre-training decoder, 0% (correct null)
- *Reference: Table 2*

### 3.4 Confound Decomposition
- Permissive hedging: any token ceasing to be FN at any higher L0 -> hedging (98.6%)
- Strict hedging: check whether specific parent-associated latents (k=5) fire at L0=176 -> 6.2%
  - Major discrepancy: strict << permissive; headline must use strict
  - z=3.51 vs. shuffled control (3.4%), p<0.001 -> signal is real but small
- Persistent core words: 8 words FN at all 4 L0 values

### 3.5 Activation Patching
- For each of 8 persistent core words: zero child feature at L0=82, check parent recovery
- Control: zero 10 random non-child features, expect no change
- *Reference: Table 5*

### 3.6 Threshold Sensitivity Analysis
- 5x4 grid: cosine {0.01, 0.02, 0.025, 0.03, 0.05} x magnitude gap {0.5, 1.0, 1.5, 2.0}
- CV=0.077 across 20 cells (rate range 0.118--0.151)
- Control failure persists at all 20 combinations
- *Reference: Table 3 / Figure 3*

### 3.7 CMI Estimation
- k-NN mutual information estimator in d'-dimensional decoder subspace
- Primary: d'=10 (pre-registered)
- Comparison: L0=82 (varying probe quality) vs. L0=22 (all probes F1=1.0)
- Partial correlation controlling for probe F1

---

## 4. Metric Audit: Universal Control Failure and Hedging Dominance

### 4.1 Universal Control Failure
- Shuffled-label controls exceed measured absorption in all 5 domains (Table 2)
  - First-letter: 74.6% shuffled vs. 15.96% measured (4.7x)
  - City-continent: 45.2% vs. 6.49%
  - City-language: 18.0% vs. 6.56%
  - Animal-class: 39.3% vs. 1.43%
  - City-country: 10.3% vs. 0.00%
- Random probe (9.2%) and untrained SAE (0%) controls behave correctly

### 4.2 Structural Explanation: Candidate Explosion in High Dimensions
- At cosine >= 0.025 in R^2304, a random unit vector identifies 3,766 of 16,384 decoder columns (23.0%)
- True probes: 3,478 candidates (21.2%); shuffled: 3,502 (21.4%)
- With L0=82, P(at least 1 candidate fires) = 1.0 for all probe types
- The metric reduces to: absorption_rate ~ false_negative_rate, because the candidate step is vacuous
- Dead features (18.8%) inflate candidate counts
- *Reference: Figure 2*

### 4.3 Threshold Sensitivity
- Absorption rate stable across threshold choices (CV=0.077, range 0.118--0.151)
- Letter-level rankings preserved (mean Kendall tau=0.977)
- Magnitude gap matters more than cosine threshold (effect 0.030 vs. 0.010)
- Control failure is structural, not resolvable by threshold tuning
- *Reference: Table 3 / Figure 3*

### 4.4 Confound Decomposition: Permissive vs. Strict Hedging
- 656 FN tokens at L0=22 (all probes F1=1.0)
- Permissive hedging: 98.6% (tokens no longer FN at higher L0)
- Strict hedging: 6.2% (parent-associated latents fire at L0=176)
  - 93.8% of FNs: NONE of 5 parent latents fire at L0=176
  - Control: shuffled strict rate 3.4%, z=3.51, p<0.001
- Letter G is an outlier: 90.5% strict hedging (19/21 FN tokens)
- The 98.6% headline is an upper bound; strict 6.2% is the defensible number
- *Reference: Table 4*

### 4.5 Activation Patching: Ruling Out Competitive Exclusion
- 8 persistent core words: eight(E), liked(L), lower(L), offer(O), often(O), other(O), under(U), until(U)
- Result: 0/8 parent recovery after child zeroing
  - None of the words show parent feature activation recovery
- Random-feature controls: no spurious parent activation
- Interpretation: these FNs are not caused by child-to-parent competitive suppression; the model simply does not represent first-letter information for these tokens via the SAE's parent latent
- *Reference: Table 5*

### 4.6 Probe Quality Confound
- rho(absorption, probe_F1) = -0.69, p < 0.001
- Letters with low probe quality show high absorption; letters with high probe quality show zero absorption
- This confound pervades all metric-based measurements at L0=82

---

## 5. L0 Phase Transition

### 5.1 Monotonic Decline
- L12-16k: 42.85% (L0=22) -> 37.49% (L0=41) -> 14.39% (L0=82) -> 0.84% (L0=176)
- Spearman rho = -1.0 (perfect monotonic decline)
- Bootstrap CIs at each L0 do not overlap
- *Reference: Figure 4*

### 5.2 Phase Transition Region
- Sharp transition between L0=41 and L0=82 (37.49% -> 14.39%)
- L0~40--80 identified as the transition zone
- Below L0~40: absorption > 35% (capacity-starved regime)
- Above L0~80: absorption < 15% (capacity-sufficient regime)

### 5.3 Cross-Layer Stability
- L10-16k: 13.88% at L0=82; L20-16k: 13.55% at L0=82; L12-16k: 14.39%
- CV = 8.6% across layers 10, 12, 20
- The phase transition is a property of the L0 operating point, not the layer

### 5.4 Cross-Architecture Comparison (Confounded)
- GPT-2 Small L1-ReLU: 61.65%--67.29% at layers 8/10/11
- Confounded by: model size, architecture, L0, vocabulary, training data
- Reported as context only; not a controlled comparison

---

## 6. Exploratory CMI Analysis and Its Failure to Replicate

### 6.1 Motivation from Rate-Distortion Theory
- If CMI(X; f_parent | f_child) > 0, absorption destroys unique parent information
- Prediction: lower CMI -> higher absorption susceptibility

### 6.2 Results at L0=82 (Marginal Signal)
- CMI at d'=10: absorbed mean 0.649 (std 0.187) vs. non-absorbed 0.861 (std 0.258)
- Cohen's d = -0.924 (large effect)
- Mann-Whitney p = 0.045
- Spearman rho = -0.383, p = 0.059 (marginal, uncorrected)
- Bonferroni corrected: p = 0.236
- Sign reversal at d' >= 20

### 6.3 Partial Correlation (Controlling for Probe F1)
- Partial rho(CMI, absorption | probe_F1) = -0.328, p = 0.118
- Restricted to F1 > 0.85 (n=10): rho = -0.113, p = 0.757
- Probe quality confound weakens and eventually eliminates the signal
- *Reference: Table 6*

### 6.4 Leave-One-Out Sensitivity
- No single letter removal changes rho by > 0.1 (max |delta| = 0.088, letter V)
- All 25 leave-one-out rho values remain negative (same sign)
- Jackknife SE = 0.186; bias-corrected rho = -0.397
- Removing both S and K: rho = -0.505, p = 0.014 (becomes significant, but this is cherry-picking)
- Conclusion: the marginal signal at L0=82 is stable across individual letter removals but not robust to probe quality controls

### 6.5 Replication at L0=22 (Null Result)
- All 25 probes F1=1.0 -> probe quality confound eliminated
- CMI Spearman rho = 0.044, p = 0.835; Bonferroni p = 1.0
- Sign is POSITIVE (reversed from L0=82)
- At d'=30: rho = 0.410, p = 0.042; d'=50: rho = 0.483, p = 0.014 (both reversed)
- Conclusion: the CMI-absorption association at L0=82 was driven by probe quality, not rate-distortion theory

### 6.6 Dimension Sensitivity
- Sign reversal at d' >= 20 at L0=82
- Positive correlations at all d' values at L0=22
- The direction depends on subspace choice and L0, undermining any theoretical interpretation
- *Reference: Figure 5*

---

## 7. Discussion

### 7.1 Summary of Contributions
- First metric audit on JumpReLU SAEs: universal control failure, strict hedging 6.2%, 0/8 activation patching recovery
- L0 phase transition: absorption is primarily controlled by L0 operating point, not architecture
- Honest negative result: CMI diagnostic does not replicate; probe quality is the confounder

### 7.2 Two Interpretations of the Core Findings
- Interpretation A (metric miscalibration): The Chanin metric is fundamentally miscalibrated for JumpReLU SAEs. Control failure, candidate explosion, and probe quality confounds mean the metric measures false-negative rate, not competitive exclusion.
- Interpretation B (genuine low absorption): JumpReLU SAEs at L0>=82 genuinely have low absorption. The metric works correctly and the answer is that there is very little competitive exclusion.
- Our activation patching result (0/8 recovery) cannot fully disambiguate: the 8 words may be a special case.

### 7.3 Implications for the Mitigation Wave
- Matryoshka SAE, OrtSAE, ATM-SAE, masked regularization all benchmark against the Chanin metric
- If the metric measures false-negative rate rather than competitive exclusion, mitigation benchmarks need recalibration
- L0 operating point is the primary control parameter; architectural interventions may be secondary

### 7.4 Negative Results as Contributions
- 4 of 7 hypotheses falsified; reported with pre-registered targets vs. actual
- CMI non-replication demonstrates the importance of: (a) perfect probes, (b) replication across L0 values, (c) controlling for probe quality
- The strict vs. permissive hedging discrepancy (6.2% vs. 98.6%) shows that hedging classification methodology matters enormously

### 7.5 Limitations
- Single model (Gemma 2 2B) and SAE family (Gemma Scope)
- First-letter spelling as primary task (unusual hierarchy properties)
- Cross-domain results uninterpretable due to control failure
- Activation patching on 8 words is small sample; cannot generalize to all FNs
- CMI estimation sensitive to subspace dimension choice
- Training-free study: cannot assess whether retraining SAEs would change results

### 7.6 Future Directions
- Decoder geometry analysis for unsupervised absorption detection
- Activation patching at scale (beyond 8 words)
- Cross-model validation (Gemma 9B, Llama 3.1)
- SAE retraining experiments: does explicit hierarchy loss reduce absorption?
- Immunological imprinting analogy: test cross-reactive absorption of semantically unrelated co-occurring features

---

## Figure and Table Plan

### Table 1: SAE Configurations (Section 3.1)
- **Purpose**: Summarize all SAE configurations used in the study
- **Type**: comparison_table
- **Content**: Model, layer, d_SAE, L0, architecture for each configuration
- **Key takeaway**: The study systematically varies L0, layer, width, and architecture
- **Generation**: data_table
- **Data source**: proposal.md Section 3.1

### Table 2: Cross-Domain Absorption and Control Results (Section 4.1)
- **Purpose**: Show universal control failure across all 5 hierarchy domains
- **Type**: comparison_table
- **Content**: Domain, measured absorption, shuffled control, random control, mean probe F1
- **Key takeaway**: Shuffled controls exceed measured absorption in ALL domains---the metric is structurally inverted
- **Generation**: data_table
- **Data source**: control_failure_diagnosis.json cross_domain_evidence

### Figure 1: Control Failure Mechanism---Candidate Explosion in High Dimensions (Section 4.2)
- **Purpose**: Explain WHY shuffled controls exceed measured absorption
- **Type**: multi-panel figure (histogram + schematic)
- **Content**: (a) Overlapping histograms of candidate counts for random/true/shuffled probes at cos>=0.025; (b) Schematic: the absorption metric pipeline showing where the candidate identification step becomes vacuous
- **Key takeaway**: At cos>=0.025 in R^2304, random directions identify 23% of decoder columns; the candidate step provides no discriminative power
- **Generation**: code (matplotlib) for histogram; manual_diagram for schematic
- **Data source**: control_failure_diagnosis.json histogram_data

### Table 3: Threshold Sensitivity Heatmap (Section 4.3)
- **Purpose**: Show absorption rate across 5x4 threshold parameter grid
- **Type**: heatmap / ablation_table
- **Content**: 5 cosine thresholds x 4 magnitude gaps, absorption rate per cell, CV, rank stability
- **Key takeaway**: CV=0.077; control failure persists at all 20 combinations; magnitude gap matters more than cosine threshold
- **Generation**: code (seaborn heatmap)
- **Data source**: threshold_sensitivity_summary.json heatmap_data

### Figure 2: Hedging Decomposition---Permissive vs. Strict Classification (Section 4.4)
- **Purpose**: Visualize the discrepancy between permissive (98.6%) and strict (6.2%) hedging rates
- **Type**: stacked bar chart + per-letter breakdown
- **Content**: (a) Stacked bar: total FN tokens split into permissive-hedging/strict-hedging/non-hedging categories; (b) Per-letter strict hedging rate with letter G highlighted as outlier
- **Key takeaway**: The 98.6% headline is an upper bound; only 6.2% of FN tokens show any parent latent firing at L0=176; letter G accounts for 46% of all strict hedging cases
- **Generation**: code (matplotlib)
- **Data source**: tightened_hedging.json

### Table 4: Tightened Hedging Classification Results (Section 4.4)
- **Purpose**: Compare permissive and strict hedging rates with controls
- **Type**: comparison_table
- **Content**: Classification, count, rate, 95% CI, shuffled control
- **Key takeaway**: Strict hedging (6.2%) significantly above shuffled control (3.4%, z=3.51) but dramatically below permissive (98.6%)
- **Generation**: data_table
- **Data source**: tightened_hedging.json tightened_classification + control_comparison

### Table 5: Activation Patching on 8 Persistent Core Words (Section 4.5)
- **Purpose**: Report per-word causal intervention results
- **Type**: data_table
- **Content**: Word, letter, child feature index, child cosine, parent recovered (yes/no), recovery magnitude, control
- **Key takeaway**: 0/8 parent recovery after child zeroing, ruling out competitive exclusion for these tokens
- **Generation**: data_table
- **Data source**: activation_patching_core_words.json persistent_core_words

### Figure 3: L0 Phase Transition (Section 5)
- **Purpose**: Show monotonic absorption decline with L0 and identify the phase transition region
- **Type**: line_plot with shaded CI bands + multi-layer overlay
- **Content**: Absorption rate vs. L0 for L12-16k (primary), with bootstrap CIs; secondary traces for L10-16k, L20-16k at L0=82; shaded region highlighting L0~40--80 transition zone
- **Key takeaway**: Absorption drops from 42.85% to 0.84% as L0 increases from 22 to 176; the sharp transition at L0~40--80 is stable across layers
- **Generation**: code (matplotlib)
- **Data source**: confound_decomposition_multi_l0.json + scaling_surface.json

### Figure 4: CMI vs. Absorption at L0=82 and L0=22 (Section 6)
- **Purpose**: Show that the CMI-absorption correlation vanishes when probe quality confounds are eliminated
- **Type**: scatter plot (two-panel)
- **Content**: (a) L0=82: CMI_d10 vs. absorption rate per letter, colored by probe F1; regression line with rho=-0.383; (b) L0=22: same axes, all probes F1=1.0; regression line with rho=0.044
- **Key takeaway**: The negative correlation at L0=82 disappears at L0=22 where probe quality is perfect; the correlation tracked probe quality, not rate-distortion theory
- **Generation**: code (matplotlib/seaborn)
- **Data source**: partial_correlation_cmi.json per_letter + cmi_replication_l0_22.json

### Figure 5: CMI Dimension Sensitivity and Sign Reversal (Section 6.6)
- **Purpose**: Show that CMI-absorption correlation direction depends on subspace dimension
- **Type**: line_plot (two traces)
- **Content**: Spearman rho vs. subspace dimension d' = {10, 20, 30, 50} for L0=82 and L0=22; horizontal line at rho=0; p-value annotations
- **Key takeaway**: Sign reversal at d'>=20 at L0=82; positive at all d' at L0=22; the theoretical prediction is not robust to analytic choices
- **Generation**: code (matplotlib)
- **Data source**: cmi_estimation.json dimension sensitivity + cmi_replication_l0_22.json

### Table 6: CMI-Absorption Correlation Summary (Section 6)
- **Purpose**: Comprehensive statistical summary of all CMI analyses
- **Type**: comparison_table
- **Content**: Analysis, N, Spearman rho, CI, p-value, interpretation for: raw L0=82, partial (controlling F1), restricted (F1>0.85 only), replicated L0=22
- **Key takeaway**: The signal weakens with every robustness check and vanishes at L0=22
- **Generation**: data_table
- **Data source**: partial_correlation_cmi.json paper_ready_table + cmi_replication_l0_22.json

---

## Visual Summary

| # | Type | Section | Title | Purpose |
|---|------|---------|-------|---------|
| Table 1 | Configuration table | 3.1 | SAE Configurations | List all SAE configs |
| Table 2 | Results table | 4.1 | Cross-Domain Control Results | Universal control failure |
| Figure 1 | Histogram + diagram | 4.2 | Candidate Explosion | WHY controls fail |
| Table 3 | Heatmap | 4.3 | Threshold Sensitivity | Robustness to thresholds |
| Figure 2 | Stacked bar + breakdown | 4.4 | Hedging Decomposition | Permissive vs. strict |
| Table 4 | Summary table | 4.4 | Tightened Hedging | Key statistical comparison |
| Table 5 | Per-word table | 4.5 | Activation Patching | 0/8 recovery result |
| Figure 3 | Line plot + bands | 5 | L0 Phase Transition | Monotonic decline + transition zone |
| Figure 4 | Two-panel scatter | 6 | CMI vs. Absorption | Non-replication with perfect probes |
| Figure 5 | Line plot | 6.6 | CMI Dimension Sensitivity | Sign reversal across d' |
| Table 6 | Summary table | 6 | CMI Correlation Summary | Progressive weakening of signal |

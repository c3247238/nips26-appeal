# Paper Outline

## Title

**The Absorption Tax: Layer-Dependent and Hierarchy-Dependent Feature Absorption Across Semantic Domains in Sparse Autoencoders**

---

## Abstract (150 words)

Sparse autoencoder (SAE) feature absorption -- where general features fail to fire because specific co-occurring features encode overlapping information -- threatens the reliability of SAE-based mechanistic interpretability. All published absorption measurements use a single proxy task (first-letter spelling). We present the first cross-domain and cross-layer absorption characterization, extending measurement to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) on Gemma 2 2B. Three results emerge:

1. Absorption varies 15x across model layers (2.2% at layer 18 to 34.5% at layer 24 for first-letter, F1=0.97), demonstrating that single-layer benchmarks are unrepresentative.
2. Measured absorption rates differ significantly across hierarchy types (Kruskal-Wallis p=0.005, 4/6 pairwise comparisons significant), though this comparison is confounded by differential probe quality (rho=-0.756).
3. Activation patching provides the first interventional causal evidence for feature suppression in SAEs (32.5% recovery vs 1.5% control, d=1.33, p<0.001).

Additionally, we show the widely-cited ~98% hedging rate is near-tautological (strict: 7.9%) and report definitive negative results for three proposed unsupervised absorption detectors (GAS, CMI, Absorption Tax).

---

## 1. Introduction (1.5 pages)

### Key arguments:
- SAE feature absorption creates a false sense of monosemanticity: features appear clean but have systematic recall failures on specific input subsets.
- Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10-40%. Anthropic's successful circuit tracing in Claude 3.5 Haiku demonstrates that reliable features enable powerful mechanistic understanding -- making absorption a critical obstacle.
- Every published absorption measurement rests on a single task (first-letter spelling) at a single model layer. Real knowledge hierarchies differ in structure (imbalanced classes, multi-level depth, context dependence).
- We present the first systematic cross-domain and cross-layer absorption characterization. Our findings reframe the absorption problem: it is both layer-dependent (15x variation) and hierarchy-dependent (ANOVA p=0.005), invalidating single-task, single-layer benchmarks.

### Paragraph structure:
1. **Opening with concrete finding**: Absorption rates on the same SAE vary 15x depending on which model layer is measured (2.2% to 34.5%). Introduce absorption concretely with a running example.
2. **Practical stakes**: DeepMind's 10-40% degradation finding. Anthropic's circuit tracing success with reliable features. The field's dependence on SAE feature quality.
3. **The single-task gap**: All existing absorption measurements (Chanin et al., SAEBench) use first-letter spelling. Describe why this proxy is limiting: 26 classes, near-uniform distribution, 100% co-occurrence by construction.
4. **Our contributions** (bulleted, 4 items):
   - First cross-layer absorption characterization: 15x variation across layers
   - First cross-domain measurement: significant variation across 4 hierarchy types
   - First interventional causal evidence: activation patching (d=1.33)
   - Tightened hedging showing ~98% figure is near-tautological + honest negative results (GAS, CMI, Tax)
5. **Pointer to Figure 1** (teaser): Layer x hierarchy absorption heatmap showing the interaction.

> **Figure 1** placement: End of Introduction

---

## 2. Background and Related Work (1.5 pages)

### Key arguments:
- Position this work precisely in the SAE absorption literature, distinguishing from related but distinct failure modes (hedging, inconsistency, dark matter).
- Summarize the canonical absorption measurement method (Chanin et al.) and explain why first-letter spelling is the universal proxy task.
- Review architectural mitigations (Matryoshka, OrtSAE, ATM, KronSAE) and note that all comparisons use first-letter only.

### Paragraph structure:
1. **SAE background** (2-3 sentences): Sparse autoencoders decompose polysemantic neural activations into approximately monosemantic features. Key references: Toy Models of Superposition (Elhage et al., 2022), Towards Monosemanticity (Bricken et al., 2023), scaling to frontier models (Templeton et al., 2024; Gao et al., 2024).
2. **Feature absorption defined**: Chanin et al. (2024): a general (parent) feature fails to fire when a specific (child) feature co-occurs, because the SAE achieves better sparsity by absorbing the parent's information into the child's decoder direction. Present the mechanistic example: "snake" fires only the snake feature, not "starts with s."
3. **The absorption-reconstruction tradeoff**: Absorption saves +1 L0 per parent-child pair. Any absorption-free solution requires higher L0. This is a fundamental tension, not a training bug.
4. **Adjacent failure modes**: Feature hedging (Chanin et al., 2025) -- correlated features merge due to insufficient width. Feature inconsistency (Song et al., 2025) -- different runs converge to different features. Dark matter (Engels et al., 2024) -- unexplained reconstruction error. Distinguish each from absorption.
5. **Architectural mitigations**: Matryoshka SAE (Bussmann et al., 2025, absorption ~0.03), OrtSAE (Korznikov et al., 2025, -70%), ATM (Li et al., 2025, 0.0068), masked regularization (Narayanaswamy et al., 2026). All evaluated on first-letter only.
6. **The evaluation gap**: SAEBench (Karvonen et al., 2025) standardized 8-metric evaluation but uses first-letter for absorption. No cross-domain or cross-layer characterization exists.

---

## 3. Method (2 pages)

### Key arguments:
- Describe the experimental setup: model, SAEs, hierarchy definitions, probe training, absorption measurement pipeline, and activation patching procedure.
- Justify each design choice and state quality gates.

### 3.1 Model and SAEs
- Model: Gemma 2 2B (unsloth/gemma-2-2b)
- SAEs: Gemma Scope JumpReLU 16k and 65k at layers 6, 12, 18, 24 (8 configs). SAEBench BatchTopK 16k and Matryoshka 32k at layer 12 (2 additional configs for architecture comparison).
- All experiments are training-free (inference-only on pre-trained SAEs).

### 3.2 Feature Hierarchies
- **First-letter** (syntactic): Token -> first letter. 26 classes, near-uniform distribution. sae_spelling pipeline with ICL prompts. Probe F1=0.97 across all layers.
- **City-continent** (factual, coarse): City -> continent. 6 classes, using RAVEL dataset (~200 cities). ICL prompts with position -1 extraction. Best probe F1=0.843 at layer 24.
- **City-country** (factual, fine): City -> country. 80 classes, highly imbalanced. Best probe F1=0.789 at layer 24.
- **City-language** (factual, medium): City -> primary language. 50 classes. Best probe F1=0.823 at layer 24.

> **Table 1** placement: Probe quality heatmap (hierarchy x layer)

### 3.3 Probe Training
- L2-regularized logistic regression (sklearn, C=1.0)
- Stratified 80/20 split, seed=42
- Trained at layers 6, 12, 18, 24 for all 4 hierarchies (16 probes)
- Quality gate: F1 >= 0.90 (strict), 0.85 (relaxed). Only first-letter passes strict gate.
- Caveat: RAVEL probes are below strict quality gate; cross-domain absorption rates carry probe-quality uncertainty.

### 3.4 Absorption Measurement
- Adapted from Chanin et al.: identify false negatives (probe classifies correctly on raw activations, incorrectly on SAE-reconstructed activations), run integrated-gradients attribution (10 steps), detect absorption via cosine threshold (>0.025) and magnitude gap (>=1.0).
- Bootstrap 95% CI (10k resamples, seed=42)
- Paired permutation test for cross-domain vs first-letter comparison

### 3.5 Activation Patching
- For each word with detected absorption: zero the highest-IG child feature in SAE output, measure recovery of correct probe prediction
- Control: zero 15 magnitude-matched random features
- 200 contexts per word
- Statistical tests: Wilcoxon signed-rank, Mann-Whitney U, bootstrap CI

### 3.6 Hedging Decomposition
- Three-category classification of false negatives:
  - **Strict hedging**: parent feature itself recovers at higher L0
  - **Compensatory resolution**: other (non-parent) features compensate at higher L0
  - **Persistent**: false negative persists at all tested L0 levels

> **Figure 2** placement: Method schematic showing the pipeline

---

## 4. Cross-Domain and Cross-Layer Absorption (2 pages) -- PRIMARY RESULT

### Key arguments:
- Absorption varies dramatically across model layers (15x for first-letter, the most cleanly measured finding).
- Measured absorption rates differ significantly across hierarchy types (ANOVA p=0.005).
- The probe quality confound must be acknowledged transparently.

### 4.1 Layer Dependence (Strongest Finding)

First-letter absorption across 8 Gemma Scope JumpReLU configs:

| Layer | 16k | 65k |
|-------|-----|-----|
| L6 | 2.4% | 2.4% |
| L12 | 5.7% | 9.2% |
| L18 | 2.2% | 4.5% |
| L24 | 34.5% | 25.5% |

15x variation (2.2% to 34.5%). This measurement is unconfounded because first-letter probes achieve F1=0.97 at all layers. L24 rates (25-35%) align with Chanin et al.'s published 15-35% range, suggesting prior work likely measured at later layers. The absorption surge at layer 24 is consistent with the model resolving its final prediction at the last residual stream position.

> **Figure 3**: Bar chart of first-letter absorption across 8 SAE configs, grouped by layer, with bootstrap 95% CI. Clear visual showing the L24 spike.

### 4.2 Cross-Domain Variation

At layer 24 (best probe quality for RAVEL hierarchies):

| Hierarchy | L24_16k | L24_65k | Probe F1 |
|-----------|---------|---------|----------|
| First-letter | 34.5% | 25.5% | 0.971 |
| City-continent | 35.8% | 26.0% | 0.843 |
| City-country | 18.5% | 12.7% | 0.789 |
| City-language | 13.6% | 13.6% | 0.823 |

Kruskal-Wallis hierarchy effect: p=0.005. Four of six pairwise comparisons significant (city-country and city-language vs first-letter at both widths, p=0.0001 to 0.015).

### 4.3 The Probe Quality Confound

Probe quality correlates strongly with false negative rate (rho=-0.756, p<0.001). First-letter's near-perfect probe (F1=0.97) detects nearly all correct classifications in the raw condition, creating a large denominator for the absorption rate. RAVEL probes (F1=0.79-0.84) miss more classifications in the raw condition, potentially masking absorption. Present probe-only FN baselines. State this limitation prominently.

> **Table 2**: Cross-domain absorption rates with probe quality, bootstrap CI, and pairwise p-values vs first-letter.

> **Figure 4**: Heatmap of absorption rate (hierarchy x layer x SAE width). Shows the layer-hierarchy interaction visually.

---

## 5. Causal Evidence and Hedging Decomposition (1.5 pages)

### Key arguments:
- Activation patching confirms absorption causally: zeroing child features recovers parent predictions.
- Tightened hedging reveals the widely-cited ~98% hedging figure is near-tautological.

### 5.1 Activation Patching

32.5% mean recovery vs 1.5% control (diff=0.310, Wilcoxon p=0.000218, Cohen's d=1.33). 16/19 words with detected absorption show positive recovery. Largest effects: conmigo (100% recovery), wikk (66.7%), backward (50.0%), zorgt (50.0%).

Discovery procedure transparency: 7 pilot words + 18 discovered via IG-guided search. Discovery procedure is biased toward finding absorption; the 58.8% absorption rate among tested words reflects selection, not corpus-wide rate.

Restricted analysis on high-confidence words (raw accuracy >= 0.50): report this subset separately to confirm causal effect generalizes to well-represented tokens.

> **Figure 5**: Paired dot plot / violin plot showing recovery rate (child-zeroed vs control) per word. Each word is a connected pair. The separation between treatment and control is visually striking.

> **Table 3**: Activation patching per-word results with raw accuracy, absorbed count, recovery rate, and control recovery.

### 5.2 Tightened Hedging Classification

At base L0=22, target L0=176, n=304 first-letter false negatives:
- Strict hedging: 7.9% (parent feature itself recovers)
- Compensatory resolution: 86.2% (other features compensate)
- Persistent: 5.9% (false negative persists)
- Loose hedging: 94.1% (strict + compensatory)

The 86.2% compensatory resolution demonstrates that at 8x higher L0, most false negatives are resolved by unrelated features adding information, not by the parent feature re-emerging. The widely-cited ~98% hedging rate is thus a near-tautological upper bound reflecting combinatorial inevitability.

> **Figure 6**: Stacked bar chart showing hedging decomposition (strict / compensatory / persistent) at different L0 multipliers.

---

## 6. Architecture Comparison (0.5 pages)

### Key arguments:
- Report honestly: architecture effect is not significant (Kruskal-Wallis p=0.87).
- Hierarchy effect dominates (p=0.005).
- Matryoshka shows lowest rates on 3/4 cross-domain hierarchies at L12 but differences are not significant.

Single paragraph with supporting table:

| Hierarchy | JumpReLU_16k | JumpReLU_65k | BatchTopK_16k | Matryoshka_32k |
|-----------|-------------|-------------|-------------|-------------|
| First-letter | 0.7% | 1.3% | 3.4% | 1.4% |
| City-continent | 17.3% | 23.1% | 13.5% | 19.2% |
| City-language | 41.2% | 38.2% | 61.8% | 35.3% |
| City-country | 47.1% | 47.1% | 52.9% | 35.3% |

Note that BatchTopK operates at L0=20 vs JumpReLU L0~75-87, making direct comparison confounded by sparsity level. Architecture comparison limited to layer 12 (only layer with all architectures available). Report power analysis: at N=16 observations, the minimum detectable effect for Kruskal-Wallis is large; the test is uninformative for moderate effects.

> **Table 4**: Architecture comparison at layer 12.

---

## 7. Negative Results (1 page)

### Key arguments:
- Three proposed unsupervised absorption detectors fail definitively. Report with the same rigor as positive results.
- Threshold sensitivity analysis confirms absorption is structural, not a detection artifact.

### 7.1 Geometric Absorption Score (GAS)

GAS (rho=0.116, AUROC=0.571) fails at 25x scale-up (5000 sequences, 640k tokens). Decoder geometry captures potential for absorption but not which features are actually suppressed during encoding. The frequency asymmetry term introduces noise for rare features.

### 7.2 Conditional Mutual Information (CMI)

CMI at L0=22: rho=0.044, p=0.83. Binary CMI formulation (A=0 vs A>0) loses fine-grained activation magnitude information.

### 7.3 Absorption Tax Ranking

T(G) ranking prediction: Spearman rho=-0.20, Kendall tau=0.0, pairwise concordance 50%. Per-letter R_pc vs absorption: rho=0.09-0.17 across 8 SAE configs (all NS). The cos-squared redundancy ratio formulation does not capture the functional dynamics of absorption.

### 7.4 Threshold Sensitivity

5x4 grid (cosine threshold x magnitude gap): false negatives constant at n=87 across all 20 cells. Absorption rate CV=0.077 (range 11.8-15.1%). Absorption is structural, not threshold-dependent. Probe quality (rho=-0.756) is a stronger predictor than any detection threshold.

---

## 8. Discussion (1 page)

### Key arguments:
1. **Practical implications for SAE evaluation**: Single-task, single-layer absorption benchmarks (SAEBench, sae-spelling) are insufficient. Layer 12 rates (2-9%) dramatically understate L24 rates (25-35%) for first-letter. Cross-domain rates differ significantly. The field needs multi-task, multi-layer evaluation.
2. **Layer-position mechanism hypothesis**: The L24 absorption spike coincides with the model resolving its final prediction. At earlier layers, the residual stream may carry distributed representations that SAEs encode without hierarchical competition. At L24, the representation sharpens toward specific tokens, creating the conditions for child-parent competition.
3. **Limitations**:
   - RAVEL probes below strict quality gate (F1=0.79-0.84); cross-domain rates have quantitative uncertainty
   - Activation patching restricted to first-letter hierarchy; cross-domain causal evidence absent
   - Architecture comparison at L12 only, underpowered
   - Gemma 2 2B only; generalization to other model families untested
4. **Broader context**: The activation patching result provides the strongest available evidence that absorption reflects a genuine competitive exclusion mechanism, not merely a measurement artifact. Combined with layer dependence, this suggests absorption is an intrinsic property of SAE encoding under sparsity constraints, not an incidental failure mode.

---

## 9. Conclusion (0.5 pages)

- Absorption is layer-dependent (15x), hierarchy-dependent (p=0.005), and causally confirmed (d=1.33).
- Single-task single-layer benchmarks are insufficient for absorption characterization.
- Three unsupervised detectors fail; absorption currently requires supervised probe-based measurement.
- Future work: degraded-probe ablation to disentangle probe-quality confound; cross-domain activation patching with improved probes; extension to safety-relevant features and other model families.

---

## Figure & Table Plan

### Figure 1: Layer-Hierarchy Absorption Heatmap (Section: Introduction)
- **Purpose**: Communicate the paper's central finding in a single glance -- absorption varies dramatically by both layer and hierarchy type
- **Type**: heatmap
- **Content**: 4 layers x 4 hierarchies x 2 widths, color = absorption rate, annotation = rate value
- **Key takeaway**: No single absorption rate characterizes an SAE; both layer and hierarchy matter
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `phase1/absorption_firstletter.json`, `phase1_absorption_crossdomain.json`

### Figure 2: Experimental Pipeline Schematic (Section: Method)
- **Purpose**: Show the complete measurement pipeline from model input to absorption rate
- **Type**: flow_chart
- **Content**: Model -> residual stream -> SAE encoding -> probe classification -> false negative detection -> IG attribution -> absorption/hedging classification. Side branch: activation patching (zero child -> measure recovery)
- **Key takeaway**: Readers understand the full measurement protocol at a glance
- **Generation**: tikz or manual_diagram
- **Data source**: N/A (conceptual)

### Figure 3: First-Letter Absorption Across Layers (Section: 4.1)
- **Purpose**: Show the 15x layer-dependent variation with error bars
- **Type**: bar_chart (grouped)
- **Content**: 4 layers x 2 widths, y-axis = absorption rate, error bars = bootstrap 95% CI
- **Key takeaway**: L24 shows dramatically higher absorption (25-35%) vs other layers (2-9%)
- **Generation**: code (matplotlib)
- **Data source**: `pilots/phase1_absorption_firstletter.json`

### Figure 4: Cross-Domain Absorption at L24 (Section: 4.2)
- **Purpose**: Compare absorption across hierarchy types at the best-probed layer
- **Type**: bar_chart (grouped by hierarchy, hue by SAE width)
- **Content**: 4 hierarchies x 2 widths at L24, with bootstrap CI. Annotate probe F1 for each hierarchy.
- **Key takeaway**: Absorption rates differ across hierarchy types, but probe quality varies too
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `pilots/phase1_absorption_crossdomain.json`, `phase1/probe_training_full.json`

### Figure 5: Activation Patching Recovery (Section: 5.1)
- **Purpose**: Visualize the causal effect of child feature zeroing on parent recovery
- **Type**: paired dot plot or box/violin plot
- **Content**: Each word as a pair (child-zeroed recovery vs control recovery). Overlay median and CI.
- **Key takeaway**: Zeroing child features recovers parent predictions at 32.5% (vs 1.5% control, d=1.33)
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `phase0/activation_patching_full.json`

### Figure 6: Hedging Decomposition (Section: 5.2)
- **Purpose**: Show that the loose hedging metric is near-tautological
- **Type**: stacked_bar_chart
- **Content**: At L0=22->176: strict (7.9%), compensatory (86.2%), persistent (5.9%). Optionally: second bar at L0=82->176 for comparison.
- **Key takeaway**: 86.2% of "hedging" is compensatory (unrelated features adding info), not parent recovery
- **Generation**: code (matplotlib)
- **Data source**: `pilots/phase0_tightened_hedging_full.json`

### Table 1: Probe Quality Across Hierarchies and Layers (Section: 3.2)
- **Purpose**: Establish the measurement foundation and transparency about probe quality limits
- **Type**: data_table
- **Content**: 4 hierarchies x 4 layers, cells = F1 score, color-coded by quality gate (pass/acceptable/below)
- **Key takeaway**: Only first-letter passes strict gate; RAVEL probes best at L24 (0.79-0.84)
- **Generation**: data_table (LaTeX)
- **Data source**: `phase1/probe_training_full.json`

### Table 2: Cross-Domain Absorption Rates with Statistical Tests (Section: 4.2)
- **Purpose**: Present the primary quantitative result with full statistical details
- **Type**: comparison_table
- **Content**: Hierarchy x SAE config: first-letter rate, cross-domain rate, difference, Cohen's d, p-value, significance
- **Key takeaway**: 4/6 pairwise comparisons significant (p=0.0001 to 0.015)
- **Generation**: data_table (LaTeX)
- **Data source**: `pilots/phase1_absorption_crossdomain.json`

### Table 3: Activation Patching Per-Word Results (Section: 5.1)
- **Purpose**: Transparent reporting of all tested words with discovery source
- **Type**: data_table
- **Content**: Word, letter, source (pilot/discovered), raw accuracy, absorbed count, child-zeroed recovery, control recovery
- **Key takeaway**: 16/19 words show positive recovery; large effects on both common and rare tokens
- **Generation**: data_table (LaTeX)
- **Data source**: `phase0/activation_patching_full.json`

### Table 4: Architecture Comparison at Layer 12 (Section: 6)
- **Purpose**: Report the architecture null result with supporting data
- **Type**: comparison_table
- **Content**: 4 architectures x 4 hierarchies, cells = absorption rate. Bold lowest per row. Note L0 values.
- **Key takeaway**: Architecture effect NS (p=0.87); hierarchy effect dominates (p=0.005)
- **Generation**: data_table (LaTeX)
- **Data source**: `pilots/phase4_architecture_comparison.json`

### Table A1: Threshold Sensitivity Grid (Appendix)
- **Purpose**: Confirm absorption is structural, not detection-threshold-dependent
- **Type**: heatmap / data_table
- **Content**: 5 cosine thresholds x 4 magnitude gaps, cells = absorption rate. FN count constant.
- **Key takeaway**: CV=0.077; false negatives constant at n=87 across all 20 cells
- **Generation**: code (matplotlib heatmap) or data_table
- **Data source**: `phase0/threshold_sensitivity_report.md`, iter_001 data

### Figure A1: GAS Validation Scatter (Appendix)
- **Purpose**: Document the GAS negative result visually
- **Type**: scatter
- **Content**: x = GAS score per letter, y = absorption rate per letter. Include rho and p-value annotation.
- **Key takeaway**: No correlation (rho=0.116, p=0.58)
- **Generation**: code (matplotlib)
- **Data source**: `phase2/gas_full.json`

### Figure A2: CMI and Absorption Tax Negative Results (Appendix)
- **Purpose**: Document CMI and Tax negative results
- **Type**: scatter (2-panel)
- **Content**: Panel A: CMI vs absorption. Panel B: T(G) rank vs observed rank per hierarchy.
- **Key takeaway**: CMI rho=0.044 (p=0.83); Tax ranking rho=-0.20
- **Generation**: code (matplotlib)
- **Data source**: `phase0/cmi_l0_22.json`, `phase3/tax_qualitative.json`

---

## Appendix Structure

- **Appendix A**: Extended probe quality analysis (per-class F1, confusion matrices, probe-only FN baselines)
- **Appendix B**: GAS negative result (full analysis with failure mode diagnosis)
- **Appendix C**: CMI negative result (analysis of binary vs continuous formulations)
- **Appendix D**: Absorption Tax theoretical framework (qualitative T(G) derivation) and empirical negative result
- **Appendix E**: Threshold sensitivity analysis (5x4 grid, subtype taxonomy stability)
- **Appendix F**: Per-letter absorption breakdown for first-letter across 8 SAE configs
- **Appendix G**: Example false negatives with qualitative analysis (city examples from cross-domain)

---

## Transition Logic

**Intro -> Background**: "Before presenting our measurements, we formalize the absorption phenomenon and review existing evaluation methodology."

**Background -> Method**: "We now describe our experimental setup for systematically characterizing absorption across layers, hierarchies, and architectures."

**Method -> Section 4**: "With this pipeline, we first examine absorption variation across layers (Section 4.1), then across hierarchy types (Section 4.2), and confront the probe quality confound directly (Section 4.3)."

**Section 4 -> Section 5**: "Having established that absorption rates vary by layer and hierarchy, we now provide causal evidence for the absorption mechanism and decompose the role of hedging."

**Section 5 -> Section 6**: "We briefly examine whether SAE architecture modulates absorption across hierarchy types."

**Section 6 -> Section 7**: "We also report three failed attempts at unsupervised absorption detection, which inform future detector design."

**Section 7 -> Discussion**: "These results -- both positive and negative -- reshape our understanding of feature absorption as a fundamental SAE failure mode."

---

## Writing Strategy Notes

1. **Lead with layer dependence** (Section 4.1) as the strongest, unconfounded finding before introducing the confounded cross-domain result.
2. **Probe quality caveat is front-and-center** in Section 4.3, not buried in limitations. This addresses the primary reviewer concern preemptively.
3. **Activation patching** is the paper's second pillar -- present the d=1.33 effect size prominently, with transparent reporting of word selection bias.
4. **Negative results** get a dedicated section (not just appendix mentions) because they are a genuine contribution: preventing others from pursuing GAS/CMI/Tax approaches.
5. **Architecture comparison** is compressed to 0.5 pages given the null result. One paragraph + one table.
6. **Target length**: 8 pages main text + appendix. NeurIPS/ICLR format.

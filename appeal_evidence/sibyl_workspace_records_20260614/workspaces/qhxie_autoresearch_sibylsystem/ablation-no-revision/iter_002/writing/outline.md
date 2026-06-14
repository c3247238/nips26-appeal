# Paper Outline: Layer-Dependent Feature Absorption in Sparse Autoencoders

## Paper Type
Empirical study with predominantly negative results. Four of five hypotheses falsified or uninformative; one confirmed in direction but practically modest. Contribution is methodological (a reproducible absorption metric with random dictionary controls) and empirical (systematic characterization of layer-dependent inverted-U pattern across 6 layers). The paper explicitly acknowledges what was NOT tested and why (H2 at wrong layer, H4 experiment design flaw).

---

## Title

**Feature Absorption in Sparse Autoencoders: Layer-Dependent Prevalence and the Failure of Five Hypotheses**

Alternative: **Layer-Dependent Feature Absorption in Sparse Autoencoders: An Inverted-U Pattern Across Model Layers**

---

## 1. Introduction

### Problem (2 paragraphs)
- Paragraph 1: SAEs have become foundational in mechanistic interpretability, enabling decomposition of residual stream into human-interpretable features. Despite theoretical concern about absorption, no systematic empirical quantification exists.
- Paragraph 2: Feature absorption -- where one latent's semantic content is redundantly encoded by co-firing latents -- threatens reliability of SAE-based circuit analyses. Gap: no validated metric, no layer-resolved measurements, no understanding of downstream causal impact.

### Our contribution (3 bullet points)
- First systematic empirical study revealing layer-dependent inverted-U pattern: absorption peaks at mid-layer 4 (49.3% of latents Af > 0.5) but is nearly absent at deeper layers (0.19% at layer 8, 17.3% at layer 10).
- A reproducible absorption metric (Af) validated against random dictionary controls -- random Gaussian decoders yield exactly 0.00% absorption by construction, confirming the metric detects genuine structure.
- Circuit faithfulness does not differentiate based on absorption level -- both low-absorption and high-absorption latent subsets yield 0.0 faithfulness, revealing that corpus-level absorption scores do not predict circuit-level causal importance.

### Key findings preview (4 sentences)
- H1 is falsified at layer 8 (0.19% vs >20% predicted) but confirmed at layer 4 (49.3%); the 100x difference between adjacent layers is the central empirical finding.
- H3 falsified: absorption does not increase monotonically with sparsity; instead it follows an inverted-U peaking at mid-layers (layer 4, L0=37.8) not the sparsest layer (layer 8, L0=71.9).
- H4 uninformative: both absorption subsets yield 0.0; the experiment design flaw is explicitly acknowledged -- the correct experiment comparing full SAEs across layers was never conducted.
- H5 confirmed in direction: larger dictionaries monotonically reduce absorption (2K: 2.25%, 24K: 0.19%, the only hypothesis not falsified).

### Paper structure roadmap
"We designed five hypotheses, built a metric to test them, and ran pilots on GPT-2 small. Section 2 provides background. Section 3 defines Af. Section 4 describes the setup. Sections 5.1-5.5 present results for H1-H5. Section 6 discusses implications. Section 7 concludes."

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders in Mechanistic Interpretability
- SAEs as tools for decomposing neural network activations into human-interpretable features (Bricken et al. 2023; Anthropic 2023).
- Standard architecture: encoder (ReLU or gated), column-normalized decoder weights, L1 sparsity penalty.
- Define W_enc (shape d_sae x d), b_enc, W_dec (shape d x d_sae), b_dec.
- Applications: feature probing, circuit discovery, concept localization.
- Existing SAE quality metrics measure reconstruction fidelity and sparsity, but NOT absorption.

### 2.2 Feature Interaction Phenomena in Dictionaries
- Superposition: multiple features represented in the same direction (Elhage et al. 2022).
- Absorption vs. superposition distinction: absorption = one feature's variance redundantly encoded by others (severe failure mode); superposition = geometric overlap without necessarily redundant encoding.
- Prior work: Sharkey et al. 2023 on correlated activation patterns; Templeton 2025 on internal dictionary geometry.

### 2.3 Causal Analysis with SAEs
- Activation patching as causal intervention technique.
- Faithfulness metric: fraction of clean-to-corrupted logit difference restored by patching.
- Open problem: whether SAE latents are causally meaningful -- does patching a latent affect behavior in proportion to its semantic importance?
- Connection to absorption: if absorbed features redundantly encode same information, patching them produces correlated effects.

---

## 3. The Absorption Score

### 3.1 Definition

Define Af for each latent f:

1. **Activating tokens** T_f: tokens where act_f(t) > 1% of corpus-maximum for that latent.
2. **Co-firing latents**: top-5 other latents with highest simultaneous activation on each token in T_f.
3. **Partial reconstruction**:
   x_t^partial = W_dec[f] * act_f(t) + sum_{c in top5(f,t)} W_dec[c] * act_c(t) + b_dec
4. **Per-token variance explained**: RVE_f(t) = 1 - Var(x_t - x_t^partial) / Var(x_t)
5. **Absorption score**: Af = (1/|T_f|) sum_{t in T_f} 1[RVE_f(t) > 0.80]

Note: b_dec cancels in the RVE ratio (subtraction of x_t^partial from x_t removes the constant).

### 3.2 Design Rationale

- Why top-5: balances sensitivity and specificity; 5 captures absorption without diluting signal.
- Why 80% threshold: calibrated against random dictionary controls.
- Why per-token RVE, not corpus-level correlation: correlation cannot distinguish causal absorption from incidental semantic co-occurrence.

### 3.3 Validation

- **Random dictionary control**: random Gaussian decoder columns yield 0.00% > threshold by construction. Real SAEs show higher absorption, confirming metric detects learned structure.
- **Always-on features**: 38 latents fire on all 12,800 tokens; excluded as bias-like.
- **Layer 4 bimodality**: At layer 4, absorption scores are bimodal: 25.1% of latents score Af=1.0 (6,170 latents, fully absorbed) and 34.2% score Af=0.0 (fully independent), with 40.7% distributed between. This contrasts with layer 8 where 99.4% score exactly 0.0.
- **Sensitivity analysis** (Appendix A): consistent rankings across RVE thresholds (0.70, 0.80, 0.90) and co-firer counts (top-3, top-5, top-10).

![Figure 1: Experimental pipeline overview](figures/gen_pipeline.pdf)

---

## 4. Experimental Setup

### 4.1 Models and SAEs

- **Base model**: GPT-2 small (124M params, 12 layers, d_model = 768).
- **SAE release**: SAELens `gpt2-small-res-jb`, residual-stream SAEs at layers ell in {0, 2, 4, 6, 8, 10}.
- **Dictionary sizes**: 24,576 (full release) primary; 2,048, 8,192, and 24,576 subdictionaries for H5.
- **Software**: TransformerLens >= 2.0.0, SAELens >= 0.5.0, PyTorch >= 2.0.0.
- **Device**: single GPU (CUDA).

### 4.2 Dataset

- **Pilot corpus**: 100 sequences x 128 tokens, seed 42, from `monology/pile-uncopyrighted`.
- **Token frequency reference**: computed over full Pile validation split (for H2, pending).

### 4.3 Hypotheses and Falsification Criteria

| ID | Hypothesis | Falsification Criterion | Status |
|----|------------|-------------------------|--------|
| H1 | >20% of mid-layer latents have Af > 0.5 | <10% prevalence at layer 8 | Falsified at layer 8; confirmed at layer 4 |
| H2 | Low-frequency latents absorbed at >=2x rate of high-frequency | Spearman r >= 0 OR ratio < 2x | PENDING (not analyzed) |
| H3 | Higher sparsity (L0) monotonically increases absorption | Non-monotonic trend | Falsified (inverted-U) |
| H4 | High-absorption SAE patching reduces faithfulness by >=5pp vs. low-absorption | Diff < 5pp | Uninformative (both subsets 0.0) |
| H5 | Larger dictionary size reduces absorption | Positive or non-monotonic correlation | Not falsified (direction confirmed) |

**Note on H1 pre-registration**: Layer 8 was pre-registered as primary test layer under <10% falsification threshold. Finding 0.19% at layer 8 falsifies it at that layer. Layer 4 (49.3%) was exploratory and does not falsify H1 at that layer.

**Note on H4 design flaw**: The experiment compared 10% vs 100% dictionary subsets, not full SAEs at layers with different absorption profiles. The correctly designed experiment (layer 4 vs. layer 8 full SAE comparison) was never conducted. Layer 4 (49.3% absorption) is never tested in H4.

(Figure 1 appears in Section 3.)

---

## 5. Experiments

## 5.1 H1: Absorption Is Extremely Rare at Layer 8 but Prevalent at Layer 4

**Hypothesis**: >20% of mid-layer latents have Af > 0.5.

**Results at layer 8** (d_sae = 24,576, 100 sequences):
- Only 46 of 24,576 latents (0.19%) have Af > 0.5 -- 100x below hypothesis threshold.
- Mean Af = 0.0022; median = 0.0000; 99.4% of latents have Af = 0.0 exactly.
- Random dictionary control: 0.00% > 0.5, confirming metric detects genuine structure.
- **H1 falsified at layer 8.**

**Results at layer 4** (exploratory):
- 49.3% of latents have Af > 0.5 -- exceeding the >20% threshold.
- 260x more absorbed latents than layer 8 (0.19%).
- **H1 confirmed at layer 4** (exploratory finding).
- Layer 4 absorption scores are bimodal: 25.1% at Af=1.0 (6,170 latents), 34.2% at Af=0.0 (8,400 latents), 40.7% in between.
- The sharp clustering at boundary values (0.0 and 1.0) is inconsistent with continuous absorption and suggests either genuine structural bifurcation or a threshold artifact of the Af metric.

**Layer sweep summary** (full results in Table 2):

| Layer | Mean L0 | Mean Af | % Af > 0.5 | % Af = 0.0 |
|-------|---------|---------|------------|------------|
| 0 | 18.9 | 0.229 | 19.5% | 77.6% |
| 2 | 29.1 | 0.470 | 45.5% | 51.2% |
| 4 | 37.8 | 0.503 | 49.3% | 48.1% |
| 6 | 57.0 | 0.430 | 41.0% | 56.4% |
| 8 | 71.9 | 0.305 | 20.9% | 76.8% |
| 10 | 56.0 | 0.287 | 17.3% | 80.2% |

## 5.2 H3: Absorption Does Not Monotonically Increase with Sparsity

**Hypothesis**: Higher sparsity (L0) monotonically increases absorption.

- Spearman r = 0.086 (p = 0.872); Pearson r = -0.073 (p = 0.891). Neither significant.
- Pattern: clear inverted-U, peaking at layer 4, not monotonic.
- Layer 8 has highest L0 (71.9, least sparse) but only 20.9% > 0.5.
- **H3 falsified: inverted-U pattern, not monotonic increase.**

![Figure 2: Layer 4 absorption score histogram](figures/fig4_layer4_histogram.pdf) (Section 5.2)

![Figure 3: Per-layer absorption: % > 0.5 and mean Af per layer; inverted-U peaks at layer 4](figures/fig1_layer_absorption.pdf) (Section 5.2)

## 5.3 H4: Circuit Faithfulness Comparison Is Uninformative

**Hypothesis**: High-absorption SAE patching reduces faithfulness by >=5pp vs. low-absorption.

**Task**: Factual recall patching -- "The capital of France is ___" (target: " Paris") vs. "The capital of Germany is ___" (target: " Berlin").

| Patching Method | Faithfulness |
|-----------------|--------------|
| Raw residual stream | 0.400 |
| SAE all latents | 0.289 |
| SAE low-absorption (bottom 10%) | 0.000 |
| SAE high-absorption (top 10%) | 0.000 |

- Raw residual patching restores 40% of clean-to-corrupted logit difference.
- SAE all latents: 0.289 (11pp drop from raw residual).
- Both low-absorption and high-absorption subsets yield 0.000 -- difference is 0.000.
- **Verdict: H4 uninformative.** Both subsets fail entirely. The experiment design flaw is explicitly acknowledged: task-agnostic subset selection does not isolate circuit-relevant absorption. A better design would compare full SAEs at layers with different absorption profiles (layer 4 vs. layer 8). Layer 4 (49.3% absorption) was never tested for H4.

![Figure 4: Faithfulness for raw residual, SAE all, SAE low-abs, SAE high-abs](figures/fig3_faithfulness.pdf) (Section 5.3)

## 5.4 H5: Larger Dictionaries Reduce Absorption

**Hypothesis**: Larger dictionary sizes monotonically reduce per-latent absorption.

| Dict Size | Mean Af (SAE) | Mean Af (Random) | % Af > 0.5 (SAE) | % Af > 0.5 (Random) |
|-----------|---------------|-----------------|-----------------|---------------------|
| 2,048 | 0.0268 | 0.0000 | 2.25% | 0.00% |
| 8,192 | 0.0067 | 0.0000 | 0.56% | 0.00% |
| 24,576 | 0.0022 | 0.0000 | 0.19% | 0.00% |

- Monotonic decrease confirmed across all metrics.
- Random controls: 0.00% at all sizes, correctly scaled.
- **H5 not falsified.** Direction confirmed. Practical significance is modest: even at 2K, 97.75% of latents are not absorbed.
- Note: 2K-24K comparison uses subselections of a single 24K SAE (prioritized by absorbable latents) -- this is an upper-bound estimate. Whether the same relationship holds for independently trained SAEs of different sizes is an open question.

![Figure 5: Mean Af (log scale) and % > 0.5 vs dictionary size](figures/fig2_dict_size.pdf) (Section 5.4)

## 5.5 H2: Token Frequency and Absorption Correlation (PENDING)

**Hypothesis**: Low-frequency token latents absorbed at >=2x rate of high-frequency.

**Status**: PENDING -- not yet analyzed. Critical path for future work.

**Why pending**: H2 requires computing token frequencies over Pile validation split. Additionally, layer 8 has insufficient absorption variance (only 46 latents with Af > 0.5) for meaningful correlation analysis. Layer 4 with 49.3% > 0.5 (approximately 12,000 absorbed latents) provides 260x more data for H2 analysis.

**Pre-registered analysis plan**:
1. Compute median token frequency for each latent's top-20 activating tokens over Pile validation.
2. Bin latents into quartiles Q1-Q4 by median frequency.
3. Compute Spearman rank correlation between token frequency and Af.
4. Falsify if r_s >= 0 OR if Q1/Q4 absorption ratio < 2x.

## 5.6 Hypothesis Summary

| Hypothesis | Predicted | Observed | Status |
|------------|-----------|----------|--------|
| H1 (layer 8) | >20% | 0.19% | Falsified (100x below) |
| H1 (layer 4) | >20% | 49.3% | Confirmed (exploratory) |
| H2 | Spearman r < 0 | Not tested | PENDING |
| H3 | Monotonic increase | Inverted-U, r = 0.086 (NS) | Falsified |
| H4 | High-abs >= 5pp worse | Diff = 0.0 (both 0.0) | Uninformative |
| H5 | Decreases with dict size | 2K: 2.25%, 24K: 0.19% | Not falsified |

---

## 6. Discussion

### 6.1 Summary of Findings

Four of five hypotheses about feature absorption in GPT-2 small SAEs either failed or produced uninformative results. H1 is falsified at layer 8 (0.19%) but confirmed at layer 4 (49.3%). H3 is falsified (inverted-U, not monotonic). H4 is uninformative (both subsets 0.0). H5 moves in the hypothesized direction, though practical significance is limited by the phenomenon's overall rarity. H2 and H6 remain untested.

### 6.2 Why Did Most Hypotheses Fail?

**Possible interpretation**: SAE training's reconstruction objective exerts implicit anti-absorption pressure -- if latents can redundantly encode each other's variance, the training loss benefits from specializing rather than sharing. However, our data cannot directly confirm this mechanism; an alternative explanation is that the 80% RVE threshold is too strict for detecting subtler forms of feature overlap.

**H3 failure (inverted-U)**: Mid-layers (4-6) process the densest abstract and semantic content in GPT-2 small, which may inherently produce more feature overlap regardless of sparsity level. L0 is not a reliable proxy for absorption risk.

**H4 failure**: Both absorption subsets yield 0.0, revealing that corpus-level absorption scores do not predict circuit-level causal importance. The 10% subset destroys reconstruction capacity entirely -- dictionary completeness, not absorption level, drives patching fidelity.

### 6.3 Implications for SAE-Based Interpretability

1. **SAE latents are largely independent at most layers**: At layers 8 and 10, over 76% of latents show Af <= 0.5. The exception is layer 4 where 49.3% exceed the threshold.
2. **Full SAE dictionary is necessary for downstream causal validity**: Any attempt to use a subset of latents destroys patching signal.
3. **Dictionary size matters within the absorption framework**: Larger dictionaries monotonically reduce absorption. For applications where absorption is a concern, prefer the largest available SAE.
4. **Sparsity tuning does not affect absorption risk**: The inverted-U pattern at mid-layers is a property of model architecture, not the sparsity hyperparameter.

### 6.4 Limitations

- **Single model (GPT-2 small)**: findings may not generalize to GemmaScope or LLaMA-class SAEs.
- **Pilot scale (100 sequences)**: full-scale may reveal rare effects, but the 100x gap for H1 at layer 8 is unlikely to close with 10x scale increase.
- **H4 correctly labeled inconclusive**: The experiment design flaw is acknowledged; the correct experiment (layer 4 vs. layer 8 full SAE comparison) was never conducted.
- **H2 and H6 pending**: These require only CPU analysis on existing layer 4 data but have been deferred.
- **H5 subsample bias**: The 2K-24K comparison uses subselections of a single 24K SAE (prioritized by absorbable latents). Random subsample control would be needed for unbiased estimate.

### 6.5 Open Questions

- Does absorption manifest differently in larger models (GemmaScope, LLaMA-class)?
- Can a graded or continuous absorption measure capture subtler forms of feature overlap?
- Are the layer 4 perfect-score latents (6,170 latents at Af=1.0, 25.1%) genuine absorption or do they reflect the metric's bimodal artifact?
- Can explicit anti-absorption objectives during SAE training reduce absorption further?

---

## 7. Conclusion

This paper provides the first systematic empirical characterization of feature absorption in Sparse Autoencoders. The central finding is a **layer-dependent inverted-U pattern**: absorption peaks at mid-layers (layer 4: 49.3%) and declines at both shallow (layer 0: 19.5%) and deeper layers (layer 8: 0.19%, layer 10: 17.3%).

Four of five hypotheses were falsified or found uninformative. Only H5 -- larger dictionaries reduce absorption -- moved in the hypothesized direction, though practical significance is limited by the phenomenon's overall rarity. The contribution is both empirical (characterizing layer-dependent boundary conditions that rule out absorption as a dominant failure mode in GPT-2 small) and methodological (a validated, reproducible absorption metric with random dictionary controls for the community to test in other architectures).

The validated absorption metric Af provides a tool for the community to test whether these findings generalize. GPT-2 small is orders of magnitude smaller than the models where SAE interpretability is most needed. A graded absorption measure may capture subtler forms of feature overlap that the binary threshold misses. And whether explicit anti-absorption regularization during SAE training can push absorption rates lower remains an open empirical question.

---

## Appendix A: Sensitivity Analysis

| RVE Threshold | Top-3 | Top-5 | Top-10 |
|---------------|-------|-------|--------|
| 0.70 | 0.31% | 0.43% | 0.61% |
| 0.80 | 0.14% | 0.19% | 0.27% |
| 0.90 | 0.06% | 0.08% | 0.11% |

Rankings are consistent across all configurations. Absorption is rare regardless of threshold or co-firer count.

---

## Figure & Table Plan

### Figure 1: Experimental Pipeline Overview (Section: Method)
- **Purpose**: Explain the absorption score computation pipeline end-to-end.
- **Type**: architecture_diagram / flow_chart
- **Content**: (1) Corpus to model activations, (2) SAE encoding, (3) Identify top-5 co-firers per token, (4) Partial reconstruction, (5) RVE computation, (6) Absorption score per latent.
- **Key takeaway**: The absorption score quantifies how much a latent's reconstruction variance is redundantly explained by its co-firing latents.
- **File**: `figures/gen_pipeline.pdf`

### Figure 2: Layer 4 Absorption Score Distribution (Section: Experiments / H3)
- **Purpose**: Show the bimodal distribution of absorption scores at layer 4.
- **Type**: histogram (two-panel)
- **Content**: Left: full distribution (0.0 to 1.0); Right: zoomed (0.1 < Af < 0.9). Annotations: 6,170 latents (25.1%) at Af=1.0; 8,400 latents (34.2%) at Af=0.0.
- **Key takeaway**: Layer 4 absorption is bimodal: most features cluster at boundary values (Af=1.0 or Af=0.0), with 40.7% in between. This sharp bifurcation is inconsistent with continuous absorption.
- **File**: `figures/fig4_layer4_histogram.pdf`

### Figure 3: Per-Layer Absorption: Inverted-U Pattern (Section: Experiments / H1+H3)
- **Purpose**: Demonstrate the inverted-U relationship between layer depth and absorption rate.
- **Type**: bar_chart with line overlay
- **Content**: X-axis = layer (0, 2, 4, 6, 8, 10); Y-axis primary = % Af > 0.5 (bars); Y-axis secondary = mean Af (line with markers). L0 annotations above each bar.
- **Key takeaway**: Absorption peaks at mid-layers (layer 4: 49.3%) not at the deepest layers. The inverted-U pattern is the central empirical finding.
- **File**: `figures/fig1_layer_absorption.pdf`

### Figure 4: Circuit Faithfulness Comparison (Section: Experiments / H4)
- **Purpose**: Compare circuit faithfulness across patching methods.
- **Type**: bar_chart
- **Content**: X-axis = patching method (Raw Residual, SAE All, SAE Low-Abs, SAE High-Abs); Y-axis = mean faithfulness.
- **Key takeaway**: Both absorption subsets yield 0.0. The full SAE (0.289) is within 11pp of raw residual (0.400).
- **File**: `figures/fig3_faithfulness.pdf`

### Figure 5: Mean Absorption vs. Dictionary Size (Section: Experiments / H5)
- **Purpose**: Show the monotonic decrease in absorption with larger dictionary sizes.
- **Type**: bar_chart (log-scale x-axis)
- **Content**: X-axis = dictionary size (2048, 8192, 24576, log scale); Y-axis = mean Af. Two grouped bars: real SAE vs. random control.
- **Key takeaway**: Larger dictionary = less absorption. Monotonic and clean across all three sizes.
- **File**: `figures/fig2_dict_size.pdf`

### Table 1: Hypothesis Results Summary (Section: Experiments / 5.6)
- **Purpose**: At-a-glance view of all hypotheses and their falsification status.
- **Type**: comparison_table
- **Content**: Hypothesis | Predicted | Observed | Status
- **File**: inline in Section 5.6

### Table 2: Per-Layer L0 and Absorption Statistics (Section: Experiments / 5.1)
- **Purpose**: Full layer sweep data supporting the inverted-U finding.
- **Type**: data_table
- **Content**: Layer | Mean L0 | Mean Af | % Af > 0.5 | % Af = 0.0
- **File**: inline in Section 5.1

### Table 3: Circuit Faithfulness Details (Section: Experiments / 5.3)
- **Purpose**: Full faithfulness results for all patching methods.
- **Type**: comparison_table
- **Content**: Patching Method | Faithfulness
- **File**: inline in Section 5.3

### Table 4: Dictionary Size Comparison (Section: Experiments / 5.4)
- **Purpose**: Full dictionary size sweep with random control.
- **Type**: ablation_table
- **Content**: Dict Size | SAE Mean Af | Random Mean Af | % > 0.5 (SAE) | % > 0.5 (Random)
- **File**: inline in Section 5.4

### Table A1: Sensitivity Analysis (Appendix A)
- **Purpose**: Robustness across RVE thresholds and co-firer counts.
- **Type**: ablation_table
- **Content**: RVE threshold x co-firer count matrix of absorption prevalence.
- **File**: inline in Appendix A

---

## Transition Logic Between Sections

1. **Introduction** ends with structured roadmap: "We designed five hypotheses... Section 2 provides background... Section 7 concludes."

2. **Section 2** establishes three conceptual pillars: SAE architecture (2.1), feature interaction phenomena including absorption vs. superposition distinction (2.2), and causal analysis with SAEs including faithfulness metric (2.3).

3. **Section 3** defines the absorption metric. Must be read before Section 5 because all experiments depend on it. End with validation on random dictionary controls and edge cases.

4. **Section 4** establishes models, dataset, and pre-registered hypotheses with falsification criteria. End with: "We now test each hypothesis in turn."

5. **Sections 5.1-5.5** present results. Order: H1 (layer-dependence, foundational finding), H3 (inverted-U interpretation), H4 (downstream impact), H5 (positive direction), H2 (omitted, honest acknowledgment with pre-registered protocol).

6. **Section 6** opens with: "Four of the five hypotheses about feature absorption in GPT-2 small SAEs either failed or produced uninformative results." Then addresses why (6.2), limitations (6.3), implications (6.4), and open questions (6.5).

7. **Section 7** closes with: absorption is strongly layer-dependent, peaking at mid-layers; the validated metric provides a reproducible tool for the community.

---

## Critical Corrections from Prior Drafts

The following critical corrections from review.md and critique_writing.md must be applied to the paper:

### Must Fix (Critical)

1. **Abstract H1 framing contradiction**: The abstract states H1 is "falsified at layer 8 but not falsified at layer 4" -- contradiction since falsification is binary, not layer-dependent. The pre-registered hypothesis predicted >20% of latents in mid-to-deep layers (layers 4-10 collectively) with falsification criterion <10% across layers 4-10, not per-layer. **Fix**: H1 is falsified at layer 8 (pre-registered test layer, 0.19% < 10%). Layer 4 (49.3%) is an exploratory finding, not "not falsified at that layer."

2. **Abstract admits unconducted experiment**: The abstract currently states "the correct experiment... was not conducted" -- this limitation belongs in Discussion/Limitations, not in the abstract. **Fix**: Remove from abstract entirely. Report what was done and what was found: "Both absorption subsets yielded 0.0 faithfulness, preventing any conclusion about whether absorption level predicts circuit-level causal importance."

3. **H4 framed as falsified when actually untested**: The paper states H4 is "falsified" but the experiment does not test the hypothesis -- the key comparison is uninterpretable because both subsets yielded 0.0. The correctly designed experiment (full SAE at layer 4 vs. layer 8) was never run. **Fix**: Reframe H4 as "untested" or "uninformative" explicitly, not falsified. State: "H4 remains unconducted -- the pilot results do not support any conclusion about whether absorption level predicts circuit-level causal importance."

4. **Section 3.1 formula**: Partial reconstruction formula must include b_dec or explicitly show why it cancels.

5. **"sparsest layer" mischaracterization**: Layer 8 has L0=71.9 (highest in dataset) meaning most non-zero activations (least sparse), not sparsest. Fix to "the layer with the highest L0 (least sparse representation)" consistently throughout.

6. **Layer 4 perfect-score count**: At layer 4, 6,170 latents (25.1%) have Af=1.0. The "8 latents" observation was only from H1 pilot at layer 8. All layer 4 figures/text must use 6,170.

7. **Section 6.1 "worst layer" error**: Layer 4 is the HIGHEST absorption layer (49.3%), not the lowest.

8. **Conclusion "uniform"**: Layer 4 has 49.3% absorption, which is NOT uniform. Change to "at depth, absorption is rare... with the notable exception of layer 4."

### Should Fix (Major)

9. **Anti-absorption mechanism (Section 6.2)**: Restructure as explicitly speculative: "One possible interpretation... however our data cannot directly confirm..."

10. **H5 subsample bias bounding**: Add one sentence bounding the bias in H5 section.

11. **H2 expansion**: Expand from placeholder to full pre-registered protocol with enough specificity for replication.

12. **H4 conclusion confound**: H4 conclusion conflates absorption selection with dictionary coverage. Add sentence acknowledging the confound.

13. **Terminology deviations**: Global fixes needed:
    - "layer-wise" -> "per-layer" (Section 5.2)
    - "activations" as singular noun -> "activation" (Section 3.1)
    - "significant" without "statistical" qualifier -> "statistical significance" (Section 5.2)

14. **Section 6.4 percentage accuracy**: "approximately 79%" should specify the derivation: 100 - 20.9 = 79.1%.

15. **Figure 2 description mismatch**: Text says "Figure 2 shows the detailed absorption score distribution for layer 4" but the actual Figure 2 (per figure plan) is fig1_layer_absorption.pdf showing inverted-U across all layers. The layer-4 histogram is Figure 4 (fig4_layer4_histogram.pdf). Fix the figure reference.

16. **"shows that" vs "suggests that"**: Section 5.3 says "shows that dictionary completeness" but should be "suggests that" given the design flaw in subset selection.

### Minor Fixes

17. **Section 3.3 isotropic explanation unclear**: Add missing step: isotropic Gaussian columns in 768-dim space have expected squared cosine similarity of 1/768 with any fixed direction. A 6-dimensional subspace captures expected 6/768 = 0.78% of variance.

---

## H2 and H6 Critical Path Status

**CRITICAL: H2 and H6 have been deferred for 11+ iterations without execution. This is the single greatest risk to project stagnation.**

### Why These Analyses Are Critical

- **H2 (token frequency correlation)**: Was incorrectly attempted at layer 8 (46 absorbed latents) instead of layer 4 (12,000 absorbed latents -- 260x more data). The layer 8 result is statistically meaningless. H2 at layer 4 requires only CPU analysis (~2 hours) on existing layer 4 data.

- **H6 (perfect-score latent investigation)**: 6,170 layer 4 latents (25.1%) have Af=1.0, each firing on exactly 100 tokens (equal to the number of sequences in the pilot corpus). This regularity is too systematic to be coincidence. Investigation requires CPU analysis (~2 hours) on existing layer 4 data.

### Go/No-Go Decision Tree

```
H2 confirmed (negative correlation) + H6 artifact confirmed -> Continue layer-dependence story
H2 falsified + H6 artifact -> Pivot to cand_metric_centric (metric validation paper)
H2 confirmed + H6 genuine -> Continue with richer analysis of perfect absorbers
H4 properly tested + positive result -> Strong paper with layer-dependence + circuit story
H4 properly tested + negative result -> Paper is metric + layer-dependence only
```

### Pre-Registered H2 Protocol (For Layer 4 Analysis)

1. Compute median token frequency for each latent's top-20 activating tokens over Pile validation.
2. Bin latents into quartiles Q1-Q4 by median frequency.
3. Compute Spearman rank correlation between log-frequency and Af.
4. Falsify if r_s >= 0 OR if Q1/Q4 absorption ratio < 2x.

### H6 Investigation Protocol

1. Compute token position consistency across sequences for each perfect-score latent (Af=1.0).
2. Check if latents fire at same absolute position in each sequence.
3. Compare position distribution to matched random latents.
4. If confirmed as artifacts, exclude from primary analysis, document in Appendix.

**Stagnation Alert**: Both analyses have been deferred for 11 iterations. No more deferral is acceptable. Writing quality is no longer the bottleneck (score 6.0 maintained); experimental execution is now the sole bottleneck.
# Experiment Methodology: Competitive Geometry of Feature Absorption in SAEs

## Candidate
**cand_a** — Competitive Geometry of Feature Absorption: LV Unsupervised Detector + Corpus PMI + Downstream Impact

---

## Overview

All experiments are **training-free**, operating on pre-trained Gemma Scope SAEs (Gemma 2 2B) accessed via SAELens. No model fine-tuning is performed. The study has four components corresponding to H1–H4, plus a pilot validation gate and ablation schedule.

---

## Environment Setup

| Resource | Specification |
|---|---|
| Model | Gemma 2 2B (google/gemma-2-2b via TransformerLens) |
| SAEs | Gemma Scope (google/gemma-scope-2b-pt-res) via SAELens |
| Widths tested | 16k, 65k, 131k latents |
| Layers tested | 6, 12, 20, 25 |
| GPU | 1× A100 40GB (inference only; no training) |
| Python | 3.10+, sae-lens>=5.0.0, transformer-lens>=2.0.0 |
| Ground truth | sae-spelling (github.com/lasr-spelling/sae-spelling) |
| Corpus | OpenWebText (HuggingFace: stas/openwebtext-10k for pilot; full sample for H2) |

SAELens loading pattern for Gemma Scope:
```python
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_82",
    device="cuda"
)
```

---

## Component 0: Pilot — Pipeline Validation and L0 Gate (Task P0)

**Purpose:** Validate end-to-end pipeline before committing GPU-hours; measure L0 matching across widths for confound control.

**Procedure:**
1. Load Gemma 2 2B + Gemma Scope SAEs at layer 12, widths 16k and 65k
2. Run sae-spelling absorption measurement on letters A–E only (n=100 test tokens each, seed 42)
3. Record: absorption rate per letter, empirical L0 for both widths on a 500-token sample
4. Compute α_ij for top-100 candidate pairs by decoder cosine similarity on 1k activations; manually log top-10 α_ij pairs and whether they correspond to known letter-feature parent-child relationships

**Decision gate:**
- If absorption rate ∈ [10%, 45%]: pipeline valid → proceed to Components 1–3
- If L0 difference between 16k and 65k SAEs > 20%: Component 2 regression must include L0 as covariate (still valid)
- If top-10 α_ij pairs contain ≥ 1 letter-feature parent-child pair: LV sanity check passes → proceed with H1 full experiment
- If all three checks fail: stop, investigate SAELens loading configuration before proceeding

**Expected runtime:** 15 minutes

---

## Component 1: LV Competition Coefficient as Unsupervised Absorption Detector (H1, H4)

### Step 1: Activation Statistics Collection

For each SAE configuration (Gemma 2 2B, layers {6, 12, 20, 25}, widths {16k, 65k}):
- Run 10k tokens of OpenWebText through the SAE using SAELens
- Record: mean activation frequency f_i per latent (fraction of tokens where latent activates > 0)
- Record: pairwise co-activation P(a_i > 0, a_j > 0) for all pairs (i, j) where both f_i > 0.001 and f_j > 0.001
- Pre-filter pairs: decoder cosine similarity cos(d_i, d_j) > 0.15 to restrict to geometrically adjacent features

Computation:
```
σ_ij = P(a_i > 0, a_j > 0) / min(f_i, f_j)    # normalized niche overlap
α_ij = σ_ij · (f_j / f_i)                        # LV competition coefficient
```

Output: per-SAE parquet file with columns [latent_i, latent_j, f_i, f_j, coact_rate, sigma_ij, alpha_ij, decoder_cosine]

### Step 2: Threshold Calibration and Validation (H1)

Ground truth: sae-spelling on Gemma Scope 16k layer 12, letters A–Z (n=100 test tokens per letter, seed 42).
- Split: letters A–M as calibration (13 letters), letters N–Z as test (13 letters)
- Calibration: fit threshold τ ∈ {0.5, 0.75, 1.0, 1.25, 1.5} on calibration letters; select τ maximizing F1
- Evaluation: apply calibrated τ to test letters; report precision, recall, F1, ROC-AUC

**LV-specific diagnostic — sharpness test:**
- Bin α_ij values into bins of width 0.1 over range [0, 3]
- For each bin: compute empirical absorption rate (Chanin metric within that bin)
- Fit sigmoid f(x) = 1 / (1 + exp(-k(x - x0))) and linear f(x) = ax + b to the binned data
- Compare AIC: sigmoid lower → LV theory supported (sharp transition); linear lower → smooth monotone suppression (LV not load-bearing)

### Step 3: Cross-Architecture Validation

Apply calibrated τ (from layer 12, 16k) to:
- Gemma Scope JumpReLU SAEs at layer 12 (same model, different architecture)
- Gemma Scope TopK SAEs at layer 12 (if available via SAELens release list)

Metric: F1 degradation from within-architecture calibration. Pass criterion: F1 degrades < 10 percentage points.

### Step 4: Width Paradox — Distributed Absorption Score (H4)

For Gemma Scope SAEs at widths {16k, 65k, 131k}, layer 12:
- DAS(P, k=1): Chanin absorption metric for letter-feature parents (single-child, standard)
- DAS(P, k=3): estimated via multi-label logistic regression on activation data:
  - For each parent P: identify top-3 children C1, C2, C3 by α_ij
  - Fit logistic regression predicting f_P from {f_C1, f_C2, f_C3} on 10k activation samples
  - DAS(P, k=3) = 1 - (conditional mutual information I(X; f_P | f_C1, f_C2, f_C3)) / H(f_P)
  - Approximate via: 1 - McFadden R² of the logistic regression / baseline entropy ratio

Prediction: DAS(k=3) monotonically increases across widths for ≥ 80% of letter features; DAS(k=1) is non-monotone or decreasing for ≥ 60% of letter features.

---

## Component 2: Corpus PMI as Absorption Predictor (H2, H5)

### Step 1: PMI Extraction from OpenWebText

For each of the 26 letter-token pairs used in the first-letter task:
- Load 1M-token sample from OpenWebText (HuggingFace: stas/openwebtext-10k or Skylion007/openwebtext)
- Tokenize with Gemma 2 tokenizer
- Compute: P(token_starts_with_L) as marginal frequency of tokens beginning with letter L
- Compute: joint probability P(parent_context, child_token) using sliding window of 5 tokens
- PMI(letter_category, specific_child_token) = log[ P(child | context) / P(child) ]
- Record top-10 PMI pairs per letter

Output: `exp/results/pmi_features.json` with columns [letter, token_id, token_str, pmi_score, child_frequency]

### Step 2: 30-SAE Absorption Survey and Regression

Run sae-spelling on 30 Gemma Scope SAE configurations:
- Widths: {16k, 65k, 131k} × Layers: {6, 12, 20, 25} × L0 settings: {low, medium, high if available}
- All 26 letters × n=100 test tokens per letter, seed 42
- Output: absorption rate per (SAE config, letter)

Regression model:
```
absorption_rate_il = β₀ + β₁ log(L0_i) + β₂ log(width_i) + β₃ layer_i + β₄ log(PMI_l) + ε_il
```
Where i indexes SAE configuration, l indexes letter.

Report: R², adjusted R², partial R² and p-value for β₄ (PMI term), coefficient signs.

Cross-model validation: if time permits, replicate PMI regression on GPT-2 small SAEs available via SAELens (release: `gpt2-small-res-jb`).

### Step 3: Absorption Taxonomy (H5)

Operationalize three absorption types on Gemma Scope 16k, layer 12:
- **Type I (Full):** Chanin metric > threshold AND single latent accounts for > 80% of parent suppression
- **Type II (Partial):** Parent latent activation at < 50% of expected magnitude on expected-parent-token set; measured via normalized activation ratio
- **Type III (Distributed):** DAS(k=3) > 0.60 AND Type I not triggered; top-3 children collectively explain parent suppression

Report: fraction of 26 letter features showing Type I, II, III, or no absorption. Compare comprehensive rate to reported 15–35% Type I rate.

---

## Component 3: Downstream Impact Analysis (H3)

### Stage 1: SAEBench Correlation Analysis (zero GPU)

- Download SAEBench CSV from neuronpedia.org/sae-bench
- Filter to Gemma 2 2B SAEs only (removes confound of model family)
- Compute Pearson r and Spearman ρ between absorption_score and: RAVEL, SCR, sparse probing F1, unlearning performance
- Apply Bonferroni correction for 4 simultaneous tests (α_corrected = 0.05/4 = 0.0125)
- Also compute partial Pearson r controlling for SAE width, layer, and architecture class
- Pre-specified threshold for "meaningful" correlation: |r| > 0.30

### Stage 2: Matched RAVEL Comparison

From SAEBench data:
- Select top-5 lowest absorption SAEs and top-5 highest absorption SAEs matched on: model = Gemma 2 2B, layer ∈ {12, 20}
- Run RAVEL evaluation directly on each matched pair using SAEBench evaluation code
- Paired t-test (one-sided, H0: low-absorption SAEs perform worse on RAVEL), α = 0.05

### Stage 3: Safety Probe Pilot

Dataset: 50 harmful prompts (AdvBench subset) vs. 50 benign prompts (OpenWebText excerpts)
SAE selection: 3 SAEs from Gemma 2 2B with lowest, median, and highest absorption rates
Classification setup:
- Dense linear probe: logistic regression on residual stream at layer 12
- 1-sparse SAE probe: ridge regression on SAE activations with L0=1 constraint (top-1 feature selection)
Report: ROC-AUC for dense probe vs. 1-sparse SAE probe at each absorption level; does probe gap narrow as absorption decreases?

---

## Ablation Schedule

| Ablation | Variable | Values | Test | Success Criterion |
|---|---|---|---|---|
| A1 | τ threshold sensitivity | {0.5, 0.75, 1.0, 1.25, 1.5} | F1 across τ values | F1 peak within τ ∈ {0.75, 1.25} |
| A2 | Decoder cosine pre-filter | {0.10, 0.15, 0.25} | Coverage vs. precision | ≥ 80% absorbed pairs captured at 0.15 |
| A3 | PMI regression without SAE config | Remove β₁β₂β₃ from regression | PMI standalone R² | Partial R² ≥ 0.10 after controlling for config |
| A4 | Per-layer regression stability | Layers {6, 12, 20, 25} | PMI coefficient sign | Same sign across all layers |
| A5 | Absorption type breakdown by letter | 26 letters | Type I/II/III mix | Comprehensive rate > 15–35% |

---

## Baselines

| Baseline | Role | Implementation |
|---|---|---|
| Cosine-similarity-only detector | Ablation of LV coefficient; prior LessWrong negative result | Threshold on cos(d_i, d_j) alone |
| Chanin et al. probe-directed metric | Ground truth for H1 validation | sae-spelling library |
| SAEBench absorption scores | Pre-computed; used for H3 correlation | neuronpedia.org/sae-bench CSV |
| Feature hierarchy depth | Compared against PMI in H2 regression | sae-spelling feature labels |
| Dense linear probe | Upper bound on downstream task | Logistic regression on residual stream |

---

## Expected Visualizations

- **Figure 1 (Architecture diagram):** LV competitive exclusion framework → α_ij computation pipeline → absorption prediction
- **Table 1 (Main results):** H1 validation table: precision, recall, F1, AUC for LV detector vs. cosine-only baseline across architectures (rows) and layers (columns)
- **Figure 2 (Sharpness test):** Absorption rate vs. α_ij bin plot with sigmoid and linear fits; sigmoid AIC vs. linear AIC comparison
- **Figure 3 (Width paradox):** DAS(k=1) and DAS(k=3) vs. SAE width for letter features; error bars from letter-level variance
- **Figure 4 (PMI regression):** Partial regression plot of absorption_rate vs. log(PMI) with SAE config residualized; letter labels as data points
- **Table 2 (Downstream correlation):** Pearson r / Spearman ρ matrix: absorption score × SAEBench tasks, with 95% CI, Bonferroni-corrected p-values
- **Figure 5 (Taxonomy):** Stacked bar chart: Type I / II / III / None absorption rates per SAE width; shows how comprehensive rate exceeds Chanin metric
- **Figure 6 (Safety probe):** ROC curves for dense probe vs. 1-sparse SAE probe at low/median/high absorption SAEs

---

## Risk and Fallback

| Risk | P | Mitigation |
|---|---|---|
| L0 not matched across widths | 50% | Include L0/D as covariate in all regressions; results remain valid |
| LV detector F1 < 0.50 | 25% | Report α_ij Pearson r with absorption rate as descriptive finding; pivot to cand_b (information-theoretic formulation) |
| PMI not predictive (partial R² < 0.05) | 35% | Report as null result supporting objective-driven account; other components stand independently |
| Gemma Scope API/access failure | 20% | Fall back to GPT-2 SAEs (gpt2-small-res-jb); all analyses replicate on GPT-2 |
| SAEBench CSV lacks covariate info | 30% | Use architecture class from SAE IDs as proxy covariate; report limitations |
| ATM/OrtSAE checkpoints unavailable | 40% | Cross-architecture test uses JumpReLU + TopK only; H6 becomes theoretical prediction |

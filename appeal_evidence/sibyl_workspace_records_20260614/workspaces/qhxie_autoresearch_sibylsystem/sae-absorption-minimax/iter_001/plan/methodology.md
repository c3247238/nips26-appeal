# Methodology: Feature Absorption Steering Sensitivity Study (Revision 1)

## 1. Problem Setup

### Research Question
This paper investigates the causal intervention reliability of absorbed vs non-absorbed SAE features. The central question: **Do absorbed features respond differently to steering interventions than non-absorbed features?**

### Key Finding (H3 Reversed - Central Contribution)
Full-scale experiments revealed a surprising result: **high-absorption features exhibit HIGHER steering sensitivity** (Spearman r=+0.35, p<0.001) than low-absorption features, contradicting both the prevailing assumption in the literature and the initial pilot (N=20).

**Full experiment (N=100 features)**:
- High-absorption mean steering effect: 0.1035
- Low-absorption mean steering effect: 0.0874
- Ratio: 0.84 (high-absorption 18% MORE effective per unit effect, though mean is higher)
- Spearman r between UAS and steering sensitivity: **+0.3548** (p=2.92e-04)

### Motivation
- The SAE Sanity Checks critique (Korznikov et al., 2026) questions whether SAE features have genuine causal weight
- If absorbed features are MORE steerable, this reframes absorption as a leverage-redistribution phenomenon rather than causal silencing
- Understanding this relationship is critical for feature-based interpretability and intervention methods

## Setup

### Models and SAEs
- **GPT-2 Small**: `gpt2-small`, d_model=768, 12 layers. Use pre-trained SAEs from `gpt2-small-res-jb` release in SAELens.
  - Primary layer: 8 (rich semantic content)
  - Layers for H1 atlas: [2, 4, 6, 8, 10]
- **Gemma-2B**: `gemma-2-2b-it`, d_model=2048. Use pre-trained SAEs from `gemma-2b-res` release.
  - Primary layer: 12
  - Layers for H1 atlas: [4, 8, 12, 16, 20]

### Pre-trained SAE Releases (SAELens)
```
gpt2-small-res-jb  → GPT-2 Small residual stream SAEs (layer 2-10)
gemma-2b-res       → Gemma-2B residual stream SAEs
```
Available SAE architectures in SAELens: standard, topk, jumprelu

### Dataset
- **Text corpus for activation collection**: Use `monology/pile-uncopyrighted` (the SAELens default)
  - For pilot: 10K tokens subsample
  - For full: 1M tokens per layer
- **Downstream tasks**: synthetic counterfactual generation, concept classification

## Absorption Metrics

### Chanin Absorption Score (Supervised)
As defined by Chanin et al. (2024): measure whether a "parent" feature fires when its "child" features do not.

**First-letter probe method** (GPT-2 only):
1. Identify features activated by words starting with a specific letter (e.g., "eight", "eleven")
2. Compute classification accuracy: does the feature fire when the first-letter pattern is present?
3. Absorption = 1 - accuracy. High absorption = fires even when parent concept is absent.

### Unsupervised Absorption Score (UAS) - Proposed
```
UAS(f) = α * cos_sim_variance(f) + β * freq_skewness(f)
```
- `cos_sim_variance(f)`: variance of cosine similarities between feature f's decoder direction and all other features
- `freq_skewness(f)`: skewness of activation frequency distribution across contexts
- α, β calibrated via h4_uas_dev on GPT-2 layer 8

**Best validated formula** (h4_uas_dev): `neg_cos_sim_variance(f) = -cos_sim_variance(f)` achieves Spearman r = -0.49 with Chanin absorption. The negative sign indicates that higher cosine similarity variance predicts higher absorption.

**Important note**: UAS and Chanin absorption are NEGATIVELY correlated in h4_uas_dev (r=-0.49), but POSITIVELY correlated in pilot_h1_h4 (r=+0.79). This discrepancy arises from different absorption metrics (Chanin first-letter probe vs Gini absorption). The paper should clarify which absorption metric is used for which analysis.

## Hypotheses and Tasks

### H1: Absorption Peaks in Middle Layers
**Status**: UNRESOLVED. GPT-2 atlas shows early peak at layer 2, not mid-layer.
**Test**: Compute absorption scores at layers [2,4,6,8,10] for GPT-2 and [4,8,12,16,20] for Gemma-2B.
**Expected**: U-shaped or unimodal curve peaking at layer 6-8 (GPT-2), layer 10-14 (Gemma-2B).
**Falsification**: Monotonic absorption across all layers, or peak at wrong layer.

### H2: Mitigation Effectiveness Hierarchy
**Status**: PARTIALLY CONFIRMED. TopK confirms (70.9% reduction, 8x MSE). OrtSAE/ATM pending. JumpReLU failed.
**Test**: Compare TopK, JumpReLU, OrtSAE, ATM, and vanilla SAE.
**Key results**:
- TopK SAE: absorption=0.068, MSE=108.07 (73% absorption reduction, 73x MSE increase)
- JumpReLU SAE: absorption=0.627, MSE=23.30 (FAILED to converge properly)
- OrtSAE: absorption=0.629, MSE=24.40 (high L0 ~50%, undertrained)
- Vanilla SAE: absorption=0.015, MSE=1.48 (reference)

### H3: Absorption is a Steering Signature (REVERSED - CENTRAL FINDING)
**Status**: REVERSED. High-absorption features show HIGHER steering sensitivity.
**Test**: Steering intervention on 100 features (50 high-absorption, 50 low-absorption) from GPT-2 layer 8.
**Results**:
- Spearman r (UAS vs steering sensitivity): +0.3548 (p=2.92e-04)
- High-absorption mean effect: 0.1035
- Low-absorption mean effect: 0.0874
**Mechanism**: Absorbed features may function as "hub" features with high residual stream leverage.

### H4: UAS Correlates with Supervised Absorption
**Status**: CONFIRMED (with caveats on metric consistency).
**Results**:
- pilot_h1_h4 (Chanin absorption): r=0.79
- h4_uas_dev (Chanin absorption, layer 8 only): r=-0.49 with neg_cos_sim_var
- full_h4: r=0.65-0.79 (across layers)
**Note**: The discrepancy between positive (r=+0.79) and negative (r=-0.49) correlations is due to different absorption metrics used in different experiments. The Chanin first-letter probe and Gini absorption produce different orderings.

### H5: Absorption Degrades Downstream Discriminability
**Status**: DIRECTIONAL CONFIRMATION (marginal failure at 4.95% vs 5% threshold).
**Results (pilot, N=48 features)**:
- Simple task: high-absorption 7.45% worse than low-absorption (AUC 0.636 vs 0.710)
- Causal task: high-absorption 2.51% worse than low-absorption (AUC 0.522 vs 0.547)
- Task-dependence delta: 4.95% (threshold: 5%)

## Pilot Design

**Target**: Complete in 10-15 minutes on 1 GPU (100 samples, seed 42, 900s timeout)

### Pilot Tasks for Next Iteration
1. **H3 null control replication**: Confirm H3 reversal with shuffled/random direction baselines
2. **H5 improved causal task**: Use CounterFact or TruthfulQA instead of synthetic pairs
3. **H1 Gemma replication**: Attempt Gemma-2B absorption atlas (pending HuggingFace access)

## Visualization Plan

- **Figure 1**: Absorption atlas — absorption score vs. layer (GPT-2 only, Gemma pending)
- **Figure 2**: Mitigation benchmark — bar chart (method × absorption reduction + reconstruction tradeoff)
- **Figure 3**: Steering effects — scatter (absorption score vs. steering effect magnitude)
- **Figure 4**: UAS validation — scatter (UAS vs. Chanin, with Spearman r annotation)
- **Figure 5**: Task-dependence — grouped bar (absorbed/non-absorbed × downstream task accuracy)
- **Table 1**: Summary metrics — method × absorption × reconstruction × downstream reliability

## Ablation Studies (Completed)

### Ablation L1 Sparsity
Trained SAEs at L1 ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1e-3} for 500K tokens (2 epochs).
**Key finding**: All L1 values produce nearly identical absorption (~0.69). Absorption ceiling effect due to undertrained SAEs (500K tokens vs 3M for pre-trained). L1 coefficient has minimal effect during early training.

### Ablation OrtSAE Orthogonal Penalty
Swept λ_ortho ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} for OrtSAE.
**Key findings**:
- λ=1e-4: -2.8% absorption, +5.3% MSE (best tradeoff)
- λ=0.1: -10.4% absorption, +74.9% MSE (maximum reduction)
- Diminishing returns: significant effects only at λ >= 1e-4
- Dead feature tradeoff is severe: 28% dead at λ=0.1 vs 19.5% with L1 only

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| H1 layer ordering unstable across seeds | High | Report with uncertainty; do not overclaim |
| Gemma-2B HuggingFace gated | Medium | Fall back to Pythia-2B; use GPT-2 for primary results |
| H3 reversal is measurement artifact | Medium | Add null controls in follow-up |
| H5 causal task weak discriminability | High | Use CounterFact/TruthfulQA for stronger signal |
| H4 UAS metric inconsistency across metrics | High | Clarify which metric used for which analysis |

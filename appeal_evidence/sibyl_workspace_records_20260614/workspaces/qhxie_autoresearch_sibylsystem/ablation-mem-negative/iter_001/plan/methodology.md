# Methodology: Systematic Analysis and Quantification of Feature Absorption in SAEs

## 1. Overview

This study systematically quantifies feature absorption across SAE architectures, establishes causal links to downstream interpretability tasks, and explores training-free detection and mitigation methods. All experiments are **training-free** (using pre-trained SAEs) unless explicitly noted.

## 2. Experimental Setup

### 2.1 Models and SAEs

**Primary Model**: Gemma-2-2B (via TransformerLens)
- Layers evaluated: 5, 10, 15, 20 (out of 18 layers; layer 20 is the final residual stream)
- Hook point: `blocks.{layer}.hook_resid_post`
- Fallback: GPT-2 Small (12 layers) if Gemma-2-2B access fails

**SAE Architectures** (all loaded/trained at width 16k):
| Architecture | Source | Notes |
|-------------|--------|-------|
| JumpReLU | GemmaScope (pre-trained) | Primary baseline; SOTA reconstruction |
| TopK | SAELens training | Standard TopK activation |
| BatchTopK | SAELens training | Batch-level adaptive k |
| Gated | SAELens training | Separate gating mechanism |
| Matryoshka | SAELens training | Nested dictionary structure |

**Pre-trained SAE Loading** (JumpReLU):
```python
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id=f"layer_{layer}/width_16k/canonical",
)
```

### 2.2 Dataset

- **Primary**: OpenWebText (10k samples for full experiments, 1k for pilots)
- **Concept evaluation**: Custom concept sets (see Section 3.2)
- **Seed**: 42 (fixed for reproducibility)

### 2.3 Software Stack

```
Base: Python 3.10+, PyTorch 2.0+
SAE: SAELens >= 2.0.0
Model: TransformerLens >= 2.0.0
Evaluation: SAEBench metrics (where applicable)
Analysis: NumPy, SciPy, Pandas, Matplotlib, Seaborn
```

## 3. Core Experiments

### 3.1 E1: Cross-Architecture Absorption Benchmark (CAAB)

**Objective**: Systematically compare absorption rates across 5 SAE architectures.

**Method**:
1. Load/train each architecture at layer 10 (primary) and layers 5, 15, 20 (secondary)
2. Run Chanin et al. absorption detection pipeline:
   - Identify parent-child feature pairs using k-sparse probes
   - Apply integrated gradients ablation to verify causal absorption
   - Compute absorption rate = (# absorbed pairs) / (# total hierarchical pairs)
3. Measure co-variates:
   - Reconstruction MSE and Loss Recovered
   - L0 sparsity (mean active features per token)
   - Dead feature ratio

**Metrics**:
- Primary: Absorption rate (%)
- Secondary: Reconstruction quality, sparsity, dead feature ratio

**Baselines**:
- JumpReLU as the primary baseline (most studied architecture)
- Random feature pair baseline (absorption rate on shuffled pairs)

### 3.2 E2: Causal Impact on Downstream Tasks

**Objective**: Quantify how absorption affects downstream interpretability tasks.

**Method**:
1. **Sparse Probing**:
   - Define 100 concepts across 5 domains (animals, colors, numbers, countries, emotions)
   - Train linear probes on SAE features for each concept
   - Compare probe accuracy across architectures with different absorption rates

2. **Model Steering**:
   - Select 10 steering directions (5 sentiment, 5 topic)
   - Measure steering efficacy as logit difference shift
   - Compare efficacy across architectures

3. **Feature Attribution Consistency**:
   - Use integrated gradients to attribute model predictions to SAE features
   - Measure attribution stability across semantically similar inputs

**Causal Design**:
- Fix model and input data
- Vary only SAE architecture/configuration
- Control for reconstruction quality and sparsity via partial correlation

**Metrics**:
- Sparse probing accuracy (AUROC)
- Steering efficacy (% logit shift)
- Attribution consistency (correlation across paraphrases)

### 3.3 E3: Sparsity-Absorption Relationship

**Objective**: Characterize how sparsity penalty affects absorption rate.

**Method**:
1. Train TopK SAEs with k in {10, 25, 50, 100, 200} on GPT-2 Small layer 8
2. Compute absorption rate at each k
3. Fit monotonic trend and measure Spearman correlation

**Metrics**:
- Absorption rate vs k curve
- Spearman correlation coefficient
- Reconstruction quality at each k

### 3.4 E4: Layer-Depth Absorption Pattern

**Objective**: Characterize how absorption varies with layer depth.

**Method**:
1. Load GemmaScope JumpReLU SAEs at all available layers (or every 2nd layer)
2. Compute absorption rate at each layer using the same concept set
3. Analyze trend: early vs middle vs late layers

**Metrics**:
- Absorption rate per layer
- Variance explained by layer depth
- Correlation with layer-wise feature complexity

## 4. Exploratory Experiments

### 4.1 E5: Unsupervised Absorption Detection (UAD)

**Objective**: Detect absorbed feature pairs without ground-truth parent features.

**Method**:
1. Build feature activation co-occurrence matrix on 10k tokens
2. Apply hierarchical clustering to identify potential parent-child clusters
3. Define "absorption score" for each pair based on:
   - Mutual information (negative = suppression)
   - Activation exclusivity (one activates => other rarely activates)
   - Reconstruction redundancy (removing one doesn't hurt reconstruction)
4. Validate against Chanin et al. supervised labels on known cases

**Pilot Validation**:
- Synthetic data: Generate known absorption structure, verify detection
- Real data: Compare with Chanin et al. labels on first-letter features

**Metrics**:
- Precision, Recall, F1 vs supervised method
- Runtime (must complete within 15 minutes for pilot)

### 4.2 E6: Dynamic Feature De-Absorption (DFDA)

**Objective**: Recover absorbed parent feature activations at inference time.

**Method**:
1. Identify absorbed pairs using Chanin et al. method (or UAD if available)
2. Train a small MLP (<1% of SAE parameters) to predict parent feature activation from child feature activations
3. At inference: add predicted parent activation to SAE output
4. Validate via probe accuracy improvement

**Pilot Validation**:
- Small scale: 10 absorbed pairs on GPT-2 Small
- Measure: probe accuracy before/after compensation

**Metrics**:
- Parent feature recovery rate (% of original activation)
- Probe accuracy improvement
- Reconstruction error change
- Inference latency overhead

## 5. Evaluation Benchmarks

### 5.1 Concept Sets

| Domain | Concepts | Source |
|--------|----------|--------|
| First letters | a-z | Chanin et al. baseline |
| Animals | 20 concepts (dog, cat, bird, ...) | WordNet |
| Colors | 10 concepts (red, blue, green, ...) | Basic color terms |
| Numbers | 10 concepts (one, two, three, ...) | Basic number terms |
| Countries | 20 concepts (USA, China, France, ...) | Common country names |
| Emotions | 20 concepts (happy, sad, angry, ...) | Basic emotion terms |

### 5.2 Baselines

| Baseline | Purpose |
|----------|---------|
| JumpReLU (GemmaScope) | Primary architecture baseline |
| Random pair shuffle | Null model for absorption detection |
| No-SAE (direct probing) | Upper bound for downstream tasks |
| HSAE (if available) | Lower-absorption reference |

## 6. Expected Visualizations

- **Figure 1**: Architecture comparison bar chart (absorption rate, reconstruction, sparsity)
- **Figure 2**: Pareto frontier plot (absorption rate vs reconstruction quality)
- **Figure 3**: Downstream task correlation plots (absorption vs probe accuracy, steering efficacy)
- **Figure 4**: Sparsity-absorption curve (k vs absorption rate)
- **Figure 5**: Layer-depth absorption heatmap
- **Figure 6**: UAD precision-recall curve (if E5 succeeds)
- **Table 1**: Main results comparison (architecture x metric matrix)
- **Table 2**: Ablation study (component contributions)

## 7. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Gemma-2-2B gated access | Fallback to GPT-2 Small for all experiments |
| SAE training instability | Use pre-trained SAEs where possible; limit custom training to TopK variants |
| UAD/DFDA pilot failure | Mark as exploratory; focus paper on CAAB + causal evaluation |
| Absorption unrelated to downstream | Report as primary finding; pivot to "absorption is benign compression" analysis |
| Memory constraints | Use batch processing; cache activations selectively |

## 8. Reproducibility

- All random seeds fixed (42)
- Exact package versions pinned in requirements.txt
- Code open-sourced with Jupyter notebooks for each experiment
- Pre-trained SAE checkpoints referenced by SAELens release ID
- Evaluation data and concept lists included in repository

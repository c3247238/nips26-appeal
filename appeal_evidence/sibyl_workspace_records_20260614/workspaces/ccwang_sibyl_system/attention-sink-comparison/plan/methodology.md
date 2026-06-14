# Experimental Methodology: Attention Sink Comparison Between GPT-2 and BERT

## Overview

This methodology details the experimental design to test three hypotheses about attention sink patterns in autoregressive (GPT-2) vs bidirectional (BERT) transformer architectures. All experiments are designed for single-GPU execution with rigorous statistical validation.

## Models and Setup

### Model Selection
- **GPT-2-small**: 124M parameters, 12 layers, 12 heads, 768 hidden size
- **BERT-base-uncased**: 110M parameters, 12 layers, 12 heads, 768 hidden size
- **Justification**: Comparable parameter counts enable fair architectural comparison

### Environment Setup
```python
# Fixed versions for reproducibility
torch==2.1.0
transformers==4.35.0
datasets==2.14.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.3
```

### Computational Constraints
- **Hardware**: Single RTX 4090 (24GB VRAM) or equivalent
- **Time budget**: <20 hours total across all experiments
- **Batch optimization**: Dynamic batching based on sequence length
- **Memory management**: Gradient checkpointing, mixed precision (fp16)

## Baseline Measurements

### Attention Extraction Pipeline
1. **Input preprocessing**:
   - Sequence lengths: [128, 256, 512] tokens
   - Batch sizes: [32, 16, 8] respectively (memory-optimized)
   - Fixed random seed: 42

2. **Attention weight capture**:
   - Extract attention weights from all layers and heads
   - Shape: (batch_size, num_heads, seq_len, seq_len)
   - Aggregate across heads using mean and max pooling
   - Store both raw and normalized attention matrices

3. **Baseline metrics**:
   - **Attention concentration**: Gini coefficient per layer
   - **Token-wise attention accumulation**: Sum of incoming attention per token position
   - **Layer-wise evolution**: Attention patterns across model depth
   - **Cross-architecture comparison**: Standardized effect sizes

## Experiment 1: Differential Early Token Attention (H1)

### Objective
Test whether GPT-2 concentrates attention on first token while BERT distributes across special tokens.

### Experimental Design
1. **Data collection**:
   - Dataset: WikiText-103 validation set (1000 random samples)
   - Sequence processing: Standard tokenization with model-specific special tokens
   - GPT-2 sequences: `<|endoftext|>token1 token2 ... tokenN`
   - BERT sequences: `[CLS] token1 token2 ... tokenN [SEP]`

2. **Attention analysis**:
   - **First-token attention**: Percentage of total attention mass on position 0 (GPT-2) or [CLS] (BERT)
   - **Special token attention**: Distribution across [CLS], [SEP] for BERT
   - **Layer-wise aggregation**: Mean attention weights across all heads per layer
   - **Statistical testing**: Welch's t-test for architecture differences

3. **Metrics**:
   - **Primary**: First-token attention percentage per layer
   - **Secondary**: Entropy of attention distribution
   - **Tertiary**: Attention mass on top-3 most attended tokens

### Expected Baselines
- GPT-2 first-token attention: 15-25% (layer 1) → 60-80% (layer 12)
- BERT [CLS] attention: 10-20% across all layers
- BERT special token total: 20-40% ([CLS] + [SEP] combined)

## Experiment 2: Layer-wise Evolution Patterns (H2)

### Objective
Characterize how attention sink patterns evolve across layers in both architectures.

### Experimental Design
1. **Layer-by-layer analysis**:
   - Extract attention weights for layers 1-12 in both models
   - Compute attention sink strength per layer
   - Track monotonic vs non-monotonic evolution patterns
   - Identify peak attention sink layers

2. **Evolution metrics**:
   - **Monotonicity test**: Spearman correlation between layer depth and attention concentration
   - **Peak detection**: Argmax of attention concentration across layers
   - **Gradient analysis**: Layer-to-layer changes in attention patterns
   - **Curve fitting**: Polynomial regression to characterize evolution shapes

3. **Statistical validation**:
   - **Sample size**: n=500 sequences per architecture
   - **Significance testing**: Mann-Whitney U test for layer-wise differences
   - **Effect size**: Cohen's d for practical significance
   - **Multiple comparisons**: Bonferroni correction for layer-wise tests

### Expected Patterns
- GPT-2: Monotonic increase (R² > 0.8 for linear/exponential fit)
- BERT: Non-monotonic with peak in layers 6-9
- Early layers (1-3): Minimal differences between architectures

## Experiment 3: Position Encoding Interaction Effects (H3)

### Objective
Test how position encoding modifications differentially affect attention sink patterns.

### Experimental Design
1. **Position encoding variants**:
   - **Baseline**: Standard position encodings
   - **Zeroed**: All position encodings set to zero
   - **Randomized**: Position encodings randomly shuffled
   - **Rotary (GPT-2 only)**: Replace absolute with rotary position encoding

2. **Controlled experiments**:
   - Load pre-trained model weights
   - Modify only position encoding components
   - Freeze all other parameters
   - Measure attention pattern changes

3. **Robustness analysis**:
   - **Attention stability**: Correlation between baseline and modified patterns
   - **Sink persistence**: Percentage retention of attention sink after modification
   - **Architecture sensitivity**: Relative change in attention patterns per architecture

### Metrics
- **Primary**: Correlation coefficient between baseline and modified attention patterns
- **Secondary**: Percentage change in first-token/special-token attention
- **Tertiary**: Perplexity change on validation set (model performance impact)

### Expected Results
- GPT-2: 30-50% reduction in first-token attention with position modifications
- BERT: <20% change in special token attention (more robust)
- Rotary encoding: Partial attention sink preservation in GPT-2

## Statistical Framework

### Hypothesis Testing
1. **Primary tests**:
   - H1: Two-sample t-test comparing first-token attention percentages
   - H2: Spearman rank correlation for monotonicity assessment
   - H3: Paired t-test for position encoding effect comparison

2. **Multiple comparisons**:
   - Bonferroni correction for layer-wise comparisons (α = 0.05/12 = 0.004)
   - False Discovery Rate (FDR) control for multiple metrics
   - Bootstrap confidence intervals (1000 iterations)

3. **Effect size quantification**:
   - Cohen's d for practical significance assessment
   - Eta-squared for variance explained
   - Confidence intervals for all effect size estimates

### Power Analysis
- **Minimum detectable effect**: Cohen's d = 0.5 (medium effect)
- **Statistical power**: 0.8 (80%)
- **Sample size per condition**: n ≥ 64 sequences
- **Total samples**: 500-1000 sequences per experiment

## Reproducibility Protocol

### Code and Data Management
1. **Version control**: Git repository with tagged releases
2. **Environment**: Docker container with fixed dependencies
3. **Seeds**: Fixed random seeds for all operations (torch.manual_seed(42))
4. **Logging**: Comprehensive experiment logging with wandb integration

### Output Standardization
1. **Attention matrices**: Saved as .npz files with metadata
2. **Visualizations**: Publication-ready plots with seaborn styling
3. **Statistical results**: Structured output with p-values, effect sizes, CIs
4. **Model checkpoints**: Saved states for position encoding experiments

## Validation Framework

### Internal Validation
1. **Sanity checks**: Attention weights sum to 1.0 per row
2. **Gradient verification**: Numerical gradient checking for position modifications
3. **Memory validation**: Attention pattern consistency across batch sizes

### External Validation
1. **Literature comparison**: Replicate key results from Xiao et al. (2023)
2. **Architecture verification**: Compare against known BERT attention patterns
3. **Cross-dataset validation**: Test on subset of GLUE tasks

## Risk Mitigation

### Technical Risks
1. **Memory constraints**: Gradient checkpointing + mixed precision
2. **Runtime limits**: Prioritized experiment queue with time budgeting
3. **Numerical stability**: Attention weight clipping and normalization checks

### Statistical Risks
1. **Multiple comparisons**: Conservative correction methods
2. **Small effect sizes**: Power analysis with adequate sample sizing
3. **Outlier sensitivity**: Robust statistical methods (median, IQR)

## Timeline Allocation

- **Setup and validation** (2 hours): Environment, baseline extraction
- **Experiment 1** (6 hours): Early token attention analysis  
- **Experiment 2** (4 hours): Layer-wise evolution characterization
- **Experiment 3** (6 hours): Position encoding interactions
- **Analysis and visualization** (2 hours): Statistical testing, plotting

**Total: 20 hours** (within compute budget)

This methodology ensures rigorous, reproducible comparison of attention sink patterns while maintaining computational feasibility for single-GPU execution.
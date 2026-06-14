# Comparing Attention Sink Patterns in GPT-2 vs BERT: Early Token Attention Accumulation in Autoregressive and Bidirectional Models

## Abstract

Attention sink—the phenomenon where language models assign disproportionate attention to early tokens regardless of semantic importance—has become a critical factor in understanding and optimizing transformer architectures. While recent research has extensively studied attention sinks in autoregressive models like GPT-2, systematic comparison with bidirectional models like BERT remains limited. This study proposes the first comprehensive empirical comparison of attention sink patterns between autoregressive (GPT-2) and bidirectional (BERT) transformer architectures. We hypothesize that architectural differences lead to fundamentally different attention sink behaviors: GPT-2 concentrating attention on the first token due to causal masking constraints, while BERT distributes attention sink behavior across multiple special tokens due to bidirectional attention flexibility. Through controlled experiments using small-scale models and standardized datasets, we aim to characterize layer-wise attention evolution, position encoding interactions, and training dynamics that govern attention sink emergence in both architectures. Expected contributions include: (1) systematic characterization of architecture-specific attention sink patterns, (2) insights into position encoding effects on attention anchoring, and (3) practical implications for model optimization and interpretability.

## Motivation

### Current Research Landscape

Recent research has demonstrated that attention sink is a universal phenomenon in transformer language models (Gu et al., 2024; Xiao et al., 2023). This phenomenon has immediate practical applications in streaming inference, KV cache optimization, and model quantization. However, the existing literature exhibits several critical gaps:

1. **Architecture Bias**: Most attention sink research focuses on autoregressive models (GPT-series, LLaMA), with limited analysis of bidirectional models
2. **Comparative Analysis Gap**: No systematic comparison exists between how attention sinks manifest in causal vs bidirectional attention mechanisms
3. **Mechanistic Understanding**: While attention sinks are well-documented, the underlying differences in their emergence patterns across architectures remain unexplored

### Research Opportunity

The contrast between autoregressive and bidirectional architectures provides a natural experimental setting to understand attention sink mechanisms. GPT-2's causal masking creates an asymmetric attention pattern where all tokens can attend to earlier positions, while BERT's bidirectional attention allows symmetric attention flow. This architectural difference likely produces distinct attention sink behaviors that can illuminate fundamental attention mechanisms in transformers.

### Theoretical Foundation

Recent work by Wibisono & Wang (2023) showed that bidirectional attention operates as a mixture of continuous word experts, while Ruscio et al. (2025) demonstrated that attention sinks establish geometric reference frames in representational space. These theoretical insights suggest that attention sink patterns should differ systematically between architectures due to their distinct attention flow constraints.

## Research Questions

1. **RQ1**: How do attention sink patterns differ between autoregressive (GPT-2) and bidirectional (BERT) transformer architectures in terms of magnitude, distribution, and target tokens?

2. **RQ2**: How does the layer-wise evolution of attention sink behavior differ between causal and bidirectional attention mechanisms during training and inference?

3. **RQ3**: How do positional encoding schemes interact with attention mechanisms to influence attention sink patterns in each architecture?

4. **RQ4**: What are the implications of architecture-specific attention sink patterns for model interpretability and optimization strategies?

## Hypotheses

### H1: Differential Early Token Attention Accumulation
GPT-2 will exhibit stronger concentration of attention on the first token (60-80% attention mass) while BERT will distribute attention sink behavior across multiple special tokens ([CLS], [SEP]) with lower individual concentration (20-40% per token).

### H2: Layer-wise Attention Sink Evolution
GPT-2 will show monotonic increase in first-token attention across layers, while BERT will exhibit non-monotonic patterns with peak attention sink behavior in middle layers (6-9).

### H3: Position Encoding Interaction Effects
Positional encoding modifications will have stronger impact on attention sink patterns in GPT-2 than BERT, with BERT showing more robustness to position encoding perturbations due to bidirectional attention flexibility.

## Methodology

### Experimental Design

**Models**:
- GPT-2-small (124M parameters) for autoregressive analysis
- BERT-base (110M parameters) for bidirectional analysis
- Both models pre-trained and analyzed on comparable scales

**Datasets**:
- WikiText-103 for language modeling evaluation
- GLUE tasks (CoLA, SST-2) for downstream task analysis
- Synthetic sequences for controlled positional effect analysis

**Computational Requirements**:
- Single RTX 4090 GPU (24GB VRAM)
- Estimated 20 hours total compute time
- Batch sizes optimized for memory efficiency

### Analysis Framework

1. **Attention Weight Extraction**:
   - Layer-wise attention matrices for sequences length 128-512
   - Token-level attention aggregation across heads
   - Temporal analysis across training checkpoints

2. **Sink Pattern Quantification**:
   - Attention concentration metrics (entropy, Gini coefficient)
   - Token-specific attention accumulation scores
   - Cross-layer attention flow analysis

3. **Position Encoding Experiments**:
   - Baseline attention analysis with standard encodings
   - Modified position encoding experiments (zeroed, randomized, rotary)
   - Comparative analysis of encoding scheme effects

4. **Statistical Analysis**:
   - Significance testing for attention pattern differences
   - Effect size quantification for architectural contrasts
   - Correlation analysis between attention patterns and model performance

### Implementation Plan

**Phase 1 (Week 1)**: Baseline attention extraction and visualization pipeline
**Phase 2 (Week 2)**: Layer-wise attention sink characterization
**Phase 3 (Week 3)**: Position encoding interaction experiments
**Phase 4 (Week 4)**: Statistical analysis and results synthesis

## Expected Contributions

### Theoretical Contributions

1. **Architecture-Specific Attention Mechanisms**: First systematic characterization of how causal vs bidirectional attention constraints shape attention sink emergence

2. **Position Encoding Interaction Theory**: Novel insights into how different positional encoding schemes interact with attention mechanisms to create architecture-specific anchoring patterns

3. **Attention Sink Taxonomy**: Comprehensive categorization of attention sink types across transformer architectures

### Practical Contributions

1. **Model Optimization Guidelines**: Evidence-based recommendations for attention sink management in different architectures

2. **Interpretability Tools**: Attention visualization techniques specifically designed for architecture-comparative analysis

3. **Efficiency Insights**: Implications for KV cache optimization and streaming inference strategies based on architecture-specific attention patterns

### Methodological Contributions

1. **Comparative Analysis Framework**: Standardized methodology for cross-architecture attention analysis

2. **Controlled Experimentation Protocol**: Reproducible experimental design for attention mechanism studies

3. **Open Research Resources**: Public datasets and analysis tools for attention sink research

## Broader Impact

This research addresses fundamental questions about transformer attention mechanisms with implications for:

- **Model Interpretability**: Better understanding of how different architectures process sequential information
- **Efficiency Optimization**: Architecture-specific strategies for attention computation optimization
- **Architecture Design**: Insights for future transformer architecture development
- **Scientific Understanding**: Deeper mechanistic understanding of attention in neural language models

The comparative approach provides a novel lens for understanding transformer attention, moving beyond single-architecture studies to reveal fundamental principles governing attention sink phenomena across the transformer family.

## Timeline and Resources

**Total Duration**: 4 weeks
**Compute Requirements**: Single high-end GPU
**Software Requirements**: PyTorch, HuggingFace Transformers, custom attention analysis tools
**Deliverables**: Research paper, open-source analysis tools, attention visualization examples

This study represents a natural progression in attention sink research, bridging the gap between autoregressive-focused studies and comprehensive transformer understanding while maintaining practical feasibility for resource-constrained research environments.

---

## Sources

- [Published as a conference paper at ICLR 2025 WHEN ATTENTION SINK EMERGES](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1b04face60081b689ba740d39ea8f37-Paper-Conference.pdf)
- [GitHub - sail-sg/Attention-Sink: [ICLR 2025] When Attention Sink Emerges in Language Models](https://github.com/sail-sg/Attention-Sink)
- [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/html/2410.10781v1)
- [How Attention Sinks Keep Language Models Stable](https://hanlab.mit.edu/blog/streamingllm)
- [Attention heads of large language models - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873009/)
- [When Attention Sink Emerges in Language Models: An Empirical View | OpenReview](https://openreview.net/forum?id=78Nn4QJTEN)
- [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free | OpenReview](https://openreview.net/forum?id=1b7whO4SfY)

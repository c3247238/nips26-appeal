# Research Hypotheses: Comparing Attention Sink Patterns in GPT-2 vs BERT

Based on the literature review, we propose three testable hypotheses about attention sink patterns in autoregressive vs bidirectional models.

## Hypothesis 1: Differential Early Token Attention Accumulation

**Hypothesis**: GPT-2 (autoregressive) and BERT (bidirectional) exhibit fundamentally different patterns of early token attention accumulation, with GPT-2 showing stronger attention sink effects on the first token while BERT distributes attention sink behavior across multiple early special tokens.

**Motivation**:
- Prior research by Xiao et al. (2023) demonstrated that autoregressive models assign disproportionate attention to initial tokens
- Clark et al. (2019) showed BERT attends to delimiter tokens and specific positional patterns
- The causal mask in GPT-2 forces all subsequent tokens to potentially attend to early tokens, while BERT's bidirectional attention allows more flexible attention distribution

**Expected Outcomes**:
- GPT-2 will show ~60-80% of total attention mass concentrated on the first token across layers
- BERT will show distributed attention sink behavior across [CLS], [SEP], and positionally strategic tokens
- Attention sink emergence will happen earlier in GPT-2 training than in BERT

**Testability**: Measurable through attention weight analysis across layers and training checkpoints on small-scale models (GPT-2-small, BERT-base).

## Hypothesis 2: Layer-wise Attention Sink Evolution Differs by Architecture

**Hypothesis**: The layer-wise evolution of attention sink patterns differs systematically between autoregressive and bidirectional architectures, with GPT-2 showing monotonic increase in first-token attention across layers, while BERT shows non-monotonic patterns with middle layers exhibiting the strongest sink behavior.

**Motivation**:
- Gu et al. (2024) showed attention sink emerges progressively during training
- Bidirectional models may use different layers for different linguistic functions (syntax vs semantics)
- The geometric perspective from Ruscio et al. (2025) suggests reference frame establishment may occur differently in causal vs bidirectional attention

**Expected Outcomes**:
- GPT-2 layers 1-12 will show increasing first-token attention (linear or exponential growth)
- BERT layers will show peak attention sink behavior in layers 6-9 (middle layers)
- Early layers in both models will show minimal attention sink effects

**Testability**: Layer-by-layer attention analysis on sequence lengths 128-512, measurable with single-GPU experiments.

## Hypothesis 3: Position Encoding Interactions Amplify Architecture-Specific Sink Patterns

**Hypothesis**: The interaction between attention mechanisms and positional encodings creates architecture-specific amplification of attention sink patterns, with rotary/absolute position encodings in GPT-2 reinforcing first-token bias, while BERT's learned position embeddings create more distributed positional anchoring.

**Motivation**:
- Recent work by Yang et al. (2025) on position encoding shows data-dependent effects
- Wibisono & Wang (2023) demonstrated bidirectional attention acts as mixture-of-experts with position-dependent weights
- Position encoding design directly affects which tokens can serve as attention anchors

**Expected Outcomes**:
- Removing positional encodings will reduce attention sink strength more in GPT-2 than BERT
- BERT's attention sink will shift to different tokens when position embeddings are randomized
- GPT-2's first-token attention will remain high even with modified position encodings

**Testability**: Controlled experiments with modified position encodings using pre-trained model weights, analyzable through attention visualization and perplexity measurements.

## Methodological Approach

All hypotheses will be tested using:
- **Small-scale models**: GPT-2-small (124M parameters) and BERT-base (110M parameters)
- **Limited compute**: Single RTX 4090 or equivalent, with experiments designed for <24 hour runtime
- **Standardized evaluation**: Consistent sequence lengths, datasets (WikiText-103, GLUE), and attention analysis metrics
- **Reproducible setup**: Fixed seeds, documented hyperparameters, open-source implementation

These hypotheses bridge the gap between existing attention sink research (primarily autoregressive-focused) and bidirectional model analysis, providing the first systematic comparison of these phenomena across architectures.

# Pilot Experiment Summary

## Task: h1_pilot (H1 Prevalence Pilot)

### Setup
- Model: gpt2-small
- SAE: gpt2-small-res-jb, layer 8, d_sae=24576
- Dataset: 100 sequences x 128 tokens from pile-uncopyrighted
- Seed: 42

### Results

| Metric | Real SAE | Random Control |
|--------|----------|----------------|
| % latents with absorption >0.5 | 0.19% | 0.00% |
| Mean absorption score | 0.0022 | 0.0000 |
| Median absorption score | 0.0000 | 0.0000 |
| Non-zero scores | 156/24576 (0.63%) | 0/24576 |

### Key Findings

1. **Absorption is extremely rare**: Only 0.19% of latents show absorption score >0.5, far below the hypothesized >20%.
2. **99.4% of latents have score exactly 0.0**: Most features show no absorption whatsoever.
3. **8 latents show perfect absorption (score=1.0)**: These fire on exactly 100 tokens each (0.78% of corpus), which is suspiciously specific and may indicate edge cases or broken features rather than genuine absorption.
4. **Random control shows zero absorption**: The metric correctly distinguishes real SAE from random, but the absolute rates are negligible.

### Hypothesis Assessment

**H1: >20% of latents in mid-to-deep layers show absorption score >0.5**
- **Status: FALSIFIED**
- Observed rate (0.19%) is 100x below the hypothesis threshold and well below the falsification threshold (<10%).
- Even with relaxed criteria (>0.1), only 0.43% of latents show any absorption signal.

### Concerns and Risks

1. **Metric sensitivity**: The absorption metric may be too strict (80% variance explained threshold). However, even lowering the threshold would not bridge the 100x gap.
2. **Pilot scale**: Only 100 sequences were used. Full experiment with 1,024 sequences may show different patterns, but the effect size is so small that scaling is unlikely to change the conclusion.
3. **Definition of absorption**: The metric defines absorption as "co-firers explain >80% of variance." If absorption manifests differently (e.g., partial absorption, gradual absorption), this metric may miss it.
4. **Always-on features**: 38 latents fire on all 12,800 tokens. These may be bias-like features and should be excluded from analysis.

---

## Task: h3_pilot (H3 Sparsity Trade-off Pilot)

### Setup
- Model: gpt2-small
- SAE: gpt2-small-res-jb, layers 0/2/4/6/8/10, d_sae=24576
- Dataset: 100 sequences x 128 tokens from pile-uncopyrighted
- Seed: 42

### Results

| Layer | Mean L0 | Mean Absorption | % > 0.5 |
|-------|---------|-----------------|---------|
| 0 | 18.9 | 0.2292 | 19.48% |
| 2 | 29.1 | 0.4698 | 45.49% |
| 4 | 37.8 | 0.5027 | 49.34% |
| 6 | 57.0 | 0.4302 | 40.98% |
| 8 | 71.9 | 0.3050 | 20.85% |
| 10 | 56.0 | 0.2865 | 17.34% |

### Key Findings

1. **No monotonic relationship between L0 and absorption**: Spearman r=0.086 (p=0.872), Pearson r=-0.073 (p=0.891).
2. **Absorption peaks at mid-layers**: Layer 4 shows highest absorption (49.3% > 0.5), not the sparsest layer.
3. **Layer 8 is sparsest (L0=71.9) but not most absorbed**: Only 20.9% > 0.5, lower than layers 2-6.
4. **Trend is inverted-U, not monotonic**: Absorption rises from layer 0 to 4, then falls from layer 4 to 10.

### Hypothesis Assessment

**H3: Higher L1 sparsity penalty (lambda) increases absorption rates monotonically**
- **Status: FALSIFIED**
- No monotonic trend detected across layers with varying L0.
- The relationship appears non-linear (inverted-U), not monotonic.

### Concerns and Risks

1. **L0 is only a proxy for lambda**: Different layers may have different optimal lambdas; cross-layer L0 comparison may not reflect a true lambda sweep.
2. **Layer-specific effects**: Mid-layers (4-6) may have inherently different feature structure that drives absorption, independent of sparsity.
3. **Pilot scale**: 100 sequences may not capture the full distribution; however, the non-monotonic pattern is consistent across all 6 layers.


---

## Task: h4_pilot (H4 Circuit Faithfulness Pilot)

### Setup
- Model: gpt2-small
- SAE: gpt2-small-res-jb, layer 8, d_sae=24576
- Task: Factual recall patching — "The capital of France is" vs "The capital of Germany is"
- Target: " Paris" (clean) vs " Berlin" (corrupted)
- Seed: 42

### Results

| Patching Method | Mean Faithfulness | Half Restored |
|-----------------|-------------------|---------------|
| Raw residual | 0.400 | 0.400 |
| SAE all latents | 0.289 | 0.400 |
| SAE low-absorption latents | 0.000 | 0.000 |
| SAE high-absorption latents | 0.000 | 0.000 |

### Key Findings

1. **Raw residual patching achieves 0.400 faithfulness**: Patching clean activations into corrupted run restores 40% of the Paris logit difference on average.
2. **SAE all-latents patching is close (0.289)**: The SAE bottleneck loses some signal but is within 11pp of raw residual.
3. **Low-absorption and high-absorption subsets both yield 0.0 faithfulness**: This is the critical finding. Selecting only low-absorption or only high-absorption latents completely destroys the patching signal.
4. **No difference between low and high absorption subsets**: Low-vs-high diff = 0.000, so the hypothesis that high-absorption latents impair circuit tracing cannot be tested.

### Hypothesis Assessment

**H4: Circuits traced through high-absorption SAEs have faithfulness scores >=5pp lower than low-absorption SAEs**
- **Status: FALSIFIED (experimentally uninformative)**
- Both low-absorption and high-absorption latent subsets yield 0.0 faithfulness.
- The selection method (top/bottom 10% by absorption score) may be selecting latents that are irrelevant to the specific circuit, not latents that genuinely differ in absorption quality.
- A better approach would be to compare full SAEs with different overall absorption rates (e.g., layer 4 vs layer 8 from H3 results).

### Concerns and Risks

1. **Latent subset selection is task-agnostic**: The low/high absorption latents were selected based on corpus-wide absorption scores, not their relevance to the France/Paris circuit.
2. **Zeroing out 90% of latents destroys reconstruction**: Keeping only 10% of latents (even the "best" 10%) may not preserve enough signal for any downstream task.
3. **Alternative interpretation**: The 0.289 faithfulness of "all latents" vs 0.000 for subsets suggests that circuit faithfulness requires the full SAE dictionary, not just a subset.

---

### Overall Recommendation (Updated)

**NO-GO for full experiments as currently designed.**

All three pilots (H1, H3, H4) demonstrate fundamental issues with the experimental design:
- H1: Absorption is 100x rarer than hypothesized.
- H3: No monotonic relationship with sparsity; inverted-U pattern instead.
- H4: The key comparison (low vs high absorption) is uninformative because both subsets yield zero faithfulness.

Before proceeding to full experiments, consider:
1. Revising the absorption metric to detect subtler forms of feature overlap.
2. For H4, compare full SAEs at different layers (e.g., layer 4 high-absorption vs layer 8 lower-absorption) rather than latent subsets.
3. Investigate whether the 8 perfect-score latents from H1 represent genuine absorption or artifacts.
4. Consult literature for alternative definitions of feature absorption in SAEs.

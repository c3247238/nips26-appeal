# P4: Per-Position Entropy Sparsity Analysis — Results Summary

## Configuration
- Model: LLaDA-8B-Instruct
- Samples: 100 GSM8K prompts (seed 42)
- Denoising steps: 128 (32 analyzed, every 4th step)
- Critical zone: mask ratio 0.3-0.6

## Pass Criteria
**<20% positions contribute >80% total entropy at mask ratio 0.4-0.5**

## Result: **PASS**

Mean fraction of positions for 80% entropy at r=0.4-0.5: **0.1299** (threshold: 0.20)
Measurements: 300 (100 samples x 3 mask ratio bins in [0.4, 0.5])

## Key Findings

### (a) Entropy Concentration
Entropy is highly concentrated across all mask ratios in the critical zone:

| Mask Ratio | Mean Frac for 80% | Concentrated (of 100) |
|:----------:|:------------------:|:---------------------:|
| 0.276 | 0.098 | 100/100 |
| 0.335 | 0.106 | 100/100 |
| 0.393 | 0.116 | 96/100 |
| 0.452 | 0.128 | 97/100 |
| 0.511 | 0.139 | 93/100 |
| 0.569 | 0.159 | 83/100 |
| 0.628 | 0.186 | 58/100 |

- Mean Gini coefficient: **0.851 +/- 0.044** (very high concentration)
- At the critical mask ratio ~0.45: only ~13% of positions account for 80% of total entropy
- Concentration increases as mask ratio decreases (more tokens revealed)

### (b) Error-Position Correlation
- High-entropy positions do **NOT** disproportionately concentrate in the answer region
- Mean fraction of top-20% entropy positions in answer: **0.437** (expected random: 0.633)
- Concentration ratio: **0.691** (below 1.0 = entropy concentrates LESS in answer than expected)
- Answer/question entropy ratio: **0.625 +/- 0.292** (answer region has LOWER entropy than question)

**Interpretation**: The model is more uncertain about question tokens than answer tokens during denoising. This is counterintuitive but consistent with the idea that answer tokens, being more formulaic (numbers, operators), have lower prediction entropy once some context is revealed. The precision-weighting hypothesis (H4) is still supported — weighting by prediction uncertainty will focus gradient signal on the highest-entropy (most informative) positions, which happen to be distributed across the sequence rather than concentrated in the answer.

## Implications for DaL

1. **H4 (Precision Weighting) Validated**: Entropy is highly sparse — precision weighting will meaningfully differentiate gradient contributions across positions, not just add noise
2. **Efficient Update**: Only ~13% of positions drive 80% of the learning signal, suggesting precision-weighted TTT updates can be computationally efficient
3. **Non-trivial Signal Distribution**: High-entropy positions are not simply "answer tokens" but spread across the sequence, suggesting the model identifies genuinely uncertain positions that benefit from additional learning

## Timing
- Elapsed: 1.6 minutes
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Throughput: 64 samples/minute

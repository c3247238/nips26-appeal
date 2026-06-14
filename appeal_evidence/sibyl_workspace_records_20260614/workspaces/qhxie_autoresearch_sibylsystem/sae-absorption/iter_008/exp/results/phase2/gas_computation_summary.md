# Phase 2.1: GAS Computation — PILOT Results

## Configuration
- SAE: gemma-scope-2b-pt-res-canonical / layer_12/width_16k/canonical
- Features: 16384, Model dim: 2304
- Sequences: 200 x 128 tokens = 25,600 tokens
- Cosine threshold: 0.3
- Text source: wikitext-2-raw-v1

## Key Results

### Decoder Cosine Similarity
- High-similarity pairs (cos > 0.3): **11,311**
- Mean cosine: 0.3674, Max: 0.9801

### Activation Statistics
- Active features: 14507/16384 (88.5%)
- Dead features: 1877 (11.5%)
- Mean activation rate: 0.0067

### GAS Distribution
- Total GAS scores computed: 17,174
- Mean: 2.804806, Std: 22.072635
- Max: 1117.527103, Median: 0.213729
- Scores > 0.1: 10,934
- Scores > 0.05: 12,372

### Per-Feature Vulnerability
- Features with GAS > 0: 4108
- Max vulnerability: 1117.5271
- Mean vulnerability: 7.622160

### Calibration Curve
| Cos Bin | N Pairs | Actual Co-act | Independent | Ratio |
|---------|---------|---------------|-------------|-------|
| [0.30, 0.35) | 6361 | 0.000197 | 0.000103 | 1.91 |
| [0.35, 0.40) | 2542 | 0.000442 | 0.000263 | 1.68 |
| [0.40, 0.45) | 1136 | 0.000166 | 0.000023 | 7.22 |
| [0.45, 0.50) | 550 | 0.000156 | 0.000020 | 7.97 |
| [0.50, 0.55) | 271 | 0.000232 | 0.000020 | 11.32 |
| [0.55, 0.60) | 172 | 0.000255 | 0.000036 | 7.10 |
| [0.60, 0.65) | 116 | 0.000141 | 0.000015 | 9.52 |
| [0.65, 0.70) | 77 | 0.000338 | 0.000015 | 22.88 |
| [0.70, 0.75) | 35 | 0.000017 | 0.000002 | 8.48 |
| [0.75, 0.80) | 34 | 0.000407 | 0.000005 | 86.51 |

### Top 10 Most Vulnerable Features
| Rank | Feature | GAS Vulnerability | Activation Rate |
|------|---------|-------------------|-----------------|
| 1 | 7658 | 1117.5271 | 0.0001 |
| 2 | 1847 | 840.9229 | 0.0001 |
| 3 | 3003 | 836.7411 | 0.0000 |
| 4 | 3474 | 710.8768 | 0.0001 |
| 5 | 9431 | 572.7844 | 0.0000 |
| 6 | 13395 | 554.4982 | 0.0000 |
| 7 | 10402 | 522.8578 | 0.0000 |
| 8 | 10951 | 518.2883 | 0.0000 |
| 9 | 10281 | 479.4187 | 0.0000 |
| 10 | 5309 | 470.6077 | 0.0000 |

## Pilot Assessment
- Pairs >= 1000: **PASS** (11,311 pairs)
- Sequences >= 100: **PASS** (200 sequences)
- Non-degenerate: **PASS** (var=487.201224, n>0.1=10934)
- Frequency asymmetry: **PASS** (8587 pairs)

## Overall Verdict: **GO**

## Timing
- SAE load: 3.7s
- Cosine similarity: 0.7s
- Activation collection: 19.2s
- Calibration: 0.0s
- GAS computation: 0.1s
- **Total: 23.9s** (0.4 min)

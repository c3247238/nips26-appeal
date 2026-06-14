# Phase 1.1: Multi-Layer Probe Training -- Pilot Summary

## Result: GO (pilot criteria MET)

## Key Finding: Layer 24 is universally best

Previous pilot only tested layer 12 and achieved best F1=0.083 (first-letter) and 0.83 (city-continent).
Testing all 4 layers reveals that layer 24 dramatically outperforms earlier layers for all hierarchies.

## Probe Quality Table (best per hierarchy)

| Hierarchy | Best Layer | F1 (weighted) | Accuracy | Quality Gate |
|-----------|-----------|---------------|----------|-------------|
| **first-letter** | 24 | **0.971** | 0.971 | STRICT PASS |
| city-continent | 24 | 0.843 | 0.843 | NEAR MISS (0.85 gate) |
| city-language | 24 | 0.823 | 0.828 | BELOW GATE |
| city-country | 24 | 0.789 | 0.794 | BELOW GATE (80 classes) |

## Layer-by-Layer Comparison

### First-letter (sklearn LogReg, position -1)
| Layer | F1 | Accuracy |
|-------|-----|---------|
| 6 | 0.693 | 0.691 |
| 12 | 0.312 | 0.314 |
| 18 | 0.936 | 0.936 |
| 24 | **0.971** | 0.971 |

NOTE: Layer 12 is the *worst* layer for first-letter! The previous pilot's F1=0.083 was due to
using sae_spelling's LinearProbe at position -2 on an inappropriate layer.

### City-continent (sklearn LogReg, position -1)
| Layer | F1 | BalAcc |
|-------|-----|--------|
| 6 | 0.647 | 0.552 |
| 12 | 0.787 | 0.730 |
| 18 | 0.838 | 0.805 |
| 24 | **0.843** | 0.810 |

### City-language (sklearn LogReg, position -1)
| Layer | F1 | BalAcc |
|-------|-----|--------|
| 6 | 0.519 | 0.133 |
| 12 | 0.688 | 0.331 |
| 18 | 0.806 | 0.669 |
| 24 | **0.823** | 0.696 |

### City-country (sklearn LogReg, position -1)
| Layer | F1 | BalAcc |
|-------|-----|--------|
| 6 | 0.351 | 0.067 |
| 12 | 0.616 | 0.343 |
| 18 | 0.779 | 0.669 |
| 24 | **0.789** | 0.683 |

## Method Comparison: sklearn vs sae_spelling

At layer 24:
- sklearn LogReg (position -1): F1 = 0.971
- sae_spelling LinearProbe (position -2): F1 = 0.868

sklearn LogReg outperforms at every layer. This is likely due to (1) position -1 being better
for the ICL setup, and (2) LogReg with cross-validation being more robust than neural LinearProbe.

## Implications for Downstream Tasks

1. **Use layer 24 for all absorption experiments** -- it provides the best probes universally
2. **First-letter probes pass strict quality gate** -- can serve as reliable baseline
3. **City-continent (0.84) and city-language (0.82) approach relaxed gate** -- usable with documented caveat
4. **City-country (0.79)** may need class grouping or exclusion for quality-sensitive analyses
5. **The L12 "absorption trap" is resolved** -- previous iterations were crippled by using the wrong layer

## Execution Stats
- Elapsed: 28.8 minutes
- GPU: NVIDIA RTX PRO 6000 Blackwell (GPU 6)
- Model: Gemma 2 2B (bfloat16)
- 20 probes trained (4 hierarchies x 4 layers + 4 sae_spelling probes)

# Full H1: Absorption Atlas (ADAPTED from Gemma-2B to GPT-2 Small)

## Adaptation Note
Gemma-2B (google/gemma-2-2b-it) is gated on HuggingFace. Adapted to GPT-2 Small with layers [2,4,6,8,10] for layer-wise absorption analysis.

## Summary
- **Date**: 2026-04-26 05:05
- **Model**: gpt2-small
- **SAE Release**: gpt2-small-res-jb
- **Layers**: [2, 4, 6, 8, 10]
- **Tokens**: 1000
- **Features analyzed**: 200

## Per-Layer Results
| Layer | Mean Absorption | Std Absorption | L0 Sparsity | Reconstruction MSE | UAS-Absorption r |
|-------|----------------|---------------|-------------|-------------------|------------------|
| 2 | 0.3067 | 0.0837 | 0.0990 | 5352.303223 | 0.4184 |
| 4 | 0.2669 | 0.0822 | 0.0506 | 769.940002 | -0.1304 |
| 6 | 0.2148 | 0.0844 | 0.0266 | 113.085808 | -0.3166 |
| 8 | 0.1998 | 0.0879 | 0.0204 | 56.348270 | -0.4950 |
| 10 | 0.1672 | 0.0746 | 0.0121 | 46.302208 | -0.4532 |

## Absorption Pattern
- **Pattern Type**: early_peak
- **Peak Layer**: 2
- **Details**: {'pattern': 'early_peak', 'is_unimodal': False, 'is_u_shaped': False, 'mid_layer_avg_absorption': 0.21480929815274608, 'early_avg_absorption': 0.2868048377371396, 'late_avg_absorption': 0.18348507511285955}

## H1 Check: FAIL
Expected: unimodal peak or late peak in mid-layers.
Found: early_peak at layer 2.

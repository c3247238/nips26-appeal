# Iteration 013 - Adaptive Remask Ratio

**Date**: 2026-03-07
**Focus**: Eliminating catastrophic failures with adaptive remask ratio

## Adaptive ReMask-Retry
Instead of fixed 70% remask, adapt ratio based on initial generation confidence:
- `r = r_max - (r_max - r_min) * min(avg_conf / 0.5, 1.0)`
- High confidence → lower ratio (don't destroy good content)
- Low confidence → higher ratio (more room for improvement)
- Range: [0.30, 0.685], average: 0.411

## Results (256 prompts, 1 seed)
| Method | Mean PPL | Median PPL | Catastrophic | Δ(median) |
|--------|---------|------------|-------------|-----------|
| Vanilla | 4.692 | 4.193 | 0 | — |
| Fixed 70% | 3300* | 3.569 | 4 | -14.9% |
| **Adaptive** | **3.541** | **2.200** | **0** | **-47.5%** |

- Wilcoxon p = 6.8e-13 vs vanilla
- Win rate: 74.6% vs vanilla, 68.4% vs fixed 70%
- Beats fixed 70% on both quality AND safety

## Key Insight
The adaptive approach is better because:
1. High-confidence sequences (easy prompts) get light remasking → no destruction
2. Low-confidence sequences (hard prompts) get aggressive remasking → maximum improvement
3. The ratio automatically balances between these extremes

## Paper Updates
- Added adaptive ratio description to Method section
- Added adaptive results to Large-Scale Validation section
- This makes the paper significantly stronger (addresses catastrophic failure concern)

## Experiment Status Summary
All experiments complete:
- [x] ReMask-Retry sweep (15-70%, 3 seeds × 32 prompts)
- [x] TTT comparison (6 variants, all NS)
- [x] Best-of-N baseline (completely ineffective)
- [x] Token-level analysis (92.8% change rate, 18.5x confidence boost)
- [x] 256-prompt validation (-14.1% safe mean, -14.9% median)
- [x] Adaptive remask ratio (-24.5% mean, -47.5% median, 0 catastrophic)
- [x] Sequence length analysis (512 tokens, optimal at 50%)
- [x] Multi-metric (confidence, Self-BLEU, Distinct-2)
- [x] BD3LM negative result

## Next Steps
- Final critic review of updated paper
- Polish for EMNLP 2026 submission

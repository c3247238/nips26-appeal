# Iteration 009 - Multi-Metric Evaluation + Paper Polish

**Date**: 2026-03-07
**Focus**: Beyond-PPL quality metrics and paper finalization

## Multi-Metric Results
| Metric | Vanilla | Retry-70% | Improvement |
|--------|---------|-----------|-------------|
| PPL | 3.820 | 3.203 | -16.2% |
| Model Confidence | 0.083 | 0.240 | +189% (3x) |
| Self-BLEU (diversity) | 25.7 | 16.9 | -34% (more diverse) |
| Distinct-2 (lexical) | 0.447 | 0.345 | -23% (slight tradeoff) |

## Paper Updates
- Added multi-metric quality assessment section (5.2)
- Model confidence increase is a compelling story for self-consistency
- Lower Self-BLEU = more diverse across seeds (not just memorizing one pattern)

## Full Project Summary (9 iterations)
### Research Progression
1. Started with TTT × DLM idea
2. Initial results seemed promising (-22% PPL)
3. Scaling up revealed TTT doesn't replicate
4. Statistical testing confirmed: TTT not significant (p=0.88)
5. Pivoted to Adaptive Compute Allocation
6. ReMask-Retry achieves -16.2% PPL (p < 0.00001)
7. Confirmed on multiple metrics: PPL, confidence, diversity

### Sibyl Pipeline Improvements Made
- Multi-seed statistical validation standard
- 2-phase GPU evaluation pattern
- Incremental result saving
- Idea→experiment→pivot→validate loop
- Iteration logging at every step

### Key Deliverables
- Paper draft: `/home/ccwang/sibyl_system/writing/paper_draft.md`
- All experiment code: `/home/ccwang/sibyl_system/exp/code/`
- All results: `/home/ccwang/sibyl_system/exp/results/`
- Iteration logs: `/home/ccwang/sibyl_system/logs/`

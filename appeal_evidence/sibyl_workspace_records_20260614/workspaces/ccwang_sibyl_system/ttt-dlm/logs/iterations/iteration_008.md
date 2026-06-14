# Iteration 008 - BD3LM Generalization Test + Paper Update

**Date**: 2026-03-07
**Focus**: Cross-model generalization and paper refinement

## BD3LM Results (Negative)
- ReMask-Retry does NOT work on BD3LM (block diffusion)
- BD3LM baseline PPL is 14.3 (vs 3.8 for MDLM) — much worse base model
- Remasking degrades BD3LM output — block structure disrupted by post-hoc remasking
- Added to paper as honest limitation

## Paper Updates
- Added BD3LM negative result to limitations section
- Clarified method specificity: MDLM-style denoising, not all DLMs
- Updated conclusion to note architecture-dependent effectiveness

## Extended Remask Sweep Results
- 70% remask: -16.2% PPL (best, p < 0.00001)
- 60% remask: -15.5% PPL
- 80% remask: catastrophic failure (too few anchors)
- Monotonic scaling from 15% to 70%

## Cumulative Research Summary
| Iteration | Focus | Key Finding |
|-----------|-------|-------------|
| 1 | Sibyl pipeline upgrade | v3 with supervisor/critic/reflection |
| 2 | TTT-DLM v3 sweep | Best config on 8 samples (misleading) |
| 3 | TTT safeguards | Catastrophic failures, block-reset discovery |
| 4 | Statistical test | TTT NOT significant (p=0.88) |
| 5 | ACA pivot | ReMask-Retry -7.9% (p=0.005) |
| 6 | ACA sweep | -11.7% with 40% remask (p<0.0001) |
| 7 | Extended sweep + paper | -16.2% with 70% remask, paper draft |
| 8 | BD3LM test | Doesn't generalize to block diffusion |

## Next Steps
- Test on downstream tasks (task #15, pending)
- Increase sample size to 64+ for tighter confidence intervals
- Investigate why BD3LM fails (block structure analysis)
- Polish paper for submission

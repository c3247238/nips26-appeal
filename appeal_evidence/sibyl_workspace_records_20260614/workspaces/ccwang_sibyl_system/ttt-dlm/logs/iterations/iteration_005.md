# Iteration 005 - Pivot to ACA-DLM: First Significant Result

**Date**: 2026-03-07
**Focus**: Adaptive Compute Allocation as alternative to TTT

## What was done
- Pivoted from TTT (parameter adaptation) to ACA (process adaptation)
- Implemented two approaches:
  1. ReMask-Retry: Re-mask low-confidence tokens and re-denoise
  2. Iterative Refinement: Multiple rounds of assessment and correction
- Ran with 3 seeds × 32 samples for statistical significance

## KEY RESULT: Statistically Significant Improvement
- **retry_2x_30pct: PPL 3.518, -7.9% vs vanilla, p=0.005**
- All ratio-based ReMask-Retry configs achieve p < 0.05
- Only ~25% compute overhead (103s vs 81s)

## Research Journey Summary
1. V2-V3: TTT shows -15% to -22% → exciting but single-seed
2. V4: 32 samples → TTT doesn't replicate, per-sample analysis shows catastrophic failures
3. V5: Safeguards (KL, grad, EMA) → don't help, wrong failure mode
4. V6: Block-reset/Prompt-TTT → single-seed shows -19%, promising
5. V7: Block-reset sweep → -4.5% to -7.3%, still noisy
6. V8: Statistical test → TTT not significant (p=0.88)
7. **ACA v1: ReMask-Retry → SIGNIFICANT -7.9% (p=0.005)**

## Sibyl Pipeline Lessons
1. Statistical significance testing is essential — without it we'd have published false results
2. The Sibyl idea→experiment→analyze→pivot loop worked: identified dead end, pivoted successfully
3. Simple approaches (re-mask and retry) can outperform complex ones (TTT parameter adaptation)
4. Always use multiple seeds for stochastic generation methods

## Next Iteration
- Fine-grained sweep of ReMask-Retry hyperparameters
- Combine with TTT to see if effects are additive
- Test on downstream tasks and different models
- Begin paper draft

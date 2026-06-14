# Iteration 006 - ACA-DLM v2: Comprehensive Sweep Confirms Significance

**Date**: 2026-03-07
**Focus**: Fine-grained ReMask-Retry sweep + TTT comparison

## Results
- **r_40pct: PPL 3.372, -11.7% vs vanilla, p < 0.0001**
- All ReMask-Retry configs significant at p < 0.001
- TTT block-reset confirmed NOT significant (p=0.88)
- More remasking = monotonically better (15% → 40%)
- More retries = diminishing returns (1x gets 75% of benefit)
- Even 8 refine steps enough for -5.5% improvement

## Key Insight
The improvement scales monotonically with remask ratio up to 40%.
This suggests the optimal might be even higher. Worth testing 50%+.

## Sibyl Pipeline Observations
1. The idea→experiment→pivot→validate loop worked excellently
2. Multi-seed testing prevented false positives (TTT) and confirmed true positives (ACA)
3. Incremental saving prevented data loss during long experiments
4. The 2-phase eval pattern is now well-established

## Next Iteration
- Test higher remask ratios (50%, 60%, 70%)
- Test on different model (bd3lm)
- Begin paper outline

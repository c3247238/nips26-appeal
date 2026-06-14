# Iteration 004 - Statistical Significance Test & Honest Assessment

**Date**: 2026-03-07
**Focus**: Validating block-reset TTT improvement with proper statistics

## What was done
- Ran 5 methods × 3 seeds × 32 samples = 480 generations + evaluations
- Paired t-tests both at seed level (n=3) and per-sample level (n=96)

## Result: NO SIGNIFICANT IMPROVEMENT
- Best config (br_lr3e4): -0.8% PPL improvement, p=0.88 (NS)
- All earlier improvements (-19% to -22%) were within seed variance
- DLM stochastic sampling has high between-seed variance

## Research Lessons
1. **Always test with multiple seeds** — single-seed DLM results are unreliable
2. **Small sample sizes (8) are especially misleading** for stochastic generation
3. **Report confidence intervals, not point estimates**
4. **Negative results are valuable** — saves others from pursuing this dead end

## Sibyl Pipeline Lessons
1. Statistical significance testing should be built into the experiment pipeline
2. Need automated seed-based replication as a standard step
3. The idea→experiment→analyze loop worked well: we went from overconfident
   claims to honest assessment in 4 iterations

## Next Direction Options
1. Different metric (task accuracy, MAUVE) — PPL may miss what TTT changes
2. Different model (LLaDA-8B) — larger model may benefit more from TTT
3. Different TTT objective — not token reconstruction but coherence
4. Adaptive noise schedule — modify denoising process itself
5. Abandon TTT-DLM direction entirely — pivot to something more promising

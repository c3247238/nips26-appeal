# Iteration 003 - From Catastrophic Failures to Block-Reset TTT

**Date**: 2026-03-07
**Focus**: Understanding and fixing TTT failure modes

## Experiment Progression

### V4: Scaled validation (32 samples)
- V3's -22% improvement on 8 samples does NOT replicate at 32 samples
- Per-sample analysis: TTT wins 19/32, but catastrophic failures (+242%) on 12/32
- Median improves -12.3%, but mean worsens due to extreme outliers
- **Lesson**: Always validate on larger sample sizes; small samples are misleading

### V5: Safeguarded TTT
- Tested KL-divergence guard, gradient norm gating, EMA blending
- **None helped**: 0 rollbacks triggered across all configs
- The issue isn't large distribution shifts but cumulative small errors
- **Lesson**: The root cause is adapting to a moving target (sequence changes during denoising)

### V6: Novel approaches
- **Block-Reset TTT**: PPL 3.286, **-19.1%** vs vanilla (4.062) — BEST RESULT
  - Lowest variance too (std 1.346 vs 1.939)
  - Key idea: reset params at each block boundary, fresh TTT per block
- Prompt-TTT (warmup only): -5.2% to -6.8%, cheap and safe
- Stable-TTT (committed tokens only): -4.8%, modest improvement
- Combined prompt+stable: -3.2%, worse than either alone

## Key Discovery
Block-reset TTT works because:
1. Prevents parameter drift across the entire denoising process
2. Each block gets fresh adaptation tailored to its local context
3. Errors don't compound — they're reset at each boundary

## Pipeline Improvements Applied
- Consistent 2-phase evaluation across all experiments
- Per-sample analysis reveals hidden failure modes
- Incremental saving prevents data loss
- Progressive research: hypothesis → test → analyze → revise

## Next Steps
- Deep analysis of block-reset (per-sample, different block sizes)
- Combine block-reset with prompt-TTT warmup
- Statistical significance testing

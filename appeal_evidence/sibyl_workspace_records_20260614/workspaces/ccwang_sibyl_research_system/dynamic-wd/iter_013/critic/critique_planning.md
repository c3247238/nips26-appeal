# Planning Critique

## Executive Summary

The methodology was well-designed with appropriate go/no-go gates, but execution deviated from plan in several important ways.

## Plan vs. Execution Gaps

| Planned | Executed | Impact |
|---------|----------|--------|
| Phase 4: 50 Optuna trials | 15 trials | Underpowered budget equivalence test |
| Budget equiv at 200 epochs | 100 epochs | Different optimal HPs than main experiments |
| ImageNet 90 epochs (standard) | 45 epochs | Non-standard regime |
| 4 control experiments | 3 executed, all appear corrupted | No valid control data |
| ResNet-101 on ImageNet | Not attempted | Reduced architecture coverage |
| AdamW experiments | Not attempted | Missing dominant optimizer |
| Layer-type ablation on CWD | Not explicitly reported | Incomplete ablation matrix |

## What Went Well

1. **Phase 1-3 executed fully**: Pilot, CIFAR core, and ImageNet experiments completed
2. **Multi-seed discipline**: 3 seeds used consistently across experiments
3. **AIS diagnostic**: Pre-experiment alignment informativeness test was completed as planned
4. **Risk mitigation**: ImageNet failure diagnosis protocol worked (sanity check, batch size fallback)

## What Went Wrong

1. **Budget equivalence truncated**: 15 trials instead of 50, 100 epochs instead of 200. This is the experiment that would have made or broken the paper.
2. **Control experiments corrupted**: The most important controls for establishing EqWD's mechanism appear to have not been properly implemented.
3. **No 90-epoch validation**: Even a single-seed 90-epoch run would have addressed the training length concern.
4. **No ResNet-101**: Architecture coverage is limited to ResNet-50 on ImageNet.
5. **No AdamW**: The paper's practical scope is severely limited without Adam-family validation.

## Resource Utilization

- 8x RTX PRO 6000 Blackwell (98GB each) available
- Estimated total from methodology: ~120 GPU-hours
- Actually used: ~100 GPU-hours (rough estimate from run logs)
- Remaining budget for missing experiments: ~40-60 GPU-hours needed for:
  - FixedWD at higher lambda on ImageNet: ~15 GPU-hours
  - 90-epoch single-seed validation: ~12 GPU-hours
  - AdamW CIFAR experiments: ~5 GPU-hours
  - Re-run corrupted controls: ~10 GPU-hours

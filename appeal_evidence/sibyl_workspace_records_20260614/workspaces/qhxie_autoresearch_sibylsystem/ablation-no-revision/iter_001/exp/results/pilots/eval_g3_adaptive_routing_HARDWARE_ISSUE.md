# G3 Adaptive Routing Evaluation - Dependency Failed

## Task: eval_g3_adaptive_routing

**Status**: BLOCKED - Dependency `train_g2_calibration` failed

## Dependency Status

| Dependency | Status | Details |
|------------|--------|---------|
| train_g2_calibration | FAILED | RTX PRO 6000 Blackwell (sm_120) incompatible with PyTorch 2.6.0 |
| eval_g0_baseline | COMPLETED | Accuracy 26%, ECE 0.313, H1 not passed |

## Root Cause

The RTX PRO 6000 Blackwell GPU has compute capability sm_120, which is NOT yet supported by PyTorch 2.6.0+cu124. PyTorch supports up to sm_90 (H100).

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## Pilot Pass Criteria (Cannot Evaluate)

- [ ] Routing accuracy >= G0 accuracy - 0.02 (G0: 26%)
- [ ] Token cost < 0.80 * G0 token cost

## Alternative Approaches

Since G2 calibration training cannot complete, consider:

1. **Base model evaluation**: Evaluate adaptive routing on DeepSeek-Math-7B-Instruct without calibration training
2. **Demonstrate potential**: Show that even without calibration, adaptive routing can reduce tokens
3. **Hardware migration**: Wait for PyTorch update supporting sm_120, or use supported GPU

## Recommendation

Since H1 (DeepSeek-Math-7B baseline >=40%) already failed, and G2/G3 cannot proceed due to hardware incompatibility, recommend:

1. Acknowledge DeepSeek-Math-7B underperformance (26% vs expected 50%+)
2. Investigate evaluation methodology (prompt format, sampling parameters)
3. Consider alternative base models or hardware configuration

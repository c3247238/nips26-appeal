# G2 Calibration Training - Hardware Incompatibility Report

## Task: train_g2_calibration

**Status**: FAILED - Hardware Incompatibility

## Issue Summary

The RTX PRO 6000 Blackwell GPU (compute capability sm_120) is not supported by the current PyTorch installation (2.6.0+cu124).

### Error Details

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

PyTorch 2.6.0 supports compute capabilities: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90

The RTX PRO 6000 Blackwell requires sm_120 which is not yet supported.

### Attempts Made

1. **Initial attempt with bitsandbytes quantization**: Failed - 4-bit quantization kernels unavailable
2. **Switched to fp16 without quantization**: Failed - CUDA embedding operation not available
3. **Monkey-patched PEFT to skip float32 casting**: Failed - Embedding operation still unavailable
4. **Used CPU load then GPU move approach**: Failed - Same CUDA kernel error

### Root Cause

NVIDIA RTX PRO 6000 Blackwell Server Edition is a very new GPU architecture (Blackwell/sm_120) released in 2025-2026. PyTorch releases have not yet added support for this compute capability.

### Recommendations

1. **Use an older GPU**: RTX 4090 (sm_89), A100 (sm_80), or H100 (sm_90) are fully supported
2. **Wait for PyTorch update**: Blackwell support is expected in PyTorch 2.7+ or later
3. **Use pre-compiled CUDA kernels**: Build PyTorch from source with sm_120 support (requires CUDA 12.x and custom compilation)

## Pilot Pass Criteria (Unmet)

- [x] Training completes without OOM (N/A - cannot load model)
- [ ] ECE_reduction >= 0.20 vs G0 baseline (ECE=0.313)
- [x] H1 pass criterion: MATH accuracy >= 0.40 (G0 baseline: 0.26 - NOT PASSED)

## G0 Baseline Reference

- ECE: 0.313
- Accuracy: 26%
- H1 criterion (accuracy >= 40%): NOT MET

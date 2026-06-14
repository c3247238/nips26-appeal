# ImageNet ResNet-50 Experiment Results
**Date**: 2026-03-25 05:31
## Configuration
- **Dataset**: ImageNet-1K (1000 classes)
- **Architecture**: ResNet-50 (25.6M parameters)
- **Epochs**: 45
- **Batch size**: 256
- **Learning rate**: 0.1 (cosine schedule)
- **AMP**: Enabled
- **Seeds**: 42, 123, 456

## Results Summary

| Method | Type | seed42 | seed123 | seed456 | Mean ± Std |
|--------|------|--------|---------|---------|------------|
| NoWD | Baseline | 70.064 | 70.282 | 69.986 | 70.111 ± 0.153 |
| FixedWD | Baseline | 71.834 | 72.150 | 71.690 | 71.891 ± 0.235 |
| SWD | NeurIPS 2023 | 72.324 | 72.224 | 71.584 | 72.044 ± 0.401 |
| CWD | ICLR 2026 | 71.666 | 71.472 | 71.034 | 71.391 ± 0.324 |
| CPR | NeurIPS 2024 | 71.978 | 71.018 | 71.150 | 71.382 ± 0.520 |
| CAWD | Ours (variant) | 71.588 | 71.450 | 71.280 | 71.439 ± 0.154 |
| EqWD | **Ours** | 72.456 | 72.064 | 72.294 | 72.271 ± 0.197 ** |

## Key Findings

1. **Best method**: EqWD with 72.271% (Δ = +0.380% vs FixedWD)
2. **Most stable**: NoWD (lowest std = 0.153)
3. NoWD: 70.111% (Δ = -1.781% vs FixedWD, std = 0.153)
3. FixedWD: 71.891% (Δ = +0.000% vs FixedWD, std = 0.235)
3. SWD: 72.044% (Δ = +0.153% vs FixedWD, std = 0.401)
3. CWD: 71.391% (Δ = -0.501% vs FixedWD, std = 0.324)
3. CPR: 71.382% (Δ = -0.509% vs FixedWD, std = 0.520)
3. CAWD: 71.439% (Δ = -0.452% vs FixedWD, std = 0.154)
3. EqWD: 72.271% (Δ = +0.380% vs FixedWD, std = 0.197)

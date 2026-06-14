# Pilot Summary: G0 Baseline (Single-pass Evaluation)

## Task: g0_qwen_baseline

**Status**: PASS
**Date**: 2026-04-28

## Configuration
- **Model**: Qwen/Qwen2.5-Math-7B-Instruct
- **Dataset**: MATH test set (pilot: 100 samples)
- **Seed**: 42
- **Temperature**: 0.7
- **Max tokens**: 1024
- **GPU**: RTX PRO 6000 Blackwell (GPU 2)

## Results

### Overall Accuracy
- **Accuracy: 47.0%** (47/100 correct)
- **PASS**: Exceeds 40% threshold

### Per-Level Accuracy
| Level | Accuracy | Correct/Total |
|-------|----------|--------------|
| Level 1 | 100.0% | 6/6 |
| Level 2 | 70.6% | 12/17 |
| Level 3 | 55.0% | 11/20 |
| Level 4 | 44.4% | 12/27 |
| Level 5 | 20.0% | 6/30 |

### Performance Metrics
- Average tokens per response: 437.3
- Elapsed time: 1110.8s (~18.5 min)
- Throughput: 0.09 problems/s

## Analysis

1. **Clear difficulty progression**: Accuracy decreases monotonically from Level 1 (100%) to Level 5 (20%), confirming that MATH difficulty levels correlate with problem difficulty.

2. **Activation Energy Theory validation**: The strong correlation between difficulty level and single-pass accuracy supports the hypothesis that harder problems require more sampling (higher activation energy).

3. **Baseline established**: 47% accuracy on pilot provides a reasonable baseline for the Arrhenius saturation curve experiments.

## Recommendation
- **GO**: Proceed to G1 (saturation curve experiment with k=1,2,4,8,16)
- The baseline confirms the model can solve ~50% of problems in single pass
- Lower difficulty problems (L1-L3) show >50% accuracy, suitable for routing experiments

## Next Steps
1. Run G1 saturation curve experiment (k=1,2,4,8,16) to validate Arrhenius kinetics
2. Estimate per-problem activation energy (Ea) from consistency signals
3. Test routing strategy on low-Ea (easy) problems

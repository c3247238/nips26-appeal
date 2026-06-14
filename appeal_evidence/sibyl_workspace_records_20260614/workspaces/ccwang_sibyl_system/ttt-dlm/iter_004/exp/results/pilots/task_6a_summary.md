# Task 6a Pilot Summary: Inference-Time Scaling Curve (16 samples)

## Experiment Design

Scanned denoising steps T in {64, 128, 256, 512} for 4 methods on 16 Countdown problems (seed=42):
- Vanilla (standard Dream origin denoising)
- ReMDM-conf (confidence-based remasking)
- DTA (denoising-time adaptation with LoRA)
- DTA+ReMDM (combined)

Each step count ran on a separate GPU (4x parallel on RTX PRO 6000 Blackwell).

## Results: Accuracy

| T    | Vanilla | ReMDM-conf | DTA   | DTA+ReMDM |
|------|---------|------------|-------|-----------|
| 64   | 6.2%    | 6.2%       | 6.2%  | 0.0%      |
| 128  | 12.5%   | 6.2%       | 0.0%  | 12.5%     |
| 256  | 0.0%    | 0.0%       | 6.2%  | 0.0%      |
| 512  | 6.2%    | 6.2%       | 0.0%  | 12.5%     |

## Results: Time per Sample (seconds)

| T    | Vanilla | ReMDM-conf | DTA   | DTA+ReMDM |
|------|---------|------------|-------|-----------|
| 64   | 1.9s    | 3.3s       | 7.7s  | 9.1s      |
| 128  | 3.7s    | 6.5s       | 15.3s | 18.1s     |
| 256  | 7.4s    | 13.0s      | 30.4s | 36.0s     |
| 512  | 14.7s   | 23.8s      | 60.9s | 57.6s     |

## Pass Criteria

- **4 step counts all ran**: PASS
- **All 4 methods ran per step**: PASS
- **H3 (DTA still improving at T=256, T=512)**: INCONCLUSIVE
- **Overall**: CONDITIONAL-GO

## Key Observations

1. **16 samples too small for scaling analysis**: With only 16 problems, 1 correct answer = 6.2%. The noise floor is far too high to observe meaningful scaling trends. All accuracy differences are within random variation.

2. **Time scaling is linear and as expected**: All methods show approximately linear time scaling with T (vanilla: 7.9x from T=64 to T=512, DTA: 7.9x). This confirms the computational overhead is proportional to step count.

3. **DTA+ReMDM at T=128 and T=512 tied best at 12.5%**: Matches vanilla T=128, suggesting the combination may have a slight advantage at higher step counts. But the sample size precludes any confidence.

4. **ReMDM-conf shows no clear saturation pattern**: With so few samples, the expected saturation behavior (H3 predicts ReMDM saturates at T>2L) cannot be verified.

5. **DTA computational overhead**: DTA costs ~4x vanilla per step (7.7s vs 1.9s at T=64, 60.9s vs 14.7s at T=512), primarily from the backward pass for LoRA updates. At T=512, DTA+ReMDM is actually slightly faster than DTA alone (57.6s vs 60.9s) due to remasking reducing the number of revealed tokens for LoRA updates.

## Conclusion

The pilot successfully verified that all 4 methods run correctly at all 4 step counts with no crashes or numerical issues. The scaling curve infrastructure works. However, 16 samples is fundamentally insufficient to test H3 (scaling behavior). **The full-scale experiment (200 samples) is required** to draw any conclusions about inference-time scaling behavior.

## Wall Clock Time

- T=64: 6.0 min
- T=128: 11.7 min
- T=256: 23.2 min
- T=512: 41.9 min
- Total (parallel across 4 GPUs): ~42 min wall clock

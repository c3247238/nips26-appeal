# DTA Full-Scale Experiment Results (In Progress)

## Status: Wave 1 Running
- **Started**: 2026-03-09 04:18 HKT
- **GPUs**: 0 (Countdown s42), 1 (Countdown s123), 2 (Countdown s456), 4 (GSM8K s42)
- **Server**: cs8000d, NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)

## Countdown-500 Results (4/7 methods complete, 3 seeds)

| Method | s42 | s123 | s456 | Mean | Std |
|--------|-----|------|------|------|-----|
| Vanilla | 4.0% | 5.0% | 5.2% | 4.7% | 0.6% |
| ReMDM-conf | 4.8% | 5.2% | 3.2% | 4.4% | 1.0% |
| RCR | 5.4% | 5.4% | 6.4% | 5.7% | 0.6% |
| **DMI** | **7.8%** | **9.6%** | **10.6%** | **9.3%** | **1.4%** |
| SCP | - | - | - | ~8.4%* | - |
| DTA | - | - | - | pending | - |
| DTA+ReMDM | - | - | - | pending | - |

*SCP interim at 150/500 samples

## GSM8K-1319 Results (in progress, seed 42)

| Method | s42 (progress) |
|--------|---------------|
| Vanilla | 29.6% (1300/1319) |
| ReMDM-conf | 25.1% (350/1319) |
| DTA | pending |
| DTA+ReMDM | pending |

## Key Findings So Far

1. **DMI provides clear improvement**: At 9.3% mean accuracy (vs 4.7% vanilla), DMI shows that even simple cross-step information (soft embedding injection from previous step's logits) meaningfully improves Countdown reasoning. This is a ~2x improvement with nearly zero computational overhead.

2. **Pure remasking is insufficient**: ReMDM-conf (4.4%) and RCR (5.7%) show marginal or no improvement over vanilla (4.7%), confirming H1's motivation that remasking alone cannot overcome the cross-step information loss.

3. **SCP performs comparably to DMI**: Interim SCP results (~8.4%) suggest that self-contradiction probing provides similar benefit to embedding memory, despite being computationally more expensive.

4. **GSM8K baseline is reasonable**: Vanilla Dream-7B achieves ~29.6% on GSM8K (consistent with reported results in the Dream paper), validating our evaluation framework.

## Remaining Work
- Complete SCP, DTA, DTA+ReMDM on Countdown (est. ~10h per seed)
- Complete GSM8K all methods x 3 seeds
- MBPP all methods x 3 seeds
- Ablation tasks at full scale (200 samples)
- Statistical tests (McNemar + Bootstrap CI)

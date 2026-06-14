# Dream-7B-Instruct Baseline Pilot Summary

## Verdict: GO

## Benchmarks (seed=42, steps=64)

| Benchmark | Accuracy | TPS |
|-----------|----------|-----|
| GSM8K (100 samples) | 0.390 | 72.3 |
| MATH500 (100 samples) | 0.150 | 147.1 |

## Cross-Model Comparison

| Model | GSM8K Acc | GSM8K TPS | MATH500 Acc |
|-------|-----------|-----------|-------------|
| Dream-7B | 0.39 | 72.29228261811322 | 0.15 |
| LLaDA-8B | 0.712 | 31.0 | 0.111 |

## Notes
- Model repo: Dream-org/Dream-v0-Instruct-7B (corrected from iter_001 wrong name)
- Dream mask_token_id=151666
- Elapsed: 11.9 min

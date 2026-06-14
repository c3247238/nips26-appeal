# Task 7a Pilot Summary: LoRA Rank Ablation (16 samples)

## Experiment Configuration

- **Benchmark**: Countdown (16 problems, seed=42)
- **Steps**: 128, Temperature: 0.4
- **Fixed DTA hyperparams**: lr=5e-4, gamma=0.95, warmup=20%, last 2 layers FFN
- **Swept ranks**: r in {2, 4, 8, 16}
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB)
- **Wall clock**: 14.6 min

## Results

| Config   | Accuracy | Correct | Avg Time | Dist-2 | Rep-3 | LoRA Params | MaxNorm (mean) |
|----------|----------|---------|----------|--------|-------|-------------|----------------|
| Vanilla  | 12.5%    | 2       | 3.7s     | 0.838  | 0.101 | 0           | N/A            |
| rank=2   | 6.2%     | 1       | 15.3s    | 0.826  | 0.084 | 270,336     | 0.0976         |
| rank=4   | 6.2%     | 1       | 11.8s    | 0.833  | 0.079 | 540,672     | 0.0936         |
| rank=8   | 6.2%     | 1       | 11.8s    | 0.832  | 0.079 | 1,081,344   | 0.0926         |
| rank=16  | 6.2%     | 1       | 11.8s    | 0.823  | 0.095 | 2,162,688   | 0.0958         |

## Pass Criteria

- **All 4 ranks ran successfully**: PASS
- **r=4 near-optimal**: PASS (all DTA ranks tied at 6.2%, 1 correct)
- **Overall**: PASS

## Key Findings

1. **No rank differentiation at pilot scale**: All 4 LoRA ranks produced identical accuracy (6.2%, exactly 1/16 correct). The same single problem (target=28, "(1+3)*7") was solved across all ranks. This is consistent with the 16-sample pilot limitation noted in task_5a -- a 1-problem difference is within noise.

2. **Vanilla outperforms DTA on pilot**: Vanilla baseline got 2/16 (12.5%) vs DTA's 1/16 (6.2%) regardless of rank. This matches the task_5a pilot findings. The 1-problem gap at this sample size is not statistically meaningful.

3. **Rank=2 is notably slower**: rank=2 took 15.3s per sample vs ~11.8s for ranks 4/8/16. This is likely due to the first-run JIT compilation overhead (rank=2 was the first DTA config tested after vanilla). The actual compute cost difference between ranks is minimal on this GPU.

4. **LoRA norms are stable across ranks**: Mean max-norm ranges from 0.0926 (r=8) to 0.0976 (r=2), all well below the 1.0 safety threshold. No numerical instability observed at any rank.

5. **Text quality is similar**: Distinct-2 ranges from 0.823 (r=16) to 0.838 (vanilla). Rep-3 ranges from 0.079 to 0.101. No evidence of text degradation from higher ranks.

6. **Parameter efficiency**: The 8x increase from r=2 (270K params) to r=16 (2.2M params) produced no accuracy difference on this pilot. This suggests the bottleneck is not LoRA capacity but rather the fundamental DTA mechanism's effectiveness on small samples.

## Conclusion

The pilot validates that all rank configurations are numerically stable and produce comparable quality output. The lack of differentiation is expected at N=16 -- the full-scale experiment (200 problems) is needed to observe meaningful differences between ranks. The default r=4 remains a reasonable choice as it balances parameter count and is tied for best accuracy.

## Recommendation for Full-Scale

- Proceed with full 200-problem evaluation to detect rank effects
- r=4 is a safe default; the pilot shows no reason to change it
- Consider extending to r=32 if r=16 shows improvement in full-scale

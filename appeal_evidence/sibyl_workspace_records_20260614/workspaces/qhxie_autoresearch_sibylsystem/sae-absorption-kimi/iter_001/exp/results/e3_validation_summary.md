# E3 Validation Summary: Task-Agnostic Absorption Metric vs. First-Letter Benchmark

## Overview

This experiment computed both the task-agnostic absorption metric (geography hierarchy probe) and the simplified first-letter absorption benchmark across 10 GPT-2 Small SAE checkpoints.

## Key Results

- **Checkpoints evaluated:** 10/10 successfully loaded and measured
- **Pearson correlation (r):** -0.592
- **Spearman correlation (ρ):** -0.529
- **p-value:** 0.116 (not statistically significant at α = 0.05)

## Per-Checkpoint Results

| Release | SAE ID | Family | Task-Agnostic | First-Letter |
|---------|--------|--------|---------------|--------------|
| gpt2-small-res-jb | blocks.0.hook_resid_pre | Standard | 0.073 | 0.000 |
| gpt2-small-res-jb | blocks.4.hook_resid_pre | Standard | 0.160 | 0.000 |
| gpt2-small-res-jb | blocks.8.hook_resid_pre | Standard | 0.240 | 0.000 |
| gpt2-small-res-jb | blocks.11.hook_resid_pre | Standard | 0.140 | 0.000 |
| gpt2-small-resid-post-v5-32k | blocks.4.hook_resid_post | TopK | 0.140 | 0.000 |
| gpt2-small-resid-post-v5-32k | blocks.8.hook_resid_post | TopK | 0.167 | 0.000 |
| gpt2-small-resid-post-v5-128k | blocks.4.hook_resid_post | TopK | 0.167 | 0.000 |
| gpt2-small-resid-post-v5-128k | blocks.8.hook_resid_post | TopK | 0.167 | 0.000 |
| gpt2-small-mlp-out-v5-32k | blocks.8.hook_mlp_out | TopK_MLP | 0.013 | 0.000 |
| gpt2-small-attn-out-v5-32k | blocks.8.hook_attn_out | TopK_Attn | 0.000 | 0.654 |

## Interpretation

1. **Negative correlation:** The task-agnostic metric and first-letter proxy are negatively correlated across this checkpoint set. This suggests they may be capturing different phenomena rather than a shared underlying "absorption" construct.

2. **First-letter metric is degenerate on most checkpoints:** 9 out of 10 checkpoints show zero first-letter absorption. This is consistent with the E1 full GPT-2 results, where only the attention-output SAE showed non-zero first-letter absorption.

3. **Task-agnostic metric shows meaningful variance:** The geography-hierarchy probe produces a spread from 0.0 to 0.24, indicating it is sensitive to architectural and layer differences even when the first-letter proxy is not.

4. **Single dominant outlier:** The attention-output TopK SAE (`blocks.8.hook_attn_out`) drives much of the negative correlation because it has high first-letter absorption (0.654) but zero task-agnostic absorption.

## Limitations

- **Small sample size:** Only 10 checkpoints, all on GPT-2 Small.
- **Simplified first-letter proxy:** This is a coarse approximation of the full Chanin et al. spelling-task metric.
- **No cross-model validation:** Gemma-2-2B is gated and inaccessible, so we cannot test whether correlations improve on modern models.
- **Single hierarchy domain:** Only geography parent-child pairs were used.

## Conclusion

The pilot does **not** support H3 (moderate-to-strong positive correlation, r > 0.4). Instead, it reveals a **weak negative correlation** that is not statistically significant. This is a valuable negative result: it suggests the first-letter benchmark may be unrepresentative of general absorption behavior, or that different SAE architectures absorb different kinds of hierarchical structure in qualitatively different ways.

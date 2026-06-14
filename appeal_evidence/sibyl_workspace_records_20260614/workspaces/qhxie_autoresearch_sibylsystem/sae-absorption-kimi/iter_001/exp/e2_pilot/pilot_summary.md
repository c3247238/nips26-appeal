# E2 Pilot Summary

**Task:** e2_pilot
**Total Time:** 6.0 min

## Results

| Checkpoint | Family | Absorption (Full) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|
| gpt2-small-res-jb/blocks.8.hook_resid_pre | Standard | 0.4371 | 66.6 | 0.0017 | 0.9887 |
| gpt2-small-resid-post-v5-32k/blocks.8.hook_resid_post | TopK | 0.1912 | 32.0 | 0.0951 | 0.9804 |
| gpt2-small-attn-out-v5-32k/blocks.8.hook_attn_out | TopK_Attn | 0.0500 | 32.0 | 0.1183 | 0.7501 |
| sae_bench_pythia70m_sweep_standard_ctx128_0712/blocks.3.hook_resid_post__trainer_0 | Standard | 0.0201 | 596.0 | 0.6743 | 0.9999 |
| sae_bench_pythia70m_sweep_topk_ctx128_0730/blocks.3.hook_resid_post__trainer_0 | TopK | 0.2823 | 20.0 | 0.0371 | 0.9421 |

## Pass Criteria

- All 5 checkpoints loaded successfully: YES
- Official absorption metric returned finite values: YES
- Dead-neuron rate computed on >=50k tokens: YES

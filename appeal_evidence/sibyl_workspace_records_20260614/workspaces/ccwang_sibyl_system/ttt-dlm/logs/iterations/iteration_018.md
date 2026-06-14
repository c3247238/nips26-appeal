# Iteration 018 - New Research Direction: TCR for Dream-7B

**Date**: 2026-03-07
**Focus**: Pivoting to a new approach — Trajectory-Consistent Remasking (TCR) on Dream-7B

## Literature Survey (New Findings)

The DLM inference landscape has evolved significantly since our ReMask-Retry work:

1. **ReMDM** (arxiv 2503.00307, Mar 2025): Principled remasking sampler with inference-time scaling
2. **PRISM** (arxiv 2510.01384, Oct 2025): Learned per-token quality scores via lightweight adapter
3. **Soft-Masked Diffusion** (arxiv 2510.17206, Oct 2025): Soft blending instead of binary masking
4. **Self-Rewarding SMC** (arxiv 2602.01849, Feb 2026): Parallel particle trajectories with resampling
5. **CoRe** (arxiv 2602.04096, Feb 2026): Context-robust remasking via sensitivity probing
6. **ProSeCo** (arxiv 2602.11590, Feb 2026): Progressive self-correction training
7. **Dream 7B** (arxiv 2508.15487, Aug 2025): Most powerful open DLM

## New Approach: Trajectory-Consistent Remasking (TCR)

**Motivation**: Our ReMask-Retry used confidence (point estimate) to decide which tokens to remask. We proved this games PPL via repetition. TCR instead uses trajectory consistency — how stable predictions are across denoising steps — as a more robust signal.

**Algorithm**:
1. Standard denoising with trajectory recording (via Dream's logits hook)
2. Compute per-position stability (fraction of steps with same top prediction)
3. Remask least-stable positions
4. Re-denoise remasked positions

## TCR v1 Results (Dream-7B, 32 prompts, 128 gen tokens, 128 steps)

| Method | PPL(med) | PPL(mean) | Diversity | Degen | Time |
|--------|---------|-----------|-----------|-------|------|
| vanilla_origin | **14.708** | 30.312 | 0.961 | 0 | 2.5s |
| vanilla_maskgit | 54.281 | 369.260 | 1.000 | 0 | 2.5s |
| remdm_p10 | 84.382 | 195.471 | 1.000 | 0 | 2.5s |
| tcr_r30 | 16.880 | **17.948** | **0.970** | 0 | 3.9s |
| tcr_r50 | 15.702 | 18.760 | 0.965 | 0 | 3.9s |

### Key Observations
1. **Dream's `origin` algorithm is strong baseline** — maskgit_plus and random remasking perform terribly
2. **TCR doesn't improve median PPL** but significantly improves mean PPL (17.9 vs 30.3)
3. **No degeneracy at 7B scale** — confirms our previous finding
4. **Diversity maintained/improved** — TCR doesn't cause repetition
5. **Model already very stable** — mean trajectory stability ~0.96, limited room for improvement

### Issues Discovered
- Dream uses `AutoModel` not `AutoModelForCausalLM` (custom architecture)
- `max_length` validation prevents re-denoising full-length sequences — had to use manual forward pass
- `nohup` background processes unreliable for TCR (died silently) — direct execution works
- `torch_dtype` deprecated in favor of `dtype`

## Sibyl Pipeline Improvements (this iteration)
1. **Parallel GPU utilization**: Launched 4 experiments simultaneously on GPUs 0, 2, 4, 5
2. **Dream API integration**: Used native hooks (generation_logits_hook_func, generation_tokens_hook_func)
3. **Background agent pattern**: Launched brainstorming + critic agents in parallel

## Code
- `exp/code/tcr_dream_v1.py`: TCR experiment code for Dream-7B
- Results: `exp/results/tcr_v1/`

## Next Steps
- Analyze why mean PPL improves but median doesn't (outlier investigation)
- Try more methods based on brainstorming agent results
- Consider training-based approaches (PRISM-style adapter)
- Set up LaTeX paper framework

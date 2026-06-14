# Iteration 002 - Hyperparameter Sweep Results

**Date**: 2026-03-07
**Focus**: Comprehensive hyperparameter sweep for TTT-DLM

## What was done
- Fixed CUBLAS error by implementing 2-phase generation/evaluation approach
- Ran complete sweep across 5 dimensions: param type, LR, remask ratio, inner steps, confidence threshold
- 23 configurations tested, 8 samples each, 256 tokens, 128 steps

## Key Findings
1. **conf_0.1 is the best overall** (PPL 2.739, -22% vs vanilla 3.511)
   - Very selective: only 119 TTT steps out of ~800+ denoising steps
   - Also the fastest TTT config (34.6s vs ~55s)
2. **Selectivity > intensity**: Less TTT applied more carefully beats more TTT
3. **MLP-only adaptation is catastrophic** (+49.7%), but q_proj+mlp works
4. **remask_0.1 best** — minimal remasking during TTT
5. **Inner steps don't help** — 1 step sufficient
6. **LR 5e-5 to 1e-3 all work** — robust to LR choice in this range

## Pipeline Improvements
- 2-phase approach: generate all samples first, then load eval model
- Incremental result saving (JSON after each config)
- Error recovery: try/except per config with GPU state restoration
- Fixed `torch_dtype` → `dtype` deprecation warning

## Issues Found
- CUBLAS_STATUS_INVALID_VALUE from corrupted CUDA state (not OOM)
- nohup unreliable for GPU jobs — tmux or direct execution preferred
- Conda path: `/home/ccwang/sibyl_system/miniconda3/` (not ~/miniconda3)

## Next Iteration Plan
- Combined best config experiment with more samples (32-64)
- lm-evaluation-harness benchmarks
- Scale test on larger model

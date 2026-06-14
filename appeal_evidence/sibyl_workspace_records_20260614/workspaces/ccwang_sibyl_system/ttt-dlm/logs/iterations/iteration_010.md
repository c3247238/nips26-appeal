# Iteration 010 - Critic Response + Related Work + Best-of-N Baseline

**Date**: 2026-03-07
**Focus**: Addressing academic reviewer feedback, paper rewrite, best-of-N experiment

## Critic Review Summary
Harsh but fair review received. Key issues:
1. **No ReMDM positioning** — grounds for rejection alone
2. **No best-of-N baseline** — must show ReMask-Retry beats naive compute scaling
3. **Single model (0.6B)** — insufficient for generality claims
4. **32 prompts too few** — need 200+ from established benchmarks
5. **PPL-only eval** — need external judge or downstream tasks
6. **Missing formal method description** — no pseudocode

## Actions Taken

### 1. Related Work Overhaul
Literature search found 10+ highly relevant papers (2025-2026):
- **ReMDM** (Wang et al., 2025a): Remasking within denoising loop — must cite, complementary
- **PRISM** (Kim et al., 2025): Learned self-correction head — requires training, we're training-free
- **ProSeCo** (Schiff et al., 2026): Trained corrector interleaved with unmasking
- **CDLM** (Zhang et al., 2025): Post-training for error correction
- **Informed Correctors** (Zhao & Linderman, 2025): Predictor-corrector paradigm
- **DSL** (2026): SNR-invariant denoiser, complementary to our approach

### 2. Paper Rewrite
Major revision addressing all critic feedback:
- Full related work section with explicit positioning against each method
- Formal method description with algorithm pseudocode and math
- Comparison table (Table 1): training-free vs learned approaches
- Explicit limitations section acknowledging all weaknesses
- References section added
- Phase transition analysis expanded with percolation hypothesis

### 3. Best-of-N Experiment — COMPLETED
Created resumable `aca_dllm_v4b_resume.py` (runs one config at a time, saves incrementally).

**Results (all 5 methods × 3 seeds complete):**

| Method | PPL | Δ% | Compute |
|--------|-----|-----|---------|
| vanilla | 3.820 | — | 1.0x |
| **retry_30pct** | **3.518** | **-7.9%** | **1.28x** |
| **retry_70pct** | **3.203** | **-16.2%** | **1.28x** |
| best_of_2 | 3.840 | +0.5% | 2.0x |
| best_of_3 | 4.086 | +6.9% | 3.0x |

**Key finding**: Best-of-N is completely ineffective — counterproductive at N=3.
- Model confidence (selection criterion) is anti-correlated with AR PPL
- Token-level confidence use (ReMask-Retry) >> sequence-level selection (Best-of-N)
- This is a very strong result for the paper

### 4. Paper Updated
- Best-of-N results integrated into Section 4.2
- Compute-quality tradeoff analysis updated (Section 5.3)
- Abstract updated with best-of-N comparison

## Sibyl Pipeline Improvements
- **Resumable experiment pattern**: `v4b_resume.py` runs one config at a time with incremental saves — robust to SSH disconnections
- **nohup/tmux/screen all unreliable** via SSH MCP — direct execution with long timeout is the only reliable method
- **CUBLAS errors**: GPU 0 had corrupted state, GPU 2 worked fine — always use fresh GPU

## Remaining Critic Issues (Not Yet Addressed)
- [ ] Larger prompt set (200+) from established benchmarks
- [ ] Second model scale (3B+)
- [ ] External PPL (larger model as judge)
- [ ] Token-level analysis (POS tags, positions)
- [ ] Downstream task evaluation

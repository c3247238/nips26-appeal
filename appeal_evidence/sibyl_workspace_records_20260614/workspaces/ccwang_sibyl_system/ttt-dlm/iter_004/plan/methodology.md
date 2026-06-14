# Methodology: Belief-State Diffusion with Reasoning-Aware Guidance for MDLMs

## 1. Overview

This experiment plan tests three core components for improving inference-time reasoning in masked diffusion language models (MDLMs):

1. **Belief-State Diffusion (BSD)**: Replacing hard mask embeddings with continuously evolving belief vectors
2. **Reasoning-Aware Classifier-Free Guidance (RACFG)**: Cross-step stability-guided CFG with temporal scheduling
3. **BSD + RACFG Combination**: Multi-layer framework combining representation-level and prediction-level improvements

The plan is structured in 4 phases across ~58 GPU·hours on 4× GPUs (cs8000d server).

## 2. Model & Environment

- **Primary model**: Dream-7B (Dream-v0-Instruct-7B) — strongest open-source MDLM
- **Fallback model**: LLaDA-8B — proven CFG compatibility (A-CFG paper)
- **Environment**: conda `sibyl_ttt-dlm` on cs8000d, `/home/ccwang/sibyl_system/projects/ttt-dlm/`
- **Framework**: HuggingFace Transformers + custom denoising loop modifications
- **Random seed**: 42 (pilot), {42, 123, 456} (full-scale)

## 3. Baselines

| Baseline | Description | Source | Compute |
|----------|-------------|--------|---------|
| Vanilla | Standard Dream-7B denoising | Built-in | 1x |
| DMI | Diffusion Memory Injection (α=0.3 fixed mixing) | Our iter 3 (9.3% Countdown-500) | ~1.05x |
| A-CFG | Adaptive CFG with confidence-based re-masking | arXiv 2505.20199 | ~2x |
| ReMDM-conf | Confidence-based remasking | arXiv 2503.00307 | ~1.5x |

## 4. Proposed Methods

### 4.1 Belief-State Diffusion (BSD)

**Key parameters to tune**:
- `k` (hard-reveal steps): {T/4, T/2, 3T/4} where T = total denoising steps
- `alpha_schedule`: EMA update rate schedule (linear ramp 0.1→0.8, cosine, constant)
- `tau_schedule`: Temperature annealing for softmax (1.0→0.3)
- `norm_type`: L2 normalization to match mask_emb norm
- `fallback_beta`: DMI-style mixing ratio for graceful degradation (0.9→0.1 linear decay)

**Implementation**: Modify the denoising loop to maintain belief vectors instead of hard mask embeddings. During Phase 1 (steps T to k+1), update beliefs via EMA of probability-weighted embedding mixtures. During Phase 2 (steps k to 1), standard confidence-based unmasking from belief states.

### 4.2 Reasoning-Aware CFG (RACFG)

**Key parameters**:
- `remask_pct` (m%): {5%, 10%, 20%} — fraction of positions to re-mask for unconditional input
- `guidance_weight` (w_base): {0.5, 1.0, 1.5, 2.0}
- `stability_ema_lambda`: 0.7 (EMA smoothing for cross-step logit history)
- `schedule_type`: {fixed, linear_ramp, cosine_ramp, threshold_70_30}
- `w_max`: 2.0 (cap to prevent over-extrapolation)

**Implementation**: Track logit history across denoising steps. Compute JSD-based stability scores per position. Select least-stable positions for re-masking to construct unconditional inputs. Apply position-adaptive, temporally-scheduled CFG.

### 4.3 BSD + RACFG Combination

- Steps T to k+1: BSD continuous belief refinement (no CFG)
- Steps k to 1: Hard reveal with RACFG guidance applied to belief-informed predictions

## 5. Benchmarks

| Benchmark | Size | Purpose | Time per run |
|-----------|------|---------|-------------|
| Countdown-16 | 16 samples | Quick pilot validation | ~2 min |
| Countdown-100 | 100 samples | Ablation studies | ~15 min |
| Countdown-500 | 500 samples | Primary evaluation (3 seeds) | ~45 min |
| GSM8K-1319 | 1319 samples | Generalization test | ~90 min |

**Primary metric**: Benchmark accuracy (NOT perplexity — lesson from 18 iterations)

## 6. Evaluation Protocol

### 6.1 Mandatory Diagnostics (per configuration)

- **Accuracy**: Primary metric on each benchmark
- **rep-2/3**: N-gram repetition rate (alert if > vanilla + 20%)
- **distinct-1/2/3**: Unique n-gram ratio (alert if < vanilla - 15%)
- **Output length distribution**: Mean, std, histogram (alert if variance < vanilla/2)
- **Qualitative samples**: 20 random outputs per configuration for manual inspection
- **Belief entropy trajectory** (BSD only): Per-position entropy at each step

### 6.2 Statistical Analysis

- **McNemar test**: Paired classification test for each method pair
- **Bonferroni correction**: α' = 0.05/N for multiple comparisons
- **Bootstrap 95% CI**: 10,000 resamples for accuracy differences
- **Difficulty-stratified subgroup analysis**: Easy/medium/hard partitions

### 6.3 Compute Fairness

| Method | Estimated FLOPs vs Vanilla | Fair Comparison |
|--------|---------------------------|-----------------|
| BSD | ~1.1x (EMA + normalization) | Vanilla with T×1.1 steps |
| RACFG | ~2x (one extra forward pass) | Vanilla with T×2 steps |
| BSD+RACFG | ~2.1x | Vanilla with T×2.1 steps |
| ReMDM | ~1.5x | Vanilla with T×1.5 steps |

## 7. Decision Gates

### Gate 1 (After Phase 1 Pilots)
- If A-CFG reproduction on Dream-7B < vanilla → switch all CFG experiments to LLaDA-8B
- If BSD pilot shows OOD collapse (rep-3 > 2× vanilla) → activate fallback_beta mixing
- If DMI reproduction fails to match 9.3% → re-examine codebase before proceeding

### Gate 2 (After Phase 2 Ablations)
- Advance best BSD config + best RACFG config to combination (Phase 3)
- If neither BSD nor RACFG > vanilla + 3pp → pivot to analysis-only paper

### Gate 3 (After Phase 3 Combination)
- If combination > 18% Countdown-500 → proceed to GSM8K extension
- If combination < max(BSD, RACFG) + 1pp → report negative combination result

## 8. Expected Visualizations

- **Figure 1**: Architecture diagram — BSD belief evolution + RACFG guidance pipeline
- **Figure 2**: Belief entropy trajectory — per-position entropy across denoising steps (BSD vs vanilla)
- **Table 1**: Main results comparison table (Method × Benchmark × Metric)
- **Figure 3**: Ablation study — bar charts for k, alpha_schedule, remask_pct, schedule_type
- **Figure 4**: Training loss / accuracy curves across denoising steps (if applicable)
- **Figure 5**: Token-level CFG impact visualization — which positions changed by guidance
- **Table 2**: Compute-fair comparison — methods at equal FLOPs
- **Figure 6**: Difficulty-stratified accuracy breakdown (easy/medium/hard)

## 9. Failure Modes and Mitigations

| Failure Mode | Probability | Detection | Mitigation |
|-------------|------------|-----------|------------|
| BSD OOD collapse | 30% | rep-3 > 2× vanilla | Fallback to DMI-style β mixing |
| Belief oscillation | 20% | Non-monotonic entropy | Reduce α, add temperature annealing |
| Dream CFG incompatibility | 15% | A-CFG repro fails | Switch to LLaDA-8B |
| CFG over-extrapolation | 25% | Degraded accuracy at high w | Cap w_max=2.0, temperature recalibration |
| JSD noise | 10% | Unstable stability scores | Increase EMA λ to 0.85 |

## 10. Compute Budget

| Phase | GPU·hours | Timeline |
|-------|----------|----------|
| Phase 1: Pilots | ~4 | Day 1 |
| Phase 2: Ablations | ~30 | Days 2-3 |
| Phase 3: Combination + GSM8K | ~30 | Days 4-5 |
| Phase 4: Analysis | ~6 | Day 6 |
| **Total** | **~70** | **~6 days** |

With 4× GPUs and parallelism: ~15-18 wall-clock hours of compute.

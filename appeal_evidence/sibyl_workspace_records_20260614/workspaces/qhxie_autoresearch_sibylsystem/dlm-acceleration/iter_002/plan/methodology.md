# ComposeAccel -- Iteration 2 Experiment Methodology

## Iteration Focus: Fix Foundation, Strengthen Evidence, Extend Generalization

Iteration 1 completed all primary experiments but the critique (score 5.5/10) identified
critical methodological problems that must be resolved before the study is publishable.
This iteration addresses them systematically in four phases.

**Key problems to resolve:**
1. M1 implementation gap: simplified Python cache achieves 1.38x vs published 15-26x
2. Coding benchmarks uninformative: HumanEval baseline 2.4%, MBPP 0% -- not viable signals
3. QAS formula inconsistency: undisclosed 0.5x penalty distorts cross-method comparison
4. Only 3/6 pairwise combinations measured (M2 dropped)
5. Single model (LLaDA-8B-Instruct only; Dream-7B download failed)
6. Pairwise Ortho on 200/1319 GSM8K (15% scale), 2 seeds only
7. Fabricated Wilcoxon p-value in task dependence analysis
8. CHR_refine=94% claimed as measured but not found in raw data
9. M1+M3 full pairwise contradicts pilot (Ortho=0.301 vs pilot 1.339)

**Design philosophy for iter_002:**
- Fix the metric and reporting first (remove QAS penalty, drop uninformative benchmarks)
- Scale up evidence where it matters (GSM8K full scale, 3 seeds)
- Integrate d2Cache for real M1 speedup measurement
- Add Dream-7B for cross-model validation
- Add MATH500 as second reasoning benchmark for task dependence (replace HumanEval/MBPP)
- Recalculate all Ortho scores under consistent metric

---

## Evaluation Framework Revisions

### Benchmarks (Revised)

| Task | Metric | Subset (Pilot) | Subset (Full) | Rationale |
|------|--------|-----------------|----------------|-----------|
| GSM8K | Exact match | 200 | 1319 (full) | Primary reasoning benchmark; baseline 71.2% -- strong signal |
| MATH500 | Exact match | 100 | 500 | Secondary reasoning; baseline 11.1% -- weaker but informative for harder tasks |
| HumanEval | pass@1 | 50 | 164 | **Kept for completeness only**; baseline 2.4% -- report but do not include in combined metrics |
| MBPP | pass@1 | -- | -- | **DROPPED**: baseline 0.0% provides no signal |

**Combined metric**: Weighted average of GSM8K (weight=0.7) and MATH500 (weight=0.3) for all
QAS, Ortho, and Pareto computations. HumanEval reported separately in appendix.

### QAS Formula (Corrected)

```
QAS = Speedup * Accuracy_Retention
Accuracy_Retention = Acc(method) / Acc(baseline)
```

**No penalty factor.** The 0.5x penalty from iter_001's `merge_igsd_pareto.py` is removed.
All methods use the same formula. Low-quality methods are distinguished by their raw
accuracy retention, not by an undisclosed multiplicative penalty.

### Orthogonality Metric (Unchanged)

```
Ortho(Ma + Mb) = QAS(Ma+Mb) / max(QAS(Ma), QAS(Mb))
```

Interpretation:
- Ortho > 1.0: synergy (combination strictly better than either component)
- Ortho 0.8-1.0: near-orthogonal (composition preserves most benefit)
- Ortho < 0.8: interference (composition degrades)

### Speedup Measurement Protocol

- Metric: wall-clock tokens-per-second (TPS) averaged over stable-state samples
- Discard first 5 warm-up samples per benchmark
- Measure: total output tokens / elapsed wall-clock time (generation only)
- Seeds: pilot = [42]; full = [42, 123, 456], report mean +/- std
- Batch size: 1 (interactive; primary) + 8 (serving; secondary for selected configs)

---

## Phase 1: d2Cache Integration and M1 Real Speedup (Gate Experiment)

### Goal
Determine whether kernel-level KV cache reuse via d2Cache achieves measurable TPS
speedup on LLaDA-8B-Instruct. This is the critical gate: if d2Cache integration fails,
we fall back to theoretical speedup estimation with clear labeling.

### Protocol
1. Integrate d2Cache (`dLLMCache`) from `iter_001/exp/code/d2Cache/src/cache/dllm_cache.py`
   into LLaDA-8B-Instruct inference wrapper
2. Sweep entropy threshold eta = {0.5, 1.0, 2.0} with d2Cache parameters:
   - kp (key projection ratio): {0.25, 0.50}
   - kr (KV reuse ratio): {2, 4}
   - rou (refresh rate): {0.1, 0.25}
3. Evaluate on 200 GSM8K + 100 MATH500, seed=42
4. Record: actual TPS, accuracy, cache hit rate, VRAM usage

### Decision Rule
- **d2Cache works** (TPS improvement >= 1.5x at < 2% accuracy drop): Use kernel-level M1
  for all subsequent experiments
- **d2Cache partially works** (1.2-1.5x or accuracy drops > 2%): Report both simplified
  and kernel-level; use kernel-level for Pareto but flag limitation
- **d2Cache fails** (< 1.2x or integration error): Report M1 with theoretical speedup
  estimation; label clearly as "projected" not "measured"

### Risk: d2Cache Incompatibility
d2Cache was designed for AR models. LLaDA's bidirectional attention may require
modification. If direct integration fails:
1. Try adapter: wrap LLaDA's attention with d2Cache's sparse KV selection
2. If adapter fails: implement simplified sparse attention using PyTorch custom
   attention mask (skip cached positions)
3. If all fail: report M1 as cache-hit-rate measurement + theoretical speedup;
   this is a legitimate finding (the gap between theory and practice)

### Estimated Runtime: 45 minutes

---

## Phase 2: Corrected Single-Method Pareto Curves

### Goal
Re-establish clean Pareto curves for M1, IGSD, and M3 with:
- Consistent QAS formula (no penalty)
- GSM8K + MATH500 as combined metric
- HumanEval reported separately
- 3 seeds for full experiments

### Methods and Sweeps

| Method | Parameter | Sweep Values | Notes |
|--------|-----------|-------------|-------|
| M1 (d2Cache or simplified) | entropy_threshold | {0.5, 1.0, 2.0} | Use d2Cache if Phase 1 succeeds |
| IGSD | tau, T_draft | tau={0.7, 0.85, 0.9}, T_draft={16, 32, 48} | Conservative T_draft=32-48 added |
| M3 (AR-guided) | guidance_weight | {0.3, 0.5, 0.7} | Qwen2.5-0.5B guide model |

### Protocol
**Pilot** (per method):
- 200 GSM8K + 100 MATH500 + 50 HumanEval, seed=42
- ~15-20 min per method

**Full** (per method):
- 1319 GSM8K + 500 MATH500 + 164 HumanEval, seeds=[42, 123, 456]
- ~60-90 min per method

### Key Changes from Iteration 1
- IGSD: Add T_draft=32 and T_draft=48 (iter_001 only tested T_draft=4,8,16)
  - Hypothesis: conservative drafting improves accuracy retention significantly
- M3: Drop gw=1.0 (too aggressive, iter_001 showed diminishing returns above 0.7)
- M2: **Excluded from composition study** -- NO_GO confirmed; report as negative result
  with explanation that simplified Saber lacks backtracking mechanism

---

## Phase 3: Complete Pairwise Composition with Full-Scale Validation

### Goal
Measure all 3 viable pairwise combinations at full GSM8K scale with 3 seeds,
using corrected QAS formula. Resolve the M1+M3 discrepancy (pilot Ortho=1.339
vs full Ortho=0.301).

### Pairs
1. **M1+IGSD**: Expected synergy (iter_001 Ortho=1.385 on 200 GSM8K)
2. **M1+M3**: Needs investigation (pilot showed 2.25x/99.7% but full showed 0.93x/54.4%)
3. **M3+IGSD**: Expected interference (iter_001 Ortho=0.493)

### Protocol
For each pair:
- Use best operating point from Phase 2 Pareto curves
- **Full scale**: 1319 GSM8K + 500 MATH500, seeds=[42, 123, 456]
- Record: TPS, accuracy, QAS, Ortho per benchmark
- Per-seed breakdown to assess stability

### M1+M3 Investigation
The dramatic discrepancy between pilot (100 samples, seed 42: Ortho=1.339) and
full pairwise (200 samples, seeds 42+123: Ortho=0.301) must be explained:
- Possible cause 1: HumanEval 0% dragging down combined AccRet
- Possible cause 2: Different M1 threshold settings
- Possible cause 3: Sample size effect
- **Action**: Re-run M1+M3 on GSM8K-only at full scale to isolate the effect

### Estimated Runtime: ~3 hours (parallelizable to ~2 hours on 2 GPUs)

---

## Phase 4: Three-Way Composition and Pareto Frontier

### Goal
Combine M1+IGSD+M3. Map the three-way Pareto frontier to identify
practically useful operating points.

### Protocol
Sweep reduced grid (informed by Phase 2/3 results):
- M1 entropy_threshold: {best_from_phase2, 1_step_more_aggressive}
- IGSD tau: {0.85, 0.9}, T_draft: {32, 48}
- M3 guidance_weight: {0.0 (off), 0.3, 0.7}

Total: 2 * 4 * 3 = 24 configurations.
Each on 200 GSM8K + 100 MATH500, seed=42 (pilot).
Top 5 configurations validated at full scale with 3 seeds.

### Expected Operating Points
1. **Max-speed**: M1+IGSD (no M3) -- highest speedup, ~30-50% accuracy retention
2. **Balanced**: M1+IGSD(conservative)+M3(gw=0.3) -- moderate speedup with quality insurance
3. **Quality-first**: M1+M3(gw=0.3) -- minimal speedup, near-lossless quality

### Estimated Runtime: ~2 hours pilot + ~2 hours full validation

---

## Phase 5: Cross-Model Generalization (Dream-7B-Instruct)

### Goal
Validate top 5 configurations on Dream-7B-Instruct to test whether composition
patterns transfer across models.

### Protocol
1. Download and validate Dream-7B-Instruct (hkunlp/dream-7b-instruct)
   - iter_001 download failed; retry with fallback mirrors
2. Run baseline (64-step) on 200 GSM8K + 100 MATH500, seed=42
3. Run top 5 configurations from Phase 4 on same benchmarks
4. Compare Ortho scores between LLaDA and Dream

### Fallback
If Dream-7B download fails again:
- Use LLaDA with different sequence lengths (128, 256, 512) as generalization test
- Report single-model limitation prominently in paper

### Estimated Runtime: ~3 hours

---

## Phase 6: AR Baseline Comparison

### Goal
Honest comparison of best composed DLM acceleration against properly optimized
AR inference.

### Protocol
1. **AR model**: Qwen2.5-7B-Instruct (closest size to LLaDA-8B) with vLLM
2. **AR acceleration**: Speculative decoding with Qwen2.5-0.5B draft model
3. **Batch sizes**: 1 (interactive), 8 (serving)
4. **Benchmarks**: 200 GSM8K + 100 MATH500, seed=42
5. Report: TPS, accuracy, quality-adjusted throughput

### Expected Result
AR with speculative decoding likely faster in raw TPS but DLM's parallel
generation may show advantage at higher batch sizes. The comparison
positions our work honestly in the broader ecosystem.

### Estimated Runtime: ~45 minutes

---

## Phase 7: IGSD Ablation Refinement

### Goal
Complete the IGSD ablation with corrected analysis (no QAS penalty).

### Ablations
1. IGSD with T_draft={16, 32, 48} at fixed tau=0.9 -- isolate draft length effect
2. IGSD with tau={0.7, 0.85, 0.9} at fixed T_draft=32 -- isolate threshold effect
3. IGSD w/o confidence partitioning (tau=0.0) vs with (tau=0.9) at T_draft=32
   - This directly addresses whether the confidence gate adds value over naive step reduction
4. Per-step KL divergence profile: Record KL(p_t || p_{t-1}) at every step for 100 GSM8K samples
   - Validate H6: "inverted-U" KL profile hypothesis

### Protocol
200 GSM8K + 100 MATH500, seed=42 for all ablations.
Full scale (1319 GSM8K, 3 seeds) for the key tau=0.0 vs tau=0.9 comparison.

### Estimated Runtime: ~2 hours

---

## Phase 8: Batch Size Sensitivity (Secondary)

### Goal
Address the batch_size=1-only limitation from iter_001.

### Protocol
For best pairwise (M1+IGSD) and best three-way (M1+IGSD+M3) configurations:
- batch_size = {1, 4, 8}
- 200 GSM8K, seed=42
- Record: TPS, accuracy, per-sample latency, GPU utilization

### Expected Finding
At higher batch sizes, confidence profiles are mixed across samples in a batch,
potentially reducing cache hit rates and accept rates. This would explain why
composition effects may differ in production vs. interactive settings.

### Estimated Runtime: ~30 minutes

---

## Expected Visualizations

- **Table 1**: Single-method Pareto table (Method x {Speedup, Acc_Ret, QAS} on GSM8K + MATH500)
- **Table 2**: Pairwise Ortho matrix (3 pairs x 2 benchmarks, corrected metric)
- **Table 3**: Three-way Pareto operating points (top 5 configs with speed/quality tradeoff)
- **Table 4**: Cross-model comparison (LLaDA vs Dream, top 5 configs)
- **Table 5**: AR vs DLM honest comparison at batch=1 and batch=8
- **Figure 1**: Architecture diagram -- IGSD draft-partition-refine pipeline
- **Figure 2**: Speed-accuracy Pareto curves (each method individually + compositions)
- **Figure 3**: Per-step KL divergence profile (IGSD H6 validation, inverted-U shape)
- **Figure 4**: Ortho score comparison across pairs (bar chart with confidence intervals from 3 seeds)
- **Figure 5**: Batch size sensitivity (TPS and accuracy vs batch size for key compositions)
- **Figure 6**: IGSD T_draft ablation (QAS vs T_draft, showing Pareto-optimal point)

---

## Baselines

| Baseline | Description |
|----------|-------------|
| LLaDA-8B default | 64-step denoising, no acceleration, bf16, greedy decoding (iter_001 validated) |
| Dream-7B default | Same protocol on Dream-7B-Instruct (cross-model validation) |
| Qwen2.5-7B + vLLM | AR baseline with speculative decoding (honest comparison) |

---

## Hardware

- **GPUs**: 2x NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM each)
- **Per-task**: Most tasks use 1 GPU; LLaDA-8B requires ~16-20 GB VRAM in bf16
- **Parallelism**: Independent tasks run on separate GPUs concurrently

---

## Timeline Estimate

| Phase | Est. Hours | Parallelizable | GPU Count |
|-------|-----------|---------------|-----------|
| Phase 1: d2Cache integration | 0.75 | No (gate) | 1 |
| Phase 2: Single-method Pareto | 3.0 | Yes (3 methods) | 2 |
| Phase 3: Pairwise composition | 3.0 | Yes (3 pairs) | 2 |
| Phase 4: Three-way Pareto | 4.0 | Partially | 2 |
| Phase 5: Dream-7B generalization | 3.0 | Yes | 1 |
| Phase 6: AR comparison | 0.75 | Independent | 1 |
| Phase 7: IGSD ablation | 2.0 | Yes | 1 |
| Phase 8: Batch sensitivity | 0.5 | Independent | 1 |
| **Total** | **~17 hours** | | |
| **Wall-clock (2 GPUs)** | **~10 hours** | | |

---

## Risk Mitigation

| Risk | Likelihood | Trigger | Mitigation |
|------|-----------|---------|------------|
| d2Cache integration fails | Medium | LLaDA bidirectional attention incompatible | Fall back to theoretical speedup; label as "projected" |
| Dream-7B download fails again | Medium | HuggingFace CDN issues | Use sequence-length variation as alternative generalization |
| IGSD T_draft=32-48 still poor accuracy | Medium | Accuracy retention < 60% at T_draft=48 | Report T_draft sweep as negative result; reframe IGSD as speed-only tool |
| M1+M3 remains low Ortho at full scale | Medium | GSM8K-only Ortho still < 0.5 | Investigate M1 slowing down M3; report as interference finding |
| AR baseline dominates all DLM configs | High | Qwen2.5-7B+vLLM faster at all quality levels | This is a legitimate finding; positions DLM acceleration gap honestly |

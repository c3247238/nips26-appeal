# ComposeAccel — Iteration 1 Experiment Methodology (REVISION CYCLE)

## Iteration 1 Focus: tau=0.0 Paradox Resolution + Paper Critical Fixes

**Previous iteration (0) completed all primary experiments.** This iteration addresses three critical issues identified in the review (score 5.5/10) before paper revision.

---

# ComposeAccel — Original Experiment Methodology (Iteration 0 Reference)

## Research Goal

Systematic study of composability, orthogonality, and failure modes of training-free acceleration
methods for Masked Diffusion Language Models (MDMs), with a new method (IGSD) filling the
self-speculative decoding gap.

---

## Phase 0: Environment Setup

### Target Model
- **Primary**: `GSAI-ML/LLaDA-8B-Instruct` (HuggingFace)
- **Secondary**: `hkunlp/dream-7b-instruct` (cross-model transferability check)

### Evaluation Framework
- **Custom MDM eval wrapper** built on top of LLaDA's official inference code
  (`ML-GSAI/LLaDA`) and DARE framework. Standard `lm-evaluation-harness` does NOT
  natively support MDMs (bidirectional masked denoising is incompatible with the
  left-to-right generation interface). We wrap LLaDA's `generate()` to emit predictions
  compatible with lm-eval task format.
- **Fallback**: Direct evaluation scripts using GSM8K/HumanEval dataset loaders + exact
  match / pass@1 metrics implemented in-house.

### Acceleration Methods (M1–M3 + IGSD)
| ID | Method | Representative Implementation | Notes |
|----|--------|-------------------------------|-------|
| M1 | KV-Cache (approximate) | EntropyCache (`mscheong01/EntropyCache`) + dKV-Cache | Primary implementation |
| M2 | Adaptive Step Scheduling | Saber (`arXiv:2510.18165`) — training-free adaptive acceleration + backtracking | |
| M3 | AR-Guided Unmasking | FlashDLM FreeCache + Guided Diffusion (`arXiv:2505.21467`) with Qwen2.5-0.5B supervisor | Qwen2.5-0.5B for small footprint |
| IGSD | Self-Speculative Denoising | Implemented from scratch per proposal Algorithm | New method |

### Benchmarks
| Task | Metric | Subset Size (Full) | Subset Size (Pilot) |
|------|--------|--------------------|---------------------|
| GSM8K | Exact match (8-shot) | 1319 (full test) | 200 samples |
| MATH500 | Exact match (4-shot) | 500 | 100 samples |
| HumanEval | pass@1 (0-shot) | 164 | 50 problems |
| MBPP | pass@1 (3-shot) | 374 | 100 problems |

**Baseline**: LLaDA-8B-Instruct with default 64-step denoising, no acceleration.

### Hardware
- Local GPU (per spec.md): up to 2 GPUs available
- Primary: 1×GPU per experiment task; 8B model requires ~18GB VRAM in bf16
- Estimated VRAM per task: 18–20GB (model) + ~4–6GB (KV-cache buffer) → fits on A100 40GB

### Throughput Measurement Protocol
- Metric: wall-clock tokens-per-second (TPS) averaged over 50 stable-state samples
  (discard first 5 warm-up samples)
- Measure: total output tokens / elapsed wall-clock time (generation only, exclude prompt
  encoding)
- Seeds: pilot = [42]; full = [42, 123, 456], report mean ± std

### Composability Metric
```
Ortho(Ma + Mb) = Speedup(Ma+Mb) / (Speedup(Ma) × Speedup(Mb))
```
- Ortho ≥ 0.90: highly orthogonal (near-multiplicative)
- Ortho 0.80–0.90: partially orthogonal
- Ortho < 0.80: sub-orthogonal / destructive interference

### Quality-Adjusted Speedup (QAS)
```
QAS = Speedup × Accuracy-Retention
Accuracy-Retention = Acc(method) / Acc(baseline)
```

---

## Phase 1: IGSD Feasibility Pilot (H6)

**Goal**: Determine whether IGSD's token acceptance rate at τ=0.85 with T_draft=8 is ≥ 50%.
This is the gate condition before investing in full IGSD implementation and composability
experiments.

**Protocol**:
1. Load LLaDA-8B-Instruct in bf16 on 1×GPU
2. For each prompt in 200 GSM8K + 50 HumanEval samples (seed=42):
   - Run 8-step denoising (T_draft=8, standard LLaDA schedule)
   - Collect per-token confidence: `c[i] = max_v softmax(logits)[v]` at final draft step
   - Record `|S_accept(τ)| / N` for τ ∈ {0.70, 0.80, 0.85, 0.90}
3. Report acceptance rate distribution (median, 25th/75th percentile) per τ
4. Decision rule (see alternatives.md):
   - accept_rate ≥ 50% at τ=0.85 → proceed with IGSD full implementation
   - 40–50% → tune τ to 0.70, report IGSD as exploratory
   - < 40% at τ=0.70 → drop IGSD, pivot plan to Backup Candidate A

**Runtime estimate**: ~12 minutes (250 short samples, 8-step forward passes each)

---

## Phase 2: KV-Cache Hit-Rate vs. Unmasking Fraction (H5 Pilot)

**Goal**: Characterize KV-cache hit-rate as a function of per-step unmasking fraction across
Saber configurations. Establishes the H5 failure-mode hypothesis mechanism.

**Protocol**:
1. Instrument EntropyCache implementation to log per-step cache hit-rate
2. Run LLaDA-8B-Instruct on 100 GSM8K prompts with:
   - M2 (Saber) at step-jump = {2x, 4x, 6x} tokens/step
   - M1 (EntropyCache) active simultaneously
3. Record: {step_jump, unmasking_fraction, cache_hit_rate, per-sequence accuracy}
4. Plot: cache_hit_rate vs. unmasking_fraction (expected: negative correlation at 4x+)

**Runtime estimate**: ~10 minutes

---

## Phase 3: Single-Method Baseline Pareto Curves

**Goal**: Establish reproducible speed-accuracy Pareto curves for each method individually.
This is the foundation for all composability measurements.

**Protocol**:
1. For each method M ∈ {M1, M2, M3, IGSD (if H6 confirmed)}:
   - Sweep acceleration aggressiveness parameter:
     - M1 (EntropyCache): entropy threshold ∈ {0.5, 1.0, 2.0, 3.0}
     - M2 (Saber): step-jump ∈ {2x, 4x, 6x, 8x} tokens/step
     - M3 (Guided Diffusion): guidance weight ∈ {0.3, 0.5, 0.7, 1.0}
     - IGSD: τ ∈ {0.70, 0.80, 0.85, 0.90}, T_draft ∈ {4, 8, 16}
   - Evaluate on: 200 GSM8K + 50 HumanEval (pilot); full benchmarks (full experiment)
   - Record: TPS, accuracy, QAS
2. Identify the "operating point" for each method: highest QAS within 2% accuracy drop

**Runtime (pilot)**: ~30 min total (4 methods × ~7–8 min each)
**Runtime (full)**: ~3–4 hours per method across all benchmarks and seeds

---

## Phase 4: Pairwise Orthogonality Analysis

**Goal**: Measure Ortho(Ma+Mb) for all C(4,2)=6 method pairs and the four-way combination.

**Pairs**:
1. M1 + M2 (KV-cache + Adaptive Scheduling) — H1: expected sub-orthogonal at 4x step-jump
2. M1 + M3 (KV-cache + AR-Guided)
3. M1 + IGSD — H2: expected highly orthogonal (≥ 0.90)
4. M2 + M3
5. M2 + IGSD
6. M3 + IGSD
7. M1 + M2 + M3 + IGSD (four-way) — H3: expected < 0.7 × product

**Protocol**:
- Each pair: use operating-point parameters from Phase 3
- Evaluate on: 200 GSM8K + 50 HumanEval (pilot); full benchmarks (full)
- Record: Speedup, Accuracy-Retention, Ortho, QAS
- For H1 specifically: sweep M2 step-jump {2x, 4x, 6x} to identify interference threshold

**Runtime (pilot)**: ~20 min (7 combinations × ~3 min each)
**Runtime (full)**: ~6–8 hours

---

## Phase 5: Task-Dependent Optimal Recipe (H4)

**Goal**: Determine whether optimal combination differs between reasoning and coding tasks.

**Protocol**:
1. Collect all pairwise Ortho scores from Phase 4 stratified by task type:
   - Reasoning: GSM8K + MATH500
   - Coding: HumanEval + MBPP
2. For each task type, rank combinations by QAS
3. Statistical test: Wilcoxon signed-rank on QAS scores per combination across tasks

**Runtime (full)**: subsumed in Phase 4 full evaluation (no extra compute)

---

## Phase 6: Failure Mode Atlas (H5 Full + Edge Cases)

**Goal**: Characterize input conditions that cause each method (or combination) to fail
catastrophically (> 5% accuracy drop vs. baseline).

**Protocol**:
1. H5 full validation: Run M1+M2 across full GSM8K with per-step unmasking fraction logging;
   identify threshold where accuracy drop crosses 2% (prediction: unmasking_fraction > 0.25)
2. Systematic stress test: for each method at aggressive settings:
   - Long sequences (256 tokens output)
   - Complex multi-step reasoning (MATH500 level 4–5)
   - Code with deep nesting (HumanEval hard problems)
3. Failure mode taxonomy: classify failures as {cache_invalidation, step_starvation,
   draft_divergence, AR_guidance_conflict}
4. Detection rule: for each failure mode, define a proactive detection signal
   (e.g., per-step unmasking fraction > 0.25 → trigger M1 cache refresh)

**Runtime (full)**: ~2 hours

---

## Phase 7: Ablation Studies

**Goal**: Validate that each component of IGSD contributes meaningfully.

**IGSD ablations**:
1. IGSD w/o confidence partitioning (uniform τ=0) — measures draft quality alone
2. IGSD w/o KV-cache in REFINE phase — measures KV synergy
3. IGSD with T_draft = {4, 8, 16} — validates T_draft=8 as Pareto-optimal

**M1 (EntropyCache) ablations**:
1. Entropy-guided refresh vs. uniform refresh (every N steps)
2. Per-layer refresh vs. global refresh

**Runtime (full)**: ~2–3 hours

---

## Expected Visualizations

- **Table 1**: Single-method Pareto table (Method × {Speedup, Accuracy, QAS} across benchmarks)
- **Table 2**: 6×3 orthogonality matrix (method pair × benchmark, value = Ortho score)
- **Figure 1**: Architecture diagram of IGSD algorithm (draft→partition→refine pipeline)
- **Figure 2**: Speed-accuracy Pareto curve for each method (Speedup vs. Accuracy-Retention)
- **Figure 3**: Cache hit-rate vs. unmasking fraction (H5 failure mode visualization)
- **Figure 4**: QAS comparison bar chart across benchmarks (best combinations per task type)
- **Figure 5**: IGSD acceptance rate distribution (per τ value, boxplot or violin plot)
- **Figure 6 (optional)**: Four-way speedup decomposition showing sub-multiplicativity

---

## Baselines

| Baseline | Description |
|----------|-------------|
| LLaDA-8B default | 64-step denoising, no acceleration, bf16, greedy decoding |
| Dream-7B default | Same protocol on Dream-7B-Instruct (cross-model check) |
| Fast-dLLM (NVlabs) | Strongest published single-method baseline; reproduce from official code |

---

## Risk Mitigation

| Risk | Trigger | Mitigation |
|------|---------|------------|
| IGSD infeasible | accept_rate < 40% at τ=0.70 | Drop IGSD; pivot composability to 3-method study (M1+M2+M3); report as negative result |
| All pairs sub-orthogonal | All Ortho < 0.80 | Elevate failure-mode atlas as primary contribution; reframe as "why methods conflict" |
| LLaDA VRAM OOM | bf16 + KV-buffer > GPU VRAM | Use int8 quantization via bitsandbytes for inference (not training); verify accuracy impact |
| Saber not available | GitHub repo unavailable | Implement simplified adaptive scheduling: unmask top-k confidence tokens per step |
| Eval framework incompatibility | lm-eval harness fails for MDM | Use direct eval scripts with GSM8K/HumanEval dataset loaders from LLaDA repo |

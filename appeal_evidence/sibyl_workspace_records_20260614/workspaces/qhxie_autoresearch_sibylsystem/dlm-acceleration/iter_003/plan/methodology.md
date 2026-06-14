# ComposeAccel -- Iteration 3 Experiment Methodology

## Iteration Focus: Full-Scale Evidence, New Method Axis, Engineering Fix, Honest Reframing

Iteration 2 achieved comprehensive experimental coverage (15 groups) but scored 5.5/10 due to
three critical deficiencies: (1) pairwise compositions at pilot scale only (100 samples, 1 seed),
(2) IGSD confidence gate proven inert (null ablation), (3) M1 is a 1.16x no-op. The reflection
identified these as recurring issues blocking publication.

**Iteration 3 addresses all three root causes AND extends the study with a genuinely new method:**

1. **Full-scale pairwise replication** (recurring critical): 1319 GSM8K, 3 seeds for all 3 pairs
2. **Fast-dLLM integration** (replaces broken M1): Kernel-level KV cache from NVIDIA's ICLR 2026 codebase
3. **JoT integration** (new axis): Per-token early stopping -- the ONE method that claims orthogonality to KV caching but has never been empirically validated in composition
4. **M3 speedup clarification**: Separate per-token speed from output-length effect
5. **IGSD honest reframing**: Rewrite contribution as "simple draft-truncate with null confidence gate"
6. **Baseline standardization**: Phase 0 canonical measurement before any speedup experiments
7. **Writing completion**: Generate missing figures, replace citation placeholders

**Key strategic shift**: Rather than defending the "three orthogonal axes multiply" thesis (falsified),
reframe the paper as "the first rigorous composability atlas with interference mechanisms explained."
The honest negative results are the paper's genuine strength. Adding JoT (a genuinely different axis:
per-token early stopping vs. per-step cost reduction) tests whether ANY cross-family composition
works on DLMs.

---

## Phase 0: Baseline Standardization (GATE -- must complete before all speedup experiments)

### Goal
Establish a single canonical baseline TPS measurement that all speedup calculations reference.
Eliminates the 31.0/33.8/58.5 TPS inconsistency from iter_001/002.

### Protocol
1. LLaDA-8B-Instruct, bf16, 64 denoising steps, greedy decoding
2. GSM8K test split, 200 samples, seed=42
3. Measurement: generation-only wall-clock time (exclude prompt encoding)
4. Discard first 10 warm-up samples
5. Record: per-sample latency, total TPS, VRAM usage, GPU utilization
6. Repeat at batch={1, 4, 8} for batch sensitivity reference
7. Write canonical reference to `exp/results/baseline/canonical_baseline.json`

### Output
- Canonical baseline TPS (batch=1): single reference number for ALL speedup calculations
- Per-batch baseline TPS: reference for batch sensitivity analysis
- Measurement protocol documented in JSON metadata

### Estimated Runtime: 20 minutes

---

## Phase 1: Fast-dLLM Integration (Replaces broken M1)

### Goal
Replace our broken M1 (EntropyCache without kernel integration, 1.16x no-op) with Fast-dLLM's
properly engineered kernel-level KV cache + confidence-aware parallel decoding. This is the
engineering fix that iter_002's d2Cache attempt failed to achieve (15.2x overhead).

### Protocol
1. Clone Fast-dLLM repository (NVlabs/Fast-dLLM, ICLR 2026)
2. Run with published configs on LLaDA-8B-Instruct: GSM8K 200 samples, seed=42
3. Sweep confidence threshold: {0.5, 0.7, 0.85, 0.95}
4. Record: actual TPS, accuracy, cache hit rate, VRAM
5. Compare against our canonical baseline AND our iter_002 M1 (1.16x)

### Decision Rule
- **Fast-dLLM works** (TPS >= 5x at < 5% accuracy drop): Use as new M1 for all compositions
- **Fast-dLLM partially works** (2-5x): Use with caveats; report engineering gap
- **Fast-dLLM fails** (< 2x or integration error): M1 remains a documented no-op;
  report the engineering-vs-algorithm gap as a key finding

### Risk: Fast-dLLM requires specific block-wise generation
Fast-dLLM was designed for block-wise decoding. LLaDA-8B-Instruct may need adapter.
If direct integration fails:
1. Use Fast-dLLM's cache mechanism with LLaDA's default generation loop
2. If that fails: use EntropyCache codebase (which has unified benchmark support for LLaDA)
3. If that fails: report M1 as no-op and focus composition study on {IGSD, M3, JoT}

### Estimated Runtime: 45 minutes

---

## Phase 2: JoT (Token-Level Early Stopping) Integration

### Goal
Integrate JoT (Just on Time, arXiv 2602.11133) as a genuinely new method axis. JoT claims
orthogonality to KV caching (it reduces step count per token, while caching reduces per-step
cost) but has NEVER been empirically validated in composition. This is the single strongest
new experimental contribution for iter_003.

### Protocol
1. Clone JoT repository (Anonym-cybersudo/JoT)
2. Run on LLaDA-8B-Instruct: GSM8K 200 samples, seed=42
3. Sweep early-stopping threshold: {0.7, 0.85, 0.9, 0.95}
4. Record: TPS, accuracy, per-token step counts, VRAM
5. Establish JoT's individual Pareto curve

### Composition Tests (Phase 4)
JoT + Fast-dLLM (KV cache + token early stop): the predicted orthogonal pair
JoT + IGSD (token early stop + step truncation): potential interference (both reduce steps)
JoT + M3 (token early stop + AR guidance): unclear interaction

### Expected Finding
If JoT + Fast-dLLM achieves > 80% of the product of individual speedups at > 90% quality
retention, this is the first validated cross-family composition in DLM acceleration --
a definitive positive result contrasting with our iter_001/002 interference findings.

### Estimated Runtime: 30 minutes

---

## Phase 3: Corrected Single-Method Pareto Curves (Full Scale)

### Goal
Complete full-scale Pareto curves for all methods under consistent measurement protocol.
Address iter_002's sample-size inconsistency (M1: N=1319, IGSD: N=200, M3: N=100).

### Methods

| Method | Source | Parameter | Sweep | Scale |
|--------|--------|-----------|-------|-------|
| Fast-dLLM (new M1) | NVlabs repo | confidence_threshold | {0.5, 0.7, 0.85, 0.95} | 1319 GSM8K, 3 seeds |
| IGSD | iter_001 code | tau, T_draft | tau={0.85, 0.9}, T_draft={32, 48} | 1319 GSM8K, 3 seeds |
| M3 (AR-guided) | iter_001 code | guidance_weight | {0.3, 0.5, 0.7} | 1319 GSM8K, 3 seeds |
| JoT (new) | JoT repo | threshold | {0.7, 0.85, 0.9, 0.95} | 1319 GSM8K, 3 seeds |

### Key Changes from Iteration 2
- ALL methods run at full GSM8K scale (1319) with 3 seeds -- no mixed sample sizes
- Fast-dLLM replaces broken EntropyCache M1
- JoT added as 4th method
- MATH500 relegated to supplementary (11.1% baseline too noisy for primary metric)
- GSM8K-only is the primary metric; MATH500 and HumanEval in appendix

### Protocol
**Pilot** (per method): 200 GSM8K, seed=42, ~15 min each
**Full** (per method): 1319 GSM8K, seeds=[42, 123, 456], ~45-60 min each

### M3 Speedup Clarification (NEW)
For M3, additionally measure:
- (a) Average output token count for M3 vs baseline
- (b) Per-sample wall-clock latency (not just TPS)
- (c) Per-token latency = wall-clock / output_tokens
This disambiguates whether M3's "1.68x speedup" is per-token speed or output-length efficiency.

### Estimated Runtime: 4 hours (parallelizable to ~2 hours on 2 GPUs)

---

## Phase 4: Complete Pairwise Composition -- FULL SCALE (CRITICAL)

### Goal
The single highest-impact improvement: run ALL pairwise compositions at full GSM8K scale
(1319 samples, 3 seeds) with bootstrap 95% CIs. This directly addresses the recurring
critical issue (C2) that has blocked publication for two iterations.

### Pairs (6 pairs from 4 methods)

| Pair | Expected Interaction | iter_002 Pilot Ortho | Rationale |
|------|---------------------|---------------------|-----------|
| Fast-dLLM + IGSD | Near-orthogonal | N/A (new) | Per-step cost reduction + step count reduction |
| Fast-dLLM + M3 | Interference? | 0.30 (M1+M3 proxy) | Both modify attention distributions |
| Fast-dLLM + JoT | **Predicted synergy** | N/A (new) | Per-step cost + per-token step count -- genuinely different axes |
| IGSD + M3 | Interference | 0.49 | Both modify denoising trajectory |
| IGSD + JoT | Interference? | N/A (new) | Both reduce step count (different granularity) |
| M3 + JoT | Unknown | N/A (new) | AR guidance + token early stop |

### Protocol
For EACH pair:
- Use best operating point from Phase 3 Pareto curves
- **Full scale**: 1319 GSM8K, seeds=[42, 123, 456]
- Report: TPS, accuracy, QAS, Ortho per seed
- Bootstrap 95% CI for Ortho
- Hard gate: Ortho claims require N>=500 AND >=2 seeds

### Key Composition: Fast-dLLM + JoT
This is the headline experiment. JoT explicitly claims orthogonality to KV caching.
If Fast-dLLM (proper KV cache, ~10-20x) + JoT (~7x) compose to > 40x at > 90% quality,
this is a landmark result. If they interfere (Ortho < 0.5), it extends the interference
finding to yet another pair.

### Estimated Runtime: ~6 hours (parallelizable to ~3 hours on 2 GPUs)

---

## Phase 5: Three-Way and Four-Way Composition Frontier

### Goal
Map the expanded composition frontier with the best-performing pairs from Phase 4.

### Protocol
**Three-way**: Top 3 three-way combinations from Phase 4 evidence:
- Fast-dLLM + IGSD + M3 (evolved from iter_002)
- Fast-dLLM + JoT + M3 (new)
- Fast-dLLM + JoT + IGSD (new)

**Four-way** (if Phase 4 shows multiple synergistic pairs):
- Fast-dLLM + JoT + IGSD + M3

For each: 200 GSM8K pilot (seed=42), then top 3 configs at full scale (1319, 3 seeds).

### Estimated Runtime: ~3 hours

---

## Phase 6: KV Drift Diagnostic (Explains Interference)

### Goal
Provide the mechanistic explanation for interference. Address the pragmatist perspective's
key question: WHY do methods interfere?

### Protocol
For 100 GSM8K samples (seed=42), at each denoising step record:
1. Cosine similarity of KV states between step t and step t-1 (KV drift rate)
2. KV drift under vanilla vs. M3-guided vs. JoT-pruned generation
3. Per-token entropy trajectory under each method

### Expected Finding
M3 guidance changes unmasking order -> different KV evolution trajectory -> stale caches
from Fast-dLLM become MORE stale (increased drift). JoT stops tokens early -> removes them
from attention -> changes KV for remaining tokens, but does NOT inject new information
(unlike M3). Hence JoT + cache should show lower drift amplification than M3 + cache.

### Estimated Runtime: 1 hour

---

## Phase 7: Cross-Model Validation (Dream-7B)

### Goal
Validate top 3 composition configurations on Dream-7B-Instruct.

### Protocol
1. Reuse Dream-7B checkpoint (downloaded in iter_002)
2. Run baseline + top 3 compositions at 200 GSM8K, seed=42
3. Compare Ortho patterns across models

### Caveat
Dream-7B baseline is ~36% on GSM8K (vs LLaDA 71.2%). All AccRet values will be
inflated. Report separately, do not include in primary tables.

### Estimated Runtime: 1.5 hours

---

## Phase 8: AR Baseline Comparison (Updated)

### Goal
Honest comparison with properly optimized AR inference.

### Protocol
1. Qwen2.5-7B-Instruct with HuggingFace Transformers (reuse iter_002 setup)
2. Add vLLM serving comparison IF installable within 15 min
3. Batch sizes 1 and 8
4. 200 GSM8K, seed=42
5. Include caveat: "HF-based comparison is conservative lower bound; vLLM/TRT-LLM would widen gap"

### Estimated Runtime: 45 minutes

---

## Phase 9: Degenerate Output Analysis

### Goal
Quantify output quality beyond pass/fail (new issue M8 from iter_002).

### Protocol
For all IGSD and composition experiments with AccRet < 80%:
1. Count samples with repetition loops (>3 consecutive identical tokens)
2. Count samples with premature truncation (<50 output tokens)
3. Count samples with whitespace flooding (>20% whitespace tokens)
4. Correlate accept_rate with output quality categories
5. Include 3 qualitative examples in paper

### Estimated Runtime: 30 minutes (analysis of existing outputs)

---

## Evaluation Framework

### Primary Metric: GSM8K-only
- 1319 samples, 3 seeds {42, 123, 456}
- Exact match accuracy
- Report mean +/- std across seeds
- Bootstrap 95% CI for all Ortho values

### Secondary Metrics (Appendix)
- MATH500 (500 samples) -- report with noise caveat (11.1% baseline)
- HumanEval (164 samples) -- report separately, do not include in QAS

### QAS Formula (Unchanged from iter_002)
```
QAS = Speedup * Accuracy_Retention
Accuracy_Retention = Acc(method) / Acc(baseline)
```
No penalty factor. No mixed baselines.

### Ortho Metric (Unchanged)
```
Ortho(Ma + Mb) = QAS(Ma+Mb) / max(QAS(Ma), QAS(Mb))
```
NEW: Report only when both methods have individual QAS > 1.2 (avoid M1 no-op degeneracy).

### Speedup Measurement Protocol
- Reference: canonical baseline TPS from Phase 0 (batch=1)
- Generation-only wall-clock time, post-warmup (discard first 10 samples)
- For M3: report BOTH TPS ratio AND per-sample latency ratio

---

## Expected Visualizations

- **Table 1**: Single-method Pareto (4 methods x {Speedup, AccRet, QAS} on GSM8K, N=1319, 3 seeds)
- **Table 2**: Pairwise Ortho matrix (6 pairs, with 95% CIs, N=1319, 3 seeds)
- **Table 3**: Best three-way/four-way compositions with operating points
- **Table 4**: Cross-model comparison (LLaDA vs Dream, top 3 configs)
- **Table 5**: AR vs DLM honest comparison at batch=1 and batch=8
- **Figure 1**: Architecture diagram -- ComposeAccel evaluation framework
- **Figure 2**: Speed-accuracy Pareto curves (all methods + key compositions)
- **Figure 3**: Ortho heatmap (6 pairs, color-coded synergy/interference)
- **Figure 4**: KV drift diagnostic (cosine similarity trajectories under different methods)
- **Figure 5**: Per-step KL divergence profile (from iter_002 data)
- **Figure 6**: Degenerate output analysis (bar chart of failure mode rates)

---

## Hardware

- **GPUs**: NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM), up to 4 GPUs (candidate_gpu_ids: [0, 1, 4, 5])
- **Per-task**: Most tasks use 1 GPU; LLaDA-8B requires ~16-20 GB VRAM in bf16
- **Parallelism**: Independent tasks run on separate GPUs concurrently (max 4 parallel)

---

## Timeline Estimate

| Phase | Est. Hours | Parallelizable | GPU Count |
|-------|-----------|---------------|-----------|
| Phase 0: Baseline standardization | 0.33 | No (gate) | 1 |
| Phase 1: Fast-dLLM integration | 0.75 | No (gate) | 1 |
| Phase 2: JoT integration | 0.50 | Yes (with Phase 1) | 1 |
| Phase 3: Full-scale Pareto (4 methods) | 4.00 | Yes (4 parallel) | 4 |
| Phase 4: Pairwise composition (6 pairs) | 6.00 | Yes (4 parallel) | 4 |
| Phase 5: Three/four-way composition | 3.00 | Partially | 2 |
| Phase 6: KV drift diagnostic | 1.00 | Independent | 1 |
| Phase 7: Dream-7B validation | 1.50 | Independent | 1 |
| Phase 8: AR comparison | 0.75 | Independent | 1 |
| Phase 9: Degenerate output analysis | 0.50 | CPU only | 0 |
| **Total** | **~18.3 hours** | | |
| **Wall-clock (4 GPUs)** | **~7-8 hours** | | |

---

## Risk Mitigation

| Risk | Likelihood | Trigger | Mitigation |
|------|-----------|---------|------------|
| Fast-dLLM integration fails on LLaDA-8B | Medium | Requires block-wise generation | Use EntropyCache unified benchmark code instead; report M1 as no-op with honest caveat |
| JoT integration fails | Low | Dependency issues | Re-implement core logic (50 lines: per-token confidence monitoring + early commit) |
| Fast-dLLM + JoT interference (Ortho < 0.5) | Medium | Shared confidence signals compete | This IS the finding -- extends interference thesis |
| Full-scale pairwise takes too long | Low | 1319 samples x 6 pairs x 3 seeds | Prioritize top 3 pairs first; extend to 6 if time permits |
| All compositions interfere | Medium | No synergistic pair found | Paper reframed as "interference atlas" -- equally publishable |
| AR baseline dominates everything | High | Known from iter_002 | Honest comparison is a strength; position as "the gap DLM acceleration must close" |

---

## Baselines

| Baseline | Description |
|----------|-------------|
| LLaDA-8B canonical | 64-step denoising, no acceleration, bf16, greedy, batch=1 (Phase 0) |
| Dream-7B canonical | Same protocol on Dream-7B (Phase 7) |
| Qwen2.5-7B + HF | AR baseline (Phase 8) |
| Qwen2.5-7B + vLLM | Optimized AR baseline if installable (Phase 8) |

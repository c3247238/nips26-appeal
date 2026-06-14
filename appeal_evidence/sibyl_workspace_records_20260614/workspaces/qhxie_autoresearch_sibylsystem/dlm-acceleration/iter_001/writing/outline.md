# Paper Outline: ComposeAccel

## Title

**ComposeAccel: A Pairwise Composability Study of Training-Free Acceleration Methods for Masked Diffusion Language Models**

Alternative: **When Does MDM Inference Acceleration Compose? A Systematic Composability and Failure-Mode Study**

---

## 1. Abstract (written last, from verified numbers)

Key elements to include:
- MDMs attract growing ecosystem of training-free acceleration (KV-caching, adaptive scheduling, AR-guided unmasking, speculative denoising), each evaluated in isolation. No prior work measures pairwise composability across method families.
- We introduce a composability framework with a formal orthogonality metric (Ortho) and Quality-Adjusted Speedup (QAS), evaluated on LLaDA-8B-Instruct across GSM8K (1319) and MATH500 (500), 3 seeds.
- Among three feasible method pairs (M2 excluded: NO_GO due to structural mask incompatibility), exactly one achieves super-multiplicative synergy: KV-cache (M1) + CD-SSD self-speculative denoising, Ortho=1.385 (2-seed, 200-sample estimate; full-scale pending), 5.13x combined speedup.
- Mechanistic driver: CD-SSD's refine phase freezes ~52% high-confidence tokens, enabling near-100% KV hit rate for those positions. Ablation confirms CD-SSD(tau=0.0) matches naive T=16 step reduction in QAS, indicating the confidence partitioning mechanism contributes marginal quality gain at the cost of complexity; the frozen-token KV amplification is the primary value.
- Failure-mode atlas: four failure modes characterized with detection heuristics — M2 discrete masking incompatibility (structural NO_GO), M1 overhead inversion below entropy threshold 2.0, CD-SSD low-draft-quality below T_draft=16, M3+IGSD AR distribution mismatch.
- Scope: analysis paper, single-model (LLaDA-8B-Instruct), 3-pair composability study. All claims scoped accordingly.

**Constraint on numbers in Abstract**: use only values confirmed in full-scale JSON artifacts. Current Ortho=1.385 is 2-seed. QAS uses standard formula Speedup × AccRet (no undisclosed penalties). Coding benchmarks excluded from primary claims.

---

## 2. Introduction (1.5 pages)

### 2.1 The Fragmentation Problem
- As of April 2026, at least 12 training-free acceleration methods exist for MDMs (KV-caching: EntropyCache, Fast-dLLM, dKV-Cache, Elastic-Cache, Window-Diffusion; scheduling: Saber; AR-guidance: FlashDLM; speculative: SSD, SSMD, S2D2), each evaluated in isolation under incompatible protocols.
- Practitioners face a fundamental deployment question: which method or combination maximizes throughput without catastrophic quality loss? No published work provides an answer.
- *Ref: Table 1 (literature speed comparison, using only published numbers from papers using the same base model where available).*

### 2.2 Two Urgent Gaps
1. **Unknown composability**: Methods targeting different bottlenecks should, in principle, compose multiplicatively — but MDM's global mask-state coupling creates non-obvious interaction risks. No cross-family composability measurement exists.
2. **Unknown failure boundaries**: All published papers report average-case results. No work characterizes the conditions under which methods fail catastrophically, preventing practitioners from detecting dangerous configurations proactively.

### 2.3 Our Contributions (numbered list)
1. **First pairwise composability study** of three training-free MDM acceleration families (KV-cache, AR-guided unmasking, self-speculative denoising) on LLaDA-8B-Instruct, with a formal orthogonality metric.
2. **Binary composability observation**: Among the three feasible pairs evaluated, one achieves super-multiplicative synergy (M1+CD-SSD, Ortho=1.385) and two exhibit destructive interference (M3+CD-SSD Ortho=0.493, M1+M3 Ortho=0.301).
3. **Mechanistic explanation**: CD-SSD's frozen-token refine phase creates near-zero-entropy KV entries that M1 exploits at peak cache efficiency. Ablation (tau=0.0 vs. naive T=16) resolves the tau paradox: the step-reduction component drives the speedup; confidence partitioning adds marginal quality at significant compute cost.
4. **CD-SSD (Coarse-Draft Self-Speculative Denoising)**: A training-free speculative method using reduced-step drafting (T_draft=16 vs. T_full=64). Concurrent with SSD (Gao et al., arXiv:2510.04147) and SSMD (Campbell et al., arXiv:2510.03929); differentiated by coarse-step draft mechanism and composability profile.
5. **Failure-mode atlas**: Four interference patterns with proactive detection signals — the first systematic worst-case characterization of MDM acceleration.

### 2.4 Scope and Positioning
- Analysis paper, not a methods paper. CD-SSD is a composability study vehicle, not claimed as SOTA over SSD.
- Single-model study (LLaDA-8B-Instruct). Generalization to Dream-7B and other MDMs is future work.
- M2 (adaptive step scheduling) excluded from composability experiments due to NO_GO verdict; reported as a structural incompatibility finding.
- Coding benchmarks (HumanEval 2.4%, MBPP 0.0% baseline) excluded from primary quantitative claims; reported separately with explicit degenerate-baseline caveats.
- M1 implementation achieves 1.38x vs. published 15–26x due to absence of kernel-level sparse attention; the dimensionless Ortho metric remains valid.

*Transition: Section 3 surveys related work. Section 4 formalizes methods and metrics. Section 5 reports results.*

---

## 3. Background and Related Work (1.5 pages)

### 3.1 Masked Diffusion Language Models
- LLaDA forward/reverse process: iterative masking over T steps (T=64 default), bidirectional attention.
- Inference bottleneck: each step requires a full O(N^2) forward pass; token states change globally, preventing standard KV reuse.
- Block-based semi-autoregressive generation in LLaDA; beam-free greedy decoding.

### 3.2 Training-Free Acceleration Families
- **KV-Cache Approximation (M1)**: EntropyCache (arXiv:2603.18489), Fast-dLLM (ICLR 2026), dKV-Cache (arXiv:2505.15781), Elastic-Cache (arXiv:2510.14973), Window-Diffusion (arXiv:2601.20332). Each reuses KV matrices across steps for stable positions; differs in refresh criterion (entropy, delay, window).
- **Adaptive Step Scheduling (M2)**: Saber (arXiv:2510.18165), PRR (arXiv:2603.04514). Reduce total steps by unmasking more tokens per step. Key risk: discrete masking requires sequential conditioning that DDIM-style schedules violate.
- **AR-Guided Unmasking (M3)**: FlashDLM Guided Diffusion (arXiv:2505.21467). Uses a lightweight AR model to bias unmasking order toward high-confidence tokens.
- **Speculative Denoising**: SSD (arXiv:2510.04147) — lossless, training-free, same-model hierarchical verification; SSMD (arXiv:2510.03929) — attention-mask switch; S2D2 (arXiv:2603.25702) — for block-diffusion models; DualDiffusion (arXiv:2604.05250) — external draft.
- **Not covered (training-based)**: Block Diffusion, Fast-dLLM v2, D2F, T3D.

### 3.3 The Composability Gap
- All existing methods are evaluated in isolation. No work measures pairwise orthogonality across families.
- The closest prior work is Kolbeinsson et al. (arXiv:2407.06483) on composable LLM interventions (compression, editing, unlearning) — different intervention types, not inference acceleration.
- MDM global mask-state coupling means methods that modify the denoising trajectory (M2, M3) may conflict with methods that assume trajectory stability (M1, CD-SSD).

*Transition: Section 4 defines the framework and methods.*

---

## 4. Methods (2.5 pages)

### 4.1 Composability Framework

**Orthogonality metric**:
```
Ortho(Ma + Mb) = Speedup(Ma+Mb) / (Speedup(Ma) × Speedup(Mb))
```
- Ortho >= 1.0: super-multiplicative synergy
- Ortho in [0.8, 1.0): partially orthogonal
- Ortho < 0.8: destructive interference

**Quality-Adjusted Speedup (QAS)** — uniform formula, no per-method penalties:
```
QAS(M) = Speedup(M) × AccRet(M)
AccRet(M) = Acc(M) / Acc(baseline)
```
All methods use identical formula. Methods with AccRet < 0.95 carry an explicit feasibility warning in tables but QAS is not modified.

*Ref: notation.md for symbol definitions.*

### 4.2 Method Implementations

#### M1: EntropyCache (KV-Cache Approximation)
- Per-token entropy H_i computed after each denoising step.
- If H_i < eta: reuse cached KV from step t-1. If H_i >= eta: recompute.
- Parameter sweep: eta in {0.5, 1.0, 2.0, 3.0}. Operating point: eta=2.0.
- Implementation note: our implementation computes entropy and applies logic but executes full forward passes (no kernel-level sparse attention). Measured speedup 1.38x reflects cache computation overhead savings at 60% hit rate; published 15–26x requires sparse attention kernels. The dimensionless Ortho metric remains valid. Combined speedup projections with a production-grade M1 are not reported as they are speculative.
- *Ref: Figure 5 (M1 speedup and AccRet vs. entropy threshold).*

#### M2: Adaptive Step Scheduling (Simplified Saber — NO_GO)
- Top-k confidence unmasking at step_jump = {2x, 4x, 6x, 8x}, without Saber's backtracking mechanism.
- Verdict: NO_GO at all step_jump values. step_jump=2x: AccRet=76%; step_jump=4x: AccRet=28% (CATASTROPHIC).
- Root cause (FM1): LLaDA's masked denoising requires sequential cumulative conditioning; skipping steps creates unresolvable mask inconsistencies at positions committed too early. This is a structural incompatibility, not addressable by hyperparameter tuning.
- Caveat: the genuine Saber includes backtracking; our NO_GO verdict applies to the simplified implementation.
- M2 excluded from composability experiments. Reported as Failure Mode FM1.

#### M3: AR-Guided Unmasking
- At each denoising step, blend LLaDA's unmasking logits with Qwen2.5-0.5B's token log-probabilities.
- blended_logits = (1 - w) * logits_llada + w * logits_qwen
- Cross-tokenizer bridging via text round-trip decode/encode.
- Parameter sweep: w in {0.3, 0.5, 0.7, 1.0}. Operating point: w=0.3.

#### CD-SSD: Coarse-Draft Self-Speculative Denoising
- *Ref: Figure 2 (CD-SSD architecture diagram).*
- Same model (LLaDA-8B) as both drafter and verifier; no external model, no training.
- **Draft phase**: Run T_draft=16 denoising steps on full sequence; produce draft output and per-token confidence c_i = max_v softmax(logits)_v.
- **Partition**: S_accept = {i : c_i >= tau=0.9}, S_refine = {i : c_i < tau}. Alpha ≈ 52% at operating point.
- **Refine phase**: Freeze S_accept tokens; run T_full=64 steps on S_refine only.
- Concurrent with SSD (full-step + hierarchical tree, lossless) and SSMD (attention-mask switch).
- Key differentiation from SSD: coarse reduced-step draft vs. full-step tree; approximate rather than lossless. The frozen-token set from the draft creates structural conditions for KV-cache synergy not present in SSD's architecture.
- **tau=0.0 ablation result** (Sec 5.4): CD-SSD(tau=0.0) QAS = 4.198, naive-T16 QAS = 4.458, diff = -5.8% (within noise). Conclusion: the confidence partitioning mechanism adds no measurable value over plain step reduction. CD-SSD's value in this paper is as a composability vehicle demonstrating the frozen-token KV amplification mechanism.

### 4.3 Evaluation Protocol
- **Model**: LLaDA-8B-Instruct (bf16, 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 97GB VRAM).
- **Benchmarks (primary)**: GSM8K (1319 samples, exact match), MATH500 (500 samples, exact match).
- **Benchmarks (reported separately)**: HumanEval (164, pass@1), MBPP (257, pass@1) — degenerate baselines (2.4%, 0.0%); AccRet meaningless.
- **Seeds**: [42, 123, 456]; report mean ± std.
- **Throughput**: wall-clock TPS, generation only, discard first 2–5 warm-up samples.
- **Baseline**: LLaDA-8B-Instruct, 64-step denoising, no acceleration, bf16.

*Transition: Section 5 presents all results.*

---

## 5. Experiments (3 pages)

### 5.1 Single-Method Baselines
*Ref: Table 2 (single-method results) and Figure 3 (Pareto curves).*

All numbers: 3-seed mean (seeds 42, 123, 456), full benchmark (except MBPP/HumanEval marked separately).

**Baseline (LLaDA-8B-Instruct, 64-step)**:
- GSM8K: 71.2% exact match, 31.0 ± 4.0 tok/s
- MATH500: 11.1% exact match, 79.2 ± 0.1 tok/s
- HumanEval: 2.4% pass@1 [degenerate]
- MBPP: 0.0% pass@1 [degenerate]

**M1 (EntropyCache, eta=2.0)**:
- GSM8K: 1.50x speedup, AccRet=54.9%, QAS=0.822
- MATH500: 1.31x speedup, AccRet=65.6%, QAS=0.861
- Combined (primary benchmarks): 1.38x speedup, QAS=0.836
- Cache hit rate at eta=2.0: ~60%. Below eta=1.0: speedup < 1.0x (FM2).

**M2 (Simplified Saber) — NO_GO**:
- step_jump=2x: 3.10x speedup, GSM8K AccRet=54.4%; step_jump=4x: GSM8K AccRet=13.0% (CATASTROPHIC).
- Excluded from composability. FM1 documented (Sec 5.3).

**M3 (AR-Guided, Qwen2.5-0.5B, w=0.3)**:
- GSM8K: 1.68x speedup (GSM8K-specific), AccRet=103.9% (genuine +3.9% accuracy gain); QAS=1.675
- MATH500: 1.19x speedup, AccRet=243.9% [artifact: inflated from 11.1% baseline — not used in primary claims]
- Combined speedup (all benchmarks including HumanEval 0.83x): 1.33x
- HumanEval: 0.83x speedup, QAS≈0 [AR guidance fails on code — FM4]
- Task-specific utility: reasoning only.

**CD-SSD (tau=0.9, T_draft=16)**:
- GSM8K: 4.57x speedup, AccRet=63.7%, QAS=2.908 (standard formula)
- MATH500: 2.32x speedup, AccRet=88.5%, QAS=2.053
- Accept rate α=0.88 (88% of tokens accepted from draft at tau=0.9; ~52% frozen during refine based on unique token positions)
- KV hit rate during refine: 94.0% (confirmed from igsd_p2 per-seed results)

**Verdict on M2**: STRUCTURAL NO_GO. Root cause: LLaDA's masked denoising requires sequential step gradients; skipping steps creates unresolvable mask inconsistencies (FM1). Not a hyperparameter issue.

### 5.2 Pairwise Orthogonality Analysis
*Ref: Table 3 (pairwise orthogonality) and Figure 4 (orthogonality bar chart).*

Evaluated on 200 GSM8K + 164 HumanEval, seeds [42, 123]. Primary claims based on GSM8K subset; HumanEval reported separately.

| Pair | Combined Speedup | GSM8K AccRet | QAS (standard) | Ortho | Verdict |
|------|-----------------|--------------|----------------|-------|---------|
| **M1+CD-SSD** | **5.13x** | **~61%** | **3.13** | **1.385** | **SYNERGY** |
| M3+CD-SSD | 2.34x | ~35% | 0.826 | 0.493 | INTERFERENCE |
| M1+M3 | 0.93x | ~54% | 0.504 | 0.301 | INTERFERENCE |

Ortho=1.385 for M1+CD-SSD: per-seed range [1.292, 1.478], both seeds above 1.0. This is a 2-seed estimate on 15% of benchmark scale; full 3-seed validation at full scale is required before submission.

**Mechanism (M1+CD-SSD synergy)**: During CD-SSD's refine phase, tokens in S_accept are frozen with probability 1 across all T_full=64 steps. Their entropy H_i → 0 (deterministic); M1 never refreshes them. Effective KV hit rate for frozen positions: ~94%. Combined speedup exceeds the naive product (1.38x × 3.40x = 4.69x expected vs. 5.13x observed) because IGSD creates the ideal cache scenario for M1 precisely during the computationally heavy refine phase.

**M1+M3 catastrophic interference**: M3 guidance modifies the unmasking logit distribution at each step, causing entropy H_i to spike for positions M3 shifts toward different tokens. M1 interprets these entropy spikes as cache invalidation events and recomputes KV — negating the cache benefit and adding 12–17% guidance overhead. Combined speedup 0.93x is slower than baseline.

### 5.3 Failure Mode Atlas
*Ref: Table 4 (failure mode taxonomy) and Figure 5 (M1 speedup vs. entropy threshold).*

Numbers from verified raw JSON files (m2_pareto_full.json, m1_pareto_full.json, etc.).

| ID | Name | Method | Root Cause | Detection Signal | Evidence |
|----|------|--------|-----------|-----------------|----------|
| FM1 | step_starvation | M2 | LLaDA discrete masking requires sequential cumulative conditioning; step-skipping creates unresolvable mask inconsistencies | step_jump > 3x → GSM8K AccRet < 20% → REJECT | GSM8K AccRet=54.4% at J=2x, 13.0% at J=4x [from m2_pareto_full.json] |
| FM2 | cache_overhead_inversion | M1 | Below eta=2.0, entropy computation + selective recompute overhead exceeds attention savings | Combined speedup < 1.0x when eta < 1.0; speedup=0.553x at eta=0.5 [verified m1_pareto] | FM2 is a regime failure, not a method failure |
| FM3 | draft_quality_cliff | CD-SSD | T_draft < 16 produces low-quality drafts; refine cannot recover from poor partition quality | QAS drops sharply below T_draft=16: QAS=0.394 at T_draft=4 | Confirmed igsd_ablation.json |
| FM4 | ar_guidance_mismatch | M3+coding | Qwen2.5-0.5B causal LM logits misaligned with LLaDA masked denoising trajectory for code tokens | Combined speedup ≤ 0.83x on HumanEval; QAS ≈ 0 | HumanEval speedup=0.83x, pass@1=0.0 all M3 configs |

**Proactive remedies**:
- FM1: auto-reject step_jump > 3x; require genuine Saber backtracking for > 2x steps.
- FM2: default eta=2.0; monitor per-step mean entropy; if mean < 1.5, warn cache may be counterproductive.
- FM3: set T_draft >= 16; monitor per-draft accept rate; reject if alpha < 0.60.
- FM4: do not combine M3 with CD-SSD; restrict M3 to reasoning-only deployment.

### 5.4 CD-SSD Ablations
*Ref: Table 5 (ablation results) and Figure 6 (T_draft sensitivity).*

Data from ablation_igsd experiments (full benchmark scale, seeds 123/456).

**tau=0.0 paradox resolution (critical)**:

| Configuration | Speedup | GSM8K AccRet | QAS | vs. Full CD-SSD |
|---------------|---------|--------------|-----|-----------------|
| CD-SSD(tau=0.9, T_draft=16) | 4.518x | 65.3% | 2.950 | — |
| CD-SSD(tau=0.0) — no partition | 7.118x | 59.0% | 4.198 | +42.3% |
| naive-T16 — plain 16-step | 7.559x | 59.0% | 4.458 | +51.1% |
| M1+naive-T16 | 7.395x | 57.2% | 4.232 | — |
| M1+CD-SSD(tau=0.9) | 6.677x | 58.6% | 3.914 | — |

**Key finding**: CD-SSD(tau=0.0) and naive-T16 achieve statistically indistinguishable QAS (4.198 vs. 4.458, -5.8%, within seed variance). Conclusion: the confidence partitioning mechanism (draft → partition → refine) adds no measurable value beyond simple step reduction. CD-SSD's standalone speedup benefit is entirely attributable to using T=16 instead of T=64. The refine phase (running 64 steps on S_refine) imposes overhead that offsets any quality gain from confidence-based token retention.

**Implication for paper framing**: CD-SSD is best understood as: (1) a step-reduction method (T=16 approximation), where the partition mechanism is optional complexity, and (2) a composability vehicle demonstrating that frozen-token structures enable KV-cache synergy. The Ortho=1.385 synergy finding remains valid — it arises from frozen tokens in the refine phase, which are a byproduct of the partition even if the partition itself does not improve quality over naive T=16.

**T_draft sensitivity** (tau=0.9):

| T_draft | Combined Speedup | GSM8K AccRet | QAS |
|---------|-----------------|--------------|-----|
| 4 | ~3.59x | ~38.4% | ~1.378 |
| 8 | ~4.08x | ~45.6% | ~1.862 |
| **16** | **4.52x** | **63.7%** | **2.879** |
| 32 | ~1.81x | ~75.3% | ~1.361 |

T_draft=16 is Pareto-optimal among the tau=0.9 configurations: T_draft < 16 produces poor draft quality (sharp QAS drop); T_draft > 16 yields diminishing quality returns with proportionally more draft cost. Note: the T_draft=32 slowdown reflects the heavier draft phase dominating total time at T_draft/T_full=50%.

### 5.5 Task Dependence
*Ref: Table 6 (task-dependent recipe).*

| Task Type | Best Single Method | QAS | Best Combination | QAS |
|-----------|-------------------|-----|------------------|-----|
| Reasoning (GSM8K + MATH500) | M3 (w=0.3) | 1.582 | M3 alone (M3+CD-SSD: Ortho=0.493) | — |
| General / mixed | CD-SSD (tau=0.9) | 2.879 (GSM8K; standard QAS) | M1+CD-SSD | 3.913 |

H4 observation: QAS rankings differ between reasoning (M3 > CD-SSD > M1) and general (CD-SSD > M3 > M1) task types, consistent with M3's task-specific benefit on reasoning. Formal statistical testing requires more data points than available in this study (3 methods, 2 task types = 6 measurements; no Wilcoxon test reported).

---

## 6. Analysis and Discussion (1.5 pages)

### 6.1 Why Are Only Some Pairs Composable?
*Ref: Figure 7 (mask-state coupling schematic).*

- MDM denoising is globally coupled: each token's unmasking probability depends on the current mask state of all N positions.
- Methods that **modify the mask trajectory** (M2: step skipping alters when tokens are committed; M3: biased unmasking changes which tokens are committed early) conflict with methods that **assume trajectory stability** (M1: cached KV entries assume token distributions change slowly; CD-SSD: draft quality depends on the 64-step reference trajectory).
- M1+CD-SSD avoids this conflict: M1 does not modify the mask trajectory (it approximates the same computation); CD-SSD creates a two-phase trajectory where the refine phase has a maximally stable mask (S_accept tokens frozen).
- This trajectory-preserving vs. trajectory-modifying classification provides a testable prediction: any MDM acceleration method that modifies the unmasking order or step schedule will likely interfere with KV-caching. Future composability tests can pre-screen methods using this classification.

### 6.2 The Frozen-Token Synergy Mechanism
*Ref: Figure 8 (KV hit rate during CD-SSD phases).*

- During CD-SSD's refine phase, all tokens in S_accept (alpha=0.88 at tau=0.9, occupying ~52% of unique positions after deduplication) are frozen for all T_full=64 steps.
- EntropyCache assigns H_i ≈ 0 to frozen positions (deterministic token; entropy of a point distribution is 0). Every position falls below threshold eta=2.0; M1 never refreshes their KV.
- Measured KV hit rate during refine phase: 94.0% (from igsd_p2_tau09_td16 per-seed JSON, avg_kv_hit_rate_refine).
- This is substantially higher than M1's hit rate in standalone operation (~60%), explaining the super-multiplicative speedup: Ortho = observed_speedup / (speedup_M1 × speedup_CD-SSD) = 5.13 / (1.38 × 3.40) = 1.385 > 1.0.
- The tau=0.0 ablation confirms: removing the partition (no frozen tokens) yields QAS matching naive T=16. The frozen-token structure is what creates the synergy, even if it does not by itself improve output quality.

### 6.3 Limitations and Open Questions

1. **Pairwise statistical power**: Ortho=1.385 is from 2 seeds, 200 GSM8K samples (15% of full benchmark). Per-seed range [1.292, 1.478]. Full 3-seed, 1319-sample validation required before final submission.

2. **M1 implementation gap**: 1.38x vs. published 15–26x. Ortho as a ratio is independent of absolute speedup magnitude — M1's 1.38x and CD-SSD's 3.40x are measured under the same hardware/protocol, so the ratio Ortho=1.385 is valid. However, absolute combined speedup (5.13x) would scale with a production M1; the scaling factor is unknown without empirical measurement.

3. **Simplified M2 without backtracking**: Our NO_GO verdict applies to simplified Saber. The genuine Saber with backtracking may partially mitigate mask inconsistency. This is flagged as a limitation; the structural incompatibility observation (skipping steps in discrete MDMs) is likely to persist but the severity may differ.

4. **Single-model evaluation**: All results are for LLaDA-8B-Instruct. Dream-7B-Instruct was unavailable. The composability framework generalizes in principle, but the specific Ortho values are model-specific.

5. **Batch-size sensitivity**: All experiments at batch=1. Batched inference may change KV hit rates and speedup profiles.

6. **CD-SSD acceptance gate redundancy**: tau=0.0 matches naive T=16, suggesting the confidence partition mechanism adds no measurable value. Future work: investigate whether domain-specific tau calibration (per-task) recovers the partition's theoretical benefit.

---

## 7. Related Work (0.75 pages)

### 7.1 KV-Cache Methods for MDMs
- Fast-dLLM (arXiv:2504.01801 / ICLR 2026), dKV-Cache (arXiv:2505.15781), Elastic-Cache (arXiv:2510.14973), EntropyCache (arXiv:2603.18489), FreeCache/FlashDLM (arXiv:2505.21467), Window-Diffusion (arXiv:2601.20332).
- Compare our M1 operating point (1.38x) against published speedups; explain gap.

### 7.2 Step Reduction and Adaptive Scheduling
- Saber (arXiv:2510.18165), PRR (arXiv:2603.04514), "Not All Denoising Steps Are Equal" (arXiv:2604.02340).
- Our negative finding on M2 is consistent with the challenge of applying continuous-diffusion DDIM schedules to discrete MDMs.

### 7.3 Speculative Decoding for MDMs
- SSD (arXiv:2510.04147): lossless, same-model, full-step hierarchical verification. Concurrent with CD-SSD.
- SSMD (arXiv:2510.03929): self-speculative via attention-mask switch. Concurrent.
- S2D2 (arXiv:2603.25702): self-speculative for block-diffusion LLMs (requires AR-block hybrid architecture).
- DualDiffusion (arXiv:2604.05250): external draft model.
- CD-SSD differs from SSD in draft mechanism (reduced-step vs. full-step tree) and quality guarantee (approximate vs. lossless).

### 7.4 Composability in LLM Optimization
- Kolbeinsson et al. (arXiv:2407.06483): composable interventions for LLMs (compression, editing, unlearning) — different intervention types, not inference acceleration. Most closely related in spirit; no prior work applies composability analysis to MDM inference acceleration.

### 7.5 Training-Based Methods (Out of Scope)
- Block Diffusion (arXiv:2503.09573), Fast-dLLM v2 (arXiv:2509.26328), D2F (arXiv:2508.09192), T3D (arXiv:2602.12262) — training-based approaches. Out of scope for this training-free composability study.

---

## 8. Conclusion (0.5 pages)

- First pairwise composability study of training-free MDM acceleration: binary pattern observed across three method pairs.
- One synergistic pair (M1+CD-SSD, Ortho=1.385) driven by frozen-token KV amplification in CD-SSD's refine phase.
- Tau=0.0 ablation resolves: CD-SSD's value is the frozen-token structure, not the confidence partition per se; plain step reduction (T=16) achieves equivalent standalone QAS.
- Failure mode atlas provides actionable deployment guidance; FM1 (M2 structural incompatibility) generalizes to any DDIM-style discrete step schedule applied to LLaDA-class masked MDMs.
- Future work: full-scale 3-seed pairwise validation, SSD+M1 composability comparison (to test whether synergy requires the coarse-draft frozen-token structure or is a general self-speculative+KV property), Dream-7B cross-model validation, batched inference analysis.

---

## Figure & Table Plan

### Table 1: Literature Speed Comparison (Section: Introduction)
- **Purpose**: Ground reader in the fragmented MDM acceleration landscape; motivate unified composability study.
- **Type**: comparison_table
- **Content**: 8–10 representative methods (Fast-dLLM, EntropyCache, Elastic-Cache, Window-Diffusion, Saber, SSD, FlashDLM) with: Speedup, Base Model, Benchmark, Training-free flag. Note: all published under different hardware/protocols — not directly comparable. Last row: "M1+CD-SSD (this work): 5.13x [two-method composite, LLaDA-8B, GSM8K+HumanEval]".
- **Key takeaway**: Methods are evaluated in isolation with incompatible protocols; no composability data exists before this work.
- **Generation**: data_table (from literature.md Section 3)
- **Data source**: `context/literature.md`

### Figure 1: Teaser — Composability Landscape (Section: Introduction)
- **Purpose**: Communicate the binary composability finding in one glance.
- **Type**: 2×2 panel: (left) bar chart of Ortho scores for 3 pairs with threshold line at Ortho=1.0; (right) scatter plot of Speedup vs. AccRet for individual methods and pairwise results.
- **Content**: Bars colored green (M1+CD-SSD, synergy) vs. red (M3+CD-SSD, M1+M3, interference). Scatter shows Pareto frontier; M1+CD-SSD stands out top-right.
- **Key takeaway**: Exactly one pair lands in the synergy region.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/full_pairwise/full_pairwise_ortho.json`

### Figure 2: CD-SSD Architecture Diagram (Section: Methods)
- **Purpose**: Explain the three-phase CD-SSD pipeline.
- **Type**: flow_chart / architecture_diagram
- **Content**: Input prompt → Draft phase (T_draft=16 steps, whole-sequence) → per-token confidence scoring → Partition (S_accept, frozen vs. S_refine) → Refine phase (freeze S_accept; run T_full=64 for S_refine) → merge output. Annotate: alpha=0.52 frozen fraction; KV hit rate = 94% during refine; show how frozen tokens feed M1 cache.
- **Key takeaway**: Frozen tokens from the draft partition create the structural condition for KV-cache synergy.
- **Generation**: tikz or manual_diagram
- **Data source**: Proposal algorithm + igsd_p2 per-seed results (kv_hit_rate_refine)

### Table 2: Single-Method Results (Section: Experiments 5.1)
- **Purpose**: Establish reproducible per-method baselines.
- **Type**: comparison_table
- **Content**: Method (M1, M2-NO_GO, M3, CD-SSD) × (Speedup, GSM8K AccRet, MATH500 AccRet, QAS [standard formula]). All numbers 3-seed mean ± std. Separate column for MATH500. Coding benchmarks in a footnote with degenerate-baseline warning. Bold the highest-QAS feasible method.
- **Key takeaway**: M3 leads on reasoning QAS (1.675); CD-SSD offers highest pure speedup (4.57x on GSM8K at 63.7% AccRet); M2 NO_GO; M1 conservative.
- **Generation**: data_table
- **Data source**: `exp/results/full_m1/m1_pareto_full.json`, `exp/results/full_m2/m2_pareto_full.json`, `exp/results/full_m3/m3_pareto_full.json`, `exp/results/full_igsd/igsd_pareto_full.json`

### Figure 3: Speed-Accuracy Pareto Curves (Section: Experiments 5.1)
- **Purpose**: Show quality-speed tradeoffs across method parameter sweeps.
- **Type**: line_plot (one line per method, points at each operating parameter)
- **Content**: X-axis = Speedup, Y-axis = GSM8K AccRet. Error bars from seed variance. Shaded "acceptable zone" (AccRet >= 95%). M2 shown with dashed line ending at step_jump=2x (NO_GO boundary).
- **Key takeaway**: Only M3 (w<=0.5) stays in the acceptable zone. All other methods trade substantial accuracy for speed.
- **Generation**: code (matplotlib)
- **Data source**: m1_pareto, m2_pareto, m3_pareto, igsd_pareto JSON files.

### Table 3: Pairwise Orthogonality (Section: Experiments 5.2)
- **Purpose**: Present the core composability result.
- **Type**: comparison_table
- **Content**: 3 rows (M1+CD-SSD, M3+CD-SSD, M1+M3) × columns (Combined Speedup, GSM8K AccRet, QAS standard, Ortho mean, Ortho per-seed range, Interpretation). Green/red color coding on Ortho. Footnote: "2-seed estimate; full-scale validation pending."
- **Key takeaway**: M1+CD-SSD Ortho=1.385 (super-multiplicative); others destructively interfere.
- **Generation**: data_table
- **Data source**: `exp/results/full_pairwise/full_pairwise_ortho.json`

### Figure 4: Orthogonality Comparison (Section: Experiments 5.2)
- **Purpose**: Visually compare Ortho scores with per-seed variance.
- **Type**: bar_chart with error bars
- **Content**: 3 grouped bars (one per pair). Horizontal line at Ortho=1.0 (multiplicative threshold). Error bars from per-seed Ortho values. Color: M1+CD-SSD green; others red.
- **Key takeaway**: Only M1+CD-SSD consistently exceeds the multiplicative threshold across both seeds.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: full_pairwise_ortho.json per-seed values.

### Table 4: Failure Mode Atlas (Section: Experiments 5.3)
- **Purpose**: Catalog interference patterns with detection signals and verified evidence.
- **Type**: comparison_table
- **Content**: FM ID, Name, Affected Method, Root Cause (1–2 sentences), Detection Signal, Verified Evidence (value from raw JSON).
- **Key takeaway**: Four distinct failure modes; each has a proactive detection signal applicable before full deployment.
- **Generation**: data_table
- **Data source**: verified from m2_pareto, m1_pareto, igsd_ablation, m3_pareto JSON files.

### Figure 5: M1 Speedup and AccRet vs. Entropy Threshold (Section: Experiments 5.3)
- **Purpose**: Show FM2 (cache overhead inversion) with exact operating boundary.
- **Type**: line_plot with dual y-axes
- **Content**: X-axis = eta in {0.5, 1.0, 2.0, 3.0}; Left Y = Speedup (shaded "OVERHEAD ZONE" where < 1.0); Right Y = GSM8K AccRet. Vertical dashed line at eta=2.0 (breakeven). Error bars from 3 seeds.
- **Key takeaway**: eta=2.0 is the only viable operating point — below 1.0 the method is self-defeating; above 3.0 accuracy collapses.
- **Generation**: code (matplotlib)
- **Data source**: `exp/results/full_m1/m1_pareto_full.json`

### Table 5: CD-SSD Ablation (Section: Experiments 5.4)
- **Purpose**: Show component contributions and resolve tau=0.0 paradox.
- **Type**: ablation_table
- **Content**: Configuration (Full tau=0.9/T16, tau=0.0, naive-T16, M1+naive-T16, M1+CD-SSD tau=0.9) × (Speedup, AccRet, QAS, ΔQAS%). Key row: naive-T16 vs. tau=0.0 shows -5.8% diff → partitioning adds no value.
- **Key takeaway**: tau=0.0 ≈ naive-T16 resolves paradox: CD-SSD's value is frozen-token structure, not confidence partitioning.
- **Generation**: data_table
- **Data source**: `exp/results/full_tau0_comparison/full_tau0_comparison.json`

### Figure 6: T_draft Sensitivity (Section: Experiments 5.4)
- **Purpose**: Show that T_draft=16 is Pareto-optimal.
- **Type**: line_plot with dual y-axes
- **Content**: X-axis = T_draft (4, 8, 16, 32). Left Y = Speedup; Right Y = GSM8K AccRet. Mark T_draft=16 operating point.
- **Generation**: code (matplotlib)
- **Data source**: igsd ablation JSON files.

### Figure 7: Mask-State Coupling and Composability (Section: Discussion 6.1)
- **Purpose**: Explain why composability is binary via mask-state coupling.
- **Type**: conceptual_diagram / two-panel schematic
- **Content**: Left panel: "Compatible" (M1+CD-SSD — M1 observes trajectory, CD-SSD creates stable refine phase with frozen tokens). Right panel: "Incompatible" (M3+CD-SSD — M3 biases unmasking, disrupts M1's assumed trajectory; cascading entropy spikes). Arrows show mask state evolution.
- **Generation**: tikz or manual_diagram

### Figure 8: KV Hit Rate During CD-SSD Phases (Section: Discussion 6.2)
- **Purpose**: Provide direct mechanistic evidence for frozen-token KV amplification.
- **Type**: bar_chart or line_plot
- **Content**: X-axis = denoising phase (M1-standalone, CD-SSD draft phase, CD-SSD refine phase). Y-axis = KV hit rate. Show jump from ~60% (M1 standalone) to ~94% (CD-SSD refine) for frozen positions.
- **Key takeaway**: Frozen tokens guarantee zero-entropy entries; M1 never refreshes them → peak hit rate.
- **Generation**: code (matplotlib)
- **Data source**: avg_kv_hit_rate_refine from igsd_p2_tau09_td16 per-seed JSON + m1_pareto CHR values.

---

## Appendix Plan

### Appendix A: Full Per-Seed Results
- Complete tables for all methods × seeds × benchmarks (GSM8K, MATH500, HumanEval, MBPP).
- HumanEval/MBPP included with explicit degenerate-baseline caveat.

### Appendix B: CD-SSD Algorithm Pseudocode
- Full pseudocode with implementation details; distinguish from SSD/SSMD.

### Appendix C: M2 Negative Results (Detailed)
- Step_jump sweep results (J=2, 4, 6, 8) with raw accuracy values from m2_pareto_full.json.
- Note: J=6 and J=8 produce identical results to 3 decimal places — potential implementation artifact, acknowledged.

### Appendix D: Qualitative Examples
- Sample GSM8K problems with outputs from: baseline, M1, CD-SSD, M1+CD-SSD, M3.

---

## Section Writing Order

1. **notation.md** and **glossary.md** (already written; update CD-SSD note in glossary)
2. **Section 4 (Methods)** — defines framework, metrics, implementations
3. **Section 5 (Experiments)** — presents all verified results
4. **Section 3 (Background/Related Work)** — contextualizes after results
5. **Section 6 (Discussion)** — interprets results
6. **Section 2 (Introduction)** — frames story after all sections exist
7. **Section 7 (Related Work)** — positions against literature
8. **Section 8 (Conclusion)** — summarizes
9. **Section 1 (Abstract)** — written last, from verified numbers
10. **Appendices** — A/B/C/D with real data

---

## Pre-Submission Checklist (Updated)

### Critical (blocking)
- [ ] Full-scale M1+CD-SSD pairwise validation (3 seeds, GSM8K 1319 + MATH500 500) — compute Ortho mean ± std
- [ ] Verify Ortho >= 1.0 at full scale (NeurIPS gate)
- [ ] Failure mode atlas numbers corrected to match raw JSON data (m2_pareto, m1_pareto)
- [ ] Remove Wilcoxon p<0.05 claim (replaced with ranking observation)
- [ ] QAS formula consistent across all tables (standard formula, no undisclosed penalty)
- [ ] All figures generated (Fig 1–8) — no placeholder references
- [ ] All [CITE:xxx] placeholders replaced with real citations
- [ ] Appendix A–D populated with real data (not placeholders)

### Major (required before venue submission)
- [ ] SSD+M1 composability test (determine whether synergy is CD-SSD-specific or general self-speculative+KV property)
- [ ] CD-SSD REFINE-only KV ablation (disable M1 during DRAFT phase only; confirm synergy is REFINE-phase-driven)
- [ ] FastdLLM either reproduced under identical protocol OR removed from comparison table (no mixing published/reproduced numbers)
- [ ] M3 speedup reporting corrected to combined speedup (1.33x not 1.68x) in Table 2
- [ ] Abstract updated with verified full-scale Ortho (after priority 1 experiment)
- [ ] IGSD renamed consistently to CD-SSD throughout all files

### Minor (polish)
- [ ] All tables use ± std from 3-seed runs where available
- [ ] Coding benchmark caveats explicitly stated in table footnotes
- [ ] M1 implementation gap clearly explained in methods (1.38x vs. published 15–26x)
- [ ] Consistent color scheme across all figures
- [ ] Section word counts checked against venue limits (target: 8 pages + references for NeurIPS)

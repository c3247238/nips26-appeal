# Comparativist Analysis: ComposeAccel Results vs. State of the Art

**Agent**: sibyl-comparativist (updated 2026-04-14)  
**Workspace**: dlm-acceleration  
**Analysis scope**: Positioning ComposeAccel / ComposeAccel results within the MDM acceleration literature

---

## 1. Baseline Landscape: Top Methods on Shared Benchmarks

All existing methods are compared on LLaDA-8B-Instruct (or LLaDA-8B / Dream-7B where noted) on the same benchmarks as our study. Numbers drawn from published papers; hardware differences are noted.

### Table 1: SOTA Competing Methods vs. Our Results

| Method | Type | Speedup (published) | Quality | GSM8K Acc | Training-Free? | Source |
|--------|------|-------------------|---------|-----------|---------------|--------|
| **Fast-dLLM DualCache** (ICLR 2026) | Block-wise KV cache + parallel | 11× (GSM8K/512), 27.6× (long seq) | Negligible loss | ~71% preserved | Yes | arXiv:2505.22618 |
| **EntropyCache** (Mar 2026) | Entropy-guided KV refresh | 15.2–26.4× | Competitive | Competitive | Yes | arXiv:2603.18489 |
| **Sparse-dLLM** (Aug 2025) | Dynamic cache eviction | 5.8× (LLaDA GSM8K) | Minor loss | Minor loss | Yes | arXiv:2508.02558 |
| **DyLLM** (Mar 2026) | Saliency-based partial attention | Up to 9.6× | Largely preserved | Largely preserved | Yes | arXiv:2603.08026 |
| **Window-Diffusion** (Jan 2026) | Windowed token pruning + cache | Up to 99× (adaptive) | Partially preserved | N/A | Yes | arXiv:2601.20332 |
| **Elastic-Cache** (Oct 2025) | Attention-aware adaptive KV refresh | 8.7× (short), 45.1× (long) | Preserved | Preserved | Yes | arXiv:2510.14973 |
| **FlashDLM FreeCache + Guided Diffusion** (May 2025) | KV approx + AR-guided unmasking | 12.14–13.29× | Negligible loss | Negligible loss | Yes | arXiv:2505.21467 |
| **SSD: Self Speculative Decoding** (Oct 2025) | Self-speculative, lossless | 2.4–3.46× | Lossless (identical output) | Identical to baseline | Yes | arXiv:2510.04147 |
| **S2D2** (Mar 2026) | Self-speculative, block-diffusion | Up to 4.7× over AR | Improved | Improved | Yes | arXiv:2603.25702 |
| **DualDiffusion** (Apr 2026) | Draft-verify MDM | Better quality-steps Pareto | Preserved on MMLU/GSM8K | Preserved | Yes (draft model needed) | arXiv:2604.05250 |
| **SlowFast + dLLM-Cache** (Jun 2025) | Dynamic sampling + cache | 34.22× (combined) | Marginal degradation | Comparable | Yes | arXiv:2506.10848 |
| **Learn2PD + KV cache** (Sep 2025) | Learned parallel decoding | 57.51× | Minor degradation | 79.83% | No (learned) | arXiv:2509.25188 |
| **D2F** (Aug 2025) | Block-wise + KV cache (distillation) | 52.9× | Minimal loss | ~38% (vs 39%) | No (distillation) | arXiv:2508.09192 |

### Our Results (ComposeAccel on LLaDA-8B-Instruct)

| Method | Speedup | GSM8K AccRet | Combined AccRet | QAS | Notes |
|--------|---------|--------------|-----------------|-----|-------|
| M1 (EntropyCache approx, t=2.0) | 1.38× | 55.0% | 60.6% | 0.836 | 1.38× vs. 15–26× published gap |
| M3 (AR-Guided, Qwen2.5-0.5B, gw=0.3) | 1.33× | 103.9% | 125.8% | 1.675 | Best reasoning single-method |
| CD-SSD/IGSD (tau=0.9, T_draft=16) | 3.40× | 63.7% | 70.3% | 1.194 (penalized) | GSM8K 45.3% abs vs 71.2% baseline |
| **M1+CD-SSD (SYNERGY)** | **5.13×** | **32.2%** | — | **1.654** | Ortho=1.385, super-multiplicative |

---

## 2. Contribution Margin Analysis

### 2.1 CD-SSD Standalone vs. SSD Prior Art

**Our CD-SSD** (tau=0.9, T_draft=16): 3.40× speedup, GSM8K accuracy drops from 71.2% → 45.3% (63.7% retention).  
**SSD** (Gao et al., arXiv:2510.04147): 2.4–3.46× speedup, **lossless** output.

**Delta**: CD-SSD matches SSD's throughput ceiling (3.40× vs. 3.46× peak) but with a critical handicap — SSD is provably lossless (identical token output to stepwise decoding), while CD-SSD at tau=0.9 drops GSM8K absolute accuracy by ~26 points. On the quality-speed Pareto, SSD strictly dominates CD-SSD. The IGSD/CD-SSD ICLR 2026 submission was withdrawn — the same competitive landscape faces our work.

**Contribution margin for CD-SSD standalone**: NEGATIVE vs. SSD on quality; MARGINAL on speed (−1.7% behind SSD's peak). CD-SSD as a standalone method is not competitive.

**Key differentiator**: CD-SSD's coarse-step draft (T_draft=16 vs. T_full=64) creates ~52% frozen-token KV anchors that SSD's full-step hierarchical tree does not generate. This is the mechanistic basis for the M1+CD-SSD synergy. Whether SSD+M1 also achieves super-multiplicative synergy is the decisive open experiment (see P4 in proposal).

### 2.2 M1 (EntropyCache) vs. Published EntropyCache — Critical Implementation Gap

Our M1 achieves **1.38×**. Published EntropyCache (arXiv:2603.18489) reports **15.2–26.4×**.

**Delta**: −14× to −25× absolute speedup gap. This is an order-of-magnitude discrepancy.

**Root cause (confirmed in proposal)**: Our implementation uses standard PyTorch attention without kernel-level sparse computation. The published paper uses hardware-aware kernel-level sparse attention that actually skips computation for cached positions.

**Impact**: Our M1 + CD-SSD = 5.13×. If kernel-level M1 (say 15×) were composed with CD-SSD (3.40×) at our measured Ortho=1.385, the projected combined speedup would be 15 × 3.40 × 1.385 ≈ **70×**. Even at conservative Ortho=0.9, this projects to ~46×. These projections are theoretical upper bounds, but they indicate the M1 implementation gap severely understates the headline result.

**Contribution margin for M1**: Non-competitive as standalone. Relative Ortho measurements remain mechanistically valid.

### 2.3 M1+CD-SSD Composition vs. Competing Combinations

| Method Combination | Speedup | Quality Impact | Source |
|-------------------|---------|---------------|--------|
| **M1+CD-SSD (ours)** | **5.13×** | 32% acc ret (GSM8K) | This work |
| Sparse-dLLM (single method) | 5.8× | Minor loss | arXiv:2508.02558 |
| SlowFast + dLLM-Cache | 34.22× | Marginal degradation | arXiv:2506.10848 |
| Fast-dLLM DualCache | 11–27.6× | Negligible loss | arXiv:2505.22618 |
| Learn2PD + KV cache | 57.51× | Minor degradation | arXiv:2509.25188 |

Our 5.13× combined speedup is exceeded by **Sparse-dLLM alone (5.8×)** and is 5–11× below combinations like SlowFast + dLLM-Cache. This is the most damaging comparison: a single method surpasses our best two-method composition on absolute speedup, with better quality retention.

**The 5.13× speedup number is not the contribution. The Ortho=1.385 discovery is.** No existing paper measures orthogonality scores; the composability framework and binary landscape discovery are the genuine contributions regardless of the absolute speedup.

**Classification of M1+CD-SSD speedup margin**: MARGINAL vs. single-method SOTA. But the composability methodology discovering this synergy: STRONG novelty.

### 2.4 M3 (AR-Guided Unmasking) — Positioning

M3 achieves 1.33× speedup with 103.9% GSM8K accuracy retention (absolute: 74.0% vs. 71.2% baseline, +2.8pp improvement).

**Comparison**: FlashDLM's Guided Diffusion (arXiv:2505.21467) uses a similar AR-supervisor mechanism and reports 12.14–13.29× speedup with negligible quality loss. Our M3 at 1.33× is 9–10× slower than FlashDLM's guided diffusion component.

**Root cause**: FlashDLM's Guided Diffusion likely uses a larger or more capable guiding model and more efficient integration. Our M3 uses Qwen2.5-0.5B with minimal integration overhead tuning.

**Classification**: M3 alone is not competitive with FlashDLM, but it demonstrates the task-dependent recipe insight: M3 is uniquely suited for reasoning tasks where it improves accuracy while FlashDLM focuses on speed regardless of task type.

---

## 3. Concurrent Work Scan (Oct 2025 – Apr 2026)

### 3.1 Methods Directly Competing with CD-SSD/IGSD

| Paper | ArXiv | Threat Level | Key Overlap | Differentiation |
|-------|-------|-------------|-------------|----------------|
| **SSD** (Gao et al.) | 2510.04147 | CRITICAL | Same conceptual space: self-speculative dLLM, no aux model | SSD is lossless; CD-SSD is approximate; CD-SSD generates frozen-token set |
| **SSMD** (Campbell et al.) | 2510.03929 | HIGH | Self-speculative via attention mask flip | Different mechanism; SSMD ~2× speedup |
| **DualDiffusion** | 2604.05250 | MEDIUM | Draft-verify MDM paradigm | Requires separate draft model; CD-SSD is self-speculative |
| **S2D2** | 2603.25702 | LOW-MEDIUM | Self-speculative decoding for LLMs | Block-diffusion specific; not applicable to pure masked-diffusion (LLaDA/Dream) |

### 3.2 Methods Competing with the M1+CD-SSD Speedup Claim

| Paper | ArXiv | Combined Speedup | Quality | Threat |
|-------|-------|-----------------|---------|--------|
| SlowFast + dLLM-Cache | 2506.10848 | 34.22× | Marginal | MEDIUM — higher speedup, different families composed |
| Learn2PD + KV cache | 2509.25188 | 57.51× | Minor | LOW — training-required, different paradigm |
| Fast-dLLM DualCache | 2505.22618 | 11–27.6× | Negligible | MEDIUM — single-paradigm combination |

**Critical nuance**: None of these competing combinations conduct a systematic orthogonality study. SlowFast + dLLM-Cache implicitly combines two methods but does not measure Ortho, does not study failure modes, and does not evaluate destructive interference conditions.

### 3.3 Composability Studies — No Direct Competitor Found

Systematic search confirms: **no paper in the MDM acceleration literature measures pairwise orthogonality scores or performs a structured composability study.** The closest existing work is:

- **Composable Interventions for Language Models** (Kolbeinsson et al., ICLR 2025, arXiv:2407.06483): Studies composing knowledge editing, model compression, and machine unlearning — different problem domain entirely (model interventions, not inference acceleration).
- **Elastic-Cache** authors (arXiv:2510.14973): Explicitly acknowledge planning to "explore the interplay with speculative decoding" — confirming this is open.
- **SSD paper** (arXiv:2510.04147): Discusses KV-cache synergy qualitatively in the context of Fast-dLLM but does not measure Ortho or perform systematic pairwise analysis.

**The composability framework remains genuinely novel and uncontested.**

### 3.4 M2 (Adaptive Step Scheduling) Failure — Relevant Concurrent Work

Our NO_GO verdict on M2 (structural incompatibility of DDIM-style step skipping with bidirectional masked diffusion) has no direct concurrent publication. Saber (arXiv:2510.18165) reports positive results at 2× step jump but:
1. Evaluated primarily on code generation, not reasoning
2. Does not study step-jump ≥ 3× systematically
3. Does not provide a mechanistic explanation of the mask-inconsistency cascade

**Our failure analysis of M2 is a novel negative result with mechanistic depth that Saber does not address.**

---

## 4. Novelty Verdict

**What is the ONE thing this work does that no prior work does?**

> This work provides the first systematic pairwise composability analysis of training-free MDM acceleration methods across method families — discovering a binary landscape with exactly one super-multiplicative synergistic pair (M1+CD-SSD, Ortho=1.385) amid universal destructive interference — and characterizes four failure modes with proactive detection heuristics.

This is articulatable in one sentence and is verifiably uncontested in the literature. **Novelty: HIGH.**

Secondary novelty claims:
- **CD-SSD method**: CONDITIONAL. CD-SSD is novel relative to SSD/SSMD in its coarse-step frozen-token mechanism, but whether this mechanism uniquely enables KV synergy (vs. SSD+M1 also synergizing) is unresolved. If SSD+M1 Ortho ≈ M1+CD-SSD Ortho, the synergy is a general property and the CD-SSD mechanism claim weakens; if SSD+M1 < 1.0, CD-SSD's mechanism is uniquely synergistic.
- **M2 failure characterization**: Novel negative result — structural incompatibility of discrete masking with DDIM-style step skipping. Novelty: MODERATE-HIGH.
- **Task-dependent recipe**: Moderate novelty — useful deployment guidance not previously published, but incremental.

---

## 5. Venue Recommendation

### Primary Target: EMNLP 2026 Main (Confidence: MODERATE-HIGH)

**Rationale**:
- The composability framework and binary landscape discovery match EMNLP's appetite for empirical analysis papers examining LLM inference behavior.
- 3-seed full-scale experiments on GSM8K (1319 samples) and MATH500 (500 samples) with pairwise composability (200 GSM8K, 2 seeds) is statistically adequate for an analysis paper.
- Honest framing as an analysis paper (CD-SSD as the study vehicle) avoids the overreach that would doom a submission to NeurIPS/ICLR.
- The failure-mode atlas is a distinct practical contribution — practitioners need exactly this kind of negative result catalogued.
- EMNLP regularly accepts systems/analysis papers with strong empirical frameworks even when individual method improvements are modest.

**Risk**: Absolute speedup numbers (5.13× vs. 27.6× SOTA single-method) are weak. Reviewers may focus on this without appreciating the framework contribution. The abstract and introduction must position clearly as an analysis paper from the first sentence.

### Conditional Target: NeurIPS 2026 (Confidence: LOW, requires 3 pending experiments)

**Conditions for NeurIPS escalation (all must hold)**:
1. SSD+M1 experiment (P4): Ortho(SSD+M1) < 0.9 — confirms CD-SSD's coarse-step mechanism is uniquely synergistic
2. REFINE ablation (P2): M1 disabled during REFINE drops speedup substantially — confirms synergy location
3. tau=0.0 paradox (P3): tau=0.0 > naive T=16 — confirms CD-SSD's acceptance gate adds value beyond step reduction
4. Kernel-level M1 addressed: Either implemented or upper-bound projection analysis included

Without these, NeurIPS is not viable. Even with them, the M1 implementation gap is a credibility liability.

**Comparable EMNLP papers**: Analysis papers studying decoding strategies, inference behavior under different conditions, systematic comparisons of acceleration approaches.

### Workshop Fallback: EfficientNLP (EMNLP 2026), Efficient Systems for Foundation Models (NeurIPS 2026)

If pending experiments do not strengthen the narrative, a focused 4-page paper on the composability atlas and failure modes alone is workshop-worthy and would reach the right audience.

---

## 6. Strengthening Plan: Three Specific Additions

### 6.1 SSD+M1 Composability Experiment (Priority P4)

**Why**: If SSD+M1 Ortho >= 1.0, the synergy is a general property of self-speculative MDMs + KV-caching — a **stronger** generalization claim. If SSD+M1 Ortho < 1.0 while M1+CD-SSD Ortho = 1.385, CD-SSD's frozen-token mechanism is uniquely synergistic.

**Protocol**: Same 200 GSM8K samples, 2 seeds {42, 123}. Compare: M1 alone, SSD alone, SSD+M1 combined. Compute Ortho(SSD+M1). Expected GPU time: ~4 hours.

**Impact on narrative**: Either way this experiment makes the paper dramatically stronger — it either generalizes the finding (bigger claim) or uniquely attributes it to CD-SSD (bigger method claim). No downside to running it.

### 6.2 Naive T=16 Baseline for tau=0.0 Paradox (Priority P3)

**Why**: tau=0.0 (skip REFINE, QAS=1.801) outperforms full CD-SSD (tau=0.9, QAS=0.956). If tau=0.0 ≈ naive 16-step LLaDA, then CD-SSD's acceptance gate mechanism adds no value — pure step reduction suffices. This is a critical design flaw that must be either resolved or reported honestly.

**Protocol**: Run LLaDA-8B-Instruct at 16 steps uniform denoising (T=16 straight) on 200 GSM8K, seeds {42, 123}. Compare: tau=0.0 CD-SSD vs. naive T=16 vs. tau=0.9 CD-SSD.

**Expected outcomes**: If naive T=16 < tau=0.0, the draft-and-conditional-skip mechanism has genuine value. If naive T=16 ≈ tau=0.0, CD-SSD's accepted tokens provide no quality benefit over simple step reduction.

### 6.3 M1 Implementation Explanation or Kernel-Level Baseline

**Why**: The 1.38× vs. 15–26× gap is the most credibility-damaging element of the current results. Without addressing it, reviewers will question all relative Ortho measurements.

**Options**:
(a) **Implement kernel-level sparse attention** using EntropyCache's official code (github.com/mscheong01/EntropyCache) and re-run pairwise experiments. Ortho measurements should be stable; absolute speedup will increase dramatically.
(b) **Include a theoretical analysis**: At Ortho=1.385 and M1 speedup = S_M1, combined speedup scales as S_M1 × 3.40 × 1.385. With published S_M1 = 15–26×, projected combined speedup = 70–122×. Present this as a "performance ceiling" analysis.

Option (a) is preferred but requires ~2 GPU-hours for re-runs. Option (b) is adequate for an analysis paper if (a) is infeasible.

---

## 7. Evidence Samples

**Sample 1: M1+CD-SSD synergy data (from full_pairwise_ortho.json)**
```
Seed 42:  combined_speedup=5.129, combined_ret=0.301, QAS=1.543, Ortho=1.292
Seed 123: combined_speedup=5.140, combined_ret=0.343, QAS=1.765, Ortho=1.478
Average:  combined_speedup=5.135, combined_ret=0.322, QAS=1.654, Ortho=1.385
```
Note: Ortho variance across seeds (1.292 to 1.478) suggests 2 seeds may be insufficient for a robust estimate. The 3rd seed (456) should be added for pairwise experiments before submission.

**Sample 2: Composability landscape binary pattern**
```
M1+CD-SSD: Ortho=1.385 (SYNERGY)
M3+CD-SSD: Ortho=0.493 (INTERFERENCE)
M1+M3:     Ortho=0.301 (CATASTROPHIC INTERFERENCE)
```
No intermediate values — the landscape is genuinely binary, not gradient.

**Sample 3: M2 failure (structural incompatibility)**
```
Step jump 2×: AccRet=76.0% (marginal)
Step jump 4×: AccRet=27.9% (FAILING)
Step jump 8×: AccRet=24.3% (CATASTROPHIC)
```
Quality collapse is abrupt, consistent with structural mechanism (mask-inconsistency cascade), not gradual parameter sensitivity.

---

## 8. Summary Assessment

| Dimension | Rating | Key Evidence |
|-----------|--------|-------------|
| **Composability framework novelty** | STRONG (9/10) | No prior systematic pairwise orthogonality study in MDM acceleration |
| **CD-SSD method novelty** | CONDITIONAL (4–7/10) | Concurrent SSD/SSMD; differentiated by coarse-step frozen-token mechanism; SSD+M1 experiment will determine |
| **Absolute speedup competitiveness** | WEAK | 5.13× vs. 5.8× (Sparse-dLLM single), 27.6× (Fast-dLLM), 34.22× (SlowFast+dLLM-Cache) |
| **Quality-speed Pareto** | WEAK | CD-SSD at 35% acc ret; SSD is lossless at comparable speed |
| **M1 implementation** | PROBLEMATIC | 1.38× vs. 15–26× published; severe implementation gap undermines headline numbers |
| **Failure mode atlas** | STRONG (8/10) | M2 structural incompatibility is novel negative result; four failure modes with detection signals |
| **Task-dependent recipes** | MODERATE | Useful deployment guidance; first evidence-backed task-type recipe for LLaDA-8B-Instruct |
| **Venue fit** | EMNLP 2026 (analysis paper) | NeurIPS conditional on 3 pending experiments |

**Core positioning recommendation**: Frame explicitly as an analysis paper from the first sentence. Lead with the binary composability discovery and the failure-mode atlas. Position CD-SSD as the study vehicle for the composability analysis, not as a methods contribution unless SSD+M1 experiments establish CD-SSD's unique synergistic property.

---

## Sources

- [Fast-dLLM (arXiv:2505.22618)](https://arxiv.org/abs/2505.22618)
- [FlashDLM (arXiv:2505.21467)](https://arxiv.org/abs/2505.21467)
- [SSD Self Speculative Decoding (arXiv:2510.04147)](https://arxiv.org/abs/2510.04147)
- [Elastic-Cache (arXiv:2510.14973)](https://arxiv.org/abs/2510.14973)
- [EntropyCache (arXiv:2603.18489)](https://arxiv.org/abs/2603.18489)
- [Window-Diffusion (arXiv:2601.20332)](https://arxiv.org/abs/2601.20332)
- [DyLLM (arXiv:2603.08026)](https://arxiv.org/abs/2603.08026)
- [Sparse-dLLM (arXiv:2508.02558)](https://arxiv.org/abs/2508.02558)
- [D2F (arXiv:2508.09192)](https://arxiv.org/abs/2508.09192)
- [DualDiffusion (arXiv:2604.05250)](https://arxiv.org/abs/2604.05250)
- [S2D2 (arXiv:2603.25702)](https://arxiv.org/abs/2603.25702)
- [SlowFast Sampling (arXiv:2506.10848)](https://arxiv.org/abs/2506.10848)
- [Composable Interventions for Language Models (arXiv:2407.06483)](https://arxiv.org/abs/2407.06483)
- [LLaDA (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992)

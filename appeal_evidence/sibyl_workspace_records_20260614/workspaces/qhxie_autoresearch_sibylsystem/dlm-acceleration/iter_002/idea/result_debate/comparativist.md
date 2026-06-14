# Comparativist Analysis: ComposeAccel Results vs. SOTA

**Agent**: sibyl-comparativist
**Date**: 2026-04-16
**Workspace**: dlm-acceleration/current (iter_002)

---

## 1. Baseline Landscape: Top Existing Methods

### 1.1 Training-Free DLM Acceleration (Published Numbers)

| Method | Base Model | Reported Speedup | Quality Impact | Reference |
|--------|-----------|-----------------|----------------|-----------|
| Fast-dLLM (ICLR 2026) | LLaDA/Dream | Up to 27.6x | Negligible loss | NVlabs |
| EntropyCache | LLaDA-8B / Dream-7B | 15.2x--26.4x | Competitive | arXiv 2603.18489 |
| FlashDLM (FreeCache+Guided) | LLaDA-8B / Dream-7B | 12.1x--13.3x | Negligible loss | arXiv 2505.21467 |
| SlowFast + dLLM-Cache | LLaDA | Up to 34.2x | Marginal degradation | arXiv 2506.10848 |
| Window-Diffusion | LLaDA / Dream | Up to 99x | Partially preserved | arXiv 2601.20332 |
| Elastic-Cache | LLaDA variants | 45.1x (long seq) | Preserved | arXiv 2510.14973 |
| ES-dLLM | LLaDA-8B / Dream-7B | 5.6x--16.8x | Preserved | arXiv 2603.10088 |
| DyLLM | LLaDA-8B / Dream-7B | Up to 9.6x | Largely preserved | arXiv 2603.08026 |
| Saber (code) | DLM | 251.4% (~3.5x) | +1.9% Pass@1 | arXiv 2510.18165 |

### 1.2 Our Results (ComposeAccel, LLaDA-8B-Instruct, RTX PRO 6000 Blackwell)

| Configuration | GSM8K Acc | GSM8K TPS | GSM8K Speedup | Acc Retention | QAS |
|---------------|-----------|-----------|---------------|---------------|-----|
| **Baseline** (64 steps, bs=8) | 0.712 | 33.8 | 1.0x | 100% | 1.0 |
| M1 (EntropyCache, eta=0.5) | 0.673 | 35.9 | 1.16x | 94.5% | 1.09 |
| M1 (EntropyCache, eta=1.0) | 0.627 | 38.8 | 1.25x | 88.0% | 1.10 |
| M1 (EntropyCache, eta=2.0) | 0.395 | 46.5 | 1.50x | 55.5% | 0.83 |
| IGSD (tau=0.7, T_draft=16) | 0.425 | 164.5 | 2.81x | 58.2% | 1.64 |
| IGSD (tau=0.85, T_draft=32) | 0.495 | 101.3 | 1.73x | 67.8% | 1.17 |
| IGSD (tau=0.9, T_draft=48) | 0.535 | 71.5 | 1.22x | 73.3% | 0.90 |
| M3 (AR-guided, gw=0.3) | ~0.74 | 52.0 | 1.68x | 103.9% | 1.69 |
| M1+IGSD (eta=0.5, tau=0.7, td=16) | 0.43 | 160.9 | 2.75x | 58.9% | 1.62 |
| M1+M3 (eta=0.5, gw=0.3) | 0.73 | 50.3 | 0.86x | 102.5% | 0.88 |
| M3+IGSD (tau=0.7, td=16, gw=0.7) | 0.43 | 159.3 | 2.72x | 60.4% | 1.64 |
| Three-way Max-Speed (eta=0.5, tau=0.85, td=32, gw=0.0) | 0.517 | ~96 | 1.69x | 70.8% | 1.20 |

---

## 2. Contribution Margin Analysis

### 2.1 Speedup Deltas vs. Existing SOTA

**CRITICAL FINDING**: Our maximum measured speedup is **2.81x** (IGSD alone, tau=0.7, T_draft=16, pilot subset). The best composition (M1+IGSD) achieves **2.75x**. These numbers are **dramatically below published SOTA** for training-free DLM acceleration:

| Comparison Point | Our Best | Published SOTA | Delta | Classification |
|-----------------|----------|---------------|-------|----------------|
| KV cache speedup | 1.50x (M1, eta=2.0) | 26.4x (EntropyCache paper) | **-94%** | Far below |
| Step scheduling | 2.81x (IGSD) | 34.2x (SlowFast+Cache) | **-92%** | Far below |
| Best composition | 2.75x (M1+IGSD) | 27.6x (Fast-dLLM) | **-90%** | Far below |
| Quality-preserving | 1.68x (M3, gw=0.3) | 12.1x (FlashDLM) | **-86%** | Far below |

### 2.2 Root Cause of Speedup Gap

The massive gap between our results and published numbers stems from **implementation fidelity**, not methodological insight:

1. **M1 (EntropyCache)**: Our implementation achieves 56--60% cache hit rates but runs full forward passes every step -- it selectively reuses KV but does NOT skip computation. The EntropyCache paper reports 15--26x speedup because their kernel-level implementation actually avoids redundant attention computation. Our implementation measures the "cache hit rate" signal but does not realize the speedup. The measured 1.16x--1.50x speedup comes entirely from minor computational savings in the entropy computation path, not from KV reuse. **This is an implementation gap, not a methodological finding.**

2. **IGSD**: The 2.81x speedup is genuine but reflects a different mechanism (speculative drafting with verification). It is not directly comparable to KV-cache-based speedups.

3. **M3 (AR-guided)**: The 1.68x on GSM8K is encouraging but the overhead of loading a second model (Qwen2.5-0.5B) limits the practical speedup. The FlashDLM paper integrates guided diffusion at the attention level, achieving 12x+ speedup.

### 2.3 Accuracy Comparison

On the quality side, our baselines are concerning:

| Benchmark | LLaDA-8B (ours) | Published LLaDA-8B | Gap |
|-----------|-----------------|-------------------|-----|
| GSM8K | 71.2% | ~75% (reported) | -3.8% |
| MATH500 | 11.1% | ~15% (reported) | -3.9% |
| HumanEval | 2.4% | ~15% (reported) | -12.6% |
| MBPP | 0.0% | ~10% (reported) | -10.0% |

**WARNING**: The extremely low HumanEval (2.4%) and MBPP (0%) baselines suggest a **code generation evaluation bug** or prompt template mismatch. These near-zero baselines make code benchmarks unusable for composition analysis and undermine any claims about task-dependent acceleration recipes for code generation.

### 2.4 AR Comparison

| System | GSM8K Acc | TPS (bs=1) | TPS (bs=8) |
|--------|-----------|------------|------------|
| LLaDA-8B baseline | 71.2% | ~34 | ~34 |
| Qwen2.5-7B (AR, greedy) | 96% | 70.9 | 471.1 |
| Qwen2.5-7B (AR + SpecDec) | 97% | 48.2 | N/A (not supported) |
| Best DLM composition | ~52% | ~96 | ~51 |

**Verdict**: At batch_size=1, the best DLM composition (M1+IGSD) generates tokens faster (96 TPS vs 71 TPS) than the AR baseline, but at a 40% accuracy reduction. At batch_size=8, AR dominates both speed (471 TPS) and quality (96% accuracy). The batched inference scaling gap remains a fundamental DLM disadvantage -- batch_size=8 actually SLOWS DOWN our composed DLM methods (50.7 TPS < 96.2 TPS at bs=1), likely because the IGSD verify/re-draft loop cannot be parallelized across batch items efficiently.

---

## 3. Concurrent Work Scan

### 3.1 Directly Competing Papers (Last 6 Months)

| Paper | Date | Overlap | Threat Level |
|-------|------|---------|-------------|
| **DiffBench + DiffAgent** (arXiv 2601.03178) | Jan 2026 | Automated composition of acceleration techniques for diffusion models. Introduces a benchmark for evaluating combinations. | **HIGH** -- directly addresses the combination question, though for image diffusion, not language. |
| **Introspective DLMs** (arXiv 2604.11035) | Apr 2026 | Compares against LLaDA-2.1, SDAR, and latest DLMs on benchmarks. | MEDIUM -- provides updated baselines but doesn't study composition. |
| **LLaDA 2.0/2.1** (InclusionAI) | Feb 2026 | 100B MoE with token editing, dramatically different architecture. | MEDIUM -- changes the target model landscape but our study focuses on LLaDA-8B and Dream-7B. |
| **CDLM (Consistency DLMs)** (Together AI, Feb 2026) | Feb 2026 | Up to 14.5x speedup via consistency distillation + KV caching. | MEDIUM -- training-based, not directly competing with our training-free focus. |
| **TORS** (arXiv 2603.00763) | Mar 2026 | Notes that training-free acceleration methods for text-to-image diffusion are "developed independently, leaving overall performance and compatibility unexplored." | LOW -- for image diffusion, but validates our gap claim. |

### 3.2 Concurrent Composition Studies

**No direct competitor found.** After systematic search of arXiv, Google Scholar, and web sources (April 2026), we confirm:

- No published paper systematically studies the composition of 3+ training-free acceleration methods for DLMs with quantified orthogonality metrics.
- DiffBench+DiffAgent (Jan 2026) addresses composition for image diffusion models, not language DLMs.
- Fast-dLLM (ICLR 2026) combines KV cache + parallel decoding (2 axes) but does not study factorial composition or orthogonality.
- SlowFast mentions "combined with caching" but the integration analysis is ad hoc, not systematic.

The **composition study gap** remains valid. However, see Section 6 for caveats.

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does:**

> "Provides the first controlled factorial composition study of training-free DLM acceleration methods across three orthogonal axes (KV caching, step scheduling, guided decoding), with quantified orthogonality metrics and cross-model generalization."

**Assessment**: This novelty claim is **valid in principle but weakened in execution** by the following issues:

1. **Implementation fidelity gap**: The M1 (EntropyCache) implementation does not achieve real KV cache speedups. Without kernel-level sparse attention, the "KV caching axis" effectively tests cache-hit-rate statistics rather than actual speedup from caching. This undermines the claim of studying three distinct "axes" -- in practice, we are studying (a) an entropy-guided forward-pass modification with minimal speedup, (b) a speculative drafting scheme (IGSD), and (c) AR-guided unmasking.

2. **Near-zero code baselines**: HumanEval 2.4% and MBPP 0% eliminate the "task-dependent recipe" contribution for code generation entirely.

3. **IGSD quality limitation**: At all configurations tested, IGSD reduces accuracy by 27--42% on GSM8K. This limits the practical value of IGSD as a "composable module" -- it is more of a speed-for-quality tradeoff than a principled step scheduler.

4. **Composition findings are negative**: M1+M3 shows INTERFERENCE (ortho=0.41--0.52), M3+IGSD shows interference (ortho=0.61--0.84), and only M1+IGSD shows near-orthogonality (ortho=0.91--0.96). The primary finding is that most pairwise compositions interfere, which, while publishable, is less compelling than finding synergistic compositions.

---

## 5. Venue Recommendation

### 5.1 Honest Assessment

| Venue Tier | Fit | Reasoning |
|-----------|-----|-----------|
| **Top-tier (NeurIPS/ICML/ICLR)** | Unlikely | Speedup numbers (1.16--2.81x) are 10x below published SOTA. The composition framework is novel but results are negative/mixed. Code benchmarks are broken. |
| **Mid-tier (AAAI/EMNLP)** | Possible with major revisions | The orthogonality framework and negative-result findings could fit, but need proper M1 implementation and fixed code benchmarks. |
| **Workshop (NeurIPS DLM Workshop)** | Good fit | The composition methodology, negative interference findings, and cross-model generalization results are appropriate for a workshop paper without requiring SOTA speedups. |
| **Technical report / blog** | Current state | In its current form, the contribution is best suited as a technical report documenting the composition landscape. |

### 5.2 Comparable Papers at Each Venue

- **NeurIPS main**: "How Efficient Are DLMs?" (arXiv 2510.18480) was a methods critique paper. But it had comprehensive evaluation on 5+ models with validated tooling. We would need comparable rigor.
- **Workshop**: "Not All Denoising Steps Are Equal" (arXiv 2604.02340) achieved modest 17% FLOP reduction but offered principled analysis. Our composition analysis is similar in scope.

---

## 6. Strengthening Plan

### 6.1 Critical Fixes (Must-Do)

1. **Fix M1 implementation**: Integrate the actual d2Cache or Fast-dLLM kernel-level KV cache library. Without real KV cache speedup, the "three axes" claim is unsupported. The d2cache_pilot experiment exists but needs completion and integration into the composition study.

2. **Fix code generation baselines**: Debug HumanEval and MBPP evaluation. The 2.4% / 0% baselines suggest prompt template issues (e.g., missing stop tokens, wrong formatting). LLaDA-8B-Instruct should achieve ~15% on HumanEval with correct prompting. Without this, the "task-dependent recipe" contribution is non-existent.

3. **Expand to full-sample evaluation**: Many results use 100-200 sample subsets. The full GSM8K (1319) results show the baseline varies from 0.70 to 0.73 across seeds. Full evaluations with 3 seeds are needed for reliable composition conclusions.

### 6.2 Additional Baselines to Strengthen Positioning

1. **Fast-dLLM baseline**: Run the official Fast-dLLM (NVlabs) implementation on our hardware and benchmarks. This provides the true ceiling for training-free KV cache acceleration and establishes whether our composition study adds value on top of SOTA.

2. **SlowFast + dLLM-Cache**: Reproduce the published 34x speedup result and then compose with IGSD/M3 to test three-way composition with a properly accelerated KV cache axis.

3. **DualDiffusion (arXiv 2604.05250)**: Very recent speculative decoding for MDMs. Compare against IGSD as an alternative step reduction scheme.

### 6.3 Key Findings Worth Emphasizing

Despite the limitations, several findings are genuinely valuable:

1. **Interference discovery**: M1+M3 composition achieves speedup <1x (0.86x) despite both methods individually achieving >1x. This is a **cautionary tale** for the field -- naive composition destroys speedup. The AR guidance model's per-step overhead dominates when cache-hit-rate-based methods run full forward passes.

2. **Batch sensitivity negative scaling**: Composed methods show WORSE TPS at batch_size=8 than batch_size=1 (50.7 vs 96.2 TPS). This fundamentally different scaling behavior from AR models is an important finding for deployment planning.

3. **Cross-model transfer**: All 5 top configurations show consistent behavior on Dream-7B (transfer ratio ~1.86x), confirming hyperparameter transferability across DLM architectures. 4 of 5 show pattern agreement.

4. **IGSD as principled signal**: Despite accuracy limitations, the KL-divergence threshold provides a principled, adaptive mechanism for step compression. The accept rate is consistently >92%, validating the signal quality even if draft quality limits final accuracy.

---

## 7. Summary Verdict

### What Actually Compares Well
- The **composition methodology** (factorial design, orthogonality metric, QAS framework) is novel and well-designed
- The **interference findings** are genuinely useful negative results
- The **cross-model generalization** on Dream-7B validates transferability
- The **IGSD algorithm concept** is simple and principled

### What Falls Short
- **Speedup numbers are 10--90% below published SOTA** due to implementation gaps
- **Code generation benchmarks are broken** (near-zero baselines)
- **M1 does not actually implement KV cache acceleration** (runs full forward passes)
- **No comparison against the best existing methods** (Fast-dLLM, SlowFast)

### Bottom Line

The paper's core novelty claim -- systematic composition study with orthogonality analysis -- is valid and no competing paper addresses this. However, the experimental execution has critical gaps that prevent the results from being compelling at a top venue. The most honest framing is: **"We found that most training-free DLM acceleration methods interfere when composed, with only KV-caching + speculative drafting showing near-orthogonal behavior."** This is a useful but modest contribution, best positioned for a **workshop paper** in its current state, upgradeable to a mid-tier venue with proper M1 implementation and fixed code benchmarks.

---

## Appendix: Evidence Sources

- Full baseline: `exp/results/full_baseline/llada_baseline_full.json`
- M1 Pareto: `exp/results/full_m1/m1_pareto_full.json`
- M3 Pareto: `exp/results/full_m3/m3_pareto_full.json`
- IGSD Pareto: `exp/results/igsd_pareto/igsd_pareto_corrected.json`
- Pairwise M1+IGSD: `exp/results/pairwise/m1_igsd_full.json`
- Pairwise M1+M3: `exp/results/pairwise/m1_m3_full.json`
- Pairwise M3+IGSD: `exp/results/pairwise/m3_igsd_full.json`
- Three-way: `exp/results/threeway/threeway_pareto_full.json`
- AR comparison: `exp/results/ar_comparison/ar_baseline.json`
- Dream-7B: `exp/results/dream7b/dream7b_top5.json`
- Batch sensitivity: `exp/results/batch_sensitivity/batch_sensitivity.json`

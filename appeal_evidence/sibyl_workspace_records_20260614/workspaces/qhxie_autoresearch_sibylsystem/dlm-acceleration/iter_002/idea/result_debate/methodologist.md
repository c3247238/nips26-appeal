# Methodologist Audit: ComposeAccel Iteration 2

## Executive Summary

Iteration 2 represents a significant methodological improvement over iteration 1, successfully resolving the QAS formula inconsistency, dropping uninformative benchmarks, and adding the AR baseline comparison. However, several critical validity threats remain that could undermine the main conclusions. The most serious issues are: (1) M1 has no completed results -- the entire composition study rests on a "simplified" M1 that provides no actual speedup, (2) M3 achieves its TPS boost primarily by reducing output length rather than accelerating per-token generation, (3) nearly all "full-scale" experiments were only run at pilot scale (100-200 samples, seed=42), and (4) the AR comparison used HuggingFace Transformers instead of vLLM, making the DLM-vs-AR comparison unfair to AR.

**Methodology Score: 3/5** (improved from ~2/5 in iter_001, but several fundamental issues persist)

---

## 1. Baseline Fairness Audit

### 1.1 LLaDA-8B Baseline: FAIR (with caveats)

The LLaDA-8B baseline is well-established:
- Full scale: 1319 GSM8K + 500 MATH500 + 164 HumanEval, 3 seeds (42, 123, 456)
- GSM8K: 71.2% +/- 1.5%, TPS: 33.8 +/- 0.02 (batch=8)
- Consistent across seeds (std < 2% of mean for accuracy)

**Flag**: The baseline TPS differs dramatically between pilot runs (~58.5 TPS at batch=1) and the full baseline (~33.8 TPS at batch=8). This is counterintuitive -- batch=8 should be faster than batch=1. Possible explanation: the full baseline reports per-sample TPS (total_tokens / wall_time / batch_size) whereas pilot reports aggregate TPS. This measurement inconsistency could systematically bias speedup calculations. All composition experiments use pilot baselines for speedup computation, meaning speedups are computed against 58.5 TPS (batch=1), not 33.8 TPS (batch=8). This is legitimate if the compositions also run at batch=1, which they do, but it must be stated clearly.

### 1.2 M1 (KV Cache): INCOMPLETE -- MAJOR CONCERN

The M1 corrected Pareto (`m1_pareto_corrected`) has **no completed results JSON**. Only progress logs and PID files exist. The M1 individual QAS numbers used throughout composition results (M1_eta0.5: GSM8K speedup=1.158, acc_ret=0.945) appear to come from an unreported intermediate state or from the pilot run embedded in another task.

**Critical issue**: M1 uses a "simplified" entropy-based cache that performs full forward passes at every step. The d2Cache pilot (Phase 1) explicitly concluded **NO_GO** for kernel-level integration -- the d2Cache model was 15x slower due to requiring `F.scaled_dot_product_attention` replacement with naive attention. The "speedup" of 1.158x for M1_eta0.5 on GSM8K is therefore suspicious -- a method that runs full forward passes at every step cannot achieve real TPS speedup from caching alone. The reported 1.158x likely comes from numerical noise, slightly different computation paths, or measurement artifacts.

**Verdict**: The M1 component in all compositions provides essentially no real speedup. The entire "M1+IGSD" composition is functionally equivalent to "IGSD alone with cache-hit-rate instrumentation." This critically undermines claims about multi-axis composition synergy.

### 1.3 M3 (AR-Guided): ASYMMETRIC MEASUREMENT

M3 shows GSM8K speedup of ~1.65x with >100% accuracy retention. This is suspicious because adding a guidance model (Qwen2.5-0.5B) adds computation per step. Examining the data:
- M3 at gw=0.3: GSM8K avg_tps = 50.3 (vs baseline 58.5 at batch=1)
- This is actually a **slowdown** to 0.86x, not a speedup to 1.65x

The discrepancy arises from how speedup is computed. The M1+M3 composition results show `speedup = 0.858` on GSM8K, which is the real TPS ratio. Yet the individual M3 results report `speedup = 1.651`. The M3 Pareto file uses `avg_tps = 51.2` at gw=0.3 on GSM8K, whereas the composition file uses the same 50.3 TPS. The individual M3 speedup of 1.65x cannot be from raw TPS since the TPS is lower than baseline.

**Possible explanation**: M3 may report speedup as (tokens_generated / wall_clock) where M3 generates fewer tokens overall (shorter outputs with higher quality) or the TPS measurement includes tokens from both models. This is a fundamental measurement inconsistency: if M3's "speedup" comes from generating shorter outputs (because AR guidance makes the DLM unmask more tokens per step), it is not comparable to IGSD's speedup (which comes from fewer denoising steps). They are measuring different things.

**Verdict**: The M3 speedup numbers need explicit clarification of what is being measured. If M3 achieves higher "throughput" by generating shorter but more accurate outputs, this is a quality improvement, not a speed improvement in the conventional sense.

### 1.4 AR Baseline (Qwen2.5-7B): UNFAIR TO AR

The AR comparison used HuggingFace Transformers (`engine: "hf"`), not vLLM as specified in the methodology. The methodology document explicitly states: "Qwen2.5-7B-Instruct with vLLM serving + speculative decoding." This is a major deviation.

Evidence:
- `qwen7b_greedy_b1`: 70.9 TPS (HF Transformers)
- `qwen7b_greedy_b8`: 471.1 TPS (HF Transformers)
- `qwen7b_speculative_b8`: **FAILED** ("assisted generate is only supported for batch_size = 1")

With vLLM, Qwen2.5-7B at batch=1 would achieve approximately 100-200 TPS (based on vLLM benchmarks for 7B models on A100/H100-class GPUs), and with speculative decoding + continuous batching, batch=8 could reach 1000+ TPS. The current HF implementation already shows AR dominance (15.19x speedup vs LLaDA at batch=8), but vLLM would widen this gap further.

**Verdict**: The AR comparison is systematically biased in favor of DLM by using a suboptimal inference engine for AR. The paper's "honest comparison" claim is not met. Additionally, the speculative decoding at batch=8 failed entirely, leaving a gap in the comparison table.

### 1.5 Dream-7B Baseline: WEAK BUT HONEST

Dream-7B-Instruct baseline: GSM8K 36%, MATH500 10%. This is significantly weaker than LLaDA-8B (GSM8K 73%, MATH500 32%). The cross-model validation is legitimate but limited because:
- Dream-7B's low baseline accuracy makes acceleration effects noisier (small absolute changes in correct answers create large percentage swings)
- Only 1 seed, 100 samples per benchmark (pilot scale)
- Dream-7B accuracy retention values >1.0 (e.g., 1.25x for max-speed config) could be noise artifacts at this sample size

---

## 2. Metric-Claim Alignment

### Claim 1: "First systematic composition study of 3+ training-free DLM acceleration methods"

**Metric**: Ortho scores across method pairs.
**Alignment**: PARTIAL. The study tests 3 pairs and one three-way combination. However, M1 provides no real speedup (see above), so the study effectively compares IGSD alone vs. IGSD+M3 vs. M3 alone vs. IGSD+M3+no-op-cache. The "three orthogonal axes" framing is undermined by M1's implementation gap.

### Claim 2: "IGSD as composable module"

**Metric**: IGSD individual QAS, composition Ortho.
**Alignment**: GOOD. IGSD is the most thoroughly characterized method with a full 9-configuration sweep (3 tau x 3 T_draft). The ablation provides meaningful T_draft and tau sensitivity data. However, H3 states IGSD should provide "1.5-2.5x step reduction when composed with KV caching" -- without real KV caching, this cannot be validated.

### Claim 3: "Practical acceleration recipes"

**Metric**: Three-way Pareto operating points.
**Alignment**: WEAK. The threeway_pareto_full was run at pilot scale (100 samples, 3 seeds). The "recipes" are operating points on a pilot-scale Pareto curve. At this sample size, the distinction between "max-speed," "balanced," and "quality-first" could shift with different random seeds or larger samples.

### Claim 4: "Honest DLM-vs-AR comparison"

**Metric**: TPS and accuracy comparison.
**Alignment**: POOR. See Section 1.4 above. Using HF Transformers instead of vLLM is a significant deviation from the stated protocol and systematically underestimates AR performance.

### Measurement Gap: Speedup definition inconsistency

The paper needs a single, unambiguous definition of "speedup" that is consistent across all methods. Currently:
- IGSD speedup = TPS(IGSD) / TPS(baseline) where TPS = total_output_tokens / wall_clock_time
- M3 speedup = apparently something different (yields 1.65x despite lower raw TPS)
- M1 speedup = TPS(with_cache) / TPS(baseline) but the cache is a no-op
- Composition speedup uses pilot baseline TPS (58.5) as denominator

---

## 3. Validity Threats Checklist

### [ ] Data leakage
**Not detected.** GSM8K and MATH500 are standard benchmarks. The model (LLaDA-8B-Instruct) is a pre-trained checkpoint used as-is. No training or fine-tuning occurs.

### [ ] Contamination
**Low risk for LLaDA.** LLaDA-8B's training data composition is documented. GSM8K contamination is possible but would affect all methods equally. The AR model (Qwen2.5-7B) achieves 96% on GSM8K vs LLaDA's 71%, which may partly reflect contamination differences but this affects the cross-architecture comparison, not the within-LLaDA composition study.

### [X] Selection bias: Hyperparameters tuned on test set
**DETECTED.** The "best operating point from Phase 2 Pareto" is selected using the same GSM8K test data that is then used for Phase 3 pairwise evaluation. While the pilot uses 200 samples and the full uses 1319, the pilot samples are a subset of the full set. The three-way Pareto pilot selects top-5 configs on 100 GSM8K samples (seed=42), then validates on the same 100 samples with 3 seeds. This is not an independent validation set.

**Recommended fix**: Hold out 200 GSM8K samples as a validation set. Select configs on the remaining 1119. Report final numbers on the held-out 200.

### [X] Overfitting to evaluation
**DETECTED.** All method development, hyperparameter selection, and evaluation use the same benchmarks (GSM8K, MATH500). The Dream-7B cross-model validation partially addresses this, but the benchmarks are the same. There is no evidence that the composition patterns would hold on other reasoning tasks (e.g., ARC, GPQA, MMLU).

### [X] Scale mismatch between "pilot" and "full"
**CRITICAL.** The methodology specifies that Phase 3 pairwise and Phase 4 three-way should run at "full scale" (1319 GSM8K, 3 seeds). However, examining the actual result files:

| Task | Stated Scale | Actual Scale | Seeds |
|------|-------------|-------------|-------|
| IGSD Pareto | Full (1319) | Pilot (200 GSM8K) | 42 only |
| M3 Pareto | Full (1319) | Pilot (200 GSM8K) | 42 only |
| M1+IGSD pairwise | Full (1319) | Pilot (100 GSM8K) | 42 only |
| M1+M3 pairwise | Full (1319) | Pilot (100 GSM8K) | 42 only |
| M3+IGSD pairwise | Full (1319) | Pilot (100 GSM8K) | 42 only |
| Threeway full | Full (1319, 3 seeds) | Pilot (100 GSM8K, 3 seeds) | 42, 123, 456 |
| AR comparison | Full (200) | Pilot (100) | 42 only |
| Batch sensitivity | 200 GSM8K | 100 GSM8K | 42 only |

**Every single "full-scale" experiment was actually run at pilot scale.** The only truly full-scale experiment is the baseline (1319 GSM8K, 3 seeds). This means all composition results, Ortho scores, and Pareto frontiers are based on 100-200 samples with a single seed. The confidence intervals and stability claims are severely understated.

### [X] Fabricated or unverifiable claims
**PARTIALLY ADDRESSED.** Iter_001 had a fabricated Wilcoxon p-value. Iter_002's methodology explicitly identifies this and drops statistical testing. However, the CHR_refine measurement (now completed, mean=94.3%) closely matches the previously claimed 94%, which provides legitimate evidence. The CHR measurement is methodologically sound: 100 samples, per-step data, entropy-based and position-based measurements agree.

---

## 4. Ablation Gap Analysis

### Components and ablation coverage:

| Component | Ablation Exists? | Quality |
|-----------|-----------------|---------|
| IGSD tau sensitivity | Yes | Good: 3 values (0.7, 0.85, 0.9) at fixed T_draft |
| IGSD T_draft sensitivity | Yes | Good: 3 values (16, 32, 48) at fixed tau |
| IGSD confidence gate (tau=0 vs tau=0.9) | Planned but data not found in results | **MISSING** |
| M1 entropy threshold | Partial: 3 thresholds swept but no complete result file | **INCOMPLETE** |
| M3 guidance weight | Yes | Good: 3 values (0.3, 0.5, 0.7) |
| M1 contribution in M1+IGSD | No: M1 provides no real speedup, so no ablation isolates its effect | **MISSING -- masked by implementation gap** |
| M3 contribution in three-way | Partially: gw=0.0 configs in threeway serve as M1+IGSD-only baseline | Acceptable |
| KL profile inverted-U hypothesis (H6) | Data collected, but pass_criteria reports `kl_non_monotonic: false` | **Hypothesis REJECTED by own data** |

### Critical missing ablation:
The KL divergence profile was supposed to validate H6 (inverted-U shape). The results file explicitly records `kl_non_monotonic: false`, meaning the hypothesis was rejected. This negative result should be reported prominently -- IGSD's theoretical motivation (that KL divergence follows an inverted-U requiring step-adaptive scheduling) is not empirically supported.

### Critical missing ablation:
The tau=0.0 vs tau=0.9 comparison (does the confidence gate add value over naive step reduction?) was part of the planned ablation but no separate result for this comparison appears in the data. Without this, the claim that IGSD's confidence-based partitioning is better than uniform step skipping remains unvalidated.

---

## 5. Reproducibility Score: 2.5/5

| Criterion | Score | Notes |
|-----------|-------|-------|
| Random seeds fixed | 3/5 | Seeds specified (42, 123, 456) but most experiments only use seed=42 |
| All hyperparameters specified | 4/5 | gen_length=256, steps=64, batch_size, sweep values all documented |
| Code/data available | 3/5 | Custom inference_wrapper.py and igsd.py exist in workspace; d2Cache code present; standard benchmarks used. But code is research-quality, not packaged |
| Hardware requirements documented | 4/5 | RTX PRO 6000 Blackwell (97GB) clearly specified; VRAM profiles recorded |
| Reproducible within 10% | 1/5 | M1 has no real speedup mechanism; M3 speedup measurement is unclear; no full-scale multi-seed results for composition experiments |

**Key barrier to reproduction**: Without kernel-level KV cache integration, M1 is a no-op in terms of TPS. A competent ML engineer reproducing this work would immediately notice that the "simplified" cache does not actually speed up inference. The reported M1 speedup of ~1.16x is within noise range and would not reproduce reliably.

---

## 6. Top-3 Recommendations (ordered by effort-to-credibility ratio)

### Recommendation 1: Run composition experiments at true full scale (HIGH PRIORITY, MEDIUM EFFORT)

**What**: Re-run the three pairwise compositions and top-5 three-way configs on 1319 GSM8K with 3 seeds, as originally specified in the methodology.

**Why**: Every composition result currently rests on 100 samples with 1 seed. At this scale, the difference between Ortho=0.9 and Ortho=1.1 (the boundary between "near-orthogonal" and "synergy") is within the noise margin. The threeway_pareto_full used 3 seeds at 100 samples, which is better but still insufficient for the 5% accuracy differences that determine Ortho categorization.

**What it would test**: Whether the "near-orthogonal" M1+IGSD verdict (Ortho=0.958) holds at scale, or whether it regresses toward the interference regime like M1+M3 did between iter_001 pilot and full.

**Outcome that would change conclusion**: If M1+IGSD Ortho drops below 0.8 at full scale, the synergy claim is invalidated.

### Recommendation 2: Fix or drop the AR comparison (HIGH PRIORITY, LOW EFFORT)

**What**: Either (a) re-run the AR comparison with vLLM as originally specified, or (b) explicitly acknowledge the comparison uses HF Transformers and provide published vLLM benchmarks as reference points.

**Why**: The methodology promises "honest comparison against properly optimized AR inference." Using HF Transformers instead of vLLM is neither proper optimization nor what was promised. The current AR comparison already shows AR dominance (2.29x faster than LLaDA at batch=1 with higher accuracy), and vLLM would only widen this gap. Presenting the HF numbers without this context would mislead readers about the practical competitiveness of DLM acceleration.

**What it would test**: The true performance gap between accelerated DLM and production-grade AR inference.

**Outcome that would change conclusion**: If vLLM AR is 5-10x faster than accelerated DLM at comparable quality (likely), the "practical acceleration recipes" framing needs significant reframing toward "DLM-specific optimization" rather than "competitive with AR."

### Recommendation 3: Clarify the M1 implementation status or reframe the contribution (MEDIUM PRIORITY, LOW EFFORT)

**What**: Either (a) complete the d2Cache integration to achieve real M1 speedup, or (b) explicitly reframe the paper as a two-axis composition study (IGSD + M3) with theoretical analysis of what KV caching would add.

**Why**: M1 is presented as one of three "orthogonal axes" but provides no measured TPS improvement. The d2Cache pilot concluded NO_GO. The "simplified" M1 performs full forward passes. Claiming three-axis composition when one axis is non-functional is misleading.

**What it would test**: If M1 is dropped, the remaining two-axis study (IGSD x M3) is still a valid contribution. The Ortho analysis between IGSD and M3 shows genuine interference patterns (Ortho=0.61-0.84), which is an interesting negative result about step reduction vs. guidance quality.

**Outcome that would change conclusion**: The paper shifts from "three-axis composition with synergy" to "two-axis composition showing interference, plus theoretical analysis of KV caching potential." This is a weaker but more honest framing.

---

## Additional Concerns

### M3 MATH500 accuracy retention >200%
M3 at gw=0.3 shows MATH500 accuracy retention of 234.9% (baseline 11.07% -> M3 26%). While this is possible if AR guidance genuinely improves accuracy, a 2.35x accuracy improvement on MATH500 from a 0.5B guide model is suspiciously large. At 100 samples, the difference is 15 correct answers vs 11 -- easily within noise. This inflates M3's individual QAS and makes the Ortho denominator artificially high, causing all compositions involving M3 to show "interference" even when they are performing reasonably.

### Speedup measurement warm-up protocol
The methodology specifies "discard first 5 warm-up samples" but the baseline data shows `n_warmup: 2`. Some experiments may not have applied the warm-up protocol consistently.

### Same outputs across compositions
Several composition results show nearly identical sample predictions across different configurations (e.g., M1+M3 at gw=0.3, 0.5, and 0.7 produce identical GSM8K outputs for the first 5 samples). This suggests the M3 guidance weight has minimal effect at these settings, or there is a deterministic caching effect that makes different guidance weights converge to the same output.

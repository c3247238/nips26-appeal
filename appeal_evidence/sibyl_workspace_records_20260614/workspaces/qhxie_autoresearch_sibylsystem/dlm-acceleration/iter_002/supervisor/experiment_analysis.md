# Experiment Result Analysis

## Key Results Summary

### Individual Methods (LLaDA-8B-Instruct, RTX PRO 6000 Blackwell)

| Method | GSM8K Acc | Speedup | Acc Retention | QAS |
|--------|-----------|---------|---------------|-----|
| Baseline (64 steps) | 71.2% | 1.0x | 100% | 1.0 |
| M1 (EntropyCache, eta=0.5) | 67.3% | 1.16x | 94.5% | 1.09 |
| IGSD (tau=0.7, T_draft=16) | 42.5% | 2.81x | 58.2% | 1.64 |
| M3 (AR-guided, gw=0.3) | ~74% | 1.68x | 103.9% | 1.69 |

### Pairwise Compositions (iter_002, pilot-scale 100 samples, seed=42)

| Pair | GSM8K Speedup | Acc Ret | Ortho | Verdict |
|------|---------------|---------|-------|---------|
| M1+IGSD | 2.75x | 58.9% | 0.96 | Near-orthogonal |
| M1+M3 | 0.86x | 102.5% | 0.41-0.52 | Destructive interference (SLOWDOWN) |
| M3+IGSD | 2.72x | 60.4% | 0.61-0.84 | Partial interference |

### Three-way (pilot-scale 100 samples, 3 seeds)

| Config | GSM8K Ortho (mean) | Per-seed range |
|--------|-------------------|----------------|
| Max-Speed (M1+IGSD+M3_gw=0.0) | 1.02 | 0.91-1.11 |
| Quality-First (M1+IGSD+M3_gw=0.3) | 0.49 | -- |

### AR Comparison (using HF Transformers, NOT vLLM)

| System | GSM8K Acc | TPS (bs=1) | TPS (bs=8) |
|--------|-----------|------------|------------|
| LLaDA-8B baseline | 71.2% | ~34 | ~34 |
| Qwen2.5-7B (AR, greedy) | 96% | 70.9 | 471.1 |
| Best composed DLM (M1+IGSD) | ~52% | ~96 | ~51 |

### Cross-Model Transfer (Dream-7B, pilot-scale)

Transfer ratio ~1.86x across 5 configs. Dream-7B baseline GSM8K accuracy is only 36-39%, inflating retention metrics. 4/5 configs show qualitatively consistent patterns.

---

## Debate Perspectives Summary

- **Optimist**: M3 at gw=0.3 is a rare win-win (simultaneous speed + quality improvement). M1+IGSD near-orthogonality (0.96) validates core composition thesis. Cross-model transfer to Dream-7B is highly publishable. Three-way ortho > 1.0 indicates super-additive synergy. Several unexpected signals (M3 as accuracy booster, batch-size-improving accuracy) open follow-up directions.

- **Skeptic**: Three fatal flaws identified. (F1) M1+IGSD "synergy" rests on single-seed pilot of 100 samples with 95% CI spanning both synergy and interference. (F2) IGSD-no-partition (tau=0.0) outperforms full IGSD by 88.5% on QAS, potentially invalidating IGSD's KL-divergence mechanism as a contribution. (F3) M1+M3 shows interference (ortho=0.41-0.52, speedup <1x), directly refuting the "quality-first composition dominates" hypothesis. AR baseline comparison is devastating -- Qwen2.5-7B dominates on both speed and accuracy. QAS metric conflates speed and quality in misleading ways.

- **Strategist**: PROCEED to paper writing with major narrative reframing. The "multiplicative speedup" story is dead; the actual story is "composition reveals pervasive interference." Recommends centering on three claims: (1) interference taxonomy, (2) M3 as practical quality enhancer, (3) honest DLM positioning vs. AR. Begin writing immediately with 80% effort on writing, 20% on order-correction ablation. The data is comprehensive enough (15 experiment groups) despite being mostly pilot-scale.

- **Comparativist**: Speedup numbers (1.16-2.81x) are 10-90% below published SOTA (Fast-dLLM 27.6x, EntropyCache 26.4x, SlowFast 34.2x). The gap is primarily an implementation fidelity issue (M1 runs full forward passes, no kernel-level cache). No competing factorial composition study exists. Code benchmarks (HumanEval 2.4%, MBPP 0%) are broken and unusable. Venue assessment: unlikely for top-tier (NeurIPS/ICML), possible for mid-tier (AAAI/EMNLP) with major revisions, good fit for workshop (NeurIPS DLM Workshop).

- **Methodologist**: Methodology score 3/5. M1 has no completed result files -- its "speedup" of 1.16x is within measurement noise since it runs full forward passes. M3's reported 1.68x speedup may be an artifact of output-length reduction rather than per-token acceleration. Nearly ALL composition experiments were run at pilot scale (100 samples, 1 seed) despite the methodology specifying full scale (1319 samples, 3 seeds). Hyperparameters selected and evaluated on same test data (selection bias). Reproducibility score: 2.5/5.

- **Revisionist**: 4 of 6 original hypotheses refuted. H1 (subadditivity) refuted: ortho ranges 0.41-0.99. H2 (quality-first composition) refuted: M1+M3 interference with net slowdown. H5 (KL sufficient signal) refuted: no threshold achieves iso-accuracy with meaningful step reduction. Revised mental model: "DLM denoising resists modular acceleration" -- methods interact through shared denoising trajectory. Proposes reframing from "how methods compose" to "why methods interfere."

---

## Analysis

### 1. Method Feasibility

**M1 (EntropyCache) does NOT work as an acceleration method.** The kernel-level d2Cache integration was declared NO_GO (15x slower due to SDPA/FlashAttention incompatibility). The "simplified" M1 runs full forward passes at every step and achieves only 1.16x speedup, which is within measurement noise. All six debate perspectives unanimously agree on this. Consequence: the "three orthogonal axes" framing collapses to effectively two axes (step scheduling + AR guidance). M1+IGSD near-orthogonality (ortho=0.96) is mathematically correct but practically misleading -- it reflects absence of interference from a no-op, not genuine composition of two acceleration axes.

**IGSD provides genuine step reduction (2.81x) but with severe quality cost (30-42% GSM8K accuracy degradation).** The KL-divergence signal achieves 89-97% accept rates but misses subtle distributional shifts critical for reasoning. Critically, the tau=0.0 variant (no confidence gating, accept everything) achieves QAS=1.801 vs. full IGSD's QAS=0.956 -- an 88.5% improvement. This raises a fatal question: does IGSD's KL-divergence mechanism add any value over naive step reduction? This ablation remains incomplete.

**M3 (AR-guided unmasking at gw=0.3) is the genuine standout.** It is the only method that simultaneously improves accuracy (+3.9% GSM8K, +135% MATH500) and may improve throughput. However, the Methodologist raises a critical concern: the TPS data shows M3 actually generates tokens *slower* (~50 TPS vs. baseline 58.5 TPS), suggesting the reported 1.68x "speedup" comes from generating shorter/more efficient outputs, not faster per-token generation. This measurement ambiguity must be resolved.

### 2. Performance

**Against internal baselines**: Only IGSD provides substantial speedup (2.81x), at the cost of massive accuracy degradation. M3 provides quality improvement with uncertain speed properties. M1 is effectively inert.

**Against published SOTA**: Our speedup numbers (1.16-2.81x) are 10-90% below published training-free DLM acceleration results (Fast-dLLM 27.6x, EntropyCache 26.4x, FlashDLM 12.1x). This is primarily an implementation gap, not a methodological insight.

**Against AR baselines**: Qwen2.5-7B dominates LLaDA-8B on both speed (2-14x depending on batch size) and accuracy (96% vs. 71%). Even the best composed DLM acceleration cannot close this gap. At batch=8, composed methods actually *slow down* (50.7 TPS vs. 96.2 TPS at batch=1), a fundamental architectural limitation.

### 3. Improvement Headroom

**Limited for the current direction.** The M1 axis is dead without kernel-level implementation (high risk, previously failed). IGSD has a speed-quality tradeoff frontier that cannot be shifted without fundamental algorithmic changes (per-token KL, stronger verifiers -- untested hypotheses). M3 is the most promising, but its interaction with other methods is destructive (M1+M3 ortho=0.41, M3+IGSD ortho=0.61-0.84). The composition story -- the paper's central contribution -- is fundamentally a negative result.

Potential improvements that could shift outcomes:
- M3 speedup measurement clarification (1 GPU hour, critical)
- IGSD tau=0.0 ablation (2 GPU hours, critical for IGSD contribution claim)
- Full-scale replication of compositions (8-10 GPU hours, critical for statistical rigor)
- Order-correction ablation (3-4 GPU hours, could add mechanistic insight)

None of these are likely to change the core finding that composition predominantly causes interference.

### 4. Time-Cost Tradeoff

**The data is comprehensive in scope but insufficient in depth.** 15 experiment groups cover a thorough factorial design, but nearly all are pilot-scale (100 samples, 1 seed). The synthesis gave a quality score of 4/10. To bring the paper to a submittable state requires:

- Critical fixes: 11-13 GPU hours (M3 clarification, IGSD ablation, full-scale compositions)
- Paper reframing and writing: substantial effort but 0 GPU hours
- Estimated total to workshop submission: 2-3 days of computation + writing
- To upgrade from workshop to mid-tier (AAAI/EMNLP): additional 15-20 GPU hours for M1 fix or reframing, vLLM AR comparison, code benchmark fix

**Pivoting to an alternative would cost 8-11 hours for a fresh experimental cycle** but would target a potentially stronger contribution (theoretical convergence bounds, order-first acceleration, or multigrid V-cycle).

### 5. Critical Objections Assessment

**The Skeptic's three fatal flaws are all valid and partially addressable:**

- F1 (unreliable synergy claim): Addressable with full-scale replication, but even if confirmed, M1+IGSD ortho=0.96 is misleading given M1 is a no-op. The claim must be downgraded from "synergy" to "minimal interference from instrumentation overhead."

- F2 (IGSD-no-partition paradox): Addressable with tau=0.0 ablation, but likely outcome is that naive step reduction works equally well. IGSD would need repositioning from "novel algorithm" to "empirical finding."

- F3 (M1+M3 interference): This is a confirmed negative result. Not addressable without kernel-level M1 implementation, which previously failed.

**The AR comparison objection (S1) is the most devastating and is NOT addressable**: even with vLLM (which would show even greater AR dominance), DLM acceleration cannot close the gap. This fundamentally limits the paper's practical impact claim.

---

## Decision Rationale

I weigh the following factors:

**Arguments for PROCEED:**
1. The dataset is genuinely comprehensive in scope (factorial design, 15 experiment groups, cross-model, AR comparison). No competing composition study exists as of April 2026.
2. The interference taxonomy (three distinct regimes) is a novel and useful negative finding. Negative results about composition failure are arguably more valuable than incremental positive results.
3. M3 as a quality enhancer is a genuine finding worth publishing.
4. The strategist, synthesis, and verdict all recommend PROCEED with reframing.
5. Critical remediation experiments (11-13 GPU hours) are feasible and can be completed in parallel with paper writing.

**Arguments for PIVOT:**
1. Quality score is only 4/10. One of the three "axes" (M1) is non-functional.
2. Speedup numbers are 10-90% below published SOTA -- the paper cannot claim acceleration competitive with existing work.
3. Four of six hypotheses were refuted. The paper's original thesis is dead.
4. IGSD's algorithmic contribution is unvalidated (tau=0.0 may be equally good).
5. Venue ceiling is AAAI/EMNLP at best, workshop at floor. Top-tier (NeurIPS/ICML/ICLR) is unlikely.
6. Strong alternatives exist: token-level convergence theory (Alternative 1) has deeper novelty; order-first acceleration (Alternative 2) addresses the root cause identified by the experiments.

**My assessment:**

The experimental evidence tells a clear story, but it is not the story the proposal intended. The original thesis ("composition yields multiplicative speedup") is refuted. What remains is: (a) an interference taxonomy, (b) M3 quality enhancement, and (c) honest DLM positioning. These are modest but publishable contributions -- for a workshop or mid-tier venue.

However, I am concerned about opportunity cost. The alternatives are compelling:
- Alternative 2 (Order-First Acceleration) directly builds on the key insight from this iteration -- that the *order* of unmasking may be the fundamental bottleneck -- and has higher novelty potential.
- Alternative 1 (Token-Level Convergence Theory) offers deeper theoretical novelty and would be the first per-token convergence analysis for DLMs.

The critical question is: **is the reframed interference-taxonomy paper worth the 11-13 additional GPU hours and significant writing effort for a workshop/mid-tier publication, when alternatives with higher novelty ceilings exist?**

I judge yes -- but only if the team can write this paper efficiently as an "honest empirical study" without over-investing. The data exists. The reframing is clear. The critical experiments (M3 clarification, IGSD ablation, full-scale replication) are well-defined and bounded. The alternatives require starting over with new implementations and do not leverage the existing 15-experiment dataset.

**However**, I attach two conditions:
1. If the IGSD tau=0.0 ablation confirms that naive step reduction works equally well (IGSD adds no value), the IGSD algorithmic contribution must be dropped entirely, and the paper becomes a pure "empirical composition survey + M3 quality finding" -- this weakens the paper further and should trigger reconsideration.
2. If full-scale replication shows M1+IGSD ortho dropping below 0.80 (entering interference territory for all pairs), the composition narrative collapses entirely, and a PIVOT should be reconsidered.

DECISION: PROCEED

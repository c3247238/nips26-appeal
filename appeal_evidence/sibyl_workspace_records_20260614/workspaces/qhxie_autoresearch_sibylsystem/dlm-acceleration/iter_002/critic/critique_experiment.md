# Experiment Critique -- ComposeAccel iter_002

## Overall Assessment: 4/10

The experimental coverage is wide (15 experiment groups, cross-model, AR comparison) but shallow. Nearly all composition experiments -- the paper's central contribution -- are pilot-scale (100 samples, 1 seed). The methodology explicitly planned full-scale (1319 GSM8K, 3 seeds) validation that was never executed for pairwise compositions. Two of three acceleration methods have fundamental measurement issues (M1 is a no-op, M3 speedup confounds per-token speed with output length). The ablation that should have been first (IGSD vs. naive truncation) confirms the null hypothesis. Sample outputs reveal degenerate generation patterns that are never analyzed.

---

## Critical Issues

### C1: Pilot-Scale Composition Data for Central Claims

The paper's core contribution is the Ortho taxonomy: M1+IGSD near-orthogonal, M3+IGSD task-dependent, M1+M3 interference. Every one of these verdicts is based on 100-sample, single-seed pilots.

**Evidence of unreliability**: The methodology notes the M1+M3 discrepancy between iter_001 pilot (Ortho=1.339) and iter_001 full-scale (Ortho=0.301). This 4.4x discrepancy between pilot and full-scale is concrete proof that 100-sample Ortho estimates are unreliable. The iter_002 pilot (Ortho=0.41) is closer to the iter_001 full-scale, but we cannot know whether iter_002 full-scale would confirm 0.41 or reveal another surprise.

**Sample size calculation**: At 100 samples with ~45% accuracy (IGSD configurations), the standard error is sqrt(0.45*0.55/100) = 5.0 percentage points. A 95% CI on accuracy is +/-10pp. This propagates to +/-22% relative error on AccRet (10/45 = 0.22), which cascades into the Ortho calculation.

**Specific violations of the planned methodology**:
- Phase 3 (methodology.md): "Full scale: 1319 GSM8K + 500 MATH500, seeds=[42, 123, 456]" for each pair. Actually executed: 100 GSM8K, seed=42 only.
- Phase 2 (methodology.md): "Full (per method): 1319 GSM8K + 500 MATH500 + 164 HumanEval, seeds=[42, 123, 456]". M1 was run at full scale (N=1319); IGSD and M3 were not (N=200 and N=100 respectively).
- Table 3 mixes N=1319, N=200, and N=100 results in the same table without adequate visual distinction.

### C2: M1 Speedup Is Within Measurement Noise

M1 at eta=0.5 achieves 1.16x measured speedup. The "simplified" implementation runs full forward passes and applies entropy monitoring as a diagnostic overlay -- it does NOT skip any computation. The 16% TPS increase likely comes from caching overhead amortization or variance in GPU scheduling, not from actual KV cache reuse.

**d2Cache failure details**: The d2Cache native model runs at 3.85 TPS vs. HuggingFace baseline at 58.5 TPS -- a 15.2x framework OVERHEAD. d2Cache's internal speedup (4.4x over its own no-cache baseline) is real, but the framework overhead completely negates it. The paper correctly reports this, but then proceeds to compute M1+IGSD Ortho using M1's 1.16x measured speedup as if M1 were a functioning acceleration method.

**Impact on all M1-involving claims**:
- M1+IGSD Ortho=0.96: This is the Ortho between "IGSD + a 16% overhead wrapper." Not meaningful as a composition result.
- M1+M3 Ortho=0.41: The "destructive interference" is actually "M3 slowdown + M1 near-zero acceleration." This is expected, not a finding.
- Three-way compositions with M1: All use M1 eta=0.5, which contributes ~1.16x. The three-way results are effectively two-way (IGSD+M3) with M1 monitoring.

### C3: M3 Speedup Measurement Ambiguity

The M3 pareto full results (m3_pareto_full.json) show avg_tps=52.04 at gw=0.3 on GSM8K, while baseline avg_tps=58.50. M3 is SLOWER per token than baseline (52/58.5 = 0.89x). Yet the paper reports M3 speedup as 1.68x.

**The discrepancy**: The 1.68x figure likely comes from the "speedup" field computed as (M3 TPS / iter_001 baseline TPS), where iter_001 used a different measurement protocol (31.0 TPS baseline vs. iter_002's 58.5 TPS). Alternatively, M3 generates shorter outputs that complete faster per sample, inflating the TPS metric.

**Evidence from data**: M3 gw=0.3 achieves 73% accuracy on 100 GSM8K samples (same as baseline) with avg_tps=50.3 (m1_m3_full.json, M1+M3 gw=0.3 GSM8K data). This is SLOWER than baseline 58.5 TPS. If M3 were truly 1.68x faster, we would see ~98 TPS.

**Root cause**: The baseline TPS varies across experiments: 58.5 (d2cache pilot), 31.0 (iter_001 full), 33.8 (paper text). These inconsistencies suggest measurement protocol differences (generation-only vs. end-to-end, warmup handling, prompt processing inclusion). The M3 "speedup" may be computed against the slowest baseline variant.

This ambiguity cascades into: M3 QAS calculations, M1+M3 Ortho calculations, M3+IGSD Ortho calculations, and the combined Pareto frontier.

---

## Major Issues

### M1: Degenerate Outputs Not Analyzed

Sample outputs from multiple experiments show clear degenerate generation:

1. **Repetition loops**: Sample id=4 in m1_igsd_full.json: "Mariah ske ske ske ske ske ske ske ske ske ske ske ske ske ske ske ske..." (64+ repetitions). Same pattern in igsd_ablation_refined.json. Accept_rate for this sample is 0.48, indicating IGSD correctly identifies it as low-confidence, but the draft phase already corrupted it.

2. **Truncated reasoning**: Sample id=3 in m1_igsd_full.json: "To determine how much James spends on candles, we need to know the total number candles he wants = $ on candles." This is a premature answer with nonsensical formatting.

3. **Whitespace flooding**: Sample id=4 in pairwise/m1_igsd_full.json MATH500: "1. Let n find the greatest possible value of gcd and 2n+1" followed by 136+ newlines.

These degenerate outputs are counted as "incorrect" in the exact_match metric but their nature is qualitatively different from a well-formed wrong answer. The paper treats all incorrect answers identically. A method that produces 40% coherent-but-wrong answers is qualitatively better than one producing 20% coherent-wrong + 20% degenerate.

### M2: IGSD tau Insensitivity Undermines Confidence Gate Claim

The tau sweep at T_draft=32 shows:
- tau=0.7: AccRet=66.4%, QAS=1.17
- tau=0.85: AccRet=67.8%, QAS=1.17
- tau=0.9: AccRet=67.8%, QAS=1.16

The 1.4pp spread across the entire tau range is within sampling noise of 200 samples. Combined with the tau=0.0 ablation (identical accuracy to tau=0.9), this establishes that IGSD's confidence partitioning has NO sensitivity to its primary hyperparameter at T_draft=32.

The paper correctly reports this ("IGSD is insensitive to tau in the [0.7, 0.9] range") but does not draw the obvious conclusion: the confidence gate is inert. The explanation ("the monotonic KL profile explains why") is post-hoc rationalization of a null result.

### M3: Batch Sensitivity Results Are Uninterpretable

Section 5.3 reports M1+IGSD batch sensitivity:
- batch=1: 1.64x speedup, 96 TPS
- batch=4: accuracy 50%, TPS 56
- batch=8: accuracy 52%, TPS 34

**Missing information**: What is the baseline TPS at batch=4 and batch=8? Without this, "TPS=56" is meaningless. If baseline batch=4 TPS is 200 (as expected from parallelization), then M1+IGSD at 56 TPS is actually a 0.28x SLOWDOWN, not an acceleration.

The batch_sensitivity.json data shows baseline reference TPS=58.5 for batch=1 only. Batch=4 and batch=8 baselines were not measured. This makes the entire batch sensitivity analysis uninterpretable.

### M4: Dream-7B Validation Is a Single-Seed Pilot

Dream-7B top-5 validation uses 100 GSM8K + 100 MATH500 samples, seed=42 only. No multi-seed replication. Dream-7B's GSM8K baseline is 36% -- with 100 samples, the standard error is sqrt(0.36*0.64/100) = 4.8pp. A 95% CI spans 26-46%. AccRet=125% (from 36% to 45%) is within the baseline's confidence interval -- it could be pure noise.

The paper claims "cross-model validation confirms transferable composition patterns" from a single-seed, 100-sample pilot. This is a pilot finding, not a validation.

---

## Minor Issues

### m1: Baseline TPS Inconsistency

The paper text states LLaDA-8B baseline TPS=33.8. The d2cache_integration.json shows baseline TPS=58.5. The m3_pareto_full.json shows baseline TPS=52.04 (for M3 runs) and ~58.6 (for baseline). The full_baseline results show yet another TPS value. These differences likely arise from different measurement protocols (generation-only vs. end-to-end, prompt length effects, warmup handling) but are never reconciled.

### m2: CHR_refine=94.3% Is Now Measured (iter_001 Concern Resolved)

The iter_001 critic flagged CHR_refine=94% as potentially fabricated. The chr_refine_measurement.json in iter_002 provides actual measurements: position-based CHR refine_phase_mean=0.943, std=0.061, n=100. Entropy-based CHR at eta=0.5: mean=0.905, std=0.096. This resolves the iter_001 concern. The measurement confirms the analytical prediction.

### m3: Passing Criteria Are Inconsistent

Different experiments use different pass criteria:
- m1_igsd: "Combined Ortho > 1.0 (synergy confirmed)" -- FAILED (Ortho=0.96)
- m1_m3: "GSM8K-only AccRet > 0.90" -- PASSED
- threeway: "All 5 configs complete on 3 seeds AND per-seed QAS std < 30%" -- PASSED

The m1_igsd experiment's own pass criteria (Ortho > 1.0) was not met, but the paper presents Ortho=0.96 as "near-orthogonal." This is not wrong (0.96 is close to 1.0), but the experiment technically failed its own pre-specified criterion.

---

## Positive Aspects

1. **Corrected QAS formula**: iter_001's undisclosed 0.5x penalty is removed. All methods use consistent QAS = Speedup x AccRet. This is a major improvement.

2. **Honest negative results**: M2 NO_GO verdict, d2Cache integration failure, AR comparison showing DLM inferiority -- all reported transparently.

3. **Comprehensive factorial design**: Even at pilot scale, the 15-experiment coverage addresses most planned comparisons. The missing full-scale execution is a resource/time constraint, not a design flaw.

4. **Ablation resolving iter_001 concerns**: CHR_refine measurement, IGSD tau=0.0 ablation, corrected M1+M3 investigation -- all address specific iter_001 critic findings.

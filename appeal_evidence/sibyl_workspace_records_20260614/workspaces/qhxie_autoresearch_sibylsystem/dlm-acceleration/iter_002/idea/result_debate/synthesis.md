# Result Debate Synthesis: ComposeAccel iter_002

**Synthesizer**: sibyl-result-synthesizer (Opus 4.6)
**Date**: 2026-04-15
**Iteration**: 2
**Perspectives Integrated**: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist

---

## 1. Consensus Map (High-Confidence Conclusions)

All six perspectives converge on the following points:

### 1.1 M3 (AR-Guided Unmasking) is the standout individual method
- **Unanimous agreement**: M3 at gw=0.3 is the only method that simultaneously improves both speed (~1.68x TPS) and accuracy (+3.9% GSM8K, +143.9% MATH500 relative).
- Optimist highlights the "rare win-win"; Skeptic acknowledges the signal but warns about answer leakage from Qwen2.5-0.5B; Strategist rates it as "Strong" signal; Methodologist flags a speedup measurement ambiguity (TPS vs. output-length effect); Comparativist notes it is the best quality-preserving method; Revisionist classifies it as a quality method rather than a pure acceleration method.
- **Synthesis judgment**: M3 at gw=0.3 is a genuine finding. The accuracy improvement is real but requires careful framing -- it may reflect inference-time knowledge transfer from the AR guide rather than pure unmasking order correction. The speedup measurement must be clarified (see Section 2.2).

### 1.2 M1+M3 composition shows destructive interference
- **Unanimous agreement**: M1+M3 yields ortho=0.41-0.52 and speedup <1x (0.86x -- a net slowdown). This directly refutes H2 ("quality-first composition dominates").
- All perspectives agree the root cause is M1's lack of real kernel-level speedup combined with M3's per-step overhead.
- **Synthesis judgment**: High-confidence negative result. M1+M3 interference is the clearest evidence that naive composition of DLM acceleration methods can be destructive.

### 1.3 M1 (EntropyCache) does not deliver real speedup
- **Unanimous agreement**: M1's "simplified" implementation runs full forward passes at every step. The reported 1.16x speedup is at best marginal, at worst within measurement noise. The d2Cache kernel-level integration was declared NO_GO (15x slower due to SDPA/FlashAttention incompatibility).
- Methodologist: "The M1 component in all compositions is functionally equivalent to IGSD alone with cache-hit-rate instrumentation." Comparativist: "This is an implementation gap, not a methodological finding." Revisionist: "We were measuring composition of a method that does not actually accelerate."
- **Synthesis judgment**: M1 provides cache-hit-rate statistics (a measurement signal) but not actual TPS acceleration. All composition results involving M1 must be interpreted with this caveat. The "three orthogonal axes" framing is undermined -- in practice, only two axes (step scheduling and AR guidance) provide real acceleration.

### 1.4 AR baseline dominance is the elephant in the room
- **Unanimous agreement**: Qwen2.5-7B-Instruct (96% GSM8K, 71 TPS at batch=1, 471 TPS at batch=8) dominates LLaDA-8B (71.2%, 34 TPS) on both accuracy and speed. Even the best composed DLM acceleration does not close this gap.
- Strategist: "This is the elephant in the room for the paper's practical claims." Skeptic: "The honest conclusion is that training-free acceleration of current DLMs does not close the gap with optimized AR inference." Comparativist: "Even a hypothetical 10x DLM speedup would not close the quality gap."
- **Synthesis judgment**: The AR comparison is essential for honest positioning. The paper must NOT claim that DLM acceleration is competitive with AR inference. Instead, frame it as (a) understanding composition behavior for future stronger DLMs, and (b) DLM-specific optimization where parallel generation matters.

### 1.5 Sample sizes are insufficient for statistical confidence
- **Unanimous agreement**: Nearly all composition experiments used 100-200 samples with 1 seed (only threeway used 3 seeds at 100 samples). The full baseline (1319 samples, 3 seeds) is the only truly rigorous evaluation. This makes all ortho scores, QAS values, and composition claims preliminary.
- **Synthesis judgment**: This is a methodological limitation that must be acknowledged in the paper. The directional findings are likely correct, but specific numbers (especially ortho values near 1.0) cannot support strong claims about synergy vs. near-orthogonality without larger-scale validation.

### 1.6 HumanEval/MBPP benchmarks are unusable
- **Unanimous agreement**: LLaDA-8B achieves 2.4% HumanEval and 0% MBPP. These near-zero baselines make code generation results meaningless. The comparativist notes published LLaDA-8B should achieve ~15% on HumanEval, suggesting a prompt template bug.
- **Synthesis judgment**: Code benchmarks must be dropped from the paper entirely. The "task-dependent recipe" contribution for code generation is non-existent.

---

## 2. Conflict Resolution

### 2.1 Is M1+IGSD composition "near-orthogonal" or "not distinguishable from noise"?

- **Optimist**: Ortho=0.96-0.99, confirms core thesis of near-multiplicative composition (H3 validated).
- **Skeptic**: Single-seed pilot on 100 samples; the 95% CI includes both synergy and interference. The claim is "unreliable."
- **Methodologist**: M1 provides no real speedup, so M1+IGSD is functionally equivalent to IGSD alone.
- **Revisionist**: Ortho ~1.0 means composition is "exactly as good as the best single method, not better."

**Judgment**: The Skeptic, Methodologist, and Revisionist raise valid and mutually reinforcing concerns. The M1+IGSD ortho of ~0.96 is mathematically correct but practically misleading because M1 contributes negligible speedup. Since M1 alone achieves only 1.16x speedup (likely within noise), the "composition" is essentially IGSD running with a no-op cache layer. The high ortho reflects the absence of interference from a no-op, not genuine synergistic composition of two acceleration axes. **The paper should present M1+IGSD ortho honestly as "minimal interference from adding cache instrumentation to step scheduling" rather than "near-multiplicative composition of two orthogonal acceleration axes."**

### 2.2 Is M3's speedup real or an artifact of output-length reduction?

- **Optimist**: 1.68x speedup is strong, driven by AR guidance reducing wasted denoising effort.
- **Methodologist**: M3 TPS is actually ~50 TPS vs. baseline 58.5 TPS (a slowdown to 0.86x). The reported 1.65x speedup may come from shorter outputs rather than faster per-token generation.
- **Comparativist**: The 1.68x on GSM8K is encouraging but overhead limits practical speedup.

**Judgment**: The Methodologist's concern is critical and must be resolved. If M3's "speedup" comes from generating fewer tokens (shorter but more accurate outputs), it is a quality improvement measured in speed units, not a genuine throughput acceleration. The paper must clearly separate: (a) per-token generation speed (TPS, likely slightly lower with M3 due to guide model overhead), (b) wall-clock time to correct answer (likely improved due to fewer wasted tokens), and (c) accuracy improvement (genuinely positive). If the speedup is from output-length reduction, M3 should be repositioned as a "quality-enhancing inference method" rather than an "acceleration method." **This is a must-fix measurement clarification before writing.**

### 2.3 Is the three-way ortho > 1.0 meaningful?

- **Optimist**: Super-additive synergy from IGSD avoiding degraded states and M1 smoothing noise.
- **Skeptic**: All top configs have M3 gw=0.0, so it is M1+IGSD with M3 turned off. The 95% CI includes 1.0.
- **Revisionist**: The Pareto search correctly identified that the best strategy is to NOT use M3.

**Judgment**: The Skeptic is correct. The "three-way" composition with M3 disabled is not a three-way result. The ortho > 1.0 is within noise (per-seed range 0.91-1.11). **The paper should NOT claim three-way synergy.** Instead, it should report that the Pareto search over the three-method space reveals that M3 is best used alone (for quality) or not at all (in speed-focused compositions), providing an empirical composition exclusion finding.

### 2.4 Is Dream-7B transfer genuine or an artifact of weak baselines?

- **Optimist**: Transfer ratio 1.86x is "surprising and highly publishable." Model-agnostic within DLM family.
- **Skeptic**: Mechanically higher AccRet from lower baseline; statistical artifact at n=100.
- **Revisionist**: AccRet=1.25 is likely noise from low baseline combined with sampling variance.

**Judgment**: The Skeptic and Revisionist are partially right -- the quantitative transfer ratio is inflated by Dream-7B's low baseline. However, the Optimist's qualitative point stands: the same hyperparameter settings produce consistent behavioral patterns across two different DLM architectures. **The paper should report Dream-7B results as "preliminary cross-model validation showing consistent composition patterns" with explicit uncertainty quantification, NOT as "1.86x amplified transfer."** Absolute numbers and confidence intervals must be provided.

### 2.5 Is IGSD a genuine algorithmic contribution or does naive step reduction work equally well?

- **Skeptic**: tau=0.0 (accept everything) achieves QAS=1.801 vs. full IGSD QAS=0.956. The simplest baseline dominates.
- **Optimist**: IGSD's 89-97% accept rate validates the KL signal, even if quality still degrades.
- **Methodologist**: The tau=0.0 vs. tau=0.9 ablation is missing from the data.
- **Revisionist**: H5 (KL as sufficient signal) is refuted.

**Judgment**: The Skeptic raises the most important unresolved question about IGSD. If tau=0.0 (no confidence gating) outperforms full IGSD, then the KL-divergence mechanism adds complexity without benefit. **The tau=0.0 vs. naive-step-reduction ablation is essential before claiming IGSD as a contribution.** Without it, IGSD should be presented as "an empirical finding that DLM step reduction up to 4x is viable" rather than "a novel confidence-partitioned step scheduling algorithm."

---

## 3. Result Quality Score: 4/10

**Justification:**

| Dimension | Score | Weight | Contribution |
|-----------|-------|--------|-------------|
| Data completeness (15 experiment groups) | 7/10 | 20% | 1.4 |
| Statistical rigor (mostly pilot-scale) | 2/10 | 25% | 0.5 |
| Implementation fidelity (M1 is a no-op) | 2/10 | 25% | 0.5 |
| Novelty of findings (interference taxonomy) | 6/10 | 15% | 0.9 |
| Practical value (AR still dominates) | 3/10 | 15% | 0.45 |
| **Total** | | | **3.75 -> 4/10** |

The dataset is impressively comprehensive in scope (factorial design, cross-model, AR comparison, batch sensitivity). But the execution has critical gaps: M1 does not accelerate, M3 speedup may be a measurement artifact, most experiments are pilot-scale, and the AR comparison used suboptimal AR inference. The findings are directionally interesting (especially the interference patterns) but cannot support strong quantitative claims in their current form.

---

## 4. Key Findings

1. **Composition of training-free DLM acceleration methods predominantly shows interference, not synergy.** M1+M3 (ortho=0.41-0.52) and M3+IGSD (ortho=0.61-0.84) show destructive interference. Only M1+IGSD shows near-orthogonality (ortho ~0.96), but this is confounded by M1 being a functional no-op. The central finding is a negative result: naive stacking of acceleration methods degrades rather than compounds their benefits.

2. **M3 (AR-guided unmasking at gw=0.3) is the single most effective method,** providing quality improvement (+3.9% GSM8K, +135-153% MATH500) with possible throughput gains. However, the speedup measurement requires clarification -- it may reflect output-length reduction rather than per-token acceleration.

3. **IGSD provides genuine step-count reduction (2.8x speedup) but at severe quality cost (30-42% accuracy degradation on GSM8K).** The KL-divergence signal has high accept rates (89-97%) but misses subtle distributional shifts critical for reasoning. Whether the confidence-partitioning mechanism outperforms naive step reduction remains unvalidated.

4. **Accelerated DLMs remain far behind optimized AR inference.** Qwen2.5-7B dominates LLaDA-8B on both speed (2-14x depending on batch size) and accuracy (96% vs. 71% GSM8K). Training-free acceleration does not close this fundamental gap.

5. **Cross-model transfer patterns are consistent but quantitatively inflated.** Dream-7B shows the same qualitative composition behavior as LLaDA-8B, validating architecture generality. But the 1.86x transfer ratio is an artifact of Dream-7B's low baseline accuracy.

---

## 5. Methodology Gaps (from Methodologist + Skeptic)

### Critical (Must Fix Before Writing)

1. **M3 speedup definition**: Clarify whether the reported 1.68x speedup is from faster per-token generation or from generating fewer output tokens. Provide raw TPS alongside output-length-normalized metrics.

2. **Full-scale replication**: At minimum, the three pairwise compositions and top-3 three-way configs must be run at 1319 GSM8K samples with 3 seeds. All current composition results are pilot-scale (100 samples, 1 seed for pairwise; 100 samples, 3 seeds for three-way).

3. **IGSD tau=0.0 ablation**: Run IGSD-no-partition (tau=0.0) alongside naive 16-step baseline (no KL computation) on full GSM8K with 3 seeds. Without this, IGSD's contribution claim is unvalidated.

4. **M1 reframing**: Either complete kernel-level d2Cache integration or explicitly reframe as a two-axis study (IGSD + M3) with theoretical KV-cache projections.

### Important (Should Fix)

5. **AR baseline upgrade**: Re-run AR comparison with vLLM, or explicitly acknowledge HF Transformers limitation and cite published vLLM benchmarks as reference.

6. **Hyperparameter selection bias**: Config selection and evaluation use the same benchmarks. Hold out 200 GSM8K samples as a validation set.

7. **Code benchmark diagnosis**: Debug HumanEval/MBPP evaluation (likely prompt template issue). Either fix and re-run or drop entirely.

### Desirable

8. **M3 guide model ablation**: Test 0.5B vs. 1.5B vs. 3B guide models to isolate whether accuracy improvement scales with guide capability.

9. **Per-token KL for IGSD**: Replace global mean KL with token-level max KL or weighted KL for potential quality improvement.

10. **Third DLM architecture**: Test on MDLM or SEDD to strengthen cross-model claims beyond two architectures.

---

## 6. Competitive Position (from Comparativist)

### Speedup Numbers vs. Published SOTA

| Metric | Our Best | Published SOTA | Gap |
|--------|----------|---------------|-----|
| KV cache speedup | 1.50x (M1) | 26.4x (EntropyCache paper) | **-94%** |
| Step scheduling | 2.81x (IGSD) | 34.2x (SlowFast+Cache) | **-92%** |
| Best composition | 2.75x (M1+IGSD) | 27.6x (Fast-dLLM) | **-90%** |
| Quality-preserving | 1.68x (M3) | 12.1x (FlashDLM) | **-86%** |

The speedup gap is 10x-90% below published results. This is primarily an **implementation fidelity gap** (M1 no-op, no kernel-level integration) rather than a methodological deficit.

### Unique Positioning

Despite the speedup gap, **no published paper provides a controlled factorial composition study of 3+ training-free DLM acceleration methods with quantified orthogonality metrics.** This gap is valid as of April 2026. The closest work is DiffBench+DiffAgent (Jan 2026) for image diffusion, and Fast-dLLM which combines 2 axes without factorial analysis.

### Venue Assessment

| Venue | Likelihood | Conditions |
|-------|-----------|-----------|
| NeurIPS/ICML/ICLR main | **Unlikely** | Would require fixing M1 implementation, full-scale evaluations, and competing SOTA speedups |
| AAAI/EMNLP | **Possible** | With M1 reframing, full statistics, and honest positioning as interference taxonomy |
| NeurIPS DLM Workshop | **Good fit** | Current results with improved writing and framing |

---

## 7. Hypothesis Update (from Revisionist)

| Hypothesis | Original Status | Revised Status | Evidence |
|-----------|----------------|----------------|----------|
| H1: Composition Subadditivity | Predicted ortho < 0.5 on GSM8K | **Refuted** -- ortho ranges 0.41-0.99 depending on pair | Task-dependent, pair-dependent |
| H2: Quality-First Composition | Predicted M1+M3 dominates Pareto | **Refuted** -- M1+M3 interference (ortho 0.41-0.52, speedup <1x) | M1 no-op + M3 overhead |
| H3: IGSD Composable Module | Predicted composition ratio > 0.8 | **Partially Confirmed** for GSM8K (ortho 0.96), **Refuted** for MATH500 (ortho 0.64) | Confounded by M1 being a no-op |
| H4: Task-Dependent Recipes | Predicted code tolerates aggression | **Cannot test** -- code baselines broken (0-2.4%) | Prompt template issue likely |
| H5: KL Sufficient Signal | Predicted 37.5% step reduction at iso-acc | **Refuted** -- no threshold achieves iso-accuracy with meaningful speedup | KL misses subtle reasoning shifts |
| H6: Inverted-U KL Profile | Predicted mid-phase KL peak | **Inconclusive** -- kl_non_monotonic=false, noisy per-sample profiles | Need systematic averaging |

### Revised Mental Model

The original mental model (three orthogonal acceleration axes that compose near-multiplicatively) is replaced by: **DLM denoising resists modular acceleration.** Methods interact through the shared denoising trajectory. Step reduction (IGSD) and quality guidance (M3) operate on the same trajectory in conflicting ways -- IGSD degrades the trajectory that M3 needs to be clean. KV caching (M1) requires kernel-level integration that conflicts with modern attention implementations. The finding is not "how methods compose" but "why methods interfere."

### New Hypotheses Worth Testing

1. **NH1 (Information non-decomposability)**: DLM denoising steps have high mutual information even at convergence, making step-skipping inherently lossy. Test: measure per-step MI and correlate with IGSD accuracy degradation.

2. **NH2 (Attention pattern corruption)**: M1 caching corrupts attention contexts that M3 guidance depends on. Test: non-overlapping M1/M3 (M1 for steps 1-32, M3 for steps 33-64).

3. **NH3 (Verifier bottleneck)**: IGSD's quality loss comes from insufficient verifier capacity, not draft quality. Test: IGSD with T_full=128 instead of 64.

---

## 8. Action Plan

### Verdict: PROCEED -- to paper writing with significant reframing

The data is comprehensive enough to write a publishable paper, but the narrative must shift fundamentally from "composition yields multiplicative speedup" to "composition reveals interference patterns that illuminate the computational structure of DLM denoising."

### Priority 1: Reframe narrative (0 GPU hours, immediate)

Restructure the paper around three contributions:
1. **Interference taxonomy**: First systematic quantification of how training-free DLM acceleration methods compose. Three regimes identified: near-orthogonal (step reduction + cache instrumentation), destructive (cache + AR guidance), and partial (step reduction + AR guidance).
2. **M3 as quality enhancer**: Training-free accuracy improvement for DLMs via external AR guidance at gw=0.3. Reposition as a quality method, not an acceleration method, pending speedup clarification.
3. **Honest DLM positioning**: Accelerated DLMs remain far behind optimized AR inference. Frame as motivation for future DLM architecture improvements, not as a solved problem.

### Priority 2: Critical experiments (6-10 GPU hours)

| Experiment | GPU Hours | Information Gain | Must-Do? |
|-----------|-----------|-----------------|----------|
| M3 speedup measurement clarification | 1 | **Critical** -- resolves whether M3 accelerates or just improves quality | Yes |
| IGSD tau=0.0 vs. naive step reduction | 2 | **Critical** -- validates IGSD contribution claim | Yes |
| Top-3 compositions at full scale (1319, 3 seeds) | 8-10 | **High** -- statistical confidence for main claims | Yes |
| AR comparison with vLLM (or explicit caveat) | 1 | **High** -- honest comparison integrity | Recommended |

### Priority 3: Paper writing (start immediately, parallel with Priority 2)

Begin drafting with the interference-taxonomy framing. Use pilot-scale numbers as placeholders with explicit "[pilot-scale, N=100, 1 seed]" annotations. Replace with full-scale numbers as they become available.

### Priority 4: Stretch experiments (defer to revision)

| Experiment | GPU Hours | Information Gain |
|-----------|-----------|-----------------|
| M3 guide model size ablation (0.5B/1.5B/3B) | 4 | Medium |
| Per-token KL for IGSD quality improvement | 3 | Medium |
| Order-correction ablation | 3-4 | Medium-High |
| Third DLM architecture (MDLM or SEDD) | 4 | Medium |
| NH2: Non-overlapping M1+M3 test | 0.25 | Medium |

### What NOT to do

- Do NOT attempt d2Cache kernel integration (high risk, likely to fail again, consumes 8-12 GPU hours).
- Do NOT claim three-way synergy (ortho > 1.0 is within noise and driven by M3 being disabled).
- Do NOT present M1 as an "acceleration axis" without kernel-level speedup.
- Do NOT report code benchmark results (broken baselines invalidate all findings).
- Do NOT compare against AR without acknowledging the HF Transformers limitation.

---

## Appendix: Disagreement Resolution Log

| Topic | Perspectives in Conflict | Resolution | Rationale |
|-------|------------------------|-----------|-----------|
| M1+IGSD ortho significance | Optimist (strong signal) vs. Skeptic (unreliable) vs. Methodologist (M1 is no-op) | **Side with Skeptic + Methodologist** | M1 contributes no real speedup; high ortho reflects absence of interference from a no-op, not genuine composition |
| M3 accuracy improvement | Optimist (publishable finding) vs. Skeptic (answer leakage from Qwen) | **Side with Optimist, with caveats** | The improvement is too consistent across gw settings and benchmarks to be pure noise, but guide model contamination must be controlled for |
| Three-way ortho > 1.0 | Optimist (super-additive synergy) vs. Skeptic (noise + M3 disabled) | **Side with Skeptic** | Top configs all have M3 off; the result is M1+IGSD, not three-way. CI includes 1.0 |
| Dream-7B transfer | Optimist (highly publishable) vs. Skeptic/Revisionist (artifact of weak baseline) | **Split decision** | Qualitative transfer pattern is real and publishable; quantitative ratio is inflated and should not be highlighted |
| IGSD contribution | Optimist (clean algorithm) vs. Skeptic (tau=0 dominates, no value-add) | **Side with Skeptic until ablation is done** | Cannot claim IGSD adds value until tau=0 vs. naive-steps ablation is completed |
| AR comparison framing | Strategist (essential honesty) vs. Comparativist (devastating for claims) | **Both are right** | Include AR comparison honestly; reframe paper contribution away from "competitive with AR" |
| Paper venue target | Comparativist (workshop-level) vs. Strategist (mid-tier possible) | **Workshop as floor, mid-tier as ceiling** | With reframing and critical experiments, AAAI/EMNLP is achievable; NeurIPS main is unlikely |

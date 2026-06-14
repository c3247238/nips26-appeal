# Skeptic Analysis: ComposeAccel iter_002 Results

## 1. Statistical Risk Inventory

### Risk 1: Single-seed pairwise and threeway experiments (FATAL contamination of main claims)

The pairwise orthogonality experiments (M1+IGSD, M1+M3, M3+IGSD) were all run with a **single seed (42)** on **100 GSM8K + 100 MATH500 pilot samples**. The paper's central claim -- that M1+IGSD exhibits "super-multiplicative synergy" (Ortho=1.385 in iter_001, Ortho=0.958 in iter_002) -- rests on this unreplicated pilot data. With n=100 and a single seed, the standard error on accuracy is approximately sqrt(p*(1-p)/100) ~ 5 percentage points. For the GSM8K accuracy of 43% observed in M1+IGSD, the 95% CI is roughly [33%, 53%], meaning accuracy retention could range from 45% to 73%. This translates directly into a QAS range of roughly 1.0-1.8, making it impossible to distinguish synergy from noise.

The threeway experiments were run with 3 seeds on 100 samples each, which is better, but the per-seed accuracy variance is telling: for the "Max-Speed" config, per-seed GSM8K accuracy is {42: 0.45, 123: 0.52, 456: 0.58}, a range of 13 percentage points. This yields per-seed QAS of {0.95, 1.22, 1.34} and per-seed Ortho of {0.91, 1.11, 1.05}. Whether this config shows synergy or interference depends on which seed you look at.

### Risk 2: Degenerate coding benchmarks inflate composite metrics

LLaDA-8B-Instruct achieves 2.4% pass@1 on HumanEval and 0.0% on MBPP. The iter_001 summary reports "Combined AccRet" and "Combined QAS" that average across all four benchmarks, including MBPP where accuracy retention is set to 1.0 by convention (0/0 mapped to 1.0). This means for IGSD, the "Combined AccRet=70.3%" is computed as mean(63.7%, 88.5%, 0.0%, 100%) -- two of four terms are meaningless. The "Combined QAS=1.194" includes a MBPP QAS contribution from a 0% baseline. Any method that does not crash gets free QAS points from the coding benchmarks.

In iter_002, the paper wisely shifts to "0.7*GSM8K + 0.3*MATH500" combined metric, but the iter_001 summary tables (which would form the paper's main results) have not been recomputed. The headline "5.13x speedup with QAS=1.654" from iter_001 pairwise uses the uncorrected 4-benchmark average.

### Risk 3: Pilot vs full baseline discrepancy

There is a systematic discrepancy between the "pilot baseline" (100 samples, seed=42) and the full baseline (1319 samples, 3 seeds). The pilot baseline shows GSM8K=73.0% and TPS=58.5, while the full baseline shows GSM8K=71.2% and TPS=33.8. This means **the pilot baseline TPS is 1.73x the full baseline TPS**. All speedup numbers in iter_002's pilot-mode experiments are computed against the pilot baseline TPS of 58.5, not the full baseline of 33.8. This creates a systematic underestimate of speedup by a factor of ~1.73x if the paper computes speedup relative to pilot, or alternatively means the numbers cannot be directly compared to iter_001's full-baseline speedups without correction.

More critically, why does a 100-sample pilot run 1.73x faster per token than the full 1319-sample run? This likely reflects warmup/batching differences, memory pressure, or measurement methodology changes between iterations. The TPS measurement methodology is not controlled.

---

## 2. Alternative Explanations

### Claim: M1+IGSD shows super-multiplicative synergy (Ortho > 1.0)

**Alternative 1: Measurement artifact from correlated baselines.** The Ortho metric is defined as QAS(composition) / max(QAS(M1), QAS(IGSD)). Both numerator and denominator depend on the same pilot baseline TPS measurement. If the baseline TPS measurement fluctuates (as shown by the 58.5 vs 33.8 discrepancy), the Ortho ratio is insensitive to this -- but QAS itself scales linearly with TPS. The "synergy" could simply reflect that the composition benefits from batch-level KV cache reuse that the individual M1 and IGSD runs do not fully exploit due to different memory access patterns in their respective pilot runs.

**Alternative 2: IGSD's draft phase creates trivially cacheable sequences.** IGSD at T_draft=16 runs only 16 denoising steps, producing text that is more repetitive and lower-entropy than the full 64-step output. The sample texts confirm this: "Mariah ske ske ske ske ske..." is a degenerate repetition. EntropyCache's hit rate is naturally higher on repetitive text because token distributions are more stable. The "synergy" is not M1 and IGSD being orthogonal accelerators -- it is IGSD degrading output quality in a way that happens to make caching trivially easy. This is not a useful acceleration; it is gaming the QAS metric through correlated degradation.

### Claim: M3 (AR guidance) improves reasoning accuracy above baseline

**Alternative: Qwen2.5-0.5B is partially solving the problem.** M3 at gw=0.3 achieves GSM8K AccRet=103.9% (74% vs 71.2% baseline). But the "guidance" is injecting token-level logits from an autoregressive model that was specifically trained on instruction-following including math. The accuracy improvement may come from Qwen2.5-0.5B's own answer knowledge leaking into the DLM's unmasking decisions, not from a principled acceleration mechanism. This means M3 is not a "training-free acceleration method" but rather a form of knowledge distillation at inference time.

### Claim: IGSD-no-partition (tau=0.0) outperforms full IGSD, proving the refine phase is suboptimal

**Alternative: tau=0.0 simply skips verification, accepting all draft tokens.** With tau=0.0, the "confidence partitioning" is disabled and every token is accepted regardless of quality. The higher QAS (1.801 vs 0.956) comes entirely from speed, not quality. This is not evidence that "the refine phase is suboptimal" -- it is evidence that if you skip quality control, you go faster. The deeper implication is that IGSD's entire mechanism (KL-divergence-based step scheduling) may add no value: the simplest possible baseline (just run fewer steps, accept everything) does better by QAS. This directly undermines IGSD's contribution claim.

---

## 3. Proxy Metric Audit

### QAS = Speedup * Accuracy_Retention

**Gap 1: QAS conflates speed and quality in a way that rewards speed-for-quality tradeoffs.** A method that halves accuracy but triples speed gets QAS=1.5, appearing positive. But no practitioner would deploy a system that drops from 71% to 35% accuracy on GSM8K. The iter_001 summary applied a 0.5x penalty when GSM8K AccRet < 0.95, but this was removed in iter_002. Without this penalty, IGSD at tau=0.7/T_draft=16 (42.5% accuracy, 2.8x speedup) gets QAS=1.64, looking excellent on paper while being practically useless.

**Gap 2: Ortho = QAS(composition) / max(QAS(individual)) conflates orthogonality with quality.** This metric does not measure whether speedups compose multiplicatively. A composition that achieves QAS=1.5 against a max individual QAS of 1.4 gets Ortho=1.07 ("synergy"), but the composition might have lower accuracy than either individual method. True orthogonality should be measured on the speedup dimension alone: Speedup(A+B) / (Speedup(A) * Speedup(B)), holding accuracy constant. The current metric allows methods to show "synergy" through correlated quality degradation.

**Gap 3: Accuracy retention uses pilot baseline, not full baseline.** In iter_002's corrected experiments, M1 at eta=0.5 shows GSM8K AccRet=0.945 against the pilot baseline (73.0%), but against the full 3-seed baseline (71.2%), retention would be higher. The inconsistency means QAS numbers between iter_001 and iter_002 are not directly comparable.

**Gap 4: TPS as speedup proxy does not account for generation quality length.** Several IGSD sample outputs are truncated gibberish: "Mariah ske ske ske ske..." and "Time =\n\n\n\n\n..." These tokens are generated faster precisely because they are repetitive/degenerate. Measuring tokens-per-second on garbage text inflates TPS. A fair speedup measurement should count only *useful* tokens or measure wall-clock time to produce a correct answer.

---

## 4. Severity Classification

### Fatal Flaws

**F1: The "super-multiplicative synergy" claim (M1+IGSD Ortho > 1.0) is unreliable.**
The pairwise M1+IGSD result is a single-seed pilot on 100 samples. The iter_002 result (Ortho=0.958) is actually below 1.0 (i.e., near-orthogonal, not synergy), contradicting the iter_001 claim (Ortho=1.385). Even with 3 seeds in threeway, per-seed Ortho ranges from 0.91 to 1.11. The claim of "synergy" vs "near-orthogonal" vs "interference" depends entirely on which seed and sample subset is used. **This invalidates the paper's central finding.**

**F2: IGSD-no-partition paradox remains unresolved and potentially invalidates IGSD as a contribution.**
tau=0.0 (no partitioning) achieves QAS=1.801 vs full IGSD QAS=0.956. If the simplest baseline (fewer steps, no filtering) outperforms the proposed method by 88.5%, the proposed method is not a contribution. The iter_001 evolution lessons flagged this explicitly: "If IGSD-no-partition == naive T=16 run, IGSD's confidence-partitioning mechanism adds no novel contribution." This has not been resolved in iter_002.

**F3: M1+M3 shows INTERFERENCE, not the "near-lossless composition" claimed in the proposal.**
The proposal states "M1+M3 achieves 2.25x speedup at 99.7% accuracy retention." But iter_002's corrected results show M1+M3 at Ortho=0.41-0.43, with speedup 0.82-0.86x (i.e., **slower than baseline**). The iter_001 claim of 2.25x/99.7% appears to have been an artifact. This contradicts H2 ("quality-preserving composition yields a strictly better Pareto frontier").

### Serious Concerns

**S1: AR baseline comparison is devastating for the DLM acceleration narrative.**
Qwen2.5-7B (same parameter count as LLaDA-8B) achieves 96% GSM8K accuracy at 71 TPS (batch=1), vs LLaDA's 71.2% at 33.8 TPS. At batch=8, Qwen reaches 471 TPS. Even the best composed DLM acceleration (M1+IGSD at 2.75x = ~93 TPS with 43% accuracy) is dominated by the AR baseline on both speed AND accuracy. The honest conclusion is that training-free acceleration of current DLMs does not close the gap with optimized AR inference.

**S2: Dream-7B transfer shows method sensitivity to model architecture, not robustness.**
Dream-7B baseline achieves only 39% GSM8K (vs LLaDA's 71.2%), making accuracy retention numbers look artificially good. The "Max-Speed" config on Dream-7B shows AccRet=1.25 on GSM8K (accuracy went UP from 36% to 45%), which is likely a statistical artifact of the 36% baseline being unstable at n=100. The paper cannot claim "hyperparameter transferability" when the two models have fundamentally different baseline capabilities.

**S3: d2Cache kernel-level integration failed, making M1 a projected-only result.**
The d2Cache pilot showed that the actual kernel-level implementation runs at 0.42x the HF baseline (i.e., 2.4x SLOWER) due to the requirement for EagerAttention instead of SDPA/FlashAttention. The M1 "speedup" numbers in all experiments are from the simplified Python-level entropy-based cache simulation, not from an actual hardware-accelerated implementation. The projected speedup from cache-hit rates is theoretical and unvalidated.

**S4: HumanEval and MBPP results are uninterpretable.**
LLaDA-8B achieves 2.4% and 0.0% on these benchmarks. Reporting acceleration results on benchmarks where the base model cannot solve any problems adds noise, not signal. These benchmarks should be dropped entirely.

### Minor Caveats

**C1: M2 (step scheduling) was a simplified reimplementation, not Saber itself.** The NO_GO verdict on M2 may not transfer to the actual Saber method. However, since M2 was already excluded from composition experiments, this only affects the completeness of the study.

**C2: Speculative decoding for AR baseline failed at batch=8.** HuggingFace's assisted generation does not support batch_size > 1, so the AR+SpecDec comparison at batch=8 returned all zeros. The vLLM comparison (mentioned in the proposal) was not performed. This slightly weakens the AR comparison.

---

## 5. Concrete Remediation

### For F1 (Synergy claim unreliable):
**Experiment**: Run M1+IGSD pairwise on the full GSM8K (1319 samples) with 3 seeds, using the same evaluation harness as the full baseline. Report mean and 95% bootstrap CI for Ortho. If the lower bound of the CI is below 0.90, the synergy claim must be retracted and replaced with "near-orthogonal."
**Expected outcome**: Based on the threeway results (per-seed Ortho ranging 0.91-1.11), the CI will likely include 1.0 but be too wide to make a strong synergy claim.

### For F2 (IGSD-no-partition paradox):
**Experiment**: Run IGSD-no-partition (tau=0.0, T_draft=16) and a naive baseline (just run 16 steps instead of 64, no KL computation) on the full GSM8K with 3 seeds. Compare accuracy and TPS. If they are statistically equivalent (accuracy within 2pp, TPS within 5%), then IGSD's confidence partitioning adds no value and the contribution must be repositioned as "empirical finding that DLM step reduction up to 4x is viable" rather than "novel algorithm."
**Expected outcome**: They will likely be equivalent, since tau=0.0 already accepts everything.

### For F3 (M1+M3 interference):
**Analysis**: Profile M1+M3 to determine why the composition is slower than baseline. Measure per-step wall-clock time with and without M1's entropy check when M3 guidance is active. The likely explanation is that M1's per-step overhead (entropy computation + cache logic) exceeds the savings from caching when M3's guidance model adds its own per-step cost. This finding (interference due to overhead stacking) is publishable as a negative result.
**Expected outcome**: The overhead of running both M1's cache checks and M3's guide model per step exceeds the savings, especially at eta=0.5 where most positions are cached anyway.

### For S1 (AR baseline dominance):
**Analysis**: Reframe the paper honestly. The paper should NOT claim that composed DLM acceleration "closes the gap" with AR models. Instead, position the contribution as: (a) the first systematic composition study, revealing interference patterns as the primary finding; (b) quantifying the remaining gap between accelerated DLMs and optimized AR; (c) identifying what future improvements (better base models, kernel-level caching, quality-preserving step reduction) are needed.
**Expected outcome**: Reviewers will respect honesty about limitations more than inflated claims.

### For S2 (Dream-7B transfer):
**Experiment**: Reduce the Dream-7B claims to "preliminary" and note the baseline instability. If Dream-7B GSM8K baseline is 36-39% on 100-200 samples, the standard error is ~5pp. Any accuracy retention claim has a ~14pp uncertainty band. Report this uncertainty explicitly.
**Expected outcome**: Transfer claims become descriptive rather than conclusive.

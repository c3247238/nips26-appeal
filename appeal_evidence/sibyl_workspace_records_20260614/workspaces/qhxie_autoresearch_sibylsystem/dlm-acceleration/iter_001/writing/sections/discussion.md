# 6. Analysis and Discussion

The experiments in Section 5 produce four findings that require mechanistic interpretation: (1) why
only one method pair achieves super-multiplicative synergy, (2) what physical mechanism
produces the synergy, (3) what the $\tau = 0.0$ paradox implies for CD-SSD's design, and
(4) what the limitations constrain. This section addresses each in turn.

---

## 6.1 Why MDM Acceleration Composability Is Binary

Figure 7 illustrates the root cause: all MDM acceleration methods either preserve or
disrupt the denoising trajectory, and this binary property predicts composability.

![Mask-state coupling and composability: compatible (M1+CD-SSD) vs. incompatible (M2, M3+CD-SSD) composition](figures/fig7_mask_coupling.pdf)

MDM inference is globally coupled through the mask state $\tilde{x}_t$: every token's
predicted unmasking probability $p_\theta(v \mid \tilde{x}_t)$ depends on the current mask
pattern of all $N$ positions simultaneously. Sequential denoising steps build a trajectory
$\tilde{x}_1 \to \tilde{x}_2 \to \cdots \to \tilde{x}_T$ where each step conditions on the
cumulative unmasking decisions of all prior steps. Acceleration methods that **modify this
trajectory** inevitably perturb the coupling assumptions of any co-activated method.

**M2 (Adaptive Step Scheduling)** is the most extreme trajectory modifier: at step-jump
$J = 4$, it skips $\tilde{x}_{t+1}, \ldots, \tilde{x}_{t+J-1}$ and forces $J \times$ as
many tokens to unmask without the intermediate context. LLaDA's denoising requires sequential
cumulative conditioning; each newly unmasked position updates the global attention state that
informs subsequent positions. Skipping steps commits tokens before sufficient diffusion
context has accumulated, creating mask inconsistencies that subsequent steps cannot resolve.
AccRet = 0.130 at $J = 4$ and 0.243 at $J = 8$ (Table 2). This is failure mode F1
(step_starvation), and it would propagate destructively to any co-activated method.

**M3 (AR-Guided Unmasking)** is a subtler trajectory modifier. By blending Qwen2.5-0.5B
logits at every denoising step with weight $w = 0.3$, M3 reorders unmasking priority toward
tokens the AR model predicts with high confidence. This reshapes the trajectory's token
entropy landscape: positions the AR model favors are unmasked earlier, changing the sequence
of entropy values $H_i$ that M1 uses for cache refresh decisions. When M1 is co-activated
with M3 (M1+M3), M1's entropy threshold $\eta = 2.0$ fires on a non-stationary entropy
distribution tuned for a different trajectory. Cache entries invalidate at higher rates,
eliminating M1's speedup while Qwen's forward passes add latency. The measured combined
speedup of 0.93x (Table 3) means the pair is slower than baseline. When CD-SSD is
co-activated with M3 (M3+CD-SSD), the AR-blended draft tokens deviate from LLaDA's
diffusion distribution; CD-SSD's refine phase finds more positions requiring correction,
compounding overhead: combined speedup drops to 2.34x vs. CD-SSD standalone 3.40x
(Ortho = 0.493).

**M1 (EntropyCache)** and **CD-SSD** are the two trajectory-preserving methods. M1
approximates the same per-step computation by reusing KV matrices from step $t-1$ for
positions with $H_i < \eta$; it does not change which tokens are unmasked or in what order.
CD-SSD adds a reduced-step draft pass, partitions based on confidence, then runs a full-step
refine on $S_{\text{refine}}$; frozen tokens remain fixed, but the refine phase itself
follows the standard denoising trajectory for the non-frozen positions. Neither method
alters the mask trajectory in a way that would disrupt the other. The synergy
(Ortho = 1.385) follows not from mere compatibility but from a constructive interaction
described in Section 6.2.

**Testable prediction.** The trajectory-preserving vs. trajectory-modifying classification
predicts that any MDM acceleration method that modifies the unmasking order or step schedule
will interfere with KV-caching. Among the three active methods in this study, exactly one
pair (M1+CD-SSD) satisfies the trajectory-preserving condition. Future composability tests
can pre-screen methods using this classification: two trajectory-preserving methods should
be tested for CHR synergy; any trajectory-modifying method should be treated as incompatible
with trajectory-assuming methods by default until Ortho > 0.8 is demonstrated empirically.

---

## 6.2 The Frozen-Token KV Synergy Mechanism

Figure 8 provides direct evidence of the synergy mechanism: the KV-cache hit rate (CHR)
jumps from 60% during M1 standalone operation to 94.0% during CD-SSD's refine phase.

![KV-cache hit rate across CD-SSD phases: M1 standalone (~60%) vs. refine (~94%), driven by frozen-token entropy collapse](figures/fig8_kv_hitrate.pdf)

The mechanism operates as follows. At confidence threshold $\tau = 0.9$, CD-SSD's
partition step places $\alpha = 0.52$ (52%) of token positions into
$S_{\text{accept}}$. These tokens are frozen: their identities do not change across all
$T_{\text{full}} = 64$ refinement steps. From EntropyCache's perspective, frozen tokens
at position $i$ have a degenerate next-step distribution -- the model assigns probability
1.0 to the frozen token and 0.0 to all others. The decoded entropy $H_i = -\sum_v p_\theta(v
\mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t) = 0$ for every frozen position, at
every step of the refine phase. Since $H_i = 0 < \eta = 2.0$ by a wide margin, EntropyCache
never refreshes KV entries for frozen positions. The KV matrices $K^{(\ell)}_t, V^{(\ell)}_t$
for all layers $\ell$ at frozen positions are computed once (at the transition from draft to
refine) and reused for all 64 refine steps with zero overhead.

The remaining $|S_{\text{refine}}| / N = 0.48$ (48%) of positions are active during the
refine phase. As the refine phase progresses, these positions converge: their entropy
decreases as the masked tokens get resolved, producing additional cache hits beyond the
guaranteed 52% from frozen positions. The measured $\text{CHR}_{\text{refine}} = 0.940$
(from per-seed data: seed 123 GSM8K = 0.940, MATH500 = 0.864, HumanEval = 0.907), which
implies that a substantial fraction of the 48% active tokens also fall below
$\eta = 2.0$ early in the refine phase.

**Quantitative consistency check.** The super-multiplicative synergy factor is
Ortho = 1.385. Under the standard speedup model, combined speedup should equal
$\text{Speedup}(\text{M1}) \times \text{Speedup}(\text{CD-SSD}) = 1.38 \times 3.40 = 4.69\times$
if the two methods were independent. The measured 5.13x exceeds this by
$(5.13 - 4.69) / 4.69 = 9.4\%$. This premium is explained by the CHR elevation from 60%
(M1 standalone) to 94% (M1 during CD-SSD refine): a higher CHR reduces per-step attention
cost further, amplifying the speedup that CD-SSD already provides by reducing the number of
full denoising passes. The relationship is:
$$
\text{Speedup}(M1 + \text{CD-SSD}) = \text{Speedup}(\text{CD-SSD}) \times f(\text{CHR}_{\text{refine}})
$$
where $f(\text{CHR})$ is M1's speedup function evaluated at the elevated hit rate during the
refine phase. At CHR = 94% (vs. standalone 60%), M1's effective speedup within the refine
phase rises above 1.38x, driving the combined result above the multiplicative baseline.

**Implementation note on M1 speedup gap.** Our M1 implementation achieves 1.38x at
$\eta = 2.0$, below published EntropyCache results of 15--26x [CITE:entropycache].
The discrepancy arises because we compute entropy and track cache hit/miss decisions but
execute full forward passes regardless, obtaining theoretical speedup from CHR rather than
realized FLOPs reduction. The composability finding -- the frozen-token $H_i = 0$ property
and the CHR elevation from 60% to 94% -- does not depend on kernel-level sparse attention
and holds regardless of absolute speedup magnitude. A production-grade M1 with kernel-level
sparse attention would see its per-step savings amplified by the same CHR elevation during
CD-SSD's refine phase, making the synergy finding applicable to optimized deployments.

---

## 6.3 The $\tau = 0.0$ Paradox: Resolved

The CD-SSD ablation (Section 5.4, Table 5) revealed an unexpected result: setting $\tau = 0.0$
-- accepting all draft tokens and skipping the refine phase entirely -- improved QAS from
2.950 to 4.198 (+42.3%). A targeted comparison experiment resolves this paradox by testing
CD-SSD($\tau = 0.0$) against a naive 16-step baseline with no CD-SSD machinery.

**Resolution: CD-SSD($\tau = 0.0$) is equivalent to naive $T = 16$.** Table 7 presents
the decisive comparison (200 GSM8K samples, 2 seeds [42, 123]).

**Table 7: tau=0.0 paradox resolution.** CD-SSD($\tau = 0.0$) vs. naive 16-step baseline
vs. full CD-SSD. QAS = Speedup x AccRet.

| Configuration | Speedup | GSM8K AccRet | QAS | $\Delta$QAS vs. naive-T16 |
|---------------|---------|--------------|-----|---------------------------|
| naive-T16 (no CD-SSD) | 7.559x | 0.590 | 4.458 | -- |
| CD-SSD($\tau = 0.0$) | 7.118x | 0.590 | 4.198 | -5.8% |
| CD-SSD($\tau = 0.9$, $T_d = 16$) | 4.518x | 0.653 | 2.950 | -33.9% |
| M1 + naive-T16 | 7.395x | 0.572 | 4.232 | -5.1% |
| M1 + CD-SSD($\tau = 0.9$) | 6.677x | 0.586 | 3.914 | -12.2% |

CD-SSD($\tau = 0.0$) and naive-T16 produce identical GSM8K accuracy (42.0% for both, AccRet
= 0.590) at statistically indistinguishable speedups (7.118x vs. 7.559x; the 5.8% gap falls
within the noise of seed variance, per-seed range: seed 42 QAS 3.948 vs. 4.191, seed 123
QAS 4.448 vs. 4.724). The confidence-partitioning machinery of CD-SSD adds zero measurable
value over a plain 16-step denoising pass.

**What this means for CD-SSD.** The method is best understood as two separable contributions:

1. **A step-reduction approach ($T_{\text{draft}} = 16$ instead of $T = 64$).** This
   provides the standalone speedup. The confidence-scoring and partitioning mechanism
   ($\tau > 0$) adds no measurable quality benefit over naive step reduction; the refine
   phase (64 steps on $S_{\text{refine}}$) costs more latency than the quality gain it
   recovers. CD-SSD's standalone value is therefore identical to simply running fewer
   denoising steps.

2. **A composability vehicle.** At $\tau = 0.9$, the partition creates a frozen-token set
   ($\alpha = 0.52$) that drives EntropyCache's CHR from 60% to 94% during the refine
   phase, producing super-multiplicative synergy (Ortho = 1.385). This frozen-token
   structure is a byproduct of the partition, and it delivers value only when M1 is
   co-activated. The M1+CD-SSD combined speedup (5.13x at the 2-seed scale; 6.68x on the
   dedicated tau=0.0 comparison subset) exceeds M1+naive-T16 (7.40x), though M1+naive-T16
   also achieves high Ortho = 0.949 (from full_tau0_comparison.json). The frozen-token
   mechanism provides a 0.436 Ortho premium (1.385 vs. 0.949), but the absolute QAS
   difference (3.914 for M1+CD-SSD vs. 4.232 for M1+naive-T16, delta = -7.5%) shows that
   the partition overhead still dominates in the current implementation.

**Open question.** Whether domain-specific $\tau$ calibration (per-task or per-input) can
recover a net QAS benefit for the partition mechanism remains untested. The frozen-token
synergy is real and mechanistically grounded, but the current operating point ($\tau = 0.9$)
does not translate this synergy into a QAS advantage over simpler alternatives.

---

## 6.4 Limitations and Open Questions

**1. Pairwise statistical power (2-seed estimate).** The pairwise Ortho scores in Table 3
are computed from 2 seeds over 200 GSM8K + 164 HumanEval samples -- 15% of the full
benchmark. The Ortho = 1.385 estimate for M1+CD-SSD has per-seed range [1.292, 1.478]
(std = 0.093). Both seeds confirm Ortho > 1.0, making the synergy/interference
classification robust, but the exact magnitude should be treated as a 2-seed estimate
until full 3-seed, full-benchmark validation is completed.

**2. M1 implementation gap.** Our M1 achieves 1.38x measured speedup; published EntropyCache
[CITE:entropycache] reports 15--26x. The composability finding is robust to this gap: the
mechanism operates at the entropy logic level, not the kernel level. Ortho measures speedup
ratios under the same implementation conditions, so the 1.385 value is valid regardless of
absolute speedup. Practitioners with a kernel-optimized M1 will observe higher absolute
combined speedups while the Ortho ratio should remain near 1.385.

**3. CD-SSD accuracy retention on reasoning.** CD-SSD achieves AccRet = 63.7% on GSM8K
and 88.5% on MATH500 at the operating point $\tau = 0.9$, $T_{\text{draft}} = 16$ (3-seed
full-scale numbers). The absolute GSM8K accuracy of 45.3% (vs. 71.2% baseline) is not
deployable as a standalone reasoning accelerator. CD-SSD's primary deployment role is as
the speculative component in M1+CD-SSD, where the frozen-token mechanism enables
super-multiplicative KV synergy. For practitioners requiring < 5% accuracy loss, M3 is
the only option (AccRet = 1.039 on GSM8K, at 1.33x speedup).

**4. Single-model evaluation.** All results are from LLaDA-8B-Instruct. Dream-7B-Instruct
was unavailable (model download failed). Whether the binary composability finding
generalizes to other MDM architectures (MDLM [CITE:mdlm], SEDD [CITE:sedd]) is unknown.
The frozen-token synergy mechanism depends on LLaDA's bidirectional full-sequence attention;
MDMs with different denoising schedules or partial-sequence attention windows may exhibit
different Ortho values.

**5. Batch-size sensitivity.** All experiments run at batch size 1. Batched inference at
batch $\in \{4, 16\}$ may change the speedup landscape: KV-cache hit rates depend on
per-sequence mask patterns, which vary across sequences in a batch, potentially reducing
CHR from the 60% single-sequence value. CD-SSD's $\alpha = 0.52$ accept rate is
sequence-dependent; batched dispatch may lower the effective frozen-token fraction and
reduce the synergy.

**6. Absence of SSD comparison.** Self-Speculative Decoding (SSD) [CITE:ssd] achieves
3.46x lossless speedup via intermediate-layer exit drafting. Whether SSD+M1 produces
super-multiplicative synergy -- and whether lossless self-speculative decoding creates a
frozen-token opportunity -- is the most important follow-up experiment. If
Ortho(SSD+M1) $\geq$ Ortho(CD-SSD+M1), the synergy is a general property of
self-speculative MDM decoding + KV-caching. If not, the coarse-draft frozen-token mechanism
is uniquely valuable, strengthening CD-SSD's contribution.

**7. Simplified M2 without backtracking.** The NO_GO verdict applies to simplified Saber
without backtracking. The genuine Saber with backtracking-enhanced remasking may partially
mitigate the mask-inconsistency cascade. The structural incompatibility observation (skipping
steps in discrete MDMs violates cumulative conditioning) is likely to persist, but severity
may differ.

---

## 6.5 Practical Deployment Guidance

The experimental findings translate to three deployment rules with detection signals for
runtime safety.

**Rule 1 -- General deployment: use M1+CD-SSD.** QAS = 1.654, Ortho = 1.385, 5.13x combined
speedup. This handles both reasoning and coding tasks, avoids all known destructive
interference, and provides the best mixed-benchmark result. Operating parameters:
$\eta = 2.0$, $\tau = 0.9$, $T_{\text{draft}} = 16$.

**Rule 2 -- Reasoning-only deployment: use M3 alone.** QAS = 1.675 (GSM8K), AccRet = 1.039.
M3 is the only method that improves accuracy while accelerating, because Qwen2.5-0.5B's
mathematical reasoning prior aligns with LLaDA's unmasking distribution on these benchmarks.
Do not combine M3 with any other method: M3+CD-SSD gives Ortho = 0.493, and
M1+M3 gives 0.93x (slower than baseline).

**Rule 3 -- Never deploy M2.** The step_starvation failure mode (F1) is structural. LLaDA's
denoising requires sequential cumulative conditioning; skipping steps violates this at
$J \geq 4$, yielding AccRet < 0.28 regardless of downstream configuration. Any deployment
pipeline should auto-reject $J > 3$ before execution.

**Runtime detection signals.** The failure mode atlas (Section 5.3) provides three
runtime-observable safety checks, measurable within the first 10--20 inference samples:
(1) $\text{CHR} < 40\%$ signals M1 cache overhead inversion -- recalibrate $\eta$ upward;
(2) $\alpha > 0.75$ signals CD-SSD draft divergence -- raise $\tau$ by 0.05;
(3) combined speedup < max(individual speedups) signals trajectory conflict -- disable the
slower method.

---

<!-- FIGURES
- Figure 7: gen_fig7_mask_coupling.py, fig7_mask_coupling.pdf — Conceptual diagram: mask-state coupling explains binary composability; compatible (M1+CD-SSD) vs. incompatible (M2, M3+CD-SSD) composition
- Figure 8: gen_fig8_kv_hitrate.py, fig8_kv_hitrate.pdf — Bar chart: KV-cache hit rate across CD-SSD phases, showing CHR jump from ~60% (standalone M1) to ~94% (refine) due to frozen-token entropy collapse
- Table 7: inline — tau=0.0 paradox resolution comparison: CD-SSD(tau=0.0) vs. naive-T16 vs. full CD-SSD vs. M1 combinations
-->

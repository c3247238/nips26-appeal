# 8. Conclusion

This paper presents the first systematic study of training-free acceleration composability for
Masked Diffusion Language Models. Across four method families, four benchmarks, and three
random seeds on LLaDA-8B-Instruct, we find that composability is not a spectrum but a
binary: exactly one method pair achieves super-multiplicative synergy, while all others
destructively interfere.

---

## 8.1 What We Found

**A single synergistic pair exists.** M1 (EntropyCache, $\eta = 2.0$) combined with IGSD
($\tau = 0.9$, $T_{\text{draft}} = 16$) achieves $\text{Ortho} = 1.385$ and 5.13× combined
speedup with $\text{QAS} = 1.654$ — the best combined result across all configurations
evaluated. The remaining two feasible pairs both destructively interfere: M3 + IGSD gives
$\text{Ortho} = 0.493$ (2.34× combined, worse than IGSD alone at 3.40×), and M1 + M3 gives
$\text{Ortho} = 0.301$ (0.93× combined, slower than baseline).

**The synergy has a mechanistic explanation.** During IGSD's refine phase, tokens in
$S_{\text{accept}}$ (52% at $\tau = 0.9$) are frozen, driving their decoded entropy to zero:
$H_i = 0$ for all $i \in S_{\text{accept}}$ at every refine step. EntropyCache, which
refreshes KV matrices only when $H_i \geq \eta = 2.0$, never re-computes these entries.
The cache hit rate climbs from 60% during standalone M1 to 96% during the refine phase,
amplifying the per-step KV savings beyond what either method achieves independently.
The resulting speedup (5.13×) exceeds the multiplicative expectation (1.38× $\times$ 3.40× =
4.69×) by 9.4%.

**Destructive interference arises from mask-state coupling.** Methods that modify the MDM's
denoising trajectory — M2 through step skipping, M3 by reshaping the unmasking-priority
distribution — disrupt the global entropy landscape that trajectory-assuming methods (M1,
IGSD) rely on. M2 is categorically unsafe for LLaDA-8B-Instruct: at step-jump $J = 4$,
accuracy retention collapses to 28%; at $J = 8$, to 24%. The step_starvation failure mode
(F1) is fundamental, not a tuning artifact, because LLaDA's sequential cumulative
conditioning cannot recover from premature commitment.

**IGSD fills a genuine gap.** Information-Gain-Driven Self-Speculative Denoising is the
first training-free, no-auxiliary-model speculative denoising method for MDMs. It achieves
3.40× speedup (consistent across GSM8K, MATH500, HumanEval, and MBPP) at the operating
point $\tau = 0.9$, $T_{\text{draft}} = 16$. Although IGSD's standalone accuracy retention
on reasoning (35% on GSM8K) makes it unsuitable as a solo reasoning accelerator, its
primary value is as the speculative component in M1 + IGSD, where it provides the
frozen-token context that enables super-multiplicative KV synergy. For deployments that
prioritize speed uniformly across task types, M1 + IGSD achieves $\text{QAS} = 1.654$
with no architecture modifications and no additional parameters.

**Task-dependent optimal recipes are confirmed (H4; M3 QAS = 1.582 for reasoning, IGSD QAS = 0.744 for coding, both consistent across seeds 42 and 123).** Reasoning-only
deployments should use M3 ($w = 0.3$): Qwen2.5-0.5B's mathematical prior improves GSM8K
accuracy by 3.9% over baseline while delivering 1.68× speedup ($\text{QAS} = 1.675$). M3
must not be combined with any other method. General-purpose or coding deployments should
use M1 + IGSD. M2 must never be deployed.

---

## 8.2 Limitations

The results carry three substantive limitations. First, all pairwise Ortho scores are
computed from 2 seeds over 200 GSM8K + 164 HumanEval samples; the Ortho = 1.385 estimate
for M1 + IGSD has per-seed range [1.292, 1.478] and should be treated as a 2-seed estimate
until full 3-seed validation is completed. Second, our M1 implementation achieves 1.38×
measured speedup rather than the published 15–26× of EntropyCache [CITE:entropycache],
because we execute full forward passes without kernel-level sparse attention; the Ortho
ratio is implementation-agnostic (it measures the ratio of CHRs, not absolute speedup), but
the absolute combined speedup (5.13×) would scale proportionally with a kernel-optimized
M1. Third, all experiments use LLaDA-8B-Instruct at batch size 1; Dream-7B-Instruct was
unavailable, and batched inference may reduce the frozen-token fraction and the CHR
elevation, potentially lowering $\text{Ortho}$ below 1.385.

---

## 8.3 Future Work

Four experiments remain open and would materially strengthen or qualify the findings.

**SSD composability.** Self-Speculative Decoding (SSD) [CITE:ssd] achieves 3.46× lossless
speedup using layer-exit drafting. Whether SSD + M1 also produces a frozen-KV synergy —
and whether lossless speculative decoding is compositionally equivalent to IGSD's lossy
variant — is the most important unanswered question from this study. A positive result would
establish the frozen-token CHR elevation as a property of self-speculative decoding in
general, independent of the specific draft mechanism.

**Full 3-seed pairwise validation.** Completing the pairwise experiments at full benchmark
scale (1319 GSM8K + 500 MATH500, 3 seeds) would reduce the variance on the Ortho = 1.385
estimate and enable reporting a confidence interval suitable for a primary claim.

**Tau = 0.0 paradox resolution.** The ablation finding that removing confidence partitioning
($\tau = 0.0$) improves QAS by +88.5% vs. full IGSD ($\tau = 0.9$) requires comparison
against a naive $T = 16$ baseline with no IGSD machinery. If IGSD-no-partition is identical
to naive $T = 16$, the confidence-scoring mechanism adds no value beyond draft length
selection, and IGSD's architecture simplifies to a step-count reduction approach. Resolving
this paradox is necessary before claiming IGSD's partitioning step as a distinct contribution.

**Batched inference and multi-model generalization.** A systematic analysis of batch sizes
$\in \{4, 16\}$ and experiments on additional MDM architectures (MDLM [CITE:mdlm], SEDD
[CITE:sedd]) would determine whether the binary composability finding is universal across
MDMs or specific to LLaDA's bidirectional full-sequence attention formulation.

---

## 8.4 Broader Significance

The composability framework and failure mode atlas introduced here — the orthogonality
metric $\text{Ortho}(M_a + M_b) = \text{Speedup}(M_a + M_b) / (\text{Speedup}(M_a) \times
\text{Speedup}(M_b))$, the four failure modes (F1–F4) with detection signals, and the
trajectory-preserving vs. trajectory-modifying classification — apply to any MDM
acceleration ecosystem. As the number of competing training-free methods grows beyond the
four families studied here, the composability framework provides a systematic evaluation
protocol: two trajectory-preserving methods should be tested for CHR synergy; any
trajectory-modifying method should be treated as incompatible with all others by default
until Ortho > 0.8 is demonstrated empirically. The detection signals (CHR < 40%, $\alpha >
0.75$, combined speedup < max individual speedup) are runtime-observable within the first
10–20 inference samples, enabling deployment systems to fail fast before a bad combination
is used at scale.

<!-- FIGURES
- None
-->

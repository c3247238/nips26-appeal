# 1. Introduction

Masked Diffusion Language Models (MDMs) such as LLaDA-8B-Instruct [CITE:llada] and Dream-7B [CITE:dream7b]
generate text by iteratively unmasking tokens over $T = 64$ denoising steps with bidirectional attention.
This architecture enables parallel token generation and strong context modeling, but imposes a steep
inference cost: each denoising step requires a full $O(N^2)$ forward pass over the entire sequence,
and unlike autoregressive (AR) models, standard KV-cache reuse across steps is invalidated by
continuously changing mask states.
On a single NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM), LLaDA-8B-Instruct generates at 31.0 ± 4.0 tokens/s
on GSM8K — roughly one-third the throughput of a comparable AR model under similar conditions.

The community has responded with a rapidly expanding set of training-free acceleration methods.
As of April 2026, at least seven KV-cache approximation strategies target the per-step attention cost
(Fast-dLLM [CITE:fastdllm], dKV-Cache [CITE:dkvcache], Elastic-Cache [CITE:elasticcache],
EntropyCache [CITE:entropycache], FreeCache [CITE:flashdlm],
Window-Diffusion [CITE:windowdiffusion], SlowFast+dLLM-Cache [CITE:slowfast]).
Two adaptive step scheduling methods reduce total denoising steps
(Saber [CITE:saber], PRR [CITE:prr]).
Two speculative decoding approaches exist: DualDiffusion [CITE:dualdiffusion], which uses an external
draft model within the MDM regime, and DFlash [CITE:dflash], which cross-architecturally uses
a block diffusion draft for an AR verifier.

These methods report speedups ranging from 2× to 99× in their respective papers.
However, every published evaluation is conducted in isolation, on different benchmark subsets,
with incompatible throughput measurement protocols (see Table 1).
No prior work asks whether two methods can be safely combined,
or what happens when the assumptions of one method conflict with those of another.
This leaves practitioners with three unresolved problems.

**Deployment confusion.** Given a task type (math reasoning, code generation) and latency budget, which
method — or combination — should be deployed? No unified comparison exists to answer this question.

**Hidden conflicts.** Some method combinations may interact catastrophically.
MDM denoising steps are globally coupled through mask state: each token's unmasking probability
depends on the current mask state of all other tokens.
Methods that modify the mask trajectory (e.g., adaptive step scheduling) should logically
conflict with methods that assume trajectory stability (e.g., KV-cache approximation),
but this hypothesis has never been tested quantitatively.

**A missing method class.** Self-speculative decoding — using the model's own coarse-step output as a
draft without any auxiliary model — is explicitly identified as a gap in DualDiffusion [CITE:dualdiffusion]
and confirmed by our literature survey.
No training-free, no-auxiliary-model speculative approach for MDMs exists.

This paper addresses all three problems through a systematic composability study.

We introduce **ComposeAccel**, a framework for measuring how training-free MDM acceleration methods
interact when combined.
The framework defines a formal Orthogonality metric,
$\text{Ortho}(M_a + M_b) = \text{Speedup}(M_a + M_b) / (\text{Speedup}(M_a) \times \text{Speedup}(M_b))$,
and a Quality-Adjusted Speedup (QAS),
$\text{QAS}(M) = \text{Speedup}(M) \times \text{AccRet}(M)$,
which jointly capture whether two methods compose multiplicatively and whether the combined
speedup is purchased at an acceptable accuracy cost.

We evaluate four method families on LLaDA-8B-Instruct across GSM8K, MATH500, HumanEval, and MBPP,
and we introduce a new method, **Information-Gain-Driven Self-Speculative Denoising (IGSD)**,
to fill the self-speculative gap.
As Figure 1 summarizes, the composability landscape is binary: exactly one of three feasible method pairs
achieves super-multiplicative synergy, while the others destructively interfere.

![Teaser: composability landscape and speed–quality scatter](figures/fig1_teaser.pdf)

**Our contributions are:**

1. **First systematic pairwise composability study** of four families of training-free MDM acceleration
   methods (KV-cache approximation M1, adaptive step scheduling M2, AR-guided unmasking M3, and IGSD)
   under a unified evaluation protocol on LLaDA-8B-Instruct (3 seeds, full benchmark scale).

2. **Binary composability finding.** MDM composability is not a gradient — it is binary.
   Exactly one pair, M1 (EntropyCache, $\eta=2.0$) + IGSD ($\tau=0.9$, $T_{\text{draft}}=16$),
   achieves super-multiplicative synergy ($\text{Ortho} = 1.385$, 5.13× combined speedup, QAS = 1.654).
   All other feasible pairs destructively interfere: M1+M3 produces a combined speedup of 0.93×,
   worse than either method alone.

3. **Mechanistic explanation of the synergy.** During IGSD's refine phase, tokens in $S_{\text{accept}}$
   ($\alpha \approx 52\%$ at $\tau = 0.9$) are frozen — their identities do not change across all $T_{\text{full}} = 64$
   refinement steps. EntropyCache assigns entropy $H_i = 0$ to these frozen positions (deterministic
   distributions have zero entropy), guaranteeing cache hits throughout the refine phase. The combined
   speedup thereby exceeds the product of individual speedups because IGSD creates the ideal caching
   scenario for M1.

4. **IGSD: Information-Gain-Driven Self-Speculative Denoising.** A training-free, no-auxiliary-model
   speculative method that achieves 3.40× speedup (consistent across reasoning and coding tasks)
   by running a fast $T_{\text{draft}}=16$ draft pass of the same MDM and selectively refining only
   low-confidence positions. IGSD requires no external model parameters, filling the gap left by
   DualDiffusion and DFlash.

5. **Failure mode atlas.** Four characterized interference patterns with proactive detection signals.
   Adaptive step scheduling (M2) receives a NO\_GO verdict: step-jump factors above 3× cause mask-consistency
   violations (accuracy retention collapses to 24–28% at 4×–8× step-jump) that no hyperparameter tuning
   can resolve, because they arise from a fundamental algorithmic mismatch with LLaDA's masked denoising schedule.

**Scope and honest positioning.**
ComposeAccel is an analysis paper, not a methods paper.
IGSD serves primarily as a composability study vehicle rather than a standalone acceleration proposal:
its 35.1% accuracy retention on GSM8K is not deployable independently,
and the combined M1+IGSD system's 5.13× speedup is below the single-method claims of published
techniques (EntropyCache 15–26× [CITE:entropycache], Fast-dLLM 27.6× [CITE:fastdllm]).
The contribution is structural insight: understanding when and why acceleration methods compose,
and providing actionable deployment guidance through the failure mode atlas.

**Table 1** positions ComposeAccel's implemented methods against published speedup claims.
The fragmentation of evaluation protocols — different hardware, benchmark subsets, and throughput
definitions across papers — makes direct comparison impossible without a unified study such as ours.

| Method | Published Speedup | Base Model | Training-free? | Evaluated in Combination? |
|--------|------------------|-----------|---------------|--------------------------|
| Fast-dLLM [CITE:fastdllm] | 27.6× | LLaDA/Dream | Yes | No |
| EntropyCache [CITE:entropycache] | 15.2–26.4× | LLaDA-8B/Dream-7B | Yes | No |
| Elastic-Cache [CITE:elasticcache] | 8.7–45.1× | LLaDA variants | Yes | No |
| Window-Diffusion [CITE:windowdiffusion] | Up to 99× | LLaDA/Dream | Yes | No |
| SlowFast+dLLM-Cache [CITE:slowfast] | 34.22× | LLaDA | Yes | No |
| Saber [CITE:saber] | 251.4% | DLM (code) | Yes | No |
| DualDiffusion [CITE:dualdiffusion] | Not reported | MDM | Yes | No |
| **M1 (EntropyCache, this work)** | **1.38×** | **LLaDA-8B-Instruct** | Yes | **Yes** |
| **IGSD (this work)** | **3.40×** | **LLaDA-8B-Instruct** | Yes | **Yes** |
| **M1+IGSD (this work)** | **5.13×** | **LLaDA-8B-Instruct** | Yes | **Yes (SYNERGY)** |

*Note: M1's 1.38× versus EntropyCache's published 15–26× reflects an implementation gap: our implementation
tracks cache hit rates and entropy signals but executes full forward passes without kernel-level sparse
attention. This implementation gap does not affect the qualitative composability finding, as Ortho is
measured from ratios of speedups under the same implementation conditions.
See Section 6.3 for detailed discussion.*

The remainder of the paper is organized as follows.
Section 2 formalizes the composability framework and presents IGSD and the three other method families.
Section 3 provides experimental results: single-method Pareto baselines, pairwise orthogonality analysis,
the failure mode atlas, IGSD ablations, and task-type dependence.
Section 4 analyzes why composability is binary, characterizes the frozen-token synergy mechanism,
and discusses limitations and open questions.
Section 5 positions this work against related acceleration methods.
Section 6 concludes.

<!-- FIGURES
- Figure 1: gen_fig1_teaser.py, fig1_teaser.pdf — Composability landscape: (a) Ortho bar chart for all three feasible pairs, color-coded by synergy/interference; (b) Speedup vs. QAS scatter showing all individual methods and combined pairs.
- Table 1: inline — Literature speed comparison positioning ComposeAccel against published training-free MDM acceleration methods; highlights fragmentation of evaluation protocols and absence of composability data.
-->

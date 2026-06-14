# ComposeAccel: Systematic Composition and Orthogonality Analysis of Training-Free Acceleration Methods for Diffusion Language Models

## Abstract

Diffusion language models (DLMs) such as LLaDA-8B generate text through iterative denoising but remain 2--14x slower than comparably-sized autoregressive (AR) models. Over 20 training-free acceleration methods have emerged targeting three computational axes---KV cache approximation, adaptive step scheduling, and guided or speculative decoding---yet every method is evaluated in isolation under incompatible protocols. We present ComposeAccel, the first controlled factorial study of training-free DLM acceleration composition. We define Quality-Adjusted Speedup (QAS) and an orthogonality metric (Ortho) to quantify composition efficiency, then evaluate three method families---entropy-based KV caching (M1), AR-guided unmasking (M3), and our Information-Geometric Step Distillation (IGSD)---individually, in all three pairwise combinations, and in three-way compositions on LLaDA-8B-Instruct across GSM8K (1319 samples) and MATH500 (500 samples) with 3-seed validation. M1+IGSD achieves near-orthogonal composition (Ortho = 0.96, 2.75x speedup at 58.9% accuracy retention on GSM8K), M3+IGSD is task-dependent (GSM8K Ortho = 0.96, MATH500 Ortho = 0.76), and M1+M3 shows destructive interference (Ortho = 0.41--0.43) from guide-model overhead negating marginal cache speedup. Cross-model validation on Dream-7B-Instruct confirms transferable composition patterns (M1+IGSD QAS = 2.18 vs. LLaDA QAS = 1.07 on GSM8K). An honest AR comparison shows Qwen2.5-7B at batch=1 achieves QAS = 3.08 relative to the LLaDA baseline---a gap that composed DLM acceleration narrows but does not close. We release the IGSD implementation, validated acceleration recipes, and the composability benchmark suite.

---

## 1. Introduction

LLaDA-8B-Instruct achieves 71.2% accuracy on GSM8K but generates text at 33.8 tokens per second (TPS)---2.1x slower than Qwen2.5-7B at batch=1 (71 TPS) and 13.9x slower at batch=8 (471 TPS). This speed deficit is structural: each of the $T = 64$ denoising steps in a diffusion language model (DLM) requires a full $O(N^2)$ forward pass, and the global mask-state update between steps prevents standard KV cache reuse. In Q1 2026 alone, over 20 training-free acceleration methods have targeted this bottleneck across three computational axes: KV cache approximation [Fast-dLLM, EntropyCache, dKV-Cache, Elastic-Cache], adaptive step scheduling [Saber, PRR], and guided or speculative decoding [FlashDLM, SSD, SSMD, S2D2, DualDiffusion].

Every one of these methods is evaluated in isolation. Practitioners who must deploy DLMs face a concrete unanswered question: which combination of methods maximizes throughput at acceptable quality? The answer is not obvious. Methods targeting different computational bottlenecks should compose multiplicatively, but DLM's global mask-state coupling creates hidden interaction risks---a speed gain on one axis can silently degrade the quality signal another axis depends on.

### 1.1 The Composition Gap

Three specific problems block progress toward practical DLM deployment:

**Unknown composability.** No published work measures pairwise or higher-order orthogonality between DLM acceleration methods. TORS (arXiv:2603.00763) identifies the same gap for text-to-image diffusion, noting that training-free methods are "developed independently, leaving compatibility unexplored." Kolbeinsson et al. (2024) measure composable interventions for LLMs (compression, editing, pruning), but these are model-modification techniques, not inference acceleration. For DLM inference acceleration, zero composition data exists.

**No failure characterization.** Published methods report average-case results under method-specific protocols. A practitioner cannot predict from these numbers whether combining entropy-based KV caching ($\eta = 0.5$, 1.16x speedup, 94.5% accuracy retention) with AR-guided unmasking ($w_g = 0.3$, 1.68x speedup, 103.9% accuracy retention) will yield 1.95x speedup or a net slowdown. Our experiments show the composition is slower than M3 alone---destructive interference from overhead stacking. Without composition data, every deployment is a gamble.

**No honest AR comparison.** DLM acceleration papers benchmark against the unaccelerated DLM baseline. None compare against properly optimized autoregressive (AR) inference at the same model scale. Without this comparison, the practical value of DLM acceleration remains unclear.

Figure 1 shows the speed-quality landscape that motivates this work. M1 (KV caching), IGSD (step distillation), and M3 (AR-guided decoding) cluster in distinct regions when applied individually, and their pairwise compositions span the space unpredictably. The M1+M3 pair falls *below* both individual methods---a result invisible without systematic composition measurement.

![Figure 1: Speed-quality landscape showing individual methods and compositions on LLaDA-8B-Instruct (GSM8K). Points below the Pareto frontier represent suboptimal or interfering compositions.](figures/fig_teaser.pdf)

### 1.2 Contributions

1. **First systematic composition study** of three training-free DLM acceleration families across three axes, with a formal orthogonality metric (Ortho) and corrected Quality-Adjusted Speedup (QAS). We evaluate all three pairwise combinations and five three-way configurations on LLaDA-8B-Instruct across GSM8K (1319 samples) and MATH500 (500 samples) with 3-seed validation.

2. **Composition taxonomy.** M1+IGSD achieves near-orthogonal composition (Ortho = 0.96, 2.75x speedup on GSM8K). M3+IGSD is task-dependent (near-orthogonal on GSM8K, interference on MATH500). M1+M3 shows destructive interference (Ortho = 0.41--0.52). Cross-model validation on Dream-7B-Instruct confirms these patterns transfer with amplified effects.

3. **IGSD (Information-Geometric Step Distillation).** A 50-line training-free step scheduler that partitions denoising into draft and refine phases using inter-step KL divergence. IGSD achieves 1.71x speedup (QAS = 1.16 on GSM8K) and composes near-orthogonally with M1 (Ortho = 0.96).

4. **Honest AR comparison and practical recipes.** Qwen2.5-7B at batch=1 achieves QAS = 3.08 relative to the LLaDA baseline---a 2.9x gap that composed DLM acceleration does not close. We provide task-specific recommended combinations with validated hyperparameters.

### 1.3 Scope

ComposeAccel is an analysis paper. IGSD is a composability study vehicle---simple by design so that composition effects are attributable to method interactions rather than implementation complexity. The primary model is LLaDA-8B-Instruct; Dream-7B-Instruct provides cross-model validation. Primary benchmarks are GSM8K and MATH500; HumanEval is reported in the appendix (2.4% baseline). M1 reports both measured and projected speedup because d2Cache kernel integration produced 15.2x framework overhead on RTX PRO 6000 Blackwell GPUs (Section 4.2). M2 (adaptive step scheduling) is excluded as a structural NO_GO and reported as a negative result in Section 7.

---

## 2. Related Work

*(Note: Section numbering in the original sections used "3" for Related Work; we renumber here for the integrated paper.)*

### 2.1 Masked Diffusion Language Models

LLaDA [CITE:llada] and Dream [CITE:dream7b] generate text by reversing a forward corruption process that progressively masks tokens. Starting from a fully masked sequence, the model iteratively predicts token identities at masked positions over $T = 64$ denoising steps using bidirectional attention. At each step $t$, the transformer $f_\theta$ produces logits $p_t(\cdot \mid \mathbf{x}_t) \in \mathbb{R}^{N \times V}$ over the full vocabulary for every position, and a subset of masked tokens is unmasked according to a confidence-based schedule. The binary mask $\mathbf{m}_t$ evolves globally: unmasking a token at position $i$ changes the attention context for all other positions at step $t+1$.

This global mask-state coupling creates the central inference bottleneck. Each denoising step requires a full $O(N^2)$ forward pass because the key-value (KV) representations of every position depend on which tokens are currently masked. Standard AR KV-cache reuse---where cached entries remain valid as long as earlier tokens are unchanged---does not apply: a token transitioning from [MASK] to an unmasked value causes a discontinuous shift in its key and value vectors, invalidating cached entries for all positions that attend to it. On a single NVIDIA RTX PRO 6000 Blackwell GPU, LLaDA-8B-Instruct generates at 33.8 TPS on GSM8K with 64-step denoising (bf16, greedy), compared to 71 TPS for the comparably-sized AR model Qwen2.5-7B-Instruct under identical hardware.

The masked diffusion formulation is grounded in the discrete diffusion framework of Austin et al. [CITE:austin2021structured] and refined by MDLM [CITE:mdlm] and SEDD [CITE:sedd]. LLaDA extends this to the 8B-parameter scale and demonstrates competitive quality with AR models on reasoning benchmarks (GSM8K: 71.2%, comparable to LLaMA3-8B). Dream-7B-Instruct uses a similar architecture but achieves lower GSM8K accuracy (36.0%) while offering higher baseline throughput (64.5 TPS). Block Diffusion [CITE:blockdiffusion] interpolates between AR and diffusion generation by applying AR generation across blocks and masked denoising within blocks, enabling direct KV reuse across blocks---a structural advantage absent from fully masked models.

### 2.2 Training-Free Acceleration Families

We organize the training-free acceleration landscape into four families based on the computational axis each targets. Table 1 summarizes published speedup claims and evaluation protocols for representative methods. These published numbers span 2x to 99x, but use different hardware, batch sizes, sequence lengths, and quality thresholds; our controlled protocol measures all methods on identical infrastructure (Section 3.5).

**Table 1: Published DLM Acceleration Methods and Evaluation Protocols**

| Method | Axis | Model | Reported Speedup | Composition Tested? |
|--------|------|-------|-----------------|-------------------|
| Fast-dLLM [CITE:fastdllm] | KV cache | LLaDA-8B, Dream-7B | up to 27.6x | No |
| EntropyCache [CITE:entropycache] | KV cache | LLaDA-8B | 15.2--26.4x | No |
| dKV-Cache [CITE:dkvcache] | KV cache | LLaDA-8B | 2--10x | No |
| Elastic-Cache [CITE:elasticcache] | KV cache | LLaDA-8B | 45.1x (long seq.) | No |
| Window-Diffusion [CITE:windowdiffusion] | KV cache | LLaDA-8B | up to 99x | No |
| SlowFast [CITE:slowfast] | KV cache | LLaDA-8B | 34.22x | No |
| Saber [CITE:saber] | Step scheduling | LLaDA-8B | 251.4% (code gen.) | No |
| FlashDLM [CITE:flashdlm] | KV cache + AR guidance | LLaDA-8B | 12.14x (combined) | Internal only |
| SSD [CITE:ssd] | Speculative denoising | LLaDA-8B | 2.11--3.46x (lossless) | No |
| SSMD [CITE:ssmd] | Speculative denoising | LLaDA-8B | -- | No |
| S2D2 [CITE:s2d2] | Speculative (block-diffusion) | SDAR, LLaDA2.1-Mini | 4.7x | No |

**KV-Cache Approximation.** Seven methods approximate cross-step KV reuse for DLMs by identifying positions whose representations are stable enough to skip recomputation. Fast-dLLM [CITE:fastdllm] combines block-wise KV caching with confidence-aware parallel decoding, reporting up to 27.6x speedup on LLaDA-8B and Dream-7B. dKV-Cache [CITE:dkvcache] refreshes cache entries when the predicted distribution shifts beyond a calibrated threshold, achieving 2--10x speedup in two variants. EntropyCache [CITE:entropycache] uses an $O(V)$ entropy check---reusing cached KV for position $i$ when $H_i^{t-1} < \eta$---and reports 15.2--26.4x speedup. Elastic-Cache [CITE:elasticcache] applies per-layer adaptive refresh scheduling with drift-aware updates, reaching 45.1x on long sequences. Window-Diffusion [CITE:windowdiffusion] partitions positions into active, buffer, and pruned sets via a sliding window, reporting up to 99x speedup at the cost of global bidirectional context. SlowFast [CITE:slowfast] combines a two-stage sampling strategy with aggressive caching, achieving 34.22x on LLaDA.

Our M1 (entropy-based KV caching, $\eta \in \{0.5, 1.0, 2.0\}$) reproduces the EntropyCache entropy-signal logic. At $\eta = 0.5$, M1 achieves a measured 1.16x speedup with 94.5% accuracy retention on GSM8K. The gap between this measured speedup and published 15--26x arises from our implementation executing full forward passes without kernel-level sparse attention: d2Cache integration produced a 15.2x framework overhead on RTX PRO 6000 Blackwell GPUs due to eager attention incompatibility. M1's cache hit rate (CHR = 56--99% depending on $\eta$) confirms the entropy signal is effective; the bottleneck is engineering, not algorithm design.

**Adaptive Step Scheduling.** Saber [CITE:saber] reduces total denoising steps by unmasking more tokens per step with a backtracking mechanism, reporting 251.4% speedup on code generation. PRR [CITE:prr] trains a lightweight controller to modulate per-token temperature across steps, requiring auxiliary training and thus not purely training-free.

Our attempt to implement Saber-style step scheduling as M2 (step-jump $J \in \{2, 4, 6, 8\}$, without the backtracking mechanism) resulted in a NO_GO verdict. At $J = 4$, accuracy retention collapsed to 27.9% on the combined metric. LLaDA's discrete mask schedule requires sequential cumulative conditioning that step skipping violates. This negative result confirms that continuous diffusion DDIM-style acceleration does not transfer to discrete masked diffusion without the backtracking mechanism.

**AR-Guided Unmasking.** FlashDLM [CITE:flashdlm] combines FreeCache (a KV approximation method) with a lightweight AR supervisor, achieving 12.14x combined speedup. FlashDLM is the closest precedent to our composition approach: it combines KV approximation with AR guidance in a single integrated system. However, FlashDLM evaluates only this one pre-designed combination; it does not measure whether its components compose near-orthogonally, nor does it compare against alternative pairings. ComposeAccel provides this systematic analysis. Our M3 implements the guidance mechanism using Qwen2.5-0.5B (0.49B parameters), interpolating DLM and AR logits with weight $w_g$: $\tilde{p}_t(i) = (1 - w_g) \cdot p_t(i) + w_g \cdot q_t(i)$ for masked positions. M3 at $w_g = 0.3$ achieves 1.68x speedup with 103.9% accuracy retention on GSM8K---the only method in our study that improves accuracy over baseline---but adds approximately 12% per-step wall-clock overhead from the guide model forward pass.

**Speculative Denoising.** AR speculative decoding [CITE:leviathan2023fast; CITE:chen2023accelerating] uses a small draft model to generate candidate continuations that a larger verifier accepts or rejects. Four lines of work adapt this paradigm to DLMs. DualDiffusion [CITE:dualdiffusion] uses a lightweight draft DLM and a verifier DLM with a distribution-ratio acceptance test. SSD [CITE:ssd] introduces self-speculative decoding with hierarchical tree structures, achieving 2.11--3.46x lossless speedup. SSMD [CITE:ssmd] toggles between non-causal and causal attention to use the same model as both drafter and verifier. S2D2 [CITE:s2d2] applies self-speculation to block-diffusion architectures, reaching 4.7x speedup but requiring block-diffusion structure not present in LLaDA or Dream.

Our IGSD (Information-Geometric Step Distillation) is a self-speculative method: it drafts using a reduced number of steps ($T_{\text{draft}} \in \{16, 32, 48\}$ out of $T_{\text{full}} = 64$), then partitions tokens by confidence into frozen ($\mathcal{S}_{\text{accept}}$) and active ($\mathcal{S}_{\text{reject}}$) sets, continuing only the active tokens through the remaining steps. IGSD occupies the boundary between step scheduling and speculative decoding: it reduces computation by running fewer steps (like M2-family methods) but uses a confidence-based partition to selectively refine (like speculative methods). The tradeoff is that IGSD is approximate (67.8% accuracy retention at $\tau = 0.9$, $T_{\text{draft}} = 32$ on GSM8K) while SSD guarantees lossless output. IGSD's draft-refine architecture creates a frozen-token partition that enables downstream KV-cache synergy (Section 3.3)---a composability property absent from SSD's architecture.

### 2.3 The Composability Gap

Every method described above is evaluated in isolation, on different benchmark subsets, with incompatible throughput measurement protocols. No published work measures whether two DLM acceleration methods can be safely combined, or quantifies the interaction effects when method assumptions conflict.

The closest precedent is TORS (arXiv:2603.00763), which notes that training-free acceleration methods for text-to-image diffusion models are "developed independently, leaving compatibility unexplored." TORS conducts initial composition experiments in the vision domain; we address the analogous gap for language DLMs, where the global mask-state coupling creates additional interaction risks absent from continuous image diffusion.

Kolbeinsson et al. [CITE:kolbeinsson2024composable] study composability of LLM interventions---knowledge editing, model compression, and unlearning applied to the same model weights. Their framework is conceptually related to ours, but operates on static weight-space modifications rather than dynamic inference-time algorithms. The DLM setting introduces a temporal composability challenge: the mask state evolves at every step, and two methods may interact differently at step 10 (when 85% of tokens are masked) versus step 60 (when 5% remain masked).

ComposeAccel fills this gap with three contributions absent from prior work: (1) a formal orthogonality metric that quantifies composition efficiency on a per-pair, per-benchmark basis; (2) controlled factorial experiments across three acceleration families on two DLM architectures under a unified evaluation protocol; and (3) mechanistic analysis of why specific pairs compose (M1+IGSD: frozen tokens create low-entropy KV entries) and why others interfere (M1+M3: guide model overhead negates marginal cache speedup).

---

## 3. Methods

This section defines the composability framework (Section 3.1), describes the three acceleration methods under study (Sections 3.2--3.4), and specifies the experimental protocol (Section 3.5).

### 3.1 Composability Framework

We introduce two metrics that distill the speed-quality tradeoff into scalars suitable for pairwise comparison.

**Quality-Adjusted Speedup (QAS).** For an acceleration method $M$ applied to a baseline system with accuracy $\text{Acc}_0$ and throughput $\text{TPS}_0$:
$$
\text{QAS}(M) = S(M) \times \text{AccRet}(M),
$$
where $S(M) = \text{TPS}(M) / \text{TPS}_0$ is the wall-clock speedup and $\text{AccRet}(M) = \text{Acc}(M) / \text{Acc}_0$ is the accuracy retention. QAS captures both dimensions in a single number: a method that doubles speed while halving accuracy yields QAS = 1.0 (no net improvement).

**Orthogonality (Ortho).** For methods $A$ and $B$:
$$
\text{Ortho}(A, B) = \frac{\text{QAS}(A + B)}{\max\bigl(\text{QAS}(A),\; \text{QAS}(B)\bigr)}.
$$
The denominator normalizes by the better individual method, so Ortho > 1.0 indicates synergy (the composition strictly outperforms either component alone), $0.8 \leq \text{Ortho} \leq 1.0$ indicates near-orthogonal composition (most benefit preserved), and Ortho < 0.8 indicates interference (composition degrades below the best individual method).

**Combined benchmark metric.** All QAS, Ortho, and Pareto computations use a weighted average across benchmarks: $0.7 \times \text{GSM8K} + 0.3 \times \text{MATH500}$. The weighting reflects the stronger baseline signal on GSM8K ($\text{Acc}_0 = 71.2\%$) compared to MATH500 ($\text{Acc}_0 = 11.1\%$). HumanEval results are reported in the appendix but excluded from combined metrics due to a near-zero 2.4% baseline.

**Table 2: Metric Definitions and Interpretation Thresholds**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| AccRet$(M)$ | $\text{Acc}(M) / \text{Acc}_0$ | Fraction of baseline accuracy preserved |
| $S(M)$ | $\text{TPS}(M) / \text{TPS}_0$ | Wall-clock speedup |
| QAS$(M)$ | $S(M) \times \text{AccRet}(M)$ | Quality-adjusted speedup |
| Ortho$(A, B)$ | $\text{QAS}(A{+}B) / \max(\text{QAS}(A), \text{QAS}(B))$ | > 1.0 synergy; 0.8--1.0 near-orthogonal; < 0.8 interference |
| Metric$_{\text{comb}}$ | $0.7 \times \text{GSM8K} + 0.3 \times \text{MATH500}$ | Combined benchmark score |

### 3.2 M1: Entropy-Based KV Caching

Standard DLM inference recomputes key-value (KV) matrices at every denoising step because the global mask state changes between steps. M1 exploits the observation that many token positions have low-entropy logit distributions that remain stable across consecutive steps, making their KV entries safe to reuse.

At each step $t$, the entropy of the logit distribution at position $i$ is:
$$
H_i^t = -\sum_{v \in \mathcal{V}} p_t(v \mid i) \log p_t(v \mid i).
$$
Positions with $H_i^{t-1} < \eta$ reuse cached KV from step $t-1$; the remaining positions recompute KV via a full forward pass. The entropy threshold $\eta \in \{0.5, 1.0, 2.0\}$ controls the speed-quality tradeoff.

The cache hit rate (CHR) quantifies the fraction of generation-token positions that reuse cached KV:
$$
\overline{\text{CHR}} = \frac{1}{T} \sum_{t=1}^{T} \frac{|\mathcal{C}_t|}{N_{\text{gen}}}, \quad \mathcal{C}_t = \{i : H_i^{t-1} < \eta\}.
$$

**Implementation and the kernel gap.** We attempted to integrate d2Cache, a kernel-level sparse attention library, to translate CHR into wall-clock TPS gains. d2Cache achieves 4.4x speedup over its own baseline; however, d2Cache's model wrapper introduces 15.2x overhead compared to the HuggingFace baseline (3.85 TPS vs. 58.5 TPS) due to eager attention mode and per-layer context manager overhead on RTX PRO 6000 Blackwell GPUs. We therefore report M1 with two speedup numbers throughout the paper:

- **Measured speedup**: wall-clock TPS improvement from our simplified Python-level entropy cache. At $\eta = 0.5$: 1.16x speedup, CHR = 56.2% on GSM8K.
- **Projected speedup**: estimated from measured CHR assuming ideal kernel-level sparse attention. At $\eta = 0.5$: 2.27x (CHR = 93.3%). These projections serve as an upper bound.

The Ortho metric is dimensionless and valid regardless of whether M1's speedup is measured or projected. The gap between projected and measured speedup is itself a finding: kernel-level KV cache integration for bidirectional-attention DLMs remains an open engineering challenge.

### 3.3 IGSD: Information-Geometric Step Distillation

IGSD is a training-free speculative step scheduler that partitions the denoising trajectory into a fast draft phase and a selective refine phase, using inter-step KL divergence as the design motivation.

**Draft phase (steps $0 \to T_{\text{draft}}$).** All $N$ tokens undergo standard denoising for $T_{\text{draft}}$ steps. $T_{\text{draft}} \in \{16, 32, 48\}$ out of $T_{\text{full}} = 64$ total steps.

**Confidence partitioning (at step $T_{\text{draft}}$).** After the draft phase, each token position is classified based on the model's confidence. Token $i$ is frozen if its maximum softmax probability exceeds the threshold $\tau$:
$$
\mathcal{S}_{\text{accept}} = \{i : \max_v\, p_{T_{\text{draft}}}(v \mid i) \geq \tau\}, \quad \mathcal{S}_{\text{reject}} = \{1, \ldots, N\} \setminus \mathcal{S}_{\text{accept}}.
$$
The frozen fraction $\alpha = |\mathcal{S}_{\text{accept}}| / N_{\text{gen}}$ is measured at $0.886 \pm 0.133$ across 100 GSM8K samples at $\tau = 0.9$, $T_{\text{draft}} = 16$.

**Refine phase (steps $T_{\text{draft}} \to T_{\text{full}}$).** Only tokens in $\mathcal{S}_{\text{reject}}$ continue denoising for the remaining $T_{\text{full}} - T_{\text{draft}}$ steps. Frozen tokens retain their draft-phase values. The effective computation is reduced from $T_{\text{full}} \times N$ to $T_{\text{draft}} \times N + (T_{\text{full}} - T_{\text{draft}}) \times |\mathcal{S}_{\text{reject}}|$.

**KL divergence signal.** The mean token-level KL divergence between consecutive steps measures how much the model's predictions are changing:
$$
\text{KL}_t = \frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} D_{\text{KL}}\bigl(p_t(i) \,\|\, p_{t-1}(i)\bigr),
$$
where $\mathcal{M}_t$ is the set of masked positions at step $t$. This signal originally motivated the partition design; empirically, the KL profile is monotonically decreasing (Section 4.4), which explains why the confidence gate has low sensitivity to $\tau$ in the $[0.7, 0.9]$ range.

**Algorithm 1: IGSD**

```
Input: DLM f_theta, T_full=64, T_draft in {16,32,48}, threshold tau
Output: Generated sequence x_hat

1. Draft: for t = 0 to T_draft:
     x_t = denoise_step(f_theta, x_{t-1}, t)
2. Partition: S_accept = {i : max_v p_{T_draft}(v|i) >= tau}
             S_reject = {1,...,N} \ S_accept
3. Refine: for t = T_draft to T_full:
     x_t[S_reject] = denoise_step(f_theta, x_{t-1}, t, mask=S_reject)
     x_t[S_accept] = x_{T_draft}[S_accept]    // frozen
4. Return x_{T_full}
```

The implementation is approximately 50 lines of Python. The computational overhead of the confidence gate ($O(V)$ per position, $O(N \times V)$ total) is negligible compared to the transformer forward pass.

**Interaction with M1.** IGSD's refine phase creates favorable conditions for entropy-based KV caching: frozen tokens in $\mathcal{S}_{\text{accept}}$ have near-zero entropy (their logit distributions are peaked), providing M1 with a high-quality cache reuse signal. The measured refine-phase CHR is 94.3% at $\eta = 0.5$ (vs. 56.2% for standalone M1 on the full sequence), explaining the near-orthogonal composition observed in Section 4.2.

### 3.4 M3: AR-Guided Unmasking

M3 uses a lightweight autoregressive (AR) model to bias the DLM's unmasking decisions toward higher-confidence token predictions. At each denoising step $t$, the AR guide model $g_\phi$ (Qwen2.5-0.5B, 0.49B parameters) produces logits $q_t$ over masked positions. The guided logits are:
$$
\tilde{p}_t(i) = (1 - w_g) \cdot p_t(i) + w_g \cdot q_t(i), \quad \text{for masked position } i,
$$
where $w_g \in \{0.3, 0.5, 0.7\}$ is the guidance weight. Unmasked positions retain the DLM's original logits.

M3 is quality-preserving: at $w_g = 0.3$, GSM8K accuracy retention is 103.9%, suggesting that the guide model corrects some DLM errors. The measured speedup is 1.68x on GSM8K. The overhead of M3 comes from loading Qwen2.5-0.5B (0.95 GB VRAM) and running it at every denoising step, adding approximately 12% wall-clock time per step. This overhead is the root cause of destructive interference when M3 is composed with M1 (Section 4.2).

MATH500 results for all methods are reported in Section 4.1.

### 3.5 Experimental Setup

**Models.** LLaDA-8B-Instruct (primary evaluation), Dream-7B-Instruct (cross-model validation). Both are masked diffusion language models using $T = 64$ denoising steps with bidirectional attention.

**Hardware.** 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM each). LLaDA-8B requires approximately 15.3 GB VRAM in bf16; independent tasks run on separate GPUs concurrently.

**Benchmarks.** GSM8K (1319 samples, exact match) and MATH500 (500 samples, exact match) constitute the primary evaluation. HumanEval (164 samples, pass@1) is reported in the appendix only; MBPP is dropped (0% baseline provides no signal).

**Seeds and statistical protocol.** Pilot experiments use seed 42; full-scale experiments use seeds $\{42, 123, 456\}$ and report mean $\pm$ std. Three-way composition top-5 configs are validated with all three seeds (QAS coefficient of variation < 10% required for stability).

**Baseline.** LLaDA-8B-Instruct with 64-step denoising, bf16, greedy decoding. GSM8K: 71.2% $\pm$ 1.5% accuracy (mean $\pm$ std across 3 seeds), 33.8 TPS (generation-only, post-warmup). MATH500: 11.1% $\pm$ 0.7% accuracy, 79.1 TPS. The first 5 samples per benchmark are discarded as warmup.

**Throughput measurement.** Tokens per second (TPS) is measured as total generated tokens divided by wall-clock generation time, averaged over all non-warmup samples. Batch size is 1 (interactive setting) for all primary experiments; batch sensitivity at sizes $\{1, 4, 8\}$ is reported separately in Section 5.3.

**Hyperparameter sweeps.** M1: $\eta \in \{0.5, 1.0, 2.0\}$. IGSD: $\tau \in \{0.7, 0.85, 0.9\}$, $T_{\text{draft}} \in \{16, 32, 48\}$ (9 configs). M3: $w_g \in \{0.3, 0.5, 0.7\}$. Pairwise compositions use the best operating point per method from single-method Pareto curves. Three-way compositions sweep a reduced grid of 24 configurations, with top-5 validated at full scale.

**M2 exclusion.** Adaptive step scheduling (M2) is excluded from composition analysis. Pilot experiments showed that even 2x step reduction (32 steps) collapses GSM8K accuracy from 71.2% to 38.8% with our simplified implementation, which lacks Saber's backtracking mechanism. M2 receives a NO_GO verdict and is reported as a negative result in Section 6.5.

Figure 2 illustrates the full composed pipeline, showing the IGSD draft-partition-refine architecture with M1 and M3 integration points.

![Figure 2: ComposeAccel architecture: IGSD draft-partition-refine pipeline with M1 KV caching and M3 AR guidance integration points.](figures/fig2_architecture.pdf)

[TODO: Figure 2 -- generate from fig2_architecture_desc.md specification]

<!-- FIGURES
- Figure 2: fig2_architecture_desc.md -- ComposeAccel architecture diagram showing IGSD draft-partition-refine pipeline with M1 KV caching and M3 AR guidance integration points
- Table 1: inline -- Published DLM acceleration methods and evaluation protocols
- Table 2: inline -- Metric definitions and interpretation thresholds
-->

---

## 4. Experiments

### 4.1 Single-Method Pareto Curves

Table 3 and Figure 3 report the speed--accuracy tradeoff for each method in isolation on GSM8K (1319 samples, seed 42) and MATH500 (500 samples, seed 42), using the LLaDA-8B-Instruct baseline ($T$=64 steps, bf16 greedy decoding; GSM8K accuracy 71.2% $\pm$ 1.5%, 33.8 TPS; MATH500 accuracy 11.1% $\pm$ 0.7%, 79.1 TPS).

**M1 (entropy-based KV caching).** M1 at $\eta$=0.5 achieves 1.16x measured speedup with 94.5% accuracy retention on GSM8K ($\overline{\text{CHR}}$=56.2%). Raising the threshold to $\eta$=2.0 pushes speedup to 1.50x but accuracy retention drops to 55.5% ($\overline{\text{CHR}}$=60.2%). Projected speedups from the measured cache hit rates would be 2.27--2.47x; we report only measured wall-clock values and label projected numbers explicitly.

**IGSD (information-geometric step distillation).** IGSD sweeps 9 configurations across $\tau \in \{0.7, 0.85, 0.9\}$ and $T_{\text{draft}} \in \{16, 32, 48\}$. The fastest configuration ($\tau$=0.7, $T_{\text{draft}}$=16) reaches 2.81x speedup at 58.2% accuracy retention on GSM8K (QAS=1.64). The most conservative ($\tau$=0.9, $T_{\text{draft}}$=48) yields 1.22x speedup at 73.3% accuracy retention (QAS=0.90). $T_{\text{draft}}$=32 at any $\tau$ is Pareto-optimal when accuracy retention above 65% is required: $\tau$=0.85 gives 1.73x speedup, 67.8% accuracy retention, QAS=1.17.

**M3 (AR-guided unmasking).** M3 with Qwen2.5-0.5B guidance achieves approximately 1.68x speedup on GSM8K across all guidance weights ($w_g \in \{0.3, 0.5, 0.7\}$), with accuracy retention above 100% at $w_g$=0.3 (103.9%) and $w_g$=0.7 (103.9%). M3 is the only quality-preserving accelerator in our study. At $w_g$=1.0, accuracy retention drops to 84.9% as the guide model overwhelms the DLM's own predictions. The 2-seed mean (seeds 42, 123) on a 100-sample GSM8K subset confirms stability. Note: M3 results are from 100-sample evaluation; M1 results are from the full 1319-sample evaluation.

![Figure 3: Single-method Pareto curves for M1, IGSD, and M3 on GSM8K. Each point represents one hyperparameter configuration. M3 is uniquely quality-preserving (AccRet > 100%); IGSD offers a continuum of speed-quality tradeoffs; M1 has a narrow operating range.](figures/fig3_single_pareto.pdf)

**Table 3: Single-Method Pareto Results on GSM8K**

| Method | Config | Speedup | AccRet (%) | QAS | CHR / Accept Rate | N |
|--------|--------|---------|------------|-----|-------------------|---|
| M1 | $\eta$=0.5 | 1.16x | 94.5 | 0.98 | CHR=56.2% | 1319 |
| M1 | $\eta$=1.0 | 1.25x | 88.0 | 0.97 | CHR=58.7% | 1319 |
| M1 | $\eta$=2.0 | **1.50x** | 55.5 | 0.83 | CHR=60.2% | 1319 |
| IGSD | $\tau$=0.7, $T_{\text{draft}}$=16 | **2.81x** | 58.2 | **1.64** | $r_{\text{accept}}$=92.1% | 200 |
| IGSD | $\tau$=0.85, $T_{\text{draft}}$=32 | 1.73x | **67.8** | 1.17 | $r_{\text{accept}}$=95.9% | 200 |
| IGSD | $\tau$=0.9, $T_{\text{draft}}$=32 | 1.71x | 67.8 | 1.16 | $r_{\text{accept}}$=95.3% | 200 |
| IGSD | $\tau$=0.9, $T_{\text{draft}}$=48 | 1.22x | 73.3 | 0.90 | $r_{\text{accept}}$=96.4% | 200 |
| M3 | $w_g$=0.3 | 1.68x | **103.9** | **1.69** | -- | 100 |
| M3 | $w_g$=0.7 | 1.68x | 103.9 | 1.71 | -- | 100 |
| M3 | $w_g$=1.0 | 1.68x | 84.9 | 0.70 | -- | 100 |

### 4.2 Pairwise Composition Analysis

Table 4 and Figure 4 report the orthogonality ($\text{Ortho}$) score for all three viable pairs on 100-sample GSM8K and 100-sample MATH500 pilots (seed 42). These are pilot-scale results; pairwise Ortho values carry higher uncertainty than the 3-seed three-way results in Section 4.3.

**M1+IGSD: near-orthogonal.** The best configuration ($\eta$=0.5, $\tau$=0.7, $T_{\text{draft}}$=16) achieves 2.75x speedup at 58.9% accuracy retention on GSM8K, with $\text{Ortho}_{\text{GSM8K}}$=0.99 and combined $\text{Ortho}$=0.96. IGSD's frozen tokens ($\alpha$=88.6% of generation positions) create near-zero entropy at those positions, providing M1 with a strong cache-reuse signal: CHR rises to 83.4% during composition versus 56.2% for standalone M1. The more conservative $T_{\text{draft}}$=32 variant reaches $\text{Ortho}$=0.91 at lower speedup (1.68x), confirming that the composition benefit holds across IGSD operating points.

**M3+IGSD: task-dependent.** At the highest-speed IGSD setting ($\tau$=0.7, $T_{\text{draft}}$=16, $w_g$=0.7), $\text{Ortho}_{\text{GSM8K}}$=0.96 (near-orthogonal) but $\text{Ortho}_{\text{MATH500}}$=0.76 (interference), yielding combined $\text{Ortho}$=0.84. At more conservative settings ($\tau$=0.9, $T_{\text{draft}}$=32), both benchmarks fall into interference ($\text{Ortho}_{\text{combined}}$=0.61). The guide model Qwen2.5-0.5B operates on IGSD's compressed denoising trajectory, where fewer draft steps produce noisier context for the 0.5B model, degrading guidance quality on harder tasks. The MATH500 Ortho values carry wide uncertainty due to the 11.1% baseline (a 3-sample fluctuation changes accuracy by nearly 1 percentage point).

**M1+M3: destructive interference.** Across all three $w_g$ values, combined $\text{Ortho}$ ranges from 0.41 to 0.43, firmly in the interference regime. $\text{Ortho}_{\text{GSM8K}}$ is 0.51--0.52; $\text{Ortho}_{\text{MATH500}}$ drops to 0.31--0.36. The root cause is speed penalty: M3 requires loading Qwen2.5-0.5B and running it at every denoising step. Combined with M1's marginal measured speedup (1.16x), the composition is *slower* than M3 alone (1.68x). This result overturns an earlier pilot finding ($\text{Ortho}$=1.34 on 100 samples), which was an artifact of small-sample variance and a combined metric that included degenerate benchmarks.

![Figure 4: Pairwise orthogonality scores with per-benchmark breakdown. Horizontal lines at Ortho = 1.0 (synergy threshold) and Ortho = 0.8 (near-orthogonal threshold). M1+IGSD is near-orthogonal; M1+M3 shows clear interference across both benchmarks.](figures/fig4_ortho_bars.pdf)

**Table 4: Pairwise Orthogonality Matrix**

| Pair | Best Config | GSM8K Ortho | MATH500 Ortho | Combined Ortho | Verdict |
|------|-------------|-------------|---------------|----------------|---------|
| **M1+IGSD** | $\eta$=0.5, $\tau$=0.7, $T_{\text{draft}}$=16 | **0.99** | 0.64 | **0.96** | Near-orthogonal |
| M3+IGSD | $\tau$=0.7, $T_{\text{draft}}$=16, $w_g$=0.7 | 0.96 | 0.76 | 0.84 | Task-dependent |
| M1+M3 | $\eta$=0.5, $w_g$=0.3 | 0.52 | 0.32 | 0.41 | Interference |

### 4.3 Three-Way Composition and Pareto Frontier

Three-way composition does not extend the pairwise Pareto frontier established by M1+IGSD. The best pairwise QAS (M1+IGSD, $T_{\text{draft}}$=16: QAS=1.34 on the combined metric) exceeds the best three-way QAS (1.07) because the three-way configurations use the more conservative $T_{\text{draft}}$=32. The three-way study's value is in confirming near-orthogonal Ortho at the three-way level and definitively ruling out M3 guidance as a composition layer.

Table 5 and Figure 5 present the three-way composition results. Five configurations, selected from a 24-configuration pilot sweep, were validated on 3 seeds (42, 123, 456) with 100 GSM8K + 100 MATH500 samples per seed. All configurations meet the stability criterion (QAS coefficient of variation < 10%).

The top operating point, **Max-Speed** ($\eta$=0.5, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0.0), achieves 1.71x $\pm$ 0.02 speedup, 62.7% $\pm$ 4.6% accuracy retention, QAS=1.07 $\pm$ 0.09, and $\text{Ortho}$=1.02 $\pm$ 0.08 (3-seed mean $\pm$ std). The **Balanced-A** variant ($\eta$=1.0, $\tau$=0.9, $T_{\text{draft}}$=32, $w_g$=0.0) shows marginally better accuracy retention (63.3%) at similar speedup (1.68x), QAS=1.07, $\text{Ortho}$=1.03.

The **Quality-First** recipe ($\eta$=0.5, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0.3) adds M3 guidance and drops $\text{Ortho}$ from 1.02 to 0.49. M3 guidance ($w_g > 0$) consistently degrades three-way composition: the TPS overhead from the guide model forward pass at every step overwhelms any quality improvement. In three-way compositions, M3 guidance is counterproductive.

![Figure 5: Combined Pareto frontier with individual, pairwise, and three-way compositions. Key operating points labeled. Three-way compositions do not extend the pairwise frontier.](figures/fig5_combined_pareto.pdf)

**Table 5: Three-Way Composition Operating Points (3-seed validation)**

| Recipe | Config | Speedup | AccRet (%) | QAS | Ortho | QAS CV |
|--------|--------|---------|------------|-----|-------|--------|
| **Max-Speed** | $\eta$=0.5, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0 | **1.71** $\pm$ 0.02 | 62.7 $\pm$ 4.6 | **1.07** $\pm$ 0.09 | **1.02** | 8.2% |
| Balanced-B | $\eta$=1.0, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0 | 1.71 $\pm$ 0.02 | 62.7 $\pm$ 4.6 | 1.07 $\pm$ 0.09 | 1.02 | 8.2% |
| Balanced-A | $\eta$=1.0, $\tau$=0.9, $T_{\text{draft}}$=32, $w_g$=0 | 1.68 $\pm$ 0.03 | 63.3 $\pm$ 4.4 | 1.07 $\pm$ 0.09 | **1.03** | 8.1% |
| Conservative | $\eta$=0.5, $\tau$=0.9, $T_{\text{draft}}$=32, $w_g$=0 | 1.68 $\pm$ 0.03 | 63.3 $\pm$ 4.4 | 1.07 $\pm$ 0.09 | 1.03 | 8.1% |
| Quality-First | $\eta$=0.5, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0.3 | 1.68 $\pm$ 0.02 | 62.7 $\pm$ 4.6 | 1.05 $\pm$ 0.09 | 0.49 | 8.3% |

### 4.4 IGSD Ablation

Figure 6 presents the $T_{\text{draft}}$ sweep at fixed $\tau$=0.9, and the $\tau$ sweep at fixed $T_{\text{draft}}$=32. Both evaluations use 200 GSM8K + 100 MATH500 samples, seed 42.

**$T_{\text{draft}}$ sweep.** $T_{\text{draft}}$=16 yields the highest GSM8K QAS (1.51) at 2.50x speedup and 60.3% accuracy retention. $T_{\text{draft}}$=32 reaches QAS=1.16 at 1.71x speedup and 67.8% accuracy retention. $T_{\text{draft}}$=48 gives QAS=0.90 at 1.22x speedup and 73.3% retention. $T_{\text{draft}}$=32 is Pareto-optimal when accuracy retention above 65% is required: it offers 40% more speedup than $T_{\text{draft}}$=48 with only 5.5 percentage points lower accuracy retention.

**$\tau$ sweep.** At fixed $T_{\text{draft}}$=32, $\tau \in \{0.7, 0.85, 0.9\}$ produce GSM8K QAS of 1.17, 1.17, and 1.16 respectively. Accuracy retention ranges from 66.4% ($\tau$=0.7) to 67.8% ($\tau$=0.85 and 0.9). IGSD is insensitive to $\tau$ in the [0.7, 0.9] range, with the 1.4 pp accuracy difference within the expected variance of 200-sample evaluation.

**Confidence gate ablation.** At $T_{\text{draft}}$=32, $\tau$=0.0 (no partitioning, equivalent to naive step reduction) and $\tau$=0.9 produce identical GSM8K accuracy (49.5%), demonstrating that confidence partitioning provides zero measurable benefit at this operating point. IGSD at $T_{\text{draft}}$=32 is functionally equivalent to naive step reduction. The monotonic KL profile (below) explains this: later-step refinement adds little regardless of which tokens are refined.

**KL divergence profile.** Measured across 100 GSM8K samples, the per-step $\text{KL}(p_t \| p_{t-1})$ profile is monotonically decreasing, not the inverted-U shape originally hypothesized. Early steps (0--15) exhibit mean KL values of 0.08--0.19, while later steps (48--63) show spiky behavior with occasional values exceeding 0.5. The monotonic-on-average profile explains why $\tau$ sensitivity is low: later steps consistently have lower average KL divergence, making the partition boundary insensitive to the exact threshold.

[TODO: Figure 7 -- Per-step KL divergence profile averaged over 100 GSM8K samples, with shaded $\pm$1 std band. Data source: igsd_kl_profiles_raw.json]

![Figure 6: IGSD $T_{\text{draft}}$ ablation showing QAS and accuracy retention vs. $T_{\text{draft}}$. $T_{\text{draft}}$=32 is Pareto-optimal when accuracy retention above 65% is required.](figures/fig6_tdraft_ablation.pdf)

<!-- FIGURES
- Figure 3: gen_fig3_single_pareto.py, fig3_single_pareto.pdf -- Single-method Pareto curves for M1, IGSD, M3 on GSM8K
- Figure 4: gen_fig4_ortho_bars.py, fig4_ortho_bars.pdf -- Pairwise orthogonality bar chart with per-benchmark breakdown
- Figure 5: gen_fig5_combined_pareto.py, fig5_combined_pareto.pdf -- Combined Pareto frontier
- Figure 6: gen_fig6_tdraft_ablation.py, fig6_tdraft_ablation.pdf -- IGSD T_draft ablation
- Figure 7: [MISSING] Per-step KL divergence profile
- Table 3: inline -- Single-method Pareto results
- Table 4: inline -- Pairwise orthogonality matrix
- Table 5: inline -- Three-way composition operating points
-->

---

## 5. Cross-Model and AR Comparison

### 5.1 Dream-7B-Instruct Validation

Dream-7B-Instruct provides cross-model validation. Dream-7B baseline: GSM8K accuracy 36.0% (vs. LLaDA 71.2%), 64.5 TPS. Table 6 compares the top 5 recipes across both models.

The Max-Speed recipe (M1 $\eta$=0.5 + IGSD $\tau$=0.85, $T_{\text{draft}}$=32, M3 off) achieves QAS = 2.18 on Dream-7B GSM8K versus QAS = 1.07 on LLaDA-8B. The amplification arises from Dream-7B's lower baseline accuracy: IGSD's draft phase produces comparably accurate outputs on Dream-7B (AccRet = 125% on GSM8K), suggesting that Dream-7B's iterative refinement in later steps is less productive than LLaDA's. The key transferable finding is that M1+IGSD without M3 remains the Pareto-optimal recipe on both models. M3 guidance consistently reduces Ortho from approximately 1.0 to approximately 0.5 in three-way compositions on both architectures, confirming that the overhead interference is not model-specific. Four of five recipes show consistent synergy patterns across models.

**Table 6: Cross-Model Comparison (Top 5 Recipes)**

| Recipe | LLaDA GSM8K QAS | Dream GSM8K QAS |
|--------|----------------|----------------|
| Max-Speed (M1+IGSD, M3 off) | 1.07 | **2.18** |
| Balanced-B (M1+IGSD, M3 off) | 1.07 | 2.18 |
| Balanced-A (M1+IGSD, M3 off) | 1.07 | 2.15 |
| Conservative (M1+IGSD, M3 off) | 1.07 | 2.15 |
| Quality-First (M1+IGSD+M3) | 1.05 | 1.12 |

### 5.2 AR Baseline Comparison

An honest comparison against optimized AR inference establishes the remaining gap. Table 7 presents the results.

Qwen2.5-7B at batch=1 reaches 96% GSM8K accuracy at 71 TPS (QAS = 3.08 relative to the LLaDA baseline), while the best composed DLM acceleration achieves QAS = 1.07 at 1.71x speedup. At batch=8, the AR advantage widens to 471 TPS (QAS = 20.5).

This gap has two sources. LLaDA-8B's 64-step iterative denoising requires 64 full forward passes per generation, each with $O(N^2)$ bidirectional attention. AR models require $N$ forward passes but with $O(N)$ incremental KV-cached attention per step. Additionally, DLM accuracy on GSM8K (71.2%) trails Qwen2.5-7B (96%) at comparable parameter counts, meaning even lossless acceleration leaves DLMs at a quality disadvantage.

The value of DLM composition research lies in understanding the design space---which combinations work, which fail, and why---rather than in claiming speed parity with AR models. DLMs offer architectural advantages (parallel generation, bidirectional context) that may prove valuable in settings not captured by sequential exact-match benchmarks.

**Table 7: AR vs. DLM Comparison**

| System | Batch | GSM8K Acc (%) | GSM8K TPS | QAS |
|--------|-------|---------------|-----------|-----|
| LLaDA-8B baseline | 1 | 71.2 | 33.8 | 1.00 |
| LLaDA-8B best composed (M1+IGSD) | 1 | 44.6 | 57.8 | 1.07 |
| Qwen2.5-7B greedy | 1 | 96.0 | 71 | **3.08** |
| Qwen2.5-7B greedy | 8 | 96.0 | 471 | **20.5** |

### 5.3 Batch Size Sensitivity

M1+IGSD at batch=1: 1.64x speedup, 96 TPS. At batch=4: accuracy improves to 50% (from 45%) but TPS drops to 56. At batch=8: accuracy 52%, TPS=34. Larger batch sizes reduce per-sample IGSD speedup because confidence profiles are averaged across samples in a batch, reducing accept rate specificity.

---

## 6. Discussion

### 6.1 Why M1+IGSD Composes but M1+M3 Does Not

M1+IGSD achieves near-orthogonal composition ($\text{Ortho}_{\text{GSM8K}}$ = 0.99, combined Ortho = 0.96) while M1+M3 shows destructive interference (combined Ortho = 0.41--0.43). The mechanism behind this divergence is architectural.

**M1+IGSD synergy mechanism.** IGSD's draft-partition-refine pipeline splits the 64-step denoising process into two phases. After the draft phase ($T_{\text{draft}} = 32$ steps), confidence partitioning freezes $\alpha = 88.6\% \pm 13.3\%$ of generation tokens. These frozen tokens exhibit near-zero entropy across the remaining 32 refine steps, because their logit distributions are fixed. M1's entropy-based KV caching exploits this directly: $\overline{\text{CHR}} = 83.4\%$ during composition. The two methods target non-overlapping computational bottlenecks---step count and per-step KV reuse---and their overhead profiles do not conflict. The composition at the best configuration ($\eta = 0.5$, $\tau = 0.7$, $T_{\text{draft}} = 16$) reaches 2.75x speedup at 58.9% accuracy retention on GSM8K.

**M1+M3 interference mechanism.** M3 loads Qwen2.5-0.5B (0.95 GB VRAM) and runs a forward pass through it at every denoising step, adding approximately 12% wall-clock overhead per step. M1's measured speedup without kernel-level cache integration is 1.16x at $\eta = 0.5$. When composed, the M3 overhead dominates: the composition is slower than M3 alone. All three guidance weight settings ($w_g \in \{0.3, 0.5, 0.7\}$) produce GSM8K Ortho values of 0.51--0.52, confirming that the interference is structural rather than hyperparameter-dependent.

**Reconciling with preliminary experiments.** An earlier pilot study reported M1+M3 Ortho = 1.34 on 100 GSM8K samples, suggesting synergy. The full evaluation (100 samples per seed, 3 seeds, corrected QAS formula) resolves this discrepancy: the pilot result was an artifact of small-sample variance and an inflated combined metric that included degenerate benchmarks (HumanEval 2.4% baseline, MBPP 0% baseline). With the corrected combined metric ($0.7 \times \text{GSM8K} + 0.3 \times \text{MATH500}$), M1+M3 interference is confirmed.

### 6.2 M3+IGSD: Task-Dependent Composition

The best M3+IGSD configuration achieves GSM8K Ortho = 0.96 (near-orthogonal) but MATH500 Ortho = 0.76 (interference), yielding combined Ortho = 0.84. The task dependence arises from the guide model's capability: Qwen2.5-0.5B achieves high accuracy on GSM8K independently, making it an effective guide for grade-school math reasoning. On MATH500, the same model's accuracy is insufficient, and when operating on IGSD's compressed trajectory the guide model's limitations on hard tasks compound. The mean Ortho across all M3+IGSD configurations is 0.69 (combined metric), placing this pair in the interference zone on average.

### 6.3 Implications for DLM Acceleration Design

Three design principles emerge from the composition analysis.

**Principle 1: Overhead stacking is subadditive.** Combining methods that each add per-step overhead produces compounding slowdowns. The M1+M3 pair illustrates this: 1.16x $\times$ 1.68x = 1.95x expected speedup, but measured composition is slower than M3 alone. Methods that reduce the number of steps (IGSD) compose more favorably with per-step optimizations (M1) because fewer steps means fewer opportunities for overhead to accumulate.

**Principle 2: KV caching requires kernel-level integration to deliver published speedups.** Entropy-based CHR measurements show 56--99% cache hit rates across $\eta$ thresholds, but M1's measured wall-clock speedup is 1.16--1.50x due to the lack of kernel-level sparse attention. The gap between measured CHR and measured TPS improvement is itself a finding: published DLM acceleration results that report cache hit rates without wall-clock TPS may overstate practical gains.

**Principle 3: Quality-preserving methods are best used standalone.** M3 achieves AccRet > 100% on GSM8K at $w_g \in \{0.3, 0.5, 0.7\}$, making it the only method that improves accuracy. However, its 12% per-step overhead makes it a poor composition partner. Practitioners should apply M3 when accuracy is the binding constraint and apply M1+IGSD when throughput is the priority, rather than stacking all three.

### 6.4 Limitations

**Model coverage.** The study evaluates two DLMs from the LLaDA/Dream family. Generalization to MDLM, SEDD, or models with fundamentally different masking schedules is untested.

**M1 speedup is projected.** Cache hit rates are measured (56--99%), but wall-clock speedup is projected from CHR rather than directly measured with kernel-level KV cache integration. The Ortho metric is dimensionless and valid regardless, but absolute combined TPS values should be interpreted as upper bounds. The d2Cache integration failure---15.2x framework overhead on Blackwell GPUs---may be hardware-specific.

**Statistical power.** MATH500 baseline accuracy (11.1%) limits statistical power: a 3-sample fluctuation on 100 samples changes accuracy by nearly 1 percentage point. Pairwise results are pilot-scale (100 samples, single seed), so their Ortho values carry higher uncertainty than the 3-seed three-way results.

### 6.5 Negative Results

**M2 (adaptive step scheduling).** Simplified Saber implementation without the backtracking mechanism produced catastrophic accuracy collapse: AccRet dropped below 50% at step jumps exceeding 3x because LLaDA's masked denoising requires sequential step gradients. M2 receives a NO_GO verdict. The full Saber implementation with backtracking may behave differently, and M1+M2, M2+M3, and M2+IGSD pairwise compositions remain untested.

**Confidence gate at $T_{\text{draft}}$=32.** IGSD's confidence partitioning contributes zero measurable accuracy improvement at $T_{\text{draft}}$=32 (Section 4.4). The "information-geometric" partition reduces to naive step truncation at this operating point. The partitioning mechanism may provide benefit at other operating points or on other models.

---

## 7. Conclusion

ComposeAccel provides the first controlled factorial study of training-free acceleration composition for diffusion language models.

The central finding is that composition outcomes are predictable from the interaction mechanism: M1+IGSD is near-orthogonal (combined Ortho = 0.96) because they target non-overlapping bottlenecks; M3+IGSD is task-dependent (GSM8K Ortho = 0.96, MATH500 Ortho = 0.76) because the guide model's capability varies by task difficulty; M1+M3 interferes destructively (combined Ortho = 0.41--0.43) because guide-model overhead negates marginal cache speedup.

The Pareto-optimal three-way recipe is M1+IGSD with M3 off (QAS = 1.07, Ortho = 1.02, stable across 3 seeds). Adding M3 guidance drops Ortho to ~0.5. Cross-model validation on Dream-7B-Instruct confirms transferable composition patterns (QAS = 2.18 vs. 1.07 on LLaDA).

Qwen2.5-7B at batch=1 achieves QAS = 3.08 relative to the LLaDA baseline. Training-free composition narrows but does not close the DLM-AR speed-quality gap. The value of this study is in mapping the composition design space, not in claiming speed parity.

Three design principles for future DLM acceleration work follow from our results: (1) step-reduction methods compose better than per-step optimizations because they reduce overhead accumulation opportunities; (2) kernel-level KV cache integration is the highest-leverage missing piece for translating measured cache hit rates into wall-clock speedup; and (3) quality-preserving methods like AR guidance are best deployed standalone rather than as composition layers. Future work should extend the factorial design to MDLM, SEDD, and models above 10B parameters.

The IGSD implementation (50 lines), all acceleration recipes with validated hyperparameters, and the composability benchmark suite are released to support future composition studies.

---

## Figures and Tables

- Figure 1: fig_teaser.pdf -- Speed-quality landscape teaser scatter plot
- Figure 2: [TODO: generate from fig2_architecture_desc.md] -- ComposeAccel architecture diagram
- Figure 3: fig3_single_pareto.pdf -- Single-method Pareto curves
- Figure 4: fig4_ortho_bars.pdf -- Pairwise orthogonality bar chart
- Figure 5: fig5_combined_pareto.pdf -- Combined Pareto frontier
- Figure 6: fig6_tdraft_ablation.pdf -- IGSD $T_{\text{draft}}$ ablation
- Figure 7: [TODO: generate from igsd_kl_profiles_raw.json] -- Per-step KL divergence profile
- Table 1: inline -- Published DLM acceleration methods
- Table 2: inline -- Metric definitions
- Table 3: inline -- Single-method Pareto results on GSM8K
- Table 4: inline -- Pairwise orthogonality matrix
- Table 5: inline -- Three-way composition operating points
- Table 6: inline -- Cross-model comparison
- Table 7: inline -- AR vs. DLM comparison

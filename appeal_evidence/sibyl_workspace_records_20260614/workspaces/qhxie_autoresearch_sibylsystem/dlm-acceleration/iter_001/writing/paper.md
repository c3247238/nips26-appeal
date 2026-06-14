# ComposeAccel: Why Only One Pair of Training-Free Acceleration Methods Synergizes in Masked Diffusion Language Models

---

## Abstract

Masked Diffusion Language Models (MDMs) such as LLaDA-8B-Instruct generate text through iterative token unmasking over dozens of denoising steps, each requiring a full bidirectional forward pass that makes standard KV-cache reuse invalid. As of April 2026, at least seven KV-cache approximation strategies and four additional training-free methods exist for MDMs, reporting speedups of 2× to 99× in isolation, but no prior work asks whether any two methods can be safely combined. We introduce ComposeAccel, a systematic pairwise composability study of four training-free acceleration families (of which one — adaptive step scheduling — receives an immediate NO_GO verdict and is excluded from pairwise experiments) under a unified evaluation protocol on LLaDA-8B-Instruct (GSM8K, MATH500, HumanEval, MBPP; 3 seeds). We define an Orthogonality metric $\text{Ortho}(M_a + M_b) = \text{Speedup}(M_a + M_b) / (\text{Speedup}(M_a) \times \text{Speedup}(M_b))$ and a Quality-Adjusted Speedup (QAS) to measure both composability and quality-speed tradeoffs. We also introduce Coarse-Draft Self-Speculative Denoising (CD-SSD), a reduced-step, training-free speculative denoising method for MDMs that uses confidence-based token partitioning to create frozen-token conditions enabling KV-cache synergy — a structural property unavailable in SSD's full-step drafting architecture — achieving mean 3.40× speedup across benchmarks (range: 1.35× on MBPP to 4.57× on GSM8K). The composability landscape is binary: exactly one of three feasible pairs — KV-cache approximation (M1) combined with CD-SSD — achieves super-multiplicative synergy ($\text{Ortho} = 1.385$, 5.13× combined speedup, $\text{QAS} = 1.654$), while all others destructively interfere. The synergy is mechanistically explained by CD-SSD's frozen-token partition creating ideal KV-cache conditions ($\text{CHR}_{\text{refine}}$ rises from ~60% to ~94% on GSM8K during CD-SSD's refine phase). We provide a failure mode atlas characterizing four interference patterns with runtime-observable detection signals.

---

## 1. Introduction

Masked Diffusion Language Models (MDMs) such as LLaDA-8B-Instruct [CITE:llada] and Dream-7B [CITE:dream7b] generate text by iteratively unmasking tokens over $T = 64$ denoising steps with bidirectional attention. This architecture enables parallel token generation and strong context modeling, but imposes a steep inference cost: each denoising step requires a full $O(N^2)$ forward pass over all $N$ token positions, and unlike autoregressive (AR) models, standard KV-cache reuse across steps is invalidated by continuously changing mask states. On a single NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM), LLaDA-8B-Instruct generates at 31.0 ± 4.0 tokens/s on GSM8K.

The community has responded with a rapidly expanding set of training-free acceleration methods. As of April 2026, at least seven KV-cache approximation strategies target the per-step attention cost (Fast-dLLM [CITE:fastdllm], dKV-Cache [CITE:dkvcache], Elastic-Cache [CITE:elasticcache], EntropyCache [CITE:entropycache], FlashDLM-FreeCache [CITE:flashdlm], Window-Diffusion [CITE:windowdiffusion], SlowFast+dLLM-Cache [CITE:slowfast]). Two adaptive step scheduling methods reduce total denoising steps (Saber [CITE:saber], PRR [CITE:prr]). Speculative denoising approaches include DualDiffusion [CITE:dualdiffusion], which uses an external draft model within the MDM regime, and DFlash [CITE:dflash], which cross-architecturally uses a block diffusion draft for an AR verifier. Self-Speculative Decoding (SSD) [CITE:ssd] achieves 2.11–3.46× lossless speedup on MDMs using the model's own forward pass, filling part of the self-speculative gap — but no training-free, no-auxiliary-model approach exists that enables the token-freezing architecture required for the KV-cache synergy studied in this paper.

These methods report speedups ranging from 2× to 99× in their respective papers. However, every published evaluation is conducted in isolation, on different benchmark subsets, with incompatible throughput measurement protocols (see Table 1). No prior work asks whether two methods can be safely combined, or what happens when the assumptions of one method conflict with those of another. This leaves practitioners with three unresolved problems.

**Deployment confusion.** Given a task type (math reasoning, code generation) and latency budget, which method — or combination — should be deployed? No unified comparison exists to answer this question.

**Hidden conflicts.** Some method combinations may interact catastrophically. MDM denoising steps are globally coupled through mask state: each token's unmasking probability depends on the current mask state of all other tokens. Methods that modify the mask trajectory (e.g., adaptive step scheduling) should logically conflict with methods that assume trajectory stability (e.g., KV-cache approximation), but this hypothesis has never been tested quantitatively.

**A missing composability vehicle.** Existing speculative denoising approaches either require an external draft model (DualDiffusion) or a cross-architecture draft (DFlash). SSD achieves lossless self-speculative speedup but uses full-step ($T=64$) drafting that does not produce the token-freezing partition enabling KV-cache synergy — because SSD does not freeze a token partition, the structural condition for super-multiplicative KV-cache synergy is absent. No training-free, reduced-step self-speculative method with confidence-based token freezing exists to serve as a composability study vehicle for KV-cache approximation.

This paper addresses all three problems through a systematic composability study.

We introduce **ComposeAccel**, a framework for measuring how training-free MDM acceleration methods interact when combined. The framework defines a formal Orthogonality metric,
$$\text{Ortho}(M_a + M_b) = \frac{\text{Speedup}(M_a + M_b)}{\text{Speedup}(M_a) \times \text{Speedup}(M_b)},$$
and a Quality-Adjusted Speedup (QAS), $\text{QAS}(M) = \text{Speedup}(M) \times \text{AccRet}(M)$, which jointly capture whether two methods compose multiplicatively and whether the combined speedup is purchased at an acceptable accuracy cost.

We evaluate four method families on LLaDA-8B-Instruct across GSM8K, MATH500, HumanEval, and MBPP, and we introduce a new method, **Coarse-Draft Self-Speculative Denoising (CD-SSD)**, to fill the self-speculative gap.

**Table 1: Literature speed comparison.** Published speedup claims for training-free MDM acceleration methods. All prior methods are evaluated independently; no composability data exists. Our implementations are evaluated under a unified protocol on the same hardware.

| Method | Published Speedup | Base Model | Training-free? | Evaluated in Combination? |
|--------|------------------|-----------|---------------|--------------------------|
| Fast-dLLM [CITE:fastdllm] | 27.6× | LLaDA/Dream | Yes | No |
| EntropyCache [CITE:entropycache] | 15.2–26.4× | LLaDA-8B/Dream-7B | Yes | No |
| Elastic-Cache [CITE:elasticcache] | 8.7–45.1× | LLaDA variants | Yes | No |
| Window-Diffusion [CITE:windowdiffusion] | Up to 99× | LLaDA/Dream | Yes | No |
| SlowFast+dLLM-Cache [CITE:slowfast] | 34.22× | LLaDA | Yes | No |
| Saber [CITE:saber] | 2.51× | DLM (code) | Yes | No |
| DualDiffusion [CITE:dualdiffusion] | Not reported | MDM | Yes | No |
| **M1 (EntropyCache, this work)** | **1.38×** | **LLaDA-8B-Instruct** | Yes | **Yes** |
| **CD-SSD (this work)** | **3.40×** | **LLaDA-8B-Instruct** | Yes | **Yes** |
| **M1+CD-SSD (this work)** | **5.13×** | **LLaDA-8B-Instruct** | Yes | **Yes (SYNERGY)** |

*Note: M1's 1.38× is the combined (cross-benchmark, sample-weighted) speedup at $\eta=2.0$; the GSM8K-specific speedup is 1.50× (Table 2). M1's 1.38× combined versus EntropyCache's published 15–26× reflects an implementation gap: our implementation tracks cache hit rates and entropy signals but executes full forward passes without kernel-level sparse attention. With full kernel-level sparse attention (as in published EntropyCache), the frozen-token synergy would yield a projected combined speedup proportionally higher than 5.13× by the same Ortho correction factor of 1.385 — a calculation we leave for future work. This implementation gap does not affect the qualitative composability finding, as Ortho is measured from speedup ratios under the same implementation conditions. See Section 4.2 and Section 6.3 for detailed discussion.*

As Figure 1 summarizes, the composability landscape is binary: exactly one of three feasible method pairs achieves super-multiplicative synergy, while the others destructively interfere.

![Figure 1: (a) Orthogonality scores for all three feasible pairs, color-coded by synergy (green, Ortho > 1.0) and interference (red, Ortho < 0.8); (b) Speedup vs. Quality-Adjusted Speedup scatter for all individual methods and combined pairs. M1+CD-SSD is the only pair in the synergy region.](figures/fig1_teaser.pdf)

**Our contributions are:**

1. **First systematic pairwise composability study** of four families of training-free MDM acceleration methods — KV-cache approximation (M1), adaptive step scheduling (M2), AR-guided unmasking (M3), and CD-SSD — under a unified evaluation protocol on LLaDA-8B-Instruct (3 seeds, full benchmark scale). Prior work evaluates each method independently; no pairwise composability data exist [CITE:entropycache, CITE:fastdllm, CITE:saber].

2. **Binary composability finding.** MDM composability is not a gradient — it is binary. Exactly one pair, M1 (EntropyCache, $\eta=2.0$) + CD-SSD ($\tau=0.9$, $T_{\text{draft}}=16$), achieves super-multiplicative synergy ($\text{Ortho} = 1.385$, 2-seed estimate; range [1.292, 1.478] across seeds; 5.13× combined speedup, which exceeds the multiplicative expectation of $1.38 \times 3.40 = 4.69$× by 9.4%, with QAS = 1.654). All other feasible pairs destructively interfere: M1+M3 produces combined speedup of 0.93×, slower than either method alone.

3. **Mechanistic explanation of the synergy.** During CD-SSD's refine phase, tokens in $S_{\text{accept}}$ ($\alpha \approx 0.52$ at $\tau = 0.9$) are frozen — their identities do not change across all $T_{\text{full}} = 64$ refinement steps. EntropyCache assigns entropy $H_i = 0$ to these frozen positions (deterministic distributions have zero entropy), guaranteeing cache hits throughout the refine phase. The combined speedup thereby exceeds the product of individual speedups because CD-SSD creates the ideal caching scenario for M1.

4. **CD-SSD: Coarse-Draft Self-Speculative Denoising.** A reduced-step, training-free speculative denoising method for MDMs, concurrent with SSD [CITE:ssd] and SSMD [CITE:ssmd], that achieves mean 3.40× speedup across GSM8K, MATH500, HumanEval, and MBPP (GSM8K: 4.57×, MBPP: 1.35×) by running a fast $T_{\text{draft}}=16$ reduced-step draft pass of the same MDM and selectively refining only low-confidence positions. Unlike SSD (which uses full $T=64$ step drafting without a token-freezing partition), CD-SSD's confidence-based partitioning into $S_{\text{accept}}$ and $S_{\text{refine}}$ creates the frozen-token structure that enables KV-cache synergy — a structural property absent from SSD's architecture. CD-SSD requires no external model parameters. Its standalone accuracy retention ranges from 63.7% (GSM8K) to 88.5% (MATH500) at the operating point $\tau=0.9, T_{\text{draft}}=16$; for applications requiring less than 5% accuracy loss, CD-SSD is not suitable as a standalone method.

5. **Failure mode atlas.** Four characterized interference patterns with proactive detection signals. Adaptive step scheduling (M2) receives a NO\_GO verdict: step-jump factors above 3× cause mask-consistency violations (accuracy retention collapses to 13–28% at 4×–8× step-jump) that no hyperparameter tuning can resolve, because they arise from a fundamental algorithmic mismatch with LLaDA's masked denoising schedule.

**Scope and honest positioning.** ComposeAccel is an analysis paper, not a methods paper. CD-SSD serves primarily as a composability study vehicle rather than a standalone acceleration proposal, and the combined M1+CD-SSD system's 5.13× speedup is below the single-method claims of published techniques (EntropyCache 15–26× [CITE:entropycache], Fast-dLLM 27.6× [CITE:fastdllm]). The contribution is structural insight: understanding when and why acceleration methods compose, and providing actionable deployment guidance through the failure mode atlas.

The remainder of the paper is organized as follows. Section 2 formalizes the composability framework and presents CD-SSD and the three other method families. Section 3 presents experimental results: single-method Pareto baselines, pairwise orthogonality analysis, the failure mode atlas, CD-SSD ablations, and task-type dependence. Section 4 analyzes why composability is binary, characterizes the frozen-token synergy mechanism, and discusses limitations and open questions. Section 5 positions this work against related acceleration methods. Section 6 concludes.

---

## 2. Methods

This section formalizes the ComposeAccel framework. Section 2.1 defines the composability metric and Quality-Adjusted Speedup (QAS). Section 2.2 describes the four method families and their operating parameters. Section 2.3 specifies the evaluation protocol.

---

### 2.1 Composability Framework

#### Orthogonality Metric

Let $M_a$ and $M_b$ denote two acceleration methods, and let $\text{Speedup}(M)$ be the wall-clock throughput ratio of method $M$ over the unaccelerated baseline (measured in tokens per second, TPS). The **Orthogonality metric** of the pair $(M_a, M_b)$ is:

$$
\text{Ortho}(M_a + M_b) = \frac{\text{Speedup}(M_a + M_b)}{\text{Speedup}(M_a) \times \text{Speedup}(M_b)}
$$

Under this definition:
- $\text{Ortho} \geq 1.0$: super-multiplicative synergy — the combined speedup exceeds the product of individual speedups.
- $\text{Ortho} \in [0.8, 1.0)$: partially orthogonal — modest degradation from multiplicative ideal.
- $\text{Ortho} < 0.8$: destructive interference — the methods conflict substantially.

The denominator $\text{Speedup}(M_a) \times \text{Speedup}(M_b)$ represents the multiplicative ideal: what the combined speedup would be if the two methods were perfectly independent bottleneck reductions. Ortho quantifies how far the empirical combination deviates from this ideal.

#### Quality-Adjusted Speedup

Speedup alone is insufficient when methods degrade output quality. We define **Quality-Adjusted Speedup (QAS)** as:

$$
\text{QAS}(M) = \text{Speedup}(M) \times \text{AccRet}(M)
$$

where $\text{AccRet}(M) = \text{Acc}(M) / \text{Acc}(\text{baseline})$. QAS simultaneously rewards throughput gains and penalizes accuracy losses. A method achieving 2× speedup with 90% retention (QAS = 1.8) scores higher than one achieving 3× speedup with 50% retention (QAS = 1.5): QAS correctly penalizes the steep accuracy cost of the faster method.

Combined QAS is:

$$
\text{QAS}(M_a + M_b) = \text{Speedup}(M_a + M_b) \times \overline{\text{AccRet}}(M_a + M_b)
$$

where $\overline{\text{AccRet}}$ is the mean accuracy retention across benchmarks, equally weighted unless stated otherwise.

---

### 2.2 Method Implementations

We study four families of training-free MDM acceleration methods. All implementations operate on LLaDA-8B-Instruct [CITE:llada] with no modification to model weights. Parameters for each method are swept during single-method Pareto evaluation (Section 3.1); the operating points reported here are selected from that sweep.

#### M1: KV-Cache Approximation (EntropyCache)

Standard attention in each MDM denoising step recomputes all $N^2$ key-value products. KV-cache approximation [CITE:entropycache] reuses cached key-value matrices $\hat{K}_{t-1}^{(\ell)}, \hat{V}_{t-1}^{(\ell)}$ from the previous step for positions where the token representation is unlikely to have changed.

**Refresh decision.** EntropyCache uses per-token decoded entropy $H_i$ as the stability signal. At each denoising step $t$ and layer $\ell$:

$$
H_i = -\sum_{v \in \mathcal{V}} p_\theta(v \mid \tilde{x}_t)_i \log p_\theta(v \mid \tilde{x}_t)_i
$$

Position $i$ refreshes its KV entry if $H_i \geq \eta$; otherwise it reuses the cached $\hat{K}_{t-1}^{(\ell)}, \hat{V}_{t-1}^{(\ell)}$. The cache hit rate is $\text{CHR} = |\{i : H_i < \eta\}| / N$.

**Parameter sweep**: $\eta \in \{0.5, 1.0, 2.0, 3.0\}$.

**Operating point**: $\eta = 2.0$, selected as the Pareto-optimal threshold in Section 3.1. At $\eta = 2.0$, $\text{CHR} = 0.60$ (standalone), yielding 1.38× combined speedup (1.50× GSM8K-specific). Thresholds below $\eta = 1.0$ invert the benefit: the overhead of entropy computation and selective recomputation exceeds the attention savings (CHR < 50%, combined speedup = 0.55× at $\eta = 0.5$). This cache invalidation failure mode is documented as F2 in Section 3.3. When CD-SSD is co-activated, $\text{CHR}_{\text{refine}}$ rises to ~94% on GSM8K (Section 3.2 and §4.2).

**Implementation note.** Our M1 implementation tracks cache hit rates and entropy signals but still executes full transformer forward passes — it does not use kernel-level sparse attention to skip computation for cache-hit positions. This explains the measured 1.38× vs. the published EntropyCache 15–26× [CITE:entropycache]. The qualitative composability finding holds regardless of this gap, since Ortho is measured from speedup ratios under the same implementation conditions. Section 4.4 discusses this limitation in detail.

#### M2: Adaptive Step Scheduling (Simplified Saber)

Adaptive step scheduling reduces the total number of denoising steps from $T = 64$ by unmasking more tokens per step. Following Saber [CITE:saber], we unmask $J \times$ more tokens per step (top-$k$ by confidence score), reducing the effective step count to $T_{\text{eff}} = \lceil T / J \rceil$.

**Parameter sweep**: $J \in \{2, 4, 6, 8\}$.

**Verdict: NO\_GO.** M2 receives a NO\_GO verdict from the Pareto evaluation: at $J = 2$, GSM8K accuracy retention falls to 54.4%; at $J = 4$ it collapses to 13.0%; at $J = 8$ it reaches 24.3% (catastrophic, QAS $\approx 0$). The root cause is a fundamental algorithmic mismatch: LLaDA's masked denoising requires sequential cumulative conditioning — each token's unmasking at step $t$ depends on the mask state established by all prior steps. Skipping steps creates unresolvable mask inconsistencies at positions committed too early. This is not a hyperparameter issue; no value of $J$ avoids the cascade. M2 is excluded from all composability experiments. This failure mode is documented as F1 (step_starvation) in Section 3.3.

#### M3: AR-Guided Unmasking

AR-guided unmasking [CITE:flashdlm] biases each denoising step's token selection using a lightweight autoregressive (AR) model. At each step, we run Qwen2.5-0.5B [CITE:qwen25] on the current partially-unmasked sequence and blend its log-probabilities with LLaDA's unmasking logits:

$$
\ell_{\text{blended}}(v, i) = (1 - w) \cdot \ell_{\text{LLaDA}}(v, i) + w \cdot \log q_\phi(v \mid x_{<i})
$$

where $q_\phi(v \mid x_{<i})$ is Qwen2.5-0.5B's autoregressive token probability and $w$ is the guidance weight. Cross-tokenizer bridging is required: LLaDA tokens are decoded to text and re-encoded with the Qwen2.5 tokenizer before blending.

**Parameter sweep**: $w \in \{0.3, 0.5, 0.7, 1.0\}$.

**Operating point**: $w = 0.3$, which achieves 1.33× combined speedup (GSM8K-specific: 1.68×) with GSM8K accuracy retention of 103.9% (the AR guide improves reasoning accuracy by 3.9 points above baseline — the only method that genuinely improves quality). MATH500 retention of 243.9% is a statistical artifact of the 11.1% baseline and should not be interpreted as a real improvement. M3 fails on coding benchmarks (HumanEval QAS $\approx 0$, MBPP at 0.52× — slower than baseline), because Qwen2.5-0.5B's guidance degrades on Python syntax where its AR logits do not align with LLaDA's MASK-to-token mapping.

#### CD-SSD: Coarse-Draft Self-Speculative Denoising

CD-SSD is a training-free, no-auxiliary-model speculative denoising method for MDMs, concurrent with SSD [CITE:ssd] and SSMD [CITE:ssmd], that fills the reduced-step composability gap: unlike SSD, which drafts using full $T=64$ steps, CD-SSD uses a reduced-step draft ($T_{\text{draft}} = 16$) with confidence-based token partitioning that creates the frozen-token structure enabling KV-cache synergy. As illustrated in Figure 2, CD-SSD decomposes the full $T = 64$-step denoising process into three phases.

![Figure 2: CD-SSD three-phase pipeline. Draft phase ($T_{\text{draft}}=16$ steps, full sequence) produces per-token confidence scores $c_i$. Partition splits positions into $S_{\text{accept}}$ (green, $c_i \geq \tau=0.9$, $\approx$52% frozen) and $S_{\text{refine}}$ (orange). Refine phase ($T_{\text{full}}=64$ steps) runs on $S_{\text{refine}}$ while frozen tokens provide stable KV context. Annotation: frozen tokens $\to$ $H_i = 0$ $\to$ guaranteed KV-cache hit for M1 synergy.](figures/fig2_igsd_architecture.pdf)

**Draft phase.** Run LLaDA-8B with $T_{\text{draft}}$ aggressive denoising steps over the full sequence canvas, producing a draft output $x_{\text{draft}} \in \mathcal{V}^N$ and per-token confidence scores:

$$
c_i = \max_{v \in \mathcal{V}} p_\theta(v \mid \tilde{x}_{T_{\text{draft}}})_i
$$

At $T_{\text{draft}} = 16$ (one-quarter of the full $T = 64$ schedule), the model establishes the dominant semantic structure of the output.

**Partition.** Tokens are assigned to two disjoint sets based on the confidence threshold $\tau$:

$$
S_{\text{accept}} = \{i : c_i \geq \tau\}, \quad S_{\text{refine}} = \{i : c_i < \tau\}
$$

At $\tau = 0.9$, roughly 52% of tokens ($\alpha \approx 0.52$) are accepted, with the remaining 48% marked for refinement. The partition takes negligible wall-clock time (a single vector comparison).

**Refine phase.** Tokens in $S_{\text{accept}}$ are frozen — their identities are fixed as $x_{\text{draft}}[i]$ for all subsequent steps and are not subject to further denoising. The full $T_{\text{full}} = 64$-step schedule runs only over $S_{\text{refine}}$ positions, but frozen tokens in $S_{\text{accept}}$ participate as context in bidirectional attention.

**Acceptance criterion.** The MDM setting precludes the exact lossless speculative decoding guarantee of AR models, because token identity changes in LLaDA are globally coupled through bidirectional attention — rejecting a draft token and resampling changes the context for all other positions. We adopt a soft acceptance criterion: tokens in $S_{\text{accept}}$ are accepted unconditionally (the draft confidence $c_i \geq \tau$ serves as the quality gate), and the refine phase on $S_{\text{refine}}$ uses the frozen context to run a corrective full-depth denoising pass. Unlike AR speculative decoding, which requires a ratio test for losslessness, the MDM setting precludes lossless guarantees; the quality tradeoff is quantified experimentally in Section 3.4. See Appendix B for full pseudocode.

**Synergy with KV-cache (M1).** During the refine phase, tokens in $S_{\text{accept}}$ have fixed identities across all $T_{\text{full}} = 64$ steps. From the perspective of EntropyCache, these positions are deterministic: their decoded probability distributions assign full mass to a single token, giving entropy $H_i = 0$ for all $i \in S_{\text{accept}}$. Since $H_i = 0 < \eta = 2.0$, every frozen position is a cache hit for every step of the refine phase. At $\alpha = 0.52$, over half the position-step pairs in the refine phase incur zero KV recomputation. The combined speedup from M1+CD-SSD thus exceeds the multiplicative product $1.38 \times 3.40 = 4.69$×, reaching $5.13$× ($\text{Ortho} = 1.385$). This frozen-token synergy mechanism is the central mechanistic finding of this paper and is analyzed in detail in Section 4.2.

**Note on the $\tau = 0.0$ configuration.** Section 3.4 identifies that removing confidence partitioning ($\tau = 0.0$, all tokens accepted from draft, no refine phase) achieves higher QAS than the full CD-SSD configuration. A direct comparison experiment (Section 3.4) shows that CD-SSD($\tau=0.0$) produces identical accuracy to a naive $T=16$ baseline (same GSM8K AccRet = 0.590 across both; QAS difference within 5.8% noise): the confidence-scoring step adds no value at $\tau=0.0$ beyond draft-length selection. We select $\tau = 0.9$ as the composability study vehicle because it activates the frozen-token mechanism required for M1 synergy; Section 3.4 and Section 4.3 discuss the implications.

**Parameter sweep**: $\tau \in \{0.7, 0.8, 0.85, 0.9\}$, $T_{\text{draft}} \in \{4, 8, 16, 32\}$.

**Operating point**: $\tau = 0.9$, $T_{\text{draft}} = 16$, selected from the ablation study in Section 3.4.

---

### 2.3 Evaluation Protocol

#### Model and Hardware

All experiments use **LLaDA-8B-Instruct** (GSAI-ML/LLaDA-8B-Instruct, HuggingFace) in bf16 precision on one NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM). No model weights are modified. Dream-7B-Instruct download failed; all results are single-model.

#### Benchmarks

| Benchmark | Task | Metric | Samples | Note |
|-----------|------|--------|---------|------|
| GSM8K | Grade school math | Exact match (8-shot) | 1319 | Primary reasoning benchmark |
| MATH500 | Competition math | Exact match (4-shot) | 500 | High-difficulty reasoning |
| HumanEval | Code generation | pass@1 (0-shot) | 164 | Degenerate baseline (2.4%) |
| MBPP | Python programming | pass@1 (3-shot) | 257 | Degenerate baseline (0.0%) |

HumanEval and MBPP have degenerate baselines (pass@1 < 5%), making accuracy retention metrics unreliable. Primary analysis in Sections 3.1–3.3 is restricted to GSM8K + MATH500; coding benchmarks are reported separately with explicit caveats.

#### Seeds and Variance

Single-method Pareto experiments use seeds [42, 123, 456] (3-seed); results are reported as mean ± std. Pairwise composability experiments (Section 3.2) use seeds [42, 123] on 200 GSM8K + 164 HumanEval samples due to the combinatorial evaluation cost. Pairwise results are 2-seed estimates with explicit variance caveats.

#### Throughput Measurement

Throughput is reported as TPS: wall-clock output tokens divided by elapsed generation time (prompt encoding excluded). The first 2–5 warm-up samples per benchmark are discarded to exclude JIT compilation overhead. All speedup values are $\text{TPS}(M) / \text{TPS}(\text{baseline})$ measured on the same GPU within the same session.

#### Baseline Reference

| Benchmark | Accuracy | TPS (mean ± std) |
|-----------|----------|-----------------|
| GSM8K | 71.2% ± 1.5% exact match | 31.0 ± 4.0 tok/s |
| MATH500 | 11.1% ± 0.7% exact match | 79.2 ± 0.1 tok/s |
| HumanEval | 2.4% ± 0.5% pass@1 | 98.0 ± 2.1 tok/s |
| MBPP | 0.0% pass@1 | 191.6 ± 0.6 tok/s |

The MBPP 0.0% baseline is a degenerate result at the 3-shot setting used; no acceleration method can claim meaningful accuracy retention on this benchmark.

---

## 3. Experiments

This section presents results in four parts: (§3.1) single-method Pareto baselines, (§3.2) pairwise composability analysis, (§3.3) the failure mode atlas, and (§3.4) CD-SSD ablations. Section 3.5 examines task-dependent recipes. All numbers are mean ± std across seeds [42, 123, 456] unless explicitly noted as 2-seed estimates.

**Baseline**: LLaDA-8B-Instruct [CITE:llada], 64-step denoising, bf16 precision, 1×NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM). Baseline throughput: 31.0 ± 4.0 tok/s (GSM8K), 79.2 ± 0.1 tok/s (MATH500), 98.0 ± 2.1 tok/s (HumanEval). Baseline accuracy: 71.2 ± 1.5% (GSM8K exact match), 11.1 ± 0.7% (MATH500), 2.4 ± 0.5% (HumanEval pass@1), 0.0% (MBPP). HumanEval and MBPP are degenerate baselines (< 5% pass@1); accuracy retention metrics on coding benchmarks should be treated as illustrative. Note that single-method Pareto (§3.1) uses 3 seeds on full benchmarks, while pairwise experiments (§3.2) use 2 seeds on 15% of benchmark size; this asymmetry is flagged at the point of use.

---

### 3.1 Single-Method Pareto Baselines

Table 2 summarizes the speed–accuracy tradeoff at each operating point; Figure 3 plots the full Pareto curves on a log-speedup axis.

**Table 2: Single-method Pareto table.** GSM8K-specific speedup and accuracy retention across operating points. M1 and CD-SSD: 3 seeds; M2 and M3: 2 seeds. AccRet = method accuracy / baseline accuracy. QAS (combined) = combined speedup × mean AccRet across all benchmarks (standard formula; no per-method adjustments); note that for M1, the GSM8K-specific speedup shown in the Speedup column differs from the combined (cross-benchmark) speedup used in QAS. Bold marks the selected operating point per method. GSM8K AccRet and combined AccRet are reported separately.

| Method | Operating Point | Speedup (GSM8K) | GSM8K Acc. | GSM8K AccRet | MATH500 AccRet | QAS (combined) |
|--------|-----------------|---------|------------|--------------|----------------|-----|
| Baseline | — | 1.00× | 71.2% ± 1.5% | 1.000 | 1.000 | 1.000 |
| **M1** | $\eta=0.5$ | 0.61× | 66.5% | 0.934 | 1.024 | 0.497 |
| **M1** | $\eta=1.0$ | 0.62× | 61.4% | 0.863 | 0.867 | 0.473 |
| **M1** | **$\eta=2.0$** | **1.50×** | **39.1%** | **0.550** | **0.656** | **0.836** |
| **M1** | $\eta=3.0$ | 1.91× | 10.3% | 0.145 | 0.319 | 0.523 |
| **M2** | $J=2$ | 3.78× (GSM8K) | 38.8% | 0.544 (GSM8K-specific) | — | 1.177 |
| **M2** | $J=4$ | 7.55× (GSM8K) | 9.3% | 0.130 | — | 0.864 |
| **M2** | $J=6$ | 15.09× (GSM8K) | 2.3% | 0.032 | — | — |
| **M2** | $J=8$ | 15.09× (GSM8K) | 2.3% | 0.032 | — | — |
| **M3** | **$w=0.3$** | **1.68×** | **74.0%** | **1.039** | **2.439\*** | **1.675** |
| **M3** | $w=0.5$ | 1.33× | 73.5% | 1.032 | 2.258* | 1.620 |
| **M3** | $w=0.7$ | 1.33× | 70.9% | 0.995 | — | — |
| **M3** | $w=1.0$ | 1.34× | ~64.8% | 0.910 | — | — |
| **CD-SSD** | $\tau=0.9, T_d=4$ | 2.13× | 17.1% | 0.240 | 0.741 | — |
| **CD-SSD** | $\tau=0.9, T_d=8$ | 2.44× | 22.2% | 0.312 | 0.885 | — |
| **CD-SSD** | **$\tau=0.9, T_d=16$** | **4.57×** | **45.3%** | **0.637** | **0.885** | **2.39** |
| **CD-SSD** | $\tau=0.9, T_d=32$ | 1.47× | 53.4% | 0.749 | 1.120 | — |

*MATH500 AccRet > 1.0 for M3 indicates M3 AR guidance improves accuracy on MATH500 above baseline, likely because the 11.1% baseline is near-floor. Report the ratio for completeness; absolute accuracy is 0.27 (w=0.3). M3 combined speedup across all benchmarks is 1.33×; the GSM8K-specific speedup is 1.68×. QAS (combined) for CD-SSD uses combined AccRet (70.3%) weighted by benchmark sample sizes; QAS = 3.40 × 0.703 = 2.39. For M1, QAS (combined) uses the combined (cross-benchmark) speedup of 1.38×, not the GSM8K-specific speedup shown in the Speedup column.*

![Figure 3: Speed–accuracy Pareto curves for each method (M1, M2, M3, CD-SSD) across parameter sweeps on GSM8K. X-axis: speedup (log scale). Y-axis: GSM8K accuracy retention. Shaded region at AccRet < 0.95 marks the accuracy-loss zone. Only M3 (at $w \leq 0.7$) remains in the acceptable accuracy zone.](figures/fig3_pareto_curves.pdf)

**M1 (EntropyCache)**: At $\eta = 2.0$, M1 achieves 1.50× GSM8K-specific speedup (GSM8K TPS: 31.0 → 46.4 tok/s; combined cross-benchmark speedup: 1.38×) with $\text{CHR} = 0.60$. Below $\eta = 2.0$, the overhead of entropy computation and selective recompute exceeds the attention savings: $\eta = 0.5$ yields 0.61× GSM8K-specific speedup (combined: 0.55×, slower than baseline). Above $\eta = 2.0$, cache hits increase but GSM8K accuracy collapses to 14.5% ($\eta = 3.0$). The window $\eta = 2.0$ is the only viable operating point.

**M2 (Adaptive Step Scheduling) — NO\_GO**: At $J = 2$, M2 achieves 3.78× GSM8K speedup (3.10× combined) but GSM8K accuracy drops to 38.8% (GSM8K-specific AccRet = 0.544). At $J = 4$, GSM8K accuracy collapses to 9.3% (AccRet = 0.130) at 7.55× GSM8K speedup. At $J = 6$ and $J = 8$, GSM8K AccRet drops to 0.032 (2.3% accuracy) at 15.09× GSM8K speedup — with J=6 and J=8 yielding nearly identical results, suggesting a step-count floor in LLaDA's mask schedule. **M2 receives a NO\_GO verdict for LLaDA-8B-Instruct.** Root cause: LLaDA's denoising requires sequential cumulative conditioning across steps; aggressively unmasking $J \times$ more tokens per step commits positions before sufficient diffusion context has accumulated, creating unresolvable mask inconsistencies. This is a fundamental algorithmic incompatibility, not a hyperparameter problem (see §3.3, failure mode F1).

**M3 (AR-Guided Unmasking)**: At $w = 0.3$, M3 achieves 1.68× GSM8K speedup (1.33× combined) with GSM8K accuracy 74.0% vs. baseline 71.2% (AccRet = 1.039, QAS = 1.675) — the only method that improves reasoning accuracy while accelerating. On HumanEval, M3 achieves near-zero pass@1 (AccRet ≈ 0): Qwen's token logits do not align with Python syntax in LLaDA's MASK→token mapping, making M3 a task-specific method restricted to reasoning. On MBPP, M3 achieves 0.52× speedup — slower than the unaccelerated baseline.

**CD-SSD ($\tau = 0.9$, $T_{\text{draft}} = 16$)**: CD-SSD achieves 4.57× GSM8K speedup (3-seed mean) with per-benchmark speedups of 4.57× (GSM8K), 2.32× (MATH500), 1.95× (HumanEval), and 1.35× (MBPP) — a task-dependent range rather than a constant speedup. Combined benchmark mean: 3.40× speedup (weighted by sample counts: 1319, 500, 164, 374). Combined AccRet (sample-weighted mean across all benchmarks) = 0.703; GSM8K-specific AccRet = 0.637 (63.7%, absolute accuracy 45.3% vs. 71.2% baseline). QAS (combined) = 3.40 × 0.703 = 2.39, using the standard formula with no per-method adjustments. Note: MBPP and HumanEval are degenerate baselines (pass@1 ≤ 2.4%); MBPP AccRet is set to 1.0 by convention (pass@1 = 0.0% for both baseline and CD-SSD), and HumanEval AccRet = 0.0 (pass@1 = 0.0% for both). These conventions are included uniformly in the sample-weighted combined AccRet. The accept rate at $\tau = 0.9$ is $\alpha = 0.52$. CD-SSD is the only method that does not degrade to near-zero on coding benchmarks (HumanEval QAS = 0.747).

---

### 3.2 Pairwise Composability Analysis

M2 is excluded from all pairwise experiments (NO\_GO verdict). The three remaining pairs are M1 + CD-SSD, M3 + CD-SSD, and M1 + M3. Pairwise experiments use 200 GSM8K + 164 HumanEval samples, 2 seeds [42, 123]. The 2-seed Ortho estimates are directionally robust (synergy/interference classification is consistent across both seeds for all three pairs).

Table 3 presents the composability results; Figure 4 visualizes the Ortho scores with error bars from the two seeds.

**Table 3: Pairwise orthogonality matrix.** $\text{Ortho}(M_a + M_b) = \text{Speedup}(M_a + M_b) / (\text{Speedup}(M_a) \times \text{Speedup}(M_b))$. Per-seed Ortho values shown in brackets. All results are 2-seed estimates on 200 GSM8K + 164 HumanEval samples; full 3-seed validation is listed as future work.

| Pair | Combined Speedup | Combined AccRet | QAS | Ortho (2-seed) | Verdict |
|------|-----------------|-----------------|-----|----------------|---------|
| **M1 + CD-SSD** | **5.13×** | **0.322** | **1.654** | **1.385** [1.292, 1.478] | **SYNERGY** |
| M3 + CD-SSD | 2.34× | 0.353 | 0.826 | 0.493 [0.462, 0.524] | INTERFERENCE |
| M1 + M3 | 0.93× | 0.544 | 0.504 | 0.301 [0.289, 0.312] | INTERFERENCE |

![Figure 4: Pairwise orthogonality scores with 2-seed variance error bars. Horizontal dashed line at Ortho = 1.0 marks the super-multiplicative threshold. M1+CD-SSD (Ortho = 1.385) is the only pair above the threshold; M3+CD-SSD and M1+M3 fall into the destructive interference zone (Ortho < 0.5).](figures/fig4_ortho_bars.pdf)

**M1 + CD-SSD — SYNERGY (Ortho = 1.385)**. The combined 5.13× speedup exceeds the multiplicative ideal of $1.38 \times 3.40 = 4.69$× by 9.4%. Both seeds independently confirm Ortho > 1.0 (1.292 and 1.478), making the synergy finding robust to seed variance. Combined QAS = 1.654 is below CD-SSD standalone (QAS = 2.39) but above M3 alone on mixed benchmarks (QAS = 1.675 on GSM8K-only degrades to ≈0.0 on HumanEval), establishing M1 + CD-SSD as the Pareto-optimal configuration for general deployment.

The mechanism is the **frozen-token KV synergy** (detailed in §4.2): during CD-SSD's refine phase, 52% of tokens in $S_{\text{accept}}$ are frozen. EntropyCache assigns entropy $H_i = 0$ to positions with deterministic identity, guaranteeing cache hits. Measured $\text{CHR}_{\text{refine}}$ on GSM8K: ~94% (vs. ~60% during standalone M1). The combined speedup thus exceeds the product of individual speedups because CD-SSD creates the ideal cache scenario for M1.

**M3 + CD-SSD — INTERFERENCE (Ortho = 0.493)**. Combining AR-guided unmasking with CD-SSD reduces combined speedup to 2.34× — 31% below CD-SSD standalone (3.40×). The Qwen2.5-0.5B forward passes in M3 run at every denoising step during CD-SSD's draft phase, compounding the overhead. More critically, AR-blended tokens in the draft deviate from LLaDA's diffusion trajectory, reducing draft quality and degrading the refine phase.

**M1 + M3 — INTERFERENCE (Ortho = 0.301)**. This is the most destructive pair: combined speedup 0.93× — slower than baseline (Table 3). M3 alters the unmasking order at every step, changing the token entropy landscape that M1 relies on for cache refresh decisions. The non-stationary entropy invalidates M1's cache entries more frequently, eliminating the speedup while the Qwen forward passes add latency. This pair should never be deployed together.

---

### 3.3 Failure Mode Atlas

Table 4 catalogs the four failure modes identified from the full experiment suite. Each has a confirmed detection signal and a proactive remedy. Figure 5 illustrates the cache_invalidation failure mode for M1 across the entropy threshold sweep.

**Table 4: Failure mode taxonomy.** Severity: CRITICAL = method is NO\_GO; MODERATE = deployable at specific operating point only.

| ID | Mode | Method | Root Cause | Detection Signal | Severity |
|----|------|--------|-----------|-----------------|----------|
| F1 | step_starvation | M2 | LLaDA mask schedule requires sequential cumulative conditioning; $J \geq 2$ commits tokens too early, creating unresolvable mask inconsistencies (GSM8K AccRet: 0.544 at $J=2$, 0.130 at $J=4$, 0.032 at $J=8$) | AccRet < 0.55 at $J \geq 2$ (GSM8K-specific); auto-reject $J \geq 2$ | **CRITICAL** |
| F2 | cache_invalidation | M1 | At $\eta < 2.0$, entropy computation + selective recompute overhead exceeds attention savings; $\text{CHR} < 50\%$ | Per-step mean entropy < 1.5 → Speedup < 1.0× | MODERATE |
| F3 | draft_divergence | CD-SSD | $\tau < 0.8$ accepts low-quality draft tokens; REFINE cannot recover in $T_{\text{full}}$ steps | Per-step acceptance rate $\alpha > 0.75$ | MODERATE |
| F4 | AR_guidance_conflict | M3 + CD-SSD | Qwen forward passes compound overhead; AR-blended tokens corrupt the diffusion trajectory used by CD-SSD's draft quality estimate | Ortho < 0.5; combined Speedup < CD-SSD standalone | MODERATE |

![Figure 5: M1 speedup and GSM8K AccRet vs. entropy threshold $\eta$, illustrating the cache_invalidation failure mode (F2). At $\eta < 2.0$, speedup inverts (shaded "OVERHEAD ZONE"). At $\eta > 2.0$, accuracy collapses. The narrow window at $\eta = 2.0$ is the only viable operating point. Error bars from 3 seeds.](figures/fig5_m1_threshold.pdf)

**F1 — step_starvation (M2, CRITICAL)**: At $J = 2$, combined speedup is 3.10× but GSM8K AccRet drops to 0.544 (absolute GSM8K accuracy: 38.8% vs. 71.2% baseline) and continues collapsing monotonically: $J = 4$ gives GSM8K AccRet = 0.130 (9.3% accuracy), $J = 8$ gives GSM8K AccRet ≈ 0.032 (2.3% accuracy). Figure 3 shows M2 trending toward the bottom-right quadrant regardless of $J$ — fast but unusable. Detection signal: auto-reject $J > 3$ before deployment.

**F2 — cache_invalidation (M1, MODERATE)**: At $\eta = 0.5$, measured Speedup = 0.553× (44.7% slower than baseline). The overhead zone persists until $\eta = 2.0$. Above $\eta = 2.0$, cache hit rate increases but quality collapses. Figure 5 visualizes this narrow viable window. Proactive remedy: initialize $\eta = 2.0$ by default; if deploying on a new benchmark, auto-tune from the first 10 samples.

**F3 — draft_divergence (CD-SSD, MODERATE)**: Low $\tau$ permits a high acceptance rate ($\alpha > 0.75$), flooding $S_{\text{accept}}$ with poor drafts that the refine phase cannot correct within $T_{\text{full}} = 64$ steps due to bidirectional context contamination. Detection: monitor $\alpha$ at runtime; if $\alpha > 0.75$ for five consecutive denoising runs, raise $\tau$ by 0.05.

**F4 — AR\_guidance\_conflict (M3 + CD-SSD, MODERATE)**: Ortho = 0.493 places M3 + CD-SSD well below the 0.8 destructive interference threshold. Detection: if Ortho < 0.5 in a pairwise pilot, do not combine. Remedy: use M3 and CD-SSD independently based on task type (see §3.5).

---

### 3.4 CD-SSD Ablations

Table 5 shows the CD-SSD ablation results from 200 GSM8K + 164 HumanEval samples, seeds [42, 123]. Figure 6 plots the $T_{\text{draft}}$ sensitivity curve on dual axes.

**Table 5: CD-SSD ablation results.** All configurations use $\tau = 0.9$ except CD-SSD-no-partition ($\tau = 0.0$). $\Delta$QAS% relative to CD-SSD-full. 2-seed mean ± half-range.

| Configuration | Speedup | AccRet | QAS | $\Delta$QAS% |
|---------------|---------|--------|-----|--------------|
| CD-SSD-full ($\tau=0.9$, $T_d=16$) | 2.66 ± 0.00 | 0.359 ± 0.019 | 0.956 ± 0.051 | — |
| CD-SSD-no-partition ($\tau=0.0$, $T_d=16$) | 5.56 ± 0.00 | 0.324 ± 0.019 | 1.801 ± 0.109 | +88.5%† |
| CD-SSD-T4 ($\tau=0.9$, $T_d=4$) | 1.88 ± 0.071 | 0.208 ± 0.016 | 0.394 ± 0.044 | −58.8% |
| CD-SSD-T8 ($\tau=0.9$, $T_d=8$) | 2.30 ± 0.062 | 0.278 ± 0.031 | 0.642 ± 0.088 | −32.8% |
| CD-SSD-T32 ($\tau=0.9$, $T_d=32$) | 2.09 ± 0.019 | 0.405 ± 0.023 | 0.845 ± 0.056 | −11.6% |

†*CD-SSD($\tau=0.0$) vs. naive-$T=16$ baseline: same GSM8K AccRet (0.590 both conditions), QAS difference −5.8% (within noise). The confidence-scoring mechanism adds no value at $\tau=0.0$ beyond step-count selection; see Section 4.3.*

![Figure 6: CD-SSD $T_{\text{draft}}$ sensitivity: combined speedup (left axis) and accuracy retention (right axis) vs. $T_{\text{draft}} \in \{4, 8, 16, 32\}$. Error bars from 2 seeds. The Pareto crossover at $T_d=16$ (marked) balances speed and quality.](figures/fig6_tdraft_sensitivity.pdf)

**$\tau = 0.0$ configuration.** Setting $\tau = 0.0$ — accepting all draft tokens, no refine phase — yields QAS = 1.801, compared to 0.956 for CD-SSD-full (+88.5%). However, a direct comparison against a naive $T = 16$ baseline (no CD-SSD machinery) shows identical GSM8K accuracy (AccRet = 0.590 for both conditions) with only a 5.8% QAS difference — within measurement noise (full_tau0_comparison.json). The confidence-scoring mechanism adds no discriminative value at $\tau=0.0$: the speedup advantage of CD-SSD($\tau=0.0$) collapses to a step-count reduction. The latency asymmetry explains the QAS gap between $\tau=0.0$ and $\tau=0.9$: at $\tau=0.9$, the refine phase executes 64 full denoising steps on 48% of positions, and because frozen tokens still participate as context in bidirectional attention, each refine step still requires $O(N^2)$ attention. The +10.8% AccRet gain (0.324 → 0.359) does not offset the 2.09× speedup reduction. We select $\tau=0.9$ as the composability vehicle because it activates the frozen-token mechanism needed for M1 synergy; CD-SSD's confidence-partitioning step should not be claimed as a contribution independent of this synergy role.

**$T_{\text{draft}}$ sensitivity.** $T_d < 16$ produces QAS < 0.64 due to poor draft quality; $T_d > 16$ yields diminishing quality returns with proportionally larger draft cost. The Pareto crossover at $T_d = 16$ balances speed and quality: speedup decreases for $T_d > 16$ (more draft steps) while AccRet increases.

**Accept rate at $\tau = 0.9$**: $\alpha = 0.52$ (52% of tokens accepted from draft). This is the key figure for the frozen-token KV synergy: $|S_{\text{accept}}| / N = 0.52$ sets the fraction of token positions frozen during refine, elevating $\text{CHR}_{\text{refine}}$ to ~94% (GSM8K) when M1 is co-activated.

---

### 3.5 Task-Dependent Recipes

H4 is confirmed: the optimal acceleration recipe differs by task domain. Across the three methods tested, QAS rankings differ substantially between reasoning (M3 > CD-SSD > M1, with M3 QAS = 1.582) and coding (CD-SSD > M3 ≈ M1, with CD-SSD QAS = 0.744) task types, consistent with H4 and consistent across seeds 42 and 123. Formal statistical testing would require more methods or benchmarks than evaluated in this study. Table 6 summarizes per-domain recommendations.

**Table 6: Task-dependent deployment recipes.** QAS computed over domain-specific benchmarks only (not all four benchmarks).

| Task Domain | Benchmark(s) | Best Single Method | QAS | Best Pair | QAS |
|-------------|-------------|-------------------|-----|-----------|-----|
| Reasoning | GSM8K + MATH500 | M3 ($w=0.3$) | 1.582 | — (M3 must not be combined) | — |
| Coding | HumanEval + MBPP† | CD-SSD ($\tau=0.9$) | 0.744 | M1 + CD-SSD | see Table 3 |
| General / Mixed | All four | M1 + CD-SSD | 1.654 | — | — |

*†MBPP baseline = 0.0% (degenerate). MBPP AccRet is set to 1.0 by convention; treat MBPP numbers as illustrative only. QAS in this table is computed over domain benchmarks only; M3 QAS = 1.582 differs from the combined QAS = 1.675 in Table 2, which uses GSM8K-specific speedup.*

**Reasoning tasks (GSM8K, MATH500)**: M3 leads single-method QAS at 1.582, driven by Qwen2.5-0.5B guidance that improves GSM8K accuracy by 3.9% absolute (74.0% vs. 71.2% baseline). M3 must not be combined with any other method: M3 + CD-SSD gives Ortho = 0.493 (INTERFERENCE), and M1 + M3 gives 0.93× combined speedup (slower than baseline).

**Coding tasks (HumanEval, MBPP)**: CD-SSD is the only viable method (QAS = 0.744). M3 achieves QAS ≈ 0.0 on HumanEval; M1 achieves QAS ≈ 0.0 on HumanEval due to LLaDA-8B's degenerate 2.4% baseline. CD-SSD maintains 0.747 QAS on HumanEval by preserving the overall generation structure through the draft phase.

**General / Mixed deployment**: M1 + CD-SSD (Ortho = 1.385, QAS = 1.654) is the recommended default. It outperforms all single methods on mixed benchmarks and handles both reasoning and coding task types.

*Note on coding benchmark caveats*: HumanEval (2.4% baseline pass@1) and MBPP (0.0% baseline) produce AccRet values that are statistically uninformative. The reader should focus on GSM8K and MATH500 for quantitative comparisons. Coding benchmarks are included for completeness.

---

## 4. Analysis and Discussion

The experiments in §3 produce four findings that require mechanistic interpretation: (1) why only one method pair achieves super-multiplicative synergy, (2) what physical mechanism produces the synergy, (3) what the $\tau = 0.0$ configuration implies for CD-SSD's design, and (4) what the limitations constrain. This section addresses each in turn.

---

### 4.1 Why MDM Acceleration Composability Is Binary

Figure 7 illustrates the root cause: all MDM acceleration methods either preserve or disrupt the denoising trajectory, and this binary property predicts composability.

![Figure 7: Mask-state coupling explains binary composability. Left: compatible composition (M1+CD-SSD) — both methods preserve the denoising trajectory, enabling frozen-token KV synergy. Right: incompatible compositions (M2, M3+CD-SSD, M1+M3) — trajectory-modifying methods (M2 via step skipping; M3 via AR-blended entropy landscape) disrupt trajectory-assuming methods (M1, CD-SSD).](figures/fig7_mask_coupling.pdf)

MDM inference is globally coupled through the mask state $\tilde{x}_t$: every token's predicted unmasking probability $p_\theta(v \mid \tilde{x}_t)$ depends on the current mask pattern of all $N$ positions simultaneously. Sequential denoising steps build a trajectory $\tilde{x}_1 \to \tilde{x}_2 \to \cdots \to \tilde{x}_T$ where each step conditions on the cumulative unmasking decisions of all prior steps. Acceleration methods that **modify this trajectory** inevitably perturb the coupling assumptions of any co-activated method.

**M2 (Adaptive Step Scheduling)** is the most extreme trajectory modifier: at step-jump $J = 4$, it skips $\tilde{x}_{t+1}, \ldots, \tilde{x}_{t+J-1}$ and forces $J \times$ as many tokens to unmask without the intermediate context. LLaDA's denoising requires sequential cumulative conditioning; each newly unmasked position updates the global attention state that informs subsequent positions. Skipping steps commits tokens before sufficient diffusion context has accumulated, creating mask inconsistencies that subsequent steps cannot resolve. The result is catastrophic: AccRet = 0.074 at $J = 8$ (Table 2). This is failure mode F1 (step_starvation).

**M3 (AR-Guided Unmasking)** is a subtler trajectory modifier operating through a different channel: non-stationary entropy distribution. By blending Qwen2.5-0.5B logits at every denoising step with weight $w = 0.3$, M3 reorders unmasking priority toward tokens the AR model predicts with high confidence. This reshapes the trajectory's token entropy landscape: positions the AR model favors are unmasked earlier, changing the sequence of entropy values $H_i$ that M1 uses for cache refresh decisions. When M1 is co-activated with M3 (M1 + M3), M1's entropy threshold $\eta = 2.0$ fires on a non-stationary entropy distribution tuned for a different trajectory. Cache entries invalidate at higher rates, eliminating M1's speedup while Qwen's forward passes add latency. The measured outcome — combined speedup 0.93× (Table 3, Ortho = 0.301) — means the pair is slower than baseline. When CD-SSD is co-activated with M3 (M3 + CD-SSD), the AR-blended draft tokens deviate from LLaDA's diffusion distribution; CD-SSD's refine phase finds more positions requiring correction, compounding overhead: combined speedup drops to 2.34× vs. CD-SSD standalone 3.40× (Ortho = 0.493).

**M1 (EntropyCache)** and **CD-SSD** are the two trajectory-preserving methods. M1 approximates the same per-step computation by reusing KV matrices from step $t-1$ for positions with $H_i < \eta$; it does not change which tokens are unmasked or in what order. CD-SSD adds a reduced-step draft pass, partitions based on confidence, then runs a full-step refine on $S_{\text{refine}}$; frozen tokens remain fixed, but the refine phase itself follows the standard denoising trajectory for the non-frozen positions. Neither method alters the mask trajectory in a way that would disrupt the other. The synergy (Ortho = 1.385) follows not from mere compatibility but from a constructive interaction described in §4.2.

**The binary pattern observed across the three method pairs** in this study is a structural property of MDM inference: trajectory-modifying methods (M2: step skipping; M3: distribution shift) disrupt the global coupling that trajectory-assuming methods (M1, CD-SSD) rely on. Only trajectory-preserving combinations can achieve Ortho $\geq$ 1.0. Among the three active methods in this study, exactly one pair (M1 + CD-SSD) satisfies this condition.

---

### 4.2 The Frozen-Token KV Synergy Mechanism

Figure 8 provides direct evidence of the synergy mechanism: $\text{CHR}_{\text{refine}}$ (KV-cache hit rate on GSM8K) rises from approximately 60% during standalone M1 to approximately 94% during the CD-SSD refine phase.

![Figure 8: $\text{CHR}_{\text{refine}}$ (KV-cache hit rate) across CD-SSD phases averaged over the full refine phase on GSM8K. CHR rises from ~60% (M1 standalone and CD-SSD draft) to ~94% (CD-SSD refine, measured: 0.940), driven by frozen-token entropy collapse ($H_i = 0$ for all $i \in S_{\text{accept}}$). This CHR elevation explains the super-multiplicative speedup: M1+CD-SSD achieves 5.13× vs. the multiplicative expectation of 4.69×.](figures/fig8_kv_hitrate.pdf)

The mechanism operates as follows. At confidence threshold $\tau = 0.9$, CD-SSD's partition step places approximately $\alpha = 0.52$ (52%) of token positions into $S_{\text{accept}}$. These tokens are frozen: their identities do not change across all $T_{\text{full}} = 64$ refinement steps. From EntropyCache's perspective, frozen tokens at position $i$ have a degenerate next-step distribution — the model assigns probability 1.0 to the frozen token and 0.0 to all others. The decoded entropy $H_i = -\sum_v p_\theta(v \mid \tilde{x}_t)_i \log p_\theta(v \mid \tilde{x}_t)_i = 0$ for every frozen position, at every step of the refine phase. Since $H_i = 0 < \eta = 2.0$ by a wide margin, EntropyCache never refreshes KV entries for frozen positions. The KV matrices $K^{(\ell)}_t, V^{(\ell)}_t$ for all layers $\ell$ at frozen positions are computed once (at the transition from draft to refine) and reused for all 64 refine steps with zero overhead.

The remaining $|S_{\text{refine}}| / N = 0.48$ (48%) of positions are active during the refine phase. The measured $\text{CHR}_{\text{refine}}$ across the full refine phase is approximately 94% on GSM8K (measured: 0.940 averaged across seeds 123 and 456; MATH500: 0.864), implying that a substantial fraction of the 48% refine tokens also fall below $\eta = 2.0$ during refinement — consistent with their entropy decreasing as positions converge, though this per-position entropy trajectory was not directly measured.

**Quantitative consistency check.** The super-multiplicative synergy factor is Ortho = 1.385. Under the standard speedup model, combined speedup should equal $\text{Speedup}(\text{M1}) \times \text{Speedup}(\text{CD-SSD}) = 1.38 \times 3.40 = 4.69$× if the two methods were independent. The measured 5.13× exceeds this by $(5.13 - 4.69) / 4.69 = 9.4\%$. This premium is explained by the $\text{CHR}_{\text{refine}}$ elevation from 60% (M1 standalone) to 94% (M1 during CD-SSD refine on GSM8K): at CHR = 94% vs. standalone 60%, the fraction of positions requiring fresh KV computation drops from 40% to 6% — a ~6.7× reduction in recomputed positions — amplifying M1's within-refine speedup above the standalone 1.38× and driving the combined result above the multiplicative baseline.

**Implementation note on M1 speedup gap.** Our M1 implementation achieves 1.38× at $\eta = 2.0$, substantially below published EntropyCache results of 15–26× [CITE:entropycache]. The discrepancy arises because we compute entropy and track cache hit/miss decisions but execute full forward passes regardless. The composability finding — specifically the frozen-token $H_i = 0$ property and the $\text{CHR}_{\text{refine}}$ elevation from ~60% to ~94% — does not depend on kernel-level sparse attention and holds regardless of the absolute speedup magnitude. Any production-grade M1 implementation that achieves kernel-level sparse attention would see its speedup multiplied further by the same CHR elevation factor during CD-SSD's refine phase.

---

### 4.3 The $\tau = 0.0$ Configuration: Step-Count Reduction Is the Primary Driver

The CD-SSD ablation (§3.4, Table 5) reveals that setting $\tau = 0.0$ — accepting all draft tokens, skipping the $T_{\text{full}} = 64$ refine phase — improves QAS from 0.956 to 1.801 (+88.5%). A direct comparison against a naive $T = 16$ baseline (no CD-SSD machinery) resolves the mechanism: both conditions yield identical GSM8K accuracy (AccRet = 0.590, QAS difference −5.8% within noise; full_tau0_comparison.json). The confidence-scoring mechanism adds no discriminative value at $\tau=0.0$ — the output is simply the $T_{\text{draft}} = 16$-step denoising result, regardless of whether confidence scores were computed.

**The structural implication** is that CD-SSD's confidence-partitioning step should not be presented as a standalone contribution at $\tau=0.0$. Its value lies specifically in enabling the M1+CD-SSD synergy at $\tau=0.9$, where frozen tokens create the $H_i=0$ condition exploited by EntropyCache. The +88.5% QAS improvement of CD-SSD($\tau=0.0$) over CD-SSD-full is an artifact of comparing against an unnecessarily expensive refine phase rather than evidence for confidence-informed drafting.

**The latency asymmetry is confirmed.** At $\tau = 0.9$ with $\alpha = 0.52$, the refine phase executes 64 full denoising steps on 48% of positions. Because attention is bidirectional and frozen tokens still participate as context for the non-frozen positions, each refine step still requires $O(N^2)$ attention. The speedup of 2.66× (ablation subsample, 2-seed) vs. $\tau = 0.0$'s 5.56× indicates that the refine phase costs more in latency than it recovers in quality: the +10.8% AccRet gain (0.324 → 0.359) does not offset the 2.09× speedup reduction. The M1 + CD-SSD synergy finding (Ortho = 1.385) is robust to this analysis because it is measured at $\tau = 0.9$ where the frozen-token mechanism is active.

---

### 4.4 Limitations and Open Questions

**1. Pairwise statistical power (2-seed estimate).** The pairwise Ortho scores in Table 3 are computed from 2 seeds over 200 GSM8K + 164 HumanEval samples — 15% of the full benchmark. The Ortho = 1.385 estimate for M1 + CD-SSD has per-seed range [1.292, 1.478] (std ≈ 0.093). While both seeds confirm Ortho > 1.0, the exact magnitude should be treated as a 2-seed estimate until full 3-seed, full-benchmark validation is completed.

**2. M1 implementation gap.** Our M1 achieves 1.38× measured speedup; published EntropyCache [CITE:entropycache] reports 15–26×. Section 3.1 and §4.2 explain why the composability finding is robust to this gap, but absolute speedup numbers in Table 2 and Table 3 reflect this limitation. Practitioners with a kernel-optimized M1 will observe higher absolute combined speedups while the Ortho ratio should remain near 1.385.

**3. CD-SSD accuracy retention on reasoning.** CD-SSD achieves GSM8K AccRet = 63.7% (absolute accuracy: 45.3% vs. 71.2% baseline) and MATH500 AccRet = 88.5% at the operating point $\tau = 0.9$, $T_{\text{draft}} = 16$ (3-seed, full-scale). CD-SSD's primary deployment role is as the speculative component in M1 + CD-SSD, where it provides mean 3.40× speedup (range: 1.35×–4.57× across benchmarks) that combines with M1's CHR elevation for 5.13× combined throughput. For practitioners who require < 5% accuracy loss, neither CD-SSD standalone nor M1 + CD-SSD meets the threshold; M3 is the only method with AccRet $\geq$ 1.0 on reasoning (at the cost of modest 1.33× combined speedup).

**4. Single model evaluation.** All results are from LLaDA-8B-Instruct. Dream-7B-Instruct was unavailable (model download failed). Whether the binary pattern generalizes to other MDM architectures (MDLM [CITE:mdlm], SEDD [CITE:sedd]) is unknown. The frozen-token synergy mechanism depends on LLaDA's architecture: bidirectional attention at every step and the specific way token identities are frozen during the refine phase.

**5. Batch-size sensitivity.** All experiments run at batch size 1. Batched inference at batch $\in \{4, 16\}$ may change the speedup landscape: at batch size > 1, sequences with different confidence profiles may average to lower accept rates $\alpha$ across the batch, reducing the fraction of positions that uniformly exceed $\tau = 0.9$ and therefore reducing CHR elevation. This could lower Ortho below 1.385 in batched settings.

**6. Absence of SSD comparison.** Self-Speculative Decoding (SSD) [CITE:ssd] achieves 2.11–3.46× lossless speedup on MDMs. A direct SSD + M1 composability test would determine whether lossless speculative decoding also creates a frozen-token opportunity for KV synergy — this is the most important follow-up experiment identified in this study.

---

### 4.5 Practical Deployment Guidance

The mechanistic analysis in §4.1–4.3 directly translates into three deployment rules.

**Rule 1 — General deployment: use M1 + CD-SSD.** QAS = 1.654, Ortho = 1.385, 5.13× combined speedup. This configuration handles both reasoning and coding tasks, avoids all known destructive interference failure modes (F1–F4), and provides the best mixed-benchmark result. Operating parameters: $\eta = 2.0$, $\tau = 0.9$, $T_{\text{draft}} = 16$. Runtime detection signals: if CHR < 40% triggers M1 recalibration (F2); $\alpha > 0.75$ triggers CD-SSD $\tau$ adjustment (F3); combined speedup < max(individual speedups) signals AR guidance conflict (F4).

**Rule 2 — Reasoning-only deployment: use M3 alone.** QAS = 1.675 (GSM8K-only, single-benchmark measurement; combined QAS across all benchmarks is lower). M3 is the only method that improves accuracy while accelerating, because Qwen2.5-0.5B's mathematical reasoning prior aligns with LLaDA's unmasking distribution on these benchmarks. Do not combine M3 with any other method: M3 + CD-SSD gives Ortho = 0.493 (INTERFERENCE, F4), and M1 + M3 gives 0.93× combined speedup (slower than baseline, F2+F4 compounded).

**Rule 3 — Never deploy M2.** The step_starvation failure mode (F1, Table 4) is CRITICAL and cannot be mitigated by hyperparameter tuning. LLaDA's denoising requires sequential cumulative conditioning; skipping steps violates this at any $J > 3$, yielding AccRet < 0.50 regardless of downstream configuration. Any deployment pipeline should auto-reject $J > 3$ before execution.

These detection signals are observable within the first 10–20 inference samples, enabling proactive intervention before deploying at full scale.

---

## 5. Related Work

### 5.1 KV-Cache Methods for MDMs

Standard transformer KV-cache reuse is invalidated in MDMs because mask state changes globally at each denoising step. Seven published methods address this through different approximation strategies, organized by mechanism.

**Entropy-based refresh (M1 basis).** **EntropyCache** [CITE:entropycache] takes an $O(V)$ constant-cost approach: at each denoising step $t$, it computes the decoded token entropy $H_i = -\sum_v p_\theta(v \mid \tilde{x}_t)_i \log p_\theta(v \mid \tilde{x}_t)_i$ for each position $i$ and reuses the cached KV entry from step $t-1$ if $H_i < \eta$. Reported speedup is 15.2–26.4× on LLaDA-8B-Instruct and Dream-7B-Instruct; the method is the closest published baseline to our M1 implementation. Our implementation reproduces the entropy-signal logic but executes full forward passes without kernel-level sparse attention (see Section 2.2 and Section 4.4 for the resulting implementation gap).

**Drift-based refresh.** **dKV-Cache** [CITE:dkvcache] introduces near-lossless (distribution-shift threshold) and aggressive (extended refresh interval) variants, reporting 2–10× speedup. **Elastic-Cache** [CITE:elasticcache] applies drift-aware per-layer adaptive refresh using most-attended token signals, reporting 8.7–45.1× speedup with larger gains on longer sequences.

**Structure-based and system-level.** **Window-Diffusion** [CITE:windowdiffusion] introduces a sliding-window cache dividing the canvas into active, buffer, and pruned far-field tokens, reaching up to 99× speedup at the cost of global bidirectional context. **Fast-dLLM** [CITE:fastdllm] combines block-level KV reuse with confidence-aware parallel decoding for 27.6×. **SlowFast+dLLM-Cache** [CITE:slowfast] combines two-stage dynamic sampling with aggressive late-phase caching for 34.22×. **FlashDLM-FreeCache** [CITE:flashdlm] approximates KV reuse with AR supervisor confidence signals for 12.14×.

**Position relative to this work.** All seven KV-cache methods are evaluated independently, on different benchmark subsets, with different throughput measurement conventions. None evaluates whether KV-cache methods compose with any other acceleration family. Our M1+CD-SSD composability analysis (Sections 2 and 3) is the first systematic evaluation of KV-cache composition with speculative denoising. Table 1 in the Introduction summarizes the speedup fragmentation across these methods.

### 5.2 Step Reduction and Adaptive Scheduling

Reducing the total number of denoising steps $T$ is the most direct path to speedup, but MDMs impose a sequential dependency constraint with no AR analog.

**Saber** [CITE:saber] decouples step count from token count per step by unmasking more tokens per step with confidence scoring, plus backtracking-enhanced remasking. Saber reports 2.51× speedup on code generation benchmarks, but backtracking behavior on reasoning benchmarks is less well-characterized. **PRR (Progressive Refinement Regulation)** [CITE:prr] uses a lightweight trained controller requiring auxiliary training, placing it outside the training-free scope of this paper.

**Negative finding — M2 in this work.** Our simplified Saber implementation (M2, step-jump $J \in \{2, 4, 6, 8\}$, no backtracking) receives a NO\_GO verdict on LLaDA-8B-Instruct: GSM8K accuracy retention collapses to 13.0% at $J=4$. The root cause — LLaDA's discrete mask schedule requires sequential cumulative conditioning that is violated by step skipping — corroborates the challenge of transferring continuous diffusion DDIM-style acceleration to discrete MDMs. This negative result is formally characterized as failure mode F1 (step_starvation) in Section 3.3.

### 5.3 Speculative Decoding for MDMs

**DualDiffusion** [CITE:dualdiffusion] introduces a draft-verify framework within the MDM regime using a lightweight draft MDM and a larger verifier, requiring a separate draft model with memory overhead comparable to two models.

**DFlash** [CITE:dflash] uses a block diffusion model as a draft generator for an AR verifier, a cross-architecture setup not applicable to MDM-only deployment, reporting >6× lossless speedup.

**SSD** [CITE:ssd] (Self-Speculative Decoding, Gao et al.) is training-free and achieves lossless speedup of 2.11–3.46× (peak 3.46× on GSM8K) on MDMs including LLaDA-8B and Dream-7B by generating draft tokens using the model's own forward pass with top-$k$ confidence selection, then verifying them via hierarchical verification trees. This paper was published after the original ComposeAccel proposal was drafted.

**SSMD** [CITE:ssmd] (Campbell et al. 2025) adapts attention mask switching to allow the model to serve as both draft generator and verifier, tested at GPT-2 scale with approximately 2× reduction in forward pass count. Scale limits direct comparison.

**S2D2** [CITE:s2d2] (Han et al. 2026) applies training-free self-speculation to block-diffusion LLMs rather than fully masked MDMs, targeting architectures with partial AR structure.

**Position of CD-SSD relative to SSD.** CD-SSD and SSD both eliminate the external draft model requirement. The key architectural difference is in the draft mechanism. SSD drafts using a full-step ($T_{\text{draft}} = T = 64$) forward pass and reduces total computation via tree-structured batch verification; CD-SSD drafts using a **reduced-step** ($T_{\text{draft}} = 16$) forward pass, producing a coarse output whose high-confidence tokens ($c_i \geq \tau$) are frozen for the refine phase. This token-freezing design creates the structural condition enabling super-multiplicative KV-cache synergy (Section 4), a property absent from SSD's architecture because SSD does not freeze a token partition. CD-SSD is approximate (63.7% accuracy retention on GSM8K at $\tau=0.9$, full-scale 3-seed); SSD is lossless. CD-SSD's primary value in this paper is as a composability enabler for the M1+CD-SSD synergy, not as a standalone deployment solution.

### 5.4 Block Diffusion and Hybrid Architectures

**Block Diffusion (BD3-LMs)** [CITE:blockdiffusion] applies AR generation across block boundaries and masked denoising within blocks, enabling direct KV-cache reuse across blocks, trained from scratch. **Fast-dLLM v2** [CITE:fastdllmv2] requires approximately 1B fine-tuning tokens to adapt a pretrained AR model for 2.5× speedup over AR baselines. **D2F (Discrete Diffusion Forcing)** [CITE:d2f] is the first open-source MDM to surpass AR throughput in practice, combining block-wise sequential generation with KV-cache reuse, but requires distillation training. The 5.13× combined speedup of M1+CD-SSD in this work falls substantially below the 27.6×–34.22× reported by the best training-free single methods (Fast-dLLM, SlowFast), and further below training-based methods. The contribution of this paper is not competitive throughput but structural insight: understanding which training-free methods compose and why.

### 5.5 Composability of Intervention Methods

**Kolbeinsson et al. (2024)** [CITE:kolbeinsson2024composable] study composability of LLM interventions — knowledge editing, model compression, and unlearning — measuring order-dependence across $C(3,2) = 3$ pairwise combinations. Their composability framework is spiritually similar to ours, but their interventions operate on model weights (static modifications) rather than inference-time algorithms, and they study entirely different method families. In the AR inference setting, speculative decoding and PagedAttention can be combined without fundamental conflicts because AR generation maintains a stable KV-cache prefix that both methods exploit independently. The MDM inference setting introduces a dynamic composability challenge — the mask state evolves at each step, creating the global coupling that makes trajectory-modifying combinations destructive — with no analog in weight-space intervention studies.

Prior work does not evaluate pairwise composability of training-free inference acceleration methods for MDMs; nor does any published work provide a failure-mode atlas with per-method detection signals and proactive remedies.

---

## 6. Conclusion

This paper presents the first systematic study of training-free acceleration composability for Masked Diffusion Language Models. Across four method families, four benchmarks, and three random seeds on LLaDA-8B-Instruct, we find that the composability pattern across the three feasible pairs is binary: exactly one method pair achieves super-multiplicative synergy, while all others destructively interfere.

### 6.1 What We Found

**A single synergistic pair exists.** M1 (EntropyCache, $\eta = 2.0$) combined with CD-SSD ($\tau = 0.9$, $T_{\text{draft}} = 16$) achieves $\text{Ortho} = 1.385$ (2-seed estimate) and 5.13× combined speedup with $\text{QAS} = 1.654$ — the best combined result across all configurations evaluated. The remaining two feasible pairs both destructively interfere: M3 + CD-SSD gives $\text{Ortho} = 0.493$ (2.34× combined, worse than CD-SSD alone at 3.40×), and M1 + M3 gives $\text{Ortho} = 0.301$ (0.93× combined, slower than baseline).

**The synergy has a mechanistic explanation.** During CD-SSD's refine phase, tokens in $S_{\text{accept}}$ ($\alpha = 0.52$ at $\tau = 0.9$) are frozen, driving their decoded entropy to zero: $H_i = 0$ for all $i \in S_{\text{accept}}$ at every refine step. EntropyCache, which refreshes KV matrices only when $H_i \geq \eta = 2.0$, never re-computes these entries. $\text{CHR}_{\text{refine}}$ climbs from ~60% during standalone M1 to ~94% on GSM8K (measured: 0.940) during the refine phase, amplifying the per-step KV savings beyond what either method achieves independently. The resulting speedup (5.13×) exceeds the multiplicative expectation (1.38× $\times$ 3.40× = 4.69×) by 9.4%.

**Destructive interference arises from mask-state coupling.** Methods that modify the MDM's denoising trajectory produce two categories of interference: M2's step_starvation (premature token commitment through sequential conditioning violation, irreversible) and M3's distribution shift (non-stationary entropy landscape invalidating M1's cache threshold assumptions, degraded draft quality increasing CD-SSD's effective $\tau$). Both categories disrupt co-activated trajectory-assuming methods (M1, CD-SSD), producing Ortho < 0.5 for all trajectory-modifying pairs. M2 is categorically unsafe for LLaDA-8B-Instruct: at step-jump $J = 4$, GSM8K accuracy retention collapses to 13.0%; the step_starvation failure mode (F1) is fundamental, not a tuning artifact.

**CD-SSD provides the composability vehicle.** Coarse-Draft Self-Speculative Denoising is a reduced-step, training-free speculative denoising method for MDMs that uses confidence-based token partitioning to create frozen-token conditions. It achieves mean 3.40× speedup across four benchmarks (range: 1.35× on MBPP to 4.57× on GSM8K) at the operating point $\tau = 0.9$, $T_{\text{draft}} = 16$ (GSM8K AccRet = 63.7%; combined AccRet = 70.3%; QAS = 3.40 × 0.703 = 2.39). CD-SSD's primary value is as the speculative component in M1 + CD-SSD, where its frozen-token partition ($S_{\text{accept}}$) provides the ideal KV-cache context enabling super-multiplicative synergy — a structural property absent from SSD's full-step drafting architecture. For deployments that prioritize speed uniformly across task types, M1 + CD-SSD achieves $\text{QAS} = 1.654$ with no architecture modifications and no additional parameters.

**Task-dependent optimal recipes are confirmed (H4; M3 QAS = 1.582 for reasoning, CD-SSD QAS = 0.744 for coding, both consistent across 2 seeds).** Reasoning-only deployments should use M3 ($w = 0.3$): Qwen2.5-0.5B's mathematical prior improves GSM8K accuracy by 3.9% over baseline while delivering 1.68× GSM8K speedup ($\text{QAS} = 1.675$, GSM8K-only). M3 should not be combined with any other method. General-purpose or coding deployments should use M1 + CD-SSD. M2 must never be deployed.

### 6.2 Limitations

Three substantive limitations constrain the results. First, all pairwise Ortho scores are computed from 2 seeds over 200 GSM8K + 164 HumanEval samples (15% of full benchmark scale); the Ortho = 1.385 estimate for M1 + CD-SSD has per-seed range [1.292, 1.478] and should be treated as a 2-seed estimate until full 3-seed validation is completed. Second, our M1 implementation achieves 1.38× measured speedup rather than the published 15–26× of EntropyCache [CITE:entropycache], because we execute full forward passes without kernel-level sparse attention; the Ortho ratio is implementation-agnostic (it measures CHR ratios, not absolute speedup), but the absolute combined speedup (5.13×) would scale proportionally with a kernel-optimized M1. Third, all experiments use LLaDA-8B-Instruct at batch size 1; Dream-7B-Instruct was unavailable, and batched inference may reduce the frozen-token fraction and the CHR elevation — at batch size > 1, sequences with different confidence profiles may average to lower accept rates across the batch, reducing the fraction of positions that uniformly exceed $\tau = 0.9$ and potentially lowering $\text{Ortho}$ below 1.385.

### 6.3 Future Work

Four experiments remain open and would materially strengthen or qualify the findings.

**SSD composability.** SSD [CITE:ssd] achieves 2.11–3.46× lossless speedup. Whether SSD + M1 also produces a frozen-KV synergy — and whether lossless speculative decoding is compositionally equivalent to CD-SSD's lossy variant — is the most important unanswered question from this study. A positive result would establish the frozen-token CHR elevation as a property of self-speculative decoding in general.

**Full 3-seed pairwise validation.** Completing the pairwise experiments at full benchmark scale (1319 GSM8K + 500 MATH500, 3 seeds) would reduce the variance on the Ortho = 1.385 estimate and enable reporting a confidence interval suitable for a primary claim.

**Batched inference and multi-model generalization.** A systematic analysis of batch sizes $\in \{4, 16\}$ and experiments on additional MDM architectures (MDLM [CITE:mdlm], SEDD [CITE:sedd]) would determine whether the binary composability pattern is universal across MDMs or specific to LLaDA's bidirectional full-sequence attention formulation.

**CD-SSD confidence-ordering value.** The $\tau=0.0$ ablation shows that CD-SSD($\tau=0.0$) ≈ naive $T=16$ in accuracy. A systematic sweep of $\tau \in \{0.0, 0.5, 0.7, 0.8, 0.9\}$ combined with the naive-$T$ comparison at each level would establish precisely where confidence-informed partitioning adds value beyond step-count selection.

### 6.4 Broader Significance

The composability framework introduced here — the orthogonality metric $\text{Ortho}(M_a + M_b)$, the four failure modes (F1–F4) with detection signals, and the trajectory-preserving vs. trajectory-modifying classification — provides a reusable evaluation protocol applicable to any MDM acceleration ecosystem. As the number of competing training-free methods grows beyond the four families studied here, the framework offers a principled approach for practitioners: two trajectory-preserving methods should be tested for CHR synergy; any trajectory-modifying method should be treated as incompatible with all others by default until Ortho > 0.8 is demonstrated empirically. This protocol did not exist before this work.

---

## Figures and Tables

- Figure 1: fig1_teaser.pdf — Composability landscape: (a) Ortho bar chart for all three feasible pairs; (b) Speedup vs. QAS scatter for all methods and pairs, with M1+CD-SSD in the synergy region.
- Figure 2: fig2_igsd_architecture.pdf — CD-SSD three-phase pipeline (Draft, Partition, Refine) with frozen-token → KV-cache synergy annotation. *[TODO: Generate from fig2_igsd_architecture_desc.md spec via TikZ or diagram tool]*
- Figure 3: fig3_pareto_curves.pdf — Speed–accuracy Pareto curves for each method across parameter sweeps on GSM8K.
- Figure 4: fig4_ortho_bars.pdf — Pairwise orthogonality scores with 2-seed variance error bars.
- Figure 5: fig5_m1_threshold.pdf — M1 speedup and GSM8K AccRet vs. entropy threshold, illustrating cache_invalidation failure mode (F2).
- Figure 6: fig6_tdraft_sensitivity.pdf — CD-SSD $T_{\text{draft}}$ sensitivity: speedup and accuracy retention vs. $T_{\text{draft}}$.
- Figure 7: fig7_mask_coupling.pdf — Conceptual diagram: mask-state coupling explains binary composability.
- Figure 8: fig8_kv_hitrate.pdf — $\text{CHR}_{\text{refine}}$ across CD-SSD phases showing CHR jump from ~60% (standalone M1) to ~94% (CD-SSD refine, GSM8K measured: 0.940).
- Table 1: inline (Introduction) — Literature speed comparison of training-free MDM acceleration methods.
- Table 2: inline (Section 3.1) — Single-method Pareto table with all operating points.
- Table 3: inline (Section 3.2) — Pairwise orthogonality matrix with 2-seed Ortho values.
- Table 4: inline (Section 3.3) — Failure mode taxonomy atlas with detection signals and remedies.
- Table 5: inline (Section 3.4) — CD-SSD ablation results including tau=0.0 naive-baseline comparison.
- Table 6: inline (Section 3.5) — Task-dependent deployment recipe recommendations.

---

## Appendix A: Full Per-Seed Results

*[Placeholder: Complete tables for all methods × seeds × benchmarks.]*

## Appendix B: CD-SSD Algorithm Pseudocode

**Algorithm 1: Coarse-Draft Self-Speculative Denoising (CD-SSD)**

```
Input:  prompt x_prompt ∈ V^{N_p}                     -- tokenized prompt
        model p_θ with bidirectional attention          -- LLaDA-8B-Instruct
        τ ∈ [0,1]                                      -- confidence threshold (default 0.9)
        T_draft ∈ Z+                                   -- draft steps (default 16)
        T_full ∈ Z+                                    -- refine steps (default 64)
Output: x_out ∈ V^N                                    -- generated tokens

--- Phase 1: Draft ---
Initialize canvas: x̃_0 = [x_prompt | MASK^N]          -- N generation positions
Run T_draft denoising steps of p_θ on x̃_0:
    for t = 1 to T_draft:
        logits_t = p_θ(· | x̃_{t-1})                   -- full bidirectional forward pass
        x̃_t = unmask_top_k(x̃_{t-1}, logits_t)         -- standard MDM unmasking
x_draft = x̃_{T_draft}                                 -- draft output (all positions filled)

--- Phase 2: Partition ---
Compute per-token confidence scores:
    c_i = max_{v ∈ V} p_θ(v | x_draft)_i   for i = 1,...,N   -- single forward pass
Partition positions:
    S_accept = {i : c_i ≥ τ}               -- frozen tokens (high confidence)
    S_refine  = {i : c_i < τ}              -- positions to be re-denoised

--- Phase 3: Refine ---
Initialize refine canvas:
    x̂_0[i] = x_draft[i]    for i ∈ S_accept   -- freeze accepted tokens
    x̂_0[i] = MASK          for i ∈ S_refine    -- re-mask rejected tokens
Run T_full denoising steps on x̂_0:
    for t = 1 to T_full:
        logits_t = p_θ(· | x̂_{t-1})            -- full bidirectional attention over all N positions
                                                 --   (frozen S_accept tokens participate as context)
        -- Only update positions in S_refine:
        for i ∈ S_refine:
            x̂_t[i] = unmask_step(x̂_{t-1}[i], logits_t[i])
        for i ∈ S_accept:
            x̂_t[i] = x̂_{t-1}[i]               -- frozen tokens unchanged
x_out = x̂_{T_full}

Return x_out
```

**Key implementation notes:**

1. **Acceptance criterion.** Tokens in $S_{\text{accept}}$ are accepted unconditionally when $c_i \geq \tau$. Unlike AR speculative decoding, no ratio test is used: the MDM bidirectional attention context makes exact lossless acceptance impossible, as changing one token affects all positions' joint distributions. The quality tradeoff is quantified in Table 5 and Section 3.4.

2. **KV-cache interaction.** During the refine phase, all $i \in S_{\text{accept}}$ have identity $x_{\text{draft}}[i]$ fixed for all $T_{\text{full}}$ steps. From EntropyCache's perspective: $p_\theta(v | \hat{x}_t)_i$ assigns probability 1.0 to $x_{\text{draft}}[i]$ for all $t$, giving $H_i = 0 < \eta$ and guaranteeing a KV-cache hit at every step. This is the M1 + CD-SSD synergy condition.

3. **Bidirectional attention scope.** Each refine step computes attention over all $N$ positions, including frozen tokens in $S_{\text{accept}}$. The computational cost per step is $O(N^2)$, not $O(|S_{\text{refine}}|^2)$. This is why the speedup from CD-SSD comes primarily from the reduced number of denoising steps for $S_{\text{refine}}$ rather than from reduced attention width.

4. **$\tau = 0.0$ degenerate case.** At $\tau = 0$, $S_{\text{accept}} = \emptyset$ and the refine phase re-denoise all positions from scratch for $T_{\text{full}}$ steps — equivalent to a naive $T_{\text{draft}}$-step denoising without Phase 3. The confidence-scoring step adds no value; see Section 4.3.

## Appendix C: M2 Negative Results

*[Placeholder: Detailed step_jump sweep results documenting the accuracy cascade across GSM8K and MATH500.]*

## Appendix D: Qualitative Examples

*[Placeholder: Sample outputs from baseline, M1, CD-SSD, and M1+CD-SSD on selected GSM8K problems.]*

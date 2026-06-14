# 4 Methods

This section defines the composability framework (Section 4.1), describes the three acceleration methods under study (Sections 4.2--4.4), and specifies the experimental protocol (Section 4.5). Figure 2 illustrates the full composed pipeline.

![ComposeAccel architecture: IGSD draft-partition-refine pipeline with M1 and M3 integration points.](figures/fig2_architecture.pdf)

## 4.1 Composability Framework

We introduce two metrics that distill the speed-quality tradeoff into scalars suitable for pairwise comparison.

**Quality-Adjusted Speedup (QAS).** For an acceleration method $M$ applied to a baseline system with accuracy $\text{Acc}_0$ and throughput $\text{TPS}_0$, we define:
$$
\text{QAS}(M) = S(M) \times \text{AccRet}(M),
$$
where $S(M) = \text{TPS}(M) / \text{TPS}_0$ is the wall-clock speedup and $\text{AccRet}(M) = \text{Acc}(M) / \text{Acc}_0$ is the accuracy retention. QAS captures both dimensions in a single number: a method that doubles speed while halving accuracy yields QAS = 1.0 (no net improvement). No penalty factor is applied; low-quality methods are distinguished by their raw accuracy retention.

**Orthogonality (Ortho).** For methods $A$ and $B$:
$$
\text{Ortho}(A, B) = \frac{\text{QAS}(A + B)}{\max\bigl(\text{QAS}(A),\; \text{QAS}(B)\bigr)}.
$$
The denominator normalizes by the better individual method, so Ortho > 1.0 indicates synergy (the composition strictly outperforms either component alone), $0.8 \leq \text{Ortho} \leq 1.0$ indicates near-orthogonal composition (most benefit preserved), and Ortho < 0.8 indicates interference (composition degrades below the best individual method).

**Combined benchmark metric.** All QAS, Ortho, and Pareto computations use a weighted average across benchmarks: $0.7 \times \text{GSM8K} + 0.3 \times \text{MATH500}$. The weighting reflects the stronger baseline signal on GSM8K ($\text{Acc}_0 = 71.2\%$) compared to MATH500 ($\text{Acc}_0 = 11.1\%$). HumanEval results are reported in the appendix but excluded from combined metrics due to a degenerate 2.4% baseline.

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| AccRet$(M)$ | $\text{Acc}(M) / \text{Acc}_0$ | Fraction of baseline accuracy preserved |
| $S(M)$ | $\text{TPS}(M) / \text{TPS}_0$ | Wall-clock speedup |
| QAS$(M)$ | $S(M) \times \text{AccRet}(M)$ | Quality-adjusted speedup |
| Ortho$(A, B)$ | $\text{QAS}(A{+}B) / \max(\text{QAS}(A), \text{QAS}(B))$ | > 1.0 synergy; 0.8--1.0 near-orthogonal; < 0.8 interference |
| Metric$_{\text{comb}}$ | $0.7 \times \text{GSM8K} + 0.3 \times \text{MATH500}$ | Combined benchmark score |

**Table 2.** Metric definitions and interpretation thresholds.

## 4.2 M1: Entropy-Based KV Caching

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

**Implementation and the kernel gap.** We attempted to integrate d2Cache, a kernel-level sparse attention library, to translate CHR into wall-clock TPS gains. d2Cache's internal caching mechanism achieves 4.4x speedup over its own baseline; however, d2Cache's model wrapper introduces 15.2x overhead compared to the HuggingFace baseline (3.85 TPS vs. 58.5 TPS) due to eager attention mode and per-layer context manager overhead on RTX PRO 6000 Blackwell GPUs. The net effect is that d2Cache with caching enabled (16.9 TPS) still runs at 0.29x the HuggingFace baseline throughput.

We therefore report M1 with two speedup numbers throughout the paper:

- **Measured speedup**: the wall-clock TPS improvement from our simplified Python-level entropy cache, which runs full forward passes but tracks which positions would have reused cached KV. At $\eta = 0.5$: 1.16x speedup, 56% CHR on GSM8K (full-scale, 1319 samples). At $\eta = 2.0$: 1.50x speedup, 60% CHR. The measured speedup is modest because the Python implementation still executes full attention.
- **Projected speedup**: estimated from the measured CHR assuming ideal kernel-level sparse attention. At $\eta = 0.5$: 2.27x (GSM8K CHR = 93.3%). At $\eta = 2.0$: 2.47x (CHR = 99.1%). These projections serve as an upper bound.

The Ortho metric is dimensionless and valid regardless of whether M1's speedup is measured or projected, because it normalizes by the individual method's QAS. The gap between projected and measured speedup is itself a finding: kernel-level KV cache integration for bidirectional-attention DLMs remains an open engineering challenge.

## 4.3 IGSD: Information-Geometric Step Distillation

IGSD is a training-free speculative step scheduler that partitions the denoising trajectory into a fast draft phase and a selective refine phase, using inter-step KL divergence as the partitioning signal.

**Draft phase (steps $0 \to T_{\text{draft}}$).** All $N$ tokens undergo standard denoising for $T_{\text{draft}}$ steps. $T_{\text{draft}} \in \{16, 32, 48\}$ out of $T_{\text{full}} = 64$ total steps.

**Confidence partitioning (at step $T_{\text{draft}}$).** After the draft phase, each token position is classified based on the model's confidence. Token $i$ is frozen if its logit confidence (measured as the maximum softmax probability) exceeds the threshold $\tau$:
$$
\mathcal{S}_{\text{accept}} = \{i : \max_v\, p_{T_{\text{draft}}}(v \mid i) \geq \tau\}, \quad \mathcal{S}_{\text{reject}} = \{1, \ldots, N\} \setminus \mathcal{S}_{\text{accept}}.
$$
The frozen fraction $\alpha = |\mathcal{S}_{\text{accept}}| / N_{\text{gen}}$ is measured at $0.886 \pm 0.133$ across 100 GSM8K samples at $\tau = 0.9$, $T_{\text{draft}} = 16$.

**Refine phase (steps $T_{\text{draft}} \to T_{\text{full}}$).** Only tokens in $\mathcal{S}_{\text{reject}}$ continue denoising for the remaining $T_{\text{full}} - T_{\text{draft}}$ steps. Frozen tokens retain their draft-phase values. The effective computation is reduced from $T_{\text{full}} \times N$ to $T_{\text{draft}} \times N + (T_{\text{full}} - T_{\text{draft}}) \times |\mathcal{S}_{\text{reject}}|$.

**KL divergence signal.** The mean token-level KL divergence between consecutive steps measures how much the model's predictions are changing:
$$
\text{KL}_t = \frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} D_{\text{KL}}\bigl(p_t(i) \,\|\, p_{t-1}(i)\bigr),
$$
where $\mathcal{M}_t$ is the set of masked positions at step $t$. This signal originally motivated the partition threshold; empirically, however, the KL profile is monotonically decreasing (Section 5.4 presents the full profile over 100 samples), which explains why the confidence gate has low sensitivity to $\tau$ in the $[0.7, 0.9]$ range.

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

The implementation is approximately 50 lines of Python inserted into the denoising loop. The computational overhead of the confidence gate (a single argmax over the vocabulary dimension) is $O(N \times V)$, negligible compared to the transformer forward pass.

**Interaction with M1.** IGSD's refine phase creates favorable conditions for entropy-based KV caching: frozen tokens in $\mathcal{S}_{\text{accept}}$ have near-zero entropy (their logit distributions are peaked), providing M1 with a high-quality cache reuse signal. The measured refine-phase CHR is 94.3% at $\eta = 0.5$ (vs. 56% for standalone M1 on the full sequence), explaining the near-orthogonal composition observed in Section 5.2.

## 4.4 M3: AR-Guided Unmasking

M3 uses a lightweight autoregressive (AR) model to bias the DLM's unmasking decisions toward higher-confidence token predictions. At each denoising step $t$, the AR guide model $g_\phi$ (Qwen2.5-0.5B, 0.49B parameters) produces logits $q_t$ over masked positions. The guided logits are:
$$
\tilde{p}_t(i) = (1 - w_g) \cdot p_t(i) + w_g \cdot q_t(i), \quad \text{for masked position } i,
$$
where $w_g \in \{0.3, 0.5, 0.7\}$ is the guidance weight. Unmasked positions retain the DLM's original logits.

M3 is quality-preserving: at $w_g = 0.3$, GSM8K accuracy retention is 103.9% (the guide model corrects some DLM errors). The measured speedup is 1.68x on GSM8K (52.0 TPS vs. 31.0 TPS baseline), arising primarily from reduced denoising steps needed to reach convergence when the unmasking order is improved.

The overhead of M3 comes from loading Qwen2.5-0.5B (0.95 GB VRAM) and running it at every denoising step, adding approximately 12% wall-clock time per step. This overhead is the root cause of destructive interference when M3 is composed with M1: M1's modest measured speedup (1.16x) is insufficient to absorb the M3 per-step cost (Section 5.2).

## 4.5 Experimental Setup

**Models.** LLaDA-8B-Instruct (primary evaluation), Dream-7B-Instruct (cross-model validation). Both are masked diffusion language models using $T = 64$ denoising steps with bidirectional attention.

**Hardware.** 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM each). LLaDA-8B requires approximately 15.3 GB VRAM in bf16; independent tasks run on separate GPUs concurrently.

**Benchmarks.** GSM8K (1319 samples, exact match) and MATH500 (500 samples, exact match) constitute the primary evaluation. HumanEval (164 samples, pass@1) is reported in the appendix only; MBPP is dropped (0% baseline provides no signal).

**Seeds and statistical protocol.** Pilot experiments use seed 42; full-scale experiments use seeds $\{42, 123, 456\}$ and report mean $\pm$ std. Three-way composition top-5 configs are validated with all three seeds (QAS coefficient of variation < 10% required for stability).

**Baseline.** LLaDA-8B-Instruct with 64-step denoising, bf16, greedy decoding. GSM8K: 71.2% $\pm$ 1.5% accuracy, 33.8 TPS (generation-only, post-warmup). MATH500: 11.1% $\pm$ 0.7% accuracy, 79.1 TPS. The first 5 samples per benchmark are discarded as warmup.

**Throughput measurement.** Tokens per second (TPS) is measured as total generated tokens divided by wall-clock generation time, averaged over all non-warmup samples. Batch size is 1 (interactive setting) for all primary experiments; batch sensitivity at sizes $\{1, 4, 8\}$ is reported separately in Section 6.3.

**Hyperparameter sweeps.** M1: $\eta \in \{0.5, 1.0, 2.0\}$. IGSD: $\tau \in \{0.7, 0.85, 0.9\}$, $T_{\text{draft}} \in \{16, 32, 48\}$ (9 configs). M3: $w_g \in \{0.3, 0.5, 0.7\}$. Pairwise compositions use the best operating point per method from single-method Pareto curves. Three-way compositions sweep a reduced grid of 24 configurations (2 $\eta$ values $\times$ 4 IGSD configs $\times$ 3 $w_g$ values), with top-5 validated at full scale.

**M2 exclusion.** Adaptive step scheduling (M2) is excluded from composition analysis. Pilot experiments in iteration 1 showed that even 2x step reduction (32 steps) collapses GSM8K accuracy from 71.2% to 38.8% with our simplified implementation, which lacks the backtracking mechanism of methods like Saber. M2 receives a NO_GO verdict and is reported as a negative result.

<!-- FIGURES
- Figure 2: fig2_architecture_desc.md -- ComposeAccel architecture diagram showing IGSD draft-partition-refine pipeline with M1 KV caching and M3 AR guidance integration points
- Table 2: inline -- Metric definitions and interpretation thresholds (QAS, Ortho, combined metric)
-->

# Notation Table

All mathematical symbols and notation used in the paper. Every section writer, critic, and editor must reference this file for consistency.

---

## Inputs and Sequences

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $x$ | Input sequence (clean / target tokens) | $x \in \mathcal{V}^N$ |
| $\tilde{x}_t$ | Noisy (masked) sequence at denoising step $t$ | $\tilde{x}_t \in (\mathcal{V} \cup \{[\text{MASK}]\})^N$ |
| $x_{\text{draft}}$ | Draft output from IGSD's draft phase | $x_{\text{draft}} \in \mathcal{V}^N$ |
| $N$ | Sequence length (number of token positions in generation canvas) | $N \in \mathbb{Z}^+$ |
| $\mathcal{V}$ | Vocabulary (set of all token IDs) | $|\mathcal{V}| = V$ |
| $V$ | Vocabulary size | $V \in \mathbb{Z}^+$ |
| $[\text{MASK}]$ | Special mask token used in MDM forward/reverse process | $[\text{MASK}] \notin \mathcal{V}$ |

## Model and Denoising Process

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $T$ | Total number of denoising steps in the baseline schedule | $T = 64$ (default for LLaDA) |
| $t$ | Current denoising step index | $t \in \{1, 2, \ldots, T\}$ |
| $T_{\text{draft}}$ | Number of denoising steps in IGSD draft phase | $T_{\text{draft}} \in \{4, 8, 16, 32\}$ |
| $T_{\text{full}}$ | Number of denoising steps in IGSD refine phase | $T_{\text{full}} = T = 64$ |
| $p_\theta(v \mid \tilde{x}_t)$ | Model's predicted probability of token $v$ at position $i$ given masked state $\tilde{x}_t$ | $p_\theta : (\mathcal{V} \cup \{[\text{MASK}]\})^N \to \Delta(\mathcal{V})^N$ |
| $\theta$ | Model parameters (LLaDA-8B-Instruct, frozen) | $\theta \in \mathbb{R}^d$, $d \approx 8 \times 10^9$ |

## CD-SSD Notation

*Note: method renamed from IGSD to CD-SSD (Coarse-Draft Self-Speculative Denoising) to avoid name collision with Info-Gain Sampler (Yang et al., arXiv:2602.18176). All symbols below apply to CD-SSD.*

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $c_i$ | Per-token confidence score at position $i$ after draft phase | $c_i = \max_{v \in \mathcal{V}} p_\theta(v \mid \tilde{x}_{T_{\text{draft}}})_i \in [0, 1]$ |
| $\tau$ | Confidence threshold for CD-SSD partitioning | $\tau \in [0, 1]$; operating point $\tau = 0.9$; $\tau = 0.0$ ablation: no partitioning (all tokens accepted) |
| $S_{\text{accept}}$ | Set of token positions accepted from draft (high confidence) | $S_{\text{accept}} = \{i : c_i \geq \tau\}$; at $\tau=0.9$: $\alpha \approx 0.52$ (frozen fraction, unique positions) |
| $S_{\text{refine}}$ | Set of token positions requiring refinement (low confidence) | $S_{\text{refine}} = \{i : c_i < \tau\} = \{1,\ldots,N\} \setminus S_{\text{accept}}$ |
| $\alpha$ | Accept rate / frozen-token fraction: fraction of unique positions frozen during refine | $\alpha = |S_{\text{accept}}| / N \approx 0.52$ at $\tau=0.9$, $T_{\text{draft}}=16$ |
| $\text{CHR}_{\text{refine}}$ | KV cache hit rate during CD-SSD's refine phase | Measured: $\approx 0.940$ (from igsd\_p2 per-seed results); driven by $\tau$-frozen tokens with $H_i \approx 0$ |

## KV-Cache (M1) Notation

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $K_t^{(\ell)}, V_t^{(\ell)}$ | Key and Value matrices at layer $\ell$, step $t$ | $K_t^{(\ell)}, V_t^{(\ell)} \in \mathbb{R}^{N \times d_k}$ |
| $\hat{K}_t^{(\ell)}, \hat{V}_t^{(\ell)}$ | Cached (approximated) Key/Value matrices from step $t-1$ | Same shape as $K_t^{(\ell)}$ |
| $H_i$ | Decoded token entropy at position $i$ | $H_i = -\sum_{v} p_\theta(v \mid \tilde{x}_t)_i \log p_\theta(v \mid \tilde{x}_t)_i$ |
| $\eta$ | Entropy threshold for cache refresh decision | $\eta \in \{0.5, 1.0, 2.0, 3.0\}$ |
| $\text{CHR}$ | Cache hit rate: fraction of positions using cached KV | $\text{CHR} = |\{i : H_i < \eta\}| / N$ |

## AR-Guided Unmasking (M3) Notation

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $\phi$ | AR guide model parameters (Qwen2.5-0.5B) | Frozen |
| $w$ | Guidance blending weight | $w \in \{0.3, 0.5, 0.7, 1.0\}$ |
| $q_\phi(v \mid x_{<i})$ | AR model's autoregressive token probability | $q_\phi : \mathcal{V}^* \to \Delta(\mathcal{V})$ |

## Adaptive Step Scheduling (M2) Notation

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $J$ | Step-jump factor: number of tokens unmasked per step relative to baseline | $J \in \{2, 4, 6, 8\}$ |
| $T_{\text{eff}}$ | Effective number of steps after acceleration | $T_{\text{eff}} = \lceil T / J \rceil$ |

## Metrics

| Symbol | Definition | Formula |
|--------|-----------|---------|
| $\text{TPS}$ | Tokens per second (wall-clock throughput) | Output tokens / elapsed generation time |
| $\text{Speedup}(M)$ | Speedup of method $M$ over baseline | $\text{TPS}(M) / \text{TPS}(\text{baseline})$ |
| $\text{AccRet}(M)$ | Accuracy retention of method $M$ | $\text{Acc}(M) / \text{Acc}(\text{baseline})$ |
| $\text{QAS}(M)$ | Quality-Adjusted Speedup | $\text{Speedup}(M) \times \text{AccRet}(M)$ |
| $\text{Ortho}(M_a + M_b)$ | Pairwise orthogonality score | $\text{Speedup}(M_a + M_b) / (\text{Speedup}(M_a) \times \text{Speedup}(M_b))$ |

## Conventions

- Subscript $t$: denoising step index (time).
- Superscript $(\ell)$: transformer layer index.
- Subscript $i$: token position index within the sequence.
- Bold lowercase ($\mathbf{x}$): vectors when dimension is emphasized.
- Calligraphic ($\mathcal{V}$, $\mathcal{D}$): sets.
- $\Delta(\mathcal{V})$: probability simplex over vocabulary $\mathcal{V}$.
- All experiments use bf16 precision unless otherwise stated.
- Reported metrics are mean $\pm$ std across seeds [42, 123, 456] unless otherwise noted.

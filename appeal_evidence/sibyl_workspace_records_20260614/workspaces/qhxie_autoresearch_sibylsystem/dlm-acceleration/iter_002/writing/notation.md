# Notation Table -- ComposeAccel (Iteration 2)

All subsequent section writers, critics, and the editor reference this file to ensure consistent mathematical notation throughout the paper.

---

## Inputs and Data

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\mathbf{x}$ | Input sequence of tokens | $\mathbf{x} \in \mathcal{V}^N$ |
| $N$ | Sequence length (number of tokens) | Includes prompt + generation |
| $\mathcal{V}$ | Vocabulary | $|\mathcal{V}| = V$ |
| $\mathbf{m}_t$ | Binary mask at denoising step $t$ | $m_{t,i} = 1$ if position $i$ is masked |
| $\mathbf{x}_t$ | Noisy sequence at step $t$ | Masked positions replaced with [MASK] |

## Denoising Process

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $T$ | Total denoising steps | Default $T = 64$ for LLaDA |
| $t$ | Current denoising step index | $t \in \{0, 1, \ldots, T\}$ |
| $p_t(\cdot \mid \mathbf{x}_t)$ | Model logits at step $t$ | $p_t \in \mathbb{R}^{N \times V}$ |
| $\theta$ | DLM model parameters | Fixed (training-free setting) |
| $f_\theta$ | DLM forward pass (transformer) | $f_\theta(\mathbf{x}_t) \to p_t$ |

## Method M1: Entropy-Based KV Caching

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\eta$ | Entropy threshold for cache reuse | $\eta \in \{0.5, 1.0, 2.0\}$ |
| $H_i^t$ | Entropy of logit distribution at position $i$, step $t$ | $H_i^t = -\sum_v p_t(v \mid i) \log p_t(v \mid i)$ |
| $\mathcal{C}_t$ | Set of cached positions at step $t$ | $\mathcal{C}_t = \{i : H_i^{t-1} < \eta\}$ |
| $\text{CHR}_t$ | Cache hit rate at step $t$ | $\text{CHR}_t = |\mathcal{C}_t| / N_{\text{gen}}$ |
| $\overline{\text{CHR}}$ | Mean cache hit rate across steps | Reported per experiment |
| $\mathbf{K}_t, \mathbf{V}_t$ | Key and value matrices at step $t$ | $\mathbf{K}_t, \mathbf{V}_t \in \mathbb{R}^{N \times d_k}$ per head |

## Method IGSD: Information-Geometric Step Distillation

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $T_{\text{draft}}$ | Number of draft-phase denoising steps | $T_{\text{draft}} \in \{16, 32, 48\}$ |
| $T_{\text{full}}$ | Total denoising steps (refine target) | $T_{\text{full}} = 64$ |
| $\tau$ | KL-divergence confidence threshold | $\tau \in \{0.7, 0.85, 0.9\}$ |
| $\text{KL}_t$ | Mean token-level KL divergence at step $t$ | $\text{KL}_t = \frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} D_{\text{KL}}(p_t(i) \| p_{t-1}(i))$ |
| $\mathcal{S}_{\text{accept}}$ | Set of accepted (frozen) token positions after draft phase | Positions with confidence above $\tau$ |
| $\mathcal{S}_{\text{reject}}$ | Set of rejected token positions requiring refinement | $\mathcal{S}_{\text{reject}} = \{1, \ldots, N\} \setminus \mathcal{S}_{\text{accept}}$ |
| $\alpha$ | Frozen fraction | $\alpha = |\mathcal{S}_{\text{accept}}| / N_{\text{gen}}$; measured $\alpha = 0.886 \pm 0.133$ |
| $r_{\text{accept}}$ | Accept rate (fraction of draft steps accepted) | Measured $r_{\text{accept}} \approx 0.92$--$0.96$ |

## Method M3: AR-Guided Unmasking

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $g_\phi$ | AR guide model | Qwen2.5-0.5B with parameters $\phi$ |
| $q_t(\cdot \mid \mathbf{x}_t)$ | AR guide model logits at step $t$ | $q_t \in \mathbb{R}^{N \times V}$ |
| $w_g$ | Guidance weight | $w_g \in \{0.3, 0.5, 0.7\}$ |
| $\tilde{p}_t$ | Guided logits | $\tilde{p}_t(i) = (1 - w_g) \cdot p_t(i) + w_g \cdot q_t(i)$ for masked $i$ |

## Evaluation Metrics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\text{Acc}(M)$ | Accuracy of method $M$ on a benchmark | Exact match for GSM8K/MATH500; pass@1 for HumanEval |
| $\text{Acc}_0$ | Baseline accuracy (vanilla 64-step) | GSM8K: 0.712; MATH500: 0.111 |
| $\text{AccRet}(M)$ | Accuracy retention | $\text{AccRet}(M) = \text{Acc}(M) / \text{Acc}_0$ |
| $\text{TPS}(M)$ | Tokens per second | Wall-clock, generation-only, post-warmup |
| $\text{TPS}_0$ | Baseline TPS | GSM8K: 33.8; MATH500: 79.1 |
| $S(M)$ | Speedup | $S(M) = \text{TPS}(M) / \text{TPS}_0$ |
| $\text{QAS}(M)$ | Quality-Adjusted Speedup | $\text{QAS}(M) = S(M) \times \text{AccRet}(M)$ |
| $\text{Ortho}(A, B)$ | Orthogonality metric for methods $A$ and $B$ | $\text{Ortho}(A, B) = \text{QAS}(A + B) / \max(\text{QAS}(A), \text{QAS}(B))$ |
| $\text{Metric}_{\text{comb}}$ | Combined benchmark metric | $0.7 \times \text{GSM8K} + 0.3 \times \text{MATH500}$ |

## Thresholds and Interpretation

| Value | Interpretation |
|-------|---------------|
| $\text{Ortho} > 1.0$ | Synergy: composition strictly better than best component |
| $0.8 \leq \text{Ortho} \leq 1.0$ | Near-orthogonal: composition preserves most benefit |
| $\text{Ortho} < 0.8$ | Interference: composition degrades performance |

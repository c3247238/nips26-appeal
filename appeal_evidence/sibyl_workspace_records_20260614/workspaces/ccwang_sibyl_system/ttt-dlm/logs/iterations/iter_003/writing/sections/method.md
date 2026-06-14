# 3. Method: Denoising-Time Adaptation (DTA)

## 3.1 Core Algorithm

Masked diffusion language models (MDLMs) generate text by iteratively denoising a fully masked sequence $\mathbf{x}_T = ([\texttt{MASK}], \ldots, [\texttt{MASK}])$ over $T$ steps. At each step $t$, the model $f_\theta$ predicts all masked positions conditioned on the current partially revealed sequence $\mathbf{x}_t$, then reveals a subset of tokens according to a confidence-based or schedule-based policy. After $T$ steps, the fully unmasked output $\mathbf{x}_0$ is returned.

A critical limitation of this process is that **no information persists across denoising steps**: each step $t$ receives only the discrete tokens $\mathbf{x}_t$ and recomputes all representations from scratch. The continuous representations---logits, attention patterns, hidden states---computed at step $t$ are discarded before step $t-1$ begins. We call this the **Information Island** problem (cf. Xia et al., 2026).

Our key observation is that DLM denoising is structurally analogous to test-time training (TTT). At each step, the model performs masked language modeling on a progressively revealed sequence---precisely the same self-supervised objective used during pretraining. Yet current DLMs never update their parameters during this process. DTA makes this implicit TTT explicit by adding lightweight LoRA parameter updates within the denoising loop.

Concretely, DTA augments each denoising step with an alternating E-step/M-step structure:

**E-step (Standard Denoising).** Given the current sequence $\mathbf{x}_t$ and augmented model $f_{\theta + \Delta\theta}$, predict distributions over all masked positions and reveal tokens according to the base model's sampling schedule:
$$\hat{y}_i \sim p_{\theta + \Delta\theta}(x_i \mid \mathbf{x}_t), \quad \forall\, i \in \mathcal{M}_t$$
where $\mathcal{M}_t$ denotes the set of masked positions at step $t$. Tokens are revealed by confidence ranking or the `origin` sampling algorithm (Dream; Gong et al., 2025), yielding $\mathbf{x}_{t-1}$ with a strictly larger set of revealed positions $\mathcal{R}_{t-1} \supset \mathcal{R}_t$.

**M-step (DTA Update).** Let $\mathcal{R}_{t-1}$ be the set of all currently revealed positions after the E-step. We randomly mask a fraction $\rho = 0.2$ of these positions to construct a self-supervised training signal:
$$\mathcal{S} \subseteq \mathcal{R}_{t-1}, \quad |\mathcal{S}| = \lfloor \rho \cdot |\mathcal{R}_{t-1}| \rfloor$$
We compute the masked language modeling loss on the masked subset:
$$\mathcal{L}_{\text{DTA}} = -\frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \log p_{\theta + \Delta\theta}\!\left(x_i^{*} \mid \texttt{mask}(\mathbf{x}_{t-1}, \mathcal{S})\right)$$
where $x_i^{*}$ is the token revealed at position $i$ and $\texttt{mask}(\mathbf{x}_{t-1}, \mathcal{S})$ denotes the sequence with positions in $\mathcal{S}$ replaced by $[\texttt{MASK}]$. The LoRA parameters are then updated via a single AdamW step:
$$\Delta\theta \leftarrow \gamma \cdot \Delta\theta - \eta \cdot \nabla_{\Delta\theta} \mathcal{L}_{\text{DTA}}$$
where $\gamma \in [0, 1]$ is a cumulative decay factor and $\eta$ is the learning rate.

The complete DTA procedure is summarized in Algorithm 1.

---

**Algorithm 1: Denoising-Time Adaptation (DTA)**

```
Input: prompt x_prompt, base DLM f_θ, denoising steps T,
       LoRA config (rank r, layers L_lora, lr η, decay γ, warmup w)
Output: generated sequence x_0

1.  Initialize x_T = [MASK, ..., MASK]  (generation area)
2.  Initialize Δθ = 0  (zero-initialized LoRA adapters)
3.  Initialize AdamW optimizer for Δθ
4.  For t = T, T-1, ..., 1:
      // E-step: standard denoising with augmented model
5.    logits = f_{θ+Δθ}(x_t)
6.    x_{t-1} = sample_and_reveal(logits, schedule(t))

      // M-step: parameter update (skip warmup phase)
7.    If t/T < (1 - w):    // i.e., past warmup fraction
8.      S = random_mask(revealed_tokens(x_{t-1}), ratio=ρ)
9.      L = -mean_{i∈S} log p_{θ+Δθ}(x_i* | mask(x_{t-1}, S))
10.     Clip gradients: ||∇L|| ≤ 1.0
11.     Δθ ← γ · Δθ − η · AdamW_step(∇L)
12.   EndIf
13. EndFor
14. Return x_0
```

---

### Design Decisions

We now motivate the key design choices, each informed by pilot experiments on 16 Countdown problems with Dream-7B-Instruct.

**LoRA injection.** We insert rank-$r$ LoRA adapters into the `gate_proj`, `up_proj`, and `down_proj` matrices of the last $L_{\text{lora}}$ Transformer layers. With default settings $r = 4$ and $L_{\text{lora}} = 2$ (layers 26--27 of Dream-7B's 28-layer architecture), this adds approximately 540K trainable parameters---0.007% of the 7.6B total. The LoRA output is computed as:
$$h' = h + \frac{\alpha}{r} B A h$$
where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ are the low-rank factors.

**Zero initialization.** Both $A$ and $B$ are initialized such that $BA = 0$ at step $T$. This ensures the augmented model $f_{\theta + \Delta\theta}$ is exactly equivalent to the base model $f_\theta$ at the start of denoising, introducing no distribution shift before any signal has been accumulated.

**Cumulative decay ($\gamma = 0.95$).** Before each gradient step, the LoRA parameters are scaled by $\gamma$, implementing an exponential moving average that prevents unbounded parameter drift. In our ablation (Section 5.3), $\gamma = 1.0$ (no decay) leads to LoRA Frobenius norms reaching $\sim$0.96, exhibiting clear drift, while $\gamma = 0.95$ keeps norms bounded below 0.10. This mechanism is analogous to the forget gate in Titans (Behrouz et al., 2025), adapted from sequence-level to denoising-time-level memory.

**Warmup ($w = 0.2$).** DTA updates are skipped during the first 20% of denoising steps. At this stage, only $\sim$2--5 tokens have been revealed (out of 256 generation positions), providing insufficient signal for meaningful gradient updates. Empirically, warmup prevents early noisy gradients from biasing the LoRA adapter before the E-step has revealed enough context.

**Mask-and-predict loss.** The M-step masks 20% of *revealed* tokens and predicts them, rather than using a self-consistency loss (i.e., predicting the model's own logits). Pilot experiments showed that self-consistency loss yields near-zero gradients because the model already agrees with its own predictions, producing LoRA norms of effectively zero. The mask-and-predict loss provides a meaningful training signal (mean loss $\sim$0.08--0.16 per step in pilots) because re-predicting a revealed token after masking it requires integrating information from surrounding context.

**Optimizer.** We use AdamW (lr=$5 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $= 0.01$) with gradient clipping at 1.0. Earlier experiments with SGD produced LoRA norms too small to affect predictions; AdamW's adaptive learning rates provide the necessary per-parameter scaling for the sparse gradient signal.

**Computational overhead.** Each DTA update requires one additional backward pass through the last $L_{\text{lora}}$ layers, approximately doubling the per-step cost. Over $T = 128$ denoising steps with warmup, DTA performs $\sim$103 update steps, resulting in $\sim$4$\times$ wall-clock overhead compared to vanilla denoising (15.9s vs. 3.7s per sample on an NVIDIA RTX PRO 6000 Blackwell). Importantly, the LoRA parameters are reset to zero for each new input---no information leaks between samples.

**Relationship to DLM inference.** DTA operates entirely in parameter space and does not modify the token-level sampling process. The E-step uses the standard sampling algorithm of the base model. This makes DTA *orthogonal* to token-space interventions such as remasking (Section 3.4), enabling straightforward combination.

## 3.2 Variational Interpretation (VDTA)

We now provide a variational inference interpretation of DTA that formalizes the intuition of "learning during denoising" and yields two theoretical guarantees.

### Setup

Consider the joint distribution over the target sequence $\mathbf{x}_0$ and the LoRA adapter parameters $\Delta\theta$. The standard DLM denoising process optimizes only over $\mathbf{x}_0$ while keeping $\theta$ fixed. DTA extends this to a joint optimization problem:
$$\max_{\mathbf{x}_0,\, \Delta\theta}\; \log p(\mathbf{x}_0 \mid \mathbf{x}_{\text{prompt}};\, \theta + \Delta\theta)$$

We frame DTA as coordinate-wise optimization in this joint space, alternating between:
- **E-step**: Fix $\Delta\theta^{(t)}$, update $\mathbf{x}_t \to \mathbf{x}_{t-1}$ via the augmented model $f_{\theta + \Delta\theta^{(t)}}$
- **M-step**: Fix $\mathbf{x}_{t-1}$, update $\Delta\theta^{(t)} \to \Delta\theta^{(t+1)}$ by minimizing the masked LM loss

This alternation is a form of Expectation-Maximization (EM) in the extended space $(\mathbf{x}_0, \Delta\theta)$.

### Proposition 1 (ELBO Monotonicity)

Under the following regularity conditions:
1. The DTA loss $\mathcal{L}_{\text{DTA}}(\Delta\theta)$ is $\mu$-strongly convex in $\Delta\theta$ (ensured by $L_2$ regularization via weight decay),
2. The model $f_{\theta + \Delta\theta}$ is continuous in $\Delta\theta$,
3. The learning rate $\eta$ is sufficiently small,

each E-M step improves the variational lower bound:
$$\text{ELBO}(\mathbf{x}_{t-1}, \Delta\theta^{(t+1)}) \geq \text{ELBO}(\mathbf{x}_t, \Delta\theta^{(t)})$$

The ELBO is defined as:
$$\text{ELBO} = \mathbb{E}_{q(\mathbf{x}_0)}\!\left[\log p(\mathbf{x}_0 \mid \mathbf{x}_{\text{prompt}};\, \theta + \Delta\theta)\right] - D_{\text{KL}}\!\left(q(\Delta\theta) \,\|\, p(\Delta\theta)\right)$$
where $p(\Delta\theta)$ is a zero-centered Gaussian prior (induced by weight decay) and $q(\Delta\theta) = \delta(\Delta\theta^{(t)})$ is the point estimate maintained by DTA.

*Proof sketch.* The E-step improves the first term by moving $\mathbf{x}_t$ toward higher-likelihood regions under the current model. The M-step improves the first term by adapting $\Delta\theta$ to better explain $\mathbf{x}_{t-1}$, while the $L_2$ regularization bounds the KL penalty. Strong convexity ensures the single gradient step makes non-negative progress. $\square$

### Proposition 2 (Information Accumulation)

The mutual information between the LoRA parameters and the target sequence increases monotonically across denoising steps:
$$I(\Delta\theta^{(t+1)};\, \mathbf{x}_0) \geq I(\Delta\theta^{(t)};\, \mathbf{x}_0)$$

*Intuition.* At each step, the M-step update incorporates gradient information from newly revealed tokens. Since $|\mathcal{R}_{t-1}| > |\mathcal{R}_t|$ (strictly more tokens are revealed), each update integrates information from a strictly larger context. The cumulative decay $\gamma < 1$ ensures that older information is gradually down-weighted but never erased within the same denoising trajectory. Empirically, we observe that prediction confidence on held-out masked positions increases monotonically from 0.969 at step $T$ to 0.995 at step 1 (Section 5.4), consistent with this proposition.

### Contrast with Autoregressive TTT

In autoregressive TTT (Sun et al., 2024; Behrouz et al., 2025), the hidden state or fast weights are updated along the *sequence position* axis:
$$W^{(i+1)} = W^{(i)} - \eta\, \nabla_W \mathcal{L}(x_i;\, W^{(i)})$$
This requires a causal structure: token $x_i$ is processed only after $x_1, \ldots, x_{i-1}$. DTA instead updates along the *denoising time* axis:
$$\Delta\theta^{(t+1)} = \gamma \cdot \Delta\theta^{(t)} - \eta\, \nabla_{\Delta\theta} \mathcal{L}_{\text{DTA}}(\mathbf{x}_{t-1};\, \Delta\theta^{(t)})$$

This distinction has two important consequences:
1. **Bidirectional context.** Each DTA update conditions on the *full* partially revealed sequence (both left and right context of any position), exploiting the DLM's bidirectional attention. AR TTT is restricted to unidirectional (left-to-right) context.
2. **Inherent self-supervision.** The DLM's denoising objective is itself a masked LM task, providing a natural self-supervised signal without requiring an auxiliary loss function. AR TTT must construct a separate self-supervised objective (e.g., next-token prediction on a sliding window).

## 3.3 Information Augmentation Spectrum

To systematically evaluate *how much* cross-step information matters and *at what cost*, we introduce an information augmentation spectrum with four levels of increasing expressivity and computational overhead. This spectrum serves as a controlled ablation framework: each level adds a specific type of cross-step information transfer, allowing us to isolate the contribution of each mechanism.

### Level 0: Vanilla (No Cross-Step Information)

The standard DLM denoising process. Each step receives only the discrete token sequence $\mathbf{x}_t$; all continuous representations are discarded. This is the baseline against which all other levels are compared.

- **Cross-step information**: None
- **Compute overhead**: 1$\times$ (baseline)

### Level 1: DMI --- Diffusion Memory Injection

DMI injects a soft memory of the previous step's predictions into the current step's input. At step $t-1$, we compute a soft embedding from the previous step's logits:
$$\mathbf{e}_i^{\text{soft}} = \text{softmax}(\mathbf{z}_i^{(t)} / \tau_{\text{soft}}) \cdot \mathbf{E}$$
where $\mathbf{z}_i^{(t)}$ is the logit vector at position $i$ from step $t$, $\tau_{\text{soft}}$ is a temperature parameter, and $\mathbf{E}$ is the token embedding matrix. This soft embedding is mixed with the hard (discrete) embedding at masked positions:
$$\mathbf{h}_i^{(t-1)} = (1 - \alpha)\, \mathbf{e}_i^{\text{hard}} + \alpha\, \mathbf{e}_i^{\text{soft}}, \quad i \in \mathcal{M}_{t-1}$$
with mixing weight $\alpha = 0.3$ and $\tau_{\text{soft}} = 0.5$.

- **Cross-step information**: Embedding-level (soft token distributions from previous step)
- **Compute overhead**: $\sim$1.05$\times$ (one softmax-embedding matrix multiply per step)
- **Mechanism**: Injects a continuous "memory" of previous predictions, breaking the discrete information bottleneck between steps

### Level 2: SCP --- Self-Contradiction Probing

SCP uses leave-one-out probing to detect tokens that are *self-contradictory*: positions where the model's prediction changes when the token at that position is masked out and re-predicted. At each probing step:
1. For each revealed position $i \in \mathcal{R}_t$, mask position $i$ and run a forward pass
2. If the model's top-1 prediction differs from the current token $x_i$, flag position $i$ as contradictory
3. Remask all flagged positions, allowing the next denoising step to re-predict them

- **Cross-step information**: Token-level (identifies specific unreliable positions via bidirectional self-consistency checking)
- **Compute overhead**: $\sim$7$\times$ (one forward pass per revealed token per probing step; batched but still expensive)
- **Mechanism**: Exploits the DLM's bidirectional attention to verify each token against its surrounding context

### Level 3: DTA --- Denoising-Time Adaptation

The full DTA method described in Section 3.1. LoRA parameters accumulate gradient information from all previous steps, creating a persistent parameter-level memory across the entire denoising trajectory.

- **Cross-step information**: Parameter-level (gradient-based adaptation encoding patterns from all revealed tokens)
- **Compute overhead**: $\sim$4$\times$ (one backward pass through last 2 layers per step)
- **Mechanism**: Online learning that adapts the model's weights to the specific input being generated

### Spectrum Summary

| Level | Method | Information Type | Granularity | Compute | Persistent Memory |
|-------|--------|-----------------|-------------|---------|-------------------|
| 0 | Vanilla | None | --- | 1$\times$ | No |
| 1 | DMI | Soft embeddings | Embedding | $\sim$1$\times$ | 1-step lookback |
| 2 | SCP | Token contradictions | Token | $\sim$7$\times$ | No (stateless check) |
| 3 | DTA | LoRA gradients | Parameter | $\sim$4$\times$ | Full trajectory |

The spectrum is designed so that each level introduces a qualitatively different form of cross-step information. DMI (Level 1) provides a simple soft "echo" of the previous step; SCP (Level 2) provides targeted error detection but no persistent memory; DTA (Level 3) provides persistent, gradient-based adaptation that accumulates understanding across the entire trajectory. By comparing performance across levels, we can determine whether the value of cross-step information lies in the *type* of information (embedding vs. token vs. parameter), the *persistence* of memory (1-step vs. stateless vs. full trajectory), or both.

## 3.4 Combination with Remasking

DTA and remasking-based methods (e.g., ReMDM-conf; Nisonoff et al., 2025) operate in fundamentally different spaces:

- **DTA** modifies the model's *parameters*, improving the quality of predictions at all positions simultaneously. It does not change which tokens are revealed or remasked.
- **Remasking** modifies the *token sequence*, selectively replacing low-confidence tokens with $[\texttt{MASK}]$ and re-predicting them. It does not change the model.

This orthogonality makes them naturally complementary. In the combined DTA+ReMDM-conf mode, each denoising step proceeds as:

1. **E-step with remasking**: Predict masked positions using $f_{\theta + \Delta\theta}$, reveal tokens by confidence, then remask the bottom-$k$ tokens by confidence (ReMDM-conf's `conf` sampler)
2. **M-step**: Compute DTA loss on the *post-remasking* revealed set and update LoRA parameters

The combined method benefits from both mechanisms: DTA provides a progressively better model for prediction (reducing the *rate* of errors), while remasking provides a second chance to correct errors that occur despite the improved model (reducing the *impact* of errors). DTA+ReMDM-conf is our recommended configuration for production use, as it addresses complementary failure modes at a combined overhead of $\sim$5$\times$ vanilla.

<!-- FIGURES
- Figure 1: fig1_dta_overview_desc.md — Architecture diagram illustrating the E-step/M-step alternation within the DTA denoising loop, contrasting standard DLM denoising with DTA-enhanced denoising
- Figure 2: fig2_info_spectrum_desc.md — Flow chart visualizing the four levels of the information augmentation spectrum with their mechanisms and compute overhead
-->

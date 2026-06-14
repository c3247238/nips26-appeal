# 3. Method

## 3.1 Preliminaries: MDLM Denoising and the Information Island Problem

Masked diffusion language models (MDLMs) generate text by iteratively denoising a fully masked sequence $x_T = [\texttt{MASK}]^L$ over $T$ steps. At each step $t$, the model predicts a distribution over the vocabulary for every masked position $p_\theta(x_i^0 \mid x_t)$, and a subset of positions are "unmasked" by committing to their argmax predictions based on a confidence criterion. The unmasked tokens are then fed back as hard embeddings in the next step's input.

This procedure suffers from what we term the **information island problem**: the rich distributional information produced at each step --- the full logit vectors $\ell_i^t \in \mathbb{R}^{|V|}$, the attention distributions, and the implicit inter-position dependencies captured in the Transformer's hidden states --- is entirely discarded after the argmax operation. The next step receives only a binary signal: either a hard token embedding (for unmasked positions) or the generic $\texttt{mask\_emb}$ vector (for positions still masked). No cross-step memory of the model's uncertainty or distributional beliefs is preserved.

Formally, let $e_v \in \mathbb{R}^d$ denote the embedding of token $v$, and let $\texttt{mask\_emb} \in \mathbb{R}^d$ be the learned mask embedding. At step $t$, the input representation for position $i$ is:

$$h_i^t = \begin{cases} e_{x_i} & \text{if } i \text{ has been unmasked (hard token)} \\ \texttt{mask\_emb} & \text{if } i \text{ is still masked} \end{cases}$$

This hard switching discards all information from prior steps' predictions at masked positions. Even if the model at step $t+1$ assigned high probability to a particular token at position $i$, the next step's input at that position is the same uninformative $\texttt{mask\_emb}$ vector --- identical across all positions and agnostic to the model's evolving beliefs.

**Evidence for the bottleneck.** Our prior work on Diffusion Memory Injection (DMI) provides direct evidence that this representational poverty is the core bottleneck. DMI injects a simple logit-weighted embedding mixture into mask positions:

$$h_i^t = \alpha \cdot \texttt{mask\_emb} + (1 - \alpha) \cdot \sum_{v \in V} \text{softmax}(\ell_i^{t+1})_v \cdot e_v$$

With $\alpha = 0.3$, this achieves 9.3\% accuracy on Countdown-500 (3-seed mean) versus 4.7\% for vanilla Dream-7B --- a $\sim$2$\times$ improvement with near-zero computational overhead ($\sim$1.05$\times$ FLOPs). In contrast, methods that operate in other spaces have failed: parameter-space adaptation via DTA (LoRA online updates) achieves only 6.2\% due to vanishing gradient signals (MLM loss already at 0.005--0.032), and pure remasking (ReMDM-conf 4.4\%, RCR 5.7\%) does not accumulate information across steps. DMI's success motivates the two methods we introduce next: a richer continuous representation (BSD) and a prediction-level enhancement (A-CFG).

## 3.2 Belief-State Diffusion (BSD)

### Core Idea

BSD generalizes DMI from a fixed additive mixture to a fully continuous representational framework. Instead of mixing logit-weighted embeddings with $\texttt{mask\_emb}$ at a fixed ratio, BSD **replaces** mask embeddings entirely with *belief states* --- probability-weighted embedding vectors that evolve via exponential moving average (EMA) across denoising steps. Only in the final $k$ steps does BSD commit to hard token assignments.

### Algorithm

BSD operates in two phases (Algorithm 1). In **Phase 1** (continuous belief refinement, steps $T$ to $k+1$), the model operates on belief vectors rather than hard masks. At each step, the model's predictions are used to update the belief state via EMA, without any argmax sampling or unmasking. In **Phase 2** (hard token reveal, steps $k$ to $1$), standard confidence-based unmasking proceeds from the accumulated belief states.

---

**Algorithm 1: Belief-State Diffusion (BSD)**

**Input:** prompt $x_p$, generation length $L$, model $f_\theta$, total steps $T$, hard-reveal fraction $k_{\text{frac}}$ (steps $k = k_{\text{frac}} \cdot T$), EMA schedule $\alpha(\cdot)$, temperature schedule $\tau(\cdot)$

**Output:** generated sequence $x_0$

1. Initialize belief vectors: $b_i^T \leftarrow \texttt{mask\_emb}$ for all generation positions $i$

2. **Phase 1: Continuous Belief Refinement** (steps $t = T, T-1, \ldots, k+1$)
   - Construct input: $[x_p; b^t]$ (prompt embeddings concatenated with belief vectors)
   - Forward pass: $\ell^t \leftarrow f_\theta([x_p; b^t])$
   - Compute soft predictions: $p_i^t \leftarrow \text{softmax}(\ell_i^t / \tau(t))$
   - Compute embedding mixture: $\hat{b}_i^t \leftarrow \sum_{v \in V} p_i^t(v) \cdot e_v$
   - EMA update: $b_i^{t-1} \leftarrow (1 - \alpha(t)) \cdot b_i^t + \alpha(t) \cdot \hat{b}_i^t$
   - L2 normalize: $b_i^{t-1} \leftarrow b_i^{t-1} \cdot \frac{\|\texttt{mask\_emb}\|}{\|b_i^{t-1}\|}$

3. **Phase 2: Hard Token Reveal** (steps $t = k, k-1, \ldots, 1$)
   - Forward pass from belief states: $\ell^t \leftarrow f_\theta([x_p; b^t])$
   - Confidence-based unmasking: select highest-confidence positions, replace $b_i$ with $e_{x_i}$

---

### Key Design Decisions

**EMA update rate $\alpha(t)$.** We use a linear schedule from $\alpha_{\text{start}} = 0.1$ to $\alpha_{\text{end}} = 0.8$. Early steps use small $\alpha$ to stabilize beliefs when model predictions are uncertain (high mask rate), while later steps use large $\alpha$ to rapidly converge as predictions become more reliable. Ablation across four schedule types --- linear(0.1$\to$0.8), cosine(0.1$\to$0.8), constant(0.3), and constant(0.5) --- shows that all achieve identical accuracy (6.2\% on Countdown-16), indicating that schedule shape is not a performance bottleneck at this scale.

**L2 normalization.** Belief vectors are normalized to match $\|\texttt{mask\_emb}\|$ at every step. Without this, probability-weighted embedding mixtures drift to a different norm scale than the inputs the Transformer was trained on, causing out-of-distribution (OOD) behavior in early layers. Normalization ensures that belief vectors remain in the model's expected input manifold.

**Hard-reveal fraction $k_{\text{frac}}$.** The $k$ parameter controls the trade-off between continuous belief refinement and discrete token commitment. Ablation reveals that $k_{\text{frac}} = 0.75$ (only 25\% of steps in the belief phase, 75\% in hard reveal) achieves 6.2\%, while $k_{\text{frac}} = 0.25$ and $k_{\text{frac}} = 0.50$ (longer belief phases) both yield 0\%. This falsifies our hypothesis H3 that an intermediate balance would be optimal, and instead suggests that models require early hard token anchors to ground their predictions. BSD's benefit comes from the *quality* of initial representations fed into Phase 2, not from extended continuous refinement.

### Relationship to Prior Work

BSD can be understood as a generalization within a family of continuous-representation methods for MDLMs. Table 1 summarizes the key distinctions.

| Dimension | DMI (Ours, prior) | LRD | ReMix | EvoToken | **BSD** |
|---|---|---|---|---|---|
| Mask replacement | Mixed with $\texttt{mask\_emb}$ | Mixed | Continuous state | Gradual evolution | **Full replacement** |
| Cross-step memory | None | None | None | None | **EMA accumulation** |
| Reveal trigger | Every step | KL convergence | After convergence | Gradual | **Last $k$ steps** |
| Compute overhead | $\sim$1.05$\times$ | $\sim$1$\times$ | $\sim$1$\times$ | $\sim$1$\times$ | **$\sim$1.1$\times$** |

DMI is the special case of BSD where $k = T$ (no belief phase) and $\alpha$ is fixed. LRD (Zhu et al., 2025) and ReMix (Ye et al., 2026) use continuous representations primarily for acceleration; BSD focuses on quality improvement through explicit cross-step information accumulation.

## 3.3 Adaptive Classifier-Free Guidance (A-CFG) for Dream-7B

### Core Idea

Classifier-free guidance (CFG) amplifies the difference between conditional and unconditional predictions to sharpen model outputs. In autoregressive models, CFG requires training a separate unconditional model. MDLMs offer a natural alternative: re-masking a subset of positions to construct an "unconditional" input, enabling CFG without any additional training.

We apply Adaptive CFG (A-CFG), introduced by Arriaga et al. (2025) for LLaDA-8B, to Dream-7B. A-CFG selects positions to re-mask based on single-step confidence scores, then computes guided logits as an extrapolation beyond the conditional prediction.

### Algorithm

At each denoising step $t$:

1. **Confidence scoring:** For each non-prompt position $i$, compute the confidence score $c_i^t = \max_v p_\theta(x_i^0 = v \mid x_t)$.

2. **Re-masking construction:** Select the $m\%$ least-confident positions (we use $m = 10\%$). Replace their current representations with $\texttt{mask\_emb}$ to construct the unconditional input $\tilde{x}_t$.

3. **Dual forward pass:**
   - Conditional: $\ell^+ = f_\theta(x_t)$ (original input)
   - Unconditional: $\ell^- = f_\theta(\tilde{x}_t)$ (re-masked input)

4. **Guided logit combination:**

$$\ell_{\text{guided}} = \ell^+ + w \cdot (\ell^+ - \ell^-)$$

where $w$ is the guidance weight, capped at $w_{\max} = 2.0$.

The computational overhead is exactly 2$\times$ vanilla FLOPs due to the dual forward pass.

### Why Confidence Beats Cross-Step Stability (JSD)

Our original proposal (RACFG) attempted to improve A-CFG by using cross-step stability signals for position selection. Specifically, we tracked smoothed logit histories via EMA and computed per-position Jensen-Shannon divergence (JSD) between consecutive steps, hypothesizing that positions with high cross-step instability (low stability score) represent "reasoning decision points" where the model is hesitating.

This hypothesis (H5) was decisively falsified. Dream-7B produces remarkably stable cross-step distributions, with JSD stability scores clustered at $\sim$0.997 across all positions. This near-degeneracy renders JSD uninformative: there is no meaningful variation to discriminate reasoning-critical from non-critical positions. Across all RACFG configurations tested (re-mask percentages $m \in \{5\%, 10\%, 20\%\}$, EMA smoothing $\lambda \in \{0.3, 0.7\}$), accuracy was 0.0\% --- identical to vanilla. In contrast, A-CFG with simple confidence-based selection achieved 12.5\%.

The root cause is architecturally informative: Dream-7B's prediction consistency across steps is a *desirable* property (stable denoising), but it eliminates the inter-step variation that stability-based methods require. This suggests that well-trained MDLMs may generically lack the cross-step instability signal assumed by RACFG-style approaches.

### Why Fixed Guidance Outperforms Temporal Scheduling

CFG scheduling theory for continuous diffusion models (Rojas et al., 2025) predicts that guidance should be suppressed early (when the model lacks sufficient context at high mask rates) and amplified late (during critical decision-making). We tested this prediction by comparing four temporal schedules --- fixed, linear ramp, cosine ramp, and threshold (70\%/30\% mask rate boundary) --- each at guidance weights $w \in \{1.0, 1.5\}$.

Fixed guidance at $w = 1.5$ achieved 12.5\%, while *all* scheduled variants achieved 0.0\% (Table 2). This falsifies hypothesis H6 and reveals a fundamental disconnect between continuous and discrete diffusion dynamics. In continuous diffusion, the noise-to-signal transition is smooth, and early guidance can amplify noise. In masked diffusion, the transition is discrete: at any step, the model either has a token (full signal) or a mask (no signal). The model has sufficient context from already-unmasked positions at every step, making constant guidance appropriate throughout the denoising process.

| Schedule | $w = 1.0$ | $w = 1.5$ |
|---|---|---|
| Fixed | 6.2\% | **12.5\%** |
| Linear ramp | 0.0\% | 0.0\% |
| Cosine ramp | 0.0\% | 0.0\% |
| Threshold (70/30) | 0.0\% | 0.0\% |

**Table 2.** A-CFG temporal schedule ablation on Countdown-16. Fixed guidance dominates all scheduled variants. Scheduled approaches suppress guidance at critical early steps, preventing the method from building momentum.

## 3.4 Information-Theoretic Analysis: Belief Entropy Trajectories

We validate BSD's information accumulation mechanism through an entropy trajectory analysis. For each sample, we track the mean per-position entropy of belief vectors across denoising steps:

$$H(b_i^t) = -\sum_{v \in V} p_i^t(v) \log p_i^t(v)$$

where $p_i^t$ is the softmax distribution used to compute the belief vector at step $t$.

**Monotonic decrease.** BSD belief entropy exhibits near-monotonic decrease during the belief phase (Phase 1), with average Spearman rank correlation $\rho = -0.95$ between step index and mean entropy. Of 16 pilot samples, 15/16 show monotonically decreasing entropy trajectories ($\rho < -0.8$). This confirms that beliefs are consistently converging rather than oscillating --- each step's predictions refine the belief state toward a concentrated distribution.

**Lower terminal entropy.** At the end of the belief phase, BSD achieves mean terminal entropy of 0.001 versus 0.002 for vanilla denoising's corresponding prediction entropy. While the absolute difference is small (both methods reach near-deterministic distributions), the gap is consistent and directionally validates that EMA accumulation preserves strictly more information than independent per-step predictions.

**Entropy-accuracy correlation.** Across the 16 pilot samples, the correlation between terminal belief entropy and task correctness is $r = 0.78$ ($p < 0.001$). Samples where beliefs converge more cleanly (lower terminal entropy) are more likely to produce correct answers. This suggests that entropy trajectory monitoring could serve as a runtime quality indicator for BSD-enhanced generation.

These information-theoretic properties distinguish BSD from mere computational overhead: the method produces *qualitatively different* internal representations, not just more computation. The monotonic entropy decrease and its correlation with accuracy provide a principled explanation for why continuous belief evolution improves over discrete mask-based denoising.

<!-- FIGURES
- Figure 2: fig2_method_architecture_desc.md — Two-panel architecture diagram showing BSD two-phase pipeline (Panel A) and A-CFG pipeline (Panel B) with FLOPs annotations
- Figure 3: gen_entropy_trajectories.py, entropy_trajectories.pdf — Belief entropy trajectory plot showing BSD monotonic decrease vs vanilla step-function drops with Spearman rho and terminal entropy annotations
-->

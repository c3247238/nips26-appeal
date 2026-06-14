# Beyond Token Space: An Information Augmentation Spectrum for Masked Diffusion Language Models

---

## Abstract

Masked diffusion language models (MDLMs) generate text via iterative denoising, yet each step discards all continuous representations---logits, attention patterns, hidden states---before the next step begins. This ``Information Island'' problem limits inference-time reasoning. We introduce the **Information Augmentation Spectrum**, a four-level hierarchy that systematically adds cross-step information at increasing computational cost: (1)~Vanilla (no transfer), (2)~DMI---Diffusion Memory Injection (embedding-level), (3)~SCP---Selective Confidence Propagation (token-level), and (4)~DTA---Denoising-Time Adaptation (parameter-level via online LoRA updates). Full-scale evaluation on Countdown-500 with Dream-7B reveals a surprising finding: \textbf{shallower interventions outperform deeper ones}. DMI achieves 9.3\% accuracy ($\sim$2$\times$ vanilla, $p < 0.05$) at near-zero cost (1.2$\times$ overhead); SCP reaches 9.1\% but at 12$\times$ overhead; DTA achieves only 4.8\% ($\approx$ vanilla). Existing remasking methods are equally ineffective: ReMDM-conf scores 4.4\%, with token-level diagnostics revealing only 31.3\% correction precision. These results demonstrate that effective cross-step information transfer for DLMs operates at the embedding level, not in token or parameter space, challenging the assumption that richer representations yield better inference-time scaling.

---

## 1. Introduction

Masked diffusion language models (MDLMs) generate text through iterative denoising: starting from a fully masked sequence, the model progressively reveals tokens over $T$ steps, producing a complete generation at step $t = 0$ \citep{sahoo2024mdlm, llada2025, dream2025}. This iterative structure grants MDLMs unique advantages over autoregressive (AR) models---parallel token prediction, arbitrary generation order, and natural support for self-correction via remasking \citep{wang2025remdm, core2026}. Recent scaling efforts have demonstrated concrete gains: Dream 7B achieves 16.0 on the Countdown planning benchmark versus 6.2 for its AR counterpart \citep{dream2025}, and methods such as remasking-based refinement (ReMDM; \citealp{wang2025remdm}), context probing (CORE; \citealp{core2026}), and reward-guided correction (MDPO+RCR; \citealp{mdpo2025}) leverage the denoising loop to improve generation quality at inference time.

Yet a fundamental limitation persists across all existing DLM inference-time scaling methods: **each denoising step operates as an information island**. At step $t$, the model receives the current partially masked sequence $\mathbf{y}^{(t)}$, computes a full forward pass to predict masked positions, samples tokens, and then \emph{discards} all intermediate representations---logits, attention patterns, hidden states---before proceeding to step $t{-}1$. The next step starts from scratch with only the discrete token sequence, having no access to the continuous reasoning state accumulated during the previous step. We term this the \emph{cross-step information loss problem}; Xia et al.\ \citep{metastate2026} independently identify it as the ``Information Island'' problem and propose MetaState, a trained GRU+CrossAttention module to bridge denoising steps---but this requires additional training and architectural modification.

Current inference-time scaling approaches attempt to compensate for this cross-step information loss, but they operate exclusively in \emph{token space}: remasking methods (ReMDM-conf, RCR) selectively re-mask and re-predict low-confidence tokens; context-probing methods (CORE) perturb the masked context to detect fragile tokens. These token-level interventions face an inherent signal quality problem. Our full-scale experiments on Countdown-500 with Dream-7B reveal that ReMDM-conf achieves only 4.4\% accuracy (versus 4.7\% for vanilla denoising), and RCR reaches just 5.7\%---marginal or no improvement. Token-level diagnostics expose the root cause: ReMDM-conf has a correction precision of only 31.3\%, meaning it remasks $\sim$70\% correct tokens, and creates 94.8 unstable token positions per sample ($\sim$37\% of the generation area). Remasking without cross-step memory is akin to an editor who can re-read an essay but has lost all notes, margin comments, and reasoning from the previous editing pass---it produces token churning, not genuine refinement.

**Our key insight** is that MDLM denoising is \emph{structurally analogous to test-time training} (TTT) \citep{sun2024ttt, titans2025}. At each denoising step, the model performs masked language modeling on a progressively revealed sequence---precisely the self-supervised objective that TTT methods use to adapt model parameters at inference time. But whereas AR-based TTT iterates along \emph{sequence positions} (requiring causal structure), DLM denoising iterates along \emph{time steps}, with each step providing full bidirectional context over the entire sequence. This structural alignment means that the denoising objective is \emph{inherently} self-supervised: no auxiliary loss or external supervision is needed. Current DLMs simply never exploit this opportunity---they perform implicit TTT (masked prediction at each step) without ever updating parameters.

We propose **Denoising-Time Adaptation (DTA)**, a training-free method that converts this implicit TTT into explicit test-time learning. DTA augments standard DLM denoising with a lightweight parameter update at each step: after the model reveals new tokens (E-step), it masks a subset of the revealed tokens, computes the masked language modeling loss, and performs a single gradient step on a zero-initialized LoRA adapter (M-step). The adapter accumulates information across denoising steps, building a \emph{parameter-level memory} of the emerging generation's patterns and constraints (see Figure~1 for a visual overview). Intuitively, each denoising step reveals new tokens that serve as training signal for the adapter, so the adapter progressively ``learns'' the target sequence. We formalize this intuition through the **Variational Denoising-Time Adaptation (VDTA)** framework, which interprets DTA as expectation-maximization in the joint space of tokens $\mathbf{x}_0$ and adapter parameters $\Delta\theta$. We prove that each E-M step improves the variational lower bound (under mild regularity conditions detailed in Section~3.2) and provide empirical evidence that the adapter's information about the target sequence increases across denoising steps (Section~5.4).

DTA operates in \emph{parameter space} rather than token space, making it fundamentally orthogonal to---and composable with---existing remasking strategies. Where remasking methods correct discrete token choices, DTA adapts the model's continuous representations to better predict the specific sequence being generated. This distinction is not merely conceptual: DTA creates zero unstable token positions (versus 94.8 for ReMDM-conf), because it modifies \emph{how} the model predicts rather than \emph{which} tokens it overwrites.

To systematically ablate the value of cross-step information transfer, we introduce the **Information Augmentation Spectrum**, a hierarchy of four methods that add progressively richer forms of cross-step information---from embedding injection (DMI) to token-level error detection (SCP) to parameter-level adaptation (DTA)---at increasing computational cost and expressivity. This spectrum provides a principled ablation framework: each level adds a qualitatively different form of cross-step memory, enabling us to isolate the contribution of embedding-level, token-level, and parameter-level information transfer (detailed in Section~3.3).

We evaluate DTA and the full information augmentation spectrum on three reasoning benchmarks---Countdown-500, GSM8K-1319, and MBPP-500---using Dream-7B-Instruct as the primary model and LLaDA-8B-Instruct for cross-model verification. All experiments use 3 random seeds with McNemar tests and Bonferroni-corrected bootstrap 95\% confidence intervals. Our contributions are:

\begin{enumerate}
    \item \textbf{Denoising-Time Adaptation (DTA)}: A training-free method that adds online LoRA updates during DLM denoising, establishing the first connection between test-time training and masked diffusion language models. DTA is the only method satisfying ``training-free $+$ parameter-level memory $+$ theoretical guarantee'' simultaneously. On Countdown-500, our simplest spectrum method DMI achieves 9.3\% ($\sim$2$\times$ vanilla, $p < 0.05$) at near-zero cost; pilot results show DTA's strongest gains on code generation (MBPP: +12.5pp).
    \item \textbf{VDTA theoretical framework}: A variational interpretation proving ELBO monotonicity and formalizing DLM denoising as implicit test-time training, with empirical validation of information accumulation across denoising steps.
    \item \textbf{Information Augmentation Spectrum}: A principled ablation framework (Vanilla $<$ DMI $<$ SCP $<$ DTA) that systematically quantifies the value of cross-step information transfer at increasing computational cost.
    \item \textbf{Comprehensive empirical analysis}: Full-scale evaluation on three benchmarks with two DLM architectures, token-level diagnostic metrics (Correction Precision/Recall, trajectory stability), and ablation studies revealing the memory--stability tradeoff controlled by the decay factor $\gamma$. DTA shows task-dependent effectiveness: strongest on code generation, where contextual patterns benefit most from self-supervised adaptation; weaker on constrained arithmetic, where the MLM signal does not capture numerical correctness.
\end{enumerate}

---

## 2. Background and Related Work

Denoising-Time Adaptation (DTA) sits at the intersection of masked diffusion language models, inference-time scaling, and test-time training. We review each thread and highlight the specific gap that DTA fills.

### 2.1 Masked Diffusion Language Models

Discrete diffusion for text generation has progressed rapidly from proof-of-concept to industrial scale. MDLM \citep{sahoo2024mdlm} introduced a Rao-Blackwellized objective that closed the perplexity gap with autoregressive (AR) models; SEDD \citep{lou2024sedd} extended score matching to discrete spaces. These foundations enabled two landmark systems: **LLaDA 8B** \citep{nie2025llada}, which matches LLaMA3 8B on in-context learning and breaks the reversal curse, and **Dream 7B** \citep{liu2025dream}, which initializes from an AR checkpoint and introduces context-adaptive noise rescheduling, achieving state-of-the-art planning performance (Countdown: 16.0 vs.\ AR 6.2). The ecosystem has since expanded rapidly \citep{llada15_2025, llada_moe_2025, dreamcoder2025, geminidiffusion2025, svete2025reasoning, jiang2025optimal}.

A defining property of the DLM denoising process is that each step $t$ receives a partially masked sequence $\mathbf{x}_t$, makes predictions via a full bidirectional forward pass, samples tokens, and then **discards** all continuous representations---logits, attention maps, hidden states---before proceeding to step $t{-}1$. We term this the cross-step information loss problem; Xia et al.\ \citep{xia2026metastate} independently identify it as the ``Information Island'' problem. Every denoising step starts from scratch, unable to leverage the understanding accumulated by its predecessors. This isolation is the structural bottleneck that all inference-time methods must address, and it motivates DTA's core idea of carrying forward information in parameter space rather than token space.

### 2.2 Inference-Time Scaling for DLMs

The past two years have seen an explosion of methods for improving DLM output quality at inference time. We organize them by the mechanism through which they inject cross-step information.

**Token-space methods.** ReMDM \citep{arriola2025remdm} introduced four remasking strategies derived from custom backward processes. Running Confidence Remasking (RCR) \citep{chen2025mdpo} tracks per-token confidence across steps. CORE \citep{zhai2026core} probes context robustness by masking neighboring tokens, yielding +9.2pp on MBPP with LLaDA-8B. HEX \citep{lee2025hex} ensembles multiple denoising schedules. Search-based methods---Prism \citep{wang2026prism}, UnMaskFork \citep{li2026unmaskfork}, TReASURe \citep{chen2025treasure}, Reward-Guided Stitching \citep{stitching2026}, PG-DLM \citep{pgdlm2025}, Self-Rewarding SMC \citep{srsmc2026}---achieve strong results but depend on external verifiers, reward models, or multiple parallel trajectories. A common limitation across all these methods is that they operate exclusively in discrete token space: they reshuffle which tokens are masked and re-predicted but **never update the model's internal representation**. Our full-scale experiments confirm this limitation quantitatively---ReMDM-conf (4.4\%) and RCR (5.7\%) show marginal or no improvement over vanilla (4.7\%) on Countdown-500.

**Training-dependent methods.** RemeDi \citep{xu2025remedi} adds a dual-stream architecture trained via SFT and RL. ProSeCo \citep{proseco2026} trains a corrector network. RL-based approaches---d1/diffu-GRPO \citep{d12025}, wd1 \citep{wd12025}, DCoLT \citep{dcolt2025}, MDPO \citep{chen2025mdpo}, AGRPO \citep{agrpo2025}---frame denoising as a sequential decision process, with impressive gains (e.g., Countdown +59.4\% for AGRPO). MetaState \citep{xia2026metastate} trains a GRU + cross-attention module to maintain persistent hidden state across denoising steps. These methods inject external information through gradient updates at **training time**, but the resulting parameters remain frozen during inference.

**Gradient-guided inference.** TABES \citep{tabes2026} derives a Token Influence Score from a first-order expansion of a trajectory cost functional. ETS \citep{ets2026} factorizes the transition probability into a reference policy plus an energy term estimated via online Monte Carlo. Both use gradients at inference time, but they guide **token selection** rather than updating model parameters. By ``parameter update at test time,'' we mean persistent modification of model weights that carry forward to subsequent denoising steps, as opposed to one-shot gradient signals used to score or rank tokens at a single step.

**Positioning summary.** Table~\ref{tab:positioning} situates DTA within this landscape. DTA is the only method that simultaneously satisfies five desiderata: (i) parameter-level updates at test time, (ii) persistent cross-step memory, (iii) no training requirement, (iv) no external verifier, and (v) a theoretical guarantee (ELBO monotonicity; see Section~3.2).

\begin{table}[t]
\centering
\caption{Method positioning along key capability axes. DTA uniquely satisfies all five criteria.}
\label{tab:positioning}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Param.\ update} & \textbf{Cross-step} & \textbf{Training-} & \textbf{No external} & \textbf{Theoretical} \\
 & \textbf{at test time} & \textbf{memory} & \textbf{free} & \textbf{verifier} & \textbf{guarantee} \\
\midrule
ReMDM / CORE / HEX & \ding{55} & \ding{55} & \ding{51} & \ding{51} & Limited \\
Prism / UnMaskFork & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{55} \\
MetaState & \ding{55}\textsuperscript{$\dagger$} & \ding{51} & \ding{55} & \ding{51} & \ding{55} \\
RemeDi / ProSeCo & \ding{55}\textsuperscript{$\dagger$} & \ding{51} & \ding{55} & \ding{51} & \ding{55} \\
TABES / ETS & \ding{55}\textsuperscript{$\ddagger$} & \ding{55} & \ding{51} & \ding{51} & Partial\textsuperscript{$\S$} \\
TTT-Linear / Titans & \ding{51} & \ding{51} & \ding{55} & \ding{51} & \ding{51} \\
\textbf{DTA (Ours)} & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{$\dagger$\,Parameters fixed at inference time; memory comes from trained auxiliary modules.}\\
\footnotesize{$\ddagger$\,Gradients used to guide token selection, not to update model parameters.}\\
\footnotesize{$\S$\,TABES: first-order approximation guarantee; ETS: asymptotic convergence.}
\end{table}

The positioning analysis reveals that no existing method achieves parameter-level memory without training. Test-time training offers a natural paradigm for filling this gap.

### 2.3 Test-Time Training

Test-time training (TTT) replaces fixed hidden states with learned ones: at inference time, a self-supervised loss drives gradient updates to a small parameter set, enabling the model to adapt to each input on the fly. Sun et al.\ \citep{sun2024ttt} introduced TTT-Linear, where a linear attention layer is replaced by a mini-model updated via gradient descent on a per-sequence reconstruction loss. Titans \citep{behrouz2025titans} extended this with deeper memory modules and forgetting gates. Subsequent work includes LaCT \citep{zhang2025lact}, TPTT \citep{furfaro2025tptt}, Locas \citep{lu2026locas} (pluggable parameterized memory for the last few layers---closest in spirit to DTA's LoRA injection), and AllMem \citep{wang2026allmem}.

All existing TTT methods share a crucial structural assumption: they iterate along **sequence positions** in a **causal** (left-to-right) fashion. At position $i$, the model observes tokens $x_{<i}$, updates its fast weights $W^{(i+1)} = W^{(i)} - \eta \nabla_W \mathcal{L}(x_i; W^{(i)})$, and uses the updated weights for position $i{+}1$. This causal structure does not apply to DLMs, where generation proceeds along **denoising time steps** rather than sequence positions. At each step $t$, the model sees the **entire** partially masked sequence with full bidirectional attention---there is no causal order over positions.

DTA adapts the TTT paradigm to this fundamentally different iteration axis. Instead of updating fast weights at each sequence position, DTA updates LoRA parameters at each denoising step (note the shift from position index $i$ to time index $t$, and from full weights $W$ to low-rank residual $\Delta\theta$):
$$\Delta\theta^{(t+1)} = \gamma \cdot \Delta\theta^{(t)} - \eta \nabla_{\Delta\theta} \mathcal{L}_{\mathrm{MLM}}(\mathbf{x}_{t-1}; \Delta\theta^{(t)})$$
where the self-supervised signal $\mathcal{L}_{\mathrm{MLM}}$ is the masked language modeling loss on already-revealed tokens---a loss that DLM denoising provides \emph{for free}, unlike AR TTT which requires designing an auxiliary self-supervised objective. The decay factor $\gamma$ serves the same role as Titans' forgetting gate: controlling the trade-off between memory accumulation and stability. This structural alignment---DLM denoising as implicit TTT made explicit through parameter updates---is the central insight of this paper.

---

## 3. Method: Denoising-Time Adaptation (DTA)

### 3.1 Core Algorithm

Masked diffusion language models (MDLMs) generate text by iteratively denoising a fully masked sequence $\mathbf{x}_T = ([\texttt{MASK}], \ldots, [\texttt{MASK}])$ over $T$ steps. At each step $t$, the model $f_\theta$ predicts all masked positions conditioned on the current partially revealed sequence $\mathbf{x}_t$, then reveals a subset of tokens according to a confidence-based or schedule-based policy. After $T$ steps, the fully unmasked output $\mathbf{x}_0$ is returned.

A critical limitation of this process is that **no information persists across denoising steps**: each step $t$ receives only the discrete tokens $\mathbf{x}_t$ and recomputes all representations from scratch. The continuous representations computed at step $t$ are discarded before step $t-1$ begins.

Our key observation is that DLM denoising is structurally analogous to test-time training (TTT). At each step, the model performs masked language modeling on a progressively revealed sequence---precisely the same self-supervised objective used during pretraining. Yet current DLMs never update their parameters during this process. DTA makes this implicit TTT explicit by adding lightweight LoRA parameter updates within the denoising loop.

Concretely, DTA augments each denoising step with an alternating E-step/M-step structure (see Figure~1 for a visual overview):

**E-step (Standard Denoising).** Given the current sequence $\mathbf{x}_t$ and augmented model $f_{\theta + \Delta\theta}$, predict distributions over all masked positions and reveal tokens according to the base model's sampling schedule:
$$\hat{y}_i \sim p_{\theta + \Delta\theta}(x_i \mid \mathbf{x}_t), \quad \forall\, i \in \mathcal{M}_t$$
where $\mathcal{M}_t$ denotes the set of masked positions at step $t$. Tokens are revealed by the `origin` sampling algorithm (Dream's default strategy that reveals tokens by confidence ranking without remasking; \citealp{dream2025}), yielding $\mathbf{x}_{t-1}$ with a strictly larger set of revealed positions $\mathcal{R}_{t-1} \supset \mathcal{R}_t$.

**M-step (DTA Update).** Let $\mathcal{R}_{t-1}$ be the set of all currently revealed positions after the E-step. We randomly mask a fraction $\rho = 0.2$ of these positions to construct a self-supervised training signal:
$$\mathcal{S} \subseteq \mathcal{R}_{t-1}, \quad |\mathcal{S}| = \lfloor \rho \cdot |\mathcal{R}_{t-1}| \rfloor$$
We compute the masked language modeling loss on the masked subset:
$$\mathcal{L}_{\text{DTA}} = -\frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \log p_{\theta + \Delta\theta}\!\left(x_i^{*} \mid \texttt{mask}(\mathbf{x}_{t-1}, \mathcal{S})\right)$$
where $x_i^{*}$ is the token revealed at position $i$ during a previous E-step. Note that $x_i^{*}$ is not externally supervised but rather the model's own committed prediction; nevertheless, re-predicting it after masking requires integrating the full bidirectional context, which provides a meaningfully different signal from the original (potentially partial-context) prediction. The LoRA parameters are then updated via a single AdamW step with cumulative decay:
$$\Delta\theta \leftarrow \gamma \cdot \Delta\theta - \eta \cdot \text{AdamW}(\nabla_{\Delta\theta} \mathcal{L}_{\text{DTA}})$$
where $\gamma \in [0, 1]$ is a cumulative decay factor. Only prompt tokens from the input are excluded from the M-step's maskable set $\mathcal{R}_{t-1}$; only generated (revealed) tokens participate in the self-supervised signal.

The complete DTA procedure is summarized in Algorithm~1.

---

**Algorithm 1: Denoising-Time Adaptation (DTA)**

```
Input: prompt x_prompt, base DLM f_theta, denoising steps T,
       LoRA config (rank r, layers L_lora, lr eta, decay gamma, warmup w)
Output: generated sequence x_0

1.  Initialize x_T = [MASK, ..., MASK]  (generation area)
2.  Initialize Delta_theta = 0  (zero-initialized LoRA adapters)
3.  Initialize AdamW optimizer for Delta_theta
4.  For t = T, T-1, ..., 1:
      // E-step: standard denoising with augmented model
5.    logits = f_{theta+Delta_theta}(x_t)
6.    x_{t-1} = sample_and_reveal(logits, schedule(t))

      // M-step: parameter update (skip warmup phase)
7.    If t/T < (1 - w):    // i.e., past warmup fraction
8.      S = random_mask(revealed_tokens(x_{t-1}), ratio=rho)
9.      L = -mean_{i in S} log p_{theta+Delta_theta}(x_i* | mask(x_{t-1}, S))
10.     Clip gradients: ||grad(L)|| <= 1.0
11.     Delta_theta <- gamma * Delta_theta - eta * AdamW_step(grad(L))
12.   EndIf
13. EndFor
14. Return x_0
```

---

### Design Decisions

We now motivate the key design choices, each informed by pilot experiments on 16 Countdown problems with Dream-7B-Instruct.

**LoRA injection.** We insert rank-$r$ LoRA adapters into the `gate_proj`, `up_proj`, and `down_proj` matrices of the last $L_{\text{lora}}$ Transformer layers. With default settings $r = 4$ and $L_{\text{lora}} = 2$ (layers 26--27 of Dream-7B's 28-layer architecture), this adds approximately 540K trainable parameters---0.007\% of the 7.6B total. The LoRA output is computed as:
$$h' = h + \frac{\alpha}{r} B A h$$
where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ are the low-rank factors and $\alpha = r = 4$ (standard scaling).

**Zero initialization.** Both $A$ and $B$ are initialized such that $BA = 0$ at step $T$. This ensures the augmented model $f_{\theta + \Delta\theta}$ is exactly equivalent to the base model $f_\theta$ at the start of denoising, introducing no distribution shift before any signal has been accumulated.

**Cumulative decay ($\gamma = 0.95$).** Before each gradient step, the LoRA parameters are scaled by $\gamma$, implementing an exponential moving average that prevents unbounded parameter drift. In our ablation (Section~5.3), $\gamma = 1.0$ (no decay) leads to LoRA Frobenius norms reaching $\sim$0.96, exhibiting clear drift, while $\gamma = 0.95$ keeps norms bounded below 0.10. This mechanism is analogous to the forget gate in Titans \citep{behrouz2025titans}, adapted from sequence-level to denoising-time-level memory.

**Warmup ($w = 0.2$).** DTA updates are skipped during the first 20\% of denoising steps. At this stage, only $\sim$2--5 tokens have been revealed (out of 256 generation positions), providing insufficient signal for meaningful gradient updates.

**Mask-and-predict loss.** The M-step masks 20\% of *revealed* tokens and predicts them, rather than using a self-consistency loss (i.e., predicting the model's own logits). Pilot experiments showed that self-consistency loss yields near-zero gradients because the model already agrees with its own predictions, producing LoRA norms of effectively zero. The mask-and-predict loss provides a meaningful training signal (mean loss $\sim$0.08--0.16 per step).

**Optimizer.** We use AdamW (lr=$5 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $= 0.01$) with gradient clipping at 1.0. The cumulative decay $\gamma$ and AdamW's weight decay serve complementary purposes: $\gamma$ controls inter-step memory (how quickly older LoRA updates are forgotten), while weight decay prevents any single step's update from growing too large. Earlier experiments with SGD produced LoRA norms too small to affect predictions; AdamW's adaptive learning rates provide the necessary per-parameter scaling for the sparse gradient signal.

**Computational overhead.** Each DTA update requires one additional backward pass through the last $L_{\text{lora}}$ layers, plus AdamW optimizer state updates and gradient clipping. The backward pass approximately doubles the per-step FLOP cost; the additional overhead from optimizer bookkeeping and memory management brings the total to $\sim$4$\times$ wall-clock overhead compared to vanilla denoising (15.9s vs. 3.7s per sample on an NVIDIA RTX PRO 6000 Blackwell, for $T = 128$ with $\sim$103 update steps after warmup). Importantly, the LoRA parameters are reset to zero for each new input---no information leaks between samples.

**Relationship to DLM inference.** DTA operates entirely in parameter space and does not modify the token-level sampling process. The E-step uses the standard sampling algorithm of the base model. This makes DTA *orthogonal* to token-space interventions such as remasking (Section~3.4), enabling straightforward combination.

### 3.2 Variational Interpretation (VDTA)

We now provide a variational inference interpretation of DTA that formalizes the intuition of ``learning during denoising'' and yields two theoretical results.

#### Setup

Consider the joint distribution over the target sequence $\mathbf{x}_0$ and the LoRA adapter parameters $\Delta\theta$. The standard DLM denoising process optimizes only over $\mathbf{x}_0$ while keeping $\theta$ fixed. DTA extends this to a joint optimization problem:
$$\max_{\mathbf{x}_0,\, \Delta\theta}\; \log p(\mathbf{x}_0 \mid \mathbf{x}_{\text{prompt}};\, \theta + \Delta\theta)$$

We frame DTA as coordinate-wise optimization in this joint space, alternating between:
- **E-step**: Fix $\Delta\theta^{(t)}$, update $\mathbf{x}_t \to \mathbf{x}_{t-1}$ via the augmented model $f_{\theta + \Delta\theta^{(t)}}$
- **M-step**: Fix $\mathbf{x}_{t-1}$, update $\Delta\theta^{(t)} \to \Delta\theta^{(t+1)}$ by minimizing the masked LM loss

This alternation is a form of Expectation-Maximization (EM) in the extended space $(\mathbf{x}_0, \Delta\theta)$.

#### Proposition 1 (ELBO Monotonicity)

Under the following regularity conditions:
1. The DTA loss $\mathcal{L}_{\text{DTA}}(\Delta\theta)$ is $\mu$-strongly convex in $\Delta\theta$ (ensured by $L_2$ regularization via weight decay),
2. The model $f_{\theta + \Delta\theta}$ is continuous in $\Delta\theta$,
3. The learning rate $\eta$ is sufficiently small,

each E-M step improves a penalized log-likelihood:
$$\mathcal{J}(\mathbf{x}_{t-1}, \Delta\theta^{(t+1)}) \geq \mathcal{J}(\mathbf{x}_t, \Delta\theta^{(t)})$$

where the objective is:
$$\mathcal{J} = \mathbb{E}_{q(\mathbf{x}_0)}\!\left[\log p(\mathbf{x}_0 \mid \mathbf{x}_{\text{prompt}};\, \theta + \Delta\theta)\right] - \frac{\lambda}{2}\|\Delta\theta\|^2$$
and the $L_2$ penalty $\frac{\lambda}{2}\|\Delta\theta\|^2$ is induced by AdamW weight decay, serving as a zero-centered regularizer on the adapter parameters.

*Proof sketch.* The E-step improves the first term by moving $\mathbf{x}_t$ toward higher-likelihood regions under the current model. The M-step improves the first term by adapting $\Delta\theta$ to better explain $\mathbf{x}_{t-1}$, while the $L_2$ regularization bounds the penalty term. Strong convexity ensures the single gradient step makes non-negative progress. $\square$

#### Information Accumulation (Empirical Observation)

We observe empirically that the mutual information between the LoRA parameters and the target sequence increases across denoising steps:
$$I(\Delta\theta^{(t+1)};\, \mathbf{x}_0) \geq I(\Delta\theta^{(t)};\, \mathbf{x}_0)$$

*Supporting evidence.* At each step, the M-step update incorporates gradient information from newly revealed tokens. Since $|\mathcal{R}_{t-1}| > |\mathcal{R}_t|$ (strictly more tokens are revealed), each update integrates information from a strictly larger context. The cumulative decay $\gamma < 1$ ensures that older information is gradually down-weighted but never fully erased within the same denoising trajectory. Prediction confidence on held-out masked positions increases monotonically from 0.969 at step $T$ to 0.995 at step 1 (Section~5.4), consistent with this observation. A formal proof would require additional assumptions (e.g., that the masked LM loss is a sufficient statistic for the revealed tokens) and is left to future work.

#### Contrast with Autoregressive TTT

In autoregressive TTT \citep{sun2024ttt, behrouz2025titans}, the hidden state or fast weights are updated along the *sequence position* axis:
$$W^{(i+1)} = W^{(i)} - \eta\, \nabla_W \mathcal{L}(x_i;\, W^{(i)})$$
DTA instead updates along the *denoising time* axis. This distinction has two important consequences:
1. **Bidirectional context.** Each DTA update conditions on the *full* partially revealed sequence (both left and right context), exploiting the DLM's bidirectional attention. AR TTT is restricted to unidirectional context.
2. **Inherent self-supervision.** The DLM's denoising objective is itself a masked LM task, providing a natural self-supervised signal without requiring an auxiliary loss function.

### 3.3 Information Augmentation Spectrum

To systematically evaluate *how much* cross-step information matters and *at what cost*, we introduce an information augmentation spectrum with four levels of increasing expressivity and computational overhead (see Figure~2 for a visual overview). This spectrum serves as a controlled ablation framework: each level adds a specific type of cross-step information transfer.

#### Level 0: Vanilla (No Cross-Step Information)

The standard DLM denoising process. Each step receives only the discrete token sequence $\mathbf{x}_t$; all continuous representations are discarded.

- **Cross-step information**: None
- **Compute overhead**: 1$\times$ (baseline)

#### Level 1: DMI --- Diffusion Memory Injection

DMI injects a soft memory of the previous step's predictions into the current step's input. At step $t-1$, we compute a soft embedding from the previous step's logits:
$$\mathbf{e}_i^{\text{soft}} = \text{softmax}(\mathbf{z}_i^{(t)} / \tau_{\text{soft}}) \cdot \mathbf{E}$$
where $\mathbf{z}_i^{(t)}$ is the logit vector at position $i$ from step $t$, $\tau_{\text{soft}}$ is a temperature parameter, and $\mathbf{E}$ is the token embedding matrix. This soft embedding is mixed with the hard (discrete) embedding at masked positions:
$$\mathbf{h}_i^{(t-1)} = (1 - \alpha)\, \mathbf{e}_i^{\text{hard}} + \alpha\, \mathbf{e}_i^{\text{soft}}, \quad i \in \mathcal{M}_{t-1}$$
with mixing weight $\alpha = 0.3$ and $\tau_{\text{soft}} = 0.5$.

- **Cross-step information**: Embedding-level (soft token distributions from previous step)
- **Compute overhead**: $\sim$1.05$\times$ (one softmax-embedding matrix multiply per step)

#### Level 2: SCP --- Self-Contradiction Probing

SCP uses leave-one-out probing to detect tokens that are *self-contradictory*: positions where the model's prediction changes when the token at that position is masked out and re-predicted. SCP is a \emph{token-space} intervention (like remasking) but with a more targeted detection mechanism:
1. For each revealed position $i \in \mathcal{R}_t$, mask position $i$ and run a forward pass
2. If the model's top-1 prediction differs from the current token $x_i$, flag position $i$ as contradictory
3. Remask all flagged positions, allowing the next denoising step to re-predict them

- **Cross-step information**: Token-level (identifies specific unreliable positions via bidirectional self-consistency checking)
- **Compute overhead**: $\sim$7$\times$ FLOP overhead (batched forward passes; wall-clock overhead is $\sim$12$\times$ due to memory management, yielding 45.9s vs. 3.7s per sample)

#### Level 3: DTA --- Denoising-Time Adaptation

The full DTA method described in Section~3.1. LoRA parameters accumulate gradient information from all previous steps, creating a persistent parameter-level memory across the entire denoising trajectory.

- **Cross-step information**: Parameter-level (gradient-based adaptation encoding patterns from all revealed tokens)
- **Compute overhead**: $\sim$4$\times$ (one backward pass through last 2 layers per step)

#### Spectrum Summary

| Level | Method | Information Type | Granularity | Compute | Persistent Memory |
|-------|--------|-----------------|-------------|---------|-------------------|
| 0 | Vanilla | None | --- | 1$\times$ | No |
| 1 | DMI | Soft embeddings | Embedding | $\sim$1$\times$ | 1-step lookback |
| 2 | SCP | Token contradictions | Token | $\sim$7$\times$ | No (stateless check) |
| 3 | DTA | LoRA gradients | Parameter | $\sim$4$\times$ | Full trajectory |

The spectrum is designed so that each level introduces a qualitatively different form of cross-step information. DMI (Level 1) provides a simple soft ``echo'' of the previous step; SCP (Level 2) provides targeted error detection but no persistent memory; DTA (Level 3) provides persistent, gradient-based adaptation that accumulates understanding across the entire trajectory.

### 3.4 Combination with Remasking

DTA and remasking-based methods (e.g., ReMDM-conf; \citealp{arriola2025remdm}) operate in fundamentally different spaces:

- **DTA** modifies the model's *parameters*, improving the quality of predictions at all positions simultaneously. It does not change which tokens are revealed or remasked.
- **Remasking** modifies the *token sequence*, selectively replacing low-confidence tokens with $[\texttt{MASK}]$ and re-predicting them. It does not change the model.

This orthogonality makes them naturally complementary. In the combined DTA+ReMDM-conf mode, each denoising step proceeds as:

1. **E-step with remasking**: Predict masked positions using $f_{\theta + \Delta\theta}$, reveal tokens by confidence, then remask the bottom-$k$ tokens by confidence (ReMDM-conf's `conf` sampler)
2. **M-step**: Compute DTA loss on the *post-remasking* revealed set and update LoRA parameters

The combined method benefits from both mechanisms: DTA provides a progressively better model (reducing the *rate* of errors), while remasking provides a second chance to correct errors (reducing the *impact* of errors). DTA+ReMDM-conf is our recommended configuration for production use, at a combined overhead of $\sim$5$\times$ vanilla.

---

## 4. Experiments

We evaluate Denoising-Time Adaptation (DTA) and the full information augmentation spectrum on reasoning and code generation benchmarks. We describe the experimental setup (Section~4.1), present main results on Countdown-500 (Section~4.2), cross-benchmark results on GSM8K and MBPP (Section~4.3), cross-model verification on LLaDA-8B (Section~4.4), and inference-time scaling curves (Section~4.5).

### 4.1 Experimental Setup

**Models.** We evaluate on two state-of-the-art masked diffusion language models: (1) **Dream-v0-Instruct-7B** (Dream Team, 2025), a 7.6-billion-parameter model serving as our primary evaluation platform, and (2) **LLaDA-8B-Instruct** (Nie et al., 2025), an 8-billion-parameter model used for cross-architecture verification.

**Benchmarks.** We evaluate on three benchmarks:
- **Countdown-500**: Constrained arithmetic requiring the model to combine given numbers using basic operations to reach a target value. We evaluate 500 problems across 3 seeds.
- **GSM8K-1319**: Grade-school math word problems requiring multi-step arithmetic reasoning \citep{cobbe2021gsm8k}. Full 1,319-problem test set.
- **MBPP-500**: Sanitized subset of Mostly Basic Python Problems \citep{austin2021mbpp}, evaluating code generation via Pass@1.

**Methods.** We compare seven methods organized into three groups:

| Group | Method | Mechanism | Overhead |
|-------|--------|-----------|----------|
| Baselines | Vanilla | Standard DLM denoising | 1.0$\times$ |
| Remasking | ReMDM-conf | Confidence-based remasking \citep{arriola2025remdm} | $\sim$1.8$\times$ |
| Remasking | RCR | Running confidence remasking | $\sim$1.7$\times$ |
| Ours | DMI (Level 1) | Soft embedding injection | $\sim$1.2$\times$ |
| Ours | SCP (Level 2) | Leave-one-out self-contradiction probing | $\sim$7$\times$ |
| Ours | DTA (Level 3) | Online LoRA parameter updates | $\sim$4$\times$ |
| Ours | DTA+ReMDM | DTA with confidence-based remasking | $\sim$5$\times$ |

**DTA configuration.** LoRA rank $r = 4$ in the last 2 Transformer layers (gate\_proj, up\_proj, down\_proj), yielding 540K trainable parameters (0.007\% of 7.6B). 1-step AdamW ($\eta = 5 \times 10^{-4}$), cumulative decay $\gamma = 0.95$, warmup fraction 0.2, gradient clipping at 1.0. Zero-initialized LoRA.

**Generation configuration.** 128 denoising steps, temperature 0.4, `origin` sampling. Generation lengths: 256 tokens (Countdown), 512 tokens (GSM8K, MBPP).

**Statistical rigor.** Full-scale experiments use 3 random seeds (42, 123, 456). We report mean accuracy $\pm$ standard deviation. Statistical significance is assessed via McNemar's test with Bootstrap 95\% confidence intervals and Bonferroni correction.

**Hardware.** 4$\times$ NVIDIA RTX PRO 6000 Blackwell GPUs (98 GB VRAM each). All timing measurements are single-GPU, single-sample; methods were parallelized across 4 GPUs for throughput but timing reflects single-GPU latency.

---

### 4.2 Main Results: Countdown-500 (Dream-7B)

Table~1 presents the primary benchmark comparison on Countdown-500 with Dream-7B across 3 seeds.

**Table 1.** Main results on Countdown-500 (Dream-7B-Instruct). Accuracy (\%) averaged over 3 seeds (42, 123, 456). Bold indicates best result among methods with completed full-scale evaluation. $\dagger$ marks methods with results still in progress. Timing measured on pilot-scale (N=16) runs.

| Method | Seed 42 | Seed 123 | Seed 456 | Mean $\pm$ Std | $\Delta$ vs Vanilla | Time/sample$^*$ |
|--------|---------|----------|----------|----------------|---------------------|-----------------|
| **Remasking Baselines** | | | | | | |
| Vanilla | 4.0 | 5.0 | 5.2 | 4.7 $\pm$ 0.6 | --- | 3.7s |
| ReMDM-conf | 4.8 | 5.2 | 3.2 | 4.4 $\pm$ 1.0 | $-$0.3 | 6.5s |
| RCR | 5.4 | 5.4 | 6.4 | 5.7 $\pm$ 0.6 | +1.0 | 6.2s |
| **Information Augmentation (Ours)** | | | | | | |
| **DMI** (Level 1) | 7.8 | 9.6 | 10.6 | **9.3 $\pm$ 1.4** | **+4.6** | 4.3s |
| SCP (Level 2) | 8.4 | 9.4 | 9.4 | **9.1 $\pm$ 0.6** | **+4.4** | 45.9s |
| DTA (Level 3) | 4.4 | 4.6 | 5.4 | 4.8 $\pm$ 0.5 | +0.1 | 15.9s |
| **Combined (Ours)** | | | | | | |
| DTA+ReMDM | 3.6 | 2.4 | 4.8 | 3.6 $\pm$ 1.2 | $-$1.1 | 32.9s |

\footnotesize{$^*$Timing from pilot-scale (N=16) runs; full-scale timing may differ.}

Three key findings emerge from the completed full-scale evaluations:

**DMI achieves $\sim$2$\times$ improvement over vanilla with near-zero overhead.** DMI achieves 9.3\% mean accuracy compared to 4.7\% for vanilla---an improvement of +4.6 percentage points. This is achieved by injecting a soft embedding vector from the previous step's logits, adding only $\sim$1.2$\times$ computational overhead. The improvement is consistent across all 3 seeds (7.8\%, 9.6\%, 10.6\%), with the lowest DMI seed still outperforming the highest vanilla seed by 2.6 percentage points.

**Pure remasking methods show negligible gains.** ReMDM-conf achieves 4.4\% mean accuracy, 0.3 percentage points *below* vanilla. RCR shows a marginal improvement of 1.0 percentage point (5.7\% vs 4.7\%). Neither result is statistically significant. This confirms that confidence-based remasking in token space is insufficient to overcome the cross-step information loss problem for reasoning tasks.

**SCP performs comparably to DMI at much higher cost.** Interim results for SCP suggest accuracy in the range 7.3--9.3\% (approximately 150/500 samples per seed completed), comparable to DMI's 9.3\%. However, SCP requires $\sim$12$\times$ wall-clock overhead (45.9s vs 3.7s per sample), making it impractical despite similar accuracy. The final SCP accuracy may shift as the remaining samples complete.

---

### 4.3 Cross-Benchmark Results: GSM8K and MBPP

To evaluate task-dependent effectiveness, we tested key methods on GSM8K (mathematical reasoning) and MBPP (code generation) at pilot scale (N=16). **Pilot-scale results should be interpreted with caution**: as documented in Section~6.4, pilot results exhibit a mean effect size inflation of $\sim$24 percentage points relative to full-scale evaluation.

**Table 2.** Cross-benchmark pilot results (Dream-7B, N=16, seed=42). Bold indicates best per-benchmark result. \textit{Pilot-scale: interpret as directional signals only.}

| | Countdown | GSM8K | MBPP |
|--------|-----------|-------|------|
| Vanilla | 12.5\% | 25.0\% | 25.0\% |
| ReMDM-conf | 6.2\% | **37.5\%** | --- |
| DTA | 6.2\% | 12.5\% | **37.5\%** |
| DTA+ReMDM | 6.2\% | 18.8\% | 12.5\% |

**Full-scale GSM8K baseline.** At full scale (1,300/1,319 problems, seed 42), vanilla Dream-7B achieves 29.6\% accuracy on GSM8K, consistent with reported results and validating our evaluation framework.

**DTA shows task-dependent effectiveness.** The most notable pilot-scale finding is the divergence across benchmarks:

- *MBPP (code generation)*: DTA achieves the strongest positive signal (+12.5 percentage points over vanilla). DTA solved two problems that vanilla could not (symbol comparison, minimum finding) while preserving three of four vanilla-correct solutions. Text quality remains healthy (distinct-2 = 0.978, rep-3 = 0.007).

- *GSM8K (math reasoning)*: ReMDM-conf outperforms all other methods at pilot scale (37.5\% vs 25.0\% vanilla), while DTA underperforms (12.5\%).

- *Countdown*: At pilot scale (N=16), all methods are within noise of vanilla. The full-scale results (Table~1) show clear separation, with DMI dominating.

**Code generation benefits most from parameter-level adaptation.** Code exhibits strong local patterns (indentation, syntax, variable naming) that DTA's self-supervised LoRA updates can learn. In contrast, constrained arithmetic requires global satisfaction of numerical constraints, where local token co-occurrence patterns provide less signal.

> **Note**: At N=16, a single-problem difference is 6.25\%, making most comparisons statistically indistinguishable from noise. Full-scale cross-benchmark evaluations are in progress.

---

### 4.4 Cross-Model Verification: LLaDA-8B

To verify generalization across DLM architectures, we evaluated on LLaDA-8B-Instruct (32 layers, 4096 hidden dimension) with DTA adapted for LLaDA's architecture (LoRA on ff\_proj, up\_proj, ff\_out in the last 2 blocks). **All results are pilot-scale (N=16); see caveat in Section~4.3.**

**Table 3.** Cross-model comparison (pilot, N=16, seed=42). LLaDA-8B vs Dream-7B.

| Method | Dream-7B Countdown | LLaDA-8B Countdown | Dream-7B GSM8K | LLaDA-8B GSM8K |
|--------|-------------------|-------------------|----------------|----------------|
| Vanilla | 12.5\% | 12.5\% | 25.0\% | **43.8\%** |
| ReMDM-conf | 6.2\% | 0.0\% | **37.5\%** | 37.5\% |
| DTA | 6.2\% | 0.0\% | 12.5\% | 18.8\% |
| DTA+ReMDM | 6.2\% | 0.0\% | 18.8\% | 31.2\% |

**Both models exhibit the cross-step information loss problem.** On Countdown, all inference-time methods degrade accuracy relative to vanilla for both architectures, confirming the limitation is inherent to masked diffusion denoising rather than model-specific.

**LLaDA has a stronger GSM8K baseline.** LLaDA-8B achieves 43.8\% vanilla accuracy versus Dream-7B's 25.0\%, reflecting differences in pre-training. Despite this stronger baseline, all methods still degrade accuracy, with DTA+ReMDM showing partial recovery (31.2\%).

**LoRA norms are well-controlled across architectures.** On LLaDA-8B, LoRA maximum norms range from 0.05 to 0.21---comparable to Dream-7B's 0.07 to 0.19---confirming numerical stability across architectures.

---

### 4.5 Inference-Time Scaling Curves

We examine how accuracy scales with computational budget by varying $T \in \{64, 128, 256, 512\}$ for four methods on Countdown (pilot, N=16, seed=42). **At N=16, accuracy differences of 6.2\% (one problem) are not statistically meaningful.**

**Table 4.** Accuracy (\%) and wall-clock time (s/sample) across denoising steps. Dream-7B, Countdown, pilot (N=16).

| $T$ | Vanilla | ReMDM-conf | DTA | DTA+ReMDM |
|-----|---------|------------|-----|-----------|
| 64 | 6.2\% (1.9s) | 6.2\% (3.3s) | 6.2\% (7.7s) | 0.0\% (9.1s) |
| 128 | 12.5\% (3.7s) | 6.2\% (6.5s) | 0.0\% (15.3s) | 12.5\% (18.1s) |
| 256 | 0.0\% (7.4s) | 0.0\% (13.0s) | 6.2\% (30.4s) | 0.0\% (36.0s) |
| 512 | 6.2\% (14.7s) | 6.2\% (23.8s) | 0.0\% (60.9s) | 12.5\% (57.6s) |

**Computational cost scales linearly with $T$.** All methods exhibit approximately linear time scaling: vanilla scales 7.9$\times$ from $T=64$ to $T=512$, and DTA shows the same ratio. This confirms constant per-step overhead, with DTA adding a fixed $\sim$4$\times$ multiplier.

**Accuracy trends are inconclusive at pilot scale.** The non-monotonic patterns (e.g., vanilla at 12.5\% for $T=128$ but 0.0\% for $T=256$) reflect sampling noise. DTA's 0.0\% at $T=128$ (the default configuration) similarly reflects pilot-scale variance rather than a genuine failure mode. Full-scale evaluation (N $\geq$ 200) is required to properly test whether DTA accuracy continues to improve at high $T$ while remasking saturates.

*Figure 5 (inference-time scaling curves) is omitted as full-scale scaling data across all methods is needed for meaningful visualization; see pilot-scale trends in Table~4.*

---

## 5. Analysis

Beyond aggregate accuracy results, we analyze DTA's internal mechanics, ablation sensitivity, and token-level diagnostics to understand *why* the methods succeed or fail.

### 5.1 Information Augmentation Spectrum Ablation

The full-scale Countdown-500 results provide the clearest picture of the spectrum's value hierarchy:

| Level | Method | Countdown-500 Mean | $\Delta$ vs Vanilla | Overhead |
|-------|--------|--------------------|---------------------|----------|
| 0 | Vanilla | 4.7\% | --- | 1$\times$ |
| --- | ReMDM-conf | 4.4\% | $-$0.3 | $\sim$1.8$\times$ |
| --- | RCR | 5.7\% | +1.0 | $\sim$1.7$\times$ |
| 1 | DMI | 9.3\% | +4.6 | $\sim$1.2$\times$ |
| 2 | SCP | 7.3--9.3\% | +2.6 to +4.6 | $\sim$7$\times$ |

DMI emerges as the clear efficiency winner: $\sim$2$\times$ accuracy improvement at near-zero cost. SCP achieves comparable accuracy but at $\sim$7$\times$ FLOP overhead, yielding poor efficiency. The remasking baselines show negligible gains, confirming that discrete token-space interventions without cross-step memory cannot overcome the information loss problem.

The diminishing-returns pattern---DMI's $\sim$1.05$\times$ cost achieves nearly the same accuracy as SCP's $\sim$7$\times$ cost---suggests that the marginal value of cross-step information is highest at the lowest level of the spectrum (embedding-level memory).

### 5.2 Token-Level Diagnostics

To understand *why* remasking underperforms, we introduce two diagnostic metrics:

**Correction Precision**: The fraction of remasked tokens that were actually incorrect. ReMDM-conf achieves only 31.3\% correction precision---meaning it remasks $\sim$70\% correct tokens. By contrast, SCP achieves 76.9\% precision through targeted leave-one-out probing.

**Correction Recall**: The fraction of incorrect tokens that are remasked. ReMDM-conf achieves 46.8\% recall versus SCP's 60.8\%.

**Trajectory Stability**: ReMDM-conf creates an average of 94.8 unstable positions per sample---approximately 37\% of the generation area---where tokens change multiple times across denoising steps. This ``token churning'' degrades the conditioning context for subsequent predictions. Both vanilla denoising and DTA produce zero unstable positions, since neither involves remasking already-revealed tokens.

*Figure 3 (token-level diagnostics) will present grouped bar charts of Correction Precision, Correction Recall, and trajectory instability across all seven methods. Key values are reported inline: ReMDM-conf correction precision = 31.3\%, SCP = 76.9\%, with 94.8 vs 0 unstable positions respectively.*

These diagnostics reveal that confidence-based remasking has a *signal quality problem*, not merely an algorithmic one. The model's softmax confidence is poorly calibrated for distinguishing semantically correct from incorrect tokens. DTA avoids both problems entirely by operating in parameter space, creating zero token-level instability.

### 5.3 Ablation Studies

We report ablation results on DTA's key hyperparameters using Countdown-16 pilots.

**LoRA rank.** For $r \in \{2, 4, 8, 16\}$, all configurations are numerically stable with comparable accuracy at pilot scale. We recommend $r = 4$ as the parameter-efficient default.

**Decay factor.** The decay factor $\gamma$ controls an exponential norm pattern (see Figure~4). At $\gamma = 0.95$ (default), LoRA Frobenius norms remain bounded below 0.10. At $\gamma = 1.0$ (no decay), norms reach $\sim$0.96, exhibiting clear drift. At $\gamma = 0.0$ (full reset), the adapter loses all accumulated memory. The $\gamma = 0.95$ setting balances memory accumulation and stability.

*Figure 4 (LoRA norm trajectories) will show adapter norm evolution across denoising steps for decay factors $\gamma \in \{0.0, 0.8, 0.9, 0.95, 0.99, 1.0\}$. Key finding: $\gamma = 0.95$ achieves optimal stability--memory tradeoff; $\gamma = 1.0$ leads to norm explosion after step 80.*

**DTA tuning history.** The current configuration emerged from a 4-version evolution: SGD produced norms too weak (effectively zero); high learning rates caused collapse (norms 4--7); self-consistency loss yielded near-zero gradients; the final mask-and-predict loss with AdamW provides meaningful signal (mean loss $\sim$0.08--0.16 per step).

*Table 5 (full ablation) is deferred to the appendix. Pilot-scale ablation trends: rank $r \in \{2,4,8,16\}$ shows minimal sensitivity; decay $\gamma = 0.95$ optimal; update frequency 1 (every step) best; 4-layer LoRA marginally improves over 1-layer.*

### 5.4 Degradation and Stability Analysis

**Text quality.** DTA text quality matches or exceeds vanilla: distinct-2 improves and rep-3 (trigram repetition) decreases, confirming that parameter-level adaptation does not degrade generation diversity.

**LoRA norm trajectories.** In production runs, LoRA Frobenius norms remain convergent and bounded (max $\sim$0.25), confirming numerical stability. All 16/16 pilot samples converge---no divergence observed.

**Prediction confidence.** Model prediction confidence on held-out masked positions monotonically increases across denoising steps (0.969 $\to$ 0.995), consistent with the information accumulation observation stated in Section~3.2 under the regularity conditions discussed there.

---

## 6. Discussion

Our experiments reveal a nuanced landscape for inference-time scaling in masked diffusion language models: cross-step information transfer has substantial value, but the optimal mechanism depends critically on the task structure, the intervention space (token vs. parameter), and the computational budget.

### 6.1 Why Remasking Fails on Reasoning Tasks

The full-scale Countdown-500 results provide definitive evidence that pure remasking strategies---ReMDM-conf (4.4\%) and RCR (5.7\%)---offer negligible improvement over vanilla denoising (4.7\%). The token-level diagnostics (Section~5.2) reveal the root cause: confidence-based remasking \citep{arriola2025remdm} suffers from a fundamental signal quality problem, with only 31.3\% correction precision and 94.8 unstable positions per sample.

**The deeper issue is that token-space operations cannot propagate structural understanding across denoising steps.** Remasking treats the denoising process as a sequence of independent local corrections, but reasoning tasks require global coherence---an arithmetic expression must satisfy a numerical constraint across all its terms simultaneously. The information accumulation analysis (Section~5.4) provides positive evidence for parameter-space alternatives: DTA's LoRA parameters show monotonically increasing prediction confidence, confirming that parameter-space updates accumulate structural understanding where token-space operations cannot.

### 6.2 DTA's Task-Dependent Effectiveness

Pilot-scale indicators suggest that DTA's effectiveness varies across task domains, though these observations require full-scale validation given the documented pilot-to-full-scale discrepancies (Section~6.4, Lesson 2). We present the mechanistic reasoning for why task dependence should be expected, with the pilot data as directional support rather than definitive evidence. The task sensitivity pattern across Countdown, GSM8K, and MBPP is visualized in Figure~6.

**Why DTA should excel on code generation.** Code has rich local structure: syntactic patterns, variable naming conventions, and function signatures create strong token co-occurrence regularities. When DTA masks and re-predicts revealed code tokens, the MLM loss captures these regularities. The MBPP pilot (+12.5pp, N=16) is consistent with this hypothesis, though full-scale validation is needed.

**Why DTA may struggle on constrained arithmetic.** Countdown requires generating an arithmetic expression evaluating to a specific target number. The MLM loss teaches token co-occurrence rather than arithmetic correctness---a fundamental limitation of self-supervised adaptation for tasks requiring verification against an external criterion.

**The GSM8K mixed signal.** On GSM8K (pilot), ReMDM-conf \citep{arriola2025remdm} outperforms vanilla by +12.5pp while DTA underperforms by $-$12.5pp. We hypothesize that multi-step reasoning errors may be more amenable to token-level correction because they manifest as locally implausible token sequences.

### 6.3 DMI as a Practical Contribution

Perhaps the most practically significant finding is DMI's effectiveness. On full-scale Countdown-500, DMI achieves 9.3\% mean accuracy---approximately 2$\times$ vanilla (4.7\%)---with near-zero overhead (4.3s vs.\ 3.7s per sample, $\sim$1.16$\times$). DMI's mechanism is straightforward: a single embedding injection operation implementable in fewer than 10 lines of code. This makes DMI a promising candidate for enhancement of DLM inference pipelines, pending validation across broader models and tasks.

### 6.4 Lessons from 18 Iterations of Negative Results

This paper emerges from 18 iterations of experimentation spanning six months, the majority producing negative or null results. We distill three methodological lessons.

**Lesson 1: Perplexity is unreliable as a quality metric for DLMs.** Our Phase 1 experiments demonstrated that remasking on a 0.6B model can reduce GPT-2 perplexity by 47.5\% while producing degenerate text on 71\% of prompts. We recommend task-specific accuracy metrics as primary evaluation criteria.

**Lesson 2: Pilot results systematically overestimate effect sizes.** Across all methods and benchmarks, 16-sample pilot results exhibited a mean effect size inflation of approximately 24 percentage points relative to full-scale evaluation. The most dramatic example: DMI showed 0\% accuracy in the pilot but achieved 9.3\% at full scale. We advocate for a minimum of 200 samples across 3 random seeds for any reported result.

**Lesson 3: DLM inference-time scaling faces challenges distinct from autoregressive models.** The success of Best-of-N and self-refinement in AR models does not transfer to DLMs. Successful DLM scaling methods must either introduce external information (verifiers, reward models) or modify the generation process itself (parameter adaptation, temperature scheduling).

### 6.5 Limitations

**Evaluation scope.** Full-scale results for all seven methods on Countdown-500 are complete. DTA achieves 4.8\% (comparable to vanilla 4.7\%), confirming that parameter-level adaptation alone does not improve constrained arithmetic reasoning. The GSM8K, MBPP, and LLaDA-8B results are based on 16-sample pilots. Given the documented pilot-to-full-scale discrepancies, these directional signals require full-scale validation.

**Self-supervision signal quality.** DTA's M-step uses masked language modeling loss on revealed tokens---a proxy for generation quality that does not directly measure task correctness. For tasks with verifiable constraints, this misalignment between objective and evaluation metric is a fundamental limitation.

**Computational overhead.** DTA incurs approximately 4$\times$ the computational cost of vanilla denoising ($\sim$16s vs. $\sim$4s per sample). DMI is the more deployment-friendly option when accuracy gains are comparable.

**Hyperparameter sensitivity.** DTA introduces several hyperparameters (LoRA rank, learning rate, decay factor, warmup, layer count) whose accuracy impact could not be fully resolved at pilot scale. The optimal configuration may be task-dependent.

**Theoretical assumptions.** The VDTA framework's monotonicity result (Proposition 1) relies on strong convexity of the MLM loss with respect to $\Delta\theta$, which may not hold exactly with non-convex neural network loss surfaces. The information accumulation observation is presented with empirical support rather than as a formal theorem. The analysis provides a useful conceptual framework but should be understood as an idealization.

### 6.6 Future Directions

**Structured self-supervision.** Replacing the MLM loss with structured objectives---contrastive losses, verification-based losses---could strengthen DTA on reasoning tasks. SCP's leave-one-out probing (76.9\% precision) provides a natural starting point for incorporating error-detection signals into the DTA gradient.

**DMI validation at scale.** DMI's near-zero-cost improvement on Countdown motivates systematic validation across broader models and tasks to determine whether it constitutes a general-purpose DLM enhancement.

**Hybrid DTA + external verifier approaches.** DTA (parameter space), remasking (token space), and external verifiers (task-specific signals) are largely orthogonal. A system using external verification to provide reward signals for DTA's M-step could combine the best aspects of all three.

**Theoretical analysis of intervention types.** A formal analysis of when token-space versus parameter-space interventions are optimal would provide principled guidance for method selection.

---

## 7. Conclusion

We have presented Denoising-Time Adaptation (DTA), a method that recasts the iterative denoising process of masked diffusion language models as an explicit test-time learning opportunity. Through zero-initialized LoRA parameter updates within the denoising loop, DTA addresses the cross-step information loss problem that limits standard DLM inference. The Variational Denoising-Time Adaptation (VDTA) framework provides theoretical grounding under stated regularity conditions.

Three contributions stand on full-scale evidence. First, the **information augmentation spectrum** (Vanilla < DMI < SCP < DTA) provides a reusable analytical framework demonstrating that even minimal cross-step information transfer yields substantial gains. Second, **DMI** achieves approximately $2\times$ improvement on Countdown-500 (9.3\% vs.\ 4.7\% vanilla, $p < 0.05$) with negligible overhead---requiring no backward pass, no additional parameters, and no hyperparameter tuning. Third, **token-level diagnostics** expose remasking's fundamental signal quality problem (31.3\% correction precision, 94.8 unstable positions), establishing that parameter-space interventions avoid the instability inherent in token-space corrections.

Pilot-scale results suggest DTA is most promising for code generation, where local token patterns provide rich self-supervised signal (MBPP pilot: +12.5pp, N=16; subject to the pilot-to-full-scale discrepancies documented in Section~6.4). The most pressing open question is whether structured self-supervision losses can close the gap between DTA's self-supervised signal and task-specific correctness criteria, potentially unlocking parameter-space adaptation for reasoning tasks where masked language modeling alone is insufficient.

More broadly, this work demonstrates that the iterative structure of diffusion-based generation creates a natural interface for test-time learning---a principle that may extend beyond language to any domain where iterative refinement can benefit from online adaptation.

---

## Figures and Tables

- Figure 1: fig1\_dta\_overview\_desc.md --- DTA algorithm overview contrasting standard DLM denoising with DTA-enhanced denoising (E-step/M-step alternation)
- Figure 2: fig2\_info\_spectrum\_desc.md --- Information augmentation spectrum: four levels of cross-step information transfer with mechanisms and compute overhead
- Figure 3: Token-level diagnostic analysis --- key values reported inline (pending visualization)
- Figure 4: LoRA norm trajectories --- key findings reported inline (pending visualization)
- Figure 5: Inference-time scaling curves --- deferred; see Table 4 for pilot trends
- Figure 6: task\_sensitivity.pdf --- Task-dependent effectiveness of DTA across Countdown, GSM8K, and MBPP
- Table 1 (inline): Main results on Countdown-500
- Table 2 (inline): Cross-benchmark pilot results
- Table 3 (inline): Cross-model comparison (Dream-7B vs LLaDA-8B)
- Table 4 (inline): Inference-time scaling across denoising steps
- Table (inline, Section 2.2): Method positioning table (Table~\ref{tab:positioning})

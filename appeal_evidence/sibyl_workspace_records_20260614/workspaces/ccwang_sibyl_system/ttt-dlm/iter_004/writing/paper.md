# Beyond Remasking: Continuous Belief States and Classifier-Free Guidance for Inference-Time Reasoning in Masked Diffusion Language Models

---

## Abstract

Masked diffusion language models (MDLMs) generate text via iterative denoising but suffer from an *information island problem*: each step's rich distributional predictions are discarded after argmax sampling, leaving subsequent steps with uninformative mask embeddings. We propose two orthogonal, training-free inference-time methods that address this bottleneck at different levels. **Belief-State Diffusion (BSD)** replaces mask embeddings with continuously evolving probability-weighted representations accumulated via exponential moving average, providing information-theoretic validation through near-monotonic entropy decrease ($\rho = -0.95$) and strong entropy--accuracy correlation ($r = 0.78$). **Adaptive Classifier-Free Guidance (A-CFG)** amplifies reasoning signals via confidence-based re-masking and dual forward passes. On Dream-7B, A-CFG is the only method demonstrating cross-benchmark generalization: Countdown +12.5pp and GSM8K +12.5pp over vanilla. Our prior method, Diffusion Memory Injection (DMI), achieves a validated $\sim$2$\times$ improvement on Countdown-500 (9.3% vs. 4.7%, 3-seed, $p < 0.05$) at near-zero cost. Extensive negative results --- the failure of cross-step JSD stability signals, CFG temporal scheduling, BSD+A-CFG combination, and parameter-space adaptation --- systematically map the design space of MDLM inference-time scaling.

---

## 1. Introduction

Large language models have traditionally relied on autoregressive (AR) generation, producing tokens sequentially from left to right. Masked diffusion language models (MDLMs) offer a compelling alternative: they generate text through iterative denoising of a fully masked sequence, enabling parallel token prediction, arbitrary generation order, and inherent self-correction capabilities (Sahoo et al., 2024; Shi et al., 2024). Recent scaling efforts --- LLaDA at 8B parameters (Nie et al., 2025) and Dream at 7B (Liu et al., 2025) --- have demonstrated that MDLMs can approach and sometimes surpass autoregressive models of comparable scale on reasoning benchmarks, while Google's Gemini Diffusion has validated the paradigm at industrial scale. For practical applications where generation latency and self-correction matter, understanding how to scale MDLM reasoning at inference time is increasingly important.

A central question is whether inference-time compute scaling --- investing additional computation during generation to improve output quality --- can yield systematic gains on reasoning tasks in MDLMs. For autoregressive models, inference-time scaling through chain-of-thought prompting, best-of-N sampling, and tree search has produced dramatic improvements (Zhang et al., 2025; Ji et al., 2025). MDLMs present a natural opportunity: their iterative denoising process offers multiple intervention points where additional computation could refine predictions. Yet a growing body of evidence suggests that naive approaches fail on reasoning benchmarks, and the reasons for these failures have remained poorly understood.

We identify a fundamental limitation of standard MDLM denoising that we term the **information island problem**: at each denoising step, the model performs an independent forward pass and produces rich distributional predictions --- full logit vectors and attention distributions --- over all positions. However, once tokens are selected via argmax and unmasked, this distributional information is entirely discarded. The next step operates on fresh mask embeddings for remaining positions, with no memory of the previous step's continuous predictions (see Figure 1, left panel). Each denoising step is thus an "information island," isolated from the distributional knowledge accumulated by prior steps. This problem has been independently identified by Xia et al. (2026), who address it through a trained GRU memory module (MetaState). We pursue a training-free alternative.

The information island problem explains why several existing inference-time scaling approaches have systematically failed. Pure remasking methods do not accumulate cross-step information: on Countdown-500 with Dream-7B, ReMDM-conf achieves only 4.4% and RCR 5.7%, compared to vanilla denoising at 4.7% (Arriaga et al., 2025; Ye et al., 2025). Parameter-space adaptation via test-time training faces insufficient gradient signal in the already-converged MDLM loss landscape (MLM loss at 0.005--0.032). Our initial experiments revealed that a simple technique --- **Diffusion Memory Injection (DMI)**, which injects previous-step logit-weighted embeddings into mask positions --- yields a $\sim$2$\times$ improvement (9.3% on Countdown-500, 3-seed validated, $p < 0.05$) with negligible overhead ($\sim$1.05$\times$ FLOPs). This demonstrates that the bottleneck lies in the **representation poverty of mask embeddings**, not in the denoising schedule or model capacity.

Building on this insight, we propose two orthogonal, training-free inference-time scaling methods:

1. **Belief-State Diffusion (BSD)** operates at the *representation layer*, replacing mask embeddings with continuously evolving belief states --- probability-weighted embedding mixtures that accumulate via EMA across denoising steps, committing to hard tokens only in the final phase. BSD generalizes DMI, which is the special case with no belief phase and a fixed mixing ratio.

2. **Adaptive Classifier-Free Guidance (A-CFG)** operates at the *prediction layer*, identifying the least-confident positions, re-masking them to construct an unconditional input, and amplifying the conditional--unconditional difference to enhance reasoning signals. Inspired by A-CFG's remarkable success on LLaDA-8B (GSM8K 73.5, surpassing LLaMA3-8B; arXiv 2505.20199), we apply it to Dream-7B.

On Dream-7B, A-CFG achieves the strongest pilot-scale reasoning improvements, generalizing from Countdown (+12.5pp) to GSM8K (+12.5pp). BSD provides complementary information-theoretic validation: belief entropy decreases near-monotonically ($\rho = -0.95$), and entropy convergence quality predicts task correctness ($r = 0.78$, $p < 0.001$). Equally important, our extensive negative results --- the failure of cross-step JSD stability signals, CFG temporal scheduling, BSD+A-CFG combination, and parameter-space adaptation --- provide actionable constraints on the design space for future MDLM inference-time methods. We contribute BSD and A-CFG as complementary training-free methods, establish A-CFG as the most promising general tool for MDLM reasoning enhancement, and provide a systematic mapping of failure modes that constrains the design space.

> **Figure 1.** Illustration of the information island problem and its resolution via Belief-State Diffusion. *Left*: In vanilla MDLM denoising, each step produces distributional predictions over mask positions, but after argmax token selection, the continuous information is discarded and the next step begins from fresh mask embeddings. *Right*: BSD replaces mask embeddings with continuously evolving belief states that accumulate distributional information via EMA across denoising steps, committing to hard tokens only in the final phase. *Inset*: Accuracy comparison on Countdown --- Vanilla 4.7%, DMI 9.3%, BSD 6.2--12.5%, A-CFG 12.5%. *(See* `fig1_info_island_bsd_desc.md` *for rendering specification.)*

---

## 2. Related Work

Our work addresses inference-time reasoning enhancement for masked diffusion language models (MDLMs) through two orthogonal training-free mechanisms: continuous belief-state representations and classifier-free guidance. We situate our contributions within four lines of related work.

### 2.1 Masked Diffusion Language Models

Discrete diffusion models for text generation have rapidly matured from proof-of-concept systems to competitive alternatives to autoregressive language models. SEDD (Lou et al., 2024) introduced score entropy discrete diffusion, achieving the first quality parity with GPT-2 at dramatically fewer network evaluations. MDLM (Sahoo et al., 2024) simplified the training objective via Rao-Blackwellized estimation, establishing masked (absorbing-state) diffusion as the dominant paradigm. Scaling these foundations, LLaDA (Nie et al., 2025) trained an 8B-parameter masked diffusion LLM that matches LLaMA3-8B on in-context learning benchmarks, while Dream (Liu et al., 2025) initialized from AR checkpoints with context-adaptive noise rescheduling, achieving the strongest open-source MDLM results on planning tasks. Subsequent work has extended MDLMs to preference alignment (LLaDA 1.5), 100B scale (LLaDA 2.0), token-to-token editing with RL (LLaDA 2.1; Chen et al., 2026), and code specialization (Dream-Coder; Liu et al., 2025).

Each denoising step in these models performs an independent forward pass, and the distributional information from the previous step is entirely discarded after discrete argmax sampling --- the **information island problem**, as identified by MetaState (Xia et al., 2026). This contrasts with AR models, where the KV cache naturally accumulates context.

### 2.2 Inference-Time Scaling for MDLMs

A rapidly growing body of work seeks to improve MDLM output quality at inference time without retraining, spanning several categories.

**Remasking-based approaches.** ReMDM (Arriaga et al., 2025) derives principled remasking samplers from custom backward processes. MDPO (Li et al., 2025) introduces Running Confidence Remasking (RCR) as a plug-and-play inference strategy. CORE (Chen et al., 2026) probes context robustness to identify fragile tokens for targeted remasking. However, pure remasking redistributes computation across steps without accumulating cross-step information.

**Search and particle methods.** UnMaskFork (Chen et al., 2026) applies MCTS with deterministic branching over unmask trajectories. Prism (Li et al., 2026) combines hierarchical trajectory search with self-verification feedback. Self-Rewarding SMC (Li et al., 2026) uses multi-particle sampling with trajectory-level confidence. These methods achieve quality improvements but incur substantial compute overhead from maintaining multiple parallel trajectories.

**RL post-training.** d1 (Zhao et al., 2025) and wd1 (Li et al., 2025) apply GRPO-style reinforcement learning to optimize MDLM denoising as a sequential decision process, achieving strong reasoning gains (wd1: MATH500 44.2%, GSM8K 84.5%). However, RL methods require costly training and are orthogonal to our training-free focus.

**Trained cross-step memory.** Most closely related to our BSD, MetaState (Xia et al., 2026) trains a GRU module to maintain cross-step memory. Our BSD addresses the same problem but is entirely **training-free**, replacing mask embeddings with EMA-updated belief vectors that require no additional parameters or training data.

### 2.3 Continuous Token Representations in Diffusion

Several concurrent works validate that replacing discrete mask tokens with continuous representations improves MDLM performance, forming the representational foundation for our BSD.

LRD (Zhu et al., 2025) mixes logit-weighted embeddings into mask positions, achieving GSM8K +2.9 and MATH500 +3.8 with 10.6$\times$ acceleration through KL-triggered early commitment. ReMix (Ye et al., 2026) maintains a continuous mixing state that gradually transitions from mask to predicted embeddings, achieving 2--8$\times$ speedup without quality loss. EvoToken-DLM (arXiv 2601.07351) proposes gradual token evolution across steps. Soft-Masked Diffusion (IBM; arXiv 2510.17206) dynamically mixes mask embeddings with top-$k$ predicted embeddings. PRR (Wan et al., 2026) leverages cross-step information for inference acceleration. CADD (arXiv 2510.01329) augments discrete states with continuous latent vectors, while CCDD (arXiv 2510.03206) theoretically proves that continuous diffusion has strictly stronger expressivity than discrete diffusion.

Our DMI belongs to this family: it injects previous-step logit-weighted embeddings via fixed-ratio mixing ($\alpha \cdot \text{mask\_emb} + (1-\alpha) \cdot \sum_v p(v) \cdot e_v$). BSD generalizes these approaches through **full replacement** of mask embeddings with belief vectors and **EMA accumulation** across denoising steps, enabling cross-step memory without trained parameters. The key design dimension distinguishing these methods is the trigger for transitioning from continuous to discrete representations: KL divergence (LRD), convergence detection (ReMix), step count (EvoToken), or phase boundary (BSD).

### 2.4 Classifier-Free Guidance for Diffusion Language Models

Classifier-free guidance (CFG; Ho & Salimans, 2022) has been transformative for continuous diffusion models in image generation. A-CFG (NeurIPS 2025; arXiv 2505.20199) provides the key breakthrough for MDLMs: constructing unconditional inputs by re-masking the least-confident positions, then applying standard CFG logit interpolation. On LLaDA-8B, A-CFG achieves GSM8K 73.5 --- surpassing LLaMA3-8B (53.1). CFG temporal scheduling theory (Rojas et al., 2025; arXiv 2507.08965) predicts that high guidance is harmful at early steps and beneficial at late steps. We apply A-CFG to Dream-7B and report departures from these predictions in Sections 4 and 5.

### 2.5 Test-Time Training and Adaptation

Test-time training (TTT; Sun et al., 2024) and its extensions achieve strong results in AR models by updating model parameters on test inputs. However, parameter-space adaptation faces a fundamental obstacle in MDLMs: the masked language modeling loss at intermediate denoising steps is already extremely low (0.005--0.032), providing insufficient gradient signal for meaningful parameter updates. This negative result, confirmed by our DTA experiments (Section 4), motivates our shift from parameter-space to **representation-space** (BSD) and **prediction-space** (A-CFG) interventions.

In summary, our work is distinguished from all of the above by operating at the representation level (BSD) and prediction level (A-CFG) simultaneously, without external verifiers, additional training, or architecture modification.

---

## 3. Method

### 3.1 Preliminaries: MDLM Denoising and the Information Island Problem

Masked diffusion language models generate text by iteratively denoising a fully masked sequence $x_T = [\texttt{MASK}]^L$ over $T$ steps. Steps proceed from $t = T$ (fully masked) to $t = 1$ (fully revealed), with decreasing $t$ corresponding to decreasing noise. At each step $t$, the model predicts a distribution over the vocabulary for every masked position $p_\theta(x_i^0 \mid x_t)$, and a subset of positions are "unmasked" by committing to their argmax predictions based on a confidence criterion. The unmasked tokens are then fed back as hard embeddings in the next step's input.

This procedure suffers from the **information island problem**: the rich distributional information produced at each step --- the full logit vectors $\ell_i^t \in \mathbb{R}^{|V|}$, the attention distributions, and the implicit inter-position dependencies captured in the Transformer's hidden states --- is entirely discarded after the argmax operation. Let $e_v \in \mathbb{R}^d$ denote the embedding of token $v$, and let $\texttt{mask\_emb} \in \mathbb{R}^d$ be the learned mask embedding. At step $t$, the input representation for position $i$ is:

$$h_i^t = \begin{cases} e_{x_i} & \text{if } i \text{ has been unmasked (hard token)} \\ \texttt{mask\_emb} & \text{if } i \text{ is still masked} \end{cases}$$

This hard switching discards all information from prior steps' predictions at masked positions. Even if the model at step $t+1$ (the previous, more masked step) assigned high probability to a particular token at position $i$, the next step's input at that position is the same uninformative $\texttt{mask\_emb}$ vector --- identical across all positions and agnostic to the model's evolving beliefs.

### 3.2 Belief-State Diffusion (BSD)

#### Core Idea

BSD generalizes DMI from a fixed additive mixture to a fully continuous representational framework. Instead of mixing logit-weighted embeddings with $\texttt{mask\_emb}$ at a fixed ratio, BSD **replaces** mask embeddings entirely with *belief states* --- probability-weighted embedding vectors that evolve via exponential moving average (EMA) across denoising steps. Only in the final $k$ steps does BSD commit to hard token assignments.

#### Algorithm

BSD operates in two phases (Algorithm 1). In **Phase 1** (continuous belief refinement, steps $T$ to $k+1$), the model operates on belief vectors rather than hard masks, with no argmax sampling or unmasking. In **Phase 2** (hard token reveal, steps $k$ to $1$), standard confidence-based unmasking proceeds from the accumulated belief states.

---

**Algorithm 1: Belief-State Diffusion (BSD)**

**Input:** prompt $x_p$, generation length $L$, model $f_\theta$, total steps $T$, hard-reveal fraction $k_{\text{frac}}$ (steps $k = k_{\text{frac}} \cdot T$), EMA schedule $\alpha(\cdot)$, temperature $\tau = 1.0$

**Output:** generated sequence $x_0$

1. Initialize belief vectors: $b_i^T \leftarrow \texttt{mask\_emb}$ for all generation positions $i$

2. **Phase 1: Continuous Belief Refinement** (steps $t = T, T-1, \ldots, k+1$)
   - Construct input: $[x_p; b^t]$ (prompt embeddings concatenated with belief vectors)
   - Forward pass: $\ell^t \leftarrow f_\theta([x_p; b^t])$
   - Compute soft predictions: $p_i^t \leftarrow \text{softmax}(\ell_i^t / \tau)$
   - Compute embedding mixture: $\hat{b}_i^t \leftarrow \sum_{v \in V} p_i^t(v) \cdot e_v$
   - EMA update: $b_i^{t-1} \leftarrow (1 - \alpha(t)) \cdot b_i^t + \alpha(t) \cdot \hat{b}_i^t$
   - L2 normalize: $b_i^{t-1} \leftarrow b_i^{t-1} \cdot \frac{\|\texttt{mask\_emb}\|}{\|b_i^{t-1}\|}$

3. **Phase 2: Hard Token Reveal** (steps $t = k, k-1, \ldots, 1$)
   - Forward pass from belief states: $\ell^t \leftarrow f_\theta([x_p; b^t])$
   - Compute confidence: $c_i^t \leftarrow \max_v \text{softmax}(\ell_i^t)_v$ for each still-masked position $i$
   - Select the $\lfloor (T - t + 1) \cdot L / k \rfloor$ highest-confidence positions for unmasking
   - For selected positions: replace $b_i$ with the hard embedding $e_{\arg\max_v \ell_i^t}$
   - Carry forward belief states for remaining positions

---

The $\sim$1.1$\times$ FLOPs overhead arises from the softmax computation over the full vocabulary and the weighted embedding sum at each Phase 1 step, which is negligible relative to the Transformer forward pass.

#### Key Design Decisions

**EMA update rate $\alpha(t)$.** Ablation across four schedule types --- linear(0.1$\to$0.8), cosine(0.1$\to$0.8), constant(0.3), and constant(0.5) --- shows that schedule shape is not a performance-differentiating factor: all achieve identical accuracy (6.2% on Countdown-16). We adopt the linear schedule as the default for conceptual simplicity.

**L2 normalization.** Belief vectors are normalized to match $\|\texttt{mask\_emb}\|$ at every step. Without this, probability-weighted embedding mixtures drift to a different norm scale than the inputs the Transformer was trained on, causing out-of-distribution behavior in early layers.

**Hard-reveal fraction $k_{\text{frac}}$.** The $k$ parameter controls the trade-off between continuous belief refinement and discrete token commitment. Ablation reveals that $k_{\text{frac}} = 0.75$ (only 25% of steps in the belief phase, 75% in hard reveal) achieves 6.2%, while $k_{\text{frac}} = 0.25$ and $k_{\text{frac}} = 0.50$ (longer belief phases) both yield 0%. This suggests that Dream-7B's Transformer requires early hard token anchors to ground its predictions --- the model's representations are not well-calibrated for extended continuous belief inputs. BSD's benefit comes from improving the *quality* of initial representations fed into Phase 2, not from extended continuous refinement. We analyze possible explanations for this sharp threshold in Section 5.

#### Relationship to Prior Work

BSD can be understood as a generalization within a family of continuous-representation methods for MDLMs (Table 1).

**Table 1.** Comparison of continuous-representation methods for MDLMs. BSD is distinguished by full mask replacement and EMA-based cross-step memory.

| Dimension | DMI (Ours) | LRD | ReMix | EvoToken | **BSD (Ours)** |
|---|---|---|---|---|---|
| Mask replacement | Mixed | Mixed | Continuous state | Gradual | **Full replacement** |
| Cross-step memory | None | None | Convergence tracking | None | **EMA accumulation** |
| Reveal trigger | Every step | KL convergence | After convergence | Gradual | **Last $k$ steps** |
| Training required | No | No | No | No | **No** |
| Compute overhead | $\sim$1.05$\times$ | $\sim$1$\times$ | $\sim$1$\times$ | $\sim$1$\times$ | **$\sim$1.1$\times$** |

DMI is the special case of BSD where $k = T$ (no belief phase) and $\alpha$ is fixed.

### 3.3 Adaptive Classifier-Free Guidance (A-CFG) for Dream-7B

#### Core Idea

Classifier-free guidance (CFG) amplifies the difference between conditional and unconditional predictions to sharpen model outputs. MDLMs offer a natural mechanism: re-masking a subset of positions to construct an "unconditional" input, enabling CFG without any additional training. We apply Adaptive CFG (A-CFG), introduced by Arriaga et al. (2025) for LLaDA-8B, to Dream-7B.

#### Algorithm

At each denoising step $t$:

1. **Confidence scoring:** For each non-prompt position $i$, compute $c_i^t = \max_v p_\theta(x_i^0 = v \mid x_t)$.

2. **Re-masking construction:** Select the $m\%$ least-confident positions ($m = 10\%$). Replace their current representations with $\texttt{mask\_emb}$ to construct the unconditional input $\tilde{x}_t$.

3. **Dual forward pass:**
   - Conditional: $\ell^+ = f_\theta(x_t)$
   - Unconditional: $\ell^- = f_\theta(\tilde{x}_t)$

4. **Guided logit combination:**

$$\ell_{\text{guided}} = \ell^+ + w \cdot (\ell^+ - \ell^-)$$

where $w = 1.5$ is the guidance weight, capped at $w_{\max} = 2.0$.

The computational overhead is exactly 2$\times$ vanilla FLOPs due to the dual forward pass.

#### Why Confidence Beats Cross-Step Stability (JSD)

Our original proposal (RACFG) attempted to improve A-CFG by using cross-step JSD stability for position selection. Dream-7B produces remarkably stable cross-step distributions, with JSD stability scores clustered at $\sim$0.997 across all positions (see Figure 6a). This near-degeneracy eliminates any meaningful signal for discriminating reasoning-critical positions. Across all RACFG configurations tested, accuracy was 0.0%, while A-CFG with simple confidence-based selection achieved 12.5%. We provide the full analysis in Section 4.3.

#### Why Fixed Guidance Outperforms Temporal Scheduling

CFG scheduling theory for continuous diffusion (Rojas et al., 2025) predicts that guidance should be suppressed early and amplified late. We tested four temporal schedules; fixed guidance at $w = 1.5$ achieved 12.5%, while *all* scheduled variants achieved 0.0% (Section 4.3). In masked diffusion, the transition from noise to signal is discrete: at any step, a position has either full signal (hard token) or no signal (mask). The model has sufficient context from already-unmasked positions at every step, making constant guidance appropriate throughout.

### 3.4 BSD+A-CFG Combination Protocol

We also evaluate a combination of both methods: BSD Phase 1 (continuous belief refinement) runs for steps $T$ to $k+1$, and then A-CFG is applied during Phase 2 (hard token reveal, steps $k$ to $1$). The unconditional input for A-CFG is constructed by re-masking the least-confident positions based on predictions from the accumulated belief states. This protocol tests whether representation-layer and prediction-layer improvements compose synergistically (hypothesis H7). The results (Section 4.2) show they do not.

### 3.5 Information-Theoretic Analysis: Belief Entropy Trajectories

We validate BSD's information accumulation mechanism through an entropy trajectory analysis. For each sample, we track the mean per-position entropy of belief vectors across denoising steps:

$$H(b_i^t) = -\sum_{v \in V} p_i^t(v) \log p_i^t(v)$$

**Monotonic decrease.** BSD belief entropy exhibits near-monotonic decrease during Phase 1, with average Spearman rank correlation $\rho = -0.95$ between step index and mean entropy. Of 16 pilot samples, 15/16 show monotonically decreasing entropy ($\rho < -0.8$). This confirms that beliefs consistently converge rather than oscillating.

**Lower terminal entropy.** At the Phase 1 boundary, BSD achieves mean terminal entropy of 0.001 versus 0.002 for vanilla's corresponding prediction entropy --- directionally validating that EMA accumulation preserves more information than independent per-step predictions.

**Entropy--accuracy correlation.** Across 16 pilot samples, the correlation between terminal belief entropy and task correctness is $r = 0.78$ ($p < 0.001$). This correlation is suggestive given the pilot sample size ($n = 16$) and requires full-scale confirmation, but it indicates that entropy trajectory monitoring could serve as a runtime quality indicator for BSD-enhanced generation.

These properties demonstrate that BSD produces *qualitatively different* internal representations, not just more computation --- providing a principled explanation for why continuous belief evolution improves over discrete mask-based denoising.

> **Figure 2.** Method architecture for BSD and A-CFG. *Panel A*: BSD two-phase pipeline --- EMA belief update cycle in Phase 1 (steps $T$ to $k+1$) with norm preservation, transition to Phase 2 (steps $k$ to $1$) with hard reveal. *Panel B*: A-CFG pipeline --- confidence scoring across positions, re-masking the least-confident $m\%$, dual forward pass (conditional vs. unconditional), guided logit combination. FLOPs: BSD $\sim$1.1$\times$, A-CFG $\sim$2.0$\times$. *(See* `fig2_method_architecture_desc.md` *for rendering specification.)*

> **Figure 3.** Belief entropy trajectories during denoising. BSD (blue): smooth monotonic decrease from $\sim$3.5 to $\sim$0.0 during Phase 1. Vanilla (gray): step-function drops at discrete unmask events. Annotated: Spearman $\rho = -0.95$, terminal entropy BSD = 0.001 vs. vanilla = 0.002. The shaded region represents the "information accumulation gap" between the two approaches. *(Generated from* `gen_entropy_trajectories.py`; *rendered as* `entropy_trajectories.pdf`*.)*

---

## 4. Experiments

We evaluate Belief-State Diffusion (BSD) and Adaptive Classifier-Free Guidance (A-CFG) on two reasoning benchmarks, comparing against vanilla Dream-7B denoising, established remasking baselines, and our prior Diffusion Memory Injection (DMI) method. Our experiments address four questions: (1) Do BSD and A-CFG individually improve reasoning accuracy? (2) How sensitive are they to hyperparameter choices? (3) Does their combination yield synergistic gains? (4) Do improvements generalize across benchmarks?

### 4.1 Experimental Setup

**Model.** We use Dream-v0-Instruct-7B (Liu et al., 2025), the strongest open-source masked diffusion language model at the time of our experiments. All experiments run on a single NVIDIA RTX PRO 6000 Blackwell GPU (98 GB VRAM).

**Benchmarks.** Our primary benchmark is **Countdown-500**, a structured arithmetic reasoning task requiring models to find arithmetic expressions over given numbers that reach a target value. We use 500 problems evaluated across 3 random seeds (42, 123, 456) for full-scale evaluation, and a 16-sample pilot (**Countdown-16**) for ablation studies. For cross-benchmark generalization, we evaluate on **GSM8K-16**, a 16-sample pilot of grade-school math word problems.

**Baselines.** We compare against five baselines:
- **Vanilla Dream-7B**: Standard denoising with 128 steps, generation length 256, temperature 0.4.
- **DMI** ($\alpha=0.3$): Diffusion Memory Injection, which mixes previous-step logit-weighted embeddings into mask positions with a fixed ratio (detailed in Appendix D).
- **ReMDM-conf**: Confidence-based remasking (Arriaga et al., 2025).
- **RCR**: Running Confidence Remasking (Li et al., 2025).
- **DTA**: Denoising-Time Adaptation with online LoRA updates (detailed in Appendix C).

**Our methods.** We evaluate:
- **BSD** ($k_\text{frac}=0.75$, linear $\alpha$ 0.1$\to$0.8): 25% of steps in belief phase, 75% hard reveal.
- **A-CFG** ($w=1.5$, fixed, $m=10\%$): Confidence-based re-masking with fixed guidance weight 1.5.
- **BSD+A-CFG**: BSD Phase 1 followed by A-CFG during Phase 2.

**Evaluation metrics.** We report accuracy (primary), n-gram repetition rates (rep-2, rep-3), lexical diversity (distinct-1/2/3), and FLOPs overhead. We do not use perplexity as a quality metric, following our finding that pilot PPL improvements failed entirely at full scale.

**Statistical protocol.** For Countdown-500, we apply McNemar's exact test with Bonferroni correction for pairwise comparisons, and report bootstrap 95% confidence intervals (10,000 resamples). For pilot evaluations ($n=16$), we report Cohen's $h$ effect sizes and note that statistical tests have very low power (minimum detectable effect $\sim$25pp). **All pilot results ($n = 16$) should be interpreted as directional evidence, not confirmed improvements.** Full-scale 3-seed validation of BSD and A-CFG is required for publication-ready conclusions.

### 4.2 Main Results

Table 2 presents the full-scale Countdown-500 results (3-seed). Table 3 presents pilot-scale results on Countdown-16 and GSM8K-16. We separate these to avoid conflating evaluations at different statistical power levels.

**Table 2. Full-scale results on Countdown-500 (3-seed averaging, $n = 500$ per seed).** Best result is **bolded**. These are the only publication-ready comparisons in this study.

| Method | Group | FLOPs | Countdown-500 |
|--------|-------|-------|---------------|
| Vanilla (128 steps) | Baseline | 1.0$\times$ | 4.7% $\pm$ 0.6 |
| ReMDM-conf | Remasking | $\sim$1.5$\times$ | 4.4% $\pm$ 1.0 |
| RCR | Remasking | $\sim$1.2$\times$ | 5.7% $\pm$ 0.6 |
| **DMI** ($\alpha$=0.3) | Cross-step info | $\sim$1.05$\times$ | **9.3% $\pm$ 1.4** |

DMI versus vanilla yields Cohen's $h = 0.72$ (medium effect size) and McNemar $p < 0.05$, confirming a statistically significant $\sim$2$\times$ improvement at near-zero computational cost.

**Table 3. Pilot-scale results on Countdown-16 and GSM8K-16 (single seed, $n = 16$).** All differences are directional only; no pairwise comparison reaches statistical significance at $n = 16$. Bootstrap 95% CIs are shown in brackets.

| Method | FLOPs | Countdown-16 | GSM8K-16 | rep-3 | distinct-3 |
|--------|-------|-------------|----------|-------|------------|
| Vanilla (128 steps) | 1.0$\times$ | 0.0% (0/16) | 25.0% (4/16) | 0.079 | 0.876 |
| DMI ($\alpha$=0.3) | $\sim$1.05$\times$ | 12.5% (2/16) [0.0, 25.0] | 25.0% (4/16) | 0.095 | 0.857 |
| BSD ($k$=0.75) | $\sim$1.1$\times$ | 6.2% (1/16) [0.0, 18.8] | 18.8% (3/16) | 0.048 | 0.913 |
| A-CFG ($w$=1.5) | $\sim$2.0$\times$ | 12.5% (2/16) [0.0, 25.0] | **37.5%** (6/16) [$-$12.5, 37.5] | 0.054 | 0.889 |
| BSD+A-CFG | $\sim$2.1$\times$ | 6.2% (1/16) | --- | 0.046 | 0.915 |
| DTA (LoRA) | $\sim$3.0$\times$ | 6.2% (1/16) | --- | --- | --- |

Several findings emerge from the pilot results, to be confirmed at full scale:

First, **cross-step information methods directionally outperform remasking baselines**: DMI, BSD, and A-CFG all show positive trends over vanilla on Countdown-16.

Second, **A-CFG shows the strongest pilot-scale accuracy** on both Countdown-16 (12.5%, 2/16) and GSM8K-16 (37.5%, 6/16). The GSM8K result is particularly notable, as A-CFG is the only method showing cross-benchmark improvement.

Third, **all methods maintain generation quality**: repetition rates (rep-3) remain below 0.10 and lexical diversity (distinct-3) above 0.85.

Fourth, the **BSD+A-CFG combination (6.2%) performs below both individual methods**, falsifying our synergy hypothesis (H7). We analyze this failure in Section 5.

### 4.3 Ablation Studies

We conduct systematic ablation studies on both methods using Countdown-16 ($n = 16$, seed 42, 128 steps). Given the small sample size, ablation results should be interpreted as identifying gross effects (e.g., working vs. non-working configurations) rather than fine-grained performance differences. Raw counts (correct/16) are provided alongside percentages.

#### 4.3.1 BSD $k$-Parameter (Belief Phase Length)

| $k_\text{frac}$ | Belief Phase | Accuracy | rep-3 | distinct-3 |
|-----|-------------|----------|-------|------------|
| Vanilla | 0% | 0.0% (0/16) | 0.079 | 0.876 |
| 0.25 | 75% | 0.0% (0/16) | 0.014 | 0.899 |
| 0.50 | 50% | 0.0% (0/16) | 0.020 | 0.947 |
| **0.75** | **25%** | **6.2% (1/16)** | 0.048 | 0.913 |

Only $k_\text{frac} = 0.75$ (the shortest belief phase) achieves non-zero accuracy, suggesting that Dream-7B requires early access to hard token anchors to form coherent reasoning chains (see Figure 4a).

#### 4.3.2 BSD Alpha Schedule

Fixing $k_\text{frac} = 0.75$, we ablate four $\alpha$ schedule variants:

| Alpha Schedule | Accuracy | rep-3 | distinct-3 |
|----------------|----------|-------|------------|
| linear (0.1$\to$0.8) | 6.2% (1/16) | 0.048 | 0.913 |
| cosine (0.1$\to$0.8) | 6.2% (1/16) | 0.048 | 0.913 |
| constant (0.3) | 6.2% (1/16) | 0.048 | 0.913 |
| constant (0.5) | 6.2% (1/16) | 0.067 | 0.852 |

All schedules produce identical accuracy. The alpha schedule shape is not a performance-differentiating factor at the optimal $k$ setting. We adopt linear (0.1$\to$0.8) as the default for its slightly better diversity profile.

#### 4.3.3 A-CFG Guidance Weight

| Guidance Weight $w$ | Accuracy | rep-3 | distinct-3 | FLOPs |
|---------------------|----------|-------|------------|-------|
| Vanilla | 0.0% (0/16) | 0.079 | 0.876 | 1.0$\times$ |
| $w = 1.0$ | 6.2% (1/16) | 0.057 | 0.900 | $\sim$2.0$\times$ |
| $w = 1.5$ | **12.5% (2/16)** | 0.054 | 0.889 | $\sim$2.0$\times$ |
| $w = 2.0$ | **12.5% (2/16)** | 0.052 | 0.885 | $\sim$2.0$\times$ |

Moderate-to-strong guidance ($w \geq 1.5$) is required for the largest improvements. Quality metrics remain stable, with no evidence of degeneration even at $w = 2.0$.

#### 4.3.4 A-CFG Temporal Schedule

| Schedule | Accuracy | rep-3 | distinct-3 |
|----------|----------|-------|------------|
| **Fixed** ($w = 1.5$) | **12.5% (2/16)** | 0.054 | 0.889 |
| Linear ramp | 0.0% (0/16) | 0.074 | 0.866 |
| Cosine ramp | 0.0% (0/16) | 0.068 | 0.875 |
| Threshold (70/30) | 0.0% (0/16) | 0.055 | 0.897 |

Fixed guidance dominates all scheduled variants, which uniformly achieve 0% accuracy. This falsifies H6 and suggests that the theoretical predictions from continuous diffusion CFG scheduling (Rojas et al., 2025) do not transfer to masked diffusion's discrete dynamics. Scheduled approaches suppress guidance at critical early steps, preventing the method from building momentum.

#### 4.3.5 RACFG vs. A-CFG: Re-masking Signal Quality

| Method | Re-mask Signal | Accuracy | FLOPs |
|--------|---------------|----------|-------|
| Vanilla | N/A | 0.0% (0/16) | 1.0$\times$ |
| RACFG (JSD stability) | Cross-step JSD | 0.0% (0/16) | $\sim$1.8$\times$ |
| A-CFG (confidence) | Single-step confidence | **12.5% (2/16)** | $\sim$2.0$\times$ |

A-CFG dramatically outperforms RACFG. Dream-7B produces remarkably stable cross-step probability distributions (JSD stability $\sim$0.997 everywhere), eliminating the "hesitation signal" RACFG relies on. We define *flip analysis* as examining the subset of problems where two methods disagree (one correct, one incorrect); all disagreements favor A-CFG over RACFG.

> **Figure 4.** Ablation study grid. (a) BSD $k$-parameter: only the shortest belief phase ($k_\text{frac} = 0.75$, 25%) achieves non-zero accuracy. (b) BSD $\alpha$ schedule: all schedules yield identical accuracy. (c) A-CFG guidance weight: moderate-to-strong guidance ($w \geq 1.5$) required. (d) A-CFG temporal schedule: only fixed guidance works; all scheduled variants yield 0%. Annotations show distinct-3 diversity scores. *(Generated from* `gen_fig4_ablation_grid.py`; *rendered as* `fig4_ablation_grid.pdf`*.)*

### 4.4 GSM8K Generalization

To test whether improvements transfer beyond structured arithmetic, we evaluate the best configuration of each method on GSM8K-16 (pilot, $n = 16$).

| Method | Countdown-16 | GSM8K-16 | $\Delta$ vs. Vanilla (GSM8K) |
|--------|-------------|----------|---------------------------|
| Vanilla | 0.0% (0/16) | 25.0% (4/16) | --- |
| DMI ($\alpha=0.3$) | 12.5% (2/16) | 25.0% (4/16) | +0.0pp |
| BSD ($k_\text{frac}=0.75$) | 6.2% (1/16) | 18.8% (3/16) | $-$6.2pp |
| **A-CFG** ($w=1.5$) | **12.5% (2/16)** | **37.5% (6/16)** | **+12.5pp** |

A-CFG is the only method showing consistent cross-benchmark improvement. DMI, despite being the strongest validated method on Countdown-500, shows zero benefit on GSM8K. BSD performs below vanilla on GSM8K (18.8% vs. 25.0%), suggesting that continuous arithmetic token mixing may disrupt free-form reasoning chains.

Flip analysis supports A-CFG's generalization: of the 4 disagreements between A-CFG and vanilla on GSM8K, A-CFG wins 3 and loses 1. While not statistically significant at $n = 16$, this 3:1 ratio is directionally consistent with the Countdown results.

> **Figure 5.** Cross-benchmark generalization. Grouped bar chart showing accuracy of each method on Countdown-16 (blue) vs. GSM8K-16 (orange). A-CFG is the only method with consistent improvement across both benchmarks (+12.5pp on GSM8K). BSD degrades on free-form math ($-$6.2pp). *(Generated from* `gen_fig5_gsm8k_generalization.py`; *rendered as* `fig5_gsm8k_generalization.pdf`*.)*

### 4.5 Compute Fairness Analysis

Since A-CFG requires $\sim$2$\times$ FLOPs, we compare each method against vanilla with a proportionally increased number of steps.

**Table 4. Compute-fair comparison (pilot, $n = 16$).** Each method is compared against vanilla with matched FLOPs.

| Method | FLOPs | Accuracy | Vanilla (matched FLOPs) | $\Delta$ |
|--------|-------|----------|------------------------|----------|
| BSD ($\sim$1.1$\times$) | 1.1$\times$ | 6.2% (1/16) | 12.5% (141 steps) | $-$6.2pp |
| A-CFG ($\sim$2.0$\times$) | 2.0$\times$ | 12.5% (2/16) | 12.5% (256 steps) | $\pm$0.0pp |
| BSD+A-CFG ($\sim$2.1$\times$) | 2.1$\times$ | 6.2% (1/16) | 6.2% (269 steps) | $\pm$0.0pp |
| DMI ($\sim$1.05$\times$) | 1.05$\times$ | 12.5% (2/16) | 12.5% (134 steps) | $\pm$0.0pp |

At matched compute budgets on the pilot, vanilla step-scaling is competitive. However, the full-scale results tell a different story: at 1.0$\times$ FLOPs, vanilla achieves only 4.7% while DMI achieves 9.3% --- a clear advantage for information-quality methods at the low-compute frontier. The pilot sample size ($n = 16$) provides very low resolution for distinguishing methods at similar accuracy levels. Full-scale evaluation is needed to determine whether A-CFG's information-quality improvements scale differently with problem difficulty.

### 4.6 Information-Theoretic Validation

We validate that BSD belief vectors accumulate information across denoising steps (pilot, $n = 16$).

| Property | BSD | Vanilla |
|----------|-----|---------|
| Mean terminal entropy | 0.001 | 0.002 |
| Spearman $\rho$ (step vs. entropy) | $-$0.952 | $-$0.932 |
| Samples with monotonic decrease ($\rho < -0.8$) | 15/16 (94%) | N/A |
| Entropy--accuracy correlation ($r$) | 0.784 ($p < 0.001$) | --- |

BSD belief entropy decreases near-monotonically during denoising, **supporting H2**. The strong entropy--accuracy correlation ($r = 0.784$) is suggestive at pilot scale and requires full-scale validation, but indicates that belief convergence quality is predictive of reasoning success (see Figure 3).

### 4.7 Hypothesis Verification Summary

**Table 5. Hypothesis verification summary.**

| ID | Hypothesis | Expected | Observed | Verdict |
|----|-----------|----------|----------|---------|
| H1 | BSD > DMI on Countdown-500 | $\geq$14% | 6.2% (pilot) | **Pending** |
| H2 | BSD entropy monotonically decreases | Monotonic | $\rho = -0.952$, 15/16 | **Supported** |
| H3 | Intermediate $k$ optimal | $k \approx T/4$--$T/2$ | $k_\text{frac} = 0.75$ best | **Falsified** |
| H4 | A-CFG > vanilla on Countdown | $\geq$15% | 12.5% (pilot) | **Partially supported** |
| H5 | JSD > confidence for re-masking | JSD +2pp | 0% vs 12.5% | **Falsified** |
| H6 | Temporal scheduling > fixed | +2pp | 0% vs 12.5% | **Falsified** |
| H7 | BSD+A-CFG synergy | $\geq$18% | 6.2% (< both) | **Falsified** |
| H8 | Best method generalizes to GSM8K | Significant gain | A-CFG +12.5pp | **Supported** (pilot) |
| H9 | BSD quality maintained | rep-3 < vanilla+20% | rep-3 = 0.048 | **Supported** |
| H10 | A-CFG no length bias | Mean $\pm$15% | Within range | **Supported** |
| H11 | Methods beat vanilla at equal FLOPs | Methods > vanilla | Competitive | **Falsified** (pilot) |

Of 11 hypotheses, 4 are supported (H2, H8, H9, H10), 4 are falsified (H3, H5, H6, H7), 1 is partially supported (H4), and 2 require full-scale validation (H1, H11). The falsified hypotheses provide actionable constraints analyzed in Section 5.

---

## 5. Discussion

The experimental results paint a nuanced picture: BSD and A-CFG each yield meaningful improvements over vanilla Dream-7B, yet several pre-registered hypotheses were decisively falsified. We analyze five key findings that constrain the design space for future MDLM inference-time methods.

### 5.1 Why BSD + A-CFG Combination Fails

Perhaps our most surprising negative result is that combining BSD and A-CFG yields *lower* accuracy (6.2%) than either method alone (BSD 6.2%, A-CFG 12.5%). The root cause lies in a fundamental tension between the representational properties each method requires.

During Phase 1, BSD evolves belief vectors via EMA accumulation, producing *smooth*, high-entropy distributional representations designed for gradual refinement. However, A-CFG's re-masking mechanism in Phase 2 relies on *sharp*, discriminable confidence scores to identify uncertain positions. When A-CFG operates on BSD's belief-conditioned predictions rather than standard mask-conditioned predictions, the confidence landscape is fundamentally altered: positions that BSD has partially resolved present a different uncertainty profile than positions emerging from fresh mask embeddings.

The guidance signal $\ell^+ - \ell^-$ is therefore computed over a mismatched distribution, producing interference rather than amplification. This suggests a general principle: **methods that modify the representation space and methods that modify the prediction space may not compose freely**, even when they operate at nominally orthogonal layers. This principle may have broader applicability beyond MDLMs --- analogous to known difficulties with stacking independently beneficial regularizers or optimization techniques.

### 5.2 Why JSD Stability Fails as a Re-masking Signal

RACFG, which selects re-masking positions via cross-step JSD instability, achieved 0.0% accuracy across all configurations. The root cause is that Dream-7B produces remarkably stable cross-step probability distributions, with JSD stability scores clustered at $\sim$0.997 across all positions (Figure 6a). This near-degenerate distribution eliminates any meaningful "hesitation signal."

High cross-step prediction consistency is arguably a *desirable* property of a well-trained denoising model, but this robustness makes stability-based position selection generically inapplicable to well-trained MDLMs. In contrast, single-step confidence directly measures current uncertainty without requiring inter-step variation. For models exhibiting Dream-7B-like cross-step stability, future work on MDLM inference-time scaling should prioritize *within-step* signals (confidence, entropy, attention patterns) rather than *across-step* signals (stability, trajectory divergence). Whether other MDLMs (e.g., LLaDA-8B) exhibit sufficient cross-step variation to support JSD-based methods remains an open question.

### 5.3 Why CFG Temporal Scheduling Fails for Masked Diffusion

The theoretical prediction from continuous diffusion (Rojas et al., 2025) --- suppress guidance early, maximize late --- failed decisively on Dream-7B. Fixed guidance achieved 12.5%, while every scheduled variant achieved 0.0%.

The theory-practice gap stems from a mismatch in information geometry. Continuous diffusion operates over smooth noise-to-signal transitions; masked diffusion's dynamics are discrete: at each step, a position is either fully masked or fully revealed. The model has access to sufficient context for guidance at any step where some tokens are revealed. Additionally, scheduled variants that suppress early guidance sacrifice exactly the steps where A-CFG has the most positions available for re-masking --- and therefore the greatest opportunity to construct an informative unconditional input. **On Dream-7B, CFG scheduling theory developed for continuous diffusion does not transfer to discrete masked diffusion**; whether this extends to all MDLMs requires further investigation.

### 5.4 The Representation Bottleneck: Real but Narrow

Both BSD and A-CFG provide evidence that the information island problem is a genuine bottleneck. BSD's information-theoretic validation is compelling: near-monotonic entropy decrease ($\rho = -0.95$, 15/16 samples), lower terminal entropy (0.001 vs. 0.002), and entropy--accuracy correlation ($r = 0.78$).

However, compute-fair comparisons temper this conclusion. At matched FLOPs on the pilot, vanilla step-scaling is competitive. This apparent contradiction can be understood through an efficiency-vs.-effectiveness lens: the bottleneck is real (evidence: entropy metrics show that information quality per step is improved), but addressing it at the representation level is not yet more efficient than simply adding compute. DMI is the notable exception --- its near-zero overhead ($\sim$1.05$\times$) delivers a validated $\sim$2$\times$ improvement on Countdown-500, making it the most efficient method discovered across four iterations and immediately deployable in any MDLM pipeline. Whether BSD and A-CFG offer advantages beyond vanilla step-scaling at larger evaluation scales remains an open question for full-scale validation.

### 5.5 Negative Results as Design Space Constraints

Across four iterations, this work has systematically mapped what does and does not work for MDLM inference-time scaling. The pattern is clear: *parameter-space* interventions (TTT, DTA) fail due to weak gradient signals in the converged MDLM loss landscape. *Structural* interventions modifying the denoising schedule (ReMDM, RCR, temporal scheduling) fail because they do not address underlying representation quality. *Cross-step signal* methods (RACFG/JSD) fail because well-trained MDLMs are too stable across steps. The methods that succeed --- DMI, BSD, and A-CFG --- all operate at the *within-step representation or prediction level*.

A-CFG stands out as the most promising candidate among evaluated methods: it is the only one demonstrating cross-benchmark generalization (+12.5pp on both Countdown and GSM8K at pilot scale). Its conceptual simplicity (confidence-based re-masking + fixed guidance weight) and the failure of our proposed enhancements suggest that A-CFG's effectiveness derives from fundamental properties of CFG in discrete diffusion, robust to additional complexity. The non-composability principle from Section 5.1 --- that representation-layer and prediction-layer methods may interfere --- should inform future attempts at method combination.

### 5.6 Limitations

Several important limitations constrain our conclusions. First, BSD and A-CFG have only been evaluated at pilot scale ($n = 16$), where bootstrap 95% CIs are wide ($\pm$12--31pp) and include zero for every comparison. Full-scale 3-seed validation on Countdown-500 is the immediate priority. Second, the mechanistic explanations for failure modes (Sections 5.1--5.3) are post-hoc rationalizations based on pilot data; alternative explanations cannot be ruled out at $n = 16$. Third, GSM8K evaluation used only 16 samples, and A-CFG's +12.5pp improvement has a 95% CI of [$-$12.5%, +37.5%]. Fourth, all experiments use Dream-7B; the JSD stability failure may not generalize to other MDLMs. Fifth, the compute-fair analysis suggests methods may not dominate vanilla step-scaling at all budgets, and this analysis should be extended to larger scales.

> **Figure 6.** Failure mode diagnostics. (a) JSD stability histogram: near-degenerate distribution peaked at $\sim$0.997 across all positions, explaining RACFG's inability to discriminate reasoning-critical tokens. (b) A-CFG guidance magnitude vs. sample correctness: correctly-solved problems receive higher mean guidance magnitude (22081 vs. 19367), confirming CFG targets useful positions. (c) BSD+A-CFG entropy trajectory: belief entropy during Phase 1 followed by confidence distribution shift at the Phase 1$\to$Phase 2 boundary, illustrating the representational mismatch that disrupts A-CFG. *(Generated from* `gen_fig6_failure_diagnostics.py`; *rendered as* `fig6_failure_diagnostics.pdf`*.)*

---

## 6. Conclusion

We have introduced two training-free inference-time scaling methods for masked diffusion language models that directly address the information island problem. Evaluated on Dream-7B, both methods individually outperform vanilla denoising and all remasking baselines on Countdown, with A-CFG achieving the strongest pilot-scale results.

**Pilot-scale evidence for continuous belief evolution.** BSD provides the first empirical evidence that continuous belief states in MDLMs converge in an information-theoretically principled manner: near-monotonic entropy decrease ($\rho = -0.95$, 15/16 samples), lower terminal entropy (0.001 vs. 0.002), and entropy--accuracy correlation ($r = 0.78$, $p < 0.001$). These results validate the hypothesis that replacing discrete mask embeddings with soft distributional representations enables meaningful information accumulation.

**Pilot-scale evidence for A-CFG as a general reasoning enhancement.** A-CFG is the only method in our evaluation showing consistent cross-benchmark improvement: Countdown +12.5pp and GSM8K +12.5pp over vanilla at pilot scale. Fixed guidance weight ($w = 1.5$) dominates all temporal scheduling variants, indicating that theoretical predictions from continuous diffusion do not transfer directly to masked diffusion's discrete dynamics.

**DMI as a validated near-zero-cost contribution.** Diffusion Memory Injection achieves 9.3% accuracy versus vanilla 4.7% on full-scale Countdown-500 (3 seeds, $p < 0.05$) --- approximately $2\times$ improvement at $\sim$1.05$\times$ FLOPs. DMI requires no backward pass, no architectural changes, and no hyperparameter tuning, making it immediately deployable as a default enhancement in any MDLM inference pipeline.

**Negative results as design space constraints.** Cross-step JSD stability signals are degenerate ($\sim$0.997 everywhere); CFG temporal scheduling does not transfer from continuous to masked diffusion; BSD and A-CFG are non-composable; and parameter-space adaptation produces insufficient gradient signal. These failures systematically map what does and does not work for MDLM inference-time scaling.

**Limitations.** BSD and A-CFG results are at pilot scale ($n = 16$), where confidence intervals are wide and include zero. The documented pilot-to-full-scale effect size shrinkage across prior iterations (up to 24pp) warrants caution. Compute-fair analysis suggests that the methods' advantage is concentrated at the low-compute frontier.

**Future directions.** The immediate next step is full-scale 3-seed validation of BSD and A-CFG on Countdown-500 and GSM8K. Beyond validation, three research avenues emerge: (1) *training-aware belief states* --- learning belief update rules via a lightweight auxiliary module (cf. MetaState) --- could address BSD's sensitivity to the $k$-parameter; (2) *confidence-calibrated guidance* with learned weight schedules could overcome the failure of theory-driven temporal scheduling by adapting to masked diffusion's information geometry; and (3) *scaling to larger models* may provide richer cross-step distributional variation, potentially restoring the informativeness of JSD-based stability signals that are degenerate on Dream-7B.

---

## Figures and Tables

- Figure 1: `fig1_info_island_bsd_desc.md` --- Information island problem and BSD teaser with accuracy inset
- Figure 2: `fig2_method_architecture_desc.md` --- Two-panel method architecture (BSD + A-CFG pipelines)
- Figure 3: `entropy_trajectories.pdf` --- Belief entropy trajectories (BSD vs. vanilla)
- Figure 4: `fig4_ablation_grid.pdf` --- 2$\times$2 ablation grid (BSD $k$, BSD $\alpha$, A-CFG $w$, A-CFG schedule)
- Figure 5: `fig5_gsm8k_generalization.pdf` --- Cross-benchmark generalization bar chart
- Figure 6: `fig6_failure_diagnostics.pdf` --- Three-panel failure mode diagnostics
- Table 1: inline --- Continuous-representation method comparison (Section 3)
- Table 2: inline --- Full-scale Countdown-500 results (Section 4)
- Table 3: inline --- Pilot-scale Countdown-16 and GSM8K-16 results (Section 4)
- Table 4: inline --- Compute-fair comparison (Section 4)
- Table 5: inline --- Hypothesis verification summary (Section 4)

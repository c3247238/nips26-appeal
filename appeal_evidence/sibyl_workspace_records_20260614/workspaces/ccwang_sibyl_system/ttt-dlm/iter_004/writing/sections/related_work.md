# 2. Related Work

Our work addresses inference-time reasoning enhancement for masked diffusion language models (MDLMs) through two orthogonal training-free mechanisms: continuous belief-state representations and classifier-free guidance. We situate our contributions within four lines of related work.

## 2.1 Masked Diffusion Language Models

Discrete diffusion models for text generation have rapidly matured from proof-of-concept systems to competitive alternatives to autoregressive (AR) language models. SEDD (Lou et al., 2024) introduced score entropy discrete diffusion, achieving the first quality parity with GPT-2 at dramatically fewer network evaluations. MDLM (Sahoo et al., 2024) simplified the training objective via Rao-Blackwellized estimation, establishing masked (absorbing-state) diffusion as the dominant paradigm. Scaling these foundations, LLaDA (Nie et al., 2025) trained an 8B-parameter masked diffusion LLM that matches LLaMA3-8B on in-context learning benchmarks and uniquely breaks the reverse curse, while Dream (Liu et al., 2025) initialized from AR checkpoints with context-adaptive noise rescheduling, achieving the strongest open-source MDLM results on planning tasks (Countdown 16.0 vs. AR 6.2). Subsequent work has extended MDLMs to preference alignment (LLaDA 1.5; Nie et al., 2025), sparse mixture-of-experts (LLaDA-MoE), 100B scale (LLaDA 2.0), token-to-token editing with RL (LLaDA 2.1; Chen et al., 2026), and code specialization (Dream-Coder; Liu et al., 2025). Google's closed-source Gemini Diffusion demonstrated industrial viability at 1,479 tokens/sec.

Despite these advances, MDLMs exhibit a fundamental limitation that we term the **information island problem** (cf. MetaState; Xia et al., 2026): each denoising step performs an independent forward pass, and the rich distributional information (logits, attention patterns) from the previous step is entirely discarded after discrete argmax sampling. The next step operates on a fresh mask embedding that carries no memory of prior predictions. This contrasts with AR models, where the KV cache naturally accumulates context. Our BSD and A-CFG methods directly address this limitation at the representation and prediction layers, respectively.

## 2.2 Inference-Time Scaling for MDLMs

A rapidly growing body of work seeks to improve MDLM output quality at inference time without retraining. These methods span several categories:

**Remasking-based approaches.** ReMDM (Arriaga et al., 2025) derives principled remasking samplers (cap, rescale, confidence, loop) from custom backward processes, showing that increased sampling steps improve text quality toward AR levels. MDPO (Li et al., 2025) introduces Running Confidence Remasking (RCR) as a plug-and-play inference strategy alongside RL training. CORE (Chen et al., 2026) probes context robustness to identify fragile tokens for targeted remasking. RemeDi (Park et al., 2025) trains a dual-stream architecture for self-reflective remasking. However, our experiments confirm that pure remasking strategies — ReMDM-conf (4.4%) and RCR (5.7%) on Countdown-500 — fail to surpass vanilla Dream-7B (4.7%), because remasking redistributes computation across steps without accumulating cross-step information.

**Search and particle methods.** UnMaskFork (Chen et al., 2026) applies MCTS with deterministic branching over unmask trajectories. Prism (Li et al., 2026) combines hierarchical trajectory search with self-verification feedback. Self-Rewarding SMC (Li et al., 2026) uses multi-particle sampling with trajectory-level confidence as self-reward. TReASURe (Zhang et al., 2025) proposes low-variance tree search via resubstitution scoring. PG-DLM (Wang et al., 2025) provides convergence guarantees through particle Gibbs sampling. These methods achieve quality improvements but incur substantial compute overhead from maintaining multiple parallel trajectories.

**RL post-training.** d1 (Zhao et al., 2025) and wd1 (Li et al., 2025) apply GRPO-style reinforcement learning to optimize MDLM denoising as a sequential decision process, achieving strong reasoning gains (wd1: MATH500 44.2%, GSM8K 84.5%). DCoLT (Li et al., 2025) trains an Unmasking Policy Module to optimize reveal order. However, RL methods require costly training and are orthogonal to our training-free focus.

**Trained cross-step memory.** Most closely related to our BSD, MetaState (Xia et al., 2026) explicitly identifies the information island problem and trains a GRU module to maintain cross-step memory, achieving improved generation quality. Our BSD addresses the same problem but is entirely **training-free**, replacing mask embeddings with EMA-updated belief vectors that require no additional parameters or training data.

Our work is distinguished from all the above by operating at the **representation level** (BSD) and **prediction level** (A-CFG) simultaneously, without external verifiers, additional training, or architecture modification.

## 2.3 Continuous Token Representations in Diffusion

Several concurrent works validate that replacing discrete mask tokens with continuous representations improves MDLM performance, forming the representational foundation for our BSD:

LRD (Zhu et al., 2025) mixes logit-weighted embeddings into mask positions, achieving GSM8K +2.9 and MATH500 +3.8 with 10.6x acceleration through KL-triggered early commitment. ReMix (Ye et al., 2026) maintains a continuous mixing state that gradually transitions from mask embeddings to predicted embeddings, achieving 2--8x speedup without quality loss via convergence-triggered hard reveal. EvoToken-DLM (arXiv 2601.07351) proposes gradual token evolution from mask to predicted embeddings across steps. Soft-Masked Diffusion (IBM; arXiv 2510.17206) dynamically mixes mask embeddings with top-k predicted embeddings, showing consistent improvements on Dream-7B code benchmarks. PRR (Wan et al., 2026) leverages cross-step information for inference acceleration. CADD (arXiv 2510.01329) augments discrete states with continuous latent vectors, enabling controllable mode-seeking and mode-covering behavior. CCDD (arXiv 2510.03206) theoretically proves that continuous diffusion has strictly stronger expressivity than discrete diffusion or recurrent Transformers.

Our Diffusion Memory Injection (DMI), developed in a prior iteration of this work, belongs to this family: it injects previous-step logit-weighted embeddings into mask positions via fixed-ratio mixing ($\alpha \cdot \text{mask\_emb} + (1-\alpha) \cdot \sum_v p(v) \cdot e_v$), achieving 9.3% on Countdown-500 vs. vanilla 4.7% with near-zero overhead.

BSD generalizes these approaches in two key ways. First, it performs **full replacement** of mask embeddings with belief vectors rather than mixing, treating continuous representations as the primary input rather than an additive augmentation. Second, it introduces **EMA accumulation** across denoising steps, enabling cross-step memory without trained parameters. Table 1 in Section 3 details these distinctions.

## 2.4 Classifier-Free Guidance for Diffusion Language Models

Classifier-free guidance (CFG; Ho & Salimans, 2022) has been transformative for continuous diffusion models in image generation. Its adaptation to discrete diffusion language models is more recent and technically nontrivial, since MDLMs lack an explicit noise level to condition on.

A-CFG (NeurIPS 2025; arXiv 2505.20199) provides the key breakthrough: it constructs unconditional inputs by re-masking the least-confident token positions, then applies standard CFG logit interpolation ($\ell_{\text{guided}} = \ell_{\text{cond}} + w \cdot (\ell_{\text{cond}} - \ell_{\text{uncond}})$). On LLaDA-8B, A-CFG achieves GSM8K 73.5 — surpassing LLaMA3-8B (53.1) — and GPQA +3.9, establishing CFG as the most powerful training-free reasoning enhancement for MDLMs.

CFG temporal scheduling theory (Rojas et al., 2025; arXiv 2507.08965) analyzes when guidance should be applied during the denoising process, predicting that high guidance is harmful at early steps (high mask rate, insufficient context) and beneficial at late steps. This motivates our investigation of scheduled vs. fixed guidance weights.

Our A-CFG experiments on Dream-7B reveal two surprising findings that depart from prior work. First, confidence-based re-masking decisively outperforms cross-step JSD stability-based re-masking (12.5% vs. 0.0% on Countdown-16), because Dream-7B produces near-degenerate cross-step stability scores ($\sim$0.997 everywhere), eliminating the discriminative signal that JSD-based selection requires. Second, fixed guidance weight outperforms all temporal scheduling variants (linear, cosine, threshold), suggesting that the assumptions of continuous diffusion CFG scheduling theory — smooth noise-to-signal transitions — do not transfer to masked diffusion's discrete dynamics.

## 2.5 Test-Time Training and Adaptation

Test-time training (TTT; Sun et al., 2024) and its extensions (Titans; Akyürek et al., 2024; TTT with KV binding) achieve strong results in AR models by updating model parameters on test inputs. DTA, developed in a prior iteration of our work, applies LoRA-based online adaptation during MDLM denoising.

However, parameter-space adaptation faces a fundamental obstacle in MDLMs: the masked language modeling loss at intermediate denoising steps is already extremely low (0.005--0.032), providing insufficient gradient signal for meaningful parameter updates. DTA achieves only 6.2% on Countdown pilot — below vanilla (12.5%) — confirming that the gradient landscape is too flat for online learning. This negative result motivates our shift from parameter-space to **representation-space** (BSD) and **prediction-space** (A-CFG) interventions, where the signal-to-noise ratio is more favorable.

<!-- FIGURES
- None
-->

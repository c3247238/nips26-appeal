# 5. Related Work

## 5.1 KV-Cache Methods for MDMs

Standard transformer KV-cache reuse is invalidated in MDMs because mask state changes globally at each denoising step: when a token transitions from masked to unmasked, its key and value representations shift discontinuously, requiring recomputation of all attended positions. Seven published methods address this problem through different approximation strategies.

**Fast-dLLM** [CITE:fastdllm] reuses cached KV entries within each generation block and applies confidence-aware parallel decoding — tokens with prediction confidence above a threshold are committed without re-attending to later positions. Reported speedup is up to 27.6× on LLaDA-8B and Dream-7B. The parallel decoding component reduces the number of sequential refinement rounds, compounding the KV savings. Fast-dLLM requires block-wise generation and is not directly applicable to models that produce tokens in a single unstructured canvas.

**dKV-Cache** [CITE:dkvcache] introduces two variants. The near-lossless variant (dKV-Cache-Decode) refreshes cache entries when the token's predicted distribution has shifted by more than a calibrated threshold between consecutive steps; the aggressive variant (dKV-Cache-Greedy) extends the refresh interval more aggressively. Reported range is 2–10× speedup.

**EntropyCache** [CITE:entropycache] takes an $O(V)$ constant-cost approach: at each denoising step $t$, it computes the decoded token entropy $H_i = -\sum_v p_\theta(v \mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t)$ for each position $i$ and reuses the cached KV entry from step $t-1$ if $H_i < \eta$. This avoids the scaling cost of computing KV differences across context length. Reported speedup is 15.2–26.4× on LLaDA-8B-Instruct and Dream-7B-Instruct; the method is the closest published baseline to our M1 implementation. Our implementation reproduces the entropy-signal logic but executes full forward passes without kernel-level sparse attention (see Section 4.2 and Section 6.3 for the resulting implementation gap).

**Elastic-Cache** [CITE:elasticcache] applies drift-aware cache updates with per-layer adaptive refresh scheduling, using most-attended token signals to detect when global context has shifted. Reported speedups are 8.7× on short sequences and 45.1× on long sequences, owing to increased cache hit rates when the sequence is long enough that most positions stabilize early.

**Window-Diffusion** [CITE:windowdiffusion] analyzes structural locality in MDM token dependencies and introduces a sliding-window cache that divides the canvas into active tokens (full recomputation), buffer tokens (cached KV), and pruned far-field tokens. Published speedup reaches up to 99× on LLaDA and Dream. The window mechanism sacrifices global bidirectional context, which may explain the quality degradation at long-range dependency tasks.

**SlowFast+dLLM-Cache** [CITE:slowfast] combines a two-stage dynamic sampling strategy with caching: a "slow" phase runs full-quality denoising for the first fraction of steps, and a "fast" phase applies aggressive caching thereafter. Reported speedup is 34.22× on LLaDA.

**FlashDLM-FreeCache** [CITE:flashdlm] (the caching component of FlashDLM) approximates KV reuse with the Guided Diffusion AR supervisor providing token confidence signals. The combined system achieves 12.14× speedup on LLaDA-8B.

**Position relative to this work.** All seven KV-cache methods are evaluated independently, on different benchmark subsets, with different throughput measurement conventions. None evaluates whether KV-cache methods compose with any other acceleration family. Our M1 (EntropyCache, $\eta=2.0$) is one instance of this family; the M1+IGSD composability analysis (Section 2 and Section 3) is the first systematic evaluation of KV-cache composition with speculative denoising.

## 5.2 Step Reduction and Adaptive Scheduling

Reducing the total number of denoising steps $T$ from the default 64 is the most direct path to speedup, but MDMs impose a constraint with no AR analog: each step's token selections are conditioned on the mask state inherited from all previous steps, creating a sequential dependency chain.

**Saber** [CITE:saber] decouples step count from token count per step. At each step, the method unmasks more tokens (proportional to a confidence score) to complete generation in fewer steps, with a backtracking mechanism that re-masks tokens if downstream steps reveal inconsistency. Saber reports 251.4% speedup on code generation benchmarks, though the backtracking mechanism increases code complexity and its behavior on reasoning benchmarks is less well-characterized.

**PRR (Progressive Refinement Regulation)** [CITE:prr] uses a lightweight trained controller that monitors convergence trajectories across steps and modulates temperature to selectively accelerate or slow denoising per token. PRR requires training an auxiliary controller, making it a training-based method by the definitions used in this paper.

**Model scheduling** [CITE:modelscheduling] proposes running a smaller MDLM at a subset of denoising steps, based on the observation that not all steps contribute equally to final quality. The middle steps are identified as most sensitive to model capacity, while early and late steps tolerate smaller models. This is a single-method analysis focused on model allocation, not step count reduction or composability.

**Negative finding — M2 in this work.** Our simplified Saber implementation (M2, step-jump $J \in \{2, 4, 6, 8\}$, no backtracking) reaches a NO\_GO verdict on LLaDA-8B-Instruct: accuracy retention collapses to 28% at $J=4$ and 24% at $J=8$. The root cause — LLaDA's discrete mask schedule requires sequential cumulative conditioning that is violated by step skipping — corroborates the challenge of transferring continuous diffusion DDIM-style acceleration to discrete MDMs. This is a publishable negative result: practitioners who attempt to deploy Saber-style schedules on fully masked MDMs should expect similar degradation. The failure mode is characterized formally in Section 3 as interference pattern F1 (step_starvation).

## 5.3 Speculative Decoding for MDMs

Speculative decoding in AR models uses a small draft model to generate candidate continuations, which a large verifier then accepts or rejects via a ratio test, guaranteeing lossless quality while reducing the number of large-model forward passes [CITE:leviathan2023speculative]. Three lines of work adapt this paradigm to MDMs.

**DualDiffusion** [CITE:dualdiffusion] introduces a draft-verify framework within the MDM regime. A lightweight draft MDM produces a candidate masked sequence, and a larger MDM verifier accepts positions via a distribution-ratio test adapted to discrete mask states. The approach requires maintaining a separate draft model, which imposes a memory overhead comparable to running two models simultaneously. The evaluation focuses on MMLU and GSM8K; speedup figures are not reported as a primary claim.

**DFlash** [CITE:dflash] uses a block diffusion model as a draft generator for an AR verifier. This is a cross-architecture setup: the draft model (block diffusion) and the verifier (AR) operate on fundamentally different generation paradigms. DFlash reports >6× lossless speedup, 2.5× better than EAGLE-3. Because the verifier is an AR model, DFlash is not directly applicable to MDM-only deployment.

**SSD** [CITE:ssd] (Self-Speculative Decoding) is the most relevant prior work for IGSD. Gao et al. [CITE:gao2025ssd] introduce a training-free, no-auxiliary-model self-speculative framework for MDMs (LLaDA-8B, Dream-7B). SSD generates draft tokens using the model's own forward pass with top-$k$ confidence selection, then verifies them via hierarchical verification trees in a single batch forward pass. Reported speedup is 2.11–3.46× on GSM8K, MATH500, HumanEval, and MBPP — the same benchmarks used in this work. SSD achieves **lossless** output (the verified distribution is identical to full-step generation). This paper was published in October 2025, after the original ComposeAccel proposal was drafted.

**SSMD** [CITE:ssmd] (Self-Speculative Masked Diffusions, Campbell et al. 2025) adapts the attention mask switching mechanism — toggling between non-causal and causal attention to allow the model to serve as both draft generator and verifier in a single forward pass. Tested at GPT-2 scale; achieves approximately 2× reduction in forward pass count.

**S2D2** [CITE:s2d2] (Han et al. 2026) applies training-free self-speculation to block-diffusion LLMs. S2D2 uses the same pretrained model as drafter (parallel block decoding) and verifier (AR mode). Because S2D2 targets block-diffusion architectures (SDAR, LLaDA2.1-Mini with partial AR structure) rather than fully masked MDMs, it is not directly comparable to our setting.

**Position of IGSD relative to SSD.** IGSD and SSD both eliminate the external draft model requirement. The key architectural difference is in the draft mechanism. SSD drafts using a full-step ($T_{\text{draft}} = T = 64$) forward pass and reduces total computation via tree-structured batch verification; IGSD drafts using a **reduced-step** ($T_{\text{draft}} = 16$) forward pass, producing a coarse output whose high-confidence tokens ($c_i \geq \tau$) are frozen for the refine phase. This design creates a structural condition — token freezing in $S_{\text{accept}}$ — that enables super-multiplicative KV-cache synergy (Section 4), a property absent from SSD's architecture because SSD does not freeze a token partition. IGSD is approximate (35.1% accuracy retention on GSM8K at $\tau=0.9$, $T_{\text{draft}}=16$); SSD is lossless. IGSD's primary value in this paper is as a composability enabler for the M1+IGSD synergy, not as a standalone deployment solution.

## 5.4 Block Diffusion and Hybrid Architectures

Several methods restructure the generation process through training.

**Block Diffusion (BD3-LMs)** [CITE:blockdiffusion] interpolates between AR and diffusion generation by dividing the sequence into blocks, applying AR generation across block boundaries and masked denoising within blocks. This enables direct KV-cache reuse across blocks (a different mechanism than per-step approximation) and flexible-length generation. BD3-LMs are trained from scratch and achieve SOTA on language modeling benchmarks. The speedup over pure-diffusion baselines comes from the AR block structure, not from acceleration methods applied at inference time.

**Fast-dLLM v2** [CITE:fastdllmv2] adapts a pretrained AR model into a block-diffusion LLM with approximately 1B fine-tuning tokens (500× less compute than training Dream from scratch). A hierarchical block-level and sub-block cache achieves 2.5× speedup over AR baselines. The fine-tuning requirement places this in the training-based category.

**D2F (Discrete Diffusion Forcing)** [CITE:d2f] is the first open-source DLM to surpass AR throughput in practice, combining block-wise sequential generation with KV-cache reuse and AR-diffusion hybrids. D2F requires distillation training.

**Scope statement.** This paper studies training-free methods only. Block Diffusion, Fast-dLLM v2, and D2F are out of scope as primary comparisons, but they define the upper bound of achievable speedup: the 5.13× combined speedup of M1+IGSD in this work falls substantially below the 27.6×–34.22× reported by training-free methods on compatible hardware configurations (Fast-dLLM, SlowFast), and further below the 57.51× of the training-based Learn2PD pipeline. The contribution of this paper is not competitive throughput but structural insight: understanding which training-free methods compose and why.

## 5.5 Composability of Intervention Methods

**Kolbeinsson et al. (2024)** [CITE:kolbeinsson2024composable] study composability in the space of LLM interventions — knowledge editing, model compression, and unlearning applied to the same model. They measure order-dependence and interaction metrics across $C(3,2)=3$ pairwise combinations of these interventions. Their composability framework is spiritually similar to ours, but their interventions operate on model weights (static modifications) rather than inference-time algorithms, and they study entirely different method families. The MDM inference setting introduces a dynamic composability challenge (the mask state evolves at each step) with no analog in weight-space intervention studies.

No prior work evaluates pairwise composability of training-free inference acceleration methods for MDMs, or provides a failure-mode atlas characterizing the conditions under which method combinations destructively interfere. The gap encompasses both the measurement framework (our orthogonality metric $\text{Ortho}(M_a + M_b)$) and the causal analysis (the frozen-token synergy mechanism documented in Section 4 and Figure 8).

<!-- FIGURES
- None
-->

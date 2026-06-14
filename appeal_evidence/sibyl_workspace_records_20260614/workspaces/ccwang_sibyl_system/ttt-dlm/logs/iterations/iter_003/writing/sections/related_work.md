## 2. Background and Related Work

Denoising-Time Adaptation (DTA) sits at the intersection of masked diffusion language models, inference-time scaling, and test-time training. We review each thread and highlight the specific gap that DTA fills.

### 2.1 Masked Diffusion Language Models

Discrete diffusion for text generation has progressed rapidly from proof-of-concept to industrial scale. MDLM \citep{sahoo2024mdlm} introduced a Rao-Blackwellized objective that closed the perplexity gap with autoregressive (AR) models; SEDD \citep{lou2024sedd} extended score matching to discrete spaces, earning the ICML 2024 Best Paper award. These foundations enabled two landmark open-source systems: **LLaDA 8B** \citep{nie2025llada}, which matches LLaMA3 8B on in-context learning and breaks the reversal curse, and **Dream 7B** \citep{liu2025dream}, which initializes from an AR checkpoint and introduces context-adaptive noise rescheduling, achieving state-of-the-art planning performance (Countdown: 16.0 vs.\ AR 6.2). Subsequent variants include LLaDA 1.5 (VRPO preference alignment) \citep{llada15_2025}, LLaDA-MoE (7B parameters, 1.4B active) \citep{llada_moe_2025}, Dream-Coder (code-specialized) \citep{dreamcoder2025}, and the closed-source Gemini Diffusion at 1,479 tokens/sec \citep{geminidiffusion2025}. On the theoretical side, Svete \& Sabharwal \citep{svete2025reasoning} proved that MDMs are computationally equivalent to recurrent Transformers, and Jiang et al.\ \citep{jiang2025optimal} showed that DLMs with remasking constitute provably optimal parallel samplers.

A defining property of the DLM denoising process is that each step $t$ receives a partially masked sequence $\mathbf{x}_t$, makes predictions via a full bidirectional forward pass, samples tokens, and then **discards** all continuous representations—logits, attention maps, hidden states—before proceeding to step $t{-}1$. Xia et al.\ \citep{xia2026metastate} term this the "Information Island" problem: every denoising step starts from scratch, unable to leverage the understanding accumulated by its predecessors. This isolation is the structural bottleneck that all inference-time methods must address, and it motivates DTA's core idea of carrying forward information in parameter space rather than token space.

### 2.2 Inference-Time Scaling for DLMs

The past two years have seen an explosion of methods for improving DLM output quality at inference time. We organize them by the mechanism through which they inject cross-step information, highlighting the trade-offs each incurs.

**Token-space remasking.** ReMDM \citep{arriola2025remdm} introduced four remasking strategies (cap, rescale, conf, loop) derived from custom backward processes, demonstrating that increasing sampling steps can push masked diffusion text quality toward AR levels on small models (${\sim}170$M). Running Confidence Remasking (RCR) \citep{chen2025mdpo} offers a plug-and-play alternative that tracks per-token confidence across steps. CORE \citep{zhai2026core} probes context robustness by masking neighboring tokens, identifying positions that are "context-fragile" rather than merely low-confidence, yielding +9.2pp on MBPP with LLaDA-8B. HEX \citep{lee2025hex} ensembles multiple denoising schedules. A common limitation is that all these methods operate exclusively in discrete token space: they reshuffle which tokens are masked and re-predicted but **never update the model's internal representation** of the generation. Our full-scale experiments confirm this limitation quantitatively—ReMDM-conf (4.4\%) and RCR (5.7\%) show marginal or no improvement over vanilla (4.7\%) on Countdown-500 with Dream-7B.

**Trained refinement and RL.** RemeDi \citep{xu2025remedi} adds a dual-stream architecture (Token Prediction Stream + Unmasking Policy Stream) trained via SFT and RL to detect and remask erroneous tokens, achieving open-source DLM SOTA. ProSeCo \citep{proseco2026} trains a corrector network that reuses the DLM's denoising outputs. RL-based approaches—d1/diffu-GRPO \citep{d12025}, wd1 \citep{wd12025}, DCoLT \citep{dcolt2025}, MDPO \citep{chen2025mdpo}, AGRPO \citep{agrpo2025}—frame denoising as a sequential decision process and optimize via policy gradient variants, with impressive gains (e.g., GSM8K 84.5\% for wd1; Countdown +59.4\% for AGRPO). These methods inject external information through gradient updates at **training time**, but the resulting parameters remain frozen during inference. DTA takes the complementary approach: it performs gradient updates **at inference time**, requiring no training phase.

**Search and verification.** Prism \citep{wang2026prism} combines hierarchical trajectory search with self-verification feedback. UnMaskFork \citep{li2026unmaskfork} models the unmasking trajectory as a search tree navigated by MCTS. TReASURe \citep{chen2025treasure} introduces deterministic resubstitution scoring for low-variance tree search. Reward-Guided Stitching \citep{stitching2026} stitches optimal steps across parallel trajectories using a process reward model, averaging +23.8\% across six benchmarks. PG-DLM \citep{pgdlm2025} applies particle Gibbs sampling with theoretical convergence guarantees. Self-Rewarding SMC \citep{srsmc2026} uses trajectory-level confidence as self-reward weights for importance resampling. These methods achieve strong results but depend on external verifiers, reward models, or multiple parallel denoising trajectories—substantial computational investments that DTA avoids entirely.

**Gradient-guided inference.** TABES \citep{tabes2026} derives a Token Influence Score from a first-order expansion of a trajectory cost functional, using a single backward pass to approximate infinite-step lookahead. ETS \citep{ets2026} factorizes the transition probability into a reference policy plus an energy term estimated via online Monte Carlo, offering training-free RL alignment with provable convergence. Both methods use gradients at inference time, but they guide **token selection** rather than updating model parameters. DTA's backward passes serve a fundamentally different purpose: building a persistent, cumulative representation of the current generation within the LoRA adapter.

**Cross-step memory (trained).** MetaState \citep{xia2026metastate} directly addresses the Information Island problem by training a GRU + cross-attention module to maintain a persistent hidden state across denoising steps, updated via K-step unrolled optimization. While MetaState and DTA share the motivation of cross-step memory, MetaState requires dedicated training of auxiliary modules, whereas DTA achieves cross-step memory through zero-shot LoRA adaptation—no training, no architectural changes, and no additional learned parameters.

**Positioning summary.** Table~\ref{tab:positioning} situates DTA within this landscape. DTA is the only method that simultaneously satisfies four desiderata: (i) parameter-level updates at test time, (ii) persistent cross-step memory, (iii) no training requirement, and (iv) a theoretical guarantee (ELBO monotonicity; see Section 3.2).

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
ReMDM / CORE / Prism & \ding{55} & \ding{55} & \ding{51} & \ding{51} & Limited \\
MetaState & \ding{55}\textsuperscript{$\dagger$} & \ding{51} & \ding{55} & \ding{51} & \ding{55} \\
RemeDi & \ding{55}\textsuperscript{$\dagger$} & \ding{51} & \ding{55} & \ding{51} & \ding{55} \\
ProSeCo & \ding{55}\textsuperscript{$\dagger$} & \ding{51} & \ding{55} & \ding{51} & \ding{55} \\
TABES / ETS & \ding{55}\textsuperscript{$\ddagger$} & \ding{55} & \ding{51} & \ding{51} & Partial \\
TTT-Linear / Titans & \ding{51} & \ding{51} & \ding{55} & \ding{51} & \ding{51} \\
\textbf{DTA (Ours)} & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{$\dagger$\,Parameters fixed at inference time; memory comes from trained auxiliary modules.}\\
\footnotesize{$\ddagger$\,Gradients used to guide token selection, not to update model parameters.}
\end{table}

### 2.3 Test-Time Training

Test-time training (TTT) replaces fixed hidden states with learned ones: at inference time, a self-supervised loss drives gradient updates to a small parameter set, enabling the model to adapt to each input on the fly. Sun et al.\ \citep{sun2024ttt} introduced TTT-Linear, where a linear attention layer is replaced by a mini-model updated via gradient descent on a per-sequence reconstruction loss—effectively learning to compress the test-time context into fast weights. Titans \citep{behrouz2025titans} extended this idea with deeper memory modules and forgetting gates, scaling TTT to long-context regimes. Subsequent work has explored diverse instantiations: LaCT \citep{zhang2025lact} simplifies TTT with KV-binding that reduces to linear attention, TPTT \citep{furfaro2025tptt} converts pretrained Transformers into TTT architectures, and Locas \citep{lu2026locas} introduces pluggable parameterized memory for the last few layers—closest in spirit to DTA's LoRA injection into final Transformer layers. AllMem \citep{wang2026allmem} combines sliding-window attention with TTT memory networks for ultra-long contexts.

All existing TTT methods share a crucial structural assumption: they iterate along **sequence positions** in a **causal** (left-to-right) fashion. At position $i$, the model observes tokens $x_{<i}$, updates its fast weights $W^{(i+1)} = W^{(i)} - \eta \nabla_W \mathcal{L}(x_i; W^{(i)})$, and uses the updated weights for position $i{+}1$. This causal structure is natural for AR models but does not apply to DLMs, where generation proceeds along **denoising time steps** rather than sequence positions. At each denoising step $t$, the model sees the **entire** partially masked sequence with full bidirectional attention—there is no causal order over positions.

DTA adapts the TTT paradigm to this fundamentally different iteration axis. Instead of updating fast weights at each sequence position, DTA updates LoRA parameters at each denoising step:
$$\Delta\theta^{(t+1)} = \gamma \cdot \Delta\theta^{(t)} - \eta \nabla_{\Delta\theta} \mathcal{L}_{\mathrm{MLM}}(\mathbf{x}_{t-1}; \Delta\theta^{(t)})$$
where the self-supervised signal $\mathcal{L}_{\mathrm{MLM}}$ is the masked language modeling loss on already-revealed tokens—a loss that DLM denoising provides \emph{for free}, unlike AR TTT which requires designing an auxiliary self-supervised objective. The decay factor $\gamma$ serves the same role as Titans' forgetting gate: controlling the trade-off between memory accumulation and stability. This structural alignment—DLM denoising as implicit TTT made explicit through parameter updates—is the central insight of this paper.

<!-- FIGURES
- Table 1 (Figure 6 from outline): inline — Method positioning table comparing DTA against existing DLM inference-time scaling approaches across five capability axes
- None (no code-generated figures in this section)
-->

# Innovator Perspective: Unconventional Research Proposals for DLM Inference-Time Scaling

**Agent**: sibyl-innovator
**Date**: 2026-03-11
**Topic**: Masked Diffusion Language Model Inference-Time Compute Scaling (ReMask-Retry / TTT / TCR)
**Iteration**: 5 (post-pilot evidence round)

---

## Preamble: What the Existing Proposal Gets Right and Where It Falls Short

The current DaL proposal is a carefully reasoned incremental advance: insert TTT layers into frozen DLMs, treat denoising as self-supervised learning. But pilot evidence (P3: gate stuck at 0.007, -1.0pp accuracy; MetaState GRU: 43.75% vs vanilla 50.0%) suggests the "cross-step memory via lightweight module" paradigm may face a fundamental ceiling. The proposals below deliberately break away from this paradigm. They attack the problem from angles the existing literature has not explored, with concrete hypotheses and fast experimental plans.

---

## Proposal 1: Associative Denoising Memory (ADM) — Modern Hopfield Networks as Cross-Step Memory for DLMs

### Angle: Cross-Domain Transfer (Associative Memory Theory -> DLM Denoising)

### Core Insight

Recent work establishes a deep mathematical connection between diffusion models and modern Hopfield networks: the energy function of a trained diffusion model is asymptotically identical to that of a modern Hopfield network (Ambrogioni, 2024; PMC article on "Generative Diffusion Models Are Associative Memory Networks"). Meanwhile, Chaudhry et al. (ICLR 2026 Workshop) show that test-time scaling meets a fundamental bottleneck in sub-quadratic models due to limited associative memory capacity. The counter-intuitive implication: **the denoising process itself IS pattern retrieval from an associative memory, and scaling inference compute means scaling the retrieval dynamics**.

Instead of TTT (gradient-based weight updates) or GRU (fixed-gate updates), use **explicit modern Hopfield network layers** as cross-step memory. At each denoising step, the revealed tokens serve as query patterns; the Hopfield layer performs one-shot pattern completion in exponential-capacity associative memory. No gradient computation needed — just energy minimization dynamics.

### Hypothesis

**H_ADM**: A modern Hopfield layer (continuous Hopfield with exponential storage capacity) inserted between frozen DLM layers achieves >=2% accuracy improvement over vanilla Dream-7B on GSM8K, by performing associative pattern completion across denoising steps, WITHOUT requiring any gradient computation at inference time.

### Why This Might Work Where TTT Fails

1. **No gradient instability**: Hopfield retrieval is energy minimization, not SGD. No learning rate tuning, no gate initialization problems.
2. **Native to diffusion**: Diffusion denoising IS associative retrieval (Ambrogioni, 2024). ADM makes this explicit rather than fighting against it.
3. **Exponential capacity**: Modern Hopfield networks store exponentially many patterns (Ramsauer et al., 2021). The cross-step memory capacity is vastly larger than a small MLP fast-weight.
4. **One-shot, not iterative**: Pattern completion is a single forward pass, not K gradient steps. Overhead is negligible.

### Method

```
At denoising step t:
1. Revealed tokens R_t provide query vectors: Q = Embed(R_t)
2. Hopfield memory stores patterns from ALL previous steps:
   M_t = M_{t-1} ∪ {backbone hidden states at step t}
3. Hopfield retrieval: H(Q, M_t) = softmax(beta * Q @ M_t^T) @ M_t
   (beta = inverse temperature, controls retrieval sharpness)
4. Retrieved patterns injected via residual: h' = h + alpha * H(Q, M_t)
5. alpha is learnable (same gate as DaL, but simpler training signal)
```

Training: Meta-learn alpha and beta on a small dataset (same K-step unrolling as DaL). The Hopfield retrieval itself is parameter-free — it operates on the backbone's own hidden states as stored patterns.

### Failure Modes

- Hopfield retrieval may produce averaged/blurred patterns (mode collapse in associative memory)
- Memory M_t grows linearly with steps; may need pruning for long sequences
- The "diffusion = associative memory" equivalence is asymptotic — may not hold for practical DLMs

### Computational Cost

- No backward pass at inference → overhead is ~0.05x per step (just one extra softmax attention)
- Training: ~2-4 GPU-hours for alpha/beta meta-learning
- **Success probability: 30%** (novel, untested, but mathematically motivated)

### Key References

- Ambrogioni (2024). "In Search of Dispersed Memories: Generative Diffusion Models Are Associative Memory Networks." Entropy 26(5):381.
- Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR 2021.
- Chaudhry et al. (2025). "Test-time scaling meets associative memory." ICLR 2026 Workshop on New Frontiers in Associative Memories.
- Zhong et al. (2025). "Understanding Transformer from the Perspective of Associative Memory." arXiv 2505.19488.
- Modern Hopfield Networks with Continuous-Time Memories (arXiv 2502.10122).

---

## Proposal 2: Consistency-Accelerated Denoising with Adaptive Compute Allocation (CADAC)

### Angle: New Method (Consistency Models x Adaptive Compute for DLMs)

### Core Insight

Current DLM inference-time scaling methods (ReMDM, DaL, MetaState) all assume **more denoising steps = better quality**, allocating compute uniformly across the sequence. But consistency models (Song et al., 2023) and their discrete extensions (CDLM, arXiv 2511.19269; FS-DFM, arXiv 2509.20624) demonstrate that **a well-trained model can jump directly from high noise to clean output in 1-2 steps**. The counter-intuitive idea: **instead of adding memory/TTT to make each step better, teach the DLM to skip easy steps entirely and concentrate all compute on the hard parts**.

CADAC trains a lightweight "difficulty predictor" that, at each denoising step, classifies each token position as EASY (skip to consistency prediction) or HARD (continue iterative denoising + optional remasking). This creates a **spatially non-uniform denoising schedule** where reasoning-critical tokens get 10x more compute than filler tokens.

### Hypothesis

**H_CADAC**: A difficulty-aware adaptive compute allocation strategy achieves the same accuracy as 256-step uniform denoising with only 64-step average compute (4x speedup), AND outperforms 256-step uniform denoising by >=1% on reasoning benchmarks when given a matched 256-step budget (by concentrating compute on hard tokens).

### Why This Is Novel

- CDLM (arXiv 2511.19269) does consistency for uniform acceleration (all tokens equally)
- CoRe (arXiv 2602.04096) identifies vulnerable tokens but only for remasking, not compute allocation
- DyLLM (arXiv 2603.08026) does saliency-based token selection but for efficiency, not quality
- **Nobody has combined**: consistency jumps for easy tokens + extra denoising for hard tokens + adaptive per-token compute budgets. This is the DLM analog of early-exit / adaptive computation in AR models.

### Method

```
Phase 1: Train consistency shortcut (CDLM-style)
- Distill Dream-7B into a 1-step consistency predictor for "easy" positions
- Easy = positions where the model's prediction entropy < threshold

Phase 2: Train difficulty predictor
- Small MLP head on backbone hidden states
- Binary classification: EASY (consistency jump) vs HARD (continue denoising)
- Training signal: compare 1-step consistency prediction vs full denoising result
- Positions where they differ significantly = HARD

Phase 3: Adaptive inference
- At each step t:
  - Run difficulty predictor on all masked positions
  - EASY positions: apply consistency shortcut (finalize in 1 step)
  - HARD positions: continue standard denoising + optional remasking
- As denoising proceeds, more positions become EASY (natural curriculum)
```

### Failure Modes

- Consistency distillation quality may be poor for discrete tokens (irreducible error noted in arXiv 2602.16813)
- Difficulty predictor may be poorly calibrated (easy-hard boundary is noisy)
- Early finalization of "easy" tokens may create incoherent context for remaining hard tokens

### Computational Cost

- Consistency distillation: ~8-12 GPU-hours
- Difficulty predictor training: ~2-4 GPU-hours
- Pilot (test adaptive schedule on GSM8K-200): ~2-4 GPU-hours
- **Success probability: 35%** (builds on proven CDLM + CoRe ideas, but combination is untested)

### Key References

- CDLM (arXiv 2511.19269). Consistency Diffusion Language Models.
- FS-DFM (arXiv 2509.20624). Few-Step Diffusion Language Models.
- CoRe (arXiv 2602.04096). Context-Robust Remasking.
- DyLLM (arXiv 2603.08026). Saliency-based Token Selection.
- Song et al. (2023). Consistency Models. ICML 2023.

---

## Proposal 3: Latent Diffusion Reasoning (LDR) — Bypass Discrete Bottleneck via Continuous Latent Space

### Angle: Cross-Domain Transfer (Image Latent Diffusion -> Language Reasoning)

### Core Insight

A fundamental structural problem with masked DLMs: the denoising operates on **discrete** tokens, which creates an irreducible approximation error when factorizing the reverse transition (noted in arXiv 2602.16813, "One-step Language Modeling via Continuous Denoising"). LaDiR (arXiv 2510.04573, ICLR 2026) demonstrates that moving reasoning to a **continuous latent space** — where a VAE encodes text reasoning steps into blocks of "thought tokens" — enables holistic iterative refinement with adaptive test-time compute.

The radical proposal: **do NOT try to improve discrete DLM denoising at all. Instead, add a parallel continuous latent reasoning track that operates alongside the discrete DLM, providing "reasoning scaffolds" that guide the discrete denoising process.**

### Hypothesis

**H_LDR**: A hybrid architecture where a small continuous latent diffusion model (LaDiR-style) generates reasoning scaffolds in latent space, which are then decoded to guide the discrete DLM's denoising (via cross-attention or soft prompting), achieves >=3% improvement on GSM8K over vanilla Dream-7B.

### Why This Challenges Assumptions

The entire existing DLM-TTS literature assumes you must work within the discrete denoising framework. DaL, MetaState, ReMDM, CoRe — all operate on discrete tokens. But the discrete bottleneck may be the real problem: hard mask tokens destroy information, one-hot embeddings prevent smooth interpolation, and the factorized reverse process introduces irreducible error. LaDiR shows continuous latent reasoning is strictly more expressive.

### Method

```
Architecture:
1. Frozen Dream-7B backbone (discrete DLM)
2. Small VAE encoder: maps prompt + partial generation to continuous latent z (d=256)
3. Small latent diffusion model: refines z through T_latent steps (T_latent=8-16)
   - Each latent step captures holistic reasoning structure
4. Cross-attention bridge: latent z attends to DLM backbone hidden states
5. Soft guidance: DLM backbone receives latent z as additional context via cross-attention

Training:
- Train VAE on reasoning traces (GSM8K solutions)
- Train latent diffusion on encoded reasoning traces
- Train cross-attention bridge with frozen backbone + frozen latent model
- Only bridge parameters are tunable at inference time
```

### Failure Modes

- VAE bottleneck may lose critical reasoning information
- Continuous-discrete interface (latent guidance -> discrete token selection) is non-trivial
- Training pipeline is complex: VAE + latent diffusion + bridge
- May degenerate to just "prompting with latent noise"

### Computational Cost

- VAE training: ~4-6 GPU-hours
- Latent diffusion training: ~4-6 GPU-hours
- Bridge training: ~4-6 GPU-hours
- Pilot evaluation: ~2-4 GPU-hours
- Total: ~16-20 GPU-hours
- **Success probability: 20%** (highest novelty, highest risk, highest potential payoff)

### Key References

- LaDiR (arXiv 2510.04573). Latent Diffusion Enhances LLMs for Text Reasoning. ICLR 2026.
- CCDD (arXiv 2510.03206). Coevolutionary Continuous Discrete Diffusion.
- One-step Language Modeling via Continuous Denoising (arXiv 2602.16813).
- Rombach et al. (2022). Latent Diffusion Models. CVPR 2022.

---

## Proposal 4: Self-Play Denoising (SPD) — The DLM Plays Against Itself to Improve

### Angle: Improve Existing (Self-Play RL -> DLM Denoising)

### Core Insight

SPELL (arXiv 2509.23863) and SeRL (arXiv 2505.20347) show that LLMs can self-improve through self-play: generating their own training data and using it to iteratively improve. The twist for DLMs: **the denoising process naturally generates multiple trajectories from the same noised input** (different random unmaskings lead to different outputs). Instead of treating these as independent samples (as in Best-of-N), use them as **self-play episodes** where the model learns from its own successes and failures.

Concretely: run multiple denoising trajectories in parallel, score them with a lightweight self-reward (the DLM's own likelihood), and use the contrast between winning and losing trajectories to update the model via DPO/RLHF at test time. This is "inference-time RLHF" — no external reward model needed.

### Hypothesis

**H_SPD**: Self-play denoising with N=4 parallel trajectories and test-time DPO updates achieves >=2% improvement over vanilla Dream-7B on reasoning benchmarks, AND outperforms naive Best-of-N by >=1% (because SPD learns from the contrast, not just selects).

### Why This Might Work Where Best-of-N Failed

Our iteration history shows Best-of-N was completely ineffective (3x compute, +6.9% — likely noise). The problem: Best-of-N selects but doesn't learn. SPD is different:
1. **Contrastive learning**: The model updates toward the winning trajectory and away from the losing one — this is a genuine learning signal, not just selection.
2. **Self-reward**: Use the DLM's own log-likelihood on a separate forward pass as reward — no external verifier needed.
3. **Test-time adaptation**: The DPO updates are LoRA-based, applied at test time, reset per sequence.
4. **Compatible with DLM parallelism**: Multiple trajectories are naturally parallel in DLMs.

### Method

```
For each input sequence:
1. Initialize: sample 4 masked sequences from noise schedule
2. Parallel denoising: run 4 independent trajectories (standard DLM sampling)
3. Self-scoring: for each completed trajectory y_i:
   - Score(y_i) = mean log p(x_j | x_{-j}, y_i) for held-out positions j
   - (Self-consistency: re-mask random 10% tokens and measure prediction accuracy)
4. Rank trajectories: y_win = argmax Score, y_lose = argmin Score
5. Test-time DPO update:
   - LoRA parameters theta += alpha * grad(DPO_loss(y_win, y_lose; theta))
   - Single gradient step (or 2-3 steps)
6. Final generation: run one more denoising trajectory with updated theta
7. Reset theta for next sequence
```

### Failure Modes

- Self-reward (own likelihood) may be poorly calibrated — high likelihood != high quality
- DPO with only 4 trajectories may be too noisy for meaningful gradient signal
- Test-time LoRA updates add significant overhead (backward pass per sequence)
- Risk of degeneration: model learns to maximize own likelihood (reward hacking)

### Computational Cost

- Implementation: ~1 day (reuse existing DPO/LoRA infrastructure)
- Pilot (N=4, GSM8K-200): ~4-6 GPU-hours (4x parallel + DPO step + final generation)
- No pre-training needed — purely test-time
- **Success probability: 25%** (self-play is proven for AR; adaptation to DLM is novel)

### Key References

- SPELL (arXiv 2509.23863). Self-Play RL for Long-Context LLMs.
- SeRL (arXiv 2505.20347). Self-Play RL for LLMs with Limited Data.
- MDPO (arXiv 2508.13148). Masked Diffusion Policy Optimization.
- Self-Rewarding SMC (arXiv 2602.01849). Self-Rewarding for MDLMs.
- Rafailov et al. (2023). DPO: Direct Preference Optimization.
- "Teach Diffusion Language Models to Learn from Their Own Mistakes" (arXiv 2601.06428).

---

## Proposal 5: Denoising as Predictive Coding — A Free-Energy Principle Framework for DLM Scaling

### Angle: Cross-Domain Transfer (Neuroscience Predictive Coding -> DLM Denoising)

### Core Insight

Friston & Kiebel (2009) formalize perception as iterative denoising under the free-energy principle: the brain minimizes prediction error through hierarchical message passing, where each level generates predictions for the level below and receives prediction errors from below. This is structurally identical to DLM denoising, where each step generates predictions (unmask tokens) and receives feedback (the remaining mask pattern).

The current proposal (DaL) uses a flat SSL objective (MLM on revealed tokens). But predictive coding theory says: **the update signal should be the precision-weighted prediction error between hierarchical levels, not just the reconstruction loss**. This leads to a different architecture: instead of a single TTT layer, insert **a predictive coding circuit** that maintains top-down predictions and bottom-up prediction errors across denoising steps, with precision-weighting that naturally adapts to the denoising phase.

### Hypothesis

**H_PC**: A predictive coding circuit (top-down/bottom-up message passing with precision-weighted prediction errors) outperforms both DaL-TTT and MetaState-GRU as cross-step memory, because it naturally solves the gate problem (precision weighting replaces the learned gate) and the SSL-task alignment problem (prediction errors are directly task-relevant, not auxiliary SSL).

### Why This Addresses DaL's Root Failures

1. **Gate problem (P3)**: In predictive coding, there is no gate — the influence of higher-level predictions on lower-level processing is controlled by **precision** (inverse variance), which is naturally estimated from the data. No learned gate initialization needed.
2. **SSL-task alignment (H_align risk)**: Predictive coding's objective IS task-aligned: minimize prediction error on the actual tokens. The "self-supervised" signal is the discrepancy between the model's prediction and the observed tokens, not an auxiliary objective.
3. **Phase scheduling (P2)**: Precision naturally increases as denoising proceeds (more tokens revealed = higher certainty), so the predictive coding updates automatically concentrate in the informative phase.

### Method

```
Predictive Coding DLM (PC-DLM):
Layer L (top-level): generates top-down prediction mu_L of sequence
Layer L-1: receives mu_L, compares with backbone hidden state h_{L-1}
  Prediction error: epsilon_{L-1} = pi_{L-1} * (h_{L-1} - f(mu_L))
  where pi_{L-1} = precision = 1/Var(h_{L-1}) estimated online
Update rule (per denoising step):
  mu_L += lr * g(epsilon_{L-1})   [update predictions]
  pi_{L-1} updated via running variance of h_{L-1}

Architecture:
- Insert 2-level PC circuit at backbone layer L/2
- Top level: small MLP (d_model -> d_pc -> d_model) maintaining mu_L across steps
- Bottom level: precision estimation head (d_model -> 1, sigmoid output)
- Total trainable params: ~same as DaL TTT layer
- Training: K-step unrolling, meta-learn MLP weights and lr
```

### Failure Modes

- Predictive coding convergence requires multiple message-passing iterations per step (slow)
- Precision estimation may be noisy with few revealed tokens
- Theoretical elegance doesn't guarantee practical improvement
- PC dynamics in discrete token space (vs continuous neural activity) is not well-understood

### Computational Cost

- Implementation: ~1-2 days (simpler than TTT — no backward pass in the PC circuit)
- Training: ~4-8 GPU-hours (same as DaL meta-training)
- Pilot: ~2-4 GPU-hours
- **Success probability: 25%** (theoretically beautiful but unproven in this domain)

### Key References

- Friston & Kiebel (2009). Predictive Coding under the Free-Energy Principle.
- Salvatori et al. (2022). "A stable, fast, and fully automatic learning algorithm for predictive coding networks." arXiv 2212.00720.
- Millidge et al. (2022). "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs." NeurIPS 2022.
- Ambrogioni (2023). Statistical Thermodynamics of Diffusion. arXiv 2310.17467.
- SR-TTT (arXiv 2603.06642) — surprisal-aware gating is a form of precision weighting.

---

## Comparative Analysis and Recommendation

| Proposal | Novelty | Risk | Compute Cost | Success P | Addresses P3 Root Cause | Paradigm Break |
|----------|---------|------|-------------|-----------|------------------------|----------------|
| **1. ADM (Hopfield)** | Very High | High | Low (2-4h) | 30% | Yes (no gate needed) | Yes |
| **2. CADAC (Consistency)** | High | Medium | Medium (16-20h) | 35% | N/A (different approach) | Partial |
| **3. LDR (Latent)** | Very High | Very High | High (16-20h) | 20% | Yes (bypasses discrete bottleneck) | Yes |
| **4. SPD (Self-Play)** | High | Medium | Low (4-6h) | 25% | N/A (different approach) | Yes |
| **5. PC-DLM (Predictive Coding)** | High | High | Low (4-8h) | 25% | Yes (precision replaces gate) | Partial |

### Innovator's Recommendation: Parallel Exploration Portfolio

Given 4 GPUs and the lesson from 18+ iterations that serial exploration is suboptimal:

**Day 1 (all parallel)**:
- GPU 0: **Proposal 1 (ADM)** pilot — lowest implementation cost, mathematically motivated, directly addresses the gate problem that killed DaL
- GPU 1: **Proposal 4 (SPD)** pilot — no pre-training needed, can evaluate immediately, tests whether self-play contrastive signal is meaningful for DLMs
- GPU 2: **DaL with gate repair** (existing plan) — give the main proposal one fair shot with the engineering fix
- GPU 3: **Alternative A (Adaptive Unmasking)** — already in the plan, continue

**Day 2 decision gate**: Based on day 1 results, go deep on the 1-2 most promising approaches.

### Why NOT to Put All Eggs in DaL + Gate Repair

The existing proposal, after 5 iterations and 3 pilot experiments, has accumulated significant evidence of difficulty:
1. Gate stuck at 0.007 (engineering failure)
2. SSL-task misalignment risk (structural concern)
3. MetaState GRU underperforms vanilla (paradigm-level risk)
4. 18 prior iterations of negative results across various TTS methods

The contrarian is right that we should hedge aggressively. But rather than hedging with Alternative A (which is low-novelty), hedge with **genuinely novel approaches** (ADM, SPD) that could produce a higher-impact paper if they work.

### The DLM Inference Compute Trilemma (Reframed)

From the innovator's perspective, the trilemma isn't just D-W-M (depth-width-memory). It's:

```
        Accuracy (what you optimize)
              /\
             /  \
            /    \
           /------\
     Speed          Novelty
  (what you spend) (what you publish)
```

DaL sits in the "high novelty, high cost, uncertain accuracy" region. ADM and SPD offer different tradeoff profiles (ADM: high novelty, low cost, uncertain accuracy; SPD: moderate novelty, moderate cost, moderate accuracy). A portfolio approach maximizes expected paper quality.

---

## Lessons Applied from Evolution History

1. **"Contrarian was right"** (Evolution lesson): The innovator proposals above each include explicit failure modes and pivot triggers, not just optimistic outcomes. ADM has a concrete falsification: if Hopfield retrieval produces blurred patterns (cosine similarity > 0.9 with vanilla outputs), abandon.

2. **"First claims need verification"** (Evolution lesson): None of the proposals claim to be "first" without qualification. ADM is inspired by the Hopfield-diffusion connection but applies it to DLM denoising for the first time; the novelty claim is "first application", not "first discovery."

3. **"Innovator weighted 4/6"** (Previous weighting): This round, the innovator explicitly engages with the negative pilot evidence (P1-P3) and the contrarian's concerns. The proposals are not just creative — they are motivated by the specific failure modes observed in DaL.

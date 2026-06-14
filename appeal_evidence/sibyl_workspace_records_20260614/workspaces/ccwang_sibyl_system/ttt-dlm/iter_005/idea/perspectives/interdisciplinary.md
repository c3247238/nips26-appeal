# Interdisciplinary Perspective: Cross-Domain Structural Analogies for DLM Inference-Time Compute Scaling

**Agent**: sibyl-interdisciplinary
**Date**: 2026-03-11
**Topic**: Masked Diffusion Language Model Inference-Time Compute Scaling (ReMask-Retry / TTT / TCR)
**Iteration**: 5 (post-pilot evidence round)

---

## Preamble: Why Look Beyond ML?

The DaL proposal inserts TTT layers into frozen DLMs, treating iterative denoising as self-supervised learning. Pilot results show the learning mechanism works (SSL loss -52.7%) but fails to translate to downstream performance (accuracy -1.0pp, gate stuck at 0.007). Before debugging this within the ML framework alone, we should ask: **are there known systems in other sciences that perform iterative refinement with progressive information revelation, and what do they teach us about when and why such systems succeed or fail?**

The answer is yes -- at least four distinct scientific domains contain deep structural isomorphisms with the DaL/DLM denoising paradigm. Each suggests specific, testable modifications to the current approach.

---

## Analogy 1: Turbo Decoding and the Extrinsic Information Principle (Information Theory / Coding Theory)

### The Structural Correspondence

Turbo codes (Berrou et al., 1993) achieve near-Shannon-limit performance through **iterative exchange of extrinsic information** between two component decoders. The structural mapping to DLM denoising with DaL is remarkably precise:

| Turbo Decoding | DLM Denoising with DaL |
|---------------|----------------------|
| Received noisy codeword | Initial fully-masked sequence |
| Component decoder 1 (BCJR) | Frozen DLM backbone |
| Component decoder 2 (BCJR) | TTT layer |
| Interleaver | Mask schedule (permuted revelation order) |
| Extrinsic information exchange | Residual gate between backbone and TTT |
| Iterative decoding rounds | Denoising steps |
| Convergence to correct codeword | Convergence to coherent text |

The critical insight from turbo coding: **convergence depends on the two decoders exchanging only extrinsic (new) information, not intrinsic (already known) information**. If decoder 2 simply echoes what decoder 1 already computed, the iterations are wasted -- the system enters a fixed point without improving. This is precisely the "Turbo cliff" phenomenon, where performance suddenly collapses when the signal-to-noise ratio drops below a critical threshold (Hagenauer et al., 1996; Richardson & Urbanke, 2001).

### Why This Explains the P3 Failure

The P3 pilot showed SSL loss dropping 52.7% while task accuracy degraded. In turbo coding terms, this is a classic **intrinsic information feedback loop**: the TTT layer is learning to predict what the backbone already knows (revealed tokens), not providing complementary information about masked positions. The gate staying at 0.007 is a symptom, not the cause -- even with an open gate, if the TTT layer's output is highly correlated with the backbone's output (cosine similarity > 0.7), the information gain per iteration is negligible.

**Turbo coding theory makes a specific prediction**: the TTT output must be **decorrelated** from the backbone output to provide useful iterative refinement. This is the "extrinsic information principle" -- each component should contribute only the information that the other component cannot compute on its own.

### Concrete Design Transplant: Extrinsic Information Separation

Directly transplant the turbo decoding architecture:

1. **Extrinsic-only gating**: Instead of `output = backbone + gate * TTT_output`, use `output = backbone + gate * (TTT_output - proj(TTT_output, backbone_output))`, where `proj` is the projection onto the backbone output subspace. This ensures the TTT contribution is orthogonal to what the backbone already provides.

2. **EXIT chart analysis**: Borrow the EXtrinsic Information Transfer (EXIT) chart framework (ten Brink, 2001) to visualize the iterative convergence of the backbone-TTT system. Plot mutual information I_E(TTT) vs I_A(backbone) across denoising steps. If the two curves do not form an open tunnel, the system cannot converge -- this predicts failure before expensive training.

3. **Interleaver design**: In turbo codes, the interleaver's quality determines convergence. The analogous object in DLMs is the **mask schedule** (which tokens are revealed at each step). Current DLMs use random or confidence-based unmasking. Turbo coding theory suggests the mask schedule should maximize the "interleaver gain" -- reveal tokens that provide maximum extrinsic information to the TTT layer about the still-masked positions.

### Literature Grounding

- Berrou, Glavieux & Thitimajshima (1993). "Near Shannon limit error-correcting coding." The original turbo code paper establishing iterative extrinsic information exchange.
- ten Brink (2001). "Convergence behavior of iteratively decoded parallel concatenated codes." EXIT chart framework for predicting iterative decoder convergence.
- Richardson & Urbanke (2001). "The capacity of low-density parity-check codes under message-passing decoding." Density evolution analysis of iterative decoding, directly applicable to analyzing DaL convergence.
- Jiang et al. (2019, arXiv 1903.02295). "DeepTurbo: Deep Turbo Decoder." Neural turbo decoder demonstrating that the iterative extrinsic information principle can be learned end-to-end.
- Song et al. (2025, arXiv 2505.06175). "Turbo-ICL: In-Context Learning-Based Turbo Equalization." Recent work applying turbo-style iterative refinement with Transformer-based models, incorporating extrinsic information from decoder feedback as additional context.
- Hasan et al. (2026, arXiv 2601.10403). "Discrete Feynman-Kac Correctors." SMC-based framework for controlling discrete masked diffusion at inference time, including annealing -- demonstrates the connection between iterative correction and sampling in discrete diffusion.

### Testable Prediction

**H_turbo**: Measuring the cosine similarity between TTT output and backbone output at each denoising step will reveal similarity > 0.7 in the current DaL setup (explaining P3 failure). After applying extrinsic-only gating, similarity should drop below 0.3, and the accuracy improvement should become positive. Estimated compute: 2 GPU-hours for diagnostic, 4 GPU-hours for extrinsic gating implementation.

### Success Probability and Cost

- **Probability**: 40% that extrinsic separation alone fixes the P3 failure pattern
- **Compute**: 6 GPU-hours for full diagnostic + implementation
- **Novelty**: High -- no prior work applies the turbo extrinsic information principle to DLM inference-time adaptation

---

## Analogy 2: Predictive Coding and the Free Energy Principle (Neuroscience / Computational Neuroscience)

### The Structural Correspondence

The brain's cortical hierarchy implements **predictive coding** (Rao & Ballard, 1999; Friston & Kiebel, 2009): higher cortical areas generate top-down predictions of lower-area activity, and only the **prediction errors** (residuals) propagate upward. This iterative error-correction process minimizes **variational free energy** -- a bound on the negative log-evidence of the brain's generative model.

| Predictive Coding | DLM Denoising with DaL |
|-------------------|----------------------|
| Cortical hierarchy (V1 -> V2 -> V4 -> IT) | Transformer layers (L1 -> L2 -> ... -> Ln) |
| Top-down predictions | Backbone's token predictions |
| Bottom-up prediction errors | Difference between revealed tokens and predictions |
| Lateral recurrence (within-level iteration) | TTT gradient updates within denoising step |
| Temporal unfolding (across saccades) | Denoising steps (across mask ratios) |
| Free energy minimization | SSL loss minimization |
| Precision weighting of prediction errors | DaL's precision-weighted loss (pi_i = 1/Var[p]) |

The correspondence is not superficial. Friston's free energy principle states that biological systems minimize the quantity:

```
F = E_q[log q(z) - log p(x,z)] = KL[q(z)||p(z|x)] + (-log p(x))
```

DaL's SSL loss is formally a **prediction error** term, and the TTT gradient update is a **recognition model update** -- exactly the E-step of variational EM. The precision weighting already proposed in DaL (Section 2.2 of the proposal) is the neural implementation of **precision-weighted prediction error**, the core computational primitive of predictive coding (Feldman & Friston, 2010).

### What Predictive Coding Teaches That DaL Currently Misses

**1. Hierarchical prediction errors, not flat SSL loss.** In predictive coding, prediction errors are computed at **every level** of the hierarchy, not just at the output. The current DaL computes SSL loss only at the TTT layer's output (one level). Predictive coding theory predicts that distributing TTT updates across multiple layers -- each minimizing its own local prediction error -- would be more effective than a single TTT layer with a global SSL loss.

**2. Precision as a learnable, context-dependent quantity.** DaL uses precision pi_i = 1/Var[p(x_i|x_t)] as a fixed function of model uncertainty. In predictive coding, precision is itself **learned and context-dependent** -- it modulates the gain of prediction error signals based on expected reliability (Feldman & Friston, 2010). A learnable precision network (small MLP predicting pi_i from the hidden state) would be a closer transplant.

**3. Active inference and action selection.** In the free energy framework, agents don't just update beliefs (perception) -- they also select actions that minimize expected free energy (active inference). The DLM analog of "action" is the **unmasking decision** -- which tokens to reveal next. Current DLMs use random or confidence-based unmasking. Active inference predicts that the optimal unmasking policy should select tokens that maximally reduce expected free energy (prediction error) for the remaining masked positions. This connects directly to Alternative A (information-gain unmasking) in the proposal.

### Literature Grounding

- Friston & Kiebel (2009). "Predictive coding under the free-energy principle." Foundational paper on hierarchical predictive coding. (Royal Society B, 364(1521), 1211-1228)
- Feldman & Friston (2010). "Attention, uncertainty, and free-energy." Precision weighting in predictive coding; directly justifies DaL's pi_i weighting and suggests making it learnable.
- Millidge (2021, arXiv 2107.00140). "Applications of the free energy principle to machine learning and neuroscience." Bridges predictive coding to modern deep learning, including variational inference connections.
- Ali et al. (2022). "Predictive coding is a consequence of energy efficiency in recurrent neural networks." Shows predictive coding emerges naturally in energy-efficient recurrent networks (Patterns, Cell Press).
- Shaw & Berndt (2025). "A Neuro-Inspired Computational Framework for AGI: Predictive Coding, Active Inference, and Free Energy Minimisation." Recent framework connecting free energy minimization to LLM architectures.
- Salvatori et al. (2025, ScienceDirect). "A survey on neuro-mimetic deep learning via predictive coding." Comprehensive 2025 survey establishing predictive coding as a viable alternative to backpropagation for iterative inference in deep networks.
- Boutin et al. (2024, PLOS Comp Bio). "Dynamic predictive coding: A model of hierarchical sequence learning and prediction in the neocortex." Demonstrates how temporal prediction errors drive sequence learning -- directly relevant to denoising as a temporal process.

### Concrete Design Transplant: Hierarchical Predictive TTT

1. **Multi-level TTT**: Insert TTT modules at 2-3 layers (e.g., layers L/4, L/2, 3L/4) instead of just one. Each TTT module minimizes its local prediction error (difference between its hidden state and the next layer's top-down prediction). This creates a proper predictive coding hierarchy.

2. **Learnable precision network**: Replace fixed pi_i with a small MLP: `pi_i = softplus(W_pi * h_i + b_pi)`, trained jointly with the TTT layer. This allows the system to learn which prediction errors are informative vs. noisy.

3. **Active unmasking**: At each denoising step, select the next tokens to unmask by computing expected free energy reduction: `F_reduction(i) = E[DeltaF | unmask token i]`. Approximate via the TTT layer's gradient magnitude at each masked position.

### Testable Prediction

**H_pred_coding**: Multi-level TTT (3 layers) with learnable precision achieves higher accuracy than single-level TTT with fixed precision, under matched parameter budget. The effect should be especially pronounced on longer sequences (GSM8K) where hierarchical structure matters. Estimated delta: +1-2% accuracy.

### Success Probability and Cost

- **Probability**: 35% for multi-level TTT improving over single-level (moderate -- adds complexity)
- **Compute**: 8-12 GPU-hours for 3-layer TTT training
- **Novelty**: Moderate -- predictive coding in deep learning is known, but applying it to DLM denoising TTT is new

---

## Analogy 3: Simulated Annealing and Phase Transitions (Statistical Physics)

### The Structural Correspondence

DLM denoising is mathematically analogous to **simulated annealing** (Kirkpatrick, Gelatt & Vecchi, 1983) -- a global optimization method inspired by the physical process of slowly cooling a material to find its minimum-energy crystalline state.

| Simulated Annealing | DLM Denoising |
|--------------------|---------------|
| Temperature T | Mask ratio r (high r = high T) |
| Energy landscape E(x) | Negative log-likelihood -log p(x) |
| Metropolis-Hastings updates | Token prediction + sampling at each step |
| Cooling schedule T(t) | Mask schedule r(t) |
| Phase transition (liquid -> solid) | Transition from noise to coherent text |
| Critical temperature T_c | Critical mask ratio r_c ~ 0.6 (P2 evidence) |
| Quenching (too-fast cooling) | Too few denoising steps |
| Annealing (slow cooling) | Many denoising steps |

This analogy is not just metaphorical. Recent work makes it rigorous:

- Toji et al. (2024, arXiv 2412.01212) demonstrate an unambiguous **Berezinskii-Kosterlitz-Thouless (BKT) phase transition** in a context-sensitive probabilistic language model, showing that critical properties in natural language may be generically explained by BKT phases rather than requiring fine-tuning.
- Li, Karan & Chen (2025, arXiv 2502.00921) show that key features of generated output are decided in narrow "critical windows" during the generation process -- both in diffusion models and autoregressive LLMs -- connecting this to the all-or-nothing phenomenon from statistical inference.
- Arnold et al. (2024, arXiv 2405.17088) develop statistical methods for automated detection of phase transitions in LLM output distributions, showing that temperature-like parameters induce genuine phase transitions.
- Alpay & Kilictas (2026, arXiv 2601.19942) formalize Transformer forward passes as renormalization group flows, showing phase transitions in the effective dimensionality of hidden states at critical depths.
- Hasan et al. (2026, arXiv 2601.10403) derive SMC algorithms for discrete masked diffusion that explicitly perform **annealing** of the sampled distribution, connecting simulated annealing to discrete diffusion inference.

### What Statistical Physics Teaches About DaL's Phase-Transition Scheduling

**1. Critical slowing down near T_c.** In physical systems, dynamics slow dramatically near the critical temperature. The analog for DLMs: near the critical mask ratio r_c ~ 0.6 (where P2 shows maximum gradient SNR), the model is most sensitive to perturbations but also most susceptible to getting stuck. TTT updates at this point have maximum leverage but also maximum risk.

**2. The annealing schedule must be logarithmic, not linear.** Optimal simulated annealing requires T(t) = T_0 / log(1+t) (Geman & Geman, 1984). Current DLM mask schedules are typically linear or cosine. The statistical physics analogy predicts that a **logarithmic mask schedule** near the critical zone would improve generation quality, especially for reasoning tasks where the "energy landscape" is rugged.

**3. Quench-anneal hybrid.** In metallurgy, materials are sometimes quenched rapidly to a metastable state, then slowly annealed. The DLM analog: use few denoising steps (quench) to get close to a solution, then apply TTT-enriched slow refinement (anneal) near r_c. This is computationally cheaper than full TTT at every step.

**4. Replica symmetry breaking.** In spin glasses (Parisi, 1979), the energy landscape fragments into exponentially many metastable states below T_c. Carson & Reisizadeh (2025, arXiv 2506.04374) identify four latent reasoning regimes in Transformer hidden-state trajectories, modeled as a switching linear dynamical system. The DLM denoising process may similarly traverse multiple reasoning regimes, and TTT's role should be to help the system navigate between these regimes rather than just refining within one.

### Concrete Design Transplant: Annealing-Aware TTT Scheduling

1. **Logarithmic mask schedule in the critical zone**: Replace linear/cosine scheduling with T(t) = T_0/log(1+t) for mask ratio in [0.3, 0.7]. Keep uniform scheduling outside this zone.

2. **Quench-then-anneal protocol**: First 60% of denoising steps: fast, no TTT (quench to approximate solution). Last 40%: slow, TTT-enriched refinement (anneal to precise solution). This concentrates TTT compute where it has maximum effect.

3. **Replica-aware multi-modal sampling**: At the critical mask ratio, generate K parallel denoising trajectories with TTT, then select the best via a reward model. This is the DLM analog of parallel tempering in MCMC.

### Testable Prediction

**H_anneal**: The quench-then-anneal protocol (TTT only in last 40% of steps) achieves >= 95% of full-TTT accuracy at 40% of the TTT compute cost, and logarithmic scheduling in the critical zone outperforms linear scheduling by >= 0.5% accuracy on reasoning benchmarks. This refines and extends H5 from the main proposal.

### Success Probability and Cost

- **Probability**: 50% -- P2 evidence already confirms the phase structure; this is an optimization on top of confirmed structure
- **Compute**: 4 GPU-hours for schedule ablation
- **Novelty**: Moderate -- phase-aware scheduling is already in the proposal, but the logarithmic schedule and quench-anneal protocol are new

---

## Analogy 4: Somatic Hypermutation and Affinity Maturation (Immunology / Evolutionary Biology)

### The Structural Correspondence

The adaptive immune system generates high-affinity antibodies through **affinity maturation** -- a process of iterative mutation and selection in germinal centers (Victora & Nussenzweig, 2012). The structural mapping to DLM denoising with TTT is unexpectedly precise:

| Affinity Maturation | DLM Denoising with DaL |
|---------------------|----------------------|
| Naive B cell (low affinity) | Fully masked sequence (random) |
| Somatic hypermutation (SHM) | Token prediction updates at each step |
| Affinity selection (by T follicular helper cells) | Confidence-based unmasking |
| Clonal expansion of high-affinity variants | Best-of-N sampling |
| Germinal center dark zone (mutation) | High mask ratio (exploration) |
| Germinal center light zone (selection) | Low mask ratio (exploitation) |
| Memory B cells (long-term storage) | TTT fast weights (cross-step memory) |
| Successive rounds of SHM + selection | Successive denoising steps |

The biological insight: affinity maturation succeeds because it alternates between **diversification** (SHM in the dark zone) and **selection** (affinity testing in the light zone). The mutation rate is regulated -- too high and beneficial mutations are destroyed; too low and the search stalls. Critically, the immune system does NOT use gradient descent -- it uses **directed random mutation biased by sequence context** (AID enzyme targeting hotspot motifs) followed by **fitness selection**.

### What Immunology Teaches That DaL Currently Misses

**1. Mutation rate scheduling mirrors mask scheduling.** SHM rates peak during the first few rounds of germinal center reactions, then decline as antibodies approach optimal affinity (Tas et al., 2016). This is directly analogous to DaL's phase-transition scheduling -- but the biological system uses a **fitness-dependent** (not time-dependent) schedule. When affinity improvement stalls, mutation rate increases; when affinity is improving rapidly, mutation rate decreases. DaL should similarly condition TTT learning rate on the rate of SSL loss improvement, not just on mask ratio.

**2. Clonal diversification prevents premature convergence.** In germinal centers, multiple B cell clones compete simultaneously, preventing the system from getting stuck in a local optimum. Current DaL uses a single denoising trajectory. The immunological analog suggests maintaining K parallel TTT weight trajectories at early denoising steps (when mask ratio is high = dark zone), then selecting the best trajectory at a later step (light zone selection). This is more sophisticated than simple Best-of-N because the selection happens at an intermediate step, not just at the final output.

**3. Antigenic sin: past adaptations can harm future ones.** "Original antigenic sin" occurs when memory B cells from a previous infection dominate the response to a new, related pathogen, preventing optimal adaptation. The TTT analog: fast weights learned during early denoising steps (when the sequence is mostly masked and predictions are unreliable) may create **harmful inductive biases** that persist into later steps where better information is available. This could explain P3's failure -- early TTT updates learn from low-quality signal and then resist correction.

**Mitigation**: Implement **fast weight decay** that is high in early steps and low in later steps, analogous to how the immune system clears low-affinity clones. Alternatively, implement a "checkpoint-and-restart" mechanism: save fast weights at the critical mask ratio, discard weights learned before that point, and restart TTT from the checkpoint.

### Literature Grounding

- Victora & Nussenzweig (2012). "Germinal centers." Annual Review of Immunology. The canonical review of affinity maturation cycles.
- Tas et al. (2016). "Visualizing antibody affinity maturation in germinal centers." Science. Direct observation of mutation-selection cycles.
- Tang, Krantsevich & MacCarthy (2022). "Deep learning model of somatic hypermutation reveals importance of sequence context beyond hotspot targeting." iScience / Cell Press. ML model of SHM context-dependence.
- Irvine & Reddy (2024). "Advancing antibody engineering through synthetic evolution and machine learning." J Immunol. Bridges evolutionary maturation with ML-based iterative optimization.
- Jin et al. (2021, arXiv 2110.04624). "Iterative Refinement Graph Neural Network for Antibody Sequence-Structure Co-design." Directly applies iterative refinement (denoising-like) to antibody CDR design, validating the structural correspondence.
- Kucharavy, El Mhamdi & Guerraoui (2020, arXiv 2006.04720). "Host-Pathogen Co-evolution Inspired Algorithm Enables Robust GAN Training." Demonstrates that immune system co-evolution principles can improve ML training stability.

### Concrete Design Transplant: Affinity-Maturation-Inspired DaL

1. **Fitness-dependent learning rate**: Set TTT learning rate proportional to the rate of SSL loss improvement: `eta_t = eta_base * max(1, -d(L_ssl)/dt / threshold)`. When SSL loss is improving rapidly, reduce learning rate (exploitation); when it stalls, increase it (exploration).

2. **Clonal diversification at high mask ratios**: At mask ratio > 0.5, maintain K=4 parallel TTT weight trajectories (different random initializations). At mask ratio = 0.5, select the trajectory with lowest SSL loss and discard the rest. This prevents premature convergence without K-fold compute cost at every step.

3. **Antigenic sin prevention via checkpoint-restart**: Reset fast weights at the critical mask ratio r_c = 0.6, discarding all learning from the "dark zone" (high mask ratio). Restart TTT from fresh initialization at r_c, using only the high-quality signal available at lower mask ratios. This directly addresses the concern that early low-quality TTT updates harm later performance.

### Testable Prediction

**H_affinity**: Checkpoint-restart at r_c = 0.6 (discarding early TTT weights) outperforms continuous TTT across all steps by >= 1% accuracy on reasoning benchmarks, because early TTT updates from low-quality signal create harmful inductive biases. This is testable as a simple ablation with zero additional parameters.

### Success Probability and Cost

- **Probability**: 45% -- the "antigenic sin" explanation for P3 failure is independently plausible alongside the gate failure explanation
- **Compute**: 3 GPU-hours for checkpoint-restart ablation
- **Novelty**: High -- no prior work applies affinity maturation principles to test-time training scheduling

---

## Synthesis: Cross-Disciplinary Convergence on DaL's Failure Modes

The four analogies converge on a unified diagnosis of why DaL's P3 pilot failed and what to do about it:

| Analogy | Diagnosis of P3 Failure | Predicted Fix |
|---------|------------------------|---------------|
| **Turbo coding** | TTT output is not extrinsic to backbone output | Extrinsic-only gating (orthogonal projection) |
| **Predictive coding** | Single-level prediction error is insufficient; precision is not learned | Multi-level TTT + learnable precision |
| **Simulated annealing** | TTT applied uniformly instead of concentrated near phase transition | Quench-then-anneal protocol |
| **Affinity maturation** | Early low-quality TTT updates create harmful inductive bias | Checkpoint-restart at r_c |

Remarkably, these four fixes are **mutually compatible and independently testable**. They form a natural ablation sequence:

1. **First** (cheapest diagnostic): Measure cosine similarity between TTT and backbone outputs (turbo coding diagnostic). If > 0.7, extrinsic separation is the priority fix. (2 GPU-hours)
2. **Second**: Implement checkpoint-restart at r_c (affinity maturation fix). Zero additional parameters. (3 GPU-hours)
3. **Third**: Implement quench-then-anneal protocol (statistical physics fix). Compatible with checkpoint-restart. (4 GPU-hours)
4. **Fourth** (most expensive): Multi-level TTT with learnable precision (predictive coding fix). Increases parameter count. (8-12 GPU-hours)

### Integration with Existing Proposal

The interdisciplinary perspective strengthens the existing DaL proposal in three ways:

1. **D0c gets a turbo-coding-grounded diagnostic**: The SSL-task misalignment can be predicted *a priori* by measuring extrinsic information content, before expensive training.

2. **Phase-transition scheduling (H5) gains statistical physics rigor**: The logarithmic schedule and quench-anneal protocol are principled improvements over the current heuristic [0.1, 0.7] window.

3. **A new failure mode is identified**: "Antigenic sin" (early low-quality TTT updates persisting into later steps) is a distinct failure mode from gate failure, and has a zero-cost fix (checkpoint-restart).

### Combined Experimental Plan

| Experiment | Source Analogy | GPU-hours | Integrate With |
|-----------|----------------|-----------|---------------|
| E_turbo: Cosine similarity diagnostic | Turbo coding | 2 | D0c (proposal Phase 0) |
| E_anneal: Quench-then-anneal schedule | Stat physics | 4 | H5 (proposal Phase 1) |
| E_restart: Checkpoint-restart at r_c | Immunology | 3 | H_gate (proposal Phase 0) |
| E_extrinsic: Orthogonal projection gate | Turbo coding | 4 | Gate repair (proposal Phase 0) |
| E_hier_ttt: Multi-level TTT + precision | Pred coding | 10 | H1 (proposal Phase 1) |
| **Total** | | **23** | |

### Risk Assessment

| Proposal | P(success) | Impact if successful | Compute |
|----------|-----------|---------------------|---------|
| Extrinsic gating (turbo) | 40% | High -- directly addresses information redundancy | 4h |
| Checkpoint-restart (immuno) | 45% | Medium -- simple ablation, novel insight | 3h |
| Quench-anneal (physics) | 50% | Medium -- refines existing schedule | 4h |
| Multi-level TTT (neuro) | 35% | High -- fundamentally new architecture variant | 10h |
| At least one succeeds | ~87% | | 23h |

The combined probability that at least one interdisciplinary-inspired fix provides a measurable improvement is approximately 87%, making this a high-value investment at 23 GPU-hours total.

---

## References

### Information Theory / Coding Theory
1. Berrou, C., Glavieux, A. & Thitimajshima, P. (1993). Near Shannon limit error-correcting coding and decoding: Turbo-codes. ICC.
2. ten Brink, S. (2001). Convergence behavior of iteratively decoded parallel concatenated codes. IEEE Trans. Comm.
3. Richardson, T. & Urbanke, R. (2001). The capacity of low-density parity-check codes under message-passing decoding. IEEE Trans. IT.
4. Jiang, Y. et al. (2019). DeepTurbo: Deep Turbo Decoder. arXiv 1903.02295.
5. Song, Z. et al. (2025). Turbo-ICL: In-Context Learning-Based Turbo Equalization. arXiv 2505.06175.
6. Hasan, M. et al. (2026). Discrete Feynman-Kac Correctors. arXiv 2601.10403.

### Neuroscience / Predictive Coding
7. Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex. Nature Neuroscience.
8. Friston, K. & Kiebel, S. (2009). Predictive coding under the free-energy principle. Phil Trans R Soc B, 364(1521), 1211-1228.
9. Feldman, H. & Friston, K. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience.
10. Millidge, B. (2021). Applications of the free energy principle to machine learning and neuroscience. arXiv 2107.00140.
11. Ali, A. et al. (2022). Predictive coding is a consequence of energy efficiency in recurrent neural networks. Patterns (Cell Press).
12. Salvatori, T. et al. (2025). A survey on neuro-mimetic deep learning via predictive coding. ScienceDirect.
13. Shaw, A.D. & Berndt, L.C.S. (2025). A Neuro-Inspired Computational Framework for AGI. cpnslab.com.

### Statistical Physics / Phase Transitions
14. Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by simulated annealing. Science.
15. Geman, S. & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Trans. PAMI.
16. Toji, Y. et al. (2024). Berezinskii-Kosterlitz-Thouless transition in a context-sensitive random language model. arXiv 2412.01212.
17. Li, M., Karan, A. & Chen, S. (2025). Blink of an eye: critical windows in generative models. arXiv 2502.00921.
18. Arnold, J. et al. (2024). Phase Transitions in the Output Distribution of Large Language Models. arXiv 2405.17088.
19. Alpay, F. & Kilictas, B. (2026). Latent Object Permanence: Topological Phase Transitions in Deep Transformer Manifolds. arXiv 2601.19942.
20. Carson, J.D. & Reisizadeh, A. (2025). A Statistical Physics of Language Model Reasoning. arXiv 2506.04374.

### Immunology / Evolutionary Biology
21. Victora, G.D. & Nussenzweig, M.C. (2012). Germinal centers. Annual Review of Immunology.
22. Tas, J.M.J. et al. (2016). Visualizing antibody affinity maturation in germinal centers. Science.
23. Tang, C. et al. (2022). Deep learning model of somatic hypermutation. iScience (Cell Press).
24. Jin, W. et al. (2021). Iterative Refinement GNN for Antibody Sequence-Structure Co-design. arXiv 2110.04624.
25. Irvine, E.B. & Reddy, S.T. (2024). Advancing antibody engineering through synthetic evolution and ML. J Immunol.

### Fast Weights / TTT
26. Irie, K. & Gershman, S.J. (2025). Fast weight programming and linear transformers: from ML to neurobiology. arXiv 2508.08435.
27. Chaudhary, S. (2025). Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity. arXiv 2510.21908.
28. Wang, K.A. et al. (2025). Test-time regression: a unifying framework for designing sequence models with associative memory. arXiv 2501.12352.
29. Hwang, H.S. et al. (2026). Reinforced Fast Weights with Next-Sequence Prediction. arXiv 2602.16704.
30. Zhao, T. & Jones, L. (2026). Fast-weight Product Key Memory. arXiv 2601.00671.
31. Zenke, F. & Neftci, E. (2020). Brain-Inspired Learning on Neuromorphic Substrates. arXiv 2010.11931.
32. Ziyin, L. et al. (2025). Heterosynaptic Circuits Are Universal Gradient Machines. arXiv 2505.02248.

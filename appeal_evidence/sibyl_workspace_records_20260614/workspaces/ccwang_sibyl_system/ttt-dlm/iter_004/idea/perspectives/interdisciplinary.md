# Interdisciplinary Perspectives: Test-Time Computation Scaling for Masked Diffusion Language Models

## Executive Summary

We propose three interdisciplinary research directions that transplant deep structural principles from statistical physics, neuroscience, and evolutionary biology into the domain of masked diffusion language models (MDLMs). Each direction addresses a fundamental limitation revealed by the project's 18 iterations: naive remasking strategies either cause text degeneration (repetition) or fail to improve quality at scale. The key insight is that **MDLMs' iterative denoising process is structurally isomorphic to well-studied physical and biological processes**, and the solutions discovered in those fields can be formally adapted.

---

## Direction 1: Simulated Annealing Schedules for Adaptive Remasking (Statistical Physics)

### The Analogy

The MDLM denoising process — where masked tokens are iteratively unmasked over T steps — is structurally isomorphic to **simulated annealing on an Ising model**. In both systems:

| MDLM Denoising | Ising Model / Simulated Annealing |
|---|---|
| Token position i with mask/unmask state | Spin site i with state +1/-1 |
| Denoising step t (high mask ratio → low) | Temperature T (high → low) |
| Model confidence p(x_i \| x_{masked}) | Boltzmann factor exp(-E/kT) |
| Remasking low-confidence tokens | Spin flip via Metropolis acceptance |
| Fixed linear unmasking schedule | Fixed linear cooling schedule |

The critical insight: **the project's ReMask-Retry failures mirror a known pathology in simulated annealing** — using a fixed linear cooling schedule on a rugged energy landscape leads to trapping in local minima (text degeneration/repetition) or failure to converge (PPL worsening on LLaDA-8B). The solution in statistical physics is well-established: **adaptive cooling schedules** (Tsallis & Stariolo, 1995; generalized simulated annealing).

### Structural Correspondence

1. **Energy function**: Define E(x) = -log p_θ(x) as the sequence-level energy under the MDLM. The denoising trajectory is a path through the discrete energy landscape.
2. **Temperature as mask ratio**: At denoising step t, the "temperature" is the fraction of remaining masked tokens. High temperature = many masked tokens = high exploration; low temperature = few masked = exploitation.
3. **Metropolis acceptance for remasking**: Instead of deterministically remasking the k lowest-confidence tokens (ReMask-Retry), accept/reject remasking decisions with probability min(1, exp(-ΔE/T_t)), where ΔE is the energy change from remasking token i and T_t is an adaptive temperature.

### Concrete Proposal: Annealing-Aware Remasking (AAR)

**Algorithm:**
1. Run standard MDLM denoising for t steps, producing partial sequence x_t
2. Compute per-token energy: e_i = -log p_θ(x_i | x_{t,\i})
3. Define remasking temperature T_remask(t) using an **adaptive schedule**:
   - **Logarithmic**: T(t) = T_0 / log(1 + t) — guarantees convergence to global minimum (Geman & Geman, 1984)
   - **Tsallis q-annealing**: T(t) = T_0 * (2^{1-q} - 1) / ((1+t)^{1-q} - 1) — faster convergence for discrete systems
   - **Reheat**: If ΔE variance drops below threshold (plateau detection), temporarily increase T to escape local minima
4. Accept remasking of token i with probability: p_remask(i) = sigmoid((e_i - e_threshold) / T_remask(t))
5. Re-denoise remasked positions

**Why this could work where ReMask-Retry failed:**
- ReMask-Retry uses a **fixed, deterministic** remasking policy → gets trapped in repetition attractors
- AAR introduces **stochastic acceptance** calibrated by an annealing schedule → can escape repetition basins while still converging
- The logarithmic schedule has a **theoretical convergence guarantee** to the global minimum of the energy landscape

**Grounding in existing cross-disciplinary work:**
- Sanokowski et al. (2025, arXiv:2502.08696) demonstrated that discrete diffusion models can be trained as samplers for Ising model benchmarks, outperforming autoregressive approaches — directly validating the MDLM-Ising isomorphism
- Ramachandran et al. (2025, arXiv:2511.00124) showed that diffusion sampling exhibits **phase transitions** detectable via cross-fluctuations from statistical physics, and that detecting these transitions boosts sampling efficiency
- Borysenko & Byshkin (2020, arXiv:2005.14605) proved that momentum SGD is equivalent to discretized Langevin dynamics with simulated annealing (CoolMomentum)
- The "Promises and Pitfalls of Generative Masked Language Modeling" (ICML 2024) establishes that MDLM decoding is formally a **k-Gibbs sampler** on an Ising-like undirected model, with theoretical separation results between dependent and independent sampling

### Experimental Plan

| Experiment | Model | Metric | Compute |
|---|---|---|---|
| Baseline: ReMask-Retry (fixed k) | Dream-7B | PPL + MAUVE + Distinct-N | 1 GPU, 2h |
| AAR-Log (logarithmic schedule) | Dream-7B | PPL + MAUVE + Distinct-N | 1 GPU, 3h |
| AAR-Tsallis (q=1.5, 2.0, 2.5) | Dream-7B | PPL + MAUVE + Distinct-N | 1 GPU, 4h |
| AAR-Reheat (plateau detection) | Dream-7B | PPL + MAUVE + Distinct-N | 1 GPU, 3h |
| Ablation: schedule comparison | Dream-7B | PPL + MAUVE + repetition rate | 1 GPU, 4h |
| Benchmark validation (HellaSwag/ARC) | Dream-7B | Accuracy | 2 GPU, 6h |

**Testable predictions:**
1. AAR-Log should eliminate the repetition degeneration seen in ReMask-Retry (Distinct-4 should not decrease)
2. AAR-Tsallis with q≈2 should outperform logarithmic on PPL while maintaining diversity (faster convergence)
3. Reheat should show the largest gains on long sequences (>256 tokens) where energy landscapes are more rugged
4. The optimal annealing schedule parameters should correlate with the model's perplexity landscape roughness

**Success probability:** 65% — The physics is sound and the isomorphism is tight; the main risk is that MDLM energy landscapes may be too high-dimensional for annealing theory guarantees to hold practically.

**Estimated compute:** ~22 GPU-hours total

---

## Direction 2: Hopfield Associative Memory as Test-Time Refinement Layer (Neuroscience)

### The Analogy

The brain does not generate language in a single feedforward pass. Instead, it uses **recurrent attractor dynamics** in associative memory networks to iteratively refine representations before committing to output. This is precisely the mechanism formalized by **modern Hopfield networks** (Ramsauer et al., 2020), which have been shown to be mathematically equivalent to transformer attention with exponential storage capacity.

The key structural correspondence:

| MDLM Denoising | Hopfield Attractor Dynamics |
|---|---|
| Partially masked sequence x_t | Partial/noisy query pattern |
| Denoising step t → t+1 | One step of energy minimization |
| Converged sequence x_0 | Retrieved memory (fixed point) |
| Multiple denoising trajectories | Multiple attractor basins |
| No memory across sequences | No persistent memory |

The critical gap in MDLMs: **each denoising trajectory is memoryless** — the model has no mechanism to learn from the current sequence's denoising history. This is why TTT (Test-Time Training) variants showed no statistical significance (p=0.88): they tried to add memory by fine-tuning weights, which is too slow and too global. Hopfield networks offer a **fast, local, content-addressable** memory that operates at the right timescale.

### Structural Correspondence (Deep)

Modern Hopfield networks store patterns as fixed points of the energy function:

E(x) = -log Σ_μ exp(⟨x, ξ_μ⟩)

where ξ_μ are stored patterns. The update rule x_{t+1} = softmax(Xξ^T) · ξ is **exactly transformer attention** (Ramsauer et al., 2020). But crucially:

1. **Continuous-time Hopfield** (arxiv:2502.10122) compresses discrete memories into continuous-time dynamics, enabling efficient storage of denoising trajectory history
2. **Dense Associative Memories** (Hu et al., NeurIPS 2024) provide provably optimal capacity scaling O(d^{n-1}) for n-th order interactions
3. **In-context denoising via attention** (Smart et al., 2025, arXiv:2502.05164) formally proves that attention-based denoising is equivalent to associative memory retrieval

### Concrete Proposal: Hopfield-Augmented Denoising (HAD)

**Architecture: Insert a Hopfield memory layer into the MDLM denoising loop:**

1. **Memory Write**: At each denoising step t, store the current (partial) sequence representation h_t and the model's confidence vector c_t as a memory entry: M_t = {(h_1,c_1), ..., (h_t,c_t)}
2. **Memory Read**: Before the next denoising step, query the Hopfield network with the current state to retrieve the most relevant past states. This implements **trajectory-aware denoising**: the model knows which token positions have been historically unstable.
3. **Gated Fusion**: Combine the standard MDLM prediction with the Hopfield-retrieved pattern via a learned gate: x̃_{t+1} = g · MDLM(x_t) + (1-g) · Hopfield(x_t, M_t)

**Why this addresses TTT's failure:**
- TTT adapts model **weights** (slow, global, destroys pre-trained knowledge)
- HAD adapts model **memory** (fast, local, preserves pre-trained weights)
- This mirrors the neuroscience distinction between **synaptic plasticity** (slow, permanent → TTT) and **working memory** (fast, transient → HAD)

**Grounding in existing work:**
- Behrouz et al. (2025, Titans, arXiv:2501.00663) demonstrated that a neural long-term memory module combined with attention achieves SOTA on language modeling — but applied to AR models, not MDLMs
- Sun et al. (2025, Associative Transformer, CVPR) showed Hopfield-based sparse attention improves vision transformers
- Chaudhary (2025, arXiv:2510.21908) proved that Hebbian plasticity in transformers enables faster in-sequence adaptation than static weights
- Jafari & Anbarjafari (2025, Equilibrium Transformers, arXiv:2511.21882) showed that iterative refinement via energy minimization in latent space **unifies diffusion language models and TTT as special cases** — directly motivating our approach
- Masumura & Taki (2025, arXiv:2511.20698) analyzed the role of hidden states in modern Hopfield networks within transformers

### Experimental Plan

| Experiment | Model | Metric | Compute |
|---|---|---|---|
| Baseline: Dream-7B standard | Dream-7B | PPL + MAUVE + benchmarks | 1 GPU, 2h |
| HAD-Simple (concat memory) | Dream-7B | PPL + MAUVE + benchmarks | 1 GPU, 4h |
| HAD-Hopfield (modern Hopfield) | Dream-7B | PPL + MAUVE + benchmarks | 2 GPU, 6h |
| HAD vs TTT-Linear comparison | Dream-7B | PPL + statistical test | 1 GPU, 3h |
| Memory capacity ablation (depth) | Dream-7B | PPL vs memory size | 1 GPU, 4h |
| Benchmark (HellaSwag/ARC/MMLU) | Dream-7B | Accuracy | 2 GPU, 8h |

**Testable predictions:**
1. HAD should show statistically significant PPL improvement (p < 0.05) where TTT showed p=0.88 — because memory is faster than weight adaptation
2. Improvement should scale with sequence length (longer = more memory entries = more refinement signal)
3. The Hopfield memory should preferentially store tokens at positions where the model's confidence fluctuated across denoising steps (instability → memorability, mirroring hippocampal surprise signals)
4. HAD should NOT cause repetition degeneration because the memory stores diverse trajectory states, not just high-confidence ones

**Success probability:** 50% — High theoretical elegance but implementation complexity is significant; the main risk is that MDLM's denoising trajectory may not produce sufficiently diverse memory patterns for Hopfield retrieval to add value beyond simple averaging.

**Estimated compute:** ~27 GPU-hours total

---

## Direction 3: Evolutionary Fitness Landscapes and Population-Based Denoising (Evolutionary Biology)

### The Analogy

Protein evolution navigates a high-dimensional **fitness landscape** through population-based search: maintaining a diverse population of variants, applying selection pressure (fitness), mutation (variation), and recombination (crossover). This is structurally isomorphic to the problem of generating high-quality text with MDLMs:

| MDLM Text Generation | Protein Evolution |
|---|---|
| One denoising trajectory = one sequence | One organism = one protein sequence |
| Best-of-N sampling (failed: +6.9% PPL) | Naive parallel evolution (genetic drift) |
| Model confidence as quality signal | Fitness function |
| Token-level remasking | Point mutation |
| No recombination mechanism | Crossover between fit individuals |

**The critical insight**: Best-of-N failed because it generates N **independent** trajectories and picks the best — this is equivalent to running N independent evolutionary lineages with no crossover. In evolutionary biology, it is well-established that **recombination between fit individuals** is essential for efficient search in rugged fitness landscapes (Fisher-Muller effect). The MDLMs lack this mechanism entirely.

### Structural Correspondence (Deep)

The connection between diffusion models and evolutionary algorithms has been recently formalized:

1. **Diffusion Models are Evolutionary Algorithms** (Zhang et al., 2024, arXiv:2410.02543): Proved mathematically that the denoising process inherently performs selection, mutation, and reproductive isolation
2. **Heuristically Adaptive Diffusion-Model Evolutionary Strategy** (Hartl et al., 2025, Advanced Science): Demonstrated practical integration of diffusion sampling with evolutionary strategies
3. **Reward-Guided Iterative Refinement** (Uehara et al., 2025, arXiv:2502.14944): Applied iterative noising-denoising with reward guidance for protein and DNA design — directly bridging the bio-sequence and text-generation domains

The deep structural parallel: In Wright's adaptive landscape theory, evolution proceeds by maintaining a **population** on the fitness landscape, with genetic drift enabling exploration of valleys between adaptive peaks. The masked diffusion denoising process, when run as a population, can implement this same dynamics — but only if we add **recombination** (crossover between partially-denoised sequences).

### Concrete Proposal: Evolutionary Denoising with Crossover (EDC)

**Algorithm:**
1. Initialize K denoising trajectories (population) from the same prompt
2. At each denoising step t:
   a. **Selection**: Rank trajectories by a fitness function f(x) (e.g., MDLM log-probability, external reward model, or self-consistency score)
   b. **Crossover**: For the top-K/2 trajectories, create offspring by exchanging unmasked tokens at **semantically coherent boundaries** (sentence/clause boundaries identified by the model's attention pattern)
   c. **Mutation**: Apply standard MDLM denoising to all masked positions
   d. **Elitism**: Carry forward the top-2 individuals unchanged
3. After T steps, return the highest-fitness individual

**Why this could work where Best-of-N failed:**
- Best-of-N = K independent lineages → genetic drift, no information sharing
- EDC = K interacting lineages with crossover → Fisher-Muller effect, efficient exploration
- Crossover at semantic boundaries prevents the "chimera" problem (incoherent text from naive token-level crossover)

**Grounding in existing cross-disciplinary work:**
- **Protein inverse folding**: Yi et al. (NeurIPS 2023) showed that discrete graph denoising diffusion with iterative refinement achieves SOTA on protein design — the same discrete denoising + iterative refinement paradigm
- **Mask-prior guided denoising** (Bai et al., Nature Machine Intelligence, 2025): Demonstrated that using structural priors to guide which positions to denoise first dramatically improves protein sequence design — analogous to our semantic-boundary-aware crossover
- **IterRef** (arXiv:2511.05562): Showed that iterative noising-denoising refinement with reward guidance achieves 2x improvement on toxicity reward with LLaDA-8B — validates the core mechanism but without population-based search
- **Self-Rewarding SMC** (arXiv:2602.01849, from spec references): Uses parallel particle trajectories with resampling — a sequential Monte Carlo approach that is formally related to but less efficient than evolutionary crossover

**Connection to immune system clonal selection** (secondary analogy):
The adaptive immune system uses **affinity maturation** — a process where B-cells undergo hypermutation and selection, with the highest-affinity variants being amplified. This maps onto our EDC: denoising trajectories undergo mutation (standard denoising), selection (fitness ranking), and clonal expansion (crossover with top individuals). The key insight from immunology: **somatic hypermutation is targeted to complementarity-determining regions** (CDRs), not uniform — analogous to our semantic-boundary-aware crossover targeting uncertain regions.

### Experimental Plan

| Experiment | Model | Metric | Compute |
|---|---|---|---|
| Baseline: Best-of-N (K=8) | Dream-7B | PPL + MAUVE + Distinct-N | 2 GPU, 4h |
| EDC-Uniform (random crossover) | Dream-7B | PPL + MAUVE + Distinct-N | 2 GPU, 6h |
| EDC-Semantic (boundary-aware) | Dream-7B | PPL + MAUVE + Distinct-N | 2 GPU, 6h |
| EDC + IterRef reward guidance | Dream-7B | PPL + MAUVE + benchmarks | 2 GPU, 8h |
| Population size ablation (K=4,8,16) | Dream-7B | PPL + compute efficiency | 2 GPU, 6h |
| Benchmark (HellaSwag/ARC) | Dream-7B | Accuracy | 2 GPU, 6h |

**Testable predictions:**
1. EDC-Semantic should outperform Best-of-N by >5% on MAUVE score (diversity + quality) at equal compute
2. EDC-Uniform should show intermediate performance (crossover helps, but chimera artifacts hurt)
3. Population diversity (measured by pairwise edit distance) should remain higher in EDC than Best-of-N throughout denoising (Fisher-Muller effect)
4. EDC's advantage should increase with sequence length (longer sequences = more rugged fitness landscape = more value from crossover)
5. EDC should NOT increase repetition rate (unlike ReMask-Retry) because crossover introduces variation

**Success probability:** 55% — The evolutionary analogy is the most actionable of the three proposals; the main risk is computational overhead (K populations × T steps) and the difficulty of defining semantically coherent crossover boundaries.

**Estimated compute:** ~36 GPU-hours total

---

## Synthesis: Unifying Framework

The three directions can be unified under a single theoretical framework: **the MDLM denoising process as search on a discrete energy landscape**.

```
                    Energy Landscape View
                    =====================

Direction 1 (Physics):    HOW to navigate   → Annealing schedule
Direction 2 (Neuro):      WHAT to remember  → Trajectory memory
Direction 3 (Evolution):  WHO navigates      → Population + crossover

                    Combined: AAR + HAD + EDC
                    = Population of memory-augmented denoisers
                      with adaptive annealing schedules
```

This unification is not merely metaphorical. The Equilibrium Transformer framework (Jafari & Anbarjafari, 2025) formally proves that iterative refinement via energy minimization **unifies deep equilibrium models, diffusion language models, and TTT** as special cases. Our three directions add three orthogonal axes of improvement:

1. **Schedule** (AAR): Adaptive temperature control prevents trapping
2. **Memory** (HAD): Trajectory history prevents revisiting bad states
3. **Population** (EDC): Crossover enables efficient exploration

### Recommended Priority

1. **Start with Direction 1 (AAR)** — lowest implementation cost, strongest theoretical guarantees, directly addresses the observed failure mode (repetition degeneration)
2. **Then Direction 3 (EDC)** — moderate cost, most novel contribution (no prior work on evolutionary crossover in MDLM denoising)
3. **Finally Direction 2 (HAD)** — highest novelty but highest implementation risk

### Total Estimated Compute

~85 GPU-hours across all three directions (feasible on 4 GPUs in ~21 hours).

---

## References

### Statistical Physics & Discrete Diffusion
- Sanokowski et al. (2025). "Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics." arXiv:2502.08696
- Ramachandran et al. (2025). "Cross-fluctuation phase transitions reveal sampling dynamics in diffusion models." arXiv:2511.00124
- Tsallis & Stariolo (1995). "Generalized Simulated Annealing." arXiv:cond-mat/9501047
- Borysenko & Byshkin (2020). "CoolMomentum: A Method for Stochastic Optimization by Langevin Dynamics with Simulated Annealing." arXiv:2005.14605

### Generative Masked Language Models & Gibbs Sampling
- Sahoo et al. (2024). "Simple and Effective Masked Diffusion Language Models." NeurIPS 2024.
- "Promises and Pitfalls of Generative Masked Language Modeling." ICML 2024 (arXiv:2407.21046)
- "Effective Test-Time Scaling of Discrete Diffusion through Iterative Refinement." arXiv:2511.05562

### Modern Hopfield Networks & Associative Memory
- Ramsauer et al. (2020). "Hopfield Networks is All You Need." ICLR 2021.
- Hu et al. (2024). "Provably Optimal Memory Capacity for Modern Hopfield Models." NeurIPS 2024.
- Smart et al. (2025). "In-context denoising with one-layer transformers: connections between attention and associative memory retrieval." arXiv:2502.05164
- Modern Hopfield Networks with Continuous-Time Memories. arXiv:2502.10122
- Masumura & Taki (2025). "On the Role of Hidden States of Modern Hopfield Network in Transformer." arXiv:2511.20698

### TTT & Test-Time Adaptation
- Behrouz et al. (2025). "Titans: Learning to Memorize at Test Time." arXiv:2501.00663
- Zhang et al. (2025). "Test-Time Training Done Right." arXiv:2505.23884
- Chaudhary (2025). "Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity." arXiv:2510.21908

### Energy-Based & Equilibrium Models
- Jafari & Anbarjafari (2025). "Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium." arXiv:2511.21882
- "Energy-Based Transformers are Scalable Learners and Thinkers." arXiv:2507.02092
- "Energy-Based Diffusion Language Models for Text Generation." arXiv:2410.21357

### Evolutionary Algorithms & Diffusion
- "Diffusion Models are Evolutionary Algorithms." arXiv:2410.02543
- Hartl et al. (2025). "Heuristically Adaptive Diffusion-Model Evolutionary Strategy." Advanced Science.
- Uehara et al. (2025). "Reward-Guided Iterative Refinement in Diffusion Models at Test-Time." arXiv:2502.14944

### Protein Design & Biological Sequence Generation
- Yi et al. (2023). "Graph denoising diffusion for inverse protein folding." NeurIPS 2023.
- Bai et al. (2025). "Mask-prior-guided denoising diffusion improves inverse protein folding." Nature Machine Intelligence.
- Mahbub et al. (2025). "Uncertainty-Aware Discrete Diffusion Improves Protein Design." bioRxiv.

### Transformer-Attention-Memory Bridge
- Sun et al. (2025). "Associative Transformer." CVPR 2025.
- Niu et al. (2024). "Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory." arXiv:2405.08707
- Zhong et al. (2025). "Understanding Transformer from the Perspective of Associative Memory." arXiv:2505.19488
- Kosowski et al. (2025). "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain." arXiv:2509.26507

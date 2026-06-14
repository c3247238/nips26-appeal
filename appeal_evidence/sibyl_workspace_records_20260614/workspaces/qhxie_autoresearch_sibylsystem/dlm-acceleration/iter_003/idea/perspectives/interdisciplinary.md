# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Information Theory / Channel Coding

1. **McEliece, MacKay & Cheng (1998) -- "Turbo Decoding As An Instance Of Pearl's Belief Propagation"** (IEEE JSAC) -- Establishes that turbo decoding is belief propagation on a loopy factor graph. The key insight: two imperfect component decoders exchange *extrinsic information* iteratively, and the composition achieves near-Shannon-limit performance that neither decoder achieves alone. The composability of component decoders is analyzed via EXIT (Extrinsic Information Transfer) charts, which predict convergence of the iterative system from properties of the individual components.

2. **Gallager (1960) / Richardson & Urbanke (2001) -- LDPC Codes** -- Message-passing decoding on sparse factor graphs. The *threshold phenomenon*: below a certain noise level (the BP threshold), iterative decoding converges; above it, it fails. This sharp phase transition is characterized analytically via density evolution.

3. **"Dynamic Phase Transition for Decoding Algorithms"** (Montanari et al.) -- Maps iterative decoding onto statistical physics models (diluted mean-field spin glasses). Shows that decoding algorithms exhibit dynamic phase transitions: performance degrades sharply at a critical noise threshold, analogous to a thermodynamic phase transition.

4. **D-Turbo-CS (2017)** -- Denoising-based Turbo Compressed Sensing. The closest existing bridge between turbo-style extrinsic message passing and generic denoisers. Derives explicit expressions for extrinsic messages of arbitrary denoisers, showing that turbo-style composition accelerates convergence of iterative denoising in compressed sensing.

5. **EXIT Chart Analysis** -- Semi-analytical tool for predicting whether two iteratively coupled decoders will converge. The "tunnel" between EXIT curves of the two decoders must remain open for successful decoding. Provides a principled way to analyze composability of two methods without exhaustive simulation.

#### Neuroscience / Cognitive Science

6. **Friston (2005/2010) -- Free Energy Principle and Predictive Coding** -- The brain minimizes variational free energy (equivalently, maximizes ELBO) through a cortical hierarchy. Top-down predictions meet bottom-up prediction errors at each level. *Precision weighting* modulates prediction errors based on estimated reliability -- effectively implementing adaptive attention. The iterative message passing between cortical levels is structurally analogous to iterative denoising.

7. **CogDPM (2024) -- "Diffusion Probabilistic Models via Cognitive Predictive Coding"** -- Explicitly maps the progressive denoising steps of diffusion models onto precision-weighted error minimization in predictive coding. Designs precision maps for each denoising step, weighting signals by their imprecision (high-noise signals get more processing effort).

8. **Vafaii et al. (NeurIPS 2025) -- "Brain-like Variational Inference"** -- Introduces FOND (Free energy Online Natural-gradient Dynamics), deriving iterative inference from three principles: natural gradients on free energy, online belief updating, and iterative refinement. Shows iterative inference outperforms amortized (single-pass) inference, directly supporting the value of multi-step refinement.

9. **"Dynamic Predictive Coding" (2024)** -- Higher cortical levels modulate temporal dynamics of lower levels. Lower levels encode shorter timescales, higher levels encode longer timescales. This hierarchical temporal structure is analogous to how different denoising steps in DLMs operate at different noise levels (timescales of uncertainty).

#### Statistical Physics / Thermodynamics

10. **Ambrogioni et al. (2025) -- "Statistical Thermodynamics of Generative Diffusion Models"** -- Shows that diffusion models undergo second-order phase transitions corresponding to symmetry breaking. The critical instability at these phase transitions is what enables generation. The mean-field universality class governs these transitions. Near-critical behavior implies *critical slowing down*: the system needs more iterations to converge near the phase transition.

11. **Simulated Annealing and Adaptive Temperature Schedules** -- Near a phase transition, specific heat CV = dE/dT grows, signaling the algorithm should slow its cooling. This is directly analogous to DLM denoising: certain steps (near the "phase transition" where tokens first become identifiable) are more critical and need more compute. Rushing through these steps causes quality loss.

12. **Renormalization Multigrid (RMG)** -- Combines statistical mechanics renormalization group with multigrid acceleration. Coarse-graining identifies collective variables; fine-graining refines them. The statistically optimal flow between scales accelerates Monte Carlo simulation.

#### Numerical Methods / Applied Mathematics

13. **Multigrid Methods (Brandt 1977, Hackbusch 1985)** -- Hierarchy of grids accelerates convergence of iterative solvers. High-frequency errors are damped by local smoothing; low-frequency errors are corrected on coarser grids. The V-cycle achieves O(N) convergence where standard iterative methods achieve O(N^2) or worse. The key insight: *different error components are best addressed at different resolution scales*.

14. **DNN-MG (2022-2026)** -- Neural networks embedded in multigrid cycles as smoothers, prolongation operators, or coarse-grid solvers. Demonstrates 2-100x speedups over classical solvers while maintaining accuracy.

#### Biology / Immunology

15. **Clonal Selection and Affinity Maturation** -- The immune system generates thousands of B cell variants via somatic hypermutation (parallel speculative candidates), selects those with highest antigen affinity, and discards the rest. Cycles of mutation and selection progressively improve antibody quality. The *germinal center* dark zone (exploration via mutation) and light zone (selection via affinity testing) form an iterative generate-and-verify loop.

16. **"Computational Convergence of Adaptive Immunity and AI" (bioRxiv 2026)** -- Argues that four major AI innovations can be derived from immunological first principles. Key formal equivalences: softmax in transformer attention = Boltzmann distribution of antibody-antigen binding; InfoNCE loss = negative log of clonal selection probability; pre-training/fine-tuning/RLHF = germline/somatic hypermutation/T follicular helper stages.

### Cross-Disciplinary Gaps

The following cross-disciplinary transplants have NOT been attempted for DLM inference acceleration:

1. **Turbo-style composability analysis with EXIT charts**: No one has formalized the composition of DLM acceleration methods using the extrinsic information / EXIT chart framework from channel coding. This would provide a principled, semi-analytical tool for predicting which methods compose well.

2. **Multigrid-inspired hierarchical denoising for DLMs**: While "Not All Steps Are Equal" and TCCF explore coarse-to-fine ideas, no one has formally mapped the multigrid V-cycle (restriction, smoothing, prolongation, correction) onto the DLM denoising pipeline.

3. **Precision-weighted adaptive compute from predictive coding**: CogDPM bridges diffusion and predictive coding for images, but no one has applied precision-weighted prediction errors to DLM token-level denoising to allocate compute where uncertainty is highest.

---

## Phase 2: Initial Candidates

### Candidate A: Turbo-Composability Framework for DLM Acceleration (from Channel Coding / Information Theory)

- **Source principle**: In turbo codes, two imperfect component decoders exchange *extrinsic information* iteratively, achieving near-optimal performance through composition. The composability of two decoders is predicted by EXIT (Extrinsic Information Transfer) charts: each decoder's input-output transfer function is characterized independently, and the system converges if and only if the EXIT curves form an open "tunnel." This provides a semi-analytical prediction of composability without exhaustive empirical testing.

- **Structural correspondence**: 
  - Component decoder 1 / decoder 2 --> Acceleration method A / method B (e.g., KV caching / early stopping)
  - Extrinsic information --> The "residual improvement" that each method provides: method A reduces per-step compute while passing its output (a partially accelerated inference trace) to method B, which reduces step count
  - EXIT curve --> A transfer function mapping "input quality" (accuracy at a given speedup) to "output quality" for each method
  - Turbo convergence condition (open tunnel) --> Composability condition: two methods compose well if each method's acceleration does not destroy the information the other method relies on
  - Interleaver --> The denoising schedule that mediates between the two methods' operating points

- **Hypothesis**: Two DLM acceleration methods compose well (super-additive speedup with sub-additive quality loss) if and only if they operate on "orthogonal information dimensions" -- analogous to the extrinsic/intrinsic information separation in turbo codes. Specifically: KV caching (which reduces per-step FLOPS by reusing computations across steps) and early stopping (which reduces step count by committing high-confidence tokens) should compose well because caching provides extrinsic information about cross-step stability while early stopping provides extrinsic information about per-token convergence. Methods that both rely on the same signal (e.g., two different caching strategies that both depend on KV drift) should NOT compose well, analogous to correlated extrinsic information degrading turbo performance.

- **Why not just a metaphor**: The mapping is formal. In turbo codes, the extrinsic information L_e^(1) from decoder 1 becomes the a priori information L_a^(2) for decoder 2. In DLM composition, the output state of method A (e.g., which KV pairs are cached vs. refreshed) becomes the input condition for method B (e.g., which tokens are candidates for early stopping). The EXIT chart analysis can be literally computed: measure each method's quality-speedup transfer function independently, then predict the composed system's behavior from the intersection geometry of the two curves. This is a testable quantitative prediction, not a loose analogy.

- **Novelty estimate**: 8/10 -- D-Turbo-CS applies turbo principles to generic denoisers in compressed sensing, but no one has applied EXIT chart composability analysis to DLM acceleration methods.

### Candidate B: Multigrid V-Cycle Denoising for DLMs (from Numerical Methods / Applied Mathematics)

- **Source principle**: Multigrid methods accelerate iterative solvers by exploiting a hierarchy of resolution scales. High-frequency error components are efficiently damped by local smoothing on the fine grid; low-frequency (global) error components are corrected on coarser grids where they become high-frequency and are cheaply eliminated. The V-cycle (fine --> coarse --> fine) achieves optimal O(N) convergence.

- **Structural correspondence**:
  - Fine grid / coarse grid --> Full model forward pass / lightweight proxy (e.g., smaller model, subset of layers, or reduced-precision attention)
  - High-frequency error --> Token-level uncertainty (which tokens are masked, local prediction errors)
  - Low-frequency error --> Global sequence coherence (long-range dependencies, overall narrative structure)
  - Smoothing (Jacobi/Gauss-Seidel relaxation) --> Standard denoising steps on the full model (expensive but handles local details)
  - Restriction operator --> Projecting the current noisy sequence state to a coarser representation (e.g., only attend to confident tokens + random sample of uncertain ones)
  - Coarse grid solve --> A cheap forward pass on the reduced representation that corrects global coherence
  - Prolongation operator --> Mapping the coarse correction back to full resolution (updating all token logits based on the coarse-grid insight)
  - V-cycle --> Alternating between expensive fine-grain denoising (a few steps) and cheap coarse-grain correction (a few steps)

- **Hypothesis**: A multigrid-inspired denoising schedule that alternates between full-model steps (fine smoothing) and lightweight-model steps (coarse correction) will converge faster than either pure full-model denoising or pure lightweight-model denoising. The coarse steps efficiently resolve global dependencies (which tokens should unmask in which order), while the fine steps resolve local token-level quality. This predicts that the "sandwich schedule" of "Not All Steps Are Equal" works precisely because it approximates a multigrid V-cycle -- and that a proper V-cycle with explicit restriction/prolongation operators would outperform the ad hoc sandwich.

- **Why not just a metaphor**: The multigrid framework makes specific quantitative predictions: (1) the spectral radius of the error propagation operator determines convergence rate; (2) the optimal number of smoothing steps at each level is determined by the smoothing factor; (3) the restriction and prolongation operators must satisfy a Galerkin condition for optimal convergence. These can be literally measured for DLMs: compute the "smoothing factor" (how much one denoising step reduces token-level entropy), the "spectral radius" of the error propagation (how quickly token predictions stabilize across steps), and design restriction/prolongation operators that satisfy the corresponding conditions.

- **Novelty estimate**: 7/10 -- TCCF ("Think Coarse Critic Fine") and the model scheduling paper explore coarse-to-fine ideas, but neither formulates the problem as a multigrid V-cycle with explicit restriction/prolongation/smoothing operators and convergence theory.

### Candidate C: Precision-Weighted Iterative Denoising from Predictive Coding (from Neuroscience)

- **Source principle**: In the brain's predictive coding hierarchy, prediction errors are weighted by their estimated *precision* (inverse variance). High-precision errors (reliable signals from well-lit environments) drive large updates; low-precision errors (unreliable signals from noisy environments) are attenuated. This implements optimal Bayesian inference: the Kalman gain. Crucially, precision weighting is *adaptive* -- the brain continuously re-estimates precision at each level of the hierarchy and at each time step.

- **Structural correspondence**:
  - Cortical level --> Denoising step t (each step is a "level" in the iterative hierarchy)
  - Top-down prediction --> Model's predicted token distribution at step t
  - Bottom-up sensory signal --> The actual masked/noisy token sequence
  - Prediction error --> Difference between model prediction and current state (the tokens that need to change)
  - Precision --> Inverse of token-level entropy at step t: high-entropy (uncertain) tokens have low precision, low-entropy (confident) tokens have high precision
  - Precision-weighted update --> Update magnitude proportional to precision: confident tokens get committed (large update), uncertain tokens get small updates and continue refining
  - Attention as precision weighting --> The brain's attentional mechanisms implement precision weighting; in DLMs, the KV cache refresh decision can be reformulated as a precision-weighted attention allocation

- **Hypothesis**: A precision-weighted denoising strategy that allocates compute (attention computation, KV cache refreshes, number of refinement iterations) proportional to the *inverse precision* (i.e., proportional to entropy) of each token at each step will achieve a better speed-quality Pareto frontier than uniform allocation. Specifically: (1) tokens with high precision (low entropy) should be committed early and their KV cached aggressively (never refreshed); (2) tokens with low precision (high entropy) should receive full attention recomputation at every step; (3) the precision threshold for commitment should follow a schedule analogous to the brain's neuromodulatory gain control (e.g., dopaminergic precision scaling). This unifies EntropyCache (which uses entropy to decide KV refresh) with JoT (which uses confidence for early stopping) into a single precision-weighted framework.

- **Why not just a metaphor**: The mapping preserves the mathematical structure. In predictive coding, the precision-weighted prediction error update is: mu_new = mu_old + Pi * epsilon, where Pi is precision (inverse variance) and epsilon is prediction error. In DLMs, the analogous update is: logit_new = logit_old + alpha(H) * delta_logit, where alpha(H) is a function of token entropy H (playing the role of precision), and delta_logit is the change from one denoising step. The Kalman gain interpretation gives an optimal alpha(H) = sigma_prior^2 / (sigma_prior^2 + sigma_obs^2), which can be estimated from the denoising trajectory statistics. This is not just "pay more attention to uncertain tokens" (which is obvious); it provides the *optimal* weighting function derived from Bayesian first principles.

- **Novelty estimate**: 6/10 -- CogDPM maps diffusion to predictive coding for images; EntropyCache and DyLLM use entropy/saliency signals in DLMs. But no one has derived the *optimal* precision-weighted compute allocation for DLM denoising from Bayesian/predictive coding first principles, and no one has used this to unify existing methods (KV caching + early stopping) into a single framework.

---

## Phase 3: Self-Critique

### Against Candidate A (Turbo-Composability / EXIT Charts)

- **Shallow analogy attack**: This is the strongest candidate. The correspondence is genuinely structural: both turbo decoding and DLM method composition involve iteratively coupling two imperfect processing stages. The EXIT chart analysis is a literal mathematical tool that can be applied to any two iteratively coupled systems -- it does not depend on the specific domain. A channel coding theorist would recognize the structure immediately. **Verdict: NOT shallow.**

- **Scale mismatch attack**: Turbo codes operate on bit-level signals; DLMs operate on token-level logits over a vocabulary of 32K-128K tokens. The dimensionality is vastly different. However, EXIT charts characterize the *mutual information* between the true signal and the decoder's estimate, which is a scalar quantity regardless of the underlying alphabet size. The mutual information characterization should scale. **Verdict: Manageable.**

- **Prior transplant check**: Searched for "turbo decoding diffusion model extrinsic information composability" -- no results. D-Turbo-CS applies turbo principles to generic denoisers but in compressed sensing, not language models, and not for composability analysis of acceleration methods. "TurboDiffusion" is a name-only coincidence (video acceleration framework, no turbo decoding theory). **Verdict: No prior transplant found.**

- **Testability attack**: The EXIT chart analysis makes a concrete, falsifiable prediction: two methods should compose well if their EXIT curves form an open tunnel, and poorly if the curves intersect. This can be tested by: (1) measuring each method's quality-speedup transfer function independently, (2) computing the EXIT chart, (3) predicting the composed system's performance, (4) comparing with actual measurements. If the EXIT prediction matches actual composability (within reasonable error), the framework is validated. If it fails, the analogy breaks. **Verdict: TESTABLE.**

- **Verdict: STRONG**

### Against Candidate B (Multigrid V-Cycle)

- **Shallow analogy attack**: The multigrid correspondence is partially structural but has a key gap: multigrid methods require linearity (or at least bilinearity) of the error propagation operator for the Galerkin condition to hold. DLM denoising is highly nonlinear (transformer forward passes). The multigrid convergence theory (spectral radius bounds) may not transfer to nonlinear systems. A numerical analyst would point out that nonlinear multigrid (FAS -- Full Approximation Scheme) exists but has weaker convergence guarantees. **Verdict: Partially shallow -- the qualitative insight (different error scales need different resolution) is structural, but the quantitative convergence theory may not transfer.**

- **Scale mismatch attack**: Multigrid operates on continuous fields discretized on spatial grids; DLMs operate on discrete token sequences. The "restriction" and "prolongation" operators in the token domain are not obvious -- what does it mean to "coarsen" a token sequence? One could use model size as the resolution axis (small model = coarse grid), but this is different from spatial coarsening. **Verdict: Significant mismatch in the definition of "grid."**

- **Prior transplant check**: TCCF ("Think Coarse Critic Fine") uses coarse-to-fine block sizes. "Not All Steps Are Equal" schedules different-sized models across steps. Neither references multigrid theory. The model scheduling paper (arXiv 2604.02340) achieves only 17% FLOP reduction -- far from multigrid's theoretical O(N) optimal convergence, suggesting the analogy may be incomplete. **Verdict: Coarse-to-fine ideas exist but are ad hoc, not multigrid-formalized.**

- **Testability attack**: The multigrid prediction is that a V-cycle with proper restriction/prolongation should converge faster than either pure fine or pure coarse denoising. This is testable but designing the restriction/prolongation operators for discrete token sequences is a significant engineering challenge. **Verdict: Testable in principle, but operationalization is hard.**

- **Verdict: MODERATE**

### Against Candidate C (Precision-Weighted Predictive Coding)

- **Shallow analogy attack**: The correspondence between precision weighting and entropy-based compute allocation is real but arguably *already implicit* in existing methods. EntropyCache uses token entropy for KV refresh decisions; JoT uses confidence for early stopping; DyLLM uses saliency for partial attention. These are all "precision-like" signals. Formalizing them under predictive coding adds theoretical elegance but may not add practical value beyond what entropy/confidence thresholds already provide. A neuroscientist might say: "yes, this is just Bayesian inference with a biological label." **Verdict: Risk of being a rebranding rather than a genuine transplant.**

- **Scale mismatch attack**: Predictive coding operates on continuous neural activations with Gaussian noise assumptions; DLMs operate on discrete tokens with mask noise. The precision (inverse variance) concept translates cleanly to inverse entropy, but the Kalman gain formula assumes Gaussian distributions, which is wrong for discrete token distributions. The optimal weighting may differ significantly from the Gaussian-derived formula. **Verdict: The specific formula may not transfer, even if the qualitative insight does.**

- **Prior transplant check**: CogDPM (2024) explicitly maps diffusion models to predictive coding with precision weighting -- but for continuous image diffusion, not discrete language diffusion. EntropyCache (2026) uses entropy for KV refresh in DLMs but does not cite predictive coding or derive the refresh rule from Bayesian optimality. The gap between these two works is the target of this candidate. **Verdict: Partially explored in adjacent domains.**

- **Testability attack**: The specific prediction is that the Bayesian-optimal compute allocation (derived from precision weighting) outperforms heuristic entropy thresholds. This requires: (1) deriving the optimal alpha(H) from the DLM's denoising dynamics, (2) comparing against EntropyCache's heuristic threshold and JoT's confidence threshold. If the optimal weighting provides only marginal improvement over the heuristic, the theoretical contribution is weak. **Verdict: Testable, but the margin of improvement may be small.**

- **Verdict: MODERATE** (risk of being a theoretical rebranding of existing practice)

---

## Phase 4: Refinement

### Dropped

**Candidate C (Precision-Weighted Predictive Coding)** is dropped as a standalone proposal. The risk of being a theoretical rebranding of existing entropy/confidence-based methods (EntropyCache, JoT, DyLLM) is too high. The Gaussian precision framework does not neatly transfer to discrete token distributions. However, the *insight* -- that optimal compute allocation follows Bayesian optimality principles -- is preserved and incorporated into Candidate A's EXIT chart analysis as a theoretical grounding.

### Strengthened: Candidate A (Turbo-Composability Framework)

**Formalized structural correspondence:**

Let M_A and M_B be two DLM acceleration methods. Define:
- I(X; Y_A(s)) = mutual information between true output X and method A's output Y_A at speedup s
- I(X; Y_B(s)) = mutual information between true output X and method B's output Y_B at speedup s
- T_A(I_in) = method A's EXIT transfer function: given input quality I_in, outputs quality I_out
- T_B(I_in) = method B's EXIT transfer function

The turbo composability condition states: methods A and B compose well (the iterative system converges to high quality) if and only if T_A(I) > T_B^{-1}(I) for all I in [I_min, I_max], i.e., the EXIT curves do not cross in the (I_A, I_B) plane.

**Diagnostic experiment design:**
1. Measure T_A independently: for method A (e.g., EntropyCache), sweep its aggressiveness parameter and measure (speedup, quality) pairs. Convert to I(X; Y_A) using the relationship between benchmark accuracy and mutual information.
2. Measure T_B independently: same for method B (e.g., JoT early stopping).
3. Compute the EXIT chart: plot T_A and T_B^{-1} in the (I_A, I_B) plane.
4. Predict composability: if tunnel is open --> methods should compose with super-additive speedup; if tunnel is closed --> methods will interfere.
5. Validate: actually compose A+B and measure the result. Compare predicted vs. actual.

**Why this is load-bearing, not decorative:** The EXIT chart makes a specific quantitative prediction that can be right or wrong. If it correctly predicts which method pairs compose well (e.g., KV cache + early stopping = open tunnel) and which don't (e.g., two competing KV cache strategies = closed tunnel), then the framework provides genuine predictive power beyond exhaustive empirical search. If it fails, we learn that DLM acceleration composition has fundamentally different dynamics than turbo decoding composition.

### Strengthened: Candidate B (Multigrid V-Cycle) -- Absorbed into Candidate A as a secondary contribution

Rather than proposing multigrid as a standalone method, the multigrid insight is incorporated into Candidate A: the EXIT chart framework can be applied to *hierarchical* compositions (fine-model step + coarse-model step), not just to two-method compositions. The multigrid V-cycle becomes one specific composition pattern that the EXIT framework can analyze.

### Selected Front-Runner: Candidate A -- EXIT Chart Composability Analysis for DLM Acceleration

---

## Phase 5: Final Proposal

### Title
**EXIT Chart Composability Analysis: A Channel-Coding Framework for Predicting and Optimizing Multi-Method DLM Inference Acceleration**

### Source Principle
In iterative turbo decoding (information theory), two imperfect component decoders are concatenated via an interleaver and exchange extrinsic information iteratively. The composability and convergence of this iterative system is predicted -- without exhaustive simulation -- by EXIT (Extrinsic Information Transfer) charts, which characterize each component's input-output mutual information transfer function independently. The system converges to near-optimal performance if and only if the EXIT curves of the two components form an open "decoding tunnel." This provides a semi-analytical composability prediction tool that has been the gold standard in code design for two decades.

### Structural Correspondence

| Channel Coding Concept | DLM Acceleration Concept |
|------------------------|--------------------------|
| Component decoder 1 | Acceleration method A (e.g., KV caching) |
| Component decoder 2 | Acceleration method B (e.g., early stopping) |
| Codeword (transmitted signal) | Ground-truth output sequence |
| Channel noise | Denoising noise schedule (mask ratio) |
| Received noisy signal | Current noisy/masked sequence at step t |
| Extrinsic information from decoder 1 | Residual information from method A: which tokens are confidently cached, which KV pairs are stable |
| Extrinsic information from decoder 2 | Residual information from method B: which tokens have converged early, which steps can be skipped |
| Interleaver | Denoising schedule that mediates between methods (step ordering, which method acts when) |
| EXIT transfer function T(I_in) | Method's quality-speedup curve: given input sequence quality I_in, what quality I_out does the method produce at a given speedup? |
| Open decoding tunnel | Composability condition: the two methods' EXIT curves do not cross --> they can be iteratively composed for super-additive acceleration |
| Closed tunnel / crossing curves | Interference condition: methods compete for the same information signal --> composition yields sub-additive or even negative returns |
| EXIT chart area theorem | Upper bound on achievable composed speedup-quality tradeoff |

### Hypothesis
Two training-free DLM acceleration methods compose well (super-additive speedup with bounded quality loss) if and only if their EXIT transfer functions form an open tunnel -- meaning each method improves the inference state along a dimension that is orthogonal to the other method's operating dimension. Specifically:

1. **KV caching (cross-step compute reduction) + Early stopping (step-count reduction)** should compose well because they operate on orthogonal information dimensions: caching exploits cross-step KV stability (a temporal signal), while early stopping exploits per-token confidence (a spatial signal). **Predicted: OPEN TUNNEL.**

2. **Two competing KV caching strategies (e.g., EntropyCache + Elastic-Cache)** should compose poorly because they both rely on the same signal (KV drift / token stability across steps). The extrinsic information from one is correlated with the intrinsic information of the other, violating the extrinsic information separation principle. **Predicted: CLOSED TUNNEL.**

3. **Speculative decoding (draft-then-verify) + KV caching** should compose partially: the draft phase benefits from caching (draft model reuses prior KV), but the verify phase disrupts the cache (verification may change tokens, invalidating cached KV). **Predicted: PARTIALLY OPEN TUNNEL** with a specific degradation mode at the draft-verify boundary.

### Method

1. **Define the EXIT transfer function for DLM acceleration methods.**
   - For each method M and aggressiveness parameter alpha (e.g., cache refresh rate, confidence threshold, number of speculative steps):
     - Run inference on a held-out calibration set with method M at parameter alpha
     - Measure output quality Q(alpha) (e.g., accuracy on MMLU/GSM8K) and speedup S(alpha)
     - Compute mutual information I(X; Y_M(alpha)) between ground truth X and method output Y_M, estimated via the relationship I approx H(X) - H(X|Y_M) where H(X|Y_M) is the conditional entropy of the ground truth given the method's output
   - Plot T_M: I_in --> I_out as the method's EXIT transfer function

2. **Compute pairwise EXIT charts.**
   - For each pair of methods (M_A, M_B):
     - Plot T_A and T_B^{-1} in the (I_A, I_B) plane
     - Measure the "tunnel opening": min gap between the two curves
     - Predict: open tunnel --> composable; closed tunnel --> interfering

3. **Validate predictions.**
   - Actually compose each pair of methods and measure (speedup, quality)
   - Compare predicted composability (from EXIT chart) vs. actual composability
   - Report prediction accuracy as the primary metric

4. **Design optimal compositions.**
   - For method pairs with open tunnels, use the EXIT chart to find the optimal operating points (aggressiveness parameters) that maximize the tunnel area
   - The area theorem from turbo coding theory gives an upper bound on achievable composed performance

### Diagnostic Experiment
The key test that confirms the analogy is load-bearing (not decorative):

**Prediction:** The EXIT chart correctly predicts the rank ordering of method pair composability scores. Specifically, if the EXIT chart predicts that pair (A,B) has a wider tunnel than pair (C,D), then the actual composed speedup-quality tradeoff of (A+B) should be strictly better than (C+D).

**Control:** Compare against the null hypothesis that composability is simply predicted by the product of individual speedups (the "independence assumption"). If the EXIT chart prediction is no better than the product-of-speedups prediction, the turbo analogy is not load-bearing.

**Strong validation:** The EXIT chart predicts a specific failure mode -- identify a pair predicted to have a closed tunnel, and demonstrate that composing them actually degrades performance below either method alone. This would be a smoking-gun confirmation.

### Experimental Plan

**Phase 1: EXIT curve measurement (3-4 experiments, ~45 min each)**
- Model: LLaDA-8B-Instruct
- Methods to characterize: EntropyCache (M1), JoT early stopping (M2), IGSD speculative decoding (M3)
- For each method, sweep 5 aggressiveness levels and measure (quality, speedup) on GSM8K-200 and HumanEval-50
- Compute EXIT transfer functions

**Phase 2: Composability prediction + validation (3-6 experiments, ~30 min each)**
- Compute EXIT charts for all pairs: (M1,M2), (M1,M3), (M2,M3)
- Predict composability from tunnel opening
- Validate by actually running each pair's composition
- Add one "predicted failure" pair: EntropyCache + Sparse-dLLM (two KV cache methods, predicted closed tunnel)

**Phase 3: Optimal composition design (1-2 experiments, ~45 min each)**
- For the best-composing pair, use EXIT chart to find optimal operating point
- Compare EXIT-optimized composition vs. grid-search-optimized composition
- Report whether EXIT chart reduces the search space for optimal composition

Total: ~8-12 experiments, ~10-12 hours. Each individual task fits within the 1-hour budget.

### Risk Assessment

1. **The EXIT chart may be too coarse.** Turbo codes have well-defined component decoders with clean extrinsic/intrinsic separation. DLM acceleration methods may have more complex information interactions that are not captured by a scalar mutual information measure. *Mitigation:* Use multi-dimensional EXIT charts (e.g., separate curves for reasoning vs. coding tasks).

2. **Mutual information estimation is noisy.** Computing I(X; Y_M) from finite benchmark samples introduces estimation error. *Mitigation:* Use multiple benchmarks and report confidence intervals; use the relationship between accuracy and mutual information for binary-outcome tasks (GSM8K correct/incorrect) where the estimation is exact.

3. **The "orthogonality" may be task-dependent.** Two methods that compose well on GSM8K may interfere on HumanEval. *Mitigation:* Measure EXIT curves separately for each benchmark type and report task-dependent composability predictions.

4. **DLM denoising is nonlinear.** Turbo decoding theory assumes linear message passing (belief propagation), while DLM denoising is highly nonlinear (transformer forward passes). The EXIT chart analysis is known to be approximate on loopy graphs (which is also the case in turbo codes), but the approximation quality may degrade for DLMs. *Mitigation:* The EXIT chart is already used as an approximate tool in turbo codes (it ignores cycle structure); we use it in the same spirit as an approximate but useful predictor.

### Novelty Claim

**The specific cross-disciplinary insight:** The composability of DLM acceleration methods can be predicted using EXIT (Extrinsic Information Transfer) charts from channel coding theory. This transplant is novel because:

1. No published work applies EXIT chart analysis to DLM acceleration composability (confirmed via arXiv/Google Scholar/Web search for "turbo decoding diffusion model extrinsic information composability" and related queries -- zero results).

2. The closest prior work, D-Turbo-CS (2017), applies turbo principles to generic denoisers in compressed sensing but does not address (a) discrete token diffusion, (b) composability of acceleration methods, or (c) the EXIT chart as a composability prediction tool.

3. The ComposeAccel project (this workspace) addresses composability empirically through pairwise ablation. The EXIT chart framework would complement this by providing a *theoretical prediction tool* that explains *why* certain pairs compose well and others don't, and that can predict composability of new method pairs without exhaustive empirical testing.

4. The diagnostic experiment (predicting rank ordering of composability from EXIT curves) provides a sharp test: if the turbo analogy is correct, the EXIT chart should predict composability better than the naive independence assumption.

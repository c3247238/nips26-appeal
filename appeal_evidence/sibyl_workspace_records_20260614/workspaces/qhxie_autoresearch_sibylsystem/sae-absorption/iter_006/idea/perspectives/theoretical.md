# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Tang et al. (2025). "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima." arXiv:2512.05534** -- The most comprehensive theoretical treatment of SAE optimization dynamics. Casts all SDL variants (SAEs, transcoders, crosscoders) as a single piecewise biconvex optimization problem. Theorem 4.7 proves the existence of spurious local minima corresponding to absorption patterns: for any realizable absorption pattern with at least one partition of size >= 2, there exists a local minimum. Proposes feature anchoring to restore identifiability. Key mathematical result: global minima correspond to correct feature recovery; absorption arises from spurious partial minima that are optimization-landscape artifacts, not training failures.

2. **Cui et al. (2025). "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy." arXiv:2506.15963** -- First closed-form theoretical analysis of SAE feature recovery. Establishes necessary and sufficient conditions for identifiable SAEs: (1) extreme sparsity of ground-truth features, (2) sparse activation of SAEs, (3) sufficient hidden dimensions. Key negative result: standard SAEs inevitably suffer from feature shrinking and vanishing, preventing full recovery except under extreme sparsity. Proposes WSAE with principled weight selection.

3. **Chanin et al. (2024). "A is for Absorption." arXiv:2409.14507 (NeurIPS 2025)** -- Foundational empirical characterization of absorption. Contains a toy model proof that hierarchical features cause absorption under sparsity optimization. The informal argument: if features form a hierarchy where parent always co-occurs with child, the SAE saves one L0 unit by absorbing the parent direction into the child latent. This is the starting point for any formal theory. Key gap: the toy model does not predict absorption magnitude as a function of SAE configuration.

4. **Equitz & Cover (1991). "Successive Refinement of Information." IEEE Trans. Inf. Theory 37(2):269-275** -- The foundational information-theoretic result on multi-resolution coding. A source is successively refinable if and only if the individual rate-distortion solutions form a Markov chain. This is the theoretical anchor for the current proposal's Stage 5: when the Markov chain condition is violated for a hierarchical feature pair, absorption destroys information; when it holds, absorption is lossless.

5. **Bereska et al. (2024). "Superposition as Lossy Compression." arXiv:2512.13568** -- Provides the rate-distortion framework for understanding superposition. Key insight: each neural layer acts as a bandwidth-limited channel, and superposition is the optimal response to rate-distortion constraints. Measures "effective features" (the exponential of Shannon entropy of SAE activations) as a principled superposition metric. Establishes that when networks encode more effective features than physical neurons, interference is unavoidable.

6. **Chen et al. (2025). "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders." arXiv:2506.14002** -- Proposes bias-adaptation training with theoretical recovery guarantees under a statistical generative model. Group Bias Adaptation (GBA) variant handles correlated features. Key limitation for absorption theory: the generative model does not include feature hierarchies, so the guarantees do not address the hierarchical co-occurrence structure that drives absorption.

7. **Wainwright (2009). "Information-Theoretic Limits on Sparsity Recovery in the High-Dimensional and Noisy Setting." IEEE Trans. Inf. Theory** -- Establishes fundamental sample complexity bounds for sparse support recovery: the required number of measurements scales as O(k log(p/k)) where k is sparsity and p is dimension. Relevant to absorption because it establishes that sparse recovery is information-theoretically hard when features are correlated -- the mutual coherence of hierarchical features directly increases the required sample complexity.

8. **Rozell et al. (2008). "Sparse Coding via Thresholding and Local Competition in Neural Circuits."** -- The Locally Competitive Algorithm (LCA) solves sparse coding via lateral inhibition, where active neurons suppress others based on their Gram matrix entries. This provides the dynamical systems interpretation of absorption: when a child feature has high Gram matrix overlap with the parent, lateral inhibition from the child kills the parent's activation. Absorption IS lateral inhibition operating across hierarchy levels.

9. **D'Amato, Lancia & Pezzulo (2025). "The Geometry of Efficient Codes: How Rate-Distortion Trade-offs Distort Latent Representations." PLOS Computational Biology** -- Identifies three distortions emerging from rate-distortion optimization: prototypization, specialization, and orthogonalization. Feature absorption maps onto "specialization" -- where capacity constraints cause the coding system to abandon general descriptions in favor of specific ones.

10. **Reizinger et al. (2025). "From Superposition to Sparse Codes: Interpretable Representations in Neural Networks." arXiv:2503.01824** -- Connects superposition to compressed sensing theory, arguing that sparse codes can be recovered from superposed neural activations under appropriate conditions. Key theoretical insight: recovery is possible when the measurement matrix (LLM's representation) satisfies restricted isometry property (RIP) conditions, but hierarchical features systematically violate RIP in the parent-child subspace.

11. **Aksoylar & Saligrama (2014). "Information-Theoretic Characterization of Sparse Recovery." AISTATS** -- Provides non-asymptotic bounds on sparse recovery probability as a function of mutual information. Establishes that feature correlation (as in parent-child hierarchies) tightens the necessary conditions for recovery. This is the information-theoretic foundation for why absorption becomes more severe as parent-child correlation increases.

12. **Tilde Research (2025). "The Rate Distortion Dance of Sparse Autoencoders." Blog post** -- Interprets TopK SAEs through the information bottleneck lens. Key observation: TopK fixes the sparsity (rate) for all inputs uniformly, which creates an information bottleneck that is particularly constraining for hierarchically structured inputs where some inputs "need" more active features to capture both parent and child concepts.

### Theoretical Landscape Summary

The theoretical understanding of feature absorption sits at the intersection of three mature mathematical frameworks, none of which has been fully brought to bear on the problem:

**1. Optimization landscape theory (Tang et al., 2025)**: Absorption corresponds to spurious partial minima of the piecewise biconvex SDL objective. The existence of these minima is formally proven (Theorem 4.7), and feature anchoring is proposed as a remedy. However, the theory does not provide a *quantitative* prediction of absorption severity as a function of measurable feature properties (hierarchy depth, co-occurrence ratio, SAE width/L0). It characterizes the *existence* of absorption equilibria but not their *basin of attraction* size or *probability of convergence* to them.

**2. Identifiability theory (Cui et al., 2025; Chen et al., 2025)**: Necessary and sufficient conditions for SAE identifiability are established under specific generative models, but these models do not include hierarchical feature structure. The extreme sparsity requirement (Cui et al.) is a pessimistic bound that does not account for the possibility that *partial* recovery (recovering children but not parents) may be the optimal strategy under moderate sparsity. No existing identifiability result characterizes the specific failure mode of "recovering a strict subset of features where the missing features are exactly those that are hierarchically subsumed."

**3. Rate-distortion and information theory (Bereska et al., 2024; Equitz & Cover, 1991)**: Superposition is lossy compression; SAEs attempt to decompress. The successive refinement theorem provides the exact condition under which multi-resolution decompression is possible (Markov chain condition). But no one has connected this to absorption specifically: when is absorption "lossless" (parent carries no unique information beyond the child) versus "lossy" (absorption destroys recoverable information)? This is the core theoretical question.

**The critical missing piece**: A *quantitative* theory that predicts absorption rate R_abs as a closed-form function of:
- SAE configuration: width M, effective L0, sparsity penalty lambda
- Feature hierarchy properties: co-occurrence probability p(parent|child), hierarchy depth D, fan-out F
- Representation properties: parent-child decoder cosine similarity, conditional mutual information

Such a theory would unify the optimization landscape, identifiability, and rate-distortion perspectives and provide the first principled guidance for SAE hyperparameter selection to minimize absorption.

## Phase 2: Initial Candidates

### Candidate A: Rate-Distortion Bound on Absorption -- When is Absorption Information-Theoretically Necessary?

- **Formal claim (Theorem sketch)**: Let X be the LLM activation, f_P a parent feature, and f_C a child feature with f_P => f_C (parent always fires when child fires). Let an SAE have effective rate R = log(M) - H(z) where M is dictionary size and z is the sparse code. Then absorption of f_P into f_C is *rate-distortion optimal* (i.e., any SAE that separately represents f_P and f_C must use strictly more rate for the same distortion) if and only if the conditional mutual information I(X; f_P | f_C) < delta for a threshold delta that depends on the sparsity penalty strength lambda. When I(X; f_P | f_C) > delta, absorption is information-theoretically suboptimal -- it destroys recoverable information.

- **Proof sketch**:
  1. Model the SAE as a rate-limited channel from X to the sparse code z, with rate R = L0 * log(M) (each active latent selects from M dictionary elements). The sparsity penalty lambda controls the rate budget.
  2. The distortion D = E[||X - X_hat||^2] must be minimized subject to the rate constraint.
  3. For hierarchically related features f_P, f_C, consider two coding strategies: (a) Absorption: encode only f_C, absorb f_P. Rate cost: L0 = 1 per token where f_C fires. (b) Separate: encode both f_P and f_C. Rate cost: L0 = 2 per token where both fire.
  4. The distortion difference between strategies is proportional to I(X; f_P | f_C) -- the information in X about the parent direction that is not captured by the child direction.
  5. Strategy (a) is optimal when lambda * (rate_savings) > (distortion_penalty), which simplifies to I(X; f_P | f_C) < lambda / (constant depending on decoder geometry).
  6. Apply the successive refinement theorem: f_P, f_C form a successively refinable pair iff the rate-distortion-optimal descriptions form a Markov chain X -- f_C -- f_P. When this holds, absorption is lossless. When it fails, absorption is lossy by exactly I(X; f_P | f_C, Markov violation term).

- **Empirical prediction**: Measure I(X; f_P | f_C) for each first-letter feature pair. Features with high conditional MI should exhibit lower absorption rates (the SAE "pays" the extra L0 to keep them separate because the information loss would be too high). Features with low conditional MI should be preferentially absorbed (absorption is "free"). This is a testable, quantitative prediction with a clear falsification criterion.

- **Connection to existing theory**: Extends Chanin et al.'s qualitative sparsity-gain argument to a rigorous rate-distortion framework. Connects to Bereska et al.'s superposition-as-lossy-compression via the rate constraint. Recovers Cui et al.'s extreme-sparsity condition as a special case (when all features are independent, I(X; f_P | f_C) is maximized, and absorption is never optimal).

- **Novelty estimate**: 8/10. The successive refinement theorem (Equitz & Cover, 1991) is classical. Modeling SAEs as rate-limited channels is implicit in Bereska et al. and Tilde Research. The specific application to feature absorption -- deriving the absorption optimality condition as a conditional MI threshold -- is novel and produces testable quantitative predictions.

### Candidate B: Absorption as Optimization Landscape Basin Geometry -- A Stability Analysis

- **Formal claim (Proposition sketch)**: In the piecewise biconvex optimization landscape of the SAE objective (following Tang et al., 2025), the absorption minimum for a parent-child pair (f_P, f_C) has a basin of attraction whose measure (in parameter space) is proportional to:

  mu_abs ~ p(f_P & f_C) * cos(w_P, w_C) * (lambda / sigma^2_P)

  where p(f_P & f_C) is the co-occurrence probability, cos(w_P, w_C) is the decoder cosine similarity, lambda is the sparsity penalty, and sigma^2_P is the variance of f_P's contribution to the data. The probability of converging to the absorption solution (rather than the correct solution) during SGD training is approximately proportional to mu_abs.

- **Proof sketch**:
  1. Starting from Tang et al.'s piecewise biconvex formulation, restrict to the 2-feature case (f_P, f_C) with hierarchical co-occurrence.
  2. Compute the Hessian of the SAE loss at both the absorption minimum (f_P absorbed into f_C) and the correct minimum (both separately represented).
  3. The eigenvalues of the Hessian determine the local curvature. The absorption minimum has a wider basin (flatter loss landscape) when: (a) p(f_P & f_C) is high (co-occurrence is frequent, so the "savings" from absorption apply to many tokens), (b) cos(w_P, w_C) is high (the decoder directions are already aligned, reducing the distortion from absorption), (c) lambda/sigma^2_P is high (the sparsity penalty dominates the reconstruction cost for f_P).
  4. Formalize the basin measure using the volume of the set of initializations that converge to the absorption minimum under gradient flow.

- **Empirical prediction**: The quantity mu_abs should predict which parent features get absorbed. Rank parents by mu_abs; the top-ranked should have the highest observed absorption rates. This gives a per-feature absorption risk score computable from decoder weights and activation statistics alone.

- **Connection to existing theory**: Directly extends Tang et al.'s Theorem 4.7 (existence of absorption minima) by characterizing basin geometry. Connects to Chanin et al.'s observation that absorption is more common for parent features with high co-occurrence with children.

- **Novelty estimate**: 7/10. Hessian analysis of piecewise biconvex problems is standard optimization theory. The application to SAE absorption basins is novel. The quantitative prediction (mu_abs formula) is new and testable. Risk: the 2-feature restriction may not capture multi-feature interactions.

### Candidate C: Lateral Inhibition Phase Transition -- A Dynamical Systems Theory of Absorption Onset

- **Formal claim (Theorem sketch)**: Model SAE inference as a dynamical system analogous to the Locally Competitive Algorithm (Rozell et al., 2008), where features compete via lateral inhibition proportional to their Gram matrix entries. For a parent-child pair, define the "inhibition ratio" q = G_{PC} * a_C / (theta_P - u_P) where G_{PC} is the Gram matrix entry, a_C is the child's activation magnitude, theta_P is the parent's threshold, and u_P is the parent's input drive. There exists a critical inhibition ratio q* = 1 at which the parent feature undergoes a bifurcation from active to inactive (absorbed). The absorption transition is discontinuous (first-order) for TopK/JumpReLU activation functions and continuous (second-order) for L1-penalized ReLU.

- **Proof sketch**:
  1. Write the LCA dynamical system for two features with hierarchical co-occurrence: du_P/dt = b_P - u_P - G_{PC} * T(u_C), du_C/dt = b_C - u_C - G_{CP} * T(u_P), where b is the input drive, G is the Gram matrix, T is the thresholding function.
  2. Compute fixed points as a function of the inhibition ratio q. For q < 1: both features are active at equilibrium (no absorption). For q > 1: only the child is active (absorption). At q = 1: bifurcation.
  3. The nature of the bifurcation depends on the thresholding function T. For hard thresholds (TopK, JumpReLU): supercritical pitchfork bifurcation (discontinuous jump from active to inactive). For soft thresholds (ReLU + L1): continuous transition through an intermediate regime of reduced activation.
  4. The critical threshold q* depends on the sparsity penalty: higher lambda lowers the parent's threshold theta_P, making q > 1 easier to satisfy. This formalizes Chanin et al.'s "sparsity gain" argument as a bifurcation condition.

- **Empirical prediction**: (a) Absorption should be more abrupt (all-or-nothing) for TopK/JumpReLU SAEs than for L1 ReLU SAEs -- matching the SAEBench observation that JumpReLU has worse absorption. (b) The inhibition ratio q is computable from decoder weights and activation statistics; features with q > 1 should be absorbed, q < 1 should not be. (c) At the critical L0 ~ 7-14 phase transition found in iteration 5's scaling analysis, the average q should cross 1 for a significant fraction of parent-child pairs.

- **Connection to existing theory**: Directly applies the LCA framework (Rozell et al., 2008) to SAE feature dynamics. The bifurcation analysis is standard dynamical systems theory. Connects to the iteration 5 observation of a phase transition at L0 ~ 7-14, providing a mechanistic explanation.

- **Novelty estimate**: 8/10. The LCA framework is well-established but has never been applied to analyze SAE feature absorption. The bifurcation analysis connecting activation function choice to absorption behavior is entirely new. The prediction that TopK/JumpReLU produces more abrupt absorption than L1 ReLU is testable and explanatory.

## Phase 3: Self-Critique

### Against Candidate A (Rate-Distortion Bound)

- **Proof soundness attack**: The main gap is step 5, where the rate-distortion tradeoff is simplified to a comparison of two discrete strategies. In reality, the SAE can partially absorb (the parent latent fires with reduced magnitude on some tokens, not a binary all-or-nothing). The binary comparison overestimates the sharpness of the absorption threshold. **Mitigation**: Extend to a continuous absorption parameter alpha in [0,1] measuring the fraction of parent information absorbed. The rate-distortion analysis then gives an optimal alpha* as a function of CMI and lambda. Searched arXiv for "partial absorption sparse autoencoder" -- zero results. The partial absorption formulation is novel.

- **Tightness attack**: The conditional MI threshold delta depends on the sparsity penalty lambda through a proportionality constant that involves decoder geometry. If this constant is loose, the bound becomes vacuous (predicting everything is absorbed or nothing is). **Mitigation**: Compute the constant empirically on the toy model from Chanin et al. and verify it gives non-trivial predictions. If the constant is tight on the toy model, it is likely informative on real SAEs.

- **Relevance attack**: The rate-distortion framework assumes the SAE has converged to a global optimum of the rate-constrained problem. In practice, SGD converges to local optima. The gap between the rate-distortion prediction and observed absorption is the "optimization gap" -- which may be large. **Mitigation**: This gap is itself a measurable quantity. If the rate-distortion bound says absorption should happen but it does not (or vice versa), that reveals something about the optimization dynamics.

- **Novelty attack**: Searched "rate distortion sparse autoencoder absorption," "successive refinement feature absorption," "conditional mutual information SAE features." Found Bereska et al. (2024) which models superposition as lossy compression but does NOT apply the successive refinement theorem to absorption. Found the Tilde Research blog post which discusses rate-distortion for TopK but does NOT connect to absorption or CMI. The specific application of the successive refinement theorem to absorption, and the CMI threshold for absorption optimality, appear to be genuinely novel.

- **Verdict**: **STRONG**. The proof sketch has identifiable gaps (partial absorption, optimization gap) but these are addressable. The predictions are quantitative, testable, and connect three mature theoretical frameworks. The rate-distortion framing provides a principled answer to "when is absorption actually harmful?" which the field lacks.

### Against Candidate B (Basin Geometry)

- **Proof soundness attack**: The basin measure formula mu_abs requires computing the Hessian at the absorption minimum of the piecewise biconvex objective. The "piecewise" nature means the loss function has non-differentiable boundaries (kinks at activation thresholds). The Hessian is well-defined within each piece but the basin volume calculation requires integrating across piece boundaries. This is technically feasible for the 2-feature case but becomes intractable for the full M-feature problem. **This is a significant limitation.**

- **Tightness attack**: The mu_abs formula is an approximation that ignores multi-feature interactions. When features A, B, C form a 3-level hierarchy, the absorption of A may depend on whether B has already been absorbed by C. The pairwise formula misses these cascade effects. **Mitigation**: Test on the 2-feature toy model first; if the formula is predictive there, extend with interaction corrections.

- **Relevance attack**: The formula requires knowing decoder cosine similarity and co-occurrence probability, which are computable from a trained SAE. But practitioners want to know absorption risk *before* training. The formula predicts absorption for a given SAE, not how to avoid it. **Moderate relevance.**

- **Novelty attack**: Tang et al.'s Theorem 4.7 already proves absorption minima exist. Our contribution would be quantifying basin size, which is a standard follow-up in optimization theory but technically non-trivial for piecewise objectives. Searched "basin of attraction piecewise biconvex" -- found general results but none applied to SAEs.

- **Verdict**: **MODERATE**. Technically sound for 2-feature case, but the extension to realistic SAEs is hard. Less directly testable than Candidate A because it requires optimization-landscape measurements rather than information-theoretic quantities computable from activations.

### Against Candidate C (Lateral Inhibition Phase Transition)

- **Proof soundness attack**: The LCA dynamical system models SAE *inference* (forward pass), not *training*. Absorption is a phenomenon of the *trained* SAE. The connection between inference dynamics and training dynamics is: if the LCA fixed point at inference time would deactivate the parent, then gradient updates during training will further push the SAE toward the absorption configuration (because the gradient of the sparsity loss reinforces the deactivation). This causal chain is plausible but not formally proven. **Mitigation**: Prove the chain formally by showing that the LCA fixed point is a fixed point of the SAE training gradient flow, which requires showing that the gradient aligns with the LCA dynamics. This is likely true for the L1 case (where the thresholding function is exactly the proximal operator of the L1 penalty) but less clear for TopK.

- **Tightness attack**: The inhibition ratio q = G_{PC} * a_C / (theta_P - u_P) depends on the child's activation magnitude a_C, which varies across tokens. The bifurcation is not a sharp phase transition in practice but a gradual shift in the fraction of tokens where q > 1. **Mitigation**: Define absorption rate as the fraction of tokens where q > 1. This gives a natural quantitative prediction.

- **Relevance attack**: The LCA framework is most natural for L1-penalized SAEs where inference is iterative. For TopK and JumpReLU SAEs, inference is a single forward pass with hard thresholding. The dynamical systems analysis still applies (the forward pass is one step of the dynamics) but the bifurcation analysis is less rich. **Moderate concern.**

- **Novelty attack**: Searched "locally competitive algorithm feature absorption," "LCA sparse autoencoder hierarchical." Zero results. The application of LCA bifurcation analysis to SAE absorption is novel. The prediction connecting activation function choice to absorption behavior (discontinuous vs. continuous) is new and testable.

- **Verdict**: **STRONG**. The inference-training gap is the main concern but is addressable. The bifurcation framework provides mechanistic insight that rate-distortion does not: it explains *how* absorption happens dynamically, not just *when* it is optimal. The prediction about TopK vs. L1 absorption dynamics is particularly compelling.

## Phase 4: Refinement

### Dropped
- **Candidate B (Basin Geometry)** is dropped as the primary proposal. The piecewise Hessian computation is technically challenging and the formula is harder to validate empirically than the other candidates. However, the core insight (co-occurrence * cosine similarity * sparsity/variance as an absorption risk score) can be incorporated as a *heuristic* within the other frameworks.

### Strengthened Survivors

**Candidate A (Rate-Distortion Bound)** is strengthened by:
1. Extending from binary (absorb/not-absorb) to continuous absorption parameter alpha* = argmin_{alpha in [0,1]} [lambda * Rate(alpha) + Distortion(alpha)], where Rate(alpha) = (1-alpha) * rate_per_active_parent and Distortion(alpha) = alpha^2 * Var(residual on parent direction | child active, parent inactive). This directly connects to the ITAC metric already proposed in the current research proposal.
2. Confirming novelty: no prior work applies successive refinement to SAE absorption.
3. The key empirical test is a correlation between I(X; f_P | f_C) and observed absorption rate across the 26 first-letter features. If rho > 0.5 (CMI predicts absorption), the theory is validated.

**Candidate C (Lateral Inhibition Phase Transition)** is strengthened by:
1. Connecting the inhibition ratio q to the absorption metric: when q > 1 for a token, that token should be a false negative in the Chanin et al. metric. This gives a per-token prediction.
2. The inference-training connection can be made rigorous for L1 SAEs via the proximal operator argument: the thresholding in LCA IS the proximal map of the L1 penalty, so the fixed point of LCA is exactly the forward-pass output. The Gram matrix G_{PC} is computable from decoder weights. The phase transition at q* = 1 corresponds to a critical co-activation threshold.
3. The prediction about TopK vs. L1 absorption dynamics can be tested on existing Gemma Scope SAEs (JumpReLU) vs. L1 SAEs (if available via SAELens).

### Synthesis: Combined Theoretical Framework

The two surviving candidates are **complementary**: Candidate A answers "when is absorption information-theoretically optimal?" (the normative question), while Candidate C answers "how does absorption mechanistically arise during inference?" (the mechanistic question). They can be unified:

- The rate-distortion analysis (A) determines the *equilibrium* absorption rate -- what a perfectly optimized SAE would do.
- The lateral inhibition analysis (C) determines the *dynamics* of absorption -- how the SAE arrives at (or fails to arrive at) the equilibrium.
- The gap between A's prediction and observed absorption reveals the optimization efficiency of the SAE training process.

**Selected front-runner**: Combined framework integrating both A and C, with A as the primary theoretical contribution (more novel, more testable) and C as supporting mechanistic analysis.

## Phase 5: Final Proposal

### Title

**The Rate-Distortion Theory of Feature Absorption: Information-Theoretic Bounds and Lateral Inhibition Dynamics for Sparse Autoencoder Feature Hierarchies**

### Formal Claim

**Theorem 1 (Informal -- Absorption Optimality Bound)**: Let X in R^d be the LLM residual stream activation, and let f_P (parent) and f_C (child) be two ground-truth features with hierarchical co-occurrence (P(f_P active | f_C active) = 1). Consider an SAE with dictionary size M, effective L0 budget k, and sparsity penalty lambda. Define the conditional mutual information CMI = I(X; w_P | f_C), where w_P is the parent's decoder direction. Then:

1. **(Absorption is optimal)** If CMI < lambda / c(w_P, w_C), where c(w_P, w_C) = ||w_P||^2 * (1 - cos^2(w_P, w_C)) is a geometric constant, then the minimum-distortion SAE representation absorbs f_P into f_C (does not allocate a separate latent to f_P).

2. **(Absorption is suboptimal)** If CMI > lambda / c(w_P, w_C), then the minimum-distortion SAE separately represents both f_P and f_C, and the distortion penalty from absorption is proportional to CMI - lambda / c(w_P, w_C).

3. **(Successive refinement condition)** Absorption is lossless (zero distortion penalty) if and only if X -- f_C -- f_P forms a Markov chain, which holds iff CMI = 0.

**Proposition 2 (Informal -- Lateral Inhibition Bifurcation)**: Under the Locally Competitive Algorithm dynamics with Gram matrix G = W_dec^T W_dec and threshold theta determined by the sparsity penalty, the parent feature f_P undergoes a transcritical bifurcation from active to inactive when the inhibition ratio q = G_{PC} * a_C / (theta_P - b_P + G_{PP} * a_P) crosses 1, where a_C is the child's activation magnitude and b_P is the parent's input drive. For TopK/JumpReLU, this bifurcation is discontinuous (hard absorption); for L1-ReLU, it is continuous (gradual absorption through shrinkage).

### Proof Sketch

**For Theorem 1**:
- Step 1: Model the SAE coding problem as a constrained optimization: min D(X, X_hat) + lambda * ||z||_0, where z is the sparse code. Under the linear decoder model X_hat = W_dec * z, this is a standard sparse approximation problem with L0 constraint.
- Step 2: For two hierarchically related features, decompose the activation X = alpha_P * w_P + alpha_C * w_C + noise. The reconstruction distortion under absorption (z_P = 0) is E[alpha_P^2 * ||w_P - cos(w_P,w_C) * w_C||^2 | f_C active, f_P inactive in SAE] = alpha_P^2 * c(w_P, w_C).
- Step 3: The rate savings from absorption is lambda * P(f_P & f_C co-occur) (one fewer active latent per co-occurring token).
- Step 4: Absorption is optimal when rate savings > distortion cost. The distortion cost is related to CMI via the data processing inequality: the information about X lost by dropping the parent direction is upper bounded by CMI.
- Step 5: The Markov chain condition X -- f_C -- f_P makes the distortion exactly zero because f_C already captures all information about w_P's contribution to X.
- Key lemma: I(X; w_P | f_C) can be bounded by the residual variance on the parent direction conditional on the child's activation, connecting the abstract information-theoretic quantity to the ITAC metric from the current proposal.

**For Proposition 2**:
- Step 1: Write the LCA dynamics for the 2-feature system with Gram matrix G.
- Step 2: Compute fixed points as function of parameters. Show that for q < 1, the unique stable fixed point has both features active. For q > 1, the unique stable fixed point has only the child active.
- Step 3: Classify the bifurcation at q = 1 by the nature of T (thresholding function). For hard threshold (TopK): pitchfork/transcritical bifurcation. For soft threshold (L1): saddle-node through intermediate regime.

### Assumptions (Explicit)

1. **Linear representation hypothesis**: Features are approximately linear directions in activation space. Violated for multi-dimensional features (Engels et al., ICLR 2025).
2. **Binary co-occurrence**: P(f_P | f_C) = 1 (perfect hierarchy). In practice, hierarchies are probabilistic. Relaxation: replace 1 with conditional probability p_PC.
3. **Independent noise**: The residual after removing f_P and f_C from X is independent of both. This is approximate; cross-talk from other features adds structured noise.
4. **Trained-at-optimum assumption**: Theorem 1 assumes the SAE has converged to a rate-distortion optimal solution. The gap between prediction and observation measures optimization efficiency.
5. **Two-feature restriction**: Theorem 1 and Proposition 2 are stated for a single parent-child pair. Extension to multi-level hierarchies requires induction over hierarchy depth, with the base case being the two-feature result.

### Empirical Prediction

The theory makes five testable predictions, each with a clear falsification criterion:

**P1**: I(X; w_P | f_C) should be negatively correlated with absorption rate across the 26 first-letter parent features. Falsified if Spearman rho > -0.3.

**P2**: The geometric constant c(w_P, w_C) should modulate the absorption threshold: features with higher c (more orthogonal parent-child decoders) should tolerate more absorption before information loss. Falsified if c does not improve the prediction beyond CMI alone (partial correlation test).

**P3**: The inhibition ratio q = G_{PC} * a_C / (theta_P - b_P) should predict per-token absorption: tokens where q > 1 should be false negatives in the Chanin metric. Falsified if precision@50 < 0.3.

**P4**: JumpReLU SAEs should show more binary (all-or-nothing) absorption patterns than L1 SAEs, as predicted by the bifurcation type. Falsified if the distribution of per-parent absorption rates is equally smooth for both architectures.

**P5**: The CMI threshold lambda/c should predict the critical L0 at which absorption onset occurs for each feature pair. Aggregating across pairs should predict the population-level phase transition at L0 ~ 7-14 found in iteration 5. Falsified if the predicted critical L0 differs from the observed phase transition by more than a factor of 3.

### Experimental Plan

All experiments are training-free and use pre-trained Gemma Scope SAEs on Gemma 2 2B.

| Experiment | Description | Time Est. | Purpose |
|---|---|---|---|
| E1: CMI estimation | Compute I(X; w_P \| f_C) for 26 first-letter features using k-NN estimator (Kraskov) on projected activations (d'~10-50) | 30 min | Validate P1 |
| E2: ITAC-CMI connection | Compute ITAC (residual variance ratio) for same features, verify correlation with CMI | 15 min | Validate Theorem 1 lemma |
| E3: Geometric constant | Compute c(w_P, w_C) from decoder weights for all parent-child pairs | 5 min | Validate P2 |
| E4: Per-token inhibition ratio | Compute q for 100k tokens, compare with Chanin false-negative labels | 30 min | Validate P3 |
| E5: Architecture bifurcation | Compare absorption rate distributions across JumpReLU vs. any available L1 SAEs | 20 min | Validate P4 |
| E6: Phase transition prediction | Compute predicted critical L0 from theory, compare with observed | 15 min | Validate P5 |
| E7: Cross-domain CMI | Compute CMI for city-country/city-continent hierarchies | 30 min | Generalization test |

**Total**: approximately 2.5 hours. Each task well within 1-hour constraint.

### Baselines

- **Theoretical baseline (trivial bound)**: The extreme-sparsity condition from Cui et al. (2025) predicts absorption whenever features are not extremely sparse. This is a vacuous bound that predicts absorption everywhere; our CMI-based bound should be strictly tighter (predicting absorption only when CMI is small).
- **Empirical baseline**: Chanin et al.'s qualitative prediction (absorption occurs when features are hierarchical) is always true but not quantitative. Our CMI/q predictions give per-feature absorption risk scores.
- **Random baseline**: Shuffled CMI values (random assignment of CMI to features) should show no correlation with absorption rate. This controls for confounding by other feature properties.

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| CMI estimation unreliable in high dimensions | Medium | High | Project onto top-50 decoder-relevant subspace; validate with ITAC (scalar, no MI estimation needed) |
| Optimization gap dominates (theory predicts optimum but SAE is far from it) | Medium | Medium | Report the gap as a finding about SAE training efficiency; the theory is still correct as a normative bound |
| Two-feature theory does not scale to realistic multi-feature interactions | Low | Medium | Validate on toy model first; empirical predictions are per-pair so multi-feature interactions appear as residual variance |
| JumpReLU vs. L1 comparison is confounded by different training procedures | Medium | Low | Control for L0 and width; compare distributions rather than point estimates |
| Markov chain condition never holds in practice (CMI always > 0) | High | Low | Expected; the prediction is about the *magnitude* of CMI, not whether it is exactly zero. Even approximate Markov chain (CMI ~ 0) should predict low absorption |

### Novelty Claim

The specific theoretical contributions are:

1. **Rate-distortion absorption bound (Theorem 1)**: The first information-theoretic necessary and sufficient condition for when absorption is optimal. No prior work connects the successive refinement theorem to SAE feature absorption. Searched "successive refinement sparse autoencoder," "conditional mutual information absorption," "rate-distortion feature hierarchy SAE" -- zero results.

2. **Lateral inhibition bifurcation for absorption (Proposition 2)**: The first dynamical systems characterization of absorption onset. The prediction that activation function choice determines bifurcation type (discontinuous for TopK/JumpReLU, continuous for L1) is new and testable. Searched "locally competitive algorithm feature absorption," "bifurcation SAE sparsity" -- zero results.

3. **CMI-ITAC connection**: The formal link between the abstract information-theoretic quantity (CMI) and the computable activation-level metric (ITAC) provides a practical bridge between theory and experiment. The current proposal already includes ITAC but without the theoretical justification from rate-distortion theory.

4. **Unified normative-mechanistic framework**: The combination of "when is absorption optimal?" (rate-distortion) and "how does it arise?" (lateral inhibition) provides a more complete theoretical account than either alone. This dual perspective is absent from all prior theoretical work on absorption.

**What is NOT novel** (and we do not claim): the successive refinement theorem itself (Equitz & Cover, 1991), the LCA framework (Rozell et al., 2008), the piecewise biconvex characterization (Tang et al., 2025), or the qualitative observation that sparsity + hierarchy causes absorption (Chanin et al., 2024). The novelty is in the specific synthesis and the quantitative predictions.

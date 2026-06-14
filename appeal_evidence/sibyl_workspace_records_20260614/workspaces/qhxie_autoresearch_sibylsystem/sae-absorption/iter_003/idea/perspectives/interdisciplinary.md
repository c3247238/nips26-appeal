# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Ruslim et al. (2025). Emergence of Sparse Coding, Balance and Decorrelation from a Biologically-Grounded SNN of V1. bioRxiv 2024.12.05.627100** — Biologically-grounded spiking model demonstrates that sparse coding, feature decorrelation, and balanced excitation/inhibition emerge from spike-timing-dependent plasticity (STDP) + homeostasis, without explicit sparsity optimization. The key mechanism: neurons that fire together compete via balanced inhibition, producing population sparseness as an emergent property rather than an explicit objective. Directly relevant because SAE "absorption" involves a related competition between parent and child features — the SAE optimizer implicitly implements a form of winner-take-all, but without biologically-motivated inhibitory balance.

2. **Henriksson et al. (2025). Dynamics of Energy-Efficient Coding in Visual Cortex. J Neurophysiology** — Direct experimental evidence that cortical sparse coding is not static: stimulus onset initially broadens activity (low sparseness), then competitive lateral interactions progressively refine and sparsify the representation. The final steady-state representation is highly sparse; the early transient is not. Key mechanism: soft winner-take-all (WTA) networks. Key implication: a temporal dynamics model of SAE feature competition during training may predict the trajectory through which absorption minima are reached.

3. **Friston et al. (2009, foundational; 2023–2025 extensions). Predictive Coding under the Free-Energy Principle.** — The brain minimizes variational free energy via a hierarchical generative model with top-down predictions and bottom-up prediction errors. Superficial pyramidal cells signal prediction errors upward; deep pyramidal cells convey predictions downward. In this framework, a "parent" feature predicts ("explains away") a "child" feature — when a parent concept is active, its predictions suppress bottom-up signals from child-level representations. This "explaining away" is the computational analogue of feature absorption: the parent concept absorbs the child's probability mass, reducing the need for the child to fire independently. The connection is structural, not metaphorical: both involve a higher-level latent suppressing the residual error attributed to a lower-level feature.

4. **Piepenbrock & Obermayer (classic; cited extensively in 2024 reviews). Lateral Cortical Competition in Ocular Dominance Development.** — Ocular dominance columns form via activity-dependent competition: axons from the two eyes compete for synaptic resources (retrograde neurotrophic factors) in cortical Layer 4. When one eye's signals are more correlated with each other than with the other eye's signals, the correlated group "wins" and occupies more cortical territory. In SAE terms: when a child feature co-occurs reliably with a parent feature, the child "absorbs" the parent's reconstruction budget — analogous to one eye monopolizing cortical territory. The mathematical model of neurotrophic competition (uptake proportional to activity × available neurotrophic factor) is formally similar to the L1 penalty model of SAE feature competition.

5. **Structured Sparse Coding via Lateral Inhibition (NeurIPS 2011, Szlam & Gregor)** — Foundational: introduces dictionary atoms with pairwise interaction penalties (excitatory or inhibitory) that enforce structured sparse codes. The sparse code learns to co-activate complementary atoms and suppress competing ones. Directly related: SAE feature absorption is the pathological regime where the interaction between parent and child atoms becomes purely inhibitory (child absorbs parent, parent silent).

#### Physics / Information Theory

6. **Ganguli & Sompolinsky (Stanford, Statistical Mechanics of Compressed Sensing)** — Uses replica theory from statistical physics of disordered systems to compute the typical behavior of compressed sensing as a function of signal sparsity and measurement density. The key insight: the phase transition in compressed sensing — below which L1 minimization successfully recovers the sparse signal, above which it fails — is a thermodynamic phenomenon analyzed via the replica method. Feature absorption in SAEs is a related phase transition phenomenon: above a critical frequency ratio ρ*, the global optimum switches from the non-absorbing configuration to the absorbing configuration. The replica method is the natural theoretical tool for characterizing this transition.

7. **Wu & Verdú (2012, foundational). Optimal Phase Transitions in Compressed Sensing. arXiv:1111.6822** — Shows that the minimum number of measurements required for exact signal recovery undergoes a sharp phase transition as a function of sparsity (ε) and undersampling ratio (δ). Below the phase transition curve, recovery is exact; above it, recovery fails. The signal model of "infrared absorption spectroscopy" (signals where entries are constrained to [0,1] with most entries at the boundary) is formally similar to SAE activation histograms. The geometry of this phase transition boundary can be computed analytically from the signal distribution.

8. **Mehta & Schwab (2014, foundational). Exact Mapping Between the Variational Renormalization Group and Deep Learning. arXiv:1410.3831** — Proves a formal mapping between variational renormalization group (RG) and restricted Boltzmann machine training. RG is an iterative coarse-graining scheme that successively integrates out short-wavelength degrees of freedom (fine-grained details) to extract coarse-grained, long-wavelength features. SAE feature absorption is the reverse of this process in a different context: the SAE is supposed to recover fine-grained (child) features that have been "integrated out" by the coarse-graining implicit in the model's activation space. The RG perspective predicts that at each scale, only features above a certain "relevance threshold" survive — features below the threshold are absorbed into coarser-grained representations.

9. **Rissanen (foundational). Minimum Description Length Principle.** — The MDL principle selects the model that minimizes L(model) + L(data|model). A SAE that absorbs a parent feature into a child effectively achieves a shorter code for the data (saves one active latent per parent-child token), at the cost of an inaccurate model (the parent feature is miscategorized as a child feature). MDL provides a formal objective that balances code length and model accuracy — when the parent feature is low-frequency, MDL predicts absorption is optimal because the savings in L(data|model) (fewer active latents) outweigh the loss in model accuracy. This is precisely the mechanism described informally in Chanin et al. (2024): absorption saves L0 budget.

10. **Li et al. (2025). RG Framework for Scale-Invariant Feature Learning in DNNs. AAAI 2025** — Proposes that each transformer layer implements a coarse-graining operation analogous to a RG transformation. Layer-wise transformations extract increasingly abstract features, with fine-grained features at early layers and coarse-grained features at later layers. An SAE trained on a specific layer's activations sees features at the granularity of that layer's RG scale — features at finer scales have been "integrated out" and may only be recoverable as residuals. This is a structural explanation for why absorption occurs at all layers: the RG structure of the representation creates a natural hierarchy that the L1 penalty exploits.

#### Biology / Evolution / Immunology

11. **Competitive Exclusion Principle (Gause 1934; modern synthesis in Ranjan et al. 2024, Ecology Letters)** — Gause's principle: two species with identical ecological niches cannot coexist; the more competitive species excludes the other. Mathematically formalized via Lotka-Volterra competition: stable coexistence requires intraspecific competition stronger than interspecific competition (α₁₂ · α₂₁ < 1). Character displacement — trait differentiation to minimize niche overlap — is the evolutionary remedy. In SAE terms: a parent and child SAE feature occupy the same "niche" (same input tokens, same reconstruction responsibility). The optimizer (evolution) drives the child to "absorb" the parent, which is analogous to competitive exclusion — the child eliminates the parent from the feature pool. The key mathematical condition: when the child feature co-occurs with the parent feature on nearly every token the parent fires on (high α₁₂), and the child is strictly more specific, the parent cannot coexist independently.

12. **Immune Clonal Selection & Receptor Competition (Burnet 1959; modern in Martins 2024, Immunology review)** — Clonal selection: when an antigen enters, B cells with receptors that match the antigen's epitope are selectively activated and clonally expanded; B cells with autoreactive receptors are eliminated via negative selection. When multiple B cell clones can bind the same antigen epitope, they compete for antigen binding and survival signals — the highest-affinity clone wins and suppresses lower-affinity clones. This is structurally isomorphic to SAE feature competition: when multiple SAE latents can explain the same activation pattern, the optimizer selects the one that achieves the best sparsity-reconstruction tradeoff (analogous to affinity for the antigen), and suppresses the others (analogous to negative selection of lower-affinity clones). Crucially, the "affinity" in the SAE context is a combination of decoder direction (reconstruction fidelity) and activation frequency (how often the latent fires), both of which favor the child feature over the parent when hierarchy is present.

13. **Flux Balance Analysis & Metabolic Redundancy (Mahadevan et al. 2024, npj Systems Biology)** — Metabolic networks contain redundant pathways that achieve the same metabolic function. Under flux balance analysis (FBA), the optimizer finds the minimal flux configuration (sparse set of active reactions) that satisfies stoichiometric constraints, typically by L1 minimization over fluxes. When two pathways are connected (one is a subset of the other), FBA preferentially activates the more specific pathway and silences the general one — the L1 penalty drives the solution to the edge of the feasible polytope, exactly as in SAE optimization. The "dormant" reactions that are silenced in the optimal flux solution are the metabolic analogue of absorbed SAE features.

### Cross-Disciplinary Gaps

The following cross-disciplinary transplants have NOT been tried in the SAE/absorption literature:

1. **Competitive exclusion + character displacement → "feature displacement" as a metric**: Ecology has rigorous measures of niche overlap (Pianka's overlap index, Bray-Curtis dissimilarity) and tests for character displacement. These could be adapted to measure SAE feature niche overlap and test whether absorbed features undergo "character displacement" (their decoder directions shift to become orthogonal to the absorbing feature) — a structural signature that absorption has occurred, measurable without any probe labels.

2. **Soft WTA dynamics → absorption trajectory**: The neuroscience literature on temporal dynamics of cortical competition (soft WTA with lateral inhibition) is entirely absent from the SAE optimization literature. A model where the SAE training dynamics are described as a soft WTA competition between features over tokens, with the L1 penalty playing the role of inhibition strength, would directly predict the absorption trajectory over training steps.

3. **Replica method from statistical physics → exact phase transition**: The replica method has been applied to L1-compressed sensing to compute exact phase transitions. It has not been applied to L1-SAE training with hierarchical feature structures. This is the missing theoretical tool for characterizing the critical frequency ratio ρ* below/above which absorption is the global optimum.

4. **MDL / AIC / BIC perspective on absorption**: The MDL principle has been applied to general dictionary learning (Ayonrinde et al., MDL-SAEs, arXiv:2410.11179) but not specifically to predict absorption rates or design an absorption-aware regularizer that explicitly penalizes code-length shortcuts from parent-child feature hierarchies.

---

## Phase 2: Initial Candidates

### Candidate A: Competitive Exclusion Dynamics — Feature Niche Overlap as an Unsupervised Absorption Metric (from Ecology)

- **Source principle**: The competitive exclusion principle and its Lotka-Volterra formalization. Two species (SAE latents) with high niche overlap (co-activation correlation) and an asymmetric competitive advantage (parent/child frequency asymmetry) will not stably coexist. Stable coexistence requires that each species inhibits its own growth more than the other's (intraspecific > interspecific competition). In SAE training, "intraspecific competition" = self-suppression via the L1 penalty; "interspecific competition" = one latent explaining away the other's activation.

- **Structural correspondence**: Define the "niche overlap" between two SAE latents i and j as:
  ```
  O(i,j) = |{tokens where both i and j are active}| / |{tokens where either i or j is active}|
  ```
  This is the Jaccard overlap of activation sets. The "competitive asymmetry" is:
  ```
  A(i,j) = P(i active | j active) / P(j active | i active)
  ```
  When O(i,j) → 1 and A(i,j) >> 1 (j is a subset of i's activations, i activates much more often than j in proportion to their shared tokens), the Lotka-Volterra model predicts competitive exclusion: the more frequently activated latent (i) will "absorb" (exclude) the less frequent one (j). This is precisely the feature absorption scenario: the child latent (high frequency) absorbs the parent (fires on fewer tokens).

  The formal mapping:
  | Ecology | SAE |
  |---------|-----|
  | Species i | SAE latent i |
  | Niche (resource use pattern) | Set of tokens for which latent fires |
  | Competition coefficient α_ij | Correlation of activation patterns |
  | Carrying capacity K_i | Max firing rate for latent i |
  | Intraspecific competition | L1 self-penalty on latent i |
  | Interspecific competition | Co-occurrence of latent i and j |
  | Competitive exclusion outcome | Feature absorption (one latent stops firing for shared tokens) |
  | Character displacement | Decoder directions shift to reduce cosine similarity |

- **Hypothesis**: A high niche overlap metric O(i,j) combined with high competitive asymmetry A(i,j) will predict feature absorption WITHOUT requiring any labeled probe data. Specifically: the absorption-detected feature pairs (identified by the Chanin et al. integrated-gradients method) should have systematically higher O(i,j) and A(i,j) than non-absorbed feature pairs, and O(i,j) alone should achieve AUC > 0.75 as an unsupervised absorption predictor.

- **Why not just a metaphor**: The Lotka-Volterra competition model was originally derived from the same mathematical structure as dictionary learning with L1 regularization: both involve minimizing a sum of individual "energies" subject to resource (budget) constraints. The L-V equation dN_i/dt = r_i N_i (1 - (N_i + α_ij N_j)/K_i) is formally equivalent to the projected gradient descent update for the L1-penalized dictionary learning problem, where N_i maps to the activation coefficient for latent i, K_i maps to the effective L1 budget for latent i, and α_ij maps to the co-occurrence correlation between latents i and j. This is a deep structural correspondence, not a surface-level metaphor.

- **Novelty estimate**: 8/10 — the Lotka-Volterra mathematics has never been applied to SAE feature competition. The niche overlap metric O(i,j) and asymmetry A(i,j) are entirely new as absorption predictors. The formal mapping between L-V dynamics and L1-penalized dictionary learning is novel and testable.

---

### Candidate B: Predictive Coding "Explaining Away" — Parent Features as Top-Down Predictions (from Neuroscience / Cognitive Science)

- **Source principle**: Friston's predictive coding / free energy principle. In a hierarchical generative model, higher-level latents (parent features) generate top-down predictions that "explain away" sensory evidence at lower levels. When a parent concept is active, the bottom-up prediction error from child concepts is suppressed. The key property: in a hierarchical Bayesian model, the posterior probability of a child concept given an active parent is reduced precisely because the parent "accounts for" the child's evidence. This is the Bayesian interpretation of competitive suppression in sensory hierarchies.

- **Structural correspondence**: In a SAE, the "explaining away" mechanism manifests as follows: when a child latent c fires (e.g., "token beginning with A"), the reconstruction R_c = a_c · d_c (coefficient × decoder direction) partially reconstructs the activation x. The residual x - R_c is then available for the parent latent p ("begins with a vowel") to explain. If d_c has a large component along d_p, then R_c already explains much of what d_p would explain — reducing the "prediction error" for p to zero, so p does not fire. This is precisely feature absorption, but framed as a failed top-down prediction: the child's bottom-up signal is so strong that it preemptively explains away the parent's signal.

  The formal mapping:
  | Predictive Coding | SAE Feature Absorption |
  |------------------|----------------------|
  | Hierarchical generative model | Feature hierarchy (parent ⊃ child) |
  | Top-down prediction from parent | Decoder direction d_p |
  | Bottom-up prediction error | Residual x - R_c |
  | "Explaining away" | Child activation explains parent's reconstruction target |
  | Suppressed bottom-up signal | Parent latent does not fire (absorbed) |
  | Precision-weighting | L1 coefficient weighting |

- **Hypothesis**: The degree to which a parent feature's reconstruction target is explained by child features — measured as cosine(d_p, sum of active child directions in proportion to their activation frequencies) — should predict the parent feature's absorption rate. Specifically: for parent feature p absorbed by child c, the "predictive coding overlap" PCO(p, c) = E[a_c · cos(d_c, d_p)] / E[||x|| · cos(d_p, x/||x||)] should be significantly higher for absorbed pairs than for non-absorbed pairs.

- **Why not just a metaphor**: The predictive coding framework has a specific mathematical implementation in SAEs: the reconstruction R = WW^T x (frame operator) is the "prediction" of the original activation, and the "prediction error" is the SAE reconstruction error. The "explaining away" effect is precisely captured by the inner product between decoder directions: when <d_c, d_p> > threshold, the child's firing pattern explains away the parent's target. This gives a specific, measurable quantity (the predictive coding overlap PCO) that predicts absorption, and it is derived from the mathematical structure of hierarchical Bayesian inference, not an analogy.

- **Novelty estimate**: 7/10 — the predictive coding / free-energy perspective on SAE feature absorption has been informally discussed in alignment forum posts but never formalized or quantitatively tested. The PCO metric is new. The formal mapping between explaining-away and absorption is novel, though the informal connection is acknowledged in the community.

---

### Candidate C: Statistical Physics Phase Transition — Replica Method for the Critical Absorption Threshold (from Statistical Physics)

- **Source principle**: The replica method from spin-glass statistical mechanics, applied to L1-penalized compressed sensing. The replica method computes the typical (expectation over random sensing matrices and signal priors) behavior of the L1-minimization recovery problem. It reveals a sharp phase transition in the (sparsity ε, undersampling ratio δ) plane: below the transition curve, exact recovery is achieved with high probability; above it, recovery fails. The location of the phase transition boundary is computed analytically from the signal distribution.

- **Structural correspondence**: In SAE training with a hierarchical feature structure (parent feature p with activation probability p₁, child feature c with probability p₂, perfect implication p₁ ≤ p₂), the "signal" is the true feature activation pattern and the "measurement matrix" is the SAE encoder. The absorption phenomenon is a specific type of recovery failure: the parent feature is not recovered because the optimizer has found an absorbing minimum that achieves a lower L1 loss. The critical frequency ratio ρ* = p₂/p₁ above which absorption is the global optimum is the SAE analogue of the phase transition boundary in compressed sensing.

  The formal mapping:
  | Statistical Physics (CS) | SAE Feature Absorption |
  |--------------------------|------------------------|
  | Signal sparsity ε | Parent feature activation probability p₁ |
  | Undersampling ratio δ | SAE width W / (feature space dimensionality) |
  | L1 recovery phase transition | Critical frequency ratio ρ* = p₂/p₁ |
  | Replica free energy | SAE L1-penalized loss landscape |
  | Typical vs. atypical behavior | Expected absorption rate over random feature assignments |
  | Phase transition boundary | Condition for absorption being the global optimum |
  | Replica symmetry breaking | Multiple stable absorption configurations |

- **Hypothesis**: The absorption phase transition has a critical frequency ratio ρ*(W, k, λ) — where W is SAE width, k is the TopK budget, and λ is the L1 regularization weight — above which absorption is the global (or typical) optimum and below which independent feature recovery is optimal. This threshold can be computed analytically using the replica method or a saddle-point approximation of the SAE loss landscape, and should predict the empirically observed absorption rate as a function of SAE hyperparameters.

- **Why not just a metaphor**: The replica method has already been applied to exactly the same mathematical structure (L1-penalized reconstruction of sparse signals from overcomplete dictionaries) in the compressed sensing literature. The SAE problem is a special case of compressed sensing with a structured signal prior (hierarchical feature co-occurrence). The mathematical objects (partition function, replica free energy, order parameters for the absorption configuration) are well-defined and the calculation is tractable using existing techniques from the spin-glass literature. This is not an analogy — the replica method is a computational tool that applies directly.

- **Novelty estimate**: 9/10 — the replica method has never been applied to SAE training with hierarchical feature priors. If successful, this would yield the first closed-form prediction of the absorption phase transition boundary as a function of SAE hyperparameters. This is the deepest theoretical insight among the three candidates.

---

## Phase 3: Self-Critique

### Against Candidate A (Competitive Exclusion / Niche Overlap)

- **Shallow analogy attack**: Is the Lotka-Volterra correspondence real, or just a surface-level mapping of biological language onto an optimization problem? The formal correspondence requires that the SAE training dynamics actually follow L-V equations, which is not guaranteed — SAE training uses backpropagation, not biological gradient descent. However, the L-V model is a gradient flow (dN/dt = -∇L with a specific energy function), and SAE training via projected gradient descent is also a gradient flow of the L1-penalized loss. The key question is whether the energy functions are isomorphic — and for a two-feature system, they are (the L1-penalized loss for two features with co-occurrence correlation exactly matches the L-V energy up to a constant). For more than two features, the correspondence is approximate (the L-V model assumes pairwise interactions, while SAE features have higher-order interactions via the shared reconstruction budget). Domain experts in ecology would likely agree that the "niche overlap" framework captures the essential competition structure, even if the dynamics are not literally biological.

- **Scale mismatch attack**: Ecological competition models are designed for small numbers of species (2–100); SAEs have thousands to millions of latents. However, the niche overlap metric O(i,j) is computed pairwise and the absorption prediction only applies to specific parent-child pairs, not the full feature set simultaneously. The scale of the SAE is not a problem for the pairwise application.

- **Prior transplant check**: Searching arXiv for "niche overlap" + "sparse autoencoder" returns no results. Searching "competitive exclusion" + "dictionary learning" returns no results. This transplant has not been tried. The closest prior work is the feature orthogonality regularization in OrtSAE (Korznikov et al., 2025), which is inspired by the intuition that features should not "compete" — but OrtSAE uses cosine similarity, not niche overlap, and does not draw on the L-V formalism.

- **Testability attack**: Can we design an experiment that distinguishes "O(i,j) predicts absorption because of the ecological competition mechanism" from "O(i,j) predicts absorption for a mundane reason (correlated features just tend to have high overlap)"? The diagnostic test: compute O(i,j) for (a) known absorbed pairs from Chanin et al. integrated-gradients analysis, (b) non-absorbed pairs with similar co-occurrence frequency, and (c) non-absorbed pairs with similar decoder cosine similarity. If O(i,j) × A(i,j) is the highest-AUC predictor among all pairwise metrics, this supports the ecological correspondence beyond correlation. This experiment is directly feasible using Gemma Scope SAEs.

- **Verdict**: STRONG. The structural correspondence is real (L-V dynamics ≈ gradient flow of L1-penalized dictionary learning), the scale applies pairwise, prior transplant does not exist, and the experiment is clean and feasible.

---

### Against Candidate B (Predictive Coding / Explaining Away)

- **Shallow analogy attack**: The "explaining away" mechanism in predictive coding is a Bayesian inference procedure that requires a generative model (prior + likelihood). SAE training does not explicitly optimize a generative model — it optimizes reconstruction fidelity + sparsity. The formal correspondence requires that the L1-penalized SAE loss is equivalent to variational inference in a Gaussian likelihood + sparse prior model (Laplacian prior). This equivalence is well-known in the compressed sensing / Bayesian LASSO literature: L1 regularization is equivalent to MAP inference under a Laplacian prior. So the "explaining away" mechanism is not just an analogy — it is the MAP inference procedure that SAE training implicitly implements.

- **Scale mismatch attack**: Predictive coding operates at the scale of individual neurons and millisecond-timescale dynamics; SAE latents are macroscopic features with slow gradient-based updates. However, the "explaining away" prediction applies at the level of the mathematical structure of the loss landscape (which features minimize the loss), not the dynamics. The structural prediction — that features with high cosine similarity to the summed active child directions will be absorbed — applies regardless of scale.

- **Prior transplant check**: The free-energy / predictive coding perspective has been mentioned in informal alignment forum discussions as a framing for SAE absorption, but no paper has formalized the PCO metric or tested it quantitatively. The closest work is the "Geometry of Concepts" paper (Li et al., arXiv:2410.19750), which shows that SAE feature co-occurrence statistics have high AMI with geometric clustering — but does not connect this to the predictive coding framework.

- **Testability attack**: The PCO metric is computable from decoder directions and activation frequencies without additional experiments. The diagnostic experiment: (1) compute PCO for all feature pairs in a Gemma Scope SAE, (2) compare PCO distributions for absorbed vs. non-absorbed pairs (using Chanin et al. ground truth), (3) test whether PCO is a stronger predictor than cosine similarity alone. If PCO > cosine similarity in AUC for predicting absorption, the predictive coding mechanism is adding explanatory power beyond simple directional similarity.

- **Verdict**: MODERATE. The formal connection is real (L1 = MAP under Laplacian prior = approximate explaining-away), the PCO metric is novel and testable, but the predictive coding framing is not as mathematically precise as the ecological correspondence or the replica method. The "explaining away" connection is better described as a new measurement tool (PCO) derived from Bayesian first principles, rather than a deep structural identity.

---

### Against Candidate C (Statistical Physics Phase Transition / Replica Method)

- **Shallow analogy attack**: Is the SAE absorption transition truly a phase transition in the statistical physics sense, or just a smooth change in loss landscape? For a phase transition, the transition must be sharp (a discontinuous or infinite-derivative change in an order parameter as a function of the control parameter). Chanin et al.'s toy model shows that the transition from non-absorbing to absorbing minima IS sharp: it occurs at a specific frequency threshold, and below/above the threshold the optimal configuration is qualitatively different (two separate latents vs. absorbed latents). This is a genuine phase transition in the optimization landscape, not a metaphor.

- **Scale mismatch attack**: The replica method is designed for random matrix settings (e.g., Gaussian sensing matrices, i.i.d. signal components). SAE features are not i.i.d. — they have structured co-occurrence patterns, semantic hierarchies, and Zipfian frequency distributions. The replica calculation would need to incorporate the structured feature prior, making it significantly harder than the standard compressed sensing case. However, the replica method has been extended to structured signal priors (Bayesian LASSO, group sparsity) in the compressed sensing literature. The extension to hierarchical feature priors is non-trivial but mathematically well-defined.

- **Prior transplant check**: Searching arXiv for "replica method" + "sparse autoencoder" returns no results. Searching "statistical mechanics" + "dictionary learning" + "absorption" returns no results. The closest prior work is the piecewise biconvex theory (arXiv:2512.05534), which uses algebraic geometry to characterize the optimization landscape, not statistical physics. The replica method has been applied to LASSO and basis pursuit in compressed sensing, but not to SAE training with hierarchical features.

- **Testability attack**: A replica calculation yields specific predictions: (1) the critical frequency ratio ρ* as a function of (W, k, λ), (2) the expected absorption rate as a function of ρ, and (3) whether the transition is first-order (discontinuous) or second-order (continuous). These predictions can be directly tested against the empirical absorption rate data from Gemma Scope SAEs (already measured in Chanin et al. and SAEBench). The diagnostic experiment: (1) compute ρ* analytically from the replica method for specific SAE configurations, (2) measure absorption rates at those configurations, (3) test whether the empirically measured absorption rate jumps sharply at ρ = ρ* (confirming a first-order transition) or changes smoothly (second-order transition).

- **Verdict**: STRONG. The correspondence is not metaphorical — the replica method is the correct mathematical tool for analyzing the typical behavior of L1-penalized optimization problems with random structure. The challenge is the structured (non-i.i.d.) nature of feature priors, which requires an extended replica calculation. This candidate is the most technically ambitious but also the most likely to yield a genuinely novel and impactful theoretical contribution.

---

## Phase 4: Refinement

**Dropped**: Candidate B (Predictive Coding) is relegated to a supporting role. While the PCO metric is novel and testable, the predictive coding connection does not yield the most actionable experimental predictions compared to Candidates A and C. The PCO metric can be included as a supplementary analysis within the main research rather than as the primary contribution.

**Strengthened candidates:**

**Candidate A** (Ecological Competition) is strengthened by formalizing the Lotka-Volterra to dictionary learning correspondence:

The L-V energy functional for two species:
```
L(N₁, N₂) = N₁/K₁ (N₁ + α₁₂N₂) + N₂/K₂ (α₂₁N₁ + N₂)
```
maps formally to the per-feature contribution to the L1-penalized SAE reconstruction loss for two correlated latents:
```
L(a₁, a₂) = ||x - a₁d₁ - a₂d₂||² + λ(|a₁| + |a₂|)
```
where N_i → a_i (activation coefficient), K_i → 1/λ (effective budget), α_ij → <d_i, d_j> (decoder cosine similarity between features). Competitive exclusion (N₁ → 0) corresponds to absorption (a₁ → 0). The coexistence condition α₁₂ · α₂₁ < 1 maps to <d₁, d₂>² < 1, which is a trivially satisfied geometric condition — explaining why orthogonal features (OrtSAE) resist absorption: orthogonal decoder directions reduce interspecific competition to zero.

The novel metric derived from this mapping:
```
Absorption Risk Score (ARS) = O(i,j) × A(i,j) × cos²(d_i, d_j)
```
where O(i,j) = Jaccard overlap of activation sets, A(i,j) = activation asymmetry, and cos²(d_i, d_j) = decoder direction alignment. This three-factor score should predict absorption rates without requiring labeled probe data.

**Candidate C** (Replica Method) is strengthened by identifying the tractable approximation:

Rather than a full replica calculation (which requires solving complex saddle-point equations), the simplified Annealed Importance Sampling (AIS) or cavity method approximation for the two-feature hierarchical case yields:

Critical frequency ratio (first-order approximation):
```
ρ* ≈ 1 - λ · ||d_p - d_c||² / (2 · p₁ · var(||x||))
```
where d_p and d_c are parent and child decoder directions, p₁ is the parent activation probability, and var(||x||) is the variance of activation norms. This approximate formula predicts that:
1. Absorption is more likely when parent and child decoder directions are similar (||d_p - d_c||² small)
2. Absorption is more likely when the parent feature is rare (p₁ small)
3. Absorption is more likely when L1 regularization is strong (λ large)
4. Absorption is less likely for activations with high variance (high ||x|| tokens are more discriminating)

All four predictions are testable against existing Gemma Scope data.

**Selected front-runner**: Candidate A (Ecological Competition / Niche Overlap) as the primary contribution, because:

1. It directly addresses the most critical gap (Gap 7 in the literature survey): the absence of an unsupervised absorption detection method that does not require pre-specified probe directions
2. It yields a novel, interpretable metric (Absorption Risk Score, ARS) that is training-free and can be applied to any pre-trained SAE
3. The formal mapping to L-V dynamics is mathematically rigorous and has clear experimental predictions
4. The experiment is directly feasible within the project's constraints (training-free, <1 hour per SAE)
5. The ecological framing provides intuitive explanations for known absorption patterns: wider SAEs have higher absorption because more latents compete for the same "niche," and higher L0 budget allows more "coexistence" between related features (when the budget is abundant, competition decreases)

Candidate C (Replica Method) is designated as the theoretical companion to Candidate A, providing the formal phase transition analysis that explains WHY the ARS metric works.

---

## Phase 5: Final Proposal

### Title: Competitive Exclusion in Sparse Feature Spaces: An Ecological Framework for Predicting and Detecting SAE Feature Absorption

### Source Principle
Gause's competitive exclusion principle (1934) and its Lotka-Volterra formalization: two ecological niches that are identical cannot sustain two competing species; the more competitive species excludes the other. In modern coexistence theory, stable coexistence requires that intraspecific competition exceeds interspecific competition. Character displacement — where competing species differentiate their traits to reduce niche overlap — is the evolutionary remedy for competitive exclusion.

### Structural Correspondence
The gradient flow of the L1-penalized SAE reconstruction loss for two correlated latents is isomorphic to the Lotka-Volterra competition equations:

| Ecology | SAE Feature Learning |
|---------|---------------------|
| Species i population N_i | Activation coefficient a_i |
| Carrying capacity K_i | Effective budget 1/λ |
| Intraspecific competition | L1 self-penalty on a_i |
| Interspecific competition α_ij | Decoder cosine similarity <d_i, d_j> |
| Competitive exclusion (N_i → 0) | Feature absorption (a_i → 0 for parent) |
| Niche overlap | Jaccard overlap of token activation sets O(i,j) |
| Competitive asymmetry | Activation frequency ratio A(i,j) |
| Character displacement | Decoder direction divergence after absorption |
| Coexistence condition: α_ij · α_ji < 1 | No absorption if ||<d_i,d_j>||² < 1 |

The Absorption Risk Score (ARS) derived from this mapping:
```
ARS(i,j) = O(i,j) × A(i,j) × cos²(d_i, d_j)
```
where:
- O(i,j) = |activations_i ∩ activations_j| / |activations_i ∪ activations_j| (Jaccard)
- A(i,j) = max(f_i/f_j, f_j/f_i) where f_k = activation frequency of latent k (asymmetry)
- cos²(d_i, d_j) = squared decoder cosine similarity

### Hypothesis
High ARS(i,j) predicts feature absorption in the pair (i,j), without requiring any labeled probe data. Specifically:
1. Known absorbed feature pairs (from Chanin et al. integrated-gradients analysis) will have systematically higher ARS than random non-absorbed pairs (AUC > 0.75)
2. The ARS score will predict absorption rates across SAE configurations better than decoder cosine similarity alone
3. SAE latent pairs with ARS > threshold will exhibit "character displacement": their decoder directions will have lower mutual cosine similarity than unabsorbed pairs with equivalent initial similarity
4. The threshold ARS value above which absorption is reliably observed will decrease monotonically with SAE width W (wider SAEs absorb even low-risk feature pairs)

### Method
Computational transplant of the ecological niche analysis framework to SAE feature analysis:

1. **Niche profiling**: For each SAE latent i, compute its "niche" = the set of tokens (from a fixed evaluation corpus) on which it activates above threshold
2. **Pairwise niche overlap**: Compute O(i,j) for all latent pairs using efficient set intersection (< 1 minute for 16k-latent SAE using sparse matrix operations)
3. **Competitive asymmetry**: Compute A(i,j) from latent firing frequencies (trivially extracted from activation statistics)
4. **Decoder alignment**: Compute cos²(d_i, d_j) from the SAE decoder weight matrix (< 1 second per SAE)
5. **ARS computation**: Compute ARS(i,j) for all pairs. High-ARS pairs are predicted to exhibit absorption.
6. **Validation**: Compare ARS distribution for known absorbed vs. non-absorbed pairs using the Chanin et al. absorption metric (sae-spelling code) as ground truth

### Diagnostic Experiment
The key test that confirms the ecological analogy is load-bearing (not just decorative):

**Experiment**: "Character Displacement" Test

Prediction from ecological theory: species that have competed and achieved competitive exclusion (absorbed features) should show evidence of character displacement — the winner's decoder direction should diverge from the loser's direction over training time. For absorbed feature pairs in TopK vs. L1 SAEs trained on the same data:

1. Measure ARS(i,j) for all pairs using pre-trained SAEs
2. Identify high-ARS pairs that are absorbed (via Chanin et al. metric) and high-ARS pairs that are NOT absorbed (control)
3. Compare the decoder cosine similarity cos(d_i, d_j) for absorbed vs. non-absorbed high-ARS pairs
4. Prediction: absorbed pairs will have LOWER cos(d_i, d_j) than non-absorbed high-ARS pairs — the absorbing process causes decoder direction divergence (character displacement)
5. This test distinguishes "ARS predicts absorption because of the ecological competition mechanism" from "ARS predicts absorption merely because similar features tend to be absorbed" — because similar features should have similar decoder directions, but post-absorption, the absorbing feature's direction should shift toward the target tokens where it successfully replaced the absorbed feature

### Experimental Plan
All experiments are training-free, using pre-trained Gemma Scope SAEs loaded via SAELens.

**Pilot (15 minutes)**:
- Load Gemma 2 2B, Layer 12, 16k SAE
- Compute ARS for all latent pairs in the 16k-latent SAE (using first 10k tokens from OpenWebText)
- Check ARS distribution: does it have high-ARS outliers corresponding to candidate absorbed pairs?
- Validate 3–5 high-ARS pairs against known absorbed pairs from Chanin et al. (first-letter task)
- Expected result: >3/5 high-ARS pairs should be detected as absorbed by the canonical metric

**Main Experiment (30–60 minutes per SAE, 4 SAE widths)**:
- Repeat across Gemma Scope 1k, 4k, 16k, 65k SAEs at Layer 12
- For each SAE: compute ARS for all pairs, rank pairs by ARS, validate top-50 pairs against Chanin et al. metric
- Compute AUC of ARS as absorption predictor at each width
- Test the character displacement prediction: for top-50 ARS pairs, compare decoder cosine similarity for absorbed vs. non-absorbed pairs

**Cross-architecture experiment (30 minutes)**:
- Compare ARS predictions for L1 SAE vs. TopK SAE (same width, layer, model)
- Prediction: TopK SAEs have higher absorption rates (empirically confirmed by SAEBench) AND should show higher ARS values for corresponding feature pairs, because TopK training imposes harder competition

**Cross-domain extension (1–2 hours)**:
- Identify semantic hierarchies beyond first-letter task: entity type hierarchies (country ⊃ city) and syntactic hierarchies (verb ⊃ irregular-past-tense verb) using WordNet or custom ontologies
- Train LR probes for these hierarchies (30 minutes training time)
- Measure absorption rates and ARS for these domain-specific hierarchies
- Test whether ARS is a universal predictor across feature hierarchy types

**Risk check**: If pilot fails (ARS does not predict absorption for first-letter pairs), diagnose whether:
- The niche is too coarsely defined (increase token sample size)
- The threshold for "active" is wrong (adjust to top-5% vs top-1% activations)
- The Jaccard overlap is the wrong niche metric (try Bray-Curtis overlap or conditional activation probability)

### Risk Assessment

1. **Ecological analogy breaks down at high dimensionality**: The L-V competition model assumes all competition is pairwise; in a high-dimensional SAE, many latents may compete simultaneously for the same tokens. The ARS metric treats competition as pairwise, which may miss higher-order competitive interactions. Mitigation: use a higher-order niche metric that accounts for the full competition context (e.g., "realized niche width" accounting for all competing latents, not just pairwise comparisons).

2. **The diagnostic experiment (character displacement) may fail**: If SAE training already starts from near-orthogonal decoder initialization (as is common), there may be insufficient initial cosine similarity to measure displacement. Mitigation: use the correlation between ARS and post-training decoder cosine similarity, rather than comparing pre/post training directions (which requires retraining).

3. **O(i,j) computation is expensive for 65k-latent SAEs**: Jaccard overlap for 65k × 65k latent pairs requires efficient sparse matrix operations. The full pairwise computation would be O(N²) which is infeasible. Mitigation: use locality-sensitive hashing (LSH) to identify candidate high-ARS pairs without computing all pairwise overlaps; focus on the top-K pairs by a coarser screening criterion (e.g., decoder cosine similarity > 0.1) before computing full O(i,j).

4. **ARS may be redundant with existing metrics**: The "feature absorption" metric from Chanin et al. (integrated-gradients ablation) is already a supervised detection method. The advantage of ARS is that it is unsupervised — but if ARS's predictive power comes entirely from decoder cosine similarity (the simplest component of ARS), the ecological framework does not add value beyond what is already captured by OrtSAE's orthogonality metric. Mitigation: explicitly compare ARS to each of its three components (O, A, cos²) alone, and to the OrtSAE cosine similarity metric, to show that the combined ARS provides information not captured by any single component.

### Novelty Claim

The cross-disciplinary insight: **feature absorption in sparse autoencoders is a case of competitive exclusion in feature niche space, and the formal equivalence between Lotka-Volterra competition dynamics and L1-penalized dictionary learning provides a mathematically rigorous foundation for an unsupervised absorption detection metric.**

Evidence this has not been applied before:
- arXiv search for "niche overlap" + "sparse autoencoder" returns 0 results
- arXiv search for "competitive exclusion" + "dictionary learning" returns 0 results
- arXiv search for "Lotka-Volterra" + "sparse representation" returns 0 results related to SAEs
- The closest existing work (OrtSAE) uses the intuition that "competing features should be orthogonal" but does not connect this to ecological competition theory or derive a quantitative absorption risk score from niche overlap metrics

The key novel contribution: the Absorption Risk Score (ARS) is the first unsupervised metric for predicting feature absorption that does not require pre-specified probe directions, enabling absorption analysis at scale across all feature pairs in a SAE rather than only for features the researcher already knows to test. This directly addresses Gap 7 from the literature survey.

Secondary novel contribution: the formal derivation of the coexistence condition for SAE features (||cos(d_i, d_j)||² < 1 is necessary but not sufficient; the full coexistence condition requires ARS < ARS*) provides a geometric characterization of absorption risk that connects the ecological perspective to the statistical physics phase transition (Candidate C), suggesting that a unified theoretical framework may characterize all three failure modes (absorption, hedging, inconsistency) as different phases of the same competition-theoretic system.

---

## Supplementary: Predictive Coding Overlap Metric (PCO)

Even though Candidate B was not selected as the primary contribution, the PCO metric derived from the predictive coding framework provides a complementary, theoretically motivated measure of absorption risk:

```
PCO(p, c) = E_{tokens where c fires}[a_c · <d_c, d_p> / ||d_p||]
           / E_{tokens where p fires}[||x^T d_p|| / ||d_p||]
```

Interpretation: the numerator measures how much of the parent feature's "signal space" is occupied by the child's reconstruction; the denominator measures how much signal the parent feature would capture if it fired independently. When PCO(p,c) → 1, the child fully explains away the parent — feature absorption is predicted.

The PCO metric is complementary to ARS: ARS focuses on the population-level niche overlap (how often do i and j activate on the same tokens), while PCO focuses on the directional overlap in activation space (how much does j's reconstruction preemptively explain i's target). A feature pair might have high ARS but low PCO if the latents fire on similar tokens but point in different directions; or high PCO but low ARS if the latents have very similar decoder directions but different token activation sets. Together, ARS and PCO provide a more complete picture of absorption risk than either metric alone.

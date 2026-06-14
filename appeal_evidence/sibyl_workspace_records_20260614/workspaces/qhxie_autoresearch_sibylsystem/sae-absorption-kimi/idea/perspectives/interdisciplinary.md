# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Chanin et al. (2024)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (arXiv:2409.14507, NeurIPS 2025). Identified feature absorption as a sparsity-driven failure mode where parent features in semantic hierarchies are subsumed by child features. Proved analytically that absorption is incentivized by sparsity loss when hierarchical features co-occur.

2. **Olshausen & Field (1996, 1997)** — "Emergence of simple-cell receptive field properties by learning a sparse code for natural images" (Nature) and "Sparse coding with an overcomplete basis set: A strategy employed by V1?" (Vision Research).
   - **Key principle**: The visual cortex employs overcomplete sparse coding where neurons compete via lateral inhibition to represent sensory input. Overcompleteness (more basis functions than input dimensions) enables greater flexibility and sparseness.
   - **Relevance to SAE absorption**: The Olshausen-Field model does not exhibit absorption because it uses continuous-valued coefficients with soft competition. However, the *hard* sparsity constraints in SAEs (TopK, L1) introduce winner-take-all dynamics that may disrupt the smooth competition mechanism.

3. **Rozell et al. (2008)** — "Sparse coding via thresholding and local competition in neural circuits" (Neural Computation).
   - **Key principle**: The Locally Competitive Algorithm (LCA) implements sparse coding through biologically plausible neural dynamics: leaky membrane potentials, thresholding nonlinearities, and lateral inhibition proportional to receptive field similarity (G_ij = <phi_i, phi_j>).
   - **Relevance to SAE absorption**: LCA's lateral inhibition is *state-dependent* — only active neurons inhibit others. This is structurally different from SAEs where sparsity is enforced globally (TopK) or via fixed penalties (L1). The LCA suggests that **local, activity-dependent competition** may prevent absorption better than global constraints.

4. **Friston (2010)** — "The free-energy principle: a unified brain theory?" (Nature Reviews Neuroscience).
   - **Key principle**: The brain minimizes variational free energy F = U - TS (internal energy minus entropy), which under predictive coding, leads to sparse efficient representations through "explaining away" — when one cause adequately explains sensory data, it suppresses alternative explanations.
   - **Relevance to SAE absorption**: "Explaining away" in predictive coding is the *neuroscience analog of absorption*. When a specific feature (e.g., "short") explains the input, it suppresses the general feature ("starts with S"). However, in the brain, this suppression is *context-dependent and reversible*; in SAEs, it becomes *structurally frozen* through training.

5. **Rosch (1978)** — Prototype theory and basic-level categorization.
   - **Key principle**: Human conceptual hierarchies have a cognitively privileged "basic level" (e.g., "dog") that is learned first and named most quickly, sitting between superordinate ("animal") and subordinate ("Labrador"). Superordinate categories have lower within-category similarity and are harder to form mental images for.
   - **Relevance to SAE absorption**: SAEs may be "missing" superordinate features because sparsity incentives favor the basic/subordinate level. The cognitive science literature suggests that superordinate concepts require *more* representational resources (not fewer), which is the opposite of what sparsity incentives produce.

6. **Bastos et al. (2012)** — "Canonical microcircuits for predictive coding" (Neuron).
   - **Key principle**: Predictive coding in cortex involves canonical microcircuits with superficial pyramidal cells (prediction-error units), deep pyramidal cells (expectation units), and inhibitory interneurons that mediate comparison between predictions and inputs.
   - **Relevance to SAE absorption**: The hierarchical structure of predictive coding — with distinct error and representation units — may provide architectural insights for preventing absorption. SAEs lack explicit error/representation separation.

7. **Li et al. (2025)** — "The Geometry of Concepts: Sparse Autoencoder Feature Structure" (Entropy 27(4):344). Examined SAE features at three spatial scales (atomic crystals, brain-like lobes, galaxy-scale power laws), revealing non-isotropic geometric structure and clustering entropy variations by layer.

8. **Mayzel & Schneidman (2024)** — "Homeostatic Synaptic Normalization Optimizes Learning in Network Models of Neural Population Codes" (eLife). Homeostatic synaptic scaling regulates firing rates while enabling competitive feature learning. Competition mechanisms between synapses are essential for computation.

9. **Turrigiano et al. (1998); Turrigiano & Nelson (2004)** — Homeostatic synaptic scaling.
   - **Key principle**: Neurons globally scale synaptic strengths via negative feedback to maintain firing rate set-points. When activity is chronically elevated, synapses are downscaled; when reduced, they are upscaled. Critically, scaling is **proportional** (multiplicative), preserving relative synaptic strengths and thus stored information.
   - **Relevance to SAE absorption**: SAEs lack any homeostatic mechanism. Features that are rarely activated (parent features, which only fire when no child is present) receive no protection and can be "forgotten" by the training dynamics. A homeostatic mechanism that maintains minimum activation rates for all features could prevent this.

#### Physics / Statistical Mechanics / Information Theory

10. **Tishby, Pereira & Bialek (1999)** — "The Information Bottleneck Method" (arXiv:physics/0004057).
    - **Key principle**: The IB method minimizes I(X;X~) while preserving I(X~;Y) about a "relevant variable" Y. Solutions exhibit hierarchical structure with curves bifurcating at critical beta values through second-order phase transitions.
    - **Relevance to SAE absorption**: The IB bifurcation phenomenon — where representations split into finer categories at critical compression levels — is structurally analogous to feature splitting/absorption in SAEs. The key difference: IB phase transitions are *reversible* (tuning beta changes the representation), while SAE absorption is *frozen* by training.

11. **Wu & Fischer (2020)** — "Phase Transitions for the Information Bottleneck in Representation Learning" (ICLR 2020).
    - **Key principle**: IB exhibits sudden jumps in dI(Y;Z)/dbeta when new classes are learned. Each transition finds a component of maximum nonlinear correlation between X and Y, orthogonal to the learned representation.
    - **Relevance to SAE absorption**: This formalizes how hierarchical representations emerge through phase transitions. The "absorption" of parent features may correspond to the system being "stuck" in a local minimum before the bifurcation point where the parent would split.

12. **Equitz & Cover (1991)** — "Successive Refinement of Information" (IEEE Trans. Info. Theory).
    - **Key principle**: A source is successively refinable if it can be encoded in layers (coarse to fine) such that each layer optimally refines the previous. Gaussian sources under MSE are successively refinable; not all sources are.
    - **Relevance to SAE absorption**: Matryoshka SAEs explicitly implement successive refinement (nested dictionaries of increasing size). The theory predicts that *not all feature hierarchies are successively refinable* — some parent-child relationships cannot be optimally encoded in a layered fashion. This may explain why Matryoshka SAEs reduce absorption for some hierarchies but not all.

13. **Peraza Coppola, Helias & Ringel (2025)** — "Renormalization group for deep neural networks: Universality of learning and scaling laws" (arXiv:2510.25553). Applies RG formalism to deep neural networks, establishing universality in learning and scaling laws. Builds on Mehta & Schwab (2014) variational RG-deep learning mapping.

14. **Ayonrinde, Pearce & Sharkey (2024)** — "Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs" (arXiv:2410.11179). Reframed SAE training through Minimum Description Length (information theory) rather than traditional sparsity penalties.

15. **Lobacheva et al. (2025)** — "SGD as Free Energy Minimization" (arXiv:2505.23489).
    - **Key principle**: SGD implicitly minimizes a free energy function F = U - TS, where the learning rate acts as temperature. At high temperature, entropy dominates (distributed representations); at low temperature, energy dominates (sparse representations).
    - **Relevance to SAE absorption**: The sparsity-reconstruction tradeoff in SAEs can be reframed as an energy-entropy tradeoff. The "temperature" (controlled by sparsity penalty strength) determines whether the system favors distributed (high-entropy) or sparse (low-entropy) representations. Absorption may be a low-temperature phenomenon where energy (reconstruction) dominates at the expense of representational diversity.

16. **Zdeborova & Krzakala (2016)** — "Statistical physics of inference: Thresholds and algorithms" (Physica A).
    - **Key principle**: Inference problems (compressed sensing, low-rank matrix estimation) exhibit phase transitions with sharp boundaries between recoverable and non-recoverable regimes.
    - **Relevance to SAE absorption**: Dictionary learning in SAEs is an inference problem. The "absorption" regime may correspond to a phase where the sparsity-reconstruction tradeoff makes parent feature recovery statistically impossible (below the phase transition threshold).

17. **Shannon (1959); Berger (1971)** — Rate-distortion theory. The fundamental trade-off between compression rate and reconstruction distortion. For hierarchical sources, the optimal lossy code exploits statistical dependencies between parent and child features, effectively "absorbing" the parent into the child.

18. **Raya & Ambrogioni (2023/2025)** — "The statistical thermodynamics of generative diffusion models: Phase transitions, symmetry breaking and critical instability" (Entropy). Shows generative models undergo second-order phase transitions in the mean-field universality class, with self-consistency equations identical to the Ising model.

19. **Li (2025)** — "A Spin Glass Characterization of Neural Networks" (arXiv:2508.07397).
    - **Key principle**: Constructs a Hopfield-type spin glass model from feedforward neural networks and uses replica symmetry breaking (RSB) as a characteristic descriptor. The overlap between simulated replica samples reveals nontrivial structural properties.
    - **Relevance to SAE absorption**: Spin glass theory provides tools for analyzing systems with many interacting components and quenched disorder. SAE dictionary learning can be viewed as a disordered system where latents compete for activation. RSB may characterize the transition from a symmetric state (all features equally represented) to a broken-symmetry state (some features dominate, others are absorbed).

20. **Annesi, Malatesta & Zamponi (2025)** — "Exact Full-RSB SAT/UNSAT Transition in Infinitely Wide Two-Layer Neural Networks" (SciPost Physics 18, 118). Establishes exact full replica symmetry breaking transitions in infinitely wide networks.

#### Biology / Evolution / Ecology

21. **Gause (1934)** — Competitive Exclusion Principle.
    - **Key principle**: Two species competing for identical limited resources cannot coexist indefinitely — one will outcompete and eliminate the other. This is a fundamental law of ecology.
    - **Relevance to SAE absorption**: SAE latents "compete" for activation budget (sparsity constraint). When a parent feature ("starts with S") and child feature ("short") always co-occur, the child "outcompetes" the parent because it provides more specific information per activation. This is a direct structural analogy: **sparsity constraint = limited resource; latents = competing species**.

22. **Fetzer et al. (2015)** — "The extent of functional redundancy changes as species' roles shift in different environments" (PNAS).
    - **Key principle**: Functional redundancy (species performing identical ecological functions) is context-dependent. Species that are redundant in one environment become pivotal in another.
    - **Relevance to SAE absorption**: Parent features (e.g., "starts with S") may appear "redundant" when child features are present, but they are *not* redundant in contexts where only the parent applies. The SAE's failure to maintain parent features is like an ecosystem losing species that appear redundant under normal conditions but are critical under perturbation.

23. **Loreau & de Mazancourt (2013)** — "Biodiversity and ecosystem stability" (Ecology Letters).
    - **Key principle**: Competition's net effect on ecosystem stability is often neutral or negative. What promotes stability is niche complementarity (reduced competition), not competition itself.
    - **Relevance to SAE absorption**: SAEs with strong sparsity constraints create intense "competition" between latents. The ecological literature suggests that *reducing competition strength* (e.g., through architectural designs that give parent features "protected niches") may improve representational stability.

24. **Ulanowicz (2018)** — "Biodiversity, functional redundancy, and system stability" (J. Royal Society Interface).
    - **Key principle**: Information theory tools (Shannon entropy, Kullback-Leibler divergence) capture the "overhead" in ecological flow networks — redundant pathways that provide system resilience.
    - **Relevance to SAE absorption**: The "overhead" of maintaining parent features alongside child features is the price of representational resilience. SAEs optimized purely for reconstruction+sparsity may be "over-pruning" this overhead, losing resilience.

25. **Clonal Selection Theory** (Burnet, 1959; modern formulations).
    - **Key principle**: The immune system maintains diversity through clonal selection: lymphocytes compete for antigen binding, with successful clones expanding and unsuccessful ones dying. Negative selection removes self-reactive clones.
    - **Relevance to SAE absorption**: The immune system has *multiple mechanisms* to maintain diversity: (1) negative selection (remove self-reactive = remove dead features), (2) somatic hypermutation (introduce variation = feature splitting), and (3) memory cell preservation (maintain clones for rare antigens = maintain parent features). SAEs have mechanisms (1) and (2) but lack (3) — there is no "memory" mechanism to preserve parent features that are rarely needed but critical.

26. **Hart & Ross (2003)** — "Exploiting the Analogy between the Immune System and Sparse Distributed Memories" (Genetic Programming and Evolvable Machines). Explicitly connects clonal selection theory to sparse distributed memory, showing immune network dynamics parallel competitive learning.

27. **Perelson & Oster (1979)** — "Theoretical studies of clonal selection" (J. Theor. Biol.). Mathematical basis for sparse, competitive recognition in immune systems.

28. **Gallinaro et al. (2025)** — "The interplay between homeostatic synaptic scaling and homeostatic structural plasticity maintains the robust firing rate of neural networks" (eLife).
    - **Key principle**: Homeostatic synaptic scaling (HSS) and structural plasticity (HSP) work redundantly to maintain firing rate homeostasis. HSS operates on hours-days timescale via proportional weight scaling; HSP operates on days-weeks via spine formation/elimination.
    - **Relevance to SAE absorption**: SAE training has no equivalent homeostatic mechanism. A feature's "firing rate" (activation frequency) is determined purely by reconstruction needs and sparsity constraints, with no feedback mechanism to prevent features from dying out. Adding homeostatic plasticity could maintain feature diversity.

#### Signal Processing / Compressed Sensing

29. **Donoho (2006); Candes & Tao (2006)** — Compressed sensing foundations.
    - **Key principle**: Sparse signals can be recovered from far fewer measurements than Nyquist requires, provided the sensing matrix satisfies the Restricted Isometry Property (RIP) or has low mutual coherence.
    - **Relevance to SAE absorption**: The SAE decoder matrix is a dictionary. The mutual coherence mu(D) = max_{i!=j} |<d_i, d_j>| governs whether sparse recovery is possible. High coherence (overlapping decoder directions) leads to ambiguity in feature attribution — precisely what absorption manifests as. The classical bound: OMP recovers K-sparse signals iff mu < 1/(2K-1).

30. **Elad & Aharon (2006); Sapiro et al.** — Incoherent dictionary learning.
    - **Key principle**: Dictionary learning algorithms can explicitly minimize coherence during training. Methods include INKSVD (incoherent K-SVD), IP+DR (iterative projection + dictionary refinement), and UNTF-INKSVD (unit norm tight frame).
    - **Relevance to SAE absorption**: OrtSAE's orthogonality penalty is a form of incoherence enforcement. However, full orthogonality may be too restrictive. The compressed sensing literature suggests that *bounded* incoherence (not full orthogonality) is sufficient for recovery guarantees, and may allow better reconstruction.

31. **Welch (1974)** — Lower bound on coherence.
    - **Key principle**: For a dictionary with n atoms in R^m, the mutual coherence satisfies mu >= sqrt((n-m)/(m(n-1))). Dictionaries achieving this bound are Equiangular Tight Frames (ETFs).
    - **Relevance to SAE absorption**: The Welch bound sets a fundamental limit on how incoherent a dictionary can be. For overcomplete dictionaries (n >> m), the bound approaches mu >= 1/sqrt(m). This means some decoder overlap is *inevitable* in overcomplete SAEs, and absorption may have a theoretical lower bound.

### Cross-Disciplinary Gaps

**Where transplants haven't been tried yet:**

1. **No SAE architecture explicitly implements lateral inhibition** (Rozell LCA-style) with state-dependent competition. All current SAEs use global sparsity constraints (L1, TopK, BatchTopK).

2. **No SAE work draws on the information bottleneck phase transition theory** to predict when absorption will occur or to design annealing schedules that avoid frozen absorption states.

3. **No SAE work uses ecological niche partitioning principles** to design "protected niches" for parent features that prevent competitive exclusion by child features.

4. **No SAE work implements a "memory" mechanism** (analogous to immune memory cells) to preserve parent features that are rarely activated but semantically important.

5. **No SAE work treats the sparsity-reconstruction tradeoff as a thermodynamic free energy problem** with temperature-controlled annealing to navigate the phase space.

6. **No SAE work connects "explaining away" in predictive coding** to feature absorption as a structural isomorphism — the neuroscience literature has identified this mechanism for decades, but it has not been transplanted to SAE architecture design.

7. **No rate-distortion theoretical bound** has been derived specifically for SAE absorption — despite the clear mapping (sparsity = rate, reconstruction = distortion, absorption = a specific failure mode of the code).

8. **No formal Lotka-Volterra model** has been applied to SAE feature dynamics, despite the obvious structural correspondence (features competing for activation "resources").

9. **No SAE work applies compressed sensing incoherence bounds** to derive theoretical limits on absorption. The mutual coherence bound mu < 1/(2K-1) has a direct interpretation for SAEs but has not been exploited.

10. **No SAE work implements homeostatic plasticity** to maintain feature activation rates, despite the clear neuroscience precedent.

11. **No SAE work uses spin glass RSB theory** to analyze the phase transition between symmetric (all features represented) and broken-symmetry (absorption-dominated) states.

---

## Phase 2: Initial Candidates

### Candidate A: Lateral Inhibition SAE (from Neuroscience / Rozell LCA)

- **Source principle**: The Locally Competitive Algorithm (Rozell et al., 2008) implements sparse coding through state-dependent lateral inhibition: only active neurons inhibit others, and inhibition strength is proportional to receptive field similarity (G_ij = <phi_i, phi_j>).
- **Structural correspondence**:
  - LCA neuron membrane potential u_i <-> SAE pre-activation (W_enc(x - b_dec) + b_enc)
  - LCA thresholding T_lambda(u_i) <-> SAE activation function (ReLU, TopK, JumpReLU)
  - LCA lateral inhibition G_ij = <phi_i, phi_j> <-> SAE decoder weight overlap (absorption occurs when child decoder d_child has high overlap with parent decoder d_parent)
  - LCA's *state-dependent* inhibition <-> Current SAEs use *state-independent* sparsity (fixed K or fixed lambda)
- **Hypothesis**: Replacing global sparsity constraints with local, activity-dependent lateral inhibition (where similar decoder directions inhibit each other proportionally to their overlap) will reduce absorption while maintaining reconstruction quality. The key prediction: inhibition strength should scale with decoder cosine similarity, so that a "short" feature inhibits "starts with S" only when both are active, not globally.
- **Why it's not just a metaphor**: The mathematical structure is identical — both systems solve sparse approximation with competition. The LCA's inhibition matrix G_ij = <phi_i, phi_j> is precisely the decoder Gram matrix that governs absorption in SAEs. The transplant preserves the *dynamics* (differential equations) not just the vocabulary.
- **Novelty estimate**: 8/10. LCA has been applied to sparse coding but never to SAEs for LLM interpretability. The connection between LCA lateral inhibition and SAE absorption is unexplored.

### Candidate B: Rate-Distortion Bounds for Absorption Inevitability (from Information Theory)

- **Source principle**: Rate-distortion theory characterizes the fundamental trade-off between compression rate (bits) and reconstruction distortion. For a given source and distortion measure, there exists a rate-distortion function R(D) that gives the minimum achievable rate for a given distortion. No code can operate below this bound.
- **Structural correspondence**:
  | Information Theory | SAE |
  |-------------------|-----|
  | Source X | LLM activation distribution |
  | Rate R | L0 sparsity (active features per token) |
  | Distortion D | Reconstruction MSE |
  | Codebook | SAE dictionary (decoder matrix) |
  | R(D) function | Optimal sparsity-reconstruction trade-off |
  | **Absorption** | **Optimal code exploiting parent-child redundancy** |

  The key insight: when the source has hierarchical structure (parent features that always co-occur with child features), the rate-distortion optimal code may "absorb" the parent into the child to save rate. This is not a bug but a fundamental consequence of lossy compression with sparsity constraints.

  The SAE training objective L = ||x - x_hat||^2 + lambda * ||z||_0 is exactly the Lagrangian relaxation of the rate-distortion problem: minimize D subject to R <= R_0.
- **Hypothesis**: Feature absorption is an inevitable consequence of operating near the rate-distortion bound for hierarchical sources. There exists a theoretical lower bound on absorption rate given the source's hierarchical structure, the sparsity level, and the distortion tolerance. Current SAEs that reduce absorption (Matryoshka, OrtSAE) do so by effectively changing the rate-distortion problem—either by adding structural constraints that move the operating point away from the bound, or by expanding the effective rate budget.
- **Why it's not just a metaphor**: The mapping is formal. SAE training minimizes L = MSE + lambda * L0, which is exactly the Lagrangian relaxation of the rate-distortion problem. The absorption phenomenon has a direct information-theoretic interpretation as the optimal code exploiting statistical dependencies in the source.
- **Novelty estimate**: 8/10. While rate-distortion theory has been applied to autoencoders generally, explicitly deriving absorption as a rate-distortion phenomenon for hierarchical sources is novel.

### Candidate C: Niche-Partitioning SAE (from Evolutionary Ecology / Competitive Exclusion)

- **Source principle**: The competitive exclusion principle (Gause) states that species competing for identical resources cannot coexist. Niche partitioning allows coexistence by dividing resources. In ecology, "functional redundancy" provides resilience — species that appear redundant in normal conditions become critical under perturbation.
- **Structural correspondence**:
  | Ecology | SAE |
  |---------|-----|
  | Species population N_i | Feature activation strength |
  | Carrying capacity K | Expansion factor (dictionary size) |
  | Competition coefficient alpha_ij | Decoder cosine similarity |
  | Niche overlap | Feature correlation in training data |
  | Resource | Reconstruction error budget |
  | Competitive exclusion | Feature absorption / feature death |

  The Lotka-Volterra equations: dN_i/dt = r_i N_i (1 - (N_i + sum_j alpha_ij N_j)/K)

  Map to SAE: feature i's "population" grows when it reduces reconstruction error, but is suppressed by competing features j with high decoder overlap. Absorption occurs when a child feature drives a parent to near-extinction.
- **Hypothesis**: SAE feature absorption can be modeled as competitive exclusion. The absorption rate should correlate with niche overlap (decoder cosine similarity x data co-occurrence) between parent and child features. Introducing "niche differentiation" mechanisms should reduce absorption analogously to how resource partitioning reduces competitive exclusion in ecology.
- **Why it's not just a metaphor**: The mathematical structure is identical. Both systems involve entities competing for a limited resource with competitive coefficients determined by overlap. The equilibrium analysis of LV systems directly applies: coexistence requires alpha_ij * alpha_ji < 1, which maps to a concrete condition on decoder orthogonality and data correlation.
- **Novelty estimate**: 7/10. The LV-competition analogy for neural networks exists in general (e.g., NeuRD), but applying it specifically to SAE feature absorption dynamics with quantitative predictions is new.

### Candidate D: Homeostatic Plasticity SAE (from Neuroscience / Synaptic Scaling)

- **Source principle**: Homeostatic synaptic scaling (Turrigiano et al., 1998) maintains neuronal firing rates via negative feedback: when activity is chronically elevated, synapses are globally downscaled; when reduced, they are upscaled. The scaling is proportional (multiplicative), preserving relative weights and stored information.
- **Structural correspondence**:
  | Neuroscience | SAE |
  |-------------|-----|
  | Neuron firing rate | Feature activation frequency |
  | Synaptic strength | Decoder weight norm / encoder sensitivity |
  | Set-point firing rate | Target activation rate per feature |
  | Homeostatic scaling | Feature-wise normalization during training |
  | Hebbian plasticity | Reconstruction gradient updates |
  | Catastrophic excitation/quiescence | Feature death / runaway activation |

  The key insight: SAEs have "Hebbian" learning (gradient descent on reconstruction) but no "homeostatic" regulation. Parent features, which fire rarely (only when no child is present), are not protected and can be pruned by the training dynamics.
- **Hypothesis**: Adding a homeostatic plasticity term that maintains minimum activation rates for all features will prevent parent feature absorption. The mechanism: after each training batch, compute each feature's empirical activation rate. If a feature's rate falls below a threshold, boost its encoder sensitivity (equivalent to upscaling synapses). If it exceeds a threshold, reduce sensitivity. This preserves relative feature relationships while preventing extinction.
- **Why it's not just a metaphor**: The mathematical structure maps directly. Both systems have fast, input-driven plasticity (Hebbian/gradient) and slow, rate-stabilizing plasticity (homeostatic scaling). The timescale separation and proportional nature of the homeostatic term are preserved.
- **Novelty estimate**: 7/10. Homeostatic plasticity has been applied to neural networks (e.g., for catastrophic forgetting), but not specifically to SAE feature absorption. The connection to parent feature preservation is new.

### Candidate E: Spin Glass RSB Analysis of Absorption (from Statistical Physics)

- **Source principle**: Spin glass theory (Parisi, Mezard, Virasoro) analyzes disordered systems with many interacting components. Replica symmetry breaking (RSB) characterizes transitions from symmetric states (all spin configurations equally likely) to broken-symmetry states (some configurations dominate). The order parameter is the overlap distribution P(q).
- **Structural correspondence**:
  | Spin Glass | SAE |
  |-----------|-----|
  | Spin configuration | Activation pattern z |
  | Disorder (quenched couplings J_ij) | Decoder overlap matrix G = D^T D |
  | Temperature T | Inverse sparsity penalty strength 1/lambda |
  | Replica symmetry | All features equally represented |
  | Replica symmetry breaking | Some features dominate, others absorbed |
  | Overlap distribution P(q) | Feature activation correlation distribution |
  | Parisi order parameter | Hierarchical structure of feature representations |

  The key insight: SAE dictionary learning can be viewed as finding the ground state of a disordered Hamiltonian H(z) = ||x - D*z||^2 + lambda*||z||_0. At low temperature (high sparsity), the system may undergo RSB, entering a phase where some features (children) dominate and others (parents) are frozen out.
- **Hypothesis**: Feature absorption corresponds to a replica symmetry breaking phase transition in the SAE energy landscape. There exists a critical sparsity level lambda_c below which RSB occurs and absorption becomes prevalent. This transition can be characterized by the overlap distribution P(q) between independently trained SAEs: in the RS phase, P(q) is a single delta function; in the RSB phase, P(q) develops multiple peaks.
- **Why it's not just a metaphor**: The mapping is formal. Both systems involve energy minimization in high-dimensional spaces with quenched disorder. The replica method provides exact solutions for the thermodynamic limit, which maps to the limit of large dictionary size and many training samples.
- **Novelty estimate**: 9/10. Spin glass theory has been applied to neural network learning (Gardner, 1988; recent work by Li 2025), but never to SAE feature absorption specifically. The RSB-absorption correspondence is genuinely unexplored.

---

## Phase 3: Self-Critique

### Against Candidate A (Lateral Inhibition SAE)

- **Shallow analogy attack**: Is this just renaming "sparsity constraint" as "lateral inhibition"? No — the structural correspondence is exact. The LCA inhibition matrix G_ij = <phi_i, phi_j> is precisely the decoder Gram matrix. However, a domain expert might note that biological lateral inhibition is *recurrent* and *dynamic*, while SAEs are feedforward. The transplant would need to preserve the dynamical aspect (iterative settling) to be truly faithful.
- **Scale mismatch attack**: The LCA was designed for small overcomplete dictionaries (hundreds of basis functions). SAEs have dictionaries with millions of latents. Computing the full Gram matrix G_ij for millions of latents is O(n^2) and infeasible. However, the Gram matrix is sparse — most latents have near-zero overlap. Approximate methods (locality-sensitive hashing, top-k nearest neighbors) could make this scalable.
- **Prior transplant check**: Has LCA been applied to neural network interpretability? LCA has been used for sparse coding of natural images and video, and for neuromorphic hardware, but not for LLM activation decomposition. No prior work connects LCA to SAE absorption.
- **Testability attack**: Can we distinguish "this works because of lateral inhibition" from "this works because we added another regularization term"? Yes — the diagnostic experiment is to compare (1) global TopK sparsity, (2) LCA-style lateral inhibition with full Gram matrix, and (3) LCA with *random* inhibition matrix. If (2) outperforms (3), the structured inhibition (not just iterative dynamics) is the active ingredient.
- **Verdict**: STRONG. The structural correspondence is rigorous, the novelty is high, and the diagnostic experiment is clear. The main risk is computational scalability.

### Against Candidate B (Rate-Distortion / Information Theory)

- **Shallow analogy attack**: Rate-distortion theory assumes known source statistics and infinite block length. SAEs learn from finite samples with unknown distributions. The bound may be too loose to make precise predictions. However, recent work (Chen et al., 2025) provides theoretical feature recovery guarantees for SAEs under specific generative models, suggesting the theoretical approach is viable.
- **Scale mismatch attack**: Rate-distortion theory typically deals with scalar or vector sources. LLM activations are high-dimensional (e.g., 768-dim for GPT-2 small). The multi-terminal rate-distortion problem is much harder. However, the single-letter characterization still applies, and the hierarchical structure creates a specific dependency structure that can be analyzed.
- **Prior transplant check**: Rate-distortion theory has been applied to neural network compression (Gao et al., 2019), VAEs (Alemi et al., 2018), and sparse graph codes (Sourlas, 1989). The beta-VAE explicitly implements the RD Lagrangian. However, no prior work derives absorption as an RD phenomenon for hierarchical sources in SAEs.
- **Testability attack**: The key prediction—that absorption is inevitable near the RD bound—can be tested by training SAEs at different points on the sparsity-reconstruction trade-off curve and measuring absorption. If absorption increases as we approach the Pareto frontier, this supports the hypothesis. A diagnostic experiment: construct a toy model with known hierarchical features and compare the empirical absorption rate to the theoretical lower bound derived from the RD function.
- **Verdict**: STRONG. The mapping is the most formal of the candidates. The main risk is computational: deriving the RD function for realistic hierarchical sources may be intractable. But even a toy-model derivation would be valuable.

### Against Candidate C (Niche-Partitioning / Ecology)

- **Shallow analogy attack**: Is "niche partitioning" just "hierarchical SAEs" with ecological branding? There is overlap with HSAE (Luo et al., 2026) and Matryoshka SAEs. However, the ecological framing adds a specific mechanism — "protected activation territory" — that is not present in existing hierarchical SAEs.
- **Scale mismatch attack**: Ecological niche partitioning operates at the population level (many individuals per species). SAE latents are single vectors. The "population" analogy doesn't directly apply. However, the competition kernel (Lotka-Volterra) maps directly to the decoder overlap matrix, so the mathematical structure survives the scale translation.
- **Prior transplant check**: HSAE (Luo et al., 2026) is the closest prior work — it explicitly models parent-child relationships with structural constraints. The niche-partitioning idea would need to differentiate from HSAE. The key difference: HSAE uses a fixed tree structure; niche-partitioning uses a dynamic competition mechanism that doesn't require pre-specifying the hierarchy.
- **Testability attack**: Can we distinguish "niche partitioning works" from "any architecture with explicit hierarchy works"? Yes — compare (1) standard SAE, (2) HSAE with fixed tree, (3) niche-partitioning SAE with dynamic competition. If (3) outperforms (2) on hierarchies not in the pre-specified tree, the dynamic mechanism is validated.
- **Verdict**: MODERATE. The idea is novel but overlaps with existing hierarchical SAE work. The "protected activation territory" mechanism is the key differentiator but requires careful experimental design to validate.

### Against Candidate D (Homeostatic Plasticity SAE)

- **Shallow analogy attack**: Is this just "feature dropout prevention" with neuroscience branding? There is overlap with standard regularization techniques (e.g., ensuring all neurons fire). However, the homeostatic mechanism is specifically *proportional* scaling that preserves relative relationships, not just a hard constraint on minimum activation.
- **Scale mismatch attack**: Biological synaptic scaling operates on individual neurons with continuous firing rates. SAE features are discrete (binary or sparse continuous). The timescale separation (fast Hebbian vs. slow homeostatic) also doesn't directly map to standard SGD training. However, mini-batch statistics can serve as an approximation.
- **Prior transplant check**: Homeostatic plasticity has been used in artificial neural networks (e.g., Tetzlaff et al., 2011; recent work on catastrophic forgetting). However, no prior work applies it to SAEs specifically for absorption prevention.
- **Testability attack**: Can we distinguish "homeostatic plasticity prevents absorption" from "any mechanism that prevents feature death prevents absorption"? Yes — compare (1) standard SAE, (2) SAE with hard minimum-activation constraint, (3) SAE with proportional homeostatic scaling. If (3) outperforms (2), the proportional nature (preserving relative relationships) is the active ingredient.
- **Verdict**: MODERATE. The mechanism is biologically grounded and plausible, but the experimental differentiation from simpler alternatives is subtle. The main value is in the proportional scaling preserving feature relationships.

### Against Candidate E (Spin Glass RSB Analysis)

- **Shallow analogy attack**: Is RSB just a fancy way to describe "some features dominate, others don't"? The RSB framework provides quantitative predictions about the nature of the transition, the critical point, and the structure of the solution space. However, deriving these predictions for realistic SAEs requires approximations that may lose the rigor.
- **Scale mismatch attack**: Spin glass theory is exact in the thermodynamic limit (infinite system size). SAEs have finite dictionary sizes (thousands to millions). Finite-size corrections may wash out the sharp phase transitions predicted by the theory.
- **Prior transplant check**: Spin glass theory has been extensively applied to neural networks (Gardner, 1988; recent work by Li 2025; Annesi et al. 2025). However, no prior work connects RSB to SAE feature absorption specifically.
- **Testability attack**: The key prediction is a phase transition at a critical sparsity level. This can be tested by training SAEs with varying sparsity and measuring absorption. The diagnostic signature is the overlap distribution P(q) between independently trained SAEs: RS phase = single delta peak; RSB phase = multiple peaks. This is a clear, quantitative prediction.
- **Verdict**: STRONG. The framework makes precise, testable predictions about absorption as a phase transition. The main risk is that finite-size effects may blur the transition, but the qualitative prediction (absorption increases with sparsity) is robust.

---

## Phase 4: Refinement

### Dropped
- **Candidate C (Niche-Partitioning)** receives a MODERATE verdict and is deprioritized. While the Lotka-Volterra structural correspondence is exact, the experimental differentiation from existing hierarchical SAEs (HSAE, Matryoshka) is challenging. The core insight — that competition dynamics drive absorption — is better captured by Candidate A's lateral inhibition mechanism and Candidate E's spin glass analysis.

- **Candidate D (Homeostatic Plasticity)** receives a MODERATE verdict. The mechanism is plausible but the experimental differentiation from simpler alternatives (hard minimum-activation constraints) is subtle. It may be valuable as a component within another architecture rather than a standalone proposal.

### Strengthened

**Candidate A (Lateral Inhibition SAE)** is selected as the primary front-runner because:
1. It has the strongest structural correspondence (exact mathematical isomorphism between LCA inhibition matrix and SAE decoder Gram matrix).
2. It is the most testable (clear diagnostic experiment with random inhibition control).
3. It has the highest novelty (LCA has never been applied to LLM SAEs).
4. It is computationally feasible with approximations (sparse Gram matrix, top-k neighbors).
5. It directly addresses the mechanism of absorption (competition between overlapping features).

**Formalized structural correspondence**:

The LCA dynamics are:
```
tau * du_i/dt = -u_i + <phi_i, x> - sum_{j!=i} G_ij * a_j
a_i = T_lambda(u_i)
```

where G_ij = <phi_i, phi_j> is the Gram matrix.

The SAE forward pass is:
```
z = T_lambda(W_enc * (x - b_dec) + b_enc)
x_hat = W_dec * z + b_dec
```

Absorption occurs when a child latent z_child activates and suppresses a parent latent z_parent. In the LCA, this suppression is *proportional to decoder overlap* and *only occurs when both are active*. In SAEs, suppression is *global* (via TopK or L1) and *independent of overlap*.

The transplanted architecture (LCA-SAE) would be:
```python
# Standard SAE encoder
u = W_enc @ (x - b_dec) + b_enc

# LCA-style lateral inhibition (iterative settling)
for t in range(T_steps):
    z = soft_threshold(u, lambda_sparsity)
    # Lateral inhibition: active latents suppress similar competitors
    u = u - eta * (G @ z)

x_hat = W_dec @ z + b_dec
```

**Candidate B (Rate-Distortion)** is selected as the secondary front-runner because:
1. It provides the most formal theoretical framework.
2. It makes a counter-intuitive prediction: absorption is not a bug but a fundamental consequence of optimal compression.
3. It explains why ALL existing methods reduce absorption through the same mechanism.
4. It provides a unifying framework that subsumes existing empirical observations.

**Formalized structural correspondence**:

For a hierarchical source with parent f_p and child f_c where P(f_c | f_p) ≈ 1, the RD-optimal code encodes only the child. The parent is "absorbed" because encoding it separately would increase rate without proportionally reducing distortion.

The diagnostic experiment: construct a toy model with known hierarchical features, derive the RD bound, and compare empirical absorption to the theoretical lower bound.

**Candidate E (Spin Glass RSB)** is selected as the tertiary front-runner because:
1. It provides a fundamentally different perspective on absorption as a phase transition.
2. It makes quantitative predictions about critical sparsity levels.
3. It connects to a rich theoretical framework with established analytical tools.
4. It suggests novel diagnostic methods (overlap distribution analysis).

### Selected Front-Runners

**Primary: Candidate A — Lateral Inhibition Prevents Feature Absorption**
**Secondary: Candidate B — Rate-Distortion Theory of Absorption Inevitability**
**Tertiary: Candidate E — Spin Glass RSB Analysis of Absorption Phase Transition**

Rationale for triple selection:
- Candidate A is the better *engineering* proposal: it yields a concrete new architecture with clear implementation path.
- Candidate B is the better *theory* proposal: it provides a unifying explanation and makes fundamental predictions.
- Candidate E is the better *analytical* proposal: it provides a novel framework for understanding absorption as a phase transition.
- Together, they form a complete research story: theory explains the problem (B), phase analysis characterizes it (E), architecture solves it (A).

---

## Phase 5: Final Proposal

### Title
**Lateral Inhibition Prevents Feature Absorption: A Neural-Circuit-Inspired Sparse Autoencoder Architecture**

### Source Principle
The Locally Competitive Algorithm (LCA; Rozell et al., 2008) implements sparse coding in biologically plausible neural circuits through three mechanisms: (1) leaky membrane potentials, (2) thresholding nonlinearities, and (3) **state-dependent lateral inhibition** where active neurons suppress competitors proportional to receptive field similarity (G_ij = <phi_i, phi_j>). This creates a dynamic competition where suppression is *local* (only active neurons inhibit), *structured* (inhibition scales with feature overlap), and *context-dependent* (suppression varies with input).

### Structural Correspondence

| Source (Neuroscience LCA) | Target (SAE Absorption) |
|---------------------------|------------------------|
| Neuron membrane potential u_i | SAE pre-activation z_pre |
| Thresholding T_lambda(u_i) | SAE activation (ReLU/TopK) |
| Receptive field phi_i | SAE decoder direction d_i |
| Lateral inhibition G_ij = <phi_i, phi_j> | SAE decoder Gram matrix G = D^T D |
| State-dependent competition | Dynamic sparsity (vs. fixed TopK) |
| "Explaining away" in predictive coding | Feature absorption |

The key insight: **feature absorption in SAEs is the computational equivalent of "explaining away" in predictive coding, but frozen into network weights by global sparsity constraints.** The LCA's dynamic, overlap-proportional inhibition prevents explaining-away from becoming permanent absorption.

### Hypothesis
An SAE with LCA-style lateral inhibition (where active latents suppress competitors proportional to decoder cosine similarity) will exhibit lower feature absorption rates than standard TopK/BatchTopK SAEs with equivalent final sparsity and reconstruction quality. The reduction in absorption will be specifically attributable to the *structured* inhibition (decoder-overlap-proportional), not merely to iterative dynamics or additional regularization.

### Method

#### Architecture: LCA-SAE

The LCA-SAE modifies the standard SAE forward pass to include iterative lateral inhibition:

```python
def lca_sae_encode(x, W_enc, W_dec, b_enc, b_dec, lambda_sparsity, T_steps):
    # Initial pre-activation
    u = W_enc @ (x - b_dec) + b_enc

    # Compute decoder overlap matrix (sparse approximation)
    # In practice: precompute top-k nearest neighbors per latent
    G = compute_sparse_gram_matrix(W_dec, k=50)  # O(n*k) not O(n^2)

    # Iterative settling with lateral inhibition
    for t in range(T_steps):
        z = soft_threshold(u, lambda_sparsity)
        # Lateral inhibition: active latents suppress similar competitors
        u = u - eta * (G @ z)

    return z
```

Key design choices:
- **Sparse Gram approximation**: Instead of full G (n x n), store only top-k nearest neighbors per latent. For n=1M latents, k=50, this is 50M entries vs. 1T for full matrix.
- **Soft thresholding**: T_lambda(u) = sign(u) * max(0, |u| - lambda) for L1-like sparsity, or hard thresholding for L0-like sparsity.
- **Fixed T_steps**: 5-10 iterations is typically sufficient for convergence (Rozell et al., 2008).

**Training**: Standard reconstruction loss + sparsity penalty, with the LCA dynamics as the encoder. The decoder and encoder are trained jointly via backpropagation through the iterative dynamics (or with straight-through estimator for thresholding).

### Diagnostic Experiment

**Experiment 1: Structured vs. Random Inhibition (Active Ingredient Test)**
- Train three SAEs on identical data (Pythia-160M activations, layer 8):
  - (A) Standard TopK SAE (baseline)
  - (B) LCA-SAE with true decoder Gram matrix (structured inhibition)
  - (C) LCA-SAE with random Gram matrix (unstructured inhibition control)
- Measure: SAEBench absorption scores (first-letter) + semantic-hierarchy absorption (WordNet)
- Prediction: absorption(B) < absorption(A) and absorption(B) < absorption(C)
- If absorption(B) ≈ absorption(C): the inhibition structure (not just iterative dynamics) is irrelevant.
- If absorption(B) < absorption(A) but absorption(C) < absorption(A): iterative dynamics help, but structured inhibition doesn't add value.

**Experiment 2: Inhibition Strength Scaling**
- Vary the inhibition strength parameter eta across {0.0, 0.1, 0.5, 1.0, 2.0}
- Measure absorption vs. reconstruction tradeoff
- Prediction: There exists an optimal eta where absorption is minimized without significant reconstruction loss. Too much inhibition causes under-representation; too little fails to prevent absorption.

**Experiment 3: Context-Dependent Suppression**
- Test whether LCA-SAE correctly handles cases where parent features should *not* be suppressed (e.g., "snake" should activate "starts with S" but not "short").
- Measure: parent feature activation on inputs where no child feature is present
- Prediction: LCA-SAE maintains parent activation when children are absent; standard TopK SAE may suppress parents globally.

### Experimental Plan

**Models**: Pythia-160M (fast iteration) + Gemma-2-2B (standard benchmark)
**Layer**: Residual post-layer 8 (standard SAEBench layer)
**SAE width**: 16k latents (manageable scale)
**Training**: SAELens training pipeline with LCA encoder modification
**Evaluation**:
- SAEBench absorption metric (first-letter, ~26 min per SAE)
- Custom semantic-hierarchy absorption (WordNet parent-child pairs, ~10 min per SAE)
- L0, CE loss recovered, explained variance (standard metrics)

**Runtime estimate**:
- Training 3 SAEs x 30 min each = 90 min (can parallelize)
- Evaluation 3 SAEs x 40 min each = 120 min
- Total: ~3.5 GPU-hours, split across multiple 1-hour tasks

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Sparse Gram approximation loses important long-range overlaps | Medium | Ablation with full Gram on small SAE (1k latents) |
| Iterative dynamics make training unstable | Medium | Use straight-through estimator; start with small T_steps |
| LCA-SAE has worse reconstruction than TopK | Medium | Tune eta and lambda jointly; accept small reconstruction tradeoff |
| Absorption reduction is trivial (all architectures reduce it) | Low | Random inhibition control isolates structured inhibition effect |
| Computational cost too high for large SAEs | Low | Top-k neighbor approximation keeps cost O(n*k) |

### Novelty Claim

The specific cross-disciplinary insight is: **feature absorption in SAEs is structurally isomorphic to "explaining away" in predictive coding, and can be prevented by the same mechanism that prevents permanent explaining-away in biological neural circuits — state-dependent lateral inhibition proportional to feature overlap.**

Evidence this hasn't been applied before:
- LCA (Rozell et al., 2008) has 800+ citations but zero applications to LLM sparse autoencoders.
- No SAE architecture (ReLU, TopK, JumpReLU, Gated, BatchTopK, Matryoshka, OrtSAE, HSAE) uses dynamic, overlap-proportional lateral inhibition.
- The connection between LCA inhibition and SAE decoder Gram matrix has not been made in the literature.
- "Explaining away" is discussed in predictive coding and Bayesian networks but has not been connected to SAE feature absorption.

---

## Secondary Proposal: Rate-Distortion Theory of Absorption

### Title
**"Absorption as Compression: A Rate-Distortion Theory of Feature Absorption in Sparse Autoencoders"**

### Source Principle
Rate-distortion theory (Shannon, 1959; Berger, 1971) characterizes the fundamental limits of lossy data compression. For hierarchical sources where parent features imply child features (P(child|parent) ≈ 1), the optimal lossy code exploits statistical dependencies by encoding only the child and reconstructing the parent implicitly.

### Structural Correspondence

| Rate-Distortion Theory | Sparse Autoencoder |
|----------------------|-------------------|
| Source X | LLM activation distribution |
| Reconstruction X_hat | SAE reconstruction W_d * z |
| Distortion d(x, x_hat) | Squared error ||x - x_hat||^2 |
| Rate R | L0 sparsity (active features per token) |
| Optimal code p(x_hat|x) | SAE encoder/decoder |
| Hierarchical source structure | Parent-child feature co-occurrence |
| **Absorption** | **Optimal code exploiting parent-child redundancy** |

### Hypothesis
**H1 (Inevitability)**: For hierarchical sources, feature absorption is inevitable at the rate-distortion bound. There exists a theoretical lower bound on absorption rate given the source's hierarchical structure, sparsity level, and distortion tolerance.

**H2 (Operating Point)**: Current SAE architectures that reduce absorption (Matryoshka, OrtSAE, HSAE) do so by effectively operating away from the RD bound — either by adding constraints that increase rate for the same distortion, or by changing the distortion measure to penalize absorption explicitly.

**H3 (Predictive Power)**: The absorption rate of an SAE can be predicted from the source's hierarchical redundancy and the operating point on the sparsity-reconstruction trade-off curve.

### Diagnostic Experiment

**"Approaching the Bound" Experiment:**
1. Train a family of SAEs on the same data with varying sparsity levels
2. For each SAE, compute: (a) reconstruction error, (b) L0 sparsity, (c) absorption rate
3. Plot the sparsity-reconstruction Pareto frontier
4. **Prediction**: Absorption rate should increase monotonically as SAEs approach the Pareto frontier, regardless of architecture
5. **Control**: Train SAEs with a modified distortion measure that explicitly penalizes absorption. These should show lower absorption at the same sparsity-reconstruction point.

This experiment distinguishes "absorption works because of the borrowed principle" (RD theory predicts it as optimal compression) from "absorption works for a mundane reason" (e.g., just a training artifact).

### Why This Complements the Primary Proposal

- **Candidate A (LCA-SAE)** provides a concrete *engineering solution*: how to build an SAE that prevents absorption.
- **Candidate B (Rate-Distortion)** provides a *theoretical foundation*: why absorption happens and what the fundamental limits are.
- Together, they form a complete research story: theory explains the problem, architecture solves it.

---

## Tertiary Proposal: Spin Glass RSB Analysis

### Title
**"Replica Symmetry Breaking and Feature Absorption: A Statistical Physics Perspective on Sparse Autoencoder Phase Transitions"**

### Source Principle
Spin glass theory (Parisi, Mezard, Virasoro, 1987) analyzes disordered systems with many interacting components. Replica symmetry breaking (RSB) characterizes transitions from symmetric states to broken-symmetry states where some configurations dominate. The order parameter is the overlap distribution P(q).

### Structural Correspondence

| Spin Glass | SAE |
|-----------|-----|
| Spin configuration | Activation pattern z |
| Disorder (couplings J_ij) | Decoder overlap matrix G = D^T D |
| Temperature T | Inverse sparsity penalty 1/lambda |
| Replica symmetry | All features equally represented |
| Replica symmetry breaking | Some features dominate, others absorbed |
| Overlap distribution P(q) | Feature activation correlation distribution |

### Hypothesis
Feature absorption corresponds to a replica symmetry breaking phase transition. There exists a critical sparsity level lambda_c below which RSB occurs. This can be diagnosed by the overlap distribution between independently trained SAEs: RS phase shows a single delta peak; RSB phase shows multiple peaks.

### Diagnostic Experiment
Train multiple SAEs with varying sparsity levels. For each level, train 5 independent SAEs and compute the pairwise overlap distribution of their feature activations on a held-out test set. Plot the overlap distribution as a function of sparsity. Prediction: a sharp transition from unimodal to multimodal P(q) at lambda_c, coinciding with the onset of high absorption rates.

---

## Sources

### Neuroscience / Cognitive Science
- Olshausen, B.A. & Field, D.J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607-609.
- Olshausen, B.A. & Field, D.J. (1997). Sparse coding with an overcomplete basis set: A strategy employed by V1? *Vision Research*, 37(23), 3311-3325.
- Rozell, C.J., Johnson, D.H., Baraniuk, R.G., & Olshausen, B.A. (2008). Sparse coding via thresholding and local competition in neural circuits. *Neural Computation*, 20(10), 2526-2563.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127-138.
- Bastos, A.M. et al. (2012). Canonical microcircuits for predictive coding. *Neuron*, 76(4), 695-711.
- Rosch, E. (1978). Principles of categorization. In *Cognition and Categorization*.
- Turrigiano, G.G. et al. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391, 892-896.
- Turrigiano, G.G. & Nelson, S.B. (2004). Homeostatic plasticity in the developing nervous system. *Nature Reviews Neuroscience*, 5(2), 97-107.
- Gallinaro, J.V. et al. (2025). The interplay between homeostatic synaptic scaling and homeostatic structural plasticity maintains the robust firing rate of neural networks. *eLife*.

### Physics / Information Theory
- Tishby, N., Pereira, F.C., & Bialek, W. (1999). The Information Bottleneck Method. *arXiv:physics/0004057*.
- Wu, T. & Fischer, I. (2020). Phase Transitions for the Information Bottleneck in Representation Learning. *ICLR 2020*.
- Equitz, W.H. & Cover, T.M. (1991). Successive Refinement of Information. *IEEE Trans. Info. Theory*, 37(3), 269-275.
- Zdeborova, L. & Krzakala, F. (2016). Statistical physics of inference: Thresholds and algorithms. *Physica A*.
- Lobacheva, E. et al. (2025). SGD as Free Energy Minimization. *arXiv:2505.23489*.
- Shannon, C.E. (1959). Coding theorems for a discrete source with a fidelity criterion. *IRE Nat. Conv. Rec.*
- Berger, T. (1971). *Rate Distortion Theory: A Mathematical Basis for Data Compression*.
- Li, J. (2025). A Spin Glass Characterization of Neural Networks. *arXiv:2508.07397*.
- Annesi, B.L., Malatesta, E.M., & Zamponi, F. (2025). Exact Full-RSB SAT/UNSAT Transition in Infinitely Wide Two-Layer Neural Networks. *SciPost Physics*, 18, 118.
- Mezard, M., Parisi, G., & Virasoro, M.A. (1987). *Spin Glass Theory and Beyond*.

### Biology / Ecology
- Gause, G.F. (1934). *The Struggle for Existence*.
- Fetzer, I. et al. (2015). The extent of functional redundancy changes as species' roles shift in different environments. *PNAS*, 112(17), 5442-5447.
- Loreau, M. & de Mazancourt, C. (2013). Biodiversity and ecosystem stability: a synthesis of underlying mechanisms. *Ecology Letters*, 16, 106-115.
- Ulanowicz, R.E. (2018). Biodiversity, functional redundancy, and system stability. *J. Royal Society Interface*.
- Burnet, F.M. (1959). *The Clonal Selection Theory of Acquired Immunity*.
- Hart, E. & Ross, P. (2003). Exploiting the Analogy between the Immune System and Sparse Distributed Memories. *Genetic Programming and Evolvable Machines*.

### Signal Processing / Compressed Sensing
- Donoho, D.L. (2006). Compressed sensing. *IEEE Trans. Info. Theory*, 52(4), 1289-1306.
- Candes, E.J. & Tao, T. (2006). Near-optimal signal recovery from random projections. *IEEE Trans. Info. Theory*, 52(12), 5406-5425.
- Elad, M. & Aharon, M. (2006). Image denoising via sparse and redundant representations. *IEEE Trans. Image Processing*.
- Welch, L.R. (1974). Lower bounds on the maximum cross correlation of signals. *IEEE Trans. Info. Theory*, 20(3), 397-399.

### SAE Literature
- Chanin et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *arXiv:2409.14507*.
- Bussmann et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. *arXiv:2503.17547*.
- Luo et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. *arXiv:2602.11881*.
- Korznikov et al. (2025). OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. *arXiv:2509.22033*.

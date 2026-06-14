# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Kersten, Mamassian & Yuille (2004), "A Bayesian Perspective on Object Perception"** — Key mechanism: **"Explaining away"** in Bayesian inference. When two causes could explain an observation, evidence for one cause reduces belief in the other. Neural implementation via lateral inhibition produces sparse codes where only the best-fitting features remain active. Direct relevance: SAE feature absorption may be a *failure* of explaining away — child features fail to suppress parent features properly, or vice versa.

2. **Rozell et al. (2008), "Sparse coding via thresholding and local competition in neural circuits" (Neural Computation)** — Key mechanism: **Locally Competitive Algorithms (LCA)**. Neurons compete via lateral inhibition to represent inputs. The dynamics follow: `du_m/dt = (1/τ)[b_m - u_m - Σ G_{m,n} a_n]`. Strong nodes prevent weaker nodes from activating — implementing explaining away. Direct relevance: SAEs lack explicit lateral inhibition; feature absorption may stem from this architectural absence.

3. **Pecevski et al. (2015), "Causal Inference and Explaining Away in a Spiking Network" (Nature Scientific Reports)** — Key mechanism: **Spike-based recurrent inhibition** implements quadratic programming with non-negativity constraints. Activity of other neurons "explains away" the stimulus for neuron i if reconstruction error without i is zero. Direct relevance: Formal proof that explaining away produces sparse representations; SAEs approximate this without explicit recurrence.

4. **Spratling (2014), "A neural implementation of the Hough transform and the advantages of explaining away"** — Key mechanism: **Feature competition** in neural networks. Neurons representing hypotheses compete to explain image elements; strongly supported neurons inhibit others from receiving activation from the same evidence. Direct relevance: The "like-suppresses-like" motif in V1 competition mirrors how SAE latents should compete but don't.

5. **Rao & Ballard (1999), "Predictive coding in the visual cortex" (Nature Neuroscience)** — Key mechanism: **Hierarchical predictive coding**. Top-down predictions meet bottom-up prediction errors; sparse prior p(r) favors kurtotic activations. The brain transmits only prediction errors, achieving efficient (sparse) coding. Direct relevance: SAEs compress activations but lack the hierarchical prediction structure; absorption may reflect compression without predictive structure.

6. **Friston (2010), "The free-energy principle: a unified brain theory?" (Nature Reviews Neuroscience)** — Key mechanism: **Variational free energy minimization**. Living systems minimize surprise via Bayesian inference; precision-weighting (analogous to attention) induces sparsity in prediction error processing. Direct relevance: The free energy principle predicts that hierarchical generative models should naturally form sparse representations; SAEs are discriminative, not generative, which may explain absorption pathologies.

7. **Itti & Baldi (2005), "Bayesian surprise attracts human attention"** — Key mechanism: **Bayesian surprise** as KL-divergence between prior and posterior. Unexpected inputs generate large belief updates, attracting processing resources. Direct relevance: Absorbed features may generate "surprise holes" — inputs that should be expected (based on feature semantics) but generate large prediction errors because the absorbed feature fails to activate.

#### Physics / Statistical Mechanics / Information Theory

8. **Bereska et al. (2025), "Superposition as Lossy Compression" (arXiv:2512.13568)** — Key mechanism: **Shannon entropy of SAE activations** quantifies "effective features" F = exp(H). Superposition metric ψ = F/N measures compression ratio beyond lossless limit. Direct relevance: Feature absorption may be an *inevitable consequence* of lossy compression when ψ > 1 and features are hierarchically structured.

9. **Tishby, Pereira & Bialek (2000), "The Information Bottleneck Method"** — Key mechanism: **Information bottleneck tradeoff**. Minimize I(X; X̃) while maximizing I(X̃; Y). Solutions exhibit **hierarchical phase transitions** — as β increases, representations bifurcate through second-order phase transitions, forming hierarchies of relevant quantizations. Direct relevance: SAEs optimize reconstruction (not prediction); the IB framework suggests absorption emerges from optimizing the wrong information tradeoff.

10. **Rose, Gurewitz & Fox (1990), "Statistical mechanics and phase transitions in clustering"** — Key mechanism: **Deterministic annealing**. Gibbs free energy F_β(Q) = E_Q[H] - (1/β)S(Q). At high temperature, entropy dominates (non-sparse); as T decreases, phase transitions produce emergent sparsity via bifurcations. Direct relevance: SAE training with fixed sparsity penalty may correspond to "quenching" — abrupt temperature drop causing defective (absorbed) feature configurations instead of annealed ones.

11. **Buhmann & Puzicha (1998), "Unsupervised texture segmentation in a deterministic annealing framework"** — Key mechanism: **Multiscale annealing**. Computational temperature β controls sparsity/resolution; optimal stopping temperature derived from generalization bounds. Direct relevance: SAEs use fixed sparsity hyperparameters rather than annealed schedules; this may trap features in local minima where absorption occurs.

12. **Li & Wang (2018), "Neural Network Renormalization Group" (Physical Review Letters 121, 260601)** — Key mechanism: **Variational RG with reversible generative networks**. Hierarchical change-of-variables from physical space to latent space with reduced mutual information. Direct relevance: SAEs perform a crude form of RG coarse-graining but without the multi-scale structure; absorption may reflect missing intermediate scales in the coarse-graining hierarchy.

13. **Gong & Xia (2022), "Interpreting Deep Learning by Establishing a Rigorous Corresponding Relationship with Renormalization Group" (arXiv:2212.00005)** — Key mechanism: **Formal DNN-RG correspondence**. Forward propagation = coarse-graining; optimal parameters = RG fixed points. Direct relevance: Feature absorption may indicate that SAE parameters are *not* at RG fixed points — the coarse-graining is incomplete, leaving hierarchical correlations unresolved.

14. **"Variational Garrote for Statistical Physics-based Sparse and Robust Variable Selection" (arXiv:2509.06383)** — Key mechanism: **Feature selection spin variables** with variational inference. Sharp phase transition: as superfluous variables are admitted, generalization degrades abruptly. Direct relevance: Analogous to SAE width expansion — adding latents beyond optimal causes abrupt degradation via absorption/splitting rather than gradual improvement.

#### Biology / Evolution / Ecology

15. **Pastore et al. (2021), "The evolution of niche overlap and competitive differences" (Nature Ecology & Evolution)** — Key mechanism: **Niche overlap (ρ) and competitive differences (κᵢ/κⱼ) co-evolve**. Competition kernel: `a(z,z') = exp(-(z-z')²/ω²)`. Species diverge to minimize overlap. Direct relevance: SAE latents are like species competing for "activation territory"; absorption occurs when competitive exclusion fails and one latent dominates overlapping niche space.

16. **Hart & Ross (2002), "Exploiting the Analogy between Immunology and Sparse Distributed Memories"** — Key mechanism: **Clonal selection as sparse feature competition**. Antigen presentation → competitive activation of lymphocyte clones; only high-affinity (sparse, high-quality) matches selected. Direct relevance: The immune system's "distributed recognition" mirrors SAE distributed representation; absorption may correspond to immune system failure where one clone dominates and suppresses diversity.

17. **Levins (1968), "Evolution in Changing Environments"** — Key mechanism: **Niche breadth model**: `Bᵢ = 1 / Σⱼ Pᵢⱼ²`. Species with broader niches are generalists; narrow niches are specialists. Direct relevance: Absorbed parent features are "generalists" being outcompeted by "specialist" child features in the SAE's competitive landscape.

18. **Gause (1934), "The Struggle for Existence"** — Key mechanism: **Competitive exclusion principle**. Two species competing for the same limiting resource cannot coexist indefinitely. Direct relevance: Parent and child features compete for the same "representation resource" (activation budget); absorption is the competitive exclusion of the parent by the more specific child.

19. **Valentine (1971), "Resource supply and species diversity patterns"** — Key mechanism: **Resource utilization functions** determine coexistence. Substitutable resources → multiplicative competition; non-substitutable → additive competition with arbitrary trait associations. Direct relevance: SAE features may be "substitutable" (child can replace parent) or "non-substitutable" (both needed); absorption occurs when substitutability is incorrectly assumed.

### Cross-Disciplinary Gaps

| Gap | Description | Opportunity |
|-----|-------------|-------------|
| **No IB analysis of absorption** | Information bottleneck theory has not been applied to explain feature absorption as an information tradeoff | Frame absorption as suboptimal compression under wrong distortion measure |
| **No RG-SAE correspondence** | Renormalization group theory predicts hierarchical coarse-graining, but no work connects RG fixed points to SAE feature quality | Use RG framework to predict optimal SAE depth/width for hierarchical features |
| **No free energy formulation** | Friston's free energy principle predicts sparse representations from generative models, but SAEs are discriminative | Develop generative SAE variant that minimizes variational free energy |
| **No ecological competition model for SAEs** | Competitive exclusion and niche partitioning have precise mathematical forms, never applied to SAE latent dynamics | Model SAE training as ecological competition; predict absorption from niche overlap metrics |
| **No annealing schedule for SAEs** | Deterministic annealing produces better solutions via temperature schedules; SAEs use fixed penalties | Design temperature-annealed sparsity schedule to avoid absorption traps |
| **No explaining-away architecture** | Explaining away requires recurrent inhibition; SAEs are feedforward | Add lateral inhibition to SAEs and measure absorption reduction |

---

## Phase 2: Initial Candidates

### Candidate A: Feature Absorption as Failed Explaining Away (from Neuroscience / Bayesian Inference)

- **Source principle**: "Explaining away" in Bayesian networks — when two causes could explain an observation, evidence for one reduces belief in the other. Neural implementation via lateral inhibition (Rozell LCA, Spratling's networks).

- **Structural correspondence**:
  - **Bayesian network**: Nodes represent hypotheses; edges represent conditional dependencies
  - **SAE**: Latents represent features; hierarchical co-occurrence creates implicit conditional dependencies
  - **Explaining away**: P(cause₁ | observation, cause₂) < P(cause₁ | observation) when causes are correlated
  - **Feature absorption**: Child feature activation "explains away" the need for parent feature activation, but the SAE lacks the mechanism to preserve parent belief when child is active
  - **Formal mapping**: In a proper explaining-away network, parent and child would both activate with appropriate posterior probabilities. In SAEs, the sparsity constraint forces a winner-take-all dynamic where the child "wins" and the parent is suppressed entirely — a *pathological* form of explaining away.

- **Hypothesis**: If we add explicit lateral inhibition (explaining-away dynamics) to SAEs, feature absorption will decrease because parent features will maintain residual activation even when child features are active.

- **Why not just a metaphor**: The correspondence is mathematical, not lexical. Both systems compute sparse representations via competitive dynamics. The LCA dynamics `du/dt = (1/τ)[b - u - Ga]` are isomorphic to SAE encoder computations if we add recurrent connections. The key difference is that explaining-away networks preserve marginal probabilities while SAEs discard them.

- **Novelty estimate**: 7/10. Explaining away is well-known in neuroscience and has been applied to sparse coding, but no work explicitly connects it to feature absorption in SAEs or proposes architectural modifications based on this correspondence.

### Candidate B: Feature Absorption as Suboptimal Renormalization Group Flow (from Statistical Physics)

- **Source principle**: Renormalization group (RG) coarse-graining in statistical mechanics. Microscopic degrees of freedom are integrated out to produce effective theories at larger scales. RG fixed points correspond to scale-invariant (universal) behavior.

- **Structural correspondence**:
  - **RG**: Coarse-graining operator R_Λ integrates out modes with momentum > Λ; repeated application flows to fixed points
  - **SAE**: Encoder f_θ maps high-dimensional activations to sparse latents; decoder g_φ reconstructs
  - **RG hierarchy**: Each iteration produces effective theory at larger correlation length ξ
  - **SAE hierarchy**: Deeper layers (in multi-layer SAEs) or larger dictionaries should capture larger-scale features
  - **Feature absorption**: The SAE "flows" to a fixed point where hierarchical correlations are not properly resolved — parent features are "integrated out" (absorbed) when they should remain as independent degrees of freedom
  - **Formal mapping**: The SAE loss L = ||x - g(f(x))||² + λ||f(x)||₀ is analogous to a **truncated RG** that lacks the multi-scale structure needed to preserve features at all scales. Matryoshka SAEs partially fix this by providing explicit scale separation.

- **Hypothesis**: SAEs with explicit multi-scale structure (analogous to RG's scale-by-scale coarse-graining) will show reduced absorption because each scale preserves its relevant degrees of freedom before passing to the next scale.

- **Why not just a metaphor**: The correspondence is formal. Gong & Xia (2022) proved that DNN forward propagation is mathematically equivalent to RG coarse-graining for the 1D Ising model. The SAE encoder-decoder structure is a specific DNN architecture; the RG framework makes quantitative predictions about depth requirements (H2: depth scales logarithmically with correlation length).

- **Novelty estimate**: 8/10. The DNN-RG correspondence is established, but no work applies it to SAEs or uses it to predict/understand feature absorption. The connection to Matryoshka SAEs as "multi-scale RG" is unexplored.

### Candidate C: Feature Absorption as Competitive Exclusion in Ecological Niche Space (from Evolutionary Biology)

- **Source principle**: Competitive exclusion principle (Gause, 1934) and niche partitioning (Levins, 1968; Pianka, 1973). Species competing for the same limiting resource cannot coexist; they must differentiate (character displacement) or one excludes the other.

- **Structural correspondence**:
  - **Ecology**: Species i has niche utilization function Uᵢ(z) on resource gradient z; overlap αᵢⱼ = ∫Uᵢ(z)Uⱼ(z)dz / ∫Uᵢ(z)²dz
  - **SAE**: Latent i has activation pattern aᵢ(x) on input space x; "niche overlap" between latents i and j measures how similarly they activate across inputs
  - **Competitive exclusion**: If αᵢⱼ > 1/Kᵢ (niche overlap exceeds carrying capacity ratio), species j excludes species i
  - **Feature absorption**: Child latent j "excludes" parent latent i because their niche overlap is near-perfect (child only fires when parent fires) and the child's "fitness" (sparsity contribution) is higher
  - **Formal mapping**: The SAE training dynamics are isomorphic to Lotka-Volterra competition with sparsity as the "carrying capacity" constraint. The fixed points of this dynamics predict which features survive and which are absorbed.

- **Hypothesis**: We can predict which features will be absorbed by computing niche overlap metrics between SAE latents and comparing to a "carrying capacity" derived from the sparsity budget. Features with overlap > threshold will be absorbed.

- **Why not just a metaphor**: The Lotka-Volterra equations are a general model of resource competition. SAE latents compete for activation budget (the "resource"). The mathematical structure (competition coefficients, carrying capacity, fixed point analysis) applies directly. Pastore et al.'s (2021) modern framework for niche overlap co-evolution provides testable quantitative predictions.

- **Novelty estimate**: 9/10. Competitive dynamics in neural networks are studied (WTA, lateral inhibition), but the explicit ecological analogy with niche overlap metrics and Lotka-Volterra dynamics has not been applied to SAE feature absorption. This is a genuinely novel transplant.

---

## Phase 3: Self-Critique

### Against Candidate A: Explaining Away

- **Shallow analogy attack**: Is this just mapping "competition" to "competition"? No — the structural correspondence is precise. Explaining away is a specific Bayesian inference phenomenon with a known neural implementation (LCA). The LCA dynamics are mathematically isomorphic to SAE computations with added recurrence. A domain expert (e.g., Spratling) would agree the transplant preserves the key property: lateral inhibition produces sparse representations where multiple causes can jointly explain data.

- **Scale mismatch attack**: Explaining away in V1 involves single-neuron competition. Does it apply to billion-parameter SAEs? Yes — the LCA framework scales to arbitrary network sizes. The key question is whether the *mechanism* (lateral inhibition) is computationally feasible at scale. Modern GPU-friendly approximations (e.g., block-sparse inhibition) make this plausible.

- **Prior transplant check**: Has someone added lateral inhibition to SAEs? Not explicitly for absorption. The closest is OrtSAE's orthogonality penalty, which is a *global* constraint rather than local competition. Explaining-away dynamics are local and input-dependent, which is architecturally different.

- **Testability attack**: Can we distinguish "this works because of explaining away" from "this works because of any sparsity-inducing mechanism"? Yes — the diagnostic is whether parent features maintain *graded* activation proportional to their marginal probability when child features are active. A generic sparsity mechanism would still suppress parents entirely; explaining away preserves graded responses.

- **Verdict**: STRONG

### Against Candidate B: Renormalization Group

- **Shallow analogy attack**: Is this just saying "hierarchy = hierarchy"? No — the RG framework makes specific quantitative predictions: (1) depth requirements scale logarithmically with correlation length, (2) fixed points correspond to optimal parameters, (3) relevant/irrelevant operators correspond to stable/unstable features. These are formal, testable predictions.

- **Scale mismatch attack**: RG is typically applied to infinite systems with translation invariance. SAEs operate on finite-dimensional activations without spatial structure. However, "correlation length" can be generalized to feature correlation structure. The RG-neural network correspondence (Gong & Xia) has been proven for finite systems.

- **Prior transplant check**: Has RG been applied to SAEs? Not directly. RG-inspired architectures exist (e.g., multi-scale CNNs), but the explicit connection to SAE feature absorption is novel. Matryoshka SAEs implicitly implement a crude RG but without the theoretical framework.

- **Testability attack**: Can we distinguish "multi-scale structure helps because of RG" from "multi-scale structure helps because more parameters"? Yes — RG predicts that *depth* (number of scales) matters more than *width* (parameters per scale). If we hold total parameters constant and vary only depth, RG predicts absorption will decrease with depth up to a logarithmic limit.

- **Verdict**: STRONG

### Against Candidate C: Competitive Exclusion

- **Shallow analogy attack**: Is this just "competition = competition"? The risk is real — "niche" and "feature" are both high-dimensional concepts. However, the Lotka-Volterra equations are a specific mathematical model with precise parameters (carrying capacity K, competition coefficients αᵢⱼ, growth rates rᵢ). These map directly to SAE parameters (sparsity budget, activation overlap, learning rate). The correspondence is formal, not lexical.

- **Scale mismatch attack**: Ecological competition operates on generational timescales with population dynamics. SAE training operates on gradient steps. However, both are dynamical systems approaching fixed points. The timescale difference is irrelevant to the fixed-point structure.

- **Prior transplant check**: Has competitive exclusion been applied to neural feature learning? Yes — WTA networks, soft competition, and lateral inhibition all draw on this principle. But the *specific* application to SAE absorption with niche overlap metrics (Levins/Pianka) and Lotka-Volterra fixed-point analysis is novel.

- **Testability attack**: Can we distinguish "absorption predicted by niche overlap" from "absorption predicted by simple correlation"? Yes — niche overlap integrates over the *entire* input distribution, not just pairwise correlation. Two features can have high pairwise correlation but low niche overlap if their activation patterns differ in shape. The ecological model predicts absorption from overlap, not correlation.

- **Verdict**: STRONG

---

## Phase 4: Refinement

### Dropped

None of the three candidates were dropped. All survived scrutiny with STRONG verdicts.

### Strengthened

**Candidate A (Explaining Away)**:
- Formalized the structural correspondence: In a Bayesian network with parent P and child C where C → P (C implies P), proper explaining away preserves P(C | evidence) and P(P | evidence) as independent posteriors. SAEs with sparsity constraint compute argmax over joint assignments, collapsing the marginal distribution.
- Diagnostic experiment: Measure the "posterior preservation ratio" — for absorbed features, compute P(parent activates | child activates, input) in the SAE vs. in a Bayesian network with the same structure. A true explaining-away mechanism should match the Bayesian posterior.
- Additional support: Spratling (2014) proved that explaining-away networks with competition produce better shape recognition than non-competitive networks. This predicts that explaining-away SAEs will have better downstream task performance.

**Candidate B (Renormalization Group)**:
- Formalized the correspondence: Define "SAE correlation length" ξ as the decay rate of feature-feature correlations: C(r) ~ exp(-r/ξ) where r is hierarchical distance (e.g., "starts with S" → "short" has r=1). RG predicts optimal SAE depth d* ~ log(ξ). Test by training SAEs with varying depth on synthetic hierarchies with known ξ.
- Diagnostic experiment: Compute the "RG fixed point quality" — measure whether SAE parameters satisfy the fixed-point equation R(f_θ) = f_θ where R is the coarse-graining operator. Absorption should correlate with deviation from fixed-point conditions.
- Additional support: Li & Wang (2018) showed that RG-inspired neural networks identify mutually independent collective variables. This predicts that RG-structured SAEs will learn more statistically independent features.

**Candidate C (Competitive Exclusion)**:
- Formalized the correspondence: Define niche utilization Uᵢ(x) = aᵢ(x) / max_x aᵢ(x) (normalized activation). Niche overlap αᵢⱼ = ∫Uᵢ(x)Uⱼ(x)dx / ∫Uᵢ(x)²dx. Carrying capacity Kᵢ = sparsity budget × (feature importance). Lotka-Volterra dynamics: dnᵢ/dt = rᵢnᵢ(1 - (nᵢ + Σⱼαᵢⱼnⱼ)/Kᵢ). Absorption occurs when parent i has αᵢⱼ > 1/Kᵢ for child j.
- Diagnostic experiment: Compute niche overlap and carrying capacity for all feature pairs in a pretrained SAE. Predict absorption using the competitive exclusion condition α > 1/K. Compare to ground-truth absorption labels from ablation experiments.
- Additional support: Pastore et al. (2021) showed that niche overlap and competitive differences co-evolve. This predicts that absorption rates should change predictably with SAE width (which changes carrying capacity).

### Selected Front-Runner

**Candidate C: Competitive Exclusion in Ecological Niche Space**

Rationale:
1. **Highest novelty** (9/10): No prior work applies ecological competition theory to SAE feature absorption.
2. **Strongest testability**: The diagnostic experiment (predict absorption from niche overlap metrics) is straightforward and requires only pretrained SAEs — aligned with the project's training-free constraint.
3. **Richest quantitative predictions**: The Lotka-Volterra framework provides explicit formulas for absorption thresholds, critical sparsity levels, and width-dependent transitions.
4. **Natural extension to mitigation**: Once we can predict absorption from niche overlap, we can design "niche construction" interventions (e.g., modifying activation patterns to reduce overlap) without retraining.
5. **Cross-validation opportunity**: The same framework predicts phenomena in other domains (feature hedging, dead features) that have independent empirical characterization.

---

## Phase 5: Final Proposal

### Title
"Feature Absorption as Competitive Exclusion: An Ecological Niche Theory of Sparse Autoencoder Pathologies"

### Source Principle
The **competitive exclusion principle** from evolutionary ecology states that species competing for the same limiting resource cannot coexist indefinitely. When niche overlap exceeds a threshold relative to carrying capacity, one species excludes the other. This principle has precise mathematical formulation in the **Lotka-Volterra competition equations** and **niche overlap metrics** (Levins, 1968; Pianka, 1973). Modern extensions (Pastore et al., 2021) show that niche overlap and competitive differences co-evolve, with character displacement as the evolutionary response.

### Structural Correspondence

| Ecology | SAE | Formal Mapping |
|---------|-----|----------------|
| Species | SAE latent | Each latent is a "species" competing for activation |
| Resource | Activation budget | Sparsity constraint limits total active latents |
| Niche utilization Uᵢ(z) | Normalized activation pattern aᵢ(x)/max(aᵢ) | How latent i "uses" the input space |
| Niche overlap αᵢⱼ | ∫Uᵢ(x)Uⱼ(x)dx / ∫Uᵢ(x)²dx | Similarity of activation patterns |
| Carrying capacity Kᵢ | Sparsity budget × feature importance | Maximum sustainable activation for latent i |
| Competitive exclusion | Feature absorption | Parent excluded when α > 1/K |
| Character displacement | Feature orthogonalization | Latents diverge to reduce overlap |

The SAE training dynamics (gradient descent on reconstruction + sparsity loss) are isomorphic to Lotka-Volterra competition with:
- Growth rate rᵢ ∝ gradient magnitude for latent i
- Carrying capacity Kᵢ ∝ (sparsity target / number of latents) × (feature frequency)
- Competition coefficient αᵢⱼ = niche overlap between latents i and j

### Hypothesis
Feature absorption in SAEs is **predictable competitive exclusion**: a parent feature is absorbed when its niche overlap with a child feature exceeds the inverse of its effective carrying capacity (α_parent,child > 1/K_parent). This predicts:
1. Absorption rate increases with niche overlap (measurable from activation patterns)
2. Absorption rate decreases with carrying capacity (wider SAEs or higher sparsity budgets)
3. Features with intermediate specificity (neither too general nor too specific) are most at risk
4. "Niche construction" interventions (modifying activation overlap without retraining) can mitigate absorption

### Method

**Phase 1: Quantify Niche Overlap in Pretrained SAEs**
1. Load Gemma Scope SAEs (16k/65k/1m widths, layers 0-17) via SAELens
2. For each latent, compute activation pattern aᵢ(x) across ~10k inputs
3. Normalize to get niche utilization: Uᵢ(x) = aᵢ(x) / max_x aᵢ(x)
4. Compute niche overlap matrix: αᵢⱼ = ∫Uᵢ(x)Uⱼ(x)dx / ∫Uᵢ(x)²dx
5. Estimate carrying capacity: Kᵢ = (L0 target / N_latents) × (frequency of i's top-activating tokens)

**Phase 2: Predict Absorption from Niche Metrics**
1. Use existing absorption labels from Chanin et al. (2024) for ground truth
2. For each feature pair (parent, child), test if α > 1/K predicts absorption
3. Compute precision/recall of the competitive exclusion predictor
4. Compare to baseline: simple correlation-based prediction

**Phase 3: Test Width-Dependent Predictions**
1. Compare niche overlap and absorption across SAE widths (16k → 65k → 1m)
2. Test prediction: wider SAEs have lower absorption because K increases
3. Test prediction: wider SAEs have more niche overlap (more latents compete) but lower exclusion rate (higher K dominates)

**Phase 4: Niche Construction Intervention (Training-Free)**
1. For predicted absorbed features, compute the "overlap deficit": δ = α - 1/K
2. Design activation steering that reduces overlap for high-risk pairs
3. Measure post-steering absorption rate vs. pre-steering baseline

### Diagnostic Experiment

**The key test**: Can we predict absorption using *only* niche overlap and carrying capacity, without any ablation experiments?

Procedure:
1. Select 100 features with known absorption status from Chanin et al. (50 absorbed, 50 not absorbed)
2. For each feature, compute (α, K) from activation patterns alone
3. Predict absorption if α > 1/K
4. Measure: accuracy, precision, recall, AUC-ROC
5. **Success criterion**: AUC-ROC > 0.75 (significantly above chance)

This confirms the analogy is load-bearing because:
- If the ecological model is correct, niche metrics should predict absorption
- If the analogy is decorative, simple correlation or frequency metrics would perform equally well
- The prediction uses *only* activation patterns — no causal ablation — making it a training-free diagnostic

### Experimental Plan

| Task | Model | Time | Description |
|------|-------|------|-------------|
| T1: Compute niche overlap on Gemma-2-2B SAEs | Gemma Scope 16k/65k/1m | 15 min | Batch activation extraction for 10k inputs |
| T2: Validate absorption prediction | Gemma Scope + sae-spelling labels | 20 min | Compare niche-based predictions to ground truth |
| T3: Width-dependent analysis | Gemma Scope 16k/65k/1m | 15 min | Test K-dependence of absorption |
| T4: Niche construction steering | Gemma Scope 16k | 15 min | Reduce overlap via activation steering |
| T5: Cross-architecture test | GPT-2 SAEs (SAELens) | 15 min | Validate generality across models |

**Total time**: ~80 minutes (can run T1-T3 in parallel, then T4-T5)

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Niche overlap poorly predicts absorption | Medium | Fallback to hybrid model incorporating frequency and hierarchy depth |
| Carrying capacity estimation is noisy | Medium | Use multiple estimators (L0 target, empirical frequency, feature importance) |
| Ecological model too simplistic | High | Extend to modern frameworks (Pastore et al. co-evolution, Chesson coexistence theory) |
| Steering intervention ineffective | Medium | Try multiple steering targets (reduce overlap, increase parent activation, orthogonalize) |
| Results only replicate known patterns | Low | The competitive exclusion framework makes novel quantitative predictions (thresholds, width-scaling) not in prior work |

### Novelty Claim

The specific cross-disciplinary insight is: **feature absorption in SAEs is not merely "competition" in a vague sense, but follows the precise quantitative laws of ecological competitive exclusion**. This claim is supported by:

1. **Structural isomorphism**: The Lotka-Volterra competition equations map directly onto SAE training dynamics with identifiable parameters (niche overlap, carrying capacity, growth rate).

2. **Novel predictions**: The framework predicts (a) absorption thresholds as a function of overlap and capacity, (b) width-dependent absorption transitions, (c) optimal sparsity levels that minimize absorption — none of which have been derived from prior SAE theory.

3. **Novel diagnostic**: A training-free absorption detector based on niche overlap metrics, eliminating the need for causal ablation experiments that limit current methods to early layers.

4. **Novel intervention**: "Niche construction" via activation steering to reduce overlap without SAE retraining — a concept borrowed from ecological niche construction theory (Odling-Smee et al.) that has no precedent in SAE research.

5. **Evidence of novelty**: No paper in the SAE literature (including the 15 core references in the literature survey) uses ecological competition theory, niche overlap metrics, or Lotka-Volterra dynamics to analyze feature absorption. The closest analogies are "competition" mentioned informally in Chanin et al. (2024) without formal development.

---

## References

### Cross-Disciplinary Sources
- Kersten, D., Mamassian, P., & Yuille, A. (2004). Object perception as Bayesian inference. *Annual Review of Psychology*.
- Rozell, C. J., Johnson, D. H., Baraniuk, R. G., & Olshausen, B. A. (2008). Sparse coding via thresholding and local competition in neural circuits. *Neural Computation*.
- Pecevski, D., Buesing, L., & Maass, W. (2015). Causal inference and explaining away in a spiking network. *Scientific Reports*.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
- Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. *arXiv:physics/0004057*.
- Bereska, L., et al. (2025). Superposition as lossy compression. *arXiv:2512.13568*.
- Gong, W., & Xia, Y. (2022). Interpreting deep learning by establishing a rigorous corresponding relationship with renormalization group. *arXiv:2212.00005*.
- Li, S.-H., & Wang, L. (2018). Neural network renormalization group. *Physical Review Letters*, 121, 260601.
- Rose, K., Gurewitz, E., & Fox, G. C. (1990). Statistical mechanics and phase transitions in clustering. *Physical Review Letters*.
- Gause, G. F. (1934). *The Struggle for Existence*. Williams & Wilkins.
- Levins, R. (1968). *Evolution in Changing Environments*. Princeton University Press.
- Pastore, A. I., et al. (2021). The evolution of niche overlap and competitive differences. *Nature Ecology & Evolution*.
- Hart, E., & Ross, P. (2002). Exploiting the analogy between immunology and sparse distributed memories. *Proceedings of the IEEE*.
- Spratling, M. W. (2014). A neural implementation of the Hough transform and the advantages of explaining away. *Image and Vision Computing*.

### SAE Literature
- Chanin, D., et al. (2024). A is for Absorption: Studying feature splitting and absorption in sparse autoencoders. *arXiv:2409.14507*.
- Hu, N., et al. (2025). Measuring sparse autoencoder feature sensitivity. *arXiv:2509.23717*.
- Karvonen, A., et al. (2025). SAEBench: A comprehensive benchmark for sparse autoencoders. *arXiv:2503.09532*.
- Bussmann, B., et al. (2025). Learning multi-level features with Matryoshka sparse autoencoders. *arXiv:2503.17547*.
- Korznikov, A., et al. (2025). OrtSAE: Orthogonal sparse autoencoders uncover atomic features. *arXiv:2509.22033*.

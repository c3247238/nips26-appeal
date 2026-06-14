# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Olshausen & Field (1996), "Emergence of simple-cell receptive field properties by learning a sparse code for natural images"** -- Foundational sparse coding model showing V1 learns Gabor-like filters via sparsity. Establishes that sparsity pressure in biological neural circuits produces competitive dynamics where neurons suppress neighbors with overlapping receptive fields.

2. **Carandini & Heeger (2012), "Normalization as a canonical neural computation", Nature Reviews Neuroscience** -- Divisive normalization: a neuron's response is divided by the pooled activity of neighboring neurons. This normalization is ubiquitous across sensory cortex and provides a graded suppression mechanism (unlike winner-take-all which is binary). Key insight: normalization *attenuates* but does not *eliminate* correlated responses, allowing hierarchically related features to coexist.

3. **Lian & Burkitt (2025), "Relating sparse and predictive coding to divisive normalization", PLOS Computational Biology** -- Formally unifies sparse coding, predictive coding, and divisive normalization in a two-layer neural model. Shows that a homeostatic function shapes nonlinear responses to replicate divisive normalization. The two-layer model implements sparse coding with a structure originating from predictive coding, and the residual error (prediction error) drives the lower-layer response. Directly relevant: demonstrates that these three principles -- all individually proposed as relevant to absorption -- are mathematically related.

4. **Rao & Ballard (1999), "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects"** -- Seminal predictive coding paper. The cortex transmits only prediction errors (residuals) upward; higher layers send predictions downward. The hierarchical residual coding structure means parent-level predictions are *subtracted* from the signal before child-level processing, so both levels remain active. This is the exact structural solution to absorption: encode residuals, not raw features.

5. **Rawat, Heeger & Martiniani (2024), "Unconditional stability of a recurrent neural circuit implementing divisive normalization", NeurIPS 2024** -- Proves that divisive normalization provides unconditional stability to recurrent networks even when spectral radius > 1. Relevant: suggests divisive normalization is not just a coding strategy but a stability mechanism, providing theoretical grounding for its use in SAE architectures.

6. **Rozell et al. (2008), "Sparse coding via thresholding and local competition in neural circuits"** -- The Locally Competitive Algorithm (LCA): neurons compete for activation via lateral inhibition, with only the best-matching neurons surviving. The competitive dynamics are structurally identical to TopK SAE's winner-take-all mechanism. LCA's thresholding + lateral inhibition produces the same absorption-like behavior seen in SAEs.

7. **Dual-feature selectivity in visual cortex (2025), eLife/bioRxiv** -- Neurons in V1 and V4 encode TWO features: one that excites and one that suppresses activity, around an elevated baseline. This "bidirectional coding" increases representational capacity and is supported by feature-specific inhibitory connectivity (connectomics: Schneider-Mizell et al., 2025). The mechanism shows that biological sparse coding avoids absorption-like collapse by using bidirectional feature tuning.

#### Physics / Information Theory

8. **Shannon (1959), Rate-distortion theory; Equitz & Cover (1991), "Successive refinement of information", IEEE Trans. IT** -- Rate-distortion theory establishes the fundamental tradeoff between compression rate and distortion. Successive refinement asks: can a source be optimally compressed at every intermediate rate? Equitz & Cover show that Gaussian sources with squared-error distortion ARE successively refinable (coarse-to-fine descriptions are optimal), but not all sources are. Key structural correspondence: SAE sparsity = rate constraint; reconstruction error = distortion; absorption = violation of successive refinement (adding more SAE width doesn't recover absorbed features because optimization has converged to a non-refinable solution).

9. **Tishby, Pereira & Bialek (1999), "The Information Bottleneck Method"** -- Formalizes compression of X while preserving information about Y. The IB Lagrangian min I(X;T) - beta * I(T;Y) has the same structure as the SAE objective min ||x - x_hat||^2 + lambda * L0(z). The parameter beta controls the rate-distortion tradeoff. Absorption in the IB view: at low beta (high compression), hierarchically redundant information is preferentially discarded because it contributes least to I(T;Y) per unit of I(X;T).

10. **Krzakala, Mezard, Sausset, Sun & Zdeborova (2012), "Statistical-physics-based reconstruction in compressed sensing", Physical Review X** -- Uses replica method from statistical mechanics to derive exact phase diagrams for sparse recovery. Shows sharp phase transitions: below a critical measurement ratio, signal recovery is impossible; above it, recovery is possible but may be computationally hard. Key import: feature absorption may exhibit a similar phase transition -- below a critical SAE width/L0 ratio, absorption is *inevitable* (thermodynamically favored), not just an optimization failure.

11. **Kabashima, Krzakala, Mezard, Sakata & Zdeborova, "Phase transitions and sample complexity in Bayes-optimal matrix factorization"** -- Directly analyzes dictionary learning (Y = FX + W with sparse X) using replica method. Derives phase diagrams for dictionary recovery. The overcomplete regime (alpha < 1) corresponds to SAEs. Shows that Bayes-optimal inference has sharp phase transitions in dictionary recovery quality. Import: provides mathematical tools to derive absorption phase diagrams.

12. **Ayonrinde, Pearce & Sharkey (2024), "Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs", arXiv:2410.11179** -- Recasts SAE training as Minimum Description Length optimization. In the MDL view, absorption is *efficient coding*: it IS the optimal strategy when the parent feature's unique information is low relative to its description cost. This reframing is crucial: absorption may not be a "bug" but a natural consequence of optimal compression.

13. **Tang et al. (2024), "A Unified Theory of Sparse Dictionary Learning: Piecewise Biconvexity and Spurious Minima", arXiv:2512.05534** -- Proves that the SDL optimization landscape has global minima corresponding to correct feature recovery but also prevalent spurious partial minima exhibiting feature absorption. Absorption corresponds to a specific class of spurious minima where a subset of features are correctly recovered but parent-child pairs collapse. Proposes feature anchoring as remedy.

#### Biology / Evolution / Immunology

14. **Francis (1960), "On the doctrine of Original Antigenic Sin"** -- Describes immune imprinting: the immune system's first encounter with an antigen creates memory B cells that dominate subsequent responses to *related* antigens. When a new, slightly different strain appears, the immune system preferentially activates existing memory cells (from the first exposure) rather than generating new antibodies specific to the novel strain. This is a PRECISE structural analogue to feature absorption: the "child" feature (specific strain) is "absorbed" by the "parent" feature (original strain antibodies), and the parent feature monopolizes the response.

15. **Immune imprinting literature (2024), PMC/Taylor & Francis** -- "Immune imprinting: The persisting influence of the first antigenic encounter with rapidly evolving viruses." Memory B cell competition is the mechanistic driver: swift recall of affinity-matured memory B cells lowers available antigen for naive B cells, effectively preventing new lineages from maturing. This is structurally identical to SAE absorption: the child feature (high-affinity memory cell) captures the shared information, preventing the parent feature (naive cell for new strain) from activating.

16. **PNAS (2024), "A speed limit on serial strain replacement from original antigenic sin"** -- Shows that immune imprinting creates a *speed limit* on how fast viral strains can replace each other. The analogy: absorption creates a "resolution limit" on how finely SAEs can distinguish hierarchically related features. Both arise from the same mechanism: competition for a limited resource (immune cells / activation slots) with bias toward the first-encountered representation.

17. **Gause's Competitive Exclusion Principle (ecology)** -- Two species competing for the same niche cannot stably coexist; the more fit species drives the other to extinction. Feature absorption is representational competitive exclusion: parent and child features compete for activation on overlapping inputs, and the sparsity objective selects the child (more specific) over the parent (more general) because the child provides more bits per activation slot.

18. **Lotka-Volterra competition models** -- Predict coexistence requires alpha_ij < K_i/K_j (competition coefficients bounded by carrying capacity ratios). The SAE analogue: coexistence of parent and child features requires their decoder cosine similarity to be below a threshold determined by their relative frequencies and the L0 budget. Provides a quantitative framework for predicting absorption.

#### Signal Processing / Compressed Sensing

19. **Mallat (1989), "A theory for multiresolution signal decomposition: the wavelet representation"** -- Multi-resolution analysis decomposes signals into nested approximation spaces V_0 subset V_1 subset ... subset V_J, with wavelet coefficients encoding the residual between adjacent levels. This is the exact structural solution to absorption: enforce that each resolution level encodes only the information NOT present at coarser levels, guaranteeing that both parent (coarse) and child (fine) features remain active.

20. **WEBA / DeSpaWN / Wavelet-VAE (2024-2025)** -- Recent learnable wavelet autoencoder architectures (WEBA in speech, DeSpaWN in time series, Wavelet-VAE in images) demonstrate that: (a) wavelet-based multi-resolution structure can be learned end-to-end, (b) sparsity-enforcing losses work naturally with wavelet coefficients, and (c) the resulting representations are hierarchically organized with reduced parameter counts. Import: these architectures provide engineering templates for implementing multi-resolution SAEs.

#### Hierarchical Classification (Machine Learning)

21. **Giunchiglia & Lukasiewicz (2020), "Coherent Hierarchical Multi-Label Classification Networks (C-HMCNN)", NeurIPS 2020** -- Proposes coherence constraints for hierarchical multi-label classification: if a child label is predicted, all ancestor labels must also be predicted. Implementation: post-processing via min/max operations on logits to enforce hierarchy. Directly translatable to SAEs: if a child feature fires, enforce that all parent features also fire, correcting absorption at inference time.

22. **Hierarchical constraint masks for taxonomic classification (2024-2025)** -- Binary masks restrict predictions to valid descendants during training and inference. When species predictions fail, errors remain taxonomically local: 92.5% correctly at genus level vs. 67.2% for flat baselines. The hierarchical model reduces mean taxonomic distance of errors by 38.2%. Import: even imperfect hierarchy-aware constraints dramatically improve error locality, suggesting that even approximate absorption correction would yield large practical benefits.

### Cross-Disciplinary Gaps

The following cross-field transplants have NOT been attempted in the SAE/mechanistic interpretability literature:

1. **Original Antigenic Sin / Immune Imprinting --> SAE Feature Absorption**: No paper has connected these two phenomena, despite the structural correspondence being remarkably precise. The immunology literature has 60+ years of theoretical and experimental tools for understanding imprinting dynamics that could be directly transplanted.

2. **Successive Refinement Theory --> SAE Architecture Design**: Equitz & Cover's conditions for successive refinability have not been applied to analyze whether SAE representations are refinable (i.e., whether adding width/capacity can recover absorbed features). This would yield a clean theoretical answer to "is absorption avoidable?"

3. **Phase Transition Analysis (Replica Method) --> Absorption Phase Diagrams**: Krzakala/Zdeborova's statistical physics tools for sparse recovery have not been applied to predict absorption onset as a function of SAE parameters. This would yield sharp phase boundaries rather than empirical scaling curves.

4. **Divisive Normalization --> SAE Encoder Architecture**: Despite Lian & Burkitt (2025) unifying sparse coding, predictive coding, and divisive normalization theoretically, no SAE architecture implements divisive normalization in the encoder to mitigate absorption.

5. **Coherent Hierarchical Classification Constraints --> SAE Post-Hoc Correction**: The min/max coherence operations from C-HMCNN have not been applied as post-hoc corrections to SAE activations to fix absorption without retraining.

---

## Phase 2: Initial Candidates

### Candidate A: Immune Imprinting Theory for Absorption (from Immunology)

- **Source principle**: Original Antigenic Sin / immune imprinting. The immune system's first encounter with an antigen creates memory B cells that dominate subsequent responses to related antigens. When a novel but similar antigen appears, existing memory cells (high affinity to original) outcompete naive cells (which could develop new specificities), "absorbing" the novel response into the original one.

- **Structural correspondence**: The mapping is formal, not metaphorical:
  - Antigens --> input activation vectors
  - B cell clones --> SAE latent features
  - Antibody affinity --> decoder cosine similarity with input
  - Memory B cell recall --> child feature activation
  - Naive B cell suppression --> parent feature absorption
  - Affinity maturation --> SAE training (gradient descent improving decoder directions)
  - Antigenic seniority (titer hierarchy) --> absorption hierarchy (child before parent)
  - Limited germinal center capacity --> L0 sparsity budget
  - Immune imprinting strength --> absorption rate

  The key structural identity: in both systems, a resource-limited competition (germinal center slots / activation slots) with hierarchy (related antigens / hierarchical features) and training bias (memory recall / gradient-optimized directions) produces systematic suppression of the more general representation by the more specific one.

- **Hypothesis**: (H1) Absorption rate in SAEs follows the same dynamics as immune imprinting strength: it increases with (a) the number of training steps (analogous to repeated antigen exposure strengthening imprinting), (b) the similarity between parent and child features (analogous to antigenic similarity), and (c) the ratio of child-to-parent frequency (analogous to antigen dose). (H2) The immunological solution -- "heterologous prime-boost" (exposing to diverse antigens to break imprinting) -- translates to a training curriculum that varies the input distribution to break absorption. This is structurally similar to the masked regularization approach of Narayanaswamy et al. (2026).

- **Why not just a metaphor**: The correspondence preserves the core dynamical mechanism -- competitive exclusion under limited capacity with hierarchical structure. An immunologist would agree that the mathematical structure of B cell competition in a germinal center (Lotka-Volterra-like dynamics with finite carrying capacity) is the same as feature competition in an SAE encoder with L0 budget. The predictions (absorption increases with training time, similarity, and frequency ratio) are quantitatively testable.

- **Novelty estimate**: 9/10 -- No paper connects immune imprinting to SAE feature absorption. The immunology literature provides 60+ years of quantitative theory (germinal center dynamics, affinity maturation models) that has never been applied to dictionary learning.

### Candidate B: Successive Refinement and Rate-Distortion Bounds for Absorption (from Information Theory)

- **Source principle**: Successive refinement of information (Equitz & Cover, 1991). A source is successively refinable if optimal coding at rate R1 can be extended to optimal coding at rate R2 > R1 by adding incremental information. Not all sources are successively refinable; the condition is that the optimal test channels form a Markov chain.

- **Structural correspondence**:
  - Source signal --> LLM activation vector
  - Codebook atoms --> SAE decoder vectors
  - Rate R --> L0 (number of active features)
  - Distortion D --> reconstruction error
  - Successive refinability --> whether increasing SAE width/L0 recovers absorbed features
  - Rate-distortion function R(D) --> the minimum L0 needed for a given reconstruction error
  - Excess rate --> L0 "wasted" by absorption (features that fire redundantly or are absent due to absorption)

  The key structural identity: absorption violates successive refinement. At low L0 (high compression), the SAE absorbs parent into child. When L0 increases, the absorbed parent is NOT automatically recovered because the optimization has already converged to a solution without it. This is exactly the failure mode of non-successively-refinable coding: you cannot improve the description by just adding more bits if the initial description was structured incompatibly.

- **Hypothesis**: (H1) The first-letter spelling task's feature hierarchy is NOT successively refinable under the SAE objective -- i.e., there exists no monotone path from low-L0 to high-L0 SAEs that progressively recovers absorbed features without retraining. This would explain why wider SAEs still exhibit absorption. (H2) A "successively refinable SAE" can be constructed by training hierarchical residual coding (each L0 level encodes only information not captured at lower levels), which would eliminate absorption by construction. (H3) The rate-distortion function R(D) for the first-letter task can be computed analytically, yielding a lower bound on absorption rate at each L0.

- **Why not just a metaphor**: This is not an analogy -- it is a direct application of information-theoretic formalism. The SAE objective IS a rate-distortion problem with L0 as rate and MSE as distortion. The successive refinement conditions are precise mathematical conditions (Markov chain on optimal test channels) that can be checked for the specific feature distributions in LLM activations.

- **Novelty estimate**: 8/10 -- MDL-SAEs (Ayonrinde et al., 2024) frame SAE training as compression, but do not analyze successive refinability or derive rate-distortion bounds specific to absorption. The unified SDL theory (Tang et al.) analyzes optimization landscape but not information-theoretic bounds.

### Candidate C: Divisive Normalization + Predictive Coding SAE Architecture (from Computational Neuroscience)

- **Source principle**: Divisive normalization (Carandini & Heeger, 2012) combined with predictive coding (Rao & Ballard, 1999), recently unified by Lian & Burkitt (2025). In the brain: (1) higher cortical layers send top-down predictions to lower layers, (2) lower layers compute prediction errors (residuals) and send those upward, (3) within each layer, divisive normalization adjusts neural responses by dividing by pooled neighborhood activity. This three-component system ensures that (a) parent-level features remain active (via top-down prediction), (b) child-level features encode only residual information, and (c) competition between features is graded rather than winner-take-all.

- **Structural correspondence**:
  - Cortical layer hierarchy --> SAE resolution levels (multi-scale dictionary)
  - Top-down prediction --> coarse-level reconstruction
  - Prediction error (residual) --> fine-level input (activation minus coarse reconstruction)
  - Divisive normalization pool --> set of features with high decoder cosine similarity
  - Normalization constant (1 + sum of pool activity) --> soft competition strength
  - Surround suppression (attenuated but present) --> parent feature attenuated but NOT eliminated

  The key structural identity: in the brain, divisive normalization prevents surround suppression from completely eliminating the center response -- it attenuates rather than abolishes. TopK/L1 SAEs implement hard winner-take-all, which completely suppresses the loser. Replacing hard WTA with divisive normalization would convert absorption (complete parent suppression) into attenuation (reduced but present parent activation).

- **Hypothesis**: (H1) A "Divisive Normalization SAE" (DN-SAE) where the encoder output for feature i is divided by (1 + alpha * sum_j a_j * sim(d_i, d_j)) -- with a_j the activation of feature j and sim(d_i, d_j) the decoder cosine similarity -- will reduce absorption rate by 40-60% while maintaining reconstruction quality within 5% of standard SAE. (H2) Combining DN-SAE with predictive coding (hierarchical residual coding between resolution levels) will reduce absorption to near zero.

- **Why not just a metaphor**: Lian & Burkitt (2025) formally proved that sparse coding with predictive coding structure produces divisive normalization as an emergent property in a two-layer neural model. This is not analogy -- it is mathematical equivalence. The SAE is already a sparse coding system; adding the predictive coding structure and letting divisive normalization emerge naturally would address absorption through the same mechanism the brain uses to prevent surround suppression.

- **Novelty estimate**: 7/10 -- Divisive normalization has been used in deep learning (AlexNet, 2012) and increases sparsity. Lian & Burkitt (2025) unified the theory. But no SAE implements divisive normalization to mitigate absorption. The predictive coding + divisive normalization combination for SAEs is novel.

---

## Phase 3: Self-Critique

### Against Candidate A: Immune Imprinting Theory

- **Shallow analogy attack**: Is this really structural, or just vocabulary mapping? The core mathematical structure -- Lotka-Volterra-like competition with finite capacity and hierarchical structure -- is genuinely shared between germinal center dynamics and SAE feature competition. However, immune imprinting has temporal dynamics (sequential antigen exposures over months/years) while SAE training sees all data in a single pass (or at most a few epochs). The temporal ordering that creates "original sin" in immunology (first exposure dominates) maps to initialization bias and early-training dynamics in SAEs, which is a less clean correspondence. An immunologist would likely agree that the competition mechanism is structurally similar but would note that the timescale separation and stochastic dynamics of germinal centers are far more complex than gradient descent. **Verdict: the competition mechanism is structural; the temporal dynamics are weaker.**

- **Scale mismatch attack**: Germinal centers contain O(10^3) B cells competing for O(10^2) T follicular helper cell signals. SAEs have O(10^4-10^6) features competing for O(10^1-10^2) activation slots. The capacity ratio (features:slots) is much larger in SAEs, which might make the ecological/immunological intuitions less applicable. However, the fundamental inequality (N_features >> N_slots) is preserved. **Verdict: scale is different but the qualitative dynamics should be preserved.**

- **Prior transplant check**: Searched arXiv for "immune imprinting" + "sparse autoencoder", "original antigenic sin" + "dictionary learning", "clonal selection" + "sparse coding", "affinity maturation" + "feature learning". No results. Artificial Immune Systems (AIS) exist as a field but focus on feature *selection* (choosing feature subsets), not on the absorption phenomenon in learned representations. **Verdict: this specific transplant appears genuinely novel.**

- **Testability attack**: The predictions are testable: (1) absorption rate vs. training steps (measure on Gemma Scope SAEs trained for different durations -- data available from JumpReLU paper which shows longer training increases absorption), (2) absorption rate vs. parent-child cosine similarity (measurable from decoder matrix), (3) absorption rate vs. frequency ratio (measurable from data statistics). The "heterologous prime-boost" intervention translates to curriculum diversity during training, which can be implemented and tested. The diagnostic experiment: train SAEs with "immune-inspired" curriculum (systematically vary which parent-child pairs are presented) and measure whether this breaks imprinting (absorption). **Verdict: STRONG testability.**

- **Verdict**: **STRONG** -- The structural correspondence is genuine and deep. The temporal dynamics mapping is the weakest link but can be addressed by focusing on the competition mechanism rather than the sequential exposure dynamics.

### Against Candidate B: Successive Refinement Theory

- **Shallow analogy attack**: This is not an analogy -- it is a direct application of information-theoretic formalism to a problem that IS a rate-distortion problem. The SAE objective literally minimizes reconstruction error (distortion) subject to sparsity (rate). The successive refinement question ("can you improve by adding capacity?") is precisely the empirical observation that wider SAEs still exhibit absorption. An information theorist would immediately recognize this as a successive refinement question. **Verdict: NOT an analogy; a direct theoretical application.**

- **Scale mismatch attack**: Rate-distortion theory assumes asymptotic regimes (infinite block length). SAEs operate on finite-dimensional activation vectors (d = 768 for GPT-2, d = 2304 for Gemma 2 2B). Finite-dimensional rate-distortion bounds may be loose. However, the qualitative predictions (absorption is optimal at low rate; successive refinability depends on source structure) should hold even in finite dimensions. **Verdict: finite-dimension effects may weaken quantitative bounds but qualitative predictions should hold.**

- **Prior transplant check**: MDL-SAEs (Ayonrinde et al., 2024) frame SAE training as compression, which is related to rate-distortion theory. However, they do not analyze successive refinability, do not derive absorption-specific rate-distortion bounds, and do not connect to the Equitz-Cover conditions. The unified SDL theory (Tang et al.) provides optimization landscape analysis but not information-theoretic bounds. **Verdict: the successive refinement framing and absorption rate-distortion bound are novel.**

- **Testability attack**: Computing R(D) for the first-letter spelling task requires knowing the joint distribution of features in LLM activations, which can be estimated empirically. The successive refinability test requires training SAEs at multiple L0 values and checking whether the low-L0 solution's features are subsets of the high-L0 solution's features. This is computationally straightforward using Gemma Scope SAEs. The diagnostic experiment: compute theoretical R(D) and compare to empirical absorption rates across L0 values. If they match, absorption is provably optimal (rate-distortion optimal). If empirical absorption exceeds theoretical minimum, the gap quantifies training suboptimality. **Verdict: STRONG testability.**

- **Verdict**: **STRONG** -- This is the most theoretically rigorous candidate. It could definitively answer whether absorption is avoidable or inherent.

### Against Candidate C: Divisive Normalization + Predictive Coding SAE

- **Shallow analogy attack**: The connection between sparse coding and divisive normalization is not an analogy -- Lian & Burkitt (2025) proved mathematical equivalence in a two-layer model. However, extending this to multi-layer SAEs with learned features is a non-trivial step. The brain's cortical hierarchy has many architectural properties (recurrent connections, temporal dynamics, multiple neurotransmitter systems) that SAEs lack. A neuroscientist might argue that divisive normalization in the brain is a dynamic process (temporal integration) while in an SAE it would be a static activation function modification. **Verdict: the mathematical connection is established; the extension to SAEs is reasonable but requires validation.**

- **Scale mismatch attack**: Divisive normalization in V1 operates over small receptive field neighborhoods (tens of neurons). SAE features are distributed across the entire dictionary (thousands to millions of features). Computing pairwise cosine similarities for the normalization pool is O(N^2) in the number of features, which may be prohibitive for large SAEs. However, approximate normalization using only high-similarity features (sparse neighborhood) would be O(N * k) with k << N. **Verdict: scalability concern is real but addressable with sparse neighborhoods.**

- **Prior transplant check**: Divisive normalization has been used in deep learning (Local Response Normalization in AlexNet, 2012; Group Normalization). However, these are used for training stability and generalization, NOT for mitigating feature absorption in SAEs. The specific combination of divisive normalization + predictive coding for SAE absorption mitigation has not been proposed. **Verdict: the specific application is novel, though the components are known.**

- **Testability attack**: The DN-SAE architecture is simple to implement (modify encoder activation function). Absorption rate can be measured using the standard Chanin et al. metric. The diagnostic experiment: compare absorption rate of DN-SAE vs. standard TopK SAE at matched L0 and reconstruction quality. If DN-SAE reduces absorption, the follow-up is to verify that the reduction is specifically due to divisive normalization (not just a different activation function) by comparing against other normalization schemes. **Verdict: STRONG testability.**

- **Verdict**: **MODERATE-STRONG** -- Well-grounded in neuroscience theory with established mathematical foundations. Requires careful engineering and the scalability concern is real. The main risk is that divisive normalization may improve absorption only modestly because the core issue (sparsity pressure) remains.

---

## Phase 4: Refinement

### Dropped
None of the three candidates died in the self-critique phase. All have genuine structural correspondences, prior transplant checks confirm novelty, and all are testable. However, priorities differ.

### Strengthened

**Candidate B (Successive Refinement / Rate-Distortion)** strengthened as the most theoretically rigorous candidate. It is not an analogy but a direct application of information theory. However, it may yield a *negative* result (absorption is rate-distortion optimal), which is theoretically important but does not suggest a mitigation strategy.

**Candidate A (Immune Imprinting)** strengthened as the most *actionable* candidate because it imports both a theoretical framework (competition dynamics) and a specific intervention (heterologous prime-boost / curriculum diversity). The temporal dynamics weakness can be addressed by focusing on the steady-state competition mechanism.

**Candidate C (Divisive Normalization)** strengthened as the most *architecturally concrete* candidate. The implementation is well-specified and testable.

### Formalized Structural Correspondences

**Immune Imprinting --> Feature Absorption (Candidate A)**

| Immunology | SAE | Mathematical Structure |
|---|---|---|
| Antigen | Input activation x | Signal to be encoded |
| B cell clone | SAE feature (decoder direction d_i) | Dictionary atom |
| Antibody affinity to antigen | cos(d_i, x) | Similarity measure |
| Germinal center capacity | L0 budget | Resource constraint K |
| Memory B cell | Established feature (high gradient magnitude) | Incumbent representation |
| Naive B cell | Potential new feature | Challenger representation |
| Affinity maturation | SAE training (gradient descent) | Optimization of d_i toward data |
| Antigenic seniority hierarchy | Feature absorption hierarchy | Systematic suppression of general by specific |
| Immune imprinting strength | Absorption rate | Fraction of suppressed responses |
| Heterologous prime-boost | Curriculum diversity / masked regularization | Breaking imprinting via diverse exposure |

**Successive Refinement --> Absorption (Candidate B)**

The SAE at L0 = k encodes activations into k active features. Let R(D) be the rate-distortion function for the empirical distribution of LLM activations at a given layer. Then:
- If k < R(D_target), absorption is unavoidable for distortion level D_target
- If the source (activation distribution under feature hierarchy) is NOT successively refinable, then increasing k does not monotonically reduce absorption -- consistent with empirical observations
- The "absorption excess" = L0_empirical - R(D_empirical) measures how many activation slots are "wasted" by absorption

### Selected Front-Runner

**Candidate A (Immune Imprinting Theory)** is selected as the front-runner for the final proposal because:
1. It provides the richest theoretical framework with the most specific, novel predictions
2. It imports a concrete mitigation strategy (heterologous prime-boost --> curriculum diversity)
3. It is deeply structural (not just vocabulary mapping) with formal competition dynamics
4. It connects to a vast, mature literature that has never been tapped by the ML interpretability community
5. It naturally integrates with Candidate B (immune imprinting dynamics can be analyzed via rate-distortion theory on the germinal center capacity) and Candidate C (divisive normalization can be seen as the biological implementation of graded rather than binary competition)

**However**, the final proposal integrates all three candidates as a unified framework: the immune imprinting analogy provides the conceptual framework and predictions, rate-distortion theory provides the theoretical bounds, and divisive normalization provides the architectural solution.

---

## Phase 5: Final Proposal

### Title
**"Representational Immune Imprinting: An Information-Theoretic Framework for Understanding and Mitigating Feature Absorption in Sparse Autoencoders"**

### Source Principle
Original Antigenic Sin (OAS) / Immune Imprinting in immunology: the immune system's first encounter with an antigen creates high-affinity memory B cells that dominate subsequent responses to related antigens through competitive exclusion in germinal centers. This produces a quantitative hierarchy ("antigenic seniority") where antibody titers are highest against the first-encountered antigen and progressively lower against subsequent related antigens. The mechanism is: memory B cell recall outcompetes naive B cell activation for limited T follicular helper cell signals, preventing the immune system from developing new specificities even when they would be beneficial.

### Structural Correspondence
The formal mapping between immune imprinting and SAE feature absorption:

**Shared mathematical structure**: Both systems implement sparse coding under hierarchical feature structure with limited capacity. The competition dynamics follow a Lotka-Volterra-like model:

```
da_i/dt = a_i * (r_i - sum_j alpha_ij * a_j)
```

where a_i is the activation (B cell population / feature activation), r_i is the growth rate (antigen affinity / input projection), and alpha_ij is the competition coefficient (resource overlap / decoder cosine similarity). Competitive exclusion (absorption) occurs when alpha_ij * a_j > r_i for the parent feature i under the child feature j.

**Key structural identity**: In both systems, the resource constraint (germinal center capacity / L0 budget) combined with hierarchical structure (related antigens / hierarchical features) and optimization toward high-affinity/low-loss representations produces systematic suppression of the more general representation by the more specific one. The suppression is NOT random -- it follows a strict hierarchy determined by feature specificity, exactly as antigenic seniority follows exposure order.

### Hypothesis
1. **Absorption follows imprinting dynamics**: Absorption rate is a monotonically increasing function of (a) training duration (analogous to repeated antigen exposure), (b) parent-child decoder cosine similarity (analogous to antigenic similarity), and (c) child-to-parent frequency ratio (analogous to antigen dose). These three factors multiplicatively determine absorption severity.

2. **Absorption is rate-distortion optimal at low L0**: There exists a critical L0 threshold below which absorption is the information-theoretically optimal coding strategy (absorbing saves rate at minimal distortion cost). Above this threshold, absorption becomes suboptimal but persists due to optimization landscape structure (spurious local minima per Tang et al.).

3. **Curriculum diversity breaks imprinting**: A training intervention inspired by heterologous prime-boost vaccination -- systematically varying which hierarchical features are present in training batches -- will reduce absorption rate by 30-50% without sacrificing reconstruction quality. This intervention disrupts the co-occurrence statistics that drive absorption during training.

### Method
The study has three components:

**Component 1: Quantitative Imprinting Dynamics (analytical + empirical)**
- Model SAE feature competition as a Lotka-Volterra system with features competing for L0 slots
- Derive closed-form conditions for absorption onset as a function of competition coefficient (cos(d_parent, d_child)), frequency ratio (f_child / f_parent), and L0 budget
- Validate predictions on Gemma Scope SAEs (GPT-2 small and Gemma 2 2B, layers 0-26, widths 16k and 65k) using the Chanin et al. absorption metric
- Measure the three predicted absorption drivers (training duration, cosine similarity, frequency ratio) and fit the Lotka-Volterra model

**Component 2: Rate-Distortion Bounds on Absorption (theoretical + empirical)**
- Estimate the empirical rate-distortion function R(D) for the first-letter spelling task using the known feature hierarchy
- Compute the theoretical minimum absorption rate at each L0 value
- Compare to empirical absorption rates across Gemma Scope SAEs
- Test successive refinability: do higher-L0 SAEs recover features absorbed at lower L0?

**Component 3: Immune-Inspired Mitigation (experimental)**
- **Curriculum diversity intervention**: During SAE training (or fine-tuning), systematically mask parent features in random batches (analogous to heterologous prime-boost), forcing the SAE to develop independent representations for parent and child. Compare to masked regularization (Narayanaswamy et al., 2026) which was independently motivated but structurally similar.
- **Divisive normalization encoder**: Replace TopK activation with soft competition: a_i = ReLU(z_i) / (1 + alpha * sum_j ReLU(z_j) * sim(d_i, d_j)) where sim is decoder cosine similarity. This converts hard competitive exclusion into soft attenuation, mirroring biological divisive normalization.
- **Post-hoc coherence correction**: For pre-trained SAEs, detect parent-child pairs via decoder cosine similarity > tau, then enforce coherence: if child fires, force parent to also fire at an activation proportional to the child's. This is the "vaccination booster" intervention: supplement the absorbed response.

### Diagnostic Experiment
The key test that confirms the analogy is load-bearing (not just decorative):

**Imprinting Curve Test**: Plot absorption rate vs. parent-child decoder cosine similarity for different L0 budgets. The immune imprinting theory predicts a specific functional form:

```
P(absorption) = sigmoid(alpha * cos(d_parent, d_child) * (f_child/f_parent) - L0/N)
```

where alpha is the competition strength parameter fit from data. If this functional form fits the empirical data significantly better than a null model (e.g., linear or constant absorption rate), it confirms that the competition dynamics driving absorption are structurally identical to immune imprinting dynamics.

**Successive Refinement Test**: Train SAEs at L0 = {5, 10, 20, 50, 100} on the same model. For each absorbed feature at L0 = 5, check whether it is recovered at L0 = 10, 20, etc. If absorption persists across L0 values (violated successive refinement), this confirms the rate-distortion framing. If it resolves, absorption is a capacity issue, not a structural one.

**Curriculum Diversity Test**: Train matched SAEs with and without the immune-inspired curriculum intervention. If the intervention reduces absorption rate by >20% without >5% reconstruction quality loss, the transplanted principle is the active ingredient. Compare to random masking (not hierarchy-aware) to verify that hierarchy-specific masking is necessary.

### Experimental Plan
- **Models**: GPT-2 small (d=768), Gemma 2 2B (d=2304), using pre-trained Gemma Scope SAEs for Components 1-2 and training new SAEs for Component 3
- **SAEs**: SAELens for training; Gemma Scope 16k/65k SAEs for analysis
- **Task**: First-letter spelling task (canonical absorption benchmark) + city-country hierarchy (cross-domain validation)
- **Metrics**: Absorption rate (Chanin et al.), mean absorption fraction (KronSAE), reconstruction MSE, L0, sparse probing accuracy (SAEBench)
- **Compute budget**: Component 1 (analysis only, <1 hour on single GPU); Component 2 (rate-distortion estimation, ~30 min); Component 3 (SAE training with curriculum intervention, ~1 hour per configuration x 6 configurations = 6 hours total, parallelizable to 2 hours on 3 GPUs)
- **Override**: project spec allows training-free analysis as primary, but Component 3 requires fine-tuning; this is justified by the novel intervention design

### Risk Assessment
1. **Absorption may be rate-distortion optimal**: If the theoretical analysis shows absorption equals the rate-distortion minimum, the mitigation interventions will have limited headroom. This would be a *theoretically important negative result* -- proving absorption is inherent to sparse coding under hierarchy.

2. **Lotka-Volterra model may be too simple**: Real SAE competition dynamics involve nonlinear interactions across all features simultaneously, not just pairwise competition. The Lotka-Volterra model may fit poorly for complex hierarchies with >2 levels. Mitigation: test on simple (2-level) hierarchies first, then extend.

3. **Divisive normalization scalability**: For SAEs with 65k+ features, computing the normalization pool (cosine similarities) is O(N^2). Mitigation: use sparse neighborhoods (only features with cos > 0.3) or batch approximation.

4. **Curriculum diversity may hurt reconstruction**: Masking parent features during training could reduce reconstruction quality on inputs where parents are important. Mitigation: use soft masking (reduce, don't eliminate) and track reconstruction quality on a held-out set.

5. **Temporal dynamics mismatch**: Immune imprinting depends critically on temporal ordering of antigen exposures; SAE training sees batches in relatively random order. The competition mechanism (the structural core) should still hold, but the "original sin" (first-encountered dominance) may not directly map to SAE initialization dynamics.

### Novelty Claim
The specific cross-disciplinary insight is the formal structural correspondence between Original Antigenic Sin in immunology and feature absorption in SAEs. Both are instances of competitive exclusion under limited capacity with hierarchical structure. This correspondence has NOT been drawn before in the ML interpretability literature (confirmed by arXiv search). The correspondence is not merely decorative -- it imports:

1. A quantitative theoretical framework (germinal center competition dynamics --> Lotka-Volterra model for absorption)
2. A specific intervention strategy (heterologous prime-boost --> curriculum diversity training)
3. Information-theoretic bounds (rate-distortion analysis of absorption optimality)
4. An architectural solution (divisive normalization --> soft competition replacing hard WTA)

The combination of biological insight, information-theoretic rigor, and concrete architectural interventions makes this proposal uniquely positioned to provide both theoretical understanding and practical mitigation of absorption.

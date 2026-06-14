# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Olshausen & Field (1997), "Sparse coding with an overcomplete basis set: A strategy employed by V1?"** -- The foundational work establishing that V1 simple cells implement sparse coding over overcomplete dictionaries, using an L1 penalty to select a minimal subset of basis functions. This is the direct biological ancestor of SAEs. Key insight: non-orthogonality of basis functions means sparsifying the code introduces nonlinear interactions between units, causing deviations from linear superposition. These unit-unit interactions are structurally analogous to the feature competition that produces absorption.

2. **Rozell et al. (2008), "Sparse coding via thresholding and local competition in neural circuits"** -- Proposes the Locally Competitive Algorithm (LCA), where neurons implement sparse coding through lateral inhibitory competition. Each neuron inhibits others proportional to dictionary coherence. This is essentially a Lotka-Volterra competition dynamic applied to neural coding: dominant features suppress weaker ones through competitive exclusion. The LCA convergence dynamics directly model how "absorbing" features outcompete "absorbed" ones.

3. **Sparse vs. Grandmother Cell vs. Distributed Coding Debate** -- Neuroscience has grappled for decades with the spectrum from fully distributed codes (all neurons encode all stimuli) to fully localist "grandmother cells" (one neuron per concept). The resolution is sparse coding as a middle ground. Feature absorption in SAEs maps onto this debate: absorbed features are cases where the SAE has shifted too far toward grandmother-cell coding (a single specific feature encodes what should be a distributed property).

4. **Friston's Free Energy Principle and Predictive Coding** -- The brain minimizes variational free energy (equivalent to the ELBO in variational inference). Predictive coding implements this via hierarchical error minimization, with forward connections carrying prediction errors and backward connections carrying predictions. The hierarchical structure of prediction errors (general predictions subsume specific ones) creates the same parent-child feature structure that causes absorption in SAEs.

5. **Tishby's Information Bottleneck** -- Deep networks learn by compressing inputs while preserving relevant information, with a tradeoff between I(X;T) (compression) and I(T;Y) (relevance). Absorption can be understood as the sparsity objective forcing excessive compression of the feature hierarchy, where child features "absorb" parent information because maintaining both would violate the compression budget.

#### Physics / Information Theory

6. **Phase Transitions in Compressed Sensing (Donoho et al.)** -- Sparse recovery from underdetermined systems exhibits sharp phase transitions: below a critical measurement-to-sparsity ratio, recovery fails catastrophically. The LASSO phase transition partitions the (sparsity, undersampling) plane into success/failure regions. Feature absorption may correspond to operating near or beyond such a phase boundary, where the SAE's sparsity constraint forces a transition from "all features recovered" to "hierarchical features collapsed."

7. **Renormalization Group (RG) and Hierarchical Feature Learning** -- Each layer of a deep network performs a coarse-graining analogous to an RG transformation, extracting increasingly abstract features. Recent work (AAAI 2025) formalizes this, showing layer-wise transformations correspond to RG scale transformations. Absorption is the pathology that occurs when the RG flow "skips" a scale: instead of maintaining both the fine-grained (child) and coarse-grained (parent) features, the child absorbs the parent's information at its scale.

8. **Coherence in Overcomplete Dictionaries (Candes et al., 2010)** -- Compressed sensing with redundant dictionaries requires bounding the restricted isometry property generalized to coherent dictionaries. High mutual coherence between dictionary atoms (analogous to high cosine similarity between SAE decoder vectors) degrades recovery guarantees. Feature absorption occurs precisely when the coherence between parent and child dictionary atoms is high, causing the L1 solver to preferentially select the child while dropping the parent.

9. **Boltzmann Machine Phase Transitions** -- RBMs exhibit phase transitions in feature learning dynamics: discontinuous transitions where feature recovery changes abruptly with data parameters, and compositional phases where intermediate numbers of hidden units activate. The sparsity-induced absorption in SAEs may be analogous to a phase transition where the system transitions from a "compositional" phase (both parent and child features active) to a "ferromagnetic" phase (only child features active).

#### Biology / Evolution

10. **Competitive Exclusion Principle (Gause's Law)** -- Two species competing for identical resources cannot stably coexist; one always outcompetes the other. The mathematical structure is the Lotka-Volterra competition model: species i excludes species j when the interspecific competition coefficient exceeds a threshold relative to intraspecific competition. This is a precise structural analogy to feature absorption: a child feature (specialist species) excludes a parent feature (generalist species) because the child is more efficient at "explaining" (consuming) the shared variance (resource) under the sparsity constraint.

11. **Immunodominance and Clonal Suppression** -- In immune responses, T cells specific for one epitope (immunodominant) suppress expansion of T cells specific for other epitopes (subdominant), even when the subdominant response would be independently viable. The mechanism is competition for shared resources (antigen-presenting cell surface, cytokines). This creates a strict dominance hierarchy where high-affinity clones absorb the immune response at the expense of lower-affinity but still functional clones -- directly analogous to specific SAE features absorbing general features.

12. **Waddington's Canalization and Genetic Assimilation** -- Developmental pathways become "canalized" (buffered against perturbation), creating robust phenotypes. Under extreme perturbation, canalization breaks down, exposing hidden genetic variation. Feature absorption can be understood through the canalization lens: the SAE training process canalizes activation patterns into specific feature channels. Parent features are "cryptic genetic variation" -- present in the model's representations but hidden by the canalized (absorbed) pathway through child features. Decanalizing perturbations (e.g., masked regularization) can re-expose absorbed features.

13. **Hsp90 as an Evolutionary Capacitor** -- Hsp90 buffers genetic variation under normal conditions but releases it under stress. This is structurally isomorphic to how SAE sparsity pressure buffers (absorbs) parent features under normal conditions, but the parent feature's information persists in the network (as shown by the linear probe succeeding where SAE features fail). The "stress" analogue would be reducing sparsity pressure or increasing dictionary width.

#### Signal Processing / Mathematics

14. **Hierarchical Dictionary Learning and Multi-scale Wavelets** -- Signal processing has long addressed the problem of decomposing signals at multiple scales simultaneously. Multi-resolution analysis (MRA) uses nested subspaces to maintain both coarse and fine features without absorption. The key insight: wavelets solve the absorption problem by construction, using orthogonal projections across scales. SAEs lack this multi-scale structure, making them vulnerable to cross-scale absorption.

15. **Limiting Similarity and Species Packing (Pigolotti et al.)** -- The Fourier transform of the competition kernel determines whether species can coexist continuously or must cluster. Positive-definite kernels allow continuous species packing; kernels with negative Fourier components force clustering. This mathematical framework from ecology can be directly applied to SAE decoder vectors: the "competition kernel" is the coherence structure of the decoder, and absorption corresponds to forced clustering in feature space.

### Cross-Disciplinary Gaps

1. **No formal mapping from ecological competitive exclusion to SAE feature dynamics**: Despite the structural isomorphism between Lotka-Volterra competition and locally competitive sparse coding, no paper has formalized absorption as a competitive exclusion phenomenon with ecological predictions (e.g., R* rule for feature survival).

2. **Immunodominance models have not been applied to dictionary learning**: The mathematical models of immunodominance hierarchy (affinity-dependent clonal suppression) provide a ready-made framework for predicting which features absorb which, but this connection has not been drawn.

3. **Renormalization group theory has not been applied to SAE failure modes**: While RG-deep learning connections are actively studied, no work examines what happens when the RG flow goes wrong (absorption as a pathological RG transformation that collapses scales).

4. **Phase transition theory from compressed sensing has not been applied to absorption specifically**: While the unified SDL theory paper mentions spurious minima, no work characterizes absorption as a phase transition with a sharp boundary in (width, L0, hierarchy depth) space.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption as Competitive Exclusion in Feature Ecology (from Ecology / Population Dynamics)

- **Source principle**: Gause's Competitive Exclusion Principle -- two species competing for the same limiting resource cannot stably coexist. The winner is determined by Tilman's R* rule: the species that can persist at the lowest resource level excludes all others. In Lotka-Volterra dynamics, species i excludes species j when alpha_ij / K_j > 1/K_i (the interspecific competition coefficient exceeds the threshold set by carrying capacities).

- **Structural correspondence**: 
  - Species = SAE latent features (dictionary atoms)
  - Resource = variance in the activation space that needs to be explained
  - Carrying capacity K_i = the amount of activation variance feature i can uniquely explain (its "niche size")
  - Competition coefficient alpha_ij = cosine similarity between decoder vectors i and j (how much feature i's niche overlaps with j's)
  - Sparsity pressure = environmental harshness (reduces effective carrying capacity for all species)
  - Population size = feature activation magnitude
  - **R* rule prediction**: The feature that can persist at the lowest activation magnitude (lowest reconstruction error per unit of L0 cost) wins. For parent-child pairs, the child has lower R* because it explains a more specific, lower-variance signal with less activation, so it excludes the parent.

- **Hypothesis**: Absorption rate is predictable from the "niche overlap" (decoder cosine similarity) between parent and child features and the "environmental harshness" (sparsity pressure L1/L0). Specifically, absorption occurs when: (1) niche overlap exceeds a critical threshold set by the sparsity budget, AND (2) the child feature's R* (minimum viable activation) is lower than the parent's. This predicts that absorption should exhibit a sharp phase transition as a function of decoder coherence x sparsity pressure.

- **Why not just a metaphor**: The mapping preserves the mathematical structure of Lotka-Volterra competition. The SAE's locally competitive dynamics (via the sparsity-inducing objective) are formally equivalent to Lotka-Volterra equations with carrying capacities set by reconstruction error budgets. The R* rule makes a quantitative prediction: measure the minimum activation threshold for each feature to contribute meaningfully to reconstruction (the feature's R*), and the feature with lower R* in each competing pair will absorb the other.

- **Novelty estimate**: 8/10. While sparse coding via local competition (Rozell et al. 2008) connects sparse coding to competitive dynamics, no one has formalized feature absorption as competitive exclusion with R*-rule predictions, or derived absorption rates from Lotka-Volterra stability conditions.

### Candidate B: Absorption as a Pathological Renormalization Group Flow (from Statistical Physics)

- **Source principle**: The Renormalization Group (RG) is a framework for understanding how physical systems behave at different scales. An RG transformation coarse-grains microscopic degrees of freedom to extract effective theories at larger scales, preserving only the relevant operators. The key property: a well-behaved RG flow maintains a separation between scales, with each scale contributing its own effective description. Pathological RG flows can collapse multiple scales into one, losing the information that should be preserved at intermediate scales.

- **Structural correspondence**:
  - Microscopic degrees of freedom = individual token activations
  - RG scale levels = feature abstraction levels (specific token features at fine scale, general category features at coarse scale)
  - RG fixed points = stable feature representations
  - Relevant operators = features that survive at a given scale
  - Irrelevant operators = features that are integrated out
  - **Absorption = the fine-scale (child) effective theory incorrectly integrating out the coarse-scale (parent) operator**: Instead of maintaining both scales with their respective features, the sparsity pressure causes the RG flow to collapse the coarse-scale description into the fine-scale one. The parent feature becomes an "irrelevant operator" that is integrated out, even though it carries independent information.

- **Hypothesis**: Absorption severity is governed by the "scaling dimension" of features -- how their activation magnitude scales with the number of contexts in which they fire. Parent features have large scaling dimensions (fire in many contexts), child features have small scaling dimensions (fire in few contexts). Under sparsity pressure, features with large scaling dimensions are less stable (cost more L0) and are preferentially absorbed. This predicts a universal scaling law: absorption rate ~ (parent frequency / child frequency)^beta, where beta depends on the sparsity pressure.

- **Why not just a metaphor**: The RG framework provides a precise mathematical formalism for multi-scale decomposition. The "scaling dimension" of an operator in RG has a direct counterpart in SAE feature activation statistics (log-frequency of feature firing). The prediction of a power-law relationship between absorption rate and frequency ratio is falsifiable and follows from dimensional analysis of the RG flow equations.

- **Novelty estimate**: 7/10. The RG-deep learning connection is well-studied (de Mello Koch et al. 2019, AAAI 2025), but applying RG to diagnose SAE failure modes (absorption as pathological scale collapse) is novel. The specific prediction of a power-law absorption-frequency relationship has not been proposed.

### Candidate C: Absorption as Immunodominance in Feature Repertoires (from Immunology)

- **Source principle**: Immunodominance -- in an immune response to a complex antigen with multiple epitopes, T cells specific for one "immunodominant" epitope suppress expansion of T cells specific for other "subdominant" epitopes. The mechanism is competition for shared resources: immunodominant T cells arrive at the antigen-presenting cell first (higher precursor frequency or faster recognition), consume the available cytokines and APC surface, and actively suppress subdominant clones through IFN-gamma-mediated attrition.

- **Structural correspondence**:
  - T cell clones = SAE latent features
  - Epitope specificity = decoder vector direction
  - Antigen-presenting cell (APC) = input activation vector
  - Clonal expansion magnitude = feature activation magnitude
  - TCR affinity for epitope = cosine similarity between decoder vector and input direction
  - Immunodominant clone = child feature (high specificity, high cosine similarity for its niche)
  - Subdominant clone = parent feature (broader specificity, lower peak cosine similarity)
  - Cytokine competition = sparsity budget (limited total activation)
  - Active suppression via IFN-gamma = lateral inhibition in SAE encoder
  - **Key mapping**: Immunodominance hierarchy = absorption hierarchy. The immune system, like the SAE, must allocate a limited response budget (L0 for SAE, clonal expansion capacity for immunity) across features of varying specificity. High-specificity features (child/immunodominant) win because they achieve higher "affinity" (cosine similarity) per unit of resource spent.

- **Hypothesis**: Feature absorption follows the same hierarchy rules as immunodominance: (1) features with higher precursor frequency in the training data (analogous to higher naive T cell precursor frequency) are more likely to become absorbers; (2) absorption magnitude correlates with the "affinity gap" between the absorber and the absorbed (cosine similarity difference); (3) breaking the dominance hierarchy requires the immune-system equivalent of "epitope focusing" -- presenting subdominant epitopes in isolation (analogous to masked regularization or data augmentation that decouples parent-child co-occurrence).

- **Why not just a metaphor**: The immunodominance literature has quantitative mathematical models (Ymir, BIDpred, active attrition models) that predict dominance hierarchies from epitope parameters. These models are systems of ODEs for clonal dynamics that can be directly translated into dynamical systems for SAE feature activations. The "active attrition" mechanism (dominant clones actively suppress subdominant ones via IFN-gamma) maps onto the encoder's lateral inhibition, which is not just passive competition but active suppression through negative encoder weights.

- **Novelty estimate**: 9/10. The structural correspondence between immunodominance and feature absorption has never been drawn. The immune system literature provides ready-made mathematical models for predicting dominance hierarchies that could be directly transplanted to predict absorption rates. The "epitope focusing" intervention strategy from vaccinology maps directly onto existing absorption mitigations (masked regularization).

---

## Phase 3: Self-Critique

### Against Candidate A: Competitive Exclusion

- **Shallow analogy attack**: Is this really structural, or just vocabulary mapping? The mapping IS structural: Rozell et al. (2008) already showed that sparse coding dynamics are formally equivalent to competitive neural dynamics, which are themselves Lotka-Volterra systems. The addition is formalizing absorption as the exclusion outcome and using the R* rule to predict which feature absorbs which. A population ecologist would recognize the mathematical structure immediately -- the SAE's sparsity-penalized reconstruction objective creates exactly the resource-mediated competition that Lotka-Volterra models describe. **Assessment: genuinely structural.**

- **Scale mismatch attack**: Lotka-Volterra models describe populations of organisms over generations; SAE features are learned by gradient descent over training steps. Does the timescale matter? The key dynamic is not temporal but equilibrium-state: Lotka-Volterra competitive exclusion describes the stable equilibrium of the system, and trained SAEs represent the (approximately) converged solution of the optimization. The equilibrium prediction (R* rule) translates regardless of the dynamics that reach it. **Assessment: no mismatch.**

- **Prior transplant check**: Sparse coding via local competition (Rozell et al. 2008) connects sparse coding to competitive dynamics. However, this work focuses on the inference algorithm (solving sparse coding), not on learning pathologies (absorption). The ecological competitive exclusion framing with R* rule predictions for absorption has not been proposed. The unified SDL theory paper (arXiv:2512.05534) provides a different theoretical lens (piecewise biconvexity) without the ecological framing. **Assessment: partially explored territory, but absorption-specific predictions are novel.**

- **Testability attack**: The R* rule prediction is directly testable. For each feature in a trained SAE, compute its R* (minimum activation at which it meaningfully reduces reconstruction error). For each parent-child pair, check whether the feature with higher R* is the one that gets absorbed. This requires no new SAE training, only analysis of existing SAEs. The phase transition prediction (absorption onset at critical coherence x sparsity) is also testable by sweeping sparsity on existing SAE families. **Assessment: highly testable with existing infrastructure.**

- **Verdict**: STRONG

### Against Candidate B: Pathological RG Flow

- **Shallow analogy attack**: The RG-deep learning analogy has been criticized as "essentially any hierarchical model looks like RG" (de Mello Koch et al. 2019). The specific claim here -- that absorption is a pathological scale collapse -- needs the "scaling dimension" of SAE features to be a well-defined, measurable quantity that actually predicts absorption. If the scaling dimension is just a proxy for frequency, the RG framing adds complexity without insight beyond "frequent features absorb infrequent ones." **Assessment: risk of being over-dressed vocabulary for a simpler observation.**

- **Scale mismatch attack**: RG operates on spatial/energy scales in physical systems. SAE features don't have a natural scale ordering -- a "parent" feature (e.g., "starts with letter A") and a "child" feature (e.g., "the token 'apple'") are at different semantic abstraction levels, but it's unclear whether this maps cleanly onto RG scale. The RG analogy works best when there's a continuous hierarchy of scales; SAE feature hierarchies may be discrete and sparse. **Assessment: moderate mismatch risk.**

- **Prior transplant check**: The RG-DL connection is well-studied (Mehta & Schwab 2014, de Mello Koch 2019, AAAI 2025). None of these works study SAE failure modes or absorption. The connection to absorption is novel but the underlying RG-DL mapping is contested. **Assessment: novel application but building on disputed foundation.**

- **Testability attack**: The power-law prediction (absorption rate ~ frequency ratio^beta) is testable, but it could also follow from much simpler reasoning without the RG apparatus. The "scaling dimension" needs a clear operational definition. If the experiment confirms the power law, it's hard to distinguish "this works because of RG-like scale dynamics" from "this works because frequent features are cheaper under sparsity." **Assessment: prediction exists but diagnostic experiment is weak.**

- **Verdict**: MODERATE (risk of decorative analogy without unique predictive power)

### Against Candidate C: Immunodominance

- **Shallow analogy attack**: The mapping between immune repertoire competition and SAE feature competition is detailed and multi-layered. A domain expert in immunology would recognize: (a) the resource competition for cytokines/APC surface maps onto the sparsity budget; (b) the active suppression via IFN-gamma maps onto encoder lateral inhibition; (c) the precursor frequency dependence maps onto training data statistics; (d) the epitope focusing intervention maps onto masked regularization. However, immunodominance involves a temporal sequence (naive T cell activation -> clonal expansion -> suppression of subdominants) while SAE training is a single optimization. The temporal dynamics are important in immunology but may not transfer. **Assessment: structural correspondence is deep but the temporal dynamics aspect needs care.**

- **Scale mismatch attack**: The immune system operates with 10^8 distinct clonotypes; SAEs have 10^3 to 10^6 features. The dimensionality is comparable. The immune system processes antigens one at a time; SAEs process batches of activations. This is a difference in dynamics but not in the equilibrium structure. **Assessment: no fundamental mismatch.**

- **Prior transplant check**: Artificial immune systems (AIS) were a significant ML paradigm in the 2000s (clonal selection algorithm, negative selection for anomaly detection). However, AIS focused on using immune metaphors for optimization, NOT for analyzing failure modes of other ML systems. Applying immunodominance theory specifically to predict SAE absorption has never been done. **Assessment: novel.**

- **Testability attack**: The key diagnostic experiment: if absorption follows immunodominance dynamics, then "epitope focusing" (presenting parent features in isolation, without co-occurring child features) should break the dominance hierarchy and recover the absorbed features. This maps to: constructing a training set where parent-level features appear without their children (e.g., tokens starting with "A" that don't contain known child-level patterns), and showing that SAEs trained on this balanced data exhibit reduced absorption. This is a genuine causal test of the mechanism. Additionally, the immunodominance model predicts that the absorption hierarchy should be predictable from (1) feature frequency and (2) decoder similarity, with a specific functional form borrowed from clonal dynamics ODEs. **Assessment: strong diagnostic experiment with causal test.**

- **Verdict**: STRONG

---

## Phase 4: Refinement

### Dropped

**Candidate B (Pathological RG Flow)** is dropped. While intellectually appealing, the analogy risks being decorative rather than load-bearing. The core prediction (power-law absorption vs. frequency) likely follows from simpler analysis without the RG machinery. The scale mismatch between continuous physical scales and discrete semantic hierarchies weakens the correspondence. The RG framing may be incorporated as supplementary theoretical language within one of the surviving candidates, but should not be the primary driver.

### Strengthened Candidates

**Candidate A (Competitive Exclusion)** and **Candidate C (Immunodominance)** survive scrutiny. They can be unified into a single, stronger proposal because immunodominance IS a special case of competitive exclusion in biological systems -- it is competitive exclusion applied to immune repertoire dynamics with specific mechanisms (affinity-based competition, active suppression). The unification provides:

1. **From Ecology (Candidate A)**: The mathematical framework (Lotka-Volterra ODEs, R* rule, competition kernels, Fourier analysis of coexistence conditions)
2. **From Immunology (Candidate C)**: The specific dominance hierarchy predictions (affinity-dependent, precursor-frequency-dependent), the active suppression mechanism (not just passive competition), and the intervention strategy (epitope focusing / masked regularization)

### Formalized Structural Correspondence

The unified "Feature Ecology" framework maps:

| Source Domain (Ecology/Immunology) | Target Domain (SAE Feature Absorption) |
|---|---|
| Species / T cell clone | SAE latent feature (decoder atom) |
| Ecological niche / Epitope | Direction in activation space |
| Resource (food/light/cytokines) | Explained variance in input activations |
| Carrying capacity K_i | Maximum variance feature i can explain solo |
| Competition coefficient alpha_ij | |cos(d_i, d_j)| (decoder cosine similarity) |
| Population size N_i | Feature activation magnitude a_i |
| Environmental harshness | Sparsity pressure (L1 coefficient lambda) |
| R* (min viable resource level) | Min activation at which feature reduces MSE more than it costs in L1 |
| Competitive exclusion | Feature absorption (parent excluded by child) |
| Immunodominance hierarchy | Absorption hierarchy among co-occurring features |
| Niche partitioning / character displacement | Feature splitting (SAEs splitting one concept into multiple features) |
| Epitope focusing (vaccination) | Masked regularization / co-occurrence disruption |
| Paradox of the plankton | Simultaneous presence of absorption and feature splitting in the same SAE |
| Hsp90 evolutionary capacitor | Network's latent capacity to represent parent features (revealed by probes, hidden from SAE) |

### Selected Front-Runner

**Candidate C (Immunodominance) enriched with the mathematical scaffolding from Candidate A (Competitive Exclusion)**. The immunodominance framing is chosen over the pure ecology framing because:
1. It provides the most novel contribution (no prior work connecting immunodominance to SAE)
2. It naturally incorporates BOTH passive competition AND active suppression (matching both the L1 penalty and the encoder's learned inhibitory weights)
3. It provides a concrete intervention strategy (epitope focusing -> masked regularization) that connects to existing mitigation work
4. The mathematical models of clonal dynamics are well-developed and can be directly adapted

---

## Phase 5: Final Proposal

### Title

**Feature Immunodominance: Understanding and Predicting SAE Absorption through the Lens of Immune Repertoire Competition**

### Source Principle

In immunology, when the immune system encounters a complex antigen containing multiple epitopes, the response is not distributed equally across all possible epitope-specific T cell clones. Instead, a strict **immunodominance hierarchy** emerges: clones specific for one or a few "immunodominant" epitopes expand massively, while clones specific for other ("subdominant") epitopes are actively suppressed -- even though these subdominant clones are independently functional and would expand robustly if the immunodominant epitopes were absent.

The mechanism has three components:
1. **Affinity-based priority**: Clones with higher TCR affinity for their epitope are activated faster and begin expanding earlier.
2. **Resource competition**: Expanding clones compete for limited cytokines and APC surface area. Early-activated (immunodominant) clones consume these resources, starving late-activated (subdominant) clones.
3. **Active suppression**: Immunodominant clones release IFN-gamma, which actively attenuates subdominant clonal expansion -- this is not passive neglect but active inhibition.

The dominance hierarchy is predictable from measurable parameters: epitope binding affinity, naive precursor frequency, and antigen dose. Crucially, subdominant clones are not non-functional -- they are suppressed. Removing the immunodominant epitope (a technique called "epitope focusing" in vaccinology) fully restores the subdominant response.

### Structural Correspondence

The formal mapping between immune repertoire dynamics and SAE feature dynamics:

**Generative model**: Let x be an input activation. The SAE reconstructs x as a sum of feature activations times decoder vectors: x_hat = sum_i a_i * d_i, where a_i >= 0 and the objective minimizes ||x - x_hat||^2 + lambda * sum_i |a_i|.

Consider a parent feature P (e.g., "starts with A") and child feature C (e.g., "the token apple"). When a token is "apple", both P and C are "true" features, but the SAE must decide whether to activate both (costing 2 units of L0) or just C (costing 1 unit of L0, with C's decoder vector absorbing P's contribution).

In the immune system analogy:
- The input x is the **antigen**
- Features P and C are **T cell clones** with different epitope specificities
- Decoder vectors d_P and d_C are **TCR binding profiles**
- The cosine similarity cos(d_P, d_C) is the **cross-reactivity** between clones
- The sparsity budget lambda * L0 is the **cytokine/APC budget**
- Feature activation a_i is the **clonal expansion magnitude**
- The encoder weights W_{enc} implement **lateral inhibition** (analogous to IFN-gamma suppression)

**Immunodominance prediction for absorption**: Clone C (child) immunodominates over clone P (parent) when:
1. C has higher "affinity" for the shared input contexts: cos(d_C, x_context) > cos(d_P, x_context) for shared contexts
2. C has higher "precursor frequency" advantage: C fires in a subset of P's contexts but with higher activation
3. The "cytokine budget" (sparsity) is tight: high lambda forces a winner-take-all dynamic

This maps to Chanin et al.'s informal argument but adds: (a) a precise quantitative criterion (affinity gap + frequency ratio + sparsity pressure), (b) the active suppression mechanism (encoder weights learn to suppress parent features, not just neglect them), and (c) an intervention strategy (epitope focusing).

### Hypothesis

1. **Dominance hierarchy prediction**: For any parent-child feature pair in a trained SAE, the probability of absorption is a monotonically increasing function of three measurable quantities: (a) the decoder cosine similarity cos(d_P, d_C), (b) the frequency ratio freq(C) / freq(P) within shared contexts, and (c) the sparsity pressure lambda. Specifically, borrowing from clonal dynamics: P(absorption) = sigmoid(alpha * cos(d_P, d_C) + beta * log(freq_ratio) + gamma * lambda + delta).

2. **Active suppression signature**: If absorption is like immunodominance (not just passive competition but active suppression), then the encoder weights for absorbed parent features should show learned negative correlation with the absorbing child feature's encoder direction. This is stronger than the passive "not selected" prediction of pure competitive exclusion.

3. **Epitope focusing intervention**: Disrupting co-occurrence of parent and child features during training (analogous to epitope focusing in vaccinology) should reduce absorption. This is exactly what masked regularization (Narayanaswamy et al. 2026) does, and the immunodominance framework provides a principled explanation for WHY it works: removing the immunodominant epitope (child) from some training contexts allows the subdominant response (parent feature) to expand without suppression.

4. **Repertoire diversity vs. protection tradeoff**: Immunology shows that immunodominance is not purely harmful -- it concentrates the immune response on the most effective epitopes, providing better protection per unit of immune resource. Similarly, absorption is not purely harmful: it concentrates the SAE's representation budget on the most informative features. The immunodominance framework predicts an optimal absorption rate that balances "repertoire diversity" (number of distinct features) against "protection efficiency" (reconstruction quality per L0 unit).

### Method

1. **Formalize the Feature Immunodominance Model**: Adapt the ODE system for clonal dynamics to SAE feature dynamics:
   - da_i/dt = r_i * a_i * (1 - sum_j alpha_ij * a_j / K_i) - gamma_suppress * a_i * max_j(a_j * I_{j dominates i})
   - where r_i = base growth rate (gradient signal), K_i = carrying capacity (max useful activation), alpha_ij = competition coefficient (decoder coherence), gamma_suppress = active suppression strength
   - Derive equilibrium conditions that predict which features survive (are expressed) and which are absorbed (go to zero)

2. **Compute Feature Ecology Metrics on Existing SAEs**: For pre-trained Gemma Scope and GPT-2 SAEs:
   - Compute the "competition matrix" alpha_ij from pairwise decoder cosine similarities
   - Estimate R* for each feature (min activation at which MSE reduction > L1 cost)
   - Identify parent-child relationships from activation co-occurrence and decoder similarity
   - Test whether the R* rule correctly predicts absorption direction

3. **Test the Active Suppression Signature**: For absorbed features identified via the Chanin et al. metric:
   - Examine encoder weight patterns for absorbed parent features
   - Test whether there is a statistically significant negative correlation between the parent's encoder direction and the child's encoder direction (evidence of active suppression, not just passive neglect)
   - Compare with non-absorbed parent-child pairs as controls

4. **Validate Epitope Focusing Prediction**: Use the immunodominance model to predict which specific parent-child co-occurrences to disrupt, then verify that masked regularization targeting those specific co-occurrences reduces absorption more efficiently than random masking.

### Diagnostic Experiment

**The key causal test that confirms the analogy is load-bearing (not decorative):**

If feature absorption follows immunodominance dynamics with active suppression, then:
- **Prediction A**: Removing the immunodominant feature (child) during evaluation should cause the absorbed feature (parent) to immediately activate -- analogous to how removing the immunodominant epitope from a secondary challenge restores subdominant T cell expansion. **Test**: Zero-ablate the absorbing child feature's encoder weights and check whether the parent feature's false-negative rate drops. If absorption is passive neglect, the parent should NOT recover (its detector is untrained). If absorption involves active suppression, the parent SHOULD partially recover (its detector exists but is suppressed by the child).

- **Prediction B**: The absorption hierarchy should be predictable from pre-training feature statistics (decoder coherence, frequency ratio) using the immunodominance model's functional form, achieving higher prediction accuracy than a baseline model that uses only frequency or only cosine similarity.

- **Prediction C**: There should be a measurable "repertoire diversity vs. protection efficiency" tradeoff curve, analogous to the immunological tradeoff between breadth and depth of immune response. Plotting the number of non-absorbed features (diversity) against reconstruction quality per L0 (efficiency) across SAEs with varying sparsity should reveal a Pareto frontier.

### Experimental Plan

All experiments are training-free analysis of existing pre-trained SAEs, using GPT-2 Small and Gemma-2-2B via SAELens/Gemma Scope.

| Phase | Description | Time Budget |
|-------|-------------|-------------|
| 1. Feature ecology metrics | Compute decoder coherence matrix, co-occurrence stats, R* for all features in 3 SAEs | 45 min |
| 2. Absorption prediction model | Fit immunodominance model to predict absorption from ecology metrics; compare with baselines | 30 min |
| 3. Active suppression test | Encoder weight analysis for absorbed vs. non-absorbed parent features | 30 min |
| 4. Ablation recovery test | Zero-ablate absorber features, measure parent recovery | 30 min |
| 5. Epitope focusing validation | Targeted vs. random masking comparison using co-occurrence disruption | 45 min |
| 6. Diversity-efficiency frontier | Sweep sparsity, plot non-absorbed features vs. reconstruction per L0 | 30 min |

Total: ~3.5 hours across 6 tasks (each within 1-hour budget).

### Risk Assessment

1. **The competition may be purely passive**: If the encoder does not learn active suppression weights (the IFN-gamma analogue is absent), then the immunodominance framing reduces to simpler competitive exclusion, losing the most distinctive prediction. Mitigation: the active suppression test (Phase 3) is explicitly designed to check this.

2. **Feature hierarchy may not be binary parent-child**: Real feature hierarchies may involve complex multi-level, multi-parent structures that the two-species competition model doesn't capture. Mitigation: extend to multi-species Lotka-Volterra with community ecology tools (species interaction networks).

3. **Decoder cosine similarity may not be the right competition kernel**: Features may interact through more complex nonlinear relationships not captured by linear coherence. Mitigation: test with both linear and nonlinear interaction metrics.

4. **The immunodominance model may over-predict absorption**: Immunodominance is a strong phenomenon in biology because the immune system has evolved active suppression mechanisms. SAEs may only exhibit the weaker, passive competition version. Mitigation: report results for both the full immunodominance model and the passive-competition-only baseline.

### Novelty Claim

The specific cross-disciplinary insight: **Feature absorption in SAEs is structurally isomorphic to immunodominance in adaptive immunity -- not merely analogous, but governed by the same mathematical dynamics of affinity-based, resource-limited competition with active suppression.** This connection has not been drawn in any prior work. It provides:

1. A quantitative predictive model for absorption rates based on measurable feature properties (decoder coherence, frequency, sparsity)
2. A mechanistic explanation for WHY masked regularization works (it is epitope focusing -- a well-understood vaccinology technique)
3. A novel prediction about active suppression in SAE encoders that can be empirically tested
4. A principled framework for understanding the absorption-reconstruction tradeoff (the immunological breadth-depth tradeoff)

Evidence that this has not been applied before: Searched arXiv for "immunodominance" AND ("sparse autoencoder" OR "dictionary learning" OR "feature absorption") -- zero results. Searched Google Scholar for the same -- zero results. The artificial immune systems (AIS) literature from the 2000s used immune metaphors for optimization algorithms (clonal selection algorithm for search, negative selection for anomaly detection) but never for analyzing failure modes of representation learning.

# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Hierarchical Cortical Organization** — Visual cortex exhibits hierarchical feature organization where simple cells (edges, bars) combine into complex cells (shapes, objects). This suggests neural representations naturally form parent-child hierarchies where general features (edges) must coexist with specific features (object parts). **Structural analogy**: SAE feature absorption mirrors cortical "pollution" where higher-order neurons inadvertently respond to lower-order patterns when co-activated.

2. **Receptive Field Overlap and Binding Problem** — Neural assemblies face the challenge of binding related features without confusion. Hub neurons in cortical networks act as superordinate nodes connecting multiple specialized features. **Structural analogy**: Absorbing latents in SAEs function like hub neurons that "steal" activation from specialized child features during sparse optimization.

3. **Predictive Coding and Free Energy Principle** — The brain minimizes free energy by predicting sensory inputs at multiple scales. Parent predictions (high-level context) shape how child predictions (low-level details) are interpreted. **Structural analogy**: SAE absorption occurs when high-level (parent) features predict low-level (child) feature activations, causing the SAE to "skip" child features and route predictions through parent channels.

4. **Neuroplasticity and Critical Periods** — Synaptic pruning during development eliminates redundant connections. The timing of critical periods determines which representations become dominant. **Structural analogy**: SAE training with sparse penalties acts like neuroplasticity — features that co-occur frequently become "pruned" (absorbed) into dominant representations during the optimization trajectory.

5. **Attentional Bottleneck and Information Routing** — Attention acts as an information routing bottleneck in biological neural networks, selecting which features are transmitted to higher processing stages. **Structural analogy**: SAE's L1 sparsity penalty creates an attentional bottleneck that determines which features survive; features that compete for the same sparse slots undergo absorption.

#### Physics / Statistical Physics

6. **Spin Glass Energy Landscapes** — Spin glass systems have rugged energy landscapes with many local minima. When cooling (annealing), systems can get trapped in states where one minimum "absorbs" contributions from nearby states. **Structural analogy**: SAE training with gradient descent is like annealing a spin glass; absorption corresponds to the system settling into local minima that conflates parent-child feature representations.

7. **Phase Transitions in Compressed Sensing** — Compressed sensing exhibits phase transitions: below a critical sampling ratio, perfect recovery is impossible; above it, sudden recovery emerges. The transition is sharp and determined by the ratio of measurements to sparsity. **Structural analogy**: SAE absorption may exhibit a phase transition in the dictionary size / sparsity trade-off. When the SAE bottleneck is too narrow relative to feature hierarchy depth, absorption becomes inevitable.

8. **Replica Symmetry Breaking** — In spin glasses, replica symmetry breaking indicates the system has entered a complex state with multiple pure states. This creates " ultrametric" organization where states cluster hierarchically. **Structural analogy**: Feature hierarchies in SAEs may naturally form ultrametric structures when absorption occurs, with parent-child relationships encoded in the overlap structure rather than explicit hierarchy.

9. **Renormalization Group Flow** — Renormalization group (RG) theory describes how physical systems change scale. At each scale, degrees of freedom are "integrated out," with coarse-grained variables capturing essential physics while fine details are absorbed. **Structural analogy**: SAE absorption is analogous to RG flow: the decoder "integrates out" fine-grained (child) features, with coarse-grained (parent) features retaining the essential structure.

10. **Kosterlitz-Thouless Transitions** — In certain phase transitions, topological defects (vortices) persist until a critical temperature where they suddenly annihilate. **Structural analogy**: The transition from non-absorbing to absorbing SAE configurations may be analogous to defect annihilation — above a critical sparsity threshold, absorbed states suddenly dominate.

#### Biology / Evolution

11. **Immune System Clonal Selection** — B cells undergo clonal selection where high-affinity variants proliferate while low-affinity variants are suppressed. However, this can cause the immune response to "overspecialize" and lose coverage of rare antigens. **Structural analogy**: SAE's sparse optimization selects for frequently co-occurring features (high "affinity"), potentially suppressing rare but important features — analogous to the immune repertoire becoming "absorbed" by dominant pathogen families.

12. **Evolutionary Pressure and Trait Trade-offs** — Evolution often produces trade-offs where one trait improves at the expense of another (pleiotropy). A mutation beneficial for one function may be "absorbed" into the phenotype if it also affects other functions. **Structural analogy**: SAE weights face pleiotropic constraints — a single weight vector may need to represent both parent and child features, leading to absorption when optimizing for sparsity.

13. **Epigenetic Inheritance and Gene Regulation** — Epigenetic marks can propagate across generations, with certain gene expression patterns becoming dominant. Parent-of-origin effects show how one parent's genes can dominate expression. **Structural analogy**: Absorption in SAEs may reflect a form of "parent-of-origin" dominance where features from more frequent co-occurrence patterns dominate the representation, suppressing alternatives.

14. **Modular and Granular Architecture in Biological Networks** — Biological networks often show modular architecture with hierarchical organization. Modules that are highly connected tend to dominate network dynamics. **Structural analogy**: Absorbing latents in SAEs are like hub modules that dominate information flow, suppressing specialized modules.

#### Information Theory

15. **Rate-Distortion Theory and Compression Limits** — Rate-distortion theory quantifies the minimum bits needed to represent information within a fidelity constraint. Hierarchical compression introduces quantization error that can cascade across levels. **Structural analogy**: SAE absorption represents a fundamental rate-distortion trade-off: the SAE must compress hierarchical features into a fixed-rate bottleneck, causing parent features to "quantize away" child features.

16. **Bits-Back Coding and Inverse Coding** — Bits-back coding exploits the structure of probability distributions to encode data efficiently. The decoder uses prior knowledge to "guess" missing information. **Structural analogy**: When the SAE decoder reconstructs activations, it uses learned priors (feature co-occurrence patterns). Parent features serve as strong priors that can "predict away" child feature variance, causing absorption.

17. **Minimum Description Length (MDL) Principle** — MDL states that the best model is the one that minimizes the combined length of model and data encoding. Hierarchical models that are too complex fail MDL. **Structural analogy**: SAEs face an MDL trade-off: maintaining separate parent and child features increases description length; absorption reduces description length but loses information about child features.

### Cross-Disciplinary Gaps

Where transplants haven't been tried yet:

1. **Statistical physics phase transition analysis of SAE training dynamics** — No work has formally analyzed SAE absorption as a phase transition phenomenon, connecting dictionary size, sparsity, and feature hierarchy depth to critical thresholds.

2. **Renormalization group theory for SAE feature hierarchy** — While the "Toy Models of Superposition" paper draws loosely on physics analogies, no rigorous RG framework has been applied to understand how SAE features at different scales relate.

3. **Neuroplasticity models for SAE training trajectories** — The analogy between neuroplasticity critical periods and SAE feature selection during training has not been formalized.

4. **Rate-distortion theory for absorption limits** — Information-theoretic bounds on absorption rates given dictionary size, sparsity, and feature co-occurrence statistics have not been derived.

---

## Phase 2: Initial Candidates

### Candidate A: Phase Transition Theory for Absorption (from Statistical Physics)

- **Source principle**: Phase transitions in disordered systems (spin glasses, random matrices) exhibit sharp transitions in behavior at critical parameter values. The system suddenly switches from one phase to another.
- **Structural correspondence**: SAE absorption exhibits threshold behavior: below a critical sparsity-to-hierarchy-depth ratio, features are well-separated; above it, absorption dominates. This is mathematically analogous to the Anderson localization transition or percolation threshold.
- **Hypothesis**: There exists a critical phase boundary in SAE parameter space (dictated by the ratio of active features to dictionary size and the depth of feature hierarchy) where absorption transitions from rare to ubiquitous. This boundary can be characterized by an order parameter similar to magnetization in ferromagnets.
- **Why not just a metaphor**: The mathematical structure of mean-field spin glasses (Parisi solution) involves replica symmetry breaking and ultrametric organization of states. This precisely maps to the hierarchical organization of absorbed features, where parent-child relationships follow an ultrametric structure (as observed in the absorption graph analysis from E2 showing Layer 6 has the most fragmented graph with 9 components).
- **Novelty estimate**: 8/10 — No existing work applies statistical physics phase transition theory to characterize absorption thresholds in SAEs.

### Candidate B: Hierarchical Predictive Coding (from Neuroscience)

- **Source principle**: Predictive coding posits that the brain maintains a hierarchical generative model where top-down predictions are compared with bottom-up prediction errors. At each level, prediction errors are passed up when they cannot be explained by existing representations.
- **Structural correspondence**: SAE encoding is analogous to bottom-up prediction error transmission; the decoder generates top-down predictions. Absorption occurs when the decoder's top-down predictions (parent features) explain away the prediction error that should be attributed to child features.
- **Hypothesis**: Features that are highly predictable from other features (high conditional dependency) will be absorbed because their prediction error is already accounted for by the parent feature. This predicts that absorption is proportional to the conditional mutual information I(child | parent).
- **Why not just a metaphor**: The mathematical framework of predictive coding (variational inference formulation) is formally identical to the ELBO objective used in training SAEs with a Gaussian prior on features. The "explaining away" effect in Bayesian inference maps directly to absorption.
- **Novelty estimate**: 6/10 — The connection between predictive coding and SAE training has been hinted at in theoretical discussions but never formalized for absorption analysis.

### Candidate C: Rate-Distortion Bounds for Hierarchical Features (from Information Theory)

- **Source principle**: Rate-distortion theory provides lower bounds on the bits needed to represent a source within a distortion tolerance. For hierarchical sources, the rate required at each level depends on the conditional entropy given the higher level.
- **Structural correspondence**: SAE absorption represents a distortion trade-off: when the bottleneck rate is insufficient to separately encode parent and child features, the optimal solution routes information through the parent channel, causing absorption.
- **Hypothesis**: The absorption rate can be lower-bounded by the conditional entropy H(child | parent) relative to the bottleneck capacity. When H(child | parent) exceeds the available capacity per feature, absorption is inevitable.
- **Why not just a metaphor**: This is a precise information-theoretic bound, not a qualitative analogy. The same formal structure that governs compression limits in communication systems applies to SAE feature representation.
- **Novelty estimate**: 7/10 — Information-theoretic analysis of SAE limits exists (Cui et al., ICLR 2026), but specific application to absorption via conditional entropy bounds is novel.

---

## Phase 3: Self-Critique

### Against Candidate A (Phase Transition Theory)

- **Shallow analogy attack**: Is the correspondence really structural, or are you just mapping vocabulary? Phase transitions in physical systems involve thermodynamic limits (infinite system size) that SAEs do not satisfy. However, the mathematical framework of order parameters and critical exponents can be applied to finite systems as finite-size scaling.
- **Scale mismatch attack**: SAE dictionaries (10K-130K latents) are far smaller than Avogadro's number of particles in physical systems. But finite-size scaling theory addresses this: critical phenomena manifest at smaller scales with modified exponents.
- **Prior transplant check**: No existing work has explicitly applied phase transition theory to SAE absorption. The "On the Limits of Sparse Autoencoders" paper (Cui et al.) provides theoretical limits but not a phase transition analysis.
- **Testability attack**: Can we design an experiment that distinguishes "absorption due to phase transition" from "absorption due to training dynamics"? The prediction is a sharp threshold behavior: varying the sparsity parameter should show a sudden onset of absorption rather than a gradual increase.
- **Verdict**: STRONG — The phase transition framework makes specific, testable predictions about threshold behavior that are distinct from other theories.

### Against Candidate B (Hierarchical Predictive Coding)

- **Shallow analogy attack**: Predictive coding is a theory of brain function, not a mathematical theorem. Different neuroscientists disagree on its precise formulation. The mapping to SAE training may be too loose.
- **Scale mismatch attack**: Predictive coding was developed for biological neural networks with millions of neurons and billions of synapses. SAE dictionaries have thousands of latents — an entirely different scale regime.
- **Prior transplant check**: The connection between predictive coding and SAEs has been discussed informally (e.g., in the SAELens community). This is not a novel transplant.
- **Testability attack**: The prediction that I(child | parent) predicts absorption is testable with existing metrics. This would require computing conditional mutual information between feature activations, which is feasible with activation datasets.
- **Verdict**: MODERATE — The prediction is testable, but the prior transplant check shows this connection has been discussed. The novelty is in formalizing and testing the specific prediction.

### Against Candidate C (Rate-Distortion Bounds)

- **Shallow analogy attack**: Rate-distortion theory assumes ergodic sources and i.i.d. representations. SAE features exhibit complex dependencies that violate these assumptions.
- **Scale mismatch attack**: Classical rate-distortion theory operates on bit streams; SAE features are continuous-valued activations. The mapping requires adaptation to the continuous domain.
- **Prior transplant check**: The "On the Limits of SAEs" paper (Cui et al.) already provides an information-theoretic analysis. Our contribution would be a specific application to hierarchical features, which may not be sufficiently novel.
- **Testability attack**: Computing H(child | parent) from empirical activations is straightforward. However, connecting this to absorption requires a model of how the bottleneck constrains representation, which may require additional assumptions.
- **Verdict**: MODERATE — The theoretical framework is sound, but the prior work (Cui et al.) significantly overlaps. The specific application to absorption via conditional entropy may be publishable but requires careful framing.

---

## Phase 4: Refinement

### Dropped

- **Candidate B (Hierarchical Predictive Coding)**: While testable, the informal prior transplant reduces novelty. The prediction I(child | parent) is worth investigating but likely not sufficient for a standalone contribution.

### Strengthened

- **Candidate A (Phase Transition Theory)**: The observation from E2 that Layer 6 is a "hotspot" with the highest absorption rate (0.549%) and most fragmented graph (9 components) is consistent with a critical point phenomenon where multiple competing states (feature clusters) coexist. This suggests the system is near a phase boundary at this layer depth.

- **Candidate C (Rate-Distortion Bounds)**: The E5 finding that absorbed features have significantly lower coefficient of variation (CV: 1.07 vs 1.46, p=0.005) is consistent with the rate-distortion prediction: absorbed features are more "compressible" (lower variance) because they are predictable from parent features. This could be formalized as H(activation | parent) being lower for absorbed features.

### Formalized Mappings

**Phase Transition Framework**:
- Order parameter m = mean absorption score across candidate pairs
- Control parameter λ = (sparsity penalty) / (feature hierarchy depth)
- Critical point λ_c: at λ < λ_c, m ≈ 0 (no absorption); at λ > λ_c, m > 0 (absorption emerges)
- Prediction: absorption onset is sharp, not gradual

**Rate-Distortion Framework**:
- Source coding rate R = log(dictionary_size) per feature
- Distortion D = reconstruction error
- Absorption rate A = fraction of child features with absorption_score > threshold
- Bound: A ≥ 1 - H(child | parent) / R (simplified)

### Selected Front-Runner

**Candidate A: Phase Transition Theory for Absorption** — This offers the most novel framing, makes the sharpest predictions (sudden onset vs gradual increase), and the Layer 6 hotspot finding is consistent with critical behavior where multiple quasi-stable states coexist.

---

## Phase 5: Final Proposal

### Title

**"Absorption as a Phase Transition: Critical Behavior of Feature Hierarchy Encoding in Sparse Autoencoders"**

### Source Principle

Phase transitions in disordered statistical physics systems (spin glasses, random energy models) exhibit sharp transitions between phases at critical parameter values. Below the critical point, the system occupies a disordered phase with no long-range order; above it, ordered phases emerge with specific symmetry breaking patterns. The transition is characterized by an order parameter that changes discontinuously at the critical point.

### Structural Correspondence

SAE feature absorption can be mapped to a phase transition:

| Physics Concept | SAE Concept |
|-----------------|-------------|
| Order parameter (magnetization) | Mean absorption score across candidate pairs |
| Control parameter (temperature, field) | Sparsity penalty λ / feature hierarchy depth |
| Critical point λ_c | Threshold where absorption onset becomes widespread |
| Spontaneous symmetry breaking | Emergence of dominant parent features |
| Domain formation | Absorption clusters (graph components) |
| Free energy landscape | Loss landscape of SAE objective |

The Layer 6 "hotspot" finding (0.549% absorption rate, 9 graph components) maps to a system near its critical point where multiple competing domains (absorption clusters) coexist.

### Hypothesis

SAE absorption exhibits threshold behavior analogous to phase transitions:

1. **H1 (Critical Sparsity)**: There exists a critical sparsity ratio λ_c such that absorption rates are near zero below λ_c and increase sharply above λ_c.

2. **H2 (Finite-Size Scaling)**: The sharpness of the absorption onset scales with dictionary size according to finite-size scaling laws: the transition width δλ ∝ N^(-1/ν) where N is the number of latents.

3. **H3 (Layer Depth as Tuning)**: Layer depth acts as a "temperature" parameter: early layers (low temperature, λ << λ_c) show no absorption; mid layers (λ ≈ λ_c) show maximum absorption heterogeneity; late layers (high temperature, λ >> λ_c) show saturated absorption.

### Method

**Experimental Design**:
1. Train a series of SAEs on GPT-2 residual stream activations at Layer 6 with varying L1 sparsity penalties (λ ∈ [1e-5, 1e-2])
2. Measure absorption score distribution for each λ value
3. Compute mean absorption score m(λ) and susceptibility χ = dm/dλ
4. Identify the critical point λ_c where χ is maximized
5. Fit finite-size scaling to estimate λ_c and critical exponent ν
6. Validate with cross-layer experiments (layers 0, 3, 6, 9, 11) treating depth as an additional control parameter

**Analytical Tools**:
- AbsorptIon Detection: Use the revised scoring formula from E4 (removing degenerate suppression_ratio, focusing on decoder_cosine * freq_ratio)
- Graph Analysis: Absorption graph component structure (number of components, mean edge weight) as order parameter
- Statistical Analysis: Compute susceptibility and critical exponents using finite-size scaling theory

### Diagnostic Experiment

**The Key Test**: If absorption is truly a phase transition, then:
- Varying sparsity λ should produce a sharp onset (not gradual increase)
- The absorption rate curve should be fit by a universal scaling function
- Different layer depths should map to different points on the phase diagram

**Success Criterion**: If we observe:
- A clear threshold λ_c where absorption jumps from near-0 to measurable (>0.1)
- Scaling collapse of m(λ) curves across different dictionary sizes
- Layer 6 lying precisely at λ ≈ λ_c (explaining the hotspot)

Then the phase transition framework is load-bearing, not decorative.

### Experimental Plan

| Experiment | Duration | Model | SAE Config | Key Measurement |
|------------|----------|-------|------------|-----------------|
| E6: Sparsity sweep | 30 min | GPT-2-small | Layer 6, 16k latents | m(λ) curve, find λ_c |
| E7: Finite-size scaling | 40 min | GPT-2-small | Layers 6, 8k/16k/32k | Scaling collapse fit |
| E8: Layer phase diagram | 45 min | GPT-2-small | Layers 0,3,6,9,11 | Map layer to effective λ |
| E9: Cross-model validation | 45 min | Pythia-70M | Layer 6 | Test λ_c generalization |

Total pilot budget: ~3 hours (within 1-hour-per-task constraint)

### Risk Assessment

1. **Risk: The transition is gradual, not sharp** — If absorption increases smoothly with λ, the phase transition framework is wrong. Mitigation: Fall back to Candidate C (rate-distortion) which predicts smooth increase.

2. **Risk: Layer 6 hotspot is a coincidence** — If the hotspot doesn't replicate in other models, the critical point interpretation fails. Mitigation: Cross-model validation in E9.

3. **Risk: Dictionary size is too small for finite-size scaling** — With 16K latents, finite-size effects may be too large. Mitigation: Use 8K/16K/32K comparison in E7.

4. **Risk: Revised scoring formula still doesn't capture absorption** — If the decoder_cosine * freq_ratio formula still produces degenerate discrimination, we cannot measure m(λ). Mitigation: Validate formula on E2 data before running E6.

### Novelty Claim

This is the **first application of statistical physics phase transition theory to SAE absorption**. While Cui et al. (ICLR 2026) provide information-theoretic limits, they do not characterize the dynamical onset of absorption as a phase transition with critical exponents and scaling laws. The phase transition framing offers:

1. **Predictive power**: Sharp threshold prediction that can be experimentally verified
2. **Unified explanation**: The Layer 6 hotspot becomes explained as the system being near criticality
3. **Design guidelines**: Operating SAEs at λ << λ_c avoids absorption; λ >> λ_c ensures complete absorption
4. **Connection to known physics**: Methods from disordered systems (replica theory, renormalization group) become available for analysis

### References (Cross-Disciplinary)

- Mezard, Montanari, "Information, Physics, and Computation" (statistical physics of constraint satisfaction)
- Parisi, "Mean Field Theory of Spin Glasses" (replica solution)
- Fisher, "Collective运动 in Random Environments" (random energy model)
- Nishimori, "Statistical Physics of Learning" (phase transitions in neural networks)
- Engel, Van den Broeck, "Statistical Mechanics of Learning" (analytical learning theory)

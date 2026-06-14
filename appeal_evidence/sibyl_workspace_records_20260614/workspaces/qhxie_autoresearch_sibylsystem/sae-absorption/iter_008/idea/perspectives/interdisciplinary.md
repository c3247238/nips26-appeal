# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Olshausen & Field (1996, 1997)** -- Sparse coding with an overcomplete basis set: showed that V1 simple-cell receptive fields emerge from sparsity optimization on natural images. The foundational neuroscience theory behind SAEs. Key principle: overcomplete dictionaries + sparsity penalty recover localized, oriented features.

2. **Rozell, Johnson, Baraniuk & Olshausen (2008)** -- Locally Competitive Algorithm (LCA): solves sparse coding via a dynamical system of neuron-like elements with lateral inhibition. Active neurons suppress competitors proportional to decoder overlap. The membrane-potential dynamics include: (a) input drive, (b) leak, and (c) lateral inhibition from supra-threshold neurons. Key mechanism: *explaining away* -- once a neuron activates, it actively suppresses neurons with overlapping receptive fields.

3. **Paiton, Frye, Lundquist et al. (2020)** -- Selectivity and robustness of sparse coding networks: showed that population nonlinearities from lateral inhibition improve selectivity (hyperselectivity via curved iso-response surfaces) and adversarial robustness. Key insight: competitive inhibition shapes individual neurons' response properties beyond what feedforward processing alone achieves.

4. **Friston (2005, 2008)** -- Free energy principle and predictive coding: perception as hierarchical variational inference minimizing prediction error. The brain maintains a generative model at multiple scales; higher levels predict lower levels, and prediction errors propagate upward. Key structure: hierarchical message passing where abstract representations "explain away" sensory detail.

5. **Fletcher (1940) / psychoacoustic masking** -- Simultaneous auditory masking: a loud tone raises the detection threshold of nearby frequencies within the same critical band. The masker activates cochlear receptors that would otherwise respond to the masked signal. Key property: *asymmetric* masking -- low frequencies mask high frequencies more effectively than vice versa (upward spread of masking).

6. **Chalk, Marre & Tkacik (2018)** -- Toward a unified theory of efficient, predictive, and sparse coding: connects sparse coding to predictive coding under a single optimization framework, showing both are instances of efficient neural coding under different constraints.

#### Physics / Information Theory

7. **Krzakala, Zdeborova et al. (2013)** -- Phase transitions in dictionary learning: replica-method analysis of sparse dictionary learning reveals three phases -- impossible inference, possible-but-hard, and easy inference -- separated by sharp phase transitions as a function of sparsity and dictionary size. Key tool: approximate message passing (AMP) achieves optimal performance at the algorithmic threshold.

8. **Elhage et al. (2022) / Chen et al. (2023)** -- Phase transitions in toy models of superposition: superposition exhibits phase changes governed by feature sparsity and importance; singular learning theory (SLT) shows these are genuine phase transitions where k-gon critical points determine training dynamics. "Energy level"-like jumps during training.

9. **Ayonrinde, Pearce & Sharkey (2024)** -- MDL-SAE: Minimum Description Length perspective on SAEs. MDL incentivizes hierarchical features and provides theoretical motivation for sparsity penalties. The rate-distortion framework shows SAEs trade off code length (sparsity) against reconstruction distortion, with absorption being a compression strategy that reduces total description length.

10. **Rate-distortion theory (Shannon 1959, Berger 1971)** -- Establishes fundamental limits on lossy compression. A codebook with K codewords optimally partitions the source space; hierarchical codebook structures (residual vector quantization) allocate coarse codes first, then refine. Key principle: optimal codebooks exhibit *codebook collapse* when capacity is limited -- frequent codewords absorb nearby rare codewords.

11. **Arnold et al. (2024)** -- Phase transitions in LLM output distributions: draws analogy between token distributions and 1D Ising models with long-range interactions. Tokens as spins, attention as coupling. Key connection: spontaneous symmetry breaking in feature representation.

#### Biology / Evolution / Immunology

12. **Gause (1934) / Competitive Exclusion Principle** -- Two species competing for the same limited resource cannot stably coexist; one will dominate. Formalized by Lotka-Volterra competition equations where coexistence requires intraspecific competition > interspecific competition. Key result: when two species' niches overlap completely, the one with even a slight fitness advantage drives the other to extinction.

13. **Hardin (1960) / Hutchinson niche theory** -- The n-dimensional niche hypervolume; fundamental vs. realized niche. Competition compresses realized niches. Niche differentiation (partitioning) enables coexistence. Key insight: limiting similarity -- there is a minimum niche separation below which coexistence is impossible.

14. **Francis (1960) / Original Antigenic Sin (OAS)** -- The immune system preferentially recalls memory B cells from the first pathogen encounter, suppressing naive B cell responses to new but related epitopes. Memory B cells outcompete naive B cells for antigen capture due to higher affinity, faster proliferation, and pre-existing transcriptional priming. Key mechanism: the dominant (first-encountered) response *absorbs* the capacity that should serve novel responses.

15. **Immunodominance hierarchy** -- The immune response focuses on only a few of many potential epitopes. Dominant lineages suppress subdominant lineages by consuming pathogen below the threshold needed to trigger weaker responses. This is a quantitative hierarchy, not binary.

16. **Zipf's law / preferential attachment (Yule-Simon, Barabasi-Albert)** -- Power-law frequency distributions arise from rich-get-richer dynamics. Frequent items acquire more "mass" over time, suppressing rare items. SynthSAEBench explicitly uses Zipfian firing distributions for realistic SAE evaluation.

### Cross-Disciplinary Gaps

The following transplants have NOT been explored in the SAE feature absorption context:

1. **Ecological competitive exclusion as a formal model for absorption**: No paper maps the Lotka-Volterra competition dynamics to SAE feature dynamics. The structural correspondence (species = features, niche = activation subspace, carrying capacity = reconstruction budget) has not been formalized.

2. **Original Antigenic Sin as a model for feature absorption under sequential/hierarchical training**: The OAS mechanism (memory dominates naive, suppressing novel responses) is structurally identical to absorption (specific features dominate general features, suppressing their activation). This analogy has not been drawn.

3. **Psychoacoustic masking as a perceptual model for absorption**: The simultaneous masking effect (loud tones suppress nearby quiet tones within a critical band) has not been mapped to SAE feature dynamics, despite the structural similarity (dominant features "mask" overlapping weaker features within a representational bandwidth).

4. **Phase-transition analysis of the absorption-to-non-absorption boundary**: While phase transitions in superposition have been studied, no one has identified the precise phase boundary between the absorption and non-absorption regimes as a function of (width, L0, hierarchy depth).

---

## Phase 2: Initial Candidates

### Candidate A: Ecological Niche Theory -- Competitive Exclusion of Features (from Ecology / Evolutionary Biology)

- **Source principle**: Gause's Competitive Exclusion Principle and Lotka-Volterra competition dynamics. Two species occupying the same ecological niche cannot stably coexist -- the fitter species excludes the other. Coexistence requires sufficient niche differentiation (limiting similarity). Disturbance and environmental heterogeneity can prevent exclusion.

- **Structural correspondence**:
  | Ecology | SAE |
  |---------|-----|
  | Species | SAE latent feature |
  | Ecological niche (resource use) | Activation subspace (set of inputs a feature fires on) |
  | Resource (e.g., food, sunlight) | Reconstruction budget (variance to explain) |
  | Population size | Feature activation magnitude |
  | Carrying capacity K | Maximum activation / decoder norm |
  | Competition coefficient alpha_ij | Decoder cosine similarity between features i,j |
  | Competitive exclusion | Feature absorption |
  | Niche differentiation | Feature splitting / orthogonalization |
  | R* (minimum resource for survival) | Minimum activation threshold (JumpReLU theta, TopK rank) |
  | Environmental disturbance | Training data heterogeneity / masking regularization |

  The Lotka-Volterra equations for two competing species are:
  ```
  dN1/dt = r1 * N1 * (K1 - N1 - alpha12 * N2) / K1
  dN2/dt = r2 * N2 * (K2 - N2 - alpha21 * N1) / K2
  ```
  In the SAE context, N_i is the activation of feature i, K_i is the maximum activation permitted by the sparsity budget, r_i is the learning rate / gradient magnitude, and alpha_ij measures the competition between features (decoder overlap). When alpha12 > K1/K2 and alpha21 > K2/K1, one species is excluded -- the "absorbed" feature goes to zero activation.

- **Hypothesis**: Absorption occurs when and only when the "niche overlap" (decoder cosine similarity) between a parent feature and child feature exceeds a critical threshold relative to their "carrying capacities" (activation budgets). Specifically, the absorption probability for a parent-child pair should be predictable from the Lotka-Volterra coexistence criterion: absorption happens when alpha_parent,child > K_parent / K_child, where alpha is measured by decoder similarity and K by relative frequency-weighted activation magnitude.

- **Why not just a metaphor**: The correspondence preserves the core mathematical structure. The Lotka-Volterra equations describe the dynamics of competing entities sharing a limited resource, with outcomes determined by the ratio of inter- to intra-specific competition coefficients. SAE features during training are competing entities sharing a limited reconstruction budget, with outcomes (absorption vs. coexistence) determined by the ratio of inter-feature competition (decoder overlap) to intra-feature "self-regulation" (sparsity penalty). The extinction criterion translates directly to a testable inequality.

- **Novelty estimate**: 8/10

### Candidate B: Original Antigenic Sin -- Memory Dominance Suppresses Novel Responses (from Immunology)

- **Source principle**: Original Antigenic Sin (OAS). When the immune system encounters a new pathogen variant, pre-existing memory B cells (from a prior encounter with a related pathogen) are recalled preferentially and suppress naive B cell activation against the new epitopes. The memory cells are faster, higher-affinity, and pre-primed, so they outcompete naive cells for limited antigen. This creates "blind spots" in the immune repertoire -- the system fails to develop responses to novel epitopes that co-occur with familiar ones.

- **Structural correspondence**:
  | Immunology (OAS) | SAE (Feature Absorption) |
  |-------------------|--------------------------|
  | Memory B cell (experienced) | Specific/child feature (e.g., "snake" feature) |
  | Naive B cell (inexperienced) | General/parent feature (e.g., "starts with s" feature) |
  | Antigen (pathogen surface) | Input activation (token representation) |
  | Epitope (specific binding site) | Activation subspace direction |
  | Antibody affinity | Decoder-input cosine similarity |
  | Memory B cell recall speed | Feature activation magnitude (post-training convergence) |
  | Naive B cell suppression | Parent feature absorption (false negative) |
  | Serum antibody feedback | Reconstruction residual signal |
  | Germinal center access | Gradient updates during training |
  | Immunodominance hierarchy | Feature hierarchy (general -> specific) |
  | "Primary addiction" | Over-specialization of SAE latents |
  | Variant boosting to overcome OAS | Masking regularization / Matryoshka training |

  The key structural identity: in OAS, memory B cells have (1) higher antigen affinity, (2) faster activation, and (3) serum antibody feedback that blocks naive cell access to antigen. In SAE absorption, specific features have (1) higher cosine similarity to co-occurring inputs, (2) larger activation magnitudes after training converges, and (3) their activation reduces the reconstruction residual, eliminating the gradient signal that would have activated the general feature.

- **Hypothesis**: Feature absorption follows the same "primary addiction" dynamics as OAS. Once a specific feature latent is established during training, it monopolizes the gradient signal for inputs in its activation domain, preventing the corresponding general feature from developing or maintaining its response on overlapping inputs. The "absorption rate" should correlate with the "affinity gap" between specific and general features (analogous to the affinity advantage of memory over naive B cells). Moreover, the immunological insight that *variant boosting* overcomes OAS maps directly to Matryoshka training (training with different dictionary sizes forces general features to activate without specific features present, analogous to presenting variant antigens that memory cells cannot bind).

- **Why not just a metaphor**: The formal mechanism is identical: two competing response systems (memory vs. naive / specific vs. general features) share a limited resource (antigen / reconstruction budget), and the dominant system's rapid consumption of the resource starves the subordinate system. The immunological literature provides quantitative models of this competition (germinal center dynamics, serum feedback models) that can be adapted to predict SAE training dynamics. Moreover, the immunological solution (variant boosting) has already been independently reinvented in the SAE literature (Matryoshka SAE / masked regularization), validating the structural correspondence.

- **Novelty estimate**: 9/10

### Candidate C: Psychoacoustic Simultaneous Masking -- Critical Bandwidth of Feature Interference (from Auditory Neuroscience / Signal Processing)

- **Source principle**: In psychoacoustics, simultaneous auditory masking occurs when a strong tone raises the detection threshold of nearby frequencies within the same critical band. The masker activates cochlear receptors that would otherwise respond to the masked signal. Key properties: (1) masking is strongest within the critical bandwidth, (2) masking is asymmetric -- low frequencies mask high frequencies more effectively than vice versa (upward spread of masking), (3) masking strength grows with masker amplitude, (4) signals outside the critical band are largely unaffected.

- **Structural correspondence**:
  | Psychoacoustics | SAE (Feature Absorption) |
  |-----------------|--------------------------|
  | Auditory tone | SAE feature activation |
  | Frequency | Position in feature space (decoder direction) |
  | Amplitude / loudness | Activation magnitude |
  | Critical band | "Absorption band" -- set of features with high decoder cosine similarity |
  | Simultaneous masking | Feature absorption (strong feature suppresses weak overlapping feature) |
  | Masking threshold | Minimum activation needed to "survive" in the presence of a competing feature |
  | Upward spread of masking (low masks high) | Specific features absorb general features (not vice versa) |
  | Critical bandwidth | Decoder cosine similarity radius within which absorption occurs |
  | Signal-to-mask ratio | Ratio of general feature activation to specific feature activation |
  | Perceptual coding (MP3) | Sparsity optimization (discard sub-threshold features) |

  The formal structure: in audio coding (MP3, AAC), the masking model identifies which spectral components fall below the masking threshold set by dominant tones within each critical band, then discards them to achieve compression. In SAE training, the sparsity objective identifies which features fall below the activation threshold set by dominant co-occurring features, then suppresses them to achieve the sparsity target. Both are instances of perceptual/lossy compression where dominant components suppress detectable but sub-threshold components within a local bandwidth.

- **Hypothesis**: Feature absorption in SAEs follows a "masking model" analogous to psychoacoustic masking. For each feature, there exists a "critical absorption band" in decoder space (measured by cosine similarity) within which competing features are susceptible to absorption. The absorption probability of a parent feature by a child feature is determined by: (1) the "signal-to-mask ratio" (parent activation magnitude / child activation magnitude on co-occurring inputs), (2) the "critical bandwidth" (decoder cosine similarity between them), and (3) the sparsity penalty (analogous to the quantization bitrate in audio coding). This predicts that absorption is a *local* phenomenon in feature space: features whose decoders are far apart (outside the critical band) should not absorb each other, regardless of co-occurrence.

- **Why not just a metaphor**: The masking model provides a *quantitative* prediction framework. In audio coding, the masking threshold is computed as a function of masker frequency, amplitude, and the critical band filter shape. The analogous "absorption threshold" in SAEs would be computed as a function of the dominant feature's activation magnitude, the decoder similarity, and the sparsity penalty. This is not a vague analogy -- it is a specific functional form that can be fit to data. Moreover, the asymmetry property (upward spread: low frequencies mask high more than vice versa) maps to the observed asymmetry of absorption (specific features absorb general features, not vice versa), providing a non-trivial structural prediction.

- **Novelty estimate**: 7/10

---

## Phase 3: Self-Critique

### Against Candidate A: Ecological Competitive Exclusion

- **Shallow analogy attack**: The Lotka-Volterra equations describe population dynamics of organisms over generations with birth/death processes. SAE features during training undergo gradient-based optimization, not population dynamics. The correspondence between "population size" and "activation magnitude" is mathematically defensible (both are non-negative quantities that grow/shrink based on competition for shared resources), but the dynamics differ: Lotka-Volterra is continuous-time ODE with multiplicative growth; SAE training is stochastic gradient descent. A domain expert in ecology might note that ecological competition operates over generations with random variation, while SAE training is deterministic gradient flow. HOWEVER: the *equilibrium analysis* of Lotka-Volterra (which species survives at steady state) does not depend on the dynamics -- it depends only on the competition coefficients and carrying capacities. The SAE analog (which feature survives after training converges) similarly depends on the decoder overlap and sparsity budget, not on the specific optimization trajectory. The equilibrium mapping is structurally sound.

- **Scale mismatch attack**: Ecological systems have 10^0 to 10^6 species competing in continuous environments. SAEs have 10^3 to 10^6 latents competing in high-dimensional activation spaces. The scale is comparable. The key structural property (pairwise competition via resource overlap) holds at both scales. No obvious scale mismatch.

- **Prior transplant check**: Searched for "Lotka-Volterra sparse autoencoder" and "competitive exclusion sparse coding" on arXiv. No results found. The ecology-to-sparse-coding analogy has not been explored. HOWEVER, the general idea of competitive dynamics in neural coding is well-established via the Locally Competitive Algorithm (Rozell et al., 2008). The novelty here is specifically mapping the Lotka-Volterra *coexistence criterion* to the absorption/non-absorption boundary.

- **Testability attack**: The key prediction is testable: the Lotka-Volterra coexistence criterion (alpha_ij < K_i/K_j for both i,j) should predict whether a specific parent-child feature pair exhibits absorption. The competition coefficient alpha_ij is measurable (decoder cosine similarity), and the carrying capacities K_i, K_j are measurable (frequency-weighted mean activation). One can compute this criterion for all feature pairs in an existing SAE and correlate with observed absorption rates. The diagnostic experiment is clean.

- **Verdict**: **STRONG**. The equilibrium analysis provides a testable, quantitative prediction that goes beyond metaphor. The main weakness is that the dynamics differ, but the equilibrium prediction is dynamics-independent.

### Against Candidate B: Original Antigenic Sin

- **Shallow analogy attack**: The OAS mechanism involves two-stage temporal dynamics (first encounter establishes memory, second encounter triggers recall), while SAE training is a single continuous optimization. A domain immunologist might argue that OAS is fundamentally about *temporal ordering* (which pathogen you see first matters), while SAE training does not have this sequential structure -- all training data is seen in random order. HOWEVER: the key mechanism is not temporal ordering but *asymmetric competition* -- one system (memory/specific) has a structural advantage (higher affinity/cosine similarity) over the other (naive/general), and this advantage is self-reinforcing through feedback (serum antibodies/reconstruction residual reduction). This self-reinforcing asymmetric competition operates in SAE training regardless of temporal ordering.

- **Scale mismatch attack**: The immune system has ~10^9 B cell clones competing for ~10^3-10^6 distinct epitopes. SAEs have 10^3-10^6 latents competing for representation of 10^3-10^6 features. The population ratios are different, but the competition mechanism (faster/higher-affinity responder monopolizes limited resource) is scale-invariant. The main concern is that OAS involves clonal *expansion* (cells divide), while SAE features do not multiply. But the relevant quantity is activation magnitude, not feature count, and activation magnitudes do grow/shrink during training.

- **Prior transplant check**: Searched for "original antigenic sin neural network" and "immune memory feature learning" on arXiv. Found the Artificial Immune System (AIS) literature from the 2000s (de Castro & Timmis, 2002) which transplanted immune concepts to optimization algorithms, but NOT to SAE feature absorption specifically. The OAS-to-absorption mapping appears genuinely novel.

- **Testability attack**: The key prediction is that absorption severity correlates with the "affinity gap" (difference in cosine similarity to co-occurring inputs between specific and general features). This is directly measurable. The diagnostic experiment: verify that Matryoshka training overcomes absorption via the same mechanism as variant boosting overcomes OAS (forcing the subordinate system to activate in the absence of the dominant system). This is testable by comparing absorption rates in Matryoshka vs. standard SAEs and verifying the mechanism matches the immunological prediction.

  The main weakness of testability: the OAS analogy provides *qualitative* predictions (absorption should increase with affinity gap, Matryoshka should help) that are already known from existing SAE research. The analogy is post-hoc explanatory rather than generating truly novel predictions. To strengthen: the immunological model predicts that *serum feedback* (analogous to reconstruction residual reduction) is a separate, additive mechanism of suppression beyond simple competition. This predicts that absorption should be *stronger* than pure decoder-overlap competition would suggest, because the residual reduction provides an additional suppressive signal. This is a novel, testable prediction.

- **Verdict**: **MODERATE-STRONG**. The analogy is structurally deep and the independent reinvention of "variant boosting" (Matryoshka) validates the correspondence. But the immunological model's novel predictions beyond what is already known are limited. The strongest novel prediction is the dual-mechanism hypothesis (competition + feedback suppression).

### Against Candidate C: Psychoacoustic Masking

- **Shallow analogy attack**: Psychoacoustic masking is a property of cochlear mechanics (basilar membrane resonance, hair cell saturation) and early auditory neural processing. SAE feature dynamics are a property of gradient-based optimization in high-dimensional spaces. The *physical substrate* is completely different. A psychoacoustician would note that masking is caused by basilar membrane mechanics (traveling wave envelope), not by abstract competition. HOWEVER: the *functional form* of masking (local suppression within a critical band, asymmetric, amplitude-dependent) is substrate-independent and arises in any system where: (a) representations are overcomplete and locally overlapping, (b) there is a limited capacity (bandwidth/sparsity), and (c) strong signals saturate local processing before weak signals can register. These conditions hold for SAE features.

- **Scale mismatch attack**: The auditory system has ~3,500 inner hair cells organized tonotopically along the basilar membrane, mapping to ~24 critical bands. SAEs have 10^3-10^6 latents with no tonotopic organization. The critical band structure in audition arises from the physical geometry of the cochlea. In SAEs, the "critical band" would need to be defined in decoder cosine similarity space, which is isotropic (no preferred direction). The asymmetry (upward spread of masking) maps to absorption asymmetry (specific absorbs general), but the auditory asymmetry has a physical cause (basilar membrane response) with no SAE analog. The asymmetry in SAEs has a different cause (frequency imbalance between parent and child features). This weakens the structural correspondence somewhat.

- **Prior transplant check**: Searched for "psychoacoustic masking neural network features" and "auditory masking sparse coding." Found work on perceptual loss functions using psychoacoustic models for audio compression, but NOT applied to SAE feature dynamics. The MP3 analogy (discard sub-threshold spectral components) has been noted informally in SAE discussions but never formalized.

- **Testability attack**: The critical bandwidth prediction is testable: define a "cosine similarity radius" and measure whether absorption is confined within that radius. The masking threshold prediction is also testable: for each feature pair, compute the signal-to-mask ratio and check if it predicts absorption. The asymmetry prediction (specific absorbs general more than vice versa) is already known but could be quantitatively compared to the upward-spread-of-masking curve shape.

  Main weakness: the functional form borrowed from psychoacoustics (masking threshold as a function of frequency separation and masker amplitude) has specific parameters calibrated to the human auditory system. There is no principled reason why SAE absorption should follow the *same* functional form. The analogy suggests a *class* of models (local suppression within a bandwidth) but does not give the specific functional form.

- **Verdict**: **MODERATE**. The analogy is suggestive and provides a useful framework (local suppression within a critical bandwidth), but the physical basis of auditory masking does not transfer, and the specific functional form is not preserved. The strongest contribution is the "critical bandwidth" concept -- that absorption should be localized in feature space.

---

## Phase 4: Refinement

### Dropped

**Candidate C (Psychoacoustic Masking)**: While the "critical bandwidth" concept is useful, the structural correspondence is weaker than the other two candidates. The physical substrate differs fundamentally, and the specific functional form of masking (basilar membrane mechanics) does not transfer. The useful insight (absorption is local in decoder similarity space) is a prediction that can be derived more rigorously from the Lotka-Volterra framework (Candidate A). I will incorporate the "critical bandwidth" concept as a supporting element in the front-runner rather than as a standalone proposal.

### Strengthened Survivors

**Candidate A (Ecological Niche Theory)**: Formalized the mapping further. The key insight is that the Lotka-Volterra *coexistence criterion* provides a closed-form, testable boundary between absorption and non-absorption:

**Coexistence (no absorption) iff**: alpha_pc < K_p / K_c AND alpha_cp < K_c / K_p

where:
- alpha_pc = competition effect of child on parent = f(cosine_similarity(d_p, d_c), co-occurrence frequency)
- alpha_cp = competition effect of parent on child = f(cosine_similarity(d_c, d_p), co-occurrence frequency)
- K_p = parent feature "carrying capacity" = activation budget proportional to feature frequency and decoder norm
- K_c = child feature "carrying capacity" = activation budget proportional to feature frequency and decoder norm

Since child features are more specific (higher cosine similarity to their activation inputs, higher per-token reconstruction contribution), they typically have alpha_pc > K_p/K_c (child strongly suppresses parent) while alpha_cp < K_c/K_p (parent weakly suppresses child). This is exactly the condition for *parent exclusion* -- i.e., absorption.

The "limiting similarity" result from ecology further predicts a *minimum decoder angle separation* below which coexistence is impossible, analogous to the Lotka-Volterra result that species must be sufficiently different to coexist.

**Candidate B (OAS / Immunological Memory Dominance)**: Refined to focus on the novel prediction: the *dual-mechanism hypothesis*. In OAS, naive B cell suppression operates through two independent mechanisms: (1) direct competition for antigen (memory cells bind antigen faster/tighter), and (2) serum antibody feedback (circulating antibodies from memory responses block naive cell access to antigen in germinal centers). The SAE analog:

1. **Direct competition**: Specific features have higher cosine similarity to co-occurring inputs, so they receive larger gradient updates and larger activations. This is the standard "explaining away" effect.
2. **Residual feedback suppression**: When specific features activate, they reduce the reconstruction residual. The reduced residual provides less gradient signal to the general feature encoder, independently of the decoder overlap. This is analogous to serum antibody feedback.

The dual-mechanism prediction is that absorption is *stronger* than decoder overlap alone would predict, because residual feedback provides an additive suppressive mechanism. This is testable: compare absorption rates predicted by decoder overlap alone vs. absorption rates predicted by decoder overlap + residual reduction.

### Selected Front-Runner: Candidate A (Ecological Niche Theory)

Selected because: (1) the structural correspondence is most precise and quantitative, (2) the coexistence criterion provides a novel, falsifiable boundary condition, (3) the Lotka-Volterra framework is mathematically mature and well-understood, and (4) it generates genuinely novel predictions not present in existing SAE absorption literature. The OAS analogy (Candidate B) is incorporated as a complementary interpretive lens that provides the "dual mechanism" prediction.

---

## Phase 5: Final Proposal

### Title

**Feature Absorption as Competitive Exclusion: An Ecological Niche Theory of Sparse Autoencoder Feature Dynamics**

### Source Principle

Gause's Competitive Exclusion Principle (1934), formalized by Lotka-Volterra competition equations. In ecology, two species competing for the same limited resource cannot stably coexist if their niche overlap exceeds a critical threshold relative to their carrying capacities. The fitter competitor excludes the other. Coexistence requires sufficient niche differentiation -- a result known as "limiting similarity." Environmental disturbance and spatial heterogeneity can promote coexistence by periodically resetting competitive hierarchies.

### Structural Correspondence

The formal mapping between ecological competition and SAE feature dynamics:

**State variables**: Species population N_i maps to feature activation magnitude a_i (non-negative, competing for shared resource).

**Resource**: Shared ecological resource (food, sunlight) maps to reconstruction budget (total variance to be explained, constrained by sparsity).

**Competition**: Inter-species competition coefficient alpha_ij maps to decoder cosine similarity cos(d_i, d_j) weighted by input co-occurrence probability P(i AND j active). This measures how much feature j's activation reduces the gradient/residual available to feature i.

**Carrying capacity**: Species carrying capacity K_i maps to the maximum activation a feature can sustain, determined by its frequency (how often it should fire), decoder norm, and the sparsity budget.

**Dynamics**: The Lotka-Volterra competition equations:

```
da_p/dt = r_p * a_p * (K_p - a_p - alpha_pc * a_c) / K_p
da_c/dt = r_c * a_c * (K_c - a_c - alpha_cp * a_p) / K_c
```

where p = parent (general) feature, c = child (specific) feature.

**Absorption criterion**: Parent feature p is absorbed by child feature c when:

```
alpha_pc > K_p / K_c   (child's competition on parent exceeds parent's relative capacity)
```

This typically holds because:
- alpha_pc is large: the child feature has high cosine similarity with the parent's activation subspace (since child inputs are a subset of parent inputs)
- K_p / K_c is small: the parent's effective carrying capacity relative to the child is reduced by the sparsity penalty (parent fires less frequently per-token than child on overlapping inputs)

**Coexistence (no absorption) criterion**:

```
alpha_pc < K_p / K_c  AND  alpha_cp < K_c / K_p
```

This corresponds to sufficient "niche differentiation" -- the parent and child features must have sufficiently different decoder directions AND sufficiently distinct activation patterns.

**Limiting similarity**: Below a critical decoder angle separation theta*, coexistence is impossible regardless of other parameters. This theta* is a function of the sparsity penalty lambda and the feature frequency ratio.

### Hypothesis

1. **Absorption boundary**: The probability that a parent feature is absorbed by a child feature is determined by the Lotka-Volterra coexistence criterion applied to their decoder overlap and activation statistics. Features with alpha_pc / (K_p/K_c) > 1 should be absorbed with high probability.

2. **Limiting similarity threshold**: There exists a minimum decoder cosine dissimilarity below which general-specific feature pairs cannot coexist in an SAE, regardless of SAE width or architecture. This threshold depends on sparsity penalty and feature frequency ratio.

3. **Width as niche space**: Wider SAEs provide more "niche space," reducing competition coefficients (alpha) and enabling coexistence of more feature pairs. This predicts a quantitative relationship between SAE width and absorption rate.

4. **Disturbance promotes coexistence**: Training interventions that disrupt competitive equilibria (masked regularization, Matryoshka training) reduce absorption, analogous to how ecological disturbance prevents competitive exclusion by periodically resetting dominance hierarchies.

5. **Dual-mechanism suppression** (from OAS analogy): Absorption is stronger than decoder overlap alone predicts, because reconstruction residual reduction provides an additional suppressive pathway (analogous to serum antibody feedback in immunology).

### Method

**Phase 1 -- Operationalize the ecological model (training-free analysis on existing SAEs)**:

For each SAE in Gemma Scope / SAEBench:
1. Identify parent-child feature pairs using the Chanin et al. first-letter absorption metric + extend to entity-type hierarchies
2. Compute ecological competition parameters:
   - alpha_pc: decoder cosine similarity * co-occurrence probability
   - alpha_cp: same, other direction
   - K_p, K_c: frequency-weighted mean activation magnitude * decoder norm
3. Compute the Lotka-Volterra coexistence criterion for each pair
4. Correlate the criterion with observed absorption (binary: absorbed vs. not absorbed) and absorption magnitude (continuous: absorption score)

**Phase 2 -- Test the limiting similarity prediction**:

1. Across all parent-child pairs in multiple SAEs (varying width, sparsity, architecture), compute the minimum decoder angle separation at which absorption drops to zero
2. Test whether this threshold depends on sparsity and frequency ratio as the Lotka-Volterra model predicts
3. Compare to a null model (absorption rate is random given co-occurrence) and to the existing informal theory ("absorption reduces L0")

**Phase 3 -- Test the dual-mechanism hypothesis**:

1. For absorbed feature pairs, decompose the suppressive signal into: (a) decoder overlap component, (b) residual feedback component
2. Compare single-mechanism (decoder overlap only) vs. dual-mechanism predictions of absorption magnitude
3. Use integrated gradients or activation patching to measure each component's contribution

**Phase 4 -- Ecological interventions**:

1. Compare absorption rates in Matryoshka SAE (ecological "disturbance"), OrtSAE (niche differentiation via orthogonality), and standard SAE
2. Map each architectural intervention to its ecological analog and test whether the ecological prediction matches

Models: Gemma 2 2B, GPT-2 Small (via Gemma Scope SAEs, SAEBench SAEs, and SAELens pre-trained checkpoints). All analysis training-free.

### Diagnostic Experiment

The key test that confirms the ecological analogy is load-bearing (not decorative):

**Prediction**: The Lotka-Volterra coexistence criterion (alpha_pc / (K_p/K_c)) should be a *better* predictor of absorption than simpler baselines:
- Baseline 1: Decoder cosine similarity alone
- Baseline 2: Co-occurrence frequency alone
- Baseline 3: Activation magnitude ratio alone

The ecological model combines all three into a single criterion with a specific functional form (the competition ratio must exceed 1 for absorption). If the ecological model's AUC for predicting absorption significantly exceeds each baseline's AUC, the structural correspondence is validated. If the ecological model predicts no better than the best single-variable baseline, the analogy is merely decorative.

**Second diagnostic**: The limiting similarity threshold should exist as a *sharp* transition (analogous to the ecological phase transition), not a gradual decay. Plot absorption probability vs. decoder angle separation; the ecological model predicts a sigmoidal transition around theta*, while a "no-theory" null model predicts monotonic smooth decay.

### Experimental Plan

| Phase | Task | Time Estimate | Model/Data |
|-------|------|--------------|------------|
| 1a | Extract decoder directions, activation stats for all features in target SAEs | 30 min | Gemma 2 2B, Gemma Scope 16k SAE |
| 1b | Run first-letter absorption metric (adapt Chanin et al. code) | 30 min | Same |
| 1c | Compute ecological parameters (alpha, K) for all parent-child pairs | 20 min | Same |
| 2a | Fit Lotka-Volterra coexistence criterion to absorption data | 20 min | Same |
| 2b | Compare AUC vs. baselines (diagnostic experiment) | 15 min | Same |
| 3a | Test limiting similarity threshold: absorption vs. decoder angle | 20 min | Multiple SAE widths/sparsities |
| 3b | Dual-mechanism decomposition via integrated gradients | 40 min | Same |
| 4a | Cross-architecture comparison (Matryoshka, OrtSAE, standard) | 30 min | SAEBench SAEs |
| 4b | Extend to entity-type hierarchy (country->city) | 40 min | Gemma 2 2B |

Total: ~4 hours across multiple tasks, each individual task within 1-hour budget.

### Risk Assessment

1. **Dynamics mismatch**: The Lotka-Volterra equations describe continuous-time population dynamics, while SAE training is discrete-step SGD. The equilibrium analysis is dynamics-independent, but the *path* to absorption may differ. Mitigation: focus on equilibrium predictions, not dynamic trajectories.

2. **Multi-species generalization**: Ecology shows that pairwise Lotka-Volterra analysis can fail to predict outcomes in multi-species communities due to indirect effects and intransitive competition. SAEs have thousands of features competing simultaneously, and pairwise analysis may miss higher-order effects. Mitigation: test on known parent-child pairs where the interaction is approximately pairwise, then extend cautiously.

3. **Parameter estimation**: The ecological competition coefficient alpha_ij involves co-occurrence probability, which requires running the SAE on a corpus. This is computationally feasible for pre-trained SAEs but adds a data-dependence. Mitigation: use the same OpenWebText subset used for SAE training.

4. **Oversimplification**: The Lotka-Volterra model assumes linear competition; real feature interactions may be nonlinear (e.g., TopK introduces hard thresholding). Mitigation: use generalized Lotka-Volterra with nonlinear competition terms, or restrict analysis to architectures where competition is approximately linear (L1-penalty SAEs).

5. **Known result in new dress**: The risk that the ecological framing merely redescribes known results (decoder overlap predicts absorption) without adding predictive power. Mitigation: the diagnostic experiment explicitly tests this -- the ecological model must beat single-variable baselines.

### Novelty Claim

The specific cross-disciplinary insight is: **feature absorption in SAEs is structurally isomorphic to competitive exclusion in ecology**. The Lotka-Volterra coexistence criterion provides a novel, quantitative, falsifiable boundary condition for when absorption occurs. This framing:

1. Has not been applied before in SAE / mechanistic interpretability research (verified via arXiv search)
2. Generates a novel testable prediction (limiting similarity threshold as a sharp phase transition)
3. Provides a novel dual-mechanism hypothesis (competition + feedback suppression, from OAS)
4. Unifies multiple known interventions (Matryoshka = disturbance, OrtSAE = niche differentiation, wider SAE = more niche space) under a single theoretical framework
5. Imports mature mathematical tools (Lotka-Volterra equilibrium analysis, limiting similarity theory) that yield closed-form predictions rather than requiring empirical fitting

The ecological framework also connects to the broader information-theoretic perspective: rate-distortion theory shows that optimal lossy codebooks exhibit "codebook collapse" (rare codewords absorbed by frequent ones) under capacity constraints, which is the information-theoretic analog of competitive exclusion under resource limitation.

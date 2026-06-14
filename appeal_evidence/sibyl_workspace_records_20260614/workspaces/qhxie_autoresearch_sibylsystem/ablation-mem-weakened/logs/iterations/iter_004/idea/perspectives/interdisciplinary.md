# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Rozell et al. (2008)** — "Sparse coding via thresholding and local competition in neural circuits" (*Neural Computation*, 20(10), 2526-2563).
   - **Key mechanism**: The Locally Competitive Algorithm (LCA) solves sparse coding through a recurrent neural circuit with thresholding and lateral inhibition. The inhibition matrix is `G_ij = <phi_i, phi_j>` — the Gram matrix of dictionary elements.
   - **Relevance to SAE absorption**: The LCA inhibition matrix has the *exact same mathematical form* as the SAE decoder correlation matrix `W_dec^T W_dec`. This structural correspondence (already identified in the project's front-runner) provides a neuroscience-grounded mechanistic explanation for absorption as competitive suppression.
   - **Ongoing relevance**: LCA continues to be actively developed — WARP-LCA (2024), adaptive LCA for speech (2024), spiking LCA on neuromorphic hardware (2024), and iP-VAE (2024) extending LCA to variational inference.

2. **Zylberberg et al. (2011)** — "A Sparse Coding Model with Synaptically Local Plasticity and Spiking Neurons Can Account for the Diverse Shapes of V1 Simple Cell Receptive Fields" (*PLoS Computational Biology*).
   - **Key mechanism**: Mathematical proof that homeostasis + lateral inhibition constraints cause network units to remain sparse and independent, learning to approximate the optimal linear generative model.
   - **Relevance**: Provides theoretical grounding for homeostatic rebalancing as a repair mechanism. The proof that "homeostasis + lateral inhibition = stable sparse coding" directly supports the project's H10 (homeostatic rebalancing).

3. **Turrigiano & Nelson (2024)** — "Keeping Your Brain in Balance: Homeostatic Regulation of Network Function" (*Annual Review of Neuroscience*, Vol. 47).
   - **Key mechanism**: Comprehensive review of how neocortical microcircuits maintain stable function through firing rate homeostasis, information flow regulation, and sensory tuning stabilization. Different forms of homeostatic plasticity (excitatory synaptic, inhibitory synaptic, intrinsic excitability) cooperate to stabilize network function.
   - **Relevance**: The multi-mechanism compensation framework (different homeostatic processes for different aspects of network function) suggests that SAE absorption repair may require *multiple* complementary mechanisms, not just activation rebalancing.

4. **Wen & Turrigiano (2024)** — "The interplay between homeostatic synaptic scaling and homeostatic structural plasticity maintains the robust firing rate of neural networks" (*Frontiers in Cellular Neuroscience*).
   - **Key mechanism**: Synaptic scaling (hours-to-days timescale) and structural plasticity work together to maintain firing rate homeostasis. Intrinsic plasticity restores firing rates; synaptic scaling restores correlation structure.
   - **Relevance**: The distinction between "firing rate homeostasis" (first-order) and "correlation structure restoration" (higher-order) maps directly to the precision-recall asymmetry in SAE absorption. Precision (selectivity) is like firing rate — preserved. Recall (coverage) is like correlation structure — degraded.

5. **Zenke & Gerstner (2017/2024)** — "Hebbian plasticity requires compensatory processes on multiple timescales" (*Philosophical Transactions of the Royal Society B*).
   - **Key mechanism**: Hebbian learning (positive feedback, seconds-to-minutes) is inherently unstable without compensatory processes. Rapid compensatory processes (seconds-to-minutes) provide immediate stabilization; slow homeostatic plasticity (hours-to-days) provides fine-tuning.
   - **Relevance**: The SAE training objective (reconstruction + sparsity) is analogous to Hebbian learning — it has a positive feedback loop where sparsity pressure drives absorption. The "compensatory process" analogy suggests that post-hoc repair (homeostatic rebalancing) may be *necessary* because the training dynamics lack built-in stabilization against absorption.

6. **BioLogicalNeuron (Hakim et al., 2025)** — "Biologically inspired neural network layer with homeostatic regulation and adaptive repair mechanisms" (*Scientific Reports*).
   - **Key mechanism**: Novel neural network layer integrating calcium-driven homeostatic regulation, self-repair mechanisms, adaptive noise injection, and dynamic stability monitoring.
   - **Relevance**: First demonstration of homeostatic repair in artificial neural networks with strong performance. Validates that biological compensation mechanisms can be computationally effective.

#### Physics / Statistical Mechanics / Information Theory

7. **Hino & Murata (2009)** — "An Information Theoretic Perspective of the Sparse Coding" (ISNN 2009, LNCS 5551).
   - **Key mechanism**: Formulates sparse coding as a rate-distortion optimization problem. The rate-distortion function leads to an objective interpretable as an information-theoretic formulation of sparse coding.
   - **Relevance**: Provides the theoretical lens for understanding absorption as optimal compression. The SAE objective (reconstruction + sparsity) is exactly a rate-distortion tradeoff where "rate" = sparsity and "distortion" = reconstruction error. Absorption is the optimal strategy for minimizing rate when features are hierarchically correlated.

8. **Schaeffer et al. (2024)** — "Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs" (arXiv:2410.11179).
   - **Key mechanism**: Frames interpretability through rate-distortion-perception tradeoff. Proposes MDL-SAEs optimizing for description length rather than reconstruction error. Shows minimal description length yields more interpretable features.
   - **Relevance**: Directly connects SAE evaluation to rate-distortion theory. The "perception" dimension maps to interpretability — absorption reduces perception quality (parent features become invisible) even when distortion (reconstruction) is low.

9. **Thermodynamic Bayesian Inference (2024)** — "Thermodynamic Bayesian Inference" (arXiv:2410.01793).
   - **Key mechanism**: Establishes connections between thermodynamic computing, Bayesian inference, and the free energy principle from statistical physics.
   - **Relevance**: The free energy principle (Friston) provides a variational framework where absorption can be understood as the system settling into a free energy minimum where parent features are "explained away" by child features — analogous to predictive coding.

10. **Fenchel-Young Variational Learning (2025)** — "Fenchel-Young Variational Learning" (arXiv:2502.10295).
    - **Key mechanism**: Introduces FY free energy concepts generalizing classical variational free energy. Connections between variational inference, statistical physics, and learning with sparse posterior distributions.
    - **Relevance**: The FY free energy framework may provide a more general theoretical setting for understanding SAE training dynamics, including absorption as a sparsity-induced phase transition.

11. **Partial Information Decomposition (PID) — Recent Advances (2024-2025)**.
    - **PIDReg (2025)**: Multimodal regression with Gaussian PID, using Cauchy-Schwarz divergence to enforce uniqueness of information per modality.
    - **PIRD (2025)**: Partial Information Rate Decomposition extending PID to dynamic/time-series settings. Reveals that temporal correlations fundamentally alter PID measures.
    - **Layer-wise PID + Attention Knockout (2025)**: Applies PID layer-by-layer to characterize depth-wise evolution of information structure in pretrained models.
    - **Relevance**: PID decomposes information into redundancy, unique information, and synergy. In SAE absorption, the parent feature's information is *redundant* with the child feature (when child fires, parent is implied), but the SAE fails to extract the *unique* information (parent fires independently on non-child inputs). PID provides a formal framework for quantifying this decomposition.

#### Biology / Evolution / Ecology

12. **Chanin et al. (2024)** — "A is for Absorption" (arXiv:2409.14507).
    - **Key mechanism**: The paper itself uses ecological metaphors — feature splitting as "speciation" and feature absorption as "competitive exclusion."
    - **Relevance**: The competitive exclusion principle (Gause's principle) states that two species competing for identical limited resources cannot coexist. In SAEs, parent and child features "compete" for activation budget (sparsity constraint). The child feature "wins" and excludes the parent — exactly competitive exclusion.

13. **Yang & Park (2024)** — "Enhancing biodiversity through intraspecific suppression in large ecosystems" (arXiv:2305.12341v3, updated 2024).
    - **Key mechanism**: Uses cavity methods from statistical physics to study competitive exclusion and biodiversity. Shows that intraspecific suppression (self-regulation) can enhance coexistence.
    - **Relevance**: The cavity method approach (mean-field analysis of competitive systems) could be applied to SAE feature competition. The insight that "self-regulation enhances coexistence" suggests that per-feature activation regularization (not just global sparsity) could reduce absorption.

14. **Neutral vs. Niche Theory in Community Ecology (2024-2025)**.
    - **Key mechanism**: Neutral theory (Hubbell) posits that ecological communities are shaped by drift and dispersal, not niche differences. Niche theory posits that species coexistence requires differentiation in resource use.
    - **Relevance**: The "Sanity Checks" paper (arXiv:2602.14111) showing random SAE baselines match trained SAEs is analogous to neutral theory — maybe SAE features are not strongly selected by "niche" (semantic content) but by "drift" (random initialization + overparameterization). Absorption, in this view, is not a failure of niche differentiation but a consequence of neutral dynamics in a highly overcomplete dictionary.

15. **Turing Patterns / Reaction-Diffusion (2024-2025)** — Ouchdiri et al. (2025), Bisi et al. (2025).
    - **Key mechanism**: Turing patterns emerge from the interaction of short-range activation and long-range inhibition. The Nodal-Lefty system shows competitive inhibition alone creates labyrinthine patterns; adding direct inhibition creates spot patterns.
    - **Relevance**: The activation-inhibition dynamics in Turing patterns are structurally similar to SAE feature competition. The "short-range activation, long-range inhibition" principle could inspire SAE architectures where similar features (short-range in semantic space) activate each other while dissimilar features (long-range) inhibit each other — the opposite of current SAE dynamics where similar features compete.

#### Signal Processing / Source Separation

16. **ICA Non-identifiability and Sparsity (2024-2025)** — Ng et al. (2024), Zheng & Zhang (2024), Lachapelle et al. (2024).
    - **Key mechanism**: ICA faces non-identifiability when sources are Gaussian or correlated. Recent work shows structural sparsity assumptions on the mixing process can restore identifiability without requiring non-Gaussian sources or auxiliary variables.
    - **Relevance**: SAE absorption is fundamentally a non-identifiability problem — multiple SAE configurations achieve similar reconstruction, but only some correspond to "correct" feature disentanglement. The sparsity-based identifiability results from ICA suggest that *structural* sparsity (not just activation sparsity) may be needed for absorption-free SAEs.

17. **SparseICA (Wang et al., 2024)** — Published in *Journal of the American Statistical Association*.
    - **Key mechanism**: Sparse ICA for neuroimaging using relax-and-split framework. Balances statistical independence and sparsity.
    - **Relevance**: The balance between independence (orthogonality) and sparsity is exactly the tension in SAEs. OrtSAE (2025) enforces orthogonality; standard SAEs enforce sparsity. SparseICA shows both can be balanced — suggesting a principled framework for absorption reduction.

### Cross-Disciplinary Gaps

The following transplants have NOT been tried in the SAE absorption literature:

| Source Field | Principle | SAE Application Status |
|-------------|-----------|----------------------|
| **Ecology (neutral theory)** | Drift vs. niche selection | NOT explored — could explain why random SAEs match trained SAEs |
| **PID (information theory)** | Redundancy/unique/synergy decomposition | NOT explored — could quantify information lost to absorption |
| **Turing patterns** | Short-range activation, long-range inhibition | NOT explored — could inspire new SAE architectures |
| **Chemical reaction networks** | Mass action kinetics, extinction dynamics | NOT explored — could model feature competition dynamics |
| **Homeostatic plasticity (multi-mechanism)** | Different compensation for different network aspects | NOT explored — current repair uses single mechanism |
| **Predictive coding (iterative)** | Top-down prediction error minimization | Partially explored (MP-SAE, Innovator Candidate B) |
| **Rate-distortion bounds** | Information-theoretic impossibility results | Partially explored (Theoretical Candidate B, Innovator Candidate C) |

---

## Phase 2: Initial Candidates

### Candidate A: The Competitive Exclusion Principle — An Ecological Model of Feature Absorption

- **Source principle**: Gause's competitive exclusion principle from ecology: two species competing for identical limited resources cannot coexist indefinitely; the superior competitor eliminates the inferior one.
- **Structural correspondence**:
  - **Ecological system**: Species compete for resources in an ecosystem with carrying capacity.
  - **SAE system**: Latent features compete for activation budget (sparsity constraint) in the representation space.
  - **Species** <-> **SAE latent feature**
  - **Resource niche** <-> **Input activation pattern**
  - **Carrying capacity** <-> **Sparsity budget (k active latents per sample)**
  - **Competitive exclusion** <-> **Feature absorption (parent feature excluded by child feature)**
  - **Niche differentiation** <-> **Feature orthogonality / independence**
  - **Intraspecific suppression** <-> **Self-inhibition / activation regularization**
- **Hypothesis**: Feature absorption rate in SAEs follows the same mathematical form as competitive exclusion probability in ecological models. Specifically, the probability that feature i absorbs feature j is proportional to their niche overlap (decoder correlation) divided by the sparsity budget (carrying capacity).
- **Why it's not just a metaphor**: The mathematical structure is identical. Both systems involve: (1) a finite resource budget, (2) competitive interaction between entities, (3) hierarchical structure (ecological niches / feature hierarchies), and (4) extinction of weaker competitors. The Lotka-Volterra competition equations map directly onto SAE activation dynamics.
- **Novelty estimate**: 6/10. The competitive exclusion metaphor is already used in Chanin et al.'s paper. The mathematical formalization (deriving absorption rate from Lotka-Volterra dynamics) would be new but the metaphor is not.

### Candidate B: Partial Information Decomposition — Quantifying the Information Lost to Absorption

- **Source principle**: Partial Information Decomposition (PID) from information theory decomposes the mutual information between multiple sources and a target into redundant, unique, and synergistic components.
- **Structural correspondence**:
  - **PID framework**: For sources X1, X2 and target Y, I(X1, X2; Y) = Redundancy + Unique(X1) + Unique(X2) + Synergy.
  - **SAE absorption**: For parent feature P, child feature C, and input stimulus S:
    - When C fires, P is implied (redundant information)
    - When P fires without C, P provides unique information
    - Absorption = the SAE fails to extract Unique(P) when C is present
  - **Redundancy** <-> **Information captured by child feature that implies parent**
  - **Unique information** <-> **Information captured only by parent feature (lost in absorption)**
  - **Synergy** <-> **Information that emerges from parent-child co-activation**
- **Hypothesis**: The information lost to absorption can be quantified as the unique information component in a PID decomposition. An SAE with zero absorption would have zero redundancy between parent and child features (all information is unique or synergistic). The absorption rate correlates with the redundancy fraction.
- **Why it's not just a metaphor**: PID is a formal, axiomatically grounded decomposition. The mapping between "redundancy in information sources" and "absorption in SAE features" is exact — both describe situations where one signal contains all the information of another, making the second signal redundant. The mathematical structure (set-theoretic information decomposition) applies to both.
- **Novelty estimate**: 8/10. PID has been used in neuroscience for neural population coding but NEVER for SAE feature absorption. The application of PID to quantify absorption-induced information loss is entirely unexplored.

### Candidate C: The Local Inhibition Graph — A Neuroscience-Inspired Diagnostic for Feature Absorption

- **Source principle**: The Locally Competitive Algorithm (LCA) from Rozell et al. (2008), where sparse coding is implemented via a recurrent neural network with lateral inhibition. The inhibition matrix is `G_ij = <phi_i, phi_j>` — the Gram matrix of dictionary elements.
- **Structural correspondence**:
  - **LCA inhibition matrix**: `G_LCA = Phi^T Phi` (Gram matrix of dictionary elements, zeroed diagonal).
  - **SAE decoder correlation matrix**: `C = W_dec^T W_dec` (correlation matrix of decoder directions).
  - **Structural claim**: `C = G_LCA` — the SAE decoder correlation matrix is exactly the LCA inhibition matrix.
  - **LCA dynamics**: `tau * du_i/dt = -u_i + b_i - sum_{j != i} G_ij * a_j` — neuron i's state is suppressed by the activation of neuron j weighted by their dictionary correlation.
  - **SAE absorption**: When child feature j fires, it suppresses parent feature i's activation through decoder correlation — exactly the LCA inhibition mechanism.
- **Hypothesis**: Edges in the local inhibition graph (constructed from decoder correlations) predict known absorption pairs with precision significantly above chance. The graph structure explains the precision-recall asymmetry (precision = selectivity preserved; recall = coverage reduced by suppression).
- **Why it's not just a metaphor**: The correspondence is mathematically exact, not metaphorical. `W_dec^T W_dec` and `Phi^T Phi` are the same mathematical object. The LCA dynamics and SAE activation dynamics share the same differential equation structure. This is a structural isomorphism, not a surface-level analogy.
- **Novelty estimate**: 9/10. No prior work connects LCA to SAE absorption. The structural correspondence `W_dec^T W_dec = G_LCA` has not been articulated. The local inhibition graph is novel.

---

## Phase 3: Self-Critique

### Against Candidate A: Competitive Exclusion Principle

- **Shallow analogy attack**: The competitive exclusion principle is a *verbal* analogy, not a mathematical correspondence. Lotka-Volterra equations describe population dynamics over time, while SAE absorption is a static property of trained weights. The timescales and dynamics are fundamentally different.
- **Scale mismatch attack**: Competitive exclusion operates at the population level (many individuals per species), while SAE features are single entities. The stochastic dynamics of extinction in finite populations (demographic stochasticity) don't apply to deterministic SAE weights.
- **Prior transplant check**: Chanin et al. (2024) already use the competitive exclusion metaphor in their paper. The "A is for Absorption" paper explicitly frames absorption as competitive exclusion. This is not a new transplant.
- **Testability attack**: Deriving absorption rate from Lotka-Volterra dynamics requires fitting ecological parameters (growth rates, competition coefficients) to SAE data. These parameters have no natural interpretation in SAEs, making the model arbitrary.
- **Verdict**: WEAK — The metaphor is already used in the field. The mathematical formalization adds complexity without predictive power. The ecological analogy does not provide new insights beyond what the LCA framework (Candidate C) already provides more rigorously.

### Against Candidate B: Partial Information Decomposition

- **Shallow analogy attack**: PID decomposes information from *multiple sources* about a *single target*. In SAE absorption, we have *two features* (parent and child) and the "target" is the input stimulus. But the parent-child relationship is hierarchical (child implies parent), not symmetric like typical PID sources. The PID axioms may not apply cleanly.
- **Scale mismatch attack**: PID is computationally expensive for high-dimensional continuous variables. Computing PID between SAE features and input stimuli requires estimating mutual information in 768-dimensional spaces with continuous distributions. Current PID estimators (I_min, I_BROJA, I_delta) struggle with more than a few dimensions.
- **Prior transplant check**: PID has been used in neuroscience for neural population coding (e.g., PMID 34166489, 2021) and recently for layer-wise analysis of vision-language transformers (2025). However, NO prior work applies PID to SAE feature absorption. The transplant is genuinely new.
- **Testability attack**: The key prediction — "absorption rate correlates with redundancy fraction" — is testable but requires reliable PID estimation. If PID estimates are noisy, the correlation may be spurious. A diagnostic experiment: compute PID on synthetic data where ground-truth redundancy is known, then validate the correlation.
- **Verdict**: MODERATE — The transplant is genuinely novel and the structural correspondence is formal. The main weakness is computational feasibility for high-dimensional SAE features. Could be addressed by working in the latent space (low-dimensional feature activations) rather than input space.

### Against Candidate C: Local Inhibition Graph

- **Shallow analogy attack**: Is the LCA-SAE correspondence really structural? LCA is a dynamical system for *inference* (finding sparse codes given a fixed dictionary), while SAEs are *trained* end-to-end. The SAE encoder is learned, not derived from the decoder via gradient descent. The correspondence holds for the decoder matrix but not necessarily for the encoder.
- **Scale mismatch attack**: LCA was designed for small dictionaries (hundreds of elements) and image patches. SAEs have dictionaries of 24K-1M latents. The inhibition graph for 1M latents would have 10^12 potential edges — intractable without local approximation. The local graph (top-k neighbors) addresses this but introduces a hyperparameter.
- **Prior transplant check**: Rozell et al. (2008) has ~2000 citations but ZERO applications to LLM SAEs. No paper in the SAE literature mentions LCA, inhibition graphs, or decoder correlation analysis for absorption. The novelty searches conducted in previous rounds confirmed no matches. The transplant is genuinely unexplored.
- **Testability attack**: The key prediction (precision@20 >= 0.10) is falsifiable. If graph edges don't predict absorption pairs, the structural correspondence fails. However, even if edges don't predict absorption, the graph still provides a useful visualization of feature competition — a weaker but still valuable contribution.
- **Verdict**: STRONG — The structural correspondence is mathematically exact. The predictions are specific and falsifiable. The framework is scalable (local graph approximation). The prior art check confirms zero existing work. The main risk is empirical validation, but the theoretical grounding is solid.

---

## Phase 4: Refinement

### Dropped

- **Candidate A (Competitive Exclusion)**: Dropped because the metaphor is already used in Chanin et al.'s paper and the mathematical formalization adds no predictive power beyond what LCA provides. The ecological analogy is surface-level compared to the exact LCA correspondence.

### Strengthened

- **Candidate C (Local Inhibition Graph)**: Strengthened by formalizing the structural correspondence:
  1. **Exact mapping**: `W_dec^T W_dec = G_LCA` — the decoder correlation matrix is exactly the LCA inhibition matrix.
  2. **Dynamical correspondence**: SAE encoder activation `z = f(W_enc * a + b_pre)` with ReLU/TopK thresholding is analogous to LCA thresholding `a_i = T_lambda(u_i)`.
  3. **Precision-recall mechanism**: In LCA, inhibition from j to i reduces u's activation (recall loss) but does not cause false positives (precision preserved). This explains the project's H5 finding.
  4. **Diagnostic experiment**: Construct local inhibition graph, validate against Chanin absorption pairs, test precision-recall asymmetry explanation.

- **Candidate B (PID)**: Strengthened by focusing on a computationally feasible variant:
  1. Instead of computing PID in high-dimensional input space, compute PID in the low-dimensional latent activation space.
  2. Use the Gaussian PID approximation (PIDReg, 2025) which is tractable for continuous variables.
  3. Focus on the redundancy-unique decomposition for parent-child feature pairs where ground-truth hierarchy is known (first-letter features).
  4. The diagnostic experiment: measure redundancy fraction for absorbed vs. non-absorbed parent-child pairs; predict that absorbed pairs have higher redundancy.

### Selected Front-Runner

**Candidate C: The Local Inhibition Graph** is selected as the front-runner because:
1. The structural correspondence is exact (`W_dec^T W_dec = G_LCA`), not metaphorical.
2. It explains the project's strongest finding (H5: precision invariant, recall variable) via competitive suppression.
3. It provides a practical, training-free diagnostic tool.
4. It is scalable to million-latent SAEs via local graph approximation.
5. No prior work has made this connection.

**Candidate B (PID) is retained as a secondary direction** because:
1. It provides a complementary information-theoretic perspective on absorption.
2. It could quantify "how much information is lost" — a question the inhibition graph doesn't answer.
3. If the inhibition graph succeeds, PID analysis could strengthen the theoretical contribution.
4. If the inhibition graph partially fails, PID provides an alternative framing.

---

## Phase 5: Final Proposal

### Title

**"The Local Inhibition Graph: A Neuroscience-Inspired Training-Free Diagnostic for Feature Absorption in Sparse Autoencoders"**

Alternative: **"Decoder Correlations Reveal Competitive Suppression: A Local Inhibition Graph for SAE Feature Absorption"**

### Source Principle

The Locally Competitive Algorithm (LCA) from Rozell et al. (2008) is a biologically plausible sparse coding algorithm implemented as a recurrent neural network. In LCA, neurons compete via lateral inhibition: when neuron j fires, it suppresses neuron i by an amount proportional to the correlation between their receptive fields. The inhibition matrix is `G_ij = <phi_i, phi_j>` — the Gram matrix of dictionary elements.

### Structural Correspondence

For a Sparse Autoencoder with decoder matrix `W_dec` (shape `d_dict x d_model`), the decoder correlation matrix `C = W_dec^T W_dec` has the exact same mathematical form as the LCA inhibition matrix `G_LCA = Phi^T Phi`. This is a structural isomorphism:

| LCA Component | SAE Component | Mathematical Form |
|--------------|---------------|-------------------|
| Dictionary element `phi_i` | Decoder direction `W_dec[i]` | Vector in `R^{d_model}` |
| Inhibition matrix `G` | Decoder correlation `C` | `Phi^T Phi` / `W_dec^T W_dec` |
| Neuron activation `a_i` | Latent activation `z_i` | Scalar |
| Threshold `lambda` | Sparsity parameter (L1 coeff / TopK k) | Scalar |
| Feedforward input `b_i` | Encoder pre-activation `<W_enc[i], a>` | Scalar |

**Key insight**: In LCA, inhibition from child feature j to parent feature i suppresses i's activation when j fires. This is precisely feature absorption: the parent feature fails to fire because the child feature's activation suppresses it through decoder correlation.

### Hypothesis

For a pretrained SAE, edges in the local inhibition graph (top-k correlated neighbors per latent) correspond to known absorption pairs with precision significantly above chance. The competitive suppression mechanism explains why absorption affects recall (coverage) but not precision (selectivity).

### Method

**Phase 1: Construct Local Inhibition Graph**
1. For each latent i, compute decoder correlations `C_ij = <W_dec[i], W_dec[j]>` for all j != i.
2. Keep top-k neighbors per latent (k=20-50) with highest `|C_ij|`.
3. Edge weight = `C_ij` (signed correlation).
4. Complexity: `O(k * d_dict * d_model)` — feasible for 24K-1M latents.

**Phase 2: Validate Against Absorption Pairs**
- Use Chanin et al.'s absorption detection on first-letter features (A-Z) as ground truth.
- For each absorption pair (parent i, absorbing j), check if j is in N(i).
- Compute precision@k, recall@k, and Fisher exact test for enrichment.
- Compare against random baseline (shuffle latent indices).

**Phase 3: Test Precision-Recall Asymmetry Explanation**
- For each feature, compute total incoming inhibition (sum of edge weights from neighbors).
- Test correlation between total inhibition and recall loss.
- Test correlation between total inhibition and precision (predicted: no correlation).

**Phase 4: Layer-Dependent Analysis**
- Construct graphs for layers 0, 4, 8, 10 of GPT-2 Small.
- Compare graph statistics (mean edge weight, density, clustering coefficient) across layers.
- Test whether layer 8 (where H1b was significant) has stronger inhibition structure.

**Phase 5: Homeostatic Rebalancing (Exploratory)**
- For input activation a, compute original latents: `z = f(W_enc * a + b_pre)`.
- Compute inhibition per latent: `inh_i = sum_{j in N(i)} C_ij * z_j`.
- Apply boost: `z'_i = z_i + alpha * inh_i`.
- Clip negative values; constrain reconstruction error increase < 5%.
- Test whether rebalancing restores parent feature firing.

### Diagnostic Experiment

The key test that confirms the analogy is load-bearing:

**Experiment**: Construct the local inhibition graph for GPT-2 Small SAE (24K latents, layer 8). For each of the 26 first-letter features, identify its top-20 most correlated neighbors. Check how many of these neighbors are known absorption pairs (from Chanin et al.'s metric).

**Prediction**: Precision@20 >= 0.10 (vs. ~0.004 expected by chance = 20/24000). This represents a 25x enrichment over chance.

**Falsification**: If precision@20 <= 0.05, the structural correspondence between decoder correlations and absorption pairs fails.

**Why this is diagnostic**: If the LCA correspondence is correct, decoder correlations must predict absorption pairs because absorption *is* competitive suppression mediated by decoder correlation. If decoder correlations do NOT predict absorption, then absorption is not competitive suppression — it must have a different mechanism.

### Experimental Plan

| Experiment | Model | SAE | Metrics | Time |
|---|---|---|---|---|
| E1: Graph construction + validation | GPT-2 Small | gpt2-small-res-jb (24K) | Precision@k, recall@k, Fisher test | ~15 min |
| E2: Precision-recall asymmetry test | GPT-2 Small | Same | Correlation (inhibition vs recall, precision) | ~15 min |
| E3: Layer-dependent graph structure | GPT-2 Small | Same (layers 0/4/8/10) | Graph stats by layer | ~20 min |
| E4: Homeostatic rebalancing | GPT-2 Small | Same | Absorption rate change, reconstruction error | ~30 min |
| E5: Cross-model validation | Gemma-2-2B | GemmaScope 16K | All above metrics | ~30 min |

**Total estimated time**: ~2 GPU-hours (well within project constraints).

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| Graph edges don't correspond to absorption pairs | Medium | High | The structural correspondence is mathematically exact. If edges don't match, this itself is a finding about decoder correlation limitations. Fallback: diagnostic-only claims. |
| Homeostatic rebalancing degrades reconstruction | Medium | Medium | Alpha is tunable; sweep to find values that improve absorption without degrading reconstruction. Fallback: report Pareto frontier. |
| Repair doesn't improve steering/probing | Medium | Medium | This strengthens the "absorption is benign" claim. The diagnostic contribution stands independently. |
| Local graph misses long-range absorption | Medium | Medium | Test multiple k values (10, 20, 50, 100). Fallback: hierarchical clustering. |
| Gemma-2-2B access issues | High | Medium | Primary experiments on GPT-2 Small; Gemma as validation only. |
| Neuroscience analogy criticized as superficial | Low | High | The correspondence is exact (`W_dec^T W_dec = G_LCA`), not metaphorical. Ground claims in Rozell et al.'s equations. |

### Novelty Claim

1. **First LCA-SAE connection**: No prior work connects Rozell et al.'s Locally Competitive Algorithm to Sparse Autoencoder feature absorption. The structural correspondence (`W_dec^T W_dec = G_LCA`) is exact and has not been articulated.

2. **First local inhibition graph for SAE diagnostics**: No existing paper constructs a graph from decoder correlations to diagnose absorption.

3. **First mechanistic explanation for precision-recall asymmetry**: The competitive suppression framework explains why absorption affects recall but not precision — a finding from the project's data that currently lacks theoretical grounding.

4. **First training-free post-hoc repair for absorption**: All existing solutions (Matryoshka, OrtSAE, ATM) require retraining. Homeostatic rebalancing operates on pretrained SAEs.

### Secondary Direction: PID Analysis (If Primary Succeeds)

If the Local Inhibition Graph is validated, a Partial Information Decomposition analysis can strengthen the theoretical contribution:

- Compute PID (redundancy, unique, synergy) for parent-child feature pairs in the latent activation space.
- Test whether absorbed pairs have higher redundancy fraction than non-absorbed pairs.
- Quantify "information lost to absorption" as the unique information component that the SAE fails to extract.

This provides an information-theoretic complement to the dynamical-systems explanation from LCA.

---

## Synthesis with Other Perspectives

### How This Complements Existing Perspectives

| Perspective | Their Contribution | Interdisciplinary Complement |
|------------|-------------------|------------------------------|
| **Innovator** | Identified decoder correlation gap; proposed PID analysis | LCA provides exact dynamical correspondence; PID provides information-theoretic complement |
| **Theoretical** | Rate-distortion bound; spectral characterization | LCA provides mechanistic (not just phenomenological) explanation; homeostatic rebalancing connects to Zylberberg et al.'s proof |
| **Contrarian** | "Absorption is benign" finding | Competitive suppression explains WHY absorption is benign (information is redistributed, not destroyed) |
| **Empiricist** | Rigorous controls, random baselines | Inhibition graph predictions are falsifiable with clear significance thresholds |
| **Pragmatist** | Training-free constraint | LCA graph is entirely training-free; computed from pretrained weights |

### What Was Dropped from This Perspective

- **Competitive exclusion ecology (Candidate A)**: Already used as metaphor in Chanin et al.; LCA provides more rigorous formalization.
- **Turing pattern architecture inspiration**: Too speculative; no clear experimental path.
- **Chemical reaction network dynamics**: Timescale mismatch; CRN dynamics don't map cleanly onto SAE training.
- **Neutral theory of ecology**: Interesting for explaining "Sanity Checks" results but doesn't address absorption specifically.

### What Was Retained as Secondary

- **PID analysis (Candidate B)**: Retained as a complementary information-theoretic perspective. If the primary experiments succeed, PID can quantify the information-theoretic cost of absorption.
- **Multi-mechanism homeostasis**: The insight that different homeostatic processes restore different network properties suggests that single-pass rebalancing may be insufficient — iterative or multi-component repair may be needed.

---

## Sources

- [Rozell et al., 2008] "Sparse coding via thresholding and local competition in neural circuits" — [Neural Computation](https://pubmed.ncbi.nlm.nih.gov/18439138/)
- [Zylberberg et al., 2011] "A Sparse Coding Model with Synaptically Local Plasticity and Spiking Neurons" — [PLoS Computational Biology](https://pmc.ncbi.nlm.nih.gov/articles/PMC7691792/)
- [Turrigiano & Nelson, 2024] "Keeping Your Brain in Balance" — [Annual Review of Neuroscience](https://www.annualreviews.org/doi/10.1146/annurev-neuro-092523-110001)
- [Zenke & Gerstner, 2017] "Hebbian plasticity requires compensatory processes on multiple timescales" — [Philosophical Transactions B](https://royalsocietypublishing.org/rstb/article/372/1715/20160259)
- [Hino & Murata, 2009] "An Information Theoretic Perspective of the Sparse Coding" — ISNN 2009
- [Schaeffer et al., 2024] "Interpretability as Compression" — [arXiv:2410.11179](https://arxiv.org/abs/2410.11179)
- [Chanin et al., 2024] "A is for Absorption" — [arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- [Yang & Park, 2024] "Enhancing biodiversity through intraspecific suppression" — [arXiv:2305.12341](https://arxiv.org/abs/2305.12341)
- [Ouchdiri et al., 2025] "Turing Patterns in a Morphogenetic Model" — [arXiv:2509.15829](https://arxiv.org/abs/2509.15829)
- [Ng et al., 2024] "On the Identifiability of Sparse ICA" — [arXiv:2408.10353](https://arxiv.org/abs/2408.10353)
- [Tang et al., 2025] "A Unified Theory of Sparse Dictionary Learning" — [arXiv:2512.05534](https://arxiv.org/abs/2512.05534)
- [Li et al., 2025] "The Geometry of Concepts" — [Entropy 27(4):344](https://www.mdpi.com/1099-4300/27/4/344)
- [WARP-LCA, 2024] "WARP-LCA: Efficient Convolutional Sparse Coding" — [arXiv:2410.18794](https://arxiv.org/abs/2410.18794)
- [BioLogicalNeuron, 2025] — [PubMed](https://pubmed.ncbi.nlm.nih.gov/41028030/)
- [PIDReg, 2025] — [arXiv:2512.22102](https://arxiv.org/abs/2512.22102)
- [PIRD, 2025] "Partial Information Rate Decomposition" — [arXiv:2502.04550](https://arxiv.org/abs/2502.04550)

# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Predictive Coding under the Free Energy Principle** (Friston & Kiebel, 2009, Phil. Trans. R. Soc. B) — The brain minimizes "free energy" (a variational bound on surprise) by maintaining a hierarchical generative model. Top-down predictions suppress bottom-up prediction errors at every level. Key principle: *expected stimuli are actively suppressed*; only prediction errors propagate up. This is a neural mechanism for absorption-like behavior: a broad prior prediction "absorbs" specific sensory detail when the prior is highly confident.

2. **Feature-Selective Inhibitory Mechanisms Enable Expectation Suppression in Cortical Microcircuits** (Scientific Reports, 2025) — SOM interneurons deliver *feature-specific dendritic inhibition* of pyramidal cells for predicted/expected inputs; VIP neurons enable disinhibition for unexpected inputs. Provides a cellular-level substrate for hierarchical suppression. The general "expected feature" suppresses firing of its specific instances.

3. **Inhibitory Interneurons Decorrelate Excitatory Cells to Drive Sparse Code Formation in V1** (Journal of Neuroscience, 2013) — Excitatory-inhibitory spiking circuits learn sparse codes with Gabor-like receptive fields using only local plasticity. Inhibition *decorrelates* representations by suppressing predictable (redundant) spikes. Key mechanism: the interneuron network effectively implements a competitive exclusion process among correlated features.

4. **Competition, Stability, and Functionality in E-I Neural Circuits** (arXiv:2512.05252, 2025) — Game-theoretic treatment of neural E-I dynamics; lateral inhibition models function as *contrast enhancers*, selectively sharpening subtle differences through hierarchical E-I interplay. Corollary: features with high overlap are driven to compete, with dominant features suppressing weaker ones.

5. **Canonical Neural Circuit Frameworks for Competitive Selection** (eLife) — Competitive selection is a symmetry-breaking process: among options differing in "norm" (activation strength), inhibition proportional to norm causes dominant options to suppress others. The "donut-like pattern of competitive inhibition" is a neural implementation of L0 selection pressure.

#### Physics / Statistical Mechanics / Information Theory

6. **Renormalization Group Framework for Scale-Invariant Feature Learning** (AAAI 2025; ECT* Workshop, May 2024) — Layer-wise transformations in deep networks viewed as RG coarse-graining operations. Each RG step integrates out "fast degrees of freedom" (high-frequency, fine-grained features) and retains "slow degrees of freedom" (low-frequency, broad features). The *relevant* features survive renormalization; *irrelevant* ones are absorbed into the effective description of the coarser scale. Key structural correspondence: what physics calls "integrating out" is operationally analogous to SAE feature absorption—low-frequency parent features survive as effective descriptions; high-frequency children get absorbed into parents.

7. **Niche Overlap and Hopfield-Like Interactions in Generalised Random Lotka-Volterra Systems** (arXiv:2301.11703) — Random Lotka-Volterra competitive systems with niche overlap have fixed points determined by the niche overlap matrix (analogous to the feature co-occurrence or cosine similarity matrix). Key finding: when competition coefficients exceed a threshold determined by the overlap, competitive exclusion occurs—one "species" drives the other to zero. The mathematical structure is a Hopfield-like energy function over the species abundances.

8. **Compressed Sensing with Redundant Dictionaries** (Candès, Eldar, Needell, Randall, SIAM J. Math. Anal.) — For sparse recovery in overcomplete dictionaries, the D-RIP condition is sufficient for exact recovery. When the dictionary has *structured redundancy* (e.g., hierarchical parent-child atoms sharing directions), the D-RIP may be violated for pairs of correlated atoms—recovery fails precisely for the configurations that produce feature absorption in SAEs.

9. **Information Bottleneck Method** (Tishby et al., 1999; MDPI Entropy 2024 review) — The IB principle compresses input X into minimal representation T while preserving relevant information about label Y. Rate-distortion bifurcations correspond to phase transitions in which features are selectively retained or discarded. Key: IB hierarchical representations exhibit *successive refinement*—coarse representations are successively refined rather than replaced. The absorption failure mode in SAEs is formally the IB operating "in the wrong direction": instead of refining from coarse to fine, the SAE collapses fine distinctions back to coarse representations.

10. **Theory and Application of the IB Method** (MDPI Entropy, 2024) — IB with side information can maximize information about one target variable while *minimizing* information about another. This formalization could provide the right objective for an SAE that must preserve both parent and child features: jointly maximize mutual information with parent and child labels while minimizing shared coding.

#### Biology / Evolution / Ecology

11. **Competitive Exclusion Principle and Lotka-Volterra Dynamics** (Wikipedia; LibreTexts Biology; ScienceDirect) — Gause's Law: two species cannot coexist if they occupy the same ecological niche (require identical resources). Mathematically: stable coexistence requires α₁₂·α₂₁ < 1 (intraspecific competition dominates interspecific competition). If α₁₂ > K₁/K₂, species 1 is excluded. Precise analogy: SAE latents are species, activation frequency is carrying capacity, cosine similarity between decoder vectors is the competition coefficient. Feature absorption is competitive exclusion in the SAE "ecosystem."

12. **Species Competition: Coexistence, Exclusion and Clustering** (Phil. Trans. R. Soc. A, 2009) — In multidimensional niche space, intensity of competition decreases as distance between species increases. The *radius of the assemblage niche* (ability to maintain equilibrium with all species coexisting) decreases as resource overlap increases. Analogy: in SAE latent space, this defines a minimum "niche separation" (minimum decoder cosine dissimilarity) required for two features to coexist without one absorbing the other.

13. **Resource Partitioning and Niche Differentiation** (Modern Coexistence Theory, PMC 2022 review) — Stable coexistence requires each species to limit *its own* growth more than it limits others'. Translated to SAEs: a feature is "stable" (not absorbed) if it activates on tokens where no other latent activates more strongly. The condition for absorption-free coexistence is a direct mathematical analogue of the Lotka-Volterra coexistence condition.

14. **Geometry of Concepts: Sparse Autoencoder Feature Structure** (PMC, 2025) — Reviews atom-scale, brain-scale, and galaxy-scale geometric structures of SAE features. Explicitly notes that "feature absorption" and "feature splitting" are related to niche overlap concepts from ecology, though without formalizing the analogy. This is the closest existing cross-domain connection, but it remains at the level of metaphor rather than structural correspondence.

15. **Hierarchical Sparse Dictionary Learning** (Springer ICIAR 2015) — Building an adaptive dictionary regularized by an a-priori overcomplete dictionary leads to a two-level hierarchy: learned atoms are sparse in the prior dictionary. This is the signal-processing precursor to Matryoshka SAEs and provides non-asymptotic recovery bounds for hierarchical settings.

### Cross-Disciplinary Gaps

**Gap A: The Lotka-Volterra competitive exclusion analogy has been mentioned informally (Geometry of Concepts paper) but never formalized or applied as a predictive quantitative model of feature absorption.**

**Gap B: The renormalization group perspective has been applied to deep learning generally but never specifically to the question of which features are "integrated out" (absorbed) vs. retained in SAEs—and never formulated as a testable prediction about absorption severity.**

**Gap C: The predictive coding / free energy principle framework has never been connected to SAE feature absorption. Yet predictive coding explicitly models the suppression of expected (general, parent) features by specific (child) predictions—which is structurally identical to SAE absorption.**

**Gap D: The information bottleneck perspective has been applied to deep network compression generally, but never used to analyze absorption as an IB "failure mode" where fine-grained information is collapsed into coarser representations contrary to the desired IB objective.**

---

## Phase 2: Initial Candidates

### Candidate A: The Competitive Exclusion Theory of Feature Absorption (from Ecology)

- **Source field**: Population ecology, specifically the Lotka-Volterra competitive exclusion model
- **Source principle**: In competitive Lotka-Volterra dynamics, two species sharing a niche cannot coexist. Coexistence requires the competition coefficient matrix **A** (with α_ij = effect of species j on species i) to satisfy: for every pair (i,j), α_ij · α_ji < 1. If this condition fails, one species is driven to zero (competitive exclusion). The depth of the competitive interaction is proportional to niche overlap.
- **Structural correspondence**:
  - **Species** ↔ SAE latent features
  - **Niche** ↔ the set of input tokens/contexts that activate a feature
  - **Niche overlap** ↔ co-activation probability P(f_i = 1, f_j = 1) or decoder cosine similarity cos(d_i, d_j)
  - **Carrying capacity K_i** ↔ feature activation frequency f_i (the "bandwidth" of feature i)
  - **Competition coefficient α_ij** ↔ the degree to which feature j's activation reduces the marginal utility (sparsity gain) of activating feature i
  - **Competitive exclusion of species i** ↔ feature absorption of latent i into latent j
  - **Coexistence condition** ↔ the condition for two features to coexist without absorption: their "competition coefficient" must satisfy α_ij · α_ji < 1, which translates to: the product of their mutual suppression through the sparsity penalty must be subcritical
- **Hypothesis**: Given two SAE latents i (parent/rare) and j (child/common) with niche overlap σ_ij (co-activation rate divided by min activation rate), absorption of i by j occurs if and only if the Lotka-Volterra competition coefficient condition fails: α_ij = σ_ij · (f_j/f_i) > 1 (species j—the more common feature—excludes species i—the rarer feature). This predicts a sharp threshold in σ_ij · (f_j/f_i) space that separates coexistence from absorption.
- **Why not just a metaphor**: The mathematical structure is preserved: both systems involve a set of "entities" (species/features) competing for a shared resource (ecological niche space/activation budget / L0 budget), with interaction strengths that depend on overlap, and dynamics that converge to competitive exclusion when overlap exceeds a threshold. The Lotka-Volterra equilibrium analysis applies to SAE optimization: the absorbing solution is the LV fixed point where one species (feature) has been driven to zero, and the non-absorbing solution is the coexistence equilibrium.
- **Novelty estimate**: 9/10 — The "Geometry of Concepts" paper (2025) casually mentions this analogy in one sentence without formalization. No paper has applied Lotka-Volterra dynamics or derived the coexistence condition for SAE features.

---

### Candidate B: The Renormalization Group Theory of Absorption Severity (from Statistical Physics)

- **Source field**: Statistical physics, specifically Wilson's Renormalization Group and the concept of "integrating out" irrelevant degrees of freedom
- **Source principle**: In the RG framework, a physical system at length scale L is described by an effective Hamiltonian H_eff(L) obtained by integrating out all degrees of freedom at scales smaller than L. Features that are "irrelevant" (in the RG sense: their couplings flow to zero under the RG transformation) are absorbed into the effective description of more relevant (coarser, lower-frequency) features. Universality classes emerge: systems with different microscopic details but the same relevant couplings converge to the same RG fixed point.
- **Structural correspondence**:
  - **RG transformation step** ↔ one level of SAE feature hierarchy
  - **Degrees of freedom at scale L** ↔ SAE features at a given specificity level
  - **Integrating out fast degrees of freedom** ↔ feature absorption (specific/rare features absorbed by general/common features)
  - **Relevant couplings (survive RG flow)** ↔ features that survive and are not absorbed (high-frequency features corresponding to low-frequency SAE activations, i.e., common features)
  - **Irrelevant couplings (flow to zero)** ↔ absorbed features (rare-but-specific features whose information is "irrelevant" under L1 sparsity pressure)
  - **RG fixed point** ↔ the stable SAE solution after absorption has occurred
  - **Universality class** ↔ the "effective feature vocabulary" that all SAE initializations converge to under sparsity pressure
- **Hypothesis**: The severity of feature absorption follows an RG flow equation: features at specificity level k are absorbed at a rate proportional to their "coupling strength" (cosine similarity to more general features at level k-1) divided by their marginal frequency advantage. There exists a set of "fixed-point features" that survive all levels of absorption and form the stable SAE dictionary. Features outside this set are absorbed regardless of SAE width, because they are "irrelevant" in the RG sense under L1 sparsity pressure. Crucially, this predicts which specific features will be absorbed before any SAE is trained.
- **Why not just a metaphor**: The mathematical machinery is isomorphic: both involve iterative elimination of degrees of freedom based on their "coupling strength" to a coarser-scale description, leading to fixed points and universality. The RG flow equations (beta functions) can be written explicitly for the SAE case in terms of cosine similarities, frequency ratios, and L0 budget.
- **Novelty estimate**: 8/10 — RG has been applied to deep learning generally (many papers 2013-2025) but never specifically to the question of which SAE features are absorbed vs. retained. The "universality class" prediction (that all SAE initializations converge to the same fixed-point features under the same L0 budget) is a novel, testable prediction.

---

### Candidate C: Free Energy Principle / Predictive Coding as a Framework for Absorption-Free SAE Design (from Computational Neuroscience)

- **Source field**: Computational neuroscience, specifically Friston's Free Energy Principle (FEP) and predictive coding
- **Source principle**: Under the FEP, the brain minimizes variational free energy F = KL[q(z) || p(z|x)] - log p(x) through hierarchical message passing. Bottom-up connections carry prediction *errors* (what was not predicted by the higher level); top-down connections carry *predictions*. Critically: the brain does NOT suppress prediction errors—it propagates them to update the hierarchical model. Absorption in SAEs corresponds to a "wrong-sign" FEP: the SAE treats the parent feature as a prior that fully explains the child context, so the child feature encodes "error signal" but the parent feature stops firing when the child fires. The FEP design principle that prevents this is **precision-weighted prediction error**: even when a prediction is good, the prediction error is given nonzero precision (weight) if the predicted feature has intrinsic importance. This is formalized as: F = Σ_l π_l · ||ε_l||^2, where ε_l is the prediction error at level l and π_l is its precision.
- **Structural correspondence**:
  - **Hierarchical generative model** ↔ hierarchical feature structure in LLM activations
  - **Prediction error at level l** ↔ parent feature activation that is not explained by child feature
  - **Precision-weighting** ↔ importance weighting in the SAE objective
  - **Absorption** ↔ zero-precision prediction error: the parent feature's signal is treated as having zero importance when the child fires
  - **FEP remedy** ↔ an SAE objective that assigns nonzero precision to parent feature activation even when child is active: L = Σ_l (reconstruction error at level l) + λ · Σ_l ||features_at_level_l||₀ + **μ · Σ_{parent, child} precision_parent · ||x - ŷ_parent||²**
- **Hypothesis**: An SAE trained with precision-weighted prediction error objectives (where parent features have importance weights inversely proportional to child activation frequency) will exhibit significantly lower absorption rates than standard L1/TopK SAEs. The precision weights can be estimated without supervision: π_i ∝ 1 / (1 + Σ_j>i f_j · cos(d_i, d_j)), i.e., the precision is inversely proportional to the degree to which other features already "explain" feature i.
- **Why not just a metaphor**: The FEP is formally equivalent to variational inference (ELBO maximization), and SAEs are formally a specific case of unsupervised representation learning. The FEP framework provides a principled, information-theoretically motivated modification to the SAE objective that directly addresses the root cause of absorption: zero-precision errors at the parent level. The Helmholtz machine architecture (which implements the FEP) is mathematically related to the SAE through the ELBO-free energy equivalence.
- **Novelty estimate**: 8/10 — No paper has connected the FEP/predictive coding literature to SAE feature absorption. The precision-weighting approach maps directly to practical SAE objective modifications that can be tested empirically.

---

## Phase 3: Self-Critique

### Against Candidate A: Competitive Exclusion Theory

- **Shallow analogy attack**: Is the correspondence truly structural, or am I mapping vocabulary?
  - *Counter*: The mathematical structure is preserved. Both systems have (1) a set of entities competing for a shared resource (L0 budget / niche space), (2) interaction strengths that depend on overlap (cosine similarity / niche overlap), (3) dynamics that converge to stable fixed points (absorbed vs. non-absorbed / exclusion vs. coexistence), and (4) a sharp threshold condition (α_ij · α_ji < 1 / frequency ratio threshold). A population ecologist examining the SAE optimization landscape would recognize it as a Lotka-Volterra competitive system. The correspondence is structural, not metaphorical.
  
- **Scale mismatch attack**: Does the LV model operate at the right scale?
  - *Counter*: The LV model is a population-level model that applies whenever multiple entities compete for limited shared resources. SAE latents are exactly such entities: they compete for "activation slots" (L0 budget). The scale is correct. The LV dynamics are scale-free (they depend only on frequency ratios and overlap, not absolute counts).
  
- **Prior transplant check**: Has anyone formally applied LV to SAE absorption?
  - Evidence: The "Geometry of Concepts" paper (PMC 2025) mentions the niche analogy in one sentence, but there is no formal derivation, no coexistence condition analysis, no threshold prediction, and no experimental validation. arXiv search for "Lotka-Volterra sparse autoencoder" returns zero results. arXiv search for "competitive exclusion SAE features" returns zero results. **This transplant has not been formally attempted.**
  
- **Testability attack**: Can we design an experiment that distinguishes "this works because of the LV mechanism" from "this works for mundane reasons"?
  - *Yes*. The LV theory predicts: (1) a specific threshold σ_ij · (f_j/f_i) > 1 separates coexistence from absorption, with a sharp transition; (2) the transition is sharp (not gradual); (3) the "competition coefficient" α_ij = σ_ij · (f_j/f_i) can be computed from activation statistics of any SAE without ground-truth feature labels; (4) α_ij computed before absorption should predict which features will be absorbed after absorption. These predictions are *mechanistically specific* to the LV mechanism and would not be expected from a generic "sparsity suppresses rare features" explanation. Specifically, (4) provides a diagnostic: compute α_ij from SAE activations, predict which features will be absorbed, validate against the Chanin absorption metric.
  
- **Verdict**: **STRONG**

---

### Against Candidate B: Renormalization Group Theory

- **Shallow analogy attack**: Is RG → SAE absorption truly structural?
  - *Partial*. The RG analogy is weaker than LV because: (a) RG involves infinitely many degrees of freedom; SAEs have finite dictionaries; (b) RG integration order is determined by scale (wavelength); SAE absorption order is determined by frequency ratio and cosine similarity—a different organizing principle; (c) RG universality classes are determined by symmetries and dimensionality; SAE "universality" (if it exists) would be determined by different factors.
  - The analogy is partially structural (both involve iterative elimination of degrees of freedom) but the specific mathematical machinery of the RG (beta functions, fixed points, universality classes) does not map cleanly onto the discrete SAE setting.
  
- **Scale mismatch attack**: RG applies to continuous field theories with scale invariance; SAEs are discrete and not scale-invariant.
  - *Valid concern*. The strongest version of the RG correspondence requires scale invariance, which LLM features do not exhibit. A weaker version (sequential coarse-graining without scale invariance) is still valid but loses the power of universality class predictions.
  
- **Prior transplant check**: Has RG been applied to SAE feature absorption?
  - No paper has specifically applied RG to SAE absorption. Many papers (Mehta & Schwab 2014, AAAI 2025, ECT* 2024) apply RG to deep learning generally, but none to SAEs specifically and none to absorption.
  
- **Testability attack**: The "universality class" prediction (all SAEs converge to the same fixed-point features) is testable but complicated by the fact that SAEs are already known to be inconsistent across runs (Song et al. 2025). The more testable prediction is the RG flow direction: features with high cosine similarity to more-frequent features should be absorbed first, following a cascade from fine-grained to coarse-grained.
  
- **Verdict**: **MODERATE** — The analogy has structural elements but is weakened by the discrete, non-scale-invariant SAE setting. The testable prediction about absorption cascade order is valuable, but the full RG machinery doesn't cleanly transplant.

---

### Against Candidate C: Free Energy Principle / Predictive Coding

- **Shallow analogy attack**: Is FEP → SAE truly structural, or just "both involve hierarchical inference"?
  - *Strong correspondence*. The ELBO (SAE training objective when framed variationally) is literally the same mathematical object as the FEP variational free energy. SAEs minimize -ELBO = KL[q(z|x) || p(z)] - E[log p(x|z)] = sparsity penalty + reconstruction error. The FEP minimizes F = KL[q(z) || p(z|x)] + const = same object. The correspondence is mathematical identity, not analogy.
  
- **Scale mismatch attack**: The FEP operates on single-layer neural circuits; SAEs are trained on activation residuals of multi-layer transformers.
  - *Addressed*: The FEP framework is hierarchical and applies at any level of abstraction. The precision-weighting modification is layer-agnostic: it modifies the importance weights in the sparsity penalty, which is defined at the SAE level regardless of the underlying transformer.
  
- **Prior transplant check**: Has FEP been applied to SAE absorption specifically?
  - arXiv search for "free energy principle sparse autoencoder" returns papers on active inference (unrelated to SAE interpretability) but no papers specifically connecting FEP to SAE feature absorption. The connection between ELBO equivalence and the absorption failure mode has not been made.
  
- **Testability attack**: The precision-weighting modification can be implemented in SAELens and tested against the Chanin absorption metric on Gemma Scope SAEs. The diagnostic experiment: train a standard SAE and a precision-weighted SAE on the same activations with the same architecture; compare absorption rates and reconstruction quality.
  - *Concern*: The precision-weighting modification requires estimating precision weights during or before training. If the weights are estimated from the same SAE activations, there is a circular dependency risk. This is addressable by using a hold-out estimation set or by computing precisions from the base LLM activations before SAE training.
  
- **Verdict**: **STRONG** — Mathematical correspondence is exact (ELBO = FEP free energy). The precision-weighting modification has a clear implementation path and testable prediction.

---

## Phase 4: Refinement

### Dropped
**Candidate B (RG Theory)**: The RG analogy is intellectually stimulating but the mathematical correspondence is weaker than LV or FEP. The discrete, non-scale-invariant nature of SAE feature hierarchies means the most powerful RG predictions (universality classes, fixed-point features) do not cleanly apply. The testable prediction from RG (absorption cascade order) is captured more precisely by the LV model. **Dropped.**

### Strengthened Candidates

#### Candidate A Refined: Competitive Exclusion Theory

**Formalized mapping**:

Let the SAE dictionary be D = {d_1, ..., d_K} with decoder vectors d_i ∈ R^n. Let f_i = E[a_i(x)] be the mean activation frequency of feature i. Define the **niche overlap** between features i and j as:

σ_ij = P(a_i > 0, a_j > 0) / min(f_i, f_j)

where P(a_i > 0, a_j > 0) is the co-activation probability. The **competition coefficient** (analogous to the LV α) is:

α_ij = σ_ij · (f_j / f_i)

This is the LV competition coefficient: how much feature j suppresses feature i, relative to feature i's self-suppression. The absorbing condition α_ij > 1 translates to: f_j · P(a_i > 0, a_j > 0) > f_i · min(f_i, f_j).

**Coexistence condition** (for feature i to survive without absorption):
For all j: α_ij · α_ji < 1, i.e., σ_ij < (f_i · f_j) / (max(f_i, f_j))^2 · σ_ij^{-1}

Simplifying for the common case where f_j >> f_i (parent i, child j): coexistence requires σ_ij < f_i/f_j, i.e., the co-activation probability relative to the parent frequency must be less than the ratio of parent-to-child frequency. When the parent is rare (f_i/f_j → 0), coexistence requires near-zero niche overlap — which is impossible if the parent implies the child.

**Diagnostic experiment**: 
1. Compute α_ij for all feature pairs in a trained SAE using activation statistics alone (no ground-truth labels required)
2. For pairs predicted to satisfy the absorbing condition (α_ij > 1), check whether the Chanin absorption metric detects absorption
3. Measure correlation between α_ij and absorption severity across a range of SAE widths and L0 values on Gemma Scope SAEs
4. The LV-specific prediction (not explained by generic "sparsity suppresses rare features"): absorption follows a SHARP threshold in α_ij, not a gradual increase. Distinguishing sharp vs. gradual transition requires a fine-grained sweep of α_ij values.

**Additional prediction from LV theory**: The coexistence condition can be used to design an **absorption-free SAE**: an SAE where the L0 budget is allocated by solving the LV coexistence constraint for each feature pair, rather than by a flat L0 budget. Features with high competition coefficients should receive *separate activation slots* (effectively implementing niche partitioning). This is a novel training objective modification.

#### Candidate C Refined: Precision-Weighted Free Energy SAE

**Formalized mapping**:

Standard SAE training objective: L = ||x - Σ_i a_i · d_i||² + λ · ||a||₁

FEP reinterpretation: This is equivalent to maximizing ELBO under a factorized prior p(z) = Π_i Laplace(0, 1/λ), with posterior q(z|x) = argmin_z L.

The absorption failure mode corresponds to: precision(parent feature i | child feature j is active) → 0. Under FEP, this is "the prediction error from feature i is discounted to zero when feature j fires."

**Precision-weighted objective**:

L_precision = ||x - Σ_i a_i · d_i||² + λ · Σ_i π_i(a) · |a_i|

where π_i(a) = max(1, 1 / (1 + Σ_{j ≠ i} a_j · cos(d_i, d_j))) is the precision weight for feature i—it is 1 when feature i is the only active feature, and increases when highly similar features are also active. This implements "precision-weighted prediction error": the cost of activating feature i increases when i's information is already partially explained by other active features.

Wait—this is backwards. The FEP insight for *preventing absorption* is that the precision of the parent feature's prediction error should be *high* (not zero) even when the child fires. The correct modification is:

L_precision = ||x - Σ_i a_i · d_i||² + λ · Σ_i (1/π_i(a)) · |a_i|

where π_i(a) is high (reducing the sparsity penalty) for features that encode information not already captured by other active features. This implements "every precision-weighted prediction error must be accounted for": rare parent features whose information is not fully captured by their children incur a *reduced* sparsity penalty.

**Practical implementation**: π_i(a) can be estimated as the average reconstruction improvement from adding feature i conditioned on the other active features. This is computable during training via a single forward pass.

**Diagnostic experiment**: Train precision-weighted SAE (precW-SAE) on Gemma 2 2B residual stream layer 12 with the same architecture as standard TopK-SAE; measure Chanin absorption rate, RAVEL, and CE-loss. The FEP-specific prediction: absorption rate should decrease because parent features are given nonzero precision even when children are active. The diagnostic is whether the absorption reduction is *specifically* correlated with the precision weight magnitude—not just with general quality improvement.

### Selected Front-Runner

**Candidate A (Competitive Exclusion / Lotka-Volterra Theory) is the front-runner.**

Reasons:
1. The mathematical correspondence is strongest and most direct
2. The theory generates *quantitative, falsifiable predictions* about absorption thresholds that are specific to the LV mechanism
3. The competition coefficient α_ij is computable from activation statistics without ground-truth labels—enabling unsupervised absorption detection (Gap 7)
4. The coexistence condition suggests a novel, principled training objective modification (LV-regularized SAE)
5. The theory explains both why absorption occurs (competition coefficient > 1) and why wider SAEs can show MORE absorption (more species → higher probability of α_ij > 1 for some pair)

Candidate C (FEP/precision-weighted) is a strong second and could be the basis for a complementary paper on training-time mitigation.

---

## Phase 5: Final Proposal

### Title: Competitive Exclusion in Sparse Autoencoders: A Lotka-Volterra Theory of Feature Absorption

### Source Principle

In population ecology, Gause's competitive exclusion principle (formalized by Lotka-Volterra equations) states that two species competing for the same ecological niche cannot stably coexist. The stability of coexistence depends on the **competition matrix** α, where α_ij measures how much species j suppresses species i relative to i's self-suppression. Coexistence is stable if and only if α_ij · α_ji < 1 for all competing pairs. When α_ij > 1 (species j strongly outcompetes species i), species i is driven to extinction—regardless of its intrinsic carrying capacity or fitness.

The competitive dynamics are not merely metaphorical—they are characterized by exact fixed-point conditions, phase transitions at the coexistence boundary, and known results about how niche partitioning (reducing σ_ij) enables coexistence.

### Structural Correspondence

| Population Ecology | SAE Feature Absorption |
|---|---|
| Species i, j | SAE latents (dictionary atoms) i, j |
| Ecological niche | Set of input tokens/contexts that activate the feature |
| Niche overlap σ_ij | Co-activation rate normalized by rarer feature's frequency |
| Carrying capacity K_i | Mean activation frequency f_i |
| Competition coefficient α_ij | σ_ij · (f_j / f_i): how much j suppresses i per i's self-limit |
| Competitive exclusion of i by j | Feature absorption: i fails to fire because j is active |
| Coexistence condition α_ij · α_ji < 1 | Both features can be stably represented in the SAE |
| Niche partitioning (reduce overlap) | Orthogonality regularization (OrtSAE), or decoder orthogonalization |
| Carrying capacity allocation | L0 budget allocation across features |
| Lotka-Volterra fixed point | Stable SAE activation pattern after training |

The formal mapping: For SAE latents i and j, define f_i = E[a_i(x) > 0], σ_ij = P(a_i > 0, a_j > 0) / min(f_i, f_j), α_ij = σ_ij · (f_j / f_i). The LV coexistence condition α_ij · α_ji < 1 simplifies to:

σ_ij² < (f_i / f_j) · (f_j / f_i) = 1, i.e., σ_ij < 1

Since σ_ij is normalized to be ≤ 1, coexistence is possible whenever niche overlap is subcritical (σ_ij < 1). When the parent strictly implies the child (σ_ij → 1), absorption is certain. When co-activation is less than maximal (σ_ij < 1), coexistence is possible—subject to the full LV condition involving carrying capacities (frequency ratio f_j/f_i).

### Hypothesis

The competitive exclusion condition provides a **universal absorption predictor**: given two SAE latents i and j, feature i is absorbed by feature j if and only if α_ij = σ_ij · (f_j / f_i) > threshold τ ≈ 1, where σ_ij and f_i are computable from SAE activation statistics.

**Specific predictions**:
1. The absorption rate of feature i (probability that a token activating i is misclassified due to absorption) is a monotone function of max_j α_ij.
2. The transition from non-absorption to absorption as α_ij increases is *sharp* (discontinuous or steep), not gradual—consistent with LV competitive exclusion being an all-or-nothing transition near the coexistence boundary.
3. Features with α_ij > τ computed from a partially-trained SAE (at 10% of training steps) will predict final absorption rates at the end of training with Spearman ρ ≥ 0.7.
4. Wider SAEs (more latents) exhibit higher absorption because they have more latents—increasing the probability that some pair (i,j) has α_ij > 1. This explains the Chanin et al. paradox.
5. Niche partitioning (orthogonality regularization as in OrtSAE) reduces absorption by reducing σ_ij, directly satisfying the LV coexistence condition.

### Method: LV-Absorption Analysis Framework

**Training-free analysis (matches project constraints)**:
1. Load a pretrained SAE (e.g., Gemma Scope 2B layer 12 width 65k)
2. Run inference on a corpus (e.g., 10M tokens of Pile) to obtain activation statistics
3. Compute for all feature pairs (i,j) above a frequency threshold: f_i, f_j, σ_ij, α_ij
4. Rank pairs by α_ij; identify pairs above threshold τ = 1 as predicted-absorbed
5. Validate against Chanin absorption metric (sae-spelling toolkit) for letters in the predicted-absorbed set
6. Measure Spearman correlation between α_ij and absorption rate

**Novel metric derivation**:
The competition coefficient α_ij is a fully unsupervised absorption predictor—it requires no ground-truth feature labels, no probes, no in-context learning. This addresses Gap 7 (no metric for absorption without known probe directions).

**Niche partitioning analysis**:
Compute how OrtSAE's orthogonality penalty affects σ_ij distributions; test whether features with σ_ij reduced below 1 by orthogonalization show reduced absorption.

### Diagnostic Experiment

**The LV-specific diagnostic** (what distinguishes this from a generic "frequent features suppress rare features" story):

The LV theory predicts a SHARP threshold in α_ij. Specifically:
- Plot absorption rate as a function of α_ij in bins of width 0.1
- LV prediction: absorption rate should show a step-function-like increase near α_ij = 1 (sharp transition)
- Alternative hypothesis (generic sparsity suppression): absorption rate should increase *smoothly* and monotonically with α_ij, without a sharp threshold

If the empirical distribution shows a sharp transition near α_ij = 1 (adjusting for measurement noise), this confirms the LV mechanism is load-bearing. If the transition is gradual, the competition coefficient has predictive value but the LV mechanism is not specific.

**Additional diagnostic** (LV-specific to width paradox):
- For SAEs of width K = 4096, 16384, 65536 (Gemma Scope variants), compute the distribution of max_j α_ij for each feature i
- LV prediction: the fraction of features with max_j α_ij > 1 should *increase* with K (more latents → more competitors → more absorptions)
- This explains the width paradox while providing a testable quantitative prediction about the rate of increase

### Experimental Plan

**Pilot (10-15 minutes)**:
- Load Gemma Scope 2B layer 12 width 16384 SAE
- Compute activation statistics on 1M tokens
- Compute α_ij for all pairs above frequency threshold 0.001 (should yield ~10k pairs)
- Rank by α_ij; check top 100 against Neuronpedia feature descriptions to verify the predicted-absorbed features are indeed "parent" concepts (e.g., first-letter features) while absorbers are "child" concepts
- Expected runtime: 15 minutes on 1 GPU

**Main experiment (< 1 hour)**:
- Compare three SAE widths (4096, 16384, 65536) on Gemma 2B layer 12
- For each: compute α_ij distribution, run sae-spelling absorption metric on first-letter features
- Correlate α_ij with absorption rate; test for sharp vs. gradual threshold
- Expected runtime: 20 minutes per width × 3 = ~1 hour

**Extended analysis (cross-domain, ~2 hours)**:
- Extend sae-spelling to entity type hierarchies (using knowledge graph entity type data as feature hierarchy)
- Compute α_ij for entity-level features; test whether LV threshold predicts absorption in this richer domain
- Tests whether the LV theory generalizes beyond first-letter features (Gap 2)

### Risk Assessment

1. **Risk: α_ij correlation is strong but threshold is not sharp.** If the threshold is gradual, the LV model is predictively useful but not mechanistically specific. Mitigation: the predictive value of α_ij alone (regardless of sharpness) is novel and publishable as an unsupervised absorption detector.

2. **Risk: The LV coexistence condition gives τ ≠ 1.** The formal derivation assumes a precise LV form; the actual SAE optimization may produce an effective threshold τ < 1 or τ > 1 depending on the interaction details. Mitigation: fit τ empirically as a single free parameter.

3. **Risk: Absorption in wider SAEs is not well-explained by increased number of high-α pairs.** If the width paradox has a different cause (e.g., distributed absorption across many weak α pairs), the LV model needs extension to multi-species dynamics. The multi-species LV model is available and not significantly more complex.

4. **Risk: The analysis is observation-only (training-free).** The project constraints require training-free analysis—we can correlate α_ij with absorption but cannot test the LV-regularized SAE objective without training. Mitigation: (a) validate the prediction direction with existing SAE variants (OrtSAE reduces σ_ij → reduces α_ij → reduces absorption, as predicted); (b) propose LV-regularized training as future work.

### Novelty Claim

The specific cross-disciplinary insight: **SAE feature absorption is formally equivalent to competitive exclusion in Lotka-Volterra ecology.** This provides:

1. A **quantitative, unsupervised absorption predictor** (the competition coefficient α_ij) that does not require known probe directions—addressing an open problem (Gap 7) that existing approaches have not solved
2. An **explanation for the width paradox** (wider SAEs have higher absorption) that no existing theory provides: more latents → more pairs with α_ij > 1 → more absorptions
3. A **design principle for absorption-free SAEs** (niche partitioning → decoder orthogonalization) that is derived from first principles rather than heuristics
4. A **phase transition prediction** (sharp vs. gradual threshold at α_ij ≈ 1) that is specific to the LV mechanism and distinguishes this theory from generic "rare features get suppressed" stories

Evidence of novelty: No arXiv paper contains "Lotka-Volterra" AND "sparse autoencoder." The one paper that mentions the niche analogy (Geometry of Concepts, 2025) does so in a single sentence without formalization or experimental validation.

---

## Appendix: Eliminated Candidates and Summary Notes

### Eliminated: Renormalization Group Theory
Structural correspondence is weaker than LV due to discrete, non-scale-invariant SAE setting. Key predictions (universality classes, absorption cascade order) are less precise than LV. The insight that "integrating out irrelevant degrees of freedom" = absorption is intellectually valuable but does not generate unique testable predictions beyond what the LV model already captures.

### Retained as Secondary: Free Energy Principle / Predictive Coding
The ELBO equivalence makes this mathematically exact rather than analogical. The precision-weighted SAE objective is a concrete training-time modification. Recommended as a parallel research direction for mitigation rather than analysis. The precision-weighting modification (reduce sparsity penalty for features that provide unique information not already covered by active features) is complementary to the LV analysis and could be tested in a short training experiment if training-free constraints are relaxed.

### Note on Ecological "Paradox of the Plankton"
The ecological paradox of the plankton (many species coexisting on few resources, violating the competitive exclusion principle) has known explanations: spatial heterogeneity, temporal fluctuations, multi-resource dynamics. The SAE analogue would be: why do some SAEs manage to represent both parent and child features (not fully absorbing)? The LV framework predicts this happens when σ_ij < 1 (imperfect niche overlap), which occurs when the parent and child are not in strict hierarchical implication. This is testable.

Sources consulted:
- [Competitive Lotka-Volterra equations — Wikipedia](https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations)
- [Niche Overlap and Hopfield-like Interactions — arXiv:2301.11703](https://arxiv.org/html/2301.11703)
- [Feature-Selective Inhibitory Mechanisms — Scientific Reports 2025](https://www.nature.com/articles/s41598-025-28227-8)
- [Predictive Coding under the Free Energy Principle — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- [The Geometry of Concepts: SAE Feature Structure — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12025678/)
- [RG Framework for Scale-Invariant Feature Learning — AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35269)
- [Information Bottleneck Theory and Applications — MDPI Entropy 2024](https://www.mdpi.com/1099-4300/26/3/187)
- [A is for Absorption — arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- [Competitive exclusion principle — Wikipedia](https://en.wikipedia.org/wiki/Competitive_exclusion_principle)
- [Multiresolution Analysis — PyWavelets Documentation](https://pywavelets.readthedocs.io/en/latest/ref/mra.html)

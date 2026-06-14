# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Olshausen & Field (1996)** — "Emergence of simple-cell receptive field properties by learning a sparse code for natural images" (*Nature*, 381, 607-609).
   - **Key principle**: Maximizing sparseness in neural representations of natural images is sufficient to account for localized, oriented, bandpass V1 simple cell receptive fields.
   - **Relevance to SAE absorption**: The overcomplete basis extension (Olshausen & Field, 1997) explored non-orthogonal, non-linearly independent basis functions where sparsification recruits only necessary basis functions—creating deviations from linearity that parallel how absorbed SAE features fail to fire on arbitrary positive examples.

2. **Quiroga et al. tradition + 2025 synthesis** — The "grandmother cell" debate has evolved from a binary opposition to a hybrid view.
   - **Key principle**: The brain flexibly balances "sparse, privileged axes" with distributed representations depending on functional domain (motor control vs. visual areas).
   - **Relevance to SAE absorption**: The modern consensus of "recoverable sparsity"—sparse codes embedded in distributed representations—directly parallels the SAE project's goal of extracting monosemantic features from polysemantic activations. The absorption phenomenon may reflect a failure mode where this balance tips too far toward distributed encoding.

3. **Bricken (2025)** — "Sparse Representations in Artificial and Biological Neural Networks" (Harvard dissertation).
   - **Key principle**: Sparsity is a unifying principle for both biological and artificial intelligence, connecting sparse distributed memory (SDM), attention mechanisms, cerebellar function, and sparse coding.
   - **Relevance**: Establishes that SAEs are not merely engineering tools but probes into a fundamental computational principle shared across intelligent systems.

4. **Predictive Coding / Free Energy Principle (Friston, 2005-present)**.
   - **Key principle**: Neural systems minimize variational free energy through hierarchical predictive coding, where prediction errors drive learning.
   - **Relevance**: The SAE reconstruction loss can be reframed as a prediction error, and sparsity as a complexity penalty—making the SAE objective a special case of free energy minimization. Feature absorption may represent a failure of the "generative model" (the SAE decoder) to correctly infer causes.

#### Physics / Information Theory / Statistical Mechanics

5. **Donoho & Tanner (2005, 2009)** — Phase transition in sparse recovery via L1 minimization.
   - **Key principle**: Sparse recovery exhibits a sharp phase transition in the (delta, rho) plane, with a critical boundary separating success and failure regions.
   - **Relevance to SAE absorption**: SAEs perform sparse recovery from overcomplete dictionaries. The absorption phenomenon may correspond to crossing a phase boundary where the sparse recovery problem becomes ill-posed due to feature correlations (hierarchical co-occurrence).

6. **Replica Method for Compressed Sensing** — Kabashima (2011), Krzakala et al., Zdeborova & Krzakala (2016).
   - **Key principle**: The replica method from statistical physics predicts phase transitions in sparse recovery, revealing information-theoretic, algorithmic, and metastable phases.
   - **Relevance**: Could predict absorption as an algorithmic phase where gradient descent gets trapped in metastable states (absorbed representations) rather than reaching the information-theoretic optimum (clean feature separation).

7. **Singular Learning Theory (Watanabe, 2009; Murfet, Carroll et al., 2020-present)**.
   - **Key principle**: Neural networks are singular statistical models where the Fisher information matrix degenerates. Free energy asymptotics: F_n = nL(w*) + lambda log n + ..., where lambda (real log canonical threshold) measures model complexity.
   - **Relevance to SAE absorption**: Chen et al. (2023/2024) applied SLT to Anthropic's Toy Model of Superposition, discovering that regular k-gons are critical points and that Bayesian phase transitions occur as training sample size increases. The absorption-hedging trade-off may correspond to transitions between competing basins in the free energy landscape.

8. **Information Bottleneck Theory (Tishby, 2015; Saxe et al., 2018; Wu & Fischer, 2019)**.
   - **Key principle**: Deep networks optimize the trade-off between compression (I(X;Z)) and prediction (I(Y;Z)). Wu & Fischer identified a critical beta threshold where IB "turns on."
   - **Relevance**: The SAE's sparsity-reconstruction trade-off is an IB problem. Feature absorption may occur when the system operates below a critical "beta" (sparsity penalty too weak) or above a critical threshold (sparsity too strong, causing collapse into specific latents).

9. **"Superposition as Lossy Compression" (Bereska et al., 2025)**.
   - **Key principle**: Shannon entropy measures effective features in SAE activations; superposition psi = F/N ratio of virtual to physical neurons. Detects "abundance" vs. "scarcity" regimes with phase-like transitions during grokking.
   - **Relevance**: Directly connects SAE analysis to information-theoretic phase transitions.

#### Biology / Evolution / Ecology

10. **Competitive Exclusion Principle (Gause, 1934; Hardin, 1960)**.
    - **Key principle**: Two species competing for the exact same limiting resource cannot coexist stably. Niche partitioning (spatial, temporal, dietary) enables coexistence.
    - **Relevance to SAE absorption**: SAE latents competing for activation on the same inputs exhibit competitive exclusion. Specific latents outcompete general ones when features co-occur, driving the general latent toward "extinction" (absorption). Matryoshka SAEs may function like niche differentiation, partitioning the representation space.

11. **Artificial Immune Systems — Negative Selection Algorithm (Forrest et al., 1994; de Castro & Zuben, 2000)**.
    - **Key principle**: Immature T-cells recognizing "self" are eliminated; only non-self-recognizing cells survive. Computational analog: randomly generated detectors matching normal data are discarded; non-matching detectors detect anomalies.
    - **Relevance to SAE absorption**: The NSA is a formally defined one-class classification method. Could inspire an "absorption detection" mechanism: train detectors on known non-absorbed feature activation patterns, then identify anomalies—latents that deviate from expected behavior—as potential absorption cases.

12. **Clonal Selection Algorithm (CLONALG)**.
    - **Key principle**: When a lymphocyte binds to an antigen, it proliferates (clonal expansion), undergoes affinity maturation (random mutations), and only highest-affinity clones survive.
    - **Relevance**: The "activation lottery" in BatchTopK SAEs (where rare high-magnitude features outcompete consistent features) parallels clonal selection. Could inspire training algorithms where latents "compete" and "evolve" their selectivity.

13. **Ecological Niche Theory (Hutchinson, 1957; MacArthur & Levins, 1967)**.
    - **Key principle**: Species occupy n-dimensional hypervolumes in resource space. Limiting similarity determines how close niches can be before competitive exclusion occurs.
    - **Relevance**: SAE latents occupy "niches" in activation space. Absorption occurs when niche overlap exceeds the limiting similarity threshold. The "activation density" of a feature corresponds to its niche breadth.

#### Signal Processing / Source Separation

14. **Independent Component Analysis (ICA) and the Cocktail Party Problem**.
    - **Key principle**: Separate mixed signals into statistically independent sources.
    - **Relevance**: SAEs perform nonlinear ICA on neural network activations. Feature absorption corresponds to ICA's failure mode when sources are not truly independent (hierarchical co-occurrence violates independence). The "absorption" of a general feature into a specific latent parallels how ICA can fail when one source is a mixture of others.

15. **"Sparse Representations for the Cocktail Party Problem" (2006)**.
    - **Key principle**: Sparsity-based methods can separate audio sources.
    - **Relevance**: Demonstrates that sparsity constraints alone are insufficient for clean source separation when sources have structured dependencies—directly parallel to SAE absorption.

### Cross-Disciplinary Gaps

Where transplants have NOT been tried:

- **No paper** explicitly frames SAE feature absorption using ecological competition models (Lotka-Volterra, niche theory, competitive exclusion).
- **No paper** applies the replica method from statistical physics to predict absorption phase transitions in SAEs.
- **No paper** connects Singular Learning Theory's free energy landscape to the absorption-hedging trade-off.
- **No paper** uses immune system negative selection for unsupervised absorption detection.
- **No paper** applies information bottleneck phase transition theory to predict when absorption occurs.

---

## Phase 2: Initial Candidates

### Candidate A: Competitive Exclusion Theory for Feature Absorption (from Ecology)

- **Source principle**: The competitive exclusion principle (Gause's Law) states that complete competitors cannot coexist. Niche partitioning enables coexistence by reducing interspecific competition below intraspecific competition.
- **Structural correspondence**:
  - Species <-> SAE latents
  - Resource <-> activation "territory" (input examples where a latent could fire)
  - Niche overlap <-> feature correlation / co-occurrence
  - Competitive exclusion <-> feature absorption (general latent loses to specific)
  - Niche differentiation <-> Matryoshka hierarchy (partitioning representation space)
  - Carrying capacity <-> dictionary size (total latent capacity)
- **Hypothesis**: Feature absorption occurs when the "niche overlap" between a general feature and its specific child exceeds a critical threshold determined by the sparsity penalty (selection pressure). Matryoshka SAEs reduce absorption by creating explicit niche partitioning (hierarchical differentiation).
- **Why not just a metaphor**: The mathematical structure is identical. Lotka-Volterra competition equations describe how two species competing for the same resource cannot coexist. The SAE optimization dynamics (gradient descent on sparsity + reconstruction loss) create mathematically equivalent competition for activation mass. Chanin et al.'s proof that absorption decreases loss is formally equivalent to proving that competitive exclusion increases fitness.
- **Novelty estimate**: 8/10. While ecological analogies to neural networks exist (e.g., neural Darwinism), no paper has explicitly applied competitive exclusion theory to SAE feature dynamics.

### Candidate B: Replica Symmetry Breaking as Absorption Phase Transition (from Statistical Physics)

- **Source principle**: In spin glass theory, replica symmetry breaking (RSB) describes how a system's phase space fractures into disconnected valleys as temperature decreases. The replica method predicts phase transitions in sparse recovery, with distinct information-theoretic, algorithmic, and metastable phases.
- **Structural correspondence**:
  - Spin configurations <-> SAE latent activation patterns
  - Temperature <-> inverse sparsity penalty strength (1/beta ~ sparsity constraint)
  - External field h <-> reconstruction error gradient
  - Free energy F = E - TS <-> SAE loss = MSE + lambda * L_sparsity
  - Replica symmetry <-> unique optimal feature decomposition
  - Replica symmetry breaking <-> multiple equivalent decompositions (absorption vs. clean separation as degenerate minima)
  - Metastable states <-> absorbed representations (locally optimal but globally suboptimal)
  - Ground state <-> clean feature separation (globally optimal)
  - Quenched disorder <-> hierarchical feature co-occurrence structure
  - Phase transition <-> absorption-hedging trade-off boundary
- **Hypothesis**: Feature absorption corresponds to the SAE optimizer (gradient descent) getting trapped in a metastable state of the free energy landscape, while the information-theoretic optimum (clean feature separation) lies in a different valley. The absorption-hedging trade-off is a first-order phase transition between two competing basins.
- **Why not just a metaphor**: The replica method provides exact (in the thermodynamic limit) predictions for the phase diagram of sparse recovery. The order parameters (overlap between replicas, magnetization) have direct analogs in SAE analysis (feature correlation matrices, activation probabilities). Chen et al. (2023/2024) already applied related statistical mechanics (Singular Learning Theory) to the Toy Model of Superposition, finding exact phase transitions.
- **Novelty estimate**: 9/10. While SLT has been applied to toy models, no paper has used the replica method or RSB framework to analyze feature absorption in practical SAEs.

### Candidate C: Negative Selection for Unsupervised Absorption Detection (from Immunology)

- **Source principle**: The immune system's negative selection eliminates T-cells that recognize "self" proteins, leaving only detectors for "non-self" (anomalies). The V-detector algorithm achieves efficient coverage with variable-radius detectors.
- **Structural correspondence**:
  - Self proteins <-> correctly decomposed features (non-absorbed)
  - Non-self / anomalies <-> absorbed feature patterns
  - T-cell receptors <-> absorption detectors (classifiers)
  - Thymus screening <-> training on known non-absorbed features
  - Variable detector radius <-> adaptive threshold for absorption detection
  - Clonal expansion <-> amplifying detectors for suspected absorption cases
- **Hypothesis**: An immune-inspired negative selection algorithm can detect absorption without ground-truth parent features. By training detectors on "normal" (non-absorbed) feature activation patterns, the algorithm flags anomalies—latents that deviate from expected behavior—as potential absorption cases.
- **Why not just a metaphor**: The negative selection algorithm is a formally defined one-class classification method with proven properties. Its computational structure (hypersphere detectors in feature space) maps directly onto detecting outliers in SAE activation co-occurrence patterns. The V-detector's variable-radius approach directly addresses the challenge of setting absorption thresholds.
- **Novelty estimate**: 7/10. Immune-inspired algorithms are established in anomaly detection, but applying them specifically to SAE absorption detection is unexplored.

---

## Phase 3: Self-Critique

### Against Candidate A (Ecological Competition)

- **Shallow analogy attack**: Is this just mapping vocabulary? The Lotka-Volterra equations describe population dynamics over time, while SAE optimization is a static (or batch) gradient descent. The temporal dimension is missing. However, the competitive dynamics for activation mass at each training step are structurally similar to instantaneous competition rates.
- **Scale mismatch attack**: Ecological competition operates at population scale (many individuals). SAE latents are single "individuals" (one weight vector each). However, the activation mass across the training batch serves as the "population" being competed over.
- **Prior transplant check**: "Neural Darwinism" (Edelman, 1987) applied selectionist principles to neural development. "Winner-take-all" autoencoders (Makhzani & Frey, 2014) explicitly implement competition. However, no paper applies ecological niche theory or competitive exclusion to SAE feature absorption specifically.
- **Testability attack**: Can we distinguish "absorption occurs due to competitive exclusion" from "absorption occurs due to simple correlation"? The diagnostic experiment would vary the "carrying capacity" (dictionary size) and measure whether absorption follows the predicted competition curve. If absorption rate vs. dictionary size follows a Lotka-Volterra-like saturation curve rather than a simple linear correlation trend, the analogy is load-bearing.
- **Verdict**: MODERATE. The structural correspondence is strong but requires careful experimental design to distinguish from simpler explanations.

### Against Candidate B (Replica Symmetry Breaking)

- **Shallow analogy attack**: The replica method is a powerful but non-rigorous technique. Its predictions are exact only in the thermodynamic limit (infinite system size). SAEs are finite systems. However, SAEs with thousands of latents may be large enough for mean-field approximations.
- **Scale mismatch attack**: Spin glass theory deals with disordered systems with quenched randomness. SAE feature correlations are structured (hierarchical co-occurrence), not random. This is a genuine concern—the replica method's assumptions about random matrix ensembles may not apply to linguistic feature hierarchies.
- **Prior transplant check**: The replica method has been applied to compressed sensing (Kabashima, 2011) and dictionary learning (Krakala et al.). Chen et al. (2023/2024) applied Singular Learning Theory to the Toy Model of Superposition. However, no paper applies RSB specifically to SAE absorption.
- **Testability attack**: The key prediction is that absorption corresponds to a metastable state. Can we test this? Yes—by using simulated annealing or different optimization trajectories to see if some initializations escape absorption while others get trapped. If multiple random initializations converge to different absorption patterns (broken symmetry), this supports the RSB analogy.
- **Verdict**: STRONG. The mathematical framework is well-developed and directly applicable. The main risk is that linguistic feature structures violate the randomness assumptions of spin glass theory.

### Against Candidate C (Negative Selection)

- **Shallow analogy attack**: Negative selection is fundamentally about detecting anomalies (non-self). Absorption is not an anomaly in the traditional sense—it's a systematic bias in the SAE decomposition. However, from the perspective of "expected feature behavior," absorbed features are anomalous (they fail to fire where they should).
- **Scale mismatch attack**: The immune system has ~10^9 unique receptors. SAEs have ~10^4 latents. The repertoire size difference is significant but may not invalidate the analogy if the detection principle scales.
- **Prior transplant check**: Negative selection algorithms are widely used in intrusion detection and anomaly detection. Clonal selection algorithms have been applied to feature selection (MDPI Electronics, 2024). However, no paper applies these to SAE absorption detection.
- **Testability attack**: Can we distinguish "absorption detected by immune algorithm" from "absorption detected by simple thresholding"? The diagnostic experiment would compare negative selection detection against a simple co-occurrence threshold baseline. If the immune-inspired method achieves higher precision/recall on synthetic absorption cases, the transplant works.
- **Verdict**: MODERATE. The analogy is plausible but may not outperform simpler statistical methods. The value may lie in the theoretical framework rather than raw detection performance.

---

## Phase 4: Refinement

### Dropped
- **Candidate C (Negative Selection)** is dropped as the front-runner. While interesting, it is more of an engineering tool than a theoretical insight. It may be revived as a component of the experimental plan.

### Strengthened
- **Candidate A (Ecological Competition)** is strengthened by formalizing the structural correspondence:
  - Let f_g be a general feature and f_s be a specific child feature (f_s subset of f_g).
  - Let a_g and a_s be their activation probabilities.
  - The "niche overlap" is the co-occurrence probability P(f_g, f_s).
  - The "competition coefficient" alpha_gs = P(f_g, f_s) / P(f_g)^2 (normalized co-occurrence).
  - Absorption occurs when alpha_gs > 1 / (1 + lambda), where lambda is the effective sparsity penalty.
  - This predicts a critical co-occurrence threshold for absorption, testable with synthetic data.

- **Candidate B (Replica Symmetry Breaking)** is strengthened by connecting to existing SLT work:
  - Chen et al. (2023/2024) showed that the Toy Model of Superposition exhibits Bayesian phase transitions.
  - The free energy expansion F_n = nL(w*) + lambda log n predicts that different minima (clean separation vs. absorption) compete based on their learning coefficient lambda.
  - If the absorbed state has lower lambda (simpler) but higher loss, and the clean state has higher lambda (more complex) but lower loss, the system may undergo a phase transition as n (training data) increases.
  - This directly connects to the finding that SAEs interpret randomly initialized transformers (Cui et al., 2025)—the "absorbed" state may be the simpler, more probable one under the model's prior.

### Selected Front-Runner
**Candidate B: Replica Symmetry Breaking as Absorption Phase Transition**, with Candidate A as a complementary perspective.

Rationale:
1. The statistical physics framework provides the most rigorous mathematical foundation.
2. It connects directly to existing theoretical work on SAEs (SLT, Toy Models of Superposition).
3. It makes falsifiable predictions about metastable states and phase transitions.
4. The ecological competition analogy (Candidate A) provides intuitive interpretation but lacks the predictive power of the physics framework.

---

## Phase 5: Final Proposal

### Title
"Feature Absorption as a Phase Transition: A Statistical Mechanics Framework for Understanding Sparse Autoencoder Failure Modes"

### Source Principle
**Replica symmetry breaking in disordered systems** (spin glass theory). In spin glasses, as temperature decreases, the system's phase space fractures into exponentially many disconnected valleys (replica symmetry breaking). The replica method from statistical physics predicts phase transitions between distinct phases: paramagnetic (high temperature, symmetric), spin glass (low temperature, broken symmetry), and mixed phases. In sparse recovery, the replica method predicts three phases: (1) perfect recovery, (2) algorithmically recoverable but not perfect, and (3) unrecoverable.

### Structural Correspondence

| Spin Glass / Sparse Recovery | SAE Feature Absorption |
|------------------------------|------------------------|
| Spin configuration | Latent activation pattern |
| Temperature T | Inverse sparsity penalty (1/lambda) |
| External field h | Reconstruction error gradient |
| Free energy F = E - TS | SAE loss = MSE + lambda * L_sparsity |
| Replica symmetry | Unique optimal feature decomposition |
| Replica symmetry breaking | Multiple degenerate decompositions (absorption vs. clean) |
| Metastable state | Absorbed representation (locally optimal) |
| Ground state | Clean feature separation (globally optimal) |
| Quenched disorder | Hierarchical feature co-occurrence structure |
| Phase transition | Absorption-hedging trade-off boundary |

The key insight is that **feature absorption is not a bug but a phase**—a metastable state that gradient descent converges to because it is locally optimal (lower sparsity loss) even though it is globally suboptimal for interpretability.

### Hypothesis
**H1 (Phase Transition)**: Feature absorption rate undergoes a sharp phase transition as a function of the sparsity-reconstruction trade-off parameter. Below a critical sparsity penalty, features are cleanly separated (ordered phase). Above the critical penalty, the system enters an "absorbed" phase where general features are absorbed into specific latents.

**H2 (Metastability)**: The absorbed state is a metastable local minimum of the SAE loss landscape. Different random initializations and optimization trajectories converge to different absorption patterns, demonstrating replica symmetry breaking.

**H3 (Universality)**: The phase transition boundary follows a universal scaling law depending on the feature correlation structure (hierarchical co-occurrence), analogous to how spin glass phase boundaries depend on disorder strength.

### Method

1. **Construct a controlled toy model** (extending Elhage et al.'s Toy Model of Superposition):
   - Input: sparse features with known hierarchical co-occurrence structure
   - SAE: single-layer autoencoder with varying sparsity penalty
   - Measure: absorption rate as a function of sparsity penalty and feature correlation

2. **Apply the replica method** (or its modern variant, approximate message passing with state evolution):
   - Derive the phase diagram for the toy model
   - Predict the critical sparsity penalty where absorption onset occurs
   - Compare with empirical results from gradient descent training

3. **Test on real SAEs** (GemmaScope, GPT-2 Small SAEs):
   - Use Chanin et al.'s absorption metric to measure absorption rates
   - Vary effective sparsity (by comparing different SAE architectures)
   - Test whether absorption follows the predicted phase transition curve

4. **Demonstrate metastability**:
   - Train multiple SAEs with different random initializations
   - Show that different runs converge to different absorption patterns
   - Use simulated annealing to escape metastable states

### Diagnostic Experiment

The key test that confirms the analogy is load-bearing:

**Experiment: Metastability via Simulated Annealing**
- Train a standard SAE on synthetic hierarchical features until convergence (expected: absorbed state).
- Apply simulated annealing: gradually increase temperature (noise in gradients) and then cool.
- Measure: Does the system escape the absorbed state and converge to a cleaner separation?
- Prediction (if analogy is correct): Annealing reduces absorption rate because thermal noise allows escape from metastable states.
- Control: Standard gradient descent with same total training steps (no annealing) should maintain high absorption.
- If annealing does NOT reduce absorption, the metastability hypothesis is falsified.

### Experimental Plan

**Phase 1: Toy Model (15 minutes)**
- Extend Elhage's Toy Model with hierarchical co-occurrence
- Train with varying sparsity penalty
- Plot absorption rate vs. sparsity penalty
- Look for sharp phase transition

**Phase 2: Replica Method Analysis (1 hour)**
- Derive replica symmetric equations for the toy model
- Compute phase boundary analytically
- Compare with empirical phase transition point

**Phase 3: Real SAE Validation (1 hour)**
- Load GemmaScope SAEs with different sparsity levels
- Measure Chanin et al. absorption metric
- Test if absorption follows predicted scaling

**Phase 4: Metastability Test (30 minutes)**
- Multiple random initializations
- Simulated annealing vs. standard training
- Measure absorption pattern diversity

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Linguistic feature structure violates randomness assumptions | Medium | High | Use synthetic data with controlled correlation structure; test on toy model first |
| Phase transition is smooth, not sharp | Medium | High | Report as negative result; explore whether finite-size effects smooth the transition |
| Replica method predictions don't match empirical results | Medium | Medium | Fall back to phenomenological description; still publishable as exploratory work |
| Simulated annealing doesn't reduce absorption | Low | High | Falsifies metastability hypothesis; paper pivots to "absorption is not metastable" |
| Computational cost of replica method too high | Low | Low | Use AMP/state evolution as computationally tractable alternative |

### Novelty Claim

The specific cross-disciplinary insight is: **Feature absorption in sparse autoencoders can be understood as a phase transition in a disordered system, analogous to replica symmetry breaking in spin glasses.** This insight has not been applied to SAEs before. While Chen et al. (2023/2024) applied Singular Learning Theory to the Toy Model of Superposition, they did not connect to the replica method or frame absorption as a phase transition. The ecological competition analogy (Candidate A) provides an intuitive narrative, but the statistical physics framework provides quantitative predictions.

**Evidence this hasn't been applied before**:
1. No paper in the SAE literature uses terms like "phase transition," "replica," "metastable," or "free energy landscape" to describe absorption.
2. The absorption literature (Chanin et al., Bussmann et al., Korznikov et al.) focuses on architectural solutions, not theoretical characterization.
3. The statistical physics of sparse recovery (Donoho-Tanner, Kabashima, Krzakala) has not been connected to SAE feature dynamics.

### Complementary Perspective: Ecological Competition

While the statistical physics framework is the front-runner, the ecological competition analogy provides valuable intuition:
- **Niche partitioning** explains why Matryoshka SAEs reduce absorption (hierarchical differentiation)
- **Competitive exclusion** explains why absorption worsens with higher sparsity (stronger selection pressure)
- **Character displacement** suggests that feature decorrelation (OrtSAE's cosine penalty) reduces absorption by increasing niche separation

These intuitions can guide architectural design even if the formal statistical physics predictions are difficult to compute for real SAEs.

---

## Sources

- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381(6583), 607-609.
- Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an overcomplete basis set: A strategy employed by V1? *Vision Research*, 37(23), 3311-3325.
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *Proc. 37th Allerton Conference*.
- Saxe, A. M., et al. (2018). On the Information Bottleneck Theory of Deep Learning. *ICLR*.
- Wu, T., Fischer, I., Chuang, I., & Tegmark, M. (2019). Learnability for the Information Bottleneck. *Entropy*, 21(9), 924.
- Watanabe, S. (2009). *Algebraic Geometry and Statistical Learning Theory*. Cambridge University Press.
- Donoho, D. L., & Tanner, J. (2009). Counting faces of randomly projected polytopes when the projection radically lowers dimension. *J. AMS*, 22(1), 1-53.
- Kabashima, Y., Wadayama, T., & Tanaka, T. (2009). A typical reconstruction limit for compressed sensing based on Lp-norm minimization. *J. Stat. Mech.*, L09003.
- Krzakala, F., et al. (2012). Probabilistic reconstruction in compressed sensing: algorithms, phase diagrams, and threshold achieving matrices. *J. Stat. Mech.*, P08009.
- Chen, S., Lau, E., Mendel, T., Wei, S., & Murfet, D. (2023/2024). Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition. *ICLR 2024*.
- Lakkapragada, A., et al. (2025). Using physics-inspired Singular Learning Theory to understand grokking & other phase transitions in modern neural networks. *arXiv:2512.00686*.
- Bereska, L., et al. (2025). Superposition as Lossy Compression. *arXiv:2512.13568*.
- Elhage, N., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*.
- Chanin, D., et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *arXiv:2409.14507*.
- Chanin, D., et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. *arXiv:2505.11756*.
- Bussmann, B., et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. *arXiv:2503.17547*.
- Forrest, S., et al. (1994). Self-nonself discrimination in a computer. *Proc. IEEE Symp. Research in Security and Privacy*.
- de Castro, L. N., & Zuben, F. J. (2000). The clonal selection algorithm with engineering applications. *GECCO*.
- Gause, G. F. (1934). *The Struggle for Existence*. Williams & Wilkins.
- Hardin, G. (1960). The competitive exclusion principle. *Science*, 131(3409), 1292-1297.
- Makhzani, A., & Frey, B. (2014). Winner-Take-All Autoencoders. *NeurIPS*.

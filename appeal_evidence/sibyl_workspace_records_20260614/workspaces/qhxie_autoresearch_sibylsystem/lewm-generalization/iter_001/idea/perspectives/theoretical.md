# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Uselis et al. (2026) — "Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models"** (arXiv:2602.24264)
   - *Key result*: Formalizes three desiderata for compositional generalization under standard training — divisibility, transferability, stability — and proves that they impose necessary geometric constraints: representations must decompose linearly into per-concept components, and these components must be orthogonal across concepts. This is not an empirical regularity but a mathematical necessity under the stated conditions.

2. **Fu et al. (2024) — "A General Theory for Compositional Generalization"** (arXiv:2405.11743)
   - *Key result*: Establishes the first No Free Lunch (NFL) theorem for compositional generalization: no method uniformly outperforms all others across all CG problems. Provides a general bound: effective CG solutions should minimize the mutual information between the learning algorithm's output and the composition rule given the training distribution. Decomposes CG error into in-distribution learning error plus distribution shift penalty.

3. **Kratsios et al. (2025) — "Sharp Generalization Bounds for Foundation Models with Asymmetric Randomized Low-Rank Adapters"** (arXiv:2506.14530)
   - *Key result*: Proves the generalization gap for rank-r LoRA scales as O~(sqrt(r)/sqrt(N)) with high probability for N samples. Establishes a matching lower bound of O(1/sqrt(N)), showing these bounds are essentially tight. Increasing rank improves expressiveness at the cost of a wider generalization gap.

4. **Maes et al. (2026) — "LeWorldModel: Stable End-to-End JEPA from Pixels"** (arXiv:2603.19312)
   - *Key result*: Introduces SIGReg (Sketched Isotropic Gaussian Regularizer) based on the Cramer-Wold theorem: multivariate distribution matches isotropic Gaussian iff all 1D projections match. Uses Epps-Pulley test statistic on M random projections. Only two loss terms (MSE prediction + SIGReg). Latent space is 192-dimensional with ViT-Tiny encoder (~5M params) + transformer predictor (~10M params). Linear probes on frozen embeddings accurately predict physical quantities. Fails in low-diversity Two-Room environment.

5. **Locatello et al. (2019) — "Challenging Common Assumptions in Unsupervised Disentanglement"** (ICML 2019)
   - *Key result*: Proves that unsupervised disentanglement is impossible without inductive biases. KL regularization toward isotropic Gaussian prior does not guarantee disentanglement: while it encourages statistical independence in the aggregate posterior, it does not ensure individual latent units correspond to single generative factors. This result limits what SIGReg can guarantee about physical concept factorization without additional structure.

6. **PAC-Bayes Generalization Bounds for Dynamical Systems** (HAL, 2024)
   - *Key result*: Extends PAC-Bayesian bounds to non-linear state-space models (including RNNs), handling noise, infinite time horizons, and single trajectories. Directly applicable to world model dynamics learning in latent space.

7. **Systematic Generalization Scales with Information Entropy** (arXiv:2505.13089, 2025)
   - *Key result*: Neural networks achieve systematic OOD generalization when the entropy of the training distribution is high, independent of the number of unique samples. Connects training data diversity to generalization capability.

8. **Zhu et al. (2024) — "Asymmetry in Low-Rank Adapters of Foundation Models"** (arXiv:2402.16842)
   - *Key result*: In LoRA decomposition BA, the A matrix extracts features while B creates output. Fine-tuning B alone is more effective than fine-tuning A. Random untrained A performs nearly as well as fine-tuned A. Information-theoretic analysis bounds generalization of low-rank adapters.

9. **Chen et al. (2021) — "Representation Subspace Distance for Domain Adaptation Regression"** (ICML 2021)
   - *Key result*: Defines Representation Subspace Distance (RSD) via principal angles on the Grassmann manifold. Sum of sine values of principal angles between source/target representation subspaces forms a valid metric satisfying triangle inequality. Provides a geometric framework for quantifying domain shift.

10. **Redhardt et al. (2025) — "Scaling Can Lead to Compositional Generalization"** (arXiv:2507.07207) and companion work "Does Data Scaling Lead to Visual Compositional Generalization?" (arXiv:2507.07102)
    - *Key result*: Compositional generalization correlates with linear decodability of task constituents. Data diversity (combinatorial coverage), not raw scale, drives linear factored representations. Validated on DINO, CLIP. Not yet tested on world models.

11. **Cramer-Wold Auto-Encoder** (JMLR, Volume 21)
    - *Key result*: Leverages Cramer-Wold theorem to reduce distributional matching in latent space to 1D calculations with closed-form solutions. Matching all 1D projections (not just axis-aligned marginals) is necessary and sufficient to match the full joint distribution.

12. **Germain et al. — "PAC-Bayes and Domain Adaptation"** (Neurocomputing, 2020; arXiv:1707.05712)
    - *Key result*: PAC-Bayesian bound for domain adaptation decomposes target risk into three terms: KL complexity of the posterior, empirical source risk, and a domain divergence measuring structural difference between source and target distributions. Provides the formal backbone for analyzing LoRA adaptation efficiency as a domain transfer problem.

### Theoretical Landscape Summary

The theoretical landscape for compositional generalization in world models sits at the intersection of four established frameworks:

**A. Representation geometry theory** (Uselis et al., 2026; Redhardt et al., 2025): Compositional generalization requires that per-concept representations be linear and mutually orthogonal. This is a *necessary* condition (proven), not merely an empirical regularity. The question for LeWM is whether SIGReg's isotropic Gaussian prior creates this structure.

**B. Information-theoretic CG bounds** (Fu et al., 2024): There is no universal solution to CG (NFL theorem). Effective solutions minimize mutual information between the learned model and the composition rule. This implies that successful cross-domain transfer requires the model to learn *compositional structure* rather than memorizing specific factor combinations.

**C. LoRA generalization theory** (Kratsios et al., 2025; Zhu et al., 2024): Rank-r LoRA adaptation has well-characterized sample complexity O~(sqrt(r)/sqrt(N)). The A/B asymmetry reveals that feature extraction (A) vs. output generation (B) play fundamentally different roles, with implications for whether encoder vs. predictor adaptation is more efficient.

**D. Domain adaptation theory** (Chen et al., 2021; Germain et al.): The Grassmannian geometry of representation subspaces provides a formal metric for domain shift. PAC-Bayes bounds decompose adaptation error into source performance, complexity, and domain divergence. These tools can quantify the "distance" between physical domains.

**What is NOT known** (key gaps):
- Whether SIGReg's Gaussian constraint is sufficient (not just necessary) for the linear orthogonal structure required by compositional generalization theory
- How LoRA rank relates to the intrinsic dimensionality of cross-domain physics transfer
- Whether the NFL theorem for CG has a constructive exception for physically grounded world models with specific inductive biases
- Formal connection between representation subspace distance across physical domains and prediction error in the JEPA latent space

---

## Phase 2: Initial Candidates

### Candidate A: SIGReg-Induced Linear Factorization Theorem for Physical Concept Representations

**Formal claim (Theorem sketch)**: Let f: O -> R^d be an encoder trained with SIGReg regularization such that the aggregate posterior q(z) = E_{o~D}[delta(f(o))] matches N(0, I_d). If the data distribution D is generated by K independent physical factors (gravity, friction, mass, shape, etc.) with sufficient combinatorial diversity, then the encoder's representation decomposes as z = sum_{k=1}^{K} phi_k(c_k) + epsilon, where phi_k captures factor k and <phi_i, phi_j> = 0 for i != j, up to an error term bounded by O(lambda * SIGReg_residual + 1/sqrt(n_combinations)).

**Proof sketch**:
1. *Lemma 1 (Cramer-Wold + isotropy implies near-independence)*: SIGReg enforces that all 1D projections of z match N(0,1). By the Cramer-Wold theorem, this implies q(z) -> N(0, I_d). An isotropic Gaussian has independent components in any orthonormal basis, which implies the covariance matrix is identity. If each physical factor varies independently in the data, the latent must encode these factors in mutually uncorrelated directions (otherwise the joint distribution would deviate from isotropic Gaussian).
2. *Lemma 2 (Uncorrelation + Gaussian -> independence)*: For Gaussian random variables, uncorrelation implies independence. Combined with Lemma 1, this yields statistical independence of physical factor encodings.
3. *Lemma 3 (Independence + prediction pressure -> linear factorization)*: The prediction loss L_pred requires the predictor to use the latent to predict dynamics. If each physical factor independently contributes to dynamics (physical superposition), the most efficient encoding that maintains independence (from SIGReg) and supports prediction (from L_pred) assigns each factor to a separate linear subspace.
4. *Main theorem*: Combine Lemmas 1-3 with the Uselis et al. (2026) necessary conditions to argue that SIGReg + prediction loss + independent physical factors -> the exact geometric structure needed for compositional generalization.

**Empirical prediction**: If this theorem holds, then:
- Principal Component Analysis of LeWM latent representations organized by physical factor should reveal orthogonal factor subspaces
- Principal angles between per-factor subspaces should be near pi/2
- The Centered Kernel Alignment (CKA) between factor-organized representations should show block-diagonal structure
- Zero-shot transfer to unseen factor combinations should succeed proportionally to how well the orthogonality condition is satisfied

**Connection to existing theory**: Directly extends Uselis et al. (2026) from static vision embeddings to dynamic world model representations. Links SIGReg (Maes et al., 2026) to the geometric conditions required for CG. Fills Gap 2 from the literature survey.

**Novelty estimate**: 8/10 — The connection between SIGReg's Gaussian enforcement and the Uselis linear orthogonality requirement has not been made formal. The key novelty is showing that the Cramer-Wold mechanism provides a *sufficient* condition for the geometric structure that Uselis showed is *necessary*.

---

### Candidate B: Adaptation Complexity Hierarchy for Physical Concept Transfer

**Formal claim (Proposition)**: Let M = (enc, pred) be a trained LeWM. Define the adaptation complexity of transferring to a new physical domain D' as the minimum LoRA rank r* such that LoRA-adapted M achieves epsilon-optimal performance on D' with N_{adapt} samples. Then the adaptation complexity satisfies a hierarchy:

r*_encoder < r*_predictor < r*_joint

when the target domain D' shares visual appearance but differs in dynamics, AND

r*_predictor < r*_encoder < r*_joint

when D' shares dynamics but differs in visual appearance.

Furthermore, the sample complexity for epsilon-optimal adaptation scales as N_{adapt} = O(r* * d_factor / epsilon^2), where d_factor is the intrinsic dimensionality of the novel physical factor.

**Proof sketch**:
1. *Decomposition lemma*: The total representation error on a new domain can be decomposed as E_total = E_visual(enc) + E_dynamics(pred) + E_interaction, where E_interaction captures encoder-predictor coupling.
2. *Rank-sufficiency argument*: If the target domain differs from source only in K physical factors, and each factor occupies a d_k-dimensional subspace in the latent, then rank r >= max_k(d_k) suffices for the component that encodes the changed factor(s). The other component requires rank 0 (no adaptation) if it encodes unchanged aspects.
3. *Sample complexity via LoRA bounds*: Apply Kratsios et al. (2025) bound O~(sqrt(r)/sqrt(N)) to each component. The total adaptation sample complexity is dominated by the component requiring higher rank.
4. *Hierarchy derivation*: Visual-change domains require encoder adaptation (rank proportional to visual factor dimensionality). Dynamics-change domains require predictor adaptation (rank proportional to dynamics factor dimensionality). Joint changes require both, yielding the inequality chain.

**Empirical prediction**: 
- LoRA applied only to the encoder should suffice for visual domain shifts (new object shapes, colors)
- LoRA applied only to the predictor should suffice for dynamics shifts (new friction, gravity)
- The minimum sufficient LoRA rank directly measures the intrinsic dimensionality of the domain gap
- Sample efficiency of adaptation should scale linearly with the rank

**Connection to existing theory**: Builds on Kratsios et al. (2025) LoRA generalization bounds, Zhu et al. (2024) A/B asymmetry analysis, and PAC-Bayes domain adaptation (Germain et al.). Novel application to world model encoder-predictor decomposition.

**Novelty estimate**: 7/10 — The idea that encoder and predictor capture different aspects is intuitive, but formalizing the adaptation complexity hierarchy and connecting it to LoRA rank as an intrinsic dimensionality probe is new. The risk is that the decomposition E_total = E_visual + E_dynamics + E_interaction may not hold cleanly.

---

### Candidate C: Information-Theoretic Generalization Boundary for Compositional Physical Domains

**Formal claim (Theorem sketch)**: Let F = {f_1, ..., f_K} be the set of physical factors (gravity, friction, mass, shape, etc.), and let S_train be the set of factor combinations seen during training. Define the compositional gap as CG(f_new) = E_pred(f_new) - E_pred(S_train), where E_pred is the prediction error. Then:

CG(f_new) <= sum_{k in novel(f_new)} I(z; f_k | f_{-k}) + beta * d_Grassmann(V_{S_train}, V_{f_new})

where I(z; f_k | f_{-k}) is the conditional mutual information between the latent and each novel factor conditioned on shared factors, d_Grassmann is the Grassmannian distance between the training-domain representation subspace and the new-domain subspace, and beta depends on the predictor's Lipschitz constant in latent space.

The bound implies a *generalization boundary*: transfer fails (CG > threshold) when the novel combination introduces factors whose conditional mutual information with the latent is high (i.e., factors not well-separated in the representation) AND the Grassmannian distance is large.

**Proof sketch**:
1. *Factorization of prediction error*: By the chain rule of mutual information, the prediction error on a novel combination can be decomposed into contributions from each novel factor plus their interactions.
2. *Subspace distance bound*: By Chen et al. (2021), the representation quality on the target domain is bounded by the representation subspace distance, which is the sum of sines of principal angles between source and target representation subspaces.
3. *Lipschitz propagation*: The predictor's error on a new latent region is bounded by its Lipschitz constant times the distance from the training manifold in latent space. This distance is bounded by the Grassmannian distance.
4. *Combination via triangle inequality*: Total compositional gap is bounded by the sum of per-factor information terms (representation quality) plus the predictor's extrapolation error (geometric distance).

**Empirical prediction**:
- Factors that have high conditional MI (poorly separated) will cause larger CG gaps
- The Grassmannian distance between training and test domains predicts transfer failure
- Plotting CG against d_Grassmann should reveal a sharp phase transition (the "generalization boundary")
- Factors that are physically independent (e.g., gravity and object shape) should have low conditional MI and transfer well; factors that interact (e.g., mass and friction in contact dynamics) should have high conditional MI and transfer poorly

**Connection to existing theory**: Combines Fu et al. (2024) general CG theory with Chen et al. (2021) Grassmannian analysis and information-theoretic decomposition. Extends the "generalization boundary" concept from vision (Redhardt et al., 2025) to physically grounded world models.

**Novelty estimate**: 9/10 — No prior work provides an information-theoretic characterization of where world model compositional generalization breaks down. The combination of mutual information decomposition with Grassmannian geometry for predicting failure modes is original. The risk is that the bound may be loose (vacuous in practice).

---

## Phase 3: Self-Critique

### Against Candidate A: SIGReg-Induced Linear Factorization

**Proof soundness attack**: 
- *Gap in Lemma 1*: SIGReg matches 1D projections to Gaussian via the Epps-Pulley statistic on M random projections, but M is finite. The Cramer-Wold theorem requires ALL projections. With finite M, there remains a gap of O(1/sqrt(M)) in distributional matching. If M is small relative to d=192, the isotropic Gaussian enforcement may be incomplete, and hidden correlations between physical factors could survive.
- *Gap in Lemma 3*: The claim that "independence + prediction pressure -> linear factorization" assumes physical superposition of factors. But many physical interactions are nonlinear (e.g., friction force depends on both normal force and surface coefficient — these do not superpose). In such cases, the latent may need to encode factor interactions, breaking linear factorization.
- *Locatello impossibility*: Locatello et al. (2019) proved unsupervised disentanglement is impossible without inductive biases. SIGReg + MSE prediction loss may not provide sufficient inductive bias for the specific disentanglement we need. The proof needs to carefully delineate what additional assumptions are required.

**Tightness attack**: Even if the theorem holds, it only guarantees factorization "up to epsilon" where epsilon depends on SIGReg residual and combinatorial coverage. In practice, SIGReg does not achieve exact Gaussian matching (it optimizes a finite set of random projections), so epsilon could be large enough to make the bound vacuous for real-world experiments.

**Relevance attack**: The theorem addresses the *static* representation quality (encoder alone). But compositional generalization in a world model also depends on the *predictor* extrapolating correctly in novel regions of the latent space. Even perfect factorization in the encoder does not guarantee the predictor can compose dynamics correctly.

**Novelty attack**: The connection between Gaussian priors and disentanglement is well-studied in the VAE literature. While the specific application to SIGReg and JEPA world models is new, the underlying mechanism (Gaussian -> uncorrelation -> factorization) is standard. The true novelty lies in the predictor-side implications, which the current formulation underexplores.

**Verdict**: MODERATE — The proof sketch has genuine gaps (finite M projections, nonlinear physical interactions, predictor extrapolation). However, the core insight (SIGReg creates a strong inductive bias toward the geometry Uselis requires) is sound and novel enough to pursue. Needs significant tightening of assumptions.

---

### Against Candidate B: Adaptation Complexity Hierarchy

**Proof soundness attack**:
- *Decomposition validity*: The claim E_total = E_visual(enc) + E_dynamics(pred) + E_interaction assumes the interaction term is small. But in end-to-end trained models, the encoder and predictor are co-adapted; the encoder's representation may encode visual information in a predictor-specific way. Adapting one without the other may not be decomposable.
- *Rank-sufficiency is an inequality, not equality*: The claim that r >= max_k(d_k) suffices assumes the LoRA update can target exactly the factor-specific subspace. In practice, LoRA modifies attention weights globally, and the correspondence between LoRA rank and factor dimensionality may be loose.

**Tightness attack**: The hierarchy r*_encoder < r*_predictor (or vice versa) is a strict ordering. In practice, the difference may be negligible, making the hierarchy empirically undetectable. If both components contribute similarly, the theoretical prediction becomes unfalsifiable.

**Relevance attack**: This is more of an empirical hypothesis than a theorem. The "proof" relies on decomposition assumptions that may not hold for specific architectures. The practical utility depends on whether the hierarchy is *large enough* to guide experimental design (e.g., "only adapt the predictor for dynamics shifts").

**Novelty attack**: The idea that different model components encode different types of information is well-established. Zhu et al. (2024) showed A/B asymmetry in LoRA. The specific application to world model encoder/predictor is incrementally novel but not groundbreaking.

**Verdict**: MODERATE — The core insight is practical and testable, but the theoretical contribution is thin. The decomposition assumptions may not hold for co-adapted encoder-predictor pairs. Best framed as a well-motivated empirical hypothesis with theoretical backing rather than a formal theorem.

---

### Against Candidate C: Information-Theoretic Generalization Boundary

**Proof soundness attack**:
- *Conditional MI estimation*: In practice, I(z; f_k | f_{-k}) is extremely difficult to estimate reliably in high-dimensional latent spaces. The bound may be theoretically correct but practically uncomputable.
- *Grassmannian distance assumes linear structure*: d_Grassmann measures the distance between linear subspaces. If the representation is nonlinearly organized (which it will be, since the encoder is a deep network), the Grassmannian distance may not capture the true geometric relationship.
- *Lipschitz constant of predictor*: The predictor is a transformer, whose Lipschitz constant is typically very large and hard to bound tightly. The beta * d_Grassmann term may dominate and make the bound vacuous.

**Tightness attack**: The bound is a sum of two terms (MI + Grassmannian). In the worst case, both terms could be loose by orders of magnitude, making the bound vacuous. The prediction of a "sharp phase transition" is optimistic — real generalization boundaries are likely smooth and noisy.

**Relevance attack**: Despite potential looseness, the qualitative predictions are valuable: physically independent factors should transfer better than interacting ones, and Grassmannian distance should predict failure. Even if the quantitative bound is vacuous, the qualitative structure guides experimental design effectively.

**Novelty attack**: Information-theoretic decomposition of OOD error is standard in domain adaptation theory. The specific application to physical factor combinations in world models and the connection to Grassmannian geometry has not been done before, confirming novelty.

**Verdict**: STRONG — Despite the bound being potentially loose, the framework provides the first formal characterization of compositional generalization boundaries for world models. The qualitative predictions are testable and novel. The proof can be refined by using tighter MI estimators and local Lipschitz bounds.

---

## Phase 4: Refinement

### Dropped Ideas

None dropped outright. All three candidates address complementary aspects of the research question.

### Strengthened Survivors

**Candidate A (SIGReg Factorization)** -> Refined:
- Weaken the claim from "SIGReg guarantees linear factorization" to "SIGReg provides a strong inductive bias toward the geometric structure required for CG, with the approximation quality bounded by the SIGReg residual and the number of projection directions M."
- Add explicit assumption: physical factors must be *approximately* independent in the data distribution (which holds for designed experiments but may fail for naturally correlated physical parameters).
- Acknowledge the Locatello impossibility by noting SIGReg + prediction loss provides a *specific* inductive bias not covered by the general impossibility result (which applies to reconstruction-based losses only).
- Merge the predictor-side analysis from Candidate C to address the relevance attack.

**Candidate B (Adaptation Hierarchy)** -> Downgraded to supporting empirical prediction:
- The theoretical contribution is insufficient for a standalone theorem. However, the adaptation complexity hierarchy generates clear, falsifiable experimental predictions that complement Candidate C.
- Reframe as a corollary of Candidate C: the MI decomposition predicts *which* component needs adaptation (the one encoding the changed factor), and the LoRA rank measures the dimensionality of that factor's subspace.

**Candidate C (Generalization Boundary)** -> Selected as front-runner:
- Strengthen the Grassmannian distance analysis by using *local* Grassmannian distance (tangent plane analysis at the training manifold) rather than global subspace distance, tightening the bound.
- Replace intractable conditional MI with a computable proxy: linear probing accuracy for each physical factor, which lower-bounds how well the factor is represented and upper-bounds the MI contribution to CG gap.
- The "sharp phase transition" prediction is weakened to "monotonic increase in CG gap with Grassmannian distance, with a characteristic scale set by the predictor's effective Lipschitz constant in the local region."
- Incorporate the SIGReg insight (Candidate A) as a sufficient condition: when SIGReg residual is low, the MI term is small, so the bound is dominated by the Grassmannian term (which is measurable).

### Additional Evidence for Novelty

Searched for papers combining information theory, Grassmannian geometry, and compositional generalization for world models. No prior work addresses this combination. The closest work:
- Fu et al. (2024) provides general CG theory but not for world models or with geometric structure
- Chen et al. (2021) provides Grassmannian analysis but for regression, not dynamics prediction
- Uselis et al. (2026) provides geometric conditions but for static embeddings, not temporal prediction

**Selected front-runner**: Candidate C (Information-Theoretic Generalization Boundary), enriched with insights from A (SIGReg mechanism) and B (adaptation diagnostics).

---

## Phase 5: Final Proposal

### Title

**Information-Theoretic Generalization Boundaries for Compositional Physical Concept Transfer in JEPA World Models**

### Formal Claim

**Main Theorem (informal statement)**: For a JEPA world model M = (enc, pred) trained with SIGReg on a set of physical environments characterized by K independent factors F = {f_1, ..., f_K} with training combinations S_train, the prediction error on a novel factor combination f_new decomposes as:

E_pred(f_new) <= E_pred(S_train) + CG(f_new)

where the compositional gap CG(f_new) is bounded by:

CG(f_new) <= alpha * sum_{k in Delta(f_new)} [1 - R^2_linear(f_k)] + beta * sum_{j=1}^{d_eff} sin(theta_j(V_train, V_new)) + gamma * E_SIGReg

where:
- Delta(f_new) indexes the physical factors that differ between f_new and the nearest training combination
- R^2_linear(f_k) is the linear probing R-squared for factor k from frozen latent embeddings (measures how well factor k is linearly decodable)
- theta_j are the principal angles between the training-domain and new-domain representation subspaces on the Grassmann manifold
- d_eff is the effective dimensionality of the representation relevant to the novel factors
- E_SIGReg is the SIGReg regularization residual (how far the latent is from isotropic Gaussian)
- alpha, beta, gamma are constants depending on the predictor architecture

**Key Corollaries**:
1. *Factorization corollary*: When E_SIGReg is small (SIGReg well-satisfied), the gamma term vanishes and CG is dominated by linear probing quality and subspace distance. This connects SIGReg to compositional generalization via the Uselis geometric conditions.
2. *Adaptation corollary*: LoRA adaptation of rank r reduces the subspace distance term by O(r/d_eff), predicting that the minimum rank for epsilon-optimal adaptation is r* = O(d_eff * epsilon / beta).
3. *Failure mode prediction*: CG is large when (a) factor representations are entangled (low R^2_linear), (b) the novel combination moves far from training subspace (large principal angles), or (c) SIGReg is poorly satisfied (collapse or near-collapse).

### Proof Sketch

**Step 1 — Representation quality decomposition**:
Using the data processing inequality, the prediction error on any input is bounded by the mutual information between the latent representation and the ground-truth state. For compositional data, this MI decomposes additively across independent factors (by independence assumption). The per-factor MI is related to linear probing R^2 via the DPI: I(z; f_k) >= R^2_linear(f_k) * H(f_k).

**Step 2 — Extrapolation via Grassmannian geometry**:
For novel factor combinations, the encoder must produce latent representations in a region not covered by training. The distance between training and novel regions, measured on the Grassmann manifold via principal angles (following Chen et al., 2021), bounds how far the predictor must extrapolate. The predictor's error grows with this distance, bounded by its local Lipschitz constant.

**Step 3 — SIGReg regularization effect**:
When SIGReg is well-satisfied, the latent distribution is approximately isotropic Gaussian. For Gaussian variables, uncorrelation implies independence (a classical result). If each physical factor is encoded in an uncorrelated direction, the factors occupy near-orthogonal subspaces, which is exactly the Uselis et al. (2026) necessary condition for CG. The SIGReg residual E_SIGReg measures how far we are from this ideal.

**Step 4 — Combine via triangle inequality**:
The total CG gap is bounded by the sum of (a) per-factor representation deficiency (1 - R^2_linear), (b) subspace extrapolation distance, and (c) deviation from the ideal Gaussian structure. Each term is measurable from the trained model and the test domain.

### Assumptions

1. Physical factors in the data are *approximately* independently distributed (or at least have a known dependency structure that can be factored out)
2. The encoder's representation has bounded dimension d relevant to the physical factors (d_eff << d_total = 192)
3. The predictor's transition function is locally Lipschitz in the latent space (standard for bounded neural networks)
4. SIGReg optimization achieves epsilon-approximate Gaussian matching (the Epps-Pulley residual is small)
5. Sufficient training data diversity: the training factor combinations span the factor space with enough density that the linear probing evaluation is reliable

### Empirical Predictions

1. **Orthogonality test**: Principal angles between per-factor representation subspaces should be close to pi/2 when SIGReg is well-satisfied, and should deviate proportionally to the SIGReg residual.

2. **Generalization boundary visualization**: Plotting prediction error against Grassmannian distance from training domain (computed via principal angles) should reveal a monotonic relationship, with steeper slope for factors that have lower R^2_linear (less well-represented factors fail more catastrophically when extrapolated).

3. **LoRA rank as dimensionality probe**: The minimum LoRA rank for epsilon-recovery on a novel domain should correlate with the effective dimensionality of the changed factors, measurable via the eigenspectrum of the representation covariance conditioned on the changed factor.

4. **Failure mode taxonomy**: Factor combinations involving physically interacting quantities (e.g., mass x friction in contact dynamics) should have higher CG gap than non-interacting combinations (e.g., gravity strength x object color), because the interaction breaks the independence assumption.

5. **SIGReg ablation**: Removing or weakening SIGReg should degrade compositional transfer more than it degrades in-distribution performance, because the Gaussian enforcement provides the inductive bias for factorization.

### Experimental Plan

All experiments target GPT-2-scale models (ViT-Tiny encoder, small transformer predictor) consistent with LeWM architecture. Each experiment designed to complete within 1 hour on a single GPU.

**Experiment 1 — Latent geometry analysis** (est. 30 min):
- Train LeWM on DMControl environments with systematically varied physical parameters (gravity in {0.5g, 1g, 2g}, friction in {0.1, 0.5, 1.0}, mass in {0.5, 1.0, 2.0})
- Extract latent representations for each factor combination
- Compute: (a) per-factor linear probing R^2, (b) principal angles between factor subspaces, (c) CKA between factor-organized representations
- Validate the orthogonality prediction (Prediction 1)

**Experiment 2 — Compositional transfer evaluation** (est. 45 min):
- Hold out specific factor combinations from training (e.g., {gravity=2g, friction=0.1} never seen)
- Evaluate zero-shot prediction error on held-out combinations
- Compute Grassmannian distance between held-out and training representations
- Plot CG gap vs. Grassmannian distance (Prediction 2)
- Test whether physical independence of changed factors predicts transfer success (Prediction 4)

**Experiment 3 — LoRA adaptation diagnostics** (est. 45 min):
- Apply LoRA (ranks r = 1, 2, 4, 8, 16) to encoder only, predictor only, and both
- For each held-out domain, measure adaptation efficiency (samples to epsilon-recovery)
- Correlate minimum sufficient rank with factor subspace dimensionality (Prediction 3)
- Test the adaptation hierarchy (encoder vs. predictor, visual vs. dynamics changes)

**Experiment 4 — SIGReg ablation** (est. 30 min):
- Train LeWM variants with lambda_SIGReg in {0, 0.01, 0.05, 0.1, 0.2}
- Measure: in-distribution prediction error, cross-domain CG gap, latent orthogonality
- Test whether SIGReg disproportionately benefits compositional transfer (Prediction 5)

### Baselines

**Theoretical baselines**:
- Random baseline: CG gap when the encoder outputs random Gaussian embeddings (establishes the maximum gap)
- Oracle baseline: CG gap of a model trained on ALL factor combinations including held-out (establishes minimum gap)
- Uselis linear factorization score: measure how well the representation satisfies the Uselis geometric conditions, and compare to actual CG performance

**Empirical baselines**:
- DINO-WM (frozen DINOv2 encoder + learned predictor): tests whether pre-trained visual features provide better compositional structure
- LeWM without SIGReg: ablation to isolate the regularizer's effect
- Full fine-tuning (no LoRA constraint): upper bound on adaptation performance

### Risk Assessment

1. **Bound may be vacuous**: The primary risk. Information-theoretic bounds for neural networks are often orders of magnitude loose. Mitigation: focus on the qualitative predictions (rank ordering of domain difficulty, failure mode taxonomy) rather than quantitative tightness.

2. **Physical factors may not be independent**: Many physical parameters interact (e.g., mass affects both gravitational and inertial forces). If independence fails, the additive decomposition in Step 1 breaks. Mitigation: test on carefully designed environments where independence holds by construction, then examine what happens when it is violated.

3. **SIGReg residual may be too large**: If the Epps-Pulley statistic does not converge well on 192-dimensional latent with the default M random projections, the gamma * E_SIGReg term may dominate and obscure the other terms. Mitigation: measure E_SIGReg directly and report it; vary M to study convergence.

4. **Grassmannian distance may not capture nonlinear structure**: The Grassmann manifold analysis assumes linear subspaces, but the encoder is nonlinear. Mitigation: use kernel-PCA or local PCA to capture curved manifold structure; compare with CKA as an alternative similarity measure.

5. **LoRA rank may not correspond to factor dimensionality**: The relationship between LoRA rank and the intrinsic dimensionality of the domain shift may be mediated by the model's parameter structure in complex ways. Mitigation: compare with SVD analysis of the weight changes from full fine-tuning to establish ground truth for the "true" rank of adaptation.

### Novelty Claim

This is the first formal framework that:
1. **Connects SIGReg's Gaussian regularization to the geometric conditions for compositional generalization** (Uselis et al., 2026), providing a mechanistic explanation for why JEPA world models can (or cannot) generalize across physical domains.
2. **Provides a computable bound on compositional gap** using measurable quantities (linear probing R^2, principal angles, SIGReg residual) rather than intractable information-theoretic quantities.
3. **Predicts failure modes** via the interaction between representation quality and geometric distance, generating a testable taxonomy of which physical concept combinations are easy vs. hard to transfer.
4. **Uses LoRA adaptation efficiency as a diagnostic tool** with theoretical backing from recent LoRA generalization bounds (Kratsios et al., 2025), connecting the minimum adaptation rank to the intrinsic dimensionality of domain shift.

**Evidence of novelty**: No paper in the literature combines information-theoretic generalization bounds, Grassmannian geometry, and SIGReg regularization analysis for world model compositional transfer. The closest works address subsets: Uselis (geometry for static embeddings), Fu (general CG theory without physics), Chen (Grassmannian for regression). The synthesis for JEPA world models with actionable experimental predictions is new.

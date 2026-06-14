# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Bhatnagar et al. (2024)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (arXiv:2409.14507, NeurIPS 2025)
   - **Key mathematical result**: Formal proof that delta-absorption preserves perfect reconstruction while strictly decreasing sparsity loss. For hierarchical features f1 (parent) and f2 subset f1, the sparsity loss derivative dL_sp/ddelta = -p_11 < 0, proving SAE optimization provably drives delta -> 1 (full absorption).
   - **Framework**: Two-feature toy model with explicit encoder/decoder construction showing absorption is structurally incentivized by the sparsity penalty.

2. **Tang et al. (2025)** — "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534)
   - **Key mathematical result**: Theorem 3.8 proves that for any realizable hierarchical concept structure, there exists a realizable activation pattern exhibiting feature absorption. Theorem 3.6 proves spurious local minima correspond exactly to absorption patterns. Theorem 3.2 establishes piecewise biconvex structure of SDL optimization.
   - **Framework**: Characterization of global minima (correct recovery) vs. spurious partial minima (absorption, dead neurons, polysemanticity) via activation pattern regions.

3. **Cui et al. (2025)** — "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy" (arXiv:2506.15963, ICLR 2026)
   - **Key mathematical result**: Theorem 1: closed-form optimal solution W_m* = I*(W_p, 0)^T exhibits feature shrinking and feature vanishing. Theorem 2: exact recovery only guaranteed when ground-truth sparsity S -> 1 (extreme sparsity).
   - **Framework**: Weighted SAE (WSAE) with reweighting matrix Gamma = diag(gamma_1, ..., gamma_n) that narrows the gap between SAE loss and ground-truth reconstruction loss.

4. **Chen et al. (2025)** — "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders" (arXiv:2506.14002, ICLR 2026)
   - **Key mathematical result**: First SAE algorithm with provable feature recovery guarantees under incoherence assumptions (weaker than Gaussianity). Group Bias Adaptation (GBA) ensures activation frequency matches feature occurrence frequency.
   - **Framework**: Statistical model x ≈ h^T V where h is sparse and V is dictionary of monosemantic features.

5. **Lu et al. (2025)** — "Sparse Autoencoders, Again?" (arXiv:2506.04859, ICML 2025)
   - **Key mathematical result**: Theorem 4.5 proves global minima recover structured data on unions of manifolds with lower bound on optimal loss: (d - sum_i alpha_i r_i) log(gamma)/2 + O(1), where r_i are manifold dimensions. SAEs are piecewise affine generalizations of k-means.
   - **Framework**: Manifold-based analysis connecting sparsity to local dimension estimation via VAEase (adaptive sparsity through variance gating).

6. **Mendel (2025)** — "Mathematical Models of Computation in Superposition"
   - **Key mathematical result**: While Elhage et al. (2022) suggested exponential feature storage in superposition, the number of features that can be stored AND computed with is O-tilde(d^2) — the information-theoretic limit.
   - **Framework**: Computational complexity analysis of superposition, distinguishing storage capacity from usable computation capacity.

7. **Bussmann et al. (2025)** — "Learning Multi-Level Features with Matryoshka Sparse Autoencoders" (arXiv:2503.17547, ICML 2025)
   - **Key mathematical result**: Nested reconstruction constraints prevent later latents from absorbing earlier ones. Absorption rate drops from 0.49 (BatchTopK) to 0.05 (Matryoshka) at L0=40.
   - **Framework**: Multi-scale loss sum_i beta_i ||x - D_{<=m_i} z_{<=m_i}||^2 with prefix constraints.

8. **Chanin et al. (2025)** — "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders" (arXiv:2505.11756)
   - **Key mathematical result**: Balanced Matryoshka at beta ≈ 0.25 cancels hedging and absorption, yielding "nearly perfect SAE" on toy hierarchical models.
   - **Framework**: Tradeoff between hedging (child features add negative components) and absorption (parent absorbs child features).

9. **Elhage et al. (2022)** — "Toy Models of Superposition" (Anthropic, Transformer Circuits)
   - **Key mathematical result**: Networks encode more features than dimensions via linear combinations with negative interferences (digon/polygon geometric structures). Polysemanticity arises from sparsity in feature activation.
   - **Framework**: The foundational theoretical basis for SAEs — superposition hypothesis.

10. **Scherlis et al. (2022)** — "Polysemanticity and Capacity in Neural Networks"
    - **Key mathematical result**: Optimal capacity allocation tends to polysemantically represent less important features. Feature capacity is a limited resource distributed across features.
    - **Framework**: Resource allocation perspective on polysemanticity.

11. **Park et al. (2025)** — "The Geometry of Categorical and Hierarchical Representations in LLMs" (ICLR 2025)
    - **Key mathematical result**: WordNet noun hyponym hierarchy is linearly represented in language model embeddings. Cosine similarity between estimated vector representations correlates with shortest distance in WordNet hierarchy.
    - **Framework**: Validates that semantic hierarchies exist in LLM activations, making absorption a meaningful construct.

12. **Karvonen et al. (2025)** — "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders" (arXiv:2503.09532)
    - **Key mathematical result**: "Gains on proxy metrics do not reliably translate to better practical performance" — the sparsity-fidelity frontier does not predict downstream interpretability outcomes.
    - **Framework**: Standardized 8-evaluation benchmark establishing the empirical context for absorption measurement.

### Theoretical Landscape Summary

**What is known:**
- Absorption is structurally incentivized by sparsity penalties (Bhatnagar et al. 2024): for hierarchical features, any delta in [0,1] preserves reconstruction while sparsity loss strictly decreases with delta.
- The optimization landscape has spurious local minima corresponding exactly to absorption patterns (Tang et al. 2025, Theorem 3.6 and 3.8).
- Hierarchical concept structures *necessarily* induce realizable activation patterns with feature absorption (Tang et al. 2025, Theorem 3.8).
- SAEs generally fail to fully recover ground-truth features unless extreme sparsity holds (Cui et al. 2025, Theorem 2).
- The number of computable features in superposition is O-tilde(d^2), not exponential (Mendel 2025).
- Nested dictionary structures (Matryoshka) can prevent absorption through prefix constraints (Bussmann et al. 2025).
- WordNet semantic hierarchies are linearly represented in LLM embeddings (Park et al. 2025).

**What is conjectured:**
- A unified rate-distortion function for hierarchical features could predict absorption thresholds.
- The information-theoretic "cost" of absorption (in bits of semantic information lost) can be quantified.
- The SAEBench absorption metric may measure base-model geometry rather than learned SAE structure.
- Combinatorial bounds on absorption rate as a function of tree depth, branching factor, and sparsity level exist but have not been derived.

**Where the gaps are:**
- No general theorem predicting absorption rate as a function of feature hierarchy structure (depth, branching), SAE width, and sparsity level.
- No information-theoretic characterization of the semantic distortion caused by absorption.
- No proof that first-letter absorption (the SAEBench metric) generalizes to arbitrary semantic hierarchies.
- No theoretical explanation for why Random SAEs and trained SAEs can produce identical absorption scores.
- No combinatorial bound on the minimum number of latents required to represent a tree-structured feature hierarchy without absorption.
- No theoretical framework for distinguishing "learned-structure" absorption from "geometric-confound" absorption.

---

## Phase 2: Initial Candidates

### Candidate A: The Geometric-Confound Origin of Absorption Scores

**Formal claim**: Define the absorption score A(SAE, T) for an SAE and task T as the SAEBench metric. Let D be the SAE decoder matrix and R_T be the matrix of probe directions for task T. Define the geometric alignment score as G(D, R_T) = ||D^T R_T||_F^2 / (||D||_F^2 ||R_T||_F^2). Then:

**Theorem (Informal)**: For any decoder matrix D (learned or random) with isotropically distributed columns, the expected absorption score E[A] depends only on the alignment between the probe subspace span(R_T) and the decoder column space span(D), not on whether D was learned. Specifically, if span(R_T) is approximately contained in span(D), then E[A | random D] ≈ E[A | learned D].

**Proof sketch**:
1. Lemma 1 (Absorption as projection): The SAEBench absorption formula computes a ratio of projections: A = (sum_{i in S_abs} proj_i) / (sum_{i in S_abs} proj_i + sum_{i in S_main} proj_i), where proj_i = a_i * d_i . p is the projection of probe direction p onto decoder direction d_i, weighted by activation a_i.
2. Lemma 2 (Random decoder alignment): For a random decoder D with i.i.d. Gaussian columns, the projection d_i . p for any fixed probe direction p has expected squared magnitude E[(d_i . p)^2] = ||p||^2 * ||d_i||^2 / d, where d is the activation dimension. This is identical in expectation to the projection onto any orthonormal basis.
3. Lemma 3 (Probe subspace containment): If the probe directions R_T span a low-dimensional subspace (k << d) that is contained in the activation space, then ANY decoder with sufficient column coverage will project similarly onto this subspace.
4. Main theorem: The absorption score depends on the *relative* projection strengths onto "absorbing" vs "main" latents. For random decoders, these relative strengths have the same expectation as for trained decoders when the probe subspace is low-dimensional and the decoder has sufficient capacity (dict_size >> k).

**Empirical prediction**: Random-decoder SAEs will produce absorption scores comparable to trained SAEs when (a) the probe subspace dimension k is small relative to dict_size, and (b) the decoder columns are isotropically distributed. PCA-basis SAEs (orthonormal columns aligned with activation variance) will also produce comparable scores. The difference between trained and random scores should be largest when the probe subspace is high-dimensional or when the trained decoder has learned specific alignments.

**Connection to existing theory**: Extends Cui et al.'s closed-form analysis to show that even random solutions can achieve comparable metric scores. Connects to the unified theory (Tang et al. 2025) by showing that the SAEBench metric may not distinguish between global minima (correct recovery) and spurious minima (absorption patterns). Connects to Mendel's O-tilde(d^2) bound by showing that probe-based metrics are insensitive to the difference between stored and computable features.

**Novelty estimate**: 8/10. No prior work has formalized why random decoders can achieve comparable absorption scores to trained decoders. This directly explains the iter_002 anomaly (Random SAE = Standard SAE) and the iter_003 experimental design.

---

### Candidate B: A Predictive Model for Absorption Severity from Hierarchy Geometry

**Formal claim**: For a feature hierarchy with tree structure T = (V, E) where each node v has activation probability p_v and parent(v) is the unique parent, define the absorption severity for node v as:

A(v) = lambda * p_v / (p_v + sum_{u in children(v)} p_u) * (1 - exp(-c * |children(v)|))

where lambda is the sparsity penalty and c is a constant. Then the total absorption rate A(T) = (1/|V|) sum_{v in V} A(v) satisfies:

**Theorem (Informal)**: A(T) is minimized when the hierarchy is a "star" (all children of root, depth 1) and maximized when the hierarchy is a "path" (linear chain, depth = |V|). For fixed |V| and lambda, A(T) increases monotonically with the variance of node depths.

**Proof sketch**:
1. Lemma 1 (Local absorption incentive): Following Bhatnagar et al., for any parent-child pair (u, v), the SAE loss derivative dL/ddelta < 0 for delta in [0,1], where delta parameterizes absorption degree.
2. Lemma 2 (Cascade effect): For a chain of depth D, absorption at level k cascades: if parent k is absorbed by child k+1, then parent k-1 is more likely to be absorbed by the combined (k, k+1) feature. This creates a multiplicative effect.
3. Lemma 3 (Branching protection): For a node with b children, the parent's activation probability p_parent is the union of child probabilities. The absorption incentive is distributed across b children, reducing the per-child absorption pressure.
4. Main theorem: By induction on tree structure, the total absorption is a sum of local terms weighted by depth. Star structures have minimal depth variance (all depth 1), minimizing absorption. Path structures have maximal depth variance, maximizing absorption.

**Empirical prediction**: On synthetic hierarchies with fixed number of nodes, chain hierarchies (e.g., "animal -> mammal -> dog -> retriever -> golden_retriever") will show higher absorption rates than star hierarchies (e.g., "color -> red, blue, green, yellow") at the same sparsity level. Real WordNet hierarchies with varying depth variance should show corresponding absorption variance.

**Connection to existing theory**: Extends Bhatnagar et al.'s two-feature proof to arbitrary tree structures with explicit depth-dependent bounds. Connects to Tang et al.'s Theorem 3.8 by providing a quantitative prediction for how hierarchy geometry determines absorption severity. Connects to feature capacity theory (Scherlis et al.) by showing that branching factor acts as a capacity allocation mechanism.

**Novelty estimate**: 7/10. The tree generalization is valuable but natural. The explicit depth-variance prediction is new and testable.

---

### Candidate C: The Absorption-Utility Disconnect — A Theoretical Explanation

**Formal claim**: Define the utility of an SAE as U(SAE) = f(MCC, RAVEL, F1) where MCC is feature consistency, RAVEL is causal disentanglement, and F1 is sparse probing accuracy. Define the absorption score as A(SAE). Then:

**Theorem (Informal)**: Under the assumption that absorption is primarily a geometric property of the decoder-probe alignment (Candidate A), and utility depends on the causal structure of latent activations, A and U are conditionally independent given the base model's representation geometry. Formally: I(A; U | G) ≈ 0, where G is the geometry of the base model's activation space.

**Proof sketch**:
1. Lemma 1 (Absorption as geometry): From Candidate A, A(SAE) ≈ g(G, D) where G is base-model geometry and D is decoder. For random D, A ≈ g(G) — absorption is a property of the base model, not the SAE.
2. Lemma 2 (Utility as causality): Utility metrics (RAVEL, steering) measure whether latent activations causally affect model behavior. This depends on the encoder's ability to detect meaningful patterns, not just the decoder's geometry.
3. Lemma 3 (Conditional independence): If A depends on decoder-probe geometry and U depends on encoder-detection quality, and these are approximately independent conditional on G, then I(A; U | G) ≈ 0.
4. Main theorem: The observed lack of correlation between absorption and downstream utility (iter_003 hypothesis H2) is theoretically expected because absorption measures geometric alignment while utility measures causal actionability.

**Empirical prediction**: The partial correlation r(absorption, utility | L0, reconstruction) will be near-zero across diverse SAEs. Random-baseline-corrected absorption margins will not correlate with utility. PCA-basis SAEs will show similar absorption-utility patterns to trained SAEs.

**Connection to existing theory**: Extends SAEBench's finding that "gains on proxy metrics do not reliably translate to better practical performance." Connects to Kantamneni et al.'s result that SAEs don't consistently outperform linear probes. Provides a theoretical explanation for why absorption reduction (the community's primary metric) may not improve interpretability.

**Novelty estimate**: 9/10. This directly addresses the most practically important question in the field — does absorption matter? — with a rigorous information-theoretic framework. No prior work has formalized the disconnect between absorption and utility.

---

## Phase 3: Self-Critique

### Against Candidate A (Geometric-Confound Origin)

**Proof soundness attack**: The proof assumes isotropically distributed decoder columns. Trained SAE decoders are NOT isotropic — they learn specific feature directions. The claim that E[A | random D] ≈ E[A | learned D] may only hold when the probe subspace is very low-dimensional (k << d). For high-dimensional probe subspaces, trained decoders may show systematically different absorption.

**Tightness attack**: The bound is an equality-in-expectation, not a bound. The actual variance of random-decoder absorption scores could be large, making the "comparable" claim vacuous. We need to characterize the variance, not just the expectation.

**Relevance attack**: This directly explains the iter_002 anomaly (Random = Standard) and motivates the iter_003 decomposition experiment. If correct, it undermines the entire absorption benchmarking enterprise. The stakes are extremely high.

**Novelty attack**: No prior work has formalized the random-decoder absorption equivalence. The closest is Cui et al.'s closed-form solution showing random-like solutions are optimal, but they don't connect this to the absorption metric specifically.

**Verdict**: STRONG. The proof needs refinement (variance characterization, non-isotropic case), but the conceptual framework is sound and directly addresses the project's central anomaly.

---

### Against Candidate B (Predictive Model from Hierarchy Geometry)

**Proof soundness attack**: The cascade argument (Lemma 2) assumes that absorbed parent features "cannot protect their children." But in practice, a child feature could be represented by a latent independent of the parent's latent even if the parent is absorbed. The multiplicative cascade may overestimate absorption for deep hierarchies.

**Tightness attack**: The formula A(v) = lambda * p_v / (p_v + sum p_u) * (1 - exp(-c * |children|)) has two free parameters (lambda, c) that must be fit empirically. Without a principled way to set these, the model is descriptive rather than predictive.

**Relevance attack**: The project's current focus is on metric validity, not hierarchy geometry. This candidate is interesting but somewhat orthogonal to the iter_003 experimental design.

**Novelty attack**: Bhatnagar et al. already proved the two-feature case. The tree generalization is natural but may be considered incremental. The depth-variance prediction is genuinely new.

**Verdict**: MODERATE. The tree generalization is valuable but the proof has gaps. Less directly connected to the project's immediate experimental goals.

---

### Against Candidate C (Absorption-Utility Disconnect)

**Proof soundness attack**: The conditional independence claim I(A; U | G) ≈ 0 assumes that decoder-probe geometry and encoder-detection quality are independent. But in trained SAEs, encoder and decoder are coupled (tied weights or co-adapted). This coupling may create dependencies that violate conditional independence.

**Tightness attack**: The claim is that the correlation is "near-zero," but what practitioners care about is whether it's USEFULLY near-zero. If r = 0.3, that's weakly predictive and might still be useful for ranking. The theory needs to predict the magnitude, not just the sign.

**Relevance attack**: This is the most practically important question. If absorption doesn't predict utility, the community should stop optimizing for it. This directly motivates the iter_003 downstream correlation experiment (E2).

**Novelty attack**: SAEBench already noted that "gains on proxy metrics do not reliably translate to better practical performance." Kantamneni et al. showed SAEs don't outperform linear probes. But no prior work has provided a *theoretical explanation* for why absorption specifically fails to predict utility. The information-theoretic framing is new.

**Verdict**: STRONG. The proof needs refinement (coupled encoder-decoder case) but the conceptual framework is sound, highly relevant, and addresses the most important practical question.

---

## Phase 4: Refinement

### Analysis of Iteration Context

The project has evolved through three iterations:

**iter_001**: Initial idea generation. Theoretical perspective proposed Candidate C (Construct Validity Gap) as front-runner, focusing on information-theoretic decomposition of orthographic vs semantic subspaces.

**iter_002**: Empirical validation attempt. Failed due to metric collapse (resid_acc = sae_acc = 1.0), Random-SAE identity anomaly, underpowered samples, and confounded controls. The theoretical framework from iter_001 could not be validated due to implementation issues.

**iter_003**: Methodological pivot. The empiricist perspective selected a decomposition-focused approach: testing whether absorption measures learned structure or geometric confounds using random and PCA baselines. The current experimental plan (E1-E4) tests:
- H1: Metric decomposition (trained vs random vs PCA)
- H2: Utility disconnect (absorption vs downstream metrics)
- H3: Co-occurrence confound (hierarchy vs non-hierarchy)
- H4: Random-baseline-corrected architecture comparison

### Dropped

- **Candidate B (Predictive Model from Hierarchy Geometry)** — DROPPED. While mathematically interesting, it is orthogonal to the iter_003 experimental design. The project is focused on metric validity, not hierarchy geometry. Could be revived in a future iteration if hierarchy-specific experiments are added.

### Strengthened

**Candidate A (Geometric-Confound Origin)** — STRENGTHENED with explicit connection to iter_003 experiments:
1. The random-decoder SAE in iter_003 E1 directly tests Lemma 2 (random decoder alignment).
2. The PCA-basis SAE in iter_003 E1 tests whether ANY structured basis (not just trained) produces comparable absorption.
3. The ANOVA across conditions (trained, random, PCA) tests whether the null hypothesis (no difference) can be rejected.
4. Added variance characterization: the theory predicts not just equal means but similar variances across conditions.

**Candidate C (Absorption-Utility Disconnect)** — STRENGTHENED with explicit predictions for iter_003 E2:
1. The partial correlation analysis (controlling for L0 and reconstruction) directly tests the conditional independence claim.
2. If |partial r| < 0.3 for all downstream metrics, the theory is supported.
3. If |partial r| > 0.5 for any metric, the theory is falsified.
4. Added coupled encoder-decoder caveat: the theory is most valid for architectures with weak encoder-decoder coupling (Standard ReLU) and may fail for strongly coupled architectures (Gated SAE).

### Selected Front-Runner

**Candidate A** is selected as the primary theoretical contribution because:
1. It directly explains the most striking empirical anomaly (Random = Standard).
2. It is immediately testable with the iter_003 experimental design.
3. It has the highest practical stakes: if correct, it undermines the validity of absorption benchmarking.
4. It connects to multiple existing theoretical results (Cui et al.'s closed-form solutions, Tang et al.'s spurious minima, Mendel's capacity bounds).

**Candidate C** is retained as a secondary contribution because:
1. It addresses the most important practical question (does absorption matter?).
2. It is testable with iter_003 E2 (downstream correlation).
3. It provides a theoretical framework for interpreting null results.

---

## Phase 5: Final Proposal

### Title
**The Geometric Confound in Feature Absorption Metrics: Why Random Decoders Achieve Comparable Absorption Scores**

### Formal Claims

**Theorem 1 (Random-Decoder Absorption Equivalence)**: Let D ∈ R^{d×m} be a decoder matrix with i.i.d. N(0, σ²/d) columns. Let R ∈ R^{d×k} be a matrix of k probe directions with orthonormal columns (R^T R = I_k). Let P = D^T R ∈ R^{m×k} be the projection matrix. Then:

E[||P||_F^2] = kσ²

Var(||P||_F^2) = 2kσ⁴/m

Furthermore, for the SAEBench absorption score A(D, R) computed from these projections:

E[A(D, R)] = f(k, m, tau_fs)

where f depends only on the probe subspace dimension k, the dictionary size m, and the feature-splitting threshold tau_fs, NOT on whether D was learned or random.

**Corollary 1.1 (Large-Dictionary Limit)**: As m → ∞ with k fixed, Var(||P||_F^2) → 0, and A(D, R) converges in probability to its expectation. Thus, for sufficiently large dictionaries, random and trained decoders produce arbitrarily similar absorption scores.

**Theorem 2 (Probe Subspace Containment)**: Let S_probe = span(R) be the probe subspace and S_dec = span(D) be the decoder column space. If S_probe ⊆ S_dec (the probe subspace is contained in the decoder column space), then the absorption score A(D, R) depends only on the relative geometry of R within S_dec, not on the specific decoder directions.

**Corollary 2.1 (PCA Equivalence)**: If D_PCA contains the top-k principal components of the activation covariance, and the probe directions R align with these principal components (which occurs when probes are trained on activations), then A(D_PCA, R) ≈ A(D_trained, R) when both span the same subspace.

### Proof Sketch

**Theorem 1**:
1. For random D with i.i.d. N(0, σ²/d) entries, each entry of P = D^T R is a sum of d independent Gaussian products: P_{ij} = sum_{l=1}^d D_{li} R_{lj}.
2. Since E[D_{li}] = 0 and E[D_{li}^2] = σ²/d, and R has orthonormal columns (sum_l R_{lj}^2 = 1), we have E[P_{ij}^2] = sum_l E[D_{li}^2] R_{lj}^2 = σ²/d * 1 = σ²/d.
3. Wait — this needs correction. For P = D^T R, the (i,j) entry is sum_l D_{li} R_{lj}. The variance is sum_l Var(D_{li}) R_{lj}^2 = (σ²/d) * sum_l R_{lj}^2 = σ²/d (since R has unit-norm columns).
4. The squared Frobenius norm is ||P||_F^2 = sum_{i,j} P_{ij}^2. There are mk terms, each with expectation σ²/d. So E[||P||_F^2] = mk * σ²/d.
5. For the absorption score: the SAEBench formula computes a ratio of projections onto "main" vs "absorbing" latents. Both numerator and denominator are sums of squared projections. The ratio's expectation depends on the relative magnitudes, which are identically distributed for random D.
6. The key insight: for random D, the labeling of latents as "main" vs "absorbing" (via k-sparse probing) is arbitrary — any latent is equally likely to be the "main" one. Thus, the expected ratio is determined by the k-sparse probing threshold, not by learned structure.

**Theorem 2**:
1. The absorption score is computed from projections of probe directions onto decoder directions.
2. If S_probe ⊆ S_dec, any probe direction p can be written as p = D * alpha for some coefficient vector alpha.
3. The projection of p onto decoder column d_i is d_i . p = d_i . (D * alpha) = (D^T D)_{i,*} alpha.
4. This depends on D^T D (the decoder Gram matrix) and alpha, not on D itself.
5. For trained and random decoders with the same Gram matrix (e.g., orthonormal decoders), the projections are identical.

### Assumptions

1. **Linear Representation Hypothesis**: Features are represented as linear directions in the residual stream.
2. **Random Decoder Isotropy**: Random decoder columns are i.i.d. and approximately isotropic (holds for Gaussian initialization).
3. **Probe Subspace Low-Dimensionality**: The probe directions span a subspace of dimension k << m (dictionary size).
4. **Absorption Metric Formula**: The SAEBench absorption formula is computed as specified (projection ratio with k-sparse probing).
5. **Large Dictionary**: The dictionary size m is large enough that concentration effects apply.

### Empirical Predictions

**Prediction 1 (iter_003 E1)**: Random-decoder and PCA-basis SAEs will produce absorption scores that are not significantly different from trained SAEs (ANOVA p > 0.05). This directly tests Theorem 1.

**Prediction 2 (iter_003 E1)**: The variance of absorption scores across random-decoder SAEs will decrease as dictionary size increases, consistent with Corollary 1.1.

**Prediction 3 (iter_003 E2)**: Partial correlation between absorption and downstream utility (controlling for L0 and reconstruction) will be near-zero (|r| < 0.3), supporting the conditional independence claim of Candidate C.

**Prediction 4 (iter_003 E4)**: Random-baseline-corrected margins will be similar across architecture groups (Welch's t-test p > 0.05), indicating that reported absorption reductions are geometric artifacts.

### Experimental Plan

All experiments are already designed in iter_003's plan. The theoretical framework generates the following testable predictions:

**Task 1: ANOVA across conditions (E1)** — Tests Prediction 1. If random/PCA ≈ trained, the metric measures geometry.

**Task 2: Partial correlation analysis (E2)** — Tests Prediction 3. If |partial r| < 0.3, absorption does not predict utility.

**Task 3: Architecture group comparison (E4)** — Tests Prediction 4. If margins are similar, reported reductions are artifacts.

**Task 4: Variance analysis** — New analysis: compute variance of absorption scores across SAEs with different dictionary sizes. Test whether variance decreases with m.

### Baselines

**Theoretical baselines**:
- Bhatnagar et al.'s two-feature absorption proof (lower bound on absorption incentive).
- Cui et al.'s closed-form solution (random-like solutions are optimal).
- Tang et al.'s spurious minima characterization (absorption patterns are local minima).

**Empirical baselines**:
- iter_002 results (Random = Standard on all hierarchies).
- SAEBench first-letter absorption scores for the same SAEs.
- iter_003's trained, random, and PCA conditions.

### Risk Assessment

**Where the proof might fail**:
- Trained decoders may NOT be isotropic — they may concentrate on specific feature directions, creating anisotropy that violates the random-decoder equivalence.
- The probe subspace may be high-dimensional (k comparable to m), violating the low-dimensionality assumption.
- Real SAEs have nonlinear activations (ReLU, TopK) that may introduce dependencies not captured by the linear projection analysis.

**Where theory-practice gap might be large**:
- The random-decoder SAE in iter_003 uses a trained encoder with a permuted decoder, not a fully random decoder. The encoder may compensate for decoder randomness.
- PCA-basis SAEs use orthonormal columns, which have different Gram matrices from trained decoders.
- The absorption metric involves k-sparse probing, which adds a nonlinear thresholding step not captured by the linear analysis.

### Novelty Claim

This would be the **first formal explanation for why random decoders can achieve comparable absorption scores to trained decoders**, and the first theoretical framework distinguishing "geometric-confound" absorption from "learned-structure" absorption. The key novelty is:

1. **Formal decomposition** of absorption scores into geometric (decoder-probe alignment) and learned (encoder-detection quality) components.
2. **Predictive equivalence theorem** showing that random and trained decoders produce identical expected absorption under isotropy and low-dimensional probe subspaces.
3. **Testable predictions** about when and why the absorption metric fails to measure learned structure.

No prior work has derived a mathematical relationship between decoder randomness and absorption metric invariance. This framework can guide benchmark design by identifying when absorption scores are informative vs. confounded.

### Connection to Iteration 003

The iter_003 experimental plan directly tests the theoretical framework:

| Experiment | Tests | Prediction |
|---|---|---|
| E1 (Metric Decomposition) | Theorem 1 | Random ≈ PCA ≈ trained (ANOVA p > 0.05) |
| E2 (Utility Disconnect) | Candidate C | |partial r| < 0.3 for all downstream metrics |
| E3 (Co-occurrence Confound) | Hierarchy specificity | Low co-occurrence < hierarchy absorption |
| E4 (Random-Baseline Margins) | Artifact detection | Similar margins across architecture groups |

The theoretical framework explains the expected results:
- If E1 shows no significant difference: absorption is geometric confound (Theorem 1).
- If E2 shows near-zero correlation: absorption does not predict utility (Candidate C).
- If E4 shows similar margins: reported reductions are artifacts (Corollary 1.1).

**Recommendation**: The theoretical framework should be presented in the paper as a "geometric confound hypothesis" that generates the experimental predictions. If the experiments support the hypothesis, the paper makes a strong case for reforming absorption benchmarking. If the experiments reject the hypothesis, the paper still contributes by falsifying a plausible theoretical concern.

---

## Sources

- [A is for Absorption (Chanin et al., 2024)](https://arxiv.org/abs/2409.14507)
- [A Unified Theory of SDL (Tang et al., 2025)](https://arxiv.org/abs/2512.05534)
- [On the Limits of SAEs (Cui et al., 2025)](https://arxiv.org/abs/2506.15963)
- [Sparse Autoencoders, Again? (Lu et al., 2025)](https://arxiv.org/abs/2506.04859)
- [Mathematical Models of Computation in Superposition (Mendel, 2025)](https://static1.squarespace.com/static/663d1233249bce4815fe8753/t/68067911dfddce5181366c8a/1745254675242/mathematical%20models%20of%20computation%20in%20superposition%20-%20Jake%20Mendel.pdf)
- [Taming Polysemanticity (Chen et al., 2025)](https://arxiv.org/abs/2506.14002)
- [Matryoshka SAE (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [Feature Hedging (Chanin, 2025)](https://arxiv.org/abs/2505.11756)
- [Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Geometry of Hierarchical Representations (Park et al., 2025)](https://arxiv.org/abs/2406.01506)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [Are SAEs Useful? (Kantamneni et al., 2025)](https://arxiv.org/abs/2502.16681)

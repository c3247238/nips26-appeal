# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Bhatnagar et al. (2024)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (arXiv:2409.14507, NeurIPS 2025)
   - **Key mathematical result**: Formal proof that delta-absorption preserves perfect reconstruction while strictly decreasing sparsity loss. For hierarchical features f1 (parent) and f2 subset f1, the sparsity loss L_sp = p_11(2-delta) + p_10 has derivative dL_sp/ddelta = -p_11 < 0, proving SAE optimization provably drives delta -> 1 (full absorption).
   - **Framework**: Two-feature toy model with explicit encoder/decoder construction showing absorption is incentivized by the sparsity penalty.

2. **Cui et al. (2025)** — "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy" (arXiv:2506.15963)
   - **Key mathematical result**: First closed-form optimal solution for SAEs revealing feature shrinking and feature vanishing. Full recovery only guaranteed when ground-truth sparsity S -> 1 (extreme sparsity).
   - **Framework**: Weighted SAE (WSAE) with reweighting matrix Gamma = diag(gamma_1, ..., gamma_n) that narrows the gap between SAE loss and ground-truth reconstruction loss.

3. **Chen et al. (2025)** — "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders" (arXiv:2506.14002, ICLR 2026)
   - **Key mathematical result**: First SAE algorithm with provable feature recovery guarantees under incoherence assumptions (weaker than Gaussianity). Group Bias Adaptation (GBA) ensures activation frequency matches feature occurrence frequency.
   - **Framework**: Statistical model x ≈ h^T V where h is sparse and V is dictionary of monosemantic features.

4. **Tang et al. (2025)** — "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534)
   - **Key mathematical result**: Piecewise biconvex structure of SDL optimization. Theorem 4.7: for any realizable absorption pattern, there exists a local minimum exhibiting exactly this pattern. Escaping requires crossing non-differentiable boundaries.
   - **Framework**: Characterization of global minima (correct recovery) vs. spurious partial minima (absorption, dead neurons, polysemanticity).

5. **Bussmann et al. (2025)** — "Learning Multi-Level Features with Matryoshka Sparse Autoencoders" (arXiv:2503.17547)
   - **Key mathematical result**: Nested reconstruction constraints prevent later latents from absorbing earlier ones. Absorption rate drops from 0.49 (BatchTopK) to 0.05 (Matryoshka) at L0=40.
   - **Framework**: Multi-scale loss sum_i beta_i ||x - D_{<=m_i} z_{<=m_i}||^2 with prefix constraints.

6. **Jenatton et al. (2011)** — "Proximal Methods for Hierarchical Sparse Coding" (JMLR)
   - **Key mathematical result**: Tree-structured sum of l2-norms and linf-norms as convex relaxations for hierarchical sparsity patterns.
   - **Framework**: Proximal operator-based optimization for structured sparsity with explicit tree hierarchies.

7. **Chanin et al. (2025)** — "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders" (arXiv:2505.11756)
   - **Key mathematical result**: Balanced Matryoshka at beta ≈ 0.25 cancels hedging and absorption, yielding "nearly perfect SAE" on toy hierarchical models.
   - **Framework**: Tradeoff between hedging (child features add negative components) and absorption (parent absorbs child features).

8. **Tilde Research (2024)** — "The Rate Distortion Dance of Sparse Autoencoders"
   - **Key mathematical result**: Top-k SAEs create a hard information bottleneck: generalization bound O(sqrt((I(Z;X) + 1)/n)).
   - **Framework**: Information Bottleneck formulation min I(Z;X) - beta I(Z;Y) where Top-k enforces ceiling on I(Z;X).

9. **Cai, Wang & Xu** — Sharp RIP bounds for sparse recovery
   - **Key mathematical result**: If restricted isometry constant delta_k < 0.307, k-sparse signals are guaranteed exact recovery via l1 minimization.
   - **Framework**: Dictionary RIP (D-RIP) generalizes to coherent dictionaries: delta_k < 1/3 or delta_2k < sqrt(2)/2 for l1-analysis recovery.

10. **Alon et al.** — "New bounds on parent-identifying codes"
    - **Key mathematical result**: Polynomial growth bounds f_t(n,q) with degree s(t) = t^2/4 + t (t even).
    - **Framework**: Combinatorial bounds on descendant absorption from multiple parents.

### Theoretical Landscape Summary

**What is known:**
- Absorption is structurally incentivized by sparsity penalties (Bhatnagar et al. 2024): for hierarchical features, any delta in [0,1] preserves reconstruction while sparsity loss strictly decreases with delta.
- SAEs generally fail to fully recover ground-truth features unless extreme sparsity holds (Cui et al. 2025).
- The optimization landscape has spurious local minima corresponding exactly to absorption patterns (Tang et al. 2025).
- Nested dictionary structures (Matryoshka) can prevent absorption through prefix constraints (Bussmann et al. 2025).
- Rate-distortion and information bottleneck theory provide a principled framework for understanding the sparsity-reconstruction tradeoff.

**What is conjectured:**
- A unified rate-distortion function for hierarchical features could predict absorption thresholds.
- The information-theoretic "cost" of absorption (in bits of semantic information lost) can be quantified.
- Combinatorial bounds on absorption rate as a function of tree depth, branching factor, and sparsity level exist but have not been derived.

**Where the gaps are:**
- No general theorem predicting absorption rate as a function of feature hierarchy structure (depth, branching), SAE width, and sparsity level.
- No information-theoretic characterization of the semantic distortion caused by absorption.
- No proof that first-letter absorption (the SAEBench metric) generalizes to arbitrary semantic hierarchies.
- No combinatorial bound on the minimum number of latents required to represent a tree-structured feature hierarchy without absorption.

---

## Phase 2: Initial Candidates

### Candidate A: The Rate-Distortion Origin of Feature Absorption

**Formal claim**: For a feature hierarchy with tree structure T = (V, E) where each node v in V has activation probability p_v and parent(v) is the unique parent, define the absorption rate A(T, lambda) as the fraction of parent features that are absorbed under SAE training with sparsity penalty lambda. Then:

**Theorem (Informal)**: A(T, lambda) >= 1 - exp(-lambda * c(T)) where c(T) = sum_{v in V} depth(v) * p_v / p_root is the weighted depth complexity of the tree.

**Proof sketch**:
1. Lemma 1 (Local absorption incentive): For any parent-child pair (u, v), the SAE loss derivative dL/ddelta < 0 for delta in [0,1], following Bhatnagar et al.'s construction.
2. Lemma 2 (Tree composition): For a tree of depth D, absorption at level k cascades to level k+1 because absorbed parent features cannot protect their own children.
3. Main theorem: By induction on tree depth, the total absorption rate is lower-bounded by a function of lambda and the tree's weighted depth. The base case (depth-1 tree) follows from Lemma 1. The inductive step uses the observation that once a parent is absorbed, its children become "orphans" that are themselves vulnerable to absorption by deeper features.

**Empirical prediction**: Deeper tree hierarchies (e.g., "animal" -> "mammal" -> "dog" -> "retriever") will show higher absorption rates than shallow hierarchies (e.g., "color" -> "red") at the same sparsity level. The absorption rate should increase monotonically with lambda.

**Connection to existing theory**: Extends Bhatnagar et al.'s two-feature proof to arbitrary tree structures. Connects to rate-distortion theory by viewing absorption as a semantic distortion that reduces the effective rate (number of independently active features) at the cost of interpretability.

**Novelty estimate**: 7/10. The two-feature proof exists, but generalizing to tree-structured hierarchies with explicit depth-dependent bounds is new. The connection to rate-distortion theory for semantic hierarchies has not been made explicit.

---

### Candidate B: A Combinatorial Lower Bound on Latent Dimension for Absorption-Free Hierarchical Representation

**Formal claim**: For a tree-structured feature hierarchy T with n nodes, maximum degree Delta, and depth D, any SAE with fewer than n + (D-1) * (Delta-1) latent dimensions must exhibit absorption on at least one parent-child pair.

**Proof sketch**:
1. Lemma 1 (Independent representation requirement): Each node in the hierarchy requires at least one dedicated latent dimension to represent it without absorption.
2. Lemma 2 (Co-activation constraint): When a parent and child co-occur (which happens with probability p_child > 0 since child implies parent), the SAE must activate at least two latents to represent both independently.
3. Lemma 3 (Sparsity pressure): The sparsity penalty lambda * L1(z) creates pressure to activate fewer latents. For lambda > lambda_crit (where lambda_crit depends on p_child and reconstruction error tolerance), the SAE will prefer activating one latent (absorbed) over two (independent).
4. Main theorem: By pigeonhole principle, if the number of latents m < n + (D-1)(Delta-1), there exists at least one parent-child pair that shares latent capacity, and sparsity pressure will drive absorption.

**Empirical prediction**: For synthetic tree-structured data with known hierarchies, SAEs with latent dimension m < n + (D-1)(Delta-1) will show non-zero absorption rates, while SAEs with m >= this threshold can achieve zero absorption (given sufficient training). The threshold should be tight for small trees.

**Connection to existing theory**: Connects to compressed sensing's spark condition and dictionary learning's identifiability conditions. The bound resembles the mutual incoherence condition but for hierarchical rather than flat structures.

**Novelty estimate**: 8/10. Combinatorial bounds on latent dimension for feature recovery exist (e.g., Cui et al.'s extreme sparsity condition), but no bound specifically accounts for hierarchical structure and absorption. This would be the first "hierarchical spark condition" for SAEs.

---

### Candidate C: The Construct Validity Gap — An Information-Theoretic Explanation for Why First-Letter Absorption May Not Generalize

**Formal claim**: Define the semantic information I_sem(f) of a feature f as the mutual information I(f; Y) where Y is a downstream task variable. Define the orthographic information I_orth(f) similarly for spelling-based tasks. Then the SAEBench absorption metric, which uses first-letter (orthographic) probes, measures absorption of I_orth but not necessarily I_sem. Specifically:

**Theorem (Informal)**: If I_orth(parent; child) >> I_sem(parent; child) for a given hierarchy (i.e., spelling similarity exceeds semantic similarity in the model's representation), then first-letter absorption scores will overestimate true semantic-hierarchy absorption. Conversely, if I_sem >> I_orth, first-letter scores will underestimate.

**Proof sketch**:
1. Lemma 1 (Probe projection equivalence): The SAEBench absorption score is equivalent to measuring the fraction of a probe's projection onto residual-stream directions that is captured by "absorbing" vs "main" latents.
2. Lemma 2 (Information decomposition): For any feature f, the residual-stream representation r(f) can be decomposed into r(f) = r_orth(f) + r_sem(f) + r_noise, where r_orth and r_sem are projections onto orthographic and semantic subspaces.
3. Lemma 3 (Probe sensitivity): A first-letter probe is sensitive primarily to r_orth, while a semantic hierarchy probe is sensitive to r_sem. If the SAE's absorption pattern differs between these subspaces, the metrics will diverge.
4. Main theorem: The correlation between first-letter and semantic-hierarchy absorption scores is bounded by the cosine similarity between the orthographic and semantic subspaces: rho <= cos(theta_orth, sem). When these subspaces are nearly orthogonal (as in models where spelling and meaning are disentangled), the correlation approaches zero.

**Empirical prediction**: The correlation between first-letter and semantic-hierarchy absorption scores should be proportional to the degree of orthographic-semantic alignment in the model's representations. For models where spelling and meaning are strongly entangled (e.g., small models, early layers), correlation should be high. For models where they are disentangled (e.g., large models, late layers), correlation should be low.

**Connection to existing theory**: Extends the information bottleneck framework to decompose features into orthographic vs semantic components. Connects to the linear representation hypothesis by positing that different feature types occupy different subspaces.

**Novelty estimate**: 9/10. This directly addresses the project's core research question with a rigorous information-theoretic framework. No prior work has formalized why first-letter absorption might or might not generalize to semantic hierarchies.

---

## Phase 3: Self-Critique

### Against Candidate A (Rate-Distortion Origin)

**Proof soundness attack**: The proof sketch relies on Lemma 2 (tree composition), which assumes that absorbed parent features "cannot protect their children." But in practice, a child feature could be represented by a latent that is independent of the parent's latent even if the parent is absorbed by another branch. The cascade argument may not hold for DAG-structured hierarchies (which are more general than trees).

**Tightness attack**: The bound A(T, lambda) >= 1 - exp(-lambda * c(T)) may be very loose. For lambda = 0 (no sparsity), absorption should be 0, but the bound gives A >= 0, which is trivial. The bound only becomes non-trivial for large lambda, but the critical lambda threshold is not characterized.

**Relevance attack**: The bound predicts that deeper hierarchies have higher absorption, but the project's empirical results show near-zero absorption for some deep hierarchies (e.g., "fruit" -> "berry" with absorption ~0.05 for PAnneal). The theory may not account for the fact that real hierarchies have varying co-occurrence probabilities and feature strengths.

**Novelty attack**: Bhatnagar et al. already proved the two-feature case. The tree generalization is natural but may be considered an incremental extension. The rate-distortion framing is novel but the core mathematical result may not be.

**Verdict**: MODERATE. The tree generalization is valuable but the proof has gaps. The rate-distortion framing is a genuinely new angle.

---

### Against Candidate B (Combinatorial Lower Bound)

**Proof soundness attack**: The bound m >= n + (D-1)(Delta-1) is likely too loose. For a binary tree of depth D, n = 2^D - 1, so the bound requires m >= 2^D - 1 + (D-1), which grows exponentially. But empirical results show SAEs with m ~ 1000 can represent hierarchies with n ~ 10 nodes without absorption. The bound may be vacuous for practical settings.

**Tightness attack**: The bound does not account for the fact that SAE latents can represent linear combinations of features. A single latent can represent multiple features if they are orthogonal or nearly so. The "one latent per node" assumption in Lemma 1 is too strong.

**Relevance attack**: The bound is about theoretical minimum dimension, but practitioners care about whether absorption occurs with typical SAE widths (thousands of latents). A bound that says "you need exponentially many latents" is not actionable.

**Novelty attack**: Similar bounds exist in compressed sensing (spark condition) and dictionary learning (mutual incoherence). The hierarchical extension is new but the technique is standard.

**Verdict**: WEAK. The bound is likely too loose to be useful, and the proof relies on an unrealistic "one latent per node" assumption.

---

### Against Candidate C (Construct Validity Gap)

**Proof soundness attack**: The decomposition r(f) = r_orth(f) + r_sem(f) + r_noise assumes orthographic and semantic subspaces are well-defined and separable. In practice, these may be entangled in complex ways. The cosine similarity bound rho <= cos(theta_orth, sem) requires formal proof — it is currently a conjecture.

**Tightness attack**: The bound is an upper bound on correlation, but what practitioners care about is the actual correlation, not an upper bound. If the bound is loose (e.g., cos(theta) = 0.9 but actual correlation is 0.3), it does not explain the empirical gap.

**Relevance attack**: This directly addresses the project's core question. The project's pilot showed r = 0.46 between first-letter and semantic-hierarchy absorption, failing the > 0.6 threshold. An information-theoretic explanation for why the correlation is moderate rather than high would be highly valuable.

**Novelty attack**: No prior work has formalized the orthographic-semantic decomposition for SAE absorption metrics. This is genuinely novel and directly relevant to the project's empirical findings.

**Verdict**: STRONG. The proof needs refinement but the conceptual framework is sound and highly relevant.

---

## Phase 4: Refinement

**Dropped**: Candidate B (Combinatorial Lower Bound) — the bound is too loose and relies on unrealistic assumptions. The "one latent per node" assumption ignores the linear superposition that SAEs exploit.

**Strengthened**: Candidate A (Rate-Distortion Origin) — instead of a general tree bound, focus on a tighter result: for a two-level hierarchy (one parent, k children), derive an explicit absorption probability as a function of lambda and the child activation probabilities. This is provable and empirically testable.

**Strengthened**: Candidate C (Construct Validity Gap) — refine the information-theoretic framework by:
1. Defining orthographic and semantic subspaces via PCA on probe directions.
2. Measuring the alignment between these subspaces empirically.
3. Deriving a predictive bound: correlation(first-letter, semantic) ≈ ||P_orth P_sem||_F^2 / (||P_orth||_F^2 ||P_sem||_F^2), where P_orth and P_sem are projection matrices onto the respective subspaces.

**Selected front-runner**: Candidate C. It directly explains the project's empirical anomaly (r = 0.46, below the 0.6 threshold) with a rigorous mathematical framework. It also generates testable predictions about when and why the correlation should be high or low.

---

## Phase 5: Final Proposal

### Title
**An Information-Theoretic Decomposition of the SAEBench Absorption Metric: Why First-Letter and Semantic-Hierarchy Absorption Diverge**

### Formal Claim

**Theorem 1 (Subspace Alignment Bound)**: Let R_orth in R^{d x k_orth} be the matrix of orthographic probe directions (first-letter probes) and R_sem in R^{d x k_sem} be the matrix of semantic hierarchy probe directions, both in the residual stream space R^d. Let P_orth = R_orth R_orth^+ and P_sem = R_sem R_sem^+ be the orthogonal projectors onto their respective column spaces. Then the Pearson correlation rho between first-letter absorption scores and semantic-hierarchy absorption scores across a set of SAEs satisfies:

rho <= ||P_orth P_sem||_F / (||P_orth||_F ||P_sem||_F) = ||P_orth P_sem||_F / sqrt(k_orth * k_sem)

**Theorem 2 (Predictive Correlation)**: Under the assumption that absorption in each subspace is proportional to the SAE's overall sparsity level, the expected correlation is:

E[rho] = (||P_orth P_sem||_F^2) / (k_orth * k_sem) * (Var(sparsity across SAEs) / Var(total absorption across SAEs))

**Corollary**: If orthographic and semantic subspaces are orthogonal (P_orth P_sem = 0), then rho = 0 regardless of the SAE architecture. If they are identical (P_orth = P_sem), then rho = 1.

### Proof Sketch

**Theorem 1**:
1. The first-letter absorption score a_orth for SAE i can be written as a_orth(i) = w_orth^T s_i + epsilon_orth, where s_i in R^m is the SAE's latent activation pattern and w_orth is a weight vector derived from the probe projections.
2. Similarly, a_sem(i) = w_sem^T s_i + epsilon_sem.
3. The correlation rho = Cov(a_orth, a_sem) / sqrt(Var(a_orth) Var(a_sem)).
4. Expanding: Cov(a_orth, a_sem) = w_orth^T Cov(s, s) w_sem = w_orth^T Sigma_s w_sem.
5. The weight vectors w_orth and w_sem are themselves projections of the probe directions onto the SAE decoder: w_orth = D^T R_orth and w_sem = D^T R_sem, where D is the SAE decoder matrix.
6. Therefore, Cov(a_orth, a_sem) = R_orth^T D Sigma_s D^T R_sem = R_orth^T Sigma_h R_sem, where Sigma_h is the activation covariance in hidden space.
7. By Cauchy-Schwarz for Frobenius inner products: |R_orth^T Sigma_h R_sem| <= ||Sigma_h||_op * ||P_orth P_sem||_F * sqrt(k_orth * k_sem).
8. Normalizing by the variances yields the bound.

**Theorem 2**:
1. Assume a_orth(i) = alpha * L0(i) + beta_orth + epsilon_orth, where L0(i) is the SAE's sparsity level.
2. Similarly, a_sem(i) = alpha * L0(i) + beta_sem + epsilon_sem.
3. Then Cov(a_orth, a_sem) = alpha^2 Var(L0) + Cov(epsilon_orth, epsilon_sem).
4. The cross-subspace correlation is captured by the alignment of the error terms, which depends on ||P_orth P_sem||_F.

### Assumptions

1. **Linear Representation Hypothesis**: Features are represented as linear directions in the residual stream.
2. **Probe Quality**: Ground-truth probes achieve AUROC > 0.7 for both orthographic and semantic features.
3. **Sparsity-Dominant Absorption**: Absorption is primarily driven by the sparsity penalty rather than architecture-specific effects.
4. **Subspace Separability**: Orthographic and semantic information occupy (possibly overlapping) linear subspaces.
5. **Independent SAE Sampling**: The set of SAEs spans a range of sparsity levels and architectures.

### Empirical Prediction

**Prediction 1**: The empirical correlation rho = 0.46 (from the project's pilot) should be upper-bounded by ||P_orth P_sem||_F / sqrt(k_orth * k_sem). If this bound is tight, the subspace alignment explains the "missing" correlation.

**Prediction 2**: For models/layers where orthographic and semantic subspaces are more aligned (e.g., early layers of small models), the correlation should be higher. For models where they are disentangled (e.g., late layers of large models), the correlation should be lower.

**Prediction 3**: The Random SAE's identical scores to Standard SAE (both 0.352) suggest that the semantic-hierarchy absorption metric is measuring something other than learned structure — possibly the alignment between probe directions and random decoder directions. This should be quantified.

### Experimental Plan

**Task 1: Subspace Alignment Measurement (15 minutes)**
- Compute PCA on the first-letter probe directions (R_orth) and semantic hierarchy probe directions (R_sem) for Pythia-160M layer 8.
- Compute ||P_orth P_sem||_F / sqrt(k_orth * k_sem).
- Compare with the empirical correlation rho = 0.46.

**Task 2: Cross-Layer Subspace Analysis (20 minutes)**
- Repeat Task 1 for layers 0, 4, 8, 12, 16 of Pythia-160M.
- Test Prediction 2: correlation should decrease with layer depth as orthographic and semantic information disentangle.

**Task 3: Random SAE Control Analysis (10 minutes)**
- Compute the projection of semantic hierarchy probes onto random decoder directions.
- Quantify how much of the "absorption" score is due to random alignment vs. learned structure.

**Task 4: GPT-2 Replication (15 minutes)**
- Compute subspace alignment for GPT-2 small.
- Test whether the near-zero absorption scores correlate with low subspace alignment.

### Baselines

**Theoretical baselines**:
- Bhatnagar et al.'s two-feature absorption proof (lower bound on absorption incentive).
- Cui et al.'s feature recovery bounds (upper bound on recoverable features).

**Empirical baselines**:
- The project's pilot results (r = 0.46, H2 reversed).
- SAEBench first-letter absorption scores for the same SAEs.

### Risk Assessment

**Where the proof might fail**:
- The linear subspace assumption may not hold. Orthographic and semantic information could be nonlinearly entangled.
- The bound may be too loose to explain the empirical correlation. The actual correlation could be determined by factors not captured by subspace alignment.

**Where theory-practice gap might be large**:
- Real SAEs have nonlinear activations (ReLU, TopK) that violate the linear projection assumptions.
- The probe directions may not span the full orthographic/semantic subspaces.
- The Random SAE anomaly suggests a data-handling bug rather than a theoretical phenomenon.

### Novelty Claim

This would be the **first information-theoretic explanation for why the SAEBench absorption metric may fail to generalize from first-letter to semantic-hierarchy tasks**. The key novelty is:

1. **Formal decomposition** of absorption into orthographic and semantic subspace components.
2. **Predictive bound** on the correlation between different absorption metrics based on subspace alignment.
3. **Testable prediction** about cross-layer and cross-model variation in construct validity.

No prior work has derived a mathematical relationship between probe-based absorption metrics and the geometric alignment of the underlying feature subspaces. This framework can guide benchmark design by identifying when a proxy metric (first-letter absorption) is likely to be a valid measure of the target construct (semantic-hierarchy absorption).

---

## Connection to Project Context

The project's pilot results show:
- H1 (construct validity): r = 0.46, 95% CI [-0.39, 0.98] — inconclusive, below 0.6 threshold.
- H2 (hierarchy specificity): Reversed — semantic absorption (0.235) < non-hierarchy control (0.331).
- Random SAE = Standard SAE on all hierarchies — suggests data-handling bug.

The theoretical framework in Candidate C explains these anomalies:
1. The moderate correlation (0.46) is consistent with partial orthographic-semantic subspace alignment.
2. The reversed H2 could occur if non-hierarchy correlated features (e.g., "doctor-hospital") have stronger orthographic-semantic alignment than true hierarchies.
3. The Random SAE anomaly undermines all empirical conclusions and must be debugged before any theory can be validated.

**Recommendation**: Debug the Random SAE pipeline first. Then use the subspace alignment framework to predict and explain the correlation pattern across architectures, layers, and models.

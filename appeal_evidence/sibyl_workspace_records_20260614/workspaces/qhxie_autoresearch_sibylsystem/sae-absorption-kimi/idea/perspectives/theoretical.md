# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (arXiv:2409.14507, NeurIPS 2025 Oral)
   - **Key mathematical result**: Formal proof that delta-absorption preserves perfect reconstruction while strictly decreasing sparsity loss. For hierarchical features f1 (parent) and f2 subset f1, the sparsity loss L_sp = p_11(2-delta) + p_10 has derivative dL_sp/ddelta = -p_11 < 0, proving SAE optimization provably drives delta -> 1 (full absorption).
   - **Framework**: Two-feature toy model with explicit encoder/decoder construction showing absorption is incentivized by the sparsity penalty.
   - **Limitation**: Only proves the two-feature case; no generalization to arbitrary hierarchies or explanation of why explicit sparsity control (TopK) outperforms structural constraints.

2. **Tang et al. (2025)** — "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534)
   - **Key mathematical result**: Piecewise biconvex structure of SDL optimization. Theorem 4.7: for any realizable absorption pattern, there exists a local minimum exhibiting exactly this pattern. Escaping requires crossing non-differentiable boundaries.
   - **Framework**: Characterization of global minima (correct recovery) vs. spurious partial minima (absorption, dead neurons, polysemanticity).
   - **Significance**: Explains why absorption is structurally embedded in the optimization landscape but does not predict which architectural choices escape which minima.

3. **Cui et al. (2025)** — "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy" (arXiv:2506.15963)
   - **Key mathematical result**: First closed-form optimal solution for SAEs revealing feature shrinking and feature vanishing. Theorem 1: when n_m >= n and S -> 1, W_m* = (W_p, 0)^T is optimal. Theorem 2: when n_m = n and S -> 1, W_m* = W_p^T is the unique solution.
   - **Critical insight**: "SAE-based interpretability should be regarded as an approximation tool, not as a faithful feature recovery mechanism."
   - **Limitation**: Full recovery only guaranteed under extreme sparsity S -> 1, which is unrealistic for LLM features.

4. **Chen et al. (2025)** — "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders" (arXiv:2506.14002, ICLR 2026)
   - **Key mathematical result**: First SAE algorithm with provable feature recovery guarantees under incoherence assumptions. Group Bias Adaptation (GBA) ensures activation frequency matches feature occurrence frequency.
   - **Framework**: Statistical model x ≈ h^T V where h is sparse and V is dictionary of monosemantic features.
   - **Limitation**: Theory assumes a specific generative model; real-world LLM features may not fit.

5. **Bereska et al. (2025)** — "Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability" (arXiv:2512.13568)
   - **Key mathematical result**: Formal definition of superposition as lossy compression using Shannon entropy on SAE activations to compute "effective features" — the minimum neurons needed for interference-free encoding.
   - **Framework**: Nested rate-distortion problems: (1) neural network layer minimizes I(X;H) subject to task performance, (2) SAE minimizes E[||z||_1] subject to reconstruction distortion.
   - **Significance**: Recasts superposition as an optimal solution to rate-distortion constraints, providing an information-theoretic foundation.

6. **Michaud et al. (2025)** — "Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds" (arXiv:2509.02565)
   - **Key mathematical result**: Capacity-allocation model showing SAEs may over-allocate to common features at expense of rare ones. Two regimes: healthy (alpha < beta) vs. pathological (beta < alpha).
   - **Framework**: Power-law scaling of feature frequencies and per-feature loss reduction.
   - **Significance**: Explains why simply adding latents may not solve absorption — latents may tile common feature manifolds instead of discovering rare parent features.

7. **Chanin & Garriga-Alonso (2025)** — "Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders" (arXiv:2508.16560)
   - **Key mathematical result**: At incorrect L0, SAEs achieve better reconstruction by mixing correlated features (feature hedging) than by learning ground-truth correct latents. Sparsity-reconstruction tradeoff plots are unsound evaluation.
   - **Framework**: Decoder pairwise cosine similarity c_dec minimized at true L0.
   - **Significance**: Shows that L1-based sparsity control is fundamentally flawed — it creates shrinkage that distorts feature learning.

8. **Rajamanoharan et al. (2024)** — "Improving Dictionary Learning with Gated Sparse Autoencoders" (arXiv:2404.16014)
   - **Key mathematical result**: L1 penalty causes systematic activation shrinkage — the optimal compromise between reconstruction and sparsity systematically underestimates true feature magnitudes.
   - **Framework**: Gated SAE decouples feature detection (L1) from magnitude estimation (no L1).
   - **Significance**: Identifies the mechanism by which L1 sparsity creates biased feature representations, motivating TopK as an alternative.

9. **Elhage et al. (2022)** — "Toy Models of Superposition" (Transformer Circuits Thread)
   - **Key mathematical result**: ReLU-linear representation framework where m binary features are represented in d dimensions (d < m) via overlapping non-orthogonal activation patterns. Superposition emerges as optimal when features are sparse and have declining importance.
   - **Framework**: Epsilon-linear vs. ReLU-linear representations; geometric analysis of feature interference.
   - **Significance**: Foundational framework for understanding polysemanticity; all subsequent SAE theory builds on this.

10. **Gao et al. (2024)** — "Scaling and Evaluating Sparse Autoencoders" (arXiv:2406.04093, ICLR 2025)
    - **Key mathematical result**: Empirical scaling laws for SAEs. TopK SAEs achieve better reconstruction-sparsity Pareto frontiers than ReLU SAEs.
    - **Framework**: Large-scale empirical study on GPT-2 and GPT-4.
    - **Significance**: Established TopK as the dominant SAE architecture in practice, though without theoretical explanation of why.

11. **O'Neill et al. (2024)** — "Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders" (arXiv:2411.13117)
    - **Key mathematical result**: Theorem 3.1 (SAE Amortisation Gap): For K-sparse sources in R^N projected into M-dimensional space (M < N) satisfying RIP, SAEs with linear-nonlinear (L-NL) encoders must have a non-zero amortisation gap.
    - **Framework**: Compressed sensing theory applied to SAE encoder limitations.
    - **Significance**: Even with exact reconstruction and correct dictionary, latent recovery is not guaranteed due to architectural constraints.

12. **"Diagnosing and Fixing Latent Recovery in Sparse Autoencoders" (OpenReview 2025, ID: Hl3rEn7S4P)**
    - **Key mathematical result**: Complementary upper and lower bounds on latent recovery error. Upper bound captures error from dictionary coherence and sparsity. Lower bound reveals intrinsic error from unstable encoder-decoder dynamics.
    - **Framework**: Latent self-map F_W(x) = sigma(W W^T x); deviation ||x_t - x_m||_2 serves as certificate of intrinsic recovery error.
    - **Significance**: Identifies dictionary coherence as a fundamental control on SAE stability.

### Theoretical Landscape Summary

**What is known:**
- Absorption is structurally incentivized by sparsity penalties (Chanin et al. 2024): for hierarchical features, any delta in [0,1] preserves reconstruction while sparsity loss strictly decreases with delta.
- The optimization landscape has spurious local minima corresponding exactly to absorption patterns (Tang et al. 2025).
- L1 penalties cause systematic activation shrinkage, distorting feature magnitudes and directions (Rajamanoharan et al. 2024; Chanin & Garriga-Alonso 2025).
- SAEs generally fail to fully recover ground-truth features unless extreme sparsity holds (Cui et al. 2025).
- Rate-distortion and information bottleneck theory provide a principled framework: superposition is an optimal solution to rate-distortion constraints (Bereska et al. 2025).
- Capacity-allocation models predict over-allocation to common features at the expense of rare parent features (Michaud et al. 2025).
- Linear-nonlinear encoders have a fundamental amortisation gap (O'Neill et al. 2024).

**What is conjectured:**
- Explicit sparsity control (TopK) avoids the shrinkage bias of L1 penalties, enabling more accurate feature magnitude estimation.
- The absorption-sparsity correlation observed in our experiments (r ≈ +0.93) reflects a fundamental rate-distortion tradeoff: more active latents (higher rate) create more opportunities for parent-child competition.
- Hard thresholding (TopK) may provide theoretical guarantees that soft thresholding (L1) cannot, analogous to compressed sensing results for greedy methods vs. convex relaxation.
- Orthogonality penalties improve decoder geometry but do not change the activation sparsity pattern that drives absorption.

**Where the gaps are:**
- No theoretical framework predicting absorption rate as a function of sparsity level (L0) for a given feature hierarchy.
- No proof that TopK sparsity avoids absorption better than L1 sparsity under identical conditions.
- No rate-distortion bound characterizing the absorption-sparsity correlation observed empirically.
- No theoretical explanation for why orthogonality penalties improve reconstruction but not absorption.
- No information-theoretic characterization of why structural constraints (multi-scale, orthogonality) underperform explicit sparsity control.

---

## Phase 2: Initial Candidates

### Candidate A: The Sparsity-Control Theory — Why TopK Dominates Absorption Reduction

**Formal claim**: Consider an SAE with latent dimension m, trained on hierarchical features organized in a tree T with parent-child pairs H. Let L0 be the expected number of active latents per sample. Define the absorption rate A(L0) as the fraction of parent features absorbed by their children.

**Theorem (Sparsity-Control Bound)**: For any SAE with soft sparsity control (L1 penalty) achieving expected L0 = s, and any SAE with hard sparsity control (TopK with k = s) achieving exact L0 = s, the TopK SAE achieves lower absorption:

E[A_TopK(s)] <= E[A_L1(s)] - delta(s, H)

where delta(s, H) > 0 is a task-dependent margin that increases with hierarchy depth and decreases with sparsity level s. Furthermore, for fixed hierarchy H, absorption is monotonically increasing in L0:

dA/d(L0) > 0

**Proof sketch**:
1. Lemma 1 (L1 shrinkage bias): The L1 penalty creates a shrinkage bias E[z_L1] = z_true - lambda * sign(z_true) / ||W_dec||^2, causing systematic underestimation of feature magnitudes. This bias is larger for parent features (which have lower activation frequency) than child features.
2. Lemma 2 (TopK exactness): TopK hard thresholding preserves the relative magnitude ordering of features without shrinkage bias. The k-largest features are selected exactly, not penalized proportionally.
3. Lemma 3 (Parent-child competition): When L0 latents co-activate, the probability that a parent and child compete for the same latent slot is proportional to L0/m. With L1 (s ≈ 964), competition is high; with TopK (s = 50), competition is low.
4. Main theorem: Combining Lemmas 1-3, TopK reduces absorption through two mechanisms: (a) eliminating shrinkage bias that disproportionately suppresses parent features, and (b) reducing the pool of co-active latents that compete for parent-child representation.

**Empirical prediction**:
- A sparsity-sweep experiment (varying k in TopK SAE) should show monotonically increasing absorption with k.
- An L1-sparsity-sweep (varying lambda) should show the same trend but with higher absorption at matched L0 levels.
- The gap between TopK and L1 at matched L0 should be largest at intermediate sparsity levels (k ≈ 100-500).

**Connection to existing theory**: Connects to Rajamanoharan et al.'s shrinkage analysis (L1 bias), Chanin et al.'s absorption proof (sparsity incentive), and compressed sensing theory (hard thresholding vs. L1 minimization).

**Novelty estimate**: 8/10. The sparsity-control theory provides a mechanistic explanation for why TopK dominates, grounded in the well-known shrinkage properties of L1 penalties. The monotonic absorption-sparsity relationship is a new testable prediction.

---

### Candidate B: The Rate-Distortion Absorption Function — An Information-Theoretic Characterization

**Formal claim**: Consider a tree-structured feature hierarchy T with depth D and branching factor b. Let X be the residual-stream activation, Z the SAE latent, and X_hat = D Z the reconstruction. Define the absorption rate A(T) as the fraction of parent features not independently represented in Z.

**Theorem (Rate-Distortion Absorption Bound)**: For any SAE with rate R = E[||Z||_0] = L0 (in nats per token) and distortion D = E[||X - X_hat||^2], the absorption rate is lower-bounded by:

A(T) >= max(0, 1 - R / H(T))

where H(T) is the entropy of the feature hierarchy (in nats). For a b-ary tree of depth D with uniform leaf probabilities:

H(T) = log_2((b^D - 1)/(b - 1))

**Corollary (Sparsity-Absorption Tradeoff)**: For fixed hierarchy T, absorption decreases with rate R. For fixed rate R, absorption increases with hierarchy complexity H(T). There exists a critical rate R_crit = H(T) below which absorption is inevitable (A(T) > 0).

**Proof sketch**:
1. Lemma 1 (Feature counting): Each active latent can represent at most one independent feature direction without interference. With L0 active latents per token, the SAE can represent at most L0 independent feature directions per token.
2. Lemma 2 (Hierarchy entropy): A feature hierarchy T requires H(T) bits to specify which node is active. For independent representation of all nodes, we need R >= H(T).
3. Lemma 3 (Rate-distortion connection): By the rate-distortion theorem, achieving zero distortion requires R >= H(X). For hierarchical features, H(X) >= H(T).
4. Main theorem: When R < H(T), some nodes must be absorbed (represented implicitly through other nodes). The bound is tight when the SAE achieves the rate-distortion limit.

**Empirical prediction**:
- For SynthSAEBench hierarchies (128 trees, depth 3, branching factor 4): H(T) ≈ log_2(21) ≈ 4.4 bits per tree. With L0 = 50 and 128 trees, R = 50 >> 4.4 * 128 = 563 bits... wait, this doesn't work. The bound needs refinement.

**Critical issue**: The naive rate-distortion bound is vacuous for practical parameters. With m = 2048 latents and L0 = 50, the SAE has ample rate budget. The bound only becomes informative for extremely large hierarchies or extremely narrow SAEs.

**Revised claim**: The rate-distortion framework applies not to the total hierarchy but to the *local* competition at each co-activation event. When parent and child co-occur, the SAE must allocate its L0 active slots between them. The absorption probability is the probability that the parent loses this competition.

**Novelty estimate**: 5/10. The rate-distortion connection is natural but the naive bound is vacuous. A local-competition refinement might be novel but requires significant theoretical development.

---

### Candidate C: The Activation-Pattern Competition Theory — Why Structural Constraints Fail

**Formal claim**: Let P_cooc(p, c) be the co-occurrence probability of parent p and child c in the training data. Let S be the set of active latents when both p and c are present. Define the parent win probability as:

P_parent_wins = P(p in S | p, c both active)

**Theorem (Structural Constraint Ineffectiveness)**: For an SAE with decoder orthogonality penalty lambda_ortho, the parent win probability satisfies:

P_parent_wins(ortho) = P_parent_wins(baseline) + O(lambda_ortho * ||W_dec^T W_dec - I||_F^2)

For typical lambda_ortho = 10^{-3}, the improvement is negligible (second-order in a small penalty). In contrast, for TopK with k << m:

P_parent_wins(TopK) >= 1 - exp(-k * P(p activates independently) / m)

which approaches 1 as k decreases.

**Proof sketch**:
1. Lemma 1 (Orthogonality effect on activation): The orthogonality penalty affects decoder geometry but not the encoder's activation pattern. The encoder still selects latents based on pre-activation scores, which are unaffected by decoder orthogonality.
2. Lemma 2 (TopK effect on activation): TopK changes the activation pattern directly by limiting the number of active latents. With fewer active latents, each latent represents a larger fraction of the input, reducing the chance that parent features are crowded out.
3. Lemma 3 (Competition geometry): When k latents compete for m slots, the probability that a specific feature (parent) is selected depends on its relative activation strength. TopK makes the competition explicit and deterministic; L1 makes it implicit and biased.
4. Main theorem: Orthogonality is a second-order effect on decoder geometry; TopK is a first-order effect on activation patterns. This explains the observed effect size gap (d = 0.14 vs. d = 5.51).

**Empirical prediction**:
- Measuring parent-feature activation rates in the latent space should show: Baseline (low rate), TopK (higher rate), Orthogonality (same as Baseline).
- The orthogonality penalty should improve decoder cosine similarity but not change the distribution of latent activations for parent features.

**Connection to existing theory**: Connects to Korznikov et al.'s orthogonality analysis (decoder geometry) and Chanin et al.'s absorption proof (activation competition).

**Novelty estimate**: 7/10. The first-order vs. second-order distinction provides a clean explanation for the observed effect size hierarchy. The activation-pattern focus shifts attention from decoder geometry to encoder behavior.

---

## Phase 3: Self-Critique

### Against Candidate A (Sparsity-Control Theory)

**Proof soundness attack**: The shrinkage bias formula E[z_L1] = z_true - lambda * sign(z_true) / ||W_dec||^2 is an approximation that assumes fixed decoder. In practice, the decoder co-adapts with the encoder, potentially compensating for shrinkage. The true bias may be smaller than the lemma suggests.

**Tightness attack**: The theory predicts monotonically increasing absorption with L0, but our data shows Baseline (L0=964, A=0.252), Orthogonality (L0=550, A=0.245), TopK (L0=50, A=0.056). The Baseline-to-Orthogonality drop (L0 from 964 to 550, A from 0.252 to 0.245) is small, suggesting the relationship is non-linear or confounded by other factors.

**Relevance attack**: This directly addresses the paper's central finding (TopK dominates) and makes testable predictions. The shrinkage mechanism is well-established in the Lasso literature and directly applicable.

**Novelty attack**: The connection between L1 shrinkage and SAE absorption is novel, though the shrinkage phenomenon itself is classical (Tibshirani 1996). The application to explain TopK's dominance is new.

**Verdict**: STRONG. The sparsity-control theory provides the most direct explanation for the observed effect size hierarchy.

---

### Against Candidate B (Rate-Distortion Absorption Function)

**Proof soundness attack**: The naive bound is vacuous for all practical parameter regimes (R >> H(T) for tested hierarchies). The "local competition" refinement abandons the rate-distortion framework and becomes an activation-pattern argument, losing the information-theoretic elegance.

**Tightness attack**: Even with the local refinement, the bound requires knowing the exact co-occurrence structure and activation competition dynamics, which are model-dependent and hard to characterize.

**Relevance attack**: The rate-distortion framework is elegant but does not explain the specific architectural differences observed (TopK vs. L1 vs. Orthogonality). It predicts absorption as a function of rate, not as a function of architectural choice at fixed rate.

**Novelty attack**: Rate-distortion bounds for hierarchical features are a natural extension of Bereska et al.'s framework but the specific bound is not tight enough to be useful.

**Verdict**: WEAK. The bound is vacuous for practical parameters and does not explain the observed architectural differences.

---

### Against Candidate C (Activation-Pattern Competition Theory)

**Proof soundness attack**: The claim that orthogonality is "second-order" assumes the penalty is small. If lambda_ortho were much larger, the effect could be first-order. The theorem should specify the regime where the approximation holds.

**Tightness attack**: The TopK win probability formula 1 - exp(-k * P(p activates) / m) is heuristic, not derived from first principles. It assumes independent activation probabilities, which may not hold.

**Relevance attack**: This directly explains why orthogonality fails (decoder geometry != activation pattern) and why TopK succeeds (direct activation control). It aligns with the empirical observation that orthogonality improves reconstruction (decoder quality) but not absorption (activation pattern).

**Novelty attack**: The first-order vs. second-order distinction is a standard perturbation argument, but its application to explain the TopK-orthogonality gap is novel.

**Verdict**: MODERATE. The theory is relevant and makes testable predictions, but the formal claims need tightening.

---

## Phase 4: Refinement

**Dropped**: Candidate B (Rate-Distortion Bound) — the bound is vacuous for all practical parameter regimes and provides no explanatory power for the observed architectural differences.

**Strengthened**: Candidate A (Sparsity-Control Theory) — refined with explicit acknowledgment of decoder co-adaptation. The core claim is not that shrinkage is uncompensated, but that L1 creates a *systematic bias* in the encoder's activation pattern that favors child features over parent features. TopK eliminates this bias by enforcing exact sparsity without magnitude distortion.

**Strengthened**: Candidate C (Activation-Pattern Competition) — merged with Candidate A as a complementary perspective. The combined theory: (1) L1 creates shrinkage bias that disproportionately suppresses parent features (Candidate A), AND (2) high L0 creates dense co-activation that increases parent-child competition (Candidate C). TopK addresses both: (1) no shrinkage bias, (2) limited co-activation slots.

**Selected front-runner**: Combined A+C: "The Sparsity-Control and Activation-Competition Theory."

---

## Phase 5: Final Proposal

### Title
**Why TopK Dominates: A Sparsity-Control Theory of Feature Absorption in Sparse Autoencoders**

### Formal Claim

**Theorem 1 (L1 Shrinkage Bias in Hierarchical Features)**: Let z_p and z_c be the latent activations for parent and child features under an L1-sparsity SAE with penalty lambda. Let f_p and f_c be the true feature magnitudes, with f_p < f_c (parent features are typically less active). The L1 shrinkage bias is:

E[z_L1] = argmin_z {||x - W_dec z||^2 + lambda ||z||_1}

For orthogonal decoder directions and independent features, the shrinkage is:

z_L1,i = sign(z_i) * max(0, |z_i| - lambda / ||W_dec,i||^2)

**Lemma 1.1 (Parent-Child Asymmetry)**: Since parent features have lower baseline activation (f_p < f_c), the relative shrinkage |z_p - z_L1,p| / |z_p| > |z_c - z_L1,c| / |z_c|. Parents are disproportionately suppressed.

**Theorem 2 (TopK Exactness)**: For TopK with threshold k, the activation is:

z_TopK,i = z_i if |z_i| in top-k(|z|), else 0

TopK preserves the relative magnitude ordering without shrinkage: if |z_p| > |z_c| in pre-activation, then |z_TopK,p| > |z_TopK,c| in post-activation. The selection is based on ranking, not penalized magnitude.

**Theorem 3 (Absorption-Sparsity Monotonicity)**: For a fixed feature hierarchy H and SAE architecture, the expected absorption rate E[A] is monotonically increasing in the expected number of active latents L0:

dE[A]/d(L0) > 0

**Proof**: With more active latents, the probability that parent and child features compete for representation slots increases. When L0 = m (all latents active), absorption is maximized because all features co-activate and the SAE cannot distinguish parent from child. When L0 = 1 (single latent), absorption is minimized because only the dominant feature (typically child) activates, but reconstruction fails.

**Corollary (TopK Dominance)**: TopK achieves lower absorption than L1 at matched L0 because:
1. It eliminates shrinkage bias (Theorem 2 vs. Theorem 1)
2. It enforces exact L0 = k, avoiding the L1 "leakage" where L0 varies across samples

### Proof Sketch

**Theorem 1**: Standard Lasso shrinkage result (Tibshirani 1996) applied to SAE encoder output. The key insight is that the shrinkage threshold lambda / ||W_dec,i||^2 is constant across features, so features with smaller true magnitudes (parents) experience larger relative shrinkage.

**Theorem 2**: TopK is a selection operator, not a shrinkage operator. It preserves the rank order of pre-activations. The proof is immediate from the definition of the top-k operator.

**Theorem 3**: Consider the co-activation of parent p and child c. The SAE must allocate its L0 active slots between features present in the input. With more slots, the SAE can represent both p and c independently. However, the L1 penalty creates a "budget" that the SAE optimizes: representing both p and c costs more in L1 terms than representing only c (which partially reconstructs p through absorption). As L0 increases, the L1 budget increases, but the SAE still faces the incentive to absorb (Chanin et al.'s proof). The monotonicity follows because higher L0 means more features co-activate, creating more absorption opportunities.

### Assumptions

1. **Independent decoder directions**: Theorem 1 assumes orthogonal decoder columns. In practice, decoder directions are correlated, but the qualitative result (parents shrink more than children) holds as long as parent features have lower baseline activation.
2. **Parent magnitude ordering**: We assume f_p < f_c (parents are less active than children). This holds for hierarchical features where children are more specific and thus less frequent.
3. **Fixed hierarchy structure**: The theorems assume a known tree-structured hierarchy. Real LLM features may have more complex structure.
4. **Matched L0 comparison**: The TopK vs. L1 comparison assumes matched expected L0. In practice, L1 SAEs achieve variable L0 across samples while TopK achieves exact L0 = k.

### Empirical Prediction

**Prediction 1 (Sparsity-sweep absorption curve)**: Varying k in {10, 25, 50, 100, 200, 500} for TopK SAEs on SynthSAEBench should yield a monotonically increasing absorption curve. This directly tests Theorem 3.

**Prediction 2 (L1 vs. TopK at matched L0)**: Training L1 SAEs with varying lambda to achieve L0 in {50, 100, 200, 500, 964} and comparing with TopK at matched k should show TopK < L1 for all matched pairs. This tests the shrinkage bias hypothesis.

**Prediction 3 (Parent activation rate)**: Measuring parent-feature activation rates in the latent space should show: Baseline (low, due to shrinkage), TopK (higher, no shrinkage), Orthogonality (same as Baseline, since orthogonality doesn't affect encoder). This tests the activation-pattern mechanism.

**Prediction 4 (Decoder cosine similarity)**: The orthogonality penalty should reduce mean decoder cosine similarity (it does, by construction), but this reduction should not correlate with absorption reduction. This isolates decoder geometry from activation pattern.

### Experimental Plan

**Task 1: Sparsity-Sweep Experiment (30 minutes)**
- Train TopK SAEs with k in {10, 25, 50, 100, 200, 500} on SynthSAEBench-16k
- Measure absorption rate, MCC, MSE, L0 for each
- Test Prediction 1: plot absorption vs. k
- **Falsification criterion**: If absorption is not monotonically increasing in k, Theorem 3 is rejected.

**Task 2: L1-vs-TopK Matched Comparison (30 minutes)**
- Train L1 SAEs with lambda in {0.001, 0.003, 0.005, 0.01, 0.05} to achieve L0 matching TopK values
- Compare absorption at matched L0
- Test Prediction 2: TopK < L1 at all matched pairs
- **Falsification criterion**: If L1 achieves lower absorption than TopK at any matched L0, the shrinkage bias hypothesis is rejected.

**Task 3: Activation Pattern Analysis (15 minutes)**
- For each trained SAE, measure parent-feature activation rates in latent space
- Compare Baseline, TopK, Orthogonality
- Test Prediction 3: TopK > Baseline in parent activation rate; Orthogonality ≈ Baseline
- **Falsification criterion**: If Orthogonality shows increased parent activation, the first-order/second-order distinction is rejected.

**Task 4: Decoder Geometry Analysis (10 minutes)**
- Compute mean decoder cosine similarity for all variants
- Correlate with absorption rate
- Test Prediction 4: no correlation between decoder orthogonality and absorption
- **Falsification criterion**: If decoder cosine similarity correlates negatively with absorption (r < -0.5), the activation-pattern theory is challenged.

### Baselines

**Theoretical baselines**:
- Chanin et al.'s two-feature absorption proof (lower bound on absorption incentive).
- Tibshirani's Lasso shrinkage formula (shrinkage = lambda / ||W||^2).
- Compressed sensing: hard thresholding (OMP, IHT) vs. L1 minimization guarantees.

**Empirical baselines**:
- Current paper results: TopK (A=0.056, L0=50), Baseline (A=0.252, L0=964), Orthogonality (A=0.245, L0=550).
- Gao et al.'s TopK scaling laws (ICLR 2025).
- Rajamanoharan et al.'s gated SAE results (shrinkage reduction).

### Risk Assessment

**Where the proof might fail**:
- The shrinkage formula assumes orthogonal decoder directions. Real SAE decoders are correlated, and the effective shrinkage may differ.
- The monotonicity theorem assumes the L1 incentive to absorb (Chanin et al.'s proof) dominates at all L0 levels. At very low L0, reconstruction failure may override the absorption incentive.
- The TopK exactness theorem ignores the non-differentiability of TopK, which complicates training dynamics.

**Where theory-practice gap might be large**:
- Real LLM features may not follow the parent-magnitude < child-magnitude ordering assumed in Lemma 1.1.
- The shrinkage bias may be partially compensated by decoder co-adaptation during training.
- The sparsity-sweep may reveal non-monotonic behavior due to training instability at extreme k values.

### Novelty Claim

This would be the **first theoretical framework explaining why explicit sparsity control (TopK) dominates structural constraints (multi-scale, orthogonality) for absorption reduction**. The key novelty is:

1. **Mechanistic explanation**: TopK reduces absorption by eliminating L1 shrinkage bias and limiting parent-child competition, not by any architectural magic.
2. **Monotonicity prediction**: The absorption-sparsity relationship (dA/dL0 > 0) is a new testable prediction with practical implications for SAE design.
3. **First-order vs. second-order distinction**: The framework explains why orthogonality (second-order decoder effect) underperforms TopK (first-order activation effect) by an order of magnitude.
4. **Unified perspective**: The theory synthesizes classical Lasso shrinkage, Chanin et al.'s absorption proof, and compressed sensing theory into a coherent explanation of the observed effect size hierarchy.

No prior work has derived a mathematical relationship between sparsity control mechanism (L1 vs. TopK) and absorption rate. This framework can guide architecture design by identifying sparsity level — not architectural novelty — as the operative variable.

---

## Connection to Project Context

The project's actual experimental results (iter_006) show:

- **TopK dominates**: 78.0% absorption reduction (Cohen's d = 5.51) vs. Orthogonality 2.7% (d = 0.14).
- **Sparsity-absorption correlation**: r ≈ +0.93 across n=4 variants (Baseline L0=964/A=0.252, TopK L0=50/A=0.056, Orthogonality L0=550/A=0.245, MultiScale L0=50/A=0.050).
- **Orthogonality achieves near-perfect reconstruction** (MSE ≈ 3×10^{-5}) but negligible absorption reduction.
- **Hedging is invariant** (~0.24) across all variants, suggesting synthetic data does not trigger the absorption-hedging tradeoff.

The Sparsity-Control Theory explains these findings:

1. **TopK's dominance**: TopK eliminates L1 shrinkage bias and enforces extreme sparsity (L0=50), both of which reduce absorption. The effect is first-order on the activation pattern.

2. **Orthogonality's failure**: The orthogonality penalty is a second-order effect on decoder geometry. It improves reconstruction (by making decoder directions more independent) but does not change the encoder's activation pattern, which is what drives absorption.

3. **Sparsity-absorption correlation**: The positive correlation reflects the fundamental competition mechanism: more active latents create more opportunities for parent-child competition. TopK and MultiScale both achieve L0=50 and both show low absorption (~0.05).

4. **MCC invariance**: Hungarian matching on overcomplete dictionaries yields ~0.21 by chance. The theory does not predict MCC differences because MCC measures feature recovery quality, not absorption patterns.

**Recommendation**: Before running new experiments:
1. Run the sparsity-sweep experiment (Task 1) to test the monotonicity prediction.
2. Run the L1-vs-TopK matched comparison (Task 2) to isolate the shrinkage bias effect.
3. Measure parent activation rates (Task 3) to verify the activation-pattern mechanism.

If the theory is correct, the community should:
- Treat sparsity level (L0) as the primary design variable for absorption control.
- View TopK not as an architectural innovation but as a sparsity-control mechanism.
- Re-evaluate orthogonality penalties: they improve reconstruction but not absorption.
- Design SAEs with explicit sparsity budgets rather than implicit L1 penalties.

---

## Sources

- [Chanin et al., 2024] "A is for Absorption" (arXiv:2409.14507)
- [Tang et al., 2025] "A Unified Theory of Sparse Dictionary Learning" (arXiv:2512.05534)
- [Cui et al., 2025] "On the Limits of Sparse Autoencoders" (arXiv:2506.15963)
- [Chen et al., 2025] "Taming Polysemanticity in LLMs" (arXiv:2506.14002)
- [Bereska et al., 2025] "Superposition as Lossy Compression" (arXiv:2512.13568)
- [Michaud et al., 2025] "Understanding SAE Scaling" (arXiv:2509.02565)
- [Chanin & Garriga-Alonso, 2025] "Sparse but Wrong" (arXiv:2508.16560)
- [Rajamanoharan et al., 2024] "Gated Sparse Autoencoders" (arXiv:2404.16014)
- [Elhage et al., 2022] "Toy Models of Superposition" (Transformer Circuits Thread)
- [Gao et al., 2024] "Scaling and Evaluating Sparse Autoencoders" (arXiv:2406.04093)
- [O'Neill et al., 2024] "Compute Optimal Inference" (arXiv:2411.13117)
- [OpenReview 2025] "Diagnosing and Fixing Latent Recovery" (ID: Hl3rEn7S4P)
- [Tibshirani, 1996] "Regression Shrinkage and Selection via the Lasso" (J. R. Stat. Soc.)
- [Donoho & Elad, 2003] "Optimally sparse representation" (PNAS)
- [Donoho, 2006] "Compressed sensing" (IEEE Trans. Info. Theory)

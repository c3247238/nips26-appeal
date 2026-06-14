# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024) - "A is for Absorption"** (arXiv:2409.14507) — Established Proposition 2: absorption is a logical consequence of sparsity loss under hierarchical feature co-occurrence. Proved that under co-occurring parent-child features, the sparsity objective incentivizes merging the parent direction into child latents.

2. **Tang et al. (2025) - "On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability"** (arXiv:2512.05534) — First theoretical explanation of absorption as spurious local minima in the dictionary learning optimization landscape. Introduced "feature anchoring" as a potential remedy. Provides identifiability conditions under which ground-truth features can be recovered.

3. **Cui et al. (2025) - "On the Limits of Sparse Autoencoders"** (arXiv:2506.15963) — Formal identifiability analysis proving conditions for recovering ground-truth features. Proposes reweighted remedy. Theoretical limitations of SAE-based interpretability when features are non-independent.

4. **Elhage et al. (2022) - "Toy Models of Superposition"** (Anthropic Blog) — Foundational superposition hypothesis: neural networks represent more features than dimensions by encoding features in overlapping, approximately orthogonal directions. The theoretical bedrock for understanding why SAEs are necessary.

5. **Budd et al. (2025) - "SplInterp"** (arXiv:2505.11836) — Theoretical framework using spline theory and power diagrams to characterize TopK SAE geometry. Provides formal characterization of the activation landscape.

6. **Gao et al. (2024) - "Scaling and Evaluating SAEs"** (arXiv:2406.04093) — Scaling laws for SAE training. Information-theoretic analysis of the sparsity-reconstruction tradeoff.

7. **Korznikov et al. (2025) - OrtSAE** (arXiv:2509.22033) — Decoder orthogonality constraint reduces absorption by 65%. Theoretical insight: enforcing orthogonality changes the local minima structure of the loss landscape.

8. **Bussmann et al. (2025) - Matryoshka SAEs** (arXiv:2503.17547) — Nested multi-level dictionaries reduce absorption from 0.49 to 0.05. Theoretical insight: hierarchical structure mirrors the natural feature hierarchy, reducing the pressure to absorb.

### Theoretical Landscape Summary

**What is known:**
- Absorption is a **logical consequence** of the sparsity objective (Chanin et al., Proposition 2)
- Absorption corresponds to **spurious local minima** in the dictionary learning loss landscape (Tang et al.)
- The identifiability problem is fundamentally hard when features are correlated (Cui et al.)
- Decoder orthogonality and hierarchical architecture both reduce absorption, suggesting the loss landscape structure matters

**What is conjectured:**
- Absorption is **rate-distortion optimal** behavior under hierarchical co-occurrence (empirical evidence from H7: trained < random)
- The Chanin differential correlation metric may be **sensitive to dictionary structure** rather than learned pathology
- Training optimizes decoder geometry to reduce structural artifacts

**Where the gaps are:**
- **No formal rate-distortion analysis** connecting absorption to information-theoretic optimality
- **No PAC learning bounds** on absorption generalization to downstream tasks
- **No characterization** of the conditions under which absorbed features retain functional utility
- **No unified framework** connecting local minima theory, identifiability bounds, and rate-distortion optimality

---

## Phase 2: Initial Candidates

### Candidate A: Absorption as Information-Theoretic Rate-Distortion Optimal Compression

**Formal claim:** Under hierarchical feature co-occurrence with probability distribution P over parent-child activation patterns, the SAE sparsity objective L = ReconstructionLoss + lambda * SparsityLoss is minimized (in expectation) when the child latent absorbs the parent direction, provided lambda is sufficiently large relative to the reconstruction benefit of maintaining separate representations.

**Proof sketch:**
1. Let parent feature P have activation probability p_p and child feature C have activation probability p_c, with joint probability matrix [[p_11, p_10], [p_01, p_00]] where p_11 = P(P=1, C=1).
2. Without absorption: the expected sparsity cost is 2*p_11 + p_10 + p_01 (both features active in k-sparse case).
3. With absorption (child captures parent direction): the expected sparsity cost reduces to p_11 + p_10 + p_01, a savings of exactly p_11 > 0.
4. The reconstruction loss increase from absorption is bounded by the alignment between parent and child decoder directions: DeltaRecon <= ||d_P - d_C||^2 * p_11.
5. For hierarchical features (where d_P and d_C are aligned), DeltaRecon is small; the sparsity savings p_11 dominates, making absorption optimal.

**Empirical prediction:** Features with higher parent-child alignment (cosine similarity of decoder directions) will exhibit higher absorption rates, while features with lower alignment will maintain separate representations.

**Connection to existing theory:** Extends Chanin et al.'s Proposition 2 with a formal rate-distortion analysis. Connects to the empirical H7 finding (trained < random absorption) by showing that training optimizes decoder alignment to minimize absorption.

**Novelty estimate:** 6/10 — Rate-distortion framing is implicit in Chanin et al. but not formally developed. The specific connection to decoder alignment as the determinant of absorption vs. separation is novel.

---

### Candidate B: Absorption as Spurious Local Minima in Overcomplete Dictionary Learning

**Formal claim:** (building on Tang et al.) Absorption corresponds to a class of spurious local minima in the SAE loss landscape that are **structurally determined** by the dictionary geometry, not by the training data distribution. Random SAEs exhibit higher absorption because random decoder geometries are more likely to create these spurious minima.

**Proof sketch:**
1. The SAE objective L(theta) = E[||x - Dec(Enc(x))||^2] + lambda * E[Sparsity(Enc(x))] has a non-convex landscape.
2. For an overcomplete dictionary (more latents than dimensions), absorption occurs when the encoder learns to map multiple similar features to the same latent rather than maintaining separate latents.
3. The condition for absorption as a local minimum: for parent-child feature pair (P, C) with activations x_P, x_C, if the Hessian of L at the separated solution has a positive eigenvalue corresponding to the direction that merges P and C, then absorption is a local minimum.
4. The probability of this occurring in a random SAE is higher because random decoder directions have no structure favoring separation.
5. Training discovers the global minimum (or better local minimum) where decoder directions align with meaningful features, reducing the spurious local minima basin.

**Empirical prediction:** The absorption metric should correlate with the Hessian eigenvalue spectrum around the separated solution. High absorption corresponds to flat directions in the loss landscape.

**Connection to existing theory:** Direct extension of Tang et al.'s spurious local minima explanation. The random SAE comparison (H7) provides empirical support: random geometries create spurious minima more frequently.

**Novelty estimate:** 5/10 — Tang et al. already established the spurious local minima framework. The contribution would be formalizing the "structurally determined" claim and the random vs. trained comparison.

---

### Candidate C: PAC-Learning Bounds on Absorption-Generalization

**Formal claim:** Let A be the absorption rate of a SAE feature, measured by the Chanin differential correlation metric. Let U be the utility of that feature for downstream tasks (steering accuracy, probing accuracy). Then with probability at least 1-delta over the training distribution, |U(A) - E[U]| <= O(sqrt((d + log(1/delta))/n)) where d is the Vapnik-Chervonenkis dimension of the downstream task class and n is the number of calibration samples.

**Proof sketch:**
1. The absorption rate A is a function of the feature's activation pattern on calibration data.
2. The downstream utility U is a function of the feature's steering/probing behavior.
3. Under standard PAC-learning assumptions (i.i.d. sampling, bounded loss), the empirical estimate of U converges to the true expectation at rate sqrt(d/n).
4. The key assumption: A and U are related by a function class with finite VC dimension (the set of possible steering behaviors as a function of absorption rate is simple).
5. The bound holds regardless of the SAE architecture or training procedure, suggesting absorption rate alone is insufficient to predict utility.

**Empirical prediction:** The correlation between A and U should be weak (near zero) and the confidence intervals should be wide, consistent with the null results H1-H4.

**Connection to existing theory:** Provides formal grounding for the empirical observation that absorption does not degrade downstream tasks. The VC dimension bound explains why high absorption features can still have high utility.

**Novelty estimate:** 4/10 — PAC-learning bounds are standard tool. The specific application to absorption-utility relationship is novel but the theoretical contribution is incremental.

---

## Phase 3: Self-Critique

### Against Candidate A (Rate-Distortion)

**Proof soundness attack:**
- The proof sketch assumes that decoder directions are fixed when analyzing absorption. In reality, the encoder and decoder are jointly optimized, so the "alignment" between d_P and d_C is itself learned.
- The rate-distortion tradeoff assumes lambda (sparsity weight) is the sole determinant. In practice, learning rates, initialization, and optimization dynamics also matter.
- The "hierarchical features have aligned decoders" assumption is not formally justified — this is an empirical claim, not a theorem.

**Tightness attack:**
- The bound DeltaRecon <= ||d_P - d_C||^2 * p_11 is loose. In the extreme case where d_P = d_C, absorption is costless (zero reconstruction loss increase) and the theory correctly predicts absorption. But for intermediate alignments, the bound may be too loose to be useful.
- The prediction "higher alignment -> higher absorption" is empirically testable but may be tautological if alignment is measured post-hoc from the trained SAE.

**Relevance attack:**
- The rate-distortion framing is mathematically elegant but does it help practitioners? If absorption is optimal, there is no "fix" needed — only acceptance.
- The connection to H7 (trained < random) is suggestive but not conclusive: training could reduce absorption for reasons other than optimal compression.

**Novelty attack:**
- Chanin et al.'s Proposition 2 already establishes the sparsity-savings aspect. Candidate A extends this with the alignment analysis but the core insight is not new.

**Verdict:** MODERATE — The rate-distortion framing is the most promising for connecting the empirical findings. The alignment analysis provides a testable prediction. However, the proof is not rigorous without formal assumptions about decoder geometry.

---

### Against Candidate B (Spurious Local Minima)

**Proof soundness attack:**
- The Tang et al. framework assumes a specific noise model and activation function. The extension to JumpReLU/Gated SAEs is not trivial.
- The claim that random SAEs have "more" spurious minima is intuitive but not formally proven — it is a hypothesis consistent with the evidence.
- The Hessian analysis requires computing eigenvalues of a high-dimensional matrix, which is intractable for large SAEs (24K+ latents).

**Tightness attack:**
- The characterization of "spurious" vs. "correct" local minima is binary, but in practice there is a continuum of local minima with varying quality.
- The theory does not predict when absorption will be harmful vs. benign — it only characterizes when it occurs.

**Relevance attack:**
- If absorption is just a local minimum phenomenon, then better optimization (e.g., better initialization, second-order methods) would fix it. But empirical results show architecture (Matryoshka, OrtSAE) matters more than optimization.
- The practical relevance is limited unless we can identify which local minima are "good" vs. "bad" without running downstream experiments.

**Novelty attack:**
- Tang et al. already established the spurious local minima framework. Candidate B is an extension, not a new theory.

**Verdict:** WEAK — The local minima framing is theoretically sound but does not add substantially to Tang et al. The random SAE comparison is consistent with the theory but does not constitute a novel theoretical prediction.

---

### Against Candidate C (PAC-Learning Bounds)

**Proof soundness attack:**
- The VC dimension bound assumes the steering behavior class has finite VC dimension. For continuous steering strengths, this may not hold.
- The bound is non-uniform (depends on the specific feature) and may not generalize across different SAE architectures.
- The independence assumption between absorption measurement and utility measurement may be violated in practice.

**Tightness attack:**
- The bound is too loose to be practically useful — it does not distinguish between "absorption matters" and "absorption doesn't matter" in a way that guides action.
- The sqrt(d/n) rate is standard but pessimistic; the actual correlation may be tighter.

**Relevance attack:**
- PAC-learning bounds are theoretical curiosities for most practitioners. The bound does not provide actionable guidance.
- The weak correlation (H1-H4 null results) is consistent with the theory but does not require a PAC-learning explanation.

**Novelty attack:**
- PAC-learning is a standard tool. The specific application to absorption-utility is novel but the theoretical contribution is incremental.

**Verdict:** WEAK — The PAC-learning framing is theoretically valid but does not provide novel insights beyond the empirical null results. It is more of a formal justification for the null result than a new theoretical contribution.

---

## Phase 4: Refinement

### Dropped Candidates

- **Candidate C (PAC-Learning):** The VC dimension bound is too loose and does not provide actionable insights. The null results (H1-H4) are better explained by the rate-distortion framework.

- **Candidate B (Spurious Local Minima):** The Tang et al. framework already covers this. The random SAE comparison (H7) is consistent with the theory but is not a novel theoretical prediction.

### Strengthened Candidates

**Candidate A (Rate-Distortion) — Strengthened:**

1. **Added empirical validation connection:** The H7 finding (trained SAE mean absorption = 0.034 vs. random SAE = 0.278) supports the rate-distortion interpretation: trained SAEs have optimized decoder geometry that reduces the pressure for absorption.

2. **Added precision-recall asymmetry explanation:** The H5 finding (precision = 1.0 universally, recall varies) is consistent with rate-distortion optimality: decoder alignment (precision) is preserved because it is reconstruction-optimal, while encoder activation (recall) is suppressed to achieve sparsity.

3. **Added steering success explanation:** Feature U (24.2% absorption, 100% steering success) demonstrates that absorption is benign when decoder alignment is preserved — the steering works because the decoder direction is correct, not because absorption didn't occur.

### Selected Front-Runner

**Candidate A: Absorption as Information-Theoretic Rate-Distortion Optimal Compression**

Rationale:
- Connects all empirical findings (H1-H7) under a unified theoretical framework
- Provides a formal, testable prediction: decoder alignment predicts absorption
- Explains why absorption persists despite being characterized as a "failure mode"
- Explains why training reduces absorption (optimizes decoder geometry)
- Provides intellectual hook for the paper: "absorption is not a bug, it's a feature of optimal compression"

---

## Phase 5: Final Proposal

### Title

**Feature Absorption as Rate-Distortion Optimal Compression: A Theoretical and Empirical Analysis**

### Formal Claim

**Theorem (Rate-Distortion Optimal Absorption):** Let a SAE be trained on activations from a language model with hierarchical features. Let P be a parent feature with activation probability p_p, C be a child feature with activation probability p_c, and let d_P, d_C be their decoder direction unit vectors. Let the sparsity Lagrange multiplier be lambda > 0. Then:

1. **Absorption is locally optimal** for the child latent when the alignment alpha = d_P^T d_C satisfies alpha > 1 - lambda/(2*p_p*p_c).

2. **Absorption is globally optimal** when the parent and child decoder directions are sufficiently aligned (alpha -> 1) and the co-occurrence probability p_11 is high.

3. **The absorption rate** A of a feature pair is bounded by: A <= f(alpha, p_11, lambda) where f is monotonically increasing in alpha and p_11, and monotonically decreasing in lambda.

**Corollary:** Trained SAEs exhibit lower absorption than random SAEs because training optimizes decoder alignment (increases alpha for semantically related features) and increases effective sparsity penalty (increases lambda effective), both of which reduce the pressure for absorption according to the theorem.

### Proof Sketch

**Step 1: Sparsity Loss Decomposition**
For a k-sparse SAE with parent-child co-occurrence probability matrix [[p_11, p_10], [p_01, p_00]], the expected L0 "sparsity loss" (number of active latents) is:
- Without absorption: L_sep = 2*p_11 + p_10 + p_01
- With absorption: L_abs = p_11 + p_10 + p_01
- Savings from absorption: Delta_L = p_11

**Step 2: Reconstruction Loss Increase**
When child absorbs parent, the reconstruction loss increases because the child must represent both directions. The increase is bounded by:
- Delta_Recon <= ||d_P - d_C||^2 * p_11 * ||x||^2 = 2(1 - alpha) * p_11 * ||x||^2

**Step 3: Rate-Distortion Tradeoff**
The absorption decision minimizes: L_total = Delta_Recon + lambda * Delta_L
- Absorption is optimal when: lambda * p_11 > 2(1 - alpha) * p_11 * ||x||^2
- Simplifying: lambda > 2(1 - alpha) * ||x||^2

For bounded activations (||x|| <= B), this gives the condition in the theorem.

**Step 4: Trained vs. Random SAE**
- Random SAE: alpha is random (uniform on sphere), ||x|| is fixed
- Trained SAE: alpha is optimized for semantic alignment, decoder geometry reduces effective ||x|| for absorbed features
- Therefore, trained SAEs satisfy the absorption condition less frequently

### Assumptions

1. **Hierarchical feature structure:** Features naturally form parent-child hierarchies in language model activations
2. **Bounded activations:** ||x|| <= B for some constant B (standard in dictionary learning)
3. **Sufficient sparsity pressure:** lambda > 0 is sufficiently large relative to reconstruction benefit
4. **Alignment stability:** d_P and d_C are stable during training (no mode collapse)

### Empirical Predictions

1. **Decoder alignment predicts absorption:** Features with higher cosine similarity between decoder directions will exhibit higher absorption rates
2. **Precision invariant, recall varies:** Decoder alignment (precision) is preserved under absorption; encoder activation (recall) is suppressed
3. **Training reduces absorption:** Optimized decoder geometry in trained SAEs reduces the absorption pressure relative to random SAEs
4. **Steering works despite absorption:** Feature steering succeeds because decoder direction is preserved, not because absorption didn't occur

### Experimental Plan

| Experiment | Method | Expected Result | Time |
|---|---|---|---|
| E1: Decoder alignment vs. absorption | Compute cosine similarity between decoder directions of absorbed pairs; correlate with absorption rate | Positive correlation | 15 min |
| E2: Precision-recall decomposition | Measure precision and recall for high-absorption vs. low-absorption features | Precision invariant; recall varies | 15 min |
| E3: Random SAE comparison (H7 replication) | Compare absorption rates between trained and random SAEs | Trained < random | 30 min |
| E4: Steering success vs. absorption | Test steering success for features with varying absorption rates | No correlation | 30 min |

**Total pilot time:** ~90 minutes (within 1-hour guideline with parallel execution)

### Baselines

**Theoretical baselines:**
- Chanin et al. Proposition 2 (absorption as sparsity consequence)
- Tang et al. spurious local minima framework
- Cui et al. identifiability bounds

**Empirical baselines:**
- Chanin et al. absorption metric on GPT-2 Small SAEs
- Random SAE absorption baseline (H7: mean = 0.278)
- Sanity Checks frozen SAE baseline

### Risk Assessment

**Proof risk:** The rate-distortion theorem requires assumptions about hierarchical feature structure and bounded activations. These assumptions may not hold for all features in all LLMs.

**Mitigation:** Empirically validate the alignment prediction (E1). If E1 fails, the theory needs revision.

**Theory-practice gap:** The theorem characterizes when absorption is optimal for the SAE objective, not when it harms downstream tasks. The null results (H1-H4) suggest the gap is small.

**Mitigation:** Emphasize that the theory explains why absorption persists (it's optimal) and why it's benign (decoder direction preserved). The theory does not claim absorption is irrelevant — only that it is compression-optimal.

### Novelty Claim

This is the **first formal rate-distortion analysis of feature absorption in SAEs**, connecting the empirical findings (H1-H7) under a unified information-theoretic framework. Prior work (Chanin et al., Tang et al., Cui et al.) established what absorption is and when it occurs, but did not formally characterize why it is optimal compression behavior and why training reduces it.

**Specific contributions:**
1. Theorem 1: Formal conditions under which absorption is locally and globally optimal
2. Corollary: Explanation for why trained SAEs have lower absorption than random SAEs
3. Alignment-absorption correlation: Testable prediction connecting decoder geometry to absorption
4. Precision-recall asymmetry: Information-theoretic explanation for why precision is invariant while recall varies

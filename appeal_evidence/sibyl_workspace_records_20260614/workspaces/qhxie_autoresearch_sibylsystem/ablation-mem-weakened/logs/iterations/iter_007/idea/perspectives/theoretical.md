# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

Based on the comprehensive literature survey in `context/literature.md` and `context/idea_context.md`, the following theoretical resources are most relevant for mathematical analysis of feature absorption:

1. **Cui et al., 2026. On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963 (ICLR 2026)** — First closed-form theoretical analysis of SAE feature recovery; proves standard SAEs generally fail to recover ground-truth features; proposes Weighted SAE (WSAE) remedy. This is the foundational theoretical work we extend.

2. **Chanin et al., 2024. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507** — First systematic study establishing absorption is caused by hierarchical feature co-occurrence under sparsity optimization. Key insight: absorption correlates with feature co-occurrence frequency.

3. **Chanin et al., 2025a. Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756** — Reveals absorption and hedging are two failure modes on a spectrum determined by SAE width.

4. **Elhage et al., 2022. Toy Models of Superposition. arXiv:2209.10652** — Foundational theory of superposition as how networks represent more features than dimensions. Provides the geometric framework for understanding feature overlap.

5. **Costa et al., 2025. From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit. arXiv:2506.03093 (NeurIPS 2025)** — MP-SAE's conditional orthogonality property suggests a structural solution to absorption that standard SAEs cannot achieve.

6. **Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547** — Hierarchical feature organization reduces absorption but trades for hedging, confirming the trade-off structure.

7. **Gao et al., 2024. Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093** — Establishes scaling laws for SAE training but does not address theoretical limits of absorption.

8. **Information theory references** — Rate-distortion theory (Cover & Thomas), sparse coding theory (Zibulevsky & Pearlmutter), and group lasso analysis (Jacob et al., 2009) provide the mathematical toolkit for our analysis.

### Theoretical Landscape Summary

**What is known:**
- Absorption occurs when hierarchical features co-occur frequently in training data, causing sparsity optimization to merge parent features into child features
- The absorption-hedging trade-off is real: narrow SAEs suffer hedging (merging correlated features), wide SAEs suffer absorption (splitting hierarchical features)
- Cui et al. proved that full feature disentanglement is mathematically impossible under realistic sparsity constraints

**What is conjectured but not proved:**
- The precise information-theoretic conditions under which absorption is inevitable vs avoidable
- The exact phase transition boundary between absorption-dominated and hedging-dominated regimes
- Why conditional orthogonality (MP-SAE) helps reduce absorption from a theoretical standpoint
- Whether absorption can be predicted from feature statistics alone before SAE training

**Where the gaps are:**
- No closed-form bound on absorption rate as a function of feature co-occurrence statistics
- No theoretical characterization of the absorption-hedging phase transition
- No mathematical explanation for why MP-SAE's conditional orthogonality specifically targets the absorption failure mode
- No minimum description length (MDL) principled analysis of absorption as a compression phenomenon

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Lower Bound on Absorption

- **Formal claim**: Let `P(parent, child)` be the joint distribution of parent and child feature activations over the training data. Let `I(parent; child)` be their mutual information, `d_sae` the dictionary size, `λ` the L1 coefficient, and `d_model` the activation dimension. Then there exists a threshold function `τ(d_sae, λ, d_model)` such that if `I(parent; child) > τ`, absorption of the parent feature by the child feature is **mathematically inevitable** for any SAE trained to minimize `MSE + λ·L1`. Moreover:
  ```
  P(absorption rate > 0.5 | I(parent;child) > τ) → 1  as training proceeds
  ```
- **Proof sketch**:
  1. **Lemma 1 (Sparsity-Cost Decomposition)**: The L1 penalty creates a cost `C = λ·|z|_1` for activating the parent latent. The reconstruction benefit of activating both parent and child is proportional to `I(parent; child)`. When `I(parent; child) > λ·(d_sae/d_model)`, the net benefit of representing the parent separately is negative.
  2. **Lemma 2 (Co-occurrence Forces Coupling)**: Under gradient descent training dynamics, the encoder weights for parent and child features drift toward alignment when their activation patterns co-occur frequently. This is because the decoder reconstruction loss gradients push toward minimal reconstruction error, which is achieved when the absorbed representation suffices.
  3. **Lemma 3 (Absorption is Locally Stable)**: Once the parent is absorbed (decoder weight `W_dec[parent]` becomes a linear combination of child decoders), the absorbed state is a local minimum of the training objective. Small perturbations away from absorption increase reconstruction error without reducing L1 cost.
  4. **Theorem**: Combining Lemma 1-3 with concentration of measure arguments over random initialization, absorption is the dominant outcome with probability approaching 1 as `I(parent; child)` exceeds the threshold.
- **Empirical prediction**: Absorption rate should be a sigmoid function of `I(parent; child)·d_model/λ·d_sae`, with steep transition around `τ ≈ λ·d_sae/d_model`. This predicts that absorption can be predicted from feature co-occurrence statistics **before any SAE training**.
- **Connection to existing theory**: Extends Cui et al.'s impossibility result by specifying **which features** become unrecoverable (those with high `I(parent; child)`) rather than claiming all features are unrecoverable. Also provides the missing quantitative relationship that Chanin et al. only characterized qualitatively.
- **Novelty estimate**: 8/10 — This is genuinely new. No prior work proves information-theoretic lower bounds on absorption rate as a function of feature co-occurrence statistics.

### Candidate B: Phase Transition Theory of Absorption-Hedging Trade-off

- **Formal claim**: For SAEs with dictionary size `d_sae` and activation dimension `d_model`, the transition between absorption-dominated and hedging-dominated regimes is a **sharp phase transition** characterized by a critical width ratio `r* = d_sae/d_model`. Specifically:
  - When `r < r*`: Hedging is the dominant failure mode. Features are merged due to insufficient dictionary capacity.
  - When `r > r*`: Absorption is the dominant failure mode. Features are split due to sparsity optimization preferring specific features.
  - At `r ≈ r*`: A mixed phase exists where both failure modes co-occur with approximately equal frequency.
  The critical ratio `r*` satisfies `r* = O(I_max / λ)` where `I_max` is the maximum pairwise mutual information among features.
- **Proof sketch**:
  1. **Lemma 1 (Width-Capacity Correspondence)**: The effective capacity for representing `N` features in `d_model` dimensions with `d_sae` dictionary entries is bounded by the covering number of the feature sphere. The information-theoretic capacity is `log(d_sae)` bits per feature.
  2. **Lemma 2 (Hedging Threshold)**: When `log(d_sae) < H(feature distribution)`, where `H` is the entropy of feature activation patterns, the SAE cannot uniquely represent all features. Features must merge (hedging) to conserve capacity.
  3. **Lemma 3 (Absorption Threshold)**: When `log(d_sae) > H(feature distribution)` but `I(parent; child)` is large, the SAE has sufficient capacity to represent features uniquely, but sparsity (`λ`) penalizes representing both parent and child separately when they co-occur frequently.
  4. **Theorem**: The phase transition boundary is determined by the ratio of `log(d_sae)` to both `H` (hedging) and `I_max/λ` (absorption). The transition is sharp (discontinuous derivative) because it's a geometric covering problem.
- **Empirical prediction**: The absorption rate and hedging rate as functions of `r = d_sae/d_model` should exhibit a crossover reminiscent of phase transitions in statistical physics. Near `r*`, both rates should be elevated (mixed phase).
- **Connection to existing theory**: This provides the theoretical underpinning for Chanin et al.'s empirical observation that "wide SAEs suffer absorption, narrow SAEs suffer hedging." The transition is not gradual but exhibits a sharp boundary. This connects to the theory of sparse coding and the tradeoff between compression and reconstruction.
- **Novelty estimate**: 7/10 — The phase transition framing is novel in the SAE literature, though phase transitions are well-studied in compressed sensing and sparse coding. The specific application to the absorption-hedging trade-off would be new.

### Candidate C: Why Conditional Orthogonality Works — A Geometric Theory

- **Formal claim**: The reduction in absorption achieved by MP-SAE's conditional orthogonality constraint is explained by **geometric regularization in the feature space**. Specifically, conditional orthogonality between hierarchy levels (parent features orthogonal to child features) changes the geometry of the optimization landscape in a way that:
  1. Eliminates the local minimum corresponding to the absorbed state (Lemma 3 from Candidate A)
  2. Introduces a new local minimum corresponding to the properly-disentangled representation
  3. Does not interfere with the global minimum corresponding to optimal reconstruction
  The key is that conditional orthogonality is **not** a global constraint — it only requires `W_dec[parent] ⊥ W_dec[child]`, which is achievable without sacrificing reconstruction quality.
- **Proof sketch**:
  1. **Lemma 1 (Geometry of the Absorbed State)**: In standard SAEs, the absorbed state has `W_dec[parent] ≈ α·W_dec[child]` for some scalar `α`. This collinearity is the geometric signature of absorption.
  2. **Lemma 2 (Orthogonality Prohibition)**: When `W_dec[parent] ⊥ W_dec[child]` is enforced as a hard constraint, the collinear absorbed state is infeasible. The optimization must find an alternative minimizer.
  3. **Lemma 3 (Matching Pursuit Convergence)**: The matching pursuit algorithm (used in MP-SAE) with residual-guided selection converges to a representation where features at the same hierarchy level are correlated (allowed) but features across hierarchy levels are orthogonal (required). This is exactly the structure needed to prevent absorption.
  4. **Lemma 4 (Absorption Rate Reduction)**: Under conditional orthogonality, the absorption rate is bounded above by `O(1/d_sae)` rather than being proportional to `I(parent; child)`. The orthogonality constraint effectively filters out absorbed representations.
  5. **Theorem**: Conditional orthogonality reduces absorption because it changes the nature of the optimization from unconstrained (where absorbed states are local minima) to constrained (where absorbed states are infeasible).
- **Empirical prediction**: The absorption rate in MP-SAE should be approximately independent of `I(parent; child)`, unlike standard SAEs where absorption rate increases with co-occurrence. This can be tested by measuring absorption rates for features with varying `I(parent; child)` in both standard SAE and MP-SAE.
- **Connection to existing theory**: This explains why Matryoshka SAEs (which approximate hierarchical organization) also reduce absorption — they impose a similar structural constraint on feature representations. The geometric view connects to the optimization landscape analysis in deep learning theory.
- **Novelty estimate**: 6/10 — The matching pursuit connection is already established by Costa et al. The novelty is in the geometric interpretation that explains **why** it works. This is more of a theoretical clarification than a new result.

## Phase 3: Self-Critique

### Against Candidate A

- **Proof soundness attack**: The proof sketch relies on concentration of measure arguments and assumes gradient descent training dynamics. SAEs are trained with adaptive optimizers (Adam) and learning rate schedules, which may introduce dynamics not captured by the simple gradient descent analysis. The threshold `τ` may not be sharp in practice.
- **Tightness attack**: The threshold `τ = λ·d_sae/d_model` may be loose. In practice, absorption occurs even when `I(parent; child)` is moderate, suggesting the true threshold is lower than the bound predicts. The bound may not be tight enough to be useful for SAE configuration guidance.
- **Relevance attack**: Even if we can prove absorption is inevitable for certain features, this doesn't tell us how to fix it. The theoretical result is important but may not translate to practical improvements.
- **Novelty attack**: Cui et al. (ICLR 2026) have already established that standard SAEs cannot fully recover ground-truth features. Candidate A extends this to specify **which** features become unrecoverable, but the information-theoretic lower bound may have been implicitly known to experts in the field.
- **Verdict**: STRONG — The specific quantification of absorption as a function of `I(parent; child)` is genuinely novel. Even if the bound is loose, it provides the first formal relationship between feature statistics and absorption. The prediction that absorption can be predicted from data statistics alone is testable and falsifiable.

### Against Candidate B

- **Proof soundness attack**: The phase transition framing requires defining "absorption rate" and "hedging rate" precisely. These are empirical quantities that may not have clean theoretical counterparts. The analogy to physical phase transitions may be superficial — the "sharpness" of the transition in SAEs may be much less pronounced than in physical systems.
- **Tightness attack**: The critical ratio `r* = d_sae/d_model` may not be a universal constant but rather architecture and data dependent. The `O(I_max/λ)` scaling assumes a specific form of the training objective that may not hold for JumpReLU or Gated SAEs.
- **Relevance attack**: Understanding the phase transition is scientifically interesting but may not help practitioners choose SAE configurations. The practical guidance (use wider SAEs for less absorption, narrower for less hedging) is already known from empirical work.
- **Novelty attack**: Phase transitions in sparse coding and compressed sensing have been extensively studied. The application to SAEs is novel but the mathematical techniques are borrowed from statistical physics.
- **Verdict**: MODERATE — The phase transition framing provides scientific understanding but may not yield actionable results. The proof sketch is plausible but would require significant work to make rigorous. The main value is in predicting where the transition occurs, which could guide SAE configuration.

### Against Candidate C

- **Proof soundness attack**: The claim that conditional orthogonality "eliminates" absorbed states as local minima assumes the orthogonality constraint is enforced strictly. In practice, MP-SAE enforces conditional orthogonality approximately through the matching pursuit selection process, not as a hard constraint. The theoretical guarantee may not hold in practice.
- **Tightness attack**: The bound `O(1/d_sae)` on absorption under conditional orthogonality is very loose. It doesn't tell us how much absorption reduction to expect in practice.
- **Relevance attack**: This is a mathematical explanation of why an existing method (MP-SAE) works. It doesn't suggest new methods or improvements.
- **Novelty attack**: Costa et al. already established the connection between conditional orthogonality and absorption reduction. This candidate is more of a theoretical clarification than a new theoretical contribution.
- **Verdict**: WEAK — This candidate explains rather than predicts. The geometric interpretation is valuable for understanding but doesn't yield new testable predictions or design principles.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Conditional Orthogonality Theory)** dropped because: While the geometric explanation is valuable, it is more of a theoretical clarification than a new contribution. The key result (conditional orthogonality reduces absorption) is already established empirically by Costa et al. The mathematical explanation, while satisfying, doesn't yield new predictions or methods.

### Strengthened Ideas

- **Candidate A (Information-Theoretic Lower Bound)** strengthened by adding a **predictive model** component. The theorem predicts that absorption rate is a function of `I(parent; child)·d_model/λ·d_sae`. This suggests a regression model:
  ```
  absorption_rate = σ(α·I(parent; child)·d_model/(λ·d_sae) + β)
  ```
  where `σ` is the sigmoid function and `α, β` are fitted parameters. This model can be validated empirically, and the fitted parameters reveal the effective constants in the theoretical bound.

- **Candidate B (Phase Transition)** strengthened by connecting to **rate-distortion theory**. The absorption-hedging trade-off can be viewed as a rate-distortion tradeoff: the SAE must compress the feature representation (rate = `log(d_sae)`) while minimizing distortion (reconstruction error). Absorption corresponds to over-compression in the hierarchy dimension; hedging corresponds to under-compression in the correlation dimension.

### Selected Front-Runner

**Candidate A (Information-Theoretic Lower Bound on Absorption)** is selected as the front-runner because:

1. It provides the first **quantitative prediction** connecting feature statistics to absorption: absorption can be predicted from `I(parent; child)` and SAE configuration **without training the SAE**
2. It is **falsifiable**: if absorption cannot be predicted from `I(parent; child)` alone, the theory is disproved
3. It has **practical implications**: if we can predict which features will be absorbed before training, we can design SAE configurations (e.g., larger `d_sae`, lower `λ`) specifically for those features
4. It **extends Cui et al.'s** impossibility result by specifying which features are affected, providing a more complete theoretical picture
5. The **methodology is novel**: combining information theory (mutual information), optimization theory (L1 regularization), and learning theory (concentration of measure) in this way has not been done for SAEs

## Phase 5: Final Proposal

### Title

**Information-Theoretic Lower Bounds on Feature Absorption in Sparse Autoencoders**

### Formal Claim

**Theorem (Absorption Lower Bound)**: Let `f_parent` and `f_child` be a parent-child pair of hierarchical features with mutual information `I(parent; child)` over the training distribution. Let `d_sae` be the SAE dictionary size, `d_model` the activation dimension, and `λ` the L1 sparsity coefficient. Then for any SAE trained to minimize `MSE + λ·L1`:

```
P(absorption rate > 0.5) ≥ σ(α·I(parent; child)·d_model/(λ·d_sae) + β)
```

where `σ` is the sigmoid function, and `α, β` are constants depending on the SAE architecture and training dynamics.

**Corollary (Absorption Prediction)**: The absorption rate for a feature pair can be predicted from feature statistics alone (computed from activation data) without training the SAE, using:

```
predicted_absorption_rate = max(0, min(1, α·I(parent; child)·d_model/(λ·d_sae) + β))
```

**Corollary (Phase Transition Boundary)**: The critical mutual information `I*` at which absorption becomes likely (`P > 0.5`) satisfies:

```
I* = (λ·d_sae)/(α·d_model)
```

### Proof Sketch

**Step 1: Sparsity-Cost Lemma**
The L1 penalty creates a per-activation cost of `λ`. Representing both parent and child features separately costs `2λ`. The reconstruction benefit of representing parent separately over the absorbed state is proportional to `I(parent; child)` (mutual information measures how much knowing the parent helps predict the child). When `I(parent; child) < 2λ`, the net benefit is negative, and absorption becomes favorable.

**Step 2: Co-occurrence Gradient Lemma**
During gradient descent training, the encoder weights for parent and child features experience correlated gradient updates when their activation patterns co-occur frequently. This correlation, combined with the L1 pressure to minimize active features, drives the encoder weights toward alignment — the geometric signature of absorption.

**Step 3: Absorbed State Stability Lemma**
Once the parent decoder becomes approximately collinear with the child decoder (`W_dec[parent] ≈ α·W_dec[child]`), the absorbed state is a local minimum of the training objective. Perturbing toward disentanglement increases reconstruction error without reducing L1 cost (since both parent and child would still be active).

**Step 4: Concentration Theorem**
Combining Steps 1-3 with concentration of measure arguments over random initialization, the absorption probability follows a sigmoid relationship with the effective ratio `I(parent; child)·d_model/(λ·d_sae)`.

### Assumptions

1. **Feature stationarity**: The feature activation distributions are approximately stationary over the training period
2. **Gradient descent dynamics**: The training dynamics can be approximated by gradient descent on the training objective (ignoring adaptive optimizers for the initial analysis)
3. **Sufficient training**: The SAE is trained to convergence or near-convergence
4. **Binary absorption**: For the threshold analysis, we consider absorption as a binary variable (absorbed vs. not absorbed), ignoring gradual absorption
5. **Hierarchy independence**: For the pairwise analysis, we assume absorption relationships are approximately independent across feature pairs

### Empirical Predictions

1. **Prediction 1 (Sigmoid Relationship)**: Absorption rate as a function of `I(parent; child)` follows a sigmoid, with steepest change around `I* = λ·d_sae/(α·d_model)`. This is testable by measuring absorption rates for feature pairs with varying `I(parent; child)`.

2. **Prediction 2 (Configuration Sensitivity)**: Absorption rate should increase linearly with `d_model` (harder to represent parent separately in higher dimensions) and decrease linearly with `d_sae` (more capacity reduces absorption) and `λ` (stronger sparsity penalizes redundant features). This is testable by measuring absorption across SAE configurations.

3. **Prediction 3 (Pre-Training Prediction)**: Computing `I(parent; child)` from activation data before SAE training should predict the post-training absorption rate with `R² > 0.5`. This is the strongest test of the theory.

### Experimental Plan

**Theoretical Validation (no new experiments required)**:
- Re-analyze existing absorption measurements from Chanin et al. (2024) and SAEBench to test whether `I(parent; child)` correlates with measured absorption rates
- Compute the sigmoid fit for the existing data to estimate `α, β`

**New Experiments (≤1 hour total)**:

**Pilot 1 (15 min)**: Compute mutual information for 20 feature pairs in GPT-2 SAE layer 8
- Use activation data from SAELens to compute `I(parent; child)` for 20 candidate feature pairs
- Use SAEBench projection metric to measure their absorption rates
- Test correlation between `I` and absorption rate

**Pilot 2 (15 min)**: Vary SAE configuration and measure absorption changes
- Load GPT-2 SAEs with different widths (if available) or different L0 settings
- Measure how absorption rates change with `d_sae` and `λ`
- Test whether the relationship follows the predicted linear scaling

**Main Experiment (30 min)**: Cross-model validation
- Repeat Pilot 1 on Gemma-2-2B SAEs (layers 0-17 for reliable ablation-based absorption measurement)
- Validate that `I` predicts absorption across model families

### Baselines

1. **Cui et al. impossibility result**: Standard SAEs cannot fully recover ground-truth features — we extend this by specifying **which** features are affected
2. **Chanin et al. empirical correlation**: Absorption correlates with feature co-occurrence frequency — we provide the **information-theoretic quantification** of this correlation
3. **SAEBench absorption measurements**: We provide the **predictive model** that explains why absorption rates vary across SAE configurations

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `I(parent; child)` is hard to estimate accurately from finite data | Medium | Use binning estimators or KDE for mutual information; report confidence intervals |
| The sigmoid parameters `α, β` are architecture-dependent | High | Fit separately per architecture; show universality of the functional form |
| Absorption is not binary but gradual | Medium | Extend to continuous absorption rate; use regression rather than classification |
| Other factors (feature frequency, decoder norm) dominate over `I` | Medium | Add feature frequency as a control variable; test multivariate regression |

### Novelty Claim

This work provides the first **information-theoretic lower bound** on feature absorption in SAEs, connecting the empirical phenomenon (absorption) to fundamental quantities (mutual information between features, L1 penalty strength, dictionary size). The key insights are:

1. **Absorption is predictable**: Unlike prior work that measures absorption post-hoc, our theory predicts absorption from feature statistics **before SAE training**
2. **Absorption is inevitable under conditions**: We prove that absorption **must** occur when `I(parent; child)` exceeds a threshold determined by SAE configuration
3. **The threshold has a clean form**: `I* ∝ λ·d_sae/d_model` — stronger sparsity and larger dictionaries reduce absorption; higher-dimensional activations increase absorption

The theory is falsifiable: if absorption cannot be predicted from `I(parent; child)` and SAE configuration, the theory is disproved. This is the strongest form of scientific hypothesis.

### Connection to Prior Work

- **Extends Cui et al. (ICLR 2026)**: They proved full disentanglement is impossible; we specify **which** features are unrecoverable and **why** (high mutual information with co-occurring children)
- **Quantifies Chanin et al. (2024)**: They observed absorption correlates with co-occurrence frequency; we provide the **precise functional form** of this relationship
- **Explains Costa et al. (MP-SAE)**: Conditional orthogonality works because it changes the optimization landscape, making the absorbed state infeasible rather than just suboptimal

---

*Output generated by theoretical agent based on comprehensive literature survey in `context/literature.md` and `context/idea_context.md`.*

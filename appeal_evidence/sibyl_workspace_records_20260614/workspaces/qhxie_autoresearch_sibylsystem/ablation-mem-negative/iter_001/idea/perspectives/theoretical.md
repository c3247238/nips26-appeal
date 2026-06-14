# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Elhage et al. (2022)** — *Toy Models of Superposition* (Transformer Circuits Thread, arXiv:2209.10652)
   - **Key mathematical result**: Established the superposition hypothesis — neural networks can represent n > m features in m dimensions by using approximately (not strictly) orthogonal directions. Proved that ReLU nonlinear filtering enables superposition where linear models cannot. Discovered "sticky" geometric structures (digons, triangles, pentagons, tetrahedra) that features organize into, with feature dimensionalities clustering at rational values.
   - **Framework**: Weighted MSE loss L = sum_x sum_i I_i (x_i - x'_i)^2 with ReLU output model x' = ReLU(W^T W x + b).

2. **Chanin et al. (2024)** — *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders* (arXiv:2409.14507)
   - **Key mathematical result**: Formalized feature absorption as a phenomenon where parent features get "swallowed" by child features due to sparsity loss + hierarchical co-occurrence. Toy model with 4 true features showing SAE merges "starts with S" into "short" to increase sparsity while maintaining reconstruction.
   - **Framework**: Absorption metric based on k-sparse probing + integrated gradients ablation.

3. **Makkuva et al. (2024)** — *Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders* (arXiv:2411.13117)
   - **Key mathematical result**: **Theorem 1**: For sparse sources S in R^N with at most K >= 2 non-zero entries, linearly projected into M-dimensional space (K log(N/K) <= M < N) satisfying RIP, a sparse autoencoder with a linear-nonlinear (L-NL) encoder **must have a non-zero amortisation gap**. The encoder's weight matrix W = W_d * W_e has rank at most M, preventing full-rank recovery.
   - **Framework**: Disentangles dictionary learning from sparse inference; proves SAEs sacrifice optimal sparsity for computational efficiency.

4. **Chen et al. (2025)** — *Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders* (arXiv:2506.14002, ICLR 2025)
   - **Key mathematical result**: **First SAE algorithm with theoretical recovery guarantees**. Proves Bias Adaptation (BA) provably recovers all monosemantic features under conditions: network width M >= n * p^(s/(1-eps)^2), frequency range n^{-1} <= p <= min{n^{-(1+s^{-1})/2}, n^{-2(1+eps)^2/s}, d^{(1-zeta)/n}}, with probability >= 1 - n^{-4eps}.
   - **Framework**: Gaussian analysis of random feature matrices, concentration inequalities (Efron-Stein), dynamical analysis of weight trajectories.

5. **Cui et al. (2025)** — *On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy* (arXiv:2506.15963)
   - **Key mathematical result**: **First necessary and sufficient conditions for identifiable SAEs**: (1) extreme sparsity of ground truth features, (2) sparse activation (ReLU/Top-k), (3) sufficient hidden dimensions (n_m >= n). Theorem 1: Under extreme sparsity, SAEs recover ground truth with probability 1. Theorem 2: W_m* = W_p^T is the unique solution when n_m = n and sparsity approaches 1.
   - **Framework**: Superposition hypothesis (linear representation); identifies feature shrinking and feature vanishing phenomena.

6. **Chanin et al. (2025)** — *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders* (arXiv:2505.11756)
   - **Key mathematical result**: Identified hedging as the complement to absorption. Toy model proof (Appendix A.2) that MSE loss causes hedging when correlated features exist and SAE is narrower than the number of true features. For single-latent SAE with f_2 => f_1, latent l = (1-alpha)f_1 + alpha f_2 minimizes loss at alpha in (0,1).
   - **Framework**: Matryoshka SAEs trade absorption for hedging; balanced loss with coefficients beta_m achieves near-optimal representations at beta ~ 0.25.

7. **Lee et al. (2025)** — *Evaluating and Designing Sparse Autoencoders by Approximating Quasi-Orthogonality* (arXiv:2503.24277)
   - **Key mathematical result**: **Approximate Feature Activation (AFA) Theorem**: The l2-norm of sparse feature vector can be approximated using the l2-norm of dense embedding vector with error bound: ||f||_2^2 in [||z(x)||_2^2 / (1 + eps(h-1)), ||z(x)||_2^2 / (1 - eps(h-1))], where eps is the quasi-orthogonality constant. Connects to Johnson-Lindenstrauss Lemma.
   - **Framework**: Epsilon-quasi-orthogonality as geometric constraint; leads to Top-AFA architecture eliminating manual k selection.

8. **Chanin & Garriga-Alonso (2025)** — *Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders* (arXiv:2508.16560)
   - **Key mathematical result**: The sparsity-reconstruction trade-off is **non-monotonic** and misleading. When L0 is below true sparsity, SAEs engage in "feature hedging" — mixing correlated features to improve reconstruction while producing polysemantic latents. An incorrect SAE with too-low L0 can achieve better reconstruction than a ground-truth correct SAE. Proposed decoder pairwise cosine similarity c_dec as proxy metric.
   - **Framework**: c_dec = (1/C(h,2)) sum_{i<j} |cos(W_dec,i, W_dec,j)|; minimized exactly at true L0.

9. **Garg et al. (2026)** — *How Many Features Can a Language Model Store Under the Linear Representation Hypothesis?* (arXiv:2602.11246)
   - **Key mathematical result**: Nearly-matching upper and lower bounds for "linear compressed sensing": upper bound d = O_eps(k^2 log m), lower bound d = Omega_eps((k^2/log k) * log(m/k)). Separates linear representation from linear accessibility — the latter requires quadratically more dimensions.
   - **Framework**: Rank bounds for near-identity matrices (Alon 2003) + Turan's theorem on clique-free graphs.

10. **Bussmann et al. (2025)** — *Learning Multi-Level Features with Matryoshka Sparse Autoencoders* (arXiv:2503.17547)
    - **Key mathematical result**: Hierarchical nested SAEs achieve ~90% reduction in absorption (0.49 -> 0.05 at L0=40) by enforcing independent reconstruction constraints at multiple dictionary sizes. Introduces feature hedging in inner levels as trade-off.
    - **Framework**: Loss L = sum_{m in M} beta_m (||a - a_hat_m||_2^2 + lambda S_m) + alpha L_aux.

11. **Korznikov et al. (2025)** — *Orthogonal Sparse Autoencoders Uncover Atomic Features* (arXiv:2509.22033)
    - **Key mathematical result**: Cosine similarity penalty between latents achieves ~65% absorption reduction with ~50% less compute than Matryoshka. Alternative solution via orthogonality constraints rather than hierarchy.
    - **Framework**: Penalty term on decoder pairwise cosine similarities.

12. **Follow-up: Theoretical Foundation of Sparse Dictionary Learning** (arXiv:2512.05534)
    - **Key mathematical result**: Unified framework encompassing SAEs, transcoders, crosscoders. Generalizes identifiability analysis beyond tied-weight SAEs; explains absorption and neuron resampling.
    - **Framework**: Extends Cui et al. to broader class of sparse dictionary learning methods.

### Theoretical Landscape Summary

**What is known:**
- SAEs can recover ground-truth monosemantic features under extreme sparsity conditions (Cui et al., Chen et al.)
- The amortization gap is provably non-zero for standard L-NL encoders (Makkuva et al.)
- Absorption is caused by sparsity loss + hierarchical co-occurrence (Chanin et al. 2024)
- Hedging is caused by MSE loss + correlations in narrow SAEs (Chanin et al. 2025)
- Matryoshka SAEs trade absorption for hedging; balance at beta ~ 0.25 (Chanin et al. 2025)
- Quasi-orthogonality bounds feature activation norms via JL Lemma (Lee et al.)
- Linear accessibility requires d = Theta(k^2 log m) dimensions (Garg et al.)

**What is conjectured:**
- The absorption-hedging trade-off has a fundamental Pareto frontier that can be characterized theoretically
- There exists an information-theoretic lower bound on the joint probability of absorption and hedging
- The non-monotonic L0-feature quality relationship generalizes beyond toy models

**Where the gaps are:**
1. No closed-form characterization of the absorption-hedging Pareto frontier
2. No information-theoretic bound connecting sparsity, reconstruction, and feature quality
3. No theoretical prediction of absorption rate as a function of architecture parameters
4. No proof that absorption is inevitable (rather than just empirically observed) under certain conditions
5. No theoretical framework for unsupervised absorption detection

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Lower Bound on the Absorption-Hedging Trade-off

- **Formal claim**: For any SAE with latent dimension m reconstructing n > m hierarchical features with sparsity S and correlation structure C, there exists a fundamental lower bound on the joint probability of absorption and hedging: P(absorption) + P(hedging) >= f(n, m, S, C) > 0. Specifically, for features with parent-child hierarchy where P(child|parent) = p, the bound is Omega((1-S) * p * (n-m)/n).

- **Proof sketch**:
  1. Model the SAE as a communication channel: input X (ground-truth features) -> encoder -> latent Z -> decoder -> reconstruction X_hat.
  2. By the data processing inequality, I(X; X_hat) <= I(X; Z) <= H(Z).
  3. With m latents and sparsity constraint E[||Z||_0] = L0, the maximum mutual information is bounded by H(Z) <= m * h_b(L0/m) where h_b is binary entropy.
  4. For n features with hierarchical correlations, the source entropy H(X) >= n * h_b(S) + I_hier where I_hier is the hierarchical mutual information.
  5. When H(X) > H(Z), information must be lost. The lost information either manifests as absorption (parent feature information captured by child latents) or hedging (multiple features merged into single latents).
  6. Apply Fano's inequality to lower bound the error probability: P(error) >= (H(X|X_hat) - 1) / log(|X|).

- **Empirical prediction**: As L0 increases (more sparsity), absorption should decrease but hedging should increase, with their sum bounded below by a constant. This predicts a "forbidden region" in the absorption-hedging plane that no SAE architecture can enter.

- **Connection to existing theory**: Extends Makkuva et al.'s amortization gap to a joint bound on both pathologies. Connects to Cui et al.'s identifiability conditions by showing that even when identifiability is possible in principle, the finite-capacity constraint creates unavoidable errors.

- **Novelty estimate**: 8/10. Information theory has been applied to autoencoders (InfoMax, VAE) but not specifically to characterize the absorption-hedging trade-off. The connection of Fano's inequality to SAE pathologies is new.

---

### Candidate B: A Theoretical Explanation for Why Absorption Emerges from Sparsity Pressure — With a Predictive Formula

- **Formal claim**: In a toy model with parent feature f_p (activation probability alpha) and child features f_c1, ..., f_ck (each activating with probability beta < alpha, conditional on f_p), the standard SAE loss L = ||x - D z||^2 + lambda ||z||_1 has an optimal solution where the parent latent's encoder develops a "hole" (negative component for child directions) if and only if lambda > lambda_crit = (alpha - beta) * ||f_p||^2 / (k * beta). This provides a closed-form critical sparsity penalty threshold for absorption.

- **Proof sketch**:
  1. Set up the toy model: x = sum_i f_i * v_i + noise, with hierarchical constraint that child features only activate when parent does.
  2. Parameterize the SAE with parent latent l_p and child latents l_c1, ..., l_ck.
  3. Write the expected loss E[L] as a function of encoder/decoder parameters.
  4. Compute the gradient of E[L] with respect to the parent encoder's weight on child direction v_ci.
  5. Show that for lambda > lambda_crit, the gradient pushes the parent encoder weight negative (creating the "hole" characteristic of absorption).
  6. Verify by showing that at lambda = lambda_crit, the parent encoder has zero component in child directions (no absorption, no hedging — the "balanced" point).

- **Empirical prediction**: The critical lambda_crit should predict the onset of absorption in real SAEs. For GemmaScope SAEs, computing lambda_crit from empirical activation frequencies should correlate with measured absorption rates. If lambda < lambda_crit for a given feature pair, absorption should not occur; if lambda > lambda_crit, absorption should occur with probability increasing in (lambda - lambda_crit).

- **Connection to existing theory**: Extends Chanin et al. (2024)'s empirical observation that absorption is caused by sparsity loss + hierarchical co-occurrence into a quantitative prediction. The lambda_crit formula connects to Cui et al.'s "extreme sparsity" condition — when features are extremely sparse (beta << alpha), lambda_crit becomes very small, meaning absorption is inevitable at any reasonable sparsity penalty.

- **Novelty estimate**: 7/10. The toy model setup builds on Chanin et al. but the closed-form critical threshold is new. The predictive formula for real SAEs has not been proposed.

---

### Candidate C: The Absorption-Hedging Pareto Frontier is Characterized by a Single Parameter — Feature Density

- **Formal claim**: For a family of SAEs parameterized by width m (with fixed input dimension d and feature count n), there exists a single scalar "feature density" rho = n/(m * S) that characterizes the entire absorption-hedging Pareto frontier. Specifically, for any SAE achieving L0 sparsity and reconstruction error eps, the absorption rate A and hedging rate H satisfy: A * H = Theta(exp(-c * rho)) for some constant c depending on the correlation structure. As rho increases (more features per latent, or lower sparsity), the product A*H decreases exponentially, but the individual rates trade off along a hyperbolic curve.

- **Proof sketch**:
  1. Define feature density rho = n/(m * S) = (average features per latent) / (activation probability).
  2. Model the SAE as a random coding problem: each latent "covers" a random subset of features.
  3. Absorption occurs when a parent feature's coverage is entirely contained within a child feature's coverage (parent is "swallowed").
  4. Hedging occurs when multiple features share the same latent (latent "covers" multiple features simultaneously).
  5. Using coupon-collector-like analysis, compute the probability of containment (absorption) and collision (hedging) as functions of rho.
  6. Show that P(containment) ~ exp(-rho * p_child) and P(collision) ~ 1 - exp(-rho * (1-p_child)), where p_child is the conditional activation probability.
  7. The product P(containment) * P(collision) has the claimed exponential form.

- **Empirical prediction**: Sweeping SAE width m while keeping n and S fixed should trace out a hyperbolic curve in the (A, H) plane. The curve's position should be determined solely by rho, not by individual values of n, m, S. Different architectures (JumpReLU, TopK, Matryoshka) should all lie on the same frontier, with only the achievable region differing.

- **Connection to existing theory**: Connects to Garg et al.'s capacity bound (d = Theta(k^2 log m)) by showing that the effective "k" in SAEs is not just the sparsity but the feature density rho. Connects to Elhage et al.'s superposition phase changes by predicting discontinuous transitions at critical rho values.

- **Novelty estimate**: 9/10. The unification of absorption and hedging under a single parameter is genuinely novel. The random coding / coupon collector analogy has not been applied to SAEs. The prediction that all architectures share the same Pareto frontier is falsifiable and surprising.

---

## Phase 3: Self-Critique

### Against Candidate A (Information-Theoretic Lower Bound)

- **Proof soundness attack**: The data processing inequality argument is sound, but the connection from "information loss" to "absorption OR hedging" is not rigorous. Information loss could manifest in other ways (dead features, noise amplification, feature vanishing). The bound f(n, m, S, C) needs to be derived more carefully — the proposed Omega((1-S) * p * (n-m)/n) is a heuristic, not a proved bound.
  - *Counterexample search*: Look for SAE architectures where information loss manifests primarily as dead features rather than absorption/hedging. If such architectures exist, the bound does not apply.

- **Tightness attack**: The bound using binary entropy h_b(L0/m) is loose — real SAE latents are not independent binary variables. The actual H(Z) could be much smaller due to correlations between latents. This makes the bound potentially vacuous (the lower bound could be below trivial thresholds).

- **Relevance attack**: Information-theoretic bounds are often too loose to be practically useful. Even if we prove P(absorption) + P(hedging) >= 0.01, this tells practitioners little about how to design better SAEs. The bound needs to be tight enough to predict actual behavior.

- **Novelty attack**: Information theory has been applied to autoencoders extensively (VAE, InfoMax, IB principle). The specific application to absorption/hedging is new, but the techniques are standard. Need to verify no one has published a similar bound.
  - *Search result*: No direct prior work found connecting Fano's inequality to SAE absorption. However, the MRMR principle (max I(H;X) - beta sum I(h_i;h_j)) from recent OpenReview work is related — it explicitly uses mutual information between features.

- **Verdict**: MODERATE. The core idea is sound but the bound needs significant tightening to be useful. Risk of producing a "mathematical curiosity" rather than actionable insight.

---

### Against Candidate B (Critical Sparsity Threshold for Absorption)

- **Proof soundness attack**: The toy model assumes linear features and a specific hierarchical structure. Real SAEs operate on nonlinearly transformed activations from deep networks. The gradient analysis assumes continuous optimization reaching global optimum; real SAE training uses SGD with local minima. The "hole" in the encoder is necessary but not sufficient for absorption — the decoder must also misbehave.
  - *Counterexample search*: Search for cases where lambda > lambda_crit but absorption does not occur (e.g., due to decoder compensation or feature orthogonality).

- **Tightness attack**: The formula lambda_crit = (alpha - beta) * ||f_p||^2 / (k * beta) depends on knowing the ground-truth feature norms and activation probabilities. In practice, these are unknown. The formula may not be predictive without oracle knowledge.

- **Relevance attack**: This is the strongest aspect — if the formula works, it directly predicts absorption from measurable quantities (activation frequencies). However, the project is training-free, so we can only measure on pre-trained SAEs, not validate by training new ones with controlled lambda.

- **Novelty attack**: Chanin et al. (2024) already provided a toy model showing absorption emerges from sparsity pressure. The closed-form threshold is an extension, not a fundamentally new result. Cui et al. (2025) also analyzed optimal solutions in toy settings.
  - *Search result*: No prior work provides a closed-form critical threshold for absorption onset. The "Sparse but Wrong" paper analyzes L0 effects but not a per-feature-pair threshold.

- **Verdict**: STRONG. The proof sketch is plausible, the empirical prediction is falsifiable, and the novelty is clear. The main risk is that real SAEs are too complex for the toy model to apply.

---

### Against Candidate C (Feature Density Characterizes Pareto Frontier)

- **Proof soundness attack**: The random coding analogy is heuristic, not rigorous. Real SAEs are not random codes — they are optimized for reconstruction. The coupon collector analysis assumes uniform random coverage, but SAE optimization creates highly structured coverage. The exponential form A*H = Theta(exp(-c*rho)) is conjectural, not derived.
  - *Counterexample search*: Matryoshka SAEs explicitly avoid the trade-off by using multiple dictionary sizes. If Matryoshka achieves A*H << exp(-c*rho), the claim is falsified.

- **Tightness attack**: The "single parameter" claim is very strong. Even if rho captures much of the variation, architecture-specific effects (activation function, loss coefficients, training dynamics) likely matter. The claim may hold approximately but not exactly.

- **Relevance attack**: If true, this would be extremely useful — it would mean practitioners only need to tune one parameter (rho) rather than navigating a complex trade-off. However, the claim that "all architectures lie on the same frontier" might be too strong to be useful (if the frontier is just "everything bad").

- **Novelty attack**: The absorption-hedging trade-off was identified by Chanin et al. (2025). Characterizing it via a single parameter is new, but the random coding approach is standard in information theory. Need to check if similar analyses exist for other dictionary learning methods.
  - *Search result*: No prior work characterizes the absorption-hedging frontier. The "Sparse but Wrong" paper shows non-monotonic behavior but does not propose a unifying parameter.

- **Verdict**: MODERATE. The idea is bold and potentially impactful, but the proof is the weakest of the three. The random coding analogy needs to be replaced with a proper optimization analysis. However, even as a conjecture with empirical support, it could be valuable.

---

## Phase 4: Refinement

### Dropped
- **Candidate A** is dropped as the primary focus. The information-theoretic bound is too loose to be practically useful, and the risk of producing a "mathematical curiosity" is high. However, the Fano's inequality perspective may be worth a brief mention in the Discussion section.

### Strengthened
- **Candidate B** is strengthened by narrowing its scope and adding empirical validation:
  - Instead of claiming a universal formula, focus on the toy model derivation and then empirically test whether the qualitative prediction (lambda_crit exists and correlates with absorption) holds on real SAEs.
  - Add the connection to Chanin et al.'s "Sparse but Wrong" finding: if lambda_crit is small (features are dense), then any reasonable sparsity penalty will cause absorption, explaining why "most existing SAEs have L0 that is too low."
  - The toy model should be kept simple (2-3 features) with closed-form solutions, then extended to larger settings numerically.

- **Candidate C** is reframed as a **conjecture with empirical support** rather than a theorem:
  - Instead of claiming a proved bound, propose the "feature density hypothesis" as a unifying framework.
  - Design experiments that test whether sweeping rho (via width m or sparsity S) traces out a consistent frontier across architectures.
  - If the data supports it, the conjecture becomes a contribution; if not, the negative result is also valuable.

### Additional Evidence
- Searched for prior work on "critical sparsity threshold" and "absorption onset" — no direct matches found.
- Searched for prior work on "feature density" in SAEs — no direct matches found.
- The combination of a toy model with closed-form solution (Candidate B) and an empirical Pareto frontier (Candidate C) is complementary and novel.

### Selected Front-Runner
**Candidate B** (Critical Sparsity Threshold for Absorption) is selected as the primary theoretical contribution, with **Candidate C** (Feature Density Hypothesis) as a secondary exploratory direction.

Rationale:
1. Candidate B has the strongest proof sketch and clearest empirical prediction.
2. It directly addresses the project's core question (what causes absorption and when).
3. It connects to multiple existing papers (Chanin et al. 2024, Cui et al. 2025, Chanin et al. 2025 "Sparse but Wrong").
4. The training-free constraint is compatible — we can measure lambda_crit proxies on pre-trained SAEs.
5. Candidate C provides a natural extension if Candidate B's predictions are validated.

---

## Phase 5: Final Proposal

### Title
**"A Critical Sparsity Threshold for Feature Absorption in Sparse Autoencoders: Theory and Evidence"**

### Formal Claim

**Main Theorem (Toy Model):** Consider a sparse autoencoder with input x = f_p * v_p + sum_{i=1}^k f_{c,i} * v_{c,i} + eta, where f_p ~ Bernoulli(alpha), f_{c,i} ~ Bernoulli(beta) conditional on f_p = 1 (hierarchical co-occurrence), and {v_p, v_{c,1}, ..., v_{c,k}} are orthonormal feature directions. Let the SAE have parent latent l_p and child latents l_{c,i} with loss:

L = E[||x - D z||^2] + lambda E[||z||_1]

where z = ReLU(W x + b). Then there exists a critical sparsity penalty:

lambda_crit = (alpha - beta) / (k * beta) * ||v_p||^2

such that:
- For lambda < lambda_crit: The parent encoder w_p has positive component in child directions (no absorption; potential hedging if m < n).
- For lambda = lambda_crit: The parent encoder w_p has zero component in child directions (the "balanced" point).
- For lambda > lambda_crit: The parent encoder w_p develops negative components in child directions (absorption occurs).

Furthermore, the absorption severity (measured by the magnitude of negative encoder components) increases monotonically with (lambda - lambda_crit).

**Corollary (Predictive Formula):** For real SAEs, define the empirical critical threshold:

lambda_emp = (freq_parent - freq_child) / (n_children * freq_child)

where frequencies are estimated from activation data. Then lambda_emp should correlate with measured absorption rates: feature pairs with lambda_emp < lambda_actual (the SAE's effective sparsity penalty) should show absorption; pairs with lambda_emp > lambda_actual should not.

### Proof Sketch

**Step 1: Setup.** Parameterize the encoder weights as w_p = a_p * v_p + sum_i b_i * v_{c,i} + w_perp (perpendicular component), and similarly for child encoders. The decoder is D = [d_p, d_{c,1}, ..., d_{c,k}].

**Step 2: Expected Loss.** Compute E[||x - D z||^2] by conditioning on the feature activation pattern. There are 2^{k+1} possible patterns; due to hierarchy, only patterns where f_p = 1 can have f_{c,i} = 1.

**Step 3: Simplify.** Due to orthonormality, the loss decouples across feature directions. The key term is the reconstruction of v_p when only f_p is active (no children): the error is ||v_p - d_p * ReLU(a_p + b_p)||^2 where b_p is the bias.

**Step 4: Gradient Analysis.** Compute dL/db_i (gradient of loss w.r.t. parent encoder's child-direction component). Show that:
- The reconstruction gradient pushes b_i > 0 (to help reconstruct child features when they co-occur).
- The sparsity gradient pushes b_i < 0 (to reduce parent activation when children are active, saving L1 penalty).
- The balance point occurs when these gradients cancel, yielding lambda_crit.

**Step 5: Verify Monotonicity.** Show that d^2 L / db_i d lambda < 0, meaning increasing lambda beyond lambda_crit makes b_i increasingly negative (deeper absorption).

**Step 6: Extension to Non-Orthogonal Features.** Use Gram-Schmidt or perturbation analysis to extend to approximately orthogonal features (as in real superposition). The threshold shifts by a factor of 1/(1 - eps^2) where eps is the maximum pairwise cosine similarity.

### Assumptions

1. **Linear representation hypothesis**: Features are linearly encoded in the input activations.
2. **Hierarchical co-occurrence**: Child features only activate when parent does (P(child|not parent) = 0).
3. **Orthonormal feature directions**: In the toy model, features are orthogonal. Extended to approximately orthogonal in Corollary.
4. **ReLU activation**: The SAE uses ReLU or ReLU-like activation (TopK, JumpReLU can be approximated).
5. **Global optimum**: The analysis assumes the SAE reaches the global optimum of its loss function.
6. **Stationary distribution**: Feature activations follow a stationary Bernoulli process.

### Empirical Prediction

**Prediction 1 (Qualitative):** For any pre-trained SAE, feature pairs with strong hierarchical co-occurrence (high P(child|parent)) and moderate parent frequency should show absorption if the SAE's effective sparsity penalty is high.

**Prediction 2 (Quantitative):** Compute lambda_emp for known absorbed feature pairs (from Chanin et al.'s first-letter test cases). Compare with lambda_emp for non-absorbed pairs. The distribution of lambda_emp should be significantly lower for absorbed pairs (Mann-Whitney U test, p < 0.05).

**Prediction 3 (Monotonicity):** For a family of SAEs with varying L0 (e.g., GemmaScope SAEs with different width/sparsity configurations), absorption rate should increase with effective lambda. Plotting absorption rate vs. 1/L0 should show a sigmoidal curve with an inflection point.

### Experimental Plan

**Experiment 1: Toy Model Validation (15 minutes)**
- Implement the 3-feature toy model (1 parent, 2 children) in PyTorch.
- Train SAEs with varying lambda from 0.01 to 10.
- Measure encoder component b_i and verify the sign change at lambda_crit.
- **Success criterion**: b_i changes sign within 20% of predicted lambda_crit.

**Experiment 2: Pre-trained SAE Analysis (30 minutes)**
- Load GemmaScope JumpReLU SAEs (layer 8, widths 16k, 65k, 262k, 1M) for Gemma-2-2B.
- Identify first-letter features using k-sparse probing (Chanin et al. method).
- Compute lambda_emp for parent-child pairs (e.g., "starts with S" -> "starts with SH").
- Compare lambda_emp between absorbed and non-absorbed pairs.
- **Success criterion**: Significant difference in lambda_emp distributions (p < 0.05).

**Experiment 3: Cross-Architecture Validation (30 minutes)**
- Load GPT-2 Small SAEs from SAELens (TopK, Gated, Standard).
- Repeat Experiment 2 analysis.
- Test whether lambda_emp ranking is consistent across architectures.
- **Success criterion**: Spearman correlation > 0.5 between lambda_emp and absorption rate across architectures.

**Experiment 4: Feature Density Hypothesis (Pilot, 15 minutes)**
- Compute rho = n_features / (m * S) for each SAE configuration.
- Plot absorption rate vs. rho and hedging rate vs. rho.
- Test whether A * H shows the predicted relationship with rho.
- **Success criterion**: Clear monotonic trend visible; if not, report negative result.

### Baselines

**Theoretical baselines:**
- Cui et al. (2025): SAEs fail unless features are extremely sparse. Our result predicts *how* they fail (absorption vs. hedging) and *when* (lambda_crit).
- Chanin et al. (2024): Absorption is caused by sparsity + hierarchy. Our result quantifies this into a predictive formula.
- Chanin et al. (2025) "Sparse but Wrong": L0 being too low causes hedging. Our result explains *why* L0 = "too low" is relative to lambda_crit.

**Empirical baselines:**
- Random baseline: Absorption rate for randomly selected feature pairs (should be lower than hierarchical pairs).
- Frequency-only baseline: Predict absorption using only parent frequency (without the (alpha - beta)/(k*beta) structure).
- Architecture baseline: Compare predictive power of lambda_emp across JumpReLU, TopK, Gated, Matryoshka.

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Toy model does not extend to real SAEs | Medium | High | Frame as "toy model insight" with empirical correlation, not causal proof |
| lambda_emp computation is noisy | Medium | Medium | Use bootstrap confidence intervals; report effect sizes |
| Pre-trained SAEs lack hierarchical features | Low | High | Use first-letter features (established test case); fallback to synthetic data |
| Proof has gap in Step 4 | Medium | High | Include proof sketch in paper, note as "conjecture with strong empirical support" |
| Effect size is small | Medium | Medium | Pre-register analysis; report negative results prominently |

### Novelty Claim

The specific theoretical contribution is:

1. **First closed-form critical threshold** for absorption onset in SAEs. Prior work (Chanin et al. 2024) identified absorption qualitatively; Cui et al. (2025) analyzed optimal solutions but did not derive an absorption threshold; "Sparse but Wrong" (Chanin 2025) identified the non-monotonic L0 effect but did not provide a per-feature-pair prediction.

2. **First predictive formula** linking measurable activation statistics to absorption probability. Practitioners can compute lambda_emp from their SAE's activation data to identify at-risk feature pairs.

3. **Unified explanation** for why absorption and hedging are complementary: they are the two sides of the lambda_crit boundary — below the threshold, SAEs hedge; above, they absorb. This connects two previously separate literatures.

Evidence for novelty: Comprehensive literature search (arXiv, Google Scholar, Web) found no prior closed-form threshold for absorption. The closest work is Chanin et al.'s toy model (which shows absorption emerges but does not quantify when) and Cui et al.'s identifiability analysis (which gives conditions for perfect recovery but not the failure mode when conditions are violated).

---

## Sources

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Elhage et al., 2022
- [A is for Absorption](https://arxiv.org/abs/2409.14507) — Chanin et al., 2024
- [Compute Optimal Inference and Provable Amortisation Gap](https://arxiv.org/abs/2411.13117) — Makkuva et al., 2024
- [Taming Polysemanticity in LLMs](https://arxiv.org/abs/2506.14002) — Chen et al., 2025
- [On the Limits of Sparse Autoencoders](https://arxiv.org/abs/2506.15963) — Cui et al., 2025
- [Feature Hedging](https://arxiv.org/abs/2505.11756) — Chanin et al., 2025
- [Evaluating and Designing SAEs by Approximating Quasi-Orthogonality](https://arxiv.org/abs/2503.24277) — Lee et al., 2025
- [Sparse but Wrong](https://arxiv.org/abs/2508.16560) — Chanin & Garriga-Alonso, 2025
- [How Many Features Can a Language Model Store](https://arxiv.org/abs/2602.11246) — Garg et al., 2026
- [Matryoshka SAEs](https://arxiv.org/abs/2503.17547) — Bussmann et al., 2025
- [Orthogonal SAEs](https://arxiv.org/abs/2509.22033) — Korznikov et al., 2025

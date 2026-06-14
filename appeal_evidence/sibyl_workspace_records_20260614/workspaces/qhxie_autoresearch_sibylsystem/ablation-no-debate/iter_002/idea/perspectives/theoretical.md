# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024/2025)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (NeurIPS 2025, arXiv:2409.14507)
   - **Key mathematical result**: Formal proof (Appendix A.2) that for hierarchical features f_child ⊂ f_parent, the sparsity loss L_sp^{(1,2)}(δ) = p_11(2−δ) + p_10 is strictly decreasing in δ, creating a fundamental gradient incentive toward absorption. The δ-absorption parameterization shows perfect reconstruction is maintained for all δ ∈ [0,1], but sparsity loss is minimized at δ = 1 (full absorption).
   - **Framework**: First systematic definition of absorption rate; toy model with provable loss-minimization property.

2. **Marks et al. (2025/2026)** — "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534)
   - **Key mathematical result**: Unified SDL framework proving (Theorem 3.2) piecewise biconvexity within activation-pattern regions; (Theorem 3.6) prevalence of spurious partial minima exhibiting polysemanticity; (Theorem 3.8) hierarchical concept structures induce realizable activation patterns that manifest as feature-absorbing partial minima; (Theorem 3.4) zero-loss solutions are underdetermined — reconstruction loss can vanish without recovering any ground-truth features.
   - **Framework**: Connects mechanistic interpretability to classical optimization theory; introduces "feature anchoring" to restore identifiability.

3. **Cui et al. (2025)** — "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy" (ICLR 2025, arXiv:2506.15963)
   - **Key mathematical result**: First closed-form solution for SAEs (Theorem 1): W_m^* = I^*(W_p, 0)^⊤. Proves standard SAEs fail to fully recover ground-truth monosemantic features due to feature shrinking and vanishing. Full recovery guaranteed only under extreme sparsity (S → 1). Proposes WSAE with adaptive weight selection principle (Theorem 5).
   - **Framework**: Identifiability analysis under Linear Representation Hypothesis; reweighted remedy via polysemantic-level-adaptive weights.

4. **Chen et al. (2025)** — "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders" (ICLR 2026, arXiv:2506.14002)
   - **Key mathematical result**: First SAE algorithm with provable feature recovery guarantee (Theorem 6.1). Under their statistical model, with network width M ≳ n · p^{s/(1−ε)^2} and frequency range n^{−1} ≲ p ≲ min{n^{−(1+s^{−1})/2}, ...}, all features are recovered within T = ς^{−1} iterations with probability ≥ 1 − n^{−4ε}. Introduces "neuron resonance" — features recover when activation frequency matches feature occurrence frequency.
   - **Framework**: Statistical feature-recovery model; Group Bias Adaptation (GBA) algorithm with theoretical convergence.

5. **Sun & Huang (2025)** — "Global Identifiability of Overcomplete Dictionary Learning via L1 and Volume Minimization" (ICLR 2025)
   - **Key mathematical result**: Novel formulation combining ℓ_1 norm of sparse coefficients + log-volume of dictionary. Proves global identifiability for overcomplete dictionaries (m < k) under "m-strongly scattered in k-hypercube" condition. Sample complexity n = O((k^2/m) log(k^2/m)) for sparse-Gaussian model.
   - **Framework**: First global identifiability guarantee for overcomplete dictionary learning; bridges classical DL theory to SAEs.

6. **Lee et al. (2025)** — "Evaluating and Designing Sparse Autoencoders by Approximating Quasi-Orthogonality" (arXiv:2503.24277)
   - **Key mathematical result**: Formalizes ε-quasi-orthogonality (|u·v| ≤ ε for distinct unit vectors). Derives Approximate Feature Activation (AFA) — closed-form approximation of sparse feature activation magnitudes under ε-quasi-orthogonality. Introduces ε_LBO metric linking input embeddings to activation magnitudes.
   - **Framework**: Geometric framework connecting superposition to Johnson-Lindenstrauss Lemma; top-AFA architecture with theoretically justified adaptive sparsity.

7. **Pacela et al. (2026)** — "Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation" (arXiv:2603.28744)
   - **Key mathematical result**: Demonstrates that the dictionary itself (not the encoder/amortization gap) is the binding constraint for compositional generalization. Oracle per-sample FISTA with ground-truth dictionary achieves near-perfect OOD recovery (MCC ≥ 0.83), while SAE-learned dictionaries fail regardless of encoder quality.
   - **Framework**: Reframing SAE failure as dictionary learning challenge; evidence that learned dictionaries point in substantially wrong directions.

8. **Oursland (2026)** — "Deriving Decoder-Free Sparse Autoencoders from First Principles" (arXiv:2601.06478)
   - **Key mathematical result**: Derives architecture purely from implicit EM theory. Shows gradient descent on LSE objectives performs implicit EM (gradient = responsibility). Achieves 93.4% probe accuracy vs. 90.3% for standard SAE with half the parameters.
   - **Framework**: InfoMax principle; decoder-free architecture eliminates encoder-decoder asymmetry that enables absorption.

9. **Hu et al. (2025)** — "Measuring Sparse Autoencoder Feature Sensitivity" (arXiv:2509.23717)
   - **Key mathematical result**: Empirical finding that average feature sensitivity declines with increasing SAE width across 7 variants up to 1M features. Sensitivity measures fraction of LLM-generated similar texts that activate a feature.
   - **Framework**: New evaluation dimension complementary to absorption; scaling law in sensitivity vs. width.

10. **Bussmann et al. (2025)** — "Learning Multi-Level Features with Matryoshka Sparse Autoencoders" (arXiv:2503.17547)
    - **Key mathematical result**: Nested dictionaries trained simultaneously achieve Pareto-optimal sparsity-reconstruction tradeoffs. Reduced absorption from 0.49 to 0.05; reduced splitting from 3 latents to 1.
    - **Framework**: Hierarchical structure as architectural solution to absorption.

11. **Tang et al. (2025)** — "Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability" (arXiv:2512.13568)
    - **Key mathematical result**: Frames superposition through rate-distortion theory. Connects SAEs to minimum description length (MDL) principles. Measures effective "alphabet size" for neural network internal communication via entropy-based framework.
    - **Framework**: Information-theoretic foundation for understanding why superposition emerges.

12. **Luo et al. (2026)** — "Building a Structured Feature Forest with Hierarchical Sparse Autoencoders" (arXiv:2602.11881)
    - **Key mathematical result**: HSAE with explicit parent-child relationships and structural constraint loss substantially outperforms baselines on absorption, especially at larger sizes.
    - **Framework**: Explicit hierarchical modeling as absorption mitigation.

### Theoretical Landscape Summary

**What is known:**
- Feature absorption is not an artifact but a **fundamental consequence** of the SAE objective: for hierarchical features f_child ⊂ f_parent, the sparsity loss strictly decreases with absorption (Chanin et al.).
- The SDL optimization landscape is **piecewise biconvex** with **prevalent spurious minima** that exhibit polysemanticity and absorption (Marks et al.).
- Zero reconstruction loss can be achieved **without recovering any ground-truth features** — the problem is underdetermined (Marks et al., Theorem 3.4).
- Full feature recovery is only guaranteed under **extreme sparsity** (S → 1, nearly 1-sparse); general sparsity leads to shrinking and vanishing (Cui et al.).
- Global identifiability of overcomplete dictionaries is possible under strong geometric conditions (Sun & Huang).
- The **dictionary** (not the encoder) is the fundamental bottleneck for compositional generalization (Pacela et al.).

**What is conjectured:**
- A unified theoretical framework connecting sparsity, hierarchy, sensitivity, and absorption (Gap 5 in literature survey).
- Information-theoretic lower bounds on absorption rates for given sparsity/hierarchy parameters.
- Whether training-free detection of absorption is theoretically possible via encoder-decoder asymmetry.

**Where the gaps are:**
1. **No lower bound on absorption rate**: Chanin et al. prove absorption decreases loss, but no characterization of minimum achievable absorption given hierarchy structure and sparsity constraint.
2. **No information-theoretic characterization**: Rate-distortion theory has been applied to superposition but not specifically to absorption as a coding phenomenon.
3. **No theoretical link between sensitivity and absorption**: Both are empirical quality metrics, but their quantitative relationship is unexplored theoretically.
4. **Encoder-decoder asymmetry is observed but not characterized**: Chanin et al. note it as a potential detection signal, but no formal analysis exists.
5. **No sample complexity bounds for absorption detection**: How many samples are needed to reliably detect absorption in pretrained SAEs?

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Lower Bound on Absorption Rate

- **Formal claim**: For an SAE with dictionary D ∈ ℝ^{n×m} (n input dimensions, m latent dimensions, m > n), trained with L1 sparsity penalty λ on data generated from a hierarchical feature model where parent feature f_p and child feature f_c have co-occurrence probability p_{11} = P(f_p ∧ f_c), the absorption rate α satisfies a lower bound:

  **Theorem (Informal)**: Under the Linear Representation Hypothesis and assuming features are encoded as unit-norm directions with hierarchical inclusion f_c ⊂ f_p, any SAE minimizing L = L_rec + λ L_sp achieves absorption rate:
  
  α ≥ (1 − exp(−λ p_{11} / σ^2)) · I(f_p; f_c) / H(f_p)
  
  where σ^2 is the reconstruction noise variance and I(f_p; f_c), H(f_p) are mutual information and entropy of the feature occurrences.

- **Proof sketch**:
  1. Model the SAE as a lossy compression channel: the latent representation must encode both f_p and f_c, but sparsity constraint limits the "rate."
  2. By rate-distortion theory, when I(f_p; f_c) > 0 and rate is constrained, the optimal code reuses the child feature's direction to represent the parent (absorption).
  3. Quantify the tradeoff: each unit of absorption reduces the sparsity penalty by λ p_{11} while increasing reconstruction distortion by at most O(δ^2).
  4. At optimality, marginal benefit of absorption equals marginal cost, yielding the lower bound.

- **Empirical prediction**: SAEs with higher L1 penalty λ should show higher absorption rates. The bound predicts absorption rate increases with parent-child co-occurrence probability p_{11} and decreases with reconstruction noise σ^2.

- **Connection to existing theory**: Extends Chanin et al.'s proof that absorption decreases loss by providing a quantitative lower bound. Connects to Tang et al.'s rate-distortion framing of superposition.

- **Novelty estimate**: 7/10. Information theory has been applied to superposition generally but not specifically to absorption as a rate-distortion phenomenon.

### Candidate B: Encoder-Decoder Asymmetry as a Necessary Condition for Absorption

- **Formal claim**: In a standard SAE with tied or untied weights, feature absorption occurs if and only if there exists encoder-decoder asymmetry for the absorbed feature. Specifically, for a parent feature f_p absorbed into child feature f_c with absorption parameter δ:

  **Proposition**: Let W_enc ∈ ℝ^{m×n} and W_dec ∈ ℝ^{n×m} be the encoder and decoder matrices. Feature f_p is δ-absorbed into f_c if and only if:
  
  ||W_enc[f_c] · f_p − W_enc[f_p] · f_p|| / ||W_enc[f_p] · f_p|| ≥ δ  (encoder suppression)
  
  AND
  
  ||W_dec[f_c] − proj_{f_p} W_dec[f_c]|| / ||W_dec[f_c]|| ≤ √(1−δ^2)  (decoder alignment)

  Moreover, the product of encoder suppression and decoder alignment equals the reconstruction fidelity of the absorbed feature.

- **Proof sketch**:
  1. Start from Chanin et al.'s δ-parameterization of the absorbed solution.
  2. Show that encoder weights for the parent feature must shrink (suppression) to maintain sparsity when child feature activates.
  3. Show decoder weights must align with parent direction to maintain reconstruction.
  4. Prove the "if and only if" by constructing non-absorbed solutions that violate the asymmetry conditions and showing they have higher loss.

- **Empirical prediction**: Training-free detection of absorption is possible by measuring encoder-decoder asymmetry ratio A(f) = ||W_enc[f]|| / ||W_dec[f]|| for each latent. Absorbed features should have A(f) significantly different from 1, with parent features having A(f) < 1 (encoder suppressed) and child features having A(f) > 1 (decoder amplified).

- **Connection to existing theory**: Formalizes Chanin et al.'s observation that "absorption leads to an asymmetric pattern in the encoder and decoder." Connects to Pacela et al.'s finding that the dictionary (decoder) is the binding constraint.

- **Novelty estimate**: 8/10. The asymmetry is observed empirically but never characterized as a necessary and sufficient condition. Training-free detection via asymmetry is a new practical implication.

### Candidate C: A Unified Sensitivity-Absorption Characterization via Conditional Entropy

- **Formal claim**: Feature sensitivity (Hu et al.) and feature absorption (Chanin et al.) are dual manifestations of the same underlying information-theoretic quantity: the conditional entropy of feature activation given semantic class. Specifically:

  **Theorem**: For a learned SAE feature φ representing ground-truth concept c, define:
  - Sensitivity: s(φ) = P(φ activates | c is present)
  - Absorption rate: a(φ) = P(φ fails to activate | c is present, child features activate)
  
  Then under the hierarchical feature model:
  
  s(φ) + a(φ) ≤ 1 − H(φ | c) / log(2)
  
  with equality when the SAE achieves the rate-distortion optimum for the feature hierarchy.

  **Corollary**: Low-sensitivity features are disproportionately absorbed, with the relationship:
  
  a(φ) ≥ 1 − s(φ) − H(φ | c) / log(2)

- **Proof sketch**:
  1. Model feature activation as a binary channel with input c (concept present/absent) and output φ (feature activates/does not).
  2. Sensitivity is the true positive rate of this channel; absorption is the false negative rate conditional on child activation.
  3. Apply Fano's inequality to relate the error probability (1 − s(φ)) to conditional entropy H(φ | c).
  4. Decompose the error into absorption (hierarchy-induced) and other sources (noise, dead features).
  5. The bound follows from information-theoretic constraints on the channel capacity.

- **Empirical prediction**: Joint measurement of sensitivity and absorption on the same feature set should reveal a negative correlation bounded by the theoretical inequality. Features with sensitivity < 0.5 should have absorption rate > 0.3. The unified score U(φ) = s(φ) + a(φ) + H(φ|c)/log(2) should cluster near 1 for well-trained SAEs.

- **Connection to existing theory**: Bridges two previously separate literatures (sensitivity from Hu et al., absorption from Chanin et al.). Connects to Marks et al.'s unified SDL framework by showing how hierarchical structure constrains both metrics simultaneously.

- **Novelty estimate**: 9/10. No prior work connects sensitivity and absorption theoretically. This addresses Gap 5 directly and provides a unified quality score.

---

## Phase 3: Self-Critique

### Against Candidate A: Information-Theoretic Lower Bound

- **Proof soundness attack**: The rate-distortion connection is elegant but may be vacuous. The "rate" in SAEs is not Shannon rate but L0/L1 sparsity — the correspondence between ℓ_1 penalty and information rate is heuristic, not rigorous. The bound may reduce to a tautology if λ is treated as a Lagrange multiplier for an implicit rate constraint. Also, the reconstruction noise variance σ^2 is not independently estimable.

- **Tightness attack**: The bound involves mutual information I(f_p; f_c), which for real LLM features is impossible to compute (ground-truth features are unknown). In the toy model where features are known, the bound may be extremely loose — the exponential form exp(−λ p_{11}/σ^2) could be nearly 0 or nearly 1 for all realistic parameters, making the bound uninformative.

- **Relevance attack**: Practitioners care about detecting and mitigating absorption, not about lower bounds. The bound says "absorption must happen" but doesn't help quantify how much or which features are affected. It confirms intuition without enabling new methods.

- **Novelty attack**: Tang et al. (2025) already apply rate-distortion theory to superposition. The specific application to absorption is new, but the general framework is not. Marks et al.'s Theorem 3.8 already proves hierarchy induces absorption structurally.

- **Verdict**: MODERATE. The idea is intellectually appealing but the bound may be vacuous and the practical utility limited. Worth pursuing only if a constructive (non-vacuous) form can be derived.

### Against Candidate B: Encoder-Decoder Asymmetry Characterization

- **Proof soundness attack**: The "if and only if" claim is strong. For tied-weight SAEs (W_dec = W_enc^⊤), asymmetry is impossible by construction, yet absorption still occurs in the latent activations (different latents absorb different features). The claim must be restricted to untied SAEs or reformulated in terms of latent activations rather than weight matrices. Also, the norm-based conditions may not capture directional alignment properly — cosine similarity would be more appropriate.

- **Tightness attack**: The asymmetry ratio A(f) = ||W_enc[f]|| / ||W_dec[f]|| may not be a sharp indicator. In practice, both norms vary widely across features for reasons unrelated to absorption (feature frequency, scale of activation). Normalization by feature frequency or reconstruction contribution may be needed.

- **Relevance attack**: This is highly relevant to the project's training-free constraint. If asymmetry reliably signals absorption, it enables post-hoc detection without causal ablation. The practical utility is strong.

- **Novelty attack**: Chanin et al. explicitly mention encoder-decoder asymmetry as a potential detection signal but do not develop it. Pacela et al. show the decoder is the binding constraint. The formal characterization as a necessary and sufficient condition is new. No prior paper proposes using asymmetry for training-free detection.

- **Verdict**: STRONG. The proof needs refinement (restrict to untied SAEs, use cosine-based measures), but the core insight is novel and practically valuable.

### Against Candidate C: Unified Sensitivity-Absorption Characterization

- **Proof soundness attack**: The application of Fano's inequality requires the feature activation channel to be memoryless and stationary, which may not hold for SAE features that interact through the sparsity constraint. The conditional entropy H(φ | c) is not directly measurable without knowing ground-truth concept c. In practice, c must be approximated by probes or human labels, introducing measurement error that propagates into the bound.

- **Tightness attack**: The bound s(φ) + a(φ) ≤ 1 − H(φ|c)/log(2) may be very loose. For binary variables, Fano's inequality gives P_e ≥ (H(X|Y) − 1)/log(|X|), which for |X| = 2 becomes P_e ≥ H(X|Y) − 1. If H(φ|c) < 1, the bound becomes negative (vacuous). Since SAE features are typically very sparse, H(φ|c) may indeed be small, making the bound uninformative for most features.

- **Relevance attack**: If the bound is loose, the unified score U(φ) may not cluster as predicted. However, the empirical correlation between sensitivity and absorption is itself valuable — even without the theoretical bound, establishing their quantitative relationship addresses a major gap.

- **Novelty attack**: No prior work connects these two metrics. The novelty is high. However, the information-theoretic framing may be unnecessary — a simpler statistical correlation analysis might suffice and be more robust.

- **Verdict**: MODERATE. The information-theoretic bound may be too loose to be useful, but the empirical correlation analysis is valuable. Consider reformulating as a statistical characterization rather than an information-theoretic bound.

---

## Phase 4: Refinement

### Dropped
- **Candidate A (Information-theoretic lower bound)**: The bound risks being vacuous due to the heuristic rate-sparsity correspondence and the difficulty of estimating I(f_p; f_c) for real features. The practical utility is limited compared to the other candidates.

### Strengthened
- **Candidate B (Encoder-Decoder Asymmetry)**: Selected as front-runner. Strengthened formulation:
  - Restrict to untied-weight SAEs (the dominant architecture in practice: Gemma Scope, GPT-2 SAEs).
  - Replace norm ratio with **directional asymmetry**: measure the angle between encoder and decoder directions for each latent.
  - Define **Asymmetry Index**: AI(f) = 1 − cos(W_enc[f], W_dec[f]). For absorbed parent features, AI(f) → 1 (encoder suppressed, decoder aligned with parent). For child features, AI(f) is small (both aligned with child).
  - Add empirical validation: test on SynthSAEBench where ground-truth absorption is known.

- **Candidate C (Sensitivity-Absorption)**: Reformulated as a **statistical characterization** rather than information-theoretic bound:
  - Hypothesis: Low-sensitivity features are disproportionately absorbed.
  - Test: Measure Pearson/Spearman correlation between sensitivity and absorption rate on matched feature sets.
  - Prediction: Correlation ρ < −0.5, with absorbed features having mean sensitivity < 0.3.
  - Unified score: Q(φ) = (1 − a(φ)) · s(φ) as a composite quality metric.

### Additional Evidence
- Marks et al. (Theorem 3.8) proves hierarchical structures induce absorption — this supports the theoretical foundation for Candidate B.
- Pacela et al. prove the decoder is the binding constraint — this supports focusing on decoder-side asymmetry.
- Hu et al. show sensitivity declines with width — this suggests sensitivity is a structural property, not just noise.

### Selected Front-Runner
**Candidate B: Encoder-Decoder Asymmetry as a Necessary and Sufficient Condition for Feature Absorption**

This is the strongest candidate because:
1. It directly addresses Gap 4 (training-free detection) and Gap 1 (ablation-independent methods).
2. It has a clear, testable formal claim.
3. It enables a practical method (training-free absorption detection) with theoretical justification.
4. The proof sketch is concrete and builds on established results (Chanin et al.'s δ-parameterization).
5. It aligns with the project's training-free constraint.

---

## Phase 5: Final Proposal

### Title
**"Encoder-Decoder Asymmetry: A Theoretical Characterization and Training-Free Detection of Feature Absorption in Sparse Autoencoders"**

### Formal Claim

**Main Theorem (Absorption-Asymmetry Equivalence)**: Consider a sparse autoencoder with untied encoder W_e ∈ ℝ^{m×n} and decoder W_d ∈ ℝ^{n×m}, trained to minimize L = 𝔼_x[||x − W_d σ(W_e x)||^2] + λ L_sp, where σ is a sparsity-inducing activation. Let f_p, f_c ∈ ℝ^n be unit-norm hierarchical features with f_c ⊂ f_p (child ⊂ parent) and co-occurrence probability p_{11} = P(f_p ∧ f_c) > 0.

Define the **directional asymmetry** for latent j as:

A_j = arccos( (w_e^{(j)} · w_d^{(j)}) / (||w_e^{(j)}|| · ||w_d^{(j)}||) )

where w_e^{(j)} is the j-th row of W_e and w_d^{(j)} is the j-th column of W_d.

**Statement**: Under the Linear Representation Hypothesis and Chanin et al.'s δ-absorption model, there exists a threshold τ = τ(λ, p_{11}, n, m) such that:

1. **(Necessity)** If latent j represents a parent feature f_p that is δ-absorbed into child latents, then A_j ≥ arccos(1 − δ^2/2) − O(λ).

2. **(Sufficiency)** If A_j ≥ τ and the latent's decoder direction w_d^{(j)} has significant projection onto a ground-truth parent feature f_p (i.e., (w_d^{(j)} · f_p)^2 ≥ θ), then f_p is at least partially absorbed into latents with smaller asymmetry.

**Corollary (Training-Free Detection)**: The **Asymmetry Index** AI(j) = 1 − cos(A_j) provides a training-free signal for absorption: AI(j) > 1 − cos(τ) implies the feature represented by latent j is absorbed with probability at least 1 − ε, where ε depends on the false positive rate of the projection threshold θ.

### Proof Sketch

**Step 1: δ-Parameterization of Absorbed Solutions**
Following Chanin et al., parameterize the absorbed solution by δ ∈ [0,1]:
- Encoder: w_e^{(parent)} = 0 (suppressed), w_e^{(child)} contains parent direction with weight δ.
- Decoder: w_d^{(parent)} = (1−δ) f_p, w_d^{(child)} = f_c + δ f_p.

**Step 2: Compute Directional Asymmetry**
For the parent latent under δ-absorption:
- w_e^{(parent)} = 0 (or nearly zero under regularization).
- w_d^{(parent)} = (1−δ) f_p.
- The cosine similarity cos(A_parent) = (w_e · w_d) / (||w_e|| ||w_d||) → 0 as ||w_e|| → 0.
- Thus A_parent → π/2 and AI(parent) → 1.

For the child latent:
- w_e^{(child)} aligns with f_c + δ f_p.
- w_d^{(child)} aligns with f_c + δ f_p.
- cos(A_child) ≈ 1 (small asymmetry).

**Step 3: Prove Necessity**
Show that any solution minimizing the SAE loss must have the above structure when absorption occurs. Use the fact that sparsity loss decreases with δ (Chanin et al., Proposition 2) while reconstruction is maintained, so gradient descent converges to δ > 0.

**Step 4: Prove Sufficiency**
Show that if AI(j) is large and w_d^{(j)} projects onto a parent feature, then the encoder must be suppressing that feature (otherwise sparsity would not be optimal). The suppressed encoder plus aligned decoder implies absorption by the definition of Chanin et al.

**Step 5: Establish Threshold τ**
Derive τ from the tradeoff between reconstruction error and sparsity penalty. At optimality, the marginal benefit of absorption (reduced sparsity) equals the marginal cost (increased asymmetry-induced reconstruction error). This yields τ as a function of λ, p_{11}, and model dimensions.

### Assumptions

1. **Linear Representation Hypothesis (LRH)**: Features are represented as linear directions in activation space.
2. **Untied weights**: Encoder and decoder are independently parameterized (holds for Gemma Scope, most modern SAEs).
3. **Hierarchical feature model**: Ground-truth features form a DAG with parent-child inclusion relationships.
4. **Chanin et al. δ-model**: The family of absorbed solutions is adequately parameterized by δ.
5. **Sparsity-inducing activation**: σ enforces sufficient sparsity (TopK, L1, or JumpReLU).
6. **Known ground-truth features**: For validation, we assume access to ground-truth features (via SynthSAEBench or synthetic data).

### Empirical Prediction

**Prediction 1**: In pretrained SAEs (Gemma Scope, GPT-2), latents with high Asymmetry Index (AI > 0.5) are significantly more likely to represent absorbed parent features than latents with low AI.

**Prediction 2**: The correlation between AI and absorption rate (measured by Chanin et al.'s causal ablation method on layers 0-17) is ρ > 0.7.

**Prediction 3**: On SynthSAEBench where ground-truth is known, AI achieves AUROC > 0.85 for absorption detection without any ablation experiments.

### Experimental Plan

**Experiment 1: Validation on Synthetic Ground Truth (15 minutes)**
- Dataset: SynthSAEBench-16k (ground-truth features with known hierarchy).
- Load pretrained SAE or train a small SAE on synthetic data.
- Compute AI for all latents.
- Compare AI against ground-truth absorption labels.
- Metrics: AUROC, precision@k, correlation with true absorption rate.

**Experiment 2: Correlation with Causal Ablation on Gemma-2-2B (30 minutes)**
- Layers: 0-17 (where ablation works).
- SAE: Gemma Scope 16k or 65k.
- Compute AI for all latents.
- Run Chanin et al. ablation-based absorption detection on first-letter features.
- Measure Pearson/Spearman correlation between AI and absorption rate.
- Test Prediction 2.

**Experiment 3: Cross-Architecture Comparison (15 minutes)**
- SAEs: TopK, JumpReLU, ReLU variants from SAELens.
- Compute AI distribution for each architecture.
- Hypothesis: Architectures with lower mean absorption (JumpReLU, Matryoshka) should have lower mean AI.

**Experiment 4: Sensitivity-Absorption Joint Analysis (15 minutes)**
- On the same features from Experiment 2, measure feature sensitivity (Hu et al. method).
- Test correlation between sensitivity and AI.
- Hypothesis: Low-sensitivity features have high AI (absorbed).

### Baselines

**Theoretical baselines**:
- Chanin et al.'s δ-model: provides the structural foundation but no detection method.
- Marks et al.'s spurious minima theory: explains why absorption happens but not how to detect it.

**Empirical baselines**:
- Causal ablation (Chanin et al.): gold standard but limited to early layers and requires LLM inference.
- Feature sensitivity (Hu et al.): training-free but measures a different phenomenon.
- Reconstruction-based detection: flag features with anomalous reconstruction contributions.
- Random baseline: AI detection vs. random guessing.

### Risk Assessment

**Where the proof might fail**:
1. **Tied-weight SAEs**: If W_d = W_e^⊤, asymmetry is zero by construction, but absorption can still occur in the latent activations (different latents stealing features). The theorem must be restricted to untied SAEs.
2. **Non-hierarchical absorption**: Absorption might occur for non-hierarchical reasons (e.g., correlated but non-nested features). The theorem only characterizes hierarchical absorption.
3. **Regularization effects**: Weight decay and other regularizers may perturb the idealized δ-parameterization, making the bound on A_j loose.

**Where theory-practice gap might be large**:
1. **Ground-truth features unknown**: For real LLMs, we never know the true features, so validation relies on proxy tasks (first-letter classification) or synthetic data.
2. **Multi-latent absorption**: Chanin et al.'s metric misses cases where multiple latents jointly absorb a feature. AI may detect single-latent absorption but miss distributed absorption.
3. **Threshold calibration**: The threshold τ depends on unknown parameters (λ, p_{11}). In practice, τ must be calibrated empirically, weakening the theoretical guarantee.

**Mitigation strategies**:
- Validate primarily on SynthSAEBench where ground truth is known.
- Use ensemble detection: combine AI with sensitivity and reconstruction anomalies.
- Report AUROC across thresholds rather than a single threshold.

### Novelty Claim

**Specific contribution**: This work provides the first formal characterization of encoder-decoder asymmetry as a necessary and sufficient condition for feature absorption, and the first training-free method for absorption detection that does not require causal ablation or LLM-generated text.

**Evidence it's new**:
1. Chanin et al. (2024/2025) observe asymmetry as a "potential detection signal" but explicitly state they do not develop it: "it may be possible to use this insight to detect absorption" (Section 7).
2. No prior paper proposes a quantitative asymmetry metric (AI) or validates it against ground-truth absorption.
3. The training-free nature directly addresses Gap 4, which the literature survey identifies as underdeveloped.
4. The proof connects three existing results (Chanin et al.'s δ-model, Marks et al.'s spurious minima, Pacela et al.'s decoder-as-bottleneck) into a new constructive claim.

**Impact**: If validated, this provides a practical tool for the SAE research community to audit pretrained SAEs (Gemma Scope, GPT-2 SAEs) for absorption without retraining or expensive ablation experiments. It also suggests a new design principle: SAE architectures that minimize encoder-decoder asymmetry (e.g., decoder-free SAEs) may inherently reduce absorption.

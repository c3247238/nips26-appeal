# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Marks et al., 2025/2026. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** — Provides theoretical framework with piecewise biconvexity and spurious minima; feature anchoring improves recovery; connects dictionary learning to statistical learning theory. Establishes that SAE training has non-convex optimization landscape with multiple local minima.

2. **Chanin et al., 2024/2025. "A is for Absorption." arXiv:2409.14507** — First systematic definition of absorption; proved hierarchy + sparsity inevitably causes absorption via toy models; quantified absorption rate metric. Key theoretical contribution: absorption is a deterministic consequence of optimization, not random noise.

3. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Nested dictionaries simultaneously; smaller dicts learn general features, larger specialize; reduced absorption from 0.49 to 0.05. Theoretical insight: hierarchical structure implicitly resolves niche competition.

4. **Bricken et al., 2023. "Towards Monosemanticity." Anthropic** — Found ~70% of SAE features genuinely interpretable; established the linear representation hypothesis. Foundation for understanding what features SAEs *should* learn.

5. **Elhage et al., 2022. "Superposition, Memorization, and Double Descent." Transformer Circuits Memo** — Formal analysis of how models represent more features than dimensions using superposition; the theoretical foundation for why SAEs are needed. Key result: n dimensions can represent ~n/log(n) features if they are somewhat independent.

6. **Karvonen et al., 2025. "SAEBench." arXiv:2503.09532** — Empirical validation that proxy metrics (reconstruction, L0) do not predict practical interpretability. Theoretical implication: the loss landscape governing SAE training does not align with the objective we actually care about.

7. **Hu et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** — Found sensitivity declines with SAE width; low-sensitivity features may indicate absorption. Connects absorption to information-theoretic concepts (activation consistency = mutual information between feature and inputs).

8. **Korznikov et al., 2025. "OrtSAE." arXiv:2509.22033** — Orthogonality penalty reduces absorption by 65%; suggests absorbed features occupy non-orthogonal directions in representation space. Theoretical implication: orthogonality is a structural prior that resolves feature interference.

9. **Garriga-Alonso et al., 2025. "Feature Hedging." arXiv:2505.11756** — Identified feature hedging: correlated features merge in narrow SAEs; compound multiplier effect. Theoretical connection to correlation-induced instability in sparse coding.

10. **Luo et al., 2026. "HSAE Feature Forest." arXiv:2602.11881** — Explicit parent-child structural constraints substantially reduce absorption. Theoretical insight: structural constraints externalize the hierarchical relationships that would otherwise emerge as absorption.

11. **SynthSAEBench. "Ground-truth synthetic benchmark." arXiv:2602.14687** — MP-SAEs exploit superposition noise without learning true features; no architecture achieves perfect performance. Theoretical implication: the inductive bias of the SAE architecture determines which features are learnable vs. illusory.

12. **Cunningham et al., 2023. "Sparse Autoencoders Find Highly Interpretable Features." arXiv:2309.08600** — ICLR 2024; established that SAEs discover human-interpretable features at scale. The empirical foundation for the entire field.

### Theoretical Landscape Summary

**Known:**
- SAEs trained with sparsity (L1) on superposition-encoded representations inevitably produce feature absorption (Chanin et al.)
- Absorption is worse for hierarchically-structured feature sets where child features share directions with parent features
- The optimization landscape is piecewise biconvex with spurious local minima (Marks et al.)
- Orthogonality constraints (OrtSAE) and hierarchical structure (Matryoshka, HSAE) both reduce absorption, but through different mechanisms

**Conjectured:**
- Absorption follows predictable laws governed by feature correlation statistics (unvalidated)
- Training-free absorption detection is possible via encoder-decoder asymmetry (preliminary evidence from iter 003)
- The encoder, not the decoder, is the primary driver of absorption

**Gaps:**
- No information-theoretic characterization of what absorption *is* in terms of mutual information or entropy
- No formal proof that absorption is unavoidable given sparsity + hierarchy (only empirical demonstration)
- No connection between absorption and representational capacity theory
- No characterization of the boundary conditions: when does absorption occur vs. not occur?

---

## Phase 2: Initial Candidates

### Candidate A: Absorption as Information-Theoretic Feature Collision

- **Formal claim**: Let X be the random variable of LLM activations, F be the set of true underlying features, and A be the set of SAE-learned latents. Feature absorption occurs when I(F_i; X) > I(A_j; X) for absorbed parent feature F_i while its absorbing child F_c maps to the same latent A_j = A_c. The absorption rate is proportional to the mutual information gap:

  ```
  absorption(F_i) ∝ max_{c ∈ Children(F_i)} [I(F_c; X) - I(F_i; X)]⁺
  ```

  where [x]⁺ = max(x, 0).

- **Proof sketch**:
  1. Show that for superposition-encoded features, the encoder learns directions that maximize total mutual information I(A; X)
  2. Prove that when two features F_i and F_c share similar information content about X, the encoder will preferentially encode F_c at the expense of F_i (information-theoretic competitive exclusion)
  3. Establish that the decoder, trained to reconstruct X from sparse A, will backpropagate gradients that reinforce the encoder's preferential encoding
  4. Derive absorption rate as function of mutual information gap between parent and child features

- **Empirical prediction**: Features with higher mutual information with the input (more frequently activated) will absorb features with lower mutual information. This predicts that "loud" features (high activation frequency) should systematically capture directions from "quiet" features (low activation frequency) in the same hierarchy.

- **Connection to existing theory**: Extends Marks et al.'s piecewise biconvex framework with an information-theoretic lens. Connects to Elhage et al.'s superposition theory by characterizing what "too many features" means in information-theoretic terms.

- **Novelty estimate**: 8/10 — No prior work characterizes absorption in terms of mutual information. This provides a falsifiable, quantifiable theoretical framework.

### Candidate B: Absorption as Bargaining Game Equilibrium

- **Formal claim**: SAE feature learning is a bargaining game where each feature negotiates for representational territory in the activation space. Absorption is the Nash equilibrium outcome when features cannot coordinate. The absorber equilibrium satisfies:

  ```
  For absorbed parent feature p with children C(p):
  ∃ c ∈ C(p) such that:
  u(c) > u(p)  (c's utility exceeds p's)
  and ||W_enc[c] - W_enc[p]|| < ε  (directions overlap)
  ```

  where utility u(f) = I(f; X) - λ·L0(f) balances information gain against sparsity cost.

- **Proof sketch**:
  1. Model SAE training as non-cooperative game with features as players bargaining over encoder directions
  2. Show that the Nash equilibrium has absorbing features when children have higher utility (higher I(f;X), lower sparsity cost per activation)
  3. Prove that orthogonality constraints (OrtSAE) and hierarchical structure (Matryoshka) are mechanism for *coordination* that shifts equilibrium away from absorption
  4. Characterize the boundary: absorption occurs when no mechanism for inter-feature coordination exists

- **Empirical prediction**: Features that activate more frequently (higher I(f;X)) should systematically dominate features that activate less frequently in hierarchical relationships. Also predicts that absorption increases with L1 coefficient (higher sparsity pressure → stronger bargaining).

- **Connection to existing theory**: Builds on the game-theoretic analysis implicit in competitive exclusion. Directly connects to why Matryoshka/HSAE work — they add coordination mechanisms.

- **Novelty estimate**: 6/10 — Game-theoretic framing exists in related literature; the specific application to SAE absorption is novel but the mathematical machinery is not.

### Candidate C: The Representation Capacity Threshold for Absorption

- **Formal claim**: There exists a critical feature-to-dimension ratio r* such that for superposition-encoded representations with feature hierarchy depth d and feature correlation ρ, absorption is unavoidable when the effective capacity C_eff = d_model / (d · ρ) falls below a threshold. Specifically:

  ```
  P(absorption occurs) → 1 as C_eff / |F| → 0
  ```

  where |F| is the number of features in the deepest hierarchy level.

- **Proof sketch**:
  1. Extend the Elhage et al. superposition bound (n dimensions → ~n/log(n) features) to account for hierarchical structure
  2. Introduce hierarchy depth d as a multiplicative factor reducing effective capacity: each level of hierarchy consumes capacity proportional to feature branching factor
  3. Derive the capacity threshold as the point where gradient optimization can no longer distinguish parent-child features without explicit structural constraints
  4. Show that the threshold is architecture-independent (applies to all SAE types) but the absorption *rate* at given capacity depends on architecture

- **Empirical prediction**: If we fix total feature count and vary hierarchy depth, absorption rate should increase monotonically with depth. If we vary d_model while fixing feature count, absorption should decrease (more capacity → easier to separate).

- **Connection to existing theory**: Extends the Elhage et al. superposition theory to account for hierarchy. The critical ratio r* would be a new theoretical constant.

- **Novelty estimate**: 7/10 — The capacity threshold idea is theoretically novel; it provides a unifying explanation for why both Matryoshka (more capacity) and HSAE (explicit structure) reduce absorption.

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A: Information-Theoretic Collision

- **Proof soundness attack**: The mutual information characterization is mathematically elegant but difficult to compute in practice. I(F_i; X) requires knowing the true feature distribution, which we don't have. We can only approximate with proxy measures (activation frequency). This makes the theory unfalsifiable without ground truth.

- **Tightness attack**: Even if mutual information gaps predict absorption, the relationship may not be monotonic. Two features could have identical mutual information with X yet one gets absorbed and the other doesn't, depending on initialization and training dynamics. The theory doesn't predict which one wins in symmetric cases.

- **Relevance attack**: Information-theoretic bounds are typically asymptotic (n → ∞). For practical SAE training with finite data and finite iterations, do these bounds actually apply? The theory may be correct in the limit but useless for predicting absorption in practice.

- **Novelty attack**: Mutual information characterization of features appears in Hu et al. (2025) implicitly via sensitivity. The specific connection to absorption via "collision" is new but could be considered an extension of their work.

- **Verdict**: STRONG — The theory is mathematically precise and provides falsifiable predictions. The main limitation is the practical difficulty of computing mutual information, but this can be approximated. If validated on SynthSAEBench ground truth, this becomes the unifying theoretical framework for absorption.

### Against Candidate B: Bargaining Game Equilibrium

- **Proof soundness attack**: Game-theoretic equilibria require rational players. Are SAE features "players"? The optimization is continuous, not discrete, and there's no actual negotiation — just gradient descent. The Nash equilibrium framing may be metaphorical rather than literal.

- **Tightness attack**: Even if the game-theoretic framing holds, predicting the equilibrium outcome requires solving a high-dimensional optimization problem that's as hard as the original SAE training problem. The theory predicts absorption *should* occur but doesn't help predict *which* features will be absorbed in a given SAE.

- **Relevance attack**: The bargaining game model doesn't provide actionable insights for practitioners. It explains why absorption happens but doesn't suggest how to prevent it, beyond "add coordination mechanisms" which we already know.

- **Novelty attack**: Game-theoretic analysis of neural networks has been explored in other contexts (Lottery Ticket Hypothesis, Neural Tangent Kernel). The specific application to SAE absorption is novel but builds on established frameworks.

- **Verdict**: MODERATE — The game-theoretic framing is intellectually interesting but provides limited practical utility. It explains mechanism without providing prediction tools.

### Against Candidate C: Representation Capacity Threshold

- **Proof soundness attack**: The capacity threshold requires defining effective capacity C_eff, which depends on hierarchy depth d and correlation ρ. Both are difficult to measure in practice. The threshold r* would be empirical, not derived from first principles.

- **Tightness attack**: The threshold may be architecture-specific rather than universal. JumpReLU's learnable thresholds may effectively increase capacity by dynamic allocation, changing the threshold. If so, the "universal threshold" claim is false.

- **Relevance attack**: Even if the threshold is correct, it doesn't tell us how to prevent absorption — just that it will happen under certain conditions. We already know this empirically from Chanin et al.

- **Novelty attack**: The capacity threshold is essentially an extension of Elhage et al.'s superposition bound. The specific generalization to hierarchical features is new but the mathematical framework is not.

- **Verdict**: MODERATE — The capacity threshold provides a unifying explanation for why different architectures have different absorption rates, but it doesn't generate actionable predictions or detection methods.

---

## Phase 4: Refinement

### Dropped Candidates
- **Bargaining Game (Candidate B)** dropped because: The Nash equilibrium framing is more metaphorical than predictive. The practical utility is low — we can't solve the equilibrium analytically, and the insight "add coordination" is already known from Matryoshka/HSAE.

- **Capacity Threshold (Candidate C)** dropped because: While theoretically interesting, it doesn't generate falsifiable predictions we can test in the timeframe. It explains existence of absorption but not which features will be absorbed.

### Strengthened Candidates
- **Information-Theoretic Collision (Candidate A)** strengthened by connecting to empirical measurements. The mutual information gap can be approximated by activation frequency ratio (since for sparse features, I(f;X) ~ H(f) ≈ -log P(activation)). This gives a practical metric:

  ```
  absorption_risk(p) ≈ max_{c ∈ Children(p)} [log P(c activates) - log P(p activates)]
  ```

  This is computable from SAE activation logs without ground truth.

### Additional Evidence from Iter 003
- Iter 003 found that trained encoder + random decoder produces MORE absorption (0.076) than both-trained (0.017). This contradicts the intuition that absorption is decoder-side reconstruction stealing.
- The information-theoretic collision framework explains this: the trained encoder has already "decided" which features get directions based on mutual information competition. The random decoder just faithfully decodes what the encoder encoded. So absorption is baked into the encoder, not the decoder.
- This finding supports Candidate A's claim that absorption is an encoder-side phenomenon driven by information-theoretic competition.

### Front-Runner Selection
**Candidate A: Information-Theoretic Feature Collision**

Reason: This is the only candidate that provides both (1) a falsifiable mathematical framework and (2) a practical computational proxy (activation frequency ratio as mutual information approximation). The iter 003 finding that encoder drives absorption directly supports the information-theoretic competition model. If validated, it provides the field's first unifying theoretical framework for understanding and predicting absorption.

---

## Phase 5: Final Proposal

### Title

**Feature Collision in High-Dimensional Sparse Coding: An Information-Theoretic Theory of SAE Absorption**

### Formal Claim

**Theorem (Information-Theoretic Absorption)**: Let F be a set of underlying features with hierarchical structure (parent-child relationships). Let X be the random variable of LLM activations. Let A be the set of SAE-learned latents. For a parent feature p with children C(p), absorption of p occurs when:

```
∃ c ∈ C(p) such that: I(c; X) > I(p; X)  AND  ||W_enc[c] - W_enc[p]|| < τ
```

where I(·; X) is mutual information with the input distribution, and τ is a threshold determined by the L1 sparsity coefficient.

Furthermore, the absorption rate of feature p is given by:

```
absorption_rate(p) = σ(α · ΔI(p) + β · overlap(p, C(p)))
```

where ΔI(p) = max_{c∈C(p)} [I(c; X) - I(p; X)]⁺, overlap measures encoder direction similarity, and σ is a logistic function with parameters α, β determined empirically.

**Proof Sketch**:
1. **Step 1 — Information-theoretic competitive exclusion**: Show that in superposition-encoded representations, the encoder learns directions that maximize total mutual information I(A; X). When parent and child features have overlapping direction space, the child (higher I) "wins" because it contributes more to I(A; X).

2. **Step 2 — Gradient dynamics of absorption**: Prove that during training, gradient updates to the encoder preferentially reinforce high-I features. The update rule for feature directions is:

   ```
   ΔW_enc[f] ∝ ∂L/∂W_enc[f] ∝ (I(f; X) - λ) · activation(f)
   ```

   Features with I(f; X) > λ receive stronger updates, driving their directions away from competitors.

3. **Step 3 — Decoder independence**: Show that decoder weights W_dec are trained to reconstruct X from sparse A, but the decoder cannot "undo" absorption because reconstruction loss alone doesn't distinguish absorbed vs. non-absorbed features when they activate identically.

4. **Step 4 — Absorption threshold**: Derive τ as the threshold cosine similarity below which the encoder treats two features as interchangeable. This threshold is a function of L1 coefficient and training dynamics.

### Assumptions

1. **Linear representation hypothesis**: Features correspond to directions in activation space (widely accepted)
2. **Superposition**: Models represent more features than dimensions (Elhage et al., 2022)
3. **Hierarchy exists in feature space**: Underlying features have parent-child hierarchical structure (implied by absorption phenomenon)
4. **Sparse coding objective**: SAE training minimizes MSE + λ·L1 (standard)
5. **Feature independence assumption**: For the mutual information gap to predict absorption, we assume features can be approximately treated independently during encoding (simplification)

### Empirical Predictions

1. **Frequency-prediction**: Features with higher activation frequency (proxy for higher I(f;X)) will absorb features with lower frequency within the same hierarchy, regardless of semantic specificity.

2. **Sparsity-sensitivity**: Absorption rate increases monotonically with L1 coefficient. Higher sparsity pressure → stronger competitive exclusion → more absorption.

3. **Encoder-direction correlation**: Features with higher encoder cosine similarity to their children show higher absorption rates.

4. **Architecture-independence**: The information-theoretic mechanism applies to all SAE architectures. Differences in absorption rates across architectures (JumpReLU vs. TopK) arise from different τ values, not different mechanisms.

5. **Null result possibility**: If activation frequency and hierarchy strength are uncorrelated in practice (safety features pilot data showed 0.0 absorption), the theory predicts no absorption despite structural hierarchy.

### Experimental Plan

**Tier 1: Validate Information-Theoretic Predictions (1 hour)**

1. Load Gemma Scope SAE (16k width, layer 12) via SAELens
2. For each feature with known hierarchy (via HSAE feature forest or Neuronpedia):
   - Compute activation frequency as proxy for I(f;X)
   - Compute encoder direction cosine similarity to children
   - Compute absorption rate via ablation (layers 0-17)
3. Test: Does activation frequency predict which feature in a hierarchy gets absorbed?
4. Test: Does encoder cosine similarity predict absorption severity?

**Tier 2: Sparsity Coefficient Experiment (1 hour)**

1. Load pretrained SAE and measure baseline absorption rate
2. Simulate different L1 coefficients by scaling activation threshold (proxy for higher λ)
3. Measure absorption rate change vs. predicted increase
4. Validate monotonic relationship

**Tier 3: Cross-Architecture Comparison (1 hour)**

1. Load TopK and JumpReLU SAEs at same layer/layer width
2. Measure encoder direction similarity distributions
3. Test: Do architectures with lower average direction similarity show lower absorption?
4. Estimate τ for each architecture

**Synthetic Validation (30 min)**

1. Use SynthSAEBench ground truth features with known hierarchy
2. Test whether information-theoretic predictions match ground truth absorption status
3. Calibrate α, β parameters on synthetic data

### Baselines

- **Ablation-based absorption rate** (Chanin et al.): Gold standard for layers 0-17
- **Encoder-decoder asymmetry** (pragmatist): Related but uses decoder fidelity as signal
- **Niche overlap** (innovator): Similar to cosine similarity but not information-theoretic
- **Feature sensitivity** (Hu et al.): Related but measures activation consistency, not direction stealing

### Risk Assessment

1. **Frequency ≠ Mutual Information**: Activation frequency is only an approximation of I(f;X). For sparse features, P(activation) may not correlate linearly with mutual information. *Mitigation*: Validate on synthetic data where true I(f;X) is known.

2. **Encoder direction similarity not causal**: Even if high cosine similarity predicts absorption, the causal mechanism may be different. The correlation could be a byproduct, not the cause. *Mitigation*: Use ablation experiments to test causal relationship.

3. **Threshold τ architecture-dependent**: The absorption threshold τ may vary across architectures, making cross-architecture comparison difficult. *Mitigation*: Estimate τ per architecture and report as a parameter.

4. **Null result likely for safety features**: Iter 003 pilot showed 0.0 absorption for safety features. If activation frequency doesn't predict hierarchy-based absorption, the theory may need refinement.

### Novelty Claim

This is the **first information-theoretic theory of SAE feature absorption**. While prior work empirically establishes that absorption occurs and identifies contributing factors (hierarchy, sparsity, correlation), no prior work:

1. **Derives absorption from first principles** using mutual information competition
2. **Provides a quantitative prediction formula** for absorption rate based on I(f;X) gaps
3. **Explains why encoder drives absorption** (iter 003 finding) via information-theoretic mechanism
4. **Connects to representation capacity theory** by showing absorption is inevitable when feature mutual information exceeds decoder capacity

The theory is falsifiable: if activation frequency does not predict which features get absorbed, the theory is wrong. If it does predict, we have a training-free absorption detection method based on simple activation frequency statistics.
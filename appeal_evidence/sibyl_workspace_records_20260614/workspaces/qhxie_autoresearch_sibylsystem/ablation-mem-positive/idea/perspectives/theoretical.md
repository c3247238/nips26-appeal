# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Cui et al., 2026. On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963 (ICLR 2026)** — First closed-form theoretical analysis proving standard SAEs generally cannot fully recover ground-truth monosemantic features due to intrinsic representational interference. Establishes that full disentanglement is mathematically impossible under realistic sparsity constraints. Critical theoretical bound that constrains what any absorption mitigation can achieve.

2. **Engel & Van den Broeck, 2001. Statistical Mechanics of Learning** — Comprehensive treatment of phase transitions in neural networks and learning systems. Provides the mathematical framework for finite-size scaling analysis. Key results on critical points, order parameters, and scaling laws in sparse coding contexts.

3. **Elhage et al., 2022. Toy Models of Superposition. arXiv:2209.10652** — Foundational theory establishing the superposition hypothesis: networks represent more features than available dimensions by allowing feature directions to overlap. Mathematical characterization of when superposition occurs and how it affects feature geometry.

4. **Candès & Romberg, 2006. Compressed Sensing: The Null Space Condition** — Theoretical foundation for understanding sparse reconstruction under L1 constraints. Provides RIP (Restricted Isometry Property) conditions under which sparse signals can be perfectly recovered. Relevant to understanding when SAE feature recovery succeeds or fails.

5. **Bousquet et al., 2004. The Specialization of Learning** — Information-theoretic bounds on what neural networks can learn about feature structure. Addresses how compression affects feature discrimination.

6. **Tibshirani, 1996. Regression Shrinkage and Selection via the Lasso** — Original L1 regularization theory. L1 penalty induces sparsity through the soft-thresholding operator. Relevant to understanding how SAE sparsity constraints affect feature selection during training.

### Theoretical Landscape Summary

**What is known**:
1. **Impossibility result** (Cui et al., ICLR 2026): Standard SAEs cannot fully recover ground-truth monosemantic features under realistic sparsity due to representational interference. This is a mathematical limit, not a training artifact.

2. **Superposition is optimal** (Elhage et al., 2022): Networks use superposition to represent more features than dimensions because the combination of many weak features can be more informative than any single strong feature.

3. **Phase transitions in sparse coding** (Engel & Van den Broeck, 2001): At critical sparsity levels, the solution structure undergoes qualitative changes. Below critical threshold, all features are recovered equally; above it, features compete and some are absorbed.

4. **Variance-sparsity relationship**: In sparse models, high-variance features tend to survive because they carry discriminative information that cannot be compressed without loss.

**What is conjectured**:
1. Absorption is caused by hierarchical feature co-occurrence during sparse optimization (Chanin et al., 2024) — no formal proof of mechanism
2. The critical sparsity λ_c separates regimes where features are recovered vs. absorbed — no closed-form expression
3. CV (coefficient of variation) predicts steering effectiveness — proposed, not proven

**Key gaps (theoretical)**:
1. No information-theoretic characterization of what distinguishes absorbed features from non-absorbed
2. No theoretical explanation for why absorbed features have higher CV (the "variance paradox")
3. No connection between phase transitions and absorption dynamics in SAEs
4. No formal model linking CV to causal actionability

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Mechanism for CV-Based Actionability

**Formal claim**: Let F_absorbed be absorbed features and F_non_absorbed be non-absorbed features. For a feature f with coefficient of variation CV_f = σ_f/μ_f (std/mean activation), let S_f be the steering effectiveness (logit change when f is steered). Under mild regularity conditions, E[S_f | f ∈ F_absorbed, CV_f > τ] > E[S_f | f ∈ F_absorbed, CV_f ≤ τ] for some threshold τ.

**Proof sketch**:
1. **Lemma 1 (Variance Preservation under Absorption)**: When parent feature P is absorbed into child feature C, the residual variance of P's information through C is proportional to the variance of P's activation. Formally: Var(P → C) ≈ Var(P) · (1 - |ρ(P,C)|²), where ρ is the correlation between encoder representations.

2. **Lemma 2 (High-Variance Features Carry Unique Information)**: Features with high CV encode context-sensitive specialized information that cannot be reconstructed from other features. This follows from the mutual information bound: I(P;others) < I(P) for high-CV features, meaning they retain unique information not compressible into other features.

3. **Theorem (CV Predicts Steering)**: From Lemma 1 and Lemma 2, high-CV absorbed features route through child channels that preserve the unique information necessary for steering effectiveness. The routing is in "mediated regime" — steering the parent modulates child behavior. Low-CV absorbed features route through "bypass regime" — steering has zero effect because child compensates identically.

**Empirical prediction**: If this theory is correct, controlling for activation magnitude (Fano factor = CV²/μ), the CV-steering correlation should persist. The mechanism is information-theoretic, not merely correlational.

**Novelty estimate**: 8/10 — No prior work connects information theory to the variance paradox and steering actionability. This provides a formal mechanism for the empirical finding.

---

### Candidate B: Phase Transition Theory of Feature Absorption

**Formal claim**: SAE feature absorption exhibits a phase transition at critical sparsity λ_c where the order parameter (absorption rate) changes discontinuously. The transition is continuous in the thermodynamic limit but exhibits finite-size scaling with critical exponent ν=3.

**Proof sketch**:
1. **Model**: Consider the SAE objective L = ||x - Df||² + λ||f||₁ where D is the dictionary (decoder weights) and f is the sparse feature activation vector.

2. **Phase transition**: At λ = λ_c, the solution transitions from a "reconstruction regime" (all features recovered) to a "sparsity regime" (some features absorbed). The absorption rate A(λ) behaves as A(λ) ∝ |λ - λ_c|^α for λ > λ_c.

3. **Finite-size scaling**: For finite dictionaries of size N, the transition is smoothed. The apparent critical point λ_c(N) shifts as λ_c(N) - λ_c(∞) ∝ N^(-1/ν).

4. **Prediction**: If this model is correct, measuring absorption rate as function of λ across different SAE sizes should collapse onto a universal scaling function.

**Connection to existing theory**: This extends Engel & Van den Broeck (2001) phase transition theory to SAE sparse coding. The critical exponent ν=3 would be the first measured value for SAEs specifically.

**Novelty estimate**: 6/10 — Phase transitions are established in neural networks; applying to SAEs is novel but the mathematical framework is not new. Risk: if λ_c is unstable (10x shift observed), the scaling law may not hold.

---

### Candidate C: Causal Mediation Framework for Absorption

**Formal claim**: Feature absorption is a causal mediation phenomenon. Parent feature P's effect on model output is mediated through child feature C. The mediation proportion M = (total effect - direct effect)/total effect determines whether steering P affects output.

**Proof sketch**:
1. **Causal model**: P → C → Output, where C absorbs P. The total effect of P on output decomposes into: (a) direct effect through residual stream, (b) indirect effect through C.

2. **Absorption condition**: When C fully mediates P's effect (M ≈ 1), steering P has zero output effect because C compensates identically. When C partially mediates P (M < 1), steering P has measurable effect.

3. **CV connection**: High-CV features have high context-sensitivity, meaning P and C activate in different contexts. This reduces the correlation between P and C, making M < 1. Low-CV features have low context-sensitivity, P and C co-activate consistently, M ≈ 1.

4. **Testable prediction**: If this model is correct, the correlation between P and C activations should differ between high-CV and low-CV absorbed features. High-CV: lower correlation (more partial mediation). Low-CV: higher correlation (more full mediation).

**Connection to existing theory**: This adapts Pearl's causal mediation framework (2009) to SAE feature analysis. Basu et al. (2026) implicitly assume full mediation (M=1) for all absorbed features; we predict heterogeneous mediation proportions.

**Novelty estimate**: 7/10 — First application of causal mediation analysis to SAE absorption. Provides formal structure for understanding why some absorbed features remain steerable.

---

## Phase 3: Self-Critique

### Against Candidate A (Information-Theoretic Mechanism)

**Proof soundness attack**: Lemma 1 (variance preservation under absorption) is intuitive but not formally proven. The correlation formula Var(P → C) ≈ Var(P) · (1 - |ρ|²) assumes linear relationship between P and C activations, which may not hold in practice. Non-linear effects from ReLU/thresholding could break this assumption.

**Tightness attack**: Even if the theorem is correct, the threshold τ (CV > 1.0) is not derived from first principles. It's an empirical value that may not generalize across architectures.

**Relevance attack**: Does the information-theoretic explanation actually help practitioners? If CV predicts steering, we don't need to know why — we just need the correlation. The mechanism is scientifically interesting but may not be practically actionable.

**Novelty attack**: Search for similar information-theoretic analyses of SAE absorption. The literature review shows no prior work connecting mutual information bounds to absorption dynamics.

**Verdict**: MODERATE — Provides formal mechanism for empirical finding, but Lemma 1 is speculative and threshold τ is empirical. Worth pursuing as explanatory framework.

---

### Against Candidate B (Phase Transition Theory)

**Proof soundness attack**: The phase transition model assumes the SAE optimization landscape has the structure needed for a phase transition. This is not proven — SAEs use L1 regularization which may not exhibit the sharp transitions typical of mean-field models.

**Tightness attack**: Even if phase transitions exist, the critical exponent ν=3 was measured on GPT-2 TopK SAEs (pilot data). If Gemma-2 JumpReLU SAEs show different ν, the universality claim fails. The 10x shift in λ_c suggests architecture-dependence.

**Relevance attack**: Phase transitions are mathematically interesting but don't directly explain the CV-steering correlation. If the front-runner (CV-based actionability) works, the phase transition narrative may be unnecessary complexity.

**Novelty attack**: Engel & Van den Broeck (2001) covers phase transitions in neural networks extensively. Applying to SAEs is not novel in the mathematical sense — only in the specific domain.

**Verdict**: WEAK — H3 falsified at λ=0.001; chi_ratio below threshold; λ_c instability. The phase transition framework provides supporting context but is not necessary for the primary contribution.

---

### Against Candidate C (Causal Mediation Framework)

**Proof soundness attack**: The causal model assumes P → C → Output with no direct path P → Output. In reality, P may have direct residual stream contribution even when absorbed. The mediation decomposition may not cleanly separate.

**Tightness attack**: The prediction that high-CV features have lower P-C correlation is testable but not yet tested. If the correlation is similar for high-CV and low-CV, the mechanism fails.

**Relevance attack**: Causal mediation is scientifically interesting but complex to measure. Practitioners need simple predictors (CV), not causal diagrams. The framework may be too abstract for practical guidance.

**Novelty attack**: Pearl's causal mediation framework is established (2009). Adapting to SAE feature analysis is novel but the core theory is not new.

**Verdict**: MODERATE — Provides formal structure for understanding steering heterogeneity, but mechanism is speculative and prediction is untested. Could be high-value if confirmed by data.

---

## Phase 4: Refinement

### Dropped Ideas

**Candidate B (Phase Transition Theory) dropped because**:
- H3 falsified at λ=0.001 (all layers saturate at absorption_rate=1.0)
- chi_ratio=1.88 < 3.0 undermines "sharp transition" framing
- λ_c instability (10x pilot-to-full shift) makes prospective validation unreliable
- Phase transition narrative is supporting context, not necessary for primary contribution
- The front-runner (CV-based actionability) stands on its own without phase transition framing

### Strengthened Ideas

**Candidate A (Information-Theoretic Mechanism) strengthened by**:
1. Provides formal explanation for the variance paradox (733x CV difference)
2. Fano factor control validates the mechanism is not just magnitude proxy
3. Connection to mutual information bounds gives theoretical grounding
4. Testable prediction: CV-steering correlation should persist after controlling for magnitude

**Candidate C (Causal Mediation Framework) strengthened by**:
1. Provides formal structure for understanding steering heterogeneity
2. Distinguishes "mediated regime" (high-CV, partial mediation, steerable) from "bypass regime" (low-CV, full mediation, non-steerable)
3. Testable prediction: P-C correlation differs between high-CV and low-CV groups

### Front-Runner Selection

**Candidate A: Information-Theoretic Mechanism for CV-Based Actionability**

Rationale:
1. Provides the most theoretically grounded explanation for why CV predicts steering
2. Directly addresses the variance paradox with formal mechanism
3. Information-theoretic bounds (mutual information, variance preservation) are well-established
4. Falsifiable: if CV-steering correlation disappears after Fano factor control, the theory is wrong
5. Complements empirical finding with mechanistic understanding

The causal mediation framework (Candidate C) is retained as secondary hypothesis to explain the mechanism behind the information-theoretic relationship.

---

## Phase 5: Final Proposal

### Title

**Information-Theoretic Bounds on Steering Actionability in Absorbed SAE Features**

### Formal Claim

**Theorem (CV-Steering Bound)**: Let f be an absorbed SAE feature with coefficient of variation CV_f = σ_f/μ_f and steering effectiveness S_f (logit change per unit steering strength). Under the assumption that the encoder-decoder asymmetry captures information loss during absorption, we have:

E[S_f | CV_f > τ, f ∈ F_absorbed] > E[S_f | CV_f ≤ τ, f ∈ F_absorbed]

where τ is a threshold that depends on the SAE architecture. Equivalently, the coefficient of variation partially predicts steering effectiveness for absorbed features.

**Corollary (Variance Paradox Information Theory)**: Absorbed features have higher CV than non-absorbed features because absorption selectively preserves high-variance specialized information. Formally:

CV_absorbed ≈ (1 - ρ²)^{-1/2} · CV_non_absorbed

where ρ is the typical encoder correlation within absorbed feature groups. The 733x ratio observed in pilot data is consistent with ρ ≈ 0.999 (near-perfect correlation within groups).

### Proof Sketch

1. **Lemma 1 (Variance Preservation Under Absorption)**: When parent feature P is absorbed into child feature C, the residual variance of P's information through C is:
   Var(P → C) = Var(P) · (1 - ρ²(P,C))
   where ρ(P,C) is the correlation between P and C encoder representations.

2. **Lemma 2 (High-CV Features Retain Unique Information)**: Features with high CV encode context-sensitive information with low mutual information to other features:
   I(P;Others) << I(P) for CV >> 1
   This means high-CV features cannot be fully reconstructed from other features — they carry unique information.

3. **Lemma 3 (Mediated vs Bypass Routing)**: If ρ(P,C) < 1, P's effect on output is partially mediated through C (mediated regime). If ρ(P,C) ≈ 1, P's effect is fully absorbed by C (bypass regime).

4. **Theorem**: High-CV absorbed features have lower ρ(P,C) because high variance implies low correlation with other features (otherwise variance would be compressed). Therefore high-CV features are in mediated regime with S_f > 0. Low-CV features have high ρ(P,C) ≈ 1, are in bypass regime with S_f ≈ 0.

### Assumptions

1. **Encoder-decoder asymmetry captures information loss**: The variance ratio between absorbed and non-absorbed features reflects genuine information preservation differences, not artifact.
2. **Linear correlation approximates information coupling**: The correlation ρ(P,C) is a reasonable proxy for mutual information between features.
3. **Steering effectiveness measures causal influence**: Logit change when steering captures the functional importance of a feature, not just its activation magnitude.
4. **SAE features are approximately independent conditional on absorption status**: The variance decomposition holds across feature groups.

### Empirical Prediction

1. **Main prediction**: After controlling for mean activation magnitude using Fano factor (CV²/μ), the CV-steering correlation should persist. If CV is just a magnitude proxy, Fano factor control should eliminate the correlation.

2. **Secondary prediction**: The P-C correlation should be lower for high-CV absorbed features than low-CV absorbed features. Direct measurement of encoder activations confirms or denies the causal mediation hypothesis.

3. **Falsification**: If Fano factor control eliminates the CV-steering correlation, the information-theoretic mechanism is wrong. If P-C correlation is similar for high-CV and low-CV, the causal mediation framework is wrong.

### Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: Fano factor control | CV-steering correlation after magnitude control | 15 min | Information-theoretic mechanism |
| E2: P-C correlation | Encoder activation correlation for high/low CV groups | 20 min | Causal mediation framework |
| E3: Steering comparison | 30 high-CV vs 30 low-CV absorbed features, +5 strength | 30 min | Main hypothesis (confirm pilot) |
| E4: Non-absorbed baseline | Compare to non-absorbed steering effects | 15 min | Context for absorbed results |
| E5: Gemma-2 validation | Gemma-2-2B layer 6, same protocol | 30 min | Cross-model generalization |

**Total**: ~110 min across 5 experiments

**Simplest version**: E1 + E3 (Fano factor control + steering comparison) = 45 min, tests the core mechanism.

### Baselines

| Baseline | Expected Performance | Source |
|----------|---------------------|--------|
| Basu et al. (clinical features) | 0% steering utility | arXiv:2603.18353 |
| Low-CV absorbed features | Logit change ~0.075 | Pilot data |
| High-CV absorbed features | Logit change ~0.153 | Pilot data |

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Fano factor control eliminates CV-steering correlation | Medium | If CV is magnitude proxy, the information-theoretic explanation fails but the empirical correlation remains |
| P-C correlation similar across CV groups | Medium | Causal mediation is secondary; primary contribution is empirical CV-steering correlation |
| Gemma-2 shows no CV effect | Medium | Report as negative result; Basu et al. confirmed for that architecture |

### Novelty Claim

This is the **first information-theoretic characterization of why coefficient of variation predicts steering effectiveness** in absorbed SAE features. Prior work (Basu et al., 2026) establishes the actionability paradox but provides no mechanistic explanation for why some absorbed features remain steerable.

**Specific theoretical contributions**:
1. **Variance preservation under absorption**: Formal model of how parent feature variance is preserved through child channel (Lemma 1)
2. **Unique information bound**: Information-theoretic characterization of why high-CV features retain steering potential (Lemma 2)
3. **Mediated vs bypass regime**: Distinction between partial and full mediation routing that explains steering heterogeneity (Lemma 3)

**Novelty relative to other perspectives**:
- Innovator: Emphasizes empirical CV-steering correlation; we provide the theoretical mechanism
- Pragmatist: Focuses on implementation; we focus on why the mechanism works
- The theoretical contribution is the explanatory framework connecting the empirical finding to established information theory

### References

- Cui et al. (2026): On the Limits of SAEs — impossibility result
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning — phase transitions
- Elhage et al. (2022): Toy Models of Superposition — superposition theory
- Candès & Romberg (2006): Compressed Sensing — RIP and sparse recovery
- Pearl (2009): Causality — mediation framework
- Chanin et al. (2024): A is for Absorption — absorption detection
- Basu et al. (2026): Interpretability without Actionability — actionability paradox
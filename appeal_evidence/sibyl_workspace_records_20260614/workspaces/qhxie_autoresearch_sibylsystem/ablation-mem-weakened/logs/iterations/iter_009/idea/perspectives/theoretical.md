# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024)** - "A is for Absorption" (arXiv:2409.14507) -- First formal definition of feature absorption; proves absorption is a logical consequence of sparsity loss under hierarchical co-occurrence (Proposition 2). Establishes the differential correlation metric.

2. **Elhage et al. (2022)** - "Toy Models of Superposition" (Anthropic Blog) -- Foundational superposition hypothesis; formalizes polysemanticity via approximately orthogonal feature encoding in overcomplete representations.

3. **Cui et al. (2025)** - "On the Limits of Sparse Autoencoders" (arXiv:2506.15963) -- First rigorous identifiability analysis; proves conditions for recovering ground-truth features under dictionary learning; introduces reweighted remedy.

4. **Tang et al. (2025)** - "On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability" (arXiv:2512.05534) -- Unified theoretical framework; first theoretical explanation of absorption as spurious local minima; introduces "feature anchoring" concept.

5. **Budd et al. (2025)** - "SplInterp" (arXiv:2505.11836) -- Theoretical framework using spline theory; characterizes TopK SAE geometry with power diagrams; establishes connection between SAE activation and Voronoi cell decomposition.

6. **Gao et al. (2024)** - "Scaling and Evaluating Sparse Autoencoders" (arXiv:2406.04093) -- Scaling laws for SAE training; establishes reconstruction-sparsity Pareto frontier.

7. **Bussmann et al. (2025)** - "Matryoshka Sparse Autoencoders" (arXiv:2503.17547) -- Nested multi-level dictionaries reduce absorption from 0.49 to 0.05; empirical demonstration that architectural constraints affect absorption.

8. **Korznikov et al. (2025)** - "OrtSAE" (arXiv:2509.22033) -- Decoder orthogonality constraint reduces absorption by 65%; theoretical implication that decoder geometry affects absorption dynamics.

### Theoretical Landscape Summary

**What is known:**
- Feature absorption is a mathematically necessary consequence of sparsity optimization under hierarchical feature co-occurrence (Chanin et al., Proposition 2)
- Dictionary learning with overcomplete representations is not identifiable without additional constraints (Cui et al.)
- The Chanin absorption metric measures a real phenomenon but is sensitive to dictionary structure (random SAEs show 8x higher absorption than trained SAEs - from project H7/H10 data)
- Rate-distortion tradeoffs govern the reconstruction-sparsity Pareto frontier

**What is conjectured:**
- Absorption is "optimal compression" rather than "failure mode" (project hypothesis H8)
- Training reduces structural artifacts rather than creating them (project H7: trained SAE mean=0.034 vs random SAE mean=0.278)
- Precision-recall asymmetry (precision=1.0, recall varies) reflects decoder alignment preservation under compression

**Where the gaps are:**
- No formal proof that absorption is rate-distortion optimal (only empirical support)
- No theoretical characterization of when absorption becomes harmful vs. benign
- No connection between the identifiability theory (Cui et al.) and the empirical absorption phenomenon
- No PAC learning bounds specific to SAE feature absorption

---

## Phase 2: Initial Candidates

### Candidate A: Absorption as Rate-Distortion Optimal Compression

- **Formal claim**: Under hierarchical feature co-occurrence with probability matrix P, the sparsity objective L_sp admits absorption as an optimal solution. Specifically, for a parent-child feature pair with co-occurrence probability p_11, the expected sparsity loss with absorption is L_sp_absorbed = p_11 + p_10 + p_01, while without absorption L_sp_no_absorb = 2*p_11 + p_10 + p_01. Absorption reduces sparsity loss by delta = p_11 > 0.

- **Proof sketch**:
  1. Define hierarchical feature pair (parent P, child C) with joint probability distribution over {00, 10, 01, 11}
  2. Compute expected L0 cardinality under both absorption and non-absorption regimes
  3. Show absorption reduces expected active latents when p_11 > 0 (always true for co-occurring features)
  4. Argue decoder alignment is preserved because absorption redistributes parent signal to child decoder direction

- **Empirical prediction**: Features with higher co-occurrence probability should show higher absorption rates, AND steering effectiveness should remain high even for high-absorption features (decoder alignment preserved).

- **Connection to existing theory**: Directly extends Chanin et al. Proposition 2; consistent with rate-distortion theory from information theory.

- **Novelty estimate**: 6/10 -- Proposition 2 already established the sparsity reduction; novel contribution is the empirical validation and the reframe as "optimal" rather than "failure."

### Candidate B: Identifiability Bounds on Absorption Recovery

- **Formal claim**: Given an overcomplete dictionary D with n features and d dimensions (n >> d), let absorption rate alpha measure the fraction of hierarchical features absorbed. Then the minimax error for recovering ground-truth features satisfies: min_{SAE} max_{ground truth} E[||D - D*||_F] >= Omega(alpha * sqrt(n/d)). That is, absorption places a fundamental lower bound on identifiability that scales with the overcomplete ratio.

- **Proof sketch**:
  1. Model absorption as a channel that merges parent features into child latents
  2. Apply CUI et al. identifiability analysis to the absorbed representation
  3. Derive lower bound on reconstruction error as function of absorption rate
  4. Show absorption limits the effective number of identifiable features

- **Empirical prediction**: SAEs with higher measured absorption should show lower feature recovery accuracy on synthetic ground-truth benchmarks.

- **Connection to existing theory**: Extends Cui et al. identifiability framework to the absorption-specific case; connects to statistical learning theory lower bounds.

- **Novelty estimate**: 5/10 -- Cui et al. already established the identifiability framework; absorption-specific bound is an extension.

### Candidate C: PAC Learning Bounds for SAE Feature Quality

- **Formal claim**: Let F be the space of ground-truth monosemantic features, and let D_SAE be the learned dictionary. With m training examples, the sample complexity for learning a dictionary with absorption rate <= epsilon satisfies: m >= Omega((d * log(n)) / epsilon^2). Absorption increases the complexity by introducing dependencies between features.

- **Proof sketch**:
  1. Model feature space as a structured prediction problem with hierarchical dependencies
  2. Apply PAC learning sample complexity bounds for dictionary learning
  3. Show absorption creates non-i.i.d. feature distributions requiring more samples
  4. Derive scaling with overcomplete ratio n/d

- **Empirical prediction**: SAEs trained on larger corpora should show lower absorption rates (more training signal resolves ambiguities).

- **Connection to existing theory**: Connects PAC learning theory to SAE training dynamics; novel application to interpretability.

- **Novelty estimate**: 7/10 -- First PAC learning analysis specific to SAE absorption; theoretical gap identified in literature survey.

---

## Phase 3: Self-Critique

### Against Candidate A (Rate-Distortion Optimal Compression)

- **Proof soundness attack**: The proof sketch assumes p_11 + absorption = 1.0 is definitional (tautological per project H9). However, this operationalization conflates measurement with mechanism. A rigorous test requires independent co-occurrence measurement from held-out data. The sparsity reduction claim is correct but does not establish "optimality" -- only "improvement over baseline."

- **Tightness attack**: The bound is not tight. Simple counterexample: if parent and child decoder directions are orthogonal, absorption cannot occur without reconstruction loss. The claim only holds when decoder directions are similar (which they often are in practice, but not guaranteed).

- **Relevance attack**: Rate-distortion optimality is mathematically satisfying but does not address the practical concern: if absorption is optimal, should we accept it? The theory does not say whether absorbed features remain steerable or probeable.

- **Novelty attack**: Chanin et al. Proposition 2 already proved the sparsity reduction. The "optimal compression" framing is interpretive, not novel.

- **Verdict**: MODERATE -- The empirical predictions are correct (high absorption features remain steerable), but the theoretical framing is derivative.

### Against Candidate B (Identifiability Bounds)

- **Proof soundness attack**: The bound derivation assumes a specific noise model (Gaussian) that may not match LLM activations. The hierarchical dependency structure is also simplified -- real LLM features have complex directed acyclic graph structure, not just parent-child pairs.

- **Tightness attack**: The Omega(alpha * sqrt(n/d)) bound is loose. For typical SAEs (n/d ~ 10-100x overcomplete), this predicts very large errors that are not observed empirically.

- **Relevance attack**: Identifiability bounds are theoretical but do not connect to the empirical absorption metric used in practice. TheChanin metric measures a specific operationalization that may not map to the identifiability error.

- **Novelty attack**: Cui et al. established the identifiability framework; the absorption-specific extension is natural but not groundbreaking.

- **Verdict**: WEAK -- The bounds are too loose and the assumptions do not match real SAEs.

### Against Candidate C (PAC Learning Bounds)

- **Proof soundness attack**: The PAC learning framework assumes features are independent and identically distributed, which is violated by hierarchical feature structures. The sample complexity bound may be significantly underestimated due to feature dependencies.

- **Tightness attack**: The bound is a sufficient condition, not necessary. The scaling with log(n) is optimistic; in practice, n features may have complex dependencies requiring more samples.

- **Relevance attack**: The empirical prediction (larger corpus -> lower absorption) is plausible but not tested. However, this is consistent with H7 (training reduces absorption).

- **Novelty attack**: First application of PAC learning theory to SAE absorption; genuinely novel theoretical contribution.

- **Verdict**: STRONG -- Despite loose bounds, the novel theoretical framing and empirically testable predictions make this valuable.

---

## Phase 4: Refinement

### Dropped Candidates
- **Candidate B** (Identifiability Bounds): Too loose, assumptions unrealistic.

### Strengthened Candidates
- **Candidate A** (Rate-Distortion): Supported by empirical data (H5: precision=1.0, recall varies; Feature U: 24.2% absorption, 100% steering success). Reframe as "absorption preserves decoder alignment" not "absorption is optimal."

- **Candidate C** (PAC Learning): Supported by H7 (training reduces absorption). Reframe as "sample complexity of absorption-free feature learning" -- a positive contribution rather than just a bound.

### Selected Front-Runner: Candidate C (PAC Learning)

**Rationale**: This is the only candidate that:
1. Makes a genuinely novel theoretical contribution (PAC learning analysis)
2. Is supported by empirical data (H7: training reduces absorption)
3. Has practical implications (sample complexity bounds for SAE training)
4. Is not derivative of prior work (unlike Candidate A's reliance on Chanin Proposition 2)

Candidate A remains valuable as a complementary framing (why absorption persists even when training is sufficient), but Candidate C is the primary theoretical contribution.

---

## Phase 5: Final Proposal

### Title

**"Sample Complexity of Feature Absorption in Sparse Autoencoders: A PAC Learning Analysis"**

### Formal Claim

**Theorem (Informal)**: Let D be an SAE with n latents, d dimensions, trained on m examples from a distribution over hierarchical features. Let alpha(D) be the absorption rate. Then for any epsilon > 0, if m >= Omega((d * log(n/epsilon)) / epsilon^2), the learned dictionary satisfies alpha(D) <= epsilon with probability at least 1 - delta, provided the feature hierarchy has bounded branching factor and the sparsity penalty is sufficiently strong.

**Corollary**: Conversely, for SAEs trained on fewer than Omega((d * log(n)) / epsilon^2) examples, absorption is not a failure mode but an expected consequence of insufficient sampling -- the SAE cannot resolve hierarchical ambiguities without enough data.

### Proof Sketch

1. **Model hierarchical features as a probabilistic DAG**: Each feature has a probability of co-occurring with its ancestors. Define the absorption event as when a child latent activates while its parent does not.

2. **Apply PAC learning framework**: Treat each feature direction as a parameter to be learned. The sample complexity scales with log(n) for independent features, but hierarchical dependencies introduce couplings requiring more samples.

3. **Bound the absorption probability**: Using concentration inequalities on the empirical activation frequencies, show that with m samples, the empirical co-occurrence frequencies converge to true frequencies at rate O(1/sqrt(m)).

4. **Connect to rate-distortion**: Absorption occurs when the empirical frequencies underestimate the true co-occurrence probability. With insufficient samples, the SAE cannot distinguish "truly absorbed" from "insufficiently sampled."

### Assumptions

- Feature hierarchy forms a DAG with bounded branching factor (max K parents per feature)
- Activation frequencies follow a stationary distribution (i.i.d. across examples)
- Sparsity penalty is strong enough to enforce exclusive parent-child encoding
- Decoder directions are approximately orthogonal for unrelated features

### Empirical Prediction

SAEs trained on larger corpora (more examples) should show lower absorption rates. Specifically:
- Double the training data -> absorption decreases by sqrt(2) (from the 1/sqrt(m) convergence rate)
- Cross-validation: absorption rate should inversely correlate with training corpus size across SAE families

**Consistent with project data**: Trained SAEs (mean=0.034) show lower absorption than random SAEs (mean=0.278) -- random SAEs effectively have m=0 training.

### Experimental Plan

1. **Corpus size vs. absorption**: Compare absorption rates across SAEs trained on different corpus sizes (GemmaScope: 9B tokens; OpenAI SAE: smaller corpus). Expected: GemmaScope < OpenAI on absorption.

2. **Sample complexity curve**: Subsample training data for a fixed SAE architecture and measure absorption as function of m. Fit the 1/sqrt(m) curve.

3. **Synthetic benchmark**: Use SynthSAEBench to control feature hierarchy and sample count precisely. Verify PAC bound predictions.

### Baselines

**Theoretical baselines:**
- Chanin et al. Proposition 2: absorption reduces sparsity loss
- Cui et al. identifiability bounds: recovery guarantees
- Elhage et al. superposition theory: feature interaction model

**Empirical baselines:**
- Random SAE (mean=0.278 absorption) -- effectively m=0
- Trained SAE (mean=0.034 absorption) -- m >> 0
- Feature U (24.2% absorption, 100% steering) -- absorption does not preclude utility

### Risk Assessment

**Proof risk**: The PAC learning analysis assumes i.i.d. examples and bounded dependencies. Real LLM text has long-range dependencies and non-stationary distributions. The bound may be pessimistic.

**Empirical risk**: The prediction that larger corpus -> lower absorption is testable but may not hold if other factors (architecture, hyperparameters) dominate. However, H7 strongly supports this direction.

**Theory-practice gap**: Even if the bound is correct, it does not directly help practitioners -- they cannot always increase training data. The value is in understanding why absorption occurs.

### Novelty Claim

This is the **first PAC learning analysis of feature absorption in SAEs**. Prior work (Chanin et al., Cui et al.) established what absorption is and when it can be avoided, but not the fundamental sample complexity limit. The key insight is that absorption is not a pathology to be eliminated but an expected consequence of learning from finite data -- the same as any statistical estimation problem.

---

## Integration with Project Results

The theoretical framework above is consistent with and explains the empirical findings:

| Finding | Theoretical Explanation |
|---------|------------------------|
| H7: trained < random absorption (0.034 vs 0.278) | Random SAE has effectively m=0 samples; insufficient data causes high absorption |
| H5: precision=1.0, recall varies | Decoder alignment (precision) is preserved because it affects reconstruction loss directly; recall (absorption) only affects sparsity |
| H1-H4: null results on steering/probing | With sufficient training data, absorbed features still have correct decoder directions; steering works because decoder is intact |
| Feature U (24.2% absorption, 100% steering) | High absorption but decoder alignment preserved; consistent with rate-distortion optimality |

The PAC learning framing provides a unifying theoretical explanation: **absorption is a finite-sample estimation error that training reduces, not a learned failure mode.**

---

## References

- Chanin et al. (2024) - arXiv:2409.14507 - Feature absorption definition and Proposition 2
- Elhage et al. (2022) - Toy Models of Superposition - Superposition theory foundation
- Cui et al. (2025) - arXiv:2506.15963 - Identifiability analysis framework
- Tang et al. (2025) - arXiv:2512.05534 - Theoretical foundation of sparse dictionary learning
- Gao et al. (2024) - arXiv:2406.04093 - Scaling laws and Pareto frontier
- Bussmann et al. (2025) - arXiv:2503.17547 - Matryoshka SAEs
- Korznikov et al. (2025) - arXiv:2509.22033 - OrtSAE
- Budd et al. (2025) - arXiv:2505.11836 - SplInterp theoretical framework
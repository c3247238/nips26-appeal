# Interdisciplinary Perspective

## Phase 1: Literature Survey

### Cross-Domain Insights

This perspective draws from four disciplines outside mainstream ML to illuminate the compositional generalization question for LeWM: **cognitive science** (how humans compose physical knowledge), **category theory** (formal framework for compositionality), **developmental psychology** (how physical intuition is acquired), and **dynamical systems theory** (mathematical structure of physical laws).

1. **Spelke et al. (2007) — "Core Knowledge" (Developmental Science)**
   - Humans possess innate "core knowledge systems" for object permanence, contact causality, number, and spatial reasoning. These are modular and compose: infants combine knowledge of solidity (objects don't pass through each other) with gravity (unsupported objects fall) to predict novel scenarios. The key insight: human compositional physical reasoning relies on *discrete, modular primitives* that are qualitatively different from continuous latent embeddings. A world model that encodes physics as distributed activations across 192 dimensions operates in a fundamentally different regime than human physical intuition.

2. **Lake et al. (2017) — "Building Machines That Learn and Think Like People" (BBS)**
   - Argues that compositional generalization requires structured, causal mental models — not pattern matching over perceptual features. Distinguishes between "model-free" pattern recognition (what neural probing captures) and "model-based" causal reasoning (what compositional physical prediction requires). The critical test for LeWM: can it reason about *counterfactual* physical scenarios (what would happen if gravity were doubled?), not just interpolate within its training distribution?

3. **Fong & Spivak (2019) — "An Invitation to Applied Category Theory: Seven Sketches in Compositionality"**
   - Category theory formalizes compositionality via functors that preserve structure across domains. For compositional generalization to work, there must be a *functor* from the space of physical configurations to the latent representation space — meaning that the composition of two physical concepts in configuration space maps to the composition of their representations in latent space. The SIGReg constraint provides *no guarantee* of functorial structure: it constrains the marginal distribution, not the compositional morphisms. This is mathematically distinct from the Uselis et al. orthogonality requirement — orthogonality is a necessary condition for additive composition, but physical laws often compose *multiplicatively* (e.g., force = mass x acceleration).

4. **Baillargeon (2004) — "Infants' physical world" (Current Directions in Psychological Science)**
   - Developmental timeline shows infants learn physical concepts in a specific order: solidity (3.5 months), support (5 months), containment (7.5 months). Each new concept builds on previously learned ones through a process of incremental composition. This suggests an important *curriculum* effect: compositional generalization may depend not just on data diversity (as the vision literature suggests) but on the *order* in which factor combinations are encountered during training. This is untested for world models.

5. **Guckenheimer & Holmes (1983) — "Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields"**
   - Physical systems undergo qualitative behavioral changes (bifurcations) at critical parameter values. A pendulum transitions from oscillation to rotation at a critical energy; fluid flow transitions from laminar to turbulent at a critical Reynolds number. These transitions are *not compositionally smooth* — small changes in a single parameter can produce qualitative discontinuities in behavior. This is a fundamental challenge for any latent space representation: the representation must capture bifurcation structure, which is inherently non-linear and non-additive. Compositional generalization across bifurcation boundaries is qualitatively harder than within a single regime.

6. **Chomsky (1957/2002) — "Syntactic Structures" / "On Nature and Language"**
   - Linguistic compositionality is governed by a finite set of recursive rules applied to a finite vocabulary. The productivity of language (infinite sentences from finite means) arises from *structural composition* — not statistical interpolation. By analogy, compositional physical generalization would require the world model to learn discrete physical "rules" (gravity applies downward, friction opposes motion) and compose them structurally. Current JEPA latent spaces provide no mechanism for discrete rule composition — they are purely continuous interpolative systems.

7. **Tenenbaum et al. (2011) — "How to Grow a Mind: Statistics, Structure, and Abstraction" (Science)**
   - Bayesian program learning demonstrates that human-like generalization requires hierarchical probabilistic models with structured priors. Flat latent spaces (like LeWM's 192-dim embedding) lack the hierarchical structure needed for systematic generalization. The "blessing of abstraction": models that operate at the right level of abstraction (physical rules, not pixel correlations) can generalize combinatorially. LeWM operates at the wrong level.

8. **Batterman (2001) — "The Devil in the Details: Asymptotic Reasoning in Explanation, Reduction, and Emergence"**
   - Philosophical analysis of how different physical theories compose (or fail to compose) across scales. Key insight: not all physical phenomena are compositional. Emergent behaviors (turbulence, phase transitions, chaotic systems) are *irreducible* — they cannot be predicted from the composition of simpler components. If the held-out physical factor combinations in our experiments involve emergent phenomena, then the failure of compositional generalization would not be a limitation of the model but a genuine physical irreducibility. This provides a principled framework for distinguishing "fixable" failures (model limitation) from "unfixable" failures (physical irreducibility).

9. **Anderson (1972) — "More Is Different" (Science)**
   - The classic argument that at each level of complexity, new emergent properties arise that cannot be predicted from lower-level laws. For world models: combining gravity + friction + soft-body deformation may produce emergent dynamics (e.g., stick-slip oscillation) that are not predictable from separate knowledge of gravity, friction, and deformation. Compositional generalization has a *principled upper bound* set by the degree of emergence in the physical system.

10. **Marcus (2001) — "The Algebraic Mind: Integrating Connectionism and Cognitive Science"**
    - Argues that connectionist systems (which include all neural networks) are fundamentally limited in their ability to perform algebraic (rule-based) composition. A neural network can learn "gravity affects trajectories" and "friction affects velocities" but combining these into "gravity + friction jointly affect trajectory-velocity coupling" requires variable binding — a capability that distributed representations struggle with. This provides a theoretical ceiling on what LeWM (or any standard neural world model) can achieve compositionally.

### Landscape Summary

The interdisciplinary perspective reveals a fundamental **mismatch between the type of compositionality studied in ML and the type of compositionality found in physical systems**:

- **ML compositionality** is largely additive and linear: factor A occupies subspace V_A, factor B occupies subspace V_B, the composition A+B is represented by V_A + V_B. The Uselis et al. (2026) orthogonality requirement codifies this additive structure.

- **Physical compositionality** is frequently multiplicative, nonlinear, and subject to bifurcations: force = mass x acceleration (multiplicative), turbulence onset (bifurcation), three-body problem (chaotic). The composition of gravity and friction in a pendulum is not "gravity representation + friction representation" but involves coupled nonlinear ODEs where the interaction term is essential.

This mismatch suggests that the research question should be reframed: instead of asking "does LeWM generalize compositionally?" (which implicitly assumes additive composition), we should ask "for which types of physical factor combinations does the additive assumption hold, and where does it break?" The answer likely depends on the *physical regime*: in low-energy, linear-response regimes, additive composition may approximate well; near bifurcations or in strongly coupled systems, it will fail fundamentally.

---

## Phase 2: Initial Candidates

### Candidate A: Physical Regime-Dependent Generalization — Mapping the Boundary Between Composable and Emergent Dynamics

- **Hypothesis**: LeWM's compositional generalization performance depends critically on whether the held-out factor combination falls within a physically composable regime (where dynamics are approximately additive in the individual factors) or an emergent regime (where nonlinear interactions produce qualitatively new behavior). The generalization boundary in factor space corresponds to the onset of physical bifurcations or strong coupling, not to the model's representational quality.

- **Cross-domain insight**: From dynamical systems theory — physical systems exhibit distinct behavioral regimes separated by bifurcations. Near a bifurcation, small parameter changes cause large qualitative shifts. From Anderson (1972) — emergence creates principled limits on compositionality. The generalization boundary is not a model limitation but a reflection of physical structure.

- **Evidence for**: (1) Bifurcation theory provides a precise mathematical framework for predicting where qualitative transitions occur in parameter space. (2) No prior work on world model generalization accounts for the physical structure of the test domain — all evaluations treat factor combinations as uniform grid points. (3) The "phase transition" framing from the innovator perspective is consistent with bifurcation theory but lacks the physical grounding to predict *where* transitions occur.

- **Novelty estimate**: 9/10 — The idea that the *physics* of the test scenario, not just the *model*, determines compositional generalization boundaries is genuinely novel in the world model literature. All existing work treats generalization as a property of the model; this proposal treats it as a joint property of model and physical system.

### Candidate B: Curriculum-Dependent Compositional Acquisition — Does Training Order Affect Factor Separation?

- **Hypothesis**: Inspired by developmental psychology (Baillargeon, 2004), the order in which physical factor combinations are encountered during training affects the quality of compositional factorization in the latent space. Specifically, training that introduces factors one at a time (first gravity-only variation, then gravity + friction variation, then gravity + friction + mass variation) produces better factorized representations than training on all factor combinations simultaneously, because sequential introduction allows the model to establish orthogonal factor subspaces incrementally before encountering interactions.

- **Cross-domain insight**: From developmental psychology — infants learn physical concepts sequentially, with each new concept building on previously established ones. From curriculum learning — the order of training data matters for generalization in deep learning, but this has not been studied for physical factor factorization in world models.

- **Evidence for**: (1) Developmental psychology shows robust evidence for ordered concept acquisition. (2) Curriculum learning shows consistent benefits for structured learning problems. (3) Liang & Liu (2024) show that data structure drives representation structure in diffusion models — extending this to temporal ordering is a natural next step.

- **Novelty estimate**: 7/10 — Curriculum learning is well-studied, but applying developmental psychology insights to world model training for compositional factorization is new. The risk is that the effect may be small on a 15M model with limited capacity.

### Candidate C: Categorical Compositionality — Testing Functorial Structure in LeWM's Latent Space

- **Hypothesis**: LeWM's latent space lacks functorial structure — the composition of physical concepts in configuration space does not map homomorphically to the composition of their latent representations. Specifically, the model's representation of "gravity=2x AND friction=0.5x" is not systematically related to its representations of "gravity=2x" and "friction=0.5x" individually. This manifests as: (a) the vector difference between gravity=1x and gravity=2x representations depends on the value of friction, and (b) the dot product between friction-change and gravity-change displacement vectors varies across the factor space rather than being constant (which would indicate orthogonal additive composition).

- **Cross-domain insight**: From category theory — compositionality is formalized as functorial structure between categories. From linguistics — syntactic composition is recursive and rule-based, not interpolative. The test for functorial structure in a neural representation is whether composition in the input space maps to composition in the representation space *consistently*.

- **Evidence for**: (1) Category theory provides a principled mathematical framework for testing compositionality. (2) The "parallelogram test" (analogical reasoning) is a well-established proxy for additive composition in embedding spaces. (3) No prior work has tested functorial/algebraic composition in world model latent spaces — only statistical measures (probing accuracy, CKA).

- **Novelty estimate**: 8/10 — Applying category-theoretic compositionality tests to world model latent spaces is genuinely new. The parallelogram test is simple enough to implement and interpret. The risk is that the theoretical framework may be too demanding — real representations may approximate functorial structure without satisfying it exactly.

---

## Phase 3: Self-Critique

### Against Candidate A: Physical Regime-Dependent Generalization

- **Prior work attack**: Searched for "bifurcation world model generalization" and "dynamical regime transfer." Found cRSSM (Balestriero et al., 2024) which varies gravity in DMControl Walker but does not analyze bifurcation structure. Found no work connecting bifurcation theory to world model generalization boundaries. **Verdict: Genuinely novel.**

- **Methodological attack**: Identifying bifurcation points in parameterized physical environments requires either analytical computation (tractable for simple systems like pendulums) or numerical continuation methods (more complex). For DMControl environments using MuJoCo, the equations of motion are accessible but complex. **Mitigation**: Use empirical bifurcation detection — sweep parameters and look for discontinuous changes in behavior (e.g., sudden change in trajectory statistics like autocorrelation time, Lyapunov exponent, or variance). This is standard in dynamical systems analysis and does not require closed-form equations.

- **Theoretical attack**: Not all physical parameter changes produce bifurcations. In many regimes, physics is smoothly parametric (doubling gravity doubles the fall speed, linearly). In these smooth regimes, the concept of "regime-dependent generalization" reduces to standard interpolation/extrapolation, which is already captured by the pragmatist and empiricist perspectives. **Counter**: Even in smoothly parametric regimes, the *coupling strength* between factors varies — e.g., friction becomes more important relative to gravity as velocity increases. This coupling structure affects compositionality even without bifurcations.

- **Scalability attack**: Analyzing the bifurcation structure of the test environments adds significant analytical overhead. For a paper primarily about world model evaluation, the dynamical systems analysis may be a distraction. **Counter**: The analysis can be simplified to classifying each test point as "smoothly composable" vs. "strongly coupled" based on empirical trajectory statistics, without full bifurcation analysis. This adds one experimental condition, not a separate research program.

- **Verdict**: **STRONG** — The core insight (physical structure of the test domain matters for generalization) is novel and actionable. The implementation risk is manageable by using empirical proxies for bifurcation structure. The main weakness is that the effect may be hard to demonstrate in the simple physics of PushT or Walker environments, which lack dramatic bifurcations.

### Against Candidate B: Curriculum-Dependent Compositional Acquisition

- **Prior work attack**: Searched for "curriculum learning compositional generalization" and "ordered concept acquisition neural network." Found substantial curriculum learning literature but nothing on physical factor factorization in world models. Found one paper on curriculum learning for RL that varies difficulty but not physical factors. **Verdict: Novel application to world models, but the general idea (order matters) is well-established.**

- **Methodological attack**: Testing multiple training orderings requires many training runs. With 3 physical factors, there are 3! = 6 possible orderings for sequential introduction, plus simultaneous training as a baseline. With 3 seeds each, that is 21 training runs, each ~2-4 hours. Total: ~42-84 GPU-hours. This is feasible but expensive, and the effect size may be small. **Mitigation**: Test 3 orderings (sequential, reverse-sequential, simultaneous) with 3 seeds = 9 runs. ~18-36 GPU-hours.

- **Theoretical attack**: The developmental psychology analogy is suggestive but the mechanism in biological systems (maturational gating, structured experience) is fundamentally different from what can be implemented in neural network training (data ordering, learning rate scheduling). The analogy may be misleading. **Counter**: The mechanism need not be identical — the prediction is about the *outcome* (better factorization from ordered introduction), not the mechanism.

- **Scalability attack**: LeWM trains on collected datasets (offline RL data), not online interaction. Implementing curriculum requires either (a) splitting data by factor variation and training in stages, or (b) progressively introducing more diverse data during training. Option (a) is straightforward; option (b) requires modifying the training pipeline. **Verdict**: Option (a) is clean and minimal.

- **Verdict**: **MODERATE** — Interesting hypothesis but the developmental psychology analogy is loose, the effect size may be small, and the computational cost is significant. Best as a secondary experiment rather than the main contribution.

### Against Candidate C: Categorical Compositionality (Parallelogram Test)

- **Prior work attack**: Searched for "parallelogram test embedding compositionality" and "analogical reasoning embedding space." Found extensive NLP work on word analogies (king - man + woman = queen) which is essentially the parallelogram test applied to word embeddings. Found Hupkes et al. (2020) "Compositionality Decomposed" which formalizes composition types but not in the geometric embedding sense. Found no application to world model latent spaces. **Verdict: Novel application, but the method (parallelogram test) is well-known.**

- **Methodological attack**: The parallelogram test requires identifying the "factor displacement vectors" — e.g., the vector from gravity=1x to gravity=2x in latent space. These vectors must be estimated empirically by averaging over many samples, and they may vary across the latent space (non-constant displacement). **Mitigation**: Compute displacement vectors at multiple reference points and measure their variance — this variance itself is a measure of non-functoriality.

- **Theoretical attack**: The parallelogram test only captures *additive* composition (functors that map addition in factor space to addition in representation space). Physical factor composition is often multiplicative or nonlinear (as argued in Candidate A). A model could fail the parallelogram test while still achieving practical compositional generalization via nonlinear composition. **Counter**: This is a feature, not a bug — the discrepancy between parallelogram-test predictions and actual generalization performance would precisely quantify the nonlinearity of physical factor composition.

- **Scalability attack**: The parallelogram test is computationally trivial — it requires only basic linear algebra on extracted embeddings. The main cost is training the base model and collecting embeddings, which is already required for the other evaluation metrics.

- **Verdict**: **STRONG** — A clean, well-defined, computationally cheap metric that captures a specific (additive) form of compositionality. The limitation to additive composition is actually informative — it distinguishes regimes where additive composition suffices from regimes where nonlinear composition is needed. Pairs naturally with Candidate A.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B** (Curriculum Learning) is dropped as a primary contribution. The developmental psychology analogy is suggestive but the effect size is likely small, the computational cost is high, and the mechanism is underspecified. It could be a secondary experiment if compute allows.

### Strengthened Survivors

**Primary proposal: Merge Candidates A and C into "Physics-Informed Compositional Evaluation Framework for World Models"**

The two candidates are naturally complementary:
- **Candidate C (parallelogram test)** provides a *metric* for measuring additive compositional structure in the latent space
- **Candidate A (regime analysis)** provides a *physical explanation* for when and why that metric should succeed or fail

Together, they form a predictive framework: the parallelogram test measures the degree of additive composition; the physical regime analysis predicts whether additive composition is *possible* for a given factor combination; and the discrepancy between them reveals whether failures are due to model limitations (parallelogram fails in additive regime) or physical irreducibility (parallelogram fails in emergent regime).

**Specific strengthening changes:**
1. Added the **consistency of displacement vectors** as a key metric: for each factor, compute the displacement vector at multiple reference points in factor space. If the vectors are consistent (low variance), the factor is additively composed; if inconsistent (high variance), the representation is context-dependent (non-functorial).
2. Added **coupling strength estimation**: for each pair of physical factors, estimate their interaction strength from the variance of the cross-displacement (how much changing factor A affects the displacement vector for factor B). Physical factors with high interaction strength should show higher variance.
3. Added **regime classification**: classify each test point as "weakly coupled" (factors approximately independent) or "strongly coupled" (factors interact nonlinearly) based on empirical trajectory statistics. Predict that compositional generalization succeeds in weakly coupled regimes and fails in strongly coupled regimes, *independent of model quality*.
4. Incorporated the **Chomsky/Marcus insight** as an interpretive framework: the limit of compositional generalization in continuous-embedding world models is set by the requirement for variable binding (combining rules with arguments), which distributed representations handle poorly. This provides a theoretical explanation for why parallelogram composition degrades in complex physical regimes.

### Selected Front-Runner

**"Physics-Informed Compositional Evaluation Framework: Predicting World Model Generalization Boundaries from Physical Regime Structure"**

This is selected because:
1. It offers a genuinely interdisciplinary contribution that no other perspective captures: the physical structure of the test domain as a predictor of generalization
2. It provides actionable predictions that can be validated within the same experimental framework as the other perspectives (shared environments, shared training)
3. It introduces two concrete novel metrics (displacement vector consistency, coupling strength estimation) that are cheap to compute
4. It distinguishes between "model-fixable" and "physically-irreducible" failures, which has practical implications for system design

---

## Phase 5: Final Proposal

### Title

Physics-Informed Compositional Evaluation for JEPA World Models: Predicting Generalization Boundaries from Physical Regime Structure

### Hypothesis

**H1 (Regime-dependent generalization)**: LeWM's compositional generalization performance on held-out physical factor combinations is primarily determined by the physical coupling structure of those combinations, not by the model's representational quality (probing accuracy) or regularization (SIGReg strength). Specifically, factor combinations in weakly-coupled physical regimes (where factors contribute approximately additively to dynamics) will show successful zero-shot transfer, while strongly-coupled combinations (where nonlinear interactions dominate) will fail — regardless of how well the model performs in-distribution.

**H2 (Additive composition boundary)**: The latent representation satisfies additive composition (measured by the parallelogram test) only in weakly-coupled regimes. The consistency of factor displacement vectors across the latent space serves as a quantitative predictor of compositional transfer success, with Pearson r > 0.6 between displacement consistency and holdout probing accuracy.

**H3 (Model-agnostic failure)**: Failures of compositional generalization in strongly-coupled regimes persist across model variants (SIGReg, VICReg, DINO-WM), indicating that the failure is a property of the physical system, not the model architecture or regularization choice.

### Motivation

The world model community evaluates compositional generalization as a property of the *model*: does the model's representation decompose factorially? Does its regularizer promote orthogonal structure? But this framing ignores the *physics*: not all physical factor combinations are composable. When gravity and friction interact to produce stick-slip dynamics, or when parameter changes cross a bifurcation boundary, the resulting behavior is emergent — it cannot be predicted from separate knowledge of gravity and friction. Evaluating a world model on such combinations and concluding it "fails to generalize compositionally" conflates model limitation with physical irreducibility.

This perspective draws from dynamical systems theory (Guckenheimer & Holmes, 1983), the philosophy of emergence (Anderson, 1972; Batterman, 2001), and cognitive science (Spelke, 2007; Lake et al., 2017) to argue that compositional generalization has a *physics-dependent upper bound*. We propose a framework that estimates this upper bound empirically, allowing fair evaluation of model generalization against a physically-informed baseline rather than against an oracle trained on the test distribution.

The practical implication is significant: if a system designer knows which factor combinations are weakly coupled (and thus amenable to compositional transfer) vs. strongly coupled (requiring explicit training data), they can make informed decisions about training data collection — a problem of clear practical importance for robotics and simulation.

### Method

**1. Physical Regime Analysis (Novel Component)**

For each pair of physical factors (f_i, f_j) in the experimental grid:
- **Coupling strength estimation**: Collect trajectories from environments at multiple factor levels. For each trajectory, compute summary statistics (mean velocity, acceleration variance, contact frequency, energy dissipation rate). Estimate the interaction strength as the magnitude of the mixed partial derivative of the trajectory statistics with respect to f_i and f_j:

  coupling(f_i, f_j) = |d^2 S / (d f_i * d f_j)|

  where S is a trajectory statistic vector. High coupling indicates nonlinear interaction; low coupling indicates approximate independence.

- **Regime classification**: Classify each factor combination as "weakly coupled" (all pairwise coupling strengths below a threshold c*) or "strongly coupled" (at least one coupling exceeds c*). Threshold c* is calibrated from the training-distribution coupling statistics.

- **Bifurcation detection** (optional): For environments with known analytical structure (e.g., pendulum), compute the Lyapunov exponent spectrum as a function of physical parameters. Identify parameter values where the maximum Lyapunov exponent changes sign (indicating a qualitative transition from stable to chaotic dynamics).

**2. Parallelogram Test for Additive Composition**

For each physical factor f_k and each pair of test environments (env_A, env_B) that differ only in f_k:
- Extract latent embeddings z_A and z_B for matched trajectories
- Compute the factor displacement vector: delta_k = mean(z_B) - mean(z_A)
- Repeat at multiple reference points in factor space (at least 3 different background configurations of other factors)
- **Displacement consistency score**: cos_similarity between delta_k vectors computed at different reference points. Score = 1.0 means perfect additive composition (context-independent displacement); score = 0.0 means fully context-dependent (non-compositional).
- **Cross-factor orthogonality**: dot product between displacement vectors for different factors. Low dot product indicates orthogonal composition.

**3. Integration with Standard Evaluation**

All metrics from other perspectives are also computed (probing accuracy, principal angles, IIS, planning success). The novel contribution is the *prediction*: for each held-out combination, the regime analysis predicts whether compositional transfer should succeed (weakly coupled) or fail (strongly coupled), and we measure whether this prediction is more accurate than alternatives (e.g., Grassmannian distance, probing accuracy, SIGReg residual).

**4. Model-Agnostic Failure Test**

Train 3 model variants (LeWM-SIGReg, LeWM-VICReg, DINO-WM) on the same training split. If strongly-coupled failures persist across all variants while weakly-coupled successes also persist, the regime classification is validated as model-agnostic.

### Experimental Plan

| Experiment | Metric | Baselines | Falsification |
|-----------|--------|-----------|---------------|
| Coupling strength estimation | Mixed partial derivatives of trajectory statistics | Random baseline (shuffled labels) | Coupling scores not significantly above random |
| Parallelogram test (coverage sweep) | Displacement consistency per factor | Random encoder, untrained model | Consistency scores not above random at any coverage level |
| Regime-generalization correlation | Pearson r between coupling strength and holdout error | Probing accuracy as predictor, Grassmannian distance as predictor | Coupling strength is a worse predictor than alternatives |
| Model-agnostic failure test | Cross-model failure pattern consistency | Independent model failures | Failure patterns are model-specific, not regime-specific |

### Resource Estimate

- **Physical regime analysis**: ~2h (trajectory collection + statistics computation). No GPU required for analysis.
- **Parallelogram test**: ~1h (embedding extraction + linear algebra). Negligible compute.
- **Model training** (shared with other perspectives): ~60 GPU-hours for 30 training runs.
- **Additional analysis unique to this perspective**: ~5h total, primarily CPU.
- **Individual experiment tasks**: All under 1 hour. The regime analysis is a single batch computation.

### Risk Assessment

1. **Risk: Physical coupling is uniformly weak in PushT/Walker environments**
   - Likelihood: Medium (PushT is a quasi-static 2D system with mild coupling; Walker has stronger gravitational-inertial coupling)
   - Mitigation: Include at least one environment with known strong coupling (e.g., CartPole with varying pole mass and cart friction, where the stability boundary creates a clear bifurcation). If all environments show weak coupling, the prediction that "compositional generalization succeeds in weakly coupled regimes" is trivially confirmed but the interesting strongly-coupled test is missed.
   - Impact: Would reduce the paper's novelty to the parallelogram test metric, which is still a contribution.

2. **Risk: Coupling strength is too noisy to be a reliable predictor**
   - Likelihood: Medium (estimating mixed partial derivatives from finite trajectory samples is statistically challenging)
   - Mitigation: Use bootstrap confidence intervals on the coupling estimates. If CI is too wide, use a cruder regime classification (binary: "known-interacting" vs. "known-independent" based on physics knowledge, not estimated).
   - Impact: Falls back to expert-classified regimes rather than data-driven estimation. Still validates the core hypothesis.

3. **Risk: The parallelogram test gives results indistinguishable from principal angle analysis**
   - Likelihood: Medium-high (both metrics capture aspects of linear composition)
   - Mitigation: The displacement consistency metric captures *context-dependence* of composition, which principal angles do not. Report the incremental predictive value of displacement consistency over principal angles via partial correlation or incremental R^2.
   - Impact: Even if the metrics correlate, the interpretation is different (context-dependent displacement is more directly linked to the compositional operation than subspace angle).

4. **Risk: All model variants fail on all held-out combinations, making regime analysis moot**
   - Likelihood: Low-medium (the pragmatist perspective's pilot design should calibrate this)
   - Mitigation: If all combinations fail, the interesting question becomes "do some fail more than others?" The coupling strength still predicts the degree of failure. Also, LoRA adaptation may succeed on some combinations — regime analysis predicts which ones.
   - Impact: Reduces the qualitative distinction (success vs. failure) to a quantitative one (degree of failure), which is still informative.

### Novelty Claim

This proposal is interdisciplinary-novel in three specific ways:

1. **Physical regime analysis as a generalization predictor**: No prior work on world model evaluation accounts for the physical structure of the test domain. All existing evaluations treat held-out factor combinations as a uniform grid of equally difficult test points. We propose that the physical coupling structure (measurable from trajectory statistics) predicts compositional transfer success or failure, providing a physics-informed baseline for model evaluation.

2. **Displacement vector consistency as a compositionality metric**: While the parallelogram test is well-known in NLP (word analogy tasks), its application to world model latent spaces — measuring whether factor displacement vectors are consistent across the latent space — is new. The key adaptation is that we measure consistency at multiple reference points in factor space, directly testing the *functorial* property of the representation (whether composition in factor space maps consistently to composition in latent space).

3. **Model-agnostic failure analysis**: By testing whether specific failure patterns persist across model variants, we distinguish model limitations from physical irreducibility. This framing — that some failures are *desirable* (they reflect genuine physical complexity) while others are *undesirable* (they reflect model deficiency) — has not been articulated in the world model literature. It provides a fairer evaluation framework that does not penalize models for failing on physically irreducible combinations.

The cross-disciplinary synthesis draws from dynamical systems theory (coupling estimation, bifurcation structure), category theory (functorial composition, parallelogram test), developmental psychology (ordered concept acquisition as a secondary prediction), and philosophy of science (emergence, irreducibility) — all applied to the concrete empirical question of world model compositional generalization. No single discipline provides this framework; the integration is the contribution.

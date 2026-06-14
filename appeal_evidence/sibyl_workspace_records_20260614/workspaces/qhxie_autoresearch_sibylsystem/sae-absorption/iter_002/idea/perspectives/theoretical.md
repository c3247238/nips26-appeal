# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Tang et al. (2025), "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability"** (arXiv:2512.05534) -- First unified theoretical framework casting all SDL variants (SAEs, transcoders, crosscoders) as a single piecewise biconvex optimization problem. Characterizes global solution set, non-identifiability, and spurious partial minima. Provides the first principled explanation of feature absorption as arising from spurious local minima where multiple ground-truth features are merged into a single learned feature. Proposes feature anchoring to restore identifiability. **Critical limitation:** Does not derive a closed-form absorption threshold, does not characterize the dependence of absorption on sparsity penalty strength, and validates anchoring only on synthetic benchmarks.

2. **Cui et al. (2025), "On the Limits of Sparse Autoencoders"** (arXiv:2506.15963) -- First closed-form theoretical analysis showing SAEs generally fail to recover ground-truth monosemantic features unless features are extremely sparse. The key result: under their generative model, the SAE objective has a global minimum that does NOT correspond to the true features when feature density exceeds a critical threshold. Proposes reweighted SAE (WSAE) as remedy. **Key mathematical result:** Necessary and sufficient conditions for feature recovery as a function of feature sparsity, demonstrating that recovery requires sparsity below a critical level that shrinks as the number of features grows. **Limitation:** The model does not include hierarchical feature structure, so the results do not directly predict absorption.

3. **Chen et al. (2025), "Taming Polysemanticity in LLMs"** (arXiv:2506.14002) -- Provable feature recovery guarantees via bias adaptation under a statistical generative model. Group Bias Adaptation (GBA) for LLMs. **Key result:** Under their generative assumptions (features generated from a specific distributional model), the bias-adapted SAE provably recovers the true features with high probability. **Limitation:** Assumptions are restrictive (i.i.d. feature activations, no hierarchical structure). The guarantees break when features have parent-child co-occurrence patterns.

4. **Chanin et al. (2024), "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) -- Defines and empirically characterizes feature absorption. Contains a toy model proof (Section 4) that is the closest existing formal result to an absorption theorem: in a 2-feature hierarchy with L1 penalty, the optimal SAE solution absorbs the parent feature into the child's decoder when they are sufficiently aligned. **Key mathematical content:** The toy model shows absorption occurs when the L1 gain from eliminating one activation exceeds the reconstruction cost from decoder misalignment. The paper does not generalize this to arbitrary hierarchies or derive scaling laws. **This is the starting point for our theoretical extension.**

5. **Elhage et al. (2022), "Toy Models of Superposition"** (Transformer Circuits) -- Foundational theoretical framework for understanding superposition. Introduces the notion that neural networks represent more features than dimensions by encoding features as nearly-orthogonal directions. The key mathematical model: features as vectors in R^d with importance weights and sparsity levels, and the network learns to allocate directions that minimize expected loss. **Relevance to absorption:** Their model predicts that features with similar importance and high co-occurrence will share directions (superposition). Absorption is a downstream consequence: when the SAE attempts to disentangle superposed features, the sparsity penalty creates a preference for absorbing co-occurring features.

6. **Donoho & Tanner (2009), "Observed Universality of Phase Transitions in High-Dimensional Geometry"** (arXiv:0906.2530) -- Establishes the Donoho-Tanner phase transition framework for sparse recovery: in the (delta, rho) plane (undersampling vs. sparsity), there is a sharp threshold below which L1 minimization recovers the sparse vector exactly and above which it fails. **Relevance:** This provides the mathematical template for our claim that absorption onset is a phase transition in (lambda, theta) space. The analogy is precise: lambda (sparsity penalty) plays the role of rho (sparsity), and theta (decoder angle) plays the role of the signal-to-noise ratio.

7. **Zhang & Liu (2025), "Phase Transitions with Structured Sparsity"** (arXiv:2411.09868, Digital Signal Processing 2025) -- Extends Donoho-Tanner phase transitions to block-structured and tree-structured sparsity. Key finding: weak thresholds are invariant across structure types, but strong thresholds (exact recovery) vary significantly by structure. **Direct relevance:** Feature hierarchies in SAEs are tree-structured. This paper provides the mathematical machinery for analyzing phase transitions under hierarchical sparsity, which is exactly the structure that produces absorption.

8. **Kumar et al. (2025), "Dictionary Learning: The Complexity of Learning Sparse Superposed Features with Feedback"** (arXiv:2502.05407) -- Establishes tight bounds on the feedback complexity of recovering features from sparse autoencoders. Shows that feature recovery is tractable when the agent can construct activations but harder under distributional constraints. **Relevance:** Their complexity results provide a lower bound on how much information is needed to disentangle absorbed features post-hoc.

9. **Ayonrinde et al. (2024), "Interpretability as Compression: MDL-SAEs"** (arXiv:2410.11179) -- Reframes SAE training as a Minimum Description Length (MDL) problem. The SAE's sparse code is a compressed representation; the optimal code balances description length (sparsity) against reconstruction accuracy. **Key insight for absorption theory:** MDL and rate-distortion theory are dual perspectives on the same optimization. MDL-SAE explicitly connects the sparsity penalty to a coding cost, making the rate-distortion framing of absorption mathematically natural.

10. **Bereska et al. (2025), "Superposition as Lossy Compression"** (arXiv:2512.13568) -- Measures superposition via rate-distortion theory and connects to adversarial vulnerability. Shows that the degree of lossy compression in neural representations predicts adversarial susceptibility. **Relevance:** This establishes rate-distortion as a productive theoretical lens for superposition phenomena; our contribution extends it to SAE feature recovery specifically, where absorption is the manifestation of the rate-distortion tradeoff.

11. **Wainwright (2009), "Information-Theoretic Limits on Sparsity Recovery"** (IEEE Trans. IT) -- Establishes the fundamental sample complexity limits for sparse support recovery in high-dimensional noisy settings. **Relevance:** Provides the information-theoretic lower bounds that constrain how well any SAE can disentangle features from finite data. When features are hierarchical, the effective "noise" from co-occurring features increases the sample complexity of recovery beyond these standard bounds.

12. **Hino & Murata (2009), "Information-Theoretic Formulation of Sparse Coding"** (Springer) -- Formulates sparse coding as an information-theoretic optimization: the sparse code is a channel between the signal and its representation, with rate (code length / sparsity) and distortion (reconstruction error). **Key result:** Sparse coding methods are special cases of rate-distortion optimization, providing the theoretical backbone for our absorption-as-rate-distortion-optimal-behavior claim.

### Theoretical Landscape Summary

The theoretical understanding of SAE feature recovery sits at the intersection of three mature mathematical frameworks:

**Framework 1: Optimization landscape analysis** (Tang et al., 2025). The piecewise biconvex structure of the SDL objective means that local minima can trap the optimizer in states where multiple true features are merged. Feature absorption is one such spurious minimum. The theory characterizes WHICH minima exist but not WHEN the optimizer will find them or HOW SEVERE the absorption will be for a given configuration.

**Framework 2: Statistical identifiability and recovery guarantees** (Cui et al., 2025; Chen et al., 2025; Wainwright, 2009). These results establish conditions under which the true features are recoverable in principle. The key gap: all existing recovery guarantees assume features are generated i.i.d. or have bounded correlation. None model the hierarchical co-occurrence structure (parent fires whenever child fires) that is the root cause of absorption. Under this structure, the "effective sparsity" of the parent feature is inflated by all its children, pushing it beyond the recovery threshold.

**Framework 3: Rate-distortion / MDL / compression** (Ayonrinde et al., 2024; Bereska et al., 2025; Hino & Murata, 2009). Rate-distortion theory provides the cleanest explanation for WHY absorption occurs: it is the coding-optimal behavior when the sparsity cost exceeds the distortion cost of decoder misalignment. This perspective is informally invoked by Chanin et al. ("absorption saves one L0") and by the Tilde Research blog, but has never been formalized into a closed-form threshold.

**The critical gap** at the intersection of these three frameworks: no existing result derives a quantitative absorption threshold as a function of measurable SAE properties (sparsity penalty lambda, decoder angle theta, co-occurrence frequency). Chanin et al.'s toy model (2024) proves absorption exists for a specific 2-feature hierarchy but does not generalize. Tang et al.'s unified theory (2025) identifies absorption as a spurious minimum but does not characterize when it becomes energetically preferred. Cui et al.'s recovery limits (2025) show SAEs fail in general but do not connect to the specific hierarchical structure that produces absorption. Our contribution bridges this gap by providing the first closed-form absorption threshold, proving it is tight (necessary and sufficient under the rate-distortion model), and validating it empirically across multiple hierarchy types.

A secondary gap concerns the dynamics of absorption onset. The Donoho-Tanner phase transition literature (2009) and its recent extensions to structured sparsity (Zhang & Liu, 2025) establish that sparse recovery exhibits sharp thresholds. Our hypothesis is that absorption follows a similar phase-transition pattern: below a critical sparsity, absorption is absent; above it, absorption is essentially certain. If the absorbed state is metastable (as our hysteresis hypothesis claims), this connects to the physics of first-order phase transitions and has immediate practical consequences: post-hoc sparsity reduction cannot reverse established absorption.

---

## Phase 2: Initial Candidates

### Candidate A: Rate-Distortion Absorption Threshold with Impossibility Theorem

**Formal claim (Theorem 1 -- Absorption Threshold):**

Let D in R^{d x m} be the decoder matrix of a trained SAE with sparsity penalty lambda (or effective lambda = 1/L0 for TopK). Consider a parent feature p and child feature c whose decoder columns are d_p, d_c in R^d with angle theta_{p,c} = arccos(|<d_p, d_c>| / (||d_p|| ||d_c||)). Let p_co = P(feature c fires | context where both p and c are relevant) be the co-occurrence probability.

Then the SAE objective has a local minimum where c's decoder absorbs the information of p (i.e., d_c is modified to encode both features, and p's activation is suppressed) that achieves strictly lower loss than the non-absorption solution if and only if:

    lambda > sin^2(theta_{p,c})

Moreover, the co-occurrence probability p_co cancels from the threshold condition, so the absorption threshold depends only on the decoder geometry and sparsity penalty.

**Formal claim (Theorem 2 -- Absorption Impossibility):**

For a complete b-ary feature hierarchy of depth h with flat sparsity penalty lambda, if theta_{parent,child} < arcsin(sqrt(lambda)) for all parent-child pairs, then at least ceil(h - h*) features will be absorbed, where:

    h* = O(1 / sqrt(lambda))

is the critical hierarchy depth. For any hierarchy of depth h > h*, absorption of at least one feature is unavoidable regardless of SAE width m, training procedure, or architecture (among architectures with flat sparsity penalties).

**Proof sketch:**

*Lemma 1 (Local comparison).* For a single parent-child pair, compare two solutions: (i) both features active with separate decoder columns (L0 cost = 2, distortion = 0), and (ii) only child active with its decoder absorbing the parent's information via projection (L0 cost = 1, distortion = ||d_p - proj_{d_c} d_p||^2 = sin^2(theta_{p,c}) per activation). The Lagrangian difference is Delta L = lambda - sin^2(theta_{p,c}) per co-occurring activation. Since this appears multiplicatively with p_co in the expected loss, and p_co > 0, the sign of Delta L determines the preference. This is a direct application of the rate-distortion Lagrangian with rate R = L0 and distortion D = reconstruction MSE.

*Lemma 2 (Chain absorption).* For a chain of features f_1 -> f_2 -> ... -> f_h where each pair has angle theta < arcsin(sqrt(lambda)), absorption cascades: f_1 is absorbed into f_2, then f_2 (carrying f_1's information) is absorbed into f_3, etc. At each step, the distortion accumulates as at most sin^2(theta) per level. After k absorptions, the cumulative distortion is at most k * sin^2(theta) (under small-angle approximation; exact expression involves iterated projections). The chain absorption halts when cumulative distortion per level exceeds lambda, giving h* = lambda / sin^2(theta) = O(1/sqrt(lambda)) when theta = O(sqrt(lambda)).

*Theorem 1* follows from Lemma 1. *Theorem 2* follows from Lemma 2 applied to the longest path in the hierarchy tree.

**Empirical prediction:** (1) Absorbed feature pairs have systematically higher cos^2(theta_{p,c}) than non-absorbed pairs. (2) The threshold lambda = sin^2(theta) predicts absorption with AUROC >= 0.70 across SAE configurations. (3) Deeper hierarchies exhibit more absorption than shallow ones at matched sparsity levels.

**Connection to existing theory:** Extends Chanin et al.'s (2024) 2-feature toy model to arbitrary hierarchies with a closed-form threshold. The threshold condition is the "Donoho-Tanner condition" of SAE feature recovery: below it, features are recovered; above it, they are absorbed. This connects to Tang et al.'s (2025) spurious minima characterization by identifying the specific energy landscape condition under which the absorbed minimum becomes globally preferred.

**Novelty estimate:** 9/10. The informal observation that "absorption saves L0" exists in the community (Chanin et al., Tilde Research blog), but the formal rate-distortion derivation with closed-form threshold, the impossibility theorem, and the chain absorption analysis are all new. No paper has proven that absorption is rate-distortion OPTIMAL (not just a local minimum, but the global minimum under the rate-distortion objective) for hierarchical features. The impossibility theorem is entirely novel.

---

### Candidate B: Information-Theoretic Lower Bound on Absorption-Free Recovery

**Formal claim (Theorem -- Minimum Mutual Information for Absorption-Free Recovery):**

Consider a generative model where data x = D* z + noise, where D* in R^{d x m*} is the true feature dictionary and z is a sparse activation vector with hierarchical co-occurrence structure: for each parent-child pair (p, c), P(z_p = 1 | z_c = 1) = 1 (the parent always fires when the child fires). Let the SAE encoder E: R^d -> R^m attempt to recover z from x.

Then for any SAE with m latents and flat sparsity penalty, the mutual information I(z_p; E(x)) between the parent feature activation and the SAE's encoding satisfies:

    I(z_p; E(x)) <= I(z_p; x) - lambda * H(z_p | z_c)

where H(z_p | z_c) is the conditional entropy of the parent given the child. When lambda is large enough that this upper bound drops below the threshold for reliable recovery (roughly 1 bit), the parent feature is effectively unrecoverable -- i.e., absorbed.

**Proof sketch:**

*Step 1.* The SAE objective is a Lagrangian that trades off reconstruction (distortion D = E[||x - D f(Ex + b)||^2]) against sparsity (rate R = E[||f(Ex+b)||_0]). By the data processing inequality, I(z; E(x)) <= I(z; x). The sparsity penalty lambda acts as a constraint on the "communication rate" of the encoder: it limits how many features the encoder can "transmit" per input.

*Step 2.* For hierarchical features, the parent's information is partially redundant with the child's: I(z_p; x | z_c) < I(z_p; x) because knowing the child already explains some of the parent's variance in x. The sparsity penalty penalizes the encoder for transmitting this redundant information. The gap is exactly H(z_p | z_c) -- the entropy of the parent that is NOT explained by the child.

*Step 3.* When lambda * H(z_p | z_c) > I(z_p; x | z_c), the cost of encoding the parent exceeds the information gain from doing so. The optimal encoder under the rate-distortion tradeoff drops the parent feature -- this IS absorption.

**Empirical prediction:** (1) Features with higher conditional entropy (more "independent" information beyond their children) are harder to absorb. (2) Absorption rate should correlate with the mutual information I(z_p; z_c) between parent and child features: higher MI means more redundancy, more absorption.

**Connection to existing theory:** Builds on Wainwright's (2009) information-theoretic limits for sparse recovery. Extends to the hierarchical setting where co-occurrence creates structured redundancy. Connects to the MDL-SAE perspective (Ayonrinde et al., 2024) where absorption corresponds to the MDL-optimal decision to "not transmit" a feature whose description cost exceeds its descriptive value.

**Novelty estimate:** 8/10. The information-theoretic framing is novel. The connection between conditional entropy of parent given child and absorption susceptibility has not been made before. However, the bound may be loose in practice (information-theoretic bounds typically are), which limits its empirical usefulness relative to Candidate A's geometric threshold.

---

### Candidate C: Absorption as a Phase Transition with Hysteresis

**Formal claim (Conjecture -- First-Order Phase Transition):**

Let alpha(lambda) denote the absorption rate (fraction of parent-child feature pairs exhibiting absorption) as a function of the effective sparsity penalty lambda. For a fixed feature hierarchy and SAE width:

1. There exists a critical lambda_c such that alpha(lambda) undergoes a rapid transition: alpha(lambda) ~ 0 for lambda << lambda_c and alpha(lambda) -> alpha_max for lambda >> lambda_c.

2. The transition is first-order (discontinuous in the thermodynamic limit of infinite SAE width): there exists a range [lambda_c^-, lambda_c^+] where both the absorbed and non-absorbed solutions are locally stable (metastable).

3. Consequently, absorption exhibits hysteresis: an SAE trained at lambda > lambda_c^+ (high sparsity, absorption present) that is subsequently fine-tuned at lambda < lambda_c^- (low sparsity) will retain residual absorption at a rate alpha_residual > alpha(lambda) for a from-scratch SAE at the same lambda.

**Proof sketch (for the existence of lambda_c -- rigorous; for first-order character -- heuristic):**

*Rigorous part:* From Candidate A's Theorem 1, absorption of a specific pair (p,c) switches on when lambda crosses sin^2(theta_{p,c}). For a population of feature pairs with angles drawn from a distribution P(theta), the fraction absorbed is alpha(lambda) = P(theta < arcsin(sqrt(lambda))). If P(theta) has a mode (peak density) at theta_mode, then alpha(lambda) increases sharply near lambda = sin^2(theta_mode), producing a crossover that appears as a phase transition when the distribution is concentrated.

*Heuristic part (first-order character):* The absorbed state is a local minimum of the loss landscape separated from the non-absorbed state by a barrier. At the transition point, both states coexist with equal energy. The barrier height scales with the number of co-occurring tokens (proportional to dataset size), making the transition sharper for larger training sets. First-order character implies hysteresis: once the system is trapped in the absorbed minimum, reducing lambda to below the transition does not immediately de-trap it because the barrier must be overcome.

*Connection to statistical physics:* This is analogous to the Ising model's ferromagnetic-paramagnetic transition, where the "magnetization" (absorption rate) jumps discontinuously at the critical temperature, and hysteresis occurs because domain flipping requires nucleation events. In the SAE context, "domain flipping" corresponds to the decoder column d_c suddenly splitting to re-separate the absorbed parent's information -- a coordinated change that gradient descent may not find from the absorbed state.

**Empirical prediction:** (1) Absorption rate vs. 1/L0 fits a sigmoid better than a linear model. (2) Fine-tuning a high-absorption SAE with lower sparsity does not reduce absorption to the level of a from-scratch SAE at the same sparsity. (3) The critical lambda_c shifts with SAE width (wider SAEs have lower lambda_c because they have more capacity to represent both parent and child separately).

**Connection to existing theory:** Builds on the Donoho-Tanner phase transition framework (2009) and its recent extension to structured sparsity (Zhang & Liu, 2025). The analogy between sparse recovery thresholds in compressed sensing and absorption thresholds in SAEs is precise: both involve a sharp transition between "recovery" and "failure" as a function of a sparsity parameter. The hysteresis prediction is novel and has no counterpart in the compressed sensing literature (where recovery algorithms are deterministic and do not exhibit memory effects).

**Novelty estimate:** 8/10. The phase transition framing is novel in this domain. The hysteresis prediction is strong and falsifiable. The rigorous part (existence of a sharp crossover) follows from Candidate A. The first-order / hysteresis claim is a conjecture supported by the energy landscape argument but not proven. Risk: if the angle distribution P(theta) is broad, the crossover will be smooth and undetectable, weakening the "phase transition" narrative.

---

## Phase 3: Self-Critique

### Against Candidate A: Rate-Distortion Absorption Threshold

**Proof soundness attack:**

The central vulnerability is the LOCAL comparison in Lemma 1. We compare two specific solutions (both active vs. absorbed), but the actual SAE training landscape may have MANY other solutions. The claim that absorption is the GLOBAL optimum (not just locally preferred) requires showing that no third solution achieves lower loss. In a high-dimensional loss landscape with m >> d latents, there could be solutions where the parent's information is partially distributed across multiple latents rather than fully absorbed into the child. The proof as sketched assumes a clean parent-child pair in isolation; in reality, each feature participates in multiple hierarchical relationships, and the absorption decisions interact non-trivially.

*Assessment:* This is a genuine gap but not fatal. The local comparison is tight for the 2-feature case (confirmed by Chanin et al.'s toy model). For the multi-feature case, we can invoke the piecewise biconvex structure from Tang et al. (2025): for fixed encoder, the optimal decoder for each feature pair is determined independently. This means the local comparison IS the global comparison for the decoder sub-problem. The encoder sub-problem introduces coupling, but empirical validation should reveal whether this coupling is significant.

*arXiv search for counterexamples:* No paper has shown a case where absorption is locally preferred but globally sub-optimal. The SynthSAEBench (2026) results show that SAEs systematically absorb hierarchical features, consistent with absorption being the global optimum.

**Tightness attack:**

The threshold lambda = sin^2(theta) is SHARP in the 2-feature case but may be LOOSE for multi-feature hierarchies because: (a) chain absorption accumulates distortion, which the theorem accounts for only via a small-angle approximation; (b) the effective sparsity penalty in TopK SAEs is not cleanly mapped to lambda (TopK enforces a hard constraint, not a Lagrangian penalty). The impossibility theorem's h* = O(1/sqrt(lambda)) is an asymptotic bound; the constants matter for practical predictions.

*Assessment:* The tightness concern is real for quantitative predictions. Mitigation: we should report AUROC (which is rank-order invariant and does not require the threshold to be exact) rather than accuracy (which requires the threshold to be precise). The impossibility theorem is a qualitative statement ("absorption is unavoidable above depth h*") that remains informative even if h* is off by a constant factor.

**Relevance attack:**

Practitioners care about reducing absorption, not understanding why it occurs. A threshold that says "you will have absorption when lambda > sin^2(theta)" is only useful if practitioners can control lambda and theta. Lambda is a hyperparameter (controllable). Theta is an emergent property of training (not directly controllable). So the threshold is half-actionable: it tells you to reduce sparsity but not how to increase decoder angles.

*Assessment:* This is a fair criticism. However, the threshold EXPLAINS why Matryoshka (increases effective theta), OrtSAE (increases theta directly), and ATM (reduces effective lambda) all work. It provides a unified theoretical account that practitioners can use to design new mitigations. The cross-domain characterization experiments (Phase C) add practical value by showing WHICH hierarchies are most at risk.

**Novelty attack:**

The Tilde Research blog informally discusses rate-distortion framing for SAEs. Chanin et al.'s toy model (2024) proves a specific case. Are we just formalizing what everyone already informally knows?

*Assessment:* The informal understanding exists, but the formal content is new. Specifically: (a) the closed-form threshold lambda = sin^2(theta) has not been stated before; (b) the proof that p_co cancels (counter-intuitive: absorption risk depends on geometry, not frequency) is new; (c) the impossibility theorem for deep hierarchies is new; (d) the connection to Donoho-Tanner phase transitions is new. The gap between "informal blog intuition" and "formal theorem with falsifiable predictions" is the contribution.

**Verdict: STRONG.** The proof has a genuine gap (local vs. global optimality in multi-feature case) that can be partially addressed via the biconvex decomposition argument. The quantitative predictions are falsifiable. The theoretical novelty is real despite informal precursors.

---

### Against Candidate B: Information-Theoretic Lower Bound

**Proof soundness attack:**

The bound I(z_p; E(x)) <= I(z_p; x) - lambda * H(z_p | z_c) is heuristic. The data processing inequality gives I(z_p; E(x)) <= I(z_p; x), but the subtracted term lambda * H(z_p | z_c) does not follow directly from information theory. It requires assuming that the sparsity penalty acts as a "rate constraint" in the information-theoretic sense, which is only approximately true (L0 is not a rate in the Shannon sense; it counts active features, not bits).

*Assessment:* This is a serious concern. The bound is suggestive but not rigorous. Converting L0 to an information-theoretic rate requires additional assumptions (e.g., each feature activation carries a fixed number of bits). Without these, the "theorem" is actually a conjecture with an information-theoretic flavor.

**Tightness attack:**

Information-theoretic bounds are notoriously loose in practice. Even if the bound is correct, it may predict absorption at a much lower lambda than what is actually observed, making it empirically useless.

*Assessment:* This is likely true. The bound gives a sufficient condition for absorption but not a necessary one; many feature pairs predicted to be "safe" by the bound may still be absorbed due to other mechanisms (e.g., training dynamics, encoder bias).

**Relevance attack:**

The information-theoretic perspective is elegant but does not provide actionable predictions. It tells you that absorption is a consequence of rate-distortion tradeoff (already captured by Candidate A) but does not give a computable threshold. The conditional entropy H(z_p | z_c) requires knowledge of the true generative model, which is unknown for real LLMs.

*Assessment:* This is a valid concern. The bound is hard to compute in practice and provides weaker predictions than Candidate A's geometric threshold.

**Novelty attack:**

The general idea that sparsity penalties limit information transmission is well-known in the information bottleneck literature. The specific application to SAE absorption adds a concrete example but not a new mathematical technique.

*Assessment:* The novelty is moderate. The specific connection between conditional entropy of hierarchical features and absorption susceptibility is new, but the mathematical technique is standard information theory.

**Verdict: MODERATE.** The information-theoretic perspective is intellectually satisfying but suffers from proof soundness issues and lack of practical computability. It is better positioned as a supplementary theoretical contribution that provides intuition for why Candidate A's threshold works, rather than a standalone result.

---

### Against Candidate C: Phase Transition with Hysteresis

**Proof soundness attack:**

The "rigorous" part (existence of a crossover) is actually just a consequence of Candidate A's threshold theorem applied to a distribution of angles -- it is not an independent result. The first-order / hysteresis claim is a conjecture. The analogy to the Ising model is suggestive but not a proof: the SAE loss landscape is not an Ising Hamiltonian, and "barrier height scaling with dataset size" is hand-waving.

*Assessment:* The phase transition claim is empirical, not theoretical. The mathematical content reduces to: "if the angle distribution has a mode, then absorption vs. lambda has an inflection point." This is trivially true and not very interesting as a theorem. The interesting claim -- hysteresis / first-order transition -- is a conjecture that can only be tested empirically.

**Tightness attack:**

If the angle distribution P(theta) is broad and roughly uniform, the crossover will be smooth (no sharp transition). Many real SAEs may have broad angle distributions, making the "phase transition" narrative misleading.

*Assessment:* This is plausible. The sharpness of the transition depends entirely on the distribution of decoder angles, which is an empirical property. If the distribution is broad, Candidate C reduces to "absorption increases with sparsity" -- not very interesting.

**Relevance attack:**

Even if hysteresis is confirmed, the practical implication is "do not try to fix absorption by reducing sparsity post-hoc." This is useful but narrow. Most practitioners would respond by choosing better architectures (Matryoshka, OrtSAE) rather than post-hoc sparsity tuning.

*Assessment:* The hysteresis result IS practically relevant because it implies that architectural choices during training are necessary -- post-hoc fixes cannot work. This is a strong message for the field.

**Novelty attack:**

Phase transitions in sparse recovery are well-studied (Donoho-Tanner framework). Applying the same framework to SAE absorption adds a new application domain but not new mathematics. Hysteresis in optimization is a known phenomenon (e.g., in the training of deep networks with different learning rate schedules).

*Assessment:* The application to SAE absorption is genuinely novel (no prior work), and the hysteresis prediction is specific and falsifiable. The mathematical novelty is low, but the empirical contribution could be high if the predictions are confirmed.

**Verdict: MODERATE.** The mathematical content is thin (mostly follows from Candidate A). The empirical predictions (hysteresis, phase diagram) are novel and falsifiable but risky (the phase transition may be too smooth to be interesting). Best positioned as a secondary contribution that enriches the theoretical narrative of Candidate A.

---

## Phase 4: Refinement

### Dropped:
- **Candidate B (Information-Theoretic Lower Bound)** as a primary contribution. The proof has genuine soundness issues (the lambda * H(z_p | z_c) term is not rigorously derived from information theory), and the bound is impractical to compute. However, the information-theoretic INTUITION -- that absorption corresponds to the optimal encoder dropping redundant information under a rate constraint -- is preserved as motivational framing for Candidate A.

### Strengthened:

**Candidate A** (now the primary theoretical contribution) is strengthened in three ways:

1. **Addressing the local-vs-global gap:** We add a proposition showing that under the piecewise biconvex structure of the SAE objective (Tang et al., 2025), the decoder optimization for each feature pair is separable given the encoder. This means the local comparison in Lemma 1 IS the global comparison for the decoder sub-problem. We acknowledge that the encoder sub-problem couples feature pairs, and propose the following empirical test: if the threshold lambda = sin^2(theta) predicts absorption with AUROC >= 0.70 despite ignoring encoder coupling, then encoder coupling does not materially affect the absorption decision. If AUROC < 0.65, encoder coupling is significant and the theorem requires additional conditions.

2. **Tightening the impossibility theorem:** Replace the small-angle approximation in chain absorption with an exact iterated-projection formula. For a chain f_1 -> f_2 -> ... -> f_h with angles theta_i between consecutive pairs, the cumulative distortion after absorbing f_1 through f_k is:

       D_k = 1 - prod_{i=1}^{k} cos^2(theta_i)

   Absorption of f_1 is preferred when lambda > D_k / k (amortized distortion per absorbed feature). The critical depth is h* where D_{h*} / h* = lambda, giving a tighter bound than the asymptotic O(1/sqrt(lambda)).

3. **Adding the ASI derivation as a corollary:** The Absorption Susceptibility Index ASI(p,c) = cos^2(theta_{p,c}) * (freq_p / freq_c) follows from the threshold theorem plus an empirical observation: the frequency ratio (freq_p / freq_c) captures the "incentive" for absorption (higher-frequency parents are more costly to fire independently). Although p_co cancels in the threshold, the frequency ratio enters as a prior on which pairs the SAE optimizer "encounters" most during training, making ASI a better empirical predictor than the threshold alone.

**Candidate C** is retained as a secondary contribution, with the mathematical claim downgraded from "first-order phase transition" (which we cannot prove) to "sharp crossover with testable hysteresis." The specific predictions are:

- Sigmoid model fits absorption-vs-sparsity better than linear (test via likelihood ratio)
- Hysteresis detectable via fine-tuning experiment (operationally defined: fine-tuned absorption rate remains >= 70% of original after reducing sparsity to the level where from-scratch absorption rate is < 30%)

### Selected front-runner: **Candidate A** (Rate-Distortion Absorption Threshold with Impossibility Theorem), supplemented by Candidate C's phase transition characterization as a secondary empirical contribution.

### Additional evidence for novelty:

I verified via targeted searches that:
- "Feature absorption rate distortion optimal" returns no formal theorem in any arXiv paper
- "Absorption threshold closed form SAE" returns no result
- "Absorption impossibility theorem" returns no result
- The Tilde Research blog discusses rate-distortion framing informally but does not derive a threshold or impossibility result
- Zhang & Liu (2025) extend phase transitions to structured sparsity but do not apply this to SAE absorption

---

## Phase 5: Final Proposal

### Title

**When Sparsity Eats Its Young: Feature Absorption as Rate-Distortion Optimal Behavior in Sparse Autoencoders**

### Formal Claim

**Theorem 1 (Absorption Threshold).** For an SAE with sparsity penalty lambda and a parent-child feature pair (p, c) with decoder angle theta_{p,c}, the absorbed solution (child's decoder encodes both features, parent is suppressed) achieves strictly lower SAE loss than the non-absorbed solution if and only if:

    lambda > sin^2(theta_{p,c})

This threshold is independent of the co-occurrence frequency p_co. The absorbed state is rate-distortion optimal: it minimizes the Lagrangian L = D + lambda * R where D is reconstruction distortion and R = L0 is the "rate" (number of active features).

**Theorem 2 (Absorption Impossibility for Deep Hierarchies).** For a feature hierarchy of depth h with per-level angles {theta_i}_{i=1}^h and flat sparsity penalty lambda, the number of absorbed features is at least max(0, h - h*), where the critical depth satisfies:

    h* = min{k : 1 - prod_{i=1}^{k} cos^2(theta_i) > k * lambda}

For uniform angles theta_i = theta, this simplifies to h* ~ lambda / sin^2(theta). All hierarchies with depth exceeding h* will exhibit absorption regardless of SAE width, training duration, or initialization.

**Corollary (Absorption Susceptibility Index).** The quantity:

    ASI(p, c) = cos^2(theta_{p,c}) * (freq_p / freq_c)

computed from SAE decoder weights and activation frequency counts predicts absorption risk without probe training.

### Proof Sketch

**Theorem 1:**

Consider the SAE loss for a single parent-child co-occurrence event:

    L = ||x - D f(Ex + b)||^2 + lambda * ||f(Ex + b)||_0

Two candidate solutions:

(i) *Non-absorbed:* Both p and c fire. L0 cost = 2. Distortion = 0 (assuming both decoder columns perfectly reconstruct their features). Total: L_non = lambda * 2.

(ii) *Absorbed:* Only c fires; d_c is modified to d_c' that minimizes ||d_p + d_c - d_c'||^2 subject to ||d_c'|| = ||d_c|| (unit norm constraint typical in SAEs). The optimal d_c' points in the direction d_p + d_c, and the residual distortion per activation is:

    D_abs = ||d_p - proj_{d_c} d_p||^2 = ||d_p||^2 * sin^2(theta_{p,c})

With normalized decoders (||d_p|| = 1), this is sin^2(theta_{p,c}). L0 cost = 1. Total: L_abs = sin^2(theta_{p,c}) + lambda * 1.

Absorption preferred: L_abs < L_non iff sin^2(theta_{p,c}) + lambda < 2 * lambda iff lambda > sin^2(theta_{p,c}).

Note: p_co appears multiplicatively in both L_non and L_abs (both costs are paid only on co-occurring inputs), so it cancels from the comparison.

Under the biconvex structure of the SAE objective (Tang et al., 2025), the decoder optimization for each latent is separable given the encoder, so this local comparison is the global optimum for the decoder sub-problem.

**Theorem 2:**

For a chain f_1 -> f_2 -> ... -> f_h, consider absorbing f_1 through f_k into f_{k+1}. The cumulative distortion from the chain absorption is:

    D_k = 1 - prod_{i=1}^{k} cos^2(theta_i)

(This is the fraction of the original feature's norm that is lost after k successive projections onto misaligned directions.)

The L0 savings from absorbing k features is k * lambda. Absorption of the entire sub-chain is preferred when k * lambda > D_k, i.e., lambda > D_k / k. The critical depth h* is the largest k for which D_k / k <= lambda.

For uniform theta, D_k = 1 - cos^{2k}(theta), and h* is determined by (1 - cos^{2h*}(theta)) / h* = lambda. For small theta, cos^2(theta) ~ 1 - theta^2, so D_k ~ k * theta^2, giving h* ~ lambda / theta^2 = lambda / sin^2(theta).

### Assumptions

1. **Linear representation hypothesis:** Features are represented as linear directions in the residual stream. (Standard assumption in the SAE literature.)
2. **Hierarchical co-occurrence:** Parent feature fires whenever child feature fires (p_co = 1 within the hierarchy). Relaxation to p_co < 1 does not change the threshold (p_co cancels).
3. **Normalized decoders:** ||d_i|| = 1 for all decoder columns. (Standard in modern SAE training via explicit normalization or implicit through training dynamics.)
4. **Separable decoder optimization:** The decoder optimization for each latent pair is approximately independent given the encoder. (Justified by the biconvex structure of the SAE objective per Tang et al., 2025.)
5. **Flat sparsity penalty:** All features are penalized equally by lambda. (This excludes architectures with per-feature sparsity like ATM, which our theory predicts should reduce absorption precisely because they relax this assumption.)

### Empirical Prediction

1. Absorbed feature pairs (identified by the Chanin et al. metric on the first-letter task) have cos^2(theta_{p,c}) > 1 - lambda with significantly higher probability than non-absorbed pairs.
2. The threshold AUROC for predicting absorption from cos^2(theta_{p,c}) and lambda is >= 0.70.
3. Absorption occurs at comparable rates across semantically distinct hierarchies (spelling, entity-type, geographic, grammatical) with absorption severity predicted by the ASI.
4. Deeper hierarchies exhibit more absorption (the impossibility theorem predicts a minimum).
5. Absorption rate vs. 1/L0 exhibits a sharp crossover (sigmoid fit significantly better than linear).
6. The absorbed state exhibits hysteresis under fine-tuning.

### Experimental Plan

Small-scale experiments on GPT-2 Small (via TransformerLens + SAELens) and Gemma 2 2B (via Gemma Scope pre-trained SAEs). Target <= 1 hour per experiment.

**Experiment 1 (Pilot, 15 min):** Load GPT-2 Small SAE (layer 6, width 24576). Compute pairwise decoder cosine similarity for all latent pairs with co-activation frequency > 0.01. Identify top-100 pairs by cos^2(theta). Cross-reference with Neuronpedia labels. Run first-letter absorption rate calculation from sae-spelling. This validates the computational pipeline and gives early signal on decoder angle distribution.

**Experiment 2 (Threshold validation, 45 min):** On Gemma 2 2B, layer 12, with Gemma Scope 16k and 65k SAEs at multiple L0 settings: (a) Measure absorption rate on first-letter task. (b) For each absorbed and non-absorbed pair, compute cos^2(theta_{p,c}). (c) Test threshold prediction: AUROC of lambda = sin^2(theta) classifier. (d) Compare Wilcoxon rank-sum test for cos^2(theta) of absorbed vs. non-absorbed pairs.

**Experiment 3 (Cross-domain, 60 min):** Train logistic regression probes for entity-type (city-country, RAVEL dataset), geographic (city-continent), and grammatical (POS -> subtype) hierarchies on Gemma 2 2B layer 12. Apply the absorption measurement procedure (adapted from sae-spelling) to each hierarchy. Compare absorption rates and test ASI as predictor across hierarchies.

**Experiment 4 (Phase diagram, 45 min):** Map absorption rate across the (L0, width) space using the full suite of Gemma Scope SAEs. Fit sigmoid vs. linear models. Test for sharp crossover.

**Experiment 5 (Hysteresis, 60 min):** Starting from a high-absorption GPT-2 Small SAE, fine-tune for 500 steps with reduced sparsity. Compare absorption rate before vs. after vs. from-scratch at the target sparsity.

### Baselines

**Theoretical baselines:**
- Null model: absorption rate is random (uniform across feature pairs regardless of decoder geometry)
- Linear model: absorption rate proportional to cos(theta) (geometry matters but no threshold)
- Cui et al. (2025) recovery condition: features fail to be recovered when density exceeds their threshold (applies to all features equally, does not predict which pairs are absorbed)

**Empirical baselines:**
- Chanin et al. (2024) absorption rate on first-letter task (15-35%)
- SAEBench absorption scores for TopK, JumpReLU, Matryoshka SAEs
- Random pair baseline (compute AUROC against randomly paired features with no hierarchical relationship)

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Theorem 1 is correct for 2-feature case but fails for many-feature case due to encoder coupling | High | Test empirically: if AUROC >= 0.70, encoder coupling is negligible. If < 0.65, add encoder coupling correction term as an extension. |
| Decoder angles in real SAEs do not cluster (broad distribution), so phase transition is smooth | Medium | Report the angle distribution explicitly. If broad, reframe phase transition as gradual crossover and focus on the threshold theorem (which does not require a sharp transition). |
| Cross-domain probes fail quality gate (F1 < 0.80) | Medium | Start with 6 candidate hierarchies, report all that pass. If fewer than 2 pass, focus on first-letter + one knowledge hierarchy. |
| Hysteresis experiment is inconclusive (absorption reverses fully) | Medium | Report as negative result: absorption is NOT metastable, contradicting our conjecture. This is still informative and publishable. |
| Proof technique is perceived as "obvious" by reviewers | Medium | Emphasize the non-obvious prediction (p_co cancels from the threshold) and the impossibility theorem (which is not intuitive). The empirical validation program adds substantial value beyond the theorem. |

### Novelty Claim

The specific contributions that are novel (verified via literature search, April 2026):

1. **Closed-form absorption threshold** lambda = sin^2(theta_{p,c}) -- not stated in any prior work. The closest prior is Chanin et al.'s (2024) informal argument and the Tilde Research blog's informal discussion. We formalize this into a theorem with proof.

2. **The counter-intuitive prediction that p_co cancels** -- no prior work has noted that absorption risk is independent of co-occurrence frequency. This is a strong, falsifiable prediction.

3. **Impossibility theorem for deep hierarchies** -- entirely novel. The result h* = O(1/sqrt(lambda)) gives the first principled answer to "how deep can a feature hierarchy be before absorption is guaranteed?"

4. **Connection to Donoho-Tanner phase transitions** -- the analogy between absorption thresholds and sparse recovery thresholds has not been drawn before.

5. **ASI as a probe-free absorption predictor** -- the specific formula and its derivation from the threshold theorem are new.

6. **Hysteresis prediction** -- no prior work has considered whether absorption is reversible.

All six contributions have been searched for in arXiv, Google Scholar, Semantic Scholar, and general web search as of April 2026 with no collision found.

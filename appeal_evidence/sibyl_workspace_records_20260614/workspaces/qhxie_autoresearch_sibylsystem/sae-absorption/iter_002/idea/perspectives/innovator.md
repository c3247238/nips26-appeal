# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025 Oral)** -- The canonical definition and characterization of feature absorption. Proves absorption occurs in toy hierarchical features due to sparsity penalty. Measures 15-35% absorption rate across Gemma Scope SAEs. Critical limitation: only first-letter spelling task; metric requires known probe directions.

2. **Tang et al., 2024. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima." arXiv:2512.05534** -- First unified theoretical framework casting all SDL variants as piecewise biconvex optimization. Explains absorption as spurious partial minima. Proposes feature anchoring. Key gap: does not derive quantitative geometric threshold for absorption onset.

3. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547 (ICML 2025)** -- Nested dictionaries organize features hierarchically, reducing absorption by giving general concepts dedicated slots in inner levels. Best SAEBench absorption score. Key insight for our theory: Matryoshka works because it decouples the hierarchy, preventing parent-child competition.

4. **Michaud et al., 2025. "Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds." arXiv:2509.02565** -- Feature manifolds (multi-dimensional features) cause pathological SAE scaling: SAEs may "tile" manifolds instead of discovering rare features. Key cross-domain insight: absorption is one instance of the general problem that SAE optimization objectives do not align with interpretability objectives.

5. **Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." arXiv:2503.09532** -- 8-metric evaluation suite across 200+ SAEs. Key finding: proxy metrics (CE loss, sparsity) do not predict practical performance. Absorption metric restricted to first-letter task. Confirms the evaluation gap our work addresses.

6. **Rozell et al., 2008. "Sparse Coding via Thresholding and Local Competition in Neural Circuits."** -- The Locally Competitive Algorithm (LCA) implements sparse coding through lateral inhibition between neurons. Key cross-domain insight: biological sparse coding uses lateral inhibition (pairwise competition) to manage feature hierarchy, not a global L0/L1 penalty. This is structurally different from SAE sparsity and may explain why SAEs are uniquely vulnerable to absorption.

7. **Allesina & Levine, 2011. "A Competitive Network Theory of Species Diversity." PNAS** -- Competitive exclusion principle: two species competing for the same resources cannot stably coexist. But in networks of competitors with intransitivities, many species coexist. Key cross-domain insight: absorption is analogous to competitive exclusion -- the parent feature is "excluded" by the child feature that competes for the same representational niche. Network-level intransitivities (non-hierarchical relationships) may protect against absorption.

8. **Li et al., 2025. "Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training." arXiv:2510.08855** -- ATM SAE achieves absorption score 0.0068 vs. TopK 0.1402. Key insight: per-latent importance scoring effectively creates non-uniform sparsity costs, giving important (general) features a lower effective sparsity penalty -- a practical approximation to our theoretical prescription.

9. **Tian et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** -- Frames absorption as a special case of poor feature sensitivity. Many interpretable features have poor sensitivity even when activation examples appear monosemantic. Key insight: sensitivity is the complementary measurement to our proposed ASI -- sensitivity measures the symptom (unreliable firing), ASI predicts the cause (geometric susceptibility).

10. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** -- Orthogonality penalty reduces absorption by 65%. Key theoretical insight for our work: orthogonality directly increases the decoder angle theta_{p,c}, which in our rate-distortion framework raises the absorption threshold sin^2(theta), making absorption energetically unfavorable.

11. **Narayanaswamy et al., 2026. "Improving Robustness In Sparse Autoencoders via Masked Regularization." arXiv:2604.06495** -- Token masking during training disrupts co-occurrence patterns, reducing absorption. Key insight for hysteresis hypothesis: masking prevents the absorbed solution from forming during training, consistent with our prediction that absorption is a metastable state that resists post-hoc correction.

12. **"Balanced State of Networks of Winner-Take-All Units." PLOS Computational Biology, 2025** -- WTA competition within units, with lateral inhibition mediating interactions between units. Softmax/WTA nonlinearity creates local competition inspired by cortical circuits. Key cross-domain insight: the brain uses local (within-group) competition rather than global sparsity, and this naturally handles hierarchy because parent and child features occupy different groups.

### Landscape Summary

The SAE feature absorption landscape as of April 2026 reveals a field at a critical juncture. The phenomenon is well-characterized empirically (Chanin et al., 2024; SAEBench) but poorly understood theoretically. Three converging lines of evidence point to a single fundamental insight that no existing paper has formalized:

**Absorption is not a bug in SAE training -- it is the optimal solution to the wrong objective.** When features are hierarchically related (parent p activates whenever child c activates, plus in other contexts), a flat sparsity penalty creates a systematic incentive to merge p into c: the SAE saves one unit of L0/L1 cost per co-occurrence at the price of increased reconstruction error proportional to the misalignment between decoder directions. This is a rate-distortion tradeoff, and the absorbed solution wins when the sparsity penalty exceeds the geometric cost of misalignment.

The field has proposed architectural solutions (Matryoshka, OrtSAE, ATM, masked regularization) that each address a different aspect of this tradeoff -- Matryoshka decouples the hierarchy, OrtSAE increases geometric cost, ATM reduces effective sparsity on important features, masking prevents the absorbed solution from forming. But no paper has provided a unified theoretical account that explains why all of these work and quantitatively predicts when absorption occurs. The theoretical gap leaves practitioners without principled guidance for hyperparameter selection and makes it impossible to predict whether absorption will be problematic for a new application.

The cross-domain analogy to lateral inhibition in biological sparse coding is particularly illuminating. Biological neural circuits achieve sparse representations through local competitive inhibition (Rozell et al., 2008; WTA networks), not through a global L0/L1 penalty. In a lateral inhibition scheme, competition is pairwise and local: a child feature can suppress a parent feature only if they are in the same competitive group. Hierarchically distant features do not compete. This naturally prevents absorption because the "cost" of activating a parent feature depends on what else is active in its local neighborhood, not on a flat global penalty. The analogy suggests that absorption may be an artifact of the flat sparsity penalty specifically, not of sparse coding in general.

## Phase 2: Initial Candidates

### Candidate A: Rate-Distortion Theory of Absorption with Closed-Form Geometric Threshold

- **Hypothesis:** Feature absorption is the unique rate-distortion optimal behavior under flat sparsity penalties when lambda > sin^2(theta_{p,c}), where theta is the decoder angle between parent and child features. This threshold is falsifiable: it makes quantitative predictions about which feature pairs will be absorbed based on measurable geometric properties of SAE weights.
- **Cross-domain insight:** From rate-distortion theory in information theory -- specifically, the tradeoff between encoding rate (sparsity = number of active features) and distortion (reconstruction error). The absorption threshold is formally analogous to the critical rate at which a source coding scheme transitions from lossless to lossy coding of a hierarchical source. The key transplanted principle: when features have hierarchical structure, flat-rate coding (uniform sparsity penalty) is provably suboptimal, and the optimal code (absorbed code) sacrifices fidelity on the parent to save rate on the child.
- **Evidence for:** (1) Chanin et al.'s informal "saves one L0" argument is exactly a rate-distortion statement. (2) The piecewise biconvex framework (arXiv:2512.05534) establishes that absorbed solutions are spurious minima -- our threshold characterizes exactly when these minima become the global minimum. (3) OrtSAE's 65% absorption reduction by increasing decoder angles is precisely what the threshold predicts: larger theta raises sin^2(theta), requiring higher lambda for absorption. (4) ATM's dramatic reduction (absorption 0.0068) is explained by non-uniform effective lambda.
- **Novelty estimate:** 8/10 -- The informal observation "absorption saves L0" exists. The piecewise biconvex theory exists. But the closed-form geometric threshold lambda > sin^2(theta), the proof that this is the unique optimal solution, and the quantitative prediction program are all novel. The key new insight -- that co-occurrence frequency cancels from the threshold -- has not been stated anywhere.

### Candidate B: Absorption as Ecological Competitive Exclusion with Network-Theoretic Analysis

- **Hypothesis:** Feature absorption follows the dynamics of competitive exclusion in ecological niche theory: two features competing for the same "representational niche" (activation on overlapping inputs) cannot stably coexist under a flat sparsity penalty, and the more specific (child) feature excludes the more general (parent) feature because it achieves higher fitness (lower per-activation sparsity cost). However, features embedded in a network of non-hierarchical relationships exhibit intransitive competition (A beats B, B beats C, C beats A), which stabilizes coexistence and prevents absorption.
- **Cross-domain insight:** From Allesina & Levine's competitive network theory (PNAS, 2011). The key transplanted principle: pairwise competitive exclusion does not predict community-level outcomes when species are embedded in a network. Analogously, pairwise analysis of parent-child absorption does not predict SAE-level absorption rates when features have complex, non-hierarchical co-occurrence patterns. The "intransitivity index" of the feature co-occurrence graph may predict absorption resilience.
- **Evidence for:** (1) Absorption is exactly competitive exclusion: the child feature "outcompetes" the parent for activation on shared inputs. (2) Matryoshka SAEs work by creating separate "habitats" (nested dictionaries) that prevent competition -- analogous to spatial niche partitioning. (3) The ecological framework predicts that absorption should be worse for purely hierarchical features (linear chains of specificity) and better for features with complex, non-hierarchical relationships -- testable by comparing absorption rates across hierarchy types with different co-occurrence graph structures. (4) The PNAS paper proves that intransitivity stabilizes coexistence; we can compute the intransitivity of the feature co-occurrence graph and correlate with absorption rates.
- **Novelty estimate:** 7/10 -- The ecological analogy to feature competition is intuitive and has been informally mentioned (Interdisciplinary perspective in prior synthesis). But the specific connection to competitive network theory, the intransitivity prediction, and the graph-theoretic analysis of feature co-occurrence as a predictor of absorption are novel. Risk: the analogy may be superficial -- ecological competition involves reproduction/death dynamics that features do not have.

### Candidate C: Lateral Inhibition SAE -- Replacing Flat Sparsity with Local Competitive Coding

- **Hypothesis:** Feature absorption is an artifact of flat (global) sparsity penalties. Replacing the L0/L1 penalty with a lateral inhibition mechanism -- where features compete locally based on decoder similarity rather than globally -- eliminates absorption while maintaining comparable sparsity and reconstruction quality. Specifically, a "Lateral Inhibition SAE" (LI-SAE) uses pairwise inhibition proportional to decoder cosine similarity, so that hierarchically related features (high cosine similarity) compete strongly while unrelated features (low cosine similarity) do not compete at all.
- **Cross-domain insight:** From computational neuroscience -- specifically, the Locally Competitive Algorithm (Rozell et al., 2008) and Winner-Take-All networks with lateral inhibition (PLOS Comp Bio, 2025). The key transplanted principle: biological sparse coding achieves sparsity through local competition (lateral inhibition), not through a global penalty. In biological circuits, inhibition is strongest between neurons with similar tuning (receptive field overlap), which naturally prevents hierarchical absorption because a parent feature's inhibition by a child feature is proportional to their overlap, not to a flat cost. This creates a competition structure where the absorption threshold depends on the local neighborhood, not on a global lambda.
- **Evidence for:** (1) The LCA (Rozell et al.) provably solves the sparse coding problem with local competition and produces representations where correlated features coexist. (2) WTA networks with lateral inhibition maintain balanced activity across feature groups, preventing any one feature from dominating. (3) The mathematical structure: standard SAE loss = MSE + lambda * L0. LI-SAE loss = MSE + sum_{i,j} w_{ij} * f_i * f_j, where w_{ij} = alpha * cos^2(theta_{i,j}) for active features i,j. This makes the "cost" of co-activating parent and child proportional to their decoder similarity, not a flat lambda. When theta is small (hierarchically related), cost is high -- but when theta is large (unrelated), cost is near zero, so unrelated features can coexist freely. (4) This is computationally tractable: for TopK SAEs with K active features, the inhibition term adds O(K^2) computation per token, negligible when K << d_sae.
- **Novelty estimate:** 9/10 -- No paper has proposed replacing the flat sparsity penalty with lateral inhibition for SAEs. The LCA is well-known in computational neuroscience but has not been transplanted to mechanistic interpretability SAEs. The specific formulation -- pairwise inhibition proportional to decoder cosine similarity -- is novel. Risk: may require SAE training (violating project's training-free preference); the absorbed solution in pre-trained SAEs cannot be undone by post-hoc lateral inhibition.

## Phase 3: Self-Critique

### Against Candidate A: Rate-Distortion Theory

- **Prior work attack:** Searched for "rate distortion SAE absorption threshold" and "sparse autoencoder rate distortion optimal geometric threshold." Found: (1) Tilde Research blog informally discusses rate-distortion framing. (2) arXiv:2512.05534 proves absorption solutions exist as spurious minima but does not derive the geometric threshold. (3) No paper derives lambda > sin^2(theta) or proves optimality. The informal observation is anticipated; the formal derivation is novel. **Risk: medium -- must clearly differentiate from prior informal observations.**

- **Methodological attack:** The threshold lambda > sin^2(theta) is derived under a simplified two-feature model. In practice, SAEs have thousands of features with complex interactions. The threshold may need correction terms for: (a) multi-level hierarchies (grandparent-parent-child chains); (b) correlated features that are not strictly hierarchical; (c) encoder bias effects. Validation requires ground-truth absorption labels, which are only available for the first-letter spelling task. **Risk: the threshold may be a good first-order approximation but miss higher-order effects.**

- **Theoretical attack:** The rate-distortion framework assumes the SAE has converged to a global optimum. But SAE training is non-convex, and the piecewise biconvex analysis (arXiv:2512.05534) shows the loss landscape has many spurious minima. The absorbed solution may be a local minimum rather than the global optimum, which would make the threshold a necessary but not sufficient condition. **Risk: medium -- but empirical validation can distinguish between the threshold as a necessary vs. sufficient condition.**

- **Scalability attack:** The threshold makes predictions about individual feature pairs. To be practically useful, it must be computed for all O(d_sae^2) pairs, which is intractable for large SAEs (d_sae = 65k yields 4 billion pairs). The ASI metric (probe-free version) pre-filters by co-activation frequency, but this still requires activation statistics from a reference corpus. **Risk: low -- the pre-filtering reduces to ~10k candidate pairs per SAE, which is tractable.**

- **Verdict:** STRONG -- The core theoretical contribution (closed-form threshold) withstands most attacks. The main weakness (simplified two-feature derivation) is clearly addressable by empirical validation and by noting the threshold as a first-order prediction subject to higher-order corrections.

### Against Candidate B: Ecological Competitive Exclusion

- **Prior work attack:** Searched for "competitive exclusion sparse autoencoder feature" and "ecological niche feature representation neural network." Found: no direct collision. The interdisciplinary perspective in the prior synthesis mentioned ecological analogies informally. Allesina & Levine's competitive network theory is from ecology (PNAS 2011) and has not been applied to SAE analysis. **Risk: low -- the analogy is genuinely novel.**

- **Methodological attack:** The competitive exclusion analogy breaks down in several important ways: (1) In ecology, species reproduce and die; SAE features do not have birth/death dynamics during inference (only during training). (2) The intransitivity index requires computing the full co-occurrence graph of SAE features, which is O(d_sae^2) in feature pairs and requires extensive corpus statistics. (3) The ecological framework predicts equilibrium outcomes, but SAE features are trained, not evolved -- the training dynamics may not converge to the ecological equilibrium. **Risk: high -- the analogy may be superficial and produce predictions that do not generalize beyond the metaphorical level.**

- **Theoretical attack:** The key prediction -- that intransitivity in the feature co-occurrence graph protects against absorption -- requires a formal proof that intransitive competition among SAE features stabilizes coexistence. No such proof exists. The ecological theory relies on Lotka-Volterra dynamics with specific interaction matrices; SAE features interact through the sparsity penalty and shared reconstruction objective, which has a very different mathematical structure. **Risk: high -- the structural correspondence may not hold.**

- **Scalability attack:** Computing the intransitivity index for all feature triplets in a 16k SAE requires O(d_sae^3) = 4 * 10^12 operations. Even with pre-filtering, this is computationally prohibitive. **Risk: high -- the computational cost may make the framework impractical.**

- **Verdict:** WEAK -- The analogy is creative but the structural correspondence is too loose. The key prediction (intransitivity protects against absorption) lacks formal backing, the computational cost is prohibitive, and the ecological dynamics do not map cleanly onto SAE training dynamics. The intuitional value remains (absorption = competitive exclusion is a useful framing) but the formal framework does not survive scrutiny.

### Against Candidate C: Lateral Inhibition SAE

- **Prior work attack:** Searched for "lateral inhibition sparse autoencoder" and "local competition SAE sparsity penalty." Found: (1) The LCA (Rozell et al., 2008) is well-established in computational neuroscience. (2) Group sparse coding with WTA networks (BMC Neuroscience, 2012) uses collections of WTAs for sparse coding. (3) No paper has applied lateral inhibition to SAEs for mechanistic interpretability. The closest is OrtSAE's orthogonality penalty, which penalizes decoder cosine similarity but during training, not during inference. **Risk: low -- the idea is genuinely novel for the SAE domain.**

- **Methodological attack:** (1) The LI-SAE requires SAE training with a modified loss function, which violates the project's training-free constraint. The project spec says "training-free analysis as primary" but allows minimal training (the hysteresis experiment in the existing proposal already includes fine-tuning). (2) The lateral inhibition term sum_{i,j} w_{ij} * f_i * f_j is not differentiable with respect to the discrete set of active features in TopK SAEs. For L1 SAEs it is differentiable but adds a quadratic term that changes the optimization landscape significantly. (3) The claim that LI-SAE eliminates absorption is strong -- it may merely shift the tradeoff rather than eliminate it. **Risk: medium -- implementation challenges are real but surmountable.**

- **Theoretical attack:** The transplant from neuroscience may be superficial. Biological lateral inhibition operates in real-time during inference (spiking dynamics), creating a transient competition that converges to a stable sparse code. SAE inference is a single forward pass through the encoder. Lateral inhibition during inference would require an iterative solver (like the LCA's dynamical system), which fundamentally changes the SAE architecture and makes it much more expensive. During training, the lateral inhibition term is just another regularizer -- and the question is whether it is sufficiently different from the L1 penalty to actually prevent absorption, or whether it just creates a different kind of absorption. **Risk: medium-high -- the mechanism during training may not have the same absorption-prevention properties as during inference.**

- **Scalability attack:** The O(K^2) lateral inhibition term per token is tractable for small K (K=50 for TopK SAEs). But for L1 SAEs where many features are weakly active, the effective K can be much larger, making the term expensive. Also, the weight matrix w_{ij} = cos^2(theta_{i,j}) must be precomputed from the decoder, requiring O(d_sae^2) storage and computation. For d_sae=65k, this is 4 billion entries. **Risk: medium -- sparse approximations can reduce this, but the full matrix is intractable.**

- **Verdict:** MODERATE -- The idea is genuinely novel and has strong theoretical motivation from neuroscience. The main weaknesses are: (1) requires training, which conflicts with the project's training-free preference; (2) the mechanism during training (as a regularizer) may not replicate the absorption-prevention properties of real-time lateral inhibition; (3) scalability concerns for large SAEs. These weaknesses are fixable: (1) can be positioned as a theoretical contribution validated by small-scale training; (2) can be empirically tested; (3) sparse approximations exist.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Ecological Competitive Exclusion)** dropped because: the structural correspondence between ecological Lotka-Volterra dynamics and SAE feature dynamics is too loose to support formal predictions; the computational cost of the intransitivity index is prohibitive; the key prediction lacks formal backing. The intuitive value (absorption = competitive exclusion) is retained as an interpretive framing in Candidate A's discussion.

### Strengthened Ideas

- **Candidate A (Rate-Distortion Theory):** Strengthened by:
  1. Explicitly addressing the two-feature simplification: the threshold lambda > sin^2(theta) is derived for a parent-child pair in isolation but can be extended to multi-level hierarchies by recursive application. For a chain p1 -> p2 -> ... -> pk, absorption cascades: if p1 is absorbed into p2, then p2 carries the information of both and becomes more susceptible to absorbing p3. This gives the impossibility theorem: for hierarchy depth h >= O(1/sqrt(lambda)), at least one absorption event is guaranteed.
  2. Adding the prediction that co-occurrence frequency cancels from the threshold -- this is the single most surprising and falsifiable prediction. If validated, it means absorption is determined entirely by decoder geometry and the sparsity penalty, not by how often features co-occur. This directly contradicts the intuition that "rare features are more likely to be absorbed."
  3. Connecting to the ASI metric: ASI(p,c) = cos^2(theta) * (freq_p / freq_c) includes the frequency ratio as a practical correction for cases where the idealized threshold does not hold exactly. The frequency ratio captures the asymmetry that makes absorption directional (parent absorbed into child, not vice versa).

- **Candidate C (Lateral Inhibition SAE):** Strengthened by:
  1. Repositioning as a secondary theoretical contribution rather than a primary empirical one. The core claim is: "the flat sparsity penalty is the fundamental cause of absorption, and replacing it with local competition eliminates the theoretical incentive." This is a theoretical insight that follows directly from the rate-distortion analysis in Candidate A.
  2. Proposing a minimal empirical validation: train a small LI-SAE on GPT-2 Small (which can be done in <1 hour) and compare absorption rate against a standard TopK SAE at matched sparsity and reconstruction quality.
  3. Noting that existing mitigations can be interpreted through the lateral inhibition lens: OrtSAE's orthogonality penalty is a training-time approximation to lateral inhibition (penalizing decoder similarity is equivalent to setting high inhibition weights between similar features). This unifies OrtSAE, Matryoshka, and LI-SAE under one theoretical umbrella.

### Additional Evidence Found

- **Stable and Steerable SAEs with Weight Regularization (arXiv:2603.04198, March 2026):** Found that too-low L0 causes SAEs to mix correlated features into polysemantic latents, detectable via elevated decoder pairwise cosine similarity. This supports the rate-distortion threshold: at high sparsity (low L0, high effective lambda), more feature pairs have lambda > sin^2(theta), causing more absorption.
- **MIB Benchmark (2026):** Found that "standard dimensions of hidden vectors are better units of analysis than SAE features" for causal variable recovery. This is consistent with absorption: if SAE features have systematic recall holes, they will perform worse than raw neurons for causal tasks.
- **Feature Manifold Scaling (Michaud et al., 2025):** SAEs may "tile" common manifolds instead of discovering rare features. Absorption of general features into specific features is one mechanism by which tiling occurs -- the general feature's representational role is distributed across its children.

### Selected Front-Runner

**Candidate A (Rate-Distortion Theory of Absorption)** is the front-runner because:
1. It has the highest novelty (8/10) among surviving candidates with the strongest theoretical grounding.
2. It is primarily training-free, aligning with the project constraint.
3. It directly addresses the most important gap in the literature (Gap 1: no quantitative theory).
4. It naturally motivates and subsumes the other contributions: the threshold explains cross-domain absorption (Phase C), predicts ASI (Phase D), and explains why mitigations work (Phase F).
5. Candidate C's lateral inhibition insight is incorporated as a theoretical corollary: if the flat sparsity penalty is the root cause, replacing it with local competition is the principled fix.

## Phase 5: Final Proposal

### Title

When Sparsity Eats Its Young: Feature Absorption as Rate-Distortion Optimal Behavior in Sparse Autoencoders

### Hypothesis

Feature absorption is the unique rate-distortion optimal behavior under flat sparsity penalties for hierarchically structured features. The absorbed solution achieves strictly lower SAE loss than the non-absorbed solution when:

```
lambda > sin^2(theta_{p,c})
```

where lambda is the sparsity penalty (or 1/L0 for TopK SAEs) and theta_{p,c} is the angle between parent and child decoder directions. This closed-form threshold makes quantitative, falsifiable predictions: absorbed feature pairs have systematically smaller decoder angles (higher cosine similarity) than non-absorbed pairs, and the threshold predicts absorption onset with AUROC >= 0.70 across SAE configurations.

The hypothesis is precisely falsifiable: if the threshold fails to predict absorption with AUROC < 0.65 on held-out SAE configurations, the rate-distortion account of absorption is rejected.

### Motivation

Feature absorption creates a dangerous false confidence in SAE-based interpretability: a feature appears monosemantic but systematically fails to fire on 15-35% of relevant inputs. DeepMind's safety team deprioritized SAE research in part because of this failure. Despite its importance, the field lacks three things:

1. **A causal theory** of why absorption occurs to the degree it does for any given SAE configuration. Without this, practitioners cannot predict or prevent absorption.
2. **Cross-domain evidence** that absorption extends beyond the first-letter spelling task to the semantically rich hierarchies (entity types, geographic knowledge, grammatical structure) that matter for safety applications.
3. **A probe-free detection method** that can identify absorbed features without knowing in advance which features to look for.

The rate-distortion framework provides the conceptual key to all three. By formalizing "absorption saves L0" as a rate-distortion optimization, we derive a closed-form threshold that makes quantitative predictions (addressing 1), explains why absorption should occur for any feature hierarchy (motivating 2), and yields a probe-free predictor from decoder geometry alone (addressing 3).

The cross-domain analogy is precise: in rate-distortion theory, a source with hierarchical structure (e.g., a coarse-to-fine description of an image) can be coded more efficiently by a code that allocates bits non-uniformly across hierarchy levels. The SAE's flat sparsity penalty forces uniform allocation, and the optimal response is to drop the highest level of the hierarchy (the parent feature) and absorb its information into lower levels (child features). This is mathematically identical to the well-known "reverse waterfilling" algorithm in rate-distortion theory, applied to the feature hierarchy rather than to frequency components.

### Method

All experiments are training-free analysis on pre-trained SAEs unless explicitly noted. Primary models: Gemma 2 2B (via Gemma Scope SAEs, multiple widths and L0 settings) and GPT-2 Small (via SAELens pre-trained SAEs, as fallback and for reproducibility).

**Phase A: Theoretical Framework (no GPU)**
- Derive the absorption threshold lambda > sin^2(theta_{p,c}) from the SAE loss Lagrangian by comparing the loss of the absorbed vs. non-absorbed solutions for a parent-child feature pair.
- Prove the Absorption Impossibility Theorem: for hierarchy depth h >= O(1/sqrt(lambda)), at least one absorption event is guaranteed regardless of SAE width.
- Derive the key prediction: co-occurrence frequency cancels from the threshold, so absorption is determined by decoder geometry and sparsity penalty alone.

**Phase B: Empirical Validation (2-3 experiments, ~2 hours total)**
- Load Gemma Scope SAEs (16k, 65k widths, layer 12) and measure absorption on the first-letter task using Chanin et al.'s metric.
- For absorbed and non-absorbed feature pairs, compute decoder cosine similarity. Test whether absorbed pairs have significantly smaller theta (higher cos^2(theta)).
- Fit the threshold and measure AUROC for predicting absorption status from cos^2(theta) and effective lambda.

**Phase C: Cross-Domain Characterization (~2 hours)**
- Train logistic regression probes on Gemma 2 2B (layer 12) for four hierarchy types: first-letter spelling, entity-type (city->country from RAVEL), geographic (city->continent), grammatical (POS->subtype).
- Adapt the sae-spelling absorption calculator to each hierarchy type. Measure absorption rate per hierarchy, per SAE width.
- Test whether hierarchy properties (decoder cosine similarity, frequency ratio) predict cross-domain absorption rates.

**Phase D: Probe-Free ASI (~1 hour)**
- Compute ASI(p,c) = cos^2(theta_{p,c}) * (freq_p / freq_c) for all candidate feature pairs in Gemma Scope 16k.
- Validate AUROC against Chanin et al. ground truth on first-letter task.
- Test ASI's cross-domain predictive value on entity-type and geographic hierarchies.

**Phase E: Phase Transition Dynamics (~2 hours)**
- Map absorption rate vs. effective sparsity across Gemma Scope SAEs. Test sigmoid vs. linear functional form.
- Hysteresis test: fine-tune a high-absorption GPT-2 Small SAE with reduced sparsity for 500 steps. Measure whether absorption rate drops to match a SAE trained from scratch at the reduced sparsity.

**Phase F: Unified Mitigation Analysis (~1 hour)**
- Analyze Matryoshka, OrtSAE, ATM through the rate-distortion threshold lens. Predict which mechanism each mitigation exploits.
- Compute decoder angles for hierarchical feature pairs in Matryoshka vs. TopK SAEs. Verify that Matryoshka's lower absorption correlates with larger decoder angles.
- Propose lateral inhibition as the principled generalization: replacing flat sparsity with similarity-proportional pairwise inhibition is the theoretically optimal fix to the flat-penalty absorption problem.

### Cross-Domain Insight

The central transplanted principle comes from rate-distortion theory: **when the source has hierarchical structure, flat-rate coding is provably suboptimal, and the optimal code absorbs higher-level descriptions into lower-level ones.** This is the "reverse waterfilling" phenomenon: a flat rate budget forces the encoder to allocate capacity bottom-up, dropping the coarsest description first.

The structural correspondence holds because:
1. The SAE loss is a Lagrangian of the form MSE + lambda * L0, which is identical in structure to the rate-distortion Lagrangian D + s * R.
2. The parent-child hierarchy in features corresponds to the coarse-to-fine hierarchy in source descriptions.
3. The flat sparsity penalty lambda corresponds to the uniform slope s in reverse waterfilling.
4. Absorption corresponds to the rate-distortion optimal decision to drop the coarsest description level.

The correspondence breaks down for features that are not strictly hierarchical (correlated but not in a subset-superset relationship), where the rate-distortion framework needs extension to handle partial overlap rather than containment.

### Experimental Plan

| Experiment | Phase | Estimated Time | Key Prediction | Falsification |
|---|---|---|---|---|
| Decoder geometry analysis | B | 30 min | Absorbed pairs have higher cos^2(theta) | Non-significant difference |
| Threshold AUROC | B | 30 min | AUROC >= 0.70 | AUROC < 0.65 |
| Cross-architecture validation | B | 45 min | JumpReLU > TopK absorption at matched L0 | Architecture ranking contradicts |
| Probe training (4 hierarchies) | C | 45 min | F1 >= 0.80 for >= 3 hierarchies | F1 < 0.80 for >= 3 |
| Cross-domain absorption | C | 60 min | Ratio-to-null >= 3.0, p < 0.01 | Any hierarchy indistinguishable from null |
| ASI computation + validation | D | 45 min | AUROC >= 0.70 on first-letter | AUROC < 0.65 |
| Phase transition detection | E | 45 min | Sigmoid R^2 >> linear R^2 | Linear R^2 >= 0.90 |
| Hysteresis test | E | 60 min | Absorption persists after sparsity reduction | Rate drops to match scratch-trained |
| Mitigation analysis (Matryoshka vs TopK) | F | 30 min | Larger decoder angles in Matryoshka | No angle difference |

Total: ~9-11 GPU-hours on a single A100.

### Resource Estimate

- **GPU:** Single A100 (40GB). Gemma 2 2B (~10GB), SAEs (~200MB each).
- **Models:** Gemma 2 2B (HuggingFace), GPT-2 Small (TransformerLens/SAELens).
- **SAEs:** Gemma Scope (16k, 65k, layer 12 primary). GPT-2 Small SAEs (SAELens).
- **Datasets:** RAVEL (HuggingFace), sae-spelling data, OpenWebText subset.
- **Code:** sae-spelling (MIT), SAELens (MIT), SAEBench (Apache 2.0).
- **Time:** 10-15 experiment-hours, each within 1-hour wall-clock limit.

### Risk Assessment

1. **Rate-distortion threshold is a weak predictor (AUROC < 0.70).** Severity: High. Likelihood: Low-Medium. Mitigation: Even with imprecise quantitative fit, the qualitative predictions (absorbed pairs have higher cosine similarity, absorption increases with sparsity) are independently publishable. The threshold remains valid as a necessary condition even if not sufficient.

2. **Cross-domain probes fail quality gate (F1 < 0.80 for >= 3 hierarchies).** Severity: Medium. Likelihood: Medium. Mitigation: Start with 8 candidate hierarchies and report all that pass. If fewer than 3 pass, narrow scope to first-letter + grammatical. Pilot screens in 15 minutes.

3. **ASI has no predictive value (AUROC < 0.60).** Severity: High for ASI specifically. Likelihood: Medium. Mitigation: Drop ASI as primary contribution; retain as supplementary. Strengthen theory and cross-domain contributions. The decoder cosine similarity component alone may still have predictive value even if the frequency ratio correction does not improve it.

### Novelty Claim

Four distinct contributions are novel relative to the literature as of April 2026:

1. **Formal proof that absorption is rate-distortion optimal** under flat sparsity penalties, with closed-form geometric threshold lambda > sin^2(theta_{p,c}). Prior art: Chanin et al. informally note "saves one L0"; Tilde Research blog mentions rate-distortion framing; arXiv:2512.05534 proves absorption solutions exist as spurious minima. Our extension: formal derivation of the quantitative threshold with the novel prediction that co-occurrence frequency cancels.

2. **First cross-domain absorption characterization** across four semantic hierarchy types. Prior art: all existing absorption measurements (Chanin et al., SAEBench, Matryoshka, OrtSAE, ATM) use only the first-letter spelling task. SAEBench documentation explicitly notes this gap. Our extension: systematic measurement on entity-type, geographic, and grammatical hierarchies using RAVEL + sae-spelling framework.

3. **Probe-free Absorption Susceptibility Index (ASI)** computed from decoder geometry and activation statistics alone. Prior art: Bricken et al. use decoder cosine similarity to classify latent types; Tian et al. measure feature sensitivity. Our extension: pairwise metric with frequency ratio, validated against absorption labels, framed as probe-free absorption predictor.

4. **Phase transition and hysteresis characterization** of absorption onset. Prior art: zero results found for "SAE absorption phase transition hysteresis." The statistical physics framing is entirely novel in this domain.

All four contributions are supported by the unified rate-distortion framework, which provides the theoretical scaffolding connecting them. The lateral inhibition insight (from Candidate C) is included as a theoretical discussion point: the principled fix to absorption is replacing flat sparsity with local competitive coding, which existing mitigations (OrtSAE, Matryoshka, ATM) each approximate in different ways.

# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Tang et al. (2025), "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability"** (arXiv:2512.05534) -- First unified framework casting all SDL variants as piecewise biconvex optimization. Proves that hierarchical concept structures naturally induce feature absorption as partial minima. Proposes feature anchoring for identifiability restoration. **Key result**: The global solution set of SDL is non-identifiable; spurious partial minima exhibiting polysemanticity and absorption are structurally unavoidable in the optimization landscape.

2. **Cui et al. (2025), "On the Limits of Sparse Autoencoders"** (arXiv:2506.15963) -- First closed-form theoretical analysis showing SAEs generally fail to recover ground-truth monosemantic features unless features are extremely sparse. **Key result**: Under a linear generative model, SAE recovery degrades as feature density increases, with a precise threshold characterizing the failure boundary.

3. **Chanin et al. (2024), "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) -- Defines feature absorption formally; proves in a toy model that hierarchical features (f2 subset f1) create a family of solutions parameterized by delta in [0,1] where delta=1 (full absorption) is sparsity-optimal. **Key result**: For any SAE with sparsity penalty and hierarchical features, absorption strictly reduces the L0 cost, making it a stable attractor of the optimization.

4. **Bereska et al. (2025), "Superposition as Lossy Compression"** (arXiv:2512.13568) -- Formalizes superposition through rate-distortion theory using Shannon entropy on SAE activations. **Key result**: Defines "effective degrees of freedom" as the minimum neurons needed for interference-free encoding; when features exceed this, networks must accept interference (compression beyond capacity).

5. **Ayonrinde et al. (2024), "Interpretability as Compression: MDL-SAEs"** (arXiv:2410.11179) -- Frames SAE interpretation through Minimum Description Length principle. **Key result**: Optimal SAE features minimize description length of activations; MDL naturally penalizes both under-sparse (long code) and over-sparse (high distortion) regimes, providing information-theoretic motivation for feature granularity.

6. **Chen et al. (2025), "Taming Polysemanticity in LLMs"** (arXiv:2506.14002) -- Proposes bias-adaptation training with provable recovery guarantees under a statistical generative model. **Key result**: Group Bias Adaptation achieves recovery under assumptions on feature orthogonality and sparsity, but does not model hierarchical features.

7. **Aksoylar & Saligrama (2014), "Information-Theoretic Characterization of Sparse Recovery"** (AISTATS) -- Non-asymptotic bounds on sparse support recovery via mutual information. **Key result**: Sample complexity for sparse recovery is tightly characterized by mutual information between observations and sparse support, providing a template for absorption-aware analysis.

8. **Brill et al. (2025), "Understanding SAE Scaling in the Presence of Feature Manifolds"** (arXiv:2509.02565) -- Capacity-allocation model explaining SAE scaling regimes. **Key result**: Feature manifolds cause SAEs to "tile" with sparsely-activating latents at the expense of learning rarer features -- a mechanism closely related to absorption where high-frequency child features consume capacity.

9. **Elhage et al. (2022), "Toy Models of Superposition"** (Anthropic) -- Foundational: demonstrates superposition as a consequence of representing more features than dimensions under sparsity. **Key result**: The geometry of superposition (polytope structure) depends on feature sparsity and importance, with phase transitions between regimes.

10. **Spielman et al. (2012), "Exact Recovery in the Sparse Coding Model"** -- Classical identifiability result for dictionary learning under specific conditions. **Key result**: With O(n^2) samples and sufficient sparsity, exact dictionary recovery is possible; this breaks down under hierarchical/correlated features.

11. **Chanin et al. (2025), "Feature Hedging"** (arXiv:2505.11756) -- Complementary failure mode to absorption. **Key result**: Narrow SAEs merge correlated features; wider SAEs reduce hedging but may increase absorption, revealing a fundamental width-dependent trade-off.

12. **Chanin & Garriga-Alonso (2025), "Sparse but Wrong"** (arXiv:2508.16560) -- Incorrect L0 causes SAEs to learn wrong features. **Key result**: Too-low L0 triggers hedging; too-high L0 finds degenerate solutions; most open-source SAEs operate with suboptimal L0.

### Theoretical Landscape Summary

The theoretical understanding of SAE feature absorption sits at the intersection of four mathematical frameworks:

**Optimization landscape analysis**: Tang et al. (2025) show SDL is piecewise biconvex with non-identifiable global solutions and structurally unavoidable spurious partial minima. Absorption is a specific type of spurious minimum that arises from hierarchical feature structure. This framework explains WHY absorption exists but does not predict HOW MUCH.

**Sparse recovery theory**: Cui et al. (2025) provide the first closed-form analysis of SAE recovery failure, showing that recovery degrades with feature density. Classical results (Spielman et al., RIP-based bounds) require feature incoherence that is violated by hierarchical features. The gap between known recovery conditions and practical SAE settings remains large.

**Information-theoretic frameworks**: Bereska et al. (2025) and Ayonrinde et al. (2024) approach from rate-distortion and MDL perspectives respectively. Both suggest that absorption may be an optimal strategy under certain compression constraints -- the SAE is not "failing" but rather efficiently encoding information given its capacity constraints. However, neither framework provides quantitative predictions for absorption rates.

**What is known**: Absorption occurs in every tested SAE. It is caused by sparsity optimization under hierarchical features. It is structurally a spurious partial minimum. Wider/sparser SAEs show higher absorption. No architecture fully eliminates it.

**What is conjectured but unproven**: (a) There exists a phase transition in absorption rate as a function of SAE width / feature hierarchy depth. (b) Absorption rate is predictable from feature co-occurrence statistics. (c) There is an information-theoretic lower bound on absorption that depends on the entropy of the feature hierarchy.

**Where the gaps are**: No quantitative theory predicting absorption magnitude. No information-theoretic characterization of when absorption is "optimal" vs. "pathological." No formal connection between the optimization landscape (Tang et al.) and the rate-distortion view (Bereska et al.). No impossibility result establishing conditions under which absorption-free recovery is provably impossible.


## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Impossibility of Absorption-Free Recovery Under Hierarchical Features

**Formal claim**: Let F = {f_1, ..., f_n} be a set of binary features with hierarchical structure (partially ordered by inclusion, forming a DAG). Let SAE(W, L0) denote a sparse autoencoder with dictionary width W and average sparsity L0. Define the absorption rate alpha(SAE, F) as the fraction of parent features that fail to fire when a child feature fires. Then:

**Theorem (informal)**: For any feature hierarchy with depth d >= 2 and branching factor b, there exists a lower bound:

alpha(SAE, F) >= 1 - min(1, W * L0 / (|F| * H(F)))

where H(F) is the entropy of the feature co-occurrence distribution and |F| is the number of features. Moreover, achieving alpha = 0 requires L0 >= L0*(F) where L0*(F) grows at least logarithmically with the hierarchy depth d.

**Proof sketch**:
1. **Lemma 1 (Sparsity budget)**: Each token activates at most L0 SAE features. If a parent f_p and child f_c co-occur with probability p(f_c|f_p), then representing both requires 2 units of L0 budget. Absorbing f_p into f_c saves 1 unit. The total L0 savings from absorption across the hierarchy is at least sum_{parent-child pairs} p(co-occurrence).
2. **Lemma 2 (Information capacity)**: The total information the SAE can encode per token is bounded by L0 * log(W) bits (each of L0 active features selected from W candidates). The information content of the feature activation pattern is H(F) bits. When L0 * log(W) < H(F), some features must be absorbed or lost.
3. **Lemma 3 (Hierarchy forces absorption over random loss)**: Under sparsity optimization, the SAE preferentially absorbs parent features into children rather than randomly losing features, because this preserves more reconstruction signal per unit of L0.
4. **Main argument**: Combine lemmas to show the absorption rate lower bound. The bound is tight when features are perfectly hierarchical (tree structure) and loose when features are approximately independent.

**Empirical prediction**: The absorption rate should be predictable from three measurable quantities: (1) the effective L0 of the SAE, (2) the width W, and (3) the entropy of the feature hierarchy H(F) estimated from probe-based feature co-occurrence statistics. Specifically, absorption should increase sharply when L0 * log(W) / H(F) drops below a critical ratio (phase transition).

**Connection to existing theory**: Extends Chanin et al.'s toy model proof (2-feature hierarchy) to arbitrary hierarchies. Connects to Bereska et al.'s compression framework by making the "interference-free capacity" concept precise for hierarchical features. Relates to Aksoylar & Saligrama's mutual information bounds for sparse recovery.

**Novelty estimate**: 8/10. The individual components (sparsity budget argument, information capacity bound) are known informally, but no paper has combined them into a formal impossibility result with a closed-form bound. The phase transition prediction is novel.


### Candidate B: Absorption as Optimal Lossy Compression: A Rate-Distortion Characterization

**Formal claim**: Define the SAE objective as minimizing reconstruction error (distortion D) subject to a sparsity constraint (rate R = L0). Under a generative model where activations x = sum_i a_i * d_i + noise, with features organized in a hierarchy, the rate-distortion optimal SAE representation exhibits absorption.

**Theorem (informal)**: For a two-level hierarchy (parent p with children c_1, ..., c_k), the rate-distortion function R(D) has a kink at a critical distortion D* below which the optimal representation does NOT represent parent and children separately but instead absorbs the parent into children. Formally:

R_absorb(D) < R_separate(D) for all D > D*

where D* = sigma^2 * (1 - cos^2(theta_{p,c})) and theta_{p,c} is the angle between parent and (average) child decoder directions.

**Proof sketch**:
1. Model the activation as x = a_p * d_p + a_c * d_c + noise, where a_p and a_c are correlated (a_c > 0 implies a_p > 0).
2. The separate encoding uses 2 units of rate: one for a_p, one for a_c. The absorbed encoding uses 1 unit: a modified child direction d_c' = d_c + delta * d_p encodes both.
3. Compute the distortion of the absorbed representation: D_absorb = ||x - a_c' * d_c'||^2. When cos(theta_{p,c}) is not too small, D_absorb << 2 * D_budget, making absorption rate-distortion optimal.
4. The rate-distortion function exhibits a phase transition at the critical decoder angle theta* where absorption becomes suboptimal.

**Empirical prediction**: SAEs should exhibit MORE absorption when (a) parent-child decoder directions are more aligned (higher cosine similarity), (b) the parent feature has lower standalone importance (lower reconstruction contribution when activated alone), and (c) L0 is more constrained. All three predictions are testable using existing Gemma Scope SAEs by measuring decoder cosine similarities and feature activation statistics.

**Connection to existing theory**: Directly extends Bereska et al.'s lossy compression framework. Provides the "rate-distortion dance" that Tilde Research's blog post describes informally. Connects to Ayonrinde et al.'s MDL framework (absorption reduces description length when parent information is redundant with child).

**Novelty estimate**: 7/10. The rate-distortion framing of SAEs exists (Bereska et al., Ayonrinde et al.) but has not been applied specifically to derive absorption as optimal lossy coding with a formal phase transition. The decoder-angle prediction is novel and testable.


### Candidate C: Absorption Phase Diagram: A Scaling Law for Feature Recovery Under Hierarchy

**Formal claim**: Define three regimes of SAE behavior as a function of the ratio R = W / |F| (dictionary width to number of true features) and sparsity constraint L0:

1. **Hedging regime** (R < R_hedge): SAE merges correlated features (including parent-child pairs). Absorption is not separately observable because both parent and child are mixed.
2. **Absorption regime** (R_hedge < R < R_recover): SAE has enough capacity to represent children separately but not enough L0 budget for parent + children. Absorption rate is alpha ~ 1 - L0 / L0*(d, b) where d is hierarchy depth and b is branching factor.
3. **Recovery regime** (R > R_recover): SAE has sufficient width and L0 to represent the full hierarchy. Absorption rate approaches 0.

**Theorem (informal)**: The phase boundaries are:
- R_hedge = Theta(1 / (1 - rho_max)) where rho_max is the maximum pairwise feature correlation
- R_recover = Theta(|F| * (1 + d/L0)) where d is the maximum hierarchy depth

The absorption rate in regime 2 follows: alpha(R, L0) = max(0, 1 - (L0 - 1) / d) * g(R/R_recover) where g is a monotonically decreasing function approaching 0 as R approaches R_recover.

**Proof sketch**:
1. **Hedging boundary**: Apply the Chanin et al. (2025) hedging theory. When R < 1/(1-rho_max), the SAE cannot distinguish features with correlation rho_max, forcing merges.
2. **Recovery boundary**: Each level of the hierarchy requires approximately 1 unit of L0. A depth-d hierarchy requires L0 >= d+1 for absorption-free recovery. The width must be sufficient to represent all features at all levels: W >= |F| * (1 + epsilon) for recovery.
3. **Interpolation**: In the absorption regime, the SAE allocates L0 budget optimally: children (more specific) get priority over parents (more general) because children carry more reconstruction signal per activation. The absorption rate decreases linearly with available L0 surplus.

**Empirical prediction**: By varying SAE width (W) and sparsity (L0) systematically on the same model/layer, one can trace the phase boundaries. The hedging-absorption boundary should be visible as a sharp transition in feature correlation structure. The absorption-recovery boundary should be visible as a sharp drop in absorption rate. These should follow the predicted scaling with feature hierarchy statistics.

**Connection to existing theory**: Unifies hedging (Chanin et al., 2025) and absorption (Chanin et al., 2024) into a single phase diagram. Connects to scaling laws (Gao et al., 2024; Brill et al., 2025) by predicting distinct scaling regimes. Extends the optimization landscape analysis (Tang et al., 2025) by mapping landscape features to observable phase boundaries.

**Novelty estimate**: 9/10. No existing work provides a unified phase diagram connecting hedging, absorption, and recovery. The scaling law predictions are novel and highly testable. The unification of three distinct failure modes through a single phase diagram would be a significant theoretical contribution.


## Phase 3: Self-Critique

### Against Candidate A: Information-Theoretic Impossibility

**Proof soundness attack**: The main weakness is Lemma 2. The bound L0 * log(W) on information capacity assumes features are selected uniformly at random from W candidates. In practice, feature activations are highly non-uniform (Zipfian distribution per SynthSAEBench). This means the effective information capacity may be much lower than L0 * log(W), making the bound loose. Additionally, Lemma 3 (hierarchy forces absorption over random loss) is stated as an intuitive argument but proving it rigorously requires showing that the gradient of the SAE loss landscape always points toward absorption solutions -- which the Tang et al. framework suggests but does not prove for all architectures.

**Tightness attack**: The bound may be vacuous in practice. If H(F) is small relative to L0 * log(W) (which it often is for real SAEs with W = 16k-65k and L0 = 30-100), the bound gives alpha >= 0, which is trivially true. The bound is only non-trivial when the feature hierarchy is deep and the SAE is narrow/sparse. This limits practical relevance to exactly the regime where absorption is already known to be severe.

**Relevance attack**: The bound tells us absorption is unavoidable but does not tell practitioners what to DO about it. A lower bound without a matching upper bound (achievable rate) is less useful than a complete characterization.

**Novelty attack**: The informal version of this argument already appears in Chanin et al. (2024) Section 4, where they note that absorption saves L0. The Bereska et al. compression framework essentially states the capacity constraint. The formalization adds rigor but the core insight is not new.

**Verdict**: MODERATE. The formalization is worthwhile but the bound may be too loose to be practically predictive. Needs tightening of Lemma 2 for non-uniform feature distributions.


### Against Candidate B: Rate-Distortion Characterization

**Proof soundness attack**: The 2-feature model (parent + child) is analytically tractable but may not extend cleanly to multi-level hierarchies. The distortion computation assumes linear superposition of features (x = a_p * d_p + a_c * d_c), which is only approximately true in real LLMs where MLP nonlinearities introduce nonlinear interactions. The "kink" in the rate-distortion function is a clean prediction but relies on the feature alignment being the dominant factor, whereas in practice, feature frequency (Zipfian) may dominate.

**Tightness attack**: The rate-distortion analysis assumes an optimal encoder/decoder, but real SAEs are trained with gradient descent and may not find the optimal R-D operating point. The gap between the theoretical R-D function and the actual SAE behavior could be large, making the phase transition prediction approximate at best.

**Relevance attack**: The decoder-angle prediction (absorption increases with parent-child cosine similarity) is intuitively obvious and may already be known implicitly from the absorption metric (which measures cosine similarity between absorbing latent and probe direction). The theoretical framework confirms what practitioners already observe.

**Novelty attack**: Bereska et al. (2025) and Ayonrinde et al. (2024) both frame SAEs in information-theoretic terms. The specific application to derive absorption as optimal lossy coding is new, but the R-D framework for SAEs is not. The 2-feature analysis is similar in spirit to Chanin et al.'s toy model proof.

**Verdict**: MODERATE. Clean theoretical result but limited scope (2-feature model) and the main empirical prediction may be unsurprising to practitioners.


### Against Candidate C: Absorption Phase Diagram

**Proof soundness attack**: The phase boundaries are stated as scaling relations (Theta notation) which hides potentially important constants. The hedging boundary uses Chanin et al.'s result which applies to a specific generative model; extending to real LLM features requires justification. The recovery boundary assumes additive L0 requirements per hierarchy level, which is an approximation (parent-child L0 interaction is more complex than simple addition). The interpolation in regime 2 (linear decrease of absorption with L0 surplus) is assumed without proof.

**Tightness attack**: The three-regime model is a simplification. In reality, the transitions between regimes may be smooth rather than sharp, making "phase boundaries" a misleading concept. The model also ignores interactions between different feature hierarchies (real features form a complex graph, not a single tree), which could blur the regime boundaries.

**Relevance attack**: This is highly relevant. A phase diagram would directly guide practitioners in choosing SAE hyperparameters (W, L0) for their specific use case. The prediction that increasing L0 reduces absorption (up to a point) is actionable.

**Novelty attack**: The hedging regime is characterized by Chanin et al. (2025). The absorption regime is characterized by Chanin et al. (2024). The recovery regime is implied by Cui et al. (2025). The novelty is in UNIFYING these into a single diagram with explicit scaling boundaries -- this appears genuinely new. Searched for "phase diagram" + "sparse autoencoder" + "feature absorption" and found no matching results.

**Verdict**: STRONG. The unification is novel, the predictions are testable, and the result is practically useful. Main risk is that the phase boundaries are too idealized.


## Phase 4: Refinement

### Dropped Ideas

**Candidate A** (Information-Theoretic Impossibility) is dropped as a standalone proposal because the bound is likely too loose to be practically informative. The core insight (L0 budget constrains hierarchy representation) is subsumed by Candidate C's more comprehensive framework.

**Candidate B** (Rate-Distortion Characterization) is retained as a supporting theoretical tool for Candidate C. The decoder-angle analysis provides a mechanism for WHY absorption occurs at particular features, while the phase diagram provides the macro-level WHEN.

### Strengthened Survivor: Candidate C (Phase Diagram)

**Improvements**:

1. **Weaker assumptions**: Instead of requiring a known feature hierarchy, define hierarchy depth d and branching factor b as statistical properties estimated from feature co-occurrence data. Specifically, d_eff = max eigenvalue of the feature co-occurrence adjacency matrix normalized by its trace; b_eff = mean branching factor of the probe-derived feature hierarchy tree. This makes the theory applicable without ground-truth feature knowledge.

2. **Incorporating the R-D mechanism from Candidate B**: The absorption rate in regime 2 is refined to:

   alpha(R, L0, theta) = max(0, 1 - (L0 - 1) / d_eff) * (1 - cos^2(theta_avg))^{-1/2}

   where theta_avg is the average angle between parent-child decoder directions. This captures both the L0 budget constraint and the geometric alignment factor.

3. **Testable scaling predictions** refined with specific numerical predictions for Gemma Scope SAEs:
   - Gemma 2 2B, layer 12, width 16k, L0 ~ 50: Predicted absorption regime (alpha ~ 15-25%)
   - Same with width 65k, L0 ~ 100: Predicted near recovery (alpha ~ 5-10%)
   - Same with width 1k, L0 ~ 20: Predicted hedging regime (alpha undefined, features merged)

4. **Connection to safety**: The phase diagram predicts that safety-relevant features (which tend to be rare and high in the hierarchy) are disproportionately absorbed. This follows directly from the L0 budget argument: rare parent features have the lowest "defense" against absorption because the sparsity savings from absorbing them is largest relative to their activation frequency.

### Additional novelty verification

Searched specifically for papers providing a unified phase diagram of SAE failure modes. The closest work is:
- Chanin et al. (2025) on hedging: characterizes hedging alone, not unified with absorption
- Tang et al. (2025): provides optimization landscape analysis but not a quantitative phase diagram with measurable boundaries
- Brill et al. (2025): provides scaling regimes for SAE capacity allocation but not for absorption specifically

No paper provides the unified hedging-absorption-recovery phase diagram with quantitative scaling predictions. The proposed contribution is novel.


## Phase 5: Final Proposal

### Title
**The Absorption Phase Diagram: Scaling Laws for Feature Recovery Under Hierarchical Structure in Sparse Autoencoders**

### Formal Claim

**Main Proposition**: Let an SAE with dictionary width W, sparsity L0, and decoder directions {d_1, ..., d_W} operate on activations generated by a set of features F with hierarchical structure characterized by effective depth d_eff and effective branching factor b_eff. Then the feature absorption rate alpha satisfies:

1. **Hedging regime** (W / |F| < 1/(1 - rho_max)): Feature absorption is not separately measurable; parent-child features are merged (hedged). The dominant failure mode is feature hedging.

2. **Absorption regime** (1/(1 - rho_max) <= W/|F| < C * (1 + d_eff)): The absorption rate is bounded by:

   alpha in [alpha_L, alpha_U]

   where alpha_L = max(0, 1 - L0/(d_eff + 1)) and alpha_U = 1 - L0 * cos^2(theta_min) / (d_eff + 1), with theta_min being the minimum parent-child decoder angle.

3. **Recovery regime** (W/|F| >= C * (1 + d_eff) AND L0 >= d_eff + 1): alpha approaches 0 exponentially in the surplus (L0 - d_eff - 1).

The phase boundaries exhibit sharp transitions when the feature hierarchy is tree-structured and smooth crossovers when features form a general DAG.

**Corollary (Safety implication)**: Features at hierarchy depth d_max (the most general/abstract features) have absorption rate alpha_max = max(0, 1 - L0/(d_max + 1)), which for typical SAE settings (L0 ~ 50, d_max ~ 3-5) predicts alpha_max ~ 90-98%. This means the most abstract features (including many safety-relevant concepts) are almost always absorbed.

### Proof Sketch

**Step 1: L0 Budget Analysis**. Formalize the per-token L0 budget as a resource allocation problem. Each token's activation encodes features at multiple hierarchy levels. The SAE must allocate L0 slots across these levels. Under sparsity optimization, the allocation is greedy: features with the highest marginal reconstruction benefit per L0 slot are activated first.

**Step 2: Hierarchy Priority Ordering**. Show that child features (lower in hierarchy, more specific) provide strictly more reconstruction benefit per L0 slot than parent features. This is because children are more aligned with specific activation patterns. Formally, |<x, d_child>| >= |<x, d_parent>| * cos(theta_{p,c}) when the child is active.

**Step 3: Absorption Cascade**. When L0 is exhausted by child features, parent features are systematically excluded. The absorption rate is the fraction of parent features whose L0 slots are consumed by their children. This fraction depends on d_eff (more levels = more competition for L0 slots) and L0 (more budget = less competition).

**Step 4: Width Dependence**. Below R_hedge, the SAE cannot distinguish parent and child features due to insufficient dictionary atoms. Above R_recover, each feature gets dedicated atoms and the hierarchy can be represented separately. The transition from hedging to absorption occurs when W/|F| crosses the hedging threshold.

**Step 5: Phase Boundary Derivation**. Combine Steps 1-4 with the feature frequency distribution (Zipfian) to derive the phase boundary curves as functions of W, L0, d_eff, and b_eff.

### Assumptions

1. **Linear Representation Hypothesis**: Model activations are well-approximated as sparse linear combinations of feature directions.
2. **Hierarchical Feature Structure**: Features form a partial order (DAG) with measurable depth and branching factor.
3. **Sparsity-Optimized SAE**: The SAE is trained to convergence under a sparsity penalty (L1, TopK, or JumpReLU).
4. **Feature Direction Stability**: Decoder directions do not change dramatically across different inputs (approximately fixed dictionary).
5. **Moderate Superposition**: The superposition ratio |F|/d_model is large enough that some features must share directions (i.e., the setting where SAEs are needed).

### Empirical Prediction

The theory makes five testable predictions:

**Prediction 1 (L0 scaling)**: For fixed width W, absorption rate alpha decreases approximately linearly with L0 in the absorption regime. Plotting alpha vs L0 for Gemma Scope SAEs (varying L0 via JumpReLU threshold) should yield a linear trend.

**Prediction 2 (Width scaling)**: For fixed L0, there exist two critical widths W_hedge and W_recover. Below W_hedge, features are merged (measurable as high inter-feature correlation). Between W_hedge and W_recover, absorption is present but features are distinct. Above W_recover, absorption approaches zero.

**Prediction 3 (Hierarchy depth scaling)**: Absorption rate should be higher for deeper feature hierarchies. Comparing first-letter features (depth 2: letter -> token) with knowledge features (depth 3-4: entity type -> entity -> attribute -> value) should show significantly higher absorption for knowledge features.

**Prediction 4 (Decoder angle correlation)**: Within the absorption regime, features with smaller parent-child decoder angles (more aligned) should show higher absorption rates. This is directly testable by computing cosine similarity between probe-identified parent and child decoder directions.

**Prediction 5 (Safety feature absorption)**: Abstract safety-relevant features (e.g., "harmful intent" which is high in the hierarchy, subsuming many specific harmful categories) should show absorption rates near 100% in standard SAEs, explaining DeepMind's negative results on SAE-based safety probes.

### Experimental Plan

All experiments use pre-trained SAEs (training-free analysis) on small-to-medium models. Target: each experiment <= 1 hour on a single GPU.

**Experiment 1: L0-Absorption Scaling Curve** (~30 min)
- Model: Gemma 2 2B, layer 12
- SAEs: Gemma Scope JumpReLU SAEs at widths 16k and 65k
- Vary effective L0 by adjusting the activation threshold post-hoc (threshold sweep from 0.01 to 10.0)
- Measure absorption rate using the Chanin et al. metric on the first-letter task
- Plot alpha vs effective L0 and fit linear model in the absorption regime
- Tools: SAELens + sae-spelling codebase

**Experiment 2: Width-Absorption Phase Boundaries** (~45 min)
- Model: Gemma 2 2B, layer 12
- SAEs: Gemma Scope at widths 1k, 4k, 16k, 32k, 65k (if available), matched L0
- Measure: (a) inter-feature correlation (hedging indicator), (b) absorption rate, (c) feature recovery (probe accuracy)
- Identify W_hedge and W_recover empirically and compare to predicted scaling

**Experiment 3: Cross-Domain Hierarchy Depth** (~45 min)
- Model: GPT-2 Small (for reproducibility) or Gemma 2 2B
- Task 1: First-letter features (depth 2) -- baseline using Chanin et al. methodology
- Task 2: Entity-type features (depth 3) -- e.g., "is a city" -> "is in Europe" -> "is Paris"
- Task 3: Part-of-speech hierarchy (depth 2-3) -- e.g., "is a noun" -> "is an animate noun" -> "is a person"
- Compare absorption rates across depth levels
- Tools: Custom probes trained on each hierarchy using TransformerLens

**Experiment 4: Decoder Angle Analysis** (~20 min)
- Model: Gemma 2 2B, layer 12, Gemma Scope 16k SAE
- For all absorbed features identified in Experiment 1, compute cosine similarity between the absorbing latent's decoder direction and the probe direction
- Correlate with absorption rate per feature
- Test Prediction 4 directly

**Experiment 5: Safety Feature Absorption Pilot** (~30 min)
- Model: Gemma 2 2B
- Define simple safety hierarchy: "harmful" -> "violent" -> specific violent acts
- Train probes for each level
- Measure absorption rate at each level
- Compare SAE probe accuracy vs dense linear probe accuracy
- Quantify how much of the SAE-probe gap is attributable to absorption

### Baselines

**Theoretical baselines**:
- Null model: absorption rate = 0 (all features recovered independently)
- Random allocation model: alpha = 1 - L0/|F| (features compete uniformly for L0 slots)
- Chanin et al. toy model: alpha = 1 for 2-feature hierarchy (maximum absorption)

**Empirical baselines**:
- Reported absorption rates from SAEBench (15-35% for Gemma Scope)
- Dense linear probe accuracy (upper bound on recoverable information)
- Random SAE baseline (from Korznikov et al., 2026)

### Risk Assessment

**Where the proof might fail**:
- The linear priority ordering (Step 2) may not hold when parent features carry unique information not present in children. In this case, the SAE may prefer parents over children for some tokens, reducing absorption below the predicted rate.
- The Zipfian frequency assumption may not accurately describe all feature hierarchies. Safety features may have unusual frequency distributions.
- Phase boundaries may be smooth rather than sharp in practice, making them hard to identify empirically.

**Where theory-practice gap might be large**:
- Real SAEs are not trained to global optima. The phase diagram describes the landscape at convergence, but gradient-descent trained SAEs may get stuck in different local optima depending on initialization and learning rate.
- The linear superposition assumption may break down for features that interact nonlinearly through MLP layers.
- The effective hierarchy depth d_eff is a statistical estimate and may not capture the true complexity of the feature graph.

**Mitigation strategies**:
- Use multiple independent SAE training runs to estimate the spread of absorption rates around the predicted value
- Validate the theory first on synthetic benchmarks (SynthSAEBench) where ground-truth hierarchy is known, before applying to real models
- Report confidence intervals and model uncertainty alongside point predictions

### Novelty Claim

The specific theoretical contribution is threefold:

1. **First unified phase diagram** of SAE failure modes (hedging, absorption, recovery) as a function of measurable SAE parameters (W, L0) and feature hierarchy statistics (d_eff, b_eff). No prior work unifies these failure modes into a single predictive framework.

2. **Quantitative scaling laws** for absorption rate as a function of hierarchy depth and L0 budget. Prior work (Chanin et al.) proves absorption EXISTS but does not predict its MAGNITUDE.

3. **Safety implication corollary**: The phase diagram directly predicts that the most abstract (and often safety-relevant) features are the most severely absorbed, providing theoretical grounding for DeepMind's empirical finding that SAE probes fail on safety tasks.

**Evidence of novelty**: Extensive search across arXiv, Google Scholar, and web sources found no paper providing (a) a unified phase diagram of hedging/absorption/recovery, (b) quantitative absorption rate predictions as a function of hierarchy depth, or (c) formal connection between absorption severity and feature abstraction level.

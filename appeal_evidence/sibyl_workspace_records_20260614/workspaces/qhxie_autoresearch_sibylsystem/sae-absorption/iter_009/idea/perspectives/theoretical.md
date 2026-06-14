# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** -- Introduces the delta-absorption parameterization showing that for hierarchical features f_1 superset f_2, a one-parameter family of encoder/decoder weights indexed by delta in [0,1] achieves perfect reconstruction for all delta. Proves that the SAE loss landscape is flat along the absorption direction, so gradient-based training cannot distinguish absorbed from non-absorbed solutions. The sparsity penalty breaks this degeneracy in favor of absorption (fewer active latents).

2. **Cui et al. (2025), "On the Limits of SAEs" (arXiv:2506.15963)** -- First necessary and sufficient conditions for identifiable SAEs. Shows SAEs generally fail to recover ground truth monosemantic features unless features are extremely sparse. Closed-form optimal solution reveals feature shrinking and feature vanishing as systematic pathologies. Proposes reweighted SAE (WSAE) with theoretical weight-selection rule.

3. **Tang et al. (2025), "Unified Theory of SDL" (arXiv:2512.05534)** -- Casts all major sparse dictionary learning variants (SAEs, transcoders, crosscoders) as a single piecewise biconvex optimization problem. Characterizes the global solution set, non-identifiability, and spurious minima. Provides the first principled theoretical explanation for feature absorption and dead neurons as consequences of optimization landscape structure. Proposes feature anchoring to restore identifiability.

4. **Chen et al. (2025), "Taming Polysemanticity" (arXiv:2506.14002)** -- Proves feature recovery guarantees for SAEs under a statistical model of polysemantic activations. Introduces bias-adaptation training algorithm with Group Bias Adaptation (GBA) variant. Recovery guarantees rely on restrictive generative assumptions but do not model feature hierarchy.

5. **Elhage et al. (2022), "Toy Models of Superposition"** -- Foundational: neural nets encode more features than neurons via superposition. Establishes the linear representation hypothesis. The "importance-weighted" loss framework provides the theoretical basis for understanding why high-frequency features dominate SAE training.

6. **Bereska et al. (2025), "Superposition as Lossy Compression" (arXiv:2512.13568)** -- Information-theoretic framework measuring superposition via Shannon entropy of SAE activations. Defines effective features as the minimum neurons needed for interference-free encoding. Formalizes superposition as lossy compression beyond the interference-free limit. Strong correlation (r=0.94) with ground truth in toy models.

7. **Ayonrinde et al. (2024), "Interpretability as Compression" (arXiv:2410.11179)** -- MDL framework for SAEs. Argues that interpretable SAEs require "independent additivity" (features should be understandable separately). MDL naturally penalizes both over-splitting and absorption. Suggests hierarchical SAE architectures as the natural consequence of MDL optimization. Feature splitting and absorption are artifacts of using sparsity as a proxy for compression quality.

8. **Wainwright (2008), Information-theoretic limits of sparse recovery** -- Necessary and sufficient conditions for sparse support recovery in noisy linear models. Establishes fundamental sample complexity bounds. Relevant as the classical baseline for what is achievable in sparse decomposition.

9. **Aksoylar & Saligrama (2014), "Information-Theoretic Characterization of Sparse Recovery"** -- Non-asymptotic bounds on recovery probability and tight mutual information formula for sample complexity in sparse support recovery. The framework applies to the question of whether SAEs have enough data to distinguish hierarchically related features.

10. **Bussmann et al. (2025), "Matryoshka SAEs" (arXiv:2503.17547, ICML 2025)** -- Achieves absorption rate ~0.03 vs BatchTopK ~0.29 by training nested dictionaries with prefix losses. The theoretical insight: hierarchical features require hierarchical capacity allocation. Inner levels learn general features, outer levels learn specific ones, explicitly breaking the degeneracy that causes absorption.

11. **Chanin et al. (2025), "Feature Hedging" (arXiv:2505.11756)** -- Theoretically and empirically shows narrow SAEs merge correlated features. Establishes the absorption-hedging tradeoff: Matryoshka SAEs reduce absorption but increase hedging at inner levels. Proposes balanced Matryoshka with compound multiplier ~0.75.

12. **Anthropic (2025), "SAE Scaling with Feature Manifolds" (arXiv:2509.02565)** -- Formal analysis of SAE scaling behavior adapted from neural scaling laws. Shows SAEs can pathologically tile common feature manifolds instead of discovering rare features. Frequency-decay slopes of ~-0.74 for Gemma Scope SAEs. Directly relevant to understanding why high-frequency child features absorb low-frequency parent features.

### Theoretical Landscape Summary

The theoretical understanding of feature absorption sits at the intersection of three partially developed frameworks:

**1. Optimization landscape theory.** Tang et al. (2025) show that SDL objectives are piecewise biconvex with multiple global optima. Absorption corresponds to a family of equivalent global optima parameterized by delta (Chanin et al. 2024). The sparsity penalty selects the absorbed solution because it has lower L0. This is well-established but does not predict absorption *magnitude* as a function of SAE configuration.

**2. Identifiability theory.** Cui et al. (2025) provide necessary and sufficient conditions for SAE identifiability, but these conditions assume independent features and do not model hierarchical co-occurrence. Chen et al. (2025) prove recovery guarantees under a specific generative model but also ignore hierarchy. The open question is: what identifiability conditions hold when features form a DAG (directed acyclic graph) of implication relationships?

**3. Information-theoretic compression theory.** Bereska et al. (2025) formalize superposition as lossy compression. Ayonrinde et al. (2024) connect SAE training to MDL. Neither explicitly models how hierarchical feature structure affects the rate-distortion tradeoff. The gap: a rate-distortion analysis of SAE training that accounts for feature hierarchy would predict when absorption is information-theoretically optimal vs. avoidable.

**What is known:**
- Absorption is a necessary consequence of sparsity optimization under feature hierarchy (proved in toy model).
- Absorption saves exactly 1 unit of L0 per parent-child pair (empirical observation, not formally proved in general).
- All tested SAE architectures exhibit absorption, including TopK (which lacks L1).
- Wider and more sparse SAEs show higher absorption (empirical, no theoretical explanation).
- Absorption rate is 15-35% on the first-letter task across hundreds of SAEs.

**What is conjectured but unproved:**
- Absorption rate should scale with the ratio of parent-feature frequency to child-feature frequency.
- The absorption-reconstruction tradeoff is Pareto-optimal (absorption cannot be removed without increasing L0).
- MDL-optimal SAEs would have less absorption than sparsity-optimal SAEs.
- Feature anchoring (Tang et al.) should reduce absorption by breaking the flat loss landscape.

**Where the gaps are:**
- No closed-form prediction of absorption magnitude as f(width, L0, hierarchy depth, co-occurrence statistics).
- No information-theoretic lower bound on achievable absorption rate for a given reconstruction budget.
- No theoretical distinction between "true" absorption (hierarchy-driven) and "apparent" absorption (L0/hedging confound).
- No theory for absorption in multi-dimensional feature spaces (beyond 1D linear features).

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Absorption Bound via Rate-Distortion Theory

**Formal claim (Theorem sketch):**

Consider an SAE with dictionary size M, target sparsity L0 = k, operating on activations from a model with N ground-truth features organized in a hierarchy H = (V, E) where (f_parent, f_child) in E implies f_child fires only when f_parent fires. Let p_i be the marginal firing probability of feature i, and let the hierarchy have maximum depth D. Define the *absorption potential* of a parent-child pair (i,j) as:

A(i,j) = p_j / p_i (the conditional probability that the child fires given the parent fires, relative to the parent's base rate).

**Claim:** For an SAE minimizing reconstruction loss + lambda * L0 on data generated from this model, the expected absorption rate alpha satisfies:

alpha >= 1 - k / (k + sum_{(i,j) in E} min(1, lambda * A(i,j) / R(i)))

where R(i) is the reconstruction cost of feature i. In the limit of strong sparsity pressure (large lambda), alpha approaches 1 for all parent-child pairs where A(i,j) > R(i)/lambda.

**Proof sketch:**
1. Model SAE training as a rate-distortion problem: the encoder must compress N-dimensional feature activations into k active latents (rate constraint), minimizing reconstruction distortion.
2. When features form a hierarchy, the child feature carries strictly more information than the parent (f_child implies f_parent). By the data processing inequality, representing the child subsumes the parent's information.
3. Under the rate constraint (at most k active latents), the optimal encoder allocates capacity to features with the highest information-per-latent ratio. For each parent-child pair, encoding both costs 2 latents but the marginal information gain of the parent (given the child) is at most H(f_parent | f_child) = H(f_parent) - H(f_child | f_parent) bits.
4. Absorption is optimal when H(f_parent | f_child) < lambda (the Lagrange multiplier for sparsity), i.e., the information uniquely carried by the parent feature is not worth the cost of an additional latent.
5. Sum over all parent-child pairs in the hierarchy to get the total absorption rate.

**Empirical prediction:** Absorption rate should increase monotonically with:
- Sparsity pressure lambda (or equivalently, lower L0)
- Co-occurrence frequency A(i,j) (child-parent overlap)
- Hierarchy depth D (more levels of subsumption)

And decrease with:
- Dictionary width M (more capacity to "afford" both parent and child)
- Reconstruction weight (higher penalty for missing parent features)

These predictions can be tested on Gemma Scope SAEs across varying widths and sparsities, using the first-letter task (known hierarchy) and entity-type tasks (knowledge hierarchy).

**Connection to existing theory:** Extends Bereska et al.'s superposition-as-lossy-compression framework to hierarchical feature settings. Connects to Ayonrinde et al.'s MDL framework by showing absorption is MDL-optimal when the parent feature's unique information is below the description cost threshold. Generalizes Chanin et al.'s delta-absorption toy model to the multi-feature, probabilistic setting.

**Novelty estimate:** 8/10 -- No existing work provides an information-theoretic bound on absorption rate. The rate-distortion framing is natural but unexplored. The closest work (Bereska et al. 2025) measures superposition but does not model hierarchical feature structure.

---

### Candidate B: Absorption as Implicit Hierarchical Regularization -- A PAC-Bayes Generalization Perspective

**Formal claim (Proposition sketch):**

Feature absorption in SAEs is not merely a training artifact but serves as implicit regularization that improves generalization. Specifically, an SAE with absorption has lower PAC-Bayes generalization bound than an equivalent SAE without absorption, because absorption reduces the effective number of parameters (fewer distinct active latent patterns).

Define the *effective complexity* of an SAE as the entropy of its activation pattern distribution: C_eff = H(supp(z)) where supp(z) is the binary support of the latent vector z. Then:

**Claim:** For an SAE with dictionary size M and sparsity k, if the data-generating features form a hierarchy H with |E| parent-child edges, then absorption reduces C_eff by at most |E| * log(2) bits (one bit per absorbed pair), and the PAC-Bayes generalization bound improves by:

Delta_gen <= sqrt(|E| * log(2) / (2n))

where n is the number of training samples.

**Proof sketch:**
1. Without absorption, each parent-child pair can independently fire or not, contributing 2 bits to the activation pattern entropy.
2. With full absorption, the parent feature's activation is deterministically predicted by the child's, reducing the pair's entropy contribution by 1 bit.
3. By the PAC-Bayes theorem (McAllester 2003), the generalization gap is bounded by sqrt(KL(posterior || prior) / (2n)), where the KL term includes the effective complexity.
4. Absorption reduces KL by |E| * log(2), yielding the claimed bound improvement.

**Empirical prediction:** SAEs with more absorption should generalize better (lower reconstruction loss on held-out data) than SAEs trained with absorption-mitigation techniques, when evaluated at the same L0. This predicts a fundamental tension: absorption hurts interpretability but helps generalization.

**Connection to existing theory:** Extends the classical PAC-Bayes framework to SAE activation patterns. Connects to the observation in Chanin et al. that absorption saves 1 L0 per pair (our bound quantifies the generalization benefit of this savings). Relates to Ayonrinde et al.'s MDL perspective (MDL and PAC-Bayes bounds are related via prequential coding).

**Novelty estimate:** 6/10 -- The PAC-Bayes connection is somewhat standard; the novelty is in applying it specifically to SAE absorption and deriving the hierarchy-dependent bound. The claim that absorption is beneficial for generalization (while harmful for interpretability) is provocative but may be hard to distinguish from the simpler explanation that lower L0 helps generalization.

---

### Candidate C: Absorption Phase Transition -- Critical Sparsity Threshold for Hierarchy-Driven Feature Collapse

**Formal claim (Theorem sketch):**

For an SAE with M dictionary atoms trained on activations with N features organized in a tree hierarchy of depth D, there exists a *critical sparsity threshold* k* such that:

- For k > k*, the SAE can represent both parent and child features without absorption (absorption rate alpha = 0).
- For k < k*, absorption undergoes a phase transition: alpha increases sharply from 0 to a value determined by the hierarchy structure.

Formally, define k* = N - |E_active| where |E_active| is the number of parent-child edges where both features are simultaneously active (on average). The critical threshold satisfies:

k* = E[|active features|] - E[|absorption-savings|]

where the absorption-savings term counts the number of parent features that can be "absorbed" into their children without increasing reconstruction error.

**Proof sketch:**
1. Model the SAE encoder as a maximum-weight k-subset selector: given input x = sum_i a_i * d_i, the encoder selects the k atoms with highest inner product.
2. When k >= N_active (number of truly active features), all features can be represented, and absorption provides no benefit. The encoder has enough capacity.
3. When k < N_active, the encoder must choose which features to drop. Hierarchical features are the cheapest to drop: dropping a parent when its child is active costs zero reconstruction error (because the child's decoder column already encodes the parent's direction via the delta-absorption parameterization).
4. The transition from "no absorption needed" to "absorption is optimal" occurs at k* = E[N_active] - E[N_hierarchy_savings].
5. Below k*, the absorption rate follows a power law: alpha ~ (k*/k - 1)^beta where beta depends on the hierarchy structure (branching factor, depth).

**Empirical prediction:** By varying L0 (the k parameter) across pre-trained SAEs of different architectures, we should observe:
- A sharp transition in absorption rate at a predictable k* value.
- The transition should be sharper for deeper hierarchies and more predictable for cleaner hierarchical relationships.
- Below the transition, absorption rate should follow a power law with k.

This can be tested on SAEBench SAEs (which span 6 sparsity levels) and Gemma Scope SAEs (which span widths 1k-1M). The first-letter task provides a clean 2-level hierarchy; entity-type tasks can test deeper hierarchies.

**Connection to existing theory:** Phase transitions in sparse recovery are well-studied in compressed sensing (Donoho & Tanner 2009). The critical sparsity threshold is analogous to the phase transition in L1 minimization. The novelty is identifying an analogous transition specific to *hierarchical* feature structure in SAEs.

**Novelty estimate:** 7/10 -- Phase transitions in SAE behavior have not been formally characterized. The compressed sensing analogy is natural but non-trivial to make rigorous for the SAE setting (which has learned dictionaries, not fixed measurement matrices). The prediction of a sharp transition is falsifiable and surprising if confirmed.

---

## Phase 3: Self-Critique

### Against Candidate A (Rate-Distortion Bound)

**Proof soundness attack:** The main gap is in Step 2: the data processing inequality argument assumes the child feature's decoder direction perfectly encodes the parent's information. In practice, the delta-absorption parameterization shows this is only exact in the toy model (orthogonal features). In real LLMs, features are not orthogonal, and the child's decoder direction may not perfectly subsume the parent's. The bound may be loose because it ignores interference between non-hierarchically-related features. Additionally, the rate-distortion formulation assumes independent encoding of each feature, but SAE latents interact through the shared reconstruction target.

**Tightness attack:** The bound involves a sum over all parent-child edges, but in practice most parent features are not fully absorbed -- only those where the child's decoder direction has high cosine similarity with the parent's probe direction. The bound may be vacuous for shallow hierarchies with few edges. Need to verify that the predicted absorption rates match the 15-35% empirical range.

**Relevance attack:** This is directly relevant. A predictive bound on absorption rate would be the single most useful theoretical contribution for the SAE community. It would guide hyperparameter selection (choose lambda/k to keep absorption below a threshold) and architecture design (how much width is needed to avoid absorption for a given hierarchy).

**Novelty attack:** Searched for "rate distortion sparse autoencoder feature hierarchy" -- found Bereska et al. (2025) and Ayonrinde et al. (2024) but neither models hierarchical features specifically. The Tilde Research blog discusses rate-distortion for SAEs but does not address absorption. The specific application of rate-distortion theory to predict absorption magnitude appears novel.

**Verdict:** STRONG -- The proof has identifiable gaps but the framework is sound and the result is highly relevant. The main risk is that the bound is too loose to be useful, but even a qualitative prediction (absorption increases with these variables, decreases with those) would be valuable.

### Against Candidate B (PAC-Bayes Generalization)

**Proof soundness attack:** The argument that absorption reduces C_eff by |E| * log(2) bits is too clean. In practice, the activation patterns are not uniformly distributed, and the entropy reduction from absorption depends on the marginal probabilities of each feature. The PAC-Bayes bound requires a prior over weight space, but the choice of prior for SAEs is non-obvious. The bound improvement of sqrt(|E| * log(2) / (2n)) may be dominated by other terms in the PAC-Bayes bound.

**Tightness attack:** For typical SAE training (millions of tokens), n is large and the sqrt(1/n) term makes the bound improvement negligibly small. The practical generalization benefit of absorption may be real but too small to be detected experimentally, making the claim unfalsifiable.

**Relevance attack:** Moderate. The claim that absorption helps generalization is interesting but somewhat tangential to the main research question (how to detect and mitigate absorption). The interpretability community wants to remove absorption, not celebrate its regularization benefits.

**Novelty attack:** PAC-Bayes bounds for autoencoders are well-studied (Alquier et al. 2017). The specific application to SAE absorption is new but the technical contribution is incremental.

**Verdict:** WEAK -- The bound is likely too loose to be informative, the generalization benefit is probably negligible in practice, and the contribution is tangential to the community's interests. Drop this candidate.

### Against Candidate C (Phase Transition)

**Proof soundness attack:** The "maximum-weight k-subset selector" model of the SAE encoder is an approximation. Real SAE encoders use learned linear projections followed by nonlinear activation, which only approximately selects the top-k features. The phase transition argument relies on a sharp distinction between "enough capacity" and "not enough capacity," but in practice the transition may be gradual due to noise, non-orthogonal features, and optimization imperfections. The power-law prediction alpha ~ (k*/k - 1)^beta lacks a derivation of beta.

**Tightness attack:** The critical threshold k* = E[N_active] - E[N_hierarchy_savings] is defined in terms of unobservable quantities (the true number of active features and the hierarchy savings). To make this testable, we need to estimate these from the data, which requires knowing the hierarchy -- creating a circularity with the absorption metric itself.

**Relevance attack:** High if the phase transition is real. Knowing that there exists a k* below which absorption is inevitable would fundamentally change how practitioners choose sparsity levels. It would also provide theoretical justification for the empirical observation that wider/sparser SAEs show more absorption.

**Novelty attack:** Phase transitions in sparse recovery are well-known (Donoho-Tanner transition). However, the specific application to hierarchical SAE features and the prediction of a k*-dependent transition in absorption rate appears novel. No existing paper characterizes absorption as a phase transition.

**Verdict:** MODERATE -- The core idea (phase transition in absorption at a critical sparsity) is compelling and novel, but the proof is underdeveloped. The main weakness is the circularity in estimating k*. Could be strengthened by deriving k* from observable quantities (SAE width, measured L0, hierarchy statistics from probes).

---

## Phase 4: Refinement

### Dropped
- **Candidate B (PAC-Bayes)**: Fatal weakness -- the bound improvement is negligibly small for practical n, and the contribution is tangential. The insight that absorption helps generalization is worth noting in the paper's discussion section but does not merit a full theoretical treatment.

### Strengthened Survivors

**Candidate A (Rate-Distortion Bound) -- Strengthened:**

Key refinements:
1. **Weaken the orthogonality assumption.** Instead of requiring child features to perfectly subsume parents, parameterize the "information overlap" between parent and child as cos(theta_{ij}) where theta_{ij} is the angle between their decoder directions. The bound becomes:

   alpha(i,j) >= 1 - (1 - cos^2(theta_{ij})) * R(i) / (lambda * A(i,j))

   This reduces to the original bound when theta = 0 (perfect subsumption) and predicts no absorption when theta = pi/2 (orthogonal features).

2. **Observable quantities.** Replace the unobservable "true" hierarchy H with the *empirical* hierarchy estimated from SAE decoder cosine similarities. Define an edge (i,j) whenever cos_sim(d_i, d_j) > tau and feature j fires only when feature i fires. This makes the bound computable from pre-trained SAEs.

3. **Testable prediction.** For each parent-child pair, the theory predicts:
   - Absorption probability increases with cos_sim(d_parent, d_child)
   - Absorption probability increases with p_child / p_parent
   - Absorption probability decreases with reconstruction importance of parent

   These three factors can be measured for all latent pairs in a pre-trained SAE and correlated with observed absorption (via the Chanin et al. metric).

**Candidate C (Phase Transition) -- Merged into Candidate A:**

Rather than pursuing the phase transition as a separate candidate, incorporate it as a corollary of the rate-distortion bound:

**Corollary:** The rate-distortion bound implies a critical sparsity threshold k* at which absorption transitions from negligible to substantial. Specifically, k* is the smallest k such that for all parent-child pairs (i,j), the marginal information H(f_i | f_j) exceeds the sparsity cost lambda. Below k*, at least one parent-child pair must be absorbed.

This is a weaker but more defensible version of Candidate C's claim, derived from the rate-distortion framework rather than assumed independently.

### Selected Front-Runner: Candidate A (Information-Theoretic Absorption Bound)

Rationale:
- Directly addresses Gap 1 (no quantitative causal theory of absorption magnitude)
- The theoretical framework (rate-distortion for hierarchical features) is novel
- The predictions are testable with existing SAEs (training-free analysis)
- The bound connects absorption to three measurable quantities (cosine similarity, co-occurrence, reconstruction importance)
- Integrates insights from multiple existing frameworks (Bereska et al., Ayonrinde et al., Chanin et al.)

---

## Phase 5: Final Proposal

### Title

**An Information-Theoretic Theory of Feature Absorption: Rate-Distortion Bounds for Sparse Autoencoders under Hierarchical Feature Structure**

### Formal Claim

**Main Theorem (informal statement):**

For a sparse autoencoder with sparsity budget k trained on activations generated by a model with N hierarchically structured features, the absorption rate of a parent feature f_i by its child f_j is bounded below by:

alpha(i,j) >= max(0, 1 - (1 - cos^2(theta_{ij})) * R(i) / (lambda * A(i,j)))

where:
- theta_{ij} is the angle between the decoder directions of the parent and child features
- R(i) is the marginal reconstruction cost of feature i (the increase in MSE when feature i is not represented)
- lambda is the effective sparsity penalty (Lagrange multiplier for the L0 constraint)
- A(i,j) = P(f_j active | f_i active) is the conditional activation probability

**Corollary 1 (Aggregate Absorption Rate):**

The total absorption rate across all parent-child pairs satisfies:

alpha_total >= (1/|E|) * sum_{(i,j) in E} max(0, 1 - (1 - cos^2(theta_{ij})) * R(i) / (lambda * A(i,j)))

**Corollary 2 (Critical Sparsity Threshold):**

There exists a critical sparsity k* such that for k < k*, at least one parent-child pair must exhibit absorption. k* is determined by:

k* = argmin_k { k : forall (i,j) in E, lambda(k) * A(i,j) <= (1 - cos^2(theta_{ij})) * R(i) }

where lambda(k) is the sparsity penalty as a function of target L0 = k.

**Corollary 3 (Width-Absorption Scaling):**

For fixed L0, increasing SAE width M reduces absorption only if the additional dictionary atoms reduce the effective cosine similarity between parent and child decoder directions. Specifically:

d(alpha)/d(M) = -(2 * cos(theta_{ij}) * sin(theta_{ij}) * (d(theta_{ij})/d(M))) * R(i) / (lambda * A(i,j))

This predicts that wider SAEs reduce absorption only when they learn more orthogonal parent-child representations, consistent with the OrtSAE finding that orthogonality penalties reduce absorption by ~70%.

### Proof Sketch

**Step 1: Rate-Distortion Formulation.**

Model the SAE encoder as a lossy compressor operating under a rate constraint (at most k active latents). The reconstruction distortion is D = E[||x - x_hat||^2]. The rate is R = E[||z||_0] = k. We seek the encoder-decoder pair (E, Dec) that minimizes D subject to R <= k.

**Key Lemma 1:** For hierarchically related features f_i (parent) and f_j (child), the mutual information satisfies I(f_i; f_j) >= H(f_i) * cos^2(theta_{ij}), where theta_{ij} is the angle between their directions in activation space.

This follows from the fact that cos^2(theta_{ij}) is the fraction of f_i's variance explained by projection onto f_j's direction.

**Step 2: Marginal Cost-Benefit Analysis.**

For each parent-child pair, the encoder faces a binary choice: encode both (cost = 2 latents, benefit = R(i) + R(j)) or encode only the child (cost = 1 latent, benefit = R(j) + cos^2(theta_{ij}) * R(i)). Absorption occurs when:

R(i) * (1 - cos^2(theta_{ij})) < lambda

i.e., the residual reconstruction cost of the parent (after projecting out the child's contribution) is less than the sparsity penalty. This yields:

alpha(i,j) = 1 when lambda > R(i) * (1 - cos^2(theta_{ij}))
alpha(i,j) = 0 when lambda < R(i) * (1 - cos^2(theta_{ij}))

**Step 3: Continuous Relaxation.**

In practice, the transition is not sharp because (a) the encoder is a linear projection (not a perfect k-selector), (b) features are noisy, and (c) co-occurrence patterns modulate the effective cost-benefit. The continuous version:

alpha(i,j) >= 1 - (1 - cos^2(theta_{ij})) * R(i) / (lambda * A(i,j))

where A(i,j) accounts for the fact that absorption is only beneficial when both features co-occur.

**Step 4: Aggregation.**

Sum over all parent-child pairs, weighting by their frequency, to get the total absorption rate.

### Assumptions

1. **Linear representation hypothesis:** Features are approximately linear directions in activation space. (Standard in SAE literature; challenged by Engels et al. 2024 for some features.)
2. **Hierarchical structure is known:** The parent-child relationships can be identified via probing or decoder similarity analysis. (Required for the bound to be computable; the unsupervised detection of hierarchy is a separate problem.)
3. **Decoder directions approximate true feature directions:** The SAE decoder columns are reasonable estimates of the underlying feature directions. (Standard assumption; holds better for well-trained SAEs with low dead latent fraction.)
4. **Sparsity penalty is the dominant driver of absorption:** Absorption is caused by the tension between reconstruction and sparsity, not by other training dynamics. (Supported by Chanin et al.'s toy model proof and the observation that even TopK SAEs show absorption.)
5. **Features are approximately independent conditional on the hierarchy.** (May be violated for features with complex co-occurrence patterns beyond the parent-child relationship.)

### Empirical Prediction

The theory makes the following testable predictions, all verifiable on existing pre-trained SAEs (training-free):

1. **Per-pair absorption prediction.** For each identified parent-child pair (i,j), the absorption probability should increase with:
   - cos_sim(decoder_i, decoder_j): higher similarity means lower residual reconstruction cost
   - P(j active | i active): higher co-occurrence means more opportunities for absorption
   - lambda (sparsity penalty) or lower L0

   And decrease with:
   - R(i): features with higher reconstruction importance are harder to absorb

2. **Cross-architecture prediction.** Architectures that enforce orthogonality (OrtSAE) should show lower absorption because they increase theta_{ij}, reducing cos^2(theta_{ij}). Matryoshka SAEs should show lower absorption because they allocate dedicated capacity for parent features at inner levels, effectively reducing lambda for parents.

3. **Scaling prediction.** For fixed architecture, absorption rate should scale as:
   - alpha ~ 1/M^gamma for increasing width M (where gamma depends on the rate at which added capacity orthogonalizes parent-child pairs)
   - alpha ~ lambda^delta for increasing sparsity penalty (where delta depends on the hierarchy structure)

### Experimental Plan

All experiments use pre-trained SAEs (training-free, per project constraints). Target: each task <= 1 hour on a single GPU.

**Experiment 1: Per-Pair Absorption Prediction (First-Letter Task)**
- Models: Gemma-2-2B, GPT-2 Small
- SAEs: Gemma Scope 16k and 65k (multiple layers); GPT-2 SAEs from SAELens
- Method: For each SAE, (a) compute absorption rate using the Chanin et al. metric on the first-letter task, (b) for each absorbed pair, measure cos_sim(d_parent, d_child), P(child|parent), and R(parent), (c) fit the theoretical bound and measure R^2.
- Expected output: Scatter plot of predicted vs. observed absorption rate per pair; R^2 > 0.5 would be a strong result.
- Time: ~30 min per model/SAE combination.

**Experiment 2: Cross-Domain Hierarchy Validation**
- Task: Entity-type hierarchy (country > city). Train probes for "is-country" and "is-city-in-X" using in-context prompts.
- Models: Gemma-2-2B
- SAEs: Gemma Scope 16k, 65k at layers 6, 12, 18
- Method: Adapt the sae-spelling absorption metric framework to entity-type probes. Measure absorption rate and compare with theoretical prediction using the same per-pair bound.
- Expected output: Absorption rates for entity-type task; comparison with first-letter task rates; validation that the bound transfers across domains.
- Time: ~45 min (probe training + absorption measurement).

**Experiment 3: Width Scaling**
- SAEs: Gemma Scope at widths 1k, 4k, 16k, 65k, 131k (same layer, same model)
- Method: Measure absorption rate at each width. Compute the average cos_sim between parent-child decoder pairs at each width. Test whether alpha scales with 1/M^gamma as predicted.
- Expected output: Width-absorption scaling curve; estimate of gamma; comparison with theoretical prediction.
- Time: ~20 min (loading + measurement across widths).

**Experiment 4: Sparsity Scaling**
- SAEs: SAEBench SAEs (6 sparsity levels, fixed width 16k, Gemma-2-2B layer 12)
- Method: Measure absorption rate at each sparsity level. Compute effective lambda from the L0 values. Test whether absorption transitions at the predicted k*.
- Expected output: Sparsity-absorption curve; test for phase transition at k*; estimate of delta.
- Time: ~20 min.

### Baselines

**Theoretical baselines:**
- Null model: absorption rate is random (no correlation with cos_sim, co-occurrence, or reconstruction importance).
- Frequency-only model: absorption rate is predicted solely by P(child|parent) (the simplest causal explanation).
- Sparsity-only model: absorption rate is predicted solely by L0 / width.

**Empirical baselines:**
- SAEBench absorption scores for standard architectures (BatchTopK, JumpReLU, TopK).
- Matryoshka SAE absorption scores (~0.03) as the best-case mitigation baseline.
- OrtSAE absorption scores (~70% reduction) as the orthogonality-based baseline.

### Risk Assessment

1. **Bound is too loose.** The rate-distortion bound may be dominated by terms we are ignoring (inter-feature interference, non-linear effects, optimization artifacts). Risk: medium. Mitigation: even a qualitative prediction (absorption correlates with these three factors) is publishable as the first quantitative theory.

2. **Hierarchy estimation is circular.** Measuring cos_sim between decoder directions requires knowing which pairs are parent-child, which requires probes, which is what the absorption metric already provides. Risk: low. Mitigation: the bound predicts absorption *magnitude* from observable geometric quantities, not whether absorption exists. The hierarchy is estimated independently from probes.

3. **Theory-practice gap.** The bound assumes a simplified model of SAE training. Real SAEs are trained with Adam, have batch effects, dead latents, etc. Risk: medium. Mitigation: test on many SAE architectures and configurations; if the bound holds across architectures, the simplifications are justified.

4. **Limited to known hierarchies.** The bound requires knowing parent-child relationships. For truly novel features (where the hierarchy is unknown), the bound is not directly applicable. Risk: low for this paper (we focus on known hierarchies as a proving ground for the theory).

5. **Competing concurrent work.** The unified SDL theory paper (Tang et al. 2025) may extend to provide similar predictions. Risk: medium. Mitigation: our rate-distortion framework is complementary (information-theoretic rather than optimization-theoretic) and provides different, potentially tighter bounds.

### Novelty Claim

**The specific theoretical contribution is threefold:**

1. **First information-theoretic bound on absorption rate.** No prior work derives a closed-form prediction of how severe absorption will be as a function of measurable SAE properties. Chanin et al. prove absorption *exists* in toy models. Cui et al. derive identifiability conditions. Tang et al. characterize the optimization landscape. None predicts absorption *magnitude*. Our bound fills Gap 1 from the literature survey.

2. **Unification of absorption drivers.** The bound reveals that absorption is driven by the interaction of three factors (geometric similarity, co-occurrence frequency, reconstruction importance), providing the first unified quantitative framework for understanding when and why absorption occurs. This is more specific and predictive than existing qualitative explanations.

3. **Cross-domain generalization of absorption theory.** By deriving the bound from information-theoretic first principles (rather than the specific first-letter task), the theory predicts absorption in any hierarchical feature domain. Validating this on entity-type hierarchies (beyond the first-letter task) would address Gap 2 and Gap 6 from the literature survey.

**Evidence of novelty:** Extensive search of arXiv (keywords: "rate distortion sparse autoencoder feature hierarchy," "information theoretic bound absorption," "absorption scaling law SAE") returned no papers combining rate-distortion theory with hierarchical feature absorption. The Tilde Research blog discusses rate-distortion for SAEs but does not address absorption. Bereska et al. (2025) apply Shannon entropy to measure superposition but do not model hierarchy. Ayonrinde et al. (2024) connect SAEs to MDL but do not derive absorption-specific bounds.

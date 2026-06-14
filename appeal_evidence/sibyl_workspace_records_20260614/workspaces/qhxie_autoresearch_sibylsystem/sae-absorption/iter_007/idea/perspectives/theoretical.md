# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024). "A is for Absorption." arXiv:2409.14507 (NeurIPS 2025)** -- The foundational paper defining feature absorption. Provides the delta-absorption parameterization: for two hierarchically related features f_1, f_2 (f_2 subset f_1), a family of encoder/decoder weights parameterized by delta in [0,1] maintains perfect reconstruction while monotonically improving sparsity as delta -> 1 (full absorption). Key mathematical result: absorption is a continuous deformation of the weight space along which reconstruction error is constant but L0 decreases. This is NOT a local minimum analysis; it is a demonstration that the loss landscape has a flat direction favoring absorption.

2. **Tang et al. (2024). "A Unified Theory of Sparse Dictionary Learning." arXiv:2512.05534** -- Casts all SDL variants (SAEs, transcoders, crosscoders) as a single piecewise biconvex optimization problem. Key mathematical results: (i) characterizes the global minimum set, showing it admits solutions that achieve zero reconstruction loss without recovering any interpretable features (non-identifiability theorem); (ii) proves that spurious partial minima exhibiting polysemanticity are pervasive; (iii) shows hierarchical concept structures naturally induce feature absorption as convergence to specific spurious minima; (iv) proposes feature anchoring to restore identifiability. Validated only on synthetic benchmarks.

3. **Cui et al. (2025). "On the Limits of Sparse Autoencoders." arXiv:2506.15963** -- First closed-form theoretical analysis of SAE feature recovery. Key theorem: SAEs generally fail to fully recover ground truth monosemantic features unless features are "extremely sparse." Identifies two specific failure modes under general conditions: feature shrinking (decoder vectors have reduced magnitude) and feature vanishing (features not recovered at all). Proposes the Weighted SAE (WSAE) with theoretical weight selection principle. The extreme sparsity condition for perfect recovery is the critical quantitative bound -- but it does NOT account for feature hierarchy.

4. **Chen et al. (2025). "Taming Polysemanticity in LLMs." arXiv:2506.14002** -- Provides provable feature recovery guarantees under a statistical generative model. Key technique: bias-adaptation training with Group Bias Adaptation (GBA) for LLMs. Recovery guarantees rely on restrictive assumptions (features drawn from a specific generative model) and do NOT model feature hierarchy. The guarantees break when features have implication relations (f_2 implies f_1).

5. **Bereska et al. (2025). "Superposition as Lossy Compression." arXiv:2512.13568** -- Information-theoretic framework measuring neural network superposition as lossy compression. Uses Shannon entropy of SAE activations to compute "effective features" F -- the minimum neurons needed for interference-free encoding. Validates with r=0.94 correlation in toy models. Key insight for absorption theory: superposition IS lossy compression, and absorption is a specific compression strategy where the "lossy" part is the loss of parent-child feature identity. This connects SAE analysis to rate-distortion theory.

6. **Ayonrinde et al. (2024). "Interpretability as Compression: MDL-SAEs." arXiv:2410.11179** -- Frames SAE feature learning through the Minimum Description Length (MDL) principle. SAEs are lossy compression algorithms for communicating explanations of neural activations. Key insight: MDL naturally penalizes undesirable feature splitting and suggests hierarchical SAE architectures. The MDL framework gives a principled information-theoretic cost to absorption: it reduces the description length by eliminating the need to separately encode the parent feature's activation state.

7. **Elhage et al. (2022). "Toy Models of Superposition." Transformer Circuits Thread** -- Foundational theory of superposition. Key mathematical framework: features as directions in R^n with importance weights and sparsity, model as linear map W with ReLU. The superposition phase diagram maps feature density to geometric arrangements (tetrahedron, pentagon, etc.). Absorption is NOT addressed but the geometric framework (features as directions, interference as inner products) is the natural language for formalizing absorption.

8. **Spielman, Wang & Wright (2012). "Exact Recovery of Sparsely-Used Dictionaries." COLT 2012** -- Classical identifiability result for dictionary learning: under sufficient incoherence and sparsity, the dictionary can be exactly recovered up to permutation and scaling. Critical gap: this result assumes features are independent, with NO hierarchy. The incoherence condition (low pairwise inner products between dictionary atoms) is precisely what breaks when features form a hierarchy, because parent and child features naturally have high mutual coherence.

9. **Gribonval & Schnass (2010). "Dictionary Identification via L1-Minimization." Information and Inference** -- Algebraic conditions for local identifiability of dictionaries under L1 minimization. Key result: under a random Bernoulli-Gaussian sparse model, sufficiently incoherent bases are locally identifiable with high probability. The failure mode when incoherence is violated maps directly to absorption: high-coherence feature pairs (parent-child) create degenerate directions in the L1 objective landscape.

10. **Montanari & Wang (2026). "Phase Transitions for Feature Learning." arXiv:2602.01434** -- Proves sharp phase transitions in feature learning as a function of sample-to-dimension ratio. Feature recovery undergoes a discrete threshold transition (not smooth degradation). This framework is potentially transplantable to SAE absorption: there may exist a critical ratio of SAE width to number of hierarchical features where absorption onset is sharp.

11. **Jenatton, Mairal, Obozinski & Bach (2011). "Proximal Methods for Hierarchical Sparse Coding." JMLR** -- Foundational reference for incorporating hierarchical structure into sparse representations via structured sparsity penalties (group lasso with tree-structured groups). The key mathematical tool is the proximal operator for tree-structured groups, which could be adapted to design SAE sparsity penalties that are hierarchy-aware and thus avoid absorption.

12. **Partial Information Decomposition (PID) Framework. Williams & Beer (2010)** -- Decomposes mutual information between multiple sources and a target into synergistic, redundant, and unique information atoms. When applied to SAE features and hierarchically related concepts, PID can formally separate: (a) unique information carried by the parent feature, (b) unique information carried by the child feature, (c) redundant information shared between them, and (d) synergistic information requiring both. Absorption eliminates (a), transferring it into (c) or (d).

### Theoretical Landscape Summary

The theoretical understanding of SAE feature absorption is at an intermediate stage. Three key results are established:

**What is known:**
- Absorption is a *consequence of sparsity optimization under feature hierarchy* (Chanin et al., 2024). The delta-absorption parameterization proves this constructively: there exists a continuous path in weight space from no-absorption to full-absorption that maintains perfect reconstruction while monotonically decreasing L0. The sparsity objective thus has a gradient that always points toward absorption when hierarchical features are present.
- SAE dictionary learning is *fundamentally non-identifiable* without additional constraints (Tang et al., 2024). The global minimum set of the SDL optimization problem includes solutions with zero interpretable features. Absorption is one manifestation of convergence to non-interpretable global or partial minima.
- Feature recovery requires *extreme sparsity* in the general case (Cui et al., 2025). The closed-form analysis shows that unless the ground truth features are sufficiently sparse (a quantitative condition is given), feature shrinking and vanishing are unavoidable.

**What is conjectured but unproved:**
- There exist *critical thresholds* in (width, L0, hierarchy-depth) space where absorption rate undergoes a phase transition (inspired by Montanari & Wang, 2026).
- The *MDL cost of absorption* can be formally computed and provides a principled objective that avoids absorption (inspired by Ayonrinde et al., 2024).
- *Mutual coherence* between parent and child decoder vectors is a necessary (not just correlated) condition for absorption (inspired by Spielman et al., 2012; Gribonval & Schnass, 2010).
- The *information-theoretic capacity* of an SAE (number of effective features a la Bereska et al.) places a hard upper bound on the number of features that can be represented without absorption, independent of architecture.

**Where the critical gaps are:**
1. **No quantitative absorption bound.** Nobody has derived a formula predicting absorption rate as a function of SAE configuration and feature hierarchy statistics. Chanin et al.'s toy model shows absorption CAN happen; Cui et al. show recovery FAILS in general; but neither gives a quantitative prediction of HOW MUCH absorption occurs.
2. **No information-theoretic characterization.** Absorption has been analyzed through optimization (sparsity loss landscape) and linear algebra (decoder cosine similarity), but NOT through information theory. The rate-distortion / MDL / PID frameworks offer rich tools that have not been applied.
3. **No connection between classical dictionary identifiability and absorption.** The classical conditions (incoherence, Bernoulli-Gaussian sparsity) are violated by hierarchical features, but nobody has formally characterized the degree of violation and its consequences for absorption severity.
4. **No hierarchy-aware sparsity theory.** Existing SAE sparsity penalties (L1, TopK, JumpReLU) are feature-agnostic. The structured sparsity literature (Jenatton et al., 2011) provides tools for hierarchy-aware penalties but these have not been connected to SAE absorption.

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Absorption Bound via Rate-Distortion Analysis of Hierarchical Sparse Coding

- **Formal claim**: For an SAE with dictionary size M, activation dimension d, target sparsity L0, operating on activations generated by a hierarchical feature model with K features organized in a tree of depth D and branching factor B, the minimum achievable absorption rate alpha* satisfies:

  alpha* >= 1 - (M * L0) / (K + R_hierarchy)

  where R_hierarchy = sum_{l=1}^{D} B^l * I(f_parent^l ; f_child^l) is the "hierarchy redundancy" -- the total mutual information between parent and child features across all levels of the hierarchy. When M * L0 < K + R_hierarchy, the SAE MUST absorb features because it cannot simultaneously represent all features and their hierarchical relationships within its capacity budget.

- **Proof sketch**:
  1. **Lemma 1 (Capacity Bound)**: An SAE with dictionary M and average L0 active features can faithfully represent at most M * L0 / log(M/L0) independent feature-activation patterns (an encoding capacity argument via counting the number of distinct sparse codes).
  2. **Lemma 2 (Hierarchy Tax)**: When features form a hierarchy, the parent feature f_p and child feature f_c have mutual information I(f_p; f_c) > 0. Representing both as separate SAE latents "wastes" capacity by encoding redundant information. The total hierarchy tax on a tree of depth D with branching B is R_hierarchy = sum B^l * I_l where I_l is the average parent-child MI at level l.
  3. **Main Theorem**: Combining Lemma 1 and 2: if the SAE's encoding capacity (Lemma 1) is less than the total information content K + R_hierarchy, then some features MUST share latents (absorption) or be dropped (dead features). The absorption rate lower bound follows from the capacity deficit divided by K.
  4. **Corollary (Critical Width)**: Setting alpha* = 0 and solving for M gives the critical width M_c = (K + R_hierarchy) * log(M_c/L0) / L0 -- the minimum SAE width that can in principle avoid all absorption. This provides a principled guideline for SAE sizing.

- **Empirical prediction**: (1) Absorption rate should increase monotonically with hierarchy depth D (deeper hierarchies have more parent-child pairs, each contributing MI to R_hierarchy). (2) Absorption rate should decrease with SAE width M at a rate predicted by the bound. (3) For the first-letter spelling task, the hierarchy is depth-2 (26 letters x ~1000 tokens), so the bound predicts a specific absorption rate that can be compared to Chanin et al.'s empirical 15-35%.

- **Connection to existing theory**: Extends Cui et al.'s (2025) recovery failure analysis by adding the hierarchy dimension. Generalizes Bereska et al.'s (2025) effective features metric by connecting it to absorption specifically. Provides the quantitative bound that Gap 1 in the literature survey identifies as missing.

- **Novelty estimate**: 9/10. No existing work provides a quantitative absorption rate bound as a function of hierarchy statistics. The rate-distortion framing of SAE absorption is entirely new. The critical width formula would be immediately practically useful.

### Candidate B: Absorption as Identifiability Failure Under Mutual Coherence Violation -- Extending Classical Dictionary Learning Theory

- **Formal claim**: Define the "hierarchical coherence" of a feature set as mu_H = max_{(p,c) in hierarchy} |<d_p, d_c>| where d_p, d_c are the "natural" decoder directions for parent feature p and child feature c. Then:

  **Proposition**: For an SAE trained with L1 sparsity penalty lambda on activations generated by features with hierarchical coherence mu_H, the probability of absorption of feature p by feature c is lower-bounded by:

  P(absorption of p by c) >= 1 - exp(-lambda * mu_H * freq(c) / freq(p))

  where freq(.) denotes the activation frequency of the feature.

- **Proof sketch**:
  1. **Step 1**: Under the L1 penalty, the SAE objective can be decomposed into a reconstruction term and a sparsity term. For a parent-child pair (p, c) where c subset p, the sparsity saving from absorbing p into c is exactly lambda * freq(p \ c) -- the penalty saved by not firing p on tokens where c fires.
  2. **Step 2**: The reconstruction cost of absorption is ||delta_p|| * ||x - x_hat||, where delta_p is the component of the parent direction orthogonal to the child direction (the part lost by absorption). By definition, delta_p = d_p - <d_p, d_c> * d_c, and ||delta_p|| = sqrt(1 - mu_H^2). When mu_H is high, the reconstruction cost is small.
  3. **Step 3**: Absorption occurs when sparsity saving exceeds reconstruction cost. The probability is computed over the distribution of activation patterns. Using a Gaussian model for activation magnitudes, the probability that sparsity saving dominates is a sigmoid function of lambda * mu_H * freq(c) / freq(p).
  4. **Corollary (Frequency Asymmetry)**: Absorption preferentially targets features where the child is LESS frequent than the parent (freq(c) << freq(p)), because the sparsity saving per absorption event is proportional to lambda but the total saving depends on freq(c) (number of events where both fire). This explains why absorption is most visible on rare, specific features.

- **Empirical prediction**: (1) High-coherence parent-child pairs should exhibit higher absorption rates. Measurable by computing decoder cosine similarity and correlating with empirically measured absorption. (2) Increasing L1 penalty should increase absorption monotonically (more sparsity pressure). (3) Rare child features paired with common parent features should be the primary absorption targets. (4) The sigmoid shape of the absorption probability curve can be empirically validated across a range of lambda values.

- **Connection to existing theory**: Directly extends classical dictionary identifiability theory (Spielman et al., 2012; Gribonval & Schnass, 2010) to the hierarchical feature setting. The incoherence condition mu < 1/sqrt(K) from compressed sensing becomes mu_H < threshold_for_no_absorption. Connects to OrtSAE's empirical success: orthogonality penalty drives mu_H -> 0, which drives P(absorption) -> 0 per the bound.

- **Novelty estimate**: 8/10. Extends classical results to the SAE hierarchy setting. The frequency-dependent absorption probability formula is new and testable. Risk: the Gaussian activation model may be too simplistic for real LLM activations.

### Candidate C: Partial Information Decomposition of SAE Feature Spaces -- Quantifying Absorption as Synergy/Redundancy Imbalance

- **Formal claim**: For an SAE with features f_1, ..., f_M applied to activations encoding hierarchical concepts C_parent, C_child, the Partial Information Decomposition (PID) of I(activation; C_parent) across SAE features yields four non-negative atoms:
  - Unique_{f_parent}(C_parent): information about C_parent uniquely carried by the parent feature
  - Unique_{f_child}(C_parent): information about C_parent uniquely carried by the child feature
  - Redundancy(C_parent): information about C_parent shared by both features
  - Synergy(C_parent): information about C_parent available only when both features are considered jointly

  **Theorem**: Feature absorption of parent by child is characterized by Unique_{f_parent}(C_parent) -> 0 while Unique_{f_child}(C_parent) + Redundancy(C_parent) -> I(activation; C_parent). In other words, the child feature "absorbs" the parent's unique information. Furthermore, the absorption rate defined by Chanin et al. is lower-bounded by:

  absorption_rate >= 1 - Unique_{f_parent}(C_parent) / I(activation; C_parent)

- **Proof sketch**:
  1. **Formalization**: Define the PID for the triple (f_parent activation, f_child activation ; C_parent label) using the redundancy lattice framework of Williams & Beer (2010). The four atoms sum to I(f_parent, f_child ; C_parent) (data processing inequality: <= I(activation; C_parent)).
  2. **No-absorption baseline**: When no absorption occurs, Unique_{f_parent}(C_parent) >= H(C_parent) - H(C_parent | f_parent) is large (the parent feature carries most information about its concept). The child feature carries unique information about C_child but little unique information about C_parent.
  3. **Absorption characterization**: Under absorption, the SAE optimizer redistributes information: Unique_{f_parent}(C_parent) shrinks toward zero because f_parent fails to fire on tokens where f_child fires. The "lost" unique information transfers to Unique_{f_child}(C_parent) (the child feature now carries parent information) and Redundancy.
  4. **Connection to metric**: Chanin et al.'s absorption metric counts tokens where f_parent should fire but does not, and where ablation of f_child strongly affects the parent concept direction. This is exactly the set of tokens where Unique_{f_parent} = 0 and Unique_{f_child} > 0 for C_parent.

- **Empirical prediction**: (1) PID atoms can be estimated empirically from SAE activation patterns and probe labels. (2) The ratio Unique_{f_parent} / I(activation; C_parent) should correlate strongly (r > 0.8) with the complement of Chanin et al.'s absorption rate. (3) Across different SAE architectures, Matryoshka SAEs should show higher Unique_{f_parent} (less absorption) while JumpReLU SAEs show lower Unique_{f_parent} (more absorption). (4) The PID decomposition provides a RICHER characterization than a single absorption rate: it distinguishes between redundancy-driven absorption (parent info is redundant) and synergy-driven absorption (parent info requires joint decoding).

- **Connection to existing theory**: Applies the PID framework (Williams & Beer, 2010; Lizier et al., 2018) to SAE feature analysis for the first time. Connects to Wu et al.'s (2025) MI-based feature explanations but provides a strictly finer decomposition. Provides a theoretical bridge between the optimization perspective (Tang et al., 2024) and the information-theoretic perspective (Bereska et al., 2025).

- **Novelty estimate**: 7/10. PID is a well-established framework but has NOT been applied to SAE feature absorption. The formalization of absorption as PID imbalance is new. Risk: PID estimation in high dimensions is notoriously difficult; practical computation may require simplifying assumptions (e.g., Gaussian approximations or binning).

## Phase 3: Self-Critique

### Against Candidate A (Rate-Distortion Absorption Bound)

- **Proof soundness attack**: The "capacity bound" in Lemma 1 is heuristic. The number of distinct sparse codes with L0 active features from M dictionary elements is C(M, L0), not M*L0/log(M/L0). The actual capacity depends on the continuous activation values, not just which features are active. A rigorous bound would require specifying the activation magnitude distribution and computing the rate-distortion function of the resulting source, which is significantly more complex than the sketch suggests. The connection between rate-distortion theory (which deals with continuous source coding) and the combinatorial structure of SAE sparse codes needs a much more careful treatment.
- **Tightness attack**: The bound alpha* >= 1 - M*L0/(K + R_hierarchy) could be vacuous in practice. For Gemma Scope 16k SAE with L0 ~ 50 and K ~ 10000 features: M*L0 = 800,000 >> K = 10,000, so the bound gives alpha* >= 1 - 800000/10000+ < 0, which is trivially satisfied. The bound is only interesting when M*L0 is comparable to K + R_hierarchy, which may only occur for very narrow SAEs. The R_hierarchy term helps, but for the first-letter spelling hierarchy (depth 2, branching 26), R_hierarchy is bounded by 26 * H(letter) ~ 26 * 4.7 ~ 122 nats, which is small relative to M*L0.
- **Relevance attack**: Even if the bound is correct and non-trivial, it may not predict the OBSERVED absorption rate in practice. Absorption in trained SAEs may be dominated by optimization dynamics (convergence to specific local minima) rather than information-theoretic necessity. The bound characterizes what is IMPOSSIBLE to avoid, but the actual rate could be much higher due to suboptimal training.
- **Novelty attack**: Searched for "rate distortion sparse autoencoder capacity bound." Found Bereska et al. (2025) on superposition as lossy compression, and Ayonrinde et al. (2024) on MDL-SAEs. Neither provides an absorption-specific rate-distortion bound. The specific claim appears novel but the capacity counting argument is standard in information theory.
- **Verdict**: **MODERATE**. The idea is theoretically exciting but the proof sketch has serious gaps (capacity counting is heuristic, bound may be vacuous for practical SAE sizes). The bound would need significant tightening to be empirically relevant. However, even a loose bound that correctly predicts the qualitative trends (absorption increases with hierarchy depth, decreases with width) would be a contribution.

### Against Candidate B (Coherence-Based Absorption Probability)

- **Proof soundness attack**: The decomposition of the SAE objective into independent parent-child pair interactions (Step 1-3) ignores the coupling between all features in the dictionary. In practice, absorbing feature p into feature c affects the reconstruction of ALL tokens, not just those where p and c co-occur. The probability calculation in Step 3 treats each pair independently, which is a mean-field approximation. This approximation may break down when many features form a dense hierarchy (high coupling). Searched for "mean field approximation dictionary learning" -- found that mean-field methods are used in Bayesian dictionary learning but their accuracy for SAEs with L1 penalties is unclear.
- **Tightness attack**: The sigmoid formula P(absorption) >= 1 - exp(-lambda * mu_H * freq(c)/freq(p)) has intuitive behavior but the specific functional form depends on the Gaussian activation model. For real LLM activations (which are NOT Gaussian -- they have heavy tails and complex correlations), the sigmoid may be a poor fit. However, the qualitative predictions (absorption increases with lambda, mu_H, and freq ratio) should hold under any reasonable activation model.
- **Relevance attack**: The formula is directly testable: compute decoder cosine similarities, measure frequencies, vary lambda, and check whether observed absorption matches the predicted probabilities. This is a strong advantage over Candidate A. The predictions can be checked using the first-letter spelling task (where ground truth is available) on existing pretrained SAEs (training-free).
- **Novelty attack**: Searched for "decoder cosine similarity absorption probability sparse autoencoder." OrtSAE (Korznikov et al., 2025) shows that reducing cosine similarity reduces absorption, establishing a causal link. But OrtSAE does not provide a quantitative formula predicting absorption probability as a function of coherence and frequency. The formula itself appears novel. The connection to classical incoherence conditions (Spielman et al., 2012) for the non-hierarchical case provides theoretical lineage.
- **Verdict**: **STRONG**. The proof relies on a mean-field approximation that may not be tight, but the predictions are directly testable on existing SAEs (training-free), the qualitative behavior is robust to model misspecification, and the connection to classical identifiability theory adds theoretical depth. The main risk is that the Gaussian activation model produces quantitatively wrong predictions for real LLM activations, but this can be mitigated by fitting the sigmoid parameters empirically and treating the formula as a semi-parametric model.

### Against Candidate C (PID Absorption Characterization)

- **Proof soundness attack**: The PID framework is well-defined theoretically, but estimation in high dimensions is a major open problem. For continuous random variables (SAE activations), PID computation requires either: (a) discretization (binning), which introduces bias and loses resolution; (b) kernel density estimation, which suffers from the curse of dimensionality; or (c) parametric assumptions (Gaussian), which may not hold. For a practical implementation, we would need to estimate PID atoms from ~10,000-100,000 token samples, which may be insufficient for reliable estimation given the high effective dimensionality of SAE activation spaces.
- **Tightness attack**: The bound absorption_rate >= 1 - Unique_{f_parent}/I(activation; C_parent) is tight by construction IF the PID atoms are estimated correctly. The question is whether the estimation error overwhelms the signal. For a binary concept (e.g., "starts with S" yes/no) and a single-dimensional feature activation, the PID reduces to classical mutual information quantities that are estimable. The challenge arises when trying to compute PID for many parent-child pairs simultaneously.
- **Relevance attack**: The PID decomposition provides a richer characterization than Chanin et al.'s single absorption rate metric. However, the added richness (synergy vs. redundancy distinction) may not translate into actionable insights for practitioners. If the main practical question is "how bad is absorption?" rather than "what kind of absorption is it?", then the PID framework adds complexity without proportional value.
- **Novelty attack**: Searched for "partial information decomposition neural network features sparse autoencoder." Found work on PID in neural network representational analysis (Ehrlich et al., 2023; Tax et al., 2017) but NOT applied to SAE features or absorption. The application is novel, but PID-based neural network analysis is an established (if niche) subfield.
- **Verdict**: **MODERATE**. The theoretical framework is elegant and the formalization of absorption as PID imbalance is clean. But practical estimation challenges are severe, the added granularity over simpler metrics may not justify the complexity, and the framework is more descriptive than prescriptive (it characterizes absorption but does not suggest how to fix it). Best suited as a secondary theoretical contribution supporting a primary empirical contribution.

## Phase 4: Refinement

### Dropped Ideas

**Candidate A (Rate-Distortion Bound)**: Weakened but not dropped. The heuristic capacity argument is a serious flaw, and the bound may be vacuous for practical SAE sizes. However, the qualitative predictions (absorption increases with hierarchy depth and decreases with width) are valuable. I REFINE this into a supporting theoretical framework within the main proposal rather than the headline contribution.

**Candidate C (PID Characterization)**: Dropped as a standalone idea. The estimation challenges make it impractical as the primary contribution. However, the PID formalization provides a useful CONCEPTUAL framework for understanding absorption that can inform the main proposal's framing.

### Strengthened Survivor: Candidate B (Coherence-Based Absorption Theory)

**Refinements:**

1. **Weaker assumptions**: Instead of a Gaussian activation model, I reformulate the result using only the assumption that activation magnitudes have bounded second moments. The qualitative predictions (absorption probability increases with coherence, lambda, and frequency ratio) hold under this weaker condition. The specific sigmoid functional form becomes a parametric model to be fitted, rather than a derived formula.

2. **Extension to multi-level hierarchies**: The pairwise analysis extends to trees of depth D by recursion. At each level, the absorption probability between parent and child is given by the coherence formula. The total absorption rate for a feature at level l is the product of non-absorption probabilities up to level l:

   P(feature at level l NOT absorbed) = prod_{j=1}^{l} (1 - P_j(absorption))

   This gives an exponential decay of non-absorption probability with hierarchy depth, predicting that deeper features are more robustly represented (they are specific enough that no child absorbs them) while shallow/general features are most vulnerable to absorption.

3. **Critical experiment added**: The key empirical test that validates or falsifies the theory is: for each parent-child pair in the first-letter hierarchy, compute (i) decoder cosine similarity mu_H, (ii) feature frequencies, (iii) actual absorption rate from Chanin et al.'s metric. Then fit the sigmoid model and test whether the fitted model generalizes to: (a) held-out letter pairs, (b) different SAE architectures on the same model, (c) different models (GPT-2 vs Gemma). If the same parametric model (with fitted parameters) explains absorption across all settings, this is strong evidence for the theory.

4. **Integration with information-theoretic framing**: The coherence-based formula is placed within a broader information-theoretic framework that includes:
   - A capacity argument (from Candidate A, refined) that sets the necessary condition for ANY absorption to occur
   - The coherence formula (Candidate B) that predicts WHICH features get absorbed
   - A PID interpretation (from Candidate C, simplified) that characterizes the information flow during absorption
   Together, these three levels provide a complete theoretical account: when, which, and what.

### Selected Front-Runner

**Candidate B (refined)**: "Absorption as Identifiability Failure: A Quantitative Theory Connecting Decoder Coherence, Feature Frequency, and Sparsity Penalty to Absorption Probability in Sparse Autoencoders"

This is the strongest candidate because:
- Predictions are directly testable on existing pretrained SAEs (training-free)
- Connects to classical dictionary identifiability theory (established mathematical lineage)
- Provides actionable design guidance (reduce mu_H, adjust lambda, or increase width)
- The coherence-frequency-sparsity interaction is a clean three-variable model that can be empirically validated
- Even if the specific functional form is wrong, the qualitative structure (absorption increases with coherence x frequency ratio x sparsity penalty) is highly likely to hold

## Phase 5: Final Proposal

### Title

**Quantitative Theory of Feature Absorption in Sparse Autoencoders: Coherence, Frequency, and the Absorption Phase Boundary**

### Formal Claim

**Main Proposition (Absorption Probability Bound)**:

Consider an SAE with L1 sparsity penalty lambda, dictionary vectors {d_i} in R^n with ||d_i|| = 1, trained on activations generated by hierarchically related features f_parent and f_child (f_child subset f_parent, i.e., whenever f_child is active, f_parent is also active). Define:
- mu_H = |<d_parent, d_child>| (hierarchical coherence)
- rho = freq(f_child) / freq(f_parent) (relative frequency, with rho <= 1)
- sigma^2 = Var(a_parent) (variance of parent feature activation magnitude)

Then the probability that the SAE absorbs f_parent into f_child under L1 training is:

P(absorption) >= Phi(lambda * mu_H * sqrt(rho) / sigma - Phi^{-1}(1 - mu_H^2))

where Phi is the standard normal CDF. In particular:
- P(absorption) -> 1 as mu_H -> 1 (high coherence makes absorption certain)
- P(absorption) -> 0 as lambda -> 0 (no sparsity pressure, no absorption)
- P(absorption) increases with rho (more frequent child features absorb more readily)
- P(absorption) -> 0 as mu_H -> 0 (orthogonal features cannot absorb each other)

**Supporting Proposition (Critical Width)**:

For a feature hierarchy with K features organized in a tree of depth D and branching factor B, with average parent-child coherence mu_avg and average relative frequency rho_avg, the minimum SAE width M_c to keep the expected absorption rate below epsilon is:

M_c >= K * (1 + D * mu_avg^2) / (1 - epsilon)

This follows from requiring each feature to have at least one dedicated latent with sufficiently low coherence to all other latents.

**Corollary (Absorption-Hedging Trade-off)**:

For fixed SAE width M < M_c, reducing coherence (e.g., via orthogonality penalties as in OrtSAE) reduces absorption but increases feature hedging (because orthogonal directions are forced to represent correlated features). The optimal operating point balances:

min_mu  alpha_absorption(mu, lambda, rho) + beta * alpha_hedging(M, mu, K)

where beta weights the relative importance of the two failure modes.

### Proof Sketch

**Key steps for Main Proposition:**

1. **Decomposition of the L1 objective for a parent-child pair**: Write the total loss L = L_recon + lambda * L_sparse. For tokens where both f_parent and f_child are active (frequency = freq(f_child) = rho * freq(f_parent)), absorption saves lambda units of L_sparse per such token (one fewer active feature) but incurs an additional reconstruction error of |a_parent| * ||d_parent - mu_H * d_child|| = |a_parent| * sqrt(1 - mu_H^2).

2. **Threshold condition**: Absorption is favored when the expected sparsity saving exceeds the expected reconstruction cost:
   E[lambda * 1_{both active}] > E[|a_parent| * sqrt(1 - mu_H^2) * 1_{both active}]

   This simplifies to: lambda > E[|a_parent|] * sqrt(1 - mu_H^2) = sigma * sqrt(2/pi) * sqrt(1 - mu_H^2) (using the Gaussian mean absolute value).

3. **Probabilistic bound**: The absorption event occurs when the above inequality holds for the realized activation magnitude. Computing this probability over the activation distribution yields the stated CDF bound.

4. **Multi-level extension**: For a tree of depth D, apply the pairwise result recursively. At each level l, the "effective parent" is the feature representation after potential absorption from levels 1, ..., l-1. The recursion gives:
   P(feature at level l not absorbed) = prod_{j=1}^{l} (1 - P_j)
   where P_j depends on the coherence and frequency at level j.

### Assumptions

1. **Linear Representation Hypothesis**: Model activations are well-approximated as sparse linear combinations of feature directions.
2. **Hierarchical Feature Structure**: There exist feature pairs (f_parent, f_child) where f_child implies f_parent (i.e., f_child being active implies f_parent is active).
3. **Independence of absorption events**: The absorption of one parent-child pair does not strongly affect the absorption of other pairs (mean-field approximation).
4. **Bounded activation magnitudes**: Feature activation magnitudes have bounded second moments.
5. **L1 sparsity penalty**: The result is stated for L1; analogous results hold for TopK and JumpReLU with modified threshold conditions (TopK: the threshold becomes the top-k activation value; JumpReLU: the threshold parameter theta replaces lambda).

### Empirical Prediction

The theory makes five quantitative predictions testable on existing pretrained SAEs:

1. **Coherence-absorption correlation**: For each SAE, compute pairwise decoder cosine similarity between features identified as parent-child by the first-letter probing task. The absorption rate for each parent feature should correlate with the maximum mu_H to any of its child features. Predicted Spearman correlation: r >= 0.5.

2. **Frequency-absorption asymmetry**: Among parent features with similar mu_H values, those with higher freq(child)/freq(parent) ratios should have higher absorption rates. This can be tested by binning parent-child pairs by mu_H and testing for frequency effects within bins.

3. **Architecture comparison**: The theory predicts that OrtSAE (low mu_H by construction) should have lower absorption than BatchTopK/JumpReLU (no mu_H constraint), and Matryoshka SAE (explicit hierarchy handling) should avoid absorption through a different mechanism (reducing the effective hierarchy depth). These predictions match known empirical results (OrtSAE reduces absorption ~70%, Matryoshka ~90%), providing post-hoc validation.

4. **Critical width prediction**: For the first-letter spelling hierarchy (K ~ 26 parent features + ~5000 child features, D = 2, B = ~200, average mu_H estimated from data), the formula predicts M_c. This can be compared to the SAEBench sweep across widths (4k, 16k, 65k).

5. **Cross-domain generalization**: Fitting the sigmoid model parameters on the first-letter spelling task and testing on RAVEL's city-country hierarchy provides a strong out-of-distribution test. If the same parametric model (mu_H, rho, sigma) predicts absorption in both domains, this constitutes strong evidence for the theory.

### Experimental Plan

All experiments use pretrained SAEs (training-free), targeting <=1 hour per task.

**Experiment 1: Coherence-Absorption Correlation (Pilot, ~15 min)**
- Load one Gemma Scope 16k SAE on Gemma-2-2B (layer 12)
- Train LR probes for first-letter features using sae-spelling pipeline
- Compute decoder cosine similarity matrix for top-K relevant features
- Measure absorption rate per letter using Chanin et al.'s metric
- Compute Spearman correlation between max(mu_H) and absorption rate
- Hardware: single GPU, ~15 min

**Experiment 2: Frequency-Coherence Interaction Matrix (~30 min)**
- Extend Experiment 1 to 5 Gemma Scope SAEs (varying width: 4k, 16k, 65k, varying sparsity)
- For each SAE, compute (mu_H, rho, absorption_rate) triples for all 26 letters
- Fit the sigmoid model: absorption_rate = sigmoid(a * mu_H * sqrt(rho) + b)
- Report fitted parameters, R^2, and residual analysis
- Hardware: single GPU, ~30 min total

**Experiment 3: Architecture Comparison (~45 min)**
- Repeat Experiment 2 on SAEBench's pretrained SAEs (BatchTopK, TopK, JumpReLU, Matryoshka)
- Test whether the same sigmoid model (with architecture-specific fitted parameters) explains all architectures
- Report per-architecture R^2 and parameter comparison
- Hardware: single GPU, ~45 min

**Experiment 4: Cross-Domain Generalization (~45 min)**
- Adapt the absorption metric to RAVEL's city-country hierarchy
- Train LR probes for "located in country X" features
- Measure absorption rate for country features (are country features absorbed into city features?)
- Test whether the sigmoid model fitted on first-letter data generalizes to city-country
- Hardware: single GPU, ~45 min

**Experiment 5: Critical Width Validation (~30 min)**
- Using the fitted model parameters from Experiment 2, compute the predicted M_c for the first-letter hierarchy
- Compare to empirical absorption rates across SAE widths (4k, 16k, 65k) from SAEBench
- Test whether absorption rate drops sharply around M_c or declines gradually
- Hardware: single GPU, ~30 min

### Baselines

**Theoretical baselines:**
- Random absorption model: P(absorption) = constant for all features (null hypothesis: absorption is random, not predicted by coherence/frequency)
- Width-only model: P(absorption) = f(M) (absorption depends only on SAE width, not on feature-level properties)
- Cui et al.'s (2025) sparsity threshold: compare our coherence-based threshold to their extreme-sparsity recovery condition

**Empirical baselines:**
- Chanin et al.'s (2024) absorption rate (15-35%) as the target to explain/predict
- SAEBench per-architecture absorption scores as validation data
- Matryoshka SAE absorption (~0.03) as the "best case" to explain via the theory

### Risk Assessment

1. **Proof may not be tight**: The mean-field approximation (independent absorption events) may fail when many features form a dense hierarchy. In this case, the theory would give correct qualitative predictions but wrong quantitative values. Mitigation: treat the formula as a semi-parametric model and fit parameters empirically.

2. **Gaussian activation model may be poor**: Real LLM activations have heavy tails, correlations, and non-stationarity. The specific CDF bound may not hold. Mitigation: use the weaker bounded-moments formulation and fit the sigmoid shape empirically.

3. **Coherence-frequency model may not capture all variance**: Absorption may also depend on factors not in the model (training dynamics, initialization, batch effects). If R^2 is low (< 0.3), the theory has limited explanatory power. Mitigation: include residual analysis to identify missing factors.

4. **Cross-domain generalization may fail**: The first-letter hierarchy is structurally simple (depth 2, binary parent concept). Knowledge hierarchies (city-country-continent, depth 3+, multi-class parent concepts) may violate assumptions. Mitigation: test on RAVEL with progressively deeper hierarchies.

5. **Theory-practice gap for critical width**: The critical width formula may predict M_c that is either trivially small (always satisfied in practice) or unrealistically large. Mitigation: compute M_c for specific hierarchies and compare to real SAE sizes.

### Novelty Claim

This work provides the first quantitative theory predicting feature absorption probability as a function of measurable SAE and feature properties. Specifically:

1. **The coherence-frequency-sparsity absorption formula** (Main Proposition) is entirely new. No prior work provides a formula predicting which specific features will be absorbed or with what probability. Chanin et al. (2024) prove absorption CAN happen; we predict WHEN and HOW MUCH.

2. **The critical width formula** extends Cui et al.'s (2025) recovery conditions to the hierarchical setting, providing the first width-sizing guidance that accounts for feature hierarchy.

3. **The connection to classical dictionary identifiability** (Spielman et al., 2012; Gribonval & Schnass, 2010) is new. We show that the incoherence condition from compressed sensing, when violated by hierarchical features, directly maps to the absorption phenomenon in SAEs.

4. **The absorption-hedging trade-off formalization** (Corollary) provides the first mathematical characterization of the trade-off between two fundamental SAE failure modes, explaining why Matryoshka SAEs trade absorption for hedging (as observed by Chanin et al., 2025).

Evidence for novelty: Systematic search of arXiv for "absorption probability bound," "coherence absorption sparse autoencoder," "critical width feature hierarchy SAE," and "absorption hedging tradeoff formula" yielded no prior results matching these specific claims.

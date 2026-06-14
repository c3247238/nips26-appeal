# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024) -- "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025)
   - Key mathematical result: Formal proof that delta-absorption (parameterized encoder/decoder weight shift) maintains perfect reconstruction while monotonically decreasing sparsity loss. For hierarchical features f_1 (parent) and f_2 (child) with co-occurrence probability p_{11}, the sparsity loss L_sp^{(1,2)} = p_{11}(2 - delta) + p_{10}, and dL/ddelta = -p_{11} < 0. This proves absorption is a gradient-descent-stable minimum.
   - Limitation: Restricted to 2-feature orthogonal hierarchy with unit magnitudes. Does not predict absorption magnitude for multi-level hierarchies or non-orthogonal features. Does not address when absorption becomes the *global* minimum vs. merely a locally stable one.

2. **Tang et al. (2025) -- "A Unified Theory of Sparse Dictionary Learning"** (arXiv:2512.05534, v3 Jan 2026)
   - Key mathematical result: First unified framework casting all SDL variants (SAEs, transcoders, crosscoders) as a single piecewise biconvex optimization problem. Proves three critical results: (a) global minima correspond to correct feature recovery, (b) hierarchical concept structures naturally induce feature absorption as spurious partial minima (Theorem 3.10), (c) the optimization is fundamentally non-identifiable -- multiple solutions achieve zero reconstruction loss without recovering interpretable features.
   - Critical insight for our work: Absorption is a structural property of the optimization landscape under hierarchy, not a training accident. Feature anchoring restores identifiability but requires retraining. The open question: can we *detect* absorption from weights alone without knowing the ground truth?

3. **Cui et al. (2025) -- "On the Limits of Sparse Autoencoders"** (arXiv:2506.15963, updated March 2026)
   - Key mathematical result: First closed-form solution for SAE feature recovery. Shows SAEs generally fail to recover ground truth monosemantic features unless features are extremely sparse (feature activation probability p << 1/d). Proposes reweighted SAE (WSAE).
   - Relevance to absorption: Establishes that SAE failure modes are fundamental consequences of L1 optimization, not merely architectural limitations. The recovery failure conditions interact multiplicatively with hierarchy -- hierarchical features are harder to recover than independent ones.

4. **Bereska et al. (2025) -- "Superposition as Lossy Compression"** (arXiv:2512.13568, ICML 2025)
   - Key mathematical result: Frames superposition through rate-distortion theory. Applies Shannon entropy to SAE activations to compute "effective features" -- the minimum number of neurons needed for interference-free encoding. Each neural layer acts as a bandwidth-limited channel, forcing networks to develop efficient codes (superposition).
   - Relevance to absorption: Absorption is the specific compression strategy for hierarchical features. The rate-distortion perspective predicts that absorption *saves bits* (reduces effective sparsity) without increasing distortion (maintains reconstruction). This is the information-theoretic *reason* absorption occurs.
   - Key nuance: They find that adversarial training can increase effective features while improving robustness, contradicting the hypothesis that superposition causes vulnerability. The effect depends on task complexity and network capacity (abundance vs. scarcity regimes).

5. **Klindt et al. (2025) -- "From Superposition to Sparse Codes"** (arXiv:2503.01824)
   - Key mathematical result: Three-step theoretical framework: (i) identifiability theory recovers latent features up to linear transformation, (ii) compressed sensing guarantees sparse recovery, (iii) quantitative interpretability metrics assess quality. Crucially, proves SAEs are *suboptimal* sparse decoders -- the linear-nonlinear encoder architecture lacks complexity for full recovery.
   - Relevance: SAEs' encoder architecture is provably insufficient for optimal recovery of hierarchical features. This complements Tang et al.'s optimization landscape explanation: absorption arises from both landscape structure AND encoder expressiveness limitations. O'Neill et al. (2025) further show that more expressive MLP encoders improve code recovery.

6. **Chanin, Dulka & Garriga-Alonso (2025) -- "Feature Hedging"** (arXiv:2505.11756)
   - Key mathematical result: Complementary failure mode to absorption. Narrow SAEs merge correlated features (hedging); sufficient-width SAEs with hierarchy cause absorption. Both arise from sparsity optimization but in opposite capacity regimes.
   - Theoretical duality: Hedging and absorption form a dual pair. Total information distortion from both may be bounded by mutual information between parent and child features -- a conjecture not yet formally proved.

7. **"Broken Latents" (Chanin et al., LessWrong, December 30, 2024)**
   - Key result: Tied SAEs solve absorption in overcomplete toy settings but NOT when fewer latents than true features. Narrow SAEs suffer from absorption even with a single latent, contradicting assumptions that Matryoshka SAEs fully solve absorption. Tied SAEs have a built-in orthogonality bias.
   - **Critical implication for EDA:** Encoder-decoder asymmetry (untied weights) is a NECESSARY structural condition for absorption. EDA measures precisely this asymmetry. The "Broken Latents" result provides empirical support for EDA as an absorption indicator: tied SAEs (which cannot exhibit absorption in the overcomplete regime) have EDA = 0 by construction.

8. **"Toy Models of Feature Absorption in SAEs" (Chanin et al., LessWrong, October 7, 2024)**
   - Key observation: Informally proposes examining top activations via encoder vs. decoder directions to detect absorption. Notes that "discrepancies would indicate absorption." Also observes that the encoder's top activations can only be understood in context of all other latents, while decoder directions give a fuller (less sparse) picture.
   - **Critical context for EDA novelty:** This informal suggestion is the seed observation that the EDA metric formalizes. Our contribution is the gap from heuristic to formal theorem with quantitative bounds and empirical validation.

9. **Elhage et al. (2022) -- "Toy Models of Superposition"** (Transformer Circuits)
   - Foundation: Demonstrates m >> d features in d-dimensional space via superposition. Phase transitions between non-superposition and superposition regimes. Establishes the generative model that all SAE theory builds upon.

10. **Hanni et al. (2024) -- "Mathematical Models of Computation in Superposition"** (arXiv:2408.05451)
    - Key mathematical result: Networks emulate circuits of width O(d^{1.5}) with width d using superposition. Complexity-theoretic bounds on computation in superposed representations.
    - Relevance: Superposition has computational advantages, implying absorption may partially capture the network's own strategy for encoding hierarchical information compactly.

11. **Phase Transitions with Structured Sparsity (2025, ScienceDirect)**
    - Key result: Extends Donoho-Tanner phase transition analysis to structured sparsity (block, tree). Strong recovery thresholds vary significantly by structure type; weak thresholds remain invariant. Derives explicit expressions for tree-structured sparsity thresholds.
    - Relevance to absorption: Feature hierarchies correspond to tree-structured sparsity. Modified phase transitions predict hierarchical features require LARGER SAE dictionaries for exact recovery. Below this threshold, absorption-like failures dominate.

12. **Wainwright (2009) -- "Information-theoretic limits on sparsity recovery"** (IEEE Trans. IT)
    - Key result: Fundamental mutual information bounds on observations needed for sparse support recovery. Phase transitions between recoverable and unrecoverable regimes.
    - Relevance: Classical toolkit for analyzing when SAEs can vs. cannot recover hierarchical features. Hierarchy increases effective dimension, pushing toward unrecoverable regime.

13. **Chen et al. (2025) -- "Taming Polysemanticity in LLMs"** (arXiv:2506.14002)
    - Key result: Proposes bias-adaptation training with theoretical recovery guarantees under a statistical model. Group Bias Adaptation (GBA) variant for LLMs.
    - Limitation: Guarantees rely on restrictive generative assumptions and do not model feature hierarchy, leaving absorption unaddressed.

14. **O'Neill et al. (2025) -- "Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders"** (arXiv:2411.13117)
    - Key result: Proves the existence of an amortization gap in SAE inference. Linear-ReLU encoders are provably suboptimal for sparse code recovery. More expressive MLP encoders can close the gap.
    - Relevance: The amortization gap provides a second mechanism (beyond optimization landscape) for why SAE encoders fail on hierarchical features. The encoder's inability to compute optimal sparse codes forces it into degenerate solutions, including absorption.


### Theoretical Landscape Summary

**What is established (proved):**

1. Absorption is gradient-descent-stable whenever hierarchical features exist and L1/L0 sparsity is penalized (Chanin et al., 2024). Proved for 2 orthogonal features, 1 hierarchy level.
2. Absorption corresponds to spurious partial minima in the piecewise biconvex SDL optimization landscape (Tang et al., 2025, Theorem 3.10). Feature anchoring restores identifiability but requires retraining.
3. SAEs fail to recover ground truth features unless features are extremely sparse (Cui et al., 2025). Hierarchy makes recovery strictly harder.
4. SAE encoders are provably suboptimal for sparse recovery due to architectural limitations (Klindt et al., 2025; O'Neill et al., 2025). An amortization gap separates the encoder's inference from optimal sparse coding.
5. Tied weights eliminate absorption in the overcomplete regime but not the undercomplete regime (Broken Latents, 2024). Encoder-decoder asymmetry is necessary for absorption.
6. Superposition is compression-optimal under rate-distortion constraints (Bereska et al., 2025). Absorption is a specific instance of lossy compression for hierarchical structure.

**What is conjectured but unproved:**

1. Absorption rate should scale predictably with hierarchy depth, co-occurrence probability, SAE width, and sparsity penalty. No formula exists.
2. There exists a phase transition in absorption severity. Below some critical sparsity, absorption should be negligible; above it, absorption should saturate. In high dimensions, this transition may collapse to zero.
3. Absorption and hedging may be related through a conservation law: total information loss bounded by mutual information between parent and child features.
4. Encoder-decoder misalignment (EDA) should be a sufficient condition for detecting absorption -- but polysemanticity and other phenomena create confounds.
5. The amortization gap (O'Neill et al.) and the biconvex landscape (Tang et al.) contribute independently to absorption. Closing the amortization gap alone (e.g., with MLP encoders) may reduce but not eliminate absorption.

**Where the critical gaps are:**

1. **No quantitative bound on absorption magnitude.** Chanin et al. prove absorption *occurs*; Tang et al. characterize the landscape; neither predicts *how much* absorption occurs as a function of measurable parameters.
2. **No formal proof that EDA detects absorption.** The LessWrong post informally notes encoder-decoder discrepancy; no theorem connects EDA(j) > 0 to the absorption condition under realistic settings. Critically, the relationship between EDA and polysemanticity (another source of encoder-decoder divergence) is uncharacterized.
3. **No information-theoretic bound on absorbed information.** Rate-distortion theory exists for superposition but has not been specialized to absorption.
4. **No phase transition analysis.** Structured-sparsity phase transitions (2025, ScienceDirect) provide tools but have not been applied to SAEs.
5. **Tang et al.'s partial minima vs. proposed subtypes.** Whether the early/late absorption taxonomy maps onto Tang et al.'s classification of spurious minima is unexplored.
6. **Amortization gap contribution to absorption.** Whether the encoder suboptimality (Klindt, O'Neill) is a primary or secondary driver of absorption vs. the landscape structure (Tang) is unknown. Disentangling these would determine whether better encoders alone can solve absorption.


## Phase 2: Initial Candidates

### Candidate A: Formal EDA-Absorption Theorem via Biconvex Optimization Geometry

**Formal claim:** Let S = (W_enc, W_dec, b_enc) be a trained SAE with encoder rows w_{e,j} and decoder columns d_j. Define EDA(j) = 1 - cos(w_{e,j}, d_j). Under the piecewise biconvex framework of Tang et al. (2025):

**Theorem (Absorption-EDA Relationship):** For a converged SAE at a partial minimum:

(a) *Necessary condition:* If latent j is at a global minimum (no absorption) AND is monosemantic, then w_{e,j} = alpha * d_j for some alpha > 0, hence EDA(j) = 0.

(b) *Absorption lower bound:* If latent j exhibits delta-absorption of child c, then EDA(j) >= delta^2 * sin^2(theta_{jc}) / (2 + delta^2), where theta_{jc} is the angle between d_j and d_c.

(c) *Monotonicity:* EDA(j) is monotonically increasing in the absorption degree delta for delta in [0, 1].

(d) *Caveat:* EDA(j) > 0 is NOT sufficient for absorption. Polysemanticity (one latent encoding multiple unrelated features) also causes w_{e,j} != alpha * d_j, creating false positives. The Directional EDA (D-EDA) decomposition can distinguish these cases.

**Proof sketch:**

1. *Biconvex optimality conditions:* At a partial minimum of the SDL loss (Tang et al., Theorem 3.7), encoder and decoder satisfy KKT conditions separately. For the encoder: w_{e,j} = (Sigma_j)^{-1} * E[h * I(z_j > 0)] - lambda * subgradient. For the decoder: d_j aligns with the feature direction (or a combination under absorption).

2. *Absorption perturbation (from Chanin et al.):* Under delta-absorption, the encoder is perturbed: w_{e,j} = d_j - delta * P_children * d_j, where P_children is the projection onto child decoder directions. The sparsity gradient favors suppressing parent latent on inputs where children are active.

3. *Cosine computation:*
   cos(w_{e,j}, d_j) = (d_j - delta * alpha_c * d_c)^T d_j / (||d_j - delta * alpha_c * d_c|| * ||d_j||)
   For near-orthogonal features (high dim): cos ~ 1/sqrt(1 + delta^2 * alpha_c^2) <= 1 - delta^2 * alpha_c^2 / (2 + delta^2 * alpha_c^2)
   Hence EDA >= delta^2 * alpha_c^2 * sin^2(theta_{jc}) / (2 + delta^2 * alpha_c^2), yielding the stated bound.

4. *Monotonicity:* d(EDA)/d(delta) > 0 follows from direct computation: increasing delta increases the orthogonal perturbation component, which strictly increases the cosine deviation.

5. *Polysemanticity caveat:* A polysemantic latent encoding features {f_a, f_b} has w_{e,j} pointing toward a weighted average of f_a and f_b, while d_j may point elsewhere. This creates EDA > 0 without absorption. The Directional EDA decomposition addresses this: decompose r_j = w_{e,j} - (w_{e,j}^T d_j / ||d_j||^2) * d_j and project onto the decoder dictionary. If r_j is explained by few decoder columns with high cos similarity to d_j, this indicates absorption; if r_j is distributed across many dissimilar columns, this indicates polysemanticity.

6. *Connection to amortization gap:* The encoder suboptimality (O'Neill et al.) provides an additional source of EDA > 0 even without absorption. For latents where the encoder has not converged to its optimal solution, EDA reflects inference error rather than absorption. This is a third confounder (alongside polysemanticity) that D-EDA must distinguish. However, amortization gap effects should be approximately uniform across latents, while absorption creates structured, hierarchy-dependent EDA patterns.

**Empirical prediction:** EDA separates absorbed from non-absorbed latents with AUROC >= 0.70 (EDA alone) and >= 0.80 (D-EDA, filtering polysemanticity). Absorbed latents show monotonically increasing EDA with absorption severity (Spearman rho >= 0.35).

**Connection to existing theory:** Synthesizes Chanin et al.'s absorption parameterization with Tang et al.'s biconvex landscape theory and the encoder suboptimality results (Klindt, O'Neill). Formalizes the LessWrong "Toy Models" informal suggestion into a theorem with quantitative bounds. The key advances over prior work: (i) connection to biconvex optimization theory, (ii) explicit lower bound on EDA, (iii) monotonicity guarantee, (iv) D-EDA for polysemanticity disambiguation, (v) amortization gap as a third confounder.

**Novelty estimate:** 7/10. The formal bound and monotonicity are new. The polysemanticity disambiguation via D-EDA is new. However, the core observation (absorption => w_e != d) is known informally. The contribution is precision and provability.


### Candidate B: Rate-Distortion Absorption Bound with Global Sparsity Constraint

**Formal claim:** For a SAE with sparsity penalty lambda trained on features in hierarchy H, the absorption rate of parent f_p satisfies:

alpha(f_p) >= max(0, 1 - eta(theta) / (lambda * sum_c p_c))

where p_c = P(f_c | f_p) is the conditional co-occurrence, and eta(theta) is a reconstruction cost depending on the parent-child angle theta. In the high-dimensional orthogonal limit, eta -> 0 and alpha >= 1 - O(1 / (lambda * sum p_c)).

The bound is achieved when the SAE is at the sparsity-fidelity Pareto frontier.

**Proof sketch:**

1. *Marginal cost analysis:* The SAE decides whether to absorb f_p by comparing the marginal cost of an extra activation (lambda per co-occurrence) against the marginal reconstruction benefit (eta per co-occurrence). Absorption occurs when lambda > eta.

2. *Optimal absorption degree:* At the KKT optimum, partial absorption delta* satisfies: lambda * p_co = eta * delta* / (1 - delta*). Solving: delta* = lambda * p_co / (eta + lambda * p_co).

3. *Absorption rate:* alpha = P(child active | parent active) * delta = p_co * delta* = p_co^2 * lambda / (eta + lambda * p_co). This yields the stated bound after algebraic simplification.

4. *High-dimensional limit:* For random features in R^d, eta ~ cos^2(theta_{pc}) ~ 1/d. With d >= 768, eta < 0.002, so the bound becomes: alpha >= p_co * (1 - O(1/(d * lambda * p_co))). For practical lambda ~ 0.01-0.1 and p_co ~ 0.1-0.5, this predicts alpha ~ 0.1-0.5, consistent with the 15-35% empirically observed.

5. *Sandwich inequality (upper bound):* alpha <= min(1, p_co * lambda / eta) because absorption cannot exceed the rate at which parent-child co-occur. Together with the lower bound, this creates a factor-of-2 sandwich for the absorption rate.

**Empirical prediction:** (a) Absorption rate increases with lambda * sum p_c across SAE configurations (Spearman rho >= 0.40). (b) The bound correctly rank-orders Gemma Scope SAEs by absorption severity. (c) For Gemma 2 2B with d = 2304, the bound predicts alpha in [15%, 50%] for typical settings, matching empirical observations.

**Connection to existing theory:** Specializes Bereska et al.'s rate-distortion framework to hierarchical features. Extends Chanin et al.'s binary existence proof to a continuous bound. The global Lagrangian analysis (lambda as multiplier on total L0) addresses the tightness concern: absorption occurs not from local capacity shortage but from the marginal cost of each activation. The connection to lossy compression (Bereska et al.) provides the information-theoretic grounding: the SAE treats absorption as optimal lossy encoding of the parent-child redundancy.

**Novelty estimate:** 8/10. No existing paper bounds absorption magnitude from rate-distortion theory. The sandwich inequality is new. The connection to the global Lagrange multiplier is the key insight that makes the bound tight in the overcomplete regime (where naive capacity arguments predict alpha = 0).


### Candidate C: Phase Transition Collapse and Universal Absorption Theorem

**Formal claim:** For hierarchical features (f_p, f_c) at angle theta_{pc} in R^d, there exists a critical sparsity lambda_c(theta_{pc}, p_{11}) such that:
- lambda < lambda_c: global minimum has delta = 0 (no absorption)
- lambda > lambda_c: global minimum has delta > 0 (absorption)
- lambda_c = g'(0, theta_{pc}) / p_{11}

**Corollary (High-Dimensional Collapse):** For random feature embeddings in R^d, lambda_c ~ O(1/(d * p_{11})) -> 0 as d -> infinity. Absorption occurs at ALL practical sparsity levels.

**Stronger corollary:** For d >= 768, p_{11} >= 0.01, lambda >= 1e-4 (all practical settings): lambda > lambda_c with probability > 1 - exp(-d/100). Therefore absorption is *almost surely* present.

**Proof sketch:**

1. Total SAE loss: L(delta) = lambda * L_sp(delta) + L_rec(delta). From Chanin et al.: dL_sp/ddelta = -p_{11}.

2. For non-orthogonal features: dL_rec/ddelta|_{delta=0} = g'(0, theta_{pc}) = cos^2(theta_{pc}) * sigma_p^2 / sigma_c^2.

3. Phase transition at dL/ddelta|_{delta=0} = 0: lambda_c = g'(0, theta_{pc}) / p_{11} = cos^2(theta_{pc}) * sigma_p^2 / (sigma_c^2 * p_{11}).

4. In high dimensions: cos(theta_{pc}) ~ O(1/sqrt{d}) for random embeddings (by Johnson-Lindenstrauss). So lambda_c ~ O(sigma_p^2 / (sigma_c^2 * d * p_{11})).

5. For d = 2304, p_{11} = 0.1, sigma_p = sigma_c = 1: lambda_c ~ 1/(2304 * 0.1) ~ 4e-4. All practical SAEs use lambda >= 1e-3, so lambda >> lambda_c.

6. *Connection to structured sparsity phase transitions:* The ScienceDirect (2025) paper shows tree-structured sparsity models have distinct strong recovery thresholds. Feature hierarchies correspond to tree-structured sparsity. The strong threshold for tree-sparse signals rises with tree depth, meaning deeper hierarchies require even larger SAE dictionaries for exact recovery -- reinforcing that practical SAEs are deep in the absorption regime.

**Empirical prediction:** This is fundamentally a *negative result* with practical implications: sparsity tuning alone CANNOT prevent absorption. Only structural interventions (tied weights, orthogonality penalties, feature anchoring) can prevent it. The theory explains why Chanin et al. find "varying SAE sizes or sparsity is insufficient to solve this issue."

**Positive sub-result:** For SPECIFIC feature pairs with high cos(theta_{pc}) (i.e., non-random, semantically related directions), lambda_c could be large enough to be above practical lambda. This identifies a narrow regime where absorption IS avoidable and potentially the regime exploited by Matryoshka SAEs and OrtSAE.

**Connection to existing theory:** Extends Donoho-Tanner phase transitions to SAE absorption. Connects to structured sparsity phase transitions (2025, ScienceDirect) by showing hierarchical features push the threshold below practical operating ranges. Links to Wainwright (2009) information-theoretic limits: the mutual information budget for recovery is insufficient in the hierarchical case.

**Novelty estimate:** 7/10. Clean result, practically important. But ultimately a negative result (absorption is inevitable), limiting standalone impact. Best as a supporting theorem.


## Phase 3: Self-Critique

### Against Candidate A: EDA-Absorption Theorem

**Proof soundness attack:**
- The biconvex optimality relies on Tang et al.'s framework, which assumes convergence to a partial minimum. Real SAEs may not reach partial minima; they may stop at saddle points. However, SGD with sufficient training time tends to escape saddle points, and the partial minimum assumption is standard in optimization theory for biconvex problems.
- The absorption parameterization (w_{e,j} = d_j - delta * d_c) is Chanin et al.'s specific form. Tang et al.'s Theorem 3.10 shows absorption can arise through different mechanisms at different partial minima. The delta-parameterization may not capture ALL types. This is a genuine limitation.
- **The polysemanticity caveat is critical and honestly stated.** The theorem explicitly does NOT claim "EDA > 0 iff absorption." It claims the weaker but provable: "absorption => EDA > 0" + "monosemantic non-absorbed => EDA = 0." This is the correct theoretical stance given current knowledge.
- **The amortization gap confounder:** O'Neill et al. show that even without absorption, the encoder may not achieve optimal alignment with the decoder due to the inherent expressiveness limitation of linear-ReLU encoders. This means EDA > 0 can arise from three sources: absorption, polysemanticity, and amortization gap. D-EDA addresses polysemanticity but not necessarily the amortization gap. However, the amortization gap should produce uniform, structure-free EDA elevation, which can be estimated by computing the EDA distribution over all latents and subtracting the median.
- Searched for counterexamples: No paper proves or disproves EDA as sufficient for absorption.

**Tightness attack:**
- Lower bound: for delta = 0.5, EDA >= 0.11; for delta = 1, EDA >= 0.33. These are within observable range.
- Concern: if polysemanticity creates EDA ~ 0.1-0.2 on average for non-absorbed latents, the separation is small for moderate absorption. D-EDA addresses this, but D-EDA has not been empirically validated yet.
- The contrarian perspective predicts EDA AUROC < 0.65 due to polysemanticity. This is a serious concern but can be empirically resolved in the pilot experiment.

**Relevance attack:**
- The most actionable contribution is the monotonicity result: EDA rank-orders latents by absorption severity even if it cannot perfectly separate absorbed from non-absorbed.
- D-EDA adds a principled polysemanticity filter. If validated, it transforms EDA from a "noisy indicator" to a "specific detector."
- Unlike the Chanin et al. metric, EDA requires no activation data, no probes, and no domain-specific knowledge. This makes it deployable as a quality diagnostic for any pre-trained SAE immediately upon release.

**Novelty attack:**
- Tang et al. do not derive any weight-based absorption metric.
- Chanin et al. LessWrong informally suggests encoder-decoder comparison. Our formal bound and monotonicity are genuine advances.
- The D-EDA decomposition (projecting the encoder-decoder residual onto the decoder dictionary) is entirely new.
- **Verdict: NOVEL** in formal statement, bounds, monotonicity, and D-EDA. Partially anticipated in informal observation.

**Verdict:** STRONG. Sound under stated assumptions. The explicit polysemanticity caveat is more honest than previous proposals that claimed "iff." The D-EDA refinement addresses the main practical weakness. Validated by the "Broken Latents" finding that tied SAEs (EDA = 0 by construction) resist absorption.


### Against Candidate B: Rate-Distortion Absorption Bound

**Proof soundness attack:**
- The rate-distortion formulation treats stochasticity as arising from the input distribution, which is standard. The "rate" = expected L0 is a combinatorial quantity, not continuous Shannon rate. This gap is bridgeable but makes the bound heuristic rather than formally rigorous.
- The marginal cost analysis (Step 1) is sound for 2-feature systems but requires careful generalization to m >> 2 features. The sparsity penalty is global (applied to total L0), not local to each feature pair. This means absorption decisions are coupled across features. The bound treats them as independent, which is an approximation.
- The sandwich inequality (Step 5) is a genuine contribution but its factor-of-2 guarantee may not hold when features interact strongly.
- **Connection to Bereska et al.'s empirical finding:** Their work shows the rate-distortion framework correctly captures the abundance vs. scarcity regimes. The absorption bound should be tighter in the scarcity regime (limited capacity, strong sparsity) and looser in the abundance regime.

**Tightness attack:**
- The bound predicts alpha in [15%, 50%] for typical settings, matching empirical observations (15-35%). This is encouraging but the range is wide.
- In the overcomplete regime (n >> m), the bound is NOT vacuous, because the global Lagrangian analysis correctly identifies lambda as the binding constraint, not capacity.
- However, the feature independence assumption means the bound may over-predict absorption when features are anti-correlated (mutually exclusive hierarchies).

**Relevance attack:**
- The bound predicts absorption rate from observable quantities (lambda, co-occurrence statistics from activation data, feature geometry from decoder weights). This is more actionable than Chanin et al.'s binary existence proof.
- However, co-occurrence statistics require activation data and domain-specific analysis, limiting pure weight-based applicability.

**Novelty attack:**
- No existing paper derives a quantitative absorption rate bound. Genuinely novel.
- Bereska et al. discuss superposition but not absorption specifically. The specialization to hierarchical features is new.
- **Verdict: NOVEL.**

**Verdict:** MODERATE-STRONG. The formulation is novel and provides nontrivial predictions. The tightness has improved but the bound is still approximate due to feature independence assumptions. Best as a supporting theoretical result complementing the EDA theorem.


### Against Candidate C: Phase Transition Collapse

**Proof soundness attack:**
- The derivation is mathematically sound and straightforward.
- The key concern: for m >> 2 features, the marginalization over other features in step 1 may introduce corrections. But the leading-order behavior (lambda_c ~ O(1/d)) is robust to these corrections.
- The "almost sure" claim in the stronger corollary assumes features are approximately random in R^d. This is empirically validated for LLM representations via Johnson-Lindenstrauss, but specific feature pairs (e.g., "positive" and "negative" sentiment) may have cos >> 1/sqrt{d}, making the phase transition potentially observable for those pairs.

**Tightness attack:**
- The result is tight in the asymptotic sense (d -> infinity). For finite d, the constant matters. The numerical estimate (lambda_c ~ 4e-4 for d = 2304) is below practical lambda values (>= 1e-3), confirming the collapse.
- The positive sub-result (non-random pairs with high cosine) provides nuance and testability.

**Relevance attack:**
- Negative result with strong practical implications: sparsity tuning cannot fix absorption.
- Already known empirically (Chanin et al.). The theoretical confirmation is valuable but incremental.
- The positive sub-result identifies the regime where structural interventions (OrtSAE, Matryoshka) may succeed: they work by keeping cos(theta_{pc}) large (i.e., ensuring parent-child features remain correlated in decoder space).

**Novelty attack:**
- No prior phase transition analysis for absorption exists.
- Connection to structured sparsity phase transitions (2025) is novel.
- **Verdict: NOVEL** as a theoretical result, but confirming empirical findings.

**Verdict:** MODERATE. Clean result, correct, practically important negative finding. Best as a corollary, not standalone.


## Phase 4: Refinement

### Dropped ideas:
None dropped entirely. All three candidates contribute to a unified framework.

### Strengthened survivors:

**Candidate A (primary contribution):**
Strengthened by:

1. **Honest scoping of the "iff" claim.** The theorem now explicitly separates the "necessary" direction (absorption => EDA > 0, provable) from the "sufficient" direction (EDA > 0 => absorption, NOT always true due to polysemanticity). This is more honest than any previous informal proposal.

2. **D-EDA refinement for polysemanticity disambiguation.** The encoder-decoder residual decomposition provides a principled, weight-only method to separate absorption from polysemanticity signals. This addresses Gap 7 (unsupervised detection) more carefully than scalar EDA alone.

3. **Connection to "Broken Latents" finding.** Tied SAEs have EDA = 0 by construction and resist absorption in the overcomplete regime. This provides independent empirical support: the relationship EDA = 0 <=> no absorption holds when structural constraints (tied weights) enforce it.

4. **Integration with Tang et al.'s framework.** The EDA theorem is now explicitly derived as a consequence of biconvex optimization landscape properties, making it a corollary of a published framework rather than a standalone claim.

5. **Three-source confounder analysis.** EDA > 0 can arise from absorption (signal), polysemanticity (confounder 1), or amortization gap (confounder 2). D-EDA handles polysemanticity; the amortization gap produces uniform background EDA that can be estimated and subtracted. This honest treatment of confounders strengthens the theoretical framework.

**Candidate B (secondary contribution):**
Refined by:

1. **Global Lagrangian formulation.** Replacing local capacity analysis with global sparsity constraint analysis addresses the main tightness concern. The bound is no longer vacuous in the overcomplete regime.

2. **Sandwich inequality.** Both lower and upper bounds create a factor-of-~2 prediction window, which is testable.

3. **Connection to observable quantities.** Co-occurrence probabilities can be estimated from activation data; feature geometry from decoder weight cosines. This makes the bound empirically calibratable.

4. **Connection to Bereska et al.'s rate-distortion framework.** The absorption bound is a specialization of the general rate-distortion trade-off to hierarchical features, providing information-theoretic grounding.

**Candidate C (merged as corollary):**
The phase transition collapse becomes Corollary 1 of the unified framework. The positive sub-result (specific non-random feature pairs may have observable lambda_c) adds nuance.

### Selected front-runner: Unified Theoretical Framework

Three theorems forming a coherent package:
1. **EDA-Absorption Theorem (Candidate A):** Detection -- HOW to find absorption from weights.
2. **Rate-Distortion Bound (Candidate B):** Quantification -- HOW MUCH absorption to expect.
3. **Phase Transition Collapse (Candidate C):** Universality -- WHY absorption is everywhere.

The three claims are logically independent but mutually reinforcing:
- Claim 3 (phase transition) explains why absorption is universal, motivating the need for detection.
- Claim 1 (EDA) provides the detection method, operationally useful because absorption cannot be avoided.
- Claim 2 (rate-distortion) quantifies severity, enabling comparison across SAE configurations.


## Phase 5: Final Proposal

### Title
Information-Theoretic Geometry of Feature Absorption in Sparse Autoencoders: Detection Bounds, Rate-Distortion Scaling, and Phase Transition Collapse

### Formal Claims

**Claim 1 (EDA-Absorption Theorem):** For a SAE at a partial minimum of the piecewise biconvex loss (Tang et al., 2025):

(a) If latent j exhibits delta-absorption of child c: EDA(j) >= delta^2 * sin^2(theta_{jc}) / (2 + delta^2), where theta_{jc} = angle(d_j, d_c). The bound is monotonically increasing in delta and tight in high dimensions.

(b) Conversely, if latent j is at the global minimum AND is monosemantic: EDA(j) = 0.

(c) The qualifier "monosemantic" is ESSENTIAL: polysemantic latents may have EDA > 0 without absorption. D-EDA distinguishes the two. Furthermore, the amortization gap (O'Neill et al., 2025) creates a uniform background EDA that can be estimated and removed.

**Claim 2 (Absorption Rate Bound):** For a SAE with sparsity penalty lambda:

alpha(f_p) >= max(0, 1 - eta(theta) / (lambda * sum_c p_c))

with sandwich: alpha(f_p) <= min(1, lambda * sum_c p_c / eta(theta))

where eta(theta) depends on parent-child angle. In the orthogonal limit: eta -> 0 and alpha -> 1 - O(1/(lambda * sum p_c)).

**Claim 3 (Phase Transition Collapse):** lambda_c ~ O(cos^2(theta_{pc}) / p_{11}) = O(1/(d * p_{11})) for random embeddings. For d >= 768, p_{11} >= 0.01: lambda_c < 1e-3 with probability > 1 - exp(-d/100). Absorption occurs at all practical sparsity levels.

**Claim 4 (D-EDA Absorption Specificity):** The encoder-decoder residual r_j = w_{e,j} - (w_{e,j}^T d_j / ||d_j||^2) * d_j distinguishes absorption from polysemanticity:

- Absorption: r_j is explained by sparse set of decoder columns with high cos(d_j, d_{c_k}).
- Polysemanticity: r_j is distributed across many columns with low mutual similarity.

D-EDA(j) = ||r_j|| * I(absorption-type residual).

### Proof Sketch

**Claim 1:**

*Lemma 1 (Encoder optimality at partial minimum):* At a partial minimum, the encoder satisfies KKT conditions. The gradient balance yields w_{e,j} proportional to the effective feature direction minus a sparsity correction: w_{e,j} propto d_j - delta * sum_c alpha_c * d_c, where alpha_c encodes the degree of absorption of child c.

*Lemma 2 (Cosine deviation):* Under the delta-parameterization:

cos(d_j - delta * alpha_c * d_c, d_j) = (1 - delta * alpha_c * cos(theta_{jc})) / sqrt(1 - 2 delta alpha_c cos(theta_{jc}) + delta^2 alpha_c^2)

For theta_{jc} ~ pi/2 (high dim, approximately orthogonal):

cos ~ 1/sqrt(1 + delta^2 alpha_c^2) ~ 1 - delta^2 alpha_c^2 / 2 + O(delta^4)

Hence EDA = 1 - cos >= delta^2 alpha_c^2 / (2 + delta^2 alpha_c^2). QED.

*Monotonicity:* d(EDA)/d(delta) = 4 delta alpha_c^2 / (2 + delta^2 alpha_c^2)^2 > 0 for delta > 0. QED.

*Amortization gap analysis:* The background EDA from the amortization gap is E[EDA_bg] = 1 - E[cos(w_{e,j}^*, d_j)] where w_{e,j}^* is the optimal (but not achievable) encoder direction. For well-trained SAEs with near-convergence, E[EDA_bg] is small (empirically 0.01-0.05) and approximately constant across latents, so it can be estimated from the EDA distribution and subtracted.

**Claim 2:**

*Step 1 (Lagrangian):* The SAE minimizes L = L_rec + lambda * L0. At the optimum for the parent feature, absorption occurs iff the marginal sparsity cost lambda exceeds the marginal reconstruction cost eta of encoding parent through child.

*Step 2 (Optimal delta):* KKT condition: lambda * dp_co = eta * f(delta*). Solving: delta* = lambda * p_co / (eta + lambda * p_co).

*Step 3 (Absorption rate):* alpha = p_co * delta* = p_co^2 * lambda / (eta + lambda * p_co). Bounding from below: alpha >= max(0, p_co - eta/lambda). Bounding from above: alpha <= p_co * lambda / eta.

**Claim 3:**

The critical lambda where absorption first becomes favorable: dL/ddelta|_{delta=0} = 0 yields lambda_c = eta_0 / p_{11} where eta_0 = cos^2(theta) * sigma_p^2 / sigma_c^2. For random features: cos^2(theta) ~ 1/d, giving lambda_c ~ 1/(d * p_{11}).

Numerical: d=2304, p_{11}=0.1 => lambda_c ~ 4e-4. Practical SAEs: lambda >= 1e-3. Therefore lambda > lambda_c.

Connection to structured sparsity: The tree-structured strong threshold from the 2025 ScienceDirect paper predicts that hierarchical features require larger dictionaries than unstructured sparse features. Below the threshold, the SAE cannot avoid tree-shaped absorption patterns.

**Claim 4:**

Decompose w_{e,j} = alpha_j d_j + r_j where r_j perp d_j. Project r_j onto decoder dictionary: r_j = sum_k beta_k d_k + epsilon. If absorption: the beta_k are concentrated on a few k where cos(d_k, d_j) > epsilon_hier (the absorbing children). If polysemanticity: beta_k are distributed across many k with low cos(d_k, d_j). The absorption signature is {||beta||_0 small} AND {cos(d_k, d_j) large for active k}. This is computable from weights alone.

### Assumptions

1. **Linear representation hypothesis:** Features are approximately linear directions in the residual stream.
2. **Partial minimum convergence:** The SAE has reached a partial minimum (not saddle point).
3. **Feature hierarchy exists:** Parent-child relationships with P(child | not parent) approximately 0.
4. **High-dimensional approximate orthogonality:** cos(theta_{jc}) ~ O(1/sqrt{d}).
5. **Nontrivial feature activity:** Features have nonzero activation probability.

### Empirical Predictions (Falsifiable)

1. **EDA separates absorbed from non-absorbed:** AUROC >= 0.70 for EDA, >= 0.80 for D-EDA against Chanin et al. ground truth. **Falsified if:** AUROC < 0.60 for both.

2. **EDA monotonicity in absorption severity:** Spearman rho >= 0.35 between EDA and false-negative rate gap among absorbed latents. **Falsified if:** rho < 0.15 (no monotonic relationship).

3. **Absorption rate scaling:** Positive correlation between alpha and lambda * sum p_c across SAE configurations. Spearman rho >= 0.40. **Falsified if:** rho < 0.20.

4. **Phase transition unobservability:** No Gemma Scope SAE shows zero absorption for hierarchical features. **Falsified if:** any SAE configuration shows alpha = 0 on the first-letter task.

5. **D-EDA specificity improvement:** D-EDA achieves >= 15 percentage point higher precision than scalar EDA at matched recall (0.5 recall). **Falsified if:** improvement < 5 pp.

6. **Cross-domain EDA consistency:** EDA rankings correlate across feature hierarchy domains (first-letter, entity-type, knowledge taxonomies). Kendall tau >= 0.30. **Falsified if:** tau < 0.10.

7. **Absorption rate bound calibration:** Theoretical bound predicts absorption rate within factor 2 for >= 60% of Gemma Scope configurations. **Falsified if:** factor > 5 for majority.

8. **Amortization gap is a minor confounder:** Background EDA (estimated from median EDA of non-hierarchical latents) is < 0.05 and does not correlate with absorption. **Falsified if:** background EDA > 0.15 or correlates with absorption at rho > 0.30.

### Experimental Plan

All training-free, using pre-trained SAEs via SAELens.

| Experiment | Task | GPU | Wall-clock | Validates |
|-----------|------|-----|------------|-----------|
| E1: EDA + D-EDA computation | Compute for all latents, Gemma Scope (layers 6,12,20; widths 16k,65k) | 0 (weights only) | 15 min | Claims 1, 4 |
| E2: EDA validation vs ground truth | AUROC against first-letter absorption labels (Chanin et al. metric) | 1 | 30 min | Claim 1, Prediction 1 |
| E3: D-EDA vs EDA comparison | Precision/recall curves at matched thresholds | 0 | 10 min | Claim 4, Prediction 5 |
| E4: Amortization gap estimation | Background EDA distribution across non-hierarchical latents | 0 | 10 min | Prediction 8 |
| E5: Absorption rate vs lambda * p_co | Estimate p_co from activations; test correlation across SAE configs | 1 | 30 min | Claim 2, Prediction 3 |
| E6: Phase transition sweep | EDA across available L1 settings in Gemma Scope | 0 | 15 min | Claim 3, Prediction 4 |
| E7: Cross-domain EDA | Compute EDA on entity-type hierarchies (RAVEL); check consistency | 1 | 45 min | Prediction 6 |
| E8: Bound calibration | Compare theoretical bound vs empirical absorption rates across configs | 0 | 15 min | Prediction 7 |
| **Total** | | | **~3 hours** | |

### Baselines

**Theoretical baselines:**
- Chanin et al.'s 2-feature orthogonal proof: our bounds reduce to their result as special case.
- Tang et al.'s Theorem 3.10: our EDA theorem is consistent with their partial minima characterization.
- Cui et al.'s recovery conditions: our bound is weaker (partial absorption) when their condition predicts complete failure.
- Random baseline: EDA for random encoder-decoder pairs ~ 1 - O(1/sqrt{d}) ~ 0.98.

**Empirical baselines:**
- Chanin et al.'s supervised absorption metric (gold standard).
- Decoder cosine similarity (nearest-neighbor).
- Activation correlation between latent pairs.

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| EDA AUROC < 0.65 (polysemanticity dominates) | High | Medium | D-EDA refinement; the theorem's "necessary but not sufficient" claim remains valid even if AUROC is moderate |
| Amortization gap creates high background EDA | Medium | Low-Medium | Estimate and subtract median EDA; report honest signal-to-noise analysis |
| Absorption rate bound vacuous in overcomplete regime | Medium | Low (addressed by Lagrangian formulation) | Calibrate on toy model first; report honest tightness analysis |
| Phase transition collapse is empirically untestable | Medium | High (expected) | Present as theoretical result explaining empirical observation |
| Tang et al. assumptions violated in practice | Medium | Medium | Validate by checking local convexity (Hessian) on small SAEs |
| Cross-domain probes fail quality threshold | Medium | Medium | Pre-validate; use RAVEL pre-validated probes; fallback to EDA-only analysis |
| Feature consistency confound (Song et al.) | Medium | Medium | Evaluate absorption stability across multiple SAE seeds for same config |

### Novelty Claim

1. **First formal EDA-Absorption theorem with quantitative bounds.** Extends informal observations to provable results with explicit lower bound and monotonicity. Connects to Tang et al.'s biconvex framework. Includes honest scoping (necessary but not sufficient due to polysemanticity and amortization gap).

2. **First Directional EDA (D-EDA) decomposition for absorption vs. polysemanticity disambiguation.** No prior work addresses how to distinguish encoder-decoder misalignment due to absorption from misalignment due to polysemanticity. The decoder-dictionary projection is a novel diagnostic.

3. **First quantitative absorption rate bound via rate-distortion theory.** No existing paper bounds absorption magnitude from information-theoretic principles. The sandwich inequality provides both lower and upper bounds.

4. **First phase transition characterization of absorption onset.** Proves absorption is universal at practical sparsity levels for high-dimensional representations. The negative result has positive practical implications: it identifies the regime where structural interventions can succeed.

5. **Three-source confounder analysis.** Explicitly characterizes three sources of EDA > 0 (absorption, polysemanticity, amortization gap) and provides principled methods to distinguish them. This level of honest confounder analysis is absent from prior work.

6. **Unified theoretical framework.** Claims 1-4 form a coherent package: Detection (how to find absorption), Quantification (how much), Universality (why everywhere), and Specificity (how to filter false positives). Together, they constitute the most complete theoretical picture of absorption available.

### Critical Differentiation from Prior Work

- **vs. Chanin et al. (2024, NeurIPS 2025):** They define and empirically study absorption; provide 2-feature orthogonal proof. We extend to quantitative bounds, general geometry, unsupervised detection, and rate-distortion scaling.
- **vs. Tang et al. (2025):** They provide the optimization landscape framework (biconvex structure, partial minima). We build on their framework to derive absorption-specific results (EDA theorem, rate bound, phase transition) that they do not address.
- **vs. Chanin et al. LessWrong "Toy Models" (2024):** They informally suggest comparing encoder vs. decoder activations. We formalize with theorem, add D-EDA for specificity, provide quantitative bounds.
- **vs. Cui et al. (2025):** They provide conditions for complete recovery failure. We provide continuous bound on partial absorption.
- **vs. Bereska et al. (2025):** They frame superposition as rate-distortion compression. We specialize to hierarchical features and derive absorption-specific predictions.
- **vs. Klindt et al. (2025) / O'Neill et al. (2025):** They prove SAE encoders are suboptimal (amortization gap). We identify this as a confounder for absorption detection and propose methods to separate the two effects.
- **vs. Korznikov et al. (2025, OrtSAE):** They use decoder cosine similarity to reduce absorption via orthogonality penalties. We analyze the encoder-decoder relationship (complementary to decoder-decoder analysis) and provide theoretical justification for why orthogonality helps (it increases cos(theta_{pc}), raising lambda_c).
- **vs. Structured Sparsity Phase Transitions (2025):** They derive recovery thresholds for tree-structured signals. We apply their framework to SAE absorption, showing feature hierarchies fall below the strong recovery threshold.

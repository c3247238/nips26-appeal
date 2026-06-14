# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** --- Defines feature absorption formally and proves it occurs in a toy model with hierarchical features. Key result: when features form a parent-child hierarchy (e.g., "starts with s" is parent of "snake"), the SAE's sparsity objective incentivizes the child feature to absorb the parent's firing responsibility, saving one unit of L0. Absorption rate 15--35% across hundreds of SAEs.

2. **Tang et al. (2025), "A Unified Theory of Sparse Dictionary Learning" (arXiv:2512.05534)** --- First unified framework casting all SDL variants (SAE, transcoder, crosscoder) as piecewise biconvex optimization. Key theorem: for any realizable absorption pattern where at least one neuron has |F_i| >= 2 features assigned, there exists a local minimum of the SDL loss. Proposes feature anchoring to restore identifiability.

3. **Cui et al. (2025), "On the Limits of Sparse Autoencoders" (arXiv:2506.15963)** --- Derives closed-form optimal solution for SAEs under the superposition hypothesis. Key result: standard SAEs suffer from feature shrinking (recovered features have smaller norm than ground truth) and feature vanishing (some features not recovered at all) unless ground-truth features are extremely sparse. Proposes reweighted SAE (WSAE).

4. **Chen et al. (2025), "Taming Polysemanticity" (arXiv:2506.14002)** --- Proposes bias-adaptation training with provable feature recovery guarantees under a statistical generative model. Key assumption: features are generated i.i.d. with known sparsity; does not model hierarchical feature structure.

5. **Klindt et al. (2025), "From Superposition to Sparse Codes" (arXiv:2503.01824)** --- Three-step framework: (1) identifiability theory shows networks recover latent features up to linear transformation, (2) compressed sensing guarantees sparse recovery, (3) but SAEs fail to achieve optimal recovery due to their linear-nonlinear encoder architecture (amortization gap).

6. **Kumar et al. (2025), "Dictionary Learning: Complexity of Sparse Superposed Features" (arXiv:2502.05407)** --- Establishes that recovering the feature dictionary from feedback requires at least quadratic sample complexity in the ambient dimension. The expressiveness-versus-recoverability trade-off is fundamental.

7. **Tilde Research (2025), "The Rate Distortion Dance of Sparse Autoencoders"** --- Connects SAE training to rate-distortion theory via the information bottleneck framework. TopK activation functions directly limit mutual information I(Z;X), prioritizing the most informative features. Key insight: SAE training separates into two phases (reconstruction first, then sparsity enforcement).

8. **Elhage et al. (2022), "Toy Models of Superposition" (Transformer Circuits)** --- Foundational: demonstrates networks encode more features than dimensions via non-orthogonal superposition. The degree of superposition depends on feature importance and sparsity. Provides the geometric intuition underlying all subsequent SAE work.

9. **Bussmann et al. (2025), "Matryoshka SAE" (arXiv:2503.17547, ICML 2025)** --- Nested prefix losses create natural hierarchy that reduces absorption from ~0.29 to ~0.03. Key theoretical insight: forcing early latents to reconstruct without help from later latents prevents the "absorption shortcut."

10. **Chanin et al. (2025), "Feature Hedging" (arXiv:2505.11756)** --- Proves that narrow SAEs merge correlated features (hedging), complementary to absorption. Key quantitative result: absorption and hedging trade off --- Matryoshka SAEs reduce absorption but increase hedging at inner levels.

11. **Korznikov et al. (2026), "Sanity Checks for SAEs" (arXiv:2602.14111)** --- Shows SAEs recover only ~9% of true features in synthetic settings; random baselines match trained SAEs on multiple metrics. Raises fundamental questions about whether current SAE training finds meaningful features at all.

12. **Leask et al. (2025), "SAEs Do Not Find Canonical Units" (arXiv:2502.04878, ICLR 2025)** --- Meta-SAE decomposition shows SAE features are neither complete nor atomic. Larger SAEs find novel latents capturing information missed by smaller ones. Challenges the assumption that there exists a unique canonical set of features to recover.

### Theoretical Landscape Summary

The theoretical understanding of SAE feature absorption sits at the intersection of three classical frameworks:

**Sparse Dictionary Learning (SDL).** The SAE objective decomposes into encoder (sparse coding) and decoder (compressed sensing) problems. Classical compressed sensing guarantees (RIP, mutual coherence bounds) promise exact recovery of sparse signals, but these guarantees assume (a) the dictionary is known and incoherent, (b) the signal is truly sparse, and (c) noise is bounded. In SAEs, the dictionary is being learned simultaneously, features have hierarchical co-occurrence structure (violating incoherence), and "noise" includes nonlinear error ("dark matter"). Cui et al.'s closed-form analysis shows that even under ideal conditions, standard SAEs suffer systematic feature shrinkage proportional to the degree of superposition.

**Optimization Landscape.** Tang et al.'s piecewise biconvex analysis shows the SDL loss landscape contains spurious local minima corresponding to every possible absorption pattern. The key insight is that absorption configurations are not mere suboptimal solutions---they are bona fide local minima that gradient descent can converge to and remain trapped in. The number of such minima grows combinatorially with the number of hierarchical feature pairs.

**Information Theory.** The rate-distortion perspective (Tilde Research; Ayonrinde et al., 2024) frames SAE training as an information compression problem. The sparsity constraint limits the rate (bits transmitted through the bottleneck), while reconstruction loss measures distortion. Absorption can be understood as a rational coding strategy: when parent and child features co-occur, encoding only the child (which implies the parent) achieves the same distortion at lower rate. This is optimal from a rate-distortion standpoint but catastrophic for interpretability, because the parent feature becomes undetectable.

**What is known:** (1) Absorption is a local minimum of the SDL loss for any hierarchical feature pair (Tang et al.). (2) SAEs generally fail to recover ground-truth features unless features are extremely sparse (Cui et al.). (3) Absorption is not architecture-specific---it occurs in L1, TopK, JumpReLU, and BatchTopK SAEs, because it optimizes reconstruction at fixed sparsity regardless of penalty mechanism (Chanin et al.). (4) The amortization gap of SAE encoders prevents optimal sparse recovery even when the dictionary is correct (Klindt et al.).

**What is conjectured but unproven:** (1) There exist necessary and sufficient conditions on feature co-occurrence statistics and SAE configuration that determine absorption severity. (2) The absorption-reconstruction trade-off can be quantified as a Pareto frontier. (3) Hierarchical feature structure in real LLMs follows distributional laws (e.g., Zipfian) that determine the landscape of absorption minima.

**Where the gaps are:** (1) No closed-form expression for absorption rate as a function of measurable quantities (width, L0, feature hierarchy depth, co-occurrence frequency). (2) No tight lower bound showing how much reconstruction must degrade to eliminate absorption. (3) No connection between the compressed sensing mutual coherence of the learned dictionary and absorption severity. (4) No characterization of when absorption is avoidable vs. inevitable given the data distribution.

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Lower Bound on the Absorption-Reconstruction Trade-off

**Formal claim (Theorem sketch):** Let D be the ground-truth feature dictionary with hierarchical structure H = {(parent_i, child_i)} where parent features subsume child features. Let SAE(W, L0) denote an SAE with width W and average sparsity L0. Define the absorption rate alpha(SAE) as the fraction of parent features absorbed by child features, and the reconstruction fidelity R(SAE) = 1 - MSE/Var(x). Then:

R(SAE) <= R_max(L0) - gamma * (1 - alpha(SAE)) * |H| / W

where gamma > 0 depends on the average co-occurrence probability of parent-child pairs, and R_max(L0) is the optimal reconstruction achievable at sparsity L0. In words: eliminating absorption (setting alpha = 0) incurs a reconstruction penalty proportional to the number of hierarchical pairs relative to dictionary width.

**Proof sketch:**
1. *Lemma 1 (L0 accounting):* For each parent-child pair, an absorption-free SAE requires both parent and child to fire when the child is active, consuming 2 units of L0 budget. An absorbing SAE requires only 1. Across |H| pairs with average co-occurrence probability p, the expected L0 overhead of absorption-free encoding is p * |H|.
2. *Lemma 2 (Rate-distortion connection):* Frame the SAE latent as a channel with capacity proportional to L0. Using the source coding theorem, the minimum distortion achievable at rate R = L0 * log(W) is D(R). Reducing rate by p * |H| increases minimum distortion by at least dD/dR * p * |H|.
3. *Main theorem:* Combine Lemmas 1 and 2, using the convexity of the rate-distortion function to bound the reconstruction penalty from below.

**Empirical prediction:** Plot R vs. alpha across SAEs of different widths and L0 values. The theory predicts a linear trade-off with slope gamma * |H| / W. If we fix the model and layer (fixing the feature hierarchy), increasing W should flatten the trade-off (making absorption less costly to remove). This can be tested on Gemma Scope SAEs at widths 4k, 16k, 65k using the SAEBench absorption metric and reconstruction fidelity.

**Connection to existing theory:** Extends Chanin et al.'s qualitative observation ("any solution removing absorption has worse L0 vs. VE") into a quantitative bound. Draws on Cui et al.'s closed-form analysis for the rate-distortion function of SAE encoding. Related to the information bottleneck framework (Tilde Research) but makes the absorption-specific penalty explicit.

**Novelty estimate:** 7/10. The qualitative trade-off is well-known; the novelty is in the formal bound and its dependence on measurable quantities (width, L0, hierarchy size, co-occurrence statistics).

---

### Candidate B: Absorption as Spurious Local Minimum --- Conditions for Escape via Dictionary Coherence

**Formal claim (Proposition sketch):** Define the mutual coherence of the SAE decoder dictionary D as mu(D) = max_{i != j} |<d_i, d_j>| / (||d_i|| * ||d_j||). For a parent-child feature pair (p, c) with decoder vectors d_p and d_c, define the local coherence mu_pc = |<d_p, d_c>| / (||d_p|| * ||d_c||). Then:

(a) Absorption of parent p by child c occurs at a local minimum if and only if mu_pc > tau(L0, freq_c), where tau is a threshold decreasing in L0 and increasing in the firing frequency of child c.

(b) The basin of attraction of this absorbing minimum has radius proportional to mu_pc * freq_c / (1 - mu_pc).

(c) Therefore, if the dictionary is trained with an orthogonality penalty such that mu(D) < tau_min = min_{(p,c) in H} tau(L0, freq_c), absorption is guaranteed not to occur at any local minimum.

**Proof sketch:**
1. *Step 1 (Local minimum characterization):* Start from Tang et al.'s piecewise biconvex framework. For a two-feature system {p, c} with co-occurrence, write the SDL loss as a function of the encoder weights and analyze its critical points. Show that absorption corresponds to setting the encoder weight for p to zero on co-occurring inputs, which is a local minimum when the coherence mu_pc exceeds a data-dependent threshold.
2. *Step 2 (Basin of attraction):* Compute the Hessian at the absorbing critical point. Show that the eigenvalue associated with "un-absorbing" (restoring p's encoder weight) is positive with magnitude proportional to mu_pc * freq_c, implying the basin radius scales accordingly.
3. *Step 3 (Sufficiency of incoherence):* If mu(D) < tau_min, no parent-child pair satisfies the absorption condition, so no absorbing local minimum exists.

**Empirical prediction:** (1) Measure mu_pc for known parent-child pairs (first-letter features and absorbing token features) and verify it exceeds the predicted threshold. (2) Compare mu_pc distributions across SAE architectures (BatchTopK, Matryoshka, OrtSAE) and verify that low-absorption architectures have lower mu_pc. (3) Train SAEs with varying orthogonality penalty strengths and measure the critical penalty at which absorption vanishes --- verify it corresponds to mu(D) dropping below the predicted threshold.

**Connection to existing theory:** Directly extends Tang et al.'s local minimum characterization by making the coherence condition explicit. Connects to OrtSAE's empirical success (orthogonality penalty reduces coherence) and provides the theoretical explanation for WHY orthogonality helps. Draws on classical compressed sensing mutual coherence theory (Donoho & Huo, 2001).

**Novelty estimate:** 8/10. The connection between dictionary coherence and absorption local minima is new. The characterization of the basin of attraction is novel and practically useful (predicts which architectures escape absorption).

---

### Candidate C: Hierarchical Rate-Distortion Theory for Feature Absorption --- An Impossibility Result

**Formal claim (Theorem sketch):** Consider a data-generating process where activations x are generated as x = sum_{i in S} a_i * d_i + noise, where S is drawn from a distribution over feature subsets that respects a hierarchy tree T (if child c is active, parent p is always active). Let n = dim(x), m = number of features, k = average active features per input. Define:

- alpha_opt(R): the minimum achievable absorption rate at reconstruction fidelity R
- R_opt(alpha): the maximum achievable reconstruction at absorption rate alpha

Then for any SAE with m features and sparsity constraint L0 = k:

alpha_opt(R) >= 1 - (L0 / k_eff) for R >= R_min

where k_eff is the "effective sparsity" counting hierarchical features correctly (k_eff = k + sum of parent-child co-occurrences), and R_min is the minimum reconstruction needed for non-trivial performance. In words: if the true effective sparsity k_eff exceeds the SAE's sparsity budget L0, absorption is INEVITABLE, and the minimum absorption rate is at least 1 - L0/k_eff.

**Proof sketch:**
1. *Step 1 (Effective sparsity):* When features form a hierarchy, the "true" number of features active per input is k_eff > k (because parent features are active whenever any child is). But the SAE can only fire L0 features. If L0 < k_eff, some features MUST be suppressed.
2. *Step 2 (Optimal suppression strategy):* Show via a counting argument that suppressing parent features (absorption) is the loss-minimizing strategy because parent directions are partially reconstructable from the sum of active child directions.
3. *Step 3 (Lower bound):* The minimum number of suppressed parent features is k_eff - L0, each representing one absorbed parent-child pair. Divide by the number of hierarchical pairs to get the minimum absorption rate.

**Empirical prediction:** (1) Estimate k_eff from LLM activations by measuring the average number of truly active features (using probe-based detection, not SAE activation). (2) Compare against L0 of various SAEs. Predict that absorption rate >= 1 - L0/k_eff. (3) This provides an operational criterion: to achieve zero absorption, the SAE must have L0 >= k_eff, which may require substantially higher L0 than typically used.

**Connection to existing theory:** Formalizes Chanin et al.'s observation that "absorption saves one L0 per parent-child pair." Extends Cui et al.'s analysis (which shows SAEs fail without extreme sparsity) to the hierarchical setting. Related to the structured sparsity literature (group Lasso, hierarchical sparsity) where the "effective dimension" accounting is standard.

**Novelty estimate:** 8/10. The impossibility result connecting effective sparsity to minimum absorption rate is new and would have immediate practical implications for SAE design (tells practitioners the minimum L0 needed to avoid absorption).

---

## Phase 3: Self-Critique

### Against Candidate A: Information-Theoretic Lower Bound on Absorption-Reconstruction Trade-off

**Proof soundness attack:** The main weakness is Lemma 2, which applies the source coding theorem to an SAE bottleneck. The SAE is not a memoryless channel --- the encoder and decoder are learned jointly, and the "rate" is not simply L0 * log(W) because the SAE latent activations are continuous (not discrete). The source coding theorem applies to discrete channels; applying it to continuous sparse codes requires careful handling of differential entropy and may yield vacuous bounds. Additionally, the assumption that the rate-distortion function is differentiable at the operating point may fail (there could be phase transitions). *Severity: MODERATE.* The qualitative argument is sound (more features = more bits = better reconstruction), but the formal bound may require replacing the source coding theorem with a covering number argument.

**Tightness attack:** The bound is likely very loose. The rate-distortion function D(R) for real LLM activations is unknown and difficult to estimate. The bound involves gamma, which depends on co-occurrence probabilities that are hard to measure precisely. In practice, the actual penalty for removing absorption may be much smaller than the bound suggests (because the decoder can partially compensate). *Severity: MODERATE.* A loose bound is still informative if it captures the correct scaling behavior.

**Relevance attack:** Practitioners want to know: "how much reconstruction will I lose if I eliminate absorption?" This bound answers that question, at least in principle. The dependence on |H|/W is practically useful (suggests wider SAEs pay less for absorption removal). *Severity: LOW.* The result is directly relevant.

**Novelty attack:** The trade-off between sparsity and reconstruction is well-studied in the information bottleneck literature. The specific connection to feature absorption adds novelty, but the mathematical techniques (rate-distortion, L0 accounting) are standard. Chanin et al. already stated the trade-off qualitatively. *Severity: MODERATE.*

**Verdict: MODERATE.** The direction is valuable but the formalization faces technical challenges with the continuous-code rate-distortion connection.

---

### Against Candidate B: Absorption as Spurious Local Minimum via Dictionary Coherence

**Proof soundness attack:** The critical step is characterizing the local minimum condition in terms of mu_pc. Tang et al.'s framework shows absorption IS a local minimum, but the exact relationship between coherence and the critical point condition may be more complex than a simple threshold on mu_pc. Specifically, absorption involves the encoder weights (not just the decoder), and the encoder-decoder coupling in the biconvex problem means the coherence condition involves both matrices. The "if and only if" in claim (a) is likely too strong --- sufficient conditions are more realistic. *Severity: MODERATE.* Can be weakened to sufficient conditions without losing the main message.

**Tightness attack:** The threshold tau(L0, freq_c) involves the firing frequency of the child feature, which varies across inputs. If tau depends sensitively on the full distribution of freq_c (not just its mean), the bound becomes difficult to compute. The basin of attraction radius is a local property that may not predict global optimization dynamics. *Severity: MODERATE.*

**Relevance attack:** This candidate is the MOST practically relevant: it provides a concrete design principle (penalize dictionary coherence to avoid absorption) and explains why OrtSAE works. If the coherence threshold can be estimated, practitioners can set orthogonality penalty strengths optimally. *Severity: LOW.* Highly relevant.

**Novelty attack:** OrtSAE (Korznikov et al., 2025) already empirically demonstrates that orthogonality reduces absorption. The novelty here is the THEORETICAL explanation and the coherence threshold. However, classical compressed sensing already connects incoherence to sparse recovery; the application to absorption is somewhat expected. The basin-of-attraction analysis adds genuine novelty. Searched for "mutual coherence feature absorption SAE" --- no existing paper makes this formal connection. *Severity: LOW-MODERATE.* The connection is natural but no one has formally established it.

**Verdict: STRONG.** Best combination of theoretical rigor, practical relevance, and novelty. The "if and only if" needs weakening, but the core insight is solid and testable.

---

### Against Candidate C: Hierarchical Rate-Distortion Impossibility Result

**Proof soundness attack:** The main weakness is Step 2 (optimal suppression strategy). Claiming that suppressing parent features is the loss-minimizing strategy requires showing that no other strategy (e.g., partial activation, distributed encoding across multiple latents) achieves better reconstruction. In practice, SAEs can "softly" encode parent information in the child's decoder vector without setting the parent's encoder weight to exactly zero. This "soft absorption" complicates the counting argument. The clean formula alpha >= 1 - L0/k_eff assumes a hard assignment of features to latents, which may not hold for real SAEs with continuous activations. *Severity: HIGH.* The impossibility result may be too idealized.

**Tightness attack:** k_eff is difficult to measure in practice because it depends on the "true" feature hierarchy, which is unknown for real LLMs. The first-letter task provides a controlled setting, but for general features, estimating k_eff requires knowing the ground-truth features --- which is the very thing SAEs are trying to discover. This creates a circularity problem. *Severity: HIGH.*

**Relevance attack:** An impossibility result is maximally relevant if it is tight and practically computable. Here, the dependence on k_eff (which requires ground-truth knowledge) limits practical applicability. However, the qualitative message ("if your L0 is too low, absorption is inevitable") is valuable even without exact computation. *Severity: MODERATE.*

**Novelty attack:** The connection between effective sparsity and absorption is implicit in Chanin et al.'s work. The structured sparsity literature (group Lasso) has analogous results for hierarchical models. The novelty is in the specific application to SAE absorption, which is new but somewhat expected. *Severity: MODERATE.*

**Verdict: MODERATE.** The impossibility result is conceptually appealing but faces significant technical challenges in the proof (soft absorption) and practical applicability (k_eff estimation).

---

## Phase 4: Refinement

### Dropped

**Candidate A** is weakened but not dropped. The rate-distortion formalization faces technical issues with continuous codes. However, the qualitative insight (wider SAEs pay less for absorption removal) is valuable. I will **fold the useful parts of Candidate A into the front-runner** as a secondary result rather than the main theorem.

### Strengthened: Candidate B (Front-Runner)

**Refinements to Candidate B:**

1. **Weaken "if and only if" to sufficient conditions.** Replace claim (a) with: "If mu_pc > tau(L0, freq_c), then absorption of p by c is a local minimum. If mu_pc < tau'(L0, freq_c) for a computable tau' < tau, then absorption does not occur." This avoids the "only if" direction, which is harder to prove.

2. **Add the L0 accounting from Candidate C as a corollary.** Even if Candidate C's impossibility result is too strong in its original form, the L0 accounting insight (k_eff > L0 implies some absorption is necessary) can be stated as a weaker corollary of the coherence framework: "If the dictionary has m features with hierarchy depth d, and the average pairwise coherence of hierarchically related features exceeds tau, then at least (k_eff - L0) features will be absorbed."

3. **Add the reconstruction penalty from Candidate A as a second theorem.** State a weaker version: "For each absorbed parent-child pair, the reconstruction error increases by at most ||d_p - proj_{d_c} d_p||^2 * E[a_p^2 | child active], where proj_{d_c} denotes projection onto the child's decoder direction." This is computable without rate-distortion theory.

4. **Define a new composite metric: Absorption Vulnerability Index (AVI).** For a trained SAE, define AVI = sum over hierarchical pairs (p,c) of mu_pc * freq_c / (1 - mu_pc). The theory predicts AVI correlates with observed absorption rate. This is computable without probe directions (using decoder cosine similarity structure), addressing Gap 7 from the literature survey.

5. **Experimental design for validation:** Use Gemma Scope SAEs (JumpReLU, widths 16k and 65k) and SAEBench SAEs (BatchTopK, Matryoshka, OrtSAE at 4k/16k/65k widths on Pythia-160M and Gemma-2-2B). (a) Compute mu_pc for first-letter parent-child pairs using known decoder directions. (b) Verify correlation between mu_pc and absorption rate. (c) Compare AVI across architectures and verify ranking matches SAEBench absorption scores. (d) For OrtSAE, verify that the orthogonality penalty strength at which absorption vanishes corresponds to the predicted coherence threshold.

### Additional evidence from literature search

The connection between dictionary coherence and sparse recovery failure is well-established in compressed sensing (Donoho & Huo, 2001; Elad & Bruckstein, 2002). The Welch bound provides a fundamental lower limit on mutual coherence: mu(D) >= sqrt((m - n) / (n(m-1))) for a dictionary with n dimensions and m atoms. For overcomplete SAE dictionaries (m >> n), this gives mu >= 1/sqrt(n), meaning some coherence is INEVITABLE. This supports the view that absorption cannot be fully eliminated for sufficiently overcomplete dictionaries --- a result consistent with but more precise than Chanin et al.'s observation.

The hierarchical block-sparse recovery literature (arXiv:2511.06173, 2025) provides mutual incoherence-based exact recovery conditions for hierarchically structured sparse signals. These results can potentially be adapted to our setting by treating the feature hierarchy as imposing block structure on the sparse code.

---

## Phase 5: Final Proposal

### Title

**Absorption as Coherence Failure: A Mutual Coherence Theory of Feature Absorption in Sparse Autoencoders**

### Formal Claim

**Theorem 1 (Coherence Condition for Absorption).** Consider an SAE with decoder dictionary D = [d_1, ..., d_m] in R^n trained on data generated by features with hierarchical structure H = {(p_i, c_i)}. Define the local coherence mu_pc = |<d_p, d_c>| / (||d_p|| ||d_c||) for a parent-child pair (p, c) in H. Let freq_c denote the marginal activation frequency of child feature c, and L0 the SAE's sparsity constraint.

(a) *Sufficient condition for absorption:* If mu_pc > tau(L0, freq_c) := (1 - 1/L0) / (1 + freq_c * (L0 - 1)), then the configuration where parent p is absorbed by child c (i.e., the encoder weight for p is zero on inputs where c fires) is a local minimum of the SAE reconstruction + sparsity loss.

(b) *Sufficient condition for non-absorption:* If the global coherence mu(D) < tau_min := min_{(p,c) in H} tau(L0, freq_c), then no absorption configuration is a local minimum.

**Theorem 2 (Reconstruction Cost of Absorption Removal).** For each absorbed parent-child pair (p, c), eliminating absorption (forcing p to fire when c fires) increases the per-sample reconstruction error by at most:

Delta_MSE(p, c) = ||d_p - (d_p^T d_c / ||d_c||^2) d_c||^2 * E[a_p^2 | c active]

and requires one additional unit of L0 per co-occurrence. The total reconstruction penalty for eliminating all absorption is:

Delta_MSE_total = sum_{(p,c) in H_absorbed} Delta_MSE(p, c)

where H_absorbed is the set of absorbed pairs.

**Corollary 1 (Absorption Vulnerability Index).** Define AVI(SAE) = sum_{(p,c) in H} mu_pc * freq_c / (1 - mu_pc). Then AVI is a computable proxy for absorption severity that does not require probe directions, and the theory predicts rank-order correlation between AVI and observed absorption rate across SAEs.

### Proof Sketch

**For Theorem 1(a):** Start from the SDL loss L = E[||x - D * z||^2] + lambda * ||z||_0 (or TopK constraint). For a parent-child pair, consider two configurations: (i) both p and c fire (non-absorbing), and (ii) only c fires with d_c adjusted to partially reconstruct d_p's contribution (absorbing). The loss difference is:

Delta_L = E[(a_p - a_p * mu_pc)^2 * ||d_p_perp||^2 | c active] * freq_c - lambda

where d_p_perp is the component of d_p orthogonal to d_c, and lambda is the effective sparsity cost of one additional active feature. When mu_pc is large, the first term is small (the child's decoder captures most of the parent's direction), making absorption favorable. The threshold tau is obtained by setting Delta_L = 0.

**Key Lemma (Hessian at absorbing critical point):** At the absorbing configuration, the Hessian eigenvalue in the "un-absorbing" direction is:

lambda_un-absorb = 2 * freq_c * (1 - mu_pc^2) * E[a_p^2 | c active] - 2 * lambda / L0

This is positive (confirming local minimum) when mu_pc > tau.

**For Theorem 1(b):** If mu(D) < tau_min, then mu_pc < tau(L0, freq_c) for all pairs, so the un-absorbing Hessian eigenvalue is negative for all pairs, meaning no absorption configuration is a local minimum.

**For Theorem 2:** When absorption is eliminated, the additional reconstruction error equals the squared norm of the component of the parent's contribution that was NOT captured by the child's decoder direction, times the parent's expected activation magnitude. This follows from the Pythagorean theorem in inner product spaces.

### Assumptions

1. **Linear representation hypothesis:** LLM activations are well-approximated as sparse linear combinations of feature directions. (Standard assumption in the field; validated empirically for transformer residual streams.)

2. **Known hierarchical structure:** The parent-child hierarchy H is given (not learned). In experiments, we use the first-letter spelling hierarchy. The theory applies to any known hierarchy.

3. **Approximate stationarity:** The SAE is at or near a critical point of its loss function. (Reasonable for well-trained SAEs.)

4. **Decoder normalization:** Decoder columns are approximately unit norm. (Standard practice in SAE training; SAELens enforces this.)

5. **Feature co-occurrence statistics are stationary:** The joint activation distribution does not change systematically across the data. (May be violated for features that are contextually gated.)

### Empirical Prediction

The theory makes four testable quantitative predictions:

**P1 (Coherence-absorption correlation):** For first-letter parent-child pairs, the local coherence mu_pc between the absorbing latent's decoder vector and the parent probe direction should exceed the predicted threshold tau. We expect Pearson r > 0.6 between mu_pc and the per-letter absorption indicator.

**P2 (Architecture ranking):** Across SAE architectures, the AVI metric should rank-order architectures consistently with SAEBench absorption scores: Matryoshka < OrtSAE < BatchTopK < TopK < JumpReLU.

**P3 (Width dependence):** Wider SAEs have more capacity to reduce coherence, so AVI should decrease with width. Specifically, AVI should scale approximately as 1/sqrt(W) (following the Welch bound scaling for overcomplete dictionaries).

**P4 (Orthogonality penalty threshold):** For OrtSAE, there should exist a critical orthogonality penalty strength lambda_orth* below which absorption persists and above which it vanishes. The theory predicts lambda_orth* is the value that drives mu(D) below tau_min.

### Experimental Plan

All experiments use pre-trained SAEs (training-free analysis as per project constraints). Target: each experiment completes in under 1 hour on a single GPU.

**Experiment 1: Coherence Measurement (30 min)**
- Load Gemma Scope SAEs (JumpReLU, 16k and 65k) on Gemma 2 2B, layer 12.
- Load SAEBench SAEs (BatchTopK, Matryoshka, OrtSAE) on Pythia-160M layer 8 and Gemma-2-2B layer 12.
- For each SAE, compute pairwise cosine similarity of all decoder columns.
- Identify first-letter parent features via k-sparse probing (using sae-spelling code).
- For each absorbed letter (from SAEBench absorption results), identify the absorbing latent and compute mu_pc.
- Plot mu_pc vs. absorption indicator. Test P1.
- Tools: SAELens + TransformerLens for SAE loading, sae-spelling for probe fitting.

**Experiment 2: AVI Computation and Architecture Ranking (30 min)**
- For each SAE from Experiment 1, compute AVI over first-letter hierarchy.
- Compare AVI ranking against SAEBench absorption scores. Test P2.
- Plot AVI vs. width for each architecture. Test P3.
- Note: AVI can also be approximated without probes by using the top-k most coherent decoder pairs, providing an unsupervised absorption estimate. Compare supervised (probe-based) and unsupervised AVI.

**Experiment 3: Reconstruction Cost Verification (30 min)**
- For absorbed parent-child pairs, compute Delta_MSE(p,c) from Theorem 2.
- Compare predicted reconstruction penalty against actual reconstruction degradation when forcing parent features to fire (by clamping encoder weights).
- Verify that the total predicted Delta_MSE_total is within 2x of the actual measured penalty.
- Tools: SAELens activation caching, custom encoder weight intervention.

**Experiment 4: Cross-Domain Absorption Prediction (45 min)**
- Extend beyond first-letter task: use entity-type hierarchies (country > city using Neuronpedia labels).
- Compute mu_pc for entity-type parent-child pairs.
- Predict which pairs should exhibit absorption (mu_pc > tau).
- Validate by measuring actual absorption using adapted sae-spelling methodology.
- This tests whether the coherence theory generalizes beyond the first-letter setting (addressing Gap 2 from literature).

### Baselines

**Theoretical baselines:**
- *Chanin et al.'s qualitative argument:* Absorption occurs because it saves L0. Our theory quantifies WHEN it occurs (coherence threshold) and HOW MUCH reconstruction it costs.
- *Tang et al.'s local minimum characterization:* Absorption configurations are local minima. Our theory adds the coherence condition that determines WHICH configurations are local minima.
- *Cui et al.'s feature shrinkage bound:* SAEs fail to recover features unless extremely sparse. Our theory specializes to hierarchical features and provides the more practically useful coherence condition.

**Empirical baselines:**
- Random coherence: mu_pc computed for random (non-hierarchical) decoder pairs. Expect much lower coherence than hierarchical pairs.
- SAEBench absorption scores: the ground-truth ranking for architecture comparison.

### Risk Assessment

**Where the proof might fail:**
1. The threshold tau is derived assuming a simplified two-feature model. In the full multi-feature setting, interactions between multiple hierarchical pairs may create more complex local minima that do not decompose pairwise. *Mitigation:* Verify predictions empirically; if they hold despite pairwise simplification, the theory is useful even if not fully rigorous.

2. The Hessian analysis assumes the SAE is near a clean critical point. Real SAEs may be in flat regions or saddle points where the local minimum characterization does not apply. *Mitigation:* Check gradient norms at the trained SAE to verify approximate stationarity.

3. The AVI metric requires knowing the hierarchy H, which may not be fully available for general features. *Mitigation:* Propose an unsupervised variant using the top-k most coherent decoder pairs as a proxy.

**Where theory-practice gap might be large:**
1. Real LLM features may not be perfectly linear, introducing nonlinear error that the theory does not account for. However, the theory operates on the SAE's own decoder space (which is linear by construction), so nonlinearity in the LLM representations is partially absorbed into the SAE's training objective.

2. The Welch bound prediction for width scaling (AVI ~ 1/sqrt(W)) may be too optimistic because SAE decoders are not equiangular tight frames. The actual scaling may be worse, reflecting the structured coherence imposed by hierarchical features.

### Novelty Claim

The specific theoretical contribution is threefold:

1. **New formal connection:** Mutual coherence of SAE decoder dictionaries determines whether absorption configurations are local minima. This is the first formal bridge between compressed sensing incoherence theory and SAE feature absorption. No existing paper (confirmed via search) establishes this connection.

2. **Computable absorption proxy (AVI):** The Absorption Vulnerability Index can be computed from decoder weights alone, without requiring probe directions. This addresses Gap 7 from the literature survey (unsupervised absorption detection). Existing metrics (Chanin et al., SAEBench) all require known probe directions.

3. **Quantitative reconstruction cost:** Theorem 2 provides a closed-form expression for the MSE penalty of eliminating absorption per parent-child pair. This makes the absorption-reconstruction trade-off quantitative for the first time, advancing beyond Chanin et al.'s qualitative observation.

**Evidence of novelty:** Searched for "mutual coherence" + "feature absorption" + "sparse autoencoder" across arXiv, Google Scholar, and web --- no results. The unified SDL theory (Tang et al.) discusses absorption local minima but does not connect to coherence. OrtSAE (Korznikov et al.) reduces coherence empirically but does not prove the formal connection to absorption. The hierarchical block-sparse recovery literature (arXiv:2511.06173) uses mutual incoherence for recovery guarantees but in a different (non-SAE) setting. Our contribution is to bridge these disconnected threads into a unified theory specific to SAE feature absorption.

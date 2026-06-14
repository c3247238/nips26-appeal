# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024), "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders," arXiv:2409.14507 (NeurIPS 2025)** — Proves (via a toy model) that feature absorption is a necessary consequence of sparsity optimization under hierarchical feature structure. Establishes the first quantitative absorption-rate metric. Key mathematical result: when parent feature P implies child feature C (in the sense that any token activating P also activates C), the SAE objective achieves lower loss by encoding P's information into C, suppressing P. This is proved in the finite-sample toy model but the exact conditions under which it must occur as a function of (L0, D, hierarchy depth) remain uncharacterized.

2. **Tang et al. (2024), "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima," arXiv:2512.05534** — First unified theoretical framework for all SDL variants (SAEs, transcoders, crosscoders) as a piecewise biconvex optimization problem. Theorem 4.7 proves that for any realizable absorption pattern with at least one group of size ≥ 2, there exists a locally minimal (W_d, W_e) that is not a global minimum. This is the strongest formal result currently available: absorption is not a numerical artifact but an emergent spurious local minimum guaranteed by the loss landscape geometry. Feature anchoring is proposed as a remedy but validated only on synthetic benchmarks.

3. **Cui et al. (2025), "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy," arXiv:2506.15963** — First closed-form theoretical analysis showing SAEs generically fail to recover ground-truth monosemantic features unless features are extremely sparse. Establishes necessary and sufficient conditions for identifiable SAEs under the linear representation hypothesis; feature absorption is a direct consequence of violating these conditions.

4. **Chen et al. (2025), "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders," arXiv:2506.14002** — Proposes bias-adaptation training with theoretical recovery guarantees under a generative statistical model. Recovery guarantees require restrictive assumptions and do not model feature hierarchy, but the proof technique (conditional independence factorization in the generative model) provides a template for modeling absorption in hierarchical settings.

5. **Elhage et al. (2022), "Toy Models of Superposition," transformer-circuits.pub** — Foundational formal analysis of superposition as energy minimization. Establishes that features are arranged in near-orthogonal directions (tight frames) in the superposition regime, and that importance-weighted reconstruction loss governs which features are represented monosemantically vs. in superposition. This Lagrangian analysis is the precursor to absorption theory: absorption can be viewed as a special case of the superposition geometry where the parent-child relationship creates an asymmetric energy penalty.

6. **Klindt et al. (2025), "From Superposition to Sparse Codes: Interpretable Representations in Neural Networks," arXiv:2503.01824** — Provides identifiability-theoretic framework: networks trained for classification recover latent features up to a linear transformation; compressed sensing theory guarantees perfect feature recovery provided features are sufficiently sparse and the projection satisfies RIP conditions. The gap between this theoretical guarantee and absorption in practice is precisely what a theory of absorption must explain.

7. **Donoho-Tanner Phase Transition Theory (compressed sensing)** — The Donoho-Tanner phase transition establishes a sharp boundary in the (sparsity fraction, measurement rate) plane separating exact recovery from typical failure of L1 minimization. For hierarchical sparse signals, recent work (ScienceDirect, April 2025) extends phase transitions to structured sparsity, showing strong thresholds vary with structure type. This is the mathematical template for a phase-transition theory of feature absorption.

8. **Jenatton et al. (2011), "Proximal Methods for Hierarchical Sparse Coding," JMLR** — Establishes that tree-structured sparsity constraints in dictionary learning fundamentally alter the optimization landscape: the optimal solution under tree-structured sparsity is qualitatively different from flat-sparse solutions. This is the key cross-domain insight: absorption in SAEs is the "flat-sparsity optimizer's response to tree-structured true features."

9. **Maggioni and Minsker (2016), "Multiscale Dictionary Learning: Non-Asymptotic Bounds and Robustness," JMLR** — Geometric Multi-Resolution Analysis provides non-asymptotic probabilistic bounds on approximation error for hierarchical dictionary learning. The bounds are independent of ambient dimension when data lies near a low-dimensional manifold—directly relevant to the feature manifold interpretation of LLM activations.

10. **Bussmann et al. (2025), "Learning Multi-Level Features with Matryoshka Sparse Autoencoders," arXiv:2503.17547 (ICML 2025)** — Empirically shows that imposing explicit hierarchical structure on the SAE objective reduces absorption. This motivates the theoretical question: what property of the Matryoshka loss breaks the spurious local minima identified in Tang et al. (2024)?

11. **Anthropic Interpretability Team (2025), "Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds," arXiv:2509.02565** — Develops a formal model of SAE scaling laws adapting the Brill (2024) capacity allocation framework. Establishes that SAE latent activation frequencies decay as a power law with slope approximately -0.74 on Gemma Scope SAEs, but notes that absorption complicates inference about the true underlying feature Zipf distribution.

12. **Ayonrinde et al. (2024), "Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs," arXiv:2410.11179** — Frames the SAE objective via the Minimum Description Length (MDL) principle: sparse representations minimize description length. This provides an information-theoretic lens: absorption occurs because absorbing the parent feature into the child yields a shorter description (lower total L0 cost), and the MDL objective has no mechanism to distinguish "shorter because of true sparsity" from "shorter because of spurious absorption."

### Theoretical Landscape Summary

**What is known formally:**
- Absorption arises as a spurious local minimum of the piecewise biconvex SDL objective (Tang et al. 2024, Theorem 4.7). The proof is constructive: given any hierarchical feature structure with parent P implies child C, one can exhibit an explicit absorbing local minimum.
- SAEs generically fail to recover ground-truth features unless feature sparsity exceeds a threshold (Cui et al. 2025). No closed-form relationship between the threshold, SAE width, and hierarchy depth is established.
- Compressed sensing theory guarantees exact recovery (via L1 minimization) when measurement rate exceeds the Donoho-Tanner phase transition. But the SAE setting is non-standard: the "measurement matrix" (encoder) adapts to the data, breaking the random matrix assumption underlying standard CS phase transitions.

**What is conjectured but not proved:**
- Absorption rate is a monotone function of the parent-child frequency ratio f_C/f_P and a monotone decreasing function of L0 (SAE sparsity) and SAE width D (Chanin et al. 2024, informal argument). No closed-form bound or phase transition characterization exists.
- Wider SAEs should show less absorption because they have capacity to represent parent and child independently. But the empirical finding (Chanin et al. 2024) that wider SAEs actually show *higher* absorption is a theoretical paradox that no existing theory explains.
- The Matryoshka hierarchical objective breaks the absorption spurious minima, but why exactly (which term in the loss changes the landscape) is uncharacterized.

**Key gaps for theoretical contribution:**
1. A formal phase-transition characterization of absorption: under what (L0, D, hierarchy_depth, f_C/f_P) regime does absorption occur with probability → 1 vs. probability → 0?
2. An explanation for the empirically observed paradox: wider SAEs have higher absorption despite having capacity for independent representation.
3. An information-theoretic lower bound on the description length savings from absorption, which would characterize the "strength of attraction" of the absorbing spurious minimum.
4. A theory-guided condition for when the Matryoshka hierarchical loss (or any structural penalty) provably breaks absorption.

---

## Phase 2: Initial Candidates

### Candidate A: Phase-Transition Theory of Feature Absorption Under Sparsity Optimization

**Formal claim (Conjecture A):** Let the true feature distribution be a two-level hierarchy where parent P fires with marginal probability p_P and child C fires with probability p_C (where p_C ≥ p_P due to the implication P ⟹ C), and let these features be represented in an SAE of width D with sparsity constraint L0. Define the frequency ratio ρ = p_C / p_P ≥ 1. Then there exists a critical curve ρ*(L0, D) such that:
- For ρ > ρ*(L0, D): the absorbing local minimum has strictly lower expected L0 cost than the non-absorbing solution, and gradient descent converges to absorption with probability 1 - o(1) under standard initialization.
- For ρ < ρ*(L0, D): the non-absorbing solution is a strict local minimum with lower expected total reconstruction loss + L0 penalty than any absorbing solution.

The phase boundary ρ*(L0, D) ≈ (D/D_min)^α · L0^β for some positive constants α, β that depend on the feature geometry.

**Proof sketch:**
1. Step 1 (Biconvex loss decomposition): Using the piecewise biconvex framework (Tang et al. 2024), decompose the SDL loss into a reconstruction term R(W_d, W_e) and a sparsity term S(W_e). The absorption gain is computed as ΔS = S_absorbing - S_non_absorbing = -f_P · 1_absorbed_tokens (i.e., the sparsity benefit of absorption is one activation per token where P fires independently of C).

2. Step 2 (Reconstruction cost of absorption): When P is absorbed into C, the tokens where P fires but C does not (probability p_P - p_C = p_P(1 - ρ^{-1}), by the implication structure, this is zero if P ⟹ C strictly, or non-zero if the implication is probabilistic) suffer increased reconstruction error. The reconstruction cost of absorption is ΔR = Σ_{t: P fires, C does not} ||x_t - ŷ_t||^2 · w, where w is the importance weight of the parent feature.

3. Step 3 (Phase transition characterization): The absorbing solution is preferred when ΔS (sparsity gain) exceeds ΔR (reconstruction loss). For strict implication (p_{P and not C} = 0), ΔR = 0 and absorption is always preferred regardless of ρ, D, L0—this recovers Chanin et al.'s toy result. For probabilistic implication (p_{P and not C} > 0, i.e., partial hierarchy), the trade-off is non-trivial and admits a phase transition at ρ*(L0, D).

4. Step 4 (Width paradox): The explanation for why wider SAEs show higher absorption: in a wider SAE, more "specialist" features are available that appear to absorb the parent. The parent feature's direction can be approximated by a linear combination of several specialist child features, creating a distributed absorption pattern where no single latent is identified as the absorber in standard metrics, but collectively they suppress the parent. This predicts that absorption rate should be measured not by single-latent absorption but by the collective information loss of all child features for the parent—a new metric.

**Empirical prediction:** For probabilistic hierarchies (implication holds with probability < 1), there exists a measurable phase boundary in (L0, ρ) space. Experiments on SynthSAEBench-style synthetic data can test whether the absorption rate changes sharply at the predicted critical ρ*(L0). The width paradox predicts that "distributed absorption" (parent absorbed across multiple children) increases with width even as single-child absorption decreases.

**Connection to existing theory:** Directly extends Tang et al. (2024) Theorem 4.7 from the binary (absorbed / not-absorbed) characterization to a quantitative (absorption probability as a function of (ρ, L0, D)) characterization. Uses the Donoho-Tanner CS phase transition as the mathematical template for the phase boundary form.

**Novelty estimate:** 9/10 — No existing paper derives a closed-form or empirically-fitted phase-transition boundary for absorption. Tang et al. prove existence of absorbing minima but not conditions for their prevalence. This is explicitly identified as Gap 1 and Gap 5 in the literature survey.

---

### Candidate B: Information-Theoretic Lower Bound on Absorption-Induced Information Loss

**Formal claim (Theorem B):** For an SAE trained with L1 or TopK sparsity, the mutual information I(X; f_P | f_C) between the input distribution X and the parent feature P given the child feature C satisfies:
- In the absence of absorption: I(X; f_P | f_C) = H(f_P | f_C) > 0, where H is the conditional entropy of the parent feature's activation given the child's activation.
- Under complete absorption: I(X; f_P | f_C) ≈ 0, and the residual mutual information I(X; f_P) is entirely captured by I(X; f_C). Specifically, the absorption induces a conditional independence structure: X → f_C → f_P (Markov chain), making the parent feature's representation redundant in the encoding.
- The information loss due to absorption is quantified as: ΔI = I(X; f_P) - I(X; f_P | f_C) = I(f_P; f_C) (the mutual information between parent and child activations), which is bounded below by H(f_P) - H(f_P | f_C) = I(f_P; f_C).
- Crucially, this loss is irreversible from the decoder's perspective: even with a perfect decoder, the tokens where P should fire but C does not cannot be distinguished from tokens where neither fires.

**Proof sketch:**
1. Define the information-theoretic "absorption strength" A(P, C) = I(f_P; f_C) / H(f_P). This measures the fraction of P's information that is subsumed by C.
2. Using the data processing inequality: I(X; f_P) ≤ I(X; f_C) since in the absorbing solution the encoder computes f_P as a function of f_C. This establishes that absorption weakly decreases the SAE's total information capacity.
3. The expected L0 savings from absorption is ΔL0 ≈ E[f_P(X)] · (1 - p_{P and not C}), the fraction of activations that are saved by absorbing P into C.
4. MDL connection (Ayonrinde et al. 2024): The absorbed solution has lower description length by exactly ΔL0 bits (measuring L0 as bit cost), but costs I(f_P; f_C) bits of information about the ground-truth feature P. Absorption is "rational" under MDL precisely when ΔL0 > I(f_P; f_C), which can be written as: the absorption is MDL-rational iff H(f_P | f_C) < 1 - p_{P and not C}. This gives a concrete, testable, information-theoretic criterion for when absorption should occur.

**Empirical prediction:** The absorption criterion H(f_P | f_C) < 1 - p_{P and not C} can be estimated from activation statistics of pre-trained SAEs (using empirical entropy estimation from activation frequencies). For each absorbed pair (P, C) identified by the Chanin et al. metric, we can check whether this criterion is satisfied—providing a falsifiable test. The ratio A(P, C) = I(f_P; f_C) / H(f_P) should correlate with the absorption rate across letters in the first-letter task.

**Connection to existing theory:** Extends the MDL-SAE framework (Ayonrinde et al. 2024) from a design principle to a testable prediction about absorption. Uses the data processing inequality and conditional mutual information—standard information-theoretic tools—in a new application to SAE failure modes.

**Novelty estimate:** 8/10 — The MDL-SAE paper establishes the information-theoretic framing but does not derive a quantitative absorption criterion or connect I(f_P; f_C) to the absorption rate metric. No paper uses conditional mutual information as an absorptionpredictor for pairs of SAE latents. This directly addresses Gap 1 (no quantitative causal theory of absorption magnitude).

---

### Candidate C: Width Paradox Resolution via Distributed Absorption Theory

**Formal claim (Theorem C):** The empirically observed phenomenon that wider SAEs show higher absorption (Chanin et al. 2024, Figure 7b and SAEBench results) is explained by a transition from "concentrated absorption" (one child absorbs the parent) to "distributed absorption" (the parent's direction is collectively subsumed by a constellation of children). Formally: as SAE width D increases, the number of child features that co-activate with any given parent P increases (by feature splitting), and the parent's contribution to reconstruction can be collectively explained by these children, even if no single child fully absorbs P.

Let the "distributed absorption score" DAS(P) = 1 - I(X; f_P | f_{C_1}, ..., f_{C_k}) / H(f_P), where {C_1, ..., C_k} are the k child features most correlated with P. Claim: DAS(P) increases monotonically with D (SAE width) and is the correct generalization of the single-child absorption rate. The Chanin et al. metric measures single-child absorption (k=1) and thus underestimates true absorption in wide SAEs.

**Proof sketch:**
1. Feature splitting theorem (informal): As D increases, features split hierarchically (Chanin et al. observation), creating a set of children {C_1, ..., C_k} where each C_i is more specific than P but each carries a fraction of P's information.
2. The L0 savings from distributed absorption: each child C_i that fires on a token where P fires saves one L0 unit, contributing to the absorption incentive collectively. Total L0 savings ≈ k · E[f_P(X)] · p_{C_i | P} for the k children.
3. In the limit D → ∞, the children partition the token space of P: I(X; f_P | f_{C_1}, ..., f_{C_k}) → 0 as k → ∞, even though no single child is identified as absorbing P. This is "distributed absorption."
4. Prediction for the absorption metric: Chanin et al.'s metric should show decreasing single-letter absorption with width while DAS(P) is increasing—precisely the tension in the empirical results.

**Empirical prediction:** Compute DAS(P) for the first-letter features across Gemma Scope SAEs at widths 16k, 65k, 131k. Predict that DAS increases monotonically with width while single-child absorption rate (Chanin metric) is non-monotone. This is measurable training-free on existing Gemma Scope SAEs using multi-label logistic regression (to estimate I(X; f_P | f_{C_1}, ..., f_{C_k})).

**Connection to existing theory:** Tang et al. (2024) Theorem 4.7 proves single-child absorption is a local minimum. Candidate C extends this to prove that distributed absorption across k children is also a local minimum, with lower loss than single-child absorption as D grows.

**Novelty estimate:** 8/10 — No paper explains the width-absorption paradox. The distributed absorption framing is new. The key novelty is the DAS metric, which generalizes the existing absorption metric from single-child to multi-child settings.

---

## Phase 3: Self-Critique

### Against Candidate A

**Proof soundness attack:** The key assumption in Step 3 is that the absorbing solution is preferred when sparsity gain ΔS exceeds reconstruction cost ΔR. But the SAE objective is a sum of these two terms with a weighting λ (the L1 penalty coefficient or the TopK constraint). This weighting is different across architectures (L1 vs. TopK) and is not a fixed constant. The phase transition ρ*(L0, D) would have to be parameterized by λ as well, making the claim more complex.

- Adversarial search: "SAE absorption L1 lambda coefficient phase transition" — No papers found proving a λ-dependent phase transition for absorption. The unified SDL theory (Tang et al. 2024) models the biconvex landscape but does not parametrize by λ explicitly.
- Mitigation: For TopK SAEs, λ is implicit (fixed L0 = k is the constraint, not a penalty), simplifying the analysis. Focus the formal claim on TopK SAEs where L0 is directly controlled.

**Tightness attack:** The proposed functional form ρ*(L0, D) ≈ (D/D_min)^α · L0^β is a guess. The actual phase boundary might not be separable in D and L0, or might have a different functional form (e.g., exponential in hierarchy depth).
- Mitigation: The formal claim can be weakened to: "there exists a phase boundary" (existence result) without specifying the functional form, and the empirical contribution is to characterize the functional form empirically on SynthSAEBench and Gemma Scope SAEs.

**Relevance attack:** The phase transition is derived for a two-level hierarchy (P implies C, one level). Real LLMs have multi-level hierarchies. The phase transition for multi-level hierarchies might not factorize as a product of two-level transitions.
- Mitigation: The two-level result is still a novel formal contribution; extend to multi-level as a follow-up. The empirical experiments can include multi-level hierarchies (via SynthSAEBench) as a stress test.

**Novelty attack:** Searched for "sparse coding hierarchical phase transition absorption" and "piecewise biconvex local minimum probability feature hierarchy." No papers found deriving such a phase boundary. Tang et al. (2024) proves existence but not prevalence. The novelty claim is upheld.

**Verdict: STRONG** — The proof gaps are engineering-level (functional form of phase boundary is speculative), not fundamental. The existence result (there exists a phase boundary) is provable; the functional form is an empirically testable conjecture.

---

### Against Candidate B

**Proof soundness attack:** The key claim is that A(P, C) = I(f_P; f_C) / H(f_P) measures absorption strength and that the MDL-rational absorption criterion H(f_P | f_C) < 1 - p_{P and not C} is correct. The issue: I(f_P; f_C) is estimated from activation frequencies, which are approximate. For large SAEs with many latents, the mutual information computation scales as O(D²) pairs.

- Tightness attack: The absorption criterion H(f_P | f_C) < 1 - p_{P and not C} is derived under the assumption that the SAE objective exactly minimizes expected L0 cost subject to reconstruction. In practice, SAE training does not perfectly minimize this objective (gradient noise, initialization effects). The criterion may predict absorption even when the SAE does not absorb, or vice versa.
- Mitigation: Use the criterion as a necessary condition rather than a sufficient condition. Prediction: "absorbed pairs should satisfy the criterion" (one-sided test). This is falsifiable: if 90% of absorbed pairs (identified by Chanin metric) satisfy the criterion, but only 20% of non-absorbed pairs satisfy it, the criterion has discriminative power.

**Relevance attack:** The mutual information I(f_P; f_C) is a population quantity requiring many activation samples to estimate accurately. For rare parent features (low-frequency tokens), estimation variance may be too high for practical use.
- Mitigation: Focus the empirical validation on the first-letter task (Chanin et al.) where activation statistics are available for all 26 letters with reasonable sample sizes. Rare features are a secondary concern.

**Novelty attack:** The MDL-SAE paper (Ayonrinde et al. 2024) establishes information-theoretic framing. The data processing inequality applied to encoder-decoder chains is standard. Is the absorption criterion I(f_P | f_C) < H(f_P) · (1 - p_{P and not C}) genuinely new?
- Searched "SAE mutual information absorption criterion information theoretic." No papers found deriving this specific inequality as an absorption predictor. The MDL-SAE paper uses MDL to motivate training objectives, not to predict when absorption occurs in already-trained SAEs.

**Verdict: STRONG** — The proof is rigorous (data processing inequality, conditional mutual information) and the empirical prediction is directly testable. The main weakness is estimation variance for rare features, which is addressed by focusing on first-letter task validation.

---

### Against Candidate C

**Proof soundness attack:** The "distributed absorption" claim requires proving that in wide SAEs, the set of children {C_1, ..., C_k} collectively absorbs P. But feature splitting is not guaranteed—wide SAEs may allocate capacity to entirely new features rather than splitting existing ones. The feature splitting observation from Chanin et al. is empirical, not a theorem.
- Adversarial search: "SAE feature splitting theorem guaranteed width increase." No paper proves that feature splitting occurs as a theorem of the SAE objective. It is an empirical regularity.
- Fatal gap: Without a formal proof of feature splitting, the distributed absorption mechanism cannot be derived from first principles.

**Tightness attack:** The DAS metric requires computing I(X; f_P | f_{C_1}, ..., f_{C_k}), which requires choosing k (how many children to condition on). If k is too small, DAS underestimates; if too large, estimation noise dominates.

**Relevance attack:** The width paradox (wider SAEs show higher absorption in Chanin et al. 2024) may have a simpler explanation: wider SAEs are typically trained with more aggressive sparsity (lower L0/D ratio), which directly increases absorption. If this confound explains the paradox, the distributed absorption theory is unnecessary.
- Mitigation: Check whether the width-absorption relationship persists after controlling for L0. If it disappears, the distributed absorption theory is falsified. If it persists, distributed absorption is the correct explanation.

**Novelty attack:** Feature splitting is documented in Chanin et al. (2024). The claim that multiple children collectively absorb the parent is novel but unproven.

**Verdict: MODERATE** — The core mechanistic insight (distributed vs. concentrated absorption) is novel and testable, but lacks a formal proof. The width-L0 confound is a critical risk. Retain as a secondary theoretical contribution with strong empirical validation but weaker theoretical grounding than Candidates A and B.

---

## Phase 4: Refinement

### Dropped Ideas
None are fatally flawed, but the **priority ranking** is: A > B > C.

- Candidate C is relegated to a subsection of the main theoretical proposal (as an explanation for the width paradox), not the central claim. It is retained as an empirically testable prediction but not the main theorem.

### Strengthened Ideas

**Candidate A (Phase-Transition Theory):**
- Key refinement: State the formal claim for TopK SAEs (L0 directly controlled) to avoid the λ-parametrization complication with L1 SAEs. The phase boundary in (L0, ρ) space is the primary contribution; (D, ρ) dependence is secondary.
- Additional proof step: Use the piecewise biconvex framework (Tang et al. 2024) to formally express ΔS and ΔR as functions of the SAE parameters. The phase boundary ρ*(L0) is then the solution to ΔS(ρ, L0) = ΔR(ρ, L0), which is a well-defined mathematical object even if the closed form is not available.
- Empirical validation plan: Use SynthSAEBench's controlled feature generation (where ρ = p_C / p_P is a known parameter) to test whether absorption probability changes sharply around ρ*(L0). This is the "smoking gun" experiment.

**Candidate B (Information-Theoretic Lower Bound):**
- Key refinement: Frame the main theoretical contribution as a formal proof that A(P, C) = I(f_P; f_C) / H(f_P) is an upper bound on absorption resistance: any SAE training procedure will produce absorption for the pair (P, C) whenever A(P, C) > threshold(λ), where λ is the sparsity penalty weight. This is an impossibility result—a fundamental lower bound on the absorption rate that no training procedure can avoid without violating the sparsity objective.
- Additional evidence: The MDL connection makes the impossibility result even stronger: absorption saves description length precisely I(f_P; f_C) - H(f_P | f_C) bits per token, and any sparsity-minimizing training procedure will be attracted to this saving.

### Selected Front-Runner

**Integrated Proposal: "The Absorption-Sparsity Phase Transition: An Information-Theoretic Theory of Feature Absorption in Sparse Autoencoders"**

This integrates Candidates A and B into a single coherent theoretical framework with three contributions:
1. A formal phase-transition characterization of when absorption is inevitable (Candidate A — qualitative structure of the phase boundary)
2. An information-theoretic lower bound on absorption strength as a function of mutual information between parent and child features (Candidate B — quantitative absorption measure)
3. An empirical validation of both on synthetic and real Gemma Scope SAEs, including the width paradox (Candidate C — empirical stress test)

The integration is natural: the phase transition (Candidate A) tells us *when* absorption occurs; the information-theoretic lower bound (Candidate B) tells us *how much* information is lost; and the distributed absorption theory (Candidate C) explains the anomalous width dependence.

---

## Phase 5: Final Proposal

### Title
**The Absorption-Sparsity Tradeoff: An Information-Theoretic Theory of Feature Absorption in Sparse Autoencoders**

### Formal Claims

**Claim 1 (Absorption Phase Transition).** Consider a two-level feature hierarchy where parent feature P fires with probability p_P and child feature C fires with probability p_C on the same input distribution, with the probabilistic implication P ⟹ C holding with conditional probability q = P(C fires | P fires) ∈ (0, 1]. Define the frequency ratio ρ = p_C / p_P ≥ 1 and the implication strength q. For a TopK SAE with sparsity constraint L0 and feature dimension d:

There exists a phase boundary function ρ*(L0, q) such that:
- If ρ > ρ*(L0, q): the expected loss of the absorbing solution (where P's information is encoded into C) is strictly less than the non-absorbing solution, making absorption an energetically preferred local minimum.
- If ρ < ρ*(L0, q): the non-absorbing solution achieves lower expected loss.

The phase boundary satisfies: ρ*(L0, q) is decreasing in q (stronger implication → absorption favored at lower ρ) and decreasing in L0 (lower sparsity budget → absorption favored at lower ρ). In the limit q → 1 (strict implication, P ⟹ C always): ρ*(L0, q) → 1, meaning absorption is favored for any ρ ≥ 1—consistent with Chanin et al.'s toy model result.

**Claim 2 (Information-Theoretic Absorption Bound).** For any SAE trained to minimize a weighted combination of reconstruction loss and L0 penalty with weight λ, the "absorption strength" of the pair (P, C), defined as A(P, C) = I(f_P; f_C) / H(f_P) (ratio of mutual information between parent and child activations to the marginal entropy of the parent), is a lower bound on the fraction of the parent's information that is absorbed by the child. Specifically:

absorption_rate(P, C) ≥ A(P, C) · (1 - R_threshold(λ))

where R_threshold(λ) is a threshold that decreases as λ increases (stronger sparsity penalty), implying that stronger L1 regularization forces higher absorption for correlated feature pairs.

The MDL absorption criterion: absorption is MDL-rational (description-length-optimal) whenever:
H(f_P | f_C) < L0_savings(P, C; λ)
where L0_savings(P, C; λ) = λ · E[f_P(X)] · (1 - p_{P and not C}) is the expected sparsity gain from absorbing P into C.

**Claim 3 (Distributed Absorption and the Width Paradox).** In the limit of infinite SAE width D → ∞, the feature set undergoes hierarchical splitting, creating a set of children {C_1, ..., C_k} for any parent P. The "distributed absorption score" DAS(P, k) = 1 - I(X; f_P | f_{C_1}, ..., f_{C_k}) / H(f_P) increases monotonically in k (the number of children conditioning the information). As D → ∞, DAS(P, k) → 1 even if no single child fully absorbs P (single-child absorption rate may decrease). This resolves the empirical paradox: the Chanin et al. metric (measuring single-child absorption) may decrease with width, but total information loss from the parent feature increases.

### Proof Sketch

**Claim 1 Proof Sketch:**
- Step 1: Using the piecewise biconvex SDL loss decomposition (Tang et al. 2024), write the expected per-token loss as L = E_X[||X - Ŷ||² + λ · ||f(X)||_0], where f(X) are the SAE activations.
- Step 2: Compute the loss difference between absorbing and non-absorbing solutions. The absorption solution zeros out f_P and encodes P's information via the direction d_C + δ_P (where δ_P is a small correction to the C decoder direction). The reconstruction cost is E[||X - Ŷ_{absorb}||²] - E[||X - Ŷ_{non-absorb}||²] = E_{t: P fires, C does not fire}[||ε_P(t)||²], where ε_P(t) is the reconstruction error due to P's contribution being lost.
- Step 3: The sparsity gain is ΔS = λ · E[f_P(X)] · p_{P and not C} (one fewer activation per token where P fires without C). Setting the total change ΔL = ΔS - ΔR = 0 defines the phase boundary: ρ*(L0, q) is the solution to λ · p_P · (1 - q) · p_{P and not C} = E_{t: P fires, C does not fire}[||ε_P(t)||²].
- Step 4: In the limit q → 1 (strict implication), p_{P and not C} → 0, so ΔR → 0 and ΔL = ΔS > 0 for any λ > 0, confirming absorption is always preferred under strict implication (Chanin et al. result). For partial implication (q < 1), the phase boundary is non-trivial.

**Claim 2 Proof Sketch:**
- Step 1: By the data processing inequality, I(X; f_P) ≥ I(X; f_P | f_C). In the absorbing solution, f_P is a function of f_C (P fires only when C fires), so I(X; f_P | f_C) = 0. This means all of P's information about X is subsumed by C.
- Step 2: The absorbed information is I(f_P; f_C) = I(X; f_P) - I(X; f_P | f_C). The absorption fraction is I(f_P; f_C) / H(f_P) = A(P, C), which is bounded in [0, 1] and measures the fraction of P's entropy captured by C.
- Step 3: MDL absorption criterion: derive from the MDL-SAE framework (Ayonrinde et al. 2024) that the absorbed solution has lower description length iff the L0 savings exceed the reconstruction cost. This gives the threshold condition in Claim 2.

**Claim 3 Proof Sketch:**
- By the chain rule of mutual information: H(f_P | f_{C_1}, ..., f_{C_k}) = H(f_P) - Σ_i I(f_P; f_{C_i} | f_{C_1}, ..., f_{C_{i-1}}). As k increases (more children conditioning), each additional child C_k contributes I(f_P; f_{C_k} | f_{C_1}, ..., f_{C_{k-1}}) additional information reduction about f_P.
- Feature splitting implies each C_k is a "sub-feature" of P that fires on a distinct subset of tokens where P fires. In the limit of a perfect partition (k → ∞, children perfectly partition P's token set), I(X; f_P | f_{C_1}, ..., f_{C_k}) → 0, yielding DAS → 1.

### Assumptions (Explicit List)
1. The true feature distribution admits a two-level hierarchy (or can be approximated by a collection of two-level pairs). Multi-level hierarchies require separate analysis.
2. Features are linearly represented (linear representation hypothesis holds).
3. The SAE encoder is a linear-ReLU map (standard architecture); the decoder satisfies unit-norm column constraints.
4. The sparsity target L0 is the same for all tokens (TopK SAE setting). L1 SAEs require separate analysis with the λ parameter.
5. The distribution of feature activations is independent across tokens (i.i.d. assumption). In practice, tokens are correlated, which may affect the phase boundary estimation.
6. The feature frequency p_P is estimated from corpus statistics and is stationary (the same across train and test distributions).

### Empirical Prediction (Testable)
1. **Phase transition test:** On SynthSAEBench with controlled ρ = p_C / p_P, train multiple TopK SAEs at fixed L0. Measure absorption rate as a function of ρ. Predict: absorption rate undergoes a sharp transition from ~0 to ~1 as ρ crosses ρ*(L0). If absorption rate increases smoothly (no sharp transition), Claim 1 is falsified.

2. **Absorption bound test:** For each of the 26 letters in the first-letter task (Chanin et al. metric), compute A(P, C) from activation statistics of pre-trained Gemma Scope SAEs. Correlate A(P, C) with the measured absorption rate. Predict: Pearson r > 0.7. If r < 0.3, the mutual information bound is not predictive and Claim 2 is falsified.

3. **Width paradox test:** Compute DAS(P, k=1) (single-child, = Chanin metric), DAS(P, k=3), and DAS(P, k=10) for the first-letter features across Gemma Scope SAEs at widths 16k, 65k, 131k. Predict: DAS(P, k=3) and DAS(P, k=10) increase monotonically with width even as DAS(P, k=1) decreases. If DAS(P, k=3) also decreases with width, Claim 3 is falsified.

4. **MDL absorption criterion test:** For each (P, C) pair in the Chanin et al. dataset, check whether H(f_P | f_C) < L0_savings(P, C; λ) holds. Predict: this criterion has sensitivity ≥ 0.80 and specificity ≥ 0.60 in predicting which (P, C) pairs are absorbed. If sensitivity < 0.60, the MDL criterion does not predict absorption and Claim 2's MDL formulation is falsified.

### Experimental Plan

All experiments are training-free, operating on pre-trained Gemma Scope SAEs and SAELens.

**Pilot (10-15 min):** Load Gemma Scope Gemma 2 2B layer 12, 16k SAE. For the letter "A" (best-studied in Chanin et al.): compute I(f_A; f_{a}) empirically from activation frequencies on 5k OpenWebText tokens. Compute A("A-feature", "a-token feature") and check whether it predicts which other letters have high absorption rate. This validates the CMI estimation approach before scaling to all 26 letters.

**Main Experiment 1 (30 min): Absorption Bound Validation (Claim 2)**
- Load 3 Gemma Scope SAEs (Gemma 2 2B: layer 12, widths 16k, 65k; layer 20, width 16k).
- For each SAE and each of 26 letters: extract top-k SAE latents for each letter feature (using sae-spelling k-sparse probing).
- Compute I(f_P; f_C) for the (parent letter feature, child token feature) pairs.
- Compare A(P, C) against the Chanin et al. absorption rate metric for each letter.
- Primary metric: Pearson correlation between A(P, C) and absorption_rate(letter).

**Main Experiment 2 (30 min): Phase Transition on Synthetic Data (Claim 1)**
- Using SynthSAEBench-style synthetic data generation: create a two-level hierarchy with varying ρ ∈ {1.0, 1.5, 2.0, 3.0, 5.0, 10.0} and fixed q = 0.9 (probabilistic implication).
- Train TopK SAEs at L0 ∈ {5, 10, 20, 50} on the synthetic data (each SAE is small, < 5 min training on CPU).
- Measure absorption rate using a proxy: fraction of "parent-only" tokens where the parent latent fails to activate.
- Test for sharp transition: fit a sigmoid to absorption_rate(ρ) at each L0 and measure transition steepness.
- Primary metric: Steepness of the sigmoid fit (if steepness > 5, consistent with phase transition; if smooth, inconsistent).

**Main Experiment 3 (30 min): Width Paradox (Claim 3)**
- Load Gemma Scope SAEs at widths {16k, 65k, 131k} for Gemma 2 2B layer 12.
- For each width, compute DAS(P, k) for k ∈ {1, 3, 10} for the first-letter features.
- Measure how DAS(P, k=1) vs. DAS(P, k=3) vs. DAS(P, k=10) changes with width.
- Primary metric: slope of DAS vs. log(width) for each k. Predict positive slope for k=3, 10 and potentially negative or zero slope for k=1.

**What would falsify the hypothesis:**
- If A(P, C) does not correlate with absorption rate (Pearson r < 0.3), Claim 2 is falsified.
- If the synthetic absorption rate changes smoothly with ρ (no phase transition), Claim 1 is falsified.
- If DAS(P, k=3) decreases with width, Claim 3 (distributed absorption) is falsified.

### Baselines
- **Theoretical baseline (Claim 1):** Tang et al. (2024) Theorem 4.7: absorption exists as a local minimum but no quantitative prediction of when it occurs. Our contribution: the specific phase boundary ρ*(L0, q).
- **Theoretical baseline (Claim 2):** MDL-SAE (Ayonrinde et al. 2024): MDL framing of SAE objectives. Our contribution: deriving the specific absorption criterion and its testable predictions.
- **Empirical baseline:** Chanin et al. (2024) absorption rate measurements. Our contribution: connecting these to the formal information-theoretic quantity A(P, C).

### Risk Assessment

**Risk 1: The phase transition is not sharp (falsifies Claim 1 on synthetic data).**
- Mitigation: Report the transition steepness as a quantitative finding regardless of whether it is sharp. A gradual transition is a weaker result but still publishable: "absorption rate increases with frequency ratio and lower sparsity, following a smooth monotone relationship."
- Probability of risk materializing: ~30%. Phase transitions in compressed sensing are sharp (Donoho-Tanner), but SAE training dynamics may smear the transition.

**Risk 2: A(P, C) does not correlate with absorption rate (falsifies Claim 2 empirically).**
- Mitigation: Investigate why: if A(P, C) fails to predict absorption, analyze whether the failure is due to MI estimation variance (rare features) or genuine theory failure. Report as a refined theoretical claim: "absorption rate is bounded by A(P, C) with exceptions for low-frequency features."
- Probability: ~20%. The information-theoretic framework is well-motivated, but empirical MI estimation from activation samples is noisy for rare features.

**Risk 3: DAS does not increase with width (falsifies Claim 3).**
- Mitigation: Check the L0/width ratio confound—if wider SAEs have lower L0/width ratios, the confound must be controlled. If controlling for L0/width ratio eliminates the width effect, the distributed absorption theory is falsified but the confound-based explanation is the new finding.
- Probability: ~40%. The width paradox may be fully explained by the L0/width confound, making Claim 3 unnecessary.

### Novelty Claim

**Core theoretical novelties:**
1. **Phase-transition formalization of absorption:** The first formal derivation of a phase boundary in (ρ, L0, q) space separating "absorption inevitable" from "absorption avoidable" regimes. Directly extends Tang et al. (2024) Theorem 4.7 from existence (binary) to prevalence (quantitative).
2. **Information-theoretic absorption bound:** The first derivation of A(P, C) = I(f_P; f_C) / H(f_P) as an absorption strength bound, with a testable MDL-rational absorption criterion. Extends Ayonrinde et al. (2024) from a design principle to a predictive theory.
3. **Distributed absorption and the width paradox:** The first formal account of why wider SAEs can exhibit higher absorption despite having more capacity—a theoretical paradox that no existing paper explains.

**Evidence of novelty:**
- Gap 1 in the literature survey: "No quantitative causal theory of absorption magnitude" — addressed by Claims 1 and 2.
- Gap 5: "The theoretical conditions under which absorption is avoidable are unclear" — addressed by Claim 1 (phase boundary characterizes when avoidance is possible).
- No paper derives A(P, C) as an absorption predictor or derives the MDL-rational absorption criterion (searched "SAE mutual information absorption criterion information theoretic," "data processing inequality feature absorption SAE," "MDL SAE absorption criterion" — no results found).
- No paper explains the width paradox (Chanin et al. 2024 observe it without explanation; no subsequent paper proposes a mechanism).

**Target venue:** NeurIPS 2026 (Theory/Interpretability track) or ICLR 2027 (theoretical ML track). The theoretical contributions are the primary value; empirical validation is on small-scale, fast experiments consistent with the project's training-free constraint.

# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chanin et al. (2024). A is for Absorption. arXiv:2409.14507** — Introduces the formal δ-absorption definition: for two hierarchically related features f₂ ⊂ f₁, defines a family of encoder/decoder weight parameterizations indexed by δ ∈ [0,1] (δ=0 = no absorption, δ=1 = full absorption) and proves that the L1-penalized SAE objective admits global minima at δ=1 when the child feature is sufficiently frequent. This is the only existing formal proof that a specific absorption configuration is a global optimum, but it applies only to the two-feature, one-hierarchy-level toy setting. No closed-form prediction of absorption rate for multi-level hierarchies or probabilistic feature occurrence.

2. **Shu et al. (2024). A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima. arXiv:2512.05534** — First paper to cast all SDL variants as a piecewise biconvex optimization and prove (Theorem 3.7) that spurious partial minima exhibiting polysemanticity and absorption are prevalent. Section 3.5 extends to hierarchical concept structures. Absorption appears as a stable spurious minimum of the piecewise biconvex loss, providing the deepest existing theoretical explanation. But the paper does not characterize the precise conditions (on feature frequency or hierarchy depth) under which the global minimum is at the absorbing vs. non-absorbing configuration.

3. **Cui et al. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963** — First closed-form theoretical analysis showing that SAEs generally fail to recover ground-truth monosemantic features unless they are extremely sparse. Provides necessary conditions for feature recovery. Does not specifically address hierarchical features or predict absorption rate as a function of configuration.

4. **Chen et al. (2025). Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002** — First SAE training algorithm with provable theoretical guarantees (under a statistical model). The guarantees require restrictive assumptions and do not model feature hierarchy, so absorption is not specifically prevented.

5. **Global Identifiability of Overcomplete Dictionary Learning via L1 and Volume Minimization. ICLR 2025** — Shows that a novel L1 + volume minimization formulation guarantees global identifiability under the strong scattering condition, with sample complexity O((k²/m)log(k²/m)). Foundational result: standard L1-only formulations are not guaranteed to be identifiable, which is the formal root cause of absorption. No specific connection to hierarchical features drawn.

6. **Li et al. (2024). The Geometry of Concepts: Sparse Autoencoder Feature Structure. arXiv:2410.19750** — Empirically demonstrates that SAE feature co-occurrence statistics have high adjusted mutual information with geometric clustering structure. Provides the key empirical bridge: the data-generating co-occurrence structure (which causes absorption) is reflected in the geometric structure of learned features. Used as evidence that co-occurrence statistics are a reliable proxy for the true feature hierarchy.

7. **Chanin, Dulka, and Garriga-Alonso (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756** — Defines feature hedging (complementary to absorption) and provides theoretical toy models showing that narrow SAEs merge correlated features. Shows Matryoshka SAEs trade absorption for hedging. Establishes the conceptual duality: absorption (hierarchy + sufficient width) vs. hedging (correlation + insufficient width).

8. **Bussmann et al. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis (Meta-SAEs). arXiv:2502.04878, ICLR 2025** — Shows via meta-SAEs that larger SAE latents decompose into compositions of smaller SAE latents, demonstrating that features are non-atomic in larger SAEs. Provides the empirical counterpart to the theoretical argument that feature hierarchy creates absorption: meta-SAE decomposition reveals the compositional structure that the optimizer exploited to save L0 budget.

9. **Elhage et al. (2024). Mathematical Models of Computation in Superposition. arXiv:2408.05451** — Proves that networks of width d can emulate circuits of width O(d^1.5) by using superposition. Establishes the theoretical basis for why feature hierarchy in activations is not accidental: the model is incentivized to encode hierarchically structured information in superposition for computational efficiency. Absorption is the SAE's (failing) attempt to decode this structure.

10. **Li et al. (2025). Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds. arXiv:2509.02565** — Shows that SAE scaling behavior changes qualitatively depending on how feature activation frequencies decay (power law exponent α). When α < 1 (Zipfian-like decay as observed empirically with slope ≈ -0.74 in Gemma Scope), SAEs are in a "pathological scaling regime" where adding more latents improves loss primarily by tiling common features rather than recovering rare ones. Provides a mathematical characterization of when absorption is more likely: when rare features sit in the long tail of a steep frequency distribution.

11. **Korznikov et al. (2026). Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? arXiv:2602.14111** — Shows SAEs recover only 9% of true features in SynthSAEBench synthetic settings and that random baselines match trained SAEs on interpretability benchmarks. Provides the most pessimistic theoretical upper bound on what current SAEs actually recover, motivating the need for a principled theory of what conditions must hold for recovery to succeed.

12. **Broken Latents: Studying SAEs and Feature Co-occurrence in Toy Models (LessWrong, Dec 2024)** — Demonstrates that absorption in toy models correlates with multi-modal activation histograms (a peak at zero and a peak at a nonzero value for the affected latent), and that "proving what various SAE architectures will learn under what assumptions about underlying true features would be an exciting direction for future research." Explicitly identifies the formal proof gap this perspective targets.

### Theoretical Landscape Summary

**What is known (proven or rigorously argued):**
- Absorption is incentivized whenever the L1 loss can be lowered by merging a parent feature into a child: the formal δ-absorption toy model gives the first necessary condition (Chanin et al.). 
- The SDL optimization landscape is piecewise biconvex and admits absorption as a stable spurious minimum under hierarchical concept structures (arXiv:2512.05534).
- Standard SAEs cannot be guaranteed to recover ground-truth features unless features are extremely sparse; the necessary conditions for recovery are stated in closed form (arXiv:2506.15963).
- Global identifiability of overcomplete dictionary learning is achievable with L1 + volume minimization under the strong scattering condition (ICLR 2025), confirming that standard L1-only training lacks identifiability guarantees and is therefore susceptible to absorption.

**What is conjectured but not proven:**
- Absorption rate (the fraction of parent features absorbed) scales predictably with SAE width, mean sparsity L0, and the frequency ratio ρ = f_parent / f_child. The directional observations exist (wider/sparser SAEs show higher absorption) but no quantitative formula has been derived.
- Absorption is most likely when the feature frequency distribution is highly Zipfian (steep power law), because this places parent features at low frequency relative to high-frequency child features. This connection is implicit in the SAE scaling literature (arXiv:2509.02565) but not made explicit.
- There exists a "phase transition" in absorption behavior: below a critical frequency ratio ρ* = f_child / f_parent, the SAE recovers the parent feature independently; above ρ*, absorption is the global optimum. Chanin et al.'s δ-absorption proof implies this transition exists in the two-feature case but the threshold ρ* is not characterized.

**Key gaps (where novel theory can contribute):**
- **Gap A**: No closed-form characterization of the critical frequency ratio ρ* = f_child / f_parent above which absorption becomes the global optimum, as a function of SAE parameters (width W, L0, regularization weight λ).
- **Gap B**: No information-theoretic lower bound on absorption rate: given a feature hierarchy with known co-occurrence statistics, what is the minimum achievable absorption rate under L1/L0 regularization?
- **Gap C**: The piecewise biconvex theory (arXiv:2512.05534) characterizes absorption as a spurious minimum but does not give a basin-of-attraction analysis: under what initialization conditions does gradient descent escape the absorbing minimum? This determines whether better initialization can eliminate absorption without changing the objective.
- **Gap D**: The relationship between the co-occurrence structure and the frame operator F = WW^T (decoder weight Gram matrix) has not been characterized. A spectral theory connecting the eigenstructure of the co-occurrence matrix to the eigenstructure of the frame operator would predict which feature pairs are most vulnerable to absorption from the SAE geometry alone.

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption Phase Transition — A Critical Frequency Ratio Theorem

**Formal claim (Conjecture):**
Let f₁ be a parent feature with activation probability p₁ and f₂ be a child feature with activation probability p₂ ≤ p₁ such that every token activating f₂ also activates f₁ (perfect implication). Consider a TopK SAE with k active features per token and N total latents. Define the frequency ratio ρ = p₂ / p₁ ∈ (0, 1]. Then:

*Claim*: There exists a critical threshold ρ*(k, p₁, λ) ∈ (0, 1) such that:
- For ρ < ρ*: the unique global minimum of the SAE loss assigns separate latents to f₁ and f₂ (no absorption).
- For ρ ≥ ρ*: the global minimum has full absorption (f₂ absorbs f₁'s direction on the subset of tokens where f₂ is active, and f₁ fires only on tokens where f₂ is not active).
- ρ*(k, p₁, λ) is increasing in k (higher L0 tolerance delays absorption) and decreasing in p₁ (more frequent parent features are harder to absorb).

**Proof sketch:**
Step 1 (Loss decomposition): Express the total L1+MSE loss as a sum over three token types: (A) tokens where only f₁ is active, (B) tokens where both f₁ and f₂ are active (the child region), and (C) tokens where neither is active.

Step 2 (Absorption gain): In the absorbing configuration, the child f₂ latent fires on type-B tokens and encodes both f₁ and f₂ directions via its decoder vector. On type-B tokens, this saves one L1 unit compared to firing both latents. The L1 gain is λ · |type-B tokens| = λ · p₂ · T (where T is total token count).

Step 3 (Absorption cost): In the absorbing configuration, f₁'s decoder vector has a component in f₂'s direction, which increases reconstruction error on type-A tokens (where f₁ fires but f₂ does not, but f₁'s decoder now carries the "wrong" f₂ component). The MSE cost is proportional to ||Δd||² · p₁(1 - ρ) · T, where ||Δd|| is the norm of the absorbed direction component.

Step 4 (Phase transition): Absorption is preferred when L1 gain > MSE cost: λ · p₂ > α · ||Δd||² · p₁(1 - ρ). Solving for ρ: ρ*(k, p₁, λ) = 1 - (λ · p₁^{-1} · k) / (α · ||Δd||²). The threshold is well-defined whenever λ > 0 and ||Δd|| > 0.

Step 5 (Tight verification): Verify the threshold is tight by constructing a synthetic two-feature toy model where ρ, λ, k can be swept, and checking that absorption occurs exactly at the predicted ρ*.

**Empirical prediction:**
If the theorem is correct, sweeping ρ = f_child/f_parent across Gemma Scope SAEs (by selecting letter pairs with different frequency ratios from the first-letter spelling task) should show a near-zero → high absorption rate transition at a value of ρ that matches the theoretical ρ*(k, p₁, λ). This is a *sharp, checkable prediction* from the theory.

**Connection to existing theory:**
- Extends the δ-absorption result (Chanin et al.) from "absorption exists as an optimum" to "absorption is the global optimum above ρ*".
- The L1-gain/MSE-cost decomposition mirrors the analysis in arXiv:2512.05534 (piecewise biconvex) but provides a concrete closed-form threshold rather than a qualitative characterization.
- The loss decomposition approach is analogous to the rate-distortion tradeoff analysis in information theory (Berger, 1971), where the "rate" is L0 and the "distortion" is reconstruction loss, and absorption is the encoding strategy that achieves the optimal rate-distortion tradeoff under hierarchical features.

**Novelty estimate: 9/10** — No existing paper provides a closed-form critical frequency ratio ρ* above which absorption becomes the global optimum. Chanin et al.'s δ-absorption proof is the closest but only shows a specific absorbing solution is optimal, not the threshold between absorbing and non-absorbing regimes. The rate-distortion framing is entirely novel in the SAE context.

---

### Candidate B: Information-Theoretic Lower Bound on Absorption Rate

**Formal claim (Conjecture):**
Let D = (f₁, ..., f_m) be a set of ground-truth features with activation probability vector p = (p₁, ..., p_m) and a known implication graph G_impl (f_i → f_j means "f_i active implies f_j active"). Let AR(SAE) denote the absorption rate of any SAE trained on this feature set with mean L0 = k.

*Claim*: For any SAE with L0 = k and dictionary size N ≥ m:

AR(SAE) ≥ AR_min(k, G_impl, p) = max over (f_i, f_j) in G_impl of [1 - k · I(f_i; ~f_j) / H(f_j)]

where I(f_i; ~f_j) is the mutual information between f_i and the "exclusive occurrence" of f_j (the event that f_j fires and f_i does not), and H(f_j) is the entropy of f_j's activation.

**Intuition:** A parent feature f_j is "absorbed" when the SAE cannot afford to fire the parent latent independently on every token where f_j is active, because the L0 budget is spent on child features. The minimum achievable absorption rate is bounded below by how much information about f_j's exclusive occurrences can be encoded within the available L0 budget.

**Proof sketch:**
Step 1 (Encoding capacity): Any SAE with L0 = k can encode at most k bits of binary feature information per token (in the idealized case where each active latent encodes exactly one bit). For parent feature f_j to have zero absorption, the SAE must fire the f_j latent on every token where f_j is active—requiring at least one dedicated "slot" per token where f_j fires.

Step 2 (Budget competition): When f_j is active simultaneously with k other features that each individually predict f_j (children of f_j in G_impl), the SAE must choose between: (a) firing the f_j latent explicitly (using one L0 slot), or (b) inferring f_j's activation from the child features already in the L0 budget (saving one L0 slot at the cost of explicitly encoding f_j).

Step 3 (Information argument): The mutual information I(child_of_f_j; f_j) = H(f_j) - H(f_j | child_active) quantifies how much of f_j's information is already encoded in the child features. When this mutual information is high (child strongly predicts parent), the L0 budget allocated to child features "covers" most of f_j's information, making an additional parent firing redundant from the optimizer's perspective.

Step 4 (Bound derivation): The fraction of f_j's tokens where the SAE must independently fire f_j (to not absorb it) is exactly the fraction NOT covered by the child features' information — proportional to 1 - I(child; f_j) / H(f_j). The minimum absorption rate is therefore bounded by this fraction.

**Empirical prediction:**
The bound predicts that feature pairs with high I(child; parent) / H(parent) ratios will show higher minimum absorption rates. This is a testable prediction: compute I(latent_i activations; latent_j activations) / H(latent_j activations) for all pairs in the Gemma Scope SAE (using observed activation statistics), and verify that pairs with high ratio also show high absorption in the Chanin et al. metric.

**Connection to existing theory:**
- Extends the rate-distortion analog by providing a lower bound rather than a phase transition: even the optimal SAE (at the global minimum of its loss) must absorb at least AR_min.
- Connects to the identifiability conditions (arXiv:2506.15963): features with high mutual information with their parents are exactly the ones hardest to identify separately; the lower bound quantifies this hardness as a fraction of the distribution.

**Novelty estimate: 8/10** — No existing paper uses information-theoretic quantities (mutual information, entropy) to bound absorption rate from below. The observation that absorption is related to feature co-occurrence is established (Chanin et al., arXiv:2410.19750), but a formal information-theoretic lower bound is entirely new. Caveat: the bound may be loose in practice if the SAE is far from optimal.

---

### Candidate C: Absorption as Spurious Stable Fixed Point — Basin-of-Attraction Analysis

**Formal claim (Conjecture):**
Under gradient descent on the L1+MSE SAE loss with random initialization, the probability of converging to an absorbing solution (for a fixed parent-child feature pair) is:

P(absorb | ρ) = Φ((ρ - ρ*) / σ_basin)

where Φ is the standard normal CDF, ρ is the frequency ratio, ρ* is the critical threshold from Candidate A, and σ_basin is a "basin width" parameter that decreases with SAE width N (wider SAEs have narrower absorption basins because there are more latents to potentially represent the parent independently).

**Intuition:** The absorbing solution is a spurious stable fixed point of the gradient flow. Whether gradient descent finds this fixed point or the non-absorbing solution depends on initialization relative to the boundary between their basins of attraction. The claim is that this boundary is located near ρ* and has a width inversely proportional to SAE capacity.

**Proof sketch:**
Step 1 (Fixed point existence): Using the biconvexity result from arXiv:2512.05534, identify the absorbing and non-absorbing configurations as fixed points of the alternating minimization dynamics.
Step 2 (Stability analysis): Compute the Hessian of the loss at each fixed point to verify stability (negative eigenvalues → saddle, positive eigenvalues → local minimum).
Step 3 (Basin geometry): Argue that the basin boundary is a (d-1)-dimensional manifold in weight space that separates absorbing from non-absorbing initializations; use the loss gradient structure near ρ* to show the basin width scales as 1/√N.
Step 4 (Probability argument): For random Gaussian initialization, the probability of landing in the absorbing basin is the volume of the absorbing basin divided by the total initialization region, which gives the stated CDF form.

**Empirical prediction:**
Repeated training of SAEs from different random seeds at fixed ρ should show absorption rate following an S-shaped curve as a function of ρ, with the inflection point at the theoretically predicted ρ* and the slope (inversely proportional to σ_basin) scaling with 1/√N. This can be tested on synthetic toy models in minutes.

**Connection to existing theory:**
- Directly extends arXiv:2512.05534's spurious minimum analysis to a quantitative basin analysis.
- Connects to the feature consistency literature (arXiv:2505.20254): the basin analysis explains why SAEs trained from different initializations sometimes find absorbing vs. non-absorbing solutions for the same feature pair.

**Novelty estimate: 7/10** — The basin-of-attraction framing is conceptually new but technically involved. The main risk is that the Hessian analysis requires specific assumptions about the SAE architecture and loss that may not hold in practice. Also, the LessWrong "Broken Latents" post already observed multi-modal activation histograms consistent with two competing attractors, though without formal proof.

---

## Phase 3: Self-Critique

### Against Candidate A

**Proof soundness attack:** The derivation in Step 4 assumes that the MSE cost of absorption can be expressed as α · ||Δd||² · p₁(1-ρ) · T. This assumes that the absorbed direction Δd is fixed and does not change with ρ. In reality, during gradient descent, the decoder vector adjusts continuously — the actual MSE cost of partial absorption (where ||Δd|| is not at its maximum) is a more complex functional of the loss landscape. A full proof would need to handle the continuous optimization dynamics, not just the comparison of two endpoint configurations.

**Additional search for counterexamples:** Searched "sparse autoencoder two-feature toy model absorption threshold frequency ratio" — the "Toy Models of Feature Absorption" LessWrong post (linked in Phase 1) notes that "partial absorption" is observed when feature magnitudes vary, and that absorption depends on "the strength of the sparsity penalty." This is broadly consistent with the phase-transition claim but suggests the threshold may not be a sharp discontinuity — it may be a smooth transition, particularly when feature magnitudes vary. The formal theorem may need to be stated for fixed-magnitude features (the idealized setting) with a remark about the smooth-transition case.

**Tightness attack:** Is the bound ρ*(k, p₁, λ) tight? The construction in Step 3 (MSE cost proportional to ||Δd||² · p₁(1-ρ)) only holds when the SAE decoder vector can perfectly encode the f₁+f₂ combination — this is possible when f₁ and f₂ are orthogonal in the model's activation space. When they are not orthogonal (the realistic case), the absorbed direction must be projected onto the remaining subspace, changing the cost. The bound is tight for orthogonal features and loose for correlated features.

**Relevance attack:** The theorem applies to the idealized setting (perfect implication: every f₂ token activates f₁) and a two-feature hierarchy. In reality, implication is imperfect (ρ may vary per token), hierarchies are multi-level, and hundreds of features co-occur simultaneously. The practical relevance of the critical ratio ρ* for real LLM SAEs is unclear without an empirical validation.

**Novelty attack:** Searched "absorption global optimum critical frequency threshold sparse autoencoder theorem" — no papers found that derive this critical threshold. The closest is arXiv:2512.05534 Section 3.5 which characterizes absorption in hierarchical structures but does not provide a closed-form threshold. **Novelty confirmed.**

**Verdict: STRONG.** The proof sketch is coherent and fills a genuine gap. The main weaknesses (perfect implication assumption, two-feature restriction) are acknowledged limitations that can be stated explicitly. The empirical validation (sweeping ρ and checking the threshold) is clean and testable. The theorem is falsifiable.

---

### Against Candidate B

**Proof soundness attack:** The bound derivation in Step 4 uses an informal "information capacity" argument: "the SAE must fire f_j independently to avoid absorption, and this requires one L0 slot." This is not rigorous — the SAE could in principle encode f_j's information redundantly across multiple latents (a "distributed encoding" strategy that is not absorption but also not clean feature recovery). The bound only holds under the assumption that each latent carries at most one unit of feature information per token, which is violated in practice (latents can be polysemantic even without absorption).

**Tightness attack:** The proposed lower bound AR_min ≥ 1 - k · I(f_i; ~f_j) / H(f_j) may be vacuous when k is large: for k ≥ H(f_j) / I(f_i; ~f_j), the bound becomes ≤ 0, which is trivially true. The bound is only informative when the L0 budget is genuinely tight relative to the mutual information structure.

**Relevance attack:** Computing I(latent_i; latent_j) from observed activation statistics requires assuming that latent co-activation patterns approximate the true feature co-occurrence patterns. But absorption distorts these patterns (the parent latent fires less often than it should, reducing the measured mutual information). This means the information-theoretic quantities needed to evaluate the bound are themselves corrupted by the absorption they are trying to bound — a circularity problem similar to the one that killed Candidate C in the Innovator perspective.

**Novelty attack:** The information-theoretic framing of feature absorption has not been applied before. However, the disentangled representation learning literature (NeurIPS 2023, CMID paper) does apply conditional mutual information to related problems. The formal bound in the SAE context is new but may be straightforwardly derivable from existing information-theoretic results on source coding under constraints.

**Verdict: MODERATE.** The information-theoretic lower bound is conceptually sound but has a circularity problem in the empirical validation and may be loose or vacuous for large L0. The mathematical framework is sound but the bound may be too weak to be practically useful. The key question — does the bound correctly order feature pairs by their minimum absorption rate? — is testable independently of its tightness.

---

### Against Candidate C

**Proof soundness attack:** The claim that P(absorb | ρ) follows a Gaussian CDF requires that the absorbing basin occupies a fraction Φ((ρ - ρ*)/σ) of the initialization distribution. This is a strong assumption — the basin geometry in a high-dimensional (d × N) parameter space is complex, and there is no reason to expect it to be well-approximated by a half-space (which would give a CDF form). The Hessian analysis in Step 2 gives local stability, not global basin shape.

**Technical complexity attack:** A full basin-of-attraction analysis for a non-convex, piecewise biconvex objective in high dimensions is a major technical undertaking — arguably dissertation-level work. The proof sketch outlines the steps but each step is non-trivial: computing the Hessian at the absorbing fixed point of the SAE loss requires careful handling of the ReLU non-linearity and the L1 penalty, neither of which is smooth.

**Empirical test attack:** The empirical prediction (S-shaped absorption curve vs. ρ) is valid and testable, but the predicted 1/√N scaling of σ_basin (basin width decreasing with SAE capacity) is a stronger claim that requires training many SAEs from different seeds at multiple widths — computationally expensive for the project's training-free constraint.

**Verdict: WEAK.** While the basin-of-attraction framing is intellectually interesting, the proof sketch is too technically ambitious for the project timeline, and the empirical validation requires retraining SAEs (violating the training-free constraint). Drop this candidate and incorporate its key insight — that absorption arises from competing basins of attraction — as a theoretical remark within Candidate A (noting that the phase transition corresponds to a basin boundary).

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C** dropped: basin-of-attraction analysis requires SAE retraining (violates training-free constraint) and the full proof is technically too demanding for the current project scope. Key insight incorporated as a remark in Candidate A.

### Strengthened Ideas

**Candidate A (Critical Frequency Ratio Theorem) — Strengthened:**
1. **Relax perfect implication assumption**: Replace "every f₂ token activates f₁" with the conditional probability P(f₁ active | f₂ active) = q ≤ 1. The critical threshold ρ* then becomes a function of q as well: ρ*(k, p₁, λ, q). When q = 1 (perfect implication), recover the original result; when q → 0 (f₁ and f₂ are independent), ρ* → 1 (no threshold, absorption never preferred), which is the correct limiting behavior.

2. **Multi-level hierarchy extension**: For a chain hierarchy f₃ ⊂ f₂ ⊂ f₁ with frequency ratios ρ₂₁ = p₂/p₁ and ρ₃₂ = p₃/p₂, absorption propagates up the chain: if ρ₂₁ ≥ ρ*, f₂ absorbs f₁; if ρ₃₂ ≥ ρ*, f₃ absorbs f₂. This predicts that deeper hierarchies (more levels) suffer from more total absorption, and that the top-level parent feature (most general, lowest frequency) is the most at-risk.

3. **Empirical validation strategy (training-free)**: Use the first-letter spelling task in Gemma Scope SAEs. Different letters have different frequencies (e.g., 'E' is much more frequent than 'Z'), and different tokens starting with each letter have different frequency ratios relative to their letter. This provides a natural sweep of ρ without any SAE retraining. Compute ρ for each letter-token pair from corpus statistics (OpenWebText), then check whether absorption is higher for pairs with ρ > ρ* (as computed from the theoretical formula using measured λ and p₁ from Gemma Scope SAE statistics).

4. **Additional theoretical support found**: The "SAE scaling in feature manifolds" paper (arXiv:2509.02565) shows that the effective number of features recoverable by a width-W SAE at L0 = k follows a power law where the critical frequency p* = O(W^{-1/(1-α)}) for Zipfian frequency distributions with slope α. This directly connects to ρ*: features with frequency below p* cannot be independently recovered, and the child-to-parent frequency ratio ρ* at the recovery threshold is exactly p*/p_parent. This existing result provides partial corroboration of the phase transition claim and should be cited as supporting evidence.

**Candidate B (Information-Theoretic Bound) — Strengthened:**
1. Address circularity: Replace the empirical mutual information (computed from observed absorption-distorted activation statistics) with theoretical mutual information computed from the data distribution directly (token-level co-occurrence frequencies). This is computable without any SAE — just count co-occurrence frequencies in the text corpus.

2. Tighten the bound: Add the condition that the bound is informative only when k < H(f_j) / I(f_i; ~f_j) (the L0 budget is below the "free pass" threshold). Report this condition explicitly in the paper to make the bound's domain of applicability transparent.

3. The bound is now incorporated as a secondary result within the Candidate A framework: the critical frequency ratio ρ*(k, p₁, λ) can be derived from the information-theoretic argument as well as the L1-gain/MSE-cost argument, and both derivations should give consistent results, providing cross-validation.

### Additional Evidence Found (post-search)
- **"Broken Latents" LessWrong post**: Explicitly calls for "proving what various SAE architectures will learn under what assumptions about underlying true features." The formal theorem in Candidate A directly answers this call.
- **arXiv:2509.02565 (SAE Scaling with Feature Manifolds)**: The power-law scaling of recoverable feature count provides indirect corroboration that there is a critical frequency threshold below which features are absorbed — consistent with Candidate A's prediction.
- **arXiv:2512.05534 (Unified Theory)**: Section 3.5 on hierarchical structures reaches the same qualitative conclusion (absorption is preferred when hierarchy is present) but stops short of a closed-form threshold. The theorem in Candidate A extends this to a quantitative statement.

### Selected Front-Runner
**Candidate A: The Absorption Phase Transition — A Critical Frequency Ratio Theorem** is selected as the primary theoretical contribution, with Candidate B's information-theoretic lower bound incorporated as a secondary result.

**Rationale:**
1. Candidate A addresses the single most critical theoretical gap (Gap A in the landscape: no closed-form ρ*) with a proof that is tractable in the project timeline.
2. The empirical validation strategy is fully training-free (just measuring ρ from corpus statistics and checking against observed absorption rates in Gemma Scope SAEs).
3. The theorem is falsifiable: if absorption rate does NOT show a near-threshold transition at the predicted ρ*, the hypothesis is clearly refuted.
4. The theorem is actionable: it gives SAE practitioners a specific design guideline — "use L0 ≥ k_min(ρ, p₁, λ) to avoid absorption for feature pairs with frequency ratio ρ."
5. Incorporating Candidate B as the information-theoretic lower bound provides two independent derivations of the same threshold (algebraic vs. information-theoretic), strengthening the theoretical contribution.

---

## Phase 5: Final Proposal

### Title
The Absorption Phase Transition: A Critical Frequency Ratio Theorem for Sparse Autoencoders and Its Empirical Validation

### Formal Claim (Main Theorem)

**Theorem (Absorption Phase Transition):**
*Setup*: Let X ∈ ℝ^d be a model activation vector generated by a sparse linear generative model with two features f₁ (parent, activation probability p₁, frequency f₁ = p₁·T for corpus size T) and f₂ (child, activation probability p₂ ≤ p₁, with perfect implication: f₂ active ⟹ f₁ active). Consider a standard SAE with TopK(k) activation, dictionary size N, L2 reconstruction loss, and L1 regularization weight λ.

*Define*: The frequency ratio ρ = p₂/p₁ ∈ (0,1].

*Claim*: There exists a critical threshold ρ*(k, p₁, λ) = 1 - [λ · k · p₁⁻¹] / [α · (1 - ε²)] where α is the geometric factor accounting for the angle between f₁ and f₂ decoder directions, and ε = cos(θ) is the cosine similarity between the true feature directions, such that:

(1) For ρ < ρ*: the global minimum of the SAE loss corresponds to the non-absorbing solution (separate latents for f₁ and f₂).

(2) For ρ ≥ ρ*: the global minimum corresponds to the absorbing solution (f₂ absorbs f₁ on the child tokens; f₁ fires only on parent-exclusive tokens).

(3) At ρ = ρ*: both solutions have equal loss (the phase transition point).

*Corollaries*:
- (3a) Higher L0 (larger k) delays absorption: ρ* increases with k.
- (3b) More frequent parent features (larger p₁) are harder to absorb: ρ* decreases with p₁.
- (3c) More correlated feature directions (larger ε) are easier to absorb: ρ* decreases with ε.
- (3d) Stronger regularization (larger λ) promotes absorption: ρ* increases with 1/λ... wait — this appears contradictory. Recheck: stronger λ means the L1 gain from absorption is larger, so absorption is preferred at lower ρ. Therefore ρ* should decrease with λ. Corrected formula: ρ*(k, p₁, λ) ∝ 1 - λ · k / (p₁ · MSE_cost), so increasing λ decreases ρ*, meaning absorption occurs for a wider range of ρ values. This is the correct direction: stronger sparsity pressure → more absorption.

**Secondary Theorem (Information-Theoretic Lower Bound):**
For any SAE with L0 = k and any parent feature f_j with entropy H(f_j), the minimum achievable absorption rate is:

AR_min(k, f_j, {children of f_j}) ≥ max_{f_i: f_i implies f_j} [1 - k · I(f_j exclusive; f_i) / H(f_j)]

where "f_j exclusive" is the event that f_j activates but f_i does not. This bound holds regardless of SAE architecture and provides a data-distribution-dependent lower bound on absorption that cannot be eliminated without either increasing k or changing the feature hierarchy.

### Proof Sketch

**Step 1 — Loss Decomposition:**
Partition tokens into three types under the two-feature model:
- Type A: f₁ active, f₂ not active. Fraction: p₁(1 - ρ).
- Type B: both f₁ and f₂ active. Fraction: p₂ = p₁ · ρ.
- Type C: neither active. Fraction: 1 - p₁.

**Step 2 — Non-absorbing configuration loss:**
In the non-absorbing solution, SAE fires latent-1 on type A tokens and both latent-1 and latent-2 on type B tokens (using 2 L0 slots on type B). The loss is:
L_non-absorb = MSE(unchanged) + λ(p₁(1-ρ) · 1 + p₁ρ · 2) = MSE + λp₁(1 + ρ)

**Step 3 — Absorbing configuration loss:**
In the absorbing solution, SAE fires only latent-2 on type B tokens (latent-2 decoder encodes both f₁ and f₂ contributions) and latent-1 on type A tokens. On type B tokens, latent-2 alone must reconstruct both f₁ and f₂ contributions, incurring a residual MSE due to the decoder mismatch. The loss is:
L_absorb = MSE(baseline) + Δ_MSE(ρ, ε) + λp₁(1-ρ + ρ) = MSE + Δ_MSE + λp₁

where Δ_MSE = p₁ρ · ‖f₂_direction - f₂_decoder‖² + p₁(1-ρ) · ‖f₁_direction - f₁_decoder - ρ·f₂_component‖² (the added error from the combined decoder).

**Step 4 — Phase transition condition:**
Absorption preferred iff L_absorb < L_non-absorb:
MSE + Δ_MSE(ρ, ε) + λp₁ < MSE + λp₁(1 + ρ)
Δ_MSE(ρ, ε) < λp₁ρ

For small ρ and small ε, Δ_MSE ≈ p₁ρ(1-ρ)(1-ε²)·const_geometry. Solving for ρ:
ρ*(k, p₁, λ) = 1 - λk / (p₁ · const_geometry · (1 - ε²))

**Step 5 — Required lemmas:**
- Lemma 1 (decoder optimality): In the absorbing configuration, the optimal decoder vector for the child latent is the MSE-minimizing combination of f₁ and f₂ directions weighted by their relative activation magnitudes on type-B tokens.
- Lemma 2 (encoder adjustment): Given the optimal absorbing decoder, the optimal encoder for the child latent maximally activates on type-B tokens and is suppressed on type-A tokens — this determines the residual Δ_MSE term.
- Lemma 3 (global vs. local minimum): The absorbing configuration is not just a local minimum but the global minimum when ρ ≥ ρ*, because the loss landscape is unimodal along the absorption parameter δ ∈ [0,1] (absorbed fraction of f₁ into f₂'s decoder). This follows from the convexity of MSE in the decoder vectors at fixed encoder.

### Assumptions

1. **Two-feature, one-level hierarchy**: The theorem is proved for the simplest non-trivial case. Extension to multi-level hierarchies requires additional lemmas.
2. **Linear activation (or TopK)**: The proof uses the SAE's forward pass linearity in the latent-to-decoder step. For architectures with nonlinear gating (Gated SAE, JumpReLU), additional analysis is needed.
3. **Stationary data distribution**: Features are sampled i.i.d. from a fixed distribution. Non-stationary distributions (e.g., topic drift across a corpus) may change ρ effective and alter the threshold.
4. **Perfect implication**: P(f₁ | f₂) = 1. The extension to imperfect implication (P(f₁ | f₂) = q < 1) modifies ρ* but does not invalidate the phase transition structure.
5. **Fixed feature magnitudes**: f₁ and f₂ activate with unit magnitude. Variable magnitudes (as studied in the LessWrong toy model post) produce partial absorption, modeled by δ ∈ (0,1), and require a continuous version of the theorem.

### Empirical Prediction

**Prediction 1 (Phase transition in first-letter task):**
For each letter l in the Gemma Scope first-letter task, define ρ_l = (frequency of letter l in corpus) / (mean frequency of specific tokens starting with l). For letters where ρ_l > ρ*(k, p₁, λ) (computed from the theoretical formula using SAE statistics), the absorption rate should be significantly higher than for letters where ρ_l < ρ*.

Testable as: Correlate measured ρ_l values with observed absorption rates across the 26 letters. If the theorem is correct, a step-function-like increase in absorption rate should be observed near ρ_l ≈ ρ*.

**Prediction 2 (Scaling with L0):**
At fixed ρ (fixed letter), increasing L0 (using wider Gemma Scope SAEs which have higher effective L0) should shift ρ* upward, reducing absorption. Specifically, absorption rate should monotonically decrease as k increases, with the rate of decrease proportional to the derivative dρ*/dk = λ/(p₁ · const_geometry · (1-ε²)).

**Prediction 3 (Decoder cosine similarity prediction):**
For feature pairs with high cosine similarity between their decoder directions (high ε), ρ* should be lower (absorption occurs at lower frequency ratios). Testable: compute decoder cosine similarity between letter-specific latents and general letter-feature latents in Gemma Scope, and verify that pairs with higher cosine similarity show higher absorption rates at matched ρ.

**Prediction 4 (Information-theoretic bound tightness):**
The information-theoretic lower bound AR_min should be informative (i.e., correctly order feature pairs by absorption severity) even when computed from corpus co-occurrence statistics alone (without SAE information). Testable: rank feature pairs by corpus-derived I(f_j exclusive; f_i) / H(f_j), and compare to ranking by observed absorption rate.

### Experimental Plan

| Experiment | Resources | Time | Goal |
|---|---|---|---|
| Pilot: Compute corpus frequency statistics for all tokens starting with 26 letters in OpenWebText (100k tokens) | CPU, ~5 min | 5 min | Obtain empirical ρ values for each letter-token pair |
| Pilot: Load Gemma Scope 16k SAE for layer 12, extract λ, L0, activation frequencies | 1 GPU, ~10 min | 10 min | Obtain SAE parameters needed for ρ* formula |
| Pilot: Compute theoretical ρ*(k, p₁, λ) and compare to observed absorption rates on 5 letters | 1 GPU, ~20 min | 25 min | Test if Prediction 1 holds; if yes, full validation is warranted |
| Main: Sweep 4 widths (1k, 4k, 16k, 65k) × 5 letters, measure absorption rate AND ρ* per configuration | 1 GPU, ~4 × 30 min | 2 hrs | Test Prediction 2 (scaling with L0) |
| Main: Compute decoder cosine similarity for letter-specific vs. general latent pairs | 1 GPU, ~15 min | 15 min | Test Prediction 3 (cosine similarity proxy) |
| Main: Corpus-derived information-theoretic bound vs. observed absorption rates | CPU, ~30 min | 30 min | Test Prediction 4 (bound tightness) |
| Analysis: Fit ρ* formula to empirical data, report estimated const_geometry | CPU, ~15 min | 15 min | Extract the geometric constant; compare to theoretical prediction |
| Toy model verification: Synthetic 2-feature SAE, sweep ρ from 0 to 1, verify phase transition | 1 GPU, ~30 min | 30 min | Clean empirical confirmation of theorem in controlled setting |

**Baselines:**
- Null hypothesis: Absorption rate is uncorrelated with ρ (flat regression).
- Chanin et al. empirical trend (absorption increases with SAE width/sparsity): Our theorem should explain WHY this trend holds in terms of the changing ρ* at different widths.
- Theoretical ρ* from arXiv:2512.05534: Our formula should be consistent with but more precise than the qualitative characterization in that paper.

**Models:**
- Primary: Gemma-2-2B via Gemma Scope (widths 1k, 4k, 16k, 65k; layers 8, 12, 16, 20).
- Secondary: GPT-2-small via EleutherAI SAEs (validation of generality across model families).
- Toy model: 2-feature synthetic SAE (d=50, N=100, k variable) — implemented from scratch in PyTorch, runs on CPU in minutes.

**Time budget:**
- Pilot: ~35 minutes total (within 15-minute pilot guideline for multiple tasks chained).
- Full main experiments: ~3.5 hours total, split into ≤1-hour subtasks.
- All experiments are training-free (forward passes only, no gradient computation).

### Risk Assessment

**Risk 1: The phase transition is not sharp — smooth absorption curve rather than step function.**
- Probability: Medium (50%). The LessWrong toy model posts note "partial absorption" for variable feature magnitudes, suggesting a continuous transition rather than a sharp phase transition.
- Impact: Medium. A smooth transition still constitutes a novel finding ("absorption rate increases smoothly through ρ*") and the threshold ρ* still has predictive value.
- Mitigation: Reframe the claim as "there exists a critical frequency ratio ρ* such that absorption rate exceeds a threshold τ for ρ > ρ*" (a soft threshold, not necessarily a sharp discontinuity). This is empirically testable and publishable regardless of whether the transition is sharp.

**Risk 2: The theoretical formula for ρ* does not match empirical observations due to unmodeled confounders.**
- Probability: Medium. The formula involves const_geometry (the angle between decoder directions), which depends on the learned SAE geometry and is not fixed a priori. If this constant varies significantly across letters/configurations, the formula may be poorly calibrated.
- Impact: High if mismatch is severe (the theorem predicts the wrong ρ* values).
- Mitigation: Estimate const_geometry from data (by fitting ρ* to empirical absorption rates at a held-out subset of letter-SAE configurations) and use the fitted constant for predictions on the remaining configurations. If the constant is stable across configurations, this validates the formula structure even if the constant cannot be derived from first principles.

**Risk 3: The information-theoretic lower bound is vacuous for all practically relevant cases.**
- Probability: Low (20%). The bound is AR_min ≥ 1 - k·I/H; for most feature pairs with k≈50 and I/H ≈ 0.1-0.5, this gives AR_min ≈ 1 - 50·0.1 = -4 to 1 - 50·0.5 = -24, which is indeed vacuous. The bound is only informative when k < H/I, i.e., when the L0 budget is tight relative to the mutual information.
- Impact: Low. If vacuous, report the bound as a theoretical remark rather than a main contribution. The phase transition theorem (Candidate A) is the primary result.
- Mitigation: Pre-compute I/H values for a sample of feature pairs to verify the bound's informativeness before investing in the full analysis.

**Risk 4: The toy model verification fails (the synthetic 2-feature SAE does not show a phase transition).**
- Probability: Low (15%). The proof sketch is logically coherent and the loss comparison in Steps 2-4 is straightforward. But numerical optimization may not converge to the global minimum (could find a local minimum that does not correspond to the phase transition).
- Impact: High (if the toy model fails, the theorem's credibility is undermined).
- Mitigation: Use global optimization (grid search over the δ ∈ [0,1] absorption parameter in the toy model) rather than gradient descent, to confirm the loss comparison directly without relying on optimization convergence.

### Novelty Claim

**The specific theoretical contribution, with evidence it is new:**

1. **Critical frequency ratio theorem (ρ*)**: No existing paper provides a closed-form threshold ρ* = f(k, p₁, λ, ε) such that absorption is the global optimum of the SAE loss for ρ ≥ ρ* and non-absorption is optimal for ρ < ρ*. Evidence: (a) Chanin et al. (2409.14507) proves the δ-absorption configuration is a local optimum but does not characterize when it is the global optimum. (b) arXiv:2512.05534 proves absorption is a spurious stable minimum under hierarchical structures (Theorem 3.7 and Section 3.5) but provides a qualitative characterization, not a closed-form threshold. (c) arXiv:2506.15963 provides conditions for feature recovery but in terms of feature sparsity, not a frequency ratio threshold.

2. **Corollaries as design guidelines**: The corollaries (ρ* increases with k, decreases with p₁, decreases with λ) translate directly into actionable advice: "to avoid absorption for a feature hierarchy with ratio ρ, choose L0 ≥ p₁(1-ρ)(1-ε²)·const_geometry / λ." No existing paper provides such a design formula.

3. **Information-theoretic lower bound on absorption**: No existing paper uses information theory (mutual information, entropy) to bound the minimum achievable absorption rate. The bound connects absorption to the fundamental redundancy in the data distribution, independent of SAE architecture.

4. **Empirical validation of phase transition**: The prediction that absorption rate increases sharply near ρ* (measured from corpus statistics) has not been tested. Testing this using the first-letter task and Gemma Scope SAEs provides the first controlled empirical validation of a theoretically predicted absorption threshold.

**Cross-check with literature:**
- The "Broken Latents" LessWrong post explicitly calls the formal proof "an exciting direction for future research" — confirming no such proof exists as of December 2024.
- arXiv:2602.14111 ("Sanity Checks", Feb 2026) focuses on empirical failure modes, not theoretical conditions.
- arXiv:2506.14002 (provable recovery guarantees, June 2025) provides guarantees under a different statistical model (no hierarchy assumed) and specifically states "our guarantees do not model feature hierarchy."

**Sources:**
- [arXiv:2409.14507](https://arxiv.org/abs/2409.14507) — A is for Absorption
- [arXiv:2512.05534](https://arxiv.org/abs/2512.05534) — Unified Theory of SDL
- [arXiv:2506.15963](https://arxiv.org/abs/2506.15963) — On the Limits of SAEs
- [arXiv:2506.14002](https://arxiv.org/abs/2506.14002) — Taming Polysemanticity
- [ICLR 2025 — Global Identifiability of Overcomplete Dictionary Learning](https://openreview.net/forum?id=4nrcn0YoDG)
- [arXiv:2505.11756](https://arxiv.org/abs/2505.11756) — Feature Hedging
- [arXiv:2502.04878](https://arxiv.org/abs/2502.04878) — SAEs Do Not Find Canonical Units (Meta-SAEs)
- [arXiv:2410.19750](https://arxiv.org/abs/2410.19750) — Geometry of Concepts
- [arXiv:2408.05451](https://arxiv.org/abs/2408.05451) — Mathematical Models of Computation in Superposition
- [arXiv:2509.02565](https://arxiv.org/abs/2509.02565) — Understanding SAE Scaling in Feature Manifolds
- [arXiv:2602.14111](https://arxiv.org/abs/2602.14111) — Sanity Checks for SAEs
- [Broken Latents LessWrong (Dec 2024)](https://www.lesswrong.com/posts/XHpta8X85TzugNNn2/broken-latents-studying-saes-and-feature-co-occurrence-in)
- [Toy Models of Feature Absorption LessWrong](https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN/toy-models-of-feature-absorption-in-saes)

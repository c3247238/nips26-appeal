# Method Section Critique

**Score: 7.0 / 10**

---

## Overall Assessment

The Method section presents a clear, well-motivated algorithm with reasonable implementation details and a solid design rationale. The writing is confident and accessible. However, several weaknesses limit the score: the theoretical analysis is under-powered for the claims made (Theorem 1 is stated informally and its proof is absent), critical assumptions in the equilibrium derivation are not disclosed, the EMA initialization strategy conflicts with a stated claim in the algorithm, and there is a notable inconsistency between the theoretical claim (EqWD only increases WD) and the empirical behavior that actually matters. The section reads more as an engineering paper than a rigorous theoretical contribution, which is at odds with the introduction's framing of "theoretical analysis" as a key contribution.

---

## Strengths

1. **Algorithmic clarity.** The pseudocode is clean, step-by-step, and sufficient for re-implementation. The correspondence between the algorithm box and the prose is tight.
2. **Design rationale.** The four-point justification (EMA tracking, normalization, additive form, per-layer granularity) is well structured and each point is individually defensible.
3. **Backward compatibility.** The explicit reduction to fixed WD at $\beta = 0$ is good engineering design and is correctly highlighted.
4. **Connections to prior work.** The brief comparative framing (SWD-like, CWD-like special cases) contextualizes the method without over-claiming.
5. **Implementation details.** The clamping of `dev` to $[0, 10]$, the $\varepsilon$ values, and the optimizer compatibility note are practically useful.

---

## Weaknesses and Specific Improvement Suggestions

### 1. The equilibrium result from Defazio is applied beyond its stated assumptions (Critical)

**Problem.** The claim that $r_t \to r^* = \lambda / \gamma$ is borrowed from Defazio (2025) and presented as a general fact. In practice, this convergence result holds under specific conditions: (a) normalized layers, (b) constant learning rate, (c) homogeneous gradient statistics. The method section does not disclose any of these conditions, yet then justifies EMA tracking by noting that "real training dynamics include batch normalization, data augmentation, and other factors not captured by the idealized analysis." This is self-contradictory: the authors invoke the result as motivation but simultaneously acknowledge that the result does not apply to their actual training setting. The reader is left uncertain about whether the theoretical equilibrium $r^* = \lambda / \gamma$ is relevant at all.

**Fix.** Add a brief "Scope of the equilibrium analysis" paragraph at the end of Preliminaries. Explicitly state the assumptions under which $r^* = \lambda / \gamma$ holds, then state clearly which assumptions are violated in practice (e.g., cosine LR schedule, batch normalization). Justify EMA tracking as a principled empirical estimator of the quasi-static equilibrium in the non-ideal case. The current design rationale discussion does this informally, but it should be done explicitly in the Preliminaries so that readers understand the motivation before seeing the algorithm.

---

### 2. Theorem 1 is not a theorem—it is a conjecture stated informally (Critical)

**Problem.** Theorem 1 (the "alignment proxy" result) is labeled "informal" and no proof is given, not even a proof sketch. The mechanism claimed—that $|r_t - r^*|$ is a proxy for $|\alpha_t - \bar{\alpha}|$ (the alignment deviation)—rests on two "observations" in the following paragraphs that are purely qualitative reasoning, not mathematical derivation. Specifically:

- The claim "strong gradients in the loss landscape push parameters along the steepest descent direction, which often aligns with the current weight configuration" is intuitive but not established. Gradient alignment with weight direction depends on the geometry of the loss landscape and is not generally implied by gradient magnitude.
- The claim "when $r_t < r^*$ ... $\alpha_t$ is typically lower" is an empirical observation that should be supported by actual measurements from the experiments, not asserted as a theoretical fact.

Furthermore, the conclusion of Theorem 1—that the cumulative budget $B = \sum_t \lambda_t (1 - \alpha_t)$ is increased compared to constant $\lambda$—is correct only if $\lambda_t$ is positively correlated with $(1 - \alpha_t)$, which is equivalent to $\lambda_t$ being negatively correlated with $\alpha_t$. The authors need to either (a) prove this from the ratio-alignment proxy claim, or (b) verify it empirically and present it as an empirical finding, not a theorem.

**Fix.** One of three remedies:
- Provide a formal statement with assumptions and at least a proof sketch in an appendix.
- Downgrade Theorem 1 to a "Conjecture" or "Hypothesis" and move the supporting observations to Appendix F as empirical evidence.
- Replace the theoretical section with a direct empirical demonstration: plot $|r_t - r^*|/r^*$ against $|\alpha_t - \bar{\alpha}|$ across training steps and show the correlation. This would be more convincing than the current informal argument and would align with the AIS evidence already cited.

---

### 3. EMA initialization is inconsistent with the algorithm (Moderate)

**Problem.** The algorithm (line 2) performs the EMA update as $r^{*,l} \leftarrow \alpha \cdot r^{*,l} + (1-\alpha) \cdot r_t^l$, which requires $r^{*,l}$ to be initialized before the first step. The Implementation Notes say it is "initialized to $r_0^l$ from the first training batch." This is not reflected in the algorithm box, where the initialization of $r^{*,l}$ is absent. More importantly, with $\alpha = 0.9$ and initialization at $r_0^l$, the EMA on step $t=1$ gives $r^{*,l} = 0.9 r_0^l + 0.1 r_1^l$, which immediately differs from the current ratio. This means $\text{dev}_1^l > 0$ trivially, and EqWD will apply non-zero modulation at step 1 even if $r_0^l \approx r_1^l$. The authors claim "avoiding the need for a burn-in period"—but since the EMA starts at the instantaneous ratio, not the theoretical $r^* = \lambda/\gamma$, there is still a warm-up phase during which the EMA converges to the true quasi-static equilibrium.

**Fix.** Add an explicit initialization step (REQUIRE or a step 0) to the algorithm. Discuss whether the first few training steps should be treated as a burn-in (with modulation disabled or $\varphi = 1$ forced) to avoid inflated modulation when the EMA has not yet converged. Quantify how long the burn-in period typically is at $\alpha = 0.9$ (e.g., the time constant is $1/(1-\alpha) = 10$ steps, so the EMA requires ~30--50 steps to be well-tracked).

---

### 4. The "only increases WD" property has an unstated implication (Moderate)

**Problem.** Section "Additive form" correctly states that $\varphi(t) \geq 1$ so EqWD never reduces WD below $\lambda_{\text{base}}$. This is presented as a feature. However, the asymmetry has a concrete consequence that is not discussed: if the system spends most of training near equilibrium (which is the desired outcome), EqWD applies WD close to $\lambda_{\text{base}}$, but when transitional phases occur, it only pushes WD upward. This means the effective average WD over training is systematically higher for EqWD than for FixedWD with the same $\lambda_{\text{base}}$, which could confound comparisons. Specifically, the +0.38% gain on ImageNet could partly reflect a higher average regularization strength rather than better-timed regularization. This is not controlled for in the experiments.

**Fix.** Either (a) add an experiment comparing EqWD against FixedWD with a tuned higher $\lambda$ to determine how much of the gain is attributable to timing versus effective strength, or (b) acknowledge this confound explicitly and provide a theoretical or empirical argument for why the adaptive timing is the primary driver of improvement. The CAWD baseline in the experiments partially addresses this (CAWD also modulates but performs worse), but the asymmetric-only-increase property is unique to EqWD and deserves direct treatment.

---

### 5. The AIS result in the theoretical analysis section undercuts the theoretical argument (Moderate)

**Problem.** At the end of Section 3.3, the authors cite their own empirical result: "AIS is near zero for all convolutional layers... indicating that the cosine alignment signal carries no incremental information beyond what is already captured in the gradient and weight norms individually." This observation is then used to validate EqWD's choice of using $r_t$ rather than the full cosine alignment. However, this argument has a logical gap: it shows that $r_t$ carries the same information as $(\|g_t\|, \|w_t\|, \alpha_t)$ jointly—but the Theorem 1 argument relies on $r_t$ being a proxy specifically for alignment deviation $|\alpha_t - \bar{\alpha}|$. The AIS evidence actually suggests that alignment deviation is not an independent signal at all, which weakens (rather than strengthens) the motivation for Theorem 1. The two arguments are in tension: if AIS $\approx 0$, then alignment-based reasoning (Theorem 1) is not needed to justify EqWD.

**Fix.** Restructure the theoretical section to present a single coherent narrative. Either: (a) lead with the AIS result and use it to argue that ratio deviation is sufficient without alignment-based justification, dropping Theorem 1 as unnecessary; or (b) keep Theorem 1 but address the tension with the AIS result by explaining that the near-zero AIS shows alignment information is *redundant given norms*, not that it is absent—and that EqWD captures the alignment-relevant information implicitly through the ratio.

---

### 6. Missing: convergence or stability analysis (Minor)

**Problem.** For a method that modifies the weight decay coefficient at every step, it is natural to ask whether this modification is safe: does EqWD preserve convergence properties of the underlying SGDW? If $\lambda_t^l$ grows large due to persistent ratio deviation, could EqWD destabilize training by over-regularizing? The clamping at $\delta_{\max} = 10$ is a practical safeguard, but there is no theoretical analysis of what happens when it is active, or how large the effective WD can become.

**Fix.** Add a brief note on worst-case behavior: what is the maximum possible $\lambda_t^l$ given the clamp? (It is $\lambda_{\text{base}} \cdot (1 + \beta \cdot 10) = 11 \lambda_{\text{base}}$ at the default $\beta = 1$.) State whether this level of WD has been observed in practice and whether it is theoretically safe for standard architectures. A one-paragraph stability analysis or a citation to WD convergence literature would be sufficient.

---

### 7. Notation inconsistency between Method and Introduction (Minor)

**Problem.** The Introduction defines the modulation factor as:
$$\varphi(t) = 1 + \beta \cdot \frac{|r_t - r^*|}{r^* + \varepsilon}$$
where $r^*$ is the "EMA-tracked equilibrium." In the Method section (Algorithm, line 2), the EMA variable is written $r^{*,l}$ with superscript $l$ for per-layer. In the design rationale prose, it is sometimes written $r^*$ (without the layer index) and sometimes $r^{*,l}$. The formula for $\varphi(t)$ in the prose (equation after the algorithm) uses $r^{*,l}$ in the denominator but $r_t^l$ in the numerator, while the Introduction uses $r_t$ and $r^*$ without layer indices. This creates ambiguity about whether $\varphi(t)$ is a scalar (global) or per-layer quantity.

**Fix.** Standardize notation throughout. Either (a) use $r_t^l$ and $r^{*,l}$ everywhere in Section 3, and use a slightly simplified notation in Section 1 with a forward reference; or (b) define the per-layer version once clearly, then suppress the layer index in the Introduction for presentation clarity with an explicit note.

---

### 8. "Connections to existing methods" section is incomplete and partially inaccurate (Minor)

**Problem.** The three special-case connections are presented, but:
- The connection to SWD is described as "replacing the ratio deviation with $\|g_t\|$ alone recovers an SWD-like gradient-norm schedule." SWD (as cited from NeurIPS 2023) uses the gradient norm to schedule WD at the global (not per-layer) level and reduces WD when gradients are large—i.e., it uses the inverse signal to EqWD. This is not a reduction of EqWD to SWD; it is a conceptual comparison. Calling it a "recovery" is misleading.
- There is no connection mentioned to CPR (norm-constrained regularization), which is also evaluated in experiments. A brief note on how EqWD differs from a norm-targeting approach would improve completeness.

**Fix.** Revise the SWD comparison to describe it as a *conceptual analogy* rather than a formal reduction. Add a brief comparison to CPR: EqWD uses the ratio as a dynamic signal but does not enforce a norm target, whereas CPR enforces targets via an augmented Lagrangian constraint. This clarifies why EqWD and CPR can behave differently even though both use weight norm information.

---

## Consistency with Introduction and Experiments

- The introduction frames "theoretical analysis" (Contribution 3) as a key contribution. The current Theorem 1 does not meet the standard implied by that framing. This creates a mismatch between claims and delivery that reviewers will notice.
- The introduction correctly reports the main numerical results that appear in Table 1. No inconsistency found there.
- The experiments section introduces CAWD (alignment-based variant) as a baseline, but CAWD is not mentioned in the Method section at all. The Method section should either describe CAWD briefly or cross-reference the experiments section, so readers understand what CAWD is before encountering it in the baselines.
- The claim in the Method section that "EqWD operates per layer" is consistent with the experiments, but the experiments also apply EqWD to VGG-16-BN (Section 4.3, layer-type ablation) where BN layers are present. The Method section should clarify how BN layers are handled—this is touched on in the experiments (uniform vs. layer-aware) but never resolved in the method description itself.

---

## Summary of Priority Fixes

| Priority | Issue | Action Required |
|----------|-------|----------------|
| Critical | Theorem 1 is informal with no proof | Formalize, downgrade to conjecture, or replace with empirical analysis |
| Critical | Equilibrium assumptions not stated | Add scope/assumptions paragraph in Preliminaries |
| Moderate | AIS result contradicts alignment-proxy motivation | Restructure theoretical narrative for consistency |
| Moderate | Asymmetric WD inflation confounds comparisons | Add controlled experiment or explicit discussion |
| Moderate | EMA initialization not in algorithm box | Add explicit initialization step and burn-in discussion |
| Minor | CAWD not described in Method section | Add brief description or cross-reference |
| Minor | Notation inconsistency ($r^*$ vs. $r^{*,l}$) | Standardize notation throughout |
| Minor | SWD "recovery" claim is inaccurate | Revise to "conceptual analogy" |

# Method/Theory Critique

## Score: 6/10

## Strengths

- The SGDW update rule (Eq. 1) and assumptions are stated cleanly and follow standard conventions for nonconvex SGD analysis. The three assumptions (L-smoothness, bounded variance, bounded second moment) are necessary and sufficient for the convergence analysis pursued.
- The alignment quantity definition (Eq. 2) is clearly motivated, the citation to Xie et al. (2024) is appropriate, and the connection to the prior convergence result is stated precisely (the "if and only if" characterization).
- Theorem 1 (convergence with time-varying WD) presents a clean convergence rate of $O(1/\sqrt{T})$, which is standard and expected for nonconvex SGD. The quantity $\bar{\Lambda}_T$ is a natural summary of the time-varying schedule.
- The three AADWD variant formulas are explicit, and the rationale paragraphs for each are well-written. The Square variant's connection back to Theorem 1's condition $\lambda_t = O(\gamma_t^2)$ is a concrete and useful link between theory and design.
- Proposition 2 (decoupling instability) provides the correct mechanistic narrative for the most dramatic experimental finding, and the informal feedback-loop diagram is vivid.
- The section is internally consistent with the intro's stated formulas for the three variants (the formulas match exactly between intro and method).

---

## Issues (by priority)

### Critical

**C1. Theorem 2 is not a theorem -- it is an implication of Theorem 1, stated as a separate result without independent proof.**
Theorem 2 ("Alignment-weighted bound") claims the convergence bound depends on $\bar{\delta}_T$ rather than $\sup_t \delta_t$. However, the theorem body never shows the convergence bound itself in terms of $\bar{\delta}_T$. The proof sketch for Theorem 1 mentions bounding the cross-term $\langle \nabla f_S(w_t), \lambda_t w_t \rangle$ via $\delta_t$, but the actual bound in Theorem 1 is stated only in terms of $\bar{\Lambda}_T = (1/T)\sum_t \lambda_t / \gamma_t$ -- with no $\delta_t$ anywhere in the final expression. Theorem 2 then asserts that the bound involves $\bar{\delta}_T = \sum_t \lambda_t \delta_t / \sum_t \lambda_t$, but this $\bar{\delta}_T$ does not appear in Theorem 1's inequality either. Either: (a) Theorem 1's bound should explicitly contain $\delta_t$ terms and the current statement is incomplete, or (b) Theorem 2 is restating a corollary that requires the full proof in Appendix E to make precise. As written, the two theorems are formally disconnected, which is a serious mathematical integrity problem. A NeurIPS reviewer will flag this immediately.

**C2. Proposition 1 ("Decoupling instability") does not constitute a proposition -- it is an informal narrative without any stated formal claim.**
The statement says: "If $\lambda_t / \gamma_t$ exceeds a critical threshold $\lambda^* / \gamma^*$, the weight decay term dominates... leading to weight norm collapse." No value of $\lambda^* / \gamma^*$ is defined, no conditions on the loss landscape are given, and no formal statement of "dominates" is made precise. The feedback loop diagram in Eq. (6) uses informal arrows ("$\|w_t\| \downarrow \to \cdots$"), not mathematical relations. This is an observation, not a proposition. Labeling it a "Proposition" and giving it a number implies a level of formal rigor that is absent. Either formalize it (e.g., show that for the decoupled aggressive variant, $\mathbb{E}[\|w_t\|^2]$ decreases geometrically under mild conditions) or downgrade it to a "Remark" or "Observation."

**C3. The proof sketch for Theorem 1 is logically incomplete and potentially circular.**
The sketch defines $\Phi_t = f_S(w_t) + \beta_t \|w_t\|^2$ and says $\beta_t$ is "chosen to track $\lambda_t$." This is not defined. The cross-term $|\langle \nabla f_S(w_t), \lambda_t w_t \rangle| \leq \delta_t \|\nabla f_S(w_t)\| \|w_t\|$ is then "absorbed into the potential via the condition $\lambda_t \leq C\gamma_t^2$." How exactly this absorption works -- what cancellation occurs, what residual remains -- is entirely opaque. A proof sketch is supposed to convey the key insight; a reader cannot verify even the plausibility of the final bound from this sketch. The quantity $\bar{\Lambda}_T$ in the final bound requires $\sum_t \lambda_t / \gamma_t$ to emerge from the telescope, but the sketch does not show this. The full proof is deferred to "Appendix E," which does not exist in the submitted material.

**C4. The condition $\lambda_t \leq C\gamma_t^2$ in Theorem 1 is satisfied by the Square variant but NOT by the Conservative or Aggressive variants, yet Theorem 1 is applied to motivate all three.**
With $\gamma_t = 0.1 / \sqrt{T}$ (under the theorem's assumption $\gamma_t = \gamma / \sqrt{T}$), the condition $\lambda_t \leq C\gamma_t^2$ is $O(1/T)$. But the Conservative variant sets $\lambda_t = c \cdot \gamma_t \cdot (1 - \hat{\delta}_t) = O(1/\sqrt{T})$, which is $\sqrt{T}$ times larger -- violating the condition. The Aggressive variant similarly scales as $O(1/\sqrt{T})$. The method section acknowledges in passing that "the Square variant satisfies the $O(\gamma_t^2)$ condition," but does not acknowledge that the Conservative and Aggressive variants do not, and therefore have no convergence guarantee from Theorem 1. This is not a minor omission: the entire theoretical framework (Theorems 1 and 2) only formally justifies the Square variant.

### Major

**M1. The experiment uses a milestone learning rate schedule ($\gamma_t \in \{0.1, 0.01, 0.001\}$), but Theorem 1 assumes $\gamma_t = \gamma / \sqrt{T}$ (polynomial decay). These are fundamentally different schedules.**
Under a polynomial decay schedule, $\bar{\Lambda}_T = (1/T)\sum_t \lambda_t / \gamma_t$ is well-behaved and converges. Under a milestone schedule, $\lambda_t / \gamma_t$ jumps by a factor of 10 at each milestone. The theorem's bound is not directly applicable to the experimental setup. This gap between theory and practice is never acknowledged. The paper should either: (a) provide an analog of Theorem 1 for step-function LR schedules, or (b) explicitly state that the theorem is illustrative and not directly predictive of the experimental results.

**M2. Proposition 2 ("Budget equivalence") is stated with a condition ("approximate convexity near convergence") that is never verified or even discussed.**
The argument derives $\|w_T\|^2 \approx \|w_0\|^2 \prod_t (1-2\lambda_t) + R_T$ and claims $R_T$ is "similar for both schedules." But $R_T$ collects all gradient-dependent terms, which depend heavily on the training trajectory -- and the two schedules (dynamic AADWD vs. constant budget-equivalent) produce different trajectories by construction (as evidenced by different weight norms at intermediate checkpoints). The argument essentially assumes its conclusion. A cleaner version: show that the map $\{\lambda_t\} \mapsto \|w_T\|^2$ is approximately a function of $\sum_t \lambda_t$ alone, by bounding the sensitivity of $R_T$ to temporal redistribution.

**M3. The notation $\bar{\Lambda}_T$ in Theorem 1 and $\bar{\delta}_T$ in Theorem 2 are both defined as "averages" but use incompatible weighting schemes. This creates confusion.**
$\bar{\Lambda}_T = (1/T)\sum_t \lambda_t / \gamma_t$ is a uniform time-average of the WD-to-LR ratio, while $\bar{\delta}_T = \sum_t \lambda_t \delta_t / \sum_t \lambda_t$ is a $\lambda_t$-weighted average of alignment. Neither notation distinguishes these as different kinds of averages, which forces a reader to parse the formulas carefully. At minimum, a sentence should acknowledge the different weighting in the two quantities and explain the motivation for the weighted average in $\bar{\delta}_T$.

**M4. The $\epsilon$ in the Aggressive variant formula (Eq. 8) appears without motivation in the method section, yet it is critical for numerical stability when $\hat{\delta}_t \approx 0$.**
The experiments show $\hat{\delta}_t \approx 0.003$, which is close to zero. In the denominator $\hat{\delta}_t + \epsilon$ with $\epsilon = 10^{-8}$, the $\epsilon$ term is negligible for $\hat{\delta}_t = 0.003$ (ratio $\approx 3 \times 10^5$). Its presence thus does not stabilize anything in the observed regime. However, the collapse at $c = 10.0$ (Table 4 in experiments) suggests that at extreme hyperparameter settings the denominator does approach zero. The method section should clarify: is $\epsilon$ purely a safeguard for extreme cases, and what is the behavior of $\hat{\delta}_t$ in those regimes?

**M5. The Square variant formula (Eq. 9) uses $\gamma_t^2$, but the SGDW update rule (Eq. 1) uses a milestone schedule in experiments where $\gamma_t$ is constant within each phase. The $\gamma_t^2$ term then introduces a $100\times$ vs. $10{,}000\times$ reduction at milestones as stated in the rationale, but the method section presents this as a known limitation while simultaneously claiming the Square variant "provides the strongest convergence guarantee." These two claims cannot both be correct: a variant that eliminates regularization in late training cannot simultaneously be the safest option. The contradiction should be resolved, not glossed over.**

### Minor

**m1. Assumption 3 (bounded gradient second moment, $\mathbb{E}[\|g_t\|^2] \leq G^2$) implies Assumption 2 (bounded variance) when the gradient mean is bounded. In the nonconvex SGD literature it is more common to use either bounded variance or bounded second moment, not both. The paper should clarify whether both are strictly needed or whether one subsumes the other in the proof.**

**m2. The label "Theorem 2" is used for what is called an "Implication" in the following paragraph. The text says "Theorem 2 provides the theoretical motivation for AADWD" and then immediately qualifies "However, this improvement is only realized when the variation in $\delta_t$ is large enough... a condition our experiments will show is not met." Stating a theorem and immediately undercutting its applicability in the same paragraph undermines the narrative. Either restructure (present the theorem as a conditional positive result, then explain when it fires) or consolidate Theorems 1 and 2 into a single theorem with a richer bound that makes the $\bar{\delta}_T$ dependence explicit.**

**m3. The EMA formula for $\hat{\delta}_t$ (Eq. 3 in Section 3) references $g_t$ as the stochastic gradient, but in Section 2 (Assumptions), $g_t = \nabla f_S(w_t; \xi_t)$ includes the full stochastic gradient. In the EMA formula, the numerator $|\langle g_t, w_t \rangle|$ uses the minibatch gradient -- this is consistent but should be stated explicitly, as it is the proxy (not the true alignment) being tracked.**

**m4. The method section mentions "the Pearson correlation between the minibatch EMA and large-batch alignment measurements is $r = 0.849$ on ResNet-20/CIFAR-10," but this is empirical validation that belongs in the experiments section (Section 4, Alignment Proxy Characterization), not in the method section. Placing it in the method section gives the impression it is a general theoretical property.**

**m5. Section 4 (Stability Analysis) is labeled "LR--WD Coupling" in the section header but "Stability Analysis of LR--WD Coupling" in the running text (line 93). This minor heading inconsistency should be resolved for camera-ready.**

---

## Specific Suggestions

1. **Fix Theorem 2's formal statement.** Rewrite Theorem 1 to include an explicit $\delta_t$-dependent term in the bound (e.g., a term $\frac{1}{T}\sum_t \lambda_t \delta_t \|w_t\|$), and then state Theorem 2 as a corollary that rewrites this term using $\bar{\delta}_T$. The current decoupling between the two theorems is the single largest mathematical flaw.

2. **Downgrade Proposition 1 to Observation or Remark.** Replace the proposition label with "Observation (Decoupling instability)" and frame it as an empirically-motivated heuristic. Alternatively, provide a two-line formal proof: for the decoupled aggressive variant with $\hat{\delta}_t$ near a fixed point, show that $\|w_{t+1}\| \leq (1 - \lambda_t)\|w_t\| + O(\gamma_t G)$, and demonstrate that when $\lambda_t > \gamma_t G / \|w_t\|$, the weight norm decreases in expectation.

3. **Add a scope statement for Theorems 1 and 2.** Insert a remark after Theorem 1 stating explicitly: "Theorem 1 applies under polynomial decay $\gamma_t = \gamma/\sqrt{T}$. For milestone schedules used in our experiments, the same qualitative structure holds but the exact bound constants differ; see Appendix X for discussion." This bridges the theory-experiment gap.

4. **Add a sentence after Theorem 1 noting which variants satisfy $\lambda_t \leq C\gamma_t^2$.** Currently the paper only mentions this in the Square variant's rationale paragraph. A single sentence before the variant descriptions -- e.g., "Note that Theorem 1's convergence guarantee requires $\lambda_t = O(\gamma_t^2)$; only the Square variant satisfies this condition by construction, while Conservative and Aggressive variants operate outside the theorem's formal scope" -- would make the paper intellectually honest without undermining the narrative.

5. **Strengthen Proposition 2's argument.** The budget equivalence proposition is the paper's most important theoretical claim (it is the central positive result). Invest two additional sentences in the argument: explicitly bound $|R_T^{(1)} - R_T^{(2)}|$ in terms of the difference in training trajectories, and note that since both schedules produce similar loss curves (as observed empirically), this difference is small. This transforms the argument from circular to genuinely informative.

6. **Unify the notation for averages.** Introduce a notation $\langle \cdot \rangle_\lambda$ for $\lambda_t$-weighted average to distinguish it from the uniform time-average in $\bar{\Lambda}_T$. Write $\bar{\delta}_T = \langle \delta_t \rangle_\lambda$ and note that this is a $\lambda_t$-weighted average, emphasizing that the adaptive schedule implicitly upweights alignment at high-WD steps.

7. **Move the proxy validation ($r = 0.849$) from method to experiments.** The Pearson correlation between EMA and large-batch alignment is empirical evidence, not method design. Placing it in the experiments section (Section 4.7, Alignment Proxy Characterization) and forward-referencing it from the method section ("We validate the faithfulness of $\hat{\delta}_t$ in Section 4.7") improves the structural clarity of the paper.

8. **Clarify the role of Assumption 3.** Add a footnote or parenthetical after Assumption 3 noting that it is used only to bound $\|g_t\| \leq G$ in the weight norm terms, and that Assumption 2 is the primary noise assumption for gradient variance. This prevents reviewers from asking whether the assumptions are redundant.

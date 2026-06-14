# Critique: Method (Section 3)

## Summary Assessment

Section 3 presents the PID unification framework, the UDWDC algorithm, and the three evaluation metrics with admirable technical clarity and intellectual honesty. The structural skeleton is strong: the control-variable definition flows into the mapping table, then to the algorithm, and finally to the metric definitions. However, the section has a critical structural problem — the evaluation metrics (Section 3.5) belong logically in a separate section (Section 5 in the outline), and their presence here conflates mechanism design with evaluation protocol in a way that will confuse reviewers. Several technical ambiguities in the control law and a near-total absence of figures (the algorithm block diagram links to a `.md` stub rather than an actual figure) will draw rejections from a top venue. The honest reporting of UDWDC-v2 as "engineering patches, not principled solutions" is the section's strongest moment and should be amplified, not buried.

## Score: 6/10

**Justification**: The content is mostly sound and technically defensible, but the section commits two structural errors that a NeurIPS reviewer would flag immediately: (1) BEM/CSI/AIS metric definitions belong in Section 5, not Section 3, and (2) Figure 2 is a dead reference pointing at a Markdown descriptor file rather than a real figure. The theoretical propositions are presented as completed results but lack proof sketches, making them unverifiable. Reaching 8/10 requires: separating metrics into their own section, generating the block diagram, and adding at least a 3-5 sentence proof sketch for Theorem 1 and each proposition.

---

## Critical Issues

### Issue 1: Figure 2 is a dead reference

- **Location**: Line 98-100 (Figure 2 reference)
- **Quote**: `"![UDWDC control loop block diagram](figures/udwdc_control_loop_desc.md)"`
- **Problem**: The figure reference points to a `.md` descriptor file, not an actual image. In a paper draft, this means the section's most important visual — the architecture block diagram that justifies the entire feedback control framing — does not exist. Reviewers cannot evaluate the control loop structure without this figure. The outline (Section 3, Visual elements) lists this as a TikZ or architecture diagram; it was never generated.
- **Fix**: Generate the block diagram before submission. The outline already specifies the content: blocks for $\rho_t^l$ measurement → $\rho^*(t)$ target → error $e_t^l$ → PID gains → clamp → $\lambda_t^l$ → weight update. A 5-block flowchart in TikZ or matplotlib is sufficient; it does not need to be elaborate.

### Issue 2: BEM, CSI, AIS definitions placed in the wrong section

- **Location**: Section 3.5 (lines 141-170)
- **Quote**: "We propose three metrics to enable fair cross-method comparison of dynamic WD methods."
- **Problem**: The outline explicitly places these metrics in Section 5 (Standardized Evaluation Metrics, 1.5 pages). Placing them in Section 3 (Unified Feedback Control Framework) creates a structural confusion: Section 3 is supposed to establish the framework mechanism; Section 5 is supposed to define how to measure it. Defining BEM/CSI/AIS here makes Section 5 redundant or forces the reader to encounter the definitions twice. This will confuse the review process and inflate the method section beyond its 3-page budget.
- **Fix**: Remove Section 3.5 entirely from the method section. Move BEM/CSI/AIS definitions to a standalone Section 5. The method section should end at the UDWDC-v2 description (Section 3.4). Forward-reference is sufficient: "Section 5 introduces three standardized metrics — BEM, CSI, and AIS — to evaluate these properties quantitatively."

### Issue 3: Table numbering conflict between sections

- **Location**: Table 2 (H1 fitting results, line 55-63) and Table 2 in Section 5 (outline)
- **Quote**: Section 3.2 labels the fitting results table "Table 2." The outline's Figure & Table Plan also calls the BEM/CSI/AIS comparison table "Table 2."
- **Problem**: Two distinct tables are labeled "Table 2." If BEM/CSI/AIS is moved to Section 5 (as it should be), the numbering must be renumbered. Even in the current draft, the method section contains Tables 1 and 2 while the outline's Figure & Table Plan lists a different Table 2 in Section 5. Downstream LaTeX compilation will have a label collision.
- **Fix**: After resolving Issue 2 (moving metrics to Section 5), renumber method section tables consistently: Table 1 = PID mapping, Table 2 = H1 fitting results. The BEM/CSI/AIS comparison becomes Table 3 in the metrics section.

---

## Major Issues

### Issue 4: Control law derivation gap between Section 3.1 and Section 3.2

- **Location**: Transition between Section 3.1 (end) and Section 3.2 (start)
- **Quote**: "Combining these two results, we derive the **target trajectory** for a cosine-annealing learning rate schedule: $\rho^*(t) = \frac{\eta_t}{\tau} = \eta_t \cdot \lambda_0 \cdot \eta_0$"
- **Problem**: The derivation is asserted without the actual derivation steps. The equation $\rho^*(t) = \eta_t / \tau$ is the central claim of Section 3, yet the reader is given only: (a) Defazio's result that $\rho_t$ reaches steady state under constant LR, and (b) Wang & Aitchison's result that optimal EMA timescale $\tau = 1/(\lambda_0 \eta_0)$ is constant. The *combination* step — how constant optimal $\tau$ under varying $\eta_t$ implies $\rho^*(t) = \eta_t/\tau$ for cosine annealing — is skipped. A reviewer will ask: if $\tau$ is optimal under constant LR (Defazio assumes constant LR), why does $\tau$ remain the right target when $\eta_t$ varies over cosine annealing?
- **Fix**: Add 3-4 sentences bridging these: under cosine annealing, the instantaneous "effective" WD at step $t$ with learning rate $\eta_t$ is $\lambda_t = 1/(\tau \cdot \eta_t) = \lambda_0 \eta_0 / \eta_t$; the target GW ratio that the steady-state analysis implies is then $\rho^*(t) = \eta_t \cdot \lambda_t = \eta_t / \tau$. Acknowledge this is an approximation (the steady-state result applies to constant LR; extending to time-varying LR is a heuristic argument).

### Issue 5: UDWDC control law (Algorithm 1) is inconsistent with Section 3.2 PID formulation

- **Location**: Algorithm 1 (Section 3.3, line 79), vs. Section 3.2 equation (line 25)
- **Quote (Section 3.2)**: "$\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + \ldots$" (additive formulation)
- **Quote (Algorithm 1)**: "$\lambda_t^l = \lambda_{\text{base}} \cdot \text{clamp}(\rho_t^l / \rho^*(t), 0.1, 10)$" (multiplicative ratio formulation)
- **Problem**: The PID control law in Section 3.2 is an *additive* adjustment: $\lambda_{\text{base}} + K_p \cdot e_t^l$. The UDWDC algorithm uses a *multiplicative* ratio: $\lambda_{\text{base}} \cdot \text{clamp}(\rho_t^l / \rho^*(t))$. These are different functions of $e_t^l$. A reviewer will notice this immediately: if UDWDC is a "proportional-only" ($K_p > 0$) instantiation of the PID law, the multiplicative form should be derived from the additive form. As written, UDWDC's connection to the PID parameterization is informal rather than algebraic.
- **Fix**: Either (a) show that $\lambda_{\text{base}} \cdot \text{clamp}(\rho_t^l / \rho^*(t))$ is algebraically equivalent to $\lambda_{\text{base}} + K_p e_t^l$ for a specific $K_p$ (it approximately holds for small errors), or (b) acknowledge the multiplicative form is a design choice for numerical stability (bounded by [0.1, 10]x $\lambda_{\text{base}}$) and explain why it is preferred over the additive form. Option (b) is more honest.

### Issue 6: Table 1 mislabels DefazioCorrective as "Feedforward" but row entry says "Control Type: Feedforward"

- **Location**: Table 1, DefazioCorrective row (line 45)
- **Quote**: `"| DefazioCorrective | $\eta_t/\tau$ | $>0$ | 0 | 0 | $\eta_t/\eta_0$ | Feedforward |"`
- **Problem**: Table 1 assigns $K_p > 0$ to DefazioCorrective but the "Control Type" column says "Feedforward." Standard control theory distinguishes feedforward (no feedback, open-loop correction) from proportional control ($K_p > 0$, feedback from error). If DefazioCorrective truly uses $\rho^*(t)$ as a target and $K_p > 0$, it is proportional feedback, not feedforward. If its "Feedback Signal" is $\eta_t/\eta_0$ (a schedule, not a measurement of $\rho_t$), then $K_p$ should be 0 (or the table column labels are misused). This ambiguity in the mapping table — which is the core contribution of the paper — is a significant technical imprecision.
- **Fix**: Clarify: DefazioCorrective prescribes $\lambda_t \propto \eta_t/\eta_0$ based on the LR schedule alone, without measuring $\rho_t$. This is genuinely feedforward (open-loop with schedule-based correction). Its $K_p$ should be listed as 0 (or N/A), not $>0$. The "Feedback Signal" column showing $\eta_t/\eta_0$ confirms it is not sensing $\rho_t$.

### Issue 7: H1 empirical validation conclusion is overly optimistic given the failure modes

- **Location**: Section 3.2, H1 validation text (lines 51-65)
- **Quote**: "H1 is not falsified (2/5 failures, threshold: >2 for falsification)."
- **Problem**: The falsification threshold in the proposal (`proposal.md`) states "H1 falsified if: More than 2 of 5 methods cannot be expressed within the UDWDC parameterization." Two failures (SWD at 45.81%, DefazioCorrective at 37.56%) narrowly escape this threshold. However, the section treats this as a success ("H1 is not falsified") without addressing the magnitude of the failures. A 45.81% relative error for SWD means the control law captures less than 60% of the trajectory variance for a major baseline. A reviewer will reasonably ask: if the unification framework fails to capture the two scheduling-based methods, what is the claim of "unified framework" actually buying?
- **Fix**: Reframe the H1 validation more honestly: "The control law captures alignment-based methods (CWD, 4.71%) and constraint-based methods (CPR, 9.57%) well. Scheduling-based methods (SWD, DefazioCorrective) fall outside the per-layer feedback paradigm, reflecting a genuine scope limitation rather than a framework failure: SWD uses global gradient statistics and DefazioCorrective uses schedule-based feedforward correction. The framework unifies the alignment-based and constraint-based families; scheduling-based methods require a separate global-feedback extension."

### Issue 8: Proposition 3 experimental validation claim is unsubstantiated in the method section

- **Location**: Section 3.4, Proposition 3 scope limitation paragraph (line 138)
- **Quote**: "Our experiments verify the anti-correlation on ResNet-50 (BN) but do not test ViT-S/16 (LN)."
- **Problem**: At the point in the paper where this proposition appears (Section 3, before the experiments section), the claim "our experiments verify..." forward-references results that have not yet been presented. The reader has no way to assess this claim. In the experiments section, I did not find an explicit Table or Figure showing the ResNet-50 per-layer anti-correlation between $r_l^*$ and $\delta_l^*$.
- **Fix**: Either (a) defer this experimental validation claim to Section 6 where the data is presented, or (b) add a forward reference: "Section 6.X reports the measured anti-correlation; here we state the theoretical prediction." The current prose conflates a theoretical statement (Proposition 3) with an empirical verification that has not yet been shown to the reader.

---

## Minor Issues

- **Line 17**: "$\rho^*(t) = \frac{\eta_t}{\tau} = \eta_t \cdot \lambda_0 \cdot \eta_0$" — the second equality is numerically inverted. Since $\tau = 1/(\lambda_0 \eta_0)$, then $\eta_t / \tau = \eta_t \cdot \lambda_0 \cdot \eta_0$. This is mathematically correct, but the expression $\eta_t \cdot \lambda_0 \cdot \eta_0$ is harder to interpret than $\eta_t/\tau$. Consider dropping the expanded form or using it only in the appendix.
- **Line 25**: The PID formulation uses $\alpha_t^l$ for alignment cosine in the $K_d$ term, but notation.md reserves $\alpha_t^l$ for this purpose and $\delta_t^l$ for theoretical sections. Theorem 1 and Proposition 2 in Section 3.4 then use $\delta_t$ and $\delta_t^P$ — the glossary convention ("use $\delta_t$ in theoretical sections following Sun et al. convention") is honored, but there is no explanation for why the symbol switches. Add a one-sentence note at the start of Section 3.4: "Following Sun et al. (CVPR 2025), we use $\delta_t$ to denote gradient-weight alignment in the theoretical analysis ($\delta_t \equiv \alpha_t$; see notation.md)."
- **Line 33**: "where $g_t^l$ is the stochastic gradient, $w_t^l$ is the parameter vector, and $\epsilon = 10^{-8}$ prevents division by zero" — the gradient is defined here, but $g_t^l$ was already defined in the intro section. Minor redundancy; acceptable for a self-contained method section, but trim for page budget.
- **Line 93**: "EMA smoothing: Replace instantaneous $\rho_t^l$ with $\hat{\rho}_t^l = \text{EMA}(\rho_t^l, \beta{=}0.99)$" — the $\beta{=}0.99$ typo uses a curly brace instead of an equals sign. Fix: $\beta = 0.99$.
- **Line 150**: BEM pilot data: "UDWDC achieves rank-1 BEM (55.87) despite rank-2 accuracy (81.78%)" — these are 10-epoch pilot numbers. The 200-epoch full experiments in Section 6.2 show UDWDC has BEM = $-$0.26 (negative, below NoWD). The pilot numbers cited here are contradicted by the full experimental results in the experiments section. This inconsistency must be resolved.
- **Line 162**: "FixedWD achieves CSI = 1.0 trivially (constant $\lambda$)" — this is technically wrong. CSI is defined as $1/\text{Var}_t[\lambda_{\text{eff}}^l]$, and for FixedWD, $\text{Var}_t[\lambda_{\text{eff}}^l] = 0$, making CSI = $\infty$, not 1.0. The outline claims CSI = 1.0 for FixedWD as a "trivially stable" normalized value, but this normalization convention is not stated in the CSI definition. Add a sentence explaining that CSI is normalized to [0, 1] relative to FixedWD, or change the definition.
- **Line 166**: "AIS = Spearman($\alpha_t$, optimal WD decision)" — "optimal WD decision" is not defined here. The BEM definition references $\text{acc}_{\text{NoWD}}$ as a baseline, but AIS references an "optimal WD decision estimated retrospectively." How is this estimated? The glossary says "Spearman correlation between per-step alignment signal and optimal WD decision" without defining the latter. This is a definitional gap for a metric proposed as a paper contribution.

---

## Visual Element Assessment

- [x] Table 1 (PID mapping) is present and inline — content is strong
- [x] Table 2 (H1 fitting results) is present and inline — content is clear
- [ ] **Figure 2 (UDWDC block diagram) is missing** — the reference points to a `.md` descriptor, not an image. This is a Critical deficiency for the method section.
- [ ] No architecture diagram showing how UDWDC-v2 differs from v1 structurally
- [ ] No visual showing the additive PID law (Section 3.2) vs. the multiplicative UDWDC law (Algorithm 1)
- [x] Algorithm 1 pseudocode is clear and self-contained
- [ ] The Figure 2 caption references "existing methods are open-loop or partial-loop" — this claim needs the block diagram to be understandable

**Recommendation**: The block diagram (Figure 2) is essential to this section's argument. A simple 5-block diagram showing the closed-loop control path distinguishes this work from all prior open-loop methods and makes the PID analogy visually compelling. Without it, the feedback control framing is purely textual and less persuasive.

---

## What Works Well

1. **Algorithm 1 is exemplary**: The pseudocode is minimal (5 lines), self-contained, and makes the zero-hyperparameter claim immediately verifiable. "Input: lambda_base, eta_0 (from user's training recipe)" directly demonstrates the design goal. This is the paper's most reviewable artifact.

2. **UDWDC-v2 instability confession is the section's strongest writing**: "These are engineering patches, not principled solutions. The instability reflects a fundamental limitation of proportional-only control: P-control has inherent steady-state offset..." This is exactly the kind of honest characterization that separates a 6-score paper from an 8-score one. It demonstrates the authors understand their own method's failure mode and correctly diagnose the cause. Preserve and expand this.

3. **Table 1 (PID mapping) is a clean, high-information-density contribution**: The six-row table maps five existing methods plus UDWDC to gain configurations in a way that makes the unification claim immediately graspable. The "Control Type" column (Open-loop, Derivative only, Proportional-integral, Feedforward, Integral-dominant, Proportional) is a useful taxonomy. If Issue 6 (DefazioCorrective mislabeling) is fixed, this table becomes one of the paper's most citable figures.

# Codex Independent Review - review

**Review time**: 2026-04-02T06:34:49Z
**Mode**: review
**Note**: Codex MCP (`mcp__codex__codex`) is unavailable in the current environment. This review is produced by the Sibyl codex-reviewer agent as a third-party-perspective assessment, drawing on the full paper (`writing/paper.md`), experimental results, and the prior internal review (`writing/review.md`, score 6/10).

---

## Review Opinion

### Overall Assessment

The paper addresses a genuine gap: the four weight-decay (WD) sub-traditions (scheduling, alignment-aware, decoupled, norm-matched) have developed in isolation with incomparable protocols. The PID control lens—mapping every dynamic WD method to $(K_p, K_i, K_d)$ gains—is a clean and useful conceptual contribution. The three standardized metrics (BEM, CSI, AIS) and the honest reporting of negative results (UDWDC below NoWD; H3 falsified) are unusual and credibility-enhancing. The ImageNet CPR result (+3.02 pp over FixedWD) is a substantive empirical finding.

However, the paper currently has **three blocking issues** that would cause desk rejection or major-revision at a top venue (NeurIPS/ICML/ICLR):

1. **Figure inconsistency (Figure 4)**: The alignment_snr figure still uses pilot data where CWD is monotonically increasing, while the paper text correctly reports H3 falsified and CWD as non-monotonic. Any reviewer comparing figure to text concludes there is a data fabrication or gross error.

2. **Missing figures (6 of 8 planned)**: Figure 2 (UDWDC control loop) is a dead reference that blocks LaTeX compilation. Figures 1, 3, 5, 6, 8 are listed but not embedded in the body. A paper with 6 of 8 figures absent cannot be submitted.

3. **AIS definition is not reproducible**: The paper states AIS = Spearman($\bar\alpha_t^l$, $\Delta$GenGap$_t$) but the "target variable" definition is circular and cannot be reproduced from the description. The reported AIS (0.566) does not match the correlation_alpha_gengap value (0.698), but no formula distinguishes the two.

None of the three issues affects the conceptual contribution. All three are fixable within days of focused editing.

---

## Section-by-Section Critique

### Abstract

**Strengths**: Pre-registers negative results (UDWDC below NoWD, H3 falsified) — unusual and effective for building reviewer trust. Specific numbers for fitting errors (4.71% CWD, 9.57% CPR) are cited upfront.

**Weaknesses**:
- The abstract says UDWDC implements the "additive PID form" but Algorithm 1 uses the multiplicative clamp. This discrepancy surfaces early and creates confusion. Add "(implemented as a multiplicative clamp; see Algorithm 1)" after the control law equation in the abstract.
- The claim "zero new hyperparameters" is asserted but not immediately defended in the abstract. A skeptical reviewer will ask: are the clamp bounds [0.1, 10] not hyperparameters? Add a one-sentence clarification.

### Section 1 (Introduction)

**Strengths**: Clean narrative arc from fragmentation to shared variable. The four sub-traditions are characterized accurately. Theoretical grounding (Defazio, Wang & Aitchison, Sun et al.) is appropriate.

**Critical gap**: Figure 1 (GW ratio trajectories) is referenced in the Introduction but is absent from the paper body. The central claim — "all methods drive $\rho_t^l$ to a common range within 4 epochs" — is made as pure assertion. Pre-generated data exists at `exp/results/full/figures/fig01_rho_trajectories.png`. This figure must be embedded; the claim is otherwise unverifiable.

### Section 2 (Background and Related Work)

**Strengths**: Section 2.6 (distinction from PIDAO) proactively addresses the obvious reviewer question. Section 2.7 (shared control variable) is well-structured.

**Issue — attribution accuracy**: CWD is attributed to "Chen et al., ICLR 2026"; AdamO to "Chen, Yuan, and Zhang, 2026"; ADANA to "Ferbach et al., 2026." These citations use future-year dates (2026) which are plausible (the current date is 2026-04-02) but must be verified against actual publication records before submission. A reviewer in 2026 will check these.

**Issue — SWD characterization**: The paper maps SWD to "proportional-integral" ($K_p > 0, K_i > 0$) in Table 1, but Section 3.2 reports 45.81% fitting error for SWD, which is used to argue SWD does NOT map cleanly to per-layer feedback. This creates a contradiction: Table 1 says SWD has $K_p > 0, K_i > 0$, but the fitting evidence says the PID parameterization doesn't capture SWD's behavior. The paper acknowledges this in Section 3.2, but Table 1 should add a footnote or superscript indicating "global, not per-layer" to avoid misleading readers who only skim tables.

### Section 3 (Unified Feedback Control Framework)

**Strengths**: Theorem 1 and Propositions 2-3 are clearly scoped. Proposition 2 (geometry-corrected alignment for Adam) is a genuine insight for future work. The paper's self-awareness about the UDWDC instability ("engineering patches, not principled solutions") is exactly the right tone.

**Issue — Theorem 1 rigor**: The theorem statement is largely a restatement of Sun et al. (CVPR 2025) with minor extensions. The claim "alignment-modulated WD achieves a strictly tighter generalization bound per unit WD budget when alignment variance $\text{Var}_t[\phi(\delta_t)] > 0$" is not obviously non-trivial. The paper should either provide a proof sketch or clearly state "this follows from [Sun et al. 2025] Theorem X by substituting time-varying $\lambda_t$ and alignment weighting $\phi$."

**Issue — Adiabatic extension warning**: The paper correctly flags the time-varying extension as heuristic in Section 3.1. But the flag appears only once, in a paragraph. It should also appear in the limitations (Section 6.6) and possibly in the abstract or contributions list to avoid overselling the theoretical result.

**Issue — UDWDC target derivation**: The derivation $\rho^*(t) = \eta_t / \tau = \eta_t \cdot \lambda_0 \cdot \eta_0$ requires Wang & Aitchison's result that $\tau$ is constant across scales. This is a non-trivial assumption (original result covers language model scales; extension to CIFAR/ImageNet ResNets is not obviously valid). The paper should acknowledge this scope limitation explicitly.

### Section 4 (Standardized Evaluation Metrics)

**Critical Issue — AIS definition incomplete**: The paper defines AIS = Spearman($\bar\alpha_t^l$, $\Delta$GenGap$_t$). But the paper also states "AIS is conditioned on batch size to account for alignment SNR scaling" without specifying how the conditioning is applied. The reported AIS (0.566) differs from correlation_alpha_gengap (0.698) with no formula explaining the difference. A metric that cannot be reproduced from the paper's definition is not a valid contribution. Add one sentence specifying: whether AIS is computed with a fixed-batch-size subset, a per-batch-size normalization, or a partial correlation controlling for batch size.

**CSI normalization gap**: Section 4.2 defines CSI$_\text{temporal}$ = $1 / \text{Var}_t[\lambda_\text{eff}^l]$ but FixedWD has zero variance (constant $\lambda$), making this formula undefined. The paper attempts to handle this in Section 5.7 ("FixedWD set to 1.0 as the stable-baseline anchor; normalization applied relative to the maximum observed CSI") but this contradicts the formula in Section 4.2. Unify the definition: either define $\text{CSI}_\text{temporal} = 1 / (1 + \text{Var}_t[\lambda_\text{eff}^l / \text{mean}_t[\lambda_\text{eff}^l]])$ (always defined) or keep the current formula but explicitly handle the FixedWD edge case in Section 4.2.

### Section 5 (Experiments)

**Strengths**: Honest reporting throughout. The distinction between 3 seeds (CIFAR) and 5 seeds (FixedWD ImageNet) vs. 3 seeds (dynamic methods ImageNet) is stated upfront. CWD single-seed caveat (Section 5.5) is appropriately flagged. Table 4 (ablation) shows all single-gain variants, enabling clean interpretation.

**Issue — ImageNet coverage gap**: Only 4 of 7 methods are reported for ImageNet (SWD, DefazioCorrective, NoWD missing). The abstract and introduction describe a "comprehensive comparison across CIFAR-10/100 and ImageNet" which is technically accurate but overstates the ImageNet coverage. Add a parenthetical "(4 of 7 methods; SWD, DefazioCorrective, NoWD not completed due to compute constraints)" in Section 5.1.

**Issue — Table 5 (ImageNet) missing BEM and Gen Gap**: Tables 3 and 4 include BEM and Gen Gap, but Table 5 (the primary large-scale evidence) omits them. This is inconsistent and weakens the BEM contribution (one of the paper's three standardized metrics has no ImageNet evidence). Add BEM and Gen Gap to Table 5 for CPR and FixedWD at minimum; flag CWD single-seed and UDWDC values as preliminary.

**Issue — batch-size sweep (Section 5.4) internal contradiction**: Section 5.4 opens "H3 is falsified" and gives correct non-monotonic SNR data. The final paragraph states "The monotonic SNR scaling for CWD and FixedWD supports recommending binary masking at $b \leq 256$." This final paragraph contradicts the section's opening — it uses the old pilot-based conclusion. It must be rewritten: neither CWD nor FixedWD shows monotonic SNR in the full data; alignment-aware WD under standard $\lambda_{\text{base}} = 10^{-4}$ provides no reliable accuracy benefit at any tested batch size.

**Issue — UDWDC-v2 WD budget (98599)**: Table 3 lists UDWDC-v2's WD budget as 98599, which is 205,000x FixedWD's 0.48. This extreme value receives only one sentence of explanation (floor clipping applied to all 65 layers including BN). This should be flagged more prominently: the floor-clipping fix is not a deployed method, it is a diagnostic. The paper should explicitly state that UDWDC-v2 in its current form is not practical and exists only to diagnose the P-only instability.

### Section 6 (Discussion)

**Strengths**: Section 6.3 (CWD magnitude confound) is methodologically rigorous — it correctly identifies the confound and flags the needed ablation as future work. Section 6.4 correctly concludes UDWDC instability is a genuine finding. Section 6.5 negative results are comprehensive and specific.

**Issue — Section 6.1 granularity**: The claim "scheduling-based methods are feedforward" and "constraint-based methods use integral control" is correct at the family level, but the paper should note this is a descriptive taxonomy, not a proof of optimality. CPR's integral control outperforming SWD's "proportional-integral" control does not mean integral is universally better — CPR's gains could come from norm-explicit targets rather than integral accumulation per se.

**Issue — Section 6.4 last paragraph**: Contradicts H3 falsification (noted above). This is the only place in the paper where a conclusion is inconsistent with the evidence presented in the same section.

### Section 7 (Conclusion)

**Strengths**: The conclusion correctly summarizes the three feedback channel findings (integral dominates, proportional destabilizes, derivative marginal). The CSI as necessary complement to accuracy is well-argued.

**Issue — full PID failure understated**: The conclusion focuses on individual gain findings but does not adequately address the Full PID failure (69.29% on CIFAR-100, worse than all single-gain variants). This is the paper's most surprising negative result: a complete PID controller performs worse than any of its components. The conclusion should dedicate one sentence to this finding, as it has implications for the entire PID design approach.

---

## Methodology Critique

### Framework Validity

The PID framing is descriptive, not prescriptive: the paper maps existing methods to gain configurations post-hoc, then proposes UDWDC as the "natural" P-only controller. This is valid as a unification contribution but does not prove the PID parameterization is the optimal design space. A reviewer may ask: why not use a higher-order control law? Why is $\rho_t^l$ the right control variable (vs. $\|w_t^l\|$ directly, or $\alpha_t^l$ directly)?

The paper partially addresses this through the Defazio/Wang & Aitchison theoretical grounding, but the connection from "Defazio proves $\rho_t^l$ has a steady state" to "therefore $\rho_t^l$ is the right control variable" needs one more sentence: the steady state makes $\rho_t^l$ predictable and therefore controllable; methods that ignore this steady state (scheduling methods) are open-loop.

### Empirical Validity

The 3-seed protocol is adequate for CIFAR but marginal for ImageNet. CPR's ImageNet result (74.74 ± 0.05%, 3 seeds) has unusually tight standard deviation — tighter than FixedWD's 5-seed result (71.72 ± 0.36%). This warrants a note: CPR's tight std could reflect either genuine stability or an artifact of the 3-seed selection. Reporting median accuracy alongside mean ± std would add robustness.

The batch-size sweep on CIFAR-100/VGG-16-BN (Section 5.4) uses 3 seeds but the large batch sizes (bs=512, bs=1024) show high variance (std=4.64 and 7.16 pp respectively). These high-variance results require more seeds (at least 5) for reliable conclusions. The current framing "H3 is falsified" is too strong given the high variance at large batch sizes — "H3 is not confirmed" is more defensible.

### Standardized Metrics

BEM as defined (accuracy improvement per unit WD budget) has a denominator problem: if NoWD accuracy exceeds FixedWD accuracy (which does not occur here but is plausible), BEM would be negative for FixedWD. The metric is well-motivated but needs a note about this edge case.

CSI: As noted in Section 4 critique, the formula is undefined for FixedWD (zero variance). The paper's workaround (normalize relative to FixedWD's CSI) contradicts the formula. This inconsistency must be resolved.

AIS: Not reproducible from current definition. This is the weakest of the three metrics from a scientific rigor standpoint.

---

## Specific Issues by Priority

### Priority 1 — Blocking (must fix before submission)

1. **Figure 4 wrong data**: Replace `writing/figures/alignment_snr.pdf` with `exp/results/full/figures/fig04_batchsize_snr.pdf`. The current figure directly contradicts the text on H3 falsification.

2. **Figure 2 dead reference**: Generate the UDWDC control loop diagram (5-block: Target Generator → Summation → Controller → Plant → Measurement). Save as `writing/figures/udwdc_control_loop.pdf`. This is a LaTeX compilation blocker.

3. **6 of 8 figures absent**: At minimum embed Figures 1 (rho trajectories) and 6 (H7 bimodal R² distribution) from pre-generated files at `exp/results/full/figures/`. Remove figures listed in the end-of-paper summary if they are not embedded in the body.

4. **AIS not reproducible**: Add one sentence to Section 4.3 operationalizing how batch-size conditioning is applied and resolving the AIS (0.566) vs. correlation_alpha_gengap (0.698) discrepancy.

### Priority 2 — Major (required for top-venue acceptance)

5. **Section 5.4 final paragraph contradicts H3 falsification**: Rewrite to remove the monotonic-SNR-based recommendation.

6. **CSI formula undefined for FixedWD**: Unify the formula in Section 4.2 with the normalization procedure in Section 5.7.

7. **Table 5 (ImageNet) missing BEM and Gen Gap**: Add at least CPR and FixedWD BEM values.

8. **Full PID failure conclusion**: Add one sentence to Section 7 on why Full PID underperforms all single-gain variants.

### Priority 3 — Minor (improve clarity)

9. **Abstract: additive vs. multiplicative form discrepancy** — Add "(implemented as a multiplicative clamp; see Algorithm 1)" after the PID control law in the abstract.

10. **Table 1: SWD footnote** — Add "global, not per-layer" annotation for SWD to avoid the Table 1 vs. fitting-error contradiction.

11. **Theorem 1 proof sketch**: Either provide one or explicitly cite the Sun et al. theorem it extends.

12. **CPR tight std note**: Mention that CPR's 74.74 ± 0.05% over 3 seeds is unusually tight; note that median accuracy is not reported.

13. **FixedWD 71.72% context**: The note "below standard recipes (76–77%) because minimal augmentation" is correct and helpful — but add that CPR's gain (+3.02 pp) is measured under the same protocol, so the relative comparison is fair.

---

## What Works Well

1. **Honest pre-registration of negative results** in the abstract (UDWDC below NoWD, fitting errors, H3 falsification). This is rare in ML papers and strongly increases reviewer trust.

2. **CPR integral control interpretation** is the paper's cleanest theoretical contribution. The connection from augmented Lagrangian → penalty accumulation → integral control is precise and testable.

3. **CWD magnitude confound analysis** (Section 6.3) is methodologically rigorous. Identifying that the CWD binary mask halves effective WD and hypothesizing that the halving (not the alignment) drives the gain is the type of mechanistic analysis top venues reward.

4. **UDWDC v1 instability documentation** (CSI = −2.41, accuracy below NoWD). A paper that proposes a method, then honestly reports the method fails, and quantifies the failure mechanism with a new metric (CSI), is a contribution even if UDWDC itself is not practically useful.

5. **Table 4 (gain ablation)** provides the cleanest evidence: Kd_only beats FixedWD; Kp_only underperforms; integral partially compensates for proportional noise. This 7-row table is the most informative single table in the paper.

---

## Score

**5/10**

The conceptual contribution (PID unification taxonomy, three standardized metrics, honest negative results) is publishable. The three blocking issues (Figure 4 wrong data, missing figures, AIS not reproducible) and the internal contradiction in Section 5.4 prevent submission in current form. Fixing these four issues would raise the score to 7-8/10. The paper's unusual honesty about its own method's failure is a genuine differentiator; protect this in any revision.

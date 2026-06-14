# Critique: Discussion

## Summary Assessment

The Discussion section is among the strongest in the paper. It precisely scopes each positive claim, honestly acknowledges tensions and failure modes, and consistently leads with numbers. Section 5.3 (Informative Failure Modes) is particularly well-executed: the three failure modes are named, quantified, and given mechanistic interpretations without overclaiming. The section closely fulfills its outline promise and is internally consistent with the experiments.

The main weaknesses are: a numerical discrepancy in Section 5.4 (absorption rate range inconsistent with the experiments section), a redundancy between the EDA magnitude tension discussion appearing in both Section 5.2 and the paper's Method section (Sec. 3.3), and an absence of any forward pointer to figures in a section that references specific geometry without a visual anchor. These are correctible without structural rework.

## Score: 8/10

**Justification**: The section earns its high score through disciplined, evidence-backed writing, honest treatment of negative results, and precise scoping of claims. It reaches 8 rather than 9 because: (1) the absorption rate numbers in Sec. 5.4 contradict Sec. 4.4 (0.876–0.978 vs. 0.876–0.978 are quoted consistently in the discussion, but the opening sentence cites "0.876–0.978" while the conclusion paragraph at the end of Section 5 cites different bounds — see Critical Issue 1); (2) cross-section referencing of Proposition 2 leaves one mechanistic claim slightly underspecified in Sec. 5.2 (decoder AUROC = 1.000 vs. encoder AUROC = 0.991 is dropped); (3) the closing summary paragraph duplicates content efficiently but could link more explicitly to implications for practitioners. To reach 9/10, fix the numerical inconsistency and either add a figure reference in Sec. 5.2 or add a sentence making clear Figure 4 is the basis for the decoder vs. encoder alignment claim.

---

## Critical Issues

### Issue 1: Absorption Rate Bounds — Inconsistency Between 5.4 and Closing Summary

- **Location**: Section 5.4, paragraph 1 vs. closing summary paragraph (after the horizontal rule)
- **Quote (5.4)**: "Absorption rates are 0.876–0.978 across all 11 tested configurations (layers 2–10, widths 12k–98k, L1-penalized and AJT training regimes, $L_0$ range 18–81)."
- **Quote (closing summary)**: "absorption is phase-stable across the tested hyperparameter range (rates 0.876–0.978, no phase transition...)"
- **Problem**: These two numbers are identical, which is actually correct — but cross-checking against the Experiments section (Sec. 4.4) reveals a discrepancy: Sec. 4.4 states the mean is 0.950, but the outline and conclusion cite the range as 0.919–0.968 for the "positive finding" summary (Conclusion: "Absorption rates span only 0.919–0.968"), while Discussion Sec. 5.4 uses 0.876–0.978. Both represent "all 11 configurations" but the bounds differ. The Conclusion section (6) uses 0.919–0.968, while the Discussion uses 0.876–0.978 — a direct contradiction for the same claim about the same dataset.
- **Fix**: Verify the actual empirical range from `exp/results/full/E1_phase_transition.json`. Use exactly one pair of bounds across Discussion 5.4, the closing summary, and the Conclusion. If 0.876–0.978 is the correct range for all 11 configs, update the Conclusion. If 0.919–0.968 reflects a narrower subset, clarify which subset each range applies to. Cross-reference Sec. 4.4 which states the standard jb suite alone spans 0.938–0.967.

---

## Major Issues

### Issue 2: Section 5.2 References Encoder-Decoder AUROC Numbers That Are Not Stated in This Section

- **Location**: Section 5.2, paragraph 1
- **Quote**: "decoder-probe cosine alignment averages $0.383$ versus encoder-probe cosine alignment of $0.139$ (paired $t$, diff $= -0.244$, $p = 3.5 \times 10^{-38}$)"
- **Problem**: The paragraph correctly cites the paired t-test result and the means. However, it omits the AUROC-based confirmation that is central to the mechanistic interpretation: decoder AUROC = 1.000, encoder AUROC = 0.991 (both stated in Sec. 4.2). The Discussion section is the correct place to synthesize this — the decoder is effectively a perfect discriminator for letter features, while the encoder is not. Omitting this number makes the claim "the decoder points toward the letter-identity direction; the encoder does not" slightly underspecified, because a reader might expect both to be quite high given they are looking at letter features by construction.
- **Fix**: After the paired t-test result, add one sentence: "As binary predictors of letter-feature membership, the decoder achieves AUROC = 1.000 versus the encoder's AUROC = 0.991 — a gap that is small in absolute terms but confirms that the decoder is strictly more aligned with the letter concept than the encoder, even within this already-selected set."

### Issue 3: Figure 4 Is Not Referenced in Section 5.2

- **Location**: Section 5.2, opening paragraph
- **Problem**: Section 5.2 presents the encoder-decoder decomposition evidence from Figure 4 (referenced explicitly in Sec. 4.2: "Figure 4 decomposes EDA..."), but the Discussion section provides no figure reference. A reader encountering Section 5.2 in isolation cannot locate the visual support for the claim about decoder vs. encoder probe alignment. The opening sentence "The most direct mechanistic evidence comes from Figure 4" appears in Section 4.2, not here — yet Section 5.2 is the primary interpretive site for this finding.
- **Fix**: Add "(Figure 4)" after "the most direct mechanistic evidence" claim is introduced, or open with "As shown in Figure 4, for 50 letter features at layer 6..." This applies the paper's own rule that figures must be referenced where the claim they support is made.

### Issue 4: Width Analysis Is Missing from Section 5.3

- **Location**: Section 5.3 (Informative Failure Modes)
- **Problem**: Section 5.3 covers three failure modes — L10 reversal, AJT polarity reversal, and encoder norm dominance. However, the width analysis from Section 4.3 also represents a form of failure: EDA_delta decreases monotonically as width increases from 12k (+0.028) to 98k (−0.017), and the signal disappears at 98k. This is a fourth failure mode — width-induced signal dilution — that is mechanistically distinct from the AJT reversal and has a clear interpretation (feature splitting reduces per-feature absorption concentration). Omitting it from Section 5.3 creates an incomplete picture of EDA's failure modes and may mislead practitioners into thinking EDA works for any L1-penalized SAE.
- **Fix**: Add a short fourth subsection in 5.3: "**Width saturation.** As SAE width increases from 12k to 98k at layer 8 (matched L0 ≈ 51), EDA_delta decreases from +0.028 to −0.017, crossing zero between 49k and 98k. The signal dilutes because feature splitting at wider widths distributes child-concept representation across multiple features, not all of which share the absorption geometry. EDA thresholds validated at 24k width cannot be transferred to 49k+ without recalibration."

### Issue 5: Section 5.4's Hysteresis Limitation Claim Is Slightly Circular

- **Location**: Section 5.4, penultimate paragraph
- **Quote**: "The saturation of absorption rates throughout the tested range also means that the classical hysteresis experiment...cannot be executed in this regime: no non-absorbed stable state is accessible via parameter changes within the tested L0 range."
- **Problem**: This is a valid point, but the section first establishes hysteresis as showing the absorbed state is metastable (fine-tuning doesn't escape it), then says classical hysteresis cannot be tested. The two claims are not contradictory, but the ordering makes the limitation feel like a retraction of the positive finding. The reader might reasonably conclude: "so the hysteresis result is meaningless?" The section should make explicit that the 500-step fine-tuning experiment establishes one half of hysteresis (stability of the absorbed state under perturbation), while the other half (existence of a competing non-absorbed state) requires a different experimental design.
- **Fix**: Reframe: "The hysteresis experiment establishes one half of the hysteresis picture: the absorbed state is stable under sparsity perturbation within the tested range. The other half — whether a non-absorbed stable state exists at sufficiently low sparsity — cannot be tested here because no configuration in the tested L0 range (18–81) produces a non-absorbed SAE. Testing the full bistability claim requires either architectural modification or L0 values outside this range, which we leave for future work."

---

## Minor Issues

- **Sec. 5.1, paragraph 2, sentence about AUPRC**: "Features ranked in the top EDA percentile are enriched for absorbed features (AUPRC $= 2.09\times$ base rate at L6)" — the parenthetical "at L6" is ambiguous: does this mean at L6 specifically, or across all tested layers at L6? This should be "at L6 against exact Chanin labels ($n_+ = 18$)" for precision.

- **Sec. 5.2, "EDA magnitude tension" subsection**: This tension is already introduced and explained in Section 3.3 (Method) with nearly identical language ("We do not resolve this tension in the present work; it could indicate partial alignment only, or that the relevant parent-child decoder angles are not as small as the $\lambda \approx 0.02$ threshold..."). The Discussion adds one new element (the "temporal reconciliation" hypothesis about post-convergence encoder drift), but a reader may feel this ground was already covered. Either (a) explicitly cross-reference ("As noted in Section 3.3, this tension is unresolved; here we add a candidate temporal explanation...") or (b) consolidate so the Method mentions the tension briefly and the Discussion contains the full reconciliation attempt.

- **Sec. 5.5, "ratio-to-null = 10.0, 120 events, GO"**: The format "(GO)" and "(NO_GO)" borrows from the experimental notation in Sec. 4.5 without the definitions being re-introduced in the Discussion. For a discussion section that might be read without Sec. 4.5 immediately preceding it, these are unexplained terms. Add a brief gloss: "(GO: statistically above null; NO_GO: indistinguishable from null)" on first use.

- **Sec. 5.6, last sentence**: "The EDA framework provides the measurement tool." — this is an unsupported forward assertion. The claim that EDA can measure $\theta_{p,c}$ changes in OrtSAE/Matryoshka/ATM SAEs is a prediction, not a demonstrated capability. Soften: "Measuring $\theta_{p,c}$ distributions in trained OrtSAE and Matryoshka SAE weights using EDA would test these predictions directly and is a tractable next experiment."

- **Sec. 5.7, "Proxy label quality" limitation**: "The AUROC results using proxy labels should be interpreted with this in mind" — this is vague. Specify: "The AUROC = 0.659 with proxy labels measures EDA's ability to detect features with high decoder-probe alignment, not confirmed absorbed features; it should be read as a separate, complementary result rather than a stronger version of the exact-label AUROC = 0.650."

- **Closing summary paragraph**: The paragraph restates four key numbers (z_null = 2.49, AUROC = 0.650/0.730, rates 0.876–0.978). This is appropriate for a closing synthesis, but note the inconsistency flagged in Critical Issue 1 — this paragraph's bounds (0.876–0.978) conflict with the Conclusion section's bounds (0.919–0.968).

---

## Visual Element Assessment

- [x] Discussion contains no new figures, consistent with the outline plan ("None (Discussion section contains no new figures; references figures from earlier sections)")
- [ ] Section 5.2 discusses Figure 4's data without citing Figure 4 — fix as noted in Major Issue 3
- [x] No text-heavy explanation requires additional visual support beyond what is already planned
- [x] No redundant visuals

---

## What Works Well

1. **Failure mode documentation (Sec. 5.3)** is exemplary. Each failure mode (L10 reversal: AUROC = 0.256, Cohen's d = -0.890; AJT: EDA_delta = -0.204 to -0.217; encoder norm: AUROC = 0.757, DeLong p = 0.153) is presented with exact numbers, two competing interpretations, and an explicit statement of what data would distinguish them. This is the model for how a mechanistic interpretability paper should handle anomalies.

2. **Proposition 1 unification (Sec. 5.6)**: The account of why OrtSAE, Matryoshka SAE, and ATM SAE reduce absorption — all unified under "increasing $\sin^2(\theta_{p,c})$" — is the kind of theoretical payoff a venue like NeurIPS expects from a theory paper. The mechanistic linking is explicit, falsifiable, and properly marked as a prediction rather than a verified result.

3. **Label sparsity limitation (Sec. 5.7)**: The honest quantification of the label sparsity problem ("$n_+ = 18$ out of 24,576... small changes in the label set can substantially alter the apparent metric") is exactly what a reviewer wants to see upfront. The suggestion of a concrete fix (run FeatureAbsorptionCalculator at lower activation thresholds) is actionable rather than vague.

# Critique: Discussion — REVISION ROUND

## Summary Assessment

This is the REVISION round critique. The Discussion section shows marked improvement from Round 1. Three major Round 1 issues are now fixed: (a) Section 6.2 title changed from "When Adaptive WD Should Help" to "When Dynamic WD Should Help" (terminology violation resolved); (b) Section 6.2 bullet points now carry explicit epistemic qualifiers ("partially supported," "directional," "speculative"); (c) Section 6.3 PMP-WD now includes "empirical validation pending, Section 6.5" qualification and "is designed to do so by construction." The remaining critical gap — that Section 6.1 still does not show the Theorem 1 AIS threshold computation — persists. There are also two downstream issues from the unfixed experiments problems: Section 6.2's NoBN discussion depends on incomplete rho_low/NoBN data that has not been updated in the experiments section, and the discussion does not cross-reference the Figure 6 gap.

## Score: 7/10

**Justification**: Round 1 score was 7/10. The section fixed the three major Round 1 issues (terminology, epistemic qualifiers, PMP-WD overclaiming), which is substantial work. The score holds at 7/10 because the one remaining Critical issue — the missing AIS threshold computation in 6.1 — is still absent. The discussion says "AIS values fall below the Theorem 1 threshold" without ever stating what the threshold value is, leaving the central theoretical claim empirically unverifiable in the paper itself. Fixing this single issue (one sentence with computed threshold) would bring the section to 8/10.

---

## REVISION FOCUS: Review.md Issue Downstream Effects on Discussion

The four review.md issues (Table 3 footnote, Figure 6, rho_low data, run count) are experiments-section errors, but two have downstream Discussion effects:

**Effect 1 (rho_low → Section 6.2 "Elevated rho"):** Section 6.2 bullet 1 states the elevated-rho prediction is "partially supported" and references Section 5.3. If rho_low cwd_hard data were added (90.09% vs. constant 90.13%), Section 6.2 should update its characterization to "further supported by rho_low partial data: CWD does not gain at rho_low either." Currently the section says nothing about rho_low CWD, only citing the SGD and rho_high gap. Adding the rho_low CWD data point to the discussion would strengthen the multi-regime argument.

**Effect 2 (Figure 6 missing → Section 6.2 "Without BN"):** Section 6.2 discusses the NoBN AIS increase but cannot visually reference Figure 6 because it does not exist. When Figure 6 is generated, Section 6.2 should add: "As shown in Figure 6, the BN vs. NoBN difference in constant-WD accuracy (Cohen's d = 9.14) dwarfs any inter-method spread, confirming BN is the dominant accuracy factor at this scale."

These are secondary dependencies — they will resolve when the experiments issues are fixed.

---

## Changes From Round 1 — What Was Fixed

The following Round 1 major issues are resolved:

1. **Section 6.2 title fixed.** Changed from "When Adaptive WD Should Help" to "When Dynamic WD Should Help" — terminology violation resolved per glossary. ✓
2. **Epistemic qualifiers added to 6.2.** All three regime bullets now explicitly state their data support level: "(partially supported; rho-high data gap, Section 5.3)", "(directional; AIS increase observed but Phi spread not yet confirmed, Section 5.5)", "(speculative; no experimental data)." ✓
3. **PMP-WD overclaiming fixed.** Section 6.3 bullet 1 now reads "PMP-WD is designed to do so by construction (empirical validation pending, Section 6.5)" — no longer asserts "PMP-WD does compensate" for an unimplemented algorithm. ✓
4. **NoBN gap precision.** Section 6.2 adds "within 1-std margin" for the 0.10% gap. ✓
5. **RG derivation regime qualifier.** Section 6.3 bullet 2 now references "with agreement within 15% over the moderate-alignment regime $\hat{\delta}_t \in [0.3, 0.7]$ (Remark 3.1)." ✓

---

## Critical Issues

### Issue 1: Theorem 1 validation in 6.1 never shows the threshold computation

- **Location**: Section 6.1, paragraphs 1–2
- **Quote**: "AIS values in our BN experiments (0.18--0.40) fall below the Theorem 1 threshold."
- **Problem**: The threshold is $\text{AIS}^* = (C\sigma^2/n) \cdot \Delta\text{CSI} / \bar{\lambda}$ (notation.md). The discussion asserts that measured AIS (0.18–0.40) falls below this threshold but never reports what the threshold value actually is. The reader cannot evaluate whether the margin is narrow (AIS = 0.38, threshold = 0.41) or wide (AIS = 0.38, threshold = 2.1). Without this, the theoretical framework's empirical bite is invisible. The conclusion section (Section 7) also just says "7/7 predictions confirmed" but does not show the threshold number anywhere in the main body.
- **Fix**: Add one sentence with the computed threshold value (or range), e.g., "At our experimental settings ($n = 50{,}000$, estimated $C\sigma^2 \approx X$, $\Delta\text{CSI} \approx Y$, $\bar{\lambda} = 5 \times 10^{-4}$), the Theorem 1 threshold evaluates to $\text{AIS}^* \approx Z$, well above the measured range of 0.18–0.40." If $C$ is unknown, bound it or note it is estimated empirically.

---

## Major Issues

### Issue 2: "Adaptive WD" used where glossary prescribes "dynamic WD"

- **Location**: Section 6.2 title "When Adaptive WD Should Help", repeated in body: "potentially enabling direct alignment-based WD modulation without EMA smoothing"
- **Quote**: "When Adaptive WD Should Help" (section header)
- **Problem**: The glossary (glossary.md) explicitly prescribes: "Dynamic weight decay" = any WD strategy where the effective decay rate varies; "Adaptive weight decay" = specifically PMP-WD using state-feedback. Section 6.2 is discussing the regime conditions where *dynamic* WD methods in general become beneficial — not specifically PMP-WD. Using "adaptive WD" for this generic discussion violates the established terminology and could confuse readers into thinking the section is specifically about PMP-WD (which is the true "adaptive WD" in this paper's taxonomy).
- **Fix**: Change section header to "When Dynamic WD Should Help" and audit the body text for any additional uses of "adaptive WD" in the generic (non-PMP) sense.

### Issue 3: Three-regime predictions in 6.2 have inconsistent epistemic status

- **Location**: Section 6.2, three bullet points
- **Quote**: "The theory predicts three regimes where adaptive WD becomes beneficial: Elevated rho. [...] Without BN. [...] Large-batch training."
- **Problem**: The three predictions have very different levels of empirical support, but they are presented in an undifferentiated list:
  - "Elevated rho": two data points exist (rho-low constant-only and SGD) but the critical rho-high (rho = 5.0) data failed. The section notes this honestly at the end but the opening framing ("theory predicts") does not convey the data gap.
  - "Without BN": NoBN shows higher AIS (0.50 vs 0.35) but only 2 methods completed; the section admits the AIS increase "is not sufficient" but the opening is still phrased as a positive prediction.
  - "Large-batch training": purely theoretical; no experimental evidence at all. The LLM-scale claim (4K–64K tokens) is speculation presented alongside partially-supported empirical claims.

  A NeurIPS reviewer will notice this epistemic conflation and penalize the paper for overstating the data-supported scope of the theory.
- **Fix**: Add a brief qualifier to each bullet: "(partially supported: rho-high data gap, Section 5.3)", "(directional: AIS increase observed but Phi spread not yet confirmed, Section 5.4)", "(speculative: no experimental data; Proposition 1 predicts the noise floor drops, but this is unvalidated)". This takes 6–10 words per bullet and significantly strengthens the paper's credibility.

### Issue 4: PMP-WD section makes claims about generalization performance without evidence

- **Location**: Section 6.3, bullet 1
- **Quote**: "When the actual trajectory deviates from the plan---due to data distribution shift, batch noise, or learning rate schedule interactions---feedforward methods cannot compensate; PMP-WD does."
- **Problem**: This is a strong empirical-sounding claim ("PMP-WD does [compensate]") made for an algorithm that is explicitly unimplemented. The previous section 6.5 Limitation 5 acknowledges "PMP-WD not implemented." Saying a theoretical algorithm "does compensate" in the middle of the discussion — without the disclaimer visible at that point — is misleading. A reviewer checking the claim at 6.3 before reaching 6.5 will mark this as overclaiming.
- **Fix**: Add a parenthetical qualifier in 6.3: "...feedforward methods cannot compensate; PMP-WD is designed to do so by construction (empirical validation pending, Section 6.5 Limitation 5)." or restructure bullet 1 to say "is designed to correct" rather than "does."

---

## Minor Issues

- **6.1, paragraph 2**: "A 10$\times$ variation in effective WD budget (BEM from 0.0 to 0.90) produces $< 0.25\%$ accuracy change." The BEM range 0.0 to 0.90 corresponds to constant (BEM=0) vs. SWD (BEM≈0.90), which is actually the full BEM range in the data. Writing "BEM from 0.0 to 0.90" as if it were a controlled test of the same method understates the actual claim. Clarify: "Across all 7 methods, a 10$\times$ variation in effective WD budget (BEM spanning 0.0 to 0.90) produces..."

- **6.2, bullet 2**: "The available NoBN data (2 methods, constant slightly outperforming CWD by 0.10%) does not yet resolve whether the AIS increase is sufficient to cross the Theorem 1 threshold." The gap is 0.10% (87.74% vs. 87.64%), which is smaller than the 3-seed std (0.20% and 0.17% respectively). This non-result deserves a brief nod to the statistical ambiguity: "...outperforming CWD by 0.10% (within 1-std margin)."

- **6.3, bullet 2**: "Dual derivation. The same functional form emerges from both stochastic PMP (control theory) and RG beta function analysis (statistical physics)." This claim's strength depends on how similar the forms actually are. The method section (Remark 3.1) notes the convergence holds "in the moderate-alignment regime (δ̂ ∈ [0.3, 0.7])." The discussion here claims "same functional form" without the regime qualifier, which is slightly stronger than the method section's wording. Add the regime qualifier or reference Remark 3.1.

- **6.4, recommendation 1**: "Dynamic scheduling adds implementation complexity without accuracy benefit. The 0.25% Phi spread on CIFAR-10 and 0.16% on VGG-16-BN are within inter-seed variance." The claim that they are "within inter-seed variance" requires checking. ResNet-20 inter-seed std is 0.31% (constant), so 0.25% spread is indeed within 1 std. VGG-16-BN std is 0.06% (constant), and 0.16% spread is larger than the constant std (though Table 3 shows half-lambda's std is 0.13%). The statement should be more precise: "The 0.25% spread is within the 0.31% inter-seed std of the best method; VGG-16-BN's 0.16% spread exceeds the individual method std but falls below any single method's confidence interval width."

- **6.5, Limitation 3**: "Seed 42 constant shows anomalous 76.12% (5 epochs only)". The experiments section (Table 5) correctly footnotes this with $^\dagger$. The limitation section re-states this clearly. No change needed. Good.

- **Glossary check — "scale-invariance"**: The paper uses "scale-invariance" (hyphenated) in 6.1. The glossary has no entry for this term. This is fine since it refers to BN's standard mathematical property, not a paper-specific term, but the first use in method or experiments should have a citation or definition cross-reference if "BN's scale-invariance limits alignment benefit" is a load-bearing claim.

- **Cross-section consistency**: The conclusion section says "Phi spread ≤ 0.25% under AdamW (CIFAR-10) and ≤ 0.16% on VGG-16-BN." The discussion section says "the 0.25% Phi spread on CIFAR-10 and 0.16% on VGG-16-BN." These are consistent. Good.

- **Cross-section consistency**: The method section's Theorem 1 says "CWD achieves lower expected test loss than constant WD if and only if AIS > (Cσ²/n)·ΔCSI/λ̄." The discussion section's 6.1 refers to "the Theorem 1 threshold" without reprinting the formula. This is acceptable in a Discussion section (the reader has seen the theorem), but adding a brief inline reminder "(AIS > C·σ²·ΔCSI / (n·λ̄))" would help readers who read sections non-linearly.

---

## Visual Element Assessment

- [x] No figures/tables planned for Discussion section in the outline — this is correct; Discussion is appropriately text-only
- [x] The `<!-- FIGURES: None -->` comment is correctly present
- [ ] No issue with visual elements since none are expected

---

## What Works Well

1. **Limitations subsection is exemplary.** Section 6.5 lists 7 specific, quantified limitations with honest data counts ("only 2 of 7 methods have 3-seed data," "3 seeds gives limited power for effect sizes below 0.3%"). Most ML papers bury limitations in vague sentences; this section names each gap and its consequences. This will build reviewer trust.

2. **The state-feedback vs. feedforward distinction in 6.3 is crisp and well-evidenced.** The contrast between PMP-WD (measures $\hat{\rho}_t$ and corrects deviations) and AdamC (follows predetermined $\gamma_t$ regardless of state) is made concrete with a specific failure mode ("when the actual trajectory deviates from the plan"). This is precise, not generic.

3. **Practical recommendations in 6.4 are specific and actionable.** The three recommendations have precise numeric thresholds ("$\lambda \sim 10^{-4}$--$10^{-3}$, BN architectures," "EMA with $k \geq 10$") and identify the reporting gap in prior work ("new dynamic WD methods should report gradient-to-weight ratio as a context variable"). This makes the paper immediately useful to practitioners.

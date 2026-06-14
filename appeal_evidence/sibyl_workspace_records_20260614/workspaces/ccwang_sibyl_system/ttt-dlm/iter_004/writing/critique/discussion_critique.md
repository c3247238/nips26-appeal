# Discussion Section Critique

**Section**: 5. Discussion
**Reviewer**: section-critic (automated)
**Score**: 7/10

---

## Summary

The Discussion section provides thoughtful analysis of five key findings, with particularly strong mechanistic explanations for why BSD+A-CFG combination fails (Section 5.1) and why JSD stability is uninformative (Section 5.2). The negative-results-as-constraints framing (Section 5.5) is a genuine strength that elevates the paper beyond a simple methods report. However, there are several issues ranging from critical to minor that should be addressed before submission.

---

## Issues

### CRITICAL

**C1. Missing Figure 6 — failure diagnostics are described but not rendered**
- Location: Section 5.2 references "Figure 6a" (line 15), and a comment block (lines 49–52) lists three panels for Figure 6, but the figure itself is only present as an HTML comment placeholder.
- Impact: The three-panel failure diagnostics figure (JSD histogram, guidance magnitude scatter, entropy discontinuity) is planned in the outline as the primary visual element for the Discussion. Without it, the mechanistic arguments in Sections 5.1–5.3 lack the data-driven visual evidence that would make them convincing. The JSD stability histogram (panel a) is especially important — it is the single most compelling piece of evidence for why RACFG fails.
- Suggestion: Generate Figure 6 from `token_diagnostics_summary.md` and `racfg_remask_pct_ablation_summary.md` as specified in the outline's Figure & Table Plan. Insert it after Section 5.2 or at the end of Section 5.3, and reference all three panels explicitly in the text.

**C2. Missing Table 3 — hypothesis verification summary**
- Location: Section 5.5 (line 39) states "We summarize these constraints in Table 3" but the table is not included — only referenced.
- Impact: Table 3 is planned in the outline as a structured hypothesis verification matrix (H1–H11 with expected/observed/verdict). Its absence means the "systematic mapping" claim in Section 5.5 is unsupported. This is a key organizational contribution of the paper.
- Suggestion: Insert Table 3 inline in Section 5.5, or provide a concrete markdown/LaTeX table. The outline specifies the exact format.

### MAJOR

**M1. Section 5.4 makes a factual claim that contradicts itself within two paragraphs**
- Location: Lines 31–35. The section opens by calling the information island problem a "genuine bottleneck" with "particularly compelling" evidence, then immediately argues that vanilla step-scaling matches the methods at equivalent FLOPs. The final paragraph then calls DMI "the most efficient method discovered" with a "validated ~2x improvement."
- Impact: The reader receives three conflicting messages: (1) the bottleneck is real and important, (2) simple step-scaling is equally effective, (3) DMI is still the best. The logical tension is not adequately resolved.
- Suggestion: Restructure Section 5.4 to explicitly acknowledge the apparent contradiction and resolve it: the bottleneck is real (evidence: entropy metrics), but addressing it at the representation level is not yet more efficient than simply adding compute. Then position DMI separately as the exception (near-zero overhead). Consider using a "efficiency vs. effectiveness" framing.

**M2. The Limitations subsection is buried and undersized relative to the statistical weaknesses**
- Location: Lines 46–47, within Section 5.5 as a subsection.
- Impact: The paper's pilot-scale evaluation (n=16) is arguably its most significant weakness — Bootstrap CIs include zero for every comparison. Burying this as a subsection within "Negative Results as Design Space Constraints" underplays it. Additionally, the limitations section mentions five points but does not discuss the most obvious one: the paper draws mechanistic conclusions (why methods fail) from samples too small to reliably confirm that they fail.
- Suggestion: (1) Promote Limitations to a top-level subsection (Section 5.6) for visibility. (2) Add a sixth limitation: mechanistic explanations for failure modes (Sections 5.1–5.3) are post-hoc rationalizations based on pilot data — alternative explanations cannot be ruled out at n=16. (3) Explicitly state which conclusions are robust to sample size and which are tentative.

**M3. Section 5.3 overgeneralizes from a single model**
- Location: Lines 21–27. The section concludes that "CFG scheduling theory developed for continuous diffusion does not transfer to discrete masked diffusion" (bold text, line 27).
- Impact: This is a strong universal claim based on experiments with one model (Dream-7B) on one benchmark (Countdown-16, n=16). The Limitations section (line 47) acknowledges single-model scope but does not temper the bold claim in Section 5.3 itself.
- Suggestion: Qualify the bold claim: "**On Dream-7B, CFG scheduling theory developed for continuous diffusion does not transfer to discrete masked diffusion**" or add a caveat noting this is a single-model observation.

### MINOR

**m1. Inconsistent accuracy reporting format**
- Location: Throughout. Section 5.1 uses "6.2%" and "12.5%" without confidence intervals; Section 5.4 uses "12.5%" and then "9.3% vs. 4.7%, 3-seed validated" with different notation.
- Suggestion: Standardize: always report CIs for pilot results (n=16), always note "(3-seed)" for Countdown-500 results.

**m2. Section 5.2 last paragraph makes a prescriptive recommendation without sufficient grounding**
- Location: Lines 18–19. "Future work on MDLM inference-time scaling should therefore focus on *within-step* signals... rather than *across-step* signals."
- Impact: This recommendation is based on one model's behavior. LLaDA or other MDLMs may exhibit different cross-step dynamics.
- Suggestion: Soften to "For models exhibiting Dream-7B-like cross-step stability, future work should prioritize..." or reference the Limitations caveat.

**m3. The BSD+A-CFG failure explanation (Section 5.1) could benefit from a concrete numerical example**
- Location: Lines 9–11. The mechanistic explanation is entirely verbal.
- Suggestion: Add a brief quantitative illustration: e.g., report the entropy distribution of BSD belief states entering Phase 2 vs. standard mask states, or show the confidence score distribution shift. This data may already be in Figure 6c (once generated).

**m4. Missing forward-reference to Conclusion**
- Location: End of Section 5.5 (line 43).
- Impact: The outline specifies a transition to Section 6, but the section ends abruptly after the A-CFG paragraph with no bridge to the Conclusion.
- Suggestion: Add a brief transition sentence, e.g., "We summarize our contributions and outline future directions informed by these constraints in Section 6."

**m5. The bold-text "general principle" in Section 5.1 (line 11) is insightful but not revisited**
- Location: "methods that modify the representation space and methods that modify the prediction space may not compose freely"
- Impact: This is potentially the paper's most broadly applicable insight, but it appears once and is never connected to the broader ML community's experience with method composition.
- Suggestion: Revisit this principle in Section 5.5's synthesis or the Conclusion, and consider whether it connects to known phenomena (e.g., regularization stacking, optimization landscape interference).

---

## Visual Element Review

| Planned Element (from Outline) | Present? | Status |
|-------------------------------|----------|--------|
| Figure 6: Failure Mode Diagnostics (3-panel) | No | Only HTML comment placeholder. **Must be generated.** |
| Table 3: Hypothesis Verification Summary | No | Referenced in text but not included. **Must be created.** |

The Discussion section is the weakest in terms of visual support — both planned elements are missing. The text-heavy explanations in Sections 5.1–5.3 would significantly benefit from the diagnostic visualizations. In particular:
- Section 5.2's JSD stability argument is much stronger with a histogram (Figure 6a) than with prose alone.
- Section 5.5's "systematic mapping" claim requires Table 3 to be credible.

---

## Strengths

1. **Mechanistic depth**: The explanations for why methods fail are not hand-wavy — they identify specific technical mechanisms (EMA smoothing vs. confidence sharpness, JSD degeneracy, discrete vs. continuous information geometry).
2. **Honest reporting**: The section does not hide negative results; it frames them as contributions.
3. **Actionable design principles**: The bold-text takeaways (lines 11, 27) give readers concrete guidance for future work.
4. **Compute-fairness analysis** (Section 5.4): Including this analysis shows intellectual honesty and prevents overclaiming.

---

## Score Justification: 7/10

The Discussion demonstrates strong analytical thinking and honest scientific reporting. However, two critical issues — the missing Figure 6 and Table 3 — mean the section currently lacks its primary visual evidence, which is essential for a venue like NeurIPS. The internal contradiction in Section 5.4 and the overgeneralization in several subsections also need attention. With the visual elements generated and the major issues addressed, this section could reach 9/10.

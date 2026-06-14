# Critique: Discussion

## Summary Assessment
The Discussion section successfully synthesizes six hypotheses across the phase transition framework, correctly framing negative results and genuine discoveries. The writing is clear and the logical flow from validation to paradox to implications is well-structured. However, the section suffers from critical gaps in visual communication (all figures are referenced only in comments, not in the body), a speculative mechanism that lacks direct validation, and several cross-section consistency issues that undermine the section's credibility.

## Score: 6.5/10
**Justification**: The Discussion section has strong synthesis and framing quality, correctly reframing H3 and H6 as informative negatives and properly elevating the variance paradox as a genuine discovery. However, it loses points for: (1) complete absence of in-body figure references despite a detailed figure plan in the outline, (2) a speculative mechanism for the actionability paradox without direct validation, and (3) terminology inconsistencies with other sections (e.g., "absorption rate α" vs "absorption_rate" in tables).

## Critical Issues

### Issue 1: No In-Body Figure References
- **Location**: Throughout Section 5 (missing entirely)
- **Quote**: N/A — figures are only referenced in HTML comment blocks (`<!-- FIGURES -->`)
- **Problem**: The outline specifies 4 figures for the Discussion section (fig1_phase_transition, fig2_scaling_collapse, fig3_cv_comparison, fig4_cooccurrence), and the body text discusses these results extensively (e.g., "Figure 1", "Figure 2", "Figure 3" mentions in the text), but there are zero actual in-text figure references. A reader cannot determine from the prose alone which figure illustrates which result.
- **Fix**: Add explicit figure references in each subsection. For example:
  - Section 5.1: "The order parameter m(λ) shows a smooth susceptibility peak at λ_c = 5 × 10⁻⁵, with χ_max = 11.19 (Figure 1)"
  - Section 5.3: "This striking reversal appears consistently at every layer tested (Figure 3)"
  - Section 5.4: "The CV-based decomposition offers a hypothesis (Figure 4)"

### Issue 2: Speculative Mechanism for Actionability Paradox Lacks Direct Validation
- **Location**: Section 5.4, lines 41-53
- **Quote**: "The CV-based decomposition offers a hypothesis for which absorbed features might remain steerable: low-CV absorbed features may have less specialized child channels, allowing parent steering to produce measurable output changes. Testing this hypothesis requires activation patching experiments comparing steering effectiveness across the CV spectrum."
- **Problem**: This paragraph proposes a mechanism for the actionability paradox but explicitly states the validation is not done. However, the experiments section (4.8) already reports a steering effectiveness experiment (Table 7) showing "High-CV absorbed features exhibit approximately 2× larger steering effects than low-CV absorbed features (0.153 vs 0.075)." This directly contradicts the Discussion's claim that validation is missing. The Discussion should reference these results, not claim the hypothesis is untested.
- **Fix**: Revise Section 5.4 to reference the steering validation results from Section 4.8. The mechanism is supported by preliminary evidence, not purely speculative. Use: "Our steering experiments (Section 4.8) show high-CV absorbed features exhibit ~2× larger steering effects than low-CV absorbed features (0.153 vs 0.075), partially validating the specialized child channel mechanism."

## Major Issues

### Issue 3: Cross-Section Terminology Inconsistency — "absorption rate α" vs "absorption_rate"
- **Location**: Section 5.2 vs experiments.md Table 3 vs conclusion.md Section 6.1
- **Quote**: "all tested layers {0, 3, 6, 9, 11} show α = 1.0 (absorption rate)" (Section 5.2 line 15)
- **Problem**: The notation.md glossary defines α as "absorption rate (fraction of features absorbed)" and notes it ranges from 0 to 1. However, in the experiments section (4.3), the same concept is labeled as "absorption_rate" with values "1.0" (no Greek alpha). The conclusion uses "absorption rate α" interchangeably. This inconsistency may confuse readers about whether α and absorption_rate are the same variable.
- **Fix**: Ensure consistent notation across sections. If α is the defined symbol for absorption rate, use it uniformly: "absorption rate α = 1.0" in experiments Table 3 header, and reference "absorption rate α" in Discussion Section 5.2.

### Issue 4: Cross-Layer Measurement at Wrong Sparsity is Labeled as "Limitation" Not "Future Work"
- **Location**: Section 5.6 (Limitations), line 69-70
- **Quote**: "Cross-layer absorption measurement at λ_c = 5 × 10⁻⁵ (not λ = 0.001) to test layer-criticality"
- **Problem**: The outline (Section 5.2) explicitly lists this as a critical experiment for validating layer-criticality, and the conclusion (6.2 Limitation L3) frames it as "Cross-layer measurement at wrong sparsity." But in the Discussion Section 5.6, it's listed under "Future Directions" not "Limitations." This creates inconsistency about whether this is a completed limitation or a pending experiment.
- **Fix**: Clarify the framing — if this was an executed experiment that produced an informative negative result, it should appear in Section 5.2's revised narrative, not as an unexecuted future experiment. The text should explicitly state whether the cross-layer measurement at λ_c was actually performed or is still pending.

### Issue 5: The "Genuine Discovery" Framing for Variance Paradox is Inconsistent with Outline
- **Location**: Section 5.3, line 22 vs outline.md Section 5.3
- **Quote**: "The coefficient of variation (CV) analysis reveals a striking reversal from the predicted pattern" (line 23)
- **Problem**: The outline (Section 5.3) explicitly states "H4 SUPPORTED (REVERSED: CV_reversal)" and frames the variance paradox as a discovery. However, Section 5.3 opens with "The variance paradox: A Genuine Discovery" as a subsection title, which is appropriate. But the table in Section 5.3 (lines 25-32) shows "inf" ratio for all layers because CV_non-absorbed = 0.0. The text states "ratio: inf" but earlier in the experiments (4.4) this is reported as "733x ratio" with specific numbers 7.33/0.01. The discrepancy needs reconciliation.
- **Fix**: If ratio is truly "inf" due to 0/0, the "733x ratio" claim from experiments needs clarification. The discussion should explain: 733x is computed using the means (7.33/0.01 ≈ 733) but individual layer ratios are undefined due to division by zero in the non-absorbed CV.

## Minor Issues

- **Location**: Section 5.1, line 9: "R² = 0.951 at ν = 3 (Figure 2)" — Figure 2 is referenced in-text but the reference should appear before describing the result, not after.
- **Location**: Section 5.3, line 33: "All t-statistics exceed 1000 (p < 0.01)" — The exact t-values are reported in experiments.md Table 4 as ranging from 1093 to 1474. The Discussion should cite these specific values for precision.
- **Location**: Section 5.5, line 59: "Operating below the critical point" — Use consistent terminology: "below λ_c" or "below the critical sparsity" not both in the same paragraph.
- **Location**: Section 5.6, line 74: "does not account for the CV reversal" — The term "CV reversal" should be "variance paradox" per the glossary's preferred terminology.
- **Location**: Section 5.4, line 51: "The variance paradox mechanism requires theoretical formalization" — This sentence is redundant with Section 5.3's framing and could be consolidated.

## Visual Element Assessment

- [ ] **Figures match outline plan**: The outline specifies 4 figures for Discussion (fig1, fig2, fig3, fig4). However, NONE of these figures are referenced in the body text of the Discussion section — they only appear in HTML comment blocks.
- [ ] **All visuals referenced before appearance**: No figure references appear in the body text at all. All figure mentions are in comments.
- [ ] **Captions are self-explanatory**: Cannot assess — figures are not visible in the markdown, only referenced in comments.
- [ ] **No text-heavy sections that need visual support**: Section 5.3 (Variance Paradox) would benefit significantly from the inline Figure 3 reference — the text describes the 733x ratio but does not point to the visual evidence.
- **Missing**: Figure 3 (CV comparison) should be explicitly referenced in Section 5.3 to support the "733x ratio" claim with visual evidence.
- **Missing**: Figure 1 and Figure 2 should be referenced in Section 5.1 to support the phase transition validation claims.

## What Works Well

1. **Subsection 5.2 (Layer Saturation Puzzle)**: The revised narrative correctly explains why H3 failed — measuring at λ = 0.001 was past the critical point for all layers. The framing is precise and the logical pivot to "measurement at λ_c" is well-argued. Quote: "The genuine layer-critical behavior should manifest at finer sparsity values near λ_c = 5 × 10⁻⁵." This is a model of how to handle falsified hypotheses.

2. **Subsection 5.3 (Variance Paradox)**: Correctly elevates the CV reversal from "failed hypothesis" to "genuine discovery." The interpretation via Pearl's causal mediation framework is principled and the connection to "context-sensitive" vs "degraded" signal is compelling. The table with exact layer-by-layer CV values is excellent evidence.

3. **Subsection 5.5 (Implications)**: Provides genuinely actionable guidance for interpretability practitioners. The three concrete recommendations (operating below λ_c, CV-based feature triage, architectural interventions targeting variance profile) are specific and testable. This is the strongest subsection.

4. **Framing of Informative Negatives**: The discussion correctly handles H3 and H6 as informative negatives (not failed hypotheses), properly contextualizing them within the quasi-critical framework. This demonstrates sophisticated understanding of the results.
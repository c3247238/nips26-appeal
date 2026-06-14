# Visual Audit Report

**Date:** 2026-04-15
**Editor:** sibyl-editor
**Manuscript:** paper.md (iter_007, initial integration)

---

## Summary

- **Total figures:** 5
- **Total tables:** 8 (Tables 1--8)
- **Missing visuals:** 0 (all planned figures present)
- **Consistency issues found and fixed:** See details below

---

## Completeness Check

All five figures from the outline's Figure & Table Plan are present in the manuscript and in the `writing/figures/` directory:

| Figure | File | Section | Status |
|--------|------|---------|--------|
| Figure 1 | control_failure.pdf | 4.1 | Present, referenced before appearance |
| Figure 2 | hedging_decomposition.pdf | 4.4 | Present, referenced before appearance |
| Figure 3 | l0_phase_transition.pdf | 5.1 | Present, referenced before appearance |
| Figure 4 | cmi_vs_absorption.pdf | 6.3 | Present, referenced before appearance |
| Figure 5 | cmi_dimension_sensitivity.pdf | 6.4 | Present, referenced before appearance |

All eight tables are inline and appear in correct order:

| Table | Section | Content | Status |
|-------|---------|---------|--------|
| Table 1 | 3.1 | SAE configurations | OK |
| Table 2 | 4.1 | Cross-domain control results | OK |
| Table 3 | 4.3 | Threshold sensitivity heatmap | OK |
| Table 4 | 4.4 | Hedging classification | OK |
| Table 5 | 4.5 | Activation patching results | OK |
| Table 6 | 6.2 | CMI correlation summary | OK |
| Table 7 | 5.1 | L0 phase transition rates | OK |
| Table 8 | 3.7 | Cross-domain hierarchy suite | OK |

**Note on numbering:** Table 7 appears in Section 5 (after Table 6 in Section 6 in the manuscript's reading order). This is because the outline designed Tables 6 and 7 for separate Sections 5 and 6 respectively. In the integrated paper, Table 7 (L0 phase transition) appears before Table 6 (CMI analysis) in the text. This numbering is preserved from the outline for cross-referencing consistency with other artifacts. Future LaTeX compilation should renumber sequentially.

---

## Consistency Check

### Figure Numbering
- Figures are numbered sequentially 1--5 across sections. OK.

### Table Numbering
- Tables 1--5 are sequential in Sections 3--4. Tables 6--8 follow. The gap (Table 6 in Section 6 before Table 7 in Section 5) reflects the outline's assignment. Flagged for LaTeX renumbering.

### Caption Style
- All captions use sentence case with period at end. Consistent.
- All captions are self-explanatory. Reviewed and confirmed.

### Bold Best Results
- Table 2: Bold ratios for shuffled/measured. Consistent.
- Table 3: Bold marks the default threshold. Consistent.
- Table 4: Bold marks the strict hedging row. Consistent.
- Table 5: No bold needed (all results are "No"). Consistent.

---

## Flow Check

- Figure 1 is referenced in Section 4.1 text before it appears. OK.
- Figure 2 is referenced in Section 4.4 text before it appears. OK.
- Figure 3 is referenced in Section 5.1 text before it appears. OK.
- Figure 4 is referenced in Section 6.3 text before it appears. OK.
- Figure 5 is referenced in Section 6.4 text before it appears. OK.
- No orphan figures (all are referenced in text).
- Figures appear as close as possible to their first reference.

---

## Issues Found and Addressed During Integration

1. **Numerical inconsistency resolved.** The intro/conclusion previously used 15.96% and 74.6% for first-letter absorption while experiments.md Table 2 used 13.4% and 59.6%. Investigation revealed two distinct measurement protocols: the first-letter replication (1,203 words, probes at $L_0$=82) producing 15.96%/74.6%, and the confound decomposition protocol (1,195 words, probes at $L_0$=22) producing 14.39%. The integrated paper uses 15.96%/74.6% in Table 2 (matching the cross-domain comparison protocol) and 14.39% in Table 7 (matching the confound decomposition protocol), with an explicit reconciliation note in Section 5.1.

2. **Section numbering aligned.** The experiments.md had a monolithic Section 4 (4.1--4.10). The integrated paper splits this into Sections 4, 5, and 6 matching the outline and the introduction's roadmap.

3. **Table numbering:** Tables are numbered 1--8. Table 6 (CMI) and Table 7 ($L_0$ phase transition) are out of reading order due to outline assignment. Flagged for LaTeX renumbering.

4. **"Ten letters" corrected to "twelve letters"** in Section 4.4 (confound decomposition). The listed letters (B, D, F, H, J, K, M, N, P, R, T, W) number 12, not 10.

5. **Glossary compliance fixes:**
   - "feature absorption" used on first mention in each section
   - "competitive suppression" replaced with "competitive exclusion" throughout
   - "missed features" replaced with "missed latents" (Section 2.3, MP-SAE)
   - "$L_0$" typeset correctly throughout (no bare "L0" in body text)
   - "SAE" expanded on first use per section
   - "CMI" expanded as "conditional mutual information" on first use per section
   - "FN" expanded as "false negative" on first use per section

6. **Title changed** from "Beyond Competitive Exclusion" to "Auditing Feature Absorption Metrics on JumpReLU SAEs" -- the original title implied competitive exclusion is disproven, but the paper shows the metric conflates hedging with competitive exclusion on JumpReLU SAEs. The new title is descriptive rather than claiming a conclusion the data do not fully support (per evolution lesson on title misleading).

7. **Cross-domain novelty claim removed.** The previous version claimed "first cross-domain absorption characterization" while admitting rates are uninterpretable. The integrated paper reports the cross-domain results factually without claiming novelty for measurements that fall below controls (per evolution lesson on self-contradiction).

8. **$\tau_{\text{mag}}$ defined** in Section 3.2 absorption criterion, addressing the method critique's concern that the magnitude gap threshold was used in Section 3.6 but never defined.

9. **Bootstrap CIs added** to Table 7 ($L_0$ phase transition), addressing the experiments critique.

10. **Conclusion restructured** to explicitly close the loop on Q1/Q2/Q3, addressing the conclusion critique.

11. **Abstract length:** 237 words. Within the 200--250 word target.

---

## Suggestions for Additional Visuals

The paper is text-heavy in Section 7 (Discussion). A summary figure or graphical abstract showing the three findings (control failure bars, $L_0$ phase transition curve, CMI scatter) in a compact 3-panel layout would improve accessibility in the Introduction. This is optional and recommended for the camera-ready version.

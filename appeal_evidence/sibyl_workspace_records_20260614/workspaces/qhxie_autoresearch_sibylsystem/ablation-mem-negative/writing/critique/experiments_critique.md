# Critique: Experiments

## Summary Assessment
The Experiments section presents UAD and DFDA results with commendable honesty about metric limitations, particularly the DFDA caveat. However, it suffers from significant structural issues: missing visual elements (Figures 2, 3, 4 referenced but not verifiable), inconsistent table numbering that deviates from the outline plan, and a logical gap where the cross-layer pilot (P3) is discussed before being introduced. The section is technically accurate but lacks the narrative scaffolding expected at a top venue.

## Score: 6/10
**Justification**: The section gets the numbers right and is admirably honest about limitations, earning a solid base. To reach 7-8, it needs: (1) fixable table/figure numbering consistency with the outline, (2) a clearer narrative arc from pilot to full experiments, (3) explicit discussion of why 1,000 samples is sufficient, and (4) addressing the missing E4 experiment ID gap. To reach 9+, it would need richer analysis of *why* precision is 0.569 and what the 22 false positives look like.

---

## Critical Issues

### Issue 1: Missing E4 Experiment Creates Confusion
- **Location**: Section 4.5 heading "E5: DFDA Scaling"
- **Quote**: "## 4.5 E5: DFDA Scaling (8 Pairs)"
- **Problem**: The experiment numbering jumps from E3 to E5 with no explanation. Readers will search for E4 and conclude content is missing. This is a structural defect that undermines paper completeness.
- **Fix**: Either (a) renumber to E4, or (b) add a one-sentence note: "Experiment E4 (cross-layer full validation) was merged into E3 after pilot results indicated layer 4 fell below threshold." Check `exp/results/full/` directory to confirm what E4 was and explain its absence.

### Issue 2: Table Numbering Inconsistent with Outline Plan
- **Location**: Sections 4.5 and 4.6
- **Quote**: "Table 5 reports per-pair details" and "Table 6 presents the comprehensive UAD results"
- **Problem**: The outline plan specifies Table 1 = Main UAD Results, Table 2 = DFDA Detail, Table 3 = Prior Work Comparison. The section uses Tables 5 and 6, which conflicts with the outline's Figure & Table Plan. This will cause LaTeX compilation issues and confuse readers cross-referencing the outline.
- **Fix**: Rename Table 5 -> Table 2 (DFDA Per-Pair Detail) and Table 6 -> Table 1 (Main Results), matching the outline plan. Ensure all in-text references are updated.

---

## Major Issues

### Issue 3: Figure References Without Verification
- **Location**: Sections 4.4, 4.5, 4.6
- **Quote**: "![Cross-Layer F1 Comparison](figures/fig3.pdf)", "![DFDA Per-Pair Results](figures/fig4.pdf)", "![UAD Performance Summary](figures/fig2.pdf)"
- **Problem**: Three figures are referenced but the markdown uses image links to `.pdf` files. At the draft stage, these figures may not exist or may not match the data. The section assumes figures are available without confirming their content aligns with the reported numbers.
- **Fix**: Verify that `figures/fig2.pdf`, `figures/fig3.pdf`, and `figures/fig4.pdf` exist and contain data matching the tables. If figures are not yet generated, add a `[FIGURE PENDING]` marker and ensure the caption description in the outline is implementable. The outline specifies Figure 2 should have error bars -- verify the multi-seed data supports this (std = 0.000, so error bars would be invisible -- this is a design issue).

### Issue 4: P3 Referenced Before Introduction
- **Location**: Section 4.1, third paragraph
- **Quote**: "Pilot P3 (UAD, cross-layer). UAD across layers 4, 8, and 10 yielded F1 = 0.432..."
- **Problem**: P3 is the third pilot mentioned, but the reader has not yet been told what P3's purpose was. P1 and P2 have clear purpose statements ("UAD, layer 8" and "DFDA, 2 pairs"), but P3 appears without context. The narrative jumps from P2 to P3 without a transition explaining *why* cross-layer validation was tested.
- **Fix**: Add a purpose clause: "**Pilot P3 (UAD, cross-layer).** To test whether UAD's detection signature generalizes across model layers, we evaluated layers 4, 8, and 10. UAD across these layers yielded..."

### Issue 5: Insufficient Justification for Sample Size
- **Location**: Section 4.2
- **Quote**: "The analysis covered 15,000 token positions and completed in 7.6 seconds."
- **Problem**: The section reports 1,000 samples and 15,000 token positions but never justifies why this sample size is sufficient for stable phi coefficient estimation. A skeptical reviewer will question whether 1,000 samples from OpenWebText (which contains ~8M documents) provides adequate coverage. The determinism claim (Section 4.3) relies on this sufficiency but does not argue for it.
- **Fix**: Add one sentence: "With 1,000 samples averaging 15 tokens each, the co-occurrence matrix captures ~15,000 activation events per feature, sufficient to stabilize phi coefficient estimates for the top 500 most active features." Cite a standard or pilot sensitivity analysis if available.

### Issue 6: No Analysis of False Positives
- **Location**: Section 4.2
- **Quote**: "The precision of 0.569 indicates that 43% of same-cluster pairs are false positives."
- **Problem**: The section acknowledges false positives but does not analyze them. What characterizes the 22 false positive pairs? Are they semantically related (e.g., co-occurring but not absorbed)? Are they random? This is a missed opportunity to strengthen the method's credibility by showing the authors understand *why* their method errs.
- **Fix**: Add a paragraph analyzing false positives. Example: "The 22 false positive pairs cluster into two categories: (1) features that co-occur frequently due to topical correlation but lack parent-child hierarchy, and (2) features that share a common super-absorber child. Feature X (example) illustrates category 1: it co-occurs with feature Y on 89% of tokens containing 'the', but neither is absorbed by the other." If no such analysis exists in the results files, add it as a limitation note.

### Issue 7: DFDA Metric Caveat Buried in Subsection
- **Location**: Section 4.5, paragraph 3
- **Quote**: "The near-100% improvement is artifactual..."
- **Problem**: While the caveat is present and appropriately strong, its placement at the end of the DFDA subsection means readers scanning the section may miss it. The abstract, introduction, and conclusion all flag DFDA as preliminary, but the experiments section itself does not prominently label the DFDA subsection as preliminary until the reader reaches the caveat paragraph.
- **Fix**: Add a prominent preliminary marker at the start of Section 4.5: "**Preliminary Result -- Metric Under Revision.**" Consider moving the caveat to the first paragraph of the subsection.

---

## Minor Issues

- **Section 4.1, P2**: "99.999% mean MSE improvement" -> The table shows 99.999% for most pairs but 96.225% for pair 5. The mean in the table is 99.528%. The text should say "99.5% mean" (matching Section 4.5) not "99.999% mean."
- **Section 4.2**: "15,000 token positions" -- clarify whether this is 1,000 samples x ~15 tokens/sample or a different calculation.
- **Section 4.3**: "corpus sampling randomness does not perturb the co-occurrence structure enough to change cluster assignments" -- this is an assertion without evidence. A footnote or brief justification would help.
- **Section 4.4**: "Layer 8's optimality is consistent with prior work" -- the citation [Elhage et al., 2022] is about superposition, not specifically about layer-wise hierarchy optimality. Verify this citation supports the claim or find a more specific one.
- **Table 5 (DFDA)**: All phi values are identical (0.812) for pairs 1-4 and 6. This seems suspicious -- verify these are correct or add a footnote explaining why multiple pairs share the same phi coefficient.
- **Section 4.6**: "Table 6 presents the comprehensive UAD results" -- this table repeats data from earlier subsections. Consider whether a summary table is necessary if all data has already been presented, or restructure so Table 6 is the *first* presentation of all data and earlier subsections reference it.
- **Missing: Statistical significance testing**: No p-values, confidence intervals, or bootstrap estimates are reported for any metric. The notation.md defines bootstrap CI but notes it is "not yet implemented." At minimum, acknowledge this as a limitation.
- **"PARTIAL_PASS" terminology**: Used in Section 4.1 for P3 but nowhere else. If this is internal validation language, it should not appear in the paper draft.

---

## Visual Element Assessment

- [ ] **Figures/tables match outline plan**: NO. Table numbering (5, 6) does not match outline (1, 2). Figure numbering (2, 3, 4) matches but cannot be verified.
- [ ] **All visuals referenced before appearance**: PARTIAL. Figures 2, 3, 4 are referenced before their image links, but the image links appear inline without clear "Figure X shows..." textual introductions in some cases.
- [ ] **Captions are self-explanatory**: CANNOT VERIFY. The markdown uses `![alt text](path)` syntax without separate captions. The outline specifies detailed captions but these are not in the section draft.
- [x] **No text-heavy sections that need visual support**: The DFDA table (Table 5) is appropriate; the cross-layer discussion would benefit from Figure 3 being present.

**Additional visual concern**: Figure 2 (grouped bar chart) is described in the outline as having "error bars where applicable." But multi-seed std = 0.000, so error bars would be invisible. The figure design should be adjusted -- perhaps show individual seed bars instead of a mean bar with error bars.

---

## What Works Well

1. **Honest DFDA caveat (Section 4.5)**: The explicit disclosure that "the near-100% improvement is artifactual" and the clear explanation of why (near-zero prediction on child-dominant examples) is exemplary scientific writing. This builds reviewer trust rather than undermining it.

2. **Determinism vs. robustness distinction (Section 4.3)**: The section correctly identifies that perfect multi-seed consistency reflects determinism (fixed SAE, sufficient samples) rather than robustness (to SAE retraining, corpus change, model change). This nuance shows sophisticated understanding of experimental design.

3. **Precision contextualization (Section 4.2)**: The framing that "UAD is a detection tool that identifies an enriched candidate set requiring post-hoc filtering, not a classifier" appropriately manages expectations and positions the method honestly.

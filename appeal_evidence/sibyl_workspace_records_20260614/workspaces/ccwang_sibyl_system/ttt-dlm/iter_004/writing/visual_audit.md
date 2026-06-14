# Visual Element Audit Report

**Date**: 2026-03-10
**Auditor**: sibyl-editor (integration pass)

---

## Summary

- **Total figures**: 6
- **Total tables**: 5 (inline)
- **Figures with rendered artifacts**: 4 of 6
- **Figures with description-only specs**: 2 of 6

---

## Completeness Check

| Figure/Table | Outline Plan | Present in Paper | Artifact Status |
|---|---|---|---|
| Figure 1: Info Island + BSD Teaser | Section 1 | Yes, with caption | Description spec only (`fig1_info_island_bsd_desc.md`). Needs tikz/manual rendering. |
| Figure 2: Method Architecture (BSD + A-CFG) | Section 3 | Yes, with caption | Description spec only (`fig2_method_architecture_desc.md`). Needs tikz rendering. |
| Figure 3: Belief Entropy Trajectories | Section 3 | Yes, with caption | **Rendered**: `entropy_trajectories.pdf` + `.png` exist. Generation script: `gen_entropy_trajectories.py`. |
| Figure 4: Ablation Grid (2x2) | Section 4 | Yes, with caption | **Rendered**: `fig4_ablation_grid.pdf` exists. Generation script: `gen_fig4_ablation_grid.py`. |
| Figure 5: GSM8K Generalization | Section 4 | Yes, with caption | **Rendered**: `fig5_gsm8k_generalization.pdf` exists. Generation script: `gen_fig5_gsm8k_generalization.py`. |
| Figure 6: Failure Mode Diagnostics | Section 5 | Yes, with caption | **Rendered**: `fig6_failure_diagnostics.pdf` + `.png` exist. Generation script: `gen_fig6_failure_diagnostics.py`. |
| Table 1: Method Comparison | Section 3 | Yes, inline | Complete. |
| Table 2: Full-scale Countdown-500 | Section 4 | Yes, inline | Complete. |
| Table 3: Pilot-scale Results | Section 4 | Yes, inline | Complete. |
| Table 4: Compute-fair Comparison | Section 4 | Yes, inline | Complete. |
| Table 5: Hypothesis Verification | Section 4 | Yes, inline | Complete. |

**Additional artifacts found**: `task_sensitivity.pdf/.png` and `gen_task_sensitivity.py` exist in the figures directory but are not referenced in the outline or paper. These appear to be from an earlier iteration.

---

## Consistency Check

### Figure Numbering
- Figures are numbered sequentially 1--6 across sections. **Consistent.**
- Tables are numbered sequentially 1--5 across sections. **Consistent.**
- Note: The original section files used different table numbering (e.g., the method section had "Table 1" for the comparison table, and experiments had its own "Table 1", "Table 2", "Table 3"). These have been **renumbered** in the integrated paper to avoid conflicts: method comparison is Table 1, full-scale results Table 2, pilot results Table 3, compute-fair Table 4, hypotheses Table 5.

### Caption Style
- All captions use sentence case with period at end. **Consistent.**
- All captions provide sufficient context for standalone comprehension. **Consistent.**

### Color Scheme
- No `style_config.py` found in figures directory, but figure description files specify consistent color schemes:
  - Gray for mask embeddings / vanilla baselines
  - Green-to-blue gradient for belief states / BSD
  - Red for discarded info / low-confidence positions
- Generation scripts (`gen_fig4`, `gen_fig5`, `gen_fig6`) should be checked to ensure they follow the same palette.

---

## Flow Check

### Reference-before-appearance
- Figure 1: Referenced in Section 1, paragraph 3 ("see Figure 1, left panel"). Appears at end of Section 1. **OK.**
- Figure 2: Referenced implicitly in Section 3.2--3.3 method descriptions. Caption placed at end of Section 3. **OK.**
- Figure 3: Referenced in Section 3.5 (entropy analysis) and Section 4.6. Caption placed at end of Section 3. **OK.**
- Figure 4: Referenced in Section 4.3.1 ("see Figure 4a") and Section 4.3.4. Caption placed after Section 4.3.5. **OK.**
- Figure 5: Referenced in Section 4.4 narrative. Caption placed after Section 4.4. **OK.**
- Figure 6: Referenced in Section 3.3 ("see Figure 6a"), Section 5.2 ("Figure 6a"). Caption placed at end of Section 5. **OK.**

### Orphan Figures
- None. All 6 figures are referenced in the text before they appear.

### Visual Narrative Flow
- Method diagrams (Figures 1--2) precede detailed method descriptions. **Good.**
- Entropy trajectories (Figure 3) appear with the information-theoretic analysis. **Good.**
- Ablation grid and generalization chart (Figures 4--5) accompany results. **Good.**
- Failure diagnostics (Figure 6) support Discussion analysis. **Good.**

---

## Quality Check

### Captions
- All 6 figure captions are self-explanatory, describing what is shown and the key takeaway. **Good.**
- Table captions include scope annotations (full-scale vs. pilot). **Good.**

### Tables
- Table 2 (full-scale) and Table 3 (pilot-scale) are now **separated** --- this addresses the critique about mixing evaluation scales in one table. **Improvement over original sections.**
- Raw counts (e.g., "2/16") included alongside percentages in pilot tables. **Good.**
- Bootstrap CIs included in Table 3 for key comparisons. **Good.**
- Best results bolded in each table. **Consistent.**

### Redundancy
- No redundant figures identified. Each figure serves a distinct purpose.

---

## Missing Visuals (Action Required)

1. **Figure 1** (`fig1_info_island_bsd_desc.md`): Detailed description spec exists but no rendered PDF. **Needs tikz or manual diagram creation for LaTeX.**
2. **Figure 2** (`fig2_method_architecture_desc.md`): Detailed description spec exists but no rendered PDF. **Needs tikz or manual diagram creation for LaTeX.**

---

## Suggestions for Additional Visuals

1. **Pareto frontier figure**: Section 4.5 discusses Pareto optimality verbally. A FLOPs-vs-accuracy scatter with Pareto frontier line would strengthen the compute-fairness argument. Consider adding as supplementary Figure 7.
2. **Per-sample binary heatmap**: At $n = 16$, showing which individual problems each method solves (methods $\times$ problems binary grid) would be more informative than aggregate accuracy. Consider for Appendix.
3. **Forest plot of effect sizes**: Bootstrap CIs for all pairwise method comparisons plotted as a forest plot would powerfully convey the uncertainty at pilot scale. Consider for Appendix.

---

## Critique Feedback Addressed in Integration

| Critique Issue | Section | Resolution |
|---|---|---|
| C1 (intro too long) | Intro | Trimmed ~30%, removed detailed results enumeration, compressed contributions to prose |
| C2 (weak opening motivation) | Intro | Added practical motivation sentence about latency and self-correction |
| H1 (DMI/BSD relationship unclear) | Intro | Restructured: insight first, DMI as simplest fix, then BSD/A-CFG as full methods |
| H2 (pilot vs full-scale) | Intro, Experiments | Explicitly labeled pilot vs full-scale throughout; separated Tables 2 and 3 |
| H3 (contributions too granular) | Intro | Compressed to prose summary |
| M2 (prior iteration language) | All sections | Replaced with "initial experiments" / "detailed in Appendix" |
| M3 (boilerplate outline) | Intro | Removed section-by-section outline paragraph |
| C1 (RW: no taxonomy) | Related Work | Added organizing principle for Section 2.3 (continuous-to-discrete trigger) |
| M1 (RW: contribution framing) | Related Work | Moved "our methods address" to end-of-section summary |
| M2-M3 (RW: own results in RW) | Related Work | Moved own experimental numbers to forward references |
| C1 (method: prelim bleeds) | Method | Trimmed 3.1 to formal definition + info island problem only |
| C2 (method: schedule rationale) | Method | Led with ablation finding, removed unsupported "explore/exploit" rationale |
| M1 (method: k failure unexplained) | Method | Added forward reference to Section 5 analysis |
| M2 (method: missing combo protocol) | Method | Added Section 3.4 describing BSD+A-CFG combination |
| M3 (method: Phase 2 detail) | Method | Expanded Algorithm 1 Phase 2 with confidence metric and selection procedure |
| M4 (method: n=16 hedging) | Method | Added "suggestive" qualification for entropy-accuracy correlation |
| m5 (method: temperature unspecified) | Method | Specified $\tau = 1.0$ in Algorithm 1 |
| C1-C2 (experiments: mixed scales) | Experiments | Split into Table 2 (full-scale) and Table 3 (pilot-scale) |
| M4 (experiments: missing error bars) | Experiments | Added raw counts and Bootstrap CIs |
| M1 (experiments: baseline inconsistency) | Experiments | Added caveat about pilot statistical limitations |
| C1-C2 (discussion: missing Fig 6 and Table 3) | Discussion | Figure 6 rendered; hypothesis table moved to Section 4.7 |
| M1 (discussion: contradicts itself) | Discussion | Restructured with efficiency-vs-effectiveness framing |
| M2 (discussion: limitations buried) | Discussion | Promoted to Section 5.6 |
| M3 (discussion: overgeneralizes) | Discussion | Added "on Dream-7B" qualifier to bold claims |
| C1 (conclusion: overclaims in headers) | Conclusion | Added "(pilot-scale)" qualifiers to subsection headers |
| M2 (conclusion: redundancy) | Conclusion | Compressed negative results to listing without re-explanation |
| M3 (conclusion: missing full-scale next step) | Conclusion | Added immediate next step as priority future direction |

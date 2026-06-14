# Visual Element Audit Report

**Date**: 2026-03-10 (updated)
**Paper**: Denoising-Time Adaptation: Turning Diffusion Iterations into Test-Time Learning for Masked Language Models

---

## Summary

- **Total figures planned**: 6 (Figures 1--6)
- **Total tables planned**: 6 (Tables 1--4 inline, positioning table inline, ablation Table 5)
- **Figures with descriptions ready**: 2 (Figure 1, Figure 2)
- **Figures generated**: 1 (Figure 6: task_sensitivity.pdf)
- **Figures pending generation**: 3 (Figures 1, 2, 3, 4)
- **Figures pending data**: 1 (Figure 5: scaling curves need full-scale data)
- **Tables present inline**: 5 (Tables 1--4, positioning table)
- **Tables pending**: 1 (Table 5: full ablation results)

---

## Completeness Check

| Visual Element | Outline Plan | Status | Referenced in Text | Notes |
|---|---|---|---|---|
| Figure 1: DTA Algorithm Overview | Section 3.1 | Description ready (`fig1_dta_overview_desc.md`), **not generated** | Yes (Intro para 5, Section 3.1) | HIGH PRIORITY: Needs TikZ generation |
| Figure 2: Information Augmentation Spectrum | Section 3.3 | Description ready (`fig2_info_spectrum_desc.md`), **not generated** | Yes (Section 3.3 opening) | HIGH PRIORITY: Needs TikZ generation |
| Figure 3: Token-Level Diagnostics | Section 5.2 | **Not generated** | Yes (marked as TODO) | Needs matplotlib code from pilot data |
| Figure 4: LoRA Norm Trajectories | Section 5.3 | **Not generated** | Yes (marked as TODO) | Needs matplotlib code from `task_7b_decay_gamma.json` |
| Figure 5: Scaling Curves | Section 4.5 | **Not generated, data insufficient** | Yes (marked as TODO) | Pending full-scale data |
| Table (positioning): Method Positioning | Section 2.2 | **Present** (inline LaTeX table) | Yes (Table~\ref{tab:positioning}) | OK --- split Prism/search into separate row |
| Figure 6: Task Sensitivity | Section 6.2 | **Generated** (`task_sensitivity.pdf`) | Yes (Section 6.2) | OK |
| Table 1: Main Results | Section 4.2 | **Present** (inline) | Yes | OK --- bold only on Mean column |
| Table 2: Cross-Benchmark | Section 4.3 | **Present** (inline) | Yes | OK --- pilot caveat prominently added |
| Table 3: Cross-Model | Section 4.4 | **Present** (inline) | Yes | OK --- pilot caveat in subsection intro |
| Table 4: Scaling | Section 4.5 | **Present** (inline) | Yes | OK --- pilot caveat in subsection intro |
| Table 5: Full Ablation | Section 5.3 | **Not present** | Marked as TODO | Needs pilot data compilation |

---

## Consistency Check

### Figure Numbering
- Figures are numbered 1--6 sequentially. The outline's original Figure 6 (method positioning table) was implemented as an inline LaTeX table, and the task sensitivity figure is numbered Figure 6. **Consistent.**

### Table Numbering
- Tables 1--4 are numbered sequentially in Sections 4.2--4.5. Positioning table uses `\ref{tab:positioning}` without a number in the markdown flow. Table 5 (ablation) is planned but not yet present. **Consistent.**

### Caption Style
- Table captions use sentence case with period at end: consistent across Tables 1--4.
- Figure captions (from description files): sentence case with period at end. **Consistent.**
- TODO placeholders use brackets: `[TODO: Figure X]` format. **Consistent.**

### Color Scheme
- Figure descriptions specify: Gray=masked, Blue=revealed, Green=E-step, Orange=M-step/DMI, Red=SCP contradictions, Purple/Dark Blue=DTA/LoRA. No `style_config.py` file exists yet. **Recommendation**: Create a shared color config before generating figures.

### Notation Consistency
- Cross-step information loss problem: consistently used throughout (Introduction, Section 2.1, Section 3.1, Section 7). "Information Island" attributed to Xia et al. as independent terminology.
- LoRA parameters: $\Delta\theta$ with superscript counting updates (not denoising steps). Consistent.
- Decay factor: $\gamma$ throughout. No conflicts with other uses of gamma.
- Denoising steps: $T$ for total, $t$ for current step. Consistent.
- Revealed positions: $\mathcal{R}_t$. Masked positions: $\mathcal{M}_t$. Consistent.

---

## Flow Check

### References Before Appearance
- Figure 1: Referenced in Introduction (para 5) and Section 3.1 --- appears in Section 3. **OK** (intro reference is appropriate for teaser).
- Figure 2: Referenced in Section 3.3 opening --- appears in Section 3.3. **OK**.
- Figure 3: Referenced in Section 5.2 (TODO) --- would appear in Section 5.2. **OK**.
- Figure 4: Referenced in Section 5.3 (TODO) --- would appear in Section 5.3. **OK**.
- Figure 5: Referenced in Section 4.5 (TODO) --- would appear in Section 4.5. **OK**.
- Figure 6: Referenced in Section 6.2 --- appears in Section 6.2. **OK**.
- All inline tables are referenced before or at their location. **OK**.

### Orphan Figures
- No orphan figures detected. All planned figures have text references.

### Visual Narrative
- Method diagram (Figure 1) appears before detailed method description. **OK**.
- Spectrum diagram (Figure 2) appears alongside spectrum description. **OK**.
- Results tables appear with results discussion. **OK**.
- Diagnostic analysis figures support claims in Analysis section. **OK**.
- Task sensitivity figure supports Discussion claims. **OK**.

---

## Quality Check

### Captions
- Figure descriptions include self-explanatory captions with key takeaways. **OK**.
- Table captions describe content and note key qualifiers (pilot-scale, pending results). **OK**.
- Table 1 caption explicitly notes timing is from pilot-scale runs. **Fixed** from previous version.

### Table Formatting
- Tables have clear headers and method grouping (Remasking Baselines, Information Augmentation, Combined). **OK**.
- Best results bolded in Mean column only (not per-seed). **OK**.
- Dagger marks ($\dagger$) for incomplete results with footnote explaining. **OK**.
- Pilot tables labeled with "pilot" in caption and section introductions. **OK**.

### Redundancy
- No redundant figures detected. Each figure serves a distinct purpose.

---

## Issues Found and Fixed During Integration

1. **Introduction over-detailed**: Removed LoRA rank, layer count, decay factor, warmup specifics from intro; kept high-level intuition only. (Critique Issue 1)
2. **Contributions/findings merge**: Merged results preview and contributions list into a single enumerated structure. (Critique Issue 2)
3. **VDTA accessibility**: Added intuitive framing before formalism; qualified monotonicity as empirical observation rather than proven proposition. (Critique Issue 5, Method Critique C1)
4. **ELBO formulation**: Replaced KL divergence with point-mass $q$ with explicit $L_2$ penalty formulation to avoid mathematical inconsistency. (Method Critique C2)
5. **Algorithm vs text inconsistency**: Harmonized update rule to use `AdamW_step()` notation in both algorithm and text, with explicit discussion of $\gamma$-decay vs weight decay interaction. (Method Critique M1)
6. **Proposition 2 downgraded**: Changed from "Proposition" to "Empirical Observation" with supporting evidence. (Method Critique C1)
7. **Text/table mismatch**: Fixed "four desiderata" to "five desiderata" in Section 2.2 text. (Related Work Critique Issue 2)
8. **Concurrent work handling**: Defined "Information Island" in paper's own terms first, then attributed. (Related Work Critique Issue 6)
9. **Positioning table**: Split Prism/search into separate row; added footnote explaining "Partial" for TABES/ETS. (Related Work Critique Issues 10, 11)
10. **SCP accuracy range**: Reports 7.3--9.3% range instead of cherry-picked midpoint. (Experiments Critique C2)
11. **SCP overhead clarification**: Clarified ~7x as FLOP overhead vs ~12x wall-clock. (Experiments Critique m1, Method Critique M2)
12. **Pilot caveats**: Added prominent pilot-scale warnings in Sections 4.3, 4.4, 4.5. (Experiments Critique M3)
13. **DMI overhead consistency**: Used 1.16x (from Table 1 timing 4.3s/3.7s) in Discussion instead of ~1.05x. (Discussion Critique Issue 9)
14. **Discussion redundancy with Analysis**: Condensed Section 6.1 to reference Section 5.2 rather than repeating statistics. (Discussion Critique Issue 3)
15. **DTA overclaiming in conclusion**: Reframed pilot results with explicit caveats; led with fully validated contributions. (Conclusion Critique C1, C2)
16. **Conclusion length**: Cut redundant discussion overlap; added strong closing sentence on broader significance. (Conclusion Critique M1, m3)
17. **Parameter update definition**: Added explicit definition in Section 2.2 distinguishing persistent weight modification from one-shot gradient scoring. (Related Work Critique Issue 5)
18. **Prompt tokens in M-step**: Explicitly stated that only generated tokens participate in M-step. (Method Critique M4)
19. **"origin" sampling**: Added parenthetical definition. (Method Critique m1, Experiments Critique m6)
20. **LoRA alpha**: Specified $\alpha = r = 4$. (Method Critique m2)
21. **Essay analogy**: Refined to "editor who can re-read but has lost all notes and reasoning." (Intro Critique Issue 4)
22. **Roadmap paragraph**: Removed entirely. (Intro Critique Issue 9)
23. **Figure references in text**: Added explicit `Figure~1` and `Figure~2` references. (Method Critique visual review)
24. **Transition 2.2->2.3**: Added bridge sentence about TTT filling the training-free gap. (Related Work Critique Issue 8)
25. **Hardware details**: Added batch size/parallelization note. (Experiments Critique m4)
26. **Related work section streamlined**: Consolidated five subcategories into three, removed repetitive inline DTA differentiation. (Related Work Critique Issues 3, 4)
27. **Section 2.1 density**: Moved secondary model names to a single citation cluster. (Related Work Critique Issue 7)

---

## Suggestions for Additional Visuals

1. **Figure 3 (Token-Level Diagnostics)**: HIGH PRIORITY. This is the most impactful missing figure --- it visually demonstrates remasking's fundamental flaw and would dramatically strengthen Sections 5.2 and 6.1.

2. **Figure 4 (LoRA Norm Trajectories)**: MEDIUM PRIORITY. Supports the decay factor ablation argument. Data available in `task_7b_decay_gamma.json`.

3. **Figures 1 and 2 (TikZ diagrams)**: HIGH PRIORITY. These are method-defining visuals critical for reader comprehension. The paper is text-heavy without them.

4. **Bar chart for Table 1**: OPTIONAL. A grouped bar chart comparing Countdown-500 accuracy across completed methods would be more visually impactful than the table alone.

5. **Shared color config**: Create `writing/figures/style_config.py` to ensure color consistency across all generated figures.

---

## Action Items (Priority Order)

1. **Generate Figure 1** (DTA overview TikZ) --- critical for paper comprehension
2. **Generate Figure 2** (Spectrum TikZ) --- critical for ablation framework
3. **Generate Figure 3** (Token diagnostics bar chart) --- high impact for key argument
4. **Generate Figure 4** (LoRA norm trajectories) --- medium priority
5. **Complete full-scale DTA/SCP experiments** to fill Table 1 gaps and enable Figure 5
6. **Run McNemar tests** on completed methods (DMI vs vanilla) to deliver promised statistical significance
7. **Create style_config.py** for cross-figure color consistency
8. **Compile Table 5** (ablation) from existing pilot data files

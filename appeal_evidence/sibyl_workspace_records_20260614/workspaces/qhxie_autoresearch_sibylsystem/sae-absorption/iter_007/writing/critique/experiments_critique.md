# Section Critique: Experiments (Section 4)

**Section**: experiments.md (Section 4)
**Reviewer**: sibyl-section-critic
**Date**: 2026-04-15
**Overall Score**: 7 / 10

---

## Summary Assessment

The Experiments section is substantively strong: it presents a well-structured, multi-pronged audit of the Chanin absorption metric on JumpReLU SAEs, progressing logically from control failure (4.1) through structural explanation (4.2), threshold sensitivity (4.3), confound decomposition (4.4), causal intervention (4.5), probe quality confound (4.6), the L0 phase transition (4.7), and the exploratory CMI analysis (4.8--4.10). The statistical reporting is thorough (bootstrap CIs, effect sizes, Bonferroni corrections, leave-one-out sensitivity). The section's greatest strength is its honest treatment of negative results, particularly the CMI non-replication and the transparent acknowledgment of post-hoc cherry-picking in Section 4.9.

However, several structural, notational, and cross-section consistency issues reduce clarity and introduce risk of reviewer confusion. The section attempts to cover all three research questions (Q1--Q3) in a single experiments section, whereas the outline (outline.md) clearly partitions them across Sections 4, 5, and 6. This mismatch is the most critical structural issue.

---

## Critical Issues (Must Fix)

### 1. Section Structure Mismatch with Outline (Severity: HIGH)

The outline specifies:
- **Section 4**: Metric Audit (Sections 4.1--4.5 cover control failure, candidate explosion, threshold sensitivity, hedging decomposition, activation patching)
- **Section 5**: L0 Phase Transition (separate section)
- **Section 6**: Exploratory CMI Analysis and Its Failure to Replicate (separate section)

But the current experiments.md combines all three into Section 4 (subsections 4.1--4.10). This creates a monolithic 180-line section that violates the paper's stated organization. The Introduction (intro.md, line 25) explicitly says: "Section 4 presents the metric audit... Section 5 reports the L0 phase transition... Section 6 describes the exploratory CMI analysis." The Discussion (discussion.md) and Conclusion (conclusion.md) also reference this three-section structure.

**Action**: Split experiments.md into three files matching the outline:
- Section 4 (experiments.md): Sections 4.1--4.5 (plus 4.6 Probe Quality Confound, which belongs with the metric audit)
- Section 5 (l0_phase_transition.md or a new file): Current 4.7
- Section 6 (cmi_analysis.md or a new file): Current 4.8--4.10

### 2. Table Numbering Discontinuity (Severity: HIGH)

Table numbering jumps from Table 5 (Section 4.5) to Table 7 (Section 4.7), skipping Table 6. Table 6 appears later in Section 4.8 (the CMI correlation summary). This suggests the tables were numbered for the three-section structure in the outline (where Table 6 belongs in Section 6 and Table 7 in Section 5), but the current single-section layout makes the gap confusing. Readers encountering "Table 7" in Section 4.7 will look for Table 6 and find it later in Section 4.8, out of order.

**Action**: If keeping the single-section structure, renumber tables sequentially (Tables 2--8). If splitting into three sections per the outline, the current numbering may be correct but should be verified against the figure/table plan in outline.md.

### 3. Inconsistent Absorption Rate for First-Letter at L0=82 (Severity: HIGH)

The experiments section reports different first-letter absorption rates at L0=82 in different subsections:
- Section 4.1 (Table 2): 13.4% measured absorption
- Section 4.3 (Table 3): 14.6% at default thresholds (bold cell)
- Section 4.7 (Table 7): 14.39%
- Introduction (intro.md): 15.96% measured
- Conclusion (conclusion.md): 15.96% measured

The 13.4% in Table 2 vs. 14.39% in Table 7 vs. 15.96% in the Introduction are three distinct values for what should be the same quantity. Possible explanations include different quality gates (Table 2 may include all 25 letters vs. only the 10 passing F1 > 0.85), different vocabularies (1,204 vs. 1,196), or different threshold settings. But none of these differences are explained.

**Action**: Reconcile all first-letter absorption rate values. Add a footnote or parenthetical to Table 2 explaining which letters/vocabulary subset is used. Ensure the Introduction, Experiments, and Conclusion cite the same canonical rate, or explicitly note the difference (e.g., "aggregate across all 25 letters" vs. "restricted to 10 letters passing quality gate").

---

## Major Issues (Should Fix)

### 4. Opening Paragraph References Sections That Do Not Exist in This File

Line 1 states: "metric validity (Section 4.1--4.5), sparsity dynamics (Section 4.6--4.7), and the exploratory CMI analysis (Section 4.8--4.10)." If the section is split per the outline, these cross-references become invalid. Even in the current structure, the mapping is not clean: Section 4.6 (Probe Quality Confound) is about metric validity, not sparsity dynamics.

**Action**: Reassign Section 4.6 to the metric audit cluster and adjust the opening paragraph's mapping accordingly.

### 5. Missing Bootstrap CIs in Table 7 (L0 Phase Transition)

Table 7 reports "---" for all four 95% CI values, despite the opening paragraph promising "Bootstrap confidence intervals (CIs) use 10,000 resamples with seed 42 throughout." The text below the table claims "bootstrap CIs at each L0 do not overlap," but the table does not show them. Figure 3's caption mentions "Error bars are bootstrap 95% CIs," suggesting the CIs exist but were not tabulated.

**Action**: Add bootstrap CIs to Table 7 or explain why they are omitted (e.g., "see Figure 3 for CIs").

### 6. Notation Inconsistency: "L0" vs. "$L_0$"

The glossary (glossary.md, line 15) mandates: "Always typeset as $L_0$, not L0, in the manuscript body." Several instances in experiments.md use raw "L0" in figure captions and inline text:
- Line 52: "At $L_0 = 22$" (correct)
- Line 99: "$L_0$" (correct in heading)
- Line 151: "L0 = 82" and "L0 = 22" in Section 4.10 heading and body

While most instances are correct, a careful pass is needed to catch all raw "L0" occurrences, especially in figure captions (line 103: "L0 Phase Transition", line 155: "L0 = 82 and L0 = 22").

**Action**: Replace all bare "L0" with "$L_0$" in the manuscript body and figure captions.

### 7. Cross-Architecture Section (4.7 subsection) Lacks Sufficient Context

The "Cross-Architecture Context (Confounded)" subsection at lines 122--124 reports GPT-2 Small absorption rates of 61.65%--67.29% without specifying which L0 values were used for those SAEs. The Method section (method.md) notes GPT-2 Small L1-ReLU SAEs have "--" for L0 in Table 1, implying the L0 is not directly comparable. This should be stated explicitly in the experiments section, not just in the methodology.

**Action**: Add a note such as "(L0 not directly comparable; L1-ReLU SAEs use continuous penalty rather than hard thresholds)" to the cross-architecture subsection.

---

## Minor Issues (Nice to Fix)

### 8. Table 2 Caption Refers to "Untrained SAE Control (C4)" as "Not Shown"

The caption says "The untrained SAE control (C4) produces 0% absorption in all domains (not shown)." Since this is a key validation result, consider adding it as a row in the table or as a footnote value rather than relegating it to parenthetical.

### 9. Twelve Letters Listed for Zero Strict Hedging (Line 70)

Line 70 lists "Ten letters (B, D, F, H, J, K, M, N, P, R, T, W) show zero strict hedging." Counting the listed letters gives 12, not 10. This is a factual error.

**Action**: Change "Ten letters" to "Twelve letters" or verify which letters are correct.

### 10. Missing First Use Expansion of Abbreviations

The glossary mandates expanding abbreviations on first use per section. The experiments section uses "SAE" (line 3), "CMI" (line 1), "FN" (line 52), and "CI" (line 3) without expansion. "SAE" is expanded as "Sparse Autoencoders (SAEs)" only implicitly via "JumpReLU Sparse Autoencoders (SAEs)." CMI is never expanded in this section.

**Action**: Add "(Conditional Mutual Information)" after the first use of "CMI" in the section, and ensure all other abbreviations are expanded on first use.

### 11. Inconsistent Shuffled-Label Rate Between Sections

Section 4.1 reports the shuffled-label control for first-letter as 59.6% (Table 2, line 11). The Introduction reports it as 74.6%. Section 4.3 references 74.6%. The 59.6% in Table 2 may be the domain-specific (first-letter only at L0=82) rate, while 74.6% may be the aggregate or from a different L0/threshold. This discrepancy needs explicit reconciliation.

**Action**: Add a clarifying note to Table 2 or the surrounding text explaining which configuration produces 59.6% vs. 74.6%.

### 12. Figure References Inconsistent with Outline's Figure Plan

The outline assigns "Figure 3" to the L0 Phase Transition (Section 5 in the outline) and "Figure 4" to CMI vs. Absorption (Section 6). The experiments section also uses Figure 3 for L0 Phase Transition and Figure 4 for CMI, which is consistent with the outline numbering but inconsistent with the section placement (everything is in Section 4 here).

### 13. The Word "Hedging" Should Be Expanded on First Significant Use

While the glossary defines hedging, the experiments section first uses it in Section 4.4 without a brief inline definition. A reader who skipped the methodology might benefit from a one-clause reminder: "hedging---information spreading across many SAE latents due to insufficient capacity."

### 14. Activation Patching Section Could Benefit from a Figure

Section 4.5 describes three distinct patching methods and their results but has no visual. A small schematic or bar chart showing 0/8 recovery for all three methods would strengthen this pivotal result visually.

---

## Cross-Section Consistency Checklist

| Item | Experiments | Intro | Method | Discussion | Conclusion | Status |
|------|:-----------:|:-----:|:------:|:----------:|:----------:|:------:|
| First-letter absorption at L0=82 | 13.4% / 14.39% | 15.96% | -- | 15.96% | 15.96% | INCONSISTENT |
| Shuffled rate (first-letter) | 59.6% | 74.6% | -- | 74.6% | 74.6% | INCONSISTENT |
| Strict hedging rate | 6.2% | 6.2% | -- | 6.2% | 6.2% | OK |
| Activation patching result | 0/8 | 0/8 | -- | 0/8 | 0/8 | OK |
| CMI rho at L0=82 | -0.383 | -0.383 | -- | -0.383 | -0.383 | OK |
| CMI rho at L0=22 | 0.044 | 0.044 | -- | 0.044 | 0.044 | OK |
| L0 phase transition rates | 42.85%/37.49%/14.39%/0.84% | same | -- | same | same | OK |
| Persistent core words count | 8 | 8 | 8 | 8 | 8 | OK |
| FN count at L0=22 | 656 | 656 | -- | -- | 656 | OK |
| Bootstrap parameters | 10k, seed 42 | -- | 10k, seed 42 | -- | -- | OK |
| Section numbering structure | 4.1--4.10 | 4/5/6 | -- | 4/5/6 | 4/5/6 | INCONSISTENT |

---

## Notation Consistency with notation.md

| Symbol | notation.md | experiments.md | Status |
|--------|:-----------:|:--------------:|:------:|
| $L_0$ | Correctly defined | Mostly correct, some bare "L0" in captions | MINOR FIX |
| $\tau_{\cos}$ | Defined | Used correctly | OK |
| $\tau_{\text{mag}}$ | Defined | Used correctly | OK |
| $\rho_s$ | Spearman rank correlation | Used correctly throughout | OK |
| CMI / $I(X; f_p \mid f_c)$ | Defined | CMI used without first-use expansion | MINOR FIX |
| Cohen's $d$ | Defined | Used correctly | OK |
| CV | Defined | Used correctly | OK |
| $d'$ | Subspace dimension | Used correctly | OK |
| $k$ | Sparsity of probe, $k=5$ | Used correctly | OK |

---

## Strengths

1. **Rigorous statistical reporting**: Every claim is accompanied by effect sizes, p-values, confidence intervals, and multiple comparison corrections. The Bonferroni corrections are applied transparently.

2. **Layered evidence structure**: The progression from observational (control failure) to structural (candidate explosion) to parametric (threshold sensitivity) to causal (activation patching) builds a compelling argument.

3. **Transparent negative results**: The CMI non-replication is reported with full statistical detail, including the dimension sensitivity analysis that further undermines the original finding. The explicit labeling of post-hoc removal as "cherry-picking" (Section 4.9, line 147) is commendable.

4. **Pre-registered elements clearly identified**: The $d' = 10$ subspace dimension is noted as pre-registered, and the distinction between pre-registered and exploratory analyses is maintained.

5. **Multiple control conditions**: The four-control suite (C1--C4) is well-designed and the universal control failure is convincingly demonstrated across five domains.

---

## Scoring Breakdown

| Criterion | Score (1-10) | Notes |
|-----------|:---:|-------|
| Completeness | 8 | All experiments from outline are present; some missing details (CIs in Table 7) |
| Accuracy | 6 | Absorption rate and shuffled-label rate inconsistencies across sections; "ten letters" counting error |
| Clarity | 7 | Well-organized within subsections; the monolithic structure reduces navigability |
| Notation consistency | 7 | Mostly consistent with notation.md; minor L0 and CMI expansion issues |
| Cross-section consistency | 5 | Section numbering and key numerical values conflict with Introduction, Discussion, Conclusion |
| Statistical rigor | 9 | Excellent reporting of CIs, effect sizes, corrections, controls |
| Figure/table quality | 7 | Tables are informative; missing CIs in Table 7; no figure for activation patching |
| **Overall** | **7** | Strong empirical content undermined by structural mismatch and numerical inconsistencies |

---

## Priority Ranking of Fixes

1. **(Critical)** Resolve section structure mismatch: split into Sections 4/5/6 per the outline, or update all cross-references in Introduction, Discussion, and Conclusion.
2. **(Critical)** Reconcile first-letter absorption rate (13.4% vs. 14.39% vs. 15.96%) and shuffled-label rate (59.6% vs. 74.6%) across all sections.
3. **(Critical)** Fix table numbering discontinuity (Table 5 -> Table 7, with Table 6 appearing later).
4. **(Major)** Add bootstrap CIs to Table 7 or cross-reference Figure 3.
5. **(Major)** Fix the "Ten letters" count error (should be twelve).
6. **(Major)** Ensure all L0 instances use $L_0$ notation per glossary mandate.
7. **(Minor)** Expand CMI and other abbreviations on first use in this section.
8. **(Minor)** Reconcile or annotate the different absorption rate denominators (all 25 letters vs. quality-gated subset).
9. **(Minor)** Add C4 results to Table 2 as a row rather than parenthetical.
10. **(Minor)** Add inline hedging definition on first use in Section 4.4.

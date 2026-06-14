# Critique: Conclusion

## Summary Assessment
The conclusion is crisp, technically dense, and structurally disciplined — it mirrors the three-contribution framing of the introduction and delivers concrete numbers throughout. At 270 words it is slightly over the 150-word target but fits well within the 0.25-page budget. The principal weaknesses are: (1) a discrepancy between what the conclusion lists as future-work items and what the outline specified; (2) one overclaimed phrase that invites reviewer challenge; and (3) a cross-section inconsistency with the Discussion section regarding the citation needed for "EDA" and "\citeauthor{eda2025}".

## Score: 8/10
**Justification**: The section does its job well — numbered contributions match reported data, numbers are correct, future work is specific and actionable, and the final "core message" sentence is the right 25-word send-off. To reach 9/10 it needs: the overclaim fixed, the future-work list aligned with the outline, and a citation for the EDA baseline. To reach 7 would require the current issues to be left unaddressed.

---

## Critical Issues

*(None found — no central claims are unsupported or factually wrong.)*

---

## Major Issues

### Issue 1: Future-work list does not match outline commitments
- **Location**: "Future work" bullet paragraph
- **Quote**: "(3) testing whether absorption generalizes to non-spelling-task feature hierarchies using probes trained on the target model; and (4) empirically validating whether masked regularization \citep{narayanaswamy2026masked} achieves measurable absorption reduction consistent with our H2 falsification."
- **Problem**: The outline (Section 8, "Future work") specifies three items: "cross-hierarchy absorption measurement; functional recovery validation; safety attribution analysis." The conclusion replaces "functional recovery validation" and "safety attribution analysis" with "obtaining a larger gold-label dataset" and "resolving the hook confound" — items that are limitations rather than future directions. The outline's "safety attribution analysis" (H5 from the proposal) is entirely absent from the conclusion, yet it is the highest-stakes downstream implication of the work.
- **Fix**: Either (a) update the conclusion's future-work list to include safety attribution analysis as item (4) or (5), or (b) revise the outline to reflect that hook-confound and label-size are the paper's actual stated future directions. The mismatch will be noticed by a reviewer who reads outline → conclusion carefully.

### Issue 2: "significantly outperforming EDA" — missing citation and context
- **Location**: Second bullet (Methodological), sentence beginning "significantly outperforming EDA"
- **Quote**: "significantly outperforming EDA (DeLong $p = 0.0012$)"
- **Problem**: "EDA" is used without a citation here. In the Discussion section (`discussion.md`), EDA is cited as "\citeauthor{eda2025}" — a placeholder that itself lacks a full reference. The conclusion introduces EDA as a known baseline without defining it or citing it. A reader encountering the conclusion first (as reviewers sometimes do) will not know what EDA is. Additionally, the p-value is stated but does not appear in the experiments section's Table 1 or in the body of Section 4.1 — it appears in the Table 1 prose only. Ensure the citation is consistent across sections.
- **Fix**: Add a citation for EDA at its first mention in the conclusion (e.g., "\citep{chanin2024absorption}" or whichever paper introduced EDA). If EDA is the authors' own prior contribution from iter_001/iter_002, mark it as such (e.g., "our prior EDA baseline"). Cross-check that `discussion.md`'s `\citeauthor{eda2025}` placeholder is resolved before submission.

---

## Minor Issues

- **"first direct empirical support"** (Mechanistic bullet): This is an overclaim. Tang et al. (2512.05534) already provides theoretical and some empirical support for the sparsity landscape account. The conclusion's phrasing could be challenged: the OMP experiment provides the first *controlled experimental* support, not the first *empirical* support. Fix: change to "provides the first controlled experimental evidence for the sparsity landscape account."

- **Jaccard co-occurrence — missing AUPRC context**: The Methodological bullet cites AUROC = 0.721 for O_jaccard but omits AUPRC = 0.075 (13.9× enrichment), which was the stronger signal for practical use (Section 4.2 in `experiments.md`). Given the tiny positive class (n_pos=18), AUPRC is arguably more informative than AUROC. Fix: consider adding "(AUPRC = 0.075, 13.9× enrichment at Precision@50)" to give the metric proper context for the class-imbalanced detection setting.

- **"The remaining 33% require training-time structural changes"** (Practical bullet): This is technically correct but slightly overstated. The experiments (Section 4.4) note the hook confound — the 33%/67% split is across different hook points (resid_pre vs. resid_post), so it may partially reflect activation-space differences rather than genuine coverage gaps. The conclusion presents this as a clean result without the caveat. Fix: append "(subject to hook-confound caveat, Section 5.2)" or fold the caveat into the sentence.

- **Word budget**: The section runs approximately 270 words against a 150-word target. The "Future work" paragraph is 103 words on its own. Given this is a workshop paper, the longer form may be appropriate, but if space is tight, the future-work paragraph could be trimmed to 2 items rather than 4.

---

## Visual Element Assessment
- [x] No figures/tables are required in the Conclusion — this check is N/A for this section.
- [x] No references to visual elements appear; none are needed.
- [x] No text-heavy sections that unexpectedly require visual support.

---

## What Works Well

1. **Numbered-contribution structure mirrors the introduction exactly.** The Mechanistic / Methodological / Practical triplet maps precisely to the introduction's contribution list with matching numbers, creating the closing loop a reviewer expects. The section delivers this without needing to re-read anything.

2. **Every number in the conclusion is traceable to a specific experiment.** AUROC 0.757–0.837, DeLong p=0.0012, 67%/33% split, ρ=0.044 — all appear in the experiments section with supporting tables. There are no round numbers or inflated claims.

3. **The final "core message" sentence is precisely scoped.** "Feature absorption in SAEs is primarily a training problem, and the interpretability community's diagnostic and remediation tooling should be oriented accordingly." This is specific, actionable, and stays within what the evidence actually supports ("primarily a training problem" rather than "entirely a training problem"). Well calibrated.

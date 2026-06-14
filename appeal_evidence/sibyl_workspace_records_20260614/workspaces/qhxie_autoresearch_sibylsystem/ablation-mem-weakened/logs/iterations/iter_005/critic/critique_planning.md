# Critique: Planning

## CP-1: Planning-Methodology Mismatch (CRITICAL)

The `plan/methodology.md` exists in this workspace, but it describes a different set of hypotheses (H1-H5, H7, H8) than what appears in either proposal.md (H1-H8) or writing/paper.md (H6-H10). The hypothesis numbering is inconsistent across documents:

| hypothesis | proposal.md | methodology.md | paper.md |
|---|---|---|---|
| Steering degradation | H1, H1b | H1 | H1 (mentioned) |
| Probing degradation | H2 | H2 | not prominent |
| Cross-layer consistency | H3 | H3 | not prominent |
| EC50 correlation | H4 | H4 | not prominent |
| Precision-recall asymmetry | H5 | H5 | H7 |
| Graph prediction | H6 | H6 | H6 (falsified) |
| Trained < random | H7 | H7 | not prominent |
| Optimal compression | H8 | H8 | not prominent |
| Homeostatic rebalancing | not in proposal | not in methodology | H10 (not executed) |

This means there is no single document that accurately describes what hypotheses were tested, which were supported, and which were falsified. The planning phase generated a methodology that was then not followed by the paper.

**Severity**: Critical

---

## CP-2: Critical Experiments Not Executed (CRITICAL)

The paper claims four contributions (Section 1.4 of paper.md):
1. LCA-SAE connection -- theoretical, valid
2. Local inhibition graph -- built, but H6 falsified
3. Mechanistic explanation for precision-recall asymmetry -- supported by H7
4. Homeostatic rebalancing -- **not executed**

Contribution #4 was planned, proposed, and claimed as a contribution, but never executed. The planning phase did not enforce that the experiment be completed before the contribution was claimed. This is a planning failure: the plan did not distinguish between "planned" and "executed."

**Severity**: Critical

---

## CP-3: Planning Stagnation Warning Ignored (MAJOR)

The methodology.md contains a warning: "PLANNING STAGNATION WARNING: Zero experiments or revisions for 2+ consecutive iterations." This warning appeared in the planning document but was not acted upon. The result is a paper with:
- Multiple unfinalized framings
- One experiment not executed but claimed
- No cross-model validation
- Statistical inconsistencies across documents

The planning phase should have recognized that all experiments were complete and triggered a consolidation/writing phase rather than continuing to iterate on the ideation.

**Severity**: Major

---

## CP-4: Cross-Model Validation Marked Completed But Is Not (MAJOR)

Both proposal.md and methodology.md list "Cross-model (Pythia-70M)" as "Completed" in the evaluation benchmarks table. However, the methodology.md describes the result as "Inconclusive; limited feature overlap." Being inconclusive is not the same as completed. The cross-model validation should have been marked as "inconclusive" or "partial" in the status column, not "Completed."

This mislabeling creates a false impression that cross-model validation was successful when it was not. A reviewer asking about cross-model validation would find that the results are not usable.

**Severity**: Major

---

## CP-5: Expected Visualizations Not Produced (MINOR)

The methodology.md lists 6 figures and 2 tables expected for the paper:
- Figure 1: Absorption rate distribution
- Figure 2: Steering success vs absorption rate
- Figure 3: Precision-recall decomposition
- Figure 4: Inhibition graph precision@k
- Figure 5: Random SAE vs trained SAE absorption
- Figure 6: [implied but not listed]
- Table 1: Hypothesis testing summary
- Table 2: Feature-level data

No figure files are present in the workspace. The planning document lists expected outputs that were never produced. This suggests the planning phase planned figures that were never generated from the experimental data.

**Severity**: Minor

---

## CP-6: NO-GO Branches Not Followed Through (MINOR)

The methodology.md lists two NO-GO branches:
1. H9 (co-occurrence tautology) -- correctly identified as NO-GO
2. Encoder-correlation-based absorption prediction -- marked as "NOT planned"

But the paper (writing/paper.md) proposes encoder-correlation-based analysis in Section 3.3 when discussing why H6 failed and mentions that future work should explore "larger k, adaptive neighborhood sizes, or context-dependent edge weighting." This is essentially the encoder-correlation idea without explicitly calling it that.

The NO-GO decision was not enforced or revisited when the paper's writing phase needed additional positive results.

**Severity**: Minor

---

## Summary

The planning phase has two critical failures:
1. **No single source of truth for hypotheses** -- the three hypothesis sets (proposal, methodology, paper) are inconsistent
2. **No execution tracking** -- homeostatic rebalancing was planned and claimed as a contribution but never executed

The planning stagnation warning was correct: the writing-review loop was running without experimental input, producing documents that are internally inconsistent. The planning phase should have triggered a consolidation step to resolve inconsistencies before writing continued.

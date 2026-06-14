# Critique of Experiments

**Score:** 8/10

## Strengths

- The section reports the decisive comparisons directly and keeps the sham-control logic central.
- Table 1 and the two planned charts are appropriate and well matched to the paper's claims.
- The MBPP subsection avoids overclaiming and uses failure modes in the right diagnostic role.

## Issues

1. **Major — Paragraph 2:** The batch-probe disclosure is honest, but it raises an immediate reviewer question: why were the audited runs executed at batch size 8 if the setup probe found a safe ceiling of 28? Without a short explanation, the reader may suspect uncontrolled implementation drift.  
   **Suggestion:** Add one sentence clarifying whether batch size 8 was inherited from the already-completed pilot, chosen for reproducibility parity, or constrained by another unreported factor in the packaged run.

2. **Major — Paragraph 3 / Table 1:** The table mixes per-dataset accuracy with total correct counts across both datasets, which is mathematically valid here but slightly awkward rhetorically.  
   **Suggestion:** Either add a note that the total is over the full 100-sample audited slice, or split the table into reasoning-side and code-side panes so the reader does not mentally compare unlike aggregates.

3. **Minor — Paragraph 4:** The term "real" in "the localized signal is therefore real" is understandable, but it may still invite pushback because the section immediately denies validated efficacy.  
   **Suggestion:** Replace "real" with "auditable" or "present in the active-control comparison" to keep the wording ceiling fully aligned.

4. **Minor — Paragraph 6:** The MBPP failure-mode analysis is good, but a short pointer back to the risk-marker framing would strengthen the connection between reasoning-side targeting and code-side boundary evidence.  
   **Suggestion:** Add one sentence noting that the risk-marker interpretation does not guarantee cross-task transport, which is exactly why MBPP is discussed as a boundary.

## Overall recommendation

This is the strongest section in the current draft. The remaining work is mostly about removing small sources of reviewer confusion, especially the safe-batch-size versus executed-batch-size discrepancy and the wording around what kind of signal the experiments actually establish.

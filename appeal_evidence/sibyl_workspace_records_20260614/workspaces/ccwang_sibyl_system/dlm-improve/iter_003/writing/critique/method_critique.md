# Critique of Method

**Score:** 7/10

## Strengths

- The section clearly defines the audited slice and the roles of the four arms.
- The paper object is protected by an explicit claim policy, which is unusually strong and appropriate for this project.
- The outcome-decomposition equation is simple and well aligned with the packaged artifacts.

## Issues

1. **Major — Paragraph 1:** The section explains that the slice is audit-oriented rather than benchmark-representative, but it does not spell out the selection bias implications. A reviewer may ask whether the entropy-stratified slice exaggerates or suppresses the intervention effect.  
   **Suggestion:** Add one sentence stating that the slice is designed for bounded interpretive auditing, not for estimating population-level benchmark accuracy, and that all claims are restricted accordingly.

2. **Major — Paragraph 3 / Figure 1 reference:** The method text references Figure 1, but only a description artifact exists at this stage. That is acceptable for the pipeline, yet the prose could do more to describe the figure's logic inline in case the rendered diagram is delayed.  
   **Suggestion:** Add a short sentence after the Figure 1 caption explaining that the sham-control stage is the point at which the positive controller interpretation fails.

3. **Minor — Paragraph 4:** The runtime-contract paragraph is careful, but the compile disclosure may still confuse readers because setup logged a compile attempt while the arm metrics report no compiled execution.  
   **Suggestion:** Rephrase the final sentence to make the distinction explicit: compile was attempted during setup, but the reported experimental arms remained on the eager path.

4. **Minor — Paragraph 6:** The claim-policy bullets are strong, though the trajectory add-on statement may feel abrupt to readers who have not seen the internal planning artifacts.  
   **Suggestion:** Add a brief phrase such as "to prevent rescue narratives from widening the claim beyond the audited slice."

## Overall recommendation

The method section is structurally solid and unusually transparent. The next revision should mainly sharpen the selection-bias caveat and make the runtime/figure logic even more reviewer-proof.

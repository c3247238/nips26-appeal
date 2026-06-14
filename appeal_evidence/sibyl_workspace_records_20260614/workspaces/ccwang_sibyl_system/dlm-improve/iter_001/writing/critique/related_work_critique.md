# Section Critique: Related Work

**Reviewer**: main control plane
**Date**: 2026-03-12
**Score**: 7.5/10

## Summary

The section is aligned with the new iteration-1 paper identity. It no longer treats the manuscript as a controller paper and instead positions it as a diagnostic study about comparison protocol, observer-controller factorization, and task dependence. That is the correct high-level move. The remaining issues are mostly about scholarly precision and taxonomy clarity rather than framing.

## Major Issues

### M1: Citation anchors are still too implicit

The prose names several families of work, but the section still reads more like an informed narrative than a submission-ready related-work section. Before integration into LaTeX, each cluster should receive explicit citation anchors, especially for LLaDA, Dream, Fast-dLLM-style acceleration, and revision/remasking methods.

### M2: The acceleration / revision / calibration split could be one sentence sharper

The section already separates these threads, but the transition into the paper's contribution would be stronger if it explicitly said that prior work typically optimizes **within** one of these threads, while this paper audits the assumptions shared **across** them.

## Minor Issues

### m1: The final paragraph should reference the benchmark-role hierarchy

The section would connect more cleanly to the rest of the draft if it mentioned that the paper's evidence is organized by benchmark roles: GSM8K for headline comparison, MATH500 for transfer, and HumanEval for boundary testing.

### m2: A short caveat on concurrent work would help

Some of the families discussed are current enough that the final integrated draft should likely use wording such as “recent” or “concurrent” rather than sounding like a settled historical taxonomy.

## Strengths

1. The section is now consistent with the diagnostic-study framing.
2. It correctly repositions calibration as a diagnostic object rather than the basis of a new controller.
3. It makes the paper's novelty methodological instead of algorithmic.

## Recommended Revisions

1. Add concrete citation anchors when integrating references.
2. Add one explicit sentence connecting prior work categories to the paper's benchmark-role design.
3. Tighten the final paragraph so the paper's novelty is stated as “evaluation protocol + interpretation discipline,” not just “difference in emphasis.”

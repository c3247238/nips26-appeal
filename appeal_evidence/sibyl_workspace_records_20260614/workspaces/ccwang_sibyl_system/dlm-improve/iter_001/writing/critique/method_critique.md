# Section Critique: Method / Protocol

**Reviewer**: main control plane
**Date**: 2026-03-12
**Score**: 8/10

## Summary

This section does the most important conceptual work in the revised paper: it redefines “Method” as an evaluation protocol rather than a new decoding algorithm. That is the right structural choice for the current evidence. The section is clear, disciplined, and well aligned with the rewritten Introduction and Experiments. The main remaining issues are about concreteness and figure integration.

## Major Issues

### M1: “Figure X” should become a concrete figure reference during integration

The section correctly includes a protocol flow artifact, but the prose still uses a placeholder label. Before the integrated draft is assembled, the paper should assign a stable figure number so the section reads like a completed manuscript rather than a staging draft.

### M2: The observer/controller definitions would benefit from one explicit example

The section defines the distinction well, but it would be even stronger if it added one sentence such as: “Calibration is treated only as an observer in this study, whereas entropy-guided and instability-guided revision instantiate controller families.” The content is already implied; making it explicit would reduce reader effort.

## Minor Issues

### m1: The failure taxonomy is intentionally minimal, but should say so

The code taxonomy currently names syntax failure and runtime failure. Adding “at minimum” or “in this diagnostic slice” would make it clearer that this is a deliberately scoped taxonomy, not a claim of exhaustiveness.

### m2: The non-claims subsection is a strength and should stay concise

It already prevents overclaiming effectively. Avoid expanding it too much in later drafts; its current compactness is part of why it works.

## Strengths

1. The section fully supports the new paper identity.
2. Honest-compute variables are clearly listed and easy to operationalize.
3. The benchmark-role hierarchy is clean and supports the later task-dependent analysis.
4. The non-claims paragraph is appropriately cautious and reviewer-friendly.

## Recommended Revisions

1. Replace “Figure X” with a stable figure number in the integrated manuscript.
2. Add one explicit observer/controller example sentence.
3. Add a tiny qualifier that the code failure taxonomy is minimal rather than exhaustive.

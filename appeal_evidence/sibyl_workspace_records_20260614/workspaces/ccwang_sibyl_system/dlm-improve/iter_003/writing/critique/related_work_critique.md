# Critique of Related Work

**Score:** 6/10

## Strengths

- The section groups the literature into meaningful themes instead of listing papers one by one.
- It clearly distinguishes DLM inference engineering, uncertainty-guided control, and audit-centric evaluation.
- The final positioning paragraph preserves the negative-case contribution margin and avoids method inflation.

## Issues

1. **Critical — Entire section:** The section mentions paper titles and years, but it does not provide citation anchors, shorthand labels, or any placeholder reference syntax. In a paper draft, this is a high-risk omission because Related Work is the first place reviewers look for scholarly grounding.  
   **Suggestion:** Introduce citation placeholders consistently, for example with author-year markers or temporary keys that the later integration stage can map into the bibliography.

2. **Major — Paragraphs 1-3:** The prose explains how neighboring literatures differ from the present paper, but it does not explicitly identify which gap is new enough to justify publication. The answer is implied rather than argued.  
   **Suggestion:** Add one short synthesis paragraph stating that the gap is not "better DLM control" but "a reviewer-facing audit template showing how a sham control rewrites a small-gain interpretation."

3. **Major — Paragraph 2:** The uncertainty literature is summarized carefully, but the bridge from predictive usefulness to intervention invalidity could be sharper.  
   **Suggestion:** Add one sentence that explicitly states: predictive observer signals can coexist with null sham-control separation, which is exactly the failure mode documented in this paper.

4. **Minor — Paragraph 4:** The survey discussion is useful, but it could better differentiate survey-style taxonomy from protocol-style contribution.  
   **Suggestion:** Tighten the last two sentences so the contrast between "cataloging DLM methods" and "auditing DLM claims" becomes more memorable.

## Overall recommendation

This section has the right conceptual structure but still reads more like a literature memo than a submission-ready related-work section. Citation scaffolding is the main blocker, and the contribution gap should be made more explicit in one final synthesis move.

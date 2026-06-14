# Section Critique: Introduction (intro.md)

**Reviewer**: sibyl-section-critic
**Date**: 2026-03-12
**Score**: 8/10

---

## Summary

The Introduction is already much closer to the right paper than the old CARD framing. It clearly presents the project as a compute-normalized diagnostic study rather than a new controller paper, and it stays consistent with the rewritten Related Work, Protocol, Experiments, Discussion, and Conclusion sections on the manuscript's three core claims: honest compute, observer-controller mismatch, and task-dependent revision response.

The main remaining weakness is not factual inconsistency but emphasis. Paragraphs 3-5 read like a compressed results section, with many method names and exact numbers arriving before the section fully settles the study's scope and conceptual vocabulary. That makes the framing slightly less clean than the outline intends. A few targeted edits would make the section sharper, easier to scan, and more obviously diagnostic-study-first.

---

## Issues

### CRITICAL

*None.*

### MAJOR

**Issue M1: The result preview is too dense, which blurs the diagnostic-study framing**

- **Location**: paragraphs 3-5, especially [intro.md line 7](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L7), [intro.md line 9](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L9), and [intro.md line 11](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L11).
- **Problem**: The outline asks the Introduction to preview three claims, but the current draft goes further and starts reporting a large amount of method-level evidence: six method names, multiple exact accuracies, multiple latencies, and several audit metrics. The result is accurate, but it makes the Introduction feel partially like Section 4 before the paper has finished establishing its scope. This is slightly at odds with the cleaner division used later in [method.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/method.md) and [experiments.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/experiments.md), where the protocol and the evidence are separated more crisply.
- **Why it matters**: For a diagnostic-study paper, the Introduction should primarily teach the reader what is being diagnosed and why that diagnosis requires a different evaluation protocol. Right now, the numeric detail competes with that message.
- **Suggestion**: Keep one anchor example per claim, but compress the rest. For example:
  - keep the `CORE-proxy-64` vs. `Entropy-Revise-64+3` runtime contrast as the honest-compute teaser;
  - restate the observer-controller point in plain language with at most one metric pair;
  - keep the HumanEval boundary fact as the main negative-result teaser;
  - add one scope-setting sentence before the preview, such as: "We study a six-method training-free shortlist across GSM8K, MATH500, and HumanEval to separate comparison hygiene, signal diagnosis, and task-boundary behavior."

**Issue M2: The observer-controller argument introduces metric language before the reader has enough conceptual scaffolding**

- **Location**: paragraph 4 at [intro.md line 9](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L9).
- **Problem**: The Introduction's most distinctive claim is the observer-controller split, but the paragraph presents it in audit jargon: "diagnostic score," "deployed control gain," "lightweight screen," and "diagnostic association." Those terms are defined much more clearly later in [method.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/method.md#L35), where observers and controllers are formalized. In the Introduction, however, the reader has not yet been taught that vocabulary, so the paragraph lands more as a metric dump than as a conceptual payoff.
- **Why it matters**: This is one of the paper's central contributions. If the first explanation feels technical before it feels intuitive, the paper risks underselling what is actually novel about the framing.
- **Suggestion**: Rewrite the paragraph in more reader-facing language, then leave the score terminology to Section 3/4. For example: "Calibration is the best signal for identifying likely errors, but that does not make it a useful revision policy. Entropy remains informative but gives little extra gain once deployed. Instability motivates a controller, yet shows weak evidence that it is the right signal to control on." That version would still be consistent with the later tables while making the concept easier to absorb.

### MINOR

**Issue m1: The Introduction should hedge the study scope slightly more explicitly**

- **Location**: paragraph 6 at [intro.md line 13](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L13).
- **Problem**: The section correctly calls the paper a "compute-normalized diagnostic study," but it does not briefly signal that the evidence is still a focused shortlist-based diagnostic package rather than a broad benchmark sweep. Later sections do this more carefully, especially [experiments.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/experiments.md#L5) and [discussion.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/discussion.md#L17).
- **Suggestion**: Add one modest hedge in the contributions or setup sentence, e.g., "Across a focused six-method shortlist and three benchmark roles, we make three claims..." This would better align the Introduction with the manuscript's later caution.

**Issue m2: The opening field overview needs citation placeholders or anchors**

- **Location**: paragraph 1 at [intro.md line 3](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L3) and paragraph 3 at [intro.md line 7](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L7).
- **Problem**: The Introduction names `LLaDA`, `Dream`, and several controller families, but the section currently has no citation markers at all. Because [related_work.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/related_work.md) now gives a clean taxonomy of prior work, the Introduction can stay light on literature detail, but it still needs at least a few anchor citations for the reader to trust the opening survey.
- **Suggestion**: Add citation placeholders after `LLaDA` and `Dream`, and a compact citation cluster when mentioning the representative training-free methods or controller families. Two or three anchor cites are enough here.

**Issue m3: Figure 1 is referenced, but the prose could do more to explain what the teaser actually contains**

- **Location**: paragraph 2 at [intro.md line 5](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L5), plus the figure comment at [intro.md line 22](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L22).
- **Problem**: The figure is correctly referenced before appearance, which is good. But the prose only says that Figure 1 "previews the three resulting takeaways." The outline's Figure 1 plan is more concrete: left panel for honest-compute reorder, middle panel for observer-controller gap, right panel for HumanEval boundary failure. A reader would benefit from one extra sentence mapping the visual panels to the three claims.
- **Suggestion**: Add a short clause such as: "Its three panels summarize the compute reorder, the observer-controller gap, and the code-boundary failure pattern." That would make the teaser figure work harder without adding clutter.

**Issue m4: The contribution list could foreground the evaluation-protocol contribution more explicitly**

- **Location**: paragraph 6 at [intro.md line 15](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L15).
- **Problem**: The list is accurate, but the paper's positioning in the outline is "diagnostic study + evaluation protocol + failure taxonomy." Right now, contribution 1 does mention an honest-compute comparison protocol, but the list overall still reads more like a set of empirical findings than a framing contribution. This is a mild mismatch with the outline's intended identity.
- **Suggestion**: Tighten the opening sentence before the list or slightly revise item 1 to say that the paper contributes an evaluation protocol for comparing revision methods under honest compute, then uses that protocol to surface observer-controller mismatch and task-boundary failure.

---

## Visual Communication Assessment

- **Figure 1 planned in the outline**: Yes. The outline specifies a three-panel teaser for compute reorder, observer-controller mismatch, and code-boundary failure.
- **Referenced before appearance**: Yes. The Introduction explicitly mentions Figure 1 in paragraph 2.
- **Assessment**: The teaser figure is the right visual for this section and matches the paper's current framing much better than a method pipeline would. The only missing piece is a slightly more explicit panel map in the prose so the figure does explanatory work, not just promotional work.
- **Additional visual need**: None. The Introduction does not need another figure or table as long as Figure 1 is fully integrated.

---

## Positive Strengths

1. **The paper identity is now correct.** The Introduction consistently frames the manuscript as a diagnostic study rather than a new hero-method paper, matching [related_work.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/related_work.md), [discussion.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/discussion.md), and [conclusion.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/conclusion.md).
2. **The three-claim structure is strong.** Honest compute, observer-controller mismatch, and task dependence are introduced in a clear order and then mirrored by the rest of the manuscript.
3. **The negative-result framing is honest and useful.** Bringing the HumanEval failure into the Introduction is the right move for this paper; it builds credibility and prevents the story from sounding selectively positive.
4. **The section roadmap is aligned with the current manuscript order.** The closing paragraph correctly transitions to Related Work, Protocol, Experiments, Discussion, and Conclusion in the same structure used by the rewritten sections.

---

## Score Justification

**8/10**

This is a solid Introduction with the right thesis, the right paper identity, and good alignment with the rest of the rewritten manuscript. The score is held back mainly by presentation density: too many exact numbers and partially undefined audit terms arrive before the conceptual scaffolding is fully in place. If the authors compress the numeric preview, make the observer-controller claim more intuitive, add a few citation anchors, and let Figure 1 do slightly more explanatory work, this section should move comfortably into the 9/10 range.

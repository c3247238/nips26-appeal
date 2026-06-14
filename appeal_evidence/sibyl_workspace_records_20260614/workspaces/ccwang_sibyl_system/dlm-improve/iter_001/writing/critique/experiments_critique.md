# Critique: Experiments Section

**Reviewer**: Section Critic Agent (Codex)
**Date**: 2026-03-12
**Score**: 7.0 / 10

---

## Overall Assessment

This section is substantially aligned with the new iteration-1 framing. It reads like a compute-normalized diagnostic study rather than the older CARD-forward paper, and its three-part structure matches the thesis established in the outline, introduction, method, discussion, and conclusion: honest compute, observer-controller mismatch, and task-dependent revision response. The closing subsection is especially good at restating the paper's safe scope.

The main problems are not conceptual drift, but evidence presentation and calibration of claims. Several key points are asserted more strongly than the displayed tables support, the runtime-fairness metadata promised by the protocol is only partially shown, and the task-dependence evidence needs more explicit uncertainty framing given the small diagnostic slices. There is also a terminology/labeling issue in the HumanEval table that makes the cross-section narrative less clean than the rest of the draft.

---

## Issues by Severity

### MAJOR

**M1 — Table 1 does not fully deliver the honest-compute protocol promised by the outline and method section**

Paragraph reference: lines 13-28, especially Table 1 at lines 15-24.

Cross-check:
- Outline, Figure & Table Plan, Table 1 (lines 177-183) promises accuracy, actual NFE, nominal NFE, latency, throughput, batch size, **backend**, and compile status.
- Method section 3.2 (lines 20-29) also defines backend as part of the honest-compute protocol.

The current Table 1 reports accuracy, nominal NFE, actual NFE, latency, tokens/s, batch, and compile, but it omits **backend**. That omission matters because the whole point of this paper's framing is that runtime configuration is part of the scientific claim, not a footnote. In addition, the prose emphasizes a concrete compute reorder between `CORE-proxy-64` and `Entropy-Revise-64+3`, but the table does not make compute ordering explicit, so the reader must infer the reorder mentally.

*Fix*: Expand Table 1 to include backend, and consider one compact column for nominal-rank vs actual-rank or a short note in the caption identifying the key reorder. That would make the section match the protocol already established in [method.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/method.md#L18) and the outline.

---

**M2 — The observer-controller mismatch subsection relies on evidence that is not visible in the displayed table**

Paragraph reference: lines 30-45, especially lines 42-44.

Table 2 reports `Diagnostic score`, `Control effectiveness`, and `Gap`, but the prose then makes two more specific claims that are not grounded in the visible display:

- entropy does not outperform random revision because both reach `0.37` accuracy;
- calibration is not a deployed control policy in the shortlist.

The first claim introduces a random-revision baseline that is not present in Table 2. The second is true, but the subsection never reminds the reader how `control effectiveness` is operationalized or why a signal with no deployed controller is scored as `0.0000` rather than "not applicable." As written, the logic is understandable if one has already internalized [method.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/method.md#L35), but the section asks the reader to trust an unseen comparison at the exact point where the paper's second headline claim is being made.

*Fix*: Either add a row or footnote for the random-revision screen baseline, or explicitly state in one sentence how `control effectiveness` is computed and why calibration receives a zero. If space is tight, the cleanest solution is to fold the random baseline into the Figure 3 caption and define the metric briefly in the text.

---

**M3 — The task-dependence subsection is slightly over-confident for a diagnostic slice and needs stronger uncertainty framing**

Paragraph reference: lines 46-71, especially lines 48, 59, and 71.

Cross-check:
- Setup (line 7) says GSM8K, signal audit, and MATH500 use `100` examples; HumanEval uses `50`.
- Discussion limits (lines 17-26 in [discussion.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/discussion.md#L17)) explicitly emphasize single-seed and small-slice limits.
- The outline guardrails (lines 17-21 and 130-138 in [outline.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/outline.md#L17)) ask for a modest diagnostic interpretation.

Despite that framing, the subsection uses strong formulations such as "It does not," "the result matters because it blocks a simple generalization," and "This is exactly the pattern we would expect." These are close to the right interpretation, but the tone is firmer than the evidence scale shown in the section. For iteration 1, the paper should sound a little more like "in the current diagnostic slice" and a little less like "we have fully established the boundary."

*Fix*: Add `n` to the Table 3 and Table 4 captions, and soften two or three sentences with phrasing such as "in this diagnostic slice," "under the tested setup," or "consistent with the interpretation that...". That would align the section more tightly with the caution already present in the introduction, discussion, and conclusion.

---

**M4 — Table 3 does not match the outline's planned evidence for task dependence**

Paragraph reference: lines 50-57.

Cross-check:
- Outline Table 2 plan (lines 201-207 in [outline.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/outline.md#L201)) calls for GSM8K and MATH500 accuracy, **actual NFE**, and ordering for the four main methods.

The current Table 3 gives only GSM8K and MATH500 accuracy. That is enough to show a ranking shift, but not enough to preserve the paper's compute-normalized framing across sections. Because the whole paper argues that runtime-fairness metadata changes interpretation, dropping actual NFE from the cross-dataset table makes this part of the section feel weaker and slightly inconsistent with its own protocol.

*Fix*: Add actual NFE for each method, or add a short "ordering" note in the caption if full rank columns feel too heavy. This is especially important because the subsection is meant to show that transfer changes even within reasoning, not merely that two columns have different winners.

---

**M5 — The HumanEval table uses ambiguous method labels that break consistency with the rest of the draft**

Paragraph reference: lines 61-69.

Every other section uses the shortlist names consistently: `Standard-64`, `Entropy-Revise-64+3`, and `TIGER-Instability-64+3`. Table 4 switches to:

- `Standard`
- `Entropy/TIGER Ungated Revision`
- `Gated TIGER`

This is understandable informally, but it leaves several questions unanswered:

- Is `Standard` the same as `Standard-64`, and should that be stated?
- Why are entropy and TIGER merged into one row here? Are they exactly tied, averaged, or pooled?
- Why is the gated variant only attributed to TIGER, while the ungated row appears to combine two controller families?

Because the section's job is diagnostic clarity rather than method marketing, this ambiguity is avoidable and should be fixed.

*Fix*: Rename the rows so they preserve the main-paper method vocabulary, and add one sentence explaining why entropy and TIGER are collapsed (if they truly are identical in this boundary test). If the row is an average or pooled condition, say so explicitly.

---

### MINOR

**m1 — The setup paragraph uses operational wording that sounds like a workspace note rather than paper prose**

Paragraph reference: line 7.

"Our primary evidence comes from the diagnostic assets produced in the current workspace" is too implementation-facing for the paper voice established in the rewritten sections. The introduction, method, discussion, and conclusion all read like manuscript prose; this sentence reads like an internal project memo.

*Fix*: Replace "current workspace" with something like "the current diagnostic study" or "the diagnostic runs used in this paper."

---

**m2 — Table captions need sample sizes to be self-explanatory**

Paragraph reference: lines 24, 40, 57, and 69.

The captions are concise and readable, but they omit `n`, even though sample size is one of the most important caveats in this draft. Since this is not a leaderboard paper, the captions should help the reader remember the evidence scale at the point of interpretation.

*Fix*: Add sample sizes directly to the captions, e.g., "Matched-compute comparison on GSM8K (`n=100`)" and "HumanEval boundary study (`n=50`)."

---

**m3 — The section introduces an extra result table that is not reflected in the current outline plan**

Paragraph reference: Table 2 at lines 34-40.

The current outline plans one GSM8K compute table, one cross-dataset task-dependence table, one appendix runtime table, and figures for the signal gap and code boundary. The experiments section adds a standalone signal-gap table that is not currently reflected in the Figure & Table Plan. The table itself is useful, but this creates planning drift between the outline and the draft.

*Fix*: Either update the outline to acknowledge this extra table, or absorb the same information into Figure 3/caption if you want to keep the visual plan tighter and reduce table count.

---

## Visual Communication Assessment

| Element | Status | Assessment |
|---------|--------|------------|
| Table 1 | Present | Strong core table, but missing `backend` and explicit compute-order signal |
| Figure 2 | Referenced before interpretation | Good |
| Table 2 | Present but unplanned in outline | Useful, though it creates plan drift |
| Figure 3 | Referenced before the table | Good, but the random baseline used in text is not visibly attached |
| Table 3 | Present | Needs actual NFE and sample size to match the outline's task-dependence plan |
| Figure 4 | Referenced before Table 4 | Good |
| Table 4 | Present | Boundary logic is strong, but row labels are ambiguous and `n` is missing |

The section does satisfy the high-level requirement that experiments use both tables and charts. Its visual problem is not absence, but incomplete alignment between what the visuals show and what the prose asks the reader to conclude.

---

## Positive Highlights

- The section is clearly aligned with the new thesis in [intro.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/writing/sections/intro.md#L3), not the older CARD-centered framing.
- The subsection order mirrors the paper's three claims cleanly and makes the section easy to follow.
- Lines 28 and 75 are especially strong because they state the paper's non-claims explicitly.
- The HumanEval interpretation is directionally right for this draft: it functions as boundary evidence, not as a second success story.

---

## Concrete Suggestions for Revision

1. Add `backend` to Table 1 and make the key compute reorder visible rather than only verbal.
2. Make the observer-controller metrics self-contained by exposing the random baseline or defining the control-effectiveness calculation in one sentence.
3. Add `n` to every experiment-table caption and slightly hedge the strongest task-dependence claims.
4. Expand Table 3 so it matches the outline's plan for cross-dataset ordering under compute-normalized framing.
5. Rename the HumanEval rows so they match the method family terminology used everywhere else.
6. Replace "current workspace" with manuscript-facing prose.
7. Update the outline if Table 2 is meant to stay as a standalone result table.

---

## Score: 7.0 / 10

**Justification**: The section already has the right scientific posture for iteration 1. It is focused, modest in its final claims, and consistent with the rewritten introduction, method, discussion, and conclusion. The score is held back because the evidence display is not yet fully aligned with the paper's own protocol: some runtime-fairness metadata is missing, one key observer-controller claim depends on off-table evidence, and the task-dependence subsection needs slightly better uncertainty signaling for a diagnostic-slice paper. These are fixable issues, and fixing them would materially improve the section without changing its core argument.

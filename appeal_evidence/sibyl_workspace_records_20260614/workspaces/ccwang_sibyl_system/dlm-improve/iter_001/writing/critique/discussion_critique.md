# Section Critique: Discussion (Section 5)

**Reviewer**: sibyl-section-critic  
**Date**: 2026-03-12  
**Score**: 7/10

---

## Summary Judgment

This Discussion section is directionally strong and, importantly, already reflects the new iteration-1 framing rather than the older CARD-centric story. It correctly narrows the paper to a diagnostic contribution, keeps the claims modest, and includes a useful limitations/next-steps structure. The main weakness is that the section often states the interpretation without re-anchoring it to the paper's concrete evidence, figures, and benchmark-role logic. As a result, the section is more credible than flashy, but not yet as analytically sharp or as tightly integrated with the rest of the manuscript as it should be.

---

## Strengths

**S1. The framing is now aligned with the paper's actual contribution.**  
Lines 3-4 correctly present the paper as "a diagnostic account" rather than a new controller paper. This is consistent with the outline's one-sentence thesis and with the current Introduction, Method, and Conclusion drafts.

**S2. The limitations are appropriately honest.**  
Lines 17-26 do not overclaim. The section explicitly names single-seed evaluation, slice sizes, incomplete benefit-bucket analysis, and the `CORE-proxy` approximation. That honesty fits the paper's intended tone.

**S3. The next-step package is sensible and focused.**  
Lines 28-36 prioritize uncertainty accounting and mechanism analysis instead of reopening a controller search. That is exactly the right direction for this diagnostic-study framing.

---

## Issue 1 — The synthesis is too detached from the paper's own quantitative evidence

**Severity**: Major  
**Location**: Lines 7, 11, 15-16, 26-36

**Problem**: The Discussion states the correct interpretations, but it rarely reattaches them to the specific evidence that earned those interpretations. For example:

- Section 5.1 discusses honest compute in general terms, but does not remind the reader of the key GSM8K facts from Table 1 and Figure 2: `CORE-proxy-64` keeps the best raw accuracy (`0.46`) while paying much higher latency (`482.95s`) than `Entropy-Revise-64+3` (`210.67s`) and shifting under actual-NFE ordering.
- Section 5.2 describes the observer/controller split, but does not restate the most important numerical asymmetry from Table 2 / Figure 3: calibration has the strongest diagnostic score (`0.6225`) but no deployed control gain, while instability is weak on both sides.
- Section 5.3 correctly names the code boundary, but does not explicitly remind the reader of the HumanEval tradeoff from Table 4 / Figure 4: syntax failure improves from `0.48` to `0.28`, while runtime failure worsens from `0.50` to `0.68`, and `pass@1` still remains below `Standard`.

Because this paper is selling a diagnostic interpretation rather than a new algorithm, the Discussion has to feel tightly evidence-coupled. Right now it reads a little too abstractly for that role.

**Suggestion**: Add one explicit quantitative reminder and one figure/table reference in each of §§5.1-5.3. The section does not need new visuals, but it should visibly synthesize the paper's existing ones.

---

## Issue 2 — The observer/controller claim needs tighter scoping to stay consistent with the rest of the paper

**Severity**: Major  
**Location**: Lines 9-11

**Problem**: Section 5.2 is mostly correct, but the phrasing is slightly broader than the paper's safer formulation elsewhere. The outline, Introduction, Method, and Conclusion all carefully say that the observer-controller mismatch is established **under the tested policies**. Here, the wording risks sounding closer to a general law:

> "Calibration is highly informative as an observer and still unconvincing as a controller. Entropy remains the most practical signal but is not transformative."

This is a little too compressed for the evidence actually presented. Calibration is not a deployed shortlist controller; its "controller weakness" is inferred from the audit and from the absence of a convincing policy instantiation, not from a head-to-head deployed calibration controller. Similarly, entropy being "not transformative" is true in the current screen/shortlist evidence, but should stay explicitly tied to those tested interventions.

**Suggestion**: Add the caveat directly inside §5.2 rather than leaving it implicit. For example: "Under the tested policies, calibration is strongest as an observer, while neither entropy nor instability yields a comparably strong controller."

---

## Issue 3 — The code framing is incomplete relative to the outline and other rewritten sections

**Severity**: Major  
**Location**: Lines 13-16

**Problem**: Section 5.3 correctly says that revision has a "recoverability boundary," but it stops short of one of the paper's planned Discussion jobs: explaining **why code belongs in the paper**. The outline says Discussion should make explicit that code is included as boundary evidence rather than as a second main success domain. The Introduction and Experiments sections already support that framing, but the Discussion never says it plainly.

That omission matters because HumanEval could otherwise look like a negative side-result rather than a designed stress test. The whole paper becomes cleaner if the Discussion explicitly tells the reader: code is in the main paper because it reveals the structural limit of local revision, not because the paper claims broad code gains.

**Suggestion**: Add one sentence at the end of §5.3 such as: "That is why HumanEval belongs in the main paper: not as a second headline domain, but as a structural stress test showing where local revision ceases to be safe."

---

## Issue 4 — The section does not fully exploit the benchmark-role hierarchy established earlier

**Severity**: Major  
**Location**: Lines 13-16, 26-36

**Problem**: The Method and Experiments sections establish a clear benchmark-role hierarchy:

- GSM8K = headline comparison hygiene
- MATH500 = transfer check
- HumanEval = structural boundary test

The Discussion gestures toward task dependence, but it does not explicitly reuse this hierarchy when synthesizing the results. As written, MATH500 and HumanEval are presented mainly as additional negative evidence. That is weaker than the paper's actual design, which is that each benchmark has a distinct interpretive role.

This missing synthesis leaves value on the table. The Discussion should explain that the paper is not just reporting "different datasets disagree"; it is showing that different benchmark roles reveal different failure modes of revision.

**Suggestion**: Add 2-3 sentences in §5.3 or at the start of §5.5 clarifying that GSM8K establishes the compute-normalized comparison, MATH500 tests reasoning transfer, and HumanEval probes recoverability under stronger structural constraints.

---

## Issue 5 — The future-work package is useful but too shorthand-heavy for a paper discussion

**Severity**: Minor  
**Location**: Lines 28-36

**Problem**: The priorities in §5.5 are good, but the prose sounds more like internal project planning than reader-facing discussion prose. Terms such as `benefit-bucket audit`, `seed-sensitivity spot-check`, and `asset-lineage tables` are understandable inside the workspace, but the section does not yet explain why each item is the highest-value follow-up for a reviewer or future reader.

For example, the benefit-bucket audit is the most important of the three, but the Discussion never says what it would resolve: whether revision mostly fixes draft-wrong cases, harms draft-correct cases, or simply redistributes errors.

**Suggestion**: Keep the three-item package, but gloss each item with one short rationale. That will make the future-work paragraph feel like scientific prioritization rather than a to-do list.

---

## Issue 6 — The section should name the paper's durable positive contribution more explicitly in the limitations block

**Severity**: Minor  
**Location**: Lines 17-26

**Problem**: The limitations subsection does a good job narrowing the claims, but it ends on what the paper cannot claim without restating what still survives. The risk is tonal: the section can leave the reader with the impression that the limitations mostly weaken the paper, rather than clarifying the narrower contribution that remains defensible.

This is especially important because the outline positions the contribution as a **diagnostic study + evaluation protocol + failure taxonomy**. The current Discussion names the diagnostic story well, but says less about the evaluation-protocol and failure-taxonomy contribution at exactly the point where skeptical readers will be asking what still stands.

**Suggestion**: After line 26, add one sentence such as: "Even with these constraints, the paper still supports a narrower contribution: a compute-normalized evaluation protocol and a task-structured failure account for revision in DLMs."

---

## Visual Element Check

**Status**: Partial pass

The outline does not plan a new figure or table for the Discussion section, so `<!-- FIGURES - None -->` is acceptable. However, the section should still reference the paper's existing visuals during synthesis:

- §5.1 should point back to Table 1 and/or Figure 2.
- §5.2 should point back to Table 2 and/or Figure 3.
- §5.3 should point back to Table 3, Table 4, and/or Figure 4.

No additional figure is required, but stronger visual back-references would materially improve the section.

---

## Cross-Section Consistency Check

**C1. Framing consistency is mostly good.**  
The Discussion matches the Introduction, Method, Experiments, and Conclusion in treating the paper as a compute-normalized diagnostic study rather than a new controller paper. There is no obvious fallback to the older CARD-paper framing.

**C2. One caveat from the outline is only partially realized.**  
The outline says the observer-controller result should be described as stable **under the tested policies**. The Discussion currently implies that point but does not state it as explicitly as the other sections.

**C3. One planned discussion function is still missing.**  
The outline says the section should explain why code belongs in the main paper as boundary evidence rather than a second success domain. The current draft strongly implies this, but does not yet say it outright.

---

## Recommended Revision Priority

1. Re-anchor §§5.1-5.3 to the paper's actual tables/figures and key quantitative facts.
2. Tighten §5.2 with the explicit caveat "under the tested policies."
3. Add one sentence in §5.3 explaining why HumanEval is included as boundary evidence, not as a second success domain.
4. Reuse the benchmark-role hierarchy from Method/Experiments when synthesizing task dependence.
5. Expand the §5.5 future-work items with one-line scientific rationales.
6. End the limitations block by restating the narrower contribution that still stands.

---

## Score: 7/10

**Justification**: The section is already honest, coherent, and aligned with the iteration-1 diagnostic-study framing. It loses points because the synthesis is not yet evidence-dense enough, the observer/controller claim needs tighter scoping, and the code-boundary interpretation is not stated as explicitly as the outline and other sections require. Fixing Issues 1-4 would likely raise the section to 8.5/10.

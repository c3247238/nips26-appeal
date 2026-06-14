# Critique: Writing

## CR-1: Proposal/Paper Coherence (CRITICAL)

The `idea/proposal.md` and `writing/paper.md` describe **two fundamentally different papers**:

- **proposal.md**: "Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts" — primary narrative is that trained SAEs have lower absorption than random baselines (H7), absorption is a structural artifact, and the optimal compression framing explains why absorption persists but is benign.
- **paper.md**: "Competitive Suppression in SAEs: Connecting the Locally Competitive Algorithm to Feature Absorption" — primary narrative is the LCA-SAE structural correspondence, local inhibition graph, and competitive suppression mechanism.

These are **not compatible narratives** sharing the same evidence. They require different emphases, different literature, different theory, and different conclusions. A reader cannot understand what this paper's actual contribution is from the two documents together.

**Severity**: Critical
**Category**: Writing

---

## CR-2: H1b Contradictory Status (CRITICAL)

In `proposal.md`:
- Table: H1b (delta-corrected) = "NOT SUPPORTED after correction" with p=0.028 uncorrected, p=0.334 Bonferroni
- Text (H1): "SUPPORTED (null hypothesis)" with r=+0.008 (L4), r=-0.301 (L8), neither significant at alpha=0.05

In `paper.md` Section 4.2:
- "delta-corrected steering at L8: r=-0.431, p=0.028" presented as the strongest signal

This is a direct contradiction. The same hypothesis (H1b) is simultaneously described as a null result and as evidence of a real effect. This inconsistency appears in the abstract of `proposal.md` ("zero hypotheses survive multiple comparison correction") while Section 4.2 of `paper.md` presents H1b as a significant result. The claim of "honest null-result reporting" — the paper's strongest asset according to the evolution lessons — is directly undermined by this contradiction.

**Severity**: Critical
**Category**: Writing

---

## CR-3: Phantom Contribution — Homeostatic Rebalancing (CRITICAL)

Section 1.4 of `paper.md` lists four contributions:

1. LCA-SAE connection (valid — theory is developed)
2. Local inhibition graph (valid — built, though H6 falsified)
3. Mechanistic explanation for precision-recall asymmetry (valid — empirically supported by H7)
4. **"First training-free post-hoc repair for absorption"** — homeostatic rebalancing

Contribution #4 is **not executed**. Section 4.7 explicitly states: "The homeostatic rebalancing experiment was not executed due to the negative H6 result." The abstract does not reflect that this was not executed. The contribution list presents it as a done thing.

A paper that claims a contribution and then does not execute the corresponding experiment is claiming phantom results.

**Severity**: Critical
**Category**: Writing

---

## CR-4: Abstract/Body Mismatch (MAJOR)

The abstract of `proposal.md` states:
> "zero hypotheses survive multiple comparison correction (12 tests, Bonferroni alpha = 0.00417)"

But Section 4.2 of `paper.md` frames r=-0.431 at L8 (p=0.028 uncorrected) as "the strongest signal in the dataset" — presenting it as evidence of a real effect. A reader of the abstract would expect zero significant results; a reader of Section 4.2 would conclude one marginally significant result exists.

This creates an inconsistent narrative about what the study actually found.

**Severity**: Major
**Category**: Writing

---

## CR-5: Cross-Model Validation Absent (MAJOR)

Both `proposal.md` and `methodology.md` note that cross-model (Pythia-70M) validation was "inconclusive; limited feature overlap." The paper presents only GPT-2 Small results throughout. The abstract and introduction do not prominently flag this limitation. Claims about the LCA framework and competitive suppression mechanism — which are framed as general (not GPT-2-specific) — have no cross-model validation.

A paper about a general mechanistic phenomenon (competitive suppression via decoder correlations) should either validate across models or prominently acknowledge single-model limitation as a scope constraint, not bury it in the risk table.

**Severity**: Major
**Category**: Writing

---

## CR-6: Framing Instability Across Documents (MAJOR)

Three different framings appear across documents:
1. **proposal.md**: "absorption as optimal compression" — frames absorption as rate-distortion optimal behavior
2. **writing/paper.md**: "absorption as competitive suppression via LCA" — frames absorption as a neuroscience-inspired mechanism
3. **alternatives.md**: "metric validation study" — frames absorption as a structural artifact measured by a potentially miscalibrated metric

These framings are **mutually inconsistent** in their implications:
- Optimal compression framing: absorption is expected and benign
- LCA/competitive suppression framing: absorption is caused by decoder correlations (testable, H6 falsified)
- Metric validation framing: absorption may be artifact of the Chanin metric itself

The paper has not resolved which framing is correct and should drive the narrative.

**Severity**: Major
**Category**: Writing

---

## CR-7: Table Referencing Errors (MINOR)

The proposal references Table 1 (Hypothesis Testing Summary) and Table 2 (Feature-Level Data) in the methodology section, but these tables are not present in the document. The first table in `proposal.md` is the Evidence-Driven Revisions table, not the hypothesis testing summary. This suggests copy-paste errors from an earlier version.

**Severity**: Minor
**Category**: Writing

---

## CR-8: Supplementary Materials Promised But Absent (MINOR)

The paper references "supplementary materials" for raw graph data and detailed statistics, but no supplementary materials are present in the workspace. Per-feature absorption rates, steering correlations, and probe F1 scores needed to reproduce the hypothesis tests are not accessible from any file in the workspace.

**Severity**: Minor
**Category**: Reproducibility

---

## Summary

The writing critique reveals **two critical coherence problems**:

1. **Two different papers** in proposal.md and writing/paper.md with incompatible framings, titles, and primary claims
2. **H1b presented as both null and significant** across documents, directly contradicting the paper's stated commitment to honest null-result reporting

The phantom contribution (homeostatic rebalancing) further undermines credibility. Before submission, these documents must be reconciled into a single coherent paper with consistent framing and accurate reporting of what was and was not executed.

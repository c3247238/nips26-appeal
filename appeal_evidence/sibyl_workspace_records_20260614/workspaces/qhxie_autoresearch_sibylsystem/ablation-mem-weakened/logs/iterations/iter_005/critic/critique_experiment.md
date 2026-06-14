# Critique: Experimentation

## CE-1: Steering Success Criterion Is Too Coarse (CRITICAL)

**Claim**: `methodology.md` uses "top-5 token contains target letter" as the steering success criterion. This is too coarse a metric.

**Problem**: Steering could reduce the probability mass on the target letter by 80% while keeping it in the top-5 tokens. The criterion only measures directional correctness (is the target in top-5?), not effectiveness (how much does steering shift probability?). Absorption may suppress parent feature activation, reducing steering effectiveness substantially without eliminating the target from top-5 entirely.

This means:
- The null correlations (H1: r=+0.008, r=-0.301) could be masking large absorption effects that don't fully suppress the target
- A feature with 90% steering degradation might still pass the top-5 criterion
- The steering metric cannot detect subtler absorption effects that reduce but don't eliminate steering

**Evidence**: proposal.md H1 shows r=-0.301 at L8 (p not significant) but the proposal simultaneously presents r=-0.431 at L8 (p=0.028) as evidence of real effect for delta-corrected steering. The difference between raw and delta-corrected suggests there is a directional shift, but the magnitude of that shift is not measured.

**Severity**: Critical

---

## CE-2: Random SAE Baseline Is Confounded (MAJOR)

**Claim**: H7 compares trained SAE (mean=0.034) against Random SAE (mean=0.278) to conclude "training reduces structural artifacts."

**Problem**: The Random SAE uses a "frozen orthonormal decoder, random encoder" (proposal.md Phase 3), while the trained SAE uses a pretrained decoder. The orthonormal constraint is an architectural difference, not just a training-state difference. An orthonormal decoder has fundamentally different correlation structure than a learned decoder — it cannot represent the same hierarchy. The 8x difference in absorption could be due to the orthonormal constraint forcing low correlations, not due to training reducing artifacts.

**Proper control**: Random SAE should use the same architecture as trained SAE (untied weights, same dictionary size) but with random initialization — not an orthonormal constraint. Alternatively, compare pretrained decoder + random encoder vs. pretrained decoder + trained encoder to isolate encoder training effects.

**Severity**: Major

---

## CE-3: Single-Feature Evidence Generalized (MAJOR)

**Claim**: RQ4 ("Do high-absorption features retain functional steering capability?") is answered with Feature U (24.2% absorption, 100% steering success) as the sole evidence.

**Problem**: n=1 is anecdotal, not systematic. H, S, U, V at L8 all have absorption >14% (19.0%, 16.0%, 24.2%, 14.7% respectively). If all four steer successfully, that is evidence. If only one does, that is also informative. But presenting U alone as supporting RQ4 implies the answer is general when it is only a single data point.

**Severity**: Major

---

## CE-4: Precision-Recall Probe Ceiling Effect (MAJOR)

**Claim**: "Precision = 1.0 universally at k >= 5" is cited as evidence that absorption does not affect precision.

**Problem**: Precision=1.0 at k>=5 indicates the probe has saturated — with 5 or more features, it can perfectly classify positive vs. negative examples. This ceiling effect means the precision invariance finding is trivially true by probe construction, not a meaningful finding about absorption. The probe always has enough capacity to be correct when it fires.

At k=1, precision std=0.195 — there is variance at low k that disappears at high k. This pattern is consistent with probe capacity saturation, not absorption insensitivity.

**Severity**: Major

---

## CE-5: Cross-Model Validation Absent (MAJOR)

**Claim**: The paper frames competitive suppression and LCA connection as general phenomena (not GPT-2 Small-specific).

**Problem**: All experimental validation is on GPT-2 Small. Pythia-70M validation was attempted but deemed inconclusive. No cross-model results are presented. The LCA connection (Section 3.1 of paper.md) is presented as a general structural correspondence, yet it is validated on only one model family.

Claims that "deeper layers show stronger inhibition structure" (Section 5.2 of paper.md) and that the framework explains "all prior findings" are only validated within GPT-2 Small.

**Severity**: Major

---

## CE-6: Homeostatic Rebalancing Not Executed But Claimed (CRITICAL)

**Claim**: Section 3.4 of paper.md proposes homeostatic rebalancing; Section 1.4 Contribution #4 claims it as a contribution.

**Problem**: Section 4.7 explicitly states the experiment was not executed: "was deferred pending improved graph construction." The contribution is claimed but the corresponding experiment was never done. A contribution requires executed validation, not just theoretical proposal.

**Severity**: Critical

---

## CE-7: Delta-Correction Method Not Fully Specified (MINOR)

**Claim**: "Random baseline subtraction for delta-corrected analysis" is mentioned but the protocol is underspecified.

**Problem**: How is the random baseline computed? How many random features? What distribution? The delta-corrected analysis yields r=-0.431 at L8 (p=0.028) which is the paper's strongest signal, but the method is not described in enough detail to reproduce. If the random baseline is computed from a small number of random features, the delta-correction could be noisy.

**Severity**: Minor

---

## CE-8: EC50 Analysis Appears Incomplete (MINOR)

**Claim**: H4 tests EC50 correlation with absorption; proposal.md reports L4: r=-0.166, p=0.439; L8: r=+0.180, p=0.380.

**Problem**: The EC50 analysis uses "linear interpolation" (methodology.md) rather than the Hill equation fitting described in the methodology. The methodology claims "Hill equation: S(s) = S_max * s^n / (EC50^n + s^n)" but the results section only reports correlations without specifying the fitted parameters (S_max, n, EC50). This makes it impossible to evaluate the quality of the dose-response fits or whether the Hill equation was actually used.

**Severity**: Minor

---

## Summary

The experimental design has three critical flaws:

1. **Steering metric too coarse**: top-5 criterion masks absorption effects that don't fully suppress targets
2. **H7 random baseline confounded**: orthonormal decoder constraint confounds the trained-vs-random comparison
3. **Homeostatic rebalancing claimed but not executed**: contribution #4 is phantom

Two major issues compound these: the probe ceiling effect means precision invariance is trivially satisfied at k>=5, and single-feature (n=1) evidence is used to answer a general research question. The cross-model absence is a significant scope limitation that should be prominently flagged.

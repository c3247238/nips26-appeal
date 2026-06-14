# Writing Critique

## Overall Judgment

- The paper now has the right identity: a diagnostic/evaluation paper, not a hero-controller paper.
- The remaining writing problem is not direction; it is claim discipline. Several sentences still promise more than the workspace currently proves.

## P0 Claim Corrections

- Soften the honest-compute headline.  
  Evidence: `exp/results/diag_compute_curve_gsm8k.json` shows one explicit reorder, not a broad ranking rewrite.
- Remove or soften the phrase `failure taxonomy` in the abstract and contribution list.  
  Evidence: the paper claims it, while the key bucket audit is still future work.
- Recast the observer/controller claim as “diagnostic signals with weak realized gains under the tested policies,” not as a stronger general statement.  
  Evidence: `exp/results/diag_signal_gap_audit.json` does not yet provide paired benefit buckets.
- Keep MATH500 explicitly in the boundary-evidence lane.  
  Evidence: `exp/results/diag_math500_shortlist.json` is `n=100`, single-seed, and all accuracies are tightly clustered.

## Section-Level Outline

### Abstract and Introduction

- State the study scale earlier: single seed, `n=100` slices for GSM8K and MATH500, `n=50` for HumanEval boundary evidence.
- Reduce numerical overload on page 1; the conceptual framing is strong enough without previewing every number.
- Replace pluralized “key comparisons and Pareto conclusions” with wording that matches the actual evidence strength.

### Related Work

- Add explicit citation anchors and make the novelty claim precise: this is an evaluation-correction paper with a protocol contribution, not a new algorithm paper.
- Connect related work directly to the shortlist taxonomy actually used in the experiments.

### Protocol Section

- Define `diagnostic score` and `control effectiveness` operationally, with one worked example per signal family.
- State whether “honest compute” means algorithmic efficiency, realized system cost, or both.

### Results Section

- The MATH500 table should either include runtime metadata or be explicitly labeled as accuracy-only boundary evidence.
- The HumanEval table should not merge `Entropy` and `TIGER` unless the merge is justified by provenance and identical behavior.
- If the paper keeps the `failure taxonomy` language, it needs an actual bucket table in the main text or appendix.

### Discussion and Conclusion

- Promote missing benefit buckets and seed robustness from generic future work to explicit current limitations that cap the claims.
- End with the narrow but credible recommendation the current evidence really supports: report runtime-fairness metadata, separate observation from control, and treat code as a structural stress test.

## Manuscript Sync Issues

- Resolve the canonical-source ambiguity between `writing/paper.md` and the integrated LaTeX build.
- Remove all visible TODO placeholders from the review-facing manuscript.
- Ensure every paper claim points to one canonical result asset, not two conflicting JSONs.

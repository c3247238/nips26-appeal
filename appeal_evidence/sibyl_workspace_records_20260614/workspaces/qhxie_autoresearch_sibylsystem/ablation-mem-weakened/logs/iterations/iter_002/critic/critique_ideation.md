# Ideation Critique: Local Inhibition Graph Framework

## Summary

The pivot to the Local Inhibition Graph framework represents a decisive and intellectually sound reframing of the project's narrative. The LCA-SAE structural correspondence is genuinely novel---zero prior work connects these two frameworks. The six-perspective result debate converged on a PROCEED recommendation with bounded risk. However, the ideation process has weaknesses: (1) the synthesizer ignored the result debate's recommendation to engage with Alternative C (trade-off analysis); (2) the novelty claim depends entirely on H6 validation; (3) the decision tree is biased toward "PROCEED"; (4) the "exact" correspondence claim is overstated for untied SAEs.

## Critical Issues

### 1. Synthesizer Ignored Trade-off Interpretation (Alternative C)

The result debate synthesis (verdict.md) explicitly recommended engaging with Alternative C: "The data strongly supports the view that absorption is a benign byproduct of the interpretability-compression trade-off." The comparativist, revisionist, and strategist all identified this as the strongest intellectual direction. Yet the current paper mentions trade-off only briefly ("absorption may be a benign byproduct") and does not treat it as a central contribution.

This is a strategic error. The trade-off interpretation:
- Is directly supported by the data (Feature U: 24.2% absorption, 100% steering; precision invariant; EC50 null)
- Provides intellectual value beyond "we found nothing"
- Explains why absorption-fixing interventions (Matryoshka, OrtSAE) increase other pathologies
- Was the Contrarian's core insight and the result debate's recommended pivot

**Fix**: Expand Section 6.2 into a serious engagement with the trade-off interpretation. Treat it as a plausible alternative reading of the null results that strengthens the paper's intellectual contribution. See findings.json for suggested text.

## Major Issues

### 2. Novelty Claim Is Conditional on H6 Validation

The ideation documents claim four "firsts":
1. First LCA-SAE connection
2. First inhibition graph for SAE diagnostics
3. First mechanistic explanation for precision-recall asymmetry
4. First training-free post-hoc repair

Claims 2-4 depend on H6-H10 validation. The Skeptic's critique is valid: the correspondence alone (Claim 1) is a definitional identity for tied-weight SAEs, not a derived result. Its scientific value depends on whether it generates validated predictions.

**Fix**: Acknowledge the conditional nature of the novelty claim. Frame Claim 1 as "exact by definition; its scientific value lies in the predictions it generates." Frame Claims 2-4 as "proposed contributions pending empirical validation."

### 3. Decision Tree Is Biased Toward "PROCEED"

The pivot decision tree in proposal.md has multiple "PROCEED" branches:
- H6 validated (precision@20 >= 0.10) -> PROCEED with full framework
- H6 partially validated (0.05-0.10) -> PROCEED with diagnostic-only claims
- H6 not validated (<= 0.05) -> "retain as theoretical speculation in Discussion"

Even the "failure" branch does not lead to a clean pivot. The Skeptic correctly identified this as confirmation bias.

**Fix**: Sharpen the decision tree:
- precision@20 < 0.05 -> PIVOT to Alternative C (trade-off analysis)
- precision@20 0.05-0.10 -> PROCEED with diagnostic-only claims, explicit caveats
- precision@20 >= 0.10 -> PROCEED with full framework

Remove the "retain as speculation" branch. If H6 fails, the framework's empirical claims collapse.

### 4. "Exact" Claim Overstated for Untied SAE

The proposal and paper repeatedly call the correspondence "exact." However, gpt2-small-res-jb uses untied weights. The Methodologist in the result debate explicitly flagged this. The correspondence is approximate for untied SAEs, and the approximation quality is unknown.

**Fix**: Quantify the approximation. Report correlation between W_dec^T W_dec and W_enc^T W_enc. Soften "exact" to "approximate" for the actual SAE used.

### 5. H10 Claimed as Contribution Before Testing

The proposal lists "first training-free post-hoc repair" as a contribution, but H10 has not been executed. The rebalancing formula has sign ambiguity (Skeptic's critique). Claiming an untested method as a contribution is premature.

**Fix**: Remove H10 from the contribution list until executed. Frame it as "we propose" rather than "we contribute."

## Minor Issues

6. **proposal.md contradicts paper on H1b**: proposal.md still states H1b is "SUPPORTED" with p=0.028, contradicting the paper's correct statement that H1b does not survive multiple comparison correction.

7. **Random baseline miscalculation**: proposal.md states 0.004 expected precision@20, but correct value is 0.000814.

## What Works Well

1. **Genuinely novel theoretical bridge**: The LCA-SAE connection has not been articulated in the literature despite ~2000 LCA citations.
2. **Clear falsification criteria**: Each hypothesis has explicit pass/fail thresholds.
3. **Bounded risk**: The decision tree (even if biased) includes a pivot option.
4. **Honest tracking of H1-H5 status**: All prior hypotheses are correctly labeled as refuted.
5. **Six-perspective synthesis**: The result debate process was thorough and converged on a clear recommendation.
6. **Fallback alternatives**: Alternative A-E provide genuine backup directions if LIG fails.

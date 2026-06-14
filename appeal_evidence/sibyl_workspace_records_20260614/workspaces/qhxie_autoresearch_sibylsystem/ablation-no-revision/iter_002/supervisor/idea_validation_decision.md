# Idea Validation Decision

## Pilot Evidence Summary

| Candidate | Hypothesis | Key Metric | Result | Falsified? |
|-----------|-----------|-----------|--------|-------------|
| cand_absorption_metric | H1 (Prevalence) | % latents with absorption >0.5 | 0.19% (predicted >20%) | YES |
| cand_absorption_metric | H3 (Sparsity monotonicity) | Spearman r(L0, absorption) | 0.086 (p=0.872) | YES |
| cand_downstream_impact | H4 (Circuit faithfulness) | low_vs_high_diff | 0.0 (both subsets 0.0) | YES |

### cand_absorption_metric
- **H1**: Only 0.19% of latents show absorption score >0.5, vs predicted >20%. The metric is 100x too low.
- **H3**: No monotonic relationship between L0 sparsity and absorption rate. Absorption peaks at layer 4 (49.3%) rather than increasing with sparsity.
- **8 latents** have suspiciously perfect scores (exactly 100) — suggests metric artifacts or data leakage.
- The absorption metric may measure something orthogonal to the intended phenomenon.

### cand_downstream_impact
- SAE patching faithfulness (0.289) is below raw residual (0.400), confirming SAE introduces noise.
- Critical failure: both low-absorption (0.0) and high-absorption (0.0) latents yield identical 0.0 faithfulness. The latent selection method is fundamentally broken.
- Cannot measure downstream absorption impact because the selection metric does not distinguish meaningful latents.

## Decision Matrix

| Criterion | Weight | cand_absorption_metric | cand_downstream_impact |
|-----------|--------|------------------------|------------------------|
| Pilot signal strength | 0.30 | 1 — signal absent (0.19% vs >20%) | 1 — negative (both subsets 0.0) |
| Hypothesis survival | 0.25 | 1 — H1, H3 both falsified | 1 — H4 falsified |
| Path to full result | 0.20 | 1 — metric needs total redesign | 1 — selection method broken |
| Novelty | 0.15 | 2 — first systematic audit (but results are null) | 2 — downstream impact unmeasurable |
| Resource efficiency | 0.10 | 1 — no credible path forward | 1 — no credible path forward |
| **Weighted Score** | | **1.20** | **1.10** |

Scoring guide: 5=Strong positive, 4=Positive minor uncertainties, 3=Ambiguous, 2=Weak significant concerns, 1=Negative no credible path.

## Decision Rationale

Both candidates failed their primary falsification criteria:
- **cand_absorption_metric**: H1 falsified by 100x (0.19% vs >20% threshold). H3 shows no sparsity-absorption relationship. The absorption metric appears to capture noise or an unrelated phenomenon.
- **cand_downstream_impact**: H4 shows SAE patching reduces faithfulness but the absorption-based latent selection produces identical (0.0) results for both low and high absorption subsets. The selection mechanism is fundamentally broken.

Both weighted scores < 2.5. No candidate has a credible path to publishable results. Prior work on these candidates is sunk cost — irrelevant to the forward decision.

## Next Actions

1. **Drop both candidates**: cand_absorption_metric and cand_downstream_impact are not worth additional GPU budget.
2. **Return to ideation**: Generate new candidates that address the failure modes:
   - The metric artifact problem (8 latents with exactly 100 score) suggests the absorption metric needs fundamentally different operationalization
   - The selection method failure suggests need for orthogonal selection criteria
3. **Consider backup ideas**: The project memory mentions "Activation Energy Theory" (novelty 8/10) as a backup — this could be re-evaluated as a fresh candidate.
4. **Do NOT advance any current candidate**: Both require redesign that is equivalent to starting over.

## Verdict

SELECTED_CANDIDATE: none
CONFIDENCE: 0.95
DECISION: PIVOT

Both candidates failed their core falsification criteria. The absorption metric and downstream impact frameworks are broken at a fundamental level. System must pivot to new ideation.

Evidence triggering pivot: H1 falsified (0.19% vs >20%), H3 falsified (Spearman r=0.086), H4 falsified (0.0 difference between low/high absorption subsets). All three main hypotheses are false.
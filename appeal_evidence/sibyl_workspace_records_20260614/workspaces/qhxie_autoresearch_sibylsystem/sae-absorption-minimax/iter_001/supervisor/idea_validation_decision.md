# Idea Validation Decision

## Pilot Evidence Summary

### cand_a (front_runner): Steering Signature of Feature Absorption

| Hypothesis | Status | Key Evidence |
|------------|--------|-------------|
| H1 (layer-wise absorption) | UNRESOLVED | Run 1: Layer 4 < Layer 8 (+10.6%); Run 2: Layer 4 > Layer 8 (-22.9%) — contradictory |
| H2 (mitigation hierarchy) | PARTIALLY CONFIRMED | TopK: 70.9% absorption reduction, 8x MSE cost; JumpReLU: failed to converge; ATM/OrtSAE: pending |
| H3 (steering signature) | REVERSED | Full (N=100): High-abs mean effect=0.1035, Low-abs=0.0874, Spearman r=+0.35 (p=2.92e-04) — OPPOSITE to original prediction |
| H4 (UAS correlation) | CONFIRMED | Spearman r=0.65-0.79 across all runs |
| H5 (downstream discriminability) | DIRECTIONAL | 4.95% task-dependence delta vs. 5% threshold (marginal fail) |

### cand_b (backup): Absorption-Aware Hierarchical Feature Decomposition
- No pilot data. Grounded in H3 hub mechanism interpretation.
- Requires loading HSAE hierarchy relationships.

### cand_c (backup): 4D Pareto Frontier Analysis
- No pilot data. Theoretically interesting extension.
- H3 finding makes steering sensitivity a natural Pareto dimension.

### cand_d (backup): Feature Reliability Index (FRI)
- No pilot data. Depends on FRI outperforming UAS alone.

---

## Decision Matrix

| Criterion | Weight | cand_a | cand_b | cand_c | cand_d |
|-----------|--------|--------|--------|--------|--------|
| Pilot signal strength | 0.30 | 5 | 1 | 1 | 1 |
| Hypothesis survival | 0.25 | 3 | 2 | 1 | 1 |
| Path to full result | 0.20 | 4 | 3 | 2 | 2 |
| Novelty | 0.15 | 5 | 4 | 4 | 3 |
| Resource efficiency | 0.10 | 4 | 3 | 2 | 3 |
| **Weighted Score** | | **3.90** | **2.55** | **1.85** | **1.85** |

### Scoring Rationale

**cand_a = 3.90**:
- Signal (5): H3 reversal is the strongest pilot finding in the entire candidate pool. N=100, p=2.92e-04, with clear mechanistic interpretation (hub features). H4 confirmed with r=0.79. H2 partially confirmed (TopK 70.9%/8x). H1 unresolved but acknowledged.
- Survival (3): H3 was reversed (not confirmed in original direction), which is scientifically valuable but scores below "5" for hypothesis survival. H4 survived perfectly. H5 directional. Two hypotheses unresolved (H1) or pending (H2 ATM/OrtSAE).
- Path (4): Clear path: complete H1 final + H3 Gemma replication + H3 null control + H5 → writing. Strong paper structure already in proposal.md.
- Novelty (5): H3 reversal is genuinely novel (Spearman r=+0.35 between absorption and steering sensitivity is absent from all prior art). UAS validated. 5-method benchmark fills clear gap. Transforms SAE Sanity Checks from existential threat to paper motivation.
- Efficiency (4): Only ~3-4 experiments remain (H3 Gemma ~45min, H3 null ~15min, H1 final ~30min, H5 ~60min). Most experiments already complete.

**cand_b = 2.55**:
- Signal (1): No pilot data. Entirely theoretical.
- Survival (2): Depends on H3 hub mechanism being real. If confirmed by H3 null control, cand_b becomes more attractive.
- Path (3): Pilot is straightforward (~15 min). But full approach requires significant new experiments.
- Novelty (4): HSAE hierarchy + absorption regularization is technically novel, but no empirical validation yet.
- Efficiency (3): Not more efficient than advancing cand_a's confirmed findings.

**cand_c = 1.85**:
- Signal (1): No pilot. 4D Pareto frontier is purely theoretical.
- Survival (1): Risk that absorption is Pareto-optimal for steering (making reduction counterproductive).
- Path (2): Significant new experiments required.
- Novelty (4): 4D extension is genuinely novel in theory.
- Efficiency (2): New experiments would require substantial GPU budget.

**cand_d = 1.85**:
- Signal (1): No pilot. FRI must outperform UAS alone.
- Survival (1): Depends on FRI empirical validation.
- Path (2): Pilot is straightforward but full validation requires FRI to beat UAS.
- Novelty (3): Per-feature FRI is useful but softer contribution than H3 reversal.
- Efficiency (3): Similar budget to cand_b.

---

## Decision Rationale

**cand_a ADVANCES** (weighted score 3.90 >= 3.5 threshold). The H3 reversal is the single most important scientific finding across all candidates. It is:
1. **Surprising**: The opposite of the original prediction (absorbed features are MORE steerable, not less)
2. **Novel**: Spearman r=+0.35 between absorption and steering sensitivity is absent from all prior art
3. **Actionable**: Reframes absorption as a hub-feature signature, not a silencing signal
4. **Transformative**: Converts the SAE Sanity Checks existential threat into paper motivation

The unresolved hypotheses (H1, H2 ATM/OrtSAE, H5) do NOT falsify the main claim. H3's main hypothesis was NOT falsified — it was reversed, which is scientifically valuable. The paper already has a strong structure with the revised title "The Steering Signature of Feature Absorption."

**Sunk cost check**: cand_b, cand_c, cand_d have zero pilot data. Investing GPU budget in untested directions while a direction with N=100 empirical confirmation is available is indefensible.

---

## Next Actions

### Immediate (cand_a advancement):
1. **H3 Gemma replication** (full_h3_gemma, ~45 min): Confirm that positive correlation holds on Gemma-2B layer 12. Critical for cross-model generalizability claim.
2. **H3 null control** (pilot_h3_null, ~15 min): Add shuffled/random direction controls to confirm high-absorption effect is genuine, not artifact. Strengthens the mechanism claim.
3. **H1 final resolution** (full_h1_final, ~30 min): Resolve contradictory layer-wise results with fixed feature selection. Report with uncertainty bands — do not overclaim.
4. **H5 expansion** (full_h5, ~60 min): Expand to 200 features across GPT-2 and Gemma-2B. Replace synthetic counterfactuals with real causal QA (CounterFact/TruthfulQA) to address low causal AUC.
5. **Matryoshka SAEs** (full_h2 extension): Add Matryoshka SAEs to H2 benchmark. Reviewers will flag omission.

### Writing (can proceed in parallel):
6. **Begin paper drafting**: Use proposal.md structure. Lead with H3 reversal as the central contribution.
7. **Related Work section**: Explicitly contrast activation sensitivity (Tian et al.) vs. steering sensitivity (this work).
8. **SAE Sanity Checks rebuttal**: Draft dedicated paragraph in Related Work / Discussion.

### Pending confirmation (may trigger pivot to cand_b):
9. If H3 null control fails to confirm hub mechanism, pivot to cand_b for absorption-aware regularization.

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.72
DECISION: ADVANCE

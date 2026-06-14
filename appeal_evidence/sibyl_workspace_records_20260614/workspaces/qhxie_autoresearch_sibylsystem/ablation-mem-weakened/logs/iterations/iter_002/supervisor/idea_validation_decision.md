# Idea Validation Decision

## Pilot Evidence Summary

**CRITICAL NOTE:** This is Iteration 2 with a new front-runner candidate (cand_f: Local Inhibition Graph). The pilot experiments for H6-H10 have NOT been executed yet. The existing pilot data (pilot_summary.json) is from Iteration 1 (cand_a) and is irrelevant to the new direction.

### Prior Evidence from Iteration 1 (cand_a) - Context Only

| Hypothesis | Layer 4 | Layer 8 | Verdict |
|---|---|---|---|
| H1 (Raw steering vs absorption) | r = +0.008, p = 0.970 | r = -0.301, p = 0.136 | FALSIFIED |
| H1b (Delta-corrected steering vs absorption) | r = +0.245, p = 0.227 | r = -0.431, p = 0.028 | NOT SUPPORTED after MCP |
| H2 (Probing F1 vs absorption) | r = -0.003, p = 0.987 | r = -0.107, p = 0.604 | FALSIFIED |
| H3 (Cross-layer consistency) | Opposite signs; CV = 1.53 | | FALSIFIED |
| H4 (EC50: efficiency vs absorption) | p = 0.232 | p = 0.435 | NOT SUPPORTED |
| H5 (Precision invariance) | Precision std = 0.054, Recall std = 0.199 | Precision std = 0.028, Recall std = 0.192 | SUPPORTED (at k>=5) |

**Key methodological findings from prior round:**
- Zero hypotheses survive multiple comparison correction
- H1b uncorrected p=0.028 does NOT survive Bonferroni (p=0.334) or BH-FDR (q=0.107)
- Steering-encoder confound is fundamental (steering bypasses SAE encoder)
- Random baseline heterogeneity is extreme: layer 8 range [0.0, 1.0], std=0.402
- Quality plateau at 6.0/10 for 2 rounds despite 8 revision rounds

### Why cand_f Was Selected as Front-Runner

The Local Inhibition Graph framework was selected in Iteration 2 because:
1. **Exact mathematical correspondence:** W_dec^T W_dec = G_LCA (Rozell et al. 2008) --- not metaphorical
2. **Verified novelty:** No prior work connects LCA to SAE absorption; no prior work constructs decoder correlation graphs for SAE diagnostics
3. **Explains existing findings:** Precision-recall asymmetry (H5), layer-dependent effects (H1b), steering robustness all have natural explanations in competitive suppression framework
4. **Training-free and scalable:** Computed from pretrained SAE weights; O(k * d_dict) complexity
5. **Clear falsifiable predictions:** H6 (precision@20 >= 0.10 vs ~0.004 chance) has an unambiguous threshold

### Pilot Evidence for cand_f: NOT YET COLLECTED

The following experiments are planned but not executed:
- H6: Graph edges predict absorption pairs (precision@20 >= 0.10)
- H7: Inhibition explains precision-recall asymmetry
- H8: Graph predicts at-risk features
- H9: Layer-dependent graph structure
- H10: Homeostatic rebalancing (exploratory)

---

## Decision Matrix

### cand_f: Local Inhibition Graph (Current Front-Runner)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | No direct pilot data yet. However, the mathematical correspondence (W_dec^T W_dec = G_LCA) is exact and derivable. Prior data (H5 precision invariance, H1b layer 8 effect) is consistent with the framework. |
| Hypothesis survival | 0.25 | 4 | H6-H10 are new hypotheses with clear falsification criteria. H5 (precision invariance) from prior round is SUPPORTED and directly explained by competitive suppression. The framework does not depend on the failed H1-H4. |
| Path to full result | 0.20 | 4 | Clear route: (1) construct graph, (2) validate against Chanin pairs, (3) test precision-recall explanation, (4) layer analysis, (5) optional repair. All experiments are training-free and estimated at ~2 GPU-hours total. |
| Novelty | 0.15 | 5 | Verified novelty: no prior LCA-SAE connection, no prior inhibition graph for SAE diagnostics, no prior mechanistic explanation for precision-recall asymmetry, no prior training-free repair. Novelty searches returned NO MATCHES. |
| Resource efficiency | 0.10 | 4 | Pilot: ~30 min (graph construction + H6 validation). Full: ~2 GPU-hours. Extremely low cost for the potential contribution. All experiments use existing pretrained SAEs. |
| **Weighted Score** | | **3.75** | |

### cand_c: Feature Absorption as Optimal Compression (Trade-off Analysis)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Data strongly supports trade-off view: 100% steering for most absorbed feature, precision invariant, full-activation F1=1.00, EC50 no degradation. Null results on H1-H4 are evidence FOR this framing. |
| Hypothesis survival | 0.25 | 4 | The data from cand_a directly supports the trade-off hypothesis. No new experiments needed --- reinterpret existing data. |
| Path to full result | 0.20 | 4 | Can reuse all existing data. Requires reframing paper narrative, not new experiments. |
| Novelty | 0.15 | 3 | First quantification of absorption-pathology trade-off is novel, but less theoretically grounded than cand_f. |
| Resource efficiency | 0.10 | 5 | Zero additional GPU time. All data already collected. |
| **Weighted Score** | | **3.55** | |

### cand_a: Feature Absorption and Downstream SAE Reliability (Previous Front-Runner)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | ZERO hypotheses survive multiple comparison correction. Only 1/6 hypotheses supported (H5 at k>=5). |
| Hypothesis survival | 0.25 | 1 | 0/6 survive after MCP. Central empirical claim is unsupported. |
| Path to full result | 0.20 | 2 | Paper written, 8 revision rounds, score stagnant at 6.0. Critical issues (EC50 mismatch, steering confound) are not fixable by writing. |
| Novelty | 0.15 | 2 | First systematic correlation attempt is novel, but null result with zero significant findings limits contribution. |
| Resource efficiency | 0.10 | 1 | ~20 GPU-hours spent with diminishing returns. Quality plateaued. |
| **Weighted Score** | | **1.30** | |

---

## Decision Rationale

**Decision: ADVANCE with cand_f (Local Inhibition Graph)**

**Confidence: 0.72**

### Why ADVANCE cand_f:

1. **Strongest weighted score (3.75 vs 3.55 for cand_c).** cand_f edges out cand_c on novelty (5 vs 3) and hypothesis survival (4 vs 4, but cand_f's hypotheses are fresh and not contaminated by prior failures).

2. **Exact mathematical correspondence provides theoretical bedrock.** The structural correspondence W_dec^T W_dec = G_LCA is not an analogy --- it is an exact equality when weights are tied. This gives the framework a rigor that cand_c's trade-off analysis lacks.

3. **Verified novelty across all search dimensions.** Novelty searches confirmed: no prior LCA-SAE connection, no prior inhibition graph for SAE diagnostics, no prior training-free repair. The contribution margin is genuinely new.

4. **Explains the one robust finding from prior work (H5: precision invariance).** Competitive suppression naturally explains why precision is invariant (selectivity preserved) while recall varies (coverage reduced). This turns a "null result finding" into a "mechanistically explained phenomenon."

5. **Low risk, high upside.** Pilot cost is ~30 min GPU time. If H6 fails (precision@20 <= 0.05), the pivot to cand_c is clean and immediate. If H6 succeeds, the contribution is substantially stronger than cand_c.

6. **Training-free = scalable = practical impact.** The diagnostic tool works on any pretrained SAE without retraining. This is a genuine practical contribution that practitioners can use immediately.

### Why Not cand_c:

- cand_c is a strong backup but is primarily a reframing of existing null results
- The trade-off analysis lacks the theoretical hook that cand_f provides
- cand_c's novelty score (3/5) is lower than cand_f's (5/5)
- If cand_f pilots succeed, the paper is significantly stronger

### Why Not cand_a:

- cand_a has been thoroughly tested and failed to produce significant results
- Quality plateau at 6.0/10 after 8 revision rounds
- Steering-encoder confound is fundamental and not fixable
- The direction has exhausted its predictive power

### Risk Acknowledgment:

The primary risk is that **no pilot data exists yet for cand_f.** The ADVANCE decision is based on:
- Theoretical strength (exact mathematical correspondence)
- Verified novelty (independent searches confirm no prior work)
- Consistency with existing data (H5 explained, H1b layer-dependence explained)
- Low cost of validation (~30 min pilots)

If H6 fails (precision@20 <= 0.05), the decision should immediately pivot to cand_c.

---

## Sanity Checks

- [x] Compared ALL candidates (cand_a, cand_c, cand_f)
- [x] cand_a's main hypothesis was falsified by multiple comparison correction
- [x] Not swayed by sunk cost --- 20 GPU-hours on cand_a are irrelevant
- [x] cand_f scores highest (3.75) among all candidates
- [x] Defaulting to ADVANCE with explicit go/no-go criteria for pilots
- [x] Clear pivot path to cand_c if H6 fails

---

## Next Actions

1. **Execute pilot experiments for cand_f (H6-H8):**
   - Construct inhibition graph for GPT-2 Small layer 8 (k=20 neighbors)
   - Validate H6: precision@20 >= 0.10 vs ~0.004 chance
   - Test H7: correlation between inhibition and recall/precision
   - Test H8: graph statistics predict at-risk features
   - Estimated time: ~30 min GPU

2. **Apply go/no-go decision after pilots:**
   - GO (precision@20 >= 0.10, Fisher p < 0.001): Proceed with full H6-H10 + optional cross-model
   - CAUTION (precision@20 = 0.05-0.10): Proceed with diagnostic-only claims, drop H10 repair
   - NO-GO (precision@20 <= 0.05): Pivot to cand_c (trade-off analysis)

3. **If pilots pass, execute full experiments:**
   - H6: Graph validation across layers 0, 4, 8, 10
   - H7: Precision-recall asymmetry analysis
   - H8: At-risk feature prediction
   - H9: Layer-dependent graph structure
   - H10: Homeostatic rebalancing (exploratory)
   - Estimated time: ~2 GPU-hours

4. **If pilots fail, execute cand_c pivot:**
   - Reframe paper narrative to trade-off/compression interpretation
   - Reuse all existing data from iterations 1-8
   - Add Matryoshka SAE comparison if feasible
   - Estimated time: ~1 GPU-hour

SELECTED_CANDIDATE: cand_f
CONFIDENCE: 0.72
DECISION: ADVANCE

# Verdict: Local Inhibition Graph Framework

## Executive Summary

**Result Quality Score: 6/10**
**Recommendation: PROCEED with cand_f (Local Inhibition Graph), with bounded risk and clear fallback**

The project has pivoted from a failed correlation study (score 3/10, all H1-H5 refuted) to a neuroscience-inspired theoretical framework connecting Rozell et al.'s Locally Competitive Algorithm (LCA) to SAE feature absorption. The structural correspondence W_dec^T W_dec = G_LCA is mathematically exact and genuinely novel --- zero prior work makes this connection. The existing data provides strong indirect support (precision invariance, delta-corrected steering, layer-dependence). The critical test (H6: graph edges predict absorption pairs) requires ~15 minutes of computation and has a clear pass/fail threshold. Risk is bounded: if H6 fails, pivot to cand_c (trade-off analysis) with no sunk cost.

---

## Score & Justification

| Dimension | Score | Why |
|---|---|---|
| Theoretical grounding | 2/2 | Exact mathematical correspondence; ~2000-citation foundation; zero prior SAE applications |
| Novelty | 2/2 | First LCA-SAE connection; first inhibition graph for SAE diagnostics; first training-free repair (if validated) |
| Empirical validation | 0.5/2 | Strong indirect support (precision invariance, delta-corrected steering); BUT direct test (H6) not yet run |
| Methodological rigor | 0.5/2 | Clear falsification thresholds; random baselines; BUT n=26 small, H3 circular, multiple comparisons unaddressed |
| Falsifiability | 1/2 | H6 has clear pass/fail; BUT decision tree has too many "proceed" branches; H10 sign ambiguity |
| **Total** | **6/10** | Strong theory + pending empirical validation. Rises to 8-9/10 if H6 validates; drops to 3-4/10 if H6 fails. |

---

## Key Conclusion

**The Local Inhibition Graph framework is a genuinely novel theoretical contribution with practical utility, but its empirical foundation hinges on a single gatekeeper experiment (H6).**

The old hypothesis --- that absorption proportionally degrades downstream SAE reliability --- is refuted by the evidence. The new framework does not rescue it; it replaces it with a mechanistic theory that explains why absorption affects recall but not precision, why steering remains robust, and why effects are layer-dependent.

---

## What All 6 Analysts Agree On

1. **The old hypotheses (H1-H5) are refuted.** No primary hypothesis survived multiple comparison correction.
2. **The LIG framework is a new research program, not a rescue attempt.** It replaces H1-H5 with mechanistically grounded hypotheses (H6-H10).
3. **The structural correspondence W_dec^T W_dec = G_LCA is mathematically exact and novel.** No prior work connects LCA to SAEs.
4. **Precision invariance (H5) is the strongest supported finding.** 21-25/26 features have precision=1.0; recall varies 0.10-1.0.
5. **The critical test (H6) has not been run.** All empirical claims hinge on whether decoder correlation edges predict absorption pairs.
6. **The novelty claim is strong.** Four "firsts" if validated: first LCA-SAE connection, first inhibition graph diagnostic, first mechanistic explanation for precision-recall asymmetry, first training-free repair.

---

## Key Points of Disagreement (Resolved)

| Disagreement | Resolution |
|---|---|
| **Is LCA correspondence a "discovery" or "rebranding"?** | The correspondence is exact and novel, but its scientific value depends on H6 validation. If H6 succeeds, it is a productive theoretical bridge. If H6 fails, the "rebranding" critique dominates. |
| **Is precision invariance real or a ceiling effect?** | Likely real (full-activation probing also achieves F1=1.0), but should be tested on semantic features (WordNet) to rule out task-specific artifacts. |
| **Does the decision tree always yield "PROCEED"?** | Partially valid concern. Remedy: pre-register H6 as sole gatekeeper with fixed threshold (precision@20 >= 0.10) and commit to pivoting if it fails. |
| **Is H3 (at-risk prediction) viable?** | Underpowered (n=26) and circular (same data). Must use LOOCV or cross-layer prediction. Treat as exploratory, not confirmatory. |
| **Is H10 (rebalancing) theoretically sound?** | Sign ambiguity in update rule. Must test both additive and subtractive signs empirically before any claims. |

---

## Competitive Position

| Dimension | Assessment |
|---|---|
| vs Chanin et al. (2024) | We explain *why* absorption happens mechanistically; they identified the phenomenon |
| vs SAEBench (ICML 2025) | We predict at-risk features *without* running absorption metrics; they standardize measurement |
| vs Matryoshka SAE (ICML 2025) | We explain *why* multi-level structure helps; they provide the architectural solution |
| vs OrtSAE (2025) | We predict orthogonality's effectiveness from first principles; they engineer the constraint |
| vs Tang et al. (2025) | Complementary theoretical lens (dynamical vs. optimization) |
| **Venue if H6 validated** | NeurIPS/ICML/ICLR plausible |
| **Venue if H6 partially validated** | AAAI/EMNLP/ACL or COLM likely |
| **Venue if H6 fails** | ICLR MiF Workshop / arXiv fallback |

---

## Action Plan

### Immediate (Day 1, ~0.5 GPU hours)

**E1: H6 Gatekeeper --- Graph Construction + Validation**
- Construct inhibition graph for GPT-2 Small L8 (24K latents, k=20)
- Compute precision@20 against Chanin absorption pairs (26 first-letter features)
- Include random baseline and non-absorbed correlated pair control
- **Pass**: precision@20 >= 0.10 -> proceed to Phase 2
- **Fail**: precision@20 <= 0.05 -> pivot to cand_c (trade-off analysis)

### Core Validation (Days 1-2, ~2 GPU hours)

- **E2: H7** --- Precision-recall asymmetry test (correlate inhibition with recall/precision)
- **E3: H8** --- At-risk prediction with LOOCV (fix circularity)
- **E4: H9** --- Layer-dependent graph structure (L0, L4, L8, L10)

### Repair + Generalization (Days 2-5, ~4 GPU hours)

- **E5: H10** --- Homeostatic rebalancing (test BOTH signs of update rule)
- **E6: Cross-model** --- Replicate on Gemma-2-2B or Pythia-160M
- **E7: Semantic features** --- WordNet hierarchy test (optional but recommended)

### Mandatory Methodological Fixes

- [ ] Correct random baseline: 0.00083 (not 0.004)
- [ ] Pre-register H6 at L8, k=20 as primary analysis
- [ ] Report both uncorrected and corrected p-values
- [ ] Use LOOCV or cross-layer prediction for H8
- [ ] Add non-absorbed correlated pair control
- [ ] Report W_dec^T W_dec vs W_enc^T W_enc correlation for untied SAE
- [ ] Test both additive and subtractive rebalancing rules for H10

---

## Decision Tree

```
Current Proposal: Local Inhibition Graph (cand_f)
|
|-- H6 validated (precision@20 >= 0.10):
|   --> PROCEED with H7-H9 (core framework validated)
|   --> If H10 succeeds: Full paper with diagnostic + repair
|   --> If H10 fails: Paper with diagnostic only (still strong)
|   --> Paper target: NeurIPS/ICML/ICLR
|
|-- H6 partially validated (precision@20 = 0.05-0.10):
|   --> PROCEED with diagnostic-only claims (no repair)
|   --> Paper becomes "Decoder Correlations Reveal Competitive Suppression in SAEs"
|   --> Paper target: ICLR workshop, COLM, or AAAI
|
|-- H6 not validated (precision@20 <= 0.05):
|   --> PIVOT to cand_c (trade-off analysis)
|   --> Paper becomes "Feature Absorption as Optimal Compression: Evidence from GPT-2 Small"
|   --> Inhibition framework retained as theoretical speculation in Discussion
|   --> Paper target: COLM, ICLR workshop, or arXiv
|
|-- H7 fails (inhibition doesn't explain precision-recall):
|   --> Refine explanation (add L1 sparsity contribution)
|   --> Diagnostic claims (H6) still stand
|
|-- H10 fails (repair doesn't work):
|   --> DROP repair claims
|   --> Diagnostic contribution stands independently
```

---

## Bottom Line

**PROCEED with the Local Inhibition Graph framework.**

Run E1 (H6 validation) first --- it is the gatekeeper experiment. If precision@20 >= 0.10, the framework's central claim is validated and we proceed with H7-H9. If precision@20 <= 0.05, pivot immediately to cand_c (trade-off analysis).

The inhibition graph transforms the project's narrative from "we tried to correlate absorption with tasks and found nothing" to "we discovered that decoder correlations encode competitive suppression, explaining why absorption affects recall but not precision." This is a theoretical contribution with practical utility, not a null result.

**The path forward is a decisive experiment, not a rescue mission.**

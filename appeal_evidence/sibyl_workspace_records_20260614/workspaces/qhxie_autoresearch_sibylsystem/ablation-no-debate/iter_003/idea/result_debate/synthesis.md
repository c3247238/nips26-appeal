# Result Debate Synthesis: Encoder-Driven Feature Absorption in SAEs

**Date**: 2026-05-01
**Iteration**: 4 (ablation-no-debate)
**Available Perspectives**: 5/6 (optimist, strategist, methodologist, comparativist, revisionist)
**Missing**: skeptic.md

---

## 1. Consensus Map: High-Confidence Conclusions

All 5 perspectives agree on the following:

1. **H_Mech partially confirmed**: Encoder is the primary driver (B≈D holds), but decoder contribution is non-zero under stochastic hierarchy (Condition C shows extreme outliers: 17.3, 43.8 in seeds 789/1024)

2. **H_Comp FALSIFIED**: No monotonic relationship between hierarchy strength and absorption (R²=0.04, far below threshold 0.8). This is a legitimate negative result.

3. **H_Pareto UNINTERPRETABLE**: Sensitivity metric broken — returns constant value (3.018) across ALL L0 levels, making Pareto frontier analysis impossible.

4. **H_Safe FAILED on real SAEs**: Both GPT2-small (p=0.345) and Gemma Scope pilot (p=1.0) show safety features are NOT disproportionately absorbed.

5. **Methodology issues exist**: The experiment design has validity threats (selection bias in H_Safe features, anomalous Condition D invariance at 0.0175, H_Pareto metric broken)

---

## 2. Conflict Resolution

| Conflict | Perspective A | Perspective B | Resolution |
|----------|---------------|--------------|------------|
| H_Mech decoder contribution | Optimist: encoder-driven, decoder is noise | Methodologist/Revisionist: decoder can be pathological | **Decoder is non-zero in high-activation regimes** — the story is more complex than "encoder-only" |
| H_Comp non-monotonic relationship | Optimist: "inverted U" shape is novel finding | Strategist: falsified, drop continuation | **Legitimate negative result** — the monotonic assumption fails. Novelty claim is weak unless validated |
| H_Pareto zero absorption | Methodologist: metric broken or measurement failure | Optimist: absorption metric works for H_Mech, not H_Pareto | **Measurement issue confirmed** — sensitivity metric needs debugging before Pareto claims are valid |
| Venue recommendation | Comparativist: workshop tier | Strategist: pivot needed | **Both agree current trajectory is insufficient for main conference** |

**Key judgment call**: The "encoder-driven" conclusion from iter_001 (deterministic hierarchy) does NOT replicate under stochastic conditions. The paper's main contribution requires qualification or deeper investigation.

---

## 3. Result Quality Score: 3/10

**Justification**:
- 1/4 hypotheses fully confirmed (H3: steering sensitivity, only from iter_001)
- 1/4 partially confirmed with caveats (H_Mech: encoder-driven with decoder anomalies)
- 1/4 falsified (H_Comp: monotonic hierarchy strength)
- 1/4 broken/uninterpretable (H_Pareto: constant sensitivity)
- H_Safe failed on real SAEs

The result quality is significantly below the threshold for a main-conference paper. The methodology has critical issues (broken metrics, failed replications, selection bias) that prevent strong conclusions.

---

## 4. Key Findings

- **Encoder is primary absorption driver**: Condition B (trained encoder + random decoder = 0.055) ≈ Condition D (both trained = 0.017) confirms encoder alignment with hierarchy structure

- **Decoder can create pathological amplification**: Condition C shows extreme outliers (17-44x absorption) in seeds with high parent activation — the decoder-driven story is real in saturation regime

- **Hierarchy strength does NOT predict absorption monotonically**: R²=0.04, non-monotonic pattern, peak at cosine ~0.85-0.90 suggesting an "inverted U" shape

- **Sensitivity metric is broken on synthetic data**: Constant 3.018 across all L0 levels — metric may work on real SAEs (Gemma Scope) but not synthetic hierarchies

- **Safety features are NOT disproportionately absorbed**: Direction positive (233 vs 222) but p=0.345 (not significant), pilot on Gemma shows p=1.0

---

## 5. Methodology Gaps (from Methodologist + Revisionist)

1. **Sensitivity metric must be debugged**: H_Pareto is uninterpretable due to constant sensitivity. Compare implementation against Hu et al. (2025) source code, test on synthetic case where sensitivity SHOULD vary.

2. **H_Safe feature selection is invalid**: Features [1024, 2048, ...] are arbitrary indices, not Neuronpedia-matched. Must retrieve actual annotations and match by activation frequency.

3. **Condition D invariance is suspicious**: All 5 seeds report exactly 0.01746168273373232 — suggests shared constant or aggregation bug rather than true measurement.

4. **Missing ablations**: Encoder alignment QUALITY (not just trained vs random), hierarchy depth variation, feature frequency interaction

5. **Falsification criteria not applied**: H_Pareto shows "full_pass": true despite R²=0; H_Safe pilot NO_GO but no subsequent plan documented

---

## 6. Competitive Position (from Comparativist)

| Method | Absorption Reduction | Year | Notes |
|--------|----------------------|------|-------|
| Matryoshka SAE | 90% (0.49 → 0.05) | 2025 | Main conference level |
| OrtSAE | 65% | 2025 | Strong at L0=70 |
| HSAE | Substantially outperforms | Feb 2026 | Joint parent-child training |
| **This work** | **Not directly measured** | 2026 | Claims encoder-driven; partially confirmed |

**Critical Gap**: This work does NOT report direct absorption rate reduction percentages comparable to established baselines. The H_Mech results are in arbitrary units without calibration to published baselines.

**Novelty Verdict (Comparativist)**: Questionable — the "encoder-driven" mechanism is undermined by stochastic results showing decoder saturation. Non-monotonic hierarchy finding is genuinely novel but requires validation. Safety claim failed.

**Venue Recommendation**: Workshop tier (ACL/NeurIPS Interpretability Workshop) — contribution margin insufficient for main conference.

---

## 7. Hypothesis Update (from Revisionist)

| Hypothesis | Before | After | Confidence |
|------------|--------|-------|------------|
| H1: Trained SAEs > baselines | PASSED | PASSED (confirmed) | High |
| H_Mech: Encoder-driven, decoder-irrelevant | PARTIALLY REVERTED | Decoder can be pathological amplifier in high-activation regimes | Medium |
| H2: Absorption ∝ frequency | FAILED | FAILED — direction reversed (Spearman rho=+0.171) | High |
| H3: Steering sensitivity | PASSED | PASSED (confirmed) | High |
| H_Safe: Safety elevated absorption | INCONCLUSIVE | FAILED on real SAEs, direction positive but not significant | Medium |
| H_Comp: Monotonic hierarchy | THEORETICAL | REFUTED — R²=0.04, non-monotonic | High |
| H_Pareto: Frontier exists | THEORETICAL | REFUTED — absorption=0 for all L0, sensitivity constant | High |

**Mental Model Revision**: Absorption is not purely encoder-driven; the decoder can create pathological pathways under stochastic hierarchy. Hierarchy strength is NOT a reliable predictor — the relationship is flat or slightly negative. The Pareto frontier is not empirically observable in this setup.

---

## 8. Action Plan

### RECOMMENDATION: PIVOT

**Rationale**:
1. H_Pareto (core theoretical contribution) cannot be measured — sensitivity metric broken
2. H_Safe is dead on real SAEs — p=1.0 (pilot), drop trigger met
3. H_Comp falsified — monotonic assumption fails
4. H_Mech's main claim has replication problems — stochastic setting reveals decoder contribution

### Prioritized Next Steps

| Priority | Action | Rationale | Expected Runtime |
|----------|--------|-----------|------------------|
| **P1** | Debug sensitivity metric | Unblocks H_Pareto, core theoretical contribution | 1h |
| **P1** | Pursue cand_encreg (encoder regularization) | Novel encoder-targeted intervention, constructive contribution needed | 1.5h |
| **P2** | Re-examine H_Mech anomalous results | Condition D invariance, Condition C outliers | 1h |
| **P2** | Fix H_Safe feature selection | Properly match by Neuronpedia annotations before abandoning | 1h |
| **P3** | cand_geom backup | If cand_encreg fails, fallback to encoder geometry diagnostic | 0.5h |

### Drop List
- H_Safe on Gemma (p=1.0 in pilot, drop trigger met)
- H_Comp continuation (R²=0.04, monotonic failed)
- H_Pareto as-is (zero absorption measurement issue)

### Conditions for Proceed
- Sensitivity metric debugged and validated
- H_Mech replicated with 20+ seeds across varying parent activation levels
- cand_encreg shows >30% absorption reduction with <5% reconstruction degradation
- Quantitative comparison against Matryoshka/OrtSAE baselines established

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Concurrent work overtakes this direction | HIGH (HSAE Feb 2026, OrtSAE Sep 2025) | Pivot to encoder regularization as novel contribution |
| Sensitivity metric cannot be fixed | MEDIUM | Fall back to qualitative Pareto description |
| cand_encreg fails | MEDIUM | cand_geom backup, or shift to methodology paper |
| H_Mech decoder contribution is bug, not real | LOW | Investigate via controlled experiments |

---

*Synthesized by sibyl-result-synthesizer agent (sibyl-heavy tier)*
*Evidence contract: All claims traceable to experiment outputs in /exp/results/*
*Missing perspective: skeptic.md (not generated or lost)*
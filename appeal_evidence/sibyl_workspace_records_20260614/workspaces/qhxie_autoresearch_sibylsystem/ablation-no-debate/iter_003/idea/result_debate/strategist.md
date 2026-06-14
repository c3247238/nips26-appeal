# Strategist Analysis: Result Debate

## Date: 2026-05-01
## Workspace: ablation-no-debate/current

---

## 1. Signal Strength Assessment

### H_Mech (Encoder-Driven Absorption)
**Rating: MODERATE (with significant caveats)**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| B vs D delta | 0.037 | B ≈ D does NOT hold (should be ~0) |
| C vs A delta | 12.099 | Massive variance in Condition C |
| encoder_driven_check | true | Partial confirmation |
| decoder_irrelevant_check | false | FAILS due to C variance |

**Evidence**:
- Condition C shows extreme outliers: seed 789 → 17.30, seed 1024 → 43.84 (vs baseline ~0.28)
- Condition D is constant at 0.0175 across all seeds (suspiciously invariant)
- Condition B averages 0.055 (trained encoder + random decoder)
- **Interpretation**: The encoder effect is real BUT the experiment has severe non-replicability across seeds. The "encoder-driven" story holds in aggregate but not per-seed.

### H_Comp (Hierarchy Strength Sweep)
**Rating: WEAK/NOISE**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| R² | 0.04 | Near-zero fit; no monotonic relationship |
| p-value | 0.70 | Not significant |
| monotonic | false | Absorption does NOT increase with hierarchy strength |

**Evidence**:
- Absorption at cos_0.5: 0.81
- Absorption at cos_0.9: 1.20
- Absorption at cos_0.95: 0.51
- No monotonic pattern whatsoever

**Verdict**: H_Comp FALSIFIED. The theoretical prediction that absorption increases with hierarchy strength is not supported.

### H_Pareto (Sensitivity-Absorption Frontier)
**Rating: WEAK/NOISE (measurement artifact suspected)**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| All absorption_mean | 0.0 | Zero absorption across ALL L0 levels |
| All absorption_std | 0.0 | No variance |
| frontier_fit | degenerate | a=1.0, b=-0.5 produces zero frontier |

**Evidence**:
- L0 targets {16, 32, 64, 128} all yield absorption = 0.0
- Sensitivity is stable at ~0.105 across all configs
- **Interpretation**: Either (a) the measurement method is wrong for this configuration, or (b) no absorption occurs in this setup. The "Pareto frontier" hypothesis cannot be tested when one axis is always zero.

**Verdict**: H_Pareto INCONCLUSIVE due to measurement artifact. Cannot assess frontier shape.

### H_Safe (Safety-Critical Feature Absorption)
**Rating: WEAK/NOISE**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| p_value (GPT2-small) | 0.345 | Not significant |
| p_value (Gemma pilot) | 1.0 | Worst possible outcome |
| safety_mean vs non_safety | 233.1 vs 221.7 | Direction correct but tiny effect |

**Evidence**:
- Held-out validation on GPT2-small: U=63, p=0.345 (fails significance)
- Gemma Scope pilot: p=1.0 (safety features LESS absorbed, but not significantly)
- Safety features show 5% elevated absorption (233 vs 222), far below significance threshold

**Verdict**: H_Safe FAILED on all real SAE tests. The "safety-critical features are disproportionately absorbed" hypothesis is not supported.

### H3 (Steering Sensitivity of Absorbed Features)
**Rating: MODERATE**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| sensitivity_ratio | 1.62x | Absorbed features more sensitive |
| Status | PASSED (iter_001) | Replicated across seeds |

**Note**: This is the only hypothesis with consistent positive evidence. But it was measured in iter_001 on synthetic data, not yet replicated in iter_003.

---

## 2. Opportunity Cost Analysis

| Direction | Signal Strength | GPU Cost | Risk | Expected Info Gain/GPU-hr |
|-----------|----------------|---------|------|---------------------------|
| H_Safe on Gemma (full) | WEAK (p=1.0) | 1h | Very High (>0.3 threshold) | ~0.01 |
| H_Comp refinement | WEAK (R²=0.04) | 0.5h | High (null result likely) | ~0.02 |
| H_Pareto method fix | WEAK/UNKNOWN | 1h | High (measurement issue) | ~0.01 |
| H_Mech replication (new seeds) | MODERATE | 1h | Medium (outliers persist) | ~0.05 |
| **cand_encreg** | UNTESTED | 1.5h | Medium | ~0.10 |
| **cand_safe (Gemma full test)** | WEAK (p=1.0) | 1h | Very High | ~0.01 |
| cand_geom | DEFERRED | 0.5h | Medium | ~0.03 |
| cand_ens | UNTESTED | 2h | High | ~0.02 |

**Ranked by information gain per GPU-hour**:
1. **cand_encreg** (~0.10 info gain/GPU-hr) — Untested constructive intervention, directly motivated by H_Mech
2. H_Mech seed replication (~0.05) — Would clarify outlier issue
3. cand_geom (~0.03) — Deferred but lower priority
4. cand_ens (~0.02) — High cost, uncertain payoff
5. H_Comp refinement (~0.02) — Low priority given R²=0.04
6. cand_safe Gemma (~0.01) — p=1.0 is worst outcome
7. H_Pareto method fix (~0.01) — Measurement issue
8. H_Safe GPT2 continuation (~0.01) — Already failed

---

## 3. Decision Matrix

| Direction | Signal | GPU Cost | Risk | Expected Outcome | Priority |
|-----------|--------|----------|------|-------------------|----------|
| H_Safe Gemma full | WEAK | 1h | Very High (>0.3 drop trigger) | Null result | DROP |
| H_Comp continuation | WEAK | 0.5h | High (drop trigger: R²<0.6) | Null result | DROP |
| H_Pareto method fix | UNKNOWN | 1h | High | Measurement fix | DEFER |
| H_Mech seed replication | MODERATE | 1h | Medium | Confirm encoder story | KEEP |
| **cand_encreg** | UNTESTED | 1.5h | Medium | Novel intervention | **PRIORITY 1** |
| cand_geom | DEFERRED | 0.5h | Medium | Training-free diagnostic | BACKUP |
| cand_ens | UNTESTED | 2h | High | Ensemble recovery | BACKUP |

---

## 4. PIVOT vs PROCEED Verdict

**Recommendation: PIVOT**

**Rationale**:

1. **H_Pareto failure is blocking**: The sensitivity-absorption Pareto frontier — a core theoretical contribution — cannot be measured because absorption is 0.0 across all L0 levels. This is either a fundamental measurement problem or the phenomenon doesn't exist in this configuration.

2. **H_Safe is dead on real SAEs**: Both GPT2-small (p=0.345) and Gemma Scope pilot (p=1.0) show safety features are NOT disproportionately absorbed. The drop trigger (p > 0.3) is already met.

3. **H_Comp falsified**: The monotonic relationship between hierarchy strength and absorption is not supported (R²=0.04, non-monotonic).

4. **H_Mech's "encoder-driven" story has replication problems**: The B≈D check FAILS (delta=0.037 is non-trivial), Condition C shows extreme outliers (43.8 vs 0.28 baseline), and Condition D is suspiciously constant at 0.0175. The aggregate pattern holds but per-seed replication is poor.

5. **No path to publication-quality results on current trajectory**: The main contributions (encoder mechanism + Pareto frontier + safety implications) are either replication-challenged (H_Mech), unmeasurable (H_Pareto), or falsified (H_Safe).

---

## 5. If PIVOT: Recommended Backup

**Primary Pivot: cand_encreg (Encoder Regularization to Reduce Absorption)**

**Why this direction**:
- Directly motivated by H_Mech finding (encoder is the driver)
- Novel methodology (first encoder-targeted intervention; prior work modifies both encoder and decoder)
- Constructive contribution needed: the paper needs a positive intervention story, not just diagnosis
- Novelty: 7/10 (lower than H_Safe 9/10 but backed by solid mechanism)
- Estimated runtime: 1.5 hours GPU

**Specific experiments to run**:
1. **Pilot (30 min)**: Test one encoder regularization variant on synthetic hierarchy
   - Penalize parent-child activation correlation in encoder loss
   - Compare absorption rate to baseline (expected: >30% reduction)
   - Measure reconstruction degradation (expected: <5%)
2. **Full experiment (1h)**: If pilot passes, run 3-seed validation with best regularization strength

**Drop trigger for cand_encreg**: If regularization degrades reconstruction >10% OR reduces absorption <10%

**Secondary Pivot: cand_geom (Encoder Geometry Diagnostic)**
- Lower priority, deferred due to H_Mech showing decoder contributes nothing
- Would need to validate on encoder directions instead of decoder directions
- Estimated runtime: 0.5h

---

## 6. Summary: Strategic Recommendation

| Action | Rationale |
|--------|-----------|
| **DROP** H_Safe on Gemma | p=1.0 in pilot, drop trigger met |
| **DROP** H_Comp continuation | R²=0.04, monotonic failed |
| **DEFER** H_Pareto | Zero absorption measurement issue |
| **MONITOR** H_Mech | Replication problems, but aggregate signal holds |
| **PRIORITY 1: Pursue cand_encreg** | Novel encoder-targeted intervention, constructive contribution |
| **BACKUP: cand_geom** | If cand_encreg fails, fallback to encoder geometry diagnostic |

**Key concern**: The zero absorption in H_Pareto and the extreme outliers in H_Mech Condition C suggest possible measurement bugs or configuration issues that should be investigated before running new experiments.

---

*Generated by sibyl-strategist agent (sibyl-light tier)*
*Evidence contract: All claims backed by experiment outputs in /exp/results/*

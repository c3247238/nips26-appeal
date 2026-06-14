# Experiment Result Analysis

## Key Results Summary

| Hypothesis | Status | Key Metrics |
|------------|--------|-------------|
| H_Mech (encoder-driven absorption) | PARTIAL FAIL | B≈D holds (delta=0.037), but C≠A (delta=12.10) — decoder contributes in high-activation regimes |
| H_Comp (monotonic hierarchy) | FALSIFIED | R²=0.04, non-monotonic, peak at cos~0.85-0.90 |
| H_Pareto (sensitivity-absorption frontier) | UNINTERPRETABLE | All sensitivity=3.018 (constant), absorption=0 across all L0 levels |
| H_Safe (safety-critical features) | FAILED | GPT2 p=0.345, Gemma pilot p=1.0, direction positive but not significant |
| H3 (steering sensitivity) | PASSED | 1.62x sensitivity improvement, only confirmed from iter_001 |

**Result Quality Score: 3/10** — Three of four main hypotheses are falsified, broken, or failed.

## Debate Perspectives Summary

- **Optimist**: H_Mech partially confirmed (encoder is primary driver), decoder shows "saturation amplification" at high parent activations. Non-monotonic hierarchy strength is genuinely novel finding (inverted U at cos~0.85-0.90). H_Safe direction positive but underpowered (needs 50/group for significance). Safety result requires larger sample.

- **Skeptic**: (File not available — not generated or lost in iter_004)

- **Strategist**: PIVOT recommended. H_Pareto failure is blocking (core theoretical contribution unmeasurable). H_Safe dead on real SAEs (p=1.0 in pilot, drop trigger met). H_Comp falsified. H_Mech has severe replication problems (Condition C outliers: 17.3, 43.8). Prioritized backup: cand_encreg (encoder regularization).

- **Comparativist**: Contribution margin MARGINAL (<1%). H_Pareto broken (constant sensitivity). H_Safe failed. H_Mech replication-challenged. No quantitative comparison against Matryoshka (90% reduction) or OrtSAE (65% reduction). Venue recommendation: Workshop tier only.

- **Methodologist**: Methodology Score: 2/5. H_Pareto sensitivity metric BROKEN (identical values 3.018 across all L0). H_Safe feature selection invalid ([1024, 2048...] arbitrary indices). Condition D suspiciously invariant at 0.0175 (5 seeds = exact same value). Falsification criteria not being applied. Priority fixes: debug sensitivity metric, fix H_Safe feature selection, investigate H_Mech anomalies.

- **Revisionist**: H_Mech partially reverted — decoder contributes non-zero under stochastic hierarchy (not deterministic). NH1 generated: decoder pathological absorption requires specific hierarchy decoding geometry. NH2: absorption=0 due to feature sparsity (branching factor artifact). NH3: safety absorption is cross-model inconsistent.

## Analysis

### 1. Method Feasibility

**Assessment: IMPAIRED**

The core experimental design (2x2 factorial decomposition) is sound, but critical metrics are broken:
- Sensitivity metric produces constant values (3.018) across all L0 levels, making H_Pareto uninterpretable
- H_Safe measurement on Gemma Scope yields zero absorption for all features, suggesting measurement failure
- Condition D shows suspicious invariance across all 5 seeds (identical 0.0175), suggesting potential aggregation bug

The mechanistic story (encoder-driven absorption) is partially confirmed but complicated:
- Condition B (trained encoder + random decoder = 0.055) ≈ Condition D (both trained = 0.017) confirms encoder alignment
- BUT Condition C (random encoder + trained decoder = 12.28 mean) >> Condition A (0.184) reveals decoder CAN amplify in some seeds
- Extreme outliers in Condition C (seeds 789: 17.3, 1024: 43.8) suggest decoder "short-circuit" in high-activation regimes

### 2. Performance

**Assessment: WEAK**

No quantitative absorption reduction reported compared to established baselines:
- Matryoshka SAE: 90% absorption reduction (0.49 → 0.05)
- OrtSAE: 65% absorption reduction
- This work: H_Mech results in arbitrary units (B=0.055, D=0.017) without calibration

The hypotheses that COULD show performance (H_Pareto, H_Safe) are either broken or failed.

### 3. Improvement Headroom

**Assessment: LIMITED**

Path to main-conference publication requires:
1. Debug sensitivity metric (H_Pareto core contribution) — 1h
2. Replicate H_Mech with 20+ seeds at varying parent activation levels — 1h
3. Run cand_encreg pilot (encoder regularization) — 0.5-1.5h
4. Properly power H_Safe with 50/group and Neuronpedia-matched features — 1h
5. Quantitative comparison against Matryoshka/OrtSAE baselines

Even if all fixes succeed, the expected contribution is marginal improvement over existing methods.

### 4. Time-Cost Tradeoff

**Assessment: UNFAVORABLE**

| Direction | GPU Cost | Expected Info Gain | Risk |
|-----------|----------|-------------------|------|
| Fix sensitivity metric | 1h | Medium | High (may not fix) |
| cand_encreg pilot | 1.5h | Medium | Medium (novel but unproven) |
| H_Mech 20-seed replication | 1h | Low | Medium (anomalies persist) |
| H_Safe proper matching | 1h | Low | High (p=1.0 in pilot) |

Expected total time to get publication-quality results: 4-6h additional GPU time with no guarantee of success. Meanwhile, concurrent work (HSAE Feb 2026, OrtSAE Sep 2025) has already surpassed this approach.

### 5. Critical Objections

**Assessment: FATAL for current trajectory**

1. **H_Pareto BLOCKING**: Sensitivity metric is broken — cannot claim Pareto frontier contribution
2. **H_Safe DEAD**: p=1.0 in Gemma pilot, p=0.345 in GPT2 — safety hypothesis not supported
3. **H_Comp FALSIFIED**: R²=0.04, monotonic assumption does not hold
4. **H_Mech REPLICATION PROBLEMS**: Stochastic hierarchy reveals decoder contribution that deterministic hierarchy missed
5. **NO QUANTITATIVE BASELINE COMPARISON**: Cannot claim improvement over Matryoshka/OrtSAE
6. **CONCURRENT WORK RISK**: HSAE (Feb 2026) addresses hierarchical absorption with better results

## Decision Rationale

**PIVOT is required** because:

1. **Core theoretical contribution (H_Pareto) is unmeasurable**: The sensitivity-absorption Pareto frontier cannot be established with a broken metric. This was meant to be the paper's main theoretical contribution.

2. **Highest-novelty hypothesis (H_Safe) failed**: Safety-critical features are NOT disproportionately absorbed on real SAEs (Gemma pilot p=1.0). This 9/10 novelty finding is dead.

3. **No publication-quality result on current trajectory**: H_Mech is replication-challenged, H_Comp is falsified, H_Pareto is broken, H_Safe is failed. Result quality score is 3/10.

4. **Time-cost unfavorable**: 4-6h additional GPU time with no guarantee, while concurrent work has already surpassed this approach.

5. **Consensus across ALL perspectives**: Strategist, Comparativist, Methodologist, and Synthesis ALL recommend pivot or downgrade to workshop tier.

**Recommended Pivot Direction**: cand_encreg (encoder regularization to reduce absorption)
- Directly motivated by H_Mech finding (encoder is primary driver)
- First encoder-targeted intervention (prior work modifies both encoder and decoder)
- Novelty: 7/10, with constructive contribution the paper currently lacks
- Drop trigger: If regularization degrades reconstruction >10% OR reduces absorption <10%

**Alternative if cand_encreg fails**: cand_geom (encoder geometry diagnostic) or shift to methodology-only workshop paper documenting boundary conditions.

## DECISION: PIVOT

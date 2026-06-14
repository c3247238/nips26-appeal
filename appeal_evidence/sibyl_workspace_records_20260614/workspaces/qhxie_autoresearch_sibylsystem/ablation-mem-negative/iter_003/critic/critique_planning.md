# Planning Critique: Experimental Design Review

> Critic: sibyl-critic (heavy tier)
> Date: 2026-04-29
> Scope: methodology.md + plan/task_plan.json + plan/pilot_plan.json

---

## Overall Planning Assessment

**Strength**: The experimental plan is well-structured with clear pilot/full separation, pre-registered pass/fail criteria, and honest go/no-go gates. The pivot from Iteration 2 to Iteration 3 shows good planning adaptability.

**Weakness**: The plan was designed for a positive-result framing ("UAD works") and was not fully re-calibrated when the negative result emerged. Several planned experiments were skipped with weak justification.

---

## 1. Pilot Design Review

### P1: Collision Proxy Validation

**Design**: Validate collision rate as proxy for absorption rate on first-letter hierarchy (10 pairs).

**Result**: PASS (r = 0.711, CI excludes 0)

**Critique**:
- The pilot uses first letters (a-z), but the full experiment switches to numbers and punctuation. This is a design inconsistency.
- The pilot summary notes a critical concern: "feature 11746 dominates all 26 letters, suggesting a first-letter super-feature." This concern was not carried forward into the full experiment design.
- The pilot's "ground truth" (shared top-10 features) may be too permissive, but this was not addressed before proceeding to full experiments.

**Verdict**: Adequate pilot, but concerns raised were not acted upon.

### P2: UAD Reproducibility

**Design**: Reproduce UAD on pretrained SAE with number hierarchy.

**Result**: FAIL (F1 = 0.0, Recall = 14.3%)

**Critique**:
- The FAIL result is clear and well-documented.
- Root cause analysis (token-level mutual exclusivity) is provided with evidence.
- However, the pilot only tests ONE hierarchy type (numbers). A second hierarchy type in the pilot would have strengthened the generalization claim.

**Verdict**: Good pilot, but limited scope.

### P3: Random Baseline

**Design**: Compare UAD to random baselines.

**Result**: FAIL (UAD F1 = same-cluster random F1)

**Critique**:
- The same-cluster random baseline is well-designed.
- However, the "frequency-weighted random" baseline achieves F1 = 0.0, which is worse than global random (F1 = 0.00011). This suggests the frequency weighting is miscalibrated or the baseline is not meaningful.
- The pass criterion "UAD F1 - random F1 >= 0.3" is absurdly high for this domain. With 7 GT pairs out of ~125,000 possible pairs, the maximum achievable F1 is bounded by the base rate. A more realistic criterion would have been specified.

**Verdict**: Baseline design is good, but pass criteria poorly calibrated.

---

## 2. Full Experiment Design Review

### F1: UAD Multi-SAE

**Status**: SKIPPED

**Justification**: "P2/P3 failure made this moot"

**Critique**: This is weak justification. Testing UAD on multiple SAEs would have strengthened the generalization claim. Even if UAD fails on one SAE, showing it fails on multiple SAEs would make the negative result more robust. The "moot" justification assumes the failure is architecture-independent without evidence.

**Verdict**: Should have been run. Would have added ~30 minutes of compute time.

### F2: UAD Ablations

**Status**: COMPLETE

**Results**: All variants fail. Best: K-means F1 = 0.0037, Recall = 85.7%.

**Critique**:
- Comprehensive ablation design (6 variants).
- The K-means result (85.7% recall) is interesting but not analyzed further.
- Missing: ablation of cluster count (k=50 fixed; what about k=10, k=100?)
- Missing: ablation of feature selection count (500 fixed; what about 100, 1000?)

**Verdict**: Good ablation study, but missed opportunities for deeper analysis.

### F3: UAD Multi-seed

**Status**: SKIPPED

**Justification**: "P2/P3 failure made this moot"

**Critique**: Multi-seed validation is especially important for a negative result. If UAD's F1 varies wildly across seeds, the conclusion is fragile. If it is consistently near-zero, the conclusion is strengthened. Skipping this removes an important robustness check.

**Verdict**: Should have been run. Would have added ~5 minutes of compute time.

### F4: Extended Collision Correlation

**Status**: COMPLETE

**Results**: Spearman r = 0.869, n=56, CI=[0.780, 0.938]

**Critique**:
- Good expansion from 10 to 56 pairs.
- Two hierarchy types (numbers, punctuation) provide some diversity.
- However, the combined correlation (0.869) is inflated by between-hierarchy variance (see Experiment Critique Section 9).
- Missing: a semantic hierarchy (animal/dog) that would test whether collision rate works for non-token-disjoint concepts.

**Verdict**: Adequate but could be stronger with semantic hierarchy.

### F5: False Positive Analysis

**Status**: COMPLETE

**Results**: 99.98% FP rate, token-level mutual exclusivity confirmed.

**Critique**:
- Good root cause analysis.
- The three "failure modes" are well-articulated.
- However, the analysis is post-hoc (not pre-registered) and based on a single example.

**Verdict**: Good qualitative analysis, but limited quantitative depth.

### F6: DFDA Validation

**Status**: SKIPPED

**Justification**: "UAD cannot identify pairs"

**Critique**: DFDA (Dead Feature Distribution Analysis) was part of the original UAD+DFDA proposal. Skipping it is reasonable since UAD fails, but the paper should acknowledge that DFDA was not tested.

**Verdict**: Reasonable skip.

---

## 3. Planning Decisions: What Was Right and Wrong

### Right Decisions

1. **Pivoting to negative result**: When pilots failed, the plan correctly pivoted rather than forcing a positive result.

2. **Running ablations**: The F2 ablation study provides valuable evidence that the failure is structural, not hyperparameter-related.

3. **Validating collision rate**: The F4 experiment provides a constructive positive finding to balance the negative result.

4. **Honest reporting**: All results are reported without spin, including the limitations.

### Wrong Decisions

1. **Skipping F1 (Multi-SAE)**: Would have strengthened generalization claims with minimal cost.

2. **Skipping F3 (Multi-seed)**: Would have provided robustness check for the negative result.

3. **Not testing semantic hierarchies**: The universal claim about mutual exclusivity is unsupported without semantic hierarchy tests.

4. **Not analyzing K-means success**: The K-means ablation achieving 85.7% recall is the most interesting result but was not analyzed.

5. **Not re-calibrating pass criteria**: The pass criteria (F1 >= 0.5, UAD-random >= 0.3) were designed for positive-result framing and are absurdly high for this domain.

---

## 4. Resource Allocation

| Experiment | Planned | Actual | Value |
|------------|---------|--------|-------|
| P1: Collision proxy | 15 min | 25s | High |
| P2: UAD reproduce | 15 min | 149s | High |
| P3: Random baseline | 10 min | 151s | High |
| F1: Multi-SAE | - | SKIPPED | Would be Medium |
| F2: Ablations | - | 148s | High |
| F3: Multi-seed | - | SKIPPED | Would be Low |
| F4: Extended correlation | - | 8s | High |
| F5: FP analysis | - | <1s | Medium |
| F6: DFDA | - | SKIPPED | N/A |

**Total runtime**: ~8 minutes

The experiments are extremely fast. There is no computational reason to skip F1 or F3.

---

## 5. Suggested Additional Experiments

### High Priority (should run before submission)

1. **Multi-seed UAD**: Run UAD with seeds 0, 123, 999. Report variance in F1. (~5 min)

2. **K-means analysis**: Investigate why K-means achieves 85.7% recall. What property does it use? (~15 min)

3. **Filter no-op investigation**: Report how many features/pairs were actually filtered by dead feature and phi filtering. (<5 min)

### Medium Priority (would strengthen paper)

4. **Semantic hierarchy test**: Test on one semantic hierarchy (e.g., animal/dog/cat) to check mutual exclusivity claim. (~15 min)

5. **Different layer**: Test layer 4 or 12 to check feature structure variation. (~15 min)

### Low Priority (nice to have)

6. **Multi-SAE**: Test on a different SAE architecture. (~30 min)

---

## Summary: Planning Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Pilot design | 7/10 | Good structure, but concerns not acted upon |
| Full experiment coverage | 6/10 | Skipped experiments that should have run |
| Pass/fail criteria | 5/10 | Poorly calibrated for negative-result framing |
| Resource efficiency | 9/10 | Fast experiments, good use of compute |
| Adaptability | 8/10 | Good pivot from positive to negative result |
| Constructive balance | 7/10 | Good mix of negative and positive findings |
| **Overall** | **7/10** | Good planning with some missed opportunities |

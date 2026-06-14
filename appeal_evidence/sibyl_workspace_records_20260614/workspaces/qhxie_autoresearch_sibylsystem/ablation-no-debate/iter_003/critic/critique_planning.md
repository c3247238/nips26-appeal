# Critique: Planning and Methodology

## Overview

The planning document (plan/methodology.md) is comprehensive with good experimental design rationale. However, there are critical gaps between what is planned, what is claimed, and what is actually executed.

---

## Critical Issues

### 1. Plan Claims 5 Seeds, Execution Has 1

**Location**: plan/methodology.md vs actual execution

**Planned** (methodology.md):
- H_Mech: "5 seeds × 4 conditions = 20 runs"
- H_Comp: "3 seeds for robust fit"
- H_Pareto: "3 seeds"

**Actual**:
- h_mech_pilot_seed42.json: Only seed 42
- h_comp_pilot.json: Only seed 42
- h_pareto_pilot.json: Only seed 42

The pilot JSON files consistently show only a single seed (42), not the multiple seeds the methodology promises.

**Impact**: The methodology claims robust multi-seed validation but only single-seed pilots exist. The paper claims "multi-seed validation: Across 5 seeds, Condition B consistently produces higher absorption than Condition D" but the actual data supporting this claim is unclear.

---

### 2. Experiment State Shows Tasks Still Running

**Location**: exp/experiment_state.json

```json
"h_mech_full": {"status": "running", "gpu_ids": [0], ...},
"h_pareto_pilot": {"status": "running", "gpu_ids": [3], ...}
```

The h_mech_full (which should be the multi-seed validation of H_Mech) and h_pareto_pilot are still RUNNING according to the experiment state. But the paper presents results as if they are complete.

**Gap**: There is a disconnect between the planning/execution state and the claims in the paper.

---

### 3. Timeline is Unrealistic

**Location**: plan/methodology.md lines 104-112

**Plan**:
| Phase | Task | Duration |
|-------|------|----------|
| 1 | H_Mech factorial (5 seeds, stochastic hierarchy) | 45 min |
| 2 | H_Comp: hierarchy strength sweep | 30 min |
| 3 | H_Pareto: sensitivity-absorption frontier | 45 min |
| 4 | H_Safe on Gemma Scope | 60 min |
| 5 | Held-out validation | 30 min |

**Total**: 3.5 hours

**Reality Check**:
- 5 seeds × 4 conditions × 100 samples (H_Mech) cannot complete in 45 minutes
- SAELens + Gemma Scope loading alone can take 10+ minutes
- "Held-out validation" is vague and unspecified

The timeline significantly underestimates task complexity.

---

## Major Issues

### 4. Task Dependencies Don't Match Execution

**Location**: plan/methodology.md lines 167-181

**Planned Dependency Graph**:
```
[setup_sae_env]
       ↓
   ┌───┴───┬─────────────┐
   ↓       ↓             ↓
[h_mech] [h_comp]    [h_pareto]
   │       │             │
   └───────┴─────────────┤
            ↓            │
       [h_safe_gemma]◄───┘
              ↓
       [held_out_validation]
```

**Actual State**:
- h_safe_gemma_pilot: completed
- h_mech_pilot: completed
- h_mech_full: running
- h_pareto_pilot: running

The dependency graph shows h_safe_gemma depends on others completing first, but h_safe_gemma_pilot is already completed. The execution order doesn't match the planned dependency structure.

---

### 5. Falsification Criteria Table is Inconsistent

**Location**: plan/methodology.md lines 132-141

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| H_Mech | Condition B ≈ Condition D, C ≈ A | PASSED (pilot) |
| H_Comp | R² > 0.8 for monotonic fit | NOT TESTED |
| H_Pareto | Detectable frontier shape | NOT TESTED |
| H_Safe | Mann-Whitney p < 0.05 | NOT TESTED |
| H2 (archive) | rho < -0.3 | FAILED (+0.171) |

**Issue**: The table says H_Comp and H_Pareto are "NOT TESTED" but the paper Section 4.2 and 4.3 report CONFIRMED results for both with specific metrics (R² = 0.984 for H_Comp, Δ = 0.179 for H_Pareto).

This suggests either:
- The methodology.md wasn't updated after experiments ran, OR
- The paper is reporting results from experiments that haven't been marked as complete in the experiment state

---

### 6. H_Safe Execution Did Not Follow Plan

**Location**: plan/methodology.md Section 2.4 vs h_safe_gemma_pilot.json

**Planned**:
1. Load Gemma Scope SAE via SAELens
2. Select 20 safety-relevant features from Neuronpedia annotations
3. Match 20 non-safety features by activation frequency ±10%
4. Measure absorption via multi-child proportional method
5. Mann-Whitney U test

**Actual**:
```json
"n_safety_features": 5,
"n_non_safety_features": 5,
"safety_feature_indices": [1024, 2048, 3072, 4096, 5120],
```

Only 5 features each (not 20), placeholder indices (not Neuronpedia-validated), and no matching by activation frequency.

**Impact**: The actual experiment deviates significantly from the plan, but the paper presents it as a valid pilot finding.

---

## Minor Issues

### 7. Pilot vs Full Experiment Confusion

The plan distinguishes between pilot and full experiments:
- h_mech_pilot (seed 42 only) vs h_mech_full (should be 5 seeds)
- h_pareto_pilot (2 L0 levels) vs presumably h_pareto_full (4 L0 levels)

But it's unclear if h_pareto_full exists or what its state is. The h_pareto_pilot_DONE marker exists but the pilot had only L0=16 and L0=64 (2 levels, not the 4 promised).

---

### 8. Figures Referenced but Not Present

The plan lists expected visualizations (Figure 1-5) but the actual figure generation status is unclear. The full/figures directory was empty in the file listing.

---

## Recommendations

### Must Fix
1. **Execute the multi-seed validation as planned**: Complete h_mech_full with all 5 seeds before claiming multi-seed robustness
2. **Update methodology.md to reflect actual execution**: The falsification criteria table should match what was actually tested
3. **Either follow H_Safe plan or update the plan**: 5 placeholder features is not "20 Neuronpedia-validated features"

### Should Fix
4. **Revise timeline to be realistic**: 3.5 hours for all experiments is insufficient
5. **Update experiment state properly**: Mark experiments as completed when done so planning and state are consistent
6. **Clarify pilot/full distinction**: Make it clear which results come from pilots vs full experiments

### Consider
7. **Add execution log tracking**: Document actual execution times vs planned times for better future planning
8. **Specify held-out validation more concretely**: What exactly does this mean operationally?
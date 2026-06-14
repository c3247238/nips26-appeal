# Planning Critique: Feature Absorption Study

## Overall Assessment

The planning phase produced a well-structured experimental design with appropriate priorities, but the plan was not fully executed. Key planned experiments (H_Safe, H_Mech, multi-seed validation) were never run, and the time budget estimates were optimistic given the actual implementation challenges encountered.

---

## Critical Issues

### 1. Plan Not Executed (CRITICAL)

**Evidence**: task_plan.json lists 6 tasks with estimated 2.75 hours total. The actual executed experiments appear to be only the original pilot tasks (p1_multichild_absorption, p1_frequency_correlation, p1_steering) from the first iteration.

**Planned but not executed**:
- h_safe_pilot (20 min) -- highest novelty, no training needed
- h3_fix_pilot (30 min) -- debug broken steering
- multichild_multiseed_pilot (30 min) -- address zero-variance
- h_mech_factorial (45 min) -- decompose geometric vs learned
- h_safe_full (30 min) -- extended safety analysis

**Analysis**: The paper presents results as if the full plan was executed, but only the initial pilot was completed. This is a significant gap between planned and delivered work.

**Fix**: The paper must clearly state which experiments were actually run and which are planned future work.

---

### 2. H3 Fix Deprioritized Despite Being Critical for Causal Claims (MAJOR)

**Evidence**: methodology.md Priority 2 is "H3_fix - Steering Intervention Validation" with explicit debug steps. The proposal states "Required Fix: Verify steering changes activations." Yet the fix was never implemented.

**Analysis**: The plan correctly identified that H3 was broken and needed fixing before publication. However, the paper was written with the broken H3 data anyway, drawing causal conclusions from it.

**Fix**: Either fix H3 before publication or remove all causal claims.

---

### 3. Time Budgets Were Optimistic (MINOR)

**Evidence**: Pilot tasks were estimated at 20-30 minutes each, but the actual pilot (p1_multichild_absorption with 100 samples) likely took longer given the SAE training component.

**Analysis**: The time estimates did not account for SAE training time (20,000 steps mentioned in methodology). Even "no training needed" tasks like H_Safe require downloading and loading Gemma Scope SAEs.

**Fix**: Time budgets should include model loading and initialization overhead.

---

### 4. Multi-Seed Validation Planned but Not Executed (MAJOR)

**Evidence**: methodology.md Priority 3 is "Multi-Seed Validation for H1" with explicit design (3 seeds: 42, 43, 44). The proposal states this is needed to "address zero-variance concern."

**Analysis**: The zero-variance issue was correctly identified as a concern, but the planned fix was never implemented. The paper presents single-seed results without acknowledging this limitation.

**Fix**: Either run multi-seed validation or prominently state that results are from a single seed.

---

## What Works Well

1. **Clear prioritization**: H_Safe correctly identified as highest novelty
2. **Appropriate baselines**: Three Korznikov-style baselines follow best practices
3. **Pass/fail criteria**: Each task has explicit pass criteria
4. **Negative result planning**: H2 archiving and H3 fallback documented

---

## Score: 5/10

Good planning undermined by lack of execution. The plan correctly identified what needed to be done, but the paper was written before doing it.

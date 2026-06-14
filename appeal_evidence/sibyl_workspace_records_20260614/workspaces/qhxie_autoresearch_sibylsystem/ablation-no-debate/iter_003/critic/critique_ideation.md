# Critique: Ideation and Proposal Quality

## Overview

The research proposal (idea/proposal.md) is well-structured with clear hypotheses, falsification criteria, and experimental plans. However, several critical concerns exist around the self-contradictory H_Mech framing and the unrealistic plan to execute all experiments within 3.5 hours.

---

## Critical Issues

### 1. H_Mech Interpretation is Mathematically Inconsistent

**Location**: proposal.md lines 22-31, hypothesis table line 53

**Issue**: The key prior finding table shows:

| Condition | Encoder | Decoder | Absorption Rate |
|-----------|---------|---------|-----------------|
| A | Random | Random | 0.299 |
| B | Trained | Random | **0.490** |
| C | Random | Trained | **0.299** |
| D | Trained | Trained | **0.484** |

The pass criterion states: "Condition B ≈ Condition D, C ≈ A"

**The Math Doesn't Work**:
- B (0.490) vs D (0.484): These ARE approximately equal (delta = 0.006) ✓
- C (0.299) vs A (0.299): These ARE equal ✓

Wait—the VALUES in lines 23-28 of proposal.md show DIFFERENT numbers than the pilot JSON:
- proposal.md: A=0.299, B=0.490, C=0.299, D=0.484
- h_mech_pilot_seed42.json: A=0.004, B=0.076, C=0.000, D=0.017

**Two different result sets**. The proposal.md table is from an earlier iteration (iter_001) while the pilot JSON is from a later one (iter_003).

The proposal appears to conflate results from different experimental runs, or the table in proposal.md was not updated to reflect the most recent data.

**Impact**: The hypothesis status table claims H_Mech "PASSED" based on values that no longer match the actual experimental results.

---

### 2. Timeline is Unrealistic

**Location**: proposal.md lines 104-112

**Table**:
| Phase | Task | Duration | Hypothesis |
|-------|------|----------|------------|
| 1 | H_Mech factorial (5 seeds, stochastic hierarchy) | 45 min | H_Mech |
| 2 | H_Comp: hierarchy strength sweep | 30 min | H_Comp |
| 3 | H_Pareto: sensitivity-absorption frontier | 45 min | H_Pareto |
| 4 | H_Safe on Gemma Scope | 60 min | H_Safe |
| 5 | Held-out validation | 30 min | Cross-validate |

**Total**: 3.5 hours GPU time

**Reality Check**:
- H_Mech full experiment (5 seeds × 4 conditions × 100 samples) cannot realistically complete in 45 minutes
- H_Safe requires SAELens + Gemma Scope loading which alone can take 10+ minutes
- "Held-out validation" is vague about what exactly this means

This timeline appears to underestimate task complexity significantly.

---

## Major Issues

### 3. H2 Failure Not Adequately Integrated

**Location**: proposal.md line 54, alternatives.md

**Issue**: H2 (frequency correlation) FAILED with +0.171 instead of expected negative correlation. The proposal notes this as a negative result but:

1. The theoretical framework (competitive exclusion) predicted negative correlation
2. The finding (positive correlation) contradicts the theory
3. The alternatives.md file mentions this as "DEFERRED" but doesn't adequately explain why the theory was wrong

The positive frequency correlation suggests efficient coding (common features are easier to represent) rather than competitive exclusion (rare features get absorbed more). This needs better theoretical integration.

---

### 4. Pivot Decision Tree Not Updated

**Location**: alternatives.md lines 119-142

The pivot decision tree references conditions that may no longer match current data:

```
H_Mech confirmed (encoder-driven)?
 /          \
No            Yes
 |            |
H_Safe       H3 replicated?
analysis     /              \
            |               No
            |                |
            v                v
     Proceed with     H_Safe on Gemma
     encoder         Scope?
```

This tree should be regenerated based on actual experimental outcomes rather than expected outcomes.

---

## Minor Issues

### 5. Hypothesis Table Inconsistency

The hypothesis table shows different status indicators across rows:
- Some have "Status" filled (PASSED, FAILED)
- Some have "NOT TESTED"
- The "Front-runner" in metadata is "cand_p1 (Encoder-Driven Absorption with Safety Validation)"

But the proposal says H_Safe is highest novelty at 9/10 while the H_Mech result (encoder-driven) is the actual confirmed finding that should be the front-runner.

---

## What the Proposal Does Well

1. **Clear hypothesis structure**: Each hypothesis has explicit falsification criteria
2. **Good experimental design**: 2×2 factorial is the correct approach for mechanism decomposition
3. **Honest negative result reporting**: H2 failure is documented with actual vs expected values
4. **Perspectives integration table**: Shows how 6 different perspectives contributed to the proposal
5. **Novelty assessment**: Clear differentiation from prior work

---

## Recommendations

### Must Fix
1. **Reconcile proposal.md data with actual experimental results**: The H_Mech table in proposal.md shows values from iter_001 but the pilot JSON shows iter_003 values
2. **Update pivot decision tree** to reflect actual outcomes
3. **Revise timeline** to be more realistic

### Should Fix
4. **Better integrate H2 failure** into theoretical framework discussion
5. **Clarify front-runner hypothesis** based on what was actually confirmed

### Consider
6. **Add uncertainty quantification** to the sensitivity-absorption frontier measurements
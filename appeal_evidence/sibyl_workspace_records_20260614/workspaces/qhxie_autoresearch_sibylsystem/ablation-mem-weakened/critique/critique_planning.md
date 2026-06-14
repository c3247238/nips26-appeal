# Critique of Planning: Feature Absorption Project

## Overview

The planning process went through 7+ iterations but did not adapt quickly to falsified hypotheses. The result is a paper that promises a predictive tool but delivers only theoretical speculation.

## Critical Issues

### 1. No Pivot After H6 Falsification (Critical)

**Timeline**:
- Iteration 1-4: H1-H4 null results
- Iteration 3: H6 (decoder graph) falsified
- Iteration 4-5: Continued with H7-H10
- Iteration 7: Final paper still anchored on the graph

**Problem**: H6 was falsified in iteration 3 (precision@20=0.0). The project continued without pivoting away from the graph-based approach.

**What should have happened**: After H6 falsification, the planning should have pivoted to either:
1. Metric validation study (H10 showed metric sensitivity)
2. Theoretical paper without empirical validation
3. Precision-recall asymmetry as the main contribution

**Verdict**: Planning did not adapt to falsified hypotheses.

### 2. H9 Wasted an Iteration (Minor)

**H9** (co-occurrence): Identified as definitional tautology in pilot_summary.md.

**Problem**: This was labeled "NO_GO" but still went through a full experiment iteration. The tautological nature was known from the start.

**Verdict**: Filter tautological hypotheses earlier in planning.

### 3. Resource Allocation Was Inefficient (Major)

**Total experiments**: 10 major analyses across multiple layers and models.

**Yield**:
- 1 robust replicable finding (H5: precision-recall asymmetry)
- 0 statistically significant results after MCP
- 1 falsified primary hypothesis (H6)
- 1 tautology (H9)
- 1 ambiguous result (H10)

**Problem**: 10 experiments to produce 1 robust finding is low yield.

**Verdict**: Future planning should set a higher bar for continuing in a given direction.

### 4. No Pre-Registration (Major)

**Problem**: H1b (delta-corrected steering) emerged as "significant" only in post-hoc analysis. If it was the primary hypothesis, it should have been pre-registered.

**Verdict**: Pre-register primary hypotheses to avoid post-hoc cherry-picking.

## What Works

1. **Clear research questions**: RQ1-RQ5 are well-defined.

2. **Cross-layer validation**: Good practice to check consistency.

3. **Multiple metric types**: Steering, probing, CMI all measured.

## Recommendations

1. **Set pivot triggers**: If primary hypothesis falsified by iteration 3, pivot.
2. **Pre-register hypotheses**: Commit to H1 before seeing data.
3. **Filter tautologies**: Check whether H9-type hypotheses are definitional before running.
4. **Resource budget**: Set a max iteration limit and evaluate at each step.

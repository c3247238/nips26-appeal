# Critique of Planning: Feature Absorption as Optimal Compression

## Overview

The planning documents are comprehensive and well-organized. The methodology.md and proposal.md cover the experimental design thoroughly. However, there are critical gaps: the plan does not account for the possibility of zero significant results, and the alternative pivot options are underspecified.

## Critical Issues

### 1. Plan Does Not Anticipate Zero Significant Results (CRITICAL)

**Problem**: The plan assumes H1-H4 will show significant correlations (or at least some will). It does not have a contingency for the scenario where all tests fail MCP. The plan says "H1-H4: SUPPORTED (null hypothesis)" in the expected outcomes table, but this is a post-hoc reframe — the plan was designed to find significant effects, not to document null results.

**Timeline analysis**:
- Iteration 1: H1-H4 null, but framed as "supported null hypothesis"
- Iteration 2-4: Zero new significant results
- Iteration 8: Still zero significant results after MCP

**Verdict**: The plan should have pre-specified: "If no results survive MCP, reframe as null-result study with H5 (precision-recall asymmetry) as the main empirical contribution."

### 2. H9 Included Despite Known Tautology (CRITICAL)

**Problem**: H9 co-occurrence measurement is a definitional tautology (p_11 + absorption = 1.0). This was known from iteration 1. The plan still includes H9 as a hypothesis in iteration 8.

**Verdict**: H9 should have been removed from the plan at iteration 1. Its inclusion pollutes the hypothesis count (12 tests instead of 11) and creates unnecessary multiple comparison burden.

### 3. Alternative Pivot Options Underspecified (MAJOR)

**Problem**: The alternatives.md lists 6 pivot options but does not specify trigger conditions or execution plans. "When to pivot" sections are vague ("if optimal compression framing rejected").

**Verdict**: Pre-specify exact trigger conditions for each pivot. Example: "If H6 falsification is judged fatal by reviewers, pivot to Alternative A (Metric Validation Study). Execution: report trained vs. random comparison, remove graph claims."

### 4. H10 Not Pre-Specified as Deferred (MAJOR)

**Problem**: H10 (homeostatic rebalancing) is extensively discussed in the plan (Section 3.4, 5.3) but was never executed. The plan does not specify under what conditions H10 would be deferred.

**Verdict**: Pre-specify contingencies for deferred experiments. If H6 fails and the graph does not identify parent-child relationships, H10 cannot be tested — this should have been anticipated.

## What Works

1. **Comprehensive experimental design**: 10 major analyses across multiple layers and models. Coverage is thorough.

2. **MCP pre-specification**: Bonferroni (alpha=0.00417) and BH-FDR (q<0.05) are correctly specified before data collection.

3. **Random baseline inclusion**: H7 (trained vs. random SAE) is a strong methodological contribution that emerged from planning.

4. **Cross-layer validation**: L4 and L8 tested independently. This is correct experimental design.

5. **Cross-model pilot**: Pythia-70M pilot attempted. The attempt is methodologically sound even though results were inconclusive.

## Planning-Specific Recommendations

1. **Pre-specify null-result contingency**: "If zero results survive MCP, the paper reframes as a null-result study with H5 (precision-recall asymmetry) as the main empirical contribution and the LCA-theoretical framework as the theoretical contribution."

2. **Remove H9 from plan** — it is a tautology, not a testable hypothesis.

3. **Pre-specify pivot trigger conditions**: "If [specific condition], then [specific pivot] with [specific execution plan]."

4. **Pre-specify H10 contingency**: "If H6 fails (graph does not identify parent-child pairs), H10 cannot be validated and will be removed from the paper."

5. **Power analysis before data collection**: Compute required n for detectable effect size r=0.3 at alpha=0.05, power=0.80. State upfront: "n=26 is insufficient for medium effects; study is powered for large effects only."

6. **Matched L0 pre-specification**: If OrtSAE ablation is a key comparison, pre-specify matching criteria. "OrtSAE with and without penalty will be compared at matched L0 values (±10%). If L0 cannot be matched, the comparison will be excluded."
# Planning Critique: Encoder-Driven Feature Absorption

## Overall Assessment

The methodology document is detailed and well-organized, with good task dependencies and risk assessment. However, several critical gaps undermine the plan's credibility: (1) H_Downstream is listed as Priority 0 but never executed, (2) the B>D anomaly explanation is incomplete, (3) H_Pareto remains suspended but still appears in the plan.

## Strengths

1. **Clear experimental design**: Each hypothesis has specific design, measurement, and falsification criteria
2. **Risk assessment**: Identifies H_Downstream null result as high-impact risk with mitigation
3. **Task dependencies**: Properly identifies parallel vs sequential execution
4. **Resource estimation**: All tasks fit within 1-hour budget

## Critical Gaps

### H_Downstream Listed as Priority 0 But Never Executed

The plan identifies H_Downstream as the "contrarian's challenge" and Priority 0 (highest). The risk table explicitly states: "If absorbed vs non-absorbed features show < 10% difference in steering success rate, absorption does not have practically significant downstream impact."

But the paper contains no H_Downstream results. This is the most practically important question in the entire proposal - whether the metric the field measures actually predicts utility - and it was never executed.

**This gap fundamentally undermines the paper's practical significance claims.**

### B>D Anomaly: Planned but Not Adequately Analyzed

The plan highlights B>D as requiring investigation but provides no investigation design:
- What statistical test will determine if B>D is real vs noise?
- What sample size is needed to detect a 0.006 difference?
- What's the post-hoc explanation mechanism?

The plan needs a specific investigation protocol for the B>D finding.

### H_Pareto Suspended But Still Listed

The plan lists H_Pareto as Priority 4 with "45 min" estimated runtime. But the risk table acknowledges "Sensitivity formula unfixable" as a risk, and the pilot shows the formula produces out-of-bounds values.

**The plan should either:**
1. Remove H_Pareto from the plan until formula is fixed
2. Add a dedicated "fix sensitivity formula" task with exit criteria

## Design Issues

### H_Safe: Placeholder vs Validated Features

The plan correctly identifies that prior pilots used arbitrary indices as safety features. But the plan doesn't specify:
1. How to access Neuronpedia-validated features programmatically
2. How many validated safety features are expected to be available
3. What to do if insufficient validated features exist

Without these details, H_Safe execution could fail at runtime.

**The plan needs an explicit feature selection protocol with expected yield.**

### H_Mech: No B>D Investigation Design

The B>D finding is highlighted as an "ANOMALY" requiring explanation. The plan states:
> "Key Insight from Contrarian: The B > D finding suggests decoder regularization partially counteracts absorption."

But no specific experiment is designed to test this hypothesis. The plan needs:
1. Specific prediction from decoder regularization hypothesis
2. Experiment to test whether adding regularization to decoder reduces B-D gap
3. Alternative explanations and how to distinguish them

### Missing: H_Comp Results Integration

H_Comp (hierarchy strength dependence) passed in pilot but is absent from the paper. The plan doesn't explain:
1. Why H_Comp was dropped from the paper
2. Whether full H_Comp results exist
3. Where results should appear in the paper

## Task Dependencies Issue

The dependency diagram shows:
```
[setup_env]
       ↓
   ┌───┴───┬───────────┬─────────┐
   ↓       ↓           ↓         ↓
[h_mech] [h_comp]  [h_safe] [h_downstream]
```

But the paper omits h_comp and h_downstream. The plan's execution doesn't match the paper's content.

**This suggests either:**
1. The plan wasn't followed (what happened to these tasks?)
2. The paper is incomplete (what happened to these results?)

## Risk Assessment Gaps

| Risk | Listed Impact | Actual Outcome |
|------|--------------|----------------|
| H_Downstream: No effect found | HIGH (landmark negative) | Task never executed |
| Sensitivity formula unfixable | Medium | Still unfixed |
| H_Safe: No difference | Medium | FAILED (p=0.665) |

The risk assessment correctly identified H_Downstream as high-impact but the task was never executed.

## Recommendations

1. **Either execute H_Downstream or remove from plan** - don't list Priority 0 and then not do it
2. **Add B>D investigation protocol** - specific experiment to test decoder regularization hypothesis
3. **Remove or separately track H_Pareto** - don't include suspended tasks in active plan
4. **Specify H_Safe feature selection protocol** - how to get Neuronpedia features, expected yield
5. **Update plan to match actual paper content** - if h_comp/h_downstream were dropped, document why

## Execution Gap Summary

| Task | Plan | Actual | Gap |
|------|------|--------|-----|
| h_mech | 5 seeds, stochastic | Done | None |
| h_comp | 6 levels, 3 seeds | Not in paper | Missing |
| h_safe | Neuronpedia features | Done, null result | OK |
| h_downstream | Priority 0 | Never executed | Critical |
| h_pareto | Fix formula, then run | Suspended | OK |
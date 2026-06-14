# Critique: Experiments

**Score: 7/10**

## Strengths
- Table 1 correct (no BatchTopK)
- Table 2 uses "Collision Rate" with correct k=200 value (19.2)
- Exact p-values used (0.870, 0.873, 0.868)
- Figure references for all experiments
- E3 title corrected to "Sparsity-Collision"

## Weaknesses
- Table 2 under E2 uses f2_causal data while E3 references f3_sparsity; values differ slightly (e.g., k=100 MSE 27.3 vs 26.6). Need a footnote clarifying Table 2 is from the causal experiment (f2).
- Section 4.8 Summary is redundant with Conclusion

## Improvements
1. Add footnote to Table 2: "Data from causal impact experiment (f2_causal). Sparsity sweep (f3_sparsity) yields similar trends with slightly different values."
2. Remove or shorten 4.8 Summary

## Consistency
- E2-E4 consistently use "collision rate"
- DFDA correctly notes per-pair residual scale

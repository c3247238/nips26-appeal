# Writing Quality Review (Iteration 9)

## Summary

Iteration 9 applied 12 structural fixes to the paper: removed undefined Lyapunov/certified band section, removed build manifest, added evaluation metric specification, defined alignment deviation symbols, added experiment count decomposition, downgraded Proposition 1, fixed SWD h() specification, added conjecture qualifiers, integrated NoBN data, and renumbered figures 1-8.

## Reviewer Scores

| Reviewer | Score | Key Concerns |
|----------|:-----:|--------------|
| Supervisor | 6.0 | Scope collapse (planned experiments not executed), data provenance mismatch, no ImageNet |
| Critic | ~6.0 | 15 findings (3 critical): missing ImageNet, scope collapse from plan, CSI arbitrary weights |

## Dimension Scores (Supervisor)

| Dimension | Score |
|-----------|:-----:|
| Novelty | 7.0 |
| Soundness | 5.0 |
| Experiments | 6.0 |
| Reproducibility | 5.5 |

## Fixes Applied in Iter 9

1. Removed Section 5.7 (Certified Band / Lyapunov) - undefined theory
2. Removed build manifest (lines 432-448) from paper end
3. Added "best test accuracy" evaluation metric specification in Section 4
4. Defined alignment deviation symbols in Section 5.7 (renumbered from 5.8)
5. Added experiment count decomposition (84 + 21 = 105)
6. Specified SWD h() function form
7. Downgraded Proposition 1 to inline remark
8. Added conjecture qualifiers to abstract
9. Integrated NoBN data into Section 6.2
10. Fixed "rapidly growing" filler text
11. Added note to Figure 3 caption about subset vs tables
12. Renumbered Figures consistently (1-8)

## Remaining Issues

1. **[Critical] Data provenance**: Tables use iter_003 data, some figures use iter_006 data
2. **[Critical] No ImageNet**: Limits venue eligibility
3. **[Major] PMP-WD in figures**: Still appears in Figure 3 image but not in method catalog
4. **[Major] Figure 4 BEM scatter**: May still show old BEM values for half_lambda

## Score Analysis

Score dropped from 6.5 to 6.0 because the iter 9 planner created an ambitious experiment plan that was skipped. The critic penalizes "scope collapse" - promising experiments in the plan but not delivering them. The paper text improvements (symbol definitions, evaluation metric, NoBN data) were offset by the plan-paper gap.

SCORE: 6.0

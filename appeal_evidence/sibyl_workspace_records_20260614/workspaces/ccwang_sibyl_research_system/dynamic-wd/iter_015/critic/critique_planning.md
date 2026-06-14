# Planning Critique

## Overall Assessment

The research plan evolved from a focused theoretical investigation (dynamic alignment-aware WD for nonconvex SGD) to a broader unification framework with diagnostic metrics. This pivot was well-motivated by the discovery that four WD sub-traditions share a common control variable (rho_t^l). However, the execution has spread too thin: 14+ iterations have not completed ImageNet experiments, critical ablations, or theorem proofs.

## What Went Right

1. **Adaptive scope refinement**: The original plan targeted novel theoretical contributions (Theorem Targets A-D). When the theory proved harder than expected, the team pivoted to a descriptive/diagnostic contribution (PID taxonomy + evaluation metrics). This pivot was strategically sound.

2. **Evidence-driven direction changes**: The discovery that UDWDC underperforms NoWD led to an honest reframing rather than result cherry-picking. The CWD magnitude confound was identified early. These demonstrate scientific integrity in planning.

3. **Literature coverage**: The literature survey (55+ references) is exceptionally thorough. The identification of 11 research gaps and the taxonomy of 30+ methods across four families provides strong positioning.

## What Went Wrong

### Problem 1: Chronic Under-Execution of Flagged Tasks

The lessons_learned.md shows the same items flagged across multiple iterations:
- "COMPLETE IMAGENET" -- flagged 10+ iterations, still 4/7 methods
- "CWD halved-lambda ablation" -- flagged as P0 for 3+ iterations, never run
- "Fix CSI formula" -- three contradictory formulas persist
- "Resolve theorem status" -- 6+ iterations without proofs or downgrade

The planning system generates correct priorities but fails to enforce execution. There is no mechanism to prevent an item from being eternally deferred.

### Problem 2: Theory Ambition Exceeded Capacity

The original plan targeted four theorems (Target A-D). Zero formal proofs were produced in 14 iterations. Theorem 1 is stated without proof and contradicted by its own Lyapunov function empirics. The planning should have triggered a forced decision point: "Either produce proofs by iteration X or downgrade to conjectures."

### Problem 3: Scope Creep in Metrics

The plan introduced three novel metrics (BEM, CSI, AIS). Each requires independent validation:
- BEM needs NoWD baselines on ImageNet (blocked by incomplete experiments)
- CSI has three contradictory formulas and impossible values
- AIS has fabricated reference values and zero predictive power

Having three half-validated metrics is worse than having one well-validated metric. The plan should have prioritized validating one metric thoroughly.

### Problem 4: Resource Allocation Mismatch

The project has 8x RTX PRO 6000 GPUs (98GB each). A single ImageNet/ResNet-50 run takes approximately 5 hours on 2 GPUs. The 11 missing ImageNet runs would take 28 GPU-hours (3.5 hours wall-clock with 8 GPUs). After 14 iterations, there is no computational reason these runs are incomplete. The bottleneck is not compute but planning/execution discipline.

## Strategic Recommendations

### 1. Enforce a "Do or Drop" Rule

Any action item flagged as P0 for 3+ iterations must be either executed in the next iteration or dropped from the paper's scope. Currently: 4 items qualify (ImageNet completion, CWD ablation, CSI formula, theorem status).

### 2. Reduce Metric Count

Pick one metric to validate thoroughly (recommendation: BEM, as it is the simplest and most interpretable). Demote CSI and AIS to "exploratory analysis" in an appendix. This reduces the validation burden from three metrics to one.

### 3. Lock the Paper Scope

The paper's contribution should be frozen as:
1. PID taxonomy (descriptive, not prescriptive)
2. Diagnostic negative findings (UDWDC instability, CWD magnitude confound)
3. One validated evaluation metric (BEM)
4. Controlled multi-benchmark comparison

Remove: "Unified framework" claims, "Theoretical analysis" contribution, UDWDC as "proposed method."

### 4. Time-Box Remaining Work

- Week 1: Fix data integrity (AIS, CSI formula, CWD Table 1 mapping) + downgrade theorems
- Week 2: Complete ImageNet runs + run CWD halved-lambda ablation
- Week 3: Fix UDWDC-v2 BN bug + run CPR budget-matched ablation
- Week 4: Integrate results, final editing pass

Total: 4 weeks to submission-ready paper. This is achievable with the available resources.

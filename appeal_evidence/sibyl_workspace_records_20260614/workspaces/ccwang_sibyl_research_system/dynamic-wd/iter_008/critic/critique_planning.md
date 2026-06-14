# Planning Critique

## Overall Assessment: 5/10

The task_plan.json shows 5 fixes (all analysis/writing, zero GPU hours), focused on known issues from prior iterations. This is appropriate given the paper's maturity. However, the planning misses the most impactful zero-compute improvements and fails to address the figure regeneration problem that underlies multiple critical findings.

## What the Plan Gets Right

1. **BEM half_lambda fix** (task 1): Correctly identified, low effort, high impact on data integrity.
2. **Remove Lyapunov claims** (task 2): Correct priority. The Lyapunov certificate has been flagged for 5+ iterations. Removing it is the pragmatic choice.
3. **Demote Theorem 2** (task 3): Correct. Non-significant results should not be framed as validated contributions.
4. **Remove appendix refs** (task 4): Necessary cleanup.
5. **Rewrite paper** (task 5): Depends on all fixes above. Logical ordering.

## Critical Planning Gaps

### 1. No Figure Regeneration Task

The most critical issue---PMP-WD ghost method in Figures 3, 8, 9---has no fix task. The task_plan.json does not mention figures at all. This is a planning failure because the figure-text inconsistency is the most reviewer-visible problem. A single task ("Regenerate Figures 3, 8, 9 with correct 7-method set") would address 2 of the 5 critical findings.

### 2. No NoBN Integration Task

Lessons_learned.md explicitly states: "NoBN partial (iter_005) ALL exist and NONE are in the paper body. This is zero-compute, maximum-ROI." The plan does not include integrating NoBN data. This is the highest-ROI missing task: it resolves the AdamW vs BN confound without any new experiments.

### 3. No VGG-16-BN AdamW Integration

Similarly, if AdamW + VGG-16-BN experiments exist (implied by the 105-experiment count), integrating them is zero-compute and completes the factorial design.

### 4. Plan Repeats Prior Iteration Failures

The lessons_learned.md warns: "Stop rewriting paper from scratch every iteration. Quality oscillation (8.2 -> 5.5 -> 7.0 -> 5.5 -> 6.0) is caused by full rewrites." Yet task 5 is "Apply all fixes to paper.md" with a 60-minute estimate. If this becomes another full rewrite, quality will oscillate again. The plan should specify INCREMENTAL EDITS to specific sections, not a blanket "rewrite."

### 5. No ImageNet Experiment Planning

The project memory says "ImageNet (user explicitly requested)." The paper's weakest point is CIFAR-only scale. Even a minimal ImageNet experiment (4 methods x 3 seeds = 12 runs on ResNet-50) would dramatically improve the paper. The plan should at least include a task to design the ImageNet experiment configuration, even if execution is deferred.

## Recommended Additional Tasks

```json
[
  {
    "id": "regenerate_figures",
    "name": "Regenerate Figures 3, 8, 9 with correct method set",
    "type": "visualization",
    "estimated_minutes": 30,
    "priority": "HIGHEST"
  },
  {
    "id": "integrate_nobn_data",
    "name": "Integrate NoBN iter_005 data into paper",
    "type": "analysis",
    "estimated_minutes": 45,
    "priority": "HIGH"
  },
  {
    "id": "specify_accuracy_metric",
    "name": "Add best-vs-final accuracy specification to Section 4",
    "type": "writing",
    "estimated_minutes": 5,
    "priority": "MEDIUM"
  },
  {
    "id": "design_imagenet_experiments",
    "name": "Design minimal ImageNet experiment plan",
    "type": "planning",
    "estimated_minutes": 30,
    "priority": "MEDIUM"
  }
]
```

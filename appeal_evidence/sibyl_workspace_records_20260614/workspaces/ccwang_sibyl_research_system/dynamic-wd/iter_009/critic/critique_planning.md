# Planning Critique

## Overall Assessment

The methodology document is thorough and well-structured—6 phases, clear baselines, pre-registered falsification criteria, proper compute budget estimation. The plan itself is good. The problem is execution: only ~60% of planned experiments were completed, and the missing 40% includes the most scientifically important ones.

## Critical Issues

### 1. Execution Gap

| Phase | Planned | Executed | Status |
|-------|---------|----------|--------|
| Phase 1: VGG-16-BN CIFAR | 42 runs | 21 runs (SGD CIFAR-10 only) | Partial |
| Phase 2: Non-BN Ablation | 9 runs | 0 | Missing |
| Phase 3: PMP/Hill Variants | 18 runs | 0 | Missing |
| Phase 4: ImageNet | 15 runs | 0 | Missing |
| Phase 5: Steady-State Validation | 3 runs | 0 | Missing |
| Phase 6: Analysis & Visualization | N/A | Done | Complete |
| **Total** | **87 runs** | **~42 + 63 prior = 105** | **~48% of new** |

The 105 total experiments cited in the paper include 63 from prior iterations (iter_003 ResNet-20 SGD data) plus ~42 new runs (21 VGG-16-BN + 21 AdamW ResNet-20). The new experiments are heavily concentrated on AdamW ResNet-20 (which are the least novel, being a different optimizer on the same architecture as iter_003).

### 2. Phase Prioritization Error

Phases 2 (non-BN) and 4 (ImageNet) are marked HIGH priority in the proposal but were not executed. Phase 1 (VGG-16-BN) was executed but only partially (SGD/CIFAR-10 only, not the full 7x3x2 = 42 run plan). The execution prioritized breadth (adding AdamW to existing ResNet-20 data) over depth (new architectures/datasets).

### 3. Pre-Registered Falsification Criteria Not Applied

The methodology defines specific falsification thresholds for H1-H7. None of these are evaluated in the paper because the experiments needed to test them (steady-state formula for H1, non-BN for H5, Hill functions for H7, PMP for H4) were not executed. Pre-registering falsification criteria and then not testing them is worse than not pre-registering at all—it signals awareness of what was needed and failure to deliver.

### 4. Compute Budget Underutilized

The plan estimated 111 GPU-hours total. With 8 GPUs, this is ~14 hours wall-clock. The AdamW ResNet-20 experiments (the newly executed ones) are the cheapest runs (~5 GPU-hours). The expensive but important ImageNet runs (~75 GPU-hours) were not attempted despite available compute.

## Strengths

- Falsification criteria are well-defined and scientifically appropriate
- Compute budget estimates are realistic
- Baseline selection (constant, no_wd, half_lambda, random_mask) provides excellent controls
- The planned visualization suite (8 figures) is appropriate for the paper's scope
- Phase structure allows incremental validation and early stopping

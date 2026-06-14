# Planning Critique -- Iteration 017

## Experimental Plan Assessment

The experimental plan covers two datasets (ImageNet, CIFAR-100), two CIFAR architectures (ResNet-20, VGG-16-BN), seven methods, multi-seed evaluation, and ablation studies. This is adequate but has notable gaps.

## Strengths

1. **Multi-seed ImageNet with 7 methods** is a substantial and correctly prioritized experiment. Most dynamic WD papers skip multi-seed ImageNet comparison.

2. **CAWD ablation isolates the modulation signal** -- a well-designed controlled experiment.

3. **Beta and alpha ablations** cover appropriate ranges and provide useful sensitivity information.

## Planning Gaps

### Critical

1. **Budget-matched FixedWD was identified as P0 blocking in iter_016 lessons_learned but was not planned or executed in iter_017.** This is the most impactful experiment per compute-hour and should have been the FIRST experiment run. Estimated cost: ~9 GPU-hours. ROI: resolves the paper's central unfalsifiable claim.

2. **No plan to resolve the WD heatmap lambda discrepancy.** The heatmap colorbar values (0.00052-0.00062) are inconsistent with stated lambda_base=1e-4. This should have been caught during figure generation.

### Major

3. **No 90-epoch ImageNet plan.** The 45-epoch regime creates regime-selection bias favoring EqWD. At least one 90-epoch comparison should have been planned. ~18 GPU-hours.

4. **No ImageNet AIS plan.** The AIS diagnostic is validated on the wrong benchmark. An ImageNet AIS measurement would take minimal additional compute.

5. **No AdamW experiment planned.** In 2026, publishing a weight decay method tested only with SGDW is a significant scope limitation. Even a single AdamW + CIFAR-100 experiment (~30 min) would address this.

6. **Beta=5.0 multi-seed validation not planned.** A 30-minute experiment that resolves a major narrative tension.

7. **CIFAR-10 EqWD/CAWD runs not planned.** Completing the benchmark table is ~0.5 GPU-hours.

## Priority Ordering (If Limited Compute Budget)

If only ~30 GPU-hours are available, the priority ordering should be:

1. **Budget-matched FixedWD on ImageNet** (9 hours) -- resolves core claim
2. **CIFAR-10 EqWD + CAWD** (0.5 hours) -- completes benchmark table
3. **Beta=5.0 multi-seed on CIFAR-100** (0.5 hours) -- resolves narrative tension
4. **90-epoch ImageNet** (18 hours) -- validates standard regime
5. **AIS on ImageNet** (2 hours) -- strengthens Contribution 3

Total: ~30 GPU-hours. Items 1-3 are the minimum viable set.

## Process Observations

- The iter_016 lessons_learned document is well-written and correctly prioritized, but iter_017 appears to have focused on writing refinement rather than executing the identified experiments. This is the wrong priority ordering: experiments should precede writing when blocking confounds exist.
- The 17-iteration history shows a pattern of identifying critical experiments but not executing them before the next writing pass. A stricter "experiments first, writing second" policy would improve iteration efficiency.

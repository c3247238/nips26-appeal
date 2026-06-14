# Planning Critique

## Overall Assessment

The planning (methodology.md) is ambitious and well-structured with 7 phases, explicit resource estimates, and risk mitigation tables. However, the plan overpromised and underdelivered: approximately 40% of the planned experiments were not completed, and the completed experiments revealed issues (UDWDC instability, CWD confound) that should have triggered plan revisions.

## Strengths

1. **Explicit falsification criteria.** The proposal defines clear conditions under which each hypothesis is falsified. This is exemplary scientific methodology.

2. **Phased design with gates.** The pilot-first approach correctly identified UDWDC's instability early (Phase 1), enabling the v2 fix before committing to full runs.

3. **Resource estimates were reasonable.** The 295 GPU-hour estimate for ~76h wall-clock on 8 GPUs was realistic. The actual Phase 1 took ~9.2h (33,253s), consistent with the ~2h per-method estimate scaled by 8 methods x 3 seeds.

## Critical Issues

### 1. Plan Did Not Trigger Pivot Despite Criteria Being Met
The proposal defines explicit pivot criteria:
- H2 falsified if: "UDWDC with rho* underperforms optimally tuned fixed-lambda by > 0.5% on 2+ benchmarks"
- Result: UDWDC underperforms FixedWD by 0.53pp (CIFAR-10), 2.01pp (CIFAR-100), 1.79pp (ImageNet) -- all three benchmarks. **H2 is falsified.**
- H1 partially failed: 2/5 methods exceed 20% fitting threshold.

According to the pivot decision tree, H2 failure should lead to "Keep taxonomy + metrics, drop algorithm claim." This pivot was NOT executed -- the paper still presents UDWDC as Contribution 2.

### 2. Missing Experiments Never Triggered Replanning
Large portions of the plan were not completed:
- Phase 4b (Budget-Matched Controls): Only 2-epoch pilots, no full 90-epoch runs
- Phase 5 (Architecture Generalization): Only pilots for ResNet-101 and ViT-S
- Phase 3 (Alignment Informativeness): Only partial grid sweep
- CWD vs halved-lambda ablation: Not executed
- ImageNet: 3 methods entirely missing (SWD, Defazio, NoWD)

The plan should have been revised to prioritize completing critical experiments over running less important ones. The CWD halved-lambda ablation (~2 GPU-hours) was more important than UDWDC-v2 revalidation (~5 GPU-hours) but was deprioritized.

### 3. UDWDC-v2 Fix Was Insufficient
The v2 fix (EMA smoothing + floor clipping) was an engineering patch that created a new problem (absurd WD budget). The plan identified "If UDWDC-v2 shows improved CSI, re-run key experiments" but did not include a condition for what happens if v2 introduces new anomalies. The floor clipping issue (applying WD to BN layers) was a foreseeable problem that should have been caught in design review.

## Major Issues

### 4. ImageNet Seed Count Inconsistency
The plan specifies "5 seeds (42, 123, 456, 789, 2024) for ImageNet main." Only FixedWD achieved 5 seeds. CPR and UDWDC have 3 seeds. CWD has 1 seed. The plan's minimum seed requirement was not enforced.

### 5. Resource Underutilization
With 8x RTX PRO 6000 GPUs available and ~190 GPU-hours planned, the experiments should have been completable in ~24h wall-clock (assuming 100% GPU utilization). The actual execution spans multiple days with significant gaps. Better experiment scheduling could have completed all planned phases.

### 6. Statistical Protocol Not Fully Applied
The plan specifies: "Paired t-test, Welch's t-test, TOST at delta=0.5%, Cohen's d." None of these appear in the paper's results. The paper reports means and standard deviations but does not apply the planned statistical tests. For a paper emphasizing rigorous evaluation, the absence of formal statistical testing is a planning-execution gap.

## Minor Issues

### 7. Duplicate Ablation Variant
Full_PID and UDWDC_v2 use identical gains but are listed as separate conditions in the ablation plan. This wasted 3 full 200-epoch runs (one per seed) that could have been used for a more informative configuration.

### 8. Phase 7 Never Triggered
Phase 7 (UDWDC-v2 stability fix + re-run) was contingent on UDWDC-v2 showing improved CSI. The paper does not report whether this condition was evaluated, and Phase 7 appears to not have been executed. The plan's conditional phases were not properly tracked.

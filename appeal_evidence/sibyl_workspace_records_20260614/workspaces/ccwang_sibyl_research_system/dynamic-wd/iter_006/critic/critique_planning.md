# Planning Critique

## Overall Assessment: 6/10 (Well-designed plan, poor execution coverage)

The methodology document is well-structured with clear phases, dependencies, falsification criteria, and time estimates. The problem is that only Phase 0-1 were completed. Phases 2-4 are partially done or entirely missing.

## Plan Execution Audit

| Phase | Planned | Executed | Status |
|-------|---------|----------|--------|
| Phase 0: PMP-WD + Instrumentation | PMP-WD implementation + instrumented training | Done | COMPLETE |
| Phase 1: PMP-WD Validation | CIFAR-10/100 x 3 seeds | CIFAR-10 done, CIFAR-100 done | COMPLETE |
| Phase 2: Certificate Visualization | Instrumented reruns + certified band analysis | Instrumented reruns done; NO band visualization | PARTIAL |
| Phase 3A: VGG-16-BN CIFAR-100 | 8 methods x 3 seeds | NOT DONE (CIFAR-10 partial in iter_005) | MISSING |
| Phase 3B: NoBN Ablation | 7+ methods x 3 seeds | 2 methods x 3 seeds + 1 method x 1 seed | INCOMPLETE |
| Phase 3C: ImageNet | 4 methods x 3 seeds | ALL FAILED | FAILED |
| Phase 4: Cumulative Alignment Grid | 96 configs x 3 seeds | NOT DONE | MISSING |
| Phase 5: Analysis & Visualization | 9 figures, 3 tables | 5 figures, 1 table | INCOMPLETE |

**Execution rate: ~35% of planned experiments completed.**

## Falsification Criteria Assessment

The proposal defined 5 falsification criteria. Status:

1. **Certified band violation >20%**: UNTESTABLE -- no subsumption data published, V_t increasing suggests possible violation
2. **Cumulative alignment not better (|rho diff| < 0.05)**: UNTESTABLE -- Phase 4 not executed
3. **PMP-WD diverges or >2% below constant**: PASSED -- PMP-WD converges and performs within noise
4. **Alignment-aware WD >0.5% better on BN (p<0.05)**: PASSED -- no method exceeds 0.5% margin
5. **Both |rho| < 0.3**: UNTESTABLE -- no correlation computed

Only 2 of 5 falsification criteria are testable with current data. This undermines the "pre-registered falsification" framing.

## Planning Flaws

### 1. ImageNet Not Risk-Mitigated
ImageNet experiments are listed as "Phase 3C" with "Estimated: 4-6 hours per run" but no fallback plan for failure. All ImageNet runs failed, and the plan provides no diagnostic protocol. A robust plan would include: (a) a single-epoch smoke test before full runs, (b) memory/OOM diagnostics, (c) reduced-ImageNet (ImageNet-100 or ImageNet-1k subset) as fallback.

### 2. Phase Dependencies Not Enforced
Phase 4 (cumulative alignment grid) depends on Phase 2 (certified band analysis), but Phase 2 was only partially completed. The plan does not make dependencies blocking, allowing the system to "complete" early phases while skipping critical components.

### 3. Time Estimates Were Accurate but Ignored
The plan estimated ~213 GPU-hours total. Approximately 50-60 GPU-hours were consumed. The remaining ~150 GPU-hours were never scheduled.

## Recommendations

1. Gate paper submission on Phase 3B (NoBN) and Phase 4 (alignment grid) completion
2. Add ImageNet-100 as a lower-risk scale validation alternative
3. Mark certified band visualization as a blocking dependency for Phase 2 completion
4. Require Theorem 2 validation before claiming it as a contribution

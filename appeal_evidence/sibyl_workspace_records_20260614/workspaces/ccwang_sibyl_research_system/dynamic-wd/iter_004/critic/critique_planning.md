# Planning Critique: Unified Dynamic Weight Decay Framework

## Phase Execution Assessment

### Phase 1 (Pilot): Executed
- CIFAR-10, ResNet-20, {AdamW, CWD, SWD, cosine-WD}, seed 42 — completed
- Soft-CWD beta sweep (10, 50, 100, 1000) — completed (in pilots/)
- Baseline validation achieved (AdamW constant > 90% ✓)

### Phase 2 (CIFAR Comprehensive): Partially Executed
- CIFAR-10 + CIFAR-100, ResNet-20, all 7 AdamW methods, 3 seeds — completed ✓
- SGD negative control, CIFAR-10 + CIFAR-100, 3 seeds — completed ✓
- **NOT completed**: WD Stability warmup ablation (K in {1,10,50,200,1000})
- **NOT completed**: Soft CWD proximal verification (already done in pilots actually — this is an inconsistency)
- **NOT completed**: AdamWN and AlphaDecay experiments
- **NOT completed**: Composition tests (CWD+Cosine, CWD+AdamWN)

### Phase 3 (ImageNet): Not Executed
- Per project constraints, ImageNet is a hard requirement. This was not executed.
- This is the most significant planning failure — the final paper explicitly lacks the ImageNet experiments that the proposal identified as Phase 3.

### Phase 4 (ViT): Not Executed
- Expected given Phase 3 was skipped.

---

## Critical Planning Flaw: Stopping Before Phase 3

The proposal defined Phase 3 (ImageNet) as a required validation phase, not optional. The project memory explicitly lists ImageNet as a hard constraint ("datasets: CIFAR-10, CIFAR-100, **ImageNet** (user explicitly required)"). The current paper skips ImageNet entirely.

The final_review.md's NeurIPS review correctly identifies this as the top deal-breaker (#1 of 3). The review was written by the same system that knows ImageNet was required. This creates a contradiction: the review correctly identifies the gap, but the gap was never addressed.

**Recommended action**: Before the paper is considered submission-ready, Phase 3 (at minimum: ResNet-50 on ImageNet, constant vs. CWD vs. no_wd vs. cosine_schedule, 3 seeds, 90 epochs) must be executed. With 8x RTX PRO 6000 Blackwell GPUs, 4 methods x 3 seeds = 12 runs at ~4 GPU-hours each = ~48 GPU-hours total. With 8 GPUs in parallel, wall clock ~6 hours.

---

## Methodology Gaps vs. Plan

| Planned Experiment | Status | Impact |
|--------------------|--------|--------|
| WD warmup ablation (K ∈ {1,10,50,200,1000}) | NOT DONE | Minor: only tests H2 |
| Soft CWD verification | NOT DONE (but pilots has beta sweep) | Minor: H1 partially covered |
| AdamWN | NOT DONE | Critical: target-norm axis uncovered |
| AlphaDecay-style per-layer WD | NOT DONE | Critical: spatial axis uncovered |
| Composition tests (CWD+Cosine, CWD+AdamWN) | NOT DONE | Major: H7 untested |
| ImageNet/ResNet-50 | NOT DONE | Critical: project-level hard requirement |
| ViT-S architecture | NOT DONE | Major: H6 architecture comparison |
| More seeds (5+) for key comparisons | NOT DONE | Major: statistical power for null claim |

---

## Resource Utilization Assessment

The total experiment compute used is approximately:
- 7 methods × 3 seeds × 2 datasets × 200 epochs × 2 optimizers = 84 runs
- Plus pilots: ~8 runs
- Estimated: ~100 GPU-hours total out of ~117 planned

This is a reasonable utilization rate. The unused budget (~17 GPU-hours) should have been allocated to AdamWN (6 runs), WD warmup ablation (5 runs), and beginning ImageNet experiments (12 runs → requires additional budget).

---

## Recommendation for Iteration 4

If the paper is to be submitted, the following minimum changes are needed:

1. **Run ImageNet experiments** (Phase 3): 4 methods × 3 seeds × 90 epochs × 2 datasets → ~50 GPU-hours, ~6 wall-clock hours on 8 GPUs.

2. **Add AdamWN** to CIFAR experiments: 3 seeds × 2 datasets = 6 runs, ~3 GPU-hours.

3. **Add 2 extra seeds** for constant, no_wd, CWD comparisons on both datasets: 2 extra × 3 methods × 2 datasets × 2 optimizers = 24 runs, ~12 GPU-hours.

4. **Run WD warmup ablation** on CIFAR-10: 5 configurations × 1 seed = 5 runs, ~2 GPU-hours.

Total additional investment: ~67 GPU-hours, ~8 wall-clock hours on 8 GPUs. Feasible in a single session.

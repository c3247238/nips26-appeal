# Planning Critique: Unified Dynamic Weight Decay Framework

## Plan vs. Execution Gap

The methodology.md describes an ambitious 4-phase experimental plan. Comparing against what was actually executed:

| Phase | Planned | Executed | Gap |
|-------|---------|----------|-----|
| Phase 1: Pilot (CIFAR-10, 4 methods, seed 42) | 3 GPU-hours, ~1h | Executed | Complete |
| Phase 2: CIFAR Comprehensive (11 configs, 3 seeds, CWD battery, WD Stability, composition) | 18 GPU-hours, ~3h | 7 methods × 3 seeds × 2 datasets = 42 experiments | **Partial: no CWD battery (C1/C3/C4), no WD Stability, no composition tests** |
| Phase 3: ImageNet Validation (ResNet-50, top 5 methods, 3 seeds) | 72 GPU-hours, ~9h | NOT executed | **Completely missing** |
| Phase 4: Architecture Generalization (ViT-S, CIFAR-100) | 24 GPU-hours, ~3h | NOT executed | **Completely missing** |

**Actual execution: ~25% of the planned work.** Phases 3 and 4 were never run, and Phase 2 was incomplete (CWD falsification battery, WD Stability Condition, composition tests were all dropped).

---

## Critical Planning Failures

### 1. ImageNet Experiments Never Ran

The research proposal explicitly listed ImageNet as a project constraint ("**ImageNet**（用户明确要求）"). The methodology describes Phase 3 as 72 GPU-hours on 8x RTX PRO 6000. The compute was available (8x RTX PRO 6000 Blackwell, 98GB each). Yet no ImageNet experiments were executed.

The reviewer (final_review.md) identified CIFAR-only experiments as the top deal-breaker for NeurIPS. The paper acknowledges this as its first limitation. The problem is not a resource constraint---it is a planning failure: ImageNet experiments should have been the highest priority once the CIFAR null result was established.

**Why this matters:** The paper's central claim (Phi Invariance Conjecture) is about AdamW generally. Without ImageNet data, reviewers cannot distinguish "AdamW invariance is real" from "CIFAR is too easy a benchmark to detect WD effects." The paper even provides the order-of-magnitude argument for why the null result is expected at CIFAR scale---which makes the failure to test at a scale where WD matters more conspicuous.

### 2. CWD Falsification Battery Abandoned

The proposal described a rigorous battery for testing whether CWD's alignment-awareness genuinely contributes:
- C1: Effective-lambda matched constant WD
- C2: Random binary mask (executed)
- C3: Inverted (anti-alignment) mask
- C4: Continuous cosine-similarity-weighted WD

Only C2 was run. The missing tests (C1, C3, C4) were the most scientifically important: C1 would directly test if CWD's benefit is just reduced WD, and C3 would test if the alignment *direction* matters. Without C1 and C3, the paper cannot conclude that "CWD's alignment-awareness is irrelevant"---only that "CWD and random_mask produce similar accuracy under AdamW," which is a weaker claim.

### 3. WD Stability Condition Never Tested

The proposal described H2 (WD Stability Condition) as a key theoretical contribution: warmup ablations with K ∈ {1, 10, 50, 200, 1000} steps, testing whether violating the stability condition produces loss spikes. The methodology even computed K_critical = 200 for WD=5e-3. These experiments were never run. The WD Stability Theorem appears in the proposal but not the paper.

This is a planning failure: theoretical claims should be accompanied by their empirical tests. Proposing a theorem and then not testing it leaves it as speculation.

### 4. Architecture Diversity Dropped

VGG-16-BN and ViT-S experiments were never run. The methodology described these as necessary for:
- VGG-16-BN: testing whether the null result is specific to ResNet skip connections
- ViT-S: testing BN vs LN architecture differences (H6)

Without this, the paper's claim that "BatchNorm is the key mechanism" is not validated. The paper identifies LayerNorm architectures as a boundary condition where the conjecture may fail, but provides no data to locate this boundary.

---

## Resource Allocation Assessment

The project had 8x RTX PRO 6000 Blackwell GPUs available. The methodology estimated ~117 GPU-hours total. Actual usage appears to be:
- 42 AdamW experiments (7 × 3 × 2): ~2-3 GPU-hours (ResNet-20 trains in ~40min/run)
- ~21 SGD experiments: ~3-4 GPU-hours
- Total: roughly 6-8 GPU-hours used vs. 117 planned

This is severe underutilization given the stated project constraints (user explicitly requested ImageNet). The planning correctly identified ImageNet as essential and allocated 72 GPU-hours for it, but the execution stopped at CIFAR.

---

## Recommendations for Future Iterations

1. **Prioritize ImageNet immediately.** Run 3 seeds of constant, no_wd, cosine_schedule, and CWD on ResNet-50/ImageNet. If the invariance holds at ImageNet, the paper becomes strongly publishable. If it fails, the boundary is identified.

2. **Fix the SGD data integrity issue first.** Before running more experiments, recompute all SGD statistics from raw data and update Table 5. The paper cannot be submitted with inflated p-values.

3. **Execute C1 and C3 of the CWD battery.** These two experiments take ~6 GPU-hours total and would directly test whether CWD's alignment-awareness contributes anything beyond budget reduction.

4. **Run the WD Stability Condition test.** The warmup ablation with K ∈ {1, 10, 50, 200, 1000} takes ~5 GPU-hours and would provide the first empirical test of H2.

5. **Scope the claims to match the evidence.** If ImageNet experiments cannot be run before the submission deadline, narrow the Phi Invariance Conjecture explicitly to CIFAR-scale, and present the paper as a "benchmark and framework infrastructure" contribution rather than a general claim about AdamW.

# Experiment Critique — Iteration 7 (Updated)

**Critic**: sibyl-critic (Opus 4.6)
**Date**: 2026-03-19

---

## Overall Assessment: Significant Progress, Critical Gaps Remain

Since the last critique, VGG-16-BN is now fully complete (7 methods x 3 seeds, 200 epochs each), NoBN has expanded to 2 of 3 methods complete, and matched-rho SGD has 2 methods partially running. However, the two most scientifically important experiments — rho_high and PMP-WD implementation — remain entirely unexecuted.

---

## Critical Issue 1: rho_high Still at 5-Epoch Pilot

The paper's central empirical prediction ("method sensitivity scales with rho, with a transition at rho* ~ 1-5") requires data at elevated rho. Current state:

| rho regime | Status | Phi spread |
|-----------|--------|-----------|
| rho ~ 0.005 (SGD) | Complete (but confounded with optimizer) | 0.91% |
| rho ~ 0.05 (AdamW low) | Partial: constant 3 seeds, CWD 1 seed complete | >= 0.04% (incomplete) |
| rho ~ 0.5 (AdamW standard) | Complete | 0.25% |
| rho ~ 5.0 (AdamW high) | **5-epoch pilot only** | Unknown |

Without rho_high data, the paper has exactly ONE clean AdamW data point (rho=0.5, spread=0.25%) on the spread-vs-rho curve. The SGD point (rho=0.005, spread=0.91%) is confounded with optimizer type. The rho_low point is incomplete.

Figure 1's three-zone diagram is effectively a hypothesis illustration, not an empirical finding. The paper should either:
1. Run rho_high (strongly preferred, ~3 GPU-hours)
2. Relabel Figure 1 as "hypothesized regime structure"

---

## Critical Issue 2: PMP-WD Not Implemented

PMP-WD = clip(kappa * (rho* - rho_hat_t)+, 0, lambda_max) is approximately 30 lines of code. It has not been implemented. This is the paper's primary algorithmic contribution. A reviewer will reasonably ask: "You derived an optimal WD law and didn't test it?"

Even a null result (PMP-WD = constant at rho=0.5) validates the theory: the optimal adaptive law correctly identifies when adaptation is unnecessary. At rho=5.0 (if rho_high is run), PMP-WD should outperform constant according to Theorem 3's prediction. Testing this closes the loop between theory and experiment.

**Effort**: ~30 LOC implementation + 6 runs (2 rho values x 3 seeds) = ~3 GPU-hours.

---

## Critical Issue 3: Data-Paper Inconsistencies (NEW)

Verified against actual epoch_metrics.jsonl files:

### VGG-16-BN (Table 3)
Every entry in Table 3 is incorrect. All means are inflated by 0.04-0.10%. See critique_writing.md W-CRIT-1 for full comparison table. The phi spread is 0.18% (actual) vs 0.16% (paper).

### NoBN (Table 5)
Table 5 reports seed maxima as means. Constant: paper says 87.74+/-0.20, actual mean = 87.49+/-0.23. CWD: paper says 87.64+/-0.17, actual mean = 87.46+/-0.19. The constant-CWD gap shrinks from 0.10% to 0.03%.

### "105 completed runs" claim
The abstract claims 105 completed 200-epoch runs. Audit needed:
- iter_003 ResNet-20: 84 runs (4 optimizer-dataset configs x 7 methods x 3 seeds) -- status of CIFAR-100 SGD no_wd unclear (possibly n=1)
- iter_005 VGG-16-BN: 21 runs (complete)
- iter_005 NoBN: 7 complete + 2 incomplete
- iter_005 matched-rho: 4 complete + 1 partial
- iter_005 rho_low: 4 complete + 3 partial

The actual count of fully completed 200-epoch runs may differ from 105. A comprehensive audit is required.

---

## Major Issue: Incomplete Experiments Misrepresented

### NoBN
Paper states "2 of 7 methods completed" but no_wd has only 1 of 3 seeds complete (seed_42 at 200 epochs, seed_123 at 95 epochs, seed_456 missing). So the actual state is: constant=3 seeds complete, CWD=3 seeds complete, no_wd=1 seed complete, 1 at 95/200, 1 missing. The paper cannot compute Phi spread for NoBN without at least 3 methods.

### Matched-rho SGD
Constant: 2 complete seeds (123, 456), seed_42 only 5 epochs.
CWD: 2 complete seeds (42, 123), seed_456 at 119/200 epochs.
no_wd: entirely missing.
Without no_wd, the matched-rho Phi spread cannot be computed.

### rho_low
Constant: 3 seeds complete (90.01, 90.08, 89.94 = mean 90.01+/-0.07).
CWD: seed_42 complete (90.09), seed_123 at 346 epochs (incomplete), seed_456 at 200 epochs (89.83).
half_lambda: seed_42 at 106 epochs (incomplete).
The rho_low spread cannot be computed reliably with 1 complete method.

---

## Major Issue: Statistical Power

N=3 remains. MDE at 80% power = ~0.77%. TOST equivalence at delta=+/-0.3% has power ~15-20%. This was flagged in the previous critique and is unaddressed. The VGG-16-BN spread (0.18%) is well within noise given random_mask std of 0.36%.

---

## Positive Developments Since Last Critique

1. **VGG-16-BN fully complete**: 7 methods x 3 seeds x 200 epochs. Phi spread = 0.18% confirms cross-architecture null result. This is a genuine improvement -- the paper now has 2 architectures.

2. **NoBN expanded**: constant and CWD both have 3 complete seeds. The NoBN constant accuracy (87.49% mean) and CWD accuracy (87.46% mean) show a 0.03% gap (smaller than BN's 0.07%), directionally consistent with Theorem 1 (higher AIS but gap does not widen).

3. **Matched-rho SGD expanding**: CWD data now available (90.43%, 90.63% for seeds 123, 42). The matched-rho spread can be partially estimated once no_wd is added.

4. **rho_low constant complete**: 3 seeds showing 90.01+/-0.07. Nearly identical to standard rho (90.13+/-0.31), suggesting rho_low does not change accuracy for constant WD.

---

## Experiment Priority Queue (Updated)

| Priority | Experiment | Est. Cost | Scientific Value |
|----------|-----------|-----------|-----------------|
| **1** | rho_high (lambda=5e-3, 4 methods x 3 seeds) | ~3 GPU-h | Tests Theorem 1 non-trivial prediction; fills Figure 1 |
| **2** | PMP-WD implementation + pilot (2 rho x 3 seeds) | ~3 GPU-h | Tests Theorem 3; converts theory to contribution |
| **3** | NoBN no_wd completion (2 more seeds) | ~40 min | Enables NoBN Phi spread computation |
| **4** | Matched-rho SGD no_wd (3 seeds) | ~1.5 GPU-h | Enables confound resolution |
| **5** | Matched-rho SGD cwd_hard seed_456 completion | ~15 min | 81 more epochs |
| **6** | rho_low CWD seed_123/456 completion + half_lambda | ~2 GPU-h | Fills low-rho regime |
| **7** | n=5 seed extension (key methods) | ~6 GPU-h | Improves TOST power |
| **8** | ImageNet ResNet-50 | ~12 GPU-h | Scale validation |

Items 1-6 total ~10 GPU-hours and would resolve all critical and most major findings.

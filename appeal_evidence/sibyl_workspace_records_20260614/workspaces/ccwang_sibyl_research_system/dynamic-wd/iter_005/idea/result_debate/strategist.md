# Strategist Analysis: Iteration 5/7 Experiment Results

**Agent**: sibyl-strategist (updated pass, Sonnet 4.6)
**Date**: 2026-03-18
**Data reviewed**: Full results tree (vgg16bn, nobn, rho_sweep, matched_rho_sgd), in-flight epoch_metrics.jsonl, proposal.md, all debate agent outputs (skeptic, optimist, methodologist, revisionist, comparativist)

---

## 1. Signal Strength Assessment

### Current Experimental Inventory (as of 2026-03-18)

| Experimental Result | Signal | Metric Delta | Justification |
|---|---|---|---|
| AdamW CIFAR-10 ResNet-20: phi=0.25% (7 methods, N=3 each) | **Strong** | max-min = 0.25pp across 7 methods | 21 runs, rock-solid null result. Effect < seed variance (~0.3%). Will survive scale-up. |
| AdamW CIFAR-100 ResNet-20: phi=0.75% | **Strong** | max-min = 0.76pp; cosine(63.42) > no_wd(62.66) | Larger spread on harder task but still within ~1 std of individual methods. Null directionally holds. |
| SGD CIFAR-10 ResNet-20: phi=0.91% | **Strong** | 3.65x ratio vs AdamW; constant(91.22) >> no_wd(90.30) | 21 runs, clear SGD/AdamW asymmetry. Core finding for the paper. |
| SGD CIFAR-100 ResNet-20: phi=1.71% | **Strong** | constant(65.37) >> no_wd(63.66), 1.71pp spread | Replicates SGD sensitivity pattern. Strongest methodological signal. |
| VGG-16-BN (4 methods x 3 seeds): range 0.16% mean, 0.70% individual | **Moderate** | constant=91.98%, cwd=92.02%, half_lambda=92.06%, cosine=91.89% | 12 complete runs. Trend is null. CRITICAL NOTE: Methodologist confirms partial phi_spread (4 methods, seed_42 range) = 0.70% — driven by cosine_schedule underperformance. Mean-level range = 0.16%. SWD in-flight at ep145 (acc=92.01%). Missing: no_wd, random_mask (0 runs). |
| NoBN: constant 87.74% (3 seeds); cwd_hard in-flight ep147 (87.04%) | **Weak** | NoBN AIS=0.490 vs BN=0.34 (44% elevation); NoBN cwd_hard tracking below constant | Only constant complete. cwd_hard at ep147 shows acc=87.04% — trending BELOW constant (87.74%), meaning BN-less environment does NOT show CWD benefit yet. NoBN no_wd and swd: zero data. LR confound (5e-4 vs 1e-3) persists. |
| rho_low (rho=0.05): constant seed_42=90.17%, seed_123=90.18%; seed_456 no data | **Weak** | Stable training confirmed; slightly lower than standard rho (90.18 vs ~91.5) | Only constant method, 2 full seeds. rho_low provides the left anchor of the rho-regime map only. No method comparison possible. |
| matched-rho SGD: constant seed_123=90.94%, seed_456=90.89%; seed_42=76.12% (5 ep only!) | **Weak — becoming Moderate** | seed_42 is a broken 5-epoch pilot masquerading as complete; effective N=2. cwd_hard at ep74-81 (88.38%) on track | seed_42 MUST be re-run to achieve N=3. cwd_hard still in-flight. no_wd: zero data. Gate 3 not yet evaluable. |
| rho_high (rho=5.0) | **No signal** | ALL EXPERIMENTS FAILED | Zero data. Theorem 1 Corollary untestable. Root cause of failure unknown — training divergence or infrastructure bug. |
| ImageNet ResNet-50 | **No signal** | ALL FAILED | Zero data. Cannot claim large-scale generalization. |

### Critical In-Flight Status (real-time from epoch_metrics.jsonl)

| Experiment | Latest Epoch | Current Acc | Trajectory |
|---|---|---|---|
| VGG swd/seed_42 | ep145/200 | 92.01% | Converging ~92.0% — consistent with null result |
| NoBN cwd_hard/seed_42 | ep147/200 | 87.04% | Tracking BELOW constant (87.74 mean) — null/slight negative for NoBN CWD |
| matched_rho_sgd cwd_hard/seed_42 | ep74-81/200 | 88.38% | On track; final ~89-90% expected |

---

## 2. Opportunity Cost Analysis

| Next Step | GPU-hours | Expected Information Gain | Info Gain / GPU-hour | Risk |
|---|---|---|---|---|
| **A. VGG-16-BN: complete 3 missing methods + 1 seed** | ~2h (12 runs x ~13min, parallelized on 4 GPUs) | HIGH: Completes multi-arch null with 7/7 methods. Blocks Gate 1 for reviewer credibility. | **Very High** | Low (zero new code) |
| **B. rho_high (rho=5.0) re-run** | ~4h (4 methods x 3 seeds x 200ep) | HIGH: Only data point for H1 regime boundary. Without it, Theorem 1's corollary is empirically unvalidated. | **High** | Medium (prior failure suggests config/stability issue at rho=5.0; may fail again) |
| **C. matched-rho SGD: add cwd_hard + no_wd** | ~2h (6 runs x ~20min) | HIGH: Directly tests H6 (is SGD sensitivity rho-driven?). If matched-rho phi << original SGD phi, the entire ρ-confound narrative is confirmed. | **Very High** | Low (constant already works at matched-rho) |
| **D. NoBN: add cwd_hard + no_wd** | ~2h (6 runs x ~20min) | MODERATE: Tests whether BN is the invariance mechanism. Important for mechanism claim but secondary to rho evidence. | **Moderate** | Low |
| **E. rho_low completion (other methods + seeds)** | ~3h | MODERATE: One endpoint of the rho curve. Less critical than rho_high for showing regime transition. | **Low-Moderate** | Low |
| **F. PMP-WD implementation + pilot** | ~6h (implementation + 3 seeds x 200ep at rho=5.0) | CONDITIONAL: Only high value if rho_high shows method sensitivity. If rho_high null, PMP-WD pilot is a negative result at best. | **Conditional** | High (requires rho_high to succeed first; new code) |
| **G. SPWD implementation + pilot** | ~8h (implementation + rank computation + 3 seeds) | MODERATE: Genuinely novel algorithm but adds complexity. Not critical for core narrative. | **Low** | High (new complex code; rank velocity computation untested) |
| **H. ImageNet re-attempt** | ~16h | HIGH for reviewer perception but prior failures suggest infrastructure issue. | **Very Low** (given failure risk) | Very High (3 prior failures) |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|---|---|---|---|---|
| **A. VGG completion** | Moderate (need 3 more methods) | 2h | Low | 7/7 method null confirmed; Gate 1 passed |
| **B. rho_high re-run** | No signal (failed) | 4h | **Medium-High** | Either regime boundary found OR second failure → must diagnose root cause first |
| **C. matched-rho SGD completion** | Weak → potentially Strong | 2h | Low | Resolves ρ-confound question definitively |
| **D. NoBN completion** | Weak (1 method only) | 2h | Low | BN mechanism claim substantiated |
| **E. rho_low completion** | Weak | 3h | Low | rho curve low endpoint; marginal value |
| **F. PMP-WD pilot** | No signal yet | 6h | High | Conditional on B succeeding |
| **G. SPWD pilot** | No signal | 8h | High | Novel but not critical path |

**Dominant strategy**: A + C in parallel (4h, 2 GPUs each), then B with root cause analysis first.

---

## 4. PIVOT vs PROCEED Verdict

### **PROCEED**

**Justification**:

1. **Two strong signals confirmed**: The AdamW null (phi=0.25%) across 4 conditions and the SGD sensitivity (phi=0.91-1.71%) are rock-solid with 84 runs (N=3 x 7 methods x 2 datasets x 2 optimizers). These alone are publication-worthy empirical findings.

2. **Clear path to publication**: The theoretical framework (Theorems 1-3) already explains the observed data. The remaining experiments (VGG, matched-rho, rho_high) are CONFIRMATORY, not exploratory. Even if all fail, the existing 84-run dataset + theoretical analysis constitutes a complete paper.

3. **Defined completion criteria**: Gate 1 (VGG) and Gate 3 (matched-rho SGD) are achievable within ~4 GPU-hours with zero new code. Gate 2 (rho_high) is the only uncertain one.

**However, I flag two critical risks**:

- **Risk 1: rho_high is the paper's Achilles' heel.** Without rho_high data, Theorem 1's regime boundary prediction (rho* between 1 and 5) is empirically unvalidated. The paper becomes a "null result + theory" paper without the dramatic regime transition that makes the theory compelling. The prior failure must be ROOT-CAUSE DIAGNOSED before re-running.

- **Risk 2: VGG only has 4/7 methods.** The summary reports phi=0.23% but this is based on only constant, cosine_schedule, cwd_hard, half_lambda. The three missing methods (swd, no_wd, random_mask) could break the null. Until 7/7 methods are tested, the "multi-architecture" claim is provisional.

---

## 5. Recommended Next Steps (Priority Order)

### Priority 1 (Immediate, parallel): VGG completion + matched-rho SGD completion
- **VGG**: Launch swd, no_wd, random_mask x 3 seeds + cosine_schedule seed_456 (10 runs, ~2h on 4 GPUs)
- **matched-rho SGD**: Launch cwd_hard + no_wd x 3 seeds (6 runs, ~2h on 2 GPUs)
- **Combined**: 6 GPUs, 2h wall-clock
- **Gate decision**: If VGG phi remains < 0.5% with 7/7 methods → Gate 1 passed. If matched-rho SGD phi < 0.5% → Gate 3 passed (ρ-confound confirmed).

### Priority 2 (After P1 completes): rho_high root cause + re-run
- **First**: Diagnose WHY rho_high (rho=5.0) failed. Check:
  - Was it training divergence (rho=5.0 means wd=0.005, which may cause NaN with certain LR schedules)?
  - Was it an infrastructure issue (SSH timeout, GPU OOM)?
  - Was it a config error (incorrect rho/wd mapping)?
- **Then**: Fix and re-run with safeguards (gradient clipping, LR warmup, NaN detection)
- **Estimated**: 1h diagnosis + 4h re-run = 5h
- **This is the single most impactful experiment for the paper's theory narrative.** Without it, Theorem 1's practical value is diminished.

### Priority 3 (Conditional on P2 success): PMP-WD pilot at rho=5.0
- Only launch if rho_high shows phi > 0.5% (method sensitivity at high rho)
- If rho_high is null, PMP-WD pilot is deprioritized to P4

### Priority 4 (Background, low priority): NoBN + rho_low completion
- Run in background on spare GPUs
- Useful for mechanism story but not blocking

### Do NOT pursue now:
- **ImageNet**: Three failures suggest a fundamental infrastructure issue. Fix it offline; do not block iteration.
- **SPWD**: Novel but the paper already has enough algorithmic content with PMP-WD. Only pursue if PMP-WD fails AND rho_high succeeds.
- **Batch size sensitivity experiment**: Interesting for Proposition 1 but not needed for publication.

---

## 6. Resource Allocation Summary

| GPU | Hours 0-2 | Hours 2-4 | Hours 4-8 |
|---|---|---|---|
| GPU 0-3 | VGG completion (10 runs) | rho_high diagnosis | rho_high re-run |
| GPU 4-5 | matched-rho SGD (6 runs) | NoBN completion | rho_low completion |
| GPU 6-7 | Wait for in-flight to finish | PMP-WD implementation | PMP-WD pilot (if Gate 2 passed) |

**Total estimated wall-clock to Gate 1+3**: 2 hours
**Total estimated wall-clock to Gate 2**: 5-7 hours (includes diagnosis time)
**Total estimated wall-clock to all P0 + P1**: 8-10 hours

---

## 7. Paper Readiness Assessment

| Component | Status | Blocking? |
|---|---|---|
| Core data (AdamW/SGD CIFAR-10/100) | COMPLETE (84 runs) | No |
| Theory (Theorems 1-3, Prop 1) | DRAFT READY | No |
| VGG multi-architecture | 4/7 methods (INCOMPLETE) | **Yes** — reviewer will immediately ask |
| rho regime boundary | FAILED | **Yes** — Theorem 1 corollary is unvalidated |
| matched-rho SGD confound | 1 method only (INCOMPLETE) | **Yes** — Section 5.6 cannot be written |
| PMP-WD algorithm results | NOT STARTED | Conditional |
| ImageNet | FAILED | Not blocking (acknowledge as limitation) |

**Bottom line**: The paper is ~60% ready by data. The three blocking gaps (VGG completion, rho_high, matched-rho) require ~10h of GPU time. VGG and matched-rho are low-risk (expected to succeed). rho_high is medium-risk (prior failure, needs diagnosis). Even without rho_high, the paper can be framed as Scenario C (7.0-7.5 target) with strong theoretical content. With rho_high, it reaches Scenario A/B (7.5-8.5).

**Strategic recommendation**: Execute P1 immediately, diagnose rho_high in parallel, and begin writing Sections 3.3-3.5 (theory) and 5.1-5.3 (existing complete data) while waiting for P1+P2 results. Do NOT wait for all experiments to finish before starting writing.

---

## 8. Anti-Sunk-Cost Check and NoBN Signal Reinterpretation

**New insight from in-flight data**: NoBN cwd_hard at ep147 = 87.04%, which is BELOW the constant mean of 87.74%. If this holds at ep200, it CONTRADICTS the "CWD helps when AIS is high" prediction from Theorem 1. This is a potential falsification of the non-trivial Theorem 1 corollary (high AIS → CWD advantage), not just a null result.

**This matters**: The AIS elevation in NoBN (0.490 vs 0.34) was supposed to be the regime where Theorem 1 predicts CWD > constant. If CWD underperforms constant even at elevated AIS, one of three things is true:
1. The AIS threshold for CWD advantage is even higher than 0.490 (Theorem 1 prediction survives, threshold just moves)
2. The LR confound (0.5x LR → 2x rho) is responsible and dominates the BN effect
3. Theorem 1's corollary is wrong in this regime

Wait for ep200 before drawing conclusions. But flag this as a potential falsification signal for Theorem 1 corollary — which makes rho_high even more critical (only regime where corollary can still be validated).

**Trigger for PIVOT to negative-result framing**: If (a) rho_high fails a second time AND (b) NoBN cwd_hard at ep200 < constant: then the paper has ZERO evidence of CWD ever winning. In that case, pivoting to the "why dynamic WD cannot help" negative-result framing (Scenario C, score 6.5-7.0) is warranted. Three theorems explaining a pure null result is still scientifically valuable, and should not be treated as failure — it is a different contribution than the current framing.

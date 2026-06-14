# Methodology Audit — Iteration 5/7 Experiments

**Auditor**: Methodologist Agent
**Date**: 2026-03-18 (Updated for Iter 7 proposal)
**Scope**: Internal validity, external validity, metric appropriateness, reproducibility, and ablation completeness for the Iter 5/7 experimental campaign supporting "When Does Adaptive Weight Decay Help? A Stability-Optimal Control Theory of Dynamic Weight Decay"

**Status Note**: The Iter 7 proposal refines the Iter 6 framing with dual derivation of Theorem 3 (PMP + RG beta function) and Proposition 1 (alignment noise constraint). The experimental evidence has not substantially changed since Iter 5 — the same data gaps persist. This audit reflects the current evidence state as of Iter 7.

---

## 1. Baseline Fairness Audit

### 1.1 Hyperparameter Budget Asymmetry

**CRITICAL FLAG — NoBN uses different learning rate than BN counterpart.**
- ResNet-20 (BN): lr=1e-3, wd=5e-4
- ResNet-20-NoBN: lr=5e-4, wd=5e-4
- The halved learning rate for NoBN was justified as "stability requirement," but this introduces a confound: the effective rho (grad-norm/weight-norm ratio) changes with lr. Any difference in phi_spread between BN and NoBN could be attributed to lr change rather than BN removal. The NoBN ablation does NOT cleanly isolate BN's role.
- **Severity**: High. The NoBN comparison (H5-2) is confounded. Claims about "ell-infinity path" vs "BN scale-invariance" cannot be made with confidence from this data alone.

**FLAG — matched-rho SGD seed_42 ran only 5 epochs.**
- Seed 42: 5 epochs (best_test_acc=76.12%). Seeds 123, 456: 200 epochs (90.94%, 90.89%).
- The seed_42 summary.json shows `"epochs": 5` while seeds 123/456 show `"epochs": 200`. This means seed_42 was a pilot run that was never re-run at full scale, or the 200-epoch run overwrote this file incorrectly.
- **Impact**: If seed_42 truly only ran 5 epochs, then "3 seeds x 3 methods" for matched-rho SGD is actually "2 full seeds + 1 pilot seed" for the constant method. The task_plan.json marks this task as "completed" which is misleading.
- **Severity**: High for the matched-rho SGD constant method specifically. The mean/std computation across seeds is unreliable if one seed ran 5 vs 200 epochs.

**FLAG — matched-rho SGD has only "constant" method results.**
- The plan specifies methods=[constant, cwd_hard, no_wd] x 3 seeds = 9 runs.
- Only 3 completed summary.json files exist, all for `constant`. No cwd_hard or no_wd results.
- The task is marked "completed" in task_plan.json despite missing 6/9 runs.
- **Impact**: The phi_spread for matched-rho SGD CANNOT be computed without multiple WD methods. A single method's accuracy across seeds measures seed variance, not method sensitivity.
- **Severity**: CRITICAL. The H5-4 hypothesis ("SGD-AdamW effect ratio collapses at matched rho") is untestable with current data.

### 1.2 Method Coverage Gaps

| Experiment | Planned Methods | Methods with Results | Gap |
|---|---|---|---|
| NoBN (full) | constant, cwd_hard, no_wd | constant only (3 seeds) | **cwd_hard, no_wd missing** — cannot compute phi_spread |
| rho_low (rho=0.05) | constant, cwd_hard, half_lambda, no_wd | constant only (2/3 seeds) | **3 methods missing, 1 seed missing** |
| rho_high (rho=5.0) | constant, cwd_hard, half_lambda, no_wd | **FAILED** (task status: failed) | **All missing** |
| matched_rho_sgd CIFAR-10 | constant, cwd_hard, no_wd | constant only (2 full + 1 pilot) | **cwd_hard, no_wd missing** |
| matched_rho_sgd CIFAR-100 | constant, cwd_hard, no_wd | **No results found** (task "completed" but no data located) | **All missing** |
| VGG-16-BN | 7 methods x 3 seeds | constant(3), cwd_hard(3), half_lambda(3), cosine_schedule(3) = 12/21 | **swd, random_mask, no_wd missing** (0/9 runs) |
| ImageNet | All 3 phases | **FAILED** (all 3 tasks failed) | **All missing** |

**Summary**: Of the 5 core hypotheses (H5-1 through H5-5), NONE can be fully tested with current data:
- H5-1 (ImageNet phi-invariance): No data.
- H5-2 (NoBN): Only one method (constant), no spread computable; also confounded by lr change.
- H5-3 (rho monotonicity): rho_high failed; rho_low has only one method.
- H5-4 (matched-rho SGD ratio collapse): Only one method; no spread computable.
- H5-5 (VGG invariance): 4/7 methods complete; partial spread can be computed but not the full 7-method comparison.

---

## 2. Metric-Claim Alignment

### 2.1 Phi Spread as Primary Metric

The paper defines `phi_spread = max(best_test_acc) - min(best_test_acc)` across WD methods. This is appropriate for the core "method invariance" claim. However:

**FLAG — phi_spread conflates test-set noise with method difference.**
- With only 3 seeds and best_test_acc differences of ~0.05-0.3%, the standard error of mean accuracy is potentially of the same order as the spread itself. No statistical test (paired t-test, TOST equivalence test at delta=0.5%) has been reported alongside raw spread values.
- The methodology document mentions TOST at delta=1%, but no TOST results appear in the data files.
- **Recommendation**: Report phi_spread with bootstrap confidence intervals AND TOST p-values. Without these, a spread of 0.25% cannot be distinguished from random seed noise.

### 2.2 CSI, AIS, BEM Metrics

- **CSI (Coupling Stability Index)**: Recorded in all summary.json files. Values range from 0.43 (low-epoch SGD) to 2.06 (VGG cosine_schedule). The metric is meaningful for comparing within-architecture perturbation stability.
- **AIS (Alignment Informativeness Score)**: Recorded. NoBN AIS (0.37-0.59) consistently higher than VGG-BN AIS (0.18-0.31). This is an interesting signal worth reporting but has no statistical significance test attached.
- **BEM (Budget Equivalence Metric)**: Most values are 0.0 (constant and methods with no budget deviation) or -0.5 (half_lambda, cosine_schedule, cwd_hard). The metric appears to be a simple `(mean_wd_actual / wd_nominal) - 1` or similar. Its diagnostic value is limited — it essentially reports whether the method reduces effective WD.
- **Missing metric**: Per-layer rho tracking is mentioned as a key diagnostic but no per-layer rho data is visible in the summary.json files (only aggregate metrics). If it exists in epoch_metrics.jsonl, its absence from summaries makes cross-experiment comparison difficult.

### 2.3 Metric-Hypothesis Mapping Gaps

| Hypothesis | Required Metric | Available? |
|---|---|---|
| H5-1: ImageNet phi-invariance | phi_spread across 4 methods, N=3 seeds | NO (ImageNet failed) |
| H5-2: BN not required | phi_spread NoBN vs BN; Cohen's d | NO (only 1 NoBN method) |
| H5-3: rho monotonicity | phi_spread at 3 rho values | NO (rho_high failed, rho_low incomplete) |
| H5-4: SGD-AdamW ratio collapse | phi_spread ratio: original_SGD vs matched_SGD | NO (only 1 matched method) |
| H5-5: VGG invariance | phi_spread across 7 methods, N=3 | PARTIAL (4/7 methods) |

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: No evidence of data leakage. CIFAR-10/100 train/test splits are standard. Each run uses standard `torchvision` splits.
- [x] **Contamination**: Not applicable (image classification, not NLU/NLG).
- [ ] **Selection bias — POTENTIAL CONCERN**: The methodology specifies "best_test_acc" as the primary metric. This is the maximum test accuracy across all 200 epochs, which introduces optimistic bias (selecting the best epoch rather than final or averaged). This is standard practice for CIFAR but should be acknowledged. The final_test_acc values are ~0.1-0.3% lower than best_test_acc, confirming the selection effect is small.
- [ ] **Overfitting to evaluation — MODERATE CONCERN**: All experiments use CIFAR-10 with the same augmentation pipeline (presumably random crop + horizontal flip). No experiments on CIFAR-100 for the new Iter 5 comparisons (NoBN, rho sweep, VGG). The CIFAR-10 focus risks overfitting conclusions to one dataset's characteristics.
- [x] **Hyperparameter tuning on test set**: No evidence. All configurations are pre-specified in task_plan.json.

### 3.1 Additional Threats

**CONCERN — Incomplete experiment marking as "completed".**
The task_plan.json marks several tasks as "completed" that have significant gaps:
- `full_nobn_cifar10`: marked completed, but only constant method has results (plan required constant + cwd_hard + no_wd).
- `full_matched_rho_sgd_cifar10`: marked completed, only constant method present.
- `full_rho_low_cifar10`: marked completed, only constant method, 2/3 seeds.
- `full_vgg16bn_cifar10`: marked completed, but 3/7 methods (swd, random_mask, no_wd) have no data.

This false "completed" status creates a serious risk: downstream analysis and writing stages may proceed believing the experimental foundation is solid when it is not.

**CONCERN — rho_high task FAILED with no fallback executed.**
The methodology specifies: "If rho=5.0 diverges in pilot, fallback to rho=2.0 (wd=2e-3)." The pilot for rho_high shows it completed (summary.json exists at pilot path), but the full task is marked "failed." No rho=2.0 fallback was attempted. This means the trichotomy hypothesis (H5-3) has zero data at the high-rho end.

---

## 4. Ablation Gap Analysis

### 4.1 Required Ablations vs. Completed

| Proposed Component | Ablation Needed | Status |
|---|---|---|
| BN scale-invariance as mechanism | NoBN ablation (remove BN, keep everything else) | **INCOMPLETE**: lr changed (confound), only 1 method |
| rho as governing variable | rho sweep (0.05, 0.5, 5.0) | **INCOMPLETE**: rho=5.0 failed, rho=0.05 has 1 method |
| AdamW vs SGD optimizer effect | matched-rho comparison | **INCOMPLETE**: only 1 method, no spread computable |
| Architecture generalization | VGG-16-BN vs ResNet-20 | **PARTIAL**: 4/7 methods for VGG |
| Scale generalization (ImageNet) | ResNet-50 / ImageNet-1K | **FAILED**: no data |

### 4.2 Missing Ablations Not Planned

- **Batch size ablation**: The Contrarian (Iter 7 proposal) identifies alignment noise at batch=128 as a critical concern. No batch size ablation (e.g., 256, 512, 1024) is planned for CIFAR.
- **Learning rate sensitivity**: All CIFAR experiments use lr=1e-3 (AdamW) or lr=0.01 (SGD). No sensitivity analysis across learning rates. If the null result (constant WD wins) is specific to this lr, the conclusion may not generalize.
- **WD magnitude sensitivity**: Only three wd values tested (5e-5, 5e-4, 5e-3). A finer sweep (e.g., 10 points on log scale) would better characterize the rho-spread relationship claimed in the trichotomy.

---

## 5. Reproducibility Score: 3.5 / 5

| Criterion | Score | Notes |
|---|---|---|
| Random seeds fixed | 5/5 | All runs specify seeds 42, 123, 456. Deterministic. |
| Hyperparameters specified | 4/5 | All HPs in summary.json. Missing: gradient clipping norm for NoBN (specified in plan as max_norm=1.0 but not recorded in outputs). |
| Code/data availability | 3/5 | Code in `iter_004/exp/code/`. Dataset is standard CIFAR. However, no requirements.txt or Docker specification. PyTorch version, CUDA version, etc. not recorded in outputs. |
| Hardware documented | 3/5 | GPU type (RTX PRO 6000 Blackwell) documented in methodology. Specific GPU IDs per run recorded (good). No CPU/RAM/storage specifications. |
| Reproducibility within 10% | 3/5 | For completed VGG constant runs: seed_42=92.03%, seed_123=92.00%, seed_456 not yet checked. The 0.03% cross-seed difference is excellent. However, for NoBN: 87.97, 87.66, 87.58 — 0.39% spread, still within 1% easily reproducible. Main concern: incomplete experiments mean overall reproducibility cannot be assessed for the full set of claims. |

**Key reproducibility gap**: The `cwd_beta=100.0` parameter appears in all configs but is never explained in the methodology. What is it? If it controls CWD's alignment sensitivity, its value may be critical for reproducing CWD results. Not mentioned in the 7-method table.

---

## 6. Top-3 Highest-Impact Methodology Improvements

### Recommendation 1: Complete the missing WD methods before any analysis or writing (Effort: Medium, Credibility Impact: CRITICAL)

**What**: Run cwd_hard and no_wd for NoBN, rho_low, and matched_rho_sgd experiments. Run swd, random_mask, no_wd for VGG. These are the minimum runs needed to compute phi_spread for any hypothesis.

**Why**: Currently, ZERO of the five Iter 5 hypotheses can be tested. The experiments have achieved "constant method across seeds" consistency, which demonstrates seed stability — a necessary but not sufficient result. Without multiple methods, the paper's central claim (WD method invariance) has no new supporting evidence from Iter 5 beyond the existing Iter 3 data.

**What changes**: With even 3 methods per experiment, all five hypotheses become testable. The delta between "one method" and "three methods" in terms of scientific value is enormous — far larger than going from 3 to 7 methods.

### Recommendation 2: Acknowledge and correct the NoBN lr confound (Effort: Low, Credibility Impact: High)

**What**: Either (a) re-run NoBN with lr=1e-3 and accept potential instability (documenting any instability as itself informative), or (b) acknowledge the lr confound explicitly in the paper and frame NoBN results as "suggestive" rather than "confirmatory." Additionally, record the actual rho values for NoBN to verify the confound magnitude.

**Why**: A reviewer will immediately notice that NoBN uses lr=5e-4 vs BN's lr=1e-3. Halving lr approximately halves the effective gradient norm, directly changing rho by ~2x. Since the paper's core thesis is that rho governs sensitivity, this confound undermines the NoBN ablation's conclusions.

**What changes**: Transparent acknowledgment of this limitation (Option b) costs 15 minutes of writing. A fair comparison (Option a) costs ~5 GPU-hours but dramatically strengthens the BN mechanism claim.

### Recommendation 3: Add statistical significance testing to all phi_spread comparisons (Effort: Low, Credibility Impact: High)

**What**: For every phi_spread value reported, include: (a) bootstrap 95% CI from the 3-seed data, (b) TOST equivalence test with delta=0.5% (or the planned delta=1%), (c) effect size (Cohen's d) for each method vs. constant baseline.

**Why**: The paper's central contribution is a null result: WD method choice does not matter (under certain conditions). Null results require stronger statistical evidence than positive results. A spread of 0.25% from 3 seeds is indistinguishable from noise without a formal equivalence test. Reviewers at NeurIPS/ICML will demand this.

**What changes**: ~2 hours of analysis code. Transforms "spread is small" into "methods are statistically equivalent at p < 0.05," which is a publishable claim. Without this, the paper asserts informally what it should prove statistically.

---

## Appendix: Data Inventory Summary

### VGG-16-BN CIFAR-10 (best_test_acc from seed_42)
| Method | best_test_acc | Status |
|---|---|---|
| constant | 92.03 | 3 seeds complete |
| cwd_hard | 92.32 | 3 seeds complete |
| half_lambda | 92.00 | 3 seeds complete |
| cosine_schedule | 91.62 | 3 seeds complete |
| swd | — | NO DATA (epoch_metrics.jsonl exists for seed_42 but no summary.json) |
| random_mask | — | NO DATA |
| no_wd | — | NO DATA |

**Partial phi_spread (4 methods)**: 92.32 - 91.62 = 0.70%
**Note**: This 4-method partial spread is interesting — it exceeds the 0.5% invariance threshold, driven primarily by cosine_schedule underperformance. CWD_hard (92.32) slightly outperforms constant (92.03), which contradicts the Iter 3 ResNet-20 finding where constant was best-or-tied. This VGG result deserves careful examination.

### NoBN ResNet-20 CIFAR-10 (constant only)
| Seed | best_test_acc |
|---|---|
| 42 | 87.97 |
| 123 | 87.66 |
| 456 | 87.58 |
| Mean +/- std | 87.74 +/- 0.20 |

### Matched-rho SGD CIFAR-10 (constant only)
| Seed | best_test_acc | Epochs |
|---|---|---|
| 42 | 76.12 | **5 (PILOT ONLY)** |
| 123 | 90.94 | 200 |
| 456 | 90.89 | 200 |

**Warning**: Seed 42 data is not usable for full comparison. Effective N=2 for this condition.

### rho_low (rho=0.05) CIFAR-10 (constant only)
| Seed | best_test_acc |
|---|---|
| 42 | 90.17 |
| 123 | 90.18 |
| 456 | In progress (ep83/200 at acc=88.42) |

---

## 7. Iter 7 Proposal-Specific Methodology Issues

### 7.1 VGG Phi Spread Miscomputed in iter5_summary

The iter5_summary.json reports VGG-16-BN phi_spread=0.2317% from 4 methods (constant, cosine_schedule, cwd_hard, half_lambda). However, computing phi spread as max(method_means) - min(method_means) from the per-seed data yields:

- constant mean: 92.05%
- cosine_schedule mean: 91.99% (only 2/3 seeds complete at summary time; now 3/3 complete)
- cwd_hard mean: 92.06%
- half_lambda mean: 92.15%

Max=92.15 (half_lambda), Min=91.99 (cosine_schedule), partial spread = **0.16%**. But note: the raw seed-level range within cosine_schedule (91.62, 92.21, 92.14) spans 0.59%, which already exceeds the < 0.5% invariance threshold at the seed level. CWD_hard seed-level range (92.32, 92.04, 91.81) spans 0.51%. **The invariance threshold is applied to method means, but the within-method seed variance can exceed the threshold — a statistical inconsistency in interpretation.**

Furthermore, the summary reports phi_spread=0.2317% which does not match the manual calculation of 0.16%. The discrepancy likely stems from cosine_schedule having only 2 seeds in the analysis (seed_456 was in-flight). This inconsistency should be documented.

### 7.2 The 18.3x → 3.65x Ratio Discrepancy

The methodology document states the SGD/AdamW effect ratio is 18.3×. The Iter 7 proposal document also states 18.3×. But the iter5_summary.json reports original_ratio=3.65. The 18.3× figure appears to come from an effect size (Cohen's d) ratio rather than the raw phi_spread ratio. The phi_spread ratio is SGD_spread/AdamW_spread = 0.9133/0.2500 = **3.65×**, not 18.3×. The proposal repeatedly cites 18.3× as the "effect size ratio" without clearly distinguishing Cohen's d from phi spread. **The paper must specify which metric produces the 18.3× figure and ensure consistency throughout.**

### 7.3 Baseline Accuracy Asymmetry (AdamW vs SGD)

SGD baseline achieves 91.22% (CIFAR-10) vs AdamW baseline at 90.13% — a 1.09% gap. A method with 1.09% higher baseline accuracy is simply performing better, which changes the regularization landscape. If SGD finds sharper minima (higher baseline), WD method sensitivity could reflect convergence geometry differences rather than WD method properties alone. **This optimizer-performance confound is never discussed in the methodology or the proposal, despite being directly relevant to the SGD vs AdamW comparison.**

### 7.4 Matched-rho SGD cwd_hard In-Flight Status

The matched-rho SGD cwd_hard seed_42 run is at epoch 79 (test_acc=87.27%) with CSI=0.228. For comparison, matched-rho SGD constant at convergence achieves ~90.9% with CSI=0.826. The large CSI difference between methods (0.228 vs 0.826) at this intermediate epoch is striking and may indicate that cwd_hard under SGD dynamics produces a fundamentally different optimization trajectory — not just a different accuracy level. If cwd_hard converges to substantially different accuracy than constant under matched-rho SGD, this would provide evidence that the SGD/AdamW ratio is NOT fully explained by rho, falsifying H5-4. The methodology should pre-specify the convergence criterion for "substantially different" in matched-rho SGD results.

### 7.5 Dual-Derivation of Theorem 3: Assumption Sharing Risk

The Iter 7 proposal claims Theorem 3 (PMP-WD) is strengthened by an "independent" derivation via RG beta function theory. However, the proposal itself notes both derivations assume "near-steady-state dynamics and linearized rho evolution." If both derivations share the same linearization assumption, they are not methodologically independent — they are the same approximation derived through different formalisms. A reviewer will probe whether the two routes share hidden assumptions. The methodology should explicitly state the assumptions of each derivation and verify they are distinct.

### 7.6 PMP-WD Algorithm: Missing Sensitivity Analysis for κ

The PMP-WD formula λ*(t) = clip(κ·(ρ* − ρ̂_t)⁺, 0, λ_max) has feedback gain κ described as "treated as a hyperparameter in experiments, default κ=1." No sensitivity analysis for κ is planned or executed. If performance depends critically on κ, then PMP-WD is not a hyperparameter-free improvement over constant WD — it introduces a new tuning parameter. The paper must either: (a) provide theory for the optimal κ value, or (b) run a κ sensitivity sweep (e.g., {0.1, 0.5, 1.0, 2.0, 5.0}) to demonstrate robustness.

---

## 8. Revised Summary Assessment (Iter 7)

**What the current experimental evidence supports** (well-established, N=21 runs each):
1. AdamW, ResNet-20, CIFAR-10: phi_spread=0.25% across 7 methods × 3 seeds. Solid.
2. AdamW, ResNet-20, CIFAR-100: phi_spread=0.75% across 7 methods × 3 seeds. Solid.
3. SGD (original rho), ResNet-20, CIFAR-10: phi_spread=0.91%. Solid.
4. SGD (original rho), ResNet-20, CIFAR-100: phi_spread=1.71%. Solid.
5. VGG-16-BN partial: phi_spread=0.16% from 4/7 methods (suggestive, not conclusive).

**What the current evidence does NOT support**:
- Any hypothesis requiring rho_high data (rho=5.0): FAILED.
- Any hypothesis requiring matched-rho SGD phi_spread: only 1 method, no spread computable.
- Any hypothesis requiring NoBN phi_spread: only 1 method, no spread computable.
- Any ImageNet claim: FAILED.
- The VGG "null result" with all 7 methods: 3 methods (no_wd, random_mask, swd partially) missing.

**The Iter 7 proposal's theoretical framework (Theorems 1-3, Proposition 1) is internally coherent and well-articulated. The methodological problem is not the theory — it is that the experiments designed to test the theory's critical predictions have not been completed. The proposal's target score of 7.5-8.0 is contingent on Gates 1-3 passing. Currently, no gate can be formally evaluated.**

**Critical path actions in strict priority order**:
1. Complete VGG swd (ep150→200) + launch VGG no_wd × 3 seeds + VGG random_mask × 3 seeds → enables Gate 1
2. Complete matched-rho SGD cwd_hard (ep79→200) + launch matched-rho no_wd × 3 seeds → enables Gate 3
3. Complete NoBN cwd_hard (ep148→200) + launch NoBN no_wd × 3 seeds → enables BN mechanism test
4. Launch rho_high (rho=5.0 or fallback rho=2.0) × 4 methods × 3 seeds → enables Gate 2
5. Add TOST equivalence tests to all reported phi_spread values

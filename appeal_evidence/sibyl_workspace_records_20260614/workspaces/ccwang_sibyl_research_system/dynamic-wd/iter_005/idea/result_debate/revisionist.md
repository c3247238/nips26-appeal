# Revisionist Analysis — Iteration 7 Result Debate (Updated)

**Agent**: sibyl-revisionist
**Date**: 2026-03-18
**Input**: iter_003 full results (AdamW + SGD, 7 methods × 3 seeds × 2 datasets); iter_005 analysis (VGG-16-BN: 4 methods × 3 seeds complete + SWD ep182/200; rho_low: constant 2 seeds complete; matched_rho_sgd: constant 2 seeds complete; NoBN pilot); Iter 7 proposal and hypotheses H1–H10

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| **H1: ρ-Regime Hypothesis** (ρ* ∈ [1,5] transition) | **Inconclusive — critical gap** | rho_low (ρ=0.05): constant only, 2 seeds complete (90.17%, 90.18%). rho_high (ρ=5.0): ALL RUNS FAILED repeatedly across multiple iterations. No multi-method phi spread at any non-standard ρ computable. | 15% — The hypothesis' core empirical test has zero data at the boundary |
| **H2: Binary Masking Stability Cost** (Theorem 1) | **Weakly confirmed for ordinal direction; null on statistical significance** | 7/7 directional predictions hold on iter_003: constant >= CWD on AdamW (all p>0.05). SGD constant (91.22%) > CWD (90.87%, d=1.13) > SWD (90.71%, d=3.48, p<0.001). BUT on VGG-16-BN: ordering REVERSES — half_lambda (92.147%) > SWD (92.16% best) > CWD (92.057%) > constant (92.050%). Constant ranks 4th of 5 methods on VGG. | 50% — The directional prediction fails on VGG; the null result is confirmed but not directed |
| **H3: PMP-WD Effectiveness** | **Untested** | Not implemented. No code, no pilot, no data. | 0% |
| **H4: BN Mechanism** | **Directionally supported, causally confounded** | NoBN constant mean ~87.49% vs BN constant 90.13% (diff ~2.6%, d=9.14). BUT NoBN uses lr=5e-4 vs BN lr=1e-3, changing ρ from ~0.5 to ~1.0. Cannot isolate BN effect from ρ/LR change. | 30% — Strong signal, weak causal claim |
| **H5: Multi-Architecture (VGG-16-BN)** | **Confirmed for tested methods** | VGG phi spread = 0.157% across 4 complete methods × 3 seeds. SWD at ep182/200: best_so_far=92.16%, mean_wd=5e-5 (10× less than constant). Missing: swd full completion (seeds 123/456), no_wd (all), random_mask (all). | 70% — Strong for 4/7 complete methods; 3 still missing |
| **H6: Matched-ρ SGD** | **Inconclusive — partial data only** | Matched-ρ SGD constant: seed_123=90.94%, seed_456=90.89% (seed_42: 5 epochs only, discarded). Mean N=2: 90.91% vs original SGD constant 91.22% — matched-ρ is 0.31% LOWER, not higher. CWD: ep77/200, acc=87.54% (still training). Cannot compute phi spread. | 20% — 1 complete method; ρ-confound unresolved; accuracy direction surprising |
| **H7: Layer-wise ρ Dynamics (ImageNet)** | **Untested** | All ImageNet experiments failed. Zero data. | 0% |
| **H8: AIS as Regime Indicator** | **Inconclusive** | rho_high failed; rho_low has only constant method. Pilot observation: rho_low constant AIS≈0.474 (5ep) vs standard-ρ VGG AIS~0.217 — suggestive that lower ρ has higher AIS, but not confirmable without multi-method data. | 15% — Suggestive trend, not testable |
| **H9: Alignment Noise Aggregation** | **Untested** | No alignment variance diagnostic run. Evidence base: SimiGrad (2021) citation only. | 5% — Entirely theoretical |
| **H10: SPWD** | **Untested** | Not implemented. | 0% |

---

## 2. Surprise Analysis

### Surprise 1: VGG-16-BN method ordering REVERSES from ResNet-20 — half_lambda and SWD beat constant

**Deviation expectation**: The proposal states "constant WD is best or tied-best under all tested conditions" (confirmed 7/7 on iter_003). On VGG-16-BN with 4 complete methods, constant ranks 4th: half_lambda (92.147%) > CWD (92.057%) > constant (92.050%) > cosine_schedule (91.990%). SWD at ep182/200 shows best_so_far=92.16% — also above constant.

**Quantification**: half_lambda exceeds constant by +0.097% (vs constant std=0.062%, CWD std=0.255%). Directionally wrong, marginally significant. The entire ordering is noise-level differences.

**Wrong assumption traced**: We over-interpreted Theorem 1's prediction as a strict inequality (constant > dynamic). The theorem actually says the stability cost of dynamic WD raises the generalization gap bound — it predicts dynamic WD cannot improve beyond constant, not that it strictly underperforms. The VGG data is consistent with a true ordering of exact equality, with observed values reflecting random variation. The assumption "constant ranks first" was never warranted by the theorem.

**Implication for framing**: Remove all directional claims about constant WD being "best." The correct paper claim is: "All budget-equivalent methods produce statistically indistinguishable accuracy in this regime." This is more defensible and more accurate.

### Surprise 2: SWD achieves best VGG accuracy (92.16% at ep149) with 10× WD reduction

**Deviation**: SWD's mean_wd at ep182 is 5e-05 — 10× less than constant's wd=0.0005. Yet its best test accuracy (92.16%) exceeds every complete method, including constant (92.05%). Final CSI for SWD is ~1.10 vs constant's 1.58.

**Quantification**: +0.11% over constant. Not yet finalized (ep182/200), but the best epoch has already passed (ep149), suggesting the final result will be close to 92.16%.

**Wrong assumption traced**: We assumed that "budget-equivalent" methods (same wd budget averaged over training) produce comparable regularization. SWD applies a 10× lower effective WD for extended periods. This is NOT budget-equivalent to constant wd=0.0005; it is a different regularization level. The "budget equivalence" framing conflates the total WD integral with the adaptive schedule path. If VGG-16-BN is over-regularized at wd=0.0005, then SWD's reduction is actually moving toward the optimum, not exploring a budget-equivalent path.

**Implication**: The experimental design has a confound. Methods that reduce WD below the constant level are not budget-equivalent — they are exploring a different regularization strength. This confound is present in all SWD and half_lambda results throughout the paper.

### Surprise 3: Matched-ρ SGD accuracy is 0.31% LOWER than original low-ρ SGD

**Deviation**: We predicted matched-ρ SGD (ρ~0.5 via lr=0.01, wd=5e-3) would produce accuracy comparable to AdamW (90.13%) or similar to original SGD (91.22%). Actual: 90.91% for constant, N=2 complete seeds.

**Quantification**: -0.31% vs original SGD constant (91.22%). The direction is wrong: increasing ρ by 100× decreased accuracy.

**Wrong assumption traced**: We assumed that ρ=0.005 (original SGD) and ρ=0.5 (matched) are in the "same regime" and that the accuracy difference reflects method sensitivity. But wd=5e-3 is a very strong regularizer for a 270K-parameter ResNet-20 on CIFAR-10. Original SGD uses wd=5e-5, which may be closer to optimal. The matched-ρ experiment changed both ρ and the absolute WD strength simultaneously — it did not isolate the ρ effect.

**Implication**: The matched-ρ SGD design is methodologically flawed. To properly resolve the ρ confound, we need to match ρ via learning rate (lr increase with fixed wd) rather than via wd increase. This is a significant redesign.

### Surprise 4: rho_low (ρ=0.05) constant accuracy is marginally HIGHER than standard ρ=0.5

**Deviation**: We expected rho_low (wd=5e-5, 10× less than standard) to produce slightly lower or similar accuracy. Actual: 90.17%, 90.18% vs standard 90.13% — marginally higher (+0.045%).

**Quantification**: +0.045% over standard ρ constant. Trivially small but directionally consistent with the "over-regularization at wd=0.0005" hypothesis.

**Wrong assumption traced**: We assumed wd=0.0005 is well-tuned for ResNet-20 AdamW CIFAR-10. The data suggests the network performs equally well or slightly better at 10× lower WD. If the optimal wd is around 5e-5, then all our phi spread measurements at wd=0.0005 are in a slightly supra-optimal regime. This does not invalidate the null result, but it means the "standard ρ" is not an unambiguous canonical operating point.

---

## 3. Mental Model Revision

**Before these results**, we believed three things: (a) constant WD directionally dominates at standard ρ across architectures; (b) ρ is the key explanatory variable for the SGD/AdamW sensitivity difference; (c) wd=0.0005 is a well-calibrated operating point for our experiments.

**After these results**, the data forces three revisions:

First, we assumed constant WD produces a strict ordering advantage over dynamic methods. VGG-16-BN refutes this directionality. The correct mental model is that **the accuracy distribution of all budget-equivalent methods overlaps completely** — no method is reliably better or worse, and the apparent ordering in any single condition is noise. Theorem 1 should be framed as proving upper bounds on improvement, not proving a ranking.

Second, we assumed ρ is the primary causal variable distinguishing SGD from AdamW sensitivity. The matched-ρ SGD result undermines this: increasing ρ via wd increase decreases accuracy and probably does not reproduce the AdamW phi spread pattern. The causal analysis requires matching ρ via learning rate (not wd), which has not been tested. We may have misidentified the confound.

Third, we assumed wd=0.0005 is a neutral, well-calibrated operating point. The rho_low accuracy (+0.045%) and SWD VGG accuracy (+0.11% at 10× lower wd) both suggest the network operates near or slightly above the optimal WD level. This means our phi spread measurements may reflect behavior in a mild over-regularization regime rather than a canonical one.

---

## 4. Reframing Test

**Original research question**: "When does adaptive weight decay help? A stability-optimal control theory of dynamic weight decay."

**Would we frame it the same today?** No. This framing implies we will identify conditions where dynamic WD helps and propose a validated algorithm. After 7 iterations: (a) we have found no condition where dynamic WD provably helps; (b) PMP-WD has not been implemented or tested; (c) the ρ-regime transition has no data.

**Revised research question**: "Why are weight decay methods interchangeable in normalized networks? A unifying stability theory with implications for adaptive WD design."

Justification for the revision:
1. It centers the confirmed finding (method equivalence across architectures and conditions) rather than the speculative one (ρ-regime transition)
2. Theorems 1–2 become the core positive contributions: they explain why equivalence holds rather than explaining when it breaks
3. The SGD sensitivity anomaly (phi spread 3.7× larger) becomes the key puzzle that the theory illuminates, motivating the rho-regime prediction as a direction for future work
4. PMP-WD becomes a theoretical contribution ("this is what the optimal control law would look like in a non-trivial regime") rather than a validated algorithm

**What this costs**: The revised framing is a negative-result + theory paper. Expected reviewer score ceiling drops from 8.5 to 7.0–7.5. The algorithm novelty is reduced.

**What this preserves**: The gap between theory and evidence closes substantially. The paper makes only claims it can defend with data. The theoretical contribution (Theorems 1–2 + Proposition 1) stands on its own.

**Recommendation**: Maintain the original framing only if rho_high experiments can be debugged and run, AND matched-ρ SGD phi spread is completed, AND PMP-WD pilot is run. If any two of these three fail to materialize in the next iteration, pivot to the revised framing.

---

## 5. New Hypotheses from Surprising Results

### NH1: Absolute WD Level Dominates ρ Ratio as the Confounding Variable (from Surprise 3)

**Statement**: The accuracy decrease in matched-ρ SGD (91.22% → 90.91%) is caused by the absolute WD level change (wd: 5e-5 → 5e-3), not by the ρ ratio change. Prediction: SGD with the same wd=5e-5 but higher lr (to achieve ρ=0.5) will recover accuracy to ≥ 91.0% and show phi spread ≤ 0.3%.

**Falsification experiment**: SGD CIFAR-10 ResNet-20 at lr=0.1, wd=5e-5 (ρ≈0.5 via lr increase). 3 seeds × 200 epochs × {constant, cwd_hard, no_wd}. If accuracy ≥ 91.0% and phi spread < 0.5%, NH1 is confirmed and the matched-ρ design was flawed. Estimated time: ~3 GPU-hours.

**Why this matters**: If confirmed, the entire ρ-confound narrative needs revision. The paper's claim that "ρ explains the SGD/AdamW difference" would need to be replaced by "absolute WD level explains the difference."

### NH2: VGG-16-BN Optimal WD is Below the Experimental wd=0.0005 (from Surprise 2)

**Statement**: SWD's superior performance on VGG at mean_wd≈5e-5 indicates the optimal constant WD for VGG-16-BN on CIFAR-10 is closer to 1e-4 to 5e-5 rather than 5e-4. Prediction: VGG-16-BN with constant wd=1e-4 will achieve ≥ 92.15% — above the current wd=0.0005 baseline of 92.05%.

**Falsification experiment**: VGG-16-BN CIFAR-10 with constant wd ∈ {5e-5, 1e-4, 2e-4, 5e-4}. One seed each. If the best wd is 5e-4 (our current baseline), NH2 is falsified. Estimated time: ~2 GPU-hours.

**Why this matters**: If NH2 is confirmed, the phi spread measurements on VGG are at a suboptimal operating point. The "budget equivalence" framing requires wd to be at or near optimal for a fair comparison.

### NH3: Method Ordering is a Zero-Mean Random Variable Across Architectures (from Surprise 1)

**Statement**: The accuracy ordering of WD methods (constant vs CWD vs SWD etc.) has no consistent signal across architectures. ResNet-20 AdamW: constant > cosine_schedule > random_mask. VGG-16-BN: half_lambda > CWD > constant. Prediction: a third architecture (e.g., WideResNet-28-10 CIFAR-10) will produce yet another ordering, with no method consistently ranking in the top 2 across all three.

**Falsification experiment**: WideResNet-28-10 CIFAR-10, AdamW, {constant, cwd_hard, half_lambda, no_wd} × 3 seeds. If constant ranks first or second on WideResNet-28-10 as well as ResNet-20, NH3 is partially falsified. Estimated time: ~4 GPU-hours.

**Why this matters**: NH3 would directly determine whether Theorem 1's directional prediction (constant ≥ dynamic) is a reproducible regularility or an artifact of the ResNet-20 architecture.

---

## 6. Summary Assessment

### What the current evidence establishes with high confidence

| Finding | Evidence | Defensibility |
|---|---|---|
| AdamW phi spread < 0.5% on CIFAR-10, two architectures | ResNet-20: 0.25% (7 methods × 3 seeds); VGG-16-BN: 0.157% (4 methods × 3 seeds) | High |
| SGD phi spread 3.7× larger than AdamW at CIFAR-10 | 0.91% vs 0.25%, p<0.001 | High (confound unresolved) |
| NoBN removes ~2.6% accuracy from constant WD baseline | BN: 90.13% vs NoBN: 87.49%, d=9.14 | High (causation confounded) |
| No WD method reliably ranks first across conditions | VGG ordering ≠ ResNet-20 ordering | Medium — only 2 architectures |

### Unconfirmed claims carried forward from the proposal

- ρ-regime transition: No data at rho_high. **High risk of irreproducibility.**
- PMP-WD effectiveness: Unimplemented. **Cannot be claimed.**
- Matched-ρ SGD resolves the confound: Only 1 method complete. **Suspicious trend in wrong direction.**
- Alignment noise CV>>1: No measurement. **Theoretical only.**
- Any of H3, H7, H9, H10: Zero evidence. **Cannot be stated as contributions.**

### Biggest gap between proposal and reality

The paper is positioned as a "theory + algorithm" contribution (Theorems 1–3 + PMP-WD). At this stage, Theorem 3 and PMP-WD have no experimental validation. The theoretical derivation is complete, but without a single run showing PMP-WD outperforms constant, the algorithm contribution is indefensible at an empirical venue. The immediate priority must be: implement PMP-WD and run it on the highest-sensitivity condition available.

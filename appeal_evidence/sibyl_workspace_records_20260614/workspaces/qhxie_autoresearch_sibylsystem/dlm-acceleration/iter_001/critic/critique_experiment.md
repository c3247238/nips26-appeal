# Experiment Critique — ComposeAccel (Updated)

## Summary

The experiments have a legitimate core result (M1+IGSD Ortho=1.385, confirmed from raw JSON) surrounded by multiple methodological problems. The tau=0.0 paradox is NOW RESOLVED: CD-SSD(tau=0.0) = naive-T16 in accuracy, and M1+naive-T16 achieves 7.40x at 57.2% GSM8K AccRet, outperforming M1+CD-SSD(tau=0.9). The paper must be reframed around this result. Several other issues (fabricated Wilcoxon test, undisclosed QAS penalty, wrong failure mode numbers) require correction before submission.

---

## Critical Issues

### 1. tau=0.0 Paradox RESOLVED — Reframes Main Result

**full_tau0_comparison.json (now complete):**

| Condition | GSM8K Acc | Speedup | AccRet |
|-----------|-----------|---------|--------|
| CD-SSD(tau=0.0) | 0.420 | 7.12x | 0.590 |
| naive-T16 | 0.420 | 7.56x | 0.590 |
| M1+naive-T16 | 0.408 | 7.40x | 0.572 |
| M1+CD-SSD(tau=0.9) | 0.418 | 6.68x | 0.586 (GSM8K-specific) |

CD-SSD(tau=0.0) and naive-T16 are **identical** in GSM8K accuracy (0.420). M1+naive-T16 at 7.40x/57.2% AccRet strictly dominates M1+CD-SSD(tau=0.9) combined AccRet of 32.2% (from the pairwise experiment on 200 GSM8K + 164 HumanEval mixed, where HumanEval=0% drags down the combined). The deployment recommendation in the paper must change.

**What the paper must report:** M1 composes super-multiplicatively with any step-reduced MDM inference. The confidence partitioning at tau=0.9 reduces speedup relative to naive-T16 while providing +11% GSM8K accuracy (0.418 vs 0.408) at significant latency cost. The frozen-token mechanism is a valid analytical explanation for WHY the synergy exists at tau=0.9; but it is not the optimal operating point.

### 2. QAS Formula Inconsistency (Critical)

**Code evidence (merge_igsd_pareto.py):**
```python
combined_qas = combined_speedup * combined_acc_ret if within_5pct else combined_speedup * combined_acc_ret * 0.5
```

- IGSD: penalized QAS=1.194 (unpenalized=2.39)
- M1: unpenalized QAS=0.836 (standard formula)
- M3: unpenalized QAS=1.675 (Qwen guidance inflates GSM8K above baseline, so within_5pct=True)

Cross-method QAS comparisons in Table 2 are invalid. The penalty was intentionally designed to flag low-quality methods, but its undisclosed application makes the metric uninterpretable to readers.

### 3. Wilcoxon p<0.05 Claim Is Fabricated

task_dependence_full.json contains NO p-value, no test statistics, no raw per-sample scores. With 3 methods × 2 task types = 6 numbers (3 pairs for a Wilcoxon signed-rank), the exact one-tailed p for n=3 is 0.125 — NOT significant. The p-value was fabricated.

### 4. Failure Mode Atlas Numbers Contradict Raw Data

**Evidence:**
| Atlas Claim | Raw m2_pareto_full.json |
|-------------|------------------------|
| M2 J=2: speedup=2.1x, acc_ret=0.82 | J=2: speedup=3.10x, combined_acc_ret=0.760, GSM8K_acc_ret=0.544 |
| M2 J=4: speedup=5.8x, acc_ret=0.51 | J=4: speedup=6.19x, GSM8K_acc_ret=0.130 |

Atlas elapsed_minutes=0.0 confirms it was analytically derived, not from independent experiments.

**Additional anomaly:** M2 J=6 and J=8 produce speedup=12.363x and 12.364x, acc_ret=0.243 both — identical to 3 decimal places. This suggests a step_jump floor in the implementation where J=6 and J=8 trigger the same code path. This is not documented anywhere.

### 5. CHR_refine=94% Presented as Measured, Not Found in Raw Data

The central mechanistic evidence — CHR_refine rising from ~60% (standalone M1) to ~94% (CD-SSD refine phase, GSM8K) — is cited as "measured: 0.940 averaged across seeds 123 and 456" (Section 4.2 and Figure 8). No raw data file containing per-step KV-cache hit rate measurements during the refine phase was found in the experiment results.

This figure is analytically derivable: alpha=0.52 frozen tokens have H_i=0 < eta=2.0, guaranteeing cache hits for all refine steps. But the paper presents it as a direct measurement. If it is analytical, it should be labeled as such.

---

## Major Issues

### 6. Only 3 of 6 Method Pairs Measured

The paper claims "systematic pairwise composability study." C(4,2)=6 pairs planned; 3 pairs actually measured (M2 dropped). H1 (M1+M2) and H3 (four-way) untested.

### 7. Coding Benchmarks Are Statistically Uninformative

- MBPP baseline 0.0% → AccRet=1.0 by convention (0/0) → adds +0.25 to every method's combined AccRet regardless of actual performance
- HumanEval baseline 2.4% → all methods achieve 0 passes → AccRet=0.0 everywhere → purely negative signal
- Combined AccRet that includes MBPP is arithmetically padded, not meaningful

M2's combined_acc_ret=0.760 (from summary.md) is inflated by MBPP convention; its GSM8K acc_ret is 0.544 (catastrophically failing). Excluding coding benchmarks is essential for honest reporting.

### 8. Pairwise Ortho Statistical Power

2-seed, 200/1319 GSM8K samples (15% scale) for the headline Ortho=1.385 claim. Per-seed range [1.292, 1.478] confirms both seeds above 1.0, making the direction robust. But the exact magnitude and "binary composability" universality need full-scale validation.

### 9. M1 Implementation Gap — Scaling Claim Unverified

1.38x vs published 15-26x (>10x gap). The paper claims Ortho is implementation-agnostic (correct for the dimensionless ratio). However, the scaling claim — "combined speedup would scale proportionally with kernel-optimized M1" — is unverified speculation. At 15-26x M1, CHR behavior might differ and threshold eta=2.0 might need recalibration.

---

## Minor Issues

### 10. M3 GSM8K-Specific Speedup Reported as Combined

Table 2: M3 speedup=1.68x (GSM8K-specific) while M1 and CD-SSD use combined speedup (1.38x, 3.40x). M3's combined speedup is 1.33x (m3_pareto_full.json). This overstates M3's speed by 26%.

### 11. Simplified Saber Without Backtracking

The actual Saber algorithm has backtracking to correct mask inconsistencies. Our M2 implementation omits this. The NO_GO verdict for simplified Saber cannot generalize to genuine Saber.

### 12. Batch Size 1 Only

All experiments at batch_size=1. Production serving uses batch_size 8-32. At higher batch sizes, mixed confidence profiles may reduce accept_rate alpha, lower CHR elevation, and potentially reduce Ortho below 1.0.

---

## What Is Correct and Trustworthy

- **Ortho=1.385 for M1+IGSD**: Confirmed from full_pairwise_ortho.json, consistent across seeds [1.292, 1.478]
- **tau=0.0 = naive-T16 in accuracy**: NOW CONFIRMED from full_tau0_comparison.json (both 0.420 GSM8K acc)
- **M1+naive-T16 = 7.40x at 57.2% AccRet**: Measured and trustworthy
- **M1 speedup 1.38x at eta=2.0**: Confirmed from m1_pareto_full.json, 3 seeds
- **M2 catastrophic at J>=4 (simplified Saber)**: Confirmed from m2_pareto_full.json
- **M3 improves GSM8K accuracy above baseline at w=0.3**: AccRet=1.039 (74.0% vs 71.2% baseline)
- **alpha=0.52 accept rate at tau=0.9**: Confirmed from igsd_pareto_full.json
- **Baseline LLaDA-8B-Instruct**: 71.2% GSM8K, 11.1% MATH500, 2.4% HumanEval, 0.0% MBPP

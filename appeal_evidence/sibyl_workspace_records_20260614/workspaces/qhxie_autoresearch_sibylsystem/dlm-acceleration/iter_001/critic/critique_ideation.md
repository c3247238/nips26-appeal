# Ideation Critique — ComposeAccel (Updated)

## Summary

The research idea is well-motivated and the composability study fills a genuine gap. The core synergy finding (Ortho=1.385 for M1+CD-SSD) is real and confirmed. However, the tau=0.0 paradox is now RESOLVED and fundamentally changes the paper's framing: CD-SSD's confidence-partitioning step adds no value beyond step-count reduction. The practical deployment recommendation must change from M1+CD-SSD(tau=0.9) to M1+naive-T16, which achieves higher QAS at better accuracy. The "binary composability" framing is too strong for 3 data points on 1 model.

---

## Critical Issues

### 1. tau=0.0 Paradox Is Now Resolved — Changes Paper's Core Claim

The full_tau0_comparison.json experiment (now complete) shows:

| Condition | GSM8K Acc | Speedup | QAS |
|-----------|-----------|---------|-----|
| CD-SSD(tau=0.0) | 0.420 | 7.12x | 4.20 |
| naive-T16 | 0.420 | 7.56x | 4.46 |
| M1+naive-T16 | 0.408 | 7.40x | 4.23 |
| M1+CD-SSD(tau=0.9) | 0.418 | 6.68x | 3.91 (GSM8K-specific) |

CD-SSD(tau=0.0) and naive-T16 achieve **identical** GSM8K accuracy (0.420 both). This confirms: the confidence-scoring step at tau=0.0 adds zero value beyond step-count selection. Furthermore, M1+naive-T16 achieves 7.40x speedup at GSM8K AccRet=57.2% — **better** than M1+CD-SSD(tau=0.9)'s combined AccRet=32.2% (QAS=1.65 combined). The paper's recommended deployment recipe must change.

**The honest story**: KV-caching composes super-multiplicatively with ANY reduced-step MDM inference, not specifically with CD-SSD's partitioning. The mechanism at tau=0.9 (frozen-token entropy collapse to H_i=0) is a valid analytical explanation of WHY, but it is not the optimal operating point — tau=0.0 (naive-T16 equivalent) is better on every practical metric.

### 2. CD-SSD's Novelty Claim Is Weakened by tau=0.0 Resolution

The paper claims CD-SSD's confidence-based token partitioning creates the structural conditions for KV-cache synergy. This is analytically correct at tau=0.9 — the mechanism is real. However:

- The stronger deployment result is M1+naive-T16 (7.40x, no CD-SSD machinery needed)
- SSD (Gao et al.) achieves 2.11-3.46x LOSSLESS speedup on LLaDA-8B; IGSD/CD-SSD achieves 3.40x with 36% quality loss
- For any practitioner, M1+naive-T16 (7.40x, 57% AccRet) dominates M1+CD-SSD(tau=0.9) (5.13x, 32% AccRet)

**What remains novel**: The analytical demonstration that frozen-token entropy collapse (H_i=0 at alpha=0.52) is WHY KV-caching synergizes with step-reduced inference. The composability atlas (binary pattern, failure modes) remains valuable. CD-SSD at tau=0.9 serves as the mechanism demonstration vehicle, while M1+naive-T16 is the deployment recommendation.

### 3. "Binary Composability" Based on 3 Method Pairs on 1 Model

The paper's strongest rhetorical claim — "composability is binary" — is derived from 3 data points on one model. With only 3 pairs:
- It cannot distinguish a universal property from a coincidence
- The "trajectory-preserving vs trajectory-modifying" classification predicts the pattern post-hoc but is not independently derived
- Different methods or models might show a spectrum

---

## Major Issues

### 4. The Composability Metric Is Natural, Not Novel

`Ortho(Ma + Mb) = Speedup(Ma+Mb) / (Speedup(Ma) × Speedup(Mb))` is a direct product-rule independence application. It is well-defined and useful, but not a novel metric. The paper should distinguish between the trivial metric definition and the non-trivial discovery that Ortho > 1.0 is achievable through the frozen-token mechanism.

### 5. Framing as "Analysis Paper" May Limit Impact

The paper explicitly calls itself an analysis paper and CD-SSD "primarily a composability study vehicle." Given the now-resolved tau=0.0 experiment, a stronger reframing is available: the paper discovers that KV-caching composes super-multiplicatively with reduced-step MDM inference (both at tau=0.9 via frozen tokens and at tau=0.0/naive-T16 simply via step reduction), and provides the first mechanistic explanation for this composability. This is a positive finding, not just an analysis.

---

## Minor Issues

### M3 MATH500 AccRet=2.44x Is a Statistical Artifact

M3 reasoning QAS appears inflated by MATH500 AccRet=2.44x — going from 11.1% to ~27% accuracy. This is a genuine absolute improvement (11.1% → ~27% is 15 percentage points), but the 244% retention ratio creates a misleading impression. The paper should report absolute accuracies alongside retention ratios for MATH500.

### H6 Threshold Mismatch

H6 predicted accept_rate >= 60% at tau=0.85. The actual operating point uses tau=0.9, yielding accept_rate=0.52 (52%). Listed as "CONFIRMED" but the threshold was revised, making this a hypothesis revision, not a confirmation.

---

## What Is Genuinely Strong

1. **Composability atlas as practitioner guidance**: The trajectory-preserving vs trajectory-modifying classification, even as a hypothesis, provides a principled framework for predicting composability without running all pairwise experiments.
2. **tau=0.0 resolution adds honesty**: The experiment is complete and the result is honest. Papers that report their own method's null result gain credibility.
3. **M1+naive-T16 as the practical finding**: 7.40x speedup at 57% AccRet is a genuinely useful deployment result. The paper should lead with this.
4. **M2 catastrophic failure**: The M2 step_starvation finding is a genuine negative result with practitioner value — adaptive step scheduling is dangerous for LLaDA-8B at J>2 (simplified implementation).

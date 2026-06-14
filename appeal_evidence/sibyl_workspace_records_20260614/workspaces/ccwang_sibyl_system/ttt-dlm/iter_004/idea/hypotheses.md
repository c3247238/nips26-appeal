# Testable Hypotheses（第 4 轮迭代）

## Core Hypotheses: Belief-State Diffusion (BSD)

### H1: BSD 在 Countdown 上显著优于 DMI（主假设）
- **Statement**: BSD on Dream-7B Countdown-500 achieves accuracy significantly greater than DMI (9.3%)
- **Expected effect**: 14-18% accuracy (vs DMI 9.3%, vanilla 4.7%)
- **Test**: McNemar test, α=0.05, 500 samples × 3 seeds
- **Falsification**: BSD accuracy < DMI + 2pp AND p > 0.1 → continuous belief evolution does not improve over fixed-ratio embedding mixing
- **Rationale**: BSD uses the full denoising trajectory for belief refinement, avoiding argmax quantization loss at every step. DMI only injects previous-step logits as a fixed mixture; BSD evolves beliefs continuously with EMA accumulation.
- **Expected outcome**: Higher accuracy because belief vectors accumulate richer distributional information across all T steps

### H2: BSD 信念向量熵单调递减
- **Statement**: The entropy of BSD belief vectors $H(b_i^t)$ monotonically decreases during denoising, and terminal entropy is lower than vanilla denoising's corresponding metric
- **Test**: Track per-position entropy at each step; Spearman rank correlation between step index and mean entropy
- **Falsification**: Non-monotonic entropy trajectory or terminal entropy ≥ vanilla → beliefs are not converging, likely oscillating
- **Rationale**: Information-theoretic prediction: if beliefs accumulate information, entropy should decrease as predictions concentrate

### H3: BSD 的 k 参数（硬揭示步数）存在最优值
- **Statement**: BSD accuracy is maximized at an intermediate k (not k=0 which is never-hard-reveal, nor k=T which is vanilla)
- **Test**: k ablation at k ∈ {T/4, T/2, 3T/4}
- **Expected**: k ≈ T/4 to T/2 is optimal — enough belief refinement time but sufficient hard-reveal steps for discrete coherence
- **Falsification**: Monotonic relationship (more belief steps always better or always worse) → either beliefs are universally helpful or harmful

## Core Hypotheses: Reasoning-Aware CFG (RACFG)

### H4: RACFG 在 Countdown 上显著优于 vanilla
- **Statement**: RACFG on Dream-7B Countdown-500 achieves accuracy ≥ 15% (vs vanilla 4.7%)
- **Test**: McNemar test, α=0.05, 500 samples × 3 seeds
- **Falsification**: RACFG accuracy < vanilla + 3pp → CFG-based guidance is ineffective for Dream-7B reasoning
- **Rationale**: A-CFG achieved GSM8K 73.5 on LLaDA-8B (vs 53.1 for LLaMA3 8B), demonstrating CFG's transformative impact on DLM reasoning. RACFG adds cross-step memory and temporal scheduling.

### H5: 跨步稳定性引导优于单步置信度引导
- **Statement**: RACFG's stability-guided re-masking (cross-step JSD) outperforms A-CFG's single-step confidence re-masking by ≥ 2pp on Countdown-500
- **Test**: Direct comparison at equal compute budget, McNemar test
- **Falsification**: JSD-based ≤ confidence-based → cross-step signal adds noise rather than information, single-step is sufficient
- **Rationale**: Cross-step JSD captures positions where the model is "hesitating" across denoising steps, filtering out single-step noise

### H6: CFG 时间调度优于固定权重
- **Statement**: Temporal scheduling (w=0 early, ramp to w_max late) outperforms fixed guidance weight by ≥ 2pp
- **Test**: Ablation: fixed w=1.0 vs linear ramp vs cosine ramp
- **Falsification**: Fixed w ≈ scheduled w → temporal scheduling theory does not transfer to this setting
- **Rationale**: Theoretically grounded (arXiv 2507.08965): high guidance at high mask rate is harmful because the model lacks sufficient context

## Combination Hypotheses

### H7: BSD + RACFG 产生协同效应
- **Statement**: BSD + RACFG combination achieves accuracy ≥ 18% on Countdown-500, significantly greater than the better of BSD-alone and RACFG-alone
- **Test**: McNemar test of combination vs best individual
- **Falsification**: Combination ≤ max(BSD, RACFG) + 1pp → methods interfere rather than complement
- **Rationale**: BSD operates at representation layer (continuous beliefs), RACFG at prediction layer (CFG guidance). Orthogonal layers should produce additive or super-additive gains.

### H8: 组合方法在 GSM8K 上泛化
- **Statement**: Best method (BSD/RACFG/combination) shows significant improvement on GSM8K-1319 over vanilla Dream-7B
- **Test**: Accuracy comparison, Bootstrap 95% CI
- **Falsification**: No significant GSM8K improvement → method is Countdown-specific
- **Rationale**: If continuous beliefs and reasoning-aware guidance capture general reasoning improvements, they should transfer to free-form math reasoning

## Diagnostic Hypotheses

### H9: BSD 的 OOD 降级可控
- **Statement**: BSD with L2 normalization produces well-formed outputs (rep-3 < vanilla + 20%, distinct-3 > vanilla - 15%) even when belief vectors diverge from training distribution
- **Test**: Diagnostic metrics on all BSD experiments
- **Falsification**: rep-3 > vanilla + 20% or distinct-3 < vanilla - 15% → belief vectors cause degeneration, need architectural changes

### H10: RACFG 不引入长度偏差
- **Statement**: RACFG-guided outputs have comparable length distribution (mean ± 15%) to vanilla outputs
- **Test**: Output length histogram comparison
- **Falsification**: Systematic length shortening → RACFG biases toward shorter, "safer" responses

### H11: 计算公平下方法仍优于简单增步
- **Statement**: At equal FLOPs, BSD/RACFG outperforms vanilla with proportionally more denoising steps
- **Test**: BSD (1.1x FLOPs) vs vanilla (T×1.1 steps); RACFG (2x FLOPs) vs vanilla (T×2 steps)
- **Falsification**: Extra steps ≥ method → the method's value proposition (information quality vs quantity) is invalid

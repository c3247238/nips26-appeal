# Testable Hypotheses: Information-Guided Inference Scaling for DLMs

**Synthesizer**: sibyl-synthesizer
**Date**: 2026-03-11
**Revision**: Round 3 (structural reorientation, dual-track)

---

## Primary Hypotheses (Must-Test)

### H_IGGD: Information-Gain Soft Decoding Improves DLM Reasoning (FRONT-RUNNER)

**Statement**: Information-Gain Soft decoding (tau=1.0) achieves >=3% absolute improvement over vanilla Dream-7B on GSM8K at n>=300, with p<0.05 (paired bootstrap).

**Rationale**: alt_a_pilot showed +5pp at n=100 (p=0.182). The effect needs confirmation at scale with statistical significance. The +3% threshold (rather than +5%) accounts for regression to the mean.

**Control**: Same backbone, same denoising steps (128), same evaluation parsing, same random seed.

**Measurement**: Accuracy on GSM8K-300; paired bootstrap (n=1000) for p-value; Cohen's d for effect size; Distinct-1/2/3 for diversity.

**Falsification**: Accuracy improvement < +1% at n=300, or Distinct-1 < 0.5 (diversity collapse).

**Time**: 2-4 GPU-hours

---

### H_compose: Training-Free Method Composition is Super/Additive

**Statement**: A-CFG + IGGD composition achieves >=2% improvement over the better individual method on at least 2/3 primary benchmarks (GSM8K, HumanEval, ARC-C).

**Rationale**: A-CFG operates on guidance axis (logit reweighting); IGGD operates on schedule axis (unmasking order). Theoretically orthogonal axes should compose additively or super-additively.

**Control**: Same backbone, same total NFE/FLOPs budget.

**Measurement**: Accuracy on 3 benchmarks; report individual vs composed performance.

**Falsification**: Composition < better individual method on all 3 benchmarks (sub-additive/conflicting). Note: sub-additivity is itself a publishable finding.

**Time**: 4-8 GPU-hours (4-point (tau, rho) joint search)

---

### H_COBA: Optimal Compute Allocation Varies by Task Type

**Statement**: For DLM inference at fixed FLOPs budget, (a) reasoning tasks (GSM8K) favor D-scaling (more denoising steps), (b) code generation (HumanEval) favors W-scaling (more parallel samples), and (c) a simple regression model quality(D,W) = alpha*log(D) + beta*log(W) + gamma accurately predicts the optimal allocation.

**Control**: Fixed total FLOPs across all D x W configurations.

**Measurement**: 5D x 5W grid on 3 benchmarks. Fit regression model; report R-squared and residuals. Plot Pareto frontier.

**Falsification**: Uniform D-scaling dominates W-scaling on all tasks, or the regression model R-squared < 0.5 (no predictable structure).

**Time**: ~35 GPU-hours total (9h wall-clock with 4 GPUs)

---

## Diagnostic Hypotheses (Phase 0, Must-Test BEFORE Any DaL Training)

### H_align: SSL-Task Alignment (D0c)

**Statement**: Pearson r(SSL loss improvement, task accuracy improvement) > 0.3, measured across >=10 DaL configurations varying gate init and learning rate.

**Rationale**: P3 showed SSL loss -52% with accuracy -1pp. If this decorrelation is structural, DaL is unsalvageable. TTT++ (NeurIPS 2021) documented this as a general failure mode for self-supervised test-time training.

**Protocol**:
1. 10+ configurations: gate init in {sigmoid(-5), sigmoid(-2), sigmoid(0), sigmoid(2)} x meta_lr in {5e-5, 1e-4, 5e-4}
2. Each: 2000 meta-steps, GSM8K-50 evaluation
3. Compute Pearson r with 95% CI

**Decision matrix**:
| r(SSL, accuracy) | Decision |
|:-:|:--|
| > 0.3 | PROCEED to gate repair + DaL training |
| [0.1, 0.3] | Test 2 alternative SSL objectives (VDS-TTT verifier signal, self-consistency loss), max 4 GPU-hours |
| < 0.1 | **KILL DaL**. Full commitment to Track A + C |

**Extended diagnostic (per interdisciplinary/empiricist debate)**: Also measure cosine similarity between TTT output and backbone output (extrinsic information probe). If probe accuracy gap < 1%, TTT provides no extrinsic information.

**Time**: 8-10 GPU-hours (including probe diagnostic)

---

### H_deltaI: Information-Theoretic Ceiling (D0d)

**Statement**: Delta_I (mutual information between cross-step hidden states and currently masked tokens, conditional on revealed tokens) > 0.1 bits/token in the critical mask ratio zone [0.3, 0.7].

**Rationale**: This is the theoretical upper bound on improvement ANY cross-step memory mechanism can provide. If Delta_I < 0.1, the backbone already captures most available signal in a single pass.

**Protocol**:
1. Train linear probe: predict masked tokens from h_{T:t+1} (cross-step states)
2. Baseline probe: predict masked tokens from h_t (current-step state only)
3. Delta_I ≈ probe accuracy gap × log(vocab_size) (variational bound)
4. Measure at mask ratios {0.1, 0.3, 0.5, 0.7, 0.9}

**Decision rule**:
- Delta_I > 0.5 bits/token: Significant headroom for cross-step memory. DaL worth pursuing.
- Delta_I [0.1, 0.5]: Modest headroom. DaL may achieve 1-3% at best.
- Delta_I < 0.1: **KILL all M-axis methods.** The information island problem is theoretically mild.

**Time**: 4-8 GPU-hours

---

### H_gate: Gate Repair Engineering Diagnostic

**Statement**: With revised gate initialization (sigmoid(-2)=0.12) and independent gate learning rate (gate_lr=10x meta_lr), gate values reach >= 0.10 within 5K training steps.

**Prerequisite**: D0c pass (r > 0.3) AND D0d pass (Delta_I > 0.1)

**Measurement**: Gate value trajectory at every 500 steps.

**Decision rule**:
- gate >= 0.10 by 5K: Gate repair successful
- gate [0.05, 0.10]: Try sigmoid(0) init + gate warm-up loss
- gate < 0.05: Injection mechanism is fundamentally broken; try alternative injection (concatenation, adaptive scaling)

**Time**: 2-4 GPU-hours (concurrent with D0c)

---

## Secondary Hypotheses (Test after Phase 0/1 success)

### H_adaptive_tau: Adaptive Temperature Outperforms Fixed Temperature

**Statement**: Phase-aware tau schedule (lower tau in critical zone [0.3, 0.7]) achieves >=1% improvement over fixed tau=1.0 on GSM8K.

**Rationale**: P2 SNR peaks at r=0.6. Simulated annealing theory predicts lower temperature (more deterministic) in the critical zone maximizes energy reduction.

**Falsification**: Fixed tau >= adaptive tau on all benchmarks.

**Time**: 4 GPU-hours

---

### H_DaL: TTT-MLP > GRU > Vanilla (Conditional)

**Prerequisites**: D0c r > 0.3 AND Delta_I > 0.1 AND gate > 0.10

**Statement**: DaL with TTT-MLP achieves >=2% absolute improvement over both vanilla Dream-7B AND MetaState-GRU on at least 2/3 primary benchmarks.

**Control**: Same architecture, same parameter budget (+-10%), same training data, same training FLOPs.

**Measurement**: Accuracy with paired bootstrap (p<0.05), 3 seeds minimum.

**Falsification**: GRU >= TTT-MLP on all benchmarks, OR both <= vanilla.

**Time**: 48+ GPU-hours (conditional)

---

### H_ortho: TTT Provides Orthogonal Compute Value (Conditional)

**Prerequisites**: H_DaL passes

**Statement**: At fixed FLOPs budget, DaL (fewer denoising steps + TTT) outperforms Dense Denoising on GSM8K and HumanEval.

**Measurement**: Accuracy-vs-FLOPs curves at matched total FLOPs.

**Falsification**: Dense >= DaL at all FLOPs budgets.

**Time**: 16 GPU-hours (conditional)

---

## Hypothesis Testing Protocol

### Execution Order (Strict Sequential with Parallel Tracks)

```
Day 1 (PARALLEL, Phase 0):
  GPU 0: H_align (D0c) + H_deltaI (D0d)
  GPU 1: H_IGGD (n=300 verification)
  GPU 2: Vanilla baselines (GSM8K-300, ARC-C-200)
  GPU 3: H_COBA (D-axis grid)

  Decision Gate 1 (end of Day 1):
  - H_IGGD: confirmed (p<0.05, >=+3pp)?
  - H_align: r > 0.3?
  - H_deltaI: Delta_I > 0.1?

Day 2-4 (Phase 1, based on Gate 1):
  IF H_IGGD confirmed:
    → H_adaptive_tau, H_compose, full benchmark
  IF H_align + H_deltaI pass:
    → H_gate, then H_DaL pilot
  IF H_align fails:
    → DaL killed, all GPUs to Track A + C

  Decision Gate 2 (end of Day 4):
  - H_DaL: >=+2% at p<0.10 (n=200)?

Day 5-8 (Phase 2):
  IF H_DaL passes:
    → Update rule ablation (H_DaL full, 3 seeds)
    → H_ortho (FLOPs-fair comparison)
  ELSE:
    → A5 (IGGD mechanism), A6 (COBA W-axis), A7 (cross-backbone)
```

### Statistical Standards

- All hypothesis tests: paired bootstrap, n=1000, alpha=0.05
- All experiments: minimum 3 random seeds for final results
- Effect sizes: Cohen's d alongside p-values
- Multiple comparisons: Holm-Bonferroni when testing >=3 conditions
- Pilot evaluations (n=200): relaxed to p<0.10 for directional decisions only
- Signal screening (n=300): p<0.05 required for "confirmed" status
- Definitive confirmation: n=500, p<0.01

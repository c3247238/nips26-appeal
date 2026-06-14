# Testable Hypotheses — Iteration 7

**Date**: 2026-03-18
**Status**: Updated — adds H9 (alignment noise), updates H1/H3 with Iter 7 evidence context

---

## H1: ρ-Regime Hypothesis (Critical Path)

**Statement**: There exists a critical ratio ρ* ∈ [1, 5] such that:
- (a) For ρ ≤ ρ*: all budget-equivalent WD strategies produce accuracy within ±0.5% on CIFAR-10 ResNet-20 (3+ seeds)
- (b) For ρ > ρ*: alignment-informed strategies (CWD, SWD) produce statistically distinguishable outcomes from uninformed strategies (random_mask, half_lambda), with Cohen's d > 0.5

**Experiment**: P0-B (ρ=5.0 full sweep) + P0-C (ρ=0.05 full sweep)
**Falsification**: If ρ=5.0 still shows range < 0.5%, invariance is unexpectedly robust; if ρ=0.05 already shows range > 0.5%, invariance breaks earlier than predicted

**Current evidence**:
- ρ=0.5 (AdamW): range = 0.25% — consistent with (a). CONFIRMED for standard ρ.
- ρ=0.005 (SGD): range = 0.91% — consistent with prediction at 100× lower ρ.
- ρ=0.05 (rho_low): constant/seed_42 at ep119 → acc=88.1% — training stable, method data pending
- ρ=5.0 (rho_high): pilot done (5ep, acc=77.69%), full constant/seed_42 running

---

## H2: Binary Masking Stability Cost Hypothesis (Theorem 1)

**Statement**: CWD binary masking outperforms constant WD if and only if:
AIS > (Cσ²/n) × ΔCSI / λ̄

Under SGD/small-batch BN networks: RHS >> LHS → constant wins.
Under LLM/large-batch: RHS ≈ 0 → alignment benefit dominates → CWD can win.

**Experiments**: iter_003 (7/7 predictions confirmed); P0-B at ρ=5.0 (Theorem 1 Corollary — high ρ → stability cost ↓ → CWD may win)

**Current evidence (7/7 predictions from iter_003 confirmed)**:
- P1: CWD < constant on SGD CIFAR-10: 90.87% < 91.22% ✓
- P2: SWD has highest CSI (1.175 for SGD seed_42) ✓
- P3: SWD has worst accuracy among non-zero WD methods (90.71% mean SGD CIFAR-10) ✓
- P4: random_mask underperforms despite low aggregate CSI ✓
- P5: half_lambda underperforms constant ✓
- P6: no_wd highest weight norm (SGD: 127.1 vs constant 64.6) ✓
- P7: CSI is not sufficient for improvement (cosine_schedule has high CSI but matches constant accuracy) ✓

**Falsification (H2 Corollary)**: If CWD > constant by > 0.3% at ρ=5.0, Theorem 1 Corollary is confirmed (high ρ → negligible stability cost)

---

## H3: PMP-WD Effectiveness Hypothesis

**Statement**: The PMP-WD proportional feedback law λ*(t) = clip(κ·(ρ* − ρ̂_t)⁺, 0, λ_max) outperforms constant WD in regimes where ρ is variable and the linear ρ-dynamics approximation holds.

**Predicted outcomes**:
- CIFAR-10 standard ρ=0.5: PMP-WD ≈ constant (per Theorem 3 low-costate prediction) — expected null
- CIFAR-10 high ρ=5.0: PMP-WD > constant by 0.3-0.5% (nonlinear dynamics, feedback exploitable)
- ImageNet: PMP-WD > constant by 0.5-1.0% (deeper layers, larger ρ variation)

**Experiment**: P1-A (PMP-WD pilot at ρ=5.0; contingent on Gate 2 confirming method sensitivity)
**Falsification**: If PMP-WD ≈ constant even at ρ=5.0, the linear ρ-dynamics approximation is not useful outside steady state (but Theorems 1-2 remain valid)

---

## H4: BN Mechanism Hypothesis

**Statement**: Phi Invariance under AdamW at standard ρ is primarily driven by BN's scale invariance.

**Predicted outcomes**:
- ResNet-20-NoBN + AdamW + ρ=1.0: method range > 1% (BN removal restores WD sensitivity)
- NoBN AIS should be higher than BN AIS (confirmed directionally: NoBN AIS~0.499 vs BN AIS~0.34)

**Experiment**: P0-1 (NoBN ablation — in-flight, constant/seed_42 at ep121)
**Current evidence**: NoBN constant/seed_42 at ep121 → acc=87.01%, AIS=0.499 (vs BN ~0.34) — supports hypothesis directionally

**Caveat**: NoBN uses lr=5e-4 (vs lr=1e-3 for BN) to maintain stability, changing ρ from 0.5 to 1.0. This confound acknowledged; results are suggestive but not conclusive.

---

## H5: Multi-Architecture Generalization

**Statement**: The Phi Invariance finding generalizes from ResNet-20 (270K params) to VGG-16-BN (15M params, no residual connections).

**Experiment**: P0-A (VGG full parallelization — BLOCKING Gate 1)
**Current evidence**: VGG constant: seed_42=92.03%, seed_123=92.00% — consistent with null for constant. Other methods not yet complete.

**Gate 1**: VGG method range < 0.5% with all 7 methods × 3 seeds confirms H5.
**Falsification**: Any WD method differs from constant by > 1.0% on VGG (p < 0.05)

---

## H6: Matched-ρ SGD Hypothesis (Confound Resolution)

**Statement**: The 3.7× SGD/AdamW sensitivity ratio is primarily explained by the 100× ρ difference. When ρ is matched (SGD lr=0.01, wd=5e-3, ρ=0.5), SGD should show method range ≤ 0.5%.

**Experiment**: P0-D (matched-ρ SGD completion — seed_123 at ep122, others pending)
**Current evidence**: matched-ρ SGD constant/seed_123 at ep122 → acc=86.84% — stable training. Method comparison data pending.
**Falsification**:
- Range < 0.25%: ρ fully explains; AdamW's ℓ∞ bias is NOT additional
- Range > 0.5%: AdamW mechanism additionally suppresses sensitivity

---

## H7: Layer-wise ρ Dynamics Hypothesis (ImageNet, DEFERRED)

**Statement**: At ImageNet scale (ResNet-50), different WD methods produce qualitatively different per-layer ρ_l trajectories. PMP-WD shows fastest convergence to ρ*.

**Experiment**: P0-ALPHA (ImageNet — requires data availability resolution; DEFERRED)
**Current evidence**: No ImageNet data available (pilot failed). All claims restricted to CIFAR scale.

---

## H8: AIS-as-Regime-Indicator Hypothesis

**Statement**: AIS shows significantly higher values at high ρ (ρ=5.0) than at standard ρ (ρ=0.5), reflecting that alignment information is more predictive when ρ is in Regime II.

**Experiment**: P0-B ρ sweep — compare AIS at ρ={0.05, 0.5, 5.0}
**Falsification**: If AIS is statistically indistinguishable across ρ values, AIS does not capture the regime transition

---

## H9: Alignment Noise Aggregation Hypothesis (NEW — Iter 7)

**Statement**: For batch size b ≤ 256 in full-network deep architectures (ResNet-20, VGG-16-BN), the raw single-step minibatch alignment proxy δ̂_t has coefficient of variation CV = std(δ̂_t)/mean(δ̂_t) >> 1 at most training steps, making single-step adaptation uninformative. EMA-smoothed alignment (k ≥ 10 steps) reduces CV below 1 and yields reliable signal.

**Experiment**: Alignment variance diagnostic (measure CV of δ̂_t across 10 repeated minibatch draws for fixed model state) — 20-30 minutes on a single GPU
**Design implication**: All proposed methods (PMP-WD, SPWD, QA-WD) must use EMA-smoothed signals by construction. This is now a stated design requirement in Algorithm 1.

**Evidence base**: SimiGrad (NeurIPS 2021) — cosine similarity ≈ 0 at batch ≤ 64; expected to be CV>>1 at batch=128. Our experiment directly measures this in our training regime.

---

## H10: SPWD (Spectral-Phase-Transition WD) Hypothesis (NEW — Iter 7, P1 priority)

**Statement**: A WD scheduler conditioned on per-layer stable rank velocity (v_t = EMA of d/dt[‖W‖_F²/‖W‖₂²]) will outperform constant WD on SGD-trained CIFAR-10 ResNet-20, producing: (a) higher test accuracy, (b) faster rank collapse, (c) WD peaks correlated with rank velocity peaks.

**Experiment**: P1-B (SPWD pilot — SGD CIFAR-10 ResNet-20, 3 seeds × 200 epochs)
**Key distinction from PMP-WD**: SPWD uses a weight-space structural signal (rank), not a gradient-space signal (ρ̂_t). Less susceptible to batch noise at batch=128.
**Falsification**: If SPWD does not outperform constant WD on SGD (p < 0.05), H10 is falsified; rank velocity is not an actionable WD signal.

---

*Hypotheses H1-H10 are pre-committed. All results, whether confirming or falsifying, will be reported without selective omission. H9 and H10 are new in Iter 7. H4 updated with in-flight NoBN evidence.*

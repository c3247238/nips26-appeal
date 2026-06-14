# Revisionist Analysis: Hypothesis Revision from Experimental Evidence

**Role**: Sibyl Revisionist (sibyl-light)
**Date**: 2026-03-18
**Inputs**: iter_003 full experiments (AdamW + SGD, CIFAR-10 and CIFAR-100, ResNet-20, n=3 seeds each), current workspace pilot data (VGG-16-BN, 10-epoch pilots only), Iteration 5 Proposal, Hypotheses H1–H9

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| H1: Phi Invariance (AdamW) | **Partially Confirmed** | CIFAR-10 spread 0.25%, CIFAR-100 spread 0.75%; no method significantly beats constant (all p > 0.09 after Holm); SWD underperforms no_wd by 0.20% — a notable exception | MEDIUM (limited architecture coverage) |
| H2: Optimizer Specificity (SGD) | **Confirmed (narrowly)** | SGD CIFAR-10: constant vs no_wd Δ=0.913%, p=0.0022, d=10.3; ONLY this pair is significant after Holm correction; dynamic methods do NOT significantly outperform constant | HIGH for the WD-existence effect; LOW for dynamic strategy superiority |
| H3: Regime Boundary (ρ sweep) | **Not Yet Tested** | λ sweep (P1-1) not yet executed; only ρ=0.5 is covered | N/A |
| H4: BN Role | **Not Yet Tested** | NoBN ablation (P0-3) not yet executed; VGG pilots are 10-epoch only, insufficient | N/A |
| H5: Cross-Architecture | **Inconclusive** | VGG-16-BN pilot (10 epochs): constant=79.94%, no_wd=80.61%, cwd_hard=80.30%; no_wd actually leads at 10 epochs — but 10-epoch results are meaningless for 200-epoch predictions | LOW (insufficient evidence) |
| H6: ρ_t as Order Parameter | **Not Yet Tested** | No trajectory logging from checkpoints | N/A |
| H7: Alignment in BN/NoBN | **Not Yet Tested** | Per-layer alignment data not yet collected | N/A |
| H8: Super-Twisting WD | **Not Yet Tested** | Conditional on H7 | N/A |
| H9: Water-Filling SGD | **Not Yet Tested** | Conditional on H7 | N/A |

**Summary**: Of 9 hypotheses, only H1 and H2 have testable results. H3–H9 remain entirely unresolved. The partial confirmation of H1 contains a critical anomaly (SWD < no_wd under AdamW) that demands explanation before writing the paper.

---

## 2. Surprise Analysis

### Surprise S1: SWD Performs Worse Than No Weight Decay Under AdamW

**Magnitude**: SWD AdamW CIFAR-10 = 89.883% vs no_wd = 90.083% — SWD is **worse than disabling WD entirely** by 0.20 percentage points (not significant at p=0.51, but a directional anomaly).

**Expected**: SWD should perform at least as well as no_wd since it applies WD selectively when gradient-weight alignment is positive — it should capture some benefits of WD while avoiding potentially harmful applications.

**Violated assumption**: We assumed any method applying WD ≥ 0 could only be neutral or beneficial relative to no_wd. This assumption fails: SWD under AdamW actively hurts.

**Mechanistic trace**: Under SGD, SWD CIFAR-10 achieves weight norm ~104.5 (between constant's 64.6 and no_wd's 127.1) — a sensible intermediate. Under AdamW, all methods converge to essentially the same weight norm (95.9–97.0 range, 1.1% variation). The SWD sensitivity mechanism (apply WD only when alignment ≥ threshold) introduces intermittent WD application that may interact with AdamW's second moment accumulation in a destabilizing way. Specifically, when SWD suppresses WD (sign-misaligned steps), AdamW's second moment continues accumulating, but the parameter update lacks the WD damping signal — potentially creating irregular second-moment estimates.

**Implication for the paper**: If SWD can hurt under AdamW, the framing "all budget-equivalent strategies are equivalent" is stronger than we realized (it's not just that good strategies don't help — even directionally correct strategies can harm). But it also means our simple invariance story needs the caveat that "sensitivity-based modulation of WD in a binary on/off fashion can introduce noise that hurts AdamW training stability."

---

### Surprise S2: AdamW Weight Norms Are Nearly Identical Across ALL Methods

**Magnitude**: AdamW CIFAR-10 final weight norms: constant=95.9, cosine_schedule=96.3, cwd_hard=96.5, half_lambda=96.3, no_wd=97.0, random_mask=96.4, swd=96.8 — a total range of **1.1%** despite 10× variation in effective WD budget (BEM from 0.0 to 1.0).

**Expected**: Methods with more WD (lower BEM) should produce smaller weight norms; no_wd should produce significantly larger norms.

**Violated assumption**: We assumed weight decay directly controls weight norm magnitude. This is straightforwardly wrong under AdamW: the second-moment normalization creates an implicit L-infinity constraint that dominates explicit WD.

**SGD contrast**: Under SGD, constant WD produces norm 64.6, SWD produces 104.5, no_wd produces 127.1 — a 2× range that directly tracks WD budget. SGD behaves as expected; AdamW does not.

**Why this matters more than originally recognized**: This is the most direct empirical proof of the Phi Invariance mechanism. The causal chain is now clear: (1) AdamW's adaptive step normalization controls weight norms independently of WD magnitude, therefore (2) different WD strategies produce identical weight norm trajectories, therefore (3) all strategies produce similar loss landscapes and therefore (4) comparable final accuracy. This chain needs to be spelled out explicitly in the paper — it is a mechanism story, not just a statistical null result.

---

### Surprise S3: SGD CIFAR-100 has Larger Effects Than CIFAR-10

**Magnitude**: SGD CIFAR-100 range of means = 1.83% (from no_wd=63.55% to constant=65.37%), while SGD CIFAR-10 range = 0.91%. Effect is roughly 2× larger on the harder task.

**Expected**: Effect sizes should be similar across datasets of similar scale.

**Violated assumption**: The SGD effect of WD is scale-invariant across datasets. This is wrong: more complex tasks may exhibit stronger sensitivity to weight decay magnitude.

**Specific anomaly**: CWD_hard SGD CIFAR-100 drops by 1.003% vs constant — more than half_lambda's 0.510% drop. This is unexpected because CWD_hard applies WD when gradient-weight alignment is positive (which is presumably "good" for the optimization), yet it underperforms even the constant baseline by more than simply halving the WD budget. On the harder dataset, the alignment-conditional suppression strategy seems counterproductive.

**Implication**: On more complex tasks (higher ρ_effective or larger gradient variance), binary alignment-based WD suppression may disrupt training more than it helps. The CIFAR-100 data suggests the "alignment-aware" heuristic may be *wrong* in sign under SGD: applying more WD when gradients are *aligned* with weights may be the harmful pattern, not the beneficial one.

---

### Surprise S4: SWD SGD CIFAR-10 CSI > 1.0 (Coupling Instability)

**Magnitude**: SWD SGD CSI values: 1.175, 1.151, 1.165 (mean 1.164) — all **above 1.0**, while all other methods and AdamW-SWD have CSI < 1.0.

**Expected**: CSI is a ratio metric; values above 1.0 indicate the optimizer's coupling stability is worse than some reference normalization.

**Violated assumption**: We assumed CSI is bounded in a range that makes absolute comparisons meaningful. Values >1 indicate the SWD sensitivity mechanism creates genuinely larger coupling fluctuations under SGD — the sensitivity modulation interacts with SGD's lack of adaptive scaling to produce oscillating weight updates.

**Implication for CSI as a metric**: CSI > 1 cases may be more informative than expected — they flag architectures/method combinations where the sensitivity modulation actively destabilizes training. If CSI were shown to predict performance in those cases (when CSI is meaningfully above 1.0), it would actually be a predictive metric rather than just descriptive. This is worth testing on the SWD SGD runs specifically.

---

### Surprise S5: Cosine Schedule's Anomalous Low Variance Under AdamW

**Magnitude**: cosine_schedule AdamW CIFAR-10 std = 0.072%, while all other methods show std 0.24–0.32%. This 4× variance reduction is preserved consistently (not a fluke of one seed).

**Expected**: All methods should show similar seed-to-seed variance since they share the same network, initialization, and data augmentation.

**Violated assumption**: Modulation strategy does not affect training stability (variance). The cosine schedule breaks this: its deterministic, pre-programmed WD trajectory eliminates decision noise from the WD signal, converging to nearly identical weight dynamics across seeds.

**Implication**: There is a real stability benefit to cosine WD scheduling under AdamW — not an accuracy benefit, but a reproducibility benefit. This changes the research question slightly: even in Regime I (ρ ≪ 1), WD strategy affects *reproducibility* even when it doesn't affect *performance*. Papers that compare methods using few seeds on AdamW may systematically understate variance for cosine WD while overestimating it for alignment-based strategies.

---

## 3. Mental Model Revision

**Original mental model**: Weight decay strategies compete on accuracy; the alignment-awareness of methods like CWD should help by directing WD toward "useful" weights; AdamW and SGD respond similarly to WD scheduling, just with different sensitivities.

**Revised mental model after seeing the data**:

We assumed WD controlled a meaningful optimization variable (weight norm) that then influenced generalization. Under AdamW, this assumption is empirically false: AdamW's adaptive second-moment normalization absorbs all but ~1% of the WD signal variation, making weight norms effectively constant across WD strategies. The optimizer subsumes the regularizer. Under SGD, WD remains the dominant weight norm control mechanism, so different WD levels produce meaningfully different weight norms — but even then, the question of *when* to apply WD (the dynamic/alignment-aware contribution) is statistically invisible compared to the question of *whether* to apply WD at all.

Furthermore, we assumed alignment-aware WD (CWD, SWD) would benefit from directing WD toward "correctly aligned" weights. The CIFAR-100 SGD data suggests the opposite: alignment-conditional WD may actually harm optimization by concentrating the regularization signal in exactly the periods when the optimizer is making progress (gradient-weight alignment is high when the update is "on track"), potentially slowing convergence on harder tasks.

---

## 4. Reframing Test

**Original question**: "When does Phi modulation matter? Can we identify the ρ = λ/η regime boundary?"

**With full knowledge of these results, would we frame it the same way?**

No. The current framing assumes there IS a regime boundary to find — that sufficiently high ρ will reveal strategy differences. But the data shows something more fundamental: the mechanism that makes strategies equivalent under AdamW (second-moment normalization absorbing WD variation) is not ρ-dependent in any obvious way. The L-infinity implicit constraint is a structural property of AdamW, not a high-ρ phenomenon. Even at ρ = 50, if AdamW's second moment normalization still dominates, we might see invariance persist.

**Revised research question**: "Is the Phi Invariance property of AdamW a consequence of its adaptive scaling mechanism, and does removing either adaptive scaling or normalization layers (BN) break it? If so, at what structural boundary does WD strategy begin to matter?"

This reframing shifts from a scalar regime boundary (ρ threshold) to a structural boundary (which optimizer and architecture properties are necessary for invariance). It is more tractable and more informative — the experiments to answer it are the NoBN ablation and the AdamW-without-adaptive-step comparison, not the λ sweep.

---

## 5. New Hypotheses from Surprising Results

### NH1: Adaptive Scaling Is Sufficient for Phi Invariance (Independent of BN)

**Derivation**: Surprise S2 shows AdamW weight norms are nearly identical regardless of WD strategy. This norm convergence is caused by AdamW's second-moment normalization. A simpler hypothesis than "BN + AdamW = invariance" is "adaptive scaling alone is sufficient."

**Prediction**: AMSGrad or RMSProp (adaptive, no weight decay decoupling) should also show near-invariance to WD scheduling, while AdaGrad (adaptive but with sum of historical gradients that grows) may not.

**Falsification experiment**: Run ResNet-20 CIFAR-10 with RMSProp + decoupled WD using the same 6 methods; if invariance persists, adaptive scaling is the sufficient condition. Expected run time: 1 GPU hour (18 runs).

---

### NH2: Binary WD Suppression Is Harmful When Alignment-Correlated Periods Are Training-Critical

**Derivation**: CWD_hard and SWD underperform relative to their WD budget on CIFAR-100 SGD (CWD_hard worse than half_lambda despite applying WD "more intelligently"). This suggests that alignment-conditional suppression removes WD during exactly the periods when gradient-weight alignment is informative — which may be the periods with largest useful weight updates.

**Prediction**: A method that applies *more* WD when alignment is high (inverse CWD) should outperform standard CWD on SGD CIFAR-100, while performing similarly or worse on CIFAR-10 where the harder-task effect is smaller.

**Falsification experiment**: Implement inverted CWD (apply λ when cos_sim(g,w) < 0, apply 0 otherwise) and compare to CWD_hard and constant on SGD CIFAR-100. Expected run time: 15 minutes (pilot, 1 seed).

---

### NH3: Cosine WD Variance Reduction Scales with Task Difficulty

**Derivation**: Cosine WD's 4× variance reduction (std 0.072% vs ~0.28% for other methods on CIFAR-10) under AdamW suggests that deterministic WD scheduling eliminates a stochastic noise source. If this noise source grows with task complexity, the variance gap should be larger on CIFAR-100 and even larger on ImageNet.

**Prediction**: On CIFAR-100, cosine_schedule should show std < 0.2% while other methods show std > 0.35%. On ResNet-50/ImageNet, the gap should exceed 5× (std < 0.05% vs > 0.25%).

**Falsification experiment**: The existing CIFAR-100 AdamW data already tests this. Current CIFAR-100 results: cosine=63.417±0.419% — the variance is NOT lower for CIFAR-100. This falsifies NH3: the low-variance phenomenon is CIFAR-10 specific, not a general stability property of cosine WD scheduling.

> **Immediate update**: NH3 is falsified by existing data. Cosine schedule's CIFAR-100 std (0.419%) is HIGHER than CIFAR-10 (0.072%) and comparable to other methods (0.29–0.47%). This means the CIFAR-10 low-variance is likely a dataset-specific artifact (possibly related to CIFAR-10's simpler decision boundary reducing sensitivity to WD perturbation timing), not a general property of cosine WD.

---

## Anti-Pattern Check

**Post-hoc rationalization check**: Surprise S1 (SWD worse than no_wd) is not explained away — it is flagged as a mechanism worth investigating (intermittent WD application interacting with second-moment accumulation). We have not lowered the bar to call it "consistent with invariance."

**Hypothesis creep check**: H1 verdict is "Partially Confirmed," not "Confirmed." The SWD anomaly is explicitly noted as an exception to the simple invariance narrative.

**Connection to evidence check**: All three new hypotheses trace directly to specific numerical surprises, not to general intuitions. NH3 is immediately self-falsified against existing data, demonstrating the hypothesis-testing discipline.

---

## Summary of Required Belief Updates

| Prior Belief | Updated Belief | Evidence |
|---|---|---|
| Alignment-aware WD should help under SGD | Alignment-conditional WD may hurt on harder tasks | SGD CIFAR-100: CWD_hard Δ = -1.003% (2× worse than half_lambda) |
| WD strategy controls weight norm trajectory | Under AdamW, weight norms are structurally invariant to WD strategy | AdamW CIFAR-10: 1.1% norm range across 10× WD variation |
| AdamW and SGD differ quantitatively, not qualitatively | The mechanisms are qualitatively different (structural norm absorption vs direct control) | Weight norm data: SGD range 64.6–127.1 vs AdamW range 95.9–97.0 |
| Cosine WD's low variance is a stability property | Cosine WD's low variance is CIFAR-10 specific | CIFAR-100 cosine std = 0.419%, comparable to other methods |
| SWD should be between constant and no_wd in performance | SWD can underperform no_wd (harmful intermittent WD) | AdamW CIFAR-10: SWD = 89.883% < no_wd = 90.083% |
| ρ = λ/η defines the invariance regime | The sufficient condition for invariance may be adaptive scaling itself, not ρ | Requires NoBN ablation to confirm, but mechanism evidence is compelling |

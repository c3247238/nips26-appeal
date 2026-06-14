# Contrarian Perspective

**Topic**: Unified Dynamic Weight Decay Framework (Unified Dynamic WD: WD Scheduling, Alignment-Aware WD, Decoupled WD, Norm-Matched WD)
**Agent**: sibyl-contrarian
**Date**: 2026-03-18

---

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: The alignment signal (gradient-weight cosine similarity δ̂_t) is a reliable, informative, and low-noise signal for adapting WD in real training.**
   - Evidence challenging it:
     - SimiGrad (NeurIPS 2021) shows that for small batch sizes (≤ 64), gradient cosine similarity is essentially 0 due to gradient noise — the signal is dominated by minibatch variance, not the true geometry.
     - Gradient-Weight Alignment paper (arXiv:2510.25480) explicitly notes that "cosine similarity degrades in high dimensions" and restricts computation to the linear classifier head to avoid this. Full-network alignment proxies may be pure noise.
     - The CVPR 2025 paper (Sun et al.) acknowledges δ_t in theory requires full-data gradients; stochastic proxy δ̂_t inherits concentration inequalities only under strict assumptions that real training may violate.
     - Anti-correlated SGD noise (arXiv:2306.05300): standard i.i.d. noise assumptions underlying alignment proxy concentration arguments are violated in epoch-based training with data sampled without replacement.
   - Verdict: The core signal driving alignment-aware WD adaptation is far noisier and less reliable than typically assumed.

2. **Assumption: The four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) are meaningfully distinct mathematical objects requiring unification.**
   - Evidence challenging it:
     - D'Angelo et al. (NeurIPS 2024) argue all four reduce to the same underlying mechanism: WD as a training dynamics modifier controlling weight norm and effective learning rate. The "conceptual diversity" may be illusory.
     - Chou (arXiv:2512.08217) shows that norm-matched WD (γ² scaling) and decoupled WD converge to the same steady-state behavior; the "correction" is negligible empirically.
     - Defazio (arXiv:2506.02285) shows WD's dominant effect is driving ‖g‖/‖w‖ to a steady state across all normalized layers — this single mechanism already unifies scheduling, alignment, and decoupled WD's core practical effect.
   - Verdict: The proposed "unification" may discover that all four approaches are already effectively the same thing, producing a framework with zero explanatory gain.

3. **Assumption: Weight decay provides genuine regularization in modern architectures with BatchNorm/LayerNorm, justifying alignment-sensitive modulation.**
   - Evidence challenging it:
     - Van Laarhoven (arXiv:1706.05350): L2/WD has no regularizing effect when combined with normalization layers — it only adjusts the effective learning rate. This is the foundational result.
     - NeurIPS 2024 (D'Angelo et al.): "The effective learning rate interpretation only emerges as a consequence of scale-invariance and therefore does not apply to general architectures." Even the ELR interpretation has limited scope.
     - Robust Layerwise Scaling (arXiv:2510.15262): Non-affine normalizers (BatchNorm/LayerNorm/RMSNorm without bias) preserve forward-scale invariance but introduce backward scale sensitivity, creating complex width-dependent interactions.
     - If WD on BN/LN layers is just LR scaling, then "alignment-aware WD" on those layers is just "alignment-aware LR scheduling" — a much less novel claim.
   - Verdict: The proposed framework may be studying a phenomenon that doesn't exist in the form assumed for a large fraction of modern networks.

4. **Assumption: Alignment-aware dynamic WD meaningfully improves over well-tuned fixed WD baselines on standard benchmarks.**
   - Evidence challenging it:
     - NOVAK (arXiv:2601.07876) shows coupling WD with effective LR (α_eff) rather than base LR degrades generalization by 4-8pp on CIFAR-100 — suggesting alignment-based adaptation can hurt when miscalibrated.
     - SGDW failed to converge on VGG16/CIFAR-10 at high WD (5e-2) — extreme WD schedules can be catastrophic, not just suboptimal.
     - Choi et al. (2019, optimizer benchmarking): most claimed improvements of adaptive methods over well-tuned SGD vanish when hyperparameter budgets are equalized. The same confound likely applies to dynamic WD comparisons.
     - CWD (ICLR 2026): 20,000 H100 GPU hours for consistent but "modest" gains. No catastrophic failure cases, but the effect size is small — roughly 0.1-0.3 nats improvement in validation loss.
   - Verdict: Dynamic WD may provide marginal gains, not the substantial improvements the theoretical motivation suggests.

5. **Assumption: Budget Equivalence Metric, Coupling Stability Index, and Alignment Informativeness Score are well-defined, non-trivial metrics that will reveal genuine differences between WD methods.**
   - Evidence challenging it:
     - Fragility audit (arXiv:2510.18934): "Many generalization measures for deep learning are fragile" — popular post-mortem measures change qualitative predictions under mild hyperparameter tweaks. Proposed new metrics may suffer the same fate.
     - "Understanding and Scheduling Weight Decay" (OpenReview): "The optimal decoupled weight decay hyperparameter in AdamW can be very different from L2 regularization and SWD, meaning AdamW requires re-tuning the weight decay hyperparameter in practice." Any BEM metric must account for this — equating budgets is harder than it appears.
     - AIS (Alignment Informativeness Score): if the alignment signal is noisy (Challenge 1), the informativeness score may measure noise rather than signal, giving a falsely negative verdict for alignment-aware methods in certain regimes.
   - Verdict: The proposed metrics may be self-undermining — noisy input signals produce noisy metric outputs.

6. **Assumption: The Sun et al. (CVPR 2025) theoretical framework provides a stable foundation for extending to dynamic WD.**
   - Evidence challenging it:
     - Sun et al. prove WD does NOT accelerate convergence but improves generalization. This creates a fundamental tension: if dynamic WD is designed to preserve convergence speed while improving generalization, it must navigate a proven convergence-generalization trade-off. The theoretical tools for this may require fundamentally new approaches, not extensions.
     - The framework is limited to SGD. Extension to Adam/AdamW requires handling adaptive preconditioning, which the CVPR 2025 paper explicitly does not address. The convergence proof for AdamW (arXiv:2310.08858) shows AdamW minimizes a "dynamically regularized loss" rather than the original loss — qualitatively different behavior.
   - Verdict: Building on Sun et al. as a foundation for a unified framework risks inheriting fundamental limitations that preclude the claimed scope.

### Landscape of Doubt

The dominant narrative in this area is: "weight decay methods are fragmented; we need unification and standardized metrics." But the contrarian reading of the same evidence says:

1. The approaches may be fragmented because they target fundamentally different phenomena (scale-invariant architectures vs. non-scale-invariant; adaptive optimizers vs. SGD; LLMs vs. vision), and forcing them into one framework may produce a vacuous or trivially simple unification.

2. The alignment signal that motivates the most novel contributions (alignment-aware WD, the δ̂_t proxy) is much noisier in practice than in theory, potentially invalidating the practical value of the alignment modulation.

3. Every proposed "improvement" in this space involves tuning new hyperparameters (c, λ_min, λ_max, p, ε in the proposed dynamic rule). If these require as much tuning as fixed WD, the claim of improvement may only hold in the paper author's chosen hyperparameter neighborhood.

4. The "standardized evaluation metrics" are themselves new objects that could be gamed, fragile, or uninformative — reproducing the fragmentation problem at a meta-level.

---

## Phase 2: Initial Candidates

### Candidate A: The Alignment Signal Is Too Noisy to Drive WD Adaptation — A Controlled Empirical Falsification

- **Challenged assumption**: The minibatch alignment proxy δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε) is a reliable, low-variance signal that carries meaningful information about the gradient-weight geometry, sufficient to drive WD adaptation.
- **Evidence against it**: SimiGrad shows cosine similarity is near zero at small batch sizes. GWA paper (2510.25480) restricts to classifier head due to high-dimensional degradation. Anti-correlated SGD noise (2306.05300) invalidates i.i.d. concentration bounds. Sun et al. acknowledge the observability problem themselves.
- **Contrarian hypothesis**: In typical training regimes (batch size ≤ 512, deep ResNet/VGG, CIFAR-10/100), the variance of δ̂_t is so large that alignment-aware WD rules perform no better than randomly-modulated WD, and their apparent gains (when present) are attributable to implicit learning rate scheduling effects rather than alignment sensitivity.
- **Exploitation plan**:
  1. Train ResNet-20/CIFAR-10 and VGG-16-BN/CIFAR-100 with alignment-aware WD at batch sizes 64, 128, 512, 2048.
  2. At each step, record δ̂_t and its variance across multiple minibatches.
  3. Create a "noise-injected" control: replace δ̂_t with a random variable from the same empirical distribution (matching mean and variance but with no true signal).
  4. If the noise-injected version performs identically to the alignment-aware version, the alignment signal carries no information beyond its statistical moments.
  5. Additionally, test whether the effective WD schedule λ_t under the alignment rule is indistinguishable from a simple cosine schedule by Kolmogorov-Smirnov test on the λ_t trajectory distribution.
- **Novelty estimate**: 8/10 — directly falsifies a key assumption underlying alignment-aware WD without requiring new theory.

### Candidate B: Scale Invariance Degeneracy — WD Unification Is Only Needed for Parameters Without Normalization, Which Are a Shrinking Minority

- **Challenged assumption**: The proposed unified framework applies to the full network and all parameter types uniformly; alignment and norm-matching are meaningful for all parameters.
- **Evidence against it**: Van Laarhoven (1706.05350), NeurIPS 2024 (D'Angelo et al.): WD has no regularizing effect on scale-invariant (BN/LN) layers — only LR modulation. In modern ViT/ResNet-BN networks, the majority of parameters (weight matrices preceding BN/LN layers) fall in the scale-invariant regime. Kobayashi et al. (NeurIPS 2024): applying WD to attention layers induces nuclear norm regularization, which can damage performance.
- **Contrarian hypothesis**: The meaningful WD design space is much smaller than assumed. For scale-invariant parameters, WD optimization is just LR scheduling and adds no new freedom. Alignment-aware WD is only non-trivial for the small minority of parameters in non-scale-invariant positions (output layer, embedding layers, certain bias terms). A "unified framework" that ignores this is studying a mostly vacuous problem.
- **Exploitation plan**:
  1. For ResNet-20/VGG-16-BN (CIFAR), categorize all parameters into scale-invariant (pre-BN conv weights) and non-scale-invariant (BN parameters, final FC layer).
  2. Measure the actual proportion of parameters where WD has a genuine regularizing effect vs. pure LR scaling effect.
  3. Apply alignment-aware WD only to non-scale-invariant parameters; apply no WD (or pure LR correction) to scale-invariant parameters.
  4. Compare against the "unified framework" approach that applies alignment WD uniformly.
  5. Prediction: selective WD (based on scale-invariance classification) outperforms uniform application of alignment-aware WD.
- **Novelty estimate**: 7/10 — proposes a structural decomposition that would redefine the scope of the unified framework.

### Candidate C: The "Unification" Is Trivial — All Dynamic WD Approaches Reduce to ELR Scheduling

- **Challenged assumption**: The four sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) represent genuinely distinct mathematical mechanisms that require a non-trivial framework to unify.
- **Evidence against it**: D'Angelo et al. (NeurIPS 2024) show all WD effects reduce to: (a) norm stabilization → (b) ELR control → (c) stochastic noise calibration. Defazio (2506.02285) shows the ‖g‖/‖w‖ steady state is the dominant mechanism. CWD's sign alignment is binary LR masking in the sign update interpretation. AdamWN's target norm is equivalent to a particular LR schedule. If the unifying variable is simply "effective learning rate trajectory," the framework is a special case of known LR scheduling theory.
- **Contrarian hypothesis**: The proposed unified framework, when fully formalized, will reduce to a known class of adaptive LR schedules modulated by a noise signal. This would make the contribution a repackaging rather than a genuine unification.
- **Exploitation plan**:
  1. Derive the effective LR trajectory induced by each WD method (scheduling, alignment-aware, decoupled, norm-matched) under the Defazio-style ‖g‖/‖w‖ steady-state analysis.
  2. Compare these trajectories (KL divergence between schedule distributions, or time-series correlation) against standard LR schedules (cosine, linear, WSD, log-time).
  3. If alignment-aware WD produces an ELR trajectory indistinguishable from cosine annealing + noise, the "unification insight" is trivial.
  4. Use mathematical derivation to characterize the exact class of LR schedules each WD approach implicitly implements.
- **Novelty estimate**: 6/10 — if hypothesis is correct, this is a useful negative/clarifying result; if false, it strengthens the case for genuine unification.

---

## Phase 3: Self-Critique

### Against Candidate A (Alignment Signal Noise)

- **Steelman**: The SWD paper (NeurIPS 2023) uses gradient norms (not cosine similarity) for scheduling, and shows robust improvements across seeds. CWD uses sign alignment (binary, low-dimensional) rather than full cosine similarity — this is less susceptible to high-dimensional noise. The Sun et al. theory requires only that δ̂_t concentrates around its expectation in aggregate, not that each single step is accurate. With EMA smoothing, high-frequency noise could be averaged away. Papers by GWA (2510.25480) show that restricting alignment computation to the classifier head yields stable, informative signals.
- **Cherry-picking check**: I cited SimiGrad (batch size 4) as evidence for noise, but CWD's ICLR 2026 success was at large-scale LLM training where effective batch sizes are much larger (≥ 1024 tokens per step). The noise problem may only apply to small-batch vision training, not the regime where alignment-aware methods are primarily claimed to work.
- **Confounding check**: CWD's improvements could stem from its implicit effect on the loss landscape geometry (finding Pareto-optimal stationary points) rather than the alignment signal per se. The mechanism could be beneficial even if the alignment signal is noisy — as long as the binary masking has the right expected behavior.
- **Actionability check**: Even if the alignment signal is noisy, this finding would precisely characterize the regimes where alignment-aware WD works vs. fails, which is valuable. Proposing a noise-robust alignment proxy (EMA-smoothed, lower-dimensional) would be constructive.
- **Verdict**: MODERATE — the noise concern is real and underexplored, but the steelman case is strong for large-batch regimes. The finding would need to be scoped to specific training regimes to avoid a false universal claim.

### Against Candidate B (Scale Invariance Degeneracy)

- **Steelman**: D'Angelo et al. (NeurIPS 2024) explicitly argue WD's mechanism is NOT the same for SGD (loss stabilization via noise + WD) and for near-one-epoch LLM training (bias-variance tradeoff). Neither is the scale-invariance ELR argument — it's a third distinct mechanism specific to BN/LN architectures. The WD design problem may be architecture-specific but still requires a unified framework to handle all architectures. Additionally, Kobayashi et al. (NeurIPS 2024) show WD on attention layers induces nuclear norm regularization — this is a distinct, non-trivial effect beyond simple LR scaling, proving that WD has genuine structural effects even in scale-invariant layers.
- **Cherry-picking check**: I focused on Van Laarhoven's result for scale-invariant layers, but modern networks have a mix of scale-invariant (BN/LN-preceded) and non-scale-invariant layers. For ResNets, the final FC layer and early conv layers can be genuinely regularized by WD. For transformers, embedding layers and final projections are non-scale-invariant.
- **Confounding check**: Even if WD = ELR scaling for BN-preceded layers, the specific ELR trajectory matters for generalization. Two networks with the same final accuracy can have different representational quality if their ELR trajectories differ.
- **Actionability check**: Identifying "which parameters actually need alignment-aware WD" is highly actionable — it would reduce computational overhead of alignment tracking to only the relevant parameters.
- **Verdict**: MODERATE — the core insight about scale invariance is valid but the conclusion (framework is vacuous) overstates the case. A more precise version: "alignment-aware WD is only non-trivial for non-scale-invariant parameters, and the unified framework should explicitly partition the parameter space."

### Against Candidate C (Trivial Unification)

- **Steelman**: LR scheduling theory operates in the global learning rate space; WD sub-approaches operate in the parameter-specific, geometry-aware, norm-constrained space. Even if the net effect on ‖g‖/‖w‖ is similar, the mechanism matters: two identical ‖g‖/‖w‖ trajectories could result from alignment-aware WD (which conditions on gradient direction) and a scalar LR schedule (which does not), leading to different per-parameter behavior and different generalization properties on non-spherically-distributed problems. CWD's Pareto-optimality interpretation is a property of WD, not of LR scheduling — you cannot replicate the "search for locally Pareto-optimal stationary points" with LR scheduling alone.
- **Cherry-picking check**: Defazio (2506.02285) analyzes the steady-state aggregate ‖g‖/‖w‖ ratio, but dynamic WD methods operate at each step based on current geometry. The steady-state analysis averages away the within-trajectory dynamics where alignment information would have the most impact.
- **Confounding check**: The ELR equivalence may hold for the final solution quality but not for the path to convergence. Alignment-aware WD might reach better local minima faster even if the endpoint ELR is the same.
- **Actionability check**: Even a negative result (proving Candidate C is correct) would be highly publishable and important — it would redirect the community from studying "better alignment signals" to studying "better effective LR trajectories."
- **Verdict**: WEAK-to-MODERATE — the CWD Pareto-optimality argument provides a clean counter-example showing WD has properties not replicated by LR scheduling. The trivial unification hypothesis is likely false, but the "partially equivalent" version is worth formalizing precisely.

---

## Phase 4: Refinement

### Dropped: Candidate C
The steelman case is strong enough that "trivial unification" is likely false. CWD's Pareto-optimality interpretation, the nuclear norm induction in attention layers, and the structural low-rank bias are all properties genuinely absent from scalar LR scheduling. The more interesting version of this critique is: "the unified framework's mathematical formulation, λ(t, w, g) = f(alignment, norm, schedule, target_norm), can be simplified to a lower-dimensional parametrization without losing any empirically significant degrees of freedom." This is a refinement critique, not a falsification.

### Strengthened: Candidate A (Front-Runner)

After steelmanning, the noise concern is real but scoped. The refined, strengthened version:

**Precise claim**: For typical vision training regimes (CIFAR-10/100, ResNet/VGG, batch size ≤ 256 — exactly the experimental regime proposed in this workspace's spec), the minibatch alignment proxy δ̂_t has variance-to-signal ratio >> 1 at most training steps. Alignment-aware WD under these conditions performs no better than a weight-norm-triggered schedule: λ_t = f(‖w_t‖) rather than λ_t = f(δ̂_t, ‖w_t‖). The alignment component is noise-corrupted to the point of being uninformative.

**Constructive proposal**: Rather than abandoning alignment-aware WD, this finding would justify:
1. EMA-smoothed alignment proxy (averaging over k steps, k ≥ 10) as a precondition for any alignment-aware WD method
2. Restricting alignment computation to a low-dimensional subspace (e.g., layer-wise alignment after projecting to top-k singular value space) to reduce noise
3. A theoretically principled noise budget: alignment-aware WD should switch to a fixed schedule when the minibatch estimate's noise-to-signal ratio exceeds a threshold

**Additional corroboration found**: The GWA paper (2510.25480) explicitly notes "cosine similarity degradation in high dimensions" and restricts computation to the classifier head. The SimiGrad paper shows that for batch size 4-64, cosine similarity is "almost always 0." In the proposed spec, batch size is 128 — right in the danger zone.

### Strengthened: Candidate B (Secondary Focus)

The scale invariance argument is sharpened: not "the framework is vacuous" but "the framework is not uniform — it has two qualitatively different operating regimes that require separate analysis." This becomes a positive contribution: parameterize the unified framework with a regime indicator (scale-invariant vs. regularization-active), and show experimentally that cross-contamination (applying alignment WD uniformly) degrades performance on scale-invariant layers relative to a layer-type-aware strategy.

**Selected front-runner**: Candidate A (noise-signal analysis of alignment proxy), because:
1. It is directly falsifiable with the experimental setup already planned (ResNet-20/VGG-16-BN, CIFAR-10/100)
2. It produces a constructive proposal (noise-robust alignment proxy) that adds value regardless of outcome
3. It targets the single most novel and unverified assumption in the entire framework
4. The experiments fit within the 1-hour budget per task for CIFAR (batch size 128, ResNet-20, 200 epochs ≈ 25 minutes)

---

## Phase 5: Final Proposal

### Title
**"When Alignment Signals Are Just Noise: Empirical Bounds on the Informativeness of Gradient-Weight Alignment for Weight Decay Adaptation in Finite-Batch Training"**

*(Or, constructively framed: "Noise-Robust Alignment-Aware Weight Decay: When and How the Alignment Signal Actually Helps")*

### Challenged Assumption
The minibatch alignment proxy δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε), which drives alignment-aware dynamic WD adaptations, carries reliable, low-variance geometric information about the gradient-weight relationship at each training step in typical deep learning training regimes.

### Evidence For and Against

**For the mainstream assumption** (alignment signal is reliable):
- Sun et al. (CVPR 2025) prove theoretical convergence bounds depending on aggregate alignment (cumulative Σ δ_t) — only average behavior needs to be correct, not each step.
- CWD (ICLR 2026) uses binary sign alignment which is more noise-robust than continuous cosine similarity.
- GWA (2510.25480) shows that with appropriate computational focus (classifier head, EMA smoothing), alignment signals are stable and informative.
- With EMA smoothing (k ≥ 10), i.i.d. noise variance drops by 1/√k.

**Against the mainstream assumption** (alignment signal is unreliable):
- SimiGrad: gradient cosine similarity ≈ 0 for batch size ≤ 64; near 0 for batch size = 128 at many training steps.
- High-dimensional degradation: in a D-dimensional space, random vectors have cosine similarity O(1/√D). For D = 10^5 - 10^7 (typical ResNet layer), the signal-to-noise ratio is extremely low without dimensionality reduction.
- Anti-correlated SGD noise (2306.05300): epoch-based training violates the i.i.d. assumption underlying concentration inequalities for δ̂_t.
- No existing paper has measured the empirical distribution of δ̂_t variance across training steps under realistic conditions for CIFAR-scale tasks.

### Hypothesis
For vision classification with batch size ≤ 256 (the typical regime for CIFAR-10/100 experiments), the single-step minibatch alignment proxy δ̂_t has a coefficient of variation (CV = std/mean) greater than 1 for most of the training trajectory, making it uninformative as a real-time WD adaptor. Consequently:
1. Alignment-aware WD rules using raw δ̂_t perform no better than noise-matched random modulation with the same mean and variance.
2. However, EMA-smoothed alignment (k ≥ 10 steps) significantly reduces CV below 1 and yields statistically significant improvements over fixed WD baselines.
3. This implies the true informativeness of the alignment signal requires temporal aggregation, not single-step adaptation — contradicting the computational motivation for real-time alignment adaptation.

### Method

**Experiment 1: Characterize δ̂_t variance** (diagnostic, ~10 minutes per run)
- Train ResNet-20 on CIFAR-10 with fixed WD=5e-4, batch size in {64, 128, 512, 2048}
- At each training step, record: δ̂_t (mean over all layers), per-layer δ̂_t, variance across 10 repeated minibatch draws from the same batch
- Compute CV(δ̂_t) = std/mean across training, and the autocorrelation structure
- Expected finding: CV >> 1 for batch size ≤ 256, CV < 1 for batch size ≥ 1024

**Experiment 2: Noise injection test** (~25 minutes per configuration)
- Four conditions for λ_t rule on ResNet-20/CIFAR-10, ResNet-20/CIFAR-100:
  a. Fixed WD: λ_t = λ (baseline)
  b. Alignment-aware WD: λ_t = clip(c * γ_t * (1 - δ̂_t), λ_min, λ_max)
  c. Noise-injected control: λ_t = clip(c * γ_t * (1 - ε_t), λ_min, λ_max) where ε_t ~ empirical distribution of δ̂_t, but with shuffled time-indices (no real signal, same distribution)
  d. EMA-smoothed alignment: λ_t = clip(c * γ_t * (1 - EMA_k(δ̂_t)), λ_min, λ_max), k = 10, 50

- Primary comparison: if (b) approximately equals (c) statistically (over 3 seeds), the alignment signal is uninformative. If (b) >> (c), alignment carries genuine information beyond its distribution.

**Experiment 3: Norm-based surrogate comparison** (~15 minutes per configuration)
- Compare alignment-aware WD against a norm-aware schedule: λ_t = clip(c * γ_t * (1 - ‖w_t‖/‖w_0‖), λ_min, λ_max)
- If the weight norm already captures most of the alignment signal's benefit, the additional alignment computation is redundant.

### Baselines
- Fixed SGDW (well-tuned, grid-searched λ in {1e-4, 5e-4, 1e-3, 5e-3})
- Cosine WD schedule tied to LR schedule (standard practice)
- SWD/AdamS (gradient-norm-aware baseline)
- CWD variant (sign alignment — more noise-robust comparison)

All baselines receive equal hyperparameter tuning budget (grid search over same parameter space as dynamic methods). This directly addresses the confound identified in Choi et al. (2019).

### Experimental Plan
- Architecture: ResNet-20, VGG-16-BN
- Dataset: CIFAR-10, CIFAR-100
- Batch size: 128 (primary), 512, 2048 (scale analysis)
- Seeds: 42, 123, 456
- Time budget: Experiments 1+2+3 approximately 3-4 hours total on a single RTX PRO 6000
- Compute: local (single GPU per experiment)

### Risk Assessment
**What if the mainstream view turns out to be correct?**

Scenario A: δ̂_t is reliable even at batch size 128. This would be a significant positive finding for alignment-aware WD — the worry is unfounded. The contribution becomes: "we empirically verified that alignment signals are reliably informative even at small batch sizes, providing justification for alignment-aware WD adaptation." This is still publishable as an important validation.

Scenario B: δ̂_t is noisy at batch size 128 but EMA-smoothing at k ≥ 10 recovers reliability. This confirms the constructive proposal: single-step adaptation is unreliable, aggregate adaptation works. This is a nuanced and publishable finding that improves the practical design of alignment-aware WD.

Scenario C: δ̂_t is noisy but the alignment-aware rule still outperforms noise-injected control. This suggests the benefit is not from the alignment information per se, but from the dynamic schedule's implicit structure (e.g., correlation with training phase). This would identify the true mechanism and suggest simpler alternatives.

### Novelty Claim
The specific insight: **single-step minibatch alignment is unreliable as a WD adaptation signal in typical training regimes, but temporally-aggregated alignment (EMA ≥ 10 steps) is informative.** This defines a previously uncharacterized "minimum aggregation horizon" for alignment-based WD adaptation — a concept absent from all existing alignment-aware WD papers (CWD, AdamO, SPD). This would also reveal whether the Sun et al. theory's reliance on worst-case alignment rather than step-wise alignment is not just a theoretical convenience but an empirical necessity.

### Connection to the Unified Framework
This contrarian finding is directly constructive for the proposed Unified Dynamic WD Framework:
- It would constrain the design space of λ(t, w, g): raw single-step alignment should not be a direct input; only temporally-aggregated alignment qualifies.
- It would motivate a new metric: "Alignment Aggregation Horizon" (AAH) — the minimum smoothing window over which the alignment proxy becomes informative (CV < 1). This is more actionable than the vaguely-defined Alignment Informativeness Score.
- It would provide a direct criterion for when to use alignment-aware WD (large batch, or EMA smoothed) vs. simpler alternatives (small batch, short training).

---

## Summary Table

| Candidate | Core Challenge | Phase 3 Verdict | Status |
|-----------|---------------|-----------------|--------|
| A: Alignment signal noise | δ̂_t is unreliable noise at batch ≤ 256 | MODERATE | **Front-runner** (refined) |
| B: Scale invariance degeneracy | Most params do not need alignment-aware WD | MODERATE | Secondary insight (embedded in proposal) |
| C: Trivial unification | Reduces to LR scheduling | WEAK-MODERATE | Dropped; partially absorbed |

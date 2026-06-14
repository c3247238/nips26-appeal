# Contrarian Perspective

**Agent**: Contrarian (Devil's Advocate)
**Topic**: Unified Dynamic Weight Decay Framework
**Date**: 2026-03-18 (Refreshed with additional evidence 2026-03-18)

---

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption**: "The gradient-weight cosine alignment δ̂_t provides a reliable, informative signal for dynamically adjusting weight decay strength in practical stochastic training."
   - **Evidence challenging it**: The project's own risk assessment acknowledges "δ̂_t 噪声太大导致算法不稳定" (δ̂_t noise too large for stability). SimiGrad (NeurIPS 2021) showed cosine similarity estimates are "almost always 0" at small batch sizes due to gradient noise. Even at standard batch sizes (b=128), for large parameter tensors (e.g., 512×512 matrices), the variance of a minibatch cosine similarity estimate scales as O(1/sqrt(b·d)) — potentially comparable to the true signal.
   - **Key papers**: [SimiGrad NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/abea47ba24142ed16b7d8fbf2c740e0d-Paper.pdf), [Anti-correlated noise in SGD](https://arxiv.org/abs/2306.05300)

2. **Assumption**: "Alignment-aware weight decay research on Batch Normalization architectures (ResNet-20+BN, VGG-16-BN) provides valid evidence for the alignment mechanism."
   - **Evidence challenging it**: In BN architectures, the loss function is approximately scale-invariant with respect to pre-BN weights, meaning the gradient is approximately orthogonal to the weight vector for BN-preceding layers. This structurally suppresses δ̂_t ≈ 0 for most parameters — not because alignment is uninformative, but because BN erases the radial gradient component before it reaches the weight update. Chou (arXiv:2512.08217) explicitly tested alignment/rotation-balanced AdamW variants on ImageNet with ViT-S/16 for 90 epochs and found "almost no differences in relevant metrics."
   - **Key papers**: [Jane Street: L2 Regularization and Batch Norm](https://blog.janestreet.com/l2-regularization-and-batch-norm/), [Correction of Decoupled WD (Chou 2025)](https://arxiv.org/abs/2512.08217), [Kobayashi et al. 2024 BN-WD interaction](https://arxiv.org/abs/2410.23819)

3. **Assumption**: "The four WD sub-fields (scheduling, alignment-aware, decoupled, norm-matched) are fundamentally distinct methods requiring unification through a common framework."
   - **Evidence challenging it**: D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) argue that *all* these methods work via the same root mechanism — modifying training dynamics rather than providing explicit regularization. If they share one root cause, unifying them may produce a taxonomy rather than a theory. Meanwhile, the Ye (arXiv:2410.00232) preconditioning framework already proposed a unified view of AdamW, L1-regularization analogues, and normalization methods, yet produced no new methods or resolved practical questions.
   - **Key papers**: [Why Do We Need WD (D'Angelo 2024)](https://arxiv.org/abs/2310.04415), [Preconditioning for Optimization and Regularization (Ye 2024)](https://arxiv.org/abs/2410.00232)

4. **Assumption**: "CWD, AdamWN, SWD, and decoupled WD are genuinely independent 'special cases' of a single optimization principle."
   - **Evidence challenging it**: These methods have incompatible mathematical foundations. CWD uses a binary sign mask from a bilevel Pareto interpretation. AdamWN uses target-norm control (Lagrangian relaxation). SWD uses gradient-norm scheduling (empirical heuristic). A sufficiently general formula lambda(t, w, g) = f(...) can algebraically subsume any method — this is vacuously true. The Lipton & Steinhardt "Troubling Trends in ML" explicitly warns against papers whose contribution is "providing a general framework" without new predictive power.
   - **Key papers**: [CWD ICLR 2026](https://arxiv.org/abs/2510.12402), [Weight Norm Control (Loshchilov 2023)](https://arxiv.org/abs/2311.11446)

5. **Assumption**: "Sun et al.'s (CVPR 2025) theoretical framework for nonconvex SGDW can be cleanly extended to alignment-aware, time-varying λ_t."
   - **Evidence challenging it**: Sun et al. explicitly prove WD does NOT accelerate convergence in nonconvex SGD. Their proof structure requires static alignment conditions (worst-case δ_T). When λ_t depends on δ̂_t (which depends on g_t), λ_t becomes a random process with complex dependence on past iterates via the filtration, invalidating clean Lyapunov arguments. This is listed as a "known technical difficulty" in the project spec but may be a fundamental barrier to useful theoretical guarantees, not just an engineering challenge.
   - **Key paper**: [CVPR 2025 WD Nonconvex SGD (Sun et al.)](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Investigating_the_Role_of_Weight_Decay_in_Enhancing_Nonconvex_SGD_CVPR_2025_paper.pdf)

6. **Assumption**: "Standardized evaluation metrics (BEM, CSI, AIS) will reduce the field's fragmentation problem and enable fair comparisons."
   - **Evidence challenging it**: The field already has multiple competing diagnostic tools: gradient norms, weight norms, OUI indicator (arXiv:2504.17160), spectral density (Yunis et al. 2024), gradient-to-weight ratio (Defazio 2025), and gradient-weight alignment (arXiv:2510.25480). Adding three more metrics without rigorous statistical validation of their predictive power risks "metric proliferation" — more opportunity for cherry-picking, not less fragmentation. The AIS metric specifically could be undefined or trivially near-zero in normalized architectures (see Assumption 2), making it meaningless on standard benchmarks.
   - **Key papers**: [OUI metric (2025)](https://arxiv.org/abs/2504.17160), [GWA proxy (2025)](https://arxiv.org/abs/2510.25480), [Defazio gradient-to-weight (2025)](https://arxiv.org/abs/2506.02285)

7. **Assumption**: "The alignment informativeness of δ̂_t provides something beyond what CWD's binary sign mask already captures."
   - **Evidence challenging it**: CWD's ablation comparing its sign-alignment mask against a *random mask* of equal weight shows substantial performance gaps (AdamW loss 2.56 CWD vs 2.82 random mask). However, the paper does not compare against a simpler baseline: constant WD scaled by the average masking fraction. If CWD's benefit comes from reducing the *amount* of decay applied rather than from alignment-specific conditioning, then continuous δ̂_t adds no value over CWD — and continuous modulation adds noise without benefit.
   - **Key paper**: [CWD ICLR 2026 ablations](https://openreview.net/pdf/46912f6bea26cf709d99b0df502e307fb4654c93.pdf)

8. **Assumption**: "WD over normalized networks provides a domain with rich alignment dynamics worth studying."
   - **Evidence challenging it**: Recent work on "Gradient-to-weight ratio" (Defazio 2025) shows WD drives all normalized layers to a *single steady-state* ‖g‖/‖w‖ ratio — meaning under steady state, the radial dynamics are governed entirely by a scalar ratio, not by a high-dimensional alignment geometry. If steady-state dominates most of training, dynamic alignment modulation is only relevant transiently during the initial phase when the system has not yet reached its steady state.
   - **Key paper**: [Why Gradients Rapidly Increase Near Training End (Defazio 2025)](https://arxiv.org/abs/2506.02285)

9. **Assumption**: "Cosine similarity between gradient and weight vectors (δ̂_t) is a well-behaved quantity that can be reliably estimated from a minibatch in high-dimensional parameter spaces."
   - **Evidence challenging it**: The GWA paper (arXiv:2510.25480) explicitly states that "the expected cosine similarity between vectors with independent noise components *diminishes rapidly with increasing dimensions*" — making cosine similarity degradation in high dimensions a primary obstacle, not an edge case. For ResNet convolutional layers with d = 36,864 parameters, cosine similarity estimates are dominated by this degradation. The paper resolves this by restricting measurement to the linear classifier head — but the dynamic WD framework needs per-tensor estimates for all layers, not just the head. This is a structural obstacle that no existing alignment-aware WD paper addresses.
   - **Key paper**: [Gradient-Weight Alignment as Train-Time Proxy (arXiv:2510.25480)](https://arxiv.org/abs/2510.25480)

10. **Assumption**: "WD benefits generalize across training regimes (over-training on vision, under-training on LLMs)."
    - **Evidence challenging it**: D'Angelo et al. (NeurIPS 2024) explicitly show WD operates via fundamentally different mechanisms in different regimes. For SGD on vision (over-training): loss stabilization mechanism. For Adam on LLMs (under-training): bias-variance tradeoff. PNAS 2024 "Different Regimes of SGD" shows "adding weight decay in the noise-dominated regime is akin to stopping the dynamics early, thus *lowering* performance in the case of the perceptron with no label noise." A single unified framework that treats WD as one phenomenon may be describing two different mechanisms with incompatible math.
    - **Key papers**: [Why Do We Need WD (D'Angelo NeurIPS 2024)](https://arxiv.org/abs/2310.04415), [On Different Regimes of SGD (PNAS 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10907278/)

### Landscape of Doubt

**Four structural problems undermine the proposed framework:**

1. **Signal suppression + high-dimensional degradation**: The key signal (δ̂_t) is structurally suppressed in BN architectures (scale-invariance kills the radial gradient component), noisy at practical batch sizes due to the cosine similarity degradation in high dimensions (rigorously documented in arXiv:2510.25480 — expected value diminishes rapidly with parameter dimension), and degenerate in normalized steady-state training (all layers converge to the same ‖g‖/‖w‖ ratio). The AIS metric may be measuring noise or architecture artifacts, not genuine alignment information.

2. **Theoretical incompatibility**: Time-varying λ_t that depends on δ̂_t creates a stochastic process with non-trivial filtration structure that invalidates the clean Lyapunov convergence arguments used in Sun et al. (CVPR 2025). Proving useful guarantees for fully adaptive alignment-aware WD in nonconvex stochastic settings may be harder than the project anticipates — and the simpler version (λ_t = schedule(t) only) may already be addressed by existing theory.

3. **Vacuous unification**: Writing existing methods as special cases of lambda(t, w, g) = f(...) is algebraically trivially achievable. Scientific unification requires the formula to have predictive power — to say which instantiation will outperform others in a new setting. Without this, the "unified framework" is a naming exercise.

4. **Regime fragmentation**: D'Angelo et al. (NeurIPS 2024) demonstrate that WD's mechanism is fundamentally regime-dependent — loss stabilization for SGD on vision, bias-variance tradeoff for Adam on LLMs, and potentially *harmful* in the noise-dominated SGD regime. A unified framework that uses a single parameterization across regimes may be misdescribing two different physical phenomena under one mathematical notation, without capturing the actual mechanism in either regime.

---

## Phase 2: Initial Candidates

### Candidate A: "The Continuous Alignment Signal Is Systematically Suppressed in Normalized Networks"

- **Challenged assumption**: The gradient-weight cosine alignment δ̂_t provides a reliable continuous signal for WD modulation in standard BN-based benchmarks.
- **Evidence against**: (1) Scale-invariance of BN makes pre-BN layers' weight gradient approximately orthogonal to the weight vector, suppressing δ̂_t ≈ 0 for most parameters. (2) Chou (2512.08217) found no differences from alignment-balanced variants on ImageNet/ViT. (3) Kobayashi et al. (2410.23819) show L2 on attention multiplicative parameters (post-LN parameters) is equivalent to nuclear norm and damages performance — alignment-aware WD that applies higher decay when δ̂_t is low could amplify this damage. (4) Project's own acknowledged risk: "δ̂_t 噪声太大".
- **Contrarian hypothesis**: For BN architectures, the continuous δ̂_t signal is suppressed or noise-dominated for most parameters. Any gain from "continuous" alignment-aware WD over CWD's binary mask will be small or indistinguishable from noise. The project should run alignment diagnostics on normalization-free architectures to even establish that the phenomenon exists.
- **Exploitation plan**: 30-minute diagnostic on CIFAR-10: log per-layer δ̂_t distributions for ResNet-20+BN vs ResNet-20-NF. If BN suppresses the signal to near-zero, the entire experimental plan needs redesigning.
- **Novelty estimate**: 8/10 — raises a specific, testable confound not addressed by any existing paper on alignment-aware WD.

### Candidate B: "Alignment Informativeness Improvements Over CWD Are Due to Effective WD Reduction, Not Alignment Information"

- **Challenged assumption**: The improvement of continuous alignment-modulated WD over CWD comes from using alignment information, not from reducing the effective amount of decay applied.
- **Evidence against**: CWD's ablation does not compare against a "constant WD × 0.5" baseline that applies the same average fraction of decay without alignment conditioning. The SWD paper found benefits from any strategy that reduces WD when gradients are large — not specifically because of alignment. The "amount of decay" and "alignment conditioning of decay" are not properly disentangled in any existing paper.
- **Contrarian hypothesis**: The improvements from alignment-aware WD collapse when compared against a properly tuned constant-reduced-WD baseline. In that case, the alignment signal provides no incremental value — only the *average* decay reduction matters.
- **Exploitation plan**: Run CWD vs. (λ × mask_fraction_of_CWD) on CIFAR-10. If they match, alignment information adds no value — only the reduction in total WD.
- **Novelty estimate**: 7/10 — directly tests whether AIS = 0 for all continuous alignment methods.

### Candidate C: "The SGDW Convergence Theory Extension to Dynamic λ_t Is Fundamentally Blocked by Filtration Problems"

- **Challenged assumption**: The theoretical goal of proving useful convergence/generalization guarantees for alignment-aware λ_t = f(δ̂_t) in nonconvex stochastic SGD is achievable with standard tools.
- **Evidence against**: Sun et al. (CVPR 2025) prove WD does NOT improve convergence speed — it only affects generalization via a stability mechanism. Their proof uses static λ with worst-case δ_T. When λ_t depends on δ̂_t = f(g_t, w_t), λ_t is a random variable adapted to the filtration F_t. This creates: (i) λ_t appearing on both sides of the stability recursion Π_s(1 - λ_s), making closed-form analysis impossible without martingale tools; (ii) Concentration inequalities for δ̂_t requiring assumptions about gradient distribution that are not standard in nonconvex theory; (iii) The key theorem target "Target D" in the project spec explicitly requires concentration that may not hold in the heavy-tailed gradient noise regime recently documented in practice.
- **Contrarian hypothesis**: The theoretically interesting version of alignment-aware WD (fully adaptive λ_t = f(δ̂_t)) has no tight convergence guarantees in the nonconvex stochastic setting. The only provable results will be for simplified variants (stagewise schedule, deterministic δ_t) that do not actually use alignment information.
- **Exploitation plan**: Write down the proof attempt for Target D (minibatch proxy concentration) explicitly and identify the blocking step. If the blocking step requires O(1/√T)-level concentration for δ̂_t with T training steps, check whether that concentration bound is achievable in typical deep learning gradient regimes.
- **Novelty estimate**: 6/10 — technical but important for scoping the paper's theoretical contribution.

---

## Phase 3: Self-Critique

### Against Candidate A (BN Signal Suppression)

- **Steelman**: Defazio (arXiv:2506.02285) provides a strong counter: even in normalized networks, WD drives gradient-to-weight ratio ‖g‖/‖w‖ to a steady state. This is not LR scaling — it is a specific dynamic that WD controls. In this view, the radial dynamics do carry real information, and δ̂_t may vary meaningfully when the system is far from steady state (early and middle training). The GWA paper (arXiv:2510.25480) further shows that gradient-weight alignment is *more stable* than pairwise gradient alignment at standard batch sizes, with "fewer fluctuations" — directly countering the noise claim.
- **Cherry-picking check**: The SimiGrad failure at b=4 is extreme. At b=128, the cosine similarity estimate has much lower variance. The BN orthogonality claim applies to coupled L2 regularization — for decoupled WD (AdamW), the scale-invariance properties are different because the WD step is multiplicative, not additive.
- **Confounding check**: Even if δ̂_t is near-zero on average for BN-adjacent layers, it may vary systematically during training phases (high early in training when norms are not yet equilibrated, low when near steady state). This temporal variation could still carry useful information for WD scheduling.
- **Actionability check**: Experiment 1 (30 minutes, CIFAR-10) directly tests whether BN suppresses δ̂_t. This is a clean, fast diagnostic that can either validate or refute the entire critique.
- **Verdict**: STRONG. The BN suppression effect is real and documented, but it is partial and architecture-specific. The key refinement: the critique is strongest for *continuous* δ̂_t modulation (the project's proposal), less strong for sign-level modulation (CWD). This is the correct targeting.

### Against Candidate B (Effective WD Reduction Confound)

- **Steelman**: CWD's ablation comparing the sign mask to a random mask of equal expected application rate shows substantial performance gaps (2.56 vs 2.82). This rules out the "amount of decay" explanation for most of the CWD gain. Furthermore, the literature survey papers (SWD, ADANA) show that timing-aware WD scheduling provides benefits beyond pure amount reduction — consistent with alignment information mattering.
- **Cherry-picking check**: The experiment required to test Candidate B (CWD vs. constant λ × mask_fraction) is not in the existing literature. This is actually an unexplored but important control.
- **Confounding check**: The confound between "alignment information" and "effective decay reduction" is a genuine theoretical confound. CWD's ablations do not fully disentangle these.
- **Actionability check**: High — requires a single well-controlled experiment that is fast (CIFAR-10, ResNet-20).
- **Verdict**: MODERATE. The claim cannot be steelmanned away by existing ablations because the exact control experiment is missing. This is an open empirical question, but less likely to be fatal than Candidate A given CWD's existing ablations.

### Against Candidate C (Filtration Barrier)

- **Steelman**: The project's spec explicitly targets stagewise-schedule proofs first (Target A, B) before fully adaptive rules. Stagewise λ_t is deterministic given the schedule, avoiding the filtration problem entirely. A paper that proves convergence for stagewise-aligned WD and shows empirical results for adaptive WD can succeed even if the fully adaptive theory is intractable.
- **Cherry-picking check**: I emphasized the hardest case (fully adaptive λ_t = f(δ̂_t)). The project's spec prudently acknowledges this and plans the "conservative" Target A first.
- **Confounding check**: Martingale tools can in principle handle random λ_t if mild independence conditions hold (e.g., λ_t depends only on g_t and not on past λ_s). This is achievable if the adaptive rule uses only current-step information.
- **Actionability check**: Requires actual theoretical work to verify — cannot be done in a short experiment.
- **Verdict**: WEAK as a "fatal" critique, MODERATE as a "scoping" critique. The filtration problem exists but is manageable if the project focuses on stagewise theory rather than fully adaptive theory. The critique is useful for scoping, not for rejection.

---

## Phase 4: Refinement

**Dropped**: Candidate C (Filtration Barrier) as a fatal critique. It is a useful scoping concern — the theory should stay focused on stagewise and deterministic-schedule variants — but it does not invalidate the project.

**Strengthened from Candidate A**: The BN signal suppression critique, with precise formulation:

For layers in ResNet-20+BN (the primary experimental testbed), the parameter tensors are organized as: [Conv weights → BN scale/shift → ReLU → Conv weights → BN → ...]. The BN scale parameters (γ, β) are *not* subject to the radial suppression (they are 1D scalars, not high-dimensional weight tensors). However, the convolutional weight matrices *before* each BN layer are subject to scale-invariance. For a ResNet-20 with 3×3×64×64 convolutional kernels, the parameter tensor has d = 36,864 dimensions. The minibatch cosine similarity variance at b=128 is approximately σ ≈ √(1/128·36864) ≈ 0.0015 — very small. But the true δ̂_t for these layers may be similarly near-zero in expectation due to scale-invariance, making the signal-to-noise ratio potentially order-1 or worse.

The refined claim: **for the high-dimensional convolutional layers in BN architectures (which contain >95% of the network's parameters), the expected δ̂_t ≈ 0 due to BN scale-invariance, making the AIS metric undefined (0/0 form) on these benchmarks**. This is not a noise problem — it is a structural problem.

**Strengthened from Candidate B**: The "effective WD reduction" confound is retained as a secondary critique. The specific control experiment needed: compare CWD against (a) standard AdamW with λ' = λ × E[mask_CWD] and (b) standard AdamW with λ' found by grid search. If (a) matches CWD, alignment information provides no value over amount reduction. If (b) matches CWD, alignment information provides no value over better hyperparameter tuning.

**Additional corroboration found**:
- Weight decay induces neural collapse and low-rank bias in wide networks (ICLR 2025, arXiv:2410.04887). This means WD's effects are non-linear and architecture-dependent — the same λ can drive one architecture toward low-rank solutions while having minimal effect on another. A "unified framework" that uses a single scalar λ(t, w, g) ignores this architectural heterogeneity.
- The PNAS 2024 "different regimes of SGD" paper shows that WD's effect on alignment dynamics is *regime-dependent*: "for large learning rates, the dynamics of alignment is different in the small batch and large batch regimes." This suggests a unified framework must condition on training regime, not just instantaneous alignment.
- GWA (arXiv:2510.25480) measures *per-sample* gradient-weight alignment (not per-parameter), and found it more stable. But their metric aggregates across all samples, not across parameter dimensions — this is a different quantity than δ̂_t in the project, where alignment is per-parameter-tensor, not per-sample.

**Selected front-runner**: Candidate A, refined — "**The Continuous Alignment Signal Is Structurally Suppressed in Standard BN Benchmarks, Making the AIS Metric Uninterpretable Without Controlled Experiments on Normalization-Free Architectures**"

---

## Phase 5: Final Proposal

### Title: "When Alignment Is Silent: Batch Normalization Suppresses the Radial Gradient Signal That Motivates Continuous WD Modulation"

**Challenged assumption**:
The gradient-weight cosine alignment δ̂_t = ⟨g_t, w_t⟩ / (‖g_t‖ ‖w_t‖ + ε) provides a reliable, non-trivially varying signal for continuous weight decay modulation in standard deep learning benchmarks using Batch Normalization (ResNet-20+BN on CIFAR-10/100, VGG-16-BN on CIFAR-10/100, ResNet-50 on ImageNet with BN).

**Evidence**: Both for and against, honestly presented:

*For the assumption (conventional view is partially right)*:
- CWD (ICLR 2026) demonstrates that sign-level alignment information (the coarsest binary form of δ̂_t) is beneficial and non-redundant — replacing its sign-aligned mask with a random mask substantially degrades performance (loss 2.56 vs 2.82 for AdamW), ruling out the "amount of decay" explanation at the sign level.
- GWA (arXiv:2510.25480) shows gradient-weight alignment is more stable than pairwise gradient similarity at standard batch sizes, with "fewer fluctuations in the mid-to-late training stage" — suggesting continuous alignment signals are more reliable than previously thought.
- Defazio (arXiv:2506.02285) shows WD acts as a real dynamical controller even in normalized networks — the gradient-to-weight ratio does converge to a specific steady state, meaning alignment dynamics carry real physical information.
- Sun et al. (CVPR 2025) show that the alignment quantity δ_T is "much less than 1" in practice on ImageNet/CIFAR experiments, confirming it is a non-degenerate quantity (not trivially 0 or 1).

*Against the assumption (the contrarian case)*:
- **Scale-invariance theorem**: For any layer where BN immediately follows, the pre-BN weight tensor W has the property that the training loss L is approximately invariant under W → αW (for α > 0 in the gradient-flow limit). By the chain rule, this means ∂L/∂W ≈ orthogonal to W, so ⟨g_t, w_t⟩ ≈ 0 and δ̂_t ≈ 0 in expectation for these layers. This is not a noise problem — it is a theoretical near-zero expected value.
- **Chou (2512.08217) direct negative evidence**: Tested alignment-balanced and rotation-aware AdamW variants on ImageNet/ViT-S/16 for 90 epochs, found "almost no differences in relevant metrics." This is the closest existing experiment to what the project proposes.
- **Project's own acknowledged risk**: "δ̂_t 噪声太大导致算法不稳定" (δ̂_t noise too large for algorithmic stability). This is not a minor risk — it is listed as one of four primary risks.
- **Confounding with effective LR**: Even if δ̂_t varies non-trivially, in BN networks the gradient-to-weight ratio effect (Defazio 2025) and the scale-invariance effect mean that "alignment-aware" changes to λ_t may operate via effective LR adjustment rather than genuine alignment-based regularization. These mechanisms are not separable without normalization-free controls.
- **Architecture specificity**: Kobayashi et al. (arXiv:2410.23819) show that WD on multiplicative parameters (post-LN parameters, BN γ/β) acts as nuclear norm regularization, which can *damage* LM performance. If alignment-aware WD applies higher decay to poorly-aligned parameters, and those parameters happen to be post-BN scalars, the alignment signal may cause unintended nuclear norm pressure.

**Hypothesis**:
For the convolutional layers in BN architectures (which contain >95% of network parameters in ResNet-20 and VGG-16-BN), the expected gradient-weight cosine alignment δ̂_t is structurally near-zero (due to BN scale-invariance) and may not provide a meaningful signal for *continuous* modulation of WD strength. Consequently:

1. The Alignment Informativeness Score (AIS), defined as the improvement in WD decisions attributable to alignment information, will be near-zero or negative on standard BN benchmarks — not because alignment doesn't matter in principle, but because BN prevents its observation in the standard benchmark architectures.
2. Any improvements from continuous alignment-modulated WD on BN architectures are confounded with: (a) normalization-induced effective LR adjustments, and (b) reductions in total WD applied, neither of which requires alignment information.
3. The differentiation claim between continuous δ̂_t modulation and CWD's binary sign alignment cannot be validated on BN architectures — only on normalization-free architectures where δ̂_t has non-trivially varying, structurally meaningful values.
4. The convergence of the gradient-to-weight ratio to a single steady state (Defazio 2025) means alignment-aware WD is primarily relevant during the *transient* phase of training — possibly too short to validate in short CIFAR experiments.

**Method**: Controlled ablation to characterize alignment signal quality and disentangle alignment benefit from effective-WD-reduction benefit.

**Experimental Plan**:

*Experiment 1 (δ̂_t signal characterization, ~30 minutes on CIFAR-10)*:
- Train ResNet-20+BN vs ResNet-20-NF (no batch norm; use fixed initialization scaling for stability)
- Log per-layer δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε) for all parameter tensors at every epoch
- Compute: (a) mean(δ̂_t) per layer, (b) std(δ̂_t) per layer over time, (c) autocorrelation of δ̂_t trajectory
- Hypothesis: BN-adjacent layers show δ̂_t ∈ [0, 0.05] with std < 0.02; NF layers show δ̂_t ∈ [0.1, 0.4] with meaningful temporal variation
- This single diagnostic determines whether standard benchmarks can support AIS measurement

*Experiment 2 (Disentangling alignment benefit from WD amount reduction, ~1 hour on CIFAR-10)*:
- Compare four WD configurations on ResNet-20+BN (seeds 42, 123, 456):
  - (a) Standard AdamW with λ = λ_opt (optimally tuned)
  - (b) CWD (ICLR 2026 binary sign mask), same λ
  - (c) CWD baseline: constant AdamW with λ' = λ × E[mask_CWD] (same total WD as CWD without alignment conditioning)
  - (d) Continuous alignment WD: λ_t = clip(c·γ_t·(1 - δ̂_t), λ_min, λ_max)
- Key contrarian prediction: (b) ≈ (c) in final accuracy → alignment information adds nothing over amount reduction. If (b) > (c) but (d) ≈ (b), continuous modulation adds nothing over binary sign.

*Experiment 3 (BN vs NF controlled comparison, ~2 hours on CIFAR-10)*:
- Run Experiment 2's configurations on ResNet-20+BN AND ResNet-20-NF
- Contrarian prediction: the ordering (a) < (b) ≈ (c) < (d) should only emerge on NF, not BN.
- If NF shows (d) > (b) but BN does not, the entire framework needs to qualify its scope to normalization-free architectures.

**Baselines**: Standard AdamW (properly tuned), CWD (ICLR 2026 published, properly tuned), constant λ × E[mask_CWD] (critical missing control)

**Risk Assessment**:

*If the mainstream view is correct (alignment is informative even in BN networks)*:
- δ̂_t distributions in BN-adjacent layers would show meaningful temporal variation (not structurally near-zero)
- Continuous modulation (d) would outperform both (b) and (c) on BN architectures
- The AIS metric would show non-zero values even on standard benchmarks
- This outcome strengthens the project's central claims — Experiment 1 as a fast diagnostic is justified precisely to confirm this

*If the contrarian hypothesis is confirmed*:
- Finding that BN suppresses δ̂_t structurally is a significant negative result — it directly explains why alignment-aware methods show inconsistent results across papers
- The project would need to: (i) add normalization-free architectures as primary benchmarks, (ii) qualify all framework claims with "in non-normalized settings," (iii) reinterpret AIS metric definition
- The paper's framing would shift from "unified evaluation framework" toward "when does alignment-aware WD work: the role of normalization in suppressing the alignment signal" — this is itself a publishable contribution

**Novelty Claim**: The specific mechanistic insight that BN scale-invariance structurally suppresses the radial gradient component (making δ̂_t ≈ 0 for BN-adjacent layers), combined with the experimental methodology to detect this suppression and disentangle it from noise, is not present in any existing paper on alignment-aware WD. This is not merely a critique — it identifies a specific, architectural confound that must be addressed before any continuous alignment-aware WD framework can be validated on standard benchmarks.

---

## Summary of Contrarian Findings

| Risk Category | Severity | Testable? | Current Project Plan Addresses? |
|---|---|---|---|
| δ̂_t structurally suppressed to ~0 in BN-adjacent layers (scale-invariance) | **HIGH** | Yes, Experiment 1 (~30 min) | NO |
| Cosine similarity degrades in high-dimensional parameter spaces (d >> 1000) | **HIGH** | Yes, measure per-layer SNR (~30 min) | NO |
| Continuous modulation gain over CWD confounded with effective WD reduction | **HIGH** | Yes, Experiment 2 (~1 hr) | Partially (CWD comparison planned, but missing the λ × E[mask] control) |
| AIS metric undefined/near-zero on standard BN benchmarks | **HIGH** | Yes, Experiments 1+3 (~2 hr) | NO |
| WD improvements on BN networks confounded with effective LR adjustment | **MEDIUM** | Yes, BN vs NF comparison | NO |
| Sun et al. theory incompatible with time-varying λ_t from random filtration | **MEDIUM** | Requires theory work | Acknowledged as risk; manageable if stagewise theory is prioritized |
| WD mechanism is regime-dependent (over-training vs. under-training) — unified framework misdescribes both | **MEDIUM** | Yes, cross-regime ablations | Partially (different datasets, but same architectural family) |
| Unified framework formula is vacuously true without predictive power test | **LOW** | Requires new-domain prediction | Not planned |
| Neural collapse/low-rank effects from WD are architecture-specific, not captured by scalar λ(t,w,g) | **LOW** | Requires theory extension | Not addressed |

**Strongest Actionable Recommendation**: Before large-scale experiments (ImageNet, 3-seed CIFAR-100, VGG experiments), run Experiment 1 (~30 minutes, CIFAR-10, ResNet-20+BN vs ResNet-20-NF) to characterize δ̂_t distributions per layer. If BN-adjacent layers show δ̂_t concentrated structurally near zero (expected ≈ 0 due to scale-invariance), the entire experimental design needs redesigning around normalization-free architectures. This 30-minute diagnostic could prevent days of compute spent on confounded results and a framework evaluation that cannot distinguish alignment informativeness from architecture artifacts.

**Secondary Recommendation**: Add the missing control (c) from Experiment 2 — constant AdamW with λ reduced to match CWD's average masking fraction — to every CWD comparison. Without this control, it is impossible to attribute CWD's benefits to alignment information rather than simply to less total weight decay.

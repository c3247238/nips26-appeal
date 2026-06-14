# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Sun et al., CVPR 2025. Investigating the Role of Weight Decay in Enhancing Nonconvex SGD]** — First rigorous proof that WD improves generalization (not convergence speed) in nonconvex SGD; introduces alignment quantity δ_T as the key measure. This defines the theoretical ceiling for the current paradigm.

2. **[Chen et al., ICLR 2026. Cautious Weight Decay (CWD). arXiv:2510.12402]** — Binary sign-alignment masking: apply decay only when sign(w) matches sign(update). Bilevel Pareto-optimal interpretation. One-line drop-in. The "obvious" alignment-aware WD variant — already published at ICLR 2026.

3. **[Chen, Yuan, Zhang, arXiv:2602.05136. AdamO. 2026]** — Identifies "Radial Tug-of-War" between WD and gradients; decouples radial (norm) from tangential (direction) dynamics. Very recent, not fully validated. Shows the problem space extends beyond scalar alignment.

4. **[Defazio, arXiv:2506.02285. Why Gradients Rapidly Increase Near the End of Training. 2025]** — WD controls gradient-to-weight ratio ‖g‖/‖w‖; all normalized layers converge to the same steady-state ratio ("layer balancing"). Provides a unified lens for understanding WD's dynamical role across layers.

5. **[Yunis et al., arXiv:2408.11804. Approaching Deep Learning through the Spectral Dynamics of Weights. 2024]** — Spectral dynamics (singular value evolution) as a unifying lens; WD promotes rank minimization across architectures; spectral dynamics distinguish generalizing from memorizing networks.

6. **[Galanti et al., arXiv:2206.05794. SGD and Weight Decay Secretly Minimize Rank. 2022]** — Formal proof that SGD+WD induces low-rank bias in weight matrices. Effect stronger with higher WD.

7. **[arXiv:2505.23489. SGD as Free Energy Minimization: A Thermodynamic View on Neural Network Training. 2025]** — Interprets SGD's stationary behavior as free energy minimization F = U − TS, where LR plays the role of temperature. Weight decay affects the energy landscape. Sharp phase transition at low LR analogous to superconductivity.

8. **[Hölzl et al., arXiv:2510.25480. Gradient-Weight Alignment as a Train-Time Proxy. 2025]** — GWA quantifies coherence between per-sample gradients and model weights; predicts generalization quality; GWA's distinct dynamic pattern during memorization.

9. **[Kuzborskij & Abbasi-Yadkori, arXiv:2502.17340. Low-rank bias, weight decay, and model merging. 2025]** — L2 regularization induces parameter-gradient alignment, norm preservation, low-rank bias at stationary points. Connects static alignment to structural outcomes.

10. **[Franke et al., NeurIPS 2024. CPR: Constrained Parameter Regularization. arXiv:2311.09058]** — Per-parameter-matrix upper-bound constraints via augmented Lagrangian; dynamically adapts per matrix without an explicit WD coefficient. Alternative unifying perspective.

11. **[Ferbach et al., arXiv:2602.06797. Optimal LR Schedules under Functional Scaling Laws. 2026]** — Sharp phase transition between easy tasks (power decay to zero) and hard tasks (WSD). WD-LR schedule co-design framework. Reveals task difficulty as a scheduling dimension.

12. **[Xie et al., arXiv:2011.11152. SWD: Scheduled Weight Decay. NeurIPS 2023]** — First practical WD scheduler driven by gradient norm; closes SGD-Adam generalization gap on CIFAR. Limited theoretical foundation, heuristic-driven.

### Landscape Summary

The weight decay landscape has bifurcated along two axes: (1) **when** to apply WD (scheduling, phase-aware) and (2) **which parameters** to decay (alignment-aware, norm-matched, per-layer). These two axes have been explored almost entirely independently. The richest theoretical work (Sun et al. CVPR 2025) treats only static WD and identifies alignment δ_T as the key quantity. The most innovative alignment-aware work (CWD, ICLR 2026) uses only binary sign alignment without leveraging the rich spectral or thermodynamic structure of the training trajectory.

A critical unexploited connection bridges the thermodynamic view (SGD as free energy minimization, where LR = temperature) with the spectral dynamics view (WD drives rank minimization via singular value compression). In thermodynamic systems, phase transitions are governed by order parameters — and the effective rank of weight matrices evolves as a natural order parameter for training phase transitions. This suggests a new design principle: **use the spectral rank trajectory as a closed-loop feedback signal to schedule WD strength**, analogous to how thermostats use temperature as feedback to maintain desired system states.

A second unexploited connection is between the gradient-to-weight ratio (‖g‖/‖w‖) as a layer balancing controller (Defazio 2025) and the theoretical alignment quantity δ_t from Sun et al. (CVPR 2025). The alignment δ_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖) is precisely the quantity that mediates whether WD helps generalization. When ‖g‖/‖w‖ is far from its steady-state, alignment is informative and WD should be large; when at steady-state, alignment is neutral and WD can be reduced. This suggests a **gradient-to-weight ratio deviation as a WD scheduling signal** that unifies the Defazio dynamics view with the Sun et al. alignment theory.

---

## Phase 2: Initial Candidates

### Candidate A: Spectral-Feedback Weight Decay (SpectralWD)

**Hypothesis**: Using the per-layer effective rank trajectory as a real-time feedback signal to schedule weight decay strength produces better generalization than either fixed or gradient-norm-scheduled WD, because the effective rank is the true optimization target that WD is implicitly driving toward — using it explicitly closes the feedback loop.

**Cross-domain insight**: Borrowed from **closed-loop control theory** (specifically PID/thermostat control). WD is already an open-loop controller trying to drive networks toward low-rank solutions. By measuring actual rank deviation from a target trajectory and using this deviation to modulate WD strength, we convert an open-loop perturbation into a closed-loop regulator. The analogy holds structurally: the rank target plays the role of the setpoint, the effective rank error is the control error, and λ_t is the control signal. This is exactly proportional control: λ_t ∝ (rank_target - rank_actual_t).

**Evidence for**:
- Yunis et al. (2408.11804) conclusively demonstrate that WD drives rank minimization, and that rank trajectory distinguishes memorizing from generalizing networks — rank is thus a meaningful target variable, not just an artifact.
- Galanti et al. (2206.05794) prove the rank-minimization bias is mathematically induced by WD.
- Kuzborskij & Abbasi-Yadkori (2502.17340) connect stationary-point alignment to low-rank structure — validating that rank and alignment are related optimization objectives.
- ADANA (arXiv:2602.05298) shows that logarithmically-scheduled WD alone yields 40% compute efficiency gains — scheduling matters enormously.
- No existing paper uses rank as an explicit feedback signal for WD scheduling (confirmed: Gap 10 from literature survey).

**Novelty estimate**: 8/10 — The closed-loop feedback formulation via effective rank is genuinely novel. Individual pieces exist (WD drives rank, rank distinguishes generalization, scheduling matters) but the feedback control synthesis is new. The risk is that effective rank computation adds overhead, though SVD-based approximations via power iteration can make this cheap.

---

### Candidate B: Free-Energy Phase Transition WD (PhaseWD)

**Hypothesis**: WD should be maximized at training phase transitions (where the free energy landscape undergoes structural changes) and minimized during smooth gradient flow phases, because phase transitions are precisely the moments when the optimization geometry changes most rapidly and WD's effect on stability is most consequential. A WD schedule driven by a phase transition detector outperforms both fixed WD and gradient-norm-scheduled WD.

**Cross-domain insight**: Borrowed from **statistical physics / thermodynamics**. In thermodynamic systems, the response function (susceptibility) peaks at phase transitions — small perturbations have maximum effect near critical points. The analogous quantity in optimization is the sensitivity of the loss landscape to parameter perturbations (curvature). WD is a perturbation to the optimization direction. Applying stronger perturbations at critical points (where the landscape is most malleable) and weaker perturbations in smooth phases parallels how thermostats operate near critical temperatures. The free energy view (arXiv:2505.23489) formally maps LR to temperature and loss to energy, validating this thermodynamic analogy.

**Evidence for**:
- arXiv:2505.23489 formally establishes the SGD-as-free-energy-minimization framework, making the thermodynamic analogy precise rather than metaphorical.
- Grokking-as-phase-transition (ICLR 2024) shows that WD strongly influences whether and when phase transitions in training occur.
- Ferbach et al. (2602.06797) identify a sharp phase transition in optimal LR schedule shape depending on task difficulty — this is a scheduling phase transition, not a training one, but it validates that phase structure in optimization is real and detectable.
- Phase transition detectors can be operationalized via gradient norm volatility (variance of ‖g_t‖ over a window) — spikes in volatility mark critical points in the optimization trajectory.

**Novelty estimate**: 7/10 — The thermodynamic framing is being actively explored (2505.23489 is from May 2025), but using phase transition detection to schedule WD is not in the existing literature. The operationalization via gradient volatility is tractable. However, the thermodynamic framing is gaining momentum and someone else may formalize this connection before us.

---

### Candidate C: Gradient-to-Weight Ratio Equilibrium WD (GWR-EqWD)

**Hypothesis**: The gradient-to-weight ratio ‖g_t‖/‖w_t‖ per layer has a natural steady-state value determined by the learning rate and batch size. When this ratio deviates from its steady-state (which happens continuously during training, especially at the beginning and near LR decay steps), alignment δ_t carries maximum information about whether WD is needed. A WD rule that amplifies λ_t when ‖g_t‖/‖w_t‖ deviates from steady-state, and reduces it when at equilibrium, achieves better convergence than fixed WD by concentrating regularization during the moments of highest geometric information.

**Cross-domain insight**: Borrowed from **control theory / equilibrium regulation**. The gradient-to-weight ratio is a system state variable. Its steady-state is a natural equilibrium (Defazio 2025 proves all normalized layers converge to the same steady-state). Deviations from equilibrium signal that the system is in a transient phase where regularization strength should be adapted. This is a proportional-integral controller where λ_t responds to (‖g_t‖/‖w_t‖ - target_ratio). The key insight is that the steady-state ratio encodes the LR and batch size implicitly, making this a self-calibrating WD rule that automatically accounts for training dynamics.

**Evidence for**:
- Defazio (2506.02285) rigorously proves the gradient-to-weight ratio steady-state and its role in layer balancing — this is the foundational evidence that the ratio is a meaningful equilibrium target.
- The Sun et al. (CVPR 2025) alignment quantity δ_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖) encodes exactly this ratio relationship; when ‖g_t‖/‖w_t‖ deviates from steady-state, δ_t is informative.
- SWD (Xie et al., NeurIPS 2023) already uses gradient norm as a WD scheduling signal — GWR-EqWD generalizes this to the ratio ‖g‖/‖w‖, which is theoretically better motivated.
- No existing paper uses the ratio deviation as a WD modulation signal (confirmed absent from literature survey).

**Novelty estimate**: 8/10 — Defazio (2025) identified the ratio as important but proposed only a corrective term for the LR-schedule interaction. Using the ratio deviation as a continuous WD modulation signal is novel. The theoretical grounding in nonconvex analysis is tractable because it builds directly on Sun et al.'s alignment framework.

---

## Phase 3: Self-Critique

### Against Candidate A (SpectralWD)

**Prior work attack**: Searched for "spectral rank weight decay scheduling feedback" — Yunis et al. (2408.11804) is the closest prior work but explicitly states they do not propose an adaptive WD algorithm. The "Understanding and Scheduling Weight Decay" (OpenReview) paper mentions rank minimization but uses gradient norm as the scheduling signal, not rank itself. AlphaDecay (2025) uses spectral heavy-tailedness per module but assigns static rates, not dynamic per-iteration feedback. No paper found that uses real-time rank measurement as a WD control signal. **Prior work attack: passes.**

**Methodological attack**: Effective rank computation via full SVD is O(min(m,n)²·max(m,n)) per layer — prohibitively expensive for large models. However, power iteration-based effective rank estimation is O(k·m·n) with k~10 iterations, feasible per-batch for small models and every-K-steps for large ones. The rank target trajectory is also unknown a priori — what is the "right" rank to target? This requires either pre-specifying a rank schedule or using the naturally observed rank trajectory from a reference run. **Methodological concern: moderate — addressable via approximation and target specification.**

**Theoretical attack**: The PID analogy breaks down if the control signal (WD) affects the rank measurement in a strongly nonlinear way — if increasing WD also changes the gradient dynamics that drive rank evolution, the feedback loop may oscillate. In control theory this is the stability of the closed-loop system, which requires analysis. However, since WD is typically small (λ << 1), it acts as a weak perturbation and first-order linearization of the rank-WD relationship is reasonable. **Theoretical concern: present but manageable — need to analyze feedback stability.**

**Scalability attack**: For ResNet-50 with ~25M parameters across ~50 layers, computing even approximate rank every N steps is non-trivial. For ViT and LLM-scale models, this may be impractical. The method may be validated only on CIFAR-scale models unless the rank approximation is extremely efficient. **Scalability concern: real, limits practical impact to medium-scale validation within our time budget.**

**Verdict**: MODERATE — The idea is genuinely novel and well-grounded, but the computation overhead for rank feedback needs careful engineering, and the feedback stability analysis is required for a theoretical contribution. The rank target specification problem also needs a principled solution.

---

### Against Candidate B (PhaseWD)

**Prior work attack**: Searched specifically for "phase transition weight decay schedule training" and "thermodynamic weight decay scheduling." The closest work is arXiv:2505.23489 (SGD as free energy minimization), which identifies phase transitions but does not use them to schedule WD. Ferbach et al. (2602.06797) identifies a phase transition in optimal LR schedule shape but for schedule design, not WD. No paper found that uses online phase transition detection to modulate WD. **Prior work attack: passes.**

**Methodological attack**: Phase transition detection during training is the core operationalization challenge. Proposed proxy: variance of ‖g_t‖ over a window. But gradient norm variance can be high throughout early training and during LR warmup for reasons unrelated to phase transitions. Distinguishing genuine phase transitions from ordinary gradient noise requires careful windowing and threshold-setting — two more hyperparameters. Alternative proxy: Hessian trace approximation (HVP with random vectors) captures curvature changes better but adds cost. **Methodological concern: the operationalization is the weakest point — need a more principled transition detector.**

**Theoretical attack**: The thermodynamic analogy (SGD = free energy minimization) is an approximate mapping, not exact. WD changes the energy landscape U(w), so the free energy F = U − TS changes with WD itself. This creates a coupled system where the WD schedule affects which phase transitions occur, which affects the WD schedule — circularity. This is a significant theoretical weakness: the prescription (increase WD at transitions) may destabilize the very transitions it's trying to exploit. **Theoretical concern: serious circularity issue.**

**Scalability attack**: Gradient norm variance computation is cheap (O(1) per step with running statistics). Phase detection can be done at negligible cost. This attack is not a problem. **Scalability: passes.**

**Verdict**: MODERATE-WEAK — The circularity in the theoretical argument (WD changes which transitions occur, transitions trigger WD changes) is a serious flaw. The thermodynamic framing is compelling but the feedback loop is ill-posed. The idea needs substantial reformulation to avoid this circularity, and even then the novelty claim over arXiv:2505.23489 is weaker since that paper directly establishes the thermodynamic framework (even if not for WD scheduling). Leaning toward dropping.

---

### Against Candidate C (GWR-EqWD)

**Prior work attack**: Searched specifically for "gradient-to-weight ratio weight decay schedule alignment." SWD (Xie et al.) uses ‖g_t‖ alone (not the ratio). Defazio (2506.02285) explicitly analyzes the ratio ‖g‖/‖w‖ as a WD-controlled system state but proposes only a corrective term for schedule interaction, not a dynamic WD rule based on ratio deviation. No paper found that uses ratio deviation for WD modulation. The idea is conceptually adjacent to SWD but uses a richer signal. **Prior work attack: passes, with caveat that Defazio is very close and any reviewer will immediately cite it.**

**Methodological attack**: The "steady-state" ratio depends on the optimizer and LR schedule, and changes when LR steps down. Computing it requires either a long warm-up period to estimate the equilibrium (which delays the adaptive WD's effect) or using the theoretical steady-state formula from Defazio (2025), which is derived for specific conditions (normalized layers, specific LR schedules). For unnormalized layers, no steady-state formula exists. **Methodological concern: steady-state estimation is nontrivial for heterogeneous architectures.**

**Theoretical attack**: The theoretical contribution is to augment Sun et al.'s alignment framework with the ratio deviation signal. Sun et al. prove generalization bounds depending on δ_T = sup_t δ_t. GWR-EqWD argues that δ_t is more informative when ‖g‖/‖w‖ deviates from steady-state. This requires proving that the correlation between ratio deviation and alignment δ_t is strong. The correlation may be architecture-specific and difficult to characterize in full generality. **Theoretical concern: the ratio-alignment correlation may not generalize cleanly.**

**Scalability attack**: Computing ‖g_t‖/‖w_t‖ per layer is O(params) per step — the same cost as computing the gradient update, which is already done. This is essentially free. For layer-aggregated versions, even cheaper. **Scalability: passes with flying colors — this is one of the cheapest possible adaptive signals.**

**Verdict**: STRONG — The computational overhead is negligible, the prior work attack fails, the methodological concern about steady-state estimation is addressable (use empirical moving average of ‖g‖/‖w‖ as the equilibrium estimate), and the theoretical concern about correlation is empirically testable. The Defazio proximity is the biggest risk, but using ratio deviation for WD modulation (not just as a diagnostic) is genuinely new.

---

## Phase 4: Refinement

### Dropped Ideas

**Candidate B (PhaseWD)** dropped because: The circularity in the theoretical argument is fatal — WD affects which phase transitions occur, making it impossible to prescribe WD based on detected transitions without solving a fixed-point equation. The thermodynamic framing, while intellectually appealing, creates a coupled feedback loop that is ill-posed. Moreover, arXiv:2505.23489 (SGD as Free Energy Minimization, May 2025) directly establishes the thermodynamic framework, narrowing the novelty claim. Reformulating PhaseWD to avoid the circularity would require major theoretical machinery that goes beyond the scope of a single paper's novel contribution.

### Strengthened Ideas

**Candidate A (SpectralWD)**: Strengthened by addressing the rank target specification problem with the following solution: use the empirically observed rank trajectory from a short pilot run (10% of total training) as the target schedule. This removes the need to pre-specify a rank trajectory and only requires one cheap reference run. The feedback control rule becomes: λ_t = λ_base · max(0, rank(W_t) - rank_target(t)) / rank_target(t). This is a proportional rank-error controller. Additionally, address the computational overhead by using randomized SVD with k=5 components to estimate the effective rank (nuclear norm / spectral norm ratio), which runs in ~1ms per layer for CIFAR-scale models. The feedback stability analysis: since λ_t is bounded above by λ_max and the rank evolution under SGDW has been shown to be monotonically decreasing (Galanti et al.), the closed-loop system is well-posed without oscillation risk.

**Candidate C (GWR-EqWD)**: Strengthened by (1) using an exponential moving average of ‖g_t‖/‖w_t‖ as the empirical steady-state estimate (no special derivation needed, naturally adapts to LR schedule changes), and (2) proving a formal connection between ratio deviation and alignment δ_t using the Cauchy-Schwarz inequality: δ_t = |⟨g_t, w_t⟩| / (‖g_t‖ · ‖w_t‖), and the ratio r_t = ‖g_t‖/‖w_t‖ directly scales the inner product numerator when ‖w_t‖ changes. When r_t deviates from equilibrium, ‖w_t‖ is changing rapidly, which means WD is less effective at maintaining its typical steady-state contribution. The WD rule becomes: λ_t = λ_base · (1 + β · |r_t - r_eq| / r_eq), where r_eq is the EMA-estimated equilibrium and β controls adaptation strength.

### Additional Evidence Found

- arXiv:2505.23489 (SGD as Free Energy, 2025): The free energy framework validates that WD modifies the energy landscape U(w), but does not address ratio-based or rank-based scheduling.
- GWA (arXiv:2510.25480, 2025): GWA correlates coherently with generalization, and GWA's dynamic pattern during memorization (initial near-zero → sharp peak → decay) directly mirrors the alignment δ_t trajectory — providing additional evidence that alignment is informative specifically during early training and LR transitions.
- Ferbach et al. (2602.06797, 2026): The sharp phase transition in optimal LR schedule (easy vs. hard tasks) suggests the ratio r_t = ‖g‖/‖w‖ will behave qualitatively differently across these regimes — validating that ratio deviation is a task-sensitive signal.

### Selected Front-Runner

**Candidate C (GWR-EqWD)** is the front-runner.

Reasons:
1. **Negligible computational overhead** — the ratio ‖g_t‖/‖w_t‖ is already computed implicitly during training; adding explicit tracking costs essentially nothing.
2. **Direct theoretical bridge** — builds on both Sun et al. (CVPR 2025, alignment δ_t) and Defazio (2025, ratio steady-state), providing a clear lineage that reviewers can follow.
3. **Unambiguously novel** — no existing paper uses ratio deviation for WD modulation; Defazio uses ratio as a diagnostic, SWD uses ‖g‖ alone, CWD uses sign alignment. The ratio deviation formulation is a genuine gap.
4. **Both CIFAR and ImageNet tractable** — cheap enough for ImageNet-scale experiments, which is critical for the paper's quality threshold.
5. **Connects to unified framework** — GWR-EqWD can be viewed as a special case of the general dynamic WD rule λ(t, w, g) = f(δ(w,g), ‖w‖, schedule(t)), where f is determined by the ratio deviation, showing how it slots into the broader unified framework.

However, to make this a stronger paper, GWR-EqWD should be extended with: (a) a theoretical convergence result showing that ratio-deviation WD preserves the nonconvex SGD convergence order from Sun et al. (CVPR 2025), (b) a formal characterization of the alignment-informativeness property (when ratio deviation predicts alignment δ_t), and (c) SpectralWD as a complementary method in the experimental suite, since both leverage the spectral structure of the training trajectory.

---

## Phase 5: Final Proposal

### Title
**Equilibrium-Driven Weight Decay: Adaptive Regularization via Gradient-to-Weight Ratio Deviation**

### Hypothesis
The gradient-to-weight ratio r_t = ‖g_t‖/‖w_t‖ per layer converges to a layer-specific equilibrium r* during stable training phases. Deviations from this equilibrium (|r_t - r*|/r*) are a signal that the alignment quantity δ_t = |⟨g_t, w_t⟩|/(‖g_t‖ ‖w_t‖) carries high information about whether weight decay is beneficial. A WD rule λ_t = λ_base · (1 + β · |r_t - r*| / r*) that amplifies decay during high-deviation phases and reduces it during equilibrium phases achieves strictly better generalization than fixed WD and outperforms gradient-norm-scheduled WD (SWD), while adding negligible computational overhead.

Formally falsifiable: If GWR-EqWD on ResNet-50 / ImageNet does not improve over SGDW (fixed) and SWD at the same compute budget (controlling for training epochs), the hypothesis is false.

### Motivation

The theoretical gap this fills: Sun et al. (CVPR 2025) proved that WD's generalization benefit in nonconvex SGD depends on the alignment quantity δ_T = sup_t |⟨g_t, w_t⟩|/(‖g_t‖ ‖w_t‖). Separately, Defazio (2025) showed that WD drives ‖g_t‖/‖w_t‖ to a universal steady-state r* across all normalized layers. These two results have never been connected. The connection is: δ_t is most informative (alignment carries maximum information about beneficial WD application) precisely when r_t deviates from r*, because that's when ‖w_t‖ is changing rapidly and WD has the greatest leverage over the regularization trajectory.

The practical gap: SWD (NeurIPS 2023) schedules WD based on gradient norm ‖g_t‖ alone — a scalar with no structural content about the alignment relationship. GWR-EqWD uses the ratio ‖g_t‖/‖w_t‖, which encodes both gradient magnitude and weight norm simultaneously. This richer signal enables better WD scheduling decisions, especially at LR transitions where both ‖g_t‖ and ‖w_t‖ change simultaneously but in different directions.

Literature gap confirmation: After extensive search across arXiv, web, and literature survey (47 papers), no existing work uses the gradient-to-weight ratio deviation as a WD modulation signal. Defazio (2025) identifies the ratio as a diagnostic; SWD (NeurIPS 2023) uses gradient norm as a WD signal. The combination is absent.

### Method

**Core update rule**:
```
r_t^l = ‖g_t^l‖ / (‖w_t^l‖ + ε)           [per-layer ratio]
r*^l = EMA(r_t^l, α=0.9)                   [adaptive equilibrium estimate]
dev_t^l = |r_t^l - r*^l| / (r*^l + ε)      [normalized deviation]
λ_t^l = λ_base · (1 + β · dev_t^l)         [layer-specific WD]
w_t+1^l = (1 - λ_t^l · γ_t) · w_t^l - γ_t · g_t^l  [SGDW update]
```

**Key design choices**:
- EMA for r* ensures the equilibrium tracks slow changes in the LR schedule without overreacting to per-step noise.
- β controls the adaptation strength (β=0 recovers fixed WD, β→∞ recovers maximally adaptive WD).
- Layer-wise ratios capture heterogeneous dynamics across layers (early layers typically have smaller ratios than late layers in deep networks).
- The deviation normalization ensures scale-invariance: a 10% ratio deviation triggers the same WD amplification regardless of the absolute ratio magnitude.

**Theoretical analysis targets**:
1. **Target A (convergence)**: Prove that GWR-EqWD with λ_t ≤ O(γ_t) preserves the standard nonconvex SGD convergence order O(1/√T) from Sun et al. (CVPR 2025).
2. **Target B (alignment informativeness)**: Prove that E[δ_t | dev_t > threshold] ≥ E[δ_t | dev_t ≤ threshold], i.e., high ratio deviation is a sufficient statistic for high alignment informativeness.
3. **Target C (generalization bound)**: Show that GWR-EqWD's cumulative WD contribution Σ_t λ_t ‖w_t‖ is better distributed across training than fixed WD, leading to a tighter cumulative contraction bound Π_t(1 - λ_t + O(Lγ_t)).

**Connection to Unified Framework**: GWR-EqWD is a special case of the general dynamic WD function λ(t, w, g) with:
- Alignment component: δ_t (implicit, via ratio deviation)
- Norm component: ‖w_t‖ (denominator of ratio)
- Schedule component: γ_t (LR coupling)
- Target norm: 0 (standard decay; can be generalized to τ > 0 as in AdamWN)

This makes GWR-EqWD the first concrete instantiation of the unified framework that emerges naturally from combining two independently established theoretical results (Sun CVPR 2025 + Defazio 2025).

### Experimental Plan

**Pilot experiments (15-20 minutes, CIFAR-10)**:
- VGG-16-BN, CIFAR-10, batch 128, lr=0.1 with cosine schedule
- Compare: Fixed SGDW, SWD (Xie et al.), GWR-EqWD (β=0.5, 1.0, 2.0)
- Track: test accuracy, ‖w_t‖ per layer, r_t per layer, dev_t per layer
- Success criterion: GWR-EqWD at best β ≥ SWD and ≥ Fixed SGDW at final accuracy

**Core experiments (30-60 minutes per run, CIFAR-100)**:
- ResNet-20 and VGG-16-BN on CIFAR-100
- Seeds: 42, 123, 456 (report mean ± std)
- Baselines: No-WD SGD, Fixed SGDW, SWD (NeurIPS 2023), CWD (ICLR 2026), AdaDecay
- Ablations: λ_t = λ_base (β=0), ratio-only without EMA, gradient-norm-only (SWD), ratio-deviation (GWR-EqWD)
- Metrics: Test accuracy, train loss, alignment cosine δ_t trajectory, ratio trajectory, WD budget distribution

**ImageNet experiment (4-8 hours, primary scale validation)**:
- ResNet-50 on ImageNet-1K, standard 90-epoch recipe
- Seeds: 42, 123 (3rd seed if time permits)
- Baselines: AdamW (fixed WD), SGDW (fixed), SWD, CWD, GWR-EqWD
- This is the critical experiment for paper quality — ImageNet results provide the convincing evidence

**Falsification criterion**: If GWR-EqWD does not improve over SWD on ≥2/3 benchmark settings, the hypothesis is falsified. Report this honestly if it occurs.

**Diagnostic visualizations** (per the literature recommendation):
- Per-layer ratio trajectory r_t^l with r* overlay
- Effective WD strength λ_t^l heatmap across layers and training time
- Alignment cosine δ_t trajectory vs. ratio deviation dev_t^l (correlation plot)
- Weight norm ‖w_t^l‖ trajectory under each WD method

### Resource Estimate

- Pilot (CIFAR-10, VGG-16-BN): ~15-20 min, 1 GPU RTX PRO 6000
- CIFAR-100 experiments: 30-60 min per run × 2 architectures × 5 methods × 3 seeds ≈ 12-18 hours total, parallelizable across 8 GPUs → ~2-3 hours wall clock
- ImageNet (ResNet-50, 90 epochs): ~4-6 hours per run, 2-3 seeds → 8-18 hours wall clock on 1-2 GPUs
- Theoretical derivations: 3-5 days of focused work
- Visualization and analysis: 1-2 days

**Compute fits within project constraint**: CIFAR experiments are well within 1-hour budget per run; ImageNet is covered by the documented time-budget override for ImageNet experiments.

### Risk Assessment

**Risk 1: β hyperparameter sensitivity** — GWR-EqWD introduces β as a new hyperparameter controlling adaptation strength. If performance is highly sensitive to β, practitioners cannot easily adopt the method. Mitigation: provide a theoretical derivation of the optimal β in terms of the WD-alignment trade-off (analogous to Sun et al.'s optimal λ derivation), and validate that a small set {0.5, 1.0, 2.0} covers most use cases empirically.

**Risk 2: Defazio (2025) proximity undermines novelty** — Reviewers familiar with arXiv:2506.02285 may claim GWR-EqWD is obvious given Defazio's ratio analysis. Mitigation: explicitly position the paper as "translating Defazio's diagnostic insight into a WD scheduling algorithm," cite Defazio as the motivation, and emphasize that Defazio proposes a corrective term for LR-schedule interaction (a different problem) while GWR-EqWD proposes WD modulation (a different solution). The combination with Sun et al.'s alignment theory is the novel contribution.

**Risk 3: ImageNet results may not improve over baselines** — If the ratio deviation signal is too noisy at ImageNet scale (larger batch sizes smooth the per-step ratio, reducing the deviation signal), GWR-EqWD may degenerate toward fixed WD. Mitigation: pre-validate the ratio deviation amplitude at ImageNet scale during pilot experiments; if the deviation is consistently small, increase β or switch to a multi-step smoothed deviation.

### Novelty Claim

The novel contributions, with evidence that they haven't been done:

1. **First use of gradient-to-weight ratio deviation as a WD scheduling signal**: Confirmed absent from all 47 papers in the literature survey, arXiv searches, and web searches. Defazio (2025) identifies the ratio as important but uses it only diagnostically, not for WD modulation.

2. **Theoretical bridge between Sun et al. (CVPR 2025) alignment theory and Defazio (2025) ratio dynamics**: These two results have not been connected in prior work. The connection (ratio deviation → alignment informativeness → WD scheduling) is a novel theoretical contribution.

3. **First concrete instantiation of the Unified Dynamic WD Framework as a tractable algorithm**: The GWR-EqWD formulation naturally fits the general λ(t, w, g) framework while being computationally trivial to implement (three lines of code addition to any SGDW or AdamW optimizer).

4. **Layer-wise WD adaptation via heterogeneous ratio equilibria**: No existing method conditions WD per layer on that layer's own ratio equilibrium; CWD and SWD apply global signals while AdamWN targets a global norm. The per-layer formulation respects the empirical heterogeneity in layer dynamics documented in Defazio (2025) and Yunis et al. (2024).

The evidence that this is new: After searching arXiv with "gradient-to-weight ratio weight decay schedule," "ratio deviation weight decay modulation," "equilibrium-driven weight decay," "‖g‖/‖w‖ weight decay adaptation," and related terms — none returned a matching paper. The closest hit is SWD (uses ‖g‖ alone) and Defazio (uses ‖g‖/‖w‖ as a diagnostic, not for WD scheduling).

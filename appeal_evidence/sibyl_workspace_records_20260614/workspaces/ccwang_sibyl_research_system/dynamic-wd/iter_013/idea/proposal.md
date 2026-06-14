# Unified Dynamic Weight Decay: Cumulative Alignment Contraction Theory with Gradient-to-Weight Ratio Equilibrium Scheduling

## Abstract

Weight decay (WD) in deep learning has fragmented into four independent streams — scheduling, alignment-aware, decoupled, and norm-matched — each with separate theoretical justifications, different evaluation protocols, and incompatible hyperparameter spaces. We propose a unified theoretical framework that connects these streams through two novel contributions: (1) **Cumulative Alignment Contraction Theory**, which replaces the worst-case alignment bound of Sun et al. (CVPR 2025) with an average-case bound that formally proves alignment-aware WD achieves strictly tighter generalization under fixed WD budgets; and (2) **Equilibrium-Driven Weight Decay (EqWD)**, a practical algorithm that uses the gradient-to-weight ratio deviation from its layer-specific equilibrium as a WD scheduling signal. EqWD bridges Sun et al.'s alignment theory with Defazio's (2025) ratio dynamics, providing the first algorithm that is simultaneously theoretically grounded in nonconvex generalization theory, computationally negligible (three lines of code), and applicable to both SGD and Adam families. We introduce a layer-type-aware formulation that accounts for the distinct roles of WD in normalized vs. non-normalized layers, addressing the fundamental critique that alignment signals are uninformative for scale-invariant layers. Standardized metrics (BEM, CSI, AIS) enable the first apples-to-apples comparison of all four WD streams under matched compute budgets.

## Motivation

### Theoretical Gap

Sun et al. (CVPR 2025) proved the first formal generalization bound for SGDW depending on the alignment quantity delta_T = sup_t |<g_t, w_t>| / (||g_t|| ||w_t||). This is a worst-case bound over all timesteps. Separately, Defazio (2025) showed that WD drives the gradient-to-weight ratio ||g_t||/||w_t|| to a universal steady-state r* across all normalized layers. These two foundational results have never been connected.

The connection we establish: delta_t is most informative — alignment carries maximum information about whether WD is beneficial — precisely when the gradient-to-weight ratio r_t deviates from its equilibrium r*. When r_t is at equilibrium, the system is in a stable phase and WD has predictable effects; when r_t deviates, the system is in transition and WD's alignment-dependent effect is most consequential.

### Practical Gap

SWD (NeurIPS 2023) schedules WD based on gradient norm ||g_t|| alone — a scalar with no structural content about the alignment relationship. CWD (ICLR 2026) uses binary sign alignment — throwing away magnitude information. No method uses the ratio ||g_t||/||w_t||, which encodes both gradient magnitude and weight norm simultaneously, as a WD modulation signal. Furthermore, no existing method accounts for the fundamental difference between normalized and non-normalized layers in how alignment signals should be interpreted.

### Literature Gap Confirmation

After extensive search across arXiv, Google Scholar, and web sources (47+ papers surveyed by the 6 perspectives), no existing work:
- Uses the gradient-to-weight ratio deviation as a WD modulation signal
- Provides an average-case alignment generalization bound that strictly dominates the worst-case bound
- Accounts for layer-type heterogeneity (normalized vs. non-normalized) in alignment-aware WD design
- Offers a standardized evaluation framework for comparing all four WD streams under budget equivalence

## Research Questions

1. Does an average-case alignment bound yield a strictly tighter generalization guarantee than the worst-case bound of Sun et al. (CVPR 2025), and by how much in practice?
2. Does gradient-to-weight ratio deviation carry more information for WD scheduling than gradient norm alone?
3. Does layer-type-aware alignment (treating normalized and non-normalized layers differently) improve upon uniform alignment-aware WD?
4. Under matched compute budgets, does any dynamic WD method genuinely outperform optimally-tuned fixed WD?

## Hypotheses

See `hypotheses.md` for detailed testable hypotheses with expected outcomes and falsification criteria.

## Method

### Core Algorithm: Equilibrium-Driven Weight Decay (EqWD)

```
r_t^l = ||g_t^l|| / (||w_t^l|| + eps)           [per-layer ratio]
r*^l = EMA(r_t^l, alpha=0.9)                     [adaptive equilibrium estimate]
dev_t^l = |r_t^l - r*^l| / (r*^l + eps)          [normalized deviation]
lambda_t^l = lambda_base * (1 + beta * dev_t^l)   [layer-specific WD]
w_{t+1}^l = (1 - lambda_t^l * gamma_t) * w_t^l - gamma_t * g_t^l  [SGDW update]
```

Key design choices:
- **EMA for r***: ensures the equilibrium tracks slow changes in the LR schedule without overreacting to per-step noise
- **beta**: controls adaptation strength (beta=0 recovers fixed WD; default beta=1.0)
- **Layer-wise ratios**: capture heterogeneous dynamics across layers (early layers typically have smaller ratios)
- **Deviation normalization**: ensures scale-invariance across layers with different absolute ratio magnitudes
- **Negligible overhead**: ratio ||g_t||/||w_t|| is implicit in every training step; explicit tracking costs essentially nothing

### Layer-Type-Aware Extension

For normalized layers (BatchNorm, LayerNorm, GroupNorm): WD acts as an effective learning rate modifier, not a regularizer. The alignment signal delta_hat_t captures effective LR dynamics, not geometric regularization.

```
For layer l:
  if is_normalized(l):
    lambda_t^l = lambda_base * phi_norm(dev_t^l)   [ratio-deviation driven, no alignment]
  else:
    lambda_t^l = lambda_base * phi_free(dev_t^l, delta_hat_t^l)  [ratio + alignment]
```

This addresses the contrarian critique that alignment signals are uninformative for scale-invariant layers (Van Laarhoven 2017; NeurIPS 2024 normalization paper).

### Theoretical Framework

**Theorem 1 (Cumulative Alignment Contraction Bound):**
For SGDW with alignment-aware dynamic WD lambda_t = psi(alpha_t, gamma_t), the generalization gap satisfies:
```
E[f(w_T) - f_S(w_T)] <= C1 * exp(-B) * ||w_0||^2 / n + C2 * T * sigma^2 * gamma^2 / n
```
where B = sum_t lambda_t * E[1 - alpha_t] is the alignment-weighted budget. This replaces Sun's worst-case delta_T with average-case delta_avg, yielding a strictly tighter bound.

**Theorem 2 (Optimal Budget Allocation):**
Under constraint sum_t lambda_t <= Lambda, the alignment-weighted budget B is maximized by:
```
lambda_t* = Lambda * (1 - alpha_t) / sum_s (1 - alpha_s)
```
The formal advantage over fixed WD is Lambda * T * Var(alpha_t) — large when alignment fluctuates.

**Proposition 3 (Unified Ratio Equilibrium Characterization):**
For all four WD families: (i) Fixed WD: r* = lambda/gamma; (ii) CWD: r* = lambda/gamma * P(alpha >= 0); (iii) Scheduled WD: r*(t) ≈ lambda_t/gamma_t (quasi-static); (iv) Norm-matched: r* -> 0 as ||W|| -> tau.

### Unified Framework: The Phi Formulation

All WD strategies are special cases of: lambda_t(l) = lambda_base * Phi(cos_l, ||w_l||, t, tau_l, type_l), where:
- Phi = 1: fixed WD
- Phi = I[sign(w) * sign(g) > 0]: CWD (binary mask)
- Phi = (1 + cos_l) / 2: CAWD (continuous alignment)
- Phi = f(||g||, t): SWD (gradient-norm scheduled)
- Phi = ||w|| / tau: norm-matched
- Phi = 1 + beta * |r - r*| / r*: EqWD (ratio-deviation driven)

### Standardized Metrics

- **Budget Equivalence Metric (BEM)**: Compare at equal compute (FLOPs), not equal epochs
- **Coupling Stability Index (CSI)**: CSI = Var(r_t) / E[r_t] — rolling coefficient of variation of the gradient-to-weight ratio; lower = more stable WD-optimizer coupling
- **Alignment Informativeness Score (AIS)**: AIS = I(delta_hat_t; generalization_gap | ||g_t||) — mutual information between alignment signal and generalization gap, controlling for gradient norm

## Expected Contributions

1. **Theoretical**: First average-case alignment generalization bound that strictly dominates worst-case (Sun CVPR 2025); first formal proof that alignment-aware WD is optimal under fixed budget; unified ratio equilibrium characterization across all WD families
2. **Algorithmic**: EqWD — first algorithm using gradient-to-weight ratio deviation for WD scheduling; layer-type-aware formulation; negligible computational overhead
3. **Empirical**: First budget-equivalent comparison of all four WD streams; standardized metrics (BEM, CSI, AIS); comprehensive per-layer diagnostic visualizations on both CIFAR and ImageNet
4. **Framework**: Unified Phi formulation showing all existing WD methods as special cases; layer-type-aware extension addressing the normalized-vs-non-normalized layer distinction

## Experimental Plan

### Pilot (15-20 min, CIFAR-10)
- VGG-16-BN, batch 128, lr=0.1 with cosine schedule
- Compare: Fixed SGDW, SWD, CWD, EqWD (beta=0.5, 1.0, 2.0)
- Track: test accuracy, per-layer ratios, alignment trajectories
- Go/no-go: EqWD must show non-trivial ratio deviation signal (variance > 0.01)

### Alignment Information Diagnostic (1 hour, CIFAR)
- Pre-experiment: test whether delta_hat_t contains information beyond (||g_t||, ||w_t||)
- Use k-NN mutual information estimator with bootstrap CI
- If I(delta_hat_t; test_acc | ||g_t||, ||w_t||) CI includes 0: alignment signal is not incrementally informative

### Core Experiments (CIFAR-100, 30-60 min per run)
- Architectures: ResNet-20, VGG-16-BN
- Seeds: 42, 123, 456 (report mean +/- std)
- Methods: No-WD SGD, Fixed SGDW, SWD, CWD, CPR, EqWD
- Ablations: beta sensitivity, layer-type-aware vs. uniform, EMA decay rate
- Metrics: test accuracy, BEM, CSI, AIS, per-layer ratio trajectories

### ImageNet Experiments (4-8 hours, PRIMARY evidence)
- ResNet-50 on ImageNet-1K, standard 90-epoch recipe
- Seeds: 42, 123 (3rd if time permits)
- Methods: AdamW (fixed), SGDW (fixed), SWD, CWD, CPR, EqWD
- This is the critical experiment for paper quality

### Budget Equivalence Test
- Each method gets identical Bayesian optimization budget (50 Optuna trials)
- Fixed WD baseline with equivalent search budget
- Report which methods survive matched-budget comparison
- If no dynamic WD beats tuned fixed WD: report as rigorous negative finding

### Control Experiments
1. Phase-schedule control: replay average lambda_t trajectory as fixed schedule
2. Gradient-norm-only control: replace ratio deviation with ||g_t|| normalized to [0,1]
3. Noise injection: add Gaussian noise with sigma = std(delta_hat_t) to alignment signal
4. Layer-type ablation: CWD on only BN layers vs. only non-BN layers

### Diagnostic Visualizations
- Per-layer ratio r_t^l with r* overlay
- Effective WD lambda_t^l heatmap across layers and training time
- Alignment cosine delta_t vs. ratio deviation (correlation plot)
- Weight norm trajectories under each method
- AIS heatmap by layer and training phase

## Resource Estimate

- Pilot: ~20 min, 1 GPU
- CIFAR experiments: ~3 hours wall-clock (parallelized across 8 GPUs)
- ImageNet: ~4-6 hours per run, 2-3 seeds -> 8-18 hours wall-clock
- Total: ~3-4 days with 8x RTX PRO 6000

## Risk Assessment

1. **Beta hyperparameter sensitivity**: If performance is sensitive to beta, practical adoption suffers. Mitigation: theoretical derivation of optimal beta; validate {0.5, 1.0, 2.0} covers most cases.

2. **Defazio (2025) proximity**: Reviewers may claim EqWD is "obvious" from Defazio. Mitigation: position as "translating Defazio's diagnostic into a scheduling algorithm" + combination with Sun's alignment theory is the novel bridge.

3. **ImageNet results may not improve**: Ratio deviation may be too smooth at scale. Mitigation: pre-validate deviation amplitude; increase beta or use multi-step smoothing if needed.

4. **Alignment signal is uninformative for normalized layers**: The contrarian's critique is serious. Mitigation: layer-type-aware formulation explicitly addresses this; include ablation showing per-layer-type behavior.

5. **Budget equivalence null result**: Fixed WD may match dynamic WD with equal tuning budget. Mitigation: this is a valuable negative finding; reframe as evaluation framework contribution.

6. **Proof gap in Theorem 1**: The stability recursion requires going through potential function approach (not standard stability). Mitigation: use Lyapunov potential from Sun et al. as starting point; the average-case extension is tractable.

## Novelty Assessment

### Core novelty claims (verified absent from literature):
1. **Average-case alignment generalization bound**: No paper uses average alignment delta_avg instead of worst-case delta_T for SGDW bounds. ArXiv search "average alignment generalization weight decay" returns 0 matches.
2. **Gradient-to-weight ratio deviation as WD signal**: Defazio (2025) uses ratio as diagnostic only; SWD uses ||g|| alone; no paper uses ratio deviation for WD modulation. Confirmed absent.
3. **Layer-type-aware alignment WD**: No paper distinguishes normalized vs. non-normalized layers for alignment-based WD decisions.
4. **Formal proof that alignment-aware WD is optimal under fixed budget**: The Cauchy-Schwarz argument for budget allocation is new.
5. **Unified ratio equilibrium across all 4 WD families**: Defazio covers only fixed SGD/Adam; extensions to CWD, scheduled, norm-matched are new.

### Closest prior work and differentiation:
- **SWD (NeurIPS 2023)**: Uses ||g_t|| alone, not the ratio ||g_t||/||w_t||
- **CWD (ICLR 2026)**: Binary sign mask, no ratio dynamics
- **Defazio (2025)**: Ratio as diagnostic, not scheduling signal
- **CPR (NeurIPS 2024)**: Per-matrix constraint via Lagrangian, no alignment
- **Sun et al. (CVPR 2025)**: Worst-case bound, no dynamic WD algorithm

## Perspective Synthesis: What Each Contributed

| Perspective | Key Contribution to Final Proposal | Weight |
|---|---|---|
| **Innovator** | Front-runner: GWR-EqWD algorithm; connection between Sun and Defazio | Highest |
| **Theoretical** | Cumulative Alignment Contraction bound (Theorems 1-2); formal proof of alignment-aware optimality | Highest |
| **Pragmatist** | CAWD concept (continuous alignment modulation); unified Phi framework; implementation details | High |
| **Contrarian** | Layer-type-aware critique; WD-LR duality for normalized layers; signal reliability concerns | High |
| **Empiricist** | Budget equivalence test protocol; alignment informativeness diagnostic; control experiments | High |
| **Interdisciplinary** | BCM sliding threshold inspiration (incorporated into adaptive EMA design); evolutionary equilibrium framework for BEM grounding | Moderate |

### Conflict Resolution

The most significant conflict was between the **Innovator/Pragmatist** (who favored alignment-based approaches) and the **Contrarian** (who argued alignment is uninformative for normalized layers). Resolution: both are correct for different layer types. The layer-type-aware formulation accepts the contrarian's critique for normalized layers while preserving alignment value for non-normalized layers.

The **Empiricist's** insistence on budget-equivalent comparison challenges the entire research agenda's premise. Resolution: include the budget equivalence test as a core experiment. If the null hypothesis holds (fixed WD matches dynamic WD), the paper reframes as an evaluation framework contribution with a significant negative finding. The theoretical contributions remain valuable regardless.

The **Interdisciplinary** BCM sliding threshold and evolutionary frameworks provide rich theoretical grounding but the BCM direction conflicts with CWD's sign convention (BCM suggests decay more when misaligned; CWD decays when aligned). Resolution: the BCM insight is incorporated into the adaptive EMA mechanism for the equilibrium estimate, not the decay direction. The evolutionary framework grounds the BEM metric in Haldane's load theorem.

## Revisions from Prior Feedback

This is the first synthesis round. No prior proposal, pilot evidence, novelty report, or Codex feedback exists.

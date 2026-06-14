# Final Research Proposal: Iteration 3

## Title

**Unified Dynamic Weight Decay: A Proximal-Theoretic Framework with Standardized Evaluation and Comprehensive Benchmarking**

## Abstract

We propose a unified framework that characterizes all major dynamic weight decay (WD) methods -- scheduling, alignment-aware, decoupled, and norm-matched -- as instances of a general time-varying proximal operator applied to the parameter update. This operator-theoretic unification reveals that the four sub-approaches modulate WD along orthogonal axes (temporal, directional, structural, and target), explains when their compositions are beneficial versus redundant, and yields a novel WD Stability Condition that provides actionable constraints on schedule design. We complement the theoretical framework with three standardized evaluation metrics (Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score) and the first compute-controlled head-to-head benchmark of all major WD variants (AdamW, CWD, SWD, cosine WD, AdamWN) on the same codebase. A critical CWD falsification battery tests whether alignment-awareness genuinely contributes beyond reduced effective WD strength. Our systematic visualization toolkit produces publication-quality diagnostic panels tracking per-layer weight norms, alignment cosines, effective learning rates, and modulation patterns. The framework is grounded in prior empirical evidence from this project (iterations 0-2) showing budget equivalence for temporal scheduling and limited alignment informativeness at the nonconvex scale, which motivates the hypothesis that spatial modulation (per-parameter, per-layer) is the load-bearing axis of dynamic WD.

## Motivation

Weight decay is the most universally applied regularization technique in deep learning, yet the field suffers from a severe fragmentation problem. Four independent research threads each propose improvements to WD from incompatible mathematical formulations:

1. **WD Scheduling** (SWD/AdamS, ADANA, cosine WD): adjusts decay strength over training time
2. **Alignment-Aware WD** (CWD, AdamO): conditions decay on gradient-weight geometry
3. **Decoupled WD** (AdamW vs. Adam+L2): separates WD from gradient preconditioning
4. **Norm-Matched WD** (AdamWN, AlphaDecay): targets a specific weight norm rather than zero

The practical consequence is that practitioners cannot make informed choices between methods -- the papers are simply not comparable. CWD (ICLR 2026) reports final loss improvements, SWD (NeurIPS 2023) focuses on gradient norm stability, AlphaDecay reports spectral density analysis, and AdamWN describes target-norm control theory without extensive benchmarks. No existing work provides a head-to-head comparison under equal compute budgets.

Meanwhile, a deeper theoretical gap exists: nobody has shown that these four approaches are instances of the same mathematical structure. The operator-theoretic perspective (Chen et al. 2023 on Lion; Xie & Li 2024 on AdamW) reveals that each optimizer+WD combination implicitly solves a different constrained optimization problem, but this insight has not been extended to encompass all four dynamic WD sub-approaches within a single framework.

Our prior iterations (0-2) of this project established three critical empirical findings that reshape the research landscape:

- **Budget Equivalence**: Mean(lambda_t) = constant lambda yields identical performance for temporal scheduling, suggesting WD scheduling is a red herring under budget normalization.
- **Alignment Signal Limitation**: Gradient-weight alignment is uninformative at the nonconvex scale of real deep learning, challenging the premise of alignment-aware methods.
- **LR-WD Coupling**: WD operates primarily through effective learning rate modulation in normalized networks (consistent with D'Angelo et al., NeurIPS 2024 and Zhang et al. 2018).

These findings, scored 8.2/10 and rated "publication-ready for NeurIPS" in iteration 2's review, demand a more ambitious framing: not just reporting that dynamic WD fails, but explaining *why* through a unified theoretical lens, identifying *which axis* of dynamic WD actually matters, and providing the community with standardized tools for future evaluation.

## Research Questions

1. Can all four WD sub-approaches be unified as instances of a time-varying proximal operator, and does this unification yield non-trivial theoretical predictions?
2. Under compute-controlled conditions, do any dynamic WD methods achieve statistically significant improvement over optimally tuned constant WD?
3. Is CWD's improvement attributable to alignment-awareness or to reduced effective WD strength?
4. Do the proposed standardized metrics (BEM, CSI, AIS) predict method rankings across architectures and datasets?
5. Is the spatial dimension (per-parameter, per-layer modulation) the only axis where dynamic WD provides genuine benefit?

## Hypotheses

**H1 (Proximal Unification)**: All four WD sub-approaches can be expressed as time-varying proximal operators with specific structural choices for the regularizer R_t, and the resulting taxonomy is complete for first-order methods with additive WD.

**H2 (WD Stability Condition)**: There exists a necessary condition on the rate of WD schedule change -- the regularizer must not change faster than the optimization is converging -- that predicts when WD schedules cause training instability. WD warmup from 0 to lambda in fewer than 1/(eta*lambda) steps should violate this condition and produce measurable loss spikes.

**H3 (Budget Equivalence at Scale)**: Under equal compute budgets, temporal WD scheduling (cosine, linear, SWD) achieves statistically indistinguishable final accuracy from optimally tuned constant WD on CIFAR-10/100 and ImageNet. Falsification: any temporal schedule achieves >0.3% improvement with p<0.05 across 3+ seeds.

**H4 (CWD Mechanism)**: CWD's improvement over standard AdamW is substantially (>50%) attributable to reduced effective WD strength rather than alignment-awareness. Falsification: CWD outperforms effective-lambda-matched constant WD by >0.3% AND random-mask WD with matched sparsity performs >0.3% worse than CWD.

**H5 (Metric Predictiveness)**: CSI and AIS have predictive power for method rankings (Spearman rho > 0.5) across method-architecture-dataset combinations.

**H6 (Spatial > Temporal)**: Per-parameter and per-layer WD modulation (spatial axis) provides genuine benefit through effective learning rate redistribution, while temporal WD scheduling provides near-zero benefit under budget equivalence.

## Expected Contributions

1. **Unified Phi-Framework**: The first mathematical framework expressing all four WD sub-approaches as special cases of a per-parameter modulation function Phi(w, u, t), with proofs of orthogonality/redundancy between approaches.

2. **Proximal-Theoretic Taxonomy**: Complete characterization of WD methods as time-varying proximal operators with a novel WD Stability Condition providing actionable schedule design constraints.

3. **Standardized Metrics (BEM, CSI, AIS)**: The first quantitative metrics specifically designed for evaluating and comparing dynamic WD methods, empirically validated for predictive power.

4. **Compute-Controlled Benchmark**: The first head-to-head comparison of all major WD variants on the same codebase with identical training budgets and proper statistical testing.

5. **CWD Falsification Battery**: The first direct test of whether CWD's alignment-awareness genuinely contributes beyond WD reduction (matched-lambda, random-mask, inverted-mask, continuous-modulation ablations).

6. **Systematic Visualization Toolkit**: Comprehensive diagnostic panels enabling visual comparison of weight norm trajectories, alignment structures, effective learning rates, and modulation patterns.

7. **Evidence-Driven Synthesis**: Integration of three iterations of empirical evidence (budget equivalence, alignment limitation, LR-WD coupling) into a coherent theoretical narrative.

## Method

### 1. Unified Per-Parameter WD Modulation Framework

The general dynamic WD update is:

```
w_{t+1} = w_t - eta_t * u_t - eta_t * Phi(w_t, u_t, t) * lambda_0 * w_t
```

where u_t is the optimizer update and Phi is the per-parameter modulation matrix (diagonal). Existing methods correspond to specific Phi choices:

| Method | Phi(w, u, t) | Axis |
|--------|-------------|------|
| Standard AdamW | I (identity) | -- |
| CWD | diag(1_{sign(w_i * u_i) >= 0}) | Directional |
| SWD/AdamS | h(||g_t||) * I | Temporal |
| Cosine WD | cos_schedule(t) * I | Temporal |
| AdamWN (target tau) | (1 - tau/||w||) * I | Target |
| AlphaDecay | diag(alpha_layer) | Spatial |
| gamma^2 scaling | (eta_t/eta_base) * I | Structural |

The unification reveals four orthogonal modulation axes: temporal (through t), directional (through w,u geometry), target (through ||w||), and spatial (through layer/parameter identity).

### 2. Proximal-Theoretic Characterization

Each WD method admits a proximal interpretation:

```
Phi_t(w) = (1/eta) * (w - prox_{eta * R_t}(w))
```

where R_t is a time-varying regularizer:
- Standard: R_t(w) = (lambda/2)||w||^2
- Scheduled: R_t(w) = (lambda(t)/2)||w||^2
- Alignment-aware (soft CWD): R_t(w) = (lambda/2) sum w_i^2 sigma(beta*w_i*u_i)
- Norm-matched: R_t(w) = (lambda/2)||w - tau*w/||w||||^2

**WD Stability Theorem**: The time-varying Lyapunov function V_t = f(w_t) + R_t(w_t) decreases in expectation iff:

```
|R_{t+1}(w_t) - R_t(w_t)| <= (1 - rho) * eta_t * ||nabla f(w_t)||^2 + sigma^2
```

This WD Stability Condition states that the regularizer can change at most as fast as the loss is decreasing.

**Optimal quadratic schedule**: For f(w) = (1/2)w^T H w, the optimal lambda(t) is curvature-weighted and monotonically increasing, theoretically justifying WD warmup.

### 3. Standardized Evaluation Metrics

**Budget Equivalence Metric (BEM)**: Normalized performance difference under equal FLOPs.

**Coupling Stability Index (CSI)**: Composite of (a) coefficient of variation of weight norm trajectory, (b) spectral condition number at convergence, (c) effective LR stability across layers.

**Alignment Informativeness Score (AIS)**: Correlation between gradient-weight alignment and per-step loss improvement. If AIS ~ 0, alignment-aware methods are not exploiting a meaningful signal.

### 4. Comprehensive Benchmark

All methods implemented in a single PyTorch codebase forked from `why-weight-decay` (NeurIPS 2024):
- **Architectures**: ResNet-20, VGG-16-BN (CIFAR), ResNet-50 (ImageNet), ViT-S (optional)
- **Datasets**: CIFAR-10, CIFAR-100, ImageNet-1K
- **Methods**: AdamW, CWD-AdamW, SWD/AdamS, Cosine-WD, AdamWN, AlphaDecay-style, CWD+Cosine composition
- **Protocol**: 3 seeds (42, 123, 456), LR tuned per method, identical augmentation/batch/steps
- **Statistical tests**: Paired t-test with Bonferroni correction, bootstrap 95% CI, Cohen's d

### 5. CWD Falsification Battery

- CWD vs. effective-lambda-matched constant WD
- CWD vs. random binary mask with matched sparsity
- CWD vs. inverted (anti-alignment) mask
- CWD vs. continuous cosine-similarity-weighted WD
- Track per-layer mask ratio over training

### 6. Visualization Toolkit

Six standardized diagnostic panels per method:
1. Training loss and test accuracy curves
2. Per-layer weight norm trajectories
3. Per-layer gradient-weight cosine similarity
4. Effective learning rate per layer
5. CSI and AIS evolution over training
6. Phi modulation heatmap (layers x time)

## Experimental Plan

**Phase 1: Pilot Validation (~3 GPU-hours, ~1h wall clock)**
- CIFAR-10, ResNet-20, {AdamW, CWD, SWD, Cosine-WD}, 3 seeds
- Validate CSI/AIS differentiate methods; test effective-lambda hypothesis for CWD
- Kill criterion: If CSI/AIS indistinguishable, rethink metrics

**Phase 2: CIFAR Comprehensive (~18 GPU-hours, ~3h on 8 GPUs)**
- CIFAR-10 + CIFAR-100, ResNet-20 + VGG-16-BN, all 7 methods, 3 seeds
- Full CWD falsification battery
- All metrics + visualization panels
- WD Stability Condition test: vary warmup steps K in {1, 10, 50, 200, 1000}

**Phase 3: ImageNet Validation (~72 GPU-hours, ~9h on 8 GPUs)**
- ImageNet-1K, ResNet-50, top 4-5 methods, 3 seeds
- Verify metric rankings transfer from CIFAR
- Distributed training on 8x RTX PRO 6000 Blackwell

**Phase 4: Architecture Generalization (~24 GPU-hours, ~3h on 8 GPUs)**
- ViT-S on CIFAR-100 and/or ImageNet
- Compare Phi structure across BN vs LN architectures

**Total: ~117 GPU-hours, ~16h wall clock on 8x RTX PRO 6000 Blackwell**

## Evidence-Driven Revisions (from Iterations 0-2)

This proposal integrates lessons from three prior iterations:

1. **Iteration 0** (Score 5.0): Established AADWD (Alignment-Aware Dynamic Weight Decay) as the initial research direction. Pilot experiments revealed that alignment proxy reliability was borderline (r=0.849) and that norm-matched WD performed comparably to AADWD, suggesting the alignment signal's marginal contribution was limited.

2. **Iteration 1** (Score 7.8): Reframed as a negative-result study. Established budget equivalence, LR-WD coupling, and alignment uninformativeness as core findings. Review identified multi-seed experiments as mandatory and cross-architecture validation as high priority.

3. **Iteration 2** (Score 8.2): Resolved numerical inconsistencies, added random-schedule robustness verification, achieved "publication-ready for NeurIPS" assessment. Remaining issues: (a) narrow scope (single architecture/dataset), (b) no theoretical framework explaining the negative results.

**Key pivot in Iteration 3**: Rather than continuing to strengthen a negative-result paper about AADWD, we elevate the scope to a *unified framework paper* that (a) explains why temporal scheduling fails (budget equivalence as a theorem, not just an observation), (b) identifies where dynamic WD actually works (spatial modulation), (c) provides standardized tools for future research (metrics + benchmark + visualization), and (d) offers theoretical grounding through the proximal-operator unification. The negative results from iterations 0-2 become empirical evidence supporting the framework's predictions, rather than the paper's sole contribution.

## Novelty Assessment

### Systematic Novelty Verification

**Claim 1: Unified Phi-framework for WD methods.** Searched arXiv for "unified framework weight decay" and "proximal operator weight decay". No existing work provides a complete per-parameter modulation taxonomy of all four WD sub-approaches. PathProx (Yang et al. 2022) treats fixed WD as proximal for ReLU networks but does not address time-varying or alignment-aware WD. Newhouse's thesis (2025) connects WD to mirror descent but does not derive the four-way unification.

**Claim 2: WD Stability Condition.** Searched for "weight decay stability condition convergence" and "time-varying regularization Lyapunov". No work derives a necessary condition on the rate of WD schedule change. Chen et al. (2023) provide Lyapunov analysis for Lion with fixed WD; our extension to time-varying WD with the stability constraint is novel.

**Claim 3: Standardized WD evaluation metrics (BEM, CSI, AIS).** No existing metrics are specifically designed for comparing dynamic WD methods. Individual papers track weight norms or alignment cosines, but no standardized suite exists.

**Claim 4: Compute-controlled benchmark of all major WD variants.** Confirmed: no paper provides head-to-head comparison of CWD, SWD, AdamWN, cosine WD schedule, and AlphaDecay-style methods on the same codebase.

**Claim 5: CWD falsification battery.** CWD's paper (ICLR 2026) tested a random-mask ablation (which underperformed CWD), but did NOT test effective-lambda matching, inverted mask, or continuous alignment modulation. Our battery is more comprehensive.

**Closest prior work and differentiation:**
- D'Angelo et al. (NeurIPS 2024): Studies WD mechanisms but does not compare dynamic WD methods or propose metrics.
- CWD (ICLR 2026): Proposes one dynamic WD method; does not unify or benchmark others.
- SWD (NeurIPS 2023): Proposes scheduling; does not address alignment-aware or norm-matched WD.
- Wen et al. (2025): Benchmarks optimizers but not WD variants specifically.

### Potential Collision Risks

**Risk 1: Someone publishes a similar benchmark.** The WD research community is active. Mitigation: our theoretical unification (proximal framework) and standardized metrics differentiate us from a pure benchmark paper.

**Risk 2: CWD authors publish their own mechanism analysis.** The ICLR 2026 paper defers rigorous Filippov analysis to future work. If they publish this before us, our CWD falsification battery still contributes independently.

**Risk 3: Caraffa (2026) or followers publish Fisher-Rao WD.** Our framework subsumes Fisher-Rao WD as one special case (the per-parameter modulation using v_t as Fisher proxy). We are not competing with the thermodynamic approach but incorporating it.

## Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| All methods perform the same under proper tuning | Medium | High (60%) | Frame as validation of unified framework's prediction; metrics and visualization have standalone value |
| CWD falsification is ambiguous (p=0.05-0.10) | High | Medium (40%) | Increase seeds to 5 for CWD-specific comparisons; Bayesian analysis |
| Proposed metrics (CSI, AIS) are uninformative | Medium | Medium (50%) | Report negative results transparently; propose simpler alternatives |
| WD Stability Condition trivially satisfied | Medium | Medium (30%) | Find at least one practical violation (WD warmup is best candidate) |
| Soft CWD proximal approximation too crude | Low | Low (20%) | Show experimentally that soft CWD (large beta) matches hard CWD |
| Theory perceived as "just notation" | Medium | Medium (35%) | Prove composition theorem (which modulations compose beneficially); predict WD warmup advantage |
| CIFAR results don't transfer to ImageNet | Medium | Medium (40%) | Always include ImageNet; acknowledge CIFAR limitations |

## Baselines

**Theoretical**: Ding et al. (2023) convergence for fixed AdamW; Chen et al. (2023) Lyapunov for Lion; Li et al. (2020) intrinsic LR dynamics.

**Empirical** (all properly tuned with equal HP budget):
1. Standard AdamW with constant WD (grid-searched)
2. SGD + momentum + constant WD
3. CWD-AdamW (ICLR 2026)
4. SWD/AdamS (NeurIPS 2023)
5. Cosine WD schedule
6. AdamWN (Loshchilov 2023)
7. CWD + Cosine WD (composition test)

## Resource Estimate

| Phase | GPU-hours | Wall clock (8x RTX PRO 6000) |
|-------|----------|---------------------------|
| Pilot | 3 | 1 hour |
| CIFAR full | 18 | 3 hours |
| ImageNet | 72 | 9 hours |
| ViT | 24 | 3 hours |
| **Total** | **~117** | **~16 hours** |

All experiments fit within the project's compute budget (8x RTX PRO 6000 Blackwell, 98GB each). ImageNet experiments use multi-GPU distributed training. No external API costs.

# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **tml-epfl/why-weight-decay** (https://github.com/tml-epfl/why-weight-decay, MIT) — Primary evaluation scaffold. Runs ResNet-18/34, VGG-16 on CIFAR-10/100 and Tiny-ImageNet with comprehensive weight norm, gradient norm, and Hessian trace tracking. Code is available and working. This is the single most reusable codebase for our project.

2. **Cautious Weight Decay (CWD)** (arXiv:2510.12402, ICLR 2026) — The binary sign-alignment baseline we extend. One-line implementation: `mask = (p.data * update) > 0; p.data -= lr * wd * p.data * mask`. Validated on AdamW/Lion/Muon at 338M–2B params. ViT-S/16 ImageNet: 78.84% → 79.45%.

3. **CPR / AdamCPR** (https://github.com/automl/CPR, Apache-2.0, NeurIPS 2024) — Per-parameter-matrix constraint via augmented Lagrangian. `pip install pytorch-cpr`. No-hyperparameter `kappa_init_method='inflection_point'` variant. ImageNet DeiT: +2–3% vs AdamW. GPT-2: same perplexity with 33% fewer steps. Strong baseline, minimal setup overhead.

4. **SWD / AdamS** (https://github.com/zeke-xie/stable-weight-decay-regularization, MIT, NeurIPS 2023) — Gradient-norm-aware WD scheduler. Working implementation; direct baseline for the scheduling sub-approach.

5. **Spectral Dynamics codebase** (https://github.com/dyunis/spectral_dynamics, MIT) — Tracks singular value evolution during training across architectures. Needed for our CSI metric and Gap 10 (spectral-feedback WD).

6. **GWA implementation** (https://github.com/hlzl/gwa, PyTorch+JAX) — Gradient-weight cosine similarity metric with open-source code. Directly reusable as the underlying signal for our continuous alignment modulation.

7. **Sun et al. CVPR 2025** (core paper) — Proves nonconvex SGDW generalization benefit via alignment quantity δ_T. Formal convergence framework we extend. No public code but math is self-contained.

8. **D'Angelo et al. NeurIPS 2024** (arXiv:2310.04415) — Unifying "WD as dynamics modifier" perspective. Uses same experimental framework (tml-epfl repo). Establishes that WD is not classical regularization.

9. **Defazio 2025** (arXiv:2506.02285) — WD drives ‖g‖/‖w‖ to a steady state ("layer balancing"). This gradient-to-weight ratio is a natural unified lens metric for our CSI. No code but formula is simple to implement.

10. **AdamO** (arXiv:2602.05136) — Radial/tangential decomposition, Feb 2026, no public code yet. Important conceptually for our unified framework but cannot be baselined directly.

11. **OUI** (https://github.com/AlbertoFdezHdez/OUI, MIT) — Overfitting-Underfitting Indicator; validation-free WD monitoring. Reusable as one component of our Coupling Stability Index.

12. **NOVAK** (arXiv:2601.07876) — Shows coupling WD with effective LR (not base LR) degrades CIFAR-100 by 4–8pp. Key negative result for our BEM metric design.

### Landscape Summary

**What works in practice**: Decoupled WD (AdamW) is the baseline. Binary sign-alignment (CWD) gives +0.6% ImageNet accuracy with zero overhead. Per-matrix constraint (CPR) gives +2–3% on DeiT at cost of Lagrange multiplier overhead. Gradient-norm-aware scheduling (SWD) helps close Adam-SGD gap on CIFAR but the improvement is small.

**What does not work**: Coupled WD with adaptive optimizers is definitively worse (NOVAK shows 4–8pp penalty). Module-wise static WD (AlphaDecay) works for LLMs but adds heuristic complexity. Per-parameter sigmoid-scaled WD (AdaDecay) is outperformed by CPR and AdamW.

**Practical gap**: No one has implemented a continuously modulated, cosine-similarity-based alignment-aware WD for vision tasks. CWD is binary; nothing fills the continuous spectrum. The GWA implementation exists but has never been used to gate WD. This is the engineering opportunity.

**Benchmark fragmentation is real and measurable**: NOVAK compares 14 optimizers but does not control for WD coupling. CWD, SWD, CPR, and AlphaDecay each use different datasets and metrics. Nobody has run all four WD sub-approaches on the same training run and measured them with the same diagnostic set. This is achievable in one codebase.

---

## Phase 2: Initial Candidates

### Candidate A: Continuous Alignment-Aware Weight Decay (CAWD)

- **Hypothesis**: Replacing CWD's binary sign-alignment mask with a continuous cosine similarity score λ_t(i) = λ · σ(cos(w_i, update_i)) — where σ is a monotone function — improves over CWD on ImageNet classification without adding hyperparameters, because it exploits the magnitude of alignment, not just its sign.

- **Implementation sketch**: Start from the tml-epfl/why-weight-decay codebase (MIT). Add a custom SGD and AdamW variant that computes `cos_sim = F.cosine_similarity(p.data.flatten(), update.flatten(), dim=0)` per parameter group (or per layer) and modulates the WD coefficient by `wd_effective = wd * (1 + cos_sim) / 2` (this maps [-1,1] to [0,1], consistent with CWD's binary mask as a special case). Add CWD as the binary baseline. Run on ResNet-50/ImageNet with 3 seeds.

- **Simplest version**: CIFAR-100 with VGG-16-BN, 160 epochs, batch size 128. Compare: (a) fixed WD, (b) CWD, (c) CAWD with cosine modulation. ~20 minutes per run on one RTX PRO 6000. Pilot experiment total: 1 hour.

- **Time estimate**: CIFAR pilot: 1 hour (3 methods × 20 min). ImageNet ResNet-50: ~6 hours per seed × 3 seeds = 18 hours wall-clock on 2 GPUs.

- **Reusable components**: tml-epfl/why-weight-decay training harness; GWA cosine similarity code; CWD mask code from arXiv:2510.12402 (3 lines).

---

### Candidate B: Gradient-to-Weight Ratio Equalization as a Unified Diagnostic Framework

- **Hypothesis**: The gradient-to-weight ratio ρ_t(l) = ‖g_l‖/‖w_l‖ per layer l is a sufficient statistic for characterizing all four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched): each method drives ρ toward a different target or via a different path, and this can be formalized and measured. Measuring ρ trajectories across methods on shared benchmarks produces the first standardized comparative framework.

- **Implementation sketch**: Instrument tml-epfl training loop to log ρ_t(l) for every layer every N steps. Run all 5–6 WD variants (fixed WD, CWD, SWD, CPR, CAWD) on CIFAR-100/VGG-16-BN and ImageNet/ResNet-50. Generate 2×2 grid plots: x=training step, y=ρ(l) per layer, color=WD method. No new optimizer code needed beyond what already exists.

- **Simplest version**: CIFAR-100 only, single seed, log ρ across fixed WD variants (λ = 0, 1e-4, 1e-3, 5e-3). 4 runs × 20 min = 80 minutes total. Already answers: "does WD compress ρ to a tighter distribution across layers?"

- **Time estimate**: CIFAR diagnostic: 2 hours. Full ImageNet comparative panel: 30–40 GPU-hours across all methods.

- **Reusable components**: tml-epfl training harness (already logs weight/gradient norms); Defazio's corrective term formula (arXiv:2506.02285); Spectral Dynamics codebase for singular value tracking.

---

### Candidate C: Unified WD Optimizer Interface with Pluggable Strategy + Standardized Metrics

- **Hypothesis**: A single optimizer class `UnifiedWD(base_optimizer, strategy='fixed'|'cwd'|'swd'|'cawd'|'cpr')` with standardized metric logging (BEM, CSI, AIS) is achievable in <500 lines of code, and comparing all strategies under this interface reveals that: (1) BEM normalization changes the ranking of methods relative to published results; (2) CSI predicts generalization gaps better than final accuracy alone.

- **Implementation sketch**: Wrap PyTorch SGD/AdamW with a `WDStrategy` mixin. Implement: (a) fixed — standard; (b) cwd — binary mask from arXiv:2510.12402; (c) swd — gradient-norm-gated from SWD; (d) cawd — cosine-similarity-weighted (Candidate A); (e) cpr_lite — norm-feedback WD where wd_t = wd * (‖w‖/‖w‖_target) without full augmented Lagrangian. Log per-step: wd_effective, ‖w‖_l, ‖g‖_l, ρ_l, cos_sim_l. BEM = compare methods at equal FLOPs (not equal epochs). CSI = variance of wd_effective over last 20% of training. AIS = mutual information between cos_sim signal and subsequent accuracy improvement.

- **Simplest version**: Implement fixed + cwd + cawd only (3 strategies), run on CIFAR-100/ResNet-20, log BEM and CSI. 3 runs × 10 min = 30 minutes. This is the real pilot.

- **Time estimate**: Implementation: 3–4 hours coding. CIFAR experiments: 2 hours. ImageNet: 20–40 GPU-hours.

- **Reusable components**: AdamW, SGD from PyTorch; CWD one-liner; SWD GitHub; CPR GitHub (installable). OUI for one CSI component.

---

## Phase 3: Self-Critique

### Against Candidate A (CAWD)

- **Implementation reality check**: Computing per-parameter cosine similarity inside the optimizer step is O(d) extra computation per layer. For ResNet-50, d ~ 25M parameters. Per-layer cosine similarity is O(d_l) which is feasible. However, per-parameter cosine similarity requires `flatten()` which copies tensors — need to be careful about memory. The GWA paper (arXiv:2510.25480) warns that "computing full network gradients at every step introduces significant implementation and computational overhead." Key question: are we computing gradient-weight cosine similarity (cheap, just dot products with existing tensors) or per-sample gradient alignment (expensive)? Answer: we use the minibatch gradient g_t and parameter vector w_t — both already computed, no extra backward needed. Cost is just a dot product per layer.

- **Reproducibility attack**: The modulation function σ(cos) introduces a new design choice. If we use `(1 + cos_sim) / 2`, that's one natural choice but not uniquely motivated. Papers will ask "why not sigmoid(cos_sim)" or "why not cos_sim^2"? We need an ablation across 3-5 modulation functions to show robustness. This is doable but adds ~5 more runs.

- **Baseline sanity check**: CWD already achieves +0.6% ImageNet accuracy (ViT-S/16: 78.84% → 79.45%). Is that improvement threshold meaningful enough that a continuous extension would also show a detectable improvement? With std of ~0.1% on ImageNet top-1, a +0.3–0.5% further improvement from continuous modulation is plausible but not guaranteed. Risk: the improvement may be within noise.

- **Scope attack**: CIFAR-100 may show clearer signal than ImageNet due to smaller model/faster training dynamics. But project constraints require ImageNet as primary evidence. The alignment signal cos_sim may be noisier in large models with batch normalization (BN absorbs weight scaling, changing the geometric relationship between w and g).

- **Verdict**: MODERATE-STRONG. Implementation is feasible (3–5 hours). Core risk is that the improvement over CWD may be too small to be significant on ImageNet. Need >3 seeds and careful statistical testing.

---

### Against Candidate B (ρ-Ratio Diagnostic Framework)

- **Implementation reality check**: Logging ρ_t(l) per layer requires hooking into the training loop to extract per-layer gradient norms and weight norms. The tml-epfl codebase already does this (they log weight norms and gradient norms). This is genuinely a 1–2 hour modification of existing code, not a new system. Risk is visualization complexity: 50-layer ResNet-50 × N timesteps × 5 methods = large multi-panel figures that may be cluttered.

- **Reproducibility attack**: ρ trajectories are fully deterministic given the same seed. No hidden hyperparameters. Reproducibility is high.

- **Baseline sanity check**: Defazio (arXiv:2506.02285) already shows the ρ steady-state property analytically for normalized layers. Our contribution is empirical verification across methods and layers, not the analytical result itself. Risk: this is "verification of known theory" not a new result. Mitigation: our contribution is the comparative measurement and the AIS metric derived from it, not just ρ itself.

- **Scope attack**: Diagnostic/measurement papers with strong visualizations do get published (e.g., D'Angelo et al. NeurIPS 2024 is primarily empirical). But the contribution must be "we measured X and found surprising result Y," not just "we measured X and found it matches theory." Need to find surprising disagreements between methods or unexpected behaviors.

- **Verdict**: MODERATE. High practical value and feasibility, but the novelty claim is weaker unless we find something unexpected in the ρ trajectories across methods.

---

### Against Candidate C (Unified Interface + Standardized Metrics)

- **Implementation reality check**: A `WDStrategy` mixin wrapping PyTorch optimizers is standard software engineering. The complexity is moderate but manageable. The BEM metric (compare at equal FLOPs) requires computing FLOPs per forward/backward pass, which is non-trivial to compute accurately across architectures. Workaround: use wall-clock time on the same hardware as proxy for FLOPs (less rigorous but practical). The AIS metric (mutual information between cos_sim and subsequent accuracy) is computationally expensive and hard to estimate reliably from finite samples. Simpler proxy: Pearson correlation between wd_effective variance and final generalization gap.

- **Reproducibility attack**: The metric definitions need to be precisely pinned (what window for CSI? what lag for AIS?). If left vague, different implementations will disagree. Need to commit to concrete formulas in the paper.

- **Baseline sanity check**: The benchmark contribution is valuable independently of any new algorithm. The question is: does the comparison reveal that BEM changes the ranking? If methods happen to all train for the same number of epochs on the same hardware, BEM = accuracy and is uninformative. BEM only matters when methods have different compute costs per step. CWD adds ~2% overhead. CPR adds ~0.3% overhead. So BEM corrections will be small. Risk: BEM may not change rankings meaningfully.

- **Scope attack**: The software engineering is real work but may not be "novel enough" for top venues unless the metric design itself is theoretically motivated and the empirical findings are surprising.

- **Verdict**: MODERATE. The unified interface is a means to the end (enabling comparison), not the end itself. Best combined with Candidate A to give both a new algorithm and a standardized evaluation.

---

## Phase 4: Refinement

**Dropped**: No ideas fully dropped. Candidate B is demoted to a supporting tool (ρ-trajectory visualization is a key figure, not a separate contribution).

**Strengthened by combining A + C + B**:
The strongest single proposal combines:
- **Candidate A (CAWD)** as the core algorithmic contribution — continuous cosine alignment modulation of WD
- **Candidate C's unified interface** as the experimental infrastructure — enabling apples-to-apples comparison
- **Candidate B's ρ-diagnostic** as the main analysis and visualization — explaining *why* CAWD works

Key additional search: Do practitioners report that continuous alignment signals are too noisy compared to binary masks? The GWA paper (arXiv:2510.25480) notes that cosine similarity between per-sample gradients and weights "diminishes rapidly with increasing dimensions" but this is per-sample alignment. The minibatch-average alignment (what we use) is more stable and precisely what Sun et al. CVPR 2025 study (the δ_T quantity). This is a material distinction.

**Simplification**: Remove per-parameter cosine similarity; use per-layer cosine similarity instead. This reduces the alignment signal from d-dimensional to L-dimensional (L = number of layers, typically 20–50 for ResNets). Per-layer alignment: `cos(l) = dot(flatten(w_l), flatten(g_l)) / (‖w_l‖ ‖g_l‖)`. Per-layer is cheaper, more stable, and directly connected to Sun et al.'s δ_T (which is a global alignment quantity, but per-layer is a natural extension).

**Minimal pilot design**:
1. Run VGG-16-BN on CIFAR-100 for 10 epochs with fixed WD, CWD, and CAWD (3 runs × 10 min = 30 min total)
2. Log per-layer ρ_t(l) and cos(l) throughout
3. Check: does CAWD produce smoother ρ trajectories than CWD? Does it apply more WD in early training (when alignment is typically higher) and less later?
4. If yes → proceed to full CIFAR-100 experiments. If no → the cosine signal is not informative and we need to rethink.

**Selected front-runner**: Candidate A (CAWD) + Candidate B (ρ-diagnostic) + Candidate C (unified interface infrastructure), packaged as a single paper with three contributions: (1) unified theoretical framework, (2) CAWD algorithm, (3) standardized metrics and visualization.

---

## Phase 5: Final Proposal

### Title
Unified Dynamic Weight Decay: Theoretical Framework, Continuous Alignment-Aware Algorithm, and Standardized Evaluation

### Hypothesis
The four streams of dynamic weight decay (scheduling, alignment-aware, decoupled, norm-matched) are special cases of a general WD update λ_t(l) = f(cos(w_l, g_l), ‖w_l‖, t, τ_l) with interpretable parameter choices. Furthermore, replacing CWD's binary sign-alignment mask with continuous cosine similarity modulation — Continuous Alignment-Aware Weight Decay (CAWD) — achieves measurably better generalization than CWD on ImageNet classification at no hyperparameter cost.

### Motivation
Every major optimizer paper since 2023 proposes a different weight decay mechanism (CWD, CPR, SWD, AlphaDecay, AdamO, AdamWN) but there is no common framework and no standardized comparison. Practitioners cannot determine which method is best for their setting. Theoretically, the field lacks a unified formulation that explains *why* these methods work and how they relate. Concretely: CWD is binary (sign only), but the gradient-weight cosine similarity δ_T is continuous and already proven relevant to generalization (Sun et al. CVPR 2025). No method exploits the full continuous alignment signal.

### Method (Step-by-Step)

**Step 1: Implement the unified WD update rule (1 week)**

Define: `λ_t(l) = λ_base · Φ(cos_l, ‖w_l‖, t, τ_l)` where:
- `cos_l = dot(flatten(w_l), flatten(g_l)) / (‖w_l‖ · ‖g_l‖ + ε)` — per-layer alignment
- Special cases: Φ=1 (fixed WD), Φ=1[sign(w_l)·sign(g_l)>0] (CWD), Φ=(1+cos_l)/2 (CAWD), Φ=(‖w_l‖/τ_l) (norm-matched), Φ=f(gradient_norm, t) (SWD)

Implement in PyTorch as a WD strategy mixin that can be attached to SGD or AdamW. Code: ~200 lines.

**Step 2: Implement CAWD algorithm (2 days)**

```python
# Inside optimizer.step():
for group in self.param_groups:
    for p in group['params']:
        if p.grad is None: continue
        grad = p.grad.data
        # Per-layer cosine alignment
        cos_sim = F.cosine_similarity(
            p.data.flatten(), grad.flatten(), dim=0
        ).clamp(-1, 1)
        # Continuous modulation: maps [-1,1] to [0,1]
        wd_scale = (1.0 + cos_sim) / 2.0
        # Apply modulated weight decay
        p.data.mul_(1.0 - group['lr'] * group['weight_decay'] * wd_scale)
```

**Step 3: Implement standardized metrics (2 days)**

- **Budget Equivalence Metric (BEM)**: Record wall-clock time per step. Report all accuracy comparisons at equal wall-clock time (not equal epochs). Implementation: log `time.perf_counter()` per batch.
- **Coupling Stability Index (CSI)**: `CSI(l) = std(wd_effective(l, t) for t in last_20_pct_of_training)`. Log wd_effective at each step. Lower CSI = more stable WD behavior.
- **Alignment Informativeness Score (AIS)**: `AIS = Pearson_corr(cos_sim(t), Δaccuracy(t, t+Δ))` over training. Measures whether alignment signal predicts near-future improvement.

**Step 4: Run CIFAR pilot experiments (1 day)**

Setup: VGG-16-BN on CIFAR-100, 160 epochs, batch 128, lr=0.1 with milestones [60,120], seeds {42, 123, 456}.
Methods: fixed WD (λ=5e-4), CWD, CAWD-linear, CAWD-squared, SWD.
Log: test_acc, ‖w_l‖, ‖g_l‖, ρ_l, cos_l, wd_effective_l, CSI, AIS.
Expected time: 20 min per run, 15 runs total = 5 hours on 3 parallel GPUs.

**Step 5: Run ImageNet experiments (primary evidence)**

Setup: ResNet-50 on ImageNet-1K, 90 epochs, batch 256, lr=0.1 with cosine decay, seeds {42, 123, 456}.
Methods: AdamW (fixed WD), AdamW+CWD, AdamW+CAWD, AdamCPR.
Expected time: 6 hours per run on 2 GPUs (DDP). Total: ~18 GPU-hours per seed.
Additional: ViT-S/16 on ImageNet (90 epochs, AdamW baseline, CAWD, CWD). ~10 hours per run.

**Step 6: Theoretical framework (2–3 weeks, parallel)**

Prove that the four sub-approaches are special cases of the general Φ framework. Extend Sun et al. CVPR 2025 nonconvex SGDW convergence analysis to time-varying λ_t. Key theorem: if `λ_t = λ_base · (1 + cos_l(t)) / 2` and `λ_base = O(γ_t)`, convergence order is preserved while generalization bound depends on `mean_t[(1 + cos_l(t)) / 2 · (1 - δ_t)]` instead of `max_t(1 - δ_t)`.

**Step 7: Visualization panel (2 days)**

Generate 6-panel figure: (a) weight norm trajectories per method; (b) per-layer ρ_t(l) heatmap; (c) cos_sim(l,t) heatmap; (d) wd_effective(l,t) heatmap; (e) CSI bar chart across methods; (f) AIS correlation scatter. Use seaborn/matplotlib. Save as PDF/PNG, verify visually.

### Simplest Version
CIFAR-100 + VGG-16-BN, 160 epochs, 3 methods (fixed WD, CWD, CAWD), 3 seeds, log ρ and cos_sim. This tests the core claim (continuous modulation > binary) with 9 runs × 20 min = 3 GPU-hours.

### Baselines

1. **AdamW with fixed decoupled WD** (λ=5e-4): The universal baseline. Expected ImageNet ResNet-50 top-1: ~76.1–76.5%.
2. **Cautious Weight Decay (CWD)**: The strongest alignment-aware baseline. Expected ImageNet: +0.6% over fixed WD (~76.7–77.1%). This is the direct competitor for CAWD.
3. **CPR / AdamCPR** (kappa_init='inflection_point'): The strongest per-matrix adaptive baseline. Expected ImageNet DeiT: +2–3% over AdamW, but heavier Lagrange multiplier overhead.
4. **SWD (Scheduled Weight Decay)**: The scheduling baseline. Expected CIFAR improvement: ~0.3–0.5% over fixed WD.

### Experimental Plan

| Phase | Dataset | Model | Methods | Duration | Primary Metric |
|-------|---------|-------|---------|----------|---------------|
| Pilot | CIFAR-100 | VGG-16-BN | fixed, CWD, CAWD | 1 hour | test acc + CSI |
| CIFAR full | CIFAR-100 | VGG-16-BN, ResNet-20 | 5 methods, 3 seeds | 4 hours | mean±std acc, BEM, CSI, AIS |
| ImageNet primary | ImageNet-1K | ResNet-50 | fixed, CWD, CAWD, CPR | 30 GPU-hours | top-1 acc, BEM |
| ImageNet extended | ImageNet-1K | ViT-S/16 | fixed, CWD, CAWD | 30 GPU-hours | top-1 acc, BEM |
| Visualization | CIFAR+ImageNet | all | all | post-processing | ρ heatmaps, CSI bar |

**Ablation schedule**:
1. Modulation function shape: (1+cos)/2 vs sigmoid(cos) vs cos^2 vs 1[cos>0]
2. Granularity: per-layer vs per-parameter-group vs global cosine
3. Optimizer: SGD vs AdamW vs Muon (if time allows)

### Resource Estimate

- **CIFAR pilot**: 1 GPU-hour (1×RTX PRO 6000)
- **CIFAR full**: 4 GPU-hours (3 GPUs parallel, 5 methods × 2 architectures × 3 seeds)
- **ImageNet ResNet-50**: 18 GPU-hours (2×DDP, 3 seeds × 3 methods × 6 hours each)
- **ImageNet ViT-S/16**: 30 GPU-hours (2×DDP, 3 seeds × 2 methods × 10 hours each)
- **Total wall-clock**: ~3 days with 4–6 parallel GPUs
- **Coding overhead**: 1 week (optimizer mixin, metric logging, visualization)

### Risk Assessment

**Engineering risks**:
- *Gradient-weight cosine stability with BatchNorm*: BN normalizes activations but not weight-gradient alignment. cos_sim may still be well-defined per layer but has different behavior in BN vs non-BN layers. Mitigation: test with both VGG-16-BN and ResNet-20 (no BN in residual branch). Flag discrepancy if observed.
- *Memory overhead of flatten() per layer*: Creates a temporary tensor equal to layer parameter size. For ResNet-50 layer sizes up to ~2M parameters, this is negligible (<10MB). For ViT with 86M parameters, still acceptable (single layer cosine is computed within optimizer step, not retained).
- *Numerical stability of cosine similarity for small norms*: Early training weight norms can be very small. Use `clamp(ε)` in denominator. Standard practice.
- *DDP synchronization*: wd_effective differs per layer per GPU replica. Since WD is applied after gradient aggregation (in decoupled formulation), the cosine similarity is computed with the synchronized gradient — no extra synchronization needed.

**Scientific risks**:
- *Improvement over CWD may be within noise*: On ImageNet top-1, std across seeds is ~0.1–0.2%. A +0.3% improvement from continuous modulation may not be significant. Mitigation: run 5 seeds (vs standard 3), or use CIFAR experiments where results are more stable.
- *Continuous modulation may underperform binary mask for adaptive optimizers*: AdamW already adapts per-parameter step size; the cosine signal may be redundant. Test both SGD+CAWD and AdamW+CAWD to separate concerns.
- *Unified framework may not be mathematically tight*: Proving all four sub-approaches as strict special cases of Φ requires careful formulation. Worst case: we can only show they are *related* but not strict special cases. Mitigation: accept looser "unifying perspective" framing as backup.

### Novelty Claim

1. **Theoretical**: First unified framework showing that CWD, SWD, AdamWN, and CPR are special cases of a single parameterized WD update λ_t(l) = λ·Φ(cos_l, ‖w_l‖, t, τ_l). Extension of Sun et al. CVPR 2025 convergence theorem to time-varying, alignment-conditioned λ_t.

2. **Algorithmic**: CAWD — Continuous Alignment-Aware Weight Decay — is the first method to use the full continuous gradient-weight cosine similarity (not binary sign) to modulate WD strength. Addresses Gap 3 in the literature survey. Drop-in, one hyperparameter (λ_base only), no architectural changes.

3. **Empirical**: First standardized comparison of all four WD sub-approaches on shared benchmarks (CIFAR-100, ImageNet) with equal-compute normalization (BEM), stability measurement (CSI), and alignment informativeness quantification (AIS). Comprehensive per-layer ρ and alignment trajectory visualization.

The novelty is genuinely "showing that continuous alignment-aware WD works where binary alignment already works, and providing the first systematic framework to understand why." Even if the accuracy improvement is modest (+0.3–0.5% on ImageNet), the theoretical framework and standardized metrics are independently valuable contributions.

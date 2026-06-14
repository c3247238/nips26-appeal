# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **tml-epfl/why-weight-decay** (GitHub, 70 stars, MIT) — NeurIPS 2024. Complete codebase for ResNet/VGG/ViT experiments on CIFAR-10/100, with weight/gradient norm tracking built-in. 3-seed protocol. **Has usable code. Our best starting point for the evaluation framework.**

2. **zeke-xie/stable-weight-decay-regularization** (GitHub, MIT) — NeurIPS 2023 SWD implementation. Gradient-norm-aware dynamic WD via AdamS optimizer. Runnable on CIFAR. **Has usable code. Direct baseline for WD scheduling.**

3. **CWD (Cautious Weight Decay)** (arXiv:2510.12402, ICLR 2026) — One-line drop-in modification for any optimizer. Binary sign-alignment mask. Demonstrated on AdamW/Lion/Muon at 338M-2B scale on LM and ImageNet ViT-S/16. No separate repo needed—the modification is literally one line of code. **Has usable code (trivial to implement).**

4. **Sun et al. (CVPR 2025)** — "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD." Foundational theory: WD improves generalization (not convergence) in nonconvex SGD. Key quantity: worst-case alignment delta_T. **No public code found, but the theory is straightforward to build on.**

5. **D'Angelo et al. / why-weight-decay (NeurIPS 2024)** — "Why Do We Need Weight Decay in Modern Deep Learning?" Unifying perspective: WD is a dynamics modifier, not a regularizer. Loss stabilization mechanism for SGD, bias-variance tradeoff for LLMs. **Their codebase is our primary experimental framework.**

6. **Defazio (arXiv:2506.02285, 2025)** — WD controls gradient-to-weight ratio ||g||/||w||; all normalized layers converge to same steady-state ratio ("layer balancing"). Proposes corrective term for LR-schedule interaction. **No public code, but the corrective term is simple to implement (~5 lines).**

7. **Kosson et al. (arXiv:2510.19093, 2025)** — WD matters more than muP for LR transfer in practice. Shows alignment assumptions in muP fail after initial training. **Important context for why WD dynamics matter at scale.**

8. **Loshchilov (arXiv:2311.11446, 2023)** — Weight Norm Control (AdamWN). Generalizes decoupled WD to target-norm control. Target=0 recovers standard WD. **No public code, but target-norm WD is ~10 lines to implement.**

9. **AlphaDecay** (GitHub: hed-ucas/AlphaDecay, Apache-2.0) — Module-wise decay via spectral heavy-tailedness. LLM-specific but the spectral analysis tools are reusable. **Has usable code.**

10. **dyunis/spectral_dynamics** (GitHub, MIT) — Singular value tracking during training across architectures. **Has usable code for spectral analysis.**

11. **Schedule-Free optimization** (GitHub: facebookresearch/schedule_free) — Eliminates LR scheduling entirely. Weight decay handling built-in. Important comparison baseline—if schedule-free can handle WD implicitly, our scheduling arguments need to account for this. **Has usable code.**

12. **NOVAK** (arXiv:2601.07876, 2026) — Unified optimizer integrating 5 techniques. Shows coupling WD with effective LR (not base LR) degrades generalization 4-8pp on CIFAR-100. **Critical negative result: naive WD-LR coupling hurts.**

### Landscape Summary

From an engineering standpoint, the weight decay landscape looks like this:

**What actually works in practice:**
- AdamW with constant WD (0.01-0.1) is the default that beats most alternatives when properly tuned.
- CWD (ICLR 2026) provides consistent small improvements (0.1-0.5% accuracy, 0.01-0.08 loss) with zero hyperparameter overhead—a one-line change.
- SWD (NeurIPS 2023) helps close the SGD-Adam generalization gap on CIFAR but hasn't demonstrated convincing gains on modern large-scale setups.
- For LLMs, simply using higher WD with constant LR can outperform AdamW without WD (Benchmarking Optimizers, 2025).

**What doesn't work or is unproven:**
- Per-parameter adaptive WD (AdaDecay, 2019) is outperformed by simple well-tuned AdamW at GPT-2 scale (per AlphaDecay 2025 comparison).
- AdamO (2026) has interesting theory (radial/tangential decomposition) but no public code and limited large-scale validation.
- Complex multi-component approaches (NOVAK's 5-in-1 optimizer) are harder to debug, tune, and maintain.

**The engineering gap that matters:**
There is no comprehensive, controlled comparison of WD methods on the same codebase, same hyperparameter budget, same evaluation protocol. Every paper compares against a different subset of baselines with different experimental setups. A practitioner reading these papers cannot determine which method to use.

---

## Phase 2: Initial Candidates

### Candidate A: Unified WD Benchmark + Diagnostic Toolkit

- **Hypothesis**: A systematic, controlled comparison of 6-8 WD methods on a shared codebase will reveal that the differences between methods are smaller than reported, with most gains attributable to implicit hyperparameter tuning interactions rather than the WD method itself. The exception will be architectures without batch normalization, where alignment-aware methods genuinely help.
- **Implementation sketch**: Fork `tml-epfl/why-weight-decay`. Implement CWD (1 line), SWD (use existing AdamS code), cosine WD schedule (5 lines), AdamWN target-norm (10 lines), Defazio's corrective term (5 lines), and our proposed continuous alignment-modulated WD (20 lines). Run all on CIFAR-10/100 (ResNet-20, VGG-16-BN, VGG-16-no-BN) and ImageNet (ResNet-50). Track weight norms, gradient-weight alignment, gradient-to-weight ratio, spectral dynamics—all per layer.
- **Simplest version**: Just implement CWD, SWD, and constant WD on ResNet-20/CIFAR-10 with full diagnostic tracking. This alone takes ~2 hours compute and generates the core comparison.
- **Time estimate**: CIFAR experiments: ~4 GPU-hours total. ImageNet ResNet-50: ~20 GPU-hours per method x 6 methods x 3 seeds = ~360 GPU-hours. With 4 GPUs, ~4 days wall-clock for the full suite.
- **Reusable components**: why-weight-decay codebase (training loop, norm tracking), standard PyTorch optimizers, torchvision datasets/models.

### Candidate B: Continuous Alignment-Modulated WD (Practical Version)

- **Hypothesis**: Replacing CWD's binary sign-alignment mask with a continuous cosine-similarity modulation—lambda_t(i) = lambda_0 * max(0, cos(w_i, g_i))—will provide strictly better performance than CWD because it applies proportional decay rather than all-or-nothing, especially in early training when alignment is noisy.
- **Implementation sketch**: Modify CWD's mask from binary sign(w)*sign(update) > 0 to a soft version: mask = sigmoid(alpha * cos(w, g)), where alpha controls sharpness (alpha -> infinity recovers CWD). This is still a drop-in one-liner, just with a different mask function.
- **Simplest version**: Implement the soft mask in AdamW on ResNet-20/CIFAR-10. Compare against CWD and constant WD. Track the alignment distribution to see whether continuous modulation makes a measurable difference.
- **Time estimate**: ~1 GPU-hour for the pilot (3 methods x 1 seed x ~15 min each).
- **Reusable components**: Standard PyTorch AdamW, CWD implementation.

### Candidate C: "Strong Baseline Done Right" — Properly Tuned Constant WD with Diagnostic Explanation

- **Hypothesis**: A properly tuned constant WD, combined with the gradient-to-weight ratio corrective term from Defazio (2025), will match or beat all dynamic WD methods on standard benchmarks. The "paper" contribution would be showing this empirically and explaining *why* via the Lyapunov certified band analysis from the theoretical perspective—the band is narrow for BN architectures, so constant WD is near-optimal.
- **Implementation sketch**: Implement the corrective WD term (lambda_corrected = lambda * (1 - gamma_dot/gamma * something)), tune constant WD carefully (grid over 6-8 values), and compare against CWD/SWD/cosine on CIFAR/ImageNet. The theoretical contribution is showing the certified band width is narrow for BN networks, explaining why all methods converge to similar performance.
- **Simplest version**: Grid search WD in {0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2} on ResNet-20/CIFAR-10 with and without BN, and show the performance spread.
- **Time estimate**: ~2 GPU-hours for the CIFAR pilot.
- **Reusable components**: Standard PyTorch, no special libraries needed.

---

## Phase 3: Self-Critique

### Against Candidate A (Unified Benchmark)

- **Implementation reality check**: The why-weight-decay codebase exists and runs. CWD is literally one line. SWD has a public implementation. The diagnostic tracking (norms, alignment, spectra) adds compute overhead but is standard. **Feasible within the time budget.**
- **Reproducibility attack**: Using a single shared codebase with fixed seeds eliminates the main reproducibility concern. The risk is hyperparameter selection fairness: each method has a "sweet spot" and we need to tune each fairly. Mitigation: use the same HP budget (e.g., same number of grid search evaluations) for each method.
- **Baseline sanity check**: The strongest baseline is well-tuned constant AdamW. CWD (ICLR 2026) already showed only 0.1-0.5% improvement over this baseline. If we confirm this finding, the "benchmark" paper is less exciting—it would essentially say "nothing beats well-tuned constant WD by much." That could be a valuable negative result, but it's hard to publish as a top venue paper.
- **Scope attack**: CIFAR-10/100 and ImageNet are standard, but the LLM community increasingly cares more about language modeling. Without LLM experiments, reviewers may question generality. Mitigation: add one GPT-2 experiment using NanoGPT as a lightweight validation.
- **Verdict**: **MODERATE** — Valuable engineering contribution but may lack the theoretical novelty for a top venue unless combined with strong theoretical insights (e.g., the certified band analysis explaining *why* methods perform similarly).

### Against Candidate B (Continuous Alignment WD)

- **Implementation reality check**: Trivial to implement. The cos(w, g) computation is already available in any framework. The sigmoid soft mask is straightforward. **No engineering risk.**
- **Reproducibility attack**: The method introduces one new hyperparameter (alpha, the sharpness). This is manageable—we can show robustness over a range and also derive it from theory. **Low risk.**
- **Baseline sanity check**: CWD (ICLR 2026) already tried ablating the sign-alignment mask against random masks and gradient-based masks. The gradient-based mask performed poorly (loss 2.82 vs CWD's 2.56 on their LM benchmark). This is a WARNING: simply using gradient magnitude information instead of sign information hurt performance in CWD's experiments. Our continuous cosine modulation is different (it uses the alignment angle, not magnitude), but this negative result demands careful attention.
- **Scope attack**: If continuous modulation only provides marginal improvement over CWD's binary mask, the contribution is too thin for a standalone paper. It needs to be part of a larger framework (e.g., the unified theory showing continuous alignment as the natural interpolation between methods).
- **Verdict**: **MODERATE** — The CWD ablation showing gradient-based masks hurt is concerning. The continuous cosine approach is different but needs empirical validation before we can be confident. Best used as one component of a larger proposal, not standalone.

### Against Candidate C (Strong Baseline Done Right)

- **Implementation reality check**: Grid search of constant WD is trivial. The corrective term from Defazio is simple. **No engineering risk.**
- **Reproducibility attack**: Grid search is maximally reproducible. **No risk.**
- **Baseline sanity check**: This IS the baseline. The question is whether it beats everything else. If it does, that's a publishable finding ("Revisiting Weight Decay: Why Constant WD is Hard to Beat"). If it doesn't, the paper has no contribution.
- **Scope attack**: A "strong baseline" paper needs to be convincing that the baseline was truly optimized and the comparisons are truly fair. This requires extremely thorough experiments—more than the other candidates. The theoretical explanation (narrow certified band for BN architectures) adds substance.
- **Verdict**: **MODERATE** — Publishable if the empirical story is compelling and the theoretical explanation (narrow band for BN) is novel. But the risk of being scooped by any future benchmark paper is high.

---

## Phase 4: Refinement

### Dropped ideas

None fully dropped—all three have merit but none stands alone for a top venue paper.

### Strengthened combination: Unified Diagnostic Framework + Certified Band Theory

The strongest paper combines elements of all three candidates:

1. **From Candidate A**: The systematic benchmark on a shared codebase, with comprehensive diagnostic tracking.
2. **From Candidate B**: The continuous alignment-modulated WD as *one instantiation* within the unified framework.
3. **From Candidate C**: The "why constant WD is hard to beat" narrative, backed by the certified band analysis.

The key engineering insight that ties everything together: **the certified band width (from the theoretical perspective) is an empirically measurable quantity that predicts when dynamic WD methods will and won't help.** This is both a theoretical contribution and a practical diagnostic tool.

### Additional searches

Confirmed: No existing paper provides a controlled comparison of CWD, SWD, constant WD, cosine WD, and norm-matched WD on the same codebase. This gap is real.

Confirmed: The why-weight-decay codebase tracks weight norms and gradient norms but does NOT track gradient-weight alignment (cos(w,g)). We need to add this tracking—trivial but necessary.

Confirmed: PyTorch has no built-in weight decay scheduler. A clean WD scheduler API would be a useful software contribution alongside the paper.

### Selected front-runner

**Unified Dynamic Weight Decay: A Diagnostic Framework with Certified Convergence Bands**

This combines:
- Comprehensive benchmark (8 WD methods, shared codebase, fair HP budget)
- Certified band analysis as the theoretical backbone (explaining when and why methods differ)
- Diagnostic toolkit (alignment, norms, spectral dynamics, band width)
- One novel method: continuous alignment-modulated WD as the natural generalization of CWD
- Engineering artifact: PyTorch WD scheduler API

---

## Phase 5: Final Proposal

### Title

Unified Dynamic Weight Decay: When Does Adaptive Scheduling Actually Help?

### Hypothesis

For architectures with batch normalization and standard training setups, the Lyapunov-certified convergence band for weight decay is narrow, and all major WD methods (constant, cosine, SWD, CWD) fall within it—explaining empirically observed small performance differences. For architectures without BN or under non-standard training regimes (high WD, long training, non-adaptive optimizers), the band widens and alignment-aware methods provide genuine benefits.

### Motivation

Practitioners face decision paralysis: 15+ WD methods published since 2023, each claiming improvements on different benchmarks with different evaluation setups. No existing paper provides:
1. A controlled comparison on a shared codebase with fair hyperparameter budgets
2. A theoretical explanation for why most methods perform similarly in standard setups
3. A diagnostic criterion that predicts when dynamic WD is worth the complexity

### Method

**Step 1: Implement WD Methods on Shared Codebase** (days 1-3)
- Fork `tml-epfl/why-weight-decay` (NeurIPS 2024, MIT license)
- Implement all methods as pluggable WD schedulers:
  - Constant WD (baseline)
  - Cosine WD decay (tied to LR schedule)
  - SWD (gradient-norm-aware, from `zeke-xie/stable-weight-decay-regularization`)
  - CWD (sign-alignment binary mask, one-line implementation)
  - Continuous Alignment WD (our proposal: lambda_i(t) = lambda_0 * sigmoid(alpha * cos(w_i, g_i)))
  - AdamWN-style target-norm WD
  - Defazio corrective WD term
  - (Optional) AlphaDecay module-wise WD

**Step 2: Add Diagnostic Tracking** (days 2-3)
- Per-layer gradient-weight cosine similarity (alignment delta_t)
- Per-layer weight norms and gradient norms
- Gradient-to-weight ratio ||g||/||w|| per layer
- Effective WD strength (lambda_effective = lambda * decay_mask_fraction)
- Certified band width estimate: lambda_max(t) - lambda_min(t) per the Lyapunov analysis

**Step 3: Run Experiments** (days 3-7)
- CIFAR-10: ResNet-20 (with BN), VGG-16 (with and without BN) — 3 seeds each
- CIFAR-100: ResNet-20, VGG-16-BN — 3 seeds each
- ImageNet: ResNet-50 — 3 seeds each (longer, parallelized across GPUs)
- WD grid: {0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2} for constant WD tuning
- Fair comparison: each dynamic method gets same HP budget (tune its method-specific params with same number of trials)

**Step 4: Theoretical Analysis** (parallel with experiments)
- Compute certified band [lambda_min(t), lambda_max(t)] for each architecture/dataset
- Verify that all implemented methods fall within the band
- Characterize band width as function of BN presence, layer depth, training phase
- Prove continuous alignment WD is the smooth interpolation within the certified band

**Step 5: Visualization and Analysis** (day 7)
- Phase portraits: (||w||, ||g||/||w||, alignment) trajectories per method
- Certified band width evolution over training
- Per-method effective WD trajectories
- Correlation: band width vs. method performance spread

### Simplest version

Run CWD, SWD, constant WD, and our continuous alignment WD on ResNet-20/CIFAR-10 (with and without BN). Track alignment and norms. Compute certified band width. This takes ~2 GPU-hours and produces the core story: narrow band + BN = all methods equivalent; wide band + no BN = alignment-aware methods win.

### Baselines

1. **Constant WD (grid-tuned)**: The strongest simple baseline. Expected accuracy: ~92-93% on CIFAR-10 ResNet-20.
2. **CWD (ICLR 2026)**: The current SOTA alignment-aware method. Expected: ~0.1-0.3% above constant WD.
3. **SWD (NeurIPS 2023)**: The current SOTA scheduling method. Expected: comparable to well-tuned constant WD.
4. **No WD**: Control. Expected: 1-3% below constant WD on CIFAR.

### Experimental plan

| Phase | Architecture | Dataset | Methods | Seeds | Est. GPU-hours |
|-------|-------------|---------|---------|-------|----------------|
| Pilot | ResNet-20 (BN) | CIFAR-10 | 4 methods | 1 | 1 |
| Pilot | VGG-16 (no BN) | CIFAR-10 | 4 methods | 1 | 1 |
| Core | ResNet-20, VGG-16-BN, VGG-16-noBN | CIFAR-10 | 7 methods | 3 | 20 |
| Core | ResNet-20, VGG-16-BN | CIFAR-100 | 7 methods | 3 | 20 |
| Scale | ResNet-50 | ImageNet | 5 methods | 3 | 200 |
| **Total** | | | | | **~242** |

With 4x RTX PRO 6000 GPUs, the CIFAR experiments complete in ~1 day, ImageNet in ~3 days.

Ablation schedule:
1. BN vs no-BN (the key diagnostic variable)
2. WD strength sweep (how does band width change with WD magnitude?)
3. Alignment sharpness alpha in continuous alignment WD (convergence to CWD as alpha -> infinity)
4. Training phase breakdown (early/mid/late: where does alignment modulation matter most?)

### Resource estimate

- Total GPU-hours: ~242 (CIFAR: 42, ImageNet: 200)
- Wall-clock with 4 GPUs: ~5 days for full experiment suite
- Pilot (first useful results): ~2 hours
- Models: ResNet-20 (0.27M params), VGG-16 (138M params), ResNet-50 (25M params)
- Software: PyTorch 2.x, torchvision, standard CIFAR/ImageNet datasets
- No external compute needed: all runs on local 4x RTX PRO 6000

### Risk assessment

1. **All methods perform within noise on BN architectures** (Probability: HIGH). This is actually our predicted finding. The paper's contribution becomes the explanation (narrow certified band) rather than a new SOTA method. Risk: reviewers may want a method that wins, not an explanation of why nothing wins.
   - Mitigation: Frame as "diagnostic framework" paper. Show the certified band width predicts when methods differentiate, validated on non-BN architectures where differences are meaningful.

2. **Continuous alignment WD underperforms CWD** (Probability: MODERATE). CWD's ablation showed gradient-based masks hurt. Our cosine-based continuous mask is different but might suffer from the same issue.
   - Mitigation: If it underperforms, this is a finding we report honestly. The paper's value doesn't depend on our method winning—it depends on the diagnostic framework.

3. **ImageNet experiments take longer than estimated** (Probability: LOW). ResNet-50 training is well-characterized; 200 GPU-hours across 5 methods x 3 seeds is conservative.
   - Mitigation: Start ImageNet early, run 24/7. CIFAR results alone are publishable.

4. **Certified band computation is too expensive** (Probability: LOW). The band depends on L-smoothness, gradient variance, and current norms—all cheaply computable.
   - Mitigation: Use per-epoch estimates rather than per-step.

5. **Library compatibility issues** (Probability: LOW). We're using standard PyTorch with standard models and datasets.
   - Mitigation: Use the why-weight-decay codebase which is already validated.

### Novelty claim

The novelty is NOT in proposing a new WD method that beats everything. The novelty is in:

1. **First controlled comparison** of 7+ WD methods on a single codebase with fair HP budgets and comprehensive diagnostics. No existing paper does this.

2. **Certified band width as a diagnostic metric**: The Lyapunov-certified band width is a new, theoretically grounded quantity that predicts when dynamic WD methods will provide genuine benefit. This is practically useful for practitioners deciding whether to invest in WD tuning.

3. **Continuous alignment WD as natural generalization of CWD**: Even if it doesn't beat CWD, showing it as the smooth limit of the sign-alignment mask within the certified band framework provides theoretical closure.

4. **Empirical validation of the "narrow band" conjecture**: Showing that BN architectures have narrow certified bands, explaining the puzzling finding that 15+ WD methods all perform similarly on standard benchmarks.

5. **PyTorch WD scheduler API**: A clean software contribution that makes it trivial to plug in any WD scheduling strategy.

This is a "understanding and explaining" paper in the vein of D'Angelo et al. (NeurIPS 2024, "Why Do We Need Weight Decay?"), not a "new method beats SOTA" paper. That paper got into NeurIPS with strong empirical analysis and clear explanations. We aim for the same venue type with deeper theoretical grounding.

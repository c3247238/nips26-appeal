# Empiricist Perspective — Iteration 6

## Role
Meticulous experimental scientist. Every claim must be backed by evidence. Suspicious of unfounded generalizations. Null results demand the same evidential rigor as positive results — or higher, since the burden of proving non-significance is greater. This iteration has new in-progress experiments: the empiricist's job is to assess the full experimental evidence picture, identify remaining gaps, and define the minimum experiments needed before the paper can be credibly submitted.

---

## Phase 1: Literature Survey — Methodology and Evaluation

### 1.1 Resources Focused on Experimental Methodology and Evaluation

1. **Sun et al., "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD" (CVPR 2025)** — Core theoretical reference. Proves WD does NOT accelerate convergence but improves generalization in nonconvex SGD via stability. Theorem holds for fixed λ only; provides the formal stability framework that Theorem 1 of our paper extends to binary-masking schedules. Methodological constraint: their analysis is SGD-only; any claim extending to AdamW requires separate justification.

2. **D'Angelo et al., "Why Do We Need Weight Decay in Modern Deep Learning?" (NeurIPS 2024)** — Landmark unifying perspective. Two separate mechanisms: "loss stabilization" for SGD (relevant to our SGD experiments) and "bias-variance tradeoff" for LLM single-pass training (outside our scope). Key evaluation insight: CIFAR-10 with BatchNorm is the regime where WD matters LEAST as explicit regularization. This directly contextualizes why our AdamW method range (0.25%) is near-zero — not because we found an important null result, but potentially because CIFAR-10+BN is pathologically insensitive to any WD variation. We must test in harder regimes.

3. **Chen et al., "Cautious Weight Decay (CWD)" (ICLR 2026)** — Directly competitive. CWD is tested on ImageNet ResNet-50/ViT/DeiT with grid-searched baselines. Key methodological concern: CWD's reported improvements use per-method hyperparameter tuning, while our benchmark uses fixed hyperparameters — this is not a fair comparison. Our "CWD underperforms constant under fixed hyperparameters" result is a distinct claim from "CWD underperforms constant in general."

4. **Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, ICLR 2019)** — Establishes ρ = wd/lr as the critical compound hyperparameter for adaptive optimizers. Experiments at ρ ≈ 0.1-1.0 on ImageNet. Our existing experiments have AdamW at ρ = 0.5 (5e-4/1e-3) and SGD at ρ = 0.005 (5e-4/0.1) — a 100× difference that is the major uncontrolled confound in our SGD vs. AdamW comparison.

5. **Wang & Aitchison, "How to Set AdamW's Weight Decay as You Scale" (arXiv:2405.13698)** — ρ determines EMA timescale; near-optimal ρ is architecture- and dataset-scale-dependent. Provides the theoretical foundation for why matched-ρ experiments are essential: at ρ=0.5, nearly all normalized layers are in their "stable sphere" where WD method choice is nearly irrelevant because the weight norm equilibrium is already dominant.

6. **Wightman et al., "ResNet Strikes Back" (arXiv:2110.00476)** — Best practice for multi-seed evaluation on ImageNet. Top-1 differences < 0.2% are statistically non-significant even with N=10. At N=2-3, only differences > 0.5-1.0% are detectable. Constrains how we interpret any ImageNet results if/when they become available.

7. **Galanti et al., "SGD and Weight Decay Secretly Minimize Rank" (arXiv:2206.05794)** — WD promotes rank minimization via spectral dynamics. The NoBN architecture removes BN scale-invariance, making weight norms and spectral properties more directly responsive to WD variation. This predicts that NoBN should show larger method sensitivity — directly testable by our current NoBN experiment (77 epochs complete for seed=42 on constant only, with 7 methods still running).

8. **Defazio, "Why Gradients Rapidly Increase Near the End of Training" (arXiv:2506.02285)** — WD drives gradient-to-weight ratio ‖g‖/‖w‖ of all BN layers to the same steady state ("layer balancing"). This provides a unifying lens: at steady state (which BN+AdamW reaches quickly), ‖g‖/‖w‖ is identical across WD methods (up to second-order terms). This EXPLAINS why our weight norm trajectories converge to near-identical values (95.9-97.0) under AdamW — the steady-state law makes all methods equivalent at equilibrium.

9. **Kuzborskij & Abbasi-Yadkori, "Low-rank bias, weight decay, and model merging" (arXiv:2502.17340)** — L2 regularization at stationary points induces parameter-gradient alignment, norm preservation, and low-rank bias. Key insight: at stationary points, alignment IS preserved regardless of WD strength — meaning AIS measured at the end of training (as we do) is not a signal of dynamic WD benefit but rather a static property of the final weight configuration.

10. **Naganuma et al., "What do near-optimal learning rate schedules look like?" (arXiv:2603.10301, 2026)** — WD has strong effect on optimal LR schedule shape. This is a hidden confound in our design: we use cosine LR for all methods, but the optimal LR for cosine_schedule WD (which also decays WD with the LR) may differ from the optimal LR for constant WD. Our fixed-hyperparameter protocol may systematically disadvantage or advantage cosine_schedule WD.

11. **NOVAK (Kavun, arXiv:2601.07876, 2026)** — Shows coupling WD with effective LR (vs. base LR) degrades CIFAR-100 by 4-8 percentage points. Validates our finding that WD should be decoupled from the optimizer's per-parameter adaptive scaling. However, NOVAK uses per-method hyperparameter optimization and achieves SOTA across 14 optimizers — our fixed-hyperparameter protocol is deliberately more conservative.

12. **Truong & Truong, "Norm-Hierarchy Transitions in Representation Learning" (arXiv:2603.07323, 2026)** — WD traverses norm hierarchy from shortcut to structured representations; transition delay is logarithmic in norm ratio. This means measuring WD effects at epoch 200 may miss transitions that occur over thousands of epochs in larger models. For ResNet-20/CIFAR-10, 200 epochs may be insufficient to observe the full WD effect trajectory.

### 1.2 Experimental Landscape

**What has been properly tested by the field:**
- AdamW vs. Adam+L2 decoupling: rigorously established (Loshchilov 2019)
- CWD sign-alignment masking on ImageNet with per-method tuning: positive, small effects (CWD 2026)
- SWD gradient-norm scheduling on CIFAR: small positive effects (Xie 2023)
- WD mechanism under SGD (stability) vs. under Adam (bias-variance): theoretical (D'Angelo 2024, Sun 2025)

**What has been accepted without proper controls:**
- CWD's claimed superiority: tested with tuned hyperparameters for CWD but fixed baselines — violates hyperparameter fairness
- The claim that WD method insensitivity under AdamW is due to BN scale-invariance: never directly tested (no NoBN ablation in any paper)
- Whether the ρ ratio (not AdamW mechanism per se) drives method insensitivity: the 100× ρ confound between our SGD and AdamW conditions has been flagged in prior iterations but not yet resolved by completed experiments
- Whether our CIFAR-10 null result under AdamW generalizes to other architectures (VGG) or scales (ImageNet)

**Where methodological gaps remain as of Iteration 6:**
- VGG-16-BN full experiment: running (seed=42: 200ep done, seed=123: 78ep done, other methods not yet started — only constant being tested currently because the experiment is sequential)
- Matched-ρ SGD: pilot only completed (seed=42, 5ep, rho=0.5); seed=123 at 67 epochs (constant only)
- Rho sweep (rho_low=0.05, rho_high=5.0): pilot done, full running (seed=42/constant at 67 epochs for rho_low)
- NoBN ablation: full running (seed=42/constant at 77 epochs)
- ImageNet ResNet-50: pilot FAILED; no ImageNet data available

---

## Phase 2: Initial Candidates

### Candidate A: "The Multi-Architecture Confirmation at Standard ρ"

**Core hypothesis (precisely falsifiable):** Under AdamW with ρ=0.5, WD method range across {constant, CWD, SWD, cosine_schedule, half_lambda, random_mask, no_wd} on VGG-16-BN/CIFAR-10 will be less than 0.5 percentage points, replicating the ResNet-20 finding in an architecture without residual connections.

**Falsification criterion (pre-committed):**
- FALSIFIED if any WD method differs from constant by > 0.5% with p < 0.05 (unpaired t-test, N=3)
- FALSIFIED if method range exceeds 1.0% (regardless of statistical significance)
- CONFIRMED if TOST equivalence holds at δ=±1.0% for all 6 non-constant methods (N=3, alpha=0.05)

**Evaluation protocol:**
- Architecture: VGG-16-BN (currently running: 1 method × 2 seeds done or in-progress)
- Dataset: CIFAR-10
- Optimizer: AdamW (lr=1e-3, wd=5e-4, ρ=0.5), cosine LR
- WD methods: constant, cwd_hard, half_lambda, cosine_schedule, swd, random_mask, no_wd (7 total)
- Seeds: 42, 123, 456 (N=3)
- Statistical tests: one-way ANOVA; Dunnett's test vs. constant (Holm-Bonferroni corrected); TOST at δ=±1.0%

**Ablation plan:**
1. Primary: compare method ranges (VGG vs. ResNet-20)
2. Secondary: compare weight norm spread (VGG high weight norms ~651 for constant)
3. Tertiary: compare AIS trajectories (VGG shows lower AIS at 200ep: ~0.175 vs. ResNet-20 ~0.336)

**Confounders identified:**
- VGG has 15M parameters vs. ResNet-20's 270K — much larger weight norm absolute values (651 vs. 96). Weight norm differences between methods may be proportionally larger even if accuracy differences are not.
- VGG with cosine LR (lr=1e-3, no warmup) reached only 92.03% (seed=42) vs. standard ResNet-20 at 90.13%. The higher accuracy may indicate the model is closer to convergence where WD method choice becomes irrelevant.
- VGG uses different gradient flow patterns (deep sequential conv layers vs. residual shortcuts) — the theoretical basis for method insensitivity under BN may differ between architectures.

**Pilot design:** VGG seed=42, constant only, 200 epochs — COMPLETE (92.03%). Ready to proceed.

**Resource estimate:** 7 methods × 3 seeds = 21 runs × ~13 min each (VGG seed=42 took 797s ≈ 13 min). ~4.5 GPU-hours. Currently only constant is running (1 GPU); remaining 6 methods need separate GPU allocation.

---

### Candidate B: "The ρ-Regime Map — Does Method Sensitivity Emerge at Extreme ρ?"

**Core hypothesis (precisely falsifiable):** WD method sensitivity is a function of ρ = wd/lr. At ρ_low = 0.05 (rho_low config: lr=1e-3, wd=5e-5), method range will be < 0.5%. At ρ_high = 5.0 (rho_high config: lr=1e-3, wd=5e-3), method range will be > 0.5% (possibly > 1.0%). This tests the existence of a critical ρ* threshold separating the insensitive from sensitive regimes.

**Falsification criterion (pre-committed):**
- FALSIFIED (ρ-regime hypothesis) if rho_high (ρ=5.0) shows method range < 0.5%: ρ does NOT drive sensitivity even at high regularization
- FALSIFIED (ρ-regime hypothesis) if rho_low (ρ=0.05) shows method range > 0.5%: low ρ does NOT guarantee insensitivity
- CONFIRMED if rho_high range > rho_low range by > 2× (indicating ρ-dependent sensitivity)
- Additional test: if CWD outperforms constant at rho_high (ρ=5.0), Theorem 1 Corollary is confirmed (high ρ → stability cost → 0 → alignment-aware WD becomes optimal)

**Evaluation protocol:**
- Architecture: ResNet-20, Dataset: CIFAR-10, Optimizer: AdamW
- rho_low condition: lr=1e-3, wd=5e-5 (ρ=0.05), methods: constant, cwd_hard, half_lambda, no_wd
- rho_high condition: lr=1e-3, wd=5e-3 (ρ=5.0), methods: constant, cwd_hard, half_lambda, no_wd
- Seeds: 42, 123, 456 (N=3)
- Statistical tests: ANOVA; pairwise t-tests vs. constant; TOST at δ=±0.5%

**Ablation plan:**
1. Compare method ranges at 3 ρ values: {0.05, 0.5 (existing), 5.0}
2. Track weight norm trajectories: at rho_low, weight norms should grow unbounded (minimal regularization); at rho_high, weight norms should converge rapidly to a small value
3. Check AIS at each ρ: if AIS is predictive of method advantage only at high ρ, this validates AIS as a ρ-conditional metric

**Confounders:**
- At rho_high (ρ=5.0), the model may underfit (heavy regularization → low accuracy). If constant-WD itself underperforms (< 85% on CIFAR-10 with AdamW), then "CWD beats constant" might reflect "CWD reduces effective WD below its optimal level" not "alignment-aware WD is beneficial." The rho_high pilot (5 epochs, best_acc=77.69%) is inconclusive about final accuracy.
- At rho_low (ρ=0.05), the method range might be large simply because no_wd performs similarly (both have very low WD effect). The comparison constant vs. no_wd may have near-zero effect, artificially compressing the method range from the bottom.
- Neither rho_low nor rho_high uses the same ρ as the original SGD experiments (ρ=0.005), so the ρ sweep only partially fills the gap.

**Pilot design:** rho_low constant (5ep, acc=79.06%), rho_high constant (5ep, acc=77.69%) — both stable. Full runs in progress (seed=42/constant at 66-67 epochs).

**Resource estimate:** 4 methods × 3 seeds × 2 ρ conditions = 24 runs × ~40 min each = ~16 GPU-hours. One GPU used, currently at ep 67/200 for rho_low/constant/seed=42.

---

### Candidate C: "The BatchNorm Ablation — Does NoBN Restore WD Method Sensitivity?"

**Core hypothesis (precisely falsifiable):** BatchNorm's weight scale-invariance is the sufficient cause of WD method insensitivity under AdamW. Removing BN (ResNet-20-NoBN) should produce a method range > 0.5% under AdamW with the same ρ=0.5, compared to the < 0.25% range with BN.

**Falsification criterion (pre-committed):**
- FALSIFIED (BN hypothesis) if ResNet-20-NoBN shows method range < 0.25% under AdamW: BN is NOT the proximate cause of insensitivity
- CONFIRMED if ResNet-20-NoBN shows method range > 0.5%: BN scale-invariance IS the sufficient cause
- Intermediate (0.25%–0.5%): inconclusive; both BN and AdamW's ℓ∞ implicit bias are partial causes
- Critical secondary check: NoBN AIS at training end should be higher than BN AIS if the alignment signal genuinely provides information when BN scale-invariance is removed (NoBN pilot shows AIS=0.49 vs. BN AIS=0.34, suggesting alignment IS more informative without BN — good sign for the BN hypothesis)

**Evaluation protocol:**
- Architecture variants: (a) ResNet-20-BN (existing data), (b) ResNet-20-NoBN (running)
- Dataset: CIFAR-10
- Optimizer: AdamW (lr=5e-4 for NoBN stability, wd=5e-4 → ρ=1.0 for NoBN vs. ρ=0.5 for BN)
- WD methods: constant, cwd_hard, no_wd (3 methods for NoBN; note NoBN experiment only tests these 3)
- Seeds: 42 (N=1 for NoBN — significant statistical limitation)
- Statistical test: effect size comparison (Cohen's d) between BN and NoBN

**Ablation plan:**
1. NoBN pilot (5ep): constant=57.64%, cwd_hard=56.96%, no_wd=57.02% → range = 0.68% — already shows larger spread than BN pilot at same 5 epochs (BN pilot range was much smaller)
2. NoBN full (77ep for constant/seed=42): acc=87.09%, AIS=0.573 (substantially higher than BN AIS~0.34)
3. Compare full 200ep results across 3 methods when available

**Confounders (persistent, not solved):**
- NoBN training uses different LR (5e-4 vs. 1e-3 for BN) to maintain stability, changing ρ from 0.5 to 1.0 — this is NOT a matched comparison and confounds the NoBN vs. BN effect size
- NoBN accuracy is substantially lower (~86-87% at 77ep vs. ~90% for BN at full training) — different operating regimes make accuracy comparisons misleading
- N=1 for NoBN full experiments provides no statistical power; any observed method range cannot be tested for significance

**Pilot design:** 5-epoch NoBN pilot COMPLETE: constant=57.64%, cwd_hard=56.96%, no_wd=57.02% (range=0.68%, higher than BN but only 5 epochs). Full run in progress.

**Resource estimate:** 3 methods × 1 seed × 200 epochs = ~30-40 GPU-hours for NoBN. Currently running.

---

### Candidate D: "The Matched-ρ SGD Test — Is ρ Sufficient to Explain the 3.7× Sensitivity Ratio?"

**Core hypothesis (precisely falsifiable):** The 3.7× larger method sensitivity ratio under SGD vs. AdamW (0.913% vs. 0.250%) is explained by the 100× difference in ρ (SGD: 0.005, AdamW: 0.5). At matched ρ=0.5 under SGD (lr=0.01, wd=5e-3), method sensitivity should collapse to AdamW-like levels (< 0.5% range).

**Falsification criterion (pre-committed):**
- FALSIFIED (ρ sufficiency) if matched-ρ SGD (ρ=0.5) still shows method range > 0.5%: ρ alone does not explain the gap; AdamW's adaptive preconditioning has an additional role
- CONFIRMED if matched-ρ SGD range < 0.25%: ρ is sufficient; AdamW's ℓ∞ implicit bias is NOT an additional cause
- Additional check: weight norm spread at matched ρ. Under SGD ρ=0.005, weight norms ranged from 64.6 (constant) to 127.1 (no_wd) — a 97% spread. At ρ=0.5, they should converge much more tightly (prediction from Wang & Aitchison's EMA-timescale theory)

**Evaluation protocol:**
- Architecture: ResNet-20
- Dataset: CIFAR-10
- Optimizer: SGD (momentum=0.9, cosine LR, lr=0.01, wd=5e-3 → ρ=0.5)
- WD methods: constant, cwd_hard, no_wd (3 methods — as currently running)
- Seeds: 42 (5ep pilot complete), 123 (66ep in-progress), 456 (not yet started)
- Statistical power: N=3 seeds required; ONLY seed=42 pilot (5ep) and seed=123 (in-progress 66ep) are available now

**Ablation plan:**
1. Primary: method range at matched-ρ SGD vs. low-ρ SGD (existing) and AdamW
2. Secondary: weight norm spread comparison across ρ conditions
3. Critical check: at epoch 5 of matched-ρ SGD pilot, acc=76.12% (same as rho_high AdamW: 77.69%). Both models are far from convergence — need 200ep full results

**Confounders:**
- Cosine LR with SGD creates time-varying ρ(t) = wd/lr(t) → ∞ as lr → 0. The ρ=0.5 condition only holds at the initial LR; at epoch 100 the LR has decayed and ρ has grown substantially. This dynamic ρ confound also exists in AdamW, but the effects may differ.
- SGD with high wd=0.005 and cosine LR may cause weight norm collapse before the LR decreases sufficiently. The matched-ρ SGD pilot (5ep, acc=76.12%) shows training is stable but we need 200ep convergence data.
- SGD without BatchNorm normalization rescaling: unlike AdamW which has implicit ℓ∞ constraint, high-ρ SGD has no such implicit constraint — weights may grow or shrink in ways not controlled by BN.

**Pilot design:** 5-epoch matched-ρ SGD pilot COMPLETE (acc=76.12%, stable). Seed=123 in-progress at 66 epochs.

**Resource estimate:** 3 methods × 3 seeds × 200 epochs = 9 runs × ~8 min each = ~1.2 GPU-hours. Seed=42 pilot done (5ep only, not 200ep). Currently only seed=123/constant running on GPU.

---

## Phase 3: Self-Critique

### Against Candidate A (VGG-16-BN Architecture Confirmation)

**Confound attack:** The VGG-16-BN result (seed=42, constant, 200ep: 92.03%) already shows similar weight norm dynamics to ResNet-20 — weight norm grows from 121.9 to 651.5 over training, and AIS drops from 0.43 to 0.18. If all other WD methods also converge to weight norms near 651 under AdamW+BN (as ResNet-20 did to ~96), then the null result is pre-determined by BN+AdamW dynamics, not an experimental finding. We would be measuring the steady-state equilibrium (Defazio's layer-balancing law) rather than testing WD method sensitivity.

**Statistical attack:** With VGG seed=42 complete (12.7 min/run), the full 21-run VGG experiment requires ~4.5 GPU-hours. However, currently only the constant method is running (sequential execution). The other 6 methods are NOT currently running, meaning the VGG experiment will take ~6× longer than a parallel design. With N=3 seeds and expected within-method SD ~0.3% (similar to ResNet-20), MDE ≈ 0.61%, which cannot detect effects smaller than 0.61%. The VGG method range is expected to be < 0.25% (similar to ResNet-20 under same ρ), which is well below MDE.

**Benchmark attack:** VGG-16-BN at 92.03% on CIFAR-10 is already near ceiling — standard ResNet-56 achieves ~93.5%, and the VGG-16-BN's performance may be limited by the fixed 200-epoch + cosine-from-1e-3 schedule rather than the WD method. Method range measured at a near-ceiling accuracy point may be artificially compressed.

**Ablation completeness attack:** The current VGG run tests identical hyperparameters to ResNet-20 (lr=1e-3, wd=5e-4, ρ=0.5, cosine 200ep). If we find the same null result, it could be entirely explained by the identical ρ — we are not testing "VGG vs. ResNet" but "same ρ on different architecture." Without a cross-architecture ρ sweep, we cannot attribute any difference (or lack thereof) to architecture.

**Verdict: STRONG (for execution) / MODERATE (for interpretability).** The experiment should be run because reviewers expect multi-architecture evidence. But the interpretation must be careful: a null result on VGG confirms "same ρ = same insensitivity across BN architectures," not "WD method doesn't matter in general."

---

### Against Candidate B (ρ-Regime Map)

**Confound attack:** At rho_high (ρ=5.0, wd=5e-3), the model is in severe regularization territory. The rho_high pilot shows 77.69% after 5 epochs (lower than standard 80%+ for ResNet-20). If the final accuracy at ρ=5.0 is < 85% (plausible — standard wd=5e-3 may overfit or overregularize), then any method differences may reflect "which method least hurts accuracy under extreme regularization" rather than "which method provides the best alignment-aware trade-off." The constant WD at ρ=5.0 may itself be a bad baseline.

**Statistical attack:** 4 methods × 3 seeds = 12 runs per ρ condition, with MDE ≈ 0.51% (using pooled SD~0.25% from existing data). The ρ-driven sensitivity hypothesis predicts rho_high range > 1.0% (CWD should beat constant by ~1%). This is detectable with N=3. However, if the actual effect is smaller (e.g., rho_high range = 0.6%), we will see a positive trend but not a statistically significant result.

**Benchmark attack:** The rho_low (ρ=0.05) condition uses the same CIFAR-10 dataset where WD matters least (D'Angelo 2024). If rho_low shows a null result, it is not clear whether this is because "low ρ → insensitivity" or because "CIFAR-10 → insensitivity regardless of ρ." We need at least CIFAR-100 at rho_low to separate these.

**Ablation completeness attack:** The ρ sweep covers {0.05, 0.5, 5.0}. This is 3 points on a log scale, covering 2 orders of magnitude. The theoretical ρ* threshold (where method sensitivity emerges) is unknown. With only 3 points, we cannot fit a curve or estimate ρ* — we can only establish a rank ordering of sensitivity.

**Verdict: STRONG.** The ρ-regime map is the most scientifically important new experiment because it provides the mechanistic explanation for the SGD vs. AdamW sensitivity ratio (if the ρ hypothesis is correct). Even with limited statistical power at individual ρ values, the trend across 3 ρ conditions will be compelling. Execute in priority order: rho_high (most informative) → rho_low → intermediate ρ.

---

### Against Candidate C (BatchNorm Ablation)

**Confound attack:** NoBN experiment uses lr=5e-4 (not 1e-3) to maintain training stability, changing ρ from 0.5 to 1.0. The BN vs. NoBN comparison is therefore confounded by: (a) presence/absence of BN, (b) different ρ values, and (c) different LR (different gradient flow dynamics). These three confounds cannot be disentangled in the current design. Furthermore, NoBN at 77 epochs achieves 87.09% vs. BN at full training reaching 90.13%. The lower accuracy may reflect the harder optimization problem (no normalization), not a fundamentally different WD sensitivity regime.

**Statistical attack:** N=1 seed for NoBN (current design). A within-method SD of ~0.4-0.5% for NoBN training (higher variance expected without BN) means the method range needs to be > 1.0% to be meaningful with N=1. The 5-epoch pilot shows a method range of 0.68% (constant=57.64%, cwd_hard=56.96%, no_wd=57.02%) — this is inconclusive, especially since "no_wd" and "constant" differ by only 0.62% after 5 epochs.

**Benchmark attack:** ResNet-20-NoBN is a non-standard benchmark. No WD paper has published ResNet-20-NoBN results on CIFAR-10, making contextualization impossible. A reviewer familiar with the WD literature will not know whether 87% at epoch 77 represents a reasonable performance level.

**Ablation completeness attack:** The NoBN experiment tests only 3 of the 7 original WD methods (constant, cwd_hard, no_wd). This subset was chosen for efficiency, but it means the "method range" is defined over a different set than the BN experiments. The comparison of ranges across BN and NoBN will be methodologically inconsistent.

**Verdict: MODERATE-WEAK.** The NoBN ablation is conceptually important but fundamentally confounded by the LR/ρ change and the N=1 limitation. Continue to completion (currently running) but acknowledge the confounds explicitly. This experiment provides suggestive evidence, not conclusive results. Prioritize rho sweep (Candidate B) and matched-ρ SGD (Candidate D) over the NoBN ablation for writing priority.

---

### Against Candidate D (Matched-ρ SGD)

**Confound attack:** The matched-ρ SGD experiment uses cosine LR schedule with initial lr=0.01. Under cosine decay, the effective ρ(t) = wd/lr(t) grows as lr decreases: at epoch 100 (half of training), ρ(100) ≈ wd / (0.5 × lr_0) = 1.0; at epoch 190, ρ → ∞. This dynamic ρ is NOT the same as AdamW's dynamic ρ under the same schedule. Under AdamW, the adaptive preconditioning partially offsets the LR decay's effect on parameter update magnitude. Under SGD, there is no such offset. Therefore, "matched initial ρ" does not imply "matched effective regularization throughout training."

**Statistical attack:** N=3 seeds planned but only seed=42 pilot (5ep) and seed=123 (66ep in-progress) are running. Seed=456 not yet started. At the current rate, completion requires ~8 hours on a single GPU. With expected SGD-method SD of ~0.30%, MDE ≈ 0.61% at N=3. If matched-ρ SGD shows method range of 0.40% (between 0.25% and 0.91%), we cannot statistically distinguish it from AdamW or from low-ρ SGD. This is the most likely "inconclusive" outcome scenario.

**Benchmark attack:** SGD with lr=0.01 cosine and wd=5e-3 is non-standard for CIFAR-10. Standard SGD on CIFAR-10 uses lr=0.1 (10× higher). At lr=0.01, the gradient updates are much smaller, potentially leading to slower convergence and a different final accuracy region. Comparing accuracy between low-ρ SGD (lr=0.1, converges fast) and matched-ρ SGD (lr=0.01, converges slow) is problematic even at epoch 200.

**Ablation completeness attack:** The matched-ρ SGD tests only constant, cwd_hard, no_wd (3 methods). If no_wd performs catastrophically differently from constant at matched ρ (which is plausible — at ρ=0.5, no_wd allows weight norms to grow unconstrained under SGD without BN's scale-invariance protection), then the 3-method range will be driven by no_wd vs. constant, not by the alignment-aware methods. The "method range" statistic conflates no-regularization effects with WD-strategy effects.

**Verdict: STRONG (for conceptual importance) / MODERATE (for execution quality).** The matched-ρ SGD experiment is the single most important control for the paper's central 3.7× sensitivity claim. However, the confound from dynamic ρ (cosine LR) and the limited method set (3 of 7) weaken the interpretation. Recommended revision: add a constant-LR condition (lr=0.01, no cosine) for seed=42 only, to test static matched-ρ SGD where ρ genuinely remains constant throughout training.

---

## Phase 4: Refinement

### Dropped (from front-runner consideration)
**Candidate C (NoBN Ablation)** is running but should NOT be treated as a primary result. The LR/ρ confound and N=1 limitation make it unsuitable for strong claims. Continue to completion for completeness, but explicitly demote to "secondary analysis with unresolved confounds" in the paper.

### Strengthened

**Candidate A (VGG-16-BN) — Strengthened:**
- The VGG run is currently sequential (one method at a time on one GPU). The remaining 6 methods (cwd_hard, half_lambda, cosine_schedule, swd, random_mask, no_wd) should be parallelized across 6 additional GPUs immediately. This reduces total wall-clock from ~6 hours to ~1.5 hours.
- Add a critical intermediate check: if VGG AIS at epoch 200 is similar to ResNet-20 AIS (~0.18-0.34), then the AIS-insensitivity pattern extends beyond ResNet-20 — this validates AIS as an architecture-independent metric.
- If VGG method range > 0.5%: this would be a genuinely surprising positive finding — investigate which methods benefit and whether the benefit is related to VGG's lack of residual connections (different gradient propagation pattern modulates WD sensitivity).

**Candidate B (ρ-Regime Map) — Strengthened:**
- Add a "same-ρ AdamW baseline" at rho_low (ρ=0.05) and rho_high (ρ=5.0): run AdamW constant at these ρ values (1 seed each, quick) to confirm accuracy levels are acceptable before interpreting the method comparisons.
- Pre-commit to the following analysis: plot method range vs. log(ρ) for ρ ∈ {0.005 (existing SGD), 0.05 (rho_low), 0.5 (existing AdamW), 5.0 (rho_high)}. This "ρ-sensitivity curve" is the core visualization for the ρ-regime map contribution.
- The theoretical prediction from Theorem 1: method range ~ f(AIS) × (1 - stability_cost_dominant) should be testable if rho_high data shows CWD > constant.

**Candidate D (Matched-ρ SGD) — Strengthened:**
- Revise the interpretation scope: we cannot claim "constant-ρ SGD matches AdamW insensitivity" — only "initial-ρ-matched SGD." In the paper, report rho(t) trajectories for both SGD and AdamW to show how their effective regularization diverges over training.
- Add the "dead simple" no_wd check: at ρ=0.5 under matched SGD, no_wd (no regularization) should dramatically differ from constant because SGD lacks AdamW's implicit ℓ∞ norm constraint. If no_wd catastrophically underperforms (Cohen's d > 5), the experiment demonstrates that SGD at high ρ is more sensitive to WD presence/absence, even if less sensitive to WD modulation strategy (CWD vs. constant). This is a distinct and important finding.

### Additional Experiments Needed (Beyond Currently Running)

1. **VGG-16-BN multi-method parallelization (CRITICAL):** Start all 6 remaining WD methods on VGG-16-BN in parallel across GPUs 2-7. Currently only constant is running (GPU 1). Remaining experiments represent the single highest-priority action.

2. **Rho_high at CIFAR-100 (HIGH VALUE, easy):** Add rho_high CIFAR-100 condition with 3 methods × 3 seeds × 200 epochs. If rho_high shows method sensitivity, testing it on the harder CIFAR-100 benchmark (where method range under standard AdamW was already 0.75%) strengthens the finding. Expected: rho_high CIFAR-100 range > 1.0%.

3. **Matched-ρ SGD with constant LR (MECHANISTIC):** Run a single seed (seed=42) of matched-ρ SGD with constant LR = 0.01 (no cosine). This eliminates the dynamic ρ confound and provides a cleaner test of static ρ-matched SGD vs. AdamW. 200 epochs at constant lr=0.01 will converge more slowly but the interpretation is cleaner.

4. **ImageNet pilot (HIGHEST REVIEWER DEMAND):** The existing pilot FAILED due to data availability. Before claiming "our results have implications for ImageNet-scale training," we must either (a) confirm ImageNet data is available and run a 5-epoch pilot, or (b) explicitly state that all evidence is CIFAR-scale and explicitly NOT generalized to ImageNet.

### Selected Front-Runner

**Front-runner: Candidate A (VGG-16-BN) + Candidate B (ρ-Regime Map), executed in parallel.**

Rationale:
- VGG-16-BN provides the second-architecture evidence that reviewers universally expect. It is straightforward to interpret and requires no additional methodological explanations.
- ρ-Regime Map provides the mechanistic explanation for the SGD vs. AdamW 3.7× sensitivity ratio. If rho_high (ρ=5.0) shows CWD > constant, it directly validates Theorem 1 Corollary and provides the paper's most novel empirical finding.
- Both experiments are currently running (partially) and can be completed within 6-8 GPU-hours with proper parallelization.

---

## Phase 5: Final Proposal — Iter 6 Experimental Program

### Title
**"Phi Insensitivity at Standard ρ: Architecture-Confirmed and Regime-Mapped Evidence for WD-Method Equivalence Under AdamW"**

### Hypothesis (Primary, Precisely Falsifiable)
Under AdamW with ρ ≥ 0.5, WD method choice among {constant, CWD-hard, SWD, cosine-schedule, random-mask, half-lambda, no-WD} is statistically equivalent within δ = ±1.0% on CIFAR-scale benchmarks, across ResNet-20 and VGG-16-BN architectures. Furthermore, method sensitivity emerges when ρ is increased to 5.0, with CWD outperforming constant by > 0.5% at rho_high, consistent with Theorem 1 Corollary (alignment-aware WD is beneficial when stability cost is negligible at high regularization).

### Falsification Criteria (Pre-Committed, Before Seeing Results)

| Claim | Falsification Condition |
|-------|------------------------|
| VGG null result at ρ=0.5 | Any method differs from constant by > 1.0% (VGG CIFAR-10, N=3) |
| ρ-driven sensitivity emergence | rho_high (ρ=5.0) method range ≤ 0.5% after 200ep (fails to show sensitivity) |
| CWD optimal at high ρ (Theorem 1 Cor.) | CWD does NOT outperform constant at rho_high (ρ=5.0) by > 0.3% |
| Matched-ρ SGD convergence to AdamW | Matched-ρ SGD method range > 0.5% (ρ alone does not explain insensitivity) |

### Current Evidence State (March 18, 2026 — Iteration 6)

#### Completed Experiments

| Dataset | Arch | Optimizer | ρ | Methods | N | Method Range | Status |
|---------|------|-----------|---|---------|---|-------------|--------|
| CIFAR-10 | ResNet-20 | AdamW | 0.5 | 7 | 3 | **0.250%** | Complete |
| CIFAR-100 | ResNet-20 | AdamW | 0.5 | 7 | 3 | **0.753%** | Complete |
| CIFAR-10 | ResNet-20 | SGD | 0.005 | 7 | 3 | **0.913%** | Complete |
| CIFAR-100 | ResNet-20 | SGD | 0.005 | 7 | 3 | **1.707%** | Complete |
| CIFAR-10 | VGG-16-BN | AdamW | 0.5 | 1 (const) | 2 (partial) | N/A (ongoing) | Running |

#### Key Numerical Facts (Hard Evidence)

- **AdamW CIFAR-10 method range: 0.250%** (constant=90.133%, swd=89.883%, max-min = 0.250%)
- **SGD CIFAR-10 method range: 0.913%** (3.65× ratio vs. AdamW; constant=91.217%, no_wd=90.303%)
- **SGD ρ vs. AdamW ρ: 100× difference** (SGD: 5e-4/0.1 = 0.005; AdamW: 5e-4/1e-3 = 0.5) — uncontrolled confound
- **Weight norm spread**: AdamW 95.9-97.0 (range 1.2, CV=1.2%); SGD 64.6-127.1 (range 62.5, CV=50%) — extreme compression under AdamW
- **TOST at δ=±1.0%**: All 6 non-constant methods EQUIVALENT under AdamW CIFAR-10 (p < 0.015)
- **TOST at δ=±0.5%**: Only cosine_schedule equivalent (p=0.028); other methods inconclusive (p=0.060-0.167)
- **CWD underperforms constant under SGD**: 90.867% vs. 91.217% (Cohen's d = 1.133) — mechanistically important
- **AIS is similar across all methods** under both SGD and AdamW (~0.34-0.42 for AdamW, ~0.37-0.42 for SGD) — alignment signal present but not predictive of method advantage
- **NoBN AIS higher (0.49-0.57)** vs. BN AIS (~0.34): partial support for hypothesis that BN suppresses alignment informativeness

#### Critical Gap Analysis

| Gap | Scientific Importance | Current Status |
|-----|----------------------|----------------|
| VGG-16-BN full N=3 (second architecture) | CRITICAL | Running: seed=42 done, seed=123 at 78ep, other methods NOT running |
| ρ-regime map (rho_low=0.05, rho_high=5.0) | CRITICAL | Running: only constant/seed=42 at 66-67ep for both ρ conditions |
| Matched-ρ SGD full N=3 | HIGH | Running: pilot done, seed=123 at 67ep constant only |
| NoBN full ablation (N=1, 3 methods) | MODERATE | Running: constant/seed=42 at 77ep |
| ImageNet ResNet-50 | CRITICAL (reviewer demand) | FAILED (no data available) |
| N=5 for TOST at δ=±0.5% | HIGH | Not started |

### Minimum Required Experiments Before Submission

**Gate 1 (Required for "multi-architecture" claim):**
- VGG-16-BN: all 7 methods × 3 seeds × 200 epochs complete
- Expected: method range 0.2-0.5% (null result confirmed across architectures)
- BLOCKING: Currently only constant running; must start other 6 methods immediately

**Gate 2 (Required for "ρ-regime" claim and Theorem 1 Corollary validation):**
- rho_high: 4 methods × 3 seeds × 200 epochs complete for ρ=5.0
- rho_low: 4 methods × 3 seeds × 200 epochs complete for ρ=0.05
- Expected: rho_high range > 0.5% (confirming ρ drives sensitivity emergence)
- BLOCKING: Only constant/seed=42 running for each ρ condition

**Gate 3 (Required for "matched-ρ explains sensitivity ratio" claim):**
- Matched-ρ SGD: constant, cwd_hard, no_wd × 3 seeds × 200 epochs
- Expected: method range between 0.25% (AdamW-like) and 0.91% (standard SGD-like)
- BLOCKING: Only seed=42 pilot (5ep) and seed=123 partial (67ep) done

**Optional but HIGH VALUE (for "scale generalization" claim):**
- ImageNet: resolve data availability, run 5-epoch pilot, then full experiment
- Without ImageNet: all claims are explicitly restricted to CIFAR scale

### Statistical Test Plan (Pre-Committed)

1. **Primary analysis:** one-way ANOVA across 7 WD methods (separately for each architecture/dataset/optimizer cell)
2. **Pairwise comparison:** Dunnett's test (each method vs. constant), Holm-Bonferroni corrected
3. **Equivalence testing:** TOST at δ=±1.0% (N=3, detectable MDE=0.51%) for CIFAR; at δ=±0.5% (N=5, detectable MDE=0.36%) if seed extension completed
4. **ρ-trend analysis:** Spearman correlation between ρ and method range across the 3-4 ρ conditions; fit log-linear model: range = a × log(ρ) + b
5. **Stability cost vs. alignment benefit:** Compute (Theorem 1) AIS_threshold from σ², n, λ̄ for each (ρ, architecture) condition; check whether empirical CWD advantage (or disadvantage) is correctly predicted by the threshold

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| rho_high (ρ=5.0) shows null result (range < 0.25%) | Medium | Contradicts Theorem 1 Corollary; weakens regime map | Report as falsification; strengthens theoretical claim that "ρ is not the sole driver" |
| VGG-16-BN shows larger method range > 1.0% | Low | Contradicts "null result generalizes across architectures" | Investigate per-layer weight norms and AIS; determine if residual connections modulate sensitivity |
| Matched-ρ SGD shows range = 0.40-0.60% (inconclusive zone) | High | Cannot confirm or deny ρ-sufficiency hypothesis | Report as inconclusive; attribute to dynamic-ρ confound from cosine LR; plan static-LR follow-up |
| NoBN shows same range as BN despite higher AIS | Medium | BN hypothesis falsified | Accept falsification; attribute insensitivity to AdamW mechanism, not BN |
| ImageNet data permanently unavailable | High | Cannot make ImageNet claims | Restrict all claims to CIFAR scale; explicitly note the limitation |
| VGG other-method runs still sequential | Current issue | Delays Gate 1 completion by ~6× | Start parallel GPU allocation for methods 2-7 immediately |

### Novelty Claims (Experimental Contribution)

This program provides:

1. **Architecture-confirmed Phi-insensitivity**: The null result (constant WD is best or tied for best under AdamW at standard ρ) extends from ResNet-20 to VGG-16-BN — two architecturally distinct models representing different gradient flow patterns. Together, these cover the most common CIFAR-scale evaluation architectures in the WD literature.

2. **First empirical ρ-regime map for WD method sensitivity**: A systematic characterization of how method range varies with ρ across 3-4 orders of magnitude. If the theoretical prediction holds (sensitivity emerges at ρ > ρ*), this is the first experimental validation of Theorem 1 Corollary. If not, it constrains the regime where the theorem applies.

3. **Partial isolation of the ρ confound**: The matched-ρ SGD experiment provides the first controlled test of whether ρ or AdamW's implicit ℓ∞ bias drives method insensitivity. Even with the dynamic-ρ confound from cosine LR, this experiment narrows the possible explanations.

4. **Evidence on BN's role (with acknowledged limitations)**: The NoBN ablation provides suggestive evidence that BN scale-invariance contributes to WD-method insensitivity, consistent with the higher AIS values observed without BN. This represents the first systematic measurement of alignment informativeness in BN vs. NoBN settings under identical optimizer conditions.

---

## Evidence Inventory Summary (Current State, March 18, 2026 — Iteration 6)

### What We Have (Hard Evidence)
| Data | Runs | Status | Quality |
|------|------|--------|---------|
| AdamW ResNet-20 CIFAR-10/100, 7 methods, N=3 | 42 | Complete | High (MDE~0.51%) |
| SGD ResNet-20 CIFAR-10/100, 7 methods, N=3 | 42 | Complete | High (same MDE) |
| VGG-16-BN CIFAR-10 constant only, N=1-2 | 1-2 | Partial | Low (N<3, 1 method) |
| Rho_low constant only, seed=42, 66ep | 1 | Partial | Very Low |
| Rho_high constant only, seed=42, 66ep | 1 | Partial | Very Low |
| Matched-ρ SGD constant seed=42 pilot (5ep) | 1 | Pilot only | Very Low |
| Matched-ρ SGD constant seed=123 (66ep) | 1 | Partial | Very Low |
| NoBN constant seed=42, 77ep | 1 | Partial | Very Low |

### Critical Pre-Writing Checklist

Before entering writing phase, ALL of the following must be verified:

1. **Figure 2 SGD p-value annotation**: Text states p=0.071 for SWD under SGD; figure annotation reportedly shows p=0.004. Verified from raw data (seed42/123/456 for SGD/CIFAR10/swd): raw t-test p needed. Compute: `scipy.stats.ttest_ind([90.57, 90.63, 90.93], [91.3, 91.18, 91.17])` → t-test p-value.
2. **Cohen's d formula**: All tables must consistently use "unpaired pooled SD" Cohen's d formula.
3. **Figure 4 trajectory source**: Must confirm all curves are from actual `epoch_metrics.jsonl` data, not illustrative.
4. **Figure 3 correlation label**: If it shows r=0.48, the caption must not say "No correlation."
5. **Gate 1 (VGG)**: ALL 7 methods × 3 seeds complete before claiming "multi-architecture confirmation."

### The Empiricist's Bottom Line

The existing evidence (84 complete runs, both optimizers, both CIFAR datasets, N=3, full 200 epochs) is **sufficient for the central null result** — under AdamW at ρ=0.5, WD method choice has negligible effect on CIFAR-10/100 accuracy (TOST confirmed at δ=±1.0%). This result is real, reproducible, and statistically defensible.

The existing evidence is **insufficient for the mechanistic claims**:
- "ρ explains the SGD/AdamW sensitivity gap" — no matched-ρ data yet
- "The null result generalizes to other architectures" — VGG data is <10% complete for the full experiment
- "CWD is beneficial at high ρ (Theorem 1 Corollary)" — no rho_high method comparison data yet
- "BN causes insensitivity" — NoBN data is N=1, 1 method, 77 epochs, with unresolved LR/ρ confound
- "The null result generalizes to ImageNet scale" — no ImageNet data (pilot failed)

**The minimum acceptable paper** (if no additional experiments complete):
- Central claim: "WD method choice is equivalent within ±1.0% under AdamW at standard ρ=0.5 on CIFAR scale" — supported
- Supporting claims: "Mechanism is unknown but consistent with BN scale-invariance and high-ρ steady-state" — speculative
- The matched-ρ confound must be acknowledged as a limitation, not papered over

**The target paper** (if Gates 1-3 complete):
- Central claim: same as above
- VGG architecture confirmation: "null result extends to architectures without residual connections"
- ρ-regime map: "sensitivity emerges at ρ ~ 5.0 under AdamW; CWD beneficial at high ρ" (if confirmed)
- Matched-ρ SGD: "ρ partially explains sensitivity gap; AdamW's ℓ∞ implicit bias may contribute additionally" (or "ρ alone explains gap" if matched-ρ shows insensitivity)

Every experiment that completes between now and the writing deadline transforms a speculative claim into an evidence-backed finding. The computing resources are available. The bottleneck is parallelization — start all VGG methods simultaneously, immediately.

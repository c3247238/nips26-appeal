# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **[tml-epfl/why-weight-decay](https://github.com/tml-epfl/why-weight-decay)** (MIT, NeurIPS 2024) — The primary evaluation infrastructure used in our prior iterations. Includes ResNet-20/VGG-16-BN on CIFAR-10/100 and ResNet-50 on ImageNet, full weight/gradient norm tracking. **Has code. Already adopted.**

2. **[Investigating the Role of Weight Decay in Enhancing Nonconvex SGD](https://arxiv.org/abs/2405.10234)** (Sun et al., CVPR 2025) — The theoretical core of this project. Proves WD's generalization benefit in nonconvex SGD via alignment quantity δ_T. Proves WD does NOT accelerate convergence but DOES improve generalization. Limited to fixed WD. **Key reference: directly motivates alignment-aware WD.**

3. **[Cautious Weight Decay (CWD)](https://arxiv.org/abs/2510.12402)** (Chen et al., ICLR 2026) — Binary sign-alignment mask; one-line drop-in. Our experiments show it performs WORSE than constant WD on CIFAR-10/100 with SGD (mean -0.29% on CIFAR-10, mean -0.58% on CIFAR-100). Code: one-line mask. **Critical negative result in our data.**

4. **[Scheduled Weight Decay / AdamS](https://github.com/zeke-xie/stable-weight-decay-regularization)** (Xie et al., NeurIPS 2023) — Gradient-norm-aware dynamic WD. Our experiments show it performs worse than constant WD on both CIFAR datasets (CSI notably higher at ~1.15 vs ~0.84, indicating optimizer coupling instability). **Code available. Our data shows it is detrimental under SGD.**

5. **[Weight Norm Control / AdamWN](https://arxiv.org/abs/2311.11446)** (Loshchilov, 2023) — Target-norm control framework. Generalizes decoupled WD to target arbitrary norm levels. ~20 lines of PyTorch. No standalone repo but trivially implementable. **Theoretically the cleanest norm-matched WD formulation.**

6. **[AlphaDecay](https://github.com/hed-ucas/AlphaDecay)** (He et al., arXiv 2025) — Module-wise adaptive WD via spectral heavy-tail analysis. Apache-2.0. LLM-focused but borrows spectral analysis approach. **Has code. Useful for layer-wise WD ablations.**

7. **[Spectral Dynamics Codebase](https://github.com/dyunis/spectral_dynamics)** (Yunis et al., arXiv 2408.11804) — Singular value tracking during training. MIT license. Relevant for rank-aware WD scheduling (Gap 10 from literature survey). **Has code. Low implementation cost for adding per-layer spectral tracking.**

8. **[CPR / AdamCPR](https://github.com/automl/CPR)** (Franke et al., NeurIPS 2024) — Per-parameter-matrix upper-bound constraints via augmented Lagrangian. `pip install pytorch-cpr`. Has demonstrated advantages over AdamW on CIFAR-100, ImageNet, GPT-2. **Has code. Strong competing approach that we have not yet benchmarked against.**

9. **[OUI (Overfitting-Underfitting Indicator)](https://github.com/AlbertoFdezHdez/OUI)** (arXiv 2504.17160, MIT) — Diagnostic metric for WD quality monitoring. Could serve as validation-free component of our CSI metric. **Has code. Low integration cost.**

10. **[Defazio gradient-to-weight ratio](https://arxiv.org/abs/2506.02285)** (2025) — Shows WD drives ‖g‖/‖w‖ of normalized layers to steady state ("layer balancing"). Explains Adam vs AdamW gap. Proposes corrective term for LR-schedule interaction. **No standalone code but <10 lines to implement. High theoretical relevance as unified lens.**

11. **[Correction of Decoupled Weight Decay](https://arxiv.org/abs/2512.08217)** (Chou, 2025) — Derives WD ∝ γ² for stable weight norm via Total Update Contribution analysis. Actionable scaling rule. **No standalone code but direct formula to implement.**

12. **[VGG-16-BN on CIFAR experiments](https://github.com/tml-epfl/why-weight-decay)** — Our plan includes VGG-16-BN as a second architecture per project constraints. The `why-weight-decay` repo already has this. **Gap in our current data: all iter_003 results are ResNet-20 only.**

### Landscape Summary

Here is the engineering reality after three iterations of experiments:

**What the data actually shows (our iter_003 results, SGD on CIFAR):**
- CIFAR-10/ResNet-20: constant WD wins or ties on best accuracy (mean 91.22% vs all dynamic methods ≤91.20%). CWD is -0.30% on average.
- CIFAR-100/ResNet-20: constant WD wins clearly (mean 65.37%). CWD is -0.68% on average. SWD is -1.02% on average — the worst performer.
- CSI: SWD has anomalously high CSI (~1.15 vs baseline ~0.83), exactly as predicted by the WD Stability Condition (rapid gradient-norm fluctuation causes optimizer coupling instability).
- AIS: 0.28–0.46 range across all methods and seeds, suggesting alignment signal carries real but low informativeness for WD decisions.

**What the prior AdamW experiments showed (iter_003 writing):**
- In the AdamW regime (the paper's Section 5.1 results), performance spread is only 0.25% on CIFAR-10 and 0.76% on CIFAR-100. No method is statistically significant.

**The critical engineering question for the next iteration:**
The current results cover ResNet-20 + CIFAR only. Project constraints require VGG-16-BN on CIFAR and ResNet-50 on ImageNet. Before proposing theory extensions, the pragmatist priority is: **do the findings generalize across architectures?** VGG-16-BN has no batch normalization in the same positions as ResNet-20, and the alignment dynamics may differ. ResNet-50/ImageNet is a higher-stakes validation.

**What is missing from our experimental coverage:**
- VGG-16-BN on CIFAR-10/100: NOT yet run
- ResNet-50 on ImageNet: NOT yet run
- Adaptive optimizer (AdamW) at larger scale: iter_003 had results but only CIFAR-10/100/ResNet-20
- CPR comparison: never benchmarked
- Continuous alignment modulation (not binary mask like CWD): never tested

---

## Phase 2: Initial Candidates

### Candidate A: Architecture Generalization + CPR Baseline (The Missing Empirical Evidence)

- **Core hypothesis**: The null result (no dynamic WD method beats constant WD) that we established for ResNet-20/CIFAR generalizes to VGG-16-BN/CIFAR and ResNet-50/ImageNet. Additionally, CPR (the strongest competing approach from NeurIPS 2024) also fails to beat well-tuned constant WD under compute-controlled conditions.

- **Implementation sketch**: Reuse our existing unified training infrastructure from iter_003. Add VGG-16-BN as architecture (already in `why-weight-decay` repo, ~5 lines to add). Add `pytorch-cpr` (`pip install pytorch-cpr`) as one more WD method. Run the same 7 methods + CPR on CIFAR-10/100/VGG-16-BN. For ImageNet, use ResNet-50 with reduced epoch count (90 epochs, the standard torchvision baseline) and our 3 seeds. The VGG experiments can run in parallel with CIFAR-ResNet (same GPU hours).

- **Simplest version**: Run 3 methods (constant, cosine_schedule, CWD) + CPR on VGG-16-BN/CIFAR-10 with 1 seed each. If the constant vs. CWD gap stays similar to ResNet-20 results, the generalization hypothesis holds. ~45 minutes on one GPU.

- **Time estimate**: VGG-16-BN CIFAR pilot (3 methods × 1 seed × ~20 min) = 1 GPU-hour. Full VGG suite (8 methods × 3 seeds × 2 datasets × ~25 min) = 30 GPU-hours. ImageNet/ResNet-50 (8 methods × 3 seeds × ~5 hours) = 120 GPU-hours. Feasible on 8 GPUs: VGG in ~2 hours, ImageNet in ~15 hours wall-clock.

- **Reusable components**: Existing iter_003 training code + `pytorch-cpr` + torchvision models.

### Candidate B: Continuous Alignment Modulation Ablation (Does Softening CWD Help?)

- **Core hypothesis**: CWD's binary sign-alignment mask is too aggressive and the wrong design choice. A continuous alignment modulation — λ_t × f(cosine_sim(w_t, g_t)) — where f is a smooth sigmoid or power function, should perform better than binary CWD while remaining comparable to constant WD. The point is to test whether the issue is "binary vs continuous" or "alignment-awareness fundamentally doesn't help."

- **Implementation sketch**: Implement 3 variants of continuous alignment modulation: (a) λ_t = λ_0 × sigmoid(k × cos(w,g)) where k=1,5,10; (b) λ_t = λ_0 × (1 - |cos(w,g)|) (suppress WD when strongly aligned OR opposed); (c) λ_t = λ_0 × (1 - cos(w,g))^2 (the "quadratic suppression" variant). All are ~5 additional lines in the existing Phi framework. Compare against constant WD and binary CWD on CIFAR-10/100/ResNet-20 with 3 seeds each.

- **Simplest version**: Just variant (b) on CIFAR-10/ResNet-20 with 3 seeds. If it matches constant WD, the binary vs continuous distinction doesn't matter. If it beats CWD, we have identified the design flaw in CWD. 3 × 15 min = 45 min pilot.

- **Time estimate**: 3 variants × 3 seeds × 2 datasets × 15 min = 4.5 GPU-hours total.

- **Reusable components**: Existing Phi modulator interface from iter_003 is designed to accept exactly these variants. Zero new infrastructure needed.

### Candidate C: Theoretical Framework Completeness — Quantitative Connection Between δ_T and CSI/AIS

- **Core hypothesis**: The alignment quantity δ_T from Sun et al. (CVPR 2025) has a direct mathematical relationship to our AIS (Alignment Informativeness Score) metric. Specifically, AIS ≈ 1 - δ_T in the SGD setting. If we can prove this relationship formally, it closes the theoretical loop: low AIS → high δ_T → WD provides strong generalization benefit but alignment-aware scheduling adds no further value because δ_T is already controlled by constant WD.

- **Implementation sketch**: Pure theory work + visualization. Compute δ_T from our existing epoch_metrics.jsonl data (all 42 SGD experiments have alignment tracked as AIS). Re-derive Sun et al.'s stability recursion with time-varying λ_t. Show that if AIS ≈ constant over training (which our data hints at), then the dynamic WD schedule reduces to constant WD in expectation. Produce LaTeX proof + scatter plot of δ_T vs AIS from experimental data.

- **Simplest version**: Extract cosine similarity (AIS proxy) per epoch from our iter_003 data, compute the running average δ̂_T = mean(AIS over epochs), and check correlation with final accuracy gaps. Entirely using existing data — zero new experiments needed. ~2 hours of analysis work.

- **Time estimate**: 0 GPU-hours (analysis only). 4-6 hours of analysis + writing.

- **Reusable components**: iter_003 epoch_metrics.jsonl files for all 42 experiments.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Implementation reality check**: VGG-16-BN is already in the `why-weight-decay` repo. CPR is `pip install pytorch-cpr`. The experimental addition is small. For ImageNet, 3 seeds × 5 hours × 8 methods = 120 GPU-hours is non-trivial but feasible on 8 GPUs in under 2 days wall-clock. The main risk is GPU memory for ImageNet experiments (ResNet-50 with batch 256 needs ~8GB per GPU — no problem on 98GB GPUs).

- **Reproducibility attack**: Our iter_003 infrastructure is already reproducible (seeds controlled, same base hyperparameters). Adding VGG and CPR is a straight extension. The only fragility is CPR's per-matrix constraint initialization (does it converge with our default hyperparameters?). Mitigation: run CPR with its recommended defaults from the paper.

- **Baseline sanity check**: For ImageNet/ResNet-50, the well-known PyTorch baseline is 76.1% top-1 with SGD, cosine LR, WD=1e-4, 90 epochs. If our constant WD achieves this, the baseline is correct. CWD one-line drop-in should be straightforward.

- **Scope attack**: Three architectures (ResNet-20, VGG-16-BN, ResNet-50) across three datasets (CIFAR-10, CIFAR-100, ImageNet) is strong generalization scope. The weakness is that we're still in the SGD + vision classification regime. No language models, no AdamW at scale. But project constraints specify these architectures — this is intentional scope.

- **Verdict**: **STRONG.** Fills the clearest gap in our current evidence (architecture generalization + ImageNet). Implementation risk is low. Results will either strongly confirm the null finding (most likely) or reveal an architecture-dependent effect (interesting and publishable either way).

### Against Candidate B

- **Implementation reality check**: The continuous modulation variants are literally 5 lines of code within the existing Phi interface. The compute cost is trivial (< 5 GPU-hours). The question is whether the results are *interesting enough* to include in the paper.

- **Reproducibility attack**: High reproducibility — same infrastructure, same seeds, same hyperparameters. The only question is what value of k to use for sigmoid variant. Use k=1 (soft), k=5 (medium), k=10 (hard approaching binary). All three in one sweep.

- **Baseline sanity check**: The best baseline is our existing constant WD result (CIFAR-10: 91.22%, CIFAR-100: 65.37%). The continuous variants need to get within noise range of this. Given that binary CWD performs *worse*, continuous variants may perform *better* or still *worse* — both are informative.

- **Scope attack**: This is only on CIFAR-10/100/ResNet-20. If the finding is "continuous alignment modulation doesn't help either," we need to report it at ImageNet scale too for credibility. Limit to CIFAR for the pilot, extend if interesting.

- **Verdict**: **MODERATE.** Scientifically interesting but low-risk/low-reward. If continuous variants also fail, this strengthens the overall null message. If one variant wins, it's a positive finding but a small one. Worth including as an ablation section, not as the core contribution.

### Against Candidate C

- **Implementation reality check**: This is 100% analysis of existing data. No new experiments required. The theoretical derivation is the main work: connecting Sun et al.'s δ_T (full-data gradient alignment) to our AIS (minibatch proxy, averaged over epochs). This connection requires careful handling of the stochastic process.

- **Reproducibility attack**: Pure math + data analysis. The scatter plot is entirely reproducible from our data files. The proof requires review but does not rely on any undiscovered facts — it's a formalization of relationships that are already visible in the data.

- **Baseline sanity check**: The prior iterations have established δ̂_T ≈ AIS ≈ 0.3-0.45 across all our experiments. This is the quantitative "constant" that makes dynamic WD uninformative. The theory should explain why this value is the fixed point.

- **Scope attack**: Pure theory based on CIFAR/SGD. The δ_T concept applies to nonconvex SGD generally. Whether it generalizes to AdamW or ImageNet is less clear. Frame carefully as "for SGD in the CIFAR regime."

- **Verdict**: **STRONG (for theoretical contribution).** Zero compute cost, uses only existing data. High theoretical leverage: if we can formally connect AIS to δ_T, the paper has a tight loop from Sun et al. (CVPR 2025) theory → our dynamic WD failure explanation → actionable implications. This is the kind of theoretical contribution that makes a NeurIPS/ICML paper rather than just a benchmarking paper.

---

## Phase 4: Refinement

**Dropped ideas**: None fully dropped. Candidate B demoted to ablation status (not the core contribution).

**Strengthened ideas**:

**Candidate A** (architecture generalization): This is the clearest missing piece. We must run VGG-16-BN on CIFAR and ResNet-50 on ImageNet to be credible at submission. The CPR comparison adds a strong competing method we have not yet addressed. Priority: HIGH. Start with CIFAR-VGG pilot immediately.

**Candidate C** (theoretical connection): Reframed as the theoretical backbone of the paper. The connection between our AIS metric and Sun et al.'s δ_T is not just a nice derivation — it IS the unified theoretical framework that explains all our empirical results. If AIS ≈ 1 - mean(δ̂_t), then:
  - Low AIS → high δ_T → WD helps generalization (Sun et al. proven)
  - Constant AIS over training → dynamic WD averages to constant WD (our budget equivalence finding)
  - AIS < 0.5 in all our experiments → alignment signal is below the threshold where dynamic modulation would matter

This gives the paper a **falsifiable theoretical prediction**: if AIS > 0.6 (high alignment informativeness), dynamic WD SHOULD help. We can test this by finding training regimes where AIS is high (e.g., very early training, high WD, or specific architectures).

**Candidate B** (continuous alignment modulation): Run as a 5 GPU-hour ablation on CIFAR-10/ResNet-20 only. Results go in an ablation table. If any continuous variant achieves AIS > 0.5 while beating constant WD, this becomes an interesting secondary finding.

**Additional search** (engineering risk check): ImageNet ResNet-50 with 90 epochs + WD=1e-4 SGD standard baseline must be validated against PyTorch official numbers (76.1% top-1). Quick check: use the `why-weight-decay` repo's ImageNet setup, which already achieves this. Confirmed feasible.

**Selected front-runner**: Candidate A (architecture generalization) as the experimental core, supported by Candidate C (theoretical backbone). The combination gives us:
1. A theory that predicts when dynamic WD fails (AIS < 0.5 → δ_T is constant → dynamic modulation averages to constant WD)
2. Experiments confirming the prediction across 3 architectures + 3 datasets
3. A falsification test (find a regime where AIS > 0.6 and test whether dynamic WD helps there)

---

## Phase 5: Final Proposal

### Title
**When Does Alignment-Aware Weight Decay Fail? A Unified Framework with AIS Threshold Predictions and Multi-Architecture Validation**

### Hypothesis
**Primary (falsifiable)**: When the Alignment Informativeness Score (AIS) — the mean cosine similarity between gradient and weight vectors during training — is below 0.5 across all layers, alignment-aware dynamic WD (CWD, continuous variants) achieves statistically indistinguishable accuracy from well-tuned constant WD. Specifically:
- In our SGD/CIFAR-10/100/ResNet-20 experiments: AIS ≈ 0.35–0.45 throughout training → null result confirmed (iter_003 data)
- Prediction for VGG-16-BN/CIFAR: AIS will be in the 0.30–0.50 range → same null result
- Prediction for ResNet-50/ImageNet: AIS will be in the 0.30–0.50 range → same null result
- Falsification: if AIS > 0.55 on any architecture/dataset, dynamic WD should show measurable improvement

**Secondary**: SWD performs worse than constant WD because it operates by gradient-norm-driven schedule changes, which increase the Coupling Stability Index (CSI) above 1.0, indicating optimizer coupling instability. Verified by our data (SWD CSI ≈ 1.15 vs constant CSI ≈ 0.84).

### Motivation
Three iterations of experiments have established a robust null result: no dynamic WD method (scheduling, alignment-aware, or mixed) beats well-tuned constant WD on CIFAR-10/100/ResNet-20 under compute-controlled conditions. This finding needs two things to become publishable:

1. **Theoretical explanation**: WHY do these methods fail? The connection between AIS and Sun et al.'s alignment quantity δ_T provides this. When AIS is consistently low (~0.35), gradient and weight vectors are nearly orthogonal — in this regime, conditional decay (CWD) or gradient-norm-based scaling (SWD) has no signal to act on that constant WD does not already handle.

2. **Generalization evidence**: The null result must be shown to hold beyond ResNet-20/CIFAR. VGG-16-BN and ResNet-50/ImageNet are required by project constraints and by the standards of the field.

The practical gap: practitioners who read CWD (ICLR 2026) or SWD (NeurIPS 2023) will try to apply these methods to their SGD-based vision training pipelines. Our finding that they are harmful or neutral is actionable and currently undocumented.

### Method

#### Step 1: Theoretical Connection (AIS ↔ δ_T, 1-2 days)

Starting from Sun et al.'s stability bound: at step T, if alignment quantity δ_T < 1, WD improves generalization. Extend to the dynamic case:
- With time-varying λ_t, the effective δ_T becomes a trajectory-weighted average: δ̄_T = Σ_t λ_t δ_t / Σ_t λ_t
- When the per-step alignment proxy δ̂_t (minibatch cosine similarity ≈ our AIS) is approximately constant over training (empirically: AIS ≈ 0.35-0.45, σ < 0.05 across epochs), then δ̄_T ≈ AIS regardless of the λ_t schedule
- This means any WD schedule that integrates to the same total budget achieves the same stability bound — the budget equivalence result is derived directly from constant AIS
- Formal statement: **Proposition 1** (budget equivalence under constant alignment): If δ̂_t = δ̄ ± ε with ε < 0.05 for all t, then E[generalization gap with schedule {λ_t}] = E[generalization gap with λ_constant = mean(λ_t)] up to O(ε) correction terms

This provides a **testable threshold**: AIS variation > 0.1 per epoch is the condition where dynamic WD could theoretically help.

#### Step 2: Architecture Generalization Experiments

**VGG-16-BN/CIFAR-10/100 (3 seeds × 7 methods × 2 datasets = 42 experiments)**:
- Fork our existing iter_003 training script, change `--arch resnet20` to `--arch vgg16_bn`
- `why-weight-decay` repo already has this model
- Expected time: ~25 min/experiment × 42 = ~17.5 GPU-hours
- Run on 4-8 GPUs in parallel: ~3-4 hours wall-clock
- Collect: best accuracy, CSI, AIS, weight norm trajectories

**ResNet-50/ImageNet (3 seeds × 7 methods = 21 experiments)**:
- Standard 90-epoch SGD schedule: LR=0.1 with cosine decay, WD=1e-4, batch=256
- Expected time: ~5 hours/experiment × 21 = 105 GPU-hours
- Run on 8 GPUs: ~13-14 hours wall-clock
- Add CPR as 8th method: `pip install pytorch-cpr` + 2 lines in optimizer setup

#### Step 3: Continuous Alignment Modulation Ablation (existing architecture only)

On ResNet-20/CIFAR-10/100 (3 seeds × 3 variants × 2 datasets = 18 experiments):
```python
# Variant A: sigmoid suppression
phi = torch.sigmoid(-k * cos_sim)  # k=5, low phi when aligned
# Variant B: anti-alignment boost
phi = 1.0 - cos_sim.abs()  # suppress when strongly aligned OR opposed
# Variant C: quadratic suppression
phi = (1.0 - cos_sim) ** 2 / 4.0  # normalized quadratic
```
All variants are 5-line additions to the existing Phi framework. Expected time: ~15 min × 18 = 4.5 GPU-hours.

#### Step 4: AIS Threshold Validation

After collecting all data, compute AIS trajectories for all methods/architectures. Test the threshold hypothesis:
- Plot AIS distribution across training for all architectures — do they all stay below 0.5?
- Find the one subset with highest AIS (likely: first 5 epochs, or smallest dataset)
- Run targeted experiments in the high-AIS regime: shorter training (20 epochs), very small dataset (10% of CIFAR-10), or larger WD (5× baseline)
- This is the falsification test: in the one regime where AIS > 0.5, does CWD or continuous modulation help?

### Baselines

1. **Constant WD** (our primary baseline): AdamW/SGD with fixed λ = 5e-4. Expected performance on CIFAR-10/ResNet-20: ~91.2% (confirmed in iter_003). On ImageNet/ResNet-50: ~76.1% (torchvision standard).

2. **CWD (binary sign-alignment)**: Our iter_003 data shows -0.30% on CIFAR-10, -0.58% on CIFAR-100 relative to constant. Expected to maintain same direction on VGG and ImageNet.

3. **SWD (gradient-norm schedule)**: Our iter_003 data shows -0.46% on CIFAR-10, -0.72% on CIFAR-100. Expected to be the weakest performer.

4. **CPR (AdamCPR)**: Per-matrix constraint regularization from NeurIPS 2024. Claims to outperform AdamW on CIFAR-100 and ImageNet. This is the strongest competing method we have not yet tested. If it beats constant WD, the unified framework must explain why (hint: spatial modulation, the one axis we have not fully validated).

5. **Cosine WD schedule**: Our iter_003 data shows near-parity with constant on CIFAR-10 (+0.03%) and slight underperformance on CIFAR-100 (-0.12%). Expected to be the "closest competitor" and establish the upper bound on temporal scheduling benefit.

### Experimental Plan

| Phase | Experiments | GPU-hours | Wall-clock | Priority |
|-------|------------|-----------|------------|----------|
| Pilot: VGG-16-BN/CIFAR-10 3 methods | 9 runs × 25 min | 3.75 | 1h (4 GPUs) | HIGH |
| Full VGG-16-BN CIFAR-10/100 | 42 runs × 25 min | 17.5 | 3h (8 GPUs) | HIGH |
| Continuous modulation ablation | 18 runs × 15 min | 4.5 | 1h (4 GPUs) | MEDIUM |
| ResNet-50/ImageNet full suite | 21 runs × 5h | 105 | 14h (8 GPUs) | MEDIUM |
| AIS threshold falsification | 12 runs × 20 min | 4 | 1h (4 GPUs) | LOW |

**Total: ~135 GPU-hours, ~20 hours wall-clock on 8 GPUs**

Ablation schedule: constant > cosine_schedule > CWD > random_mask > SWD > continuous variants > CPR

Datasets: CIFAR-10, CIFAR-100, ImageNet-1K
Models: ResNet-20 (existing), VGG-16-BN (new), ResNet-50 (new)
Metrics: best test accuracy, final CSI, final AIS, weight norm trajectory, per-epoch alignment

### Resource Estimate

- GPU: 8× RTX PRO 6000 Blackwell (98GB each), all local — confirmed available
- Total GPU-hours: ~135 (2 days on 8 GPUs)
- CIFAR experiments: ~20 min per run, parallelizable across 8 GPUs (8 runs simultaneously)
- ImageNet experiments: ~5 hours per run; up to 8 runs in parallel
- Disk space: ~2GB per experiment for full logging → ~50GB total

### Risk Assessment

**Engineering risks**:

1. **VGG-16-BN numerical instability**: VGG without BN has training instability. VGG-16-BN is stable. The `why-weight-decay` repo's VGG implementation has been tested — low risk.

2. **CPR hyperparameter sensitivity**: AdamCPR has 2 hyperparameters (lambda_p and xi_p). The paper's recommended defaults may not work well with our SGD setup (CPR is usually paired with Adam). Mitigation: run CPR with AdamW base optimizer as its recommended setup.

3. **ImageNet training duration**: 5 hours per run is within budget but any SSH/system interruption could waste time. Mitigation: use our existing sentinel/heartbeat infrastructure for automatic recovery.

4. **AIS threshold falsification failure**: If AIS stays below 0.5 in ALL regimes, the falsification test is negative and we cannot demonstrate when dynamic WD DOES work. This is actually fine for the paper — a strong null result across all tested regimes is still publishable. It changes the framing to "we sought the conditions under which dynamic WD helps and found them narrower than expected."

5. **CWD performing differently on ImageNet**: If CWD helps on ImageNet (opposite of our CIFAR result), it complicates the narrative. This would be a finding, not a failure. The theory would then need to predict WHY CWD works at ImageNet scale (perhaps AIS is higher on ImageNet due to larger gradient variance).

**Theoretical risks**:

1. **Proposition 1 proof requires tight concentration bounds**: The stochastic gradient makes the δ_t → AIS connection require Azuma-Hoeffding type bounds on the martingale difference sequence. This is standard but requires careful handling. Mitigation: frame as a semi-formal derivation with empirical validation, not as a rigorous theorem.

2. **AdamW vs SGD setting mismatch**: Sun et al.'s theory is for SGD. Our AIS↔δ_T connection should hold for SGD but may not directly transfer to AdamW (where preconditioning changes the effective alignment). Mitigation: restrict formal theory claims to SGD; treat AdamW results as empirical validation only.

### Novelty Claim

The novelty is threefold:

1. **AIS Threshold Hypothesis**: The first falsifiable prediction about WHEN alignment-aware WD will and will not work. Prior papers (CWD, SWD) made universal claims; we provide a condition (AIS > 0.5) under which their methods are informative and below which they are noise.

2. **Multi-architecture null result**: The first compute-controlled comparison of dynamic WD methods showing consistent null results across three architectures (ResNet-20, VGG-16-BN, ResNet-50) and three datasets (CIFAR-10, CIFAR-100, ImageNet). The "null result is the finding" is itself a publishable contribution at scale.

3. **AIS-δ_T theoretical connection**: Formalizing the relationship between our AIS metric and Sun et al.'s alignment quantity δ_T creates a theoretically grounded explanation for the null result. This is not just showing that methods fail — it explains the mechanism through a derived proposition.

The simplest version of what is new: *"We prove that when gradient-weight alignment is consistently below 0.5 (our AIS threshold), any dynamic WD schedule is budget-equivalent to constant WD, and we validate this prediction across three architectures at two scales."* This is one sentence that captures the core contribution.

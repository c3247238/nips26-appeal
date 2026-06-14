# Pragmatist Perspective

> Agent: Pragmatist (Iteration 4)
> Date: 2026-03-18
> Focus: Maximum impact/effort ratio; executable experiments; what actually works

---

## Phase 1: Literature Survey

### Key Resources Found

1. **tml-epfl/why-weight-decay (GitHub, MIT)** — https://github.com/tml-epfl/why-weight-decay — NeurIPS 2024 codebase. Tracks weight norm, gradient norm, Jacobian norm across ResNet/VGG/ViT on CIFAR/ImageNet with comprehensive per-layer diagnostics. Directly reusable as evaluation scaffold. Has code.

2. **zeke-xie/stable-weight-decay-regularization (GitHub, MIT)** — https://github.com/zeke-xie/stable-weight-decay-regularization — NeurIPS 2023 SWD implementation. The AdamS optimizer extends AdamW with gradient-norm-aware WD scheduling. Our existing code already implements SWD variant. Has code.

3. **Cautious Weight Decay / CWD (ICLR 2026, arXiv:2510.12402)** — https://arxiv.org/abs/2510.12402 — One-line formula: mask WD by sign(u_t ⊙ w_t ≥ 0). Drop-in for AdamW/Lion/Muon. Already implemented in our iter_003 code as cwd_hard. No separate repo yet.

4. **Defazio: Why Gradients Increase Near End of Training (arXiv:2506.02285)** — https://arxiv.org/pdf/2506.02285 — WD drives ||g||/||w|| to a steady-state; "layer balancing" effect. Provides clean unifying lens. Simple corrective term derivable from existing data. No code released but formula is simple.

5. **Sun et al. CVPR 2025 — Nonconvex SGDW theory** — First proof that WD benefits generalization (not convergence) in nonconvex SGD via alignment quantity delta_T. Core theoretical foundation. No open code.

6. **AdamW Implicit Bias — ell_inf constraint (arXiv:2404.04454)** — https://arxiv.org/abs/2404.04454 — AdamW ≈ Frank-Wolfe on ell_inf ball. Explains why WD scheduling may be absorbed by AdamW's built-in norm control. Theoretically validates the Phi Invariance hypothesis.

7. **Spectral Dynamics codebase (GitHub, MIT)** — https://github.com/dyunis/spectral_dynamics — Singular value tracking during training. Connects WD → rank minimization. Potential feedback signal for rank-aware scheduling.

8. **Wang & Aitchison (arXiv:2405.13698)** — WD as EMA timescale; tau = 1/(eta·lambda) = constant across scales. Optimal WD scales inversely with eta. Explains coupling between LR schedule and WD.

9. **D'Angelo et al. NeurIPS 2024 (arXiv:2310.04415)** — https://github.com/tml-epfl/why-weight-decay — WD as dynamics modifier (loss stabilization for SGD; bias-variance tradeoff for LLMs). Key empirical grounding.

10. **gd-zhang/Weight-Decay (GitHub)** — https://github.com/gd-zhang/Weight-Decay — Three mechanisms of WD regularization. Baseline reference implementation.

11. **AlphaDecay (GitHub, Apache-2.0)** — https://github.com/hed-ucas/AlphaDecay — Module-wise WD guided by spectral heavy-tailedness. Useful for spectral analysis module.

12. **OUI diagnostic (GitHub, MIT)** — https://github.com/AlbertoFdezHdez/OUI — Per-training-run WD quality indicator. Could be integrated as one component of CSI.

### Landscape Summary

**What works**: Standard decoupled WD (AdamW) is a settled engineering decision. One-line modifications (CWD) ship improvements for free at LLM scale. Gradient-norm-aware scheduling (SWD) demonstrably improves over static WD with no extra hyperparameters. The tml-epfl/why-weight-decay codebase provides a reusable, battle-tested evaluation scaffold for vision models.

**What does not work in practice**: Per-parameter sigmoid-scaled WD (AdaDecay 2019) was outperformed by CPR/AdamW in more recent comparisons. Module-wise static WD (AlphaDecay) requires spectral analysis overhead and is LLM-specific. Full per-iteration adaptive WD with heavy theoretical machinery tends to add implementation fragility without proportionate gain at CIFAR/ImageNet scale.

**Practical gap**: Nobody has run all four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) on the same benchmark with the same compute budget and the same diagnostic metrics. This is the most tractable and most impactful contribution — not proposing a new algorithm, but proving that existing approaches are special cases and quantifying when each matters.

**Current project state**: We have SGD baseline data for 7 WD methods × CIFAR-10/100 × 3 seeds on ResNet-20. The existing cwd_hard implementation captures binary sign alignment. Key data already on disk: SGD constant vs no_wd shows delta=0.91% (Cohen's d≈12); AdamW shows delta=0.05% (noise level). Missing: VGG-16-BN, ImageNet, continuous cosine-modulated decay, and lambda sweep.

---

## Phase 2: Initial Candidates

### Candidate A: Gradient-to-Weight Ratio as Unifying Diagnostic + Corrective Schedule

**Hypothesis**: The gradient-to-weight ratio rho_t = ||g_t||/||w_t|| is a sufficient statistic for deciding WD strength at each step. Methods that track rho_t explicitly will outperform both fixed WD and sign-based methods (CWD), and the resulting rho_t-conditioned schedule is a special case of the unified lambda(t, w, g) framework.

**Implementation sketch**:
- Start from existing train_unified.py in iter_003/exp/code/
- Add per-step logging of rho_t = ||g_t||/||w_t|| (2 lines)
- Implement corrective WD term from Defazio (2506.02285): lambda_corrected = lambda × (eta_max / eta_t)
- Compare: constant lambda / cosine_schedule lambda / corrective lambda / CWD / SWD
- Pilot: ResNet-20 + CIFAR-10, seed=42, 5 WD variants, approximately 25 minutes on 1 GPU

**Simplest version**: Plot rho_t trajectory for all existing iter_003 methods from existing epoch_metrics.jsonl (zero training cost). Compare rho_t equilibrium values across methods.

**Time estimate**: Pilot analysis from existing data = 0 GPU hours. New experiments: 5 variants × 3 seeds × 2 datasets = 30 runs × ~12 min = ~6 GPU hours.

**Reusable components**: Existing train_unified.py, existing results directory, tml-epfl/why-weight-decay diagnostic style.

---

### Candidate B: The Unification Taxonomy Experiment — Each WD Approach on the Same Budget

**Hypothesis**: Under compute-budget equivalence (equal epochs, same LR schedule), the four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) show statistically distinguishable performance profiles on ResNet-20 + VGG-16-BN on CIFAR-10/100, and these profiles are predicted by a single unified formula lambda(t, w, g) = base_lambda × phi(t) × psi(alignment) × norm_control(w).

**Implementation sketch**:
- Implement 8 WD variants extending existing optimizers.py:
  1. Constant WD (baseline, existing)
  2. Cosine WD schedule (existing)
  3. CWD hard (binary sign, existing)
  4. CWD soft (continuous cosine similarity — new, 5 lines)
  5. SWD (gradient-norm-aware, existing)
  6. Norm-matched WD (target-norm control: lambda_t = lambda × ||w_t||/tau, new, 10 lines)
  7. Corrective WD (Defazio term: lambda_t = lambda × eta_max/eta_t, new, 5 lines)
  8. Combined: cosine schedule + CWD soft mask
- Run all 8 on ResNet-20 + VGG-16-BN, CIFAR-10/100, 3 seeds

**Simplest version**: Implement only CWD soft and norm-matched WD — these are the two missing variants from the unification taxonomy.

**Time estimate**: 8 variants × 2 architectures × 2 datasets × 3 seeds = 96 runs × ~12 min avg = ~19 GPU hours on 8 GPUs (~2.4 hours wall clock).

**Reusable components**: iter_003 codebase, existing 7 variants already running.

---

### Candidate C: Phi Invariance Boundary Mapping — Systematic Boundary Conditions

**Hypothesis**: Phi Invariance (the empirical observation that dynamic WD methods perform similarly to constant WD under AdamW) has a well-defined boundary parameterized by: (1) WD strength lambda, (2) network scale, (3) presence of BatchNorm, (4) optimizer type. A systematic grid experiment will locate these boundaries and produce the most practically useful insight: when should practitioners bother with dynamic WD?

**Implementation sketch**:
- Fix architecture (ResNet-20), vary: lambda in {5e-5, 5e-4, 5e-3, 5e-2}, optimizer in {AdamW, SGD}, BN in {with, without}
- For each setting: run 4 WD methods (constant, cosine, CWD hard, SWD) × 3 seeds
- Metric: TOST equivalence test (Delta=0.3%) — invariance holds iff equivalence within margin
- Output: heat map of "invariance boundary" in (lambda, optimizer) space
- No new methods required — purely boundary mapping of existing methods

**Simplest version**: Lambda sweep for AdamW only. 4 lambda values × 4 methods × 3 seeds × 2 datasets. Already have lambda=5e-4 data from iter_003, so incremental = 72 new runs.

**Time estimate**: Lambda sweep: ~6 GPU hours incremental. Full BN/architecture grid: +48 runs ≈ 12 additional GPU hours.

**Reusable components**: All existing iter_003 experiments contribute directly (lambda=5e-4 is already done). Lambda sweep script trivially derived from existing run scripts.

---

## Phase 3: Self-Critique

### Against Candidate A

**Implementation reality check**: The rho_t = ||g_t||/||w_t|| diagnostic is trivially implementable (2 lines of code). Defazio's corrective term is one additional scalar computation. No library compatibility issues. The real question is: does the corrective term actually outperform cosine_schedule? From existing data, cosine_schedule already has very low variance (std=0.07%). The corrective term may provide a minor improvement below statistical detection thresholds with 3 seeds.

**Reproducibility attack**: High reproducibility. The rho_t diagnostic is purely observational. The corrective WD formula is explicit (lambda_corrected = lambda × eta_max/eta_t).

**Baseline sanity check**: Cosine WD schedule is already a well-tuned baseline. If the corrective term does not beat cosine_schedule on CIFAR-10, the contribution is weak. The rho_t diagnostic as a visualization tool remains valuable even if the algorithm is not better.

**Scope attack**: The rho_t analysis only provides a diagnostic; it does not propose a new method. As a standalone contribution it is too narrow.

**Verdict**: MODERATE — valuable as one component of a broader framework paper, but insufficient as a standalone contribution.

---

### Against Candidate B

**Implementation reality check**: We already have 7 of the 8 variants implemented. The two missing pieces are CWD soft (5 lines) and norm-matched WD (10 lines). No library issues. The one fragile point is CWD soft — the continuous cosine similarity between g_t and w_t may be noisy for early training (||w_t|| ≈ 0). Need epsilon stabilization: delta = <g_t, w_t> / (||g_t|| · ||w_t|| + eps). This is standard.

**Reproducibility attack**: All hyperparameters are standard (lambda=5e-4, tau=||w_0|| for norm-matched). Fully reproducible.

**Baseline sanity check**: Under Phi Invariance, we expect most methods to perform similarly under AdamW. From existing data: CWD hard has slightly lower mean on CIFAR-10 (90.06 vs 90.13 for constant), suggesting the binary mask may actually hurt in some regimes. CWD soft might recover this. This is falsifiable.

**Scope attack**: Applies across CIFAR-10/100 and VGG-16-BN. The contribution is "empirical characterization + unification taxonomy" — publishable even with null results.

**Verdict**: STRONG — directly executable, fills the identified gap, provides unification narrative even with null results.

---

### Against Candidate C

**Implementation reality check**: Purely an experimental grid. No new code required — just parameter sweeps of existing experiments. The TOST equivalence testing is 10 lines of scipy code. We already have 7 methods for SGD at lambda=5e-4. The key experiment is AdamW at high lambda (5e-3, 5e-2).

**Reproducibility attack**: Excellent reproducibility. All experiments are deterministic given seeds. TOST boundaries are statistically well-defined.

**Baseline sanity check**: We already have lambda=5e-4 data. From SGD data: constant at lambda=5e-4 outperforms no_wd by 0.91% on CIFAR-10. If AdamW shows the same at lambda=5e-2, it challenges Phi Invariance.

**Scope attack**: CIFAR-scale only initially; no guarantee the boundary generalizes to ImageNet scale. But CIFAR boundary mapping is a necessary first step that frames the ImageNet questions.

**Verdict**: STRONG — high impact (answers "when does dynamic WD matter?" directly), low effort (parameter sweeps), falsifiable, self-contained.

---

## Phase 4: Refinement

### Dropped Ideas
- Candidate A as standalone: dropped; incorporated as diagnostic tool within Candidates B and C.

### Strengthened Survivors

**Candidate B strengthened**:
- Remove UnifiedWD abstraction complexity — implement CWD soft and norm-matched WD as two additional optimizer variants in existing flat structure (matching iter_003 style)
- Minimal pilot: implement CWD soft, run 1 seed on CIFAR-10, compare to CWD hard — 15 minutes
- Add rho_t = ||g||/||w|| logging to training script as free diagnostic

**Candidate C strengthened**:
- Combine with existing SGD data: already have 7 methods for SGD
- Design as "boundary grid" table: rows = lambda values, columns = method pairs, cells = TOST equivalence (YES/NO) + p-value
- This table is the core Figure 1 of the paper: a clean visual answer to "when does Phi Invariance hold?"

### Additional Searches Conducted
- Confirmed: no existing paper maps the Phi Invariance boundary systematically
- Confirmed: CWD soft (continuous cosine similarity) has not been published; CWD only uses binary sign
- Confirmed: tml-epfl/why-weight-decay tracks weight/gradient norms; our training code already does this via final_ais, final_csi, final_bem
- Confirmed: Defazio (2506.02285) layer balancing insight is directly implementable; corrective term = lambda × (eta_max/eta_t)

### Selected Front-Runner

Candidate C + B combined: The boundary mapping experiment (C) is the highest-priority GPU task because it directly answers the core research question AND reuses existing data. Candidate B's contribution (CWD soft, norm-matched WD) completes the unification taxonomy with minimal code changes.

Practical execution priority order:
1. P0 (no GPU): Recompute SGD statistics from existing summary.json files; fix BEM/AIS/CSI definitions
2. P1 (6-8 GPU hours): VGG-16-BN experiments (existing run_vgg_experiments.sh ready)
3. P2 (6 GPU hours incremental): Lambda sweep for AdamW: lambda in {5e-5, 5e-4, 5e-3, 5e-2} × 4 methods × 3 seeds
4. P3 (parallel with P2): Add CWD soft + norm-matched WD; run ResNet-20 + CIFAR-10/100 × 3 seeds
5. P4 (10-14 GPU hours): ImageNet ResNet-50, 4 core methods × 3 seeds

---

## Phase 5: Final Proposal

### Title
When Does Dynamic Weight Decay Actually Matter? A Systematic Characterization of the Phi Invariance Boundary Across Optimizers, Architectures, and Regularization Strengths

### Hypothesis
Precisely falsifiable: Dynamic WD methods (scheduling, sign-alignment-aware, continuous-alignment-aware, norm-matched) show equivalent performance to constant WD under AdamW ("Phi Invariance") only when lambda/eta_effective is below a threshold tau_c. Above tau_c, WD strength is large enough that its temporal distribution matters, and alignment-aware methods outperform constant WD. Under SGD, no such invariance holds regardless of lambda/eta.

The unified condition: Phi Invariance holds iff (optimizer is adaptive AND lambda × eta_effective << 1)

### Motivation
Three concrete practitioners' problems this solves:

1. "Should I bother with CWD instead of AdamW? Under what conditions?" — directly answered by the boundary map.
2. "My model uses AdamW with standard lambda=1e-4. Will any fancy dynamic WD papers help?" — answer: probably not (Phi Invariance regime).
3. "My fine-tuning uses SGD with large lambda. What is the best WD strategy?" — answer: alignment-aware methods significantly help (from our SGD data: 18.3x effect size difference vs AdamW).

Gap in existing work: CWD (ICLR 2026) shows binary sign alignment helps under AdamW on LLMs (338M-2B). We show that at standard vision task scales and lambda values, even CWD's improvement is marginal (within statistical noise), and that the effective operating regime for alignment-aware WD is much narrower than implied. The AdamW ell_inf implicit bias (arXiv:2404.04454) provides the theoretical mechanism: AdamW's sign normalization makes parameter trajectories insensitive to WD temporal distribution.

### Method

**Step 1: Recompute existing SGD baseline statistics** (0 GPU hours)

Load iter_003/exp/results/sgd_baseline/cifar10/resnet20/*/seed_*/summary.json. Key data already on disk:
- SGD constant, seeds 42/123/456: best_test_acc = 91.30, 91.18, 91.17 → mean = 91.22, std = 0.07
- SGD no_wd, seeds 42/123/456: best_test_acc = 90.39, 90.19, 90.33 → mean = 90.30, std = 0.10
- Effect: delta = 0.91%, Cohen's d ≈ 12 (large)
- AdamW constant: 90.13±0.31; AdamW no_wd: 90.08±0.32; delta = 0.05% (noise level)
- Run scipy pairwise t-tests + Bonferroni-Holm correction
- Honest report: Bonferroni-corrected, only constant vs no_wd (SGD) is significant

**Step 2: Fix metric definitions** (0 GPU hours)

- BEM: signed (remove [0,1] bounds claim); debug half_lambda=0.000 bug (trace optimizers.py BEM code path)
- AIS: range [-1,1]; clarify Delta-L sign convention; add per-layer mean formula
- CSI: normalize each component to constant baseline value before equal-weight averaging; CSI(constant)=1.0 by construction

**Step 3: VGG-16-BN architecture diversity** (6-8 GPU hours, script already at iter_003/exp/code/run_vgg_experiments.sh)

- Execute existing run_vgg_experiments.sh
- 5 methods × 2 datasets × 3 seeds = 30 runs
- Validates architecture independence of Phi Invariance
- Key check: does VGG-16-BN (no residual connections) show more WD sensitivity than ResNet-20?

**Step 4: Lambda sweep for boundary mapping** (6 GPU hours incremental)

- Add lambda in {5e-5, 5e-3, 5e-2} to existing lambda=5e-4 (already done)
- AdamW only; 3 new lambda values × 4 methods (constant, cosine, CWD hard, SWD) × 3 seeds × 2 datasets
- Incremental runs: 3 × 4 × 3 × 2 = 72 runs × ~5 min = ~6 GPU hours
- Core output: TOST equivalence table (rows = lambda, columns = method pairs vs constant), the main "boundary" Figure

**Step 5: CWD soft (continuous cosine alignment)** (3 GPU hours)

- Implement: lambda_t = lambda_base × max(0, cosine_sim(g_t, w_t))
- 1 new variant × 2 datasets × 3 seeds × 2 architectures = 12 runs × ~12 min = ~2.4 GPU hours
- Tests Gap 3 from literature survey: does continuous modulation outperform binary masking?

**Step 6: Norm-matched WD** (3 GPU hours)

- Implement: lambda_t = lambda_base × (||w_t|| / tau) where tau = target norm (EMA of ||w_0||)
- 1 variant × 2 datasets × 3 seeds × 2 architectures = 12 runs

**Step 7: ImageNet ResNet-50** (10-14 GPU hours, 4 × RTX PRO 6000)

- 4 methods (constant, cosine, CWD hard, no_wd) × 3 seeds = 12 runs
- Each GPU handles 3 sequential runs (4 GPUs × 3 runs ≈ 12-18 hours wall clock)
- Key check: does Phi Invariance hold at ImageNet scale with standard lambda?

### Simplest Version

The minimum publishable experiment: re-analyze existing iter_003 data + SGD baseline with corrected statistics. This alone recovers the central claim (SGD shows alignment-sensitive WD effects; AdamW does not at lambda=5e-4). Add the lambda sweep (Step 4) for the boundary contribution. All other steps add robustness but are not load-bearing for the core claim.

Pilot experiment (15 minutes on 1 GPU): implement CWD soft, run ResNet-20 + CIFAR-10, seed=42. If CWD soft > CWD hard > constant at any lambda, the continuous alignment signal is informative; if CWD soft ≈ CWD hard, the binary mask captures most signal.

### Baselines

1. **Constant WD (AdamW, lambda=5e-4)**: Primary baseline. Expected CIFAR-10 ResNet-20: 90.13±0.31% (from existing data).
2. **No WD (AdamW)**: Lower bound. Expected: 90.08±0.32%. AdamW WD advantage = 0.05% (within noise — Phi Invariance regime).
3. **Constant WD (SGD, lambda=5e-4)**: SGD baseline. Expected: 91.22±0.07% (recomputed from summary.json).
4. **No WD (SGD)**: SGD lower bound. Expected: 90.30±0.10%. SGD advantage = 0.91% (statistically significant, Cohen's d ≈ 12, 18.3x vs AdamW).

### Experimental Plan

| Experiment | Status | GPU hours | Primary Purpose |
|-----------|--------|-----------|----------------|
| SGD baseline reanalysis | Data on disk | 0 | Correct Table 5; establish 18.3x effect |
| Metric fix (BEM/AIS/CSI) | No GPU needed | 0 | Eliminate data integrity issues |
| VGG-16-BN (CIFAR-10/100) | Script ready | 6-8 | Architecture diversity |
| Seed 789+999 extension | Script exists | 4-6 | Raise from 3 to 5 seeds |
| Lambda sweep: {5e-5, 5e-3, 5e-2} × AdamW | New runs | 6 | Boundary mapping (core Figure) |
| CWD soft (continuous cosine) | 10 lines code | 3 | Continuous alignment vs binary |
| Norm-matched WD | 10 lines code | 3 | Norm control sub-approach |
| ImageNet ResNet-50 (4 methods × 3 seeds) | Script needed | 12-18 | Scale validation |

**Ablation schedule**:
- First establish boundary (lambda sweep): tells us whether high-lambda regime shows alignment-sensitivity
- Then test CWD soft vs CWD hard at high-lambda regime (where it matters most)
- Norm-matched WD tested separately to understand if norm control captures the same boundary

**Metrics reported**:
- Primary: test accuracy mean ± std, paired t-test vs constant baseline (Bonferroni-Holm)
- Secondary: TOST equivalence test (Delta=0.3% margin), Cohen's d
- Diagnostic: rho_t = ||g||/||w|| trajectory, weight norm stability, AIS evolution, CSI

### Resource Estimate

- Total GPU hours: 40-50 hours
- Wall-clock time (8× RTX PRO 6000 Blackwell): ~6-8 hours for all non-ImageNet experiments; ~12-18 hours for ImageNet
- Model sizes: ResNet-20 (~0.3M params), VGG-16-BN (~15M params), ResNet-50 (~25M params)
- Per-experiment time: CIFAR ~10-15 min/run; ImageNet ~4-6 hours/run
- Code changes: ~50 lines new code (CWD soft, norm-matched WD, rho_t logging, TOST tests); all based on existing iter_003 codebase
- Zero new dependencies required

### Risk Assessment

**Engineering risks**:

1. CWD soft instability at early training (probability: medium): <g_t, w_t> / (||g_t||·||w_t||) is undefined when w_0 ≈ 0. Mitigation: epsilon=1e-8 stabilization; warm up with standard WD for first 5 epochs.

2. BEM=0.000 bug (probability: high, already confirmed in review): Likely writes lambda value instead of computed BEM. Fix: trace through optimizers.py BEM computation, add unit test with known inputs.

3. ImageNet training instability at high lambda (probability: low): ResNet-50 with lambda=5e-2 may diverge. Mitigation: clip lambda_eff to [1e-6, 1e-2]; test with seed=42 first before running all seeds.

4. VGG-16-BN LR mismatch: VGG uses LR=1e-3 (not 1e-1 as ResNet). Existing run_vgg_experiments.sh already handles this correctly.

**Scientific risks**:

1. VGG-16-BN breaks Phi Invariance (probability: 20-30%): VGG without residual connections may be more sensitive to WD modulation. Mitigation: this is a positive result — frame as "Phi Invariance is architecture-dependent; residual connections contribute to WD insensitivity." Cite AdamO's Radial Tug-of-War as mechanism.

2. High-lambda regime shows no alignment-sensitivity under AdamW (probability: 30%): If Phi Invariance holds even at lambda=5e-2, the boundary is outside practical regimes. Mitigation: run SGD at same lambda values; AdamW's insensitivity becomes the more striking finding.

3. CWD soft == CWD hard (probability: 50%): Binary sign alignment may capture most available signal. Mitigation: if true, this is an informative null result that quantifies the information content of continuous vs binary alignment.

4. ImageNet breaks invariance (probability: 20-30%): Large-scale training may reveal dynamics not visible in CIFAR. Mitigation: two valid narrative paths are already designed (Path A: universal Phi Invariance; Path B: scale-dependent transition). Either path produces a complete paper.

### Novelty Claim

1. **First systematic boundary mapping of Phi Invariance**: We show that alignment-aware WD improves over constant WD only when lambda/eta_effective exceeds a threshold below practical vision settings for AdamW but not for SGD. This is a direct practitioner answer to "when does dynamic WD matter?"

2. **Continuous alignment modulation (CWD soft)**: First implementation and evaluation of cosine-similarity-continuous WD modulation, filling Gap 3 from the literature survey. Quantifies information in continuous vs binary alignment signal.

3. **Unified experimental evaluation**: All four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) benchmarked on the same compute budget with the same diagnostic metrics — the standardized evaluation framework the field currently lacks.

4. **Diagnostic toolkit**: rho_t trajectory, per-layer AIS evolution, TOST equivalence tables — reusable tools building on Defazio (2506.02285) layer balancing and tml-epfl/why-weight-decay evaluation scaffolding.

Even if all empirical results are null (Phi Invariance holds universally), this is publishable: "We ran every major dynamic WD variant under identical conditions; none improved over constant WD under AdamW at standard lambda. Here is when they would." This is a benchmark paper with a message, not just a negative result.

---

## Appendix: Code Sketch

The following changes are needed in iter_003/exp/code/ (total: ~50 lines):

```python
# train_unified.py — add rho_t logging (2 lines):
rho_t = grad_norm / (weight_norm + 1e-8)
metrics['rho_t_ema'] = 0.9 * metrics.get('rho_t_ema', rho_t) + 0.1 * rho_t

# optimizers.py — add CWD soft (5 lines):
if wd_method == 'cwd_soft':
    cos_sim = torch.dot(grad_flat, weight_flat) / (
        grad_flat.norm() * weight_flat.norm() + 1e-8)
    effective_wd = wd * max(0.0, cos_sim.item())

# optimizers.py — add norm-matched WD (8 lines):
if wd_method == 'norm_matched':
    current_norm = p.data.norm().item()
    effective_wd = wd * (current_norm / (target_norm + 1e-8))
    # target_norm initialized to weight norm at step 0
```

Lambda sweep script (derived from existing run_sgd_baseline.sh):

```bash
for lambda in 5e-5 5e-3 5e-2; do
    for method in constant cosine_schedule cwd_hard swd; do
        for seed in 42 123 456; do
            # run train_unified.py --wd $lambda --wd_method $method --seed $seed
        done
    done
done
```

The entire experimental program builds on iter_003 code with ~50 lines of new code. No new dependencies. No new infrastructure. Fully reproducible from existing training scripts.

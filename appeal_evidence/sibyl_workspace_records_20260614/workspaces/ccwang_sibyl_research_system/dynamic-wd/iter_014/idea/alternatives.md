# Backup Ideas for Pivot

## Alternative 1: Spectral-Homeostatic Weight Decay via Renormalization Group Flow

### Summary
If the PID control law unification proves too coarse (H1 falsified) or alignment signals are genuinely uninformative (H3, H7 both fail), pivot to a two-factor WD strategy grounded in interdisciplinary principles:
- **Temporal factor** (from statistical physics): WD schedule lambda_schedule(t) driven by the spectral condition number kappa(t) = sigma_1 / sigma_r of weight matrices, following a Renormalization Group (RG) flow interpretation. The optimal schedule is non-monotone: low early (allow relevant singular modes to emerge), high mid-training (compress irrelevant modes), low late (avoid overshooting).
- **Per-parameter factor** (from neuroscience): Homeostatic synaptic scaling where decay is proportional to the degree of non-constructive participation: phi(g_k, w_k) = 1 - max(0, cos(angle(g_k, w_k))).

### Novelty Score: 8/10 (higher raw novelty, fewer close prior works)

### Why This Survives If Front-Runner Fails
This alternative does NOT depend on the PID unification or the alignment signal being informative in the control-theoretic sense:
- The RG schedule depends only on spectral statistics (SVD), a fundamentally different signal from gradient-weight alignment
- The homeostatic factor uses alignment but with specific neuroscience-grounded motivation (activity-weighted downscaling)
- Both interdisciplinary connections have formal mathematical correspondences, not just metaphors

### Key Risks
- SVD computation overhead: O(min(m,n)^2 * max(m,n)) per layer per step. Mitigation: compute every K=100 steps with EMA, or use nuclear-norm/Frobenius-norm ratio as tractable proxy.
- The non-monotone RG schedule prediction may not hold empirically. Mitigation: diagnostic experiment tracking kappa(t) under standard schedules.
- AlphaDecay (NeurIPS 2025) partially occupies the spectral-based WD niche. Differentiation: AlphaDecay uses static heavy-tail alpha exponent; we propose dynamic condition number tracking via RG flow.

### Estimated Additional Work
- Implementation: 1-2 days
- Theory: 1-2 days (formalize RG flow equations)
- Experiments: Same benchmarks, add spectral condition number tracking

---

## Alternative 2: Fisher-Informed Weight Decay -- Geometry-Aware Regularization via Natural Gradient Alignment

### Summary
If the unified framework approach encounters complexity barriers or fails to produce competitive results, pivot to a single, clean algorithmic contribution: Fisher-weighted weight decay.

**Core idea**: Apply weight decay as a natural gradient by scaling per-parameter decay strength by the inverse Fisher information (approximated by Adam's v_t):
```
fisher_weight_i = 1.0 / (sqrt(v_hat_i) + epsilon)
fisher_weight_i = fisher_weight_i / mean(fisher_weight_i)  # normalize
w_i -= eta * lambda * fisher_weight_i * w_i
```

### Novelty Score: 9/10 (highest raw novelty -- no paper implements or tests this)

### Why This Survives If Front-Runner Fails
- Completely independent of the PID unification framework
- Does not depend on the alignment signal being informative
- One-line modification to AdamW with zero computational overhead (reuses v_t)
- Clean theoretical motivation from information geometry and Bayesian optimal shrinkage
- FAdam (ICLR 2025 submission) proposes the principle conceptually but does NOT implement or test Fisher-weighted WD

### Key Risks
- Noisy v_t: Adam's second moment is a biased estimator of the diagonal Fisher. Dividing WD by sqrt(v_t) could amplify noise for rarely-updated parameters. Mitigation: normalization by mean(fisher_weight) + epsilon floor.
- Fisher heterogeneity may be too small to matter: If v_t is approximately constant across parameters, Fisher-weighted WD reduces to standard WD. Mitigation: measure heterogeneity first on ResNet and ViT.
- The improvement may be too small for ImageNet. Mitigation: focus on theoretical contribution (information-geometric motivation, cold posterior connection) rather than claiming large accuracy gains.

### Estimated Additional Work
- Implementation: 1 day (one-line change to AdamW)
- Theory: 1-2 days (Bayesian shrinkage analysis, cold posterior connection)
- Experiments: Same benchmarks, add Fisher information heterogeneity tracking

---

## Alternative 3: Budget-Efficient Weight Decay -- A Pre-Registered Empirical Falsification Study

### Summary
If the theoretical framework encounters mathematical obstacles or CPR proves functionally equivalent and impossible to beat, pivot to a purely empirical contribution:

**"Does Gradient-Weight Alignment Carry Marginal Information for Weight Decay? A Pre-Registered Falsification Study with Unified Evaluation Metrics"**

This paper asks and answers:
1. Is delta_hat_t a temporal proxy or does it carry genuine geometric information? (Temporal predictability gate)
2. Does average alignment predict generalization better than worst-case alignment? (Sun et al. CVPR 2025 extension test)
3. What is the measured AIS of gradient-weight alignment on CIFAR-100 and ImageNet? (First quantitative AIS measurement)
4. Does BEM reverse any WD method rankings? (Compute-fairness validation)
5. Does CWD work through alignment information or WD magnitude reduction? (Random-mask and halved-WD ablation)

### Novelty Score: 6/10 (reduced post-GWA NeurIPS 2025)

### Why This Survives If Everything Else Fails
- Stands regardless of theory: valuable as positive or negative result
- Pre-registered design with explicit falsification criteria
- Three standardized metrics are publishable as standalone contribution
- All experiments already planned as part of the front-runner's design -- no wasted compute

### Key Risks
- Effect sizes may be too small to detect at CIFAR scale (need 5+ seeds or focus on ImageNet)
- Negative results harder to publish (mitigated by pre-registration + TMLR/NeurIPS D&B track)
- GWA (NeurIPS 2025) partially answers "does alignment carry information?" affirmatively

### Estimated Additional Work
- Implementation: 1 day (instrumentation)
- Experiments: Same as front-runner (no additional compute)
- Writing: shift focus from theory to empirical methodology

---

## Alternative 4: Cooperative Alignment Control -- Bio-Inspired Noise-Robust WD

### Summary
If the contrarian's noise analysis proves devastating (alignment SNR < 1 at all practical batch sizes), pivot to a noise-robust WD design inspired by biological feedback systems:
- **Dead zone** (from bone mechanostat): No WD adaptation when alignment is in [delta_low, delta_high], filtering stochastic noise
- **Hill-function cooperativity** (from quorum sensing): Tunable switch sharpness n interpolating between CWD (n=inf) and linear (n=1), with noise-dependent optimal n
- **Inter-layer cooperative sensing** (from quorum sensing): Per-layer WD modulated by both local and global alignment signal

### Novelty Score: 7/10

### Why This Survives
- Directly addresses the noise problem that may kill the front-runner's continuous alignment
- Dead zone + Hill function family provides first continuous parameterization containing CWD as a limit
- Testable prediction: optimal n increases with batch size

### Key Risks
- Too many hyperparameters (K_base, dead_width, n, alpha, beta) unless defaults derived from alignment statistics
- Biological motivation may be viewed as superficial by reviewers

---

## Pivot Decision Tree

```
Start with Front-Runner (UDWDC / PID Unification)
    |
    +-- H1 passes (unification works) --> Continue with UDWDC
    |   |
    |   +-- H2 passes (prescriptive rho* works) --> Full paper
    |   |
    |   +-- H2 fails (rho* not competitive) --> Keep taxonomy + metrics,
    |       drop algorithm, merge with Alt 3
    |
    +-- H1 fails (unification too coarse) --> Check H7 (temporal gate)
        |
        +-- H7 passes (alignment informative) --> Pivot to Alt 1 or Alt 2
        |   (Spectral-Homeostatic or Fisher-Weighted WD)
        |
        +-- H7 fails (alignment is temporal proxy) --> Pivot to Alt 3
            (negative result paper) or Alt 4 (noise-robust design)
```

## Decision Timing
- **After Phase 2 (CIFAR Full, ~20 GPU-hours)**: Check H1, H2, H7. If both H1 and H7 fail, early pivot.
- **After Phase 3 (ImageNet, ~120 GPU-hours)**: Final decision. All experiments needed for any alternative have been run.

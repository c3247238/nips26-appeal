# Glossary

Unified terminology for the paper. All section writers, critics, and editors must use these exact terms.

---

## Core Concepts

| Term | Definition | Preferred Usage |
|------|-----------|----------------|
| Weight decay (WD) | Multiplicative shrinkage of parameters toward zero at each optimization step | "weight decay" (lowercase), abbreviate as "WD" after first use per section |
| Decoupled weight decay | WD applied as a separate multiplicative step, independent of gradient scaling (as in AdamW) | "decoupled weight decay" or "decoupled WD" |
| Dynamic weight decay | Any WD strategy where the effective decay rate varies across parameters or time | "dynamic WD" (not "adaptive WD" except when referring to PMP-WD specifically) |
| Adaptive weight decay | WD that adapts based on measured training state (e.g., PMP-WD using rho feedback) | "adaptive WD" -- use specifically for state-feedback methods like PMP-WD |
| Constant weight decay | WD with phi = 1 (uniform, time-invariant decay) | "constant WD" (not "fixed WD" or "static WD") |

## Framework Terms

| Term | Definition | Preferred Usage |
|------|-----------|----------------|
| Phi modulator | The function phi(t, w, g) that modulates per-parameter WD | "Phi modulator" or "phi function" (capitalize "Phi" when referring to the framework) |
| Phi framework | The unified mathematical abstraction subsuming all dynamic WD methods as phi special cases | "the Phi framework" (not "Phi Modulator Framework" in body text -- too long) |
| Phi invariance | The empirical observation that accuracy is invariant to phi choice under AdamW at standard rho | "Phi invariance" (not "method invariance" or "schedule invariance") |
| Phi spread | max(accuracy) - min(accuracy) across all 7 WD methods for a given configuration | "Phi spread" or "method spread" (interchangeable) |
| Modulation axis | The dimension along which phi varies: temporal, directional, spatial, or target-norm | "modulation axis" |

## Diagnostic Metrics (abbreviations)

| Abbreviation | Expansion | Notes |
|-------------|-----------|-------|
| BEM | Budget Equivalence Metric | Always expand on first use per section |
| CSI | Coupling Stability Index | Always expand on first use per section |
| AIS | Alignment Informativeness Score | Always expand on first use per section |

## WD Methods (7 experimental methods)

| Method Name | Code Name | Description |
|-------------|-----------|-------------|
| Constant WD | `constant` | phi = 1; the baseline |
| Cosine schedule | `cosine_schedule` | phi = 0.5*(1 + cos(pi*t/T)); temporal decay |
| CWD (hard) | `cwd_hard` | phi = indicator[sign(w) = sign(u)]; binary sign-alignment mask (Chen et al. ICLR 2026) |
| SWD | `swd` | phi = ||g|| / ||g||_mean; gradient-norm scaling (Xie et al. NeurIPS 2023) |
| Half-lambda | `half_lambda` | phi = 0.5; constant at half the baseline WD |
| Random mask | `random_mask` | phi = Bernoulli(0.5); stochastic binary mask (ablation control) |
| No WD | `no_wd` | phi = 0; zero weight decay |

## Proposed Algorithms

| Algorithm | Expansion | Definition |
|-----------|-----------|------------|
| PMP-WD | Pontryagin Maximum Principle Weight Decay | lambda*(t) = clip(kappa * (rho* - rho_hat_t)^+, 0, lambda_max); state-feedback optimal WD |
| QA-WD | Quadratic-Alignment Weight Decay | lambda*(t) = beta_0 * delta_hat^2_t; derived from RG beta function (equivalent to PMP-WD in moderate-alignment regime) |
| SPWD | Spectral-Phase-Transition Weight Decay | lambda_t = lambda_0 * (1 + alpha * tanh(-v_t)); based on stable rank velocity (backup candidate, not primary contribution) |

## Key Physical Quantities

| Term | Definition | Preferred Usage |
|------|-----------|----------------|
| Gradient-to-weight ratio | rho = ||g|| / ||w||; the central quantity governing WD regime | "gradient-to-weight ratio" on first use, then "rho" or "the ratio" |
| Alignment | Cosine similarity between gradient and weight vectors | "gradient-weight alignment" (not "alignment" alone, which is ambiguous) |
| Alignment signal | The cosine similarity delta_hat used as feedback in adaptive WD | "alignment signal" |
| Stability cost | The generalization penalty from WD-induced coupling instability | "stability cost" (not "instability penalty" or "regularization cost") |
| Alignment benefit | The generalization gain from WD modulation that exploits informative gradient-weight geometry | "alignment benefit" |

## Architecture Terms

| Term | Preferred Usage | Notes |
|------|----------------|-------|
| Batch normalization | "batch normalization" or "BN" | Always abbreviate as "BN" after first use |
| ResNet-20 | "ResNet-20" | Hyphenated, capital R, capital N |
| VGG-16-BN | "VGG-16-BN" | Always include "-BN" to distinguish from VGG without BN |
| ResNet-20-NoBN | "ResNet-20-NoBN" | The BN-removed ablation architecture |
| ResNet-50 | "ResNet-50" | ImageNet architecture (future work) |

## Optimizer Terms

| Term | Preferred Usage | Notes |
|------|----------------|-------|
| AdamW | "AdamW" | Capital A, capital W; never "Adam-W" or "adam-w" |
| SGD | "SGD" | With momentum unless stated otherwise |
| AdamC | "AdamC" | Defazio's feedforward WD schedule; distinguish from PMP-WD (state-feedback) |

## Statistical Terms

| Term | Preferred Usage | Notes |
|------|----------------|-------|
| Paired t-test | "paired t-test" | Lowercase; for comparing methods using same seeds |
| Bonferroni correction | "Bonferroni correction" | For multiple comparison adjustment |
| TOST | "Two One-Sided Tests" or "TOST" | For equivalence testing |
| Cohen's d | "Cohen's d" | Lowercase d, italic in LaTeX |
| Effect size | "effect size" | Prefer Cohen's d as the specific measure |

## Control Theory Terms

| Term | Preferred Usage | Notes |
|------|----------------|-------|
| Pontryagin Maximum Principle | "Pontryagin Maximum Principle" or "PMP" | Expand on first use |
| State-feedback | "state-feedback" (hyphenated) | PMP-WD uses measured rho_hat as state |
| Feedforward | "feedforward" (one word) | AdamC uses scheduled gamma_t, no measurement |
| Riccati equation | "Riccati equation" | Source of feedback gain kappa |
| Renormalization group | "renormalization group" or "RG" | The physics framework for dual derivation |

## Banned Terms (do not use)

| Banned | Use Instead |
|--------|-------------|
| fine-tuning (in WD context) | "WD scheduling" or "WD modulation" |
| novel (without quantification) | state the specific novelty |
| significantly (without p-value) | provide exact p-value or effect size |
| state-of-the-art | "best reported result" with citation |
| groundbreaking / game-changing / revolutionary | do not use |
| furthermore / moreover / it is worth noting | remove or restructure sentence |

---

*Version: 2.0 | Iteration: 7 | Generated: 2026-03-18*

# Glossary

Unified terminology for the paper. All section writers must use these exact terms.

## Core Concepts

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| Weight decay (WD) | Regularization technique that shrinks parameters toward zero by subtracting $\eta_t \lambda w_t$ at each step | Always "weight decay", never "weight regularization" or "L2 penalty" (the latter is mathematically distinct for Adam) |
| Gradient-to-weight ratio | Per-layer ratio $\rho_t^l = \|g_t^l\| / \|w_t^l\|$ measuring the relative magnitude of gradient updates to parameter norms | Abbreviated as "GW ratio" or "$\rho_t$" after first use; never "gradient ratio" alone |
| Control error | Difference between measured and target GW ratio: $e_t^l = \rho_t^l - \rho^*(t)$ | Always "control error", not "tracking error" or "deviation" |
| Target trajectory | Prescribed trajectory $\rho^*(t) = \eta_t / \tau$ that the controller tracks | Always "$\rho^*(t)$" in equations; "target trajectory" in prose |
| Gradient-weight alignment | Cosine similarity between gradient and weight vectors: $\alpha_t^l = \cos(g_t^l, w_t^l)$ | Use "$\alpha_t$" for alignment cosine; "$\delta_t$" reserved for theoretical sections following Sun et al. convention |
| Alignment signal | The gradient-weight alignment $\alpha_t$ used as a feedback signal for WD modulation | "Alignment signal", not "alignment feature" or "alignment information" |
| EMA timescale | The time constant $\tau = 1/(\lambda_0 \cdot \eta_0)$ governing the exponential decay rate of parameters under WD | Always "EMA timescale", not "decay timescale" or "relaxation time" |

## Methods

| Term | Abbreviation | Definition |
|------|-------------|-----------|
| Fixed Weight Decay | FixedWD | Standard SGDW/AdamW with constant $\lambda$ across training |
| Cautious Weight Decay | CWD | Alignment-aware WD with binary sign mask: apply WD only when $\alpha_t < 0$ (ICLR 2026) |
| Scheduled Weight Decay | SWD | Gradient-norm-aware WD scheduling (NeurIPS 2023) |
| Constrained Parameter Regularization | CPR | Augmented Lagrangian constraint on weight norms (NeurIPS 2024) |
| Defazio Corrective WD | DefazioCorrective | LR-proportional corrective WD term (Defazio 2025) |
| No Weight Decay | NoWD | Null baseline with $\lambda = 0$ |
| Unified Dynamic Weight Decay Control | UDWDC | Our proposed proportional controller closing the $\rho_t$ control loop |
| UDWDC-v2 | UDWDC-v2 | Stability-fixed variant with EMA-smoothed $\rho_t$ and floor clipping |

## Evaluation Metrics

| Term | Abbreviation | Definition |
|------|-------------|-----------|
| Budget Equivalence Metric | BEM | Accuracy improvement per unit of total WD budget: $(\text{acc} - \text{acc}_{\text{NoWD}}) / \text{TotalWDBudget}$ |
| Coupling Stability Index | CSI | Inverse variance of $\rho_t$ over training, measuring stability of the gradient-weight coupling |
| Alignment Informativeness Score | AIS | Spearman correlation between alignment signal and optimal WD decision |
| Signal-to-noise ratio | SNR | Ratio $|\mathbb{E}[\alpha_t]| / \text{Std}[\alpha_t]$ measuring alignment signal quality |
| Total WD budget | TotalWDBudget | Cumulative $\sum_t \sum_l \lambda_t^l \|w_t^l\|^2$ over training |

## Control Theory Terms

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| PID controller | Proportional-Integral-Derivative controller from classical control theory | Always "PID controller", not "PID algorithm" |
| Proportional gain ($K_p$) | Gain applied to current error $e_t$ | "Proportional gain" or "$K_p$" |
| Integral gain ($K_i$) | Gain applied to accumulated (EMA-smoothed) error | "Integral gain" or "$K_i$" |
| Derivative gain ($K_d$) | Gain modulated by alignment signal $\alpha_t$ | "Derivative/alignment gain" or "$K_d$"; note this maps to alignment, not temporal derivative |
| Open-loop control | Control without feedback from the controlled variable | FixedWD is open-loop (no feedback from $\rho_t$) |
| Closed-loop control | Control with feedback from the controlled variable | UDWDC closes the loop by measuring $\rho_t$ |
| Steady state | Equilibrium value that $\rho_t$ converges to under constant conditions | "Steady state" or "fixed point", not "convergence point" |
| Control law | The rule mapping measurements to control actions ($\rho_t \to \lambda_t$) | "Control law", not "control policy" or "control strategy" |

## Architecture Terms

| Term | Abbreviation | Definition |
|------|-------------|-----------|
| Batch Normalization | BN | Normalization over mini-batch dimension (Ioffe & Szegedy, 2015) |
| Layer Normalization | LN | Normalization over feature dimension (Ba et al., 2016) |
| ResNet-20 | ResNet-20 | 20-layer residual network for CIFAR |
| VGG-16-BN | VGG-16-BN | 16-layer VGG network with batch normalization for CIFAR |
| ResNet-50 | ResNet-50 | 50-layer residual network for ImageNet |
| ResNet-101 | ResNet-101 | 101-layer residual network for ImageNet |
| Vision Transformer Small | ViT-S/16 | Small Vision Transformer with 16x16 patch size |
| Distributed Data Parallel | DDP | PyTorch's data-parallel training across multiple GPUs |

## Statistical Terms

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| Seeds | Random seed values for reproducibility: {42, 123, 456} (CIFAR) or {42, 123, 456, 789, 2024} (ImageNet) | Always report as "mean $\pm$ std across $N$ seeds" |
| TOST | Two One-Sided Tests for equivalence testing at $\delta = 0.5\%$ | Used for null-result claims only |
| Cohen's d | Standardized effect size measure | Report alongside p-values |
| Bonferroni correction | Multiple comparison correction | Apply when comparing $>2$ methods simultaneously |

## Preferred Phrasing

| Preferred | Avoid |
|-----------|-------|
| weight decay | weight regularization, L2 regularization (distinct for Adam) |
| fine-tuning | finetuning, fine tuning |
| hyperparameter | hyper-parameter |
| per-layer | per layer (use hyphen as compound modifier) |
| closed-loop | closed loop (use hyphen as compound modifier) |
| batch size | batchsize, batch-size |
| learning rate | learning-rate (no hyphen as noun) |
| cosine annealing | cosine decay, cosine schedule (use "cosine annealing" consistently) |
| generalization gap | generalization error, test-train gap |
| multi-seed | multi seed |
| budget-efficient | budget efficient (use hyphen as compound modifier) |

## Abbreviations (First Use)

All abbreviations must be defined at first use in each section. Common abbreviations:

| Abbreviation | Expansion |
|-------------|-----------|
| WD | Weight Decay |
| GW ratio | Gradient-to-Weight ratio |
| BEM | Budget Equivalence Metric |
| CSI | Coupling Stability Index |
| AIS | Alignment Informativeness Score |
| SNR | Signal-to-Noise Ratio |
| PID | Proportional-Integral-Derivative |
| EMA | Exponential Moving Average |
| SGD | Stochastic Gradient Descent |
| SGDW | SGD with decoupled Weight decay |
| DDP | Distributed Data Parallel |
| BN | Batch Normalization |
| LN | Layer Normalization |

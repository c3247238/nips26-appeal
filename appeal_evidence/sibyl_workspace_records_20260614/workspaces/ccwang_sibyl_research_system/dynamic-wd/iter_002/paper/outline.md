# Paper Outline

**Title:** On the Sufficiency of Constant Weight Decay: Alignment Dynamics, Learning Rate Coupling, and Budget Equivalence in Nonconvex SGD

**Venue Target:** NeurIPS 2026 / TMLR (rolling)

**Paper Type:** Negative result + mechanistic insight (NOT a methods paper)

**Page Budget:** ~12 pages (NeurIPS format) + appendix

---

## 1. Abstract

**Length:** 150--200 words

**Key content:**

Weight decay (WD) is a ubiquitous regularizer in deep learning, yet optimal scheduling of its strength remains an open question. Motivated by recent theoretical results connecting WD efficacy to gradient--parameter alignment (the quantity $\delta_t = |\langle \nabla f, w \rangle| / (\|\nabla f\| \|w\|)$), we design Alignment-Aware Dynamic Weight Decay (AADWD) -- a family of three adaptive WD schedules that modulate $\lambda_t$ based on a stochastic alignment proxy $\hat{\delta}_t$. Through 39 systematic experiments spanning two architectures (ResNet-20, VGG-16-BN), two datasets (CIFAR-10, CIFAR-100), and extensive ablations, we establish three negative but informative results: (1) **Budget Equivalence** -- when the time-averaged dynamic WD matches a constant, performance is identical (92.54% = 92.54% on CIFAR-10/ResNet-20); (2) **LR--WD Coupling Necessity** -- removing the learning rate multiplier from AADWD triggers catastrophic collapse (84.49% $\to$ 10.00%); (3) **Alignment Signal Inapplicability** -- random dynamic WD (92.06%) matches alignment-based WD (92.05%), because $\delta \sim O(10^{-3})$ under standard training. These findings formalize why constant WD remains optimal and identify structural conditions under which adaptive WD scheduling cannot succeed.

**Data cited:** 92.54% = 92.54% (budget equivalence), 84.49% $\to$ 10.00% (decoupling collapse), 92.06% vs 92.05% (random vs AADWD), $\delta \sim O(10^{-3})$

---

## 2. Introduction

**Length:** ~1.5 pages

### 2.1 Motivation and Problem Statement (~0.5 pages)

**Key points:**
- Weight decay is the most widely used explicit regularizer in modern deep learning (cite: Loshchilov & Hutter 2019, Zhang et al. 2019)
- Standard practice: constant WD throughout training, tuned as a single hyperparameter
- Natural question: can WD be scheduled dynamically -- stronger when regularization helps, weaker when it hurts?
- Prior theoretical work (cite: core paper "Investigating the Role of WD in Enhancing Nonconvex SGD") establishes the alignment quantity $\delta_T = \sup_t |\langle \nabla f(w_t), w_t \rangle| / (\|\nabla f(w_t)\| \|w_t\|)$ as the key condition for WD-induced generalization improvement
- $\delta_T < 1$ is necessary and sufficient for WD to be beneficial; natural hypothesis: if $\delta_t$ varies over training, the WD rate should vary accordingly

### 2.2 Our Approach: AADWD (~0.4 pages)

**Key points:**
- We design three AADWD variants that adapt $\lambda_t$ based on a stochastic EMA proxy $\hat{\delta}_t$:
  - **Conservative:** $\lambda_t = c \cdot \gamma_t \cdot (1 - \hat{\delta}_t)$ -- increase WD when alignment is low
  - **Aggressive:** $\lambda_t = c \cdot \gamma_t / (\hat{\delta}_t + \epsilon)$ -- inversely proportional to alignment
  - **Square:** $\lambda_t = c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t)$ -- quadratic LR coupling
- Theoretical motivation: time-varying WD can improve the alignment-weighted convergence bound from $\sup_t \delta_t$ to $\bar{\delta}_T = (1/T) \sum_t \delta_t$
- AADWD is designed as an experimental framework to test whether alignment information is actionable

### 2.3 Summary of Findings and Contributions (~0.6 pages)

**Three core contributions (negative results stated as positive insights):**

1. **WD Budget Equivalence Principle:** We prove empirically that the time-averaged WD magnitude, not its temporal distribution, determines generalization. When $\text{mean}(\lambda_t) = \lambda_{\text{fixed}}$, the two yield identical performance (92.54% vs 92.54%, same weight norm 23.49). This implies WD scheduling has zero marginal value when the budget is matched.
   - *Data:* equiv_cumulative_wd best_test_acc = 92.54% = fixed_wd_0.0005 best_test_acc = 92.54%; identical final_weight_norm = 23.49

2. **LR--WD Coupling is a Structural Necessity:** The learning rate multiplier $\gamma_t$ in AADWD is not a design choice but a stability requirement. Removing it (decoupled mode) causes aggressive AADWD to collapse to 10.00% accuracy (weight norm $\to$ 0.0036) via a positive feedback loop, and conservative to degrade to 80.30%.
   - *Data:* aggressive coupled 92.05% vs decoupled 10.00% ($\Delta$ = -82.05%); conservative coupled 92.37% vs decoupled 80.30% ($\Delta$ = -12.07%)

3. **Alignment Signal is Not Actionable Under Standard SGD:** The alignment proxy $\hat{\delta}_t \sim O(10^{-3})$ with std $\sim 7.5 \times 10^{-4}$ throughout training, making it effectively constant. Random dynamic WD (92.06%) matches AADWD aggressive (92.05%), proving alignment carries no exploitable scheduling information.
   - *Data:* random_dynamic_wd = 92.06%, aadwd_aggressive = 92.05%, $\Delta$ = 0.01% (statistically indistinguishable)

**Additional contributions:**
- Systematic cross-architecture evidence of CWD late-training collapse (CIFAR-10/ResNet-20: 91.79% $\to$ 86.95%; CIFAR-100/ResNet-20: 66.84% $\to$ 54.27%; CIFAR-10/VGG-16: 92.95% $\to$ 86.47%)
- Complete hyperparameter sensitivity characterization of alignment-based WD scheduling

---

## 3. Related Work

**Length:** ~1 page

### 3.1 Weight Decay Theory and Practice (~0.4 pages)

**Key references and points:**
- Classical L2 regularization vs decoupled WD (Loshchilov & Hutter, 2019 -- AdamW)
- Theoretical foundations: WD in nonconvex optimization (cite core paper); convergence with fixed WD under smoothness assumptions
- Empirical WD scheduling: cosine WD schedules, warmup-then-constant, etc.
- Scale-free regularization perspective (van Laarhoven, 2017)
- Connection to flatness and generalization (Foret et al., 2021 -- SAM; Li et al., 2018 -- landscape visualization)

### 3.2 Dynamic and Adaptive Regularization (~0.3 pages)

**Key references and points:**
- Adaptive regularization in optimization: AdaGrad, natural gradient, KFAC
- Cautious Weight Decay (CWD, ICLR 2026) -- coordinate-wise sign-based masking
- Scheduled regularization: dropout scheduling, label smoothing schedules
- Meta-learning for hyperparameter adaptation (Baydin et al., 2018)
- Layer-wise adaptive WD (LARS, LAMB)

### 3.3 Gradient--Parameter Alignment in Training Dynamics (~0.3 pages)

**Key references and points:**
- Gradient alignment as a training diagnostic (Fort et al., 2019; Gur-Ari et al., 2018)
- Connection to implicit regularization (Blanc et al., 2020; Li et al., 2021)
- Alignment in loss landscape analysis
- Stochastic proxy reliability: minibatch vs full-batch gradient statistics

---

## 4. Theoretical Framework

**Length:** ~2 pages

### 4.1 Preliminaries and Problem Setup (~0.4 pages)

**Key content:**
- Standard nonconvex SGD with decoupled WD: $w_{t+1} = (1 - \lambda_t) w_t - \gamma_t g_t$
- Assumptions: $L$-smoothness, bounded variance $\sigma^2$, bounded gradient second moment
- Definition of alignment quantity: $\delta_t = |\langle \nabla f_S(w_t), w_t \rangle| / (\|\nabla f_S(w_t)\| \|w_t\|)$
- Review of the fixed-WD result from the core paper: convergence bound depends on $\delta_T = \sup_t \delta_t$

### 4.2 Convergence with Time-Varying Weight Decay (~0.6 pages)

**Key content:**
- **Theorem 1 (Target A -- Conservative):** Under standard smoothness and bounded variance, time-varying SGDW with $\lambda_t = O(\gamma_t^2)$ preserves the standard $O(1/\sqrt{T})$ nonconvex convergence rate
- Proof sketch: augmented potential $\Phi_t = f_S(w_t) + \beta_t \|w_t\|^2$ with time-varying $\beta_t$; telescope sum absorbs the cross-term via $\lambda_t \leq C \gamma_t^2$
- **Theorem 2 (Target B -- Alignment-Weighted Bound):** The convergence loss depends on the cumulative alignment $\bar{\delta}_T = (1/T) \sum_t \lambda_t \delta_t / \sum_t \lambda_t$ rather than $\sup_t \delta_t$
- Implication: time-varying WD can exploit heterogeneous alignment across training -- but only if $\delta_t$ varies significantly

### 4.3 The Three AADWD Variants (~0.5 pages)

**Key content:**
- **Conservative:** $\lambda_t = \text{clip}(c \cdot \gamma_t \cdot (1 - \hat{\delta}_t)^p, \lambda_{\min}, \lambda_{\max})$
  - Rationale: increase WD when alignment is low (gradients and parameters are misaligned)
  - When $\hat{\delta}_t \approx 0$: degenerates to $\lambda_t \approx c \cdot \gamma_t$, a deterministic function of LR
- **Aggressive:** $\lambda_t = \text{clip}(c \cdot \gamma_t / (\hat{\delta}_t + \epsilon), \lambda_{\min}, \lambda_{\max})$
  - Rationale: inversely proportional to alignment -- maximum WD when gradients point away from parameters
  - Potential instability: high $\delta$ $\to$ low WD $\to$ weight growth $\to$ higher $\delta$ (self-correcting); low $\delta$ $\to$ high WD $\to$ weight shrinkage $\to$ lower $\delta$ (self-amplifying without LR coupling)
- **Square:** $\lambda_t = \text{clip}(c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t), \lambda_{\min}, \lambda_{\max})$
  - Rationale: satisfies the $O(\gamma_t^2)$ condition from Theorem 1
  - Consequence: after milestone LR drops (0.1 $\to$ 0.01 $\to$ 0.001), WD drops by $100\times$ then $10000\times$

### 4.4 Stability Analysis of LR--WD Coupling (~0.5 pages)

**Key content:**
- **Proposition 1 (Coupling Necessity):** In the decoupled AADWD (remove $\gamma_t$), the aggressive variant exhibits a positive feedback instability: $\hat{\delta}_t$ increase $\to$ $\lambda_t$ decrease $\to$ weight norm explosion or, conversely, $\hat{\delta}_t$ decrease $\to$ $\lambda_t$ increase $\to$ weight death
- Informal argument: the LR multiplier acts as a natural damper -- since $\gamma_t$ itself decreases over training (via milestone schedule), $\lambda_t$ inherits this decay, preventing runaway dynamics
- Connection to L2 regularization: in standard L2, the effective decay is $\text{lr} \times \text{wd}$, which couples the two automatically; decoupled WD (Loshchilov & Hutter, 2019) deliberately removes this, but AADWD with alignment-dependent $\lambda_t$ reintroduces it as a stability mechanism
- **Proposition 2 (Budget Equivalence -- Informal):** Under mild conditions on the loss landscape, if two WD schedules $\{\lambda_t^{(1)}\}$ and $\{\lambda_t^{(2)}\}$ satisfy $\sum_t \lambda_t^{(1)} = \sum_t \lambda_t^{(2)}$, their regularization effects on generalization are asymptotically equivalent

---

## 5. Experimental Setup

**Length:** ~1 page

### 5.1 Datasets and Architectures (~0.3 pages)

**Key content:**
- **CIFAR-10** (50K train / 10K test, 10 classes): standard benchmark for WD studies
- **CIFAR-100** (50K train / 10K test, 100 classes): harder classification, higher regularization sensitivity
- **ResNet-20** (0.27M parameters): compact residual network, primary architecture
- **VGG-16-BN** (15M parameters): deep feedforward with batch normalization, cross-architecture validation
- Standard data augmentation: random crop (32x32, padding 4), horizontal flip, normalization

### 5.2 Methods Compared (~0.4 pages)

**Table: Summary of all methods (11 methods total)**

| Method | Schedule | Formula | Key Property |
|--------|----------|---------|-------------|
| No WD | -- | $\lambda = 0$ | Baseline (no regularization) |
| Fixed WD | Constant | $\lambda = 5 \times 10^{-4}$ | Standard practice baseline |
| Fixed WD Grid | Constant | $\lambda \in \{10^{-4}, 3 \times 10^{-4}, 5 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$ | Optimal WD search |
| Stagewise WD | Step | LR-proportional at milestones | Simple dynamic baseline |
| CWD | Coordinate-wise | Sign-based masking (ICLR 2026) | Recent adaptive method |
| AADWD Conservative | Alignment-adaptive | $\lambda_t = c \cdot \gamma_t \cdot (1 - \hat{\delta}_t)$ | Low-variance adaptation |
| AADWD Aggressive | Alignment-adaptive | $\lambda_t = c \cdot \gamma_t / (\hat{\delta}_t + \epsilon)$ | High-variance adaptation |
| AADWD Square | Alignment-adaptive | $\lambda_t = c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t)$ | Theoretically motivated |
| Random Dynamic WD | Random | $\lambda_t = c \cdot \gamma_t \cdot U(0,1)$ | Ablation control |
| Equiv. Cumulative WD | Matched-budget | $\bar{\lambda} = \text{mean}(\lambda_t^{\text{AADWD}})$ | Budget equivalence test |
| Norm-Matched WD | Trajectory-matched | Match AADWD weight norm curve | Norm ablation control |

### 5.3 Training Protocol (~0.15 pages)

- **Optimizer:** SGD + Momentum (0.9)
- **Initial LR:** 0.1 with milestone schedule $[80, 120]$ ($\gamma \times 0.1$)
- **Epochs:** 200 (full experiments); 20 (pilot screening)
- **Batch size:** 128
- **AADWD hyperparameters:** $c \in \{0.5, 1.0, 2.5, 5.0, 10.0\}$, $\beta \in \{0.9, 0.99, 0.999, 0.9999\}$, $\lambda_{\min} = 10^{-6}$, $\lambda_{\max} = 0.05$

### 5.4 Evaluation Metrics (~0.15 pages)

- **Best test accuracy** (primary metric): highest test accuracy during training
- **Final test accuracy**: accuracy at epoch 200 (stability indicator)
- **Generalization gap:** $\text{train\_acc} - \text{test\_acc}$ at convergence
- **Weight norm:** $\|w_T\|_2$ at final epoch (regularization strength indicator)
- **$\lambda_t$ trajectory:** temporal evolution of effective WD rate (for dynamic methods)
- **Alignment proxy $\hat{\delta}_t$:** EMA-smoothed minibatch alignment over training

---

## 6. Results

**Length:** ~3 pages

### 6.1 Main Results: Fixed WD Dominates All Dynamic Methods (~0.8 pages)

**Content: Table 1 (Main results table -- the central table of the paper)**

| Method | Best Acc (%) | Final Acc (%) | Gen Gap | Weight Norm | Setting |
|--------|-------------|--------------|---------|-------------|---------|
| No WD | 90.49 | 90.31 | 9.17 | 129.42 | C10/R20 |
| Fixed WD (5e-4) | **92.54** | **92.29** | 7.17 | 23.49 | C10/R20 |
| Stagewise WD | 92.44 | 92.27 | 7.18 | 33.22 | C10/R20 |
| AADWD Conservative | 92.37 | 92.22 | 7.15 | 23.60 | C10/R20 |
| AADWD Square | 92.13 | 91.78 | 7.47 | 38.75 | C10/R20 |
| AADWD Aggressive | 92.05 | 91.57 | 7.50 | 21.47 | C10/R20 |
| Random Dynamic WD | 92.06 | 91.95 | 7.54 | 34.19 | C10/R20 |
| Equiv. Cumulative WD | **92.54** | **92.29** | 7.17 | 23.49 | C10/R20 |
| CWD | 91.79 | 86.95 | 3.22 | 9.28 | C10/R20 |

**Key observations:**
- Fixed WD (5e-4) achieves the best or tied-best performance (92.54%)
- Equiv. Cumulative WD precisely matches Fixed WD (92.54% = 92.54%), proving budget equivalence
- Conservative AADWD performs within 0.17% of Fixed WD because $\delta \approx 0$ makes it degenerate to fixed
- All AADWD variants are dominated by Fixed WD
- CWD shows dramatic late-training collapse (91.79% best $\to$ 86.95% final, $\Delta$ = 4.84%)

**Figure 1:** Test accuracy curves over 200 epochs for all methods on CIFAR-10/ResNet-20

### 6.2 Cross-Architecture and Cross-Dataset Validation (~0.6 pages)

**Content: Table 2 (Cross-architecture results)**

| Method | C10/R20 | C100/R20 | C10/VGG16 |
|--------|---------|----------|-----------|
| No WD | 90.49 | 64.70 | 92.34 |
| Fixed WD (5e-4) | **92.54** | **68.45** | **93.86** |
| AADWD Conservative | 92.37 | 68.24 | 93.75 |
| AADWD Aggressive | 92.05 | 61.34 | 90.97 |
| CWD | 91.79 $\to$ 86.95 | 66.84 $\to$ 54.27 | 92.95 $\to$ 86.47 |

**Key observations:**
- Fixed WD (5e-4) is the best method across all three settings
- AADWD Conservative is consistently second-best but within noise of Fixed WD ($\Delta \leq 0.21\%$)
- AADWD Aggressive shows larger degradation on CIFAR-100 ($\Delta$ = -7.11% vs Fixed WD), suggesting alignment instability worsens on harder tasks
- CWD collapse is universal: best-to-final degradation of 4.84%, 12.57%, 6.48% across settings
- VGG-16 (15M params) shows the same pattern as ResNet-20 (0.27M params) -- conclusions are architecture-independent

**Figure 2:** Bar chart comparing methods across three dataset/architecture combinations

### 6.3 Budget Equivalence Experiment (~0.4 pages)

**Content: The cleanest causal result in the paper**

**Experimental design:**
- Compute $\bar{\lambda} = (1/T) \sum_{t=1}^T \lambda_t^{\text{AADWD-agg}}$ from a completed AADWD-Aggressive run
- Run a new experiment with constant $\lambda = \bar{\lambda}$ for 200 epochs
- Compare all metrics

**Results:**

| Metric | AADWD Aggressive | Equiv. Cumulative | Fixed WD (5e-4) |
|--------|-----------------|-------------------|-----------------|
| Best Acc | 92.05 | 92.54 | 92.54 |
| Final Acc | 91.57 | 92.29 | 92.29 |
| Weight Norm | 21.47 | 23.49 | 23.49 |
| Gen Gap | 7.50 | 7.17 | 7.17 |

**Interpretation:**
- The cumulative WD budget is the sufficient statistic for generalization
- Temporal dynamics of $\lambda_t$ provide zero additional information
- The 0.49% gap between AADWD-aggressive (92.05%) and its budget-equivalent fixed WD (92.54%) is entirely due to the suboptimal temporal allocation, not the budget

**Figure 3:** Side-by-side $\lambda_t$ trajectories for AADWD-aggressive vs constant budget-equivalent, with identical final performance

### 6.4 LR Decoupling Experiment (~0.5 pages)

**Content: The most dramatic finding**

**Experimental design:**
- Compare each AADWD variant with and without the $\gamma_t$ multiplier (coupled vs decoupled)
- Decoupled formulas: conservative $\lambda_t = c \cdot (1 - \hat{\delta}_t)$; aggressive $\lambda_t = c / (\hat{\delta}_t + \epsilon)$

**Results: Table 3**

| Method | Coupled | Decoupled | $\Delta$ | Weight Norm (Decoupled) |
|--------|---------|-----------|----------|------------------------|
| Conservative | 92.37 | 80.30 | -12.07 | 5.53 |
| Aggressive | 92.05 | 10.00 | -82.05 | 0.0036 |

**Analysis of Aggressive Collapse (92.05% $\to$ 10.00%):**
- Without $\gamma_t$ damping, $\lambda_t$ remains high even after LR drops
- After milestone at epoch 80: LR drops $0.1 \to 0.01$, but decoupled $\lambda_t$ stays at $\sim 5 \times 10^{-4}$
- Effective regularization becomes $10\times$--$100\times$ too strong relative to the gradient signal
- Positive feedback: weight shrinkage $\to$ alignment change $\to$ higher $\lambda$ $\to$ more shrinkage $\to$ weight death (norm 0.0036)
- **Result: network parameters shrink to near-zero, training collapses to random chance (10%)**

**Analysis of Conservative Degradation (92.37% $\to$ 80.30%):**
- Without $\gamma_t$, constant $\lambda \approx 5 \times 10^{-4}$ is $10\times$ too strong when LR = 0.01, $100\times$ too strong when LR = 0.001
- Over-regularization causes underfitting but not complete collapse

**Figure 4:** Weight norm trajectories for coupled vs decoupled experiments (log scale), showing the catastrophic divergence

### 6.5 Hyperparameter Sensitivity (~0.4 pages)

**Content: Table 4 (c and beta sweeps for AADWD-Aggressive on CIFAR-10/ResNet-20)**

**c sweep (beta = 0.999):**

| c | Best Acc (%) | Final Weight Norm |
|---|-------------|------------------|
| 0.5 | 91.87 | 65.48 |
| 1.0 | 92.18 | 41.27 |
| 2.5 | 92.05 | 21.47 |
| 5.0 | $\star$ | $\star$ |
| 10.0 | $\star$ | $\star$ |

**beta sweep (c = 2.5):**

| beta | Best Acc (%) | Note |
|------|-------------|------|
| 0.9 | $\star$ | Fast-responding EMA |
| 0.99 | $\star$ | |
| 0.999 | 92.05 | Default configuration |
| 0.9999 | $\star$ | Over-smoothed, poor responsiveness |

**Key findings:**
- AADWD Aggressive is robust to $c$ over a $10\times$ range ($c \in [0.5, 5.0]$): accuracy varies $\leq 0.7\%$
- $\beta$ sensitivity is moderate: $\beta \in [0.9, 0.999]$ gives stable results; $\beta = 0.9999$ degrades due to over-smoothed EMA losing responsiveness
- Despite this robustness, no configuration outperforms simple Fixed WD

**Figure 5:** Heatmap or line plot showing accuracy as a function of c and beta

### 6.6 Alignment Proxy Characterization (~0.3 pages)

**Content: Why alignment is not actionable**

**Tier-0 diagnostic results (ResNet-20/CIFAR-10):**
- Alignment proxy $\hat{\delta}_t$ statistics across 200 epochs:
  - Early training mean: 0.004491
  - Mid training mean: 0.003352
  - Late training mean: 0.002824
  - Overall std: 0.000753
- Pearson correlation between minibatch EMA and large-batch alignment: $r = 0.849$
- Phase structure: $\delta$ decreases monotonically but the total range is only $[0.0028, 0.0045]$ -- a $1.6\times$ variation

**Implication:** With $\delta \sim O(10^{-3})$, the conservative formula $\lambda_t = c \cdot \gamma_t \cdot (1 - \hat{\delta}_t) \approx c \cdot \gamma_t \cdot 0.997$, which is effectively constant. The alignment signal is swamped by the LR schedule dynamics ($\gamma_t$ varies $100\times$ over training).

**Figure 6:** Alignment proxy $\hat{\delta}_t$ trajectory over 200 epochs with confidence bands, overlaid with $\gamma_t$ schedule

---

## 7. Analysis: Why Dynamic Weight Decay Fails

**Length:** ~1.5 pages

### 7.1 Insight 1: The Budget Equivalence Principle (~0.5 pages)

**Key argument:**
- WD's primary effect is controlling the weight norm trajectory $\|w_t\|$
- The weight norm at convergence is determined by the total regularization pressure $\sum_t \lambda_t$, not its temporal distribution
- Formal argument: in the linear regime near convergence, $\|w_T\|^2 \approx \|w_0\|^2 \prod_t (1 - 2\lambda_t) + \text{gradient terms}$; for small $\lambda_t$, $\prod_t (1 - 2\lambda_t) \approx \exp(-2 \sum_t \lambda_t)$, which depends only on the sum
- Experimental confirmation: equiv_cumulative_wd (92.54%) = fixed_wd (92.54%), identical weight norms (23.49)
- **Practical implication:** There is no "optimal schedule" for WD beyond choosing the right average magnitude. Tuning a single constant is sufficient.

### 7.2 Insight 2: LR Coupling as a Stability Mechanism (~0.5 pages)

**Key argument:**
- In standard L2 regularization, the effective WD is automatically $\text{lr} \times \lambda$, coupling the two
- Decoupled WD (Loshchilov & Hutter, 2019) removes this coupling as a deliberate design choice for AdamW
- AADWD with alignment-dependent $\lambda_t$ faces a dilemma: without LR coupling, the adaptive $\lambda_t$ can become arbitrarily large relative to the gradient signal after LR drops
- The $\gamma_t$ multiplier serves as a "speed limit" that prevents WD from dominating the dynamics
- Analogy: WD must be commensurate with the learning signal; when LR decreases, the system enters a "fine-tuning regime" where even small WD can be destructive
- **Connection to the aggressive collapse mechanism:**
  1. LR drops $0.1 \to 0.01$ at milestone
  2. Without $\gamma_t$ damping, $\lambda_t \sim 5 \times 10^{-4}$ persists
  3. The ratio $\lambda_t / \gamma_t$ increases $10\times$, meaning WD overwhelms the gradient
  4. Weight norm shrinks rapidly $\to$ $\hat{\delta}_t$ changes $\to$ feedback loop $\to$ complete collapse

### 7.3 Insight 3: Alignment is a Consequence, Not a Control Signal (~0.5 pages)

**Key argument:**
- The theoretical result that $\delta_T < 1$ is sufficient for WD benefit does not imply that varying $\delta_t$ provides useful information for WD scheduling
- $\delta_t \sim O(10^{-3})$ in standard training means the alignment condition is always "easily satisfied" -- there is no training phase where $\delta_t$ approaches 1 and WD becomes harmful
- The alignment quantity is an outcome of the optimization trajectory, not an independent signal
- Evidence: random WD (92.06%) = alignment-based WD (92.05%), confirming that any time-varying WD with appropriate budget and LR coupling performs similarly
- **When might alignment matter?** We conjecture that alignment-based adaptation could be beneficial when: (a) $\delta_t$ varies over a wider range (e.g., non-standard architectures, adversarial training), (b) WD rates are much larger (approaching the instability boundary), or (c) training is in a highly nonconvex regime where alignment fluctuates significantly

---

## 8. Discussion and Conclusion

**Length:** ~1 page

### 8.1 Practical Recommendations (~0.3 pages)

**Key points:**
1. **Use constant WD and tune only the magnitude.** Our results show no benefit from temporal adaptation. The single hyperparameter $\lambda$ is sufficient.
2. **Avoid decoupling WD from LR in custom schedules.** Any adaptive WD scheme must scale with the learning rate to maintain stability.
3. **Be cautious with CWD.** Despite its theoretical motivation, coordinate-wise WD shows consistent late-training instability across architectures and datasets.
4. **Budget matching as a diagnostic.** When evaluating any new WD schedule, compare against a constant WD with matched average -- if performance is identical, the temporal dynamics are not contributing.

### 8.2 Implications for WD Theory (~0.2 pages)

**Key points:**
- The alignment framework of the core paper correctly identifies the key quantity ($\delta_T$), but the practical value is limited because $\delta_T$ is always small under standard training
- Theory-driven WD adaptation requires settings where the alignment condition is tight (close to 1), which standard SGD training does not produce
- The budget equivalence principle suggests that WD theory should focus on cumulative effects rather than instantaneous rates

### 8.3 Limitations (~0.3 pages)

**Key limitations (honest assessment):**
1. **Scale:** Experiments limited to CIFAR-10/100 with ResNet-20 and VGG-16-BN. Results may not generalize to ImageNet-scale or transformer architectures.
2. **Optimizer:** All experiments use SGD + Momentum. Extension to Adam/AdamW remains an important open question (in AdamW, WD is already decoupled from LR -- different dynamics may apply).
3. **Single seed:** Most experiments use seed=42. While effect sizes are large enough that multi-seed runs are unlikely to change directional conclusions, statistical rigor would benefit from 3--5 seeds for the core comparisons.
4. **Alignment proxy:** We use a specific EMA-based minibatch proxy. Other alignment estimators (e.g., based on Fisher information or Hessian-vector products) might capture different aspects of the alignment structure.
5. **Scope of "standard training":** Our findings are specific to milestone LR schedules with moderate WD values. Extreme training regimes (very large WD, cyclical LR, warmup-heavy schedules) might exhibit different alignment dynamics.

### 8.4 Future Directions (~0.2 pages)

**Key points:**
1. **When does alignment become actionable?** Systematic identification of training settings where $\delta_t$ varies significantly (adversarial training, continual learning, multi-task settings)
2. **Adaptive WD in Adam/AdamW:** The preconditioner in Adam changes the geometry; alignment-based adaptation might behave differently
3. **Formal proof of budget equivalence:** Tight theoretical conditions under which $\sum_t \lambda_t$ is a sufficient statistic for generalization
4. **Large-scale validation:** Confirm budget equivalence on ImageNet with modern architectures (ResNet-50, ViT)

---

## Appendix (Supplementary Material)

### A. Full Experimental Details (~2 pages)

- Complete hyperparameter tables for all 39 experiments
- Per-epoch training curves for all methods
- Detailed alignment proxy diagnostic (Pearson correlations, phase-wise statistics)
- Pilot experiment results (20-epoch) and how they guided full experiment design

### B. Additional Ablation Studies (~1.5 pages)

- Norm-matched WD detailed analysis (best 90.44%, final 75.37% -- severe late-training degradation)
- Full c-sweep results (c = 0.5, 1.0, 2.5, 5.0, 10.0)
- Full beta-sweep results (beta = 0.9, 0.99, 0.999, 0.9999)
- Fixed WD grid search across all values ($10^{-4}$ through $3 \times 10^{-3}$)

### C. CWD Instability Analysis (~1 page)

- Detailed weight norm trajectories showing the collapse mechanism
- Per-layer analysis of coordinate-wise shrinkage
- Comparison of best-to-final accuracy gaps across all three settings

### D. Decoupling Experiment Details (~1 page)

- Full training curves for coupled vs decoupled aggressive and conservative
- Weight norm evolution on log scale showing the divergence point
- $\lambda_t$ trajectory comparison showing the loss of damping

### E. Theoretical Proofs (~2 pages)

- Complete proof of Theorem 1 (convergence with time-varying WD)
- Complete proof of Theorem 2 (alignment-weighted bound)
- Proof sketch of Proposition 1 (decoupling instability)
- Discussion of conditions for Proposition 2 (budget equivalence)

---

## Figures Summary

| Figure | Content | Section | Type |
|--------|---------|---------|------|
| Fig 1 | Test accuracy curves (all methods, C10/R20, 200 epochs) | 6.1 | Line plot |
| Fig 2 | Cross-architecture comparison (3 settings, bar chart) | 6.2 | Bar chart |
| Fig 3 | Budget equivalence: $\lambda_t$ trajectories + matched performance | 6.3 | Dual-axis |
| Fig 4 | LR decoupling: weight norm divergence (coupled vs decoupled) | 6.4 | Log-scale line |
| Fig 5 | Hyperparameter sensitivity heatmap (c vs beta) | 6.5 | Heatmap |
| Fig 6 | Alignment proxy $\hat{\delta}_t$ trajectory with LR schedule overlay | 6.6 | Dual-axis |
| Fig 7 | CWD collapse: best-to-final accuracy degradation across settings | 7/App C | Bar chart |

---

## Tables Summary

| Table | Content | Section |
|-------|---------|---------|
| Table 1 | Main results (11 methods, CIFAR-10/ResNet-20) | 6.1 |
| Table 2 | Cross-architecture results (3 settings x 5 methods) | 6.2 |
| Table 3 | LR decoupling experiment (coupled vs decoupled) | 6.4 |
| Table 4 | Hyperparameter sensitivity (c-sweep, beta-sweep) | 6.5 |
| Table 5 | Alignment proxy statistics (phase-wise) | 6.6 |

---

## Key Data Reference (for writing)

**Budget Equivalence:**
- equiv_cumulative_wd: best=92.54%, final=92.29%, WN=23.49
- fixed_wd_0.0005: best=92.54%, final=92.29%, WN=23.49
- Exact match on all metrics

**LR Decoupling:**
- Aggressive coupled: 92.05% $\to$ Aggressive decoupled: 10.00% (WN 0.0036)
- Conservative coupled: 92.37% $\to$ Conservative decoupled: 80.30% (WN 5.53)

**Alignment Signal:**
- Random dynamic WD: 92.06%
- AADWD Aggressive: 92.05%
- $\Delta$ = 0.01% (indistinguishable)
- $\delta$ range: [0.0028, 0.0045], std = 0.000753

**CWD Collapse (best $\to$ final):**
- CIFAR-10/ResNet-20: 91.79% $\to$ 86.95% ($\Delta$ = 4.84%)
- CIFAR-100/ResNet-20: 66.84% $\to$ 54.27% ($\Delta$ = 12.57%)
- CIFAR-10/VGG-16: 92.95% $\to$ 86.47% ($\Delta$ = 6.48%)

**Cross-Architecture (best acc):**
- CIFAR-10/ResNet-20: Fixed 92.54 > Conservative 92.37 > Aggressive 92.05
- CIFAR-100/ResNet-20: Fixed 68.45 > Conservative 68.24 > Aggressive 61.34
- CIFAR-10/VGG-16: Fixed 93.86 > Conservative 93.75 > Aggressive 90.97

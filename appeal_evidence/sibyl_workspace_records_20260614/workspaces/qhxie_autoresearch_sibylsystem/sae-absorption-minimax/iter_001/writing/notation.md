# Notation Table

## Model Parameters

| Symbol | Definition | Dimensionality |
|--------|------------|----------------|
| $x \in \mathbb{R}^d$ | Input token representation (residual stream) | $d$ = model dimension |
| $d_{model}$ | Model hidden dimension | scalar |
| $L$ | Number of transformer layers | scalar |
| $\ell \in \{1, ..., L\}$ | Layer index | scalar |

## SAE Architecture

| Symbol | Definition | Dimensionality |
|--------|------------|----------------|
| $W_{enc} \in \mathbb{R}^{d_{model} \times d_{SAE}}$ | SAE encoder weight matrix | $d_{model}$ × $d_{SAE}$ |
| $W_{dec} \in \mathbb{R}^{d_{SAE} \times d_{model}}$ | SAE decoder weight matrix | $d_{SAE}$ × $d_{model}$ |
| $b_{enc} \in \mathbb{R}^{d_{SAE}}$ | SAE encoder bias | $d_{SAE}$ |
| $b_{dec} \in \mathbb{R}^{d_{model}}$ | SAE decoder bias | $d_{model}$ |
| $d_{SAE}$ | SAE dictionary size (number of features) | scalar |
| $f \in \{1, ..., d_{SAE}\}$ | Feature index | scalar |
| $\hat{x} \in \mathbb{R}^{d_{model}}$ | SAE reconstruction | $d_{model}$ |

## SAE Forward Pass

| Symbol | Definition |
|--------|------------|
| $pre_f = W_{enc}[f] \cdot x + b_{enc}[f]$ | Pre-activation for feature $f$ |
| $act_f = ReLU(pre_f)$ | Post-activation for feature $f$ (standard SAE) |
| $act_f = max(0, pre_f - \tau)$ | Post-activation for feature $f$ (JumpReLU), $\tau$ = threshold |
| $\hat{x} = W_{dec} \cdot act + b_{dec}$ | Reconstruction from active features |
| $z_f \in \{0,1\}$ | Binary activation indicator (TopK) |
| $k$ | TopK sparsity parameter (number of active features) |

## Sparsity and Loss

| Symbol | Definition |
|--------|------------|
| $L_0 = \sum_f \mathbb{1}[act_f > 0]$ | Number of active features |
| $\mathcal{L}_{recon} = \|x - \hat{x}\|^2$ | Reconstruction loss (MSE) |
| $\mathcal{L}_{L1} = \lambda \sum_f \|act_f\|_1$ | L1 sparsity penalty |
| $\mathcal{L}_{SAE} = \mathcal{L}_{recon} + \mathcal{L}_{L1}$ | Total SAE training loss |
| $\lambda$ | L1 regularization coefficient |
| $\mathcal{L}_{ortho} = \gamma \sum_{i \neq j} (W_{enc}[i] \cdot W_{enc}[j])^2$ | OrtSAE orthogonal penalty |

## Absorption Metrics

| Symbol | Definition | Range |
|--------|------------|-------|
| $A(f)$ | Chanin absorption score for feature $f$ | $[0, 1]$ |
| $UAS(f)$ | Unsupervised Absorption Score for feature $f$ | $[0, \infty)$ |
| $cos\_sim(f, g) = \frac{W_{dec}[f] \cdot W_{dec}[g]}{\|W_{dec}[f]\| \|W_{dec}[g]\|}$ | Cosine similarity between decoder directions | $[-1, 1]$ |
| $cos\_var(f) = Var_{g \neq f}(cos\_sim(f, g))$ | Variance of cosine similarities for feature $f$ | $[0, 4]$ |
| $freq\_skew(f)$ | Skewness of activation frequency distribution for feature $f$ | $(-\infty, \infty)$ |
| $\alpha$ | Weight for cosine similarity variance in UAS | scalar |
| $\beta$ | Weight for frequency skewness in UAS | scalar |
| $N_{top}$ | Number of top-activated tokens for absorption evaluation | scalar |

## Steering Intervention

| Symbol | Definition | Dimensionality |
|--------|------------|----------------|
| $\alpha_{steer}$ | Steering coefficient (intervention strength) | scalar |
| $\delta_f = \alpha_{steer} \cdot W_{dec}[f]$ | Steering direction for feature $f$ | $d_{model}$ |
| $x_{steered} = x + \delta_f$ | Steered residual stream | $d_{model}$ |
| $p_{orig}(token)$ | Original model probability for target token | scalar |
| $p_{steered}(token)$ | Steered model probability for target token | scalar |
| $\Delta_{logit} = \log p_{steered} - \log p_{orig}$ | Logit change from steering | scalar |
| $effect_f$ | Mean steering effect across prompts for feature $f$ | scalar |

## Statistical Measures

| Symbol | Definition |
|--------|------------|
| $\rho$ or $r$ | Spearman rank correlation coefficient |
| $p$ | P-value for statistical significance |
| $\mu$ | Mean |
| $\sigma$ | Standard deviation |
| $CI_{95}$ | 95% confidence interval |
| $N$ | Sample size (number of features or tokens) |

## Evaluation Metrics

| Symbol | Definition |
|--------|------------|
| $MSE$ | Mean squared error (reconstruction quality) |
| $AUC$ | Area under ROC curve (discriminability) |
| $Accuracy$ | Classification accuracy |
| $CE$ | Cross-entropy loss |

## Experiment Variables

| Symbol | Definition |
|--------|------------|
| $S_{high}$ | Set of high-absorption features (UAS > threshold) |
| $S_{low}$ | Set of low-absorption features (UAS < threshold) |
| $UAS_{threshold}^{high}$ | High-absorption threshold (typically 1.0) |
| $UAS_{threshold}^{low}$ | Low-absorption threshold (typically 0.3) |
| $N_{high}$ | Number of high-absorption features in experiment |
| $N_{low}$ | Number of low-absorption features in experiment |
| $N_{prompts}$ | Number of test prompts for steering evaluation |

## Hypothesis Status

| Hypothesis | Description | Status |
|------------|-------------|--------|
| H1 | Absorption peaks in middle layers | UNRESOLVED |
| H2 | Mitigation reduces absorption at reconstruction cost | PARTIALLY CONFIRMED |
| H3 | High-absorption features show lower steering sensitivity | REVERSED |
| H4 | UAS correlates with supervised absorption (r > 0.6) | CONFIRMED |
| H5 | Absorption degrades downstream discriminability | DIRECTIONAL |

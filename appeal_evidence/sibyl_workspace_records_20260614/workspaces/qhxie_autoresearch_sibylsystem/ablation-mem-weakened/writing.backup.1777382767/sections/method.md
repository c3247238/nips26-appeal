# 4. Methodology

## 4.1 Overview

Our approach is entirely training-free. We analyze pre-trained SAEs using SAELens, TransformerLens, and custom evaluation code. No SAE training is required, making the methodology accessible to any researcher with a GPU and open-source tools.

The experimental pipeline consists of four phases. Phase 1 detects feature absorption using the Chanin et al. differential correlation metric. Phase 2 measures steering effectiveness by adding feature directions to model activations, with a critical random feature baseline control. Phase 3 trains k-sparse linear probes for first-letter classification. Phase 4 correlates absorption rates with task performance to test our hypotheses.

![Four-phase experimental pipeline: (1) Absorption Detection loads pre-trained SAEs and runs the Chanin et al. metric; (2) Feature Steering extracts decoder directions and measures probability lift with random baseline subtraction; (3) Sparse Probing trains k-sparse classifiers on SAE latents; (4) Correlation Analysis tests H1--H3 with Pearson/Spearman correlation and linear regression.](figures/fig1_pipeline.pdf)

**Figure 1:** The four-phase experimental pipeline. Each phase feeds into the next, with absorption rates from Phase 1 paired with task performance from Phases 2--3 for correlation analysis in Phase 4. The random baseline control in Phase 2 (highlighted) is essential for isolating feature-specific steering effects.

## 4.2 Model and SAE Configuration

We use GPT-2 Small ($M$, 124M parameters, 85M non-embedding, 12 layers, $d = 768$ hidden dimensions) with the gpt2-small-res-jb SAE release (24,576 latents, $n_{\text{latent}} = 24{,}576$). GPT-2 Small provides a well-studied baseline with publicly available SAEs.

We test four layer indices: $\ell \in \{0, 4, 8, 10\}$, all at the hook_resid_pre hook point (residual stream activations before the attention and MLP sublayers). Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network where feature abstraction increases.

## 4.3 Phase 1: Absorption Detection

For each layer $\ell$, we load the corresponding pre-trained SAE via SAELens and run the Chanin et al. differential correlation metric on 26 first-letter features ($\mathcal{F} = \{A, B, \ldots, Z\}$). For each letter, we identify the parent latent $\phi(f)$ that maximally activates on tokens starting with that letter, then compute the absorption rate $A(f)$ as the fraction of child features showing differential correlation above a threshold.

The differential correlation $\rho(f, c)$ between parent $f$ and child $c$ is computed by correlating their activations on a dataset where the parent concept is present. A child $c$ is flagged as absorbing if $\rho(f, c)$ exceeds the Chanin et al. threshold, indicating that the child's activation reliably coincides with the parent's absence.

We classify features into three categories:
- HIGH: $A(f) \geq 0.10$
- MEDIUM: $0.05 \leq A(f) < 0.10$
- LOW: $A(f) < 0.05$

The threshold of 10% follows the SAEBench convention and reflects the empirical distribution: most features show near-zero absorption, and the 10% cutoff separates the tail of absorbed features from the bulk.

## 4.4 Phase 2: Feature Steering

For each first-letter feature $f$ identified in Phase 1, we extract its decoder direction $d_f = W_{\text{dec}}[\phi(f)] \in \mathbb{R}^d$. We generate $N = 100$ test prompts per feature, each containing a word starting with the target letter (e.g., "Apple", "Ant", "April" for feature A). Prompts are drawn from a curated vocabulary list with fixed random seed 42.

Steering is applied at layer $\ell$ by modifying the residual stream activation:
$$
a'_\ell = a_\ell + s \cdot d_f
$$
where $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$ is the steering strength. The model generates the next-token distribution under steered and unsteered conditions.

The raw steering success rate $S(f, s)$ is the fraction of prompts where the probability of a target-letter token increases relative to the unsteered baseline:
$$
S(f, s) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[ P_f(t_i) > P_0(t_i) \right]
$$
where $P_f(t_i)$ is the probability of the target token under steering and $P_0(t_i)$ is the baseline probability.

**Random baseline control.** We include a critical control condition: 26 randomly selected SAE latents (drawn uniformly from $\{0, \ldots, n_{\text{latent}}-1\}$ with seed 42) are steered with the identical protocol. The random baseline success rate $S_{\text{rand}}(s)$ measures the generic steering effect from arbitrary decoder directions. Feature-specific steering significantly exceeds random baseline at both layers (layer 4: $t = 6.41$, $p < 0.0001$, Cohen's $d = 1.26$; layer 8: $t = 6.02$, $p < 0.0001$, Cohen's $d = 1.18$), validating that feature-specific directions capture meaningful structure.

**Delta steering success.** Our primary steering metric subtracts the random baseline to isolate the feature-specific contribution:
$$
\Delta S(f, s) = S(f, s) - S_{\text{rand}}(s)
$$
This controls for the baseline steering effect that arbitrary directions produce and isolates the true feature-specific steering contribution. H1 tests raw steering; H1b tests delta steering.

## 4.5 Phase 3: Sparse Probing

We train k-sparse linear probes for first-letter classification using SAE latent activations as features. For each feature $f$, we collect SAE latent activations $z \in \mathbb{R}^{n_{\text{latent}}}$ from the $N = 100$ prompts and train a logistic regression classifier with L1 regularization, selecting the $k$ features with the largest absolute weights to enforce the sparsity constraint $\|w_k\|_0 \leq k$.

We test four sparsity levels: $k \in \{1, 5, 10, 20\}$. The $k = 5$ level is our primary analysis point: it is sparse enough to isolate individual feature contributions but rich enough to capture correlated latents that may compensate for an absorbed parent.

Probe performance is measured by F1 score:
$$
\text{F1}(f, k) = 2 \cdot \frac{\text{Prec}(f, k) \cdot \text{Rec}(f, k)}{\text{Prec}(f, k) + \text{Rec}(f, k)}
$$

We also compute $\text{F1}_{\text{full}}$ using all 24,576 latents with L2-regularized logistic regression (no sparsity constraint) to measure whether task-relevant information is recoverable from the full SAE representation even when individual features are absorbed.

## 4.6 Phase 4: Correlation Analysis

For each layer where both absorption and task data are available ($\ell \in \{4, 8\}$), we compute:

1. **Pearson correlation** $r$ between $A(f)$ and $S(f, 50)$ (H1), between $A(f)$ and $\Delta S(f, 50)$ (H1b), and between $A(f)$ and $\text{F1}(f, 5)$ (H2).
2. **Spearman rank correlation** $\rho$ as a non-parametric robustness check.
3. **Linear regression**: $\text{task\_performance} = \beta \cdot A(f) + \epsilon$, reporting slope $\beta$, $R^2$, and two-tailed p-value.

For H3 (cross-layer consistency), we compare the regression slopes $\beta$ across layers. The coefficient of variation $\text{CV} = \sigma / |\mu|$ quantifies consistency: $\text{CV} < 0.5$ indicates consistent degradation coefficients. The absolute value on the denominator is necessary because slopes can have opposite signs, making the raw mean near zero and the raw CV uninterpretable.

**Falsification criteria.** H1, H1b, and H2 are not supported if the Pearson correlation is non-negative ($r \geq 0$) or fails to reach significance ($p \geq 0.05$). A negative but non-significant trend does not support the hypothesis. H3 is not supported if slopes have opposite signs or differ substantially in magnitude ($\text{CV} \geq 0.5$).

All statistical tests use $n = 26$ features (the full first-letter set). This sample size provides approximately 65% power to detect a Pearson correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ (two-tailed). To achieve 80% power for detecting $|r| = 0.50$, approximately $n = 32$ features would be required. Correlations in the $-0.3$ to $+0.1$ range, as we observe for H1 and H2, fall below this detection threshold.

## 4.7 Software and Reproducibility

All experiments use Python 3.12 with the following stack: SAELens (SAE loading), TransformerLens (model hooks and activation caching), PyTorch (tensor operations), NumPy/SciPy (statistics), and Matplotlib (visualization). The random seed is fixed at 42 for all stochastic operations. All SAEs are from publicly available releases (gpt2-small-res-jb via SAELens). Code and evaluation protocol are released with the paper.

<!-- FIGURES
- Figure 1: gen_fig1_pipeline.py, fig1_pipeline.pdf — Flow diagram showing the four-phase pipeline with data flow arrows, key outputs, and random baseline control highlighted in Phase 2
- None (other visual elements belong to the Results section)
-->

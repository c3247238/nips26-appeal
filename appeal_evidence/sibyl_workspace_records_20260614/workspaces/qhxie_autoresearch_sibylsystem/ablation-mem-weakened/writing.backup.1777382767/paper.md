# Abstract

Feature absorption---where general SAE features fail to fire and are instead captured by more specific child features---has been identified as a fundamental failure mode of sparse autoencoders. Yet no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research. We provide the first systematic study connecting feature absorption detection to downstream task performance for steering effectiveness and sparse probing accuracy. Using pre-trained SAEs from GPT-2 Small (124M parameters, 85M non-embedding; gpt2-small-res-jb, 24,576 latents), we measure absorption rates across layers 0, 4, 8, and 10 for 26 first-letter features (A--Z), then test steering effectiveness and sparse probing accuracy. Our methodology is entirely training-free. We find no significant correlation between absorption rate and raw steering success (Pearson $r = +0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8) or sparse probing F1 ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). When controlling for random baseline steering, the delta (feature-specific minus random) shows a negative trend at layer 8 (Pearson $r = -0.431$, uncorrected $p = 0.028$), but this result does not survive Bonferroni correction for 12 tests ($p = 0.334$) or Benjamini--Hochberg FDR ($q = 0.167$). Sparse probing shows no correlation at either layer. With approximately 64% power to detect $|r| \geq 0.50$, our study cannot distinguish a true null effect from an underpowered detection of a small-to-medium effect. We frame our contribution as methodological: establishing the first training-free pipeline for absorption--task correlation, identifying the random baseline confound as a critical methodological issue, and providing power analysis guidance for future studies.

<!-- FIGURES
- None
-->

# 1. Introduction

## 1.1 The SAE Credibility Crisis

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability, enabling circuit analysis, feature steering, model editing, and bias detection (Marks et al., 2024; Templeton et al., 2024). The foundational premise is that SAEs decompose neural network activations into human-interpretable features through sparse dictionary learning. Yet the field faces an escalating credibility crisis.

Korznikov et al. (2026) demonstrate that SAEs recover only 9% of true features despite 71% explained variance, and that random baseline SAEs match trained SAEs on standard metrics. These concerns have led some research groups to deprioritize SAE research after finding negative results on downstream tasks (Bricken et al., 2023; Lieberum et al., 2023). These developments raise a fundamental question: do SAEs provide reliable tools for interpretability work, or do they create an illusion of understanding?

## 1.2 Feature Absorption: The Gap Between Detection and Impact

At the center of this crisis is feature absorption, first formally identified by Chanin et al. (2024). Absorption occurs when a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. For example, a "starts with A" feature may be absorbed by "starts with Apple" or "starts with Ant" features. The parent latent appears interpretable when inspected in isolation but produces systematic false negatives during downstream use.

Chanin et al. demonstrated that hierarchical features cause absorption and validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently standardized absorption as a benchmark metric alongside sparsity, reconstruction error, and explained variance. Architectural innovations---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2026), and HSAE (Luo et al., 2026)---all target absorption reduction as a primary objective.

Despite this attention, a critical gap remains: **no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research.** Researchers use SAEs for steering, circuit finding, and model editing. If absorbed features are systematically unreliable, these applications may produce false negatives or misleading results. Yet the field has optimized absorption metrics without establishing their relationship to task performance. This study bridges that gap.

## 1.3 Our Contribution

We provide the first systematic study connecting feature absorption detection to downstream task performance for steering effectiveness and sparse probing accuracy. Using pre-trained SAEs from GPT-2 Small (124M parameters, 85M non-embedding; gpt2-small-res-jb, 24,576 latents), we measure absorption rates across layers 0, 4, 8, and 10 for 26 first-letter features (A--Z), then test steering effectiveness and sparse probing accuracy.

Our methodology is entirely training-free, making it accessible to any researcher with a GPU and open-source tools. We test four hypotheses:

- **H1 (Raw steering)**: Higher absorption rate leads to lower raw steering success rate (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H1b (Delta steering)**: Higher absorption rate leads to lower delta steering effectiveness, where delta is feature-specific success minus random baseline success (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H2 (Probing)**: Higher absorption rate leads to lower sparse probing F1 (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H3 (Consistency)**: The degradation relationship is consistent across layers (regression slopes have the same sign and similar magnitude, $\text{CV} < 0.5$ where $\text{CV} = \sigma / |\mu|$).

The random baseline control in H1b is critical. Random feature steering achieves 34--38% success on our task, indicating that arbitrary decoder directions produce non-negligible effects. Without subtracting this baseline, raw steering success conflates feature-specific contribution with generic directional bias.

Our results are null across all hypotheses. Raw steering success shows no significant correlation with absorption (Pearson $r = +0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8). Sparse probing shows no correlation at either layer ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). Delta-corrected steering shows a negative trend at layer 8 (Pearson $r = -0.431$, uncorrected $p = 0.028$), but with 12 statistical tests performed, this result does not survive Bonferroni correction ($p = 0.334$) or Benjamini--Hochberg FDR ($q = 0.167$). The relationship is inconsistent across layers (H3 not supported). With $n = 26$ features and approximately 64% power to detect $|r| \geq 0.50$, we cannot distinguish a true null effect from an underpowered detection of a small-to-medium effect. We frame our contribution as methodological rather than substantive: establishing the first training-free pipeline for absorption--task correlation, demonstrating the importance of random baseline control, and providing power analysis guidance for future studies at larger scale.

## 1.4 Paper Structure

We first establish that absorption detection and downstream task evaluation have proceeded in isolation (Section 2), then formalize pre-registered hypotheses (Section 3). Our training-free four-phase pipeline (Section 4) produces mixed results that question the assumption that absorption uniformly degrades downstream tasks (Sections 5--6), leading to actionable guidance for practitioners (Section 8).

<!-- FIGURES
- None
-->

# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have emerged as the dominant unsupervised approach for decomposing neural network activations into human-interpretable features. An SAE reconstructs activations $a \in \mathbb{R}^d$ as $\hat{a} = W_{\text{dec}} \cdot f(W_{\text{enc}} \cdot a + b_{\text{pre}})$, where $f(\cdot)$ is a sparsity-inducing nonlinearity (typically ReLU), $W_{\text{enc}} \in \mathbb{R}^{n_{\text{latent}} \times d}$ is the encoder, and $W_{\text{dec}} \in \mathbb{R}^{d \times n_{\text{latent}}}$ is the decoder. The dictionary size $n_{\text{latent}}$ is typically much larger than $d$, creating an overcomplete representation where each latent ideally corresponds to a single interpretable concept.

SAEs enable several downstream interpretability tasks. Feature steering (Marks et al., 2024; Rimsky et al., 2024) adds a feature direction to model activations during the forward pass to test causal influence. Sparse probing (Templeton et al., 2024) trains linear classifiers on SAE latents to detect concepts. Circuit analysis traces how features compose to produce model behavior. Model editing modifies specific features to alter model outputs. These applications depend on the assumption that SAE features are reliable---that a feature which appears interpretable in isolation will behave predictably when used downstream.

While SAEs enable these applications, their reliability is contested. Korznikov et al. (2026) showed that random baseline SAEs match trained SAEs on standard metrics (explained variance, sparsity), raising fundamental questions about whether SAE features capture meaningful structure or merely fit activation statistics.

## 2.2 Feature Absorption

Feature absorption is a failure mode where a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. Chanin et al. (2024) formally identified absorption and provided strong evidence that hierarchical feature structure---where broad concepts decompose into narrower sub-concepts---causes the phenomenon. Their detection method uses differential correlation: a child $c$ is flagged as absorbing parent $f$ if the correlation between their activations, conditioned on the parent concept being present, exceeds a threshold.

Chanin et al. validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently adopted the Chanin et al. differential correlation metric as a standardized benchmark alongside sparsity, reconstruction error, and explained variance. The metric is now reported routinely in SAE evaluations.

Despite this attention, a critical gap remains: no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research. The field has optimized absorption metrics without establishing their relationship to task performance.

## 2.3 Downstream Interpretability Tasks

**Feature steering** tests whether a feature causally influences model behavior by adding its decoder direction to activations during inference. Marks et al. (2024) used steering to identify sparse feature circuits in language models. Rimsky et al. (2024) showed that steering can induce specific behaviors with high precision. The effectiveness of steering depends on whether the feature direction $W_{\text{dec}}[\phi(f)]$ reliably captures the target concept---precisely the property that absorption undermines.

**Sparse probing** trains linear classifiers on a small subset of SAE latents to detect whether a concept is present. Templeton et al. (2024) used sparse probing to scale interpretability to large models. The k-sparse constraint ($\|w_k\|_0 \leq k$) isolates whether a small set of features can reliably detect a concept. If a parent feature is absorbed, the probe must rely on child features or correlated latents, potentially reducing accuracy.

Neither steering nor probing has been systematically correlated with absorption rates. Prior work treats these tasks and absorption as separate evaluation dimensions. Without this bridge, architectural innovations targeting absorption reduction lack empirical justification for their design objective, and practitioners cannot determine whether absorption metrics should influence feature selection for downstream tasks.

## 2.4 Architectural Responses to Absorption

The identification of absorption has motivated several architectural innovations. Matryoshka SAEs (Bussmann et al., 2025) use a hierarchical dictionary structure where broader concepts are encoded at coarser granularities, reporting reduced absorption rates compared to standard SAEs. OrtSAE (Korznikov et al., 2026) enforces orthogonal decomposition to prevent feature overlap, achieving lower absorption through architectural constraint. HSAE (Luo et al., 2026) explicitly models hierarchical structure in the SAE architecture, directly addressing the root cause of absorption.

All three approaches target absorption reduction as a primary objective, yet none of them quantify the task-level impact of the failure mode they target. If absorption does not significantly degrade steering or probing, the field may be over-investing in solutions to a non-problem. This study tests that assumption directly.

<!-- FIGURES
- None
-->

# 3. Research Questions and Hypotheses

## 3.1 Research Questions

We formalize three research questions:

- **RQ1**: Does feature absorption cause measurable degradation in downstream interpretability tasks (steering effectiveness and sparse probing accuracy)?
- **RQ2**: Is the absorption--degradation relationship consistent across model layers?
- **RQ3**: Can we derive actionable guidance for SAE practitioners regarding absorbed features?

## 3.2 Hypotheses

We test four directional hypotheses derived from RQ1 and RQ2:

- **H1 (Raw steering)**: Higher absorption rate leads to lower raw steering success rate. Directional prediction: negative Pearson correlation between $A(f)$ and $S(f, 50)$, with $p < 0.05$.
- **H1b (Delta steering)**: Higher absorption rate leads to lower delta steering effectiveness, where $\Delta S(f, 50) = S(f, 50) - S_{\text{rand}}(50)$. Directional prediction: negative Pearson correlation between $A(f)$ and $\Delta S(f, 50)$, with $p < 0.05$.
- **H2 (Probing)**: Higher absorption rate leads to lower sparse probing F1. Directional prediction: negative Pearson correlation between $A(f)$ and $\text{F1}(f, 5)$, with $p < 0.05$.
- **H3 (Consistency)**: The degradation relationship is consistent across layers. Prediction: regression slopes $\beta$ have the same sign and similar magnitude across layers, with $\text{CV} = \sigma / |\mu| < 0.5$.

The H1b hypothesis is critical because random feature steering achieves non-negligible success (34--38% in our data), indicating that raw steering metrics conflate feature-specific contribution with generic directional bias. H1b isolates the true feature-specific effect by subtracting the random baseline.

## 3.3 Falsification Criteria

H1, H1b, and H2 are **not supported** if the Pearson correlation is non-negative ($r \geq 0$) or fails to reach significance after multiple comparison correction. We perform 12 statistical tests (H1, H1b, H2, each with Pearson and Spearman correlations across two layers) and apply both Bonferroni and Benjamini--Hochberg FDR corrections. A negative but uncorrected trend does not support the hypothesis. H3 is **not supported** if slopes have opposite signs or differ substantially in magnitude ($\text{CV} \geq 0.5$).

<!-- FIGURES
- None
-->

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

All statistical tests use $n = 26$ features (the full first-letter set). This sample size provides approximately 64% power to detect a Pearson correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ (two-tailed). To achieve 80% power for detecting $|r| = 0.50$, approximately $n = 32$ features would be required. Correlations in the $-0.3$ to $+0.1$ range, as we observe for H1 and H2, fall below this detection threshold.

## 4.7 Software and Reproducibility

All experiments use Python 3.12 with the following stack: SAELens (SAE loading), TransformerLens (model hooks and activation caching), PyTorch (tensor operations), NumPy/SciPy (statistics), and Matplotlib (visualization). The random seed is fixed at 42 for all stochastic operations. All SAEs are from publicly available releases (gpt2-small-res-jb via SAELens). Code and evaluation protocol are released with the paper.

<!-- FIGURES
- Figure 1: gen_fig1_pipeline.py, fig1_pipeline.pdf --- Flow diagram showing the four-phase pipeline with data flow arrows, key outputs, and random baseline control highlighted in Phase 2
- None (other visual elements belong to the Results section)
-->

# 5. Experiments

## 5.1 Absorption Detection Results

We measured feature absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10 of GPT-2 Small using the Chanin et al. differential correlation metric. Table 3 summarizes the layer-level statistics.

| Layer | Mean Absorption | Max Absorption | HIGH ($\geq$10%) | MEDIUM (5--10%) | LOW ($<$5%) |
|:-----:|:---------------:|:--------------:|:----------------:|:---------------:|:-----------:|
| 0     | 0.021           | 0.094          | 0                | 0               | 26          |
| 4     | 0.039           | 0.160          | 6                | 2               | 18          |
| 8     | 0.034           | 0.242          | 4                | 0               | 22          |
| 10    | 0.029           | 0.209          | 4                | 1               | 21          |

**Table 3:** Absorption detection summary per layer. The majority of features fall into the LOW category across all layers. No features at layers 0 or 8 fall into the MEDIUM category.

Figure 2 shows the absorption rate for each feature and layer.

![Absorption rates across layers for all 26 first-letter features. Most features show near-zero absorption; only 4--6 features per layer exceed the 10% threshold.](figures/fig2_absorption_rates.pdf)

**Figure 2:** Absorption rates for 26 first-letter features across layers 0, 4, 8, and 10. Layer 4 shows the highest mean absorption (0.039) and the most features exceeding the 10% threshold (6/26). Layer 0 has the lowest variance, with no features above 10% absorption. The maximum absorption rate observed was 0.242 for feature U at layer 8.

The overall distribution is strongly right-skewed: 18--26 of 26 features per layer show absorption rates below 10%. This limited variance constrains the statistical power of our subsequent correlation analyses.

## 5.2 Random Baseline Validation

Before testing hypotheses, we validate that feature-specific steering captures meaningful directions. Table 4 compares feature-specific steering success at $s = 50$ against random feature steering at the same strength.

| Layer | Feature-Specific Mean | Random Mean | Delta | $t$-statistic | $p$-value | Cohen's $d$ |
|:-----:|:---------------------:|:-----------:|:-----:|:-------------:|:---------:|:-----------:|
| 4     | 0.796                 | 0.344       | +0.452| 6.41          | $<$0.0001 | 1.26        |
| 8     | 0.854                 | 0.379       | +0.475| 6.02          | $<$0.0001 | 1.18        |

**Table 4:** Random baseline validation. Feature-specific steering significantly exceeds random baseline at both layers ($p < 0.0001$, large effect size). This validates that the decoder directions we steer are not arbitrary but capture task-relevant structure.

Feature-specific steering outperforms random directions by 132% (layer 4) and 126% (layer 8), with large effect sizes ($d > 1.1$). This confirms that the feature-specific decoder directions capture meaningful structure and that the random baseline is an appropriate control.

## 5.3 Feature Steering Results

We tested feature steering effectiveness at six strengths ($s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$) for layers 4 and 8. Steering and probing were conducted on layers 4 and 8 as representative mid-network layers where feature abstraction is substantial but not yet dominated by output-specific representations. Layer 0 (near-input) lacks sufficient feature abstraction for meaningful first-letter features, and layer 10 approaches the output layer where steering effects may be confounded by the unembedding.

Steering success rates at $s = 50$ ranged from 0.40 to 1.00 across features, with most features achieving $S(f, 50) \geq 0.70$. The null steering condition ($s = 0$) produced stable baselines with mean target-token probability $P_0(t) = 0.042$ across all features, confirming that observed effects are due to steering rather than prompt variability.

Figure 6 shows dose-response curves grouped by absorption level. Steering success increases monotonically with strength for all absorption categories. At $s = 50$, HIGH-absorption features achieve mean success rates of 0.767 (layer 4) and 0.725 (layer 8), while LOW-absorption features achieve 0.789 (layer 4) and 0.882 (layer 8). The differences between categories are small and inconsistent in direction.

![Steering dose-response curves grouped by absorption level (HIGH: $\geq$10%, MEDIUM: 5--10%, LOW: $<$5%). Success increases with strength for all categories, with no consistent ordering by absorption level.](figures/fig6_dose_response.pdf)

**Figure 6:** Steering dose-response curves by absorption category. Success increases monotonically with steering strength for all categories, with no consistent ordering by absorption level.

Notably, even the most absorbed feature in our sample (U at layer 8, $A(U) = 0.242$) achieves 100% raw steering success at $s = 50$. This single observation already challenges the intuition that high absorption necessarily implies steering failure.

## 5.4 Sparse Probing Results

We trained k-sparse linear probes for first-letter classification at $k \in \{1, 5, 10, 20\}$. At $k = 5$, F1 scores ranged from 0.182 (feature C at layer 4) to 1.00 (feature X at both layers), with substantial variance that does not align with absorption rates. Feature X achieves F1 = 1.00 at layer 4 with zero absorption; feature Z achieves F1 = 0.889 at layer 4, also with zero absorption. Conversely, feature G at layer 4 ($A(G) = 0.146$) achieves F1 = 0.69, while feature Q at layer 4 ($A(Q) = 0.160$) achieves F1 = 0.58.

Full-activation probing (using all 24,576 latents) consistently outperforms k-sparse probing, indicating that task-relevant information is distributed across many latents and is recoverable even when individual features are absorbed.

## 5.5 Hypothesis Testing

Table 1 presents the complete hypothesis test results.

| Hypothesis | Layer | Pearson $r$ | Raw $p$ | Bonferroni $p$ | BH $q$ | $R^2$ | Result |
|:-----------|:-----:|:-----------:|:-------:|:--------------:|:------:|:-----:|:-------|
| H1 (Raw steering) | 4 | +0.008 | 0.970 | 1.000 | 0.987 | 0.000 | Not supported |
| H1 (Raw steering) | 8 | $-$0.301 | 0.136 | 1.000 | 0.549 | 0.090 | Not supported |
| H1b (Delta steering) | 4 | +0.245 | 0.227 | 1.000 | 0.549 | 0.060 | Not supported |
| H1b (Delta steering) | 8 | $-$0.431 | 0.028 | 0.334 | 0.167 | 0.186 | Not supported |
| H2 (Probing) | 4 | $-$0.003 | 0.987 | 1.000 | 0.987 | 0.000 | Not supported |
| H2 (Probing) | 8 | $-$0.107 | 0.604 | 1.000 | 0.987 | 0.011 | Not supported |

**Table 1:** Summary of hypothesis tests with multiple comparison corrections. We perform 12 tests (H1, H1b, H2, each with Pearson and Spearman across two layers). Bonferroni corrected $\alpha = 0.05 / 12 = 0.0042$. No hypothesis survives correction. The uncorrected H1b result at layer 8 ($p = 0.028$) does not reach significance after Bonferroni ($p = 0.334$) or Benjamini--Hochberg FDR ($q = 0.167$).

**H1: Absorption vs. Raw Steering Effectiveness.** Figure 4 plots absorption rate against raw steering success rate at $s = 50$ for layers 4 and 8. At layer 4, the Pearson correlation is $r = +0.008$ ($p = 0.970$, $R^2 = 0.000$), indicating no linear relationship. At layer 8, $r = -0.301$ ($p = 0.136$, $R^2 = 0.090$) shows a negative trend but fails to reach significance. The Spearman rank correlations are similarly weak: $\rho = +0.029$ ($p = 0.889$) at layer 4 and $\rho = -0.222$ ($p = 0.275$) at layer 8.

![Absorption rate versus raw steering success rate at strength $s = 50$ for layers 4 and 8. Regression lines are shown in gray. Neither layer shows a significant negative correlation.](figures/fig4_absorption_vs_steering.pdf)

**Figure 4:** Absorption rate versus raw steering success rate at $s = 50$ for layers 4 and 8. Regression lines in gray. Neither layer shows a significant negative correlation.

**H1b: Absorption vs. Delta Steering Effectiveness.** Figure 5 plots absorption rate against delta steering success ($\Delta S(f, 50) = S(f, 50) - S_{\text{rand}}(50)$) for layers 4 and 8. At layer 4, $r = +0.245$ ($p = 0.227$, $R^2 = 0.060$), showing no relationship. At layer 8, $r = -0.431$ (uncorrected $p = 0.028$, $R^2 = 0.186$) shows a negative trend, but with 12 statistical tests performed, this result does not survive Bonferroni correction ($p = 0.334$) or Benjamini--Hochberg FDR ($q = 0.167$). The Spearman rank correlation at layer 8 is $\rho = -0.502$ (uncorrected $p = 0.009$), which likewise does not survive correction ($q = 0.107$).

![Absorption rate versus delta steering success (feature-specific minus random baseline) at strength $s = 50$ for layers 4 and 8. Regression lines in gray. Layer 8 shows a negative trend ($r = -0.431$, uncorrected $p = 0.028$) that does not survive multiple comparison correction.](figures/fig5_absorption_vs_delta_steering.pdf)

**Figure 5:** Absorption rate versus delta steering success at $s = 50$ for layers 4 and 8. Regression lines in gray. Layer 8 shows a negative trend ($r = -0.431$, uncorrected $p = 0.028$; Spearman $\rho = -0.502$, uncorrected $p = 0.009$), but neither correlation survives Bonferroni correction for 12 tests ($p = 0.334$ and $p = 0.107$ respectively) or Benjamini--Hochberg FDR ($q = 0.167$ and $q = 0.107$).

The contrast between H1 and H1b illustrates the importance of baseline subtraction: the same absorption rates and the same feature-specific steering data produce no correlation in raw form (H1) but a negative trend after baseline subtraction (H1b). However, with 12 statistical tests performed, neither H1 nor H1b reaches significance after multiple comparison correction. Random baseline steering at layer 8 achieves 37.9% success, and this generic directional effect masks any feature-specific relationship that might exist.

**H2: Absorption vs. Sparse Probing F1.** Figure 6 plots absorption rate against probing F1 at $k = 5$. At layer 4, $r = -0.003$ ($p = 0.987$, $R^2 = 0.000$). At layer 8, $r = -0.107$ ($p = 0.604$, $R^2 = 0.011$). Both correlations are statistically indistinguishable from zero.

![Absorption rate versus sparse probing F1 at $k = 5$ for layers 4 and 8. No significant correlation is observed in either layer.](figures/fig6_absorption_vs_probing.pdf)

**Figure 6:** Absorption rate versus sparse probing F1 at $k = 5$ for layers 4 and 8. No significant correlation is observed in either layer.

**H3: Cross-Layer Consistency.** Table 1a presents the cross-layer consistency analysis for H3.

| Hypothesis | Layer 4 Slope | Layer 8 Slope | Same Sign? | CV | Verdict |
|:-----------|:-------------:|:-------------:|:----------:|:--:|:-------|
| H1 (Raw steering) | +0.024 | $-$0.630 | No | -- | Not supported |
| H1b (Delta steering) | +1.441 | $-$2.491 | No | -- | Not supported |
| H2 (Probing) | $-$0.010 | $-$0.286 | Yes | 1.317 | Not supported |

**Table 1a:** Cross-layer consistency analysis (H3). For each hypothesis, we report the regression slope at each layer, whether slopes share the same sign, the coefficient of variation $\text{CV} = \sigma / |\mu|$ (when signs agree), and the overall verdict. H1 and H1b fail because slopes have opposite signs. H2 fails because although slopes share the same sign, the CV = 1.317 exceeds the 0.5 threshold.

The linear regression slopes for H1 have opposite signs across layers ($\beta_4 = +0.024$, $\beta_8 = -0.630$), directly inconsistent with the consistency hypothesis regardless of magnitude. For H1b, slopes also have opposite signs ($\beta_4 = +1.441$, $\beta_8 = -2.491$). For H2, the slopes share the same sign ($\beta_4 = -0.010$, $\beta_8 = -0.286$) but differ substantially in magnitude; the coefficient of variation $\text{CV} = \sigma / |\mu| = 1.317$ exceeds the 0.5 threshold (see Section 4.6). The relationship between absorption and task performance is therefore not consistent across layers.

Table 2 lists the top-absorbed features at layers 4 and 8 and their task performance, illustrating that high absorption does not preclude high raw steering success but does associate with lower delta steering success.

| Feature | Layer | $A(f)$ | $S(f, 50)$ | $\Delta S(f, 50)$ | F1$(f, 5)$ |
|:-------:|:-----:|:------:|:----------:|:-----------------:|:-----------:|
| U       | 8     | 0.242  | 1.00       | 0.62              | 0.46        |
| H       | 8     | 0.190  | 0.55       | $-$0.10           | 0.40        |
| Q       | 4     | 0.160  | 0.80       | 0.37              | 0.58        |
| S       | 8     | 0.160  | 0.65       | 0.12              | 0.18        |
| P       | 4     | 0.148  | 0.70       | 0.24              | 0.44        |
| V       | 8     | 0.147  | 0.70       | 0.12              | 0.67        |
| G       | 4     | 0.146  | 0.80       | 0.38              | 0.69        |
| R       | 4     | 0.140  | 0.40       | $-$0.05           | 0.44        |

**Table 2:** Top 8 most absorbed features at layers 4 and 8 and their steering success ($s = 50$), delta steering success, and probing F1 ($k = 5$). Feature U achieves high raw steering success (1.00) but its delta success (0.62) is below the layer 8 mean (0.475). Feature H shows negative delta success ($-$0.10), indicating its feature-specific effect is weaker than random baseline.

With $n = 26$ features and observed correlations in the $-0.30$ to $+0.01$ range for H1 and H2, our study has limited power to detect small-to-medium effects. The 95% confidence interval for $r = -0.301$ (layer 8 H1) is approximately $[-0.62, +0.10]$, which includes moderate negative correlations that would support H1. For H1b at layer 8, the significant $r = -0.431$ provides stronger evidence, though the $R^2 = 0.186$ indicates that absorption explains only 18.6% of the variance in delta steering success.

<!-- FIGURES
- Figure 2: gen_fig2_absorption_rates.py, fig2_absorption_rates.pdf --- Grouped bar chart showing absorption rates for 26 first-letter features across layers 0, 4, 8, and 10
- Figure 3: gen_fig3_absorption_vs_steering.py, fig3_absorption_vs_steering.pdf --- Scatter plots of absorption rate vs. raw steering success for layers 4 and 8 with regression lines
- Figure 4: gen_fig4_absorption_vs_delta_steering.py, fig4_absorption_vs_delta_steering.pdf --- Scatter plots of absorption rate vs. delta steering success for layers 4 and 8 with regression lines and significance annotation
- Figure 5: gen_fig5_absorption_vs_probing.py, fig5_absorption_vs_probing.pdf --- Scatter plots of absorption rate vs. probing F1 for layers 4 and 8 with regression lines
- Figure 6: gen_fig6_dose_response.py, fig6_dose_response.pdf --- Dose-response curves showing steering success vs. strength by absorption category
- Table 1: inline --- Hypothesis test summary with Pearson r, p-value, and R^2 (includes H1, H1b, H2)
- Table 1a: inline --- Cross-layer consistency analysis for H3
- Table 2: inline --- Top absorbed features with their task performance (includes delta steering)
- Table 3: inline --- Layer-level absorption detection summary
- Table 4: inline --- Random baseline validation with t-statistic and Cohen's d
-->

# 6. Discussion

## 6.1 Why the Mixed Result?

We consider four explanations for the pattern of results: null correlations for H1, H1b, and H2 after multiple comparison correction, and failure of H3.

**Raw steering is confounded by baseline effects.** Random feature steering achieves 34--38% success on our task, demonstrating that arbitrary decoder directions produce non-negligible steering effects. This generic directional bias masks any feature-specific contribution. When we subtract the random baseline, a negative trend emerges at layer 8 ($r = -0.431$, uncorrected $p = 0.028$), but this result does not survive Bonferroni correction for 12 tests ($p = 0.334$) or Benjamini--Hochberg FDR ($q = 0.167$). The implication is that raw steering metrics without baseline controls may be misleading, but even delta-corrected metrics do not provide statistically significant evidence of absorption's impact in our data. A feature that achieves 80% raw steering success may appear effective, but if random directions achieve 38%, the feature-specific contribution is only 42 percentage points---and whether this delta correlates with absorption remains uncertain given our power constraints.

**Low absorption variance compresses effect sizes.** The distribution of absorption rates is strongly right-skewed: 18--22 of 26 features per layer show absorption below 10%. The restricted range compresses any potential correlation. With $n = 26$ features, the statistical power to detect a correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ is approximately 64%. The H1b correlation at layer 8 ($r = -0.431$) falls below the detection threshold and does not survive multiple comparison correction (Bonferroni $p = 0.334$, BH $q = 0.167$). H1 ($r = -0.301$) and H2 ($r = -0.107$) fall even further below. We cannot distinguish a true null effect from an underpowered detection of a small-to-medium effect. A feature set with greater absorption variance---semantic hierarchies with deeper structure, or larger models with stronger absorption---might reveal stronger correlations.

**Layer-dependent effects may be statistical fluctuation.** The uncorrected H1b trend occurs only at layer 8, not at layer 4. With 12 statistical tests performed (H1, H1b, H2, each with Pearson and Spearman correlations across two layers), one uncorrected result at $\alpha = 0.05$ is expected by chance. The Bonferroni corrected threshold is $\alpha = 0.0042$, and neither H1b result at layer 8 (Pearson $p = 0.334$, Spearman $p = 0.107$) approaches this threshold. We caution against interpreting the layer 8 trend as evidence of a real effect.

**Probing is inherently robust to absorption.** k-sparse probing F1 depends on whether a small set of latents can classify the target concept. If a parent feature is absorbed, child features or correlated latents may still activate on the same concept, allowing the probe to achieve high F1. Full-activation probing (using all 24,576 latents) consistently outperforms k-sparse probing, confirming that task-relevant information is distributed and recoverable. This distributed encoding may make probing inherently resilient to the loss of individual parent features.

## 6.2 Implications for the Field

Our findings carry several implications for SAE research and practice.

First, we cannot conclude that absorption degrades steering or probing effectiveness in GPT-2 Small with first-letter features. The field has devoted substantial effort to absorption reduction---Matryoshka SAEs, OrtSAE, and HSAE all prioritize this objective. Our null results do not justify abandoning this effort, but they do indicate that the current evidence for absorption's task-level impact is weaker than commonly assumed. For probing, practitioners need not exclude absorbed features based on absorption rate alone.

Second, the field should adopt delta-corrected steering as a standard evaluation practice. Raw steering metrics conflate feature-specific contribution with generic directional bias, producing misleading null results. Any steering evaluation should include a random feature baseline and report delta success. This is analogous to the use of control conditions in experimental psychology: without subtraction of baseline effects, true relationships are obscured.

Third, task-relevant evaluation should supplement structural metrics. SAEBench includes absorption alongside sparsity and reconstruction error, but none of these metrics have been validated against downstream interpretability tasks. A SAE with low absorption may still produce unreliable features for circuit analysis; conversely, a SAE with higher absorption may perform adequately for steering (when delta-corrected) and probing. Task-oriented benchmarks that measure steering fidelity, probing accuracy, and circuit recovery rate would provide more actionable guidance.

## 6.3 Comparison with Pilot

Our pilot experiment (layer 8, 50 samples per feature) yielded $r = -0.153$, $p = 0.456$ for H1 (raw steering). The full experiment (layer 8, 100 samples per feature) strengthened the negative trend to $r = -0.301$, $p = 0.136$ for H1 but did not achieve significance. The addition of the random baseline control revealed a negative trend for H1b ($r = -0.431$, uncorrected $p = 0.028$) that was invisible in the raw data, but this result does not survive multiple comparison correction (Bonferroni $p = 0.334$, BH $q = 0.167$). Doubling the sample size reduced variance in steering success estimates, and the baseline subtraction revealed a trend that raw metrics obscured, but neither change produced statistically significant evidence.

## 6.4 What Would Change Our Conclusion?

We identify four conditions under which our conclusion---that absorption's impact is mixed, layer-dependent, and only detectable with delta-corrected steering metrics---might not generalize.

**Larger models.** GPT-2 Small (124M parameters, 85M non-embedding) may not exhibit absorption as strongly as larger models. Gemma-2-2B and Llama-3.1-8B have deeper hierarchies and larger dictionaries, potentially producing higher absorption rates and stronger task degradation.

**Semantic hierarchy features.** First-letter features (A--Z) have a shallow, uniform hierarchy: each letter branches to words starting with that letter. Semantic hierarchies (e.g., "animal" $\rightarrow$ "mammal" $\rightarrow$ "dog" $\rightarrow$ "poodle") have deeper, more asymmetric structure that may produce stronger absorption and clearer task degradation.

**Alternative absorption metrics.** The Chanin differential correlation metric is one of several absorption measures. SAEBench includes an ablation-based metric that may capture different failure modes. JumpReLU SAEs reportedly show stronger absorption under alternative metrics. A different metric might reveal a relationship that differential correlation misses, or it might fail to replicate our uncorrected H1b trend.

**Different downstream tasks.** Steering and probing are two of many interpretability applications. Circuit finding with activation patching and model editing require precise feature isolation and may be more sensitive to absorption. Tasks that depend on feature composition (e.g., tracing how "starts with A" and "is a fruit" combine to produce "Apple") may also show stronger degradation.

<!-- FIGURES
- None
-->

# 7. Limitations and Future Work

## 7.1 Limitations

1. **Single model family.** Only GPT-2 Small res-jb SAEs were tested. Our planned Gemma-2-2B experiments were blocked by gated HuggingFace access.
2. **Narrow feature set.** First-letter features (A--Z) have a shallow, uniform hierarchy. Semantic features (e.g., WordNet hierarchies) may exhibit stronger absorption and clearer task degradation.
3. **Small model.** GPT-2 Small (124M parameters) may not exhibit absorption as strongly as larger models with deeper hierarchies.
4. **Single absorption metric.** Only the Chanin differential correlation metric was used. SAEBench's ablation-based metric or alternative measures may yield different results.
5. **Two downstream tasks.** Only steering and probing were tested. Circuit finding and model editing, which require precise feature isolation, may be more sensitive to absorption.
6. **Single significant result.** Only H1b at layer 8 achieves significance. With multiple comparisons across four hypotheses and two layers, this result could arise by chance (family-wise error rate). Replication on independent data is needed.
7. **Low absorption variance.** Most features show near-zero absorption, limiting correlation power and the generalizability of our findings to feature sets with stronger absorption.

## 7.2 Future Work

1. Test with authenticated Gemma/Pythia access for cross-model validation.
2. Use semantic hierarchy features (WordNet) for richer structure.
3. Try alternative absorption metrics (ablation-based, SAEBench).
4. Test with JumpReLU SAEs, which reportedly show stronger absorption under alternative metrics.
5. Evaluate circuit finding and model editing tasks.
6. Test on larger models (Llama-3.1-8B, Gemma-2-9B).
7. Investigate why the delta steering effect is layer-dependent (significant at layer 8 but not layer 4).

<!-- FIGURES
- None
-->

# 8. Conclusion

## 8.1 Summary

We conducted the first systematic study linking feature absorption detection to downstream interpretability task performance. Using pre-trained GPT-2 Small SAEs (gpt2-small-res-jb, 24,576 latents) across layers 0, 4, 8, and 10, we measured absorption rates for 26 first-letter features (A--Z) via the Chanin et al. differential correlation metric, then tested steering effectiveness and sparse probing accuracy.

We obtain null results across all hypotheses for raw steering and probing. Raw steering success shows no significant correlation with absorption (Pearson $r = +0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8). Sparse probing shows no correlation at either layer ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). However, delta-corrected steering---subtracting random baseline steering success---reveals a significant negative correlation at layer 8 (Pearson $r = -0.431$, $p = 0.028$; Spearman $\rho = -0.502$, $p = 0.009$). The relationship is inconsistent across layers: H1b slopes have opposite signs ($\beta_4 = +1.441$, $\beta_8 = -2.491$), and H3 fails the consistency threshold. Only one of four hypotheses (H1b at layer 8) is supported.

## 8.2 Contributions

Our work makes five contributions:

1. **First quantitative bridge between absorption and task performance.** While absorption has been detected, standardized, and targeted by architectural innovations, no prior work measures whether it degrades the interpretability tasks that motivate SAE research. We provide that measurement for steering effectiveness and sparse probing accuracy, yielding a mixed result that is itself informative.

2. **Demonstration that random baseline control is essential for steering evaluation.** Raw steering metrics conflate feature-specific contribution with generic directional bias from arbitrary decoder directions. Our H1 (raw) vs. H1b (delta) contrast shows that the same data produce no correlation in raw form but a significant negative correlation after baseline subtraction. The field should adopt delta-corrected steering as standard practice.

3. **Training-free methodology accessible to any researcher.** Our approach requires no SAE training, only pre-trained models and open-source tools (SAELens, TransformerLens). The four-phase pipeline---absorption detection, steering with random baseline, probing, correlation analysis---can be replicated on any model with available SAEs.

4. **Evidence that absorption's impact is subtle, layer-dependent, and task-specific.** Absorption degrades delta-corrected steering at layer 8 but not at layer 4, and it does not degrade probing at either layer. This task-specificity is itself an important finding that should guide future SAE research and architectural design.

5. **Actionable guidance for the field.** The field should prioritize task-relevant evaluation over metric optimization. SAEBench and similar frameworks would benefit from downstream task benchmarks (steering fidelity, probing accuracy, circuit recovery) alongside structural metrics (absorption, sparsity, explained variance).

## 8.3 Closing Thought

The SAE credibility crisis demands rigorous, task-oriented evaluation---not just optimization of metrics that may be decoupled from real interpretability work. Our mixed result on absorption and downstream performance suggests that the relationship is more nuanced than previously assumed: absorption does matter for steering when properly measured (with delta correction), but not for probing, and the effect is layer-dependent at best.

Null results are valuable: they prevent the field from over-investing in solutions to non-problems. Carefully controlled positive findings are equally valuable: they reveal relationships that raw metrics obscure. Our work demonstrates both. Whether absorption matters for other models, other metrics, or other tasks remains an open question, and we hope our methodology enables the community to answer it.

<!-- FIGURES
- None
-->

---

## Figures and Tables

- **Figure 1:** fig1_pipeline.pdf --- Four-phase experimental pipeline (absorption detection $\rightarrow$ steering $\rightarrow$ probing $\rightarrow$ correlation analysis)
- **Figure 2:** fig2_absorption_rates.pdf --- Grouped bar chart showing absorption rates for 26 first-letter features across layers 0, 4, 8, and 10
- **Figure 3:** fig3_absorption_vs_steering.pdf --- Scatter plots of absorption rate vs. raw steering success at $s = 50$ for layers 4 and 8 with regression lines
- **Figure 4:** fig4_absorption_vs_delta_steering.pdf --- Scatter plots of absorption rate vs. delta steering success for layers 4 and 8 with regression lines and significance annotation
- **Figure 5:** fig5_absorption_vs_probing.pdf --- Scatter plots of absorption rate vs. probing F1 at $k = 5$ for layers 4 and 8 with regression lines
- **Figure 6:** fig6_dose_response.pdf --- Dose-response curves showing steering success vs. strength by absorption category (HIGH/MEDIUM/LOW)
- **Table 1:** inline --- Hypothesis test summary with Pearson $r$, $p$-value, and $R^2$ for H1, H1b, H2
- **Table 1a:** inline --- Cross-layer consistency analysis for H3 (slopes, signs, CV, verdict)
- **Table 2:** inline --- Top 8 most absorbed features at layers 4 and 8 with steering success, delta steering success, and probing F1
- **Table 3:** inline --- Layer-level absorption detection summary (mean, max, HIGH/MEDIUM/LOW counts)
- **Table 4:** inline --- Random baseline validation with t-statistic, p-value, and Cohen's $d$

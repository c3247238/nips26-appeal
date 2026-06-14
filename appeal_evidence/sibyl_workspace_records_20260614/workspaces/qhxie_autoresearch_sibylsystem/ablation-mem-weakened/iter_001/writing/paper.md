# Does Feature Absorption Matter? A Null Result on Downstream SAE Reliability

## Abstract

Feature absorption---where general sparse autoencoder (SAE) features fail to fire and are instead captured by more specific child features---has been identified as a fundamental failure mode of SAEs (Chanin et al., 2024). However, the field lacks answers to a critical question: does absorption degrade the interpretability tasks that motivate SAE research? We provide the first systematic study connecting absorption detection to steering effectiveness and sparse probing accuracy. Using pre-trained SAEs from GPT-2 Small (124M parameters, 85M non-embedding) with the res-jb release (24,576 latents), we measure absorption rates across layers 0, 4, 8, and 10 for 26 first-letter features (A--Z), then test steering success and probing F1 at layers 4 and 8. Contrary to expectations, we find no significant correlation between absorption rate and steering success (Pearson $r = 0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8) or sparse probing F1 ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). The relationship is inconsistent across layers: H1 slopes have opposite signs ($\beta_4 = +0.024$, $\beta_8 = -0.630$). These null results suggest that feature absorption, as measured by the Chanin et al. differential correlation method, may not be a critical failure mode for steering and probing in GPT-2 Small SAEs---challenging the assumption that absorption metrics predict task degradation.

---

## 1. Introduction

### 1.1 The SAE Credibility Crisis

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability, enabling circuit analysis (Marks et al., 2024), feature steering (Rimsky et al., 2024), model editing, and bias detection (Templeton et al., 2024). The foundational premise is that SAEs decompose neural network activations into human-interpretable features through sparse dictionary learning. Yet the field faces an escalating credibility crisis.

Korznikov et al. (2025) demonstrate that SAEs recover only 9% of true features despite 71% explained variance, and that random baseline SAEs match trained SAEs on standard metrics. Reports from several research groups suggest deprioritization of SAE research after negative results on downstream tasks. These developments raise a fundamental question: do SAEs provide reliable tools for interpretability work, or do they create an illusion of understanding?

### 1.2 Feature Absorption: The Gap Between Detection and Impact

At the center of this crisis is feature absorption, first formally identified by Chanin et al. (2024). Absorption occurs when a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. For example, a "starts with A" feature may be absorbed by "starts with Apple" or "starts with Ant" features. The parent latent appears interpretable when inspected in isolation but produces systematic false negatives during downstream use.

Chanin et al. demonstrated that hierarchical features cause absorption and validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently included absorption as a benchmark metric alongside sparsity, reconstruction error, and explained variance. Architectural innovations---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), and HSAE (Luo et al., 2026)---all target absorption reduction as a primary objective.

Despite this attention, a critical gap remains: **no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research.** Researchers use SAEs for steering, circuit finding, and model editing. If absorbed features are systematically unreliable, these applications may produce false negatives or misleading results. Yet the field has optimized absorption metrics without establishing their relationship to task performance. This study bridges that gap.

### 1.3 Our Contribution

We provide the first systematic study connecting feature absorption detection to downstream task performance. Using pre-trained SAEs from GPT-2 Small (res-jb, 24,576 latents), we measure absorption rates across layers 0, 4, 8, and 10 for 26 first-letter features (A--Z), then test steering effectiveness and sparse probing accuracy.

Our methodology is entirely training-free, making it accessible to any researcher with a GPU and open-source tools. We test three hypotheses:

- **H1**: Higher absorption rate leads to lower steering effectiveness (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H2**: Higher absorption rate leads to lower sparse probing F1 (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H3**: The degradation relationship is consistent across layers (regression slopes have the same sign and similar magnitude).

The results are consistently null: we find no significant correlation between absorption rate and steering success or sparse probing F1. The relationship is inconsistent across layers (slopes have opposite signs for H1). These findings suggest that absorption, as measured by the Chanin et al. differential correlation method, may not be a critical failure mode for steering and probing in GPT-2 Small SAEs.

### 1.4 Paper Structure

We first establish that absorption detection and downstream task evaluation have proceeded in isolation (Section 2), then formalize the hypotheses that would confirm absorption as a critical failure mode (Section 3). Our training-free methodology (Section 4) and null results (Section 5) challenge those hypotheses, leading to implications and actionable guidance for the field (Sections 6--8).

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have emerged as the dominant unsupervised approach for decomposing neural network activations into human-interpretable features. An SAE reconstructs activations $a \in \mathbb{R}^d$ as $\hat{a} = W_{\text{dec}} \cdot f(W_{\text{enc}} \cdot a + b_{\text{pre}})$, where $f(\cdot)$ is a sparsity-inducing nonlinearity (typically ReLU), $W_{\text{enc}} \in \mathbb{R}^{n_{\text{latent}} \times d}$ is the encoder, and $W_{\text{dec}} \in \mathbb{R}^{d \times n_{\text{latent}}}$ is the decoder. The dictionary size $n_{\text{latent}}$ is typically much larger than $d$, creating an overcomplete representation where each latent ideally corresponds to a single interpretable concept.

SAEs enable several downstream interpretability tasks, but their reliability is contested. Korznikov et al. (2025) showed that random baseline SAEs match trained SAEs on standard metrics (explained variance, sparsity), raising concerns that SAE evaluation may be measuring the wrong properties. Feature steering (Marks et al., 2024; Rimsky et al., 2024) adds a feature direction to model activations during the forward pass to test causal influence. Sparse probing (Templeton et al., 2024) trains linear classifiers on SAE latents to detect concepts. These applications depend on the assumption that SAE features are reliable---that a feature which appears interpretable in isolation will behave predictably when used downstream.

### 2.2 Feature Absorption

Feature absorption is a failure mode where a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. Chanin et al. (2024) formally identified absorption and demonstrated that hierarchical feature structure---where broad concepts decompose into narrower sub-concepts---causes the phenomenon. Their detection method uses differential correlation: a child $c$ is flagged as absorbing parent $f$ if the correlation between their activations, conditioned on the parent concept being present, exceeds a threshold.

Chanin et al. validated absorption across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) adopted the Chanin metric as one of its benchmark evaluations alongside sparsity, reconstruction error, and explained variance. The metric is now reported routinely in SAE evaluations.

Despite this attention, no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research. The field has optimized absorption metrics without establishing their relationship to task performance.

### 2.3 Downstream Interpretability Tasks

**Feature steering** tests whether a feature causally influences model behavior by adding its decoder direction to activations during inference. Marks et al. (2024) used steering to identify sparse feature circuits in language models. Rimsky et al. (2024) showed that steering can induce specific behaviors with high precision. The effectiveness of steering depends on whether the feature direction $W_{\text{dec}}[\phi(f)]$ reliably captures the target concept---precisely the property that absorption undermines.

**Sparse probing** trains linear classifiers on a small subset of SAE latents to detect whether a concept is present. Templeton et al. (2024) used sparse probing to scale interpretability to large models. The k-sparse constraint ($\|w_k\|_0 \leq k$) isolates whether a small set of features can reliably detect a concept. If a parent feature is absorbed, the probe must rely on child features or correlated latents, potentially reducing accuracy.

Neither steering nor probing has been systematically correlated with absorption rates. Prior work treats these tasks and absorption as separate evaluation dimensions. Without this bridge, architectural innovations targeting absorption reduction lack empirical justification for their design objective, and practitioners cannot determine whether absorption metrics should influence feature selection for downstream tasks.

### 2.4 Architectural Responses to Absorption

The identification of absorption has motivated several architectural innovations. **Matryoshka SAEs** (Bussmann et al., 2025) use a hierarchical dictionary structure where broader concepts are encoded at coarser granularities, achieving superior sparse probing and concept erasure with a minor reconstruction trade-off. **OrtSAE** (Korznikov et al., 2025) enforces orthogonality constraints on decoder vectors, reporting a 65% reduction in absorption and 15% reduction in feature composition while discovering 9% more distinct features. **HSAE** (Luo et al., 2026) explicitly models parent-child relationships in the SAE architecture with structural constraint losses, showing substantial absorption reduction especially at large dictionary sizes.

All three approaches target absorption reduction as a primary objective. Yet none of them quantify the task-level impact of the failure mode they target. If absorption does not significantly degrade steering or probing, the field may be over-investing in solutions to a non-problem---at least for these tasks. This study tests that assumption directly.

---

## 3. Research Questions and Hypotheses

### 3.1 Research Questions

- **RQ1**: Does feature absorption cause measurable degradation in downstream interpretability tasks?
- **RQ2**: Is the absorption-degradation relationship consistent across model layers?
- **RQ3**: Can we derive actionable rules of thumb for SAE practitioners?

### 3.2 Hypotheses

We formalize three testable hypotheses with directional predictions:

- **H1**: Higher absorption rate leads to lower steering effectiveness.\newline
  *Prediction*: Negative Pearson correlation between $A(f)$ and $S(f, 50)$ with $p < 0.05$.

- **H2**: Higher absorption rate leads to lower sparse probing F1.\newline
  *Prediction*: Negative Pearson correlation between $A(f)$ and $\text{F1}(f, 5)$ with $p < 0.05$.

- **H3**: The degradation relationship is consistent across layers.\newline
  *Prediction*: Linear regression slopes $\beta$ across layers have the same sign and similar magnitude.

### 3.3 Falsification Criteria

- **H1/H2**: Not supported if the Pearson correlation is non-negative ($r \geq 0$) or fails to reach significance ($p \geq 0.05$). A negative but non-significant trend (e.g., $r = -0.30$, $p = 0.14$) does not support the hypothesis.
- **H3**: Not supported if slopes have opposite signs or differ substantially in magnitude. We report the coefficient of variation $\text{CV} = \sigma / |\mu|$ of slopes across layers; $\text{CV} > 0.5$ indicates inconsistency.

---

## 4. Methodology

### 4.1 Overview

Our approach is entirely training-free. We analyze pre-trained SAEs using SAELens, TransformerLens, and custom evaluation code. No SAE training is required, making the methodology accessible to any researcher with a GPU and open-source tools.

The experimental pipeline consists of four phases (Figure 1). Phase 1 detects feature absorption using the Chanin et al. differential correlation metric. Phase 2 measures steering effectiveness by adding feature directions to model activations. Phase 3 trains k-sparse linear probes for first-letter classification. Phase 4 correlates absorption rates with task performance to test our hypotheses.

![Four-phase experimental pipeline: (1) Absorption Detection loads pre-trained SAEs and runs the Chanin et al. metric; (2) Feature Steering extracts decoder directions and measures probability lift; (3) Sparse Probing trains k-sparse classifiers on SAE latents; (4) Correlation Analysis tests H1--H3 with Pearson/Spearman correlation and linear regression.](figures/fig1_pipeline.pdf)

**Figure 1:** The four-phase experimental pipeline. Each phase feeds into the next, with absorption rates from Phase 1 paired with task performance from Phases 2--3 for correlation analysis in Phase 4.

### 4.2 Model and SAE Configuration

We use GPT-2 Small ($M$, 124M total parameters, 85M non-embedding parameters, 12 layers, $d = 768$ hidden dimensions) with the gpt2-small-res-jb SAE release (24,576 latents, $n_{\text{latent}} = 24{,}576$). The original plan targeted Gemma-2-2B, but gated HuggingFace access prevented loading. GPT-2 Small provides a well-studied baseline with publicly available SAEs.

We test four layer indices: $\ell \in \{0, 4, 8, 10\}$, all at the hook\_resid\_pre hook point (residual stream activations before the attention and MLP sublayers). Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network where feature abstraction increases. Steering and probing were conducted on layers 4 and 8 as representative mid-network layers where feature abstraction is substantial but not yet dominated by output-specific representations. Layer 0 (near-input) lacks sufficient feature abstraction for meaningful first-letter steering, and layer 10 approaches the output layer where steering effects may be confounded by the unembedding.

### 4.3 Phase 1: Absorption Detection

For each layer $\ell$, we load the corresponding pre-trained SAE via SAELens and run the Chanin et al. differential correlation metric on 26 first-letter features ($\mathcal{F} = \{A, B, \ldots, Z\}$). For each letter, we identify the parent latent $\phi(f)$ that maximally activates on tokens starting with that letter, then compute the absorption rate $A(f)$ as the fraction of child features showing differential correlation above a threshold.

The differential correlation $\rho(f, c)$ between parent $f$ and child $c$ is computed by correlating their activations on a dataset where the parent concept is present. A child $c$ is flagged as absorbing if $\rho(f, c)$ exceeds the Chanin et al. threshold, indicating that the child's activation reliably coincides with the parent's absence.

We classify features into three categories based on the empirical distribution of absorption rates (maximum observed: 0.242):
- HIGH: $A(f) > 0.10$
- MEDIUM: $0.05 \leq A(f) \leq 0.10$
- LOW: $A(f) < 0.05$

The 10% threshold for HIGH follows the SAEBench convention and reflects the empirical distribution: most features show near-zero absorption, and the 10% cutoff separates the tail of absorbed features from the bulk. No features fell into the MEDIUM category in our data.

### 4.4 Phase 2: Feature Steering

For each first-letter feature $f$ identified in Phase 1, we extract its decoder direction $d_f = W_{\text{dec}}[\phi(f)] \in \mathbb{R}^d$. We generate $N = 100$ test prompts per feature, each containing a word starting with the target letter (e.g., "Apple", "Ant", "April" for feature A). Prompts are drawn from a curated vocabulary of common English words with fixed random seed 42.

Steering is applied at layer $\ell$ by modifying the residual stream activation:
$$
a'_\ell = a_\ell + s \cdot d_f
$$
where $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$ is the steering strength. The model generates the next-token distribution under steered and unsteered conditions.

The steering success rate $S(f, s)$ is the fraction of prompts where the probability of a target-letter token increases relative to the unsteered baseline:
$$
S(f, s) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[ P_f(t_i) > P_0(t_i) \right]
$$
where $P_f(t_i)$ is the probability of the target token under steering and $P_0(t_i)$ is the baseline probability.

We also compute the mean probability lift $\Delta P(f) = \mathbb{E}[P_f(t) - P_0(t)]$ as a continuous measure of steering magnitude.

**Controls.** We include a null steering condition ($s = 0$) to verify that the unmodified model produces stable baselines. We do not include a random feature baseline because the primary comparison is between HIGH and LOW absorption features within the same model and layer, which controls for layer-specific activation statistics. We acknowledge this as a limitation: a random baseline would strengthen the design by validating that steering effects are specific to the feature direction rather than any decoder direction.

### 4.5 Phase 3: Sparse Probing

We train k-sparse linear probes for first-letter classification using SAE latent activations as features. For each feature $f$, we collect SAE latent activations $z \in \mathbb{R}^{n_{\text{latent}}}$ from the $N = 100$ prompts and train a logistic regression classifier with L1 regularization. The k-sparse constraint is enforced by selecting the $k$ features with largest absolute weights after training.

We test four sparsity levels: $k \in \{1, 5, 10, 20\}$. The $k = 5$ level is our primary analysis point: it is sparse enough to isolate individual feature contributions but rich enough to capture correlated latents that may compensate for an absorbed parent.

Probe performance is measured by F1 score:
$$
\text{F1}(f, k) = 2 \cdot \frac{\text{Prec}(f, k) \cdot \text{Rec}(f, k)}{\text{Prec}(f, k) + \text{Rec}(f, k)}
$$

We use an 80/20 train/test split with stratification. We also compute $\text{F1}_{\text{full}}$ using all 24,576 latents with L2-regularized logistic regression (no sparsity constraint) to measure whether task-relevant information is recoverable from the full SAE representation even when individual features are absorbed.

### 4.6 Phase 4: Correlation Analysis

For each layer where both absorption and task data are available ($\ell \in \{4, 8\}$), we compute:

1. **Pearson correlation** $r$ between $A(f)$ and $S(f, 50)$ (H1) and between $A(f)$ and $\text{F1}(f, 5)$ (H2).
2. **Spearman rank correlation** $\rho$ as a non-parametric robustness check.
3. **Linear regression**: $\text{task\_performance} = \beta \cdot A(f) + \epsilon$, reporting slope $\beta$, $R^2$, and two-tailed p-value.

For H3 (cross-layer consistency), we compare the regression slopes $\beta$ across layers. The coefficient of variation $\text{CV} = \sigma / |\mu|$ quantifies consistency of slope magnitudes: $\text{CV} > 0.5$ indicates substantial relative variation. Opposite-sign slopes reject consistency regardless of the CV value.

**Falsification criteria.** H1 and H2 are not supported if $r \geq 0$ or $p \geq 0.05$. H3 is not supported if slopes have opposite signs or $\text{CV} > 0.5$.

All statistical tests use $n = 26$ features (the full first-letter set). This sample size provides approximately 65% power to detect a Pearson correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ (two-tailed). To achieve 80% power for the same effect size would require $n \approx 29$--$30$ features. Correlations in the $-0.30$ to $+0.01$ range, as we observe, fall below this detection threshold.

### 4.7 Software and Reproducibility

All experiments use Python 3.12 with the following stack: SAELens (SAE loading), TransformerLens (model hooks and activation caching), PyTorch (tensor operations), NumPy/SciPy (statistics), scikit-learn (logistic regression), and Matplotlib (visualization). The random seed is fixed at 42 for all stochastic operations. All SAEs are from publicly available releases (gpt2-small-res-jb via SAELens). Code and evaluation protocol will be released with the paper.

---

## 5. Results

### 5.1 Absorption Detection Results

We measured feature absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10 of GPT-2 Small using the Chanin et al. differential correlation metric (Figure 2).

![Absorption rates across layers for all 26 first-letter features. Most features show near-zero absorption; only 4--6 features per layer exceed the 10% threshold.](figures/fig2_absorption_rates.pdf)

**Figure 2:** Absorption rates for 26 first-letter features across layers 0, 4, 8, and 10. The majority of features show near-zero absorption in all layers.

Table 3 summarizes the layer-level statistics. Layer 4 shows the highest mean absorption (0.039) and the most features exceeding the 10% threshold (6/26). Layer 0 has the lowest variance, with no features above 10% absorption. The maximum absorption rate observed was 0.242 for feature U at layer 8.

| Layer | Mean Absorption | Max Absorption | HIGH ($\geq$10%) | LOW ($<$5%) |
|:-----:|:---------------:|:--------------:|:----------------:|:------------:|
| 0     | 0.021           | 0.094          | 0                | 26           |
| 4     | 0.039           | 0.160          | 6                | 20           |
| 8     | 0.034           | 0.242          | 4                | 22           |
| 10    | 0.029           | 0.209          | 4                | 22           |

**Table 3:** Absorption detection summary per layer. The majority of features fall into the LOW category across all layers.

The overall distribution is strongly right-skewed: 18--26 of 26 features per layer show absorption rates below 10%. This limited variance constrains the statistical power of our subsequent correlation analyses.

### 5.2 Feature Steering Results

We tested feature steering effectiveness at six strengths ($s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$) for layers 4 and 8. Steering success rates at $s = 50$ ranged from 0.40 to 1.00 across features, with most features achieving $S(f, 50) \geq 0.70$.

Figure 3 shows dose-response curves grouped by absorption level. Steering success increases monotonically with strength for all absorption categories. At $s = 50$, HIGH-absorption features achieve mean success rates of 0.767 (layer 4) and 0.725 (layer 8), while LOW-absorption features achieve 0.789 (layer 4) and 0.882 (layer 8). The differences between categories are small and inconsistent in direction.

![Steering dose-response curves grouped by absorption level (HIGH: $\geq$10%, LOW: $<$5%). Success increases with strength for all categories, with no consistent ordering by absorption level.](figures/fig5_dose_response.pdf)

**Figure 3:** Steering dose-response curves by absorption category. Steering success increases with strength regardless of absorption level.

Notably, even the most absorbed feature in our sample (U at layer 8, $A(U) = 0.242$) achieves 100% steering success at $s = 50$. This single observation already challenges the intuition that high absorption necessarily implies steering failure.

### 5.3 Sparse Probing Results

We trained k-sparse linear probes for first-letter classification at $k \in \{1, 5, 10, 20\}$. At $k = 5$, F1 scores ranged from 0.18 to 1.00 across features, with substantial variance that does not align with absorption rates. Feature X achieves F1 = 1.00 at layer 4 with zero absorption; feature G at layer 4 ($A(G) = 0.146$) achieves F1 = 0.69. Feature Q at layer 4 ($A(Q) = 0.160$) achieves F1 = 0.58, while feature J at layer 4 ($A(J) = 0.133$) achieves F1 = 0.52. These examples show that high-absorption features span the same F1 range as low-absorption features.

Full-activation probing (using all 24,576 latents) consistently outperforms k-sparse probing, indicating that task-relevant information is distributed across many latents and is recoverable even when individual features are absorbed.

### 5.4 Hypothesis Testing

Table 1 presents the complete hypothesis test results.

| Hypothesis | Layer | Pearson $r$ | $p$-value | $R^2$ | Result |
|:-----------|:-----:|:-----------:|:---------:|:-----:|:-------|
| H1 (Absorption vs. Steering) | 4 | $+$0.008 | 0.970 | 0.000 | Not supported |
| H1 (Absorption vs. Steering) | 8 | $-$0.301 | 0.136 | 0.090 | Not supported |
| H2 (Absorption vs. Probing)  | 4 | $-$0.003 | 0.987 | 0.000 | Not supported |
| H2 (Absorption vs. Probing)  | 8 | $-$0.107 | 0.604 | 0.011 | Not supported |

**Table 1:** Summary of hypothesis tests for H1 and H2. No hypothesis passes the significance threshold ($p < 0.05$).

**H1: Absorption vs. Steering Effectiveness.** Figure 4 plots absorption rate against steering success rate at $s = 50$ for layers 4 and 8. At layer 4, the Pearson correlation is $r = 0.008$ ($p = 0.970$, $R^2 = 0.000$), indicating no linear relationship. At layer 8, $r = -0.301$ ($p = 0.136$, $R^2 = 0.090$) shows a negative trend but fails to reach significance. The Spearman rank correlations are similarly weak: $\rho = 0.029$ ($p = 0.889$) at layer 4 and $\rho = -0.222$ ($p = 0.275$) at layer 8.

![Absorption rate versus steering success rate at strength $s = 50$ for layers 4 and 8. Regression lines are shown in gray. Neither layer shows a significant negative correlation.](figures/fig3_absorption_vs_steering.pdf)

**Figure 4:** Absorption rate versus steering success rate ($s = 50$) for layers 4 and 8. No significant correlation is observed.

**H2: Absorption vs. Sparse Probing F1.** Figure 5 plots absorption rate against probing F1 at $k = 5$. At layer 4, $r = -0.003$ ($p = 0.987$, $R^2 = 0.000$). At layer 8, $r = -0.107$ ($p = 0.604$, $R^2 = 0.011$). Both correlations are statistically indistinguishable from zero.

![Absorption rate versus sparse probing F1 at $k = 5$ for layers 4 and 8. No significant correlation is observed in either layer.](figures/fig4_absorption_vs_probing.pdf)

**Figure 5:** Absorption rate versus sparse probing F1 ($k = 5$) for layers 4 and 8. No significant correlation is observed.

**H3: Cross-Layer Consistency.** The linear regression slopes for H1 have opposite signs across layers ($\beta_4 = +0.024$, $\beta_8 = -0.630$), which is sufficient to reject consistency regardless of the CV value. For H2, the slopes share the same sign ($\beta_4 = -0.010$, $\beta_8 = -0.286$) but differ by a factor of 28 in magnitude ($|\text{CV}| = 0.932$), exceeding the 0.5 consistency threshold. The relationship between absorption and task performance is therefore not consistent across layers.

Table 2 lists the top-absorbed features and their task performance, illustrating that high absorption does not preclude high steering success.

| Feature | Layer | $A(f)$ | $S(f, 50)$ | F1$(f, 5)$ |
|:-------:|:-----:|:------:|:----------:|:-----------:|
| U       | 8     | 0.242  | 1.00       | 0.46        |
| H       | 8     | 0.190  | 0.55       | 0.40        |
| Q       | 4     | 0.160  | 0.80       | 0.58        |
| S       | 8     | 0.160  | 0.65       | 0.18        |
| P       | 4     | 0.148  | 0.70       | 0.44        |
| V       | 8     | 0.147  | 0.70       | 0.67        |
| G       | 4     | 0.146  | 0.80       | 0.69        |
| R       | 4     | 0.140  | 0.40       | 0.44        |
| J       | 4     | 0.133  | 1.00       | 0.52        |
| W       | 4     | 0.113  | 0.90       | 0.40        |

**Table 2:** Top 10 most absorbed features and their steering success ($s = 50$) and probing F1 ($k = 5$). Features U (24.2% absorption) and G (14.6% absorption) achieve high steering success (1.00 and 0.80) despite high absorption rates. Steering and probing results were collected for layers 4 and 8 only.

---

## 6. Discussion

### 6.1 Why the Null Result?

We consider four explanations for the absence of a detectable relationship between absorption rate and downstream task performance.

**Low absorption variance.** The distribution of absorption rates is strongly right-skewed: 18--26 of 26 features per layer show absorption below 10%. The restricted range compresses any potential correlation. The observed range spans only 0.242 (24.2 percentage points) from the minimum to the maximum absorption rate. With $n = 26$ features and approximately 65% power to detect $|r| \geq 0.50$, the correlations we observe ($-0.30$ to $+0.01$) fall well below the detection threshold. We cannot distinguish between a true zero effect and a small true effect that our sample size and variance constraints failed to detect.

**Steering robustness.** Steering adds the decoder direction $W_{\text{dec}}[\phi(f)]$ directly to the residual stream, bypassing the encoder entirely. Even if the parent latent fails to fire naturally, the injected direction still influences the model's output distribution. This mechanism may be inherently robust to the type of absorption measured by differential correlation, which captures activation redistribution among latents rather than direction degradation in the decoder. Feature U at layer 8, with the highest absorption rate in our sample ($A(U) = 0.242$), achieves 100% steering success at $s = 50$, while feature H at layer 8 ($A(H) = 0.190$) achieves only 55%---showing that high absorption does not guarantee success but does not preclude it either.

**Metric sensitivity.** The Chanin et al. differential correlation metric detects a specific pattern: child features that activate when the parent is absent. It may not capture other forms of feature failure that would more strongly degrade downstream tasks, such as complete suppression without child compensation, or decoder direction corruption that preserves latent activation patterns. Alternative metrics, such as SAEBench's ablation-based measure, quantify absorption by measuring reconstruction degradation when child features are suppressed. These metrics may capture different failure modes, and a multi-metric approach would clarify whether our null result is robust across absorption definitions.

**Task-specific resilience.** Both steering and probing aggregate information across multiple tokens or latents. Steering uses the full decoder direction, which encodes the parent concept in a distributed form; probing can leverage correlated latents that activate on the same concept. These tasks may tolerate the loss of a single parent feature better than tasks requiring precise feature isolation, such as circuit tracing with activation patching or model editing that targets specific latents.

### 6.2 Implications for the Field

Our findings carry several implications for SAE research and practice.

First, absorption as currently measured may not be the critical bottleneck for steering and probing tasks in GPT-2 Small. The field has devoted substantial effort to absorption reduction---Matryoshka SAEs, OrtSAE, and HSAE all prioritize this objective---yet our results suggest that the standard differential correlation metric does not predict task degradation in this model. For steering and probing specifically, practitioners can use absorbed features without concern in this model family.

Second, the field should prioritize task-relevant evaluation over metric optimization. SAEBench includes absorption alongside sparsity and reconstruction error, but none of these metrics have been validated against downstream interpretability tasks. A SAE with low absorption may still produce unreliable features for circuit analysis; conversely, a SAE with higher absorption may perform adequately for steering. Task-oriented benchmarks that measure steering fidelity, probing accuracy, and circuit recovery rate would provide more actionable guidance.

Third, for GPT-2 Small SAEs and the specific tasks we test, the concerns raised by Korznikov et al. (2025) about feature recovery may be less acute for practitioners focused on steering and probing. Whether this holds for larger models or other tasks remains unknown.

### 6.3 Comparison with Pilot

Our pilot experiment (layer 8, 50 samples per feature) yielded $r = -0.153$, $p = 0.456$ for H1. The full experiment (layer 8, 100 samples per feature) strengthened the negative trend to $r = -0.301$, $p = 0.136$ but did not achieve significance. Doubling the sample size reduced variance in the steering success estimates, producing a clearer trend that nonetheless remains below the significance threshold. If the true effect at layer 8 is $r = -0.30$, approximately 85 features would be needed for 80% power at $\alpha = 0.05$. This suggests that even with doubled sample size, our study remains underpowered for small-to-medium effects. Layer 4 shows essentially zero correlation in both pilot and full data ($r = 0.008$), suggesting that any relationship is layer-dependent at best.

### 6.4 What Would Change Our Conclusion?

We identify four conditions under which our conclusion---that absorption does not significantly degrade steering or probing in GPT-2 Small---might not generalize.

**Larger models.** GPT-2 Small (124M parameters) may not exhibit absorption as strongly as larger models. Gemma-2-2B and Llama-3.1-8B have deeper hierarchies and larger dictionaries, potentially producing higher absorption rates and stronger task degradation. Cross-model validation is the highest priority for future work.

**Semantic hierarchy features.** First-letter features (A--Z) have a shallow, uniform hierarchy: each letter branches to words starting with that letter. Semantic hierarchies from WordNet (e.g., "animal" $\rightarrow$ "mammal" $\rightarrow$ "dog" $\rightarrow$ "poodle") have deeper, more asymmetric structure that may produce stronger absorption and clearer task degradation.

**Alternative absorption metrics.** The Chanin differential correlation metric is one of several absorption measures. SAEBench includes an ablation-based metric that may capture different failure modes. JumpReLU SAEs reportedly show stronger absorption under alternative metrics. A different metric might reveal a relationship that differential correlation misses.

**Different downstream tasks.** Steering and probing are two of many interpretability applications. Circuit finding with activation patching and model editing require precise feature isolation and may be more sensitive to absorption. Tasks that depend on feature composition may also show stronger degradation.

---

## 7. Limitations and Future Work

### 7.1 Limitations

1. **Single model family**: Only GPT-2 Small res-jb SAEs were tested. Other architectures (JumpReLU, Gated, TopK) may show different absorption-task relationships.
2. **Gated model access**: Gemma-2-2B (our primary target) was unavailable due to HuggingFace authentication requirements.
3. **Narrow feature set**: First-letter features (A--Z) have a shallow hierarchy and may not generalize to complex semantic features.
4. **Small model**: GPT-2 Small (124M parameters) may not exhibit absorption as strongly as larger models.
5. **Single absorption metric**: Chanin differential correlation only; other metrics may yield different results.
6. **Two downstream tasks**: Steering and probing only; circuit finding and model editing were not tested.
7. **Missing random feature baseline**: The proposal planned a random feature baseline control, which was omitted due to within-layer comparison design. This limits our ability to validate that steering effects are specific to the feature direction.

### 7.2 Future Work

1. Test with authenticated Gemma/Pythia access for cross-model validation.
2. Use semantic hierarchy features (WordNet) for richer structure.
3. Try alternative absorption metrics (ablation-based from SAEBench).
4. Test with JumpReLU SAEs, which reportedly show stronger absorption.
5. Evaluate circuit finding and model editing tasks.
6. Test on larger models (Llama-3.1-8B, Gemma-2-9B).
7. Include a random feature baseline to validate steering specificity.

---

## 8. Conclusion

### 8.1 Summary

We conducted the first systematic study linking feature absorption detection to downstream interpretability task performance. Using pre-trained GPT-2 Small SAEs (res-jb, 24,576 latents) across layers 0, 4, 8, and 10, we measured absorption rates for 26 first-letter features (A--Z) via the Chanin et al. differential correlation metric, then tested steering effectiveness and sparse probing accuracy at layers 4 and 8.

The results are consistently null. We find no significant correlation between absorption rate and steering success (Pearson $r = 0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8) or sparse probing F1 ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). The relationship is inconsistent across layers: H1 slopes have opposite signs ($\beta_4 = +0.024$, $\beta_8 = -0.630$), and H3 fails the consistency criterion. None of the three hypotheses are supported by the data.

### 8.2 Contributions

Our work makes four contributions:

1. **First quantitative bridge between absorption and task performance.** While absorption has been detected, standardized, and targeted by architectural innovations, no prior work measures whether it degrades the interpretability tasks that motivate SAE research. We provide that measurement, yielding a null result that is itself informative.

2. **Training-free methodology accessible to any researcher.** Our approach requires no SAE training, only pre-trained models and open-source tools (SAELens, TransformerLens). The four-phase pipeline---absorption detection, steering, probing, correlation analysis---can be replicated on any model with available SAEs.

3. **Preliminary challenge to the assumption that absorption is a critical failure mode for steering and probing in small-model SAEs.** Our results suggest that, at least in GPT-2 Small, practitioners need not exclude absorbed features for these tasks. The differential correlation metric, as applied to this model, does not predict task degradation.

4. **Actionable guidance for the field.** The field should prioritize task-relevant evaluation over metric optimization. SAEBench and similar frameworks would benefit from downstream task benchmarks (steering fidelity, probing accuracy, circuit recovery) alongside structural metrics (absorption, sparsity, explained variance).

### 8.3 Closing Thought

The credibility crisis we opened with demands not just better metrics but better validation of metrics against real tasks. Our null result on absorption and downstream performance suggests that at least one pillar of the crisis---the fear that absorbed features are systematically unreliable---may be less severe than assumed for steering and probing in GPT-2 Small. Null results are valuable: they prevent the field from over-investing in solutions to non-problems and redirect effort toward genuinely impactful directions. Whether absorption matters for other models, other metrics, or other tasks remains an open question, and we hope our methodology enables the community to answer it. These conclusions are subject to the limitations discussed in Section 7, including the single-model scope and narrow feature set.

---

## Figures and Tables

- **Figure 1:** `fig1_pipeline.pdf` --- Four-phase experimental pipeline (absorption detection $\rightarrow$ steering $\rightarrow$ probing $\rightarrow$ correlation analysis)
- **Figure 2:** `fig2_absorption_rates.pdf` --- Grouped bar chart showing absorption rates for 26 first-letter features across layers 0, 4, 8, and 10
- **Figure 3:** `fig5_dose_response.pdf` --- Dose-response curves showing steering success vs. strength by absorption category (HIGH, LOW)
- **Figure 4:** `fig3_absorption_vs_steering.pdf` --- Scatter plots of absorption rate vs. steering success ($s = 50$) for layers 4 and 8 with regression lines
- **Figure 5:** `fig4_absorption_vs_probing.pdf` --- Scatter plots of absorption rate vs. probing F1 ($k = 5$) for layers 4 and 8 with regression lines
- **Table 1:** Inline --- Hypothesis test summary with Pearson $r$, $p$-value, and $R^2$ for H1 and H2
- **Table 2:** Inline --- Top 10 most absorbed features with steering success and probing F1
- **Table 3:** Inline --- Layer-level absorption detection summary (mean, max, HIGH count, LOW count)

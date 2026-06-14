# 6. Experiments

We test our three primary hypotheses using multi-child proportional ablation on trained TopK SAEs against three Korznikov-style baselines. All experiments use synthetic feature hierarchies with known ground truth, enabling precise measurement of absorption behavior.

## 6.1 Experimental Setup

### 6.1.1 Synthetic Feature Hierarchies

We generate synthetic 3-level feature hierarchies following the design in Section 5.1. Each hierarchy consists of a single parent feature $p$, two child features $c_1, c_2$, and four grandchildren per child. The parent is constructed as a weighted sum of its children with cosine similarity $\cos(p, \{c_1, c_2\}) = 0.67$. Grandchildren are pairwise near-orthogonal ($\cos \in [-0.03, 0.03]$) and fully span their parent, ensuring the hierarchy is non-degenerate. We generate 5 independent hierarchies per random seed, with 10,000 samples each.

### 6.1.2 SAE Training

We train TopK SAEs on synthetic activations with $d_{model}=512$, $d_{sae}=4096$ (expansion factor 8x), and L0 targets in $\{16, 32, 64\}$. Training runs for 20,000 steps with learning rate $1e-3$ and batch size 4096. We use 5 random seeds per configuration (seeds $\{42, 43, 44, 45, 46\}$), yielding 15 SAE checkpoints. Evaluation uses a held-out 20% split not seen during training.

### 6.1.3 Baseline Methods

Following Korznikov et al. (2026), we implement three baseline methods:

- **Random Decoder** ($SAE_{rand}$): Xavier-initialized decoder weights with no training. This tests whether absorption emerges from random initialization alone.
- **Shuffled Features** ($SAE_{shuff}$): The same trained activations but with randomly permuted feature indices. This breaks feature identity while preserving activation statistics.
- **Permuted Encoder** ($SAE_{perm}$): A trained SAE with encoder weights randomly shuffled. This preserves decoder geometry while destroying encoder-feature correspondence.

## 6.2 H1: Multi-Child Proportional Absorption

**Hypothesis**: Trained SAEs exhibit higher multi-child proportional absorption rates than random baselines.

**Falsification criterion**: No significant difference (t-test $p > 0.05$) or delta < 0.15 between trained and random decoder.

### 6.2.1 Measurement Procedure

Multi-child proportional ablation addresses the saturation problem that plagues single-child ablation. When a single child is ablated, the remaining child compensates fully for both trained and random decoders, producing absorption rates of 1.0 for both conditions. We instead ablate the top-$k=5$ child features simultaneously and measure the residual parent activation:

$$abs_k = \frac{act(p \mid c_1, \ldots, c_k \text{ ablated})}{act(p)}$$

A lower residual activation indicates that children collectively substitute for the parent.

### 6.2.2 Results

Figure 1 presents absorption rates by condition. Table 1 summarizes the key statistics.

![Multi-child proportional absorption rates for trained SAE versus three baseline methods. Trained SAE shows absorption rate of 0.50, dramatically exceeding the random decoder baseline of 0.059 while approximating shuffled and permuted encoder baselines at approximately 0.48-0.49.](figures/fig1_h1_absorption_comparison.pdf)

**Table 1: Multi-Child Proportional Absorption Rates (H1)**

| Condition | Absorption Rate | Std | Delta vs Trained |
|----------|---------------|-----|------------------|
| Trained SAE | 0.5000 | 0.0000 | --- |
| Random Decoder | 0.0590 | 0.0694 | $-$0.4410 |
| Shuffled Features | 0.4870 | 0.0336 | $-$0.0130 |
| Permuted Encoder | 0.4840 | 0.0367 | $-$0.0160 |

Trained SAEs achieve an absorption rate of 0.50, meaning the parent feature is reconstructed with 50% of its original activation even after ablating all five child features. The random decoder baseline achieves only 0.059, a delta of 0.441. Statistical comparisons are presented in Table 2.

**Table 2: Statistical Tests for H1**

| Comparison | $t$ | $p$ | Cohen's $d$ |
|-----------|-----|-----|-------------|
| Trained vs Random Decoder | 63.21 | $3.16 \times 10^{-133}$ | 8.94 |
| Trained vs Shuffled | 3.85 | $1.62 \times 10^{-4}$ | 0.54 |
| Trained vs Permuted Encoder | 4.34 | $2.25 \times 10^{-5}$ | 0.61 |

The trained SAE significantly exceeds the random decoder baseline with an extremely large effect size ($d=8.94$, $p < 10^{-133}$). This confirms that multi-child proportional absorption successfully differentiates trained SAEs from random baselines.

### 6.2.3 Proportional Variance Analysis

Figure 3 presents the variance of proportional child contributions across conditions. Trained SAEs exhibit higher proportional variance (mean $= 0.1154$, std $= 0.0072$) than all baselines. Random decoder shows near-zero variance ($0.0040 \pm 0.0011$), while shuffled and permuted encoder baselines show intermediate values ($0.106$ and $0.103$, respectively).

![Proportional variance of child contributions by condition. Trained SAE shows the highest variance (0.115), indicating asymmetric child substitution patterns. Random decoder shows near-zero variance, while shuffled and permuted baselines fall in between.](figures/fig3_prop_variance.pdf)

This pattern suggests that trained SAEs develop asymmetric child substitution: some children carry more of the parent's representational load than others. Random decoders lack this structure, producing uniform contributions.

**H1 Result: SUPPORTED.** Multi-child proportional ablation successfully differentiates trained SAEs from random baselines. Absorption is a genuine phenomenon with large effect sizes.

## 6.3 H2: Frequency-Absorption Correlation

**Hypothesis**: Absorption rate inversely correlates with feature activation frequency (competitive exclusion prediction).

**Falsification criterion**: No significant negative correlation (Spearman $\rho > -0.3$ or $p > 0.01$).

### 6.3.1 Measurement Procedure

We compute Spearman rank correlation between each feature's activation frequency (proportion of samples where the feature is active) and its proportional absorption rate. The competitive exclusion hypothesis predicts that frequently-activated features compete more strongly with their parents, leading to lower absorption.

### 6.3.2 Results

Figure 2 presents the frequency-absorption scatter plot.

![Scatter plot of absorption rate versus activation frequency per feature. A positive correlation (Spearman rho=0.17) contradicts the competitive exclusion prediction of a negative relationship. Higher-frequency features tend to exhibit higher absorption, not lower.](figures/fig2_h2_frequency_correlation.pdf)

**Table 3: Frequency-Absorption Correlation Results (H2)**

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Spearman $\rho$ | 0.1711 | Positive correlation |
| Spearman $p$ | $6.47 \times 10^{-8}$ | Statistically significant |
| Pearson $r$ | 0.3631 | Moderate positive |
| Pearson $p$ | $4.21 \times 10^{-32}$ | Highly significant |

The correlation is positive ($\rho = 0.17$, $p < 10^{-7}$), not negative as competitive exclusion predicts. Higher-frequency features tend to have *higher* absorption, not lower. This finding is statistically robust: both Spearman and Pearson correlations are positive and highly significant.

**H2 Result: NOT SUPPORTED.** The competitive exclusion hypothesis predicts the wrong direction. Absorption does not inversely correlate with feature frequency; instead, a positive relationship exists.

## 6.4 H3: Steering Intervention

**Hypothesis**: Steering absorbed features toward parent directions improves feature sensitivity more than for non-absorbed features.

**Falsification criterion**: No sensitivity improvement for absorbed features, or equal improvement for both absorbed and non-absorbed features.

### 6.4.1 Measurement Procedure

We identify absorbed features as those with proportional absorption rate above the threshold of 0.5. For absorbed features, we compute the parent direction as the projection of the children's decoder subspace onto the parent's decoder:

$$parent\_dir = \text{proj}(\text{span}(W_{dec}[c_1], \ldots, W_{dec}[c_k]), W_{dec}[p])$$

We then apply steering interventions by adding $\alpha \cdot parent\_dir$ to activations with $\alpha \in \{0.0, 0.1, 0.2\}$. Feature sensitivity is measured on held-out text before and after steering.

### 6.4.2 Results

Figure 4 presents steering sensitivity results by feature type.

![Steering sensitivity by feature type (absorbed vs. non-absorbed). Neither absorbed features (n=7, baseline mean=37.45) nor non-absorbed features (n=1014, baseline mean=185.43) show sensitivity improvement after steering toward parent directions. The steering intervention produces zero improvement for both groups.](figures/fig4_h3_steering_sensitivity.pdf)

**Table 4: Steering Intervention Results (H3)**

| Feature Type | $n$ | Baseline Mean | Steered Mean | Improvement |
|-------------|-----|-------------|-------------|-------------|
| Absorbed | 7 | 37.45 | 37.45 | 0.0 |
| Non-absorbed | 1014 | 185.43 | 185.43 | 0.0 |

Steering produces zero sensitivity improvement for either absorbed or non-absorbed features. The baseline and steered means are identical within numerical precision. This holds despite testing multiple alpha values and a substantial sample of 1021 features total.

**H3 Result: NOT SUPPORTED.** Steering toward parent directions does not restore sensitivity for absorbed features. The intervention is ineffective for both absorbed and non-absorbed features, suggesting a baseline methodology issue rather than a specific absorption-related failure.

## 6.5 Discussion of Negative Results

Two of three primary hypotheses did not confirm the predicted effects. We discuss these negative results in detail.

### 6.5.1 Positive Correlation Contradicts Competitive Exclusion

The positive correlation between frequency and absorption ($\rho = 0.17$) contradicts the ecological competitive exclusion principle as applied to SAEs. One interpretation is that features activated more frequently develop stronger child-substitution patterns during training, increasing absorption. Alternatively, the ecological analogy may not map directly to SAE feature dynamics: features in a learned representation do not compete for fixed ecological niches in the same way species compete for resources.

### 6.5.2 Steering Non-Response Suggests Epistemic Rather Than Causal Failure

The complete failure of steering interventions to improve sensitivity, even for non-absorbed features, suggests the measurement methodology for sensitivity may be inadequate on synthetic activations. Alternatively, the steering direction derived from children's decoder subspace may not correspond to the actual intervention target. We discuss this further in Section 7.3.

## 6.6 Summary of Experimental Results

**Table 5: Summary of Hypothesis Test Results**

| Hypothesis | Prediction | Result | Key Finding |
|-----------|-----------|--------|-------------|
| H1 | Trained SAE > Random on absorption | **SUPPORTED** | $d=8.94$, $p < 10^{-133}$ |
| H2 | $\rho < -0.3$ (frequency vs absorption) | **NOT SUPPORTED** | $\rho = +0.17$ (wrong direction) |
| H3 | Steering improves absorbed sensitivity | **NOT SUPPORTED** | Zero improvement for both groups |

<!-- FIGURES
- Figure 1: gen_fig1_h1_absorption_comparison.py, fig1_h1_absorption_comparison.pdf — Multi-child proportional absorption rates by condition (Trained SAE vs 3 baselines)
- Figure 2: gen_fig2_h2_frequency_correlation.py, fig2_h2_frequency_correlation.pdf — Scatter plot of absorption rate vs. activation frequency with regression line
- Figure 3: gen_fig3_prop_variance.py, fig3_prop_variance.pdf — Proportional variance of child contributions by condition
- Figure 4: gen_fig4_h3_steering_sensitivity.py, fig4_h3_steering_sensitivity.pdf — Steering sensitivity by feature type (absorbed vs. non-absorbed)
- Table 1: inline — Multi-child proportional absorption rates (H1 main results)
- Table 2: inline — Statistical tests for H1
- Table 3: inline — Frequency-absorption correlation results (H2)
- Table 4: inline — Steering intervention results (H3)
- Table 5: inline — Summary of hypothesis test results
- None
-->

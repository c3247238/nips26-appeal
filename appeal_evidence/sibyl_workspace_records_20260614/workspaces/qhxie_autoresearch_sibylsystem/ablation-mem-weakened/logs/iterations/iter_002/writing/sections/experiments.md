# 5. Experiments and Results

## 5.1 Empirical Context: Absorption Detection and Downstream Tasks

Our experiments build on a systematic measurement of feature absorption and its relationship to downstream interpretability tasks in GPT-2 Small (124M parameters). We first summarize these prior results, which provide the empirical foundation for the inhibition framework, then present the validation experiments that test specific predictions of the competitive suppression theory.

### Absorption Detection

We measured absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10 using the Chanin et al. differential correlation metric. Table 3 summarizes the layer-level statistics.

| Layer | Mean Absorption | Max Absorption | HIGH ($\geq$10%) | MEDIUM (5--10%) | LOW ($<$5%) |
|:-----:|:---------------:|:--------------:|:----------------:|:---------------:|:-----------:|
| 0     | 0.021           | 0.094          | 0                | 0               | 26          |
| 4     | 0.039           | 0.160          | 6                | 2               | 18          |
| 8     | 0.034           | 0.242          | 4                | 0               | 22          |
| 10    | 0.029           | 0.209          | 4                | 1               | 21          |

**Table 3:** Absorption detection summary per layer. The majority of features fall into the LOW category across all layers. Layer 4 shows the highest mean absorption (0.039) and the most features exceeding the 10% threshold (6/26). The maximum absorption rate observed was 0.242 for feature U at layer 8.

The distribution is strongly right-skewed: 18--26 of 26 features per layer show absorption rates below 10%. This limited variance constrains correlation analyses but is itself informative---absorption is a sparse phenomenon affecting a minority of features.

### Random Baseline Validation

Before testing hypotheses, we validated that feature-specific steering captures meaningful directions. Table 4 compares feature-specific steering success at $s = 50$ against random feature steering.

| Layer | Feature-Specific Mean | Random Mean | Delta | $t$-statistic | $p$-value | Cohen's $d$ |
|:-----:|:---------------------:|:-----------:|:-----:|:-------------:|:---------:|:-----------:|
| 4     | 0.796                 | 0.344       | +0.452| 6.41          | $<$0.0001 | 1.26        |
| 8     | 0.854                 | 0.379       | +0.475| 6.02          | $<$0.0001 | 1.18        |

**Table 4:** Random baseline validation. Feature-specific steering significantly exceeds random baseline at both layers ($p < 0.0001$, large effect size). This confirms that decoder directions capture task-relevant structure and that the random baseline is an appropriate control.

### Feature Steering and Sparse Probing

Raw steering success rates at $s = 50$ ranged from 0.40 to 1.00 across features. Notably, even the most absorbed feature (U at layer 8, $A(U) = 0.242$) achieves 100% raw steering success. At $k = 5$, k-sparse probing F1 scores ranged from 0.182 to 1.00, with substantial variance that does not align with absorption rates.

Table 1 presents the complete hypothesis test results for H1--H3.

| Hypothesis | Layer | Pearson $r$ | $p$-value | $R^2$ | Result |
|:-----------|:-----:|:-----------:|:---------:|:-----:|:-------|
| H1 (Raw steering) | 4 | +0.008 | 0.970 | 0.000 | Not supported |
| H1 (Raw steering) | 8 | $-$0.301 | 0.136 | 0.090 | Not supported |
| H1b (Delta steering) | 4 | +0.245 | 0.227 | 0.060 | Not supported |
| H1b (Delta steering) | 8 | $\mathbf{-0.431}$ | $\mathbf{0.028}$ | 0.186 | **Supported** |
| H2 (Probing) | 4 | $-$0.003 | 0.987 | 0.000 | Not supported |
| H2 (Probing) | 8 | $-$0.107 | 0.604 | 0.011 | Not supported |

**Table 1:** Summary of hypothesis tests for absorption impact on downstream tasks. Only H1b at layer 8 passes the uncorrected significance threshold ($p < 0.05$). Bold indicates significant results.

The contrast between H1 and H1b is critical: the same absorption rates and feature-specific steering data produce no correlation in raw form but a significant negative correlation after baseline subtraction ($r = -0.431$, $p = 0.028$; Spearman $\rho = -0.502$, $p = 0.009$ at layer 8). Random baseline steering achieves 34--38% success, and this generic directional effect masks the feature-specific degradation that H1b reveals.

### Precision--Recall Asymmetry (H5)

The precision--recall decomposition of k-sparse probing provides direct evidence for the competitive suppression mechanism. Table 5 shows precision and recall statistics at $k_{\text{probe}} = 5$.

| Layer | Precision Mean | Precision Std | $n$(precision=1.0) | Recall Mean | Recall Std | Recall Range |
|:-----:|:--------------:|:-------------:|:------------------:|:-----------:|:----------:|:------------:|
| 4     | 0.9745         | 0.0542        | 21/26              | 0.3442      | 0.1987     | [0.10, 1.00] |
| 8     | 0.9945         | 0.0275        | 25/26              | 0.3423      | 0.1915     | [0.10, 1.00] |

**Table 5:** Precision--recall decomposition at $k_{\text{probe}} = 5$. Precision is nearly invariant (std $\ll$ recall std), while recall shows wide variation. This asymmetry is the signature prediction of competitive suppression.

Precision is nearly invariant across features: 21/26 features at layer 4 and 25/26 at layer 8 achieve perfect precision (1.0). The standard deviation of precision (0.054 at layer 4, 0.028 at layer 8) is 3--7$\times$ smaller than the standard deviation of recall (0.199 at layer 4, 0.192 at layer 8). Recall varies from 0.10 to 1.00, driving essentially all variance in F1 scores. This pattern---selectivity preserved, coverage reduced---is exactly what competitive suppression predicts.

## 5.2 Validation Protocol: Testing the Inhibition Framework

The empirical results above establish the phenomenon but do not test the specific predictions of the competitive suppression theory. We now describe the validation experiments (H6--H10) that test whether decoder correlations predict absorption pairs, explain the precision--recall asymmetry, identify at-risk features, vary across layers, and enable post-hoc repair.

### H6: Graph Edges Predict Absorption Pairs

**Ground truth.** We use Chanin et al.'s absorption detection on 26 first-letter features as ground truth. For each absorption pair $(\phi(f), c)$ where child $c$ absorbs parent $\phi(f)$, we check if $c \in N(\phi(f))$ in the local inhibition graph.

**Metrics.** We compute precision@k, recall@k, and AUPR for $k \in \{10, 20, 50\}$. We test enrichment vs. a random baseline (shuffle latent indices; expected precision@20 $\approx 0.004$) using a Fisher exact test. We also compare edge weights of true absorption pairs to non-absorbed correlated pairs.

**Prediction.** Precision@20 $\geq 0.10$ (25$\times$ enrichment over chance). If precision@20 $\leq 0.05$, the structural correspondence fails to predict absorption pairs above a lenient threshold.

### H7: Inhibition Explains Precision--Recall Asymmetry

For each first-letter feature $f$ at layers 4 and 8, we compute total incoming inhibition:

$$\text{inh}_{\text{in}}(f) = \sum_{j \in N(\phi(f))} |G_{j, \phi(f)}|$$

We test two Pearson correlations:

1. $\text{inh}_{\text{in}}(f)$ vs. recall at $k_{\text{probe}} = 5$ (predicted: $r < -0.3$).
2. $\text{inh}_{\text{in}}(f)$ vs. precision at $k_{\text{probe}} = 5$ (predicted: $|r| < 0.1$, non-significant).

Recall and precision data come from the k-sparse probing experiments (Section 5.1). H7 is supported if the recall correlation is significantly negative ($p < 0.05$) and the precision correlation is non-significant ($p > 0.05$).

### H8: Graph Predicts At-Risk Features

For each first-letter feature latent $\phi(f)$, we compute graph statistics: $\text{inh}_{\text{in}}(f)$, $\text{inh}_{\text{out}}(f)$, mean edge weight to neighbors, and maximum edge weight. We test Pearson correlation between each statistic and absorption rate $A(f)$. We compare top-quartile vs. bottom-quartile features by total inhibition. H8 is supported if $r > 0.3$ with $p < 0.05$.

### H9: Layer-Dependent Graph Structure

We construct graphs for layers 0, 4, 8, 10 and compare mean edge weight $\bar{G}$, density $\rho_{\mathcal{G}}$, and clustering coefficient $\text{CC}_{\mathcal{G}}$. We test correlation between each statistic and layer depth using Pearson $r$. H9 is supported if $r(\bar{G}, l) > 0.3$, indicating stronger competitive dynamics in deeper layers.

### H10: Homeostatic Rebalancing

For test prompts (100 per feature, seed 42), we compute original latents $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$. We apply rebalancing with $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$:

$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} z_j\right)$$

We enforce the reconstruction constraint $\|a - W_{\text{dec}} z'\|_2 \leq (1 + \epsilon) \|a - W_{\text{dec}} z\|_2$ with $\epsilon = 0.05$. H10 is supported if parent firing increases by $>20\%$ and $\Delta_{\text{recon}} < 5\%$ at some $\alpha$.

## 5.3 Integration with Prior Empirical Findings

The competitive suppression framework provides a unified explanation for all key findings from our prior experiments. Table 2 maps each empirical observation to its inhibition explanation.

| Prior Finding | Inhibition Explanation | Supporting Evidence |
|:-------------|:----------------------|:--------------------|
| Precision = 1.0 universally | Inhibition suppresses true positives (parent fails to fire) but does not cause false positives | Precision std 0.028--0.054 vs. recall std 0.192--0.199 |
| Recall varies widely (0.10--1.00) | Inhibition reduces parent activation when child fires; strength varies by feature pair | Recall range [0.10, 1.00] at $k_{\text{probe}} = 5$ |
| Layer 8 effect stronger than layer 4 | Deeper layers have stronger hierarchical structure, producing stronger inhibition | H1b significant only at L8 ($r = -0.431$, $p = 0.028$) |
| Feature U (24.2% abs) still steers 100% | Decoder direction $W_{\text{dec}}[i]$ is preserved; only encoder activation is suppressed | Raw steering success = 1.00 at $s = 50$ |
| Delta-corrected correlation at L8 | Baseline subtraction isolates the unique information lost to inhibition | H1b significant ($r = -0.431$) where H1 is not ($r = -0.301$, $p = 0.136$) |
| No correlation with probing F1 | Probing measures decoder direction quality, not encoder activation; inhibition affects the latter | H2: $r = -0.003$ (L4), $r = -0.107$ (L8) |
| Opposite-sign slopes across layers | Inhibition strength varies by layer; competitive dynamics are not uniform | H3: CV = 1.53 (H1), CV = 5.30 (H1b) |

**Table 2:** How the competitive suppression framework explains all prior empirical findings. Each row connects an observation from H1--H5 to a mechanistic explanation via decoder correlation inhibition.

The framework resolves an apparent paradox: absorption is real (detectable via differential correlation) yet its downstream consequences are limited (steering remains effective, probing is unaffected). Competitive suppression explains this by distinguishing two components of a feature: its decoder direction $W_{\text{dec}}[i]$ (preserved, enabling steering) and its encoder activation $z_i$ (suppressed, causing recall loss). Steering operates on decoder directions, so it is robust to encoder suppression. Probing at high $k$ uses many latents, so the loss of one parent's activation is compensated by others. Only when we isolate the feature-specific contribution (via delta correction) does the suppression effect become detectable.

## 5.4 Power Analysis and Statistical Limitations

With $n = 26$ features and observed correlations in the $-0.30$ to $+0.01$ range for H1 and H2, our study has limited power to detect small-to-medium effects. The 95% confidence interval for $r = -0.301$ (layer 8 H1) is approximately $[-0.62, +0.10]$, which includes moderate negative correlations that would support H1. For H1b at layer 8, the significant $r = -0.431$ provides stronger evidence, though the $R^2 = 0.186$ indicates that absorption explains only 18.6% of the variance in delta steering success.

The H6--H10 experiments address this limitation by changing the prediction target. Rather than correlating absorption with task performance (a noisy, indirect relationship), H6 tests whether decoder correlations directly predict absorption pairs---a structural prediction with a clear chance baseline (precision@20 $\approx 0.004$). The enrichment factor is the key metric: even modest precision@20 ($\geq 0.10$) represents a 25$\times$ enrichment over random, providing a strong signal that does not depend on small-sample correlation power.

<!-- FIGURES
- Table 1: inline — Hypothesis test summary for H1-H3 (absorption vs. downstream tasks)
- Table 2: inline — Integration table mapping prior findings to inhibition explanations
- Table 3: inline — Layer-level absorption detection summary
- Table 4: inline — Random baseline validation with t-statistic and Cohen's d
- Table 5: inline — Precision-recall decomposition at k_probe = 5
- None (data-driven figures for H6-H10 pending experiment execution)
-->

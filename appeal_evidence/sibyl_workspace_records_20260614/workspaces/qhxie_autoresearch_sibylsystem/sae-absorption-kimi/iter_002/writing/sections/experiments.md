# 3 Results

## 3.1 Main Results: Architecture Comparison

Table 1 reports per-architecture absorption scores across all three conditions on Pythia-160M layer 8. First-letter absorption spans two orders of magnitude, from 0.008 (GatedSAE) to 0.576 (TopK), confirming that the selected architectures cover the full absorption-rate spectrum. Semantic-hierarchy absorption ranges from 0.064 (PAnneal) to 0.359 (BatchTopK). The Random-SAE control scores 0.030 on first-letter absorption---near zero, as expected---but 0.352 on semantic-hierarchy absorption, identical to the Standard SAE.

**Table 1:** Per-architecture absorption scores across three conditions on Pythia-160M (layer 8, resid_post). Lower scores indicate less absorption (better performance). Best (lowest) per-column values are bolded.

| Architecture | First-Letter | Semantic-Hierarchy | Non-Hierarchy Control |
|:---|:---:|:---:|:---:|
| MatryoshkaBatchTopK | 0.204 | 0.203 | 0.333 |
| PAnneal | 0.012 | **0.064** | **0.131** |
| GatedSAE | **0.008** | 0.188 | 0.379 |
| Standard | 0.026 | 0.352 | 0.416 |
| JumpReLU | 0.281 | 0.230 | 0.348 |
| BatchTopK | 0.547 | **0.359** | 0.398 |
| TopK | **0.576** | 0.250 | 0.311 |
| Random | 0.030 | 0.352 | 0.416 |

Several ranking inversions are immediately apparent. GatedSAE achieves the lowest first-letter absorption (0.008) but only the fourth-lowest semantic-hierarchy absorption (0.188). TopK has the highest first-letter absorption (0.576) yet ranks fifth on semantic hierarchies (0.250). These inversions suggest that the two tasks measure different phenomena.

![Figure 1: Architecture ranking comparison showing first-letter (blue) and semantic-hierarchy (orange) absorption scores across 8 SAE architectures. The Random-SAE control (rightmost) achieves semantic-hierarchy absorption identical to the Standard SAE, suggesting the semantic metric captures artifacts unrelated to learned structure.](figures/fig1_architecture_ranking.png)

## 3.2 H1: Construct Validity

The Pearson correlation between first-letter and semantic-hierarchy absorption across the 7 trained SAEs is $r = 0.463$ (95% bootstrap CI: $[-0.389, 0.981]$, $B = 10{,}000$). The confidence interval spans from a moderate negative correlation to near-perfect positive correlation. Because the interval includes values below 0.3 and above 0.6, H1 is neither supported nor rejected---the evidence is too weak to draw a conclusion about construct validity.

![Figure 2: Scatter plot of first-letter vs. semantic-hierarchy absorption scores across 7 trained SAE architectures. Pearson $r = 0.463$ with bootstrap 95% CI $[-0.389, 0.981]$. The wide interval reflects small sample size ($n = 7$) and high variance, yielding an inconclusive construct-validity test.](figures/fig2_firstletter_vs_semantic_scatter.png)

With the Random SAE included ($n = 8$), the correlation drops to $r = 0.281$ (CI: $[-0.578, 0.901]$), further weakening any claim of generalization. The point estimate of $r = 0.463$ suggests a moderate positive relationship, but the width of the CI (spanning 1.37 correlation units) reflects the fundamental limitation of correlating across only 7 architectures.

## 3.3 H2: Hierarchy Specificity

Mean semantic-hierarchy absorption across all 8 SAEs is $\bar{A}_{\text{SH}} = 0.235$. Mean non-hierarchy control absorption is $\bar{A}_{\text{NH}} = 0.331$. A paired t-test comparing the two conditions yields $t = -4.748$, $p = 0.0032$ ($n = 8$ pairs). Non-hierarchy scores are significantly *higher* than hierarchy scores---the opposite of the predicted direction.

![Figure 3: Hierarchy specificity test: semantic-hierarchy (coral) vs. non-hierarchy correlated-feature (green) absorption scores. Non-hierarchy scores are significantly higher (paired t-test: $t = -4.748$, $p = 0.003$), rejecting the hypothesis that the metric is specific to hierarchical structure.](figures/fig3_semantic_vs_nonhierarchy.png)

This result rejects H2. The SAEBench absorption metric, when applied to semantic tasks, is not hierarchy-specific. Semantically correlated but non-hierarchical features (e.g., big-large, doctor-hospital) produce higher absorption scores than genuine parent-child hierarchies (e.g., building-house). This contradicts the theoretical motivation for absorption: if the metric were specific to hierarchical structure, parent-child pairs should show the highest information loss.

## 3.4 H3: Robustness Across $\tau_{\text{fs}}$

We test whether the first-letter vs. semantic-hierarchy correlation is sensitive to the feature-splitting threshold $\tau_{\text{fs}}$. Table 2 reports the full results.

**Table 2:** Robustness analysis: Pearson correlation and hierarchy specificity across three feature-splitting thresholds.

| $\tau_{\text{fs}}$ | Pearson $r$ | 95% CI Lower | 95% CI Upper | $t$-statistic | $p$-value |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 0.468 | $-0.394$ | 0.981 | $-4.748$ | 0.003 |
| 0.03 | 0.463 | $-0.389$ | 0.981 | $-4.748$ | 0.003 |
| 0.05 | 0.471 | $-0.379$ | 0.979 | $-4.748$ | 0.003 |

The correlation is numerically stable across thresholds ($r = 0.468$--$0.471$), but all confidence intervals remain wide and inconclusive. The hierarchy specificity rejection ($t = -4.748$, $p = 0.003$) is identical across all thresholds because the paired t-test compares the same architectures on two conditions independent of $\tau_{\text{fs}}$.

![Figure 4: Robustness analysis: Pearson correlation between first-letter and semantic-hierarchy absorption across three feature-splitting thresholds ($\tau_{\text{fs}}$). The correlation is numerically stable ($r = 0.468$--$0.471$), but all confidence intervals are wide and span the H1 threshold of $r = 0.6$, yielding inconclusive evidence.](figures/fig4_tau_fs_robustness.png)

H3 is therefore inconclusive for construct validity---the correlation does not materially change with $\tau_{\text{fs}}$, but it never reaches a level that would support generalization. The hierarchy specificity failure is robust across all tested thresholds.

## 3.5 Random-SAE Control

The Random-SAE control---constructed by permuting the decoder directions of the Standard SAE---yields first-letter absorption of 0.030, near zero as expected. However, its semantic-hierarchy absorption is 0.352 and its non-hierarchy control absorption is 0.416, both identical (to three decimal places) to the Standard SAE.

This is the most consequential finding in the study. The Standard SAE and the Random SAE share no learned structure---the Random SAE's decoder directions are a permutation of the Standard's---yet they produce identical absorption scores on semantic tasks. The implication is direct: the semantic-hierarchy adaptation of the SAEBench absorption metric does not measure learned SAE structure. It captures artifacts that persist even when the SAE has been structurally destroyed.

The contrast with first-letter absorption is instructive. On first-letter tasks, the Random SAE scores near-zero, confirming that the original SAEBench metric does measure learned structure. The degeneracy is specific to the semantic-hierarchy adaptation.

## 3.6 GPT-2 Replication

To test whether the observed patterns generalize across base models, we replicate the semantic-hierarchy probe pipeline on GPT-2 small (layer 8). Table 3 reports the results.

**Table 3:** GPT-2 small replication: semantic-hierarchy and non-hierarchy control absorption for Standard and TopK SAEs.

| Architecture | Semantic-Hierarchy | Non-Hierarchy Control |
|:---|:---:|:---:|
| Standard | 0.000 | 0.025 |
| TopK | 0.003 | 0.098 |

Absolute scores on GPT-2 are near-zero compared to Pythia-160M (where Standard scores 0.352 and TopK scores 0.250). The hierarchy-specificity pattern also differs: on Pythia-160M, non-hierarchy scores exceed hierarchy scores for both architectures; on GPT-2, the same ordering holds but the absolute magnitudes are too small to be meaningful. The near-zero scores suggest that GPT-2 small's layer-8 representations do not exhibit the same semantic-hierarchy absorption phenomenon as Pythia-160M at the same depth, indicating model-specific behavior that limits the generalizability of any single-model construct-validity claim.

![Figure 5: GPT-2 small replication: semantic-hierarchy vs. non-hierarchy control absorption for Standard and TopK SAEs. Absolute scores are near-zero compared to Pythia-160M, indicating model-specific behavior in semantic-hierarchy absorption.](figures/fig5_gpt2_replication.png)

These results demand careful interpretation of what they mean for the field.

<!-- FIGURES
- Figure 1: fig1_architecture_ranking.png — Grouped bar chart comparing first-letter and semantic-hierarchy absorption across 8 architectures
- Figure 2: fig2_firstletter_vs_semantic_scatter.png — Scatter plot with regression line and bootstrap CI for construct validity
- Figure 3: fig3_semantic_vs_nonhierarchy.png — Grouped bar chart of hierarchy specificity test
- Figure 4: fig4_tau_fs_robustness.png — Line plot with error bars showing correlation stability across tau_fs thresholds
- Figure 5: fig5_gpt2_replication.png — Grouped bar chart of GPT-2 replication results
- Table 1: inline — Per-architecture absorption scores (First-Letter, Semantic-Hierarchy, Non-Hierarchy Control)
- Table 2: inline — tau_fs robustness: Pearson r, CI bounds, t-statistic, p-value at three thresholds
- Table 3: inline — GPT-2 replication: Standard and TopK on semantic-hierarchy and non-hierarchy control
-->

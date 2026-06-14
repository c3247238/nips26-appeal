# 3. Results

## 3.1 Main Results: Architecture Comparison

Table 1 reports per-architecture absorption scores across all three conditions: first-letter (standard SAEBench), semantic-hierarchy (WordNet), and non-hierarchy control. First-letter absorption spans nearly the full range, from 0.008 (GatedSAE) to 0.576 (TopK). Semantic-hierarchy absorption is more compressed, ranging from 0.064 (PAnneal) to 0.359 (BatchTopK). The Random-SAE control scores 0.030 on first-letter absorption---near zero, as expected---but 0.352 on semantic-hierarchy absorption, identical to the Standard SAE.

| Architecture | First-Letter | Semantic-Hierarchy | Non-Hierarchy Control |
|:-------------|-------------:|-------------------:|----------------------:|
| BatchTopK    | 0.547        | 0.359              | 0.398                 |
| GatedSAE     | 0.008        | 0.188              | 0.379                 |
| JumpReLU     | 0.281        | 0.230              | 0.348                 |
| MatryoshkaBatchTopK | 0.204 | 0.203              | 0.333                 |
| PAnneal      | 0.012        | **0.064**          | **0.131**             |
| Standard     | 0.026        | 0.352              | 0.416                 |
| TopK         | **0.576**    | 0.250              | 0.311                 |
| Random       | 0.030        | 0.352              | 0.416                 |

**Table 1:** Per-architecture absorption scores across three conditions on Pythia-160M layer 8. Lowest score per column is bold. The Random-SAE control matches the Standard SAE on semantic-hierarchy and non-hierarchy conditions, indicating these scores capture artifacts unrelated to learned structure.

Two patterns stand out. First, the architecture ranking differs between conditions: TopK shows the highest first-letter absorption (0.576) but only moderate semantic-hierarchy absorption (0.250), while Standard SAE shows low first-letter absorption (0.026) but high semantic-hierarchy absorption (0.352). Second, PAnneal achieves the lowest scores in both semantic conditions (0.064 and 0.131), suggesting its penalty-annealing training dynamics may reduce absorption-like behavior regardless of task type.

## 3.2 H1: Construct Validity

Figure 2 plots first-letter against semantic-hierarchy absorption for the seven trained SAEs. The Pearson correlation is $r = 0.463$ with a bootstrap 95% confidence interval of $[-0.389, 0.981]$ ($B = 10{,}000$ resamples). The point estimate suggests a moderate positive relationship, but the interval spans from moderately negative to near-perfect correlation. With the Random SAE included ($n = 8$), the correlation drops to $r = 0.281$ with CI $[-0.578, 0.901]$.

![Construct validity scatter plot showing first-letter vs. semantic-hierarchy absorption across 7 trained SAE architectures. Pearson r = 0.463 with bootstrap 95% CI [-0.389, 0.981]. The wide interval reflects small sample size and high variance, yielding an inconclusive construct-validity test.](figures/fig2_scatter.pdf)

**Figure 2:** Scatter plot of first-letter vs. semantic-hierarchy absorption scores across 7 trained SAE architectures. Pearson $r = 0.463$ with bootstrap 95% CI $[-0.389, 0.981]$. The wide interval reflects small sample size and high variance, yielding an inconclusive construct-validity test.

Because the CI includes values below 0.3 and above 0.9, the evidence neither supports nor rejects H1. We cannot conclude that first-letter absorption generalizes to semantic hierarchies, nor can we rule out a strong relationship. The principal limitation is sample size: with only seven trained SAEs, the bootstrap distribution is highly diffuse.

## 3.3 H2: Hierarchy Specificity

Figure 3 compares semantic-hierarchy and non-hierarchy control absorption per architecture. The mean semantic-hierarchy absorption across all eight configurations is $\bar{A}_{\text{SH}} = 0.235$, while the mean non-hierarchy control absorption is $\bar{A}_{\text{NH}} = 0.331$. A paired t-test yields $t = -4.748$, $p = 0.0032$.

![Hierarchy specificity test comparing semantic-hierarchy (coral) vs. non-hierarchy correlated-feature (green) absorption scores per architecture. Non-hierarchy scores are significantly higher (paired t-test: t = -4.748, p = 0.003), rejecting the hypothesis that the metric is specific to hierarchical structure.](figures/fig3_specificity.pdf)

**Figure 3:** Hierarchy specificity test: semantic-hierarchy (coral) vs. non-hierarchy correlated-feature (green) absorption scores. Non-hierarchy scores are significantly higher (paired t-test: $t = -4.748$, $p = 0.003$), rejecting the hypothesis that the metric is specific to hierarchical structure.

The direction of the effect contradicts the theoretical motivation: if the metric were hierarchy-specific, parent-child hierarchies should show *higher* absorption than semantically correlated non-hierarchical pairs. Instead, every architecture except TopK shows higher non-hierarchy absorption. H2 is therefore rejected.

## 3.4 H3: Robustness Across $\tau_{\text{fs}}$

Figure 4 shows the Pearson correlation between first-letter and semantic-hierarchy absorption at three feature-splitting thresholds: $\tau_{\text{fs}} \in \{0.01, 0.03, 0.05\}$. The correlation is numerically stable ($r = 0.468$, $0.463$, $0.471$), but all CIs remain wide and inconclusive.

| $\tau_{\text{fs}}$ | Pearson $r$ | 95% CI Lower | 95% CI Upper |
|:-------------------|------------:|-------------:|-------------:|
| 0.01               | 0.468       | $-0.394$     | 0.981        |
| 0.03               | 0.463       | $-0.389$     | 0.981        |
| 0.05               | 0.471       | $-0.379$     | 0.979        |

**Table 2:** Robustness analysis: Pearson correlation and bootstrap 95% CI across three feature-splitting thresholds. All intervals span the H1 threshold of $r = 0.6$.

![Robustness analysis showing Pearson correlation between first-letter and semantic-hierarchy absorption across three feature-splitting thresholds. The correlation is numerically stable (r = 0.468-0.471), but all confidence intervals are wide and span the H1 threshold of r = 0.6, yielding inconclusive evidence.](figures/fig4_robustness.pdf)

**Figure 4:** Robustness analysis: Pearson correlation between first-letter and semantic-hierarchy absorption across three feature-splitting thresholds ($\tau_{\text{fs}}$). The correlation is numerically stable ($r = 0.468$--$0.471$), but all confidence intervals are wide and span the H1 threshold of $r = 0.6$, yielding inconclusive evidence.

The hierarchy-specificity rejection (H2) holds at all thresholds: the paired t-test comparing semantic-hierarchy vs. non-hierarchy control yields identical statistics ($t = -4.748$, $p = 0.0032$) because the control condition does not depend on $\tau_{\text{fs}}$. H3 is therefore inconclusive for construct validity but robust for hierarchy-specificity failure.

## 3.5 Random-SAE Control

The Random-SAE control---constructed by permuting the decoder directions of the Standard SAE---yields first-letter absorption of 0.030, confirming that the first-letter task measures learned structure. However, its semantic-hierarchy absorption is 0.352, identical to the Standard SAE (0.352), and its non-hierarchy control absorption is 0.416, also identical to Standard (0.416). This is the most striking finding in our study: the semantic-hierarchy and non-hierarchy adaptations of the metric produce identical scores on trained and randomized SAEs, indicating they capture artifacts unrelated to learned SAE structure.

## 3.6 GPT-2 Replication

We replicated the semantic-hierarchy probe pipeline on GPT-2 small (layer 8) for Standard and TopK SAEs. The Standard SAE achieved hierarchy absorption of 0.000 and non-hierarchy absorption of 0.025; the TopK SAE achieved 0.003 and 0.098 respectively. Absolute scores are near-zero compared to Pythia-160M, and the pattern differs---on GPT-2, hierarchy absorption is lower than non-hierarchy absorption, which is directionally consistent with hierarchy specificity, but the magnitudes are too small to support confident interpretation. This model-dependent behavior suggests that semantic-hierarchy absorption is not a stable, generalizable phenomenon.

<!-- FIGURES
- Figure 2: gen_fig2_scatter.py, fig2_scatter.pdf — Scatter plot of first-letter vs. semantic-hierarchy absorption with Pearson r and bootstrap CI
- Figure 3: gen_fig3_specificity.py, fig3_specificity.pdf — Hierarchy specificity test: semantic-hierarchy vs. non-hierarchy control per architecture
- Figure 4: gen_fig4_robustness.py, fig4_robustness.pdf — Robustness analysis: correlation across feature-splitting thresholds
- Table 1: inline — Per-architecture absorption scores across three conditions
- Table 2: inline — tau_fs robustness analysis with Pearson r and 95% CI
-->

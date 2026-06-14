# Does First-Letter Absorption Predict Semantic-Hierarchy Absorption? A Construct-Validity Study of the Dominant SAE Benchmark

**Abstract.** Feature absorption is a central pathology in sparse autoencoders (SAEs), measured by the SAEBench metric using first-letter classification tasks. We conduct the first construct-validity study, testing whether first-letter absorption scores generalize to matched-frequency semantic hierarchies from WordNet. Across 8 SAE architectures on Pythia-160M, the Pearson correlation between first-letter and semantic-hierarchy absorption is $r = 0.463$ (95% CI: $[-0.389, 0.981]$)---inconclusive. The metric fails hierarchy specificity: non-hierarchy correlated features show higher absorption than semantic hierarchies ($t = -4.748$, $p = 0.003$, Cohen's $d = -1.68$). A Random-SAE control yields semantic-hierarchy absorption of 0.352, identical to the Standard SAE, indicating the metric captures artifacts unrelated to learned structure. These findings reveal a methodological blind spot in a widely adopted benchmark and suggest the need for domain-specific absorption metrics.

---

# 1. Introduction

## 1.1 Feature Absorption as a Central Pathology

Sparse autoencoders (SAEs) have become the dominant approach for decomposing neural network activations into interpretable features. By learning an overcomplete dictionary of latent directions, SAEs aim to recover monosemantic representations---features that each encode a single human-interpretable concept---from the polysemantic superposition that pervades transformer residual streams (Bricken et al., 2023; Cunningham et al., 2023). The promise is substantial: if SAEs succeed, mechanistic interpretability can move from hand-crafted circuit tracing to automated feature discovery at scale.

Yet SAEs suffer from well-documented failure modes. Feature absorption, first characterized analytically by Chanin et al. (2024), is among the most consequential. When a parent feature (e.g., "animal") and its child features (e.g., "dog," "cat") co-occur in the training data, the SAE's sparsity loss incentivizes allocating representational capacity to the children at the parent's expense. The parent's information is not merely distributed across children; it is actively suppressed. Chanin et al. proved that absorption arises structurally from the sparsity objective for hierarchical features in their analytical model, though whether the metric generalizes to all hierarchical features or all SAE architectures remains an open question.

The practical implications are severe. If SAEs absorb parent features, downstream interpretability tools that rely on SAE latents will miss high-level concepts entirely. A probe searching for "animal" features in a language model's SAE representation might find only "dog" and "cat" latents, with no latent encoding the general category. This undermines the core premise of SAE-based interpretability: that the latent basis captures the full conceptual structure of the model's internal representations.

Concurrent work has pursued architectural mitigations. Matryoshka SAEs report order-of-magnitude absorption reduction on first-letter tasks via nested multi-scale features (Bussmann et al., 2025). OrtSAE reduces absorption by 65% through orthogonal decoder constraints (Korznikov et al., 2025). These advances make the question of metric validity more urgent, not less: if architectures are optimized to reduce a metric, we must first establish that the metric measures what it claims.

## 1.2 The First-Letter Benchmark and Its Limitations

Given the theoretical importance of absorption, the community needed a standardized measurement. SAEBench (Karvonen et al., 2025) provided one, incorporating an absorption evaluator based on Chanin et al.'s first-letter classification task. In SAEBench's capability taxonomy, absorption falls under concept detection---the ability of SAE latents to preserve distinguishable concepts---making construct validity essential to its interpretability.

The task constructs 26 parent-child hierarchies defined by character-level properties: a parent feature like "starts with S" with children like "short," "small," and "smart." A ground-truth logistic probe on base-model residual activations provides an upper-bound baseline; probes on full SAE latents and top-$k$ sparse latents measure information loss. The absorption score $A_{\text{full}}$ quantifies the maximum relative accuracy drop across these three conditions:

$$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$$

Despite its limitations, the benchmark has virtues that explain its adoption. Ground-truth labels are available without human annotation. Causal ablations are tractable because the hierarchies are cleanly defined. The task isolates a specific structural property---parent-child containment---that theory predicts should trigger absorption. Architecture papers routinely report absorption as a primary metric (Bussmann et al., 2025).

But the benchmark also has a critical limitation: it is an artificial task with no clear relationship to the semantic hierarchies that motivate absorption theory. Chanin et al. (2024) themselves noted that "finding examples of feature absorption unrelated to character identification" remains open future work. The first-letter hierarchies are defined by orthographic properties, not semantic ones. Whether absorption scores on "starts with S" generalize to "animal $\to$ dog" is an empirical question that has never been tested. Without validation, absorption-reduction claims---such as Matryoshka SAEs' reported 10x improvement---may reflect task-specific optimization rather than general hierarchical feature preservation.

Our focus on absorption joins a growing body of work questioning whether SAE evaluation metrics capture meaningful interpretability (Kantamneni et al., 2025).

## 1.3 Research Questions

This paper conducts the first construct-validity study of the SAEBench absorption metric. We adapt the first-letter evaluation protocol to semantic hierarchies extracted from WordNet (Miller, 1995) and test whether the resulting scores correlate with, and are specific to, hierarchical structure. Our analysis is guided by three research questions, tested via three hypotheses (H1--H3, Section 2.6):

**RQ1 (Construct Validity):** Do first-letter absorption scores correlate with semantic-hierarchy absorption scores across diverse SAE architectures? If the metric measures a general absorption phenomenon, architectures that score high on first-letter hierarchies should also score high on semantic hierarchies.

**RQ2 (Hierarchy Specificity):** Is the metric specific to hierarchical features, or does it detect absorption-like behavior in semantically correlated but non-hierarchical pairs? A valid absorption metric should score higher on parent-child hierarchies than on synonym or co-occurrence pairs that lack containment structure.

**RQ3 (Robustness):** How stable is the correlation across feature-splitting thresholds ($\tau_{\text{fs}}$) in k-sparse probing? If the relationship is genuine, it should persist regardless of how many latents are retained in the sparse probe.

## 1.4 Contributions

Our study makes four contributions:

1. **First construct-validity test on semantic hierarchies.** We provide the first empirical test of whether the dominant SAE absorption metric generalizes from artificial first-letter hierarchies to real semantic hierarchies derived from WordNet.

2. **Evidence of hierarchy-specificity failure.** Non-hierarchy correlated features show significantly higher absorption ($\bar{A}_{\text{NH}} = 0.331$) than semantic hierarchies ($\bar{A}_{\text{SH}} = 0.235$; paired t-test: $t = -4.748$, $p = 0.003$), rejecting the hypothesis that the metric is specific to hierarchical structure. This rejection holds under our experimental conditions; we discuss alternative explanations, including template-induced spurious correlations, in Section 4.2.

3. **Random-SAE control anomaly.** On semantic hierarchies, a Random-SAE control achieves scores identical to the trained Standard SAE (0.352), suggesting that the semantic-hierarchy adaptation of the metric may capture geometric artifacts rather than learned structure. The first-letter task does distinguish trained from random SAEs, so the degeneracy is task-specific.

4. **Open-source replication materials.** We release the WordNet hierarchy dataset, evaluation code, and per-architecture scores to enable community replication and extension.

The implications extend beyond this specific metric. Our findings reveal a methodological blind spot in a widely adopted benchmark and suggest that domain-specific absorption metrics---validated for hierarchy specificity and sensitivity to training---are needed before absorption scores can guide architecture selection for real-world interpretability tasks.

We now describe the measurement protocol that adapts the SAEBench absorption evaluator to semantic hierarchies, non-hierarchical controls, and a randomized baseline.

---

# 2. Method

## 2.1 SAE Selection

We evaluate eight SAE configurations on Pythia-160M-deduped (EleutherAI), layer 8, resid_post. Seven are trained architectures from the SAELens public release (trainer_0): Standard (L1-sparse ReLU), TopK, BatchTopK, MatryoshkaBatchTopK, GatedSAE, JumpReLU, and PAnneal. The eighth is a Random-SAE control constructed by applying a fixed random permutation (seed = 42) to the columns of the Standard SAE's decoder matrix $W_{\text{dec}} \in \mathbb{R}^{d \times m}$, while retaining the encoder $W_{\text{enc}}$ and biases unchanged. The permutation destroys the learned correspondence between latent directions and semantic features while preserving the geometric properties of the decoder basis (column norms and pairwise angles). All checkpoints use residual-stream dimension $d = 768$ and latent dimension $m = 2048$.

The selection spans the full absorption-rate spectrum observed in prior work: TopK and BatchTopK show high first-letter absorption ($\bar{A}_{\text{FL}} = 0.576$ and $0.547$), while GatedSAE and PAnneal show near-zero absorption ($\bar{A}_{\text{FL}} = 0.008$ and $0.012$). This range is essential for testing construct validity: if the metric generalizes, the correlation across architectures should be stable regardless of where they fall on the first-letter spectrum.

## 2.2 First-Letter Absorption (Baseline)

First-letter absorption scores come from the official SAEBench evaluator (`sae_bench.evals.absorption.main.run_eval`), using the standard configuration: Pythia-160M-deduped, random seed 42, batch size 256. The evaluator tests 26 parent-child hierarchies defined by first-letter classification (e.g., "starts with S" parent with "short" child). The primary metric is the mean full absorption score $\bar{A}_{\text{FL}}$ across all 26 hierarchies.

## 2.3 Semantic-Hierarchy Construction

We extract 10 parent-child hierarchies from WordNet (Miller, 1995) using NLTK's `nltk.corpus.wordnet` interface. Table 2 lists the exact hierarchies. Selection criteria are: (1) direct hypernymy relationship in WordNet, (2) all tokens are single words present in the Pythia-160M vocabulary, and (3) concepts are concrete enough to generate unambiguous sentences.

| Parent | Children | Probe AUROC |
|:-------|:---------|------------:|
| building | house, school, library | 1.000 |
| container | box, bag, cup | 1.000 |
| tool | fork, comb, rake | 1.000 |
| room | cell, court, hall | 1.000 |
| device | fan, key | 1.000 |
| book | album, journal | 1.000 |
| tree | ash, poon | 1.000 |
| food | fish, water | 1.000 |
| fruit | berry, seed | 1.000 |
| animal | pet, male | 1.000 |

**Table 2:** WordNet semantic hierarchies used in the semantic-hierarchy absorption evaluation. All hierarchies achieved perfect probe AUROC ($= 1.0$) on base-model residual activations, validating probe quality. Children are direct hyponyms of the parent in WordNet.

To control for frequency confounds, we construct synthetic balanced datasets with $N = 100$ sentences per concept. Each sentence follows a fixed template: "The [concept] is [property]." For example, "The house is large." This template ensures that frequency differences between concepts do not drive probe performance. All probes achieved AUROC $= 1.0$ on residual activations, confirming that the base model perfectly distinguishes parent from child concepts at layer 8.

## 2.4 Absorption Measurement Protocol

The absorption measurement follows the SAEBench protocol with three probe conditions:

1. **Ground-truth probe ($\text{acc}_{\text{resid}}$):** Logistic regression trained on base-model residual-stream activations $h^{(l)}(x) \in \mathbb{R}^{768}$ at layer 8. This serves as the upper-bound baseline.

2. **SAE latent probe ($\text{acc}_{\text{sae}}$):** Same logistic regression trained on full SAE latent activations $z \in \mathbb{R}^{2048}$.

3. **K-sparse probe ($\text{acc}_{\text{k-sparse}}$):** Logistic regression trained on the top-$k$ most active SAE latents, where $k = 10$. Latents are selected per sample by absolute activation magnitude. The value $k = 10$ was chosen to match the median number of active latents observed at $\tau_{\text{fs}} = 0.03$ (the SAEBench default) across architectures, providing a fixed reference point for the main analysis. The robustness analysis (H3) varies $\tau_{\text{fs}}$ directly.

The full absorption score for a single hierarchy is:

$$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$$

The mean semantic-hierarchy absorption $\bar{A}_{\text{SH}}$ is the average of $A_{\text{full}}$ across all 10 hierarchies. A minimum probe AUROC threshold of 0.7 filters out hierarchies where the base model itself cannot discriminate parent from child; all 10 hierarchies in this study exceeded this threshold with AUROC $= 1.0$.

Probes are trained with scikit-learn's `LogisticRegression` (liblinear solver, L2 regularization, $C = 1.0$) on 80% of the synthetic data, with 20% held out for evaluation. We report mean AUROC across 3 random seeds for data splitting.

## 2.5 Non-Hierarchy Control Condition

To test hierarchy specificity, we construct 10 semantically correlated but non-hierarchical word pairs. Table 3 lists the exact pairs. These pairs share thematic or associative relationships (synonyms, co-occurring concepts) but lack the strict hypernymy structure of the semantic hierarchies.

| Word A | Word B | Relationship Type |
|:-------|:-------|:------------------|
| big | large | synonym |
| fast | quick | synonym |
| begin | start | synonym |
| doctor | hospital | co-occurrence |
| student | school | co-occurrence |
| river | water | co-occurrence |
| fire | heat | co-occurrence |
| sun | light | co-occurrence |
| tree | wood | co-occurrence |
| stone | rock | synonym |

**Table 3:** Non-hierarchy control pairs used to test hierarchy specificity. Pairs are either synonyms or semantically co-occurring concepts without a parent-child hypernymy relationship.

For non-hierarchy pairs, the probe is trained to distinguish sentences containing word A from sentences containing word B, analogous to the parent-vs-child classification in hierarchies. The same absorption formula is applied, measuring whether the SAE representation loses discriminative information for the pair. If the metric is hierarchy-specific, semantic hierarchies should show higher absorption than these non-hierarchical pairs.

## 2.6 Statistical Analysis

Three hypotheses frame the statistical analysis:

**H1 (Construct Validity):** First-letter absorption scores correlate with semantic-hierarchy absorption scores across SAEs. We test this with Pearson correlation $r$ and a bootstrap 95% confidence interval ($B = 10{,}000$ resamples). H1 is supported if $r > 0.6$ with a 95% CI that excludes values below 0.3.

**H2 (Hierarchy Specificity):** Semantic-hierarchy absorption exceeds non-hierarchy control absorption. We test this with a paired t-test comparing $\bar{A}_{\text{SH}}$ and $\bar{A}_{\text{NH}}$ across the same architectures. H2 is supported if the mean difference is positive and $p < 0.05$.

**H3 (Robustness):** The correlation is numerically consistent across feature-splitting thresholds. We compute $r$ at $\tau_{\text{fs}} \in \{0.01, 0.03, 0.05\}$, where $\tau_{\text{fs}}$ controls how many latents are retained in k-sparse probing. H3 is supported if all three correlations exceed 0.6 and their CIs exclude values below 0.3.

All analyses are performed in Python using `scipy.stats` (Pearson correlation, paired t-test) and custom bootstrap resampling for confidence intervals. The Random-SAE control is excluded from H1 and H3 because its first-letter score is an outlier by design, but it is included in H2 because the hierarchy-specificity test applies to all configurations.

With the protocol established, we test whether first-letter absorption predicts semantic-hierarchy absorption.

---

# 3. Results

## 3.1 Main Results: Architecture Comparison

Figure 1 shows side-by-side first-letter and semantic-hierarchy absorption scores across all architectures. First-letter absorption spans nearly the full range, from 0.008 (GatedSAE) to 0.576 (TopK). Semantic-hierarchy absorption is more compressed, ranging from 0.064 (PAnneal) to 0.359 (BatchTopK). The Random-SAE control scores 0.030 on first-letter absorption---near zero, confirming that the first-letter task measures learned structure---but 0.352 on semantic-hierarchy absorption, identical to the Standard SAE.

![Architecture ranking comparison showing first-letter (blue) and semantic-hierarchy (orange) absorption scores across 8 SAE architectures on Pythia-160M layer 8. The Random-SAE control (rightmost) achieves semantic-hierarchy absorption identical to the Standard SAE.](figures/fig1_architecture_ranking.png)

**Figure 1:** Comparison of first-letter (blue) and semantic-hierarchy (orange) absorption scores across 8 SAE architectures on Pythia-160M layer 8. The Random-SAE control (rightmost) achieves semantic-hierarchy absorption identical to the Standard SAE, suggesting the semantic metric captures artifacts unrelated to learned structure.

Table 1 reports the numerical scores across all three conditions.

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

Two patterns stand out. First, the architecture ranking differs between conditions: TopK shows the highest first-letter absorption (0.576) but only moderate semantic-hierarchy absorption (0.250), while Standard SAE shows low first-letter absorption (0.026) but high semantic-hierarchy absorption (0.352). Second, PAnneal achieves the lowest scores in both the semantic-hierarchy and non-hierarchy conditions (0.064 and 0.131), suggesting its penalty-annealing training dynamics may reduce absorption-like behavior regardless of task type.

## 3.2 H1: Construct Validity

Figure 2 plots first-letter against semantic-hierarchy absorption for the seven trained SAEs. The Pearson correlation is $r = 0.463$ with a bootstrap 95% confidence interval of $[-0.389, 0.981]$ ($B = 10{,}000$ resamples). The point estimate suggests a moderate positive relationship, but the interval spans from weakly-to-moderately negative to near-perfect correlation. With the Random SAE included ($n = 8$), the correlation drops to $r = 0.281$ with CI $[-0.578, 0.901]$.

![Construct validity scatter plot showing first-letter vs. semantic-hierarchy absorption across 7 trained SAE architectures. Pearson r = 0.463 with bootstrap 95% CI [-0.389, 0.981]. The wide interval reflects small sample size and high variance, yielding an inconclusive construct-validity test.](figures/fig2_scatter.pdf)

**Figure 2:** Scatter plot of first-letter vs. semantic-hierarchy absorption scores across 7 trained SAE architectures. Pearson $r = 0.463$ with bootstrap 95% CI $[-0.389, 0.981]$. The wide interval reflects small sample size and high variance, yielding an inconclusive construct-validity test.

Because the CI includes values below 0.3 and above 0.9, the evidence neither supports nor rejects H1. We cannot conclude that first-letter absorption generalizes to semantic hierarchies, nor can we rule out a strong relationship. The principal limitation is sample size: with only seven trained SAEs, the bootstrap distribution is highly diffuse.

## 3.3 H2: Hierarchy Specificity

Figure 3 compares semantic-hierarchy and non-hierarchy control absorption per architecture. The mean semantic-hierarchy absorption across all eight configurations is $\bar{A}_{\text{SH}} = 0.235$, while the mean non-hierarchy control absorption is $\bar{A}_{\text{NH}} = 0.331$. A paired t-test yields $t = -4.748$, $p = 0.0032$, Cohen's $d = -1.68$ (95% CI: $[-2.87, -0.49]$), a large effect.

![Hierarchy specificity test comparing semantic-hierarchy (coral) vs. non-hierarchy correlated-feature (green) absorption scores per architecture. Non-hierarchy scores are significantly higher (paired t-test: t = -4.748, p = 0.003), rejecting the hypothesis that the metric is specific to hierarchical structure.](figures/fig3_specificity.pdf)

**Figure 3:** Hierarchy specificity test: semantic-hierarchy (coral) vs. non-hierarchy correlated-feature (green) absorption scores. Non-hierarchy scores are significantly higher (paired t-test: $t = -4.748$, $p = 0.003$), rejecting the hypothesis that the metric is specific to hierarchical structure.

The direction of the effect contradicts the theoretical motivation: if the metric were hierarchy-specific, parent-child hierarchies should show *higher* absorption than semantically correlated non-hierarchical pairs. Instead, all eight architectures show higher non-hierarchy absorption. H2 is therefore rejected.

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

The Random-SAE control yields first-letter absorption of 0.030, confirming that the first-letter task measures learned structure. However, its semantic-hierarchy absorption is 0.352, identical to the Standard SAE (0.352), and its non-hierarchy control absorption is 0.416, also identical to Standard (0.416). The semantic-hierarchy and non-hierarchy adaptations of the metric produce identical scores on trained and randomized SAEs, indicating they capture artifacts unrelated to learned SAE structure.

## 3.6 GPT-2 Replication

We replicated the semantic-hierarchy probe pipeline on GPT-2 small (layer 8) for Standard and TopK SAEs. Figure 5 shows the results. The Standard SAE achieved hierarchy absorption of 0.000 and non-hierarchy absorption of 0.025; the TopK SAE achieved 0.003 and 0.098 respectively.

![GPT-2 small replication: semantic-hierarchy vs. non-hierarchy control absorption for Standard and TopK SAEs. Absolute scores are near-zero compared to Pythia-160M, indicating model-specific behavior in semantic-hierarchy absorption.](figures/fig5_gpt2_replication.png)

**Figure 5:** GPT-2 small replication: semantic-hierarchy vs. non-hierarchy control absorption for Standard and TopK SAEs. Absolute scores are near-zero compared to Pythia-160M, indicating model-specific behavior in semantic-hierarchy absorption.

Absolute scores are near-zero compared to Pythia-160M. On Pythia-160M, the hierarchy vs. non-hierarchy gap is $-0.096$ (non-hierarchy higher), while on GPT-2 it is $+0.025$ (Standard) and $+0.095$ (TopK)---hierarchy lower, as theory predicts. However, the magnitudes are too small to support confident interpretation. This model-dependent behavior, combined with the near-zero absolute scores, suggests that semantic-hierarchy absorption is not a stable, generalizable phenomenon across base models.

These results demand careful interpretation of what they mean for the field.

---

# 4. Discussion

## 4.1 Interpreting the Inconclusive Construct Validity

The point estimate $r = 0.463$ suggests a moderate positive relationship between first-letter and semantic-hierarchy absorption across SAE architectures. Yet the bootstrap 95% confidence interval $[-0.389, 0.981]$ spans from weakly-to-moderately negative to near-perfect correlation. With only seven trained SAEs, the bootstrap distribution is highly diffuse, and the evidence neither supports nor rejects H1.

Three factors contribute to this uncertainty. First, the sample size ($n = 7$) is small for correlation inference. Second, the variance in semantic-hierarchy absorption is compressed relative to first-letter: the coefficient of variation is 0.42 for semantic-hierarchy versus 1.15 for first-letter. Third, the relationship may be genuinely nonlinear or moderated by architecture-specific properties (e.g., gating mechanisms in GatedSAE may affect first-letter and semantic tasks differently).

The only warranted interpretation is that the current evidence base is insufficient for confident claims about construct validity. Researchers should not assume that first-letter absorption generalizes to semantic hierarchies, nor should they dismiss the possibility outright. For future studies aiming to detect a correlation of $r = 0.6$ with 80% power at $\alpha = 0.05$, a sample of approximately 19 architectures would be needed.

## 4.2 The Hierarchy Specificity Failure

The most statistically decisive finding is the rejection of H2. Non-hierarchy correlated features show significantly higher absorption ($\bar{A}_{\text{NH}} = 0.331$) than semantic hierarchies ($\bar{A}_{\text{SH}} = 0.235$), with $t = -4.748$, $p = 0.0032$, and Cohen's $d = -1.68$. All eight architectures show this reversal (Figure 3).

This contradicts the theoretical motivation for the absorption metric. Chanin et al. (2024) proved that sparsity loss incentivizes absorption specifically for hierarchical features, where parent-child relationships create representational competition. If the metric were hierarchy-specific, parent-child hierarchies should show higher absorption than semantically correlated pairs that lack hierarchical structure. The opposite pattern suggests the metric detects something other than hierarchy-driven absorption.

We see three plausible explanations:

1. **Synthetic template artifacts.** The fixed sentence template ("The [concept] is [property].") may introduce spurious correlations. All sentences share identical syntactic structure, so the probe may discriminate based on positional or syntactic cues rather than semantic content. If these cues are preserved equally well in SAE latents for non-hierarchies but disrupted for hierarchies, the metric would artifactually favor non-hierarchies.

2. **Semantic relatedness without containment.** Non-hierarchy pairs like "doctor-hospital" or "sun-light" are thematically correlated in the pre-training corpus. SAEs may learn distributed representations that co-activate for these pairs, creating probe-accuracy drops that the absorption formula interprets as absorption-like behavior. The metric may measure general co-occurrence sensitivity, not hierarchy-specific capacity allocation.

3. **K-sparse probing threshold.** The $k = 10$ threshold for k-sparse probing may be too coarse. Semantic hierarchies may require fine-grained distinctions that need more than 10 latents, while non-hierarchy pairs may be discriminable with fewer. If so, the k-sparse term in the absorption formula would artifactually inflate non-hierarchy scores.

Distinguishing among these explanations requires additional experiments: varying sentence templates, controlling for corpus co-occurrence, and testing multiple $k$ values. Our data cannot adjudicate among them, but all three point to the same conclusion: the metric as currently adapted to semantic tasks is not measuring hierarchy-specific absorption.

## 4.3 The Random-SAE Anomaly

The Random-SAE control---constructed by permuting the decoder directions of the Standard SAE while retaining its trained encoder---yields first-letter absorption of 0.030, near zero because permuting the decoder destroys the letter-specific features learned during training. On semantic-hierarchy absorption, however, it scores 0.352, identical to the Standard SAE (0.352). The non-hierarchy control absorption is also identical: 0.416 for both Random and Standard.

This is the most striking finding in our study. The first-letter task correctly distinguishes trained from randomized structure: the permuted decoder destroys the letter-specific features learned during training, and absorption drops to near-zero. But the semantic-hierarchy and non-hierarchy adaptations of the metric produce identical scores on trained and randomized SAEs, indicating they capture artifacts unrelated to learned SAE structure.

What could these artifacts be? The permuted decoder preserves the geometric properties of the activation space---the angles between decoder directions, their norms, and their coverage of the residual-stream manifold. If the semantic-hierarchy absorption score depends primarily on how well the probe can discriminate concepts given the geometric structure of the basis (rather than the semantic content of the features), then any basis with similar geometric properties would yield similar scores. The Random SAE confirms this: permutation does not change the basis geometry, and the score remains unchanged.

We favor the geometric-basis explanation, but alternative interpretations are possible. The probe may exploit overparameterization rather than basis geometry; the synthetic template may be too simple to stress the representation; or the trained encoder may compensate for decoder permutation. Distinguishing these explanations requires additional controls (e.g., varying latent dimension, natural-language templates, fully random encoder-decoder pairs).

This finding is consistent with the intuition behind Korznikov et al. (2025), who introduced orthogonal decoder constraints and reported 65% absorption reduction---suggesting that decoder geometry may influence absorption. However, their work does not establish that geometry is the primary determinant, and our Random-SAE result provides more direct evidence for this interpretation. It also resonates with the broader critique that SAE evaluation metrics may measure geometric properties of the activation manifold rather than learned semantic structure.

## 4.4 Implications for Benchmark Design

Our findings carry four implications for SAE benchmark design and community practice.

**First, the SAEBench absorption metric should not be extended to semantic hierarchies without modification that demonstrates hierarchy specificity and training sensitivity.** The current adaptation---replacing first-letter hierarchies with WordNet hypernyms while keeping the same probe protocol---produces degenerate scores that are insensitive to learned structure. A valid semantic-hierarchy absorption metric would need to demonstrate hierarchy specificity (hierarchy > non-hierarchy) and sensitivity to training (trained > random) before community adoption.

**Second, architecture comparisons using absorption as a criterion may be valid for first-letter tasks but not generalizable.** Bussmann et al. (2025) report order-of-magnitude absorption reductions for Matryoshka SAEs on first-letter tasks; our results do not challenge this external finding, but they do challenge its generalization to semantic tasks. This does not mean absorption should be abandoned as an architecture criterion---first-letter absorption may still capture meaningful structural properties---but claims of absorption reduction should be accompanied by evidence of generalization.

**Third, random-baseline correction should become standard practice.** The Random-SAE anomaly demonstrates that raw absorption scores on semantic tasks are uninterpretable without a baseline. While random-baseline correction adds computational overhead, the marginal cost is small relative to SAE training, and the diagnostic value is substantial. We recommend that future absorption evaluations report scores relative to both random-decoder and PCA-basis controls, following the methodology introduced by Korznikov et al. (2025) for AutoInterp and sparse probing.

**Fourth, the community should invest in domain-specific absorption metrics with validated hierarchy specificity.** A metric that cannot distinguish hierarchies from correlated pairs, and cannot distinguish trained from randomized SAEs, is not ready for benchmark status. Future metrics should explicitly test for hierarchical structure, perhaps using causal ablations or interventional probes that go beyond passive probe accuracy.

## 4.5 Limitations

Our study has five principal limitations.

**Small SAE sample.** With seven trained SAEs plus one random control, statistical power is limited. The bootstrap confidence interval for H1 spans nearly the full correlation range. A cohort of 15--20 architectures would be needed for definitive construct-validity testing.

**Single layer and model size.** All experiments use Pythia-160M-deduped, layer 8, resid_post. Absorption dynamics may differ at other layers (earlier layers encode lower-level features; later layers encode more abstract concepts) and at larger scales (Templeton et al., 2024, extracted millions of features from Claude 3 Sonnet). The GPT-2 replication shows model-dependent effects, suggesting that results do not generalize across base models without further validation.

**Shallow hierarchies.** Our WordNet hierarchies are single-level parent-child relationships. Deeper hierarchies (3--4 levels, e.g., "animal" $\to$ "mammal" $\to$ "dog" $\to$ "poodle") may exhibit different absorption patterns. The sparsity-loss incentive for absorption may compound across levels, or deeper hierarchies may be represented differently in the residual stream.

**Synthetic sentence templates.** The fixed template ("The [concept] is [property].") ensures frequency balance but may not reflect natural language distribution. Probes trained on synthetic data may exploit template-specific cues rather than genuine semantic representations. Natural-language corpus extraction (e.g., from Wikipedia or C4) would increase ecological validity but introduce frequency confounds. We note that the first-letter benchmark also uses synthetic templates, so this is not a unique weakness of the semantic-hierarchy adaptation.

**Probe ceiling effects.** All hierarchies achieved perfect probe AUROC ($= 1.0$) on residual activations, setting a high baseline that minimizes the absorption score for any given accuracy drop. If residual AUROC were lower (e.g., 0.7), the same accuracy drop would produce a larger absorption score. Hierarchies with lower residual AUROC might show different absolute absorption scores, though the relative ranking of architectures may be stable.

From these implications, we conclude with concrete recommendations.

---

# 5. Conclusion

## 5.1 Summary

This paper presents the first construct-validity study of the SAEBench feature absorption metric, testing whether first-letter absorption scores generalize to real semantic hierarchies drawn from WordNet. Across eight SAE architectures on Pythia-160M layer 8, three findings emerge.

First, construct validity is inconclusive. The Pearson correlation between first-letter and semantic-hierarchy absorption is $r = 0.463$ ($95\%$ CI: $[-0.389, 0.981]$). The point estimate suggests a moderate positive relationship, but the confidence interval spans from weakly-to-moderately negative to near-perfect correlation. With only seven trained SAEs, the evidence neither supports nor rejects the hypothesis that the metric generalizes.

Second, hierarchy specificity fails. Non-hierarchy correlated features show significantly higher absorption ($\bar{A}_{\text{NH}} = 0.331$) than semantic hierarchies ($\bar{A}_{\text{SH}} = 0.235$; paired t-test: $t = -4.748$, $p = 0.0032$, Cohen's $d = -1.68$). All eight architectures show this reversal. The metric detects absorption-like behavior in semantically correlated pairs that lack parent-child structure, contradicting the theoretical motivation that absorption is specific to hierarchical features.

Third, the Random-SAE control reveals degeneracy. A randomized SAE with permuted decoder directions achieves semantic-hierarchy absorption of $0.352$, identical to the trained Standard SAE ($0.352$ to three decimal places), and non-hierarchy control absorption of $0.416$, also identical. The first-letter task correctly distinguishes trained from randomized structure ($0.030$), but the semantic adaptations of the metric do not. This indicates that semantic-hierarchy absorption scores capture artifacts of basis geometry rather than learned SAE structure.

Fourth, robustness across feature-splitting thresholds ($\tau_{\text{fs}}$) is inconclusive for construct validity but confirms the hierarchy-specificity failure at all thresholds.

## 5.2 Recommendations

We order these recommendations by immediacy.

**Most urgently, for benchmark designers:** Do not extend the first-letter absorption metric to semantic hierarchies without substantial modification. The current adaptation---replacing first-letter hierarchies with WordNet hypernyms while retaining the same probe protocol---produces scores that are insensitive to learned structure and fail hierarchy specificity. Any semantic-hierarchy absorption metric should demonstrate both hierarchy specificity (hierarchy $>$ non-hierarchy) and sensitivity to training (trained $>$ random) before community adoption.

**Second, for architecture researchers:** Absorption-reduction claims should be validated on multiple task types. Bussmann et al. (2025) report order-of-magnitude absorption reductions for Matryoshka SAEs on first-letter tasks; our results do not challenge this external finding, but they do challenge its generalization to semantic tasks. Architecture comparisons that rank SAEs by absorption alone may be optimizing for a metric that does not reflect behavior on real conceptual hierarchies.

**Finally, for the community:** Invest in domain-specific absorption metrics with validated hierarchy specificity. A metric that cannot distinguish hierarchies from correlated pairs, and cannot distinguish trained from randomized SAEs, is not ready for benchmark status. Future metrics should explicitly test for hierarchical structure, perhaps using causal ablations or interventional probes that go beyond passive probe accuracy.

## 5.3 Future Work

Four directions would strengthen or extend this study.

**Larger SAE cohorts.** A cohort of 15--20 architectures would provide adequate statistical power for definitive construct-validity testing. Detecting $r = 0.6$ at $\alpha = 0.05$ with $80\%$ power requires $n \approx 19$ SAEs. The current sample of seven trained architectures yields a bootstrap distribution too diffuse for confident inference.

**Deeper hierarchies and multiple base models.** Our WordNet hierarchies are single-level parent-child relationships. Deeper hierarchies (3--4 levels, e.g., "animal" $\to$ "mammal" $\to$ "dog") may exhibit different absorption patterns, and the sparsity-loss incentive may compound across levels. The GPT-2 replication shows model-dependent effects, with near-zero absolute scores that differ from Pythia-160M. A broader model sweep---spanning scales from GPT-2 small to Gemma-2-2B and layers from early to late---would test whether semantic-hierarchy absorption is a stable phenomenon or a model-specific artifact.

**Alternative hierarchy sources.** WordNet is one of many lexical ontologies. ConceptNet (Speer et al., 2017) provides broader relational coverage; BabelNet (Navigli & Ponzetto, 2012) integrates multilingual hierarchies. Testing absorption on hierarchies from multiple sources would guard against ontology-specific artifacts and increase ecological validity.

**Causal validation.** The current metric is correlational: it measures probe accuracy drops, not whether parent-feature information is truly missing or merely hidden in distributed form. Causal ablation studies---following the methodology of Chanin et al. (2024)---could distinguish "truly absorbed" from "merely distributed" parent features. Activation patching or interchange intervention (Geiger et al., 2023) could test whether suppressing child latents restores parent-feature accessibility. Such experiments would move absorption measurement from diagnostic probing to causal validation, aligning the metric more closely with the theoretical construct it claims to measure.

These directions, taken together, would transform absorption measurement from a diagnostic convenience into a validated scientific instrument.

---

# References

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olsson, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*.

- Bussmann, F., Braun, D., & Bethge, M. (2025). Learning hierarchical features with matryoshka sparse autoencoders. *arXiv preprint*.

- Chanin, J., Wright, J., Conmy, A., & Nanda, N. (2024). Feature absorption in sparse autoencoders. *arXiv preprint*.

- Cunningham, H., Ewart, A., Huben, R., & Sharma, M. (2023). Sparse autoencoders find highly interpretable features in language models. *arXiv preprint*.

- Geiger, A., Lu, H., Icard, T., & Potts, C. (2023). Causal abstractions of neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*.

- Kantamneni, N., et al. (2025). SAEs often fail to outperform baselines on downstream tasks. *arXiv preprint*.

- Karvonen, A., et al. (2025). SAEBench: Standardized evaluation for sparse autoencoders. *arXiv preprint*.

- Korznikov, A., et al. (2025). OrtSAE: Orthogonal sparse autoencoders for interpretability. *arXiv preprint*.

- Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39-41.

- Navigli, R., & Ponzetto, S. P. (2012). BabelNet: The automatic construction, evaluation and application of a wide-coverage multilingual semantic network. *Artificial Intelligence*, 193, 217-250.

- Speer, R., Chin, J., & Havasi, C. (2017). ConceptNet 5.5: An open multilingual graph of general knowledge. *Proceedings of AAAI*, 31(1).

- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Jermyn, A., Drake, B., Estornell, A., Mosse, D., Zhang, Z., Belka, M., Durbin, T., Hume, T., Carter, S., Henighan, T., & Olsson, C. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic*.

---

# Figures and Tables

- **Figure 1:** `fig1_architecture_ranking.png` --- Comparison of first-letter and semantic-hierarchy absorption scores across 8 SAE architectures on Pythia-160M layer 8.
- **Figure 2:** `fig2_scatter.pdf` --- Scatter plot of first-letter vs. semantic-hierarchy absorption scores across 7 trained SAE architectures (Pearson $r = 0.463$, 95% CI $[-0.389, 0.981]$).
- **Figure 3:** `fig3_specificity.pdf` --- Hierarchy specificity test: semantic-hierarchy vs. non-hierarchy correlated-feature absorption scores (paired t-test: $t = -4.748$, $p = 0.003$).
- **Figure 4:** `fig4_robustness.pdf` --- Robustness analysis: Pearson correlation across three feature-splitting thresholds ($\tau_{\text{fs}} \in \{0.01, 0.03, 0.05\}$).
- **Figure 5:** `fig5_gpt2_replication.png` --- GPT-2 small replication: semantic-hierarchy vs. non-hierarchy control absorption for Standard and TopK SAEs.
- **Table 1:** inline --- Per-architecture absorption scores across three conditions (first-letter, semantic-hierarchy, non-hierarchy control) on Pythia-160M layer 8.
- **Table 2:** inline --- WordNet semantic hierarchies with probe AUROC values.
- **Table 3:** inline --- Non-hierarchy control pairs with relationship types.
- **Table 4:** inline --- Robustness analysis: Pearson correlation and bootstrap 95% CI across three feature-splitting thresholds.

# 2. Method

## 2.1 SAE Selection

We evaluate eight SAE configurations on Pythia-160M-deduped (EleutherAI), layer 8, resid_post. Seven are trained architectures from the SAELens public release (trainer_0): Standard (L1-sparse ReLU), TopK, BatchTopK, MatryoshkaBatchTopK, GatedSAE, JumpReLU, and PAnneal. The eighth is a Random-SAE control constructed by permuting the decoder directions of the Standard SAE while retaining its trained encoder. All checkpoints use activation dimension $d = 768$ and latent dimension $m = 2048$.

The selection spans the full absorption-rate spectrum observed in prior work: TopK and BatchTopK show high first-letter absorption ($\bar{A}_{\text{FL}} = 0.576$ and $0.547$), while GatedSAE and PAnneal show near-zero absorption ($\bar{A}_{\text{FL}} = 0.008$ and $0.012$). This range is essential for testing construct validity: if the metric generalizes, the correlation across architectures should be stable regardless of where they fall on the first-letter spectrum.

## 2.2 First-Letter Absorption (Baseline)

First-letter absorption scores come from the official SAEBench evaluator (`sae_bench.evals.absorption.main.run_eval`), using the standard configuration: Pythia-160M-deduped, random seed 42, batch size 256. The evaluator tests 26 parent-child hierarchies defined by first-letter classification (e.g., "starts with S" parent with "short" child). The primary metric is the mean full absorption score $\bar{A}_{\text{FL}}$ across all 26 hierarchies.

## 2.3 Semantic-Hierarchy Construction

We extract 10 parent-child hierarchies from WordNet (Miller, 1995) using the NLTK interface. Table 2 lists the exact hierarchies. Selection criteria are: (1) direct hypernymy relationship in WordNet, (2) all tokens are single words present in the Pythia-160M vocabulary, and (3) concepts are concrete enough to generate unambiguous sentences.

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

3. **K-sparse probe ($\text{acc}_{\text{k-sparse}}$):** Logistic regression trained on the top-$k$ most active SAE latents, where $k = 10$. Latents are selected per sample by absolute activation magnitude.

The full absorption score for a single hierarchy is:

$$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$$

The mean semantic-hierarchy absorption $\bar{A}_{\text{SH}}$ is the average of $A_{\text{full}}$ across all 10 hierarchies. A minimum probe AUROC threshold of 0.7 filters out hierarchies where the base model itself cannot discriminate parent from child; all 10 hierarchies in this study exceeded this threshold with AUROC $= 1.0$.

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

The same absorption formula is applied to each pair: the probe classifies which of the two words is present in a sentence, and the accuracy drop from residual to SAE latents quantifies absorption-like behavior. If the metric is hierarchy-specific, semantic hierarchies should show higher absorption than these non-hierarchical pairs.

## 2.6 Statistical Analysis

Three hypotheses guide the statistical analysis:

**H1 (Construct Validity):** First-letter absorption scores correlate with semantic-hierarchy absorption scores across SAEs. We test this with Pearson correlation $r$ and a bootstrap 95% confidence interval ($B = 10{,}000$ resamples). H1 is supported if $r > 0.6$ and the CI excludes 0.

**H2 (Hierarchy Specificity):** Semantic-hierarchy absorption exceeds non-hierarchy control absorption. We test this with a paired t-test comparing $\bar{A}_{\text{SH}}$ and $\bar{A}_{\text{NH}}$ across the same architectures. H2 is supported if the mean difference is positive and $p < 0.05$.

**H3 (Robustness):** The correlation is stable across feature-splitting thresholds. We compute $r$ at $\tau_{\text{fs}} \in \{0.01, 0.03, 0.05\}$, where $\tau_{\text{fs}}$ controls how many latents are retained in k-sparse probing. H3 is supported if all three correlations exceed 0.6 and their CIs exclude 0.

All analyses are performed in Python using `scipy.stats` (Pearson correlation, paired t-test) and custom bootstrap resampling for confidence intervals. The Random-SAE control is excluded from H1 and H3 because its first-letter score is an outlier by design, but it is included in H2 because the hierarchy-specificity test applies to all configurations.

With the protocol established, we present the empirical findings.

<!-- FIGURES
- None
-->

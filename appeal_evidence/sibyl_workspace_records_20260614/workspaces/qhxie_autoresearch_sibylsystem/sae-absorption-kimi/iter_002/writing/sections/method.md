# 2 Method

## 2.1 SAE Selection

We evaluate eight sparse autoencoder (SAE) configurations that span the absorption-rate spectrum, drawn from the SAELens public release for Pythia-160M (Biderman et al., 2023). All SAEs target the `resid_post` activation at layer 8, where the residual-stream dimension is $d = 768$ and the SAE latent dimension is $m = 2048$. Table 1 lists the architectures.

| Architecture | Sparsity Mechanism | Expected Absorption |
|-------------|-------------------|---------------------|
| MatryoshkaBatchTopK | Nested multi-scale Top-K (Bussmann et al., 2025) | Low |
| PAnneal | Penalty annealing with gradual L1 reduction | Low-Moderate |
| GatedSAE | Gated magnitude-direction separation | Low-Moderate |
| Standard | ReLU with L1 sparsity penalty (baseline) | Moderate |
| JumpReLU | Jumping-threshold ReLU (Rajamanoharan et al., 2024) | Moderate |
| BatchTopK | Batch-wise Top-K gating | Moderate-High |
| TopK | Per-sample Top-K hard threshold | High |
| Random-SAE | Permuted decoder directions from Standard | Near-zero (control) |

The Random-SAE control permutes the decoder matrix $W_{\text{dec}}$ of the Standard SAE, destroying any learned structure while preserving the marginal activation statistics. If the absorption metric on semantic tasks is sensitive to learned structure, the Random-SAE should score near-zero; if it scores comparably to trained SAEs, the metric captures artifacts unrelated to training.

## 2.2 First-Letter Absorption (Baseline)

First-letter absorption scores are computed using the official SAEBench evaluator (`sae_bench.evals.absorption.main.run_eval`) with the following configuration: model `EleutherAI/pythia-160m-deduped`, random seed 42, batch size 256, and dtype `float32`. The primary metric is `mean_full_absorption_score`, which averages the full absorption score $A_{\text{full}}$ across all first-letter parent-child hierarchies in the SAEBench suite.

The SAEBench protocol trains logistic-regression ground-truth probes on base-model residual-stream activations $h^{(l)}(x)$ for first-letter classification tasks (e.g., "starts with S" vs. "short"), then applies the absorption formula to SAE latents using k-sparse probing with feature-splitting threshold $\tau_{\text{fs}} = 0.03$.

## 2.3 Semantic-Hierarchy Construction

We construct ten semantic hierarchies from WordNet (Miller, 1995) where each parent is a direct hypernym of its children. All parent and child tokens are verified as single tokens in both the GPT-2 and Pythia tokenizers. Table 2 documents the exact hierarchies.

| Parent | Children | Probe AUROC |
|--------|----------|-------------|
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

**Selection criteria.** Each hierarchy satisfies four constraints: (i) the parent is a direct or near-direct hypernym of every child (maximum two steps in WordNet); (ii) all tokens are single tokens in the model vocabulary; (iii) concepts are concrete and semantically distinct; (iv) each parent has at least two children to enable meaningful probe training. All hierarchies achieved perfect probe AUROC ($1.0$), confirming that the ground-truth probes can reliably discriminate parent from child concepts in the base-model activations.

**Frequency matching.** To control the frequency confound identified by Chanin et al. (2024), we construct a synthetic balanced dataset where parent and child tokens appear with equal frequency. For each concept, we generate $N = 100$ sentences using simple templates (e.g., "The {concept} is a place with rich history."). Parent and child samples are exactly balanced within each hierarchy, ensuring that any observed absorption is not driven by frequency imbalance.

## 2.4 Absorption Measurement Protocol

We replicate the exact SAEBench absorption logic on our WordNet hierarchies, measuring information loss at three representation levels.

**Ground-truth probe.** For each parent concept $p \in \mathcal{H}$, we train a logistic regression probe on Pythia-160M residual-stream activations $h^{(8)}(x)$ at layer 8 (`resid_post`) to classify "parent vs. child." The probe input is the mean-pooled residual activation over non-padding positions. Training uses 200 Adam epochs with learning rate $0.05$. The probe accuracy on residual activations, $\text{acc}_{\text{resid}}$, serves as the upper-bound baseline.

**SAE latent probe.** The same probe is trained on the full SAE latent vector $z = \phi(h^{(8)}(x))$, yielding accuracy $\text{acc}_{\text{sae}}$.

**K-sparse probe.** The probe is trained on the top-$k$ sparse latent vector $z^{(k)}$, where only the $k = 10$ largest entries are retained and all others are zeroed. This tests whether the most relevant features alone suffice for the classification task. The accuracy is $\text{acc}_{\text{k-sparse}}$.

**Absorption score.** The full absorption score for a single hierarchy is:

$$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$$

The score ranges from $0$ (no absorption: SAE latents preserve all parent-feature information) to $1$ (complete absorption: parent-feature information is fully lost). The mean semantic-hierarchy absorption $\bar{A}_{\text{SH}}$ is the average of $A_{\text{full}}$ across all ten hierarchies.

A minimum probe AUROC threshold of $0.7$ is enforced; all hierarchies in our suite achieved AUROC $= 1.0$, so none were excluded.

## 2.5 Non-Hierarchy Control Condition

To test whether the absorption metric is specific to hierarchical structure, we compute absorption scores on ten semantically correlated but non-hierarchical word pairs. Table 3 lists the pairs.

| Pair | Relationship Type |
|------|-------------------|
| big -- large | Synonym |
| fast -- quick | Synonym |
| begin -- start | Synonym |
| doctor -- hospital | Co-occurrence |
| student -- school | Co-occurrence |
| river -- water | Co-occurrence |
| fire -- heat | Attribute |
| sun -- light | Attribute |
| tree -- wood | Material |
| stone -- rock | Synonym |

For each pair, we train a binary probe (word A vs. word B) and apply the identical absorption formula. If the metric is hierarchy-specific, these non-hierarchy scores should be near-zero and significantly lower than semantic-hierarchy scores. The mean non-hierarchy control absorption is denoted $\bar{A}_{\text{NH}}$.

## 2.6 Statistical Analysis

We test three hypotheses with pre-registered criteria:

**H1 (Construct Validity).** The Pearson correlation $r$ between first-letter absorption $\bar{A}_{\text{FL}}$ and semantic-hierarchy absorption $\bar{A}_{\text{SH}}$ across the $n = 7$ trained SAEs (excluding Random-SAE) is computed with a bootstrap 95% confidence interval ($B = 10{,}000$ resamples). H1 is supported if $r > 0.6$ and the CI excludes $0$; rejected if the CI includes values $< 0.3$.

**H2 (Hierarchy Specificity).** A paired t-test compares semantic-hierarchy absorption $\bar{A}_{\text{SH}}$ to non-hierarchy control absorption $\bar{A}_{\text{NH}}$ across the same SAEs. H2 is supported if $\bar{A}_{\text{SH}} > \bar{A}_{\text{NH}}$ and $p < 0.05$.

**H3 (Robustness).** The Pearson correlation is recomputed at $\tau_{\text{fs}} \in \{0.01, 0.03, 0.05\}$. H3 is supported if $r$ remains positive and exceeds $0.5$ across all thresholds.

All statistical tests are performed on the per-SAE mean scores, treating each architecture as an independent observation. The small sample size ($n = 7$ trained SAEs) is acknowledged as a limitation; bootstrap CIs are reported to quantify uncertainty.

<!-- FIGURES
- Table 1: inline — SAE architecture selection with sparsity mechanisms and expected absorption levels
- Table 2: inline — WordNet semantic hierarchies with parent, children, and probe AUROC
- Table 3: inline — Non-hierarchy control word pairs with relationship types
- Figure 1: fig_method_pipeline_desc.md — Method pipeline architecture diagram
-->

# 3 Methods

## 3.1 Experimental Setup

All experiments use GPT-2 Small (124M parameters) [Radford et al., 2019] with the gpt2-small-res-jb SAE pretrained via SAELens [Templeton et al., 2024]. The SAE has a dictionary size of $d_{\text{sae}} = 24{,}576$ features mapping from the residual stream at layer 8 ($d_{\text{model}} = 768$). We use OpenWebText [Gokaslan and Cohen, 2019] as our corpus, sampling 1,000 sequences with a maximum length of 128 tokens. All experiments use seed 42 for reproducibility. Code and data are available in the supplementary materials.

## 3.2 Ground Truth Definition

We define ground truth absorption using manually constructed concept hierarchies:

**Number hierarchy:** The digits "one" through "eight" form a natural hierarchy where broader number concepts (e.g., "one through four") absorb more specific ones (e.g., "one," "two"). We analyze all $\binom{8}{2} = 28$ pairs.

**Punctuation hierarchy:** Punctuation marks (period, comma, exclamation, question, semicolon, colon, quote, apostrophe) form a flat hierarchy where the general concept "punctuation" absorbs specific marks. We analyze all 28 pairs.

**Case hierarchy (control):** Uppercase and lowercase letter pairs (a/A, b/B, ..., z/Z) serve as a control condition where no absorption is expected.

For each concept, we identify its **absorption feature set** $A(c)$ as the set of SAE features with top-$K$ mean activation on tokens belonging to $c$. The **true absorption rate** between concepts $c_i$ and $c_j$ is:
$$R_{\text{abs}}(c_i, c_j) = \frac{|A(c_i) \cap A(c_j)|}{|A(c_i) \cup A(c_j)|}$$

We set $K = 10$ based on pilot analysis showing that top-10 features capture the dominant activation pattern while avoiding noise from tail features. Seven true absorption pairs were identified in the number hierarchy (where one number concept's features overlap with another's in a parent-child relationship).

## 3.3 UAD Pipeline

We implement UAD following the original specification [Chanin et al., 2024]:

1. **Feature selection:** Select the 500 features with highest mean activation across the corpus.
2. **Co-occurrence matrix:** For each feature pair $(f_i, f_j)$, compute the phi coefficient:
   $$\phi(f_i, f_j) = \frac{n_{11}n_{00} - n_{10}n_{01}}{\sqrt{n_{1\cdot}n_{0\cdot}n_{\cdot 1}n_{\cdot 0}}}$$
   where $n_{11}$ counts tokens where both features activate (above mean activation threshold), $n_{10}$ counts tokens where only $f_i$ activates, etc.
3. **Hierarchical clustering:** Apply Ward linkage clustering on the phi coefficient matrix, producing $k = 50$ clusters.
4. **Pair extraction:** All feature pairs within the same cluster are flagged as candidate absorption pairs.
5. **Dead feature filtering:** Remove features with near-zero variance (coefficient of variation $\text{CV} < 0.01$).

A pair is classified as "detected" by UAD if both features fall in the same cluster after filtering.

## 3.4 Collision Rate Computation

For each concept pair $(c_i, c_j)$, we compute the **collision rate** as the Jaccard overlap of their top-$K$ activating features:
$$R_{\text{collision}}(c_i, c_j) = \frac{|T(c_i) \cap T(c_j)|}{|T(c_i) \cup T(c_j)|}$$
where $T(c)$ is the set of top-$K$ features by mean activation on tokens belonging to $c$. We validate collision rate against the true absorption rate $R_{\text{abs}}$ using Spearman and Pearson correlation.

## 3.5 Ablations and Baselines

To isolate the source of UAD's failure, we test the following variants:

- **Full UAD:** Complete pipeline as described above.
- **No dead feature filtering:** Skip step 5.
- **No phi filtering:** Use all pairs (no clustering threshold).
- **No clustering:** Consider all $\binom{500}{2}$ pairs as candidates.
- **Single linkage clustering:** Replace Ward linkage with single linkage.
- **K-means clustering:** Replace hierarchical clustering with K-means ($k = 50$).

We also compare against two random baselines:
- **Global random:** Randomly sample pairs from all possible feature pairs.
- **Same-cluster random:** Randomly sample pairs from within each UAD cluster (controls for cluster structure).

For each method, we report Precision, Recall, F1, and the number of true positives (TP), false positives (FP), and false negatives (FN) relative to the 7 ground truth absorption pairs.

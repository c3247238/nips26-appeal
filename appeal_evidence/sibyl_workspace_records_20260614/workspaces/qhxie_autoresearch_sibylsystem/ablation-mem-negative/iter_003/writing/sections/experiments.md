# 4 Results

## 4.1 UAD Fails Catastrophically

Table 1 summarizes UAD's performance across all variants and baselines. The full UAD pipeline detects 4,155 candidate pairs, of which only 1 is a true positive (recall = 14.3%). Precision is 0.024%, yielding an F1 score of 0.00048. These results are not merely poor---they are **statistically indistinguishable from random chance**.

The most striking finding is that UAD's F1 (0.00048) is **identical** to the same-cluster random baseline (0.00048). This means that UAD's entire pipeline---phi coefficient computation, hierarchical clustering, dead feature filtering, and specificity checks---provides **zero discriminative value** over randomly sampling pairs from within the same clusters. The clustering step happens to place 1 of the 7 ground truth pairs in the same cluster by chance, and neither UAD's sophisticated filtering nor random sampling can distinguish this true positive from the 4,154 false positives.

Global random performs even worse (F1 = 0.00011), as expected, since most feature pairs are not absorption pairs. However, the margin between UAD and global random (0.00037) is three orders of magnitude below any practical threshold for usefulness.

## 4.2 Ablation Results

Table 1 shows that **all UAD variants fail**. Removing dead feature filtering or phi filtering leaves F1 unchanged at 0.00048. Removing clustering entirely (considering all pairs) reduces F1 to 0.000056. Single linkage clustering achieves F1 = 0.0, suggesting it fragments clusters too aggressively.

K-means clustering achieves the best performance among variants (F1 = 0.0037, Recall = 85.7%), detecting 6 of 7 true positives. However, precision remains near-zero (0.043%), meaning K-means produces 13,919 false positives for every true positive. This confirms that the clustering approach itself---regardless of algorithm---is the fundamental problem. Clustering groups features by some similarity metric, but the similarity metric (co-occurrence) does not capture absorption relationships.

## 4.3 Collision Rate Validation

In contrast to UAD's failure, collision rate shows a **strong positive correlation** with true absorption rate. Figure 3 displays the scatter plot of collision rate vs. true absorption rate for all 56 concept pairs (28 number pairs + 28 punctuation pairs). Spearman $\rho = 0.869$ ($n = 56$, 95% CI $[0.780, 0.938]$), and Pearson $r = 0.815$. The correlation holds across both hierarchy types (numbers and punctuation) and is robust to outliers.

Table 2 validates collision rate across experimental conditions. The pilot experiment on first letters ($n = 10$ pairs) yielded $\rho = 0.711$ (95% CI $[0.219, 0.887]$). The full experiment combining numbers and punctuation ($n = 56$) yielded $\rho = 0.869$ (95% CI $[0.780, 0.938]$). The consistency across hierarchy types and sample sizes supports collision rate as a reliable proxy metric.

## 4.4 Root Cause: Token-Level Mutual Exclusivity

Figure 2 visualizes the token-level activation patterns that explain UAD's failure. The heatmap shows four absorption features for the number hierarchy (features 11513, 12413, 22971, 24189) across tokens "one" through "eight." Each feature activates on a disjoint set of tokens:

- Feature 11513: activates exclusively on "three" (mean activation = 29.4)
- Feature 12413: activates exclusively on "one" (mean activation = 15.3)
- Feature 22971: activates exclusively on "two" (mean activation = 24.2)
- Feature 24189: activates on "four" through "eight" (mean activation = 14.3--18.9)

The key observation is **complete mutual exclusivity**: no two absorption features activate on the same token. This is not coincidental but structural: absorption features represent different child concepts (different numbers), and each token belongs to exactly one child concept.

This mutual exclusivity has a direct consequence for UAD. The phi coefficient measures co-occurrence: $\phi(f_i, f_j)$ is high when $f_i$ and $f_j$ frequently activate on the same tokens. For absorption features, $\phi \approx 0$ (or negative), because they never co-occur. Hierarchical clustering then places these features in **different clusters**, since the clustering algorithm groups features by similarity (co-occurrence), and absorption features are maximally dissimilar by this metric.

The one true positive that UAD detects (feature 24189, which activates on "four" through "eight") is detected not because of co-occurrence but because this feature spans multiple child concepts and happens to share a cluster with another feature by chance. This explains why UAD achieves the same F1 as random sampling: the single true positive is a statistical accident, not a successful detection.

## 4.5 False Positive Analysis

The 4,154 false positives produced by UAD are not random errors but systematic consequences of the co-occurrence clustering mechanism. Because clustering groups features by shared activation patterns on the same tokens, it naturally captures features that activate in similar contexts---for example, features for "number words" and features for "counting contexts." These features are semantically related but not hierarchically related. The clustering algorithm has no mechanism to distinguish contextual similarity from hierarchical absorption, because both produce high co-occurrence scores. This confirms that UAD's detection mechanism is structurally misaligned with the target phenomenon.

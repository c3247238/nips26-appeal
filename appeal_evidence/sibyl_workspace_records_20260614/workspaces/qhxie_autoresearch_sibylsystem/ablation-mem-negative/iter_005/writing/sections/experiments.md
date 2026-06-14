# 4 Results

## 4.1 UAD Fails Catastrophically

Table 1 summarizes UAD's performance across all variants and baselines. The full UAD pipeline detects 4,155 candidate pairs, of which only 1 is a true positive (recall = 14.3%). Precision is 0.024%, yielding an F1 score of 0.00048.

The most important observation is that UAD and the same-cluster random baseline detect **exactly the same number of true positives** (1 out of 4,155 candidates). This is not a statistical coincidence but an arithmetic identity: both methods sample from the same set of cluster-internal pairs, and the clustering happens to place 1 of the 7 ground truth pairs in the same cluster by chance. We report the bootstrap 95% confidence interval for F1 as $[0.00012, 0.00102]$, confirming that the performance is near the floor of detectability.

Global random performs even worse (F1 = 0.00011), as expected, since most feature pairs are not absorption pairs. The margin between UAD and global random (0.00037) is three orders of magnitude below any practical threshold for usefulness.

**Ground truth limitation:** Our ground truth comprises 7 true absorption pairs (6 distinct pairs + 1 self-pair) in the number hierarchy. This small sample limits statistical power; conclusions about UAD's failure should be interpreted as strong evidence rather than definitive proof.

## 4.2 Ablation Results

Table 1 shows that **all UAD variants fail**. Removing dead feature filtering or phi filtering leaves F1 unchanged at 0.00048. Removing clustering entirely (considering all pairs) reduces F1 to 0.000056. Single linkage clustering achieves F1 = 0.0.

K-means clustering achieves the best performance among variants (F1 = 0.0037, Recall = 85.7%), detecting 6 of 7 true positives. However, precision remains near-zero (0.043%), meaning K-means produces 2,324 false positives for every true positive. The difference between K-means and Ward linkage is informative: K-means uses hard assignment, forcing all features into clusters regardless of pairwise distances. This means features with phi ≈ 0 (including absorption features) may be placed in the same cluster due to random initialization or centroid proximity. Ward linkage, which minimizes within-cluster variance, is more sensitive to the near-zero phi values and places absorption features in different clusters. This confirms that the clustering approach itself---regardless of algorithm---is the fundamental problem, as even the "best" variant achieves F1 < 0.004.

## 4.3 Collision Rate Internal Consistency

In contrast to UAD's failure, collision rate shows strong internal consistency across concept pairs. Figure 3 displays the scatter plot of collision rate vs. absorption rate for all 56 concept pairs (28 number pairs + 28 punctuation pairs). Spearman $r = 0.869$ ($n = 56$, 95% CI $[0.780, 0.938]$), and Pearson $r = 0.815$.

**Interpretation:** Since both metrics are computed from the same top-$K$ feature sets, this correlation measures the internal consistency of our operationalization, not an independent proxy relationship. The strong correlation indicates that our definition of absorption via top-$K$ feature overlap is structurally coherent: concept pairs with known hierarchical relationships show systematically higher overlap than unrelated pairs. This validates the operationalization as a useful screening tool, though it does not establish causality or independent predictive validity.

## 4.4 Root Cause: Token-Level Mutual Exclusivity

Figure 2 visualizes the token-level activation patterns that explain UAD's failure. The heatmap shows four absorption features for the number hierarchy across tokens "one" through "eight." Each feature activates on a disjoint set of tokens:

- Feature 11513: activates exclusively on "three" (mean activation = 29.4)
- Feature 12413: activates exclusively on "one" (mean activation = 15.3)
- Feature 22971: activates exclusively on "two" (mean activation = 24.2)
- Feature 24189: activates on "four" through "eight" (mean activation = 14.3--18.9)

The key observation is **complete mutual exclusivity**: no two absorption features activate on the same token. This is structural: absorption features represent different child concepts (different numbers), and each token belongs to exactly one child concept.

This mutual exclusivity has a direct consequence for UAD. The phi coefficient measures co-occurrence: $\phi(f_i, f_j)$ is high when $f_i$ and $f_j$ frequently activate on the same tokens. For absorption features, $\phi \approx 0$ (or negative), because they never co-occur. Hierarchical clustering then places these features in **different clusters**, since the clustering algorithm groups features by similarity (co-occurrence), and absorption features are maximally dissimilar by this metric.

The one true positive that UAD detects is detected not because of co-occurrence but because this feature spans multiple child concepts and happens to share a cluster with another feature by chance. This explains why UAD achieves the same F1 as random sampling: the single true positive is a statistical accident, not a successful detection.

## 4.5 False Positive Analysis

The 4,154 false positives produced by UAD are systematic consequences of the co-occurrence clustering mechanism. Because clustering groups features by shared activation patterns on the same tokens, it naturally captures features that activate in similar contexts---for example, features for "number words" and features for "counting contexts." These features are semantically related but not hierarchically related. The clustering algorithm has no mechanism to distinguish contextual similarity from hierarchical absorption, because both produce high co-occurrence scores. This confirms that UAD's detection mechanism is structurally misaligned with the target phenomenon.

# 2 Background and Related Work

## 2.1 Feature Absorption: Definition and Mechanism

Feature absorption was first systematically characterized by Chanin et al. [2024] as a failure mode in SAEs where parent features are suppressed by their child features. In a well-structured SAE, one might expect to find features encoding broad concepts (e.g., "animal") alongside features encoding specific instances (e.g., "dog," "cat"). However, during training, the SAE may learn to allocate nearly all activation mass to the most specific features, causing the parent feature to have negligible activation.

Formally, let $f_p$ be a parent feature and $f_{c_1}, \ldots, f_{c_k}$ be child features. Absorption occurs when:
$$\mathbb{E}[|f_p(x)| \mid f_{c_1}(x) > 0 \lor \cdots \lor f_{c_k}(x) > 0] \ll \mathbb{E}[|f_p(x)|]$$

That is, the parent's expected activation is significantly reduced when any child is active. This suppression is not merely correlation but a causal consequence of the SAE's reconstruction objective: if the children can fully reconstruct the input, the parent becomes redundant and is pruned by the sparsity penalty.

The danger of absorption lies in interpretability illusions. When researchers inspect the top-activating tokens for a parent feature, they may find no coherent pattern and conclude the feature is uninterpretable or noise. In reality, the feature may encode a meaningful broad concept that is simply suppressed in contexts where specific instances are present.

## 2.2 Existing Detection Methods (All Supervised)

All prior work on absorption detection requires supervised ground truth. Chanin et al. [2024] propose a supervised probe-based approach: (1) identify a candidate parent feature using known concept labels, (2) train a linear probe to predict the parent feature's activation from child features' activations, (3) ablate child features and measure parent recovery. If the parent can be accurately predicted from children and recovers when children are removed, absorption is confirmed.

Karvonen et al. [2025] extend this approach with causal mediation analysis, measuring the direct and indirect effects of child features on parent activation. While more rigorous, this method still requires known parent-child relationships and trained probes.

The common limitation of all supervised approaches is scalability: they require labeled parent features, which are only available for a tiny fraction of the SAE dictionary. With 24,576 features in a typical SAE, manual labeling is infeasible. An unsupervised method would enable screening the entire dictionary for absorption patterns.

## 2.3 Unsupervised Absorption Detection (UAD)

UAD [Chanin et al., 2024] proposes co-occurrence clustering as an unsupervised alternative. The method assumes that features with hierarchical relationships will frequently co-occur on the same tokens: a parent and its child should both activate when the child concept is present. The pipeline consists of five steps:

1. **Feature selection:** Select the $n$ most active features (by mean activation) to reduce computational cost.
2. **Co-occurrence matrix:** Compute phi coefficients $\phi(f_i, f_j)$ for all feature pairs, measuring statistical association.
3. **Hierarchical clustering:** Cluster features using Ward linkage on the phi matrix, producing $k$ clusters.
4. **Pair extraction:** All feature pairs within the same cluster are flagged as candidate absorption pairs.
5. **Dead feature filtering:** Remove near-zero variance features that may produce spurious correlations.

UAD's appeal is its training-free nature: it requires no labeled data, no probes, and no model retraining. However, its core assumption---that hierarchical features co-occur---has never been rigorously validated on pre-trained SAEs.

## 2.4 Collision Rate as a Proxy Metric

Collision rate, defined as the Jaccard overlap of top-$K$ activating features between two concepts, has been mentioned in passing as a potential proxy for absorption strength [Chanin et al., 2024]. The intuition is that if two concepts share many top-activating features, they may have hierarchical relationships. However, collision rate has never been systematically validated against ground-truth absorption rates at scale.

Our work provides the first large-scale validation of collision rate as an absorption proxy, testing it across 56 concept pairs spanning two hierarchy types (numbers and punctuation). We also provide the first empirical evaluation of UAD on pre-trained SAEs, revealing a fundamental flaw in its underlying assumption.

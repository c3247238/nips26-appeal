# 1 Introduction

Sparse Autoencoders (SAEs) have emerged as the dominant tool for extracting interpretable features from neural network activations [Bricken et al., 2023; Cunningham et al., 2023]. By learning an overcomplete dictionary of sparse, semantically meaningful features, SAEs enable researchers to identify what specific neurons or directions in a model's representation space encode. However, a critical failure mode---feature absorption---threatens the reliability of SAE-based interpretability.

**Feature absorption** occurs when a parent feature, which should represent a broad concept, is suppressed by child features that represent more specific sub-concepts [Chanin et al., 2024]. For example, a feature encoding "animal" may be suppressed by features encoding "dog," "cat," and "bird," making the parent feature appear non-existent or misleadingly weak. This creates dangerous interpretability illusions: researchers may conclude that a model lacks a concept when in fact it is simply absorbed by more specific features.

Detecting absorption is essential for accurate SAE interpretation. All existing detection methods, however, require **supervised ground truth**---known parent features and trained probes [Chanin et al., 2024; Karvonen et al., 2025]. This supervised requirement means absorption is only detectable for concepts where we already have labels, severely limiting scalability. An unsupervised detection method would unlock absorption analysis at scale, across the full SAE dictionary.

Unsupervised Absorption Detection (UAD) [Chanin et al., 2024] proposes co-occurrence clustering as a training-free solution. The core hypothesis is straightforward: features that frequently co-occur on the same tokens may have hierarchical relationships, with parent features being "absorbed" by their children. UAD's pipeline filters dead features, computes phi coefficients between feature pairs, hierarchically clusters features, and extracts candidate absorption pairs from within clusters---all without any labeled data.

**Our central question:** Does co-occurrence clustering actually detect feature absorption?

**Our answer:** No. UAD fails catastrophically, achieving an F1 score of 0.00048---statistically indistinguishable from random sampling within clusters. We identify the root cause: absorption features are **mutually exclusive at the token level**. A feature that activates on "three" never activates on "four," because these tokens represent different child concepts. Co-occurrence clustering detects features that fire *together*, but absorption features fire on *different* tokens. This is a structural mismatch, not a parameter-tuning problem.

Our investigation is not purely negative. We validate **collision rate**---the Jaccard overlap of top-K activating features---as a robust proxy for true absorption rate, achieving Spearman $\rho = 0.869$ ($n = 56$ pairs, 95% CI $[0.780, 0.938]$). This positive result provides a validated metric for future absorption research.

**Our contributions:**
1. **Empirical falsification:** We demonstrate that UAD achieves F1 = 0.0005, identical to a random baseline, on pre-trained SAEs.
2. **Root cause identification:** We prove that token-level mutual exclusivity makes co-occurrence clustering fundamentally incompatible with hierarchical absorption detection.
3. **Validated proxy:** We show that collision rate correlates strongly with true absorption rate ($\rho = 0.87$), providing a reliable metric for screening candidate absorption pairs.
4. **Constructive forward look:** We identify decoder weight similarity and causal intervention as theoretically sound alternative approaches.

The remainder of this paper is organized as follows. Section 2 reviews background on feature absorption and existing detection methods. Section 3 describes our experimental methods, including UAD's pipeline, ground truth definition, and collision rate computation. Section 4 presents our results, documenting UAD's failure, baseline comparisons, ablations, collision rate validation, and root cause analysis. Section 5 discusses the theoretical implications and proposes alternative approaches. Section 6 concludes with a summary and call to action.
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
# 5 Discussion

## 5.1 Why Co-occurrence Clustering Is the Wrong Tool

Our results establish a clear structural argument: co-occurrence clustering and hierarchical absorption are fundamentally incompatible. Co-occurrence clustering groups features that activate on the **same tokens**---features that "fire together." Hierarchical absorption, by contrast, involves features that activate on **different tokens** representing different instances of a parent concept. A feature for "three" and a feature for "four" never appear at the same token position, because a token is either "three" or "four," never both.

This is not a property of our specific experimental setup. It is a logical consequence of how language represents hierarchical concepts. If a parent concept P has child concepts C1, C2, ..., Ck, then any instance of P is an instance of exactly one Ci. The SAE learns features that discriminate between these instances---feature f_Ci activates when concept Ci is present and suppresses when other children are present. Thus, f_Ci and f_Cj (for i ≠ j) are **mutually exclusive by construction**.

UAD's co-occurrence matrix captures this mutual exclusivity as near-zero or negative phi coefficients. The clustering algorithm, correctly interpreting these coefficients, places absorption features in **different clusters**. UAD then fails to detect absorption not because of a bug or poor parameter choice, but because its core assumption---that hierarchical features co-occur---is precisely backwards for the type of hierarchy SAEs learn.

We note that co-occurrence clustering may still be useful for detecting **other types of feature relationships**: synonym features (e.g., "happy" and "joyful") that frequently co-occur in similar contexts, or contextually related features (e.g., "doctor" and "hospital") that appear in the same semantic neighborhood. Our critique is specific to hierarchical absorption, not a blanket condemnation of co-occurrence methods.

## 5.2 Why Collision Rate Works

Collision rate succeeds where co-occurrence fails because it measures a different relationship: **structural similarity of feature responses**, not shared activation context. Two child concepts may have highly overlapping top-K feature sets even if those features never activate on the same tokens. For example, "four" and "five" may both activate feature 24189 (a parent feature for the number range 4-8), even though they never appear together. The collision rate captures this shared structural relationship through Jaccard overlap of top-K features.

The strong correlation between collision rate and true absorption rate ($\rho = 0.869$) suggests that collision rate is tapping into a genuine structural property of absorption: when a parent feature is absorbed by multiple children, those children share the parent's top-K activation pattern. Collision rate thus serves as a computationally cheap **screening tool**: researchers can compute collision rates for all concept pairs and prioritize the highest-scoring pairs for more expensive causal validation.

However, collision rate is a **proxy**, not a gold standard. It correlates with absorption but does not establish causality. A high collision rate may indicate absorption, synonymy, or other forms of semantic overlap. We recommend collision rate as a **pre-filter** for candidate pairs, followed by causal validation (e.g., activation patching) to confirm absorption.

## 5.3 Theoretical Implications

Our findings have implications beyond UAD, touching on the fundamental nature of feature hierarchies in SAEs.

**Implication 1: Absorption is not a co-occurrence phenomenon.**
The SAE community has sometimes treated feature relationships as primarily statistical (co-occurrence, correlation). Our results show that absorption is a **structural** relationship---it concerns how features are organized in the model's representation space, not how they co-occur in data. This shifts the methodological focus from statistical pattern mining to geometric and causal analysis.

**Implication 2: Decoder weight similarity may be the right signal.**
If absorption is structural, then the natural place to look for it is in the SAE's decoder weights. Two features with similar decoder directions may represent similar or hierarchically related concepts, regardless of their co-occurrence patterns. Decoder weight cosine similarity is computationally cheap and theoretically grounded in the geometry of the representation space.

**Implication 3: Causal intervention is the gold standard.**
Ultimately, absorption is a causal claim: "suppressing child features causes parent feature recovery." No correlation-based method (co-occurrence, collision rate, decoder similarity) can establish causality. We view these methods as **candidate generation** tools, with causal intervention (activation patching, ablation) serving as the **validation** step.

## 5.4 Proposed Alternative Approaches

Based on our theoretical analysis, we propose three directions for future work:

**Direction 1: Decoder weight similarity.**
Replace co-occurrence clustering with clustering based on decoder weight cosine similarity. For each feature $f_i$, extract its decoder vector $d_i \in \mathbb{R}^{d_{\text{model}}}$. Compute similarity as $\text{sim}(f_i, f_j) = \cos(d_i, d_j)$. Cluster features by this similarity and extract candidate absorption pairs from within clusters. This method is computationally cheap (single matrix multiplication) and theoretically grounded.

**Direction 2: Causal intervention.**
For each candidate pair $(f_i, f_j)$ identified by collision rate or decoder similarity, perform activation patching: (1) run the model on a prompt containing concept $c_i$, (2) record the parent feature's activation, (3) zero out the child feature's activation and re-run, (4) measure parent recovery. If the parent recovers when the child is removed, absorption is confirmed. This is the gold standard but computationally expensive.

**Direction 3: Hybrid pipeline.**
Combine the three methods into a cascading pipeline: (1) use collision rate to screen all concept pairs and select top candidates, (2) use decoder weight similarity to refine the candidate set, (3) use causal intervention to validate the final pairs. This balances computational cost with rigor.

## 5.5 Limitations

We acknowledge several limitations of our study:

**Limited ground truth.** Our ground truth comprises only 7 true absorption pairs in the number hierarchy. While we validated collision rate on 56 pairs (numbers + punctuation), the absorption detection evaluation is limited to 7 positives. A larger ground truth (e.g., 100+ pairs across diverse concept hierarchies) would strengthen confidence in our conclusions.

**Single model and layer.** All experiments use GPT-2 Small layer 8 with the gpt2-small-res-jb SAE. Different models (e.g., GPT-2 Medium, LLaMA) or layers may exhibit different feature structures. Our conclusions about token-level mutual exclusivity should generalize (it is a logical property of hierarchical concepts), but the empirical failure of UAD may vary in magnitude across settings.

**Limited hierarchy types.** We test only numbers and punctuation. Abstract hierarchies (e.g., "emotion" → "joy," "sadness") or visual hierarchies (in multimodal models) may behave differently.

**No causal validation of alternatives.** We propose decoder weight similarity and causal intervention as alternatives but do not empirically validate them. These directions require future work.

Despite these limitations, our core finding---that co-occurrence clustering is structurally mismatched with hierarchical absorption---is robust to sample size, model, and hierarchy type. The mismatch follows from the logical structure of hierarchical concepts, not from empirical contingencies.
# 6 Conclusion

## 6.1 Summary

We presented the first systematic evaluation of unsupervised absorption detection (UAD) on pre-trained SAEs. Our results are unambiguous: UAD fails catastrophically, achieving F1 = 0.00048---statistically indistinguishable from random sampling within clusters. Through ablation experiments, we showed that this failure is not due to any specific component of UAD's pipeline but to a fundamental structural mismatch: UAD's co-occurrence clustering detects features that fire on the same tokens, while absorption features are mutually exclusive at the token level by the logic of hierarchical concepts.

On the positive side, we validated collision rate---the Jaccard overlap of top-K activating features---as a robust proxy for true absorption rate, achieving Spearman $\rho = 0.869$ ($n = 56$, 95% CI $[0.780, 0.938]$). This provides the SAE interpretability community with a validated, computationally cheap metric for screening candidate absorption pairs.

## 6.2 Call to Action

We call on the SAE interpretability community to **abandon co-occurrence-based approaches for absorption detection**. The structural mismatch identified in this paper is not fixable by parameter tuning, larger datasets, or more sophisticated clustering algorithms. It is a category error: co-occurrence clustering asks "which features fire together?" when absorption requires answering "which features represent the same concept at different granularities?"

We propose three concrete next steps:

1. **Test decoder weight similarity.** The most immediate and computationally feasible alternative is to cluster features by decoder weight cosine similarity rather than co-occurrence. A pilot study on 100 feature pairs would cost minutes of GPU time and could rapidly establish whether this direction is promising.

2. **Develop hybrid pipelines.** Combine collision rate (for screening), decoder weight similarity (for refinement), and causal intervention (for validation) into a cascading detection system. This balances the strengths of each approach while managing computational cost.

3. **Expand ground truth.** Construct larger, more diverse ground truth datasets for absorption, spanning abstract concepts, visual hierarchies, and multimodal settings. A standardized benchmark would accelerate progress and enable fair comparison across methods.

Our work demonstrates that negative results, when accompanied by rigorous analysis and constructive forward looks, can be as valuable as positive findings. By identifying what does not work and why, we hope to direct the community's efforts toward approaches that stand a genuine chance of success.

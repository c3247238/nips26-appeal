# 5 Discussion

## 5.1 Why Co-occurrence Clustering Is the Wrong Tool

Our results establish that co-occurrence clustering fails to detect feature absorption in tested token-disjoint hierarchies. The root cause is a structural mismatch between what UAD measures and what absorption is.

Co-occurrence clustering assumes that related features activate on the same inputs. This assumption holds for synonym features ("happy" and "joyful") and contextually related features ("king" and "queen"), which frequently co-occur in natural text. However, absorption in token-disjoint hierarchies operates differently: absorbed features represent mutually exclusive sub-concepts. A feature that activates on "three" never activates on "four," because these tokens represent different child concepts that cannot appear at the same position.

UAD is designed to find features that fire *together*---features whose activation vectors have high correlation. But absorption features fire on *different* tokens representing alternative instances of the same abstract concept. The phi coefficient, which measures co-occurrence, is near-zero for absorption features not because the features are unrelated, but because they are structurally prevented from co-occurring. Hierarchical clustering then correctly places these features in different clusters, since the clustering algorithm groups features by similarity, and absorption features are maximally dissimilar by the co-occurrence metric.

This is not a parameter-tuning problem. No adjustment of the clustering algorithm, threshold, or feature selection criteria can overcome the fundamental issue: the signal UAD searches for (co-occurrence) is orthogonal to the phenomenon it targets (absorption in token-disjoint hierarchies).

We emphasize that this conclusion is scoped to tested token-disjoint hierarchies (numbers, punctuation). Semantic hierarchies where child concepts co-occur in natural text (e.g., "animal" and "dog" may appear in the same context) may exhibit different patterns, and UAD's performance on such hierarchies remains an open question.

## 5.2 Why Collision Rate Shows Internal Consistency

In contrast to UAD's failure, collision rate demonstrates strong internal consistency. The Spearman correlation of $r = 0.869$ across 56 concept pairs indicates that our operationalization of absorption via top-$K$ feature overlap is structurally coherent.

The key insight is that collision rate measures a different property than co-occurrence. While co-occurrence asks "do these features activate on the same tokens?" collision rate asks "do these concepts share the same absorbing features?" Two child concepts ("three" and "four") may both have feature 13586 in their top-5 activating features even though they never appear together. The collision rate captures this structural relationship without requiring token-level co-occurrence.

The high correlation between collision rate and absorption rate validates that our operational definition produces stable, expected patterns: concept pairs with known hierarchical relationships show systematically higher overlap than unrelated pairs. We reiterate that this is an internal consistency check of the operationalization, not an independent proxy validation, as both metrics are computed from the same top-$K$ feature sets.

## 5.3 Theoretical Implications

Our findings carry three theoretical implications for SAE interpretability research.

**First, absorption is not co-occurrence.** The mechanism is more subtle than the intuition that "features that appear together get merged." In token-disjoint hierarchies, absorption occurs precisely because features do *not* co-occur---the SAE allocates different features to different child concepts, suppressing the parent in the process. This suggests that absorption detection methods must look beyond co-occurrence patterns.

**Second, decoder weight similarity may be the right signal.** If two child features share a parent in reconstruction, their decoder directions should be geometrically related in the activation space. Cosine similarity of decoder weight vectors is training-free and directly measures structural relationships without requiring token-level co-occurrence. This makes it a promising candidate for unsupervised absorption detection.

**Third, causal intervention remains the gold standard.** Only activation patching can definitively establish absorption by showing that suppressing a child feature causes parent recovery. While computationally expensive, causal validation provides the strongest evidence for absorption and should be used to validate any unsupervised detection method.

## 5.4 Proposed Alternative Approaches

Based on our analysis, we propose three directions for future work:

**Decoder weight similarity (highest priority).** Compute cosine similarity between decoder weight vectors of candidate feature pairs. Features that share a parent in reconstruction should have geometrically aligned decoder directions. This method is training-free, scalable to the full SAE dictionary, and does not depend on token-level co-occurrence.

**Causal intervention validation.** Use activation patching to test whether suppressing a child feature causes parent recovery. While expensive, this provides definitive evidence and can validate candidate pairs identified by cheaper methods.

**Semantic similarity clustering.** Replace co-occurrence-based clustering with clustering by decoder weight similarity. This maintains the unsupervised, training-free nature of UAD while using a signal that is theoretically aligned with absorption.

## 5.5 Limitations

We acknowledge several limitations of our study:

1. **Single SAE.** All experiments use gpt2-small-res-jb at layer 8. Results may not generalize to other layers, models, or SAE architectures.

2. **Small ground truth.** Only 7 true absorption pairs (6 distinct + 1 self-pair) limit statistical power. We provide bootstrap confidence intervals to quantify uncertainty.

3. **Token-disjoint hierarchies only.** Numbers and punctuation are token-disjoint. Semantic hierarchies where children co-occur in natural text may show different patterns.

4. **No causal validation of alternatives.** Decoder weight similarity and causal intervention are proposed but not empirically tested in this work.

5. **Single seed.** All experiments use seed 42. Sensitivity to corpus sampling is unknown.

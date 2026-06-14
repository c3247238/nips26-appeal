# Backup Ideas for Potential Pivot

## Alternative 1 (cand_b): Absorption-Aware Hierarchical Feature Decomposition

**Candidate ID**: cand_b
**Pivot Trigger**: If H3 contradiction is confirmed in replication; if absorption features are genuinely hub-like

**Title**: Absorption-Aware Hierarchical SAE: Leveraging HSAE Structure to Prevent and Detect Absorption

**Core Idea**: Combine HSAE's hierarchical feature discovery with absorption-aware regularization. The H3 finding (absorbed features are more steerable) suggests absorbed features may function as hub-like structures. HSAE's parent-child relationships provide structural priors to: (a) detect absorption by comparing child feature directions to their expected parent projections, and (b) prevent absorption by regularizing hub features away from their children.

**Why Attractive Now**: The H3 reversal suggests absorbed features are hub-like (high leverage, high activation frequency). cand_b's mechanism of using HSAE's hierarchy to regularize hub-child relationships directly addresses this finding.

**Novelty**: No prior work combines HSAE's hierarchical discovery with absorption mitigation. HSAE discovers the hierarchy; our regularization prevents hub-like features from collapsing into absorption proxies.

**Key Experiments**:
1. Load HSAE parent-child relationships from Luo et al.
2. Test H_b1: Are HSAE-identified child features more absorbed? (Are child features the high-UAS, high-steering hub features?)
3. Test H_b2: Does absorption-aware regularization (penalize child-parent cosine similarity) reduce absorption without harming reconstruction?
4. Compare against OrtSAE and ATM on absorption reduction and reconstruction

**Pivot Decision**: If H3 replication confirms absorbed features are hubs, pivot to cand_b to investigate the hierarchical structure of hub features.

---

## Alternative 2 (cand_c): Absorption as Information Compression Tradeoff

**Candidate ID**: cand_c
**Pivot Trigger**: If H5 is strongly confirmed; if absorption is Pareto-optimal

**Title**: Is Feature Absorption Optimal? Information-Theoretic Analysis of the Sparsity-Absorption Pareto Frontier

**Core Idea**: Test the contrarian hypothesis rigorously: is absorption actually a locally-optimal solution to the SAE training objective? Use information-theoretic analysis (mutual information between features and downstream representations) to map the Pareto frontier between sparsity, reconstruction quality, and absorption. The H3 finding (absorbed features are MORE steerable) adds a new dimension: steering sensitivity should also be on the Pareto frontier.

**Why Attractive Now**: The H3 finding adds a new twist to the Pareto analysis. If absorbed features are more steerable, then absorption may be a *feature*, not a bug -- a consequence of the SAE learning high-leverage representations.

**Novelty**: First 4D Pareto frontier (sparsity, reconstruction, absorption, steering sensitivity). Previous work on HierarchicalTopK only maps 2D.

**Key Experiments**:
1. Train SAEs across a wide range of lambda_sparse values
2. For each, compute: reconstruction CE, L0, absorption score (UAS), steering sensitivity
3. Map the 4D Pareto frontier
4. Test whether OrtSAE/ATM lie on the frontier or off it

**Pivot Decision**: If the 4D Pareto frontier reveals that absorption is genuinely optimal for steering sensitivity, the paper reframes: absorption is not a failure mode but a consequence of learning high-leverage representations.

---

## Alternative 3 (cand_d): Feature Reliability Index for Interpretability Practitioners

**Candidate ID**: cand_d
**Pivot Trigger**: If H5 is confirmed and H3 is consistently reversed; if absorption is one of several failure modes

**Title**: Beyond Absorption: A Multi-Factor Feature Reliability Index for SAE-Based Interpretability

**Core Idea**: Rather than focusing solely on absorption, develop a comprehensive Feature Reliability Index (FRI) that combines absorption score, activation frequency, steering sensitivity, and downstream causal contribution into a single score. The H3 finding suggests steering sensitivity should be a *positive* component of the FRI (high sensitivity = reliable feature), not negative.

**Why Attractive Now**: The H3 reversal changes the FRI calculus. Absorbed features are more steerable, so they may actually be MORE reliable for some interventions. The FRI should capture this nuance.

**Novelty**: First multi-dimensional reliability scoring that incorporates steering sensitivity as a positive signal (not just purity/specificity as negatives).

**Key Experiments**:
1. Define FRI = weighted combination: steering_sensitivity (positive), activation_frequency (positive), UAS (negative), specificity (positive)
2. Validate FRI against human interpretability judgments (annotator study)
3. Show that FRI predicts intervention effectiveness better than any single axis
4. Release FRI computation tool integrated into SAELens

**Pivot Decision**: If experiments reveal that absorbed features are useful for steering but not for classification, the FRI framework captures this dual nature.

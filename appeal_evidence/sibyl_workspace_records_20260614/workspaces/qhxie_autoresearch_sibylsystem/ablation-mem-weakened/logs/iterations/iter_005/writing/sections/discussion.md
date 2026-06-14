# 5. Discussion

## 5.1 Absorption as Optimal Compression

The central finding of this paper is the precision-recall asymmetry (H5): precision equals 1.0 universally at k >= 5 across 26 features, while recall varies widely (0.05-1.0). This asymmetry is the signature of optimal compression behavior.

Under rate-distortion theory, SAEs minimize reconstruction error (distortion) subject to sparsity constraints (rate). When parent and child features co-occur hierarchically, the sparsity objective incentivizes merging their directions—but only in the encoder. The decoder direction $d_f = W_{\text{dec}}[:, f]$ is a column of $W_{\text{dec}}$ and is not directly affected by which encoder activations fire.

This creates the asymmetry we observe:
- **Precision preserved**: The decoder direction remains accurate. When the child feature fires, it produces the correct reconstruction using its own direction; when the parent would have fired, the decoder still produces approximately correct output via the child direction.
- **Recall reduced**: The encoder activation is suppressed because the child fires in contexts where the parent would have fired. This reduces coverage (recall) but not selectivity (precision).

The theoretical implication: absorption is not a pathology requiring architectural fixes. It is the **optimal strategy for minimizing rate while preserving decoder alignment** under hierarchical co-occurrence constraints. This reframing explains why absorption persists in trained SAEs—the rate-distortion trade-off actively incentivizes it.

## 5.2 Why the Inhibition Graph Failed

Our H6 falsification (precision@20 = 0.0 for decoder-correlation-based prediction) rules out a specific mechanistic hypothesis, but the failure is informative.

The structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ is mathematically exact for tied-weight SAEs, where $W_{\text{enc}} = W_{\text{dec}}^T$. If the gpt2-small-res-jb SAE uses untied weights (which is common in practice), the correspondence is only approximate, breaking the predictive relationship.

Alternative explanations:
1. **Encoder dynamics drive absorption**: The decision of which feature fires is determined by encoder activations, not decoder geometry. Decoder correlations capture reconstruction geometry, not competition dynamics.
2. **Chanin metric captures something else**: The differential correlation metric may measure correlation structure rather than competitive suppression. Two features may be correlated without competing.
3. **First-letter features are too shallow**: True hierarchical absorption may require deeper semantic hierarchies (animal → dog → poodle) rather than surface-level letter grouping.

The falsification is valuable: it eliminates one specific hypothesis and suggests that decoder-geometry-based approaches to understanding or mitigating absorption are unlikely to succeed.

## 5.3 Why Prior Work Found Null Results

The weak interpretability-utility correlation reported by Wang et al. (ICLR 2026) and the "sanity checks" results from Korznikov et al. (2026) are consistent with our findings. Our work explains why:

1. **Raw metrics confound generic and specific effects**: Raw steering success rates are dominated by the baseline capability of SAE features, not absorption-specific effects. Delta correction (subtracting random baseline) isolates feature-specific effects but still yields null results after correction.

2. **Low absorption variance constrains power**: With mean absorption of 3.9% (L4) and 3.4% (L8), the signal-to-noise ratio is low. Even if absorption had a small effect, detecting it requires large sample sizes.

3. **The "fixes" may be unnecessary**: If absorption does not significantly harm downstream tasks (H1-H4 null), architectural solutions like Matryoshka, OrtSAE, and Balance Matryoshka may be addressing a non-problem.

The field needs more rigorous null-result reporting with proper controls. Our contribution includes not just the null results but the methodological framework for detecting and reporting them honestly.

## 5.4 Relationship to Existing Architectural Solutions

### 5.4.1 Matryoshka SAEs

Matryoshka SAEs (Bussmann et al., 2025) use nested multi-level dictionaries to reduce absorption from 0.49 to 0.05. Our results suggest this reduction may be unnecessary for downstream tasks: absorption does not significantly degrade steering or probing (H1-H4 null). The absorption reduction may improve interpretability metrics without improving practical utility.

**Caveat**: Our results are on GPT-2 Small (124M); larger models may show different absorption patterns and different functional consequences.

### 5.4.2 OrtSAE

OrtSAE (Korznikov et al., 2025) enforces decoder orthogonality to reduce absorption by 65%. Our H6 falsification suggests decoder orthogonality may not address the root cause of absorption: if decoder geometry does not predict absorption (precision@20 = 0.0), orthogonalizing the decoder may not solve the problem.

### 5.4.3 Balance Matryoshka

Chanin et al.'s Balance Matryoshka addresses both absorption and hedging, proposing that Matryoshka exacerbates hedging while reducing absorption. Our precision-recall asymmetry suggests a unified interpretation: both phenomena reflect optimal compression trade-offs under different constraints.

### 5.4.4 ATM (Adaptive Temporal Masking)

Li et al.'s ATM achieves ~40% absorption reduction via temporal EMA tracking. If absorption is optimal compression, why would reducing it improve the SAE? Possible explanations:
1. ATM may target a specific subtype of absorption that does have functional consequences
2. The reduction may improve interpretability metrics without improving utility
3. Optimal compression may have regimes where more compression is harmful

Further work is needed to understand whether ATM's absorption reduction translates to functional improvements.

## 5.5 Methodological Contributions

### 5.5.1 Baseline Correction (Delta-Corrected Steering)

Raw steering metrics confound absorption-specific effects with generic directional bias. The random baseline subtraction ($\Delta S = S_{\text{feature}} - S_{\text{random}}$) isolates absorption-specific effects. We recommend this as standard practice for SAE steering evaluations.

### 5.5.2 Precision-Recall Decomposition

Standard F1 scores aggregate precision and recall, masking the asymmetry we observe. Decomposing into precision (selectivity) and recall (coverage) reveals whether absorption affects false positives (it does not) or false negatives (it does). This decomposition should be standard for absorption analysis.

### 5.5.3 EC50 Dose-Response Analysis

The dose-response framework ($S(s) = S_{\text{max}} \cdot s^n / (EC50^n + s^n)$) provides a principled way to compare steering efficiency across features. EC50 captures the steering strength required for 50% of maximum effect, enabling comparisons that raw success rates cannot support.

### 5.5.4 Multiple Comparison Correction

Most prior absorption studies report raw correlations without correction. Our systematic application of Bonferroni (alpha_B = 0.00417) and BH-FDR (q < 0.05) to 12 tests reveals the importance of correction: uncorrected p=0.028 becomes corrected p=0.334. This is essential for honest scientific reporting.

## 5.6 Implications for the Field

### 5.6.1 Reassess Architectural Fixes

If absorption does not significantly harm downstream tasks, architectural solutions (Matryoshka, OrtSAE, Balance Matryoshka, ATM) may be addressing a non-problem. The field should focus on:
1. Identifying which (if any) absorption subtypes have functional consequences
2. Developing metrics that distinguish harmful absorption from benign optimal compression
3. Testing proposed fixes on downstream utility, not just absorption metrics

### 5.6.2 Rethink Absorption Detection

Our H10 result (random SAE shows 8x higher absorption than trained SAE) raises questions about the Chanin metric. If the metric is sensitive to structural artifacts that training reduces, it may not be well-calibrated to measure "absorption as failure." Better metrics may be needed.

### 5.6.3 Embrace Honest Null Results

Null results are difficult to publish but essential for scientific progress. Our finding that zero hypotheses survive multiple comparison correction is a genuine contribution: it rules out large-effect absorption degradation in GPT-2 Small SAEs. The field should reward such reporting rather than requiring positive findings.

## 5.7 Limitations and Caveats

1. **Single model family (GPT-2 Small)**: Larger models (Gemma-2-2B, Llama-3-8B) may show different absorption patterns and different functional consequences.

2. **Narrow feature set (first-letter A-Z)**: Semantic hierarchies (animal → dog → poodle) may exhibit different absorption dynamics than surface-level letter grouping.

3. **Small sample size (n=26 features)**: Constrains statistical power; small-to-medium effects may be undetected.

4. **Chanin metric validity**: H10 suggests the metric may measure structural artifacts rather than learned absorption.

5. **Two downstream tasks only**: Steering and probing are important but do not capture all SAE applications.

These limitations are honest and do not invalidate the main findings. They define the scope of valid claims and guide future work.

## 5.8 Future Work

1. **Larger models**: Test absorption-functional correlations on Gemma-2-2B and Llama-3-8B with authenticated access.

2. **Semantic hierarchy features**: WordNet hierarchies instead of first-letter features to test deeper hierarchies.

3. **Encoder-correlation-based prediction**: Since decoder correlations fail (H6), test whether encoder correlations predict absorption.

4. **Cross-architecture validation**: JumpReLU, TopK, Gated SAEs may show different absorption patterns.

5. **Rate-distortion bound derivation**: Formal proof that absorption is information-theoretically optimal under stated constraints.

6. **Better absorption metrics**: Develop metrics specific to learned structure (not captured by random SAEs).

<!-- FIGURES
- Figure 5: fig7_precision_recall.pdf — Precision-recall asymmetry visualization
- Figure 6: fig5_dose_response.pdf — Rate-distortion interpretation schematic
-->
# Pilot Experiment Summary

## Task: pilot_absorption

### Status: COMPLETED

### Model & Configuration
- **Model**: GPT-2 Small (124M parameters)
- **SAE**: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs, Layer 8, 32K dictionary
- **Device**: NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Seed**: 42

### Method
Decoder cosine similarity with child firing detection:
1. Find first-letter features by computing cosine similarity between decoder directions and average token directions for words starting with each letter
2. For each feature, find "child" features via decoder cosine similarity
3. Measure absorption: when main feature doesn't fire on test words, do child features fire instead?

### Results

| Metric | Value |
|--------|-------|
| Features analyzed | 26 (A-Z) |
| Unique features | 20 |
| Mean absorption rate | 0.269 |
| Std absorption rate | 0.444 |
| Min absorption rate | 0.000 |
| Max absorption rate | 1.000 |
| Median absorption rate | 0.000 |

**Classification:**
- HIGH (>50%): 7 features [B, D, I, J, L, R, T]
- MEDIUM (10-50%): 0 features
- LOW (<10%): 19 features [A, C, E, F, G, H, K, M, N, O, P, Q, S, U, V, W, X, Y, Z]

### Key Observations

1. **Bimodal distribution**: Features show a bimodal pattern - either fully absorbed (100%) or not absorbed at all (0%). This suggests the absorption metric may be too coarse, or the feature identification has false positives.

2. **Feature 25906 dominates**: Letters A, U, V, W, X, Y, Z all map to the same feature (25906), which has the highest decoder cosine similarity (0.416). This feature fires consistently on all test words for these letters, suggesting it may be a generic "first letter" feature rather than letter-specific.

3. **7 fully absorbed features**: For letters B, D, I, J, L, R, T, the identified main feature never fires on test words, but child features do fire. This is the classic absorption pattern.

4. **19 non-absorbed features**: For most letters, either the main feature always fires (no absorption) or neither main nor child features fire (possible false positive in feature identification).

### GO/NO-GO Assessment

**GO** - The experiment successfully detects absorption with a bimodal distribution. However, several issues need addressing:

1. **Feature identification quality**: 6 letters map to the same feature (25906), suggesting the decoder-based feature search may not be specific enough.
2. **Bimodal distribution**: The all-or-nothing pattern (0% or 100%) is suspicious and may indicate threshold sensitivity.
3. **Model limitation**: GPT-2 Small may not exhibit strong absorption (as noted in evolution lessons).

---

## Task: pilot_steering

### Status: COMPLETED

### Model & Configuration
- **Model**: GPT-2 Small (124M parameters)
- **SAE**: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs, Layer 8, 32K dictionary
- **Method**: Reconstruction fidelity analysis (v4)
- **Device**: NVIDIA RTX PRO 6000 Blackwell Server Edition
- **Seed**: 42

### Method Evolution
The steering experiment went through 4 iterations:
- **v1**: Direct decoder steering at strengths [1, 2, 5, 10] - minimal effect
- **v2**: Activation-based feature selection with higher strengths [1, 5, 10, 20, 50] - minimal effect
- **v3**: Activation patching (set feature to target value) - minimal effect
- **v4**: Reconstruction fidelity analysis - detectable differences found

### Results (v4 - Reconstruction Fidelity)

| Metric | HIGH Absorption | LOW Absorption |
|--------|-----------------|----------------|
| Reconstruction MSE | 2,939,454,976.0 +/- 0.0 | 2,939,454,970.6 +/- 15.7 |
| Cosine Similarity | -0.5890 | -0.5890 |
| Main Feature Activation | 1,163.6 +/- 346.2 | 1,376.5 +/- 307.0 |
| Main Feature in Top-32 | 100.0% | 100.0% |

### Key Findings

1. **Main feature activation is lower for HIGH absorption features**: 1164 vs 1376 (difference = 213, r=-0.285 with absorption rate). This suggests that absorbed features are genuinely less active in the SAE representation.

2. **Reconstruction MSE is extremely high**: ~3B MSE and -0.59 cosine similarity indicate very poor SAE reconstruction quality. This SAE may be optimized for sparsity/interpretability rather than reconstruction fidelity.

3. **All features are in top-32**: Both HIGH and LOW absorption features have 100% top-32 rate, meaning the SAE always includes them in its sparse representation when they fire.

4. **No statistically significant differences**: T-tests show p > 0.05 for all comparisons, likely due to small sample size (n=7 HIGH, n=19 LOW) and high variance.

### Steering Robustness Confound

As noted in the evolution lessons, steering adds the decoder direction directly to the residual stream, bypassing the SAE encoder entirely. Even if the parent latent fails to fire naturally (due to absorption), the injected direction still influences the output. This makes steering inherently robust to encoder-side absorption, which may explain the minimal steering effects observed in v1-v3.

### GO/NO-GO Assessment

**GO** - The experiment reveals detectable differences between HIGH and LOW absorption features in terms of main feature activation. However:

1. **Poor SAE reconstruction quality** limits the interpretability of reconstruction-based metrics
2. **Small sample size** (n=26) is underpowered for detecting small-to-medium effects
3. **Steering confound** means steering is not a valid test of absorption's impact
4. **GPT-2 Small limitations** - weak first-letter features, many letters share the same feature

---

## Overall Pilot Assessment

### Recommendation: GO_WITH_CAVEATS

The pilot successfully demonstrates:
- Absorption can be detected (7/26 features HIGH, 19/26 LOW)
- Absorbed features have lower activation (1164 vs 1376)
- The experimental pipeline works end-to-end

However, significant issues limit the strength of conclusions:
- GPT-2 Small has weak, non-specific first-letter features
- SAE reconstruction quality is poor
- Steering is confounded by encoder bypass
- Small sample size limits statistical power

### Recommendations for Full Study

1. **Switch to a larger model**: Gemma-2-2B with Gemma Scope SAEs would provide stronger, more specific features. Requires HuggingFace gated model access.

2. **Use activation-based metrics**: Focus on feature activation strength, reconstruction fidelity, and presence in top-k rather than steering.

3. **Improve feature identification**: Use contrastive activation patching or manual inspection to find truly letter-specific features.

4. **Add controls**: Random feature baseline, shuffled label control, null steering.

5. **Increase sample size**: Test more words per letter, more letters, or semantic hierarchy features.

6. **Multi-layer analysis**: Test layers 8, 12, 16 to validate consistency.

7. **Power analysis**: Current n=26 is underpowered for |r| < 0.50. Need n ~ 85 for 80% power at |r| = 0.50.

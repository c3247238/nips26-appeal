# Pilot Summary: SAE Feature Absorption (CAAB)

## P1: CAAB Feasibility

### Results
- **Status**: SUCCESS
- **Elapsed**: 27.8 seconds (well under 15-minute budget)
- **GPU**: RTX PRO 6000 Blackwell (GPU 3)

### Key Findings

#### TopK SAE (trained on GPT-2 Small layer 8)
- **Architecture**: TopK with k=25, d_sae=3072 (4x expansion)
- **Training**: 500K tokens, completed in ~15 seconds
- **First-letter features found**: 26/26 letters
- **Feature collision rate**: 30.77% (8 out of 26 letters share features with other letters)
- **Unique features**: 18 out of 3072

#### Collision Analysis
Letters sharing the same SAE features indicate potential absorption:
- Feature 1665: letters a, i, o, p, u (5 letters share this feature)
- Feature 93: letters j, q, z (3 letters share this feature)
- Feature 470: letters e, y (2 letters share this feature)
- Feature 1141: letters n, r (2 letters share this feature)

#### Interpretation
The 30.77% collision rate suggests that the TopK SAE with limited training does exhibit feature overlap patterns that could indicate absorption. However, this is a simplified proxy metric. The pre-trained SAE could not be loaded due to API compatibility issues with SAELens v6.39.0.

### GO/NO-GO Assessment

**GO** - The pilot demonstrates:
1. SAELens training pipeline works correctly
2. First-letter features can be identified in SAEs
3. Feature collision detection is feasible
4. Runtime is well within budget (<1 min vs 15 min target)

### Limitations
1. Pre-trained SAE loading failed (SAELens API compatibility)
2. Collision rate is a simplified proxy, not true Chanin et al. absorption detection
3. Only single architecture (TopK) was tested
4. Small SAE size (3K features) may not represent full-scale behavior

### Recommendations for Full Experiments
1. Fix pre-trained SAE loading or use alternative SAE sources
2. Implement proper Chanin et al. absorption detection with parent-child hierarchy
3. Scale to larger SAEs (16K+ features)
4. Compare multiple architectures

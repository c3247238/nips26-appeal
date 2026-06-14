# Critique: Experimental Design and Execution

## Summary

The experimental design is reasonable but execution has critical flaws. The multiple comparison correction was properly applied (credit to the authors) but the results are presented incorrectly throughout the paper. Key experiments (H6, OrtSAE ablation) have methodological issues that undermine conclusions.

## Critical Issues

### 1. Multiple Comparisons Correction Not Reflected in Writing
**The facts**:
- 12 statistical tests performed
- Bonferroni-corrected alpha = 0.00417
- BH-FDR q < 0.05 threshold
- **Zero significant results after correction**

**The problem**: The paper presents H1b at layer 8 (p=0.028 uncorrected) as evidence throughout:
- Abstract: Implies significant relationship exists
- Section 4.4: "r = -0.431, p = 0.028" without immediately noting it does not survive correction
- Conclusion: Cites H1b as demonstrating "absorption-specific effect"

After proper correction:
- H1b Bonferroni p = 0.334 (not significant)
- H1b BH-FDR q = 0.107 (not significant)

**Required fix**: Every presentation of H1b results must prominently note the correction status.

### 2. H6 Inhibition Graph Predicts Nothing
**Design**: Construct local inhibition graph from decoder correlations, test whether top-k neighbors correspond to absorption pairs.

**Result**: Precision@20 = 0.0. Of 520 predictions (26 features x 20 neighbors), ZERO corresponded to known absorption pairs. Fisher exact test p = 1.0 (no enrichment over chance).

**Interpretation problem**: The paper frames this as "informative negative result" but then continues to rely on the LCA framework throughout. The logical inconsistency is:
- If LCA competitive suppression explains absorption, the graph should predict absorption pairs
- The graph predicts nothing
- Therefore, the LCA framework (as operationalized) does not explain absorption

**Required fix**: Either (a) drop LCA claims entirely and acknowledge the mechanism is not validated, or (b) provide alternative empirical support for the LCA mechanism that does not depend on H6.

### 3. OrtSAE Ablation at Unmatched L0
**Design**: Compare OrtSAE with orthogonality penalty (L0~550) vs without penalty (L0~920).

**Problem**: The paper criticizes unmatched comparisons throughout but performs one:
- L0~550 vs L0~920 is a ~40% difference in sparsity
- L0 is a key determinant of absorption rate
- Comparing absorption at different L0 values is comparing different sparsity levels, not different penalties

**Conclusion**: "The orthogonality penalty does not appear to reduce absorption" is invalid given the L0 mismatch.

**Required fix**: Match L0 values before comparing, or acknowledge this comparison cannot support the stated conclusion.

### 4. Random SAE Baseline Construction Underspecified
**Current specification**: "frozen orthonormal decoder, random encoder"

**Missing details**:
- Encoder initialization distribution (Gaussian? Uniform? Xavier?)
- Whether encoder was trained before freezing (it was not, but this should be explicit)
- Orthonormalization method (QR decomposition? SVD? Gram-Schmidt?)
- Random seed value

**Why this matters**: Different random encoder initializations could yield substantially different absorption rates. Without a fixed seed, this experiment cannot be reproduced.

**Required fix**: Provide exact initialization code and random seed.

### 5. MCC Failure for Random SAE Not Interpreted
**Observation**: MCC~0.21 for random SAE when running Hungarian matching to recover ground truth absorption pairs.

**Implication not addressed**: If matching yields MCC~0.21 on a random SAE (where we know the ground truth), then:
- The matching procedure may not work correctly
- All MCC comparisons are suspect
- The trained vs random comparison may be comparing metric noise, not real differences

**Required fix**: Acknowledge this limitation explicitly: "The MCC failure for random baseline suggests Hungarian matching may not reliably recover absorption pairs, limiting interpretation of MCC-based comparisons."

### 6. Beta-Conditional Effects Not Flagged
**Observation**: The absorption-steering correlation is only observable at high steering magnitudes (beta=10, 20). At beta=5 (median), zero difference.

**Problem**: The paper presents this as a "robust" finding without noting the conditional nature.

**Required fix**: State clearly: "The absorption-steering correlation is only statistically detectable at steering magnitudes beta >= 10. At beta=5 (median), no correlation is observed."

### 7. Post-Hoc Power Analysis
**Location**: Section 3.6 (approximately)

**Problem**: The paper computes power after seeing results (post-hoc), which is methodologically questionable. Power should be computed a-priori to determine sample size.

**Required fix**: Remove post-hoc power analysis. If discussing power, cite a-priori calculation that justified the sample size.

## Moderate Issues

### 8. H9 Layer-Dependent Structure (n=4)
**Design**: Compare mean edge weight across 4 layers (0, 4, 8, 10).

**Problem**: With n=4 data points, statistical inference is not meaningful. r=0.82 is descriptive, not inferential.

**Required fix**: Either add more layers (0, 2, 4, 6, 8, 10, 12) or downgrade to "descriptive observation" without statistical language.

### 9. Dead Feature Ratio Not Addressed
**Observation**: 89-99% dead feature ratio across experiments.

**Problem**: The analysis does not stratify by dead vs active features. Conclusions about absorption may not apply to active features if dead features have different properties.

**Required fix**: Acknowledge as limitation. Consider stratifying analysis by L0 to determine if dead features behave differently.

### 10. Sample Size Justification Absent
**Problem**: n=26 features (A-Z) is small for correlation analysis. No justification for this sample size is provided.

**Required fix**: Provide power calculation a-priori: "With n=26, we have 80% power to detect r>0.5 at alpha=0.05. Effects smaller than this are not reliably detectable."

## Minor Issues

### 11. H1b Delta Correction Applied Inconsistently
**Observation**: H1b uses delta-corrected steering (feature-specific minus random baseline) but H1 uses raw steering.

**Question**: Why delta correction for H1b but not H1? If delta correction is necessary to isolate absorption effects, it should be applied consistently.

**Required fix**: Explain why delta correction is applied selectively.

### 12. Cross-Model Validation Incomplete
**Observation**: Pythia-70M cross-model validation was attempted but is described as "inconclusive; limited feature overlap."

**Problem**: Without successful cross-model validation, claims may not generalize beyond GPT-2 Small.

**Required fix**: Either complete the cross-model validation or explicitly frame findings as GPT-2 Small specific.
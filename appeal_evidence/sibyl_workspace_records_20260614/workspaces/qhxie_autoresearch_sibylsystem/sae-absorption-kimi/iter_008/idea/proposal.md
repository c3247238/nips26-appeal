# Research Proposal: Iteration 8

## Title
L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders

## Core Research Questions
- **RQ1**: Does natural L0 confound cross-architecture absorption comparisons?
- **RQ2**: Does absorption rate causally predict downstream interpretability performance?

## Key Improvements from Iteration 7
1. **Fix dead latent reporting**: Convert raw counts to percentages (TopK ~82%, Matryoshka ~62%)
2. **Debug explained_variance calculation for OrtSAE**: Custom OrthogonalitySAE class incompatible with eval function's activation scaler
3. **Address MCC metric limitation**: Acknowledge that MCC measures feature recovery (decoder alignment), not downstream performance; supplement with reconstruction MSE which shows clear training effect
4. **Add statistical tests**: Welch's t-test, Cohen's d, Pearson correlation with CI
5. **Remove duplicate MultiScale variant from analysis** (already verified: multiscale and matryoshka are NOT identical)
6. **Verify training actually occurred**: Add loss curve logging

## Methodology
1. Re-analyze existing experimental data with corrected metrics
2. Fix OrtSAE evaluation code to properly compute explained_variance
3. Add comprehensive statistical analysis
4. Update paper with corrected data and honest metric limitations discussion

## Expected Contributions
1. First systematic demonstration that L0 is the dominant driver of absorption
2. Proof that Baseline L1 cannot match sparsity levels of TopK/Matryoshka
3. Dose-response study falsifying causal absorption-interpretability link
4. Critical methodological contribution: L0-matching protocol and its limitations

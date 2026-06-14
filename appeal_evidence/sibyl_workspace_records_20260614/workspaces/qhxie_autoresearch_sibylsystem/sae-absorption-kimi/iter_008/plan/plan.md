# Experiment Plan: Iteration 8

## Objective
Fix critical data quality issues identified in iteration 7 review and re-analyze existing experimental data.

## Tasks

### Task 1: Fix Data Analysis Pipeline (No GPU needed)
- Fix dead latent percentage calculation in analysis script
- Add proper statistical tests (Welch's t-test, Cohen's d, Pearson correlation with CI)
- Re-analyze all existing experimental data
- Generate corrected Table 1 and Table 2

### Task 2: Fix OrtSAE Evaluation (GPU needed)
- Debug explained_variance calculation for custom OrthogonalitySAE class
- The issue: eval_sae_on_synthetic_data uses activation_scaler, but OrtSAE training does not
- Fix: Ensure consistent scaling or bypass scaler for OrtSAE evaluation
- Re-run OrtSAE evaluation only (5 seeds)

### Task 3: Add Loss Curve Logging (No GPU needed)
- Modify run_full.py to log training loss per batch
- Verify training actually occurs (not cached/shortcut)

### Task 4: Generate Statistical Analysis (No GPU needed)
- Compute ANOVA across all variants
- Pairwise t-tests with Bonferroni correction
- L0-absorption Pearson correlation
- Effect sizes (Cohen's d) for all comparisons

## Configuration
- Use existing experimental data where possible
- Only re-run OrtSAE evaluation (not full retraining)
- Training samples: 2M (unchanged)
- Seeds: 42, 123, 456, 789, 1011 (unchanged)
- Expected time: ~30 minutes for OrtSAE re-evaluation + analysis

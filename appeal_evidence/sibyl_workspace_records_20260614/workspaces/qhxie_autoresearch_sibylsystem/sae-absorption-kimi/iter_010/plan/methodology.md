# Methodology: Iteration 10

## Objective
Run corrected pilot experiments with improved data quality and sanity checks.

## Key Improvements from Iteration 9
1. **Dead latent percentages**: Report `dead_latents_pct` (percentage) instead of raw counts
2. **Sanity checks**: Automatic post-experiment validation:
   - explained_variance in [-1, 1]
   - training_time > 10s for 2M tokens
   - dead_latents_pct < 50% for healthy dictionaries
   - MCC variation > 0.01 across conditions (checked in analysis)
3. **Loss curve logging**: Extract and save first 100 loss points for convergence verification
4. **Statistical tests**: Welch's t-test, Cohen's d, ANOVA in analysis phase

## Experimental Design
- Synthetic data: 1024 features, 256 hidden dim, 32 root nodes, branching factor 4, depth 3
- Training: 2M tokens, batch size 1024, seed 42
- Variants: Baseline ReLU (L1), TopK (k=50), MultiScale (Matryoshka), Random control
- Metrics: absorption_rate, MCC, reconstruction_MSE, explained_variance, L0, dead_latents_pct

## Analysis Plan
1. Cross-variant comparison with Welch's t-test
2. Effect sizes (Cohen's d)
3. Sanity check validation
4. GO/NO-GO decision based on metric discrimination and convergence

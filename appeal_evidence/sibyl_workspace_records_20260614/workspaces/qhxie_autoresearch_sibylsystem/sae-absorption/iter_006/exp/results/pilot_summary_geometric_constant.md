# Pilot Summary: Geometric Constant Validation

## Task
Compute c(w_P, w_C) = ||w_P||^2 * (1 - cos^2(w_P, w_C)) from SAE decoder weights for all 25 first-letter parent-child pairs. Test whether CMI/c correlates more strongly with absorption rate than raw CMI.

## Key Finding: Unit-Normalized Decoder Weights

**Gemma Scope SAE decoder weights are unit-normalized** (||w|| = 1.0 for all features). This has a critical consequence for the geometric constant:

- c(w_P, w_C) = ||w_P||^2 * (1 - cos^2) = 1.0 * (1 - cos^2) = **sin^2(angle(w_P, w_C))**
- The norm component ||w_P||^2 contributes zero variability
- The geometric constant reduces to a pure angular measure

## Results

| Metric | Spearman rho | p-value |
|--------|-------------|---------|
| Raw CMI vs absorption | -0.383 | 0.059 |
| CMI/c(mean) vs absorption | -0.375 | 0.065 |
| CMI/c(median) vs absorption | -0.385 | 0.057 |
| c alone vs absorption | 0.171 | 0.414 |
| Mean cosine vs absorption | -0.184 | 0.378 |

## Pass Criteria Assessment

| Criterion | Result | Status |
|-----------|--------|--------|
| c computed for all 25 letters | 25/25 | PASS |
| CMI/c correlation computed | Yes | PASS |
| Comparison with raw CMI reported | Yes | PASS |
| c modulates beyond CMI | No (-2.2%) | FAIL |

## Why c Does Not Improve Over CMI

1. **Near-constant c values**: mean=0.960, std=0.021, CV=2.16%
2. **Near-orthogonal directions**: mean cos(w_P, w_C) = 0.17, so sin^2 ~ 0.97 for all pairs
3. Dividing by a near-constant (~0.96) preserves rank order but adds no information
4. Bootstrap 95% CI for |rho(CMI/c)| - |rho(CMI)|: [-0.113, +0.085] (includes zero)

## Theoretical Implication

For **normalized SAEs** (which includes Gemma Scope and most modern SAEs), the rate-distortion absorption threshold lambda/c(w_P, w_C) is dominated by lambda and CMI, not by decoder geometry. The geometric constant provides negligible additional modulation because:

1. Unit normalization removes ||w_P||^2 variability
2. SAE training encourages feature decorrelation, making parent-child pairs nearly orthogonal
3. This leaves only CMI as the effective predictor of absorption

This is a **theoretically informative negative result**: it constrains the rate-distortion theory by showing that the geometric term in the absorption criterion degenerates for practical SAE architectures. The CMI alone captures the relevant information-theoretic structure.

## Recommendation

**GO** (for reporting): This is a clean result that constrains the theory. Report as: "The geometric constant c degenerates for normalized SAEs, reducing the rate-distortion criterion to a pure CMI-based diagnostic." This simplifies the theoretical framework (which is a positive for the paper) and should be reported in the methodology/discussion section.

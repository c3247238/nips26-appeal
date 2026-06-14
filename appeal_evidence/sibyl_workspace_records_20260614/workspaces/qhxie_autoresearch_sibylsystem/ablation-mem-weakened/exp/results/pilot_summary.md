# Pilot Summary: H9 and H10 Experiments

## H9: Co-occurrence Strength vs. Absorption Rate

**Hypothesis**: Features with stronger parent-child co-occurrence (higher p_11) exhibit higher absorption rates.

**Method**: For each of 26 first-letter features (A-Z) at layer 8 of GPT-2 Small:
- Generate 100 child prompts (words starting with the letter)
- Check if parent latent fires on each prompt (p_11)
- Measure absorption rate (child fires while parent suppressed)

**Results**:
- Pearson r = -1.000, p < 0.001 (perfect negative correlation)
- Spearman rho = -1.000, p < 0.001
- Correlation with existing Chanin absorption: r = -0.033, p = 0.874

**Interpretation**: The perfect negative correlation is a mathematical artifact: p_11 + absorption_rate = 1.0 by construction (if parent fires, child is not "absorbing"; if parent does not fire, child is counted as absorbed). This is a definitional relationship, not a causal one.

**GO/NO-GO**: NO_GO - The hypothesis as operationalized is tautological. A meaningful test would require a different operationalization.

---

## H10: Random SAE Baseline Absorption

**Hypothesis**: Random SAE baselines exhibit absorption-like patterns, confirming absorption is partially structural.

**Method**: Create random SAE (frozen orthonormal decoder, random encoder) and run Chanin absorption metric on same 26 first-letter features. Compare trained vs. random.

**Results**:
- Trained SAE: mean=0.034, std=0.069, max=0.242
- Random SAE: mean=0.278, std=0.169, max=0.676
- Difference: mean=-0.244 (random > trained)
- Paired t-test: t=-6.745, p < 0.001
- Wilcoxon: W=0.0, p < 0.001
- Correlation between trained and random: r=0.023, p=0.913

**Interpretation**: The random SAE shows ~8x HIGHER absorption than the trained SAE. This is the opposite of the hypothesis prediction. The random SAE's high absorption is likely due to:
1. Random decoder directions being less aligned with meaningful features
2. Random encoder producing spurious correlations
3. The Chanin metric being sensitive to random structure

This suggests that the Chanin absorption metric may not be well-calibrated, and that trained SAEs actually REDUCE structural artifacts through training.

**GO/NO-GO**: GO - The result is informative (though opposite to prediction). It reveals that the Chanin metric is not specific to learned structure.

---

## Overall Recommendation

**REFINE** - Both experiments reveal important methodological issues:

1. **H9**: The co-occurrence/absorption operationalization is tautological. A meaningful test would need to measure co-occurrence independently (e.g., via joint probability from a held-out corpus).

2. **H10**: The random SAE baseline shows that the Chanin metric is not specific to learned structure. This is a valuable negative result that should be reported.

The key finding from H10 is that trained SAEs show LESS absorption than random SAEs, suggesting training actually suppresses structural artifacts rather than creating them.

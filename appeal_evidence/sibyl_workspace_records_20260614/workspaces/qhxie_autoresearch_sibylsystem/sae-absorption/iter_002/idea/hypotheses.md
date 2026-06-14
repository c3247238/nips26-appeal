# Testable Hypotheses

## Primary Hypotheses

### H1: Rate-Distortion Absorption Threshold

**Statement:** For a parent feature p and child feature c in a trained SAE, absorption of p into c is the rate-distortion optimal behavior when the sparsity penalty lambda exceeds sin^2(theta_{p,c}), where theta_{p,c} is the angle between their decoder directions:

```
Absorption(p, c) is energetically preferred iff  lambda > sin^2(theta_{p,c})
```

The co-occurrence frequency p_co cancels from this expression: absorption risk depends only on decoder geometry and sparsity penalty, not on how often the parent and child co-occur.

**Expected outcome:** Absorbed feature pairs have significantly smaller decoder angle (higher cosine similarity) than non-absorbed pairs (Wilcoxon rank-sum, Cohen's d ≥ 0.5). AUROC ≥ 0.70 when using the threshold to predict which feature pairs are absorbed from the Chanin et al. ground truth.

**Mechanism:** When lambda > sin^2(theta), the L0 savings from absorbing one activation (cost = lambda) exceed the reconstruction cost from decoder misalignment (cost = sin^2(theta) × p_co per co-occurrence), and p_co cancels. Hence absorption is the unique energetically preferred solution under flat sparsity penalties.

**Key measurement:** For each absorbed and non-absorbed feature pair identified by the Chanin et al. metric, compute cos^2(theta_{p,c}). Test AUROC of threshold classifier on held-out SAE configurations.

**Falsification:** AUROC < 0.65 after fitting the threshold to one SAE and predicting absorption in held-out SAE configurations.

---

### H2: Cross-Domain Generality of Absorption

**Statement:** Absorption occurs at statistically detectable rates (permutation test p < 0.01 after Bonferroni correction for 8 comparisons: 4 hierarchies × 2 SAE widths) across at least 3 of 4 tested hierarchy types:
- First-letter spelling (baseline)
- Entity-type (city→country from RAVEL)
- Geographic (city→continent from RAVEL)
- Grammatical (POS→subtype, e.g., noun→proper noun)

**Expected outcome:** Ratio-to-shuffled-null ≥ 3.0 for at least 3 hierarchy types. Absorption rates for entity-type and geographic hierarchies fall in the range [5%, 50%]. Shuffled-label controls show absorption rates < 5% for all hierarchy types.

**Key measurement:** For each hierarchy type, compute mean absorption rate across the 16k and 65k Gemma Scope SAEs at layer 12. Report ratio-to-shuffled-null with 95% bootstrap CI (1000 resamples).

**Falsification:** If ALL non-spelling hierarchy types show ratio-to-shuffled-null < 1.5 (indistinguishable from noise), absorption is specific to syntactic/artificial hierarchies and does not generalize to semantic domains.

---

### H3: Absorption Susceptibility Index (ASI) Validity

**Statement:** The Absorption Susceptibility Index:
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
computed from SAE decoder weights and activation frequencies alone (no probe training required), predicts absorbed feature pairs with AUROC ≥ 0.70 against the Chanin et al. ground truth on the first-letter task.

**Expected outcome:** Among the top-k pairs ranked by ASI, precision ≥ 0.30 and recall ≥ 0.30 for the absorbed-pair class. AUPRC > 0.20 (well above the base rate of absorbed pairs). Permutation null (shuffled ASI scores) achieves AUPRC < 0.10, confirming the result is not a statistical artifact.

**Key measurement:** Compute pairwise ASI for all feature pairs with co-activation frequency > 0.01. Compare ASI ranking against the Chanin et al. absorption labels. Compare against ablated versions: (a) cosine similarity only; (b) frequency ratio only; (c) random ranking.

**Falsification:** AUROC < 0.60 (at or near chance level given the imbalanced class distribution) means decoder geometry has no predictive value for absorption.

---

### H4: Absorption Onset Phase Dynamics

**H4a (Phase transition):** Absorption rate, plotted as a function of effective sparsity pressure (1/L0 for TopK SAEs), exhibits a rapid transition at a critical L0_c — identifiable as a sigmoid-like crossover rather than a linear increase. Specifically: a sigmoid functional form fits absorption-vs-sparsity data significantly better than a linear model (likelihood ratio test, p < 0.05).

**H4b (Hysteresis):** When a SAE trained at high sparsity (absorption present) is fine-tuned with reduced sparsity, the absorption rate decreases by at most 50% of the gap between the original rate and the rate of a SAE trained from scratch at the reduced sparsity.

**Expected outcome for H4a:** Sigmoid model R^2 > 0.85 vs. linear model R^2 < 0.70 on the absorption-vs-sparsity data across Gemma Scope SAE configurations.

**Expected outcome for H4b:** Fine-tuned SAE absorption rate remains ≥ 70% of the original high-sparsity absorption rate, while a from-scratch SAE at the same target L0 achieves absorption rate ≤ 30% of the high-sparsity rate. This would confirm hysteresis: the absorbed state is metastable.

**Falsification for H4a:** Linear fit R^2 ≥ 0.90 (absorption increases uniformly with sparsity, no phase transition).

**Falsification for H4b:** Fine-tuned absorption rate drops below 50% of the original rate within the fine-tuning budget (absorption is fully reversible, no hysteresis).

---

## Secondary Hypotheses

### H5: Architectural Mitigations Increase Effective Decoder Angles

**Statement:** Matryoshka SAE and OrtSAE reduce absorption by mechanisms interpretable through the rate-distortion threshold: Matryoshka increases the effective decoder angle between hierarchically related features (by constraining inner-level features to cover general concepts), and OrtSAE does so directly via the orthogonality penalty.

**Expected outcome:** Matryoshka and OrtSAE SAEs have significantly larger mean cos^2(theta_{p,c}) for known absorbed feature pairs compared to TopK SAEs at matched width — i.e., the mitigations push feature pairs below the absorption threshold.

**Key measurement:** For the first-letter hierarchy, compute decoder cosine similarity between known absorbed pairs across Matryoshka, OrtSAE, and TopK SAEs.

**Falsification:** No significant difference in decoder angles between architectures despite significant difference in absorption rates — would imply the mitigations work via a different mechanism than the rate-distortion theory predicts.

### H6: ASI Predicts Cross-Domain Absorption Rates

**Statement:** The cross-domain absorption rate of a given hierarchy type is predictable from the mean ASI of feature pairs in that hierarchy (higher mean ASI → higher absorption rate).

**Expected outcome:** Spearman correlation between mean ASI and absorption rate across hierarchy types is rho ≥ 0.5 (directional claim given the small number of hierarchy types).

**Key measurement:** Compute mean ASI for each hierarchy type's probe-identified feature pairs. Correlate with measured absorption rate.

**Falsification:** Mean ASI does not rank hierarchy types in the same order as absorption rates — would suggest ASI is hierarchy-specific and not a universal predictor.

---

## Summary Table

| Hypothesis | Prediction | Success Metric | Falsification |
|---|---|---|---|
| H1: RD threshold | Absorbed pairs have higher cos^2(theta); threshold AUROC ≥ 0.70 | AUROC ≥ 0.70 | AUROC < 0.65 |
| H1 corollary | co-occurrence frequency cancels from threshold | Frequency-stratified AUROC stable | Large difference in AUROC by frequency bin |
| H2: Cross-domain | Absorption detectable in ≥ 3 of 4 hierarchy types | Ratio-to-null ≥ 3.0, p < 0.01 Bonf. | All non-spelling types indistinguishable from null |
| H3: ASI probe-free | Decoder geometry predicts absorption | AUROC ≥ 0.70, AUPRC > permutation null | AUROC < 0.60 |
| H4a: Phase transition | Sigmoid fit better than linear | LRT p < 0.05, sigmoid R^2 > 0.85 | Linear R^2 ≥ 0.90 |
| H4b: Hysteresis | Fine-tuning does not reverse absorption | Rate > 70% of original after FT | Rate drops below 50% |
| H5: Mitigation mechanism | Mitigations increase decoder angles between hierarchical pairs | Significant angle increase for Matryoshka vs. TopK | No angle difference |
| H6: ASI cross-domain | Mean ASI predicts cross-domain absorption | Spearman rho ≥ 0.5 | rho < 0.3 |

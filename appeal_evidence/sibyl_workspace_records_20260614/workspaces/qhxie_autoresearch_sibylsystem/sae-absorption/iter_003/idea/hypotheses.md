# Testable Hypotheses (Iteration 2 — Synthesizer)

## Status Overview

This is a fresh hypothesis set for Iteration 2. All hypotheses are pre-registered before any experiments are run in this iteration.

| Hypothesis | Direction | Falsification Criterion | Priority |
|------------|-----------|------------------------|----------|
| H1 (ARS Unsupervised Detection) | ARS AUROC > 0.70 vs. Chanin labels | AUROC < 0.55 on Gemma Scope 16k L12 | 1 — Primary |
| H2 (Amortization Gap Dominant) | OMP absorption rate < 50% of feedforward | OMP >= 80% of feedforward rate | 1 — Primary |
| H3 (Cross-Hierarchy Generalization) | Entity-type absorption > negative control by > 10 pp | p > 0.1 Bonferroni AND ρ-Spearman < 0.2 | 1 — Primary |
| H4 (Phase Transition in ρ) | Step-like increase in absorption rate near ρ* | R² < 0.15 for step-function fit | 2 — Supplementary |
| H5 (Safety Attribution) | >= 20% false-negatives absorption-attributable | < 10% AND random latent shows same improvement | 3 — Supplementary |

---

## H1: Absorption Risk Score as Unsupervised Detector

**Full statement:**
The Absorption Risk Score ARS(i,j) = O(i,j) × A(i,j) × cos²(d_i, d_j), where O is Jaccard activation overlap, A is activation frequency asymmetry, and cos²(d_i, d_j) is squared decoder cosine similarity, achieves AUROC > 0.70 against Chanin et al. canonical absorption labels on Gemma Scope 16k at layer 12 of Gemma-2-2B.

**Operationalization:**
- Ground truth: Chanin et al. integrated-gradients absorption labels for letters A, E, I, O, T on Gemma Scope L12-16k
- Predicted signal: Absorption Fingerprint Score AFS(i) = max_{j: f_j > 3×f_i} ARS(i,j)
- Threshold for AUROC: primary metric AUROC; secondary metrics AUPR and precision at top-k
- Component decomposition: compare AUROC of O(i,j) alone, A(i,j) alone, cos²(d_i, d_j) alone vs. full ARS product to establish additive value of each component

**Pilot criterion (go/no-go at 10 min):**
If top-10 high-ARS pairs contain >= 5 Chanin-labeled absorbed pairs (out of expected ~26 × absorption_rate × 16k absorbed features), proceed to full validation. If top-10 high-ARS pairs contain < 2 absorbed pairs, diagnosis and possible component reweighting before full run.

**Theoretical motivation:**
The ecological competitive exclusion principle formally maps to the L1-penalized SAE loss: O(i,j) captures niche overlap (which tokens both latents compete for), A(i,j) captures competitive asymmetry (which latent has carrying-capacity advantage from higher frequency), and cos²(d_i, d_j) captures interspecific competition strength (how much the more-frequent latent's decoder direction conflicts with the rarer latent's direction). The product combines all three factors into a single "absorption risk" score.

**Relation to previous iteration:**
EDA (iter_001) detected absorption via encoder-decoder angular divergence — theoretically expected to have signal only for "late absorption" (decoder-present, encoder-suppressed, ~13% of cases). ARS operates on co-occurrence statistics, which are observable regardless of decoder geometry, and should therefore provide signal for "early absorption" (decoder-absent, ~75% of cases). The two detectors are theoretically complementary.

**What would falsify H1:**
AUROC < 0.55 on Gemma Scope 16k layer 12 (no better than chance), AND high-ARS pairs are not enriched for Chanin-labeled absorbed features (enrichment ratio < 1.2). Both conditions must hold simultaneously — partial failure (e.g., AUROC = 0.62 with significant enrichment) would motivate recalibration of the ARS product, not full abandonment.

---

## H2: Amortization Gap Controlled Dictionary Experiment

**Full statement:**
When the pre-trained decoder dictionary of Gemma Scope L12-16k is fixed and encoding method is switched from standard feedforward to Orthogonal Matching Pursuit (OMP) at matched L0, absorption rate is reduced by >= 50% relative to feedforward encoding if the amortization gap is the primary driver, or is unchanged (< 20% reduction) if the sparsity optimization landscape is the primary driver.

**Operationalization:**
- Control: Pre-trained decoder matrix W_dec from Gemma Scope L12-16k (frozen, identical across conditions)
- Condition A (Feedforward): Standard SAE encoder produces activation coefficients
- Condition B (OMP): Orthogonal Matching Pursuit on W_dec at K matched to mean L0 of Condition A
- Condition C (2-pass): Feedforward + one residual correction pass
- Measurement: Chanin et al. absorption metric adapted to each encoding condition's activation patterns (same probe directions, same thresholds, same ground-truth labels)
- Statistical test: Paired Wilcoxon signed-rank test on per-letter absorption rates (A vs. B, A vs. C)

**Pre-committed decision rule:**
- If (AR_A - AR_B) / AR_A >= 50%: amortization gap is the dominant driver → recommend iterative encoding for all absorption-sensitive SAE applications immediately
- If (AR_A - AR_B) / AR_A < 20%: sparsity landscape is the dominant driver → encoder changes are insufficient; hierarchically-aware training is the necessary fix
- If 20% ≤ reduction < 50%: mixed causes; both interventions are partially effective

**Subtype stratification:**
- Run analysis separately for early-absorbed latents (decoder-absent, from iter_001 taxonomy) vs. late-absorbed latents (decoder-present)
- Prediction: if amortization gap dominates early absorption, OMP should show especially large reduction for early-type cases; if sparsity landscape dominates, reduction should be uniform across types

**What would falsify the amortization-gap-dominance hypothesis:**
OMP absorption rate >= 80% of feedforward rate on the same dictionary (< 20% reduction), AND the 2-pass encoder also shows < 20% reduction. This would confirm the sparsity landscape explanation and recommend against investing in encoder improvements for absorption reduction.

**Prior literature connection:**
- O'Neill et al. (arXiv:2411.13117) proves theoretically that a linear-nonlinear encoder has a provably non-zero amortization gap. This experiment tests whether that gap is large enough to explain most absorption.
- Tang et al. (arXiv:2512.05534) argues absorption is a stable spurious minimum of the piecewise biconvex loss. This experiment tests whether the minimum is reachable with a better encoder or only avoidable through loss landscape changes.
- MP-SAE (Costa et al., arXiv:2506.03093) reduces absorption via iterative encoding, but retrains jointly — cannot distinguish encoder vs. dictionary effects. Our experiment controls for the dictionary.

---

## H3: Cross-Hierarchy Absorption Generalization

**Full statement:**
Absorption rate in entity-type hierarchies (ANIMAL ⊃ specific animals) is significantly higher than in matched-frequency-ratio non-hierarchical pairs (paired t-test, p < 0.05, Bonferroni-corrected), and frequency ratio ρ = p_child/p_parent is the strongest single predictor of absorption rate across all four hierarchy types (Spearman ρ > 0.40 after controlling for hierarchy type).

**Hierarchy types tested:**
1. **First-letter (baseline):** Chanin et al. direct replication — alphabetical hierarchy where ground truth is known
2. **Entity-type:** ANIMAL ⊃ {dog, cat, bird, fish, horse} — semantically richer, less symmetrically distributed frequency ratio
3. **Syntactic:** PAST-TENSE ⊃ IRREGULAR-PAST-TENSE — purely syntactic hierarchy
4. **Negative control:** "mentions weather" AND "mentions time of day" — co-occurring but non-hierarchical, matched on ρ to Hierarchy 2

**Probe quality gate (pre-specified):**
- LR probe accuracy for entity-type: must achieve > 80% at some layer (tested at layers 8, 12, 16, 20)
- LR probe accuracy for syntactic: must achieve > 75%
- If probe accuracy fails: replace with a simpler hierarchy (capitalized word ⊃ first word in sentence)

**Statistical design:**
- Fixed SAE: Gemma Scope 16k at layer 12 (controlled across all hierarchy types)
- Minimum 3 probe training seeds per hierarchy type
- Primary test: paired t-test between entity-type and negative control absorption rates (matched on ρ)
- Secondary test: Spearman ρ between frequency ratio and absorption rate across all positive hierarchies
- Effect size reporting: Cohen's d alongside p-values

**Contrarian's insight explicitly tested:**
The contrarian perspective argues that first-letter absorption rates may be systematically lower than semantic hierarchy rates because the first-letter task has artificially balanced frequency ratios (each letter covers ~3-5% of vocabulary), while real semantic hierarchies are far more imbalanced (ANIMAL >> specific animal). H3 directly tests this by comparing absorption rates across hierarchies with different ρ values.

**What would falsify H3:**
All four hierarchy types show similar absorption rates (including negative control), AND frequency ratio ρ does not predict absorption across hierarchy types (Spearman ρ < 0.2). This would suggest the absorption metric is not hierarchy-specific and may be detecting generic feature unreliability rather than hierarchy-driven failure.

---

## H4: Absorption Phase Transition in Frequency Ratio

**Full statement:**
The absorption rate curve as a function of parent-child frequency ratio ρ (measured from the 26-letter natural sweep in the first-letter task) shows a step-like increase near a threshold ρ* that is predictable from the Critical Frequency Ratio Theorem: ρ*(k, p₁, λ) ∝ 1 - λk / (p₁ · const_geometry · (1-ε²)), where ε is the decoder cosine similarity between parent and child latents.

**Operationalization:**
- Compute ρ for each of the 26 letters from OpenWebText corpus token statistics (letter frequency / mean token-starting-with-letter frequency)
- Measure absorption rate for each letter from the Chanin et al. sae-spelling metric on Gemma Scope L12-16k
- Fit a step-function model (absorption = a for ρ < ρ*, b for ρ ≥ ρ*) and compare R² to linear model
- Compute theoretical ρ* from SAE parameters (k = mean L0, p₁ = mean parent letter frequency, λ from L1 weight, ε from decoder cosine similarity between identified parent-child latent pairs)
- Compare observed ρ* (fitted) to theoretical ρ* (computed)

**Expected finding:**
- Letters with ρ > ρ*_theoretical should show absorption rates in the 25-40% range
- Letters with ρ < ρ*_theoretical should show absorption rates in the 5-15% range
- The step-function fit should have R² > linear fit, indicating threshold structure

**What would falsify H4:**
Flat relationship between ρ and absorption rate (Spearman ρ < 0.3, R² < 0.15 for both step-function and linear fits). This would suggest absorption rate is not primarily driven by frequency ratio and the phase transition prediction does not hold empirically.

**Theoretical grounding:**
The Critical Frequency Ratio Theorem (derived by the Theoretical perspective, building on Chanin et al. 2024 and Tang et al. 2512.05534) predicts a phase transition between non-absorbing and absorbing optimization minima as a function of ρ. This is the first empirical test of whether the transition exists and where it falls relative to the theoretical prediction.

---

## H5: Safety Attribution Analysis

**Full statement:**
At least 20% of the cases where the SAE probe fails but the dense linear probe succeeds (false-negative rate on harmful intent classification) can be attributed to feature absorption. Specifically: for each false-negative token, identifying the absorbing latent via the Chanin metric and manually adding the absorbed parent latent to the active set will convert that token from false-negative to true-positive. The improvement should be specific (not replicated by adding a random high-frequency latent as control).

**Operationalization:**
- Dataset: AdvBench (200 harmful + 200 harmless) and ToxiGen (200 toxic + 200 non-toxic)
- Extract Gemma-2-2B layer 12 residual stream activations + Gemma Scope 16k SAE latent activations
- Train: (a) dense LR probe on residual stream, (b) 1-sparse SAE probe, (c) k-sparse SAE ensemble (k=1,5,10,20)
- Identify false-negative tokens: dense probe says harmful, SAE probe says harmless
- For each false-negative token: run absorption metric on the "harmful intent" feature direction (identified by linear probe); identify absorbing latents
- Intervention test: add absorbed parent latent to active set; re-run SAE probe; report change in classification
- Control test: add random high-frequency latent (not identified by absorption analysis) to active set; verify no improvement

**Statistical design:**
- McNemar test for paired binary classification (harmful vs. harmless): SAE probe vs. linear probe gap
- Bootstrap 95% CI on absorption-attributable fraction
- Two-dataset validation (AdvBench + ToxiGen) for generalizability

**What would falsify H5:**
Absorption-attributable fraction < 10% on both datasets, AND random latent addition provides equivalent improvement to absorbed-parent latent addition (specificity test fails). This would indicate that absorption is NOT the primary cause of the SAE-vs-linear probe gap, pointing to alternative explanations (dark matter, polysemanticity, etc.).

**Connection to contrarian's concern:**
The contrarian perspective correctly notes that DeepMind's harmful intent failure was "about half explained by corpus mismatch" even after controlling for co-occurrence. This experiment provides the first direct absorption attribution measurement for this specific gap, avoiding the confound of corpus mismatch by using a fixed model and SAE.

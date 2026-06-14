# Backup Ideas and Pivot Alternatives

## Overview

The front-runner proposal ("Competitive Geometry of Absorption") integrates the LV unsupervised detector (H1), corpus PMI predictor (H2), and downstream impact analysis (H3). Each component can succeed or fail independently. If the front-runner's core claims fail, the following alternatives represent the strongest pivots.

---

## Alternative A: The Absorption-Sparsity Phase Transition — A Formal Information-Theoretic Theory

**Source:** Theoretical perspective (Candidates A+B from theoretical agent)

**Core claim:** Feature absorption in SAEs exhibits a phase transition in (ρ, L0, q) space, where ρ = p_C/p_P is the parent-child frequency ratio, L0 is the sparsity budget, and q is the implication strength (P(C fires | P fires)). The information-theoretic absorption strength A(P, C) = I(f_P; f_C) / H(f_P) predicts absorption rate with Pearson r > 0.70.

**Why it's strong:**
- Directly extends Tang et al. (2024) Theorem 4.7 from existence (binary) to prevalence (quantitative)
- The MDL-rational absorption criterion (H(f_P | f_C) < L0_savings(P, C; λ)) is directly testable: it should correctly classify ≥ 80% of absorbed vs. non-absorbed pairs
- Explains why strict implication (q→1) guarantees absorption regardless of width or L0

**Pivot trigger:** If H1 (LV detector) fails (F1 < 0.50), the problem may be with the LV competition coefficient specifically. The information-theoretic quantity A(P, C) = I(f_P; f_C) / H(f_P) is a different path to the same destination (unsupervised absorption prediction) with complementary theory.

**Experimental implementation (training-free, ~1 hour):**
1. For each of 26 letter features on Gemma Scope 16k layer 12: extract activation frequency distributions for parent and child latents
2. Estimate I(f_P; f_C) from empirical activation frequencies using k-NN mutual information estimator (EDGE estimator)
3. Compute A(P, C) = I(f_P; f_C) / H(f_P) for each letter
4. Correlate A(P, C) against Chanin et al. absorption rates across 26 letters
5. Test MDL criterion: check whether H(f_P | f_C) < L0_savings holds for absorbed pairs with sensitivity ≥ 0.80

**Key risk:** Information-theoretic estimation is noisy for rare features (low-frequency parent latents). Mitigation: focus on letters with activation counts > 100 in the test corpus.

---

## Alternative B: Absorption Scaling Laws — Empirical Iso-Absorption Curves for Practitioners

**Source:** Pragmatist perspective (Candidate A+B from pragmatist agent)

**Core claim:** Absorption rate is a predictable function of (L0, SAE width, layer) expressible as a log-linear model with R² > 0.70, yielding practical iso-absorption contour curves that tell practitioners which (L0, width) combinations achieve absorption rate < 10%.

**Why it's strong:**
- Directly actionable for SAE practitioners
- Builds entirely on pre-existing sae-spelling + SAELens + Gemma Scope — minimal new code
- Fills a clear community need (Gap 1 from literature: no quantitative prediction tool)
- The secondary cross-domain component (country-name, given-name hierarchies) tests generalization

**Pivot trigger:** If H2 (corpus PMI) fails and H1 (LV) partially fails, the pragmatist scaling law is the clearest path to a publication — even if the causal mechanism is unexplained, a working prediction formula is valuable.

**Experimental implementation (training-free, ~10 GPU-hours):**
1. Run sae-spelling on 30 Gemma Scope SAEs (3 widths × multiple L0 × 4 layers) — collect 780 (layer, width, L0, letter, absorption_rate) tuples
2. Fit: `absorption_rate = β₀ + β₁ log(L0) + β₂ log(width) + β₃ layer + ε`
3. Plot iso-absorption contours over (L0, width) space at fixed layer
4. Cross-domain: add 2 hierarchy domains (country-name tokens, given-name tokens)
5. Key claim: practitioners can now select SAE configurations to target a desired absorption threshold

**Key risk:** Chanin et al. Figure 7b already shows the qualitative trend. The regression formula must add value beyond the known trend — value comes from: (a) explicit quantitative coefficients, (b) iso-absorption curves, (c) cross-domain validation. If the model R² is low (< 0.50), this weakens the main claim but the iso-absorption framework remains useful exploratorily.

---

## Alternative C: Controlled Causal Audit — Does Anything Causally Drive Absorption?

**Source:** Empiricist perspective (Candidate A from empiricist agent)

**Core claim:** The commonly reported empirical relationships between SAE hyperparameters and absorption rate (L0 → absorption, width → absorption) are partially confounded by the L0/D ratio. The first controlled experiment with all confounders held fixed provides the first valid causal identification of absorption rate determinants.

**Why it's strong:**
- The methodological gap is real and acknowledged by the field (SAEBench notes, Chanin et al. limitations)
- Pre-registered predictions are strongly falsifiable
- Even if all predictions are wrong (confounds don't explain the trends), the methodology is stronger than anything published

**Pivot trigger:** If the Component 0 pilot shows L0 is not matched across widths, this alternative becomes the methodological foundation — the confounding is itself the finding.

**Experimental implementation (training-free, ~8 GPU-hours):**
1. Vary L0 at fixed width=16k, layer=12: L0 ∈ {10, 20, 50, 100, 200} — 5 SAEs
2. Vary width at matched L0=50 (if feasible per pilot): widths {1k, 4k, 16k, 65k, 131k} — 5 SAEs
3. Compare L1 vs. TopK at matched L0=50 — 2 SAEs
4. 3 random seeds at one configuration (if Gemma Scope provides)
5. ANOVA with L0 as covariate; bootstrap CIs; mixed-effects for letter clustering

---

## Alternative D: ATM SAE Replication — Testing an Extraordinary Claim

**Source:** Empiricist perspective (Candidate C from empiricist agent)

**Core claim:** ATM SAE's reported 20× absorption reduction (Chanin metric: 0.0068 vs. TopK 0.1402) is the most extraordinary result in the absorption literature and has not been independently replicated. Replicating or failing to replicate this is a high-value contribution regardless of outcome.

**Why it's strong:**
- Independent replication of extraordinary claims is always publishable
- The LessWrong community and SAEBench maintainers have not yet independently verified the ATM result
- If the result doesn't replicate on GPT-2 small, it is either model-specific or a metric artifact — both are important findings

**Pivot trigger:** If Component 3 (downstream analysis) reveals that absorption metric is decoupled from downstream tasks but ATM shows anomalously good downstream performance despite its absorption reduction, then ATM has discovered something genuinely useful that the absorption metric alone doesn't capture.

**Experimental implementation (depends on checkpoint availability):**
1. Download ATM SAE checkpoints for Gemma-2-2B (if available)
2. Run Chanin metric on two test sets: original distribution and adversarially constructed distribution with different PMI statistics
3. Compute decoder cosine similarity distribution: compare ATM vs. TopK at matched L0
4. If code available: train ATM-equivalent on GPT-2 small, compare absorption rates
5. Compare ATM vs. TopK on SAEBench RAVEL and SCR at matched L0

---

## Minimal Fallback: "Three Gaps and Three Measurements" Paper

If all four primary/alternative hypotheses fail empirically, the following minimal contribution remains publishable:

1. **Comprehensive absorption taxonomy** (Types I/II/III) showing true absorption rate substantially exceeds 15–35%
2. **LV competition coefficient as descriptive tool** — even if it doesn't achieve F1 ≥ 0.65 as a detector, showing it correlates with absorption rate at r > 0.50 is valuable
3. **SAEBench downstream correlation analysis** — whether it supports or falsifies H3, this is the first systematic test of the assumption motivating all absorption research

These three contributions together are sufficient for a short empirical paper at NeurIPS/ICLR, even without the causal claims.

---

## Priority Pivot Decision Tree

```
Front-runner fails?
├─ H1 (LV detector) fails (F1 < 0.50)
│   ├─ H2 (PMI) succeeds → Focus paper on corpus causal account + downstream analysis
│   └─ Both H1 and H2 fail → Pivot to Alternative A (information-theoretic theory)
│
├─ H2 (PMI) fails (non-significant)
│   ├─ H1 (LV) succeeds → Publish LV detector + downstream analysis as main contribution
│   └─ Both fail → Pivot to Alternative B (scaling laws) for practical contribution
│
├─ H3 is falsified (|r| > 0.3 — absorption predicts downstream)
│   → Celebrate: front-runner still publishes but with STRONGER motivation for H1+H2
│
└─ H4 (distributed absorption) fails
    → Width paradox remains unexplained; add L0/D confound analysis from Alternative C
      as the competing explanation; both are publishable findings
```

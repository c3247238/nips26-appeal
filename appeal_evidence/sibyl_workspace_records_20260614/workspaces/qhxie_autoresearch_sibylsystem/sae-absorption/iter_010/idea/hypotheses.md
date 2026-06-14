# Testable Hypotheses (Synthesis Round 7 -- Iteration 10)

## Summary Table

| Hypothesis | Status | Metric | Full-Mode Target | Falsification | Phase | GPU-hours |
|-----------|--------|--------|-----------------|---------------|-------|-----------|
| H1 (Cross-domain variation) | SUPPORTED | Absorption rate difference | Within-RAVEL p<0.01, vs first-letter p<0.05 Bonferroni | Rates within 5 pp across all types | 0.2 (write) | 0 |
| H2' (Semantic > Syntactic at L24) | SUPPORTED | Semantic - Syntactic absorption | Semantic >= Syntactic by >= 5 pp at L24 | Reversed after probe quality matching (H10) | 0.2 (write) | 0 |
| H3 (Hedging decomposition) | SUPPORTED | Hedging ratio by hierarchy | Ratio varies by hierarchy (>10 pp) | No variation across hierarchies | 0.2 (write) | 0 |
| H4 (GAS detector) | REFUTED | Spearman rho | N/A (negative result) | N/A | 3.1 (appendix) | 0 |
| H5 (Absorption tax) | NOT_SUPPORTED | Absorption-MSE correlation | Qualitative direction only | Already partially falsified | 3.1 (appendix) | 0 |
| H6 (Architecture generalization) | INCONCLUSIVE | Architecture ranking consistency | Report effect sizes | No consistent ranking detectable | 0.4 (reframe) | 0 |
| H7 (Causal absorption) | SUPPORTED UNIVERSALLY | Activation patching recovery | d>0.5, p<0.01 all hierarchies | Recovery = control (p>0.05) | 0.2 (write) | 0 |
| H8 (Decoder direction magnitude) | SUPPORTED (reframed) | Logit change on direction ablation | 3.98 nats > control (0.12) | Logit change = control | 2.1 | 0.5 |
| H9 (Rate-distortion predictors) | UNTESTED | Spearman rho of 3-factor model | rho > 0.5 | rho < 0.3 or p > 0.05 | 2.2 | 1 |
| **H10 (Probe degradation -- NEW)** | **UNTESTED** | **Absorption rate at degraded F1** | **First-letter at F1=0.80 differs from RAVEL by >5 pp** | **Rates match within 5 pp** | **1.1** | **2** |

---

## H1: Cross-Domain Absorption Variation -- SUPPORTED

**Statement:** Absorption rates on entity-attribute hierarchies differ significantly from first-letter spelling rates on the same SAEs.

**Full-mode evidence:**
- Within-RAVEL: Kruskal-Wallis p=7.4e-66 (3 hierarchies)
- City-language vs. first-letter: p_Bonf=0.003 at L24 16k -- statistically significant
- City-continent vs. first-letter: p_Bonf=1.0 -- NOT significant pairwise
- City-country vs. first-letter: p_Bonf=1.0 -- NOT significant pairwise (F1=0.73 confound)
- Descriptive range: 11.6% (city-language) to 45.1% (city-country)

**This iteration's task:** Propagate corrected data to paper. Distinguish descriptive from inferential claims. Within-RAVEL variation is strong; vs. first-letter, only city-language survives Bonferroni.

**Falsification:** Rates within 5 pp across ALL hierarchy types after probe quality matching (H10).

---

## H2': Semantic Hierarchies Show Higher Absorption at L24 -- SUPPORTED

**Statement:** Semantic/knowledge hierarchies show HIGHER absorption than the syntactic first-letter task at layer 24.

**Evidence:**
- L24 16k: first-letter 27.1%, city-country 45.1%, city-continent 42.9%, city-language 11.6%
- All probes F1 >= 0.96 at L24 (quality gate met)
- At L12: first-letter 3.9% with binary probe F1=0.083 -- too poor to compare

**Key caveat:** City-language (11.6%) is LOWER than first-letter (27.1%), so the "semantic > syntactic" claim holds for city-country and city-continent but not universally.

**Falsification:** After probe degradation ablation (H10), first-letter absorption at degraded F1 matches RAVEL rates, suggesting the variation is entirely probe-driven.

---

## H3: Absorption-Hedging Decomposition Varies by Hierarchy -- SUPPORTED

**Statement:** The absorption-to-hedging ratio varies systematically by hierarchy type.

**Evidence:**
- Strict hedging: 0-22.6% across hierarchies (vs. 92.6% loose classification)
- Compensatory resolution dominates the bulk (85.3%)
- Three-way decomposition (strict/compensatory/persistent) with bootstrap CIs is a reusable methodological contribution

**This iteration's task:** Ensure correct data propagated to paper. The strict/loose hedging finding is independently publishable.

**Falsification:** Absorption-hedging ratio identical across all hierarchy types (within 10 pp).

---

## H4: GAS Detector -- REFUTED (Negative Result)

**Statement (original):** GAS achieves Spearman rho >= 0.6 with probe-based absorption rate.

**Result:** rho = 0.12 (p=0.56). AUROC = 0.571.

**Status:** Negative result. Document in appendix with failure analysis:
1. GAS conflates absorption pairs with legitimately correlated features
2. Rare features produce noisy GAS scores
3. Static geometric signals may be inherently insufficient -- absorption requires interventional signals

---

## H5: Absorption Tax -- NOT SUPPORTED (Quantitative)

**Statement:** Higher absorption correlates with lower MSE at matched L0.

**Result:**
- T(G) = 0.414 computed
- Absorption-MSE correlation: rho = 0.08 (p=0.87)
- Ranking rho = -0.20, concordance 50%

**Status:** Quantitative predictions NOT SUPPORTED. Report as qualitative framework in appendix.

---

## H6: Architecture Generalization -- INCONCLUSIVE

**Statement:** Architecture rankings for absorption resistance are hierarchy-dependent.

**Evidence:**

| Hierarchy | JumpReLU | BatchTopK | Matryoshka |
|-----------|----------|-----------|------------|
| first-letter | 3.2% | 12.8% | 10.3% |
| city-continent | 17.1% | 23.2% | 34.2% |
| city-language | 13.9% | 31.9% | 20.5% |

**This iteration's reframing:**
- "We could not detect an architecture effect with the available data, though the test had limited statistical power (12 observations, 3 architectures x 4 hierarchies)."
- Report effect sizes regardless of significance.
- Matryoshka 32k vs JumpReLU 16k/65k width confound acknowledged.

**Falsification:** All architectures show identical absorption rates (within 5 pp). Already partially falsified: JumpReLU consistently lowest.

---

## H7: Causal Absorption via Activation Patching -- SUPPORTED UNIVERSALLY

**Statement:** Zeroing a child feature's SAE latent recovers the parent feature's probe prediction, confirming competitive exclusion.

**FULL-mode corrected evidence:**
- First-letter: d=1.33, p=0.000218 (n sufficient)
- City-continent: d=1.50, p<1e-20 (FULL-mode corrected from buggy pilot d=-0.91)
- City-language: d=0.75, p<1e-18 (FULL-mode corrected)

**This is the paper's strongest causal finding.** The universal mechanism (competitive exclusion confirmed across ALL hierarchy types) is stronger than the original pilot's domain-specific narrative.

**This iteration's task:** Propagate corrected FULL-mode data to paper. Replace "concentrated vs. distributed" dichotomy with "universal competitive exclusion."

**Falsification:** Recovery rate indistinguishable from control (p > 0.05) with n > 50. Already falsified.

---

## H8: Decoder Direction Magnitude -- SUPPORTED (Reframed from Benign/Pathological)

**Original statement:** >= 30% of absorption instances are benign (parent computationally redundant).

**Result:** 0% benign (3.98 nats vs. 0.12 control). All 1,471 instances from city-continent at L24 show large logit changes.

**Reframing (circularity acknowledged):**
- The diagnostic measures whether the child decoder carries significant parent-direction information that contributes to model output.
- It does NOT measure whether the model independently needs the parent feature.
- The child decoder IS known to carry parent information (that IS absorption); removing it will always produce large logit changes.
- The 3.98 nats result establishes magnitude, not pathology.

**This iteration's tasks:**
1. Reframe in paper with circularity acknowledgment
2. Replicate on first-letter at L24 (Step 2.1) to test cross-domain consistency
3. Describe what a genuine computational-redundancy test would require

**Falsification:** Logit change indistinguishable from random-direction control. Already falsified (3.98 vs. 0.12).

---

## H9: Rate-Distortion Three-Factor Predictor -- UNTESTED

**Statement:** Per-pair absorption probability is predicted by:
- cos_sim(decoder_parent, decoder_child): geometric overlap
- P(child active | parent active): co-occurrence frequency
- R(parent): reconstruction importance

**Method:**
1. From Phase 1 data, identify all parent-child pairs
2. Compute three predictors per pair
3. Fit: absorption_probability ~ beta_1 * cos_sim^2 + beta_2 * P(child|parent) - beta_3 * R(parent)
4. Spearman rho between predicted and observed absorption

**Target:** rho > 0.5 across pooled data from all hierarchies.

**Falsification:** rho < 0.3 or p > 0.05. (Consistent with H5 quantitative failure, this may fail.)

**Estimation:** ~1 GPU-hour for computation.

---

## H10: Probe Degradation Ablation -- NEW, HIGHEST PRIORITY

**Statement:** Cross-domain variation in absorption rates is a genuine hierarchy effect, not a probe quality artifact. Specifically: degrading first-letter probe quality from F1=1.0 to F1=0.80 (matching typical RAVEL probe quality) will NOT reproduce RAVEL-level absorption rates.

**Rationale:** The core uncertainty in the paper is whether the cross-domain variation (first-letter 27.1% vs. city-country 45.1% at L24) is driven by genuine differences in hierarchy structure or merely by differences in probe quality (first-letter F1=1.0, city-country F1=0.73-0.87). If degrading the first-letter probe to F1=0.80 causes its absorption rate to jump to ~45%, the cross-domain finding is an artifact. If it stays near 27-30%, the hierarchy effect is genuine.

**Method:**
1. Load trained first-letter probes at L24 (F1=1.0)
2. Inject label noise to degrade test-set F1 to {0.70, 0.80, 0.85, 0.90}
3. Re-run absorption measurement pipeline at L24 with 16k SAE at each degraded level
4. Plot absorption rate vs. probe F1
5. Compare: does the curve intersect with RAVEL absorption rates at matched probe quality?

**Expected outcome (if genuine hierarchy effect):** First-letter absorption rate increases modestly with probe degradation (27% -> maybe 30-35% at F1=0.80) but remains well below city-country (45.1%) and city-continent (42.9%) at matched quality. The gap persists.

**Expected outcome (if probe artifact):** First-letter absorption rate increases dramatically with probe degradation (27% -> ~40-45% at F1=0.80), closely matching RAVEL rates. The cross-domain variation collapses.

**Both outcomes are publishable:**
- Genuine effect: strengthens the core cross-domain finding
- Probe artifact: important methodological caution for the field (probe quality must be matched before comparing absorption rates across tasks)

**Falsification:** Degraded first-letter rates at F1=0.80 fall within 5 pp of city-language/city-country rates at equivalent probe quality.

**Estimation:** ~2 GPU-hours (4 degradation levels x ~30 min each).

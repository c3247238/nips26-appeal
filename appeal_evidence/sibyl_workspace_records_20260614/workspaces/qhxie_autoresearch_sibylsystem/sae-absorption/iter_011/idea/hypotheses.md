# Testable Hypotheses (Synthesis Round 8 -- Iteration 11)

## Summary Table

| Hypothesis | Status | Full-Mode Target | Falsification | Phase | GPU-hours |
|-----------|--------|-----------------|---------------|-------|-----------|
| H1 (Cross-domain variation) | SUPPORTED | Within-RAVEL p<0.01, vs first-letter p<0.05 Bonferroni | Rates within 5 pp across all types | 0.2 (write) | 0 |
| H2' (Semantic > Syntactic at L24) | PARTIALLY SUPPORTED | Semantic >= Syntactic by >= 5 pp at L24 | Reversed after probe quality matching | 0.2 (write) | 0 |
| H3 (Hedging decomposition) | SUPPORTED | Ratio varies by hierarchy (>10 pp) | No variation across hierarchies | 0.2 (write) | 0 |
| H4 (GAS detector) | REFUTED | N/A (negative result) | N/A | Appendix | 0 |
| H5 (Absorption tax) | NOT SUPPORTED | Qualitative direction only | Already partially falsified | Appendix | 0 |
| H6 (Architecture generalization) | INCONCLUSIVE | Report effect sizes | No consistent ranking detectable | 0.4 (reframe) | 0 |
| H7 (Causal absorption) | SUPPORTED UNIVERSALLY | d>0.5, p<0.01 all hierarchies | Recovery = control (p>0.05) | 1.1 (spot-check) | 0.5 |
| H8 (Decoder direction magnitude) | SUPPORTED (reframed) | 3.98/6.16 nats > control (0.12) | Logit change = control | Write | 0 |
| H9 (Rate-distortion predictors) | UNTESTED | Spearman rho > 0.5 | rho < 0.3 or p > 0.05 | 3.2 (optional) | 1 |
| H10 (Probe degradation) | SUPPORTED | R^2=0.777, rho=-1.0 | Rates match within 5 pp at matched F1 | 2.1 (reframe) | 0 |

---

## H1: Cross-Domain Absorption Variation -- SUPPORTED

**Statement:** Absorption rates on entity-attribute hierarchies differ significantly from first-letter spelling rates on the same SAEs.

**Full-mode evidence:**
- Within-RAVEL: Kruskal-Wallis p=7.4e-66 (3 hierarchies)
- City-language vs. first-letter: p_Bonf=0.003 at L24 16k -- statistically significant
- City-continent vs. first-letter: p_Bonf=1.0 -- NOT significant pairwise
- City-country vs. first-letter: p_Bonf=1.0 -- NOT significant pairwise (F1=0.73 confound)
- Descriptive range (quality-gated): 2.7x (31.4/11.6)

**This iteration's task:** Recompute with per-token aggregation (Phase 0.2). Verify significance holds. Distinguish descriptive from inferential claims throughout.

**Falsification:** Rates within 5 pp across ALL hierarchy types after probe quality matching (H10).

---

## H2': Semantic Hierarchies Show Higher Absorption at L24 -- PARTIALLY SUPPORTED

**Statement:** Semantic/knowledge hierarchies show HIGHER absorption than the syntactic first-letter task at layer 24.

**Evidence:**
- L24 16k: first-letter 27.1%, city-country 45.1%, city-continent 31.4%, city-language 11.6%
- Probe degradation (H10) explains city-continent rates (within 0.6 pp of curve)
- City-language is LOWER than first-letter, contradicting the blanket "semantic > syntactic" claim
- City-country has F1=0.73, below quality gate

**Key nuance:** The claim holds for city-country and city-continent but NOT universally. City-language (-21.3 pp residual after probe degradation) is the genuinely novel cross-domain finding and goes in the OPPOSITE direction.

**Falsification:** After probe degradation ablation, first-letter absorption at degraded F1 matches RAVEL rates (already partially occurred for city-continent).

---

## H3: Absorption-Hedging Decomposition Varies by Hierarchy -- SUPPORTED

**Statement:** The absorption-to-hedging ratio varies systematically by hierarchy type.

**Evidence:**
- Strict hedging: 0-22.6% across hierarchies (vs. 92.6% loose classification)
- Compensatory resolution dominates (85.3%)
- Three-way decomposition (strict/compensatory/persistent) with bootstrap CIs

**This iteration's task:** Ensure correct data propagated with per-token aggregation. The strict/loose finding is independently valuable.

**Falsification:** Absorption-hedging ratio identical across all hierarchy types (within 10 pp).

---

## H4: GAS Detector -- REFUTED (Negative Result)

**Statement (original):** GAS achieves Spearman rho >= 0.6 with probe-based absorption rate.

**Result:** rho = 0.12 (p=0.56). AUROC = 0.571.

**Status:** Negative result documented in appendix. Failure analysis: GAS conflates absorption pairs with legitimately correlated features; rare features produce noisy scores; static geometric signals are insufficient -- absorption requires interventional signals.

---

## H5: Absorption Tax -- NOT SUPPORTED (Quantitative)

**Statement:** Higher absorption correlates with lower MSE at matched L0.

**Result:** Absorption-MSE rho = 0.08 (p=0.87). Ranking rho = -0.20, concordance 50%.

**Status:** Quantitative predictions NOT SUPPORTED. Qualitative framework in appendix only.

---

## H6: Architecture Generalization -- INCONCLUSIVE

**Statement:** Architecture rankings for absorption resistance are hierarchy-dependent.

**Evidence:**

| Hierarchy | JumpReLU | BatchTopK | Matryoshka |
|-----------|----------|-----------|------------|
| first-letter | 3.2% | 12.8% | 10.3% |
| city-continent | 17.1% | 23.2% | 34.2% |
| city-language | 13.9% | 31.9% | 20.5% |

**Reframing:** "We could not detect a statistically significant architecture effect with the available data, though the test had limited statistical power (12 observations, 3 architectures x 4 hierarchies). JumpReLU showed consistently lowest absorption, but this is exploratory."

**Falsification:** All architectures show identical rates (within 5 pp). Already partially falsified: JumpReLU consistently lowest.

---

## H7: Causal Absorption via Activation Patching -- SUPPORTED UNIVERSALLY

**Statement:** Zeroing a child feature's SAE latent recovers the parent feature's probe prediction, confirming competitive exclusion.

**FULL-mode corrected evidence:**
- First-letter: d=1.33, p=0.000218
- City-continent: d=1.50, p<1e-20
- City-language: d=0.75, p<1e-18

**This is the paper's strongest causal finding.** Universal competitive exclusion confirmed across ALL tested hierarchy types.

**This iteration's task:** Spot-check 20 city-continent entities (Phase 1.1) to verify sign reversal correction. Re-run first-letter at L24 (Phase 1.3) for figure consistency. Document or execute city-country patching (Phase 1.2).

**Falsification:** Recovery rate indistinguishable from control (p > 0.05). Already falsified for all three hierarchies.

---

## H8: Decoder Direction Magnitude -- SUPPORTED (Reframed)

**Statement (reframed):** Child decoder vectors carry large-magnitude parent-direction information.

**Evidence:**
- City-continent: mean 3.98 nats (N=1,464/1,471), vs. 0.12 control
- First-letter: mean 6.16 nats (N=158), 100% above control threshold
- Cross-hierarchy consistency confirmed

**Circularity acknowledged:** The child decoder IS known to carry parent information (that IS absorption); removing it will always produce large logit changes. The measurement establishes magnitude, not computational independence. A genuine computational-redundancy test would require activation-level ablation or path patching through separate circuits.

**Falsification:** Logit change indistinguishable from random-direction control. Already falsified (3.98/6.16 vs. 0.12).

---

## H9: Rate-Distortion Three-Factor Predictor -- UNTESTED in full mode

**Statement:** Per-pair absorption probability is predicted by cos_sim(decoder_parent, decoder_child) x P(child|parent) / R(parent).

**Target:** Spearman rho > 0.5 across pooled data from all hierarchies.

**Falsification:** rho < 0.3 or p > 0.05.

**Status:** Optional Phase 3.2. May fail given H5's quantitative failure (rho=0.08). Both outcomes are informative.

---

## H10: Probe Degradation -- SUPPORTED

**Statement:** Cross-domain variation in absorption rates is partly a genuine hierarchy effect and partly a probe quality artifact. Probe quality is a major confound in absorption measurement.

**Evidence:**
- R^2=0.777 (linear fit, 7 F1 levels)
- Spearman rho=-1.0, p=0.009 (perfect monotonicity: lower F1 -> higher measured absorption)
- City-continent absorption rate falls on the first-letter degradation curve (within 0.6 pp)
- City-language is a genuine outlier: -21.3 pp residual (LOWER absorption than predicted by probe quality)

**Interpretation:**
- Probe quality IS a major confound (R^2=0.777 of variance explained)
- City-continent rates are EXPLAINED by probe quality degradation (not genuine hierarchy effect)
- City-language shows a genuine hierarchy-dependent anomaly (lower absorption than expected from probe quality)
- City-country excluded from comparison (F1=0.73, below quality gate)

**This iteration's task:** Reframe Section 4.6 to emphasize monotonicity over R^2. Hedge binary-to-multiclass extrapolation. Optionally run city-continent probe degradation (Phase 3.1).

**Falsification:** Already falsified for the "entirely probe artifact" direction -- city-language cannot be explained by probe quality.

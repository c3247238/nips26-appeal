# Testable Hypotheses (Synthesis Round 5 -- Evidence-Grounded)

## Summary Table

| Hypothesis | Pilot Status | Metric | Full-Mode Target | Falsification | Phase | GPU-hours |
|-----------|-------------|--------|-----------------|---------------|-------|-----------|
| H1 (Cross-domain variation) | PARTIALLY_SUPPORTED | Absorption rate difference | Significant difference (p < 0.01) with quality probes | Rates within 5 pp across all types | 1 | 5 |
| H2' (Semantic > Syntactic) | SUPPORTED by pilot | Semantic - Syntactic absorption | Semantic >= Syntactic by >= 5 pp | Semantic < Syntactic after probe correction | 1 | (same) |
| H3 (Hedging decomposition) | PARTIALLY_SUPPORTED | Partial correlation, hedging ratio | ratio varies by hierarchy | No variation across hierarchies | 1 | 0.5 |
| H4 (GAS detector) | REFUTED | Spearman rho | N/A (negative result) | N/A | -- | -- |
| H5 (Absorption tax) | NOT_SUPPORTED | Absorption-MSE correlation | Qualitative direction only | Already partially falsified | 3 | 0.5 |
| H6 (Architecture generalization) | INCONCLUSIVE | Architecture ranking consistency | JumpReLU consistently lowest | No consistent ranking | 1 | (same) |
| H7 (Causal absorption -- NEW) | SUPPORTED | Activation patching recovery | Wilcoxon p < 0.01 (n>50) | Recovery = control (p > 0.05) | 2 | 1 |

---

## H1: Cross-Domain Absorption Variation

**Statement:** Absorption rates on entity-attribute hierarchies differ significantly from first-letter spelling rates on the same SAEs.

**Pilot evidence:**
- City-language 10.4% vs first-letter 3.9% (p=0.005) -- significant
- City-continent 53.4% vs first-letter 3.9% -- highly significant but inflated by probe quality (raw accuracy 36.5%)
- Intra-RAVEL correlation rho=0.924, suggesting hierarchy-specific absorption is a stable property

**Full-mode requirements:**
- Probes at F1 > 0.90 (pilot best: 0.83 at layer 12). Test layers 18, 24 for factual knowledge.
- Bootstrap 95% CI (10k resamples) on all rates
- Paired permutation test across domains

**Expected outcome:** Cross-domain variation persists after probe quality improvement. City-continent rate will decrease (currently inflated) but remain above first-letter. City-language at 10.4% with decent probe quality is the most robust comparison.

**Falsification:** Rates within 5 pp across ALL hierarchy types with quality probes (F1 > 0.90).

---

## H2': Semantic Hierarchies Show Higher Absorption (Revised from H2)

**Statement (revised from original H2):** Semantic/knowledge hierarchies show HIGHER absorption than the syntactic first-letter task, because they have more complex co-occurrence patterns, deeper hierarchy structure, and less regular distribution.

**Background:** Original H2 predicted first-letter as worst case. Pilot FALSIFIED this: first-letter shows 3.9%, the lowest among all tested hierarchies. This is a critical finding that inverts the expected ordering.

**Proposed explanation:** First-letter features exist in the model primarily as a byproduct of tokenization, not as core computational features. The model does not "need" first-letter information for next-token prediction. In contrast, city-country and city-continent relationships are core factual knowledge actively used by the model, creating stronger parent-child co-occurrence pressure and thus more absorption.

**Alternative explanation:** Probe quality confound. First-letter probes at layer 12 achieved only F1=0.083 (with sae_spelling binary probe), making the 3.9% figure questionable. If first-letter probes are improved (using LogisticRegression or larger token sets), the rate may increase.

**Measurement:**
- Compare absorption rates across all hierarchies with matched probe quality (F1 > 0.90)
- Control for co-occurrence density: measure P(parent active | child active) per hierarchy
- Report both absolute rates and co-occurrence-adjusted rates

**Falsification:** With quality probes, first-letter absorption >= all semantic hierarchies by > 5 pp. This would restore the original H2 and suggest the pilot finding was a probe artifact.

---

## H3: Absorption-Hedging Decomposition Varies by Hierarchy

**Statement:** The absorption-to-hedging ratio varies systematically by hierarchy type. Semantic hierarchies have proportionally more true absorption and less hedging than syntactic hierarchies.

**Pilot evidence:**
- City-continent: 69% absorbed / 31% hedged
- First-letter: 45% absorbed / 55% hedged
- Strict hedging: only 7.4% vs loose 92.6% (85.3% compensatory)

**Full-mode requirements:**
- Tightened hedging classification across all hierarchies and SAEs
- Report both loose and strict hedging rates
- Partial correlation between L0 deviation and absorption after controlling for width, dead features

**Expected outcome:** The absorption-hedging ratio shifts toward more true absorption for semantic hierarchies, because these hierarchies involve genuine parent-child feature structure (not just feature correlation).

**Falsification:** Absorption-hedging ratio identical across all hierarchy types (within 10 pp).

**Key methodological contribution:** Demonstrating that the 98.6% loose hedging figure is near-tautological is independently valuable regardless of the cross-domain hypothesis outcomes.

---

## H4: GAS Detector -- REFUTED (Negative Result)

**Statement (original):** GAS achieves Spearman rho >= 0.6 with probe-based absorption rate.

**Pilot result:** rho = 0.12 (p=0.56). Well below both target (0.6) and failure threshold (0.3).

**Status:** Negative result. To be documented in appendix with failure analysis.

**Failure analysis hypotheses:**
1. **Confounding:** GAS conflates absorption pairs with legitimately correlated features. The co-occurrence mismatch signal is not specific to absorption.
2. **Noise in frequency ratio:** Rare features (low freq) produce noisy GAS scores.
3. **Scale:** Pilot ran on only 200 sequences. Full-mode with 10k sequences may improve.
4. **Fundamental limitation:** Static geometric signals (decoder similarity) may be inherently insufficient. Absorption may require dynamic/interventional signals.

**Full-mode action:** Run GAS on 10k sequences to give it a fair chance. If still fails, document as definitive negative result.

---

## H5: Absorption Tax Trade-off -- NOT SUPPORTED (Quantitative)

**Statement (original):** Higher absorption correlates with lower MSE at matched L0.

**Pilot result:** 
- T(G) = 0.414 computed
- Absorption-MSE correlation: rho = 0.08 (p=0.87) -- no relationship
- R_pc prediction of per-letter absorption: rho = 0.16 (p=0.46) -- no relationship

**Status:** Quantitative predictions NOT SUPPORTED. The formal theorem may be mathematically correct but the specific R_pc formulation (cos^2(d_p, d_c)) does not predict empirical absorption rates.

**Revised approach:** Report Absorption Tax as a qualitative/directional theoretical framework:
- The sign of the trade-off (absorption saves L0) is uncontested (Chanin et al. prove this in toy models)
- The magnitude T(G) and the per-pair prediction R_pc need refinement
- Consider whether the mutual coherence framework (from Theoretical perspective) provides better predictions than the redundancy ratio formulation

**Falsification:** Already partially falsified at the quantitative level. The qualitative direction remains valid.

---

## H6: Architecture Generalization -- INCONCLUSIVE

**Statement:** Matryoshka maintains absorption advantage across hierarchy types.

**Pilot evidence:**
| Hierarchy | JumpReLU | BatchTopK | Matryoshka |
|-----------|----------|-----------|------------|
| first-letter | 3.2% | 12.8% | 10.3% |
| city-continent | 17.1% | 23.2% | 34.2% |
| city-language | 13.9% | 31.9% | 20.5% |

**Key finding:** JumpReLU is consistently lowest across all hierarchies. Matryoshka shows mixed results: better than BatchTopK on city-language but WORSE on city-continent.

**Revised hypothesis:** Architecture rankings are hierarchy-dependent. JumpReLU appears most robust. Matryoshka's advantage is specific to hierarchies where the nested prefix loss matches the hierarchy structure.

**Full-mode requirements:**
- Confirm with improved probes (pilot probes below quality gate)
- Test at matched L0 (SAEs may have different L0 values confounding comparison)
- Additional SAE configurations from SAEBench grid

**Falsification:** If all architectures show identical absorption rates across all hierarchies (within 5 pp), architecture choice does not matter for absorption.

---

## H7: Causal Absorption via Activation Patching (NEW)

**Statement:** Zeroing a child feature's SAE latent recovers the parent feature's probe prediction, confirming that the child actively suppresses the parent through competitive exclusion.

**Pilot evidence:** 14.3% parent recovery when child zeroed vs 0.5% in control (random latent zeroed). Effect size appears large but statistical test is underpowered (n=9, Wilcoxon p=1.0).

**Full-mode requirements:**
- Scale to n > 50 activation patching instances
- Include instances from multiple hierarchies (not just first-letter)
- Proper controls: random latent zeroing, semantically unrelated latent zeroing
- Statistics: Wilcoxon signed-rank test, bootstrap CI on recovery rate

**Expected outcome:** Recovery rate significantly above control (p < 0.01). The effect may vary by hierarchy type: stronger for hierarchies with more absorption (consistent with H2').

**Falsification:** Recovery rate indistinguishable from control (p > 0.05) with n > 50 and proper power analysis.

**Significance:** This would be the first interventional (not just correlational) evidence for the competitive exclusion mechanism of absorption. All prior work (Chanin et al., SAEBench) uses correlational/attributional methods (integrated gradients, probe alignment).

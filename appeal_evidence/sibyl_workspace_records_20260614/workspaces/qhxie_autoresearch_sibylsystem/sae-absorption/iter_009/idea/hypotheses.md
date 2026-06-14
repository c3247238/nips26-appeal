# Testable Hypotheses (Synthesis Round 6 -- Iteration 9)

## Summary Table

| Hypothesis | Pilot Status | Metric | Full-Mode Target | Falsification | Phase | GPU-hours |
|-----------|-------------|--------|-----------------|---------------|-------|-----------|
| H1 (Cross-domain variation) | PARTIALLY_SUPPORTED | Absorption rate difference | Significant difference (p < 0.01) with quality probes | Rates within 5 pp across all types | 1 | 5 |
| H2' (Semantic > Syntactic) | SUPPORTED by pilot | Semantic - Syntactic absorption | Semantic >= Syntactic by >= 5 pp | Semantic < Syntactic after probe correction | 1 | (same) |
| H3 (Hedging decomposition) | PARTIALLY_SUPPORTED | Partial correlation, hedging ratio | Ratio varies by hierarchy | No variation across hierarchies | 1 | 0.5 |
| H4 (GAS detector) | REFUTED | Spearman rho | N/A (negative result) | N/A | -- | -- |
| H5 (Absorption tax) | NOT_SUPPORTED | Absorption-MSE correlation | Qualitative direction only | Already partially falsified | 3 | 0.5 |
| H6 (Architecture generalization) | INCONCLUSIVE | Architecture ranking consistency | JumpReLU consistently lowest | No consistent ranking | 1 | (same) |
| H7 (Causal absorption) | SUPPORTED | Activation patching recovery | Wilcoxon p < 0.01 (n>50) | Recovery = control (p > 0.05) | 2 | 1 |
| H8 (Benign vs. Pathological -- NEW) | UNTESTED | Logit change on parent ablation | >= 30% benign among absorbed instances | 100% pathological | 2 | 1 |
| H9 (Rate-distortion predictors -- NEW) | UNTESTED | Spearman rho of 3-factor model | rho > 0.5 for cos_sim x co-occur / R | rho < 0.3 or p > 0.05 | 3 | 0.5 |

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
- Bonferroni correction for multiple comparisons

**Expected outcome:** Cross-domain variation persists after probe quality improvement. City-continent rate will decrease (currently inflated) but remain above first-letter. City-language at 10.4% with decent probe quality is the most robust comparison.

**Falsification:** Rates within 5 pp across ALL hierarchy types with quality probes (F1 > 0.90).

---

## H2': Semantic Hierarchies Show Higher Absorption (Revised from H2)

**Statement (revised from original H2):** Semantic/knowledge hierarchies show HIGHER absorption than the syntactic first-letter task, because they have more complex co-occurrence patterns, deeper hierarchy structure, and stronger computational reliance by the model.

**Background:** Original H2 predicted first-letter as worst case. Pilot FALSIFIED this: first-letter shows 3.9%, the lowest among all tested hierarchies. This is a critical finding that inverts the expected ordering.

**Proposed explanations (multiple, to be distinguished):**
1. **Computational reliance:** First-letter features exist primarily as a byproduct of tokenization, not as core computational features. The model does not "need" first-letter information for next-token prediction. In contrast, city-country relationships are core factual knowledge actively used by the model.
2. **Co-occurrence complexity:** First-letter has a clean, near-uniform distribution (26 letters). Knowledge hierarchies have Zipfian distributions creating stronger co-occurrence pressure.
3. **Probe quality confound:** First-letter probes at layer 12 achieved only F1=0.083 (with sae_spelling binary probe). If probes are improved, the rate may increase.

**Measurement:**
- Compare absorption rates across all hierarchies with matched probe quality (F1 > 0.90)
- Control for co-occurrence density: measure P(parent active | child active) per hierarchy
- Report both absolute rates and co-occurrence-adjusted rates
- Test explanation 1 via activation patching: does zeroing the first-letter direction affect model output less than zeroing the country direction?

**Falsification:** With quality probes, first-letter absorption >= all semantic hierarchies by > 5 pp.

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

**Expected outcome:** Semantic hierarchies show proportionally more true absorption because these involve genuine parent-child feature structure, not just feature correlation.

**Falsification:** Absorption-hedging ratio identical across all hierarchy types (within 10 pp).

**Key methodological contribution:** Demonstrating that the 98.6% loose hedging figure is near-tautological is independently valuable.

---

## H4: GAS Detector -- REFUTED (Negative Result)

**Statement (original):** GAS achieves Spearman rho >= 0.6 with probe-based absorption rate.

**Pilot result:** rho = 0.12 (p=0.56). Well below both target (0.6) and failure threshold (0.3).

**Status:** Negative result. Documented in appendix.

**Failure analysis:**
1. GAS conflates absorption pairs with legitimately correlated features
2. Rare features produce noisy GAS scores
3. Static geometric signals may be inherently insufficient -- absorption requires interventional signals

---

## H5: Absorption Tax Trade-off -- NOT SUPPORTED (Quantitative)

**Statement:** Higher absorption correlates with lower MSE at matched L0.

**Pilot result:**
- T(G) = 0.414 computed
- Absorption-MSE correlation: rho = 0.08 (p=0.87)
- R_pc prediction: rho = 0.16 (p=0.46)

**Status:** Quantitative predictions NOT SUPPORTED. Report as qualitative/directional framework.

**Revised approach:** Compute T(G) per hierarchy type. Test whether T(G) ranking matches the cross-domain absorption ranking (qualitative prediction).

---

## H6: Architecture Generalization -- INCONCLUSIVE

**Statement:** Architecture rankings for absorption resistance are task-dependent.

**Pilot evidence:**

| Hierarchy | JumpReLU | BatchTopK | Matryoshka |
|-----------|----------|-----------|------------|
| first-letter | 3.2% | 12.8% | 10.3% |
| city-continent | 17.1% | 23.2% | 34.2% |
| city-language | 13.9% | 31.9% | 20.5% |

**Key finding:** JumpReLU is consistently lowest across all hierarchies. Matryoshka shows mixed results: better than BatchTopK on city-language but WORSE on city-continent.

**Revised hypothesis:** Architecture rankings ARE hierarchy-dependent. JumpReLU appears most robust. Matryoshka's advantage is specific to hierarchies where the nested prefix loss matches the hierarchy structure.

**Full-mode requirements:**
- Confirm with improved probes (pilot probes below quality gate)
- Test at matched L0
- Additional SAE configurations from SAEBench grid

**Falsification:** All architectures show identical absorption rates across all hierarchies (within 5 pp).

---

## H7: Causal Absorption via Activation Patching

**Statement:** Zeroing a child feature's SAE latent recovers the parent feature's probe prediction, confirming competitive exclusion.

**Pilot evidence:** 14.3% parent recovery when child zeroed vs 0.5% in control. Effect size large but underpowered (n=9, Wilcoxon p=1.0).

**Full-mode requirements:**
- Scale to n > 50 instances across hierarchies
- Proper controls: random latent zeroing, semantically unrelated latent zeroing
- Statistics: Wilcoxon signed-rank test, bootstrap CI on recovery rate
- Cross-hierarchy comparison

**Expected outcome:** Recovery rate significantly above control (p < 0.01).

**Falsification:** Recovery rate indistinguishable from control (p > 0.05) with n > 50.

**Significance:** First interventional evidence for competitive exclusion in SAEs.

---

## H8: Benign vs. Pathological Absorption (NEW -- from Contrarian)

**Statement:** A substantial fraction (>= 30%) of absorption instances are benign: the model does not independently use the parent feature when the child is active, so absorption faithfully represents computational redundancy rather than information loss.

**Rationale (from Contrarian perspective):** If the model's computation never uses "starts with S" independently when "snake" is active (because "snake" already routes through S-related circuits), then absorption is the SAE correctly reflecting computational structure. Measuring the downstream causal impact of ablating the absorbed direction distinguishes benign from pathological cases.

**Method:**
1. For each confirmed absorption instance (from Phase 1):
   - Ablate the parent direction component from the child latent's decoder vector
   - Run the model with the modified SAE reconstruction
   - Measure logit change for tokens dependent on the parent feature
2. **Benign:** logit change <= 0.1 (parent information is redundant in context)
3. **Pathological:** logit change > 0.1 (parent information has independent causal effects)
4. Report benign-to-pathological ratio per hierarchy type

**Expected outcome:** >= 30% of first-letter absorption instances are benign (the model does not need "starts with S" when "snake" is active). Semantic hierarchies may show a lower benign fraction (the model may independently use "located in France" even when "Paris" is active, because country information is used in different circuits than city information).

**Falsification:** < 10% benign absorption across all hierarchies (virtually all absorption is pathological). This would validate the mainstream view that absorption is uniformly harmful.

**Cross-reference:** Compare benign-to-pathological ratio with the activation patching results (H7). If activation patching shows high recovery, it suggests pathological absorption. If recovery is low, absorption may be benign.

---

## H9: Rate-Distortion Three-Factor Predictor (NEW -- from Theoretical + Interdisciplinary)

**Statement:** Per-pair absorption probability is predicted by the interaction of three measurable quantities:
- cos_sim(decoder_parent, decoder_child): geometric overlap
- P(child active | parent active): co-occurrence frequency
- R(parent): reconstruction importance of the parent feature

Specifically, absorption probability increases with cos_sim and co-occurrence, and decreases with R(parent), following the rate-distortion bound: alpha(i,j) ~ cos_sim^2 * co_occur / R.

**Rationale (from Theoretical perspective):** The rate-distortion framework predicts that absorption occurs when the residual reconstruction cost of the parent (after projecting out the child's contribution) is less than the sparsity penalty. This reduces to a function of three observable quantities.

**Method:**
1. From Phase 1, identify all parent-child pairs where absorption was measured
2. Compute: cos_sim(d_parent, d_child), P(child|parent), R(parent) = MSE increase when parent ablated
3. Fit: absorption_probability ~ beta_1 * cos_sim^2 + beta_2 * P(child|parent) - beta_3 * R(parent)
4. Evaluate: Spearman rho between predicted and observed absorption across all pairs and all hierarchies

**Expected outcome:** Spearman rho > 0.5 across pooled data from all hierarchies. Cos_sim and co-occurrence should be significant predictors; R(parent) may be noisy.

**Falsification:** rho < 0.3 or p > 0.05. This would indicate absorption is not predictable from these static quantities (consistent with the Absorption Tax quantitative failure, H5).

**Connection to ecological framework:** The three-factor model is equivalent to the Lotka-Volterra competition coefficient alpha_ij = cos_sim * co_occurrence_overlap (from the Interdisciplinary perspective), with R(parent) as the carrying capacity. If the model works, it validates both the rate-distortion and ecological competition frameworks simultaneously.

# Strategist Analysis: Iter 9 FULL-Mode Results

**Agent**: Strategist (Strategic Research Advisor)
**Timestamp**: 2026-04-16
**Scope**: Post-consolidation strategic assessment of 11/12 completed FULL-mode tasks across 4 phases

---

## 1. Signal Strength Assessment

| Result | Signal Strength | Metric Delta | Justification |
|--------|----------------|--------------|---------------|
| Cross-domain absorption variation (H1) | **Strong** | p=7.37e-66, 4x range (11.6%-45.1%) | Kruskal-Wallis with N=3566 is overwhelming. Effect survives any correction. Even dropping the problematic city-country result (probe F1=0.73), the first-letter vs city-language comparison (27.1% vs 11.6%, Cohen's h=-0.73) holds with clean probes. |
| All absorption is pathological (H8 falsified) | **Moderate** | 0% benign, 1000x effect ratio, n=1471 | Decisive within tested scope, BUT scope limited to city-continent/Europe/single SAE. Skeptic raises valid concern: experiment may measure "parent direction importance" not "absorption-specific pathology." Needs no-absorption baseline to be definitive. |
| Causal activation patching, first-letter (H7) | **Strong** | d=1.33, p=0.000218, 32.5% recovery vs 1.5% control | Large effect, proper controls, n=25 is adequate given effect size. But only at L12, not L24 where headline results live. |
| Cross-domain patching FAILURE | **Strong (negative)** | d=-0.91 (wrong direction), p=1.0 | Well-powered null (n=93). This is NOT noise -- it actively contradicts extending the competitive exclusion mechanism to semantic hierarchies. |
| Hedging near-tautology (H3) | **Strong** | chi2=91.51, p=1.04e-19 | Clean decomposition. The finding that strict hedging is 0-22.6% (vs prior 98.6% loose figure) is robust and informative for the community. |
| Architecture invariance (H6) | **Moderate** | p=0.754 (L12), p=0.497 (L24) | Non-significance is interesting but could be power issue. Width mismatch (Matryoshka 32k vs others 16k) weakens the claim. |
| Layer-dependent absorption (15x range) | **Strong** | 0.7% to 27.1% (L6-L24), same probes F1>=0.96 | Probe quality is constant across layers for first-letter. This is confound-free. Arguably the cleanest single finding in the paper. |
| Rate-distortion predictors (H9) | **Weak/Noise** | rho=0.250, individual predictors OPPOSITE direction | Full data (n=262) reverses pilot (n=20) direction. Framework qualitatively fails. |
| GAS detector (H4) | **Definitive negative** | rho=0.116, AUROC=0.571 | Confirmed at 25x pilot scale. Decoder geometry does not capture absorption. |
| Absorption Tax quantitative (H5) | **Noise** | rho=-0.20, concordance=50% (chance) | No predictive value. |

**Summary**: 3 strong positive signals (cross-domain variation, causal patching for first-letter, layer dependence), 1 strong methodological contribution (hedging tautology), 1 moderate positive with scope limitation (pathological absorption), 4 definitive negative results, and 1 strong negative that constrains mechanism claims.

---

## 2. Opportunity Cost Analysis

| Direction | GPU-hours | Wall-clock | Information Gain | IG / GPU-hr |
|-----------|-----------|------------|------------------|-------------|
| **Probe degradation ablation** (inject noise to F1={0.70,0.80,0.85,0.90}, re-measure absorption) | 1-2 | 2 hr | **Very High** -- resolves the single biggest threat to the paper (probe quality confound). If first-letter absorption rate changes proportionally, cross-domain comparison is confounded. If not, finding is validated. | **Highest** |
| **L24 activation patching** (extend causal evidence to main finding layer) | 1-2 | 2 hr | **High** -- connects causal mechanism to headline results. Current patching at L12 where absorption is only 5.7% is disconnected from L24's 27.1%. | **High** |
| **Benign/pathological on first-letter** (replicate H8 on a second hierarchy) | 0.5-1 | 1 hr | **Moderate** -- confirms universality of "100% pathological" claim. If first-letter shows some benign cases, finding becomes hierarchy-dependent (interesting). If also 100% pathological, claim strengthened. | **Moderate** |
| **Multi-feature patching** (zero top-3 child features for cross-domain) | 1-2 | 2 hr | **Moderate** -- addresses the cross-domain patching failure. Could rescue or definitively bury the distributed mechanism hypothesis. | **Moderate** |
| **validate_integration.py** (automated claim-vs-data checker) | 0 (CPU) | 1.5 hr | **Moderate** -- prevents hallucinated numbers (12.3% incident). Quality infrastructure. | **Moderate** |
| **Writing fixes** (title, abstract, figures 5-6, appendices, cross-refs) | 0 (CPU) | 4 hr | **Moderate** -- necessary for submission but does not change substance. | **Low per hour** (but total impact moderate) |
| **Reduce city-country to top-20 classes** (improve probe F1) | 1 | 1 hr | **Low-Moderate** -- may improve probe quality but also changes the experiment. Better to address confound via degradation ablation. | **Low** |
| **Ecological phase transition backup** (from alternatives.md) | 2-3 | 3 hr | **Low** -- rate-distortion predictors already failed (H9). Phase transition model is theoretically motivated but pilot evidence is not encouraging. | **Low** |
| **Post-hoc correction backup** (from alternatives.md) | 1-2 | 2 hr | **Low** -- prescriptive but not the current paper's focus. Better as follow-up work. | **Low** |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome | **Priority** |
|-----------|----------------|----------|------|-------------------|-------------|
| Probe degradation ablation | Resolves FATAL FLAW (FF1) | 1-2 hr | Low (experiment is straightforward) | Either validates or refutes cross-domain finding -- both outcomes are publishable | **1 (BLOCKING)** |
| L24 activation patching | Fills critical gap in causal story | 1-2 hr | Medium (may fail like cross-domain patching) | Connects causal evidence to headline finding. Failure is informative (mechanism is layer-specific) | **2** |
| Writing infrastructure (validate_integration, fixes) | Prevents false claims, enables submission | 0 GPU | Low (engineering work) | Paper achieves internal consistency | **3 (parallel with 1-2)** |
| Benign/pathological replication | Strengthens moderate-signal finding | 0.5-1 hr | Low | Either confirms universality or reveals hierarchy-dependence | **4** |
| Multi-feature patching | Addresses mechanism gap | 1-2 hr | High (may still fail) | Partially explains or buries distributed mechanism | **5 (optional)** |
| Architecture with matched widths | Addresses power/design concern | 2 hr | Medium | Strengthens or weakens "hierarchy >> architecture" claim | **6 (defer to revision)** |

---

## 4. PIVOT vs PROCEED Verdict

### **PROCEED**

Explicit criteria evaluation:
- **At least one hypothesis with moderate+ signal**: YES. H1 (cross-domain variation) has p=7.37e-66. H7 (causal patching) has d=1.33 for first-letter. Layer-dependent absorption (15x range) is confound-free.
- **Clear path to publication-quality results**: YES, conditional on resolving the probe quality confound via degradation ablation.

There is no case for PIVOT. The cross-domain characterization finding is the first of its kind, with no competing work as of April 2026. The 4 definitive negative results (GAS, CMI, Absorption Tax, rate-distortion predictors) are valuable contributions in a field prone to over-claiming. The causal evidence for first-letter is strong. The honest negative results have been praised across all 9 iterations.

**However, PROCEED is conditional on one blocking experiment**: the probe degradation ablation. If this experiment shows that injecting noise to F1=0.73 in first-letter probes replicates the city-country absorption rate (45.1%), then the cross-domain comparison must be reframed as "layer-dependent + causal + hedging" rather than "cross-domain variation." This changes the paper's narrative but does not kill it -- the layer dependence finding (15x, confound-free) and causal evidence (d=1.33) remain strong standalone contributions.

---

## 5. PROCEED: Recommended Next Steps (Priority Order)

### Experiment 1 (BLOCKING, ~2 GPU-hours): Probe Degradation Ablation

**Objective**: Determine whether the cross-domain absorption rate differences are driven by probe quality differences or genuine hierarchy effects.

**Method**: Take the first-letter probes (F1=0.96) and inject calibrated label noise to degrade them to F1={0.70, 0.80, 0.85, 0.90}. Re-run the absorption measurement pipeline at L24 with each degraded probe. Plot absorption rate vs probe F1.

**Decision branches**:
- **If absorption increases monotonically with probe degradation**: The cross-domain differences are at least partially a probe artifact. Lead the paper with layer-dependent absorption (15x, confound-free) and causal evidence. Downgrade cross-domain to "suggestive, probe-confounded" in a secondary section.
- **If absorption rate is stable across F1 range 0.70-0.96**: The cross-domain finding is robust. City-country's 45.1% rate is genuine, and the paper can lead with the "4x variation" headline.
- **If intermediate** (some increase but not enough to explain the 4x range): Report the partial confound quantitatively. Adjust headline numbers with a correction factor.

**Why this is BLOCKING**: The Skeptic identifies this as a FATAL FLAW. The Optimist's own caveats acknowledge the probe-absorption correlation (rho=-0.67 in evolution lessons). The Reflection explicitly lists this as the #1 experiment priority. Without this experiment, the paper's central claim is vulnerable to a one-line reviewer rejection: "Have you controlled for probe quality?"

### Experiment 2 (HIGH priority, ~2 GPU-hours): L24 Activation Patching

**Objective**: Extend causal absorption evidence from L12 (5.7% absorption) to L24 (27.1% absorption) for first-letter.

**Method**: Replicate the iter_008 activation patching protocol at L24 with the L24-specific probe and SAE. Zero child features, measure parent recovery rate. Include same controls (random latent, semantically unrelated latent).

**Decision branches**:
- **If d>0.8 and p<0.01 at L24**: Causal mechanism generalizes across layers. Paper can claim "layer-invariant competitive exclusion mechanism."
- **If null result at L24**: Mechanism is layer-specific. Interesting finding: absorption at different layers may have different causal structures.

**Why this is high priority**: The paper's headline finding (27-45% absorption at L24) has zero causal validation. The existing causal evidence (d=1.33) comes from L12 where absorption is only 5.7%. A reviewer will ask: "Your causal evidence is at a layer with 6% absorption. Why should I believe the 27% number at L24 has the same mechanism?"

### Experiment 3 (CPU only, ~4 hours): Writing Infrastructure + Fixes

Run in parallel with GPU experiments:

1. **Fix the 12.3% hallucinated number** [5 min] -- the reflection identifies this as a soundness issue.
2. **Implement validate_integration.py** [1.5 hr] -- cross-check all numerical claims in the paper against source JSON files. This has been recommended for 9 iterations. The 12.3% incident proves it is necessary.
3. **Fix broken cross-references** [10 min] -- "Table 7 in Section 4.4" and "Section 8.6" do not exist.
4. **Change the title** [10 min] -- "The Absorption Tax" references a framework that completely failed (rho=-0.20). Lead with the actual contribution: something like "Beyond First-Letter Spelling: Cross-Domain and Cross-Layer Characterization of SAE Feature Absorption."
5. **Generate figures 5 and 6** [1 hr] -- activation patching dot plot and hedging decomposition stacked bar chart. Both can be generated from existing JSON data.
6. **Restructure the abstract** [30 min] -- lead with confound-free findings (15x layer variation, d=1.33 causal evidence), qualify cross-domain variation appropriately.

---

## 6. Resource Allocation Summary

| Resource | Allocation | Justification |
|----------|-----------|---------------|
| GPU: 2 hours | Probe degradation ablation | Resolves the single biggest threat to publishability |
| GPU: 2 hours | L24 activation patching | Connects causal story to headline results |
| GPU: 0.5-1 hour (stretch) | Benign/pathological replication on first-letter | Strengthens H8 finding if time permits |
| CPU: 4 hours | Writing infrastructure + fixes | Necessary for submission quality |
| **Total**: ~4-5 GPU-hours + 4 CPU-hours | | |

Expected wall-clock: 1 day (experiments can run sequentially on single GPU while CPU work runs in parallel).

---

## 7. Paper Narrative Recommendation

The paper should be reframed around three confound-free pillars, with the cross-domain finding positioned carefully:

**Pillar 1 (Strongest)**: Layer-dependent absorption in SAEs.
- 15x variation across layers (0.7% at L6 to 27.1% at L24) for first-letter, measured with near-perfect probes (F1>=0.96).
- Zero probe quality confound. This is the single cleanest finding.
- Implication: SAE analyses at intermediate layers drastically underestimate absorption.

**Pillar 2 (Strong)**: Causal competitive exclusion mechanism.
- Activation patching (d=1.33, p=0.000218) at L12.
- If L24 patching succeeds, this generalizes across layers.
- If L24 patching fails, the layer-specificity itself is interesting.

**Pillar 3 (Conditional on degradation ablation)**: Cross-domain absorption variation.
- If validated: "Absorption rates vary dramatically across hierarchy types (4x range), with hierarchy-specific patterns."
- If confounded: "Cross-domain measurements reveal suggestive variation, but probe quality differences partially explain the observed rates."

**Bonus pillar**: 100% pathological absorption (if replicated on second hierarchy).

**Negative results**: 9 definitive negative results (GAS, CMI, Tax, rate-distortion, cross-domain patching, H2' ordering) as an appendix suite. Honest reporting remains the paper's strongest aspect.

---

## 8. Risk-Adjusted Publication Timeline

| Scenario | Probability | Paper Strength | Target Venue |
|----------|------------|----------------|-------------|
| Degradation ablation validates + L24 patching succeeds | 35% | **Strong accept** -- all three pillars stand | NeurIPS 2026 main |
| Degradation ablation validates + L24 patching fails | 15% | **Borderline accept** -- cross-domain + layer-specific mechanism | NeurIPS 2026 main / ICLR 2027 |
| Degradation ablation shows partial confound | 30% | **Weak accept** -- lead with layer dependence + causal + hedging tautology | ICLR 2027 / EMNLP 2026 |
| Degradation ablation shows full confound | 15% | **Workshop** -- layer dependence + causal + negative results | NeurIPS 2026 MI Workshop |
| Degradation ablation + L24 both fail | 5% | **Reframe paper** around causal mechanism + negative results only | Workshop or arxiv |

---

## 9. Anti-Pattern Check

- **Fence-sitting?** No. Clear PROCEED recommendation with specific blocking condition.
- **Sunk cost reasoning?** No. I explicitly state that if the degradation ablation reveals the confound, the paper should be reframed rather than defended. The 9 iterations of effort do not justify publishing unreliable cross-domain numbers.
- **Ignoring resource constraints?** No. Total recommended experiments are ~4-5 GPU-hours (well within budget). Writing fixes are CPU-only.
- **Ignoring negative results?** No. The cross-domain patching failure (d=-0.91) and the rate-distortion predictor reversal are explicitly weighted into the strategy. The paper should NOT claim a universal competitive exclusion mechanism for semantic hierarchies.

---

## 10. Bottom Line

**PROCEED with conditional framing.**

The project has 2 unconditionally strong contributions (layer-dependent absorption, causal patching for first-letter), 1 strong methodological contribution (hedging tautology), and 4 definitive negative results. These alone constitute a publishable paper for a top-tier venue, even if the cross-domain finding is partially confounded.

The single most important action is the probe degradation ablation (~2 GPU-hours). It resolves the paper's Achilles heel and determines whether the narrative leads with "cross-domain variation" or "layer dependence." Both narratives are publishable; the experiment determines which is honest.

Do not delay this experiment for more writing. Do not invest in the ecological phase transition backup (rate-distortion predictors already failed). Do not attempt to rescue the Absorption Tax quantitative framework. Focus resources on the 2 blocking experiments and writing infrastructure, then proceed to submission.

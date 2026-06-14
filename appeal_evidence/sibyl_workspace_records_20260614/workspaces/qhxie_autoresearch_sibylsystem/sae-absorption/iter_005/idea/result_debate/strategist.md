# Strategist Analysis: Iteration 5 Result Debate

**Agent**: Strategist (sibyl-light)
**Iteration**: 5
**Date**: 2026-04-15

---

## 1. Signal Strength Assessment

| Result | Signal Strength | Metric Delta | Justification |
|--------|----------------|-------------|---------------|
| H1: Sparse probing partial r strengthened to -0.746 after L0 control | **Strong** | Delta = -0.082 (suppression effect) | Suppression effects are rare and informative. The fact that controlling for L0 *increased* the absorption-quality association is a striking finding unlikely to vanish at larger sample. p = 1.16e-09 on 48 SAEs is robust. |
| H1: 3/4 quality metrics retain |partial_r| > 0.2 after L0 control | **Strong** | SCR: -0.570, TPP: -0.331 | Convergent evidence across independent metrics. Only unlearning fails, but unlearning is known to be a noisy metric. |
| H1: Rosenbaum Gamma = 2.65 (Mahalanobis) | **Moderate** | - | Gamma > 2.0 is conventionally "strong" robustness, but this is on cross-width matching. The within-width null (Gamma = 1.0) is a serious caveat that weakens the causal claim. |
| H1: Within-width matching shows no significant quality differences | **Moderate (negative)** | All within-width p > 0.10 | This is an honest negative finding that limits the causal interpretation. The absorption-quality link may operate primarily through cross-width architecture differences, not as a within-width causal mechanism. |
| H1: Mediation (2/4 full Baron & Kenny, 3/4 significant Sobel) | **Moderate** | SCR mediation full, TPP mediation full | Mediation analysis supports the L0 -> Absorption -> Quality pathway for SCR and TPP. But sparse probing shows unstable proportion mediated (4.785, which is uninterpretable), and unlearning is null. |
| H2: Dominance-based absorption 51-85% on knowledge hierarchies | **Noise** | Shuffled control = 100% | The metric does not discriminate real from shuffled hierarchies. This is not evidence of absorption -- it is evidence that the dominance-based metric captures background feature concentration. Signal = 0. |
| H2: Cosine-calibrated absorption = 0% | **Moderate (negative)** | 0% across all thresholds | Clean null. No probe-direction absorption is detectable in GPT-2 Small knowledge hierarchies with this SAE. But: GPT-2 Small has 98% dead features and limited factual knowledge, so this is a floor, not a ceiling. |
| H3: GAM interaction p = 3.11e-15 on 420 SAEs | **Strong** | R^2 improvement: 0.488 -> 0.620 -> 0.693 | Massive sample (420 SAEs), highly significant interaction, monotonic R^2 improvement through additive to interaction model. This result will hold at any scale. |
| H3: Phase boundary at log2(L0) range [2.7, 3.8] | **Strong** | Max gradient magnitude = 0.987 | A detectable ridge in the absorption surface with steep gradient. This is the most actionable finding: L0 below ~6-14 is the "danger zone" for absorption. |
| H5: Taxonomy correction 92.3% -> 92.3% | **Moderate** | Delta = 0.0% | The original rate is validated, not corrected. This is a useful null: the measurement artifact concern from iter_4 was not substantiated. But it limits the "taxonomy correction" as a standalone contribution. |

## 2. Opportunity Cost Analysis

| Direction | GPU Cost | Time | Expected Information Gain per GPU-Hour |
|-----------|----------|------|---------------------------------------|
| **Proceed to writing with H1 + H3 as dual pillars** | 0 GPU-hours | 4-6h writing | **Very high**. Two publication-ready contributions require zero additional experiments. The paper can ship now. |
| **Fix Phase 2 with Gemma 2B model** | 4-6 GPU-hours | 6-10h total (gating + probes + measurement) | **Medium**. Requires Gemma 2B HF token. Uncertain whether probe quality will be sufficient. Even with perfect execution, the cross-domain contribution adds breadth but the current two pillars are already strong. |
| **Fix Phase 2 with Chanin IG-based method on GPT-2** | 2-3 GPU-hours | 3-5h | **Medium-High**. The IG method is the gold standard for absorption detection. If it shows non-zero absorption on knowledge hierarchies in GPT-2, the Phase 2 contribution is rescued. If not, the GPT-2 null becomes a cleaner negative result. Either outcome is publishable. |
| **Add GPT-2 SAEs to Phase 3 for cross-model validation** | 0.5 GPU-hours | 1-2h | **Low-Medium**. Phase 3 is already strongly supported on 420 Gemma SAEs. Cross-model validation is nice-to-have but not essential for publication. |
| **Implement hurdle model for PMI analysis** | 0 GPU-hours | 1h | **Low**. The PMI finding is interesting (significant in beta/hurdle but not OLS) but peripheral to the main contributions. Can be mentioned in discussion without being a headliner. |
| **Run immunodominance R* analysis** | 2-3 GPU-hours | 3-4h | **Low**. Does not address any blocking issue. The immunodominance framework is intellectually interesting but irrelevant to the current paper's three-pillar structure. Defer to next iteration or supplementary material. |

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|-----------|----------------|----------|------|-----------------|
| **Write paper with H1 + H3 (2-pillar)** | Strong + Strong | 0h | LOW | Two-contribution paper: confound resolution + scaling surface. Missing the cross-domain pillar but both pillars are publication-ready. Risk: reviewers ask "why no cross-domain?" |
| **Write paper with H1 + H3 + negative H2 (3-pillar)** | Strong + Strong + Moderate (neg) | 0h | LOW-MEDIUM | Three-contribution paper where Contribution 2 is a methodological finding (dominance metric failure + GPT-2 Small negative result). Risk: "negative result as contribution" may be undersold. |
| **Fix H2 with IG method then write (3-pillar)** | Strong + Strong + TBD | 2-3h | MEDIUM | If IG method works: strongest possible paper with positive cross-domain finding. If not: still three pillars, but IG-validated negative result is much stronger than current dominance-metric null. |
| **Fix H2 on Gemma 2B then write** | Strong + Strong + TBD | 4-6h | HIGH | Requires model access (unknown), probe quality may fail, most expensive option. Payoff is high only if probes reach >85% accuracy AND absorption is detected. |
| **Pivot to cand_phase_diagram as primary** | Strong (H3) | 0h | MEDIUM | Phase diagram is strong but thin as sole contribution. Would need theoretical scaffolding (compressed sensing analogy) to fill a paper. Worse than current plan. |

## 4. PIVOT vs PROCEED Verdict

### **PROCEED**

Rationale:
- **H1** (confound resolution) has strong signal across multiple independent analyses. The suppression effect finding (sparse probing partial r strengthened after L0 control) is a genuine surprise that adds credibility. Mediation, Rosenbaum sensitivity, and Bradford Hill criteria all converge. The within-width null is an important caveat but does not invalidate the finding -- it constrains the interpretation.
- **H3** (scaling surface) is the single strongest result: p = 3.11e-15 on 420 SAEs with clear monotonic R^2 improvement. This is actionable, novel, and unambiguous.
- **H2** (cross-domain) is the weakest pillar but reveals a publishable methodological insight: the standard dominance-based absorption metric breaks on knowledge hierarchies because it measures feature concentration, not probe-direction absorption.
- Two of three contributions are publication-ready NOW. The critical path is writing, not more experiments.

Why not PIVOT:
- No backup candidate (cand_phase_diagram, cand_immunodominance, cand_honest_audit) is stronger than the current combination. The phase diagram is already subsumed as H3. The honest audit is not supported (H1 survived). Immunodominance is orthogonal.
- Sunk cost is NOT the reason to proceed -- the evidence IS strong. The partial r = -0.746 and GAM p = 3.11e-15 are not borderline results.

## 5. Recommended Next Steps (Priority Order)

### Priority 1: Write paper immediately with current results (0 GPU-hours, 4-6h)

Structure the paper as follows:
- **Contribution 1**: Confound resolution via epidemiological methods (Phase 1 results)
- **Contribution 2**: Absorption scaling surface with nonlinear interaction (Phase 3 results)
- **Contribution 3**: Cross-domain metric analysis + GPT-2 Small negative result (Phase 2 results, framed as a methodological finding and model-scaling hypothesis)

The Phase 2 finding should be framed NOT as "absorption does not generalize" but as "the standard absorption metric does not generalize to knowledge hierarchies, revealing a measurement validity gap. On GPT-2 Small with 98% dead features, no probe-direction absorption is detected, suggesting absorption may require sufficient model capacity and SAE dictionary coverage."

The taxonomy correction (H5) becomes a paragraph in the discussion, not a standalone contribution: "We attempted to correct the 92.3% combined rate reported in [Chanin et al.] but found the correction was minimal (delta = 0.0%). The high rate reflects genuine feature selectivity rather than measurement artifact."

### Priority 2: Before or during writing, attempt IG-based absorption on GPT-2 knowledge features (2-3 GPU-hours, 3-5h)

This is the single highest-ROI additional experiment because:
1. It uses a validated gold-standard method (Chanin's integrated gradients)
2. It runs on an already-available model (GPT-2 Small)
3. Either outcome strengthens the paper:
   - IG detects absorption: cross-domain contribution is rescued as a positive finding
   - IG detects no absorption: validates the cosine-calibrated null as genuine (not metric failure)
4. It directly addresses the dominance-vs-cosine discrepancy that is the current weakness

However, this should NOT gate writing. Start writing Contributions 1 and 3 in parallel with this experiment.

### Priority 3: Do NOT pursue Gemma 2B cross-domain (deferred)

Rationale: Gemma 2B access is uncertain (gating), probe quality is unknown, and the GPU cost is 4-6h. The expected information gain does not justify the delay. If this paper is accepted, the Gemma 2B cross-domain study is a natural follow-up paper or an extended version.

---

## Resource Allocation Summary

| Resource | Allocation | Justification |
|----------|-----------|---------------|
| **GPU budget** | 2-3h max (IG experiment, optional) | Paper can ship with 0 additional GPU hours. IG experiment is insurance. |
| **Writing time** | 4-6h (critical path) | Two contributions are publication-ready. Writing is the bottleneck, not experiments. |
| **Iteration time** | 6-10h total to complete iteration | Write in parallel with optional IG experiment. |

## Risk Assessment for PROCEED

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Reviewers demand within-width causal evidence for H1 | Medium | Report the within-width null transparently. Frame as "absorption-quality association is robust to confound control but cannot be localized to within-width variation, suggesting it operates through cross-width architecture choices." |
| Reviewers view Phase 2 negative result as insufficient | Low | Frame as methodological contribution (metric validity). The GPT-2 Small + 98% dead features context provides a clear explanation for the null. |
| H3 scaling surface is "just a regression" | Low | The GAM framework with interaction testing is methodologically rigorous. The phase boundary detection adds interpretive value beyond a raw regression. |
| Field velocity: concurrent work covers same ground | Medium | The novelty report scores 8/10 with zero prior art on mediation/Rosenbaum for SAE evaluation. Speed matters: ship within this iteration. |

## Bottom Line

This iteration produced two strong, publication-ready contributions (confound resolution and scaling surface) and one methodological finding (cross-domain metric limitation). The dominant strategy is: **write the paper now, optionally run IG-based absorption in parallel, do not pursue Gemma 2B**. The expected outcome is a paper with contributions that are individually novel and collectively form a coherent narrative about moving absorption from anecdotal evidence to rigorous empirical characterization.

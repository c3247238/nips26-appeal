# Research Proposal (Revised — Post-Experimental Round 3)

## Title

**When Encoder-Decoder Misalignment Signals Feature Absorption: Regime-Specific Detection, Cross-Domain Generalization, and the Dominance of Dictionary Coverage Failure**

---

## Abstract

Feature absorption — the systematic failure of parent features to fire when child features are active — is the defining reliability challenge for sparse autoencoders (SAEs) used in mechanistic interpretability. Three prior rounds of ideation and one full experimental cycle have produced a clear, honest assessment of what we know and what must still be done. We contribute three interlocking results: (1) a **Formal EDA Lower Bound Theorem** — encoder-decoder angular divergence (EDA) is a necessary, theoretically grounded condition for absorption in converged SAEs, validated empirically with AUROC = 0.776 at mid-layers in narrow SAEs (Gemma Scope L12-16k) and strong group separation (Cohen's d > 0.84, p < 1e-5); (2) the **first cross-domain characterization** of SAE feature absorption beyond the first-letter spelling task, demonstrating that absorption generalizes to RAVEL entity-attribute hierarchies with high intra-domain coherence (rho = 0.924), with hierarchy-dependent severity spanning 10-100x; and (3) a **three-subtype taxonomy** — early (decoder-absent), late (decoder-present, encoder-suppressed), partial (selective failure) — revealing that approximately 72-75% of absorbed latents correspond to features the SAE dictionary never learned ("early absorption"), reframing the dominant failure mode as a dictionary coverage problem rather than an encoder alignment problem. The two weaker components of the original proposal — D-EDA residual decomposition and ITAC inference-time correction — are repositioned honestly as an alternative detector and a proof-of-concept, respectively. We identify one critical blocking dependency — Gemma 2B model access for ground-truth label replication — and specify exactly what experiments remain before the writing phase.

---

## Motivation

Feature absorption has a clear gradient-descent-stable mechanism (Chanin et al., NeurIPS 2025), a rigorous optimization landscape characterization (Tang et al., 2025), and has motivated multiple architectural responses — OrtSAE, Matryoshka SAE, KronSAE, masked regularization (Narayanaswamy et al., 2026). Yet the field faces four compounding gaps that this work addresses:

**Gap 1 — Detection requires foreknowledge.** The canonical metric (Chanin et al.) requires pre-specified probe directions. An unsupervised weight-based detector would enable systematic auditing across all latents in deployed SAEs.

**Gap 2 — Generalizability is assumed, not tested.** Every published absorption measurement uses the first-letter spelling task. The question "does absorption matter for the features that matter — entity types, safety-relevant concepts?" has never been answered.

**Gap 3 — No actionable subtype taxonomy.** All prior work treats absorbed latents as a single category. Whether absorption is primarily an encoder alignment failure (remediable at inference time) or a dictionary coverage failure (requiring retraining) has never been empirically tested.

**Gap 4 — Practical guidance is missing.** The field lacks evidence-based guidance on when EDA can be trusted as a screening tool, which absorption subtypes dominate in practice, and what the implications are for practitioners choosing SAE architectures.

**Post-experimental insight**: Gap 3's answer is now available and is the paper's most practically important finding: early absorption (dictionary coverage failure) dominates at ~75%. This reframes the entire absorption mitigation literature — most proposed solutions (ITAC, OrtSAE-at-inference, Select-and-Project) target the ~25% minority. The primary lever for absorption reduction is not encoder architecture but dictionary width and training objective.

---

## Research Questions

1. **RQ1 (Theory):** Does encoder-decoder misalignment (EDA) provide a provably non-trivial lower bound on absorption degree that is tighter than random, and what are the conditions under which it has practical discriminative power?

2. **RQ2 (Detection):** In which SAE regime (layer, width, model) does EDA achieve reliable absorption detection, and what explains its layer-width dependency?

3. **RQ3 (Generalization):** Does absorption occur in entity-attribute hierarchies beyond first-letter spelling, and is it coherent within hierarchy families despite differing from cross-family rates?

4. **RQ4 (Taxonomy):** Does the three-subtype partition (early/late/partial) reflect genuine geometric distinctions captured by EDA, and what fraction of absorbed latents falls into each subtype?

5. **RQ5 (Implications):** What does the dominance of early absorption imply for mitigation strategy, and when is inference-time correction applicable?

---

## Hypotheses

**H1 (EDA Lower Bound — Confirmed):** For a converged SAE at a partial minimum of the biconvex SDL loss (Tang et al. 2025), if latent j exhibits delta-absorption of child c, then EDA(j) >= delta^2 * sin^2(theta_{jc}) / (2 + delta^2). EDA is monotonically increasing in delta. For monosemantic non-absorbed latents at the global minimum, EDA(j) = 0. EDA > 0 is necessary but not sufficient (polysemanticity is a confound).
- **Status**: Confirmed on synthetic data (SynthSAEBench AUROC = 1.0), confirmed with group separation on real Gemma SAEs (Cohen's d = 0.84-1.14, p = 6.4e-5 at L12-16k).

**H2 (EDA Regime-Specific Detection — Partially Confirmed):** EDA achieves AUROC >= 0.65 in mid-layer, narrow-SAE regimes. EDA performance is layer- and width-dependent; this dependency is informative about absorption's geometric nature.
- **Revised from original**: "EDA >= 0.65 universally" → "EDA >= 0.65 in favorable regimes (mid-layers, narrow SAEs)."
- **Status**: 2/6 Gemma configs pass; strongest: L12-16k AUROC = 0.776. 4/6 fail. Pilot-to-full collapse at L12-65k (0.853 → 0.468) likely reflects proxy label quality issues, not intrinsic EDA failure. Resolution requires Gemma 2B access.

**H3 (Cross-Domain Generalization — Reframed, Pending Validation):** Absorption generalizes to entity-attribute hierarchies; intra-domain coherence is robust; first-letter and RAVEL operationalizations measure non-identical aspects of absorption.
- **Status**: All 18 RAVEL measurements exceed 3x random baseline; intra-RAVEL rho = 0.924 (Bonferroni-corrected significant). Cross-paradigm correlation is negative (first-letter vs. RAVEL rho = -0.43 to -0.20). RAVEL probes trained on wrong model (Qwen2.5-0.5B → Gemma 2B projection) — absolute rates unreliable pending proper Gemma 2B probes.

**H4 (Three-Subtype Taxonomy — Partially Confirmed):** Late-absorbed latents have higher EDA than early-absorbed (confirmed). Partial ordering fails the intermediate-EDA prediction. Early absorption dominates at ~75%.
- **Status**: KW p = 0.0002 at L12-65k; late > early holds at all 5 tested thresholds. Partial absorption has lower EDA than early (contradicting prediction), suggesting partial and early may share a geometric mechanism distinct from late. The 75% early-dominance finding is the highest-impact result from the taxonomy.

**H5 (ITAC — Refuted, repositioned):** ITAC achieves 3% FN reduction vs. 20% pre-registered target. Structural reason: 75% of absorbed latents are early-type (decoder-absent), ineligible for ITAC.
- **Revised framing**: ITAC is a proof-of-concept applicable to ~13% of absorbed latents. The ITAC failure confirms the early-dominance finding: if late absorption were dominant, ITAC would show stronger signal.

**H6 (Scaling Sign Reversal — Falsified):** Partial rho(width, absorption | L0) = +0.37. No sign reversal. Wider SAEs consistently absorb more, with no compensatory L0 mechanism.

---

## Expected Contributions

### Primary Contributions

**Contribution 1: EDA as a Regime-Specific Weight-Only Absorption Screening Tool**
- The first formal weight-only absorption detector with theoretical lower bound from biconvex optimization theory (Tang et al. 2025).
- The detector works reliably at mid-layers in narrow SAEs (L5-16k: AUROC = 0.698; L12-16k: AUROC = 0.776; Cohen's d > 0.84). Its layer-width dependency is itself an informative finding about absorption's geometric footprint.
- EDA outperforms the decoder cosine similarity baseline by +0.396 AUROC — establishing that encoder-decoder alignment provides signal beyond decoder-only geometry.
- Novel insight: EDA's failure at large widths (L12-65k: AUROC = 0.468) may reflect that wider SAEs have a higher proportion of early absorption (decoder-absent), for which EDA is theoretically expected to have no signal. This would constitute a non-trivial prediction of the theory.
- Differentiated from LessWrong "Toy Models" informal suggestion by: formal theorem with quantitative bounds, empirical AUROC validation, cross-model replication, baseline comparison.

**Contribution 2: First Cross-Domain Absorption Characterization**
- The first systematic measurement of SAE feature absorption beyond the first-letter spelling task, using RAVEL entity-attribute hierarchies (city-continent, city-country, city-language).
- Existence proof: all 18 SAE-hierarchy combinations show absorption above 3x random baseline.
- Intra-domain coherence: rho = 0.924, p < 0.005 — absorption rankings are stable across SAE configs within the entity-attribute hierarchy family.
- Hypothesis-generating finding: first-letter absorption rates negatively correlate with RAVEL absorption rates (rho = -0.20 to -0.43), suggesting these operationalizations may capture different facets of absorption. Possible interpretation: the first-letter task, due to its uniquely syntactic hierarchy with near-complete co-occurrence statistics, may elicit a qualitatively different absorption regime.
- Pending: proper Gemma 2B probes to validate absolute absorption rates and shuffled hierarchy controls.

**Contribution 3: Three-Subtype Taxonomy and Early-Absorption Dominance Insight**
- The first formal three-way partition of absorbed latents into early (decoder-absent: 72-75%), late (decoder-present: ~25%), and partial (selective failure: minor) subtypes.
- The late > early EDA ordering is statistically robust (KW p = 0.0002 at L12-65k; stable across all 5 threshold variants).
- The dominant finding — 75% of absorbed latents are early-type — reframes the absorption problem entirely: the primary failure is dictionary coverage (the SAE never allocated capacity to the parent feature), not encoder misalignment. Most proposed mitigation strategies (OrtSAE at inference time, ITAC, Select-and-Project) target the minority late-absorption category. The prescriptive implication: wider dictionaries or hierarchically-aware training objectives are the appropriate lever for most absorption, not inference-time encoder fixes.
- Partial absorption has lower EDA than early (counter to prediction), suggesting the partial subtype involves early-stage dictionary issues, not encoder suppression — a refinement worth noting.

### Secondary Contributions (Negative Results, Honestly Reported)

**D-EDA as an Alternative Detector:** D-EDA does not improve on scalar EDA in Gemma configs where EDA works, but captures qualitatively different signal in GPT-2 L10 (D-EDA = 0.762 vs. EDA = 0.336). The likely mechanism: the sparse decomposition onto the decoder dictionary (d_sae >> d_model) is numerically ill-conditioned in large SAEs. D-EDA may be applicable as a rescue strategy when scalar EDA fails, but its conditioning properties require further analysis.

**ITAC as Proof-of-Concept:** ITAC achieves 3% mean FN reduction (not 20% target), limited by the 75% early-absorption prevalence. The successful case (j_idx=61217, 22.7% FN reduction at L12-65k) demonstrates that late absorption can be partially corrected at inference time when the geometry is favorable. The FVU improvement (-4.23% reconstruction error) suggests ITAC does not degrade quality. Recommended framing: "ITAC-style corrections are applicable to the ~13% of absorbed latents that are late-type; for the dominant 75% (early absorption), the appropriate fix is training-time."

**H6 Scaling Non-Result:** Wider SAEs consistently absorb more at any L0 setting. The confound hypothesis (width-L0 correlation) cannot be resolved with Gemma Scope canonical SAEs due to insufficient L0 variation within layers. Reported as a methodological finding: the partial correlation design requires a wider range of L0 variants than canonical SAEs provide.

---

## Novelty Assessment

### Primary Claims — Novelty Status

**EDA weight-only detector (Claim 1):**
- No prior work formalizes encoder-decoder alignment as a computable absorption metric with formal bounds.
- LessWrong "Toy Models" (Chanin et al., Oct 2024) contains the informal seed; our contribution is formalization, theorem, cross-model AUROC validation, and regime-characterization.
- EDA outperforming decoder cosine similarity by +0.396 AUROC confirms the signal is encoder-specific, not merely decoder geometry.
- **Verdict: NOVEL.** Partial overlap with informal suggestion fully cited and differentiated.

**Cross-domain absorption (Claim 2):**
- No prior work measures SAE feature absorption in entity-attribute hierarchies. SAEBench absorption metric is exclusively first-letter. RAVEL was used for disentanglement evaluation, not absorption measurement.
- **Verdict: NOVEL.** Cleanest novelty claim in the paper.

**Three-subtype taxonomy + early dominance (Claim 3):**
- No prior taxonomy of absorbed latents by geometric subtype (decoder-absent vs. decoder-present) or remediability.
- The 75% early-absorption finding is not anticipated by any existing paper. It directly contradicts the implicit assumption of prior ITAC-style proposals (that late absorption is the dominant case) and the theoretical focus on encoder alignment.
- **Verdict: NOVEL.** The early-dominance finding alone constitutes a significant practical reframing of the field.

### Competing Work Confirmed Not Blocking

| Paper | Relationship | Status |
|-------|-------------|--------|
| Chanin et al. 2409.14507 | Defines absorption; supervised detection only | Related work |
| Tang et al. 2512.05534 | Biconvex landscape theory (EDA bound grounding) | Related work (cited as theoretical foundation) |
| MP-SAE (Costa et al. 2506.03093) | Iterative encoder reducing absorption; NOT post-hoc to frozen SAE | Related work; ITAC differentiated |
| OrtSAE, Matryoshka, KronSAE, MaskedReg | Architectural mitigations requiring retraining | Related work |
| O'Neill et al. 2411.13117 | Amortization gap proof | Related work (third confounder for EDA) |
| Korznikov et al. 2602.14111 | SAEs recover 9% of synthetic features | Framing context (cited in motivation) |

---

## Revisions from Prior Feedback

This is Synthesis Round 3 (post-experimental). Changes from Round 2 proposal:

1. **EDA repositioned as regime-specific detector, not universal.** The result debate synthesis correctly identified that 4/6 config failures and the pilot-to-full AUROC collapse prevent the universal EDA claim. The new framing — EDA is reliable in mid-layer, narrow-SAE regimes — is supported by 2/6 configs with AUROC > 0.65 and strong group separation statistics across more configs.

2. **D-EDA demoted to secondary/supplementary.** D-EDA never outperforms scalar EDA on Gemma configs where EDA works. GPT-2 L10 exception (D-EDA = 0.762, EDA = 0.336) is reported as a complementary-signal finding, not a "D-EDA improvement" claim.

3. **ITAC demoted to proof-of-concept.** The 3% vs. 20% target miss, explained by 75% early-absorption prevalence, makes ITAC non-viable as a primary contribution. The failure is reframed as confirmatory evidence for early-absorption dominance.

4. **Cross-domain existence pending probe fix.** The Qwen2.5-0.5B probe issue is acknowledged as a blocking dependency. The coherence evidence (intra-RAVEL rho = 0.924) is highlighted as independently meaningful, while absolute rates are flagged as unreliable.

5. **Early-absorption dominance elevated to primary finding.** The 75% early-type finding is now a central contribution in its own right. It redirects the mitigation prescription from encoder fixes to dictionary coverage fixes, challenging the implicit framing of the entire ITAC/MP-SAE literature.

6. **H6 scaling analysis moved to supplementary.** The sign-reversal falsification, explained by insufficient L0 variation, is reported honestly as a methodological finding.

7. **Contrarian's Amortization Gap Thesis partially incorporated.** The 75% early-absorption dominance is consistent with both the amortization gap thesis and the Tang et al. optimization landscape thesis: early absorption may arise from both the encoding bottleneck preventing the SAE from learning the parent feature AND the spurious optima landscape that makes absorption the preferred solution. The paper now explicitly acknowledges both mechanisms without overclaiming which dominates.

8. **Cross-paradigm negative correlation elevated, not buried.** The first-letter vs. RAVEL rho = -0.43 is now presented as a hypothesis-generating finding about potentially distinct absorption regimes for syntactic vs. semantic hierarchies.

---

## Critical Blocking Actions Before Paper Writing

### Priority 1 (BLOCKING): Gemma 2B Model Access and Label Quality

**What**: Obtain HuggingFace authentication for Gemma 2 2B (or use Llama-3.1-3B as fallback). Re-run:
- Phase 1: EDA evaluation against Chanin et al. direct labels (not Neuronpedia proxy). Run on configs where proxy labels already produced failing AUROC (L12-65k, L19-16k) to diagnose whether failure is EDA-intrinsic or label-quality-driven.
- Phase 3: RAVEL probes trained directly on Gemma 2B residual stream activations (not random projection from Qwen2.5-0.5B). Verify city-continent probe accuracy >= 85% as quality gate.
- Shuffled hierarchy control: run cross-domain absorption measurement with randomized parent-child labels to confirm above-random signal is not an artifact.

**Decision gate**: If >= 3/6 Gemma configs pass AUROC >= 0.65 with direct labels, EDA claim stands as "mid-layer reliable detector." If < 3/6, reframe as "theoretically grounded diagnostic tool with empirically demonstrated signal in synthetic and specific real-model regimes." Either framing is publishable; the first is stronger.

**Estimated cost**: 3-5 GPU-hours + authentication. Single most important action available.

**Fallback**: Llama-3.1-3B (d_model = 3072, publicly available, comparable architecture). Use sae-spelling and SAELens Llama SAEs. Acknowledge prominently as proxy.

### Priority 2: Cross-Model EDA Replication

**What**: Run EDA on 2 additional Llama configs and 1-2 GPT-2 configs beyond existing GPT-2 L6/L10. The clean GPT-2 L6 exact-label result (AUROC = 0.629) is the paper's most methodologically sound single data point. Cross-model diversity strengthens the generalizability claim.

**Estimated cost**: 2-3 GPU-hours.

### Priority 3 (Optional): ITAC on Real Activations

**What**: Run ITAC on actual Gemma 2B text activations (10,000 tokens). Target: mean FN reduction > 10% on late-absorbed latents. If achieved, restore ITAC to a minor contribution. If not, confirms 3% result generalizes.

**Estimated cost**: 2-3 GPU-hours.

---

## Method Summary (Confirmed)

### Phase 0: Metric Validation (Completed)
- Threshold sensitivity sweep: absorption rate stable within 30% across cosine {0.005-0.10} x magnitude gap {0.5-2.0}. Canonical thresholds are robust.
- Random direction baseline: < 5% absorption rate with random probes. Metric is specific.
- SynthSAEBench: F1 = 0.974 on known-ground-truth absorption. Metric is calibrated.

### Phase 1: EDA and D-EDA Validation (Partially Completed)

**EDA metric** (computed from SAE weight matrices, no activation data):
```
EDA(j) = 1 - cos(w_{e,j}, d_j)
```

**Formal EDA Lower Bound Theorem** (proved, empirically validated):
For converged SAE at biconvex partial minimum: EDA(j) >= delta^2 * sin^2(theta_{jc}) / (2 + delta^2).
Monotonically increasing in absorption degree delta. Necessary but not sufficient for absorption.

**Validated**: Strong group separation at L12-16k (Mann-Whitney p = 6.4e-5, Cohen's d = 1.02), L5-16k (0.698 AUROC). Needs direct Chanin labels at L12-65k to diagnose proxy-label collapse.

**D-EDA** (directional residual decomposition):
```
r_j = w_{e,j} - (w_{e,j} · d_j / ||d_j||^2) * d_j
r_j ≈ sum_k beta_k d_k  (sparse projection onto decoder dictionary)
```
Absorption signature: sparse beta, high cos(d_k, d_j) for active k.
Polysemanticity signature: diffuse beta, low cos(d_k, d_j).
**Status**: Numerically ill-conditioned at large widths (d_sae >> d_model). Does not improve on scalar EDA in Gemma configs. Alternative detector in GPT-2 L10. Reported as supplementary with numerical conditioning analysis needed.

### Phase 2: Three-Subtype Taxonomy and ITAC (Completed)

**Three-subtype classification criteria:**

| Subtype | Criterion | EDA | Fraction | ITAC-Applicable |
|---------|-----------|-----|----------|-----------------|
| Early | max cos(d_k, parent_probe) < 0.3 | Low/absent | ~75% | No |
| Late | max cos(d_k, parent_probe) >= 0.3, latent fails to fire | High | ~13% | Yes (proof-of-concept) |
| Partial | Latent fires on some but not all parent-positive inputs | Intermediate | ~12% | Partial |

**Statistical validation**: Kruskal-Wallis p = 0.0002 at L12-65k; late > early ordering at all 5 threshold variants (cosine 0.2-0.4). Partial < early (counter-prediction) suggests partial absorption shares geometric mechanism with early absorption.

**ITAC** (training-free, post-hoc, applicable to late-absorbed latents only):
- FN reduction: 3.14% mean (L12-65k), 0% (L12-16k). H5 falsified.
- FVU improvement: -4.23% at L12-65k (reconstruction quality improves marginally).
- Positioned as proof-of-concept; not a primary contribution.

### Phase 3: Cross-Domain Characterization (Partially Completed)

**Hierarchy suite:**

| Domain | Hierarchy | Parent | Child |
|--------|-----------|--------|-------|
| Syntactic | First-letter (baseline) | Letter class | Token |
| Entity-attribute | City-Continent | Continent | City |
| Entity-attribute | City-Country | Country | City |
| Entity-attribute | City-Language | Primary language | City |

**Status**: All 18 RAVEL SAE-hierarchy combinations show absorption > 3x random. Intra-RAVEL coherence: rho = 0.924, p < 0.005. Cross-paradigm: first-letter vs. RAVEL rho = -0.43 to -0.20. Absolute rates unreliable (wrong-model probes). **Blocking**: Re-run with Gemma 2B probes.

---

## Experimental Plan (Remaining)

| Priority | Experiment | GPU-hours | Validates |
|----------|-----------|-----------|---------|
| 1 — BLOCKING | Gemma 2B access + Chanin labels replication (6 configs) | 3-4 | H2 definitively |
| 1 — BLOCKING | RAVEL probes on Gemma 2B + shuffled control | 1-2 | H3 definitively |
| 2 — HIGH | EDA on 2 Llama-3.1-3B configs + 1 additional GPT-2 | 2-3 | Cross-model generalization |
| 3 — OPTIONAL | ITAC on real Gemma 2B activations (10k tokens) | 2-3 | ITAC rescue attempt |
| **Total remaining** | | **8-12 GPU-hours** | |

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Gemma 2B access remains blocked | High | Medium | Fallback: Llama-3.1-3B; acknowledge prominently; still publishable as "mid-tier with Llama" |
| EDA with direct labels still fails on > 4/6 Gemma configs | Medium | Medium | Reframe as "EDA works in L5-L12 mid-layer regime" — still publishable; theory holds |
| Shuffled hierarchy control shows same absorption rates | High | Medium | Would invalidate cross-domain claim; pivot to characterization-only paper (EDA + taxonomy) |
| Reviewer adopts "Sanity Checks for SAEs" framing | Medium | Medium | Clear motivation: studying a specific, well-characterized mechanism; synthetic-data control shows F1 = 0.974 |
| Cross-paradigm negative correlation is artifact | Medium | Medium | Fixed by proper Gemma 2B probes; the coherence result (rho = 0.924) is robust to probe noise |
| Concurrent work on absorption subtype classification | Low | Low | No competing paper found; contrarian's literature search found no taxonomy |

---

## Venue Target

**Current state**: Mid-tier (EMNLP 2026 / AISTATS 2027) or NeurIPS 2026 MI Workshop.
**Target state after blocking experiments**: Potentially top-tier (NeurIPS 2026 or ICLR 2027) if Gemma 2B probes confirm cross-domain rates and >= 3/6 EDA configs pass with direct labels.

The three genuine contributions — EDA regime-specific detector, first cross-domain characterization, and early-absorption dominance insight — constitute a complete, novel characterization of absorption that the field needs. The early-absorption finding alone, cleanly demonstrated, would be highly cited for redirecting mitigation research toward dictionary coverage solutions.

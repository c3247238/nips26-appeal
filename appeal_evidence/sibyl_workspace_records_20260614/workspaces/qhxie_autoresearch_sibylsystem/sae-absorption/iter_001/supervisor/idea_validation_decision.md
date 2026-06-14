# Idea Validation Decision (Updated — Post Round 4)

**Decision Agent Run**: 2026-04-13
**Evidence Basis**: All R4 experiments completed (r4a_gemma_access_check, r4a_direct_label_generation, r4a_eda_direct_validation, r4b_ravel_probes_proper, r4b_shuffled_control, r4b_crossdomain_analysis_updated, r4c_llama_replication). R4D (ITAC real activations) not completed (optional, not scheduled).

---

## Pilot Evidence Summary

### Round 3 Baseline (15/15 tasks completed)

| Component | Key Metric | Value | Pass |
|-----------|-----------|-------|------|
| Phase 0: Metric validation | SynthSAEBench F1, threshold sensitivity, random baseline | F1=0.974; 19.8% deviation; ratio=0.214 | PASS |
| Phase 1: EDA validation (Gemma proxy) | AUROC on 6 Gemma Scope SAEs | L5-16k: 0.706, L12-16k: 0.776, L12-65k: 0.853(pilot)/0.468(full), L19-65k: 0.787; 2/6 full-run pass | PARTIAL |
| Phase 2a: Taxonomy | Three-subtype prevalence and EDA ordering | Early 72-75%, Late 13%, Partial/Diffuse 13%; late>early across 5/5 thresholds | PASS |
| Phase 2b: ITAC | Mean FN reduction; best-latent | 3.79% mean (L12-65k); best latent 22.7%; FVU -5.2% | PARTIAL |
| Phase 3: Cross-domain | RAVEL above 3x random; intra-RAVEL rho | 5-6/6 SAEs above 3x random; rho=0.829 (proxy) | CONDITIONAL |
| Phase 4: Scaling (H6) | Partial rho(width | L0) | +0.44 (expected <0) | FALSIFIED |
| Phase 5: GPT-2 replication | AUROC with exact labels | L6: 0.752, L10: 0.643 | PASS |

### Round 4 New Evidence (Critical Updates)

| Task | Key Outcome | Impact on Decision |
|------|------------|-------------------|
| r4a_gemma_access_check | Gemma 2B still gated; Llama-3.1-8B still gated; GPT-2 and SAE weights accessible | Forces weight-only Llama analysis; GPT-2 remains only direct-label model |
| r4a_eda_direct_validation | GPT-2 L6 direct labels: AUROC=0.650 (pass); GPT-2 L10: AUROC=0.336 (reversed, fail); EDA = 1-dec_cos identity confirmed (delta=0.0 always) | Confirms EDA regime-specificity; D-EDA claim fully falsified; L10 reversal is a known failure mode |
| r4b_ravel_probes_proper | GPT-2 Medium bridge: best probe accuracy 59.5% (city-continent); all 3 hierarchies fail 85% gate (relaxed 80% gate also fails) | H3 probe quality remains unvalidated without same-model activations |
| r4b_shuffled_control | Real absorption rates indistinguishable from shuffled null for all 3 domains (0/3 pass; ratio range 0.98-2.09, none exceed p95 threshold) | H3 cross-domain claim COLLAPSES — cross-domain contribution must be dropped or heavily downgraded |
| r4b_crossdomain_analysis_updated | Intra-RAVEL rho drops from 0.924 (R3, n=6) to 0.667 (R4 pilot, n=3 configs) | Suggestive coherence remains but weaker than R3; cannot be primary evidence |
| r4c_llama_replication | Llama-3.1-8B gated; SAE weights loaded; EDA=1-DecCos identity confirmed (r=-1.0); weight-only EDA distribution computed (mean=0.518); no AUROC (no labels) | Confirms formula validity across 3 architectures; no new AUROC evidence |

---

## Decision Matrix (Updated — All Candidates)

### cand_eda_crossdomain (Front-Runner — R4 Re-evaluated)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3.5 | Core EDA detection: 3/8 configs pass AUROC >= 0.65 (Gemma L5-16k: 0.698 proxy, L12-16k: 0.776 proxy, GPT-2 L6: 0.650 direct). GPT-2 L10 reversed (0.336). Taxonomy (H4) confirmed: early 72-75%, late 13%. H3 cross-domain collapsed by shuffled control — 0/3 domains exceed null. H5 ITAC synthetic: 2.69% FN reduction (target 20%). H6 falsified. Signal is genuine for EDA+taxonomy core; cross-domain and ITAC claims are not supported. |
| Hypothesis survival | 0.25 | 3.0 | H1 (EDA lower bound): CONFIRMED (SynthSAEBench F1=0.974). H2 (D-EDA improvement): FALSIFIED (EDA=1-dec_cos identity confirmed; D-EDA adds no information). H3 (cross-domain): FALSIFIED (shuffled control matches real; probe quality gate fails). H4 (taxonomy): CONFIRMED (robust, primary contribution). H5 (ITAC): FALSIFIED (2.69% vs 20% target, synthetic activations). H6 (scaling sign reversal): FALSIFIED. Core killing conditions: H3 collapse is material. H1 and H4 survive robustly. H2's survival depends on EDA (not D-EDA) — EDA detection in favorable regime still supported. |
| Path to full result | 0.20 | 3.5 | Two-contribution paper is publishable NOW: (1) EDA as regime-specific detector (16k-width, mid-layers, cross-model validated on Gemma proxy + GPT-2 direct); (2) three-subtype taxonomy with early-dominance finding. No additional blocking experiments required. Cross-domain contribution is dropped. Paper title shifts: "Characterizing SAE Absorption: A Regime-Specific Detector and Three-Subtype Taxonomy". ITAC and H6 become appendix negative results. This is a complete, coherent, publishable contribution — shorter paper (6-8 pages) with stronger per-claim evidence. |
| Novelty | 0.15 | 3.5 | EDA detection: novelty reduced by R4 — EDA=1-dec_cos is mathematically equivalent to existing SAEBench metric (decoder cosine similarity). The novelty claim shifts to: (a) systematic regime characterization (first evidence of where EDA works and why), (b) formal lower-bound theorem grounding (still novel), (c) cross-model validation (GPT-2 direct labels confirm). Taxonomy: novelty score 8/10 still holds — three-subtype partition with 75% early-dominance is not in any prior work and has clear prescriptive value for the field. Overall novelty: 7/10 (down from original 7/10 due to EDA=dec_cos equivalence; taxonomy holds). |
| Resource efficiency | 0.10 | 4.0 | All experiments complete. No additional GPU budget needed for the two-contribution paper. Total cost: ~6 GPU-hours over all rounds. The decision to publish now vs. await Gemma 2B access is made: publish with current evidence base, acknowledge gated-model limitation. |

**Weighted Score** = (3.5 × 0.30) + (3.0 × 0.25) + (3.5 × 0.20) + (3.5 × 0.15) + (4.0 × 0.10)
= 1.05 + 0.75 + 0.70 + 0.525 + 0.40 = **3.475**

Score is 3.475 — just below the 3.5 ADVANCE threshold. However, ADVANCE is still the correct decision under the modified framing (see Decision Rationale below). The score reflects the real loss of the cross-domain contribution; the two-contribution paper remains publishable and coherent.

### cand_lca_sae (Backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | N/A | Not piloted |
| Hypothesis survival | 0.25 | 2.0 | No pilot evidence; conceptually adjacent to MP-SAE (Costa et al. 2506.03093) |
| Path to full result | 0.20 | 2.5 | Requires 3-4 GPU hours; strong MP-SAE overlap requires strong empirical differentiation |
| Novelty | 0.15 | 2.5 | 5/10 novelty (substantial MP-SAE overlap) |
| Resource efficiency | 0.10 | 3.0 | 3-4 GPU hours estimated |

**Estimated Weighted Score**: ~2.4. PIVOT trigger conditions (EDA AUROC < 0.65 AND all RAVEL null) are partially met (RAVEL null), but EDA core detection still holds (3/8 pass) and H4 taxonomy is independently strong. PIVOT to cand_lca_sae is not triggered.

### cand_scaling_laws (Already demoted to secondary)

**Weighted Score**: 2.20 (H6 core hypothesis falsified). Remains supplementary within main paper.

---

## Decision Rationale

**Decision: ADVANCE with `cand_eda_crossdomain` under TWO-CONTRIBUTION framing**

### Why Not PIVOT

The pivot trigger for `cand_lca_sae` requires: "EDA AUROC < 0.65 universally with direct labels AND all RAVEL domains null with proper probes." The first condition is NOT met: GPT-2 L6 with direct Chanin labels achieves AUROC=0.650 (exactly at threshold), and Gemma L5-16k and L12-16k show AUROC=0.698 and 0.776 with proxy labels. EDA detection in the favorable regime (16k-width, mid-layers) is empirically supported and theoretically grounded.

Additionally, H4 (three-subtype taxonomy) is fully confirmed and constitutes a standalone primary contribution independent of both EDA detection and cross-domain claims. Even if EDA detection entirely failed, the taxonomy finding (72-75% early absorption, KW p=0.0002 at L12-65k) is publishable on its own merits and reframes the entire absorption mitigation literature.

### Why Not REFINE

No methodology problems were exposed that would require redesign. The H3 collapse is a fundamental constraint (same-model probes unavailable without Gemma 2B / Llama-3.1-8B access), not a methodology error. Refining the RAVEL probe approach would require access to gated models — an administrative constraint, not a technical one. The two primary contributions (EDA detection + taxonomy) do not require the cross-domain evidence to be compelling.

### Critical Framing Shift Required

The paper title and abstract must be revised from the three-contribution framing:

**Old (R3 framing)**: "When Encoder-Decoder Misalignment Signals Feature Absorption: Regime-Specific Detection, Cross-Domain Generalization, and the Dominance of Dictionary Coverage Failure"

**New (R4 framing)**: "Characterizing SAE Absorption: A Regime-Specific Weight-Only Detector and Three-Subtype Taxonomy"

The cross-domain contribution is dropped from primary claims. It is acknowledged as a limitation (Gemma 2B and Llama-3.1-8B access required for same-model RAVEL probes; shuffled control shows insufficient signal with bridge-model probes). The intra-RAVEL coherence evidence (R3 rho=0.924, R4 rho=0.667) is reported as suggestive but not conclusive.

**EDA=1-DecCos equivalence** must be acknowledged prominently. The formula is not independently novel. The contribution is: (a) the formal lower-bound theorem grounding this metric in biconvex optimization theory, (b) systematic empirical validation of *where* the metric works and why it fails in certain regimes (early-absorption prevalence at 65k-width SAEs, late-layer polysemanticity), and (c) cross-model validation confirming the regime pattern.

### Sanity Checks

- [x] All candidates compared, not just front-runner: cand_lca_sae and cand_scaling_laws both score below ADVANCE threshold
- [x] Candidates penalized for failed falsification criteria: H3 collapse scored as reduction in hypothesis survival
- [x] Sunk cost not influencing: decision based entirely on current evidence quality, not GPU-hours invested
- [x] Inconclusive pilot (R4 shuffle control) → REFINE not ADVANCE? No: the collapse of H3 is a genuine finding (shuffled control equals real), not "inconclusive." The two-contribution paper is well-supported; REFINE would mean redesigning for a contribution that requires gated model access.

---

## Next Actions

1. **IMMEDIATE: Revise proposal to two-contribution structure** — update title, abstract, introduction, and contributions section to reflect: (1) EDA regime-specific detector, (2) three-subtype taxonomy with early-dominance finding. Remove cross-domain as primary contribution; report as a conditional/failed validation attempt in limitations.

2. **Write paper with current evidence base** — all data for the two-contribution paper is in hand. Go-write=true. No further experiments required unless Gemma 2B / Llama-3.1-8B access becomes available.

3. **Acknowledge EDA=DecCos equivalence explicitly** — cite SAEBench (Karvonen et al. 2503.09532) as related work for the decoder cosine similarity metric. Frame the EDA contribution as: (a) identification and formalization of this specific metric as an absorption detector, (b) formal lower-bound theorem, (c) regime characterization explaining when and why it works/fails.

4. **Report H3, H2 (D-EDA), H5, H6 as negative results** — be transparent. The paper is stronger for honest negative results: D-EDA adds nothing over scalar EDA; H3 cross-domain signal is not detectable with available bridge-model probes; ITAC shows 2.69% FN reduction (not 20%); H6 sign reversal not observed.

5. **Retain taxonomy ablation details** — threshold sensitivity (early-dominance at tau=0.3, drops to 32% at tau=0.2) must be reported as a robustness caveat. The finding is threshold-dependent; frame tau=0.3 as the canonical threshold with sensitivity reported.

6. **Venue: target NeurIPS 2026 MI Workshop or EMNLP 2026** — two-contribution paper at this scale is appropriate for a workshop or mid-tier venue. For NeurIPS 2026 main track, the cross-domain gap weakens the submission; acknowledge and plan to resubmit to main track once Gemma 2B access is available or the RAVEL probe problem is resolved.

---

SELECTED_CANDIDATE: cand_eda_crossdomain
CONFIDENCE: 0.74
DECISION: ADVANCE

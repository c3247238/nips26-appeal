# Novelty Report

**Research Topic**: The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains
**Assessment Date**: 2026-04-15
**Search Coverage**: 12 arXiv-targeted queries, 12 web search queries, 50+ papers reviewed (including full literature survey)

---

## Executive Summary

**Overall novelty: HIGH.** The front-runner candidate's core contribution -- the first systematic cross-domain absorption characterization beyond the first-letter spelling task -- has no direct prior work as of April 2026. All published absorption measurements are restricted to the spelling task. The striking pilot finding that first-letter shows the LOWEST absorption (3.9%) while semantic hierarchies show substantially higher rates (10.4%-53.4%) is unprecedented and reframes the entire absorption literature. All backup candidates also have high novelty.

---

## Candidate 1 (Front-Runner): Cross-Domain Absorption Characterization + Causal Mechanism + Absorption Tax

**Candidate ID**: `cand_crossdomain_causal_tax`
**Novelty Score**: 8/10
**Recommendation**: PROCEED

### Core Contribution Claims Checked

#### Claim 1: First cross-domain absorption measurement (Novelty 9/10 in proposal)

**Search result: GENUINELY NOVEL.** Every published absorption measurement -- Chanin et al. (2024), SAEBench (2025), SynthSAEBench (2026), HSAE (2026), H-SAE (2025), OrtSAE (2025), KronSAE (2025), ATM (2025), masked regularization (2026) -- uses the first-letter spelling task exclusively. SAEBench includes RAVEL as a *separate* disentanglement metric, but does NOT use RAVEL entity-attribute hierarchies to measure *absorption rates*. SynthSAEBench (2026) studies hierarchy effects on absorption but only in synthetic settings with known ground truth, not on real LLM knowledge hierarchies.

No paper was found that measures absorption on city-country, city-continent, city-language, or any other entity-attribute hierarchy. The proposal's extension of absorption measurement to RAVEL-based semantic hierarchies on Gemma 2 2B is a first.

**Closest prior work**:
- SAEBench (arXiv:2503.09532): RAVEL used for disentanglement, not absorption. Severity: *related_work*.
- SynthSAEBench (arXiv:2602.14687): Synthetic hierarchy absorption. Severity: *partial_overlap* (synthetic only, no real entity hierarchies).
- HSAE (arXiv:2602.11881): Evaluated on SAEBench absorption (first-letter) and RAVEL (disentanglement, not absorption). Severity: *partial_overlap*.

**Novelty assessment: 9/10 confirmed.** The finding that first-letter is atypical (lowest absorption) strengthens this further.

#### Claim 2: Causal absorption via activation patching (Novelty 8/10 in proposal)

**Search result: GENUINELY NOVEL.** Chanin et al. (2024) use integrated-gradients ablation to *detect* absorption, which is correlational. Their ablation study zeros SAE latents and measures downstream logit change, but this tests whether a latent is causally important for the task -- it does not test whether zeroing a *child* feature *recovers* a *parent* feature's activation. The proposal's activation patching protocol (zero child feature, measure parent probe recovery rate) is a distinct and novel causal test for competitive exclusion dynamics.

No paper was found that uses activation patching specifically to demonstrate that child features suppress parent features in an SAE.

**Novelty assessment: 8/10 confirmed.**

#### Claim 3: Benign vs. pathological absorption diagnostic (Novelty 8/10 in proposal)

**Search result: GENUINELY NOVEL.** No published work distinguishes between benign absorption (computationally redundant parent) and pathological absorption (parent has independent causal effects). The term "benign vs. pathological absorption" does not appear in any search result. The community discusses whether absorption is harmful in general terms (e.g., LessWrong discussions, Contrarian perspectives), but no paper operationalizes this distinction via downstream logit measurement.

**Novelty assessment: 8/10 confirmed.**

#### Claim 4: Tightened hedging classification (Novelty 8/10 in proposal)

**Search result: NOVEL CONTRIBUTION.** Chanin et al. (2025) define feature hedging and study the absorption-hedging tradeoff, but do not provide a strict vs. loose hedging classification. The proposal's finding that strict classification yields 7.4% vs. loose 92.6% (with 85.3% compensatory resolution) is a new methodological contribution that reveals the near-tautological nature of the widely cited hedging figure.

**Novelty assessment: 8/10 confirmed.**

#### Claim 5: Rate-distortion predictor framework (Novelty 7/10 in proposal)

**Search result: NOVEL.** The three-factor model (cos_sim x co-occurrence / reconstruction importance) has no direct precedent. The unified SDL theory (arXiv:2512.05534) provides theoretical background on absorption causes but does not propose or test this specific predictor framework empirically. The "Geometry of Concepts" paper (arXiv:2410.19750) studies co-occurrence vs. cosine similarity relationships but not as predictors of absorption.

**Novelty assessment: 7/10 confirmed.** Downgraded from higher because pilot evidence shows weak quantitative predictions, which the proposal honestly acknowledges.

#### Claim 6: "Absorption Tax" concept

**Search result: NOVEL TERM AND FRAMEWORK.** The phrase "Absorption Tax" does not appear in any published work. The concept -- the minimum additional L0 cost for absorption-free representation -- is related to Chanin et al.'s observation that absorption saves one L0 per parent-child pair, but formalizing this as T(G) per hierarchy type is new.

### Collision Risk Assessment

**Low risk.** No competing cross-domain absorption characterization was found. The field is active on architectural *mitigations* (Matryoshka, OrtSAE, ATM, KronSAE, masked regularization, HSAE) but NOT on cross-domain *characterization*. The closest potential competitor would be a group extending SAEBench's absorption metric to RAVEL hierarchies, but no evidence of this was found.

---

## Candidate 2 (Backup): Controlled Dictionary Experiment

**Candidate ID**: `cand_controlled_dictionary`
**Novelty Score**: 8/10
**Recommendation**: PROCEED (as supplementary or if front-runner hits obstacles)

**Search result: GENUINELY NOVEL.** No prior work holds the decoder dictionary constant while varying the encoder (feedforward vs. OMP vs. 2-pass). MP-SAE (Matching Pursuit SAE) in SynthSAEBench uses a different encoding strategy but does NOT hold the dictionary constant -- it trains encoder and decoder jointly. The Select-and-Project approach (arXiv:2509.10809) explores encoder-only features but does not study absorption under controlled dictionary conditions.

**Closest prior work**:
- SynthSAEBench MP-SAE evaluation: Different encoder but joint training. Severity: *partial_overlap*.
- Unified SDL theory: Analyzes encoder-decoder interaction theoretically. Severity: *related_work*.

---

## Candidate 3 (Backup): Ecological Phase Transition

**Candidate ID**: `cand_ecological_phase_transition`
**Novelty Score**: 9/10
**Recommendation**: PROCEED (as supplementary analysis when Phase 1 data available)

**Search result: GENUINELY NOVEL.** Zero prior work applies Lotka-Volterra competitive exclusion or phase transition frameworks to SAE feature absorption. The ecological competition analogy has no precedent in the SAE or mechanistic interpretability literature. The prediction of sharp phase transitions at critical cosine similarity thresholds is entirely novel.

**Closest prior work**:
- Tonolo et al. (2025) on generalized LV models with sparse interactions: Pure ecology, no SAE connection.
- Chanin et al. (2024): Informal competition observation, not formalized.

---

## Candidate 4 (Backup): Absorption-Aware Post-Hoc Correction

**Candidate ID**: `cand_absorption_aware_correction`
**Novelty Score**: 8/10
**Recommendation**: PROCEED (if reviewers ask for prescriptive guidance)

**Search result: GENUINELY NOVEL.** No published work proposes or evaluates a post-hoc correction method for absorbed features at inference time. All existing mitigations are training-time modifications (Matryoshka, OrtSAE, ATM, KronSAE, masked regularization, HSAE, feature anchoring). LessWrong discussions explicitly note that post-hoc correction is unexplored.

---

## Dropped Candidates: Collision Confirmation

| Candidate | Drop Reason | Collision Confirmed? |
|-----------|------------|---------------------|
| `cand_gas_primary` | GAS refuted by pilot (rho=0.12) | N/A (empirically falsified) |
| `cand_cmi_taxonomy` | CMI refuted by pilot (rho=0.044) | N/A (empirically falsified) |
| `cand_absorption_tax_quantitative` | Quantitative predictions not supported (rho=0.08) | N/A (empirically downgraded) |
| `cand_eda_universal` | Replaced by GAS, which also failed | N/A |
| `cand_itac_primary` | Core unsupervised hypothesis refuted | N/A |
| `cand_hierarchy_coherent_loss` | Crowded by Muchane et al. (arXiv:2506.01197), Luo et al. HSAE (arXiv:2602.11881), KronSAE | YES -- crowded field. Correct to drop. |
| `cand_scaling_laws` | Low novelty (5/10), incorporated | N/A |
| `cand_pac_bayes_generalization` | Tangential, bound too loose | N/A |

---

## Summary of Novelty Assessment

| Candidate | Score | Recommendation | Key Differentiator |
|-----------|-------|---------------|-------------------|
| Cross-domain + Causal + Tax (front-runner) | 8/10 | **PROCEED** | First cross-domain absorption measurement; first-letter is atypical (lowest absorption) |
| Controlled Dictionary (backup) | 8/10 | PROCEED | First encoder-decoder separation experiment for absorption |
| Ecological Phase Transition (backup) | 9/10 | PROCEED | Zero precedent for LV/phase transition framework in SAE absorption |
| Post-Hoc Correction (backup) | 8/10 | PROCEED | All existing mitigations are training-time; inference-time correction unexplored |

**Note on front-runner score**: The proposal self-assessed its primary contribution at 9/10 novelty, which is defensible. I rate the overall candidate at 8/10 (rather than 9/10) because: (a) the absorption metric framework is adapted from Chanin et al., not invented from scratch; (b) RAVEL and Gemma Scope are existing tools being composed in a new way; (c) the secondary/tertiary contributions (rate-distortion predictors, Absorption Tax) have weak pilot support. However, the core finding -- that semantic hierarchies show HIGHER absorption than spelling, inverting field assumptions -- is a 9/10 finding if confirmed in full-mode experiments with quality probes.

---

## Potential Risks to Novelty

1. **SAEBench v2 could add cross-domain absorption**: If the SAEBench team extends their absorption metric to RAVEL entity hierarchies, this would be the most direct collision. No evidence of this found, but SAEBench is actively maintained.

2. **Chanin et al. follow-up**: The Chanin et al. group (who defined absorption) could extend their own work to knowledge hierarchies. No evidence of this found.

3. **HSAE / H-SAE groups**: The hierarchical SAE groups (Luo et al., Muchane et al.) could study absorption on their own hierarchies. Current papers only evaluate on SAEBench first-letter task.

4. **Field velocity**: The SAE interpretability field is moving fast (multiple 2026 papers already). Speed of execution matters.

**Recommendation**: Proceed with execution promptly. The novelty window is open but the field is active.

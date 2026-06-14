# Novelty Report

**Date:** 2026-04-14 (Round 3 update; prior versions: 2026-04-11 Round 2, 2026-04-10 Round 1)
**Workspace:** sae-absorption
**Agent:** sibyl-novelty-checker
**Candidates evaluated:** 4 (cand_eda_crossdomain [front-runner], cand_amortization_gap_experiment, cand_lca_sae, cand_scaling_laws)
**Proposal revision:** Round 3 (post-experimental). Key structural changes from Round 2:
- EDA repositioned from universal detector to regime-specific screening tool
- D-EDA demoted to secondary/supplementary
- ITAC demoted to proof-of-concept (3% vs. 20% target)
- Early-absorption dominance (~75%) elevated to primary contribution
- Amortization gap controlled experiment added as backup candidate

---

## Summary (Round 3 Update — Post-Experimental)

The Round 3 proposal retains HIGH novelty. All three primary contributions — (1) EDA as regime-specific weight-only absorption screening tool with formal biconvex lower bound, (2) first cross-domain absorption characterization via RAVEL entity-attribute hierarchies, and (3) three-subtype taxonomy with 75% early-absorption dominance insight — remain genuinely novel. The restructuring from Round 2 (demoting D-EDA and ITAC; elevating early-absorption dominance) does not weaken novelty; if anything, the 75% early-dominance finding is the proposal's strongest novelty claim because it directly contradicts the implicit framing of all ITAC-style and inference-time mitigation literature.

The new backup candidate (cand_amortization_gap_experiment) is assessed for the first time; it occupies very high novelty territory (9/10) because no published work holds the dictionary constant while comparing encoder quality across feedforward, OMP, and 2-pass encoders.

Fresh searches (April 2026) confirm no new paper has appeared that: (a) formalizes encoder-decoder angular divergence as a theoretical absorption lower bound with AUROC validation, (b) measures SAE feature absorption in entity-attribute hierarchies beyond first-letter, or (c) provides a three-way taxonomy with the decoder-absent vs. decoder-present distinction.

**Overall novelty: HIGH** — proceed with cand_eda_crossdomain as front-runner.

---

## Candidate 1: cand_eda_crossdomain (Front-runner) — Round 3

### Core Claims (Round 3 — Revised from Round 2)

1. **EDA as regime-specific weight-only absorption screening tool** (not universal): probe-free, computed from SAE weight matrices as `EDA(j) = 1 - cos(w_{e,j}, d_j)`, with formal biconvex lower bound from Tang et al. (2512.05534) optimization theory. AUROC = 0.776 at L12-16k; Cohen's d > 0.84 across favorable configs. Layer-width dependency is a feature of the theory, not a bug — characterizes which absorption types have an encoder-decoder geometric footprint.

2. **First cross-domain absorption characterization**: RAVEL entity-attribute hierarchies (city-continent, city-country, city-language); all 18 measurements above 3x random baseline; intra-RAVEL coherence rho = 0.924. Pending Gemma 2B probe validation for absolute rates.

3. **Three-subtype taxonomy with early-absorption dominance insight**: Early (~75%): decoder-absent, dictionary coverage failure; Late (~13%): decoder-present, encoder-suppressed; Partial (~12%): selective failure. The 75% early-dominance finding reframes the entire mitigation literature — most proposed fixes (OrtSAE at inference, ITAC, Select-and-Project) target the minority late-absorbed category.

4. **Secondary/proof-of-concept** (D-EDA and ITAC): D-EDA as complementary detector in GPT-2 L10 where scalar EDA fails; ITAC as proof-of-concept applicable only to late-absorbed (~13%) latents.

---

### Prior Art Search Results — Round 3 Updates

#### Claim 1: EDA Regime-Specific Detector with Formal Lower Bound

**Searches conducted (April 2026):**
- "encoder decoder alignment SAE absorption weight-only detector formal bound 2025" — most relevant hit remains the LessWrong post (informal); no formal EDA detector found
- "weight-only unsupervised absorption detection sparse autoencoder without probe direction 2024 2025" — KronSAE and OrtSAE address absorption via architectural training changes; no weight-only detection metric found
- "feature absorption sparse autoencoder encoder direction decoder direction misalignment metric detector 2025" — returned only the LessWrong "Toy Models" post as a relevant hit

**Key papers found and assessed (Round 3 additions/updates):**

| Paper | Relevance | Severity | New in Round 3? |
|-------|-----------|----------|-----------------|
| Chanin et al., "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025) | Supervised probe-based detection; does not formalize encoder-decoder alignment as metric | related_work | No |
| "Toy Models of Feature Absorption" (Chanin et al., LessWrong, Oct 2024) | Informal suggestion to compare encoder vs. decoder direction top activations to detect absorption; no metric, no theorem, no AUROC | **partial_overlap** | No |
| Tang et al., "Unified Theory of SDL" (arXiv:2512.05534, Dec 2025) | Formal biconvex theory grounding the EDA lower bound theorem; proposes "feature anchoring" to fix identifiability; does NOT use EDA as detector | related_work (theoretical foundation) | No |
| OrtSAE (Korznikov et al., arXiv:2509.22033, Sept 2025) | Penalizes pairwise cosine similarity BETWEEN decoder features; requires retraining; not a detection metric | related_work | No |
| KronSAE (Kurochkin et al., arXiv:2505.22255, May 2025) | Architectural absorption reduction; measures absorption post-hoc via first-letter metric; not encoder-decoder alignment based | related_work | No |
| Feature Hedging paper (arXiv:2505.11756) | Notes encoder-decoder asymmetry has opposite sign for hedging vs. absorption; does not formalize into a scalar detection metric | related_work | No |
| SynthSAEBench (Chanin et al., arXiv:2602.14687, Feb 2026) | Synthetic benchmark with ground-truth absorption; validates Phase 0 (F1 = 0.974 on first-letter ground truth); does not propose encoder-decoder alignment as detector | related_work | Yes |
| Narayanaswamy et al. (arXiv:2604.06495, Apr 2026) | Masked regularization for absorption reduction; uses first-letter metric only; requires retraining; no EDA | related_work | No (confirmed still no EDA) |
| "Mechanistic Interpretability Should Prioritize Feature..." (arXiv:2505.20254) | Argues for feature-level rather than latent-level interpretability; does not propose encoder-decoder alignment metric | related_work | Yes |

**Verdict on EDA (Round 3):** NOVEL — unchanged from Round 2. The repositioning from "universal detector" to "regime-specific screening tool" does not weaken novelty; it makes the claim more precise and defensible. No paper has yet formalized encoder-decoder angular divergence as a detection metric with a formal theorem, AUROC validation against supervised labels, and regime characterization. The regime-dependency finding (EDA works at mid-layers, narrow SAEs; fails at wide/late-layer) is itself a novel empirical contribution consistent with the theory.

**Key differentiation note for paper (updated for Round 3):** The LessWrong "Toy Models" informal suggestion must be cited; EDA's contribution over it is: (a) precise scalar metric definition, (b) formal biconvex lower bound theorem connecting EDA > 0 to absorption degree delta, (c) AUROC validation against supervised Chanin et al. labels, (d) regime characterization explaining when EDA has discriminative power (mid-layer, narrow SAEs) and why (early-absorbed latents — 75% of all absorbed — are theoretically expected to have low EDA signal because the encoder and decoder are both absent; only late-absorbed latents should elevate EDA).

---

#### Claim 2: First Cross-Domain Absorption Characterization

**Searches conducted (April 2026):**
- "RAVEL entity attribute hierarchy sparse autoencoder feature absorption cross-domain 2025" — RAVEL is used for disentanglement evaluation; no SAE absorption measurement in entity-attribute hierarchies found
- "SAE feature absorption cross-domain entity attributes generalization entity types safety concepts beyond spelling task 2025" — confirms all published absorption measurements use first-letter task; SAEBench absorption metric is exclusively first-letter
- Checked HSAE (arXiv:2602.11881) — uses RAVEL as disentanglement benchmark, NOT as absorption measurement benchmark

**Key papers found and assessed:**

| Paper | Relevance | Severity | New in Round 3? |
|-------|-----------|----------|-----------------|
| Chanin et al. 2409.14507 | Only first-letter absorption measurements | related_work | No |
| SAEBench (Karvonen et al., arXiv:2503.09532) | Absorption metric exclusively first-letter; RAVEL metric is disentanglement evaluation, not absorption measurement | related_work | No |
| HSAE (Luo et al., arXiv:2602.11881) | Evaluates on SAEBench RAVEL (disentanglement); does NOT measure absorption in entity-attribute hierarchies | related_work | Yes |
| RAVEL (Huang et al., arXiv:2402.17700) | Provides the entity-attribute dataset and disentanglement evaluation; does NOT measure feature absorption | related_work | No |
| Chanin et al., SynthSAEBench (arXiv:2602.14687) | Synthetic hierarchical absorption in first-letter-style tasks; no cross-domain entity-attribute measurement | related_work | Yes |

**Verdict on Cross-Domain Characterization (Round 3):** NOVEL — unchanged. No paper has measured SAE feature absorption in semantic entity-attribute hierarchies. The finding that intra-RAVEL rho = 0.924 (absorption rankings are stable within the entity-attribute family) and that cross-paradigm first-letter vs. RAVEL correlations are *negative* (rho = -0.43 to -0.20) is entirely new. The negative cross-paradigm correlation is the most hypothesis-generating finding — it suggests that the two most commonly studied absorption paradigms may capture different phenomena (syntactic vs. semantic hierarchy absorption regimes). This appears in no prior paper.

---

#### Claim 3: Three-Subtype Taxonomy with Early-Absorption Dominance (75%)

**Searches conducted (April 2026):**
- "feature absorption sparse autoencoder subtype classification geometric early late absorption decoder present absent 2025" — only returned Chanin et al. (2409.14507) and related work; no subtype taxonomy found
- "SAE absorption early late partial subtype dictionary coverage encoder failure retraining 2025" — no taxonomy found; confirms dictionary coverage vs. encoder failure as an open empirical question
- "SAE absorption dictionary coverage failure versus encoder alignment failure empirical 2025" — search confirmed this question is open; no paper has empirically distinguished the two causes

**Key papers found and assessed:**

| Paper | Relevance | Severity | New in Round 3? |
|-------|-----------|----------|-----------------|
| Chanin et al. 2409.14507 | Defines absorption as single category; informal split into encoder-absent vs. encoder-suppressed implicitly shown in toy model (late absorption only) | related_work | No |
| "Toy Models" (LessWrong, Oct 2024) | Shows one subtype (late absorption: decoder-present, encoder-suppressed) in toy model; no decoder-absent category; no partial subtype; no prevalence measurement | partial_overlap (weak) | No |
| Tang et al. 2512.05534 | Optimization landscape theory of absorption; does not subdivide into subtypes; does not distinguish encoder-absent from encoder-suppressed | related_work | No |
| MP-SAE (Costa et al., arXiv:2506.03093) | Shows iterative encoding reduces absorption; implicitly targets late absorption (decoder-present type); does not classify subtypes or report early-dominance | related_work | No |
| O'Neill et al. (arXiv:2411.13117, ICML 2025) | Proves amortization gap in SAE encoders; proposes OMP as improved encoder; does not classify subtypes or report early-dominance prevalence | related_work | No |
| Feature Hedging paper (arXiv:2505.11756) | Notes encoder-decoder asymmetry differs for hedging vs. absorption; does not formalize into a subtype taxonomy | related_work | No |

**Verdict on Three-Subtype Taxonomy (Round 3):** NOVEL — and the contribution is strengthened in Round 3. The 75% early-absorption dominance finding is the most practically impactful result in the paper and is not anticipated by any prior work. It directly contradicts the implicit assumption of the entire ITAC-style and inference-time correction literature — these approaches all assume late absorption (encoder-suppressed, decoder-present) is the dominant or prevalent type. The finding that 75% of absorbed latents are early-type (decoder-absent, dictionary coverage failure) fundamentally reframes which interventions are effective: retraining with wider dictionaries or hierarchy-aware objectives, not inference-time encoder corrections.

**Specific novelty of the three-way partition vs. prior work:**
- The "early" category (decoder-absent: SAE never allocated capacity to parent feature) was not formally proposed anywhere before this work
- The remediability prediction (ITAC works for late but not early; confirmed by 3% vs. 20% FN reduction with early-type dominance explaining the gap) constitutes a testable, confirmed prediction absent from any prior paper
- The partial subtype's unexpectedly low EDA (below early-type) is a new empirical finding suggesting the partial subtype may be geometrically more similar to early absorption than to late absorption

---

#### Secondary Claims: D-EDA and ITAC (Demoted in Round 3)

**D-EDA assessment (unchanged):** NOVEL. No paper proposes decomposing the SAE encoder residual (after subtracting projection onto own decoder direction) as a sparse sum of other decoder columns to distinguish absorption from polysemanticity. The numerical conditioning limitation at large widths (d_sae >> d_model) is an honest finding; the GPT-2 L10 exception (D-EDA = 0.762 vs. EDA = 0.336) shows genuine complementary signal. Reporting as secondary/supplementary with honest conditioning analysis.

**ITAC proof-of-concept assessment (repositioned but still novel):** NOVEL in principle. The 3% vs. 20% target miss, explained mechanistically by 75% early-absorption dominance, is itself an important negative result that confirms the early-dominance finding. No paper proposes training-free post-hoc absorption correction for frozen SAEs; ITAC remains the only such method. The failure story is publishable: ITAC's structural limitation (only applies to ~13% of absorbed latents) follows directly from the taxonomy finding, making ITAC's failure a confirmation of the taxonomy rather than a refutation of ITAC's underlying approach.

**No new papers threatening ITAC novelty were found in April 2026 searches.** The ITDA paper (arXiv:2505.17769) is completely different (dictionary construction via matching pursuit, not absorption correction).

---

### Novelty Score: 7/10 (overall for the Round 3 bundled three-contribution paper)

**Per-contribution scores:**
- EDA regime-specific detector: 7/10 — formal metric + theorem + regime characterization are novel; LessWrong seed overlap remains; repositioning as regime-specific (not universal) makes the claim stronger
- Early-absorption dominance + three-subtype taxonomy: 9/10 — most novel result in the paper; no competing paper; directly contradicts implicit assumptions of the mitigation literature
- Cross-domain characterization: 9/10 — cleanest novelty claim; no prior work; negative cross-paradigm correlation is bonus hypothesis-generating finding

**Recommendation: PROCEED** with cand_eda_crossdomain. All three primary contributions are novel and the paper's framing is strengthened by the honest reporting of Round 3 experimental results.

**Round 3 specific actions required:**
1. Cite LessWrong post (Chanin et al., Oct 2024) explicitly in Related Work; frame EDA as formalization + theorem + AUROC validation + regime characterization of the informal heuristic
2. Cite Tang et al. 2512.05534 as the theoretical foundation for the biconvex lower bound
3. Cite MP-SAE (Costa et al., 2506.03093) in both ITAC and taxonomy sections; clarify that MP-SAE implicitly targets late absorption (decoder-present type), that early absorption (~75%) would not be reduced by MP-SAE, and that the early-dominance finding has implications for MP-SAE's expected real-world impact
4. Cite O'Neill et al. 2411.13117 as a theoretical complement: amortization gap proves encoder is limited, but this work empirically distinguishes which absorbed latents are encoder-limited vs. dictionary-coverage-limited
5. Cite SynthSAEBench 2602.14687 for Phase 0 validation
6. Cite Narayanaswamy et al. 2604.06495 as concurrent mitigation work using first-letter metric; note EDA focuses on detection/characterization, not mitigation
7. State explicitly in Related Work: "No prior work has (a) formalized encoder-decoder angular divergence as a formal absorption lower bound, (b) measured absorption in semantic entity-attribute hierarchies, or (c) provided an empirically validated three-way partition of absorbed latents by decoder occupancy and remediability."

---

## Candidate 2: cand_amortization_gap_experiment (Backup) — First Assessment

### Core Claim

Controlled dictionary experiment: Take the SAME pre-trained decoder dictionary D from a Gemma Scope SAE and measure absorption rates under standard feedforward encoding, OMP encoding, and 2-pass encoding. Separates sparsity-landscape from amortization-gap causes of early absorption (~75%). If OMP dramatically reduces early absorption with the same dictionary, the encoder is implicated in early absorption causation.

---

### Prior Art Search Results

**Searches conducted (April 2026):**
- "controlled dictionary experiment OMP feedforward encoder absorption rate sparse autoencoder mechanistic interpretability 2025" — no paper found combining these elements
- "OMP sparse autoencoder amortization gap absorption encoder quality O'Neill 2025" — found O'Neill et al. 2411.13117 (amortization gap proof) and SynthSAEBench
- "MP-SAE Costa multi-pass iterative encoder hierarchical feature absorption 2025" — confirmed MP-SAE uses matching pursuit iteratively but does NOT hold dictionary constant

**Key papers found and assessed:**

| Paper | Relevance | Severity |
|-------|-----------|----------|
| O'Neill et al. (arXiv:2411.13117, ICML 2025) | Proves amortization gap; compares feedforward SAE vs. OMP/SAE+ITO on same dictionary; measures MCC (feature recovery), not absorption rate; does not specifically measure absorption rate in SAEBench sense | partial_overlap |
| MP-SAE (Costa et al., arXiv:2506.03093) | Iterative encoder; does NOT hold dictionary constant; trains a new encoder jointly with absorption measurement | related_work |
| SynthSAEBench (Chanin et al., arXiv:2602.14687) | Identifies that MP-SAEs exploit superposition noise; measures MCC on ground-truth features; does not use SAEBench absorption metric | related_work |
| Tang et al. 2512.05534 | Optimization landscape theory; proposes feature anchoring; does not compare feedforward vs. OMP at fixed dictionary | related_work |

**Verdict on cand_amortization_gap_experiment:** HIGH NOVELTY with one important partial overlap.

O'Neill et al. (2411.13117) compares feedforward encoder vs. OMP-style inference on a fixed decoder dictionary, but their metric is **MCC** (feature recovery quality, a correlation with ground-truth synthetic features) — NOT the SAEBench absorption rate metric (probe-based recall on hierarchically organized features). The proposed controlled experiment specifically measures **absorption rate** (parent feature recall failures when child is active) under feedforward vs. OMP encoding with a fixed Gemma Scope decoder dictionary. This is a distinct and complementary measurement. O'Neill et al. do not test on real-world Gemma Scope SAEs; they use synthetic data. The proposed experiment uses real Gemma Scope L12-16k decoder weights on real activations.

**Novelty Score: 9/10** — first controlled experiment separating encoder quality from dictionary coverage as causes of the specific absorption failure mode (using SAEBench probe-based absorption metric on fixed real-world decoder dictionary).

**Recommendation: HIGH-PRIORITY SUPPLEMENTARY.** Provides direct empirical adjudication between Tang et al. (sparsity landscape → early absorption regardless of encoder) and O'Neill et al. (amortization gap → encoder causes early absorption). Cost: 1-2 GPU-hours, no new downloads. Activatable immediately. Should run alongside Priority 1 blocking experiments.

---

## Candidate 3: cand_lca_sae (Backup) — Round 3 (Unchanged from Round 2)

**Core Claim:** Replace feedforward SAE encoder with unrolled LCA iterations; lateral inhibition via decoder Gram matrix; reduce absorption without changing decoder weights.

**Novelty Score: 5/10** — MP-SAE (Costa et al., arXiv:2506.03093) covers iterative encoding for absorption reduction with only the mechanism differing (sequential greedy residual pursuit vs. parallel lateral competition). The LCA has biological plausibility that MP-SAE lacks, but this is not sufficient differentiation for a primary contribution.

**Recommendation: BACKUP ONLY.** Activate only if EDA AUROC < 0.60 on all configs with direct Chanin labels AND cross-domain results are null.

---

## Candidate 4: cand_scaling_laws (Backup) — Round 3 (H6 Falsified)

**Core Claim:** Partial correlation analysis: H6 falsified (rho = +0.37, no sign reversal). Wider SAEs consistently absorb more. Methodological explanation: insufficient L0 variation in canonical SAEs.

**Novelty Score: 6/10** — negative result with methodological framing; Chanin et al. 2409.14507 notes wider SAEs have more absorption but does not conduct partial correlation analysis; SAEBench confirms inverse scaling but does not test the confound hypothesis. The methodological note (canonical SAEs provide insufficient L0 variation to test the confound) is independently useful for the field.

**Recommendation: INCLUDE AS SUPPLEMENTARY** in the main paper. Move H6 falsification analysis to supplementary; mention methodological limitation in main paper discussion.

---

## Key Prior Art Summary Table (Round 3 — Updated)

| Paper | arXiv/Source | Relevance | Severity | Round |
|-------|-------------|-----------|----------|-------|
| Chanin et al., "A is for Absorption" | 2409.14507 (NeurIPS 2025) | Defines absorption; supervised metric; no formal EDA bound; no subtype taxonomy; no cross-domain | related_work | 1 |
| "Toy Models of Feature Absorption" | LessWrong, Oct 2024 | Informal encoder-decoder detection suggestion; one subtype (late) implicitly shown; seed of EDA idea | **partial_overlap** | 1 |
| Tang et al., "Unified Theory of SDL" | 2512.05534 | Biconvex theory grounding EDA lower bound; proposes feature anchoring (different approach) | related_work (theoretical foundation) | 1 |
| OrtSAE | 2509.22033 | Decoder inter-latent orthogonality penalty; reduces absorption during training; no detection metric | related_work | 1 |
| Matryoshka SAE | 2503.17547 | Hierarchical nested SAE reducing absorption during training | related_work | 1 |
| KronSAE | 2505.22255 | Kronecker encoder reducing absorption; measures absorption by first-letter metric | related_work | 1 |
| MP-SAE | 2506.03093, 2506.05239 | Iterative matching pursuit encoder; closest comparison to ITAC; requires encoder retraining; implicitly targets late absorption type | **partial_overlap (ITAC)** | 1 |
| SAEBench | 2503.09532 | Standard benchmark; absorption metric exclusively first-letter | related_work | 1 |
| RAVEL | 2402.17700 | Entity-attribute disentanglement benchmark; no absorption measurement | related_work | 1 |
| O'Neill et al. | 2411.13117 (ICML 2025) | Proves amortization gap; compares feedforward vs. OMP on MCC (not absorption rate) | related_work / **partial_overlap (cand_amortization_gap)** | 1 |
| Select-and-Project | 2509.10809 | Retraining-free encoder inference-time intervention; addresses steering, not absorption | related_work | 2 |
| Narayanaswamy et al. | 2604.06495 | Masked regularization reducing absorption; first-letter metric only; requires retraining | related_work | 2 |
| HSAE (Luo et al.) | 2602.11881 | Hierarchical SAE; uses RAVEL for disentanglement (not absorption) | related_work | 3 |
| SynthSAEBench | 2602.14687 | Synthetic SAE benchmark; validates Phase 0 F1 = 0.974; identifies MP-SAE exploits superposition noise | related_work | 3 |
| Feature Hedging | 2505.11756 | Encoder-decoder asymmetry for hedging vs. absorption (qualitative); no scalar metric | related_work | 3 |
| Korznikov et al. "SAEs Don't Beat Random Baselines" | 2602.14111 | Sanity check — SAEs recover 9% of true features | related_work (framing) | 1 |
| Gemma Scope | 2408.05147 | Pre-trained SAE suite used in experiments | infrastructure | 1 |

---

## Recommendations (Round 3)

### cand_eda_crossdomain (front-runner)

**PROCEED.** Three genuine novel contributions with no blocking prior art:

1. EDA regime-specific detector — formalized, validated, regime-characterized. Seed overlap from LessWrong post is citable and differentiable.
2. Cross-domain absorption characterization — cleanest novelty; no prior work; negative cross-paradigm correlation is a bonus hypothesis.
3. Three-subtype taxonomy with 75% early-dominance — most impactful result; directly reframes the mitigation literature; no prior work.

**Pre-writing required actions:**
1. Cite Chanin et al. LessWrong "Toy Models" post explicitly; differentiate EDA as formal theorem + AUROC + regime characterization
2. Cite Tang et al. 2512.05534 as theoretical grounding
3. Cite MP-SAE (2506.03093) noting it implicitly targets late absorption only; discuss early-dominance implication for MP-SAE's expected field impact
4. Cite O'Neill et al. 2411.13117 as complementary to taxonomy (amortization gap affects encoder, but which absorbed latents are encoder-limited vs. dictionary-limited is now answered)
5. Cite SynthSAEBench 2602.14687 for Phase 0 validation
6. Cite Narayanaswamy et al. 2604.06495 as concurrent mitigation work
7. Cite Feature Hedging paper 2505.11756 noting EDA specifically targets absorption (not hedging) and the sign distinction

### cand_amortization_gap_experiment (new backup)

**HIGH-PRIORITY SUPPLEMENTARY.** Run immediately as 1-2 GPU-hour experiment alongside Priority 1 blocking experiments. Directly adjudicates Tang et al. (landscape) vs. O'Neill et al. (amortization gap) explanations of early absorption. Highly novel (9/10). Include as supplementary experiment in main paper; potentially elevate to standalone contribution if result is dramatic (OMP < 20% of feedforward absorption rate).

### cand_lca_sae (backup)

**BACKUP ONLY.** Activate only if EDA AUROC < 0.60 universally AND cross-domain results null. Even then, must differentiate from MP-SAE.

### cand_scaling_laws (backup)

**SUPPLEMENTARY.** Include H6 falsification as supplementary negative result. The methodological note about canonical SAE L0 variation is useful for the field.

---

## Overall Novelty Assessment (Round 3)

**Overall: HIGH**

The Round 3 restructuring strengthens the paper's novelty position. The 75% early-absorption dominance finding — the highest-impact result — is entirely unantipated by prior work and directly falsifies the implicit assumption (late absorption dominance) underlying the entire ITAC-style mitigation literature including MP-SAE, OrtSAE at inference time, and Select-and-Project. The cross-domain measurement is the cleanest gap in the field; no paper has done it. The EDA formal detector, even repositioned as regime-specific, remains the only formalized weight-only absorption screening metric in the literature.

The two negative results (H5 ITAC at 3%, H6 scaling non-result) are honestly reported and strengthen rather than weaken the paper: ITAC's failure confirms the early-dominance finding; H6's falsification provides a methodological note about canonical SAE design. These are publishable negative results that the field needs.

The critical dependency (Gemma 2B authentication for direct Chanin labels) is a blocking experiment with a clear fallback (Llama-3.1-3B). The paper is publishable at EMNLP 2026 / NeurIPS 2026 MI Workshop with current results; the Gemma 2B validation is required for a stronger top-tier claim.

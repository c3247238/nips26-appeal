# Novelty Report: SAE Feature Absorption Research Proposal

**Search date:** April 13, 2026  
**Workspace:** `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current`  
**Candidates assessed:** cand_a (front-runner), cand_b, cand_c, cand_d  
**Prior report updated:** Yes — new papers from 2025–2026 incorporated

---

## Summary

Overall novelty: **HIGH** — the core contributions of the front-runner (cand_a) are not anticipated by any single existing paper as of April 2026. Multiple pieces of prior art from mid-2025 and early 2026 were found and assessed; none executes the unified program proposed here. Key risks remain precisely characterized below.

**New papers found since initial report:** SynthSAEBench (arXiv:2602.14687), Sparse but Wrong (arXiv:2508.16560), On the Limits of SAEs (arXiv:2506.15963), Taming Polysemanticity (arXiv:2506.14002), Feature Consistency Position Paper (arXiv:2505.20254), SAE Feature Manifolds (arXiv:2509.02565). These papers sharpen differentiation requirements but do not undermine core novelty.

---

## Candidate A: Rate-Distortion Theory + Cross-Domain + ASI + Phase Transition (Front-Runner)

### Core claims and search findings

**Claim 1: Formal proof that feature absorption is rate-distortion optimal (closed-form threshold lambda > sin^2(theta_{p,c}))**

*Prior art found:*
- Chanin et al. (2024) "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025 Oral) — the canonical paper on feature absorption. Provides informal explanation: absorption "saves one L0 per parent-child pair." Does NOT formalize a closed-form threshold or prove optimality.
- Tilde Research Blog "The Rate Distortion Dance of Sparse Autoencoders" — discusses rate-distortion framing of SAEs, including the L1/L0 Lagrangian and sparse coding theory. Addresses shrinkage and load-balancing problems. Does NOT derive an absorption threshold as a function of decoder angle. Does NOT address absorption specifically.
- Tang et al. (2025) "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534, v3 January 2026) — proves that spurious partial minima exhibiting feature absorption exist in the piecewise biconvex optimization landscape. Establishes necessary and sufficient conditions for correct feature recovery globally. Does NOT derive a closed-form threshold as a function of decoder angle and sparsity penalty for predicting which specific feature pairs are at risk. Different question: existence of spurious minima vs. prediction of per-pair absorption risk.
- Chanin, Dulka, Garriga-Alonso (2025) "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders" (arXiv:2505.11756) — analyzes feature hedging using a parameterized family of encoder/decoder solutions. Shows that absorption is a special case of feature hedging when correlation is hierarchical. Does NOT provide the rate-distortion absorption threshold.
- Chanin, Garriga-Alonso (2025) "Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders" (arXiv:2508.16560) — shows that too-low L0 causes SAEs to mix correlated features (a manifestation of hedging/absorption). Proposes a method to detect the correct L0 via decoder projection magnitude. This is a related L0-confound paper (directly cited in Phase C.4 of our proposal as the confound control). Does NOT provide a formal rate-distortion threshold.
- Cui et al. (2026) "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy" (arXiv:2506.15963, ICLR 2026) — provides a closed-form solution revealing that SAEs fail to fully recover ground truth monosemantic features unless features are extremely sparse. Proposes a reweighting strategy. This is a feature recovery identifiability theorem, NOT an absorption threshold. Does not derive the lambda > sin^2(theta) condition.
- Chen et al. (2025) "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders" (arXiv:2506.14002) — proposes a novel Group Bias Adaptation training algorithm with theoretical guarantees for feature recovery. Focuses on encoder bias adaptation, not on absorption threshold or rate-distortion analysis.

*Verdict:* **Partial overlap only.** The informal observation that "absorption saves L0" is established. Multiple 2025–2026 theoretical works provide formal analyses of SAE optimization but none derives the specific quantitative threshold `lambda > sin^2(theta_{p,c})` for predicting per-pair absorption risk. Critically, none predicts that co-occurrence frequency cancels from the threshold. **Novelty: HIGH for the formal derivation (8/10).**

*Differentiation:* Our contribution is a falsifiable, quantitative threshold (closed-form function of measurable quantities: lambda, decoder angle), plus the key counterintuitive prediction that co-occurrence frequency cancels. The existing theoretical works (2512.05534, 2506.15963, 2506.14002) address global identifiability and existence of pathological solutions — a different theoretical question. We provide a per-pair prediction tool.

---

**Claim 2: First cross-domain characterization of absorption across multiple semantic hierarchy types**

*Prior art found:*
- Chanin et al. (2024) — measures absorption exclusively on the first-letter spelling task (syntactic hierarchy). The paper itself notes this limitation.
- SAEBench (Karvonen et al., 2025, arXiv:2503.09532, ICML 2025) — implements absorption metric using first-letter features only. Documentation explicitly states: "SAEBench evaluates feature absorption by using features for 'word starts with X', which is not useful for evaluating domain-specific feature absorption."
- Matryoshka SAE paper (Bussmann et al., 2025, arXiv:2503.17547) — evaluates absorption on the first-letter task and a synthetic tree model, but only on these two task types.
- SynthSAEBench (Chanin, Garriga-Alonso, 2026, arXiv:2602.14687, Feb 2026) — a large-scale synthetic benchmark for evaluating SAEs on hierarchical features. IMPORTANT: This paper evaluates absorption on synthetic hierarchical data, not on semantic hierarchy types (entity-type, geographic, grammatical) in real LLM contexts. The gap we address (systematic real-LLM absorption measurement across semantic hierarchy types with rigorous null controls) remains open.
- RAVEL dataset (Huang et al., 2024) — provides entity-attribute data including city→country hierarchies, integrated into SAEBench for disentanglement tasks. Has NOT been used specifically for absorption measurement.
- OrtSAE (Korznikov et al., 2025, arXiv:2509.22033) — reduces absorption but does not measure cross-domain absorption rates.

*Verdict:* **No collision on the specific contribution.** SynthSAEBench addresses synthetic hierarchies; our contribution addresses real semantic hierarchies in real LLMs (Gemma Scope), with rigorous shuffled-label null controls, bootstrap CIs, and Bonferroni correction. The gap remains explicitly acknowledged in SAEBench documentation. **Novelty: VERY HIGH (9/10).**

*Note:* SynthSAEBench should now be cited in related work as a complementary approach (synthetic benchmarking) vs. our real-LLM cross-domain characterization.

---

**Claim 3: Probe-free Absorption Susceptibility Index (ASI) from decoder geometry**

*Prior art found:*
- Chanin et al. (2024) — absorption metric requires specifying probe directions in advance.
- Bricken et al. (2023/2025) — uses decoder cosine similarity to classify latent types (threshold at 0.7 for GPT-2, 0.4 for GemmaScope). ROC analysis for predicting latent behavior. Pairwise ASI differs: (1) pairwise metric for feature pairs, not per-feature; (2) includes frequency ratio term freq_p/freq_c; (3) framed as absorption susceptibility prediction for unknown features; (4) validated against Chanin et al. absorption labels.
- Tian et al. (2025) "Measuring Sparse Autoencoder Feature Sensitivity" (arXiv:2509.23717, NeurIPS 2025 Workshop Spotlight) — measures feature sensitivity (reliability of activation on similar texts via GPT-4.1). Does not use decoder geometry. Different concept.
- Song et al. (2025) "Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs" (arXiv:2505.20254) — proposes PW-MCC metric for feature consistency across training runs. Related but different: measures training stability, not absorption susceptibility for specific feature pairs.
- SAE Feature Manifolds (Michaud et al., 2025, arXiv:2509.02565, NeurIPS 2025 MI Workshop) — studies how feature manifolds (multi-dimensional features) cause pathological decoder direction clustering (high cosine similarity between neighboring latents). Related geometric analysis but does not frame as absorption susceptibility predictor.

*Verdict:* **Partial overlap with Bricken et al. decoder cosine similarity work.** The ASI formula `cos^2(theta) * (freq_p / freq_c)` adds the frequency ratio term, uses pairwise framing, and validates against absorption labels. No probe-free absorption susceptibility predictor for unknown feature pairs exists. **Novelty: HIGH (7/10), with clear differentiation possible.**

*Differentiation notes:* (1) ASI is pairwise (parent-child), not per-feature. (2) Frequency ratio captures absorption asymmetry absent from existing metrics. (3) Validated against Chanin absorption labels (AUROC). (4) Training-free and probe-free for detection of previously unknown absorptions — no prior metric achieves this.

---

**Claim 4: Phase transition and hysteresis characterization of absorption onset**

*Prior art found:*
- No paper found on "SAE absorption phase transition hysteresis" or on whether the absorbed state is metastable.
- Chanin et al. (2024) shows absorption rate vs. L0 plots qualitatively (Figure 8b) showing nonlinear dependence on L0, but does not frame this as a phase transition, does not fit functional forms, does not test for hysteresis.
- SAEBench (2503.09532) notes that JumpReLU/TopK SAEs significantly worsen feature absorption at lower L0, but this is an architectural observation, not a phase transition analysis.
- Sparse but Wrong (2508.16560) identifies an optimal L0 and discusses transition-like behavior, but does not apply phase transition frameworks, likelihood ratio tests between sigmoid and linear models, or hysteresis tests.
- SAE Feature Manifolds (2509.02565) discusses distinct scaling regimes but in the context of SAE scaling with dictionary size, not absorption onset as a function of sparsity penalty.

*Verdict:* **No collision.** Phase transition framing with functional form fitting (sigmoid vs. linear likelihood ratio test) and hysteresis test are entirely novel. **Novelty: VERY HIGH (9/10).**

---

### Cand A Overall Assessment

| Sub-contribution | Novelty Score | Closest Prior Art | Severity |
|---|---|---|---|
| Rate-distortion formal proof + closed-form threshold | 8/10 | Tang et al. 2512.05534, Tilde Blog | partial_overlap |
| Cross-domain absorption characterization (real LLMs) | 9/10 | SynthSAEBench (synthetic only), SAEBench | minor (synthetic vs. real gap) |
| Probe-free ASI | 7/10 | Bricken et al. decoder cosine similarity | partial_overlap |
| Phase transition + hysteresis | 9/10 | None | none |

**Cand A Novelty Score: 8/10**  
**Recommendation: PROCEED**

Key differentiation required:
- Position clearly against 2512.05534: existence of spurious minima vs. quantitative per-pair prediction threshold
- Note that SynthSAEBench covers synthetic hierarchies; our contribution is first real-LLM cross-domain measurement with rigorous null controls
- Distinguish ASI from Bricken et al. cosine similarity classifier (pairwise, with frequency ratio, validated against absorption labels)
- Cite Tilde Research blog as motivating informal observation; position formal derivation as the extension

---

## Candidate B: Absorption-as-Representational-Diagnostic

*Core claim:* The absorption graph encodes the model's internal feature hierarchy; cross-layer absorption profiles reveal computational structure.

*Prior art:*
- Chanin et al. (2024) studies absorption systematically but does not construct or analyze the absorption graph structure.
- Song et al. (2025) arXiv:2505.20254 — shows features are only ~0.80 consistent across SAE runs (PW-MCC metric). **This is a significant risk for the representational diagnostic framing: if the absorption graph varies by ~20% across seeds, claims about it encoding "model structure" are weakened.**
- No paper frames absorption as a representational diagnostic or constructs an absorption graph for structural analysis.

**Cand B Novelty Score: 7/10 — Proceed only as fallback.**

*Risk:* Feature consistency at 0.80 (Song et al.) means ~20% of features will differ across runs. The absorption graph derived from different SAE seeds will differ substantially, undermining the "canonical model structure" claim. Must be addressed explicitly or restricted to seed-averaged absorption graphs.

---

## Candidate C: Systematic Mitigation Benchmark

*Core claim:* Controlled head-to-head comparison of Matryoshka, OrtSAE, ATM, masked regularization on same model/layer/metric.

*Prior art found:*
- SAEBench (2503.09532, ICML 2025) — already compares Matryoshka and OrtSAE on absorption and other metrics systematically.
- Matryoshka SAE paper (2503.17547) — compares vs. BatchTopK, not vs. OrtSAE or ATM.
- OrtSAE paper (2509.22033) — compares vs. Matryoshka but not ATM.
- SynthSAEBench (2602.14687) — compares architectures on synthetic hierarchical data. Does not compare ATM or masked regularization.
- No single paper compares all four methods (Matryoshka, OrtSAE, ATM, masked regularization) specifically on the absorption+downstream metrics trade-off.

**Cand C Novelty Score: 5/10 — Proceed only as supplement to cand_a.**

*Risk:* SAEBench and SynthSAEBench already provide partial comparison infrastructure. Without theoretical framing from cand_a, this is below top-venue bar. SynthSAEBench (Feb 2026) further reduces the incremental value of a pure benchmark paper.

---

## Candidate D: Feature Anchoring for Real LLM Absorption Reduction

*Core claim:* First empirical evaluation of feature anchoring on real LLMs for absorption reduction.

*Prior art:*
- Tang et al. arXiv:2512.05534 proposes feature anchoring and validates "its effectiveness with extensive experiments across diverse SDL methods and settings" including real neural representations.

**COLLISION CONFIRMED — EXACT MATCH:** Tang et al. (2512.05534) already introduces and evaluates feature anchoring on real LLMs. The v3 (January 2026) update expands these experiments. The proposal for cand_d claiming "currently only validated on synthetic benchmarks" is factually incorrect.

**Cand D Novelty Score: 3/10 — DROP.**

*Verdict:* Exact match on core contribution. The only residual novelty would be testing anchoring specifically against the Chanin absorption metric — too narrow for a standalone contribution.

---

## New Papers Found Since Initial Report — Impact Assessment

| Paper | Key Finding | Impact on Proposal |
|---|---|---|
| SynthSAEBench (arXiv:2602.14687, Chanin, Feb 2026) | Synthetic benchmark for hierarchical SAE absorption evaluation | Must cite as complementary; real-LLM cross-domain gap remains open |
| Sparse but Wrong (arXiv:2508.16560, Chanin, Aug 2025) | Correct L0 diagnosis via decoder projection | Validates our L0 deconfounding control (Phase C.4); must cite |
| On the Limits of SAEs (arXiv:2506.15963, ICLR 2026) | Closed-form SAE identifiability limits theorem | Related theoretical work; must distinguish from our absorption threshold |
| Taming Polysemanticity (arXiv:2506.14002, Jun 2025) | Provable feature recovery via bias adaptation | Related provable method; not specific to absorption, must cite |
| Feature Consistency (arXiv:2505.20254, May 2025) | SAE feature consistency across runs ~0.80 | Risk for cand_b; also supports canonical Gemma Scope choice for cand_a |
| SAE Feature Manifolds (arXiv:2509.02565, Sep 2025) | Feature manifolds cause decoder clustering (high cosine similarity) | Related geometric analysis; partially supports ASI intuition (high cosine sim between absorber/absorbee); must cite |

**None of these papers undermine cand_a's core novelty claims.** They sharpen the literature map and add required citations.

---

## Key Prior Work Citation Summary

| Paper | Relevance to Proposal | Severity | Must Cite |
|---|---|---|---|
| Chanin et al. (2024) arXiv:2409.14507 | Foundation: defines absorption, first-letter task, absorption metric | related_work | YES |
| Tang et al. (2025) arXiv:2512.05534 | Partial: piecewise biconvex theory; existence of spurious minima; feature anchoring | partial_overlap | YES |
| Tilde Research Blog (2024) | Partial: informal rate-distortion framing of SAEs | partial_overlap | YES |
| Bussmann et al. (2025) arXiv:2503.17547 | Related: Matryoshka SAEs reduce absorption | related_work | YES |
| Korznikov et al. (2025) arXiv:2509.22033 | Related: OrtSAE reduces absorption via decoder orthogonality | related_work | YES |
| Karvonen et al. (2025) arXiv:2503.09532 | Related: SAEBench (explicitly acknowledges cross-domain gap) | related_work | YES |
| Chanin, Dulka et al. (2025) arXiv:2505.11756 | Related: Feature hedging — different failure mode, same domain | related_work | YES |
| Chanin, Garriga-Alonso (2025) arXiv:2508.16560 | Related: Sparse but Wrong — correct L0 diagnosis | related_work | YES |
| Tian et al. (2025) arXiv:2509.23717 | Related: Feature sensitivity (different concept from absorption susceptibility) | related_work | NO |
| Bricken et al. (2023/2025) | Partial overlap: decoder cosine similarity to predict latent behavior | partial_overlap | YES |
| ATM (arXiv:2510.08855) | Related: adaptive temporal masking reduces absorption 40% | related_work | YES |
| MDL-SAEs Ayonrinde et al. (2024) | Related: compression/MDL language for SAEs | related_work | NO |
| Chanin, Garriga-Alonso (2026) arXiv:2602.14687 | Related: SynthSAEBench — synthetic hierarchical benchmark | related_work | YES |
| Cui et al. (2026) arXiv:2506.15963 | Related: SAE theoretical limits — identifiability theorem | related_work | YES |
| Chen et al. (2025) arXiv:2506.14002 | Related: Provable feature recovery via bias adaptation | related_work | NO |
| Song et al. (2025) arXiv:2505.20254 | Related: Feature consistency across SAE runs | related_work | YES |
| Michaud et al. (2025) arXiv:2509.02565 | Related: SAE scaling with feature manifolds — geometric clustering | related_work | YES |

---

## Recommendations by Candidate

| Candidate | Recommendation | Rationale |
|---|---|---|
| cand_a | PROCEED — front-runner | Four distinct novel contributions; all partial overlaps are defensible. New 2025–2026 papers sharpen differentiation but do not undermine novelty. Overall novelty 8/10. |
| cand_b | MODIFY — backup only | Conceptually novel framing but SAE feature consistency risk (~0.80 PW-MCC) weakens the "canonical representation" claim. Proceed as fallback if cand_a theory fails. |
| cand_c | SUPPLEMENT — not standalone | SynthSAEBench (Feb 2026) further reduces incremental value. Only viable as empirical supplement with theoretical framing from cand_a. |
| cand_d | DROP | Core contribution (feature anchoring on real LLMs) already published in arXiv:2512.05534 v3 (Jan 2026). |

---

## Anti-Patterns Avoided

- The rate-distortion framing is NOT the same as the Tilde Research blog: the blog is informal, does not address absorption specifically, and does not derive a geometric threshold.
- The ASI is NOT the same as Bricken et al. decoder cosine similarity: different metric (pairwise, with frequency ratio), different application (absorption susceptibility for unknown feature pairs), different validation (AUROC against Chanin absorption labels).
- The cross-domain contribution is NOT the same as SynthSAEBench: SynthSAEBench uses synthetic data with controlled hierarchies; we measure real-LLM absorption on semantic hierarchy types (entity-type, geographic, grammatical) with rigorous null controls.
- The phase transition analysis is NOT the same as L0 sweep plots in prior papers: we apply functional form fitting (sigmoid vs. linear likelihood ratio test) and test for hysteresis — framing and quantitative methodology are new.

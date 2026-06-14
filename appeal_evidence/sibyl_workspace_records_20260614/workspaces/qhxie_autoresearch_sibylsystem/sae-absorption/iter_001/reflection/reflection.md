# Reflection Report — Iteration 1

**Date**: 2026-04-13
**Score**: 5.5 / 10 (Supervisor Review)
**Verdict**: CONTINUE (scope restructuring + experimental expansion required)
**Stage**: Post-R4 (after shuffled control falsification of H3)

---

## Iteration Summary

Iteration 1 completed a full research pipeline from idea through writing and peer-simulated review for the paper "Where and When Encoder-Decoder Misalignment Signals Feature Absorption in Sparse Autoencoders." The iteration produced 22 completed experimental tasks (plus 1 failed/skipped: r4d_itac_real_activations), covering: EDA metric validation across 8 SAE configurations, a three-subtype absorption taxonomy, cross-domain RAVEL analysis with shuffled control (R4B), scaling analysis, GPT-2 replication with exact Chanin et al. labels (R3), four R4 remediation rounds, and a full paper draft with critic review.

The final score of 5.5/10 reflects genuine intellectual merit (correct problem identification, honest negative-result reporting, sound statistical methodology) that is blocked from a top-venue submission by four structural defects: an unproved theorem forming the novelty foundation, experimental results that contradict stated claims (EDA ordering), a near-null ITAC result still partially present in the abstract, and no exact-label validation for the primary model family (Gemma).

**What improved in this iteration versus iter_0**: The R4 round added substantial value — GPT-2 exact-label validation (AUROC=0.629, n_pos=67), shuffled hierarchy control correctly falsifying H3 (0/9 pass), and honest paper restructuring around these findings. Negative result reporting improved significantly. The paper is now more credible even if weaker in scope.

**What did not improve**: Score remains at 5.5. The core structural defects (Theorem 1 proof, EDA ordering factual error, Gemma access constraint) were not resolved. Two of the three highest-priority issues from iter_0 remain present in the paper.

---

## 1. Issue Analysis by Category

### NOVELTY (Critical)

**EDA = SAEBench encoder_decoder_cosine_sim (RECURRING from iter_0)**

EDA (1 - cosine(encoder_j, decoder_j)) is mathematically identical to the SAEBench `encoder_decoder_cosine_sim` metric (Pearson r > 0.999; delta AUROC = 0.0 in R4 direct validation). The Introduction still presents EDA as a new metric without disclosing the SAEBench precedent. The entire novelty claim now rests on Theorem 1 — which is itself a proof sketch.

Source evidence: `critic/findings.json` finding #2; `supervisor/review.json` issues[0]; `r4a_eda_direct_validation.json` (delta=0.0).

**Status**: RECURRING (was iter_0 issue_06; must-fix priority)

---

### SOUNDNESS (Critical + Major)

**Theorem 1 proof is circular (RECURRING from iter_0)**

The proof sketch in Section 3.2 asserts its conclusion: "absorption introduces a perturbation proportional to delta * ||d_c|| * sin(theta_{jc})" without deriving this from SDL loss stationarity conditions. Delta-absorption is not formally defined before the theorem. The bound formula delta^2*sin^2(theta_{jc})/(2+delta^2) appears without algebraic derivation. Appendix B is referenced twice in the main text but absent from paper.md.

Source evidence: `supervisor/review.json` issues[1]; `critic/findings.json` finding #3; `writing/review.md` issue 2.

**Status**: RECURRING (was iter_0 issue_05; must-fix priority)

**D-EDA/EDA equivalence tension (NEW)**

R4 direct validation confirms EDA = 1 - dec_cos as a mathematical identity (delta AUROC = 0.0). However, R3 Phase 5 shows D-EDA AUROC = 0.762 at GPT2-L10 vs EDA AUROC = 0.336. The paper does not reconcile this inconsistency. Either R4 did not correctly implement the LASSO D-EDA decomposition, or R3's D-EDA was using a different calculation than described in Section 3.3.

Source evidence: `supervisor/review.json` issues[6]; `phase5_gpt2_replication.json`.

**Status**: NEW

---

### EXPERIMENT (Multiple Critical + Major)

**EDA ordering claim factually contradicts source data (RECURRING from iter_0 — was BLOCKING)**

Section 6.2 claims "EDA ordering (late > partial > early) holds at L12-65k (KW p=0.0002)." Cross-validating against phase2a_taxonomy.json: `eda_ordering_holds=FALSE` for both L12-16k and L12-65k at threshold 0.3. At L12-65k: partial median EDA = 0.652 < early median EDA = 0.668. The KW p=0.0002 establishes late > early only; the full three-way ordering fails. Table 3 describes early absorption as having "Low (decoder-absent)" EDA — directly contradicted by the data.

This was listed as "BLOCKING: Correct before any further writing" in iter_0 (issue_02). It remains uncorrected in the paper.

Source evidence: `phase2a_taxonomy.json` eda_ordering_holds=false; `supervisor/review.json` issues[2]; `critic/findings.json` finding #6.

**Status**: RECURRING (critical, was blocking — highest urgency)

**Taxonomy rests on two configurations, one underpowered (RECURRING from iter_0)**

The "early absorption dominance" finding (72-75% early-type) rests on n=16 (L12-16k, KW p=0.237 — non-significant) and n=65 (L12-65k). The finding is highly threshold-sensitive: at tau=0.2 early proportion drops to 32%; at tau=0.3 it is 72%; at tau=0.4 it is 95%. Expansion to all 6 Gemma Scope + GPT-2 configurations was HIGH priority in iter_0 but was not completed. R4C addressed Llama weight-only EDA only, not taxonomy expansion.

Source evidence: `critic/findings.json` finding #1; `phase2a_taxonomy.json`; `r4_writing_gate.json`.

**Status**: RECURRING (was iter_0 issue_10; HIGH priority)

**ITAC near-null result partially fixed but abstract still misleading (PARTIALLY FIXED)**

Progress since iter_0: ITAC is now in Section 7.3 negative results. The paper no longer leads with ITAC as a primary contribution. Remaining issues: (a) abstract still contains ITAC detail; (b) Table 4 "NULL test" row is misleading (ITAC was never applied to early-type latents; the 0.00% FN reduction is tautological — no correction was attempted); (c) the r4d_itac_real_activations task failed and was skipped, so ITAC remains evaluated only on synthetic activations.

Source evidence: `critic/findings.json` finding #8; `supervisor/review.json` issues[3]; `phase2b_itac.json`.

**Status**: PARTIALLY FIXED (was iter_0 issue_04)

**EDA go/no-go gate failure not explicitly disclosed (PARTIALLY FIXED)**

EDA meets AUROC >= 0.65 at 2/6 Gemma configurations — a NO_GO by the paper's own pre-registered criterion. The paper uses "regime-specific" framing, which is an improvement, but does not explicitly state the gate outcome: "2/6 configurations — a NO_GO under our pre-registered criterion."

Source evidence: `supervisor/review.json` issues[4]; `phase1_eda_deda_validation.json`.

**Status**: PARTIALLY FIXED (was iter_0 issue_03)

**AUPRC not reported for class-imbalanced settings (RECURRING from iter_0)**

At L12-65k with 0.024% positive prevalence, AUROC is unreliable. AUPRC values exist in the data but are not in Table 1 or discussed in main text. Was listed as HIGH priority in iter_0 (issue_07). Not addressed in iter_1.

Source evidence: `phase1_eda_deda_validation.json` (AUPRC = 0.0024 at L12-16k); `supervisor/review.json` issues[8].

**Status**: RECURRING (was iter_0 issue_07; must add in iter_2)

**H6 scaling analysis partial correlation methodologically invalid (RECURRING from iter_0)**

Canonical L0 values used (range 65-72, no variation within layer-matched width comparisons). The partial rho(width | L0) cannot control for L0. "H6 falsified via partial correlation" language is methodologically unsound. Was flagged in iter_0 issue_08, remains unchanged.

Source evidence: `supervisor/review.json` issues[7]; `phase4_scaling_analysis.json`.

**Status**: RECURRING (was iter_0 issue_08)

**Polysemanticity AUROC CI unreliable at n_pos=3 (NEW)**

The polysemanticity stratification reports AUROC = 0.922 [0.842, 0.979] from n_pos=3 positive examples. Bootstrap CIs with n_pos=3 are numerically unreliable and present false precision.

Source evidence: `ablation_polysemanticity.json`; `critic/findings.json` finding #10.

**Status**: NEW

**Pilot-to-full AUROC collapse root cause understated (PARTIALLY FIXED)**

The paper explains the L12-65k collapse (pilot 0.853, full 0.468) as "enriched positive prevalence" and "proxy label noise" but understates the fundamental cause: n_pos=16/65536 = 0.024% prevalence makes AUROC unreliable regardless of label quality.

Source evidence: `supervisor/review.json` issues[9].

**Status**: PARTIALLY FIXED

---

### WRITING (Major + Minor)

**Abstract DeLong comparison is ambiguous (NEW factual error)**

The abstract states "outperforming the decoder cosine baseline by +0.396 AUROC" immediately after "AUROC = 0.776 on Gemma Scope L12-16k." The +0.396 is from L5-16k (EDA=0.698, decoder cosine=0.302); the L12-16k difference is +0.553 (EDA=0.776, decoder cosine=0.224). The sentence creates a misleading juxtaposition.

Fix: "EDA achieves AUROC = 0.776 on Gemma Scope L12-16k [0.700, 0.863], outperforming the decoder cosine baseline by +0.553 AUROC at this configuration (DeLong p ≈ 0), and AUROC = 0.629 on GPT-2 Small L6 [0.561, 0.692] with exact Chanin et al. labels."

Source evidence: `critic/critique_writing.md` Error 2; `writing/review.md` Claim-Evidence Integrity section.

**Status**: NEW (must-fix before submission)

**"Reframes the absorption problem" overclaims (RECURRING from iter_0)**

Sections 6.3 and Conclusion use "reframes the absorption problem" language. With n=2 SAE configurations (one non-significant at p=0.237) and high threshold-sensitivity, this claim is not defensible. iter_0 lessons explicitly flagged this.

Source evidence: `critic/critique_writing.md` issue #5.

**Status**: RECURRING (from iter_0 lessons)

**Section 7.1 "56% / 20x" figures unsourced (NEW confirmed)**

"Focusing on the top 5% of latents by EDA contains approximately 56% of absorbed latents, reducing supervised evaluation budget by 20x." Cross-checking: 5% of 16,384 = 819 candidates; 819 × Prec@50 (0.0035) ≈ 2.87 absorbed; 2.87/16 positives ≈ 18%, not 56%. These figures are arithmetically inconsistent with the source data.

Source evidence: `supervisor/review.json` issues[11]; `writing/review.md` Issue 5.

**Status**: NEW (confirmed unsourced)

**Figure 1 missing as rendered PDF (NEW)**

Figure 1 (the core method diagram illustrating absorption mechanism and EDA geometry) is referenced in Introduction and Section 3.1 but only exists as `writing/figures/fig1_absorption_mechanism_desc.md` (text descriptor, no PDF). The writing review confirms all other figures (2-7) are present as PDFs.

Source evidence: `writing/review.md` Issue 1.

**Status**: NEW (blocking for submission)

**Appendices B and C absent from paper.md (PARTIALLY FIXED)**

Appendix B (D-EDA conditioning analysis, referenced in Sections 3.3 and 7.1) and Appendix C (RAVEL probe details, referenced in outline) are absent from paper.md.

Source evidence: `writing/review.md` Issue 2; `critic/critique_writing.md` issue #4.

**Status**: PARTIALLY FIXED (Figs 5/6/7 resolved from iter_0; appendices remain absent)

**Missing citations (RECURRING from iter_0)**

DeLong test first used in Section 4.1 without citation. SynthSAEBench citation ('synthasbench2026') unverified. Both were in iter_0 issue_09 and remain unresolved.

Source evidence: `supervisor/review.json` issues[9]; `critic/critique_writing.md` issue #10.

**Status**: RECURRING (from iter_0 issue_09)

**Minor notation inconsistency (NEW)**

Spearman rho notation: $\rho$ used throughout but notation.md specifies $\rho_s$. Affects abstract, Sections 5.3, 7.2. ITAC result "3.14% mean false-negative reduction" is ambiguous (relative vs. absolute: actual is 1.4 percentage point absolute reduction).

Source evidence: `critic/findings.json` finding #15; `writing/review.md` Notation section.

**Status**: NEW (minor)

---

### REPRODUCIBILITY (Major)

**Gemma 2B HF-gated; all Gemma AUROC values on proxy labels (RECURRING)**

The R4 access check (r4a_gemma_access_check) confirmed Gemma 2 2B remains gated. All 6 Gemma Scope AUROC values use Neuronpedia proxy labels. The only exact-label validation is GPT-2 Small L6. This is the single highest-leverage unresolved issue — if 3+/6 Gemma configurations pass AUROC >= 0.65 with exact labels, the score rises to 7.0.

Source evidence: `r4a_access_status.json`; `supervisor/review.json` risks[0].

**Status**: RECURRING (structural blocker from iter_0 issue_01)

---

## 2. Fix Tracking Summary

| Issue | Origin | Status |
|---|---|---|
| RAVEL probe model mismatch (Qwen proxy) | iter_0 issue_01 | PARTIALLY FIXED (H3 falsified via shuffled control; root cause acknowledged) |
| EDA ordering factual error | iter_0 issue_02 | RECURRING (still present in Section 6.2 — was blocking) |
| EDA regime-specific framing | iter_0 issue_03 | PARTIALLY FIXED |
| ITAC negative result reframing | iter_0 issue_04 | PARTIALLY FIXED (section moved; abstract still has ITAC detail) |
| Theorem 1 proof | iter_0 issue_05 | RECURRING (unchanged) |
| EDA = SAEBench metric disclosure | iter_0 issue_06 | RECURRING (Introduction unchanged) |
| AUPRC as co-primary metric | iter_0 issue_07 | RECURRING (not added) |
| H6 partial correlation invalid | iter_0 issue_08 | RECURRING (unchanged) |
| Missing Figs 5/6/7; DeLong citation | iter_0 issue_09 | PARTIALLY FIXED (Figs exist; citations still missing) |
| Taxonomy expansion to 6+ configs | iter_0 issue_10 | RECURRING (only 2 configs) |
| ITAC null test is tautological | iter_0 issue_11 | RECURRING |
| Intra-RAVEL Bonferroni correction | iter_0 issue_12 | PARTIALLY FIXED (H3 framing improved) |
| SynthSAEBench tautology | iter_0 issue_13 | PARTIALLY FIXED |
| Abstract word count | iter_0 issue_14 | PARTIALLY FIXED |
| Delta-absorption definition circular | iter_0 issue_15 | RECURRING |

**Issues fixed from iter_0**: Figs 5/6/7 confirmed as PDFs; H3 properly falsified with shuffled control; GPT-2 exact-label anchor added; paper restructured around honest negative results; "proxy label" caveats added throughout; early ITAC demotion to Section 7.3.

**Issues recurring**: EDA ordering factual error (was BLOCKING), Theorem 1 proof, EDA/SAEBench novelty disclosure, AUPRC metric, H6 partial correlation, DeLong citation, SynthSAEBench citation, taxonomy expansion.

---

## 3. Pattern Recognition

**Cross-stage recurring patterns:**
1. **Model access constraint** propagates throughout: Gemma gating blocked Phase 1 exact labels, Phase 3 RAVEL probes, and limits reproducibility. This is a structural bottleneck that no amount of experimental iteration can resolve without model access or model substitution.
2. **Factual corrections not applied before writing**: iter_0 action plan listed the EDA ordering correction as a blocking requirement before writing. The writing phase proceeded without it. This suggests the writing agent does not read the action plan's blocking flags before starting.
3. **Proof/theorem handling**: Theorem 1 has been flagged in every review pass (iter_0 and iter_1). Without a formal proof or explicit downgrade, this will continue to block novelty credibility.
4. **Scope remains too wide**: Five contributions (EDA, D-EDA, cross-domain, taxonomy, ITAC) despite recommendation to reduce to three. This dilutes each contribution's evidence base.

**Quality trend**: Stagnant at 5.5/10. Iter_0 baseline was 5.5; iter_1 is also 5.5. The genuine improvements (H3 falsification, GPT-2 anchor, honest reporting) balanced against unresolved core defects produce the same score.

---

## 4. Resource Efficiency Analysis

**GPU utilization**: 22/23 tasks completed (1 failed: r4d_itac_real_activations, deprioritized as optional). Single GPU (GPU 5). Approximately 7 hours total GPU compute across the full iteration. Near-continuous utilization during active experiment phases. `gpu_progress.json` shows `running: {}` at end — no running tasks were stranded.

**Parallelization opportunities missed**:
- Phase 3 city subtasks (city_continent, city_country, city_language) ran sequentially — all mutually independent, could have run in parallel (3x wallclock speedup for these tasks).
- Phase 5 GPT-2 replication could have been parallelized with Phase 3.
- R4 subtasks: r4a (access check + direct labels) and r4b (RAVEL shuffled control) could have run in parallel with r4c (Llama replication).

**Bottleneck stages**:
1. `phase3_probe_quality` + Phase 3 subtasks: Blocked by model access (Gemma gated). No matter how efficiently these tasks ran, the probe quality was fundamentally limited.
2. Writing revision: Multiple mandatory factual corrections (EDA ordering error) should have been completed as a pre-writing gate, but were not enforced.

**iter_2 scheduling recommendations**:
- Allocate 2+ GPUs if available; run Phase 1 and Phase 5 in parallel; run Phase 3 subtasks in parallel.
- Make blocking corrections from action_plan.json a mandatory pre-writing gate (verified before writing agent starts).
- The r4d_itac_real_activations task (real activation ITAC evaluation) should be explicitly scheduled and not deprioritized, since its absence is a reviewable gap.

**GPU idle time**: ~15 minutes total estimated (between task dispatch). Negligible.

---

## 5. Success Patterns

1. **Honest negative result reporting**: H3 falsification (0/9 domain-SAE combinations pass), ITAC near-null (3.14% vs 20% target), EDA go/no-go gate failure (2/6) — all reported with specific numbers and pre-registered targets. Section 7.3 is the strongest part of the paper and should remain a model for future iterations.

2. **GPT-2 as open-model anchor**: The Phase 5 GPT-2 exact-label validation (AUROC=0.629, n_pos=67) provides the cleanest, most reproducible evidence in the paper. This strategy should be expanded in iter_2 to GPT-2 L10 and additional configurations for taxonomy analysis.

3. **R4 adaptive remediation round was effective**: The R4 round identified and plugged the most critical evidence gaps (direct-label validation, shuffled control). This should be formalized as a standard pre-writing remediation round in the pipeline.

4. **Statistical methodology**: Bootstrap CIs (10k resamples), DeLong test, Cohen's d, Mann-Whitney U, Kruskal-Wallis, shuffled null column in Table 1 — appropriate and exemplary. The shuffled null column is a model of methodological rigor.

5. **Writing quality baseline maintained**: No banned patterns found. Specific quantitative leads in all sections. Abstract first sentence is clear, stakes-specific, and non-formulaic.

6. **Result debate mechanism effective**: Multi-perspective (optimist/skeptic/strategist/synthesizer) debate prevented over-optimistic bias and generated specific correction paths. Should continue in all future iterations.

---

## 6. System Self-Check Response

No `logs/self_check_diagnostics.json` was found in the workspace. The `logs/` directory contains only an empty `iterations/` subdirectory. No system self-check diagnostics require response in this iteration.

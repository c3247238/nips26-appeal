# Critique: Experimental Design and Results (Post-R4 Update)

**Updated:** 2026-04-13 | **Round:** Post-R4 (final experimental state)

## Summary

Post-R4 experiments (R4A-R4C) have confirmed the GPT-2 direct-label EDA result, properly executed the shuffled hierarchy control for H3 (confirming H3's falsification), and obtained Llama Scope weight-only EDA statistics. The overall experimental picture is clearer and more honestly reported than R3. Key remaining problems: the taxonomy's central claim is threshold-dependent, the taxonomy analysis has not been run on GPT-2, and the Gemma 2B direct-label gap persists.

---

## R4 Experiment Assessment

### R4A: GPT-2 Direct Label Validation

**Status: Pass for GPT-2-L6; confirms existing R3 finding.**

- GPT-2-L6: AUROC=0.650 [0.531, 0.761] with Chanin et al. direct labels, n_pos=18
  - Note: R3 reported AUROC=0.629 [0.561, 0.692] with n_pos=67. R4A gives 0.650 with n_pos=18.
  - The discrepancy in n_pos (67 vs 18) suggests R4A and R3 used different absorption thresholds or word sampling parameters. The paper's main text cites AUROC=0.629 [0.561, 0.692] — this should be clarified which version is canonical.
- GPT-2-L10: AUROC=0.336 (consistent with R3; reversed direction confirmed)
- **Key finding**: EDA = 1 - dec_cosine identity confirmed (AUROC delta = 0.0 for both configs). This definitively establishes that EDA is the same as the SAEBench metric, not a new quantity.
- **Critical limitation**: R4A did NOT generate Gemma 2B direct labels. Gemma 2B remains gated. All Gemma Scope AUROC values still use proxy labels.

**Action needed:** Clarify which GPT-2-L6 result (R3: n_pos=67, AUROC=0.629 or R4A: n_pos=18, AUROC=0.650) is the canonical paper result and update Table 1 accordingly.

---

### R4B: RAVEL Probes on Correct Model + Shuffled Control

**Status: H3 definitively falsified. Correctly executed.**

- Bridge model fallback to GPT-2-Medium (Gemma 2B and Llama-3.1-8B both gated)
- Probe quality: city-continent 59.5% (gate: 85%), city-country 47.9%, city-language 53.3% — all fail
- Shuffled control: 0/9 domain-SAE combinations pass (real absorption rate ≤ shuffled p95 threshold)
- Ratios (real/shuffled): city-continent 0.98, city-country 2.09, city-language 0.83-1.14 — all within noise
- **This is correct methodology.** The shuffled control is the right null test for cross-domain absorption.

**Remaining issue:** The intra-RAVEL coherence reported in Section 5.3 (rho=0.924, n=6 configs from R3) is from a larger experiment than the R4B pilot (n=3 configs). The R4B pilot's coherence estimate would be different. The paper should disclose this sample-size discrepancy between the coherence claim and the shuffled control.

---

### R4C: Llama-3.1-8B Scope Weight-Only EDA

**Status: Weight-only analysis only; AUROC not computable.**

- Llama-3.1-8B model is gated; only SAE weights loaded
- EDA computed from Llama Scope weights (d_in=4096, d_sae=32768)
- EDA identity confirmed (r=-0.9999, as expected mathematically)
- No absorption labels available; AUROC not computed
- Llama mean EDA = 0.518 (mean z-score vs Gemma: +3.4-3.6)

**Assessment:** This extends cross-architecture EDA distribution characterization but provides no AUROC validation. The paper's Table 5 cross-model summary can include Llama only for distribution statistics, not absorption detection performance. The current paper (Table 1) does not include Llama results — this is appropriate.

---

## Pre-Existing Experimental Problems (Still Unresolved After R4)

### Problem 1: Taxonomy Threshold Sensitivity Is Underreported

The paper's central actionable finding — early absorption dominates at 72-75% — is entirely contingent on tau=0.3. The data shows:

| tau | L12-65k Early | L12-65k Late | L12-65k Partial |
|-----|--------------|-------------|----------------|
| 0.2 | 32.3% | 33.8% | 33.8% |
| 0.25| ~55-60% (interpolated) | — | — |
| 0.3 | 72.3% | 13.8% | 13.8% |
| 0.35| ~85-90% | — | — |
| 0.4 | 95%+ | ~3% | ~2% |

At tau=0.2, there is no early dominance — the three subtypes are nearly equally distributed. The paper acknowledges this in the r4_writing_gate.json limitations ("tau=0.3 baseline. tau=0.2 reduces early% to 32% at L12-65k — report threshold sensitivity prominently") but the main text Section 6.2 does not show the full table of tau-varying prevalence.

**Scientific consequence:** The claim "72-75% of absorbed latents are early-type" is not a robust empirical finding — it is a threshold-conditional statement. At tau=0.2, a researcher would conclude "early and late absorption are equally prevalent." The paper must either justify tau=0.3 scientifically (why is the 30th percentile of cosine similarity the right boundary?) or present the full sensitivity table and describe the finding as "at tau=0.3, early-type dominates."

---

### Problem 2: Taxonomy Not Validated on GPT-2 (Missed Opportunity)

The GPT-2-L6 configuration has exact Chanin et al. labels and n_pos=67 (the most abundant absorbed-latent set with exact labels in the paper). The taxonomy analysis (Phase 2) was run only on L12-16k and L12-65k. Running the taxonomy on GPT-2-L6 would:
- Provide a third, cross-model data point for the early-dominance claim
- Use the most methodologically clean dataset (exact labels, accessible model)
- Strengthen the generalizability claim substantially

This is a low-cost experiment (~2 GPU-hours, weight-only taxonomy analysis) that would significantly improve the paper's statistical standing. It is not too late to run before submission.

---

### Problem 3: EDA Statistical Power at Low Positive Prevalence

At L12-65k: n_pos=16 out of 65,536 total latents = 0.024% positive prevalence. AUROC at this prevalence is dominated by noise in the ranking of negative examples. The paper reports AUROC=0.468 [0.315, 0.620] — a 95% CI spanning from 0.315 to 0.620 encompasses both "EDA fails" and "EDA barely works." The AUPRC values are available in the JSON files but are not reported in the main text.

At L12-16k: n_pos=16 out of 16,384. This is 0.10% prevalence — still very low, but 4× the relative frequency of L12-65k.

**Standard practice:** For highly imbalanced classification problems (positive prevalence < 1%), AUPRC is a more informative metric than AUROC. The paper should include AUPRC in Table 1 and discuss why AUROC is being used as the primary metric despite the prevalence problem.

---

### Problem 4: GPT-2-L6 n_pos Discrepancy Between R3 and R4

R3 (Phase 5) reports n_pos=67 for GPT-2-L6 with AUROC=0.629 [0.561, 0.692]. R4A (r4a_direct_labels.json) reports n_pos=18 for GPT-2-L6 with AUROC=0.650 [0.531, 0.761]. The paper uses the R3 result (AUROC=0.629, n_pos=67) in Table 1. This is correct since R3 used more words per letter (up to 50 per letter, 7,637-word vocabulary). But the R4A run with n_pos=18 gives a different estimate. The paper should clarify which result is canonical and why they differ (likely: R4A used fewer words_per_letter and stricter ablation threshold).

---

### Problem 5: ITAC Null Test is Not an ITAC Null Test

Table 4 Row 3 ("Early/Partial — all | — | — | 0.00% | — | NULL") is described as confirming "0% FN reduction." But examining phase2b_itac.json: the null_test_early field reports baseline FN rates of early-type latents WITHOUT applying ITAC. The 0.00% reduction is because no correction was applied, not because ITAC was tried and found ineffective on early-type latents.

If the paper claims "ITAC has no effect on early-type latents, confirming taxonomy prediction," the ITAC correction must actually be applied to early-type latents to verify this. Since early-type latents by definition have max_cos(d_k, v_p) < tau (no parent decoder direction), the D-EDA decomposition should find zero parent candidates and ITAC should be inapplicable — but this needs to be empirically confirmed, not assumed.

---

### Problem 6: ITAC Mean Is Driven by Non-Applicable Targets

Of 10 ITAC targets at L12-65k (phase2b_itac.json):
- 4 targets: n_parent_candidates=0 (D-EDA finds no absorbing sources; ITAC cannot be applied)
- 5 targets: FN reduction = 0.00%
- 1 target (j_idx=61217): FN reduction = 18.9%

The reported mean of 3.14% averages a 18.9% improvement with nine zeros, producing a misleading average. The appropriate characterization is: "ITAC is applicable to 6/10 late-type targets; of these, 1/6 shows meaningful improvement (18.9%); overall mean FN reduction = 3.14%, driven by a single latent."

---

## Strongest Experimental Results (To Preserve and Emphasize)

1. **GPT-2-L6 EDA validation** (AUROC=0.629, n_pos=67, exact Chanin et al. labels): The cleanest, most reproducible result. This should be emphasized as the primary EDA validation since all Gemma results use proxy labels.

2. **H3 shuffled hierarchy control** (R4B): Correctly designed null test. 0/9 combinations pass. This is methodologically exemplary.

3. **Taxonomy KW p=0.0002 at L12-65k** (n=65 absorbed latents): The most statistically robust result in the taxonomy analysis. The late > early EDA ordering holds at all 5 tested thresholds.

4. **SynthSAEBench AUROC=1.000**: Confirms EDA computation is correct (though not that it detects Chanin et al. behavioral absorption independently).

5. **EDA outperforms decoder cosine baseline by +0.553 AUROC at L12-16k**: The decoder cosine baseline should have no information about absorption beyond dictionary geometry; EDA's advantage (+0.553) confirms encoder direction provides additional discriminative signal.

---

## Recommendations for Final Pre-Submission Experiments

| Priority | Experiment | Effort | Impact |
|----------|-----------|--------|--------|
| High | Run taxonomy on GPT-2-L6 (exact labels, n_pos=67) | 1-2 GPU-hours | Third data point for early-dominance claim |
| High | Apply ITAC to early-type latents as actual null test | <1 GPU-hour | Corrects the null test description |
| Medium | Generate Figure 5, 6 captions aligned with null result | Writing only | Consistency |
| Medium | Write Appendix B (D-EDA conditioning) | Writing only | Fixes missing appendix reference |
| Low | Add AUPRC to Table 1 | Analysis + writing | Better metric reporting |

# Supervisor Review — SAE Absorption Paper
## Independent Research Assessment (NeurIPS-calibrated)

**Score: 5.5 / 10 — Verdict: CONTINUE**

---

## Executive Summary

This paper introduces Encoder-Decoder Dissociation (EDA) as a probe-free, weight-only detector of absorbed SAE features and presents a rate-distortion theoretical framework for predicting when absorption is training-preferred. The core empirical finding is genuine and reproducible: EDA achieves AUROC = 0.650 against exact Chanin labels (n_pos=18, z_null=2.49) and the cross-directional metric cos(ê_p, d_c) achieves AUROC = 0.730 against proxy labels (n_pos=50–63, Cohen's d=0.552, p=2.8e-9) — both verified from source JSON files (D1_eda_validation.json, B1_pairwise_eda.json).

However, two fundamental science problems prevent acceptance in the current form, accompanied by several major issues that require targeted iteration:

1. **Theory falsification framing**: Proposition 1's primary geometric prediction — that absorbed pairs should have smaller decoder-decoder angles than non-absorbed pairs — is directly contradicted by B1_decoder_geometry.json (absorbed pos_mean_cos2=0.052 < non-absorbed neg_mean_cos2=0.127; AUROC=0.318). The paper's three-sentence "post-convergence equilibrium" reframing is asserted without derivation. The audit explicitly classifies this as "H1 FALSIFIED."

2. **Shortlisting claim vs. Precision@k=0**: D1_eda_validation.json shows Precision@50 = Precision@100 = Precision@500 = 0 for EDA against exact labels. The paper claims EDA provides "a tractable candidate set for downstream IG-based verification" but EDA does not return any absorbed features in the top 500 candidates.

These two issues together mean the paper overstates both the theoretical framework's empirical support and EDA's practical utility as presented. The fix for both is framing, not new experiments: honest reordering of contributions and reframed practical claim.

---

## Dimension Scores

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | 6/10 | EDA as probe-free detector is novel; rate-distortion threshold reformulation is incremental over Tang et al. Cross-directional cosine is an independent empirical discovery. ASI (originally proposed) was falsified — the novelty is narrower than the paper implies. |
| Technical Soundness | 5/10 | Proposition 1 proof is algebraically valid but its geometric prediction is empirically reversed. Proposition 2 is a mechanistic conjecture with partial support. Precision@k data contradicts the shortlisting claim. EDA_norm (0.738) significantly outperforms EDA_base (0.650) but the weaker metric is foregrounded. |
| Experimental Rigor | 5/10 | Strong statistical methodology (bootstrap CIs, DeLong tests, permutation null) but extreme label sparsity (n_pos=18) and Precision@500=0 limit practical conclusions. Hook-point confound in B3 unacknowledged. Semantic hierarchy null is underpowered (n=20). Missing cross-directional metric validation against exact labels. |
| Reproducibility | 5/10 | Source JSON files exist for all claims and are internally consistent. However, sae-spelling commit hash, FeatureAbsorptionCalculator parameters, and SAELens release hash are absent. The 18 absorbed feature IDs are not in the paper. Label count discrepancy (paper says n_pos=50 for cross-directional metric; raw data shows n_pos=63). |

---

## Critical Issues

### 1. Proposition 1 Direction Falsification

The theoretical threshold λ > sin²(θ_{p,c}) predicts that absorbed feature pairs require **small** decoder-decoder angle (sin²(θ) < λ ≈ 0.02). The empirical prediction is thus: absorbed pairs should have **smaller** cos²(θ) between parent and child decoders than non-absorbed pairs.

B1_decoder_geometry.json cross-validates this directly:
- Absorbed (letter) features: pos_cos2_mean = 0.052, pos_cos2_std = 0.039
- Non-absorbed features: neg_cos2_mean = 0.127, neg_cos2_std = 0.181
- AUROC of cos² classifier = 0.318 (below random)
- Cohen's d = -0.57 (direction reversed)

The paper reframes this in Section 3.3 as: "once absorption is established, the child decoder drifts away from d_p." This is plausible but derives from no theorem. The audit_report_summary.md is unambiguous: "H1 is FALSIFIED by this direction."

**Required fix**: The Introduction and Abstract must not present Proposition 1 as empirically predictive in the geometric direction without acknowledging this falsification. The correct framing is: Proposition 1 identifies the **onset condition** for absorption to be loss-preferred; the **post-convergence geometry** shows the opposite pattern, consistent with subsequent decoder drift. Reorder contributions to put EDA empirical discovery first.

### 2. Cross-Directional Metric Label Count Discrepancy

The paper states "cross-directional metric with proxy labels (n_pos=50)." B1_pairwise_eda.json uses threshold=0.29, yielding n_pos=63. The source data for AUROC=0.730 with Cohen's d=0.552, p=2.78e-9 is unambiguously n_pos=63 (cos_enc_p_dec_c_max in B1_pairwise_eda.json).

This means three label sets are in play: exact (n=18), proxy (n=50 in D1/D2), and proxy (n=63 in B1). Table 1 does not disambiguate which rows use which label set. This is a methodological error that a reviewer will flag immediately.

### 3. Shortlisting Claim vs. Raw Precision

D1_eda_validation.json:
```
"precision_at_50": 0.0,
"precision_at_100": 0.0,
"precision_at_500": 0.0
```

The paper (Abstract, Section 5.1, Conclusion) claims EDA provides "a tractable candidate set for downstream IG-based verification." This claim is inconsistent with Precision@500 = 0. The minimum k at which any absorbed feature is found in EDA-ranked features is not reported.

The correct practical claim, supported by data, is: "EDA provides statistical enrichment at the distributional level (AUROC = 0.650, AUPRC = 2.09× base rate) but does not reduce the candidate pool to fewer than ~1000 features for the 18-positive label set."

### 4. EDA_norm Suppressed

D2_eda_variants.json shows EDA_norm AUROC = 0.7377, DeLong p = 0.0007 better than EDA_base (0.6503). This is a statistically significant improvement. The paper buries EDA_norm in a supplementary note. The mechanistic connection is clear: absorbed features experience competing gradient pressure → elevated encoder norm → EDA_norm = EDA_base × ‖enc_j‖ naturally captures both angular dissociation and norm inflation. EDA_norm should be the primary metric.

---

## Major Issues

### 5. Absorption Rate Range Inconsistency

Three values for the same quantity:
- Abstract: "92–97%" (= 0.919–0.968, corresponds to jb suite approximate)
- Section 4.4: "0.876–0.978" (verified from E1_phase_transition.json, all 11 configs)
- Conclusion: "0.919–0.968" (does not appear in any source file)

Cross-validated range from E1: AJT configs include 0.876 (gpt2-small-res_scl-ajt) and 0.978 (gpt2-small-res_sle-ajt). Unified correct value: **0.876–0.978**.

### 6. First-Letter Absorption Heterogeneity

C2_child_suppression_absorption.json: 8 letters tested, only letter 'h' shows absorption (rate=0.067), 7/8 letters at 0. The aggregate 0.0083 with ratio-to-null=10.0 is technically accurate but hides that the effect is concentrated in one letter. The paper's claim "confirming genuine absorption" is overstated given one-in-eight letters show the effect.

### 7. Hysteresis Uninterpretable

E2_hysteresis.json: absorption trajectory 0.959 → 0.959 → 0.960 → 0.960 → 0.960 at 100-500 fine-tuning steps. From-scratch at target sparsity: 0.964. The Conclusion states this is "consistent with the absorbed state being metastable." Metastability requires a second stable state; none has been found. The saturation itself is the finding, not hysteresis.

### 8. B3 Hook-Point Confound

Standard SAE: `blocks.6.hook_resid_pre`. TopK SAE: `blocks.6.hook_resid_post`. These are different network positions. The paper reports TopK showing "weaker EDA signal" and attributes this to the exact-sparsity constraint. The hook-point difference is a competing explanation that is not acknowledged.

### 9. Reproducibility Gaps

From D1_eda_validation.json: `label_source_file` = `iter_001/exp/results/r4/r4a_direct_labels.json`. The 18 absorbed feature IDs are listed there but not in the paper. The sae-spelling repository, commit hash, FeatureAbsorptionCalculator parameters (threshold, n_tokens, letter set), and SAELens release identifier are all absent from Section 3.4.

---

## What Works Well

1. **Statistical methodology**: Bootstrap CIs (10k resamples), permutation null z-scores, DeLong test comparisons, Cohen's d effect sizes are all appropriately applied throughout.

2. **Negative result reporting**: Three AJT configurations all show inverted polarity; phase transition hypothesis falsified with exact LRT p-value; hysteresis untestable (honestly stated in Section 5.4); semantic hierarchies null at n=20 (acknowledged as underpowered). This is exemplary.

3. **Cross-validation pipeline**: The audit_report.json system (79 claims, 0 discrepancies) demonstrates rigorous claim-evidence linkage. The author's willingness to document and flag the tautological EDA variant (AUROC=1.0) rather than exploit it shows scientific integrity.

4. **EDA decomposition** (Section 4.2): Decoder-probe alignment 0.383 vs. encoder-probe alignment 0.139 (diff = -0.244, t=-38.3, p=3.5e-38) is among the strongest mechanistic evidence in the paper, verified from B1_eda_decomposition.json. The asymmetry is absent for non-letter features (diff = -0.043), ruling out a universal SAE geometry effect.

5. **GPT-2 Small as validation anchor**: Open model with exact labels provides clean, reproducible evidence anchoring all AUROC claims.

---

## Scope Assessment vs. Original Proposal

The original proposal promised: (1) cross-domain characterization on Gemma 2 2B, (2) ASI with AUROC ≥ 0.70, (3) phase-transition dynamics, (4) absorption impossibility theorem. The delivered paper has: (1) GPT-2 Small only, (2) EDA replacing ASI (ASI falsified), (3) phase stability (no transition found), (4) no impossibility theorem. This pivot is scientifically honest but the paper should explicitly state that the original ASI proposal failed and EDA emerged as the empirical replacement.

---

## Recommended Path for Next Iteration

Priority-ordered actions that would move the score from 5.5 to 6.5+:

1. **Reframe theory**: Move Proposition 1 to "characterizes onset condition," not "predicts absorbed geometry." Add B1 falsification result explicitly. Cost: 2-3 sentences added.

2. **Fix label count**: Reconcile n_pos=50 vs. n_pos=63 discrepancy. Report cross-directional metric against exact labels (n_pos=18). Cost: one analysis run + table update.

3. **Reframe shortlisting**: Replace "tractable candidate set" with "statistical enrichment at distributional level." Report min-k. Cost: prose change only.

4. **Promote EDA_norm**: Move AUROC=0.738 to primary position in Table 1, Abstract, Conclusion. Cost: prose change + table update.

5. **Unify absorption rate range**: Use 0.876–0.978 everywhere. Cost: find-and-replace.

6. **Fix Spearman rho**: Choose one canonical value (rho=-0.482, absorption_rate vs. 1/L0 from E1) and use consistently. Cost: prose change.

Items 1–6 collectively would resolve all critical and major issues without new experiments. The resulting paper would be borderline-accept at a top venue given its genuine empirical contribution, strong statistical practice, and honest negative result reporting. Cross-model generalization (Gemma) would be needed to reach clear accept (7.5+).

---

*This review was conducted by independent supervisor agent with direct cross-validation of all quantitative claims against source JSON files in `exp/results/full/`. Score of 5.5 reflects the current state of the manuscript; the research contribution itself, if properly framed, warrants a 6.5 score.*

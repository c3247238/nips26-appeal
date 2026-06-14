# Methodologist Analysis: SAE Feature Absorption Experiment Audit

**Agent:** sibyl-methodologist  
**Date:** 2026-04-13  
**Source files examined:**
- `exp/results/full/audit_report.json` and `audit_report_summary.md`
- `exp/results/full/D1_eda_validation.json`
- `exp/results/full/B1_decoder_geometry.json`
- `exp/results/full/B1_pairwise_eda.json`
- `exp/results/full/C2_child_suppression_absorption.json`
- `exp/results/full/E1_phase_transition.json`
- `exp/results/full/B2_scaling_curve.json`
- `plan/methodology.md` and `plan/task_plan.json`
- `idea/proposal.md`
- All pilot summaries under `exp/results/pilots/`

---

## 1. Baseline Fairness Audit

**EDA vs. baselines in D1 (AUROC comparison):**
The primary detector (EDA = 1 − cos(enc_j, dec_j)) is evaluated against four baselines: random, cos(enc, dec), activation frequency, and decoder norm. The baseline set is reasonable for a detection problem of this type. However, two asymmetries are present:

- **Encoder norm baseline was discovered post-hoc.** Encoder norm achieves AUROC = 0.757, outperforming EDA (0.650). This baseline was not specified in the original methodology (methodology.md lists "cos(encoder_j, decoder_j) alone," "feature activation frequency alone," and "decoder norm alone"). Adding encoder norm in the analysis creates a comparison where the proposed metric (EDA) is not the best-performing detector, yet the paper may still frame EDA as the contribution. The paper must acknowledge that a simpler baseline (encoder norm) outperforms EDA, with DeLong p = 0.153 (not significant at α=0.05, but the effect is directionally clear).

- **Hyperparameter budget asymmetry (threshold selection):** EDA's AUROC shows substantial variation depending on the probe threshold used to define "letter features": the same metric achieves AUROC = 0.681 under one threshold (thr=0.32, n=50) and AUROC = 0.469 under a slightly lower threshold (thr=0.29, n=63) in two pilots run days apart. This threshold-sensitivity is a significant fairness concern: if baselines had been similarly threshold-optimized, they might also vary. The final threshold must be fixed before comparing against baselines, not selected after observing the desired comparison.

**Assessment:** Baselines are conceptually fair but not equally tuned to an optimal hyperparameter. The encoder norm omission from the original methodology, followed by post-hoc inclusion when it outperforms EDA, is a weak form of baseline suppression risk. **Action required:** Declare the threshold and baseline set a priori in the methods section.

---

## 2. Metric-Claim Alignment

| Claimed Contribution | Evaluation Metric | Alignment Assessment |
|---|---|---|
| EDA detects absorbed features (probe-free) | AUROC vs. Chanin exact labels (n_pos=18) | **PARTIAL FIT** — AUROC captures ranking, but AUPRC=2.1x base (target was 3x). With n_pos=18 in a set of 24,576 features, precision-at-k is 0.0 for k=50, 100, 500. The paper cannot claim practical utility at high precision. |
| Rate-distortion threshold (lambda > sin^2(theta)) predicts absorption | AUROC of threshold classifier | **MISMATCH** — H1 falsified. Absorbed features have LOWER cos^2 (AUROC=0.32 < 0.5), i.e., the direction is inverted. The metric correctly captures direction; the theory was wrong. |
| ASI predicts absorption | AUROC vs. exact labels | **MISMATCH** — H3 falsified. ASI AUROC=0.42, below null (0.50). The metric correctly captures failure. |
| Cross-domain absorption exists | Absorption rate ratio-to-null | **PARTIAL FIT** — First-letter hierarchy shows ratio=10 but absolute absorption rate is 0.83% (1/120 events). Semantic hierarchies show ratio=1.0 (no effect). The claim requires reformulation: absorption is primarily an orthographic, not semantic, phenomenon in GPT-2 Small. |
| Phase transition in sparsity | BIC comparison (sigmoid vs. linear) | **MISMATCH** — H4a not supported: BIC diff=−3.22, LRT p=0.456. The EDA proxy from the earlier pilot (LRT p=0.027) was measuring a different quantity (EDA_delta, not true absorption rate). Conflation of proxy and true metric inflated apparent support. |

**Most serious gap:** The paper claims "probe-free detection" as a contribution, but the practical precision at the top-k is zero (Precision@50 = Precision@100 = Precision@500 = 0.0 for EDA). AUROC captures rank ordering but not actionability. A researcher using EDA to find absorbed features in a 24,576-feature SAE cannot identify any absorbed features in the top 500 ranked by EDA. This gap must be explicitly acknowledged.

---

## 3. Validity Threats Checklist

- [x] **Data leakage:** Not applicable. Analysis is on pre-trained SAE weights; no training was performed on the evaluation labels.

- [x] **Contamination:** The exact Chanin labels come from the same SAE (GPT-2 L6, jb release) that the detectors are evaluated on. This is appropriate — the goal is detecting absorbed features in a given SAE, not generalizing to unseen SAEs. However, **the pilot used proxy labels (thr=0.32 letter features) before accessing the exact labels**. The proxy and exact label sets have Jaccard similarity = 0.115, meaning they capture largely different features. Any metric that appeared to work on proxy labels (such as the early EDA pilot AUROC=0.681) is actually measuring something different from what the exact Chanin labels capture. This is an epistemic contamination risk: the choice to pursue EDA was made after observing its proxy performance, but the exact label result is modestly lower (0.650) and the interpretation of what is being detected is different.

- [x] **Selection bias / threshold sensitivity:** EDA AUROC varies between 0.469 and 0.681 depending on the feature-identification threshold. This is post-hoc threshold selection risk. The threshold was originally set to produce ~67 letter features (consistent with expected n_pos from Chanin), then later lowered to 50, then 63, across different pilots. The final reported AUROC depends on which threshold is used.

- [x] **Overfitting to evaluation task:** All EDA results are from GPT-2 L6, jb SAE, first-letter spelling task. The finding that EDA fails at L10 (AUROC=0.337, inverted direction) shows that even within the same model, the metric does not generalize across layers. Cross-architecture failure (AJT SAEs show inverted EDA delta) was also observed. The claim that EDA is a "probe-free detector" cannot be supported without showing it works on more than one layer and architecture.

- [ ] **Statistical power concern:** The exact Chanin labels yield n_pos=18 out of 24,576 features (base rate 0.073%). AUROC is computable but imprecise at this imbalance: the null AUROC std = 0.064 means the 95% CI on EDA's AUROC (0.650) spans approximately [0.524, 0.776]. The paper must report confidence intervals on AUROC, not point estimates, for any claim of "statistically significant detection."

---

## 4. Ablation Gap Analysis

| Proposed Component | Ablation Present? | Gap Assessment |
|---|---|---|
| EDA = 1 − cos(enc_j, dec_j) | Partially — baselines include cos alone, freq alone | **GAP:** No ablation isolating EXACTLY which part of EDA drives the signal. Encoder norm AUROC=0.757 > EDA AUROC=0.650 suggests the signal in EDA comes primarily from encoder norm, not the angular component. A proper ablation would be: EDA with encoder norm removed vs. EDA with norm retained (i.e., cos(enc, dec) alone vs. EDA). The data shows both cos and cos_inverted give identical AUROC=0.650 — meaning the 1-transform adds zero information beyond the cosine. The angular component alone is the contribution, but we don't know if it is driving the result or if encoder norm is a confounder. |
| Revised theory (Proposition 2: encoder pulls toward parent decoder) | No ablation — prediction tested at L6 only | **GAP:** B1_eda_decomposition found the OPPOSITE: decoder aligns with letter probe more than encoder does (at BOTH L6 and L10). This directly falsifies the mechanistic mechanism proposed in Proposition 2 (encoder-pulls-toward-parent). The paper acknowledges this as an "unresolved tension" but does not provide a corrected mechanism. Without the correct mechanism, the theory section is post-hoc rationalization. |
| Phase transition claim | No adequate ablation — cross-layer confound | **MAJOR GAP:** The phase transition analysis (E1) varies L0 by comparing different layers (layers 2, 4, 6, 8, 10 at fixed width). Different layers represent different representation maturity, not just different sparsity. The methodology explicitly acknowledges this confound ("cross-layer comparison conflates absorption with representation maturity") but does not resolve it. The BIC test with n=5 cross-layer points has too little power to distinguish functional forms even if the confound were absent. |
| Cross-domain absorption | No ablation distinguishing measurement failure from absence of effect | **GAP:** C2-redesign (child-feature suppression) finds first_letter absorption rate = 0.83% and semantic hierarchy rates = 0.0. The paper cannot distinguish between (a) semantic absorption genuinely does not occur in GPT-2 Small, and (b) the child-feature suppression measurement protocol is too conservative to detect it. A minimum ablation would be: apply the identical measurement protocol to the first-letter task with known absorbed pairs and confirm it recovers the expected signal. The sanity check shows ratio_to_null=10 for first-letter, which passes, but the absolute rate (0.83%) is far below the rates reported by Chanin et al. (15–35%), suggesting the measurement protocol misses most absorption events even for the known-positive hierarchy. |

---

## 5. Reproducibility Score

**Score: 3 / 5**

Reasoning:
- **Positive factors:** Seeds are fixed (seed=42). GPU hardware is documented (NVIDIA RTX PRO 6000). SAE releases are named specifically (gpt2-small-res-jb). The sae-spelling repository is cited and MIT-licensed. Pre-trained SAEs are from SAELens with fixed release tags. Scripts ran in a single GPU session of <2 hours per experiment.
- **Negative factors:**
  - The letter-feature threshold (thr=0.29 vs. 0.32) is not uniquely determined and affects results substantially. A reproducer using a different threshold will get different AUROCs.
  - The exact Chanin labels come from a file at a specific path (`iter_001/exp/results/r4/r4a_direct_labels.json`) that is an artifact of the iter_001 run, not a public data release. A reproducer cannot access these without rerunning iter_001's IG-ablation experiment.
  - The cross-directional cosine metrics (cos(enc_p, dec_c)) require identifying "parent candidates" via a search over all other features in the SAE, with the best parent selected by max cosine. The parent identification protocol is not fully specified (which candidate pool? what threshold?).
  - EDA AUROC is sensitive to which Chanin labels are used (proxy vs. exact) and the pilot/full experiment distinction is not always clear from the JSON outputs.

---

## 6. Top-3 Methodology Improvement Recommendations

**Priority 1 (Highest impact on paper credibility): Resolve the encoder-norm confound in EDA**

The critical finding is that encoder norm (AUROC=0.757) outperforms EDA (0.650). If encoder norm alone is the real signal, then the EDA contribution reduces to "absorbed features have larger encoder rows" — which is a simpler and less theoretically motivated finding. The paper must either (a) show that EDA outperforms encoder norm after proper matching/normalization, or (b) change the primary claim to "absorbed features have high encoder norm" with EDA as an interpretable proxy for this norm signal. **Experiment required:** Compute EDA-norm variant (EDA × ||enc_j|| / ||dec_j||) and compare AUROC against both EDA and encoder norm alone. If EDA-norm > encoder norm > EDA, encoder norm is a confounder; if EDA > EDA-norm ≈ encoder norm, the angular component is the true signal. This is a 30-minute computation already specified as task_D2_eda_variants and must be reported before any paper claim.

**Priority 2 (Protects cross-domain H2): Replace C2 child-suppression rate with a validated protocol**

The C2-redesign absorption rate for first-letter (0.83%, 1/120 events) is vastly lower than Chanin et al.'s reported rates (15–35%). This means the redesigned measurement misses approximately 95% of known absorption events. Two consequences: (a) negative results for semantic hierarchies are uninterpretable (could be measurement miss, not absence of effect); (b) positive results at 0.83% vs. 0% null are too noisy to draw cross-domain conclusions. **Fix required:** Calibrate the child-feature-suppression protocol against the exact Chanin labels for first-letter before applying it to semantic hierarchies. The sanity check passes the ratio criterion but not the absolute sensitivity criterion. A minimum acceptable sensitivity is recovery of ≥50% of known absorbed pairs from the Chanin list. If the protocol cannot meet this bar for first-letter, cross-domain comparisons are invalid.

**Priority 3 (Protects phase transition H4a): Add within-layer sparsity variation**

The current phase transition analysis varies sparsity by using different model layers (2, 4, 6, 8, 10), not different sparsity settings within the same layer. This confounds sparsity with representation depth. The methodology document explicitly flags this as a limitation ("cross-layer comparison conflates absorption with representation maturity") but the fix (using AJT SAEs at different L0 within layer 6) has not been executed. The AJT SAEs at layer 6 show inverted EDA signals, creating an additional complication. The result (BIC diff=−3.22, LRT p=0.456) is the most honest available answer: no phase transition is detected with current evidence. The paper may claim this as a null result for H4a. However, claiming "phase transition not found due to cross-layer confound" is different from "phase transition not present." The paper must distinguish these clearly.

---

## Summary Table

| Issue | Severity | Status | Action |
|---|---|---|---|
| Encoder norm outperforms EDA (post-hoc baseline) | HIGH | Unresolved | D2 must compare EDA-norm variant |
| Threshold sensitivity of EDA AUROC (0.469–0.681) | HIGH | Unresolved | Fix threshold a priori before final reporting |
| n_pos=18 exact labels: AUROC CI too wide for strong claims | MEDIUM | Reported | Report 95% CI on AUROC; soften language |
| Precision@k = 0.0 for all k (practical utility absent) | HIGH | Reported | Cannot claim "practical" probe-free detection |
| C2 protocol recovers only 0.83% of first-letter absorption | HIGH | Unresolved | Calibrate protocol before cross-domain inference |
| Phase transition: cross-layer L0 variation confound | MEDIUM | Acknowledged | Report as limitation; do not claim H4a supported |
| H1 falsified (direction reversed, cos^2 AUROC < 0.5) | HIGH | Documented | Remove H1 RD threshold claim from paper |
| H3 falsified (ASI AUROC < null) | HIGH | Documented | Remove ASI as contribution; report as negative result |
| Proposition 2 mechanism contradicted by B1_eda_decomposition | HIGH | Unresolved | Theory must be revised or demoted to conjecture |
| Jaccard=0.115 between proxy and exact labels | MEDIUM | Documented | Clearly separate proxy from exact results throughout |
| EDA fails at L10 and for AJT architecture | MEDIUM | Documented | Scope claims to GPT-2 L6 jb SAE explicitly |
| Table 1 incomplete: no L10 or Gemma columns | LOW | Documented | Scope paper to available results |

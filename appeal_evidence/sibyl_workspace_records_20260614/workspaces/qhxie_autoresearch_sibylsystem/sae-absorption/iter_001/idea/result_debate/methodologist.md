# Methodologist Audit: Feature Absorption in SAEs — Post-Round 4 Update

**Auditor:** Methodologist Agent  
**Date:** 2026-04-13  
**Scope:** Complete methodology audit covering Rounds 1-4 for "Characterizing SAE Absorption: A Regime-Specific Detector and Three-Subtype Taxonomy"

---

## 1. Baseline Fairness Audit

### 1.1 EDA Evaluation — Label Source Asymmetry Across Models

**Gemma 2B configs (R3):** Ground-truth labels are Neuronpedia proxy scores, NOT Chanin et al. direct FeatureAbsorptionCalculator outputs. The Gemma 2B model remained HF-gated through all four rounds. The two strongest AUROC results in the paper (L5-16k: 0.698, L12-16k: 0.776) rest entirely on these proxy labels. No calibration or cross-validation between proxy and direct labels has been performed on Gemma SAEs.

**GPT-2 configs (R3+R4):** Labels from the actual FeatureAbsorptionCalculator. GPT-2 L6 achieves AUROC = 0.650 with direct labels. This is the single most methodologically clean data point in the paper. However, GPT-2 L10 shows AUROC = 0.344 with **reversed direction** (absorbed latents have LOWER EDA, Cohen's d = -0.374), which is a fundamental failure mode, not merely a weak result.

**Llama-3.1-8B configs (R4):** Model also HF-gated. Only SAE weight-level statistics were computed. No absorption labels were generated. The "cross-model validation on 3 architectures" claim is misleading: Llama contributes only EDA distribution statistics, not AUROC validation. The self-supervised percentile labels (top-1%/2%/5% of EDA as "absorbed") produce AUROC = 1.0 by construction (selecting high-EDA features then testing whether EDA discriminates them is circular).

**Asymmetry verdict:** The three passing configs use two different label sources (proxy for Gemma, direct for GPT-2). The paper must clearly disclose that Gemma and GPT-2 results are not directly comparable. The Llama contribution should be characterized as "weight-level structural analysis" rather than "cross-model EDA validation."

### 1.2 SynthSAEBench — Uninformative for Real SAEs

SynthSAEBench achieves AUROC = 1.0 with F1 = 0.974 on synthetic SAEs (500 features, 64 dimensions). Real SAEs operate at 16,384-65,536 features in 768-4,096 dimensions. The ratio is >100x in dimensionality. This control validates only that EDA computation is correct, not that it has discriminative power on real SAEs. It should be positioned as a sanity check, not as evidence for EDA efficacy.

### 1.3 RAVEL Probes — Cascading Validity Failure (R3 to R4)

**R3 (Qwen2.5-0.5B bridge):** Probes trained on wrong model (d_model=896), projected via random QR to Gemma space (d_model=2304). All probes failed quality gate (best: 71.4% city-continent). Intra-RAVEL rho = 0.924 (n=6 configs).

**R4 (GPT-2 Medium bridge):** Probes retrained on GPT-2 Medium (d_model=1024), projected to Gemma space. Quality further degraded: best = 59.5% city-continent (down from 71.4%). All probes again fail quality gate (85% threshold). The shuffled hierarchy control then showed real absorption rates indistinguishable from shuffled null for all 3 domains and all SAE configs (0/9 combinations exceed shuffled p95).

**Critical finding:** The H3 collapse is definitive. The shuffled control experiment — which was missing in the R3 audit and explicitly recommended — has now been run and confirms the worst-case scenario: the "cross-domain absorption" signal measured with bridge-model probes is indistinguishable from noise. The R3 intra-RAVEL rho = 0.924 likely reflects consistent artifacts of the projection method rather than genuine absorption coherence.

### 1.4 ITAC Baselines — Upper Bound Still Missing

The LCA-SAE iterative encoding upper bound listed in the methodology (Table of baselines) was never implemented across any round. ITAC's 2.69% FN reduction at L12-65k (0% at L12-16k) cannot be contextualized without knowing what the upper bound achieves. R4D (ITAC on real activations) was not completed. The ITAC evaluation remains limited to synthetic activations (decoder column inputs), which is acknowledged as a methodological limitation.

### 1.5 Phase 5 / R4A GPT-2 — EDA = Decoder Cosine Identity

R4A revealed that EDA AUROC and negated decoder cosine AUROC are **identical** (delta = 0.000 on both GPT-2 configs). This was confirmed on all three architectures (Gemma, GPT-2, Llama) with Pearson r = -1.000. EDA(j) = 1 - cos(w_enc, d_j) is mathematically equivalent to the negated decoder-encoder cosine similarity already present in SAEBench.

**Impact on novelty:** The primary detection metric is not a new formula but a re-derivation of an existing SAEBench metric. The contribution must be repositioned as systematic empirical characterization of an existing metric rather than a novel detector. The formal lower bound theorem provides theoretical novelty, but the empirical metric itself is not new.

---

## 2. Metric-Claim Alignment

| Claimed Contribution | Evaluation Metric | Alignment Assessment |
|---|---|---|
| EDA as regime-specific detector | AUROC vs. proxy/direct labels | **Moderate gap.** AUROC is appropriate for ranking quality. However, 3/8 configs passing is presented as "regime-specific success" rather than "minority of configs work." No operating-point analysis (precision-recall at specific thresholds) is provided. The claim "0.65-0.78 in favorable regimes" omits that 5/8 configs fail, including the reversed-direction GPT-2 L10 result. |
| Three-subtype taxonomy | KW test, Mann-Whitney, threshold sensitivity | **Partially valid, but underpowered.** L12-65k shows KW p = 0.0002 (n=65 absorbed); L12-16k fails all tests (KW p = 0.237, n=16 absorbed). The late > early ordering holds, but partial < early (contradicting the predicted intermediate position). Only 2 SAE configs tested. |
| Early-dominance finding (72-75%) | Proportion of early-type at tau=0.3 | **Threshold-sensitive.** At tau=0.2, early proportion drops to ~32% at L12-65k (per threshold sensitivity ablation). The "72-75%" figure is specific to tau=0.3. The paper must report threshold sensitivity prominently, not in supplementary material. |
| Cross-model generalization | AUROC across model families | **Weakened.** Only 2 model families (Gemma proxy, GPT-2 direct) contribute AUROC. Llama contributes no AUROC. "3 model families" is accurate only for weight-level statistics, not for the detection claim. |
| Cross-domain absorption | Absorption rate > random; intra-RAVEL rho | **COLLAPSED.** R4B shuffled control shows real = shuffled for all domains. The intra-RAVEL coherence claim (rho = 0.924) is no longer credible given that both real and shuffled rates are in the same range. |

---

## 3. Validity Threats Checklist

- [ ] **Data leakage:** LOW RISK. EDA is weight-only; labels come from separate pipelines. No train-test contamination detected.
- [x] **Model substitution (systematic):** HIGH RISK — PARTIALLY MITIGATED BY R4. Gemma 2B remained gated. R4 added GPT-2 direct labels (genuine mitigation for GPT-2 SAEs). But Gemma SAE results still rely on proxy labels with no ground-truth calibration. Llama model also gated.
- [x] **Proxy label validity:** HIGH RISK — UNRESOLVED. No experiment in any round calibrates Neuronpedia proxy labels against FeatureAbsorptionCalculator direct labels on Gemma SAEs. The two Gemma configs that pass AUROC >= 0.65 (L5-16k: 0.698, L12-16k: 0.776) could be inflated or deflated relative to direct-label AUROC. The GPT-2 direct-label AUROC (0.650) at a comparable config is notably lower than the Gemma proxy-label AUROCs, raising questions about whether proxy labels are more favorable.
- [x] **Circular self-supervised labels (Llama):** HIGH RISK. The Llama "AUROC = 1.0" on self-supervised percentile labels is a tautology (selecting high-EDA features then testing EDA discrimination). This should not appear in any results table as if it were a genuine validation.
- [x] **Dimensionality mismatch (RAVEL probes):** HIGH RISK — CONFIRMED BY R4. Both Qwen bridge (R3) and GPT-2 Medium bridge (R4) fail quality gates. Shuffled control confirms the "absorption" signal is an artifact. H3 contribution is no longer viable.
- [ ] **Selection bias / HARKing:** MODERATE RISK. The pivot from "universal detector" to "regime-specific detector" after seeing which configs pass is post-hoc framing. However, the pass criteria (AUROC >= 0.65) were pre-specified, mitigating this concern. The threshold tau=0.3 for taxonomy was pre-specified with sensitivity analysis.
- [x] **Small N for taxonomy:** MODERATE RISK. Taxonomy classification operates on n=16 (L12-16k) and n=65 (L12-65k) absorbed latents. At n=16, statistical tests have very low power (KW p = 0.237). The 75% early-dominance at L12-16k rests on 12 out of 16 features.
- [x] **Class imbalance for AUROC estimation:** MODERATE RISK. At L12-65k, n_pos = 16 out of 65,536 latents (0.024% prevalence). Bootstrap CIs are wide. The pilot-to-full AUROC collapse (0.853 to 0.468) is attributed to this imbalance. At GPT-2 L6, n_pos = 18 out of 24,576 (0.073%). AUROC reliability at such extreme imbalance is questionable regardless of label quality.

---

## 4. Ablation Gap Analysis

| Proposed Component | Ablation Present? | Assessment |
|---|---|---|
| EDA as absorption indicator | Yes (vs. decoder cosine, random shuffled, null distribution) | Adequate baselines. **However:** EDA = negated decoder cosine identity means the "EDA vs. decoder cosine baseline" comparison is meaningless (they are the same metric). The +0.396 AUROC advantage claimed in the proposal is an artifact of comparing 1-cos against a different baseline, not against the identical metric. |
| Three-subtype taxonomy | Yes (5 threshold values: 0.2-0.4) | Present. Results show early-dominance is threshold-sensitive (75% at tau=0.3 vs. 32% at tau=0.2 for L12-65k). This is a meaningful sensitivity that must be prominently reported. |
| ITAC correction | Partial (null test on early-absorbed) | **Missing:** LCA-SAE upper bound comparison. Missing: ITAC hyperparameter sensitivity. Missing: real-activation evaluation (R4D not completed). |
| Polysemanticity stratification | Yes (full ablation on L12-16k, L12-65k) | Present. Counterintuitive result: EDA AUROC is HIGHER in polysemantic stratum (0.92 vs. 0.64 at L12-16k), contradicting the hypothesis that polysemanticity confounds EDA. |
| Cross-domain generalization | Yes (R4B shuffled hierarchy control) | **Completed and falsified H3.** This was the most important missing control from R3, and its execution is a methodological strength. The negative result is definitive. |
| Cross-model generalization | Partial (GPT-2 direct + Gemma proxy) | **Missing:** Same-label-type comparison across models. Gemma proxy labels and GPT-2 direct labels may have different sensitivity/specificity profiles, confounding cross-model comparison. |

---

## 5. Reproducibility Score: 3/5

| Factor | Score | Notes |
|---|---|---|
| Random seeds fixed | 5/5 | seed=42 used consistently across all experiments |
| Hyperparameters specified | 4/5 | All thresholds, bootstrap counts, SAE configs documented in JSON artifacts |
| Code available | 3/5 | Code in `exp/code/` with versioned scripts. No standalone release package or environment specification beyond dependency list |
| Hardware documented | 2/5 | "RTX PRO 6000 Blackwell 95GB" mentioned in task plan; actual per-experiment GPU usage, peak memory, and wall time not systematically logged |
| Independent reproducibility | 1/5 | **BLOCKING:** Gemma 2B and Llama-3.1-8B both HF-gated. Proxy labels from Neuronpedia may not be regeneratable from the exact same snapshot. A reviewer cannot reproduce the Gemma AUROC results without model access. GPT-2 results ARE reproducible. |

**Overall: 3/5.** The GPT-2 pipeline is reproducible by any researcher. The Gemma pipeline requires resolving HuggingFace access (a known gating issue, not the authors' fault, but still a barrier). Llama pipeline is partially reproducible (weight-only statistics) but not validation (model gated).

---

## 6. Top-3 Highest-Impact Methodology Improvements

### Recommendation 1: Resolve EDA = DecCos Equivalence in Framing (Effort: LOW, Credibility Impact: CRITICAL)

R4 definitively established that EDA(j) = 1 - cos(w_enc_j, d_j) is mathematically identical to the negated decoder-encoder cosine similarity already in SAEBench. This means:

- The "EDA vs. decoder cosine baseline" comparison in Phase 1 was comparing EDA against itself (the +0.396 AUROC advantage reported in the proposal likely compared against a different baseline definition, not the identical metric).
- The paper cannot claim EDA as a "novel detector." It must be positioned as: (a) a formal theoretical lower bound connecting an existing metric to absorption, and (b) a systematic regime characterization of when this known metric has discriminative power.

**Specific action:** Rewrite the contribution as "We provide the first formal theoretical lower bound (Theorem 1) connecting encoder-decoder cosine similarity to absorption degree, and systematically characterize its regime-dependent discriminative power across 3 model families and 8 SAE configurations." Drop any language suggesting EDA is a new metric.

### Recommendation 2: Report Threshold Sensitivity of Early-Dominance Claim in Main Text (Effort: LOW, Credibility Impact: HIGH)

The "72-75% early absorption" finding is the paper's strongest and most impactful result. However, at tau=0.2 (only 0.1 below the default), early proportion drops to ~32% at L12-65k. This means the dominance finding is a function of the threshold, not a robust structural property.

**Specific action:** Include a figure or table in the main text (not appendix) showing early/late/partial proportions across all 5 threshold values for both SAE configs. Frame the finding as: "At the standard threshold (tau=0.3), early absorption dominates (72-75%). The proportion is threshold-sensitive (dropping to 32% at tau=0.2), reflecting a continuum rather than a sharp boundary." This framing is actually more interesting and nuanced than a simple "75% early" claim.

### Recommendation 3: Obtain Gemma 2B Access or Accept Two-Model Paper (Effort: MEDIUM, Credibility Impact: HIGH)

After four rounds, the Gemma 2B access issue remains unresolved. The paper currently reports results on Gemma SAEs with proxy labels that cannot be independently verified. Two paths forward:

**Path A (preferred):** Obtain Gemma 2B access. Re-run Phase 1 with direct FeatureAbsorptionCalculator labels on all 6 Gemma configs. This would either confirm or correct the L5-16k (0.698) and L12-16k (0.776) results with ground-truth labels.

**Path B (acceptable):** Accept the paper as a GPT-2-primary paper with Gemma supplementary evidence. Lead with the GPT-2 L6 direct-label result (AUROC = 0.650) as the primary validation, add Gemma proxy-label results as supplementary with explicit "unverified with direct labels" caveat, and drop the Llama AUROC claims entirely (report only weight-level statistics). This is honest and still publishable — the GPT-2 direct-label validation + taxonomy + theory constitute sufficient contributions.

---

## Summary Assessment

**What improved from R3 to R4:**
- The shuffled hierarchy control was executed, definitively resolving the H3 question (collapsed).
- GPT-2 direct labels via FeatureAbsorptionCalculator were generated, providing the cleanest validation data point.
- The EDA = DecCos equivalence was empirically confirmed on 3 architectures, clarifying the metric's relationship to prior work.
- Honest hypothesis status tracking: H2, H3, H5, H6 all correctly marked as FALSIFIED.

**What remains problematic:**
- Gemma 2B model access was not resolved. The two strongest AUROC results (0.698, 0.776) cannot be verified with direct labels.
- The Llama "cross-model validation" is weight-level statistics only, not AUROC validation. The self-supervised AUROC = 1.0 is circular.
- The paper went from 6 hypotheses to 4 falsified + 1 partially confirmed + 1 confirmed. This is a very high failure rate that requires honest framing.
- The early-dominance finding (the strongest remaining claim) is threshold-sensitive and rests on 2 SAE configs.
- EDA is not a novel metric — it is a known SAEBench metric with a new theoretical justification.

**Bottom line:** The paper should be framed as: (1) a formal theoretical lower bound connecting encoder-decoder cosine similarity to absorption, validated empirically in a regime-specific manner; (2) a three-subtype taxonomy revealing threshold-dependent early-absorption dominance; and (3) a systematic negative-results survey covering D-EDA, ITAC, cross-domain generalization, and scaling predictions. This honest framing is publishable and credible. Over-claiming any of the collapsed hypotheses would severely damage reviewer trust.

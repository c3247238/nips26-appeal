# Result Debate Synthesis: SAE Feature Absorption (iter_003)

**Synthesized:** 2026-04-14  
**Evidence sources:** 6 result debate perspectives (optimist, skeptic, strategist, methodologist, comparativist, revisionist) + full experiment JSON outputs (A1, A2, A3, B1, B2, C1, C2, D1, D2, E1, F1)

---

## 1. Consensus Map

The following conclusions are endorsed by all 6 perspectives and are considered high-confidence:

| Conclusion | Evidence | Confidence |
|---|---|---|
| encoder_norm AUROC=0.757 (Standard/L1, L6) significantly exceeds EDA AUROC=0.650 | DeLong z=3.046, p=0.0012 (A1, A3, B2) | HIGH |
| encoder_norm AUROC=0.837 (TopK-32k, L6) is the strongest result in the package | n_pos=77, tight CI [0.807-0.870]; Cohen's d=1.23 (A3) | HIGH |
| OMP oracle achieves 0% FN reduction vs. feedforward encoder across all 18 absorbed features | Mean AR_feedforward=0.978, Mean AR_OMP=0.978; omp_reduction_ratio=0.0 for all letters (C2) | HIGH |
| O_jaccard AUROC=0.721 (AUPRC=0.075, Precision@50=0.10) — second strongest single detector | B2 full results | MODERATE-HIGH |
| F1: 67% of absorbed features (12/18) recovered in wider 32k SAE at cos_sim>0.80 | mean_best_cosine=0.791, median=0.815 (F1) | MODERATE |
| H3 entity-type absorption: AR=0.000, probe methodology likely invalid | D2 entity AR=0.0 vs control AR=0.05; Methodologist: Qwen→GPT-2 projection via random QR is unvalidated | LOW (artifact) |
| Hook confound exists in cross-architecture comparison | Standard SAE: resid_pre; TopK SAE: resid_post; explicitly flagged in A3 JSON | CONFIRMED LIMITATION |

---

## 2. Conflict Resolution

### Conflict A: Is the paper ready to submit? (Optimist vs. Skeptic)

**Skeptic concern:** n_pos=18 at L6, wide CIs (~±0.09), possible label errors from Neuronpedia proxy.  
**Optimist counter:** DeLong test is significant at p=0.0012; the L10 result (n_pos=39) replicates direction (encoder_norm 0.645 >> EDA 0.344); TopK result (n_pos=77) further validates.

**Judgment:** The skeptic is technically correct about n_pos=18 being small for a standalone paper. However, the convergence of three independent results (L6 IG labels, L10 proxy labels, TopK-32k proxy labels) all showing encoder_norm >> EDA substantially reduces the single-sample-fragility concern. The paper should present all three results together with CIs and explicit discussion of label quality differences. The skeptic's concern is a *presentation requirement*, not a fatal flaw.

### Conflict B: Does the hook confound invalidate the cross-architecture claim?

**Skeptic/Methodologist concern:** AUROC difference (0.757 vs. 0.837) may reflect resid_pre vs. resid_post rather than Standard vs. TopK architecture.  
**Optimist/Comparativist counter:** The finding that encoder_norm *works on both* is the claim, not that the AUROC values are directly comparable.

**Judgment:** Both sides are right about different claims. The claim "encoder_norm works on both Standard and TopK architectures" is valid and unconfounded by hook — both architectures show strong detection. The claim "TopK shows higher AUROC than Standard due to architectural differences" IS confounded. The paper must make the first claim and explicitly disavow the second. This is a clear limitation statement requirement.

### Conflict C: Is H2 falsification convincing? (Skeptic questions OMP calibration)

**Skeptic concern:** OMP k=53 may not be a true oracle if k is too small to represent absorbed features.  
**Methodologist concern:** A truly unrestricted oracle (LASSO with very small λ) might differ.

**Judgment:** The skeptic's concern is theoretically valid but empirically unpersuasive. The key evidence is: absorption_rate under feedforward = 0.978; absorption_rate under OMP = 0.978; reduction ratio = 0.0 across all three letters (a, e, s). The null result is as strong as any oracle can demonstrate given the experimental design. If OMP at k=53 (mean L0 of feedforward encoding) cannot reduce absorption, the amortization gap hypothesis would require implausibly high sparsity levels to be rescued. The correct framing is: "under fixed-sparsity OMP with k matching feedforward mean L0, amortization gap explains 0% of absorption." This is a valid falsification of the strong form of the hypothesis with an honest caveat about fixed-k assumption.

### Conflict D: Does H3 zero-result mean anything? (Optimist vs. Methodologist)

**Optimist:** H3 negative is fine because entity-type absorption is an extension; first-letter remains the core.  
**Methodologist:** H3 result is a methodology failure (Qwen→GPT-2 via random QR), not a genuine negative.

**Judgment:** Methodologist wins. D2 AR=0.0 for entity-type is inconsistent with negative control AR=0.05 from the same test, which shows the probe system works but entity-type probe projected to wrong space. The D2 result should be reported as "preliminary investigation with unvalidated cross-model probe transfer" and explicitly NOT cited as evidence that entity-type absorption does not exist. H3 remains an open question.

---

## 3. Result Quality Score

**Score: 6.5/10**

**Justification:**
- (+) Three independent replication contexts for encoder_norm (L6 IG, L10 proxy, TopK)
- (+) DeLong test confirming encoder_norm > EDA is methodologically appropriate
- (+) OMP oracle is a clean, decisive negative for amortization gap (strong form)
- (+) O_jaccard provides independent signal; high AUPRC relative to class imbalance
- (+) F1 result is practically informative (67% recovery, 33% genuine gap)
- (-) n_pos=18 at L6 is genuinely small; any 3 mislabeled features shift results materially
- (-) Hook confound in A3 prevents clean cross-architecture attribution
- (-) H3/D2 is a methodology failure, not a scientific result
- (-) AUPRC for encoder_norm (0.004) is very low despite good AUROC — severe class imbalance remains
- (-) Single task domain (first-letter spelling) limits generalizability claims

---

## 4. Key Findings

1. **Encoder norm is the best weight-only detector of SAE feature absorption.** At GPT-2 L6 with gold-standard IG labels: AUROC=0.757 (95% CI [0.655, 0.849]), significantly exceeding EDA (0.650, DeLong p=0.0012) and all other formulations. Replicated in TopK-32k: AUROC=0.837 (CI [0.807, 0.870]; Cohen's d=1.23 absorbed vs. non-absorbed).

2. **Amortization gap does not explain absorption — sparsity landscape does.** OMP oracle at k=53 achieves identical absorption rate to feedforward encoder (0.978 vs 0.978) across all tested letters. This decisively rules out the O'Neill et al. amortization gap hypothesis under fixed-sparsity conditions, supporting Tang et al.'s partial minimum theory.

3. **Co-occurrence structure (O_jaccard) provides independent detection signal.** AUROC=0.721, AUPRC=0.075 (dramatically higher than encoder_norm's 0.004 due to different score distribution), Precision@50=0.10. The two signals (encoder geometry and activation co-occurrence) are nearly uncorrelated (Spearman ρ=0.044), suggesting complementary information.

4. **Wider dictionary width partially but incompletely remediates absorption.** 12/18 (67%) absorbed features in the 24k SAE have a direction-aligned counterpart in the 32k SAE (cos_sim>0.80). 6/18 (33%) have no close match, suggesting genuine semantic coverage gaps not addressable by capacity alone.

5. **Entity-type absorption could not be measured due to probe transfer methodology failure.** D2 results (AR=0.0) reflect Qwen-to-GPT-2 projection artifacts, not genuine absence of entity-type absorption. This remains an open question.

---

## 5. Methodology Gaps (Critical Improvements)

From methodologist and skeptic consensus:

1. **Hook-confound correction for A3 (PRIORITY: HIGH).** Run encoder_norm detection on Standard SAE at resid_post or TopK SAE at resid_pre to isolate architecture effect from hook effect. This is a low-cost additional experiment.

2. **Power analysis for n_pos=18 results (PRIORITY: HIGH).** Compute and report minimum detectable effect size at current n. Consider obtaining additional absorbed feature labels or presenting sample efficiency analysis.

3. **H3 methodology replacement (PRIORITY: MEDIUM).** Train probes directly on GPT-2 activations (not transferred from Qwen) for entity-type features, or explicitly frame D2 as a null due to methodology rather than biology.

4. **Functional recovery definition for F1 (PRIORITY: MEDIUM).** Confirm that recovered 32k features (cos_sim>0.80) actually activate on the same inputs as the absorbed 24k features, not just that decoder directions are similar.

5. **AUPRC interpretation under class imbalance (PRIORITY: LOW).** O_jaccard's high AUPRC (0.075) vs encoder_norm's low AUPRC (0.004) requires explanation. Positive class prevalence ≈ 18/24576 = 0.073%. Expected random AUPRC ≈ 0.00073%. Both are informative; note the score distribution difference.

---

## 6. Competitive Position

### vs. Chanin et al. (2024)
- Chanin: activation-based (IG), requires activation data collection, slow
- Ours: weight-only (encoder_norm), requires only SAE weight matrices, <1 second for 65k SAE
- AUROC trade-off: encoder_norm ≈ 0.76 vs. Chanin IG ≈ 0.85 (estimated)
- **Positioning:** "Weight-only pre-screening complement to Chanin IG" — enables audit-at-scale before activation data is collected

### vs. Karvonen et al. (2025 SAEBench)
- SAEBench: aggregate absorption rate metric across dictionaries
- Ours: per-latent detection + mechanistic explanation + partial remediation evidence
- **Positioning:** "Mechanistic account underlying SAEBench's metric"

### vs. Tang et al. (2025)
- Tang: theoretical formalization of partial minima causing absorption
- Ours: first empirical evidence supporting Tang vs. O'Neill (OMP oracle test)
- **Positioning:** "Empirical validation of Tang et al.'s partial minimum theory"

**Overall competitive position: SOLID for workshop; BORDERLINE for main track at top venue.** The encoder_norm result is not a breakthrough (AUROC 0.76 is good but not dramatically better than EDA 0.65), but the H2 falsification is genuinely novel and the theoretical framing is clean. With hook-confound correction and expanded features, a main-track submission becomes more defensible.

---

## 7. Hypothesis Update

| Hypothesis | Original Status | Update |
|---|---|---|
| H1: encoder_norm detects absorption better than EDA | Predicted | CONFIRMED across 3 contexts; cross-architecture (with hook caveat) |
| H2: Amortization gap causes absorption | Competing hypothesis | FALSIFIED (strong form); OMP = 0% reduction |
| H3: Entity-type features show cross-hierarchy absorption | Open | UNMEASURABLE with current methodology; not falsified |
| F1: Wider SAE recovers absorbed features | Hypothesis | PARTIALLY CONFIRMED (67%), PARTIALLY REJECTED (33%) |
| Theory: High encoder norm = competition pressure from absorbing children | Mechanistic account | CONSISTENT but post-hoc; Cohen's d=0.97-1.23 for absorbed vs. non-absorbed |

**Revised primary framing:** "Feature absorption in SAEs is a training-time dictionary coverage problem (partial minimum / sparsity landscape), not an encoder approximation problem (amortization gap). Encoder norm is the best weight-only proxy for detecting absorbed features, likely because absorption creates elevated encoder weights due to gradient competition during training."

---

## 8. Action Plan

### Recommendation: PROCEED to writing, with two parallel tracks

**Track A (immediate — 1-2 days):** Begin writing with current results. Lead with H2 falsification as primary contribution. Include hook confound as explicit limitation.

**Track B (parallel — 2-3 days):** Run hook-confound correction experiment (Standard SAE at resid_post, or confirm resid_pre ≈ resid_post for L6). This single experiment turns the limitation into a controlled result.

### Prioritized next steps:

1. **Write paper outline now** (strategist recommendation: lead with H2 negative, then detection). Do not wait for Track B experiments to begin writing.

2. **Hook-confound correction experiment** (A3 follow-up): Compute average cosine similarity between resid_pre and resid_post at L6, or load a TopK SAE trained on resid_pre. Expected time: 1-2 hours.

3. **Power analysis** (skeptic requirement): Compute minimum detectable AUROC difference at n=18. Report bootstrap CI prominently in paper.

4. **Frame H3 as future work**: Do not attempt to fix D2 probe methodology for this iteration — costs too much for a tangential result. Write one paragraph: "Probe transfer methodology was insufficient to measure entity-type absorption; we recommend direct probe training on target model as future work."

5. **Finalize paper title**: Strategist/Revisionist both recommend H2-leading title. Recommended: *"Feature Absorption in Sparse Autoencoders is Primarily a Sparsity Landscape Problem, Not an Amortization Gap"* with subtitle noting encoder_norm as the detection tool enabling this conclusion.

### Decision: **PROCEED**

The evidence base justifies writing a complete paper. The hook confound is a limitation to acknowledge, not a fatal flaw. The OMP oracle result is clean and novel. The multi-architecture encoder_norm result is practically valuable. A strong workshop paper is achievable with current results; main-venue submission should include hook-confound correction.

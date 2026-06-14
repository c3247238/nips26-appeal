# Supervisor Review: Research Contribution Assessment

**Paper**: Where and When Encoder-Decoder Misalignment Signals Feature Absorption in Sparse Autoencoders
**Date**: 2026-04-13 (updated; prior review 2026-04-12)
**Score**: 5.5 / 10
**Verdict**: CONTINUE (scope restructuring + experimental expansion required)

---

## Executive Summary

This paper addresses a genuine and important problem in mechanistic interpretability. Feature absorption in Sparse Autoencoders represents a systematic reliability failure that undermines causal analyses of language model internals. The paper's conceptual direction is sound and the intellectual contributions are real.

Since the previous review (April 12), Round 4 experiments have completed: GPT-2 exact-label direct validation (AUROC = 0.629 with n_pos=67 from R3, confirmed), the H3 RAVEL cross-domain falsification via proper shuffled hierarchy control (0/9 domain-SAE combinations pass), and Llama weight-only EDA characterization. The paper has been substantially restructured to reflect these findings, now honestly reporting the H3 collapse, the pivot to a characterization-only framing, and the regime-specific EDA limitation.

Despite these updates, four structural problems that would cause rejection at a top venue remain unresolved.

**Score calibration**: 5.5 represents work with genuine intellectual merit but experimental evidence that does not support the claims as stated. The score would rise to 7.0-7.5 with targeted fixes described below.

---

## What Has Improved Since the April 12 Review

- H3 is now honestly reported as FALSIFIED, with the shuffled hierarchy control correctly described (0/9 pass).
- The paper correctly attributes GPT2-L6 AUROC = 0.629 to exact Chanin et al. labels with n_pos = 67 (from R3 Phase 5 full mode, 10,000 bootstrap resamples), not the R4 pilot subset.
- The EDA regime-specific framing is clearer: mid-layer, narrow SAEs only.
- The pivot away from D-EDA and ITAC as primary contributions has been implemented in the paper's Introduction and abstract.
- The paper reports the cross-domain null result with appropriate caveats.

---

## Dimension 1: Novelty and Significance — Score: 6/10

**EDA metric**: The underlying quantity EDA = 1 - cosine(encoder_j, decoder_j) is near-identical to SAEBench's existing `encoder_decoder_cosine_sim` metric, confirmed by Pearson r > 0.999 (paper's own Section 4.2) and R4's finding of delta AUROC = 0.0 between EDA and dec_cosine_negated. The paper presents EDA as a new metric in the Introduction without disclosing this pre-existing quantity. The novelty argument rests on Theorem 1 — if Theorem 1 is a genuine formal proof, it represents a real contribution; if it is an informal proof sketch (which it is in its current form), the novelty reduces to a systematic empirical AUROC evaluation of an existing SAEBench signal.

**Cross-domain characterization**: The cross-domain contribution has correctly pivoted to an honest null result. The falsification of H3 is itself a useful finding — absorption cannot be measured outside the first-letter task without same-model probes. The intra-RAVEL coherence claim (rho=0.924 from R3) is retained as suggestive evidence, but the R4 shuffled null also shows rho=1.0 on n=3, meaning the coherence is consistent with measurement artifact on small samples.

**Three-subtype taxonomy**: The early-absorption dominance finding (72-75% early-type) is the paper's most actionable insight. If validated on more configurations, it directly reframes the SAE absorption mitigation literature toward dictionary coverage architectures. In its current form (two configurations, n=16 and n=65), it is preliminary but suggestive.

**ITAC**: Not a novel contribution in its current form. The near-null result (3.14% mean FN reduction, 1/10 targets improved, 4/10 targets unable to identify parent candidates) should be the headline, not '3.14% mean FN reduction achieved.'

---

## Dimension 2: Technical Soundness — Score: 5/10

**Theorem 1 remains a proof sketch.** The critical derivation step — showing that absorption at a partial minimum produces a gradient perturbation proportional to delta * ||d_c|| * sin(theta_{jc}) — is asserted, not derived. Delta-absorption is not formally defined before the theorem. The bound formula appears without algebraic derivation. A reviewer with optimization theory background will reject this.

**EDA ordering prediction fails in the data.** Cross-validating against phase2a_taxonomy.json: eda_ordering_holds = FALSE for both L12-16k and L12-65k at threshold 0.3. Partial-type latents have LOWER median EDA than early-type latents at both configurations (L12-65k: partial = 0.652, early = 0.668). The paper claims 'EDA ordering (late > partial > early) holds at L12-65k (KW p=0.0002)' but the KW test only establishes late > early; the full three-way ordering fails. Table 3 describes early absorption as having 'Low (decoder-absent)' EDA — this is contradicted by the actual data where early-type has higher median EDA than partial-type.

**D-EDA/EDA equivalence tension.** R4 direct validation shows delta AUROC = 0.0 between EDA and dec_cosine_negated on both GPT-2 configurations, confirming EDA = 1 - dec_cos as a mathematical identity. However, R3 Phase 5 shows D-EDA AUROC = 0.762 at GPT2-L10 versus EDA AUROC = 0.336 — a result that requires D-EDA to be a different calculation from EDA. The paper does not reconcile this inconsistency.

---

## Dimension 3: Experimental Rigor — Score: 5/10

**Critical cross-validation findings from raw experimental files:**

| Claim in Paper | Source File | Status |
|---|---|---|
| AUROC = 0.776 at L12-16k (proxy labels) | phase1_eda_deda_validation.json | VERIFIED |
| AUROC = 0.629 at GPT2-L6 (exact labels, n_pos=67) | phase5_gpt2_replication.json | VERIFIED (R3 full mode) |
| AUROC = 0.336 at GPT2-L10 (EDA reversed) | phase5_gpt2_replication.json | VERIFIED |
| D-EDA AUROC = 0.762 at GPT2-L10 | phase5_gpt2_replication.json | VERIFIED (R3 full mode) |
| Taxonomy L12-16k 75%/12.5%/12.5% | phase2a_taxonomy.json | VERIFIED |
| Taxonomy L12-65k 72.3%/13.8%/13.8% | phase2a_taxonomy.json | VERIFIED |
| KW p=0.0002 at L12-65k | phase2a_taxonomy.json | VERIFIED (L12-65k only) |
| KW p at L12-16k | phase2a_taxonomy.json | VERIFIED: p=0.237, NOT significant — not reported in main text |
| EDA ordering holds | phase2a_taxonomy.json | CONTRADICTED: eda_ordering_holds=FALSE for both configs |
| Partial < early in median EDA at L12-65k | phase2a_taxonomy.json | VERIFIED but contradicts Table 3 description |
| ITAC 3.14% mean FN reduction | phase2b_itac.json | VERIFIED (dominated by zeros) |
| ITAC 4/10 targets have n_parent_candidates=0 | phase2b_itac.json | VERIFIED — not reported in main text |
| ITAC null test on early-type | phase2b_itac.json | CAVEAT: ITAC never applied to early-type — not a null test |
| H3 falsified, 0/9 combinations pass | r4b_shuffled_control.json | VERIFIED |
| Intra-RAVEL rho=0.924 | From R3 phase3e_crossdomain_analysis | R4 PILOT on n=3 shows shuffled null rho=1.0 — R3 value may be artifact |
| Scaling partial rho=0.368 | phase4_scaling_analysis.json | VERIFIED but L0 variation insufficient (range 65-72) |

**Go/No-Go gate performance**: The pre-registered gate was AUROC >= 0.65 on at least 3/6 Gemma configurations. Result: 2/6 — a NO_GO by the paper's own criterion. The paper does not explicitly state this gate failure.

---

## Dimension 4: Reproducibility — Score: 4/10

1. Gemma 2 2B remains HuggingFace-gated. All Gemma Scope AUROC values use Neuronpedia proxy labels. Reproduction requires HuggingFace ToS compliance and access approval.

2. Figures 5, 6, and 7 are referenced in the text without confirmed PDF existence.

3. SynthSAEBench citation (synthasbench2026 / arXiv:2602.14687) requires verification.

4. ITAC evaluated on synthetic activations from decoder columns, not real model activations.

5. DeLong test citation missing at first use (Section 4.1).

6. The R3/R4 relationship for GPT-2 results is not clearly disclosed: the paper uses R3 Phase 5 full-mode values (n_pos=67, n_bootstrap=10000) for GPT2-L6, while R4 used a different pilot subset (n_pos=18, n_bootstrap=1000).

---

## Verdict and Path Forward

**Score: 5.5/10**
**Verdict: CONTINUE**

The paper has genuine intellectual content worth pursuing. The three viable contributions — EDA as a regime-specific indicator, cross-domain absorption null result (H3 falsified), and early-absorption dominance from the taxonomy — are scientifically meaningful. The paper is well-written and honest about its limitations. The critical blocking actions:

1. **Obtain Gemma 2 2B access** (~1 day admin + 3-5 GPU hours): Re-run all 6 Gemma Scope configurations with exact Chanin et al. labels. If 3+/6 pass AUROC >= 0.65, the EDA contribution becomes a defensible regime-specific detector. This is the single highest-leverage action.

2. **Expand taxonomy to all 6 Gemma + 2 GPT-2 configurations** (~2-3 GPU hours): If early-dominance replicates at 70%+ across 6+ configurations, this becomes the paper's central validated finding rather than preliminary evidence.

3. **Provide a formal proof of Theorem 1** or explicitly downgrade to a heuristic claim. This is the primary novelty risk.

4. **Resolve the D-EDA/EDA equivalence tension**: Either implement D-EDA as the genuine LASSO decomposition with separate AUROC computation, or drop D-EDA from Table 1 and rename Section 3.3.

5. **Correct the EDA ordering claim**: The full late > partial > early ordering fails in the data. Acknowledge what the data actually shows (late > early holds; partial > early does not hold).

A paper with fixes (1)-(3) and the EDA ordering correction would plausibly score 7.0-7.5 — a defensible contribution to SAE interpretability research at the level of a top-venue workshop or mid-tier venue main track.

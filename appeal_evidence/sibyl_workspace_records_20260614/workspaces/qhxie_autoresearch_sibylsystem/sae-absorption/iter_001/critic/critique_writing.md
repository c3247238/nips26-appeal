# Critique: Writing and Presentation (Post-R4 Update)

**Updated:** 2026-04-13 | **Round:** Post-R4 (after H3 shuffled control falsification)

## Summary Assessment

The paper's writing is substantially improved since R3: the cross-domain section (Section 5) now correctly reports the shuffled-control null result and H3 falsification, the abstract accurately reflects the two-contribution structure, and negative-result reporting throughout is honest and specific. However, six specific writing problems remain — two are factual errors that will be immediately flagged by reviewers.

---

## Factual Errors (Must Fix Before Any Submission)

### Error 1: EDA Ordering Claim is Incorrect (Section 6.2 and Abstract)

**Current text (Section 6.2):** "EDA ordering (late > partial > early) holds at L12-65k (KW p = 0.0002). Threshold stability: the EDA ordering late > early holds at all 5 tested thresholds (0.20, 0.25, 0.30, 0.35, 0.40) for both L12-16k (5/5) and L12-65k (5/5). The finding is robust."

**What the data shows (phase2a_taxonomy.json):** At L12-65k threshold 0.3, late median=0.723, early median=0.668, partial median=0.652. The eda_ordering_holds field is `false` for both configurations at all tested thresholds. The full ordering late > partial > early is VIOLATED because partial < early. Late > early holds (which the paper correctly notes), but the paper implies the full three-way ordering holds, which it does not.

**Fix:** Replace with: "The ordering late > early holds at all 5 tested thresholds for both configurations (threshold-stable). The predicted full ordering late > partial > early fails: partial-type latents have lower median EDA than early-type latents at both configurations and all thresholds. This suggests the decoder-coverage threshold (tau) that separates early from partial latents does not correspond to EDA ordering — partial-type latents, despite having some decoder coverage of the parent concept, exhibit encoder-decoder alignment similar to or below early-type latents. The partial category may involve early-stage coverage (near-tau decoder directions) rather than encoder suppression."

---

### Error 2: Abstract DeLong Comparison is Ambiguous (Abstract)

**Current text (Abstract):** "EDA achieves AUROC = 0.776 on Gemma Scope L12-16k (Gemma 2 2B) and AUROC = 0.629 on GPT-2 Small L6 with exact Chanin et al. labels, outperforming the decoder cosine baseline by +0.396 AUROC."

**The problem:** The +0.396 is the DeLong difference at L5-16k (EDA=0.698, decoder cosine=0.302), NOT at L12-16k. At L12-16k, the DeLong difference is +0.553 (EDA=0.776, decoder cosine=0.224). The abstract places "+0.396 AUROC" after "L12-16k" making readers infer L12-16k is the configuration for the +0.396 claim. This is confirmed in Section 4.2: "DeLong test: EDA vs. decoder cosine baseline at L5-16k: difference = +0.396, p ≈ 0; at L12-16k: difference = +0.553, p ≈ 0."

**Fix:** "EDA achieves AUROC = 0.776 on Gemma Scope L12-16k [0.700, 0.863], outperforming the decoder cosine baseline by +0.553 AUROC at this configuration (DeLong p ≈ 0), and AUROC = 0.629 on GPT-2 Small L6 [0.561, 0.692] with exact Chanin et al. labels."

---

## Critical Writing Issues

### 3. EDA Identity with SAEBench metric Not Disclosed (Introduction / Section 3)

The Introduction states "We introduce EDA, a weight-only absorption screening metric" without disclosing that the encoder-decoder cosine similarity is already reported by SAEBench. The R4 validation confirmed EDA = 1 - dec_cos with r=0.9999. A reviewer familiar with SAEBench will flag this as a novelty problem.

**Fix:** Add to Section 3.1 before the metric formula: "The cosine similarity between encoder row j and decoder column j has been reported in SAEBench (Karvonen et al., 2025) under the label encoder_decoder_cosine_sim. EDA formalizes this existing observation by (1) providing a theoretical lower bound theorem connecting encoder-decoder misalignment to absorption degree, and (2) systematically evaluating the metric as an absorption classifier against ground-truth labels with AUROC measurement across configurations and model families. The metric formula is not novel; its theoretical grounding and empirical characterization are."

---

### 4. Theorem 1 Proof is Circular and Appendix B is Missing

The proof sketch in Section 3.2 assumes the conclusion: "absorption introduces a perturbation: by the delta-absorption definition, the gradient contribution from child-active inputs pushes w_{e,j} away from d_j by an amount proportional to delta * ||d_c|| * sin(theta_{jc})." This is what the proof should derive, not assume. Additionally, Appendix B is referenced twice (Sections 3.3 and 7.1) but absent from paper.md.

**Fix:** Either (a) write Appendix B with a proper derivation starting from the SDL loss stationarity condition, formally defining delta as the gradient bias magnitude from child-active inputs, and deriving the bound algebraically; or (b) downgrade Theorem 1 to Proposition 1 (Intuition) and add a note that the full proof requires detailed landscape analysis. Write Appendix B with the D-EDA conditioning analysis (the ill-conditioning argument when d_SAE >> d_model) at minimum.

---

### 5. "Reframes the Absorption Problem" is Overclaiming (Section 6.3, Conclusion)

The paper states in Section 6.3 and the Conclusion: "This reframes the absorption problem: the dominant failure mode is a training-time dictionary coverage gap." This language implies a field-level conclusion, but the result rests on n=16 (non-significant) and n=65 latents in two SAE configurations (both L12), with the finding being threshold-sensitive (at tau=0.2, early is 32%, not dominant). The two-configuration result cannot "reframe" a field.

**Fix:** Replace "reframes the problem" with "provides preliminary evidence for a training-time coverage explanation" and qualify: "If the early-type dominance (72-75%) at the canonical tau=0.3 threshold replicates across SAE configurations and architectures, it would redirect attention toward dictionary-coverage solutions. The current finding from two configurations is suggestive rather than definitive."

---

### 6. ITAC Null Test Description is Misleading (Table 4, Section 6.4)

Table 4 includes "Early/Partial — all | — | — | 0.00% | — | NULL" and the text says this "validates subtype selectivity." But ITAC was never applied to early-type latents — the 0.00% FN reduction reflects that no correction was attempted. Saying ITAC "operates exactly where the taxonomy predicts (late-type latents) and has no effect where it should not" is misleading since ITAC was simply not run on early-type latents.

**Fix:** Remove the NULL row from Table 4 or relabel: "Early/Partial: ITAC not applied (early-type: no parent decoder direction by construction; parent candidates = 0 for all early-type latents tested). Zero applicable targets confirms taxonomy prediction." If ITAC is actually applied to early-type latents and finds no parent candidates, that IS the null test. This should be run and reported.

---

## Major Writing Issues

### 7. Abstract Length and Word Count

The abstract is 247 words vs. the NeurIPS 200-word target. The ITAC detail ("training-free proof-of-concept targeting late-type latents, achieves 3.14% mean false-negative reduction") can be compressed to one clause since ITAC is a negative result. The code release sentence can move to the Conclusion.

**Fix:** Remove: "We additionally introduce Inference-Time Absorption Correction (ITAC), a training-free proof-of-concept for late-type latents. ITAC achieves 3.14% mean false-negative reduction on L12-65k (best individual case: 18.9% for latent j = 61217) without degrading reconstruction quality (FVU change: −4.23%). ITAC is structurally limited to the ~13% late-absorption minority, confirming the taxonomy's prediction." Replace with one sentence: "An attempted inference-time correction (ITAC) achieves only 3.14% mean FN reduction, revealing that late-type encoder suppression exceeds what linear projection can recover." Save ~40 words.

---

### 8. Section 5.3 Needs Clarifying Sentence (Coherence vs. Shuffled Control Sample Discrepancy)

Section 5.3 presents intra-RAVEL rho=0.924 from n=6 SAE configurations as "suggestive existence evidence." The shuffled control was run on n=3 configurations. A reviewer noticing both Sample sizes may question whether the coherence would survive if the shuffled control were run on all 6 configurations. The R4 pilot with n=3 gives rho=0.667 (per r4_writing_gate.json), lower than 0.924.

**Fix:** Add at the opening of Section 5.3: "The intra-RAVEL coherence reported below derives from the full R3 cross-domain experiment (n=6 SAE configurations), while the shuffled hierarchy control in Section 5.2 was run on a pilot subset of n=3 configurations. The coherence rho=0.924 is therefore not directly tested by the shuffled control, which targets absolute absorption rate validity rather than rank coherence across SAE configurations. A pilot coherence estimate on the 3-configuration subset gives rho=0.667, suggesting the full-experiment rho=0.924 may be sensitive to configuration selection."

---

### 9. Section 7.1 "56% / 20x" Figures Are Unsourced

"Focusing on the top 5% of latents by EDA contains approximately 56% of absorbed latents (Prec@50 = 0.0035), reducing the supervised evaluation budget by 20×." These figures are not derivable from reported data. Top 5% of L12-16k = 819 candidates; 819 × 0.0035 ≈ 2.87 absorbed; 2.87/16 ≈ 18%, not 56%.

**Fix:** Replace with the correct derivation: "At L12-16k, Prec@50 = 0.0035 — absorbed latents are 3.5× enriched in the top-50 EDA candidates versus random selection. Prioritizing the top 5% of the dictionary (819 latents in a 16k SAE) for supervised inspection via the Chanin et al. metric reduces the evaluation surface by 20× while maintaining meaningful enrichment of absorbed latents in the inspection set."

---

### 10. Missing Citations

- DeLong test: first used in Section 4.1 without citation. **Fix:** Add "DeLong & DeLong (1988)" at first use.
- SynthSAEBench citation 'synthasbench2026' is a placeholder. **Fix:** If internally constructed, remove citation and write: "a synthetic benchmark constructed following the absorption injection procedure described in Appendix A." If an external paper, verify arXiv:2602.14687 exists and replace placeholder with correct citation format.

---

### 11. Minor Notation Issues

- Spearman rho notation: used as $\rho$ throughout but notation.md specifies $\rho_s$. Affects abstract, Section 5.3, Section 7.2.
- LASSO coefficient $\mu$ in Section 3.3 is not in notation.md.
- Section 5.1 uses "SAE d_in = 2304" but notation.md uses d_model for model dimension.
- Table 4 "WEAK" and "NULL" pass/fail labels are non-standard; add footnote defining these.
- Section 4.4 polysemantic AUROC CI derived from n_pos=3 should be footnoted as unreliable.

---

## What Works Well

1. **H3 null result handling** (Section 5): The shuffled control introduction and H3 falsification are reported clearly and mechanistically. The "This hypothesis (H3) was falsified" sentence is direct. Section 5.3's conditional framing of intra-RAVEL coherence is appropriately hedged.

2. **Section 4.3 (pilot-to-full discrepancy)**: Honest, detailed explanation of the L12-65k collapse with two structural factors. This is model scientific reporting.

3. **Section 7.3 (negative results)**: Three pre-registered hypotheses falsified with specific targets (20% FN reduction, negative partial rho, full ordering) reported honestly. The mechanistic explanation connecting each negative result to early-absorption dominance is clear and non-circular.

4. **Table 1**: Comprehensive with EDA, D-EDA, decoder cosine, shuffled null, Cohen's d, and pass/fail columns. The shuffled null column is a good methodological safeguard.

5. **Abstract first sentence**: "Feature-based causal analyses built on Sparse Autoencoders silently produce incorrect conclusions when parent latents are suppressed by child latents" — specific, stakes-clear, no filler.

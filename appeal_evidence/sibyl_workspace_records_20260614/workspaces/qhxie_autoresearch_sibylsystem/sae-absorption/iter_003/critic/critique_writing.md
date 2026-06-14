# Writing Critique (Iter 003 — Updated by Critic)

## Overall Assessment

The paper is technically well-written. Claims lead with numbers, hedging is quantified, the hook-confound caveat appears proactively throughout, and the pre-committed falsification criterion is a genuine methodological strength. However, several critical writing problems undermine the paper's credibility: (1) the decisiveness language ("decisively falsifies," "unambiguous," "decisive") is inconsistent with the experiment's severe power limitations (n=90 observations at 97.8% baseline); (2) the cross-architecture AUROC comparison presents proxy-label results as absorption replication when E1 shows 0% actual absorption in TopK-32k; (3) all 6 required figures are missing; (4) the weak vs. strong amortization gap hypothesis distinction is absent, causing the practical recommendation to be potentially misleading. These are submission blockers.

## Critical Issues

### 1. Language Mismatch Between Claimed Certainty and Statistical Power

The paper uses strong decisiveness language throughout:
- "OMP achieves 0% absorption reduction...decisively falsifying the amortization gap hypothesis" (Abstract)
- "The result is unambiguous" (Section 4.3)
- "This decisively falsifies the amortization gap hypothesis" (Section 4.3)
- "Our result is decisive" (Introduction)

These claims are not supported by the statistical evidence presented. The experiment has n=30 tokens per letter, 3 letters, baseline AR=0.978 — approximately zero power to detect differences less than ±20% AR. A null result under zero-power conditions is not "decisive." A reviewer with statistical knowledge will reject this framing.

**Required fixes:**
- Replace "decisively falsifies" with "is consistent with falsifying" or "provides evidence against"
- Replace "The result is unambiguous" with a confidence-interval statement: "The observed absorption rates under feedforward (0.978) and OMP (0.978) are indistinguishable within measurement precision [95% bootstrap CI on the difference: X, X]"
- Add a power analysis to the limitations section or a footnote

### 2. All 6 Figures Are Missing

The paper.md references:
- Figure 1 (teaser): `fig_teaser.pdf`
- Figure 2 (enc-norm-hist): `fig_enc_norm_hist.pdf`
- Figure 3 (omp-design): `fig_omp_design.pdf`
- Figure 4 (roc-curves): `fig_roc_curves.pdf`
- Figure 5 (ablation): `fig_ablation.pdf`
- Figure 6 (omp-results): `fig_omp_results.pdf`

None of these files exist in `writing/latex/figures/`. The existing PDFs (`fig1_eda_method.pdf` through `fig5_phase_stability.pdf`) are from an earlier iteration and do not match the current paper's content. The paper would not compile correctly with figure references pointing to non-existent PDFs.

**Required fix:** Generate all 6 figures from the experiment JSON result files. This is a pre-submission blocker.

### 3. Table 1 Mixes Incomparable AUROC Values Without Sufficient Warning

Table 1 reports EncNorm AUROC=0.757 (Standard/L1, n_pos=18, gold IG labels) and EncNorm AUROC=0.837 (TopK-32k, n_pos=77, proxy labels). These are in the same table with the same column header "AUROC." The footnote explains the hook confound but does not clarify that the n_pos=77 proxy labels are not absorption labels at all (the IG pipeline shows 0% absorption in TopK-32k). A reader skimming the table will interpret both rows as reporting absorption detection performance, which is incorrect for the TopK-32k row.

**Required fix:** Add a second footnote to Table 1 clarifying that the TopK-32k row uses decoder-alignment proxy labels (cosine ≥ 0.30 to letter probes), not absorption-based labels, and that the 0% absorption rate under IG measurement means the AUROC measures letter-probe decoder alignment, not absorption detection.

## Major Issues

### 4. Section 4.1 Duplicates Section 3.4 Verbatim

Section 3.4 ("Experimental Setup") and Section 4.1 ("Experimental Setup Summary") are nearly identical. This wastes paper space and annoys reviewers. The review.md correctly identifies this as the paper's most significant writing problem. Section 4.1 should be deleted; Section 4.2 should begin with a single-sentence pointer to Section 3.4.

### 5. Jaccard Definition Inconsistency

Section 3.2 uses the condition `f_k > f_j` (any feature with higher frequency than j), while notation.md and the B2 results notes use `f_i > 3*f_j` (3x higher frequency). The actual experiment used `f_i > 3*f_j` (as confirmed by B2 notes: "A_cooccur values are bounded at ~0.33 (= f_j/f_i < 1/3 by construction when f_i > 3*f_j)"). The paper body should be corrected to match the actual experimental definition.

### 6. Introduction Parent/Child Terminology Confusion

The introduction states "a child SAE latent fails to activate on inputs where it should be active, because a more specific child latent absorbs the activation budget." Two things are called "child" in the same sentence. The paper means: "a **parent** SAE latent (the letter-level feature) fails to activate... because a more specific **child** latent (the specific-word feature) absorbs the activation budget." The review.md correctly identifies this as a terminology error.

### 7. "Absorption Rate Ratio" Defined Inconsistently

The paper uses "absorption rate ratio = 1.000" (meaning OMP/FF = 1.000, no reduction) in Table 2 and the abstract. But in Section 3.3, the ratio is defined as AbsRate_OMP / AbsRate_FF (correct for the Table 2 interpretation). The notation.md defines r_OMP as (AR_FF - AR_OMP) / AR_FF (a reduction ratio, which would be 0.000 for no reduction). The paper toggles between these formulations. The "Reduction" column in Table 2 (showing 0.0%) is clear and unambiguous — the ratio column is redundant and confusing.

**Fix:** Standardize on reduction ratio = (AR_FF - AR_OMP) / AR_FF throughout. Remove the "Ratio OMP/FF" column from Table 2 and replace with a single "AR Reduction (%)" column.

### 8. Contribution 3 Is Labeled "Practical" But Contains Methodological Caveats

The three contributions in the paper are labeled: (1) Mechanistic, (2) Methodological, (3) Practical. Contribution 3 describes the width recovery analysis (67% recovery with hook confound caveat). The hook confound is mentioned in the parenthetical: "subject to hook-confound caveat, Section~\ref{sec:discussion}." This is technically accurate but the caveat is so significant (it may invalidate the 67% figure entirely) that labeling this result "Practical" overstates its current reliability.

### 9. Discussion Section Contains Mild Hedging on the Primary Result

Section 5.1 contains: "This is most naturally explained by the sparsity landscape account." The word "most naturally" hedges a result the paper has just claimed to be decisive. If the OMP null result definitively supports the sparsity landscape account, say so directly. If it only "most naturally" supports it, the decisiveness framing in the rest of the paper is inconsistent.

## Minor Issues

### 10. "≤ 30 tokens per letter" vs. Actual 30 tokens

Section 3.3 states "≤ 30 tokens per letter" but the actual C2 data shows exactly 30 for all letters. Change to "30 tokens per letter."

### 11. Notation Table Specifies 200k Tokens for Co-occurrence, Paper Says 10k

The notation.md defines T=200,000 for co-occurrence graph but the B1 and B2 experiments used T=10,000. The notation table should be corrected to 10,000, and a note should explain why the smaller corpus was used.

### 12. "One-sided p-value" Not Specified for DeLong Test

The DeLong test reports p_one_sided = 0.001160 in the source data. The paper reports p = 0.0012 in the abstract and Section 4.2 without specifying one-sided. This should be written as "one-sided p = 0.0012 (H_A: EncNorm > EDA)" to be precise.

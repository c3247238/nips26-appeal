# Writing Quality Review

## Summary

This paper presents a controlled experiment falsifying the amortization gap hypothesis as a dominant cause of SAE feature absorption, using an OMP oracle with a fixed decoder at matched sparsity (K=53). It simultaneously introduces encoder weight norm as a weight-only absorption detector (AUROC=0.757 on Standard/L1, 0.837 on TopK-32k) and documents a Jaccard co-occurrence signal as an independent complement. The paper is structurally coherent, internally consistent, and consistently self-corrects for its key confound (resid_pre vs. resid_post hook). Terminology is rigorously defined and stable. The primary weaknesses are: (1) a figures section that lists planned figures but delivers no embedded figures in the paper body, preventing reviewers from evaluating visual communication; and (2) a minor number-rounding inconsistency in the abstract. The core writing is technically precise and avoids most banned patterns.

---

## Detailed Assessment

### Structural Coherence: 8/10

The narrative arc is well-designed: problem statement → competing theories → controlled experiment → supporting detection result → practical implications. Each section flows logically into the next, and the introduction accurately summarizes all reported results.

Minor structural issues:
- The paper is organized as 1. Introduction, 2. Background, 3. Methods, 4. Experiments, 5. Discussion, 6. Conclusion. But the outline specified a "Related Work" section (Section 7 in the outline), which was merged into Background (Section 2). This is fine for a workshop format but the section heading says "Background and Related Work" — a reviewer would still expect the related work to cover all relevant citations. The section as written is appropriate.
- Section 4.1 ("Experimental Setup Summary") is an almost verbatim repetition of Section 3.4 ("Experimental Setup"). The information in 4.1 adds nothing to what was presented in Section 3.4. This redundancy should be removed or the two sections merged.
- The transition between Section 3.3 (OMP experiment design) and Section 4 (Experiments) is weak. Section 3.3 describes the experiment in detail, and Section 4.4 then re-presents the result. A reviewer may find the structure slightly redundant: the methods section sets up the design but the reader must jump to Section 4.4 to find the result.
- The introduction mentions "Figure~\ref{fig:teaser}" and the methods mention multiple other figures, but the paper.md body does not contain the actual figures — only a "Figures and Tables" metadata section at the end. This is a LaTeX compilation artifact (figures are compiled in the .tex file), but a plain-text reviewer of paper.md cannot evaluate whether figures are visually effective.

### Notation & Terminology Consistency: 9/10

Notation is well-controlled. All key symbols are defined in the notation table and used consistently.

Specific issues found:
1. **Jaccard definition inconsistency**: In Section 3.2 (paper), the Jaccard co-occurrence score condition is `f_k > f_j` (strictly higher activation frequency than `j`). In the notation.md file, the definition shows `max_{i: f_i > 3f_j}`. The paper body uses the looser condition `f_k > f_j` while the notation table specifies the stricter `f_i > 3f_j`. This is a substantive discrepancy that must be resolved — the experimental results should clarify which definition was used in practice.

2. **Parent/child labeling inconsistency**: The paper text (Introduction, Section 2.1) describes the absorbed feature as the "child" (`child latent absorbs the activation budget`), but the glossary.md explicitly clarifies that in the first-letter task, the **absorbed latent is the parent** and the **absorbing latent is the child** with higher frequency. The paper's use of "parent concept" and "child latent" in Section 2.1 is technically correct (parent concept fails to fire, child fires instead), but the notation.md uses `i` for parent (higher frequency) and `j` for child (absorbed). Confusion arises because "child latent" in colloquial usage maps to the narrower latent that fires more — but in the paper's actual task, the absorbed latent (parent concept for the letter) fires less. The glossary's CAUTION note addresses this, but the introduction and Section 2.1 do not. A reviewer reading quickly will be confused.

3. **"EncNorm" vs. "encoder weight norm"**: The glossary prefers "encoder weight norm" on first use, then "encoder norm" for brevity. The paper uses "encoder weight norm ($\|\mathbf{w}_{\text{enc},j}\|_2$)" on first use in Section 3.1, which is correct. However, the abstract uses "encoder weight norm" without introducing the shorter form — this is acceptable.

4. Minor: The abstract reports `n_\text{pos} = 77` for the TopK-32k proxy labels. The experimental data (A3 results) shows `n_pos = 77` for TopK-32k, which matches. Consistent. OK.

5. Minor: Section 4.1 claims "OpenWebText corpus, 10k tokens for co-occurrence." The notation table states `T = 10,000` for absorption experiments and `200,000` for co-occurrence. The paper should either confirm that 10k was used throughout (which is what the B2 results file confirms, `n_tokens_for_cooccurrence: 10000`), or correct the notation table. The discrepancy between the notation table (200k) and experimental reality (10k) should be corrected.

### Claim-Evidence Integrity: 9/10

All primary claims are backed by specific numbers and cross-verified against source data.

Specific verifications:
- AUROC = 0.757 [0.665, 0.849]: Source A3 reports 0.7565988 with CI [0.6654, 0.8492]. The paper rounds to 0.757. CI reported as [0.665, 0.849] — consistent with rounding.
- AUROC = 0.837 [0.807, 0.870]: Source A3 reports 0.8369921 with CI [0.8072, 0.8698]. Consistent.
- DeLong z = 3.05, p = 0.0012: Source A3 reports z = 3.0457, source B2 reports z = 3.0457, p = 0.001160. The paper's p = 0.0012 matches (rounded). Consistent.
- OMP AR = FF AR = 0.978: Source C2 confirms both conditions show mean_absorption_rate = 0.9777. Consistent.
- F1 width recovery: Source confirms n_recovered = 12 (67%), n_not_recovered = 6 (33%), mean_best_cosine = 0.7909 (paper rounds to 0.791). Consistent.
- Cohen's d = 0.971 (Standard): Source A3 reports 0.9711. Consistent.
- Cohen's d = 1.235 (TopK-32k): Source A3 reports 1.2349. Consistent.
- O_jaccard AUROC = 0.721: Source B2 reports 0.7210673. Consistent.
- O_jaccard AUPRC = 0.075: Source B2 reports 0.07475. Consistent.
- Precision@50 = 0.100: Source B2 reports 0.10. Consistent.
- Spearman rho(EncNorm, O_jaccard) = 0.044: Source B2 reports 0.0439507. Consistent.

**One discrepancy found**: The abstract reports the DeLong test: "AUROC = 0.650; DeLong z = 3.05, p = 0.0012". Looking at source data: the DeLong test in B2 uses a ONE-SIDED p-value of 0.001160 (encoder_norm > EDA). The paper reports p = 0.0012 in the abstract and p = 0.0012 in Section 4.2. However, the DeLong test direction (z positive = enc_norm > EDA) is stated correctly. The p-value interpretation is correct (enc_norm is significantly better), but the paper should note "one-sided p-value" to be precise.

**One minor number discrepancy**: The abstract states "absorption rate ratio = 1.000" and Table 2 shows ratio = 1.000 for all three letters. Source C2 JSON shows `omp_reduction_ratio: 0.0` for all letters, confirming ratio = 1.000 (i.e., OMP/FF = 1.000). Consistent.

**One data note**: The C2 experiment used 30 tokens per letter (not the "≤ 30" stated in Section 3.3). Source C2 shows `n_tested: 30` for all three letters. The paper should change "≤ 30 tokens per letter" to "30 tokens per letter" to match the actual experimental data.

### Visual Communication: 4/10

This is the paper's most significant weakness in the paper.md form.

The outline specifies 6 figures:
- Figure 1 (teaser): feedforward vs. OMP absorption rates + ROC curves
- Figure 2 (enc-norm-hist): histogram of encoder norm, absorbed vs. non-absorbed
- Figure 3 (omp-design): architecture diagram of the controlled experiment
- Figure 4 (roc-curves): ROC curves for all detectors, two-panel
- Figure 5 (ablation): horizontal bar chart of AUROC per method
- Figure 6 (omp-results): per-letter absorption rates under FF vs. OMP

The paper.md body references all of these with LaTeX `\ref{}` calls (Figure~\ref{fig:teaser}, Figure~\ref{fig:enc-norm-hist}, etc.), and the "Figures and Tables" section at the end lists data sources. However:

1. **No figures are embedded in the paper.md itself.** A reader of paper.md sees cross-references but cannot evaluate figure clarity, caption quality, or whether figures are self-explanatory.
2. The LaTeX build directory does contain PDFs: `fig1_eda_method.pdf`, `fig2_eda_distributions.pdf`, `fig3_eda_scaling.pdf`, `fig4_enc_dec_alignment.pdf`, `fig5_phase_stability.pdf`. These file names do NOT match the figures described in the paper (which should be fig_teaser, fig_enc_norm_hist, fig_omp_design, fig_roc_curves, fig_ablation, fig_omp_results). The actual figures in the LaTeX directory appear to be from an older iteration (EDA figures, not the current encoder-norm/OMP figures).
3. Tables 1–4 are embedded as inline markdown tables, which are readable and well-formatted.
4. For what can be assessed from table structure: Table 1 is clear and well-labeled; Table 2 format is effective (shows identical FF/OMP values explicitly); Table 3 is concise; Table 4 (implications matrix) is a useful practitioner summary.

The mismatch between figure filenames in LaTeX and figure descriptions in the paper is a major concern. Either the figures have not been generated yet, or they were generated under old names and not updated. This must be resolved before submission.

### Writing Quality: 8/10

The writing is technical and precise. No headline banned patterns found ("In recent years", "It is worth noting that", "Furthermore").

Positive characteristics:
- Claims lead with numbers: "OMP achieves 0% absorption reduction across all tested features" — concrete before explanation.
- Hedging is quantified: "67% of absorbed features have direction-aligned counterparts... but 33% do not" — specific fractions rather than vague qualifiers.
- Limitations are proactively disclosed: the hook confound is mentioned in the abstract, methods, results, discussion, and conclusion — appropriate for a result that affects a key comparison.

Issues found:

1. **Paragraph-level redundancy**: Section 4.1 ("Experimental Setup Summary") repeats virtually all of Section 3.4 ("Experimental Setup"). This is the most significant writing problem: one of these sections should be removed. Keeping Section 3.4 and removing 4.1 is recommended; the section-opening in 4.2 can briefly remind readers of the experimental setup without a full repetition.

2. **Abstract precision issue**: "OMP achieves 0% absorption reduction across all tested features (mean AR_OMP = AR_FF = 0.978; absorption rate ratio = 1.000)". The ratio 1.000 means OMP/FF = 1.000, i.e., zero reduction. But "absorption rate ratio" is defined inconsistently: in Section 3.3 it says ratio = AbsRate_OMP / AbsRate_FF, while in Section 2 the notation.md defines `r_OMP = (AR_FF - AR_OMP) / AR_FF` (reduction ratio = 0.0, not absorption rate ratio = 1.000). The paper uses both "absorption rate ratio = 1.000" and "reduction = 0%" — these are consistent numerically, but the reader must toggle between two framings. Recommend standardizing on one throughout.

3. **Sentence-level complexity**: Several sentences in the Discussion (Section 5.1) are lengthy multi-clause constructions that require re-reading. Example: "The only interventions that can escape this attractor are ones that alter the loss landscape during training: masked regularization (suppressing the competing gradient), hierarchically-aware objectives, or structural changes to the dictionary." — the colon+list structure works, but "attractor" is not defined until this point in the Discussion and was only briefly mentioned in Section 2.2. A one-sentence forward reference would help.

4. **"Resid_pre" and "resid_post" notation mixing**: The paper uses `resid_pre` and `resid_post` interchangeably with `hook_resid_pre` and `hook_resid_post`. Table 1 uses the short form (`resid_pre`), while the text uses the full hook name. This is a minor consistency issue but could confuse readers who check the SAELens documentation.

5. **Contribution 3 labeled as "Methodological" but contains only AUROC results**: In the contributions list (end of introduction), Contribution 2 is labeled "Methodological: Encoder weight norm, a weight-only absorption heuristic." This is accurate. But the contribution parenthetical states "architectures not directly comparable due to hook confound — see Section~\ref{sec:method}" which redirects to the Methods section, but the hook confound warning is also in Section 3.4 (Experimental Setup). The cross-reference is correct but the caveat's location could be clearer.

6. **One mild banned pattern**: Section 5.1 contains "This is most naturally explained by the sparsity landscape account" — while not a canonical banned pattern, "most naturally" is slightly hedging language that undercuts the decisive OMP result. If the OMP result is decisive, the explanation should be stated without qualification hedges.

---

## Issues for the Editor

1. **[Critical] Figure filenames in LaTeX do not match the paper's figure plan**: The `writing/latex/figures/` directory contains `fig1_eda_method.pdf`, `fig2_eda_distributions.pdf`, `fig3_eda_scaling.pdf`, `fig4_enc_dec_alignment.pdf`, `fig5_phase_stability.pdf`. None of these match the figures described in the paper (fig_teaser, fig_enc_norm_hist, fig_omp_design, fig_roc_curves, fig_ablation, fig_omp_results). The paper compiles with figure references that may be broken or linked to wrong figures. **Fix**: Generate the correct figures (6 total) from the experiment result JSON files, name them consistently with the paper's figure labels, and place them in `writing/latex/figures/`.

2. **[Major] Section 4.1 duplicates Section 3.4 verbatim**: The experimental setup is presented in full in Section 3.4 and repeated nearly identically in Section 4.1. This wastes approximately 120 words of a workshop paper's limited budget and will annoy reviewers. **Fix**: Delete Section 4.1 entirely. Replace the first sentence of Section 4.2 with a one-line reminder: "Setup follows Section~\ref{sec:setup}; detection experiments use Standard/L1 (n_pos=18) and TopK-32k (n_pos=77) at GPT-2 L6."

3. **[Major] Jaccard definition inconsistency (paper body vs. notation.md)**: The paper (Section 3.2) defines O_jaccard with condition `f_k > f_j`, while notation.md shows `max_{i: f_i > 3f_j}`. The B2 results notes state `A_cooccur` uses `f_i > 3*f_j` as a bound. **Fix**: Reconcile the definition — verify which threshold was used in the actual experiment and update both the paper and notation.md to match. If `f_k > f_j` was used (as in the paper text), update notation.md. If `f_i > 3f_j` was used (as in notation.md), update Section 3.2.

4. **[Major] "≤ 30 tokens per letter" vs. actual 30 tokens**: Section 3.3 says "≤ 30 tokens per letter" but source C2 JSON shows exactly 30 tokens for all three letters. **Fix**: Change to "30 tokens per letter" to accurately represent the experimental data.

5. **[Minor] Parent/child labeling in Introduction (Section 1, second paragraph)**: The introduction says "a child SAE latent fails to activate... because a more specific child latent absorbs the activation budget." The phrase "a child SAE latent fails to activate because a child latent absorbs" is confusing (two things called "child"). The correct framing is: the **parent** SAE latent fails to activate because a more specific (child) latent absorbs the activation budget. **Fix**: Change "a child SAE latent fails to activate" to "a parent SAE latent fails to activate."

---

## What Works Well

1. **Abstract and introduction calibrate claims exactly to results**: The abstract provides AUROC values, CIs, sample sizes, and DeLong test results — everything needed to assess statistical strength — without overstating. The hook-confound caveat appears in the abstract itself, which is unusually thorough.

2. **Pre-committed falsification criterion is explicitly stated**: Section 3.3 states the falsification criterion before reporting the result ("established in the experimental proposal before running experiments: if OMP achieves ≥ 80% of feedforward absorption rate, H2 is falsified"). This is a methodological strength that few ML papers include and that significantly increases credibility of the OMP result.

3. **Negative results are reported with equal prominence as positive results**: ARS_v2 (AUROC = 0.586, worse than EncNorm) is reported in Section 4.3 with a clear interpretation ("Product formulations dilute both signals"). The entity-type probe failure (H3) is reported in Section 5.2 as a limitation rather than buried. This honest negative-result reporting is appropriate for the venue.

SCORE: 7


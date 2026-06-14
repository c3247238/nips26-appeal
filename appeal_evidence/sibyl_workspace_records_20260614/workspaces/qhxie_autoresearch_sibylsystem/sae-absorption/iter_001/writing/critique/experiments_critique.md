# Critique: Experiments (Sections 4, 5, 6) -- Round 2

## Summary Assessment

The experiments section has matured considerably since the prior critique round. Several earlier critical issues (GPT-2 vocabulary description, Cohen's d footnote, layer pre-registration note, polysemanticity transition) have been addressed. The section now spans three major paper sections (4: EDA Validation, 5: Cross-Domain, 6: Taxonomy + ITAC), each internally coherent and data-driven. The honest reporting of negative results (ITAC's 3.14% vs. 20% target, pilot-to-full collapse, probe quality below gate) remains a genuine strength. However, Table 1 still lacks the baseline AUROC columns promised by the outline, the best-case ITAC number inconsistency between Section 6.4 and the proposal/discussion persists, and the cross-domain Section 5 has a structural weakness: the probe quality limitations are so severe (37.8% accuracy) that the "existence proof" framing needs tighter defense. Section 6's taxonomy and ITAC results are the paper's strongest empirical contribution and are well presented.

## Score: 7/10

**Justification**: Up from 6/10 in the prior round. Several critical and major issues have been fixed (vocabulary description, Cohen's d footnote, transition, layer footnote, gated architecture clarification). Score is held back by: (1) Table 1 still missing baseline columns, (2) the best-case ITAC FN reduction number differs between experiments (18.9%) and discussion (22.7%), (3) Section 5's probe accuracy of 37.8% weakens the cross-domain claim more than the text acknowledges, and (4) no polysemanticity stratification figure despite the result being a headline finding. Reaching 8.5/10 requires completing Table 1 per the outline, resolving the ITAC number, adding a stratification visual, and tightening Section 5's defense of the existence proof claim.

---

## Critical Issues

### Issue 1: Table 1 still lacks baseline AUROC columns from the outline
- **Location**: Section 4.2, Table 1
- **Quote**: Table header: `Config | Layer | d_SAE | AUROC (EDA) | 95% CI | AUROC (D-EDA) | n_pos | Cohen's d | Pass`
- **Problem**: The Figure & Table Plan in `outline.md` specifies: "Columns: Config, Layer, Width, AUROC (EDA), 95% CI, AUROC (D-EDA), **AUROC (Decoder Cosine Baseline)**, Cohen's d, Mann-Whitney p, Pass?". The decoder cosine baseline column is absent from the table. The baseline is referenced in Section 4.2 prose ("AUROC = 0.302 at L5-16k, a +0.396 improvement") but not in the table. Reviewers will want to see EDA vs. baseline in a single row to judge significance per configuration. The shuffled null baseline (mentioned in Section 3.5 as AUROC = 0.565 at L5-16k) is also absent.
- **Fix**: Add at minimum a "Decoder Cos. Baseline" column to Table 1. If values are available only for a subset of configs, use "--" with a footnote. Move the DeLong test reference to a table footnote. This is the single most important structural fix remaining in Section 4.

### Issue 2: The +0.396 AUROC comparison mixes configurations
- **Location**: Section 4.2, paragraph 2
- **Quote**: "The strongest result, L12-16k with AUROC = 0.776, places EDA well above the decoder cosine similarity baseline (AUROC = 0.302 at L5-16k, a +0.396 improvement; DeLong $p \approx 0$)."
- **Problem**: This sentence compares L12-16k EDA AUROC (0.776, "the strongest result") against a baseline at L5-16k (0.302). The +0.396 improvement is a cross-config comparison: it is the difference between EDA and baseline at L5-16k (0.698 - 0.302 = 0.396), not the improvement at L12-16k. The sentence structure makes it read as if L12-16k beats the baseline by 0.396. This is confusing and a careful reviewer will flag it as sloppy at best, misleading at worst.
- **Fix**: Separate the two claims: "The strongest EDA result is L12-16k (AUROC = 0.776). EDA consistently outperforms the decoder cosine similarity baseline; the largest gap is at L5-16k (EDA = 0.698 vs. baseline = 0.302, +0.396 improvement; DeLong $p \approx 0$)."

---

## Major Issues

### Issue 3: Section 5 cross-domain "existence proof" rests on probes with 37.8% accuracy
- **Location**: Section 5.1, "Probes" paragraph
- **Quote**: "Probe accuracy is below the 85% quality gate: city-continent 71.4%, city-country 37.8%, city-language 36.8%."
- **Problem**: A 37.8% accuracy probe for city-country (with ~100 classes, so multi-class chance is ~1%) is far above chance but also far below the pre-specified quality gate. The section correctly states "absolute absorption rates are therefore unreliable" and argues "relative comparisons remain interpretable because the same probe noise affects all SAE configurations equally." This defense is valid for intra-RAVEL coherence (rho = 0.924) but insufficient for the "existence proof" claim. A reviewer could argue: if the probe direction is substantially wrong, the cosine threshold (>= 0.30) that identifies candidate latents may select latents unrelated to the true parent concept. The "above 3x random baseline" finding would then reflect the probe's inaccuracy pattern rather than genuine absorption. The text does not address this specific counterargument.
- **Fix**: Add a sentence explicitly addressing the counterargument: "Even with noisy probes, the cosine threshold selects latents whose decoder columns align with *some* consistent linear direction in residual stream space. The 4.6x--137.5x elevation over random baselines (which use equally arbitrary but unstructured directions) indicates the signal is systematic, though its attribution to the specific parent concept remains provisional until probes meet the quality gate." Also consider reporting the per-hierarchy random baseline absorption rates so readers can compute the absolute difference, not just the ratio.

### Issue 4: D-EDA column in Table 1 has missing entries despite prose stating values
- **Location**: Section 4.2, Table 1
- **Quote**: Table 1 shows "--" for D-EDA at L5-65k, L19-16k, L19-65k. Prose states: "D-EDA does not consistently improve over scalar EDA on Gemma Scope (0.471--0.602 across six configs, all below their corresponding EDA values except L19-16k)."
- **Problem**: The text claims D-EDA was computed for all six Gemma configs and reports the range (0.471--0.602), but the table shows missing entries for three configs. Incomplete tables invite reviewer suspicion of selective reporting.
- **Fix**: Fill all D-EDA entries in Table 1. If individual config values are available from source data, include them. If the "--" entries correspond to configs where D-EDA computation failed, explain this in a table footnote.

### Issue 5: Polysemanticity stratification (Section 4.5) has no dedicated figure
- **Location**: Section 4.5
- **Quote**: "EDA achieves AUROC = 0.922 [0.842, 0.979] on the polysemantic half... versus 0.643 [0.518, 0.763] on the monosemantic half"
- **Problem**: The polysemanticity stratification is a headline finding (AUROC = 0.922 vs. 0.643) that appears in the abstract and introduction. It has no dedicated figure. The outline's Figure & Table Plan assigns Figure 4 to "EDA Distribution Stratified by Polysemanticity" with a two-panel violin plot design. The actual Section 4.3 uses Figure 4 for group separation (absorbed vs. non-absorbed), and the stratification is relegated to text-only in Section 4.5. A result this prominent deserves visual support.
- **Fix**: Either add a Figure 4b panel showing the stratified AUROC or create a supplementary figure. A simple grouped bar chart (2 configs x 2 strata, y-axis = AUROC, error bars = bootstrap CI) would be sufficient and straightforward to generate.

### Issue 6: ITAC best-case FN reduction number inconsistency
- **Location**: Section 6.4 vs. Section 7.3 (Discussion)
- **Quote (Section 6.4)**: "The best individual case (latent $j = 61217$) achieves 18.9% FN reduction with decoder-parent cosine similarity of 0.954"
- **Quote (Section 7.3)**: "the best individual case reaching 22.7% reduction ($j_{\text{idx}} = 61217$)"
- **Problem**: The same latent (j = 61217) is reported as achieving 18.9% FN reduction in the experiments section and 22.7% in the discussion section. The proposal also cites 22.7%. One of these numbers is wrong. Inconsistency in reported data across sections is a critical credibility risk.
- **Fix**: Verify the best-case FN reduction from source data and use the correct number in both sections. Also update the proposal if needed.

### Issue 7: L12-16k ITAC results are uninterpretable as reported
- **Location**: Section 6.4, Table 3
- **Quote**: "L12-16k ... FN Before 0.000 ... FN After 0.000 ... FN Reduction 0.0% ... FVU Change +0.221"
- **Problem**: A late-absorbed latent with FN rate = 0.000 before ITAC means the parent latent was already firing on all parent-positive inputs --- it has no false negatives to correct. Yet the FVU change is +0.221 (a large *degradation* in reconstruction quality). This row raises two questions: (a) Why is ITAC being applied to a latent that has no FN problem? (b) Why does ITAC degrade FVU by 22.1% at L12-16k while improving it by 4.2% at L12-65k? Neither question is addressed in the text. The n=2 late-type latents at L12-16k makes this even more puzzling --- the sample is too small to be meaningful.
- **Fix**: Add a sentence explaining the L12-16k result: why FN = 0 for a late-absorbed latent, and why FVU degrades. If the answer is that the n=2 sample at L12-16k is too small to be informative, say so explicitly and consider reporting only L12-65k ITAC results in the main text, relegating L12-16k to supplementary.

---

## Minor Issues

- **Section 4.1, "Metrics and statistics"**: States "The pass threshold is AUROC >= 0.65" without justification. Why 0.65 and not 0.60 or 0.70? Add a one-sentence rationale: "The 0.65 threshold was pre-registered as the minimum for meaningful screening, following [source]."
- **Section 4.2, paragraph 2**: "EDA cross-validates against SAEBench's precomputed `encoder_decoder_cosine_sim` with Pearson $r > 0.999$" -- this is a computational sanity check, not a cross-validation. Reword to "EDA values match SAEBench's precomputed `encoder_decoder_cosine_sim` (Pearson $r > 0.999$), confirming implementation correctness."
- **Section 4.3**: "low positive-class prevalence ($n_{\text{pos}} = 29$ out of 65,536 latents) makes rank-based AUROC unreliable" is imprecise. AUROC is still a valid statistic; the issue is that with 29 positives the estimate is noisy. Reword: "makes AUROC estimation noisy."
- **Section 4.5**: "Caution: the polysemantic AUROC relies on only 3 positive examples" -- good, this caveat is present. Consider adding: "With n_pos = 3, AUROC is determined by the ranks of 3 observations, making it sensitive to single-example perturbation."
- **Section 5.2**: The absorption-to-random ratios (4.6x to 137.5x) are reported without the actual random baseline rates. Add the random baseline rate (e.g., "random baseline: X% for city-continent") so readers can compute the absolute difference.
- **Section 5.3**: "Bonferroni correction at alpha = 0.05/7 = 0.007" -- the denominator 7 is unexplained. Three intra-RAVEL pairwise comparisons would suggest alpha = 0.05/3. The 7 presumably includes cross-paradigm comparisons, but this is not stated. Clarify the multiple comparison family.
- **Section 5.2**: "EDA correlates positively with RAVEL absorption rates (mean Spearman rho = 0.37 for city-continent, 0.69 for city-country, 0.67 for city-language across SAE configs)" -- these are computed over n=6 configs. Rho = 0.37 with n=6 is nowhere near significant. Report p-values or acknowledge low power.
- **Section 6.1**: "primary threshold tau = 0.3" is introduced without justification. Add: "We set tau = 0.3 following the SAEBench convention for cosine alignment thresholds; robustness is verified across tau in {0.20, ..., 0.40}."
- **Section 6.2**: "suggests that partial and early absorption share a geometric mechanism" -- this is speculative interpretation. Prefix with "one possible explanation:" or "we hypothesize:".
- **Section 6.3**: States "wider dictionaries or hierarchically-aware training objectives" as the appropriate lever. However, H6 (Section 7.3) shows wider SAEs absorb *more* at matched L0. Add a qualifying sentence noting this tension: "Simply increasing width without changes to the training objective does not reduce absorption (Section 7.3); the intervention must include hierarchically-aware loss terms."
- **Section 6.4**: The ITAC formula uses $d_j^\top$ but notation.md defines $d_j$ as a column vector. The transpose is correct for the inner product but inconsistent with the dot-product convention used elsewhere. Standardize notation.
- **Section 6.4**: How is the absorbing child identified for each parent latent? The ITAC formula uses $z_{\text{abs}}$ and $d_{\text{abs}}$ without specifying the child identification method.
- **Terminology**: $n_{\text{pos}}$ in Table 1 is not in notation.md. Either add it to notation.md or define inline.
- **Terminology**: "positive-class prevalence" (Section 4.4) is not in the glossary. Use a defined term or add to glossary.

---

## Visual Element Assessment

- [x] Figure 3 (EDA AUROC heatmap) is referenced before it appears in Section 4.2: "visualized in Figure 3" -- correct
- [x] Figure 4 (EDA distributions) is referenced before it appears in Section 4.3: "Figure 4 shows the statistical group separation" -- correct
- [x] Figure 5 (cross-domain rates) is referenced before it appears in Section 5.2: "As shown in Figure 5" -- correct
- [x] Figure 6 (RAVEL coherence) is referenced before it appears in Section 5.3: "Figure 6 shows pairwise Spearman correlations" -- correct
- [x] Figure 7 (subtype EDA) is referenced before it appears in Section 6.2: "As shown in Figure 7" -- correct
- [ ] **Table 1 is missing baseline AUROC columns** promised by the outline -- must be completed
- [ ] **D-EDA column in Table 1 has missing entries** for configs where values are stated in prose -- must be completed
- [ ] **No figure for polysemanticity stratification** (Section 4.5) despite this being a headline finding -- add figure or supplementary panel
- [x] Table 2 (taxonomy distribution) and Table 3 (ITAC) are referenced before appearance -- correct
- [x] All figure captions are self-explanatory

---

## Cross-Section Consistency

### Checked against Method (Section 3)
- **Consistent**: EDA formula (Equation 2 in method, applied identically in experiments), D-EDA limitation language, Theorem 1 referenced correctly in Section 6.3
- **Consistent**: Decoder cosine baseline AUROC = 0.302 at L5-16k matches between Section 3.5 and Section 4.2
- **Consistent**: The method's D-EDA limitation ("does not outperform scalar EDA on Gemma Scope") matches experiments' D-EDA reporting

### Checked against Discussion (Section 7)
- **Consistent**: Section 7.1 references Table 1 numbers (AUROC = 0.776, 0.698, 0.629) correctly
- **Consistent**: Section 7.3 ITAC discussion ("3.14% mean FN reduction") matches Section 6.4
- **INCONSISTENT**: Section 7.3 states "best individual case reaching 22.7% reduction ($j_{\text{idx}} = 61217$)" but Section 6.4 states "18.9% FN reduction" for the same latent. (Issue 6 above.)
- **Consistent**: Section 7.4 limitations on RAVEL probes match Section 5.1 disclosures

### Checked against Introduction (Section 1)
- **Consistent**: Introduction's "AUROC = 0.776 at L12-16k" matches Table 1; "AUROC = 0.629 at GPT2-L6" matches Table 1
- **Consistent**: Introduction's "~72--75%" early absorption matches Section 6.2's Table 2
- **Consistent**: Introduction's "3.14% mean false-negative reduction" matches Section 6.4
- **Consistent**: Introduction's "+0.396 AUROC" matches experiments (but the experiments section conflates configs in the sentence structure -- Issue 2)

### Checked against notation.md
- **Consistent**: $d_{\text{SAE}}$, $w_{e,j}$, $d_j$, $\text{EDA}(j)$, $v_p$, $\tau$, $\rho_j$ all used per notation table
- **Note**: $n_{\text{pos}}$ used in Table 1 is not in notation.md. Add it or define inline.

### Checked against glossary.md
- **Consistent**: "feature absorption," "parent latent," "child latent," "EDA," "D-EDA," "early absorption," "late absorption," "partial absorption," "ITAC" all match glossary definitions
- **Consistent**: "feature density" in Section 4.5 matches glossary definition
- **Minor**: "positive-class prevalence" in Section 4.4 not in glossary -- add or replace with defined term

---

## What Works Well

1. **Honest negative result reporting throughout**: The pilot-to-full AUROC collapse (Section 4.4), the ITAC falsification of H5 (Section 6.4), and the probe quality limitations (Section 5.1) are reported with no hedging. The Section 4.4 analysis attributing the collapse to two specific mechanisms (subset enrichment and proxy label noise) is rigorous and would earn reviewer respect. The sentence "EDA is a regime-specific screening tool for mid-layer narrow SAEs, not a universal absorption detector" is a model of calibrated framing.

2. **Section 6 taxonomy is the paper's strongest empirical contribution**: The three-subtype partition is well-motivated (Section 6.1), the empirical distribution is clearly presented with appropriate statistical tests (Section 6.2, Table 2), the early-dominance insight is drawn carefully and connected to practical implications (Section 6.3), and ITAC's null test on early-type latents (Section 6.4) provides clean confirmatory evidence for the taxonomy's predictive power. The threshold stability analysis (10/10 tests) is particularly convincing.

3. **Cross-domain coherence analysis (Section 5.3)**: The intra-RAVEL Spearman rho = 0.924 with Bonferroni correction is a well-designed analysis that stands independently of the probe quality issues. The honest treatment of the cross-paradigm negative correlation (rho = -0.43 presented as hypothesis-generating) avoids overclaiming. The two-explanation framework (scale incomparability vs. genuinely distinct regimes) gives reviewers a fair analytical handle.

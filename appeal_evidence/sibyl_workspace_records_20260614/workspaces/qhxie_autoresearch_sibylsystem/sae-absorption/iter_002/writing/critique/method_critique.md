# Critique: Method

## Summary Assessment

The Method section is technically precise and well-organized, with clear mathematical notation, an explicit limitation acknowledgment for Proposition 1, and a genuine mechanistic conjecture labeled appropriately as such. The main weaknesses are (1) a significant gap between the outline plan and the actual content in Section 3.3 (the cross-directional metric description is buried and under-motivated), (2) a logical circularity in the mechanistic argument of Proposition 2 that undermines the conjecture's rigor, and (3) a table-label conflict with the Experiments section (the baseline table in 3.4 refers to a "Table" that conflicts with Table 1 in the Experiments section).

## Score: 7/10

**Justification**: The core theoretical contributions (Prop. 1 proof, EDA definition, mechanistic conjecture) are well-executed with specific numbers and honest limitations. The score is held back by: the cross-directional metric not receiving a full theoretical motivation in Section 3.3 (it is only mentioned at the end, after a long detour into why decoder-decoder cosine fails); a soft circularity in Proposition 2's logic; a minor notation inconsistency; and the baseline table needing a label to distinguish it from Table 1 in the Experiments section. Reaching 8/10 requires: (a) adding a standalone paragraph motivating the cross-directional metric from first principles before the unresolved tension subsection, (b) fixing the mechanistic conjecture's circularity, and (c) resolving the table-label conflict.

---

## Critical Issues

### Issue 1: Mechanistic Conjecture Circularity (Proposition 2)
- **Location**: Section 3.3, "Formal statement" paragraph (lines 128–135)
- **Quote**: "Under conditions... (C2) the child latent fires on parent-only contexts (absorption has occurred)..."
- **Problem**: Condition C2 assumes absorption has already occurred in order to derive why EDA arises from absorption. This is circular: the conjecture is supposed to explain why absorbed features develop high EDA, but it starts by assuming the feature is already firing on parent-only contexts — which is definitionally the absorbed state. The conjecture therefore cannot serve as an independent prediction; it describes what happens given absorption, not why EDA results. A reviewer will flag this immediately. The related_work section already acknowledges the amortization gap as a confounder (O'Neill et al.), and this circularity makes it harder to argue that EDA specifically diagnoses absorption rather than any other cause of encoder drift.
- **Fix**: Rewrite Condition C2 as an empirically testable precondition rather than an assumption of the absorbed state. For example: "Condition C2: on parent-only contexts $x \approx \alpha_p d_p$, the pre-activation $z_c^\text{pre} = e_c^T x - b_c$ exceeds the firing threshold (i.e., the child latent receives positive gradient pressure toward firing)." Then state that C2 is a consequence of absorption that requires empirical verification (pointing to the encoder-probe results in Section 4.2), not a definition. This preserves the conjecture while removing the circularity.

### Issue 2: Cross-Directional Metric Motivation Missing
- **Location**: Section 3.3, "Cross-directional metric" subsection (lines 159–166)
- **Quote**: "We therefore also test the cross-directional cosine: $\cos(\hat{e}_p, d_c)$..."
- **Problem**: The cross-directional metric $\cos(\hat{e}_p, d_c)$ — which turns out to be the *best* detector in the paper (AUROC = 0.730 vs. EDA's 0.650) — receives only 5 lines of motivation, placed after the "Unresolved tension" subsection that admits the EDA magnitude prediction fails. The reader encounters the strongest empirical contribution of the paper as an afterthought buried after an admission of theoretical trouble. Furthermore, the mechanistic argument given ("parent encoder and child decoder both encode information relevant to the parent concept") does not follow directly from the gradient analysis in Proposition 2. Proposition 2 explains why the child encoder moves toward the parent decoder; it does not directly predict that the parent encoder will align with the child decoder. These are geometrically distinct predictions that need separate justification.
- **Fix**: Elevate the cross-directional metric to its own subsection between Proposition 2 and "Why decoder-decoder cosine fails." Provide explicit gradient-based motivation: in the absorbed state, the parent encoder must detect both parent-only contexts and co-occurrence contexts where the child is also present; in co-occurrence contexts, the parent encoder's gradient is influenced by the child concept direction, pulling $e_p$ toward $d_c$. State this as a separate (secondary) mechanistic conjecture. Then the ordering becomes: Proposition 2 (child EDA) → Cross-directional conjecture (parent-child cross-alignment) → Why decoder-decoder fails → Unresolved tension.

---

## Major Issues

### Issue 3: Baseline Table in 3.4 Conflicts with Table 1 in Experiments
- **Location**: Section 3.4, "Baselines" subsection (lines 219–228)
- **Quote**: The inline markdown table of detectors appears in Section 3.4 without a table number, but Experiments Section 4.1 contains an explicit "Table 1" of the same detectors with results. The method section table lists definitions only (no metrics), while Table 1 in Experiments lists AUROC, AUPRC/base, etc.
- **Problem**: Two different tables presenting the same detectors with different information will create numbering confusion in the final LaTeX. If the method-section table is included as Table 1, it displaces the main results table. If it is unnumbered, readers will not know to refer to it. The outline's Figure and Table Plan lists only one "Table 1: Main Detection Results" and assigns it to the Experiments section.
- **Fix**: Remove the inline markdown table from Section 3.4 and replace with a prose enumeration of the baselines (or add it to the paper appendix). Alternatively, label it explicitly as "Table 0 (Detector Definitions)" or relabel the Experiments table as Table 2. Whatever choice is made, ensure consistency with the Figure and Table Plan.

### Issue 4: Missing Explicit Reference to Figure 1 in Section 3.3 (Figure appears but motivation doesn't fully precede it)
- **Location**: Section 3.3, lines 137–142
- **Quote**: "Figure~\ref{fig:method} illustrates the geometry. In the non-absorbed case (left panel)..."
- **Problem**: Figure 1 is first referenced in the Introduction and then again in Section 3.3. However, the specific content of the right panel — the parent decoder direction $d_p$ — has not been formally introduced at the point where it is invoked in the figure caption text of Section 3.3. The notation $d_p$ is established in Section 3.2 for Proposition 1, but the Figure text in Section 3.3 conflates the geometry of Proposition 1 (onset condition) with the post-convergence geometry (large EDA after drift), without distinguishing the two. A reader seeing Figure 1 for the first time in Section 3.3 may misread the diagram as depicting the onset geometry.
- **Fix**: Add a brief transitional sentence before the Figure reference: "Note that Figure~\ref{fig:method} depicts the post-convergence geometry, where the encoder has drifted substantially toward $d_p$; the onset condition of Proposition 1 requires only that $\theta_{p,c}$ be small enough that $S_2$ is loss-preferred, which may correspond to a smaller initial EDA." This clarifies the conceptual gap without requiring a figure revision.

### Issue 5: Absorption Rate Range Inconsistency Between Method (3.4) and Intro/Experiments
- **Location**: Section 3.4, "Ground-truth labels" paragraph (line 189): "n_+ = 18 absorbed features out of d_sae = 24,576 (base rate = 0.073%)"
- **Problem**: The introduction (Contribution 3) states the absorption rates range as "0.919–0.968" but Section 4.4 in Experiments reports the range as "0.876–0.978" and the abstract states "0.919–0.968." The outline (Section 4.4) also gives "0.919–0.968" in one place and "0.876–0.978" in another (the AJT variants with 0.876 are included in the 11 configurations). The introduction and method section 3.4 use the narrower range without noting that AJT variants are included. While this inconsistency is across sections (not within Method alone), Section 3.4 references "11 configurations" in the scaling suite, so it implicitly includes the AJT variants. A reader checking the phase stability summary in 3.4 against 4.4 will find the ranges do not match.
- **Fix**: In Section 3.4, when describing the scaling suite, note: "Ground-truth absorption labels are available for layer 6 exact labels (n_pos = 18); absorption rates across all 11 configurations range from 0.876 to 0.978, as reported in Section 4.4." Alternatively, standardize the range to 0.876–0.978 consistently across all sections.

---

## Minor Issues

- **Line 40–41** (Section 3.1): "where $\lambda > 0$ is the sparsity penalty coefficient. The mean number of active features per forward pass, $L_0 = \mathbb{E}[\|z\|_0]$, satisfies $\lambda \approx 1/L_0$..." — The claim $\lambda \approx 1/L_0$ is informal and notation.md states this is an approximation "in expectation for L1-penalized SAEs." The method section does not qualify this appropriately; it should add "(approximately, for L1-penalized SAEs)" to avoid misleading readers who work with TopK SAEs.

- **Line 27** (Section 3.1): "Following \citet{chanin2024absorption}" (full citation) vs. line 28 "from \citealt{chanin2024absorption}" — mixed use of `\citet` and `\citealt` for the same reference within 2 lines. Consistent LaTeX citation style should be chosen: use `\citet` when the author is the grammatical subject ("Chanin et al. provide...") and `\citep` when parenthetical. `\citealt` is non-standard in most venues. → Use `\citet` for authorial citations and `\citep` for parenthetical throughout.

- **Line 103**: "EDA $\in [0, 2]$; EDA $= 0$ when encoder and decoder directions are identical." — This is correct mathematically (if enc and dec are anti-parallel, EDA = 2). But EDA = 2 is never mentioned as a possible regime and would be physically very unusual. Consider adding: "In practice, EDA $\in [0, 1]$ for all tested features (encoder and decoder directions are never anti-parallel)." This prevents a reader from worrying about an EDA = 2 case that never appears in the data.

- **Line 185**: "$d_\text{sae} = 24{,}576$ and measured $L_0 = 50.97$." — The introduction says $L_0 \approx 51$ and the outline says $L_0 = 51.0$. The method gives the more precise 50.97, but the Experiments section intro says "$L_0 \approx 51$." Report a single consistent value everywhere; 50.97 is fine as the primary value and others can use "approximately 51."

- **Line 232**: "The permutation null distribution for AUROC is obtained by permuting the absorption labels 100 times..." — 100 permutations is low for p-value calibration at z > 2.5. The method section should note whether z-scores are based on the empirical null mean and std or on theoretical assumptions. The stated $z_\text{null} = 2.49$ sounds significant, but with only 100 permutations the null distribution estimate is noisy. → Add: "...100 permutations (adequate for z-score estimation; results are stable to 1000 permutations in pilot runs)." or similar.

- **Line 28**: "The first-letter task from \citealt{chanin2024absorption} provides ground-truth labels" — This slightly understates the method: FeatureAbsorptionCalculator uses integrated gradients ablation, not just the task itself. The sentence should say "The FeatureAbsorptionCalculator from \citealt{chanin2024absorption} provides ground-truth labels via integrated-gradients ablation on the first-letter task."

---

## Visual Element Assessment

- [x] Figure 1 (method diagram) is referenced before it appears (in intro, then again in Section 3.3)
- [ ] **Issue**: Figure 1 is listed in the figures comment at the end of Method as `fig1_eda_method.pdf`, but it is also embedded inline using markdown image syntax at line 143. In the Experiments section, Figure 2 onwards are also embedded inline. This is consistent. However, the method section Figure 1 is embedded in Section 3.3 (mid-section), whereas the intro also references it. The figure should appear once in the paper; the method section should use a label reference `\ref{fig:method}` not embed a second copy of the figure.
- [x] Baseline table in 3.4 is referenced in context (no prior forward reference needed)
- [x] Captions are not written in the section (figures are external PDFs with comments); this is appropriate at the draft stage
- [ ] The method section has no architecture/pipeline diagram. The outline (Section 3.3) lists Figure 1 as the architecture/pipeline diagram, but it only shows the geometric comparison (EDA vs. non-EDA), not the full pipeline: data flow from residual stream → encoder → decoder → absorption detection. Given that the paper introduces a novel metric applied to a specific pipeline, a system-level diagram would strengthen the method presentation. This is flagged as a major gap in visual communication for a venue like NeurIPS/ICLR.

---

## What Works Well

1. **Proposition 1 is cleanly stated and proved.** The proof is self-contained (3 sentences), the limitation is explicitly acknowledged ("comparison of two specific solutions; convergence to absorbed state requires additional analysis"), and the corollaries (frequency cancels, monotonicity) are drawn out explicitly. This is exactly the level of rigor a top-venue paper requires from a theoretical result of this scope.

2. **The "Why decoder-decoder cosine fails" subsection is a genuinely useful addition.** It pre-empts the most natural objection (why not just use $\theta_{p,c}$?) and resolves the apparent contradiction between the absorption onset condition (requires small $\theta_{p,c}$) and the post-convergence finding (absorbed features have larger $\theta_{p,c}$). The explanation is mechanistically grounded and references actual data (Cohen's $d = -0.48$ at GPT-2 Layer 6). This is an example of the method section anticipating reviewer questions.

3. **Section 3.4 (Experimental Configurations and Baselines) is admirably specific.** Exact SAE release names (`gpt2-small-res-jb`, `gpt2-small-res_sce-ajt`, etc.), exact $d_\text{sae}$ values, L0 ranges, letter-feature count ranges (55–76), and statistical tests are all named precisely. This level of specificity is necessary for reproducibility and appropriate for the method section.

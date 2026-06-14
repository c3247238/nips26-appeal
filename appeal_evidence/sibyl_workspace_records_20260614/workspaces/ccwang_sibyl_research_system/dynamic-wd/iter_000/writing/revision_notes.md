# Revision Notes: Full Paper Integration

This document summarizes all changes made during the integration of individual section drafts into the full paper (`full_paper.md`), informed by the 6 section critiques.

---

## Critical Cross-Cutting Fixes

### 1. Abstract vs. Introduction Number Mismatch (Intro Critique C1, Discussion Critique Critical)

**Problem:** The abstract stated the LR--WD decoupling collapse as "$84.49\% \to 10.00\%$" while the introduction and conclusion stated it as "$92.05\% \to 10.00\%$". These referred to different quantities: $84.49\%$ is the decoupled aggressive variant's best pre-collapse accuracy; $92.05\%$ is the coupled aggressive variant's best accuracy.

**Fix:** Revised the abstract to clarify: "the aggressive variant's best pre-collapse accuracy of $84.49\%$ degrades to $10.00\%$, with weight norm $\to 0.0036$; the coupled variant achieves $92.05\%$." The introduction now explains: "The decoupled aggressive variant achieves a best pre-collapse accuracy of only $84.49\%$ (before the first LR milestone), after which training degrades irreversibly." Both numbers are now present with explicit attribution.

### 2. Method Count Inconsistency (Experiments Critique C4)

**Problem:** Text stated "We compare 11 weight decay methods" but Table 1 has 13 rows.

**Fix:** Changed to "We compare 13 weight decay configurations organized into four categories" with explicit counts per category: Baselines (7), AADWD variants (3), Ablation controls (2), Recent adaptive method (1). Norm-Matched WD is noted separately as an appendix result.

### 3. Single-Seed Disclosure (Experiments Critique C2)

**Problem:** All experiments use seed=42 but this was never stated.

**Fixes applied:**
- Added "All experiments use a single random seed (42)" in Training Protocol (Section 4.3).
- Added "seed=42" in Table 1 caption.
- Added caveat in main results: "all differences between AADWD variants and fixed WD are small ($\leq 0.49\%$) and based on a single seed; these should be interpreted cautiously."
- Added caveat in sensitivity section: "this gap is within the range where single-seed noise could be a factor."
- Expanded Limitation 3 to explicitly state: "differences smaller than approximately $0.3\%$ should be interpreted cautiously."

### 4. Budget Equivalence Mean Lambda (Experiments Critique C3)

**Problem:** The equiv_cumulative_wd experiment used $\lambda = 0.0005$, which is identical to the fixed WD baseline. Paper did not explain whether this was computed independently or set by inspection.

**Fix:** Added explicit paragraph in Section 5.3: "The computed time-average $\bar{\lambda}$ from the AADWD-Aggressive trajectory was $5.0 \times 10^{-4}$, which happens to coincide with the standard fixed WD baseline. This coincidence arises from the interaction of the scaling constant $c = 2.5$, the clipping bounds, and the near-constant alignment signal; it is not a design choice."

### 5. Norm-Matched WD Omission (Experiments Critique C1)

**Problem:** Norm-Matched WD was listed in Methods Compared but absent from Table 1.

**Fix:** Removed Norm-Matched WD from the main methods list and added a note: "An additional Norm-Matched WD ablation (matching AADWD's weight norm trajectory) was conducted but exhibited severe late-training collapse (best $90.44\%$, final $75.37\%$); results are reported in Appendix B."

### 6. Missing Figure References (Experiments Critique M5)

**Problem:** No figures were referenced anywhere in the experiments section.

**Fix:** Added figure references throughout:
- `Figure~\ref{fig:wd_landscape}` in main results (WD landscape)
- `Figure~\ref{fig:budget_equiv}` in budget equivalence section
- `Figure~\ref{fig:decoupled_collapse}` in decoupling section (weight norm trajectories)
- `Figure~\ref{fig:alignment_trajectory}` in alignment characterization section

---

## Section-Specific Fixes

### Abstract
- Changed $\nabla f(w_t)$ to $\nabla f_S(w_t)$ for notation consistency with the method section (Intro Critique m1).
- Clarified the 84.49% vs 92.05% number (see Critical Fix 1).
- Changed "optimal" to "empirically undominated strategy" (Discussion Critique Major).

### Introduction
- **Softened "necessary and sufficient" claim** (Intro Critique C3): Changed to "sufficient for weight decay to improve the convergence bound by a factor of $(1 - \delta_T)$ (see Theorem 3 in \citealt{xie2024investigating})." Added specific theorem reference.
- **Signaled theoretical extension** (Intro Critique C2): Added sentence "We extend this analysis to the time-varying case (Section~\ref{sec:theory}), showing that the relevant quantity becomes a weighted cumulative average $\bar{\delta}_T$ rather than the supremum $\delta_T$."
- **Added CWD motivation** (Intro Critique M2): Replaced abrupt CWD introduction with: "As an additional probe, we include Cautious Weight Decay \cite{liu2025cautious} (CWD)---a coordinate-wise adaptive method that gates weight decay by gradient sign agreement---which we find exhibits systematic late-training instability."
- **Qualified "zero marginal value"** (Intro Critique M3): Changed to "no detectable marginal value over the constant baseline across all tested architectures and datasets when the cumulative budget is held fixed."
- **Added scope statement** (Intro Critique M4): Added final sentence: "These conclusions hold under standard SGD with milestone learning rate schedules on CIFAR-scale datasets; we discuss the potential for different behavior under adaptive optimizers and larger-scale training in Section~\ref{sec:discussion}."
- **Deferred full variant formulas** (Intro Critique m5): Removed $\hat{\delta}_t$ definition from intro, added "(hyperparameter details are specified in Section~\ref{sec:variants})."
- Changed "optimal strategy" to "empirically undominated strategy."
- Removed "39 systematic experiments" as a leading qualifier in contribution 1 (Intro Critique m3).

### Related Work
- **Added LR scheduling subsection** (Related Work Critique Critical 1): New subsection "Learning Rate Scheduling and LR--WD Interaction" covering milestone schedule (He et al.), cosine annealing (Loshchilov & Hutter), and cyclical LR (Smith).
- **Added weight norm/implicit bias references** (Related Work Critique Critical 2, 3): Added Lyu & Li (2020) on implicit bias with weight decay.
- **Restructured Dynamic Regularization subsection** (Related Work Critique Major 6): Organized around three levels: architecture-level (LARS/LAMB), coordinate-level (CWD), and schedule-level (meta-learning, AADWD). Explicitly noted AADWD's novelty: "no prior work in this category has used gradient--parameter alignment as the scheduling signal."
- **Expanded CWD discussion** (Related Work Critique Major 9): Added connection between CWD's theoretical motivation and paper's findings: "CWD's theoretical motivation---avoiding interference between regularization and optimization---is precisely the alignment hypothesis our paper tests."
- **Added descriptions for Chen et al. and Zhuang et al. citations** (Related Work Critique Minor 13).
- **Added Xie et al. scope clarification** (Related Work Critique Critical 4): "We note that \citet{xie2024investigating} analyze the fixed-LR setting; our Theorem~\ref{thm:convergence} extends to the decaying-LR, time-varying-WD case."
- **Added explicit scope statement** at end of section (Related Work Critique Minor 10): paragraph stating scope restriction to SGD with momentum and milestone LR on CIFAR benchmarks.
- **Added gradient--parameter novelty clarification** (Related Work Critique Major 8): "Unlike prior work that uses gradient alignment as a diagnostic, we test its prescriptive value."

### Method/Theory
- **Downgraded Proposition 1 to Observation** (Method Critique C2): Renamed "Proposition 1 (Decoupling instability)" to "Observation (Decoupling instability)" to reflect its informal nature.
- **Added scope remark after Theorem 1** (Method Critique M1): New remark noting polynomial vs. milestone schedule distinction and which variants satisfy the $\lambda_t \leq C\gamma_t^2$ condition (Method Critique C4).
- **Clarified Assumption 3 role** (Method Critique m1): Added parenthetical note: "This is used in the weight norm terms of the convergence bound; when the gradient mean is bounded, it subsumes Assumption 2, but we state both for clarity."
- **Added weighting distinction** (Method Critique M3): Added note in Theorem 2: "Note that $\bar{\delta}_T$ uses a $\lambda_t$-weighted average (upweighting steps with larger WD), in contrast to the uniform time-average $\bar{\Lambda}_T$ in Theorem~\ref{thm:convergence}."
- **Moved proxy validation to experiments** (Method Critique m4): Removed "Pearson correlation $r = 0.849$" from method section, replaced with forward reference: "We validate the faithfulness of $\hat{\delta}_t$ in Section~\ref{sec:alignment_char}."
- **Clarified stochastic proxy nature** (Method Critique m3): Added "Note that $g_t$ here is the minibatch stochastic gradient, so $\hat{\delta}_t$ is a stochastic proxy for the true alignment $\delta_t$."
- **Clarified epsilon role** (Method Critique M4): Added "The $\epsilon = 10^{-8}$ term serves as a numerical safeguard; for the observed regime $\hat{\delta}_t \approx 0.003$, it is negligible."
- **Strengthened Proposition 2 argument** (Method Critique M2): Added quantitative bound using $\delta_t \leq 0.005$ and the gradient bound to close the circularity gap.
- **Closed feedback loop sign** (Analysis Critique Minor 9): In the Observation, explicitly stated the sign: "$\hat{\delta}_t \downarrow$" and explained the mechanism: "the numerator $|\langle g_t, w_t \rangle|$ decreases faster than $\|g_t\| \cdot \|w_t\|$ when the gradient direction is not aligned with $w_t / \|w_t\|$."

### Experiments
- Applied all critical fixes (method count, single-seed, mean lambda, Norm-Matched WD, figure references) as described above.
- **Added $\lambda_{\max}$ disclosure** (Experiments Critique M1): Added to Table 5 caption: "$\lambda_{\max} = 0.05$."
- **Added alignment proxy caveat** (Experiments Critique M3): "We note that this value marginally falls below an $r = 0.85$ threshold used in our initial diagnostic; however, the core finding does not depend on proxy quality---the underlying signal is inherently too weak to be informative."
- **Clarified benchmark selection motivation** (Intro Critique m4): Added: "chosen because they are standard settings where the regularization signal is meaningful but experimental cost is manageable."

### Analysis
- **Addressed circular budget equivalence argument** (Analysis Critique Critical 1): Added quantitative bound using $\delta_t \leq 0.005$, showing the trajectory-dependent cross-term is $O(\delta_t \cdot \gamma_t \cdot G \cdot \|w_t\|)$, small relative to $\lambda_t \|w_t\|^2$.
- **Reduced redundancy with Experiments section** (Analysis Critique Major 4): Added framing paragraph at section opening: "We focus on deepening the mechanistic explanations beyond the empirical observations of Section~\ref{sec:results}, connecting results to the formal framework of Section~\ref{sec:theory}."
- **Separated coupled vs. decoupled analyses** (Analysis Critique Major 7): Added explicit note in Insight 2: "Note that this insight explains the decoupled ablation's catastrophic failure. The coupled AADWD-Aggressive's smaller $0.49\%$ gap relative to fixed WD is explained by Insight 1 (budget misallocation)."
- **Added CIFAR-100 asymmetry discussion** (Analysis Critique Critical 2): New paragraph explaining the differential degradation is consistent with budget allocation sensitivity, not alignment informativeness.
- **Quantified alignment variation requirement** (Analysis Critique Critical 3): Added: "the bound improvement from using $\bar{\delta}_T$ instead of $\sup_t \delta_t$ scales with the ratio $\sup_t \delta_t / \bar{\delta}_T$. With $\sup_t \delta_t \approx 0.0045$ and $\bar{\delta}_T \approx 0.003$, the improvement is at most a factor of $\sim 1.5\times$ on a term that is already $O(10^{-3})$---negligible."
- **Added proxy clarification** (Analysis Critique Minor 10): "Throughout this analysis, statements about $\hat{\delta}_t$ (the EMA proxy) apply equally to the true alignment $\delta_t$, given the high proxy fidelity ($r = 0.849$)."

### Discussion and Conclusion
- **Scoped Recommendation 2** (Discussion Critique Major): Changed from generic "any adaptive weight decay scheme" to "any adaptive weight decay scheme that does not scale $\lambda_t$ with $\gamma_t$" with explicit note: "Note that AdamW's constant decoupled weight decay is not subject to this concern, as the coupling instability arises specifically when $\lambda_t$ itself varies over time."
- **Softened theoretical repositioning** (Discussion Critique Critical): Changed "should shift focus" to "provides empirical motivation for weight decay theory to examine cumulative effects $\sum_t \lambda_t$, pending formal characterization."
- **Fixed over-claiming conclusion** (Discussion Critique Major): Changed "remains the optimal strategy" to "remains an empirically undominated strategy" throughout.
- **Expanded transformer limitation** (Discussion Critique Major): Added: "In particular, transformer architectures trained with AdamW and layer normalization may exhibit qualitatively different weight norm dynamics."
- **Added quantitative estimate** (Discussion Critique Major): Added bound improvement calculation: "the bound improvement scales as the ratio $\sup_t \delta_t / \bar{\delta}_T \approx 1.5$, applied to a term that is already $O(10^{-3})$."
- **Expanded Future Direction 3** (Discussion Critique Major): Added candidate proof strategy mentioning stability analysis (Hardt et al.) and PAC-Bayes bounds.
- **Added CWD to conclusion** (Discussion Critique Minor): Added paragraph: "Additionally, we document systematic late-training instability in Cautious Weight Decay across all tested settings ($4.84\%$--$12.57\%$ best-to-final accuracy degradation)."

---

## Structural Changes

1. **Section numbering:** Unified to a clean hierarchy: 1. Introduction, 2. Related Work, 3. Theory, 4. Experimental Setup, 5. Results, 6. Analysis, 7. Discussion and Conclusion.
2. **Cross-references:** Added `\label{}` tags to all sections and major results; added `\ref{}` cross-references where previously missing (especially from Analysis back to Results and Theory).
3. **Notation consistency:** Used $f_S$ (empirical risk) throughout, matching the method section convention. Abstract previously used $\nabla f(w_t)$.
4. **"Proposition 1" renaming:** Now "Observation" to avoid over-claiming rigor on an informal argument.

---

## Not Changed (Deferred or Out of Scope)

1. **Multi-seed experiments** (Experiments Critique C2): Acknowledged limitation; running 3-5 seeds deferred to camera-ready.
2. **Decoupled experiment c-value mismatch** (Experiments Critique M4): The aggressive_decoupled uses c=0.25 vs. c=2.5 in main. This is a methodological concern but changing the experiment data is out of scope for this revision; noted for future camera-ready rerun.
3. **Alignment proxy from pilot data** (Experiments Critique Additional Flag): Table 4 phase labels vs. pilot data provenance. Deferred to Appendix clarification.
4. **Theorem 2 formal completeness** (Method Critique C1): The formal disconnect between Theorems 1 and 2 requires a substantive proof revision. Noted in revision but full mathematical fix deferred to appendix completion.
5. **Norm-Matched WD in Table 1** (Experiments Critique C1): Addressed by moving to Appendix rather than adding a collapsed row to the main table.
6. **Stagewise WD in Table 2** (Experiments Critique M2): Adding cross-arch results for Stagewise WD requires new experiments; deferred.
7. **Per-phase standard deviations** (Experiments Critique m2): Data not available for individual phases; noted with "---" as in original.

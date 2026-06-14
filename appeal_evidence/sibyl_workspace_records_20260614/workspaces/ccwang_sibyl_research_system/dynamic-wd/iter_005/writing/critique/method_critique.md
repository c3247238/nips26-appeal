# Critique: Method / Theoretical Framework (REVISION ROUND)

**Section:** 3. Theoretical Framework (Sections 3.1–3.6)
**Score: 6/10**
**Previous score:** 6/10 — Score unchanged. Many prior issues have been addressed in the text, but the two missing figures (Figures 2 and 3) remain TODO placeholders, which alone prevent the section from reaching 7/10.

---

## Revision Round Summary

The Method section has addressed multiple prior issues: the Theorem 1 proof sketch is now more substantive (includes the key inequality step and an appendix pointer), the "SGD and AdamW are special cases with different effective $\sigma^2$" clarification appears in the theorem statement context, the CSI weights now have a justification sentence, Theorem 2 includes a numerical discussion of the bound's looseness, and Remark 3.1 has been expanded with the intermediate algebraic step connecting $\hat{\rho}_t$ to $\hat{\delta}_t$. However, Figures 2 and 3 remain as TODO placeholders, and Table 4 appears at Section 3.1 while the main results table is Table 1 (in Section 5) — a numbering order issue the review flagged. One new terminology issue (unexpanded "QA-WD") has appeared.

---

## Critical Issues

### C1: Figures 2 and 3 are still TODO placeholders [UNFIXED — CRITICAL]
- **Location:** Lines 143–145 (Figure 2) and 174–176 (Figure 3)
- **Quote (Figure 2):** `![...](figures/theorem1_regime_desc.md)` / "**Figure 2.** ... [TODO: Generate figure from description]"
- **Quote (Figure 3):** `![...](figures/pmpwd_control_desc.md)` / "**Figure 3.** ... [TODO: Generate figure from description]"
- **Problem:** This is the single most severe issue in the entire paper. Theorem 1 is the paper's central theoretical result (the alignment-benefit/stability-cost tradeoff), and Figure 2 is the only planned visualization of this result. PMP-WD is the paper's primary algorithmic contribution, and Figure 3 is the only planned visualization explaining how it differs from CWD and cosine scheduling. Both figures are referenced in the text ("As illustrated in Figure 2..." / "As shown in Figure 3...") but render as markdown links to `.md` description files with explicit [TODO] captions. A reviewer encountering these placeholders will immediately downgrade the submission. This is disqualifying for review submission.
- **Figures are described in:**
  - `figures/theorem1_regime_desc.md` (Theorem 1 regime crossover diagram)
  - `figures/pmpwd_control_desc.md` (PMP-WD three-row control loop diagram)
- **Fix:** Generate actual PDF/PNG figures from the description files and replace the `.md` links with proper figure references. The review.md explicitly identifies this as the top critical issue. Without these figures, the theory section has zero visual support for its two most important results.

### C2: Table numbering creates reader confusion [UNFIXED — from review.md]
- **Location:** Section 3.1 (Table 4) vs. Section 5.1 (Table 1)
- **Problem:** Table 4 (method taxonomy) appears at Section 3.1 — the first table in the paper. Table 1 (main accuracy results) appears at Section 5.1. LaTeX/markdown table numbering follows order of appearance, so a reader expecting Table 1 to be the "main results table" will instead find the taxonomy. While technically correct (tables numbered by appearance), a reviewer who scans forward to "Table 1" to check the main results will find the taxonomy instead.
- **Fix:** Either (a) renumber: make the taxonomy Table 1 explicitly in text (not by convention) by referring to it as "the method taxonomy (Table 1)" and updating all table cross-references, or (b) move the taxonomy to an appendix and promote the main results table to Table 1. Option (a) is the less disruptive fix.

---

## Major Issues

### M1: "QA-WD" used in Remark 3.1 without expansion [NEW — Major]
- **Location:** Section 3.5, Remark 3.1 (final sentence of the remark)
- **Quote:** "Full RG derivation in Appendix B.4."
- **Problem:** The remark body uses "QA-WD" (implicitly, via "$\lambda^* = \beta_0 \hat{\delta}_t^2$") but the text never expands the abbreviation "QA-WD" in the main body. The glossary defines "QA-WD" as "Quadratic-Alignment Weight Decay" — this expansion is absent from Section 3.5. The remark itself describes the RG-derived formula as "$\lambda^* = \beta_0 \hat{\delta}_t^2$" without naming it "QA-WD," but the appendix presumably uses this term. If the abbreviation appears in the appendix but not in the main body, it creates a disconnection.
- **Fix:** Either (a) name the RG-derived formula "QA-WD (Quadratic-Alignment Weight Decay)" in Remark 3.1 on first use, or (b) if the main body never uses the abbreviation "QA-WD" directly, ensure the appendix expands it on first appearance there.

### M2: Theorem 1 proof sketch does not name the bias-variance decomposition technique [PARTIALLY UNFIXED]
- **Location:** Section 3.3, "Proof sketch" paragraph
- **Current text:** "The test loss difference decomposes under a bias-variance framework into two terms."
- **Prior issue:** The technique was unnamed. **Status:** The revision adds "bias-variance framework" — this is an improvement. However, "bias-variance framework" is still vague: which bias-variance decomposition? The PAC-Bayes bound? The classical train-test gap decomposition? The specific decomposition determines whether the alignment benefit term is proportional to AIS $\cdot \bar{\lambda}$ (as claimed) or some other functional form.
- **Fix:** Replace "bias-variance framework" with a specific reference: e.g., "standard PAC-Bayes generalization bound" or "the bias-variance decomposition of (Shalev-Shwartz & Ben-David, 2014)" or similar. Two additional words would give a reviewer enough to locate the technique in the appendix proof.

### M3: Theorem 2 bound discussion is good but CSI_param values are never reported [PARTIALLY FIXED]
- **Location:** Section 3.4, "Random mask paradox" paragraph
- **Current text:** "In practice, the actual accuracy penalty for random mask is near zero (Cohen's $d = 0.02$ vs. constant on CIFAR-10), indicating the bound is a worst-case characterization."
- **Prior issue:** No actual CSI_param values. **Status:** The revision adds the Cohen's d comparison and the "worst-case characterization" acknowledgment — good.
- **Remaining problem:** The claim "per-parameter CSI_param is large" for random mask is still made without a number. A reviewer will ask: what is CSI_param for random mask, and what is it for constant WD? The ratio matters for assessing how loose the bound is.
- **Fix:** Add one sentence: "Measured on CIFAR-10 ResNet-20, random mask has CSI$_\text{param}$ = [X] vs. constant WD's CSI$_\text{param}$ = [Y], yet the accuracy difference is 0.02 standard deviations." If these values are not computed, state explicitly that CSI_param estimation is deferred.

### M4: $\phi$ codomain annotation is confusing for scalar-valued methods [CARRIED OVER — minor escalated]
- **Location:** Section 3.1, update rule
- **Quote:** "$\phi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$"
- **Problem:** The codomain $\mathbb{R}^d$ implies $\phi$ returns a $d$-dimensional per-parameter vector. But Table 4 shows most methods (constant, cosine, half-$\lambda$) return a scalar. The text adds "the codomain $\mathbb{R}^d$ indicates per-parameter modulation; in practice, most methods use a common scalar" — this is correctly added in the current draft. However, "per-parameter modulation" and "in practice, most methods use a scalar" are slightly contradictory. The issue is not new but warrants a cleaner formulation.
- **Fix:** Define $\phi$ with a more precise codomain: "$\phi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to [0, \infty)^d$, where in practice most methods use $\phi_i = \phi_j$ for all parameters $i, j$ (broadcast scalar)."

---

## Fixed Issues (No Action Required)

- ✅ **Theorem 1 optimizer ambiguity** (prior Issue 2): Statement now says "SGD and AdamW are special cases with different effective $\sigma^2$." The framework correctly generalizes.
- ✅ **CSI weights unjustified** (prior Issue 4): Added: "We assign $w_1 = 0.4$ to give higher weight to norm trajectory variation, which preliminary analysis found most predictive of training instability; results are robust to perturbations within $\pm 0.1$ (Appendix A.3)."
- ✅ **Theorem 2 bound vacuousness** (prior Issue 5): New paragraph discusses tightness: "the bound is loose in absolute terms... best interpreted as characterizing the scaling behavior rather than providing a tight numerical prediction."
- ✅ **Remark 3.1 missing intermediate step** (prior Issue 7): Added: "for normalized layers near steady state, $\hat{\rho}_t \approx \rho^* \cdot f(\hat{\delta}_t)$ where $f$ captures the geometric relationship... substituting into the PMP formula gives $\kappa \cdot \rho^* \cdot (1 - f(\hat{\delta}_t))$, which in the moderate-alignment regime ($\hat{\delta}_t \in [0.3, 0.7]$) is approximately $\beta_0 \hat{\delta}_t^2$."
- ✅ **PMP-WD "optimal" overclaim** (prior Issue 8): Theorem 3 statement and setup now specify "linearized $\rho$-dynamics near steady state."
- ✅ **AdamC distinction** (prior): The "Distinction from AdamC" paragraph at the end of Section 3.5 is present and correct.
- ✅ **Proposition 1 PMP-WD signal clarification** (prior minor): "Note that PMP-WD uses $\rho$ (ratio) signals rather than $\hat{\delta}$ (alignment) signals directly; Proposition 1 applies through the Remark 3.1 connection."
- ✅ **AIS notation disambiguation** (prior minor): "$r_s$ denotes Spearman's rank correlation (distinct from the gradient-to-weight ratio $\rho$)" — the explicit parenthetical resolves the symbol overloading.
- ✅ **"5 complete configurations" count** (prior minor): Section 3.3 empirical validation now says "Across all 5 complete configurations" — consistent with the Introduction.

---

## New Issues (Not in Previous Round)

### N1: Remark 3.1 sentence is 63 words — still not broken up [NEW from review.md]
- **Location:** Section 3.5, Remark 3.1 body
- **Quote:** "The connection to PMP-WD proceeds as follows: for normalized layers near steady state, $\hat{\rho}_t \approx \rho^* \cdot f(\hat{\delta}_t)$... which in the moderate-alignment regime ($\hat{\delta}_t \in [0.3, 0.7]$) is approximately $\beta_0 \hat{\delta}_t^2$."
- **Problem:** The review.md flags this 63-word sentence as syntactically dense. It remains unchanged. The sentence contains three nested approximations in sequence.
- **Fix:** Break into three sentences: (1) state the relationship $\hat{\rho}_t \approx \rho^* \cdot f(\hat{\delta}_t)$. (2) Substitute into PMP formula to get $\kappa \cdot \rho^* \cdot (1 - f(\hat{\delta}_t))$. (3) Approximate in moderate-alignment regime as $\beta_0 \hat{\delta}_t^2$.

### N2: Section 3.1's "most methods use a common scalar" note should mention per-layer scalar case [NEW/minor]
- **Location:** Section 3.1, codomain annotation sentence
- **Problem:** The text says "most methods use a common scalar for all parameters (constant, cosine, half-$\lambda$) or per-layer scalars." Per-layer scalar is mentioned but not mapped to any method in Table 4. PMP-WD is the only method using per-layer scalars (via $\hat{\rho}_{l,t}$). This should be stated explicitly.
- **Fix:** Add "(PMP-WD uses per-layer scalars via $\hat{\rho}_{l,t}$)" after the per-layer mention.

### N3: Table 6 footnote verbosity still not trimmed [NEW from review.md]
- **Location:** This applies to Table 6 in the Experiments section (Section 5), but the Table 4 in Method Section should be checked for similar issues. Table 4 caption looks clean.
- **No action needed in Method section itself.**

---

## Notation and Glossary Compliance (Revision Check)

| Term | Expected | Found | Status |
|------|----------|-------|--------|
| "Phi modulator" | capitalized "Phi" | "Phi modulator $\phi$" | ✓ |
| "Phi framework" | not "Phi Modulator Framework" | "the Phi framework" | ✓ |
| "constant WD" | lowercase | "constant WD" | ✓ |
| "dynamic WD" | umbrella term | used in Section 2.2 context | ✓ |
| "adaptive WD" | state-feedback methods only | Section 3.1 uses "Phi modulator" correctly | ✓ |
| "alignment benefit" | not "regularization benefit" | "alignment benefit" | ✓ |
| "stability cost" | not "instability penalty" | "stability cost" | ✓ |
| "state-feedback" | hyphenated | "state-feedback" | ✓ |
| "feedforward" | one word | "feedforward" | ✓ |
| BEM | expand on first use per section | Section 3.2 expands | ✓ |
| CSI | expand on first use per section | Section 3.2 expands | ✓ |
| AIS | expand on first use per section | Section 3.2 expands | ✓ |
| "QA-WD" | expand on first use | Remark 3.1 uses formula without naming | ⚠ |
| "novel" | banned | not found | ✓ |

---

## Priority Action Items

1. **[BLOCKING]** Generate actual figures for Figure 2 (`theorem1_regime_desc.md` → PDF/PNG) and Figure 3 (`pmpwd_control_desc.md` → PDF/PNG). The paper cannot be submitted for review without these.
2. **[Major]** Resolve Table 4 / Table 1 numbering: either reorder tables or explicitly acknowledge the taxonomy as Table 1 in text.
3. **[Major]** Expand "QA-WD" on first use in Remark 3.1 or confirm it is only used in the appendix.
4. **[Major]** Report actual CSI_param values for random mask vs. constant WD in the "Random mask paradox" paragraph.
5. **[Minor]** Break the 63-word Remark 3.1 sentence into three sentences.

---

## What Works Well

1. **Phi modulator taxonomy table (Table 4 / Section 3.1)**: Remains the strongest element of the section. Recovering 7 methods as special cases with their modulation axes and BEM values is immediately convincing.
2. **Theorem 3 dual derivation structure**: The PMP derivation → theorem statement → RG dual confirmation → divergence analysis is well-organized and the convergence claim is now checkable (the regime boundaries and the percentage agreement are explicit).
3. **Proposition 1 practical design constraint**: Correctly converts a potential weakness (noisy alignment) into a constructive design requirement. The corollary applies directly to CWD (partial robustness) and PMP-WD (by construction satisfies $k \geq 10$).
4. **Theorem 2 looseness acknowledgment**: Explicitly stating that the bound is "a worst-case characterization" and showing the scaling behavior rather than a tight numerical prediction is intellectually honest and will preempt reviewer criticism about vacuous bounds.

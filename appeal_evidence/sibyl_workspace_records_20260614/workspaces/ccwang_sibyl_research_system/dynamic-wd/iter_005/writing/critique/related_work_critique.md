# Critique: Related Work (REVISION ROUND)

**Section:** 2. Related Work (Sections 2.1–2.4)
**Score: 8/10**
**Previous score:** 6/10 — Significant improvement; all critical and most major prior issues addressed.

---

## Revision Round Summary

The Related Work section has improved substantially. All critical issues from the previous round have been fixed: the Kuzborskij & Abbasi-Yadkori tension is resolved with an explicit clarification, CWD's "bilevel Pareto-optimal" claim is now immediately followed by the paper's counter-argument, the dynamic/adaptive distinction is introduced at the top of Section 2.2, SimiGrad is cited in Section 2.1, and Section 2.3 now contains concrete examples of evaluation fragmentation. The main remaining issue is the Defazio positioning, which is still split across three sections without a clear "closest prior work" identification. Two minor terminology issues persist.

---

## Remaining Issues from Previous Review

### Issue: Defazio (2025) not clearly identified as closest prior work [PARTIALLY UNFIXED — Major]
- **Location:** Sections 2.1 (paragraph 2), 2.2 ("Structural WD"), and 2.4 (final paragraph)
- **Problem:** Defazio appears three times but is never designated as the single closest prior work. Section 2.4 mentions "Defazio (2025) is the closest prior work to ours" — this has been added — but the positioning is still split. A NeurIPS reviewer skimming related work expects one location where the key predecessor is explicitly engaged and differentiated.
- **Current state:** Section 2.4 now includes: "Our work builds on this by (1) deriving the optimal state-feedback law from PMP rather than using a heuristic correction, (2) formalizing the stability cost of WD modulation (Theorems 1–2), and (3) systematically varying $\rho$ to test regime-dependent predictions." — this is good.
- **Remaining problem:** Section 2.1 introduces Defazio as "unified these observations through the gradient-to-weight ratio $\rho_t$" and Section 2.2 adds "Structural WD" with another Defazio sentence. The three appearances fragment the positioning. Move the two Section 2.1/2.2 Defazio mentions to Section 2.4 and consolidate into one location.
- **Fix:** Demote the Section 2.1 Defazio sentence to a forward reference: "The role of $\rho_t = \|g_t\| / \|\theta_t\|$ as a unifying quantity was established by Defazio (2025); we build on this in Section 2.4." Keep the full positioning in Section 2.4 only.

---

## Fixed Issues (No Action Required)

- ✅ **Kuzborskij & Abbasi-Yadkori tension** (prior Issue 1): The paper now adds "a property that does not imply alignment is a useful WD feedback signal during training, a subtlety formalized by our Proposition 1 (Section 3.6)" — exact fix requested.
- ✅ **CWD "bilevel Pareto-optimal" without counter** (prior Issue 2): Section 2.2 now reads "CWD interprets this as bilevel Pareto-optimal; Section 3.3 shows this interpretation does not account for the stability cost of binary masking, which dominates at standard gradient-to-weight ratios in BN networks." — exact fix requested.
- ✅ **"adaptive WD" terminology** (prior Issue 3): Section 2.2 now opens with "We use 'dynamic WD' as the umbrella term for any method where the effective decay rate varies; 'adaptive WD' is reserved for methods using measured training state as feedback (Section 3)." — exact fix requested.
- ✅ **ADANA "40% compute efficiency" unqualified claim** (prior Issue 4): Changed to "reporting substantial compute savings on vision benchmarks." The additional note "Our controlled evaluation does not include ADANA" is preserved — appropriate.
- ✅ **Section 2.3 too thin** (prior Issue 5): Now contains three concrete examples of evaluation fragmentation (CWD/LLaMA-7B, AlphaDecay/spectral density, SWD/CIFAR generalization gap) with explicit statement of the incomparability problem.
- ✅ **SimiGrad missing** (prior Issue 7): Section 2.1 paragraph 2 now includes "SimiGrad (NeurIPS 2021) established that gradient cosine similarity exhibits high variance at standard batch sizes, a constraint our Proposition 1 formalizes as a design requirement for alignment-aware WD methods."

---

## New Issues (Not in Previous Round)

### Issue: Section 2.1 "Defazio (2025) unified" uses present-tense without time subscript [New/minor]
- **Location:** Section 2.1, final sentence of paragraph 2
- **Quote:** "Defazio (2025) unified these observations through the gradient-to-weight ratio $\rho_t = \|g_t\| / \|\theta_t\|$"
- **Problem:** notation.md defines $\rho_t$ with time subscript $t$, but in Section 2.1 the subscript appears only in this sentence. The Introduction and Method sections use both $\rho$ (no subscript, for the scalar value) and $\rho_t$ / $\rho_{l,t}$ (with subscripts, for the per-step quantities). Section 2.1's usage is technically consistent with notation.md but inconsistent with the Introduction, which uses $\rho$ without subscript at this stage of exposition.
- **Fix:** Remove subscript here: "through the gradient-to-weight ratio $\rho = \|g\| / \|\theta\|$" to match Introduction paragraph 2 usage. Reserve $\rho_t$ for sections where time-dependence is being analyzed.

### Issue: "Structural WD" family in Section 2.2 has weak internal coherence [New/major]
- **Location:** Section 2.2, final paragraph ("Structural WD")
- **Quote:** "Defazio (2025) proposes a layer-balancing framework... Truong & Truong (2026) analyze norm-hierarchy transitions..."
- **Problem:** The "Structural WD" category groups two very different works: Defazio's layer-balancing (which is primarily an analysis of $\rho$ dynamics) and Truong & Truong's norm hierarchy (which concerns representation quality, not WD specifically). These do not constitute a coherent family in the way "alignment-aware" or "temporal scheduling" do. Defazio belongs in Section 2.4 as the closest prior work; Truong & Truong may not warrant a standalone mention if not directly related to WD scheduling.
- **Fix:** Merge the "Structural WD" paragraph into Section 2.4 (for Defazio) or Section 2.1 (for Truong & Truong as a dynamics effect). Do not list it as a fourth WD "family" alongside the other three, as it is not a comparable category.

### Issue: D'Angelo "never functions as explicit regularization" remains too strong [New/minor — partially fixed from prior round]
- **Location:** Section 2.1, paragraph 1
- **Current text:** "D'Angelo et al. (NeurIPS 2024) showed that WD does not function primarily as explicit regularization in modern deep learning"
- **Status:** This was softened from the previous round ("never functions" → "does not function primarily") — this is the correct fix per prior Issue m1.
- **Remaining concern:** The qualifier "primarily" is still slightly ambiguous. More precise: "showed that WD's dominant effect in modern deep learning is dynamics modification (loss stabilization for SGD, bias-variance tradeoff for LLMs) rather than explicit regularization in the classical sense."

---

## Notation and Glossary Compliance (Revision Check)

| Term | Expected | Found | Status |
|------|----------|-------|--------|
| "dynamic WD" (umbrella) | "dynamic WD" | Section 2.2 opener | ✓ |
| "adaptive WD" (state-feedback only) | reserved for PMP-WD | Section 2.2 correctly restricts | ✓ |
| "constant WD" | lowercase | "constant WD" / "constant baseline" | ✓ |
| "Phi invariance" | not used in Related Work | not used | ✓ (appropriate) |
| "alignment" | "gradient-weight alignment" | used as "alignment" alone in 2.2 | ⚠ minor |
| AdamW | capital A, capital W | "AdamW" | ✓ |
| "CWD (hard)" vs. "CWD" | "CWD (hard)" in taxonomy, "CWD" in prose | "CWD" in prose | ✓ (acceptable) |
| "state-of-the-art" | banned | not found | ✓ |
| "novel" without quantification | banned | not found | ✓ |
| "significant" without p-value | banned | not found | ✓ |

**Minor notation issue:** Section 2.2 uses "alignment" alone (e.g., "sign-alignment mask") where the glossary specifies "gradient-weight alignment." Not a serious issue but worth flagging.

---

## Cross-Reference: review.md Issues Applicable to Related Work

- **Kuzborskij & Abbasi-Yadkori (2025) citation year:** review.md Section "Notation & Terminology" flags this as "not in the outline's reference list." The citation now has a clarifying note attached but the year (2025) has not been verified in the paper. Authors should confirm this reference exists and the year is correct.
- **Terminology consistency with Section 5.1:** review.md notes "Phi invariance under AdamW" usage in Section 5.1/5.2. Related Work does not use this term (correct — Phi invariance is a result, not a background concept), but the transition from Related Work to Method/Experiments should eventually introduce it.

---

## What Works Well

1. **The four-family taxonomy structure** maps cleanly onto the Phi framework's modulation axis classification (Table 4 in Section 3.1). A reader can directly trace each family to its column in the taxonomy table.
2. **Section 2.3 (Evaluation Fragmentation)** is now substantive and well-argued. The three concrete examples of incomparable protocols provide a strong motivation for the paper's controlled design.
3. **Section 2.4's PMP connection to Li & Tai (2017)** correctly positions this paper as applying PMP to WD rather than learning rates. The feedforward/state-feedback distinction is precise and important.
4. **CWD challenge is now immediate.** The phrase "Section 3.3 shows this interpretation does not account for the stability cost" means a reader of Related Work knows upfront that the paper will challenge CWD's claims — this sets correct expectations before the Theory section.

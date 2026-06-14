# Related Work Section Critique

**Section**: 2. Related Work
**Reviewer**: Section Critic (Automated)
**Score**: 7 / 10

---

## Overall Assessment

The Related Work section is comprehensive and well-organized, covering five distinct lines of prior work with clear positioning of the paper's contributions. The writing is technically precise and demonstrates strong command of the rapidly evolving MDLM literature. However, the section suffers from several structural and presentation issues that reduce its effectiveness as a reader-friendly survey.

---

## Issues

### CRITICAL

**C1. No visual elements despite outline specification.**
The outline's Figure & Table Plan does not assign any figure/table to the Related Work section, and the section itself contains none. However, the section explicitly references "Table 1 in Section 3" (paragraph 5, Section 2.3) — this cross-reference is appropriate but the section itself would benefit from a compact comparison table (e.g., a taxonomy table grouping prior methods by category: remasking / search / RL / trained memory / continuous representations / CFG / TTT, with columns for training-free?, cross-step memory?, compute overhead). This would dramatically improve readability for a 1.5-page section that currently lists 25+ papers in dense prose.

- **Severity**: Critical
- **Suggestion**: Add a small taxonomy table (Table 0 or fold into Table 1 in Section 3) summarizing the landscape of inference-time MDLM methods. At minimum, include: Method | Category | Training-free? | Cross-step Info? | Overhead.

### MAJOR

**M1. Section 2.1 conflates model overview with the paper's contribution framing (paragraph 2).**
The final paragraph of Section 2.1 (line 9) introduces the "information island problem" and positions BSD and A-CFG as solutions. This is contribution framing, not related work. It blurs the boundary between surveying prior work and making claims. The term "information island problem" is attributed to MetaState but rebranded by the authors ("we term") — this contradiction should be clarified.

- **Severity**: Major
- **Location**: Line 9, "Despite these advances..." paragraph
- **Suggestion**: Move the contribution positioning sentence ("Our BSD and A-CFG methods directly address this limitation...") to the end of the full Related Work section or to the Introduction. Keep the information island problem description but clarify attribution: either MetaState coined it or the authors did — not both.

**M2. Section 2.2 embeds experimental results that belong in Section 4.**
Line 15 states: "our experiments confirm that pure remasking strategies — ReMDM-conf (4.4%) and RCR (5.7%) on Countdown-500 — fail to surpass vanilla Dream-7B (4.7%)." Including specific experimental numbers in Related Work is unusual and creates redundancy with the Experiments section. Similarly, line 17 mentions "wd1: MATH500 44.2%, GSM8K 84.5%" — mixing the paper's own results with others' reported numbers in the same subsection is confusing.

- **Severity**: Major
- **Location**: Lines 15, 17, 19
- **Suggestion**: In Section 2.2, describe what remasking/RL methods do and their limitations conceptually. Reserve the paper's own experimental comparisons (4.4%, 5.7% etc.) for Section 4. The wd1 numbers are fine since they are cited results, but the paper's own results should be clearly deferred.

**M3. Section 2.4 includes novel experimental findings that constitute results, not related work.**
Lines 43 contains two full paragraphs of the paper's own findings: "Our A-CFG experiments on Dream-7B reveal two surprising findings..." with detailed analysis of JSD stability scores (~0.997) and CFG scheduling failures. This is a results/discussion contribution, not a survey of prior work.

- **Severity**: Major
- **Location**: Line 43, "Our A-CFG experiments..." paragraph
- **Suggestion**: Reduce to a brief forward reference: "We apply A-CFG to Dream-7B and report surprising departures from prior findings in Section 4." Move the detailed analysis of JSD stability degeneracy and scheduling failure to the Experiments or Discussion section.

**M4. Inconsistent citation format and missing formal references.**
Several works are cited only by arXiv ID without author names (line 29: "EvoToken-DLM (arXiv 2601.07351)", "Soft-Masked Diffusion (IBM; arXiv 2510.17206)", "CADD (arXiv 2510.01329)", "CCDD (arXiv 2510.03206)"). This is inconsistent with the rest of the section which uses Author (Year) format. Additionally, "Gemini Diffusion" (line 7) has no citation at all.

- **Severity**: Major
- **Location**: Lines 7, 29
- **Suggestion**: Add proper author-year citations for all referenced works. If author information is unavailable, use the first author's surname from the arXiv paper. Add a citation for Gemini Diffusion or remove the claim.

### MINOR

**m1. Section 2.5 is disproportionately short and its placement feels abrupt.**
At only 2 paragraphs (lines 47-49), Section 2.5 on Test-Time Training is significantly shorter than other subsections. The DTA results (6.2% vs vanilla 12.5%) are again experimental findings embedded in Related Work.

- **Severity**: Minor
- **Location**: Lines 47-49
- **Suggestion**: Either expand with more TTT/DTA context (e.g., mention the 6 TTT variants from Appendix C) or merge into Section 2.2 as a "Parameter adaptation" subcategory to maintain consistent subsection granularity.

**m2. Dense listing style in Sections 2.2 and 2.3 reduces readability.**
Sections 2.2 and 2.3 read as annotated bibliography lists rather than narrative surveys. For example, Section 2.3 (line 29) lists 7 works in a single paragraph with minimal connecting prose. Each work gets a one-sentence description but the relationships between them are not discussed.

- **Severity**: Minor
- **Location**: Lines 13-23, 27-33
- **Suggestion**: Group works by the insight they share rather than listing them sequentially. For Section 2.3, organize around the key design dimension: what triggers the transition from continuous to discrete (KL divergence for LRD, convergence for ReMix, step count for EvoToken, etc.).

**m3. The closing positioning statement (line 23) is weak.**
"Our work is distinguished from all the above by operating at the representation level (BSD) and prediction level (A-CFG) simultaneously" — this claim appears before discussing Sections 2.3-2.5, so "all the above" only refers to Section 2.2 methods. It should be moved to the end of the entire Related Work section.

- **Severity**: Minor
- **Location**: Line 23
- **Suggestion**: Remove the premature summary at the end of Section 2.2. Add a consolidated positioning paragraph at the end of the full Related Work section that distinguishes the paper from all surveyed lines of work.

**m4. "cf." usage is informal (line 9).**
"the information island problem (cf. MetaState; Xia et al., 2026)" — "cf." means "compare/contrast" but here it seems to mean "as identified by" or "following." This is a minor but potentially confusing usage.

- **Severity**: Minor
- **Location**: Line 9
- **Suggestion**: Replace with "as identified by MetaState (Xia et al., 2026)" or "following MetaState (Xia et al., 2026)" depending on the intended meaning.

**m5. The FIGURES comment block is empty (line 51-53).**
The HTML comment `<!-- FIGURES - None -->` is a placeholder that should be removed or populated.

- **Severity**: Minor
- **Location**: Lines 51-53
- **Suggestion**: Remove the empty comment block before final compilation. If a taxonomy table is added per C1, reference it here.

---

## Score Justification: 7/10

**Strengths** (contributing to the score):
- Comprehensive coverage of 25+ relevant works across 5 well-defined categories
- Clear identification of the paper's novelty relative to MetaState (training-free vs. trained memory)
- Good use of specific numbers from prior work to contextualize contributions
- Logical subsection organization following the paper's two main contributions (BSD → 2.1/2.3, A-CFG → 2.4)

**Weaknesses** (reducing the score):
- Boundary violation: ~30% of the section presents the paper's own experimental results rather than surveying prior work (-1.5)
- No visual elements: a taxonomy table would significantly aid comprehension for this dense section (-0.5)
- Inconsistent citation format and missing references (-0.5)
- Dense listing style reduces narrative flow (-0.5)

A revised version addressing M1-M4 (moving own results to Experiments, adding a taxonomy table, fixing citations) would score 8.5-9.

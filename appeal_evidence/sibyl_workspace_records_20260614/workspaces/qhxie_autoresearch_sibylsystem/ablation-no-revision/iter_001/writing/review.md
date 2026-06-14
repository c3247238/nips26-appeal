# Writing Quality Review

## Summary

This paper tests whether answer consistency, framed as "activation energy" ($E_a$), can predict whether a mathematical reasoning problem is solvable in a single pass by a language model. The authors confirm that aggregate accuracy follows exponential Arrhenius-like saturation ($R^2 = 0.924$) and that $E_a$ correlates with MATH difficulty level (Spearman $\rho = 0.448$), but they falsify the critical routing hypothesis: $E_a$ has zero predictive power for single-pass correctness (AUC = 0.436, $\rho = -0.063$). The paper argues that consistency measures answer stability, not correctness, and quantifies an irreducible ~25 percentage point error floor for consistency-based routing.

## Detailed Assessment

### Structural Coherence: 7/10

The paper follows a clear logical arc: problem motivation (Introduction) → theoretical framework (Activation Energy Theory) → experimental protocol → hypothesis-by-hypothesis results → diagnosis and implications → conclusion. The argument structure is sound: descriptive predictions (H1, H2) are confirmed before the prescriptive prediction (H3) is falsified, creating a narrative of "promising framework, but routing fails."

The abstract accurately represents the paper's content and results. Transitions between sections are generally smooth, though the Discussion (Section 6) is long and could benefit from sub-section tightening. The Related Work section (Section 2) is well-positioned before the theory section and effectively establishes the gap.

One structural issue: the paper has no separate "Related Work" section after the results (the outline originally planned Section 7 as Related Work, but the visual audit notes this was merged into Section 2). This is the correct decision, but the outline still references Section 7, which no longer exists in the paper. The paper itself is consistent on this point.

### Notation & Terminology Consistency: 8/10

Notation is largely consistent with `notation.md` and `glossary.md`. Key symbols ($E_a$, $P_\infty$, $k_0$, $c_0$) are defined before first use and used consistently throughout.

**Specific violations:**

1. **"Arrhenius kinetics" vs. "Arrhenius-like kinetics"**: The abstract uses "Arrhenius-like kinetics" but Section 5.1 heading uses "Aggregate Arrhenius Saturation." The glossary prefers "Arrhenius-like kinetics" and explicitly avoids "Arrhenius equation" and "Arrhenius law." The section heading should be "Aggregate Arrhenius-Like Saturation" for consistency.

2. **"Confirmed" vs. "confirmed" in table**: Table 7 uses "Confirmed (aggregate only)" and "Confirmed (modest effect)" with inconsistent capitalization compared to "**Falsified**" (bolded). Standardize to lowercase with bold for falsified only, or bold all status entries.

3. **$c_0$ definition inconsistency**: In Section 3.2, $c_0$ is defined as "the answer consistency at $k = 16$ samples." But in the formula $E_a = -\ln(c_0)$, $c_0$ is used without the subscript $i$ for problem index, while $c_{i,k}$ is defined with problem index. This is minor but could confuse readers about whether $E_a$ is per-problem or aggregate.

4. **"percentage points" abbreviation**: The glossary specifies "pp" (e.g., "25pp gap"), but the paper uses "25 percentage points" in Section 6.3 and "~8pp" is not used at all. Standardize to "pp" for consistency with the glossary.

### Claim-Evidence Integrity: 7/10

Most claims are well-supported with specific numbers. The paper is commendably honest about negative results and caveats.

**Specific issues:**

1. **Critical: H3 summary misrepresents the source data**. The `analysis_h3_summary.md` reports H3 status as "CONFIRMED" with the note "Low-Ea accuracy (75.0%) meets threshold (75%). Ea is a useful routing signal." This directly contradicts the paper's narrative that H3 is falsified. The paper correctly interprets the AUC = 0.436 and Spearman = -0.063 as evidence of falsification, but the source summary file says "CONFIRMED." The paper's interpretation is correct (the 75% threshold pass is a post-hoc artifact), but this creates a documentation inconsistency that could confuse readers or reviewers. The paper should explicitly address this discrepancy --- the 75% figure is achieved only through post-hoc threshold optimization, which the paper does note, but the contrast with the "CONFIRMED" source label is jarring.

2. **"80% of problems fail to fit"**: This claim appears in Section 5.1 and the abstract. The `analysis_h1.json` shows per-problem median $R^2 = 0.000$ for all three models, but does not explicitly state "80% fail to fit." The paper should clarify the criterion for "fails to fit" (e.g., $R^2 < 0.1$ or failure to converge).

3. **"~25 percentage points" irreducible error floor**: This is calculated as 100% - 75.0% = 25pp. But the 75.0% figure is from a post-hoc threshold. The paper correctly notes this is a "statistical artifact," yet still uses it to compute the error floor. The logic is internally consistent ("at best, 75%"), but the phrasing could be clearer that this is an upper bound on achievable performance, not a measured floor.

4. **"ACAR reports an ~8pp ceiling"**: No citation is provided for ACAR. The paper should add a citation or remove this comparison if the source cannot be located.

5. **Missing citation for Li (2026)**: The error depth hypothesis is cited as [10] but no bibliography is included in the paper. The paper references [1]--[11] but there is no reference list. This is a critical omission for any paper intended for external review.

### Visual Communication: 7/10

The paper plans 6 figures and 9 tables, which is a healthy ratio. All figures and tables are referenced before they appear, per the visual audit. Captions are self-explanatory and state key takeaways.

**Issues:**

1. **Figure 1 is referenced but not described in the Introduction**: The text says "Figure 1 shows this saturation pattern" but the figure itself is not included in the markdown (only a reference to `fig1_teaser.pdf`). For a writing review, I cannot assess whether the figure actually shows what the text claims. The visual audit confirms the PDF exists, but I cannot verify its content.

2. **Figure 6 caption mismatch**: The text says "Figure 6 visualizes the routing signal landscape" but the figure reference is `figures/hypothesis_summary.pdf`. The caption in the markdown says "Routing signal comparison. Consistency measures agreement but not correctness..." This is consistent in content but the filename (`hypothesis_summary.pdf`) does not match the figure number (Figure 6). The visual audit does not flag this, but it could cause compilation issues.

3. **Table 8 (Error Classification) lacks counts/proportions**: The table has three error types with descriptions and examples, but no quantitative data (e.g., "Execution errors: 12/20 failures"). The outline planned Table 4 (Error Classification) to include "count/proportion," but the paper only provides qualitative descriptions. This is a significant gap --- the Discussion claims "Execution errors dominate low-$E_a$ failures" but provides no numbers to support this.

4. **Missing appendix content**: The outline plans Appendix sections (per-problem fit statistics, $E_a$ distribution, full problem data, extraction audit) but these are not present in the paper. For a pilot-scale study, the full problem data table would strengthen reproducibility claims.

### Writing Quality: 6/10

The writing is generally clear and direct, with good use of specific numbers. The paper avoids most banned patterns.

**Banned patterns found:**

1. **"Moreover"**: Not found. Good.

2. **"Furthermore"**: Not found. Good.

3. **"It is worth noting that"**: Not found. Good.

4. **"In recent years"**: Not found. Good.

5. **"To the best of our knowledge"**: Not found. Good.

6. **Vague "significantly improves" without numbers**: Not found. Good.

**Unclear or problematic sentences:**

1. **Section 3.2, paragraph 2**: "The product $P_\infty \cdot k_0$ has units of samples and can be interpreted as a problem-difficulty parameter: larger values indicate slower convergence toward the ceiling." This is confusing because $P_\infty$ is dimensionless (probability) and $k_0$ has units of samples, so the product has units of samples, not "samples." More importantly, the claim that "larger values indicate slower convergence" is misleading: for fixed $P_\infty$, larger $k_0$ means slower convergence, but the product $P_\infty \cdot k_0$ conflates ceiling and rate. A problem with high $P_\infty$ and low $k_0$ could have the same product as one with low $P_\infty$ and high $k_0$ but very different behavior. This sentence should be removed or clarified.

2. **Section 5.3**: "The decisive evidence is AUC = 0.436 and Spearman = $-0.063$. These values prove $E_a$ cannot predict single-pass solveability." The word "prove" is too strong for statistical evidence. The glossary recommends "confirmed" and "falsified" rather than "proven" and "disproven." Replace "prove" with "demonstrate" or "show."

3. **Section 6.1**: "The bimodal distribution of $E_a$ values (clusters at approximately 9.47 and 10.0) reflects two stability modes, not two correctness modes." This is a good insight, but the next sentence says "Problems with $E_a \approx 9.47$ ($c_0 \approx 0.106$) show moderate consistency." This is incorrect: $c_0 = e^{-9.47} \approx 7.7 \times 10^{-5}$, not 0.106. The notation.md says $c_0 \approx 0.106$ corresponds to $E_a \approx 2.24$ (since $-\ln(0.106) \approx 2.24$). There is a serious numerical inconsistency here. Wait --- re-checking: the paper says $E_a = -\ln(c_0)$, so $c_0 = e^{-E_a}$. For $E_a = 9.47$, $c_0 = e^{-9.47} \approx 7.7 \times 10^{-5}$. But the problem data shows $c_0 = 0.10565$ for $E_a = 9.465$. This is inconsistent with the formula $E_a = -\ln(c_0)$, since $-\ln(0.10565) \approx 2.247$, not 9.465. **There is a critical inconsistency between the formula and the data.** The paper must resolve this: either $E_a$ is computed differently (e.g., $E_a = -\ln(c_0)$ with $c_0$ being something other than consistency fraction), or there is a data error. This is a **Critical** issue.

4. **Section 6.3**: "The 25pp gap represents the maximum theoretical improvement from perfect routing; any real-world system will perform worse due to threshold estimation error, distribution shift, and other confounds." This is unclear. If perfect routing knows the ground truth, the improvement from perfect routing over the current low-$E_a$ accuracy would be 25pp (100% - 75%). But the text says "maximum theoretical improvement from perfect routing" which could be misread. Clarify that 25pp is the gap between the observed low-$E_a$ accuracy and perfect accuracy, not the improvement over a baseline.

5. **Section 6.5, paragraph on bimodal artifacts**: "The near-zero variance of $E_a$ at Level 5 ($\sigma \approx 1.9 \times 10^{-6}$) suggests algorithmic saturation rather than genuine consistency." This is good, but the paper does not explain what "algorithmic saturation" means or why it occurs. A brief explanation would help.

## Issues for the Editor

1. **[Critical] $E_a$ formula inconsistency**: Section 3.2 defines $E_a = -\ln(c_0)$, but the data shows $E_a = 9.465$ with $c_0 = 0.10565$, which violates this formula ($-\ln(0.10565) \approx 2.25$, not 9.465). The notation.md Usage Note 3 says $E_a \approx 9.47$ corresponds to $c_0 \approx 0.106$, which is mathematically impossible under $E_a = -\ln(c_0)$. **Fix**: Verify the actual formula used to compute $E_a$ in the experiment code and correct either the formula or the data throughout the paper. This is a showstopper for external review.

2. **[Critical] Missing bibliography**: The paper cites [1]--[11] but has no reference list. **Fix**: Add a References section with full citations for all numbered references.

3. **[Major] Table 8 lacks quantitative data**: The error classification table (Section 6.2) claims execution errors "dominate" but provides no counts or proportions. **Fix**: Add a "Count" or "Proportion" column to Table 8 with actual numbers from the error analysis. If the error analysis was not performed quantitatively, either perform it or soften the claim to "we hypothesize that execution errors dominate."

4. **[Major] H3 source data contradiction**: The `analysis_h3_summary.md` labels H3 as "CONFIRMED" while the paper labels it "Falsified." The paper's interpretation is correct (post-hoc artifact), but the contradiction between source summary and paper narrative could confuse reviewers. **Fix**: Add a footnote or parenthetical in Section 5.3 explicitly noting that the raw threshold test passes but the AUC/Spearman evidence overrides this, and update the source summary file to reflect the correct interpretation.

5. **[Minor] Section 3.2 product interpretation**: The sentence about $P_\infty \cdot k_0$ as a "problem-difficulty parameter" is misleading. **Fix**: Remove or rephrase to avoid conflating ceiling and rate parameters.

## What Works Well

1. **Honest reporting of negative results**: The paper does not try to spin the AUC = 0.436 into a positive finding. Sections 5.3 and 6.1 clearly state that $E_a$-based routing fails, and the abstract leads with the negative result. This is scientifically admirable and reviewer-friendly.

2. **Strong caveats**: The "Critical caveat" in Section 5.1 ("The exponential pattern is a group-level statistical regularity, not a universal physical law") and the limitations section (6.5) show intellectual honesty and prevent overclaiming.

3. **Clear figure-table referencing**: Every visual element is referenced before it appears, and captions state key takeaways. The visual audit confirms this discipline was maintained throughout.

SCORE: 7

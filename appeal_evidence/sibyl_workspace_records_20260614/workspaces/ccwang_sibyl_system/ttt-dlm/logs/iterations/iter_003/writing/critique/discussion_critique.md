# Critique: Discussion Section

**Reviewer**: Section Critic Agent
**Section**: 6. Discussion (discussion.md)
**Score**: 7/10

---

## Summary

The Discussion section covers six subsections: why remasking fails (6.1), DTA's task dependence (6.2), DMI as a practical contribution (6.3), lessons from 18 iterations (6.4), limitations (6.5), and future directions (6.6). The writing is generally clear and technically grounded, with strong mechanistic explanations for negative results. However, several issues reduce its effectiveness — notably a reliance on incomplete data combined with strong claims, redundancy with the Analysis section, a missing visual element, and an overly long limitations list that could be more concisely organized.

---

## Issues

### CRITICAL

1. **Self-contradictory use of pilot evidence (Section 6.2 vs 6.4 Lesson 2)**
   - Section 6.2 opens by calling DTA's task-dependent effectiveness "a central finding of this work" and uses pilot-scale results (N=16) as primary evidence: "+12.5pp" on MBPP, "-12.5pp" on GSM8K. Yet Section 6.4 Lesson 2 explicitly states pilot results "systematically overestimate effect sizes" with ~24pp mean inflation. The Discussion cannot simultaneously use pilot results as evidence for strong claims AND argue those same results are unreliable. This logical tension will be immediately noticed by reviewers.
   - **Location**: Section 6.2, paragraph 1 ("A central finding of this work...")
   - **Suggestion**: Either (a) downgrade the framing from "central finding" to "preliminary hypothesis supported by pilot-scale indicators" with explicit cross-reference to Lesson 2, or (b) remove the specific pilot-scale numbers from 6.2 and focus on the mechanistic argument for why DTA should be task-dependent (MLM loss vs. task correctness).

2. **Incomplete full-scale results undermine the narrative arc (Section 6.2, para 1)**
   - The section acknowledges that "DTA full-scale results remain pending" for Countdown, meaning the paper's most novel method (DTA) lacks definitive full-scale evidence on the primary benchmark. Combined with pilot-only GSM8K and MBPP results, this means the task-dependence discussion — which occupies a full subsection — rests entirely on data the paper itself warns against trusting. The paper needs to either complete these experiments or restructure the Discussion to present DTA task-dependence as a clear hypothesis rather than a confirmed finding.
   - **Location**: Section 6.2, paragraph 1

### MAJOR

3. **Significant redundancy with Analysis section (5.2)**
   - Section 6.1 repeats token-level diagnostic numbers verbatim: correction precision 31.3%, unstable positions 94.8, SCP precision 76.9%. Approximately 60% of 6.1 restates quantitative findings from Section 5.2 rather than providing new interpretive depth. The Discussion should synthesize and elevate, not recapitulate.
   - **Suggestion**: Condense 6.1 by roughly half. Keep the conceptual insight (the bolded paragraph about token-space vs. parameter-space propagation of structural understanding) and the Information Island connection, but replace repeated statistics with references to "Section 5.2" and "Table X."

4. **Missing Figure 7 reference in text (Section 6.2)**
   - The outline's Figure & Table Plan specifies Figure 7 (task_sensitivity.pdf) for this section, and the HTML comment at the end confirms it. However, the figure is never referenced in the body text. A visual comparison of DTA effectiveness across Countdown/GSM8K/MBPP would significantly strengthen the task-dependence argument and reduce the wall-of-text feel of 6.2.
   - **Suggestion**: Add an explicit reference in the opening paragraph of 6.2: "Figure 7 summarizes DTA's task-dependent effectiveness across our three benchmarks (pilot-scale)." Ensure the figure is generated and included.

5. **DMI universality claim is overclaimed (Section 6.3 and 6.6)**
   - Section 6.3 calls DMI "an attractive candidate for deployment as a default enhancement in any DLM inference pipeline," and 6.6 compares it to "nucleus sampling in autoregressive models." This is a sweeping claim based on a single benchmark (Countdown-500). DMI has not been validated at full scale on GSM8K, MBPP, or any non-reasoning task. Additionally, the overhead figure is inconsistent (see Minor issue #9).
   - **Suggestion**: Temper the universality language in 6.3 to "promising on Countdown-500" and reserve the broader claim for Future Directions (6.6), clearly framed as a research question.

6. **Limitations list is structurally unorganized (Section 6.5)**
   - Six numbered items without grouping or prioritization. Items 1 and 2 overlap substantially (both concern incomplete full-scale evaluation at different granularity). Item 6 (theoretical assumptions) is conceptually distinct from the empirical limitations but is not separated.
   - **Suggestion**: Group into: (a) Evaluation scope (merge items 1-2 into one), (b) Methodological limitations (items 3-5), (c) Theoretical limitations (item 6). This reduces repetition and improves readability.

### MINOR

7. **Lesson 4 (implementation bugs) is tangential**
   - While honest and valuable, the temperature annealing bug anecdote (Section 6.4, Lesson 4) describes a specific implementation error in earlier iterations that was caught and corrected. It reads more like a lab diary entry than a generalizable methodological lesson. The general principle ("implementation verification is important") could be stated in one sentence rather than a full paragraph. Consider moving the detailed account to Appendix F (DTA tuning history).

8. **Missing quantitative support for the "structural understanding" argument (Section 6.1, bolded paragraph)**
   - The key conceptual contribution of 6.1 — "token-space operations cannot propagate structural understanding across denoising steps" — is stated as a bold assertion but supported only by the negative evidence (remasking fails). The positive evidence exists: Section 5.4 shows mutual information I(Delta_theta; x_0) monotonically increases for DTA. This should be cited here to complete the argument.
   - **Suggestion**: Add a sentence after the bolded paragraph: "The information accumulation analysis (Section 5.4) provides positive evidence for this view: DTA's LoRA parameters show monotonically increasing mutual information with the target sequence, confirming that parameter-space updates do accumulate structural understanding."

9. **Inconsistent overhead numbers for DMI**
   - Section 6.3 states DMI's overhead as "~1.05x," but Table 1 (Section 4.2) reports DMI time as 4.3s vs. vanilla 3.7s, which is ~1.16x. Section 6.5 item 4 states DTA as "4-5x" while Section 4.2 shows ~4x. These discrepancies, though minor, invite reviewer skepticism.
   - **Location**: Section 6.3 paragraph 1, Section 6.5 item 4.
   - **Suggestion**: Use the exact ratios from Table 1 consistently throughout.

10. **SCP comparison uses approximate numbers when exact ones are available (Section 6.3)**
    - "SCP (~7x cost, ~8.4% accuracy)" uses tildes for both. Since full-scale SCP results were run with 3 seeds, exact mean and std should be reported rather than approximations.

11. **Redundancy between Section 6.3 and 6.6 on DMI universality**
    - The "DMI as a universal DLM enhancement" paragraph in 6.6 repeats the claim from 6.3 almost verbatim, including the nucleus sampling analogy. State the claim once and cross-reference.

12. **Citation density is uneven**
    - Sections 6.2, 6.3, 6.4, and 6.5 contain zero citations. For a Discussion section, more engagement with related work is expected. For example: (a) cite Nisonoff et al. when discussing why their remasking approach fails, (b) cite MetaState (Xia et al.) when discussing DMI's relationship to cross-step memory, (c) cite TTT literature when discussing DTA's task dependence.

---

## Strengths

- The mechanistic explanation of why remasking fails (6.1) is insightful and well-reasoned, particularly the connection between confidence calibration and correction precision, and the concept of "token churning" degrading conditioning context.
- The "Lessons from 18 iterations" subsection (6.4) demonstrates commendable scientific transparency. Lessons 1-3 are genuinely useful for the DLM community and rarely seen in published work.
- The Information Island framing is consistently maintained throughout, providing good conceptual unity across the entire paper.
- The limitations section (6.5) is thorough and honest, covering all major weaknesses.
- Section 6.6 proposes concrete, actionable future directions (structured self-supervision, hybrid DTA+verifier) rather than vague handwaving.

---

## Score Justification: 7/10

The Discussion section demonstrates strong analytical thinking and scientific honesty. The conceptual framework (token-space vs. parameter-space interventions, Information Island problem) is well-articulated and the lessons from negative results are a genuine contribution. However, the score is held back by: (1) a critical logical contradiction between using pilot results as evidence for "central findings" while simultaneously arguing those results are unreliable (CRITICAL), (2) incomplete full-scale data for the paper's most novel method undermining the task-dependence narrative (CRITICAL), (3) significant redundancy with the Analysis section (MAJOR), and (4) a missing figure and overclaiming on DMI universality (MAJOR). Resolving the two CRITICAL issues — either by completing full-scale experiments or by restructuring the narrative to frame pilot-based conclusions as hypotheses — and trimming the redundancy would raise this to an 8-9.

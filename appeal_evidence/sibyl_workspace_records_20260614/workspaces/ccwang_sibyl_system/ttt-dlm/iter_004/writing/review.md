# Final Critic Review: Beyond Remasking — Continuous Belief States and Classifier-Free Guidance for Inference-Time Reasoning in Masked Diffusion Language Models

**Reviewer:** Sibyl Final Critic (NeurIPS/ICML-level holistic review)
**Date:** 2026-03-10

---

## 1. Summary

This paper identifies the "information island problem" in masked diffusion language models (MDLMs) — the discarding of rich distributional predictions after argmax sampling at each denoising step — and proposes two training-free inference-time methods: Belief-State Diffusion (BSD), which replaces mask embeddings with EMA-accumulated belief states, and Adaptive Classifier-Free Guidance (A-CFG), which uses confidence-based re-masking with dual forward passes. A simpler predecessor, Diffusion Memory Injection (DMI), is the only method validated at full scale (Countdown-500, 3 seeds, p < 0.05), achieving ~2x improvement (9.3% vs. 4.7%). BSD and A-CFG are evaluated only at pilot scale (n=16). The paper reports extensive negative results constraining the MDLM inference-time scaling design space.

---

## 2. Novelty and Significance (5/10)

**Strengths:**
- The "information island problem" framing is intuitive, well-articulated, and provides a unifying lens for understanding why several inference-time approaches fail. Although independently identified by MetaState (Xia et al., 2026), the training-free perspective is a meaningful differentiator.
- BSD's conceptual design — full mask-embedding replacement with EMA-accumulated belief vectors, two-phase architecture — is cleanly formulated and distinct from prior mixed-representation approaches (LRD, ReMix, EvoToken), even if the core mechanism (soft embedding mixtures) is shared.
- The extensive negative results (JSD stability failure, CFG scheduling failure, BSD+A-CFG non-composability, parameter-space adaptation failure) are a genuinely valuable empirical contribution. The non-composability finding — that representation-layer and prediction-layer methods interfere rather than compose — is a novel design principle with potential broader applicability.
- The taxonomy of failure modes (parameter-space → structural → cross-step signal → within-step representation/prediction) is a useful organizational contribution for the community.

**Weaknesses:**
- **A-CFG is not novel.** The paper explicitly acknowledges it was introduced by Arriaga et al. (2025) for LLaDA-8B. Applying it to Dream-7B without modification is an empirical transfer experiment, not a methodological contribution. Yet the paper's title and abstract give A-CFG co-equal billing with BSD, creating a misleading impression of two novel methods.
- **DMI is very simple** (fixed-ratio mixing of logit-weighted embeddings) and belongs to a concurrent family (LRD, ReMix, Soft-Masked Diffusion, EvoToken). The paper acknowledges DMI "belongs to this family" (Section 2.3), weakening the novelty claim. DMI's contribution is narrowly practical rather than conceptually significant.
- **BSD underperforms DMI** on Countdown and degrades on GSM8K (18.8% vs. 25.0% vanilla). A generalization of a method that performs worse than its special case across all evaluated settings is a weak positive contribution. The paper would benefit from explaining why the more flexible method fails to match its simpler predecessor.
- The absolute performance levels are low: the best validated result is 9.3% accuracy on Countdown-500, meaning the model fails >90% of the time. While relative improvements matter, the paper does not convincingly argue why these results are significant for the field beyond documenting failure modes.

---

## 3. Technical Soundness (4/10)

**Strengths:**
- The BSD algorithm (Algorithm 1) is precisely specified with clear mathematical notation, L2 normalization justification, and design decision documentation.
- The statistical protocol is commendably rigorous for full-scale experiments: McNemar's test with Bonferroni correction, bootstrap 95% CIs, Cohen's h effect sizes.
- The explicit separation of full-scale (Table 2) and pilot-scale (Table 3) results avoids a common conflation trap. The authors repeatedly caveat pilot results as "directional evidence, not confirmed improvements."
- The hypothesis verification table (Table 5) is a model of scientific transparency — 4 of 11 hypotheses falsified, with forthright reporting.

**Critical Weaknesses:**

- **Fundamental underpowering of central claims.** BSD and A-CFG — the paper's two headline contributions — have ONLY been evaluated at n=16. At this sample size with ~5% base rate, the expected correct count is 0.8 per run. The differences between 0/16, 1/16, and 2/16 correct are literally 1-2 samples. All bootstrap 95% CIs include zero. **No meaningful statistical inference is possible.** The paper acknowledges this but then proceeds to draw extensive mechanistic conclusions (Sections 5.1-5.3) that the data cannot support.

- **Pilot-to-full-scale calibration is alarming.** Vanilla scores 0% on Countdown-16 but 4.7% on Countdown-500; DMI scores 12.5% on pilot but 9.3% at full scale. The paper notes "pilot-to-full-scale effect size shrinkage" and documents up to 24pp shrinkage across iterations. Given this track record, A-CFG's pilot 12.5% could easily shrink to match or trail vanilla at full scale. This is not idle speculation — it is the established empirical pattern of this research program.

- **The entropy-accuracy correlation (r = 0.78, p < 0.001) at n=16 is suspect.** With only 1/16 correct samples in the BSD condition, this correlation is effectively driven by one data point's entropy value relative to 15 others. The reported p < 0.001 from 16 binary-outcome data points warrants careful scrutiny. The paper should provide the raw (entropy, correctness) pairs so reviewers can verify this claim.

- **Compute-fair analysis undermines the methods.** Table 4 shows that at matched FLOPs on the pilot, vanilla step-scaling is competitive with every proposed method. The paper argues DMI is different because of full-scale results, but this comparison is between DMI (full-scale) and other methods (pilot-only), creating an apples-to-oranges argument. If vanilla at 256 steps matches A-CFG at 128 steps + dual forward pass, A-CFG's value proposition collapses to "same accuracy, same compute, more complexity."

- **BSD's $k$-parameter sensitivity is concerning.** Only $k_\text{frac} = 0.75$ works; $k_\text{frac} \leq 0.50$ yields 0% accuracy. This means only 25% of steps use the "belief phase" — BSD's distinctive feature — while 75% revert to standard hard-token denoising. The conceptual claim that "continuous belief evolution improves reasoning" is substantially weakened when the method requires minimizing the belief phase to function at all.

- **Inconsistency in BSD performance.** BSD gets 6.2% on Countdown-16 (vs. DMI's 12.5%) and 18.8% on GSM8K-16 (vs. vanilla's 25.0%). The generalization of DMI performs strictly worse than the special case. The paper offers post-hoc explanations but does not address the fundamental issue: why should anyone use BSD over DMI?

---

## 4. Clarity and Presentation (7/10)

**Strengths:**
- The paper is well-organized with a logical flow: problem identification → methods → experiments → discussion → conclusion.
- Algorithm 1 (BSD) is unambiguous and implementable.
- The related work section (Section 2) is comprehensive, covering five relevant research lines with appropriate positioning and fair comparisons.
- Table 1 (continuous-representation method comparison) effectively positions BSD within the landscape.
- The discussion section (Section 5) is the paper's intellectual highlight — each failure mode receives a thoughtful mechanistic explanation. The analysis of why BSD+A-CFG fail to compose (representation-space vs. prediction-space interference) is particularly insightful.

**Weaknesses:**
- **The abstract overpromises.** It presents BSD's ρ = −0.95 and r = 0.78 and A-CFG's +12.5pp without qualification, giving readers scanning the abstract no indication these come from n=16 samples with CIs including zero. The mismatch between abstract confidence and actual evidence level is significant.
- **A-CFG prominence is misleading.** A-CFG appears in the title and is framed as a co-equal contribution, but it is a direct application of prior work. The paper would be more honest to frame A-CFG as an evaluation/transfer experiment rather than a proposed method.
- **DMI — the only validated method — is relegated.** Full description is in Appendix D, while the unvalidated BSD gets the most detailed treatment in Section 3. This is structurally inverted: the strongest evidence supports the least-developed method.
- **Figures are text descriptions only.** For a final review, rendered figures are essential. The six proposed figures sound informative but cannot be evaluated in their current form.
- **The paper is long relative to its empirical content.** The discussion of failure modes spans 4 pages but is built on n=16 data. Condensing Sections 5.1-5.3 would make room for full-scale validation.
- Minor: entropy-accuracy correlation reported as r = 0.784 (Section 4.6) and r = 0.78 (abstract, Section 3.5). Use consistent precision throughout.

---

## 5. Experimental Rigor (3/10)

This is the paper's most critical weakness.

**Strengths:**
- DMI's full-scale validation (500 samples × 3 seeds, McNemar p < 0.05) meets rigorous standards.
- Ablation studies are systematic, varying one factor at a time with consistent evaluation protocol.
- Flip analysis (examining method disagreement patterns) goes beyond aggregate accuracy.
- Both accuracy and generation quality metrics (rep-3, distinct-3) are reported.
- The hypothesis-driven design with pre-registered hypotheses (Table 5) is commendable.

**Weaknesses:**
- **Two of three proposed methods lack meaningful-scale validation.** This is a fatal flaw for a top venue. BSD and A-CFG are each evaluated on 16 samples — insufficient to distinguish any of the observed differences from noise.
- **All ablation studies use n=16.** Every ablation comparison (BSD k-parameter, alpha schedule, A-CFG guidance weight, temporal schedule) compares 0/16, 1/16, or 2/16 correct. These identify gross working-vs.-not-working distinctions at best, but cannot support claims about relative performance or optimal hyperparameters.
- **Only two benchmarks, one pilot-only.** Countdown-500 is the sole full-scale benchmark; GSM8K uses only 16 samples from a benchmark with 1,319 test problems. Top venues expect broader evaluation (MATH, ARC, or at minimum full GSM8K).
- **Single model (Dream-7B).** All experiments use one MDLM. The JSD stability degeneracy, temporal scheduling failure, and BSD+A-CFG non-composability could all be Dream-7B-specific. LLaDA-8B evaluation is essential, especially since A-CFG was originally designed for it and reported dramatically different results (GSM8K 73.5 on LLaDA vs. directional +12.5pp on Dream).
- **No comparison with strongest concurrent methods.** The paper discusses wd1 (GSM8K 84.5%), Prism, and d1 in related work but does not compare against any of them, even on shared benchmarks. While these require training, their existence contextualizes the 4.7-9.3% accuracy range.
- **Missing qualitative error analysis.** Beyond flip analysis, there is no examination of what kinds of problems each method solves or fails on. Do BSD, DMI, and A-CFG solve overlapping or complementary problem subsets?
- **A-CFG re-masking percentage (m=10%) is not ablated.** This is a critical hyperparameter but its sensitivity is never examined.
- **Reproducibility gap.** With accuracy at 0-12.5% on 16 samples, different pilot sample selections or random seeds could yield entirely different conclusions. The pilot sample selection procedure (which 16 from Countdown? which 16 from GSM8K?) is not described.

---

## 6. Reproducibility (6/10)

**Strengths:**
- Algorithm 1 is fully specified and implementable.
- Hyperparameter settings are clearly tabulated.
- Single-GPU setup (RTX PRO 6000 Blackwell, 98GB VRAM) is precisely documented.

**Weaknesses:**
- No code availability mentioned.
- Pilot sample selection procedure undescribed.
- Complete generation hyperparameters (top-k, top-p beyond temperature=0.4) not fully specified.
- Appendices C (DTA) and D (DMI) are referenced but not included in this draft.

---

## 7. Minor Issues

1. Section 3.2 asserts Dream-7B "requires early hard token anchors" to explain BSD's k-parameter sensitivity. An alternative hypothesis — that the BSD implementation introduces distributional shift at longer belief phases — should be explicitly considered and ideally ruled out.
2. Table 3: BSD+A-CFG shows "---" for GSM8K-16 — why was this combination not evaluated on GSM8K?
3. Section 4.5: Vanilla at 256 steps achieves 12.5% vs. A-CFG at 128 steps also at 12.5%. This directly undermines A-CFG's practical value at the pilot scale where it has been tested.
4. The claim that DMI is "immediately deployable in any MDLM inference pipeline" (Section 5.4, Conclusion) is overclaimed given DMI showed zero benefit on GSM8K-16. "Immediately deployable for structured arithmetic reasoning" would be more precise.
5. Section 2.1 mentions "LLaDA 2.0" (100B scale) and "LLaDA 2.1" — these should include proper citations or be clearly labeled as contemporaneous/not-yet-published if no citation exists.

---

## 8. Questions for Authors

1. **Why was full-scale validation not performed for BSD and A-CFG?** DMI was validated at full scale on Countdown-500. What prevented the same evaluation for the paper's two primary proposed methods?
2. **Entropy-accuracy correlation data:** Can you provide the 16 individual (terminal entropy, correctness) data points? With 1/16 correct in the BSD condition, the r = 0.78 claim needs transparent verification.
3. **Vanilla at 256 steps on Countdown-500:** Has this been evaluated? This directly determines whether A-CFG offers any advantage at matched compute at full scale.
4. **A-CFG on LLaDA-8B:** Has this been tested in your setup for comparison with the original paper's much stronger results (GSM8K 73.5)?
5. **BSD k_frac > 0.75:** Have you tested k_frac = 0.85 or 0.90 (even shorter belief phases)? The trend suggests accuracy might continue increasing with shorter belief phases.
6. **What is the variance across 3 seeds for DMI's full-scale result?** Is 9.3% ± 1.4% the standard deviation, standard error, or 95% CI?

---

## 9. Overall Assessment

This paper addresses a real and important problem — inference-time scaling for MDLMs — with a well-motivated conceptual framework (the information island problem). The scientific transparency is exemplary: pre-registered hypotheses with honest failure reporting, explicit separation of validated vs. pilot results, and appropriate statistical caveats. The negative results and design space analysis constitute a genuine intellectual contribution.

However, the paper has a fundamental gap between its ambitions and its evidence:

1. **BSD**, the most conceptually novel contribution, has not been shown to work at meaningful statistical scale (1/16 correct), underperforms its own special case (DMI), and degrades on GSM8K.
2. **A-CFG**, the most promising pilot result, is not novel (direct application from Arriaga et al., 2025), matches vanilla step-scaling at equal compute, and has not been validated beyond n=16.
3. **DMI**, the only validated method, is extremely simple, belongs to a concurrent family of similar ideas, and shows zero benefit on GSM8K.
4. The mechanistic analyses in Sections 5.1-5.3 are insightful but built on n=16 data that cannot support the level of inference drawn.

The paper reads as an honest interim report from an iterative research program rather than a finished study with validated conclusions. The negative results alone would make a strong contribution to a workshop or "lessons learned" venue, but the positive contributions do not yet meet the bar for a top-venue main track paper.

---

## 10. Recommendation

**Verdict:** Below acceptance threshold for NeurIPS/ICML main track.

**Confidence:** 4/5 — High. The statistical limitations are unambiguous and the experimental gaps are concrete.

**Key reasons for score:**
- Both headline methods (BSD, A-CFG) lack statistically meaningful evaluation (n=16 only)
- The only validated method (DMI) is simple and concurrent with similar ideas
- A-CFG is prior work applied to a new model, not a novel contribution
- Compute-fair analysis undermines the methods' value proposition
- Single model, two benchmarks (one pilot-only), no comparison with strongest concurrent work

**Actionable path to acceptance:**
1. **Full-scale 3-seed validation** of BSD and A-CFG on Countdown-500 and full GSM8K — this is the immediate priority and would fundamentally change the paper's strength.
2. **Multi-model evaluation** — at minimum Dream-7B + LLaDA-8B, to determine whether findings are model-specific.
3. **Reframe the paper** to position it as a systematic empirical study of MDLM inference-time scaling, with DMI as the practical contribution and BSD/A-CFG as promising directions pending validation, rather than a methods paper claiming three headline contributions.
4. **Tighten the abstract and title** to accurately reflect the evidence level — currently the abstract reads as if BSD and A-CFG are validated methods.
5. **Include rendered figures** and complete appendices (C, D) for reviewability.

SCORE: 4

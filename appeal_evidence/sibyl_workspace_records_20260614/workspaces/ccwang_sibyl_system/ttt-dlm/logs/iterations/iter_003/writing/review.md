# Final Critic Review: Denoising-Time Adaptation (DTA)

**Reviewer**: Final Critic (NeurIPS/ICML-level holistic review)
**Date**: 2026-03-10

---

## Review Summary

This paper proposes Denoising-Time Adaptation (DTA), a training-free method that adds online LoRA parameter updates within the denoising loop of masked diffusion language models (MDLMs), framing MDLM denoising as an explicit test-time training (TTT) opportunity. It introduces the Information Augmentation Spectrum (Vanilla < DMI < SCP < DTA) as an ablation framework, provides a variational interpretation (VDTA) with ELBO monotonicity claims, and evaluates on Countdown-500, GSM8K, and MBPP using Dream-7B and LLaDA-8B. The simplest method, DMI, achieves ~2x improvement on Countdown-500 at near-zero cost, while the headline method DTA shows a null result (4.8% vs. 4.7% vanilla) at full scale on the same benchmark.

---

## 1. Novelty and Significance (6/10)

**Strengths:**
- The conceptual connection between MDLM denoising and TTT is genuinely novel and well-articulated. The observation that each denoising step performs masked language modeling on a progressively revealed sequence---precisely the self-supervised objective TTT exploits---is a clean and insightful contribution that opens a new research direction.
- The Information Augmentation Spectrum is a useful analytical framework for reasoning about cross-step information transfer in DLMs. It provides a principled way to ablate the value of different types of memory.
- DMI (Diffusion Memory Injection) is a simple, practical contribution with near-zero overhead and meaningful gains---the kind of finding that could see immediate adoption.
- The token-level diagnostic metrics (Correction Precision/Recall, trajectory stability) are novel and useful for the DLM inference-time scaling community.

**Weaknesses:**
- The headline method (DTA) does not work on the primary benchmark at full scale: 4.8% vs. 4.7% vanilla on Countdown-500 is a null result. The paper's title and entire framing center on DTA, but the actual empirical contribution is DMI, a much simpler method that does not involve parameter updates or the TTT connection.
- The novelty of DMI itself is limited---soft embedding injection from previous-step logits is a straightforward engineering idea, not a conceptual breakthrough.
- The only positive DTA result (MBPP +12.5pp) is pilot-only (N=16) and the paper itself explicitly warns that pilot results inflate effect sizes by ~24pp on average (Section 6.4, Lesson 2).

---

## 2. Technical Soundness (5/10)

**Strengths:**
- The E-step/M-step decomposition is clean and Algorithm 1 is well-specified.
- Design decisions are carefully motivated with pilot experiments: zero initialization, mask-and-predict vs. self-consistency loss, cumulative decay, AdamW vs. SGD.
- Statistical methodology is appropriate when applied: 3 seeds, McNemar tests, bootstrap CIs, Bonferroni correction.

**Weaknesses:**

**2a. VDTA theoretical framework has significant gaps.**
- Proposition 1 assumes $\mu$-strong convexity of $\mathcal{L}_{\text{DTA}}$ in $\Delta\theta$, "ensured by $L_2$ regularization via weight decay." However, weight decay only adds a convex regularizer to a non-convex neural network loss---the sum is not necessarily strongly convex. The paper acknowledges this in Section 6.5 but still claims "ELBO monotonicity" prominently in the abstract and contributions. This is misleading.
- The proof sketch is incomplete: it asserts E-step improvement ("moves toward higher-likelihood regions") without addressing the discrete sampling operation that breaks differentiability. Standard EM convergence arguments require continuous optimization, not discrete token sampling.
- The information accumulation claim (mutual information monotonicity) is presented with theorem-like formatting but is acknowledged as empirical. The supporting evidence---that more tokens are revealed at each step and prediction confidence increases---is trivially true of any denoising process regardless of DTA.

**2b. Self-reinforcing error risk is insufficiently analyzed.**
- The M-step masks revealed tokens and re-predicts them, but these are the model's own committed predictions. The paper dismisses the circularity concern by noting that "re-predicting after masking requires integrating the full bidirectional context." However, if the model is confidently wrong (common in constrained arithmetic), bidirectional context will likely reproduce the same error. This circularity plausibly explains DTA's null result on Countdown.

**2c. Orthogonality claim is empirically falsified.**
- Section 3.4 claims DTA and remasking are "orthogonal" and "naturally complementary." However, DTA+ReMDM achieves 3.6% on Countdown-500---worse than either DTA alone (4.8%) or ReMDM-conf alone (4.4%). This direct contradiction is not adequately explained.

---

## 3. Clarity and Presentation (7/10)

**Strengths:**
- The writing is generally clear, well-organized, and flows logically from background to method to experiments.
- The "Information Island" framing is effective. The editor analogy ("an editor who has lost all notes") is memorable and apt.
- Algorithm 1 is clear and reproducible.
- Honest reporting of 18 iterations of negative results (Section 6.4) is commendable and scientifically valuable.

**Weaknesses:**
- **The abstract oversells DTA** relative to actual results. It highlights "DTA's strongest gains on code generation (MBPP: +12.5pp)" without adequate hedging that this is a 16-sample pilot. A reader skimming the abstract would conclude DTA works well, which is misleading given the full-scale null result.
- **Three figures are TODO placeholders** (Figures 3, 4, 5) and one table is TODO (Table 5). This is a significant incompleteness.
- **Narrative-evidence mismatch.** The paper builds toward DTA as the climax of the spectrum, but the results show DMI (the simplest method) wins. The paper should be restructured to match its evidence.
- The term "training-free" is used for a method that performs gradient-based parameter updates at inference time. While technically "no pre-training required," this terminology is confusing. The paper should clarify.
- Section 4.4 presents universal degradation across architectures as "cross-model verification," which is more accurately "cross-model failure replication."

---

## 4. Experimental Rigor (4/10)

This is the paper's most critical weakness.

**Strengths:**
- Full-scale Countdown-500 evaluation with 3 seeds for 4/7 methods is properly executed.
- Token-level diagnostics (Correction Precision/Recall, trajectory stability) are insightful new metrics.
- The pilot-to-full-scale inflation analysis (~24pp mean inflation, Section 6.4) is a valuable methodological contribution.

**Weaknesses:**

**4a. Only one benchmark has full-scale results, and DTA shows null effect there.**
- DTA achieves 4.8% vs. 4.7% vanilla on Countdown-500---effectively zero improvement at 4x computational cost.
- GSM8K and MBPP are pilot-only (N=16). The paper's strongest DTA claim (MBPP +12.5pp) relies entirely on the pilot scale the paper itself discredits.
- SCP results are "interim" (~150/500 samples per seed), adding further incompleteness.

**4b. Missing critical comparisons.**
- No comparison with MetaState (Xia et al., 2026), which directly addresses the same "Information Island" problem with a trained cross-step memory module. Even if MetaState requires training, the performance comparison is essential.
- No comparison with CORE on the evaluated benchmarks, despite CORE achieving +9.2pp on MBPP with LLaDA-8B---potentially stronger than DTA's unvalidated pilot claim.

**4c. Incomplete ablations.**
- Table 5 is TODO. Section 5.3 reports qualitative findings without quantitative tables at full scale.
- No full-scale ablation on warmup fraction, mask ratio, or learning rate.

**4d. Missing statistical tests.**
- The paper promises McNemar tests with Bonferroni-corrected bootstrap 95% CIs (Section 4.1) but the results tables report only mean +/- std. No p-values or confidence intervals appear in Tables 1-4.
- The conclusion claims "$p < 0.05$" for DMI without showing the test in the results section.

**4e. Inference-time scaling experiment is uninformative.**
- Table 4 is entirely pilot-scale (N=16), producing wild non-monotonic patterns (vanilla: 12.5% at T=128, 0.0% at T=256, 6.2% at T=512). The paper acknowledges this but still devotes a full section to it.

---

## 5. Reproducibility (7/10)

**Strengths:**
- Algorithm 1 is fully specified with all hyperparameters.
- Hardware (RTX PRO 6000 Blackwell), generation config (128 steps, temperature 0.4, origin sampling), and LoRA config are clearly documented.
- Seeds are reported (42, 123, 456).
- The design decision section explains failed alternatives.

**Weaknesses:**
- No code release mentioned.
- DMI and SCP implementation details are in prose, not pseudocode.
- The "origin" sampling strategy is referenced by name but not formally defined.

---

## 6. Minor Issues

1. Table 1 mixes pilot-scale timing with full-scale accuracy without clear labeling.
2. The variational interpretation uses increasing superscripts $\Delta\theta^{(t+1)}$ while denoising goes from T to 0---the indexing convention is inconsistent.
3. Figure 6 is referenced as "task_sensitivity.pdf" but its completeness status is unclear.
4. DTA at 0.0% on LLaDA Countdown (Table 3) is buried rather than discussed prominently.
5. The paper references \citep commands in Markdown; ensure proper LaTeX compilation for submission.

---

## 7. Questions for Authors

1. What explains the DTA+ReMDM degradation (3.6%) below both DTA (4.8%) and ReMDM-conf (4.4%), given the orthogonality claim?
2. Can you provide qualitative examples where DTA produces different outputs from vanilla? The aggregate metrics show near-identical accuracy.
3. How does DTA behave when early predictions are substantially wrong? Does the adapter compound errors?
4. Why not compare against MetaState, which targets the identical "Information Island" problem?

---

## Overall Assessment

The paper contains a genuinely insightful conceptual contribution: the structural analogy between MDLM denoising and test-time training, and the principled Information Augmentation Spectrum framework. DMI is a practical finding with real utility. The honest treatment of negative results is scientifically commendable.

However, critical gaps prevent acceptance:

1. **The headline method (DTA) shows a null result at full scale** (4.8% vs. 4.7% vanilla on Countdown-500). The paper is titled "Denoising-Time Adaptation" but the evidence supports DMI, a much simpler method disconnected from the TTT framework.

2. **The theoretical contribution is overstated.** Proposition 1's strong convexity assumption does not hold for neural networks. The proof sketch is incomplete, and the information accumulation claim is asserted rather than proven.

3. **Experimental evaluation is substantially incomplete.** Three TODO figures, one TODO table, SCP interim results, all cross-benchmark and cross-model results at pilot-only scale, missing critical comparisons (MetaState, CORE), and missing statistical tests.

4. **Pilot-scale results dominate the narrative despite documented unreliability.** The paper's own Lesson 2 warns about ~24pp pilot-to-full-scale inflation, yet most positive DTA claims are based on N=16 experiments.

5. **Framing-evidence mismatch.** The paper promises parameter-level adaptation but delivers embedding-level injection as the working method.

---

## Recommendation

**Major Revision Required.** The research direction is promising but the paper is not ready for a top venue. Priority actions:

1. Complete full-scale DTA evaluation on all three benchmarks. If MBPP confirms gains, the narrative strengthens significantly.
2. Add MetaState and CORE comparisons.
3. Complete all TODO figures, tables, and ablations.
4. Provide the promised statistical tests (McNemar, bootstrap CIs).
5. Reframe the paper to match the evidence: either (a) demonstrate DTA works at full scale on code generation, or (b) restructure around DMI and the diagnostic framework as the primary contributions.
6. Weaken theoretical claims to match the level of rigor achieved (conceptual framework, not theorem).

---

SCORE: 4

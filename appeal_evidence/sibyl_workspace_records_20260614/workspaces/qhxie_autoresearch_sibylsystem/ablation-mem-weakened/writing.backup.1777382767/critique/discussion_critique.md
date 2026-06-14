# Critique: Discussion

## Summary Assessment
The Discussion section provides thoughtful explanations for the null results and meaningful implications for the field. The "What Would Change Our Conclusion?" subsection is a particular strength. However, the section contains a duplicated explanation (also in Results), an unsupported implication claim, and misses deeper statistical reflection.

## Score: 6/10
**Justification**: Good structure with strong forward-looking content, but duplication with Results, an overreached implication claim, and thin statistical reflection hold it back. Tightening unique contributions and adding statistical depth would raise to 7-8.

---

## Critical Issues

### Issue 1: Near-Duplicate Content with Results Section
- **Location**: Section 6.1 (all four explanations)
- **Problem**: The four explanations in 6.1 (Low absorption variance, Steering robustness, Metric sensitivity, Task-specific resilience) are nearly identical to the four explanations in Results 5.5. The Discussion should synthesize and deepen, not repeat. For example, 6.1's "Steering robustness" paragraph and 5.5's point 2 share the same core argument about decoder direction bypassing the encoder.
- **Fix**: Remove the duplication. In Results 5.5, keep only a brief bullet list. In Discussion 6.1, expand each explanation with deeper analysis: connect to prior literature, discuss mechanism-level details, and explore interactions between explanations.

### Issue 2: "Interpretability Illusion May Be More Benign" Is Overstated
- **Location**: Section 6.2, paragraph 3
- **Quote**: "the 'interpretability illusion' may be more benign than feared---at least for these tasks"
- **Problem**: This conclusion overreaches. The study tests only GPT-2 Small with first-letter features on two tasks. Generalizing from this to "the interpretability illusion may be more benign" is not warranted. The Discussion should be more circumspect about scope.
- **Fix**: Qualify heavily: "For GPT-2 Small SAEs and the specific tasks we test, the concerns raised by Korznikov et al. about feature recovery may be less acute for practitioners focused on steering and probing. Whether this holds for larger models or other tasks remains unknown."

---

## Major Issues

### Issue 3: Missing Statistical Reflection
- **Location**: Section 6.1
- **Problem**: The Discussion never addresses what the null results mean statistically. With n=26 and low variance, the study may be underpowered to detect small but real effects. A proper discussion would calculate the minimum detectable effect size and discuss whether the observed correlations (e.g., r=-0.301 at layer 8) could reflect a true small effect that the study lacked power to confirm.
- **Fix**: Add a paragraph: "Statistical power considerations temper our interpretation. With n=26 features, our study has approximately 65% power to detect |r| >= 0.50 at alpha=0.05. The observed r=-0.301 at layer 8 corresponds to a small-to-medium effect (Cohen's guidelines) that falls below our detection threshold. We cannot distinguish between (a) a true zero effect and (b) a small true effect that our sample size and variance constraints failed to detect."

### Issue 4: Pilot Comparison Is Underanalyzed
- **Location**: Section 6.3
- **Quote**: "Doubling the sample size reduced variance in the steering success estimates, producing a clearer trend that nonetheless remains below the significance threshold."
- **Problem**: The pilot comparison is interesting but underanalyzed. The pilot had 50 samples/feature; the full study has 100. The H1 correlation strengthened from r=-0.153 to r=-0.301. What does this suggest about the true effect size and required sample size? A simple power calculation would show how many samples would be needed to detect r=-0.301.
- **Fix**: Add: "If the true effect at layer 8 is r=-0.30, approximately 85 features would be needed for 80% power at alpha=0.05. This suggests that even with doubled sample size, our study remains underpowered for small-to-medium effects."

### Issue 5: Missing Discussion of Absorption Metric Validity
- **Location**: Section 6.1
- **Problem**: The "Metric sensitivity" explanation notes that differential correlation may not capture the right phenomenon, but the Discussion doesn't explore what alternative metrics might reveal. The proposal mentioned SAEBench's ablation-based metric and JumpReLU SAEs showing stronger absorption under alternative metrics.
- **Fix**: Expand the metric sensitivity discussion: "The Chanin differential correlation metric assumes that absorption manifests as child features activating when the parent is absent. Alternative metrics, such as SAEBench's ablation-based measure, quantify absorption by measuring reconstruction degradation when child features are suppressed. These metrics may capture different failure modes, and a multi-metric approach would clarify whether our null result is robust across absorption definitions."

### Issue 6: Implications for Architectural Innovation Is One-Sided
- **Location**: Section 6.2, paragraph 2
- **Quote**: "If absorption does not significantly degrade steering or probing, the field may be over-investing in solutions to a non-problem."
- **Problem**: This is the paper's strongest claim, but it's one-sided. Even if absorption doesn't degrade steering/probing, it might degrade other tasks (circuit finding, model editing) that the paper didn't test. The Discussion should acknowledge this limitation when making the "over-investing" claim.
- **Fix**: Add qualification: "For steering and probing specifically, our results suggest that the field's substantial investment in absorption-reducing architectures may not yield proportional task-level improvements in these applications. However, absorption may still matter for tasks requiring precise feature isolation, and architectural innovations may have benefits beyond absorption reduction."

---

## Minor Issues

- **Section 6.1, paragraph 1**: "We consider four explanations" -- the number "four" is unnecessary and creates maintenance risk if explanations are added/removed.
- **Section 6.1, "Steering robustness"**: "Feature U at layer 8, with the highest absorption rate in our sample (A(U) = 0.242), achieves 100% steering success at s = 50" -- this is a strong anecdote but n=1. Note that feature H at layer 8 (A=0.190) achieves only 55% success, showing that high absorption doesn't guarantee success but doesn't preclude it either.
- **Section 6.2, paragraph 1**: "Practitioners need not avoid absorbed features for steering or probing in this model family" -- "need not avoid" is awkward. Use "need not exclude" or "can use absorbed features without concern for".
- **Section 6.4**: "Our planned Gemma-2-2B experiments were blocked by gated HuggingFace access" -- this is the third mention of the Gemma access issue (also in Method and Limitations). Consider consolidating.
- **Section 6.4**: "Semantic hierarchy features" -- the outline mentions WordNet specifically, but the Discussion doesn't. Be consistent.

---

## Visual Element Assessment
- [ ] Figures/tables match outline plan -- N/A for discussion
- [x] All visuals referenced before appearance -- N/A
- [x] Captions are self-explanatory -- N/A
- [ ] No text-heavy sections that need visual support -- A figure showing "conditions that would change our conclusion" as a decision tree or flowchart would be visually compelling and useful.

---

## What Works Well
1. **"What Would Change Our Conclusion?" is excellent**: This subsection is rare in ML papers and demonstrates scientific maturity. The four conditions (larger models, semantic features, alternative metrics, different tasks) are specific and testable.
2. **Pilot comparison adds value**: Section 6.3 shows that the authors learned from pilot data and adjusted their analysis, which increases confidence in the methodology.
3. **Task-specific resilience explanation is insightful**: The observation that steering and probing aggregate information across tokens/latents, making them robust to single-feature failure, is a genuinely interesting mechanistic insight.

---

## Revision Notes (Post-Fix)

The following critical issues from this critique have been addressed in the revised sections:

- Power analysis corrected: ~65% power (was 80%)
- Model size fixed: 124M parameters (85M non-embedding)
- 'Need not avoid' → 'can use without concern'

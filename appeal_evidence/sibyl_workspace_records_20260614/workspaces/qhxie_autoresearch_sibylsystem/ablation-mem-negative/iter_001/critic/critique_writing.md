# Writing Critique: Feature Absorption in Sparse Autoencoders

## Overall Assessment

The manuscript is well-structured and clearly written, with honest limitation disclosure that is exemplary. However, several writing issues inflate the perceived contribution and obscure methodological weaknesses.

---

## Critical Issues

### 1. Abstract Overclaims DFDA Improvement

The Abstract states: "a dynamic feature de-absorption (DFDA) module that improves reconstruction MSE by 11.1% with <0.4% parameter overhead."

**Problem**: This omits the critical qualifier "per-pair residual" that appears in the paper text. The actual improvement is on per-pair residual MSE at 10^-6 scale (absolute improvement ~10^-7), not overall reconstruction MSE. The Abstract's phrasing makes DFDA sound like it improves the SAE's overall reconstruction by 11%, which is false.

**Fix**: "improves per-pair residual MSE by 11.1% with 388 total parameters"

---

### 2. "Collision Rate is a Poor Proxy" Overgeneralizes

The Conclusion states: "Collision rate is a poor proxy for absorption harm."

**Problem**: The paper never measures "absorption harm" directly. It measures sparse probing accuracy on 26 first-letter concepts. Generalizing from this narrow task to "absorption harm" broadly is an unsupported leap.

**Fix**: "Collision rate does not correlate with first-letter sparse probing accuracy on GPT-2 Small, suggesting limited predictive value for this specific downstream task."

---

### 3. Post-Hoc Power Analysis is Misleading

Section 3.6: "With n = 6 layers or k values, the study has approximately 20% power to detect a medium effect size (r = 0.5) at alpha = 0.05."

**Problem**: Post-hoc power analyses are methodologically questionable because they are mathematically related to the observed p-value. A non-significant result always yields low post-hoc power, making this statement tautological rather than informative.

**Fix**: Replace with confidence intervals for the correlation coefficients, or report the minimum detectable effect size for 80% power given the sample size.

---

### 4. Gemma-2-2B Fallback is Buried

The outline and methodology position Gemma-2-2B as the primary model, but all experiments use GPT-2 Small. The explanation ("GemmaScope experiments were blocked by API issues") appears only in Section 3.3.

**Problem**: Readers skimming the Introduction may assume results are on Gemma-2-2B. This is a significant deviation that should be prominent.

**Fix**: Add a sentence in the Introduction: "All reported experiments use GPT-2 Small; Gemma-2-2B experiments were blocked by SAELens API compatibility issues."

---

### 5. Terminology: "Absorption" vs "Collision" is Still Slippery

While the paper has improved terminology consistency, the title still uses "Feature Absorption" while the experiments measure "collision rate." The Abstract uses "absorption" 4 times and "collision" 3 times, blurring the distinction.

**Problem**: The paper measures collision (multiple concepts sharing a feature), not absorption (parent suppressing child). Using "absorption" in the title and framing inflates the perceived relevance.

**Fix**: Consider retitling: "Feature Collision in Sparse Autoencoders: A Cross-Architecture Benchmark."

---

## Minor Issues

1. **"[anonymous repository]" placeholder** in Conclusion needs a real URL or removal.
2. **Glossary inconsistencies**: "pre-trained" vs "pretrained", "training-free" vs "SAE-retraining-free".
3. **Section 5.1**: "CAAB uses collision rate as the primary metric, but collision rate is measurable and may not measure harm" --- this meta-commentary is awkward.
4. **"Our conclusion"** in Conclusion is still slightly colloquial; "In summary" is more standard.

---

## Strengths

1. **Honest limitation disclosure** (Section 5.5) is exemplary---5 major limitations listed with future work mapping.
2. **Clear RQ-contribution mapping** makes the paper easy to follow.
3. **Bolded key finding in Abstract** effectively communicates the main result.
4. **Two revision rounds** have addressed most previously flagged issues.

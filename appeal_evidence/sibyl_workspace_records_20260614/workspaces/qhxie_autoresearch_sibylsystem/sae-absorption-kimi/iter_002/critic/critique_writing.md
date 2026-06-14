# Writing Critique: Construct Validity of SAEBench Absorption Metric

## Overall Assessment

The paper is well-structured and clearly written, with strong negative-result reporting and appropriate epistemic humility in the Discussion. However, several writing issues weaken its credibility, including a critical internal contradiction, overclaiming in the abstract, and ambiguous statistical framing.

**Score: 6/10** (would be 8/10 without the critical issues)

---

## Critical Writing Issues

### 1. Internal Contradiction: Encoder vs. Decoder Permutation (CRITICAL)

**Section 3.1 (Method):** "The Random-SAE control permutes the encoder matrix W_enc of the Standard SAE, destroying any learned feature-detection structure while preserving the marginal activation statistics."

**Section 4.5 (Results):** "This exact identity occurs because the absorption formula depends on the SAE encoder output, and the Random SAE only permutes the decoder directions, leaving encoder activations unchanged."

These are directly contradictory. An encoder permutation changes latent activations; a decoder permutation does not. The paper cannot claim both. This contradiction undermines confidence in the experimental implementation and the interpretation of the Random-SAE finding.

**Fix:** Verify the actual implementation and correct whichever description is wrong. If decoder permutation was used, the "degeneracy" interpretation must be reconsidered.

---

### 2. Abstract Overclaiming (MAJOR)

The abstract states: "A Random-SAE control achieves semantic-hierarchy absorption of 0.352, identical to the Standard SAE, despite scoring near-zero (0.030) on first-letter absorption. This dissociation indicates that the semantic-hierarchy adaptation of the SAEBench absorption metric captures artifacts unrelated to learned SAE structure."

This is an overclaim. As noted in the experiment critique, if the Random SAE permutes decoder directions (as Section 4.5 claims), identical encoder outputs produce identical absorption scores by construction. The "dissociation" is expected behavior, not evidence of metric degeneracy.

**Fix:** Soften to: "The Random-SAE control produces identical semantic-hierarchy scores to the Standard SAE, consistent with the metric's dependence on encoder geometry. This suggests the semantic-hierarchy adaptation measures encoder-output properties rather than learned hierarchical structure."

---

### 3. "Inconclusive" vs. "Insufficient" Framing (MAJOR)

The paper repeatedly describes the construct-validity result as "inconclusive" (r=0.463, CI: [-0.389, 0.981]). But with n=7 SAEs and a CI spanning the entire possible range, the result is not merely "inconclusive"---it is **uninformative**. The test lacked adequate statistical power from the outset.

"Inconclusive" implies the test was well-powered but the data were ambiguous. "Uninformative" or "insufficiently powered" implies the test design was inadequate. The latter is more accurate and more honest.

**Fix:** Reframe throughout: "The evidence base is insufficient to evaluate construct validity" rather than "The construct-validity test is inconclusive."

---

### 4. Ambiguous Percentage Claim (MINOR)

Section 5.2: "Mean non-hierarchy absorption (0.331) exceeds mean semantic-hierarchy absorption (0.235) by 0.096 points; non-hierarchy scores are 41% higher than hierarchy scores."

The 41% figure is relative to the hierarchy mean ((0.331-0.235)/0.235 = 0.408), but a reader might misread it as relative to the non-hierarchy mean.

**Fix:** Clarify: "41% higher relative to the hierarchy mean" or use absolute difference.

---

### 5. Burying the Key Observation (MINOR)

Section 3.4 contains a critical observation buried mid-paragraph: "In practice, for all semantic hierarchies in our suite, acc_sae = acc_resid = 1.0, so the absorption score simplifies to (acc_resid - acc_k-sparse) / acc_resid."

This explains why the semantic-hierarchy metric measures only k-sparse loss, not SAE encoding loss. It should be highlighted as a standalone observation or methodological caveat.

**Fix:** Pull this out as a standalone paragraph, highlighted observation, or footnote.

---

## Structural Issues

### Related Work is Overlong

Section 2 (Related Work) is ~800 words, consuming nearly 20% of the body. For a short paper, this is excessive. Subsections 2.4 (Construct Validity in NN Evaluation) and 2.5 (Semantic Hierarchies) could be condensed or merged.

**Fix:** Condense 2.4 and 2.5 into a single "Broader Context" subsection of ~200 words.

### Abrupt Transition from Limitations to Conclusion

Section 5.5 (Limitations) ends with "...are not repeated here" and jumps straight to "This study reports..." in the Conclusion. A one-sentence bridge would smooth this transition.

**Fix:** Add: "Despite these limitations, the findings carry actionable implications for benchmark design."

---

## Strengths

1. **Abstract clarity:** The abstract is a model of concision, stating the problem, method, three key findings, and implication in ~150 words.
2. **Hypothesis-driven structure:** Results map cleanly to pre-registered hypotheses (H1, H2, H3).
3. **Epistemic humility:** Section 5.1 explicitly states "The current evidence base is insufficient to conclude whether first-letter absorption predicts semantic-hierarchy absorption."
4. **Negative result reporting:** The hierarchy specificity failure is reported without hedging or smoothing.
5. **Specific numbers:** Claims are backed by specific statistics throughout.

---

## Banned Patterns Check

| Pattern | Found? | Location |
|---------|--------|----------|
| "Furthermore" | Yes | Section 1.1 |
| "It is worth noting that" | No | -- |
| "In recent years" / "Recently" | No | -- |
| "Moreover" | No | -- |
| Vague "significantly" without numbers | Partial | Section 4.3 (followed by stats) |
| "To the best of our knowledge" | No | -- |

The writing is generally clean with minimal filler.

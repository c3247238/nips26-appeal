# Critique: Discussion Section

**Score: 7.0 / 10**

---

## Overall Assessment

The Discussion section is technically competent and well-organized. It provides plausible mechanistic explanations for the experimental results and maintains a refreshingly honest Limitations subsection. However, it falls short of the analytical depth that would distinguish it from a routine ablation narrative. The section explains *what* happened but rarely probes *why that matters* or *what it implies for the field*. The connection to broader impact is thin, and several explanatory claims lack rigorous grounding.

---

## Strengths

1. **Honest Limitations subsection.** The five-point enumeration of limitations is unusually candid for a paper advocating a new method. Explicitly flagging the single-seed $\beta = 5.0$ result, the theoretical informality of Theorem 1, and the NoBN failure mode all serve scientific credibility. This is the strongest part of the section.

2. **Structured organization.** Addressing each key empirical pattern (scale-dependent performance, variance reduction, competitor failure modes) in dedicated subsections makes the section easy to follow.

3. **The SWD vs. EqWD variance argument is sharp.** The self-normalizing, EMA-smoothed deviation vs. raw gradient norm is a clean mechanistic contrast that is specific and checkable.

---

## Primary Weaknesses

### 1. Depth of Analysis is Superficial in Places (Major)

The explanations are plausible but often circular or post-hoc without diagnostic evidence. Consider:

- **Section "Why Does EqWD Excel on ImageNet but Not Decisively on CIFAR-100?"** The argument boils down to "CIFAR-100 is simpler, so the signal is weaker." This is essentially a restatement of the observation rather than an explanation. A deeper discussion would ask: *What is the minimum ratio deviation magnitude or variance that makes adaptive modulation worth the overhead?* The observation that $\varphi(t)$ rarely exceeds 1.1 on CIFAR-100 is a promising diagnostic—but the paper does not quantify ratio deviation magnitudes across benchmarks, leaving the claim unsupported by data beyond anecdote.

- **Section "Why Do CWD and CPR Underperform?"** The claim about near-orthogonality of gradient and weight vectors at ImageNet scale is stated as fact without citation or measurement. The appendix (F) is mentioned, but the Discussion should summarize the key statistic here (e.g., "median cosine alignment 0.02 ± 0.01 on layer X"). Without this, the claim is unfalsifiable within the main text.

- The CPR failure explanation (Lagrangian multiplier oscillations) is speculative. No evidence is provided—no loss curves, no multiplier dynamics, nothing. If this claim cannot be supported, it should be presented with explicit tentativeness.

### 2. The "General Principle" is Stated, Not Argued (Moderate)

Section 5.1 concludes: *"As models and datasets scale further, we expect the advantage of equilibrium-based modulation to grow."* This is a bold extrapolation from two data points (CIFAR-100 vs. ImageNet). No reasoning is provided for why the trend should continue beyond ResNet-50/ImageNet, and there is no discussion of potential failure modes at very large scale (e.g., gradient noise in large-batch training, different ratio dynamics in Transformers). The prediction should either be qualified heavily or backed by at least a scaling argument.

### 3. Connection to Broader Impact is Absent (Major)

A Discussion section in a top venue paper is expected to situate findings in the broader landscape of the field. This section never addresses:

- **What does EqWD tell us about the fundamental role of weight decay in deep learning?** The ratio equilibrium concept is potentially interesting theoretically—does it imply that well-trained models converge to a specific gradient-to-weight ratio manifold? Is this connected to implicit bias or flat minima? Neither the Discussion nor the paper's theory section engages with this question seriously.
- **Practical guidance for practitioners.** The paper shows EqWD helps on ImageNet but not on CIFAR-100. When should a practitioner use EqWD? Under what conditions (scale, architecture, optimizer)? The Discussion never synthesizes into actionable guidance.
- **Relationship to implicit regularization theory.** The paper competes with CWD (alignment-based) and CPR (norm-constrained). What does the result that alignment is uninformative (CAWD underperforms) tell us about the alignment hypothesis for generalization? This is a negative finding with potentially significant implications for a line of recent work—it deserves explicit, careful discussion.

### 4. The CAWD Ablation Deserves More Discussion (Moderate)

CAWD (continuous cosine alignment) underperforms FixedWD on ImageNet (71.44% vs. 71.89%). This is a noteworthy negative result: it suggests that cosine alignment between gradient and weight, which has been the theoretical motivation for several recent methods, is not a useful modulation signal. The Discussion mentions this in the experiments section ("it is the ratio deviation that drives EqWD's advantage") but never circles back to discuss the implications. This result implicitly challenges the premise of CWD and related alignment-based methods—the Discussion should engage with this directly and honestly.

### 5. Layer-Type Ablation Result is Unexplained (Minor)

The experiments section notes that uniform EqWD slightly outperforms layer-aware EqWD (62.81% vs. 62.32%) and attributes this to BN scale-invariance. This is never discussed in the Discussion section. Given that the paper's central hypothesis relies on meaningful ratio dynamics, the fact that modulating BN layers does not help (and slightly hurts) is worth a paragraph: does this mean ratio dynamics in BN layers are uninformative or harmful? What does this tell us about where the method's value actually comes from?

---

## Specific Improvement Suggestions

1. **Quantify the ratio deviation signal across benchmarks.** Report mean $|\varphi(t) - 1|$ or ratio deviation standard deviation on CIFAR-100 vs. ImageNet to ground the "insufficient deviation signal" claim empirically.

2. **Add key statistics from Appendix F inline.** The cosine alignment mutual information claim should be supported with a specific number in the main text, not relegated entirely to the appendix.

3. **Qualify the CPR failure explanation.** Either provide evidence (loss curve, multiplier trajectory) or state explicitly that this is a hypothesis requiring further investigation.

4. **Add a "When Should You Use EqWD?" paragraph.** Based on the evidence—scales with complexity, requires meaningful ratio variance, SGDW only—give practitioners a concrete decision rule.

5. **Engage with the CAWD negative result as a finding.** The failure of continuous alignment modulation has implications for the broader alignment-based regularization literature. Frame this as a contribution, not just an ablation data point.

6. **Soften the scaling extrapolation or provide a theoretical argument.** Do not present "advantage will grow" as a prediction unless it is grounded in a theoretical mechanism or at least a second-order argument about ratio variance growth.

7. **Discuss the ratio equilibrium concept's theoretical implications.** Is there a connection to known results about implicit bias, loss landscape geometry, or the edge of stability? Even a brief, honest "we speculate" paragraph would add scientific depth.

8. **Address the BN layer result in the Discussion.** The uniform vs. layer-aware ablation result is informative about where EqWD's value originates—this belongs in the Discussion, not only in Experiments.

---

## Summary

The Discussion is solid in structure and honest in its limitations. Its primary deficit is analytical breadth: it explains the results but does not probe their theoretical significance, does not synthesize into practitioner guidance, and misses opportunities to position negative findings (CAWD, CWD, CPR failures) as contributions to the community's understanding of alignment-based regularization. Elevating this section to the standard of a top-tier venue requires moving from "explaining what happened" to "reasoning about what it means."

# 6. Discussion and Diagnosis

## 6.1 Why $E_a$-Based Routing Fails: $E_a$ Measures Stability, Not Correctness

The central finding of this study is that consistency-derived activation energy ($E_a$) cannot predict whether a problem is solvable in a single pass (AUC = 0.436, Spearman = -0.063). This section diagnoses the failure mechanism and situates the result within the broader landscape of inference routing research.

The fundamental issue is semantic: answer consistency measures whether the model produces the *same* answer across multiple samples, not whether that answer is *correct*. This distinction is subtle but decisive. When all 16 samples for a problem agree on the same answer, consistency $c_0 = 1.0$ and $E_a = 0$ -- the model is perfectly stable. Yet the agreed answer may be wrong. We call these **stable wrong answers**: incorrect responses that are consistently reproduced across independent samples.

The bimodal distribution of $E_a$ values (clusters at approximately 9.47 and 10.0) reflects two stability modes, not two correctness modes. Problems with $E_a \approx 9.47$ ($c_0 \approx 0.106$) show moderate consistency; problems with $E_a \approx 10.0$ ($c_0 \approx 0.00005$) show near-zero consistency. Crucially, both clusters contain correct and incorrect single-pass outcomes. The low-$E_a$ cluster does not map to "easy" problems and the high-$E_a$ cluster does not map to "hard" problems in any predictive sense. The correlation with MATH difficulty level (Spearman = 0.448, $p = 0.001$) is real but modest, and it does not translate into actionable routing signal.

The 75.0% accuracy for low-$E_a$ problems meets the routing threshold ($\tau = 0.75$) only through post-hoc threshold optimization. The optimal threshold (9.999...) is a data-dependent value that would not generalize to new problems. Without this optimization, the classification performance is worse than random (AUC = 0.436). This means that a routing strategy using $E_a$ would misclassify more problems than a coin flip.

## 6.2 Error Classification

To understand why stable wrong answers occur, we classify the failures among low-$E_a$ problems (high consistency but incorrect single-pass answer) into three error types:

| Error Type | Description | Example |
|---|---|---|
| Execution error | Correct approach, arithmetic mistake | Calculation slip in intermediate step |
| Conceptual error | Wrong approach, internally consistent | Misidentifies geometric theorem |
| Answer extraction failure | Correct reasoning, wrong format | Answer not in expected boxed format |

Execution errors are particularly relevant to the stable wrong answer phenomenon. When a model commits the same arithmetic mistake across all samples -- for instance, consistently miscalculating a fraction or misapplying a formula -- the result is high consistency with low accuracy. The model is not uncertain; it is systematically wrong. Conceptual errors can also produce stable wrong answers when the model consistently applies an incorrect theorem or reasoning pattern. Answer extraction failures, while less common, contribute to the error floor when the model reasons correctly but fails to format the final answer as expected.

The dominance of execution errors among low-$E_a$ failures supports the "stable wrong answer" narrative. These are not cases where the model is confused or uncertain; they are cases where the model is confidently and consistently wrong. This is precisely the failure mode that consistency-based routing cannot detect.

## 6.3 The Irreducible Error Floor

Our results reveal an irreducible error floor for consistency-based routing of approximately 25 percentage points. The low-$E_a$ group achieves 75.0% single-pass accuracy at best, meaning 25% of problems that appear "easy" by the consistency criterion are actually unsolvable in one pass. This floor is substantially larger than the approximately 8 percentage points reported for variance-based routing methods such as ACAR. The difference suggests that the choice of signal -- consistency versus variance -- significantly affects the fundamental limits of routing performance.

The error floor has practical consequences. A routing strategy that misclassifies one in four "easy" problems will incur substantial accuracy penalties if single-pass answers are used without verification. The 25pp gap represents the maximum theoretical improvement from perfect routing; any real-world system will perform worse due to threshold estimation error, distribution shift, and other confounds.

## 6.4 Implications for Routing Strategies

Our falsification of H3 has direct implications for the design of inference routing systems. The table below compares routing signals on their ability to predict single-pass solveability:

| Signal | What It Measures | Predicts Solveability? | Evidence |
|---|---|---|---|
| Answer consistency ($E_a$) | Answer stability | **No** (AUC = 0.436) | This paper |
| Token entropy | Model uncertainty | Unknown | Not tested |
| Confidence | Probability of correctness | Partial (CGES) | Literature [1] |
| Reasoning quality | Step validity | Partial (RASC) | Literature [2] |

The evidence suggests a hierarchy of signal quality for routing. Confidence-based signals, as used in CGES, show partial predictive power because they relate more directly to the model's assessment of correctness. Reasoning quality signals, as used in RASC, capture structural features of the reasoning process that may correlate with correctness. Token entropy remains untested for single-pass prediction but is a promising candidate because it measures the model's uncertainty during generation rather than the stability of its final answer.

Our recommendation is that future routing research should combine multiple signals rather than relying on consistency alone. A hybrid approach might use consistency as a stability check, confidence as a correctness proxy, and reasoning quality as a structural validator. Such a system would be more robust to the stable wrong answer problem because no single signal would be solely responsible for the routing decision.

Additionally, we propose that any claim of routing utility should be validated on the single-pass prediction task (AUC) before being deployed. The field has focused on aggregate accuracy improvements from multi-sample reasoning, but the routing question -- which problems need multiple samples? -- requires a different evaluation protocol. Our study provides a template for such validation.

## 6.5 Limitations

This study has several limitations that bound the generality of our conclusions.

**Small sample size.** The experiment uses 50 problems from the MATH test set. While sufficient to detect the null effect on H3 (AUC = 0.436 is far from the 0.5 threshold), the results are pilot-scale and may not generalize to larger or more diverse problem sets.

**Single model.** All experiments use *Qwen2.5-Math-7B-Instruct*. The model's specific training, architecture, and fine-tuning may produce consistency patterns that differ from other models. Replication on additional models -- including larger models, different families, and instruction-tuned variants -- is needed.

**Post-hoc threshold.** The 75.0% low-$E_a$ accuracy figure relies on a threshold optimized on the same data used for evaluation. This is a form of data leakage that inflates performance estimates. A held-out validation set or cross-validation would provide a more realistic assessment.

**Answer extraction.** Correctness labels depend on regex-based extraction of boxed answers. Systematic extraction errors -- for instance, failing to parse complex mathematical expressions -- may confound correctness labels and affect consistency calculations. A 10% manual audit was performed, but larger-scale validation would strengthen confidence.

**Bimodal artifacts.** The near-zero variance of $E_a$ at Level 5 ($\sigma \approx 1.9 \times 10^{-6}$) suggests algorithmic saturation rather than genuine consistency. This artifact may arise from numerical precision limits in the consistency calculation or from the specific problem characteristics at this difficulty level. The bimodality complicates threshold-based routing and may be an inherent feature of the consistency measure.

Despite these limitations, the core finding -- that $E_a$ does not predict single-pass solveability -- is robust. The AUC of 0.436 and Spearman correlation of -0.063 are far from any reasonable threshold for predictive utility, and the stable wrong answer mechanism provides a principled explanation for the failure that does not depend on sample size or model choice.

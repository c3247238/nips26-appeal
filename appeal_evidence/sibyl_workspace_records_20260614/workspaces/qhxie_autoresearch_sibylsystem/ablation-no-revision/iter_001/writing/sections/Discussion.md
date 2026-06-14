# 6 Discussion

## 6.1 Why Ea-Based Routing Fails

The falsification of H3 is the central negative result of this work. Activation energy estimated from answer consistency measures **agreement** but not **correctness**. A problem with low activation energy means all samples agree on the same answer — it does not mean that answer is correct. This distinction is fundamental and explains why consistency-based routing cannot achieve reliable single-pass accuracy.

Three failure modes dominate low-$E_a$ problems where routing fails:

**Execution errors** are the primary culprit. The model applies the correct reasoning approach but makes a calculation mistake — miscalculating $7 \times 8 = 54$, dropping a negative sign, or truncating a decimal. These errors are internally consistent: every sample makes the same arithmetic mistake because the error is deterministic given the same input. The answer is wrong, but consistency is 100%. A consistency-based signal assigns low $E_a$ (easy problem) when the model has actually made a systematic error.

**Conceptual errors** represent a fundamentally flawed approach that is internally consistent across all samples. All samples converge on an incorrect answer because the model misinterprets the problem — solving for the wrong variable, applying the wrong formula, or misreading the problem statement. Unlike execution errors, conceptual errors reflect a deeper misunderstanding that no amount of sampling will fix.

**Answer extraction failures** occur when the reasoning is correct but the final answer is formatted incorrectly or the extraction step fails. The model arrives at the correct intermediate result but outputs it in the wrong format or misses the final step entirely. Again, all samples extract the answer the same way, producing high consistency for an incorrect final output.

The consequence is a **fundamental ceiling** on consistency-based routing: even with an arbitrarily low $E_a$ threshold, single-pass accuracy cannot exceed the model's actual single-pass error rate, which includes all three failure modes. The ceiling is not determined by routing signal quality — it is determined by the model's inherent error distribution on single-pass inference.

## 6.2 Implications for Routing Strategies

Our findings challenge the implicit assumption in reasoning-aware inference methods that answer consistency is a reliable proxy for problem solvability. Table 5 compares routing signals by what they measure and whether they predict solveability.

**Table 5: Routing signal comparison.**

| Signal | Measures | Predicts Solveability? | Evidence |
|--------|----------|------------------------|----------|
| Answer consistency ($E_a$) | Agreement among samples | **No** | H3 falsified: 68.4% < 75% |
| Token entropy | Uncertainty in generation | Not yet tested | — |
| Self-consistency confidence | Probability mass on majority | Not yet tested | — |
| Attention-based features | Reasoning depth | Not yet tested | — |

The routing problem requires signals that capture **reasoning quality** — not merely answer agreement — in a single pass. Token entropy measures uncertainty about what to generate next; if low entropy correlates with correct reasoning, it could serve as a routing signal. Attention patterns may reveal whether the model is "checking its work" or making systematic errors. Per-layer activations could capture whether the model has committed to a reasoning path.

A key insight from our analysis is that routing signals must be **single-pass** — they cannot rely on multiple samples to estimate consistency, because the routing decision must be made before samples are generated. This constraints the design space: we need signals computable from one forward pass, not from comparing multiple outputs.

Figure 6 visualizes this by contrasting consistency-based routing (which failed) with alternative single-pass signals that remain untested.

![Routing signal comparison: consistency measures agreement (Y-axis) but not correctness (X-axis). Ea-based routing attempts to use agreement as a proxy for correctness, but execution errors create a ceiling. Alternative signals — token entropy, attention patterns, per-layer activations — capture reasoning quality directly and may avoid this limitation. Conceptual diagram illustrating the fundamental mismatch between what consistency measures and what routing requires.](figures/hypothesis_summary.pdf)

## 6.3 Relationship to Prior Work

Our exponential saturation result (H1) corroborates the probabilistic inference scaling theory of Yang et al. (arXiv:2508.16456), who derive the same exponential saturation form under a different theoretical framing. Both approaches predict that accuracy follows a diminishing-returns curve as sampling count increases. We cannot claim the exponential saturation model as novel; it is well-established. Our contribution is the systematic falsification of consistency-based routing (H3), which has not been previously demonstrated.

The negative result directly challenges the implicit assumption in RASC [Wan et al., 2024], which uses reasoning quality estimation alongside consistency for early stopping. If consistency does not predict correctness, weighting by reasoning quality may not fully compensate. CGES [Aghazadeh et al., 2025] uses confidence signals rather than consistency, suggesting an orthogonal approach that may avoid our failure mode — though this remains untested.

CoDE-Stop [Qin et al., 2025] and LEASH [Stahl et al., 2025] both focus on confidence dynamics rather than consistency, which is a promising direction. The key question is whether confidence-based signals are more robust to execution errors than consistency-based signals. Our theoretical analysis suggests they should be: confidence reflects the model's probability estimate, which is sensitive to whether the computation is correct, whereas consistency is a binary property (same answer or not).

## 6.4 Limitations

Three limitations constrain the generalizability of these findings:

**Sample size.** The H3 analysis uses $n = 30$ problems, which limits statistical power for detecting small-to-medium effect sizes. The confidence interval around the 68.4% accuracy estimate is wide, and the 6.8 percentage-point gap between low-$E_a$ and high-$E_a$ accuracy (68.4% vs. 63.6%) may not be statistically significant. Larger-scale validation is needed before drawing strong conclusions.

**Single model.** *Qwen2.5-Math-7B-Instruct* is a specialized mathematics model with above-average baseline performance on MATH. Different architectures — general-purpose LLMs, smaller models, or models trained on different corpora — may exhibit different error mode distributions. A model with lower baseline accuracy might show different saturation curves or different sensitivity to $E_a$ as a routing signal.

**Synthetic Ea estimation.** We estimate activation energy from answer consistency convergence rather than fitting per-problem saturation curves. A more accurate $E_a$ estimate — obtained by running the full saturation experiment for each problem — might yield different routing performance. The current approach trades accuracy for efficiency; a hybrid strategy could use rough $E_a$ estimates for initial routing and refine estimates for borderline cases.

## 6.5 Future Directions

The most pressing question is whether alternative routing signals can succeed where consistency failed. Three directions warrant investigation:

**Token entropy** is computable in a single forward pass and measures uncertainty about what to generate next. High entropy indicates the model is "guessing," which may correlate with incorrect reasoning. If low entropy correlates with correct single-pass answers, entropy thresholds could serve as routing signals.

**Attention-based diagnostics** could reveal whether the model is attending to relevant parts of the problem. Execution errors may produce distinctive attention patterns — the model "gives up" on certain calculations, producing incorrect outputs while maintaining consistent attention to the problem statement.

**Hybrid routing** could combine multiple signals: use entropy for initial classification, refine with attention diagnostics, and fall back to multi-sample inference when signals conflict. The key insight is that no single signal is perfect; routing strategies should ensemble multiple weak signals rather than relying on one strong signal.

---

## 6.6 Conclusion

The falsification of H3 reveals a fundamental limitation in consistency-based inference routing. Activation energy from answer consistency captures difficulty but not solveability: execution errors, conceptual errors, and answer extraction failures produce consistent but incorrect answers that fool consistency-based signals. This is not a limitation of our experimental design — it is a structural property of consistency as a signal.

The theoretical framework of Activation Energy Theory remains useful for understanding why multi-sample reasoning works and for predicting the saturation curve of accuracy with additional samples. However, consistency-based $E_a$ is insufficient as a routing oracle. Future routing strategies must look beyond agreement and develop signals that capture reasoning quality directly.

---

<!-- FIGURES
- Figure 6: hypothesis_summary.pdf — Routing signal comparison: consistency vs. alternative single-pass signals
- Table 5: inline — Routing signal comparison (measures, predictiveness, evidence)
- None
-->

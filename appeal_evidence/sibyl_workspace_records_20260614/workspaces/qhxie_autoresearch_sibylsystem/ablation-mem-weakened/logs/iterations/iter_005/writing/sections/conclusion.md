# 6. Conclusion

## 6.1 Summary

We conducted a systematic, multi-method investigation of feature absorption in GPT-2 Small SAEs across 26 first-letter features. Our key findings:

1. **Null results on downstream tasks (H1-H4)**: Zero hypotheses survive multiple comparison correction. Absorption does not significantly degrade steering effectiveness or sparse probing accuracy.

2. **Precision-recall asymmetry (H5)**: The one robust finding. Precision equals 1.0 universally at k >= 5; recall varies widely (0.05-1.0). This asymmetry is the signature of optimal compression behavior.

3. **Decoder correlation graph falsified (H6)**: The inhibition graph (constructed from decoder correlations $G = W_{\text{dec}}^T W_{\text{dec}}$) achieves precision@20 = 0.0, ruling out decoder-geometry-based absorption prediction.

4. **Random SAE baseline comparison (H7/H10)**: Trained SAEs show 8x lower absorption than random SAEs (0.034 vs. 0.278, p < 0.001), reframing absorption as a structural artifact that training reduces.

## 6.2 Theoretical Contribution

Our central theoretical contribution is the **optimal compression reframing**: under hierarchical co-occurrence and sparsity constraints, absorption is the optimal strategy for minimizing rate (sparsity loss) while preserving decoder alignment (reconstruction fidelity). The precision-recall asymmetry is the empirical signature of this behavior.

The key insight: the decoder direction $d_f = W_{\text{dec}}[:, f]$ is unaffected by which encoder activations fire. This is why precision is preserved even when recall is reduced. The parent feature's decoder direction remains accurate; the child feature handles the reconstruction.

## 6.3 Methodological Contributions

We contributed a reusable methodological framework:
- **Baseline correction**: Random baseline subtraction isolates absorption-specific effects from generic steering bias
- **Precision-recall decomposition**: Distinguishes coverage problems (recall) from selectivity problems (precision)
- **EC50 dose-response analysis**: Principled comparison of steering efficiency across features
- **Multiple comparison correction**: Bonferroni and BH-FDR applied systematically to all 12 tests

## 6.4 Practical Implications

For practitioners using SAEs for interpretability:
- Absorption does not appear to harm steering or probing in GPT-2 Small SAEs
- High-absorption features (e.g., Feature U with 24.2% absorption) retain full functional capability
- Architectural fixes (Matryoshka, OrtSAE) may be unnecessary for downstream tasks
- The field should focus on functional validation, not just absorption metrics

For researchers developing SAE improvements:
- Proposed fixes should be tested on downstream utility, not just absorption metrics
- The Chanin metric may need recalibration against random baselines
- Decoder-geometry-based approaches to understanding absorption are unlikely to succeed

## 6.5 Limitations

This work is limited to GPT-2 Small (124M parameters) and first-letter features (A-Z). Larger models and semantic hierarchies may show different patterns. The Chanin metric may not be well-calibrated for measuring absorption as a failure mode. Only two downstream tasks (steering and probing) were tested.

## 6.6 Final Statement

Feature absorption has been characterized as a failure mode requiring architectural mitigation. Our systematic investigation challenges this framing through honest null-result reporting, a falsified mechanistic hypothesis, and a precision-recall asymmetry that supports the optimal compression interpretation.

Absorption is not a pathology requiring fixes. It is the consequence of the rate-distortion trade-off in sparse coding: under hierarchical co-occurrence and sparsity constraints, absorbing parent features into child features minimizes rate while preserving decoder alignment. The precision-recall asymmetry is the signature of this behavior.

The field should shift from treating absorption as a failure to understanding it as a signal of the underlying optimization landscape. Future work should focus on functional validation of proposed fixes, better metrics for absorption as a meaningful phenomenon (not a structural artifact), and testing on larger models with deeper semantic hierarchies.

<!-- FIGURES
- None
-->
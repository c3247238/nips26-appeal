# 8. Conclusion

We conducted the first systematic, training-free, multi-objective evaluation of absorption-mitigation methods across 314 SAEBench checkpoints and 27 GPT-2 Small checkpoints. Three findings emerge.

First, no architecture family dominates the full Pareto front. Feature splitting eliminates dead neurons (mean $\delta_{\text{dead}} = 0.000$ vs. $0.197$ for Standard) and improves CE loss recovered ($1.172$ vs. $1.054$), yet Mann-Whitney U tests show no significant advantage on absorption ($p = 0.754$) or hedging ($p = 0.810$). Mitigations trade one pathology for another rather than delivering universal improvement.

Second, absorption has a unique negative causal effect on downstream interpretability utility. After controlling for $L_0$ and CE loss recovered, absorption shows a significant negative partial correlation with sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$), RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$), and RAVEL Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$). OLS regression with cluster-robust standard errors confirms absorption as a significant negative predictor of all three outcomes. This justifies continued attention to absorption---but only as one objective among many.

Third, the canonical first-letter absorption benchmark may not generalize to arbitrary hierarchies. Our task-agnostic metric pilot on 10 GPT-2 Small checkpoints reveals a weak negative correlation with the first-letter benchmark ($r = -0.592$, $p = 0.12$), with the first-letter metric degenerate at $0.0$ on 9 of 10 checkpoints. H3 is not supported.

The takeaway is clear: the SAE research agenda should move from "fixing absorption" to "navigating unavoidable tradeoffs." Future work should treat multi-objective evaluation as standard practice, develop task-adaptive selection criteria, and build absorption benchmarks that span multiple semantic domains rather than relying on single-task proxies.

<!-- FIGURES
- None
-->

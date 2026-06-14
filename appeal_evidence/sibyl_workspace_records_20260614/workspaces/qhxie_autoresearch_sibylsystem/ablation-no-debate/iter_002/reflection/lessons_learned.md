# Lessons from Iteration 2

## Must Improve

- **Pre-registration discipline**: The H_Mech criterion revision (failed 14/15 runs) reveals that pre-registration is not optional. Before running any experiment, especially those whose results will be central to the paper, pre-register the stopping criteria in a public registry (OSF). If original criteria fail, do not revise post-hoc; instead, acknowledge failure and design a new experiment with pre-registered criteria.

- **Single absorption metric standardization**: Three different absorption metrics (overlap fraction, Jaccard overlap, cosine-based proportional absorption) across experiments prevents legitimate cross-experiment comparisons. Before designing experiments, establish ONE canonical metric and use it consistently throughout. Recompute any existing results in that metric before writing the paper.

- **H3 steering numbers traceability**: The irreconcilable mismatch between paper-reported values (ratio=0.776, p=0.273) and experiment_summary.json (ratio=0.974, p=0.936) indicates a data provenance problem. Every experiment must write its canonical output to a single source of truth JSON that the paper absolutely must match. Add a verification step that the paper values exactly match the JSON before finalizing any section.

## Watch Out

- **Synthetic-only d_model=128 is a serious limitation**: Real LLM SAEs operate at d_model=768 to 12,288. Claims about encoder dominance may not scale. Plan d_model=512 validation as a required step, not optional.

- **Random baseline zero-variance is suspicious**: Random SAE baselines showing std=0 across all seeds suggests either measurement insensitivity or deterministic behavior. This undermines effect size claims (Cohen's d > 10 is mathematically inflated). Investigate and document the root cause.

- **Safety analysis saturating metric**: At ~97% absorption for both groups, there is no dynamic range to detect differences. This makes the H_Safe null result uninterpretable as evidence of equivalence. The metric choice for safety analysis needs a fundamental redesign.

- **Capacity-pressure mechanism may be TopK artifact**: The L0-absorption correlation could be an artifact of TopK sparsity patterns, not a genuine mechanism. A JumpReLU SAE comparison is needed to validate the mechanism claim.

## Keep Doing (success patterns)

- **Honest negative results**: Continue reporting H3 and H_Safe negative results without spin. This is the paper's strongest aspect across all reviews. The specific p-values and effect sizes build credibility.

- **Factorial decomposition method**: The 2x2 design isolating encoder vs decoder contributions is genuinely novel and well-executed. Preserve this as the core contribution.

- **Clear contribution framing**: The six contributions are clearly enumerated. Maintain this structure.

- **Efficient experiment execution**: Experiments completed in 0-30 min vs 20-45 min planned. This efficiency allows more iteration cycles. Preserve tight experiment design with clear pass/fail criteria.

- **Limitation transparency**: Explicitly discussing five limitations builds reviewer trust. Continue this practice.
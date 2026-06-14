# Paper Outline: Honest Compute for Revision in Diffusion LMs

## Title Options

1. **When Revision Helps Diffusion LMs, When It Hurts, and Why Honest Compute Matters**
2. **Honest Compute for Revision in Diffusion LMs: Observer-Controller Mismatch and Task-Dependent Failure**
3. **A Compute-Normalized Diagnostic Study of Revision in Diffusion Language Models**

**Recommended title**: Option 1. It is specific enough to foreground the main findings, but still modest about scope.

## One-Sentence Thesis

This paper is not about proposing another winning revision policy; it is a compute-normalized diagnostic study showing that honest compute accounting changes key comparisons, strong diagnostic signals do not reliably become strong control policies under the tested interventions, and revision response is sharply task-dependent across reasoning and code.

## Positioning and Guardrails

- Frame the paper as a **diagnostic study + evaluation protocol + failure taxonomy**, not as a benchmark standard-setter.
- Avoid method-forward claims for `TIGER`; keep it as a failed-but-informative controller case.
- State honest-compute conclusions as **changing key comparisons / Pareto conclusions**, not as a universal ranking rewrite.
- State observer-controller conclusions as **under the tested policies**, not as a universal law.
- Keep code evidence in the main paper only as boundary evidence for task dependence; move implementation details and extended tables to the appendix.

## Section 1: Introduction

### Key content

- Motivate the central confusion in training-free DLM decoding: published comparisons often headline nominal steps while hiding meaningful differences in actual compute, runtime, and intervention overhead.
- Introduce the second confusion: entropy, calibration, and instability are often treated as both observers and controllers, even though those are distinct roles.
- Preview the paper's three claims:
  - honest compute can change key method comparisons;
  - good observers do not reliably become good controllers under the tested policies;
  - revision helps some reasoning settings but fails at the code boundary.

### Evidence to foreshadow

- On GSM8K, `CORE-proxy-64` has the best accuracy (`0.46`) but also far higher latency (`482.95s`) and a worse actual-compute position than its nominal label suggests.
- In the signal audit, calibration has the strongest diagnostic score (`0.6225`) yet no deployed control gain.
- On HumanEval, gating reduces syntax failure by `20` points but still stays below the standard baseline on `pass@1` (`0.04` vs `0.06`).

### Transition

Move from the field-level problem to the paper's evaluation protocol: if the claims are about compute, signals, and revision response, the paper must first define a protocol that measures all three honestly.

## Section 2: Experimental Setting and Honest-Compute Protocol

### Key content

- Define the method family under study: `Standard-64`, `DNB-84`, `Prophet-64`, `CORE-proxy-64`, `Entropy-Revise-64+3`, and `TIGER-Instability-64+3`.
- Explain the protocol variables that matter for fairness: actual NFE, latency, throughput, batch size, backend, and compile status.
- Describe benchmark roles:
  - GSM8K as the primary reasoning benchmark;
  - MATH500 as the transfer check on a second reasoning regime;
  - HumanEval as a boundary test for structured-output fragility.

### Evidence to include

- GSM8K runs used heterogeneous runtime settings, including `batch_size=115` for `Standard-64` and `batch_size=1` for `CORE-proxy-64`.
- `Standard-64`, `DNB-84`, and `Prophet-64` ran with `compile_enabled=true`, while revision-heavy methods and `CORE-proxy-64` did not.
- The protocol therefore needs both quality metrics and runtime-fairness metadata to support any comparison.

### Transition

Once the protocol is defined, the first empirical question is whether honest compute actually changes the story on the headline reasoning benchmark.

## Section 3: Honest Compute Changes Key Comparisons on GSM8K

### Key content

- Present the matched-compute GSM8K comparison as the paper's first main result.
- Show that nominal method labels and actual compute positions diverge enough to alter pairwise conclusions and Pareto interpretations.
- Emphasize that this section is about **comparison hygiene**, not about crowning a single universally best method.

### Evidence to include

- `CORE-proxy-64` is nominally a 64-step method, but uses `actual_nfe=69.0`, moving from nominal compute rank `3` to actual compute rank `5`.
- `Entropy-Revise-64+3` moves ahead of `CORE-proxy-64` under actual-compute ordering, creating one explicit reorder in `diag_compute_curve_gsm8k.json`.
- `CORE-proxy-64` keeps the best GSM8K accuracy (`0.46`) but with much higher latency than `Entropy-Revise-64+3` (`482.95s` vs `210.67s`) and `TIGER-Instability-64+3` (`213.81s`).
- `Prophet-64` is slightly faster than `Standard-64` (`147.13s` vs `157.04s`) despite sharing the same nominal 64-step headline.

### Transition

After showing that honest compute changes key comparisons, the next question is whether the signals used by revision methods actually justify their control role.

## Section 4: Good Observers Do Not Reliably Become Good Controllers

### Key content

- Separate diagnostic signal quality from control effectiveness.
- Compare three signal families: calibration, entropy, and instability.
- Argue that the strongest surviving contribution is not a better controller, but a cleaner observer/controller factorization.

### Evidence to include

- Calibration is the strongest observer in the audit: diagnostic score `0.6225`, but no direct deployed control gain.
- Entropy has a meaningful diagnostic score (`0.414`) yet nearly no extra benefit over random revision in the signal screen, where both entropy and random revision sit at `0.37` accuracy.
- Instability shows only weak diagnostic association (`0.0555`) and only marginal control effectiveness (`0.01` over random in the screen setting).
- On the GSM8K shortlist, entropy revision and TIGER tie at `0.39`, so the more elaborate instability-flavored controller does not beat the simpler entropy baseline.

### Transition

If strong signals do not automatically imply strong controllers, then the remaining explanation must involve task structure and recoverability rather than signal strength alone.

## Section 5: Revision Response Is Task-Dependent

### Key content

- Contrast revision behavior across two reasoning datasets and one code dataset.
- Show that even within reasoning, the ordering is not stable across GSM8K and MATH500.
- Use HumanEval to demonstrate that shallow syntactic repair does not imply deep functional recovery.

### Evidence to include

- GSM8K ordering is `CORE-proxy > Entropy = TIGER > Standard`, while MATH500 ordering becomes `Standard > CORE-proxy > TIGER > Entropy`.
- On MATH500, `Standard-64` reaches `0.23`, ahead of `CORE-proxy-64` (`0.21`), `TIGER-Instability-64+3` (`0.20`), and `Entropy-Revise-64+3` (`0.19`).
- On HumanEval, ungated revision drops to `0.02` `pass@1`, while `Gated TIGER` recovers only to `0.04`, still below `Standard` at `0.06`.
- The gate reduces syntax failure from `0.48` to `0.28`, but runtime failure worsens from `0.50` to `0.68`, supporting a shallow-fix / deep-failure interpretation.

### Transition

With the task-dependent pattern established, the paper can now integrate the three threads into a single interpretation of what revision studies should measure next.

## Section 6: Discussion, Limits, and What the Results Actually Support

### Key content

- Synthesize the paper's message as a diagnostic account of revision benefit, not a new decoding algorithm paper.
- Explain why code belongs in the paper as boundary evidence rather than as a second main success domain.
- Clarify the safe scope of claims and acknowledge what is still missing.

### Points to make explicitly

- Honest compute should be described as changing key comparisons and Pareto conclusions, not as invalidating every nominal-step ranking.
- The observer-controller mismatch should be described as stable under the tested policies, not as a universal theorem.
- Current evidence is still limited by single-seed evaluation and the absence of a completed benefit-bucket audit.
- The most valuable next follow-up package is:
  - `benefit_bucket_audit`;
  - a minimal `seed_sensitivity_spotcheck`;
  - appendix-grade runtime fairness and asset-lineage summaries.

### Transition

The conclusion should restate the paper as a narrowing contribution: less about a new method, more about what a credible revision evaluation protocol should reveal.

## Section 7: Conclusion

### Key content

- Restate the three takeaways in one paragraph:
  - honest compute matters because nominal labels can hide real comparison shifts;
  - strong diagnostic signals do not automatically deliver strong control policies;
  - revision response depends on task structure and recoverability.
- Emphasize the paper's durable contribution as a compute-normalized diagnostic study for revision in diffusion LMs.
- End on the practical implication: future DLM revision papers should report runtime-fairness metadata, separate observer from controller claims, and treat code as a structural stress test rather than automatic evidence of generality.

## Section 8: Related Work

### Key content

- Position the paper relative to training-free DLM decoding methods such as Prophet, CORE-style probing, entropy-guided revision, and instability-based control.
- Contrast this paper with calibration work that studies predictive confidence without the DLM denoising-process distinction.
- Frame the novelty as methodological: prior work proposes controllers, while this paper asks what evidence is needed to compare them honestly.

### Placement note

Keep this section compact and place it either before the conclusion or after the introduction, depending on venue style. The preferred draft order is after the conclusion in the first full manuscript so the main story stays result-first.

## Figure & Table Plan

### Figure 1: Why Headline Step Counts Are Not Enough (Section: Introduction)
- **Purpose**: Give readers a one-glance teaser that the paper is about comparison hygiene, signal-role mismatch, and task dependence.
- **Type**: composite_panel
- **Content**: Left panel shows nominal vs actual compute ordering on GSM8K; middle panel shows diagnostic vs control gap for the three signals; right panel shows syntax failure drop without `pass@1` recovery on HumanEval.
- **Key takeaway**: The central problems are already visible before any method story begins.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/diag_compute_curve_gsm8k.json`, `current/exp/results/diag_signal_gap_audit.json`, `current/exp/results/diag_humaneval_guard_boundary.json`

### Table 1: Matched-Compute Comparison on GSM8K (Section: Honest Compute)
- **Purpose**: Establish the primary reasoning comparison under honest compute.
- **Type**: comparison_table
- **Content**: Report accuracy, actual NFE, nominal NFE, latency, throughput, batch size, backend, and compile status for all six shortlist methods.
- **Key takeaway**: Key conclusions depend on actual compute and runtime metadata, not nominal labels alone.
- **Generation**: data_table
- **Data source**: `current/exp/results/diag_compute_curve_gsm8k.json`, `current/exp/results/gsm8k_main_shortlist.json`

### Figure 2: Pareto View Under Honest Compute (Section: Honest Compute)
- **Purpose**: Visualize how the GSM8K shortlist moves once latency and actual NFE are taken seriously.
- **Type**: scatter
- **Content**: Plot accuracy against latency, annotate actual NFE, and highlight the Pareto frontier reported in `diag_compute_curve_gsm8k.json`.
- **Key takeaway**: `CORE-proxy-64` remains strong on raw accuracy, but its Pareto position is much less clean than the headline label suggests.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/diag_compute_curve_gsm8k.json`

### Figure 3: Observer Quality vs Controller Gain (Section: Observer-Controller Mismatch)
- **Purpose**: Make the observer/controller split visually explicit.
- **Type**: scatter
- **Content**: One point each for calibration, entropy, and instability; x-axis is diagnostic score, y-axis is control effectiveness.
- **Key takeaway**: Calibration is the strongest observer and the weakest controller; entropy is useful but not transformative; instability is neither a strong observer nor a clearly better controller.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/diag_signal_gap_audit.json`

### Table 2: Task-Dependent Ordering Across GSM8K and MATH500 (Section: Task Dependence)
- **Purpose**: Show that even within reasoning, revision ordering is not stable across datasets.
- **Type**: comparison_table
- **Content**: Include GSM8K and MATH500 accuracy, actual NFE, and ordering for `Standard-64`, `CORE-proxy-64`, `Entropy-Revise-64+3`, and `TIGER-Instability-64+3`.
- **Key takeaway**: The revision story is dataset-dependent, not a single monotonic ladder.
- **Generation**: data_table
- **Data source**: `current/exp/results/gsm8k_main_shortlist.json`, `current/exp/results/diag_math500_shortlist.json`

### Figure 4: Code Boundary Failure Breakdown (Section: Task Dependence)
- **Purpose**: Show why syntax guarding is only a shallow repair.
- **Type**: bar_chart
- **Content**: Bars for `pass@1`, syntax failure, and runtime failure across `Standard`, `Entropy/TIGER Ungated Revision`, and `Gated TIGER`.
- **Key takeaway**: The gate repairs syntax but does not restore execution success.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/diag_humaneval_guard_boundary.json`

### Table 3: Runtime Fairness Appendix Table (Section: Appendix)
- **Purpose**: Provide the exact implementation metadata needed to defend the comparisons.
- **Type**: table
- **Content**: For every reported run, include batch size, backend, compile flag, peak VRAM when available, throughput, and source asset lineage.
- **Key takeaway**: Runtime configuration is part of the result, not an implementation footnote.
- **Generation**: data_table
- **Data source**: `current/exp/results/diag_compute_curve_gsm8k.json`, `current/exp/results/diag_math500_shortlist.json`, `current/exp/results/diag_humaneval_guard_boundary.json`, `current/exp/results/diag_signal_gap_audit.json`

## Writing Notes for the Next Stage

- Keep the paper result-first. The method family is introduced only to support the comparisons, not to foreground `TIGER`.
- Avoid “benchmark” as a noun for the paper's contribution in the abstract and title; prefer “diagnostic study,” “evaluation protocol,” and “failure taxonomy.”
- Report negative results plainly, especially the HumanEval boundary evidence and the entropy-vs-TIGER tie on GSM8K.
- If the drafting stage needs one sentence on future work, prioritize `benefit_bucket_audit` and a minimal seed-sensitivity check rather than proposing a new controller.

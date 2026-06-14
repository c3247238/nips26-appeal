# 6. Conclusion

This paper revisits training-free revision in diffusion language models as a diagnostic study rather than a controller paper. We show that honest compute accounting changes key comparisons on GSM8K, that strong diagnostic signals do not reliably become strong control policies under the tested interventions, and that revision response is task-dependent across reasoning and code.

These findings matter because they redefine what a persuasive revision paper should report. Nominal step counts are not enough. Observer and controller claims should be separated. Boundary failures on structured tasks should be treated as core evidence, not hidden negatives. Under that framing, the most durable contribution of this work is a compute-normalized diagnostic study and evaluation protocol for revision in DLMs.

Future work should extend the current evidence with benefit-bucket analysis, runtime-fairness follow-up tables, and minimal seed-sensitivity checks. But even in its current form, the paper offers a concrete recommendation for the field: evaluate revision methods as systems with real runtime costs, explicit observer/controller separation, and code used as a structural stress test rather than as automatic evidence of generality.

<!-- FIGURES
- None
-->

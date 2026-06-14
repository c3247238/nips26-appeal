# Lessons from Iteration 004

## Must Improve

- **Never inherit data values from previous iterations without verification**: Iter_004 inherited the Random-SAE semantic-hierarchy score of 0.352 from iter_003's estimated data (e1_decomposition_results.json), but the actual iter_004 experimental output (semantic_hierarchy_pythia_results.json) shows 0.175. This is a credibility-destroying error. Rule: when writing a new iteration's paper, every number must be verified against the CURRENT iteration's experimental output, not inherited from previous iterations.
- **Pre-writing data integrity check is mandatory**: Before LaTeX compilation, run an automated script that compares all numbers in the paper against source JSON files. The Random-SAE error (0.352 vs 0.175) and Cohen's d inconsistency (-1.68 vs -1.794) would have been caught immediately. Rule: NO-GO to LaTeX compilation if any number in tables or text does not match the source JSON.
- **Cohen's d must be sourced from the correct file**: The value -1.68 has persisted across 3 iterations despite e3_ttest_results.json showing -1.794. The statistical_analysis_summary.md does not include Cohen's d, so writers may not know where to find it. Rule: all effect sizes must be explicitly included in the statistical summary markdown file, and writers must verify against the JSON source.
- **Multiple comparison correction must be implemented**: This has been flagged in every iteration's reflection but never addressed. The statistical_analysis.py code must include Bonferroni or Benjamini-Hochberg correction. Rule: NO-GO if statistical analysis code does not include multiple comparison correction when >1 test is reported.
- **Reflection recommendations must propagate to experiment code**: Issues flagged in reflection (ceiling effect sensitivity, k-sensitivity, multiple comparison correction) are not automatically incorporated into the next iteration's experiment plan. Rule: action_plan.json items with category "analysis" or "experiment" must be reviewed by the planner before designing the next iteration's experiments.

## Watch Out

- **Score improvements can mask persistent data errors**: The score improved from 6.5 to 7.0, but the most critical issue (Random-SAE data error) was introduced or inherited during this "improving" iteration. Do not trust score increases without examining data integrity.
- **Cross-iteration data inheritance is risky**: When reframing or reusing previous experiments, old incorrect values can propagate. Always re-source from raw experimental output.
- **Perfect scores (AUROC=1.0) are a ceiling effect**: All 80 hierarchy probes achieved perfect AUROC. This means the absorption metric collapses to k-sparse probing loss. Do not interpret perfect scores as evidence of strong probe performance without considering the implications for the metric.
- **GPT-2 near-zero scores may indicate metric failure**: The GPT-2 replication shows near-zero absolute scores with perfect k-sparse accuracy. This could mean the metric adaptation simply doesn't work on GPT-2, not just "model-specific behavior."
- **Non-hierarchy controls may not be clean**: Pairs like "tree-wood" (meronymy) and "river-water" (thematic) may share structural properties with hierarchies. Future controls should use more clearly non-hierarchical pairs.
- **Post-hoc power analysis is methodologically inappropriate**: If included, remove it. Pre-registered power analyses are valid; post-hoc ones are not.

## Keep Doing (success patterns)

- **Honest reframing when evidence doesn't support claims**: Iter_004 successfully reverted from the Goodhart's Law framing (which overreached the evidence) back to a clean construct-validity study. This demonstrates epistemic humility and strengthens credibility.
- **All data is real (no estimates)**: Unlike iter_003, iter_004 contains no estimated or synthetic data presented as empirical. All values come from actual pipeline execution.
- **Honest negative result reporting**: H2 rejection (non-hierarchy > hierarchy) is reported without spin, with specific statistics (t=-4.748, p=0.003, d=-1.794). This is the paper's strongest aspect.
- **Writing quality remains high**: The writing review score is 8/10. Specific numbers throughout, clear structure, hypothesis-driven Results section, and a model abstract.
- **Transparent raw data in JSON**: All per-hierarchy and per-pair scores are available in JSON files, enabling verification.
- **Bootstrap CIs for small-n inference**: B=10,000 bootstrap CIs are appropriate and well-executed.

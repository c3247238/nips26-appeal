# Lessons from This Iteration

## Must Improve

1. **Match claim confidence to data completeness**: When only 3 of 6 variants have full data, NEVER present rankings or conclusions definitively. Always add "(provisional, based on X of Y variants)" or "(among tested components)" to every claim. The Abstract and Conclusion must reflect the actual data, not the planned data.

2. **Downgrade small-n correlations from contributions to exploratory observations**: A correlation with n=4 (or n=3 after removing incomparable points) is not a primary contribution. It is an exploratory observation requiring validation. Never present small-n correlations as "striking" or "near-perfect" in abstracts or contribution lists.

3. **Separate pilot and full-experiment results completely**: Never mix pilot data (1 replicate, 1024 features) with full-experiment data (5 replicates, 16384 features) in the same table, figure, or ranking. Use separate tables with clear labels. Do not rank pilot results against full-experiment results.

4. **Deliver promised controls**: If the Method section promises L0-matched comparisons, ANOVA, or individual replicate values, these MUST appear in the Results. Unfulfilled promises undermine credibility. Either deliver the control or remove the promise before writing the Method.

5. **Scope experiments to what can be completed in one iteration**: A 6-variant x 5-replicate design is too ambitious. Plan for 3-4 variants per iteration maximum. Split large designs across multiple iterations with explicit go/no-go criteria.

## Watch Out

- **MCC ~0.21 is chance-level across all variants including Random**: This is not a minor metric issue. It raises fundamental questions about whether absorption reduction reflects genuine learning or sparsity-induced suppression. Address this explicitly in Discussion.

- **"Order of magnitude" phrasing is easily misused**: Reserve it for ~10x differences. Use accurate multipliers ("fivefold", "fortyfold") for other comparisons. Vary phrasing to avoid repetition.

- **Verbatim repetition across sections**: "This redirects the research question" appeared identically in Sections 4.6 and 5.1. Always vary phrasing when making similar points in different sections.

- **Synthetic-to-real gap**: Without real-LLM validation, the practical impact of findings is conditional. Frame conclusions with "If this transfers to real LLMs..." and prioritize validation experiments.

- **Hyperparameter claims require hyperparameter evidence**: Claiming "orthogonality has no absorption benefit" without tuning lambda_ortho is premature. Always acknowledge untuned hyperparameters as limitations.

## Keep Doing (success patterns)

- **Honest negative results**: H1 REJECTED, H3 NOT SUPPORTED, incomplete variant set flagged---this builds reviewer trust and is the paper's strongest aspect. Continue reporting negative results with specific expected vs. observed values.

- **Ground-truth synthetic data**: The pivot from probe-based metrics to SynthSAEBench-16k eliminated major confounds. Continue using ground-truth metrics where available.

- **Random controls**: The Random SAE control (absorption=0.560) validates metric discrimination. Always include appropriate controls.

- **Effect sizes with confidence intervals**: Cohen's d values with standard deviations provide clear statistical context. Continue reporting effect sizes, not just p-values.

- **Scope notes**: Section 1.5 is a model of epistemic honesty. Continue prominently flagging data limitations.

- **Pilot validation before full experiment**: The 4-condition pilot with go/no-go criteria is excellent design. Continue using pilots to validate experimental designs before scaling up.

- **Well-crafted abstracts**: The abstract leads with problem, states method, reports key finding with numbers, states implication. Continue this structure.

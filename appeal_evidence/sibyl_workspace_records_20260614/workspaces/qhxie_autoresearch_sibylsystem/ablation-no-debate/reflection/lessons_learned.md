# Lessons from This Iteration

## Must Improve

1. **Paper narrative must be updated when new experimental findings emerge**: The H_Mech pilot revealed absorption is encoder-driven (not decoder-geometric), but the paper was never updated. When follow-up pilots contradict the initial narrative, the paper draft MUST be revised before proceeding to review. Add a "narrative sync" checkpoint after each new pilot batch.

2. **Critic findings must be enforced before writing finalization**: The Section 5.1 typo ("Single-child ablation" describing multi-child) was identified by the critic but never fixed. Implement a mandatory fix-verification step: all CRITICAL and MAJOR critic issues must be resolved and verified before the writing_integrate stage completes.

3. **Deterministic measurements must be flagged and investigated, not celebrated**: Trained SAE absorption at exactly 0.5 with std=0.0 across all seeds should have triggered an investigation into whether the metric formula reduces to a constant. Zero variance is a red flag, not a sign of robustness. Always ask: "Does this metric have within-condition variance, or is it deterministic?"

4. **Feature selection claims must be validated before writing**: H_Safe used arbitrary indices (500-519) but the paper claimed "Neuronpedia-annotated safety features." Any claim about feature annotation must be backed by actual API calls or database lookups. When validation is not possible, state "heuristic selection" explicitly.

5. **Pilot results must be clearly labeled as pilot**: The paper presents pilot results without distinguishing them from full experiments. Always label results with their experimental phase (pilot vs full) and sample size. The pilot summary recommended full experiments that were never run -- this gap must be acknowledged.

## Watch Out

- **Overclaiming from broken experiments**: H3 steering in the first pilot was broken (baseline=steered to 15 decimal places, t=NaN) but was presented as a negative result with causal conclusions. A broken experiment is not a negative result -- it is a bug. Always verify that the measurement instrument works before interpreting null results.
- **Config discrepancies between paper and data**: Paper claimed d_model=512 but data shows d_model=128. Always cross-check methodology claims against actual experiment configs before writing.
- **Zero-inflated data driving correlations**: H2's rho=+0.17 was driven by 15 non-zero features out of 1024. When >95% of data is zeros, correlation analysis is unreliable. Always report the zero-inflation rate.
- **Circular experimental designs**: H_Mech's Condition C reused the trained encoder from Condition B, confounding decoder isolation. When designing factorial experiments, ensure each condition is independently initialized and trained.
- **Division-by-near-zero in percentage calculations**: H3 steering showed billion-percent values due to dividing by near-zero baseline. Always sanity-check percentage calculations; if baseline can be zero, use absolute values instead.

## Keep Doing (success patterns)

- **Honest negative result reporting**: H2 failure (rho=+0.17, wrong direction) and H_Safe null result (p=0.665) were documented without spin, with specific expected vs observed values. This is the paper's strongest aspect and should be maintained.
- **Multi-child proportional ablation methodology**: The core methodological contribution is genuinely novel and addresses a real saturation problem. Continue developing and refining this approach.
- **Strong claim-evidence alignment**: Every quantitative claim in the paper matches source data exactly. The writing review verified this with a claim-evidence table. Maintain this rigor.
- **Clear hypothesis structure with falsification criteria**: Each hypothesis has explicit pass/fail criteria and unambiguous results. This pre-registration-style presentation strengthens credibility.
- **Efficient pilot execution**: 4 pilot tasks completed in ~15 minutes total GPU time. The pilot design is well-scoped and fits the 1-hour guideline.
- **Unexpected finding identification**: The encoder-driven absorption finding is genuinely novel and counter-intuitive. Even when it contradicts the original hypothesis, reporting it honestly adds scientific value.

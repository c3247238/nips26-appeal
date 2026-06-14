# Lessons from Iteration 1

## Must Improve

- **[EXPERIMENT] Metric consistency across pilot and full-scale**: Pilot H2 used Chanin first-letter probe absorption; full H2 used Gini absorption. This was never caught before writing. Every experiment must pre-specify the exact absorption metric (Chanin probe, Gini, etc.) in the experiment design document BEFORE running. Never mix metrics between pilot and full-scale without explicit documentation.

- **[EXPERIMENT] Post-hoc threshold fabrication destroys credibility**: H5 used a retroactively invented 5% threshold when the actual pass criterion was >8%. Always pre-register pass/fail criteria in the experiment protocol before seeing results. Never create thresholds after the fact to make results look better.

- **[EXPERIMENT] Null controls must match main experiment protocol**: Null controls used alpha=5 only; main H3 aggregated [1,3,5,10,20]. The null controls showed NO effect (p=0.94) while main H3 showed 18% effect. Null controls must use identical protocols to the main experiment to be meaningful controls.

- **[EXPERIMENT] Feature selection decisions must be tracked**: Paper stated UAS > 1.0 threshold but actual selected features had mean UAS ~0.57. Track the exact feature selection threshold used in experiment metadata, not just in paper drafts.

- **[WRITING] Causal language requires causal evidence**: "Causally important" was used in the abstract but steering sensitivity does not measure causal importance. Only use causal claims when there is causal ablation evidence. Otherwise use "highly steerable" or "computationally central."

- **[WRITING] Scope must match promises**: Title and abstract promised multi-model scope (GPT-2 + Gemma-2B). Gemma-2B experiments were skipped. Either deliver what you promise or update promises to match what was actually done.

## Watch Out

- **H1 layer-wise instability**: Three different experimental runs produced contradictory absorption patterns by layer. Do not claim any established layer-wise pattern without confirming consistency across protocols.

- **H4 full-scale was actually pilot-mode**: Named "full_h4" but run with only 3 pairs, not full-scale. Spearman r declined from 0.81 to 0.59-0.71. The naming discipline was not enforced — experiment names must reflect actual scope.

- **Entanglement hypothesis without mechanism evidence**: r=+0.35 correlation does not establish a causal mechanism. Presenting speculation as a named hypothesis without empirical support undermines credibility.

- **Effect size inconsistency within the same paper**: Section 4.2 said "~15%" when the correct figure is 18.4%. Abstract said "18%". This inconsistency suggests the paper was not proofread carefully enough before submission.

## Keep Doing (success patterns)

- **Large sample H3**: N=200 features with cluster-robust standard errors and N=314 E2 meta-analysis is the right scale for credibility. The H3 finding (r=+0.35, p<0.001) is the paper's strongest asset.

- **UAS metric validation**: Clear presentation of Spearman correlations across layers (r=0.65-0.79) with a dedicated table is reproducible and useful to the community.

- **Ablation studies**: L1 sparsity and orthogonality lambda ablations add methodological depth beyond the main correlation result.

- **Honest negative results**: The system correctly identified that H2 full-scale shows all mitigation variants failing (higher absorption than vanilla) and did not hide this. This honest reporting is the right approach.

- **Pilot before full-scale**: Running pilots first to validate methodology before committing full-scale resources is sound experimental practice. The failure was in using different metrics between pilot and full-scale, not in the pilot-then-full approach itself.

## System Process Improvements

- **Pre-experiment protocol document**: Before running any new experiment, write a one-paragraph protocol specifying: metric to be used, pass/fail criteria, sample sizes, feature selection criteria. This prevents metric mixing and threshold fabrication.

- **Post-experiment consistency check**: After full-scale experiments complete, automatically check: (1) do full-scale results directionally match pilot results? (2) are all metrics consistent with what was specified? Flag contradictions before writing begins.

- **Experiment naming enforcement**: Scripts should fail if an experiment named "full_" contains fewer than the required sample size. This prevents "full" experiments from being run in "pilot" mode.

- **Iteration history**: Write to research_diary.md and update quality_trend.md at end of each iteration. These files are essential for tracking long-term quality trajectory.

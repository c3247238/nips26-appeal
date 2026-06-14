# Lessons from Iteration 3

## Must Improve

- **[EXPERIMENT] Protocol must pre-specify absorption metric**: Pilot H2 used Chanin first-letter probe, full-scale H2 used Gini absorption. This was the #1 issue in iterations 1, 2, and 3. Before running any experiment, the protocol document MUST specify the exact absorption metric (Chanin probe, Gini, etc.). Do not allow pilot and full-scale to use different metrics.

- **[EXPERIMENT] Post-hoc threshold fabrication destroys credibility**: H5 used a retroactively invented 5% threshold when actual criterion was >8%. This was identified in iteration 1 lessons and iteration 2 lessons. It is still in the paper. Pre-register all pass/fail criteria before seeing results.

- **[EXPERIMENT] Null controls must match main experiment protocol**: Null controls used alpha=5 only; main H3 aggregated [1,3,5,10,20]. This was in iteration 1 lessons. full_h3_null is now running with full alpha range -- use its results. If results confirm the protocol mismatch, acknowledge it as a limitation.

- **[ANALYSIS] CMI dimension instability is a qualitative failure**: CMI-absorption correlation reverses sign across d' dimensions. Do not cherry-pick d'=10 as primary. Report all dimensions or restrict to quality-gated letters (F1>0.85). Compute partial correlation controlling for probe F1.

- **[ANALYSIS] OrtSAE ablation is self-contradictory**: Comparing with-penalty (L0~550) against without-penalty (L0~920) at different L0 is the very confound the paper criticizes. Redesign ablation at matched L0.

- **[WRITING] Generate missing figures**: Paper references Figure 1-6 but no figures are included. Central H3 result (scatter, bar chart) has no visual support. This is a compilation blocker.

## Watch Out

- **Entanglement hypothesis without mechanism evidence**: r=+0.35 does not establish causal mechanism. ablation_hub_interpretation is running -- wait for results before deciding whether to keep or remove the hypothesis framing.

- **n=4 correlation is fragile**: Two points at identical L0=50, driven by only 3 unique L0 values. Present as preliminary observation, not robust finding.

- **MCC=0.21 across Random control**: Hungarian matching yields chance-level recovery regardless of training. Address what this means for absorption metric validity in Discussion.

- **Unfulfilled promises**: Section 3.5 promised ANOVA but did not report it. Remove promise or deliver it.

- **Writing stage applies spin**: Results that are failures (H2 full-scale, H5) are being reframed as successes. Writing agent needs explicit instruction: report failures as failures.

## Keep Doing (success patterns)

- **Honest negative results**: H2 falsified, H4 regression, H5 failed -- all correctly reported without spin. This is the paper's strongest scientific contribution.

- **Large sample H3**: N=100 features with cluster-robust standard errors is the right scale. The finding (r=+0.35, p<0.001) is genuine and interesting.

- **UAS validation**: r=0.65-0.79 across layers with proper statistical methods. A genuine contribution to the community.

- **E2 meta-analysis**: N=314 sample with partial correlation is the paper's strongest empirical signal.

- **Pilot before full-scale**: The two-stage approach is sound; the failure is always in protocol consistency, not in the approach itself.

## System Process Improvements

- **Pre-experiment protocol enforcement**: Before running any experiment, verify: (1) absorption metric matches pilot, (2) null control alpha range matches main experiment, (3) pass/fail criteria are pre-registered, (4) experiment naming reflects actual scope.

- **Post-experiment consistency check**: After full-scale experiments complete, automatically check: (1) do full-scale results directionally match pilot? (2) are all metrics consistent with pilot? Flag contradictions before writing.

- **Writing spin detection**: Add a writing stage check: for each result, verify the framing matches the actual numbers. If pass criterion was >8% and achieved 2.51%, the framing must be "failed," not "marginal miss" or "not significant at 5%."

- **Figure generation**: Every figure promised in the outline must exist before the paper is considered complete. Add figure generation to the experiment workflow, not as an afterthought.

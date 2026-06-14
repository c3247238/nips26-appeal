# Lessons from Iteration 001

## Must Improve

- **Degenerate absorption proxy invalidates E1/E3 claims**: The simplified first-letter proxy returned 0.0 on 96% of checkpoints. Next iteration must integrate the full `sae-spelling` / SAEBench absorption pipeline before presenting absorption comparisons as primary evidence.
- **Failed pilot GO/NO-GO gate**: The pilot explicitly rated metric quality as "NO-GO for publication-ready numbers," yet the full experiment proceeded anyway. Future iterations must treat pilot evaluation as a hard gate: NO-GO means repair first, scale second.
- **Construct validity crisis in E3**: A negative correlation (-0.59) between two operationalizations of the same construct is not a "weak negative result"—it is a serious validity problem. Next iteration must either explain the divergence theoretically or conclude that "absorption" lacks cross-domain construct validity.
- **Causal overclaim in observational meta-analysis**: E2 uses "causal cost" and "unique causal effect" language for an observational design. Replace with "predictive cost" or "associational relationship" in titles and abstract.

## Watch Out

- **Training-free constraint narrows empirical scope**: The constraint limited E1 to GPT-2 Small checkpoints. Do not let the paper's framing imply broader coverage (e.g., Gemma-2-2B direct evaluation) than what the controlled experiments actually provide.
- **Family collapse masks patterns**: Collapsing resid_pre, resid_post, mlp_out, attn_out, and hook_z into a single "Standard" bucket hides meaningful variation. Always disaggregate by hook point and architecture family in comparison tables.
- **Figure and table references**: Figure 4 appeared without a forward reference; Tables 1-3 were inline markdown instead of labeled LaTeX. Run a pre-submission checklist for all figures, tables, and terminology alignment.
- **Gated model access**: The Gemma experiment failed due to missing HF authentication. Always do a pre-flight access check before scheduling gated-model experiments.

## Keep Doing (success patterns)

- **Large-N meta-analysis with rigorous statistics**: E2 (N=314) is the paper's strongest asset. The use of partial correlations, cluster-robust SEs, and proper controls should be maintained and expanded.
- **Honest negative result reporting**: E3's unsupported H3 was reported without spin. This is exactly the right tone for a top venue and should be preserved.
- **Tight claim-evidence coupling**: Most claims are anchored to specific statistics (e.g., "$r_{\text{partial}} = -0.385$, $p < 0.001$"). Continue this practice.
- **Clear conceptual reframing**: The shift from "fixing absorption" to "navigating tradeoffs" is intellectually sharp and memorable. Keep this framing as the paper's backbone contribution.
- **Training-free ecological validity**: Evaluating publicly released checkpoints without retraining is a principled design that ensures reproducibility and low cost. Retain this approach where appropriate.

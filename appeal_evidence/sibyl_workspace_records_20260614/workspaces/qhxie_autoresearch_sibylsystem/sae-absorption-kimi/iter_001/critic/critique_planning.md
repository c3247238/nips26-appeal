# Planning Critique

## Overall Assessment

The methodology (`plan/methodology.md`) is well-structured and follows rigorous experimental design principles contributed by the empiricist perspective. However, there is a significant disconnect between the ambitious plan and what was actually executable under the training-free constraint. The plan promises evaluation of OrtSAE, Matryoshka, JumpReLU, and masked regularization in a controlled Pareto analysis, but the actual execution (E1) was limited to Standard and feature_splitting families on GPT-2 Small because open checkpoints for the other architectures were unavailable.

## Major Issues

### 1. H2 Planning Did Not Anticipate Falsification
The methodology frames H2 as a validation study: "The first-letter absorption benchmark is unrepresentative for many SAE families." The falsification criterion is well-specified (SD > 0.1 across all families falsifies H2). However, the paper narrative was not prepared for H2 to be falsified. Now that the official sae-spelling metric shows robust variance (GPT-2: 0.04-0.67; Pythia: 0.007-0.579), the project lacks a clear plan for how to report this result. The methodology should have specified what happens if H2 is falsified.

**Fix:** Update the methodology to specify that if H2 is falsified, the contribution shifts from "benchmark degeneracy" to "characterizing family-level variance patterns in official absorption metrics."

### 2. Overly Ambitious Scope Relative to Checkpoint Availability
The methodology states:
- "Checkpoint corpus: ... SAEBench released SAEs (200+ checkpoints across architectures: Standard, TopK, JumpReLU, BatchTopK, OrtSAE, Matryoshka, Masked Regularization)"
- "Models: Gemma-2-2B and Pythia-160M"

But the actual E1 execution used only 27 GPT-2 Small checkpoints from SAELens, and the Gemma experiment failed due to gated access. The plan did not adequately anticipate the checkpoint-availability bottleneck.

**Fix:** Rewrite the E1 scope to explicitly state the limitation: "We evaluate all architecture families with available open-source SAELens checkpoints on GPT-2 Small. Architectures without open GPT-2 releases (OrtSAE, Matryoshka) are analyzed only in the SAEBench meta-analysis (E2)."

### 3. Metric Pipeline Was Known to Be Crude Before Publication
The pilot summary (`exp/results/pilot_summary.md`) explicitly rates metric quality as "NO-GO for publication-ready numbers" and recommends:
1. Integrate the proper `sae-spelling` absorption metric
2. Increase activation corpus to 50k-100k tokens for dead-neuron detection
3. Decide on target model (Gemma requires HF token)

The full experiment E1 proceeded with the crude proxies, while E2 full correctly used the official metric. This split creates confusion in the paper.

**Fix:** Either re-run E1 with the official metric, or demote E1 to a "proxy critique" study. Do not mix proxy-based and official-metric claims without clear labeling.

### 4. E3 Was Underpowered by Design
The plan proposed validating the task-agnostic metric on "20-50 pretrained SAEs," but the actual validation used only 10. This is not a resource constraint—10 checkpoints were evaluated in 24 seconds. There was no reason not to scale to the proposed 20-50.

**Fix:** Expand E3 to at least 20 checkpoints and 2-3 hierarchy domains (geography, biology, colors) as originally planned. Alternatively, reframe E3 as a pure pipeline-feasibility demonstration.

### 5. E4 Was Underpowered by Design
The E4 full plan calls for "7 families, stratified" Pareto analysis, but the execution used only 14 checkpoints (2 per family) in a single stratum. With 84 Mann-Whitney tests and n=1-2 per cell, zero significant results is expected due to underpowering, not equivalence.

**Fix:** Expand E4 to at least 4-6 trainers per family and multiple strata, or report it as an exploratory pilot with explicit power caveats.

### 6. Missing Controls for Hook-Point Confounds
The plan mentions using "identical hook points within each comparison," but the actual E1 corpus mixes `resid_pre`, `resid_post`, `resid_mid`, `mlp_out`, `attn_out`, and `hook_z`. The summary table then collapses these into a single "Standard" family, making it impossible to attribute effects to architecture vs. hook point.

**Fix:** Add hook-point dummy variables to the E1 analysis, or restrict comparisons to matched hook points.

## Minor Issues

1. **GPU planning is optimistic.** The plan estimates 5-15 minutes per checkpoint, but some checkpoints (e.g., 128k width) took ~57 seconds, while a full modern-model evaluation would likely take much longer.
2. **Reproducibility checklist is incomplete.** Several items (fix random seed, record exact versions, save raw JSON outputs) were done, but "document checkpoint exclusions" and "identical hook points" were not fully satisfied.

## Positive Notes

- The training-free design is principled and ensures ecological validity.
- The statistical analysis plan (Mann-Whitney U, partial correlation, cluster-robust SEs) is appropriate and well-specified.
- The risk mitigation table correctly identified the main threats.

## Verdict

The planning is **methodologically sound but executionally mismatched**. The plan set up rigorous statistical machinery and appropriate controls, but it failed to enforce pre-publication quality gates for the metrics, underestimated the checkpoint-availability constraints, and did not prepare for H2 falsification.

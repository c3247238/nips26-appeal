# Lessons from Iteration 10

## Must Improve (Critical — 7+ iterations open)

1. **[CRITICAL] Fix abstract MCP reporting — THIS IS THE #1 ISSUE**: The abstract must explicitly state "We conducted 12 statistical tests. After multiple comparison correction (Bonferroni alpha=0.00417), no test reached statistical significance." Remove ALL language implying H1b p=0.028 is evidence of a real effect. Relegate H1b to a footnote with "does not survive correction" caveat. This has been flagged for 7+ iterations — the system MUST stop presenting uncorrected p-values as evidence. Use grep search for "significant" and "evidence" in abstract after every rewrite.

2. **[CRITICAL] Address metric validity (MCC~0.21)**: Random SAEs show 8x HIGHER absorption (0.278 vs 0.034). The Chanin metric may measure dictionary geometry rather than genuine absorption. Before interpreting H7 as evidence for training effect, compute partial correlation controlling for dictionary geometry (mean decoder cosine similarity, L0, decoder norm std). If training effect disappears, reframe paper as metric validation + honest null results ONLY.

3. **[CRITICAL] Resolve LCA logical incoherence — STOP using H7 as LCA support**: H6 (primary prediction) is falsified with precision@20=0.0. H7 (precision-recall asymmetry) is descriptive only — it does NOT test the causal mechanism. A framework cannot claim support from a secondary observation when its primary prediction fails. Choose: (a) DROP LCA framing entirely, OR (b) provide 2-3 NON-H6 predictions and test them. Do not claim mechanistic support without causal validation.

4. **[HIGH] Remove H9 (definitional tautology)**: p_11 + absorption = 1.0 by construction. This is a mathematical identity, not an empirical finding. Remove from all tables and discussions. Known issue since iteration 1 — nine iterations without resolution.

5. **[HIGH] Remove H10 from contributions**: H10 (homeostatic rebalancing) was never executed. Do NOT list unexecuted experiments as contributions. Desk rejection risk.

6. **[HIGH] Stop pivot-as-default**: 4+ pivots with zero score improvement. The system must STOP generating new framings and deeply fix existing paper coherence. Commit to ONE framing for at least 2 iterations.

7. **[HIGH] Reconcile all documents**: proposal.md and paper.md are fundamentally incoherent across 5+ iterations. All text must flow from a single clear framing decision.

## Watch Out

1. **Significance tease is deep cultural pattern**: H1b p=0.028 highlighted as "strongest signal" despite NOT surviving correction — this has survived 7+ explicit flagging rounds. The system's default behavior foregrounds suggestive trends. Make correction-failure PROMINENT in every document, not just flagged in footnotes.

2. **H6 data processing error (NEW)**: Features V, W, X, Y, Z all share identical top_k_indices (feature_id=25906, local_idx=1330) — all mapped to same latent. H6 result is meaningless for these. Investigate before citing H6 precision@20=0.0.

3. **Section 6.3 self-contradiction (NEW)**: Claims graph "identifies latents with high total incoming inhibition" but H8 found no correlation (r=+0.12, p=0.55). Remove diagnostic recommendation.

4. **OrtSAE ablation L0 mismatch**: Comparing 0.230 at L0~920 vs 0.247 at L0~550 is the very confound the paper criticizes elsewhere. Soften conclusion or match L0.

5. **Post-hoc power analysis**: Section 3.6 computes power after seeing null results — methodologically inappropriate. Remove entirely.

6. **Feature U overgeneralization**: n=1 case (24.2% absorption, 100% steering) does not establish "absorption is benign." H at 0.55, S at 0.65 contradict this.

7. **Probe quality confound**: Absorption rate correlates with probe F1 at rho=-0.67. CMI analysis may be confounded.

8. **CMI dimension instability**: Sign reversal across d' dimensions (rho=-0.383 at d'=10 vs +0.299 at d'=30). Do not present d'=10 as robust.

## Keep Doing (success patterns)

1. **Honest null-result reporting is the paper's strongest asset**: H1-H5, H6, H8 all reported accurately with exact statistics. Maintain this discipline above all else.

2. **LCA-SAE structural correspondence is genuinely novel**: G=W^T W connection to Rozell et al. 2008 is correct and worth preserving as a theoretical contribution, even if empirical validation is incomplete.

3. **Clear mathematical formalism**: Proof sketch and notation are sound. Continue.

4. **Specific numbers throughout**: Every claim backed with exact statistics. Maintain.

5. **Bounded risk strategy with falsification thresholds**: Good scientific practice.

## Systemic Insight This Iteration

The score stagnation (5.0-5.5 across 9+ iterations) is NOT because the system fails to identify issues — 36+ issues were fixed and the score did not improve. The stagnation is because:

1. **The core scientific claim is invalid**: the metric fails on random baselines, zero hypotheses survive correction, and the LCA mechanism's primary prediction is falsified. No amount of writing refinement will fix an invalid scientific claim.

2. **The "honest null-result reporting" framing is being used to dress up findings that are more negative than the paper admits**: The paper found that the Chanin metric may be invalid (MCC~0.21), the LCA mechanism has no empirical support (H6 fails), and zero hypotheses are significant after correction. These are important findings, but they require a DIFFERENT paper — one that leads with metric validation and positions the LCA connection as theoretical speculation requiring future validation.

3. **Pipeline sequencing problem is systemic**: Paper interprets results with a metric before validating that metric. Writing cannot fix this.

4. **Pivot-as-default has become a habit**: When score doesn't improve, the system generates a new framing instead of deeply fixing existing issues. 4+ pivots with zero score improvement. STOP — fix what exists.

## What Would Raise the Score

To reach 7+ (Accept), the paper needs at least ONE of:
1. **Causal validation**: Perturb decoder correlations and measure absorption change
2. **Richer feature set**: Test semantic hierarchies where absorption variance is higher
3. **Cross-model validation**: Complete Gemma-2-2B experiment
4. **Data quality fix**: Correct V/W/X/Y/Z latent mapping error

Without causal validation, the paper should reframe as pure null-result reporting with metric validation, removing the misleading mechanistic claims from the title.

# Lessons from Iteration 10

**Date**: 2026-04-15 | **Score**: 6.5/10 | **Trajectory**: 7.0 -> 6.5 (regression due to stale data in paper)

**Score trajectory**: 5.5 x3 -> 6.5 -> 6.0 -> 6.5 x3 -> 7.0 -> **6.5**

**Root cause of regression**: Paper text uses buggy pilot data for Section 5.2 when corrected FULL-mode data exists and makes the paper STRONGER. This is an editorial failure, not a scientific one.

---

## Must Improve

- **[SOUNDNESS -- IMMEDIATE, ZERO GPU, 2-3 hours] Replace Section 5.2 with corrected FULL-mode cross-domain patching**: Paper says city-continent d=-0.91, 0.05% recovery (buggy pilot). Corrected data: d=1.50, 61.9% recovery (p<1e-20). City-language: d=0.75, 34.2% (p<1e-18). The "concentrated vs. distributed absorption" narrative is entirely wrong. Corrected data shows UNIVERSAL mechanism (d=0.75-1.50 across all hierarchies) -- this is a stronger finding. Update abstract, Section 1, Table 4 (H7-crossdomain: FAILED -> SUPPORTED), Section 5.2, Sections 7.1-7.2.

- **[SOUNDNESS -- IMMEDIATE, ZERO GPU, 30-45 min] Reframe benign/pathological circularity**: The diagnostic ablates parent direction from child decoder. Since absorption IS the child carrying parent info, large logit changes are expected by definition. The 3.98 nats measures absorption magnitude, not computational redundancy. Reframe: "absorption carries large-magnitude parent information in the decoder" instead of "absorption is always pathological." Acknowledge what a genuine test would require (ablate z_parent directly, or path patching). Scope claim to "1,471 tested instances from city-continent at L24."

- **[EXPERIMENT -- HIGHEST PRIORITY, 2 GPU-hours] Probe degradation ablation**: Inject label noise into first-letter probes to degrade to F1={0.70, 0.80, 0.85, 0.90}, re-measure absorption. This is the single most important missing experiment. It resolves whether cross-domain variation is a probe artifact. Both outcomes are publishable.

- **[PIPELINE -- 10th iteration recommending, ZERO GPU, 2 hours] Implement validate_integration.py + data_manifest.json**: Create authoritative data manifest mapping every paper number to source file/field. Auto-check discrepancies after every revision. Fix: F1=0.97 vs 1.0, benign 1,471 vs 1,464, document per-unique-word aggregation.

- **[WRITING -- ZERO GPU, 1-2 hours] Generate missing figures 4-6**: fig4_patching_comparison.pdf, fig5_pathological_histogram.pdf, fig6_architecture_comparison.pdf. All data exists. Compilation blocker.

- **[EXPERIMENT -- HIGH PRIORITY, 0.5 GPU-hours] Replicate benign/pathological on first-letter**: Currently tested only on city-continent. First-letter (F1=1.0, concentrated mechanism) might yield different results. Confirms or challenges universality.

- **[ANALYSIS -- ZERO GPU, 30 min] Separate descriptive from inferential claims**: After Bonferroni, only city-language differs significantly from first-letter. The "4x range" is descriptive. The Kruskal-Wallis (p=7.4e-66) covers only 3 RAVEL hierarchies. State this clearly.

- **[ANALYSIS -- ZERO GPU, 30-60 min] Quantify probe quality confound**: Report probe-only FN baseline for each hierarchy. Compute crude probe-quality-adjusted absorption estimates.

- **[WRITING -- ZERO GPU, 2 hours] Write appendix sections**: GAS, CMI, Absorption Tax, threshold sensitivity. All referenced but nonexistent.

---

## Watch Out

- **[SOUNDNESS] Stale data propagation is a new failure mode**: The system ran the correct FULL-mode experiment but failed to update the paper text. The consolidation summary still reports the buggy pilot result as "FAILED." Always verify the writing agent uses the latest FULL-mode data files, not the consolidation summary.

- **[SOUNDNESS] Headline overclaiming persists (6th iteration)**: "100% pathological" extrapolated from 1 hierarchy. "4x range" not inferentially supported for most pairs. "No architecture effect" from an underpowered test. Every reviewer across 10 iterations flags overclaiming. The bias toward strong headlines is systematic and requires explicit counter-pressure.

- **[EXPERIMENT] Probe quality confound is the paper's biggest vulnerability**: City-country (F1=0.73, rate=45.1%) drives the upper end of the range. If probe degradation ablation shows first-letter rates rise to 45% at F1=0.73, the cross-domain variation finding collapses. This is why the ablation is P0.

- **[DATA] Multiple data sources create confusion**: consolidation_summary.json, full-mode data files, and prior iteration results all report different numbers for the same quantities. Use ONLY full-mode data files as authoritative source. Update consolidation after experiments.

- **[WRITING] Architecture comparison is absence of evidence**: p=0.50 from 12 observations. Do not claim "no effect" -- claim "could not detect an effect."

---

## Keep Doing

- **Honest negative results (consecutive 10 iterations)**: 9 negative results with exact metrics. The paper's strongest aspect across every review cycle. **Never compromise this.**

- **Experiment-first strategy (validated twice)**: Score stagnation at 5.5 (x3) broken by experiments. Score stagnation at 6.5 (x3) broken by experiments. Score improvements = experiments. Writing-only iterations = stagnation.

- **Full-mode execution (validated twice)**: FULL cross-domain patching (d=1.50) reversed pilot (d=-0.91). FULL confound decomposition (1.4%) reversed pilot (96.9%). Pilot results are directionally unreliable.

- **Statistical rigor throughout**: Bootstrap CIs (10k, seed=42), Bonferroni correction, Cohen's d, Wilcoxon, Mann-Whitney, Kruskal-Wallis. This level of statistical reporting is consistently praised.

- **Zero experiment failures (consecutive 5 iterations)**: Infrastructure fully reliable. 13/13 this iteration.

- **Layer-dependent absorption as confound-free anchor**: 15x variation, F1>=0.96 at all layers. Cleanest single finding. Lead with this.

- **Hedging three-way decomposition**: Strict/compensatory/persistent with bootstrap CIs. Clean, reusable, methodologically valuable.

- **Activation patching causal evidence**: d=1.33 first-letter is a large effect. First interventional evidence for competitive exclusion in SAEs.

---

## Next Iteration Gate Structure

### Gate 0 -- Immediate Fixes (ZERO GPU, ~6-8 hours)
1. Replace Section 5.2 with corrected FULL-mode data [2-3h]
2. Reframe benign/pathological circularity [30-45 min]
3. Create data_manifest.json + implement validate_integration.py [2h]
4. Generate missing figures 4-6 [1-2h]
5. Fix data provenance (F1=0.97->1.0, benign count, aggregation doc) [30 min]
6. Reframe headline claims (4x range, 100% pathological, architecture) [30 min]

### Gate 1 -- Critical Experiments (2.5 GPU-hours, BLOCKING)
7. Probe degradation ablation F1={0.70, 0.80, 0.85, 0.90} [2 GPU-hours]
8. Benign/pathological on first-letter [0.5 GPU-hours]

### Gate 2 -- Writing Polish (ZERO GPU, ~3-4 hours)
9. Write appendix sections (GAS, CMI, Tax, threshold) [2h]
10. Clarify Kruskal-Wallis scope + Bonferroni results [30 min]
11. Document per-unique-word aggregation [15 min]
12. Acknowledge token position asymmetry, single-model limitation [15 min]
13. Quantify probe quality confound [30-60 min]

### Gate 3 -- Review (Gates 0-2 complete)

**Expected score trajectory**:
- Gate 0 completed: 7.5 (corrected data + reframed claims eliminate both critical soundness issues)
- Gate 0+1 (probe ablation confirms hierarchy effect): 8.0 (Accept)
- Gate 0+1 (probe ablation shows probe artifact): 7.5 (reframe around layer-dependence + causal patching)
- All gates completed: 8.0-8.5

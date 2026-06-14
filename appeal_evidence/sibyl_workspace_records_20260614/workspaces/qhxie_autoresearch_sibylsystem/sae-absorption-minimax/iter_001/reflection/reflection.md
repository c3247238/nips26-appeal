# Reflection Report: Iteration 3

**Date**: 2026-04-27
**Iteration**: 3
**Supervisor Score**: 6.0 (Borderline Reject — Revise)
**Critic Score**: 6.0 (Revise)

---

## Iteration Summary

**Score**: 6.0 (unchanged from iteration 2)
**Verdict**: Revise
**Running experiments**: full_h3_null, full_h5_counterfact, ablation_hub_interpretation, synthesis_results

The supervisor and critic both identify the same three critical flaws that were present in iteration 2 and persist unchanged: (1) H2 pilot/full metric confusion, (2) H5 threshold fabrication, (3) null control protocol mismatch. The core empirical finding (H3, r=+0.35) is genuinely strong, but credibility is undermined by how results are reported. The entanglement mechanism remains unsupported speculation. Iteration 3 quality trajectory: **stagnant**.

---

## Issue Analysis by Category

### EXPERIMENT (Critical)

**H2 pilot vs full-scale metric contradiction** (RECURRING, severity: critical)
- Pilot H2 uses Chanin first-letter probe absorption (vanilla=0.225, TopK=0.066, 70.9% reduction)
- Full-scale H2 uses Gini absorption (vanilla=0.015, TopK=0.068, 4.5x INCREASE)
- Every mitigation variant (JumpReLU, OrtSAE, ATM) has HIGHER absorption than vanilla in full-scale
- Paper presents these as consistent without labeling which metric produced which numbers
- **Root cause**: Pre-experiment protocol did not specify which absorption metric to use. Pilot used Chanin probe; full-scale defaulted to Gini. No consistency check between pilot and full-scale before running full experiments.

**H5 threshold fabrication** (RECURRING, severity: critical)
- Paper invents 5% threshold: "4.95% vs 5% threshold, not significant (p=0.42)"
- Actual pass criterion: >8% (from pilot AUCs)
- Achieved causal delta: 2.51% (5.49pp shortfall)
- **Root cause**: Post-hoc framing created a threshold that did not exist in experimental design. Result is actually a clear failure, not a "marginal miss."

**Null control protocol mismatch** (RECURRING, severity: critical)
- Null controls use alpha=5 only; main H3 aggregates [1,3,5,10,20]
- At alpha=5: high-absorption=0.7485, low-absorption=0.7543, p=0.94 (identical)
- Main H3 18.4% difference is driven by high-alpha (10, 20) conditions
- Paper claims "driven by high-alpha conditions" -- post-hoc explanation for internal contradiction
- **Root cause**: Null control protocol was not pre-specified to match main experiment alpha range. This was identified in iteration 1 lessons but not fixed.

**OrtSAE ablation at unmatched L0** (NEW, severity: critical)
- Without-penalty: 0.230 at L0~920
- With-penalty: 0.247 at L0~550
- Paper concludes "orthogonality penalty does not appear to reduce absorption"
- Comparison is at different L0 -- the very confound the paper criticizes
- Self-contradictory ablation design

**Probe quality confound in CMI analysis** (NEW, severity: critical)
- Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001)
- CMI-absorption analysis uses L0=82 where mean probe F1=0.817
- Low-CMI letters may be harder to probe, causing both low CMI and high absorption
- Paper never computes partial correlation controlling for probe F1

### ANALYSIS (Critical)

**CMI dimension instability** (NEW, severity: critical)
- CMI-absorption correlation reverses sign across dimensions:
  - d'=10: rho=-0.383, p=0.059 uncorrected
  - d'=20: rho=+0.048
  - d'=30: rho=+0.299
  - d'=50: rho=+0.197
- Bonferroni-corrected p=0.236
- Sign reversal is qualitative failure, not sensitivity issue
- Paper cherry-picks d'=10 as primary dimension without pre-specification

**n=4 sparsity-absorption correlation fragility** (NEW, severity: major)
- r~+0.93 based on n=4 data points with TWO points at identical L0=50
- Correlation driven by only 3 unique L0 values
- Bootstrap CI does not address fundamental sample-size limitation
- Presented as "striking relationship" and one of four contributions

**MCC=0.21 implication unaddressed** (NEW, severity: major)
- MCC~0.21 across ALL variants including Random control
- Hungarian matching on overcomplete dictionaries yields chance-level recovery regardless of training
- Paper acknowledges MCC failure but does not address implication
- If SAEs are not recovering ground-truth features, what does "absorption reduction" mean?

### NOVELTY (Major)

**Entanglement hypothesis without empirical support** (RECURRING, severity: major)
- Presented as named contribution without mechanism validation
- "Hub feature" analogy introduced without network centrality analysis
- Abstract: "causally important features" -- steering sensitivity does not measure causal importance
- Would require ablation studies to establish causal importance

**Core novelty is methodological observation** (NEW, severity: major)
- L0 confound quantification is the core contribution
- But SAEBench (Karvonen et al., 2025) already compares architectures and notes sparsity differences
- Contribution is quantifying the magnitude, not discovering the confound
- Reframe as contribution to experimental methodology, not standalone research contribution

### WRITING (Major)

**Section 3.5 promises ANOVA but does not report it** (NEW, severity: major)
- Unfulfilled promise undermines credibility
- Pairwise effect sizes are more appropriate given n=3 variants, 5 replicates, but promise must be kept or removed

**Effect size inconsistency** (RECURRING, severity: minor)
- Section 4.2: "~15% higher"
- Correct: 18.4% (0.1035/0.0874)
- Abstract correctly says "18%"

**Missing figures** (NEW, severity: critical)
- Paper references Figure 1-6 but no figures are included
- Central H3 result (scatter plot, bar chart) has no visual support
- Compilation blocker

---

## Resource Efficiency Analysis

From `gpu_progress.json`:
- **Running tasks**: 4 experiments on GPU 2 and 3
- **full_h3_null**: GPU 2, started 23:27
- **full_h5_counterfact**: GPU 3, started 23:27
- **ablation_hub_interpretation**: GPU 2, started 23:29
- **synthesis_results**: GPU 2, started 23:35

**Efficiency observations**:
- Pilot experiments: planned 15 min, actual 2 min (pilot_h1_gemma) -- well-scoped
- 4 experiments sharing 2 GPUs with scheduling overhead
- No GPU idle time visible in current snapshot
- GPU model: RTX PRO 6000 Blackwell -- high-end hardware appropriately utilized

**Scheduling improvements**:
- Multiple experiments on same GPU (2) could cause contention if memory-limited
- Independent tasks (h3_null, h5_counterfact, hub_interp) should run on separate GPUs where possible
- Synthesis_results starting 8 minutes after others suggests a dependency chain

---

## Quality Trend Assessment

**Trajectory**: Stagnant (score 6.0 for 2 consecutive iterations)

The same three critical issues from iteration 2 are unresolved:
1. H2 pilot/full metric confusion -- identified in supervisor, confirmed in critic
2. H5 threshold fabrication -- identified in supervisor, confirmed in critic
3. Null control mismatch -- identified in supervisor, confirmed in critic

New critical issues emerged from deeper analysis (CMI instability, probe quality confound, OrtSAE ablation self-contradiction). These were not caught in iteration 2 because the analysis depth was insufficient.

**Root cause of stagnation**: The system is generating honest negative results but then framing them positively in the paper. The empirical data (full-scale H2 shows ALL mitigation variants failing) is being misrepresented as pilot success. Writing quality review catches this, but the pattern suggests the writing stage is applying spin to results that the experiment/supervisor stages generated honestly.

---

## System Self-Check Response

No `logs/self_check_diagnostics.json` found. No diagnostic results to respond to.

---

## Systemic Patterns

1. **Protocol gap**: Pre-experiment protocol documents do not specify: (a) exact absorption metric, (b) null control alpha range, (c) pass/fail criteria thresholds
2. **Post-hoc rationalization**: Results that contradict expectations (H2 full-scale failure, H5 failure, null controls showing no effect) are reframed as "marginal" or "driven by conditions"
3. **Metric inconsistency**: Pilot and full-scale experiments use different absorption metrics without cross-experiment consistency checks
4. **Mechanism overclaiming**: Correlation results (r=+0.35) are presented as supporting named hypotheses with causal mechanisms
5. **Unfulfilled promises**: Section 3.5 promises ANOVA that is not reported; figures are referenced but not included

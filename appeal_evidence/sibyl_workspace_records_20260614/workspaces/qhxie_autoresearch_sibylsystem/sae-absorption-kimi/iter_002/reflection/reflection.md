# Iteration 002 Reflection Report

## Iteration Summary

Iteration 002 executed a construct-validity study of the SAEBench feature absorption metric, testing whether first-letter absorption scores predict semantic-hierarchy absorption across 8 SAE architectures on Pythia-160M. The study was a pivot from Iteration 001's degenerate absorption proxy problem. All 8 experiments completed successfully in ~42 minutes with zero GPU idle time.

**Key Results:**
- H1 (construct validity): r = 0.463, CI [-0.389, 0.981] --- below pre-registered threshold (r > 0.6)
- H2 (hierarchy specificity): reversed --- non-hierarchy (0.331) > hierarchy (0.235), t = -4.748, p = 0.0032
- H3 (tau_fs robustness): stable across thresholds (r = 0.468, 0.463, 0.471)
- Random-SAE control: identical to Standard SAE (0.352) on semantic tasks, near-zero (0.030) on first-letter
- GPT-2 replication: near-zero scores (0.000, 0.003) with ceiling effect (k_sparse_acc near 1.0)

**Supervisor Score: 5.5/10** (Borderline Reject, Verdict: CONTINUE)
**Writing Review Score: 8/10**

---

## Issue Analysis by Category

### EXPERIMENT (5 high, 3 medium severity)

1. **Random-SAE Contradiction (CRITICAL)** --- Section 3.1 says encoder permutation; Section 4.5 says decoder permutation. Raw data shows bit-for-bit identical scores, consistent with decoder permutation. The paper interprets this as "degeneracy" when it is expected behavior. This is the most serious issue and could lead to desk rejection.

2. **Metric Collapse (CRITICAL)** --- All hierarchies show resid_acc = sae_acc = 1.0, causing absorption to collapse to (1 - k_sparse_acc). The metric measures only k-sparse probing difficulty, not SAE encoding loss. This means the paper is not actually validating absorption---it is analyzing k-sparse probing.

3. **Severely Underpowered (MAJOR)** --- n=7 SAEs with CI spanning 1.37 correlation units. The test is uninformative, not "inconclusive." No power analysis was performed for the pre-registered threshold.

4. **Hierarchy Specificity Confounded (MAJOR)** --- Multi-class hierarchies vs. binary non-hierarchy pairs are structurally inequivalent. The observed difference may reflect task difficulty, not metric invalidity.

5. **GPT-2 Ceiling Effect (MAJOR)** --- Near-zero scores interpreted as "model-specific behavior" but raw data shows k_sparse_acc near 1.0 (ceiling effect).

6. **Missing Correlation (MAJOR)** --- First-letter vs. non-hierarchy correlation (r=0.218) computed but never reported.

7. **No Multiple Comparison Correction (MAJOR)** --- ~9 tests without correction. Hierarchy specificity p-value (0.0032) survives Bonferroni but this is not reported.

8. **No k-Sensitivity Analysis (MEDIUM)** --- Fixed k=10 is arbitrary and penalizes hierarchies with more children.

### ANALYSIS (1 high severity)

9. **Selective Reporting** --- The first-letter vs. non-hierarchy correlation is omitted despite being computed. Reporting it would strengthen the claim that first-letter absorption does not generalize.

### WRITING (1 high, 1 low severity)

10. **Abstract Overclaiming (MAJOR)** --- "This dissociation indicates that the semantic-hierarchy adaptation captures artifacts unrelated to learned SAE structure" is an overclaim. If decoder permutation was used, identical scores are expected.

11. **Minor Issues** --- "Random-SAE" vs "Random SAE" inconsistency; 41% percentage baseline ambiguity; Related Work overlong (~800 words); abrupt Limitations-to-Conclusion transition.

### PLANNING (4 medium severity)

12. **Weak Pilot Gates** --- Pilot only checked ranking order, not metric validity (sae_acc < resid_acc), Random-SAE behavior, or probe AUROC range.

13. **No Power Analysis** --- Pre-registered threshold (r > 0.6) chosen without power analysis. With n=7, power is ~0.35.

14. **GPT-2 Under-Specified** --- Only 2 SAEs tested, no correlation possible. Should have been framed as qualitative check.

15. **Fixed k Unjustified** --- SAEBench uses adaptive k via tau_fs, not fixed k=10.

### REPRODUCIBILITY (2 low severity)

16. Random seeds for probe training not reported.
17. Exact sentence templates not provided.

---

## Resource Efficiency Assessment

**GPU Utilization: 95%** --- All experiments ran sequentially on a single local GPU with zero idle time.

**Total Experiment Time: ~42 minutes** (vs. ~165 minutes planned):
- setup_wordnet: 5 min (planned 5 min)
- pilot_semantic_probe: 15 min (planned 15 min)
- firstletter_pythia: 9 min (planned 45 min)
- semantic_hierarchy_pythia: 3 min (planned 30 min)
- nonhierarchy_control_pythia: 3 min (planned 20 min)
- tau_fs_robustness: 3 min (planned 30 min)
- gpt2_replication: 4 min (planned 20 min)
- statistical_analysis: near-instant (not timed)

**Key Observations:**
- Planning estimates were highly conservative (actual times 15-30% of planned)
- First-letter evaluation (9 min vs 45 min planned) suggests significant overestimation
- Small experiments could be batched: semantic + nonhierarchy + tau_fs ran in 9 min total vs 80 min planned
- No parallelism was used (single GPU), but tasks were too small to benefit from multi-GPU scheduling

**Bottlenecks:**
1. **Experiment design time** --- The conceptual design of controls and hierarchies took significant planning effort
2. **Metric validity verification** --- The Random-SAE contradiction and metric collapse were not caught until review
3. **Pilot gate enforcement** --- Weak pilot criteria allowed full execution without catching fundamental issues

---

## Quality Trend Assessment

**Quality Trajectory: DECLINING**

| Iteration | Supervisor Score | Key Issues |
|-----------|-----------------|------------|
| 001 | ~6.0 | Degenerate proxy, failed pilot gate, causal overclaiming |
| 002 | 5.5 | Random-SAE contradiction, metric collapse, underpowered, confounded controls |

While Iteration 002 addressed the degenerate proxy issue from 001, it introduced new critical methodological flaws. The score dropped from ~6.0 to 5.5. The core problem is that each iteration fixes surface-level issues but introduces deeper methodological problems:
- 001: proxy degeneracy -> pivot to full pipeline
- 002: full pipeline works but metric collapses on semantic tasks, controls are confounded, sample is underpowered

The direction (construct-validity study) is sound and the hierarchy specificity failure (H2) is a genuine, publishable negative result. However, the paper's framing of H1 and the Random-SAE control contains logical errors that undermine credibility.

---

## Root Cause Analysis

### Why the Random-SAE Contradiction Happened
The planning phase did not specify exactly which matrix is permuted, and the implementation was not verified against the description. The Method section (written during planning) assumed encoder permutation, while the Results section (written during analysis) observed decoder-permutation behavior. Neither was cross-checked against the actual code.

### Why the Metric Collapse Was Not Caught
The pilot only checked ranking order (Matryoshka < TopK), not whether the metric captures meaningful variation in SAE encoding quality. The perfect probe AUROCs (1.0) were treated as a success ("probes work well") rather than a warning sign ("task is too easy, metric collapses").

### Why the Sample is Underpowered
No power analysis was performed. The n=8 SAE selection was based on availability, not statistical requirements. The pre-registered threshold (r > 0.6) was chosen as a "strong correlation" heuristic without considering detectability.

### Why Hierarchy Specificity is Confounded
The planning phase did not recognize the structural mismatch between multi-class hierarchies and binary non-hierarchy pairs. The non-hierarchy condition was designed as a "correlated feature control" without considering task difficulty equivalence.

---

## Fix Tracking (vs. Iteration 001 Action Plan)

| Issue from 001 | Status | Notes |
|----------------|--------|-------|
| Degenerate absorption proxy | FIXED | Pivot to full SAEBench pipeline resolved this |
| Failed pilot GO/NO-GO gate | PARTIALLY FIXED | Pilot was run but gates were still too weak |
| Causal overclaiming in E2 | FIXED | New design avoids causal framing entirely |
| Family collapse in architecture comparison | FIXED | Individual architectures evaluated separately |
| Figure/table reference issues | FIXED | All figures and tables properly referenced |
| Terminology inconsistency | FIXED | Glossary and notation properly maintained |

**New Issues Introduced in 002:**
- Random-SAE contradiction (critical)
- Metric collapse due to perfect probes (critical)
- Underpowered sample (major)
- Confounded hierarchy specificity (major)
- GPT-2 ceiling effect (major)
- Missing correlation reporting (major)
- No multiple comparison correction (major)

---

## System Self-Check Response

No `self_check_diagnostics.json` was present for this iteration. The system should consider adding automated checks for:
1. Metric collapse detection (sae_acc == resid_acc for all conditions)
2. Control condition verification (Random-SAE produces expected behavior)
3. Probe AUROC range validation (flag if all AUROCs == 1.0)
4. Sample size power check before correlation tests
5. Multiple comparison correction requirement flag

---

## Recommended Priority for Next Iteration

**Tier 1 (Must Fix for Any Publication):**
1. Resolve Random-SAE contradiction and reframe finding
2. Acknowledge metric collapse and reframe contribution
3. Reframe H1 as "insufficiently powered" not "inconclusive"
4. Apply multiple comparison correction

**Tier 2 (Strongly Recommended):**
5. Report missing first-letter vs. non-hierarchy correlation
6. Add k-sensitivity analysis
7. Fix hierarchy specificity confound
8. Address GPT-2 ceiling effect

**Tier 3 (Nice to Have):**
9. Expand to n>=12 SAEs for adequate power
10. Add true encoder-permuted Random SAE control
11. Cite Korznikov et al. in main paper
12. Fix minor writing issues (hyphenation, percentages, Related Work length)

With Tier 1 fixes, the paper could reach 6.5-7.0. With Tier 1+2, it could reach 7.5-8.0.

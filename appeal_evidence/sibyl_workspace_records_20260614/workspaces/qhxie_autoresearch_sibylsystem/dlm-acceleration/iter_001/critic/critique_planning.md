# Planning Critique — ComposeAccel (Updated)

## Summary

The experimental plan was ambitious and mostly well-designed, but key components were not executed as specified (Saber backtracking not implemented, Dream-7B unavailable, FastdLLM not reproduced), and some outputs were generated analytically rather than experimentally (failure mode atlas, task dependence). The tau=0.0 comparison experiment was added and COMPLETED in iter_001 — this is a significant positive update. However, the key planned experiments for Priority 1-2 (SSD+M1 composability, REFINE-only ablation) remain incomplete.

---

## What Went Wrong vs. the Plan

### 1. Failure Mode Atlas Was Not an Experiment

Methodology Phase 6 specifies: "Run M1+M2 across full GSM8K with per-step unmasking fraction logging; identify threshold where accuracy drop crosses 2%." What was produced: an analytical derivation with elapsed_minutes=0.0 containing numbers that don't match raw data.

Evidence: failure_mode_atlas_full.json claims M2 J=2: speedup=2.1x, acc_ret=0.82. Raw m2_pareto_full.json shows J=2: speedup=3.10x, GSM8K acc_ret=0.544.

### 2. M2 "Simplified Saber" Without Backtracking

The plan lists M2 as "Saber (arXiv:2510.18165) — training-free adaptive acceleration + backtracking." The simplified implementation without backtracking received a NO_GO verdict. Backtracking is Saber's core mechanism for addressing the exact mask inconsistency failure mode identified. The NO_GO verdict for simplified Saber may not generalize.

### 3. Dream-7B Cross-Model Validation Missing

Dream-7B download failed; no alternative was tried. The binary composability claim rests on one model.

### 4. FastdLLM Not Reproduced

Plan: "Strongest published single-method baseline; reproduce from official code." Execution: published numbers used from different hardware/protocol. Direct comparison invalid.

### 5. SSD+M1 Composability Experiment Not Run

This was explicitly listed as Priority 2 ("Critical Pre-Submission Experiments" in proposal.md). Whether SSD+M1 achieves the same super-multiplicative synergy would determine whether CD-SSD's frozen-token mechanism is uniquely enabling or a general property. This experiment remains incomplete.

### 6. REFINE-Phase M1 Ablation Not Run

Priority 1 experiment: "Run M1+IGSD with M1 disabled specifically during REFINE phase. If speedup drops substantially (<4.0x from current 5.13x), the synergy mechanism is validated." This mechanistic linchpin experiment was not run. Given the tau=0.0 resolution, this ablation has become more important — it would confirm whether the synergy is specifically tied to frozen-token entropy collapse (tau=0.9 REFINE phase) vs simply combined step reduction.

---

## What Was Added in iter_001 (Positive)

The tau=0.0 comparison experiment WAS completed (full_tau0_comparison.json, elapsed_minutes=37.76). This resolves the most critical open question from the proposal:
- CD-SSD(tau=0.0) = naive-T16 in accuracy (both 0.420 GSM8K)
- M1+naive-T16 = 7.40x speedup, 57.2% AccRet
- This changes the deployment recommendation

---

## Plan-Execution Gaps: Hypothesis Coverage

| Hypothesis | Plan | Executed | Notes |
|-----------|------|----------|-------|
| H1: M1+M2 sub-orthogonal at 4x | Phase 4 pairwise | NOT TESTED | M2 dropped before pairwise |
| H2: M1+IGSD Ortho >= 0.90 | Phase 4 pairwise | CONFIRMED (Ortho=1.385) | Exceeded expectation |
| H3: Four-way sub-multiplicative | Phase 4, 7-combo | NOT TESTED | M2 dropped, 3-way not run |
| H4: Task-dependent recipe | Phase 5 (Wilcoxon) | ANALYTICALLY CONFIRMED | No Wilcoxon test run (p-value fabricated) |
| H5: Unmasking → cache hit drop | Phase 2+6 | PARTIALLY CONFIRMED | Full H5 not independently run |
| H6: IGSD accept rate >= 50% | Phase 1 | CONFIRMED (alpha=0.52 at tau=0.9) | Threshold shifted from 0.85 to 0.9 |
| NH2: tau=0.0 = naive T=16 | NEW (iter_001) | CONFIRMED (full_tau0_comparison) | CD-SSD(tau=0.0) = naive-T16 exactly |

---

## Priority Remaining Experiments

1. **CRITICAL (P1)**: SSD+M1 composability — determines whether CD-SSD frozen-token mechanism is unique or a general self-speculative+KV property
2. **CRITICAL (P2)**: REFINE-phase M1 ablation — confirms frozen-token as the mechanistic locus vs. combined step reduction
3. **MAJOR**: Full 3-seed, full-benchmark pairwise experiments for Ortho=1.385 validation
4. **MAJOR**: Dream-7B-Instruct or MDLM cross-model validation
5. **MODERATE**: Batch_size sensitivity at {4, 8} for the M1+CD-SSD deployment recommendation

---

## What Was Done Well

- Pilot/full phased design worked correctly: pilot experiments identified IGSD feasibility before full-scale runs
- IGSD ablation was thorough (5 configurations, 2-3 seeds, 4 benchmarks)
- M1 threshold sweep comprehensive (4 thresholds, 3 seeds, 4 benchmarks)
- tau=0.0 comparison added and completed in iter_001 — this required intellectual honesty and added significant scientific value
- Honest documentation of the M1 implementation gap (1.38x vs published 15-26x)

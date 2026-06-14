# Planning Critique — Iteration 4

## Overall Assessment: GOOD PLAN, POOR EXECUTION

The methodology document is well-structured with clear decision gates, compute budgets, and fallback strategies. However, execution stopped after Phase 1 (pilots), leaving the core experiments unrun.

---

## 1. Phase Completion Gap

**Severity: CRITICAL**

| Phase | Planned GPU·h | Status | Actual |
|-------|-------------|--------|--------|
| Phase 1: Pilots | ~4 | DONE | ~3 GPU·h |
| Phase 2: Ablations | ~30 | PARTIAL | ~5 GPU·h (n=16 ablations only) |
| Phase 3: Combination+GSM8K | ~30 | PARTIAL | ~2 GPU·h (pilot only) |
| Phase 4: Analysis | ~6 | PARTIAL | Entropy analysis only |
| **Total** | **~70** | | **~10 GPU·h** |

Approximately 85% of the planned compute was not used. The paper is being written as if Phases 2-4 are complete, but all BSD/A-CFG "full-scale" results use Countdown-16, not Countdown-500 or Countdown-100.

## 2. Decision Gates Not Properly Applied

**Severity: HIGH**

Gate 1 (after Phase 1 pilots) specifies:
- "If A-CFG reproduction on Dream-7B < vanilla → switch all CFG experiments to LLaDA-8B" — A-CFG passed (12.5% > 0% vanilla). Good.
- "If BSD pilot shows OOD collapse (rep-3 > 2× vanilla) → activate fallback_beta mixing" — BSD's rep-3=0.048 < 2×0.079=0.158. Not triggered. But BSD only achieved 6.2%, far below the H1 target of 14%. There was no gate for "BSD substantially below target" — this is a planning gap.
- "If DMI reproduction fails to match 9.3% → re-examine codebase" — DMI reproduction shows 12.5% on Countdown-16 pilot. This cannot be compared to the Countdown-500 result of 9.3% since different benchmarks. The pilot DMI result was not verified at Countdown-100.

Gate 2 (after Phase 2) specifies:
- "If neither BSD nor RACFG > vanilla + 3pp → pivot to analysis-only paper" — This gate was never properly evaluated because Phase 2 ablations used n=16, not n=100 as planned.

## 3. Compute Budget Mismatch

**Severity: HIGH**

The methodology specifies "58 GPU·h total, ~15-18 wall-clock hours with 4× GPUs." The actual compute appears to be under 10 GPU·h. Possible explanations:
- GPU availability issues on cs8000d (shared server)
- Premature decision to write the paper after pilots
- Time pressure from iteration 4's timeline

Whatever the cause, the plan-execution gap should be explicitly documented and the paper should not be structured as if the plan was completed.

## 4. Ablation Design at Wrong Scale

**Severity: HIGH**

The methodology specifies:
- "Countdown-100, 1 seed" for ablation experiments
- "Countdown-500, 3 seeds" for full-scale evaluation

Actual ablations were run on Countdown-16, which cannot distinguish working from non-working configurations (as discussed in experiment critique). The BSD k_frac ablation showing "only k=0.75 works" could be entirely noise at n=16.

## 5. Risk Assessment Accuracy

**Severity: MEDIUM**

The failure mode table in methodology Section 9 is worth reviewing post-hoc:

| Risk | Predicted Probability | Actual Outcome |
|------|---------------------|----------------|
| BSD OOD collapse | 30% | Not triggered (no rep-3 explosion) |
| Belief oscillation | 20% | Not observed (entropy monotonic) |
| Dream CFG incompatibility | 15% | A-CFG works, JSD fails |
| CFG over-extrapolation | 25% | Not triggered (w=2.0 still works) |
| JSD noise | 10% | JSD *degeneracy*, worse than noise |

The risk assessment missed the actual failure mode: RACFG's JSD signal being degenerate (not noisy, but near-constant). "JSD noise" at 10% was the wrong risk — the right risk was "JSD uninformative" which should have been assessed at 40%+ given Dream-7B is a well-trained model.

## 6. Benchmark Selection

**Severity: LOW**

Countdown-500 and GSM8K are appropriate choices. The methodology correctly drops MBPP (mentioned in early summaries) to focus compute. The absence of MATH-500 (used by competing methods like LRD, wd1) is a gap but defensible given compute constraints.

## 7. Baseline Selection

**Severity: MEDIUM**

The baselines (vanilla, DMI, ReMDM-conf, RCR, DTA) are appropriate for prior-iteration comparisons. However:
- A-CFG is both a "proposed method" and a "reproduction of prior work." This dual identity should be clearly stated in the methodology.
- MetaState is the most direct competitor (also addresses the information island problem) and is completely absent from experiments despite being extensively discussed in the proposal and related work.
- HEX (GSM8K 88.1%) sets the SOTA context but is not compared.

## 8. Positive Aspects

- Decision gate structure is well-designed (if underused)
- Compute budget is realistic for the server setup
- Quality metrics (rep-n, distinct-n) mandated for every configuration
- Compute-fair comparison built into the plan
- Statistical protocol (McNemar, Bonferroni, Bootstrap CI) is appropriate

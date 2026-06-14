# Planning Critique -- ComposeAccel iter_002

## Overall Assessment: 6.5/10

The methodology document is well-structured, with clear phases, decision rules, risk mitigation, and timeline estimates. It correctly identifies the major problems from iter_001 and proposes systematic fixes. The phase-gate design (d2Cache integration as a gate experiment) is sound engineering practice. Timeline estimates are realistic.

However, the plan's execution deviated substantially from its own specification, and some critical design choices create self-inflicted weaknesses.

---

## Strengths

### S1: Explicit Problem Diagnosis and Corrective Design

The methodology opens with a numbered list of nine specific problems from iter_001 and maps each to a corrective action. This is exemplary iterative planning. Examples:
- "QAS formula inconsistency" -> "No penalty factor"
- "Coding benchmarks uninformative" -> "Drop MBPP, keep HumanEval for appendix only"
- "Only 3/6 pairwise combinations measured" -> "Complete pairwise analysis"

### S2: Phase-Gate Architecture

Phase 1 (d2Cache integration) is explicitly designed as a gate: "If d2Cache works -> use kernel M1; If partially works -> report both; If fails -> report projected." This prevents wasted GPU hours on downstream compositions with a broken M1. The gate was correctly triggered (d2Cache failed) and the fallback executed.

### S3: Risk Mitigation Table

The risk table at the end identifies five specific risks with likelihood, triggers, and mitigations. Four of five materialized (d2Cache failed, Dream-7B accuracy inflated, IGSD accuracy retention mediocre, AR baseline dominates). The plan survived because the mitigations were pre-specified.

---

## Weaknesses

### W1: Full-Scale Execution Was Planned but Not Enforced

The methodology specifies full-scale experiments (1319 GSM8K, 3 seeds) for Phases 2-4 but does not specify a minimum acceptable scale for publication. The experiments defaulted to pilot scale for all pairwise compositions. The three-way composition got 3-seed validation at 100 samples, but the pairwise compositions -- which are the paper's primary evidence -- did not.

**Missing from the plan**: A "minimum viable experiment" specification. For example: "Pairwise Ortho claims require at least 500 samples and 2 seeds. Claims based on <500 samples must include bootstrap confidence intervals and be labeled 'pilot estimate.'"

### W2: M3 Speedup Measurement Protocol Not Specified

The methodology specifies "Metric: wall-clock tokens-per-second (TPS) averaged over stable-state samples" and "total output tokens / elapsed wall-clock time (generation only)." However, this metric conflates per-token speed with output-length effects. If M3 produces shorter outputs (fewer total tokens) in less time, TPS could go up even if per-token latency is worse.

A better protocol would specify:
- Report both TPS (total tokens / time) AND per-sample latency (time per benchmark problem)
- Report average output length for each method
- Define "speedup" as TPS ratio if all methods produce similar output lengths; otherwise define speedup as latency ratio

This omission allows the M3 "1.68x speedup" anomaly to propagate through all downstream Ortho calculations without detection.

### W3: Baseline TPS Not Standardized

The plan does not specify which baseline TPS to use for speedup calculations. Experiments use different baselines:
- d2cache pilot: 58.5 TPS (generation-only, 100 GSM8K, seed 42)
- iter_001 full: 31.0 TPS (presumably end-to-end, 1319 GSM8K)
- Paper text: 33.8 TPS (cited as "generation-only, post-warmup")

The 1.9x discrepancy between 58.5 and 31.0 directly affects all speedup ratios. A method with 100 TPS is 1.71x faster vs. 58.5 TPS but 3.23x faster vs. 31.0 TPS. If different experiments use different baselines, their Ortho values are incomparable.

**The plan should have specified**: "All speedup calculations use the same baseline measurement from Phase 1, sample size >= 100, seed 42, generation-only timing. Record this reference TPS in a shared config file."

### W4: IGSD vs. Naive Truncation Not Prioritized

Phase 7 (IGSD Ablation) includes "IGSD w/o confidence partitioning (tau=0.0) vs with (tau=0.9) at T_draft=32." This is the most critical ablation for IGSD's contribution claim, yet it is scheduled in Phase 7 (second-to-last phase). If IGSD's confidence gate is inert (as confirmed by the results), all IGSD-involving Ortho calculations remain valid, but the paper's Contribution 2 (IGSD as a novel algorithm) is invalidated.

**Better planning**: Run the tau=0.0 ablation in Phase 2 (before any composition experiments). If the confidence gate adds nothing, reframe IGSD as "naive step truncation" throughout all subsequent phases and adjust the paper's contribution claims immediately.

### W5: Dream-7B Validation Is Under-Specified

Phase 5 allocates 3 hours for Dream-7B but specifies only "200 GSM8K + 100 MATH500, seed=42" (single-seed pilot). The methodology correctly identifies Dream-7B's lower accuracy but does not plan for the statistical consequences: with 36% baseline accuracy, 100-sample AccRet calculations have ~13% relative standard error.

The plan should have specified: "If Dream-7B baseline accuracy < 50%, increase GSM8K sample size to 500 for adequate statistical power."

---

## Execution Deviations

| Planned | Executed | Impact |
|---------|----------|--------|
| Phase 2: 3 seeds for full Pareto | M1 3-seed, IGSD 1-seed, M3 2-seed | IGSD and M3 Pareto curves lack replication |
| Phase 3: 1319 GSM8K, 3 seeds for pairwise | 100 GSM8K, 1 seed for all pairwise | Pairwise Ortho unreliable |
| Phase 4: 24-config pilot, top-5 full-scale | 24 pilot, 5 validated at 100 samples x 3 seeds | Acceptable for three-way, but pairwise gap unfilled |
| Phase 5: Retry Dream-7B download | Succeeded | Good |
| Phase 6: Qwen2.5-7B with vLLM | Qwen2.5-7B with HF Transformers | AR baseline is conservative (vLLM would be faster) |
| Phase 7: Full-scale tau=0.0 ablation | 200 GSM8K, seed 42 only | Ablation is pilot-scale for the most critical comparison |
| Phase 8: Batch=1,4,8 with baselines | Batch=1,4,8 WITHOUT batch-specific baselines | Batch sensitivity uninterpretable |

---

## Recommendations for Future Planning

1. **Specify minimum viable experiment sizes**: "Publication claims require N >= 500 and >= 2 seeds. Pilot claims (N < 500) must include confidence intervals."

2. **Standardize baseline measurement first**: Phase 0 should be "Measure baseline TPS under a fixed protocol. Use this single reference for all speedup calculations."

3. **Front-load critical ablations**: If a method claims algorithmic novelty, the null comparison (novel vs. naive) should be Phase 1, not Phase 7.

4. **Separate measurement definitions for per-token speed vs. output efficiency**: TPS conflates both. Define separate metrics or ensure methods produce comparable output lengths.

5. **Add "paper impact gates"**: After each phase, assess: "Does this result change the paper's contribution claims? If yes, adjust subsequent phases." This prevents spending GPU hours on compositions of a method (IGSD) whose core mechanism is already invalidated.

# Full A-CFG Reproduction Summary

**Task**: pilot_acfg_repro
**Date**: 2026-03-10 14:15
**Model**: Dream-v0-Instruct-7B
**Benchmark**: Countdown-100 (full)
**Seed**: 42
**Duration**: 1117s (18min 37s)

## Results

| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time |
|--------|----------|-------|-------|------------|----------|
| Vanilla | 4.0% | 0.1481 | 0.0862 | 0.7449 | 3.7s |
| A-CFG (w=1.0) | 2.0% | 0.1017 | 0.0525 | 0.8189 | 7.4s |

## Decision Gate 1

**Decision**: NO-GO

A-CFG (2.0%) <= vanilla (4.0%) → Decision Gate 1 FAILED: consider switching CFG experiments to LLaDA-8B

## Degeneration Check

No degeneration detected.

## Qualitative Observations

- Vanilla correct: 4/100
- A-CFG correct: 2/100
- A-CFG avg guidance steps: 127
- A-CFG time overhead: 2.0x

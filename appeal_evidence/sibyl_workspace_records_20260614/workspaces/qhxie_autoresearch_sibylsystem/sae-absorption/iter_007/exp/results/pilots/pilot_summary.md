# Data Integrity Validation Report (Pilot)

**Date:** 2026-04-15 17:34

**Verdict:** GO


## Summary

- Total claims validated: 68
- Matches: 63
- Mismatches: 5
- Missing data: 0
- Requires computation: 0
- Integrity score: 92.7%

## Persistent Core Words
- Identified: 5/9
  - **eight** (letter E)
  - **liked** (letter L)
  - **lower** (letter L)
  - **offer** (letter O)
  - **often** (letter O)
- Missing 4 words from letters: {'O': 1, 'U': 2, 'W': 1}
- The confound_decomposition_multi_l0.json hierarchy_details is truncated to 5 entries at each L0 level. The JSON records 9 hierarchy-driven words but only lists the 5 exemplar words (eight/E, lower/L, liked/L, offer/O, often/O). For U: 'under' and 'until' are identified with HIGH confidence (exact count match across L0=82 and L0=176). For O and W: 1 word each remains uncertain among multiple candidates. Identifying them definitively requires re-running the confound decomposition with full word-level output across all 4 L0 values.

## Mismatches
- **probe_f1_absorption_rho**: Paper=-0.67, Source=-0.69 (Paper says -0.67, computed = -0.69)
- **dense_probe_f1**: Paper=0.962, Source=0.929 (Paper says 0.962, source has 0.929)
- **gpt2_l8_rate**: Paper=0.6429, Source=0.6729 (Paper says 0.6429, computed = 0.6729)
- **gpt2_l10_rate**: Paper=0.6729, Source=0.6426 (Paper says 0.6729, computed = 0.6426)
- **gpt2_l11_rate**: Paper=0.6426, Source=0.6165 (Paper says 0.6426, computed = 0.6165)

## Recommendations
- FIX 5 number mismatches between paper.md and source JSONs before proceeding.
- Re-run confound decomposition with full word-level output to identify the 4 unnamed persistent core words (expected from letters: {'O': 1, 'U': 2, 'W': 1}).
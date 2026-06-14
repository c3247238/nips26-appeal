# Phase 2.2: GAS Validation Against Probe-Based Absorption — PILOT Results

## Configuration
- SAE: Gemma Scope L12 16k JumpReLU
- Letters tested: 25 (all with probe data)
- Letters with non-zero absorption: 9
- Overall absorption rate: 0.0390
- GAS source: Phase 2.1 (200 sequences, 25600 tokens)

## Primary Result
- **Spearman rho (GAS vs absorption): 0.1235** (p=0.5564)
- Bootstrap 95% CI: [-0.2566, 0.5187]
- Target: rho >= 0.6, Failure threshold: rho < 0.3
- **Verdict: FAIL**

## AUROC
- GAS AUROC (absorption detection): **0.5903**
- GAS AUROC (strict absorption): 0.47000000000000003
- Cosine baseline AUROC: 0.5417

## Alternative GAS Aggregation Strategies
| Strategy | Spearman rho | p-value |
|----------|-------------|---------|
| Main GAS vulnerability | 0.1235 | 0.5564 |
| weighted_gas_x_cos | 0.1235 | 0.5564 |
| log_gas | 0.1235 | 0.5564 |
| n_high_gas_neighbors | 0.0000 | 1.0000 |
| max_absorber_gas | 0.0000 | 1.0000 |

**Best strategy: main_gas_vulnerability** (rho=0.1235)

## Per-Letter Detail
| Letter | AbsRate | StrictRate | MainFID | Cos | GAS Vuln |
|--------|---------|------------|---------|-----|----------|
| a | 0.0909 | 0.0909 | 6622 | 0.632 | 0.0000 |
| b | 0.1000 | 0.1000 | 13496 | 0.740 | 0.0000 |
| c | 0.0833 | 0.0000 | 15631 | 0.619 | 0.0091 |
| d | 0.0000 | 0.0000 | 6210 | 0.694 | 0.0026 |
| e | 0.0833 | 0.0833 | 9108 | 0.635 | 0.0549 |
| f | 0.1333 | 0.0667 | 1920 | 0.705 | 0.0012 |
| g | 0.0000 | 0.0000 | 4984 | 0.649 | 0.0000 |
| h | 0.0000 | 0.0000 | 10486 | 0.702 | 0.0000 |
| i | 0.0000 | 0.0000 | 1634 | 0.577 | 0.1042 |
| j | 0.0000 | 0.0000 | 6147 | 0.677 | 0.0000 |
| k | 0.0000 | 0.0000 | 394 | 0.687 | 0.0000 |
| l | 0.0000 | 0.0000 | 205 | 0.711 | 0.0038 |
| m | 0.0000 | 0.0000 | 10931 | 0.660 | 0.0000 |
| n | 0.0000 | 0.0000 | 8740 | 0.591 | 0.0000 |
| o | 0.0909 | 0.0000 | 12334 | 0.711 | 0.0380 |
| p | 0.0000 | 0.0000 | 12186 | 0.591 | 0.0000 |
| q | 0.0000 | 0.0000 | 8414 | 0.690 | 0.0135 |
| r | 0.0714 | 0.0000 | 12279 | 0.667 | 0.0000 |
| s | 0.1818 | 0.1818 | 6508 | 0.665 | 0.0000 |
| t | 0.0000 | 0.0000 | 6355 | 0.601 | 0.0000 |
| u | 0.0769 | 0.0000 | 14110 | 0.517 | 2.1122 |
| v | 0.0000 | 0.0000 | 3039 | 0.688 | 0.0037 |
| w | 0.0000 | 0.0000 | 5879 | 0.690 | 0.0000 |
| y | 0.0000 | 0.0000 | 3811 | 0.640 | 0.3417 |
| z | 0.0000 | 0.0000 | 13942 | 0.387 | 0.0000 |

## Diagnostics
- Main letter features with GAS > 1.0: 1/25. GAS vulnerability of main features may be low because the main feature is the one that DOES the absorbing (high freq), not the one being absorbed.
- GAS direction check: per_feature_gas[fid] = max_i GAS(i -> fid). This measures vulnerability of the letter feature to absorption by any other feature. Higher GAS vulnerability should correlate with higher absorption rate. If it does NOT correlate, it may indicate that: (a) The main feature's GAS vulnerability is determined by geometry, not functional absorption; (b) Absorption happens through features NOT captured by the top-1 cosine match; or (c) The pilot sample (200 seqs, 25600 tokens) is too small for reliable co-activation statistics.

## Discovery: Top 10 Non-First-Letter Candidates
| Rank | Feature | GAS Vulnerability |
|------|---------|-------------------|
| 1 | 7658 | 1117.5271 |
| 2 | 1847 | 840.9229 |
| 3 | 3003 | 836.7411 |
| 4 | 3474 | 710.8768 |
| 5 | 9431 | 572.7844 |
| 6 | 13395 | 554.4982 |
| 7 | 10402 | 522.8578 |
| 8 | 10951 | 518.2883 |
| 9 | 10281 | 479.4187 |
| 10 | 5309 | 470.6077 |

## Timing: 1.4 seconds (0.0 minutes)

## Pilot Assessment: **ALL CRITERIA PASS**
- [x] Spearman rho computed
- [x] Bootstrap CI computed
- [x] AUROC computed
- [x] Negative/positive result documented
# Phase 4.1: Architecture Comparison Results (v2)

Mode: PILOT | Layer: 12 | Elapsed: 2.0 min | Architectures: 4

## Probe Quality at Layer 12

| Hierarchy | F1 | Quality Gate |
|-----------|-----|-------------|
| first-letter | 0.29360544217687073 | below |
| city-continent | 0.7872672575874279 | below |
| city-language | 0.6883605848644696 | excluded |
| city-country | 0.6161091303716675 | excluded |

## Summary Table

| Hierarchy | Architecture | d_sae | Absorption Rate | Strict Rate | 95% CI | L0 | FN / Total |
|-----------|-------------|-------|-----------------|-------------|--------|-----|-----------|
| first-letter | JumpReLU_16k | 16384 | 0.0067 | 0.0067 | [0.000, 0.020] | 87 | 1 / 149 |
| first-letter | JumpReLU_65k | 65536 | 0.0134 | 0.0134 | [0.000, 0.034] | 71 | 2 / 149 |
| first-letter | BatchTopK_16k | 16384 | 0.0338 | 0.0203 | [0.007, 0.061] | 20 | 5 / 148 |
| first-letter | Matryoshka | 32768 | 0.0137 | 0.0068 | [0.000, 0.034] | 93 | 2 / 146 |
| city-continent | JumpReLU_16k | 16384 | 0.1731 | 0.0577 | [0.077, 0.269] | 76 | 9 / 52 |
| city-continent | JumpReLU_65k | 65536 | 0.2308 | 0.0962 | [0.115, 0.346] | 71 | 12 / 52 |
| city-continent | BatchTopK_16k | 16384 | 0.1346 | 0.1154 | [0.058, 0.231] | 20 | 7 / 52 |
| city-continent | Matryoshka | 32768 | 0.1923 | 0.0385 | [0.096, 0.289] | 98 | 10 / 52 |
| city-language | JumpReLU_16k | 16384 | 0.4118 | 0.0882 | [0.265, 0.588] | 81 | 14 / 34 |
| city-language | JumpReLU_65k | 65536 | 0.3824 | 0.2647 | [0.235, 0.559] | 75 | 13 / 34 |
| city-language | BatchTopK_16k | 16384 | 0.6176 | 0.2941 | [0.441, 0.794] | 20 | 21 / 34 |
| city-language | Matryoshka | 32768 | 0.3529 | 0.0588 | [0.205, 0.529] | 108 | 12 / 34 |
| city-country | JumpReLU_16k | 16384 | 0.4706 | 0.1765 | [0.235, 0.706] | 77 | 8 / 17 |
| city-country | JumpReLU_65k | 65536 | 0.4706 | 0.2941 | [0.235, 0.706] | 72 | 8 / 17 |
| city-country | BatchTopK_16k | 16384 | 0.5294 | 0.2353 | [0.294, 0.765] | 20 | 9 / 17 |
| city-country | Matryoshka | 32768 | 0.3529 | 0.1765 | [0.176, 0.588] | 100 | 6 / 17 |

## Pairwise Comparisons

### first-letter

| Comparison | Rate A | Rate B | Diff | z-stat | p-value | Sig. |
|-----------|--------|--------|------|--------|---------|------|
| JumpReLU_16k vs JumpReLU_65k | 0.0067 | 0.0134 | -0.0067 | -0.580 | 0.5617 | no |
| JumpReLU_16k vs BatchTopK_16k | 0.0067 | 0.0338 | -0.0271 | -1.658 | 0.0973 | no |
| JumpReLU_16k vs Matryoshka | 0.0067 | 0.0137 | -0.0070 | -0.598 | 0.5498 | no |
| JumpReLU_65k vs BatchTopK_16k | 0.0134 | 0.0338 | -0.0204 | -1.157 | 0.2475 | no |
| JumpReLU_65k vs Matryoshka | 0.0134 | 0.0137 | -0.0003 | -0.020 | 0.9837 | no |
| BatchTopK_16k vs Matryoshka | 0.0338 | 0.0137 | 0.0201 | 1.129 | 0.2587 | no |

### city-continent

| Comparison | Rate A | Rate B | Diff | z-stat | p-value | Sig. |
|-----------|--------|--------|------|--------|---------|------|
| JumpReLU_16k vs JumpReLU_65k | 0.1731 | 0.2308 | -0.0577 | -0.733 | 0.4637 | no |
| JumpReLU_16k vs BatchTopK_16k | 0.1731 | 0.1346 | 0.0385 | 0.544 | 0.5867 | no |
| JumpReLU_16k vs Matryoshka | 0.1731 | 0.1923 | -0.0192 | -0.254 | 0.7997 | no |
| JumpReLU_65k vs BatchTopK_16k | 0.2308 | 0.1346 | 0.0962 | 1.269 | 0.2045 | no |
| JumpReLU_65k vs Matryoshka | 0.2308 | 0.1923 | 0.0385 | 0.480 | 0.6311 | no |
| BatchTopK_16k vs Matryoshka | 0.1346 | 0.1923 | -0.0577 | -0.796 | 0.4263 | no |

### city-language

| Comparison | Rate A | Rate B | Diff | z-stat | p-value | Sig. |
|-----------|--------|--------|------|--------|---------|------|
| JumpReLU_16k vs JumpReLU_65k | 0.4118 | 0.3824 | 0.0294 | 0.248 | 0.8043 | no |
| JumpReLU_16k vs BatchTopK_16k | 0.4118 | 0.6176 | -0.2059 | -1.698 | 0.0894 | no |
| JumpReLU_16k vs Matryoshka | 0.4118 | 0.3529 | 0.0588 | 0.499 | 0.6177 | no |
| JumpReLU_65k vs BatchTopK_16k | 0.3824 | 0.6176 | -0.2353 | -1.940 | 0.0523 | no |
| JumpReLU_65k vs Matryoshka | 0.3824 | 0.3529 | 0.0294 | 0.252 | 0.8014 | no |
| BatchTopK_16k vs Matryoshka | 0.6176 | 0.3529 | 0.2647 | 2.184 | 0.0290 | YES |

### city-country

| Comparison | Rate A | Rate B | Diff | z-stat | p-value | Sig. |
|-----------|--------|--------|------|--------|---------|------|
| JumpReLU_16k vs JumpReLU_65k | 0.4706 | 0.4706 | 0.0000 | 0.000 | 1.0000 | no |
| JumpReLU_16k vs BatchTopK_16k | 0.4706 | 0.5294 | -0.0588 | -0.343 | 0.7316 | no |
| JumpReLU_16k vs Matryoshka | 0.4706 | 0.3529 | 0.1176 | 0.697 | 0.4858 | no |
| JumpReLU_65k vs BatchTopK_16k | 0.4706 | 0.5294 | -0.0588 | -0.343 | 0.7316 | no |
| JumpReLU_65k vs Matryoshka | 0.4706 | 0.3529 | 0.1176 | 0.697 | 0.4858 | no |
| BatchTopK_16k vs Matryoshka | 0.5294 | 0.3529 | 0.1765 | 1.036 | 0.3001 | no |

## ANOVA: absorption ~ architecture * hierarchy

Architecture effect: {'H_stat': 0.7024336283185866, 'p_value': 0.8726313863323003, 'significant_005': False, 'test': 'Kruskal-Wallis'}
Hierarchy effect: {'H_stat': 12.793141592920357, 'p_value': 0.005105989112822576, 'significant_005': True, 'test': 'Kruskal-Wallis'}

## JumpReLU Advantage Analysis

- **first-letter**: JumpReLU 0.0067 vs best other Matryoshka 0.0137 → LOWEST
- **city-continent**: JumpReLU 0.1731 vs best other BatchTopK_16k 0.1346 → NOT lowest
- **city-language**: JumpReLU 0.3824 vs best other Matryoshka 0.3529 → NOT lowest
- **city-country**: JumpReLU 0.4706 vs best other Matryoshka 0.3529 → NOT lowest

## H6 Verdict: JumpReLU advantage -- **PARTIALLY_SUPPORTED**
- JumpReLU lowest in 1/4 hierarchies

## Caveats

- All comparisons at layer 12 (only layer with all architectures)
- Cross-domain probes below strict quality gate (F1 < 0.90)
- Matryoshka has variable width (32k) vs 16k for JumpReLU/BatchTopK -- not width-matched
- JumpReLU_65k included for within-architecture width comparison
- Absorption rates for cross-domain hierarchies may be confounded with probe errors

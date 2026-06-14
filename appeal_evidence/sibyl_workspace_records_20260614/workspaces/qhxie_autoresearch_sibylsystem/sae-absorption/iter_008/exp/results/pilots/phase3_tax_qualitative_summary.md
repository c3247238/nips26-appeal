# Phase 3: Absorption Tax Qualitative Validation (PILOT)

## Key Result: T(G) Ranking vs Observed Absorption

**Qualitative Verdict: PARTIAL_SUPPORT**

### Ranking Table

| Hierarchy | T(G) | T(G) Rank | Observed Abs (L24_16k) | Obs Rank |
|-----------|------|-----------|----------------------|----------|
| first-letter | 0.416701 | 1 | 0.3448 | 2 |
| city-country | 0.086697 | 2 | 0.1850 | 3 |
| city-language | 0.053215 | 3 | 0.1364 | 4 |
| city-continent | 0.013345 | 4 | 0.3584 | 1 |

### Ranking Correlation

- Spearman rho: -0.19999999999999998
- Kendall tau: 0.0
- Bootstrap 95% CI (Spearman): [-1.0, 1.0]

### Qualitative Details
- Top hierarchy: T(G)=first-letter, Obs=city-continent (match=False)
- Bottom hierarchy: T(G)=city-continent, Obs=city-language (match=False)
- Pairwise concordance: 3/6 = 50.00%
- First-letter: T(G) rank=1/4, Obs rank=2/4

### T(G) Per Hierarchy (at best probe layer)

| Hierarchy | Layer | T(G) | n_classes |
|-----------|-------|------|-----------|
| first-letter | 18 | 0.445821 | 26 |
| first-letter | 24 | 0.416701 | 26 |
| first-letter | 6 | 0.198281 | 26 |
| city-country | 24 | 0.086697 | 80 |
| first-letter | 12 | 0.058486 | 26 |
| city-language | 24 | 0.053215 | 50 |
| city-continent | 24 | 0.013345 | 6 |

### Per-Letter R_pc vs Absorption (Improved Probes)

| SAE Config | Probe Layer | n_letters | Spearman rho | p-value | Verdict |
|-----------|-------------|-----------|-------------|---------|---------|
| L12_16k | 12 | 24 | 0.0862 | 0.6888 | NOT_SUPPORTED |
| L12_65k | 12 | 24 | 0.1399 | 0.5143 | NOT_SUPPORTED |
| L18_16k | 18 | 24 | -0.3898 | 0.0597 | WEAK_SUPPORT |
| L18_65k | 18 | 24 | 0.0098 | 0.9638 | NOT_SUPPORTED |
| L24_16k | 24 | 24 | 0.1680 | 0.4326 | NOT_SUPPORTED |
| L24_65k | 24 | 24 | -0.0602 | 0.7798 | NOT_SUPPORTED |
| L6_16k | 6 | 24 | 0.1022 | 0.6347 | NOT_SUPPORTED |
| L6_65k | 6 | 24 | 0.0106 | 0.9608 | NOT_SUPPORTED |

## Pilot Assessment

- **Hierarchies with T(G):** 7
- **Letters with R_pc:** 26
- **Pilot pass:** YES
- **Elapsed time:** 0.2 minutes

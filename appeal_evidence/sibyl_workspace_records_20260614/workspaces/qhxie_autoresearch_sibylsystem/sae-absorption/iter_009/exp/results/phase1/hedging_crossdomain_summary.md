# Phase 1.3: Cross-Domain Absorption-Hedging Decomposition (FULL)

**Timestamp**: 2026-04-16T05:50:16.949489
**Mode**: FULL
**Elapsed**: 0.9s

## Decomposition Categories

| Category | Definition |
|----------|-----------|
| **Strict absorbed** | FN where main parent feature does NOT fire (genuine feature gap) |
| **Compensatory** | FN where main parent feature fires but probe is wrong (interference) |
| **Persistent** | Remainder (probe error, ambiguous boundary) |

## Cross-Hierarchy Comparison at L24_16k

| Hierarchy | FN | Strict | Compensatory | Persistent | Abs Rate |
|-----------|-----|--------|-------------|------------|----------|
| first-letter | 291 | 0 (0.0%) | 291 (100.0%) | 0 (0.0%) | 0.3003 |
| city-continent | 418 | 26 (6.2%) | 392 (93.8%) | 0 (0.0%) | 0.3143 |
| city-language | 124 | 28 (22.6%) | 96 (77.4%) | 0 (0.0%) | 0.1156 |
| city-country | 515 | 19 (3.7%) | 496 (96.3%) | 0 (0.0%) | 0.4510 |

## Cross-Hierarchy Comparison at L24_65k

| Hierarchy | FN | Strict | Compensatory | Persistent | Abs Rate |
|-----------|-----|--------|-------------|------------|----------|
| first-letter | 202 | 0 (0.0%) | 202 (100.0%) | 0 (0.0%) | 0.2124 |
| city-continent | 416 | 6 (1.4%) | 410 (98.6%) | 0 (0.0%) | 0.3128 |
| city-language | 83 | 19 (22.9%) | 64 (77.1%) | 0 (0.0%) | 0.0774 |
| city-country | 376 | 17 (4.5%) | 359 (95.5%) | 0 (0.0%) | 0.3292 |

## Width Comparison (16k vs 65k)

| Hierarchy | 16k Strict% | 65k Strict% | Delta Strict | 16k Comp% | 65k Comp% |
|-----------|-------------|-------------|-------------|-----------|-----------|
| first-letter | 0.0 | 0.0 | +0.0pp | 100.0 | 100.0 |
| city-continent | 6.2 | 1.4 | -4.8pp | 93.8 | 98.6 |
| city-language | 22.6 | 22.9 | +0.3pp | 77.4 | 77.1 |
| city-country | 3.7 | 4.5 | +0.8pp | 96.3 | 95.5 |

## Layer Profile (First-Letter)

| Config | Layer | FN | Strict% | Comp% | Persistent% | AbsRate |
|--------|-------|----|---------|-------|-------------|---------|
| L6_16k | 6 | 30 | 46.7 | 53.3 | 0.0 | 0.3704 |
| L6_65k | 6 | 24 | 54.2 | 45.8 | 0.0 | 0.2892 |
| L12_16k | 12 | 9 | 100.0 | 0.0 | 0.0 | 0.1286 |
| L12_65k | 12 | 27 | 100.0 | 0.0 | 0.0 | 0.4091 |
| L18_16k | 18 | 477 | 40.5 | 59.5 | 0.0 | 0.3712 |
| L18_65k | 18 | 316 | 32.9 | 67.1 | 0.0 | 0.2490 |
| L24_16k | 24 | 291 | 0.0 | 100.0 | 0.0 | 0.3003 |
| L24_65k | 24 | 202 | 0.0 | 100.0 | 0.0 | 0.2124 |

## Statistical Tests

### Chi-square (4 hierarchies x strict/compensatory at L24_16k)
- Chi2(3) = 91.51, p = 0.000000
- Chi-square(3)=91.51, p=0.0000. Significant difference in strict/compensatory ratio across 4 hierarchies.

### Pairwise Fisher Exact Tests (Bonferroni-corrected)

| Pair | OR | p-value | Significant |
|------|-----|---------|-------------|
| first-letter vs city-continent | 0.000 | 0.0000 | Yes* |
| first-letter vs city-language | 0.000 | 0.0000 | Yes* |
| first-letter vs city-country | 0.000 | 0.0003 | Yes* |
| city-continent vs city-language | 0.227 | 0.0000 | Yes* |
| city-continent vs city-country | 1.731 | 0.0904 | No |
| city-language vs city-country | 7.614 | 0.0000 | Yes* |

*Bonferroni-corrected significance

## Key Findings

- first-letter: compensatory dominates at 100.0% (N_FN=291)
- city-continent: compensatory dominates at 93.8% (N_FN=418)
- city-language: compensatory dominates at 77.4% (N_FN=124)
- city-country: compensatory dominates at 96.3% (N_FN=515)
- Strict absorbed ranges from 0.0% (first-letter) to 22.6% (city-language). Chi-square p=0.0000.
- Compensatory FNs (main feature fires but probe wrong) dominate in 4/4 hierarchies at L24_16k. This means the SAE typically has features relevant to the concept but reconstruction distorts the representation.
- Strict absorbed fraction changes with layer: L6=50.4%, L12=100.0%, L18=36.7%, L24=0.0%. Compensatory: L6=49.6%, L12=0.0%, L18=63.3%, L24=100.0%
- Width change (16k->65k) effects: first-letter: delta_strict=+0.0pp; city-continent: delta_strict=-4.8pp; city-language: delta_strict=+0.3pp; city-country: delta_strict=+0.8pp

## Comparison with iter_008

- Source: iter_008 Phase 0.2 tightened hedging (L0=22->176, L12_16k, first-letter)
- iter_008: strict=7.4%, compensatory=85.3%, persistent=7.4%
- iter_009 L12_16k: strict=100.0%, compensatory=0.0%, persistent=0.0%
- Note: iter_008 used multi-L0 analysis (L0=22->176) at L12_16k. iter_009 uses single-L0 main-feature-firing proxy at each config's default L0 across multiple layers/widths. The 'strict_absorbed' (main absent) in iter_009 roughly maps to 'strict_hedging' in iter_008, while 'compensatory' (main present but probe wrong) maps to compensatory_resolution.

## Methodology Notes
- **classification**: Three-way decomposition of false negatives: (1) strict_absorbed = FN where main parent feature does NOT fire (genuine feature gap, no SAE feature encodes this concept); (2) compensatory = FN where main parent feature DOES fire but probe is wrong on SAE reconstruction (feature fires but other features compensate/interfere, distorting information in probe direction); (3) persistent = remainder (probe error, ambiguous class boundary).
- **main_feature_identification**: Parent feature = SAE latent with highest cosine similarity to the probe direction for that class (top-5 considered, top-1 used for main_fires check).
- **limitation_single_l0**: Uses single L0 at each SAE config's default sparsity level. iter_008 Phase 0.2 used multi-L0 (22->176) for more precise classification. The 'strict_absorbed' here is a stricter criterion than iter_008's 'strict_hedging' which required FN resolution at higher L0.

### probe_quality_caveats
- **first-letter**: Probes validated at each layer; quality varies.
- **city-continent**: F1=0.8709952678012601
- **city-language**: F1=0.8178942823784625
- **city-country**: F1=0.7260000505669184 (below strict gate)

### data_sizes
- **first-letter**: 500 words x 3 prompts
- **city-continent**: 1567 entities
- **city-language**: 1229 entities
- **city-country**: 1405 entities
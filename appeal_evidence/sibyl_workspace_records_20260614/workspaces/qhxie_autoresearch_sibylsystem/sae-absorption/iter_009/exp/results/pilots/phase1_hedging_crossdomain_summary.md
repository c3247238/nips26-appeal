# Phase 1: Cross-Domain Hedging Decomposition (Pilot)

**Timestamp**: 2026-04-16T02:57:07.809444
**Mode**: PILOT
**Elapsed**: 0.1s

## Decomposition Categories

| Category | Definition |
|----------|-----------|
| **Strict absorbed** | FN where main parent feature does NOT fire (genuine feature gap) |
| **Compensatory** | FN where main parent feature fires but probe is wrong (interference) |
| **Persistent** | Remainder (probe error, ambiguous boundary) |

## Cross-Hierarchy Comparison at L24_16k

| Hierarchy | FN | Strict | Compensatory | Persistent | Abs Rate |
|-----------|-----|--------|-------------|------------|----------|
| first-letter | 9 | 3 (33.3%) | 6 (66.7%) | 0 (0.0%) | 0.4286 |
| city-continent | 22 | 2 (9.1%) | 20 (90.9%) | 0 (0.0%) | 0.2857 |

## Layer Profile (First-Letter)

| Config | FN | Strict% | Compensatory% | Persistent% | AbsRate |
|--------|-----|---------|--------------|-------------|---------|
| L6_16k | 2 | 0.0 | 100.0 | 0.0 | 0.0556 |
| L6_65k | 2 | 0.0 | 100.0 | 0.0 | 0.0526 |
| L12_16k | 2 | 0.0 | 100.0 | 0.0 | 0.0690 |
| L12_65k | 5 | 60.0 | 40.0 | 0.0 | 0.1786 |
| L18_16k | 2 | 50.0 | 50.0 | 0.0 | 0.0417 |
| L18_65k | 2 | 0.0 | 100.0 | 0.0 | 0.0400 |
| L24_16k | 9 | 33.3 | 66.7 | 0.0 | 0.4286 |
| L24_65k | 6 | 66.7 | 33.3 | 0.0 | 0.2727 |

## Key Findings

- first-letter: most FNs are compensatory (66.7%, N=9)
- city-continent: most FNs are compensatory (90.9%, N=22)
- At L24_16k: city-continent has +24.2pp compensatory, -24.2pp strict vs first-letter
- Compensatory fraction: L24 avg=50.0%, early layers (L6/L12) avg=85.0%

## Fisher's Exact Test (L24_16k)
- OR = 5.000, p = 0.1313
- OR=5.00, p=0.1313. Not significant difference in strict/compensatory ratio between hierarchies.

## Comparison with iter_008
- Source: iter_008 Phase 0.2 tightened hedging (L0=22->176, L12_16k)
- iter_008 (multi-L0): strict=7.4%, compensatory=85.3%, persistent=7.4%
- Note: iter_008 used multi-L0 analysis (L0=22->176) for classification. iter_009 uses single-L0 main-feature-firing proxy at each config's default L0. The 'strict_absorbed' (main absent) in iter_009 roughly corresponds to 'strict_hedging' (FN resolves at higher L0 + feature fires) in iter_008, while 'compensatory' (main present but probe wrong) maps to the compensatory category.

## Methodology Notes
- **classification**: Three-way decomposition of false negatives: (1) strict_absorbed = FN where main parent feature does NOT fire (genuine feature gap, no SAE feature encodes this concept); (2) compensatory = FN where main parent feature DOES fire but probe is wrong on SAE reconstruction (feature fires but other features compensate/interfere, distorting information in probe direction); (3) persistent = remainder (probe error, ambiguous class boundary).
- **main_feature_identification**: Parent feature = SAE latent with highest cosine similarity to the probe direction for that class (top-5 considered, top-1 used for main_fires check).
- **limitation_single_l0**: Uses single L0 at each SAE config's default sparsity level. iter_008 Phase 0.2 used multi-L0 (22->176) for more precise classification. The 'strict_absorbed' here is a stricter criterion than iter_008's 'strict_hedging' which required FN resolution at higher L0.
- **probe_quality_caveat**: Cross-domain probe F1=0.8710 (below strict 0.90 gate). Some FNs may reflect probe error, not SAE-induced information loss.
# Phase 1.1: Cross-Domain Patching Spot-Check

**Verification Verdict**: VERIFIED
**Time**: 2.1 minutes

## Purpose

Verify the sign-reversed city-continent activation patching data from iter_009.
Expected mean primary recovery rate: 61.9%
Expected Cohen's d: 1.5

## Sampling

- **Method**: Stratified by continent (seed=42)
- **N entities**: 20
- **N completed**: 10
- **N contexts per entity**: 50

### Entities sampled per continent

- Allende (North America)
- Maracay (South America)
- Lobos (South America)
- Uberaba (South America)
- Yambio (Africa)
- Roma (Oceania)
- Pernik (Europe)
- Ganzhou (Asia)
- Arad (Asia)
- Tofino (North America)
- Alpena (North America)
- Sarande (Europe)
- Poso (Asia)
- Magelang (Asia)
- Bytom (Europe)
- Denver (North America)
- Braila (Europe)
- Yomou (Africa)
- Agordat (Africa)
- Ranong (Asia)

## Aggregate Results

- **Mean primary recovery**: 0.6265
- **Overall primary recovery**: 0.4256
- **Mean all-absorbers recovery**: 0.9684
- **Mean control recovery**: 0.0154
- **Total absorbed contexts**: 242

## Statistical Tests

- **Cohen's d**: 2.0408 (large)
- **Wilcoxon**: p=0.000977 (sig at 0.01: True)
- **Paired t-test**: p=0.000087 (sig at 0.01: True)

## Verification Against iter_009

- **Expected recovery rate**: 61.9%
- **Observed recovery rate**: 62.7%
- **Difference**: +0.8 pp
- **Verdict**: VERIFIED

## Pass Criteria

- recovery_within_10pp: **PASS**
- cohens_d_gt_0.5: **PASS**
- p_lt_0.01_wilcoxon: **PASS**
- p_lt_0.01_ttest: **PASS**

**Overall**: ALL PASS

## Per-Class Breakdown

| Continent | Entities | Absorbed | Primary Rate | Control Rate |
|-----------|----------|----------|--------------|-------------|
| Africa | 1 | 4 | 1.0000 | 0.0250 |
| Asia | 3 | 43 | 0.8485 | 0.0348 |
| Europe | 4 | 188 | 0.3050 | 0.0012 |
| North America | 1 | 5 | 1.0000 | 0.0200 |
| South America | 1 | 2 | 0.5000 | 0.0000 |

## Per-Entity Results

| City | Continent | Absorbed | Primary Rec | Control | Diff |
|------|-----------|----------|-------------|---------|------|
| Allende | North America | 5 | 1.0000 | 0.0200 | +0.9800 |
| Maracay | South America | 0 | insufficient_absorption | - | - |
| Lobos | South America | 0 | insufficient_absorption | - | - |
| Uberaba | South America | 2 | 0.5000 | 0.0000 | +0.5000 |
| Yambio | Africa | 0 | insufficient_absorption | - | - |
| Roma | Oceania | 0 | insufficient_absorption | - | - |
| Pernik | Europe | 42 | 0.5238 | 0.0048 | +0.5190 |
| Ganzhou | Asia | 1 | insufficient_absorption | - | - |
| Arad | Asia | 6 | 0.6667 | 0.0000 | +0.6667 |
| Tofino | North America | 0 | insufficient_absorption | - | - |
| Alpena | North America | 1 | insufficient_absorption | - | - |
| Sarande | Europe | 49 | 0.1633 | 0.0000 | +0.1633 |
| Poso | Asia | 33 | 0.8788 | 0.0545 | +0.8242 |
| Magelang | Asia | 4 | 1.0000 | 0.0500 | +0.9500 |
| Bytom | Europe | 50 | 0.3200 | 0.0000 | +0.3200 |
| Denver | North America | 0 | insufficient_absorption | - | - |
| Braila | Europe | 47 | 0.2128 | 0.0000 | +0.2128 |
| Yomou | Africa | 4 | 1.0000 | 0.0250 | +0.9750 |
| Agordat | Africa | 0 | insufficient_absorption | - | - |
| Ranong | Asia | 0 | insufficient_absorption | - | - |
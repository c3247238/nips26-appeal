# Phase 2.1: Cross-Domain Activation Patching -- Iter 9 Pilot

**Verdict**: INSUFFICIENT_EVIDENCE
**Pilot**: FAIL
**Time**: 2.1 minutes

## Design

- **Hierarchy**: city-continent at L24
- **SAE**: L24_16k
- **Token position**: -2
- **Contexts per entity**: 50

## Aggregate Results

- **FN instances**: 3751
- **Child-zeroed recovery**: 0.0005 (CI [0.000, 0.001])
- **Control recovery**: 0.1445 (CI [0.016, 0.020])
- **Residual recovery**: 0.0008
- **Recovery difference**: -0.1440

## Statistical Tests

- **Wilcoxon signed-rank**: p=1.000000 (n=93 pairs, significant p<0.05: False)
- **Cohen's d**: -0.9062 (large)
- **Permutation test**: diff=-0.1700, p=1.0000

## Per-Class Patching

| Class | Entities | FN | Child Recovery | Control | Residual |
|-------|----------|-----|---------------|---------|----------|
| Asia | 4 | 66 | 0.0000 | 0.5909 | 0.0000 |
| Europe | 83 | 3567 | 0.0000 | 0.1242 | 0.0000 |
| North America | 4 | 74 | 0.0270 | 0.5541 | 0.0405 |
| Oceania | 1 | 32 | 0.0000 | 0.2500 | 0.0000 |
| South America | 1 | 12 | 0.0000 | 0.9167 | 0.0000 |

## Per-Entity Results

| City | Label | FN | Child Rec | Control Rec | Residual Rec |
|------|-------|-----|-----------|-------------|--------------|
| Aarau | Europe | 49 | 0.0000 | 0.1224 | 0.0000 |
| Abha | Asia | 38 | 0.0000 | 0.5526 | 0.0000 |
| Alesund | Europe | 50 | 0.0000 | 0.2400 | 0.0000 |
| Almenara | Europe | 5 | 0.0000 | 0.0000 | 0.0000 |
| Angers | Europe | 50 | 0.0000 | 0.0600 | 0.0000 |
| Appenzell | Europe | 47 | 0.0000 | 0.0426 | 0.0000 |
| Arad | Europe | 37 | 0.0000 | 0.0000 | 0.0000 |
| Arezzo | Europe | 48 | 0.0000 | 0.1250 | 0.0000 |
| Arras | Europe | 45 | 0.0000 | 0.0444 | 0.0000 |
| Ascension | South America | 12 | 0.0000 | 0.9167 | 0.0000 |
| Asti | Europe | 46 | 0.0000 | 0.1957 | 0.0000 |
| Augsburg | Europe | 50 | 0.0000 | 0.0400 | 0.0000 |
| Aurora | North America | 24 | 0.0417 | 0.5000 | 0.0417 |
| Bakal | Europe | 10 | 0.0000 | 0.0000 | 0.0000 |
| Balakhna | Europe | 22 | 0.0000 | 0.0000 | 0.0000 |
| Barletta | Europe | 49 | 0.0000 | 0.1633 | 0.0000 |
| Bath | Europe | 47 | 0.0000 | 0.1064 | 0.0000 |
| Bergamo | Europe | 50 | 0.0000 | 0.1200 | 0.0000 |
| Biel | Europe | 47 | 0.0000 | 0.0426 | 0.0000 |
| Bielefeld | Europe | 50 | 0.0000 | 0.1400 | 0.0000 |
| Bilibino | Europe | 48 | 0.0000 | 0.1042 | 0.0000 |
| Blackpool | Europe | 50 | 0.0000 | 0.2600 | 0.0000 |
| Bologna | Europe | 48 | 0.0000 | 0.1667 | 0.0000 |
| Braila | Europe | 50 | 0.0000 | 0.0200 | 0.0000 |
| Bristol | Europe | 37 | 0.0000 | 0.1622 | 0.0000 |
| Brno | Europe | 49 | 0.0000 | 0.1429 | 0.0000 |
| Bucharest | Europe | 49 | 0.0000 | 0.0408 | 0.0000 |
| Caen | Europe | 44 | 0.0000 | 0.1136 | 0.0000 |
| Caserta | Europe | 50 | 0.0000 | 0.1200 | 0.0000 |
| Catania | Europe | 50 | 0.0000 | 0.1400 | 0.0000 |

## Interpretation

**INSUFFICIENT_EVIDENCE**: Zeroing child features recovers correct continent prediction in 0.1% of absorbed contexts (vs 14.4% control). Statistical significance not reached with current sample size.
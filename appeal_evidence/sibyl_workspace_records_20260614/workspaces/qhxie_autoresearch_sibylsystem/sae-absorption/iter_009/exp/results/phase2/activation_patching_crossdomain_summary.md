# Phase 2.1 FULL: Cross-Domain Activation Patching

**Verdict**: CROSS_DOMAIN_CAUSAL_ABSORPTION_CONFIRMED
**Time**: 34.1 minutes

## Methodology Fix

**Pilot bug**: Pilot zeroed features with highest POSITIVE cosine to probe (features supporting correct class). Recovery was 0.05%.
**Fix**: Now uses iter_008 contribution-based approach: contribution = mean_activation * cosine_with_probe. Features with most NEGATIVE contribution are identified as absorbers (they push prediction away from true class). Zeroing absorbers should RECOVER correct prediction.

## city-continent

- **Entities**: 128
- **Total absorbed contexts**: 4902
- **Primary absorber recovery**: 0.6193 (CI [0.550, 0.689])
- **All absorbers recovery**: 0.7997
- **Control recovery**: 0.0516
- **Diff (primary-control)**: +0.5677

- **Wilcoxon**: p=0.000000 (sig: True)
- **Cohen's d**: 1.5018 (large)

### Per-Class Results

| Class | Entities | Absorbed | Primary Rec | All Rec | Control |
|-------|----------|----------|-------------|---------|---------|
| Africa | 10 | 108 | 0.8426 | 0.9722 | 0.1352 |
| Asia | 28 | 727 | 0.9381 | 1.0000 | 0.0253 |
| Europe | 30 | 2124 | 0.5325 | 0.9435 | 0.0073 |
| North America | 25 | 756 | 0.7817 | 0.9947 | 0.1066 |
| Oceania | 29 | 1121 | 0.0419 | 0.2417 | 0.0301 |
| South America | 6 | 66 | 0.8485 | 0.9848 | 0.1311 |

## city-language

- **Entities**: 201
- **Total absorbed contexts**: 7814
- **Primary absorber recovery**: 0.3418 (CI [0.289, 0.393])
- **All absorbers recovery**: 0.6998
- **Control recovery**: 0.0682
- **Diff (primary-control)**: +0.2736

- **Wilcoxon**: p=0.000000 (sig: True)
- **Cohen's d**: 0.7529 (medium)

### Per-Class Results

| Class | Entities | Absorbed | Primary Rec | All Rec | Control |
|-------|----------|----------|-------------|---------|---------|
| Afrikaans | 6 | 192 | 0.0208 | 0.5833 | 0.1760 |
| Aimar√°,Ket¬öua,Spanish | 17 | 1098 | 0.1530 | 0.6648 | 0.0448 |
| Arabic | 26 | 826 | 0.2203 | 0.7058 | 0.0818 |
| Arabic,Berberi | 10 | 466 | 0.2811 | 0.4721 | 0.0361 |
| Arabic,Kurdish,Turkish | 4 | 90 | 0.0556 | 0.1889 | 0.0583 |
| Aymara | 13 | 838 | 0.0215 | 0.0060 | 0.0128 |
| Chinese | 1 | 65 | 0.1538 | 0.4308 | 0.0318 |
| Chinese,Korean | 10 | 238 | 0.0672 | 0.9118 | 0.1286 |
| English | 3 | 150 | 0.3600 | 0.9467 | 0.0192 |
| French | 4 | 45 | 0.8222 | 1.0000 | 0.1704 |
| German | 10 | 374 | 0.7219 | 0.8396 | 0.0737 |
| Hindi | 4 | 193 | 0.1917 | 0.2383 | 0.0159 |
| Indonesian | 6 | 81 | 0.6914 | 0.8642 | 0.1534 |
| Italian | 9 | 213 | 0.3052 | 0.7559 | 0.0614 |
| Japanese | 5 | 125 | 0.0960 | 0.1920 | 0.0858 |
| Kazakh | 9 | 515 | 0.6369 | 0.7612 | 0.0511 |
| Persian | 9 | 116 | 0.3448 | 0.6552 | 0.1480 |
| Portuguese | 9 | 253 | 0.2609 | 0.3676 | 0.0311 |
| Romanian | 6 | 91 | 0.3407 | 0.4176 | 0.1197 |
| Russian | 10 | 380 | 0.3763 | 0.6868 | 0.0502 |
| Spanish | 13 | 394 | 0.3299 | 0.8934 | 0.0534 |
| Thai | 4 | 127 | 0.0000 | 0.1811 | 0.0466 |
| Turkish | 13 | 944 | 0.0858 | 0.9693 | 0.0052 |

## Cross-Hierarchy Comparison

| Hierarchy | Entities | Primary Rec | Control | Diff | Wilcoxon p | d |
|-----------|----------|-------------|---------|------|-----------|---|
| city-continent | 128 | 0.6193 | 0.0516 | +0.5677 | 4.105728234341213e-20 | 1.5017508286386902 |
| city-language | 201 | 0.3418 | 0.0682 | +0.2736 | 2.4120373948657706e-18 | 0.7529140433207754 |

## Comparison with iter_008 First-Letter

- **First-letter recovery (L12)**: 0.3245
- **First-letter control**: 0.0146
- **First-letter Wilcoxon p**: 0.00021792118340564174
- **First-letter Cohen's d**: 1.3264999383997025
- Note: iter_008 first-letter patching at L12 (different layer). Cross-domain at L24. Direct comparison is qualitative.
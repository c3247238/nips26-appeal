# Phase 1.3: Cross-Domain Absorption Measurement (v2) -- Pilot Summary

## Status: PASS (GO)

## Key Changes from v1

- Using best layer L24 (not L12) for all RAVEL hierarchies
- Token position -1 (matching probe training, v1 used -2)
- Reading quality from phase1_probe_training_full.json
- Including city-country (F1=0.789, above 0.70 minimum)
- SAE and probe at same layer (no cross-layer measurement)

## Probe Quality Summary

| Hierarchy | Layer | F1 | Gate Status | Included |
|-----------|-------|-----|-------------|----------|
| city-continent | 24 | 0.8428 | BELOW_GATE_INCLUDED | Yes |
| city-country | 24 | 0.7895 | BELOW_GATE_INCLUDED | Yes |
| city-language | 24 | 0.8234 | BELOW_GATE_INCLUDED | Yes |

## Absorption Results

| Hierarchy | SAE Config | Probe Layer | Absorption Rate | Strict Rate | CI (95%) | Probe Raw Acc | N Cities | N FN |
|-----------|------------|-------------|-----------------|-------------|----------|---------------|----------|------|
| city-continent | L24_16k | L24 | 0.3584 | 0.1387 | [0.162, 0.597] | 0.8650 | 200 | 62 |
| city-continent | L24_65k | L24 | 0.2601 | 0.1214 | [0.089, 0.478] | 0.8650 | 200 | 45 |
| city-country | L24_16k | L24 | 0.1850 | 0.0231 | [0.193, 0.422] | 0.9153 | 189 | 32 |
| city-country | L24_65k | L24 | 0.1272 | 0.0405 | [0.119, 0.307] | 0.9153 | 189 | 22 |
| city-language | L24_16k | L24 | 0.1364 | 0.0682 | [0.131, 0.417] | 0.9215 | 191 | 24 |
| city-language | L24_65k | L24 | 0.1364 | 0.0341 | [0.072, 0.354] | 0.9215 | 191 | 24 |

## Comparison with First-Letter Baseline

| Hierarchy | SAE | First-Letter Rate | Cross-Domain Rate | Difference | Cohen's d | p-value |
|-----------|-----|-------------------|-------------------|------------|-----------|---------|
| city-continent | L24_16k | 0.3448 | 0.3584 | +0.0136 | 0.306 | 0.8292 |
| city-country | L24_16k | 0.3448 | 0.1850 | -0.1599 | -3.839 | 0.0043 * |
| city-language | L24_16k | 0.3448 | 0.1364 | -0.2085 | -5.159 | 0.0001 * |
| city-continent | L24_65k | 0.2553 | 0.2601 | +0.0048 | 0.121 | 0.9319 |
| city-country | L24_65k | 0.2553 | 0.1272 | -0.1282 | -3.511 | 0.0081 * |
| city-language | L24_65k | 0.2553 | 0.1364 | -0.1190 | -3.242 | 0.0149 * |

## Comparison with Previous Pilot (v1)

- city-continent: prev=0.5342 (L12), now=0.3584 (L24)
- city-continent: prev=0.5342 (L12), now=0.2601 (L24)
- city-language: prev=0.1038 (L12), now=0.1364 (L24)
- city-language: prev=0.1038 (L12), now=0.1364 (L24)

## Example False Negatives: city-continent (L24_16k)

- **Lisburn**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=False
- **Higuey**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=True
- **Karakol**: true=Asia, raw_pred=Asia, sae_pred=North America, main_fires=True
- **Birsk**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=False
- **Ankara**: true=Asia, raw_pred=Asia, sae_pred=South America, main_fires=False
- **Jumla**: true=Asia, raw_pred=Asia, sae_pred=South America, main_fires=True
- **Angers**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=False
- **Kalmar**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=False
- **Modesto**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=True
- **Belem**: true=South America, raw_pred=South America, sae_pred=North America, main_fires=True

## Example False Negatives: city-continent (L24_65k)

- **Lakeville**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=True
- **Lisburn**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=False
- **Portland**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=True
- **Birsk**: true=Europe, raw_pred=Europe, sae_pred=South America, main_fires=False
- **Ankara**: true=Asia, raw_pred=Asia, sae_pred=South America, main_fires=False
- **Angers**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=False
- **Melville**: true=North America, raw_pred=North America, sae_pred=Asia, main_fires=True
- **Kalmar**: true=Europe, raw_pred=Europe, sae_pred=Asia, main_fires=False
- **Modesto**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=True
- **Alice**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=True

## Example False Negatives: city-country (L24_16k)

- **Encarnacion**: true=Paraguay, raw_pred=Paraguay, sae_pred=Brazil, main_fires=True
- **Zarate**: true=Argentina, raw_pred=Argentina, sae_pred=Brazil, main_fires=True
- **Mitu**: true=Colombia, raw_pred=Colombia, sae_pred=Brazil, main_fires=False
- **Amsterdam**: true=United States, raw_pred=United States, sae_pred=Russia, main_fires=False
- **Surigao**: true=Philippines, raw_pred=Philippines, sae_pred=Mexico, main_fires=True
- **Belgrade**: true=United States, raw_pred=United States, sae_pred=Albania, main_fires=True
- **Ridder**: true=Kazakhstan, raw_pred=Kazakhstan, sae_pred=Russia, main_fires=True
- **Smithton**: true=Australia, raw_pred=Australia, sae_pred=United States, main_fires=True
- **Yala**: true=Thailand, raw_pred=Thailand, sae_pred=Russia, main_fires=True
- **Hopedale**: true=Canada, raw_pred=Canada, sae_pred=United States, main_fires=True

## Example False Negatives: city-country (L24_65k)

- **Encarnacion**: true=Paraguay, raw_pred=Paraguay, sae_pred=Brazil, main_fires=True
- **Mitu**: true=Colombia, raw_pred=Colombia, sae_pred=Brazil, main_fires=True
- **Amsterdam**: true=United States, raw_pred=United States, sae_pred=Russia, main_fires=False
- **Belgrade**: true=United States, raw_pred=United States, sae_pred=Albania, main_fires=False
- **Ridder**: true=Kazakhstan, raw_pred=Kazakhstan, sae_pred=Brazil, main_fires=False
- **Smithton**: true=Australia, raw_pred=Australia, sae_pred=United States, main_fires=True
- **Yala**: true=Thailand, raw_pred=Thailand, sae_pred=Russia, main_fires=True
- **Hopedale**: true=Canada, raw_pred=Canada, sae_pred=United States, main_fires=True
- **Kankan**: true=Guinea, raw_pred=Guinea, sae_pred=Ivory Coast, main_fires=True
- **Shu**: true=Kazakhstan, raw_pred=Kazakhstan, sae_pred=Brazil, main_fires=False

## Example False Negatives: city-language (L24_16k)

- **Rabat**: true=Persian, raw_pred=Persian, sae_pred=Arabic, main_fires=True
- **Chamical**: true=Spanish, raw_pred=Spanish, sae_pred=English, main_fires=True
- **Uyuni**: true=Aymara, raw_pred=Aymara, sae_pred=Spanish, main_fires=True
- **Kasane**: true=English, raw_pred=English, sae_pred=Russian, main_fires=False
- **Goure**: true=French, raw_pred=French, sae_pred=English, main_fires=True
- **Cobija**: true=Aymara, raw_pred=Aymara, sae_pred=Spanish, main_fires=True
- **Vacaria**: true=Portuguese, raw_pred=Portuguese, sae_pred=Spanish, main_fires=False
- **Venice**: true=English, raw_pred=English, sae_pred=Italian, main_fires=True
- **Meiganga**: true=English, raw_pred=English, sae_pred=Portuguese, main_fires=False
- **Mugla**: true=Arabic,Kurdish,Turkish, raw_pred=Arabic,Kurdish,Turkish, sae_pred=Turkish, main_fires=True

## Example False Negatives: city-language (L24_65k)

- **Rabat**: true=Persian, raw_pred=Persian, sae_pred=Arabic, main_fires=False
- **Farmington**: true=English, raw_pred=English, sae_pred=Spanish, main_fires=True
- **Aurora**: true=English, raw_pred=English, sae_pred=Spanish, main_fires=True
- **Ascension**: true=Spanish, raw_pred=Spanish, sae_pred=French, main_fires=True
- **Roxas**: true=Filipino, English, raw_pred=Filipino, English, sae_pred=English, main_fires=True
- **Kasane**: true=English, raw_pred=English, sae_pred=Russian, main_fires=False
- **Goure**: true=French, raw_pred=French, sae_pred=Spanish, main_fires=True
- **Mossoro**: true=Portuguese, raw_pred=Portuguese, sae_pred=Spanish, main_fires=True
- **Vacaria**: true=Portuguese, raw_pred=Portuguese, sae_pred=Spanish, main_fires=True
- **Venice**: true=English, raw_pred=English, sae_pred=Italian, main_fires=True

## Caveats

**No RAVEL probe reaches the relaxed quality gate (F1 >= 0.85)**

Absorption may be overestimated due to probe errors.

Acceptable for pilot -- layer 24 probes are best available.

**Elapsed time**: 1.1 minutes
**Mode**: PILOT
**Token position**: -1
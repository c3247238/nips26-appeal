# Phase 1.2b: Cross-Domain Absorption -- Iter 9 FULL

**Mode**: FULL
**Time**: 3.5 minutes
**Token position**: -2
**Hierarchies**: city-continent, city-language, city-country
**SAE configs**: L24_16k, L24_65k

## Probe Quality

| Hierarchy | Layer | F1 | Gate | N Classes |
|-----------|-------|-----|------|-----------|
| city-continent | 24 | 0.8710 | PASS_RELAXED | 6 |
| city-language | 24 | 0.8179 | BELOW_GATE_INCLUDED | 23 |
| city-country | 24 | 0.7260 | BELOW_GATE_INCLUDED | 80 |

## Absorption Results

| Hierarchy | SAE | Absorption Rate | Strict Rate | CI (95%) | Strict CI | Probe Raw | Probe SAE | N | N FN |
|-----------|-----|-----------------|-------------|----------|-----------|-----------|-----------|---|------|
| city-continent | L24_16k | **0.3143** | 0.0195 | [0.289, 0.339] | [0.013, 0.027] | 0.8488 | 0.6088 | 1567 | 418 |
| city-continent | L24_65k | **0.3128** | 0.0045 | [0.287, 0.338] | [0.002, 0.008] | 0.8488 | 0.6056 | 1567 | 416 |
| city-country | L24_16k | **0.4510** | 0.0166 | [0.422, 0.479] | [0.010, 0.025] | 0.8128 | 0.4505 | 1405 | 515 |
| city-country | L24_65k | **0.3292** | 0.0149 | [0.302, 0.356] | [0.009, 0.022] | 0.8128 | 0.5480 | 1405 | 376 |
| city-language | L24_16k | **0.1156** | 0.0261 | [0.097, 0.135] | [0.017, 0.035] | 0.8731 | 0.7974 | 1229 | 124 |
| city-language | L24_65k | **0.0774** | 0.0177 | [0.062, 0.094] | [0.010, 0.026] | 0.8731 | 0.8242 | 1229 | 83 |

### Per-Class Breakdown (city-continent_L24_16k)

| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |
|-------|-------|---------------|-----|-----------------|-------------------|
| Africa | 278 | 231 | 9 | 0.0390 | 278 |
| Asia | 377 | 324 | 79 | 0.2438 | 353 |
| Europe | 315 | 276 | 249 | 0.9022 | 314 |
| North America | 288 | 241 | 46 | 0.1909 | 226 |
| Oceania | 78 | 51 | 27 | 0.5294 | 76 |
| South America | 231 | 207 | 8 | 0.0386 | 223 |

### Per-Class Breakdown (city-language_L24_16k)

| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |
|-------|-------|---------------|-----|-----------------|-------------------|
| Afrikaans | 10 | 9 | 4 | 0.4444 | 10 |
| Aimar√°,Ket¬öua,Spanish | 20 | 13 | 9 | 0.6923 | 20 |
| Arabic | 37 | 29 | 15 | 0.5172 | 31 |
| Arabic,Berberi | 12 | 9 | 6 | 0.6667 | 12 |
| Arabic,Kurdish,Turkish | 18 | 11 | 0 | 0.0000 | 14 |
| Aymara | 15 | 9 | 8 | 0.8889 | 13 |
| Chinese | 100 | 98 | 2 | 0.0204 | 99 |
| Chinese,Korean | 13 | 11 | 3 | 0.2727 | 12 |
| English | 388 | 378 | 14 | 0.0370 | 51 |
| French | 82 | 63 | 2 | 0.0317 | 65 |
| German | 33 | 24 | 5 | 0.2083 | 32 |
| Hindi | 17 | 17 | 3 | 0.1765 | 15 |
| Indonesian | 25 | 21 | 1 | 0.0476 | 24 |
| Italian | 23 | 16 | 0 | 0.0000 | 23 |
| Japanese | 15 | 15 | 2 | 0.1333 | 13 |
| Kazakh | 13 | 9 | 8 | 0.8889 | 11 |
| Persian | 22 | 19 | 1 | 0.0526 | 21 |
| Portuguese | 128 | 103 | 10 | 0.0971 | 119 |
| Romanian | 12 | 12 | 1 | 0.0833 | 12 |
| Russian | 100 | 92 | 11 | 0.1196 | 87 |
| Spanish | 123 | 97 | 10 | 0.1031 | 113 |
| Thai | 10 | 8 | 0 | 0.0000 | 9 |
| Turkish | 13 | 10 | 9 | 0.9000 | 13 |

### Per-Class Breakdown (city-country_L24_16k)

| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |
|-------|-------|---------------|-----|-----------------|-------------------|
| Albania | 12 | 6 | 6 | 1.0000 | 10 |
| Algeria | 13 | 11 | 11 | 1.0000 | 13 |
| Angola | 16 | 10 | 8 | 0.8000 | 16 |
| Argentina | 25 | 11 | 11 | 1.0000 | 15 |
| Australia | 60 | 31 | 31 | 1.0000 | 53 |
| Bangladesh | 5 | 5 | 4 | 0.8000 | 5 |
| Bolivia | 15 | 4 | 3 | 0.7500 | 15 |
| Brazil | 98 | 82 | 24 | 0.2927 | 83 |
| Bulgaria | 7 | 7 | 6 | 0.8571 | 7 |
| Burkina Faso | 9 | 1 | 1 | 1.0000 | 9 |
| Cameroon | 12 | 9 | 8 | 0.8889 | 12 |
| Canada | 49 | 28 | 22 | 0.7857 | 49 |
| Central African Republic | 7 | 1 | 1 | 1.0000 | 3 |
| Chad | 5 | 1 | 1 | 1.0000 | 5 |
| Chile | 16 | 12 | 11 | 0.9167 | 16 |
| China | 100 | 96 | 15 | 0.1562 | 99 |
| Colombia | 24 | 19 | 8 | 0.4211 | 24 |
| Cuba | 5 | 4 | 2 | 0.5000 | 5 |
| Dominican Republic | 6 | 5 | 5 | 1.0000 | 6 |
| Ecuador | 9 | 8 | 8 | 1.0000 | 9 |
| Egypt | 6 | 3 | 3 | 1.0000 | 5 |
| France | 19 | 17 | 11 | 0.6471 | 19 |
| Germany | 24 | 12 | 11 | 0.9167 | 24 |
| Ghana | 5 | 5 | 5 | 1.0000 | 5 |
| Greece | 6 | 5 | 5 | 1.0000 | 6 |
| Guinea | 9 | 8 | 8 | 1.0000 | 9 |
| India | 49 | 47 | 1 | 0.0213 | 49 |
| Indonesia | 25 | 23 | 1 | 0.0435 | 22 |
| Iran | 24 | 21 | 1 | 0.0476 | 23 |
| Italy | 23 | 18 | 18 | 1.0000 | 23 |
| Ivory Coast | 8 | 6 | 3 | 0.5000 | 8 |
| Japan | 15 | 13 | 10 | 0.7692 | 15 |
| Kazakhstan | 13 | 9 | 8 | 0.8889 | 11 |
| Kenya | 12 | 12 | 10 | 0.8333 | 12 |
| Madagascar | 5 | 4 | 4 | 1.0000 | 4 |
| Malaysia | 6 | 5 | 5 | 1.0000 | 5 |
| Mali | 7 | 4 | 4 | 1.0000 | 7 |
| Mexico | 31 | 27 | 7 | 0.2593 | 23 |
| Morocco | 5 | 5 | 4 | 0.8000 | 5 |
| Mozambique | 8 | 3 | 1 | 0.3333 | 8 |
| Myanmar | 9 | 9 | 6 | 0.6667 | 7 |
| Netherlands | 6 | 4 | 4 | 1.0000 | 6 |
| New Zealand | 9 | 7 | 4 | 0.5714 | 9 |
| Nigeria | 20 | 18 | 12 | 0.6667 | 20 |
| North Korea | 6 | 5 | 4 | 0.8000 | 5 |
| Norway | 8 | 6 | 4 | 0.6667 | 7 |
| Pakistan | 9 | 8 | 3 | 0.3750 | 9 |
| Papua New Guinea | 5 | 5 | 1 | 0.2000 | 5 |
| Paraguay | 5 | 1 | 1 | 1.0000 | 5 |
| Peru | 21 | 16 | 14 | 0.8750 | 21 |
| Philippines | 10 | 5 | 2 | 0.4000 | 10 |
| Poland | 6 | 6 | 4 | 0.6667 | 6 |
| Portugal | 6 | 5 | 5 | 1.0000 | 6 |
| Romania | 12 | 12 | 2 | 0.1667 | 12 |
| Russia | 92 | 84 | 33 | 0.3929 | 85 |
| Somalia | 5 | 4 | 2 | 0.5000 | 4 |
| South Africa | 12 | 9 | 4 | 0.4444 | 12 |
| South Korea | 10 | 9 | 9 | 1.0000 | 10 |
| South Sudan | 6 | 4 | 4 | 1.0000 | 6 |
| Spain | 6 | 6 | 5 | 0.8333 | 6 |
| Sudan | 5 | 4 | 4 | 1.0000 | 5 |
| Sweden | 7 | 7 | 4 | 0.5714 | 7 |
| Switzerland | 9 | 8 | 2 | 0.2500 | 9 |
| Taiwan | 6 | 4 | 4 | 1.0000 | 6 |
| Tanzania | 21 | 18 | 18 | 1.0000 | 21 |
| Thailand | 10 | 8 | 3 | 0.3750 | 9 |
| Tunisia | 7 | 6 | 4 | 0.6667 | 7 |
| Turkey | 31 | 26 | 4 | 0.1538 | 27 |
| Uganda | 15 | 14 | 14 | 1.0000 | 15 |
| Ukraine | 9 | 8 | 1 | 0.1250 | 9 |
| United Kingdom | 19 | 11 | 2 | 0.1818 | 19 |
| United States | 176 | 176 | 0 | 0.0000 | 138 |
| Uruguay | 5 | 3 | 3 | 1.0000 | 5 |
| Venezuela | 7 | 6 | 6 | 1.0000 | 7 |
| Yemen | 6 | 6 | 5 | 0.8333 | 6 |
| Zambia | 9 | 9 | 6 | 0.6667 | 9 |
| Zimbabwe | 7 | 7 | 6 | 0.8571 | 7 |

### Per-Class Breakdown (city-continent_L24_65k)

| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |
|-------|-------|---------------|-----|-----------------|-------------------|
| Africa | 278 | 231 | 12 | 0.0519 | 278 |
| Asia | 377 | 324 | 80 | 0.2469 | 374 |
| Europe | 315 | 276 | 254 | 0.9203 | 315 |
| North America | 288 | 241 | 34 | 0.1411 | 269 |
| Oceania | 78 | 51 | 28 | 0.5490 | 77 |
| South America | 231 | 207 | 8 | 0.0386 | 231 |

### Per-Class Breakdown (city-language_L24_65k)

| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |
|-------|-------|---------------|-----|-----------------|-------------------|
| Afrikaans | 10 | 9 | 4 | 0.4444 | 10 |
| Aimar√°,Ket¬öua,Spanish | 20 | 13 | 2 | 0.1538 | 19 |
| Arabic | 37 | 29 | 6 | 0.2069 | 22 |
| Arabic,Berberi | 12 | 9 | 4 | 0.4444 | 12 |
| Arabic,Kurdish,Turkish | 18 | 11 | 0 | 0.0000 | 13 |
| Aymara | 15 | 9 | 7 | 0.7778 | 14 |
| Chinese | 100 | 98 | 2 | 0.0204 | 89 |
| Chinese,Korean | 13 | 11 | 1 | 0.0909 | 12 |
| English | 388 | 378 | 9 | 0.0238 | 370 |
| French | 82 | 63 | 3 | 0.0476 | 72 |
| German | 33 | 24 | 2 | 0.0833 | 32 |
| Hindi | 17 | 17 | 5 | 0.2941 | 17 |
| Indonesian | 25 | 21 | 0 | 0.0000 | 24 |
| Italian | 23 | 16 | 1 | 0.0625 | 23 |
| Japanese | 15 | 15 | 2 | 0.1333 | 13 |
| Kazakh | 13 | 9 | 6 | 0.6667 | 11 |
| Persian | 22 | 19 | 1 | 0.0526 | 20 |
| Portuguese | 128 | 103 | 7 | 0.0680 | 113 |
| Romanian | 12 | 12 | 0 | 0.0000 | 12 |
| Russian | 100 | 92 | 4 | 0.0435 | 73 |
| Spanish | 123 | 97 | 10 | 0.1031 | 111 |
| Thai | 10 | 8 | 0 | 0.0000 | 8 |
| Turkish | 13 | 10 | 7 | 0.7000 | 13 |

### Per-Class Breakdown (city-country_L24_65k)

| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |
|-------|-------|---------------|-----|-----------------|-------------------|
| Albania | 12 | 6 | 5 | 0.8333 | 9 |
| Algeria | 13 | 11 | 9 | 0.8182 | 13 |
| Angola | 16 | 10 | 1 | 0.1000 | 16 |
| Argentina | 25 | 11 | 10 | 0.9091 | 19 |
| Australia | 60 | 31 | 28 | 0.9032 | 56 |
| Bangladesh | 5 | 5 | 0 | 0.0000 | 5 |
| Bolivia | 15 | 4 | 3 | 0.7500 | 15 |
| Brazil | 98 | 82 | 16 | 0.1951 | 86 |
| Bulgaria | 7 | 7 | 1 | 0.1429 | 7 |
| Burkina Faso | 9 | 1 | 1 | 1.0000 | 9 |
| Cameroon | 12 | 9 | 2 | 0.2222 | 10 |
| Canada | 49 | 28 | 17 | 0.6071 | 39 |
| Central African Republic | 7 | 1 | 1 | 1.0000 | 6 |
| Chad | 5 | 1 | 1 | 1.0000 | 5 |
| Chile | 16 | 12 | 7 | 0.5833 | 16 |
| China | 100 | 96 | 15 | 0.1562 | 99 |
| Colombia | 24 | 19 | 5 | 0.2632 | 24 |
| Cuba | 5 | 4 | 2 | 0.5000 | 5 |
| Dominican Republic | 6 | 5 | 5 | 1.0000 | 5 |
| Ecuador | 9 | 8 | 4 | 0.5000 | 9 |
| Egypt | 6 | 3 | 3 | 1.0000 | 4 |
| France | 19 | 17 | 3 | 0.1765 | 19 |
| Germany | 24 | 12 | 10 | 0.8333 | 24 |
| Ghana | 5 | 5 | 4 | 0.8000 | 5 |
| Greece | 6 | 5 | 3 | 0.6000 | 6 |
| Guinea | 9 | 8 | 8 | 1.0000 | 9 |
| India | 49 | 47 | 1 | 0.0213 | 49 |
| Indonesia | 25 | 23 | 1 | 0.0435 | 23 |
| Iran | 24 | 21 | 1 | 0.0476 | 21 |
| Italy | 23 | 18 | 16 | 0.8889 | 23 |
| Ivory Coast | 8 | 6 | 1 | 0.1667 | 8 |
| Japan | 15 | 13 | 11 | 0.8462 | 15 |
| Kazakhstan | 13 | 9 | 6 | 0.6667 | 12 |
| Kenya | 12 | 12 | 11 | 0.9167 | 12 |
| Madagascar | 5 | 4 | 3 | 0.7500 | 5 |
| Malaysia | 6 | 5 | 4 | 0.8000 | 5 |
| Mali | 7 | 4 | 4 | 1.0000 | 7 |
| Mexico | 31 | 27 | 3 | 0.1111 | 23 |
| Morocco | 5 | 5 | 1 | 0.2000 | 5 |
| Mozambique | 8 | 3 | 0 | 0.0000 | 8 |
| Myanmar | 9 | 9 | 7 | 0.7778 | 7 |
| Netherlands | 6 | 4 | 4 | 1.0000 | 6 |
| New Zealand | 9 | 7 | 1 | 0.1429 | 8 |
| Nigeria | 20 | 18 | 9 | 0.5000 | 20 |
| North Korea | 6 | 5 | 4 | 0.8000 | 6 |
| Norway | 8 | 6 | 1 | 0.1667 | 7 |
| Pakistan | 9 | 8 | 2 | 0.2500 | 9 |
| Papua New Guinea | 5 | 5 | 2 | 0.4000 | 5 |
| Paraguay | 5 | 1 | 1 | 1.0000 | 5 |
| Peru | 21 | 16 | 10 | 0.6250 | 21 |
| Philippines | 10 | 5 | 3 | 0.6000 | 10 |
| Poland | 6 | 6 | 4 | 0.6667 | 6 |
| Portugal | 6 | 5 | 2 | 0.4000 | 6 |
| Romania | 12 | 12 | 1 | 0.0833 | 12 |
| Russia | 92 | 84 | 34 | 0.4048 | 86 |
| Somalia | 5 | 4 | 2 | 0.5000 | 4 |
| South Africa | 12 | 9 | 2 | 0.2222 | 12 |
| South Korea | 10 | 9 | 7 | 0.7778 | 10 |
| South Sudan | 6 | 4 | 3 | 0.7500 | 6 |
| Spain | 6 | 6 | 4 | 0.6667 | 6 |
| Sudan | 5 | 4 | 3 | 0.7500 | 4 |
| Sweden | 7 | 7 | 1 | 0.1429 | 7 |
| Switzerland | 9 | 8 | 2 | 0.2500 | 9 |
| Taiwan | 6 | 4 | 3 | 0.7500 | 6 |
| Tanzania | 21 | 18 | 4 | 0.2222 | 21 |
| Thailand | 10 | 8 | 4 | 0.5000 | 9 |
| Tunisia | 7 | 6 | 2 | 0.3333 | 7 |
| Turkey | 31 | 26 | 5 | 0.1923 | 27 |
| Uganda | 15 | 14 | 8 | 0.5714 | 15 |
| Ukraine | 9 | 8 | 0 | 0.0000 | 9 |
| United Kingdom | 19 | 11 | 3 | 0.2727 | 19 |
| United States | 176 | 176 | 0 | 0.0000 | 113 |
| Uruguay | 5 | 3 | 3 | 1.0000 | 5 |
| Venezuela | 7 | 6 | 3 | 0.5000 | 7 |
| Yemen | 6 | 6 | 1 | 0.1667 | 6 |
| Zambia | 9 | 9 | 2 | 0.2222 | 9 |
| Zimbabwe | 7 | 7 | 2 | 0.2857 | 7 |

## Cross-Hierarchy ANOVA (Kruskal-Wallis)

- **L24_16k**: H=299.947, p=0.0000 (Significant: YES)
  - city-continent: mean=0.3143, n=1330
  - city-language: mean=0.1156, n=1073
  - city-country: mean=0.4510, n=1142
  - Pairwise Mann-Whitney:
    - city-continent_vs_city-language: p=0.0000 (Bonf: 0.0000)*
    - city-continent_vs_city-country: p=0.0000 (Bonf: 0.0000)*
    - city-language_vs_city-country: p=0.0000 (Bonf: 0.0000)*
- **L24_65k**: H=238.558, p=0.0000 (Significant: YES)
  - city-continent: mean=0.3128, n=1330
  - city-language: mean=0.0774, n=1073
  - city-country: mean=0.3292, n=1142
  - Pairwise Mann-Whitney:
    - city-continent_vs_city-language: p=0.0000 (Bonf: 0.0000)*
    - city-continent_vs_city-country: p=0.3819 (Bonf: 1.0000)
    - city-language_vs_city-country: p=0.0000 (Bonf: 0.0000)*

## Permutation Tests vs First-Letter

| Config | CD Rate | FL Rate | Diff | p (perm) | p (Bonf) | Cohen's h | Sig? |
|--------|---------|---------|------|----------|----------|-----------|------|
| city-continent_L24_16k | 0.3143 | 0.4286 | -0.1143 | 0.3423 | 1.0000 | -0.2372 | No |
| city-language_L24_16k | 0.1156 | 0.4286 | -0.3130 | 0.0005 | 0.0030 | -0.7337 | Yes |
| city-country_L24_16k | 0.4510 | 0.4286 | +0.0224 | 1.0000 | 1.0000 | 0.0451 | No |
| city-continent_L24_65k | 0.3128 | 0.2727 | +0.0401 | 0.8121 | 1.0000 | 0.0881 | No |
| city-language_L24_65k | 0.0774 | 0.2727 | -0.1954 | 0.0071 | 0.0426 | -0.5353 | Yes |
| city-country_L24_65k | 0.3292 | 0.2727 | +0.0565 | 0.6527 | 1.0000 | 0.1233 | No |

## Comparison with iter_008

- **city-continent_L24_16k**: i8=0.3584, i9=0.3143, diff=-0.0441 (OK)

### Example False Negatives (city-continent_L24_16k)

- **Abha**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Abohar**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Acarau**: true=South America, raw_pred=South America, sae_pred=Africa, main_fires=True
- **Aleppo**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=False
- **Alesund**: true=Europe, raw_pred=Europe, sae_pred=South America, main_fires=True
- **Aleysk**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=True
- **Alpine**: true=North America, raw_pred=North America, sae_pred=South America, main_fires=False
- **Ambala**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Ambon**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Amderma**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=True

### Example False Negatives (city-language_L24_16k)

- **Abha**: true=Arabic, raw_pred=Arabic, sae_pred=French, main_fires=False
- **Adrar**: true=Arabic,Berberi, raw_pred=Arabic,Berberi, sae_pred=Arabic, main_fires=True
- **Agordat**: true=Arabic, raw_pred=Arabic, sae_pred=English, main_fires=True
- **Aleppo**: true=Arabic, raw_pred=Arabic, sae_pred=English, main_fires=True
- **Almaty**: true=Kazakh, raw_pred=Kazakh, sae_pred=Russian, main_fires=True
- **Almenara**: true=Portuguese, raw_pred=Portuguese, sae_pred=Spanish, main_fires=True
- **Ankara**: true=Turkish, raw_pred=Turkish, sae_pred=Arabic,Kurdish,Turkish, main_fires=True
- **Arequipa**: true=Aimar√°,Ket¬öua,Spanish, raw_pred=Aimar√°,Ket¬öua,Spanish, sae_pred=Spanish, main_fires=True
- **Arys**: true=Kazakh, raw_pred=Kazakh, sae_pred=Arabic,Kurdish,Turkish, main_fires=True
- **Asha**: true=Russian, raw_pred=Russian, sae_pred=English, main_fires=False

### Example False Negatives (city-country_L24_16k)

- **Abidjan**: true=Ivory Coast, raw_pred=Ivory Coast, sae_pred=United States, main_fires=True
- **Abuja**: true=Nigeria, raw_pred=Nigeria, sae_pred=United States, main_fires=True
- **Acarau**: true=Brazil, raw_pred=Brazil, sae_pred=United States, main_fires=True
- **Acarigua**: true=Venezuela, raw_pred=Venezuela, sae_pred=Bolivia, main_fires=True
- **Accra**: true=Ghana, raw_pred=Ghana, sae_pred=United States, main_fires=True
- **Adrar**: true=Algeria, raw_pred=Algeria, sae_pred=United States, main_fires=True
- **Albury**: true=Australia, raw_pred=Australia, sae_pred=United States, main_fires=True
- **Alesund**: true=Norway, raw_pred=Norway, sae_pred=United States, main_fires=True
- **Algiers**: true=Algeria, raw_pred=Algeria, sae_pred=United States, main_fires=True
- **Almaty**: true=Kazakhstan, raw_pred=Kazakhstan, sae_pred=United States, main_fires=True

### Example False Negatives (city-continent_L24_65k)

- **Abha**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Albury**: true=Oceania, raw_pred=Oceania, sae_pred=North America, main_fires=True
- **Aleppo**: true=Asia, raw_pred=Asia, sae_pred=South America, main_fires=True
- **Alesund**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=True
- **Aleysk**: true=Europe, raw_pred=Europe, sae_pred=Asia, main_fires=True
- **Ambala**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Ambon**: true=Asia, raw_pred=Asia, sae_pred=South America, main_fires=True
- **Amderma**: true=Europe, raw_pred=Europe, sae_pred=North America, main_fires=True
- **Amol**: true=Asia, raw_pred=Asia, sae_pred=Africa, main_fires=True
- **Amravati**: true=Asia, raw_pred=Asia, sae_pred=South America, main_fires=True

### Example False Negatives (city-language_L24_65k)

- **Adrar**: true=Arabic,Berberi, raw_pred=Arabic,Berberi, sae_pred=French, main_fires=True
- **Almenara**: true=Portuguese, raw_pred=Portuguese, sae_pred=Spanish, main_fires=False
- **Ambala**: true=Hindi, raw_pred=Hindi, sae_pred=English, main_fires=True
- **Ankara**: true=Turkish, raw_pred=Turkish, sae_pred=Arabic,Kurdish,Turkish, main_fires=True
- **Aosta**: true=Italian, raw_pred=Italian, sae_pred=French, main_fires=True
- **Arys**: true=Kazakh, raw_pred=Kazakh, sae_pred=Russian, main_fires=True
- **Asha**: true=Russian, raw_pred=Russian, sae_pred=English, main_fires=False
- **Asino**: true=Russian, raw_pred=Russian, sae_pred=French, main_fires=True
- **Aurora**: true=English, raw_pred=English, sae_pred=Spanish, main_fires=True
- **Babanusa**: true=Arabic, raw_pred=Arabic, sae_pred=English, main_fires=False

### Example False Negatives (city-country_L24_65k)

- **Abuja**: true=Nigeria, raw_pred=Nigeria, sae_pred=United States, main_fires=True
- **Acarau**: true=Brazil, raw_pred=Brazil, sae_pred=United States, main_fires=True
- **Acarigua**: true=Venezuela, raw_pred=Venezuela, sae_pred=Brazil, main_fires=True
- **Accra**: true=Ghana, raw_pred=Ghana, sae_pred=United States, main_fires=True
- **Adrar**: true=Algeria, raw_pred=Algeria, sae_pred=United States, main_fires=True
- **Albury**: true=Australia, raw_pred=Australia, sae_pred=United States, main_fires=True
- **Alesund**: true=Norway, raw_pred=Norway, sae_pred=United States, main_fires=True
- **Algiers**: true=Algeria, raw_pred=Algeria, sae_pred=United States, main_fires=True
- **Almenara**: true=Brazil, raw_pred=Brazil, sae_pred=Mexico, main_fires=False
- **Anaco**: true=Venezuela, raw_pred=Venezuela, sae_pred=Mexico, main_fires=True
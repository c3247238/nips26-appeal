# Phase 0.1 FULL: Activation Patching at Scale

## Verdict: **CAUSAL_ABSORPTION_CONFIRMED**

Activation patching provides statistically significant causal evidence for feature absorption. Zeroing child features recovers correct letter predictions at rate 0.325 vs. control 0.015 (diff=0.310, best p=0.0000 [bootstrap], Cohen's d=1.33). N=19 words with absorption, 16 show positive recovery effect.

## Configuration
- Mode: **FULL** (iter_009 full-mode experiment)
- Model: Gemma 2 2B
- SAE: gemma-scope-2b-pt-res-canonical, layer_12/width_16k/canonical
- Probe: sklearn LogReg at layer 12, F1=0.883
- Contexts per word: 200
- Control features: 15 (magnitude-matched)
- Bootstrap resamples: 10000

## Word Count
- Pilot core words: 7
- Discovered pairs: 18
- Total completed: 25
- With absorption >= 3: 19

## Statistical Tests
- Wilcoxon signed-rank (one-sided): stat=136.0, p=0.00021792118340564174
- Mann-Whitney U (one-sided): stat=319.0, p=1.9528000882969987e-05
- Paired t-test (one-sided): t=5.627862610115804, p=1.2181631230213176e-05
- Bootstrap p-value: 0.0
- Cohen's d: 1.3264999383997025
- 95% CI on recovery difference: [0.21264998204575766, 0.42079308318643494]

## Results Table

| Word | Letter | Source | Raw Acc | Recon Acc | Absorbed | Recovery (abs) | Ctrl Recovery |
|------|--------|--------|---------|-----------|----------|----------------|---------------|
| eight | e | pilot_core | 0.000 | 0.000 | 0 | 0.000 | 0.000 |
| liked | l | pilot_core | 1.000 | 1.000 | 0 | 0.000 | 0.000 |
| lower | l | pilot_core | 0.960 | 1.000 | 0 | 0.000 | 0.000 |
| offer | o | pilot_core | 1.000 | 1.000 | 0 | 0.000 | 0.000 |
| often | o | pilot_core | 0.900 | 0.880 | 12 | 0.000 | 0.000 |
| other | o | pilot_core | 1.000 | 1.000 | 0 | 0.000 | 0.000 |
| under | u | pilot_core | 1.000 | 1.000 | 0 | 0.000 | 0.000 |
| xfa | x | discovered_ig | 0.100 | 0.080 | 12 | 0.000 | 0.000 |
| transitive | t | discovered_ig | 0.780 | 0.900 | 16 | **0.250** | 0.000 |
| udy | u | discovered_ig | 0.120 | 0.020 | 20 | **0.200** | 0.000 |
| recoge | r | discovered_ig | 1.000 | 0.420 | 116 | **0.103** | 0.041 |
| washingtonpost | w | discovered_ig | 0.120 | 0.140 | 20 | **0.400** | 0.000 |
| uzu | u | discovered_ig | 0.060 | 0.000 | 12 | **0.333** | 0.000 |
| menjadikan | m | discovered_ig | 0.940 | 0.560 | 76 | **0.105** | 0.014 |
| backward | b | discovered_ig | 0.940 | 0.860 | 16 | **0.500** | 0.050 |
| wner | w | discovered_ig | 0.060 | 0.000 | 12 | **0.333** | 0.000 |
| uestions | u | discovered_ig | 0.200 | 0.060 | 32 | **0.375** | 0.000 |
| wikk | w | discovered_ig | 0.080 | 0.060 | 12 | **0.667** | 0.022 |
| uki | u | discovered_ig | 0.080 | 0.000 | 16 | **0.250** | 0.000 |
| antigos | a | discovered_ig | 0.660 | 0.480 | 36 | **0.444** | 0.007 |
| zorgt | z | discovered_ig | 0.940 | 0.820 | 24 | **0.500** | 0.067 |
| conmigo | c | discovered_ig | 0.400 | 0.300 | 40 | **1.000** | 0.040 |
| unton | u | discovered_ig | 0.080 | 0.020 | 16 | **0.250** | 0.017 |
| yaitu | y | discovered_ig | 1.000 | 0.780 | 44 | **0.455** | 0.018 |
| membuka | m | discovered_ig | 0.140 | 0.000 | 28 | 0.000 | 0.000 |

## Aggregate
- Mean absorption rate: 0.588
- Mean recovery (child zeroed): 0.325
- Mean recovery (control): 0.015
- Recovery difference: 0.310
- Words with positive recovery effect: 16/19

## Cross-Hierarchy Note
Cross-hierarchy activation patching was NOT attempted because RAVEL probes do not pass the quality gate:
- city-continent: best F1=0.843 (gate=0.85)
- city-language: best F1=0.823 (gate=0.85)
- city-country: best F1=0.789 (gate=0.85)
This remains future work pending better probe training methods.

## Elapsed: 2.3 minutes
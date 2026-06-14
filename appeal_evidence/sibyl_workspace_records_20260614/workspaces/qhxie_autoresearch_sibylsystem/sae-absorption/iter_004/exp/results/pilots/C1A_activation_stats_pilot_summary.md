# C1A Activation Stats — PILOT Summary

**GO/NO-GO: GO**

## Configuration
- Model: GPT-2 Small (open-model anchor; Gemma-2-2b requires gated access)
- SAE: gpt2-small-res-jb / blocks.8.hook_resid_pre (d_sae=24576)
- Tokens: 1000 (pilot: 1k)
- Cosine threshold: 0.15
- f_min: 0.001

## Pass Criteria Results
- Valid pairs >= 500: PASS (n=4410908)
- No NaN/inf in alpha_ij: PASS (nan=0, inf=0)
- Pairs with alpha_ij > 1.0 >= 10: PASS (n=1206570)

## Alpha_ij Distribution
- Mean: 0.9019
- Median: 0.8500
- Max: 195.5000

## Top-10 Pairs by alpha_ij
| latent_i | latent_j | f_i | f_j | alpha_ij | decoder_cosine |
|---|---|---|---|---|---|
| 3194.0 | 22852.0 | 0.0020 | 0.3910 | 195.5000 | 0.3711 |
| 793.0 | 16021.0 | 0.0020 | 0.3540 | 177.0000 | 0.3409 |
| 3165.0 | 16021.0 | 0.0020 | 0.3540 | 177.0000 | 0.2713 |
| 11242.0 | 15075.0 | 0.0030 | 0.4430 | 147.6667 | 0.1843 |
| 13861.0 | 15075.0 | 0.0030 | 0.4430 | 147.6667 | 0.1999 |
| 16752.0 | 22852.0 | 0.0030 | 0.3910 | 130.3333 | 0.1761 |
| 619.0 | 15075.0 | 0.0030 | 0.4430 | 98.4444 | 0.2012 |
| 6219.0 | 22852.0 | 0.0040 | 0.3910 | 97.7500 | 0.2362 |
| 8761.0 | 22852.0 | 0.0040 | 0.3910 | 97.7500 | 0.1554 |
| 13867.0 | 22852.0 | 0.0020 | 0.3910 | 97.7500 | 0.2187 |
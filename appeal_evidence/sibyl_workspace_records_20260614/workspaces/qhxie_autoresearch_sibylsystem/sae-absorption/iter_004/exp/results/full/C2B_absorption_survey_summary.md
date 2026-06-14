# C2B Absorption Survey (Full) — Summary

**Mode**: FULL  |  **Model**: GPT-2 Small (open-model anchor)  |  **Timestamp**: 2026-04-14T21:47:21.375637

**Configs completed**: 31/31 planned  |  **Letters**: 26  |  **N per letter**: 100
**Total valid data points**: 806 / 806 rows
**Runtime**: ~78 minutes on NVIDIA RTX PRO 6000 Blackwell (95GB VRAM)

## Key Findings

- Peak absorption at layers 3-6 (0.09-0.12 mean rate across letters)
- Width paradox confirmed: absorption rises with SAE width at layer 8 (0.047 at 768 -> 0.104 at 98304)
- Layers 10-11 show near-zero absorption (first-letter features not represented in late layers)

## Absorption Rate by Config (mean over 26 letters)

| Config | Layer | Width | Hook Type | Mean | Std | Min | Max |
|--------|-------|-------|-----------|------|-----|-----|-----|
| resid_pre_L5_24k | 5 | 24,576 | resid_pre | 0.1192 | 0.0956 | 0.0000 | 0.5000 |
| resid_pre_L8_98304 | 8 | 98,304 | resid_pre | 0.1038 | 0.1986 | 0.0000 | 1.0000 |
| resid_pre_L4_24k | 4 | 24,576 | resid_pre | 0.0984 | 0.0665 | 0.0000 | 0.2647 |
| resid_pre_L6_24k | 6 | 24,576 | resid_pre | 0.0951 | 0.0637 | 0.0000 | 0.2692 |
| resid_mid_L6_32k | 6 | 32,768 | resid_mid | 0.0931 | 0.1821 | 0.0000 | 0.5714 |
| resid_pre_L8_49152 | 8 | 49,152 | resid_pre | 0.0921 | 0.1881 | 0.0000 | 1.0000 |
| resid_pre_L3_24k | 3 | 24,576 | resid_pre | 0.0911 | 0.0604 | 0.0000 | 0.2353 |
| resid_mid_L4_32k | 4 | 32,768 | resid_mid | 0.0864 | 0.1798 | 0.0000 | 0.5915 |
| resid_pre_L8_24576 | 8 | 24,576 | resid_pre | 0.0860 | 0.1017 | 0.0000 | 0.5000 |
| resid_post_L4_32k | 4 | 32,768 | resid_post | 0.0666 | 0.1573 | 0.0000 | 0.5309 |
| resid_post_L4_128k | 4 | 131,072 | resid_post | 0.0646 | 0.1536 | 0.0000 | 0.5100 |
| resid_pre_L7_24k | 7 | 24,576 | resid_pre | 0.0579 | 0.0468 | 0.0000 | 0.1500 |
| resid_post_L6_128k | 6 | 131,072 | resid_post | 0.0496 | 0.1376 | 0.0000 | 0.5100 |
| resid_pre_L8_12288 | 8 | 12,288 | resid_pre | 0.0479 | 0.0750 | 0.0000 | 0.3571 |
| resid_post_L6_32k | 6 | 32,768 | resid_post | 0.0476 | 0.1241 | 0.0000 | 0.4571 |
| resid_pre_L8_768 | 8 | 768 | resid_pre | 0.0470 | 0.1960 | 0.0000 | 1.0000 |
| resid_pre_L8_24k | 8 | 24,576 | resid_pre | 0.0464 | 0.0392 | 0.0000 | 0.1429 |
| resid_pre_L8_6144 | 8 | 6,144 | resid_pre | 0.0254 | 0.0358 | 0.0000 | 0.1429 |
| resid_pre_L8_1536 | 8 | 1,536 | resid_pre | 0.0195 | 0.0358 | 0.0000 | 0.1324 |
| resid_post_L8_128k | 8 | 131,072 | resid_post | 0.0146 | 0.0319 | 0.0000 | 0.1000 |
| resid_mid_L8_32k | 8 | 32,768 | resid_mid | 0.0136 | 0.0350 | 0.0000 | 0.1296 |
| resid_pre_L8_3072 | 8 | 3,072 | resid_pre | 0.0087 | 0.0230 | 0.0000 | 0.1064 |
| resid_pre_L9_24k | 9 | 24,576 | resid_pre | 0.0082 | 0.0140 | 0.0000 | 0.0500 |
| resid_post_L8_32k | 8 | 32,768 | resid_post | 0.0079 | 0.0229 | 0.0000 | 0.1111 |
| resid_pre_L10_24k | 10 | 24,576 | resid_pre | 0.0044 | 0.0167 | 0.0000 | 0.0833 |
| resid_post_L11_32k | 11 | 32,768 | resid_post | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| resid_post_L11_128k | 11 | 131,072 | resid_post | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| resid_mid_L10_32k | 10 | 32,768 | resid_mid | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| resid_post_L10_128k | 10 | 131,072 | resid_post | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| resid_post_L10_32k | 10 | 32,768 | resid_post | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| resid_pre_L11_24k | 11 | 24,576 | resid_pre | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Width Effect at Layer 8 (Feature Splitting)

| Width | Mean Absorption Rate |
|-------|---------------------|
| 768 | 0.0470 |
| 1,536 | 0.0195 |
| 3,072 | 0.0087 |
| 6,144 | 0.0254 |
| 12,288 | 0.0479 |
| 24,576 | 0.0860 |
| 49,152 | 0.0921 |
| 98,304 | 0.1038 |

## Layer Effect (resid_pre, 24k width)

| Layer | Mean Absorption Rate |
|-------|---------------------|
| 3 | 0.0911 |
| 4 | 0.0984 |
| 5 | 0.1192 |
| 6 | 0.0951 |
| 7 | 0.0579 |
| 8 | 0.0464 |
| 9 | 0.0082 |
| 10 | 0.0044 |
| 11 | 0.0000 |

## Overall Statistics

- Mean: 0.0450
- Std: 0.1053
- Range: [0.0000, 1.0000]

## Artifacts

- Parquet: exp/results/full/C2B_absorption_survey.parquet
- Summary JSON: exp/results/full/C2B_absorption_survey_summary.json
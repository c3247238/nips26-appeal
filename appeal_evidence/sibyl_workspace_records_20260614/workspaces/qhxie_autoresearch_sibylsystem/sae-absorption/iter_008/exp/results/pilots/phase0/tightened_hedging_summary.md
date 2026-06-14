# Phase 0.2: Tightened Hedging Classification -- Pilot Results

**Date:** 2026-04-15T19:46:43.295520
**Mode:** PILOT (100 samples/letter)
**Model:** Gemma 2 2B, Layer 12, SAE 16k width

## Probe Quality
- Test accuracy: 0.9077
- Test macro F1: 0.9082

## Absorption Rates Across L0
- L0=22: 13.3% (340 FNs)
- L0=82: 3.2% (82 FNs)
- L0=176: 1.2% (30 FNs)

## Hedging Decomposition (L0=22 -> L0=176)

| Classification | Count | Percentage | Description |
|---------------|-------|-----------|-------------|
| **Loose hedging** | 315 | 92.6% | Any resolution (current method) |
| **Strict hedging** | 25 | 7.4% | Parent latent fires + resolves |
| **Compensatory** | 290 | 85.3% | Resolves without parent |
| **Persistent** | 25 | 7.4% | Does not resolve |

## Key Finding

Strict hedging rate: 7.4% vs loose hedging rate: 92.6% (difference: 85.3 pp)

85.3% of FNs resolve via compensatory features (not the specific parent), indicating non-specific resolution.

## Parent Latent Identification

| Letter | Feature Idx | Cosine Sim |
|--------|-----------|-----------|
| a | 9511 | 0.5970 |
| b | 4199 | 0.6766 |
| c | 10339 | 0.6065 |
| d | 6233 | 0.7187 |
| e | 8716 | 0.6202 |
| f | 4473 | 0.6207 |
| g | 6087 | 0.6561 |
| h | 1535 | 0.6861 |
| i | 14534 | 0.6238 |
| j | 6147 | 0.6613 |
| k | 5003 | 0.5957 |
| l | 13208 | 0.6442 |
| m | 2902 | 0.5903 |
| n | 1044 | 0.6600 |
| o | 9446 | 0.6815 |
| p | 4459 | 0.5215 |
| q | 5637 | 0.6604 |
| r | 10466 | 0.6180 |
| s | 6245 | 0.4842 |
| t | 7742 | 0.6307 |
| u | 12675 | 0.6168 |
| v | 2385 | 0.6748 |
| w | 4876 | 0.6883 |
| x | 6610 | 0.4590 |
| y | 7009 | 0.6010 |
| z | 10350 | 0.6037 |
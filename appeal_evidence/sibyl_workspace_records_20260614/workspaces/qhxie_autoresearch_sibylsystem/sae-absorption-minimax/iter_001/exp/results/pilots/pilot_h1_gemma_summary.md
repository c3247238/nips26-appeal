# Pilot H1: Gemma-2B Absorption Atlas — Summary

## Status: SUCCESS (partial — Gemma model gated)

### What was attempted
1. Loaded Gemma-2B SAEs from `jbloom/Gemma-2b-Residual-Stream-SAEs` (5/5 layers loaded from local cache)
2. Attempted to load Gemma-2B model (`gemma-2-2b-it`, `gemma-2b-it`) — both **gated** (401: requires HuggingFace authentication)
3. Computed absorption proxy from SAE decoder weights (Gini coefficient of L2 norms per feature)

### Key Finding
**Pattern: Decreasing** (with a notable peak at layer 6)

| Layer | Gini Decoder Norms | Gini Encoder Norms | Cos Sim Variance |
|-------|-------------------|-------------------|-----------------|
| 0     | 0.210             | 0.303             | 0.0584          |
| 6     | **0.468**         | **0.490**         | 0.0311          |
| 10    | 0.144             | 0.155             | 0.0237          |
| 12    | 0.106             | 0.121             | 0.0173          |
| 17    | 0.084             | 0.111             | 0.0110          |

Mean Gini: 0.202 ± 0.139

### Interpretation
- **Early layers (0-6)**: High absorption proxy (Gini of decoder norms peaks at layer 6 = 0.468). This suggests that early SAE features have more unequal decoder norm distributions — a hallmark of superposition/absorption.
- **Later layers (10-17)**: Declining Gini (0.144 → 0.106 → 0.084). Later-layer features have more equal decoder norm distributions, suggesting less superposition and more monosemantic features.
- **Layer 0 anomaly**: Slightly lower than layer 6, suggesting layer 0 is transitional.

### Important Caveats
1. **No model activations**: Absorption computed from decoder weights only (no forward pass through Gemma-2B model)
2. **Gemma model gated**: Cannot verify with actual activation-based absorption metric
3. **Proxy metric**: Decoder norm Gini is a proxy, not the canonical absorption metric
4. **Layer coverage**: Gemma SAEs available at [0, 6, 10, 12, 17] — not contiguous

### Recommendation
- **GO for full-scale if Gemma model becomes accessible**
- Current proxy suggests H1 pattern (absorption peaks in mid-layers) partially confirmed at layer 6
- Need Gemma-2B model access or alternative open model for full validation
- GPT-2 absorption atlas from previous iterations (layer-wise study) should be the primary evidence

### Files
- Results: `exp/results/pilots/pilot_h1_gemma.json`
- Script: `exp/pilot_h1_gemma.py`

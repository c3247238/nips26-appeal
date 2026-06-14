# d2Cache Integration Pilot Summary

## Task: d2cache_integration_pilot
## Decision: FALL_BACK_THEORETICAL (NO_GO for kernel-level integration)

### Key Findings

| Configuration | TPS | Speedup vs HF Baseline | Accuracy (GSM8K) |
|---|---|---|---|
| HF Model Baseline (64 steps) | 58.5 | 1.00x | 73% |
| d2Cache Model (no cache) | 3.85 | 0.07x | 90% |
| d2Cache + dLLMCache | 16.87 | 0.29x | 90% |
| d2Cache + PrefixCache | 24.34 | 0.42x | 80% |

### Cache Hit Rate Measurements (Entropy-Based)

| Entropy Threshold | GSM8K CHR | MATH500 CHR | Theoretical Speedup |
|---|---|---|---|
| 0.5 | 93.3% | 87.1% | 2.27x |
| 1.0 | 97.0% | 93.5% | 2.39x |
| 2.0 | 99.1% | 98.0% | 2.47x |

### Root Cause Analysis

d2Cache's model implementation requires  (needed to output
attention weights for the d2Cache algorithm) instead of SDPA/FlashAttention. On RTX PRO 6000
Blackwell (97GB VRAM), this is 15x slower than the optimized HF model with SDPA.

The cache itself works correctly: PrefixCache achieves 6.3x internal speedup. But the
model overhead is so large that the net TPS is still 0.42x the HF baseline.

### Implication for M1 Pareto

Use the simplified entropy-based cache from iter_001 with the HF model.
Report CHR measurements from this pilot as empirical evidence.
Label M1 speedup as "projected" based on CHR * attention_fraction savings.
The gap between projected and d2Cache-actual speedup is a legitimate finding.

### Accuracy Note

d2Cache model shows higher GSM8K accuracy (90%) than HF model (73%) on this 10-sample
verification set, likely due to: (1) small sample size variance, (2) d2Cache model uses
the num_transfer_tokens=1 schedule (fully sequential unmasking) vs HF model's standard
get_num_transfer_tokens schedule. This will be controlled in the full experiment.

### Runtime: 54.1 minutes on 1x RTX PRO 6000 Blackwell

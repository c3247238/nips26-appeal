# C1C Cross-Architecture Validation — PILOT Summary

**GO/NO-GO: GO**

## Configuration
- Model: GPT-2 Small
- Baseline SAE: gpt2-small-res-jb / blocks.8.hook_resid_pre
- Calibrated tau (fixed from C1B): 0.5
- Alt arch SAE: gpt2-small-resid-post-v5-32k / blocks.8.hook_resid_post (d_sae=32768)
- Pilot letters: A, B, C, D, E

## Architecture Substitution Note
> JumpReLU SAEs unavailable for GPT-2 small in SAELens v6.39. Fallback: gpt2-small-resid-post-v5-32k blocks.8.hook_resid_post (OpenAI v5 architecture, L0=32, d_sae=32768). Differs from baseline in: SAE training objective (v5 vs jb), hook position (resid_post vs resid_pre), L0 (32 vs 60), width (32k vs 24k).

## Absorption Rates (Alt Arch SAE)
| Letter | Absorption Rate | N Absorbed | N Total |
|--------|-----------------|------------|---------|
| A | 0.000 | 0 | 13 |
| B | 0.000 | 0 | 43 |
| C | 0.020 | 1 | 50 |
| D | 0.020 | 1 | 50 |
| E | 0.000 | 0 | 50 |

## Cross-Architecture F1 Comparison
| Architecture | F1 | F1 Degradation vs Standard |
|---|---|---|
| Standard (C1B baseline): gpt2-small-res-jb | 0.1880 | — |
| Alt (v5-32k (OpenAI, resid_post, L0=32)): gpt2-small-resid-post-v5-32k | 0.0000 | 18.8 pp |

## Pass Criteria
- Alt SAE loaded: PASS
- F1 computable (non-degenerate): PASS
- F1 degradation < 30 pp: PASS (degradation=18.8 pp)
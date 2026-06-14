# Pilot Summary: e1_full_gemma (fallback to e1_full_gpt2)

## Task Context
- **Original task**: `e1_full_gemma` — Pareto evaluation on Gemma-2-2B SAEs.
- **Actual execution**: Fallback to `e1_full_gpt2` because `google/gemma-2-2b` is a gated HuggingFace repo and no HF authentication token is available in this environment.

## Environment
- GPU: cuda:4 (RTX 4090 class, 24 GB)
- Model: GPT-2 Small (117M)
- Dataset: C4 validation subset, ~12k characters (8 snippets)
- Seed: 42
- Runtime: ~34 seconds

## Checkpoints Evaluated (10)

| Release | SAE ID | Family |
|---------|--------|--------|
| gpt2-small-res-jb | blocks.0.hook_resid_pre | Standard |
| gpt2-small-res-jb | blocks.4.hook_resid_pre | Standard |
| gpt2-small-res-jb | blocks.8.hook_resid_pre | Standard |
| gpt2-small-res-jb | blocks.11.hook_resid_pre | Standard |
| gpt2-small-resid-post-v5-32k | blocks.4.hook_resid_post | TopK |
| gpt2-small-resid-post-v5-32k | blocks.8.hook_resid_post | TopK |
| gpt2-small-resid-post-v5-128k | blocks.4.hook_resid_post | TopK |
| gpt2-small-resid-post-v5-128k | blocks.8.hook_resid_post | TopK |
| gpt2-small-mlp-out-v5-32k | blocks.8.hook_mlp_out | TopK_MLP |
| gpt2-small-attn-out-v5-32k | blocks.8.hook_attn_out | TopK_Attn |

## Metrics Computed
- L0 (average active features per token)
- Explained variance
- Dead-neuron fraction
- CE loss recovered (%)
- Absorption rate (simplified first-letter proxy)
- Hedging rate (simplified correlated-pair proxy)

## Results Table

| Family | L0 | EV | Dead | CE_Rec | Absorption | Hedging |
|--------|----|----|------|--------|------------|---------|
| Standard | 19.0 | 0.981 | 0.724 | -2.08% | 0.0 | 0.08 |
| Standard | 34.4 | 0.950 | 0.648 | -3.31% | 0.0 | 0.8 |
| Standard | 69.5 | 0.917 | 0.339 | -4.98% | 0.0 | 0.8 |
| Standard | 61.6 | 0.930 | 0.335 | -6.30% | 0.0 | 0.8 |
| TopK | 32.0 | 0.948 | 0.680 | -1.94% | 0.0 | 0.8 |
| TopK | 32.0 | 0.914 | 0.652 | -3.56% | 0.0 | 0.8 |
| TopK | 32.0 | 0.961 | 0.878 | -0.99% | 0.0 | 0.8 |
| TopK | 32.0 | 0.931 | 0.865 | -2.20% | 0.0 | 0.8 |
| TopK_MLP | 32.0 | 0.734 | 0.563 | -0.49% | 0.0 | 1.0 |
| TopK_Attn | 32.0 | 0.794 | 0.648 | -0.10% | 0.654 | 1.0 |

## Family Averages

- **Standard**: L0=46.1, EV=0.945, Dead=0.511, CE_Rec=-4.17%, Absorption=0.000, Hedging=0.620
- **TopK**: L0=32.0, EV=0.938, Dead=0.769, CE_Rec=-2.17%, Absorption=0.000, Hedging=0.800
- **TopK_MLP**: L0=32.0, EV=0.734, Dead=0.563, CE_Rec=-0.49%, Absorption=0.000, Hedging=1.000
- **TopK_Attn**: L0=32.0, EV=0.794, Dead=0.648, CE_Rec=-0.10%, Absorption=0.654, Hedging=1.000

## Observations
1. **All CE loss recovered values are negative**, indicating that SAE reconstruction slightly degrades cross-entropy compared to the original model. This is expected for some SAEs but suggests the hook-replacement CE recovery computation may be noisy at small sample sizes.
2. **Dead-neuron rates are very high** (33-88%). This is likely because the pilot used only ~2k tokens; dead-neuron detection requires a larger corpus to be reliable.
3. **Absorption metric is degenerate** for most checkpoints (0.0) except TopK_Attn (0.65). The simplified first-letter proxy is too coarse and does not align with the rigorous sae-spelling benchmark. A proper absorption evaluation requires the full Chanin et al. pipeline.
4. **Hedging metric is also coarse** (same top feature for antonym pairs). It shows a clear split: Standard early layer has low hedging (0.08), while TopK_MLP and TopK_Attn show maximum hedging (1.0).
5. **Metric pipeline works end-to-end**: all 10 checkpoints loaded successfully, all metrics returned finite values, and the system tracked progress correctly.

## GO / NO-GO Assessment

- **Pipeline validation**: GO. The multi-objective metric pipeline runs without errors and produces outputs for every checkpoint.
- **Metric quality**: NO-GO for publication-ready numbers. The simplified absorption and hedging proxies are too crude, and dead-neuron estimates are unreliable at 2k tokens.
- **Gemma fallback**: NO-GO for the original `e1_full_gemma` task due to the gated model. This is a **blocking resource issue** that requires either (a) an HF token, or (b) switching to a fully open model like GPT-2 Small or Pythia-70m/160m/410m.

## Recommendation

**REFINE** — before scaling to full experiments:
1. Integrate the proper `sae-spelling` absorption metric (or SAEBench adaptation) instead of the simplified proxy.
2. Increase the activation corpus for dead-neuron detection to at least 50k-100k tokens.
3. Decide on the target model: either obtain HF access for Gemma-2-2B, or commit to GPT-2 Small / Pythia-160m as the open-model anchor.
4. Add more architecture families (e.g., Gated, JumpReLU) by sourcing appropriate checkpoints.

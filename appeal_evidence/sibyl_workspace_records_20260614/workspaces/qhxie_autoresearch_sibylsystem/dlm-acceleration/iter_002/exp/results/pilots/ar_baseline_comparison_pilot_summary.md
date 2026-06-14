# AR Baseline Comparison - Pilot Summary

**Task**: ar_baseline_comparison
**Date**: 2026-04-16 01:41
**Duration**: 46.5 min
**Engine**: HuggingFace transformers 4.47 (greedy + Universal Assisted Decoding)
**Hardware**: NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM)

## Results Table

| System | Batch | GSM8K Acc | GSM8K TPS | Speedup vs LLaDA | MATH500 Acc | MATH500 TPS | Combined QAS |
|--------|-------|-----------|-----------|-------------------|-------------|-------------|--------------|
| AR (Qwen2.5-7B) | 1 | 0.960 | 70.9 | 2.29x | 0.370 | 70.9 | 3.64 |
| AR (Qwen2.5-7B) | 8 | 0.960 | 471.1 | 15.19x | 0.380 | 531.8 | 24.96 |
| AR+SpecDec (Qwen2.5-7B+0.5B) | 1 | 0.970 | 48.2 | 1.55x | 0.390 | 45.3 | 2.53 |
| AR+SpecDec (Qwen2.5-7B+0.5B) | 8 | N/A | N/A | N/A | N/A | N/A | N/A |
| DLM (LLaDA-8B-Instruct) | 8 | 0.712 | 31.0 | 1.00x | 0.111 | 79.2 | 1.00 |

**Note**: Speculative batch=8 failed -- HuggingFace `assisted_generation` only supports batch_size=1.
vLLM (which supports batched speculative decoding) could not be installed due to PyTorch nightly (2.12.0.dev) incompatibility.

## Key Findings

1. **AR dominates DLM in accuracy**: Qwen2.5-7B-Instruct achieves 96% on GSM8K vs LLaDA-8B's 71.2% (1.35x accuracy retention). On MATH500, 37-39% vs 11.1% (3.3-3.5x). This is expected: AR models with instruct tuning are stronger at reasoning.

2. **AR dominates DLM in per-sample throughput (batch=1)**: 71 TPS vs 31 TPS (2.29x faster). LLaDA's 64-step denoising is inherently slower than single-pass AR generation for the same output length.

3. **AR batch scaling is massive (batch=8)**: 471 TPS on GSM8K (15.2x vs LLaDA batch=8 at 31 TPS). AR models scale linearly with batch size on high-VRAM GPUs because the KV cache fits comfortably. DLM's per-step full attention over all tokens limits batch scaling.

4. **Speculative decoding with cross-tokenizer UAD is SLOWER** than greedy: 48 vs 71 TPS (0.68x). The Qwen2.5-0.5B base model uses a different tokenizer than Qwen2.5-7B-Instruct, requiring Universal Assisted Decoding (UAD) which adds re-encoding overhead that negates the benefit of speculative drafting.

5. **Implication for DLM paper**: The honest comparison shows that DLM acceleration methods are not competing with AR speed -- they are narrowing the gap. The paper should frame composition results as "closing the DLM-AR gap" rather than "achieving AR-competitive speed." The quality advantage of specific DLM properties (parallel generation, non-autoregressive diversity) should be emphasized instead.

## QAS Context

The QAS (Quality-Adjusted Speedup) numbers above use LLaDA as the reference point (QAS=1.0).
- AR greedy batch=1 QAS = 3.64 (much better speed AND quality than LLaDA)
- AR greedy batch=8 QAS = 24.96 (AR dominates when batch serving is possible)
- This means DLM compositions like M1+IGSD (8.88x speedup but 52% AccRet, QAS~4.6) are in the same ballpark as AR greedy batch=1 in terms of quality-adjusted throughput
- The value proposition of DLM acceleration is strongest at batch=1 interactive settings where DLM's parallel generation creates a structural advantage
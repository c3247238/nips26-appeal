# AR Baseline Comparison (vLLM + HF)

**Date**: 2026-04-16 16:25
**Duration**: 55.5 min
**Engine**: vLLM 0.19.0 (greedy) + HF (speculative)
**Benchmarks**: 200 GSM8K + 100 MATH500

## Results

| System | Batch | Engine | GSM8K Acc | GSM8K TPS | vs LLaDA | MATH500 Acc | MATH500 TPS | Combined QAS |
|--------|-------|--------|-----------|-----------|----------|-------------|-------------|--------------|
| AR (Qwen7B) | 1 | vllm | 0.945 | 88.6 | 2.86x | 0.430 | 93.0 | 4.926 |
| AR (Qwen7B) | 8 | vllm | 0.950 | 513.2 | 16.55x | 0.400 | 624.0 | 28.144 |
| AR+SpecDec (Qwen7B) | 1 | hf_assisted | 0.940 | 51.2 | 1.65x | 0.370 | 50.7 | 2.596 |
| AR+SpecDec (Qwen7B) | 8 | none | 0.000 | 0.0 | 0.00x | 0.000 | 0.0 | 0.000 |
| DLM (LLaDA-8B) | 8 | custom | 0.712 | 31.0 | 1.00x | 0.111 | 79.2 | 1.000 |

## Notes

- LLaDA baseline: batch=8 (MDM parallel generation), 64 denoising steps
- AR batch=1 is the fair comparison for interactive use
- Speculative decoding: Qwen2.5-0.5B draft, HF Universal Assisted Decoding
- Speculative batch=8 unavailable: vocab size mismatch prevents vLLM spec dec
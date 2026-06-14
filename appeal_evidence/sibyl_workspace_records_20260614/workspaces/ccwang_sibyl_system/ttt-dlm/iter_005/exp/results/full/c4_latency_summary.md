# C4: Latency & Throughput Benchmarking (PILOT)

## Configuration
- **Model**: LLaDA-8B-Instruct (frozen)
- **NFE**: 128 denoising steps
- **Batch size**: 1
- **Samples**: 16 (GSM8K, seed 42)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (95GB VRAM)

## Results

| Method | Time/sample (s) | Step time (ms) | Overhead (%) | Peak Memory (GB) | Slowdown |
|--------|----------------|---------------|-------------|-------------------|----------|
| Vanilla | 6.77 | 52.88 | 0.0% | 15.38 | 1.000x |
| MetaState-GRU | 6.88 | 53.75 | 1.4% (GRU) | 18.67 | 1.017x |
| ReMDM | 6.80 | 53.11 | 0.3% (remask) | 18.38 | 1.004x |
| CoRe | 7.24 | 56.54 | 6.3% (remask) | 18.61 | 1.069x |
| DaL-Linear | 7.59 | 59.31 | 10.6% (TTT) | 23.27 | 1.122x |
| DaL-Linear+Phase | 7.42 | 57.95 | 8.5% (TTT) | 23.27 | 1.096x |

## Key Findings

1. **DaL-Linear adds 12.2% latency** (7.59s vs 6.77s) due to TTT gradient computation at each denoising step. TTT overhead accounts for 10.6% of total time.

2. **Phase-transition scheduling reduces overhead to 9.6%** by only executing TTT updates when mask ratio is in [0.15, 0.80] (100/128 steps active vs 128/128). Achieves 1.02x speedup over full DaL.

3. **MetaState-GRU is extremely lightweight** -- only 1.7% slowdown (1.4% GRU overhead). The GRU forward pass is much cheaper than TTT's gradient computation.

4. **ReMDM has negligible overhead** (0.4%) -- confidence-based remasking is computationally trivial since it reuses existing logits.

5. **CoRe has moderate overhead** (6.9%) due to per-position entropy computation and context-window disagreement scoring at each step.

6. **Memory overhead**: DaL adds ~7.9GB (for TTT layer + gradient buffers), MetaState-GRU adds ~3.3GB, ReMDM/CoRe add ~3GB (from softmax/entropy computation buffers).

## Pass Criteria
- **DaL-Full (with phase scheduling) overhead < 50% vs vanilla**: **PASS** (9.6% overhead, well within budget)

## Implications for Paper
- DaL's overhead is practical (~10%) but noticeably higher than MetaState-GRU (~1.7%)
- Phase scheduling provides modest speedup (2.4%) -- the cutoff zone [0.15, 0.80] still covers 78% of steps
- The compute-accuracy tradeoff (M5) is critical: DaL must demonstrate orthogonal accuracy gains to justify the 10% latency cost

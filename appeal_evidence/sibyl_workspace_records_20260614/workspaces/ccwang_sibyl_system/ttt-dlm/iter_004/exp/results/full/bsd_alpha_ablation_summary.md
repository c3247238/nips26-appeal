# BSD Alpha Schedule Ablation Summary — Countdown-16 (PILOT)

**Verdict: GO**

Fixed: k_frac=0.75, tau=linear(1.0→0.3)

| Alpha Schedule | Accuracy | rep-3 | distinct-3 | Entropy (start→end) | Time (s) |
|----------------|----------|-------|------------|---------------------|----------|
| Vanilla | 0.0% | 0.0786 | 0.8755 | — | 3.7 |
| linear(0.1→0.8) **BEST** | 6.2% | 0.0480 | 0.9132 | 3.50→0.00 | 3.8 |
| cosine(0.1→0.8) | 6.2% | 0.0480 | 0.9132 | 3.50→0.00 | 3.8 |
| constant(0.3) | 6.2% | 0.0480 | 0.9132 | 3.50→0.00 | 3.8 |
| constant(0.5) | 6.2% | 0.0669 | 0.8517 | 3.50→0.00 | 3.8 |

### Best Config
- **Alpha schedule: linear(0.1→0.8)**
- k_frac: 0.75
- Accuracy: 6.2%

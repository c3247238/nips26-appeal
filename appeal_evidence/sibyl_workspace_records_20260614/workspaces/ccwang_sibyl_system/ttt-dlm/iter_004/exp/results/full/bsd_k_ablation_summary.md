# BSD k-Parameter Ablation Summary — Countdown-16 (PILOT)

**Verdict: GO**

| Config | k_frac | Accuracy | rep-3 | distinct-3 | Entropy (start→end) | Time (s) |
|--------|--------|----------|-------|------------|---------------------|----------|
| Vanilla | — | 0.0% | 0.0786 | 0.8755 | — | 3.7 |
| k=T/4 (75% belief) | 0.25 | 0.0% | 0.0141 | 0.8992 | 3.50→0.00 | 3.8 |
| k=T/2 (50% belief) | 0.5 | 0.0% | 0.0197 | 0.9474 | 3.50→0.00 | 3.8 |
| k=3T/4 (25% belief) **BEST** | 0.75 | 6.2% | 0.0480 | 0.9132 | 3.50→0.00 | 3.8 |

### Best Config
- **k = 0.75** (k=3T/4 (25% belief))
- Accuracy: 6.2%

### Hypothesis H3 (intermediate k is optimal)
- **Not supported**: k=0.75 (k=3T/4 (25% belief)) achieves best accuracy

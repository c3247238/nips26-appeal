## 6. Conclusion

We present the first L0-matched cross-architecture comparison for SAE feature absorption. Our findings challenge the prevailing narrative:

1. **Architecture differences vanish at matched L0.** The apparent advantage of TopK and Matryoshka is primarily a sparsity effect.
2. **Absorption does not predict downstream interpretability.** The causal link hypothesized by prior work is not supported by our data.
3. **OrtSAE's orthogonality penalty has no effect.** This important negative result directly contradicts published claims.

Our methodological contribution is a framework for fair architecture comparison. Our empirical contribution is evidence that redirects community effort: rather than pursuing ever-more-complex architectures to reduce absorption, researchers should first understand whether absorption actually matters for their interpretability goals.

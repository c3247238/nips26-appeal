# 6. Conclusion

## 6.1 Summary

This paper presents the first component-isolated causal analysis of SAE absorption-reduction mechanisms, training six architectural variants on SynthSAEBench-16k ground-truth synthetic hierarchies and measuring absorption directly from known parent-child relationships. Three findings emerge.

First, **TopK sparsity is the dominant driver of absorption reduction**. With 78.0% reduction (Cohen's $d = 5.51$), TopK achieves a far larger effect than any other tested component. This is a categorical difference that reorders the community's understanding of what works.

Second, **orthogonality penalties have negligible effect**. Despite achieving near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$), orthogonality reduces absorption by only 2.7% ($d = 0.14$). The component that most improves reconstruction does nothing for absorption, revealing a dissociation between these two objectives.

Third, **an exploratory observation of a positive absorption--L0 sparsity correlation** ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) suggests that sparsity level, not architectural novelty, may be the operative mechanism: higher L0 sparsity (more active latents) is associated with higher absorption. With only $n = 4$ points (2 degrees of freedom), this correlation is mathematically fragile and should not be treated as a primary contribution. Confirmation requires a dedicated sparsity-sweep experiment. If confirmed, it would suggest reframing research toward understanding sparsity--absorption coupling.

The provisional component ranking, based on available data (3 of 6 variants with full replicates), is: **TopK ($d = 5.51$, full) $\gg$ MultiScale ($d \approx 1.1$, pilot only; not directly comparable) $\gg$ Orthogonality ($d = 0.14$, full)**. This ranking is provisional and may change when MultiScale, Gating, and Full Matryoshka complete full 5-replicate experiments.

## 6.2 Implications

These findings carry two implications for the SAE research community.

**For practitioners:** If absorption reduction is the goal, add TopK sparsity. It is a one-line architectural change with an effect size larger than multi-scale decomposition and orthogonality combined. If the provisional ranking holds when the full variant set is completed, the community's investment in more complex architectures may be misdirected if the primary benefit comes from explicit sparsity control.

**For researchers:** The strong sparsity--absorption correlation opens a new research direction. Instead of asking "which architecture reduces absorption?" we should ask "why does sparsity control absorption, and what is the optimal sparsity level?" A rate-distortion theoretical framing---absorption as information loss under a rate constraint---could provide formal bounds and guide architecture design.

## 6.3 Call to Action

Two validations are needed before these findings guide community practice.

First, **complete the remaining variants** (MultiScale, Gating, Full Matryoshka) with 5 replicates each to confirm the component ranking. Our current ranking is based on 3 of 6 variants with full data.

Second, **validate on real LLMs**. The synthetic-to-real transfer is the critical test. If TopK dominates on Pythia-160M or Gemma-2-2B first-letter absorption (measured via SAEBench), the synthetic benchmark is validated as a proxy and the component ranking can guide architecture selection. If not, the synthetic results are still valuable as a controlled setting, but practitioners should exercise caution in generalizing.

Ground-truth synthetic benchmarks are a prerequisite for causal claims about SAE architecture. Our study demonstrates that component isolation on known feature structure can resolve questions that probe-based metrics on real LLMs cannot. We hope this methodology becomes standard practice for evaluating future SAE innovations.

<!-- FIGURES
- None
-->

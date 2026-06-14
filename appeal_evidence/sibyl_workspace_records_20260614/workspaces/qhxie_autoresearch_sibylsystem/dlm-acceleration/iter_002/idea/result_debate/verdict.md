# Result Debate Verdict: ComposeAccel iter_002

**Date**: 2026-04-15 | **Score**: 4/10 | **Decision**: PROCEED (with reframing)

---

## Result Quality Score: 4/10

The dataset is impressively comprehensive in scope (15 experiment groups, factorial design, cross-model validation, AR comparison, batch sensitivity). However, execution has critical gaps: M1 (EntropyCache) provides no real speedup (kernel-level integration failed), M3 speedup may be a measurement artifact of output-length reduction, nearly all composition experiments are pilot-scale (100 samples, 1 seed), and the AR comparison used suboptimal HF Transformers instead of vLLM. Findings are directionally interesting but cannot support strong quantitative claims without remediation.

---

## Key Conclusion

**Training-free DLM acceleration methods predominantly interfere when composed, rather than synergizing.** This is a negative result that reframes the paper from "composition yields multiplicative speedup" to "composition reveals why DLM denoising resists modular acceleration."

Three interference regimes were identified:
- **Near-orthogonal** (M1+IGSD, ortho ~0.96): confounded by M1 being a functional no-op
- **Destructive** (M1+M3, ortho 0.41-0.52): net slowdown from overhead stacking
- **Partial** (M3+IGSD, ortho 0.61-0.84): step reduction degrades guidance context

The single standout finding is **M3 (AR-guided unmasking at gw=0.3)**, which improves both accuracy (+3.9% GSM8K) and potentially throughput -- but the speedup measurement needs clarification.

Accelerated DLMs remain far behind optimized AR inference (Qwen2.5-7B: 96% GSM8K at 71-471 TPS vs. best composed DLM: ~52% at ~96 TPS batch=1). Training-free acceleration does not close this gap.

---

## Action Plan

### PROCEED to paper writing immediately, with three mandatory prerequisites:

1. **Clarify M3 speedup** (1 GPU hour): Determine whether 1.68x is from faster per-token generation or shorter output lengths. This changes whether M3 is positioned as an acceleration method or a quality method.

2. **Run IGSD tau=0.0 ablation** (2 GPU hours): Validate that IGSD's KL-divergence mechanism adds value over naive step reduction. Without this, IGSD's algorithmic contribution is unsubstantiated.

3. **Full-scale top-3 compositions** (8-10 GPU hours): Replicate M1+IGSD, M3+IGSD, and top three-way configs on 1319 GSM8K with 3 seeds. Current pilot-scale results cannot support publication-quality claims.

### Paper narrative should center on:
1. **Interference taxonomy** as the primary methodological contribution
2. **M3 as quality enhancer** (pending speedup clarification) as the practical contribution
3. **Honest DLM-vs-AR positioning** as a service to the community

### Target venue: NeurIPS DLM Workshop (floor) to AAAI/EMNLP (ceiling with full remediation)

### Do NOT:
- Claim three-way synergy (ortho > 1.0 is noise; top configs have M3 disabled)
- Present M1 as an "acceleration axis" without kernel-level implementation
- Report code generation results (HumanEval 2.4%, MBPP 0% baselines are broken)
- Frame DLM acceleration as competitive with AR inference

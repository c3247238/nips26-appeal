# Verdict: Iteration 10 Pilot Results

## Executive Summary

**Score: 6.5 / 10** — Directionally compelling, methodologically incomplete.

**Verdict: PROCEED with structured go/no-go gates.**

The core finding that "sparsity reduces feature absorption" is supported by a large effect size (85.5% reduction), cross-architecture replication (TopK + MultiScale), and consistency with prior iteration results. However, multiple methodological gaps must be closed before the claim can be made with confidence.

---

## Key Conclusion

Sparse SAE architectures (TopK, MultiScale) achieve **80-85% lower absorption rates** than standard ReLU SAEs on synthetic data. The absorption rate metric successfully discriminates trained from random models, and sparse from non-sparse models. However, three critical questions remain unanswered:

1. **Is sparsity the true cause, or just correlated?** (Needs L0-matched Baseline control)
2. **Are dead neurons driving the effect mechanically?** (Needs active-neuron-only analysis)
3. **Does it generalize to real LLMs?** (Needs Gemma Scope validation)

---

## What We Know (High Confidence)

| Finding | Evidence |
|---------|----------|
| Absorption rate discriminates trained/random/sparse/non-sparse | Monotonic ordering: Random (0.452) > Baseline (0.226) > TopK (0.033) |
| TopK/MultiScale reduce absorption by 80-85% | Large effect size (d = 5.51); replicates across iterations |
| MCC is not useful here | Near-zero variation (~0.006) across all conditions including Random |
| Dead neurons are a major tradeoff | TopK: 83.2%, MultiScale: 58.4% |

## What We Don't Know (Must Resolve)

| Question | Resolution Experiment |
|----------|----------------------|
| Sparsity vs. architecture causality | L0-matched Baseline control |
| Dead neuron artifact | Active-neuron-only absorption analysis |
| Statistical robustness | 5-seed replication |
| Real-world generalizability | Gemma Scope 2B validation |

---

## Action Plan

### Immediate (Next Iteration)

| # | Task | Effort | Go/No-Go Criteria |
|---|------|--------|-------------------|
| 1 | 5-seed full experiment | ~45 min GPU | Mean effect > 50% reduction |
| 2 | L0-matched Baseline | ~3 min GPU | Confirms "sparsity" vs "architecture" story |
| 3 | Active-neuron-only analysis | No training | Effect persists > 60% without dead neurons |

### Following Iteration

| # | Task | Effort | Purpose |
|---|------|--------|---------|
| 4 | TopK k-sweep | ~18 min GPU | Dose-response curve (strongest novelty claim) |
| 5 | Gemma Scope validation | ~30 min GPU | External validity |

### Submission Targets

- **Workshop (ACL/EMNLP)**: High feasibility if Stage 1 succeeds
- **Conference (NeurIPS/ICML)**: Medium feasibility if Stage 2 succeeds

---

## Competitive Position

Our work fills a gap in the literature:
- **Anthropic** described superposition qualitatively; we provide a **quantitative, computable proxy** (absorption rate)
- **TopK SAE** was motivated by interpretability; we identify an **unreported benefit** (reduced absorption)
- The **dose-response curve** (absorption vs. k) would be genuinely novel

Risk: Reviewers may find "sparsity reduces co-activation" obvious. Counter: Emphasize (a) this was previously qualitative, we make it quantitative, (b) architectural independence, (c) the continuous dose-response relationship.

---

*Verdict rendered by Result Debate Synthesizer | Iteration 10 | 2026-04-26*

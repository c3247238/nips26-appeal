# Iteration 015 - PPL-Diversity Tradeoff Discovery

**Date**: 2026-03-07
**Focus**: Critical finding that PPL improvements are driven by text repetition

## Critical Discovery: PPL-Diversity Tradeoff

### The Problem
Examining generated texts revealed that ReMask-Retry's "PPL improvements" are substantially driven by **text degeneracy** (repetition of common tokens):

| Method | PPL (med) | Bigram Diversity | Degenerate |
|--------|----------|-----------------|------------|
| Vanilla | 4.193 | 0.479 | 45/256 (17.6%) |
| Fixed 70% | 3.569 | 0.388 | 86/256 (33.6%) |
| Adaptive | 2.200 | 0.260 | 182/256 (71.1%) |

The adaptive variant — our "best" result — produces degenerate text on 71% of prompts. AR models assign high probability to repetitive tokens ("the the the"), artificially lowering PPL.

### Filtered Analysis
When restricting to non-degenerate text pairs:
- Fixed 70%: Still -10.6% (p<0.0001) on 168 pairs, but degrades 43 prompts
- Adaptive: No improvement (+14.4%, p=0.19) on 64 pairs, degrades 147 prompts

### Temperature Ablation (64 prompts, 50% remask)
| Refinement Temp | PPL (med) | Diversity | Degenerate |
|----------------|----------|-----------|------------|
| Vanilla | 3.474 | 0.479 | 15/64 |
| 0.2 | 2.354 | 0.312 | 39/64 |
| 0.5 | 3.227 | 0.304 | 37/64 |
| 0.7 | 5.007 | 0.370 | 30/64 |
| 1.0 | 15.526 | 0.559 | 9/64 |

**No sweet spot**: Low temp → games PPL via repetition. High temp → restores diversity but worsens PPL.

### Multi-Model Validation
Qwen2.5-Coder-0.5B MDLM (64 prompts):
- GPT-2 eval: adaptive -29.7% PPL (p=0.007), but degeneracy increases (8→37/64)
- Same pattern: PPL improves but diversity degrades

### Root Cause
The 0.6B MDLM model lacks capacity to produce good text with partial context. When remasked tokens are re-denoised at low temperature, the model fills them with safe, repetitive, high-confidence predictions. This games PPL without improving actual text quality.

## Paper Updates
- Rewrote abstract to reflect PPL-diversity tradeoff as key finding
- Added Section 5.4: PPL-Diversity Tradeoff (temperature ablation, filtered analysis)
- Added Section 5.5: Multi-Model Evaluation (Qwen2.5-Coder results)
- Rewrote Limitations (6.3) to highlight diversity issue
- Rewrote Conclusion (7) to be honest about tradeoff

## Critic Review (iteration 014 paper)
Score: 6/10 (up from 5/10)
- Strengths: cross-family eval, statistical rigor, negative TTT results
- Remaining issues: single model (now addressed), no downstream eval, suspicious PPL (now acknowledged)

## Sibyl Pipeline Lessons
14. **Always co-report PPL and diversity** — PPL can be gamed by repetition
15. **Examine generated texts manually** — statistical metrics hide qualitative failures
16. **0.6B models too weak for post-hoc correction** — the model can't produce good text with partial context
17. **Negative findings are valid contributions** — the PPL-diversity tradeoff is arguably more interesting than the original "improvement"

## Code
- `aca_dllm_v9_multimodel.py`: Multi-model validation (Qwen2.5-Coder)
- `aca_dllm_v10_temperature.py`: Temperature ablation
- Results: `v9_*.json`, `v10_*.json`

## Next Steps
- Run LLM-as-judge evaluation to get quality scores beyond PPL
- Test on larger model if accessible (LLaDA-8B)
- Consider reframing paper as a study of PPL reliability for DLM evaluation

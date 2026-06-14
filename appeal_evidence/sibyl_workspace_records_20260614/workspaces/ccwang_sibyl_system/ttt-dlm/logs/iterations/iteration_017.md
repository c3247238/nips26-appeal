# Iteration 017 - LLaDA-8B Validation: Complete Scale Picture

**Date**: 2026-03-07
**Focus**: Testing ReMask-Retry on LLaDA-8B to determine scale dependence

## LLaDA-8B Results (32 prompts)
| Method | PPL (median) | Diversity | Degenerate |
|--------|-------------|-----------|------------|
| Vanilla | 16.671 | 0.976 | 0/32 |
| Retry 50% | 21.923 | 0.987 | 0/32 |
| Retry 30% | 21.928 | 0.981 | 0/32 |

### Key Findings
1. **No degeneracy at 8B scale** — confirms model capacity hypothesis
2. **PPL worsens (+31.5%)** — remasking produces different but not better text
3. **Diversity maintained** — large model avoids repetition trap

### Complete Scale Picture
| Scale | PPL Effect | Diversity | Quality |
|-------|-----------|-----------|---------|
| 0.6B | "Improves" (artifact) | Degrades | Worse |
| 8B | Worsens | Maintained | Unchanged |

**Conclusion**: ReMask-Retry does not improve text quality at ANY tested scale.

## Paper Updates
- Added LLaDA-8B results to Section 5.5 with complete scale comparison table
- Rewrote Conclusion with three clear findings: (1) method doesn't work, (2) PPL unreliable, (3) TTT/BoN also fail
- Updated Limitations to reflect 8B results

## Sibyl Pipeline Improvements (this iteration)
- Updated orchestrator's `_analyze_and_decide()` with proxy metric validation prompts
- Updated CriticAgent with "Proxy Metric Gaming" checklist item
- Updated ExperimentAgent with quality validation requirements
- All changes in `sibyl/orchestrator.py`, `sibyl/agents/supervisor.py`, `sibyl/agents/experiment.py`

## Overall Research Summary (iterations 010-017)
The research went through a complete arc:
1. **Initial success** (iter 010-013): ReMask-Retry shows -16.2% to -47.5% PPL improvement
2. **Cross-family validation** (iter 014): GPT-2 confirms PPL improvements
3. **Critical discovery** (iter 015): PPL improvements driven by text degeneracy
4. **Temperature ablation** (iter 015): No sweet spot exists at 0.6B
5. **LLM-as-judge** (iter 015): Confirms adaptive produces 100% degenerate text
6. **Scale test** (iter 017): LLaDA-8B avoids degeneracy but PPL worsens

The final finding is a clean negative result with clear methodology and honest reporting.
Paper suitable for EMNLP Findings or workshop submission.

## Code
- `aca_dllm_v11_llada.py`: LLaDA-8B experiments
- Results: `v11_llada_*.json`

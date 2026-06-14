# Iteration 011 - Token-Level Analysis + Best-of-N Complete + Supervisor Review

**Date**: 2026-03-07
**Focus**: Deepening analysis per supervisor recommendations

## Supervisor Review Key Points
1. **Priority order**: Token analysis > larger prompt set > downstream task > second model
2. **Best-of-N result**: Valuable as paper component, not standalone finding
3. **Second model**: Lower priority; deepen 0.6B analysis instead
4. **Venue target**: EMNLP 2026 (main or Findings) most realistic
5. **Methodology concern**: Ensure PPL is by external AR model (it is — Qwen3-0.6B AR)

## Completed Experiments

### Best-of-N Baseline (All 15 configs complete)
| Method | PPL | Δ% | Compute |
|--------|-----|-----|---------|
| vanilla | 3.820 | — | 1.0x |
| retry_30% | 3.518 | -7.9% | 1.28x |
| retry_70% | 3.203 | -16.2% | 1.28x |
| best_of_2 | 3.840 | +0.5% | 2.0x |
| best_of_3 | 4.086 | +6.9% | 3.0x |

**Key insight**: Model confidence anti-correlated with AR PPL at sequence level.

### Token-Level Analysis (v5, 2 seeds × 32 prompts)
1. **73.6% of tokens have confidence < 0.01** — model is guessing for most tokens
2. **92.8% of remasked tokens change** — retry is substantive, not confirmatory
3. **18.5x confidence improvement** at remasked positions (0.031 → 0.570)
4. **Function words 1.3x more likely remasked** — model commits to content first
5. **Slight early-sequence bias** (mean pos 0.480) — early tokens less confident

### Paper Updates
- Best-of-N results integrated with analysis of why it fails
- New Section 5.3: Token-Level Analysis with all findings above
- Sections renumbered accordingly

## Sibyl Pipeline Improvements
- **Resumable experiment pattern proven** — ran 15 configs across multiple SSH sessions
- **Supervisor agent** provides strategic guidance (venue, priority, methodology)
- **Background agent pattern works** — search agents run while main work continues

## Model Landscape Research
- Only 0.6B MDLM models available in dllm-hub
- Dream-7B available but different codebase
- LLaDA-8B too large for quick iteration
- dllm framework can convert AR → MDLM but requires training
- **Decision: deepen 0.6B analysis per supervisor recommendation**

## Next Steps (Priority Order)
1. Scale to 200+ prompts from standard corpus (WikiText-103 or C4 prefixes)
2. One downstream task metric (cloze accuracy or coherence)
3. Polish paper for EMNLP 2026 submission

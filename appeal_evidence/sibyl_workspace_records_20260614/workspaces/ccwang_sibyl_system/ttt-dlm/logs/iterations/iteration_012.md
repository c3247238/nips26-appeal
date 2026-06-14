# Iteration 012 - 256-Prompt Validation + Pipeline Learnings

**Date**: 2026-03-07
**Focus**: Scaling prompt set from 32 to 256, validating generalizability

## 256-Prompt Experiment

### Lesson 1: Chat Template Required
First attempt with WikiText-103 prefixes produced garbage (PPL=35K-254K).
The 0.6B MDLM model was fine-tuned with chat template format.
Raw text prompts → degenerate output. Must use `tokenizer.apply_chat_template()`.

### Lesson 2: Batch-Resumable Pattern Essential
Created `v6c_largeset.py` with batch processing (64 prompts per invocation).
Each batch completes within SSH timeout. Progress saved incrementally.
This pattern should be standard for all Sibyl pipeline experiments.

### Results (256 diverse QA prompts, 1 seed)
| Metric | Vanilla | Retry 70% |
|--------|---------|-----------|
| Mean PPL (252 safe) | 4.723 | 4.057 (-14.1%, p=2.65e-6) |
| Median PPL (all 256) | 4.193 | 3.569 (-14.9%, Wilcoxon p<1e-10) |
| Win rate | — | 72.2% (182/252) |
| Catastrophic failures | 0 | 4/256 (1.6%) |

**Key finding**: Results generalize from 32 to 256 prompts.
The -14.1% improvement is consistent with -16.2% from the ablation set.

### Catastrophic Failures Analysis
4/256 prompts (1.6%) have retry PPL > 100K.
These prompts had low vanilla PPL (2.76), meaning initial generation was already good.
70% remasking destroyed good content on these easy prompts.

**Implication**: An adaptive remask ratio (lower for high-confidence sequences)
could eliminate catastrophic failures while preserving gains.

## Paper Updates
- Setup section updated: 256 prompts, chat template, Wilcoxon test
- New Section 4.4: Large-Scale Validation with full results table
- Sections renumbered

## Sibyl Pipeline Improvements Codified
1. **Always use chat template** for model-specific prompt formatting
2. **Batch-resumable experiments** as standard pattern (64 prompts/batch)
3. **Test prompt quality first** with 3-5 samples before scaling up
4. **Use median + Wilcoxon** for robustness to outliers alongside mean + t-test

## Remaining Items
- [ ] Adaptive remask ratio (lower for high-confidence sequences)
- [ ] One downstream task metric
- [ ] Final paper polish for EMNLP 2026

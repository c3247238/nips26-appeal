# Iteration 016 - Final Critic Review & Pipeline Improvements

**Date**: 2026-03-07
**Focus**: Final review, Sibyl pipeline improvements

## Critic Review (final)
Score: 5/10 (borderline reject)

### Key Feedback
1. **Paper is structurally confused** — Sections 1-4 are a methods paper, Section 5.4 reverses the narrative. Needs to commit to one framing.
2. **No evidence method works at ANY scale** — the central limitation. Need LLaDA-8B or similar.
3. **Cross-family PPL validation contradicts degeneracy finding** — paper says GPT-2 confirms "genuinely higher quality" then later says PPL is unreliable.
4. **LLM-as-judge too small (32 samples) with weak judge (1.5B)** — need stronger model or human eval.
5. **Metric-critique framing needs engagement with prior work** (Holtzman et al. 2020 on neural text degeneration).
6. **Could work as Findings/workshop paper** with current content.

### Viable Paths Forward
1. **Metric-critique paper** (EMNLP Findings): Restructure around "when PPL fails as a DLM quality metric." Lead with degeneracy finding, make method description brief. Add human eval.
2. **Methods paper** (needs work): Show method works on LLaDA-8B where model quality is sufficient.
3. **Workshop paper** (ACL/EMNLP workshop): Current content sufficient.

## Sibyl Pipeline Improvements Made

### 1. Proxy Metric Validation in Analysis
Updated `_analyze_and_decide()` in `orchestrator.py` to explicitly prompt for:
- Checking if proxy metric improvements correspond to genuine quality
- Flagging suspiciously large improvements (>30%)
- Verifying with secondary metrics

### 2. Critic Agent Proxy Gaming Check
Added "Proxy Metric Gaming" as item #3 in CriticAgent's review checklist:
- Check for degenerate outputs gaming metrics
- Verify with secondary metrics
- Flag >30% improvements as suspicious

### 3. Experiment Agent Quality Validation
Updated ExperimentAgent prompt to require:
- Co-reporting PPL and diversity metrics
- Qualitative inspection of example outputs
- Early validation on 8-16 examples before scaling up
- Storing generated texts (not just aggregate metrics)
- Batch-resumable experiment design

## Key Sibyl Pipeline Lessons
- **Proxy metrics can be gamed** — pipeline must validate improvements with multiple metric types
- **Early quality checks save time** — inspect outputs after 8-16 examples, not 256
- **Paper framing matters** — the pipeline should detect structural confusion (methods paper that proves method doesn't work) earlier
- **Negative findings need different framing** — pivot writing style when main hypothesis is disproven

## Experiment Summary (All Iterations)
| Iter | Focus | Key Finding |
|------|-------|-------------|
| 010 | Paper rewrite, Best-of-N started | Related work expanded |
| 011 | Token analysis, Best-of-N complete | 92.8% change rate, BoN ineffective |
| 012 | 256-prompt validation | Chat template required, -14.9% PPL |
| 013 | Adaptive remask ratio | -47.5% median PPL, 0 catastrophic |
| 014 | Cross-family eval | GPT-2 confirms PPL improvements |
| 015 | PPL-diversity tradeoff | PPL improvements driven by repetition! |
| 016 | Final review, pipeline improvements | 5/10, need larger model or full reframe |

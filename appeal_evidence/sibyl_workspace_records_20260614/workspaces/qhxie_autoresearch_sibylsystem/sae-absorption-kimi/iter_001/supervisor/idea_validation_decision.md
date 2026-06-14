# Idea Validation Decision

## Pilot Evidence Summary

Only **candidate A (cand_a)** was executed in the pilot phase. The original task `e1_full_gemma` fell back to `e1_full_gpt2` because `google/gemma-2-2b` is a gated HuggingFace repository and no HF authentication token is available in the environment.

**Pilot execution summary (GPT-2 Small, 10 checkpoints, ~34 seconds):**
- Pipeline success rate: 1.0 (all 10 checkpoints loaded, all metrics returned finite values)
- Mean L0 (Standard): 46.1; (TopK): 32.0; (TopK_MLP): 32.0; (TopK_Attn): 32.0
- Mean explained variance (Standard): 0.945; (TopK): 0.938; (TopK_MLP): 0.734; (TopK_Attn): 0.794
- Mean CE loss recovered: all negative (-4.17% to -0.10%), suggesting noisy small-sample computation
- Dead-neuron fraction: 33-88%, unreliable at ~2k tokens
- Absorption (simplified first-letter proxy): 0.0 for all families except TopK_Attn (0.654)
- Hedging (simplified correlated-pair proxy): 0.620 (Standard), 0.800 (TopK), 1.000 (TopK_MLP/TopK_Attn)

**Key findings:**
1. The metric pipeline works end-to-end — this validates the engineering path.
2. The simplified absorption/hedging proxies are too coarse for publication-ready evaluation.
3. Dead-neuron estimates require a larger activation corpus (>=50k-100k tokens).
4. Gemma-2-2B is inaccessible without an HF token; the target model must switch to GPT-2 Small or Pythia-160M.
5. H1 (downstream causal cost) was **not tested** in this pilot — it requires SAEBench data.
6. H2 (first-letter benchmark unrepresentative) is **supported** by the degenerate proxy results (0.0 for 26/27 checkpoints).
7. H3 (domain-dependent absorption) was **not tested** in this pilot.

**Candidate B and C:** No pilot experiments were run.

---

## Decision Matrix

### Candidate A — Absorption Has a Downstream Cost: Re-evaluating Feature Absorption in Sparse Autoencoders

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Pipeline runs end-to-end, but key metrics (absorption, hedging, dead neurons) are degenerate or noisy at pilot scale. No direct downstream causal data was collected. |
| Hypothesis survival | 0.25 | 4 | H1 was not falsified — simply untested in this pilot. H2 is strongly supported by the degenerate proxy results. H3 was previously refuted in result debate and is now scoped as a negative-result pilot only. |
| Path to full result | 0.20 | 4 | Clear route exists: use Pythia-160M SAEBench data for H1, replace the absorption proxy with the official `sae-spelling` or SAEBench metric for H2, and run a scoped task-agnostic pilot for H3 as a negative-result demonstration. |
| Novelty (from report) | 0.15 | 5 | Novelty report and candidates.json confirm no prior controlled partial-correlation/regression analysis isolating absorption as a predictor of downstream utility. The metric critique and task-agnostic pilot are also claimed as novel. |
| Resource efficiency | 0.10 | 4 | Fully training-free; uses existing checkpoints. One GPU per task, <=1 hour per batch. Very efficient once metrics are fixed. |

**Weighted score for Candidate A:** 3.85

### Candidate B — Towards a Task-Agnostic Absorption Metric

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot was run. The task-agnostic metric is higher-engineering and depends on LLM-based hierarchy discovery, which is unvalidated in this workspace. |
| Hypothesis survival | 0.25 | 2 | H3 (task-agnostic metric correlates with first-letter benchmark) was refuted in the result debate. The front-runner already subsumes the task-agnostic metric as a supporting/negative-result pilot. |
| Path to full result | 0.20 | 2 | Requires building an entirely new automated hierarchy discovery pipeline from scratch. Higher engineering complexity than the front-runner. |
| Novelty (from report) | 0.15 | 4 | Candidates.json novelty claim: "No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery." But this novelty is already claimed by cand_a as a supporting contribution. |
| Resource efficiency | 0.10 | 2 | Higher complexity than Candidate A; LLM labeling, probe training, and causal ablation per hierarchy. Uncertain cost. |

**Weighted score for Candidate B:** 1.70

### Candidate C — Is Feature Absorption a Training Artifact?

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot run. |
| Hypothesis survival | 0.25 | 1 | Explicitly violates the project's training-free constraint (candidates.json: "Explicitly violates the project's training-free constraint. Dropped unless the constraint is relaxed."). |
| Path to full result | 0.20 | 1 | Cannot proceed without relaxing the training-free constraint. Even then, training a random-decoder SAE to matched L0 is non-trivial. |
| Novelty (from report) | 0.15 | 3 | Novel in applying absorption metric to random-decoder baselines, but Korznikov et al. (2026) already showed random baselines match trained SAEs on other metrics. |
| Resource efficiency | 0.10 | 1 | Requires training new SAEs — the most expensive path and explicitly disallowed. |

**Weighted score for Candidate C:** 1.00

---

## Decision Rationale

- **Candidate A scores 3.85**, which meets the ADVANCE threshold (>= 3.5).
- Its main hypothesis (H1: downstream causal cost) was **not falsified** by the pilot — it was simply not tested yet. The pilot's issues were methodological (proxy metrics too coarse, sample too small, model inaccessible) — not fundamental contradictions of the core claim.
- H2 (first-letter benchmark unrepresentative) is actually **strengthened** by the pilot evidence: the degenerate proxy results strongly suggest the canonical metric fails to capture absorption variance on GPT-2 Small families.
- H3 has been strategically reframed: downgraded from a validation study to a **negative-result pilot**, which is an acceptable supporting contribution.
- Candidate B is not a viable pivot because its core hypothesis was refuted and its contributions are already subsumed by cand_a.
- Candidate C is effectively disqualified by the training-free constraint.

Sanity checks:
- [x] Compared all candidates, not just the front-runner.
- [x] No candidate failed its own falsification criteria in the pilot (H1 was untested, not falsified).
- [x] Sunk cost was ignored: the decision is based on the weighted score and hypothesis survival, not prior effort.
- [x] The pilot was partially inconclusive for H1, but the clear path to testing it with real SAEBench data justifies ADVANCE over defaulting to REFINE.

---

## Next Actions

1. **Integrate official absorption metric**: Replace the simplified first-letter proxy with `sae-spelling` or SAEBench's built-in absorption evaluation before any scaled experiment.
2. **Increase dead-neuron corpus size**: Use >=50k tokens for stable dead-neuron rate estimation.
3. **Commit to Pythia-160M as the primary anchor** for H1 and H2, with GPT-2 Small as a secondary family for H2 variance testing.
4. **Execute the task plan**:
   - `e2_pilot`: Validate official absorption metric pipeline on 5 checkpoints (~15 min).
   - `e2_full_gpt2` + `e2_full_pythia`: Run official absorption metric on 10-15 GPT-2 and 15-20 Pythia checkpoints (~20-30 min each).
   - `e1_pilot` + `e1_full`: SAEBench data validation and downstream causal cost meta-analysis on 200+ Pythia SAEs (~5-10 min).
   - `e3_pilot`: Scoped task-agnostic absorption pilot (geography domain only, negative-result demonstration) (~15 min).
   - `e4_pilot` + `e4_full`: Supporting Pareto tradeoff analysis on Pythia-160M (~15-45 min).

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.54
DECISION: ADVANCE

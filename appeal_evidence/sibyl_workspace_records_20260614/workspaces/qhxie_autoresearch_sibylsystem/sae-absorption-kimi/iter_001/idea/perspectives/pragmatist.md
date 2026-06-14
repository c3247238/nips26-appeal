# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **SAELens** (`jbloomAus/SAELens`, MIT, 1,100+ stars) — The dominant library for loading and analyzing pretrained SAEs. Integrates with TransformerLens and Neuronpedia. Supports Standard, Gated, TopK, and JumpReLU architectures. **Critical for training-free analysis.**

2. **SAEBench** (`adamkarvonen/SAEBench`, open source, ICML 2025) — Industry-standard benchmark with 8 evaluations including feature absorption. Released **200+ pretrained SAEs** on Pythia-160M and Gemma-2-2B (7 architectures: ReLU, TopK, BatchTopK, JumpReLU, Gated, P-anneal, Matryoshka BatchTopK). HuggingFace checkpoints are publicly downloadable. **Essential for multi-objective evaluation.**

3. **sae-spelling** (`lasr-spelling/sae-spelling`, MIT) — Official code for "A is for Absorption" (Chanin et al., 2024). Contains `FeatureAbsorptionCalculator` and the first-letter absorption experiment pipeline. **The canonical absorption metric implementation.**

4. **OrtSAE** (Korznikov et al., 2025, arXiv:2509.22033) — Orthogonal Sparse Autoencoders. Claims 65% absorption reduction via chunk-wise orthogonality penalty. **No public GitHub repo yet** — authors state code will be released "upon acceptance." Builds on BatchTopK.

5. **Matryoshka SAE** (`bartbussmann/matryoshka_sae`, open source) — Original implementation of Matryoshka SAEs. Also `noanabeshima/matryoshka-saes` for efficient training. **Available and runnable.**

6. **BatchTopK SAE** (`bartbussmann/BatchTopK`, open source) — Base architecture for both Matryoshka and OrtSAE. Clean, SAELens-inspired codebase. **Good fallback if OrtSAE code is unavailable.**

7. **Gemma Scope** — Pretrained SAEs for Gemma-2-2B, widely used in absorption studies. Accessible via SAELens `from_pretrained`. **Note: `google/gemma-2-2b` is a gated HuggingFace repo** — requires HF token for model download, even though SAEs themselves are public.

8. **Llama Scope** (`fnlp/Llama-Scope`, HuggingFace) — 256 SAEs on Llama-3.1-8B-Base (32K and 128K features). **Fully open, no gating.**

9. **Pythia-160M SAEBench SAEs** (`adamkarvonen/saebench_pythia-160m-deduped_width-*`, HuggingFace) — Fully open, no gating. 4k/16k/65k widths, 7 architectures. **Best open-model anchor for training-free analysis.**

10. **Feature Hedging** (Chanin et al., 2025, arXiv:2505.11756) — Complementary failure mode to absorption. Shows narrower SAEs reduce absorption but increase hedging. **Key for multi-objective framing.**

11. **"Sanity Checks for SAEs"** (Korznikov et al., 2026, arXiv:2602.14111) — Random-decoder and frozen-decoder baselines match trained SAEs on AutoInterp, sparse probing, and RAVEL. **Did NOT measure absorption on these baselines.**

12. **"A Unified Theory of Sparse Dictionary Learning"** (Tang et al., 2025, arXiv:2512.05534) — Theoretical framework explaining absorption via spurious minima. Proposes "feature anchoring." **Theory-heavy, limited empirical validation on real LLMs.**

### Landscape Summary

The field has moved fast. By 2025, feature absorption is recognized as a central SAE pathology, with multiple architectural mitigations (OrtSAE, Matryoshka, masked regularization) and a dominant benchmark (SAEBench). The key practical gap is **not** proving absorption exists — that is settled. The gap is understanding whether absorption-mitigation methods genuinely improve SAEs overall or merely shift pathologies around. No prior work has conducted a systematic, training-free, multi-objective Pareto evaluation across existing checkpoints using the full SAEBench metric suite.

From an engineering standpoint, the tooling is mature:
- SAELens loads pretrained SAEs in ~5 lines of code.
- SAEBench provides 200+ checkpoints with standardized hyperparameters.
- `sae-spelling` provides the canonical absorption metric implementation.

The main engineering constraint is **model access**: Gemma-2-2B is gated, which blocks the original pilot plan. Pythia-160M and GPT-2 Small are fully open alternatives. GPT-2 Small has fewer architecture families available, while Pythia-160M has the full SAEBench suite.

---

## Phase 2: Initial Candidates

### Candidate A: Multi-Objective Pareto Evaluation of Absorption-Mitigation Methods (Front-Runner)

- **Core hypothesis**: Absorption-mitigation architectures (OrtSAE, Matryoshka, JumpReLU, BatchTopK, Gated) do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction, dead-neuron rate, and downstream probing performance. Each occupies a distinct tradeoff region.
- **Implementation sketch**: Assemble 50-100 pretrained SAEBench checkpoints per model (Pythia-160M). Compute absorption (via SAEBench or `sae-spelling` integration), hedging, L0, explained variance, CE loss recovered, dead-neuron fraction, sparse probing F1, and RAVEL Cause/Isolation. Normalize metrics, compute empirical Pareto fronts per architecture family, and test stochastic dominance with Mann-Whitney U tests.
- **Simplest possible version**: Run SAEBench's built-in `feature_absorption` eval on 5-10 Pythia-160M checkpoints (1 per architecture family) alongside L0 and reconstruction metrics. ~15 minutes.
- **Time estimate**: ~45-60 minutes per full checkpoint batch on a single A100/RTX 4090. Highly parallelizable across GPUs.
- **Reusable components**: SAEBench evaluation pipeline, SAELens checkpoint loading, Pythia-160M HuggingFace model.

### Candidate B: Task-Agnostic Absorption Metric Pilot

- **Core hypothesis**: A task-agnostic absorption metric, built by combining automated hierarchical concept discovery with causal ablation, will correlate moderately-to-strongly (r > 0.4) with the first-letter benchmark while enabling absorption measurement across arbitrary semantic domains.
- **Implementation sketch**: Use an LLM judge to discover parent-child hierarchies (geography, biology, colors) for a pretrained SAE. Train logistic regression probes on residual-stream activations. Apply the `sae-spelling` ablation framework to detect absorption on these new hierarchies. Correlate with SAEBench first-letter scores across 20-50 SAEs.
- **Simplest possible version**: 1 SAE (e.g., Llama Scope layer 12, 32K) × 1 domain (geography: continent → country) × 5-10 parent-child pairs. ~15 minutes.
- **Time estimate**: ~30-45 minutes for a pilot with 1 SAE × 1 domain. Scaling to 20-50 SAEs is mostly batchable.
- **Reusable components**: `sae-spelling` ablation logic, SAELens, LLM API for hierarchy labeling.

### Candidate C: Random-Decoder Baseline for Absorption (Conditional)

- **Core hypothesis**: Feature absorption is primarily a geometric consequence of sparse dictionary learning on hierarchical data, not a training-dynamics pathology. Randomly initialized, frozen-decoder SAEs matched for sparsity will exhibit absorption rates comparable to trained SAEs.
- **Implementation sketch**: Load a trained SAE baseline (`gpt2-small-res-jb`). Construct a random-decoder SAE with the same decoder weights frozen, train only the encoder to matched L0 using TopK. Run `sae-spelling` absorption metric on both.
- **Simplest possible version**: GPT-2-small, layer 8. One trained SAE vs. one random-decoder SAE. ~15 minutes GPU time for encoder training.
- **Time estimate**: ~30 minutes total (training + evaluation).
- **Reusable components**: SAELens training pipeline, `sae-spelling` metric.
- **Critical constraint**: **Violates the project spec's training-free mandate.** Can only proceed if that constraint is explicitly relaxed.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Implementation reality check**: The pilot (`e1_full_gpt2`) already proved the pipeline works end-to-end. All 10 checkpoints loaded successfully, all metrics returned finite values. The main issue was proxy metric quality, not engineering feasibility. SAEBench has a proper absorption eval that replaces the crude proxy.
- **Reproducibility attack**: SAEBench checkpoints are trained with standardized hyperparameters on the same corpus. Metrics are well-documented. The main fragility is the absorption metric itself, which requires integrated-gradients ablation and is sensitive to prompt formatting. But this is the *canonical* metric — if we use it, we are directly comparable to prior work.
- **Baseline sanity check**: We are not proposing a new method; we are evaluating existing methods against each other. The "baseline" is the Standard/ReLU SAE family. This is fair because all families are evaluated on the same checkpoints with the same metrics.
- **Scope attack**: The Pareto-front claim is general across architectures on a given model. It does not claim cross-model generalization. The scope is well-defined. If results hold on Pythia-160M, they provide strong evidence for the broader claim.
- **Verdict**: **STRONG**

### Against Candidate B

- **Implementation reality check**: No pilot has been run. The LLM-based hierarchy discovery is unvalidated. There is a real risk of noisy or hallucinated hierarchies, which would poison the absorption metric. The `sae-spelling` ablation logic is solid, but the upstream concept discovery is a research question in itself.
- **Reproducibility attack**: LLM judges are non-deterministic and vary by model version. Reproducing the exact hierarchies across runs is hard. Consensus prompting or temperature=0 helps but does not fully solve it.
- **Baseline sanity check**: The baseline is the first-letter absorption metric. If correlation is weak, the paper pivots to analyzing *why* — which is still a valid negative result. But weak correlation could also mean the task-agnostic metric is simply broken.
- **Scope attack**: Even if it works for geography/biology/colors, these are still toy domains. True task-agnosticness would require validation on dozens of domains. The pilot scope is necessarily narrow.
- **Verdict**: **MODERATE** — high novelty, higher engineering risk.

### Against Candidate C

- **Implementation reality check**: Random-decoder SAEs are known to work for standard metrics (Korznikov et al., 2026). Training the encoder to matched L0 is straightforward with TopK. The absorption metric pipeline is proven. Engineering is feasible.
- **Reproducibility attack**: The result depends heavily on the random seed for decoder initialization. Multiple seeds should be run. Also, "comparable absorption" needs a pre-registered threshold (e.g., within 20% relative).
- **Baseline sanity check**: The trained SAE baseline is well-established. Comparison is fair.
- **Scope attack**: Single model (GPT-2-small), single layer. Generalization to larger models is unclear. But the claim is about the *mechanism* of absorption, not model-specific performance.
- **Verdict**: **MODERATE** — conceptually clean, but **disqualified by the training-free constraint** unless the spec changes.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Random-Decoder Baseline)** is dropped from active consideration because it violates the project's explicit training-free constraint. It remains a theoretically interesting backup if the constraint is relaxed.

### Strengthened Survivors

- **Candidate A** is strengthened by anchoring on **Pythia-160M** instead of Gemma-2-2B. The SAEBench suite provides 7 architecture families on Pythia-160M with full HuggingFace availability and no gating. This removes the HF-token blocker entirely.
- The metric pipeline should use **SAEBench's built-in absorption evaluation** (which adapts the Chanin et al. metric) rather than a custom simplified proxy. This ensures publication-ready numbers and direct comparability.
- For hedging, SAEBench does not currently include a hedging metric. We can either: (a) integrate the hedging metric from Chanin et al. (2025) manually, or (b) replace hedging with **RAVEL Cause/Isolation** and **Spurious Correlation Removal (SCR)** from SAEBench, which capture related disentanglement pathologies. Option (b) is simpler and keeps the pipeline within SAEBench.

### Selected Front-Runner

**Candidate A: Multi-Objective Pareto Evaluation on Pythia-160M SAEBench SAEs**

This is the highest-probability path. It is fully training-free, uses openly available checkpoints, leverages mature tooling, and addresses a clear, novel research gap.

---

## Phase 5: Final Proposal

### Title
**The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods**

### Hypothesis
Absorption-mitigation architectures (OrtSAE, Matryoshka, JumpReLU, BatchTopK, Gated, P-anneal) do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, reconstruction fidelity, dead-neuron rate, and downstream interpretability metrics (sparse probing, RAVEL, SCR, TPP). Instead, each architecture family occupies a distinct tradeoff region.

### Motivation
The SAE community has produced a flurry of absorption-mitigation methods, each reporting impressive single-metric improvements. But SAE quality is inherently multi-objective: sparsity, reconstruction, interpretability, and utility must all be balanced. No prior work has rigorously tested whether these "fixes" genuinely improve the overall Pareto front or merely shift pathologies. This paper conducts the first systematic, training-free, multi-objective evaluation using existing, publicly available checkpoints.

### Method

**Step 1: Corpus Assembly**
- Download all SAEBench baseline SAEs for **Pythia-160M-deduped** (layer 8) from HuggingFace:
  - 7 architectures: ReLU, TopK, BatchTopK, JumpReLU, Gated, P-anneal, Matryoshka BatchTopK
  - 3 widths: 4k, 16k, 65k
  - Multiple sparsities per width (~20 to ~640 L0)
- Total checkpoints: ~50-100 per model.

**Step 2: Metric Computation**
For each checkpoint, compute:
- **Absorption**: SAEBench `feature_absorption` metric (adapted from Chanin et al.)
- **Reconstruction**: L0, explained variance, CE loss recovered
- **Dead neurons**: Fraction of latents with near-zero activation frequency
- **Downstream utility**: Sparse probing F1, RAVEL Cause/Isolation, SCR, TPP

All metrics are computed via SAEBench's existing evaluation scripts, ensuring standardization.

**Step 3: Pareto Analysis**
- Normalize each metric to [0, 1] within model family.
- Compute empirical Pareto fronts per architecture family.
- Test stochastic dominance using Mann-Whitney U tests across the full metric suite.
- Report pairwise tradeoff curves (absorption vs. reconstruction, absorption vs. RAVEL, etc.).

**Step 4: Statistical Controls**
- Include dictionary width and L0 as covariates.
- Use cluster-robust standard errors by architecture family.
- Report bootstrap confidence intervals for Pareto front boundaries.

### Simplest Version
Run SAEBench's built-in `feature_absorption` + `reconstruction` + `sparse_probing` evaluations on **5 checkpoints** (1 per architecture family) for Pythia-160M. This validates the metric pipeline and gives early signal on variance. Expected duration: **10-15 minutes**.

### Baselines
1. **Standard/ReLU SAE** — The original Anthropic architecture. Expected to have moderate absorption, strong reconstruction, moderate downstream utility.
2. **Matryoshka BatchTopK** — Expected to have low absorption, slightly weaker reconstruction, strong RAVEL/SCR scores.

### Experimental Plan

| Stage | Task | Checkpoints | Metrics | Duration |
|-------|------|-------------|---------|----------|
| Pilot | Pipeline validation | 5 (1 per family) | Absorption, L0, EV, sparse probing | 10-15 min |
| Full | Pareto evaluation | 50-100 (all Pythia-160M) | Full SAEBench suite | ~45-60 min per batch |
| Analysis | Pareto front + stats | All results | Dominance tests, tradeoff curves | 15 min (CPU) |

### Resource Estimate
- **Model**: Pythia-160M-deduped (160M parameters, easily fits on a single GPU)
- **SAEs**: 50-100 checkpoints, all training-free
- **Compute**: Single RTX 4090 / A100. Each checkpoint evaluation takes 5-15 minutes depending on metric. Total wall-clock: ~10-20 hours, easily parallelizable across GPUs. Each independent subtask is ≤1 hour.
- **Storage**: ~5-10 GB for all checkpoints.

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OrtSAE code is unavailable; cannot evaluate OrtSAE directly | High | SAEBench does not include OrtSAE yet. We focus on the 7 architectures SAEBench *does* provide. If OrtSAE code is released during the project, we can add it. The core claim does not depend on any single architecture. |
| SAEBench absorption metric is slow (IG ablation per token) | Medium | The metric is batched and GPU-accelerated. For 50-100 checkpoints, total runtime is still within the project's time budget. We can also subsample tokens if needed. |
| One architecture actually dominates the Pareto front | Medium | The paper still provides the first rigorous multi-objective validation — a valuable contribution, though less contrarian. |
| Pythia-160M results do not generalize to larger models | Low-Medium | We explicitly scope the claim to Pythia-160M. Generalization is framed as future work. If time permits, we can add GPT-2 Small or Llama-3.1-8B as a secondary model. |

### Novelty Claim
No prior work has conducted a **systematic, training-free, multi-objective Pareto evaluation** of absorption-mitigation methods across the full suite of SAE quality metrics using existing, publicly available checkpoints. Prior comparisons (OrtSAE, Matryoshka) are pairwise and single-metric. We reframe the research question from "which architecture fixes absorption?" to "what tradeoffs does each architecture make?"

### Engineering Feasibility Score: 9/10

All required tools are open-source and mature:
- `pip install sae-lens sae-bench`
- Checkpoints download automatically from HuggingFace.
- SAEBench evaluation scripts are well-documented.
- No training required.
- No gated models needed (Pythia-160M is fully open).

The only minor uncertainty is whether SAEBench's `feature_absorption` eval runs out-of-the-box on all 7 architecture families. The pilot is designed to catch this.

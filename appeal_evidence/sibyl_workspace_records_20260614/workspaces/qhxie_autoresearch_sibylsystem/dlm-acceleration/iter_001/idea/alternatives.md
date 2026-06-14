# Backup Ideas for Pivot

*Last updated: 2026-04-14 (post result-debate, iter_001)*

---

## Backup Candidate B: Consistency Distillation for Masked Diffusion Language Models

### Status: Backup (unchanged from iter_001)

### Core Idea

Adapt consistency-style distillation to LLaDA-8B — train a lightweight adapter (~50M params) on frozen LLaDA-8B to map any intermediate masked state to the final clean output in 1-4 inference steps. Achieves 15-20x step reduction with < 3% quality drop on reasoning benchmarks.

### Current Competitive Landscape (as of April 2026)

- T3D (arXiv:2602.12262): trajectory self-distillation for DLLMs, reduces steps but not to 1-4, different objective (DDO), smaller models
- FS-DFM (arXiv:2509.20624): discrete flow-matching model with consistency objective, perplexity benchmarks only, not instruction-tuned 8B
- **Gap**: No paper has demonstrated 1-4 step inference for a LLaDA-8B-scale instruction-tuned MDM on reasoning/coding benchmarks with < 3% quality drop

### Why Not Front-Runner

1. Requires training (~2-4 GPU-days for adapter)
2. Discrete MDM consistency trajectory matching is harder than continuous diffusion (stochastic masking, not deterministic ODE)
3. Risk of being scooped between now and submission
4. Front-runner is training-free — zero-compute advantage

### Pivot Trigger

- Full-scale M1+IGSD Ortho < 0.7 AND SSD comparison shows IGSD provides no differentiated composability advantage
- Composability framework results are entirely null (all pairs sub-orthogonal with no interesting failure-mode story)

### Expected Contribution If Successful

1. First open-source consistency-distilled MDM at 8B instruction-tuned scale achieving LLaDA-8B quality in 4 steps on GSM8K/MATH500
2. 15-20x step reduction (64 → 4) with < 3% accuracy drop
3. Combination with training-free KV-caching for additional 1.5-3x speedup

### Resource Requirements

- GPU: 4-6 A100-days for adapter training + 2 A100-days for evaluation
- Timeline: 4-5 weeks total
- Risk: Convergence failure risk for discrete MDM (mitigate: pilot 100k steps first)

---

## Backup Candidate C: Batched MDM Inference Roofline Analysis and Convergence-Stratified Scheduling

### Status: Backup (elevated to "stronger backup" after result-debate)

### Core Idea

First systematic roofline analysis of MDM inference under batched workloads. Characterize compute-bound vs. memory-bound regimes on A100/H100. Develop convergence-stratified batching (group sequences by per-step unmasking fraction) and KV-budget-aware batch scheduler.

### Why This Matters More Than Previously Acknowledged

The result-debate confirmed: our implementation of EntropyCache (M1) achieves 1.38x vs. published 15.2x–26.4x. This 10x discrepancy is almost certainly due to our measuring wall-clock performance of a Python simulation without kernel-level sparse attention. The published speedups require CUDA-level implementation. This underscores that the hardware efficiency question is unsolved for MDMs in open-source settings.

A systematic roofline analysis would explain:
1. Why MDM KV-caching in Python simulation is slow (overhead-dominant regime)
2. At what batch size MDMs transition from compute-bound to memory-bound
3. Whether convergence-stratified batching can exploit the transition point

### Competitive Landscape (as of April 2026)

- SSD paper (arXiv:2510.04147) Figure 1: shows TPS vs. batch size comparison — a single figure observation, not a systematic roofline analysis
- No paper performs arithmetic intensity characterization, compute-bound/memory-bound transition analysis, or convergence-stratified batch scheduling for MDMs
- **Novelty: 8/10** (unchanged from iter_001)

### Pivot Trigger

- Both front-runner M1+IGSD results collapse (full-scale Ortho < 0.7) AND IGSD provides no differentiated composability advantage over SSD
- OR user/system indicates willingness to accept longer runtime (2-4 week engineering project)

### Expected Contribution If Successful

1. First fair throughput comparison of MDMs vs. AR models (vLLM-optimized) at batch sizes {1, 4, 8, 16, 32}
2. Roofline characterization: MDM inference arithmetic intensity vs. A100 roofline → identifies fundamental hardware bottleneck
3. Convergence-stratified batching achieving >= 2x batch throughput improvement over naive batching at batch size >= 8
4. Explains why Python-simulation speedup claims (including ours for M1) diverge from kernel-level speedup claims

### Resource Requirements

- GPU: 5-8 A100-days (measurement only); 15-20 A100-days (full optimization with custom scheduler)
- Timeline: 1-2 weeks (measurement only); 4-6 weeks (full)
- Engineering risk: custom CUDA kernels / Triton operators may be required for competitive scheduler

---

## Contingency Decision Table (Updated after iter_001)

| Scenario | Front-Runner Status | Action |
|----------|-------------------|--------|
| Full-scale Ortho >= 1.0 + SSD comparison differentiates IGSD | Strong result | Proceed to writing, NeurIPS 2026 primary target |
| Full-scale Ortho in [0.8, 1.0) + SSD synergizes equally | Moderate result | Write as "highly orthogonal", EMNLP/AAAI 2026 or NeurIPS workshop |
| Full-scale Ortho in [0.8, 1.0) + SSD does NOT synergize with M1 | IGSD differentiated | Write as synergy mechanism paper, IGSD as composability-enabling method |
| Full-scale Ortho < 0.8 | Weak result | Composability framework + failure atlas as workshop/negative-results paper; pivot to cand_c measurement |
| Full-scale Ortho >= 0.8 + tau=0.0 resolves as "IGSD = naive step reduction" | Method contribution gone | Write as composability analysis only; M1+naive-step-reduction as the synergistic pair |
| IGSD accept rate high but accuracy too low to be useful | IGSD infeasible as deployment method | Already the current situation; framed correctly as analysis vehicle |

# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **SAELens** (`jbloomAus/SAELens`, MIT, 1,100+ stars) — Dominant library for loading and analyzing pretrained SAEs. Supports Standard, Gated, TopK, JumpReLU, Matryoshka, BatchTopK architectures. Integrates with TransformerLens and Neuronpedia. **Already used in the project.**

2. **SAEBench** (`adamkarvonen/SAEBench`, open source, ICML 2025) — Industry-standard benchmark with 8 evaluations including feature absorption. Released 200+ pretrained SAEs on Pythia-160M and Gemma-2-2B. **Already used for first-letter absorption.**

3. **SynthSAEBench** (Chanin & Garriga-Alonso, arXiv:2602.14687, Feb 2026) — Synthetic benchmark with 16,384 ground-truth features including 10,884 hierarchical features (128 root trees, depth 3, branching factor 4). Critiques SAEBench metrics as having "high variance and measuring indirect proxies rather than feature recovery directly." **Built into SAELens >= 6.37.6.** HuggingFace: `decoderesearch/synth-sae-bench-16k-v1`.

4. **"Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?"** (Korznikov et al., arXiv:2602.14111, Feb 2026) — Random baselines match trained SAEs on AutoInterp (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and RAVEL (0.73 vs 0.72). On synthetic data, SAEs recover only 9% of true features despite 71% explained variance. **Did NOT test feature absorption specifically.**

5. **"Why Linear Probes and Sparse Autoencoders Fail at..."** (arXiv:2603.28744, Mar 2026) — Critical analysis of probe-based SAE evaluation. Suggests linear probes may capture spurious correlations rather than true feature structure.

6. **Matryoshka SAE official code** (`bartbussmann/matryoshka_sae`) — "Heavily inspired and basically a stripped-down version of SAELens." Provides reference implementation of nested dictionaries with multi-scale reconstruction. Reports absorption ~0.05 vs BatchTopK ~0.49 on first-letter tasks.

7. **Claude autonomous research on SynthSAEBench** (`chanind/claude-auto-research-synthsaebench`) — Claude achieved 0.97 F1 with "LISTA-Matryoshka" architecture on SynthSAEBench. Demonstrates the benchmark is actively being used for architecture innovation.

8. **OrtSAE** (Korznikov et al., arXiv:2509.22033) — Chunk-wise orthogonality penalty; claims 65% absorption reduction, ~4-11% compute overhead. **No public code yet.**

9. **H-SAE** (arXiv:2506.01197, 2025) — Hierarchical SAE with gating structure. Reduces absorption with 1/4 compute cost of Matryoshka. **Not yet in SAEBench.**

10. **"Measuring and Reallocating the Transformer's MLP Budget"** (arXiv:2603.03459, Mar 2026) — GPT-2 models are "fundamentally more linear than Pythia models." Explains GPT-2 vs Pythia divergence in SAE behavior.

11. **"The Geometry of Categorical and Hierarchical Representations"** (ICLR 2025, arXiv:2406.01506) — WordNet hierarchy is linearly represented in LLMs like Gemma-2B. Validates semantic hierarchy probe approach in principle.

12. **SAEBench documentation** (neuronpedia.org) — Absorption metric only recommended for models with "decent spelling knowledge" (>1B parameters). For Pythia-160M, "absorption could not be calculated due to insufficient first-letter features being detected."

### Landscape Summary

The field is experiencing a **skeptical turn** on SAE evaluation with four converging trends:
- **Random baselines challenge SAE validity** (Korznikov et al., 2026)
- **Probe-based metrics are under scrutiny** (arXiv:2603.28744)
- **Synthetic benchmarks emerge as alternative** (SynthSAEBench, 2026)
- **SAEBench metrics have known applicability constraints** (absorption requires >1B params for reliable spelling features)

Our prior iterations (iter_002-004) attempted construct-validity studies on real LLMs and encountered fatal anomalies: Random=Standard identity, reversed H2 (non-hierarchy > hierarchy), perfect AUROC=1.0 ceiling effect, and GPT-2 vs Pythia divergence. The pivot to SynthSAEBench component-isolation directly addresses these failures by using ground-truth synthetic data where absorption can be measured without probe-based metric collapse.

---

## Phase 2: Initial Candidates

### Candidate A: Component-Isolated Absorption Analysis on SynthSAEBench-16k (Front-Runner)

- **Core hypothesis**: Among SAE architectural innovations claimed to reduce absorption, **multi-scale dictionary decomposition** (the core mechanism in Matryoshka SAEs) is the primary driver of absorption reduction, while **per-feature thresholding** (JumpReLU) and **gating mechanisms** (Gated SAE) have negligible independent effects.
- **Implementation sketch**:
  1. Use SynthSAEBench-16k (16,384 ground-truth features with realistic correlation, hierarchy, superposition)
  2. Train 6 SAE variants: Baseline ReLU, +TopK, +MultiScale, +Orthogonality, +Gating, +Full Matryoshka
  3. Measure absorption using ground-truth features (no probes, no metric collapse)
  4. Validate top-performing component on real LLM (Pythia-160M or Gemma-2-2B)
- **Simplest version**: Baseline + TopK + MultiScale on SynthSAEBench-1k subset. Target: 15-20 min.
- **Time estimate**: ~0.5-1.0 GPU-hour for full synthetic study (6 variants x 5 replicates)
- **Reusable components**: SynthSAEBench official implementation (built into SAELens), existing Pythia-160M SAE cohort for validation

### Candidate B: Training-Dynamic Analysis of Absorption Emergence

- **Core hypothesis**: Absorption emerges early in SAE training (within first 10% of steps) and is largely irreversible; early-stopping or curriculum strategies can prevent it.
- **Implementation sketch**:
  1. Train a Standard ReLU SAE on SynthSAEBench-16k with checkpointing every 1% of training
  2. Measure absorption rate, feature recovery MCC, and reconstruction at each checkpoint
  3. Test whether absorption is reversible by temporarily removing sparsity penalty
  4. Compare emergence patterns across architectures
- **Simplest version**: Track absorption during training of 1 SAE on SynthSAEBench-1k, 20 checkpoints. Target: 15 min.
- **Time estimate**: ~0.5 GPU-hour for full dynamic analysis
- **Reusable components**: SAELens training loop with checkpointing, SynthSAEBench data loader
- **Why secondary**: Less directly addresses the community's immediate question ("what works?") but provides valuable mechanistic insight. Could be a follow-up experiment.

### Candidate C: Strong Baseline Done Right — Revisiting Standard SAE with Proper Tuning

- **Core hypothesis**: Much of the reported improvement from complex architectures (Matryoshka, OrtSAE) can be matched by a well-tuned Standard ReLU SAE with optimized hyperparameters, and the community has underinvested in baseline tuning.
- **Implementation sketch**:
  1. Grid search Standard ReLU SAE hyperparameters on SynthSAEBench-16k: L1 penalty, dictionary size, learning rate schedule
  2. Compare best-tuned Standard against reported Matryoshka/OrtSAE scores
  3. Measure absorption, MCC, reconstruction, L0
  4. Report compute-normalized comparison (FLOPs per quality unit)
- **Simplest version**: 3 L1 values x 2 dictionary sizes on SynthSAEBench-1k. Target: 20 min.
- **Time estimate**: ~1.0 GPU-hour for grid search
- **Reusable components**: SAELens training, SynthSAEBench
- **Why secondary**: The "strong baseline" angle is valuable but less novel than component isolation. Could be folded into Candidate A as a control condition.

---

## Phase 3: Self-Critique

### Against Candidate A (Component-Isolated Synthetic Study)

- **Implementation reality check**: SynthSAEBench has official code integrated into SAELens and has been used in published work (Chanin & Garriga-Alonso, 2026; Claude autonomous research project). Ground-truth features eliminate probe-based metric collapse entirely. **Risk: LOW.**
- **Reproducibility attack**: The benchmark is deterministic (fixed seed, fixed feature generation). 5 replicates per variant provide variance estimates. SAELens training is well-documented. **Risk: LOW.**
- **Baseline sanity check**: The baseline is Standard ReLU SAE on known ground-truth. There is no ambiguity about what "better" means — it means recovering more ground-truth features with less absorption. **Risk: LOW.**
- **Scope attack**: Component-isolated results transfer to any SAE architecture. If multi-scale dictionaries reduce absorption causally, this applies to Matryoshka, HSAE, and any future hierarchical variant. The Phase 2 real-LLM validation tests transfer. **Risk: MEDIUM** (synthetic-to-real gap is real but acknowledged).
- **Synthetic-to-real gap**: The main threat. SynthSAEBench features are designed to be realistic (correlation, hierarchy, superposition) but are still synthetic. **Mitigation**: Phase 2 validation on real LLM; acknowledge limitation in Discussion.
- **Verdict**: **STRONG.** Addresses all failure modes of prior iterations: no probe-based metric collapse, no small-model constraint, no estimated data, clear causal claims.

### Against Candidate B (Training-Dynamic Analysis)

- **Implementation reality check**: Training dynamics studies are common in deep learning but rare for SAEs specifically. No prior work tracks absorption emergence over time.
- **Reproducibility attack**: Requires checkpointing infrastructure; SAELens supports this. Results may be sensitive to initialization.
- **Baseline sanity check**: The baseline is "absorption at end of training," which is well-defined.
- **Scope attack**: Dynamics may differ between synthetic and real data. Limited generalizability.
- **Verdict**: **MODERATE.** Interesting but secondary. Best as a follow-up to Candidate A, not a standalone paper.

### Against Candidate C (Strong Baseline Done Right)

- **Implementation reality check**: Hyperparameter tuning is standard practice. The risk is that the grid search is incomplete and misses the true optimum.
- **Reproducibility attack**: Grid search results depend on search space. A narrow search may miss better settings.
- **Baseline sanity check**: Comparing against reported (not reproduced) Matryoshka/OrtSAE scores is unfair. Would need to reproduce those architectures too.
- **Scope attack**: Even if Standard ReLU matches complex architectures on SynthSAEBench, real LLM features may differ.
- **Verdict**: **MODERATE.** Valuable angle but better folded into Candidate A as a control. The "strong baseline" narrative is less novel than component isolation.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (training dynamics)**: Deferred to follow-up. The community's most urgent question is "what works?" not "when does it emerge?"
- **Candidate C (strong baseline)**: Folded into Candidate A as a "tuned Standard ReLU" control condition. The grid search becomes part of the experimental design rather than a separate paper.
- **Prior iteration's Goodhart's Law framing**: The pivot decision (iter_004 → iter_005) explicitly moved away from the construct-validity/Goodhart's Law approach due to fatal pipeline anomalies. The new direction is constructive and forward-looking.

### What Survives from Prior Iterations

- **H2 hierarchy specificity failure** (non-hierarchy > hierarchy, t=-4.75, p=0.003): Genuine finding that the absorption metric is not hierarchy-specific. This motivates the shift to ground-truth synthetic hierarchies.
- **Random=Standard identity on semantic hierarchies**: Confirmed the metric is driven by geometry, not learned structure. This motivates using ground-truth features where "ground truth" is known by construction.
- **Architecture ranking consistency** (Kendall's tau = 0.62): The ordinal ranking is preserved across tasks, suggesting ordinal validity even if absolute scores are suspect. This supports the component-isolation approach.
- **Data integrity lessons**: Never inherit data from previous iterations without verification. Pre-writing integrity checks are mandatory. These lessons inform the new experimental design.

### Strengthened Survivor

- **Candidate A is promoted to sole front-runner.** It directly addresses every failure mode of prior iterations:
  1. **Metric collapse eliminated:** Ground-truth features mean no probe-based metric collapse.
  2. **Underpowered sample fixed:** 5 replicates x 6 variants = 30 data points; ANOVA has adequate power.
  3. **Confounded controls eliminated:** All conditions use identical ground-truth hierarchies.
  4. **Ceiling effects eliminated:** Synthetic data can be made arbitrarily difficult.
  5. **Estimated data eliminated:** Every measurement comes from actual training.
  6. **Data integrity enforced:** SynthSAEBench is deterministic; results are reproducible.

### Selected Front-Runner

**Candidate A: Component-Isolated Absorption Analysis on SynthSAEBench-16k**

This shifts the research question from "Does first-letter absorption generalize?" (unanswerable with current tools on small models) to "What architectural components actually reduce absorption?" (answerable with ground-truth data). This is a more constructive and empirically tractable question.

---

## Phase 5: Final Proposal

### Title
**What Actually Reduces Feature Absorption? A Component-Isolated Study on Ground-Truth Synthetic Hierarchies**

### Hypothesis
Among SAE architectural innovations claimed to reduce absorption, **multi-scale dictionary decomposition** (the core mechanism in Matryoshka SAEs) is the primary driver of absorption reduction, while **per-feature thresholding** (JumpReLU) and **gating mechanisms** (Gated SAE) have negligible independent effects.

### Motivation
The SAE community has coalesced around feature absorption as a central pathology, with multiple architectures (Matryoshka, OrtSAE, HSAE, GBA) reporting absorption reduction as a primary contribution. However, no study has isolated which specific architectural component drives the improvement. Matryoshka SAEs combine at least three innovations: multi-scale dictionaries, batch-level TopK sparsity, and hierarchical loss weighting. OrtSAE adds orthogonality penalties. Gated SAEs decouple detection from magnitude. **Which of these actually matters?**

Our prior iterations (iter_002-004) revealed that probe-based absorption metrics on real LLMs suffer from metric collapse (perfect AUROC probes, Random=Standard identity, non-hierarchy > hierarchy). SynthSAEBench-16k provides ground-truth features where absorption can be measured directly — no probes, no ambiguity, no ceiling effects.

### Method

**Step 1: Baseline and Component-Isolated Variants**
Train 6 SAE variants on SynthSAEBench-16k, varying one component at a time:

| Variant | Components | What It Tests |
|---------|-----------|---------------|
| Baseline | Standard ReLU, L1 sparsity | Baseline absorption rate |
| +TopK | Replace L1 with TopK sparsity | Effect of explicit k-sparsity |
| +MultiScale | Nested dictionaries (2 levels) | Effect of hierarchical decomposition |
| +Orthogonality | Chunk-wise decoder orthogonality penalty | Effect of decoder incoherence |
| +Gating | Decoupled detection/magnitude paths | Effect of gating mechanism |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | Combined effect (replicates prior work) |

**Step 2: Ground-Truth Absorption Measurement**
For each variant, measure:
- **Absorption rate**: Fraction of parent features subsumed by child features (using known ground-truth parent-child relationships from SynthSAEBench hierarchy: 128 root trees, depth 3, branching factor 4)
- **Feature recovery MCC**: Matthews correlation between learned and ground-truth feature assignments
- **Reconstruction MSE**: Standard reconstruction quality
- **Sparsity (L0)**: Average active features per token

**Step 3: Statistical Analysis**
- One-way ANOVA across 6 variants (5 replicates each)
- Post-hoc Tukey HSD for pairwise comparisons
- Effect sizes (Cohen's d) for each component vs baseline
- Bonferroni correction across 15 pairwise comparisons

**Step 4: Real-LLM Validation (Phase 2)**
- Take the top-performing component from Step 3
- Train on Pythia-160M or Gemma-2-2B with that component added to a strong baseline
- Measure first-letter absorption via SAEBench
- Compare with existing architecture leaderboard

### Simplest Version
Baseline + TopK + MultiScale on SynthSAEBench-1k subset (1,024 features, 10M samples). Target: 15-20 minutes on 1 GPU. Hard gates:
1. All variants achieve MCC > 0.5
2. Clear ordering: MultiScale < TopK < Baseline in absorption rate
3. Variance across replicates < 20% of mean
4. Random-feature control achieves MCC < 0.1

### Baselines
1. **Standard ReLU SAE** — Expected: highest absorption, lowest MCC
2. **Full Matryoshka SAE** — Expected: lowest absorption, highest MCC (replicates prior work)
3. **Random-feature control** — Expected: near-zero MCC, high absorption (validates metrics)

### Experimental Plan

| Stage | Task | Variants | Metrics | Duration |
|-------|------|----------|---------|----------|
| Pilot | Baseline + TopK + MultiScale on 1k subset | 3 | Absorption, MCC, MSE | 15-20 min |
| Full | All 6 variants on 16k full | 6 x 5 replicates | Absorption, MCC, MSE, L0 | ~60 min |
| Validation | Top component on Pythia-160M | 1 + baseline | SAEBench absorption | ~30 min |
| Analysis | ANOVA + post-hoc + effect sizes | All | Statistical comparison | 5 min (CPU) |

### Resource Estimate
- **GPU-hours**: ~0.5-1.0 for synthetic experiments; ~0.5 for real-LLM validation
- **Model sizes**: SynthSAEBench-16k (synthetic, fast); Pythia-160M or Gemma-2-2B (validation)
- **All tasks well under the 1-hour limit**

### Risk Assessment

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Synthetic data doesn't match LLM feature structure | HIGH | Phase 2 real-LLM validation; acknowledge limitation in Discussion |
| Training instability (high variance across replicates) | MEDIUM | 5 replicates; report variance; increase if needed |
| Component interactions dominate main effects | MEDIUM | Test full Matryoshka variant; report interaction if observed |
| SynthSAEBench setup complexity | MEDIUM | Use official SAELens integration; test on subset first |
| Real-LLM SAEs don't match synthetic variants exactly | MEDIUM | Acknowledge approximate matching; focus on component presence/absence |
| Prior iteration data integrity issues recur | HIGH | Pre-writing automated check comparing all numbers to source JSON; no estimated data |

### Novelty Claim
This would be the **first component-isolated study of SAE architectural innovations using ground-truth synthetic hierarchies**. While Matryoshka SAEs, OrtSAE, and Gated SAEs have all reported absorption reductions, no study has isolated which specific component drives the improvement. By varying one component at a time on SynthSAEBench-16k — where ground-truth features are known — we can make causal claims about what actually works. The synthetic-to-real validation (Phase 2) tests whether these causal claims transfer to the real-world setting that matters for the community.

### Connection to Prior Iterations

This proposal directly addresses the failures of iter_002-004:
- **Metric collapse eliminated:** Ground-truth features mean no probe-based metric collapse.
- **Underpowered sample fixed:** 5 replicates x 6 variants = 30 data points; ANOVA has adequate power.
- **Confounded controls eliminated:** All conditions use identical ground-truth hierarchies.
- **Ceiling effects eliminated:** Synthetic data can be made arbitrarily difficult.
- **Estimated data eliminated:** Every measurement comes from actual pipeline execution.
- **Data integrity enforced:** SynthSAEBench is deterministic; pre-writing checks verify all numbers.

The research question shifts from "Does first-letter absorption generalize?" (which we couldn't answer due to metric collapse on small models) to "What architectural components actually reduce absorption?" (which we can answer with ground-truth data). This is a more constructive and empirically tractable question.

### Sources
- [SynthSAEBench (Chanin & Garriga-Alonso, 2026)](https://arxiv.org/abs/2602.14687)
- [Sanity Checks for SAEs (Korznikov et al., 2026)](https://arxiv.org/abs/2602.14111)
- [A is for Absorption (Chanin et al., 2024)](https://arxiv.org/abs/2409.14507)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [Matryoshka SAE (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [Matryoshka SAE Code (bartbussmann/matryoshka_sae)](https://github.com/bartbussmann/matryoshka_sae)
- [SAELens (jbloomAus/SAELens)](https://github.com/jbloomAus/SAELens)
- [Claude Auto-Research on SynthSAEBench](https://github.com/chanind/claude-auto-research-synthsaebench)

# Backup Ideas for Pivot

## Alternative A: Narrow Diagnostic Paper --- The SAEBench Absorption Metric: A Confounded Proxy

**Core hypothesis**: The SAEBench absorption metric on real LLMs is a confounded proxy mixing (a) a small learned-structure signal, (b) a large base-model geometry contribution, and (c) severe probe-task artifacts. Three specific claims are robust: co-occurrence confound (H3), ceiling effect (H5), and model dependence (H6).

**Why it is a viable pivot**: If the SynthSAEBench component-isolation experiment fails (e.g., L0-matched ablation confirms sparsity is sole driver, making the paper too narrow), the narrow diagnostic paper provides a fallback with zero additional GPU cost. The three claims (H3, H5, H6) are already supported by existing iter_002/003 data.

**Implementation sketch**:
1. Write focused paper with ONLY three claims:
   - H3: Co-occurrence confound (non-hierarchy > hierarchy, d = -1.79, p = 0.003)
   - H5: Ceiling effect (all AUROCs = 1.0)
   - H6: Model dependence (GPT-2 vs. Pythia divergence)
2. OMIT: Random=Standard (bug, fixed), utility disconnect (underpowered), Goodhart's Law framing (requires rejected H1)
3. Target: EMNLP 2026 Findings or workshop (ME-FoMo, ICLR MI Workshop)

**Cost**: 0 GPU-hours.
**When to pivot**: If SynthSAEBench L0-matched ablation shows sparsity is sole driver and reviewers find the contribution too incremental.

---

## Alternative B: The Rate-Distortion Origin of Feature Absorption --- A Combinatorial Bound

**Core hypothesis**: Feature absorption is inevitable under sparsity for tree-structured hierarchies, with an explicit depth-dependent bound. The absorption rate increases monotonically with tree depth and sparsity penalty.

**Why it is a viable pivot**: If the empirical experiments produce the "all sparsity" result (L0-matched Baseline matches TopK), a theoretical contribution provides a clean explanation: absorption is not a pathology to be eliminated by architecture, but a fundamental consequence of lossy compression with sparsity constraints. This reframes the null result as a theoretical insight.

**Implementation sketch**:
1. Prove a general theorem: for a tree-structured feature hierarchy with depth D and branching factor B, the absorption rate under L1-sparsity constraint lambda is bounded below by a function f(D, B, lambda)
2. Validate on synthetic hierarchical data with known ground-truth features
3. Test prediction that real SAEs with higher L0 show lower absorption rates

**Simplest version**: Synthetic 3-level binary tree (7 features) x 3 lambda values; measure absorption. Target: 15 min.
**Time estimate**: ~0.5 GPU-hour for synthetic validation.
**When to pivot**: If L0-matched ablation confirms sparsity is sole driver, and reviewers request theoretical grounding.

---

## Alternative C: Lateral Inhibition Prevents Feature Absorption --- A Neural-Circuit-Inspired SAE Architecture

**Core hypothesis**: An SAE with LCA-style lateral inhibition (where active latents suppress competitors proportional to decoder cosine similarity) will exhibit lower feature absorption rates than standard TopK/BatchTopK SAEs with equivalent sparsity and reconstruction quality.

**Why it is a viable pivot**: If the component-isolation study shows that no existing component significantly reduces absorption beyond sparsity control (null result for architecture), this provides a novel architectural direction that goes beyond existing designs. It also provides a constructive contribution if reviewers demand more than "what works" analysis.

**Implementation sketch**:
1. Implement LCA-SAE encoder with iterative lateral inhibition: u = u - eta * (G @ z) where G is decoder Gram matrix
2. Use sparse Gram approximation (top-k neighbors per latent) for scalability
3. Train on Pythia-160M or Gemma-2-2B activations
4. Compare absorption, reconstruction, and L0 against Standard, TopK, and Matryoshka
5. Diagnostic: structured vs. random inhibition control

**Simplest version**: 1 SAE architecture (LCA-SAE) vs. Standard on GPT-2 small, 5 iterations. Target: 30 min.
**Time estimate**: ~1.5 GPU-hours (training + evaluation).
**When to pivot**: If reviewers request a constructive proposal beyond component analysis; or if main experiment shows that sparsity alone explains all variance.

---

## Alternative D: Information-Theoretic Bounds on Feature Absorption

**Core hypothesis**: Feature absorption can be rigorously quantified as a conditional mutual information deficit: the information about a parent feature that is not independently encoded in the SAE representation, given that child-level features are active.

**Why it is a viable pivot**: If the component-isolation experiments reveal that the absorption metric IS measuring something real (just confounded by geometry on real LLMs), an information-theoretic reformulation could provide a principled, task-agnostic alternative that works on both synthetic and real data.

**Implementation sketch**:
1. Define IT-absorption bound: Absorption(P) >= I(P; X) - I(P; Z | C_1, ..., C_k)
2. Approximate MI using variational bounds (MINE, InfoNCE)
3. Validate against Chanin et al. first-letter absorption on 5-10 SAEs
4. Test generalization to WordNet semantic hierarchies

**Simplest version**: 5 SAEs x first-letter features; compute IT bound and correlate with Chanin metric. Target: 30 min.
**Time estimate**: ~1.5 GPU-hours for validation + semantic hierarchy tests.
**When to pivot**: If L0-matched ablation shows trained SAEs consistently outperform random/PCA baselines on synthetic data, suggesting the metric captures real structure but is confounded by geometry on real LLMs. The IT bound would provide a geometry-independent alternative.

---

## Alternative E: Homeostatic Sparse Autoencoders --- Ecologically-Inspired Activation Constraints

**Core hypothesis**: Adding a homeostatic constraint that enforces minimum average activation for all SAE latents (inspired by synaptic scaling in neuroscience) will reduce feature absorption. Frequency-dependent targets (rare features get lower minima) will outperform uniform targets.

**Why it is a viable pivot**: If the main paper shows that absorption IS a real problem but current mitigation methods are flawed (e.g., TopK achieves low absorption but 82% dead latents), a novel training objective inspired by ecological competition theory would be a constructive contribution.

**Implementation sketch**:
1. Fine-tune pretrained SAEs with homeostatic loss term: L_homeostatic = sum_i max(0, tau_i - E[z_i])
2. Compare three target schedules: uniform, frequency-dependent (ecological), random
3. Measure absorption, reconstruction, and L0 on first-letter and semantic hierarchies

**Simplest version**: 1 SAE (GPT-2 small) x 3 homeostatic conditions x 5 WordNet pairs. Target: 15 min.
**Time estimate**: ~0.5 GPU-hour for pilot; ~1.5 hours for full experiment.
**When to pivot**: If the main paper's results are accepted but reviewers request a constructive proposal beyond critique. This provides a novel architectural direction.

---

## Alternative F: TopK Dose-Response and Optimal Sparsity for Absorption Control

**Core hypothesis**: There exists an optimal k (sparsity level) that minimizes absorption while preserving reconstruction quality and minimizing dead latents. This optimal k is task-dependent and can be characterized via a dose-response curve.

**Why it is a viable pivot**: If the main finding is that "sparsity drives absorption," the natural follow-up is "what is the optimal sparsity?" This transforms the main paper's reframing into a practical contribution with clear practitioner guidance.

**Implementation sketch**:
1. Train TopK SAEs with k in {10, 25, 50, 100, 200, 500, 1000} on SynthSAEBench-16k
2. Measure: absorption rate, reconstruction MSE, dead latent rate, feature recovery MCC
3. Characterize the dose-response curve and identify the Pareto-optimal k
4. Test whether the optimal k generalizes across feature hierarchy depths

**Time estimate**: ~0.5 GPU-hour for 7 k values.
**When to pivot**: If reviewers accept the sparsity reframing but request practical guidance on "what k to use."

---

## Pivot Decision Tree

```
Main paper results (SynthSAEBench component-isolation + L0-matched):
|
|-- L0-matched Baseline matches TopK (sparsity is sole driver)
|   --> Reframe as "Absorption is a Sparsity Phenomenon"
|   --> Consider Alternative B (theory) for theoretical grounding
|   --> Consider Alternative F (dose-response) for practical guidance
|
|-- L0-matched Baseline does NOT match TopK (TopK has architectural benefit)
|   --> Continue main paper; target NeurIPS/ICML/ICLR main
|   --> TopK has genuine architectural advantage beyond sparsity
|
|-- MultiScale shows independent effect (d > 0.8 at matched L0)
|   --> Continue main paper; both TopK and MultiScale matter
|   --> Strengthens original H1 (MultiScale dominance) with nuance
|
|-- All components equivalent at matched L0
|   --> Strongest version of "sparsity phenomenon" claim
|   --> Consider Alternative B (theory) or Alternative C (LCA-SAE)
|
|-- Reviewers request constructive proposal beyond component analysis
|   --> Consider Alternative C (LCA-SAE) or Alternative E (Homeostatic SAEs)
|
|-- Reviewers request geometry-independent metric
|   --> Consider Alternative D (IT-bound) for principled alternative
```

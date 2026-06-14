# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024) - "A is for Absorption"** (arXiv:2409.14507) — First systematic study of feature absorption; introduces ablation-based absorption detection metric; proves absorption is caused by hierarchical feature co-occurrence under sparsity optimization. *Limitation: metric unreliable past layer 17.*

2. **Chanin et al. (2025a) - "Feature Hedging"** (arXiv:2505.11756) — Identifies feature hedging (correlated feature merging in narrow SAEs); reveals absorption-hedging trade-off; proposes balanced loss coefficients (~0.75).

3. **Chanin & Garriga-Alonso (2025b) - "Sparse but Wrong"** (arXiv:2508.16560) — Proves L0 is not neutral; too-low L0 causes hedging, too-high causes degeneracy; proposes decoder pairwise cosine similarity (c_dec) metric.

4. **Karvonen et al. (2025) - "SAEBench"** (arXiv:2503.09532, ICML 2025) — 8-metric evaluation suite; includes projection-based absorption metric that works across all layers (vs ablation which fails past layer 17).

5. **Gao et al. (2024) - "Scaling and Evaluating SAEs"** (arXiv:2406.04093) — k-sparse autoencoders; establishes scaling laws; baseline absorption occurrence.

6. **Bussmann et al. (2025) - "Matryoshka SAEs"** (arXiv:2503.17547) — Nested dictionaries reduce absorption but trade for hedging at inner levels.

7. **Korznikov et al. (2025) - "OrtSAE"** (arXiv:2509.22033) — Orthogonality penalty reduces absorption by 65%; training-time solution.

8. **Cui et al. (2026) - "On the Limits of SAEs"** (arXiv:2506.15963, ICLR 2026) — First closed-form theoretical analysis; proves standard SAEs generally cannot fully recover ground-truth features.

9. **Costa et al. (2025) - "MP-SAE"** (arXiv:2506.03093, NeurIPS 2025) — Matching Pursuit greedy selection reduces absorption vs Vanilla/BatchTopK.

10. **Peng et al. (2025) - "CE-Bench"** (arXiv:2509.00691) — LLM-free contrastive evaluation; 5,000 story pairs; 77% alignment with SAEBench.

11. **Gurnee et al. (2023) - "K-sparse Probing"** — Detects feature splitting; evaluates sparse concept recovery.

12. **Song et al. (2025) - "Feature Consistency Position"** (arXiv:2505.20254) — Proposes PW-MCC for consistency across training runs.

### Experimental Landscape

**What has been properly tested:**
- Absorption exists across many SAE architectures (TopK, JumpReLU, Gated)
- Absorption is caused by hierarchical feature co-occurrence during sparse optimization
- Wide SAEs suffer absorption; narrow SAEs suffer hedging
- L0 directly impacts feature correctness (not a neutral hyperparameter)

**What is accepted without evidence:**
- Generalizability of first-letter spelling absorption findings to other hierarchical features (semantic, syntactic, factual)
- Cross-model absorption patterns (most studies focus on Gemma-2-2B)
- How absorption in deeper layers (18+) differs from early layers
- Practical impact of absorption on downstream tasks (circuit discovery, steering)

**Methodological gaps:**
- No systematic ablation study quantifying absorption rate vs. feature co-occurrence frequency, dictionary size, sparsity level
- No training-free detection method for absorption in pretrained SAEs (current methods require ablation or retraining)
- No comprehensive cross-model/layer absorption measurement
- Limited understanding of absorption impact on actual interpretability workflows

---

## Phase 2: Initial Candidates

### Candidate A: Systematic Cross-Model Absorption Quantification Across Layers

**Core Hypothesis**: Absorption rates vary systematically across model families (GPT-2 vs Gemma-2B), layer depths (early vs. middle vs. late), and SAE configurations (width 16k vs 65k), following predictable patterns that can be characterized as "absorption scaling laws."

**Falsification Criterion**: If absorption rates are roughly constant (~15-20% as reported by Chanin et al.) across all models, layers, and configurations, then the hypothesis is disproven. We expect to find significant variation (e.g., later layers have 2-3x higher absorption due to more hierarchical features).

**Evaluation Protocol**:
- Primary benchmark: Ablation-based absorption metric (Chanin et al.) for layers 0-17; projection-based metric (SAEBench) for deeper layers
- Models: GPT-2-small, Gemma-2-2B (via GemmaScope)
- SAE configs: 16k and 65k latent widths
- Layers: 0-17 (ablation), 18-26 (projection-based)
- Metric: Feature absorption rate (% of parent features absorbed by child features)
- Statistical test: Bootstrap 95% CI across 3 random seeds; paired t-test for model/layer comparisons
- Sample size: 500+ prompt-feature pairs per condition

**Ablation Plan**:
- Ablate width (16k vs 65k): isolates effect of dictionary size on absorption
- Ablate layer depth (early vs. mid vs. late): isolates effect of feature hierarchy depth
- Ablate model family (GPT-2 vs Gemma): isolates architecture-specific effects

**Confounders Identified**:
- L0 variation across SAE configs must be controlled (use c_dec metric to verify optimal L0)
- Probe task differences (first-letter spelling may not generalize)
- Attention-mediated information flow in deeper layers (mitigated by projection-based metric)

**Pilot Design** (<15 min):
- Run absorption measurement on GPT-2 SAE (layers 0-10, 16k width only) using 100 prompt-feature pairs
- Verify metric reliability before scaling

**Resource Estimate**: ~3 GPU-hours (GPT-2: 30 min, Gemma-2-2B: 2.5 hours)

---

### Candidate B: Training-Free Absorption Detection via Encoder-Decoder Asymmetry Analysis

**Core Hypothesis**: Encoder-decoder weight asymmetry (measured by decoder pairwise cosine similarity, c_dec) correlates with absorption rate, enabling training-free absorption detection without ablation studies.

**Falsification Criterion**: If c_dec shows no correlation (r < 0.2) with ablation-measured absorption rates, the hypothesis is disproven. A strong correlation (r > 0.6) would validate c_dec as a proxy metric.

**Evaluation Protocol**:
- Primary metric: Pearson correlation between c_dec and ablation-based absorption rate
- Dataset: GPT-2 SAEs (layers 0-17) and GemmaScope SAEs (layers 0-17)
- c_dec computation: Mean pairwise cosine similarity between decoder column vectors
- Absorption rate: Ablation-based metric from Chanin et al.
- Validation: Cross-validate on held-out SAE configurations

**Ablation Plan**:
- c_dec vs. absorption at different layers: tests if correlation holds across depth
- c_dec vs. absorption at different widths: tests if correlation holds across SAE configurations
- Compare c_dec to projection-based absorption metric for deeper layers

**Confounders Identified**:
- c_dec may correlate with other SAE properties (dead latents, feature similarity) not just absorption
- Correlation may be indirect (both caused by sparsity level)
- Need to partial out confounding variables

**Pilot Design** (<15 min):
- Compute c_dec for 5 GPT-2 SAEs (different layers) and correlate with absorption rates from existing literature

**Resource Estimate**: ~1 GPU-hour (computation is just weight analysis, no forward passes)

---

### Candidate C: Quantifying Absorption Impact on Circuit Discovery Reliability

**Core Hypothesis**: Feature absorption degrades the reliability of circuit discovery methods that rely on SAEs, causing discovered circuits to be incomplete or to misattribute functionality to wrong components.

**Falsification Criterion**: If circuit discovery results are equally reliable whether using high-absorption or low-absorption SAEs, then absorption does not matter practically. We expect to find that high-absorption SAEs produce circuits with 20%+ lower intervention effect sizes.

**Evaluation Protocol**:
- Task: Indirect Object Identification (IOI) circuit discovery (standard circuit analysis task)
- SAEs: Gemma-2-2B with varying absorption rates (compare GemmaScope 16k vs 65k)
- Metric: Causal intervention effect size when ablating discovered circuit components
- Baseline: Compare to transcoder-based circuit discovery (which handles absorption differently)
- Statistical test: Bootstrap CI on intervention effect sizes; Cohen's d for effect size comparison

**Ablation Plan**:
- Ablate circuit components discovered via high-absorption SAE vs. low-absorption SAE: measures reliability difference
- Ablate absorbed features vs. non-absorbed features: isolates absorption effect
- Compare circuit completeness (number of components identified) across SAE types

**Confounders Identified**:
- Different SAEs may identify genuinely different circuits (not just noisy measurements of the same circuit)
- Circuit discovery method variation (different attribution methods may find different circuits)
- Model architecture differences in how circuits are implemented

**Pilot Design** (<15 min):
- Run IOI circuit discovery on 1 layer of Gemma-2-2B using 2 different SAE widths
- Compare component overlap and preliminary intervention effects

**Resource Estimate**: ~4 GPU-hours (circuit analysis is expensive; may need to limit scope)

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Model Quantification)

**Confound Attack**: The ablation-based absorption metric is only reliable up to layer 17. For deeper layers, the projection-based metric must be used, but these measure different things (probe projection contribution vs. direct ablation effect). Comparing absorption rates across layer depths using different metrics introduces a confound that could explain any observed variation.

**Statistical Attack**: The expected variation in absorption rates across models/layers may be small (e.g., 15% vs 20%) and within noise margin given the probe task variance. A null result (no systematic variation) would be indistinguishable from insufficient statistical power.

**Benchmark Attack**: The first-letter spelling task measures only one type of hierarchical feature (letter identity within words). Absorption rates for semantic hierarchies (animal -> dog), syntactic hierarchies, or factual hierarchies may be systematically different. The study would measure absorption for spelling tasks but claim general conclusions about all hierarchical features.

**Ablation Completeness Attack**: Even with successful ablation studies, the "absorption rate" metric conflates two distinct phenomena: (1) features that genuinely cannot be separated due to co-occurrence, and (2) features that could be separated with different training regimes. Without varying training conditions, we cannot disentangle these.

**Verdict**: MODERATE. The experimental design is sound but limited by metric reliability at deeper layers and probe task generalizability. This is a necessary first step but alone insufficient to establish absorption "scaling laws."

---

### Against Candidate B (Training-Free Detection via c_dec)

**Confound Attack**: c_dec (decoder pairwise cosine similarity) is a proxy for feature "compactness" or "non-overlap," which could be caused by multiple factors: high absorption (features merged), low absorption but high sparsity (few active features), or simply well-trained features with clear directions. Without disentangling these causes, c_dec cannot uniquely identify absorption.

**Statistical Attack**: Even if c_dec correlates with absorption, the correlation may be nonlinear or threshold-based. A linear correlation coefficient may miss important structure (e.g., c_dec only predicts absorption below a certain sparsity threshold).

**Benchmark Attack**: The "ground truth" absorption rates come from the same ablation metric that has known reliability issues (layers 0-17 only). Using flawed ground truth to validate a proxy metric could lead to circular validation.

**Ablation Completeness Attack**: c_dec is a static property of the trained SAE weights. It cannot capture dynamic absorption that depends on input distribution. Features that appear well-separated by c_dec may still absorb for specific input patterns.

**Verdict**: WEAK. The confounds are serious and the circular validation problem is fundamental. However, if c_dec can predict downstream task degradation (rather than just correlating with ablation metrics), the approach gains validity.

---

### Against Candidate C (Circuit Discovery Impact)

**Confound Attack**: Circuit discovery methods themselves have high variance. Different methods (activation patching, TCAV, path patching) applied to the same model often find different circuits. Observed "unreliability" with high-absorption SAEs could be due to method variance, not absorption per se.

**Statistical Attack**: Circuit intervention effects have high variance across tokens and layers. Detecting a 20% effect size difference requires large sample sizes (potentially 1000+ tokens), which may exceed time budget.

**Benchmark Attack**: The IOI task is well-studied but may not generalize to other circuits. Absorption may impact some circuits (those with hierarchical structure) more than others (those with flat structure). A single circuit task cannot establish general practical impact.

**Ablation Completeness Attack**: Even with careful controls, comparing circuits discovered via different SAEs is confounded by the SAE itself (not just its absorption properties). The SAE architecture, training data, and width all affect what circuits are discoverable, independent of absorption.

**Verdict**: STRONG. This directly addresses the most important question: does absorption matter for practical interpretability? The confound of circuit discovery variance is addressable by using multiple discovery methods and focusing on intervention effect sizes rather than circuit component overlap.

---

## Phase 4: Refinement

### Dropped Candidates
- **Candidate B (c_dec)**: Dropped due to serious confounds and circular validation risk. Too indirect; the confound structure makes it nearly impossible to establish clean causal interpretation.

### Strengthened Candidates

**Candidate A (Cross-Model Quantification)**:
- *Added missing controls*: Include SAEBench projection-based metric as a separate analysis (not averaged with ablation metric)
- *Tightened falsification criterion*: Instead of expecting specific absorption scaling laws, test the more specific hypothesis: "absorption rate increases with layer depth due to accumulated hierarchical composition"
- *Added validation*: Use CE-Bench independence scores as an orthogonal measure of feature quality to cross-validate absorption patterns

**Candidate C (Circuit Discovery Impact)**:
- *Added missing controls*: Compare circuit reliability across multiple circuit discovery methods (activation patching + path patching) to separate method variance from SAE variance
- *Tightened falsification criterion*: Changed from "20% effect size difference" to "any statistically significant degradation in intervention precision" to make the threshold more achievable
- *Added additional benchmark*: Add a second circuit task (e.g., Greater-Than comparison) to test generalizability beyond IOI

### Selected Front-Runner

**Candidate C: Quantifying Absorption Impact on Circuit Discovery Reliability**

Rationale:
1. Addresses the most important open question identified in Gap 3 and Gap 9 (actionability)
2. Directly produces evidence that matters for practical interpretability research
3. Can be scoped to fit within time budget (≤1 hour per experiment)
4. Uses existing circuit analysis infrastructure (IOI, Greater-Than)
5. Results are publishable regardless of outcome (even a null result is informative)

Candidate A remains valuable as a supporting study but should follow C (establishing relevance before quantifying patterns).

---

## Phase 5: Final Proposal

### Title

**"Does Feature Absorption Compromise Circuit Discovery? An Empirical Study on SAE-Driven Circuit Analysis"**

### Hypothesis

Feature absorption in wide SAEs (e.g., GemmaScope 65k) significantly degrades the reliability of circuit discovery methods, causing (1) lower causal intervention effect sizes when ablating discovered circuit components, and (2) incomplete circuit definitions missing absorbed features, compared to narrower SAEs (e.g., GemmaScope 16k) with lower absorption rates.

**Specific metric**: Intervention effect size (Cohen's d) when ablating SAE-identified circuit components vs. ablating equivalent random feature sets.

### Falsification Criterion

If high-absorption SAEs (65k) produce circuits with statistically indistinguishable intervention effect sizes from low-absorption SAEs (16k) (p > 0.05, paired t-test), the hypothesis is rejected. We expect d > 0.5 (medium effect) favoring low-absorption SAEs.

### Method

Compare circuit discovery reliability across SAE width configurations using causal intervention analysis on the Indirect Object Identification (IOI) task and Greater-Than comparison task.

**Circuit Discovery Protocol**:
1. Load Gemma-2-2B with GemmaScope SAEs at 16k and 65k widths, layer 10 (where IOI circuits are prominent)
2. Collect activation data for IOI prompts using both SAEs
3. Apply activation patching circuit discovery (standard method) to identify key components
4. Measure intervention effect size: ablating each discovered component and measuring output logits change
5. Compare effect size distributions between 16k and 65k SAE circuits

### Evaluation Protocol

**Primary Benchmarks**:
- IOI (Indirect Object Identification) task: Standard circuit analysis benchmark
- Greater-Than comparison task: Second circuit for generalizability

**Metrics**:
- Intervention effect size (Cohen's d) per component
- Circuit completeness score (% of total logit effect explained by discovered components)
- False positive rate (% of components with effect size not significantly > 0)

**Statistical Test Plan**:
- Paired t-test (or Wilcoxon signed-rank) comparing effect size distributions between SAE widths
- Bootstrap 95% CI on Cohen's d difference
- Minimum 3 random seeds for SAE configurations

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|--------------|------------------|
| Ablate 16k SAE circuit components | Baseline circuit reliability (low absorption) | High effect sizes (d > 0.8) |
| Ablate 65k SAE circuit components | Circuit reliability with high absorption | Lower effect sizes (d ~ 0.3-0.5) if absorption degrades reliability |
| Ablate absorbed features specifically | Direct absorption effect | Large effect when ablating absorbed vs. non-absorbed features |
| Ablate random feature sets | Null baseline | Near-zero effect (d ~ 0) |
| Cross-architecture: GPT-2 16k vs 65k | Architecture independence | Similar pattern to Gemma if effect is general |

### Control Experiments

1. **L0 control**: Verify both SAEs have optimal L0 (using c_dec metric) to avoid confounding by L0-induced hedging
2. **Probe quality control**: Verify linear probes achieve similar accuracy on both SAE widths for the IOI task (ensuring task is learnable)
3. **Circuit independence control**: Verify circuits discovered via different SAEs have minimal component overlap (to test if they find genuinely different structures)

### Pilot Design (<15 min)

Run IOI circuit discovery on a single layer (layer 10) of Gemma-2-2B using:
- GemmaScope 16k width SAE
- 50 IOI prompt examples
- Simplified patching (top 5 components only)
- Measure intervention effects

This validates the pipeline before full experiment.

### Resource Estimate

| Experiment | Time | GPU |
|-----------|------|-----|
| Pilot (1 layer, 50 examples) | 10 min | 1 GPU |
| Full IOI (all layers, 500 examples) | 45 min | 1 GPU |
| Greater-Than replication | 30 min | 1 GPU |
| GPT-2 cross-validation | 30 min | 1 GPU |
| **Total** | ~2 hours | 1 GPU |

Can be parallelized across multiple GPUs if needed.

### Risk Assessment

| Threat | Likelihood | Mitigation |
|--------|------------|------------|
| Ablation metric unreliability at deeper layers | Medium | Focus on layers 8-12 where IOI is most prominent |
| Circuit discovery method variance | High | Use multiple discovery methods; report variance |
| Confounding by SAE width (not absorption) | Medium | Control for L0; analyze decoder geometry |
| Null result (absorption doesn't matter) | Medium | Report as valid finding; shift to quantifying "when" absorption matters |

### Novelty Claim

This is the **first empirical study** quantifying the practical impact of feature absorption on circuit discovery reliability. While prior work identified absorption as a phenomenon and proposed architectural remedies, no prior work measures whether absorption actually breaks interpretability workflows in practice. This study directly measures whether "interpretability illusions" (Chanin et al.'s term for absorption-induced false negatives) translate to unreliable scientific conclusions.

### Connection to Literature Gaps

- **Gap 3** (impact on downstream tasks): Directly addresses this
- **Gap 9** (actionability): Establishes whether absorption quantification matters for practical use
- **Gap 6** (absorption-hedging trade-off): Cross-width comparison naturally tests this trade-off

---

## Companion Artifacts (JSON)

```json
{
  "perspective": "empiricist",
  "selected_candidate": "C",
  "title": "Does Feature Absorption Compromise Circuit Discovery?",
  "hypothesis": "Feature absorption in wide SAEs degrades circuit discovery reliability",
  "falsification_criterion": "p > 0.05 for 16k vs 65k SAE intervention effect size difference",
  "expected_effect_size": "d > 0.5 favoring low-absorption SAE",
  "primary_benchmark": "IOI circuit analysis",
  "replication_benchmark": "Greater-Than comparison",
  "metrics": ["Cohen's d intervention effect", "circuit completeness score", "false positive rate"],
  "statistical_tests": ["paired t-test", "bootstrap 95% CI", "Wilcoxon signed-rank"],
  "pilot_time_minutes": 10,
  "full_experiment_time_minutes": 120,
  "gpu_requirements": "1-2 GPUs",
  "key_risks": ["circuit discovery variance", "confounding by SAE width", "null result"],
  "novelty_contribution": "First empirical study connecting absorption to circuit discovery reliability in practice"
}
```

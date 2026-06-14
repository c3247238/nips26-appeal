# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (Chanin et al., NeurIPS 2025) — [github.com/lasr-spelling/sae-spelling](https://github.com/lasr-spelling/sae-spelling) — Canonical absorption metric implementation. Directly reusable `feature_absorption_calculator`, `probing`, and `feature_ablation` modules. MIT license. **Code exists and is battle-tested.**

2. **SAEBench** (Karvonen et al., ICML 2025) — [github.com/adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench) — 8-metric evaluation suite with absorption as one metric. 200+ pre-trained SAEs across 7 architectures, 3 widths (4k/16k/65k), 6 sparsities on Pythia-160M layer 8 and Gemma-2-2B layer 12. All evals run on a single 24GB GPU. Apache 2.0 license. **Code exists, pre-trained SAE weights and benchmark results are publicly available.**

3. **SAELens v6** (Bloom et al.) — [github.com/decoderesearch/SAELens](https://github.com/decoderesearch/SAELens) — Standard SAE training/loading library. `SAE.from_pretrained()` loads any Gemma Scope / SAEBench SAE in 3 lines. Supports encode/decode on arbitrary activation tensors. MIT license. **Code exists, PyPI: sae-lens v6.37.6.**

4. **Gemma Scope SAEs** (Google DeepMind) — [huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope) — 400+ JumpReLU SAEs on all layers/sublayers of Gemma 2 2B/9B/27B, widths 1k--1M. Gemma Scope 2 adds Matryoshka SAEs and transcoders on Gemma 3. **Pre-trained weights available, no training needed.**

5. **RAVEL benchmark** (Huang et al., ACL 2024) — [github.com/explanare/ravel](https://github.com/explanare/ravel) — Entity-attribute disentanglement benchmark with cities, countries, languages, continents. 3000+ cities with known hierarchical attributes on HuggingFace (`hij/ravel`). Integrated into SAEBench. **Code and data exist. Provides the exact hierarchical entity-attribute structure needed to study absorption beyond first-letter tasks.**

6. **TransformerLens** — [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — Hook-based activation extraction for 50+ models. Deep integration with SAELens. MIT license. **Code exists.**

7. **"Sparse but Wrong"** (Chanin & Garriga-Alonso, 2025) — [github.com/chanind/sparse-but-wrong-paper](https://github.com/chanind/sparse-but-wrong-paper) — L0 analysis framework showing most open-source SAEs have too-low L0, causing feature hedging. Proxy metric for correct L0. **Code exists. Critical for disentangling hedging from absorption.**

8. **"Feature Hedging"** (Chanin et al., 2025) — [github.com/chanind/feature-hedging-paper](https://github.com/chanind/feature-hedging-paper) — Feature hedging analysis, balanced Matryoshka SAE with compound multiplier ~0.75. Shows absorption and hedging trade off. **Code exists.**

9. **"Do I Know This Entity?"** (Ferrando et al., ICLR 2025) — Uses SAE latents to discover entity recognition features across entity types (cities, movies, songs, players). Shows SAE features can encode known-vs-unknown entity status across entity hierarchies. **Provides evidence that entity-level hierarchical features exist in SAE latent spaces.**

10. **"The Geometry of Concepts"** (Tegmark et al., 2025) — Studies geometric structure of SAE features including country-capital, gender analogies, parallelogram structures. Decoder cosine similarity reveals hierarchical and relational structure. **Provides methods for analyzing feature geometry that could be repurposed for absorption detection.**

11. **sae-probes** (Kantamneni et al., ICML 2025) — [github.com/JoshEngels/SAE-Probes](https://github.com/JoshEngels/SAE-Probes) — Standalone sparse probing benchmark. Tests SAE utility across data scarcity, class imbalance, label noise, covariate shift. **Code exists. Provides controlled probing infrastructure.**

12. **Neuronpedia** — [neuronpedia.org](https://neuronpedia.org) — Interactive SAE feature explorer, 50M+ latents, search/steering/circuit tracing. SAEBench results visualizable. Open-source since March 2025. **Useful for qualitative case studies of absorbed features.**

### Landscape Summary

The tooling ecosystem is exceptionally mature for a training-free analysis project:

- **SAELens + TransformerLens + Gemma Scope** provides a complete pipeline from model loading to SAE feature extraction in <10 lines of Python.
- **SAEBench** already has the absorption metric implemented and 200+ SAEs evaluated, but only on the first-letter spelling task.
- **RAVEL** provides the entity-attribute hierarchy data needed to extend absorption analysis to richer semantic domains.
- **sae-spelling** provides the canonical absorption metric code that can be adapted.

**What works**: Loading pre-trained SAEs and computing feature activations is trivially easy. The absorption metric is well-defined and implemented. Multiple SAE architectures are available pre-trained.

**What does not work well**: The absorption metric requires known probe directions (researcher must know what features to look for). Only the first-letter task has been used. No systematic cross-domain or cross-architecture comparison exists under matched conditions.

**Practical gaps**: (1) No one has measured absorption on RAVEL-style entity-attribute hierarchies. (2) No one has systematically disentangled absorption from hedging confounds across architectures. (3) No unsupervised absorption detection method exists. (4) The decoder cosine similarity structure has been studied for geometry but not specifically leveraged to detect absorption without probes.

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy — Measuring Absorption Beyond First Letters

- **Hypothesis**: Feature absorption rates vary systematically across semantic hierarchy types (syntactic vs. entity-type vs. knowledge-relational), and the severity is predictable from hierarchy depth, co-occurrence frequency, and feature specificity. Specifically, absorption will be more severe in knowledge-domain hierarchies (country > city) than in the first-letter task because entity features have higher co-occurrence rates with their parent categories.
- **Implementation sketch**: 
  1. Use SAELens to load pre-trained Gemma Scope SAEs (layer 12, widths 16k/65k on Gemma-2-2B).
  2. Adapt the sae-spelling absorption metric to work with RAVEL entity-attribute pairs (city-country, city-continent, city-language) by training logistic regression probes for each attribute and measuring false-negative rates where the probe succeeds but SAE latents fail.
  3. Construct additional hierarchy tasks: (a) taxonomy (animal > mammal > dog), (b) sentiment hierarchy (positive > enthusiastic vs. positive > calm), using existing NLP datasets with known label hierarchies.
  4. Measure absorption rate per hierarchy type, per SAE architecture (JumpReLU, TopK, BatchTopK from SAEBench), per layer.
  5. Correlate absorption rate with hierarchy depth, parent-child co-occurrence frequency, and feature specificity (probe accuracy).
- **Simplest version**: Just RAVEL city-country hierarchy on Gemma-2-2B layer 12 with 3 pre-trained SAE architectures. This alone is novel --- no one has measured absorption on entity-attribute hierarchies.
- **Time estimate**: ~2-3 hours for full experiment (probe training ~20 min per hierarchy, absorption metric computation ~30 min per SAE, 3 SAEs x 4 hierarchies). Pilot: 15 min (single hierarchy, single SAE).
- **Reusable components**: SAELens (loading), sae-spelling (absorption metric core), RAVEL dataset (HuggingFace), SAEBench pre-trained SAEs (weights), TransformerLens (activation extraction).

### Candidate B: Decoder Geometry as Unsupervised Absorption Detector

- **Hypothesis**: Feature absorption creates a detectable signature in the SAE decoder weight matrix --- specifically, absorbed features will have anomalously high cosine similarity between the "absorber" (specific feature) decoder vector and the "absorbed" (general feature) probe direction, and this signature can be detected without training probes by analyzing decoder-decoder cosine similarity clusters and their co-occurrence statistics.
- **Implementation sketch**: 
  1. Load pre-trained SAEs and compute the full decoder cosine similarity matrix W_dec @ W_dec^T.
  2. For each latent, identify its k-nearest-neighbors in decoder space.
  3. Compute co-occurrence statistics: for each pair of latents with high decoder similarity, measure how often they co-activate vs. how often one fires without the other (asymmetric co-occurrence).
  4. Define "absorption signature": latent A has high decoder similarity to latent B, A fires on a strict subset of B's expected inputs, and B's false-negative rate is correlated with A's activation.
  5. Validate against ground-truth absorption (from Candidate A's probe-based measurement) --- compute precision/recall of the unsupervised detector.
- **Simplest version**: Compute decoder cosine similarity matrix for one SAE, identify top-100 highest-similarity pairs, check if these pairs exhibit asymmetric co-occurrence patterns consistent with absorption (using the first-letter task as ground truth validation).
- **Time estimate**: ~1-2 hours. Decoder similarity matrix is O(d_sae^2 * d_model) which for 16k SAE is ~16k^2 * 2304 --- feasible on a single GPU. Co-occurrence computation requires a forward pass over ~10k tokens.
- **Reusable components**: SAELens (loading + encode), numpy/scipy (similarity computation), sae-spelling (ground-truth absorption labels for validation).

### Candidate C: Absorption-Hedging Disentanglement — Controlled Decomposition of SAE Failure Modes

- **Hypothesis**: A significant fraction (>30%) of what is currently measured as "feature absorption" in practice is actually "feature hedging" caused by incorrect L0, and the two failure modes have distinct observable signatures: absorption manifests as specific-features-absorbing-general-features (asymmetric, hierarchy-dependent), while hedging manifests as correlated-features-merging (symmetric, width-dependent). By varying SAE width and L0 independently on SAEBench SAEs, the two can be cleanly disentangled.
- **Implementation sketch**: 
  1. Use SAEBench's pre-trained SAEs which span 3 widths (4k/16k/65k) x 6 sparsities on Gemma-2-2B layer 12.
  2. For each SAE, measure: (a) absorption rate (sae-spelling metric), (b) feature hedging score (from feature-hedging-paper code), (c) L0 correctness (from sparse-but-wrong proxy metric).
  3. Build a 2D decomposition: plot absorption vs. hedging across width and L0, holding each constant while varying the other.
  4. Test predictions: if absorption is hierarchy-driven, it should be roughly constant across widths (at correct L0) but vary with feature hierarchy presence. If hedging is width-driven, it should decrease monotonically with width.
  5. Compute the "true absorption" rate = measured absorption - hedging-induced false negatives.
- **Simplest version**: Take the 6 sparsity levels at width 65k from SAEBench, compute absorption rate and hedging score for each, plot the relationship. If they are negatively correlated (as the balanced Matryoshka paper suggests), this is already a novel quantitative result.
- **Time estimate**: ~2-3 hours (SAEBench SAEs are pre-trained; metric computation is the bottleneck). Pilot: 30 min (3 SAEs at one width).
- **Reusable components**: SAEBench SAEs (pre-trained weights), sae-spelling (absorption metric), feature-hedging-paper (hedging metric), sparse-but-wrong-paper (L0 analysis).

---

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption Taxonomy

- **Implementation reality check**: The sae-spelling absorption metric is tightly coupled to the first-letter task (it uses alphabetical token filtering, spelling grader, etc.). Adapting it to RAVEL-style entity-attribute pairs requires significant refactoring: new probe targets, new token sets, new "false-negative" definitions. However, the core logic (probe → find false negatives → integrated gradients on absorbing latent) is architecture-agnostic. RAVEL already provides the entity-attribute data and template prompts. The main risk is that entity-attribute hierarchies may not exhibit the same clean parent-child structure as first-letter features in the residual stream --- city features and country features may not even be linearly represented at the same layer. Mitigation: RAVEL has already been validated on SAE features in SAEBench, showing that SAE latents do encode entity attributes. Ferrando et al. (ICLR 2025) confirmed entity-type features exist in Gemma-2 SAEs.
- **Reproducibility attack**: The probe-based absorption metric is well-defined and deterministic (given probe weights). The main reproducibility risk is in probe quality --- if probes are poor, false-negative rates are meaningless. Mitigation: report probe accuracy alongside absorption rates; use k-sparse probing (from sae-spelling) which gives a continuous measure of probe quality.
- **Baseline sanity check**: The strongest "baseline" here is Chanin et al.'s first-letter absorption results. If entity-attribute absorption rates are similar to first-letter rates (~15-35%), the result is confirmatory but not groundbreaking. The novel contribution is the cross-domain comparison and the identification of hierarchy properties that predict absorption severity. If absorption rates are *very different* across domains, that is more interesting.
- **Scope attack**: Tested on one model (Gemma-2-2B), one layer (12), and a handful of hierarchy types. Generalization to other models/layers is uncertain. Mitigation: include at least 2 layers and note that SAEBench SAEs cover both Pythia-160M and Gemma-2-2B.
- **Verdict**: **STRONG**. Novel, directly addresses the most cited gap in the literature (Gap 2), uses existing tools and data, and the simplest version is a clear contribution. The adaptation effort is moderate but manageable.

### Against Candidate B: Decoder Geometry as Unsupervised Absorption Detector

- **Implementation reality check**: Computing the full 16k x 16k decoder cosine similarity matrix is straightforward (~250M dot products, ~10 seconds on GPU). The challenge is defining what "absorption signature" looks like in decoder space without circular reasoning. If we need ground-truth absorption labels to calibrate the detector, its "unsupervised" nature is weakened --- it becomes more of a "semi-supervised" approach. Additionally, high decoder cosine similarity between two latents could indicate many things besides absorption (feature splitting, redundancy, polysemanticity). The signal-to-noise ratio may be too low for a reliable unsupervised detector.
- **Reproducibility attack**: The decoder similarity computation is fully deterministic. The risk is in the threshold choices (what cosine similarity counts as "high"? what co-occurrence asymmetry counts as "absorption"?). These hyperparameters could be tuned to look good on the first-letter validation set but fail elsewhere.
- **Baseline sanity check**: Tian et al. (2025) frame absorption as poor feature sensitivity, which is already a more general framing. The "geometry of concepts" paper studies decoder similarity structure but does not connect it to absorption. So the specific claim (decoder geometry predicts absorption) is novel, but it is unclear how much better this would be than simply looking at feature sensitivity scores.
- **Scope attack**: If the unsupervised detector only works on the first-letter task (because that is the only validation set), it has not actually generalized. To be credible, it needs to predict absorption on a second task where ground truth is independently established --- which loops back to Candidate A.
- **Verdict**: **MODERATE**. Interesting idea but high risk of either (a) requiring so much calibration that it is not really unsupervised, or (b) having too much noise to be useful. Best as a *component* of a larger paper, not the main contribution.

### Against Candidate C: Absorption-Hedging Disentanglement

- **Implementation reality check**: All three metric implementations exist (sae-spelling, feature-hedging-paper, sparse-but-wrong-paper). The SAEBench pre-trained SAEs provide the controlled width x sparsity grid. The main challenge is that the hedging metric and absorption metric may not be directly comparable --- hedging is defined on correlated feature pairs while absorption is defined on hierarchical feature pairs. Disentangling them requires a clear operational definition of which failure mode a given false-negative belongs to. This conceptual work is non-trivial but manageable.
- **Reproducibility attack**: High reproducibility --- all metrics are deterministic, all SAEs are public, all code is open-source. The main judgment call is in the attribution of false negatives to absorption vs. hedging, which needs to be clearly defined and validated.
- **Baseline sanity check**: The balanced Matryoshka paper (Chanin et al., 2025) already shows absorption and hedging trade off, but does not provide a quantitative decomposition. The "sparse but wrong" paper shows L0 confounds feature quality. Our contribution would be the first systematic decomposition across the full SAEBench grid. This is valuable but incremental.
- **Scope attack**: Limited to SAEBench SAEs on two models. The decomposition may not generalize to other training setups or models. However, SAEBench is the community standard, so results on SAEBench are inherently valuable.
- **Verdict**: **MODERATE**. Solid engineering contribution but risks being seen as "just running existing metrics on existing SAEs." Needs a strong narrative about *why* the decomposition matters (e.g., it changes which mitigation strategy you should use).

---

## Phase 4: Refinement

### Dropped Ideas

None dropped outright. Candidate B is too risky as a standalone paper but valuable as a supporting analysis.

### Strengthened Survivors

**Front-runner: Unified paper combining Candidates A and C, with Candidate B as supporting analysis.**

The strongest paper is one that:
1. **Extends absorption measurement to rich semantic hierarchies** (Candidate A core) --- this is the primary novel contribution.
2. **Disentangles absorption from hedging across the SAEBench grid** (Candidate C core) --- this provides the controlled analysis backbone.
3. **Proposes decoder-geometry-based absorption indicators** (Candidate B, simplified) --- this provides a practical tool, validated against the expanded ground truth from (1).

Simplification decisions:
- **Drop the unsupervised claim** for Candidate B. Instead, frame it as "a lightweight absorption indicator that requires only decoder weights and co-occurrence statistics, no task-specific probes." This is still useful and more honest.
- **Focus on RAVEL hierarchies** for Candidate A (city-country, city-continent, city-language) rather than constructing custom hierarchy datasets. RAVEL is established, peer-reviewed, integrated into SAEBench. No need to reinvent data.
- **Use SAEBench SAE grid** for Candidate C rather than training any new SAEs. This keeps the project strictly training-free.

### Pilot experiment design (< 15 min)

1. Load Gemma-2-2B and one Gemma Scope SAE (layer 12, 16k width) via SAELens.
2. Load RAVEL city-country dataset from HuggingFace.
3. Train a logistic regression probe for "country" attribute on model activations at the city-token position.
4. Encode the same activations with the SAE and find the top-k latents most aligned with the country probe direction (cosine similarity of decoder vectors with probe weight vector).
5. Measure false-negative rate: tokens where probe correctly classifies country but none of the top-k SAE latents activate.
6. If false-negative rate > 10%, there is signal for absorption in this domain. If close to 0%, the first-letter task may be a special case.

**Expected outcome**: Based on Ferrando et al. showing entity-type features exist but are imperfect in SAEs, and DeepMind's finding of 10-40% degradation with SAE-reconstructed activations, I expect 15-30% false-negative rates, comparable to first-letter absorption.

### Selected Front-Runner

**"Beyond First Letters: A Systematic Taxonomy of Feature Absorption Across Semantic Hierarchies in Sparse Autoencoders"**

This combines the cross-domain extension (strongest novelty), the absorption-hedging decomposition (methodological rigor), and the decoder-geometry indicator (practical utility) into a single coherent story.

---

## Phase 5: Final Proposal

### Title

Beyond First Letters: A Systematic Taxonomy of Feature Absorption Across Semantic Hierarchies in Sparse Autoencoders

### Hypothesis

Feature absorption in SAEs is not specific to the first-letter spelling task but is a general phenomenon that occurs wherever features form hierarchical relationships. The absorption rate is predictable from three measurable properties of the feature hierarchy: (1) parent-child co-occurrence frequency, (2) hierarchy depth, and (3) the specificity ratio (child frequency / parent frequency). Furthermore, a significant fraction of observed false negatives attributed to absorption in existing evaluations is actually caused by feature hedging due to incorrect L0, and these two failure modes can be disentangled through controlled analysis across SAE width and sparsity.

### Motivation

Feature absorption is the most consequential failure mode of SAEs because it creates a false sense of interpretability: features appear monosemantic but have systematic blind spots. Chanin et al. (2024) defined and measured absorption, but only on the first-letter spelling task --- a narrow, syntactic proxy that may not represent the semantically rich hierarchies (entity types, knowledge relations) that matter for safety and circuit analysis applications. Meanwhile, DeepMind deprioritized SAE research partly because SAE probes dramatically underperform dense probes on safety-relevant tasks, with absorption suspected as a key culprit. To make SAEs reliable for mechanistic interpretability, we need to understand: (a) how severe absorption is across different types of feature hierarchies, (b) what hierarchy properties predict absorption severity, and (c) how to distinguish absorption from the confounding effect of feature hedging. No existing work provides this systematic characterization.

### Method

**Phase 1: Cross-Domain Absorption Measurement**

Extend the Chanin et al. absorption metric to four semantic hierarchy types, all using pre-trained Gemma Scope SAEs on Gemma-2-2B (training-free):

| Hierarchy Type | Parent Feature | Child Feature | Data Source |
|---|---|---|---|
| Syntactic (baseline) | "starts with letter X" | specific token identity | sae-spelling (first-letter task) |
| Geographic-Political | country | city | RAVEL (HuggingFace) |
| Geographic-Regional | continent | city | RAVEL |
| Linguistic | language | city | RAVEL |

For each hierarchy:
1. Train logistic regression probes for parent and child features on Gemma-2-2B residual stream activations (layers 6, 12, 18, 24).
2. Using SAELens, encode activations with pre-trained SAEs (JumpReLU 16k/65k from Gemma Scope, BatchTopK/TopK from SAEBench).
3. Identify "split latents" via k-sparse probing: find the k SAE latents that best reconstruct the probe direction.
4. Compute absorption rate using the adapted Chanin et al. metric: false-negative tokens where probe succeeds but all split latents fail, with integrated-gradients attribution confirming a different latent is responsible.
5. Measure hierarchy properties: co-occurrence frequency (P(child | parent)), specificity ratio, hierarchy depth.

**Phase 2: Absorption-Hedging Decomposition**

Using SAEBench's pre-trained SAE grid (3 widths x 6 sparsities x 3 architectures on Gemma-2-2B layer 12):
1. Compute absorption rate (from Phase 1 metric) and hedging score (from Chanin et al. 2025 feature-hedging metric) for each SAE.
2. Compute L0 correctness using the sparse-but-wrong proxy metric.
3. Build a 2D decomposition: absorption = f(hierarchy, width, L0) and hedging = g(correlation, width, L0).
4. Test key predictions:
   - Absorption should be constant across widths at correct L0 (because it is hierarchy-driven, not width-driven).
   - Hedging should decrease with width (because wider SAEs can represent more correlated features separately).
   - At too-low L0, both metrics should worsen, but hedging should worsen disproportionately.

**Phase 3: Decoder-Geometry Absorption Indicator**

1. For each SAE, compute the decoder cosine similarity between each latent's decoder vector and each probe direction.
2. For high-similarity latent-probe pairs, compute the asymmetric co-occurrence ratio: how often the latent fires when the probe's feature is present vs. absent.
3. Define "absorption indicator score" = decoder_cosine_similarity x co-occurrence_asymmetry x (1 - latent's own probe correlation). High scores indicate latents that are geometrically aligned with a feature but fire asymmetrically and are not the "correct" latent for that feature.
4. Validate: compare absorption indicator rankings against ground-truth absorption labels from Phase 1.

### Simplest Version

The absolute minimum experiment that tests the core claim:
- Load one Gemma Scope SAE (Gemma-2-2B, layer 12, 16k JumpReLU) via SAELens.
- Load RAVEL city-country dataset.
- Train country probe. Compute absorption rate using adapted sae-spelling metric.
- Compare absorption rate to the published first-letter absorption rate for the same SAE.
- If the rates are significantly different (in either direction), this is already a publishable finding that demonstrates absorption is domain-dependent.

**Time: ~30-45 minutes.**

### Baselines

1. **First-letter absorption rate** (Chanin et al., 2024): 15-35% across Gemma Scope SAEs. Our RAVEL-based rates should be compared directly.
2. **Dense probe performance** (no SAE): Logistic regression probe accuracy on the raw residual stream, without SAE encoding. This gives the "ceiling" --- if absorption were zero, SAE-based probe performance should match dense probe performance. The gap quantifies total information loss, of which absorption is one component.
3. **Random latent baseline**: Assign random SAE latents to each probe direction and measure the "absorption rate" --- this gives the noise floor.

### Experimental Plan

| Experiment | Models | SAEs | Metrics | GPU-hours | Wall-clock |
|---|---|---|---|---|---|
| Pilot: RAVEL city-country absorption | Gemma-2-2B | 1 JumpReLU 16k | Absorption rate, probe accuracy | 0.25 | 15 min |
| Full Phase 1: 4 hierarchies x 4 layers x 4 SAEs | Gemma-2-2B | 4 (JumpReLU 16k/65k, BatchTopK 16k/65k) | Absorption rate, co-occurrence freq, specificity ratio | 4 | 3-4 hours |
| Phase 2: SAEBench grid decomposition | Gemma-2-2B | 18 (3 widths x 6 sparsities, TopK arch) | Absorption rate, hedging score, L0 correctness | 3 | 2-3 hours |
| Phase 3: Decoder geometry validation | Gemma-2-2B | 4 (from Phase 1) | Absorption indicator precision/recall | 1 | 1 hour |
| Cross-model validation | Pythia-160M | 6 (SAEBench SAEs) | Phase 1 absorption rates | 1 | 1 hour |
| **Total** | | | | **~9** | **~8-10 hours** |

All experiments are training-free (no SAE training). Compute is dominated by forward passes through the LM and SAE encoding. A single 24GB GPU (RTX 3090/4090 or A100) is sufficient.

### Resource Estimate

- **GPU**: 1x 24GB GPU (confirmed sufficient by SAEBench documentation)
- **GPU-hours**: ~9 total across all experiments
- **Wall-clock time**: ~8-10 hours total, split across multiple independent experiment runs
- **Model sizes**: Gemma-2-2B (2.6B params, ~5GB in float16), Pythia-160M (160M params, ~0.3GB)
- **SAE sizes**: 16k latents (d_model=2304, so ~74M params per SAE, ~150MB), 65k latents (~600MB)
- **Disk**: ~5GB for cached activations, ~2GB for SAE weights
- **No SAE training required** --- all analysis uses pre-trained SAEs from Gemma Scope and SAEBench

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Entity-attribute features not linearly represented at target layers | High | Low | RAVEL + Ferrando et al. confirm they are. Include multiple layers. |
| Absorption metric adaptation breaks on non-first-letter tasks | Medium | Medium | The core metric (probe → false negatives → integrated gradients) is task-agnostic. Main adaptation is defining "correct" latents via probe direction alignment, which generalizes. |
| Absorption rates are nearly identical across all hierarchy types | Medium | Medium | Even a null result (absorption is universal and domain-independent) is publishable and useful. Frame as "absorption is a fundamental SAE property, not task-specific." |
| SAEBench SAEs not sufficient for hedging decomposition | Medium | Low | SAEBench provides 18 SAEs on the exact width x sparsity grid needed. If insufficient, Gemma Scope provides 400+ additional SAEs. |
| Integrated gradients attribution noisy on entity features | Medium | Medium | Use larger token sets (RAVEL has 3000+ cities). Report confidence intervals. Fall back to decoder cosine similarity as a proxy if IG is too noisy. |
| Computational bottleneck on 65k SAEs | Low | Low | 65k x 2304 dot products are fast on GPU. Batch processing handles memory. SAEBench confirms 65k SAEs run on 24GB GPUs. |

### Novelty Claim

This paper makes three novel contributions:

1. **First cross-domain absorption characterization**: We are the first to measure feature absorption on semantically rich hierarchies (entity-attribute, geographic-political) beyond the first-letter spelling task. This directly addresses Gap 2 (the most frequently cited gap) in the feature absorption literature.

2. **First systematic absorption-hedging decomposition**: We provide the first quantitative disentanglement of feature absorption from feature hedging across the SAEBench evaluation grid, showing what fraction of observed SAE failure is due to each mechanism and how this depends on SAE width, sparsity, and feature hierarchy properties. This addresses Gaps 3 and 9.

3. **Decoder-geometry absorption indicator**: We introduce a lightweight method for detecting likely absorption candidates using only decoder weight cosine similarity and co-occurrence statistics, validated against ground-truth absorption labels across multiple semantic domains. This partially addresses Gap 7 (unsupervised detection).

The novelty is primarily in the *systematic empirical characterization* rather than in proposing a new architecture or training method. This is appropriate because the field's most pressing need is understanding *when and why* absorption occurs, which must precede principled mitigation. The paper also has high practical value: it tells practitioners which types of features are most vulnerable to absorption and provides a diagnostic tool (decoder-geometry indicator) that does not require constructing bespoke probe tasks.

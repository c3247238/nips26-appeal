# Backup Ideas for Pivot

## Alternative A: Scaling Laws for Feature Absorption Rate

**Status:** Backup (high potential, can pivot if primary yields null/weak results on replication)

**Core hypothesis:** The absorption rate A(N, k, L) as a function of SAE dictionary size N, sparsity level k, and layer depth L follows a power law with exponential saturation: A(N) = alpha_0 * N^{beta_N} * k^{beta_k} * exp(-gamma * N / N^*(L)) where N^*(L) is a critical dictionary size depending on layer depth.

**Why pivot here:** If the primary study's H1b finding (significant delta-corrected correlation at layer 8) fails to replicate on Gemma or Pythia, the field would still benefit from understanding how absorption scales. This would be a natural extension that uses the same experimental infrastructure.

**Key prediction:** On Gemma Scope SAEs across dictionary sizes (16K to 1M), absorption rate vs. dictionary size follows an inverted-U curve: initial power-law increase followed by exponential decrease for N > N^*.

**Resources needed:** Same as primary (pre-trained SAEs, SAEBench metric). Additional SAE configurations may be needed (more dictionary sizes).

**Novelty:** No existing work proposes or validates a scaling law for absorption rate. SAEBench includes absorption metrics but does not study scaling behavior.

---

## Alternative B: A Validated, Generalized Metric for Feature Absorption

**Status:** Backup (methodological contribution, can pivot if primary is confounded by metric issues)

**Core hypothesis:** The standard Chanin et al. absorption metric systematically underestimates true absorption rates by missing (1) partial absorption cases, (2) distributed absorption where multiple child latents share compensation, and (3) non-orthographic hierarchical features. An enhanced metric incorporating these cases detects 2-3x more absorption instances while maintaining zero false positives on non-hierarchical features.

**Why pivot here:** If the primary study's delta-corrected metrics are inconsistent across models, it may be because the absorption metric itself is flawed. Validating and improving the metric would be a foundational contribution that strengthens all subsequent absorption research.

**Key experiments:**
1. Implement enhanced metric with partial absorption detection, multi-latent compensation tracking, and semantic hierarchy generalization
2. Validate against SynthSAEBench ground truth
3. Test on 50+ pre-trained SAEs from Gemma Scope, Llama Scope, and SAEBench

**Resources needed:** Same infrastructure plus SynthSAEBench synthetic data.

**Novelty:** First validation of the absorption metric itself against ground truth; first generalization beyond first-letter tasks.

---

## Alternative C: Feature Absorption as Optimal Compression --- A Trade-off Analysis

**Status:** Backup (contrarian reframing, STRONG CANDIDATE for pivot if primary fails on replication)

**Core hypothesis:** Feature absorption is not a pathology to be eliminated but a predictable consequence of the interpretability-compression trade-off. Architectural interventions that reduce absorption (orthogonality constraints, hierarchical structures) increase other pathologies (dead neurons, feature composition, reconstruction error) in a quantifiable trade-off.

**Why pivot here:** The primary study's evidence already partially supports this view --- absorption coexists with functional steering and perfect probing precision. If H1b fails to replicate on other models, the contrarian reframing becomes the dominant interpretation. The data shows that high-absorption features achieve 100% steering success, directly supporting the "absorption is benign" view.

**Key experiments:**
1. Compare standard SAEs vs. OrtSAE/Matryoshka SAEs on the same model/layer
2. Measure absorption reduction AND increases in other pathologies
3. Fit a Pareto frontier showing the interpretability-compression trade-off
4. Compute effective degrees of freedom (Bereska et al., 2025) for SAEs with different absorption rates

**Resources needed:** Same infrastructure plus access to OrtSAE/Matryoshka checkpoints (available open-source).

**Novelty:** First quantification of the absorption-pathology trade-off; first information-theoretic framework connecting absorption to compression efficiency.

**Why this is the strongest backup:**
1. The data already supports it --- absorption coexists with functional steering and perfect precision
2. Lower risk than primary --- descriptive claim (trade-offs exist) vs. predictive claim (absorption degrades tasks)
3. Same infrastructure --- uses GPT-2 SAEs, same codebase
4. Fresh intellectual angle --- the field treats absorption as a pathology; reframing it as a trade-off is distinct
5. H4 (EC50) NOT SUPPORTED further strengthens the "absorption is benign" view

---

## Alternative D: The Lateral Inhibition Lens --- Post-Hoc Absorption Repair

**Status:** Backup (innovator's high-risk/high-reward idea, can pivot after primary establishes problem magnitude)

**Core hypothesis:** Feature absorption is caused by excessive fixed-strength lateral inhibition (the sparsity penalty) that lacks the homeostatic regulation present in biological sparse coding circuits. By modeling SAE activation dynamics as a competitive neural network, we can construct an "inhibition graph" and apply a "homeostatic repair" that rebalances activations to restore general feature firing.

**Why pivot here:** If the primary study establishes that absorption has significant task impact (H1b replicates and strengthens), the natural next step is to develop a repair mechanism. The innovator's neuroscience-inspired framework provides a unique angle that no prior work has explored.

**Key experiments:**
1. Construct inhibition graph from SAE encoder-decoder Jacobian
2. Apply homeostatic repair: boost suppressed features based on absorption dominance
3. Measure repair efficacy: reduction in absorption score, improvement in feature completeness, preservation of reconstruction quality
4. Test on steering tasks with repaired SAEs

**Resources needed:** Same infrastructure. No additional training required.

**Novelty:** First post-hoc (training-free) diagnostic and repair for feature absorption; first quantitative "inhibition graph" for SAE feature interactions; first "homeostatic repair" mechanism inspired by biological neural plasticity.

---

## Alternative E: Cross-Model Replication and Scale Threshold

**Status:** Backup (empirical follow-up, can pivot if primary needs stronger generalizability)

**Core hypothesis:** The absorption-degradation relationship generalizes across model families but only emerges above a model scale threshold (~1B parameters). Smaller models (GPT-2 Small, 124M; Pythia-70M, 19M) show minimal absorption prevalence and no task degradation, while larger models (Gemma-2-2B, 2B) exhibit stronger absorption and measurable degradation.

**Why pivot here:** The primary study's single-model limitation (GPT-2 Small only) is its biggest weakness. Pythia-70M cross-validation was inconclusive due to model size. A cross-model replication would dramatically strengthen the paper's contribution and test the "scale threshold" hypothesis.

**Key experiments:**
1. Replicate identical protocol on Gemma-2-2B (layers 8, 12, 16, dict sizes 16K, 65K)
2. Replicate on Pythia-160M and Pythia-2.8B for scale sweep
3. Compare absorption prevalence and task degradation across scales
4. Test whether H1b (delta-corrected correlation) strengthens with model size

**Resources needed:** Gemma-2-2B requires HF authentication; Pythia is openly available. ~15-20 GPU hours total.

**Novelty:** First cross-model study of absorption-task degradation; first test of scale threshold hypothesis.

---

## Pivot Decision Tree (Updated)

```
Current Study Results (H1b significant at layer 8, H5 supported, H4 NOT SUPPORTED):
|
|-- H1b REPLICATES on Gemma/Pythia with significance:
|   --> PROCEED with current framing + add cross-model validation
|   --> Strengthens paper significantly; targets main conference
|
|-- H1b FAILS on replication (null on other models):
|   --> PIVOT to Alternative C (trade-off analysis)
|   --> Data supports contrarian view; lower risk
|   --> H4 NOT SUPPORTED further strengthens "absorption is benign" view
|
|-- H1b significant but WEAK (R^2 < 0.2) on replication:
|   --> COMBINE primary + Alternative C
|   --> Emphasize methodological contribution (baseline correction + precision-recall)
|   --> Add trade-off analysis as secondary contribution
|
|-- Metric issues suspected (inconsistent across models):
|   --> PIVOT to Alternative B (metric validation)
|   --> Then re-run primary with improved metric
|
|-- Scale threshold confirmed (degradation only at >1B):
|   --> PIVOT to Alternative E (cross-model scale analysis)
|   --> Becomes the paper's main contribution
```

---

## Resource Comparison

| Alternative | GPU Hours | Risk | Novelty | Info Gain / GPU-hr | Best Trigger |
|---|---|---|---|---|---|
| A (Scaling laws) | ~15 | Medium | Medium | Medium | Primary fails; scaling patterns interesting |
| B (Metric validation) | ~11 | Low | Low-Medium | Medium | Metric inconsistency suspected |
| **C (Trade-off)** | **~10** | **Low-Medium** | **High** | **High** | **Primary fails on replication** |
| D (Repair) | ~8 | High | High | Medium | Primary establishes strong task impact |
| E (Cross-model) | ~20 | Medium | Medium | Medium | Need stronger generalizability |
| Null-result paper | ~0 | Low | Low | Low | All alternatives fail |

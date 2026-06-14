# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling (Chanin et al., github.com/lasr-spelling/sae-spelling)** — Canonical implementation of the absorption rate metric. MIT-licensed. Directly reusable. Contains: k-sparse probing, integrated-gradients ablation, first-letter absorption rate computation. This is the single most important starting point. Has code, has been peer-reviewed at NeurIPS 2025.

2. **SAELens (github.com/jbloomAus/SAELens, 1,100+ stars)** — Standard SAE training/evaluation library. MIT-licensed. Loads pre-trained Gemma Scope, GPT-2, Llama SAEs in 5 lines. Has `HookedSAETransformer` for seamless integration with TransformerLens. Active maintenance. This is the plumbing for any experiment in this space.

3. **Gemma Scope SAEs (huggingface.co/google/gemma-scope)** — 400+ pre-trained SAEs on Gemma 2 2B/9B, residual stream and MLP, widths 1k–1M, all layers. Training-free analysis becomes straightforward: just load, forward-pass, extract latent activations. No GPU-intensive training required.

4. **SAEBench (github.com/adamkarvonen/SAEBench, Apache 2.0)** — Standardized 8-metric evaluation suite. Absorption metric code included. 200+ SAEs pre-evaluated. Interactive results at neuronpedia.org/sae-bench. Key finding: absorption worsens with TopK/JumpReLU adoption — this is the empirical motivation for the entire research area.

5. **Chanin et al., arXiv:2409.14507 (NeurIPS 2025)** — Defines absorption, provides toy model proof, canonical metric. Has code in sae-spelling. Key limitation: only evaluates first-letter task. Key empirical fact: absorption rate 15–35% on Gemma Scope 16k/65k SAEs, present in all architectures. This is the direct prior work to build on and extend.

6. **Feature Hedging (Chanin et al., arXiv:2505.11756)** — Complementary failure mode to absorption. Important to understand because any solution to absorption may trade into hedging, and a controlled experiment must track both. Code availability: check paper; the same sae-spelling infrastructure likely applies.

7. **OrtSAE (Korznikov et al., arXiv:2509.22033)** — Achieves 65% absorption reduction via decoder orthogonality penalty. Important as a strong mitigation baseline. Also provides a richer evaluation: three absorption metrics (mean absorption fraction, full-absorption score, feature splits) from KronSAE are complementary to the Chanin metric.

8. **Matryoshka SAE (Bussmann et al., arXiv:2503.17547, ICML 2025)** — Best architecture on SAEBench absorption metric. Code: MIT-licensed (check for availability). Relevant because it directly reduces absorption by hierarchical training — but requires retraining, so out of scope for training-free analysis. Useful as a comparison baseline.

9. **DeepMind Safety Blog Post (deepmindsafetyresearch.medium.com)** — Practical motivation: SAE probes fail at harmful intent detection where dense linear probes succeed. This is the strongest external motivation for absorption research. No code, but the experimental setup (1-sparse SAE probe vs. dense linear probe on harmful intent detection) is straightforward to replicate.

10. **ATM-SAE (Li et al., arXiv:2510.08855)** — Achieves absorption score 0.0068 vs. TopK 0.1402 on Gemma-2-2B. Best reported absorption numbers. Requires retraining (out of scope) but sets the empirical target for how much improvement is achievable. Important as an upper-bound baseline.

11. **The Geometry of Concepts (Li et al., arXiv:2410.19750)** — Shows SAE feature co-occurrence structures cluster spatially. Directly relevant: provides theoretical grounding for why co-occurrence statistics are predictive of absorption. Code may be available (MIT-friendly research group).

12. **SynthSAEBench (arXiv:2602.14687)** — Synthetic benchmark with known ground-truth feature hierarchies. Could provide controlled experiments with known absorption ground truth — useful for validating detection methods without relying solely on the first-letter proxy.

### Landscape Summary

What actually works, based on engineering reality:

- The absorption rate metric from sae-spelling is the only validated, code-available measurement tool. It works but requires knowing which features to test (first-letter hierarchy is the only validated case). Running it on Gemma Scope SAEs takes ~15–30 minutes per SAE width on a single GPU.
- Matryoshka SAE reduces absorption but requires full retraining. Training a Matryoshka SAE from scratch would take several hours and likely violates the 1-hour-per-experiment budget. Not viable for this project's training-free constraint.
- OrtSAE achieves strong absorption reduction via a simple regularizer. Similarly requires retraining.
- ATM-SAE achieves the best absorption numbers but via a sophisticated training modification. Out of scope.
- Training-free analysis options: (a) measure absorption rate across existing pre-trained SAEs using the existing metric, (b) measure co-occurrence statistics and see if they predict absorption, (c) develop a post-hoc detection or partial mitigation that does not require retraining the SAE decoder.

The practical gap that is both novel and tractable: **no one has systematically measured how absorption rate varies as a continuous function of SAE configuration parameters (width, L0) using existing pre-trained SAEs across a controlled sweep.** The Chanin et al. paper plots absorption vs. L0 for some layers but does not extract a clean functional relationship. The SAEBench paper shows directional trends but does not measure absorption as a primary outcome. A systematic, training-free empirical characterization — measuring absorption rate across a grid of width × L0 × layer using Gemma Scope — is doable in <1 hour per (SAE, letter) pair, and no existing paper presents this clean result.

A second gap that is novel and tractable: **extending the absorption measurement to feature hierarchies beyond first-letter spelling, without introducing probe-training overhead.** The key observation is that the first-letter task was chosen because the ground-truth hierarchy (alphabet) is known in advance. But for entity-type hierarchies (mammal ⊃ dog ⊃ golden retriever) or syntactic hierarchies (verb ⊃ past-tense verb ⊃ irregular-past-tense verb), the same k-sparse probing + integrated gradients approach should work — we just need to construct the appropriate probe sets. This requires creativity in dataset construction, not additional training.

---

## Phase 2: Initial Candidates

### Candidate A: Systematic Absorption Rate Mapping Across SAE Configuration Space

- **Hypothesis**: Absorption rate follows a predictable functional relationship with SAE width (W) and mean sparsity (L0) that can be captured empirically as AR ≈ α · W^β · L0^γ using existing Gemma Scope SAEs, and this relationship differs systematically across model layers and SAE architectures. No such quantitative characterization currently exists.

- **Implementation sketch**: Starting from sae-spelling (canonical code), extend the absorption rate computation to run over a grid of Gemma Scope SAE configurations. The existing code already handles the measurement; we just need to loop over SAE configurations instead of a single one. The main additions: (a) batch the SAE loading across widths, (b) compute absorption rate across 5 layers × 4 widths (1k, 4k, 16k, 65k) × 2 architectures (L1, TopK) = 40 data points; (c) fit a log-linear regression to the resulting absorption vs. (W, L0) data.

- **Simplest possible version**: Measure absorption rate on 3 letters (A, B, C) across 4 SAE widths (1k, 4k, 16k, 65k) at a single layer (layer 12 of Gemma-2-2B) using both L1 and TopK architectures. Fit a log-linear regression. Time estimate: ~30 minutes per SAE × 8 SAEs = 4 hours of wall-clock compute, but can be parallelized to <1 hour per configuration. Pilot: single letter, single layer, 4 widths = ~15 minutes.

- **Time estimate**: ~4–8 GPU-hours total (parallelizable). Each individual measurement (1 letter, 1 layer, 1 SAE) takes ~10–20 minutes.

- **Reusable components**: 
  - sae-spelling: 100% reusable, exact same absorption rate computation
  - SAELens: loading Gemma Scope SAEs
  - Gemma Scope: all pre-trained SAEs already available
  - No new model training required

### Candidate B: Cross-Domain Absorption Characterization Using Controlled Hierarchy Probes

- **Hypothesis**: Feature absorption generalizes beyond first-letter spelling to semantically richer feature hierarchies (entity type hierarchy: ANIMAL ⊃ DOG, syntactic hierarchy: VERB ⊃ PAST-TENSE-VERB ⊃ IRREGULAR-PAST-TENSE, and knowledge hierarchy: COUNTRY ⊃ CAPITAL-CITY). The absorption rate on these richer hierarchies will be higher than on the first-letter task because the parent-child frequency gap is larger and the features are more semantically entangled.

- **Implementation sketch**: 
  1. Construct 3 new hierarchy probe datasets (entity types, syntax, knowledge facts) using templated sentences (e.g., "The [ANIMAL] lives in the wild." for ANIMAL probe; WordNet for entity type hierarchy).
  2. Train LR probes for parent and child concepts on model residual stream activations (≤10 minutes per concept on GPU).
  3. Reuse sae-spelling's k-sparse probing + integrated gradients pipeline to measure absorption rate on each hierarchy.
  4. Compare absorption rates across hierarchy types and correlate with observable hierarchy statistics (parent-child frequency ratio, conditional activation probability).

- **Simplest possible version**: One hierarchy (ANIMAL ⊃ DOG, with a clean probe dataset of 200 sentences per class) at a single layer of Gemma-2-2B with a single SAE width (16k). Measure absorption rate. If found, report as positive evidence; if not found, investigate why (possibly the hierarchy is not learned as expected at this layer). Time estimate: ~30 minutes end-to-end.

- **Time estimate**: ~3–5 GPU-hours total. Each new hierarchy type: ~1–2 GPU-hours for probe training + absorption measurement.

- **Reusable components**: 
  - sae-spelling pipeline (k-sparse probing + integrated gradients)
  - WordNet/ConceptNet for entity hierarchy construction
  - Penn Treebank or Universal Dependencies for syntactic hierarchy
  - Wikidata for knowledge hierarchy (country → capital)

### Candidate C: Strong Baseline Done Right — Do Linear Probes Actually Beat SAE Probes, and By Exactly How Much?

- **Hypothesis**: The DeepMind finding (SAE probes fail at harmful intent detection, dense linear probes succeed) is driven by feature absorption: specifically, the safety-relevant feature (harmful intent) is absorbed into context features that frequently co-occur with harmful content (e.g., roleplay framing, instruction-following tone). Quantifying this relationship precisely — measuring absorption rate of safety-relevant features and correlating it with the probe performance gap — is both novel and highly impactful.

- **Implementation sketch**: This is a "strong baseline done right" paper: measure the precise magnitude of the SAE-vs-linear-probe gap, decompose it into the fraction attributable to absorption vs. other causes, and correlate the absorption contribution with observable SAE statistics. Requires: (a) a dataset with labeled harmful vs. harmless instructions (can use AdvBench, or construct synthetic equivalents); (b) training dense linear probes on residual stream activations; (c) training k-sparse SAE probes; (d) measuring the precision/recall gap; (e) running the absorption metric on the "harmful intent" feature to quantify how much is absorbed.

- **Simplest possible version**: Use GPT-2-small (smaller model, SAEs available via EleutherAI) with 100 harmful/100 harmless examples. Train a dense probe and a 1-sparse SAE probe. Measure the gap. Identify the top 3 most absorbed latents for the "harmful" feature. Report. Time estimate: ~30 minutes end-to-end. This is purely analytical, no new training.

- **Time estimate**: ~2–4 GPU-hours total (mostly forward passes through model + SAE for activation collection).

- **Reusable components**: 
  - SAELens: loading SAEs and extracting features
  - sae-spelling: absorption rate metric
  - AdvBench or similar: harmful instruction dataset
  - sklearn: linear probes

---

## Phase 3: Self-Critique

### Against Candidate A

- **Implementation reality check**: The sae-spelling code is well-maintained and directly reusable. The main engineering challenge is the quadratic loop over (SAE configurations × letters × layers), but this is embarrassingly parallel. Risk: Gemma Scope SAEs for some widths/layers may not be available (check HuggingFace before committing). If not available for some widths, the scaling law fitting may have gaps. **Mitigation**: Check availability first (5 minutes); the 1k, 4k, 16k, 65k widths are documented as available for most residual stream layers.

- **Reproducibility attack**: The absorption metric uses integrated gradients with Captum, which is deterministic given the same model weights. The metric depends on k (number of features in k-sparse probing) and the threshold cosine similarity (0.025). These are fixed in the canonical implementation. Reproducing the result means running the same code on the same SAEs. Very high reproducibility.

- **Baseline sanity check**: The "baseline" here is "width does not affect absorption" (null hypothesis). Any monotonic relationship (confirmed by Chanin et al. directionally) would be a positive result. The novel contribution is quantifying this relationship precisely rather than observing it directionally.

- **Scope attack**: The result applies to Gemma Scope SAEs specifically (trained on Gemma-2-2B). Whether the same functional relationship holds for other models (GPT-2, Llama) is unknown. But establishing it for one well-characterized model family is sufficient for a publishable result; the cross-model validation becomes future work.

- **Verdict**: STRONG. The implementation path is clear, the code is reusable, and the result fills a genuine gap (Gap 1 in the literature survey). The main risk is that the relationship is noisy (not a clean power law) — but even a noisy empirical characterization is publishable and adds value.

### Against Candidate B

- **Implementation reality check**: Constructing the hierarchy probe datasets requires domain knowledge (what constitutes a good ANIMAL-vs-DOG hierarchy in the context of a language model). Risk: (1) The model may not have a clean "ANIMAL" feature at the residual stream level — it may represent this more diffusely. (2) The entity-type hierarchy may not be as cleanly absorbed as the first-letter hierarchy because the child-parent frequency ratio may be less extreme. Found after searching: no one has tried this extension yet, which means both the success and failure paths are publishable.

- **Reproducibility attack**: Probe training is stochastic (random initialization) but the absorption measurement itself is deterministic. Reporting mean ± std across 5 probe training runs is standard. The hierarchy datasets (WordNet, ConceptNet, Wikidata) are publicly available. Reproducibility is high.

- **Baseline sanity check**: The baseline is the Chanin et al. first-letter result. Any new hierarchy that shows absorption is an extension of the baseline. The surprising finding would be if absorption does NOT generalize — that would also be publishable (constrains where absorption matters).

- **Scope attack**: The entity-type and knowledge hierarchies are common in NLP benchmarks (TREC entity recognition, WikiData facts). If absorption is found there, the scope is much broader than the first-letter task. This is the key contribution.

- **Verdict**: STRONG. The implementation is straightforward using existing tools, the probe construction is laborious but not technically difficult, and the result (whether positive or negative) is novel and publishable.

### Against Candidate C

- **Implementation reality check**: The main engineering challenge is constructing a high-quality harmful intent dataset that is both diverse enough to be representative and small enough to fit within the compute budget. AdvBench is available but may be too narrow. ToxiGen, HarmBench, or similar datasets are also available. Risk: The safety-relevant features may not be cleanly localizable at a single SAE layer — harmful intent may be represented differently across layers. **Mitigation**: Start with a single layer (middle residual stream) and probe which layer has the best linear probe accuracy before running the absorption analysis.

- **Reproducibility attack**: Dense linear probes are highly reproducible (deterministic training if seed is fixed). The absorption metric is deterministic. The harmful intent dataset is publicly available. High reproducibility.

- **Baseline sanity check**: The DeepMind blog post provides the baseline: 1-sparse SAE probe fails, dense linear probe succeeds. Our contribution is to quantify how much of this gap is attributable to absorption specifically. This decomposition is the novel element.

- **Scope attack**: The result applies to harmful intent detection specifically. The broader implication (absorption limits SAE safety applications generally) is the high-impact claim. But we can only directly test this for the specific harmful intent task. Whether the same holds for bias detection, deception detection, etc., requires additional experiments.

- **Verdict**: MODERATE. The implementation is feasible, but the connection between measured absorption rate and probe performance gap may be loose (other factors besides absorption cause SAE probes to underperform). The contribution is strongest if we can show a direct causal link, not just a correlation. If the connection is correlational only, the paper is descriptive rather than causal.

---

## Phase 4: Refinement

### Dropped Ideas
No idea is dropped — all three received STRONG or MODERATE verdicts. However, the project constraint is "training-free analysis," which all three satisfy. The key question is priority.

### Strengthened Survivors

**Candidate A (Absorption Scaling Law) — Strengthened:**
- The main concern was that the relationship might be noisy. Strengthening: add a second diagnostic — correlation of absorption rate with co-occurrence conditional probability P(j fires | i fires) for the absorbing latent pair. This gives a mechanistic explanation for the scaling law (higher-frequency latents are more likely to absorb lower-frequency ones in proportion to their frequency ratio).
- Pilot experiment: Single letter, single layer, 4 widths — 15 minutes. If the trend is monotonic, the full experiment is worth running.
- Existing code from sae-spelling runs this in ~10 minutes per (SAE, letter, layer) combination on a single A100 GPU (estimated from the paper's description of their computational setup).

**Candidate B (Cross-Domain Hierarchy) — Strengthened:**
- Add a "negative control" hierarchy: measure absorption for a NON-hierarchical feature pair (e.g., "is past tense" vs. "contains the letter X") — these should show near-zero absorption even at the same frequency ratio. This controls for the possibility that any two co-occurring features show apparent absorption.
- The entity-type hierarchy (ANIMAL ⊃ DOG) is the most implementable because WordNet provides the hierarchy directly, and sentences can be constructed from the CommonCrawl data that the models were trained on.
- Pilot: 20 sentences per class, 1 layer, 1 SAE width. Check if the LR probe achieves >85% accuracy (indicating the feature is learned). If yes, run the full absorption measurement. Total: <15 minutes.

**Candidate C (Safety Implications) — Strengthened:**
- Reframe as a purely empirical analysis: do not claim to prove causation, just characterize the magnitude of the probe gap and the absorption rate of the relevant SAE latents. Let the data speak. This is more defensible than a causal claim.
- Use the existing ToxiGen or AdvBench dataset directly — no new data construction needed.
- Add a second metric: compare the F1 score of (a) 1-sparse SAE probe, (b) k-sparse SAE probe (k=10), (c) linear probe on residual stream. This shows whether more latents (a softer form of ensemble) compensates for absorption.

### Selected Front-Runner: Candidate A + Candidate B as Integrated Paper

**Rationale**: Candidates A and B are naturally complementary and together tell a complete story:
- Candidate A establishes the quantitative relationship between SAE configuration and absorption magnitude (answers "how bad is absorption as a function of SAE design choices?")
- Candidate B establishes that absorption generalizes beyond the first-letter task to semantically richer hierarchies (answers "does absorption matter for real-world feature hierarchies?")
- Together they provide the systematic analysis paper the field needs: a quantitative characterization of absorption across configuration space and feature hierarchy types.
- Candidate C can be included as a third section (motivational case study) showing the practical consequence of the measured absorption rates.

**The simplest version that would still be publishable**:
- Figure 1: Absorption rate vs. SAE width (4 data points on Gemma Scope, 1 layer, 5 letters) showing the trend
- Figure 2: Absorption rate on 3 hierarchy types (first-letter, entity-type, syntactic) at matched SAE configurations
- Figure 3: SAE probe vs. linear probe performance on a downstream task, with measured absorption rate as a mediating variable
- Tables: Report absorption rate across all tested (layer, width, architecture) configurations

This is achievable in 3–4 days of engineering work and 20–30 GPU-hours total.

---

## Phase 5: Final Proposal

### Title
Anatomy of Feature Absorption: Quantifying Absorption Severity Across SAE Configuration Space and Semantic Hierarchy Types

### Hypothesis
Feature absorption rate in pre-trained SAEs is (1) a predictable function of SAE width and mean sparsity (L0), following an approximately power-law relationship measurable from existing Gemma Scope SAEs; (2) present across semantically richer feature hierarchies beyond the first-letter spelling task (entity-type, syntactic, and knowledge-fact hierarchies), with absorption rate proportional to the parent-child feature frequency ratio; and (3) quantitatively linked to downstream performance gaps between SAE probes and dense linear probes on practical tasks. All three claims are testable using only existing pre-trained SAEs and the existing sae-spelling measurement infrastructure.

### Motivation
Feature absorption has been identified as the most consequential failure mode blocking practical SAE deployment (DeepMind 2025 deprioritization), but the field lacks two critical pieces of knowledge: (1) How severe is absorption as a function of SAE design choices? Current guidance ("absorption is worse for TopK at low L0" from SAEBench) is directional but not quantitative. (2) Does absorption generalize beyond the narrow first-letter spelling task that defines the canonical evaluation? Without quantitative characterization and cross-domain validation, the community cannot make principled decisions about when SAEs can and cannot be trusted for a given application.

This paper provides a systematic measurement-focused analysis that fills both gaps using purely training-free methods: forward passes through existing SAEs, co-occurrence statistics, and the validated sae-spelling measurement pipeline.

### Method
**Step 1: Absorption Sweep Across SAE Configuration Space** (Candidate A)
- Load Gemma Scope SAEs: widths {1k, 4k, 16k, 65k} × layers {8, 12, 16, 20} × architectures {L1, TopK} = 32 SAE configurations.
- For each SAE, run the sae-spelling absorption rate metric on letters {A, E, I, O, T} (5 letters chosen to cover diverse frequency ranges).
- Compute absorption rate for each (SAE, letter) pair. Estimated time: ~15–20 minutes per pair on a single A100 GPU, ~32 × 5 = 160 pairs total. With parallelization across 8 GPUs: ~4–5 hours total wall-clock.
- Fit log-linear regression: log(AR) = α + β·log(W) + γ·log(L0) + δ·Layer. Report coefficients and R².
- Test residuals for systematic patterns (layer-wise effects, architecture effects).

**Step 2: Cross-Hierarchy Absorption Measurement** (Candidate B)
- Construct probe datasets for 3 hierarchy types:
  - **Entity-type** (2 levels): ANIMAL ⊃ {DOG, CAT, BIRD, FISH}. Use 200 templated sentences per category from the model's training distribution. Verify LR probe accuracy >85% at layer 12 before running absorption measurement.
  - **Syntactic** (2 levels): PAST-TENSE ⊃ IRREGULAR-PAST-TENSE. Use Penn Treebank annotated sentences; irregular verbs (was/were, went, had) vs. regular (walked, talked). Verify probe accuracy >80%.
  - **Knowledge** (2 levels): EUROPEAN-COUNTRY ⊃ specific-country. Use Wikidata-sourced entity descriptions. Verify probe accuracy >75%.
  - **Negative control**: PAST-TENSE vs. CONTAINS-VOWEL-E (not hierarchically related, matched frequency ratio). Should show near-zero absorption.
- Apply sae-spelling k-sparse probing + integrated gradients pipeline to each hierarchy. Use Gemma Scope 16k at layer 12 (the most studied configuration).
- Report absorption rate per hierarchy type; compare to first-letter baseline.
- Correlate absorption rate with (a) parent-child frequency ratio, (b) conditional activation probability P(parent fires | child fires), (c) decoder cosine similarity between parent and child latents.

**Step 3: Downstream Consequence Analysis** (Candidate C, secondary)
- Use ToxiGen or AdvBench (200 harmful + 200 harmless instructions, balanced).
- Extract Gemma-2-2B residual stream activations at layer 12.
- Train: (a) dense linear probe (logistic regression on 2304-dim activations), (b) 1-sparse SAE probe (top-1 SAE latent by frequency), (c) k-sparse SAE ensemble probe (top-k latents, k=1,5,10,20).
- Measure: F1 score, precision, recall for each probe type.
- Run absorption metric on the top SAE latents for "harmful intent" feature: report what fraction are absorbed and which latents are doing the absorbing.
- Report: how much of the linear probe - SAE probe gap can be attributed to measured absorption (measured by comparing the gap before/after activating absorbed latents manually).

### Simplest Version (Pilot Experiment)
**15-minute pilot**: Single letter (A), single layer (layer 12), 4 SAE widths (1k, 4k, 16k, 65k) from Gemma Scope, L1 architecture only. Measure absorption rate for each. If absorption rate is monotonically increasing with width, the trend is confirmed and the full sweep is justified. Expected: 4 measurements × ~3 min each = ~12 minutes on a single GPU.

### Baselines
1. **Chanin et al. first-letter results**: Their reported absorption rate 15–35% on Gemma Scope 16k/65k is the exact baseline to reproduce and extend. We should exactly replicate their single data point before extending.
2. **"No systematic trend" null baseline**: Absorption rate is uniform across widths (flat regression). Our scaling law is novel only if the fitted relationship is stronger than the null hypothesis.
3. **Dense linear probe performance** (for Step 3): Taken directly from the DeepMind blog post numbers for reference; we replicate on our specific dataset.

**Expected performance ranges** (based on Chanin et al. empirical findings):
- Absorption rate at 16k width: 15–35% (matches their reported range)
- Absorption rate at 1k width: <10% (narrower SAE, less opportunity for absorption)
- Absorption rate at 65k width: 25–45% (wider SAE, more absorbing latents available)

### Experimental Plan
- **Datasets**: OpenWebText (20k tokens for co-occurrence statistics); templated probe datasets (200 sentences/class, all constructed); ToxiGen or AdvBench (400 examples).
- **Metrics**: Absorption rate (Chanin metric, primary); correlation coefficients for scaling law; F1/precision/recall for probe comparison; decoder cosine similarity (structural proxy).
- **Ablation schedule**: 
  1. Reproduce Chanin et al. on Gemma Scope 16k, layer 12, letter A. (validation)
  2. Sweep 4 widths at fixed layer 12. (scaling law)
  3. Sweep 4 layers at fixed width 16k. (layer-wise pattern)
  4. Entity-type hierarchy at Gemma Scope 16k, layer 12. (cross-domain)
  5. Syntactic + knowledge hierarchies. (cross-domain extension)
  6. Downstream probe comparison on harmful intent. (safety implication)
- **Models**: Gemma-2-2B (primary, via Gemma Scope); GPT-2-small (replication/reproducibility check, via EleutherAI SAEs).

### Resource Estimate
- **Wall-clock time**: ~6–8 hours total, parallelizable to <1 hour per individual experiment.
- **GPU-hours**: ~20–30 A100-hours total.
  - Step 1 (scaling sweep): 160 absorption measurements × ~15 min each / parallelization = ~12 A100-hours
  - Step 2 (cross-hierarchy): 4 hierarchy types × 2 hours each = ~8 A100-hours
  - Step 3 (downstream): ~4 A100-hours for activation collection + probe training
- **Model sizes**: Gemma-2-2B (2.5B parameters, fits in ~6GB VRAM in bfloat16) + SAE (at most 65k × 2304 = 150M parameters = ~0.6GB additional). Comfortably fits on a single 16GB GPU.
- **No training required**: All SAEs loaded from HuggingFace. Probe training (LR) is minutes on CPU.
- **Storage**: Gemma Scope SAE weights (~10GB total for 4 widths); activation caches for 20k tokens (~4GB).

### Risk Assessment

**Engineering risks:**
1. *Gemma Scope SAE availability at all widths*: Check upfront. If 1k or 131k SAEs are not available for certain layers, use only available widths. The scaling law can still be fitted on 3–4 data points.
   - Mitigation: Download and check availability in the pilot. Gemma Scope HuggingFace page lists all available configurations.
2. *sae-spelling code compatibility with latest SAELens*: The canonical code was written for an older SAELens version. May require version pinning.
   - Mitigation: Pin to `saelens==0.5.x` (the version used in the paper); check the sae-spelling README for installation instructions.
3. *Integrated gradients computation may fail at early layers*: The Chanin et al. paper notes that the absorption metric using integrated gradients has limited applicability at very early layers (attribution becomes unreliable). 
   - Mitigation: Restrict Step 1 sweep to layers 8–20; skip layers 0–7.
4. *Entity-type hierarchy probe accuracy below threshold*: The model may not represent ANIMAL vs. DOG in a linearly separable way at a single residual stream position.
   - Mitigation: Test probe accuracy at layers 8, 12, 16 and use the layer with highest probe accuracy.

**Scientific risks:**
1. *Absorption rate scaling law may not be a clean power law*: The relationship may be non-monotonic or noisy due to different training procedures for different Gemma Scope SAE widths.
   - Mitigation: Report empirical trend regardless of functional form; even a descriptive characterization is publishable.
2. *Cross-domain absorption rates may be similar to first-letter (uninformative)*: If all hierarchy types show 20–30% absorption, the cross-domain comparison adds limited insight.
   - Mitigation: Include the negative-control hierarchy (non-hierarchical feature pair) to establish that the metric is sensitive to hierarchy structure. If all hierarchies show similar rates, this itself is a finding (absorption is not hierarchy-type-specific).
3. *Safety implication section may be underpowered*: 400 examples for harmful intent detection may be too few for a reliable probe evaluation.
   - Mitigation: Supplement with existing benchmark results from the literature; use the step primarily as illustrative case study rather than primary evidence.

### Novelty Claim

**What is new**: 
1. First quantitative characterization of absorption rate as a continuous function of SAE configuration parameters (width, L0) using a controlled sweep of 32+ SAE configurations — no existing paper provides a clean empirical relationship.
2. First measurement of feature absorption on semantically richer hierarchies beyond first-letter spelling — extends the phenomenon's scope and tests whether it generalizes to features of safety relevance.
3. First empirical quantification of how measured absorption rate mediates the SAE probe vs. linear probe performance gap — bridges the abstract measurement (absorption rate) to its practical consequence (probe reliability).

**Why this is publishable even without a "flashy" insight**: The field's current limitation is a measurement gap — the community knows absorption exists and is bad, but does not know its precise magnitude as a function of design choices. A systematic, clean measurement paper is exactly what is needed to enable principled decisions about SAE architecture selection. This mirrors the contribution of the SAEBench paper: not algorithmic novelty, but the measurement infrastructure that enables the field to make progress.

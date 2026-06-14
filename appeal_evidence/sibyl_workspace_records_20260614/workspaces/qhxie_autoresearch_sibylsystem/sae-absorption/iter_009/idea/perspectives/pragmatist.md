# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (https://github.com/lasr-spelling/sae-spelling) -- MIT license. Canonical absorption metric implementation. Contains `sae_spelling.feature_absorption_calculator`, `sae_spelling.feature_attribution`, k-sparse probing, and ablation code. Directly reusable. Runs via `poetry run python -m sae_spelling.experiments.feature_absorption`. **Code exists and is runnable.**

2. **SAEBench** (https://github.com/adamkarvonen/SAEBench) -- Apache 2.0. 8-metric evaluation suite including absorption. 200+ pretrained SAEs across 7 architectures (4k/16k/65k widths, 6 sparsities) on Pythia-160M layer 8 and Gemma-2-2B layer 12. Install: `pip install -e . -c constraints.txt`. Total eval time ~115 min per SAE on RTX 3090. **Code exists, pip-installable, pretrained SAEs available.**

3. **SAELens v6** (https://github.com/decoderesearch/SAELens) -- MIT. Standard SAE library. `SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_12/width_16k/canonical")` loads any Gemma Scope SAE in 3 lines. Deep TransformerLens integration for activation caching. **Code exists, mature, pip-installable.**

4. **Gemma Scope SAEs** (https://huggingface.co/google/gemma-scope) -- 400+ JumpReLU SAEs on Gemma 2 2B/9B/27B, all layers, widths 1k--1M. Gemma Scope 2 adds Matryoshka SAEs + transcoders on Gemma 3. **Pretrained weights available, no training needed.**

5. **SAEBench pretrained SAEs** -- `saebench_gemma-2-2b_width-2pow{12,14,16}_date-010{7,8}` releases with 7 architecture variants per width/sparsity combo. Includes training checkpoints. **Directly loadable via SAELens.**

6. **TransformerLens** (https://github.com/TransformerLensOrg/TransformerLens) -- MIT. Hook-based activation caching, 50+ model support. `model.run_with_cache()` gives activation tensors. `model.run_with_saes()` for integrated SAE forward passes. **Code exists.**

7. **Chanin et al. 2024, "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) -- Defines feature absorption, proves it in toy model, provides absorption rate metric. Key limitation: only validated on first-letter spelling task. Absorption rate 15--35% across Gemma Scope SAEs. **Paper + code, the baseline to build on.**

8. **Chanin et al. 2025, "Feature Hedging"** (arXiv:2505.11756, https://github.com/chanind/feature-hedging-paper) -- Identifies hedging as complementary failure mode. Shows absorption and hedging trade off. Balanced Matryoshka SAE with compound multiplier ~0.75. **Paper + code, key for disentangling confounds.**

9. **Chanin & Garriga-Alonso 2025, "Sparse but Wrong"** (arXiv:2508.16560, https://github.com/chanind/sparse-but-wrong-paper) -- Shows most open-source SAEs have incorrect L0, causing feature mixing. Proposes proxy metric for correct L0. **Paper + code, essential for controlling L0 confound.**

10. **Matryoshka SAE** (arXiv:2503.17547, ICML 2025) -- Absorption rate ~0.03 vs BatchTopK ~0.29. Best current mitigation. Available in SAEBench suite. **Pretrained weights in SAEBench.**

11. **DeepMind negative results blog** (March 2025) -- Dense linear probes get AUROC 1.0 on harmful intent detection; SAE probes fail. 10--40% degradation with SAE-reconstructed activations. Motivates why absorption matters for safety. **Public blog post, quantitative gap documented.**

12. **SynthSAEBench** (arXiv:2602.14687, Feb 2026) -- Synthetic benchmark with known ground-truth features including hierarchy, correlation, Zipfian distributions. Logistic probes 0.974 F1 vs SAEs much lower. **Paper + code, useful for controlled ablations.**

### Landscape Summary

The absorption research landscape has a clear shape: one foundational paper (Chanin et al. 2024) defined the problem and metric on a single task (first-letter spelling), several architectural mitigations exist (Matryoshka, OrtSAE, ATM, masked regularization) but were evaluated under different conditions, and the practical impact is well-documented (DeepMind's negative results). The critical gap is **nobody has measured absorption beyond the first-letter task**. All published absorption numbers come from the spelling task. Meanwhile, three related failure modes (absorption, hedging, incorrect L0) are conflated in practice because they all manifest as "features fail to fire where they should."

The tooling is excellent: SAELens + Gemma Scope + SAEBench give a complete pipeline from model loading to absorption measurement with pretrained SAEs. The sae-spelling repo provides the canonical metric. Everything is MIT/Apache licensed and pip-installable. This is a "measurement paper" opportunity -- the infrastructure exists, the key contribution is designing the right experiments.

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy -- Measuring Absorption Beyond First-Letter Spelling

- **Hypothesis**: Feature absorption occurs at similar or higher rates in semantically richer feature hierarchies (entity type > specific entity, part-of-speech > specific word, sentiment > topic) compared to the first-letter spelling task. The severity of absorption is predictable from measurable properties of the feature hierarchy (depth, branching factor, co-occurrence frequency ratio).

- **Implementation sketch**: Fork sae-spelling, replace the first-letter probe task with 3--4 new hierarchical probe tasks: (1) city-belongs-to-country (knowledge hierarchy), (2) token-is-noun/verb (POS hierarchy), (3) entity-type classification (person/place/organization). For each task, train LR probes on Gemma 2 2B activations using TransformerLens, then run the Chanin absorption metric (k-sparse probing + ablation + cosine similarity check) on Gemma Scope SAEs. Compare absorption rates across domains. Analyze what hierarchy properties predict absorption severity.

- **Simplest version**: Pick ONE additional hierarchy (city-to-country) and measure absorption on a single Gemma Scope SAE (layer 12, width 16k). Compare to first-letter baseline. If the metric works and shows non-trivial absorption, expand to more hierarchies and SAEs.

- **Time estimate**: Pilot 15 min (load SAE, train probe, run absorption on one letter + one country). Full experiment ~4--6 GPU-hours across multiple layers/widths/domains.

- **Reusable components**: sae-spelling (metric code), SAELens (SAE loading), TransformerLens (activation caching), Gemma Scope SAEs (pretrained weights). Need to create new probe datasets (city-country pairs, POS-tagged tokens, entity-type labels) -- straightforward using existing NLP datasets.

### Candidate B: Disentangling Absorption from Hedging and Incorrect L0

- **Hypothesis**: A significant fraction (>30%) of what is currently measured as "absorption" in open-source SAEs is actually caused by incorrect L0 (hedging/mixing) rather than true hierarchy-driven absorption. At the correct L0, true absorption is lower but still substantial.

- **Implementation sketch**: Use Chanin's "sparse-but-wrong" L0 proxy metric to identify the correct L0 for each SAE. Then measure absorption at SAEs with L0 near the correct value vs. SAEs with L0 that is too low (the common case). Use SAEBench's suite of SAEs trained at 6 different sparsity levels to get SAEs at varying L0 without retraining. For each SAE, compute: (a) absorption rate via Chanin metric, (b) hedging degree via Chanin's hedging metric, (c) L0 correctness via proxy metric. Plot all three to see how they covary and which dominates.

- **Simplest version**: Take the SAEBench Gemma-2-2B 16k SAEs (6 sparsity levels, 7 architectures = 42 SAEs already trained). Run absorption + hedging metrics on all of them. Plot absorption vs. L0 vs. hedging. This is purely analysis of existing pretrained SAEs -- no training.

- **Time estimate**: Pilot 15 min (run absorption on 2 SAEs at different L0). Full experiment ~8--10 GPU-hours for all 42 SAEs.

- **Reusable components**: SAEBench SAEs (pretrained), sae-spelling (absorption metric), feature-hedging-paper (hedging metric), sparse-but-wrong-paper (L0 proxy). All exist and are compatible.

### Candidate C: Absorption Scaling Laws -- Predicting Absorption from SAE Configuration

- **Hypothesis**: Absorption rate follows a predictable scaling relationship with SAE width, L0, and dictionary ratio (d_sae/d_model), analogous to the reconstruction scaling laws discovered by Gao et al. (2024). Specifically: absorption_rate ~ alpha * (width)^beta * (L0)^gamma for some discoverable alpha, beta, gamma.

- **Implementation sketch**: Use the full Gemma Scope SAE suite (all layers, widths 1k--1M) plus SAEBench SAEs (3 widths, 6 sparsities, 7 architectures). Run the Chanin absorption metric on each. Fit scaling laws relating absorption rate to (width, L0, layer, architecture). Test whether the scaling law predicts absorption on held-out SAE configurations.

- **Simplest version**: Run absorption on Gemma Scope 16k and 65k SAEs across layers 0--25 of Gemma 2 2B (2 widths x 26 layers = 52 SAEs). Fit power law. Check if it extrapolates to 131k width.

- **Time estimate**: Pilot 15 min (3--4 SAEs). Full experiment ~12--15 GPU-hours for all Gemma Scope SAEs.

- **Reusable components**: Same as above. The novel contribution is the analysis, not the code.

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption Taxonomy

- **Implementation reality check**: The sae-spelling metric requires training LR probes, which is straightforward for any binary classification task. The harder part is constructing clean hierarchical probe datasets where the parent-child relationship is unambiguous. For city-country, this is easy (known ground truth). For POS, standard NLP taggers give labels. For entity types, NER datasets exist. Searched for prior attempts: no one has published absorption measurements on non-spelling tasks. The Chanin paper explicitly calls this out as the top open question (Gap 2 in the literature). The metric's dependence on ablation effects means it only works in early/middle layers (layers 0--17 for Gemma 2 2B), but SAEBench's modified metric based on probe-direction cosine similarity works at all layers.

- **Reproducibility attack**: The probe training and absorption calculation are deterministic given fixed SAE weights. The main fragility is in defining what counts as a "hierarchy" -- we need probe tasks where ground truth is unambiguous. First-letter spelling is ideal because it is 100% deterministic (every token has exactly one first letter). City-country is also deterministic. POS is slightly noisier but well-established. The methodology is clean.

- **Baseline sanity check**: The baseline is Chanin's own first-letter result (15--35% absorption). We are measuring the same metric on new domains, so the comparison is direct. If we find similar rates, the contribution is "absorption generalizes." If we find different rates, the contribution is "absorption severity depends on hierarchy structure" and we characterize how.

- **Scope attack**: This could be dismissed as "just running the same metric on more tasks." The counter: the entire field's understanding of absorption is based on ONE task. Showing it generalizes (or doesn't) across semantically diverse hierarchies is a genuine contribution, especially if we can identify hierarchy properties that predict severity. This is the kind of careful measurement work that NeurIPS publishes.

- **Verdict**: **STRONG**. Low implementation risk, clear gap, reuses existing infrastructure, results are informative regardless of direction.

### Against Candidate B: Disentangling Absorption from Hedging

- **Implementation reality check**: The hedging metric exists in the feature-hedging-paper repo but was developed for a different codebase than sae-spelling. Integration may require engineering effort. The L0 proxy metric from sparse-but-wrong exists but may not be trivial to run on all SAEBench SAEs. Searched for anyone who has jointly measured absorption + hedging + L0: no one has. The balanced Matryoshka work (Chanin 2025) shows they trade off but doesn't provide a systematic decomposition.

- **Reproducibility attack**: The decomposition "this is absorption vs. this is hedging" requires a clean diagnostic criterion. The current absorption metric detects absorption by cosine similarity with probe + ablation dominance. Hedging is measured by decoder direction alignment. These are conceptually distinct but might interact in subtle ways. There's a risk of circular reasoning (defining absorption as "not hedging" and vice versa).

- **Baseline sanity check**: Without a ground-truth decomposition, it's hard to validate the claimed percentages. SynthSAEBench could provide this (synthetic features with known hierarchy), but running on synthetic data limits real-world relevance.

- **Scope attack**: The contribution might be perceived as incremental -- "we ran two existing metrics on the same SAEs and found they correlate." The insight needs to be sharper: what fraction of "absorption" goes away when L0 is correct? This requires finding SAEs at the correct L0, which may not exist in the current pretrained suite.

- **Verdict**: **MODERATE**. Conceptually interesting but implementation is less clean, and the decomposition may not yield a sharp story. Better as a secondary analysis within a larger paper than as the main contribution.

### Against Candidate C: Absorption Scaling Laws

- **Implementation reality check**: Running absorption on 50+ SAEs is straightforward but slow (~15 min per SAE for the full metric). The main risk is that absorption doesn't follow a clean scaling law -- Chanin et al. already noted "no clear layer-wise pattern." If the relationship is noisy, a scaling law paper falls flat.

- **Reproducibility attack**: The measurements are reproducible (deterministic metric + fixed SAE weights). The scaling law fit is standard regression. The risk is overfitting to the specific Gemma 2 2B model -- does the law generalize to Pythia or Llama?

- **Baseline sanity check**: Gao et al.'s reconstruction scaling laws are the template, but those are clean power laws because MSE is a smooth continuous metric. Absorption rate is a discrete 0/1 measurement aggregated over letters, making it inherently noisier. The power law may not fit well.

- **Scope attack**: If the scaling law doesn't fit cleanly, we just have "a bunch of absorption measurements" without a unifying principle. The negative result ("absorption doesn't follow simple scaling laws") is less publishable. Also, measuring absorption on many SAEs doesn't by itself suggest solutions.

- **Verdict**: **MODERATE**. High ceiling if clean scaling laws exist, but high risk of noisy non-results. Better as a component of a larger study (e.g., combined with cross-domain measurements).

## Phase 4: Refinement

### Dropped Ideas

**Candidate C (Scaling Laws)** is too risky as a standalone paper. The absorption metric is inherently noisy and there's existing evidence (no clear layer-wise pattern) suggesting clean scaling laws may not exist. However, the core measurements (absorption across many SAEs) are valuable as supporting evidence for other ideas.

### Strengthened Ideas

**Candidate A (Cross-Domain Taxonomy)** is the clear front-runner. I strengthen it by:

1. **Merging in elements of Candidate B**: For each domain, also measure hedging degree and L0 to control for confounds. This addresses Gap 9 (confounding) alongside Gap 2 (cross-domain).

2. **Merging in elements of Candidate C**: Run across enough SAE configurations (multiple widths, sparsities, layers) to characterize how absorption varies with these factors per domain. Not a formal scaling law, but a thorough empirical characterization.

3. **Adding an unsupervised absorption signal**: For each probe task, compute the cosine similarity between the absorbing latent's decoder direction and the probe direction. This gives a lightweight unsupervised proxy for absorption (high max cosine similarity between decoder vectors = potential absorption). If this proxy correlates with the full metric, it's a step toward Gap 7 (unsupervised detection).

4. **Concrete pilot experiment confirmed**: Load Gemma Scope SAE (layer 12, width 16k) via SAELens. Train city-country LR probe on Gemma 2 2B activations. Run sae-spelling absorption metric adapted for city-country task. This requires:
   - A list of city-country pairs (Wikipedia, ~500 cities across 50+ countries)
   - An ICL prompt template: `"{city} is located in the country: {country}"`
   - Adapting sae-spelling's probe training to predict country instead of first letter
   - Running the same ablation-based absorption detection

   **I verified**: sae-spelling's `feature_absorption_calculator` is parameterized by probe directions, so swapping in a country-probe direction is straightforward.

5. **Simplification**: Drop any domain where the hierarchy is ambiguous. Keep:
   - **First-letter** (baseline, replication of Chanin)
   - **City-to-country** (knowledge hierarchy, unambiguous ground truth)
   - **City-to-continent** (coarser hierarchy, tests depth effect)
   - **Token-to-POS** (syntactic hierarchy, very different from knowledge)

### Selected Front-Runner

**Candidate A (enhanced)**: Cross-Domain Feature Absorption Characterization. The highest success probability because:
- The gap is clearly documented and widely acknowledged
- All tools exist and are tested
- Results are informative regardless of direction
- The engineering is straightforward (adapt existing metric to new probe tasks)
- Directly addresses the most-cited limitation of the foundational absorption paper

## Phase 5: Final Proposal

### Title

Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders: Beyond the Spelling Task

### Hypothesis

Feature absorption in SAEs is not specific to the first-letter spelling task but is a general phenomenon that occurs wherever features form hierarchies. The severity of absorption is predictable from measurable properties of the feature hierarchy (co-occurrence frequency ratio between parent and child features, hierarchy depth, number of children per parent). Absorption severity varies systematically across semantic domains (syntactic vs. knowledge vs. spelling), and a significant fraction of measured "absorption" at common SAE configurations is confounded with feature hedging due to incorrect L0.

### Motivation

Feature absorption is the most important known failure mode of SAEs for practical interpretability. DeepMind deprioritized SAE research partly because of the gap between SAE probes and linear probes, with absorption as a key culprit. Anthropic's circuit tracing work shows that when SAE features are reliable, they enable powerful mechanistic understanding -- absorption is the main obstacle.

Yet everything we know about absorption comes from one task (first-letter spelling) on one feature hierarchy (letter membership > token identity). The field cannot make informed decisions about SAE architecture, hyperparameters, or use cases without knowing whether absorption rates of 15--35% generalize to the semantically richer hierarchies that actually matter for safety (entity types, sentiment, harmful intent).

### Method

**Step 1: Define Cross-Domain Probe Tasks (1 day)**

Construct 4 hierarchical probe tasks with unambiguous ground truth:

| Task | Parent Feature | Child Feature | Source | Hierarchy Depth |
|------|---------------|---------------|--------|-----------------|
| First-letter (baseline) | "starts with S" | "the token 'snake'" | sae-spelling | 2 |
| City-to-country | "located in France" | "the city Paris" | Wikipedia cities | 2 |
| City-to-continent | "located in Europe" | "the city Paris" | Wikipedia cities | 2 (coarser) |
| Token POS | "is a noun" | "the token 'cat'" | Universal Dependencies | 2 |

For each task, prepare:
- A vocabulary of ~500 items with known labels (e.g., 500 cities with country/continent labels)
- An ICL prompt template (e.g., `"{city} is located in the country:"`)
- Ground-truth parent-child hierarchy labels

**Step 2: Train Linear Probes (2 hours)**

For each task and each Gemma 2 2B layer (focus on layers 5--20 where absorption metric works):
- Cache activations using TransformerLens
- Train logistic regression probes predicting the parent feature (e.g., "starts with S", "located in France")
- Record probe accuracy as a ceiling for SAE performance

**Step 3: Measure Absorption (4--6 hours)**

For each task, run the Chanin absorption metric on Gemma Scope SAEs:
- Primary: layer 12, width 16k (canonical)
- Extended: layers 5, 8, 12, 16, 20; widths 16k and 65k (20 SAEs)
- For each SAE:
  1. Train k-sparse probes on the parent feature using SAE latents
  2. Identify false-negative tokens (probe correct, all k latents silent)
  3. Run integrated-gradients ablation on false-negatives
  4. Detect absorption via cosine similarity + ablation dominance criteria
  5. Compute absorption rate

**Step 4: Measure Confounds (2 hours)**

For each SAE in the analysis:
- Compute hedging degree using Chanin's hedging metric
- Estimate L0 correctness using the sparse-but-wrong proxy metric
- Record decoder direction max cosine similarity as unsupervised absorption proxy

**Step 5: Analysis (1 day)**

- Compare absorption rates across domains at matched SAE configurations
- Correlate absorption severity with hierarchy properties (co-occurrence frequency ratio, number of children per parent)
- Decompose: what fraction of measured absorption persists after controlling for L0 and hedging?
- Evaluate unsupervised proxy (decoder cosine similarity) against full supervised metric

**Libraries/repos used:**
- `sae-spelling` (absorption metric, fork and extend)
- `SAELens v6` (SAE loading, activation caching)
- `TransformerLens` (model hooks)
- `feature-hedging-paper` (hedging metric)
- `sparse-but-wrong-paper` (L0 proxy)
- Gemma Scope SAEs (pretrained weights, no training)

### Simplest Version

Load one Gemma Scope SAE (layer 12, width 16k). Construct a city-country probe task with 200 cities. Train LR probe. Run absorption metric. Compare to first-letter baseline on the same SAE. Report: "absorption rate on city-country is X% vs Y% on first-letter." This takes ~30 minutes and gives immediate signal on whether the cross-domain direction works.

### Baselines

1. **Chanin et al. first-letter results** (replication): Absorption rate 15--35% on Gemma Scope 16k/65k. We replicate this as our baseline and sanity check.

2. **Random probe baseline**: Train probes on randomly shuffled labels. Absorption rate should be ~0%. Verifies the metric isn't detecting noise.

3. **(Optional) Matryoshka SAE**: If SAEBench Matryoshka SAEs are available, measure absorption on them for each domain. Expected: ~0.03 (Bussmann et al. 2025). Verifies the metric correctly detects known low-absorption architectures across domains.

### Experimental Plan

**Phase 0 -- Pilot (30 min, 1 GPU)**
- Load Gemma Scope SAE, train city-country probe, run absorption on 3 letters + 3 countries
- Go/no-go: does the metric produce sensible numbers?

**Phase 1 -- Cross-Domain Core (6 hours, 1 GPU)**
- All 4 tasks on layer 12, widths 16k and 65k
- Primary result: absorption rates across domains

**Phase 2 -- Layer and Width Sweep (8 hours, 1 GPU)**
- All 4 tasks on layers 5, 8, 12, 16, 20; widths 16k and 65k
- Result: how absorption varies across layers and widths per domain

**Phase 3 -- Confound Analysis (4 hours, 1 GPU)**
- Hedging + L0 measurements on all SAEs from Phase 2
- Result: decomposition of absorption vs. hedging vs. L0 effects

**Phase 4 -- Unsupervised Proxy (2 hours, 1 GPU)**
- Compute decoder cosine similarity proxy for all SAEs
- Result: correlation between proxy and supervised absorption metric

Total: ~20 GPU-hours, well within single-GPU budget across a few days.

### Resource Estimate

- **GPU**: 1x GPU with >=16GB VRAM (Gemma 2 2B fits in 16-bit on single GPU)
- **GPU-hours**: ~20 hours total (4--5 sessions of ~4 hours each)
- **Wall-clock**: 3--4 days with 1 GPU
- **Model size**: Gemma 2 2B (5GB weights) + Gemma Scope SAEs (~200MB each)
- **Storage**: ~10GB for cached activations + probe weights + results
- **No training required**: All SAEs are pretrained. Only LR probes are trained (seconds each).

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| City-country probe doesn't work well on Gemma 2 2B | Low | High | Verify Gemma 2 2B knows city-country facts first (simple inference test). If not, fall back to Gemma 2 9B or use entity-type classification instead. |
| Absorption metric fails on non-spelling tasks | Medium | High | The metric depends on ablation effects, which require the model to use information at the source token position. Test this first in pilot. If ablation fails, use SAEBench's cosine-similarity-based variant that works at all layers. |
| Absorption rates are identical across all domains | Low | Medium | This is still a useful result ("absorption generalizes uniformly"). Report it, plus analyze WHY rates are similar (similar co-occurrence statistics?). |
| Results are too noisy for clear conclusions | Medium | Medium | Use more tokens per probe task (500+). Bootstrap confidence intervals. Focus on relative comparisons (domain A vs domain B on same SAE) rather than absolute rates. |
| Someone publishes cross-domain absorption first | Low | High | This is a known gap; no preprint exists as of April 2026. Move fast -- the pilot takes 30 min. |

### Novelty Claim

The novelty is NOT in the metric (which is Chanin et al.'s) or the SAEs (which are pretrained). The novelty is:

1. **First systematic cross-domain characterization of absorption**: Moving beyond the single spelling task to show whether and how absorption generalizes across semantically diverse feature hierarchies. This is the #1 acknowledged limitation of the foundational absorption paper.

2. **Hierarchy property predictors**: Identifying which measurable properties of a feature hierarchy (co-occurrence ratio, depth, branching factor) predict absorption severity. This gives practitioners actionable guidance.

3. **Absorption-hedging decomposition**: First joint measurement of absorption, hedging, and L0 correctness on the same SAEs, quantifying how much "absorption" is actually confounded.

4. **Unsupervised absorption proxy**: If the decoder cosine similarity proxy correlates with supervised absorption, this is the first step toward an unsupervised absorption detection method (Gap 7).

This is a "strong baseline done right" paper: careful measurement using existing tools on the most important open question in the subfield, with enough analysis to provide actionable insights for SAE practitioners and architecture designers.

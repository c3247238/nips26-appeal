# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (GitHub: lasr-spelling/sae-spelling) -- MIT licensed. Canonical absorption metric implementation. Contains `feature_absorption_calculator`, k-sparse probing, integrated-gradients ablation, spelling grader. Directly reusable for first-letter task; the key extension needed is adapting probes to new feature hierarchies. **Has usable code.**

2. **SAEBench** (GitHub: adamkarvonen/SAEBench) -- Apache 2.0. 8-metric evaluation suite including absorption task. 200+ pretrained SAEs across 7 architectures on Pythia-160M and Gemma-2-2B. Absorption eval extends Chanin et al.'s metric to work across all layers via LR probe projection (no ablation needed). `pip install sae-bench` works out of the box. **Has usable code.**

3. **SAELens v6** (GitHub: decoderesearch/SAELens) -- MIT licensed. Standard library for loading/training/evaluating SAEs. Supports BatchTopK, JumpReLU, TopK, Standard architectures. Deep integration with TransformerLens. Loads Gemma Scope, GPT-2, Pythia, Llama SAEs from HuggingFace. **Has usable code.**

4. **Gemma Scope SAEs** (HuggingFace: google/gemma-scope) -- 400+ JumpReLU SAEs on Gemma 2 2B/9B/27B, all layers, widths 1k-1M. Gemma Scope 2 adds Matryoshka SAEs. Eliminates all training cost for analysis experiments. **Pretrained weights available.**

5. **"A is for Absorption" (Chanin et al., NeurIPS 2025)** -- arXiv:2409.14507. Defines and characterizes feature absorption on first-letter spelling task. Absorption rate 15-35% across Gemma Scope SAEs. Present in ALL tested architectures including BatchTopK (which lacks L1). Key limitation: only first-letter task studied.

6. **SAEBench SAEs** -- 200+ SAEs, 3 widths (4k, 16k, 65k), 6 sparsities on Pythia-160M layer 8 and Gemma-2-2B layer 12. Open-sourced with training checkpoints. Ideal for controlled comparisons. **Pretrained weights available.**

7. **"Do I Know This Entity?" (Ferrando et al., ICLR 2025)** -- Probes Gemma 2 2B for entity knowledge (cities, movies, players, songs) using Gemma Scope SAEs. Found SAE latents distinguish known vs. unknown entities, peak discriminability around layer 9. Demonstrates that entity-type probe tasks are feasible on Gemma 2 2B with existing SAEs. **Methodology directly reusable for cross-domain absorption.**

8. **sae-probes** (GitHub: sae-probes/sae-probes) -- Python package for sparse probing benchmark. Works with any SAELens-compatible SAE. Supports Gemma-2-2B with TransformerLens hooks. Ready-made framework for training probes on model activations. **Has usable code.**

9. **Matryoshka SAE paper** (Bussmann et al., ICML 2025) -- arXiv:2503.17547. Absorption rate ~0.03 vs BatchTopK ~0.29. Uses decoder cosine similarity as unsupervised metric for detecting absorption. Shows high decoder cosine similarity between parent/child latents indicates absorption. **Key insight for unsupervised detection.**

10. **"Sparse Autoencoders Do Not Find Canonical Units of Analysis" (Leask et al., ICLR 2025)** -- arXiv:2502.04878. SAE stitching via decoder cosine similarity. Thresholding on max decoder cosine similarity classifies latents as novel vs. reconstruction. Demonstrates decoder geometry captures absorption-like structure without supervised probes.

11. **Feature Hedging paper** (Chanin et al., 2025) -- arXiv:2505.11756. Code at GitHub: chanind/feature-hedging-paper. Identifies hedging as complementary failure mode. Shows Matryoshka SAEs trade absorption for hedging. Balanced Matryoshka (compound multiplier ~0.75) is best overall. **Has usable code.**

12. **TransformerLens** (GitHub: TransformerLensOrg/TransformerLens) -- MIT licensed. Hook-based activation caching/editing for 50+ models. Deep SAELens integration. Standard infrastructure. **Has usable code.**

### Landscape Summary

The tooling ecosystem is mature: SAELens + TransformerLens + Gemma Scope gives a complete activation extraction and SAE analysis pipeline. SAEBench provides standardized evaluation including absorption. The sae-spelling repo provides the canonical absorption metric code that can be adapted.

**What actually works:**
- Absorption is robustly measurable on the first-letter spelling task using sae-spelling/SAEBench code
- Matryoshka SAEs convincingly reduce absorption (~10x) but trade it for hedging
- Decoder cosine similarity is an effective unsupervised signal for absorption (used in both Matryoshka and SAE-stitching papers)
- Entity-knowledge probes are feasible on Gemma 2 2B (Ferrando et al. ICLR 2025)

**What does not work / is risky:**
- No one has extended absorption measurement beyond first-letter spelling on real LLMs
- Pure architecture comparisons without new analysis are well-trodden
- Unsupervised absorption detection has been hinted at but never formalized or validated
- Training new SAEs is expensive and our project is training-free

**Practical gaps that an engineer sees:**
- Gap 2 (cross-domain absorption) and Gap 7 (unsupervised detection) are the most implementable with existing tools
- Gap 1 (quantitative theory) requires extensive controlled experiments that may exceed our compute budget
- Gap 4 (mitigation comparison) is blocked by training-free constraint -- we cannot train new SAEs to compare

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy -- Measuring Absorption Beyond First-Letter Spelling

- **Hypothesis**: Feature absorption is not specific to the first-letter spelling task but occurs systematically across all hierarchical feature domains, with absorption severity predictable from hierarchy properties (depth, co-occurrence frequency, feature frequency ratio).
- **Implementation sketch**: Adapt sae-spelling's absorption metric pipeline to 3-4 new probe tasks with known hierarchical structure on Gemma 2 2B using Gemma Scope SAEs (16k and 65k widths). Candidate tasks: (1) city -> country (knowledge hierarchy), (2) word -> part-of-speech (syntactic hierarchy), (3) entity -> entity-type (semantic hierarchy using Wikidata categories from Ferrando et al.'s template). For each task: train LR probes on residual stream activations, find k-sparse SAE feature splits, measure false-negative rate, run attribution to detect absorption. Compare absorption rates across tasks, layers, SAE widths.
- **Simplest version**: Pick two tasks (first-letter as control + city->country) on a single Gemma Scope SAE (layer 12, 16k width). Train probes, measure absorption rate on both. If absorption rate on city->country is >5%, we have signal.
- **Time estimate**: ~8-10 GPU-hours total. Probe training: ~30 min per task per layer. Attribution/ablation: ~1 hour per task. Pilot (2 tasks, 1 layer, 1 SAE): ~2 hours.
- **Reusable components**: sae-spelling (absorption metric code), SAELens (SAE loading), TransformerLens (activation caching), Gemma Scope SAEs (pretrained weights), Ferrando et al.'s entity probing methodology.

### Candidate B: Unsupervised Absorption Detection via Decoder Geometry

- **Hypothesis**: Feature absorption can be detected without supervised probes by analyzing the decoder weight matrix geometry of pretrained SAEs. Specifically, when a child feature absorbs a parent, the child's decoder vector will have anomalously high cosine similarity with several other decoder vectors (the "parent cluster"), and the parent feature's decoder vector will be an approximate linear combination of its children's decoders.
- **Implementation sketch**: For each pretrained Gemma Scope SAE: (1) compute pairwise decoder cosine similarity matrix, (2) identify "suspicious" latent pairs with high similarity (>0.3), (3) for each suspicious pair, check if one latent's activation pattern is a strict subset of the other's (subsumption test on a text corpus), (4) when subsumption holds, flag as absorption candidate. Validate against sae-spelling's supervised absorption metric on first-letter task. If correlation is high, apply unsupervised metric to all layers and tasks where supervised probes do not exist.
- **Simplest version**: Compute decoder cosine similarity for one Gemma Scope SAE (layer 12, 16k). Find top-100 highest-similarity pairs. Check overlap with known first-letter absorption cases from sae-spelling. If >50% of known absorption cases appear in the high-similarity set, the unsupervised signal is validated.
- **Time estimate**: ~3-4 GPU-hours. Decoder matrix analysis: <10 min (no forward passes needed). Activation collection for subsumption test: ~1 hour. Correlation with supervised metric: ~30 min.
- **Reusable components**: SAELens (SAE loading, decoder weight access), sae-spelling (ground truth absorption labels), numpy/scipy (cosine similarity computation).

### Candidate C: Disentangling Absorption from Hedging and Incorrect L0

- **Hypothesis**: A substantial fraction (>30%) of what is currently measured as "absorption" in standard SAEs is actually caused by incorrect L0 (feature hedging) rather than true hierarchy-driven absorption. Controlling for L0 will sharpen absorption measurements.
- **Implementation sketch**: Use SAEBench's suite of 200+ SAEs trained at 6 different sparsity levels (L0 ~20 to ~640) on Gemma-2-2B layer 12. For each sparsity level: (1) measure absorption rate using SAEBench absorption eval, (2) measure hedging indicators (feature correlation, width utilization), (3) use Chanin & Garriga-Alonso's L0 correctness proxy metric. Plot absorption rate vs. L0 correctness, separating "true absorption" (persists at correct L0) from "L0-confounded absorption" (disappears when L0 is correct).
- **Simplest version**: Run SAEBench absorption eval on 6 BatchTopK SAEs at different sparsity levels from SAEBench's pretrained suite. Plot absorption rate vs. L0. If the curve is non-monotonic (absorption drops then rises), we have evidence of L0 confounding.
- **Time estimate**: ~4-6 GPU-hours. SAEBench absorption eval: ~30 min per SAE (probes cached per layer). Running on 12-18 SAEs: ~6-9 hours but parallelizable.
- **Reusable components**: SAEBench (absorption eval + pretrained SAEs), sparse-but-wrong-paper (L0 correctness proxy).

---

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption Taxonomy

- **Implementation reality check**: Ferrando et al. (ICLR 2025) already demonstrated entity probing on Gemma 2 2B with Gemma Scope SAEs. The sae-spelling pipeline is well-documented and modular. The main engineering risk is that non-first-letter probe tasks may have lower probe accuracy (first-letter is ~99% accurate with LR probes; city->country may be ~80-90%), which makes the false-negative identification noisier. Searched for prior work extending absorption beyond first-letter: found nothing. This is genuinely unstudied.
- **Reproducibility attack**: The pipeline is straightforward: train probes, find false negatives, run attribution. All code is open-source. The main fragility is in probe quality -- if probes are poor, absorption measurement is unreliable. Mitigation: report probe accuracy alongside absorption rate, and only interpret absorption where probe accuracy >85%.
- **Baseline sanity check**: The baseline is "run the same analysis on first-letter to reproduce Chanin et al.'s numbers." If we cannot reproduce their 15-35% absorption rate, something is wrong with our pipeline. No one has measured absorption on other domains, so there is no existing number to beat -- but the contribution is the measurement itself.
- **Scope attack**: Multiple domains (knowledge, syntactic, semantic) give breadth. Multiple SAE widths and layers give depth. This is not a one-trick result.
- **Verdict**: **STRONG**. High novelty (no prior cross-domain absorption study), feasible implementation (reuse existing code), clear contribution (characterizing absorption's generality), bounded risk (worst case: absorption does not generalize, which is also a publishable finding).

### Against Candidate B: Unsupervised Absorption Detection via Decoder Geometry

- **Implementation reality check**: Decoder cosine similarity computation is trivial. The Matryoshka SAE paper and SAE-stitching paper both use it as a diagnostic. However, the subsumption test (checking activation overlap) requires running a corpus through the SAE, which is standard but takes ~1 hour per SAE. The real risk: decoder cosine similarity may have too many false positives. Many SAE features have high decoder similarity for reasons unrelated to absorption (e.g., features in the same semantic cluster).
- **Reproducibility attack**: The method is simple and deterministic. However, the thresholds (cosine similarity >0.3, subsumption ratio) are somewhat arbitrary. Need to validate on known absorption cases to calibrate.
- **Baseline sanity check**: The baseline is the supervised absorption metric from sae-spelling. If the unsupervised metric does not correlate with it (Spearman r < 0.5), the method fails. The Matryoshka paper shows high decoder cosine similarity between absorbed parent/child pairs, but this was a qualitative observation, not a validated metric.
- **Scope attack**: If validated, this metric can be applied to ANY SAE on ANY layer without needing to define probe tasks. This is the key advantage. But validation is limited to domains where supervised probes exist (first-letter).
- **Verdict**: **MODERATE**. High potential payoff but unvalidated core assumption. The correlation between decoder geometry and supervised absorption is plausible but unproven. If it works, it is a very strong contribution. If it fails, we have a negative result that is still useful but less publishable as a standalone paper.

### Against Candidate C: Disentangling Absorption from Hedging

- **Implementation reality check**: SAEBench has all the pretrained SAEs and the absorption eval code. Running the eval is straightforward (`python -m sae_bench.evals.absorption.main`). The L0 correctness proxy from Chanin & Garriga-Alonso is also available. The main engineering concern: SAEBench absorption eval takes ~30 min per SAE with probe caching, so running on 18+ SAEs needs ~9 GPU-hours. But this is parallelizable and well within budget.
- **Reproducibility attack**: All SAEs are publicly available with fixed random seeds. The evaluation code is open-source. This is highly reproducible.
- **Baseline sanity check**: The finding that absorption and hedging are confounded was already noted qualitatively by Chanin et al. (2025) in the feature hedging paper. Our contribution is quantifying the confound. However, the analysis may yield an expected result (absorption increases with sparsity) without a surprising finding.
- **Scope attack**: This only applies to the first-letter task on one model (Gemma-2-2B layer 12). The scope is narrow unless combined with cross-domain analysis.
- **Verdict**: **MODERATE**. Sound methodology, good reproducibility, but potentially incremental contribution. Better as a supporting analysis within a larger paper (e.g., combined with Candidate A) than as a standalone.

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C as standalone**: Too incremental. The L0/hedging confound analysis is important but narrow. However, it becomes a powerful supplementary experiment within a larger paper.

### Strengthened Ideas

**Candidate A (Cross-Domain Absorption) is the front-runner.** Strengthening:

1. **Combine with Candidate B as a secondary contribution**: Use decoder cosine similarity as a complementary unsupervised absorption indicator. This gives us two contributions in one paper: (1) cross-domain absorption characterization (supervised), (2) unsupervised absorption detection via decoder geometry (validated against supervised results).

2. **Combine with Candidate C as an ablation**: Include L0-sensitivity analysis as an ablation study. Run the cross-domain absorption measurement at multiple sparsity levels to disentangle true absorption from L0 confounds.

3. **Simplify the probe task selection**: After searching the literature, the cleanest hierarchy tasks are:
   - **First-letter** (control, replicating Chanin et al.)
   - **City -> Country** (knowledge hierarchy, well-studied in Ferrando et al.)
   - **Token -> Part-of-speech** (syntactic hierarchy, e.g., "running" is a verb)
   - **Animal -> Taxonomy** (e.g., "dog" -> mammal, "salmon" -> fish)

   Each has a clear parent-child relationship, is probe-trainable, and uses Gemma 2 2B.

4. **Pilot experiment design** (< 15 min): Load one Gemma Scope SAE (Gemma 2 2B, layer 12, 16k width). Train LR probes for first-letter (to reproduce Chanin et al.) and city->country (to test generalization). Compare probe accuracy. If city->country probe accuracy >80%, proceed.

5. **Code exists for everything**: sae-spelling for absorption metric, SAELens for SAE loading, TransformerLens for activation hooks, Ferrando et al.'s entity templates for city/country probes. No new library needed.

### Selected Front-Runner

**Candidate A (extended)**: "Cross-Domain Feature Absorption in Sparse Autoencoders: A Systematic Characterization Beyond First-Letter Spelling"

Combined with decoder geometry analysis (Candidate B) as secondary contribution and L0-sensitivity ablation (Candidate C).

---

## Phase 5: Final Proposal

### Title

Cross-Domain Feature Absorption in Sparse Autoencoders: Characterization, Unsupervised Detection, and Implications for Interpretability

### Hypothesis

Feature absorption in SAEs is a general phenomenon that occurs wherever the model's internal representations encode hierarchical feature structure, not just on the first-letter spelling task. The severity of absorption is predictable from the properties of the feature hierarchy (frequency ratio between parent and child features, co-occurrence rate, hierarchy depth). Furthermore, absorption can be detected without supervised probes by analyzing decoder weight geometry.

### Motivation

Feature absorption (Chanin et al., 2024) is one of the most important failure modes of SAEs, directly undermining the reliability of mechanistic interpretability. DeepMind deprioritized SAE research in part because absorption-related failures cause 10-40% performance degradation on safety tasks. Yet all empirical study of absorption has been confined to a single task (first-letter spelling) on a narrow feature hierarchy (letter membership > token identity). This limits our understanding of: (1) how widespread absorption is, (2) what predicts its severity, and (3) whether we can detect it at scale without hand-crafting probe tasks. Our work fills these gaps with the first systematic cross-domain absorption study and the first validated unsupervised absorption detection method.

### Method

**Step-by-step implementation plan:**

**Phase 0: Infrastructure Setup (Day 1, ~2 hours)**
- Install SAELens v6, TransformerLens, sae-bench, sae-spelling
- Load Gemma 2 2B model and Gemma Scope SAEs via SAELens
- Reproduce Chanin et al.'s first-letter absorption measurement as sanity check
- Libraries: `sae-lens`, `transformer-lens`, `sae-bench`, `torch`, `scikit-learn`

**Phase 1: Cross-Domain Probe Construction (Days 1-2, ~4 hours)**
- Define 4 hierarchy tasks with clear parent-child structure:
  1. **First-letter** (control): token -> starts-with-letter-X (reuse sae-spelling prompts)
  2. **City-Country**: city name token -> belongs-to-country-Y (construct from Wikidata, template: "{city} is located in {country}")
  3. **POS tagging**: token -> part-of-speech (verb/noun/adjective; use standard POS datasets)
  4. **Animal taxonomy**: animal token -> taxonomic class (mammal/bird/fish/reptile; construct from knowledge bases)
- For each task: construct balanced prompt sets (~500-1000 examples), train LR probes on Gemma 2 2B residual stream activations (layers 6, 12, 18, 24), verify probe accuracy >85%
- Code: Adapt `sae_spelling.probing.train_multi_probe()` to new tasks

**Phase 2: Cross-Domain Absorption Measurement (Days 2-4, ~12 hours)**
- For each task x each layer x each SAE width (16k, 65k):
  1. Find k-sparse feature splits for each parent feature using k-sparse probing
  2. Identify false-negative tokens (all k splits fail to fire, but probe correctly classifies)
  3. Run integrated-gradients attribution on false-negative tokens (following sae-spelling pipeline)
  4. Compute absorption rate using Chanin et al.'s metric
- Measure hierarchy properties: parent feature frequency, child feature frequency, parent-child co-occurrence rate, hierarchy depth
- Code: Adapt `sae_spelling.feature_absorption_calculator` to new probe tasks

**Phase 3: Unsupervised Detection via Decoder Geometry (Days 3-4, ~4 hours)**
- For each pretrained SAE analyzed in Phase 2:
  1. Extract decoder weight matrix W_dec from SAE
  2. Compute pairwise cosine similarity (top-k sparse to avoid O(n^2) on 65k features)
  3. Identify high-similarity pairs (cosine >0.3 threshold, calibrated on first-letter ground truth)
  4. For each high-similarity pair: compute activation subsumption score on text corpus (does feature A always fire when feature B fires?)
  5. Define unsupervised absorption score: high decoder similarity + asymmetric activation subsumption
- Validate against supervised absorption metric from Phase 2 (Spearman correlation, precision/recall)
- If validated (r > 0.5): apply to ALL layers and SAE widths, generating absorption landscape map
- Code: Custom (~200 lines), using SAELens `SAE.W_dec` and torch cosine_similarity

**Phase 4: L0 Sensitivity Analysis (Day 5, ~3 hours)**
- Use SAEBench's pretrained SAEs at 6 sparsity levels on Gemma-2-2B layer 12
- Run absorption eval (SAEBench standard) at each sparsity level for city-country task
- Plot absorption rate vs. L0, compare with first-letter results
- Determine fraction of measured absorption attributable to L0 confound vs. true hierarchy-driven absorption
- Code: SAEBench CLI + custom plotting

**Phase 5: Analysis and Writing (Days 5-7)**
- Statistical analysis: regression of absorption rate on hierarchy properties
- Visualization: absorption heatmaps (layer x task x SAE width), decoder similarity graphs
- Main findings, ablation tables, failure case analysis

### Simplest Version

Load one Gemma Scope SAE (Gemma 2 2B, layer 12, width 16k). Train LR probes for first-letter (control) and city->country. Measure absorption rate on both tasks using sae-spelling pipeline. If city->country shows absorption rate >5% (vs. 15-35% for first-letter), we have the core finding. Total time: ~3-4 hours.

### Baselines

1. **Chanin et al. (NeurIPS 2025) first-letter absorption**: Our control condition. Expected absorption rate 15-35% on Gemma Scope 16k/65k SAEs. We must reproduce this within +/-5% to validate our pipeline.

2. **Dense LR probe performance**: For each task, the LR probe trained on raw residual stream activations sets the ceiling for what is "knowable" by the model. Absorption is measured relative to this ceiling (features the model knows but the SAE fails to represent).

### Experimental Plan

**Datasets:**
- First-letter: sae-spelling's existing prompt set (~26 letters, ~500 tokens each)
- City-Country: Wikidata cities with known countries (~1000 city-country pairs, balanced across ~50 countries)
- POS tagging: Penn Treebank or Universal Dependencies subset (~2000 tokens, 4 POS classes)
- Animal taxonomy: Curated animal list (~500 animals, 4 taxonomic classes)

**Metrics:**
- Primary: Absorption rate (Chanin et al. metric, adapted per task)
- Secondary: Unsupervised absorption score (decoder cosine similarity + subsumption)
- Auxiliary: Probe accuracy, false-negative rate, SAE reconstruction error, feature split count (k)

**Ablation schedule:**
1. **Across tasks**: 4 hierarchy tasks, compare absorption rates
2. **Across layers**: Layers 6, 12, 18, 24 of Gemma 2 2B
3. **Across SAE widths**: 16k and 65k from Gemma Scope
4. **Across sparsity levels**: 6 L0 settings from SAEBench SAEs (layer 12 only)
5. **Unsupervised vs. supervised**: Correlation analysis of decoder geometry metric vs. Chanin metric

### Resource Estimate

- **Model**: Gemma 2 2B (~5GB VRAM) -- fits on single GPU with SAE
- **SAEs**: Gemma Scope pretrained, no training needed
- **GPU-hours**: ~25-30 total across all experiments
  - Probe training: ~4 hours (4 tasks x 4 layers x ~15 min each)
  - Absorption measurement: ~12 hours (4 tasks x 4 layers x 2 widths x ~20 min each)
  - Decoder geometry analysis: ~4 hours
  - L0 sensitivity: ~3 hours
  - Pilot: ~2-3 hours
- **Wall-clock time**: ~5-7 days with single GPU, accounting for iteration
- **Disk**: ~50GB for activation caches (SAEBench standard)
- **Training-free**: All analysis uses pretrained Gemma Scope SAEs and SAEBench SAEs. No SAE training required.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Probe accuracy too low on non-first-letter tasks (<80%) | Medium | High -- invalidates absorption measurement | Use multiple probe types (LR, MLP), try different layers, report probe accuracy as quality gate |
| Absorption does not generalize beyond first-letter | Low | Medium -- still publishable as negative result | Include 4 tasks to maximize chance of finding absorption; negative result is informative |
| Unsupervised metric does not correlate with supervised | Medium | Low -- Candidate B fails, but Candidate A still stands | Report as negative finding in unsupervised section; the cross-domain results stand alone |
| Gemma 2 2B does not encode entity knowledge strongly | Low | Medium | Ferrando et al. (ICLR 2025) already demonstrated entity probing works on this model |
| Computational budget exceeded | Low | Low | Start with pilot (2 tasks, 1 layer), scale up only if pilot succeeds |
| sae-spelling code breaks on non-first-letter tasks | Medium | Medium | The code is modular (probing, attribution, metric are separate modules); main adaptation is in probe construction |

### Novelty Claim

1. **First cross-domain characterization of feature absorption**: No prior work measures absorption on anything other than first-letter spelling. We extend to 3 additional hierarchy types (knowledge, syntactic, semantic), establishing whether absorption is a general SAE pathology or a task-specific artifact.

2. **First unsupervised absorption detection method**: We propose and validate a probe-free absorption detection approach using decoder weight geometry, enabling absorption measurement at scale without hand-crafting supervision signals for each feature hierarchy.

3. **First quantitative analysis of hierarchy properties predicting absorption severity**: We relate absorption rate to measurable hierarchy properties (frequency ratio, co-occurrence, depth), providing actionable guidance for predicting where SAEs will fail.

4. **L0 confound analysis**: We disentangle the contribution of incorrect sparsity (hedging) from true hierarchy-driven absorption, sharpening the definition of what "absorption" actually measures in practice.

Even if the novelty is "showing that absorption generalizes beyond first-letter spelling and correlating it with hierarchy properties," this is a practically important finding that the community has explicitly called for (Gap 2 and Gap 6 in the literature survey).

# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (https://github.com/lasr-spelling/sae-spelling) — The canonical absorption metric implementation. MIT license. Contains `feature_absorption_calculator`, `probing`, `feature_ablation`, and `vocab` modules. Directly runnable experiments for latent evaluation, k-sparse probing, and feature absorption quantification. Code is well-structured but tightly coupled to the first-letter spelling task.

2. **SAELens** (https://github.com/decoderesearch/SAELens) — Standard SAE training and evaluation library. MIT license. Supports all major architectures (ReLU, Gated, JumpReLU, TopK, BatchTopK). Deep integration with TransformerLens. Loading pre-trained Gemma Scope SAEs is a one-liner: `SAE.from_pretrained(release, sae_id)`. Active community, >1100 stars.

3. **Gemma Scope SAEs** (https://huggingface.co/google/gemma-scope) — 400+ pre-trained JumpReLU SAEs on Gemma 2 2B/9B/27B, all layers, widths 1k-1M. Eliminates training cost entirely for analysis experiments. Gemma Scope 2 (Dec 2025) adds Matryoshka SAEs and transcoders on Gemma 3 family (270M-27B).

4. **SAEBench** (https://github.com/adamkarvonen/SAEBench) — 8-metric evaluation suite with 200+ SAEs benchmarked. Apache 2.0. Includes absorption metric as one of 8 tasks. Interactive results at neuronpedia.org/sae-bench. Key finding: Matryoshka SAEs dominate on absorption, RAVEL, sparse probing, and SCR.

5. **RAVEL dataset** (https://github.com/explanare/ravel) — Entity-attribute dataset with 5 entity types (cities, Nobel laureates, verbs, physical objects, occupations), 400-800 instances each, 4-6 attributes. Cities have country, continent, language attributes. Published ACL 2024. Already integrated into SAEBench for disentanglement evaluation.

6. **SAE-RAVEL** (https://github.com/MaheepChaudhary/SAE-Ravel) — Evaluates open-source SAEs on RAVEL for GPT-2 Small. Shows SAEs struggle to reach even the neuron baseline for disentangling city->country vs city->continent. Code exists and is runnable.

7. **sae_entities** (https://github.com/javiferran/sae_entities) — Ferrando et al. (ICLR 2025) code for discovering known/unknown entity latents in Gemma 2 2B/9B. Shows SAE latents encode entity recognition causally. Directly demonstrates entity-level feature hierarchies in SAEs.

8. **sparse-but-wrong-paper** (https://github.com/chanind/sparse-but-wrong-paper) — Chanin & Garriga-Alonso code for L0 analysis. Shows most open-source SAEs have L0 too low, causing feature hedging. Proposes proxy metric for correct L0.

9. **feature-hedging-paper** (https://github.com/chanind/feature-hedging-paper) — Companion code for hedging analysis. Proves narrow SAEs merge correlated features. Complementary failure mode to absorption.

10. **TransformerLens** (https://github.com/TransformerLensOrg/TransformerLens) — Hook-based activation access for 50+ models. Deep integration with SAELens. Needed for any activation extraction or intervention experiment.

11. **SynthSAEBench** (arXiv:2602.14687) — Large-scale synthetic benchmark with ground-truth features, hierarchy, correlation, Zipfian distributions. Logistic probes achieve 0.974 F1 while best SAEs substantially underperform. Enables controlled ablation of hierarchy depth, correlation structure, etc.

12. **ATM SAE** (arXiv:2510.08855) — Reports absorption score 0.0068 vs TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B. Best reported absorption scores. Per-latent importance tracking mechanism. Only evaluated on Gemma-2-2B.

### Landscape Summary

The absorption problem is well-defined and well-motivated, with a single canonical metric (Chanin et al.) that has been adopted into SAEBench. However, the field has a critical blind spot: **all absorption measurement has been done on a single task (first-letter spelling) with a single evaluation methodology**. This is not because the community is unaware of the limitation---multiple papers explicitly call it out---but because extending the metric to other domains requires (a) knowing the ground-truth feature hierarchy and (b) having a clean probe-based evaluation.

The practical gap is that RAVEL already provides exactly the kind of entity-attribute hierarchies needed (city -> country -> continent), SAE-RAVEL has already shown SAEs fail badly at disentangling these, but nobody has connected the dots: **measuring absorption specifically on RAVEL-style knowledge hierarchies**. The sae-spelling code is modular enough to adapt, and all the pre-trained SAEs and evaluation infrastructure already exist.

Meanwhile, the absorption-vs-hedging confound (Gap 9 in the literature) is genuinely important and tractable. Chanin himself has published code for both phenomena separately but nobody has run the controlled experiment varying L0 systematically while measuring both metrics on the same SAEs.

Engineering reality check: Loading a Gemma Scope SAE and running probes on RAVEL data can be done in <30 minutes on a single GPU. The hard part is not compute---it is adapting the absorption metric to work with knowledge hierarchies instead of first-letter hierarchies.

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Characterization via Knowledge Hierarchies

- **Hypothesis**: Feature absorption occurs in knowledge-domain hierarchies (city -> country, entity -> type) at rates comparable to or higher than first-letter hierarchies, and absorption severity is predictable from the hierarchy's co-occurrence statistics in the training corpus.
- **Implementation sketch**: (1) Use RAVEL dataset for entity-attribute pairs (cities: country, continent, language). (2) Train logistic regression probes on model residual stream activations for each attribute using TransformerLens. (3) Port the sae-spelling absorption metric to work with RAVEL probes instead of first-letter probes: identify k-sparse feature splits, find false negatives, run integrated-gradients ablation. (4) Measure absorption rate across Gemma Scope SAEs (multiple layers, widths 16k/65k). (5) Compare absorption rates across hierarchy types and relate to co-occurrence frequency of entity-attribute pairs in training data.
- **Simplest version**: Run the absorption metric on city->country probe using a single Gemma Scope SAE (layer 12, width 16k) and compare to the first-letter absorption rate from the same SAE.
- **Time estimate**: Pilot 15 min (single layer/width), full experiment 2-4 GPU-hours (all layers, 2 widths, 3 attribute pairs).
- **Reusable components**: sae-spelling (absorption metric code), SAELens (SAE loading), RAVEL (entity data), TransformerLens (activation hooks), Gemma Scope (pre-trained SAEs).

### Candidate B: Disentangling Absorption from Hedging via Controlled L0 Sweeps

- **Hypothesis**: A significant fraction (>30%) of what is measured as "absorption" in standard-L0 SAEs is actually L0-induced feature hedging, and the two phenomena can be separated by measuring absorption rate as a function of L0 at fixed SAE width.
- **Implementation sketch**: (1) Use the sparse-but-wrong-paper's L0 analysis to identify the "correct" L0 for Gemma Scope SAEs. (2) Train a set of TopK SAEs at varying k values (L0 = 20, 50, 100, 200, 400) on the same model/layer using SAELens. (3) Measure absorption rate using sae-spelling metric at each L0 value. (4) Simultaneously measure feature hedging using the hedging metric from feature-hedging-paper. (5) Plot absorption vs hedging as a function of L0, identifying the crossover point where hedging dominates vs where true hierarchy-driven absorption dominates.
- **Simplest version**: Use 3 existing Gemma Scope SAEs with different L0 values on a single layer and measure absorption + hedging on each.
- **Time estimate**: Pilot 15 min (3 existing SAEs, 1 layer). Full experiment: 8-12 GPU-hours if training new SAEs, <2 hours if using only existing Gemma Scope SAEs with varying L0.
- **Reusable components**: sae-spelling (absorption), feature-hedging-paper (hedging metric), sparse-but-wrong-paper (L0 analysis), SAELens (training), Gemma Scope (pre-trained SAEs).

### Candidate C: Unsupervised Absorption Detection via Decoder Geometry

- **Hypothesis**: Absorbed features can be detected without probe directions by identifying SAE decoder vectors that have anomalously high cosine similarity to other decoder vectors while activating on a strict subset of the parent vector's activation contexts.
- **Implementation sketch**: (1) For a pre-trained SAE, compute pairwise cosine similarity of all decoder vectors. (2) Identify high-similarity pairs (threshold > 0.3). (3) For each pair, compute the activation overlap ratio: how often does the child activate vs the parent on the same inputs? (4) Flag pairs where child activates on a strict subset of parent's contexts (absorption candidates). (5) Validate against the supervised absorption metric (sae-spelling) on first-letter task as ground truth.
- **Simplest version**: Compute decoder cosine similarity matrix for one Gemma Scope SAE and check whether known absorbed features from Chanin et al. appear as high-similarity pairs.
- **Time estimate**: Pilot 10 min (cosine similarity computation is trivial), full validation 1-2 GPU-hours.
- **Reusable components**: SAELens (SAE loading/decoding), TransformerLens (activation collection), sae-spelling (ground truth validation).

## Phase 3: Self-Critique

### Against Candidate A

- **Implementation reality check**: The RAVEL dataset and evaluation code exist and are public. SAE-RAVEL (Chaudhary & Geiger, 2024) already ran SAE interventions on RAVEL but measured disentanglement, not absorption. The sae-spelling absorption metric relies on (a) training LR probes, (b) finding k-sparse feature splits, (c) running integrated-gradients ablation. Steps (a) and (b) are generic and transfer directly to any binary classification task. Step (c) requires adapting the spelling grader to a knowledge-recall grader. This is ~200 lines of code adaptation, not a rewrite. **Feasible.**
- **Reproducibility attack**: The probe training is standard (LR on residual stream activations). The k-sparse probing is from sae-spelling, well-documented. The only fragile part is the ablation threshold for detecting absorption (cosine similarity > 0.025, ablation effect >= 1.0 larger than second-highest). These thresholds were tuned for first-letter tasks and may need adjustment for knowledge tasks. Risk: if thresholds are not robust, results depend on arbitrary choices. Mitigation: report results across a sweep of thresholds.
- **Baseline sanity check**: The natural baseline is: "do SAE probes perform worse than LR probes on RAVEL attributes?" SAE-RAVEL already showed SAEs struggle to reach the neuron baseline. This means there IS signal to find---the question is whether the performance gap is specifically due to absorption vs other causes (hedging, dead features, etc.). The paper must carefully decompose the gap.
- **Scope attack**: RAVEL covers 5 entity types with 4-6 attributes each. This gives 20-30 hierarchy pairs to test, far more than the 26 letters in first-letter task. If absorption rates are consistent across entity types, generality is strong. If they vary wildly, the paper becomes a characterization study (still valuable but less clean).
- **Verdict**: **STRONG** --- clear gap in literature, existing tools cover 80% of implementation, moderate novelty, high practical value.

### Against Candidate B

- **Implementation reality check**: The core experiment (measure absorption at varying L0) is straightforward IF you have SAEs at different L0 values. Gemma Scope SAEs are all JumpReLU with specific L0 targets---there is limited L0 variation within the existing suite. Training new SAEs at controlled L0 values would require SAELens training runs (~2-4 hours each for Gemma 2 2B). This adds significant GPU cost (5 L0 values x 2 hours = 10 GPU-hours minimum per layer). The hedging metric from feature-hedging-paper is less mature than the absorption metric and may require non-trivial adaptation.
- **Reproducibility attack**: Training SAEs is somewhat stochastic. Dead features, initialization effects, and training instability can confound L0 sweeps. Need multiple seeds per L0 value, which multiplies compute. The "correct L0" determination from sparse-but-wrong-paper is itself approximate. Risk: the experiment may show a smooth gradient rather than a clean crossover, making the story less compelling.
- **Baseline sanity check**: Chanin et al. already show absorption rates across different L0 values in their paper (Fig 4/5). The novel contribution here is specifically measuring hedging simultaneously and decomposing the total failure into absorption + hedging components. If the decomposition is messy (absorption and hedging both increase together rather than trading off), the paper's thesis weakens.
- **Scope attack**: Limited to first-letter task (unless combined with Candidate A), limited to Gemma 2 2B (unless cross-model validation is added). The L0 sweep is specific to a single SAE architecture choice.
- **Verdict**: **MODERATE** --- important question, but high compute cost, uncertain payoff, and the existing papers already partially address this.

### Against Candidate C

- **Implementation reality check**: Computing pairwise cosine similarity of decoder vectors for a 16k-width SAE produces a 16k x 16k matrix (256M entries). This is computationally cheap and fits in memory. However, the core hypothesis (absorbed features appear as high-cosine-similarity pairs with subset activation patterns) is unvalidated. The Chanin et al. paper mentions that absorbing latents have cosine similarity > 0.025 with the probe direction, which is extremely low---this suggests absorbed features may NOT have high decoder cosine similarity with each other. The hypothesis may simply be wrong.
- **Reproducibility attack**: The method is fully deterministic (cosine similarity + activation counting). Reproducibility is excellent. But the validation against ground truth (sae-spelling absorption labels) is the weak link: if the method finds different features than the supervised method, which one is "right"?
- **Baseline sanity check**: A random baseline would be: what fraction of high-cosine-similarity decoder pairs are actually absorption? If the false positive rate is high, the unsupervised method is useless. There is no prior work suggesting the cosine similarity threshold would cleanly separate absorbed vs non-absorbed features.
- **Scope attack**: If it works on first-letter features, does it work on other hierarchies? Unknown. The method's appeal is domain-generality, but validation is only possible where ground truth exists (first-letter task).
- **Verdict**: **WEAK** --- the core hypothesis is not well-supported by existing evidence (cosine similarity with probe direction is very low in Chanin et al.), high risk of failure, and validation is limited to domains where supervised methods already work.

## Phase 4: Refinement

**Dropped**: Candidate C (unsupervised absorption detection). The cosine similarity hypothesis is too speculative. The canonical absorption results show absorbed features have cosine similarity > 0.025 with the probe direction---this is nearly orthogonal, suggesting decoder geometry alone will not capture absorption. A future version could use activation co-occurrence statistics instead, but that is a different and more complex method.

**Strengthened Candidate A** (front-runner):
- Simplified scope: Focus on cities only (the richest entity type in RAVEL with 3000+ cities and clear hierarchical attributes: country, continent, language, timezone).
- Added L0 sensitivity analysis: Measure absorption on 2-3 different L0 SAEs from Gemma Scope to partially address the L0 confound without training new SAEs.
- Added layer sweep: Measure absorption across all available Gemma Scope layers (0-25) to produce layer-wise absorption profiles for knowledge features, analogous to first-letter profiles in Chanin et al.
- Added comparison with Matryoshka SAEs: Gemma Scope 2 includes Matryoshka SAEs on Gemma 3. If these are available via SAELens, compare absorption rates to confirm whether Matryoshka advantage transfers to knowledge domains.
- Confirmed code exists: RAVEL data (explanare/ravel), probe training (sae-spelling/probing), absorption metric (sae-spelling/feature_absorption_calculator), SAE loading (SAELens), activation hooks (TransformerLens).

**Folded in elements from Candidate B**:
- Include a targeted L0 analysis using existing Gemma Scope SAEs (no training needed). If Gemma Scope provides SAEs at 2-3 sparsity levels per layer, measure absorption at each and compare to Chanin et al.'s first-letter scaling. This addresses the absorption-vs-hedging confound without the cost of training new SAEs.

**Selected front-runner**: Candidate A with B elements --- "Cross-Domain Feature Absorption: Measuring Absorption in Knowledge Hierarchies Beyond First-Letter Spelling"

**Why this wins on probability of success**:
1. All infrastructure exists (sae-spelling + RAVEL + SAELens + Gemma Scope).
2. The adaptation is mechanical, not creative (swap first-letter probes for RAVEL probes).
3. The gap is explicitly called out in multiple papers (Chanin et al. Gap 2 & 6, literature survey Gap 2).
4. Negative results are publishable: if absorption rates differ dramatically between syntactic and knowledge hierarchies, that is itself an important finding.
5. A single GPU can run the full experiment in <4 hours.

## Phase 5: Final Proposal

### Title

Cross-Domain Feature Absorption in Sparse Autoencoders: From Spelling to Knowledge Hierarchies

### Hypothesis

Feature absorption is not specific to the first-letter spelling task. SAE latents representing hierarchical knowledge (city -> country -> continent) exhibit absorption at rates comparable to or exceeding first-letter absorption rates (15-35% as reported in Chanin et al.), and the absorption severity correlates with the co-occurrence frequency of the parent-child feature pair in the model's training distribution.

Precisely falsifiable prediction: On Gemma 2 2B with Gemma Scope 16k SAEs at layers 8-17, the absorption rate for city-country features (measured via adapted Chanin metric) exceeds 10%.

### Motivation

Feature absorption is the most important identified failure mode of SAEs for mechanistic interpretability. It creates a false sense of interpretability: an SAE latent appears to track "starts with S" but silently fails on arbitrary tokens. The canonical measurement uses only the first-letter spelling task---a synthetic, controlled setting with clear feature hierarchy (letter membership implies specific token identity). This leaves open the critical question: does absorption occur in the semantically rich knowledge hierarchies that actually matter for safety and interpretability research?

RAVEL (Huang et al., ACL 2024) provides entity-attribute datasets with natural hierarchies (cities: country, continent, language). SAE-RAVEL (Chaudhary & Geiger, 2024) already showed SAEs fail badly at disentangling city->country vs city->continent, but did not characterize this failure as absorption specifically. Ferrando et al. (ICLR 2025) showed SAE latents encode entity recognition (known vs unknown) causally, demonstrating entity-level features exist in SAEs. The missing piece is measuring absorption on these features.

This matters because: (1) DeepMind deprioritized SAE research partly because SAE probes fail on safety-relevant downstream tasks---if absorption is the cause, quantifying it motivates specific mitigations; (2) Anthropic's circuit tracing program relies on SAE features being reliable---absorption in knowledge features would undermine this.

### Method

**Step 1: Data Preparation** (~30 min)
- Load RAVEL city dataset (3000+ cities with country, continent, language attributes).
- Construct prompt templates analogous to sae-spelling: "{City} is located in the country:" (for country probes), "{City} is on the continent:" (for continent probes), "{City} speaks the language:" (for language probes).
- Split into train/test (80/20, stratified by attribute value).

**Step 2: Probe Training** (~30 min per attribute)
- Load Gemma 2 2B via TransformerLens.
- Extract residual stream activations at each layer (0-25) for all city prompts.
- Train multi-class logistic regression probes for each attribute at each layer using sae-spelling's `train_multi_probe()`.
- Validate probe accuracy (expect >90% for mid-layer probes based on prior work).

**Step 3: SAE Absorption Measurement** (~15 min per SAE)
- Load Gemma Scope SAEs (16k and 65k widths) via SAELens.
- For each SAE at each layer:
  1. Run k-sparse probing to find feature splits for each attribute value.
  2. Identify false-negative inputs (all k splits inactive, but probe classifies correctly).
  3. Run integrated-gradients ablation on false negatives using adapted grading metric (logit of correct attribute value minus mean logit of incorrect values).
  4. Apply absorption detection threshold (cosine similarity with probe > threshold, ablation magnitude dominance > 1.0).
  5. Compute absorption rate.

**Step 4: Cross-Domain Comparison** (~1 hr)
- Run the original first-letter absorption measurement on the same SAEs using sae-spelling code directly.
- Compare absorption rates: first-letter vs country vs continent vs language.
- Correlate absorption rates with co-occurrence statistics (estimated from OpenWebText or Pile).

**Step 5: L0 Sensitivity and Architecture Comparison** (~1 hr)
- If Gemma Scope provides SAEs at multiple L0 values per layer, measure absorption at each L0.
- If Gemma Scope 2 Matryoshka SAEs are available, compare absorption rates to standard JumpReLU.

**Libraries/repos used**:
- `sae-spelling` (MIT): absorption metric, probing, vocabulary tools
- `SAELens` (MIT): SAE loading and encoding
- `TransformerLens` (MIT): model loading and activation hooks
- `RAVEL` dataset: entity-attribute data
- Gemma Scope SAEs (HuggingFace): pre-trained SAEs

### Simplest Version

Load a single Gemma Scope 16k SAE at layer 12 on Gemma 2 2B. Train country probes on RAVEL city data. Run the absorption metric on 3 high-frequency countries (France, USA, Germany) and 3 low-frequency countries (Tuvalu, Kiribati, Nauru). Report absorption rate and compare to first-letter absorption rate from the same SAE. This takes ~15 minutes on a single GPU.

### Baselines

1. **First-letter absorption baseline** (Chanin et al.): Replicate the first-letter absorption measurement on the same SAEs. Expected absorption rate: 15-35% (Chanin et al. report). This is the direct comparison target.

2. **Random hierarchy baseline**: Assign random "attribute" labels to cities and measure "absorption" on random labels. Expected absorption rate: ~0% (no real hierarchy to absorb). This validates the metric is not producing spurious positives.

### Experimental Plan

| Experiment | SAEs | Layers | Attributes | GPU-hours |
|-----------|------|--------|------------|-----------|
| Pilot: Single SAE, 3+3 cities | 1 (16k, L12) | 1 | country | 0.25 |
| Layer sweep: All layers | 1 (16k) | 0-25 | country | 2 |
| Width comparison | 2 (16k, 65k) | 8-17 | country, continent | 3 |
| Cross-attribute | 2 (16k, 65k) | 8-17 | country, continent, language | 4 |
| First-letter replication | 2 (16k, 65k) | 8-17 | first-letter | 2 |
| L0 sensitivity | 3-4 SAEs varying L0 | 12 | country | 1 |
| Matryoshka comparison (if available) | 2 (JumpReLU, Matryoshka) | 8-17 | country | 2 |

**Ablation schedule**:
1. Absorption threshold sensitivity: Sweep cosine similarity threshold (0.01, 0.025, 0.05, 0.1) and dominance threshold (0.5, 1.0, 2.0).
2. Probe quality check: Only include attributes where probe accuracy > 85%.
3. k-sparse k sweep: k = 1, 3, 5, 10 to check stability.

**Metrics reported**:
- Absorption rate (fraction of attributes with at least one absorbed instance)
- Mean absorption fraction (fraction of false-negative activations attributable to absorbing latents)
- Probe-vs-SAE recall gap (how much worse SAE features are than probes)
- Layer-wise absorption profile
- Absorption rate vs attribute co-occurrence frequency scatter plot

### Resource Estimate

- **Model**: Gemma 2 2B (~5 GB VRAM)
- **SAE**: Gemma Scope 16k (~2 GB VRAM per SAE)
- **Total VRAM**: ~10-12 GB (fits on single RTX 3090 or A100)
- **Pilot**: 15 minutes, 1 GPU
- **Full experiment**: 4-6 GPU-hours, single GPU
- **No SAE training required**: All SAEs are pre-trained

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Probe quality too low for knowledge attributes | High | Low | RAVEL probes achieve >90% accuracy in prior work; add probe quality filter |
| Absorption metric thresholds do not transfer from spelling to knowledge tasks | Medium | Medium | Sweep thresholds; report ROC curve of detection sensitivity |
| sae-spelling code too tightly coupled to first-letter task | Medium | Low | Core modules (probing, ablation) are generic; only SpellingGrader needs replacement |
| Gemma 2 2B does not encode knowledge hierarchies cleanly | Medium | Low | Ferrando et al. (ICLR 2025) confirmed entity knowledge features in Gemma 2 2B SAEs |
| Absorption rates near zero (no absorption in knowledge domain) | Low | Low | This is a publishable negative result that would distinguish knowledge from syntactic features |
| Gemma Scope 2 Matryoshka SAEs not yet loadable via SAELens | Low | Medium | Fall back to JumpReLU-only comparison; still novel contribution |

### Novelty Claim

The novelty is showing whether feature absorption generalizes beyond its single-task demonstration. Specifically:

1. **First measurement of absorption on knowledge hierarchies**: No prior work measures absorption rate on entity-attribute tasks. This fills Gap 2 and Gap 6 from the literature survey.
2. **Cross-domain comparison**: Side-by-side absorption rates for syntactic (first-letter) vs semantic (knowledge) hierarchies on the same SAEs. This is the first apples-to-apples comparison.
3. **Co-occurrence frequency as predictor**: If absorption correlates with co-occurrence frequency, this provides the first empirical scaling law for absorption severity (partially addressing Gap 1).
4. **Practical implications for safety**: If knowledge-domain absorption is high, this directly explains why SAE probes fail on downstream tasks (DeepMind's negative results) and motivates architecture-level mitigations.

The novelty is "showing that X (a well-known problem) also applies to Y (the domain that actually matters)." This is not flashy, but it is the kind of empirical contribution that the field urgently needs to move forward.

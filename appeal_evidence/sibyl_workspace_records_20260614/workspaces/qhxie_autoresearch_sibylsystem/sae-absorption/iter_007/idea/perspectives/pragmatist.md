# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (GitHub: lasr-spelling/sae-spelling, MIT) --- Canonical absorption metric implementation by Chanin et al. Includes `feature_absorption_calculator`, `probing`, `feature_attribution`, `feature_ablation` modules. Directly reusable. Poetry-managed, well-structured code.

2. **SAEBench** (GitHub: adamkarvonen/SAEBench, Apache 2.0) --- 8-metric evaluation suite including absorption. 200+ pretrained SAEs across 7 architectures open-sourced. Single SAE evaluation ~115 min on RTX 3090. Includes RAVEL (entity disentanglement) and sparse probing. Installable via pip. Interactive results at neuronpedia.org/sae-bench.

3. **SAELens v6** (GitHub: decoderesearch/SAELens, MIT) --- Standard SAE training/evaluation library. Supports all major architectures (TopK, JumpReLU, Gated, Matryoshka). Deep integration with TransformerLens. `pip install sae-lens`. Loading pretrained SAEs is ~3 lines of code.

4. **TransformerLens** (GitHub: TransformerLensOrg/TransformerLens, MIT) --- Hook-based activation caching/editing. 50+ model support including GPT-2 and Gemma 2. Essential for activation extraction and probe training.

5. **Gemma Scope SAEs** (HuggingFace: google/gemma-scope) --- 400+ JumpReLU SAEs on Gemma 2 (2B/9B/27B). Widths 1k-1M. All layers/sublayers. Free to use for research. Gemma Scope 2 adds Matryoshka SAEs on Gemma 3.

6. **RAVEL benchmark** (GitHub: explanare/ravel, HuggingFace: hij/ravel) --- Entity-attribute disentanglement benchmark with 3000+ cities (country, continent, language attributes) and Nobel Prize winners (country of birth, field, gender). Already integrated into SAEBench. Provides natural knowledge hierarchies.

7. **sae-entities** (GitHub: javiferran/sae_entities) --- ICLR 2025 paper code. SAE latents for entity recognition in Gemma 2. Identifies known/unknown entity pairs across cities, movies, songs. Demonstrates entity-level features exist in SAEs.

8. **"A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) --- The canonical paper. Absorption rate 15-35% across all tested SAEs. Only evaluated on first-letter spelling task. No solution proposed. Absorption metric requires known probe directions.

9. **Matryoshka SAE** (arXiv:2503.17547, ICML 2025) --- Best absorption score (~0.03 vs BatchTopK ~0.29). Nested prefix losses. Code available via SAELens.

10. **"Are Sparse Autoencoders Useful?"** (arXiv:2502.16681, ICML 2025) --- SAE probes underperform logistic regression baselines across all regimes. Key context for why absorption matters: it is one reason SAEs lose information.

11. **Feature sensitivity metric** (arXiv:2509.23717) --- Frames absorption as special case of poor sensitivity. Scalable evaluation method that does not require specific probe directions.

12. **Masked Regularization** (arXiv:2604.06495, April 2026) --- Most recent mitigation. Token masking during training to disrupt co-occurrence. Simple to implement.

### Landscape Summary

**What actually works in practice:**
- Matryoshka SAEs are the clear winner on absorption benchmarks (~0.03 vs ~0.29 for BatchTopK), but trade absorption for hedging.
- OrtSAE reduces absorption ~70% via orthogonality penalty with linear overhead.
- ATM achieves the lowest reported absorption (0.0068) but is only evaluated on Gemma-2-2B.
- All mitigations are evaluated in isolation on different models/layers/metrics. No head-to-head comparison exists.

**What does not work:**
- JumpReLU SAEs (Gemma Scope default) have the worst absorption---training longer makes it worse.
- TopK/BatchTopK significantly worsen absorption at low L0.
- Dense probes dramatically outperform SAE probes on downstream tasks, with absorption being a key culprit.

**The practical gap:**
- Absorption has ONLY been measured on the first-letter spelling task.
- RAVEL already provides entity-attribute hierarchies (city->country, city->language) that could serve as cross-domain absorption test beds.
- The sae-entities code proves entity-level features exist in SAEs, so the features to measure absorption against are available.
- All the code pieces exist separately but nobody has connected them.

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy Using Existing Benchmarks

- **Hypothesis**: Feature absorption rates vary systematically across different types of semantic hierarchies (syntactic vs. knowledge vs. attribute), and the hierarchy's co-occurrence statistics and depth predict absorption severity.
- **Implementation sketch**: Extend sae-spelling's absorption metric to new probe tasks: (1) RAVEL's city->country/continent/language hierarchy, (2) entity type recognition (sae-entities code), (3) part-of-speech/syntactic hierarchies. Use Gemma Scope pretrained SAEs + SAEBench's pretrained SAEs. All infrastructure exists.
- **Simplest version**: Take sae-spelling's `feature_absorption_calculator`, replace the first-letter probes with RAVEL's city->country probes, run on 3-5 Gemma Scope SAEs. Measure absorption rate. Compare to first-letter baseline.
- **Time estimate**: Pilot 15 min (single SAE, single hierarchy). Full experiment ~3-4 hours across multiple SAEs and hierarchies, parallelizable.
- **Reusable components**: sae-spelling (absorption metric), SAELens (SAE loading), RAVEL dataset (hierarchies), Gemma Scope (pretrained SAEs), TransformerLens (activations).

### Candidate B: Disentangling Absorption from Hedging via Controlled Width-Sparsity Sweeps

- **Hypothesis**: Observed feature failure-to-fire in practice is a mixture of true hierarchy-driven absorption and L0-induced hedging, and these can be separated by measuring failure rates across a 2D grid of SAE width x L0 while holding other factors constant.
- **Implementation sketch**: Use SAEBench's 200+ pretrained SAEs (trained across 3 widths x 6 sparsities x 7 architectures on matched models). For each SAE, measure both absorption (via sae-spelling metric) and hedging (via Chanin et al. hedging metric from feature-hedging-paper). Plot 2D maps of absorption vs hedging as a function of width and L0.
- **Simplest version**: Take SAEBench's pretrained SAEs on Gemma-2-2B layer 12 (one architecture, e.g., BatchTopK, across all widths/sparsities). Measure absorption on first-letter task. Measure hedging. See if they trade off as width/L0 change.
- **Time estimate**: Pilot 10 min (3 SAEs). Full experiment ~2-3 hours for full sweep (SAEBench SAEs are already downloaded).
- **Reusable components**: SAEBench (pretrained SAEs + absorption eval code), feature-hedging-paper (hedging metric), sae-spelling (absorption metric).

### Candidate C: Decoder Geometry as Unsupervised Absorption Predictor

- **Hypothesis**: The cosine similarity structure of SAE decoder columns predicts which features will experience absorption, without needing supervised probe directions. Specifically, high cosine similarity between a "child" decoder direction and a "parent" decoder direction predicts that the parent feature's recall will be degraded by absorption.
- **Implementation sketch**: For each SAE, compute the pairwise cosine similarity matrix of decoder columns. Identify clusters of high-similarity features. For features within these clusters, measure their actual absorption rate via the supervised metric. Correlate predicted absorption (from decoder geometry) with actual absorption.
- **Simplest version**: Take one Gemma Scope 16k SAE. Compute decoder cosine similarity matrix. Identify top-50 most similar feature pairs. Check which of these exhibit absorption on the first-letter task (where ground truth is available).
- **Time estimate**: Pilot 5 min (cosine similarity is just matrix multiplication). Full experiment ~1 hour including probe training and validation.
- **Reusable components**: SAELens (load SAE decoder weights), sae-spelling (ground truth absorption), numpy/torch (cosine similarity).

## Phase 3: Self-Critique

### Against Candidate A

- **Implementation reality check**: The sae-spelling absorption metric depends on having a well-defined hierarchical feature structure where you know the "parent" and "child" features in advance, then training probes for the parent. For first-letter spelling, this is clean: parent = "starts with S", children = specific tokens starting with S. For city->country, the hierarchy is: parent = "located in France", children = specific French cities. This is structurally identical---we just need to train LR probes for "located in country X" and check which city tokens fail to activate the right SAE latents. The RAVEL dataset provides exactly this structure. The sae-entities paper confirms entity features exist. **This is very doable.**
- **Reproducibility attack**: The metric is well-defined and code exists. The main fragility is in probe quality---if probes for "located in country X" are poor, absorption measurement is meaningless. Mitigation: report probe accuracy alongside absorption rates, exclude hierarchies where probe accuracy is below a threshold (e.g., 0.85 F1).
- **Baseline sanity check**: The baseline is: "absorption on first-letter spelling task generalizes to other hierarchies at similar rates (15-35%)." If we find similar rates, the contribution is establishing universality. If rates differ systematically, that is the main finding. Either outcome is publishable.
- **Scope attack**: The risk is that results only hold for Gemma-2-2B with Gemma Scope SAEs. Mitigation: include GPT-2 SAEs as a second model (cheaper to run, well-supported by SAELens).
- **Verdict**: **STRONG**. All code exists, the extension is straightforward, and either outcome (similar or different absorption rates) is a contribution.

### Against Candidate B

- **Implementation reality check**: SAEBench already provides 200+ SAEs across the needed width/sparsity grid. The absorption metric code is in SAEBench. The hedging metric code is in feature-hedging-paper. The main engineering challenge is running the hedging metric on SAEBench SAEs (different codebase). Searched for practical experience: the hedging paper uses its own SAE training setup, not SAEBench SAEs. Adapting the hedging metric to SAEBench SAEs requires some code glue but is not hard.
- **Reproducibility attack**: The 2D grid analysis is clean and reproducible. All SAEs are public. The main risk: hedging and absorption metrics may not be directly comparable (different units, different normalization). Need careful experimental design.
- **Baseline sanity check**: The balanced Matryoshka SAE paper already shows absorption and hedging trade off. Our contribution would be a more systematic characterization across the full width/L0 space, not just Matryoshka vs. vanilla.
- **Scope attack**: This is limited to SAEBench's model/layer choices (Pythia-160M L8, Gemma-2-2B L12). Acceptable for a first study, but results may not generalize to all settings.
- **Verdict**: **MODERATE**. Useful but incremental. The balanced Matryoshka paper already established the absorption-hedging trade-off; this extends it to a 2D grid but the novelty increment is modest.

### Against Candidate C

- **Implementation reality check**: Computing decoder cosine similarity is trivial. But the hypothesis that high decoder similarity *predicts* absorption is unvalidated. The "Dense SAE Latents Are Features, Not Bugs" paper shows antipodal pairs exist in decoder geometry, but does not connect this to absorption. The OrtSAE paper shows reducing cosine similarity reduces absorption, which is suggestive but indirect.
- **Reproducibility attack**: The correlation between decoder geometry and absorption may be weak or confounded by other factors (feature frequency, L0 regime). If the correlation is weak, the paper is a negative result.
- **Baseline sanity check**: A simpler baseline for predicting absorption: "features that co-occur frequently will exhibit absorption." This co-occurrence baseline is much simpler and may explain as much variance. If decoder geometry does not beat co-occurrence statistics, the story is weak.
- **Scope attack**: Unsupervised absorption detection is important, but this specific approach (decoder cosine similarity) is just one of many possible signals. If it does not work well, the paper needs to explore multiple alternatives.
- **Verdict**: **MODERATE-to-WEAK**. High risk that the signal is too weak. Better as a component of a larger study than a standalone paper.

## Phase 4: Refinement

**Dropped**: Candidate C as a standalone proposal. Decoder geometry analysis is interesting but too speculative for a primary contribution. It can be included as an exploratory analysis within Candidate A.

**Strengthened Candidate A** by incorporating elements from B and C:

The refined proposal is a **systematic cross-domain absorption characterization** that:
1. **Primary contribution**: Extends absorption measurement beyond first-letter spelling to multiple semantic hierarchies (RAVEL city attributes, entity type recognition, part-of-speech). Establishes whether absorption is universal or domain-dependent.
2. **Secondary contribution**: Within each domain, varies SAE width and L0 (using SAEBench's pretrained SAEs) to characterize how absorption scales. This subsumes the useful parts of Candidate B.
3. **Exploratory analysis**: Tests whether decoder cosine similarity or feature co-occurrence statistics predict which features get absorbed. This includes the core of Candidate C as a bonus analysis.

**Confirmed code existence:**
- `sae-spelling/sae_spelling/feature_absorption_calculator.py` --- core absorption metric, reusable
- `SAEBench` --- pip-installable, includes RAVEL evaluation, pretrained SAEs
- `sae-entities` --- entity recognition in Gemma Scope, ICLR 2025
- `SAELens` --- all SAE loading, `pip install sae-lens`
- `TransformerLens` --- activation extraction
- RAVEL dataset on HuggingFace (`hij/ravel`) --- 3000+ cities with attributes

**Minimal pilot experiment (< 15 min):**
1. Load Gemma-2-2B + one Gemma Scope 16k SAE via SAELens (~1 min)
2. Load RAVEL city->country dataset (~30 sec)
3. Train LR probes for "city is in country X" on model activations (~2 min)
4. Adapt sae-spelling absorption calculator to use country probes instead of letter probes (~5 min code adaptation)
5. Measure absorption rate on this hierarchy (~5 min)

This gives immediate signal on whether absorption generalizes beyond spelling.

**Selected front-runner**: Candidate A (refined) --- highest success probability because it builds entirely on existing, working code, addresses the most cited gap in the literature (Gap 2: absorption only studied on spelling), and produces useful results regardless of outcome.

## Phase 5: Final Proposal

### Title
Beyond First Letters: A Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders

### Hypothesis
Feature absorption in SAEs is not an artifact of the first-letter spelling task but a universal phenomenon that occurs whenever hierarchical features are present. However, the severity of absorption varies systematically with (a) the type of semantic hierarchy, (b) the depth and branching factor of the hierarchy, and (c) SAE configuration (width, L0, architecture).

Precisely falsifiable predictions:
- Absorption rate on knowledge hierarchies (city->country) will be >10% (i.e., non-trivial, as in the spelling task).
- Absorption rate will correlate with co-occurrence frequency between parent and child features (Spearman rho > 0.3).
- Wider SAEs will show higher absorption rates at matched L0 (consistent with Chanin et al.'s finding).
- Matryoshka SAEs will show lower absorption than JumpReLU/TopK across all domains (not just spelling).

### Motivation
Feature absorption is the most fundamental failure mode of SAEs---it creates systematic "holes" in feature recall that undermine downstream interpretability. However, everything we know about absorption comes from a single narrow task (first-letter spelling). If SAE-based interpretability is to be trusted for safety-critical applications (bias detection, harmful intent classification), we need to know whether absorption occurs equally in knowledge-level features. RAVEL's entity-attribute hierarchies and existing entity recognition work provide the perfect test bed, and all code infrastructure already exists.

### Method

**Step 1: Define hierarchical probe tasks** (Training-free for SAEs; only probe training needed)

| Domain | Parent Feature | Child Features | Data Source | Hierarchy Depth |
|--------|---------------|----------------|-------------|-----------------|
| Spelling (baseline) | "starts with letter X" | Specific tokens starting with X | sae-spelling | 2 (letter -> token) |
| Geography-Country | "located in country X" | Specific cities in country X | RAVEL (hij/ravel) | 2 (country -> city) |
| Geography-Continent | "located in continent X" | Specific cities in continent X | RAVEL | 2 (continent -> city) |
| Language | "language spoken is X" | Specific cities where X is spoken | RAVEL | 2 (language -> city) |
| Entity Type | "is a city/person/song" | Specific entities of that type | sae-entities | 2 (type -> entity) |

**Step 2: Train linear probes for each parent feature**
- Logistic regression probes on model activations (same as sae-spelling)
- Report probe F1 scores as quality gate (exclude probes with F1 < 0.85)
- Use 80/20 train/test split

**Step 3: Adapt and run absorption metric for each domain**
- Core adaptation: replace `sae_spelling.prompting` with domain-specific prompt templates
- Use `sae_spelling.feature_absorption_calculator` with new probe directions
- For each (domain, SAE) pair, compute: absorption rate, false-negative rate, number of absorbing latents

**Step 4: Cross-domain analysis**
- Compare absorption rates across domains (main result)
- Correlate absorption rate with hierarchy properties: co-occurrence frequency, number of child features, feature frequency ratio (parent/child)
- Test all predictions listed in the hypothesis

**Step 5: Architecture comparison**
- Run on SAEBench's pretrained SAEs (BatchTopK, TopK, JumpReLU, Gated, Matryoshka) at matched widths/sparsities
- Confirm/refute whether Matryoshka advantage holds beyond spelling

**Step 6: Exploratory --- Structural predictors of absorption**
- Compute decoder cosine similarity between absorbing and absorbed feature directions
- Compute token co-occurrence statistics for parent/child feature pairs
- Build a simple linear model predicting absorption rate from these structural features

Libraries used: SAELens (SAE loading), TransformerLens (activation extraction), sae-spelling (absorption metric core), SAEBench (pretrained SAEs + RAVEL integration), RAVEL dataset (HuggingFace), sae-entities (entity features reference).

### Simplest Version
One Gemma Scope 16k SAE (layer 12, Gemma-2-2B) + one hierarchy (city->country from RAVEL) + the first-letter spelling baseline. Measure absorption rate on both. If absorption occurs on city->country at a non-trivial rate, the core claim is validated. This can be done in under 30 minutes on a single GPU.

### Baselines
1. **First-letter spelling absorption** (Chanin et al., 2024): The established baseline. Expected absorption rate 15-35% on Gemma Scope SAEs. We reproduce this as validation.
2. **Random-feature baseline**: Measure "absorption" using randomly selected latents as fake absorbers. This establishes the null hypothesis (noise floor). Expected: near 0%.

### Experimental Plan

**Phase 0: Validation** (~30 min)
- Reproduce Chanin et al. first-letter absorption results on Gemma Scope 16k, layer 12
- Confirm our adapted pipeline produces matching results

**Phase 1: Cross-domain characterization** (~2-3 hours)
- 5 hierarchies x 5-10 SAEs = 25-50 measurements
- Each measurement: train probes (~2 min) + run absorption (~5 min)
- Parallelize across SAEs

**Phase 2: Width/L0 sensitivity** (~2-3 hours)
- Use SAEBench SAEs: 3 widths x 6 sparsities x 1 architecture (BatchTopK) = 18 SAEs
- Run on 2 domains (spelling + city->country)
- Produces 2D absorption heatmaps

**Phase 3: Architecture comparison** (~2-3 hours)
- SAEBench's 7 architectures at 2 widths x 2 sparsities = 28 SAEs
- Run on all 5 domains
- Produces architecture ranking table

**Phase 4: Structural analysis** (~1 hour)
- Decoder cosine similarity computation
- Co-occurrence statistics
- Regression analysis

**Total GPU hours**: ~8-10 hours, all parallelizable. No training required (all SAEs are pretrained).

**Metrics:**
- Absorption rate (primary, from sae-spelling)
- False-negative rate (diagnostic)
- Probe F1 (quality gate)
- Number of absorbing latents per feature (structural insight)
- Decoder cosine similarity (exploratory)

**Ablation schedule:**
1. Domain ablation: How much does absorption vary across domains?
2. Width ablation: How does absorption scale with dictionary size?
3. L0 ablation: How does absorption change with sparsity target?
4. Architecture ablation: Which architectures resist absorption best?
5. Layer ablation: Does absorption vary across model layers?

### Resource Estimate
- **GPU**: Single GPU (RTX 3090 or A100) sufficient. Gemma-2-2B fits in 16GB with SAE.
- **Wall-clock time**: 8-10 hours total for all experiments. Pilot in 30 minutes.
- **Model sizes**: Gemma-2-2B (primary), GPT-2-small (secondary replication).
- **Storage**: ~20GB for pretrained SAEs and cached activations.
- **No training costs**: All SAEs are pretrained. Only logistic regression probes need training (seconds each).

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Probes for knowledge hierarchies are poor quality | Medium | High | Use well-studied RAVEL hierarchies; report F1 as quality gate; exclude poor probes |
| Absorption on knowledge hierarchies is too low to measure | Low | High | First-letter task already shows 15-35%; knowledge hierarchies have similar structure; if truly 0%, that is itself a major finding |
| Adaptation of sae-spelling code to new domains is non-trivial | Low | Medium | The core `feature_absorption_calculator` is domain-agnostic; only prompting and probe training need adaptation |
| SAEBench pretrained SAEs have limited coverage | Low | Low | 200+ SAEs across 7 architectures is sufficient; Gemma Scope adds hundreds more |
| Results only hold for Gemma-2-2B | Medium | Medium | Include GPT-2-small as replication target; acknowledge model-specificity as limitation |
| RAVEL entity features may not be well-represented in SAE | Medium | Medium | sae-entities paper confirms entity features exist in Gemma Scope; RAVEL is already validated in SAEBench |

### Novelty Claim

The novelty is **not** a new metric or architecture, but the first systematic cross-domain characterization of feature absorption. Specifically:
1. **First measurement of absorption beyond spelling**: All prior work uses the first-letter task. We extend to 4+ additional hierarchies covering knowledge, geography, and entity types.
2. **Cross-domain absorption scaling laws**: We characterize how absorption varies with hierarchy properties (co-occurrence frequency, depth, breadth), providing the first empirical scaling relationships.
3. **Architecture comparison on non-spelling hierarchies**: We test whether Matryoshka SAE's absorption advantage generalizes beyond the single task where it was measured.
4. **Structural predictors of absorption**: We test whether absorption can be predicted from decoder geometry and co-occurrence statistics, toward unsupervised detection.

This is a "strong baseline done right" style paper: the contribution is rigorous empirical characterization of a known phenomenon across new domains, using existing tools, to answer questions the field needs answered before building more complex solutions. The existing literature has identified absorption as critical (NeurIPS 2025, ICML 2025 papers, DeepMind deprioritization) but only measured it in one place. Our work closes this gap.

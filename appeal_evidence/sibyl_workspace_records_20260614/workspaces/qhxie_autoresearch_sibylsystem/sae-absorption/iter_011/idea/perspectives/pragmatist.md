# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (https://github.com/lasr-spelling/sae-spelling) --- MIT, canonical absorption metric implementation. Includes `sae_spelling.probing` (logistic regression probes), `sae_spelling.feature_attribution` (integrated-gradients ablation), `sae_spelling.feature_ablation` (latent ablation), and `sae_spelling.vocab` (token filtering). Directly reusable and well-documented. **Code exists, tested, runnable.**

2. **SAEBench** (https://github.com/adamkarvonen/SAEBench) --- Apache 2.0, 8-metric evaluation suite including absorption task. Extended absorption metric works across all layers (uses probe direction contribution instead of ablation effect). 200+ open-source SAEs across 7 architectures provided. ~30 min per SAE for absorption eval on RTX 3090. **Code exists, actively maintained.**

3. **SAELens v6** (https://github.com/decoderesearch/SAELens) --- MIT, standard library for loading/training SAEs. `SAE.from_pretrained()` loads Gemma Scope SAEs in 2 lines. Supports all major architectures (batchtopk, jumprelu, standard, topk). Deep TransformerLens integration. **Code exists, production-quality.**

4. **Gemma Scope SAEs** (https://huggingface.co/google/gemma-scope) --- 400+ pretrained JumpReLU SAEs on Gemma 2 2B/9B/27B, all layers, widths 1k--1M. Gemma 2 2B + 16k SAE fits on a 16 GB GPU. Gemma Scope 2 adds Matryoshka SAEs on Gemma 3. **Weights freely available, eliminates training cost.**

5. **SAEBench pretrained SAEs** --- 200+ SAEs across 7 architectures (BatchTopK, TopK, JumpReLU, Gated, ReLU, Matryoshka, Standard) on Pythia-160M L8 and Gemma-2-2B L12. Widths 4k/16k/65k, 6 sparsity levels. **Perfect for controlled comparisons; weights on HuggingFace.**

6. **RAVEL benchmark** (https://github.com/explanare/ravel) --- Entity-attribute disentanglement dataset with 5 entity types (cities, Nobel laureates, verbs, physical objects, occupations), each with 4-6 attributes. Cities have country/continent/language attributes = natural feature hierarchy for absorption study. On HuggingFace at `hij/ravel`. **Code and data exist.**

7. **SAE-RAVEL** (https://github.com/MaheepChaudhary/SAE-Ravel) --- Evaluation of open-source SAEs on RAVEL for GPT-2 Small. Shows SAEs struggle to disentangle factual knowledge. **Code exists, adaptable.**

8. **TransformerLens** (https://github.com/TransformerLensOrg/TransformerLens) --- MIT, hook-based activation caching and editing for 50+ models. Required backend for SAELens. **Code exists, standard tool.**

9. **ICLR 2025: SAE Latents for Entity Recognition in Gemma 2** --- Shows SAE latents distinguish known/unknown entities across types (players, cities, movies, songs) with layer-wise progression peaking at layer 9. Demonstrates SAE entity-type probing is feasible on Gemma 2 2B. **Published, validates approach.**

10. **Chanin et al. "Feature Hedging" paper** (https://github.com/chanind/feature-hedging-paper) --- Code for measuring hedging; shows Matryoshka SAEs trade absorption for hedging. Compound multiplier ~0.75 helps. **Code exists.**

11. **Chanin & Garriga-Alonso "Sparse but Wrong"** (https://github.com/chanind/sparse-but-wrong-paper) --- Shows most open-source SAEs have incorrect L0. Proxy metric for finding correct L0. Critical confound for absorption studies. **Code exists.**

12. **Neuronpedia** (https://www.neuronpedia.org) --- Interactive SAE feature explorer with 50M+ latents. SAEBench results visualizable at neuronpedia.org/sae-bench. Useful for qualitative case study validation. **Live platform.**

### Landscape Summary

The absorption research landscape has a clear structure: Chanin et al. (2024) defined the problem and metric using the first-letter spelling task. SAEBench (2025) standardized evaluation across architectures. Multiple mitigation architectures emerged (Matryoshka, OrtSAE, ATM, masked regularization), but none have been compared head-to-head under controlled conditions. The critical gaps are:

**What works in practice:**
- sae-spelling + SAELens + Gemma Scope gives a complete analysis pipeline
- SAEBench provides the standardized evaluation framework
- The first-letter task metric is well-established and reusable
- Pretrained SAEs across 7 architectures eliminate training cost

**What does NOT work / remains unclear:**
- Absorption has only been measured on the first-letter spelling task---nobody has confirmed it generalizes to richer hierarchies
- Absorption vs. hedging vs. incorrect L0: all three cause false negatives, but nobody has disentangled them empirically
- No unsupervised absorption detection exists---you need to know the probe direction in advance
- DeepMind deprioritized SAE research partly due to absorption, but nobody has quantified how much of the SAE-vs-probe gap is caused by absorption specifically

**What is saturated:**
- New SAE architectures that improve the reconstruction-sparsity frontier without studying absorption specifically
- Purely synthetic/toy-model analyses of absorption
- Pure architecture comparisons without mechanistic insight into WHY absorption occurs differently across architectures

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy --- Extending Absorption Measurement Beyond First-Letter Tasks

- **Hypothesis**: Feature absorption rates measured on the first-letter spelling task do NOT generalize to semantically richer feature hierarchies (entity-type > entity, country > city, sentiment > topic). Absorption severity varies systematically with hierarchy depth, feature co-occurrence statistics, and semantic distance between parent and child features.

- **Implementation sketch**: 
  1. Start from sae-spelling's probing + attribution code
  2. Define 3-4 new probe tasks with known hierarchical structure using RAVEL data: (a) city > continent (shallow, 6 classes), (b) city > country (deeper, ~50 classes), (c) city > language (correlated with country), (d) entity-type > entity (very deep)
  3. Train logistic regression probes for parent features on Gemma 2 2B activations at each layer
  4. Adapt sae-spelling's absorption metric: find k-sparse SAE latents for the parent feature, identify false negatives, run attribution to detect absorption
  5. Compare absorption rates across tasks, layers, and SAE architectures (Gemma Scope JumpReLU SAEs + SAEBench SAEs)
  6. Quantify: does hierarchy depth predict absorption? Does feature co-occurrence frequency predict it?

- **Simplest version**: Pick exactly 2 tasks (first-letter and city-continent), 1 model (Gemma 2 2B), 2 layers (L5, L12), 2 SAE widths (16k, 65k). Train probes, measure absorption on both. Compare rates. This alone answers whether first-letter results generalize. ~30 min probe training + ~30 min absorption eval per SAE.

- **Time estimate**: Pilot (2 tasks, 2 SAEs): ~1 hour. Full (4 tasks, 6 SAEs, all layers): ~6-8 GPU-hours.

- **Reusable components**: sae-spelling (absorption metric), RAVEL (city/entity data), SAELens (SAE loading), Gemma Scope SAEs (pretrained weights), SAEBench absorption eval code (extended metric).

### Candidate B: Disentangling Absorption from Hedging and L0 Confounds --- A Controlled Decomposition

- **Hypothesis**: A substantial fraction (>30%) of what is measured as "absorption" in open-source SAEs is actually L0-induced hedging or incorrect sparsity, NOT true hierarchy-driven absorption. True absorption rate is lower than reported, but it persists even at correct L0 and sufficient width.

- **Implementation sketch**:
  1. Use SAEBench's 200+ SAEs spanning 7 architectures, 3 widths, 6 sparsity levels on Gemma-2-2B L12
  2. Apply Chanin & Garriga-Alonso's L0 correctness proxy metric to classify each SAE as "correct L0" vs "incorrect L0"
  3. Apply the hedging metric from feature-hedging-paper to measure hedging severity
  4. Measure absorption rate (SAEBench absorption eval) for all SAEs
  5. Decompose observed false-negative rate = f(absorption) + f(hedging) + f(residual)
  6. Regression: FN_rate ~ absorption_rate + hedging_rate + L0_error + width + architecture

- **Simplest version**: Take the 6 BatchTopK SAEs on Gemma-2-2B L12 (one per sparsity level). Measure absorption and hedging for each. Plot absorption vs L0. If there is a clear trend, the confound is real. ~1 hour total.

- **Time estimate**: Pilot (6 SAEs): ~1 hour. Full (200+ SAEs): ~15-20 GPU-hours but highly parallelizable.

- **Reusable components**: SAEBench SAEs (all pretrained), SAEBench absorption eval, feature-hedging-paper (hedging metric), sparse-but-wrong-paper (L0 proxy).

### Candidate C: Absorption as Reconstruction Strategy --- Quantifying the Fundamental Tradeoff

- **Hypothesis**: Absorption saves exactly 1 L0 per absorbed parent-child feature pair. Therefore, absorption rate scales linearly with the number of parent-child relationships present in the data and inversely with SAE width. This predicts that wider SAEs with lower sparsity should show less absorption, but absorption cannot be eliminated without paying an L0 cost.

- **Implementation sketch**:
  1. Use first-letter task (26 known parent-child pairs per letter x token)
  2. For each absorbed letter-token pair, verify that the absorbing latent's decoder direction encodes BOTH the parent (letter) and child (specific token) information
  3. Compute the L0 savings: compare L0 of absorbed vs non-absorbed tokens
  4. Derive a simple predictive model: absorption_rate ~ f(n_hierarchical_pairs, width, target_L0)
  5. Validate the model across Gemma Scope SAEs (vary width: 1k, 4k, 16k, 65k; vary L0)
  6. Test the prediction on city-continent hierarchy (different hierarchy depth)

- **Simplest version**: Take 4 Gemma Scope SAEs at the same layer but different widths (1k, 4k, 16k, 65k). Count absorption rate and L0 savings for each. Plot absorption_rate vs 1/width. If linear, the model holds. ~45 min.

- **Time estimate**: Pilot: ~45 min. Full (all widths x multiple layers): ~4-5 GPU-hours.

- **Reusable components**: sae-spelling (full pipeline for first-letter), Gemma Scope SAEs (width sweep), SAELens.

---

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption Taxonomy

- **Implementation reality check**: The biggest risk is that defining absorption metrics for non-first-letter tasks is harder than it sounds. The first-letter task has a clean, verifiable hierarchy (token identity implies first letter). For city-continent, we need probes that are accurate enough to serve as ground truth. RAVEL data exists and the ICLR 2025 paper shows SAE entity probing works on Gemma 2 2B, so this is feasible. However, probe accuracy on entity tasks is typically 85-95%, not 99%+ like the first-letter task. This means the absorption metric will have more noise.

- **Reproducibility attack**: The probe training is standard (logistic regression on residual stream activations). The main fragility is in the absorption detection threshold---Chanin et al. use cosine similarity > 0.025 and a gap of >= 1.0 between top two attributing latents. These thresholds may need tuning for different tasks. As long as we report threshold sensitivity, this is manageable.

- **Baseline sanity check**: The strongest simple baseline is: just measure probe recall vs SAE latent recall on each task and report the gap. If SAE latents have 70% recall on first-letter but 40% recall on city-continent, that alone demonstrates that absorption (and/or other failure modes) are worse for richer hierarchies. The full attribution-based absorption metric adds mechanistic specificity but is not strictly necessary for the core finding.

- **Scope attack**: If absorption rates are similar across tasks, the paper becomes "we measured absorption on 4 tasks and it was the same." That is a useful negative result but less exciting. If rates differ, the paper is much stronger. Mitigation: include enough variation in hierarchy properties (depth, co-occurrence frequency, number of children) to guarantee some variation.

- **Verdict**: **STRONG** --- High feasibility (code exists, data exists, pretrained SAEs exist), clear novelty (nobody has done this), good worst-case (even a null result is publishable as confirming generalizability or revealing task-dependence).

### Against Candidate B: Disentangling Absorption from Hedging and L0 Confounds

- **Implementation reality check**: The L0 proxy metric from sparse-but-wrong-paper exists but has only been validated on toy models. Applying it to real SAEs on Gemma-2-2B is untested. The hedging metric from feature-hedging-paper also exists but was developed for a different context. Composing these metrics requires careful validation. Risk: the decomposition FN_rate = f(absorption) + f(hedging) + f(residual) may not be clean because absorption and hedging interact.

- **Reproducibility attack**: The regression model FN_rate ~ absorption + hedging + L0_error has many potential confounds. Different architectures have different training procedures, initialization, etc. Matched-conditions comparison is key but hard to achieve perfectly. SAEBench SAEs are the best available controlled set.

- **Baseline sanity check**: The simplest test is: does absorption rate correlate with L0 error? If yes, the confound story is supported. If no, hedging is not masquerading as absorption. This requires just plotting two numbers per SAE---trivial once the metrics are computed.

- **Scope attack**: If the decomposition doesn't cleanly separate, the paper becomes a collection of correlation plots without a clear conclusion. The theoretical contribution (conceptual framework for disentangling failure modes) is valuable but hard to make crisp without clean separation.

- **Verdict**: **MODERATE** --- Conceptually important but execution risk is higher. The clean decomposition may not exist. Better as a secondary contribution within a larger paper rather than a standalone.

### Against Candidate C: Absorption as Reconstruction Strategy

- **Implementation reality check**: The core prediction (absorption saves 1 L0 per pair) is testable by comparing L0 of absorbed vs non-absorbed inputs. The sae-spelling code already identifies absorbed inputs. The width sweep is straightforward using Gemma Scope SAEs at different widths (all pretrained). Main risk: the simple linear model may be too simple---absorption likely depends on more factors than just width and L0.

- **Reproducibility attack**: The simple model absorption_rate ~ f(n_pairs, width, L0) is easily reproducible. The main concern is that the model might have low R-squared, meaning additional factors dominate. This is actually a finding, not a failure.

- **Baseline sanity check**: Chanin et al. already showed absorption rate varies with width and sparsity. What is new here is (a) measuring the actual L0 savings per absorbed pair, (b) building a predictive model, and (c) testing it across hierarchy types. The novelty is in the quantitative theory, not the observation of width dependence.

- **Scope attack**: If the linear model fits well, great. If not, we learn that absorption is more complex than expected, which is also valuable. The theory paper would be stronger if we also show how to use the model to select hyperparameters that minimize absorption.

- **Verdict**: **MODERATE** --- Clean and simple, but the novelty may be incremental over Chanin et al. if the model is too simple. Best combined with cross-domain validation (Candidate A).

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate B (standalone)**: The decomposition of absorption vs hedging is important but too risky as a standalone paper. The clean separation may not exist. However, the L0 confound analysis should be incorporated as a control experiment in the final proposal.

### Strengthened Ideas
- **Candidate A + C merged**: The cross-domain absorption taxonomy (A) is strongest when combined with the quantitative scaling analysis (C). This gives both empirical breadth (multiple tasks) and mechanistic depth (predictive model). The merged paper answers: (1) Does absorption generalize beyond first-letter? (2) What predicts absorption severity? (3) Can we build a simple model to predict absorption from hierarchy properties and SAE configuration?

### Refined Approach
The key insight: **absorption is fundamentally about feature hierarchy, but it has only been studied on one hierarchy.** The most practical and impactful paper is one that:
1. Extends absorption measurement to 4+ hierarchy types
2. Shows which hierarchy properties predict absorption severity
3. Builds a simple scaling law relating absorption to SAE width/L0/hierarchy depth
4. Controls for L0 confounds
5. Uses ONLY pretrained SAEs (training-free, as required by project spec)

### Pilot Experiment Design (< 15 min)
- Load Gemma 2 2B + one Gemma Scope 16k SAE (L12) via SAELens
- Train a logistic regression probe for "continent of city" on residual stream activations at L12
- Run the sae-spelling absorption detection pipeline on city tokens
- Check: does the code work? Are probe accuracies sufficient? Does absorption exist on this task?

### Selected Front-Runner: **Candidate A+C merger** --- Cross-domain absorption characterization with quantitative scaling analysis.

---

## Phase 5: Final Proposal

### Title
Beyond First Letters: Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders

### Hypothesis
Feature absorption is a general phenomenon tied to feature hierarchy structure, not specific to the first-letter spelling task. Absorption severity is predictable from three factors: (1) the depth and density of the feature hierarchy, (2) SAE width relative to the number of hierarchical feature pairs, and (3) the effective sparsity level (L0). Specifically, absorption rate increases with hierarchy density, decreases with width, and increases with sparsity---but it cannot be fully eliminated without an L0 cost proportional to the number of parent-child pairs.

### Motivation
Feature absorption (Chanin et al., 2024) is one of the most important failure modes of Sparse Autoencoders, directly undermining the reliability of SAE-based circuit analysis and safety applications. DeepMind deprioritized SAE research partly because SAE probes fail where dense probes succeed, and absorption is identified as a key culprit. However, the entire empirical understanding of absorption is based on a single proxy task (first-letter spelling), leaving three critical questions unanswered:
1. Does absorption generalize to semantically richer hierarchies?
2. What hierarchy properties predict absorption severity?
3. Can we build a practical model to predict (and minimize) absorption?

Without answers, the community cannot assess whether absorption is a minor nuisance (specific to syntactic tasks) or a fundamental obstacle (affecting all hierarchical knowledge). This paper provides the first systematic cross-domain absorption characterization.

### Method

**Step 1: Define Hierarchical Probe Tasks (no GPU needed, ~2 hours)**
- Task 1: First-letter spelling (baseline, from sae-spelling)
- Task 2: City -> Continent (RAVEL data, 6 classes, ~3000 cities)
- Task 3: City -> Country (RAVEL data, ~50 classes, same cities)
- Task 4: City -> Language (RAVEL data, correlated with country)
- Task 5: Entity-type classification (RAVEL data: cities vs. occupations vs. objects)
- Each task defines a clear parent-child hierarchy with known ground truth

**Step 2: Train Ground-Truth Probes (1 GPU, ~30 min per task)**
- For each task and layer, train logistic regression probes on Gemma 2 2B residual stream activations using TransformerLens
- Validate probe accuracy (target: >85% for meaningful absorption measurement)
- Reuse sae-spelling's `train_multi_probe()` and `train_binary_probe()` functions

**Step 3: Measure Absorption Across Tasks and SAE Configurations (1 GPU, ~1 hour per SAE)**
- Load pretrained SAEs via SAELens: Gemma Scope (widths 16k, 65k at L5, L12, L20) + SAEBench (7 architectures, 3 widths at L12)
- For each (task, SAE, layer): run k-sparse probing to find relevant SAE latents, identify false negatives, detect absorption via probe direction contribution (SAEBench's extended metric)
- Record: absorption rate, false-negative rate, number of absorbed parent-child pairs, L0 of absorbed vs non-absorbed inputs

**Step 4: Quantitative Scaling Analysis (no GPU needed, analysis only)**
- For each hierarchy, compute: hierarchy depth, number of parent-child pairs, parent-child co-occurrence frequency in training data
- Fit regression model: absorption_rate ~ hierarchy_density + log(width) + L0 + architecture
- Test predictive accuracy: does the model trained on first-letter data predict city-continent absorption?

**Step 5: L0 Confound Control (1 GPU, ~1 hour)**
- Apply L0 correctness proxy (from sparse-but-wrong-paper) to each SAE
- Split SAEs into "correct L0" and "incorrect L0" groups
- Report absorption rates for both groups separately
- Show that true hierarchy-driven absorption persists even at correct L0

**Libraries**: sae-spelling (MIT), SAELens v6 (MIT), SAEBench (Apache 2.0), TransformerLens (MIT), RAVEL (available on HuggingFace)

### Simplest Version
The absolute minimum experiment that tests the core claim:
1. Load Gemma 2 2B + 1 Gemma Scope 16k SAE at L12
2. Run the existing sae-spelling first-letter absorption pipeline (baseline)
3. Train a continent-of-city probe at L12 on RAVEL data
4. Run the same absorption pipeline on the city-continent task
5. Compare absorption rates between the two tasks

If rates differ significantly, the paper's core claim is supported. If they are similar, that is also a publishable finding (absorption generalizes uniformly). Total time: ~2 hours on a single GPU.

### Baselines

1. **First-letter absorption (Chanin et al., 2024)**: Expected absorption rate 15-35% on Gemma Scope 16k/65k SAEs. This is our methodological baseline---we replicate it and extend to new tasks.

2. **Dense linear probe recall**: For each task, the logistic regression probe defines the ceiling. SAE latent recall (accounting for absorption) should be lower. The gap = absorption + hedging + other failure modes.

3. **Random feature baseline**: Following Korznikov et al. (2026), compare SAE absorption detection against random decoder directions to establish that detected absorption is non-trivial.

### Experimental Plan

**Phase 0 --- Metric Validation (Day 1, ~4 GPU-hours)**
- Replicate Chanin et al. first-letter absorption on Gemma 2 2B with Gemma Scope 16k SAE at L12
- Validate probe training pipeline works for RAVEL entity tasks
- Confirm probe accuracy >85% on at least 3 of 4 RAVEL tasks

**Phase 1 --- Cross-Domain Absorption (Days 2-3, ~8 GPU-hours)**
- 5 tasks x {2 layers (L5, L12)} x {3 SAE widths (4k, 16k, 65k)} = 30 measurements
- Use Gemma Scope JumpReLU SAEs
- Record absorption rate, false-negative count, L0 savings per absorbed pair

**Phase 2 --- Architecture Comparison (Days 3-4, ~10 GPU-hours)**
- 5 tasks x 7 architectures x 1 width (16k) at L12 = 35 measurements
- Use SAEBench pretrained SAEs (all 7 architectures on Gemma-2-2B L12)
- Focus: does Matryoshka's low absorption on first-letter generalize to entity tasks?

**Phase 3 --- Scaling Analysis (Day 5, analysis only)**
- Fit predictive model across all data points
- Cross-validate: train on first-letter, predict entity-task absorption
- Ablation: which factors (width, L0, hierarchy depth, architecture) contribute most?

**Phase 4 --- L0 Confound Analysis (Day 5-6, ~2 GPU-hours)**
- Classify SAEs by L0 correctness
- Report absorption conditional on L0 correctness
- Show absorption persists even at correct L0

**Datasets**: Model vocabulary (first-letter, from sae-spelling), RAVEL (cities/entities, from HuggingFace `hij/ravel`), OpenWebText subset (for co-occurrence statistics)

**Metrics**: Absorption rate (Chanin et al. / SAEBench extended metric), false-negative rate, L0 savings per absorbed pair, probe accuracy (ground truth quality), regression R-squared (predictive model quality)

**Ablation schedule**: 
- Width ablation: 4k vs 16k vs 65k (fixed architecture, layer, task)
- Architecture ablation: 7 architectures (fixed width, layer, task)
- Task ablation: 5 tasks (fixed architecture, width, layer)
- Layer ablation: L5 vs L12 vs L20 (fixed architecture, width, task)

### Resource Estimate
- **Model**: Gemma 2 2B (~5 GB VRAM in fp16)
- **SAEs**: 16k-width SAE adds ~150 MB; 65k adds ~600 MB
- **Total VRAM per experiment**: ~8-12 GB (fits on single 16 GB GPU)
- **Probe training**: ~30 min per task per layer
- **Absorption evaluation**: ~30 min per SAE per task
- **Total GPU-hours**: ~24-30 for full paper
- **Wall-clock time with 1 GPU**: ~5-6 days
- **With 2 GPUs (parallelized)**: ~3 days
- **Pilot (minimal version)**: ~2 hours on 1 GPU

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RAVEL probes have insufficient accuracy (<80%) on Gemma 2 2B | Low | High | ICLR 2025 paper shows SAE entity probing works on Gemma 2 2B. If accuracy is low on some tasks, drop those tasks and report the limitation. |
| Absorption metric thresholds need per-task tuning | Medium | Medium | Use SAEBench's extended metric (probe direction contribution) which is more robust. Report threshold sensitivity analysis. |
| All tasks show identical absorption rates | Low-Medium | Medium | This is a valid negative result (absorption generalizes uniformly). Still publishable. Combine with the scaling analysis for mechanistic insight. |
| Gemma Scope SAEs at L5/L20 have poor reconstruction for entity features | Medium | Low | If entity features are not well-represented at certain layers, restrict analysis to layers where probes work. The layer-wise analysis is secondary. |
| L0 confound analysis is inconclusive | Medium | Low | This is a supplementary analysis. The main contribution (cross-domain absorption) stands independently. |
| Predictive model has low R-squared | Medium | Medium | Even a weak predictive model identifies which factors matter. If the simple model fails, this motivates more complex theory as future work. |

### Novelty Claim

The novelty is threefold:
1. **First cross-domain absorption characterization**: Nobody has measured absorption on anything other than the first-letter spelling task. We are the first to measure absorption on entity-type hierarchies (city-continent, city-country, city-language) and compare rates across hierarchy types.
2. **Quantitative scaling analysis**: We provide the first predictive model relating absorption rate to hierarchy properties and SAE configuration, enabling principled SAE selection for specific downstream tasks.
3. **L0 confound control**: We are the first to explicitly control for L0-induced hedging when measuring absorption, separating true hierarchy-driven absorption from sparsity artifacts.

The novelty is not flashy---it is "the obvious thing that nobody has done yet." This is exactly the kind of paper that is both highly citable (everyone studying absorption will need to reference whether first-letter results generalize) and practically useful (the predictive model guides SAE training choices).

**What is explicitly NOT claimed as novel**: The absorption metric itself (from Chanin et al. / SAEBench), the pretrained SAEs (from Gemma Scope / SAEBench), the RAVEL dataset, or the concept of feature hierarchy in SAEs. Our contribution is the systematic cross-domain empirical analysis and the quantitative framework.

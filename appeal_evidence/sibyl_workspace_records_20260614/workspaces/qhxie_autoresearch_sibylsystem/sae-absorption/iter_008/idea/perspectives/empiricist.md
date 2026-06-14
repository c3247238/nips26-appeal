# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507)** --- Defines the canonical absorption metric: k-sparse probing + false-negative identification + integrated-gradients ablation + cosine similarity threshold (>0.025) + magnitude gap (>=1.0). Key methodological limitation: thresholds chosen via manual inspection, not statistically principled; only valid for layers 0--17 in Gemma 2 2B due to information movement to final token position.

2. **Karvonen et al. (2025), SAEBench (arXiv:2503.09532, ICML 2025)** --- Modified absorption metric that replaces ablation-effect detection with probe-direction contribution, enabling evaluation across all model layers. Introduces 8-metric evaluation suite with 200+ SAEs across 7 architectures. Computational cost: ~65 min per SAE on RTX 3090 (16K width). Critical finding: proxy metrics (CE loss, sparsity) do NOT reliably predict practical performance.

3. **Chanin & Garriga-Alonso (2025), "Sparse but Wrong" (arXiv:2508.16560)** --- Demonstrates that incorrect L0 causes SAEs to learn systematically wrong features. Provides proxy metric for finding correct L0. Essential confound control: any absorption study must verify L0 correctness before attributing failures to absorption per se.

4. **Chanin, Dulka & Garriga-Alonso (2025), "Feature Hedging" (arXiv:2505.11756)** --- Shows feature hedging (merging correlated features in narrow SAEs) manifests identically to absorption in practice (features failing to fire). Critical methodological implication: hedging and absorption must be disentangled in any valid absorption measurement.

5. **Tian et al. (2025), "Measuring SAE Feature Sensitivity" (arXiv:2509.23717)** --- Frames absorption as a special case of poor feature sensitivity. Provides scalable sensitivity evaluation methods. Key insight: many features appear monosemantic via activation examples but have poor sensitivity---absorption is the tip of a larger recall problem.

6. **Huang et al. (2024), RAVEL (arXiv:2402.17700)** --- Evaluates disentanglement of entity attributes (city -> country, continent, language). Three-stage process: filtering, probe training, causal intervention via binary mask. Provides the ONLY existing benchmark with known hierarchical entity-attribute structure suitable for studying absorption beyond first-letter tasks.

7. **SynthSAEBench (arXiv:2602.14687, 2026)** --- Synthetic benchmark with known ground-truth feature hierarchies, correlations, and Zipfian firing distributions. Logistic probes achieve 0.974 F1 while best SAEs substantially underperform. Enables controlled ablation of specific hierarchy properties.

8. **Leask et al. (2025), "SAEs Do Not Find Canonical Units" (arXiv:2502.04878, ICLR 2025)** --- Meta-SAE methodology for decomposing features into sub-features. Methodologically relevant: provides tools for detecting whether "atomic" features actually contain absorbed parent information.

9. **Kantamneni et al. (2025), "SAE probes underperform baselines" (arXiv:2502.16681)** --- Demonstrates SAE probes underperform simple linear probes on both in-domain and OOD detection. Critical baseline: any absorption study claiming SAE features miss concepts must compare against linear probe baseline to distinguish SAE-specific failures from general representation limitations.

10. **Korznikov et al. (2026), "Sanity Checks for SAEs" (arXiv:2602.14111)** --- Shows SAEs recover only 9% of true features in synthetic settings; random baselines match fully-trained SAEs. Establishes the harshest null hypothesis: absorption detection must be validated against random feature baselines.

11. **Bussmann et al. (2025), Matryoshka SAE (arXiv:2503.17547, ICML 2025)** --- Absorption rate ~0.03 vs BatchTopK ~0.29. Nested prefix losses create natural feature hierarchy. Key experimental design pattern: inner Matryoshka levels trade absorption for hedging, creating natural control group.

12. **Li et al. (2025), ATM SAE (arXiv:2510.08855)** --- Best reported absorption scores (0.0068 vs TopK 0.1402). Per-latent importance scoring. Evaluated only on Gemma-2-2B layer 12---replication on other models/layers needed before accepting claims.

### Experimental Landscape

**What has been properly tested:**
- Absorption exists in every tested SAE architecture (L1, TopK, JumpReLU, BatchTopK, Matryoshka, OrtSAE) on the first-letter spelling task
- Wider and sparser SAEs show higher absorption rates (15--35% on Gemma Scope 16k/65k)
- Architecture comparison on SAEBench's 8 metrics (but not with matched compute)
- JumpReLU has worst absorption; Matryoshka has best among standard architectures

**What is accepted without rigorous evidence:**
- That first-letter absorption rates generalize to semantically richer feature hierarchies
- That absorption severity scales predictably with SAE configuration parameters
- That SAEBench's modified absorption metric (probe-direction contribution) is equivalent to the original ablation-based metric
- That ATM SAE's reported absorption of 0.0068 is reproducible beyond a single model/layer
- That absorption and hedging are cleanly separable phenomena in real LLM SAEs

**Critical methodological gaps:**
- No study uses bootstrap confidence intervals or formal statistical tests for absorption rate differences
- Most absorption comparisons across architectures use different models, layers, training data, and evaluation protocols
- The 0.025 cosine similarity threshold and 1.0 magnitude gap in the original absorption metric are not statistically justified
- No randomized control: what absorption rate would we expect from random feature directions?
- Sample size for per-letter absorption (200 false negatives) is not power-analyzed

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Anatomy --- Measuring Absorption Beyond First-Letter Spelling

- **Hypothesis**: Feature absorption severity is determined by the ratio of parent-child feature co-occurrence frequency to overall parent feature frequency, and this relationship holds across semantically distinct hierarchical domains (syntactic: first-letter; knowledge: entity-type/attribute; compositional: sentiment/topic).
- **Falsification criterion**: If absorption rates on entity-type hierarchies (e.g., "city" parent feature absorbed by specific city child features) differ from first-letter absorption rates by more than a factor of 3 after controlling for co-occurrence statistics, the co-occurrence hypothesis is falsified.
- **Evaluation protocol**:
  - Primary benchmark: First-letter spelling task (Chanin et al. baseline, replication), RAVEL entity attributes (city -> country/continent/language), custom entity-type probes (animal -> species, country -> continent)
  - Metrics: Absorption rate per Chanin et al. metric, SAEBench modified metric, plus new per-domain absorption rate
  - Statistical tests: Bootstrap 95% CI for absorption rate per domain; paired permutation test comparing absorption rates across domains on matched SAEs
  - Minimum 3 SAE architectures (BatchTopK, JumpReLU, Matryoshka) x 2 widths (16k, 65k) x 2 models (Gemma 2 2B, Pythia-160M)
- **Ablation plan**:
  - A1: Hold SAE fixed, vary domain --- isolates domain effect on absorption
  - A2: Hold domain fixed, vary SAE width --- isolates capacity effect
  - A3: Hold domain fixed, vary L0 --- controls for L0 confound per Chanin & Garriga-Alonso
  - A4: Measure co-occurrence statistics per domain, regress absorption rate on co-occurrence ratio
- **Confounders**: (i) Probe quality differs across domains---must verify probe accuracy before measuring absorption; (ii) L0 confound: apparent absorption may be hedging if L0 is too low; (iii) Layer selection: absorption metric validity varies by layer
- **Pilot design**: Run absorption measurement on Gemma 2 2B layer 12 (single SAE, single width) for first-letter vs. RAVEL city-country hierarchy. Estimate: 10--15 min on single GPU (probe training + absorption calculation for 2 domains).

### Candidate B: Disentangling Absorption from Hedging via Controlled L0 Sweep

- **Hypothesis**: A significant fraction (>30%) of what is currently measured as "absorption" in standard SAEs is actually L0-induced feature hedging, detectable by measuring absorption rate as a function of L0 at fixed SAE width.
- **Falsification criterion**: If absorption rate is invariant to L0 changes (within +/- 5% relative) across a sweep from L0=10 to L0=200 on the same SAE architecture and width, then hedging does not confound absorption measurements.
- **Evaluation protocol**:
  - Benchmark: SAEBench absorption metric on first-letter task
  - L0 sweep: 8 L0 values (10, 20, 40, 60, 80, 100, 150, 200) on BatchTopK SAEs (width 16k) on Gemma 2 2B layer 12
  - Use Chanin & Garriga-Alonso's L0 proxy metric to identify the "correct" L0 for each SAE
  - Metrics: Absorption rate, feature hedging metric (from Chanin et al. 2025), sparse probing F1, reconstruction MSE
  - Statistical test: Spearman rank correlation between L0 and absorption rate, with bootstrap CI
- **Ablation plan**:
  - A1: Full L0 sweep on BatchTopK --- primary result
  - A2: Repeat on JumpReLU SAEs at matched L0 values --- architecture generality
  - A3: Repeat at width 65k --- width interaction
  - A4: At each L0, decompose false-negative population into "hedging-type" (correlated features mixed) vs. "absorption-type" (specific child absorbs parent direction) using decoder cosine similarity structure
- **Confounders**: (i) L0 changes affect reconstruction quality, which may independently affect absorption measurement; (ii) BatchTopK's batch-level sparsity may behave differently from per-sample L0; (iii) Pre-trained SAEs may not be available at all desired L0 values---may need to use SAEBench's open-sourced training checkpoints
- **Pilot design**: Run absorption on 3 SAEBench SAEs at L0=20, 60, 100 (BatchTopK, Gemma 2 2B, width 16k). Estimate: 15 min (3 x 5 min per absorption evaluation).

### Candidate C: Quantifying Absorption's Causal Impact on Downstream Interpretability Tasks

- **Hypothesis**: The gap between SAE probe performance and dense linear probe performance on knowledge-intensive tasks (entity recognition, sentiment detection, topic classification) is primarily driven by feature absorption of parent-level features, and this gap is proportional to the SAE's absorption rate.
- **Falsification criterion**: If the correlation between per-SAE absorption rate and the SAE-vs-probe performance gap is less than r=0.3 (Pearson) across 20+ SAEs, then absorption is not the primary driver of SAE underperformance.
- **Evaluation protocol**:
  - Primary benchmark: Sparse probing on SAEBench datasets (sentiment, topic, entity type) + absorption rate on SAEBench absorption task
  - Baseline: Dense linear probe on same tasks
  - SAE pool: SAEBench's 200+ open-sourced SAEs (7 architectures, 3 widths, 6 sparsities)
  - Metrics: (i) Performance gap = dense probe F1 - SAE probe F1; (ii) Absorption rate from SAEBench; (iii) Feature hedging metric; (iv) L0; (v) Width; (vi) Architecture
  - Statistical test: Multiple regression of performance gap on absorption rate, controlling for L0, width, and hedging. Bootstrap CI for regression coefficients.
  - Seeds: Use pre-existing SAEs (no training needed), but verify with 3 probe training seeds
- **Ablation plan**:
  - A1: Simple correlation absorption rate vs. probe gap --- establishes basic relationship
  - A2: Add hedging metric as covariate --- tests whether absorption adds explanatory power beyond hedging
  - A3: Stratify by architecture --- tests whether the relationship is architecture-specific
  - A4: Stratify by task type (syntactic vs. semantic vs. knowledge) --- tests whether absorption matters more for some task types
  - A5: Within-architecture comparison of Matryoshka (low absorption) vs. BatchTopK (high absorption) on same tasks --- most controlled causal test
- **Confounders**: (i) Performance gap may be driven by reconstruction error, not absorption specifically; (ii) SAEBench absorption metric is only on first-letter task---may not predict absorption on semantic tasks; (iii) Different SAE architectures have different L0 ranges, creating confounding; (iv) Dense probe as baseline may itself be suboptimal for some tasks
- **Pilot design**: Extract absorption scores and sparse probing scores for 10 SAEBench SAEs (2 architectures x 5 sparsities). Compute correlation. Estimate: 5--10 min (data extraction from existing SAEBench results).

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Domain Absorption Anatomy)

- **Confound attack**: The biggest confound is that probe quality varies dramatically across domains. A linear probe for "starts with letter S" achieves >95% accuracy on Gemma 2 2B, but a probe for "is a city" may have different accuracy characteristics. If the "city" probe has lower accuracy, measured absorption rates will be artificially inflated (more false negatives attributed to absorption when the probe itself is wrong). MITIGATION: Report probe accuracy and exclude domains where probe accuracy < 90%. Search for papers on probe quality for entity attributes: Ferran et al. (ICLR 2025) found SAE latents that fire consistently across entity types on Gemma 2 2B/9B, suggesting probes for entity recognition are viable.

- **Statistical attack**: The co-occurrence regression requires sufficient variance in co-occurrence statistics across domains. If all hierarchies tested have similar co-occurrence patterns, the regression will lack power. With only 3--4 hierarchical domains, degrees of freedom are very low. MITIGATION: Use per-letter/per-entity granularity within domains (26 letters, ~100 cities, ~50 countries) to increase sample size for regression. Use hierarchical/mixed-effects model.

- **Benchmark attack**: RAVEL is the only suitable benchmark for entity-attribute hierarchies, but it was designed for disentanglement, not absorption. Its entity set is relatively small (400--800 instances per type), which limits statistical power for absorption measurement. Custom probes for novel domains (animal/species) require ground-truth labels that must be constructed and validated. MITIGATION: Use RAVEL as primary + construct 2 additional custom evaluation sets with at least 500 instances each.

- **Ablation completeness attack**: A4 (regressing absorption on co-occurrence ratio) assumes a linear relationship. Feature hierarchy depth may also matter: first-letter is a 2-level hierarchy (letter -> token), while knowledge hierarchies can be deeper (continent -> country -> city -> district). The regression should include hierarchy depth as a variable. Also, A2 and A3 do not control for each other---a factorial design (width x L0 x domain) would be more informative.

- **Verdict**: STRONG. Cross-domain absorption measurement addresses the most critical gap in the field (Gap 2 and Gap 6 from the literature survey). The main risk is probe quality variation, but this is measurable and controllable.

### Against Candidate B (Disentangling Absorption from Hedging)

- **Confound attack**: The core confound is that changing L0 changes everything about the SAE's feature set, not just the hedging/absorption balance. At L0=10, features are fundamentally different from L0=200. This means comparing absorption rates across L0 values is not a clean causal test---it's observational. A study that somehow manipulated hedging *independently* of L0 would be far more convincing. The closest thing would be comparing Matryoshka SAE inner levels (which are narrow = high hedging) with outer levels (wider = less hedging) at matched sparsity. MITIGATION: Include within-Matryoshka comparison as a "natural experiment."

- **Statistical attack**: 8 L0 values on a single architecture/width/model gives only 8 data points for the correlation. Even with bootstrap CI, this is low power. Adding a second architecture and width (A2, A3) helps but introduces cross-condition variability. MITIGATION: Use finer L0 grid (16 values) and report regression on the combined dataset with architecture/width as covariates.

- **Benchmark attack**: The benchmark is the existing first-letter absorption task---same benchmark used by everyone. This is appropriate for controlled comparison but does not extend results beyond this narrow domain. MITIGATION: If Candidate A advances, its cross-domain metrics can be applied here.

- **Ablation completeness attack**: A4 proposes decomposing false negatives into "hedging-type" vs. "absorption-type" using decoder cosine similarity. This decomposition has not been validated---it's a novel methodological contribution that itself requires validation. If the decomposition is unreliable, the entire study's conclusions are questionable. MITIGATION: Validate decomposition on SynthSAEBench where ground-truth hedging vs. absorption can be computed.

- **Verdict**: MODERATE. The research question is important (Gap 9), but cleanly disentangling hedging from absorption via L0 sweep is inherently observational rather than causal. The study would be stronger as a component of a larger paper (e.g., within Candidate A or C) rather than standalone.

### Against Candidate C (Absorption's Causal Impact on Downstream Tasks)

- **Confound attack**: The most serious confound is that absorption rate is measured on the first-letter task, while performance gap is measured on semantic tasks (sentiment, topic, entity type). Absorption rate on first-letter may not predict absorption rate on semantic hierarchies. This is the exact same problem Candidate A addresses. Without Candidate A's cross-domain absorption measurements, Candidate C's correlation could be spurious (or absent) due to domain mismatch. MITIGATION: Use Candidate A's results as input; alternatively, acknowledge this limitation and interpret results as measuring "absorption on spelling task as a proxy for general feature quality."

- **Statistical attack**: With 200+ SAEs, statistical power is excellent. However, the multiple regression has many covariates (absorption, hedging, L0, width, architecture) with only moderate independence. Multicollinearity between L0 and absorption (since higher L0 correlates with lower absorption in some architectures) could inflate confidence intervals. MITIGATION: Report VIF (variance inflation factor); use partial correlation / semi-partial correlation to isolate absorption's unique contribution.

- **Benchmark attack**: SAEBench's sparse probing tasks are well-established. The concern is that "sparse probing F1" measures whether SAE features can be used for classification, not whether the SAE has correctly recovered the model's internal features. An SAE could achieve good sparse probing by learning correlated but non-causal features. MITIGATION: Include TPP and SCR metrics alongside probing F1 to capture causal disentanglement.

- **Ablation completeness attack**: A5 (Matryoshka vs. BatchTopK comparison) is the strongest test but confounds architecture with absorption level. A cleaner test would compare the SAME architecture at different training stages (early training = less absorption vs. late training = more absorption, per JumpReLU pattern from SAEBench). MITIGATION: Add A6: within-JumpReLU comparison at different training checkpoints (SAEBench provides these).

- **Verdict**: STRONG. The dataset is large (200+ SAEs), the question is directly actionable (does absorption matter for real tasks?), and the statistical framework is standard. The main limitation is the domain mismatch between absorption measurement and downstream tasks.

---

## Phase 4: Refinement

### Dropped Candidates

None dropped entirely. Candidate B is too narrow as standalone but contributes essential methodology.

### Strengthened Design: Unified Study

The strongest paper combines all three candidates into a unified three-part study:

**Part 1 (from Candidate A): Cross-Domain Absorption Characterization**
- Measure absorption on 4 hierarchical domains: first-letter (baseline replication), RAVEL city-country, RAVEL city-continent, custom entity-type (animal -> species)
- Use matched SAEs across all domains (Gemma 2 2B, 3 architectures x 2 widths)
- Report absorption rate with bootstrap 95% CI per domain
- Regress absorption rate on co-occurrence statistics and hierarchy depth
- Control for probe accuracy by excluding domains/letters with probe accuracy < 90%

**Part 2 (from Candidate B, integrated): Absorption vs. Hedging Decomposition**
- Within the cross-domain study, sweep L0 on BatchTopK SAEs (8 values)
- Measure both absorption rate and hedging metric at each L0
- Use Chanin's L0 proxy metric to identify "correct" L0
- Validate decomposition on SynthSAEBench synthetic data (ground-truth available)
- Report the fraction of false negatives attributable to hedging vs. absorption at each L0

**Part 3 (from Candidate C): Downstream Impact Quantification**
- Correlate per-SAE absorption rate (extended to cross-domain) with performance gap on sparse probing, SCR, and TPP
- Use SAEBench's 200+ open-sourced SAEs
- Multiple regression controlling for hedging, L0, width, architecture
- Within-architecture paired comparison (Matryoshka vs. BatchTopK)

### Additional Controls Added

1. **Random baseline**: Measure "absorption rate" using random feature directions as the probe, to establish a null distribution for the absorption metric
2. **Probe quality gate**: For each domain, report probe accuracy and restrict analysis to domains/features where probe accuracy exceeds 90%
3. **Cross-validation**: For all probe-based measurements, use 5-fold cross-validation with separate train/test splits to prevent leakage
4. **Multiple comparison correction**: Apply Bonferroni correction when comparing absorption rates across domains

### Selected Front-Runner

**"Beyond First Letters: A Cross-Domain Anatomy of Feature Absorption in Sparse Autoencoders"**

This unified study addresses Gaps 2, 6, 9, and 8b from the literature survey in a single paper, with each part strengthening the others: Part 1 establishes whether absorption generalizes, Part 2 controls for the hedging confound, and Part 3 quantifies the practical stakes.

---

## Phase 5: Final Proposal

### Title

**Beyond First Letters: A Cross-Domain Anatomy of Feature Absorption in Sparse Autoencoders**

### Hypothesis

Feature absorption is a general pathology of sparsity-optimized SAEs that occurs across all hierarchical feature domains---not just first-letter spelling---and its severity is predictable from the co-occurrence statistics of parent-child features in the training data. Furthermore, a significant fraction (>30%) of observed absorption in standard SAEs is confounded with L0-induced feature hedging, and after controlling for this confound, absorption rate on knowledge-intensive hierarchies predicts the performance gap between SAE probes and dense linear probes on downstream interpretability tasks.

### Falsification Criteria

1. **Domain generality**: If absorption rate on entity-type hierarchies (city/country) is < 5% while first-letter absorption is > 15% on the same SAEs, then absorption is domain-specific to spelling and the core hypothesis fails.
2. **Co-occurrence prediction**: If regression of absorption rate on co-occurrence frequency ratio has R^2 < 0.1 across all domains and SAE configs, then co-occurrence does not predict absorption and the causal theory fails.
3. **Hedging confound**: If absorption rate varies by < 5% relative across the L0 sweep (L0=10 to 200) on BatchTopK SAEs, then hedging does not confound absorption and Part 2's hypothesis fails.
4. **Downstream impact**: If the Pearson correlation between per-SAE absorption rate and downstream performance gap is < 0.3 across 200+ SAEs, then absorption is not a primary driver of SAE underperformance.

### Method

A training-free analysis study using pre-trained SAEs from Gemma Scope and SAEBench. No SAE training required---all analysis uses existing open-source SAEs and models.

**Models**: Gemma 2 2B (primary), Pythia-160M (secondary, for SAEBench SAEs)
**SAEs**: Gemma Scope JumpReLU SAEs (16k, 65k widths, all layers); SAEBench SAEs (BatchTopK, JumpReLU, Matryoshka, OrtSAE, ReLU, Gated, TopK at 4k/16k/65k widths); total ~200+ SAEs
**Layers**: Layer 12 (primary analysis layer for Gemma 2 2B, matching SAEBench); layers 5, 8, 17, 22 (layer generality check)
**Libraries**: SAELens v6, TransformerLens, SAEBench evaluation code, sae-spelling absorption metric code

### Evaluation Protocol

**Part 1: Cross-Domain Absorption**
- Domains: (D1) First-letter spelling (26 classes, baseline replication), (D2) RAVEL city-country (5 entity types x 4--6 attributes), (D3) Custom entity-type hierarchy (animal species: 50+ species, 8 parent categories), (D4) Sentiment-topic hierarchy (positive/negative sentiment absorbed by topic-specific features)
- For each domain: train logistic regression probe on model residual stream, compute k-sparse probing on SAE latents, apply modified SAEBench absorption metric
- Per-domain absorption rate reported with bootstrap 95% CI (10,000 resamples)
- Factorial design: 3 architectures (BatchTopK, JumpReLU, Matryoshka) x 2 widths (16k, 65k) x 4 domains x 5 layers (primary + 4 check layers) = 120 conditions

**Part 2: Hedging-Absorption Decomposition**
- L0 sweep: 8 values (10, 20, 40, 60, 80, 100, 150, 200) on BatchTopK SAEs (width 16k) on Gemma 2 2B layer 12
- At each L0: measure absorption rate, hedging metric, Chanin's L0 proxy metric
- Validate decomposition methodology on SynthSAEBench with known ground truth
- Spearman correlation between L0 and absorption rate, with bootstrap CI

**Part 3: Downstream Impact**
- Compute performance gap (dense probe F1 - SAE probe F1) on sparse probing tasks from SAEBench
- Multiple regression: gap ~ absorption_rate + hedging_metric + L0 + width + architecture
- Report partial R^2 for absorption_rate after controlling for other variables
- Within-architecture paired comparison: Matryoshka (low absorption) vs. BatchTopK (high absorption) on matched L0

**Statistical framework**: Bootstrap 95% CI throughout; Bonferroni correction for multi-domain comparisons; report effect sizes (Cohen's d) alongside p-values; VIF check for multicollinearity in regression.

### Ablation Schedule

| ID | Component | Tests | Expected Outcome |
|----|-----------|-------|------------------|
| A1 | Domain variation (Part 1) | Same SAE, different hierarchical domains | Absorption rates differ across domains but remain >5% everywhere; first-letter may be highest due to strong co-occurrence |
| A2 | Width variation (Part 1) | Same domain, different SAE widths | Wider SAEs show higher absorption (consistent with Chanin et al.) |
| A3 | L0 sweep (Part 2) | Same SAE, different L0 | Absorption rate decreases with increasing L0; hedging metric increases with decreasing L0 |
| A4 | Architecture comparison (Part 1) | Different architectures, same domain/width/L0 | Matryoshka << BatchTopK on absorption; JumpReLU highest (replicating SAEBench) |
| A5 | Hedging control (Part 3) | Regression with vs. without hedging covariate | Adding hedging reduces absorption coefficient but absorption remains significant |
| A6 | Random baseline | Random directions as "probe" | Absorption rate near 0% (validates that metric detects real absorption, not noise) |
| A7 | Probe quality | Vary probe training set size | Absorption rate stable when probe accuracy > 90% |

### Control Experiments

1. **Random direction control**: Replace trained probe directions with random unit vectors; measure "absorption rate" under the same metric. Expected: near-zero, validating that the metric is sensitive to real absorption rather than statistical noise in feature-probe alignment.

2. **Shuffled hierarchy control**: For each domain, shuffle parent-child labels (e.g., assign random first letters to tokens) and re-measure absorption. Expected: near-zero absorption, confirming that measured absorption reflects real hierarchical structure rather than metric artifacts.

3. **Probe-only baseline**: Measure how often the dense linear probe itself disagrees with ground truth on the false-negative population. If the probe is wrong on many of these tokens, "absorption" may actually be "probe error." Expected: probe error rate < 5% for high-quality probes.

4. **Cross-run consistency**: If SAEBench provides multiple SAEs trained with different random seeds at the same configuration, measure absorption rate consistency across seeds. Expected: CV < 20%, confirming that absorption is a stable property of the SAE configuration, not training noise.

### Pilot Design

**Phase 1 pilot** (~10 min): Load single Gemma 2 2B SAE (Gemma Scope JumpReLU 16k, layer 12) via SAELens. Run absorption metric on first-letter task (replication). Verify we can reproduce published absorption rates (15--35%).

**Phase 2 pilot** (~15 min): Using same SAE, construct a simple entity-type probe (cities from RAVEL dataset) and measure absorption on city-country hierarchy. Compare to first-letter absorption on same SAE.

**Pilot success criteria**: (i) First-letter absorption rate within 10% of Chanin et al.'s published values; (ii) Entity-type probe achieves accuracy > 85%; (iii) Entity-type absorption measurement completes without errors.

### Resource Estimate

- **GPU-hours**: ~20--30 hours total (all analysis, no training)
  - Part 1: ~8 hours (120 conditions x ~4 min each for probe training + absorption measurement)
  - Part 2: ~4 hours (8 L0 values x 2 metrics x ~15 min each including SynthSAEBench validation)
  - Part 3: ~3 hours (data extraction from SAEBench + regression analysis + paired comparisons)
  - Pilots: ~1 hour
  - Buffer for debugging and re-runs: ~10 hours
- **Models**: Gemma 2 2B (~5GB VRAM), Pythia-160M (~1GB VRAM) --- fits on single A100
- **SAEs**: Pre-trained weights downloaded from HuggingFace (Gemma Scope) and SAEBench repos
- **Time budget**: Each individual experiment task well under 1 hour; full study ~2--3 days wall clock on single GPU

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Entity-type probes have low accuracy on Gemma 2 2B | High | Use RAVEL's pre-validated entity sets; fall back to Gemma 2 9B or GPT-2 if 2B probes fail; set 90% accuracy gate |
| SAEBench absorption metric code does not extend to custom domains | Medium | Adapt sae-spelling code directly; modular metric implementation |
| Absorption rates are near-zero on all non-spelling domains | Medium | This would be a meaningful negative result (absorption is domain-specific); report it as such and pivot to analyzing why |
| L0 sweep requires training new SAEs (pre-trained not available at all L0 values) | Medium | Use SAEBench's training code on Gemma 2 2B; each BatchTopK SAE trains in <2 hours on A100 |
| Multicollinearity in Part 3 regression | Medium | Check VIF; use partial correlation; report both full and reduced models |
| Hedging-absorption decomposition is unreliable | Medium | Validate on SynthSAEBench first; report decomposition accuracy before applying to real SAEs |

### Novelty Claim

This paper makes three specific empirical contributions that do not exist in the literature:

1. **First cross-domain absorption characterization**: All prior absorption studies use only the first-letter spelling task. We provide the first measurement of absorption rates on entity-type hierarchies (RAVEL), custom knowledge hierarchies, and sentiment-topic hierarchies---establishing (or falsifying) the generality of absorption as a SAE pathology.

2. **First systematic disentanglement of absorption from hedging**: Feature hedging and absorption both cause features to fail to fire, but have different root causes. We provide the first controlled experiment that quantifies how much of observed "absorption" is actually hedging, using L0 sweeps with validation on synthetic ground truth.

3. **First quantification of absorption's downstream impact**: We provide the first regression analysis linking per-SAE absorption rate to the performance gap between SAE probes and dense linear probes on downstream interpretability tasks, controlling for hedging, L0, width, and architecture---directly answering whether absorption matters for practical SAE applications.

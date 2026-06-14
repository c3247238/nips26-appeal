# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024) "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** --- The canonical absorption measurement protocol: k-sparse probing to find feature splits, false-negative identification, integrated-gradients ablation, cosine similarity thresholding (>0.025) with gap criterion (>=1.0). Critical methodological limitation: the absorption metric requires a priori knowledge of ground-truth probe directions, which restricts evaluation to tasks where the feature hierarchy is pre-specified by the experimenter. Absorption rate 15--35% on Gemma Scope 16k/65k SAEs; validated on hundreds of SAEs across Gemma 2 2B, Llama 3.2 1B, Qwen2 0.5B.

2. **SAEBench (Karvonen et al., ICML 2025, arXiv:2503.09532)** --- Extended absorption metric that uses probe direction contribution instead of ablation effect, enabling evaluation across all layers. Key methodological finding: "gains on proxy metrics do not reliably translate to better practical performance." Also reveals a confound: after fixing ReLU training, ReLU SAEs had the HIGHEST absorption---previously low absorption was an artifact of high dead-feature rates. Recommends training multiple SAEs across L0 range [20, 200] with directly comparable baselines.

3. **RAVEL (Huang et al., ACL 2024)** --- Entity-attribute disentanglement benchmark with 5 entity types (cities, Nobel laureates, verbs, objects, occupations), each with 4-6 attributes. Provides hierarchical attribute structure (city -> country, city -> continent, city -> language) with known ground truth from Wikidata. PCA and SAE achieved the lowest disentanglement scores in the original evaluation---unsupervised methods struggle with disentanglement. Multi-task DAS was most effective.

4. **"Which Sparse Autoencoder Features Are Real? Model-X Knockoffs for FDR Control" (Enkhbayar, arXiv:2511.11711, Nov 2025)** --- First application of formal statistical hypothesis testing to SAE feature selection. Using knockoff+ on Pythia-70M SAE features for sentiment classification, finds only 25% of examined latents carry task-relevant signal; 75% are spurious. Provides finite-sample FDR guarantees. Highly relevant for establishing statistical rigor in absorption studies---currently no absorption study reports p-values, confidence intervals, or false discovery rates.

5. **Feature Sensitivity (Tian et al., arXiv:2509.23717, NeurIPS 2025 Workshop Spotlight)** --- Frames absorption as a special case of poor feature sensitivity. Critically, controls for feature frequency confounds via reweighting. Wider SAEs show lower average feature sensitivity; at a given width, SAEs with more active latents have higher sensitivity. Demonstrates a dimension of feature quality missed by all existing evaluations including absorption.

6. **SynthSAEBench (arXiv:2602.14687, Feb 2026)** --- Synthetic benchmark with known ground-truth features, including hierarchical structure and Zipfian firing distributions. Logistic probes achieve F1 = 0.974 while best SAEs substantially underperform. SAEBench metrics exhibit "substantial noise between runs"---a critical finding that most absorption studies do not account for.

7. **"Sparse but Wrong" (Chanin & Garriga-Alonso, arXiv:2508.16560, 2025)** --- Shows most open-source SAEs have incorrect L0 (too low), causing systematic feature hedging/mixing. Proposes proxy metric for finding correct L0. Essential confound for any absorption study: if L0 is wrong, measured "absorption" may partly be hedging artifacts.

8. **"Feature Hedging" (Chanin et al., arXiv:2505.11756, 2025)** --- Hedging (merging correlated features in narrow SAEs) is distinct from absorption (hierarchy-driven feature suppression) but produces the same observable symptom: features failing to fire where they should. Balanced Matryoshka SAE (compound multiplier ~0.75) helps, but reveals a fundamental absorption-hedging tradeoff.

9. **Matryoshka SAE (Bussmann et al., ICML 2025, arXiv:2503.17547)** --- Best absorption score on SAEBench (~0.03 vs BatchTopK ~0.29). Nested prefix losses create natural feature hierarchy. However: inner levels suffer from feature hedging, and minor reconstruction penalty. Key methodological point: the improvement is only validated on the first-letter task.

10. **ATM SAE (Li et al., arXiv:2510.08855, 2025)** --- Achieves absorption score 0.0068 vs TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B. Best reported absorption scores. But: evaluated only on a single model (Gemma-2-2B) and per-latent importance tracking adds training overhead. Statistical significance of the improvement not reported.

11. **HSAE "From Atoms to Trees" (Luo et al., arXiv:2602.11881, Feb 2026)** --- Follows SAEBench evaluation protocol exactly. HSAE substantially outperforms baseline SAEs on absorption score, particularly at larger dictionary sizes. Similarity to Matryoshka SAE results suggests hierarchical structure is the key ingredient for absorption mitigation, not any specific architectural trick.

12. **"Sanity Checks for SAEs" (Korznikov et al., arXiv:2602.14111, Feb 2026)** --- SAEs recover only 9% of true features in synthetic settings; random baselines match SAEs on interpretability, sparse probing, and causal editing. Raises the bar for any absorption study: we must demonstrate our measurements are not achievable by random baselines.

### Experimental Landscape

**What has been properly tested:**
- Absorption exists on the first-letter spelling task across hundreds of SAEs, multiple models, and multiple architectures (Chanin et al., SAEBench)
- Matryoshka SAE reduces absorption on the first-letter task (SAEBench, HSAE)
- Absorption persists in BatchTopK SAEs that lack L1 loss, demonstrating it is not purely an L1 artifact

**What is accepted WITHOUT proper experimental evidence:**
- That absorption rates on first-letter generalize to other feature hierarchies (never tested)
- That absorption is the primary cause of SAE underperformance on safety tasks (DeepMind blog, no controlled experiment)
- That wider SAEs should show less absorption (Chanin et al. observe "wider and more sparse SAEs show HIGHER absorption," contradicting the intuitive prediction)
- That Matryoshka SAE's low absorption generalizes beyond first-letter (only tested on one task)
- That ATM SAE's extremely low absorption score (0.0068) is statistically significant vs noise

**Critical methodological gaps:**
- No absorption study reports confidence intervals, effect sizes, or statistical tests
- The cosine similarity threshold (0.025) and gap criterion (>=1.0) in the absorption metric are arbitrary and their sensitivity has not been characterized
- SAEBench notes "substantial noise between runs" but no study accounts for this variance
- Feature frequency confounds (acknowledged in Tian et al.) are not controlled in any absorption study
- The L0 confound (Chanin & Garriga-Alonso, 2025) has never been systematically controlled in absorption measurements

---

## Phase 2: Initial Candidates

### Candidate A: Controlled Cross-Domain Absorption Measurement with Statistical Rigor

- **Core hypothesis**: Absorption rates measured on the first-letter spelling task are NOT representative of absorption severity on semantically richer feature hierarchies. Specifically, feature hierarchies with higher semantic overlap between parent and child (e.g., "country" and "continent" share more variance than "first letter" and "token identity") will show HIGHER absorption rates because the sparsity incentive to merge them is stronger.
- **Falsification criterion**: If absorption rates on entity-type hierarchies (city->continent, city->country) fall within the 95% bootstrap confidence interval of the first-letter absorption rate (15-35%) after controlling for probe accuracy and L0, the hypothesis is falsified---absorption is task-independent.
- **Evaluation protocol**: Absorption rate (Chanin et al. / SAEBench extended metric) on 4 hierarchy types, with bootstrap 95% CIs (10,000 resamples), paired permutation tests across tasks, Bonferroni correction for multiple comparisons. Report results on Gemma 2 2B residual stream at L5, L12, L20 with Gemma Scope JumpReLU SAEs (16k, 65k) + SAEBench SAEs (7 architectures at 16k). Minimum 3 random seeds for probe training.
- **Ablation plan**: (a) Task ablation: 4 tasks isolate hierarchy-type effect. (b) Width ablation: 4k/16k/65k isolate capacity effect. (c) Architecture ablation: 7 SAEBench architectures isolate training-procedure effect. (d) Layer ablation: L5/L12/L20 isolate representational-maturity effect. Each ablation has a clear prediction.
- **Confounders identified**: (1) Probe accuracy varies across tasks; if probes are less accurate on entity tasks, false negatives may be misattributed to absorption. Mitigation: report probe accuracy alongside absorption rate, exclude tasks with probe accuracy <85%. (2) L0 confound: SAEs with incorrect L0 may show hedging that mimics absorption. Mitigation: classify SAEs by L0 correctness using Chanin & Garriga-Alonso's proxy metric. (3) Feature frequency: absorption may correlate with how often parent-child features co-occur. Mitigation: compute co-occurrence rates and include as covariate. (4) Metric threshold sensitivity: the 0.025 cosine sim threshold may not be appropriate for entity features. Mitigation: sweep thresholds and report sensitivity.
- **Pilot design**: Load Gemma 2 2B + Gemma Scope 16k SAE at L12. Train LR probes for first-letter AND city->continent. Run absorption detection on both. Compare rates. Check probe accuracy and execution time. If probe accuracy for city->continent >85% and absorption is detectable, proceed. ~15 minutes.

### Candidate B: Disentangling Absorption from Hedging via Controlled L0 Sweep

- **Core hypothesis**: At least 30% of what SAEBench reports as "absorption" in open-source SAEs is actually L0-induced hedging, not true hierarchy-driven absorption. True absorption persists at correct L0 but at a lower rate than currently believed.
- **Falsification criterion**: If absorption rate shows no correlation with L0 correctness (Spearman rho < 0.1, p > 0.05) across SAEBench SAEs, the hypothesis is falsified---L0 is not a confound.
- **Evaluation protocol**: Use all 200+ SAEBench SAEs on Gemma-2-2B L12. For each SAE: (a) compute L0 correctness proxy (sparse-but-wrong-paper), (b) compute hedging metric (feature-hedging-paper), (c) compute absorption rate (SAEBench absorption eval). Run Spearman rank correlation: absorption_rate ~ L0_error, absorption_rate ~ hedging_rate. Partial correlation controlling for width and architecture. Bootstrap CIs on all correlations.
- **Ablation plan**: (a) Restrict to BatchTopK SAEs only (6 sparsity levels)---architecture held constant. (b) Restrict to SAEs with correct L0 only---measure "pure" absorption. (c) Compare absorbed-input L0 vs non-absorbed-input L0 within each SAE. Each ablation tests a specific causal claim.
- **Confounders identified**: (1) Architecture confounds L0 correctness: TopK SAEs may have systematically different L0 correctness than JumpReLU. Mitigation: within-architecture analysis. (2) Width confounds both hedging and absorption: wider SAEs reduce hedging but may increase or decrease absorption. Mitigation: partial correlation controlling for width. (3) Training duration: JumpReLU SAEs trained longer show more absorption (SAEBench finding), but duration also affects L0 convergence. Mitigation: report training steps as covariate.
- **Pilot design**: Take the 6 BatchTopK SAEs at Gemma-2-2B L12 from SAEBench (1 per sparsity level). Compute absorption rate and L0 for each. Plot absorption vs L0. If clear monotonic trend, confound is real. ~10 minutes.

### Candidate C: Statistical Audit of the Absorption Metric Itself

- **Core hypothesis**: The standard absorption metric (Chanin et al.) has a false-positive rate >10% due to (a) arbitrary threshold choices (cosine sim >0.025, gap >=1.0), (b) uncontrolled multiple testing across letters, and (c) sensitivity to probe training randomness. A statistically grounded version of the metric with FDR control and bootstrap CIs will yield substantially different (lower) absorption rates.
- **Falsification criterion**: If the standard metric and the FDR-controlled metric agree to within 5 percentage points on absorption rate across 5 SAEs, the hypothesis is falsified---the existing metric is already statistically sound.
- **Evaluation protocol**: (a) Threshold sensitivity: sweep cosine sim threshold from 0.01 to 0.10 in steps of 0.005; sweep gap criterion from 0.5 to 2.0 in steps of 0.25. Plot absorption rate as a function of both thresholds. (b) Probe training variance: train 10 probes with different random seeds; measure absorption rate for each. Report std and CV across seeds. (c) FDR control: adapt Model-X knockoffs framework (Enkhbayar, 2025) or simpler Benjamini-Hochberg procedure to absorption detection across letters. (d) Random baseline: compute "absorption rate" using random decoder directions instead of SAE latents, following Korznikov et al. (2026) sanity check protocol. Report on 3 Gemma Scope SAEs (16k, 65k at L12; 16k at L5).
- **Ablation plan**: (a) Threshold ablation: which threshold choice drives the published rate? (b) Seed ablation: how stable is the metric across probe random seeds? (c) Random baseline: is the metric detecting something non-trivial? (d) FDR correction: how many individual "absorbed letters" survive multiple testing correction?
- **Confounders identified**: (1) The metric might be stable in aggregate but unstable per-letter. (2) Different layers may have different optimal thresholds. (3) The SAEBench extended metric and original Chanin metric may disagree, creating version-dependence in the literature.
- **Pilot design**: Take 1 Gemma Scope 16k SAE at L12. Run the standard absorption metric. Then re-run with 5 different probe seeds. Compute variance. Also run with random decoder directions. ~15 minutes.

---

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption

- **Confound attack**: The most dangerous confound is PROBE ACCURACY. On first-letter tasks, probes achieve near-perfect accuracy (>99%). On entity tasks (city->continent), probe accuracy is typically 85-95% (confirmed by ICLR 2025 entity recognition paper on Gemma 2 2B, and RAVEL showing SAEs achieve lowest disentanglement scores). If probes are less accurate, the absorption metric has a higher floor: some "false negatives" are genuine probe errors, not absorption. I searched for papers on disentangling probe error from absorption---none exist. Mitigation: use probe accuracy as a normalizing factor and report absorption conditional on correctly-probed inputs only (which is what Chanin et al. already do, since they restrict to tokens "correctly classified by the probe"). But if the probe has lower accuracy on entity tasks, there are fewer "correctly classified" tokens, reducing statistical power. This is manageable if probe accuracy >85%.

- **Statistical attack**: The expected effect size is unknown. If absorption rates on entity tasks are 20% (similar to first-letter), we need ~200 tokens per letter/entity-type to detect a 10-percentage-point difference with 80% power (standard z-test for proportions). RAVEL has ~3000 cities across ~50 countries and 6 continents, so continent-level analysis has ~500 cities per continent---sufficient. Country-level analysis may have <30 cities per country for small countries---could lack power. The bootstrap CI approach avoids parametric assumptions.

- **Benchmark attack**: Is RAVEL the right benchmark for this claim? RAVEL was designed for disentanglement, not absorption specifically. The attribute structure (city->continent) is a clean hierarchy, but the probing task differs from first-letter: first-letter is a syntactic task (deterministic, observable in token embedding), while city->continent requires factual knowledge (stored in later layers). This means the relevant layers may differ, which is actually a feature, not a bug---it tests whether absorption depends on where in the network the hierarchy is represented.

- **Ablation completeness attack**: Missing ablation: co-occurrence frequency as a predictor. If "France" and "Europe" co-occur far more frequently than "s" and "snake," the sparsity incentive to absorb differs. Should include co-occurrence statistics as a covariate. Also missing: training data distribution---SAEs trained on OpenWebText may have different exposure to city names vs letter patterns. The ablation schedule does not explicitly test this.

- **Verdict**: **STRONG** --- The experimental design is sound with manageable confounds. The core measurement is feasible with existing tools. Statistical power is sufficient for continent-level analysis. The main risk (probe accuracy) is mitigable. Adding co-occurrence frequency analysis and reporting all results conditional on probe accuracy would strengthen it further.

### Against Candidate B: Disentangling Absorption from Hedging

- **Confound attack**: The L0 correctness proxy from sparse-but-wrong-paper has only been validated on toy models. Its accuracy on real Gemma-2-2B SAEs is unknown. If the proxy is inaccurate, the entire decomposition is unreliable. I searched for validation of this proxy on real LLM SAEs---none found. This is a significant risk.

- **Statistical attack**: With 200+ SAEBench SAEs, the statistical power for correlation analysis is high. However, the SAEs differ in architecture, width, and training procedure, introducing many confounders. Partial correlation controlling for these factors may leave few residual degrees of freedom. Within-architecture analysis (e.g., only the 6 BatchTopK SAEs) has N=6, which is too small for reliable correlation.

- **Benchmark attack**: Using only the first-letter task means this study inherits all limitations of that single task. If L0 affects absorption differently on different tasks (plausible: syntactic vs semantic features may have different L0 sensitivity), the results may not generalize.

- **Ablation completeness attack**: The proposed decomposition FN_rate = f(absorption) + f(hedging) + f(residual) assumes additivity. But absorption and hedging may interact: a narrow SAE with incorrect L0 might show BOTH hedging AND absorption simultaneously, with one exacerbating the other. The linear decomposition could miss nonlinear interactions. A proper test would require a factorial design (correct/incorrect L0 x narrow/wide SAE x with/without hierarchy), which requires training new SAEs---violating the project's training-free constraint.

- **Verdict**: **MODERATE** --- The conceptual framing is important, but execution has significant risks: untested L0 proxy, small within-architecture sample sizes, assumption of additive decomposition. Better as a secondary analysis within a larger study than as a standalone contribution.

### Against Candidate C: Statistical Audit of the Metric

- **Confound attack**: The metric audit is internally valid---it tests the metric itself, not a downstream claim. The main risk is that the audit reveals the metric is fine (stable across thresholds and seeds), yielding a null result. This is informative but not exciting.

- **Statistical attack**: The threshold sensitivity sweep is straightforward and well-defined. The probe seed variance test has clear power (10 seeds). The random baseline test is a strong sanity check. The FDR control adaptation is methodologically novel but may be technically complex (adapting knockoffs to the specific structure of absorption detection requires careful work).

- **Benchmark attack**: Auditing the metric only on first-letter (where the metric was designed) may be insufficient. The metric might work well for first-letter but fail on other tasks. However, we cannot audit the metric on tasks where it has not yet been applied, so this limitation is inherent to the scope.

- **Ablation completeness attack**: Missing: comparison between the original Chanin et al. metric and the SAEBench extended metric on the same SAEs. If they disagree, the field has a metric-version confound. This is a quick, high-value addition.

- **Verdict**: **MODERATE-STRONG** --- Methodologically rigorous and fills a real gap (no existing statistical audit of the absorption metric). But the finding may not be exciting enough for a standalone paper. Best integrated as a foundational component of a larger study.

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate B (standalone)**: The disentangling of absorption vs hedging is important but the untested L0 proxy and small within-architecture samples make it too risky as a primary contribution. The additive decomposition assumption is likely wrong. Key elements (L0 confound control) should be absorbed as a control experiment in the final proposal.

### Strengthened Survivors

**Candidate A (Cross-Domain Absorption)**: Strengthened with:
- Addition of co-occurrence frequency as a covariate in the analysis
- Integration of Candidate C's statistical rigor: all absorption rates reported with bootstrap 95% CIs, threshold sensitivity analysis, probe seed variance, random baseline check
- L0 confound control from Candidate B: classify each SAE by L0 correctness and report absorption rates stratified by L0 correctness
- Comparison of Chanin et al. original metric vs SAEBench extended metric on the same SAEs (from Candidate C)

**Candidate C elements integrated**: The statistical audit becomes the "Methods Validation" phase of the final proposal rather than a standalone contribution. This grounds the entire study in proper statistical methodology.

### If All Had Died
Would fall back to a pure "measurement paper": systematic characterization of the absorption metric's statistical properties (variance, sensitivity, FDR) as a community resource. This has clear utility even if less exciting.

### Selected Front-Runner
**Candidate A + C integration**: Cross-domain absorption characterization with rigorous statistical methodology. The experimental design is the star: proper controls, statistical tests, threshold sensitivity, random baselines, and FDR-aware reporting. The specific feature hierarchies are the supporting cast.

---

## Phase 5: Final Proposal

### Title
Measuring What Matters: A Rigorous Cross-Domain Evaluation of Feature Absorption in Sparse Autoencoders

### Hypothesis
Feature absorption severity is a function of feature hierarchy properties (semantic overlap, co-occurrence frequency, hierarchy depth), not a fixed property of the SAE. Specifically:
- **H1**: Absorption rate on semantically overlapping hierarchies (city->country, city->language) is significantly higher than on syntactically deterministic hierarchies (first-letter), when measured at the same SAE and layer. Significance threshold: two-sided paired permutation test p < 0.01 after Bonferroni correction for 6 pairwise comparisons.
- **H2**: Absorption rate correlates positively with parent-child feature co-occurrence frequency (Spearman rho > 0.3, p < 0.05), controlling for SAE width and L0.

### Falsification Criterion
H1 is falsified if the bootstrap 95% CI for the difference in absorption rates (entity-task minus first-letter) includes zero for ALL entity tasks across ALL tested SAEs. H2 is falsified if Spearman rho is not significantly different from zero after controlling for confounders. If both are falsified, the conclusion is that absorption is task-independent (also publishable, as it validates generalization of the first-letter metric).

### Method
Training-free analysis of existing pretrained SAEs. No SAE training required.

**Models and SAEs:**
- Gemma 2 2B (primary model, ~5 GB VRAM in fp16)
- Gemma Scope JumpReLU SAEs: widths 16k and 65k at layers 5, 12, 20
- SAEBench pretrained SAEs: 7 architectures (BatchTopK, TopK, JumpReLU, Gated, ReLU, Standard, Matryoshka) at width 16k on L12
- Total: 6 (Gemma Scope) + 7 (SAEBench) = 13 SAEs minimum

**Probe Tasks with Known Feature Hierarchies:**
- Task 1: First-letter spelling (baseline, from sae-spelling, hierarchy: token identity -> first letter, 26 parent classes)
- Task 2: City -> Continent (RAVEL, 6 parent classes, ~3000 cities, shallow hierarchy)
- Task 3: City -> Country (RAVEL, ~50 parent classes, same cities, deeper hierarchy)
- Task 4: City -> Language (RAVEL, ~30 parent classes, correlated with country, tests whether correlated hierarchies show more absorption)

### Evaluation Protocol

**Primary metric:** Absorption rate (SAEBench extended variant: probe direction contribution method) per (task, SAE, layer) triple.

**Statistical plan:**
- Bootstrap 95% confidence intervals (10,000 resamples per measurement)
- Paired permutation test (5,000 permutations) for cross-task absorption rate comparisons within each SAE
- Bonferroni correction for C(4,2) = 6 pairwise task comparisons
- Spearman rank correlation for absorption rate vs hierarchy properties (co-occurrence frequency, number of classes, semantic overlap)
- Report effect sizes (Cohen's d for task differences, rho for correlations)
- Minimum 3 random seeds for all probe training; report inter-seed variance

**Random baseline (sanity check):**
- Repeat the absorption detection pipeline using random decoder directions (uniformly sampled from the unit sphere in activation space) instead of SAE latent decoder directions
- This establishes the "floor" absorption rate: if the real SAE's absorption rate does not significantly exceed the random baseline, the metric is not detecting a real phenomenon

**Threshold sensitivity analysis:**
- Sweep cosine similarity threshold: [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
- Sweep gap criterion: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- Report absorption rate heatmap as function of both thresholds
- Identify whether conclusions are threshold-robust

**Metric version comparison:**
- Run both Chanin et al. original metric (integrated-gradients ablation) and SAEBench extended metric (probe direction contribution) on the same SAEs
- Report agreement (Pearson r on absorption rates across letters/entities)

### Ablation Schedule

| Ablation | What is varied | What is held constant | What it tests | Expected outcome |
|----------|---------------|----------------------|---------------|-----------------|
| Task | 4 hierarchy types | Architecture, width, layer | Whether hierarchy properties affect absorption | Entity tasks show different rates than first-letter |
| Width | 16k vs 65k | Architecture (JumpReLU), layer, task | Whether capacity reduces absorption | Unclear: Chanin et al. found WIDER SAEs have MORE absorption; this tests replication |
| Layer | L5, L12, L20 | Architecture, width, task | Whether representational maturity affects absorption | Entity-task absorption shifts to later layers where factual knowledge is encoded |
| Architecture | 7 SAEBench architectures | Width (16k), layer (L12), task | Whether training algorithm affects absorption | Matryoshka best, JumpReLU worst (replicating SAEBench), but does ranking hold on entity tasks? |
| L0 correctness | Correct vs incorrect L0 (proxy metric) | Everything else | Whether L0 confound inflates absorption | Absorption persists at correct L0 but at lower rate |
| Threshold | Cosine sim x gap criterion | Everything else | Whether results are threshold-robust | Main conclusions hold across reasonable threshold range |

### Control Experiments

1. **Probe accuracy control**: For every (task, layer) pair, report probe accuracy (balanced accuracy, macro-F1). If probe accuracy <80% for any task at any layer, exclude that (task, layer) pair and flag in limitations. Ensures absorption is not conflated with probe failure.

2. **Random baseline control**: Repeat absorption metric with random decoder directions. If the random baseline's absorption rate overlaps the SAE's rate (within 95% CI), flag the measurement as potentially spurious. This follows Korznikov et al. (2026) methodology.

3. **L0 confound control**: Compute L0 correctness proxy for each SAE. Report absorption rates separately for "correct L0" and "incorrect L0" SAEs. If absorption is significantly higher in "incorrect L0" SAEs, hedging contamination is confirmed.

4. **Feature frequency control**: Compute the activation frequency of each tested SAE latent involved in absorption. Report whether absorbed latents have systematically different frequencies than non-absorbed latents. This follows the frequency-reweighting methodology from Tian et al. (2025).

### Pilot Design
A <15-minute experiment that gives early signal on feasibility.

**Steps:**
1. Load Gemma 2 2B + Gemma Scope 16k JumpReLU SAE at L12 via SAELens (~2 min on GPU)
2. Train LR probe for "continent of city" on residual stream activations at L12, using RAVEL city data (~3 min)
3. Check probe accuracy. If <80%, try L20 instead. If still <80%, this task is infeasible.
4. Run sae-spelling absorption pipeline adapted for city->continent (~5 min)
5. Compare against published first-letter absorption rate for the same SAE
6. Check: does the random baseline (random decoder directions) produce lower absorption than the real SAE?

**Decision criteria:**
- If probe accuracy >85% AND absorption is detectable AND random baseline is significantly lower: **PROCEED**
- If probe accuracy 80-85%: **PROCEED WITH CAUTION** (note reduced statistical power)
- If probe accuracy <80%: **PIVOT** to a different task or layer
- If absorption rate is indistinguishable from random baseline: **INVESTIGATE** threshold choices before proceeding

### Resource Estimate

| Component | GPU-hours | Notes |
|-----------|-----------|-------|
| Pilot | 0.25 | 1 SAE, 2 tasks, 1 layer |
| Probe training | 2 | 4 tasks x 3 layers x 3 seeds, ~3 min each |
| Absorption evaluation (Gemma Scope) | 6 | 6 SAEs x 4 tasks, ~15 min each |
| Absorption evaluation (SAEBench) | 7 | 7 SAEs x 4 tasks, ~15 min each |
| Threshold sensitivity | 2 | 1 SAE, 4 tasks, 54 threshold combos |
| Random baseline | 2 | 3 SAEs, 4 tasks |
| L0 confound analysis | 1 | 13 SAEs, L0 proxy computation |
| Metric version comparison | 2 | 3 SAEs, 4 tasks, 2 metric variants |
| **Total** | **~22** | Fits on 1x 16 GB GPU; ~3-4 days wall clock |

**Model sizes:** Gemma 2 2B (~5 GB); 16k SAE adds ~150 MB; 65k SAE adds ~600 MB. Total VRAM per experiment: ~8-12 GB. Comfortably fits on a single 16 GB GPU.

**Training-free constraint:** Fully satisfied. All SAEs are pretrained (Gemma Scope, SAEBench). Only logistic regression probes are trained (<1 min each).

### Risk Assessment

| Threat to validity | Severity | Mitigation |
|-------------------|----------|------------|
| Probe accuracy too low on entity tasks (<80%) | High | Test on multiple layers (L5/L12/L20); fall back to simpler hierarchy (entity-type classification) if needed. ICLR 2025 paper shows entity probing works on Gemma 2 2B. |
| Absorption metric thresholds are task-dependent | Medium | Full threshold sensitivity analysis. Report results at multiple threshold settings. Identify threshold-robust conclusions. |
| Absorption rate is identical across all tasks | Medium | This is a valid null result (absorption generalizes uniformly). Publishable as "first-letter results replicate on entity tasks." Combine with scaling analysis. |
| L0 proxy metric is inaccurate on real SAEs | Medium | Treat L0 analysis as supplementary, not primary. Main conclusions (cross-domain absorption) do not depend on L0 proxy accuracy. |
| SynthSAEBench's finding that SAEBench has "substantial noise between runs" | High | Multiple probe seeds, bootstrap CIs, random baselines. Report variance explicitly. If absorption variance is high, increase sample sizes or aggregate across letters/entities. |
| Matryoshka SAE's low absorption on first-letter does NOT hold on entity tasks | Medium | This would be a STRONG positive finding---demonstrating that mitigation claims do not generalize. |
| Gemma Scope SAEs have systematic issues (JumpReLU has worst absorption per SAEBench) | Low | SAEBench SAEs provide 7 architecture controls. JumpReLU bias is documented and can be discussed. |

### Novelty Claim

The experimental contribution answers two specific empirical questions for the first time:

1. **Does absorption generalize beyond the first-letter spelling task?** No previous study has measured absorption on entity-type, factual-knowledge, or semantically correlated feature hierarchies. This is the single most important missing piece in the absorption literature.

2. **Is the absorption metric statistically sound?** No previous study has reported confidence intervals, conducted threshold sensitivity analysis, applied FDR-aware testing, or validated against random baselines for the absorption metric. The statistical methodology alone is a contribution to the field.

What is NOT claimed as novel: the absorption metric itself (Chanin et al.), the SAE architectures (various), the pretrained SAEs (Gemma Scope, SAEBench), the RAVEL dataset (Huang et al.), or the concept that feature hierarchy causes absorption (Chanin et al.'s toy model). The novelty is in the rigorous cross-domain experimental design and the statistical framework---measuring what was claimed but never properly tested.

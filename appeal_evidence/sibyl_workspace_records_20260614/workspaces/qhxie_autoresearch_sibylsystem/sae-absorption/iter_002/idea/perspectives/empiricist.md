# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** — Defines the canonical absorption metric: k-sparse probing + false-negative detection + integrated-gradients ablation. The gold-standard evaluation protocol, but restricted to the first-letter spelling hierarchy. Key methodological detail: absorption is detected when the highest-ablation-effect latent has cosine similarity > 0.025 with the probe direction AND is >= 1.0 larger than the second-highest. These thresholds are somewhat arbitrary and their sensitivity has not been studied.

2. **Karvonen et al. (2025), SAEBench (arXiv:2503.09532, ICML 2025)** — 8-metric evaluation suite across 200+ SAEs. Crucially, they found that proxy metrics (CE loss, sparsity) do NOT reliably predict practical performance. Their absorption implementation extends Chanin et al. with partial absorption detection and multi-latent responsibility cases. Also revealed a confound: original ReLU SAE absorption results were artifacts of high dead-feature ratios, not genuine low absorption.

3. **SynthSAEBench (arXiv:2602.14687, Feb 2026)** — Large-scale synthetic benchmark with configurable hierarchy, correlation, superposition, and Zipfian firing distributions. Ground-truth features known. Logistic probes achieve 0.974 F1 while best SAEs substantially underperform. This is the only benchmark that enables fully controlled ablation of hierarchy depth, co-occurrence frequency, and sparsity independently.

4. **Tian et al. (2025), "Measuring SAE Feature Sensitivity" (arXiv:2509.23717)** — Frames absorption as a special case of poor feature sensitivity. Develops scalable sensitivity evaluation that reveals many features appearing monosemantic in activating examples actually have poor sensitivity. This metric captures a dimension missed by the Chanin et al. absorption metric alone.

5. **Korznikov et al. (2026), "Sanity Checks for SAEs" (arXiv:2602.14111)** — Shows SAEs recover only 9% of true features in synthetic settings; random baselines match fully-trained SAEs on interpretability, sparse probing, and causal editing. Devastating sanity check that any rigorous experiment must address: claimed improvements over random baselines require explicit comparison.

6. **DeepMind Safety Research (2025), "Negative Results for SAEs on Downstream Tasks"** — Dense linear probes achieve near-perfect accuracy (even OOD) while 1-sparse SAE probes fail catastrophically on harmful intent detection. The most important negative result for motivating absorption research, but the link between absorption specifically and probe failure has NOT been quantified.

7. **Gao et al. (2024), OpenAI Scaling SAEs (arXiv:2406.04093, ICLR 2025)** — Acknowledged that results were "confounded by encoder learning rate" without further investigation. Important methodological warning: hyperparameter confounds can invalidate architecture comparisons.

8. **RAVEL benchmark (Huang et al., 2024, github.com/explanare/ravel)** — Entity-attribute disentanglement evaluation on cities (country, continent, language) and Nobel Prize winners (country, field, gender). Already integrated into SAEBench. Provides the entity-attribute hierarchies needed for cross-domain absorption measurement. 3000+ cities with structured attributes.

9. **CE-Bench (Gulko et al., arXiv:2509.00691, BlackboxNLP 2025)** — Fully LLM-free, contrastive evaluation benchmark. >70% Spearman correlation with SAEBench. Important methodological contribution: deterministic evaluation removes the non-reproducibility introduced by LLM-based interpretability scoring.

10. **Unified SDL Theory (arXiv:2512.05534, 2024)** — Casts all SDL variants as piecewise biconvex optimization; provides principled explanations for absorption and dead neurons via spurious minima analysis. Proposes feature anchoring but validates only on synthetic benchmarks. The theoretical framework is sound, but the gap between synthetic validation and real LLM behavior is a major untested assumption.

11. **SAE Probes Underperform Baselines (arXiv:2502.16681, Feb 2025)** — Over 100 datasets showing SAE probes provide no improvement over traditional logistic regression probes. Emphasizes the "robust quiver of arrows" methodology: previously claimed SAE advantages were illusory under rigorous baselines. Any absorption study claiming improvements must compare against strong non-SAE baselines.

12. **Song et al. (2025), "Feature Consistency in SAEs" (arXiv:2505.20254)** — Shows SAE features are inconsistent across training runs; TopK achieves only 0.80 consistency (PW-MCC). Raises a fundamental confounder: if features themselves vary across runs, absorption measurements on single SAE instances may not generalize. Multi-run consistency checks are essential.

### Experimental Landscape

**What has been properly tested:**
- Absorption EXISTS in every SAE architecture tested (Chanin et al., validated on Gemma Scope, Llama 3.2, Qwen2)
- Absorption rate falls in the 15-35% range on the first-letter spelling task across 16k and 65k widths
- Wider and more sparse SAEs show higher absorption
- JumpReLU SAEs (trained longer) exhibit worse absorption than TopK
- Matryoshka SAEs reduce absorption (SAEBench)
- OrtSAE reduces absorption by ~65% (orthogonality penalty)
- ATM SAE achieves absorption score 0.0068 vs. TopK 0.1402 (Gemma-2-2B only)

**What is accepted without proper evidence:**
- That absorption rates on the first-letter task generalize to ANY other hierarchy type (zero cross-domain validation)
- That decoder cosine similarity is the primary driver of absorption severity (correlational, not causally validated)
- That absorption is always harmful: no study measures the cost of eliminating absorption in terms of overall model performance
- That absorption rates are stable across SAE training seeds (no consistency data for absorption specifically)
- That the 0.025 cosine similarity threshold and 1.0 gap threshold in the Chanin metric are appropriate (no sensitivity analysis)

**Where critical methodological gaps exist:**
- No shuffled-label null baseline for any absorption measurement (what is the false-positive rate of the metric itself?)
- No bootstrap confidence intervals reported for absorption rates (all point estimates)
- No multiple-comparisons correction when reporting cross-architecture results
- No control for feature quality: dead features inflate apparent low absorption (the ReLU confound SAEBench discovered)
- No temporal stability assessment: does absorption change during training, or is it an emergent property of convergence?

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy with Rigorous Null Controls

- **Hypothesis**: Feature absorption occurs at measurable rates (statistically distinguishable from shuffled-null baseline at p < 0.01 after Bonferroni correction) across at least 3 semantically distinct hierarchy types (first-letter spelling, entity-type from RAVEL cities, geographic country-continent, grammatical POS), and absorption severity is predicted by the frequency ratio (freq_parent / freq_child) with Spearman rho > 0.4.
- **Falsification criterion**: Absorption rates on non-spelling hierarchies are indistinguishable from shuffled-label permutation null (permutation test p > 0.05 after Bonferroni correction for 4 hierarchy types). If absorption is ONLY detectable on the first-letter spelling hierarchy, the phenomenon is far narrower than claimed.
- **Evaluation protocol**: SAEBench absorption metric (extended Chanin et al.) on Gemma Scope SAEs (16k, 65k widths, layers 6, 12, 18). Per hierarchy: logistic regression probe quality gate (F1 >= 0.80), 100 shuffled-label permutations for null distribution, bootstrap 95% CI on absorption rate (1000 resamples), Bonferroni correction across 4 hierarchy types. Effect size: Cohen's d for absorption rate vs. null.
- **Ablation plan**: (i) Remove frequency-ratio predictor, test if decoder cosine similarity alone predicts absorption. (ii) Remove cosine similarity predictor, test if frequency ratio alone suffices. (iii) Control for probe quality: vary probe F1 threshold from 0.70 to 0.95 and measure stability of absorption estimates. (iv) Control for SAE width: hold L0 constant, vary width.
- **Confounders**: (a) Probe quality may differ across hierarchy types, making cross-domain comparison unfair. Mitigate: report absorption normalized by probe recall. (b) RAVEL entities may not be well-represented in the SAE training corpus. Mitigate: verify entity frequency in OpenWebText before using. (c) Different hierarchies have different branching factors, confounding frequency ratios.
- **Pilot design**: Load GPT-2 Small SAE (layer 6, width 24576). Train first-letter probe. Measure absorption rate with and without shuffled-null baseline. Verify the pipeline works and that null rate is substantially below real rate. Estimated: 10-12 minutes.

### Candidate B: Absorption Threshold Stress Test via Synthetic Feature Hierarchies

- **Hypothesis**: The theoretical absorption threshold lambda > sin^2(theta_{p,c}) predicts which feature pairs exhibit absorption with AUROC >= 0.70 when validated against ground-truth absorption labels. Specifically, for all SAE configurations in Gemma Scope (layer 12, widths 16k and 65k, multiple L0 values), the fraction of absorbed pairs among those predicted-absorbed by the threshold exceeds 60% (precision@predicted).
- **Falsification criterion**: AUROC < 0.65 for threshold prediction on held-out SAE configurations. If the threshold has no better discriminative power than random, the rate-distortion theory is quantitatively wrong even if qualitatively directional.
- **Evaluation protocol**: For each SAE, compute decoder angles between all feature pairs identified by k-sparse probing on the first-letter task. Compute predicted absorption status from threshold. Compare against Chanin et al. ground truth. Report AUROC, AUPRC, precision-recall curves. Leave-one-L0-out cross-validation across Gemma Scope L0 settings.
- **Ablation plan**: (i) Replace threshold with random baseline (random pairs predicted as absorbed). (ii) Replace threshold with cosine-similarity-only ranking (no lambda dependence). (iii) Replace threshold with frequency-ratio-only ranking. (iv) Test whether adding activation kurtosis as an additional predictor improves AUROC.
- **Confounders**: (a) The Chanin et al. ground truth itself has unknown false-positive and false-negative rates. Mitigate: report results with varying thresholds for the ground-truth metric. (b) Decoder angles may be correlated with training artifacts (dead features, highly active features). Mitigate: exclude features with activation frequency < 0.001 or > 0.5. (c) The "effective lambda" for TopK SAEs is 1/L0, which is an approximation. Mitigate: also test on L1-regularized SAEs where lambda is directly specified.
- **Pilot design**: Load one Gemma Scope SAE (16k, layer 12, lowest L0). Compute decoder angles for the 26 first-letter feature clusters. Check whether absorbed pairs have systematically smaller angles than non-absorbed pairs (Wilcoxon rank-sum test). Estimated: 12-15 minutes.

### Candidate C: Probe-Free Absorption Detection via Decoder Geometry Anomalies

- **Hypothesis**: An Absorption Susceptibility Index (ASI) computed solely from SAE decoder weights and activation frequency statistics — ASI(i,j) = cos^2(theta_{i,j}) x (freq_i / freq_j) — achieves AUROC >= 0.70 against the Chanin et al. ground-truth absorption labels on the first-letter task, WITHOUT requiring any probe training. Furthermore, the top-100 ASI-ranked pairs contain at least 30% genuinely absorbed pairs (precision@100 >= 0.30).
- **Falsification criterion**: AUROC < 0.65 OR precision@100 < 0.15. If ASI has no better predictive power than random feature pairs sorted by cosine similarity alone, the frequency-ratio component adds nothing and the metric is not useful.
- **Evaluation protocol**: For one SAE (Gemma Scope 16k, layer 12), compute pairwise ASI for all feature pairs with co-activation frequency > 0.01 (pre-filter to reduce O(d^2) to ~10k pairs). Rank by ASI. Validate against Chanin et al. ground truth. Report AUROC, AUPRC, precision@k curves for k in {50, 100, 200, 500}. Compare against three baselines: (a) cosine similarity only, (b) frequency ratio only, (c) random ranking.
- **Ablation plan**: (i) ASI with only cosine similarity (drop frequency ratio). (ii) ASI with only frequency ratio (drop cosine similarity). (iii) ASI with L2 decoder norm ratio instead of frequency ratio. (iv) ASI with encoder-decoder misalignment metric added.
- **Confounders**: (a) Co-activation frequency threshold (0.01) is arbitrary; results may be sensitive to it. Mitigate: sweep threshold from 0.001 to 0.1 and report stability. (b) Absorbed pairs are rare (maybe 100 out of 10k candidate pairs), creating severe class imbalance. Mitigate: report AUPRC (which handles imbalance) alongside AUROC. (c) The first-letter task is the only ground-truth source available; ASI may overfit to spelling-specific patterns. Mitigate: if Candidate A succeeds, validate ASI on cross-domain hierarchies.
- **Pilot design**: Load GPT-2 Small SAE. Compute cosine similarity matrix for top-1000 most active features. Identify feature pairs with cosine similarity > 0.3. Check Neuronpedia labels to see if high-cosine pairs look like parent-child relationships. Estimated: 8-10 minutes.

---

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption Taxonomy

- **Confound attack**: The biggest confound is that different hierarchy types have fundamentally different representational structures in the model. First-letter spelling is an artificial, highly structured task where the hierarchy is cleanly defined. Entity-type hierarchies (city -> country) are learned from natural language co-occurrence and may not have clean linear representations. If probes for entity hierarchies achieve lower F1, the apparent "lower absorption" might simply be lower feature quality, not less absorption. The F1 >= 0.80 gate partially mitigates this, but a probe with F1 = 0.82 for spelling vs. F1 = 0.81 for entities could still have very different calibration properties. Additionally, the SAE training corpus (likely Pile/RedPajama) may have different coverage of entities vs. spelling patterns, creating a corpus-distribution confound.

- **Statistical attack**: With only 4 hierarchy types and potentially high variance within each type (26 letters vs. hundreds of entities), the statistical power for detecting a Spearman correlation of rho = 0.4 is questionable. With n = 4 hierarchy types, a Spearman test has essentially no power. The correlation analysis needs to be done at the individual feature-pair level (hundreds of pairs per hierarchy type), not at the hierarchy-type level. Even then, the dependency structure (all pairs within one hierarchy share the same SAE) violates independence assumptions. A hierarchical bootstrap or mixed-effects model is needed.

- **Benchmark attack**: The first-letter spelling task is the ONLY established ground-truth benchmark for absorption. Extending to entity hierarchies requires adapting the metric, and there is no guarantee that the Chanin et al. metric (which uses integrated-gradients ablation) works correctly for entity-attribute features. Entity features may be distributed across multiple layers, while the absorption metric operates on a single layer. The RAVEL dataset provides entity attributes but was designed for disentanglement evaluation, not absorption measurement — the mapping from RAVEL's binary intervention framework to the Chanin et al. false-negative framework is non-trivial.

- **Ablation completeness attack**: The proposed ablations isolate frequency ratio vs. cosine similarity, which is good. However, they do not ablate the choice of SAE layer (fixed at 12), which could matter: Chanin et al. found no clear layer-wise pattern, but entity representations may peak at different layers than spelling representations. A layer sweep is needed for at least one hierarchy type. Also, no ablation of the number of k in k-sparse probing, which affects which features are identified as "splits" and thus which are counted as absorbed.

- **Verdict**: STRONG. The core experiment — measuring absorption across multiple hierarchy types with null controls — is highly valuable regardless of the specific correlational analysis. The confounds are manageable with careful design. The main risk is that entity probes simply do not achieve sufficient quality, narrowing the cross-domain scope. This is a "report whatever happens" experiment: null results (no absorption on entity hierarchies) are equally publishable.

### Against Candidate B: Absorption Threshold Stress Test

- **Confound attack**: The theoretical threshold lambda > sin^2(theta_{p,c}) treats absorption as a binary decision for each feature pair, but real absorption is graded (partial absorption exists, as SAEBench found). The AUROC evaluation handles this somewhat, but the threshold is a hard boundary while the phenomenon is continuous. More fundamentally, the threshold assumes a simplified two-feature model; real SAEs have thousands of interacting features. Third-party features that are correlated with both parent and child could confound the pairwise analysis. The decoder angle theta is measured post-training; during training, the angle co-evolves with absorption itself, creating a circularity: absorbed pairs may have small angles BECAUSE absorption happened (the child decoder rotated to incorporate the parent direction), not because small angles CAUSED absorption.

- **Statistical attack**: The ground-truth labels from Chanin et al. are binary (absorbed/not absorbed), but absorption is a threshold-dependent classification. Changing the thresholds in the Chanin et al. metric (0.025 cosine similarity, 1.0 gap) could change which pairs are labeled as absorbed, affecting AUROC. A sensitivity analysis varying these thresholds is essential. Also, the number of feature pairs where absorption is measurable (requires known hierarchy) may be small enough (<100 pairs for 26 letters) that AUROC confidence intervals are wide.

- **Benchmark attack**: The leave-one-L0-out cross-validation is appropriate for testing generalization across sparsity settings. However, all validation is still on the first-letter task. If the threshold works for spelling but not for other hierarchies, its theoretical generality is questionable. Also, the Gemma Scope SAEs at different L0 values may not actually have independently set lambda values — they are trained with different hyperparameters that affect many things simultaneously.

- **Ablation completeness attack**: The proposed ablations (random, cosine-only, frequency-only, kurtosis addition) are well-designed. Missing: an ablation of the pairwise assumption itself. What if absorption is a collective phenomenon involving 3+ features? The threshold formula considers only pairs; a group-absorption analysis could reveal limitations.

- **Verdict**: MODERATE. The threshold validation is scientifically important but faces a circularity problem (decoder angles are endogenous to absorption) that is difficult to resolve without access to SAEs trained with controlled initial decoder geometry. The small sample size for ground-truth labels is a real concern. This experiment should be framed as "testing the discriminative power of the threshold" rather than "proving the threshold causes absorption."

### Against Candidate C: Probe-Free Absorption Detection via ASI

- **Confound attack**: ASI = cos^2(theta) x (freq_i / freq_j) is a heuristic with no guarantee that high-ASI pairs are actually absorbed rather than simply co-occurring features with similar decoder directions. Many non-hierarchical feature pairs could have high cosine similarity (e.g., synonymous features, features from adjacent layers of meaning). The frequency ratio could be high for non-hierarchical reasons (one feature fires on a common word, the other on a rare word). Without ground truth beyond the first-letter task, there is no way to validate ASI on new domains — creating a chicken-and-egg problem with Candidate A.

- **Statistical attack**: The class imbalance (absorbed pairs are rare) means that even a random ranking could achieve reasonable AUROC if the absorbed pairs happen to have somewhat higher cosine similarity than average. AUPRC is the right metric, but with only ~20-50 truly absorbed pairs out of ~10k candidates, even AUPRC will have very wide confidence intervals. A permutation test (shuffle ASI scores and recompute AUPRC) is essential to establish that ASI exceeds chance.

- **Benchmark attack**: Validation is limited to the first-letter task, the only setting where ground truth exists. ASI could achieve AUROC 0.70 on spelling features and completely fail on entity features, and we would not know without Candidate A's cross-domain ground truth. The practical utility of ASI depends on domains beyond spelling, but the evaluation is restricted to spelling.

- **Ablation completeness attack**: The ablations are well-structured (separating cosine similarity from frequency ratio). Adding decoder norm ratio and encoder-decoder misalignment are good extensions. Missing: comparison against meta-SAE decomposition as an alternative probe-free detector (if available), and comparison against the feature sensitivity metric of Tian et al. (2025), which addresses a related concept without requiring probe directions.

- **Verdict**: MODERATE. ASI is a clean, simple metric that is easy to compute and evaluate. The validation challenge is severe: without cross-domain ground truth, we cannot know if ASI generalizes. This candidate is best positioned as a secondary contribution that builds on Candidate A's cross-domain ground truth. If Candidate A produces ground truth for multiple hierarchies, ASI validation becomes dramatically more convincing.

---

## Phase 4: Refinement

### Dropped Ideas
None dropped entirely. All three candidates are feasible and address complementary aspects. However, the ordering of priority is clear from the critique.

### Strengthened Survivors

**Candidate A (front-runner) — strengthened with:**
1. **Hierarchical bootstrap**: Replace standard bootstrap with a two-level bootstrap that resamples both hierarchy types and feature pairs within each type, accounting for within-hierarchy dependence.
2. **Layer sweep**: For one hierarchy type (entity-type from RAVEL), run absorption measurement on layers 6, 12, 18 to verify that layer 12 is not a cherry-picked choice. Report layer with maximum absorption per hierarchy type.
3. **Probe calibration control**: Report Brier score for probes across hierarchy types. If Brier scores differ substantially, use calibrated probes (Platt scaling) before computing absorption.
4. **Entity frequency verification**: Before running cross-domain experiments, verify that RAVEL entities appear in the SAE training corpus with sufficient frequency (>10 occurrences per entity in a 100M-token sample).
5. **k-sparse probing sensitivity**: Run k from 1 to 10 and report absorption stability. If absorption estimates change dramatically with k, this is itself a finding about metric fragility.
6. **Absorption direction analysis**: For absorbed pairs, measure whether the "absorber" feature is always the more specific (child) feature, or whether parents can also absorb children. This tests the directionality assumption in all existing absorption theories.

**Candidate B — strengthened with:**
1. **Pre-training decoder analysis**: Use the initial (random) decoder directions from a freshly initialized SAE to establish the baseline angle distribution. Compare with post-training angles for absorbed vs. non-absorbed pairs. This partially addresses the circularity concern.
2. **Threshold sensitivity analysis**: Vary the Chanin et al. ground-truth thresholds and report AUROC stability. Present results as a heatmap over (cosine threshold, gap threshold) pairs.
3. **Multi-SAE-seed validation**: If available, use SAEs trained with different random seeds on the same model to test whether absorption threshold predictions are consistent.

**Candidate C — strengthened with:**
1. **Comparison against feature sensitivity**: Compute Tian et al.'s sensitivity metric alongside ASI and compare discriminative power.
2. **Conditional on Candidate A**: Only claim cross-domain validity if Candidate A provides cross-domain ground truth. Otherwise, frame ASI strictly as a first-letter-task predictor.
3. **Permutation null**: Shuffle ASI scores 1000 times to establish the null AUPRC distribution.

### Selected Front-Runner

**Candidate A: Cross-Domain Absorption Taxonomy with Rigorous Null Controls** is the front-runner because:
1. It addresses the single most impactful empirical gap (all absorption knowledge is from one task)
2. It produces the ground truth needed to validate Candidates B and C
3. It is publishable regardless of outcome (null results = absorption is task-specific, positive results = absorption is general)
4. The experimental design is straightforward and builds on existing tools (sae-spelling + RAVEL + SAELens)
5. It provides the strongest evidence for or against the practical importance of absorption for real-world interpretability applications

---

## Phase 5: Final Proposal

### Title
**Beyond Spelling: A Controlled Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders**

### Hypothesis
Feature absorption occurs at statistically significant rates (p < 0.01, permutation test with Bonferroni correction) across at least 3 of 4 semantically distinct hierarchy types (first-letter spelling, entity-type, geographic, grammatical) in Gemma 2 2B Gemma Scope SAEs, and absorption severity within and across hierarchies is predicted by the decoder cosine similarity between parent-child feature pairs (Spearman rho > 0.4 at the feature-pair level).

### Falsification Criterion
If absorption rates on ALL non-spelling hierarchies are indistinguishable from shuffled-label permutation null (permutation test p > 0.05 after Bonferroni correction for 4 hierarchy types x 2 SAE widths = 8 comparisons), the hypothesis is rejected. This would mean absorption, as currently measured, is specific to the artificial first-letter spelling task and does NOT generalize to the semantic hierarchies that matter for safety and interpretability applications.

### Method
Training-free analysis on pre-trained Gemma Scope SAEs (Gemma 2 2B) and GPT-2 Small SAEs (SAELens). No new SAE training required.

### Evaluation Protocol

**Primary benchmarks:**
- Chanin et al. absorption metric (extended SAEBench version) on Gemma Scope SAEs
- RAVEL entity-attribute dataset for entity-type and geographic hierarchies
- Penn Treebank POS-tagged data for grammatical hierarchy
- sae-spelling first-letter task as baseline

**Metrics with statistical test plan:**
- Absorption rate per hierarchy type (point estimate + 95% bootstrap CI, 1000 resamples)
- Permutation test against shuffled-label null (100 permutations per hierarchy, report p-value)
- Bonferroni correction for 8 primary comparisons (4 hierarchies x 2 widths)
- Spearman rank correlation between ASI and absorption at feature-pair level (with bootstrap CI)
- Wilcoxon rank-sum test for decoder angle difference between absorbed and non-absorbed pairs
- Cohen's d effect size for all comparisons

**Number of random seeds:** Not applicable (training-free analysis). However, we address stochasticity via:
- 3 random train/test splits for probe training (report variance)
- Bootstrap resampling (1000 iterations) for all reported statistics
- 100 permutations for each null distribution

### Ablation Schedule

| Ablation | What it tests | Expected outcome |
|----------|--------------|-----------------|
| Remove frequency-ratio from ASI | Whether cosine similarity alone predicts absorption | AUROC drops by 0.05-0.10 (frequency adds modest predictive value) |
| Remove cosine similarity from ASI | Whether frequency ratio alone suffices | AUROC drops substantially (cosine similarity is the primary driver) |
| Vary probe F1 threshold (0.70-0.95) | Sensitivity of absorption estimates to probe quality | Absorption rates stable within 5% absolute for F1 > 0.80 |
| Vary k in k-sparse probing (1-10) | Whether choice of k affects which pairs appear absorbed | Absorption rate monotonically decreasing with k (more splits = less absorption per split) |
| Fix L0, vary width (16k vs 65k) | Width dependence of absorption independent of sparsity | Wider SAEs show modestly higher absorption at matched L0 (more capacity for specific features) |
| Fix width, vary L0 (multiple Gemma Scope settings) | Sparsity dependence of absorption independent of width | Higher sparsity (lower L0) increases absorption (confirmed replication of Chanin et al.) |
| Layer sweep (6, 12, 18) for entity-type | Whether layer 12 is representative | Absorption peaks at middle layers where entity representations are strongest |
| Shuffled-label null for every comparison | False-positive rate of the absorption metric | Null absorption rate < 5% (the metric is specific, not just sensitive to noise) |

### Control Experiments

1. **Shuffled-label null**: For every hierarchy type, shuffle the parent-child labels (e.g., randomly assign cities to countries) and re-run the full absorption pipeline. This establishes the false-positive rate of the metric itself. If the null rate is > 10%, the metric is too noisy for cross-domain claims.

2. **Random-SAE baseline**: Following Korznikov et al. (2026), measure "absorption" using a randomly initialized (untrained) SAE. If absorption rates from random SAEs are comparable to trained SAEs, the phenomenon is a statistical artifact of the metric, not a property of learned features.

3. **Dense probe comparison**: For each hierarchy, train a dense logistic regression probe on model activations (not SAE features). Compare probe accuracy against best SAE probe accuracy. Report the gap. This contextualizes absorption within the broader SAE vs. probe debate (DeepMind negative results).

4. **Dead feature control**: Report the fraction of dead features in each SAE and verify that low absorption is not confounded by high dead-feature ratios (the ReLU artifact found by SAEBench). Exclude SAEs with > 20% dead features from primary analysis.

5. **Entity frequency verification**: For RAVEL entities, count occurrences in a 100M-token sample of the SAE training corpus. Exclude entities appearing < 10 times. Report the number of valid entities per hierarchy after filtering.

### Pilot Design
Load GPT-2 Small SAE (layer 6, width 24576 from SAELens). Train first-letter logistic regression probe (F1 check). Measure absorption rate on 5 letters using the Chanin et al. metric. Run one shuffled-label null (permute letter labels). Verify that real absorption rate > null absorption rate. Compute ASI for top-100 feature pairs by cosine similarity. Estimated time: 12-15 minutes. Success criteria: absorption rate > 10% for at least 3 letters, null rate < 5%, pipeline runs without error.

### Resource Estimate
- **GPU:** Single A100 (40GB) or equivalent. Gemma 2 2B + SAE fits in ~12GB.
- **GPU-hours:** ~7-9 hours total.
  - Pilot: 15 min
  - Probe training (4 hierarchies x 3 splits): 45 min
  - First-letter absorption (baseline replication): 30 min
  - Entity-type absorption: 60 min
  - Geographic absorption: 45 min
  - Grammatical absorption: 45 min
  - Null baselines (shuffled labels, 4 hierarchies x 100 permutations): 120 min
  - Random-SAE control: 30 min
  - ASI computation and validation: 45 min
  - Layer sweep (entity-type, 3 layers): 45 min
  - Sparsity sweep (width/L0 analysis): 60 min
- **Models:** Gemma 2 2B (HuggingFace), GPT-2 Small (TransformerLens)
- **SAEs:** Gemma Scope pre-trained (16k, 65k; layer 12 primary; layers 6, 18 for sweep), GPT-2 SAEs from SAELens
- **Datasets:** RAVEL (hij/ravel), sae-spelling first-letter data, OpenWebText subset (frequency computation), Penn Treebank POS-tagged data
- **Code:** sae-spelling (MIT), SAELens (MIT), SAEBench (Apache 2.0)
- **Storage:** ~20GB (model weights + SAE weights + cached activations)
- **Time budget per task:** Target <= 1 hour per individual experiment (consistent with project constraints)

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Entity-attribute probes fail F1 >= 0.80 gate | HIGH | MEDIUM | Screen 8 candidate hierarchies in pilot; require only 3 of 4 to pass; use RAVEL's well-validated city attributes as highest-confidence non-spelling hierarchy |
| Absorption metric has high false-positive rate on non-spelling tasks | HIGH | LOW-MEDIUM | Shuffled-label null explicitly tests this; if null rate > 10%, report as a finding about metric limitations |
| Cross-domain absorption indistinguishable from null | MEDIUM | MEDIUM | Report as informative null result: "absorption is task-specific to artificial hierarchies." This is equally publishable and has important implications (all mitigation papers evaluated only on spelling) |
| Feature inconsistency across SAE seeds confounds results | MEDIUM | LOW | Use Gemma Scope SAEs which are single canonical releases; note limitation; if SAE seeds are available, add multi-seed analysis |
| ASI AUROC < 0.65 | MEDIUM | MEDIUM | Drop ASI from primary claims; retain as supplementary. Focus paper on cross-domain characterization + controls |
| Absorption metric thresholds (0.025, 1.0) interact with hierarchy type | HIGH | MEDIUM | Sweep thresholds on spelling task; use the sweep to calibrate thresholds before applying to other hierarchies; report threshold sensitivity |
| RAVEL entities underrepresented in SAE training corpus | MEDIUM | LOW | Pre-verify entity frequencies; use only entities with >= 10 corpus occurrences; report coverage statistics |
| Reviewer demands new SAE training | LOW | MEDIUM | Frame entire paper as training-free analysis of existing SAEs; argue this is a feature (reproducibility, accessibility) not a limitation |

### Novelty Claim

The primary experimental contribution is the **first systematic cross-domain characterization of feature absorption** with **rigorous null controls** that have never been applied to absorption measurement. Specifically:

1. **Cross-domain generalization**: No prior work has measured absorption on any hierarchy other than first-letter spelling. We test 4 hierarchy types, establishing whether the 15-35% absorption rate generalizes.

2. **Statistical rigor**: No prior absorption measurement includes shuffled-label permutation nulls, bootstrap confidence intervals, multiple-comparisons correction, or random-SAE baselines. Our controls close a major methodological gap.

3. **Absorption Susceptibility Index**: If validated, ASI is the first probe-free metric for predicting absorption risk, enabling broad-spectrum absorption surveys without pre-specified feature hierarchies.

4. **Hierarchy property predictors**: We test which measurable properties (decoder cosine similarity, frequency ratio, hierarchy depth) predict absorption severity, providing the first empirical basis for a quantitative absorption theory.

The novelty is not in the tools (sae-spelling, SAELens, Gemma Scope are all existing) but in the experimental design: asking the right questions with the right controls on the right range of domains.

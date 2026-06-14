# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **[SAEBench](https://arxiv.org/abs/2503.09532) (Karvonen et al., ICML 2025)** — Standardized benchmark with 8 metrics including absorption. Evaluates 200+ SAEs across 7 architectures with deterministic, reproducible metrics. Key methodological insight: single metrics miss critical tradeoffs; Matryoshka SAEs appeared worse on proxy metrics but excelled at feature disentanglement. **Lesson**: comprehensive multi-metric evaluation is essential.

2. **[SynthSAEBench](https://arxiv.org/abs/2602.14687) (Chanin et al., 2026)** — Large-scale synthetic benchmark with ground-truth features (16K features, 768 dims). Enables controlled experiments with known parent-child dependencies. Critical finding: Matching Pursuit SAEs can exploit superposition noise to improve reconstruction without learning true features — a failure mode only detectable with ground-truth access. **Lesson**: ground-truth validation is the gold standard for metric validation.

3. **["Sanity Checks for Sparse Autoencoders"](https://arxiv.org/abs/2602.14111) (Korznikov et al., 2026)** — Frozen/random SAE baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). Uses Bonferroni and BH-FDR corrections. Shows SAEs recover only 9% of true features despite 71% explained variance. **Lesson**: random baselines are mandatory; reconstruction quality does not imply meaningful feature learning.

4. **["A is for Absorption"](https://arxiv.org/abs/2409.14507) (Chanin et al., 2024/2025)** — Foundational work defining absorption metric. Uses conservative estimation: only counts absorption if a single latent has much larger ablation effect than all others, and main latents do not activate at all. Acknowledges metric likely underestimates true absorption. **Lesson**: the absorption metric is conservative and may miss complex absorption patterns.

5. **["The Reproducibility Problem in Mechanistic Interpretability Studies"](https://www.researchgate.net/publication/392626589) (Rowe & Kemp, 2025)** — Documents growing reproducibility crisis: subjective choices, sensitive hyperparameters, opaque tools, cross-lab replication failure. Proposes open-source standards, benchmark tasks, transparent reporting. **Lesson**: MI claims risk becoming anecdotal without rigorous reproducibility standards.

6. **["The Story is Not the Science"](https://arxiv.org/abs/2602.18458) (2026)** — MechEvalAgent finds 60% execution failure rate, 27.3% narrative-implementation inconsistency, 23.3% plan-implementation mismatch. Example: claimed logit correlation 0.66, replication found 0.72 (evaluated on first 20 examples without disclosure). **Lesson**: execution-grounded evaluation is essential; narrative claims are insufficient.

7. **[CE-Bench](https://aclanthology.org/2025.blackboxnlp-1.1/) (EMNLP-W 2025)** — LLM-free contrastive evaluation; 77.3% CRPR alignment with SAEBench. Built on curated contrastive story pairs. **Lesson**: lightweight, no-judge-needed evaluation is possible and correlates with comprehensive benchmarks.

8. **["Does Higher Interpretability Imply Better Utility?"](https://arxiv.org/abs/2510.03659) (Wang et al., ICLR 2026)** — Weak correlation (tau_b ~ 0.3) between interpretability and steering utility. Proposes Delta Token Confidence selection. **Lesson**: interpretability metrics do not proxy for practical utility; both must be measured independently.

9. **["Analysis of Variational Sparse Autoencoders"](https://www.lesswrong.com/posts/nPxdHt8k4bMuaC46M/analysis-of-variational-sparse-autoencoders) (LessWrong, 2025)** — Explicitly computes decoder weight cosine similarity matrices. At high sparsity (k=512), both vSAE and standard SAE achieve good orthogonality; at lower sparsity (k=64), vSAE shows significantly more feature correlation (0.0058 vs 0.0001). **Lesson**: decoder correlation structure varies systematically with sparsity level.

10. **[MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151) (2025)** — Standardized baselines for comparing MI methods. Full Vector baseline fails on RAVEL (~50% IIA, chance level); DAS partially succeeds in mid-layers. Two-digit addition: all baselines at chance on Gemma-2. **Lesson**: strong baselines expose where methods fail; weak baselines inflate apparent success.

11. **["Deriving Decoder-Free Sparse Autoencoders from First Principles"](https://arxiv.org/abs/2601.06478) (2026)** — Features emerge through competition alone; decorrelation penalties prevent redundancy. Eliminates decoder entirely; uses W^T as pseudo-decoder. **Lesson**: decoder geometry is not fundamental to feature competition; the competitive dynamics exist independently of decoder structure.

12. **["Interpretability as Compression"](https://arxiv.org/abs/2410.11179) (Schaeffer et al., 2024)** — Frames interpretability through rate-distortion-perception tradeoff. Minimal description length yields more interpretable features. **Lesson**: absorption reduces "perception" quality (parent features invisible) even when distortion (reconstruction) is low.

### Experimental Landscape

**What has been properly tested:**
- Absorption exists and is prevalent across hundreds of SAEs (Chanin et al., 2024)
- Absorption rate varies across architectures (SAEBench: 200+ SAEs, 7 architectures)
- Random SAE baselines match trained SAEs on standard metrics (Korznikov et al., 2026)
- Weak interpretability-utility correlation (Wang et al., ICLR 2026)

**What is accepted without proper evidence:**
- That decoder correlations reflect "competitive suppression" (no causal evidence; correlation only)
- That the LCA-SAE correspondence explains absorption empirically (mathematically exact for tied weights, but untested for untied weights on real LLMs)
- That precision-recall asymmetry is a deep property (only tested at k>=5; post-hoc analysis)
- That homeostatic rebalancing restores parent firing (not yet tested)

**Where methodological gaps exist:**
- No causal evidence that decoder correlations CAUSE absorption (only predictive correlation claimed)
- No random baseline comparison for inhibition graph structure
- No control for feature frequency when testing correlation-absorption relationship
- No pre-registered analysis plan for precision-recall decomposition
- No power analysis for sample size (26 first-letter features)

---

## Phase 2: Initial Candidates

### Candidate A: "The Local Inhibition Graph: A Rigorously Controlled Test of Decoder Correlation Predictions"

**Core hypothesis**: Edges in the local inhibition graph (top-k correlated neighbors per latent) predict known absorption pairs with precision significantly above chance, AND this prediction holds after controlling for confounds.

**Falsification criterion**: If precision@20 <= 0.05 after controlling for feature frequency and semantic similarity, the structural correspondence fails.

**Evaluation protocol**:
- Primary: Precision@k, recall@k, Fisher exact test for enrichment vs. random baseline
- Controls: Feature frequency matching, semantic similarity matching, random SAE baseline
- Statistical: Bootstrap 95% CI for precision@k, permutation test for enrichment significance
- Multiple comparison: Bonferroni correction across k values {5, 10, 20, 50}

**Ablation plan**:
1. Remove frequency-matched control: does precision drop?
2. Remove semantic-similarity control: does precision drop?
3. Test on random SAE: does precision drop to chance?
4. Test with different k values: is prediction robust?

**Confounders identified**:
- Feature frequency: high-frequency features may have more correlated neighbors simply due to more data
- Semantic similarity: parent-child features are semantically similar; decoder correlation may reflect semantic similarity, not competitive suppression
- Dictionary overcompleteness: any overcomplete dictionary has correlated directions by geometry alone
- Tied vs. untied weights: LCA correspondence is exact only for tied weights

**Pilot design**: Compute decoder correlation matrix for GPT-2 Small SAE (24K latents). For 26 first-letter features, compute precision@20 against Chanin absorption pairs. Control for frequency and semantic similarity. Target: ~15 min.

---

### Candidate B: "Does Decoder Correlation Predict Absorption, or Just Semantic Similarity? A Controlled Experiment"

**Core hypothesis**: Decoder correlations predict absorption pairs ONLY because they proxy for semantic similarity, not because they capture competitive suppression. A model using semantic similarity alone achieves comparable precision to the inhibition graph.

**Falsification criterion**: If a semantic-similarity-only model achieves precision@20 within 50% of the inhibition graph, the "competitive suppression" mechanism is not uniquely supported.

**Evaluation protocol**:
- Construct semantic similarity baseline: use word embeddings (GloVe/fastText) to compute similarity between feature concepts (e.g., "starts with S" vs. "starts with SH")
- Compare inhibition graph precision@k vs. semantic similarity precision@k
- Test whether adding decoder correlation to semantic similarity improves prediction (incremental validity)
- Use paired permutation test to compare models

**Ablation plan**:
1. Semantic similarity only
2. Decoder correlation only
3. Both combined (linear model)
4. Both combined (nonlinear model)

**Confounders identified**:
- Word embedding quality: semantic similarity depends on embedding choice
- Feature labeling: first-letter features have clear labels; semantic features may not
- Hierarchical structure: parent-child semantic similarity is inherently high

**Pilot design**: Compute semantic similarity for 26 first-letter features using GloVe embeddings. Compare precision@20 with inhibition graph. Target: ~15 min.

---

### Candidate C: "A Controlled Experiment Nobody Has Run: Random Baseline Comparison for Inhibition Graph Structure"

**Core hypothesis**: The inhibition graph's ability to predict absorption is a learned property of trained SAEs, not a structural artifact of overcomplete dictionaries. A random SAE (frozen random decoder weights) shows significantly lower precision@k.

**Falsification criterion**: If a random SAE achieves precision@20 within 2x of the trained SAE, the graph structure is an artifact of dictionary size, not learned competitive dynamics.

**Evaluation protocol**:
- Construct random SAE baseline following Korznikov et al. (2026): frozen decoder weights from random initialization
- Compute inhibition graph for random SAE
- Measure precision@k on same first-letter features
- Compare trained vs. random using paired permutation test
- Test multiple random seeds (n=5) for stability

**Ablation plan**:
1. Random decoder, random encoder
2. Random decoder, trained encoder
3. Trained decoder, random encoder
4. Full trained SAE

**Confounders identified**:
- Random initialization distribution: different init schemes may produce different correlation structures
- Normalization: trained SAEs may have different weight norms than random weights
- Feature detection: random SAEs may not have detectable first-letter features at all

**Pilot design**: Construct one random SAE baseline for GPT-2 Small. Compute inhibition graph and precision@20. Compare to trained SAE. Target: ~20 min.

---

## Phase 3: Self-Critique

### Against Candidate A (Rigorous Inhibition Graph Test)

**Confound attack**: The project's current design tests precision@k but does not control for feature frequency. High-frequency features (like "starts with S") have more training examples and may produce more stable decoder directions, which could artifactually increase correlation with other high-frequency features. If absorption also correlates with frequency (it does: common letters like S, T have more child features), then frequency is a confound.

**Statistical attack**: With 26 first-letter features and precision@20, the sample size is small. A precision of 0.10 with 20 neighbors means ~2 true positives per feature. With 26 features, that's ~52 predictions total. The confidence interval will be wide. Is this enough to detect a meaningful effect? Power analysis: to detect precision@20 = 0.10 vs. chance = 0.00083 with 80% power at alpha = 0.05, we need... actually, with 26 features and the extreme enrichment predicted (120x), power is likely adequate. But if the true precision is lower (e.g., 0.03 = 36x enrichment), power may be insufficient.

**Benchmark attack**: First-letter features are a narrow benchmark. They are artificially constructed (human-defined categories) and may not represent natural feature hierarchies. The Chanin metric itself is conservative and may underestimate absorption. If the ground truth is incomplete, precision estimates are biased downward.

**Ablation completeness attack**: The ablation plan controls for frequency and semantic similarity, but does not control for decoder norm. Features with larger decoder norms may have different correlation structures. Also, the plan does not test whether precision varies with absorption strength — perhaps the graph only predicts strong absorption pairs.

**Verdict**: STRONG — but only if confound controls are implemented rigorously. Without them, the result is uninterpretable.

### Against Candidate B (Semantic Similarity Control)

**Confound attack**: Semantic similarity is itself confounded by hierarchical structure. Parent-child features ("starts with S" and "starts with SH") are semantically similar BY DEFINITION. The semantic similarity baseline may achieve high precision simply because it captures the hierarchical structure that defines absorption pairs. This does not distinguish "semantic similarity" from "competitive suppression" — both predict the same pairs.

**Statistical attack**: The incremental validity test (does decoder correlation add predictive power beyond semantic similarity?) is the right approach, but the effect size may be small. If decoder correlation and semantic similarity are highly correlated (they likely are, since hierarchical features have correlated decoder directions), the incremental contribution may be undetectable with 26 features.

**Benchmark attack**: Word embeddings may not capture the specific hierarchical relationships in first-letter features. "Starts with S" and "starts with SH" may not be close in embedding space if the embeddings were trained on word-level co-occurrence, not letter-level patterns.

**Ablation completeness attack**: The ablation does not test whether semantic similarity and decoder correlation are redundant (multicollinearity). If they are, the combined model may not improve over either alone, even if both capture true signal.

**Verdict**: MODERATE — the confound structure makes this difficult to interpret. The test may be inconclusive rather than falsifying.

### Against Candidate C (Random Baseline Comparison)

**Confound attack**: Random SAEs may not have detectable first-letter features at all, making the comparison moot. If we cannot identify "parent" features in a random SAE, we cannot measure precision@k. The comparison may need to use synthetic data where ground-truth features are known regardless of SAE training.

**Statistical attack**: The comparison between trained and random SAE is a between-subjects comparison (different SAEs), not a within-subjects comparison (same features). Variability across random initializations may swamp the trained-vs-random difference. Need multiple random seeds (n>=5) for stable estimates.

**Benchmark attack**: Korznikov et al. (2026) showed random SAEs match trained SAEs on standard metrics. If random SAEs also match on inhibition graph precision, this is strong evidence that the graph is an artifact. But if they differ, it only shows that the graph is not purely artifactual — it does not prove the "competitive suppression" mechanism.

**Ablation completeness attack**: The ablation tests decoder randomization and encoder randomization separately, but does not test whether the interaction matters. Also, it does not test partial randomization (e.g., shuffling decoder directions while preserving norms).

**Verdict**: STRONG — this is the most important control experiment. It directly addresses the "Sanity Checks" challenge. Even if inconclusive for mechanism, it is essential for credibility.

---

## Phase 4: Refinement

### Dropped
- **Candidate B (Semantic Similarity Control)**: The confound structure makes it difficult to interpret. Semantic similarity and decoder correlation are likely too correlated to disentangle with 26 features. The test risks being inconclusive rather than falsifying.

### Strengthened
- **Candidate A (Rigorous Inhibition Graph Test)**: Strengthened by:
  1. Adding frequency-matched control: compare precision for features matched on frequency quartile
  2. Adding decoder-norm control: test whether precision varies with decoder norm
  3. Adding absorption-strength stratification: test whether graph predicts strong vs. weak absorption differently
  4. Pre-registering analysis plan: specify all tests, thresholds, and correction methods BEFORE running experiments
  5. Power analysis: compute minimum detectable effect size given n=26 features

- **Candidate C (Random Baseline)**: Strengthened by:
  1. Multiple random seeds (n=5) for stability
  2. Both full-random and partial-random (shuffled directions, preserved norms) baselines
  3. Synthetic data validation: test on SynthSAEBench where ground truth is known for random SAEs
  4. Comparison of graph statistics (not just precision): mean edge weight, clustering coefficient, degree distribution

### Additional Controls
- **Feature frequency residualization**: Regress out log(feature frequency) from both decoder correlation and absorption rate before testing correlation
- **Layer-matched comparison**: Compare graph statistics across layers while controlling for feature frequency distribution
- **Cross-validation**: Split first-letter features into train/test (e.g., A-M vs. N-Z) for prediction validation
- **Effect size reporting**: Report Cohen's d or Pearson r with 95% CI, not just p-values

### Selected Front-Runner
**Candidate A with integrated Candidate C controls**: The primary contribution is testing whether the inhibition graph predicts absorption with rigorous confound control. The random baseline comparison (Candidate C) is integrated as a mandatory control experiment, not a separate candidate.

---

## Phase 5: Final Proposal

### Title
"Decoder Correlations Predict Feature Absorption: A Controlled Validation of the Local Inhibition Graph Hypothesis"

Alternative: "The Local Inhibition Graph Under Scrutiny: Controlled Evidence for Decoder Correlation-Based Absorption Prediction"

### Hypothesis

**H1 (Primary)**: For a pretrained SAE, edges in the local inhibition graph (top-k correlated neighbors per latent) predict known absorption pairs with precision significantly above chance, after controlling for feature frequency, decoder norm, and semantic similarity.

**Formalization**:
- Precision@20 >= 0.10 (vs. ~0.00083 chance = 20/24000 for GPT-2 Small 24K latents)
- Enrichment factor >= 10x after frequency residualization
- Fisher exact test p < 0.001 for enrichment vs. random baseline

**Falsification criterion**: If precision@20 <= 0.05 after confound control, OR if random SAE achieves precision within 2x of trained SAE, the hypothesis fails.

### Method

#### Phase 1: Construct Local Inhibition Graph with Controls

For each latent i in SAE decoder matrix W_dec:
1. Compute normalized decoder correlations: C_ij = <W_dec[i], W_dec[j]> / (||W_dec[i]|| * ||W_dec[j]||)
2. Keep top-k neighbors per latent (k in {5, 10, 20, 50}) with highest |C_ij|
3. Edge weight = C_ij (signed correlation)

#### Phase 2: Validate Against Absorption Pairs with Confound Control

**Ground truth**: Chanin et al. absorption detection on first-letter features (A-Z)

**Primary test**:
- For each absorption pair (parent i, absorbing j), check if j is in N_k(i)
- Compute precision@k, recall@k
- Bootstrap 95% CI for precision@k (1000 resamples)

**Confound controls**:
1. **Frequency control**: Stratify features by frequency quartile. Test precision within each quartile. Predict: precision does not vary significantly across quartiles.
2. **Decoder norm control**: Regress out log(decoder norm) from correlation scores. Test whether residualized correlations still predict absorption. Predict: residualized precision >= 0.05.
3. **Semantic similarity control**: Compute word embedding similarity for feature labels. Test whether decoder correlation adds predictive power beyond semantic similarity using incremental validity test.

**Statistical tests**:
- Fisher exact test for enrichment vs. chance
- Permutation test (1000 permutations) for enrichment vs. shuffled labels
- Paired permutation test comparing trained vs. random SAE
- Bonferroni correction across k values (4 tests -> alpha = 0.0125)

#### Phase 3: Random SAE Baseline Comparison

Following Korznikov et al. (2026):
1. Construct random SAE with frozen decoder weights (5 random seeds)
2. Compute inhibition graph for each random SAE
3. Measure precision@k on same first-letter features
4. Compare trained vs. random using paired permutation test
5. Test graph statistics: mean edge weight, clustering coefficient, degree distribution

**Prediction**: Trained SAE precision@20 >= 5x random SAE precision@20

#### Phase 4: Precision-Recall Asymmetry Test

- Compute total incoming inhibition I_i = sum_{j in N_k(i)} |C_ij| for each feature
- Test correlation: I_i vs. recall loss (predicted: r < -0.3)
- Test correlation: I_i vs. precision (predicted: |r| < 0.1, n.s.)
- Control for feature frequency in both correlations

#### Phase 5: Layer-Dependent Analysis

- Construct graphs for layers 0, 4, 8, 10 of GPT-2 Small
- Compare graph statistics controlling for feature frequency distribution
- Test whether layer 8 (where H1b was significant) has stronger inhibition structure

### Experimental Plan

| Experiment | Model | SAE | Metrics | Time | Controls |
|---|---|---|---|---|---|
| E1: Graph construction + validation | GPT-2 Small | gpt2-small-res-jb (24K) | Precision@k, recall@k, Fisher test | ~15 min | Frequency, norm, semantic |
| E2: Confound control analysis | GPT-2 Small | Same | Precision by quartile, residualized precision, incremental validity | ~20 min | Frequency, norm, semantic |
| E3: Random SAE baseline | GPT-2 Small | Random decoder (5 seeds) | Precision@k, graph stats | ~30 min | Multiple seeds |
| E4: Precision-recall asymmetry | GPT-2 Small | Same | Correlation (inhibition vs recall, precision) | ~15 min | Frequency residualized |
| E5: Layer-dependent structure | GPT-2 Small | Same (layers 0/4/8/10) | Graph stats by layer | ~20 min | Frequency matched |
| E6: Cross-validation | GPT-2 Small | Same | Train/test split (A-M vs. N-Z) | ~10 min | Out-of-sample prediction |

**Total estimated time**: ~2 GPU-hours

### Baselines

1. **Random graph baseline**: Shuffle latent indices; expected precision@20 ~ 0.00083
2. **Random SAE baseline**: Frozen random decoder weights (5 seeds); tests whether graph structure is learned
3. **Frequency-matched baseline**: Predict absorption using feature frequency alone; tests whether correlation adds value
4. **Semantic similarity baseline**: Predict absorption using word embedding similarity; tests whether correlation captures more than semantics
5. **Non-absorbed pair control**: Test graph edges for correlated but non-absorbed pairs; predicted lower enrichment

### Statistical Analysis Plan (Pre-registered)

**Primary analysis**:
- Precision@20 with 95% bootstrap CI
- Fisher exact test for enrichment vs. chance
- Bonferroni correction across k = {5, 10, 20, 50}

**Secondary analyses**:
- Correlation between total incoming inhibition and absorption rate (Pearson r with 95% CI)
- Correlation between inhibition and recall loss (frequency-residualized)
- Layer-dependent graph statistics (ANOVA with frequency as covariate)

**Random baseline comparison**:
- Paired permutation test (trained vs. random precision@k)
- Effect size: Cohen's d

**Cross-validation**:
- Train prediction model on A-M, test on N-Z
- Report out-of-sample precision@k

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| Precision@k is only marginally above chance after confound control | Medium | High | Report exact precision with CI; if precision@20 > 0.02 (24x chance), still publishable as "weak but significant prediction" |
| Random SAE shows similar graph structure | Medium | High | This is itself a major finding. Paper becomes "Inhibition Graph Structure is a Dictionary Artifact, Not Learned Competition" |
| Feature frequency fully explains absorption prediction | Low | High | Report frequency-residualized precision; if still significant, frequency is not the sole explanation |
| 26 features insufficient for statistical power | Medium | Medium | Power analysis shows detectable effect size; if underpowered, frame as pilot and call for larger study |
| Semantic similarity and decoder correlation are perfectly collinear | Medium | Medium | Report variance inflation factor; if VIF > 10, models are not separable |

### Resource Estimate

| Item | Estimate |
|------|----------|
| GPU | Single 24GB GPU |
| Graph construction | ~15 min per SAE |
| Validation + controls | ~30 min per SAE |
| Random SAE baselines (5 seeds) | ~30 min |
| Cross-validation | ~10 min |
| Total GPU time | ~2 hours |
| Model sizes | GPT-2 Small (primary) |

### Novelty Claim

The experimental contribution is: **the first rigorous, controlled test of whether decoder correlations predict feature absorption**. While the inhibition graph has been proposed (in this project's synthesis), no prior work has:
1. Controlled for feature frequency when testing decoder correlation predictions
2. Compared trained vs. random SAE inhibition graphs
3. Tested incremental validity of decoder correlation beyond semantic similarity
4. Pre-registered a statistical analysis plan for absorption prediction

The novelty is methodological rigor, not just the prediction itself. The field needs controlled evidence, not just theoretical proposals.

### Integration with Project Context

The existing project data (H5: precision invariant, recall variable; H1b: delta-corrected correlation at layer 8) provides the context for this study. The empiricist perspective does not propose a new direction but insists that the existing front-runner (Local Inhibition Graph) be tested with the rigor it deserves:

- **Random baselines are mandatory** (addressing Korznikov et al., 2026)
- **Confound controls are essential** (addressing frequency/semantic confounds)
- **Pre-registration prevents p-hacking** (addressing the reproducibility crisis)
- **Effect sizes with CIs, not just p-values** (addressing statistical best practices)

If the inhibition graph survives these controls, it becomes a genuinely credible contribution. If it fails, the project pivots to descriptive analysis with honest reporting of null results.

---

## Sources

- [SAEBench](https://arxiv.org/abs/2503.09532) - Karvonen et al., ICML 2025
- [SynthSAEBench](https://arxiv.org/abs/2602.14687) - Chanin et al., 2026
- [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111) - Korznikov et al., 2026
- [A is for Absorption](https://arxiv.org/abs/2409.14507) - Chanin et al., 2024/2025
- [The Reproducibility Problem in MI](https://www.researchgate.net/publication/392626589) - Rowe & Kemp, 2025
- [The Story is Not the Science](https://arxiv.org/abs/2602.18458) - 2026
- [CE-Bench](https://aclanthology.org/2025.blackboxnlp-1.1/) - EMNLP-W 2025
- [Does Higher Interpretability Imply Better Utility?](https://arxiv.org/abs/2510.03659) - Wang et al., ICLR 2026
- [Analysis of Variational SAEs](https://www.lesswrong.com/posts/nPxdHt8k4bMuaC46M/analysis-of-variational-sparse-autoencoders) - LessWrong, 2025
- [MIB Benchmark](https://arxiv.org/abs/2504.13151) - 2025
- [Decoder-Free SAEs](https://arxiv.org/abs/2601.06478) - 2026
- [Interpretability as Compression](https://arxiv.org/abs/2410.11179) - Schaeffer et al., 2024

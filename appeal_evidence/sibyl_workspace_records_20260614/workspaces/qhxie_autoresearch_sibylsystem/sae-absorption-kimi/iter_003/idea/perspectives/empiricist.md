# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024)** — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" (NeurIPS 2025). Introduced the absorption detection metric using first-letter classification with ground-truth logistic probes. Key methodological insight: absorption is detected via k-sparse probing (k=1..10) with feature-splitting threshold tau_fs = 0.03. Limitation: toy-model focus (first-letter tasks only); authors explicitly call for "finding examples of feature absorption unrelated to character identification." **Critical for this perspective**: The metric has been replicated in controlled synthetic settings (SynthSAEBench, 2026) but never validated on real semantic hierarchies.

2. **Karvonen et al. (2025)** — "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." Standardized the Chanin metric into an 8-evaluation benchmark. Adapted technically by replacing ablation with probe-projection criteria (tau_pa = 0, tau_ps = -1), enabling all-layer evaluation. However, the underlying task remains first-letter classification. Absorption eval is expensive (~26 min per SAE). **Key insight**: "Gains on proxy metrics do not reliably translate to better practical performance" — the sparsity-fidelity frontier does not predict downstream interpretability outcomes.

3. **Chanin & Garriga-Alonso (2026)** — "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." A landmark methodological advance: ground-truth synthetic data with 16,384 features, realistic correlation, hierarchy, superposition, and Zipfian firing. Even when the Linear Representation Hypothesis holds perfectly by construction, current SAE architectures still fail to achieve perfect feature recovery. **Critical finding**: Matching Pursuit SAEs exploit superposition noise to improve reconstruction without learning ground-truth features — demonstrating that reconstruction quality can mask poor feature learning.

4. **CE-Bench (2025)** — "Towards a Reliable Contrastive Evaluation Benchmark of Interpretability of Sparse Autoencoders." Fully LLM-free contrastive evaluation using 5,000 story pairs across 1,000 subjects. Addresses non-determinism and bias in LLM-dependent benchmarks. Achieves 77.30% alignment with SAEBench rankings while being deterministic and reproducible. **Methodological lesson**: Evaluation benchmarks should be validated against ground-truth or alternative methodologies to establish construct validity.

5. **Kantamneni et al. (2025)** — "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing." Critical evaluation showing SAEs do not consistently outperform strong non-SAE baselines on downstream probing tasks. Does not isolate absorption as the cause, but raises the broader validity question: do SAE metrics predict real-world utility?

6. **Bussmann et al. (2025)** — "Learning Multi-Level Features with Matryoshka Sparse Autoencoders" (ICML 2025). Reports absorption rate ~0.05 vs. 0.49 (BatchTopK), but all measurements use the first-letter benchmark. No validation on semantic hierarchies. Inner levels suffer feature hedging (Chanin 2025).

7. **Chanin (2025)** — "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." Identified feature hedging as the opposite failure mode to absorption (reconstruction-loss-driven). Shows Matryoshka SAEs trade absorption for hedging. **Methodological insight**: Failure modes interact — reducing one may increase another. Any valid evaluation must measure multiple failure modes simultaneously.

8. **OrtSAE (Korznikov et al., 2025)** — "Orthogonal Sparse Autoencoders Uncover Atomic Features." Reduced absorption by 65% using chunk-wise orthogonality penalty. Again, first-letter benchmark only. Adds ~4-11% compute overhead.

9. **Unified Theory of SDL (2025)** — "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." Identified polysemanticity, dead neurons, and feature absorption as fundamental structural properties of SDL optimization via piecewise biconvexity analysis. Theoretical focus; limited empirical validation.

10. **Feature Sensitivity (2025)** — "Measuring Sparse Autoencoder Feature Sensitivity." Introduced feature sensitivity as a new evaluation dimension; found many "interpretable" features have poor sensitivity. Complementary to absorption — features can be interpretable-looking but not causally actionable.

11. **Kerl (2025)** — "Evaluation of Sparse Autoencoder-based Refusal Features." Master's thesis documenting that SAE features don't transfer across model variants (base to instruction-tuned), and that dense decoder contributions complicate interpretation. Raises transferability as a validity concern.

12. **Elhage et al. (2022)** — "Toy Models of Superposition." The theoretical foundation: networks encode more features than neurons via linear combinations. This superposition is the root cause of both polysemanticity and the difficulty of feature recovery. Any valid absorption metric must account for superposition geometry.

### Experimental Landscape

**What has been properly tested:**
- First-letter absorption is well-documented across hundreds of SAEs (Chanin et al. 2024, replicated by SynthSAEBench 2026)
- Multiple architectures show reduced first-letter absorption (Matryoshka, OrtSAE, HSAE)
- SAEBench provides a standardized, reproducible protocol for first-letter evaluation
- SynthSAEBench validates that absorption-like phenomena emerge even with ground-truth synthetic hierarchies

**What is accepted without evidence:**
- That first-letter absorption generalizes to semantic hierarchies (NO direct evidence found; this was the target of our iter_002 study, which failed due to metric collapse)
- That lower absorption scores mean better downstream interpretability (Kantamneni et al. 2025 casts doubt; SAEBench itself notes proxy metrics don't predict practical performance)
- That the absorption metric is specific to hierarchical (vs. merely correlated) features
- That architectures optimized for first-letter absorption improve behavior on real-world tasks

**Where methodological gaps exist:**
- No construct-validity study of the absorption metric on real semantic hierarchies (our iter_002 attempt revealed metric collapse, not invalidity)
- No causal validation that absorbed parent features are "missing" vs. "hidden"
- No systematic quantification of the downstream cost of absorption
- Small sample sizes in correlation-based claims without power analysis (iter_002: n=7, CI spanning 1.37 units)
- No validation that reconstruction-quality improvements (Matryoshka, JumpReLU) translate to better feature recovery (SynthSAEBench shows they often don't)

**Hard lessons from iter_002:**
1. Metric collapse: When resid_acc = sae_acc = 1.0, absorption = (1 - k_sparse_acc). The metric measures k-sparse probing difficulty, not SAE encoding loss.
2. Random-SAE confusion: Must verify exactly what is permuted and what behavior is expected BEFORE interpreting results.
3. Underpowered samples: n=7 is uninformative for correlation inference. Power analysis is mandatory.
4. Confounded controls: Multi-class hierarchies vs binary non-hierarchy pairs are structurally inequivalent.
5. Ceiling effects: Perfect probe AUROCs (1.0) are a warning sign, not a success.
6. Weak pilot gates: Checking only ranking order is insufficient — must verify metric validity, control behavior, and probe AUROC range.

---

## Phase 2: Initial Candidates

### Candidate A: Fixed Construct-Validity Study with Methodological Corrections

**Core hypothesis (H1 — Construct Validity):** After fixing the metric collapse and control confounds from iter_002, SAEs with low first-letter absorption rates will exhibit proportionally low k-sparse probing degradation on matched-frequency semantic hierarchies. The Pearson correlation between first-letter and semantic-hierarchy scores across n >= 12 SAEs will exceed 0.5.

**Falsification criterion:** If Pearson r < 0.5 with n >= 12, or if the 95% bootstrap CI includes 0, the hypothesis is falsified. This would confirm that first-letter absorption does not generalize to semantic hierarchies.

**Evaluation protocol:**
- Primary: SAEBench first-letter absorption (established benchmark)
- Custom: Semantic-hierarchy k-sparse probing degradation using binary parent-child pairs from WordNet (frequency-matched)
- Custom: Binary non-hierarchy correlated-feature pairs (structurally equivalent controls)
- Statistics: Pearson r with bootstrap 95% CI (B=10,000); paired t-test for hierarchy vs non-hierarchy; power analysis pre-registered

**Ablation plan:**
- Harder hierarchies where sae_acc < resid_acc (fixes metric collapse)
- Binary parent-child pairs only (fixes structural inequivalence)
- True encoder-permuted Random SAE + true decoder-permuted Random SAE (fixes confusion)
- k-sensitivity: k=5, 10, 20 (fixes arbitrary k)
- tau_fs robustness: 0.01, 0.03, 0.05

**Confounders identified:**
- Frequency imbalance between parent and child tokens (mitigated: frequency-matched synthetic datasets)
- Probe quality variation (mitigated: filter to AUROC in [0.7, 0.95] — not all 1.0)
- Tokenization artifacts (mitigated: single-token concepts only)
- Task difficulty differences (mitigated: binary pairs for both conditions)
- Sample size inadequacy (mitigated: n >= 12 SAEs with power analysis)

**Pilot design:** 3 SAEs (Matryoshka, BatchTopK, Standard) x 4 binary parent-child pairs x 2 control pairs. Target: 15 min. Success criteria (HARD GATES): (1) sae_acc < resid_acc for at least 50% of hierarchies, (2) probe AUROCs in [0.7, 0.95] (not all 1.0), (3) Random-SAE (encoder-permuted) produces near-zero scores, (4) metric does not collapse to single term.

---

### Candidate B: Cross-Architecture Controlled Analysis — Isolating What Actually Reduces Absorption

**Core hypothesis:** The architectural components that most reduce first-letter absorption are (in order): (1) multi-scale reconstruction loss (Matryoshka), (2) explicit sparsity control (TopK/BatchTopK), (3) orthogonality constraints (OrtSAE). Each component's individual contribution can be isolated via controlled ablation.

**Falsification criterion:** If a component ablation shows no significant absorption reduction compared to its baseline (p > 0.05, Bonferroni-corrected), that component's claimed contribution is falsified.

**Evaluation protocol:**
- Use SynthSAEBench-16k synthetic data with ground-truth hierarchies (10,884 hierarchical features, 128 root trees, depth 3)
- Train SAE variants varying only one component at a time:
  - Baseline: Standard ReLU
  - +TopK sparsity (vs L1)
  - +Multi-scale loss (Matryoshka-style)
  - +Orthogonality penalty (OrtSAE-style)
  - +Gated activation
- Measure: ground-truth feature recovery (MCC), absorption rate on known hierarchies, reconstruction quality
- Statistics: ANOVA across architectures; post-hoc Tukey HSD; effect sizes (Cohen's d)

**Ablation plan:**
- Vary dictionary size (2x, 4x, 8x hidden dim)
- Vary sparsity level (L0 = 50, 100, 200)
- Vary hierarchy depth in synthetic data (2, 3, 4 levels)

**Confounders identified:**
- Training dynamics differ across architectures (mitigated: fixed training budget, same initialization)
- Hyperparameter tuning unfairly benefits some architectures (mitigated: report best vs default hyperparameters separately)
- Synthetic data may not match LLM feature structure (mitigated: validate top-performing config on real LLM SAEs)

**Pilot design:** Baseline + 2 variants (TopK, Matryoshka) on SynthSAEBench-16k subset (1,024 features). Target: 20 min on 1 GPU. Success: clear ordering of absorption rates matching literature claims.

---

### Candidate C: Absorption-Downstream Impact Dose-Response — Does Lower Absorption Actually Help?

**Core hypothesis:** SAEs with lower first-letter absorption rates show better performance on downstream interpretability tasks (sparse probing, steering, circuit discovery) in a dose-response relationship. The correlation between absorption rate and downstream performance will be negative and significant (r < -0.4).

**Falsification criterion:** If correlation between absorption rate and downstream performance is non-significant (p > 0.05) or positive, the hypothesis is falsified. This would mean absorption rate is not a useful proxy for real-world interpretability.

**Evaluation protocol:**
- Select 8-10 SAEs spanning the absorption spectrum (from Matryoshka to Standard ReLU)
- Measure first-letter absorption rate (SAEBench)
- Measure downstream tasks:
  1. Sparse probing accuracy on 5 diverse concept categories (from SAEBench)
  2. Steering efficacy: can we reliably steer model behavior using top-k latents?
  3. Circuit discovery: precision/recall of automated circuit extraction
- Statistics: Pearson r (absorption vs each downstream metric); multiple regression controlling for L0 and explained variance; Bonferroni correction across 3 downstream tests

**Ablation plan:**
- Control for L0 sparsity (absorption and sparsity are confounded)
- Control for explained variance (reconstruction quality may drive downstream performance, not absorption)
- Control for dictionary size

**Confounders identified:**
- Absorption is confounded with sparsity (lower L0 often means lower absorption)
- Downstream tasks may depend on reconstruction quality, not absorption specifically
- Some architectures may be better at some downstream tasks for unrelated reasons

**Pilot design:** 4 SAEs x 2 downstream tasks (sparse probing + steering). Target: 30 min. Success: negative correlation visible in pilot data.

---

## Phase 3: Self-Critique

### Against Candidate A (Fixed Construct Validity)

- **Confound attack:** Even with binary pairs and harder hierarchies, the semantic-hierarchy task may still be too easy for modern SAEs. If sae_acc remains near resid_acc, the metric still collapses. The fix (harder hierarchies) is speculative — we don't know if any natural semantic hierarchies produce sae_acc < resid_acc.
- **Statistical attack:** n=12 gives power ~0.65 to detect r=0.5 at alpha=0.05. Still underpowered by conventional standards (power >= 0.8). The pre-registered threshold (r > 0.5) may be too ambitious.
- **Benchmark attack:** The first-letter benchmark itself may be flawed in ways that don't generalize. Even if we fix our semantic-hierarchy adaptation, we're validating against a potentially flawed standard.
- **Ablation completeness attack:** The k-sensitivity and tau_fs analyses add multiple comparisons. With 12 SAEs x 3 k values x 3 tau_fs values, we're running 108 tests — Bonferroni correction would require p < 0.0005 for significance.
- **Verdict:** MODERATE — The fixes address iter_002's specific failures, but the core risk (metric collapse on semantic tasks) remains unverified. This is the most direct continuation of our previous work but carries the baggage of previous failure.

### Against Candidate B (Cross-Architecture Controlled Analysis)

- **Confound attack:** Training dynamics differ across architectures in ways that are hard to control. Matryoshka SAEs have ~50% computational overhead — they may simply train longer effectively. The "controlled" ablation may not be as controlled as it appears.
- **Statistical attack:** ANOVA assumes normality and homogeneity of variance. SAE training is stochastic — we may need non-parametric alternatives (Kruskal-Wallis). With 5 architectures x 3 dictionary sizes x 3 sparsity levels = 45 conditions, we need multiple replicates per condition, quickly exceeding our 1-hour budget.
- **Benchmark attack:** SynthSAEBench is synthetic. Even if we find clear component effects, they may not transfer to real LLMs. The synthetic data has known ground-truth features — real LLM features may not be hierarchical in the same way.
- **Ablation completeness attack:** We can vary one component at a time, but real architectures combine multiple innovations (Matryoshka = multi-scale + BatchTopK). Interaction effects may dominate main effects.
- **Verdict:** MODERATE — Stronger experimental control than Candidate A, but synthetic-to-real transfer is uncertain. The 1-hour constraint may force insufficient replication.

### Against Candidate C (Absorption-Downstream Impact)

- **Confound attack:** Absorption rate is strongly confounded with L0 sparsity and reconstruction quality. A multiple regression can control for these, but with n=8-10 SAEs and 3 predictors, we're at risk of overfitting. The "dose-response" may actually be a sparsity-response or reconstruction-response.
- **Statistical attack:** With n=8-10 SAEs and 3 downstream tasks (plus controls), we have low power. The correlation threshold (r < -0.4) may be undetectable. A negative result could mean "no relationship" or "insufficient power."
- **Benchmark attack:** Downstream tasks themselves are debated. Kantamneni et al. (2025) showed SAEs don't consistently outperform linear probes. If downstream tasks are invalid proxies for interpretability, the whole study is moot.
- **Ablation completeness attack:** We control for L0 and explained variance, but there may be other confounders (dictionary size, training data, layer depth). A causal claim requires more than regression control.
- **Verdict:** MODERATE — The most practically important question (does absorption matter?), but also the most confounded. The dose-response framing is appealing but may be unachievable with available SAEs.

---

## Phase 4: Refinement

### Analysis of Iteration 002 Failures

Our previous construct-validity study (iter_002) failed for specific, identifiable reasons:

1. **Metric collapse** (CRITICAL): All hierarchies showed resid_acc = sae_acc = 1.0, causing absorption = (1 - k_sparse_acc). This means our "semantic-hierarchy absorption" was actually measuring k-sparse probing difficulty, not SAE encoding loss. The metric was not measuring the intended construct.

2. **Random-SAE contradiction** (CRITICAL): Section 3.1 said "encoder permutation"; Section 4.5 said "decoder permutation." Raw data showed bit-for-bit identical scores, consistent with decoder permutation. The paper interpreted this as "degeneracy" when it is expected behavior (the metric depends on encoder geometry).

3. **Underpowered sample** (MAJOR): n=7 SAEs with CI spanning [-0.389, 0.981]. The test was uninformative, not "inconclusive." Power to detect r=0.6 was ~0.35.

4. **Confounded controls** (MAJOR): Multi-class hierarchies (parent vs 2-3 children) vs binary non-hierarchy pairs (word A vs word B) are structurally inequivalent.

5. **Ceiling effects** (MAJOR): Perfect probe AUROCs (1.0) were treated as success when they indicate the task is too easy.

6. **Weak pilot gates** (MAJOR): Pilot only checked ranking order, not metric validity or control behavior.

### Dropped

- **Candidate A (Fixed Construct Validity)** — DROPPED. The risk of metric collapse on semantic hierarchies is too high and unverifiable without extensive piloting. Even if we fix all identified issues, we may still find that semantic hierarchies are too easy for modern SAEs, causing the same collapse. This would waste another iteration on the same fundamental problem.

### Strengthened

**Candidate B (Cross-Architecture Controlled Analysis on Synthetic Data)** — SELECTED AS FRONT-RUNNER with modifications:

**Key strengthening:**
1. **Use SynthSAEBench-16k** instead of LLM SAEs. Ground-truth features eliminate the metric collapse problem — we know exactly what features exist and can measure recovery directly.
2. **Focus on component isolation**, not architecture ranking. The question is: which architectural innovation actually reduces absorption, and by how much?
3. **Add real-LLM validation**: After identifying the most effective component on synthetic data, validate on 2-3 real LLM SAEs to test transfer.
4. **Pre-registered power analysis**: With ground-truth data, we can compute exact effect sizes. Target n=5 replicates per architecture x 3 architectures = 15 training runs. Each run is ~5-10 min on SynthSAEBench-16k.

**Candidate C (Absorption-Downstream Impact)** — KEPT AS BACKUP:
- If Candidate B reveals that a specific component (e.g., multi-scale loss) is the primary driver of absorption reduction, Candidate C can test whether that component improves downstream performance.
- Lower priority because it's more confounded and requires more SAEs.

### Why Candidate B is the Best Path Forward

1. **Eliminates metric collapse**: Ground-truth synthetic data means we measure actual feature recovery, not proxy metrics.
2. **Stronger causal inference**: Varying one component at a time provides cleaner causal claims than correlating across existing SAEs.
3. **Builds on iter_002's lessons**: We learned that validating against first-letter tasks is risky. Synthetic ground-truth data provides an independent validation path.
4. **Aligns with community direction**: SynthSAEBench (2026) is emerging as the gold standard for controlled SAE evaluation. Our study would be among the first to use it for architecture comparison.
5. **Falsifiable and bounded**: If no component significantly reduces absorption on ground-truth hierarchies, we have a clear negative result with strong implications.

---

## Phase 5: Final Proposal

### Title
**What Actually Reduces Feature Absorption? A Component-Isolated Analysis Using Ground-Truth Synthetic Hierarchies**

### Hypothesis

**H1 (Multi-Scale Loss):** Adding multi-scale reconstruction loss (Matryoshka-style) reduces absorption rate on ground-truth hierarchical features more than switching from L1 to TopK sparsity. Effect size: Cohen's d > 0.8 compared to TopK baseline.

**Falsification criterion:** If multi-scale loss shows no significant absorption reduction vs TopK baseline (p > 0.05, Tukey HSD), H1 is falsified.

**H2 (Component Ranking):** The relative contribution of architectural components to absorption reduction (from largest to smallest) is: multi-scale loss > orthogonality penalty > TopK sparsity > gated activation.

**Falsification criterion:** If the observed ranking differs from H2 by more than one swap (e.g., gated activation outperforms multi-scale loss), H2 is falsified.

**H3 (Synthetic-to-Real Transfer):** The component ranking observed on SynthSAEBench-16k transfers to real LLM SAEs (Pythia-160M). Kendall's tau_rank between synthetic and real rankings > 0.6.

**Falsification criterion:** If tau_rank <= 0 or is non-significant, synthetic results do not transfer to real LLMs.

### Method

**Phase 1: Synthetic Controlled Experiments (Primary)**

Use SynthSAEBench-16k (Chanin & Garriga-Alonso, 2026):
- Ground-truth features: 16,384
- Hidden dimension: 768
- Hierarchical features: 10,884 (128 root trees, branching factor 4, depth 3)
- Superposition level: rho_mm ≈ 0.15

**Architecture variants** (varying one component at a time from a Standard ReLU baseline):

| Variant | Component Changed | What It Tests |
|---------|-------------------|---------------|
| Baseline | Standard ReLU + L1 sparsity | Reference point |
| +TopK | Replace L1 with TopK sparsity | Effect of explicit sparsity control |
| +MultiScale | Add multi-scale reconstruction loss (2 scales) | Effect of hierarchical reconstruction |
| +Orthogonal | Add chunk-wise orthogonality penalty | Effect of decoder orthogonality |
| +Gated | Replace ReLU with gated activation | Effect of decoupled detection/magnitude |
| Matryoshka | TopK + MultiScale (full architecture) | Combined effect |

**Training config:**
- SAE width: 4,096 (recommended by SynthSAEBench authors)
- Training samples: 200M
- Fixed random seed per replicate
- 5 replicates per variant

**Metrics:**
- Ground-truth feature recovery: Mean Correlation Coefficient (MCC)
- Absorption rate on known hierarchies: fraction of parent features "absorbed" by child features
- Reconstruction quality: Explained Variance (R²)
- Sparsity: L0

**Phase 2: Real-LLM Validation (Secondary)**

Validate top 2-3 components on real LLM SAEs:
- Use existing pretrained SAEs from SAELens that match our variants
- Measure first-letter absorption (SAEBench) as the real-world proxy
- Compare component ranking: synthetic vs real

### Evaluation Protocol

**Primary benchmarks:**
- SynthSAEBench-16k ground-truth hierarchy absorption (novel controlled evaluation)
- Feature recovery MCC (ground-truth metric)

**Metrics with statistical test plan:**
- Absorption rate per variant (mean ± std across 5 replicates)
- One-way ANOVA across 6 variants; post-hoc Tukey HSD for pairwise comparisons
- Effect sizes: Cohen's d for each component vs baseline
- Multiple comparison correction: Bonferroni across all pairwise comparisons (15 comparisons, alpha = 0.0033)

**Random seeds:** 5 random seeds per variant; report mean ± std and individual replicates.

**Power analysis:** With 6 groups, 5 replicates each, and expected Cohen's d = 0.8 (large effect), power ≈ 0.85 at alpha = 0.05. Pre-registered.

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| Vary dictionary size (2x, 4x, 8x hidden dim) | Whether component effects scale with capacity | MultiScale benefit increases with width |
| Vary sparsity (L0 = 50, 100, 200) | Whether component effects depend on sparsity | Orthogonality most effective at low L0 |
| Vary hierarchy depth (2, 3, 4 levels) | Whether component effects depend on hierarchy complexity | MultiScale most effective at deeper hierarchies |
| Remove superposition noise | Whether results depend on superposition | Effects persist but are smaller |

### Control Experiments

1. **Random-feature control:** Train SAE on SynthSAEBench with randomly permuted feature directions. Should achieve near-zero MCC and high absorption — validates that our metrics detect absence of structure.
2. **Baseline vs null:** Standard ReLU baseline should show significantly worse absorption than all component variants — validates that our experimental setup can detect differences.
3. **Replication check:** 5 replicates per variant; report variance. High variance would indicate unstable training dynamics.

### Pilot Design

- **Scope:** Baseline + 2 variants (TopK, MultiScale) on SynthSAEBench-16k subset (1,024 features, 10M samples)
- **Target runtime:** 15-20 minutes on 1 GPU
- **Success criteria (HARD GATES):**
  1. All variants achieve MCC > 0.5 (SAEs learn something)
  2. Clear ordering: MultiScale < TopK < Baseline in absorption rate
  3. Variance across replicates < 20% of mean (stable training)
  4. Random-feature control achieves MCC < 0.1 (metrics detect absence of structure)
  5. No metric collapse (ground-truth absorption formula uses known features, not probes)

### Resource Estimate

- **GPU-hours:** ~0.5-1.0 GPU-hour for synthetic experiments (6 variants x 5 replicates x ~2 min each = 60 min; subset pilot ~15 min)
- **Model sizes:** SynthSAEBench-16k (synthetic, fast); Pythia-160M (real validation, optional)
- **Well under 1-hour limit per task**

### Risk Assessment

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Synthetic data doesn't match LLM feature structure | HIGH | Phase 2 real-LLM validation; acknowledge limitation in Discussion |
| Training instability (high variance across replicates) | MEDIUM | 5 replicates; report variance; increase if needed |
| Component interactions dominate main effects | MEDIUM | Test full Matryoshka variant; report interaction if observed |
| SynthSAEBench setup complexity | MEDIUM | Use official implementation; test on subset first |
| Real-LLM SAEs don't match synthetic variants exactly | MEDIUM | Acknowledge approximate matching; focus on component presence/absence |

### Novelty Claim

This would be the **first component-isolated study of SAE architectural innovations using ground-truth synthetic hierarchies**. While Matryoshka SAEs, OrtSAE, and Gated SAEs have all reported absorption reductions, no study has isolated which specific component drives the improvement. By varying one component at a time on SynthSAEBench-16k — where ground-truth features are known — we can make causal claims about what actually works. The synthetic-to-real validation (Phase 2) tests whether these causal claims transfer to the real-world setting that matters for the community.

### Connection to Iteration 002

This proposal directly addresses the failures of iter_002:
- **Metric collapse eliminated:** Ground-truth features mean no probe-based metric collapse.
- **Underpowered sample fixed:** 5 replicates x 6 variants = 30 data points; ANOVA has adequate power.
- **Confounded controls eliminated:** All conditions use identical ground-truth hierarchies.
- **Ceiling effects eliminated:** Synthetic data can be made arbitrarily difficult.
- **Weak pilot gates strengthened:** Hard gates include MCC threshold, variance bound, and random-feature control.

The research question shifts from "Does first-letter absorption generalize?" (which we couldn't answer due to metric collapse) to "What architectural components actually reduce absorption?" (which we can answer with ground-truth data). This is a more constructive and empirically tractable question.

### Sources

- [A is for Absorption (Chanin et al., 2024)](https://arxiv.org/abs/2409.14507)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [SynthSAEBench (Chanin & Garriga-Alonso, 2026)](https://arxiv.org/abs/2602.14687)
- [CE-Bench (2025)](https://arxiv.org/abs/2509.00691)
- [Matryoshka SAE (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [Feature Hedging (Chanin, 2025)](https://arxiv.org/abs/2505.11756)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [Are SAEs Useful? (Kantamneni et al., 2025)](https://arxiv.org/abs/2502.16681)
- [Unified Theory of SDL (2025)](https://arxiv.org/abs/2512.05534)
- [Feature Sensitivity (2025)](https://arxiv.org/abs/2509.23717)
- [Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)

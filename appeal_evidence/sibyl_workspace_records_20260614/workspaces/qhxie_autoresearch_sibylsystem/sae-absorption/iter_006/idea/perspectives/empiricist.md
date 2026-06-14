# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al., 2024. "A is for Absorption." arXiv:2409.14507 (NeurIPS 2025 Oral)** -- Defines the canonical absorption metric: k-sparse probing to find feature splits, false-negative identification, integrated-gradients attribution, cosine/magnitude threshold criteria. Reports 15-35% absorption rates across all tested SAEs. Key evaluation insight: the metric requires known probe directions and has specific, published thresholds (cosine > 0.025, magnitude gap >= 1.0). The metric has never been subjected to a threshold sensitivity analysis in the original paper.

2. **SAEBench (Karvonen et al., 2025. arXiv:2503.09532)** -- 8-metric evaluation suite including absorption as a core metric. 200+ open-source SAEs across 8 architectures on Pythia-160M and Gemma-2-2B. Critical finding: proxy metrics (CE loss, sparsity) do NOT reliably predict practical SAE performance. Matryoshka SAEs lead on absorption; JumpReLU/TopK worsen absorption despite improving sparsity-fidelity frontier. Provides standardized baselines for all future absorption research.

3. **SynthSAEBench (arXiv:2602.14687, Feb 2026)** -- Large-scale synthetic benchmark with ground-truth feature hierarchies, correlation structures, and Zipfian firing distributions. Reveals SAEs substantially underperform direct logistic probes on feature recovery (logistic probes achieve 0.974 F1 while best SAEs are much lower). Enables controlled study of absorption under known hierarchy parameters -- critical for validating any theoretical prediction of absorption rates.

4. **Tian et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** -- Frames absorption as a special case of poor feature sensitivity. Develops scalable sensitivity evaluation methods. Key methodological finding: many features that appear interpretable from their activation examples have poor sensitivity (fail to activate reliably on similar texts). Sensitivity declines with increasing SAE width across 7 SAE variants. This directly challenges the assumption that high auto-interpretability scores validate SAE quality.

5. **Heap et al., 2025. "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers." arXiv:2501.17727** -- Shows SAEs trained on randomly initialized transformers produce auto-interpretability scores similar to trained models. This is a devastating false-positive result for the standard evaluation pipeline. Implication for absorption research: any absorption metric must be validated against random baselines, not just against auto-interpretability scores.

6. **LessWrong post: "Looking for feature absorption automatically" (Aug 2025)** -- The only prior attempt at unsupervised absorption detection. Method: find SAE latents with similar causal effects (cosine similarity of effect vectors) that don't co-activate. Result: FAILED. The cosine similarity distribution was non-bimodal, and examples could be cherry-picked from both ends. Critical negative result establishing that naive decoder/effect-vector cosine similarity is insufficient for automated absorption detection.

7. **Chanin & Garriga-Alonso, 2025. "Sparse but Wrong." arXiv:2508.16560** -- Most open-source SAEs have L0 that is too low, causing feature hedging (correlated feature mixing). Too-low L0 triggers hedging; too-high L0 finds degenerate solutions. Proposes a proxy metric for finding correct L0. This is a critical confound for ALL absorption studies: what appears to be absorption may be L0-induced hedging.

8. **Song et al., 2025. "Mechanistic Interpretability Should Prioritize Feature Consistency." arXiv:2505.20254** -- SAE features are inconsistent across training runs. TopK SAEs achieve PW-MCC of ~0.80 (meaning 20% of features differ). Implication: any cross-SAE absorption comparison must account for feature inconsistency as a noise source. Different training runs of the same architecture may show different absorption patterns simply due to initialization randomness.

9. **"Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (arXiv:2502.16681, 2025)** -- SAE probes underperform baseline logistic regression probes on mean test AUC across datasets. Baseline LR probes on raw activations achieve comparable or better performance even on interpretability case studies. This constrains our expectations: if SAE probes are not reliably better than dense probes, absorption may be only one of several reasons for the gap.

10. **DeepMind Safety Research, 2025. "Negative Results for SAEs on Downstream Tasks."** -- 1-sparse SAE probes fail catastrophically at detecting harmful intent, while dense linear probes achieve near-perfect accuracy even out-of-distribution. Feature absorption is implicated as a key culprit. This provides the strongest practical motivation for studying absorption but also warns that the SAE-probe gap may reflect fundamental architectural limitations, not just absorption.

11. **Narayanaswamy et al., 2026. "Improving Robustness in SAEs via Masked Regularization." arXiv:2604.06495** -- Very recent (April 2026). Proposes token masking during training to disrupt co-occurrence patterns that drive absorption. Reduces absorption and improves OOD robustness. Key evaluation insight: masking-based regularization is the simplest proposed mitigation, testable across all SAE architectures.

12. **Korznikov et al., 2025. "OrtSAE." arXiv:2509.22033** -- Three complementary absorption metrics: mean absorption fraction, full-absorption score, and feature splits. Reduces absorption 65% via orthogonality penalty. The three-metric suite provides richer evaluation than the single Chanin metric. These metrics have NOT been adopted by SAEBench, creating a gap in standardized evaluation.

### Experimental Landscape

**What has been properly tested:**
- Absorption existence and prevalence on first-letter spelling task across multiple models (Gemma 2B, 9B, Llama 3.2 1B, Qwen2 0.5B) and SAE architectures (L1, TopK, JumpReLU, Gated, Matryoshka) -- WELL ESTABLISHED.
- Absorption metric implementation on Gemma Scope SAEs via SAEBench -- STANDARDIZED.
- The scaling behavior of absorption with SAE width and training time -- PARTIALLY TESTED (SAEBench shows trends but no formal scaling law with controlled confounds).
- The observation that proxy metrics (MSE, CE loss) do not predict absorption -- WELL ESTABLISHED.

**What is accepted without proper evidence:**
- That first-letter absorption rates (15-35%) generalize to other feature hierarchies. NO EVIDENCE. The first-letter task is uniquely clean (26 non-overlapping, exhaustive classes). Real-world hierarchies are messier.
- That absorption is the primary cause of the SAE-probe vs. dense-probe gap on safety tasks. IMPLICATED BUT NOT DECOMPOSED. No study has quantified what fraction of the gap is due to absorption vs. hedging vs. dead features vs. inherent SAE capacity limitations.
- That absorption occurs at the same rate across layers. CONTRADICTED by our iteration 1 data showing layer-dependent EDA performance.
- That wider SAEs have more absorption. SAEBench data suggests this but L0 and training duration are confounded with width.

**Where methodological gaps exist:**
- NO unsupervised absorption detection method exists. The only attempt (LessWrong) failed.
- NO cross-domain absorption measurement on an adequately sized model. Our iteration 1 attempted cross-domain on Gemma 2 2B but used a proxy metric (decoder cosine similarity as absorption proxy) rather than the canonical Chanin probe-based metric.
- NO threshold sensitivity analysis of the Chanin absorption metric itself. The cosine > 0.025 and magnitude gap >= 1.0 thresholds were chosen without published justification for robustness.
- NO formal decomposition of measured absorption into component causes (hierarchy-driven, L0-induced hedging, reconstruction error, metric sensitivity).
- NO statistical power analysis for cross-domain absorption measurement. How many entities/examples are needed per hierarchy level to reliably detect a 10% absorption rate?

## Phase 2: Initial Candidates

### Candidate A: Controlled Cross-Domain Absorption Measurement with Full Confound Decomposition

- **Core hypothesis**: Feature absorption occurs at measurable rates (>= 10%) in knowledge hierarchies (city-country, city-continent, city-language) on Gemma 2 2B Gemma Scope SAEs, when measured with the canonical Chanin et al. probe-based metric adapted for new domains, with proper confound controls.

- **Falsification criterion**: Absorption rate < 5% across ALL knowledge hierarchy domains after (a) probe quality gating (F1 > 0.85), (b) frequency-matched random baseline subtraction, (c) L0 covariate correction. The 5% threshold is chosen because our iteration 1 cross-domain data showed mean rates of 0.1-1.7% using a proxy metric -- if the full Chanin metric also shows rates this low, H1 is falsified.

- **Evaluation protocol**:
  - **Primary benchmark**: Chanin et al. absorption rate metric adapted for knowledge hierarchies on Gemma 2 2B, layers 10/12/20, widths 16k and 65k (residual stream Gemma Scope SAEs).
  - **Metrics**: Absorption rate per parent class, per domain, per SAE. F1 of k-sparse probes. False-negative rate. Precision@50 of integrated-gradients attribution.
  - **Statistical tests**: Bootstrap 95% CIs (10,000 resamples) for absorption rates. Two-sided Mann-Whitney U for cross-domain comparisons. Benjamini-Hochberg FDR correction across all pairwise comparisons. Partial Spearman correlations controlling for log(L0) and log(width).
  - **Random seeds**: Not applicable (training-free analysis), but activation extraction uses fixed random seed for any subsampling (seed=42).

- **Ablation plan**:
  - A1: Probe sparsity k={1, 3, 5, 10} -- tests whether probe granularity affects measured absorption. Expected: higher k may find more split features, potentially reducing apparent absorption by capturing more of the feature's variance across multiple latents.
  - A2: Entity sample size {25, 50, 100, 200 entities per domain} -- tests statistical sufficiency. Expected: convergence at N~50 for stable absorption rate estimates.
  - A3: Threshold sensitivity (5x4 grid: cosine in {0.01, 0.02, 0.025, 0.03, 0.05}, magnitude gap in {0.5, 1.0, 1.5, 2.0}) -- tests metric robustness. Expected: if CV < 0.3 across the grid, the metric is stable. If CV > 0.5, the specific threshold choice dominates the result (which is itself an important finding about metric reliability).
  - A4: Layer sweep (layers 5/10/12/19/20) -- tests layer dependence. Our iteration 1 showed L12-16k was best (AUROC 0.776) while L19-16k was near chance (0.458).
  - A5: Frequency-matched vs. unmatched baselines -- our iteration 1 found that raw Chanin metric on GPT-2 produced 92.3% absorption (artifact), corrected to 19.2% with frequency matching. This ablation tests whether frequency matching matters for Gemma 2 2B.

- **Confounders identified**:
  1. **L0 confound**: Most open-source SAEs have incorrect L0 (Chanin & Garriga-Alonso, 2025). Must control via partial correlation.
  2. **Width confound**: Iteration 1 showed within-width associations vanish (Rosenbaum Gamma=1.0). Width is a major confound.
  3. **Training duration confound**: JumpReLU SAEs in Gemma Scope were trained longer. Architecture confounded with training.
  4. **Prompt template sensitivity**: Different ICL prompts may trigger different model behaviors. Must test 2-3 template variants.
  5. **Entity tokenization**: Multi-token entities vs. single-token entities may have different absorption patterns. Must restrict to single-token or systematically compare.
  6. **Hierarchy messiness**: Knowledge hierarchies have overlapping classes (Istanbul: Turkey AND Europe AND Turkish-speaking). First-letter is non-overlapping. This structural difference could either increase or decrease absorption.

- **Pilot design**: 50 single-token city names, Gemma 2 2B layer 12, 16k SAE. Train k=5 sparse probe for "country" parent concept. Check probe F1. If F1 > 0.85, proceed. If F1 < 0.70, pivot to animal taxonomy or number parity. Estimated time: 15 minutes.

### Candidate B: Metric Robustness Audit -- Is the Absorption Metric Itself Reliable?

- **Core hypothesis**: The canonical Chanin absorption metric is sensitive to its threshold parameters (cosine similarity > 0.025, magnitude gap >= 1.0), and at least 30% of measured absorption is attributable to threshold choice rather than genuine feature absorption.

- **Falsification criterion**: If the coefficient of variation of absorption rates across the full 5x4 threshold grid is < 0.15 (metric is highly robust to thresholds), AND the metric produces near-zero rates on all three negative controls (random probes, shuffled labels, untrained SAEs), then the metric is validated as reliable.

- **Evaluation protocol**:
  - **Primary benchmark**: First-letter task on Gemma 2 2B (established ground truth).
  - **Metrics**: Absorption rate across 20 threshold combinations. CV, IQR, and max/min ratio across the grid. Negative control absorption rates.
  - **Statistical tests**: Friedman test for threshold effect across SAE configurations. Bootstrap CIs per threshold combination.

- **Ablation plan**:
  - A1: Vary cosine threshold alone (0.005 to 0.10 in 10 steps), holding magnitude gap at 1.0.
  - A2: Vary magnitude gap alone (0.25 to 3.0 in 10 steps), holding cosine at 0.025.
  - A3: Full grid (cosine x gap x SAE_config) -- 3-way interaction analysis.
  - A4: Compare integrated-gradients attribution with gradient*input and attention-based attribution as alternatives.

- **Confounders identified**:
  1. **Publication bias**: The thresholds in Chanin et al. may have been selected post hoc to produce the cleanest results. No robustness analysis was published.
  2. **Activation magnitude scale**: Different layers and models have different activation magnitudes, so fixed thresholds may not transfer.
  3. **Probe quality variation**: Low-quality probes produce noisy absorption measurements, inflating or deflating apparent rates.

- **Pilot design**: Run the 5x4 threshold grid on the first-letter task, L12-16k SAE, for 5 letters (S, A, T, E, Z -- spanning high and low frequency). 10 minutes.

### Candidate C: Decomposing the SAE-Probe vs. Dense-Probe Gap -- How Much Is Actually Absorption?

- **Core hypothesis**: For hierarchical classification tasks on Gemma 2 2B, the gap between dense logistic regression probe performance and SAE k-sparse probe performance can be decomposed into at least three component causes, and true hierarchy-driven absorption accounts for less than 50% of the total gap.

- **Falsification criterion**: If hierarchy-driven absorption accounts for > 70% of the gap (after controlling for dead features and hedging), AND the gap is > 10 F1 points, then absorption is confirmed as the dominant failure mode.

- **Evaluation protocol**:
  - **Primary benchmark**: First-letter and city-country tasks on Gemma 2 2B.
  - **Metrics**: Dense probe F1 (ceiling). SAE k-sparse probe F1. F1 gap. Decomposition: (a) dead features (latents with zero activation on test set), (b) hedging (multiple latents partially encoding same concept, detectable by high inter-latent correlation among top-k), (c) hierarchy-driven absorption (integrated-gradients confirms another specific latent absorbed the signal), (d) residual (unexplained gap).
  - **Statistical tests**: Chi-square test for decomposition proportions. Bootstrap CIs for each component proportion.

- **Ablation plan**:
  - A1: Vary SAE width (16k vs 65k) -- wider SAEs should have fewer dead features but potentially more absorption.
  - A2: Vary probe sparsity k -- at high k, the gap may close because more latents compensate for absorption.
  - A3: Vary hierarchy type -- first-letter (clean, non-overlapping) vs. city-country (messy, overlapping) vs. city-continent (very coarse).

- **Confounders identified**:
  1. **Decomposition attribution ambiguity**: A false negative could be caused by multiple mechanisms simultaneously. Need clear precedence rules.
  2. **Dense probe ceiling effect**: If the dense probe is perfect (F1=1.0), the gap is entirely attributable to SAE limitations. If the dense probe is imperfect, some gap is due to the task difficulty, not SAE failure.
  3. **Hedging vs. absorption overlap**: A feature that is both hedged (correlated with neighbors) and absorbed (captures parent information) is counted where? Must pre-specify the decision tree.

- **Pilot design**: Dense LR probe vs. k=5 sparse probe on first-letter task, Gemma 2 2B L12-16k. Compute gap for 5 letters, run integrated-gradients on false negatives to classify causes. 20 minutes.

## Phase 3: Self-Critique

### Against Candidate A (Controlled Cross-Domain Absorption)

- **Confound attack**: The most dangerous confound is that the Chanin metric may not transfer cleanly to knowledge hierarchies. The first-letter task has a critical property: the hierarchy is EXHAUSTIVE and NON-OVERLAPPING. Every token starts with exactly one letter. Knowledge hierarchies are not like this -- Paris is in France AND in Europe AND is French-speaking. The metric was designed for the non-overlapping case. Applying it to overlapping hierarchies may produce either false positives (counting overlapping activations as absorption) or false negatives (failing to detect absorption when multiple parent features legitimately co-activate). **Searched for papers addressing overlapping hierarchies in SAE evaluation: found none.** This is a genuine methodological gap that must be addressed in the experimental design. **Mitigation**: (a) restrict to non-overlapping sub-hierarchies where possible (e.g., each city assigned to exactly one country), (b) explicitly test sensitivity to overlap degree, (c) develop an overlap-corrected absorption metric as a contribution.

- **Statistical attack**: Our iteration 1 cross-domain data suggests very low absorption rates using proxy metrics (0.1% for city-continent, 1.7% for city-country). If the Chanin metric produces similarly low rates, detecting a 10% rate requires sufficient statistical power. For a two-sided binomial test (H0: rate=0.05, H1: rate=0.10), with alpha=0.05 and power=0.80, we need approximately N=200 entity-level observations per domain. With 20 countries * 5 cities = 100 entities, we may be underpowered for detecting a 10% vs. 5% difference. **Mitigation**: (a) report power analysis explicitly, (b) use 100+ entities per domain, (c) use one-sided test if theory predicts direction, (d) pre-specify that rates < 5% are "low absorption" (not necessarily zero).

- **Benchmark attack**: The Chanin absorption metric on Gemma 2 2B has been validated by SAEBench for first-letter only. For knowledge domains, there is no gold-standard benchmark. We would be creating a new evaluation setting, which is both a contribution and a risk (no external validation possible). RAVEL provides entity-attribute data for Gemma 2 2B, which partially validates that the model encodes these hierarchies. **Mitigation**: validate that the model has adequate knowledge representations before measuring absorption (probe F1 gate).

- **Ablation completeness attack**: The probe sparsity ablation (k=1 to 10) is important because higher k may "rescue" absorbed features by including more latents. But this conflates absorption detection with probe expressiveness. A k=10 probe that achieves high F1 does not mean absorption is absent -- it means 10 latents collectively encode the feature, which may include the absorbing latent. The absorption metric itself uses integrated-gradients attribution on false negatives, which is independent of probe sparsity for detection purposes. However, different k values produce different false-negative sets, affecting the denominator. **This interaction between probe sparsity and absorption measurement is not discussed in Chanin et al. and should be examined.**

- **Verdict**: **STRONG** -- High feasibility, addresses the most important gap (cross-domain generalization), proper confound controls. Main risk is statistical power for low absorption rates and metric transfer to overlapping hierarchies. Both are addressable.

### Against Candidate B (Metric Robustness Audit)

- **Confound attack**: The threshold sensitivity analysis is important but may produce a tautological result: any continuous metric with a discrete threshold will be sensitive to that threshold near the boundary. The more informative analysis is whether the RANKING of SAEs by absorption rate is stable across thresholds, not whether the absolute rate changes. **Mitigation**: Report both absolute sensitivity (CV of rates) and rank-order stability (Kendall tau across threshold pairs).

- **Statistical attack**: With 20 threshold combinations and 6+ SAE configurations, we have 120+ comparisons. FDR correction will be aggressive. The effect sizes may be small (threshold changes of 0.01 in cosine may not change rates much). **Mitigation**: Focus on rank-order stability rather than pairwise significance tests.

- **Benchmark attack**: This candidate audits the metric itself, which is valuable but not a standard contribution format for ML venues. Reviewers may view it as "checking someone else's work" rather than providing new findings. **Mitigation**: Frame as "evaluation methodology contribution" and pair with Candidate A's cross-domain results.

- **Ablation completeness attack**: The alternative attribution methods (gradient*input, attention-based) are reasonable ablations, but the Chanin metric specifically requires integrated-gradients because it needs signed attribution magnitudes. Switching attribution methods changes the metric fundamentally, not just a parameter. **Mitigation**: Clearly frame alternative attributions as testing whether the choice of attribution method (not just thresholds) affects conclusions.

- **Verdict**: **MODERATE** -- Important for establishing metric reliability, but insufficient as a standalone contribution. Best combined with Candidate A.

### Against Candidate C (Gap Decomposition)

- **Confound attack**: The decomposition requires assigning each false negative to exactly one cause, but causes can co-occur. A latent can be both dead on some tokens and absorbed on others. A hedged feature pair can also exhibit absorption. The decomposition is necessarily approximate. **Searched for papers doing multi-cause decomposition of ML failure modes: found decision-tree-based error analysis in NLP evaluation but nothing specific to SAE failure modes.** **Mitigation**: (a) use a priority-ordered decision tree (dead features first, then hedging, then absorption, then residual), (b) report the sensitivity to ordering, (c) acknowledge that the decomposition is an approximation.

- **Statistical attack**: The decomposition requires classifying each false negative, which means running integrated-gradients attribution on potentially thousands of tokens. On Gemma 2 2B with 16k SAE, this is computationally feasible but slow (~2 seconds per token for IG with 50 interpolation steps). For 1000 false negatives, this is ~30 minutes per domain. Across 4 domains and 6 SAE configs, this is ~12 hours. **Within 1-hour-per-task constraint, must subsample.** **Mitigation**: subsample 100 false negatives per domain-config pair, with bootstrap CIs on decomposition proportions.

- **Benchmark attack**: DeepMind's negative result (SAE probes fail on safety tasks) is the strongest motivation for this decomposition, but their data is not public. We cannot directly compare our decomposition with their findings. We can only demonstrate the methodology on our own tasks and infer applicability. **Mitigation**: explicitly state that our decomposition is demonstrated on knowledge/spelling tasks and may differ for safety tasks.

- **Ablation completeness attack**: The "hedging detection" component (high inter-latent correlation among top-k) may conflate feature splitting (normal, expected behavior as SAE width increases) with hedging (pathological mixing of correlated features). Feature splitting is desirable; hedging is not. Distinguishing them requires knowing whether the split features capture genuinely distinct sub-concepts or are redundant. **Mitigation**: use the Chanin et al. definition of hedging (features that merge when width is reduced) rather than just inter-latent correlation.

- **Verdict**: **MODERATE** -- Conceptually important and addresses DeepMind's concern directly. But execution is complex, subsample-dependent, and the decomposition is inherently approximate. Best as a secondary contribution alongside Candidate A.

## Phase 4: Refinement

### Dropped: None outright, but Candidate B (metric audit) is absorbed into Candidate A as an ablation study.

The threshold sensitivity analysis from Candidate B becomes ablation A3 in Candidate A's protocol. This is more efficient (one paper, not two) and the threshold sensitivity results strengthen the cross-domain findings by establishing that the metric is (or is not) robust before applying it to new domains.

### Strengthened: Candidate A (Cross-Domain + Confound Decomposition)

Added controls and refinements:
1. **Overlap correction for knowledge hierarchies**: For each domain, explicitly construct the non-overlapping sub-hierarchy (each entity mapped to exactly one parent class) AND the full overlapping hierarchy. Report absorption for both. The difference reveals overlap-induced measurement artifacts.
2. **Power analysis pre-registered**: For each domain, compute the minimum detectable effect size given the number of entities and the baseline rate from random controls. Report whether we are adequately powered for the pre-specified H1 (>= 10%).
3. **Rank-order stability across thresholds**: In addition to absolute threshold sensitivity (CV), compute Kendall tau of SAE absorption rankings across all threshold pairs. Stable ranking (tau > 0.7) validates the metric; unstable ranking (tau < 0.3) is a critical finding.
4. **Probe quality as a pre-registration gate**: The probe F1 > 0.85 criterion is applied BEFORE absorption measurement. Domains failing this gate are reported as negative results (model does not adequately represent this hierarchy), not as zero-absorption domains.
5. **Dense probe ceiling**: For every domain, train a dense LR probe on raw model activations. The gap (dense F1 - SAE k-sparse F1) provides context for absorption rates: if the dense probe also fails (F1 < 0.85), the hierarchy is not well-represented in the model, and absorption is not the right explanation for poor SAE performance.

### Candidate C refined into a secondary analysis within the front-runner:

The gap decomposition becomes Stage 3 of the experimental plan (after cross-domain measurement). For false negatives identified in Stage 1, we classify them into dead-feature, hedging, hierarchy-driven absorption, and residual categories. This enriches the cross-domain findings without requiring a separate paper.

### Selected Front-Runner: Candidate A (enhanced)

**Rationale**: The strongest possible experimental contribution to the absorption literature is the first rigorous cross-domain measurement on an adequately sized model, with proper confound controls, threshold sensitivity analysis, and gap decomposition. This addresses Gaps 2, 6, 7 (partially), and 9 from the literature survey. The unsupervised detection pipeline (from the proposal's H3) is included as a validation component but de-prioritized relative to the supervised cross-domain measurement, because the LessWrong negative result establishes a high bar for unsupervised detection and our prior iteration's EDA/D-EDA metrics showed mixed results (AUROC 0.47-0.78 across configs).

## Phase 5: Final Proposal

### Title

Controlled Cross-Domain Measurement and Confound Decomposition of Feature Absorption in Sparse Autoencoders

### Hypothesis

**H1 (Primary)**: Feature absorption, measured by the Chanin et al. probe-based metric, occurs at rates >= 10% in at least one knowledge hierarchy domain (city-country, city-continent, city-language) on Gemma 2 2B Gemma Scope SAEs (layer 12, 16k width), after probe quality gating (F1 > 0.85), frequency-matched baseline subtraction, and L0 covariate correction.

**Falsification**: Absorption rate < 5% across ALL domains AND all SAE configurations after all corrections. Pre-registered as a decision BEFORE examining cross-domain data (first-letter baseline is examined first to validate the pipeline).

### Method

A training-free analysis pipeline using pre-trained Gemma Scope SAEs on Gemma 2 2B, extending the canonical Chanin et al. absorption metric to knowledge hierarchy domains.

### Evaluation Protocol

**Primary benchmarks:**
- Chanin et al. absorption rate metric on first-letter task (validation against published baselines)
- Same metric adapted for city-country, city-continent, city-language hierarchies (new)
- SAEBench absorption score (external comparison)

**Metrics with statistical test plan:**
- Absorption rate per parent class, per domain, per SAE config
- Bootstrap 95% CIs (10,000 resamples) for all rates
- Two-sided Mann-Whitney U for cross-domain absorption rate comparisons
- Partial Spearman correlations controlling for log(L0) and log(width) -- following iteration 5's validated methodology
- Benjamini-Hochberg FDR correction for all multiple comparisons
- Rank-order stability: Kendall tau across threshold pairs
- Power analysis: minimum detectable effect per domain at alpha=0.05, power=0.80

**Number of random seeds**: N/A (training-free). Fixed seed=42 for any subsampling. All results are deterministic given the same SAE weights and input data.

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|--------------|------------------|
| A1: Probe sparsity k={1,3,5,10} | Whether probe granularity affects absorption measurement | Higher k finds more splits, may reduce apparent absorption by ~20-30% |
| A2: Entity sample size {25,50,100,200} | Statistical sufficiency | Convergence at N~50-100 |
| A3: Threshold grid (5x4) | Metric robustness | If CV < 0.3, metric is stable; if CV > 0.5, threshold sensitivity is a primary finding |
| A4: Layer sweep (5,10,12,19,20) | Layer dependence | Layer 12 expected best based on prior data |
| A5: Frequency matching on/off | Whether frequency artifacts inflate rates | Expected significant: iteration 1 showed 92.3% vs 19.2% on GPT-2 |
| A6: Overlapping vs non-overlapping sub-hierarchy | Whether hierarchy overlap inflates/deflates absorption | Unknown; critical methodological test |
| A7: Prompt template variants (3 templates) | Template sensitivity | Rates should be stable within 5 pp across templates |

### Control Experiments

| Control | Purpose | Expected Result |
|---------|---------|-----------------|
| C1: Random probe direction | Validate metric produces ~0% with random directions | < 2% absorption rate |
| C2: Shuffled entity-parent labels | Validate metric requires correct hierarchy | < 2% absorption rate |
| C3: Untrained (random) SAE | Validate metric requires trained features | < 2% absorption rate |
| C4: Dense LR probe ceiling | Establish upper bound on what any probe could achieve | F1 > 0.90 on well-represented hierarchies |
| C5: Matched-L0 comparison | Control for L0 when comparing across SAE configs | Partial correlations controlling L0 |

### Pilot Design

**Exp 0 (15 minutes)**: 50 single-token city names, Gemma 2 2B layer 12, Gemma Scope 16k SAE. Train k=5 sparse logistic regression probe for "country" parent concept. Measure probe F1.
- Success gate: F1 > 0.85 --> proceed to full cross-domain measurement.
- Warning gate: 0.70 < F1 < 0.85 --> proceed with caution, report probe quality prominently.
- Fail gate: F1 < 0.70 --> pivot to alternative domain (animal taxonomy using WordNet hierarchy, or number parity).

**Exp 0b (5 minutes)**: Capacity check. Compute fraction of dead SAE latents on 100 city-country ICL prompts. If > 50% dead, the SAE config is inadequate (iteration 1 found 98% dead on GPT-2 Small).

### Resource Estimate

- **Model**: Gemma 2 2B (~5GB VRAM). Single A100/H100.
- **SAEs**: Gemma Scope pre-trained (HuggingFace download). No training required.
- **Software**: SAELens >= 4.0, TransformerLens, sae-spelling (Chanin et al. metric implementation), PyTorch, scikit-learn, scipy, statsmodels.
- **Total GPU-hours**: ~6-8 hours across all experiments. Individual tasks within 1-hour budget.
- **Training cost**: Zero.
- **Disk**: ~7GB (model + SAE weights + cached activations).
- **Target models**: GPT-2 Small (secondary, for replication), Gemma 2 2B (primary). Both publicly available.

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Probe F1 < 0.85 on city-country | 20% | HIGH | Probe quality gate; backup domains; single-token entities only; 5-min capacity check |
| Absorption rates < 5% in all knowledge domains | 25% | MEDIUM | Report as informative negative result; first-letter baseline provides guaranteed positive; frame as "absorption is domain-specific" |
| Threshold sensitivity CV > 0.5 (metric is brittle) | 25% | MEDIUM | Report as critical finding about metric reliability; propose robust aggregation (median across grid) |
| Within-width absorption-quality correlations vanish (replicating iteration 5) | 40% | LOW (expected) | Report prominently as confirmatory result; the width confound IS the finding |
| L0 confound eliminates all cross-domain differences | 30% | MEDIUM | Report partial correlations; the confound structure is informative |
| Overlapping hierarchies break the metric | 15% | HIGH | Test non-overlapping sub-hierarchies separately; develop overlap correction |
| Gemma Scope SAE download/loading failures | 10% | LOW | SAELens well-tested on Gemma Scope; version pin |
| Integrated-gradients is computationally slow for large-scale decomposition | 30% | MEDIUM | Subsample 100 false negatives per domain-config; parallelize on GPU; use gradient*input as fast approximation |

### Novelty Claim

The experimental contribution is answering a specific empirical question for the first time: **does feature absorption generalize beyond the first-letter spelling task to knowledge hierarchies in language models, when measured with the canonical metric under controlled conditions?**

This is NOT a methods paper (we do not propose a new metric or architecture). It is an empirical measurement paper with rigorous methodology: proper confound controls (L0, width, frequency matching), threshold sensitivity analysis of the metric itself, gap decomposition into component causes, and statistical methods (partial correlations, FDR correction, bootstrap CIs, power analysis) that have been validated in our prior iterations.

The novelty is in the QUESTION being answered with proper CONTROLS, not in the method used to answer it. This is analogous to a replication study with extension -- the most impactful kind of empirical contribution in a field where the phenomenon has only been demonstrated in a single narrow setting.

**Specific novel empirical contributions:**
1. First cross-domain absorption measurement on Gemma 2 2B with the canonical Chanin metric (not proxy metrics)
2. First threshold sensitivity analysis of the absorption metric itself
3. First confound-controlled decomposition of measured absorption into hierarchy-driven, L0-induced, and reconstruction-error components
4. First power analysis for absorption detection in knowledge hierarchies
5. First comparison of overlapping vs. non-overlapping hierarchy effects on absorption measurement

**What we explicitly do NOT claim is novel:**
- The absorption metric (Chanin et al., 2024)
- The SAE infrastructure (SAELens, Gemma Scope, TransformerLens)
- The existence of absorption (established fact)
- Statistical methods (partial correlation, bootstrap CI, FDR correction -- textbook methods applied to a new domain)

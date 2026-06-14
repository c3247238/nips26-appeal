# Research Proposal: The Competitive Geometry of Feature Absorption in Sparse Autoencoders

## Title

**Competitive Geometry of Absorption: Unsupervised Detection, Corpus Prediction, and Downstream Impact of Feature Absorption in Sparse Autoencoders**

---

## Abstract

Feature absorption — the systematic failure of sparse autoencoders (SAEs) to represent rare, general features because their information is hijacked by more frequent specific features — is empirically well-documented but theoretically unexplained and practically underevaluated. We present three tightly integrated contributions. First, we propose an **unsupervised absorption detector** grounded in the Lotka-Volterra competitive exclusion framework: two SAE latents satisfy the absorption condition if and only if their competition coefficient α_ij = σ_ij · (f_j / f_i) exceeds a threshold τ ≈ 1, where σ_ij is the co-activation rate normalized by the rarer feature's frequency. This is computable from activation statistics alone — no probe directions required. Second, we establish that **corpus co-occurrence statistics (PMI) predict which specific feature pairs will exhibit absorption** better than feature hierarchy depth alone — reframing absorption as primarily a data-driven rather than objective-driven phenomenon, and explaining why masked regularization works. Third, we provide the **first controlled test of the downstream causal chain**: we measure whether SAEs with lower absorption scores actually improve performance on interpretability and safety-relevant downstream tasks, using SAEBench plus a targeted harmful-intent detection experiment. All analyses are training-free, operating on pre-trained Gemma Scope SAEs.

---

## Motivation

The SAE feature absorption problem has three unresolved tensions that define the research frontier as of April 2026:

**Tension 1: Measurement vs. phenomenon.** The canonical absorption rate (15–35%, Chanin et al. 2024) is explicitly acknowledged as a conservative lower bound that captures only Type I (full single-latent) absorption while ignoring Type II (partial) and Type III (distributed multi-latent) absorption. The true failure rate may be 2–5x higher. Furthermore, the metric requires pre-specified probe directions — making it impossible to study absorption for the safety-relevant features that most motivate the research.

**Tension 2: Mechanism vs. intervention.** Architecture papers (Matryoshka SAE, OrtSAE, ATM SAE, masked regularization) report large absorption reductions but offer no unified mechanistic explanation. Why does masked regularization help? Why does orthogonality regularization help? Why do wider SAEs paradoxically show *higher* absorption despite having more capacity? The field optimizes the metric without understanding the mechanism — a classic Goodhart's Law vulnerability.

**Tension 3: Metric vs. impact.** DeepMind announced in 2025 that SAEs underperform dense linear probes on harmful intent detection and deprioritized SAE research. This finding is devastating for absorption research *if* absorption reduction does not actually improve downstream safety task performance. Yet no paper directly tests whether absorption rate correlates with downstream task performance across SAEs. We are potentially optimizing for a metric that does not capture what matters.

Our proposal addresses all three tensions in a unified empirical study.

---

## Research Questions

1. **RQ1 (Detection):** Can the Lotka-Volterra competition coefficient α_ij, computed from SAE activation statistics alone, predict which feature pairs exhibit absorption with precision ≥ 0.70 and recall ≥ 0.65 against ground-truth absorption labels, without using pre-specified probe directions?

2. **RQ2 (Mechanism):** Do corpus co-occurrence statistics (PMI between tokens) predict which specific letter-feature pairs are absorbed, independently of feature hierarchy depth — supporting a data-driven rather than objective-driven causal account of absorption?

3. **RQ3 (Impact):** Is SAE absorption rate (Chanin metric) correlated with downstream interpretability and safety-relevant task performance at a practically meaningful level (Pearson |r| > 0.3), or is the assumed causal chain unvalidated?

4. **RQ4 (Width Paradox):** Does the distributed absorption score DAS(P, k=3), computed by conditioning on multiple children simultaneously, increase monotonically with SAE width even as single-child absorption rates are non-monotone — explaining the Chanin et al. paradox via distributed competitive exclusion?

---

## Hypotheses

**H1 (Competitive Exclusion Absorption Detector):** The competition coefficient α_ij = σ_ij · (f_j/f_i) predicts feature absorption with a sharp transition near α_ij ≈ 1. Features with α_ij > 1 are absorbed with probability > 0.70; features with α_ij < 0.5 are absorbed with probability < 0.15. The detector achieves F1 > 0.65 against sae-spelling ground-truth labels on the first-letter task across multiple SAE architectures, without requiring any pre-specified probe directions.

**H2 (Corpus PMI Predictor):** Token-pair PMI from the SAE training corpus (OpenWebText) predicts which of the 26 letter-token feature pairs are absorbed with Pearson r > 0.50 against measured absorption rates, controlling for SAE width, L0, and hierarchy depth. This excess predictive power supports corpus statistics as a proximal cause of absorption beyond the structural feature hierarchy.

**H3 (Downstream Disconnection):** Pearson correlation between the Chanin absorption rate and SAEBench downstream task performance (RAVEL, SCR, sparse probing) is |r| < 0.2 for all tasks, confirming that optimizing the absorption metric is decoupled from optimizing downstream interpretability. The 1-sparse SAE probe gap on harmful intent detection is not explained by absorption rate variation across SAEs.

**H4 (Distributed Absorption):** DAS(P, k=3) increases monotonically with SAE width (16k → 65k → 131k), whereas single-child absorption rate (Chanin metric, k=1) shows a non-monotone or decreasing trend. This confirms the LV-theory prediction that wider SAEs produce more distributed competitive exclusion even as concentrated exclusion decreases.

---

## Expected Contributions

1. **Probe-free unsupervised absorption detector** based on the Lotka-Volterra competition coefficient — the first validated absorption detector requiring no pre-specified probe directions, enabling systematic study of absorption across arbitrary feature hierarchies including safety-relevant ones.

2. **Corpus co-occurrence as primary predictor** of absorption patterns — shifting the research community's attention from architecture engineering toward data engineering (corpus curation, co-occurrence disruption) as the principled intervention target.

3. **First direct test of the downstream causal chain** — providing empirical evidence on whether SAE absorption reduction translates to downstream improvement, directly addressing DeepMind's findings and the field's motivating assumption.

4. **Theoretical unification via LV competitive exclusion** — the competition coefficient α_ij provides a single quantity that explains why absorption occurs (α > 1), which features are at risk (high co-activation + frequency imbalance), why wider SAEs show more absorption (more latents → higher probability of α > 1 pairs), and how OrtSAE works (orthogonality regularization reduces σ_ij → reduces α_ij).

5. **Multi-type absorption taxonomy** (Types I/II/III) — establishing that the reported 15–35% rate substantially understates the true failure scope and that different absorption types have different mechanistic causes and downstream impacts.

---

## Method

### Component 0: Baseline Pipeline Validation (Pilot, 15 min)

Load Gemma Scope Gemma 2 2B, layer 12, 16k SAE. Run sae-spelling absorption measurement for letters A–E. Confirm absorption rate ≈ 15–35% (matches Chanin et al.). Simultaneously, check empirical L0 for 16k and 65k SAEs on the same 500-token sample to determine whether controlled width comparisons are possible.

**Decision gate:** If L0 differs > 20% across widths, controlled width comparisons are confounded; proceed with within-width L0 sweep only (Ablation A2 below).

---

### Component 1: LV Competition Coefficient as Unsupervised Absorption Detector (RQ1, H1)

**Step 1: Activation statistics collection.**
For each SAE (Gemma Scope Gemma 2 2B, layers 6/12/20/25, widths 16k/65k), run 10k tokens of OpenWebText through the SAE using SAELens. Record: mean activation frequency f_i per latent; pairwise co-activation rates P(a_i > 0, a_j > 0) for all pairs (i, j) with f_i > 0.001 and f_j > 0.001.

**Step 2: Competition coefficient computation.**
For each candidate pair above the frequency threshold:
- σ_ij = P(a_i > 0, a_j > 0) / min(f_i, f_j) — normalized co-activation (niche overlap)
- α_ij = σ_ij · (f_j / f_i) — LV competition coefficient

Pre-filter: additionally require decoder cosine similarity cos(d_i, d_j) > 0.15 to focus on geometrically adjacent features, reducing the candidate pair set to O(D·k) with k ≈ 20.

**Step 3: Threshold calibration and validation.**
For the first-letter task: obtain ground-truth absorption labels from sae-spelling on Gemma Scope 16k layer 12 (26 letters × ~100 test tokens each). For each absorbed pair identified by Chanin et al.: look up the computed α_ij and max_{j} α_ij for the parent feature.

Fit threshold τ on 13 letters (training set); evaluate on 13 letters (test set). Metrics: precision, recall, F1, ROC-AUC of α_ij as absorption predictor.

**Step 4: Cross-architecture validation.**
Apply the calibrated threshold to: (a) Gemma Scope JumpReLU SAEs at layer 12; (b) Gemma Scope Matryoshka SAEs. Predict which features are absorbed; validate against sae-spelling ground truth. Measure F1 degradation from the calibrated threshold — if it degrades by < 10% F1, the detector generalizes architectures.

**LV-specific diagnostic (sharpness test):**
Plot absorption rate (Chanin metric) vs. α_ij in bins of width 0.1 over the range [0, 3]. The LV theory predicts a step-function-like increase near α_ij ≈ 1 (sharp threshold). A generic sparsity-suppression story would predict a smooth monotone increase. Fit both a sigmoid (LV-consistent) and a linear function to the data; compare AIC.

**Distributed absorption — width paradox (RQ4, H4):**
For Gemma Scope SAEs at widths {16k, 65k, 131k}, layer 12, compute:
- DAS(P, k=1) = single-child absorption (Chanin metric)
- DAS(P, k=3) = 1 - I(X; f_P | f_{C1}, f_{C2}, f_{C3}) / H(f_P) where C1..C3 are the three highest-α_ij children of P, estimated via multi-label logistic regression on activation statistics

Predict: DAS(k=3) increases monotonically with width; DAS(k=1) is non-monotone.

---

### Component 2: Corpus Co-Occurrence Prediction (RQ2, H2)

**Step 1: PMI extraction from OpenWebText.**
For each of the 26 letter-token pairs used in the first-letter task (e.g., the pair ("first_letter_A", "specific_A_token")), compute:
- P(token_starts_with_L) across a 1M-token OpenWebText sample
- P(token_starts_with_L | context contains a highly frequent co-occurring word or token) — the conditional frequency in absorbing contexts
- PMI = log P(child, parent) / (P(child) · P(parent)) for each specific child token with its parent letter category

**Step 2: Regression model.**
For each of the 30 Gemma Scope SAE configurations (widths {16k, 65k, 131k} × L0 settings × layers {8, 12, 20, 25}), collect absorption rates for all 26 letters. Fit:

```
absorption_rate = β₀ + β₁ log(L0) + β₂ log(width) + β₃ layer + β₄ log(PMI_child_parent) + ε
```

Evaluate R², Pearson r between PMI and absorption after controlling for SAE config, and per-predictor significance.

**Step 3: Absorption taxonomy.**
Operationalize three absorption types using Gemma Scope 16k at layer 12:
- Type I (Full): Single absorbing latent suppresses parent; Chanin metric > threshold
- Type II (Partial): Parent latent fires at < 50% of expected magnitude; absorbing latent partially compensates (measured via normalized activation magnitude on expected-parent-token set)
- Type III (Distributed): Parent suppressed; top-3 child latents by α_ij collectively explain > 80% of parent's firing deficit (measured via DAS(k=3))

Compare: what fraction of true absorption cases are Type I, II, III respectively? This shows whether 15–35% understates the true scope.

---

### Component 3: Downstream Impact Analysis (RQ3, H3)

**Stage 1 — SAEBench correlation (zero GPU-hours):**
Download SAEBench CSV from neuronpedia.org. Compute Pearson and Spearman correlations between the absorption_score column and each of: RAVEL, SCR, sparse probing F1, unlearning performance. Apply Bonferroni correction for 4 tests. Pre-specified threshold for "meaningful" correlation: |r| > 0.3.

**Stage 2 — Matched comparison (2 GPU-hours):**
From SAEBench, select top-5 lowest absorption SAEs and top-5 highest absorption SAEs matched on model=Gemma 2 2B, layer∈{12,20}. Run RAVEL task directly on each matched pair. Paired t-test, one-sided, α=0.05.

**Stage 3 — Safety probe pilot (1 GPU-hour):**
Select 3 SAEs (lowest, median, highest absorption from Gemma 2 2B SAEBench). Build binary classification task: 50 harmful (AdvBench) vs. 50 benign (OpenWebText) prompts. Train 1-sparse SAE probe (ridge regression with L0=1 constraint) vs. dense linear probe on residual stream. Report ROC-AUC by absorption level.

**Controlled confound identification:**
Empiricist-guided caution: run the pilot L0 check first (Component 0) to determine whether width comparisons require L0 covariate control. Report all absorption rates with per-letter breakdown and seed-level variance where Gemma Scope provides multiple checkpoints.

---

## Experimental Plan

| Experiment | Component | Time Est. | GPU | Key Output |
|---|---|---|---|---|
| Pilot: pipeline validation + L0 check | 0 | 15 min | 1× A100 | Go/no-go gate for controlled comparison |
| LV coefficient computation + cosine filter | 1 | 30 min | 1× A100 | α_ij scores for all candidate pairs |
| Absorption detector validation (F1 vs. ground truth) | 1 | 20 min | 1× A100 | Precision/recall/F1 of LV detector |
| Cross-architecture validation | 1 | 20 min | 1× A100 | Architecture generalization of LV detector |
| Width paradox — DAS(k=1,3) | 1 | 30 min | 1× A100 | DAS vs. width relationship |
| PMI extraction from OpenWebText | 2 | 30 min | CPU | PMI features for regression |
| 30-SAE absorption survey | 2 | 8 hrs | 1× A100 | Main regression dataset |
| Absorption taxonomy operationalization | 2 | 30 min | 1× A100 | Type I/II/III rates |
| SAEBench correlation (Stage 1) | 3 | 1 hr | None | Correlation matrix |
| Matched RAVEL comparison (Stage 2) | 3 | 2 hrs | 1× A100 | Downstream impact evidence |
| Safety probe pilot (Stage 3) | 3 | 1 hr | 1× A100 | SAE vs. dense probe safety gap |

**Total GPU-hours: ~14 hours on a single A100 40GB.**

### Ablation Schedule

| Ablation | Tests | Success Criterion |
|---|---|---|
| τ sensitivity: LV threshold τ ∈ {0.5, 0.75, 1.0, 1.25, 1.5} | Robustness of LV detector | F1 peak within τ ∈ {0.75, 1.25} |
| Decoder cosine pre-filter threshold (0.10, 0.15, 0.25) | Coverage vs. precision tradeoff | ≥ 80% of absorbed pairs captured at τ=0.15 |
| PMI regression without SAE config terms | How much variance PMI alone explains | Partial R² ≥ 0.10 after controlling for config |
| Per-layer regression stability | Is scaling law layer-invariant? | Slope (PMI) same sign across all layers |
| Absorption type breakdown by letter | Type I/II/III mix by letter | Shows comprehensive absorption > Chanin metric |

---

## Pilot Design (Critical < 15 min)

Load Gemma Scope Gemma 2 2B layer 12 SAEs at widths 16k and 65k. Simultaneously:
1. Confirm sae-spelling absorption pipeline reproduces ~15–35% for letters A–E (pipeline validity check)
2. Measure empirical L0 for both widths on 500-token sample (determines whether width-controlled comparison is feasible)
3. Compute pairwise α_ij for top-100 candidate pairs by decoder cosine similarity on 1k activations; manually check whether top-10 α_ij > 1 pairs correspond to known parent-child letter features (LV sanity check)

**Decision rule:** If (1) passes and α_ij sanity check shows letter-feature pairs in top-10 → proceed with full Component 1. If L0 matching fails → proceed with within-width L0 sweep only for Component 2 regression.

---

## Novelty Assessment

### Search Results (conducted April 2026)

**LV / competitive exclusion + SAE:** Zero arXiv results. "The Geometry of Concepts" (2025) mentions the niche analogy in one sentence without formalization. The LV competitive exclusion framing is **unambiguously novel**.

**Probe-free absorption detection:** The LessWrong post "Looking for Feature Absorption Automatically" reports a negative result for cosine similarity of *ablation effect vectors* (not decoder cosine + co-activation statistics). Wu et al. (arXiv:2502.15576) uses mutual information for SAE feature explanations but not for absorption detection. The competition coefficient α_ij as an unsupervised absorption predictor is **novel**.

**Corpus PMI as absorption predictor:** No paper directly tests whether training corpus co-occurrence statistics predict which specific feature pairs are absorbed. Masked regularization (arXiv:2604.06495) disrupts co-occurrence patterns but does not show corpus statistics predict absorption beforehand. **Novel**.

**Downstream causal chain test:** SAEBench (arXiv:2503.09532) provides raw data; DeepMind (2025) reports a qualitative negative finding; but no paper runs the systematic correlation analysis across 200+ SAEs with Bonferroni correction and a pre-specified minimum effect size. **Novel**.

**Distributed absorption / width paradox:** No paper proposes DAS(k=3) or measures distributed vs. concentrated absorption across widths. **Novel**.

---

## Revisions from Prior Feedback

*(This is the first iteration; no prior proposal or pilot evidence exists. This section will be populated in future rounds.)*

---

## Evidence-Driven Revisions

*(No pilot evidence yet. The proposal is designed so that the 15-min pilot provides the first piece of evidence that will directly inform whether the controlled-width ablation in Component 2 proceeds or redirects to within-width sweep only.)*

---

## Risk Assessment

| Risk | Probability | Mitigation |
|---|---|---|
| Pilot L0 check fails (widths not matched) | 50% | Redirect Component 2 to within-width L0 sweep only; still fully valid for H2 |
| LV detector F1 < 0.50 (detector fails) | 25% | α_ij is still a quantitative predictor; report its Pearson r with absorption rate as a positive partial result; investigate whether the threshold is non-sharp (smooth monotone) |
| PMI regression has R² < 0.10 for PMI term | 35% | Report as null result supporting conventional view; absorption taxonomy (Type I/II/III) remains valuable independent contribution |
| SAEBench correlation shows r < -0.3 (H3 falsified) | 30% | This is the *best* outcome for the field — H3 falsification means absorption does matter for downstream tasks; reframe as confirming the research motivation and providing the first empirical proof |
| ATM SAE checkpoints unavailable | 40% | Use OrtSAE diagnostic (decoder cosine similarity distribution) as proxy instead; report ATM replication as future work |
| DAS(k=3) does not increase with width (H4 falsified) | 40% | Report width paradox as unexplained; the L0/D confound analysis from Component 0 pilot provides a competing explanation |

### What Would Falsify the Core Hypothesis

The research is structured so that no single negative result kills the paper:
- If H1 fails (LV detector F1 < 0.50): the α_ij quantity still has descriptive value; the LV framing is wrong but the corpus PMI finding (H2) and downstream analysis (H3) stand independently.
- If H2 fails (PMI not predictive): the LV detector and downstream analysis are unaffected; we conclude absorption is primarily objective-driven, not data-driven.
- If H3 is falsified (|r| > 0.3): this is a stronger positive result — absorption does predict downstream performance, which validates the entire research motivation.
- All three hypotheses failing simultaneously would mean: (a) α_ij is not predictive, (b) corpus statistics are not predictive, and (c) absorption metric does predict downstream performance. This would be a highly surprising and publishable finding in itself.

---

## Baselines

1. **Cosine-similarity-only absorption detector** (cosine > threshold): tested by the LessWrong post to perform poorly; our competitor; used as ablation of the full LV coefficient.
2. **Chanin et al. probe-directed metric** (arXiv:2409.14507): ground truth against which the LV detector is validated; our primary ground truth.
3. **SAEBench absorption scores** (arXiv:2503.09532): pre-computed for 200+ SAEs; used for downstream correlation analysis.
4. **Feature hierarchy depth as absorption predictor**: compared against corpus PMI in regression.
5. **Dense linear probe performance**: upper bound on downstream task performance; compared against SAE 1-sparse probe in Component 3 Stage 3.

---

## Target Venue

**NeurIPS 2026** (Mechanistic Interpretability track) or **ICLR 2027**.

Length: 8–10 pages + appendix.

Primary claim: "We present the first unified empirical study of SAE feature absorption that simultaneously provides an unsupervised detector, a corpus-level causal account, and a direct test of downstream impact — using only pre-trained Gemma Scope SAEs, no retraining required."

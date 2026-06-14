# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **[Chanin et al., 2024/2025. "A is for Absorption" (arXiv:2409.14507)]** — Foundational work defining feature absorption; validates across hundreds of SAEs. Code via SAEBench. **Code: Yes**

2. **[Karvonen et al., 2025. "SAEBench" (ICML 2025)]** — Standardized benchmark with 8 metrics including absorption. Evaluates 200+ SAEs. **Code: Yes (GitHub: adamkarvonen/SAEBench)**

3. **[Korznikov et al., 2026. "Sanity Checks for SAEs" (arXiv:2602.14111)]** — Frozen/random SAE baselines match trained SAEs on key metrics. Only 9% true feature recovery. **Code: Unknown**

4. **[Wang et al., ICLR 2026. "Does Higher Interpretability Imply Better Utility?" (arXiv:2510.03659)]** — Weak correlation (tau_b ~ 0.3) between interpretability and steering utility. **Code: Unknown**

5. **[Olmo et al., NAACL 2025 Findings. "Features that Make a Difference"]** — Gradient SAEs (g-SAEs) incorporate downstream gradient information; more effective for steering. **Code: Unknown**

6. **[Gadgil et al., 2025. "Ensembling Sparse Autoencoders" (arXiv:2505.16077)]** — Independently initialized SAEs share only 30-42% of features. Boosting on residuals discovers complementary features. **Code: Unknown**

7. **[SAELens (GitHub: decoderesearch/SAELens)]** — Mature library for loading pretrained SAEs. MIT license. **Code: Yes**

8. **[GemmaScope (HuggingFace: google/gemma-scope)]** — Comprehensive JumpReLU SAE suite for Gemma 2 (2B/9B/27B). Apache-2.0. **Code: Yes (via SAELens)**

9. **[IBM/sae-steering (GitHub)]** — Layer-level steering in LLMs using SAEs. Production-ready code. **Code: Yes**

10. **[SynthSAEBench (Chanin et al., 2026, arXiv:2602.14687)]** — Large-scale synthetic benchmark with ground-truth feature directions. Enables controlled validation. **Code: Unknown**

11. **[MIT Thesis (2025). "Evaluation of SAE-based Refusal Features"]** — Critical finding: SAE feature dictionaries transfer but causal meaning does not. **Code: N/A**

12. **[Bussmann et al., ICML 2025. "Matryoshka Sparse Autoencoders" (arXiv:2503.17547)]** — Nested dictionaries reduce absorption from 0.49 to 0.05 but introduce hedging trade-off. **Code: Unknown**

### Landscape Summary

The SAE field is in a credibility crisis with three converging lines of evidence: (1) GDM/DeepMind negative results on downstream tasks, (2) Korznikov et al.'s random baseline challenge, and (3) Wang et al.'s weak interpretability-utility correlation (~0.3). The key practical question is no longer "how do we reduce absorption?" but "does absorption actually matter for anything we care about?"

No existing paper systematically correlates absorption rates with steering effectiveness or probing accuracy on real LLM SAEs. The current project occupies a genuinely novel niche. However, the experimental results are predominantly null, creating a framing and publication challenge.

The most practical engineering insight from the literature: g-SAEs (Olmo et al.) explicitly address the downstream impact problem by incorporating gradient information, suggesting that the field's focus on activation-based metrics may be fundamentally misaligned with practical utility.

---

## Phase 2: Initial Candidates

### Candidate A: The Null Result as Methodological Contribution

- **Hypothesis**: The absence of correlation between absorption and downstream tasks is itself a meaningful finding that reframes how the community should evaluate SAEs. The paper's contribution is methodological (first systematic correlation study + critical controls) rather than empirical (discovering a new phenomenon).
- **Implementation sketch**: Use all existing experimental data from GPT-2 Small (layers 0/4/8/10, 26 first-letter features). Add power analysis, multiple comparison correction, and random baseline controls. Frame as "we tested the assumption and found it unsupported."
- **Simplest version**: The data already exists. Write the paper with honest null results. 0 GPU-hours.
- **Time estimate**: 0 GPU-hours. 15-20 hours writing.
- **Reusable components**: All existing data: correlation_report_full.json, ec50_analysis.json, precision_recall_analysis.json, cross_model_pythia_combined.json.

### Candidate B: Cross-Model Validation with Gemma-2-2B

- **Hypothesis**: The null result on GPT-2 Small may be model-specific. Gemma-2-2B (with JumpReLU SAEs from GemmaScope) may show different absorption-task relationships due to different architecture, training data, and SAE configuration.
- **Implementation sketch**: Load GemmaScope SAEs for Gemma-2-2B (layer 8, 16K or 65K latents). Run identical first-letter absorption detection + steering + probing protocol. Compare correlation strengths with GPT-2 Small results.
- **Simplest version**: Load one GemmaScope SAE (layer 8, 16K), detect absorption on first-letter features, run steering at 3 strengths, compute correlation with absorption rate. Target: ~1 GPU-hour.
- **Time estimate**: ~2-3 GPU-hours for absorption + steering + probing on Gemma-2-2B. 2-3 hours analysis.
- **Reusable components**: SAELens loader, existing steering/probing code, first-letter feature detection pipeline.

### Candidate C: Absorption-Aware Feature Selection for Practical Steering

- **Hypothesis**: Even if absorption does not correlate with raw steering success, absorption-aware feature selection (avoiding highly absorbed features) may improve steering reliability in practice, particularly for precision-sensitive tasks.
- **Implementation sketch**: Use the existing absorption data to split features into HIGH/LOW absorption groups. Compare steering consistency (variance across prompts) and robustness (adversarial perturbations) between groups. Test whether LOW-absorption features produce more reliable steering.
- **Simplest version**: Split 26 features into HIGH (>=10%) and LOW (<5%) absorption. Compare steering success variance and adversarial robustness. Target: ~30 minutes analysis on existing data.
- **Time estimate**: ~0.5 GPU-hours (mostly inference for adversarial testing). 2-3 hours analysis.
- **Reusable components**: Existing absorption rates, steering results, test prompts.

---

## Phase 3: Self-Critique

### Against Candidate A (Null Result Framing)

- **Implementation reality check**: Null result papers are publishable but face high scrutiny. The methodological contribution must be genuinely valuable. The random baseline control and precision-recall decomposition are substantive methodological innovations.
- **Reproducibility attack**: The study is highly reproducible --- all code is training-free, uses open-source models, and fixed seeds. However, n=26 features is underpowered. Power analysis shows only 64% power to detect |r| >= 0.50.
- **Baseline sanity check**: The random baseline finding (some random directions achieve 100% steering success) is itself a significant contribution that challenges simple assumptions about steering.
- **Scope attack**: Single model (GPT-2 Small) is a major limitation. Pythia-70M cross-validation was inconclusive due to model size.
- **Verdict**: STRONG --- The null result is honest, the methodological controls are rigorous, and the framing as "absorption coexists with functional performance" is intellectually coherent. Best combined with Candidate B for cross-model validation.

### Against Candidate B (Cross-Model Validation)

- **Implementation reality check**: SAELens supports GemmaScope loading. Gemma-2-2B fits on a single GPU. The first-letter feature detection pipeline is model-agnostic. This is technically feasible.
- **Reproducibility attack**: JumpReLU SAEs have different dictionary sizes and sparsity levels than ReLU SAEs. Direct comparison requires careful normalization. The first-letter feature set may not be equally detectable in Gemma-2-2B.
- **Baseline sanity check**: If Gemma also shows null results, this strengthens the "absorption is benign" claim. If Gemma shows significant correlation, the null result is model-specific and the paper needs reframing.
- **Scope attack**: Two models is still limited. A truly systematic comparison would need 3+ model families.
- **Verdict**: MODERATE --- Feasible and potentially decisive, but adds GPU-hours and may not change the conclusion. Best as a supplementary validation, not the primary contribution.

### Against Candidate C (Absorption-Aware Selection)

- **Implementation reality check**: The analysis can be done on existing data for variance comparison. Adversarial robustness testing requires new inference but is straightforward.
- **Reproducibility attack**: Steering consistency is not a standard metric. Defining and justifying it requires careful methodology.
- **Baseline sanity check**: If LOW-absorption features are not more consistent, this further supports the null. If they are, it provides a practical recommendation despite the raw correlation being null.
- **Scope attack**: Consistency on first-letter steering may not generalize to other tasks.
- **Verdict**: MODERATE --- Interesting practical angle but adds complexity. The variance analysis can be done on existing data quickly.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (absorption-aware selection)** is dropped as a primary contribution. The variance analysis can be included as a supplementary analysis but does not warrant being the main claim.

### Strengthened Ideas

- **Candidate A (null result framing)** is strengthened by:
  1. Adding explicit power analysis (already done: 64% power for |r| >= 0.50).
  2. Adding multiple comparison correction (already done: Bonferroni, BH-FDR).
  3. Adding the precision-recall decomposition (H5 supported: precision invariant, recall varies).
  4. Adding the steering-encoder confound discussion as a methodological insight.
  5. Adding the random baseline heterogeneity finding as a secondary contribution.

- **Candidate B (cross-model validation)** is retained as a **pilot experiment** to test generalizability. If Gemma shows similar null results, the paper's claim strengthens. If not, the paper acknowledges model-specific effects.

### Additional Searches Conducted

- Searched for "SAE null result publishable" --- confirmed that null results are increasingly accepted in ML when methodological rigor is high and the question is important.
- Searched for "GemmaScope first letter features" --- confirmed first-letter features are detectable in Gemma models via SAELens.
- Searched for "SAE steering variance consistency" --- no direct matches, suggesting this is an unexplored angle.

### Selected Front-Runner

**Candidate A (primary) + Candidate B (pilot validation): The Methodological Null Result with Cross-Model Validation**

This is the strongest path because:
1. It requires zero new experiments for the primary claim (all data exists).
2. The cross-model pilot adds only ~2 GPU-hours but potentially strengthens generalizability.
3. The methodological contributions (random baseline, precision-recall decomposition, power analysis) are genuinely novel and useful to the community.
4. It honestly reports null results while still making a valuable contribution.
5. It addresses the credibility crisis directly --- showing that absorption metrics may be decoupled from practical utility.

---

## Phase 5: Final Proposal

### Title

**"Does Feature Absorption Matter? A Systematic Study Finds No Consistent Downstream Impact"**

Alternative: **"Feature Absorption and Downstream SAE Reliability: A Methodological Study with Null Results"**

### Hypothesis

Feature absorption in Sparse Autoencoders has been assumed to degrade downstream interpretability tasks, but this assumption has never been systematically tested. We hypothesize that:

1. **H1 (Primary)**: No significant correlation exists between absorption rate and raw steering success after appropriate statistical controls.
2. **H1b (Secondary)**: Delta-corrected steering (subtracting random baseline) may reveal subtle effects at specific layers but not consistently.
3. **H2 (Secondary)**: No significant correlation exists between absorption rate and sparse probing F1.
4. **H3 (Exploratory)**: The relationship, if any, is layer-dependent and inconsistent across model families.

### Motivation

The SAE field faces a credibility crisis. Multiple independent lines of evidence suggest that SAE structural metrics may be decoupled from downstream task performance. Yet the field continues to optimize absorption reduction without empirical justification. This study provides the first systematic test of the core assumption, with rigorous methodological controls.

### Method

**Phase 1: Absorption Detection (COMPLETED)**
- Load pretrained SAE via SAELens (GPT-2 Small, gpt2-small-res-jb, layers 0/4/8/10).
- Run Chanin et al. differential correlation absorption metric on 26 first-letter features (A-Z).
- Classify features into HIGH (>=10%), MEDIUM (5-10%), LOW (<5%) absorption categories.

**Phase 2: Feature Steering (COMPLETED)**
- Extract decoder directions for each first-letter feature.
- Generate test prompts (100 per letter).
- Run steering at strengths [1, 2, 5, 10, 20, 50].
- Measure binary success rate and continuous probability lift.
- **Critical control**: Random feature baseline (26 random latents) for delta-corrected metrics.

**Phase 3: Sparse Probing (COMPLETED)**
- Train k-sparse linear probes (k=1, 5, 10, 20) on first-letter classification.
- Decompose F1 into precision and recall.
- Compare absorbed vs non-absorbed features.

**Phase 4: Statistical Analysis (COMPLETED)**
- Compute Pearson/Spearman correlations between absorption rate and task performance.
- Test raw metrics, delta-corrected metrics, precision, recall separately.
- Apply Bonferroni and BH-FDR multiple comparison correction.
- Compute statistical power for n=26, alpha=0.05.
- Test cross-layer consistency (H3).

**Phase 5: Cross-Model Pilot (PROPOSED)**
- Load GemmaScope SAE for Gemma-2-2B (layer 8, 16K latents).
- Run identical absorption detection on first-letter features.
- Run steering at 3 strengths.
- Compute correlation and compare with GPT-2 Small results.

### Simplest Version

The absolute minimum experiment that tests the core claim has already been completed: Load GPT-2 Small SAE (layer 8), detect absorption on 26 first-letter features, run steering with random baseline, train k=5 sparse probes, compute correlations. This was a single-afternoon experiment (~3 GPU-hours) that produced null results.

The proposed cross-model pilot adds: Load Gemma-2-2B SAE (layer 8), detect absorption, run steering, compute correlation. Target: ~1 GPU-hour.

### Baselines

1. **Random feature baseline**: 26 random SAE latents steered identically. Expected near-zero success. Actual: highly heterogeneous (0.0-1.0), some random directions achieve 100% success --- a finding that itself challenges simple baseline assumptions.

2. **No-absorption baseline**: Features with <5% absorption rate. Expected high steering success (>80%). Actual: highly variable (0.4-1.0), confirming that factors other than absorption dominate steering success.

3. **Multiple comparison correction**: With 12 statistical tests, one uncorrected p<0.05 result is expected by chance. H1b at layer 8 (p=0.028) does not survive Bonferroni (p=0.334) or BH-FDR (q=0.107).

4. **Power analysis**: With n=26 features, power to detect |r| >= 0.50 is approximately 64%. The study is underpowered for small effects.

### Experimental Plan

| Phase | What | Model/SAE | Metrics | Status |
|-------|------|-----------|---------|--------|
| Absorption detection | 26 features, 4 layers | GPT-2 Small res-jb | Absorption rate per feature | COMPLETED |
| Steering | 26 features, 2 layers, 6 strengths | GPT-2 Small | Success rate, prob lift | COMPLETED |
| Random baseline | 26 random latents | GPT-2 Small | Baseline success rate | COMPLETED |
| Sparse probing | k=1,5,10,20 probes | GPT-2 Small | F1, precision, recall | COMPLETED |
| Correlation | All hypotheses | Existing data | r, p, R^2, power | COMPLETED |
| Cross-model | Gemma-2-2B layer 8 | GemmaScope 16K | Absorption + steering | PROPOSED |

### Resource Estimate

- **GPU (completed)**: ~15 GPU-hours across all iterations.
- **GPU (proposed pilot)**: ~1-2 GPU-hours for Gemma-2-2B cross-validation.
- **Remaining work**: Writing revision + potential Gemma pilot. 0-2 GPU-hours.
- **Model sizes**: GPT-2 Small (124M), Gemma-2-2B (2B).
- **Storage**: <10GB.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Zero significant results after correction | HIGH (already confirmed) | HIGH | Frame as null-result contribution. Emphasize methodological insights. |
| Steering-encoder confound undermines central test | HIGH | HIGH | Explicitly discuss as limitation and methodological insight. |
| Korznikov random baseline challenges premise | HIGH | HIGH | Address directly in Related Work. Position study as testing a specific metric's validity. |
| Single model limits generalizability | HIGH | MEDIUM | Acknowledge. Proposed Gemma pilot. Recommend future work. |
| Paper rejected as "we found nothing" | MEDIUM | HIGH | Strong framing: "We found that absorption's consequences are metric-specific and layer-dependent." |
| Underpowered study (n=26) | HIGH | MEDIUM | Add power analysis. Acknowledge limitation. Effect size is small even if real. |
| Gemma pilot shows different results | MEDIUM | MEDIUM | Report honestly. If Gemma null too, strengthens claim. If Gemma significant, reframes as model-specific. |

### Novelty Claim

This paper makes four novel contributions:

1. **First systematic correlation between absorption detection and downstream task performance** (steering + probing). No prior work measures whether absorption actually degrades the tasks that motivate SAE research.

2. **Methodological insight: baseline correction is essential**. The discovery that raw steering metrics show no correlation while delta-corrected metrics reveal a subtle effect (at one layer, uncorrected) is a critical methodological contribution.

3. **Precision-recall decomposition**. Absorption affects recall (coverage) but not precision (selectivity), revealing the mechanism by which absorption impacts (or fails to impact) downstream tasks.

4. **Power analysis guidance**. We provide the first power analysis for absorption-downstream correlation studies, showing that n=26 features is underpowered for small-to-medium effects.

No existing paper combines these elements. The honest reporting of null results, combined with rigorous methodological controls, provides actionable guidance for the SAE community.

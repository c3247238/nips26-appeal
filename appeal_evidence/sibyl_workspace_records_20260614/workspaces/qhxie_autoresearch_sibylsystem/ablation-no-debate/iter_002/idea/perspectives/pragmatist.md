# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **SAELens** ([github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) — Apache 2.0, 1,100+ stars. The de facto standard for SAE training and analysis. Supports loading pretrained Gemma Scope and GPT-2 SAEs with a single `SAE.from_pretrained()` call. Essential infrastructure. **Code exists and is actively maintained.**

2. **Gemma Scope SAEs** ([arXiv:2408.05147](https://arxiv.org/abs/2408.05147)) — Apache 2.0. Full-layer JumpReLU SAEs for Gemma 2 2B/9B (16k/65k/1m width, 27 layers). Dominant experimental platform. Available via SAELens and HuggingFace. **Weights freely available.**

3. **GPT-2 Small SAEs** (`gpt2-small-res-jb` release in SAELens) — Pretrained residual-stream SAEs for all 12 layers of GPT-2 small. Smaller, faster, ideal for pilots. **Weights freely available.**

4. **sae-spelling** ([github.com/lasr-spelling/sae-spelling](https://github.com/lasr-spelling/sae-spelling)) — MIT license. Official code for "A is for Absorption" (NeurIPS 2025). Contains `FeatureAbsorptionCalculator`, probing utilities, and first-letter classification experiments. **Directly reusable for absorption rate baselines.**

5. **SAEBench** ([github.com/adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench)) — 8-metric benchmark suite including absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, CE loss. Tested on 200+ SAEs. `pip install sae-bench`. **Essential for validation.**

6. **SAE-Sensitivity** ([github.com/nathanhu0/sae-sensitivity](https://github.com/nathanhu0/sae-sensitivity)) — Official code for Hu et al. (2025) "Measuring Sparse Autoencoder Feature Sensitivity." Uses GPT-4.1-mini to generate similar texts and measure feature activation consistency. **Reusable for sensitivity analysis.**

7. **SynthSAEBench-16k** ([decoderesearch/synth-sae-bench-16k-v1](https://huggingface.co/datasets/decoderesearch/synth-sae-bench-16k-v1)) — Ground-truth synthetic benchmark with 16k features, hierarchy, correlation, superposition. **Useful for validating training-free metrics against known ground truth.**

8. **TransformerLens** ([github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)) — BSD license. Hook-based activation extraction. Integrates seamlessly with SAELens. **Essential for activation extraction.**

9. **"A is for Absorption"** ([arXiv:2409.14507](https://arxiv.org/abs/2409.14507)) — Chanin et al., NeurIPS 2025. First systematic definition of feature absorption. Toy model (4 features, 50D) proves hierarchy + sparsity causes absorption. Validated on hundreds of LLM SAEs. **Limitation: ablation-dependent metric only works for layers 0-17; conservative underestimate.**

10. **"Feature Hedging"** ([arXiv:2505.11756](https://arxiv.org/abs/2505.11756)) — Chanin & Garriga-Alonso, 2025. Identified encoder-decoder asymmetry as the diagnostic signature of absorption (vs. symmetric mixing in hedging). **Key insight for training-free detection.**

11. **"Measuring Sparse Autoencoder Feature Sensitivity"** ([arXiv:2509.23717](https://arxiv.org/abs/2509.23717)) — Hu et al., 2025. Found sensitivity declines with SAE width; many interpretable features have poor sensitivity. **Complementary metric to absorption.**

12. **"SAEBench"** ([arXiv:2503.09532](https://arxiv.org/abs/2503.09532)) — Karvonen et al., ICML 2025. Revealed proxy metrics (reconstruction, L0) do not predict practical interpretability. **Validates need for better metrics.**

### Landscape Summary

The field has matured rapidly. Feature absorption is now a well-defined phenomenon with standardized metrics (absorption rate, mean absorption fraction, full absorption rate). The community has moved beyond visualization to quantitative evaluation. Multiple architectural solutions exist (Matryoshka, HSAE, OrtSAE, AdaptiveK), but all require training new SAEs. **The critical gap is training-free detection and quantification in existing pretrained SAEs** — exactly what this project targets.

Key practical insight: the original absorption metric requires causal ablation experiments that fail for deep layers (>17 in Gemma 2 2B). A training-free alternative would unlock analysis across all layers and all pretrained SAEs, dramatically expanding applicability.

---

## Phase 2: Initial Candidates

### Candidate A: Training-Free Absorption Detection via Encoder-Decoder Asymmetry

- **Hypothesis**: Absorbed features exhibit a characteristic encoder-decoder asymmetry (decoder tracks combined direction, encoder develops a "hole"). This asymmetry can be quantified without ablation experiments, enabling training-free absorption detection across all layers.
- **Implementation sketch**: Start from `sae-spelling` repo. Load Gemma Scope or GPT-2 SAEs via SAELens. For each latent, compute encoder-decoder cosine similarity. For latents identified as tracking hierarchical features (via first-letter probes or auto-interpretability), check if encoder direction diverges from decoder direction. Quantify divergence as an absorption signal.
- **Simplest version**: Load a single GPT-2 small SAE (layer 8). Compute encoder-decoder cosine similarity matrix. Identify latents with low self-similarity. Cross-reference with first-letter probe directions to see if low-similarity latents track hierarchical features. **Pilot: 10-15 minutes on a single GPU.**
- **Time estimate**: Full experiment (~10 SAEs across layers) = 2-4 GPU-hours. Analysis and validation = 1-2 GPU-hours. Total: **~4-6 GPU-hours**.
- **Reusable components**: SAELens (loading), sae-spelling (probing), TransformerLens (activations), SAEBench (validation).

### Candidate B: Absorption-Sensitivity Joint Analysis and Unified Quality Score

- **Hypothesis**: Low feature sensitivity (Hu et al.) and high absorption rate (Chanin et al.) are correlated — absorbed features are disproportionately insensitive. A unified score combining both metrics better predicts true feature quality than either alone.
- **Implementation sketch**: Run both absorption rate and feature sensitivity on the same set of SAE features. Use `sae-spelling` for absorption, `sae-sensitivity` for sensitivity. Compute correlation. Train a simple logistic regression or compute a weighted sum to create a unified "feature reliability score." Validate against SAEBench ground-truth metrics.
- **Simplest version**: Pick 26 first-letter features from a single GPT-2 SAE. Measure absorption rate (using existing code) and sensitivity (using LLM-generated similar texts, or a lightweight proxy). Compute Pearson correlation. **Pilot: 15-20 minutes.**
- **Time estimate**: Full joint measurement on 3-5 SAEs = 3-5 GPU-hours. Sensitivity measurement requires LLM API calls (GPT-4.1-mini, cheap). Analysis = 1 GPU-hour. Total: **~5-7 GPU-hours**.
- **Reusable components**: sae-spelling (absorption), sae-sensitivity (sensitivity code), SAEBench (validation), OpenAI API (text generation).

### Candidate C: Strong Baseline — "Absorption Is Worse Than We Thought: A Large-Scale Replication"

- **Hypothesis**: The original absorption paper (Chanin et al.) used a conservative metric limited to early layers. A more comprehensive replication using multiple architectures, all verifiable layers, and larger sample sizes will show absorption is more prevalent than previously reported.
- **Implementation sketch**: Use SAEBench's standardized absorption metric (which extends the original to more layers) and run it systematically across all available Gemma Scope SAEs (16k and 65k, all layers where ablation works). Compare with published numbers. Add cross-architecture comparison (TopK vs JumpReLU vs ReLU) using existing SAEBench data where possible.
- **Simplest version**: Run SAEBench absorption eval on 3 Gemma Scope SAEs (layer 0, 8, 16) and compare results to published baselines. **Pilot: 20-30 minutes per SAE (~90 min total).**
- **Time estimate**: Systematic eval across 20+ SAE configurations = 10-15 GPU-hours. This exceeds the 1-hour-per-task guideline and should be split into parallel subtasks (one per SAE). Total: **~15 GPU-hours split across multiple runs.**
- **Reusable components**: SAEBench (full evaluation suite), Gemma Scope SAEs (pretrained), SAELens (loading).

---

## Phase 3: Self-Critique

### Against Candidate A: Training-Free Absorption Detection

- **Implementation reality check**: The encoder-decoder asymmetry insight comes from Chanin et al.'s toy model and the "Feature Hedging" paper. Both are theoretical/analytical. No one has yet built a practical training-free detector. The asymmetry may be subtle in real SAEs (noisy, overlapping features, superposition) and hard to distinguish from normal encoder-decoder misalignment. **Risk: signal too weak to be useful.**
- **Reproducibility attack**: The metric depends on identifying "hierarchical features" to check for asymmetry. How do we reliably identify hierarchical features without human labels? First-letter probes work for spelling but not for abstract semantic hierarchies. The metric may be limited to probe-defined features. **Risk: narrow applicability.**
- **Baseline sanity check**: The baseline is the original ablation-dependent absorption rate. If our training-free metric doesn't correlate strongly with the ablation metric on layers 0-17 (where both work), it's not a valid proxy. We must validate correlation. **Risk: weak correlation with ground truth.**
- **Scope attack**: Encoder-decoder asymmetry might detect absorption-like phenomena but could also flag other issues (e.g., dead features, feature hedging). Need careful disambiguation. **Risk: low specificity.**
- **Verdict**: **MODERATE**. High novelty if it works, but significant risk of weak signal. Best approached as a pilot-first idea.

### Against Candidate B: Absorption-Sensitivity Joint Analysis

- **Implementation reality check**: Both absorption and sensitivity are established metrics with open-source code. Running them jointly is straightforward engineering. The `sae-sensitivity` repo uses GPT-4.1-mini for text generation, which is cheap and fast. **Low implementation risk.**
- **Reproducibility attack**: Sensitivity measurement requires LLM-generated similar texts, which introduces stochasticity. The paper reports human validation showing generated texts are genuinely similar, but reproducibility depends on API consistency. Fixed seeds and temperature=0 help. **Moderate risk.**
- **Baseline sanity check**: The baselines are the individual absorption rate and sensitivity scores. A unified score must outperform both in predicting some ground-truth quality measure (e.g., SAEBench's full absorption rate, or human interpretability ratings). Without a validation target, the unified score is arbitrary. **Need clear validation protocol.**
- **Scope attack**: Correlation between absorption and sensitivity might be strong only for first-letter features (the original absorption task) and weak for general features. Testing on multiple feature types is essential. **Risk: limited generality.**
- **Verdict**: **STRONG**. Low implementation risk, clear validation path via SAEBench, directly addresses a stated research gap (Gap 5 in literature survey). The main value is empirical — even if correlation is weak, publishing the joint measurement is a contribution.

### Against Candidate C: Large-Scale Replication Baseline

- **Implementation reality check**: SAEBench already provides the absorption metric and has been run on 200+ SAEs. A "replication" that simply runs existing code on more SAEs is not novel — it's a baseline study. The original paper's conservative metric is a known limitation; confirming it with SAEBench's extended metric is useful but incremental.
- **Reproducibility attack**: SAEBench is well-documented and reproducible. No issue here.
- **Baseline sanity check**: The baseline is the original Chanin et al. paper. If we only replicate with a better metric, we need to show something new (e.g., cross-architecture patterns, layer-wise trends, or interaction with SAE width/sparsity).
- **Scope attack**: Large-scale replication without a novel hypothesis risks being a "measurement paper" that reviewers dismiss as engineering rather than research. Needs a clear angle (e.g., "absorption increases monotonically with depth" or "JumpReLU reduces absorption by X% across all layers").
- **Verdict**: **WEAK** as a standalone idea. Useful as a baseline comparison within a larger study, but not sufficient as the core contribution. Should be a component of Candidate A or B, not the main idea.

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C** dropped as a standalone front-runner. It will be incorporated as a baseline comparison within the chosen idea.

### Strengthened Survivors

**Candidate B (Absorption-Sensitivity Joint Analysis) selected as front-runner.**

Reasoning:
1. **Lowest implementation risk**: Both metrics have working open-source code. We can run the pilot in under 20 minutes.
2. **Clear validation path**: SAEBench provides ground-truth absorption metrics. SynthSAEBench provides ground-truth features. We can validate our unified score against both.
3. **Directly addresses a stated gap**: Gap 5 in the literature survey — "Theoretical Understanding of Absorption-Sensitivity Relationship."
4. **Training-free constraint**: Both absorption rate (via SAEBench) and sensitivity are analysis-only, no SAE training required.
5. **Publishable even with negative results**: If absorption and sensitivity are uncorrelated, that's still a valuable finding that clarifies the relationship between two major SAE quality dimensions.

**Candidate A incorporated as a secondary analysis.**

The encoder-decoder asymmetry idea is high-risk but high-reward. We will run it as a pilot within the same project. If the pilot shows strong signal, it becomes a major component. If weak, we still have Candidate B as the solid core.

**Simplification:**
- Focus on **GPT-2 small** and **Gemma 2 2B** (16k width) — both have pretrained SAEs in SAELens.
- Limit to **layers 0-12** for pilot (where ablation works, enabling ground-truth validation).
- Use **first-letter features** as the primary test case (well-defined, probe-available, established baseline).
- For sensitivity, use a **lightweight proxy** instead of full LLM generation for the pilot: measure activation variance across a fixed set of semantically similar texts (e.g., from a paraphrase dataset).

### Additional Searches Conducted
- Confirmed `sae-sensitivity` repo exists and uses GPT-4.1-mini (cheap, fast).
- Confirmed SAEBench absorption eval takes ~26 min per SAE on RTX 3090.
- Confirmed Gemma Scope SAEs load via `SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id="layer_X/width_16k/average_l0_Y")`.
- Confirmed GPT-2 SAEs load via `SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.X.hook_resid_pre")`.

---

## Phase 5: Final Proposal

### Title
**Bridging Absorption and Sensitivity: A Joint Analysis of Sparse Autoencoder Feature Reliability**

### Hypothesis
Feature absorption rate and feature sensitivity are correlated failure modes in sparse autoencoders. A unified reliability score combining both metrics better predicts true feature quality (as measured by ground-truth benchmarks) than either metric alone.

### Motivation
Sparse autoencoders (SAEs) are the primary tool for mechanistic interpretability, but two critical failure modes undermine their reliability:
1. **Feature absorption**: Seemingly interpretable features fail to activate on arbitrary positive examples because their direction has been "stolen" by co-occurring features (Chanin et al., NeurIPS 2025).
2. **Feature sensitivity**: Features activate inconsistently on semantically similar inputs, making them unreliable as interpretable classifiers (Hu et al., 2025).

These two metrics are studied in isolation. Their quantitative relationship is unknown. Understanding whether they measure the same underlying pathology or distinct failure modes is essential for building trustworthy SAE-based interpretability pipelines. This work systematically measures both on the same feature sets and develops a unified quality score validated against ground-truth benchmarks.

### Method

**Step 1: Foundation — Load SAEs and Extract Features**
- Load pretrained SAEs via SAELens:
  - GPT-2 small (`gpt2-small-res-jb`, layers 0, 4, 8, 11)
  - Gemma 2 2B (`gemma-scope-2b-pt-res`, layers 0, 4, 8, 12, 16)
- Extract residual stream activations on first-letter classification prompts (from `sae-spelling` dataset).

**Step 2: Measure Absorption Rate**
- Use `sae-spelling` `FeatureAbsorptionCalculator` or SAEBench absorption eval.
- Train logistic regression probes for first-letter classification (A-Z).
- Run k-sparse probing to detect feature splits.
- Perform integrated-gradients ablation on false-negative tokens.
- Compute absorption rate per letter, per SAE.

**Step 3: Measure Feature Sensitivity**
- For each first-letter feature identified in Step 2, collect top-activating examples.
- Generate semantically similar texts using GPT-4.1-mini (following Hu et al. protocol) or use a lightweight proxy (paraphrase dataset + embedding similarity filtering).
- Measure sensitivity as fraction of similar texts that activate the feature.

**Step 4: Joint Analysis**
- Compute Pearson/Spearman correlation between absorption rate and (1 - sensitivity) per feature.
- Stratify by: SAE architecture, layer depth, SAE width, L0 sparsity.
- Test hypothesis: low-sensitivity features are disproportionately absorbed.

**Step 5: Unified Quality Score**
- Train a simple logistic regression or compute a weighted combination: `reliability_score = w1 * (1 - absorption_rate) + w2 * sensitivity`
- Validate against:
  - SAEBench full absorption rate (ground truth for layers where available)
  - SynthSAEBench MCC/feature uniqueness (ground truth for synthetic features)
- Optimize weights to maximize correlation with ground-truth quality.

**Step 6: Secondary Analysis — Encoder-Decoder Asymmetry Pilot**
- For absorbed features identified in Step 2, compute encoder-decoder cosine similarity.
- Test if absorbed features show lower encoder-decoder alignment than non-absorbed features.
- If signal is strong, incorporate as a training-free proxy for absorption.

### Simplest Version
The absolute minimum experiment that tests the core claim:
1. Load **one GPT-2 small SAE** (layer 8, `blocks.8.hook_resid_pre`).
2. Identify **26 first-letter features** via logistic regression probes.
3. Measure **absorption rate** for each (using `sae-spelling` code).
4. Measure **sensitivity** for each (using 10 LLM-generated similar texts per feature).
5. Compute **correlation** between absorption rate and sensitivity.

**Expected pilot output**: A scatter plot of absorption rate vs. sensitivity for 26 features, with a correlation coefficient. If correlation > 0.3, the core hypothesis is supported. **Pilot time: 15-20 minutes.**

### Baselines

| Baseline | Description | Expected Performance |
|----------|-------------|---------------------|
| **Absorption rate alone** | Chanin et al. metric | Correlates moderately with ground-truth quality (SAEBench shows variance) |
| **Sensitivity alone** | Hu et al. metric | Correlates moderately; declines with width |
| **L0 sparsity** | Standard proxy metric | SAEBench showed proxy metrics do NOT predict practical performance |
| **Reconstruction R^2** | Standard proxy metric | Same issue — poor predictor of interpretability |

The unified score must outperform all baselines in predicting SAEBench/SynthSAEBench ground-truth quality.

### Experimental Plan

| Phase | Task | Datasets/Models | Metrics | Time |
|-------|------|-----------------|---------|------|
| **Pilot** | Single SAE (GPT-2 L8) | First-letter prompts | Absorption rate, sensitivity, correlation | 20 min |
| **Full** | 8 SAEs (GPT-2 + Gemma, 4 layers each) | First-letter prompts + paraphrases | Absorption rate, sensitivity, unified score | 4-6 hrs |
| **Validation** | SAEBench + SynthSAEBench | All available SAEs | Full absorption rate, MCC, feature uniqueness | 2-3 hrs |
| **Analysis** | Correlation, stratification, score optimization | Computed features | Pearson/Spearman, ROC-AUC | 1 hr |

**Ablation schedule:**
- Does correlation hold across architectures (ReLU vs TopK vs JumpReLU)?
- Does correlation vary by layer depth?
- Does the unified score generalize to non-first-letter features (e.g., auto-interpreted semantic features)?

### Resource Estimate

| Resource | Amount | Notes |
|----------|--------|-------|
| **GPU** | 1x NVIDIA GPU (8GB+ VRAM) | GPT-2 small fits on 8GB; Gemma 2 2B needs 12GB+ |
| **GPU-hours** | ~8 hours total | Split into 1-hour tasks |
| **LLM API** | ~$5-10 (GPT-4.1-mini) | For sensitivity text generation; optional if using proxy |
| **Storage** | ~10 GB | For activations, SAE weights, results |
| **Wall-clock** | ~2-3 days | Including analysis, plotting, writing |

All tasks fit within the **1-hour-per-task** guideline. The full experiment can be parallelized across SAEs.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Weak correlation** | Medium | High | Publish negative result; still valuable for field. Have secondary analysis (encoder-decoder asymmetry) as backup. |
| **SAEBench absorption eval too slow** | Low | Medium | Use `sae-spelling` lightweight implementation instead of full SAEBench suite. |
| **LLM API rate limits** | Low | Low | Use temperature=0, cache responses, fallback to paraphrase proxy. |
| **Gemma Scope loading issues** | Low | Low | GPT-2 small SAEs are reliable fallback. |
| **Sensitivity measurement noise** | Medium | Medium | Average over multiple generated texts per feature; use statistical tests. |
| **Reviewer: "just two existing metrics"** | Medium | High | Emphasize (1) first joint measurement, (2) unified score with validation, (3) practical impact on SAE quality assessment. |

### Novelty Claim

The novelty is **empirical and practical**, not architectural:
1. **First systematic joint measurement** of absorption and sensitivity on the same feature set.
2. **Unified reliability score** validated against ground-truth benchmarks, providing practitioners with a single number to assess SAE feature quality.
3. **Clarification of relationship**: Are absorption and sensitivity the same problem or different? Either answer is a contribution.
4. **Training-free**: All analysis runs on existing pretrained SAEs, making it immediately applicable to the community's existing SAE collections (Gemma Scope, GPT-2, Pythia).

Even if the correlation is weak, the paper establishes a methodological standard for joint SAE quality evaluation that future work can build on.

---

## Sources

- Chanin, D., et al. (2024/2025). "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507. NeurIPS 2025. [Link](https://arxiv.org/abs/2409.14507)
- Hu, N., et al. (2025). "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717. [Link](https://arxiv.org/abs/2509.23717)
- Karvonen, A., et al. (2025). "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability." arXiv:2503.09532. ICML 2025. [Link](https://arxiv.org/abs/2503.09532)
- Chanin, D., & Garriga-Alonso, A. (2025). "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756. [Link](https://arxiv.org/abs/2505.11756)
- Lieberum, T., et al. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." arXiv:2408.05147. [Link](https://arxiv.org/abs/2408.05147)
- SAELens: [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)
- sae-spelling: [github.com/lasr-spelling/sae-spelling](https://github.com/lasr-spelling/sae-spelling)
- SAEBench: [github.com/adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench)
- SAE-Sensitivity: [github.com/nathanhu0/sae-sensitivity](https://github.com/nathanhu0/sae-sensitivity)
- LessWrong — "Toy Models of Feature Absorption in SAEs": [lesswrong.com/posts/kcg58WhRxFA9hv9vN](https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN/toy-models-of-feature-absorption-in-saes)

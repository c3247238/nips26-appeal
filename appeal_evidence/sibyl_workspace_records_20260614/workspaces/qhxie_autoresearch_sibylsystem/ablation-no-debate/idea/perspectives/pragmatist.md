# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **SAELens (GitHub: jbloomAus/SAELens)** — Full SAE training/analysis framework; supports Gemma Scope, GPT-2, Pythia; Apache 2.0. Essential infrastructure. **Code exists**: Yes, comprehensive.

2. **Gemma Scope SAEs (DeepMind, arXiv:2408.05147)** — Gemma 2 full-layer SAEs (16k/65k/1m width, 27 layers); dominant experimental platform; Apache 2.0. **Code exists**: Yes, via HuggingFace.

3. **SAEBench (arXiv:2503.09532)** — 8-metric benchmark (absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, CE loss) on 200+ SAEs; proxy metrics don't predict practical performance. **Code exists**: Yes, github.com/ai-safety-institute/SAEBench.

4. **sae-spelling (GitHub: lasr-spelling/sae-spelling)** — Absorption rate metric implementation and toy models; MIT license. **Code exists**: Yes, well-documented.

5. **Chanin et al. 2024/2025 (arXiv:2409.14507)** — First systematic definition of feature absorption; absorption rate metric; proves hierarchy + sparsity inevitably causes absorption. **Code exists**: Partial (sae-spelling repo).

6. **SynthSAEBench (arXiv:2602.14687)** — Ground-truth synthetic benchmark (16k features, hierarchy, correlation, superposition). **Code exists**: Yes, HuggingFace dataset (decoderesearch/synth-sae-bench-16k-v1).

7. **JumpReLU SAE (Bricken et al. 2023/2025)** — Learnable per-feature thresholds; 0.0114 absorption vs TopK 0.1402; better sparsity-fidelity trade-off. **Code exists**: Yes, SAELens supports.

8. **Matryoshka SAE (arXiv:2503.17547)** — Nested dictionaries; reduced absorption 0.49→0.05; ~50% computational overhead. **Code exists**: Limited reference code available.

### Landscape Summary

The field has moved from architectural proliferation to evaluation maturity. SAEBench and SynthSAEBench provide standardized benchmarks, but key practical gaps remain:

**What works**: Training-free analysis is feasible on Gemma Scope SAEs using SAELens. Absorption rate varies dramatically by architecture (JumpReLU: 0.0114, TopK: 0.1402). Matryoshka and hierarchical SAEs reduce absorption but require custom training.

**What doesn't work**: Proxy metrics (reconstruction, L0) don't predict absorption. Simple ablation-based methods only work up to layer 17 in Gemma 2B. Pure visualization is insufficient for quantitative work.

**Practical gaps**:
- No training-free absorption detection method for deep layers (>17)
- No established protocol for measuring absorption impact on downstream steering
- Cross-architecture comparison at identical sparsity levels is incomplete

---

## Phase 2: Initial Candidates

### Candidate A: Anchor Feature Recovery via Decoder Direction Projection

- **Hypothesis**: Absorbed features leave detectable "echoes" in the decoder weight space. By projecting decoder directions onto orthogonal subspaces and measuring reconstruction residual, we can identify and partially recover absorbed features without retraining.
- **Implementation sketch**: Use SAELens to load Gemma Scope SAEs → compute decoder weight covariance matrix → identify low-rank subspaces where absorbed features "hide" → reconstruct feature directions from residuals → validate against ablation-based ground truth on layers 0-17.
- **Simplest version**: Measure decoder weight orthogonality score per feature. Test whether features with low orthogonality (more "absorbed" direction) correlate with higher absorption rates.
- **Time estimate**: ~30 min for pilot on single layer (layer 12, 16k width SAE). Full experiment across 18 layers: ~2-3 GPU-hours.
- **Reusable components**: SAELens for SAE loading, sae-spelling for absorption rate metric, Gemma Scope SAEs from HuggingFace.

### Candidate B: Cross-Architecture Absorption Pareto Analysis

- **Hypothesis**: Different SAE architectures (TopK, JumpReLU, Matryoshka, vanilla ReLU) represent different trade-off points on an absorption vs. reconstruction Pareto frontier. By characterizing this frontier, we can predict which architecture suits a given downstream task.
- **Implementation sketch**: Load SAEs of different architectures for the same model/layer via SAELens → measure absorption rate and reconstruction (R²) for each → plot Pareto frontier → identify architecture sweet spots for different task requirements.
- **Simplest version**: Compare TopK vs. JumpReLU absorption at matched L0 sparsity on Gemma 2B layer 12. Measure if JumpReLU's lower absorption translates to better steering success rates.
- **Time estimate**: ~20 min for pilot (2 architectures, 1 layer). Full: ~1-2 GPU-hours across 4 architectures, 5 layers.
- **Reusable components**: SAELens, Gemma Scope SAEs, SAEBench metrics.

### Candidate C: Safety-Critical Feature Absorption Audit

- **Hypothesis**: Safety-relevant features (deception, harm, bias) are disproportionately absorbed due to their hierarchical structure (general "unsafe behavior" vs. specific instances). This creates hidden risks in interpretability-based safety measures.
- **Implementation sketch**: Identify safety-related features via Neuronpedia or steering-based probes → measure their absorption rates using ablation → compare to non-safety features matched in activation frequency → quantify risk.
- **Simplest version**: Use 5 known safety feature indices from Neuronpedia (or steering-based identification) and 5 matched non-safety features. Compare absorption rates on Gemma Scope SAE layer 12.
- **Time estimate**: ~15 min for pilot (5+5 features, 100 samples). Full: ~45 min across 5 layers with 10 features each.
- **Reusable components**: Gemma Scope SAEs, sae-spelling absorption metric, Neuronpedia for feature identification.

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A: Anchor Feature Recovery

- **Implementation reality check**: Decoder direction analysis is a known technique (Marks et al. use it). The "recovery" aspect is novel but untested. Need to validate that recovered directions actually correspond to absorbed features, not noise.
- **Reproducibility attack**: The orthogonality metric is mathematically defined but may not generalize across architectures. What is "low orthogonality" for one SAE architecture may be normal for another.
- **Baseline sanity check**: Is this actually better than just using Matryoshka SAEs which explicitly reduce absorption? The practical value is unclear unless we can demonstrate concrete recovery.
- **Scope attack**: Recovery may work for partially absorbed features but fail for fully absorbed ones. The boundary conditions matter for practical utility.
- **Verdict**: MODERATE — Technically implementable and addresses a real gap (training-free absorption detection), but novelty is incremental. Could be valuable as a diagnostic tool.

### Against Candidate B: Cross-Architecture Pareto

- **Implementation reality check**: This is essentially a measurement study, not a new method. The "Pareto frontier" framing is interesting but may just confirm what SAEBench already shows.
- **Reproducibility attack**: Pareto frontiers are sensitive to hyperparameter choices. Comparing architectures at "matched L0" is tricky because L0 behaves differently across architectures.
- **Baseline sanity check**: JumpReLU already shows lower absorption than TopK in prior work (0.0114 vs 0.1402). What's the incremental contribution of a systematic comparison?
- **Scope attack**: The practical takeaway is unclear. If the frontier shows one architecture dominates, why not just use it? If no dominance, the result is inconclusive.
- **Verdict**: WEAK — Measurement study with unclear novelty. Prior work (SAEBench) already compares architectures. Limited potential for a paper contribution unless we discover non-obvious interaction effects.

### Against Candidate C: Safety Feature Audit

- **Implementation reality check**: Safety feature identification via Neuronpedia is feasible but labor-intensive. Steering-based probes are well-established (已有成熟方法).
- **Reproducibility attack**: "Safety-relevant" is subjective and context-dependent. The selection of which features count as safety-critical may bias results.
- **Baseline sanity check**: The pilot data shows safety features have 0.0 absorption rate on Gemma Scope SAEs. This contradicts the hypothesis — safety features may not be disproportionately absorbed after all. The hypothesis may be wrong.
- **Scope attack**: Even if safety features show absorption, demonstrating "hidden risk" requires showing downstream consequences (e.g., steering fails for absorbed features). A measurement without consequence demonstration is weak.
- **Verdict**: MODERATE (downgraded from strong) — The pilot data showing 0.0 absorption for safety features is a significant finding that contradicts the starting hypothesis. However, this null result is itself publishable as a "myth-busting" contribution. The practical risk may be lower than assumed.

---

## Phase 4: Refinement

### Dropped Ideas
- **Cross-Architecture Pareto (Candidate B)** dropped because: It's a measurement study without clear novelty. SAEBench already provides cross-architecture comparison. The Pareto framing doesn't generate actionable insights.

### Strengthened Ideas
- **Anchor Feature Recovery (Candidate A)**: Strengthened by narrowing scope to "training-free detection" rather than "recovery." Focus on the orthogonality metric as absorption predictor. Use SynthSAEBench ground truth to validate.
- **Safety Feature Audit (Candidate C)**: Despite null result in pilot, reframed as "myth-busting" study — the assumption that safety features are more absorbed is untested and pilot suggests false. This is a legitimate contribution.

### Additional Analysis from Prior Iterations
- Iter 003 experiments show: encoder-driven absorption (Condition B: trained encoder + random decoder = 0.076 absorption rate vs Condition D: both trained = 0.017). This is a surprising finding — random decoder with trained encoder produces *more* absorption.
- Safety feature pilot on Gemma Scope shows 0.0 absorption for both safety and non-safety features — contradicts the hypothesis.
- Hierarchy cosine similarity (h_comp) pilot shows absorption monotonically increases with hierarchy strength (0.58→0.67→0.80) — confirms the competition mechanism.

### Selected Front-Runner
**Candidate A: Anchor Feature Recovery via Decoder Direction Analysis**

Reason: The encoder-driven absorption finding from iter 003 is the most interesting result and points to a concrete, implementable follow-up. The orthogonality metric as training-free absorption predictor is novel, falsifiable, and addresses a real gap. The iter 003 h_mech_pilot result (encoder drives absorption, decoder irrelevant) suggests we should focus on encoder behavior, not decoder recovery.

This idea connects to Gap 4 (training-free analysis) and Gap 5 (absorption-sensitivity relationship) from the literature survey.

---

## Phase 5: Final Proposal

### Title

**Encoder-Decoder Asymmetry as Training-Free Absorption Indicator in Sparse Autoencoders**

### Hypothesis

Absorbed features exhibit predictable encoder-decoder asymmetry: trained encoders produce feature representations that, when decoded by random weights, show higher absorption rates than random encoders with trained decoders. This asymmetry suggests absorption is primarily an encoder-side phenomenon driven by feature correlation during training, not a decoder-side reconstruction artifact. The encoder-decoder asymmetry ratio serves as a training-free absorption predictor.

### Motivation

The iter 003 pilot result revealed a surprising asymmetry: a trained encoder with random decoder (Condition B) produced 0.076 absorption rate — 4x higher than both trained (Condition D: 0.017) and even higher than random encoder with random decoder (Condition A: 0.004). This contradicts the intuition that absorption is about "stealing" decoder reconstruction capacity. Instead, absorption appears to be baked into the encoder's feature representations during training.

This matters because: (1) it explains *why* architectural interventions (Matryoshka, OrtSAE) work — they modify training dynamics, not just decoder weights; (2) it suggests absorption prediction doesn't require ablation — just analyzing encoder weight geometry; (3) it shifts mitigation strategies from architecture design to training dynamics modification.

### Method

**Step 1: Compute Encoder Feature Geometry**
- Load Gemma Scope SAE layer 12 (16k width) via SAELens
- Extract encoder weight matrix W_encoder (features × hidden)
- Compute feature feature cosine similarity matrix
- Identify high-correlation feature clusters as absorption risk indicators

**Step 2: Define Asymmetry Metric**
- For feature f, compute:
  - encoder_coherence(f) = average cosine similarity between f and its top-k correlated features
  - decoder_coherence(f) = average cosine similarity between f's decoder weights and top-k correlated decoder weights
- asymmetry_ratio(f) = encoder_coherence(f) / decoder_coherence(f)

**Step 3: Validate Against Ablation Ground Truth**
- On layers 0-17 where ablation-based absorption rates exist (via sae-spelling)
- Test whether asymmetry_ratio predicts absorption rate
- Compute Pearson correlation between predicted risk and measured absorption

**Step 4: Extend to Deep Layers**
- If correlation holds, apply asymmetry metric to layers 18-26 where ablation fails
- Produce absorption risk rankings for all layers

### Experimental Plan

1. **Pilot (20 min)**: Layer 12 Gemma Scope SAE, 50 randomly sampled features, compute asymmetry ratio, correlate with existing absorption measurements (if any exist for these features)
2. **Layer 0-17 Validation (45 min)**: Full layer sweep, compare asymmetry-predicted absorption vs. ablation-measured absorption across 18 layers
3. **Synthetic Validation (30 min)**: Use SynthSAEBench ground truth features with known absorption status; test whether asymmetry ratio correctly identifies absorbed vs. non-absorbed
4. **Cross-Architecture Test (1 hour)**: Compare asymmetry patterns in TopK vs. JumpReLU SAEs; test if the metric generalizes across architectures

### Resource Estimate

- **Model**: Gemma-2-2B with Gemma Scope SAEs (16k width, all 27 layers)
- **Dataset**: Activation logs from SAELens (~10k tokens); SynthSAEBench for ground truth
- **Time**: ~2.5 hours total on 1 GPU (pilot + validation + synthetic + architecture comparison)
- **Compute**: Feature extraction + cosine similarity is matrix operations; ~10M operations per layer; negligible cost

### Baselines

- **Ablation-based absorption rate** (sae-spelling): "Gold standard" but only works up to layer 17
- **Decoder orthogonality score** (Marks et al.): Alternative training-free metric for comparison
- **Feature sensitivity** (Hu et al.): Related but distinct — measures activation consistency, not directional stealing

Expected performance: Asymmetry ratio should correlate with ablation absorption at r > 0.5 to be useful. Lower correlation suggests the metric needs refinement.

### Risk Assessment

1. **Pilot showed opposite effect**: The encoder-random condition produced *more* absorption than both-trained. This could be an artifact of the specific features or the small sample size. *Mitigation*: Replicate with larger sample (100+ features) and multiple random seeds.

2. **Correlation may not hold across architectures**: If asymmetry is architecture-specific, it won't generalize. *Mitigation*: Test on SynthSAEBench which provides ground truth across different SAE training runs.

3. **Deep layer extension uncertain**: If the asymmetry-absorption correlation is only strong for early layers, the value for deep layer prediction is limited. *Mitigation*: Report boundary conditions honestly; note where the metric fails.

### Novelty Claim

This is the first work to systematically investigate the encoder-decoder asymmetry of absorption using controlled ablation conditions. While prior work (Marks et al., Hu et al.) hints at encoder-decoder behavior differences, no prior work:
1. Provides controlled experimental evidence that encoder, not decoder, drives absorption
2. Proposes asymmetry ratio as a training-free absorption predictor
3. Validates the predictor against ablation ground truth across 18 layers

The finding that a trained encoder with random decoder produces more absorption than both-trained is a surprising result that may shift how the community thinks about absorption mechanism.
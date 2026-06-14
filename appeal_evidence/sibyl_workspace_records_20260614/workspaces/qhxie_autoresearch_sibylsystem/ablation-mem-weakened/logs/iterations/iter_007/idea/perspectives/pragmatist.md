# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

Based on the comprehensive literature survey in `context/literature.md` and `context/idea_context.md`, plus the SAELens skill knowledge, the following resources are most directly useful for engineering-focused implementation:

1. **SAELens** (https://github.com/jbloomAus/SAELens) — 1,100+ stars, MIT license. The de facto standard library for loading pretrained SAEs, analyzing features, and performing steering. `SAE.from_pretrained()` API makes loading GemmaScope and GPT-2 SAEs trivial. **Code definitely exists and is production-ready.**

2. **sae-spelling** (https://github.com/lasr-spelling/sae-spelling) — Direct implementation of Chanin et al.'s absorption metric using ablation studies. Contains the first-letter spelling task probes. **Code exists but limited to early layers (0-17).**

3. **SAEBench** (https://github.com/adamkarvonen/SAEBench) — ICML 2025 benchmark with 8 metrics including projection-based absorption metric that works across all layers. **Code exists and is the standard evaluation framework.** Key insight: projection-based absorption metric is 26x faster than ablation-based (~1 min vs ~26 min per SAE).

4. **feature-hedging-paper** (https://github.com/chanind/feature-hedging-paper) — Contains code for analyzing hedging vs absorption trade-off in Matryoshka SAEs. Uses balanced loss coefficients (~0.75).

5. **sparse-but-wrong-paper** (https://github.com/chanind/sparse-but-wrong-paper) — Implements c_dec metric (decoder pairwise cosine similarity) for identifying optimal L0. Critical for ensuring L0 is not a confounding variable.

6. **GemmaScope Pretrained SAEs** (via SAELens) — Hundreds of pretrained SAEs on Gemma-2-2B across all layers, MLP/Attn/Residual, 16k/65k/131k widths. **Ready to use, no training needed.**

7. **GPT-2 SAEs** (via SAELens, `gpt2-small-res-jb` release) — All residual stream layers, ideal for rapid prototyping on a smaller model. **Best for pilot experiments under 15 minutes.**

8. **CE-Bench** (arXiv:2509.00691) — LLM-free contrastive evaluation with 5,000 story pairs. 77% alignment with SAEBench but no LLM API needed. Useful for validating absorption impact on interpretability.

### Landscape Summary

The SAE ecosystem is mature and well-tooled. SAELens provides turnkey access to pretrained SAEs across Gemma-2B, GPT-2, and Pythia. The main gap is not in tooling but in **systematic measurement** — Chanin et al. only studied layers 0-17, and SAEBench includes absorption as one of 8 metrics without deep analysis of patterns.

**What works in practice:** TopK and JumpReLU SAEs are stable and well-understood. The projection-based absorption metric from SAEBench is computationally tractable (1 min vs 26 min). SAELens steering API is straightforward to use.

**What doesn't work:** Ablation-based absorption measurement is too slow for systematic study. Feature steering as demonstrated by Basu et al. has a fundamental actionability problem — even 98.2% probe AUROC translates to 0% behavioral change. L0 is not a neutral knob.

**Practical gaps:** No comprehensive cross-model, cross-layer absorption quantification exists. The relationship between absorption and downstream task performance (circuit discovery, concept erasure) is unquantified. Training-free absorption detection in pretrained SAEs (without ground-truth probes) remains unsolved.

## Phase 2: Initial Candidates

### Candidate A: Cross-Model Absorption Benchmark

- **Hypothesis**: Absorption rates are predictable across model families and layer depths when controlling for SAE width and sparsity level, enabling configuration guidance for practitioners.
- **Implementation sketch**: Use SAELens to load GemmaScope (2B, JumpReLU, 16k/65k/131k), LlamaScope (8B, TopK, 32k/128k), and GPT-2 SAEs. Apply SAEBench's projection-based absorption metric across all available layers (not just 0-17). Compute absorption rates at different layer depths (early: 0-10, middle: 10-20, late: 20+) and SAE widths.
- **Simplest version**: Run projection-based absorption metric on GPT-2 residual stream SAEs (all layers) using first-letter spelling probes from sae-spelling repo. Target: 12 SAEs (layers 0-11), ~15 minutes total.
- **Time estimate**: Full cross-model study: ~4-6 GPU hours. Pilot on GPT-2: ~15 minutes.
- **Reusable components**: SAELens (pretrained SAE loading), SAEBench (projection metric), sae-spelling (probe tasks), first-letter spelling dataset.

### Candidate B: Absorption Impact on Downstream Tasks

- **Hypothesis**: Absorption systematically degrades SAE-based steering reliability — features identified as "high quality" by standard metrics (L0, CE loss recovered) may still produce zero behavioral change due to absorption structure.
- **Implementation sketch**: Select top-10 highest-performing SAE configurations by SAEBench metrics. For each, identify feature pairs with high absorption relationships (using projection-based metric). Run steering experiments comparing: (a) steering absorbed features alone, (b) steering absorbing features alone, (c) steering both together, (d) steering matched non-absorbed features. Measure behavioral change on downstream tasks (question answering, text generation).
- **Simplest version**: Pick GPT-2 SAE at layer 8. Identify top-20 features by CE loss recovered. Classify which are absorbed vs not using projection-based metric. Run steering experiments with 3 conditions (absorbed only, absorbing only, both) on a simple question answering task. Target: ~20 min.
- **Time estimate**: Full study with 10 SAE configs and multiple steering conditions: ~8-10 GPU hours. Pilot (1 SAE, 3 conditions): ~20 min.
- **Reusable components**: SAELens (SAE loading + steering API), SAEBench (metric computation), TransformerLens (model hook infrastructure).

### Candidate C: L0-Absorption Interaction Study

- **Hypothesis**: The c_dec metric (decoder pairwise cosine similarity) can predict which SAEs will have high absorption before computing expensive absorption metrics, enabling efficient SAE configuration selection.
- **Implementation sketch**: For a grid of SAE configurations (varying width: 16k/65k/131k, L0: different k values for TopK), compute c_dec (fast, ~1 min per SAE) and then compute absorption rate (slower, ~1-5 min per SAE with projection method). Establish correlation between c_dec and absorption rate. Validate that c_dec can serve as a proxy for absorption when computing optimal SAE configuration.
- **Simplest version**: On GPT-2 SAEs, vary L0 (k=20, 50, 100, 200) and measure c_dec + absorption rate correlation. Target: 4 SAEs, ~10 min.
- **Time estimate**: Grid study across 3 model families x 3 widths x 4 L0 values: ~6-8 GPU hours. Pilot: ~10 min.
- **Reusable components**: sparse-but-wrong-paper (c_dec implementation), SAELens (SAE loading), SAEBench (absorption metric).

## Phase 3: Self-Critique

### Against Candidate A

- **Implementation reality check**: SAEBench already measures absorption across 200+ SAEs. While not a dedicated cross-model study, the infrastructure exists and data is partially available. Extending to all layers (not just 0-17) is genuinely novel but may reveal similar patterns to what Chanin et al. found on Gemma-2-2B layers 0-17.
- **Reproducibility attack**: The projection-based metric is faster but may have different sensitivity than ablation-based. Need to validate projection method against ablation on a subset before trusting cross-model comparisons.
- **Baseline sanity check**: What is the "strong simple baseline"? Simply reporting absorption rates across models without connecting to downstream impact is descriptive, not causal. Would need to show that absorption rate predicts something practically meaningful (e.g., steering reliability).
- **Scope attack**: A full cross-model study across Gemma/Llama/GPT-2 with multiple widths and all layers would produce a large table of numbers. Without clear actionability (e.g., "use width X at layer Y to minimize absorption"), this risks being a dataset paper.
- **Verdict**: MODERATE — Valuable as a reference study but incremental if it just produces more absorption numbers without connecting to practical guidance. Would need a strong "configuration guide" framing with actionable recommendations.

### Against Candidate B

- **Implementation reality check**: Basu et al. (2026) already showed steering fails (98.2% AUROC but 45.1% output sensitivity) and attributed it to residual stream compensation. This is a fundamental limitation, not something that can be fixed by steering strategy alone. The result may not be actionable.
- **Reproducibility attack**: Steering experiments have high variance and are sensitive to prompt phrasing, generation parameters, and model state. Reproducing behavioral changes reliably is difficult.
- **Baseline sanity check**: A "zero baseline" would be random feature steering or no steering. If absorption-aware steering also produces zero effect (as Basu et al. suggest), the negative result is already known.
- **Scope attack**: This is a single downstream task (clinical domain in Basu et al.) — may not generalize to other domains. The failure mode may be domain-specific.
- **Verdict**: WEAK — The Basu et al. result suggests the fundamental problem is residual stream compensation, not absorption structure per se. Unless the absorption-aware approach specifically addresses residual stream compensation, it likely won't work. Risk of rediscovering a known negative result.

### Against Candidate C

- **Implementation reality check**: c_dec measures decoder weight similarity, which is a static property of the SAE. Absorption is a dynamic property of how features represent hierarchical concepts. The correlation between static weight geometry and dynamic absorption behavior may be weak or architecture-dependent.
- **Reproducibility attack**: The correlation between c_dec and absorption rate needs validation across multiple SAE architectures (JumpReLU, TopK, Gated). If the correlation only holds for one architecture, the proxy is not generalizable.
- **Baseline sanity check**: A useful proxy must predict absorption better than random. Need to establish a baseline correlation coefficient and show c_dec exceeds it significantly.
- **Scope attack**: This is an efficiency optimization (avoiding expensive absorption computation) rather than addressing the core scientific question. If the correlation holds, it's a useful engineering contribution but may not be novel enough for a top venue.
- **Verdict**: MODERATE — If c_dec reliably predicts absorption, this is a valuable tool for practitioners to quickly assess SAE configurations without expensive computation. The risk is that the correlation is too weak to be useful.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Absorption Impact on Steering)** dropped because: Basu et al. (2026) have already established the fundamental limitation — residual stream compensation means steering interventions fail regardless of absorption structure. This is a known negative result that would require a fundamentally different intervention approach to overcome. The risk of rediscovering a known fundamental limitation is too high for a novel contribution.

### Strengthened Ideas

- **Candidate A (Cross-Model Absorption Benchmark)**: Strengthened by adding an **actionable configuration guide** framing. The output should be a decision tree or set of rules like: "For layer depths 0-10, use width 65k to minimize absorption; for layers 20+, absorption is unavoidable regardless of configuration." This moves beyond descriptive measurement to practical guidance.

- **Candidate C (L0-Absorption Interaction)**: Strengthened by making it a **predictive tool** rather than just a correlation study. If c_dec predicts absorption, build a regression model: `absorption_rate ~ f(c_dec, width, layer, architecture)`. This enables practitioners to predict absorption for new SAE configurations without running expensive metrics. Also adds value by identifying which SAE configurations to avoid.

### Selected Front-Runner

**Candidate A + C combined**: "Absorption Benchmark with Predictive Modeling"

Rationale: This combines the systematic measurement of Candidate A with the predictive efficiency of Candidate C. The combined approach delivers both a comprehensive reference dataset (valuable for the community) AND a practical prediction tool (actionable for practitioners). The prediction model using c_dec as a feature addresses the efficiency problem of Candidate C while the benchmark addresses the scope limitation.

**Implementation plan for pilot (15 min)**:
1. Load GPT-2 SAE layer 8 from SAELens (`gpt2-small-res-jb`, `blocks.8.hook_resid_pre`)
2. Compute c_dec (decoder pairwise cosine similarity) for different L0 values (k=32, 64, 128, 256)
3. Compute absorption rate using SAEBench projection metric for each
4. Validate correlation between c_dec and absorption rate
5. If correlation holds, build a simple linear model for absorption prediction

If pilot succeeds, scale to full cross-model study with the established prediction model.

## Phase 5: Final Proposal

### Title

**"Absorption in Pretrained SAEs: A Cross-Architecture Benchmark and L0-Based Prediction Model"**

### Hypothesis

1. Absorption rates vary systematically across model families, layer depths, and SAE configurations in a predictable pattern that enables configuration guidance.
2. The decoder pairwise cosine similarity (c_dec) metric serves as a reliable predictor of absorption rate, enabling efficient SAE configuration selection without expensive ablation or projection-based absorption computation.

### Motivation

Feature absorption in SAEs creates an "interpretability illusion" where features appear monosemantic but have systematic false negatives. This undermines the reliability of SAE-based interpretability research. While Chanin et al. (2024) established the phenomenon and SAEBench (Karvonen et al., 2025) provides evaluation infrastructure, there is no comprehensive study of absorption patterns across different model families, layer depths, and SAE configurations. Furthermore, practitioners lack efficient methods to assess absorption in their specific SAE configurations without running expensive computation.

This work addresses a practical gap: helping researchers and practitioners select SAE configurations that minimize absorption for their specific use case.

### Method

**Step 1: Data Collection**
- Use SAELens to load pretrained SAEs from 3 model families: Gemma-2-2B (via GemmaScope), Llama-3.1-8B (via LlamaScope), GPT-2-small (via SAELens pretrained)
- Cover multiple SAE widths (16k, 65k, 131k for Gemma; 32k, 128k for Llama; 16k for GPT-2)
- Sample layers across depth: early (0-10), middle (10-20), late (20+)

**Step 2: Absorption Measurement**
- Use SAEBench's projection-based absorption metric (works across all layers, ~1-2 min per SAE vs 26 min for ablation)
- Validate projection method against ablation on a subset (3 SAEs) to ensure equivalence
- Compute absorption rate for each SAE configuration

**Step 3: c_dec Computation**
- Implement decoder pairwise cosine similarity (c_dec) following Chanin & Garriga-Alonso (2025b)
- Compute c_dec for all SAE configurations (~1 min per SAE)

**Step 4: Predictive Modeling**
- Train regression model: `absorption_rate ~ c_dec + width + layer_depth + architecture`
- Evaluate prediction accuracy against held-out SAE configurations
- Identify which factors most strongly predict absorption

**Step 5: Validation**
- Validate prediction model on held-out SAE families
- Test whether the model's configuration recommendations improve downstream interpretability tasks (using CE-Bench for evaluation)

### Simplest Version

**Pilot (15 min on single GPU)**:
- Load GPT-2 SAE layer 8 from SAELens
- Vary L0 (k=32, 64, 128, 256) — use TopK re-interpretation or analyze existing configs
- Compute c_dec and absorption rate for each
- Establish correlation and fit simple linear model
- If pilot succeeds, proceed to full cross-model study

**Minimum viable experiment**: Demonstrate c_dec absorption prediction on 4 GPT-2 SAEs with different L0 configurations. Validate against ground-truth absorption rates. Show R² > 0.5 for the prediction model.

### Baselines

1. **SAEBench absorption measurements** (Karvonen et al., 2025): Provides absorption rates for 200+ SAEs but does not systematically analyze patterns or build predictive models. Expected absorption range: 0.1-0.6 depending on configuration.

2. **Chanin et al. ablation-based absorption** (2024): Gold standard measurement but limited to layers 0-17 of Gemma-2-2B. Expected absorption rate for early layers: ~0.3-0.5.

3. **L0 sensitivity baseline** (Chanin & Garriga-Alonso, 2025b): Shows c_dec varies with L0 and correlates with feature correctness. Expected c_dec range: 0.1-0.4 for well-tuned SAEs.

### Experimental Plan

**Datasets**:
- First-letter spelling dataset (from sae-spelling) for absorption probes
- CE-Bench (5,000 story pairs) for downstream interpretability validation

**Metrics**:
- Primary: Absorption rate (projection-based, SAEBench implementation)
- Secondary: c_dec, L0, CE loss recovered
- Validation: CE-Bench contrastive score

**Ablation Schedule**:
1. Pilot: GPT-2 layer 8, 4 L0 configs — 15 min
2. Cross-model: Gemma-2-2B (layers 0, 8, 16, 24), Llama-8B (layers 0, 16, 32), GPT-2 (all layers) — 3 hours
3. Width ablation: 16k vs 65k vs 131k on Gemma-2-2B — 2 hours
4. Prediction model validation: Train on 80% of data, test on 20% — 30 min

### Resource Estimate

| Experiment | GPU Hours | Wall-Clock | Model Size |
|------------|-----------|------------|------------|
| Pilot (GPT-2, 4 configs) | 0.25 | 15 min | GPT-2 (124M) |
| Cross-model benchmark | 3.0 | 3 hours | Gemma-2-2B, Llama-8B, GPT-2 |
| Width ablation | 2.0 | 2 hours | Gemma-2-2B only |
| Prediction model | 0.5 | 30 min | CPU |
| **Total** | **~6** | **~6 hours** | |

**Note**: All experiments use pretrained SAEs (no training). Gemma-2-2B requires ~18GB GPU memory. GPT-2 fits in 8GB.

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| c_dec not correlated with absorption | Medium | Fall back to direct absorption measurement; still produce benchmark |
| Projection method not equivalent to ablation | Low | Validate on 3 SAEs before full study |
| SAE loading failures | Low | SAELens is stable; have fallback configs |
| GPU memory on large models | Medium | Use Gemma-2-2B for main study; GPT-2 for pilots |

### Novelty Claim

This work provides the first:
1. **Cross-architecture absorption benchmark** covering Gemma, Llama, and GPT-2 families with consistent methodology
2. **Predictive model** linking c_dec (fast to compute) to absorption rate (expensive to measure), enabling efficient SAE configuration selection
3. **Actionable configuration guide** derived from systematic measurement, helping practitioners choose SAE configurations that minimize absorption for their specific use case

The novelty is not in discovering absorption (known) but in **systematizing the measurement and making it actionable** for practitioners who need to select SAE configurations without running expensive experiments.
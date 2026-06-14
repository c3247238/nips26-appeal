# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling (GitHub: lasr-spelling/sae-spelling)** -- The canonical absorption metric implementation. MIT licensed. Key modules: `feature_absorption_calculator`, `probing` (logistic regression), `feature_ablation`, `vocab` (alphabetic token filtering). Directly reusable for first-letter task; would need adaptation for cross-domain hierarchies. **Has code.**

2. **SAEBench (GitHub: adamkarvonen/SAEBench)** -- Comprehensive 8-metric evaluation suite including absorption. Apache 2.0. Evaluates 200+ SAEs across 8 architectures on Pythia-160M and Gemma-2-2B. Absorption metric is built on top of Chanin et al.'s code. The interactive results on neuronpedia.org/sae-bench provide existing baselines. **Has code.**

3. **SAELens (GitHub: decoderesearch/SAELens)** -- Standard library for loading, training, and analyzing SAEs. Supports all major architectures (Standard, Gated, TopK, JumpReLU). Deep integration with TransformerLens. Handles Gemma Scope SAE loading out of the box. MIT licensed. **Has code.**

4. **Gemma Scope SAEs (HuggingFace: google/gemma-scope)** -- 400+ pre-trained JumpReLU SAEs on Gemma 2 2B/9B/27B, all layers, widths 1k-1M. Primary evaluation target. Eliminates all training costs. **Pre-trained weights available.**

5. **TransformerLens (GitHub: TransformerLensOrg/TransformerLens)** -- Hook-based activation caching for 50+ models including Gemma 2 2B. Required for activation extraction. MIT licensed. **Has code.**

6. **Chanin et al., 2024. "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** -- Defines absorption, provides toy model proof, proposes canonical metric. Empirical finding: 15-35% absorption rate across all tested SAEs. Limitation: only first-letter task, metric requires known probes. **The paper we must extend, not reinvent.**

7. **SAEBench ICML 2025 paper (arXiv:2503.09532)** -- Key finding: Matryoshka SAEs best on absorption metric; JumpReLU/TopK worst (absorption increases with training time). Proxy metrics (CE loss, sparsity) do NOT predict practical performance. **Provides existing baselines to compare against.**

8. **AxBench (GitHub: stanfordnlp/axbench, ICML 2025 Spotlight)** -- Shows well-crafted prompts and simple statistical checks outperform SAEs on steering and concept detection. Reinforces that SAE-based methods need rigorous baselines. **Important context for framing our contribution.**

9. **Chanin & Garriga-Alonso, 2025 (arXiv:2508.16560)** -- Most open-source SAEs have incorrect L0 (too low), causing feature hedging. Critical confound for absorption studies. Code available at github.com/chanind/sparse-but-wrong-paper. **Must be controlled for.**

10. **LessWrong post: "Looking for feature absorption automatically"** -- Attempted unsupervised absorption detection using decoder cosine similarity of effect vectors. **Failed: non-bimodal distribution, co-occurrence invalidated.** This is a critical negative result we must learn from.

11. **Matryoshka SAE (arXiv:2503.17547, ICML 2025)** -- Best architecture for absorption reduction. Nested dictionaries organize features hierarchically. Code at github.com/noanabeshima/matryoshka-saes. **Comparison target, not our method.**

12. **OrtSAE (arXiv:2509.22033)** -- Reduces absorption 65% via orthogonality penalty. Three complementary absorption metrics (mean absorption fraction, full-absorption score, feature splits). **Useful metrics to adopt.**

### Landscape Summary

The engineering landscape for absorption research is remarkably mature in terms of infrastructure (SAELens, TransformerLens, Gemma Scope, SAEBench all work together out of the box) but remarkably narrow in terms of evaluation domains (everything is first-letter spelling). The practical gaps are:

1. **No one has run the Chanin absorption metric on anything other than first-letter spelling with an adequately sized model.** GPT-2 Small has been shown to be insufficient (98% dead features on city prompts in iteration 5). Gemma 2 2B + Gemma Scope is the obvious target.

2. **Unsupervised detection has failed exactly once** (LessWrong post) using a naive approach (raw cosine similarity). The failure mode is well-understood: cosine similarity is non-bimodal and effect-vector similarity does not discriminate absorption from mere semantic relatedness.

3. **No controlled cross-architecture absorption comparison exists.** SAEBench measures absorption but each architecture was trained with different hyperparameters/durations. This is a gap but probably not worth filling given our training-free constraint.

4. **The L0 confound is real and poorly controlled.** Chanin & Garriga-Alonso showed most SAEs have too-low L0. Any cross-SAE absorption comparison must account for this.

The simplest thing that would be publishable: extend the existing sae-spelling metric to 3-4 knowledge hierarchy domains on Gemma 2 2B, report absorption rates, and show whether hierarchy properties predict absorption severity. Everything else (unsupervised detection, information theory) is bonus.

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Audit (The Simple Version)

- **Hypothesis**: Feature absorption occurs at rates >= 10% in knowledge hierarchies (city-country, city-continent) on Gemma 2 2B, measured with the existing Chanin et al. probe-based metric adapted for new domains.

- **Implementation sketch**: Fork sae-spelling. Replace the first-letter prompt template with knowledge hierarchy templates ("Paris is a city in the country: France"). Train k-sparse logistic regression probes for parent concepts (country, continent) using SAELens-loaded Gemma Scope SAEs on Gemma 2 2B layer 12. Run the existing absorption detection pipeline (identify false negatives, integrated-gradients attribution, cosine/magnitude thresholds). Compute absorption rates per domain.

- **Simplest version**: Take 20 countries, 5 cities per country (100 total single-token entities), one SAE (Gemma 2 2B, layer 12, 16k width), and run the absorption metric. This is literally the existing sae-spelling pipeline with different prompts.

- **Time estimate**: Pilot: 15 minutes (probe quality check). Full experiment across 4 domains: ~3 hours. Scaling analysis across multiple SAE configs: ~2 more hours.

- **Reusable components**: sae-spelling (absorption metric, probing, ablation), SAELens (SAE loading), TransformerLens (activation extraction), Gemma Scope (pre-trained SAEs). Approximately 80% code reuse.

### Candidate B: Unsupervised Absorption Detection via Decoder Geometry + ITAC

- **Hypothesis**: Absorption can be detected without external probes by combining decoder cosine similarity analysis with residual-based verification (ITAC), achieving Spearman rho > 0.5 correlation with the gold-standard probe metric.

- **Implementation sketch**: (1) Compute pairwise decoder cosine similarity matrix (16k x 16k, feasible on GPU). (2) Apply hierarchical clustering at multiple resolution thresholds. (3) Filter candidate pairs by firing rate asymmetry (child fires more specifically than parent). (4) For each candidate pair, compute ITAC = Var(residual projected onto parent decoder direction | child active, parent inactive) / Var(same | neither active). (5) Validate against probe-based absorption on first-letter task.

- **Simplest version**: Skip the multi-resolution clustering. Just compute pairwise decoder cosine similarity, threshold at 0.3, filter by firing rate asymmetry (parent fires > 10x more than child in co-occurring contexts), and compute ITAC on the top 100 candidate pairs. Validate on first-letter task.

- **Time estimate**: Cosine similarity matrix: ~5 minutes on GPU for 16k features. Clustering + filtering: ~10 minutes. ITAC computation on 100k tokens: ~30 minutes. Validation: ~15 minutes. Total: ~1 hour.

- **Reusable components**: SAELens (SAE loading, decoder weights), TransformerLens (activations), scipy (clustering), numpy (variance computation). Custom code needed: ~200 lines for ITAC, ~100 lines for filtering heuristics.

### Candidate C: Strong Baseline Done Right -- Probe vs. SAE Absorption Gap Analysis

- **Hypothesis**: The gap between logistic regression probe performance and SAE k-sparse probe performance on hierarchical classification tasks is primarily explained by feature absorption (rather than other SAE failure modes), and this gap varies systematically with hierarchy properties.

- **Implementation sketch**: For each domain: (1) Train a dense logistic regression probe on model activations (the ceiling). (2) Train k-sparse probes on SAE activations (the SAE-based approach). (3) Compute the F1 gap. (4) Decompose the gap: what fraction of the dense probe's correct predictions correspond to SAE false negatives that are absorption (integrated-gradients attribution confirms another feature absorbed the signal) versus other failure modes (dead features, hedging, noise)? This directly addresses DeepMind's 2025 concern about SAE probes failing vs. dense probes.

- **Simplest version**: Run on first-letter task + city-country task, single SAE config. Report the F1 gap and the fraction attributable to absorption. This is 3 numbers that tell a powerful story.

- **Time estimate**: ~2 hours (probe training + gap decomposition for 2 domains).

- **Reusable components**: sae-spelling (probing, absorption detection), scikit-learn (logistic regression). ~90% code reuse.

## Phase 3: Self-Critique

### Against Candidate A (Cross-Domain Absorption Audit)

- **Implementation reality check**: Searched for anyone who tried cross-domain absorption measurement. The current proposal's iteration 5 tried it on GPT-2 Small and it failed catastrophically (98% dead features, 0% detection). This is a KNOWN failure mode on GPT-2 Small. On Gemma 2 2B with Gemma Scope, the situation should be very different -- Gemma Scope SAEs are specifically trained and evaluated for these models, and SAEBench already shows they work well on the absorption metric for first-letter. The main risk is whether probes will achieve F1 > 0.85 on knowledge hierarchy tasks. RAVEL (used in SAEBench) provides entity-attribute data for Gemma 2 2B, suggesting feasibility. **Risk: moderate, well-mitigated by probe quality gate.**

- **Reproducibility attack**: Extremely reproducible. The entire pipeline uses public code (sae-spelling), public SAEs (Gemma Scope), and a public model (Gemma 2 2B). The only custom component is the prompt templates for new domains, which are trivial to specify. The absorption metric has specific, published thresholds (cosine > 0.025, magnitude gap >= 1.0). We add a threshold sensitivity analysis to verify robustness. **Strong on reproducibility.**

- **Baseline sanity check**: The relevant baseline is the existing Chanin et al. absorption rate on first-letter (15-35%). Our contribution is showing whether this generalizes. If knowledge hierarchy absorption is 0%, that is still a finding. If it is 10-25%, it validates the concern. If it is > 35%, it suggests first-letter underestimates the problem. We also include random probe control, shuffled label control, and dense probe ceiling. **Baselines are solid.**

- **Scope attack**: The concern is that first-letter is a uniquely clean, exhaustive hierarchy (26 classes, no overlap). Knowledge hierarchies are messier (overlapping countries, multi-country cities). This is actually the POINT -- showing whether absorption generalizes to realistic hierarchies. We test 4 hierarchy types. **Scope is good for a single contribution; not flashy enough for top-tier on its own.**

- **Verdict**: **STRONG** -- High feasibility, high reproducibility, clear contribution, but moderate novelty on its own (it is "run existing metric on new domains").

### Against Candidate B (Unsupervised Detection)

- **Implementation reality check**: The LessWrong "Looking for feature absorption automatically" post tried this and failed. The failure mode is documented: decoder cosine similarity alone is non-bimodal, and effect-vector similarity does not discriminate absorption from general semantic relatedness. Our proposal addresses this with three innovations: (1) conditional cosine similarity instead of raw, (2) firing rate asymmetry filtering, (3) ITAC residual verification. However, I am not confident these fixes are sufficient. The conditional cosine similarity requires choosing a subspace (top-k decoder components), which is a hyperparameter. The firing rate filter is reasonable but may have high false positive rate. ITAC is a novel metric with no prior validation. **Risk: HIGH. This is the most speculative component.**

- **Reproducibility attack**: Each step is computationally deterministic and uses standard operations (cosine similarity, variance, hierarchical clustering). The concern is hyperparameter sensitivity: cosine threshold, firing rate asymmetry ratio, ITAC significance threshold, subspace dimensionality for conditional cosine similarity. We need a thorough ablation of these. **Moderate reproducibility concern due to hyperparameter sensitivity.**

- **Baseline sanity check**: The gold standard is the Chanin et al. probe-based metric. We validate against it on first-letter (where we have ground truth). Target: Spearman rho > 0.5. If rho < 0.3, the method fails. The LessWrong attempt achieved approximately rho ~ 0 (their metric did not correlate). We need to substantially beat this. **Good validation plan.**

- **Scope attack**: If it works on first-letter, does it generalize to other domains? First-letter is unusually clean. But the whole point of unsupervised detection is to work without domain-specific probes, so we should test on domains where we have BOTH probe-based ground truth AND unsupervised scores. **Reasonable.**

- **Verdict**: **MODERATE** -- Genuinely novel, but builds on a known failure (LessWrong negative result). High risk of repeating the failure with fancier heuristics. The key question is whether the three-part pipeline actually provides enough discriminative signal. I would prioritize Candidate A and treat this as a secondary contribution.

### Against Candidate C (Probe vs. SAE Gap Decomposition)

- **Implementation reality check**: This is essentially combining two well-established measurements (dense probe accuracy and SAE probe accuracy) with absorption attribution. All components exist in sae-spelling. The gap decomposition is new but conceptually straightforward: for each SAE false negative, check if integrated-gradients attribution points to another active feature (absorption) vs. no feature is active (dead feature / capacity failure) vs. multiple features partially active (hedging). **High feasibility.**

- **Reproducibility attack**: Very high. Uses existing probes, existing attribution code, deterministic computation. **Strong.**

- **Baseline sanity check**: Dense probes are the baseline. AxBench (ICML 2025) already showed dense probes outperform SAEs on concept detection. Our contribution is decomposing WHY -- specifically, how much of the gap is absorption versus other failure modes. This directly responds to DeepMind's 2025 concern. **Strong motivation.**

- **Scope attack**: The decomposition requires knowing the ground-truth feature (via probe) to identify absorption. So this only works in domains where we have good probes. This limits scope to the same domains as Candidate A. **Same scope limitation as A.**

- **Verdict**: **STRONG** -- Simple, rigorous, directly addresses a high-profile concern (DeepMind's deprioritization). The gap decomposition is novel and useful. Pairs naturally with Candidate A.

## Phase 4: Refinement

### Dropped

Nothing dropped. All three candidates are viable. However, I reorganize their priority.

### Strengthened and Combined

**Front-runner: Combined Candidate A + C -- "Cross-Domain Absorption Characterization with Probe-SAE Gap Decomposition"**

The strongest proposal combines A and C into a single coherent study:

1. **Stage 1**: Extend Chanin et al. absorption metric to 4+ knowledge hierarchy domains on Gemma 2 2B (Candidate A core).
2. **Stage 2**: For each domain, compute the dense probe vs. SAE probe performance gap and decompose it into absorption, hedging, dead features, and reconstruction error components (Candidate C core).
3. **Stage 3**: Test whether hierarchy properties (co-occurrence frequency ratio, fan-out, depth) predict both absorption rate AND gap magnitude across domains.
4. **Stage 4**: Confound control -- partial correlation with L0/width, Rosenbaum sensitivity analysis.

**Candidate B (unsupervised detection)** is retained as a secondary contribution. If the conditional cosine + firing rate + ITAC pipeline achieves rho > 0.5 against probe-based ground truth on first-letter, it becomes a strong secondary contribution. If rho < 0.3, we report the negative result honestly and focus on the cross-domain characterization.

### Minimal Pilot Experiment (< 15 min)

Before committing to full experiments:
1. Load Gemma 2 2B + Gemma Scope SAE (layer 12, 16k) via SAELens.
2. Construct 50 city-country prompts using single-token city names from the Gemma 2 tokenizer.
3. Extract residual stream activations at the city token position.
4. Train a logistic regression probe for the "country" parent feature.
5. Check F1. If F1 < 0.70, the domain is infeasible for this model/layer -- pivot to animal taxonomy or number parity.

This takes ~15 minutes and is a hard go/no-go gate.

## Phase 5: Final Proposal

### Title

Cross-Domain Feature Absorption in Sparse Autoencoders: A Systematic Characterization Beyond First-Letter Spelling

### Hypothesis

Feature absorption -- where SAE sparsity pressure causes a parent feature to be silently absorbed into a co-occurring child feature -- occurs at rates >= 10% in knowledge hierarchies (city-country, city-continent, city-language) on Gemma 2 2B with Gemma Scope SAEs, and the probe-vs-SAE performance gap is primarily explained by absorption rather than other SAE failure modes.

Precisely falsifiable: absorption rate < 5% across ALL knowledge domains after probe quality gating (F1 > 0.85).

### Motivation

Feature absorption is the most studied failure mode of SAEs, but every published study uses a single evaluation domain (first-letter spelling). This domain is uniquely clean: 26 exhaustive, non-overlapping classes with deterministic membership. Real-world feature hierarchies -- the kind that matter for safety (entity type, intent classification, knowledge representation) -- are messy, overlapping, and probabilistic. We simply do not know if the 15-35% absorption rates found by Chanin et al. generalize.

Meanwhile, DeepMind publicly deprioritized SAE research in 2025 after finding SAE probes fail catastrophically on safety-relevant downstream tasks where dense probes succeed. AxBench (ICML 2025 Spotlight) showed simple baselines outperform SAEs on concept detection and steering. These negative results implicate absorption but do not quantify it across domains.

This project closes the gap by: (1) running the first cross-domain absorption characterization on Gemma 2 2B (the model where adequate SAEs exist), (2) decomposing the probe-vs-SAE performance gap into absorption vs. other failure modes, and (3) testing whether measurable hierarchy properties predict absorption severity.

### Method

**Step-by-step implementation plan:**

**Stage 0: Infrastructure setup (~30 min)**
- Install SAELens >= 4.0, TransformerLens, sae-spelling (Poetry environment)
- Load Gemma 2 2B via TransformerLens: `model = HookedTransformer.from_pretrained("gemma-2-2b")`
- Load Gemma Scope SAE via SAELens: `sae, cfg, sparsity = SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82")`
- Verify model + SAE pipeline works on a test input

**Stage 1: Cross-domain absorption measurement (~3 hours)**

For each of 5 domains (first-letter, city-country, city-continent, city-language, animal-class):

1. *Dataset construction*: Build ICL prompt templates adapted from sae-spelling's format. For knowledge hierarchies, use pattern: "{city_name} is located in the country" with completion " {country_name}". Restrict to single-token entities in Gemma 2 tokenizer. Source entities from RAVEL (for city data) and WordNet (for animal taxonomy).

2. *Probe training*: Using `sae_spelling.probing.train_multi_probe()`, train k-sparse (k=1,3,5) logistic regression probes on parent concept classification (e.g., "which country?") using residual stream activations at the entity token position.

3. *Probe quality gate*: Require F1 > 0.85 on held-out test set. Log and report domains that fail this gate as negative results. This gate is critical -- iteration 5 showed that low probe quality produces artifact absorption rates (e.g., 92.3% raw vs. 19.2% corrected).

4. *Absorption detection*: Using `sae_spelling.feature_absorption_calculator`, for each parent class:
   - Identify false-negative tokens (all k split latents inactive, probe correctly classifies)
   - Run integrated-gradients attribution (using `sae_spelling.feature_attribution`)
   - Apply Chanin et al. absorption criteria: cosine similarity > 0.025 with probe direction AND highest-ablation latent >= 1.0 larger than second-highest
   - Report per-class and per-domain absorption rates

5. *Threshold sensitivity*: Run the full 5x4 grid (cosine thresholds {0.01, 0.02, 0.025, 0.03, 0.05} x magnitude gaps {0.5, 1.0, 1.5, 2.0}). Report coefficient of variation across grid -- CV < 0.3 indicates metric stability.

**Stage 2: Probe-SAE gap decomposition (~2 hours)**

For each domain where probes pass quality gate:

1. *Dense probe ceiling*: Train full logistic regression on raw model activations (all dimensions). This is the upper bound on what ANY linear method can achieve.

2. *SAE probe*: Train k-sparse probe on SAE features. This is what SAEs can achieve.

3. *Gap computation*: F1_gap = F1_dense - F1_SAE_ksparse.

4. *Gap decomposition*: For each token where dense probe is correct but SAE probe fails:
   - **Absorption**: Integrated-gradients shows another active SAE feature has high cosine similarity with the probe direction (Chanin et al. criteria)
   - **Dead features**: No SAE feature activates at all in the relevant subspace
   - **Hedging**: Multiple SAE features partially activate but none crosses the detection threshold (ambiguous split)
   - **Reconstruction error**: The SAE residual in the probe direction is large (> 1 std of the parent direction projection)

5. *Report*: Pie chart of gap decomposition per domain. This directly answers "how much does absorption explain SAE underperformance?"

**Stage 3: Hierarchy predictor analysis (~1 hour)**

1. For each domain, compute hierarchy properties:
   - Parent-child co-occurrence frequency ratio in a 100k-token corpus
   - Number of children per parent (fan-out)
   - Hierarchy depth (1 for flat, 2+ for multi-level)
   - Parent feature frequency (how often the parent concept is active)

2. Correlate each property with absorption rate across all domain x parent-class measurements.

3. Fit a simple linear model: absorption_rate ~ co_occurrence_ratio + fan_out + parent_frequency + L0 (as covariate).

4. Report Spearman correlations with Bonferroni correction.

**Stage 4: Confound control (~1 hour)**

1. *L0 covariate analysis*: Partial correlation between absorption rate and quality metrics (SP-F1, SCR), controlling for log(L0) and log(width).

2. *Within-width analysis*: Hold width constant (16k), vary L0 across available Gemma Scope SAEs. Report absorption-quality correlations within fixed width. Iteration 5 found these vanish (Rosenbaum Gamma = 1.0) -- we verify on the new domains.

3. *Cross-width analysis*: Mahalanobis matching across widths. Report Gamma for comparison with within-width results.

4. *FDR correction*: Benjamini-Hochberg across all hypothesis tests.

**Stage 5 (secondary): Unsupervised absorption detection (~2 hours)**

1. *Decoder geometry analysis*: Compute 16k x 16k pairwise cosine similarity matrix of SAE decoder weights. Threshold at {0.2, 0.3, 0.4} to build feature graphs.

2. *Candidate pair identification*: For each pair with cosine similarity > threshold, check firing rate asymmetry: parent fires on > 5x more tokens than child among tokens where both are relevant (based on co-activation patterns).

3. *ITAC computation*: For top 200 candidate pairs, compute ITAC = Var(residual dot parent_decoder | child active, parent inactive) / Var(residual dot parent_decoder | neither active) on 100k tokens.

4. *Validation*: On first-letter task (where probe-based ground truth exists), compute Spearman correlation between unsupervised scores and probe-based absorption rates. Report rho, p-value, precision@50%, AUROC.

5. *Decision gate*: If rho > 0.5, deploy on domains without probes. If rho < 0.3, report as negative result. If 0.3-0.5, report with caveats.

### Simplest Version

The absolute minimum experiment that tests the core claim:

1. Load Gemma 2 2B + one Gemma Scope SAE (layer 12, 16k, L0~82).
2. Construct 100 city-country prompts (20 countries x 5 cities).
3. Train probe for "which country?" on residual activations. Check F1 > 0.85.
4. Run absorption detection using the sae-spelling pipeline.
5. Report: absorption rate for city-country on Gemma 2 2B.

This is one number -- the first cross-domain absorption rate ever measured on an adequately sized model. It takes ~45 minutes.

### Baselines

1. **Chanin et al. first-letter absorption rate (15-35%)**: Our primary comparison. We reproduce on Gemma 2 2B and compare cross-domain rates against it. Expected performance: 15-35% on first-letter (replication); 10-25% on city-country (hypothesis).

2. **Dense logistic regression probe (F1 ceiling)**: Trained on full model activations. Expected: F1 > 0.95 for first-letter, F1 > 0.90 for city-country (based on SAEBench RAVEL results showing strong linear probes on Gemma 2 2B). This is the ceiling. SAE probes should fall below this, and the gap is what we decompose.

3. **Random probe control (~0% absorption)**: Probes trained with random direction vectors. Should produce ~0% absorption rate. Confirms the metric is not trivially inflated.

4. **Shuffled label control (~0% absorption)**: Random city-country assignments. Should produce ~0% absorption. Confirms the metric captures real hierarchical structure.

### Experimental Plan

| Experiment | Time | Purpose |
|---|---|---|
| Exp 0: Probe quality pilot (50 cities) | 15 min | Go/no-go gate |
| Exp 1a: First-letter replication | 30 min | Reproduce baseline |
| Exp 1b: City-country absorption | 45 min | Primary new result |
| Exp 1c: City-continent absorption | 30 min | Coarser hierarchy |
| Exp 1d: City-language absorption | 30 min | Different hierarchy type |
| Exp 1e: Animal-class absorption | 30 min | Non-geographic domain |
| Exp 2: Gap decomposition (2 domains) | 2 hrs | Absorption vs. other failures |
| Exp 3: Hierarchy predictors | 1 hr | What predicts absorption? |
| Exp 4: Confound control + scaling | 1 hr | L0/width confounds |
| Exp 5: Unsupervised detection (secondary) | 2 hrs | Probe-free method |
| **Total** | **~8 hrs** | |

Each individual experiment is under 1 hour. The secondary contribution (Exp 5) can be dropped without affecting the primary story.

### Resource Estimate

- **GPU**: Single GPU with >= 24GB VRAM (A100, H100, or RTX 4090). Gemma 2 2B is ~5GB; Gemma Scope SAE 16k adds ~300MB.
- **GPU-hours**: ~8 hours total. No training. All analysis is forward passes + probing.
- **Software**: SAELens >= 4.0, TransformerLens, sae-spelling, PyTorch, scikit-learn, scipy, matplotlib.
- **Data**: Gemma Scope SAEs from HuggingFace (~2GB download). Entity lists from RAVEL/WordNet (public).
- **Disk**: ~7GB total (model + SAEs + cached activations).
- **Cost**: Zero beyond GPU time. All components are open-source.

### Risk Assessment

| Risk | Probability | Mitigation |
|---|---|---|
| Probes fail on city-country (F1 < 0.85) | 15% | Probe quality gate at Exp 0; backup domains (animal taxonomy, number parity); try multiple layers (10, 12, 20); single-token entities only |
| Absorption rate < 5% in all knowledge domains | 20% | Report as important negative result ("absorption is domain-specific to first-letter-type hierarchies"); first-letter provides guaranteed baseline for the rest of the paper |
| Unsupervised detection fails (rho < 0.3) | 35% | Drop to secondary/negative result; paper still stands on cross-domain characterization + gap decomposition |
| L0 confound eliminates within-width correlations | 40% | Report transparently as a key finding about confound structure; this replicates iteration 5's within-width null |
| Threshold sensitivity shows metric is brittle (CV > 0.5) | 25% | Report as important methodological finding; use robust median across grid; this would motivate better metrics |
| SAELens/TransformerLens version incompatibility | 20% | Pin versions in requirements.txt; test pipeline end-to-end before committing to full experiments |
| Multi-token entity names complicate analysis | 30% | Restrict to single-token entities; use last-token for unavoidable multi-token; document and run sensitivity analysis |

### Novelty Claim

The novelty is modest but concrete and valuable:

1. **First cross-domain absorption characterization on an adequately sized model**: No prior work measures absorption on anything other than first-letter spelling with a model where SAEs actually work (Gemma 2 2B). This is the most practically useful contribution.

2. **First probe-vs-SAE gap decomposition by failure mode**: Decomposing the F1 gap into absorption, hedging, dead features, and reconstruction error is new and directly addresses DeepMind's 2025 concern.

3. **Hierarchy property predictors of absorption**: Testing whether co-occurrence frequency ratio, fan-out, and depth predict absorption rate across domains is new.

4. **Unsupervised detection pipeline (if validated)**: The conditional cosine + firing rate + ITAC combination is new, though this builds on a known failure (LessWrong attempt).

What is NOT novel: the absorption metric itself, the SAE infrastructure, the existence of absorption, individual statistical methods. We are explicit about this.

The honest novelty framing: "We show that a well-known but domain-limited finding (absorption at 15-35%) does/does not generalize to the domains that actually matter for SAE-based interpretability, and we quantify exactly how absorption contributes to SAE underperformance."

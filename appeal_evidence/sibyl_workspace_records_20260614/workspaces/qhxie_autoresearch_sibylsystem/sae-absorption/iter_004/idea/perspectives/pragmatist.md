# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling (Chanin et al., 2024)** — https://github.com/lasr-spelling/sae-spelling — MIT license. The canonical absorption implementation: k-sparse probing, integrated-gradients ablation, absorption rate metric. Directly usable; ~200 lines of core metric code. Code exists and is well-documented.

2. **SAELens (Bloom et al.)** — https://github.com/jbloomAus/SAELens — MIT license, 1,100+ stars. Standard library for loading and running pre-trained SAEs. Has `SAE.from_pretrained()` for Gemma Scope, GPT-2. Complete workflow from activation extraction to feature encoding. Code exists, actively maintained.

3. **Gemma Scope** — https://huggingface.co/google/gemma-scope — 400+ pre-trained SAEs on Gemma 2 (1B, 2B, 9B, 27B), widths 1k–1M, all layers. Non-commercial research license. Eliminates training cost entirely. Primary evaluation target across all recent absorption papers.

4. **SAEBench (Karvonen et al., 2025, arXiv:2503.09532)** — https://github.com/adamkarvonen/SAEBench — Apache 2.0. Standardized 8-metric evaluation including absorption. 200+ pre-evaluated SAEs. Key finding: TopK/JumpReLU significantly worsen absorption vs. Standard ReLU; Matryoshka best for absorption. Directly provides comparative baselines. Code exists.

5. **"A is for Absorption" (Chanin et al., arXiv:2409.14507, NeurIPS 2025)** — Defines the phenomenon, proves toy model, measures absorption rate 15–35% on Gemma Scope SAEs, shows empirical relationship between L0 and absorption rate (lower L0 → higher absorption). Key finding: wider SAEs also show higher absorption. No code for extensions beyond first-letter task.

6. **"Feature Hedging" (Chanin et al., 2025, arXiv:2505.11756)** — Complementary failure mode (narrow SAEs merge correlated features). Shows Matryoshka SAE trades absorption for hedging. Provides the absorption–hedging duality framing. No standalone code repo identified.

7. **"Looking for Feature Absorption Automatically" (LessWrong post)** — https://www.lesswrong.com/posts/z7iyek97dAeQMxdSd/looking-for-feature-absorption-automatically — Practitioner analysis of unsupervised detection via downstream SAE causal similarity. Key negative result: cosine similarity of ablation effect vectors does NOT produce bimodal distribution; cannot distinguish absorbed pairs from non-absorbed pairs. Important engineering lesson: this approach fails in practice.

8. **SAEBench absorption task methodology** — Extends Chanin et al. to handle partial absorption (multiple latents share responsibility). Runs on RTX 3090, ~65 min per SAE for full evaluation suite. Published evaluation numbers allow direct comparison without re-running.

9. **OrtSAE (Korznikov et al., arXiv:2509.22033)** — Orthogonality penalty reduces absorption 65%. Linear overhead. Code available from paper authors. Provides a simple training-time baseline for comparison.

10. **Gemma Scope 2** — https://www.neuronpedia.org/gemma-scope-2 — Released Dec 2025 for Gemma 3 models, includes transcoders. Expands evaluation surface to newer model family.

11. **SynthSAEBench (2026, arXiv:2602.14687)** — Synthetic ground-truth dataset with known feature hierarchy, correlation, and Zipfian firing distributions. Allows controlled study of absorption without relying on first-letter proxy. Code status uncertain but paper is very recent.

12. **ATM SAE (Li et al., arXiv:2510.08855)** — Adaptive Temporal Masking achieves mean absorption score 0.0068 vs TopK 0.1402 on Gemma-2-2B. Best reported absorption reduction. Training required; not training-free. Mentions per-latent importance tracking during training.

### Landscape Summary

The field has the following practical state as of April 2026:

**What works reliably:**
- Loading pre-trained Gemma Scope SAEs via SAELens takes ~5 lines of code
- The sae-spelling absorption metric is implemented and validated; running it on one SAE for one letter takes ~minutes on a single GPU
- SAEBench provides absorption scores for 200+ SAEs, so many comparisons are already done
- The empirical relationship between L0 and absorption is well-documented (lower L0 → higher absorption, Chanin et al. Figure 7b)
- Absorption is present in ALL tested architectures (L1 ReLU, TopK, JumpReLU, Gated); Matryoshka reduces it most, ATM reduces it further

**What doesn't work:**
- Automatic unsupervised detection via downstream causal similarity fails (negative result, LessWrong post)
- Cross-domain absorption measurement (beyond first-letter task) has no existing implementation
- Predicting absorption rate from corpus statistics alone has no validated approach

**Key practical gap:** All existing absorption measurements use the first-letter spelling task proxy. Extending to other domains requires building a new probe evaluation harness, which is concrete ~2-day implementation work. The core absorption metric code (sae-spelling) generalizes; the task-specific probe training is the only new component needed.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption Rate Scaling Laws from Pre-trained SAE Survey

**Hypothesis**: Absorption rate is a predictable function of SAE configuration parameters (L0, width, layer depth) and corpus-derived co-occurrence statistics (parent-child feature frequency ratio), expressible as a closed-form scaling relationship.

**Implementation sketch**: Start from sae-spelling's absorption rate measurement code. Run absorption rate computation across the Gemma Scope SAE matrix (3 widths: 16k, 65k, 131k × multiple L0 settings × layers 0–26 of Gemma 2 2B) for the first-letter task. Collect (L0, width, layer, absorption_rate) tuples. Fit power-law and log-linear models. Additionally compute corpus co-occurrence statistics (P(child | parent) for letter-token pairs) from a fixed text corpus, and test whether adding these to the regression improves prediction.

**Simplest possible version**: Run sae-spelling on 10 Gemma Scope SAEs (5 widths × 2 L0 settings) at layers 8 and 20. Fit a linear model absorption_rate ~ log(L0) + log(width) + layer. If R² > 0.7, the hypothesis is supported.

**Time estimate**: Loading Gemma 2 2B + SAE + running k-sparse probing on 26 letters × ~500 test tokens per letter ≈ 15 min per SAE on a single A100. 30 SAEs = ~7 GPU-hours total. Well within budget if spread across pilot + main experiments.

**Reusable components**:
- sae-spelling: absorption rate metric (drop-in reuse)
- SAELens: SAE loading (drop-in reuse)
- Gemma Scope: all SAEs pre-trained (zero training cost)
- Standard statsmodels/sklearn for regression fitting

---

### Candidate B: Cross-Domain Absorption Characterization Using Existing Probe Tasks

**Hypothesis**: Feature absorption rates and patterns on the first-letter task generalize to semantically richer feature hierarchies (entity-type ⊃ specific-entity, sentiment-category ⊃ sentiment-intensity, grammatical-role ⊃ specific-morpheme). The factors that predict absorption severity (co-occurrence frequency, hierarchy depth) are domain-independent.

**Implementation sketch**: Identify 3–5 additional feature hierarchies where (a) a known parent-child relationship exists, (b) linear probes can be trained, (c) the child features co-occur reliably with parent features. Candidates: (1) "word is a number" ⊃ specific number tokens, (2) "word is a common English name" ⊃ specific name tokens, (3) "token is a verb" ⊃ specific verb lemmas, (4) "token is a country" ⊃ specific country tokens. Build probe training datasets using simple in-context learning or token lists. Adapt sae-spelling's k-sparse probing + absorption measurement to each hierarchy. Run on Gemma Scope 16k SAEs at layers 12 and 20.

**Simplest possible version**: One additional hierarchy (e.g., "word is a common English first name" ⊃ specific name tokens). Build a wordlist-based test set (top 100 English names). Train LR probe on residual stream activations. Run absorption metric. Compare absorption rate to first-letter baseline.

**Time estimate**: Building probe datasets: 2 hours coding. Probe training + absorption measurement: 20 min per hierarchy per SAE. Full 5 hierarchies × 4 SAEs = ~7 GPU-hours. Pilot (1 hierarchy × 1 SAE): 30 min.

**Reusable components**:
- sae-spelling k-sparse probing + absorption metric (extend, not rewrite)
- SAELens + Gemma Scope (drop-in)
- WordNet/NLTK for hierarchy construction
- Simple LR probe training (sklearn)

---

### Candidate C: Absorption Predictor from Decoder Geometry (Training-Free, No Labeled Probes)

**Hypothesis**: Absorption is detectable from the geometry of the SAE decoder weight matrix alone, without requiring labeled probes or downstream ablation. Specifically, pairs of SAE latents (d_i, d_j) where cosine_similarity(W_dec[i], W_dec[j]) > threshold AND frequency(latent_i) >> frequency(latent_j) are candidate absorption pairs.

**Implementation sketch**: For a given pre-trained SAE, extract all decoder directions (W_dec, shape [d_sae, d_model]). Compute pairwise cosine similarity matrix (sparse, only pairs above threshold 0.3). For each high-similarity pair, check activation frequency ratio using the SAE's cached activation statistics. Pairs with high similarity AND high frequency imbalance are hypothesized to be absorption candidates. Validate against the ground-truth absorption labels from sae-spelling's first-letter experiment.

**Simplest possible version**: On GPT-2 small layer 8 SAE (16k latents), compute 16k × 16k cosine similarity (using fast matrix multiplication). Find top-100 high-similarity pairs. Cross-reference with sae-spelling absorption labels. Compute precision/recall of the geometry-based detector.

**Time estimate**: Cosine similarity of 16k vectors: ~5 seconds on GPU. Activation frequency statistics: pre-computed in SAELens. Validation against ground truth absorption labels: need to run sae-spelling once (~15 min). Total: <1 GPU-hour.

**Reusable components**:
- SAELens (decoder weights available directly via `sae.W_dec`)
- Neuronpedia activation frequency statistics (pre-computed for many SAEs)
- sae-spelling (for ground truth labels)

---

## Phase 3: Self-Critique

### Against Candidate A

**Implementation reality check**: The core computation is straightforward — load SAE, run sae-spelling, collect (config, absorption_rate) tuples, fit regression. The sae-spelling repo has a working example notebook. No hidden implementation complexity. Risk: Gemma Scope SAEs are large (~10 GB for 2B model); need careful memory management when switching SAE widths.

**Reproducibility attack**: High. Gemma Scope SAEs are public, sae-spelling is open-source, absorption metric is deterministic given the probe training seed. The regression fitting is standard. A reader could reproduce this in a weekend.

**Baseline sanity check**: Chanin et al. already have "absorption rate vs. L0" for some layer configurations (Figure 7b). We need to claim something beyond this: (a) systematic survey across ALL layers and widths, (b) quantitative prediction model (not just trend), (c) incorporating co-occurrence statistics as an explanatory variable. The question is whether the regression adds value beyond "lower L0 → more absorption." Risk: the result may be "L0 and width explain 90% of variance, corpus statistics add nothing." This is still a useful null result for the community.

**Scope attack**: The first-letter task is a known limitation. However, Candidate A's value is precisely in being systematic about this one well-validated domain — the scaling laws themselves are the contribution, not domain breadth. The result would be: "here is a formula that predicts absorption rate, which enables practitioners to select SAE configurations to target a desired absorption threshold."

**Verdict**: STRONG — Engineering-straightforward, builds directly on existing validated code, produces actionable quantitative output, fills a specific identified gap (Gap 1).

---

### Against Candidate B

**Implementation reality check**: The key risk is building the hierarchy probe datasets. "Word is a country" sounds simple but requires: a reliable country wordlist, handling capitalization and tokenization (GPT-2/Gemma tokenizers split "France" differently than "france"), and enough positive examples per class to train a stable LR probe. Expected issue: some hierarchies will have very few positive tokens in the test corpus, making probes unstable. Mitigation: filter to hierarchies with ≥ 500 positive examples in OpenWebText.

**Reproducibility attack**: Moderate. The hierarchy definitions are researcher choices ("which 100 country names to include?"). Need to make wordlists public. The probe training is stochastic but can be seeded.

**Baseline sanity check**: The comparison to first-letter absorption (15–35%) is the baseline. If absorption in country-name hierarchy is also 15–35%, the result is "absorption generalizes." If it's 5% or 60%, something interesting is happening. Either outcome is publishable.

**Scope attack**: The core risk is that the paper becomes "we measured absorption in 5 more domains and found similar rates." Without a theoretical explanation for why rates differ across domains, this is a purely descriptive contribution. Mitigation: pair with Candidate A (scaling laws) — use the domain comparison to validate that the L0/width scaling model generalizes across domains.

**Verdict**: MODERATE — Implementation is doable but requires more custom work than Candidate A. Best as a secondary experiment that extends Candidate A rather than a standalone.

---

### Against Candidate C

**Implementation reality check**: Decoder geometry analysis is genuinely fast (minutes). The core question is whether it actually detects absorption. The LessWrong "Looking for Feature Absorption Automatically" post is a direct negative result for a closely related approach (downstream causal similarity → didn't produce bimodal distribution). Decoder cosine similarity is a different proxy, but the failure mode is similar: high cosine similarity between decoder directions does not guarantee they were involved in absorption (two features can have similar directions because they encode similar concepts, not because one absorbed the other).

**Reproducibility attack**: Very high — fully deterministic from decoder weights.

**Baseline sanity check**: From the LessWrong analysis, the naive causal approach had poor precision/recall. Decoder geometry alone is weaker evidence (no causal information). If precision on the first-letter task is <20%, the approach is not useful.

**Scope attack**: If this works, it's a major contribution (unsupervised absorption detection, Gap 7). If it doesn't work (likely based on the related failure), it's a negative result that's still publishable but weaker.

**Verdict**: WEAK-to-MODERATE — There's a strong prior that geometry-only approaches fail (negative result from LessWrong), and the theoretical justification is speculative. Worth a 30-minute pilot to check precision/recall; drop immediately if precision < 30% vs. ground truth absorption labels. Keep only if pilot shows surprisingly strong signal.

---

## Phase 4: Refinement

**Dropped:** Candidate C (geometry-only approach) — The negative result from the LessWrong post is a strong empirical prior against this approach. The 30-minute pilot is still worth doing to confirm, but this should not be the primary direction.

**Strengthened front-runners:**

**Candidate A (Scaling Laws Survey)** is elevated to front-runner with the following refinements:
- Expand scope beyond just L0 and width: also vary **layer index**, **training duration** (Gemma Scope provides checkpoints), and add one corpus-derived predictor (co-occurrence frequency ratio of the specific token pairs used in the first-letter task)
- The absorption_rate ~ log(L0) + log(width) + layer + log(cooc_ratio) regression is the main empirical result
- Secondary: an iso-absorption curve showing which (L0, width) combinations achieve absorption rate < 10%, directly actionable for practitioners
- This is training-free, uses only pre-trained Gemma Scope SAEs, and the full experiment across 30 SAEs is feasible in ~8 GPU-hours

**Candidate B (Cross-Domain)** becomes the complementary experiment validating that the scaling model found in Candidate A generalizes:
- Reduce to 3 domains instead of 5: (1) first-letter (baseline, already done), (2) country-name tokens, (3) common English given name tokens
- For each additional domain, use the SAME Gemma Scope SAE subset as Candidate A
- Key question: do the scaling law coefficients (slope of absorption vs. L0) hold across domains?
- If yes: strong generalization claim. If no: the domain-specific differences become a new finding.

**Additional pilot (Candidate C, 30 min):** Load GPT-2 SAE, extract W_dec, compute top-100 cosine-similar pairs, cross-check against sae-spelling first-letter absorption labels. If precision > 40%, develop further. If not, report as negative result in related work section.

**Selected front-runner: Candidate A + Candidate B (paired)**

The combination is uniquely compelling because:
1. Candidate A produces a quantitative tool (scaling law formula) that practitioners can use immediately
2. Candidate B validates the formula is domain-general
3. Together they address Gap 1 (no quantitative theory) and Gap 2 (only first-letter task)
4. The entire analysis is training-free (uses only pre-trained Gemma Scope SAEs)
5. Full implementation reuses sae-spelling + SAELens with minimal new code
6. Total experiment budget: ~15 GPU-hours across all conditions

---

## Phase 5: Final Proposal

### Title
Absorption at Scale: Empirical Scaling Laws for Feature Absorption in Sparse Autoencoders

### Hypothesis
Feature absorption rate in pre-trained SAEs is a predictable function of (L0, SAE width, layer index, corpus co-occurrence frequency ratio between parent and child features), expressible as a log-linear model that generalizes across different feature hierarchy domains.

### Motivation
The community currently has no principled way to select SAE hyperparameters to minimize absorption. Practitioners know "lower L0 → more absorption" (Chanin et al., 2024, Figure 7b), but this informal observation is insufficient for: (a) predicting absorption rate at a given configuration, (b) identifying which (L0, width) combinations achieve acceptable absorption thresholds, or (c) understanding whether the relationship holds in safety-relevant feature domains beyond first-letter spelling. This paper fills that gap by providing the first systematic empirical scaling analysis of feature absorption across the full Gemma Scope SAE matrix, yielding an actionable formula and iso-absorption curves that practitioners can use to configure SAEs for interpretability-critical applications.

### Method

**Step 1: Absorption Rate Survey on Gemma Scope SAEs (Candidate A)**

Using sae-spelling and SAELens:
1. Select 30 Gemma Scope SAEs on Gemma 2 2B: widths {16k, 65k, 131k} × L0 settings {~20, ~50, ~100, ~200, ~400} × layers {8, 12, 20, 25} (some overlap, exact set chosen by availability in Gemma Scope HuggingFace)
2. For each SAE, run the sae-spelling k-sparse probing + absorption measurement across all 26 first-letters
3. Record: (layer, width, L0, letter, absorption_rate, false_negative_rate, mean_f1)
4. Compute corpus co-occurrence statistics: for each letter L, compute P(token_starts_with_L | token_appears_in_context_of_absorbing_token) from OpenWebText sample (~1M tokens); this captures how tightly letter-token co-occurrences are coupled

**Step 2: Regression Modeling**
Fit: `absorption_rate = β₀ + β₁ log(L0) + β₂ log(width) + β₃ layer + β₄ log(cooc_ratio) + ε`
Evaluate: R², per-predictor significance, leave-one-letter-out cross-validation
Derive: iso-absorption contour curves over the (L0, width) space for layers 12 and 20

**Step 3: Cross-Domain Generalization (Candidate B)**
Select 6 Gemma Scope SAEs (3 widths × 2 L0 values) at layer 12. For each:
- Task 1 (baseline): first-letter absorption (re-use Step 1 results)
- Task 2: country-name absorption (define parent = "token is a country name", children = {France, Germany, Japan, ...} — top 100 country names from a curated wordlist, check tokenization)
- Task 3: given-name absorption (parent = "token is a common English given name", children = {Alice, Bob, Charlie, ...} — top 100 names from US SSA data)

For each task: train LR probe on residual stream activations → run k-sparse probing → measure absorption rate. Test whether regression coefficients from Step 2 predict absorption rates across tasks.

**Step 4 (Pilot, 30 min): Geometry-Based Detection**
Load one GPT-2 SAE. Compute top-100 cosine-similar decoder pairs. Cross-reference with sae-spelling absorption labels. Report precision/recall. Drop this thread if precision < 30%; report as negative result.

**Specific libraries:**
- `sae-lens` (SAE loading, activation extraction)
- `sae-spelling` (absorption metric, k-sparse probing)
- `transformers` (Gemma 2 2B)
- `sklearn` (LR probe, regression)
- `statsmodels` (regression with R², significance testing)
- `matplotlib/seaborn` (iso-absorption curves)

### Simplest Version (Pilot Experiment, 15 min)
Load one Gemma Scope SAE (Gemma 2 2B, layer 12, 16k width, L0 ≈ 50). Run sae-spelling absorption measurement for letters A–E only. Confirm absorption rate ≈ 15–35% (matches Chanin et al.). This validates the pipeline works before investing 15 GPU-hours in the full experiment.

### Baselines
1. **Chanin et al. (2024) Figure 7b**: Absorption rate vs. L0 for Gemma Scope layers 0–17. Our contribution: systematic coverage of all layers + regression model + multi-domain validation.
2. **SAEBench absorption scores for Gemma 2 2B SAEs**: Pre-computed for many architectures. Our contribution: causal analysis (decomposing into L0 vs. width vs. layer effects) rather than simple ranking.

Expected performance range on absorption rate from existing literature: 15–35% for standard Gemma Scope SAEs at typical L0 range (20–100).

### Experimental Plan

| Experiment | SAEs | Time estimate | Output |
|---|---|---|---|
| Pilot: 5 letters, 1 SAE | 1 | 15 min | Validate pipeline |
| Survey: all 26 letters, 30 SAEs | 30 | 8 GPU-hours | Main dataset |
| Regression fitting | n/a | 1 hour CPU | Scaling law formula |
| Cross-domain: 3 tasks, 6 SAEs | 6 | 2 GPU-hours | Domain generalization |
| Geometry pilot (Candidate C) | 1 | 30 min | Positive/negative result |

**Ablation schedule:**
- Ablation 1: Regression without co-occurrence term — quantifies whether corpus statistics add explanatory power beyond SAE config alone
- Ablation 2: Per-layer vs. pooled regression — tests whether the scaling relationship is layer-invariant
- Ablation 3: Width-matched L0 comparison — varies only L0 at fixed width, confirms causal direction

**Datasets:** Gemma Scope pre-trained SAEs (400+ available, select 30), OpenWebText for corpus statistics (~1M token sample), constructed wordlists for country-name and given-name tasks.

**Metrics:** Absorption rate (Chanin et al. metric), false-negative rate, mean F1, regression R², cross-validation RMSE.

### Resource Estimate

| Task | GPU | Wall-clock | Model size |
|---|---|---|---|
| Gemma 2 2B load | 1× A100 40GB | 5 min | 2B params |
| Per-SAE absorption measurement | 1× A100 40GB | ~15 min | SAE 16k: ~100MB |
| 30-SAE survey | 1× A100 40GB | 8 hours | Sequential |
| Cross-domain extension | 1× A100 40GB | 2 hours | Same setup |
| **Total** | 1× A100 40GB | **~10 GPU-hours** | — |

All experiments are training-free (loading pre-trained weights, running forward passes and probes). No gradient computation after probe training.

### Risk Assessment

**Engineering risks:**
- *Tokenizer handling for cross-domain hierarchies*: Country names split differently across tokenizers. Mitigation: use only tokens that map to a single token ID in the Gemma 2 tokenizer; pre-filter wordlists accordingly.
- *Memory pressure when comparing SAEs of different widths*: 131k SAE decoder weights are large (~1.3 GB for 2B model). Mitigation: process one SAE at a time, clear GPU memory between SAEs.
- *sae-spelling incompatibility with newer SAELens versions*: sae-spelling was written against SAELens 3.x; check version compatibility before full run. Mitigation: pin SAELens to the version used in sae-spelling's requirements.txt.
- *Probe instability for low-frequency tokens*: Country names occur rarely in OpenWebText; may have <100 positive examples per token, making LR probe unstable. Mitigation: aggregate all country names into a single probe ("is this a country name?") rather than per-country probes.

**Research risks:**
- *The regression R² is low (<0.5)*: This would mean absorption rate is not predictably from these config parameters — itself a publishable finding but a weaker paper. Mitigation: report per-letter and per-layer breakdowns to identify where the relationship holds.
- *Cross-domain absorption rates are identical to first-letter*: Would support "absorption is universal" conclusion. Still publishable; reframe as "absorption severity is architecture-determined, not domain-determined."
- *SAEBench or Chanin et al. results already cover our regression*: Re-read Chanin et al. Figure 7b carefully before submission. Key differentiator we add: (a) regression model with explicit coefficients, (b) co-occurrence predictor, (c) cross-domain validation.

### Novelty Claim

What is genuinely new:
1. **First quantitative regression model** predicting absorption rate from (L0, width, layer, co-occurrence frequency) — enabling practitioners to target a desired absorption rate when selecting SAE hyperparameters
2. **Iso-absorption curves** over the (L0, width) space — a practical tool for SAE configuration selection
3. **Cross-domain validation** showing the first-letter absorption metric generalizes (or how it fails to generalize) to semantically richer feature hierarchies
4. **Co-occurrence frequency as an explanatory variable** — first empirical test of whether corpus statistics predict absorption severity beyond architectural factors

Novelty is "showing that X works surprisingly well (or fails) for Y" territory, but the quantitative regression and iso-absorption curves are concrete actionable outputs that no existing paper provides.

**Strongest related work to distinguish from:** Chanin et al. Figure 7b (absorption vs. L0 trends, but no regression model), SAEBench (absorption score rankings, but no decomposition into causes). Our paper is the causal decomposition and domain-generalization study.

**Target venue:** ICLR 2027 or NeurIPS 2026 (mechanistic interpretability track). Length of work is aligned with a focused empirical paper (4–8 pages main + appendix).

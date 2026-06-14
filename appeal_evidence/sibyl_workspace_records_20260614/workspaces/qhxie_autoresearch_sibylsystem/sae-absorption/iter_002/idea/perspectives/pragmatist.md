# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (https://github.com/lasr-spelling/sae-spelling) — MIT license. Canonical absorption metric implementation by Chanin et al. Directly reusable `feature_absorption_calculator` module. Built on TransformerLens + SAELens. **Code exists and is well-documented.**

2. **SAELens** (https://github.com/jbloomAus/SAELens) — MIT license. Standard SAE training/evaluation library. Supports loading Gemma Scope, GPT-2 SAEs. `SAE.from_pretrained()` for one-line loading of 200+ pre-trained SAEs. **Code exists, actively maintained by Decode/Neuronpedia team.**

3. **SAEBench** (https://github.com/adamkarvonen/SAEBench) — Apache 2.0. Absorption eval at `sae_bench/evals/absorption/main.py`. Extends Chanin et al. metric with partial absorption and multi-latent absorption detection. Probes are one-time cost per layer. **Code exists, includes demo notebook.**

4. **Gemma Scope** (https://huggingface.co/google/gemma-scope) — 400+ JumpReLU SAEs on Gemma 2 2B/9B, all layers, widths 16k/65k/131k/262k. Loadable via SAELens `SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82")`. **Weights exist, free to download.**

5. **Gemma Scope 2** (https://huggingface.co/google/gemma-scope-2) — Extended suite for Gemma 3 family (270M, 1B, 4B, 12B, 27B). Includes transcoders and crosscoders. **Very recent, may be useful for scaling analysis.**

6. **RAVEL dataset** (https://github.com/explanare/ravel, HuggingFace: hij/ravel) — 5 entity types (cities, Nobel laureates, verbs, objects, occupations), 400-800 instances each, 4-6 attributes. City-country-continent hierarchy directly usable for cross-domain absorption measurement. Already integrated into SAEBench for RAVEL disentanglement. **Data exists, ready to use.**

7. **Chanin & Garriga-Alonso "Sparse but Wrong" (arXiv:2508.16560)** — Shows L0 is not a free parameter; incorrect L0 causes feature mixing. Feature absorption is explicitly a consequence of L0 being too low. Code at https://github.com/chanind/sparse-but-wrong-paper. **Directly relevant: connects L0 choice to absorption severity.**

8. **Tilde Research "Rate Distortion Dance of SAEs"** (https://www.tilderesearch.com/blog/rate-distortion-saes) — Informal rate-distortion framing of SAE sparsity-reconstruction tradeoff. **No code; conceptual precursor to the proposed rate-distortion theory. Must cite.**

9. **Matryoshka SAE** (arXiv:2503.17547, ICML 2025) — Best on SAEBench absorption metric. Nested prefix training reduces absorption. Code integrated into SAELens. Pre-trained weights available via SAEBench. **Usable as a comparison baseline.**

10. **OrtSAE** (arXiv:2509.22033) — Reduces absorption 65% via decoder orthogonality penalty. **Code availability needs verification; architecture is simple to reimplement in SAELens.**

11. **SynthSAEBench** (arXiv:2602.14687) — Controlled synthetic benchmark with known ground-truth feature hierarchies. Logistic probes achieve 0.974 F1 while SAEs substantially underperform. **Useful for controlled validation of absorption threshold theory.**

12. **ARENA Probing Tutorial** (http://learn.arena.education/chapter1_transformer_interp/11_probing/intro/) — Comprehensive tutorial on TransformerLens activation extraction + logistic regression probing. Covers PCA visualization, layer sweep, difference-of-means probes. **Educational resource for implementation.**

### Landscape Summary

**What actually works:**
- Loading pre-trained SAEs and running the Chanin et al. absorption metric on the first-letter task is a well-paved road. The `sae-spelling` + `SAELens` + `Gemma Scope` pipeline is battle-tested and takes ~30 minutes per evaluation.
- Logistic regression probes on TransformerLens activations are standard practice. Multiple 2025 papers confirm this works well on Gemma 2 2B.
- RAVEL provides ready-made entity-attribute hierarchies. The city-country-continent hierarchy is the most natural extension of the first-letter task.

**What does not work or is risky:**
- The "Sparse but Wrong" paper (Chanin, Aug 2025) fundamentally complicates the theoretical picture: absorption severity depends on whether L0 was set correctly in the first place. An SAE with "too low" L0 will show more absorption as a byproduct of feature hedging, not just hierarchical absorption. Any theory must account for this confound.
- Most Gemma Scope SAEs are JumpReLU, which SAEBench finds has the **worst** absorption. This is good for measuring absorption (high signal) but means TopK comparisons require different SAE families.
- OrtSAE and ATM SAE pre-trained weights may not be publicly available for Gemma 2 2B, limiting head-to-head comparisons to what SAEBench has already evaluated.
- The hysteresis experiment (Phase E.2 in the current proposal) requires SAE fine-tuning, which violates the "training-free" constraint from the project spec. It is the riskiest experiment in the plan.

**Where the practical gaps are:**
- Nobody has run the Chanin et al. absorption metric on non-spelling hierarchies. The code is hierarchy-agnostic; only the probe target needs changing. This is a straightforward engineering task.
- Nobody has computed pairwise decoder cosine similarity for absorbed vs. non-absorbed feature pairs and checked whether there is a statistical separation. This is a ~30 minute computation.
- Nobody has plotted absorption rate vs. L0 across the full Gemma Scope suite to check for phase-transition-like behavior. The data (SAEBench results + Gemma Scope SAE suite) already exists.

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Characterization (Empirical, Training-Free)

- **Hypothesis**: Feature absorption occurs at statistically detectable rates on entity-type hierarchies (city/country from RAVEL), not just the first-letter spelling task.
- **Implementation sketch**: Fork `sae-spelling`. Replace the first-letter probe with a city-country probe trained on Gemma 2 2B activations using RAVEL data. Run the `feature_absorption_calculator` on Gemma Scope SAEs (layer 12, widths 16k and 65k). Compare absorption rates to shuffled-label null controls.
- **Simplest version**: One hierarchy (city-country), one layer (12), two SAE widths (16k, 65k). Does absorption exceed shuffled null? Binary yes/no answer.
- **Time estimate**: ~3-4 GPU-hours total. Probe training: 15 min. Absorption measurement per SAE: 30-45 min. Two widths + null controls: ~3 hours.
- **Reusable components**: `sae-spelling` (absorption calculator), `SAELens` (SAE loading), `RAVEL` (entity data), Gemma Scope (pre-trained SAEs). All MIT/Apache licensed.

### Candidate B: Decoder Geometry Predicts Absorption (Empirical, Training-Free)

- **Hypothesis**: Absorbed feature pairs have systematically higher decoder cosine similarity than non-absorbed pairs. A simple metric `ASI(p,c) = cos^2(theta_{p,c}) * (freq_p / freq_c)` predicts absorption with AUROC >= 0.70.
- **Implementation sketch**: Load Gemma Scope SAE decoder weights via SAELens. Compute pairwise cosine similarity for all feature pairs co-activating above threshold (0.01). Run Chanin et al. absorption metric to get ground-truth labels. Compute AUROC of ASI against labels.
- **Simplest version**: One SAE (Gemma Scope 16k, layer 12). Compute decoder cosine similarity for absorbed vs. non-absorbed pairs identified by sae-spelling. Wilcoxon rank-sum test. No formula fitting needed for the simplest version.
- **Time estimate**: ~2 GPU-hours. Loading SAE + computing activations: 30 min. Absorption labels from sae-spelling: 30 min. Cosine similarity computation: 15 min. Analysis: 15 min.
- **Reusable components**: SAELens (decoder weight access via `sae.W_dec`), scikit-learn (cosine similarity, AUROC), sae-spelling (ground-truth labels).

### Candidate C: Absorption Scaling Laws Across L0 and Width (Empirical, Training-Free)

- **Hypothesis**: Absorption rate follows a sigmoid function of effective sparsity (1/L0), not a linear function, with an identifiable inflection point (phase-transition-like behavior).
- **Implementation sketch**: Use the full Gemma Scope SAE suite for Gemma 2 2B layer 12 (multiple widths and L0 settings). Run the SAEBench absorption eval on each. Plot absorption rate vs. 1/L0 for each width. Fit sigmoid vs. linear. Likelihood ratio test.
- **Simplest version**: Use existing SAEBench results (already publicly available at neuronpedia.org/sae-bench for 200+ SAEs) rather than re-running evals. If SAEBench absorption scores are available via API/download, this is a pure analysis task requiring zero GPU.
- **Time estimate**: If SAEBench data downloadable: ~1 hour (pure analysis). If re-running: ~5-6 GPU-hours.
- **Reusable components**: SAEBench (evaluation code + pre-computed results), scipy (curve fitting, likelihood ratio test), matplotlib/seaborn (plotting).

---

## Phase 3: Self-Critique

### Against Candidate A: Cross-Domain Absorption Characterization

- **Implementation reality check**: The sae-spelling `feature_absorption_calculator` is specifically designed around the first-letter task's structure (k-sparse probing for letter membership, then false-negative detection). Adapting it to entity-type hierarchies requires changing the probe target but also verifying that the SAE learns identifiable entity-type features in the first place. If the SAE has no clean "city" or "country" latents (which is plausible — these are more distributed than letter membership), the absorption metric may produce garbage. The RAVEL eval in SAEBench measures disentanglement, not absorption — these are different measurements. **Risk: probe quality gate (F1 >= 0.80) may be hard to pass for some hierarchies.**

- **Reproducibility attack**: The first-letter task works because letter membership is a crisp, unambiguous binary classification. Entity-type hierarchies are fuzzier: is "Washington" a city or a state? Is "Paris" in France or Texas? These ambiguities could inflate or deflate absorption rates in hard-to-control ways. The probe quality gate is the main defense, but it only catches gross failures, not subtle label noise.

- **Baseline sanity check**: The shuffled-label null control is the correct baseline. If absorption rate on real labels is not statistically distinguishable from shuffled labels, the finding is null. This is a well-designed control. The risk is not that the comparison is unfair but that the effect is too small to detect at the SAE widths available (16k, 65k). Chanin et al. found 15-35% absorption on spelling — if entity-type absorption is 5-10%, we need more statistical power.

- **Scope attack**: Even if city-country absorption works, it is still a narrow extension (one more hierarchy type). The real question is whether absorption generalizes to **any** hierarchy, which requires testing 3-4 types. Adding grammatical POS hierarchies (noun/verb subtypes) requires POS-tagged data and probes — more engineering. The geographic and entity-type hierarchies from RAVEL are the easiest wins.

- **Verdict**: **STRONG** — the most clearly achievable and publishable result. The code path is well-defined, the data exists, and even a null result (absorption does not generalize to entity hierarchies) is informative and publishable.

### Against Candidate B: Decoder Geometry Predicts Absorption

- **Implementation reality check**: Computing pairwise decoder cosine similarity is straightforward (`W_dec @ W_dec.T`). The challenge is that for a 16k SAE, this is a 16k x 16k matrix (256M entries). Pre-filtering to co-activating pairs is necessary. The frequency ratio `freq_p / freq_c` requires computing activation frequencies on a reference corpus — about 30 minutes of forward passes through Gemma 2 2B. Several 2025 papers (Bricken et al., OrtSAE) already use decoder cosine similarity for feature analysis, so the computational pipeline is validated. **Feasible.**

- **Reproducibility attack**: The ASI formula `cos^2(theta) * (freq_p / freq_c)` has two terms. The cosine term is deterministic from SAE weights. The frequency term depends on the reference corpus — different corpora may give different frequency ratios, especially for rare features. Need to report sensitivity to corpus choice (or use the same corpus as sae-spelling). Also, the formula assumes we know which features form parent-child pairs; in the "probe-free" framing, we do not know this, so ASI is computed for **all** high-cosine pairs, and we hope the absorption pairs are enriched at the top. This enrichment may be weak if many non-hierarchical feature pairs also have high cosine similarity.

- **Baseline sanity check**: The key baseline is: does decoder cosine similarity **alone** (without the frequency ratio) already predict absorption? If cosine similarity achieves AUROC 0.65 and adding the frequency ratio only bumps it to 0.70, the ASI novelty is marginal. Must report both. Also, Bricken et al. already use decoder cosine similarity > 0.7 as a threshold for "shared" features — need to clearly differentiate.

- **Scope attack**: AUROC 0.70 on the first-letter task is a modest bar. The real test is whether ASI generalizes to entity-type hierarchies. If it only works on spelling, it is a narrow result. Cross-domain validation (Phase D.3 in the main proposal) is essential but depends on Candidate A succeeding first.

- **Verdict**: **MODERATE** — the core computation is easy, but the novelty over "just use cosine similarity" is thin. The frequency ratio term is the novel piece but may not add much. Best positioned as a secondary contribution supporting the cross-domain characterization, not a standalone finding.

### Against Candidate C: Absorption Scaling Laws

- **Implementation reality check**: The critical question is whether SAEBench absorption scores are downloadable. A search of the SAEBench repo and Neuronpedia confirms that results are available interactively at neuronpedia.org/sae-bench but individual absorption scores per SAE may require re-running the eval. If re-running is needed, the one-time probe training cost per layer makes this expensive (~30 min per layer). The Gemma Scope suite has SAEs at multiple L0 settings within each width — checking how many distinct L0 values exist per width is critical. If there are only 2-3 L0 values per width (typical for Gemma Scope), a sigmoid-vs-linear comparison has very low statistical power.

- **Reproducibility attack**: The number of data points (distinct L0 values x widths) determines whether sigmoid vs. linear is distinguishable. With Gemma Scope, each width typically has 3-5 L0 variants. That gives maybe 10-15 data points total for the scaling curve — barely enough for a likelihood ratio test. The "phase transition" claim could be an artifact of noisy data with too few points.

- **Baseline sanity check**: The linear baseline is the correct null. If a linear fit explains 90%+ of variance, there is no phase transition. This is a well-designed test. But with few data points, both sigmoid and linear may fit equally well, yielding an inconclusive result.

- **Scope attack**: Even if a phase transition is detected, it is a characterization result, not a mechanistic explanation. The rate-distortion theory (Phase A in the main proposal) provides the mechanism, but without that theory, the scaling curve alone is "interesting but so what?" It needs the theory to be compelling.

- **Verdict**: **MODERATE** — depends entirely on data availability and number of distinct L0 values. If SAEBench data is downloadable with sufficient L0 granularity, this is a cheap win. If not, the compute cost may not justify a potentially inconclusive result. Best as a supporting analysis within a larger paper.

---

## Phase 4: Refinement

### Dropped Ideas

None dropped outright — all three are MODERATE or above. But the relative priority is clear.

### Strengthened Ideas

**Candidate A (Cross-Domain) strengthened to front-runner:**
- Simplified further: Start with city-country hierarchy only (not city-continent or POS — those come later if city-country works). This is the single most impactful experiment because it breaks the "first-letter only" monopoly on absorption measurement.
- Pilot experiment (< 15 min): Train a logistic regression probe on Gemma 2 2B layer 12 activations to classify "is this token a city name?" using RAVEL data. Check F1. If F1 >= 0.80, the hierarchy is probe-accessible and absorption measurement is feasible. If F1 < 0.80, try "is this token a country name?" instead.
- Code confirmation: `sae-spelling` repo confirmed to have `feature_absorption_calculator.py` that takes a probe and an SAE as inputs. The adaptation requires: (1) new probe trained on RAVEL data, (2) new token list (cities instead of letters), (3) running the same absorption detection pipeline.

**Candidate B (Decoder Geometry) simplified to supporting analysis:**
- Drop the ASI formula as a primary contribution. Instead, frame as: "We measure decoder cosine similarity between absorbed and non-absorbed feature pairs and find X." This is a simpler, more defensible claim.
- The full ASI formula becomes a secondary analysis: "Adding the frequency ratio improves AUROC from X to Y."
- Pilot: compute `sae.W_dec @ sae.W_dec.T` for one SAE. Check distribution of cosine similarities. Verify that absorbed pairs (from sae-spelling ground truth) cluster at higher cosine similarity. 10 minutes.

**Candidate C (Scaling Laws) repositioned as low-cost bonus:**
- First check: download SAEBench results from Neuronpedia. If absorption scores per SAE are available, plot absorption vs. L0 immediately. Zero GPU cost.
- If not available, run SAEBench absorption eval on the Gemma Scope suite for layer 12 only (~3 hours). This gives the data for both the scaling analysis and the cross-domain experiments (shared probe training cost).
- Critical constraint: "Sparse but Wrong" (Chanin, Aug 2025) shows that incorrect L0 causes feature mixing. Must control for this — compare absorption rates only among SAEs where L0 is "correctly set" (per their proxy metric) vs. the full range. This is a non-trivial confound that the current proposal ignores.

### Selected Front-Runner

**Candidate A: Cross-Domain Absorption Characterization** — highest success probability, clearest novelty, most directly publishable.

Combined with decoder geometry analysis (simplified Candidate B) as the mechanistic explanation, and scaling analysis (Candidate C) as supporting evidence if data is available cheaply.

### Critical Engineering Decision: The "Sparse but Wrong" Confound

The "Sparse but Wrong" paper (Chanin & Garriga-Alonso, Aug 2025) is a late-breaking result that directly impacts this work. It shows that absorption partly overlaps with feature hedging at incorrect L0. The current proposal's rate-distortion theory assumes absorption is a distinct phenomenon from feature hedging, but "Sparse but Wrong" shows they share a common cause (sparsity pressure forces feature mixing).

**Practical implication**: Any cross-domain absorption measurement must report whether the SAEs being tested have "correct" L0 (per the "Sparse but Wrong" proxy metric). Otherwise, measured "absorption" may actually be feature hedging from incorrect L0 choice. This is a critical control that is missing from the current proposal.

**Recommendation**: Add a new control experiment — run the "Sparse but Wrong" L0 diagnostic on all Gemma Scope SAEs used in the study. Report absorption rates separately for SAEs with "correct" vs. "incorrect" L0. This adds ~30 minutes of computation but dramatically strengthens the paper.

---

## Phase 5: Final Proposal

### Title

Measuring Feature Absorption Beyond Spelling: Cross-Domain Characterization and Decoder Geometry Analysis of a Fundamental SAE Failure Mode

### Hypothesis

Feature absorption — the systematic failure of SAE latents to activate on hierarchically related inputs — generalizes beyond the first-letter spelling task to semantically rich entity-attribute hierarchies (city-country, city-continent) and is predictable from the decoder cosine similarity between parent and child feature directions.

Precisely falsifiable:
- H1: Absorption rate on city-country hierarchy exceeds shuffled-null control by ratio >= 3.0 (p < 0.01, Bonferroni-corrected).
- H2: Absorbed feature pairs have significantly higher decoder cosine similarity than non-absorbed pairs (Wilcoxon rank-sum p < 0.01, Cohen's d >= 0.5).

### Motivation

The entire quantitative understanding of feature absorption rests on one proxy task: first-letter spelling. SAEBench explicitly acknowledges this limitation: "SAEBench evaluates feature absorption by using features for 'word starts with X', which is not useful for evaluating domain-specific feature absorption." Meanwhile, the safety applications that motivate SAE research — detecting deception, bias, harmful intent — all involve semantically rich hierarchies (entity types, knowledge structures) that are nothing like letter membership. DeepMind deprioritized SAE research in 2025 partly because SAE probes fail on safety-relevant downstream tasks, with feature absorption as a key culprit. We need to know: does absorption affect the semantic features that actually matter?

### Method

**Step-by-step implementation plan:**

1. **Setup (30 min)**
   - Install: `pip install sae-lens transformer-lens` + clone `sae-spelling`
   - Load Gemma 2 2B via TransformerLens
   - Load Gemma Scope SAEs via SAELens: `SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82")`
   - Load RAVEL dataset: `datasets.load_dataset("hij/ravel")`

2. **Probe training (45 min)**
   - Extract Gemma 2 2B layer 12 residual stream activations for RAVEL city tokens
   - Train logistic regression probes (scikit-learn, L2-regularized) for:
     - "is-city" (binary: city name vs. non-city token)
     - "country-of-city" (multi-class: which country does this city belong to)
     - "continent-of-city" (multi-class: which continent)
   - Quality gate: require F1 >= 0.80 on held-out test set
   - Also train first-letter probes as baseline (reuse sae-spelling code)

3. **Cross-domain absorption measurement (60 min)**
   - Adapt `sae_spelling.feature_absorption_calculator` to take entity-attribute probes instead of letter probes
   - Core adaptation: replace "letter" with "country" in the probe target; the false-negative detection logic is hierarchy-agnostic
   - Run on Gemma Scope SAEs: layer 12, widths 16k and 65k
   - Run shuffled-label null controls: 100 permutations per hierarchy type
   - Compute: absorption rate, ratio-to-null, 95% bootstrap CI

4. **Decoder geometry analysis (30 min)**
   - For absorbed feature pairs identified in step 3, compute decoder cosine similarity: `cos_sim = sae.W_dec[idx_absorber] @ sae.W_dec[idx_absorbee] / (norms)`
   - For non-absorbed pairs with similar co-occurrence rates, compute same
   - Wilcoxon rank-sum test + Cohen's d for effect size
   - Plot distributions (absorbed vs. non-absorbed)

5. **L0 diagnostic control (30 min)**
   - Run the "Sparse but Wrong" L0 proxy metric on all Gemma Scope SAEs used
   - Report absorption rates separately for SAEs with correct vs. incorrect L0
   - This controls for the feature hedging confound

6. **Cross-architecture comparison (45 min)**
   - Compare absorption rates: Gemma Scope JumpReLU (baseline) vs. Matryoshka (from SAEBench, if available for layer 12)
   - Report whether cross-domain absorption reduction from Matryoshka matches first-letter task reduction

### Simplest version

The absolute minimum experiment that tests the core claim: Train one city-country logistic regression probe on Gemma 2 2B layer 12. Run the sae-spelling absorption calculator on one Gemma Scope 16k SAE. Compare absorption rate to 100 shuffled-label permutations. Report whether absorption is detectable (ratio-to-null >= 3.0, p < 0.01). **One afternoon, one GPU.**

### Baselines

1. **First-letter spelling absorption** (Chanin et al., 2024): Expected absorption rate 15-35% on Gemma Scope 16k/65k SAEs. This is the established baseline from the original paper, directly reproducible with sae-spelling.

2. **Shuffled-label null control**: Randomize city-country assignments and re-run absorption measurement. Expected absorption rate: < 5% (near zero, since random probe directions should not systematically align with SAE decoder directions). This is the critical control that establishes whether measured absorption is a real phenomenon or a statistical artifact.

### Experimental plan

| Experiment | Time | GPU? | Falsification |
|---|---|---|---|
| Pilot: city-country probe quality | 15 min | Yes | F1 < 0.80 -> try different hierarchy |
| Baseline: first-letter absorption on Gemma Scope 16k | 30 min | Yes | Replicates Chanin et al. |
| Core: city-country absorption on Gemma Scope 16k | 45 min | Yes | Ratio-to-null < 1.5 |
| Core: city-country absorption on Gemma Scope 65k | 45 min | Yes | Ratio-to-null < 1.5 |
| Null: shuffled-label controls (100 perms) | 30 min | Yes | N/A (establishes noise floor) |
| Decoder geometry: absorbed vs. non-absorbed | 15 min | Minimal | p > 0.05 or d < 0.3 |
| L0 diagnostic: "Sparse but Wrong" control | 30 min | Minimal | N/A (confound control) |
| Extension: city-continent absorption | 45 min | Yes | Same as above |
| Extension: grammatical POS hierarchy | 60 min | Yes | F1 < 0.80 for POS probes |
| Architecture comparison: Matryoshka vs. TopK | 45 min | Yes | No difference in cross-domain absorption |

**Total: ~6-7 GPU-hours on single A100 (or equivalent).** Gemma 2 2B fits in ~10GB VRAM; SAEs add ~200MB each. Well within a single GPU day.

### Resource estimate

- **GPU**: Single A100 (40GB) or equivalent. Gemma 2 2B: ~10GB. Activation caching for probes: ~5GB.
- **GPU-hours**: 6-7 hours total.
- **Models**: Gemma 2 2B (HuggingFace, free), Gemma Scope SAEs (HuggingFace, free).
- **Datasets**: RAVEL (HuggingFace, free), sae-spelling first-letter data (GitHub, free).
- **Code base**: sae-spelling (MIT), SAELens (MIT), SAEBench (Apache 2.0).
- **Storage**: ~20GB for model weights + activation caches.

### Risk assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| City-country probe F1 < 0.80 | High | Medium | Try city-continent (coarser hierarchy); try different layer (8, 20); try country-continent (may be easier) |
| Cross-domain absorption indistinguishable from null | High | Medium | Report as informative null result: "absorption is specific to syntactic/spelling hierarchies." Still publishable as it refines the scope of the phenomenon and addresses a concrete open question. |
| sae-spelling code incompatible with RAVEL adaptation | Medium | Low | The absorption calculator takes generic probes; main risk is prompt template mismatch. Write thin adapter layer. |
| "Sparse but Wrong" confound dominates results | Medium | Medium | L0 diagnostic control in step 5 addresses this directly. If L0 explains most absorption variance, that is itself an important finding. |
| Gemma 2 2B does not learn clean entity-type representations at layer 12 | Medium | Low | RAVEL disentanglement results (SAEBench) confirm Gemma 2 2B encodes city-country information. Probing should work. |
| Decoder cosine similarity does not separate absorbed from non-absorbed | Medium | Medium | Drop as contribution; focus paper on cross-domain characterization alone. |

### Novelty claim

**What is new:** The first systematic measurement of feature absorption on semantically rich entity-attribute hierarchies beyond the first-letter spelling task. This directly addresses the most explicitly acknowledged gap in the SAE evaluation literature (quoted by SAEBench: "not useful for evaluating domain-specific feature absorption").

**What is not new:** The absorption metric itself (Chanin et al., 2024), the probe training methodology (standard), the decoder cosine similarity analysis (used by multiple 2025 papers). The novelty is applying established tools to a new domain, not inventing new tools.

**Why this is enough:** In the current SAE discourse, the practical credibility of absorption as a failure mode rests on one narrow task. Either absorption generalizes (validates the concern, motivates mitigation research) or it does not (suggests absorption is an artifact of the spelling task's peculiar structure, which is also important to know). Both outcomes are publishable at a top venue as a systematic empirical contribution.

### Relationship to current proposal

The current proposal ("When Sparsity Eats Its Young") is ambitious and theoretically elegant but packs 4 distinct contributions (rate-distortion theory, cross-domain characterization, ASI, phase transition) into one paper. From an engineering standpoint, the highest-risk components are:

1. **Rate-distortion theory (Phase A)**: The closed-form threshold `lambda > sin^2(theta)` is elegant but may be too simplified. Real SAEs have non-convex loss landscapes, feature interference beyond pairwise interactions, and the "Sparse but Wrong" confound (incorrect L0 causes feature mixing that looks like absorption). The theory may achieve AUROC 0.55-0.65 rather than the target 0.70, which would weaken the paper's theoretical claim. **Recommendation: proceed but prepare fallback framing as "the theory provides qualitative insight even if quantitative prediction is imprecise."**

2. **ASI as probe-free detection (Phase D)**: The formula `cos^2(theta) * (freq_p / freq_c)` is simple but the frequency ratio may not add much over cosine similarity alone. Bricken et al. already use decoder cosine similarity for feature classification. **Recommendation: position as secondary contribution, not primary claim. Test whether cosine alone achieves comparable AUROC.**

3. **Phase transition and hysteresis (Phase E)**: The hysteresis test (E.2) requires SAE fine-tuning, which violates the training-free constraint. The phase transition detection (E.1) needs sufficient L0 granularity in the Gemma Scope suite. With 3-5 L0 values per width, the statistical power is low. **Recommendation: attempt with available data; if inconclusive, report as "insufficient data points to distinguish sigmoid from linear" rather than claiming a null result.**

4. **Cross-domain characterization (Phase C)**: This is the safest bet — well-defined code path, existing data, clear success/failure criteria. **Recommendation: prioritize this as the primary empirical contribution. Everything else supports it.**

My recommended ordering of effort:
1. Cross-domain absorption (Candidate A) — primary contribution, do first
2. Decoder geometry (simplified Candidate B) — mechanistic support, do second
3. L0 diagnostic ("Sparse but Wrong" control) — critical confound control, do third
4. Theory validation (rate-distortion threshold) — test predictions against cross-domain data, do fourth
5. Scaling analysis (Candidate C) — bonus if SAEBench data is available cheaply
6. Hysteresis experiment — only if all above succeed and time permits

This ordering maximizes the probability of having a publishable paper even if the more ambitious components fail. The cross-domain result stands on its own. Adding decoder geometry and theory support makes it a NeurIPS/ICLR paper. The phase transition dynamics, if they work, elevate it further.

# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024) "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) --- Defines the canonical absorption metric: k-sparse probing to find feature splits, false-negative identification via LR probe, integrated-gradients ablation to confirm causal responsibility of absorbing latent. Key evaluation insight: the metric has hard-coded thresholds (cosine similarity > 0.025, ablation effect gap >= 1.0) chosen by manual inspection rather than principled calibration. Only functional up to layer ~18 in Gemma 2 2B due to attention moving information to final token position.

2. **SAEBench (Karvonen et al., 2025)** (arXiv:2503.09532, ICML 2025) --- 8-metric evaluation suite across 200+ SAEs. Critical methodological lesson: proxy metrics (CE loss recovered, L0) do not reliably predict practical performance. Absorption is one of 8 metrics. Recommends training SAEs across a range of sparsities (L0 in [20, 200]) with matched baselines. Notes substantial inter-run noise, making small improvements hard to distinguish from statistical noise.

3. **Feature Sensitivity metric (Tian et al., 2025)** (arXiv:2509.23717) --- Frames absorption as a special case of poor feature sensitivity. Proposes LLM-generated semantic paraphrases as test inputs. Key evaluation finding: many features that appear monosemantic from activation examples have poor sensitivity. Demonstrates robustness to threshold cutoff choices. Reveals sensitivity-width tradeoff independent of frequency distribution.

4. **SynthSAEBench (2026)** (arXiv:2602.14687) --- Synthetic benchmark with known ground-truth feature hierarchies (forest structure, Zipfian firing rates, configurable correlation). Logistic probes achieve F1=0.974 while best SAEs substantially underperform. Enables controlled study of hierarchy depth and co-occurrence effects on absorption without confounds from unknown LLM feature structure.

5. **"Revising and Falsifying SAE Feature Explanations" (Ma et al., NeurIPS 2025)** --- Introduces similarity-based falsification strategy using close negative examples. Key methodological contribution: exposes recall bias in evaluation methods using random negatives. Demonstrates that structured component-based explanations and iterative refinement improve precision over one-shot generation.

6. **"Stop Probing, Start Coding" (2025)** (arXiv:2603.28744) --- Controlled ablation experiments that disentangle dictionary quality from inference quality. Key finding: swapping the encoder for per-sample inference on the same dictionary does not improve recovery; the dictionary itself is the binding constraint. This is critical for absorption studies: absorption may be a dictionary problem, not an inference problem.

7. **"Towards Principled Evaluations of SAEs" (Makelov et al., 2024)** (arXiv:2405.08366) --- Proposes interpretation-aware sufficiency/necessity tests. Documents the feature magnitude confound: when two causally relevant attributes share an activation, SAEs preferentially learn features for the higher-magnitude attribute. This is directly relevant to absorption where high-frequency parent features compete with low-frequency child features.

8. **"Sparse but Wrong" (Chanin & Garriga-Alonso, 2025)** (arXiv:2508.16560) --- Demonstrates most open-source SAEs have L0 too low, causing feature hedging/mixing. Proposes proxy metric for correct L0. Critical confound for absorption studies: what is measured as "absorption" may partially be L0-induced hedging. Any rigorous absorption study must control for this.

9. **Feature Hedging paper (Chanin et al., 2025)** (arXiv:2505.11756) --- Theoretically and empirically shows narrow SAEs merge correlated features. Balanced Matryoshka SAE with compound multiplier ~0.75 as mitigation. Key finding: Matryoshka SAEs trade absorption for hedging, meaning mitigation claims must be evaluated on BOTH failure modes simultaneously.

10. **"Sanity Checks: Do SAEs Beat Random Baselines?" (Korznikov et al., 2026)** (arXiv:2602.14111) --- SAEs recover only 9% of true features in synthetic settings; random baselines match trained SAEs on some metrics. Establishes a critical baseline: any absorption study must show that its measurements meaningfully distinguish trained SAEs from random baselines.

11. **"Which SAE Features Are Real? Model-X Knockoffs for FDR Control" (2025)** (arXiv:2511.11711) --- Applies statistical false discovery rate control to determine which SAE features are "real." Provides rigorous statistical methodology applicable to filtering genuine absorption from noise.

12. **Absorption metric implementation** (sae-spelling GitHub: https://github.com/lasr-spelling/sae-spelling) --- The `feature_absorption_calculator` module provides the canonical implementation. Key limitation: integrated-gradients ablation is computationally expensive (~minutes per letter per SAE). The paper suggests a faster alternative: projecting firing latents against LR probe direction, but this has not been validated as equivalent.

### Experimental Landscape

**What has been properly tested:**
- Feature absorption existence on the first-letter spelling task across Gemma Scope (16k, 65k widths), Llama 3.2 1B, and Qwen2 0.5B SAEs (Chanin et al., 2024)
- Architecture comparison on SAEBench absorption metric: Matryoshka (~0.03) < OrtSAE < BatchTopK (~0.29); JumpReLU worst (Karvonen et al., 2025)
- ATM SAE claims best absorption (0.0068) but only evaluated on Gemma-2-2B (Li et al., 2025)
- Absorption rate range: 15--35% across standard configurations

**What is accepted without proper evidence:**
- That the first-letter spelling task is representative of absorption in general feature hierarchies. No cross-domain validation exists.
- That the absorption metric's hard-coded thresholds (cosine sim > 0.025, gap >= 1.0) are optimal or even appropriate. No sensitivity analysis of these thresholds has been published.
- That absorption and hedging are cleanly separable phenomena in practice. Both manifest as false negatives and no study has systematically measured their relative contributions.
- That ATM and KronSAE absorption numbers are comparable to SAEBench numbers (different evaluation setups, models, and implementation details).

**Where critical methodological gaps exist:**
- No statistical power analysis for absorption rate measurements. With 26 letters and variable false-negative counts per letter, the per-SAE absorption rate estimate may have high variance.
- No bootstrap confidence intervals or significance tests reported for absorption rate comparisons between architectures.
- No control for the L0 confound: SAEs compared at different effective L0 levels conflate absorption with hedging.
- No systematic sensitivity analysis of the absorption metric's hyperparameters (cosine similarity threshold, ablation gap threshold, number of k-sparse probing features).
- The absorption metric requires pre-known probe directions, making it impossible to discover absorption of features the researcher did not anticipate.

---

## Phase 2: Initial Candidates

### Candidate A: Anatomy of Absorption -- Disentangling Hierarchy-Driven Absorption from L0-Induced Hedging

- **Hypothesis**: At least 30% of what the canonical absorption metric measures on standard pre-trained SAEs (Gemma Scope, SAEBench) is attributable to incorrect L0 (the L0 confound) rather than true hierarchy-driven absorption. Specifically, when SAEs are evaluated at their "correct" L0 (as defined by Chanin & Garriga-Alonso's proxy metric), the measured absorption rate will drop by at least 30% compared to their default L0.
- **Falsification criterion**: If absorption rate at correct L0 is within 10% of absorption rate at default L0 across all tested SAE configurations (>= 20 SAEs spanning 3+ architectures), this hypothesis is falsified. L0 is not a meaningful confound.
- **Evaluation protocol**: (1) Select 30+ pre-trained SAEs from SAEBench (BatchTopK, TopK, JumpReLU, Matryoshka) across 3 widths and 6 sparsities on Gemma-2-2B layer 12. (2) For each SAE, compute absorption rate using Chanin et al.'s canonical metric on the first-letter spelling task. (3) For each SAE, determine whether its L0 is "correct" using the proxy metric from Chanin & Garriga-Alonso (2025). (4) Group SAEs into "correct L0" vs "incorrect L0" bins and compare absorption rates. (5) Within the incorrect-L0 group, retrain or select SAEs at corrected L0 and re-measure absorption. (6) Statistical test: paired Wilcoxon signed-rank test (non-parametric, no normality assumption) for absorption rate at default vs. corrected L0.
- **Ablation plan**: (a) Ablate L0 correction: measure absorption at 5 L0 levels spanning 0.5x to 2x the default, to map the absorption-vs-L0 curve. (b) Ablate architecture: does the L0 confound affect all architectures equally, or is it specific to TopK/JumpReLU? (c) Ablate width: does the L0 confound scale with dictionary width?
- **Confounders identified**: (1) The "correct L0" proxy metric from Chanin & Garriga-Alonso may itself be inaccurate. (2) Changing L0 also changes reconstruction fidelity, so absorption differences may be driven by reconstruction quality rather than L0 correction per se. (3) SAEBench SAEs were trained with specific hyperparameters that may not be representative.
- **Pilot design**: Take 3 SAEBench SAEs (one BatchTopK, one TopK, one JumpReLU) on Gemma-2-2B layer 12 at width 16k. Run Chanin et al.'s absorption metric on each. Compute L0 correctness proxy. If any show incorrect L0, this gives early signal. Estimated time: ~10 minutes with pre-trained SAEs and cached activations.

### Candidate B: Cross-Domain Absorption Taxonomy -- Do Knowledge Hierarchies Absorb Like Spelling Hierarchies?

- **Hypothesis**: Feature absorption generalizes beyond first-letter spelling to knowledge-domain hierarchies (country-city, animal taxonomy, grammatical category-word), and the absorption rate in knowledge domains will be correlated (Spearman rho > 0.5) with the absorption rate on the spelling task across the same set of SAEs.
- **Falsification criterion**: If the Spearman correlation between spelling-task absorption rate and knowledge-domain absorption rate across >= 20 SAEs is below 0.3 (with 95% CI not overlapping 0.5), the hypothesis that absorption generalizes uniformly is falsified. This would mean absorption behavior is task-specific, and the spelling task is not a reliable proxy.
- **Evaluation protocol**: (1) Define 3 knowledge-hierarchy probe tasks with known parent-child structure: (a) City-Country (given a city token, does the model encode its country?), (b) Animal-Taxonomy (given an animal token, does the model encode its taxonomic class?), (c) Grammatical category (given a word, does the model encode its part-of-speech super-category?). (2) For each task, train LR probes to establish the "should-fire" direction, analogous to the spelling-task probe. (3) Adapt the sae-spelling absorption metric to these probe tasks. (4) Evaluate on >= 20 SAEs from Gemma Scope and SAEBench. (5) Compute Spearman rank correlation of absorption rate across tasks. (6) Bootstrap 95% CIs for all correlations.
- **Ablation plan**: (a) Ablate domain: test each knowledge hierarchy independently to identify which hierarchy types are most vulnerable to absorption. (b) Ablate hierarchy depth: compare 2-level hierarchies (city-country) vs. 3-level (city-country-continent) to test whether deeper hierarchies amplify absorption. (c) Ablate frequency: control for feature frequency by selecting city tokens with matched frequency distributions across high/low absorption groups.
- **Confounders identified**: (1) Knowledge-domain probes may have lower accuracy than spelling probes, introducing noise that inflates apparent absorption. Control: only use probes with accuracy >= 0.90. (2) The LLM may not encode these hierarchies in the same layers where spelling features live. Control: sweep across layers 5-18 for Gemma-2-2B. (3) Feature frequency distributions differ across domains and may confound absorption rate comparisons. Control: frequency-matched subsampling.
- **Pilot design**: Train city-country LR probes on Gemma-2-2B layers 8, 12, 16 (3 layers) using 50 well-known cities. Check probe accuracy (need >= 0.90). Then run absorption metric on one Gemma Scope 16k SAE at the best probe layer. Estimated time: ~15 minutes (probe training ~5 min, absorption calculation ~10 min).

### Candidate C: Absorption Metric Robustness Audit -- How Sensitive Are Published Numbers to Arbitrary Thresholds?

- **Hypothesis**: The canonical absorption metric's reported rates are highly sensitive to its hard-coded hyperparameters (cosine similarity threshold = 0.025, ablation gap >= 1.0). Varying these thresholds by +/-50% will change absorption rates by more than 2x for at least half of tested SAEs, meaning published architecture rankings may not be robust.
- **Falsification criterion**: If absorption rates change by less than 20% across all tested threshold combinations for >= 80% of SAEs, the metric is robust and the hypothesis is falsified.
- **Evaluation protocol**: (1) Select 10 SAEs from SAEBench spanning 3 architectures and 2 widths. (2) Run the full absorption pipeline with a grid of threshold combinations: cosine similarity in {0.01, 0.015, 0.025, 0.04, 0.06} and ablation gap in {0.5, 0.75, 1.0, 1.5, 2.0} (25 combinations). (3) Record absorption rate at each combination. (4) Plot absorption rate as a function of thresholds (heatmap per SAE). (5) Test whether architecture rankings (which architecture has lowest absorption) are invariant to threshold choice using Kendall's tau-b rank correlation across threshold settings. (6) Propose calibrated thresholds: use the ROC curve where "positive" = manually labeled absorption cases and "negative" = non-absorption cases, if manual labels are available from Chanin et al.'s supplementary data.
- **Ablation plan**: (a) Threshold sensitivity per architecture: do some architectures' absorption rates change more with thresholds than others? (b) False-positive rate: at very low cosine similarity thresholds, how many "absorptions" are spurious? (c) Replace integrated-gradients ablation with the faster projection-based alternative from Chanin et al. and check if rankings change.
- **Confounders identified**: (1) The thresholds may interact: changing cosine similarity threshold may render the ablation gap threshold irrelevant or vice versa. Control: test all combinations, not just marginal sweeps. (2) Different letters may have different natural threshold sensitivities. Control: report per-letter analysis. (3) Computational expense of running 25x the normal evaluation. Control: subsample to 10 SAEs.
- **Pilot design**: Take the single most-studied SAE (Gemma Scope 16k, layer 12) and run absorption at 5 cosine similarity thresholds with the default ablation gap. If absorption rate varies by more than 30%, early signal is strong. Estimated time: ~12 minutes (5x the single-threshold computation with cached activations).

---

## Phase 3: Self-Critique

### Against Candidate A: Disentangling Absorption from L0-Induced Hedging

- **Confound attack**: The "correct L0" proxy metric (Chanin & Garriga-Alonso, 2025) was validated on toy models and limited real-world settings. If the proxy itself is inaccurate, the entire disentanglement falls apart. Additionally, changing L0 changes the entire feature decomposition, not just hedging behavior --- so comparing absorption at different L0 levels compares fundamentally different feature sets, making causal attribution difficult. A cleaner experiment would vary L0 while holding the decoder dictionary fixed (e.g., by post-hoc adjusting activation thresholds), but this is only possible for JumpReLU/Gated architectures and introduces its own confounds.

- **Statistical attack**: With ~30 SAEs and comparing absorption rates between "correct" and "incorrect" L0 groups, the statistical power depends on the within-group variance. Given SAEBench reports substantial noise between runs, the 30% effect size may require more SAEs to detect reliably. Power analysis: assuming absorption rate standard deviation of ~0.08 (based on the 0.15--0.35 range reported in literature), detecting a 0.06 difference (30% of 0.20 baseline) with alpha=0.05 and power=0.80 requires approximately n=28 per group (two-sample t-test). This is feasible with 30+ SAEs per group but tight.

- **Benchmark attack**: The first-letter spelling task is well-established for absorption measurement, so the benchmark choice is appropriate for this specific question. However, the finding would have limited generalizability --- we would know that L0 confounds absorption on the spelling task, but not whether this extends to other domains.

- **Ablation completeness attack**: The ablation plan covers L0, architecture, and width, which is comprehensive for the hypothesis. One missing ablation: does the confound depend on the specific layer analyzed? Layer 12 is standard but absorption may manifest differently in earlier or later layers.

- **Verdict**: MODERATE. The core question (how much of measured absorption is actually hedging?) is important and answerable, but the reliance on the L0 correctness proxy introduces a fragile dependency. The experimental design is sound for a descriptive study but the causal claim ("30% of absorption is hedging") requires stronger controls than what is feasible with pre-trained SAEs alone.

### Against Candidate B: Cross-Domain Absorption Taxonomy

- **Confound attack**: The biggest confound is **probe quality**. The first-letter spelling task has near-perfect probe accuracy because the feature is discrete and unambiguous (a token either starts with L or it doesn't). Knowledge-domain probes (city-country, animal-taxonomy) will inevitably have lower accuracy due to: (a) ambiguity (some cities belong to multiple countries historically), (b) the model may not encode this information in the same way (knowledge may be distributed across layers rather than concentrated), (c) tokenization artifacts (multi-token city names). If probe accuracy is 0.85 instead of 0.98, the false-negative rate used to identify absorption candidates will be inflated by probe errors, not absorption. This is a fundamental confound that could invalidate the entire cross-domain comparison. Search for papers: Gurnee & Tegmark (2023, "Language Models Represent Space and Time") showed that geographic knowledge is encoded in intermediate layers of LLMs and is accessible via linear probes with reasonable accuracy (~0.80--0.90 for country identification from city tokens). This provides some support but the accuracy gap from spelling probes remains a concern.

- **Statistical attack**: The Spearman correlation test between spelling and knowledge-domain absorption requires sufficient variance in absorption rates across SAEs. If most SAEs cluster at similar absorption rates for knowledge tasks (e.g., all show high absorption because knowledge hierarchies are harder), the correlation will be noisy and underpowered. With n=20 SAEs, detecting rho=0.5 at alpha=0.05 requires power analysis showing that this is borderline adequate (power ~0.70). Increasing to n=30+ would be safer.

- **Benchmark attack**: The choice of knowledge hierarchies is well-motivated by the literature gaps. However, animal taxonomy and grammatical categories are less natural than city-country because the model may not have clear linear representations for these. The city-country task has the strongest prior support (from Gurnee & Tegmark and the RAVEL benchmark). Starting with city-country as the primary domain and treating others as extensions is methodologically sound.

- **Ablation completeness attack**: The frequency control ablation is essential and well-designed. The layer sweep is important but adds significant computational cost (3 layers x 20 SAEs x absorption metric). The hierarchy depth ablation (2-level vs. 3-level) is novel and informative but introduces the question of how to define the 3-level absorption metric (does the grandparent absorb into the parent, or does the parent absorb into the child?).

- **Verdict**: STRONG. This is the most impactful candidate because it addresses the field's most critical empirical gap (is absorption real beyond spelling?) with a well-structured experimental design. The probe quality confound is serious but manageable with careful controls (accuracy thresholds, frequency matching, layer sweeping). The city-country task in particular has strong prior support.

### Against Candidate C: Absorption Metric Robustness Audit

- **Confound attack**: This is a meta-methodological study, so the confounds are different. The main risk is that the results are specific to the particular implementation in sae-spelling and do not generalize to other absorption measurement approaches. If Chanin et al. release updated thresholds or a revised metric, this study could become immediately obsolete. Additionally, the 25-combination threshold grid may not span the relevant parameter space --- there could be a phase transition at a threshold value outside the tested range.

- **Statistical attack**: With 10 SAEs, reporting "at least half show > 2x change" means detecting 5+ out of 10. This is a binomial test. If the true proportion of sensitive SAEs is 0.5, observing 5+ out of 10 happens 62% of the time even under the null (because 5/10 is exactly the null expectation for p=0.5). The hypothesis needs to be sharper: "at least 80% of SAEs show > 2x change" would be more discriminating. Alternatively, use a quantitative metric: the coefficient of variation of absorption rate across threshold combinations, with a pre-specified threshold for "high sensitivity."

- **Benchmark attack**: The first-letter spelling task is the only domain where the absorption metric has been established, so using it for the robustness audit is correct. However, this limits the generalizability of the robustness finding.

- **Ablation completeness attack**: The plan to test the faster projection-based alternative is valuable and could yield a practical contribution (a validated faster metric). The per-letter analysis controls for letter-specific threshold sensitivity well. Missing: test robustness to the number of k-sparse probing features (k), which determines which latents are considered "feature splits" in the first place.

- **Verdict**: MODERATE. The question is important (are published numbers trustworthy?) but the study is inherently descriptive and the immediate research impact is limited to methodological refinement. The statistical framing needs sharpening, and the contribution would be strengthened by proposing calibrated thresholds rather than merely documenting sensitivity.

---

## Phase 4: Refinement

### Dropped

**Candidate A** (disentangling absorption from hedging) is downgraded to a supporting experiment rather than the primary contribution. Reason: the reliance on the L0 correctness proxy introduces too much uncertainty for the central claim, and the finding would be narrow (applicable only to the spelling task). However, including L0 as a controlled variable in the main experiment would address this concern partially.

### Strengthened: Candidate B (Cross-Domain Absorption Taxonomy)

This is elevated to the front-runner. Key strengthening moves:

1. **Tighter probe quality control**: Use only probe tasks where logistic regression probe accuracy >= 0.92 on held-out data. If animal-taxonomy or grammatical category probes fall below this threshold, drop them from the main analysis and report the negative finding (model does not encode these hierarchies linearly with sufficient fidelity to study absorption).

2. **Add the "controlled null" test**: For each knowledge-domain task, create a matched "non-hierarchical" control where the parent feature is replaced by a random feature. Absorption should be near zero for the non-hierarchical control. This rules out the alternative explanation that what we measure is just noise in probe accuracy, not absorption.

3. **Integrate the robustness audit (Candidate C) as a validation section**: Before reporting cross-domain absorption rates, first demonstrate that the metric is robust to threshold choice on the spelling task (subsample: 5 thresholds x 5 SAEs). This establishes the metric's reliability before extending it.

4. **Integrate L0 confound control (Candidate A) as a covariate**: Report absorption rates with L0 correctness as a binary covariate. This disentangles the confound without making it the central claim.

5. **Pre-register the analysis plan**: Specify all thresholds, statistical tests, and decision criteria before running the cross-domain experiments. This prevents post-hoc threshold tuning.

6. **Add frequency-controlled analysis**: For each domain, bin tokens by frequency (high/medium/low) and report absorption rates per bin. This addresses the confound that absorption severity may correlate with feature frequency rather than hierarchy structure.

7. **Add the absorption-downstream-performance link**: For city-country tasks, measure whether high-absorption SAEs also have worse performance on a downstream task (e.g., factual recall of city-country associations via SAE-reconstructed activations). This provides the motivating "so what?" for the community.

### Selected Front-Runner: Cross-Domain Absorption Taxonomy

**Title working draft**: "Beyond Spelling: Feature Absorption Across Knowledge Hierarchies in Sparse Autoencoders"

The experimental contribution would be the first systematic measurement of feature absorption beyond the first-letter spelling task, establishing (or refuting) the generality of absorption as an SAE failure mode in knowledge-rich hierarchical feature structures. The study includes proper controls (non-hierarchical null, frequency matching, probe quality filtering, L0 covariate, threshold robustness validation) that address every known confound in the existing absorption literature.

---

## Phase 5: Final Proposal

### Title

Beyond Spelling: Feature Absorption Across Knowledge Hierarchies in Sparse Autoencoders

### Hypothesis

Feature absorption in SAEs is not specific to the first-letter spelling task but generalizes to knowledge-domain hierarchies. Specifically: (1) Absorption is measurable (rate > 0.10) for at least 2 out of 3 knowledge-hierarchy tasks (city-country, animal-taxonomy, grammatical category) on Gemma-2-2B SAEs; (2) Cross-task absorption rates are positively correlated (Spearman rho > 0.3) across SAEs, indicating a common underlying mechanism; (3) SAE architectures that mitigate spelling-task absorption (Matryoshka) also mitigate knowledge-domain absorption.

### Falsification Criterion

The hypothesis is falsified if: (a) No knowledge-domain task achieves measurable absorption (rate < 0.05 for all tasks across all SAEs tested), OR (b) The Spearman correlation between spelling-task and knowledge-domain absorption rates is negative or indistinguishable from zero (95% CI includes 0), OR (c) Architecture rankings for absorption are reversed between spelling and knowledge domains (e.g., Matryoshka worst on knowledge tasks while best on spelling).

### Method

Adapt the canonical absorption metric (Chanin et al., 2024) to knowledge-domain probe tasks:

1. **Probe task definition**: For each domain, define the parent feature (general concept) and child features (specific instances):
   - City-Country: Parent = "European country", Child = specific city tokens (Paris, Berlin, Madrid, ...). Hierarchy: city is-in country.
   - Animal-Taxonomy: Parent = "mammal", Child = specific animal tokens (lion, whale, bat, ...). Hierarchy: animal is-a class.
   - Grammatical Category: Parent = "verb", Child = specific verb tokens. Hierarchy: word is-a POS.

2. **Probe training**: Train binary LR probes for each parent feature on Gemma-2-2B residual stream activations at layers 8, 12, and 16. Select the layer with highest probe accuracy for each task. Require accuracy >= 0.92 on held-out data.

3. **Absorption metric adaptation**: For each task:
   - Use k-sparse probing (k=5) to find feature-split SAE latents for the parent feature.
   - Identify false-negative tokens where all k split latents fail to activate but LR probe classifies correctly.
   - Run integrated-gradients ablation on false negatives to identify absorbing latents.
   - Apply the standard absorption criterion (cosine sim > 0.025 with probe, gap >= 1.0).

4. **Controls**:
   - Non-hierarchical control: Replace parent feature with a random binary classification (e.g., "token length > 5") and measure absorption. Expected rate: near zero.
   - Frequency-matched subsampling: For each domain, select child tokens with frequency distribution matched to the first-letter spelling task.
   - L0 covariate: Record each SAE's effective L0 and include as a covariate in regression analysis of absorption rates.

5. **SAE selection**: 24+ SAEs total:
   - 12 from Gemma Scope (JumpReLU, widths 16k/65k, layers 8/12/16 -- 2 widths x 3 layers x 2 for variation)
   - 12 from SAEBench (BatchTopK, TopK, Matryoshka, JumpReLU -- 4 architectures x 3 widths at layer 12)

### Evaluation Protocol

- **Primary benchmarks**: Absorption rate (Chanin et al. metric), computed on each of 4 tasks (spelling + 3 knowledge domains) for each SAE. Total: ~96 absorption rate measurements (24 SAEs x 4 tasks).
- **Metrics**:
  - Absorption rate per task per SAE
  - Spearman rank correlation of absorption rate across tasks (6 pairwise correlations: spelling-city, spelling-animal, spelling-grammar, city-animal, city-grammar, animal-grammar)
  - Architecture ranking concordance: Kendall's tau-b for architecture absorption rankings across tasks
  - Non-hierarchical control absorption rate (sanity check, expect near 0)
- **Statistical test plan**:
  - Bootstrap 95% CIs for all absorption rates (10,000 resamples of letter/category-level absorption)
  - Spearman correlation with permutation test (n=10,000 permutations) for cross-task correlation significance
  - Kendall's tau-b with 95% CI for architecture ranking concordance
  - Mixed-effects logistic regression: absorption (binary) ~ task + architecture + width + L0 + (1|SAE), to identify which factors drive absorption while accounting for repeated measures
- **Number of random seeds**: Not applicable (training-free analysis of pre-trained SAEs), but we will use 3 random train/test splits for probe training to ensure probe stability.

### Ablation Schedule

| # | Component Ablated | What It Tests | Expected Outcome |
|---|---|---|---|
| 1 | Remove knowledge-domain tasks, keep only spelling | Baseline: does our implementation reproduce Chanin et al.'s published numbers? | Absorption rates within 5% of published values |
| 2 | Remove non-hierarchical control | Whether measured absorption is artifact of probe noise vs. genuine hierarchy effect | Control absorption rate should be < 0.03; if not, probe quality is insufficient |
| 3 | Vary probe accuracy threshold (0.85, 0.90, 0.92, 0.95) | Whether probe quality confounds absorption rate | Higher threshold should yield cleaner (possibly lower) absorption rates |
| 4 | Vary absorption metric thresholds (cosine sim: 0.01--0.06; gap: 0.5--2.0) | Metric robustness -- are cross-domain findings stable? | Architecture rankings invariant (tau-b > 0.8) across reasonable threshold range |
| 5 | Replace integrated-gradients ablation with projection-based approximation | Speed vs. accuracy tradeoff of the faster metric | Rankings should be preserved; if not, the approximation is unreliable |
| 6 | Include L0 as covariate vs. exclude | Whether L0 confound explains away cross-domain absorption | L0 should be a significant covariate but not eliminate the task effect |
| 7 | Vary layer (8, 12, 16 for Gemma-2-2B) | Whether absorption is layer-specific or general | Absorption should be present across layers but magnitude may vary |
| 8 | Frequency-matched vs. unmatched token selection | Whether feature frequency drives absorption rate differences across domains | Matched subsampling should reduce but not eliminate cross-domain absorption differences |

### Control Experiments

1. **Non-hierarchical null**: For each domain, replace the parent feature with a non-hierarchical binary classification (e.g., "token has even character count"). Measure absorption. Expected: near-zero absorption, because there is no feature hierarchy to trigger the absorption mechanism. If non-zero, our method has a systematic false-positive problem.

2. **Random SAE baseline**: Initialize an SAE with random weights (no training) and measure absorption rate. This establishes the noise floor of the metric. From Korznikov et al. (2026), random SAEs can match trained SAEs on some metrics, so this control is essential.

3. **Probe direction randomization**: Replace the trained LR probe direction with a random unit vector and re-run the absorption metric. Expected: no absorption detected. This confirms the metric is measuring alignment with the true feature direction, not an artifact of the ablation procedure.

4. **Reproduction of Chanin et al. numbers**: Before running cross-domain experiments, exactly reproduce the published absorption rates on the first-letter spelling task for at least 3 SAEs. This validates our implementation.

### Pilot Design

**Phase 1 (5 minutes)**: Load Gemma-2-2B and one Gemma Scope 16k SAE (layer 12) using SAELens. Train city-country LR probe using 100 well-known cities. Check probe accuracy on held-out 20 cities.

**Phase 2 (10 minutes)**: If probe accuracy >= 0.92, run the adapted absorption metric on the city-country task for this single SAE. Record absorption rate and compare qualitatively to the spelling-task absorption rate for the same SAE.

**Decision gate**: If probe accuracy < 0.85 for city-country, pivot to a different knowledge hierarchy (e.g., company-industry, or language-script) before investing in the full experiment. If absorption rate is exactly 0 for city-country, investigate whether the probe direction is in the SAE's representational space at all (check cosine similarity between probe direction and SAE decoder columns).

### Resource Estimate

- **Models**: Gemma-2-2B (can run on single A100/RTX 4090), pre-trained Gemma Scope SAEs (loaded from HuggingFace), SAEBench SAEs (loaded from HuggingFace)
- **GPU-hours**: 
  - Pilot: ~0.5 GPU-hours
  - Full experiment (24 SAEs x 4 tasks x absorption metric): ~8 GPU-hours (absorption metric ~20 min per SAE-task pair with integrated gradients)
  - Ablation schedule: ~4 GPU-hours (threshold sweeps reuse cached activations; layer sweep adds ~4 GPU-hours)
  - Total: ~12--16 GPU-hours on a single A100
- **Time budget**: Pilot in 15 minutes. Full experiment parallelizable across tasks: ~4 hours wall-clock with sequential SAE processing, ~1 hour with 4 parallel GPU jobs
- **Software**: SAELens v6, TransformerLens, sae-spelling (adapted), PyTorch, scikit-learn (probes)

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Knowledge-domain probes have insufficient accuracy (< 0.85) | High | Medium | Pre-select domains with known linear representation (city-country has prior evidence from Gurnee & Tegmark). Have 2 backup domains (company-industry, language-script). |
| Absorption metric too slow for full grid (25 thresholds x 24 SAEs) | Medium | High | Use projection-based approximation for threshold sweep; validate equivalence on subset first. |
| Cross-domain absorption is indistinguishable from probe noise | High | Medium | Non-hierarchical control experiment. If control shows > 0.03 absorption, increase probe accuracy threshold or conclude the metric is unreliable for this domain. |
| SAEBench SAEs unavailable or incompatible with our pipeline | Medium | Low | Fall back to Gemma Scope SAEs only (still 12+ SAEs with width/layer variation). |
| Results are negative: no cross-domain absorption detected | Medium | Medium | Negative result is publishable and important: it would mean absorption is task-specific, not a general SAE failure mode. Report the finding honestly with full statistical power analysis showing we could have detected rho >= 0.3 if it existed. |
| ATM/OrtSAE pre-trained weights unavailable for Gemma-2-2B | Medium | High | Limit architecture comparison to SAEBench-available architectures (BatchTopK, TopK, JumpReLU, Matryoshka). |

### Novelty Claim

This would be the **first systematic empirical study of feature absorption beyond the first-letter spelling task**, answering the field's most critical open empirical question about absorption: does it generalize? The study contributes:

1. **New evaluation tasks**: Adapted absorption metrics for knowledge-domain hierarchies (city-country, animal-taxonomy, grammatical category) with proper controls.
2. **Cross-domain correlation analysis**: First quantitative test of whether absorption is a unified phenomenon or task-specific artifact.
3. **Controlled methodology**: Non-hierarchical null controls, probe quality filters, frequency matching, L0 covariate analysis, and threshold robustness validation that collectively represent the most rigorous absorption study to date.
4. **Architecture-generality evidence**: First test of whether Matryoshka SAE's absorption advantage extends beyond the spelling task.
5. **Reproducible framework**: A reusable pipeline for measuring absorption on any domain with known feature hierarchies, enabling the community to extend absorption analysis to safety-relevant features and other critical domains.

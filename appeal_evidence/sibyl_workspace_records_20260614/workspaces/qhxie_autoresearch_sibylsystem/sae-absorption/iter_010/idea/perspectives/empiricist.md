# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** -- Defines the canonical absorption metric: k-sparse probing to find feature splits, false-negative identification, integrated-gradients ablation to detect absorbing latents. The gold standard experimental protocol, but restricted to the first-letter spelling task and requires known probe directions.

2. **Karvonen et al. (2025), SAEBench (arXiv:2503.09532, ICML 2025)** -- 8-metric evaluation suite across 200+ SAEs and 7 architectures. Critically reveals that proxy metrics (CE loss, sparsity) do not reliably predict practical performance. The modified absorption metric extends Chanin et al.'s approach to work across all layers by detecting absorbing latents based on probe-direction contribution rather than ablation effect.

3. **SynthSAEBench (arXiv:2602.14687, 2026)** -- Synthetic benchmark with known ground-truth features, hierarchy, correlation, and Zipfian firing distributions. Enables controlled ablation of individual factors. Key finding: logistic probes achieve 0.974 F1 while best SAEs substantially underperform, confirming the probe-SAE gap under controlled conditions.

4. **Chanin & Garriga-Alonso (2025), "Sparse but Wrong" (arXiv:2508.16560)** -- Demonstrates incorrect L0 causes SAEs to learn systematically wrong features. Too-low L0 triggers feature hedging; too-high L0 finds degenerate solutions. Proposes proxy metric for correct L0. Most open-source SAEs have L0 that is too low. Critical confound for any absorption study.

5. **Chanin et al. (2025), "Feature Hedging" (arXiv:2505.11756)** -- Identifies feature hedging as complementary failure mode. Theoretically and empirically shows narrow SAEs merge correlated features. Balanced Matryoshka SAE with compound multiplier ~0.75. Essential for disentangling hedging from absorption in experimental design.

6. **Tian et al. (2025), "Feature Sensitivity" (arXiv:2509.23717)** -- Frames absorption as special case of poor feature sensitivity. Scalable sensitivity evaluation methods that do not require known feature hierarchies. Many interpretable features have poor sensitivity even when activation examples appear monosemantic.

7. **Bussmann et al. (2025), Matryoshka SAE (arXiv:2503.17547, ICML 2025)** -- Absorption rate ~0.03 vs BatchTopK ~0.29. Nested prefix losses create natural feature hierarchy. The best available architecture-level control for absorption. Inner levels suffer hedging, establishing the absorption-hedging tradeoff.

8. **Korznikov et al. (2025), OrtSAE (arXiv:2509.22033)** -- Reduces absorption ~70% vs BatchTopK via orthogonality penalty. MeanCosSim 2.7x lower. Provides additional absorption metrics (mean absorption fraction, full-absorption score). Slightly higher absorption than Matryoshka.

9. **Li et al. (2025), ATM SAE (arXiv:2510.08855)** -- Adaptive Temporal Masking achieves mean absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114. Best reported absorption scores to date, but evaluated only on Gemma-2-2B layer 12.

10. **Tang et al. (2024), Unified SDL Theory (arXiv:2512.05534)** -- First unified theoretical framework for SDL; provides principled explanations for absorption via piecewise biconvex optimization analysis. Feature anchoring proposed but only validated on synthetic benchmarks. Gives theoretical grounding for why absorption is a fundamental consequence of sparsity optimization under feature hierarchy.

11. **SAELens v6 (Bloom et al., 2024--2025)** -- Standard library for SAE training and evaluation. Supports all major architectures. Deep integration with TransformerLens and Gemma Scope. The practical infrastructure backbone for any training-free analysis.

12. **Gemma Scope (Lieberum et al., 2024, arXiv:2408.05147) + Gemma Scope 2 (2025)** -- 400+ open JumpReLU SAEs on Gemma 2 (2B/9B/27B), all layers, widths 1k--1M. Gemma Scope 2 adds Matryoshka SAEs and transcoders on Gemma 3. The primary resource for training-free absorption evaluation.

### Experimental Landscape

**What has been properly tested:**
- Absorption on the first-letter spelling task across Gemma Scope, Llama 3.2 1B, Qwen2 0.5B (Chanin et al.)
- Architecture comparison on SAEBench absorption metric for 7 architectures at 3 widths and 6 sparsities on Pythia-160M layer 8 and Gemma-2-2B layer 12
- Synthetic ground-truth validation showing SAEs substantially underperform probes (SynthSAEBench)
- Feature hedging as a function of SAE width (Chanin et al., 2025)
- Incorrect L0 causing feature mixing (Chanin & Garriga-Alonso, 2025)

**What is accepted without rigorous evidence:**
- That first-letter absorption rates generalize to semantically richer hierarchies (entity types, knowledge taxonomies, safety features). Chanin et al. explicitly note they "expect absorption to happen any time there's a dense feature that co-occurs with more sparse features" but state "this is something that would need to be validated experimentally."
- That absorption is the primary cause of SAE probe underperformance on safety tasks (DeepMind 2025 blog post asserts this without published quantification).
- That L0 and absorption are independent confounds (no controlled disentanglement study exists).
- That architecture-level improvements (Matryoshka, OrtSAE, ATM) compose or generalize across models/layers beyond the specific evaluation settings.

**Methodological gaps:**
- The absorption metric requires known probe directions, limiting it to researcher-selected features. No unsupervised absorption detection exists.
- SAEBench absorption metric evaluated only on 2 models (Pythia-160M, Gemma-2-2B) and specific layers.
- No study controls for L0 correctness when measuring absorption (most SAEs have too-low L0 per Chanin & Garriga-Alonso).
- Cross-domain absorption claims are purely speculative; no experimental validation exists beyond first-letter task.
- ATM's absorption scores (0.0068) were not evaluated on SAEBench, making comparison with other architectures indirect.

---

## Phase 2: Initial Candidates

### Candidate A: Taxonomy of Absorption -- Cross-Domain Absorption Characterization with Controlled Feature Hierarchies

- **Hypothesis**: Feature absorption occurs at comparable or higher rates in semantically rich feature hierarchies (entity-type, country-city, part-of-speech) compared to the first-letter spelling task, and the absorption rate is predictable from measurable hierarchy properties (co-occurrence frequency ratio, hierarchy depth, feature specificity). Specifically: absorption rate positively correlates with the frequency ratio child/parent (Spearman rho > 0.3, p < 0.05) across at least 3 distinct hierarchy types.

- **Falsification criterion**: If absorption rate on all non-spelling hierarchies is below 5% (the noise floor of the Chanin et al. metric) when measured on Gemma-2-2B layer 12 with Gemma Scope 16k SAEs, the hypothesis is falsified. If no hierarchy property correlates with absorption rate (all |rho| < 0.15), the predictability claim is falsified.

- **Evaluation protocol**: 
  - Define 5 hierarchical probe tasks with known ground-truth: (1) first-letter spelling (baseline, replicate Chanin et al.), (2) entity type (PERSON/ORG/LOC from NER datasets), (3) country-city (geographic containment), (4) part-of-speech (noun/verb/adjective), (5) taxonomic IS-A (animal > mammal > dog).
  - For each hierarchy: train LR probes for parent and child features on Gemma-2-2B activations. Compute absorption rate using the Chanin et al. / SAEBench metric pipeline.
  - Evaluate on Gemma Scope SAEs at widths 16k and 65k, layers 6, 12, 18, 24.
  - Compute hierarchy properties: co-occurrence frequency ratio, child feature frequency, parent feature frequency, number of child features per parent.
  - Statistical test: Spearman rank correlation between hierarchy properties and absorption rate, with bootstrap 95% CI (1000 resamples). Report per-hierarchy and aggregate.

- **Ablation plan**: 
  - Ablation 1: Hold hierarchy type constant, vary SAE width (4k, 16k, 65k) to test if absorption-width relationship generalizes beyond spelling.
  - Ablation 2: Hold SAE constant, vary layer to test layer-wise absorption pattern per hierarchy.
  - Ablation 3: Within each hierarchy, measure absorption for high-frequency vs. low-frequency child features to test frequency dependence.
  - Ablation 4: Compare absorption on same hierarchy across models (Gemma-2-2B, Pythia-160M) to test model dependence.

- **Confounders**: 
  - Probe quality: If the LR probe for the parent feature is weak, false negatives may not be true absorption. Control: report probe accuracy and only include hierarchies with probe F1 > 0.85.
  - L0 confound: SAEs with incorrect L0 may show feature hedging mistaken for absorption. Control: use Chanin & Garriga-Alonso's proxy to estimate correct L0 range; report results at both default and "correct" L0 SAEs.
  - Feature representation: Some hierarchies may not have clean linear representations in the model. Control: verify linear probe accuracy before measuring absorption.

- **Pilot design**: Replicate Chanin et al. first-letter absorption measurement on Gemma-2-2B layer 12, 16k SAE using sae-spelling code. Then extend to entity-type (PERSON vs. specific person names) on same SAE. Target: 10 minutes for first-letter replication, 15 minutes for entity-type extension.

### Candidate B: Disentangling the Trinity -- Absorption, Hedging, and L0 Confounds Under Controlled Conditions

- **Hypothesis**: Observed feature failure rates in SAEs are a mixture of three distinct mechanisms -- true hierarchy-driven absorption, width-driven hedging, and L0-induced feature mixing -- that can be quantitatively separated. Specifically: at the "correct" L0 (estimated via Chanin & Garriga-Alonso's proxy), at least 30% of apparent absorption in standard SAEs is actually attributable to incorrect L0 or hedging, not true hierarchical absorption.

- **Falsification criterion**: If correcting L0 to the estimated "correct" value changes absorption rate by less than 5% (absolute) across all tested SAEs, the L0-confound hypothesis is falsified. If varying SAE width from 4k to 65k at fixed L0 changes absorption by less than 5% (absolute), the hedging contribution is falsified.

- **Evaluation protocol**:
  - Use SAEBench's open-source SAEs on Pythia-160M layer 8: 3 widths (4k, 16k, 65k) x 6 sparsity levels x 7 architectures.
  - For each SAE: compute (a) standard absorption rate (Chanin et al. metric on first-letter task), (b) estimated L0 correctness (Chanin & Garriga-Alonso proxy), (c) feature hedging indicators (Chanin et al. 2025 hedging metric).
  - Decompose feature failure events into: (i) true absorption (absorbing latent has high cosine similarity with probe AND L0 is at correct value AND width is sufficient), (ii) hedging (width is insufficient for feature count, per hedging theory), (iii) L0 artifact (L0 is below correct value, features are merged rather than absorbed).
  - Fit a multivariate regression: failure_rate ~ absorption_component + hedging_component + L0_artifact_component + interactions.
  - Bootstrap 95% CI on component proportions. Paired t-test or Wilcoxon signed-rank for absorption rate change when L0 is corrected.

- **Falsification criterion (strengthened)**: If the three-component decomposition explains less than 60% of variance in feature failure rates (R^2 < 0.6), the decomposition model itself is inadequate.

- **Evaluation protocol details**:
  - Primary metric: SAEBench absorption metric (extended Chanin et al.)
  - Secondary metrics: feature hedging score, L0 correctness proxy, probe F1
  - Models: Pythia-160M (SAEBench SAEs available), Gemma-2-2B (Gemma Scope SAEs)
  - Statistical tests: Wilcoxon signed-rank for paired comparisons (same SAE before/after L0 correction), Spearman correlation for continuous relationships, bootstrap CIs throughout
  - Seeds: 3 random seeds for probe training; SAEs are deterministic (pre-trained)

- **Ablation plan**:
  - Ablation 1: Fix width=65k, sweep L0 from 20 to 200 (6 levels) to isolate L0 effect on absorption.
  - Ablation 2: Fix L0=optimal, sweep width from 4k to 65k (3 levels) to isolate width/hedging effect.
  - Ablation 3: Fix width and L0, compare architectures (Standard, TopK, JumpReLU, BatchTopK, Matryoshka, Gated) to isolate architecture effect.
  - Ablation 4: Within "true absorption" cases, ablate the absorbing latent's probe direction (per Chanin et al.) to confirm causal role.

- **Confounders**:
  - The L0 correctness proxy itself may be noisy. Control: validate proxy on SynthSAEBench where ground truth L0 is known.
  - Hedging and absorption may co-occur in the same latent. Control: analyze at individual feature level, not just aggregate.
  - Architecture differences may confound the decomposition. Control: analyze within-architecture first, then across.

- **Pilot design**: Load 3 SAEBench SAEs (Standard, TopK, Matryoshka) at width=16k on Pythia-160M. Compute absorption rate and L0 correctness proxy. Check if absorption rate differs significantly between SAEs with "correct" vs "incorrect" L0. Target: 15 minutes.

### Candidate C: The Absorption-Reconstruction Impossibility Frontier -- Quantifying the Fundamental Tradeoff

- **Hypothesis**: There exists a quantifiable, measurable tradeoff frontier between absorption rate and reconstruction quality (FVU at fixed L0), and this frontier shifts predictably with SAE architecture. Specifically: for any given model and layer, plotting (absorption_rate, FVU@L0=50) across all available SAEs will reveal a Pareto frontier where improvements in one dimension come at measurable cost in the other. Matryoshka SAEs will occupy a strictly better Pareto position than Standard SAEs on this frontier (lower absorption at same FVU, OR lower FVU at same absorption).

- **Falsification criterion**: If no consistent negative correlation exists between absorption rate and reconstruction quality across SAEs (Spearman rho > -0.1 or p > 0.05), the tradeoff hypothesis is falsified. If Matryoshka SAEs do not Pareto-dominate Standard SAEs in at least 2 out of 3 tested layers, the architecture claim is falsified.

- **Evaluation protocol**:
  - Collect all SAEBench SAEs on Gemma-2-2B: 7 architectures x 3 widths x 6 sparsities = ~126 SAEs.
  - For each SAE: compute (a) absorption rate (SAEBench metric), (b) FVU at native L0, (c) CE loss recovered, (d) dead feature fraction.
  - Plot 2D Pareto frontiers: absorption_rate vs FVU (iso-L0 curves at L0 = 25, 50, 100, 200).
  - Fit parametric model: absorption_rate = f(L0, width, architecture, FVU) to predict the tradeoff surface.
  - Per-architecture Pareto dominance test: for each pair of architectures, count how often one Pareto-dominates the other in the absorption-FVU plane.
  - 3 random seeds for any stochastic components (probe training).

- **Ablation plan**:
  - Ablation 1: Hold architecture=Standard, sweep L0 and width to map the "vanilla" tradeoff surface.
  - Ablation 2: Overlay Matryoshka, OrtSAE, and TopK onto the same surface.
  - Ablation 3: Test if the tradeoff surface shifts across layers (layers 6, 12, 18).
  - Ablation 4: Test if the tradeoff surface shifts across models (Gemma-2-2B vs Pythia-160M).

- **Confounders**:
  - SAEs trained for different numbers of steps may not be fairly comparable. Control: use SAEBench SAEs which control training compute.
  - L0 is itself confounded with absorption (Chanin et al. note absorption reduces L0). Control: report iso-L0 curves AND raw scatterplots with L0 as color dimension.
  - Architecture differences in L0 measurement (e.g., TopK has exact L0 while Standard has approximate). Control: report both nominal and measured L0.

- **Pilot design**: Load 6 SAEBench SAEs on Gemma-2-2B layer 12 (Standard and Matryoshka at 3 sparsities each). Compute absorption rate and FVU. Plot the 6 points. Check if a negative correlation is visible. Target: 15 minutes.

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Domain Absorption Characterization)

- **Confound attack**: The biggest confound is probe quality. For the first-letter task, linear probes are near-perfect because the model clearly represents first-letter information linearly. For richer hierarchies (entity type, country-city), the model may represent these less linearly or at different layers. If probe F1 is low (say, 0.70), false negatives from the probe itself will be conflated with absorption. I searched for papers on linear probing quality for entity types and knowledge hierarchies: Nanda et al.'s work on factual recall probing shows entity-type probes achieve F1 > 0.90 on GPT-2 and Gemma at middle layers, but country-city containment probes may be weaker. **Mitigation**: gate all analyses on probe F1 > 0.85; report probe quality alongside absorption rate; explicitly quantify the probe false-negative baseline.

- **Statistical attack**: With 5 hierarchy types and potentially 26 letters per hierarchy (or fewer entity types), the sample size per hierarchy may be small. Chanin et al. used 26 letters with hundreds of tokens each, giving robust statistics. A country-city hierarchy might have only 20-30 well-represented cities per country, yielding wider confidence intervals. Effect size for absorption rate differences across hierarchies may be 0.05-0.15 (absolute), requiring N >> 100 data points per hierarchy for adequate power. **Mitigation**: pre-compute expected power using Chanin et al.'s absorption rate variance as baseline; if power < 0.80, increase dataset size or merge similar hierarchies.

- **Benchmark attack**: The first-letter task is the only validated absorption benchmark. Extending to new hierarchies introduces a construct validity question: is the adapted metric still measuring "absorption" in the same sense? If the absorbing latent for entity types has a different causal structure than for spelling, the metric may not transfer. **Mitigation**: for each new hierarchy, validate the metric by checking that (a) absorbing latents exist (high cosine similarity with probe), (b) they have causal effect (ablation changes model output), and (c) they correspond to more-specific features (not noise).

- **Ablation completeness attack**: Ablation 3 (high-freq vs low-freq children) is informative, but does not control for the number of children per parent. A parent with 100 children (like "starts with S" for all S-tokens) may absorb differently than a parent with 5 children (like "European country" with 5 common countries). This is a potential moderating variable not explicitly ablated. **Mitigation**: add Ablation 5 (vary number of children per parent within same hierarchy type).

- **Verdict**: **STRONG**. The core design is sound -- extending absorption measurement to new domains is both novel and important. The main risks are probe quality (addressable with gating) and statistical power (addressable with larger datasets). The fundamental question ("does absorption generalize beyond spelling?") is worth answering regardless of outcome.

### Against Candidate B (Disentangling Absorption/Hedging/L0)

- **Confound attack**: The decomposition assumes absorption, hedging, and L0 artifacts are separable, but they may interact non-linearly. An SAE with incorrect L0 AND narrow width may exhibit a qualitatively different failure mode than either alone. The multivariate regression may not capture these interactions. More fundamentally, the L0 correctness proxy (from Chanin & Garriga-Alonso) is itself based on a toy model and may not transfer to real LLM SAEs. **Mitigation**: include interaction terms in the regression; validate L0 proxy on SynthSAEBench where ground truth is known.

- **Statistical attack**: With ~126 SAEs but 7 architectures, within-architecture sample sizes are ~18 (3 widths x 6 sparsities). Fitting a multivariate regression with 3+ predictors plus interactions on N=18 risks overfitting. **Mitigation**: use leave-one-out cross-validation; report R^2 on held-out data; consider simpler models (bivariate) before full multivariate.

- **Benchmark attack**: The decomposition is defined operationally (using the specific metrics from 3 papers), but these metrics were developed independently and may not be commensurable. Hedging score from Chanin et al. (2025) was validated on different SAEs than the absorption metric from Chanin et al. (2024). **Mitigation**: recompute all metrics on the same SAE set using the same pipeline.

- **Ablation completeness attack**: The ablation plan varies width, L0, and architecture independently, but does not include a "clean" synthetic baseline where true absorption, hedging, and L0 are known. Without ground truth, the decomposition is always relative to the metrics used. **Mitigation**: add a SynthSAEBench validation step where all three failure modes can be independently controlled.

- **Verdict**: **MODERATE**. Conceptually very important -- the field needs this decomposition -- but the execution is technically challenging. The L0 correctness proxy is the weakest link. Without a validated proxy, the decomposition may be circular. The SynthSAEBench validation step (added as mitigation) is essential. If the L0 proxy is validated, this becomes a strong candidate.

### Against Candidate C (Absorption-Reconstruction Tradeoff Frontier)

- **Confound attack**: The tradeoff between absorption and reconstruction is expected from theory (Chanin et al. note absorption saves +1 L0 per parent-child pair). A negative correlation between absorption rate and FVU is almost tautological if L0 is not carefully controlled: SAEs that "absorb" more have lower L0 (better sparsity) at the same FVU, so the correlation follows mechanically. To be non-trivial, the analysis must hold L0 constant (iso-L0 curves). But holding L0 constant across architectures is difficult because TopK has exact L0 while others have approximate L0. **Mitigation**: use only TopK-family SAEs for the cleanest L0 control; report results with and without L0 matching.

- **Statistical attack**: The Pareto frontier analysis is descriptive, not inferential. Reporting that "Matryoshka dominates Standard" without a statistical test for dominance is insufficient. With only ~18 SAEs per architecture, bootstrap CIs on Pareto frontier position will be wide. **Mitigation**: use the hypervolume indicator from multi-objective optimization as a quantitative dominance metric with bootstrap CIs.

- **Benchmark attack**: The SAEBench absorption metric is restricted to the first-letter spelling task. If this task is not representative (as Candidate A tests), the entire tradeoff surface may be task-specific. **Mitigation**: report tradeoff using multiple tasks if Candidate A validates new hierarchies.

- **Ablation completeness attack**: The plan does not ablate training compute. SAEs trained longer tend to have worse absorption (SAEBench finding: JumpReLU worst, trained longest). If training duration varies across architecture, the tradeoff surface reflects training choices, not inherent architectural properties. **Mitigation**: normalize by training compute or restrict to SAEs with matched training budgets.

- **Verdict**: **MODERATE**. The tradeoff is important to quantify, but risks being confirmatory (we already know it exists from theory) rather than falsifiable. The novelty is in precise quantification and architecture comparison, which is useful but less likely to yield a surprising finding. This is best as a secondary analysis within a broader paper, not the primary contribution.

---

## Phase 4: Refinement

### Dropped: Candidate C (Absorption-Reconstruction Tradeoff Frontier)

Candidate C is dropped as a standalone proposal because the tradeoff is already theoretically predicted and the analysis risks being purely confirmatory. However, the tradeoff quantification will be incorporated as a secondary analysis within the front-runner proposal.

### Strengthened: Candidate A (Cross-Domain Absorption Characterization)

Refinements based on self-critique:

1. **Added probe quality gate**: All analyses gated on probe F1 > 0.85. Probe quality reported alongside absorption rate in all tables.

2. **Added SynthSAEBench validation**: Before extending to real LLM hierarchies, validate the absorption metric pipeline on SynthSAEBench where ground-truth feature hierarchies are known. This confirms the metric works for hierarchies beyond first-letter before claiming results on real data.

3. **Added sample size analysis**: Pre-compute expected statistical power for each hierarchy type. If any hierarchy has insufficient data for reliable absorption measurement, explicitly note this limitation rather than reporting potentially unreliable estimates.

4. **Added children-per-parent ablation** (Ablation 5): Within each hierarchy, vary the number of child features per parent to test if absorption severity depends on hierarchy fan-out.

5. **Added metric validation step**: For each new hierarchy, before reporting absorption rates, verify that (a) absorbing latents exist with high probe cosine similarity, (b) they have causal effect via ablation, (c) they encode more-specific features. This confirms construct validity of the metric in new domains.

6. **Merged elements from Candidate B**: Include L0 correctness as a covariate in all absorption analyses. Report absorption rates at both default and "estimated correct" L0 for each hierarchy. This partially addresses the absorption/hedging/L0 confounding without requiring the full decomposition model.

7. **Added absorption-reconstruction tradeoff from Candidate C**: As a secondary analysis, plot the tradeoff frontier per hierarchy type to test if the absorption-FVU relationship varies across domains.

### Selected Front-Runner: Candidate A (refined)

**Title**: "Beyond Spelling: Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders"

**Rationale**: This is the most novel and the most likely to produce surprising results. The field currently has exactly one absorption evaluation domain (first-letter spelling). Whether absorption generalizes is the single most important open empirical question for the phenomenon. A negative result (absorption does not generalize) would be equally valuable. The training-free constraint is naturally satisfied since we analyze pre-trained SAEs. The methodology extends proven tools (sae-spelling metric, SAELens, Gemma Scope).

---

## Phase 5: Final Proposal

### Title

**Beyond Spelling: Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders**

### Hypothesis

Feature absorption is a general phenomenon that occurs in semantically rich feature hierarchies beyond the first-letter spelling task. Specifically:

H1 (Generality): Absorption rate exceeds 10% on at least 3 out of 4 non-spelling hierarchies (entity-type, country-city, part-of-speech, taxonomic IS-A) when measured on Gemma-2-2B layer 12 with Gemma Scope 16k SAE.

H2 (Predictability): Absorption rate correlates positively with the child-to-parent co-occurrence frequency ratio (Spearman rho > 0.3, p < 0.05 with bootstrap CI) across all hierarchy types pooled.

H3 (Architecture generality): Architecture rankings on absorption (Matryoshka < OrtSAE < Standard < TopK < JumpReLU, per SAEBench) are preserved on non-spelling hierarchies (Kendall's tau > 0.6 with the SAEBench ranking).

### Falsification Criterion

- H1 is falsified if absorption rate is below 5% on all non-spelling hierarchies (this threshold corresponds to the noise floor of the Chanin et al. metric, established via their toy model analysis).
- H2 is falsified if Spearman |rho| < 0.15 between co-occurrence ratio and absorption rate across all hierarchies pooled.
- H3 is falsified if architecture rankings on non-spelling hierarchies have Kendall's tau < 0.2 with SAEBench rankings.

All thresholds are specified before seeing results.

### Method

**Training-free analysis of pre-trained SAEs.** No new SAE training. The approach adapts the Chanin et al. absorption metric to new feature hierarchies and applies it systematically across domains, SAE configurations, models, and layers.

**Key methodological innovation**: Define a general-purpose "hierarchical absorption probe" framework where:
1. A parent feature P and set of child features {C_1, ..., C_n} are specified such that C_i implies P (every token where C_i is active, P should also be active).
2. LR probes are trained for P and each C_i.
3. The absorption metric is computed by identifying tokens where P's probe fires but no SAE latent corresponding to P activates, then checking if a C_i latent has "absorbed" P's direction via cosine similarity analysis.

### Evaluation Protocol

**Primary benchmarks:**
- First-letter spelling task (replication of Chanin et al., baseline)
- Entity-type hierarchy (PERSON/ORG/LOC -- extracted from standard NER annotations)
- Country-city containment (city token implies country membership -- from Wikidata/Wikipedia)
- Part-of-speech hierarchy (noun/verb/adjective token classes -- from POS annotations)
- Taxonomic IS-A (animal > mammal > dog -- from WordNet hierarchy, filtered to well-represented tokens)

**Metrics:**
- Absorption rate (Chanin et al. / SAEBench variant, adapted per hierarchy)
- Probe F1 (gate: only include hierarchies with F1 > 0.85)
- Mean cosine similarity between absorbing latent and parent probe direction
- Number of absorbed features per hierarchy

**Statistical test plan:**
- Spearman rank correlation with bootstrap 95% CI (1000 resamples) for H2
- Kendall's tau with bootstrap 95% CI for H3
- McNemar's test for within-hierarchy absorption rate comparison (spelling vs. non-spelling)
- Effect sizes reported as Cohen's d or rank-biserial correlation alongside p-values
- Bonferroni correction for multiple hierarchy comparisons (5 hierarchies)

**Random seeds:** Minimum 3 seeds for all probe training. SAEs are deterministic (pre-trained).

### Ablation Schedule

| # | Component Ablated | What It Tests | Expected Outcome |
|---|---|---|---|
| A1 | Hierarchy type (5 types) | Does absorption generalize across domains? | Absorption present in most types; rate varies with hierarchy properties |
| A2 | SAE width (4k, 16k, 65k) | Does absorption-width relationship generalize? | Wider SAEs show higher absorption across all hierarchies (per Chanin et al.) |
| A3 | Model layer (6, 12, 18, 24) | Layer-wise absorption pattern | No clear layer-wise pattern (per Chanin et al.) or potential layer-specificity for knowledge hierarchies |
| A4 | High-freq vs low-freq children | Does child frequency drive absorption? | High-frequency children more likely to be absorbed |
| A5 | Children per parent (fan-out) | Does hierarchy fan-out affect absorption? | Higher fan-out increases absorption of parent |
| A6 | Model (Gemma-2-2B, Pythia-160M) | Cross-model generality | Absorption present in both models; rates may differ |
| A7 | L0 correctness | Is absorption confounded by incorrect L0? | Correcting L0 reduces apparent absorption by 10-30% |

### Control Experiments

1. **Probe quality control**: For each hierarchy, report LR probe accuracy with 5-fold cross-validation. Only include hierarchies where probe F1 > 0.85. Report how many hierarchies fail this gate and why.

2. **Random baseline control**: For each hierarchy, create a "shuffled" version where parent-child relationships are randomized. Measure "absorption rate" on shuffled data. Expect near-zero absorption, confirming the metric detects real hierarchical structure rather than noise.

3. **Non-hierarchical feature control**: Select features that co-occur but have no implication relationship (e.g., "contains digit" and "all-caps"). Measure "absorption rate." Expect near-zero, confirming absorption requires hierarchy, not mere co-occurrence.

4. **SynthSAEBench validation**: Run the full absorption metric pipeline on SynthSAEBench synthetic features with known hierarchy. Confirm that measured absorption rate matches ground-truth absorption in the synthetic setting before trusting real-LLM measurements.

5. **L0 correction control**: For each SAE, compute absorption rate at both default L0 and estimated "correct" L0 (per Chanin & Garriga-Alonso proxy). Report the difference as a measure of L0 confounding.

### Pilot Design

**Phase 1 (5 min)**: Load Gemma-2-2B and Gemma Scope 16k layer 12 SAE via SAELens. Compute activations on 1000 tokens. Verify encode/decode works and L0 is in expected range.

**Phase 2 (10 min)**: Replicate Chanin et al. first-letter absorption metric on 3 letters (S, A, T) using sae-spelling code adapted to SAELens. Verify absorption rate is in the 15-35% range reported in the paper.

**Phase 3 (15 min)**: Train LR probe for entity-type (PERSON vs. not-PERSON) on Gemma-2-2B layer 12 activations. Check probe F1. If F1 > 0.85, run the adapted absorption metric on 100 person-name tokens. Report preliminary absorption rate.

**Success criterion for pilot**: First-letter replication within 5% of Chanin et al. reported values. Entity-type probe F1 > 0.85. Preliminary entity-type absorption rate computable without errors.

### Resource Estimate

- **GPU**: Single GPU with >= 24GB VRAM (for Gemma-2-2B + SAE inference). No training required.
- **Models**: Gemma 2 2B (pre-trained), Gemma Scope SAEs (pre-trained, downloaded from HuggingFace), Pythia-160M (pre-trained) + SAEBench SAEs.
- **Time per task**:
  - Pilot: ~30 minutes
  - Full first-letter replication: ~45 minutes
  - Each new hierarchy probe training + absorption measurement: ~30-45 minutes
  - Full ablation sweep (7 ablations x 5 hierarchies): ~8-12 hours total, parallelizable across GPU time
  - Individual tasks within 1-hour budget

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Probe quality too low for some hierarchies | Medium | Medium | Gate on F1 > 0.85; report failed hierarchies as null result |
| Absorption metric does not transfer to non-spelling tasks | High | Low-Medium | SynthSAEBench validation first; construct validity checks |
| L0 confound dominates "true" absorption signal | Medium | Medium | Report with/without L0 correction; include as covariate |
| Statistical power insufficient for rare hierarchies | Medium | Medium | Pre-compute power; merge similar hierarchies if needed |
| Gemma Scope SAEs have architecture-specific artifacts | Low | Low | Cross-validate on SAEBench SAEs and Pythia-160M |

### Novelty Claim

**The primary empirical contribution**: The first systematic measurement of feature absorption beyond the first-letter spelling task. This answers the most important open empirical question about absorption -- whether it is a general phenomenon affecting all hierarchically structured features in LLMs or a narrow artifact of the specific task used to discover it.

**Secondary contributions**:
1. A reusable "hierarchical absorption probe" framework that the community can apply to arbitrary feature hierarchies.
2. The first quantification of absorption as a function of measurable hierarchy properties (co-occurrence ratio, fan-out, child frequency).
3. A controlled assessment of L0 confounding in absorption measurements.
4. Cross-model generality evidence for absorption (or lack thereof).

This work fills Gap 2 (absorption only studied on first-letter task), partially addresses Gap 6 (absorption in hierarchically rich domains), and provides controlled data relevant to Gap 9 (L0 confounding of absorption). It does NOT require new SAE training, aligning with the project's training-free constraint.

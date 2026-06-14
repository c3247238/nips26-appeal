# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **[SAELens](https://github.com/jbloomAus/SAELens)** (MIT License) -- Mature library for loading and analyzing pretrained SAEs. One-line loading of GemmaScope, GPT-2, and LlamaScope SAEs. Essential infrastructure. Already used in this project.

2. **[SAEBench](https://github.com/adamkarvonen/SAEBench)** (MIT License) -- Comprehensive benchmark with 8 metrics including absorption. Standardized evaluation. Pip-installable. Already used in this project.

3. **[sae-spelling](https://github.com/lasr-spelling/sae-spelling)** (MIT License) -- Official code for "A is for Absorption" paper. Contains `FeatureAbsorptionCalculator`, probing utilities, and first-letter task implementation. Directly reusable.

4. **[lca-pytorch](https://github.com/lanl/lca-pytorch)** (LANL) -- PyTorch implementation of Rozell et al.'s Locally Competitive Algorithm. Maps directly to the LCA mathematical framework. Installable via `pip install git+https://github.com/lanl/lca-pytorch.git`.

5. **[TransformerLens](https://github.com/neelnanda-io/TransformerLens)** (GPL-3.0) -- Hook-based model introspection. Used for activation extraction and steering interventions. Already integrated with SAELens.

6. **"A is for Absorption"** (Chanin et al., arXiv:2409.14507) -- Foundational work defining absorption, detection metric, validation across hundreds of SAEs. No general solution proposed.

7. **"Feature Hedging"** (Chanin et al., arXiv:2505.11756) -- Identifies hedging as distinct from absorption. Shows Matryoshka exacerbates hedging. Proposes balanced Matryoshka.

8. **"OrtSAE"** (Korznikov et al., arXiv:2509.22033) -- Reduces absorption by 65% via decoder orthogonality. Slightly lower explained variance. Trade-off exists.

9. **"Matryoshka SAE"** (Bussmann et al., arXiv:2503.17547) -- Nested architecture reducing absorption from 0.49 to 0.05. Introduces hedging trade-off.

10. **"Sanity Checks for SAEs"** (arXiv:2602.14111, 2026) -- Frozen/random SAE baselines match trained SAEs on key metrics. Raises fundamental questions about SAE validity.

11. **"Does Higher Interpretability Imply Better Utility?"** (Wang et al., ICLR 2026) -- Weak correlation (tau_b ~ 0.3) between interpretability and steering utility. Absorption reduction must be validated on downstream tasks.

12. **"Train Sparse Autoencoders Efficiently by Utilizing Features Correlation"** (arXiv:2505.22255) -- KronSAE uses Kronecker product structure to model feature correlations. 54.7% fewer parameters.

### Landscape Summary

The SAE field is crowded with architectural innovations (TopK, JumpReLU, Gated, Matryoshka, OrtSAE, Balance Matryoshka, ATM, H-SAE, CB-SAE) but all share a common pattern: they require retraining. No existing work provides a training-free diagnostic or repair for absorption. The "Sanity Checks" paper and Wang et al.'s utility disconnect create a credibility crisis that any new absorption work must address explicitly.

The key practical gap: practitioners cannot identify at-risk features without running the Chanin metric (which requires prompt generation, activation extraction, and correlation computation). A decoder-correlation-based diagnostic would be instantaneous and scalable.

The LCA connection (W_dec^T W_dec = G_LCA) is mathematically exact but has never been articulated in the SAE literature. Rozell (2008) has ~2000 citations with zero LLM SAE applications.

---

## Phase 2: Initial Candidates

### Candidate A: The Local Inhibition Graph (Front-Runner from Synthesis)

- **Core hypothesis**: Edges in the local inhibition graph (constructed from decoder correlations W_dec^T W_dec) predict known absorption pairs with precision significantly above chance.
- **Implementation sketch**: Start from existing project codebase (SAELens + SAEBench + sae-spelling). Add a `InhibitionGraph` class that computes top-k correlated neighbors per latent from decoder weights. Validate against Chanin absorption pairs on first-letter features.
- **Simplest version**: For GPT-2 Small SAE (24K latents), compute decoder correlation matrix, keep top-20 neighbors per latent, test precision@20 against 26 first-letter absorption pairs. Target: ~15 minutes.
- **Time estimate**: ~2 GPU-hours total (graph construction + validation + precision-recall test + layer analysis + optional rebalancing).
- **Reusable components**: SAELens (SAE loading), sae-spelling (absorption detection), existing project steering/probing code, scipy (correlation stats), numpy (matrix ops).

### Candidate B: Strong Baseline Done Right -- Decoder Correlation as Absorption Predictor

- **Core hypothesis**: The simplest possible predictor of absorption -- decoder weight cosine similarity -- is already competitive with complex architectural solutions, and its limitations explain why absorption persists.
- **Implementation sketch**: Compute the full decoder correlation matrix for GPT-2 Small SAE. For each first-letter feature, rank all other latents by decoder correlation. Measure how well this simple ranking identifies absorbing children compared to the Chanin metric's child detection. The "strong baseline" here is: can we predict absorption better than chance using only decoder geometry?
- **Simplest version**: Compute mean precision@k for k in [5, 10, 20, 50] of decoder-correlation-ranked latents against Chanin-detected children. Compare to random baseline. Target: ~10 minutes.
- **Time estimate**: ~1 GPU-hour total.
- **Reusable components**: Same as Candidate A, but even simpler -- no graph construction, just matrix multiplication and ranking.

### Candidate C: Training-Free Absorption Repair via Activation Rebalancing

- **Core hypothesis**: A single-pass activation rebalancing (boosting parent features based on child activations and decoder correlations) restores parent firing without degrading reconstruction.
- **Implementation sketch**: For input activation a, compute SAE latents z. For each latent i, compute inhibition from neighbors: inh_i = sum_{j in N(i)} G_ij * z_j. Apply boost: z'_i = z_i + alpha * inh_i. Constrain reconstruction error increase < 5%. Test on first-letter features.
- **Simplest version**: Test rebalancing on 5 high-absorption features (H, S, U, V, B) with alpha in [0.5, 1.0, 2.0]. Measure parent firing rate change and reconstruction error. Target: ~20 minutes.
- **Time estimate**: ~1 GPU-hour total (including alpha sweep).
- **Reusable components**: SAELens encode/decode, existing project test prompts, numpy for rebalancing.

---

## Phase 3: Self-Critique

### Against Candidate A (Local Inhibition Graph)

- **Implementation reality check**: The LCA framework (Rozell et al., 2008) is well-established in neuroscience with ~2000 citations. The `lca-pytorch` implementation exists and is maintained by LANL. However, NO prior work connects LCA to SAE absorption. The structural correspondence W_dec^T W_dec = G_LCA is exact but untested empirically.
- **Reproducibility attack**: The graph construction is deterministic (matrix multiply + top-k), so fully reproducible. The validation depends on Chanin absorption pairs, which the project has already computed for 26 first-letter features. Risk: low.
- **Baseline sanity check**: Random baseline precision@20 = 20/24000 = 0.00083. The prediction of precision@20 >= 0.10 represents a 120x enrichment over chance. This is a high bar but not absurd -- decoder correlations do capture semantic similarity in SAEs (shown in multiple papers).
- **Scope attack**: If validated on GPT-2 Small, the framework should generalize to any SAE because it operates on decoder weights alone. However, the first-letter task is narrow. Cross-validation on Gemma-2-2B is planned but not guaranteed to work.
- **Verdict**: STRONG -- exact mathematical correspondence, deterministic computation, training-free, scalable, and falsifiable with existing data.

### Against Candidate B (Decoder Correlation Baseline)

- **Implementation reality check**: Decoder correlation is the simplest possible analysis. Multiple papers (KronSAE, Feature Aligned SAE, dense text embeddings paper) use decoder correlations for feature clustering. This is a well-trodden path.
- **Reproducibility attack**: Trivially reproducible -- one matrix multiplication.
- **Baseline sanity check**: The "baseline" IS the method. The question is whether it beats random chance sufficiently to be useful.
- **Scope attack**: Decoder correlations may predict semantic similarity but not necessarily absorption specifically. Absorption is about encoder suppression, not decoder geometry. The key risk: decoder correlations may be necessary but not sufficient for absorption.
- **Verdict**: MODERATE -- too simple to be the main contribution, but valuable as a baseline/validation for Candidate A.

### Against Candidate C (Activation Rebalancing)

- **Implementation reality check**: No prior work proposes training-free post-hoc repair for absorption. All existing solutions (Matryoshka, OrtSAE, ATM) require retraining. This is genuinely novel but high-risk.
- **Reproducibility attack**: The rebalancing formula is simple, but the alpha parameter requires tuning. Reconstruction error constraint is clear.
- **Baseline sanity check**: The baseline is "no repair" (original SAE performance). Improvement must be measurable on parent firing rate.
- **Scope attack**: Even if repair works for first-letter features, it may fail for semantic hierarchies or deeper layers. The mechanism assumes competitive suppression is the sole cause of absorption, which may not hold.
- **Verdict**: MODERATE -- high novelty but high risk. Best pursued as exploratory (H10) rather than primary contribution.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Decoder Correlation Baseline)** is too simple to stand alone. It is absorbed into Candidate A as the validation baseline.
- **Candidate C (Activation Rebalancing)** is retained as exploratory (H10) but not the primary claim.

### Strengthened Front-Runner (Candidate A)

1. **Simplify further**: The inhibition graph construction is already O(k * d_dict * d_model) -- feasible for million-latent SAEs. No simplification needed.
2. **Confirm code exists**: SAELens (loading), sae-spelling (absorption detection), scipy/numpy (stats) -- all confirmed available.
3. **Pilot design**: The pilot is essentially the primary experiment (E1: graph construction + validation) because it uses existing data. The only new computation is the decoder correlation matrix and precision@k measurement.

### Additional Validation

The project has already collected:
- Absorption rates for 26 first-letter features (A-Z) at layers 0, 4, 8, 10 of GPT-2 Small
- Steering success rates at 6 intervention strengths
- Sparse probing F1, precision, recall
- Random baseline steering data
- Cross-model Pythia-70M data
- EC50 analysis

This existing data can validate H7 (precision-recall asymmetry) and H8 (at-risk prediction) without new experiments. H6 (graph precision) and H9 (layer-dependence) require only the decoder correlation matrix computation.

### Selected Front-Runner

**Candidate A: The Local Inhibition Graph** -- highest success probability, not the flashiest, but the most grounded in engineering reality. It leverages all existing data, requires minimal new computation, and provides a genuinely novel theoretical contribution with clear falsification criteria.

---

## Phase 5: Final Proposal

### Title

**"The Local Inhibition Graph: A Neuroscience-Inspired Training-Free Diagnostic for Feature Absorption in Sparse Autoencoders"**

Alternative: **"Decoder Correlations Reveal Competitive Suppression: A Local Inhibition Graph for SAE Feature Absorption"**

### Hypothesis

For a pretrained SAE, edges in the local inhibition graph (top-k correlated neighbors per latent, constructed from decoder correlations G = W_dec^T W_dec) correspond to known absorption pairs with precision significantly above chance (precision@20 >= 0.10 vs. ~0.00083 expected by chance for 24K latents).

### Motivation

The SAE field has identified feature absorption as a critical failure mode but offers no training-free diagnostic. Practitioners must run the Chanin metric (prompt generation, activation extraction, correlation computation) to identify absorbed features. The Local Inhibition Graph provides an instantaneous, scalable alternative: compute decoder correlations once, then query the graph for at-risk features.

The framework is grounded in the exact structural correspondence between SAE decoder correlations and the inhibition matrix from Rozell et al.'s Locally Competitive Algorithm (LCA): W_dec^T W_dec = G_LCA. This connection has never been articulated in the SAE literature despite ~2000 citations for the original LCA work.

### Method

#### Phase 1: Construct Local Inhibition Graph

For each latent i in the SAE decoder matrix W_dec (shape d_dict x d_model):
1. Compute decoder correlations: G_ij = <W_dec[i], W_dec[j]> for all j != i
2. Keep top-k neighbors per latent (k=20-50) with highest |G_ij|
3. Edge weight = G_ij (signed correlation)
4. Complexity: O(k * d_dict * d_model) -- feasible for 24K-1M latents

#### Phase 2: Validate Graph Against Absorption Pairs

- Use Chanin et al.'s absorption detection on first-letter features (A-Z) as ground truth
- For each absorption pair (parent latent i, absorbing latent j), check if j is in N(i)
- Compute precision@k, recall@k, and Fisher exact test for enrichment
- Compare against random baseline (shuffle latent indices)

#### Phase 3: Test Precision-Recall Asymmetry Explanation

- For each feature, compute total incoming inhibition (sum of edge weights from neighbors)
- Test correlation between total inhibition and recall loss
- Test correlation between total inhibition and precision (predicted: no correlation)

#### Phase 4: Layer-Dependent Analysis

- Construct graphs for layers 0, 4, 8, 10 of GPT-2 Small
- Compare graph statistics (mean edge weight, density, clustering coefficient) across layers
- Test whether layer 8 (where H1b was significant) has stronger inhibition structure

#### Phase 5: Homeostatic Rebalancing (Exploratory)

- For input activation a, compute original latents: z = f(W_enc * a + b_pre)
- Compute inhibition per latent: inh_i = sum_{j in N(i)} G_ij * z_j
- Apply boost: z'_i = z_i + alpha * inh_i
- Clip negative values; constrain reconstruction error increase < 5%
- Test whether rebalancing restores parent feature firing

### Simplest Version

The absolute minimum experiment that tests the core claim:
1. Load GPT-2 Small SAE (24K latents, already in project)
2. Compute decoder correlation matrix: G = W_dec @ W_dec.T
3. For each of 26 first-letter features, get top-20 correlated neighbors
4. Check how many absorbing children (from Chanin metric) are in the top-20
5. Compute precision@20 and compare to random baseline

This requires ~15 minutes of computation and uses only existing data.

### Baselines

1. **Random graph baseline**: Shuffle latent indices; expected precision@20 ~ 0.00083 (20/24000).
2. **Non-absorbed pair control**: Test graph edges for correlated but non-absorbed pairs; predicted lower enrichment.
3. **Identity graph**: Only self-loops; tests whether correlations beyond self-similarity matter.

### Experimental Plan

| Experiment | Model | SAE | Metrics | Time | New Code Needed |
|---|---|---|---|---|---|
| E1: Graph construction + validation | GPT-2 Small | gpt2-small-res-jb (24K) | Precision@k, recall@k, Fisher test | ~15 min | InhibitionGraph class (~50 lines) |
| E2: Precision-recall asymmetry test | GPT-2 Small | Same | Correlation (inhibition vs recall, precision) | ~15 min | Analysis script (~30 lines) |
| E3: Layer-dependent graph structure | GPT-2 Small | Same (layers 0/4/8/10) | Graph stats by layer | ~20 min | Loop over layers (~20 lines) |
| E4: Homeostatic rebalancing | GPT-2 Small | Same | Absorption rate change, reconstruction error | ~30 min | Rebalancing function (~40 lines) |
| E5: Cross-model validation | Gemma-2-2B | GemmaScope 16K | All above metrics | ~30 min | Same code, different SAE ID |

**Total estimated time:** ~2 GPU-hours (well within project constraints).
**Total new code:** ~140 lines of Python.

### Resource Estimate

| Item | Estimate |
|------|----------|
| GPU | Single 24GB GPU (RTX 3090/4090 or A10) |
| Graph construction | ~15 min per SAE |
| Validation experiments | ~30 min per SAE |
| Cross-model (Gemma) | ~30 min (if accessible) |
| Total GPU time | ~2 hours |
| Wall-clock | 1 day |
| Model sizes | GPT-2 Small (primary), Gemma-2-2B (validation) |
| Storage | <10GB |
| New code | ~140 lines |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| Graph edges don't correspond to absorption pairs | Medium | High | Structural correspondence is mathematically exact. If edges don't match, this itself is a finding about decoder correlation limitations. Fallback: diagnostic-only claims. |
| Homeostatic rebalancing degrades reconstruction | Medium | Medium | Alpha is tunable; sweep to find values that improve absorption without degrading reconstruction. Fallback: report Pareto frontier. |
| Repair doesn't improve steering/probing | Medium | Medium | This strengthens the "absorption is benign" claim. The diagnostic contribution stands independently. |
| Local graph misses long-range absorption | Medium | Medium | Test multiple k values (10, 20, 50, 100). Fallback: hierarchical clustering. |
| Gemma-2-2B access issues | High | Medium | Primary experiments on GPT-2 Small; Gemma as validation only. |
| "Sanity Checks" challenge undermines findings | Medium | High | Include random SAE baseline comparison in analysis. If graph structure differs between trained and random SAEs, findings are not artifacts. |

### Novelty Claim

1. **First LCA-SAE connection**: No prior work connects Rozell et al.'s Locally Competitive Algorithm to Sparse Autoencoder feature absorption. The structural correspondence (W_dec^T W_dec = G_LCA) is exact and has not been articulated.

2. **First local inhibition graph for SAE diagnostics**: No existing paper constructs a graph from decoder correlations to diagnose absorption.

3. **First mechanistic explanation for precision-recall asymmetry**: Competitive suppression explains why absorption affects recall but not precision -- a finding from the project's data that currently lacks theoretical grounding.

4. **First training-free post-hoc repair**: All existing solutions (Matryoshka, OrtSAE, ATM) require retraining. Homeostatic rebalancing operates on pretrained SAEs with a single forward-pass correction.

### Engineering Feasibility Assessment

**Green lights:**
- All required libraries are installed and tested (SAELens, SAEBench, sae-spelling, TransformerLens)
- Existing project codebase has absorption detection, steering, and probing infrastructure
- Decoder correlation matrix computation is trivial (one matrix multiplication)
- No SAE training required
- No LLM inference required for graph construction (only for validation)

**Yellow lights:**
- Gemma-2-2B access requires HuggingFace authentication (may fail)
- Homeostatic rebalancing alpha parameter requires tuning
- Need to verify that decoder correlations are computed correctly (W_dec shape may be transposed depending on SAELens version)

**Red lights:**
- None identified. This is the lowest-risk proposal among all candidates.

### Pragmatist's Bottom Line

This proposal wins on engineering grounds:
1. **Minimal new code** (~140 lines vs. thousands for architectural modifications)
2. **No training required** (vs. hours/days for retraining SAEs)
3. **Deterministic computation** (vs. stochastic training runs)
4. **Scalable to any SAE size** (local graph construction is O(k * d_dict))
5. **Builds on existing data** (leverages all experiments from iterations 1-8)
6. **Clear falsification criteria** (precision@20 threshold is unambiguous)
7. **Fallback paths exist** (if graph fails, diagnostic-only claims remain; if repair fails, diagnostic contribution stands)

The primary risk is not engineering but intellectual: will reviewers accept a neuroscience analogy as a theoretical contribution? Mitigation: the correspondence is exact (W_dec^T W_dec = G_LCA), not metaphorical. The LCA framework provides equations, not just intuition.

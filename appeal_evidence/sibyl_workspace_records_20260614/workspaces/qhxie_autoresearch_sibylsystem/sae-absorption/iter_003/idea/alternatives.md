# Backup Ideas for Pivot (Iteration 2)

## Current Status

The front-runner proposal has three primary contributions: (1) ARS Unsupervised Absorption Detector, (2) Amortization Gap Controlled Dictionary Experiment, (3) Cross-Hierarchy Absorption Characterization. The backup ideas below are ordered by activation condition.

---

## Backup A: LCA-SAE — Locally Competitive Encoder for Absorption Reduction

**Status:** Strong backup. Activate if ARS AUROC < 0.60 AND amortization gap experiment shows encoder effects are significant (OMP reduces absorption > 50%).

**Core idea:** Replace the feedforward SAE encoder with an unrolled Locally Competitive Algorithm (LCA), where features compete via lateral inhibition derived from the decoder Gram matrix. If the amortization gap experiment confirms that encoder improvements can eliminate absorption, LCA provides a practical, low-overhead implementation that avoids the complexity of full iterative pursuit.

**Architecture:**
```
a^0 = TopK(W_enc @ x + b_enc)          # standard encoder as initialization
G = max(0, W_dec.T @ W_dec - tau * I)  # sparse lateral inhibition matrix (decoder Gram)
For t in 1..T:
    a^(t+1) = S_lambda(W_enc @ x - G @ a^t + a^t)  # soft-threshold with lateral inhibition
x_hat = W_dec @ a^T + b_dec            # standard decoder, unchanged
```

**Key differentiator from MP-SAE (Costa et al., arXiv:2506.03093):** MP-SAE uses sequential greedy selection; LCA uses parallel lateral competition. LCA allows both parent and child features to coexist at graded activation levels simultaneously; MP-SAE's greedy approach may still block a lower-activation parent after a high-activation child is selected first.

**Pilot (10 min, CPU):**
- Chanin et al. 2-feature hierarchical toy model
- Standard feedforward SAE: parent absorbed → child fires alone
- LCA encoder (5 iterations): test whether parent and child can coexist
- Go criterion: parent activation > 0.1 in LCA when child is active

**Compute cost:** 3-4 GPU-hours (threshold tuning + SAEBench evaluation on Gemma Scope 16k)

**Activation condition:** Amortization gap experiment shows OMP reduces absorption >= 30% (encoder matters) AND ARS AUROC < 0.60 (detection contribution fails). In this case, LCA provides a practical encoder improvement motivated by the amortization gap result.

---

## Backup B: Successive Refinement Test — Does Increasing Width Recover Absorbed Features?

**Status:** Quick supplementary experiment. Run regardless of other results as a 30-minute add-on.

**Core idea:** The H6 falsification from iter_001 showed wider SAEs absorb MORE features. But does a wider SAE eventually recover the features that were absorbed in a narrower one? If not, "just use a wider SAE" is not a valid prescription. This tests whether SAE representations are "successively refinable" in the sense of Equitz and Cover (1991).

**Key experiment (< 1 GPU-hour):**
For all Gemma Scope L12 SAEs (widths: 1k, 4k, 16k, 65k, 131k, if available):
1. Take all letters absorbed in the 16k SAE (from Chanin et al. metric)
2. Check whether a matching unabsorbed feature exists in the 65k SAE (cosine similarity > 0.80 with the probe direction)
3. Report fraction of absorbed-in-16k features that are recovered in 65k, 131k SAEs
4. If < 50% recovered in any wider SAE: NOT successively refinable

**Why this matters:** If wider SAEs absorb more but also recover previously-absorbed features (of a different type), the net effect is a change in which features are absorbed, not fewer total absorbed features. If wider SAEs absorb more AND fail to recover previously-absorbed features, the correct prescription is hierarchically-aware training (Matryoshka, KronSAE) not just wider dictionaries.

**Novelty:** 8/10. The successive refinement framing (Equitz-Cover) has never been applied to SAE absorption. The test is simple but the theoretical framing is novel.

**Compute cost:** < 1 GPU-hour (weight analysis only, no activation data needed).

---

## Backup C: Recoverability Analysis — Is Absorption Lossless or Lossy?

**Status:** Activate after amortization gap experiment if late absorption is a significant minority (> 20%).

**Core idea:** For late-absorbed latents (decoder-present, ~13% from iter_001 taxonomy), the absorbed parent feature has a decoder direction. If the child latent's activations contain enough information to reconstruct the parent feature, absorption is informationally lossless (the information is present in the SAE, just misaddressed). If not, absorption represents genuine information loss.

**Key experiment (1-2 GPU-hours):**
For each late-absorbed parent feature identified by the Chanin metric at L12-65k:
1. Extract child feature activations when parent should fire but is absorbed
2. Train a 1-layer linear probe: parent_concept ~ child_feature_activations
3. Compute recovery accuracy vs. non-absorbed case (parent probe accuracy when not absorbed)
4. Compute information ratio: I(parent; child_activations) / I(parent; parent_activations) using k-NN mutual information estimator

**Interpretation:**
- Recovery accuracy > 0.85 AND information ratio > 0.90: absorption is approximately lossless for this subtype → the safety concern is "information misaddressing" not "information loss" → ITAC-style corrections are the right fix
- Recovery accuracy < 0.70 OR information ratio < 0.70: absorption causes genuine information loss → no post-hoc correction can fully recover the information → retraining with hierarchy-aware objectives is necessary

**Novelty:** 7/10. The lossless vs. lossy absorption distinction is new. The recoverability ratio metric is novel. Practically important: determines whether inference-time corrections (ITAC, LCA, MP-SAE encoder) can in principle recover absorbed information.

**Compute cost:** 1-2 GPU-hours (forward passes + linear probing, no training on SAE).

---

## Backup D: ARS Scaling Law Analysis

**Status:** Run as supplementary if ARS achieves AUROC > 0.70 in main experiment.

**Core idea:** If ARS is a reliable unsupervised absorption predictor, does the mean ARS score across all latent pairs follow a scaling law as a function of SAE width and L0? This would extend the absorption scaling analysis (H6 falsification from iter_001) with a training-free proxy metric that does not require Chanin metric measurement for each configuration.

**Key experiment (< 1 GPU-hour):**
- Compute mean ARS, 95th-percentile ARS, and fraction of latents with AFS > threshold for each Gemma Scope L12 configuration (widths: 1k, 4k, 16k, 65k)
- Fit log-linear regression: log(mean_ARS) ~ α + β·log(W) + γ·log(L0)
- Compare regression to the directly measured SAEBench absorption scores (pre-computed) — does ARS track SAEBench scores without running the expensive Chanin metric?

**Value:** If ARS statistics are a reliable proxy for SAEBench absorption scores, this enables cheap screening of new SAE architectures without running the full absorption evaluation suite (which takes ~65 min per SAE on a GPU, per SAEBench timing).

---

## Dropped Candidates (Confirmed from Perspective Analysis)

- **Candidate C from Innovator (Label-Implication-Aware Decoder Regularization):** Requires SAE fine-tuning, violating the training-free constraint. Dropped.

- **Candidate C from Theoretical (Basin-of-Attraction Analysis):** Requires SAE retraining (for varied-seed convergence analysis). Technically too demanding for project timeline. The phase transition claim (H4) captures the key theoretical insight without requiring full basin analysis.

- **Candidate B from Pragmatist (Cross-Domain without negative control):** Absorbed into Contribution 3 of the main proposal with proper negative controls added per the empiricist's recommendation.

- **Competitive Exclusion full ecological framework:** The L-V formalism is incorporated as the theoretical motivation for the ARS metric but is not a standalone contribution. The ARS metric is the novel operationalization; the ecological theory provides the derivation but does not require a separate paper.

---

## Pivot Decision Tree

```
CURRENT STATE (Iteration 2 starting point):
- EDA detector: CONFIRMED from iter_001 (AUROC = 0.776 at L12-16k)
- 75% early absorption: CONFIRMED from iter_001
- ARS: UNTESTED (proposed in this iteration)
- Amortization gap: UNTESTED (proposed in this iteration)
- Cross-hierarchy with controls: PARTIALLY FAILED in iter_001 (wrong-model probes)

Primary action: Run ARS pilot + amortization gap pilot (both 15 min) before full commitment

├── ARS pilot passes (AUROC > 0.60 on pilot) AND amortization gap pilot shows effect
│   → PROCEED: Full 3-contribution paper as proposed
│   → Add Backup B (successive refinement) as supplementary
│   → Consider Backup C (recoverability) if late absorption is significant
│
├── ARS pilot fails (AUROC < 0.55) BUT amortization gap is significant
│   → PIVOT: 2-contribution paper (amortization gap + cross-hierarchy)
│   → Backup A (LCA-SAE) if amortization gap shows encoder can reduce absorption
│
├── ARS pilot passes BUT amortization gap shows no effect
│   → PIVOT: 2-contribution paper (ARS detection + cross-hierarchy)
│   → Add EDA + ARS comparison as a detection methodology contribution
│
├── Both pilots fail
│   → ACTIVATE: Cross-hierarchy as primary (most independently motivated)
│     + Backup B (successive refinement, quick) as secondary
│     + EDA from iter_001 as established detection baseline
│
└── All fail + Gemma 2B blocked
    → Fallback: EDA theory (iter_001) + ARS theory + amortization gap theory
      as a theoretical paper about absorption causation
      Target: MI Workshop, not main track
```

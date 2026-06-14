# Testable Hypotheses with Expected Outcomes

## Primary Hypotheses

### H1: TopK Dominance (REVISED from MultiScale Dominance)

**Statement**: Explicit k-sparsity (TopK, k=50) is the dominant driver of absorption reduction, achieving Cohen's d > 2.0 vs. baseline. The effect size dwarfs all other architectural components tested.

**Formal test**: Train SAE variants on SynthSAEBench-16k (5 replicates each). Compute absorption rate for each variant. Compare TopK vs. Baseline using one-way ANOVA with post-hoc Tukey HSD.

**Falsification criterion**: If TopK does not achieve the largest effect size (Cohen's d) among all components, or if d < 1.0, H1 is falsified.

**Expected outcome if true**: TopK absorption rate is significantly lower than Baseline (p < 0.05, d > 2.0). All other components show smaller effects (d < 1.0).
**Expected outcome if false**: Another component (e.g., MultiScale, Orthogonality) shows comparable or larger effect size than TopK.

---

### H2: Sparsity Mediation (NEW)

**Statement**: The absorption rate is primarily determined by L0 sparsity level, not by architectural choice. At matched L0, different architectures show comparable absorption rates.

**Formal test**: Train L1 SAEs with tuned lambda to achieve L0 = 50 (matching TopK) and L0 = 550 (matching Orthogonality). Compare absorption rates at matched L0.

**Falsification criterion**: If L0-matched Baseline shows significantly different absorption than TopK or Orthogonality (p < 0.05, d > 0.5), H2 is falsified.

**Expected outcome if true**: L0-matched Baseline achieves absorption comparable to TopK/Orthogonality (p > 0.05, d < 0.5). The L0-absorption correlation across all variants is r < -0.8.
**Expected outcome if false**: L0-matched Baseline differs significantly from TopK/Orthogonality, indicating architectural effects beyond sparsity.

---

### H3: Orthogonality Null (NEW)

**Statement**: Decoder orthogonality penalties have negligible effect on absorption (Cohen's d < 0.5), contradicting OrtSAE's claim of 65% reduction. Absorption is encoder-driven, not decoder-driven.

**Formal test**: Compare Orthogonality SAE vs. Baseline on absorption rate. Compute Cohen's d.

**Falsification criterion**: If Orthogonality shows d > 0.5 (medium effect) or absorption reduction > 20%, H3 is falsified.

**Expected outcome if true**: Orthogonality d < 0.5, absorption reduction < 10%. Despite excellent reconstruction (MSE ~0), orthogonality does not reduce absorption.
**Expected outcome if false**: Orthogonality shows d > 0.5, confirming OrtSAE's claim.

---

### H4: Synthetic-to-Real Transfer (REVISED)

**Statement**: The L0-absorption relationship observed on SynthSAEBench correlates with real-LLM architecture rankings (Kendall's tau > 0.6).

**Formal test**:
1. Compute L0-absorption curve from SynthSAEBench (Experiment 1)
2. Load pretrained SAEs from SAELens with varying L0 (Gemma Scope series)
3. Measure first-letter absorption via SAEBench
4. Test whether L0 predicts absorption across real SAEs
5. Compute Kendall's tau between predicted and observed absorption

**Falsification criterion**: If Kendall's tau < 0.3 (weak or no correlation), H4 is falsified.

**Expected outcome if true**: tau > 0.6, p < 0.05. Real-LLM absorption is predicted by L0.
**Expected outcome if false**: tau < 0.3. Synthetic and real-LLM behavior diverge; the synthetic-to-real gap is large.

---

## Secondary Hypotheses

### H5: TopK Dose-Response (NEW)

**Statement**: Absorption rate increases monotonically with k in TopK SAEs (lower sparsity -> higher absorption).

**Formal test**: Train TopK SAEs with k in {10, 25, 50, 100, 200, 500}. Measure absorption rate at each k.

**Falsification criterion**: If absorption is not monotonically increasing in k (e.g., U-shaped or flat), H5 is falsified.

**Expected outcome**: Absorption increases monotonically with k: k=10 (lowest) < k=25 < k=50 < k=100 < k=200 < k=500 (highest).

---

### H6: Dead Latent Crisis (NEW)

**Statement**: TopK SAEs exhibit high dead latent rates (>50%), and the absorption reduction is partially artifactual (driven by dictionary shrinkage, not better learning).

**Formal test**: Compute dead latent rate for all variants. Compute absorption rate using only active latents. Compare with full-dictionary absorption.

**Falsification criterion**: If dead latent rate < 30% OR active-latent-only absorption is comparable to full-dictionary absorption, H6 is falsified.

**Expected outcome**: Dead latent rate > 50% for TopK. Active-latent-only absorption is higher than full-dictionary absorption, indicating some artifactual reduction.

---

### H7: Random Control Validation (REVISED)

**Statement**: Random-feature control achieves absorption rate > 0.5, validating that the metric discriminates structure from randomness.

**Formal test**: Train SAE with random decoder on SynthSAEBench-16k. Compute absorption rate.

**Falsification criterion**: If random control achieves absorption < 0.3, the metric may not discriminate structure from randomness.

**Expected outcome**: Random control absorption > 0.5 (pilot: 0.56). Metric is validated.

---

## Exploratory Hypotheses

### H8: Component Interactions

**Statement**: MultiScale and TopK interact synergistically: their combined effect (Full Matryoshka) is greater than the sum of their individual effects.

**Formal test**: Compare observed Full Matryoshka absorption to predicted additive effect: predicted = Baseline - (Baseline - MultiScale) - (Baseline - TopK).

**Expected outcome**: If observed < predicted, synergy exists. If observed ≈ predicted, effects are additive. If observed > predicted, antagonism exists.

---

### H9: Training-Dynamic Emergence

**Statement**: Absorption emerges early in SAE training (within first 20% of steps) and is largely irreversible.

**Formal test**: Train Standard ReLU SAE with checkpointing every 10% of training. Measure absorption rate at each checkpoint.

**Expected outcome**: Absorption rate reaches >80% of final value by 20% of training and does not decrease significantly in later checkpoints.

---

## Hypothesis Status Tracker

| Hypothesis | Original Status | Current Status | Reason for Change |
|-----------|----------------|----------------|-------------------|
| H1 (MultiScale dominance) | Active | **REVISED to TopK dominance** | Data shows TopK d=5.51 >> all others |
| H2 (Component ranking) | Active | **REVISED to Sparsity Mediation** | L0-absorption correlation r~-0.99 |
| H3 (Absorption-hedging trade-off) | Active | **REFUTED** | Hedging ~0.24 constant across variants |
| H4 (Synthetic-to-real transfer) | Active | **REVISED** | Now tests L0-absorption correlation |
| H5 (Feature recovery correlates) | Active | **REFUTED** | MCC ~0.21-0.22 across ALL variants |
| H6 (Reconstruction-Absorption Pareto) | Active | **PARTIALLY SUPPORTED** | At least 2 variants on frontier |
| H7 (Random control) | Active | **INCONCLUSIVE** | Full random control data pending |
| H8 (Training dynamics) | Exploratory | **PENDING** | No checkpointing data available |
| H9 (Component interactions) | Exploratory | **PENDING** | Full Matryoshka data pending |

## New Hypotheses Added

| Hypothesis | Rationale |
|-----------|-----------|
| H3 (Orthogonality null) | Data shows d=0.14, contradicting OrtSAE's 65% claim |
| H5 (TopK dose-response) | Characterize sparsity-absorption relationship |
| H6 (Dead latent crisis) | TopK has 82% dead latents; may be artifactual |

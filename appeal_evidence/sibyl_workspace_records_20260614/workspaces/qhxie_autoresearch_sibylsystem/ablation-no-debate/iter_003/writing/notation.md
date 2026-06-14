# Notation Table: Encoder-Driven Feature Absorption in SAEs

This document establishes all mathematical symbols and notation used throughout the paper. All subsequent section writers must reference this file for consistency.

---

## 1. Model Architecture and SAE Components

### 1.1 Sparse Autoencoder (SAE)
| Symbol | Definition | Dimensionality |
|--------|------------|---------------|
| $\mathcal{A}$ | Sparse Autoencoder | - |
| $\mathbf{x} \in \mathbb{R}^d$ | Input activation vector | $d$ |
| $\mathbf{z} \in \mathbb{R}^n$ | SAE latent representation (sparse code) | $n$ |
| $\mathbf{\hat{x}} \in \mathbb{R}^d$ | Reconstructed input | $d$ |
| $f(\cdot)$ | SAE encoder network | - |
| $g(\cdot)$ | SAE decoder network | - |
| $L_0(\mathbf{z})$ | L0 norm of latent (number of active features) | scalar |

### 1.2 Encoder and Decoder
| Symbol | Definition | Dimensionality |
|--------|------------|---------------|
| $W_e \in \mathbb{R}^{n \times d}$ | Encoder weight matrix | $n \times d$ |
| $b_e \in \mathbb{R}^n$ | Encoder bias vector | $n$ |
| $W_d \in \mathbb{R}^{d \times n}$ | Decoder weight matrix | $d \times n$ |
| $b_d \in \mathbb{R}^d$ | Decoder bias vector | $d$ |

---

## 2. Feature Hierarchy

### 2.1 Hierarchy Structure
| Symbol | Definition | Notes |
|--------|------------|-------|
| $\mathcal{H}$ | Hierarchical feature structure | 3-level synthetic |
| $p$ | Parent feature | Level 0 |
| $c_1, c_2, \ldots, c_k$ | Child features | Level 1 |
| $gc_1, gc_2, \ldots, gc_m$ | Grandchild features | Level 2 |
| $d(p, c_i)$ | Cosine distance between parent and child $i$ | - |

### 2.2 Hierarchy Parameters
| Symbol | Definition | Value Range |
|--------|------------|--------------|
| $\alpha$ | Hierarchy strength (cosine similarity) | $[0.5, 0.95]$ |
| $\epsilon$ | Stochastic noise parameter | $\sim \mathcal{N}(0, 0.05)$ |
| $k$ | Number of child features to ablate | 5 (default) |

---

## 3. Absorption Metrics

### 3.1 Multi-Child Proportional Absorption
| Symbol | Definition | Range |
|--------|------------|-------|
| $A_{multi}(p)$ | Multi-child proportional absorption rate for parent $p$ | $[0, \infty)$ -- can exceed 1.0 when child ablation increases parent activation |
| $a_p^{before}$ | Parent activation before ablation | scalar |
| $a_p^{after}$ | Parent activation after ablating child features | scalar |

**Formula**:
$$A_{multi}(p) = \frac{a_p^{after}}{a_p^{before}}$$

**Note**: Values $> 1.0$ indicate the parent was being suppressed by children; ablating children releases the parent's true activation.

### 3.2 Feature Sensitivity
| Symbol | Definition | Reference |
|--------|------------|-----------|
| $\mathcal{S}(f)$ | Feature sensitivity (steering coefficient variance) | Hu et al., 2025 |
| $\sigma^2_{steer}$ | Steering coefficient variance | - |

---

## 4. Statistical Tests

### 4.1 Hypothesis Testing
| Symbol | Definition | Threshold |
|--------|------------|-----------|
| $H_{Mech}$ | Encoder-driven mechanism hypothesis | B ≈ D (delta < 0.1), C ≈ A (delta < 0.1) |
| $H_{Comp}$ | Hierarchy strength dependence hypothesis | $R^2 > 0.8$ for monotonic fit |
| $H_{Safe}$ | Safety-critical feature absorption hypothesis | Mann-Whitney $p < 0.05$ |
| $H_{Pareto}$ | Pareto frontier existence hypothesis | Detectable non-degenerate frontier |

### 4.2 Test Statistics
| Symbol | Definition | Notes |
|--------|------------|--------|
| $U$ | Mann-Whitney U statistic | Non-parametric |
| $p$ | p-value | - |
| $\rho$ | Spearman correlation coefficient | Rank correlation |
| $R^2$ | Coefficient of determination | Goodness of fit |

---

## 5. Experiment Configuration

### 5.1 SAE Releases
| Symbol | Definition | Notes |
|--------|------------|-------|
| $\mathcal{G}_{2B}$ | Gemma 2B model | Real SAE experiments |
| $\mathcal{G}_{2B-res}$ | Gemma Scope residual SAE | gemma-2b, layer 12, d=16384 |
| $\mathcal{M}_{GPT2}$ | GPT-2 Small model | Synthetic experiments (blocks.8.hook_resid_pre) |

### 5.2 Experiment Parameters
| Symbol | Definition | Values |
|--------|------------|--------|
| $L0_{target}$ | Target L0 sparsity | $\{16, 32, 64, 128\}$ |
| $n_{seeds}$ | Number of random seeds (H_Mech full) | 5 |
| $n_{samples}$ | Number of hierarchy samples per run | 100 |
| $n_{safety}$ | Number of safety features (Gemma pilot) | 5 |
| $n_{non-safety}$ | Number of non-safety features (Gemma pilot) | 5 |

---

## 6. Feature Sets and Distributions

### 6.1 Feature Sets
| Symbol | Definition | Notes |
|--------|------------|-------|
| $\mathcal{F}_{safety}$ | Safety-critical feature set | Neuronpedia-annotated (deception, jailbreak, harm) |
| $\mathcal{F}_{non-safety}$ | Non-safety control feature set | Frequency-matched |
| $\mathcal{F}_{absorbed}$ | Absorbed feature set | $A_{multi} > \tau$ |

### 6.2 Absorption Distribution
| Symbol | Definition | Notes |
|--------|------------|-------|
| $P_A$ | Distribution of absorption rates across features | - |
| $\mu_A$ | Mean absorption rate | - |
| $\sigma_A$ | Standard deviation of absorption rate | - |

---

## 7. 2x2 Factorial Conditions

| Symbol | Encoder | Decoder | Interpretation |
|--------|---------|---------|----------------|
| $C_A$ | Random | Random | Baseline geometry only |
| $C_B$ | Trained | Random | Encoder alignment only |
| $C_C$ | Random | Trained | Decoder geometry only |
| $C_D$ | Trained | Trained | Full training |

---

## 8. Experimental Results (Authoritative)

### 8.1 H_Mech Results (Encoder-Driven Mechanism, FULL EXPERIMENT)
5 seeds × 4 conditions × 100 samples = 20 runs

| Condition | Encoder | Decoder | Absorption Rate (mean) | Std | Min | Max | N |
|-----------|---------|---------|------------------------|-----|-----|-----|---|
| A | Random | Random | 0.184 | 0.323 | 0.0 | 0.826 | 5 |
| B | Trained | Random | 0.055 | 0.038 | 0.008 | 0.106 | 5 |
| C | Random | Trained | 12.28 | 17.13 | 0.0 | 43.84 | 5 |
| D | Trained | Trained | 0.017 | 0.0 | 0.017 | 0.017 | 5 |

**Factorial checks**:
- encoder_driven_check: TRUE (B ≈ D confirmed)
- decoder_irrelevant_check: FALSE (C ≈ A FAILS due to extreme variance)
- b_vs_d_delta: 0.037
- c_vs_a_delta: 12.10
- pass: false

**Cross-model validation** (held_out_validation, GPT-2 Small):
- encoder_driven_check: TRUE
- b_vs_d_delta: 0.0
- c_vs_a_delta: 0.0

**Key finding**: B ≈ D confirmed (encoder sufficient); C ≈ A FAILS (decoder contribution is configuration-dependent, not uniformly zero)

### 8.2 H_Comp Results (Hierarchy Strength Dependence, FULL EXPERIMENT)
3 seeds × 6 levels × 100 samples = 1800 measurements

| Cosine Similarity | Mean Absorption | Std | N |
|-------------------|-----------------|-----|---|
| 0.50 | 0.814 | 0.131 | 300 |
| 0.60 | 0.989 | 0.163 | 300 |
| 0.70 | 0.972 | 0.262 | 300 |
| 0.80 | 0.607 | 0.341 | 300 |
| 0.90 | 1.201 | 0.397 | 300 |
| 0.95 | 0.512 | 0.242 | 300 |

**Regression**: slope = -0.296, intercept = 1.068, R² = 0.04, p = 0.703
**pass_criteria**: R² > 0.8
**full_pass**: false

**Key finding**: No monotonic relationship; R² = 0.04. The relationship between hierarchy strength and absorption is non-monotonic and seed-dependent.

### 8.3 H_Pareto Results (Sensitivity-Absorption Frontier, FULL EXPERIMENT)
3 seeds × 4 L0 levels × 100 samples

| L0 Target | Absorption Mean | Absorption Std | Sensitivity Mean | Sensitivity Std | N |
|-----------|-----------------|---------------|-----------------|-----------------|---|
| 16 | 0.0 | 0.08 | 0.1054 | 0.0008 | 3 |
| 32 | 0.0 | 0.08 | 0.1054 | 0.0008 | 3 |
| 64 | 0.0 | 0.08 | 0.1054 | 0.0008 | 3 |
| 128 | 0.0 | 0.08 | 0.1054 | 0.0008 | 3 |

**Frontier fit**: a = 1.0, b = -0.5 (degenerate -- absorption collapses to zero)
**Key finding**: No Pareto frontier detected; absorption = 0 across all L0 levels. The theoretical prediction of a sensitivity-absorption trade-off is not supported.

### 8.4 H_Safe Results (Safety-Critical Feature Absorption)

**Gemma Scope Pilot** (5 safety + 5 non-safety features, 100 samples each):
| Metric | Value |
|--------|-------|
| Safety absorption (all values) | 0.0 |
| Non-safety absorption (all values) | 0.0 |
| Mann-Whitney U | 0.0 |
| p-value | 1.0 |

**GPT-2 Small Held-Out Validation** (20 safety + 20 non-safety features, 100 samples each):
| Metric | Value |
|--------|-------|
| Safety mean absorption | 233.13 |
| Non-safety mean absorption | 221.70 |
| Mann-Whitney U | 63.0 |
| p-value | 0.345 |

**Key finding**: No significant difference between safety and non-safety features in either Gemma Scope or GPT-2 Small. Safety-critical features do NOT show elevated absorption. This is a null result with positive implications for SAE-based safety analysis.

# Notation Table

This document defines all mathematical symbols and notation used throughout the paper for consistency.

---

## SAE Architecture

| Symbol | Definition | Type/Dimensions |
|--------|------------|-----------------|
| $\mathcal{D}$ | Training dataset (synthetic activations) | Dataset |
| $x \in \mathbb{R}^d$ | Residual stream activation vector | Vector, $d=128$ |
| $W_{enc} \in \mathbb{R}^{d_{sae} \times d}$ | SAE encoder weight matrix | Matrix, $d_{sae}=4096$ |
| $W_{dec} \in \mathbb{R}^{d \times d_{sae}}$ | SAE decoder weight matrix | Matrix, $d_{sae}=4096$ |
| $b_{enc} \in \mathbb{R}^{d_{sae}}$ | SAE encoder bias | Vector |
| $b_{dec} \in \mathbb{R}^d$ | SAE decoder bias | Vector |
| $f \in \mathbb{R}^{d_{sae}}$ | Sparse feature activations (SAE output) | Vector, sparse |
| $\theta$ | All trainable SAE parameters | Parameters |
| $L_0$ | Target number of active features (sparsity target) | Scalar, $\in \{20, 32, 50\}$ |
| $k$ | Top-k activation threshold | Scalar |
| $\sigma$ | Sparsity-inducing activation function (TopK) | Function |

---

## Feature Absorption

| Symbol | Definition | Type/Dimensions |
|--------|------------|-----------------|
| $p$ | Parent feature (ground-truth concept) | Feature index |
| $c_1, c_2, ..., c_k$ | Child features that substitute for parent | Feature indices |
| $g_1, g_2, ..., g_m$ | Grandchild features (lowest-level features) | Feature indices |
| $\alpha_{abs}$ | Multi-child proportional absorption rate | Scalar, $[0, 1]$ |
| $\alpha_{abs} = \frac{|\{x : f_c(x) > 0\} \cap \{x : f_p(x) > 0\}|}{|\{x : f_p(x) > 0\}|}$ | Jaccard overlap formula | Equation |
| $prop_i$ | Proportional contribution of child $i$ to parent reconstruction | Scalar |
| $cos(u, v)$ | Cosine similarity between vectors $u$ and $v$ | Scalar |

---

## Synthetic Hierarchy

| Symbol | Definition | Value |
|--------|------------|-------|
| $H$ | Synthetic hierarchy with known ground truth | Structure |
| $L_H$ | Hierarchy depth | 3 levels |
| $cos(parent, children)$ | Parent-children cosine similarity | $\{0.5, 0.67, 0.8\}$ |
| $cos(g_i, g_j)$ | Grandchildren pairwise orthogonality | $-0.03$ to $0.03$ |
| $N_{hierarchies}$ | Number of hierarchies per seed | 5 |
| $N_{samples}$ | Number of samples per hierarchy | 10,000 |
| $\sigma_{noise}$ | Stochastic noise added to hierarchy generation | 0.1 |

---

## 2x2 Factorial Decomposition (H_Mech)

| Symbol | Definition | Type |
|--------|------------|------|
| $A$ | Condition A: Random encoder + Random decoder | Condition |
| $B$ | Condition B: Trained encoder + Random decoder | Condition |
| $C$ | Condition C: Random encoder + Trained decoder | Condition |
| $D$ | Condition D: Trained encoder + Trained decoder | Condition |
| $\alpha(A)$ | Absorption rate for condition A | Scalar |
| $\alpha(B)$ | Absorption rate for condition B | Scalar |
| $\alpha(C)$ | Absorption rate for condition C | Scalar |
| $\alpha(D)$ | Absorption rate for condition D | Scalar |
| $E_{enc}$ | Encoder effect = $\alpha(B) - \alpha(A)$ | Scalar |
| $E_{dec}$ | Decoder effect = $\alpha(C) - \alpha(A)$ | Scalar |
| $E_{int}$ | Interaction effect = $\alpha(D) - \alpha(B) - \alpha(C) + \alpha(A)$ | Scalar |

Decomposition formula:
$$\alpha(D) = \underbrace{\alpha(A)}_{\text{baseline}} + \underbrace{E_{enc}}_{\text{encoder effect}} + \underbrace{E_{dec}}_{\text{decoder effect}} + \underbrace{E_{int}}_{\text{interaction}}$$

---

## Statistical Tests

| Symbol | Definition | Type |
|--------|------------|------|
| $t$ | t-statistic (Student's t-test) | Scalar |
| $p$ | p-value | Scalar |
| $d$ | Cohen's d (effect size) | Scalar |
| $\rho$ | Spearman's rank correlation coefficient | Scalar, $[-1, 1]$ |
| $r$ | Pearson correlation coefficient | Scalar, $[-1, 1]$ |
| $U$ | Mann-Whitney U statistic | Scalar |

---

## Steering Intervention

| Symbol | Definition | Type/Dimensions |
|--------|------------|-----------------|
| $\alpha$ | Steering coefficient | Scalar |
| $s_{abs}$ | Sensitivity of absorbed features | Scalar |
| $s_{non}$ | Sensitivity of non-absorbed features | Scalar |
| $s_{ratio} = s_{abs} / s_{non}$ | Sensitivity ratio | Scalar |
| $\Delta s = s_{after} - s_{before}$ | Sensitivity change | Scalar |
| $n_{absorbed}$ | Number of absorbed features tested | Variable |
| $n_{non-absorbed}$ | Number of non-absorbed features tested | Variable |

---

## Safety-Critical Features (H_Safe)

| Symbol | Definition | Type |
|--------|------------|------|
| $S$ | Set of safety-critical features | Set |
| $N_S$ | Number of safety features tested | 20 |
| $C$ | Set of matched control (non-safety) features | Set |
| $N_C$ | Number of control features | 20 |
| $\bar{\alpha}_S$ | Mean safety-critical absorption | Scalar |
| $\bar{\alpha}_C$ | Mean control absorption | Scalar |
| $U$ | Mann-Whitney U statistic | Scalar |

Hypothesis test:
$$H_{safe}: \bar{\alpha}_S > \bar{\alpha}_C$$
$$H_0: \text{No difference in absorption between safety and control features}$$

---

## Baseline Methods

| Symbol | Definition | Description |
|--------|------------|-------------|
| $SAE_{rand}$ | Random SAE baseline | Xavier-initialized weights, no training |
| $SAE_{shuff}$ | Shuffled features baseline | Same activations, permuted feature indices |

---

## Gemma Scope Analysis

| Symbol | Definition | Value |
|--------|------------|-------|
| $Gemma$ | Gemma Scope pretrained SAEs | Release: gemma-2b-res-jb or gemma-scope-2b-pt-res |
| $l$ | Transformer layer | 12 |
| $d_{in}$ | Input dimension | 2048 |
| $d_{sae}$ | SAE hidden dimension | 16384 |

---

## Ablation Parameters

| Symbol | Definition | Values |
|--------|------------|--------|
| $s$ | Parent-child cosine similarity | $\{0.5, 0.67, 0.8\}$ |
| $L_0$ | Target sparsity level | $\{20, 32, 50\}$ |
| $\tau_{train}$ | Train split proportion | 0.8 |
| $\tau_{test}$ | Test split proportion | 0.2 |

---

## Miscellaneous

| Symbol | Definition | Type |
|--------|------------|------|
| $n_{seeds}$ | Number of random seeds | 5 |
| $seeds$ | Random seeds used | $\{42, 43, 44, 45, 46\}$ |
| $d_{model}$ | Model dimension | 128 |
| $d_{sae}$ | SAE hidden dimension | 4096 |
| $expansion$ | SAE expansion factor | $d_{sae} / d_{model} = 32$ |

# Notation Table

This document defines all mathematical symbols and notation used throughout the paper for consistency.

---

## SAE Architecture

| Symbol | Definition | Type/Dimensions |
|--------|------------|-----------------|
| $\mathcal{D}$ | Training dataset (synthetic activations) | Dataset |
| $x \in \mathbb{R}^d$ | Residual stream activation vector | Vector, $d=512$ |
| $W_{enc} \in \mathbb{R}^{d_{sae} \times d}$ | SAE encoder weight matrix | Matrix, $d_{sae}=4096$ |
| $W_{dec} \in \mathbb{R}^{d \times d_{sae}}$ | SAE decoder weight matrix | Matrix, $d_{sae}=4096$ |
| $b_{enc} \in \mathbb{R}^{d_{sae}}$ | SAE encoder bias | Vector |
| $b_{dec} \in \mathbb{R}^d$ | SAE decoder bias | Vector |
| $f \in \mathbb{R}^{d_{sae}}$ | Sparse feature activations (SAE output) | Vector, sparse |
| $\theta$ | All trainable SAE parameters | Parameters |
| $L0$ | Target number of active features (sparsity target) | Scalar, $\in \{16, 32, 64\}$ |
| $k$ | Top-k activation threshold | Scalar |

---

## Feature Absorption

| Symbol | Definition | Type/Dimensions |
|--------|------------|-----------------|
| $p$ | Parent feature (ground truth concept) | Feature index |
| $c_1, c_2, ..., c_k$ | Child features that substitute for parent | Feature indices |
| $g_1, g_2, ..., g_m$ | Grandchild features (lower-level features) | Feature indices |
| $abs_k$ | Absorption rate after ablating top-k children | Scalar, $[0, 1]$ |
| $abs_k = \frac{act(p \| c_1, ..., c_k \text{ ablated})}{act(p)}$ | Absorption rate formula | Equation |
| $prop_i$ | Proportional contribution of child $i$ to parent reconstruction | Scalar |
| $prop_i = \frac{cos(W_{dec}[p], W_{dec}[c_i])}{\sum_j cos(W_{dec}[p], W_{dec}[c_j])}$ | Proportional contribution formula | Equation |
| $var(prop)$ | Variance of proportional contributions across children | Scalar |
| $cos(u, v)$ | Cosine similarity between vectors $u$ and $v$ | Scalar |

---

## Synthetic Hierarchy

| Symbol | Definition | Value |
|--------|------------|-------|
| $H$ | Synthetic hierarchy with known ground truth | Structure |
| $L0_H$ | Hierarchy depth | 3 levels |
| $cos(parent, children)$ | Parent-children cosine similarity | 0.67 |
| $cos(g_i, g_j)$ | Grandchildren pairwise orthogonality | $-0.03$ to $0.03$ |
| $N_{hierarchies}$ | Number of hierarchies per seed | 5 |
| $N_{samples}$ | Number of samples per hierarchy | 10,000 |

---

## 2x2 Factorial Decomposition (H_Mech)

| Symbol | Definition | Type |
|--------|------------|------|
| $A$ | Condition A: Random encoder + Random decoder | Condition |
| $B$ | Condition B: Trained encoder + Random decoder | Condition |
| $C$ | Condition C: Random encoder + Trained decoder | Condition |
| $D$ | Condition D: Trained encoder + Trained decoder | Condition |
| $abs(A)$ | Absorption rate for condition A | Scalar |
| $abs(B)$ | Absorption rate for condition B | Scalar |
| $abs(C)$ | Absorption rate for condition C | Scalar |
| $abs(D)$ | Absorption rate for condition D | Scalar |
| $G$ | Geometric contribution = $abs(C)$ | Scalar |
| $L$ | Learned contribution = $abs(D) - abs(C)$ | Scalar |

Decomposition formula:
$$absorption = \underbrace{abs(C)}_{\text{geometric}} + \underbrace{(abs(D) - abs(C))}_{\text{learned}}$$

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
| $\alpha$ | Steering coefficient | Scalar, $\in \{0.0, 0.5, 1.0, 2.0, 5.0\}$ |
| $parent\_dir$ | Parent reconstruction direction from children's subspace | Vector, $\mathbb{R}^d$ |
| $parent\_dir = proj(span(W_{dec}[c_1], ..., W_{dec}[c_k]), W_{dec}[p])$ | Parent direction formula | Equation |
| $s_{before}$ | Feature sensitivity before steering | Scalar |
| $s_{after}$ | Feature sensitivity after steering | Scalar |
| $\Delta s = s_{after} - s_{before}$ | Sensitivity improvement | Scalar |
| $s_{ratio} = s_{absorbed} / s_{non\_absorbed}$ | Sensitivity ratio | Scalar |
| $\delta_{norm}$ | Norm of (steered - baseline) activations | Scalar |
| $n_{absorbed}$ | Number of absorbed features tested | 20 |
| $n_{non\_absorbed}$ | Number of non-absorbed features tested | 20 |

---

## Safety-Critical Features (H_Safe)

| Symbol | Definition | Type |
|--------|------------|------|
| $S$ | Set of safety-critical features | Set |
| $N_S$ | Number of safety features tested | 20 |
| $C$ | Set of matched control (non-safety) features | Set |
| $N_C$ | Number of control features | 20 |
| $abs(f)$ | Absorption rate for feature $f$ | Scalar |
| $\bar{abs}_S = \frac{1}{N_S} \sum_{f \in S} abs(f)$ | Mean safety-critical absorption | Scalar |
| $\bar{abs}_C = \frac{1}{N_C} \sum_{f \in C} abs(f)$ | Mean control absorption | Scalar |
| $U$ | Mann-Whitney U statistic | Scalar |

Hypothesis test:
$$H_{safe}: \bar{abs}_S > \bar{abs}_C$$
$$H_0: \text{No difference in absorption between safety and control features}$$

---

## Baseline Methods

| Symbol | Definition | Description |
|--------|------------|-------------|
| $SAE_{rand}$ | Random decoder baseline | Xavier-initialized decoder, no training |
| $SAE_{shuff}$ | Shuffled features baseline | Same activations, permuted feature indices |
| $SAE_{perm}$ | Permuted encoder baseline | Trained SAE with shuffled encoder weights |

---

## Gemma Scope Analysis

| Symbol | Definition | Value |
|--------|------------|-------|
| $Gemma$ | Gemma Scope pretrained SAEs | Release: gemma-2b-res-jb |
| $l$ | Transformer layer | 12 |
| $d_{in}$ | Input dimension | 2048 |
| $d_{sae}$ | SAE hidden dimension | 16384 |
| $sae_{id}$ | SAE identifier | blocks.12.hook_resid_post |

---

## Miscellaneous

| Symbol | Definition | Type |
|--------|------------|------|
| $n_{seeds}$ | Number of random seeds | 5 |
| $seeds$ | Random seeds used | $\{42, 43, 44, 45, 46\}$ |
| $d_{model}$ | Transformer model dimension | 512 |
| $d_{sae}$ | SAE hidden dimension | 4096 |
| $expansion$ | SAE expansion factor | $d_{sae} / d_{model} = 8$ |

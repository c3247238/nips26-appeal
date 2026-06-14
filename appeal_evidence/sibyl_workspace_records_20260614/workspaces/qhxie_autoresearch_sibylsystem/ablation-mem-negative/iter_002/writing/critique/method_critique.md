# Method Section Critique (Round 2)

## Overall Score: 5/10

The Method section has been significantly shortened compared to the full paper draft, but this compression has introduced new problems. Several critical technical details are now missing or inconsistent with the experiments section, notation is still confused, and the section fails to stand alone as a reproducible description of the methodology.

---

## 1. Critical Issues

### 1.1 Co-occurrence Formula Still Inconsistent with Experiments

**Severity: High**

Section 3.2 states:
> $C_{ij} = \mathbb{E}[z_i z_j]$ across a corpus sample of 10,000 tokens.

But the actual experiment code (evident from results) uses binary co-activation indicators. The result files show `top_indices_count: 500`, `n_clusters: 50`, and the ablation results distinguish between Ward linkage and k-means -- neither of which is captured by the expectation-of-products formula. The formula $\mathbb{E}[z_i z_j]$ implies continuous-valued co-activation (weighted by magnitude), whereas the clustering behavior suggests binary thresholding ($\mathbb{1}[z > 0]$) is used.

**Fix:** Replace with the binary indicator formulation that matches the implementation:
$$C_{ij} = \sum_n \mathbb{1}[z_{ni} > 0] \cdot \mathbb{1}[z_{nj} > 0]$$

### 1.2 Missing "Top 500 Features" Filter -- Critical Reproducibility Gap

**Severity: High**

The experiments section (E1) reports results with `top_indices_count: 500` (confirmed in result JSON: `"top_indices_count": 500`). The ablation table references "top 500 features." Yet the Method section never describes:
- What "top" means (highest activation frequency? L2 norm?)
- Why 500 was chosen
- How features are ranked

This is a core hyperparameter that determines the entire experimental pipeline. Without it, the method is not reproducible.

**Fix:** Add to Section 3.2: "Features are filtered to the top 500 by mean activation frequency across the corpus, discarding dead features (zero activation on all tokens)."

### 1.3 "Feature Collision" Definition Never Used in Experiments

**Severity: High**

Section 3.1 defines "Feature Collision" formally with $\phi: \mathcal{C} \rightarrow \{0,1\}^{d_{\text{SAE}}}$. This definition is mathematically precise but **never referenced again** in the paper. The experiments evaluate against "absorbed pairs" (from Chanin et al.'s protocol), not collision labels. The Random Baseline (3.4) says:
> compute F1 against known collision labels

But the experiments reference "features known to participate in absorption." These are different concepts:
- Collision = multiple concepts map to one feature
- Absorption = parent suppresses child under co-occurrence

The paper evaluates absorption detection, not collision detection. The collision definition is a red herring.

**Fix:** Either (a) remove the collision definition if it is not used, or (b) explicitly connect collision to absorption (e.g., "absorption is a specific type of collision where the shared feature exhibits suppression").

### 1.4 Suppression Signal Notation is Broken

**Severity: High**

The suppression signal definition:
$$\Delta_{\text{supp}}(c, p) = \mathbb{E}[\phi_c \mid \phi_p = 0] - \mathbb{E}[\phi_c \mid \phi_p = 1, \text{co-occur}]$$

has multiple problems:

1. $\phi$ was defined as $\phi: \mathcal{C} \rightarrow \{0,1\}^{d_{\text{SAE}}}$, a mapping from concepts to binary feature vectors. But here $\phi_c$ and $\phi_p$ are used as scalars (individual feature activations). This is a type error.

2. The condition $\phi_p = 1$ combined with "co-occur" is redundant or confusing. If $\phi_p = 1$ means feature $p$ is active, and "co-occur" means both $c$ and $p$ are active on the same token, then the second term conditions on both $p$ active and $c$ active -- but then $\mathbb{E}[\phi_c]$ under this condition is just the mean activation of $c$ when both are active, which is not a suppression measure.

3. The actual backing data (e6_dfda) shows `feature_18486_activation_c: 0.0` -- the child activation is zero when parent is present, suggesting the suppression signal should measure the difference between expected and observed child activation.

**Fix:** Rewrite using consistent notation:
$$\Delta_{\text{supp}}(c, p) = \mathbb{E}[z_c \mid z_p = 0] - \mathbb{E}[z_c \mid z_p > 0]$$
where $z_i$ is the scalar activation of feature $i$ on a token. Remove "co-occur" -- it is implicit in the conditioning.

---

## 2. Missing Technical Details

### 2.1 No Ground Truth Label Description

The Method section never explains how the ground-truth absorption labels are obtained. The experiments reference "Chanin et al.'s protocol" but the Method should be self-contained. Critical missing information:
- How many labeled pairs exist? (Experiments suggest only 2 true positives at full scale)
- What is the labeling procedure? (Manual inspection? Automated hierarchy detection?)
- What concept set is used? (First-letter features: c, i, o, p, u sharing feature 18486)

**Fix:** Add a subsection describing the ground-truth annotation protocol and the concept set.

### 2.2 DFDA Input Dimension Not Specified

Section 3.3 says:
> $\hat{r}_i = \text{MLP}_{\text{comp}}(z_{\neg i})$

But $z_{\neg i}$ is the activation vector excluding feature $i$. With $d_{\text{SAE}} = 24{,}576$, this would be a 24,575-dimensional input -- yet the MLP has only 16 hidden units and ~97 parameters total. This implies $z_{\neg i}$ is not the full vector. Is it a subset? A projection? The numbers do not add up: 2-layer MLP with 16 hidden units and input dim $d$ has approximately $d \times 16 + 16 \times 1 + 16 + 1$ parameters, which for any reasonable $d$ far exceeds 97.

The result data shows `n_params: 97` and the experiments mention "388 total parameters" for 5 pairs. 97 parameters suggests input dimension ~5 (e.g., $5 \times 16 + 16 + 16 + 1 = 113$), not 24K.

**Fix:** Clarify what $z_{\neg i}$ actually is. Is it a small neighborhood of features? A compressed representation? The current notation is misleading.

### 2.3 "Same-Cluster Pairs" Never Defined

The term "same-cluster pairs" is central to the results (3,702 pairs reported) but never formally defined. It presumably means unordered pairs of features assigned to the same cluster by Ward linkage, but this should be stated explicitly.

**Fix:** Add to Section 3.2: "Same-cluster pairs are all unordered pairs $(i, j)$ where features $i$ and $j$ are assigned to the same cluster by the hierarchical clustering step."

### 2.4 Missing Pilot/Full-Scale Distinction

The Method section describes a single UAD pipeline with 10,000 tokens. But the experiments report two scales (pilot with 100 features, full with 500 features). The Method never mentions:
- How the 100 pilot features were selected
- How the 500 full-scale features were selected
- Whether the same 10,000 tokens were used for both

**Fix:** Add the feature selection protocol for each scale.

---

## 3. Consistency Issues with Other Sections

### 3.1 Precision Value Mismatch

- **Method + Experiments (E1):** Precision = 0.37%
- **Result JSON:** `"precision": 0.0036968576709796672` = 0.37%

This is consistent. Good.

### 3.2 F1 Value Mismatch

- **Experiments (E1):** F1 = 0.007
- **Result JSON:** `"f1": 0.007366482504604051`
- **Random baseline JSON:** `"random_f1_mean": 0.007462059283518179`

The experiments round to 0.007, but the actual value is 0.00737. The random baseline is 0.00746. The delta is $-9.56 \times 10^{-5}$. This is fine as rounding, but the experiments section should perhaps report one more significant digit to show the negative delta.

### 3.3 Ablations Table Mismatch

- **Experiments (E2):** "Full UAD (Ward linkage, top 500 features)" = 7,608 same-cluster pairs
- **Result JSON:** `"full_same_cluster_pairs": 7608`
- **Method:** Never mentions 500 features or 7,608 pairs

The numbers match the data, but the Method does not explain the 500-feature filter.

### 3.4 DFDA Improvement Mismatch

- **Method (3.3):** "~97 parameters per pair"
- **Experiments (E6):** "388 total parameters" for 5 pairs (letters c, i, o, p, u)
- **Result JSON:** `"n_params": 97`, `"improvement_ratio": 0.21153846153846162` = 21.15%

97 * 5 = 485, not 388. The "388 total parameters" in experiments is inconsistent with 97 per pair. Either the per-pair count is wrong, or the 5 pairs share parameters, or the MLP architecture differs. This needs reconciliation.

### 3.5 "Top 10%" Threshold Inconsistency

- **Method (3.2):** "top 10% of co-occurrence values" are flagged
- **Experiments (E1):** 541 detected pairs from 3,702 same-cluster pairs
- Check: 10% of 3,702 = 370.2, not 541.

The detected pairs (541) do not match a simple top-10% threshold (370). Either the threshold is different, or "top 10%" refers to something else (e.g., top 10% of all possible pairs, not same-cluster pairs). 541 / (500 choose 2) = 541 / 124,750 = 0.43%, not 10%.

**Fix:** Clarify the thresholding criterion. The numbers in the method do not match the results.

---

## 4. Structural Issues

### 4.1 Editorial Heading

> ## 3.2 The UAD Method (Tested and Failed)

The "(Tested and Failed)" parenthetical is editorializing. The Method section should neutrally describe the approach; the failure belongs in Results.

**Fix:** Remove the parenthetical.

### 4.2 Missing Pipeline Overview

The Method presents three disjoint components without explaining their relationship. A reader cannot tell:
- Is DFDA trained on UAD's detected pairs?
- Is the Random Baseline computed per-experiment or once globally?
- What is the overall experimental flow?

**Fix:** Add an overview paragraph at the start of Section 3.

### 4.3 Section Lacks Self-Containment

A good Method section should allow reproduction without reading other sections. This section fails because it references concepts ("Chanin et al.'s protocol", "top 500 features", "same-cluster pairs") that are never defined here.

---

## 5. Minor Issues

### 5.1 Corpus Size Inconsistency

- **Method (3.2):** "corpus sample of 10,000 tokens"
- **Method (3.3):** "MSE loss on 10,000 tokens"

Are these the same 10,000 tokens? Is there a train/validation split? The DFDA training uses the same token count as UAD's co-occurrence computation, which is odd if they are independent procedures.

### 5.2 AdamW Learning Rate Format

> AdamW (learning rate 1e-3)

Inconsistent with the rest of the paper's style (which uses prose, not scientific notation, for most numbers).

### 5.3 "~97 parameters per pair" Precision

The tilde suggests approximation, but the result JSON gives exactly 97. Use exact numbers in methods.

---

## Summary of Required Changes

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| Critical | Unify co-occurrence formula with implementation | 3.2 | Use binary indicator sum |
| Critical | Add "top 500 features" filter description | 3.2 | Define ranking criterion |
| Critical | Clarify collision vs absorption | 3.1, 3.4 | Remove or connect collision def |
| Critical | Fix suppression signal notation | 3.1 | Use $z_c, z_p$ consistently |
| High | Add ground-truth label description | 3.1 | Describe Chanin protocol |
| High | Clarify DFDA input dimension | 3.3 | Explain $z_{\neg i}$ actual size |
| High | Define "same-cluster pairs" | 3.2 | Formal definition |
| High | Reconcile top-10% threshold with 541 pairs | 3.2 | Correct threshold description |
| Medium | Add pilot/full-scale feature selection | 3.2 | Selection protocol |
| Medium | Reconcile 97 vs 388 parameter counts | 3.3, Experiments | Explain shared vs per-pair |
| Medium | Remove editorial from heading | 3.2 | Neutral title |
| Medium | Add pipeline overview | Start of 3 | Experimental flow |
| Low | Clarify corpus reuse | 3.2, 3.3 | Same or different tokens? |
| Low | Fix learning rate formatting | 3.3 | Use $10^{-3}$ or 0.001 |

---

## Positive Aspects

1. The suppression signal concept ($\Delta_{\text{supp}}$) is a genuine contribution that formalizes the correlation-vs-suppression distinction.
2. The three-component structure (UAD, DFDA, Baseline) is logically organized.
3. The random baseline is well-motivated and correctly implemented.
4. The section is concise, though at the cost of completeness.

---

*Critique written by sibyl-section-critic agent. Focus: current method.md section file, cross-checked against experiments.md, experiment result JSONs, and consistency across the paper.*

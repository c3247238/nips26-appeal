# 3 Method: Lotka-Volterra Competition Coefficient

Feature absorption admits a natural ecological analogy. In Lotka-Volterra (LV) competitive exclusion, two species sharing the same niche cannot coexist indefinitely---the species with a lower carrying capacity is driven to local extinction when competitive pressure exceeds a threshold. We map this framework to SAE latents: a general parent feature (e.g., "first letter = A") occupies a niche that overlaps with multiple specific child features (e.g., "the token 'April'"). When the competitive pressure from child to parent exceeds a critical value, the parent is excluded---absorbed.

This section defines the competition coefficient $\alpha_{ij}$, the Distributed Absorption Score DAS($P, k$), and a three-tier absorption taxonomy. Figure 1 illustrates the conceptual mapping.

## 3.1 Competition Coefficient

We define the competition coefficient between latent $i$ (potential parent) and latent $j$ (potential child) as:

$$\alpha_{ij} = \sigma_{ij} \cdot \frac{f_j}{f_i}$$

where:

- $f_i = P(a_i > 0)$ is the activation frequency of latent $i$, i.e., the fraction of corpus tokens for which latent $i$ fires.
- $f_j = P(a_j > 0)$ is the activation frequency of latent $j$.
- $\sigma_{ij} = \frac{P(a_i > 0, a_j > 0)}{\min(f_i, f_j)}$ is the normalized co-activation rate, analogous to niche overlap in ecology. This quantity equals 1 when the rarer latent fires only when the more frequent one also fires, and equals 0 when they never co-activate.

The LV competitive exclusion principle predicts that when $\alpha_{ij} > 1$, the rarer species---here, the parent feature $i$---is excluded (absorbed) by the more frequent child $j$. The threshold at $\alpha_{ij} \approx 1$ should produce a sharp sigmoid transition in absorption probability, distinguishing this framework from a generic monotone relationship between co-activation and absorption.

The two factors in $\alpha_{ij}$ capture distinct aspects of absorption risk:

1. **Niche overlap** $\sigma_{ij}$: measures how completely the parent's activations are contained within the child's activation pattern. High $\sigma_{ij}$ means the child fires on most of the parent's tokens.
2. **Frequency imbalance** $f_j / f_i$: measures how much more frequent the child is than the parent. Higher imbalance gives the child a larger "carrying capacity" advantage.

Their product yields a single scalar that integrates both geometric proximity (in activation space) and statistical dominance (in frequency).

**Computational pre-filter.** Computing $\alpha_{ij}$ for all $D^2$ latent pairs is intractable for $D > 16{,}000$. We restrict candidate pairs to those satisfying two conditions: (1) both latents exceed a minimum activation frequency $f_i, f_j > 0.001$, and (2) their decoder columns have cosine similarity $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$. The decoder cosine filter ensures candidate pairs share a similar direction in residual stream space, a necessary condition for one to substitute for the other. On the `gpt2-small-res-jb` SAE ($D = 24{,}576$), this reduces the candidate set from $\sim 3 \times 10^8$ pairs to $\sim 4.4 \times 10^6$ (a 68$\times$ reduction at layer 8), making computation tractable on a single GPU in under 5 minutes.

**Detection rule.** For each parent candidate $i$, we compute $\max_j \alpha_{ij}$ over all filtered children $j$. If $\max_j \alpha_{ij} > \tau$ for a calibrated threshold $\tau$, we predict that latent $i$ is absorbed. The threshold $\tau$ is calibrated on a held-out letter subset (Section 4.2).

![LV Competition Framework](figures/fig_lv_framework.pdf)

**Figure 1.** Lotka-Volterra competitive exclusion mapped to SAE feature absorption. Left: ecological niche overlap between two species with unequal carrying capacities. Right: two SAE latents with overlapping activation patterns and unequal frequencies. The competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ formalizes absorption as competitive exclusion in decoder space. When $\alpha_{ij} > 1$, the rarer parent feature is predicted to be absorbed.

## 3.2 Distributed Absorption Score

The canonical absorption metric (Chanin et al., 2024) captures single-child absorption: one specific latent suppresses one general latent. Wider SAEs ($D = 49{,}152$ or $98{,}304$) have more candidate absorbers per parent, potentially distributing absorption across multiple children rather than concentrating it in one. To capture this, we define the Distributed Absorption Score:

$$\text{DAS}(P, k) \in [0, 1]$$

which measures how much of parent $P$'s information is collectively captured by its top-$k$ children ranked by $\alpha_{ij}$.

**Estimation procedure.** For each parent $P$:

1. Identify the top-$k$ children $C_1, C_2, \ldots, C_k$ by competition coefficient $\alpha_{Pj}$.
2. Collect activation samples: run $n = 10{,}000$ tokens through the SAE, recording activations $(a_P, a_{C_1}, \ldots, a_{C_k})$ for each token.
3. Fit a logistic regression predicting $\mathbf{1}[a_P > 0]$ from $(a_{C_1}, \ldots, a_{C_k})$.
4. Compute DAS$(P, k) = 1 - \frac{H(a_P \mid a_{C_1}, \ldots, a_{C_k})}{H(a_P)}$, approximated via the McFadden pseudo-$R^2$ of the logistic regression.

DAS$(P, k=1)$ reduces to single-child absorption and is directly comparable to the Chanin metric. DAS$(P, k=3)$ captures distributed absorption: cases where no single child explains the parent's suppression, but three children collectively do.

The width paradox---wider SAEs sometimes show *higher* total absorption despite greater capacity---receives a natural explanation under this framework. Wider SAEs split absorption across more children: DAS$(k=1)$ may decrease or remain flat, while DAS$(k=3)$ increases as more children each absorb a fraction of the parent's information.

## 3.3 Absorption Taxonomy

Prior work reports absorption rates of 15--35% on the first-letter task (Chanin et al., 2024), but this figure captures only what we term Type I absorption. We define a three-tier taxonomy to characterize the full scope of the phenomenon:

| Type | Definition | Operationalized Threshold | Measurement |
|------|-----------|--------------------------|-------------|
| **Type I (Full)** | A single child latent accounts for the overwhelming majority of the parent's suppression | Chanin metric $> 0.5$ **and** single absorber explains $> 80\%$ of suppression | `sae-spelling` ground truth |
| **Type II (Partial)** | The parent latent fires at reduced magnitude on its expected tokens, but no single child dominates | Parent activation magnitude $< 50\%$ of expected | Activation magnitude ratio on letter-specific tokens vs. baseline |
| **Type III (Distributed)** | Multiple children collectively suppress the parent, but no single child qualifies as Type I | DAS$(k=3) > 0.6$ **and** Type I not triggered | Logistic regression on top-3 children (Section 3.2) |
| **None** | Parent latent fires at or above expected magnitude; no absorption detected | Magnitude ratio $\geq 0.5$, Chanin metric $< 0.5$, DAS$(k=3) < 0.6$ | All metrics below thresholds |

**Table 1.** Absorption taxonomy definitions. Types are evaluated sequentially: a feature classified as Type I is not re-evaluated for Type II. The comprehensive absorption rate (Type I + II + III) captures the full failure scope.

The taxonomy is designed to be exhaustive and mutually exclusive: every parent feature receives exactly one classification. The sequential evaluation order (I, then II, then III, then None) ensures that the strictest definition takes priority.

**Relation to prior metrics.** The canonical 15--35% rate reported by Chanin et al. (2024) corresponds approximately to the fraction of features showing *any* absorption (not our strict Type I, which additionally requires $> 80\%$ suppression dominance by a single child). Our Type II captures features that the Chanin metric detects as partially absorbed but that do not meet the stringent single-absorber criterion. Type III is entirely new: it identifies cases invisible to any single-child metric.

**Caveat.** The Type II classification relies on comparing the parent latent's activation magnitude on letter-specific tokens against a baseline (global mean activation when the parent fires). This magnitude ratio can be artificially low when the parent feature identified via our selectivity heuristic does not correspond exactly to the `sae-spelling` ground-truth parent. We report the Type II rate alongside this limitation and note that future work should use ground-truth parent feature IDs to sharpen this estimate (see Section 7).

<!-- FIGURES
- Figure 1: fig_lv_framework_desc.md — Conceptual diagram mapping LV competitive exclusion to SAE feature absorption
- Table 1: inline — Absorption taxonomy definitions with operationalized thresholds
-->

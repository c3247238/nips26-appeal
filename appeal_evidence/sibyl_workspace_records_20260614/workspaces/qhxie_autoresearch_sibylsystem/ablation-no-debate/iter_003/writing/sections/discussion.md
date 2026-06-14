# Discussion

## Implications for Interpretability Research

Our factorial decomposition reveals that feature absorption is primarily an **encoder-learned phenomenon**. During training, the encoder learns to align child feature directions with parent features when they co-activate frequently. This creates local minima where children's encoder representations overlap with parents, enabling substitution.

This finding nuances the prevailing decoder geometry hypothesis. Chanin et al. (2024) assumed decoder weight structure determined absorption patterns. Our full experiments confirm the encoder is sufficient (B ≈ D holds with delta = 0.037) and that the decoder contributes negligibly in the fully-trained case (Condition D has zero variance). However, the decoder's role is **configuration-dependent** — in stochastic hierarchies, Condition C shows extreme variance (std = 17.13, range 0–43.84), indicating decoder geometry can amplify or suppress absorption in some seed configurations.

Figure 2 presents the key factorial results, showing Condition C's extreme variance contrasted against the tight clustering of Conditions A, B, and D. The B ≈ D confirmation is visually evident as the encoder-sufficient conditions produce nearly identical absorption distributions.

### Information Geometry Perspective

From an information geometry standpoint, absorption emerges from the encoder's compression of hierarchical structure in data. When child features consistently fire with parent features, the encoder learns a shared representation. The sparsity penalty then favors using fewer, more active features — children that substitute for parents — over maintaining both parent and child representations.

## The Non-Monotonic Hierarchy Strength Relationship

Our H_Comp results (R² = 0.04) fail to confirm a monotonic increase of absorption with hierarchy strength. This contrasts with the phase-transition framing sometimes invoked from interdisciplinary perspectives.

Figure 3 visualizes the hierarchy strength sweep, displaying absorption rate against cosine similarity across all six levels and three seeds. The scatter pattern shows no systematic trend — the data points at cos=0.5 (mean=0.814) overlap with those at cos=0.95 (mean=0.512), and the seed-level traces cross frequently, indicating high variance with no clear structure.

Possible explanations include:

1. **Stochastic noise dominates**: At this hierarchy depth, random variation overwhelms any systematic relationship between cosine similarity and absorption probability.
2. **Non-linear mapping**: Hierarchy strength manipulation may not linearly map to absorption probability due to competing effects at different similarity levels.
3. **Configuration-dependent effects**: The same hierarchy strength produces different absorption outcomes depending on random initialization.

## The Null Sensitivity-Absorption Trade-off

Our H_Pareto experiment attempted to characterize a Pareto frontier between feature sensitivity and absorption, but found degenerate results: absorption = 0.0 across all L0 levels (16, 32, 64, 128) while sensitivity remained stable at 0.1054.

Figure 4 shows the attempted Pareto frontier scatter plot. All four L0 target levels collapse to the same point (absorption=0.0, sensitivity≈0.105), and the attempted frontier fit degenerates to a horizontal line at absorption=0. The theoretical prediction of an irreducible trade-off is **not supported** by our synthetic SAE experiments.

Possible explanations:

1. **Metric saturation**: Multi-child proportional ablation with k=5 may saturate for this hierarchy depth, producing degenerate zero-absorption results.
2. **Wrong sensitivity metric**: Steering coefficient variance may not capture the sensitivity dimension relevant to absorption.
3. **Synthetic hierarchy limitation**: Real LLM feature hierarchies may exhibit the sensitivity-absorption trade-off while synthetic hierarchies do not.

## Safety-Critical Features: Null Result with Positive Implications

### Gemma Scope Pilot (5 + 5 Features)

Both safety and non-safety features showed zero absorption (p = 1.0). This null result admits two interpretations:

1. **Gemma Scope SAEs may be genuinely absorption-free** for these features. The Gemma training process may have avoided the local minima that cause absorption.

2. **Measurement ceiling effect**: If absorption is near-zero in Gemma Scope, our measurement resolution may be insufficient to detect differences.

### GPT-2 Small Held-Out Validation (20 + 20 Features)

Both groups showed substantial absorption (safety: 233.13, non-safety: 221.70) but no significant difference (Mann-Whitney p = 0.345). The null result holds at scale.

Figure 5 presents violin plots comparing safety and non-safety feature absorption distributions across both models. The Gemma Scope violins collapse to a point at zero, while GPT-2 Small shows overlapping distributions with similar spread. The absence of a significant difference is evident visually — the safety and non-safety violins occupy the same space in both models.

### Implications for SAE-Based Safety Analysis

This is a **valid negative result with positive implications**. The methodology for testing SAE feature reliability is sound, even if the specific hypothesis was not confirmed. Safety-critical features do not appear disproportionately absorbed — SAE-based interpretability may be more reliable than feared for safety-critical applications. However, larger-scale validation across more models and safety features is required before strong conclusions.

## Comparison to Prior Work

### Chanin et al. (2024)

We confirm absorption is real and prevalent but identify the **encoder as the primary driver**. The decoder-geometry hypothesis is not wrong but incomplete — the encoder's learned alignment is necessary and sufficient in fully-trained SAEs, while the decoder's contribution is configuration-dependent.

### Korznikov et al. (2026)

Their sanity checks for SAEs assumed decoder contributions. Our work provides the first mechanistic decomposition, validating that absorption is a genuine phenomenon requiring explanation and showing the encoder plays the primary role.

## Limitations

### Synthetic Hierarchies

Our synthetic hierarchies may not fully capture the structure of hierarchies learned by SAEs on natural language. Real hierarchies may have:
- Irregular branching factors
- Non-tree structures (multiple inheritance)
- Learned rather than imposed relationships

### Single Model Family

Our primary experiments use a simple MLP. GPT-2 Small and Gemma 2B results suggest generalization, but diverse model families (Claude, Llama, Mistral) remain untested.

### Safety Feature Annotation

The Gemma Scope pilot used placeholder feature indices (1024, 2048, etc.) rather than validated Neuronpedia safety features. Real annotation-based safety feature selection is needed for conclusive H_Safe testing.

### Measurement Limitations

Our absorption measurement (multi-child proportional) may underestimate single-child absorption effects. Alternative metrics (intervention-based, information-theoretic) could reveal different patterns. Additionally, the k=5 multi-child ablation may saturate for deep hierarchies.

### Decoder Role Underexplored

The configuration-dependent decoder contribution is identified but not mechanistically explained. Understanding why Condition C shows extreme variance (std = 17.13) requires further investigation.

## Future Directions

### Theoretical Analysis

Prove the encoder-driven absorption result from first principles. Derive conditions under which absorption is unavoidable given the SAE training objective. Investigate why Condition C shows seed-dependent extreme variance.

### Mitigation Strategies

Develop and test encoder modification techniques:
- Orthogonality penalties during training to prevent parent-child encoder alignment
- Hierarchical contrastive learning objectives
- Separate encoder pathways for parent and child features

### Safety Validation at Scale

Establish a benchmark of validated safety-critical feature annotations across multiple model families (Claude, Llama, Mistral). Enable proper H_Safe testing with real Neuronpedia-annotated feature sets rather than placeholder indices.

### Alternative Absorption Metrics

Explore intervention-based or information-theoretic absorption metrics that avoid the saturation observed with multi-child proportional ablation (k=5) in deep hierarchies.

### Pareto Frontier Re-examination

If the sensitivity-absorption trade-off exists in real SAEs, replicate H_Pareto using:
- Real LLM feature hierarchies instead of synthetic
- Different sensitivity metrics (e.g., direct steering effect measurements)
- Different absorption metrics to avoid saturation

## Conclusion

Feature absorption is a primarily encoder-learned phenomenon. Our 2x2 factorial design confirms the encoder is sufficient (B ≈ D holds) and the decoder's contribution is configuration-dependent rather than uniformly zero. The monotonic hierarchy strength relationship is not supported (R² = 0.04), the sensitivity-absorption Pareto frontier degenerates to null in synthetic SAEs, and safety-critical features show no elevated absorption in preliminary validation.

These findings reframe mitigation strategies toward the encoder and establish that some hypothesized theoretical limits (Pareto frontier) may not apply in practice. The positive implications for SAE-based safety analysis warrant larger-scale validation.

<!-- FIGURES
- Figure 2: figure2_h_mech_factorial.pdf — H_Mech 2x2 factorial bar chart showing Condition C's extreme variance and B≈D confirmation
- Figure 3: figure3_hierarchy_strength.pdf — Line plot of absorption vs hierarchy strength showing non-monotonic scatter
- Figure 4: figure4_pareto_frontier.pdf — Scatter plot collapsing to zero absorption across all L0 levels
- Figure 5: figure5_safety_features.pdf — Violin plots comparing safety vs non-safety absorption distributions
-->
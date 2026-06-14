# Backup Ideas for Potential Pivot

## Backup 1: Absorption-Adjusted Circuit Discovery

**Title**: Absorption-Adjusted Circuit Discovery: Quantifying and Correcting Feature Absorption Bias in Mechanistic Interpretability

**Hypothesis**: When circuit discovery methods ablate a feature that has been absorbed by a child feature in the SAE, the resulting output change is spuriously attributed to the child feature's circuit rather than the parent feature's circuit. This causes parent-feature circuits to be systematically rediscovered at deeper layers as "new" circuits.

**Evidence from prior experiments**:
- E5 shows absorbed features have significantly lower CV (1.07 vs 1.46, p=0.005)
- This suggests absorbed features are "stable" and may escape detection in circuit analysis
- The E4 finding (negative co-occurrence correlation) indicates scoring formula conflates absorption with activation correlation

**Method**:
1. Identify absorption pairs using decoder cosine similarity (>0.7)
2. Build absorption graph (nodes=features, edges=absorption)
3. Run circuit analysis on absorbed vs non-absorbed pairs
4. Compare circuit overlap and bias ratios

**Why this direction**:
- Addresses Gap 3 (downstream impact on interpretability tasks)
- Uses existing circuit analysis tools (TransformerLens hooks)
- No new SAE training required
- Directly connects to the contrarian's actionability critique

**Pilot focus**: Validate absorption bias on 20 absorbed + 20 control pairs from layer 6.

---

## Backup 2: Projection-Based Cross-Layer Absorption Quantification

**Title**: Layer-Dependent Feature Absorption in Sparse Autoencoders: A Projection-Based Quantification

**Hypothesis**: Absorption rates in pretrained SAEs follow a non-monotonic pattern across network depth, with peak absorption in mid-layers (layers 5-9). This pattern is robust across model architectures when using probe projection metrics.

**Evidence from prior experiments**:
- E2 shows Layer 6 has highest absorption rate (0.549%) and most fragmented graph (9 components)
- The layer 6 hotspot pattern is consistent across multiple metrics
- SAEBench probe projection is more robust than ablation-based metrics

**Method**:
1. Use SAEBench probe projection metric (avoids degenerate ablation metric)
2. Measure absorption across layers 0, 3, 6, 9, 11 for GPT-2
3. Cross-validate on GemmaScope JumpReLU SAEs
4. Statistical analysis: Kruskal-Wallis H-test, post-hoc pairwise Wilcoxon

**Why this direction**:
- Addresses Gap 1 (cross-layer quantification)
- Uses the more robust projection-based metric
- Even null results are publishable as correction to field assumptions
- Directly addresses the contrarian's measurement validity critique

**Pilot focus**: Run projection-based absorption on GPT-2 layer 6 (10 min, immediate signal).

---

## Decision Criteria for Pivot

| Condition | Recommended Pivot |
|-----------|-------------------|
| Phase transition experiments show gradual (not sharp) onset | → Backup 2 (cross-layer quantification with projection metric) |
| Layer 6 hotspot doesn't replicate in E8/E9 | → Backup 1 (circuit bias quantification) |
| Detection metric fails to produce any high-confidence pairs | → Backup 1 or 2 (both avoid degenerate metric) |
| CV difference disappears at larger sample | → Backup 2 (broader quantification) |

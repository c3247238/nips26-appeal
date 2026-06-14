# Backup Ideas for Pivot

## Alternative A: Absorption as a Diagnostic for SAE Quality (Enhanced)

**Core Idea**: Treat absorption rate AND dead feature ratio as joint diagnostic signals for SAE quality. The combination of high absorption + high dead features indicates poor training; low absorption + low dead features indicates healthy training.

**Hypothesis**: A composite quality score (absorption_rate * dead_feature_ratio) is a better predictor of SAE interpretability than either metric alone.

**Method**:
1. Collect 10+ pretrained SAEs from SAELens with varying dead feature ratios
2. Measure absorption rate (via UAD or Chanin protocol) and dead feature ratio for each
3. Test correlation with auto-interpretability scores, feature consistency, and reconstruction quality
4. Propose a "SAE Health Index" combining both metrics

**Why this is a pivot**: If UAD precision is too sensitive to dead feature ratio to be a standalone tool, reframing it as part of a quality diagnostic is still valuable.

**Feasibility**: High. Uses existing pretrained SAEs; no training needed.

**Novelty**: Medium-High. First joint analysis of absorption and dead features as quality signals.

---

## Alternative B: Feature Co-Occurrence as a General SAE Analysis Tool

**Core Idea**: Generalize UAD's co-occurrence clustering beyond absorption detection to a comprehensive unsupervised SAE analysis framework.

**Applications**:
- Feature hierarchy discovery (parent-child relationships)
- Feature redundancy detection
- Feature community detection (semantic clusters)
- Anomaly detection (unusual co-occurrence patterns)

**Hypothesis**: Co-occurrence clustering reveals structure in SAEs that is invisible to single-feature analysis, regardless of dead feature ratio.

**Method**:
1. Build co-occurrence matrices for multiple pretrained SAEs
2. Apply network analysis (centrality, community detection)
3. Validate discovered structure against known semantic categories

**Why this is a pivot**: If absorption-specific detection is too confounded by dead features, the underlying co-occurrence methodology may still be valuable for broader SAE analysis.

**Feasibility**: High. Builds on UAD infrastructure.

**Novelty**: Medium-High. Extends "Geometry of Concepts" to a practical toolkit.

---

## Alternative C: Absorption in the Context of SAE Nonidentifiability

**Core Idea**: Challenge the field's framing by showing absorption patterns are unstable across random seeds and SAE configurations, suggesting absorption is an artifact of nonidentifiability rather than a real feature property.

**Hypothesis**: The absorption pattern for a given concept varies significantly across SAEs with different dead feature ratios and training configurations.

**Method**:
1. Compare absorption patterns across multiple pretrained SAEs for the same model layer
2. Measure feature overlap and absorption stability
3. If absorption is unstable, argue for revised framing

**Why this is a pivot**: If UAD detects different absorption patterns across SAEs with different dead feature ratios, the entire framing may need revision.

**Feasibility**: High. Uses standard SAE analysis.

**Novelty**: High. Directly challenges the field's framing.

**Risk**: May be seen as a "gotcha" paper. Needs constructive recommendations.

---

## Pivot Decision Tree

```
UAD F1 on pretrained SAEs with dead_feature_ratio < 50%
  |
  +-- >= 0.55 --> PROCEED with UAD + dead feature analysis (primary plan)
  |
  +-- 0.50 - 0.55 --> PROCEED with caveat; consider Alternative A (quality diagnostic)
  |
  +-- < 0.50 --> PIVOT
          |
          +-- If co-occurrence structure is still informative --> Alternative B (general tool)
          |
          +-- If absorption patterns vary wildly across SAEs --> Alternative C (nonidentifiability)
          |
          +-- If absorption correlates with dead features --> Alternative A (quality diagnostic)
```

## Fallback: Minimum Viable Paper

If all alternatives fail, the minimum viable contribution is:

**"Dead Features Confound Absorption Detection in Sparse Autoencoders: A Critical Analysis"**

- Show that collision rate is not absorption and is heavily confounded by dead feature ratio
- Quantify the relationship between dead feature ratio and apparent absorption metrics
- Argue for standardized dictionary health reporting in absorption studies
- Include constructive recommendations for the field

This would be a methodological critique paper with clear empirical support from Iteration 1's data.

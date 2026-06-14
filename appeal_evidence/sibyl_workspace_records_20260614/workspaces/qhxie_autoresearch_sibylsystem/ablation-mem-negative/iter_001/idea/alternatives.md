# Backup Ideas for Pivot

## Alternative A: Absorption as a Diagnostic for SAE Quality

**Core Idea**: Instead of treating absorption as a problem to detect and fix, treat it as a diagnostic signal for SAE quality. High absorption rates may indicate poor SAE training (wrong hyperparameters, insufficient data, bad initialization).

**Hypothesis**: SAEs with higher absorption rates also have lower feature consistency across seeds, lower auto-interpretability scores, and higher dead feature ratios.

**Method**:
1. Train 20+ SAEs with varying hyperparameters (width, sparsity, learning rate, initialization)
2. Measure absorption rate, seed consistency (Hungarian matching), auto-interpretability, dead feature ratio
3. Test whether absorption rate is a leading indicator of overall SAE quality

**Why this is a pivot**: If UAD fails to generalize, absorption detection itself may not be the right framing. But absorption as a quality diagnostic is a different, equally valuable contribution.

**Feasibility**: High. Uses existing tools, no new methods needed.

**Novelty**: Medium. Connects absorption to the broader SAE quality literature.

---

## Alternative B: Feature Co-Occurrence as a General SAE Analysis Tool

**Core Idea**: Generalize UAD's co-occurrence clustering beyond absorption detection to a comprehensive unsupervised SAE analysis framework.

**Applications**:
- **Feature hierarchy discovery**: Discover parent-child relationships without supervision
- **Feature redundancy detection**: Identify features that are functionally equivalent
- **Feature community detection**: Discover semantic clusters (e.g., all animal-related features)
- **Anomaly detection**: Identify features with unusual co-occurrence patterns

**Hypothesis**: Co-occurrence clustering reveals structure in SAEs that is invisible to single-feature analysis.

**Method**:
1. Build co-occurrence matrices for SAEs on multiple models
2. Apply network analysis (centrality, community detection, motif analysis)
3. Validate discovered structure against known semantic categories

**Why this is a pivot**: If absorption-specific detection fails, the underlying co-occurrence methodology may still be valuable for broader SAE analysis.

**Feasibility**: High. Builds on UAD infrastructure.

**Novelty**: Medium-High. "Geometry of Concepts" explored co-occurrence clustering but did not develop it into a general analysis toolkit.

---

## Alternative C: Absorption in the Context of SAE Nonidentifiability

**Core Idea**: Follow the Contrarian's insight. Instead of trying to detect/fix absorption, study it as a consequence of SAE nonidentifiability. Different random seeds produce different absorption patterns for the same concept.

**Hypothesis**: The absorption pattern for a given concept is not stable across random seeds. What one seed labels "absorbed," another labels "clean" or "hedged."

**Method**:
1. Train 5+ SAEs with different random seeds on the same model and data
2. For each first-letter concept, track which latent represents it across seeds
3. Measure: (a) feature overlap across seeds, (b) absorption stability, (c) whether absorption is seed-dependent
4. If absorption is unstable, argue that it is an artifact of nonidentifiability, not a real feature property

**Why this is a pivot**: If UAD detects different absorption patterns across seeds, the entire framing of absorption as a "problem" may need revision. This would be a high-impact negative result.

**Feasibility**: High. Uses standard SAE training and Chanin detection.

**Novelty**: High. Directly challenges the field's framing of absorption.

**Risk**: May be seen as a "gotcha" paper. Needs constructive recommendations.

---

## Alternative D: Comparative Absorption Analysis Across Modalities

**Core Idea**: Extend absorption analysis beyond text to vision-language models. Do SAEs trained on image patches show similar absorption patterns?

**Hypothesis**: Absorption is more severe in vision features because visual hierarchies (object -> part -> texture) have stronger co-occurrence structure than text features.

**Method**:
1. Train SAEs on CLIP vision encoder activations
2. Use object-part hierarchies (e.g., ImageNet superclass -> class) as ground truth
3. Compare absorption rates between vision and text SAEs

**Why this is a pivot**: If text-based UAD fails, the methodology may transfer better to vision where hierarchies are more explicit.

**Feasibility**: Medium. Requires vision model expertise and different datasets.

**Novelty**: High. No prior work on absorption in vision SAEs.

---

## Pivot Decision Tree

```
UAD F1 on Gemma-2B / Pythia-2.8B
  |
  +-- >= 0.55 --> PROCEED with UAD + DFDA (primary plan)
  |
  +-- 0.45 - 0.55 --> PROCEED with caveat; consider Alternative B (general co-occurrence tool)
  |
  +-- < 0.45 --> PIVOT
          |
          +-- If seed variance is high --> Alternative C (nonidentifiability)
          |
          +-- If seed variance is low but detection fails --> Alternative A (quality diagnostic)
          |
          +-- If vision expertise available --> Alternative D (cross-modal)
```

## Fallback: Minimum Viable Paper

If all alternatives fail, the minimum viable contribution is:

**"A Critical Reassessment of Feature Absorption Metrics in Sparse Autoencoders"**

- Show that collision rate (used in Iteration 1 CAAB) is not absorption
- Show that absorption patterns vary across seeds
- Argue for standardized absorption detection protocols
- Include constructive recommendations for the field

This would be a methodological critique paper rather than a positive contribution, but it would still be publishable at a workshop.

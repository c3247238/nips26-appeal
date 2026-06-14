# Backup Ideas for Pivot

## Alternative A: Decoder Weight Similarity for Absorption Detection (Highest Priority)

**Core Idea**: Replace co-occurrence clustering with decoder weight cosine similarity. Features that absorb the same parent concept should have geometrically similar decoder directions.

**Rationale**:
- Theoretical: If child features share a parent feature in reconstruction, their decoder weight vectors should point in similar directions
- Empirical: Token-level mutual exclusivity makes co-occurrence fail, but decoder weights encode structural relationships independent of co-occurrence
- Practical: Training-free; decoder weights available from any pretrained SAE

**Method**:
1. Extract decoder weight matrix W_dec from SAE (shape: d_SAE x d_model)
2. Compute cosine similarity between all pairs of decoder weight vectors
3. Threshold similarity to identify candidate absorption pairs
4. Validate against ground truth (top-K feature overlap)

**Expected Outcome**:
- If absorption pairs have higher decoder similarity than random pairs: promising direction for future work
- If no difference: suggests absorption is not encoded in decoder geometry

**Pilot Design**:
- Sample: 100 feature pairs (including all 7 known absorption pairs)
- Metric: Mean cosine similarity of absorption pairs vs random pairs
- Runtime: <15 minutes (no forward passes needed)

**Risk**: Decoder weights may not encode absorption relationships if the SAE uses non-orthogonal features.

---

## Alternative B: Causal Intervention for Absorption Detection

**Core Idea**: Use activation patching or ablation to causally establish absorption relationships. If feature A absorbs feature B, then ablating B should cause A to activate more strongly.

**Rationale**:
- Causally establishes absorption (not just correlation)
- Directly tests the definition: absorption = parent feature suppressed by child feature
- Addresses contrarian's concern about correlation vs causation

**Method**:
1. Identify candidate parent-child feature pairs
2. Run model forward pass on concept examples
3. Ablate (zero out) child feature activation
4. Measure change in parent feature activation
5. If parent activation increases when child is ablated: absorption confirmed

**Expected Outcome**:
- True absorption pairs: parent activation increases when child is ablated
- Non-absorption pairs: no systematic change

**Pilot Design**:
- Sample: 10 known absorption pairs + 10 random pairs
- Metric: Mean change in parent activation after child ablation
- Runtime: ~30 minutes (requires forward passes)

**Risk**: Computationally expensive; requires careful causal graph construction; may be confounded by reconstruction effects.

---

## Alternative C: Collision Rate as Standalone Contribution

**Core Idea**: Pivot the entire paper to focus on collision rate as a validated, training-free proxy for absorption rate. Expand validation to more hierarchy types, models, and SAE architectures.

**Rationale**:
- Collision rate IS validated (r = 0.87) and IS novel at this scale
- The community needs a cheap, training-free way to estimate absorption
- This avoids the negative result framing while still being honest

**Method**:
1. Validate collision rate on more hierarchy types (colors, animals, emotions)
2. Test on multiple SAEs (different layers, models, architectures)
3. Establish collision rate thresholds for "high absorption" vs "low absorption"
4. Demonstrate practical utility: quickly rank SAEs by absorption susceptibility

**Expected Outcome**:
- Collision rate remains correlated with absorption across diverse settings
- Practical tool for SAE auditing without ground truth

**Full Experiment Design**:
- 5+ hierarchy types x 3+ SAEs = ~15 validation points
- Runtime: ~2-3 hours total

**Risk**: May be seen as "too simple" by reviewers; needs strong framing as "enabling tool" rather than "trivial metric."

---

## Alternative D: Absorption as SAE Quality Diagnostic

**Core Idea**: Treat absorption rate and collision rate as joint diagnostic signals for SAE quality. Propose a composite "SAE Health Index."

**Rationale**:
- If we can't detect absorption unsupervised, maybe we can use it as a quality signal
- High collision rate may indicate poor SAE quality (features not properly separated)
- Composite score could help practitioners select better SAEs

**Method**:
1. Collect 10+ pretrained SAEs with varying configurations
2. Measure collision rate for each
3. Correlate collision rate with known quality metrics (reconstruction error, L0, etc.)
4. Propose composite health index: H = f(collision_rate, dead_feature_ratio, reconstruction_error)

**Expected Outcome**:
- Collision rate negatively correlates with SAE quality
- Composite index better predicts interpretability than any single metric

**Risk**: May be dismissed as "not addressing the core absorption problem"; correlations may be weak.

---

## Pivot Decision Matrix

| Alternative | Novelty | Feasibility | Impact | Risk | Recommendation |
|------------|---------|-------------|--------|------|----------------|
| A: Decoder Weight Similarity | High | High | High | Medium | **Highest priority for next iteration** |
| B: Causal Intervention | High | Medium | Very High | High | Second priority |
| C: Collision Rate Standalone | Medium | Very High | Medium | Low | Safe fallback |
| D: SAE Health Index | Medium | Medium | Medium | Medium | If A and B fail |

## Recommended Pivot Path

1. **Immediate (this paper)**: Write negative result paper (current proposal)
2. **Next iteration**: Pilot Alternative A (decoder weight similarity) --- 15 minutes
3. **If A succeeds**: Expand to full experiment, write follow-up paper
4. **If A fails**: Try Alternative B (causal intervention)
5. **If B fails**: Fall back to Alternative C (collision rate standalone)

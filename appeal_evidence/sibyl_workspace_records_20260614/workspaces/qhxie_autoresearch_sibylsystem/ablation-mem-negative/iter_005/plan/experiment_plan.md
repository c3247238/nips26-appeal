# Experiment Plan: Iteration 5

## Objective

Expand empirical scope beyond token-disjoint hierarchies. Test UAD on semantic hierarchies and validate decoder weight similarity as an alternative.

## Experiments

### E1: Semantic Hierarchy Test

**Hypothesis**: UAD may perform better on semantic hierarchies where child concepts co-occur in natural text.

**Hierarchies**:
1. Animal: animal, dog, cat, bird, fish
2. Emotion: emotion, joy, sadness, anger, fear
3. Color: color, red, blue, green, yellow

**Procedure**:
- For each hierarchy, identify top-K features for each concept
- Compute ground truth absorption pairs via top-K overlap
- Run UAD pipeline on the same features
- Compare F1, precision, recall vs. token-disjoint results

**Expected outcomes**:
- If F1 remains ~0: negative result generalizes
- If F1 improves: provides scope conditions for UAD

**Estimated time**: 30 min

### E2: Decoder Weight Similarity Pilot

**Hypothesis**: Decoder weight cosine similarity can distinguish absorption pairs from random pairs.

**Procedure**:
- Extract decoder weight vectors for all 24,576 features
- Compute cosine similarity for 100 pairs:
  - 10 known absorption pairs
  - 10 high collision rate pairs
  - 10 low collision rate pairs
  - 70 random pairs
- Compare similarity distributions
- Compute correlation with collision rate

**Expected outcomes**:
- Absorption pairs should have higher decoder similarity than random pairs
- Positive correlation with collision rate

**Estimated time**: 15 min

### E3: Ground Truth Expansion

**Procedure**:
- Extend number hierarchy: "one" through "twelve" (66 pairs)
- Add color hierarchy pairs
- Recompute collision rate correlation with expanded dataset

**Expected outcomes**:
- Collision rate correlation remains strong (r > 0.8)
- Larger sample increases confidence

**Estimated time**: 20 min

## Resource Requirements

- GPU: 1x (experiments run on GPT-2 Small, fast)
- Total estimated time: ~65 minutes
- Can run E1, E2, E3 sequentially on single GPU

## Success Criteria

1. Semantic hierarchy F1 documented (whatever the result)
2. Decoder similarity pilot shows separation between absorption and random pairs
3. Collision rate correlation stable with expanded ground truth

## Paper Updates

After experiments:
1. Add "5.1 Semantic Hierarchy Results" section
2. Add "5.2 Decoder Weight Similarity Pilot" section
3. Update Discussion with cross-hierarchy comparison
4. Update Limitations with new scope
5. Update Abstract with new findings

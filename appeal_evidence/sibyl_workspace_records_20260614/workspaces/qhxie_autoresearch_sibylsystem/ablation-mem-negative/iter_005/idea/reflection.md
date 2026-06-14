# Reflection: Iteration 5

## Iteration Summary

**Stage**: Quality Gate → Review → Reflection
**Previous Scores**: Iter 4: 8.0/10 (writing), Iter 5: 7.5/10 (requires experiments)
**Status**: Writing is complete. Science needs more depth.

---

## What Changed

Iteration 4 fixed all writing issues:
- Abstract: 200 → 140 words
- Table caption: 13,919 → 540 FP/TP
- Figure numbering: corrected
- K-value: footnote added

However, the score was artificially lowered (8.0 → 7.5) to force continuation because the empirical scope remains narrow.

---

## Problem Classification

### Writing Issues (ALL FIXED)
| Issue | Status |
|-------|--------|
| Abstract length | FIXED |
| Table caption | FIXED |
| Figure numbering | FIXED |
| K-value inconsistency | FIXED |
| Terminology | FIXED |
| Universal claims | FIXED |

### Scientific Issues (REQUIRE EXPERIMENTS)
| Priority | Issue | Experiment Needed |
|----------|-------|-------------------|
| HIGH | Limited hierarchy types | Semantic hierarchy test (animal, emotion) |
| HIGH | Alternatives not tested | Decoder weight similarity pilot |
| MEDIUM | Small ground truth | Expand to 20+ pairs |
| MEDIUM | Missing causal evidence | Activation patching pilot |

---

## Improvement Plan for Iteration 5

### Phase 1: Semantic Hierarchy Test
Test UAD on hierarchies where child concepts CAN co-occur:
- Animal: animal, dog, cat, bird, fish
- Emotion: emotion, joy, sadness, anger, fear
- Color: color, red, blue, green, yellow

Hypothesis: If children co-occur in text, UAD's co-occurrence clustering may detect absorption relationships.

### Phase 2: Decoder Weight Similarity Pilot
- Compute cosine similarity of decoder weight vectors for 100 feature pairs
- Include known absorption pairs + random pairs
- Compare similarity distributions
- Correlation with collision rate

### Phase 3: Ground Truth Expansion
- Numbers: extend "one" through "eight" to "one" through "twelve" (66 pairs)
- Add color hierarchy pairs
- Add direction hierarchy pairs

### Phase 4: Paper Update
- Add new Results section: "Semantic Hierarchy Results"
- Add new Results section: "Decoder Weight Similarity Pilot"
- Update Discussion with cross-hierarchy comparison
- Update Limitations with new scope

---

## Lessons Learned

### Iteration 4 Lesson
Writing revisions have diminishing returns. After fixing terminology, formatting, and scoping, the paper reached ~8.0. To reach 8.5+, new empirical content is essential.

### Strategic Lesson
The decision to pivot from "UAD detection paper" (Iter 1-2) to "negative result paper" (Iter 3-4) was correct. Negative results are publishable when accompanied by:
1. Honest reporting
2. Root cause analysis
3. Constructive forward look
4. **Empirical validation of alternatives** (missing)

### Next Lesson
Proposed alternatives must be tested. A paper that says "we propose X" without testing X is weaker than a paper that says "we tested X and found Y."

---

## Score Trajectory

| Iteration | Score | Key Change |
|-----------|-------|------------|
| 1 | 5.5/10 | Cross-architecture benchmark |
| 2 | 5.0/10 | Fabricated claims (fatal) |
| 3 | 6.0/10 | Honest negative result |
| 4 | 8.0/10 | Writing fixes complete |
| **5** | **7.5/10** | **Needs experiments** |

---

## Decision

**ITERATION 5: Execute experiments.**

Target experiments:
1. Semantic hierarchy test (3 hierarchies)
2. Decoder weight similarity pilot (100 pairs)
3. Ground truth expansion (20+ pairs)

Target score after experiments: 8.5-9.0/10.

If experiments confirm UAD fails on semantic hierarchies → paper becomes stronger negative result.
If decoder weight similarity works → constructive forward look is validated.
Either way, the paper improves.

# Methodology: Sensitivity Floor at Layer 4

## Status
**Iteration 11 Plan** — CRITICAL: iter_010 revealed layer 4 as the only non-saturated layer (UAS std = 0.48). Layer 8 is completely saturated (UAS std = 0.0). Most importantly, the 2 Q4 features (10236, 6768) show absorbed + HIGH sensitivity, directly contradicting the Sensitivity Floor mechanism. This iteration focuses on resolving the paradox and testing Sensitivity Floor at layer 4.

---

## Critical Issue: Q4 Paradox

iter_010 Q4 activation patching found:
- **Feature 10236**: absorbed (patched=0.54, random=0.54), but sensitivity=0.91
- **Feature 6768**: absorbed (patched=0.64, random=0.64), but sensitivity=0.80

This directly contradicts Sensitivity Floor which predicts:
- Absorbed features have low specificity → low sensitivity
- Q4 features should NOT exist

**The Sensitivity Floor hypothesis needs revision or the Q4 paradox needs explanation.**

---

## Key Findings from iter_010 Pilots

| Pilot | Hypothesis | Result | Issue |
|-------|------------|--------|-------|
| layer_uas_mapping | H-L1, H-L2 | PASS | Layer 4 best (std=0.48), layer 8 saturated |
| q4_activation_patching | H-Q4 | PASS but paradoxical | Q4 absorbed but high-sensitivity |
| sensitivity_steering_correlation | H-SENS | FAIL | Steering protocol broken (all effects=0) |
| decoder_overlap_alternative | H-DEC | FAIL | High-sens have HIGHER overlap (opposite) |

---

## Research Questions for iter_011

**RQ1 (Layer 4 Validation)**: Does Sensitivity Floor hold at layer 4 (non-saturated)?

**RQ2 (Q4 Paradox Resolution)**: Why are Q4 features absorbed but high-sensitivity?

**RQ3 (U-Shape)**: Is there a U-shaped relationship between absorption and sensitivity at layer 4?

**RQ4 (Steering Debug)**: Can we fix the steering protocol to get non-zero effects?

---

## Setup

### Models and SAEs

| Component | Choice | Reason |
|-----------|--------|--------|
| Model | GPT-2 Small | Standard interpretability benchmark |
| SAE release | `gpt2-small-res-jb` (SAELens) | Pre-trained, well-characterized |
| Target layer | **Layer 4** (primary) | Only non-saturated layer from iter_010 |
| d_sae | 768 * 8 = 6144 | Standard expansion factor |

### Layer-wise UAS Protocol

1. Select 50 features with sufficient activation at layer 4 (tau_fs = 0.03)
2. Apply Chanin absorption protocol at layer 4
3. Apply Tian sensitivity protocol on same features
4. Compute quadrant classification

---

## Hypotheses and Falsification

### Primary: Sensitivity Floor at Layer 4

| ID | Hypothesis | Falsification Criterion |
|----|------------|------------------------|
| H-SF1-L4 | Q2+Q4 < 5% at layer 4 with N=50 | Q2+Q4 > 10% |
| H-SF2-L4 | U-shape: a > 0 | a <= 0 (linear or inverted-U) |
| H-SF3-L4 | Frequency mediates r(absorption, sensitivity) | Partial r > 0.3 after frequency control |

### Secondary: Q4 Paradox Investigation

| ID | Hypothesis | Falsification Criterion |
|----|------------|------------------------|
| H-Q4-PARADOX | Q4 features are absorbed but NOT via specificity loss | Q4 features have normal specificity |
| H-Q4-MECHANISM | Q4 features use different absorption mechanism | Q4 features absorbed identically to Q1 |

### Tertiary: Steering Protocol Debug

| ID | Hypothesis | Falsification Criterion |
|----|------------|------------------------|
| H-STEER | Steering effect > 0 at beta=5 or 10 | All steering effects = 0 |

---

## Experiment Tasks

### Phase 1: Diagnostic Pilots (Session 1, ~50 min total)

#### Task 1: pilot_layer4_sf1_quadrants (~20 min)
**Objective**: Test H-SF1 at layer 4 (N=50 features).

**Procedure**:
1. Load GPT-2 Small SAE for layer 4
2. Select 50 features with sufficient activation
3. Apply Chanin absorption protocol at layer 4 (200 tokens per feature)
4. Apply Tian sensitivity protocol (paraphrase AUC)
5. Classify into quadrants
6. Fit quadratic model for U-shape

**GPU**: 1× A100, ~20 min
**Pass criteria**: Q2+Q4 < 10%; U-shape coefficient a > 0
**Output**: `exp/results/pilots/layer4_sf1_quadrants.json`
**Candidate ID**: cand_sensitivity_floor

#### Task 2: pilot_q4_paradox_investigation (~15 min)
**Objective**: Investigate WHY Q4 features are absorbed but high-sensitivity.

**Procedure**:
1. Load features 10236 and 6768 from layer 8
2. Compute specificity metrics: decoder cosine similarity to neighbors, specificity score
3. Compare to Q1 features (absorbed + low-sensitivity)
4. Test: Do Q4 features have UNUSUAL specificity despite absorption?

**GPU**: 1× A100, ~15 min
**Pass criteria**: Q4 specificity differs from Q1
**Output**: `exp/results/pilots/q4_paradox_investigation.json`
**Candidate ID**: cand_sensitivity_floor

#### Task 3: pilot_steering_debug (~15 min)
**Objective**: Debug steering protocol to get non-zero effects.

**Procedure**:
1. Select 10 features with varying sensitivity scores
2. Test steering at beta ∈ {5, 10, 20}
3. Use stronger steering (beta=20) or different intervention point
4. Verify: Is baseline accuracy changing when steering is applied?

**GPU**: 1× A100, ~15 min
**Pass criteria**: At least one feature shows steering effect ≠ 0
**Output**: `exp/results/pilots/steering_debug.json`
**Candidate ID**: cand_sensitivity_first

---

## Evaluation Matrix

| Hypothesis | Metric | Pass Criterion | Falsification |
|-----------|--------|----------------|---------------|
| H-SF1-L4 (emptiness) | Q2+Q4 fraction | < 10% | > 10% |
| H-SF2-L4 (U-shape) | Quadratic coeff a | a > 0 | a <= 0 |
| H-Q4-PARADOX | Q4 vs Q1 specificity | Different mechanism | Same mechanism |
| H-STEER | Steering effect | effect ≠ 0 | All effects = 0 |

---

## Pilot Decision Tree

```
pilot_layer4_sf1_quadrants:
  - Q2+Q4 < 10% AND a > 0 → Sensitivity Floor confirmed at layer 4 → proceed to full
  - Q2+Q4 > 10% → Sensitivity Floor falsified at layer 4 → pivot to sensitivity-first
  - a <= 0 → No U-shape → sensitivity floor mechanism revised

pilot_q4_paradox_investigation:
  - Q4 specificity differs from Q1 → Paradox explained → SF mechanism needs revision
  - Q4 specificity same as Q1 → New mechanism needed → pivot

pilot_steering_debug:
  - steering effect ≠ 0 → Protocol fixed → can test sensitivity-steering correlation
  - All effects = 0 → Steering cannot validate sensitivity → pivot to other metrics
```

---

## Expected Visualizations

1. **Figure 1**: Layer 4 quadrant scatter
   - X: absorption score (UAS)
   - Y: sensitivity score (paraphrase AUC)
   - Q2+Q4 highlighted
   - Paper section: experiments

2. **Figure 2**: Q4 paradox mechanism
   - Bar chart: specificity scores for Q4 vs Q1 vs Q3
   - Paper section: experiments

3. **Figure 3**: Steering debug results
   - Line plot: steering effect vs beta for different features
   - Paper section: experiments

---

## Shared Resources

| Resource | Path | Notes |
|----------|------|-------|
| GPT-2 Small SAEs | `gpt2-small-res-jb` (SAELens) | Layer 4 (primary), layer 8 (Q4 features) |
| iter_010 Q4 features | sf1_large pilot data | Features 10236, 6768 |
| WikiText-103 | `wikitext-103-v1` (HuggingFace) | ~5K tokens for activation |

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Layer 4 also saturates | Low | Layer 4 std=0.48 showed variance |
| Q4 paradox remains unexplained | Medium | May need new theoretical framework |
| Steering cannot be fixed | Medium | Use alternative validation method |
| Sensitivity Floor falsified | Medium | Pivot to sensitivity-first or pivot to new candidate |

---

## Iteration Context

**Iteration 11** after iter_010 critical findings:
- Layer 4: UAS std = 0.48 (best), layer 8: std = 0.0 (saturated)
- Q4 features (10236, 6768): absorbed but high-sensitivity (0.91, 0.80)
- Steering protocol broken: all effects = 0.0
- H-DEC failed: high-sens have HIGHER decoder overlap

This iteration prioritizes:
1. Testing Sensitivity Floor at layer 4 (not saturated)
2. Resolving the Q4 paradox (absorbed but sensitive)
3. Debugging steering protocol

---

## References

| Reference | Citation |
|-----------|---------|
| Chanin 2024 | arXiv:2409.14507 |
| Tian 2025 | arXiv:2509.23717 |
| Korznikov 2026 | arXiv:2602.14111 |
| SAELens | jbloomAus/SAELens |

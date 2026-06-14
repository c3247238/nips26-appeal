# Experiment Methodology: Feature Absorption - Focus on Safety and Causal Validation

## Research Context

**Iteration**: 4 (post-pilot synthesis with empirical evidence)
**Front-runner**: cand_p1 (with revised mechanism interpretation)
**Status**: Post-pilot, planning next experiments

**Pilot Summary**:
- H1: PASSED (strong evidence: d=8.94, p<10^-133)
- H2: FAILED (wrong direction: rho=+0.17 instead of -0.3)
- H3: BROKEN (steering not applied: baseline=steered mean)
- H_Safe: NOT TESTED
- H_Mech: NOT TESTED

**Planning Focus**:
1. H_Safe: Safety-critical feature absorption (highest novelty, no training)
2. H3_fix: Debug steering implementation
3. Multi-seed validation: Statistical robustness for H1

---

## Setup

### Synthetic Hierarchy Generation (Existing - Reuse from Pilot)
- 3-level hierarchy: parent (L0=1) → 2 children (L0=2) → 4 grandchildren per child
- Parent-children cosine similarity: 0.67
- Grandchildren pairwise orthogonality: ~0
- 5 hierarchies per seed
- Pilot: 100 samples; Full: 10,000 samples per seed

### SAE Architectures
| Config | d_model | d_sae | L0 Target | Expansion |
|--------|---------|-------|-----------|----------|
| Config A | 512 | 4096 | 16 | 8x |
| Config B | 512 | 4096 | 32 | 8x |
| Config C | 512 | 4096 | 64 | 8x |

- Framework: SAELens TopK SAE
- Seeds: 42, 43, 44, 45, 46 (5 seeds per config for multi-seed validation)
- Training: 20,000 steps, lr=1e-3, batch_size=4096

### Korznikov-Style Baselines
1. **Random Decoder**: Xavier-initialized SAE decoder, no training
2. **Shuffled Features**: Same activations, permuted feature index assignments
3. **Permuted Encoder**: Trained SAE with shuffled encoder weights

---

## Priority 1: H_Safe - Safety-Critical Feature Absorption

### Objective
Test whether safety-critical features (deception, jailbreak, harm, manipulation) show elevated absorption rates compared to matched non-safety features in real Gemma Scope SAEs.

### Hypothesis
H_Safe: Safety-critical features have higher absorption rates than matched non-safety features (Mann-Whitney p < 0.05)

### Methodology (No Training Required)
1. Load pretrained Gemma Scope SAEs from HuggingFace
2. Select 20 safety-relevant features from layer 12 via Neuronpedia annotations
3. Match 20 non-safety features by activation frequency and layer
4. Measure absorption via overlap method (top-k child feature overlap)
5. Statistical comparison: Mann-Whitney U test

### Implementation Notes
```python
# Load Gemma Scope SAE
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release="gemma-2b-res",
    sae_id="blocks.12.hook_resid_pre",
    device="cuda"
)

# Safety features identified via Neuronpedia annotations
# Match non-safety features by frequency quartile
```

### Pass Criteria
- Mann-Whitney p < 0.05: H_Safe PASS
- p >= 0.05: Document as null result (still valuable contribution)

---

## Priority 2: H3_fix - Steering Intervention Validation

### Problem from Pilot
Baseline = Steered mean (37.445), indicating steering was NOT applied. t-statistic = NaN.

### Required Debug Steps
1. **Verify steering is applied**: Compute `||steered_activations - baseline_activations|| > 0`
2. **Check alpha scaling**: Confirm alpha produces proportional changes
3. **Validate on simpler case**: Test steering on clearly non-absorbed features first
4. **Alternative approach**: Measure logit-level steering effect (Basu et al. 2026)

### Revised Steering Protocol
```python
# Verify steering actually changes activations
def verify_steering(sae, feature_idx, parent_direction, alpha):
    """Verify steering produces measurable change."""
    baseline_act = sae.encode(test_activations)  # shape: [batch, seq, d_sae]
    steered_act = apply_steering(sae, test_activations, feature_idx, parent_direction, alpha)

    delta = (steered_act - baseline_act).norm()
    assert delta > 1e-6, f"Steering had no effect! delta={delta}"
    return delta
```

### Alternative: Logit-Level Steering (Basu et al. 2026)
If feature-level steering remains problematic:
1. Identify steering vector from decoder weights
2. Apply at residual stream level
3. Measure logit change for target concept
4. Compare absorbed vs non-absorbed feature steering effects

### Pass Criteria
- `||steered - baseline|| > 0`: Steering is applied
- Paired t-test p < 0.01 for absorbed vs non-absorbed steering effect
- If steering broken after debug: Document as negative result (no causal absorption evidence)

---

## Priority 3: Multi-Seed Validation for H1

### Objective
Address zero-variance concern (std=0.0 for trained SAE in pilot). Validate with multiple seeds.

### Design
- 3 seeds: 42, 43, 44 (quick validation)
- L0=32 (single config for speed)
- Measure: Multi-child proportional absorption rate

### Expected Outcome
- Trained SAE: absorption ~0.50 with non-zero variance across seeds
- Random decoder: absorption ~0.06 with higher variance
- If trained SAE std=0 across seeds: document as deterministic geometric result

### Pass Criteria
- Non-zero variance in absorption across seeds
- All 3 seeds show absorption > 0.3 for trained SAE
- At least 2/3 seeds show absorption < 0.2 for random decoder

---

## Secondary: H_Mech - Geometric vs Learned Decomposition

### Objective
2x2 factorial to decompose absorption into geometric vs learned components.

### Design
| Condition | Encoder | Decoder |
|-----------|---------|---------|
| A | Random | Random |
| B | Trained | Random |
| C | Random | Trained |
| D | Trained | Trained |

### Expected
- Condition C ≈ D ≈ 0.48 (geometric dominates)
- Condition A ≈ 0.06 (no structure)
- If D >> C: absorption is primarily learned

### Implementation
```python
# Condition C: Random encoder + Trained decoder
trained_decoder = trained_sae.W_dec.clone()
random_encoder = initialize_random_encoder(d_in, d_sae)
condition_c_sae = SAEModel(random_encoder, trained_decoder)

# Measure absorption for all 4 conditions
```

---

## Metrics

### Primary: Multi-Child Proportional Absorption Rate
```
absorption_k5 = parent_activation_after_ablating_top_k_children / parent_activation_before
```

### Secondary Metrics
- Steering effect size: ||steered - baseline||
- Safety absorption difference: mean(safety) - mean(non_safety)
- Multi-seed variance: std across seeds

---

## Evaluation Plan

### Phase 1: Quick Pilots (Priority Order)
| Task | Duration | Focus |
|------|----------|-------|
| H_Safe pilot | 20 min | Safety-critical features |
| H3_fix pilot | 30 min | Debug steering |
| Multi-seed pilot | 30 min | 3 seeds validation |

### Phase 2: Full Experiments (if Phase 1 succeeds)
| Task | Duration | Focus |
|------|----------|-------|
| H_Mech full | 45 min | 2x2 factorial |
| H_Safe full | 30 min | Extended safety analysis |
| H3 full | 45 min | Complete steering validation |

---

## Expected Visualizations

1. **Figure 1**: Safety vs non-safety absorption (H_Safe) - bar chart
2. **Figure 2**: Steering effect by feature type (H3_fix) - bar chart
3. **Figure 3**: Multi-seed absorption distribution (H1 validation) - box plot
4. **Figure 4**: 2x2 factorial decomposition (H_Mech) - grouped bar chart

---

## Time Budget

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| H_Safe pilot | 1 | ~20 min |
| H3_fix pilot | 1 | ~30 min |
| Multi-seed validation | 1 | ~30 min |
| H_Mech (optional) | 1 | ~45 min |
| **Total** | 3-4 | **~1.5 hours** |

All tasks within 1-hour budget per CLAUDE.md guidelines.

---

## Shared Resources
- Synthetic hierarchies: reuse from pilot
- Gemma Scope SAEs: downloaded from HuggingFace
- Pilot data: reuse for diagnostic checks
- Existing trained SAEs: reuse for multi-seed validation

---

## Negative Results Documentation

### H2: Frequency-Absorption Correlation (ARCHIVED)
- Finding: rho = +0.171 (positive), contradicts hypothesis
- Interpretation: Higher-frequency features are MORE absorbed, not less
- Three-regime model: rare=ignored, moderate=absorbed, frequent=preserved

### H3 (if steering remains broken)
- Finding: Steering produces no measurable effect
- Interpretation: Cannot make causal claims about absorption impact
- Action: Document as negative result, no H3 in paper

---

## References
- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
- Tang et al. (2025). Theoretical Foundation of SDL. arXiv:2512.05534

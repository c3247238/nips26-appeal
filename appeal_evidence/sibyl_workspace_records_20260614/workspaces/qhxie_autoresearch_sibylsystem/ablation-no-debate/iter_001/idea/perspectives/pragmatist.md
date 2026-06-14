# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **DSPA (Dynamic SAE Steering for Preference Alignment)** - arXiv:2603.21461 (March 2026)
   - Novel inference-time steering method using SAE features
   - Shows how to compute "conditional-difference map" linking features to steering vectors
   - **Directly relevant** for H3 steering implementation - provides methodology for feature-level steering

2. **Feature-Guided SAE Steering** - arXiv:2511.00029 (Bhargav & Zhu, 2025)
   - Uses contrasting prompts for feature selection in safety-utility tradeoff
   - Provides feature selection methodology for H_Safe
   - Shows practical steering implementation details

3. **SAELens** - https://github.com/jbloomAus/SAELens
   - Production-ready library for SAE training/analysis
   - Gemma Scope support with pretrained weights
   - Feature visualization and evaluation tools
   - **Code exists**: Yes, well-maintained

4. **sae-spelling** (lasr-spelling/sae-spelling)
   - MIT licensed, contains absorption measurement code
   - Foundation for overlap/ablation methods
   - **Code exists**: Yes, needs extension for multi-child proportional absorption

5. **Hierarchical SAE** - arXiv:2506.01197 (Muchane et al., 2025)
   - Explicitly models semantic hierarchy in SAE architecture
   - Related to the phenomenon being studied
   - Shows what a "solution" to absorption could look like

6. **Improving steering vectors by targeting SAE features** - arXiv:2411.02193 (Chalnev et al., 2024)
   - Documents cases where steering produces degradation without interpretable change
   - **Important warning** for H3: steering may not always improve sensitivity

7. **SAEs Are Good for Steering** - arXiv:2505.20063 (Arad et al., 2025)
   - Distinguishes input vs output features
   - Filtering low-output-score features improves steering
   - Relevant for H3 interpretation

### Landscape Summary

The SAE field has converged on several practical insights:

1. **Absorption is real but measurement-dependent**: The overlap method (0.25 delta in pilot) works better than ablation-based measurement (saturates at 1.0)

2. **Steering is actionable but fragile**: Recent work shows steering effectiveness depends on feature selection quality and output scores. DSPA and Bhargav provide working implementations.

3. **Gemma Scope is the standard platform**: Pretrained SAEs on Gemma 2 with 27 layers are publicly available and well-documented for safety-critical feature analysis.

4. **Engineering reality**: The pilot code in this workspace shows that synthetic hierarchy generation and SAE training work. The bottleneck is measurement methodology and multi-seed validation.

---

## Phase 2: Initial Candidates

### Candidate A: Multi-Child Proportional Absorption with Refined Measurement

- **Hypothesis**: Multi-child proportional absorption (k=5 top children) will differentiate trained SAE from baselines where single-child ablation fails
- **Implementation sketch**: Extend current pilot code to ablating top-k children collectively; measure proportional contribution variance
- **Simplest version**: Just fix the ablation methodology on existing synthetic hierarchy data - no new training
- **Time estimate**: 20 minutes (measurement only)
- **Reusable components**: Existing pilot code, existing synthetic data
- **Verdict**: STRONG candidate for quick signal

### Candidate B: Gemma Scope H_Safe Pilot

- **Hypothesis**: Safety-critical features on Gemma Scope have elevated absorption compared to matched non-safety features
- **Implementation sketch**: Use SAELens to load Gemma Scope SAEs; use Neuronpedia for feature annotation; compute overlap-based absorption
- **Simplest version**: Select 10 features from Neuronpedia (5 safety, 5 non-safety) and compute absorption via overlap method
- **Time estimate**: 30-45 minutes (no training needed, pretrained SAEs only)
- **Reusable components**: SAELens Gemma Scope support, Neuronpedia annotations
- **Verdict**: MODERATE - highest novelty but requires human annotation effort

### Candidate C: Steering Intervention on Synthetic Features (H3)

- **Hypothesis**: Steering absorbed features toward parent directions improves sensitivity
- **Implementation sketch**: Use DSPA-style conditional-difference map on synthetic features; test alpha values {0.05, 0.1, 0.15, 0.2, 0.25}
- **Simplest version**: Implement steering on pilot data (100 samples, 3-level hierarchy)
- **Time estimate**: 30 minutes
- **Reusable components**: Existing pilot code, DSPA methodology from arXiv:2603.21461
- **Verdict**: STRONG - provides causal evidence, directly addresses validation feedback

---

## Phase 3: Self-Critique

### Against Candidate A (Multi-Child Proportional Absorption)

- **Implementation reality check**: The proportional variance approach is theoretically sound but has not been validated in prior work. Could produce non-interpretable results.
- **Reproducibility attack**: If proportional variance doesn't correlate with absorption, the metric fails. Need a backup metric ready.
- **Baseline sanity check**: Random baseline may still show high proportional variance if children happen to span the parent subspace.
- **Scope attack**: The fix addresses only the measurement methodology, not the core phenomenon understanding.
- **Verdict**: MODERATE - necessary but not sufficient for the overall research story.

### Against Candidate B (Gemma Scope H_Safe)

- **Implementation reality check**: Human annotation of 40 features is time-consuming. Need to validate that safety-critical features are reliably identified.
- **Reproducibility attack**: "Safety-critical" is subjective. Different annotators may label features differently.
- **Baseline sanity check**: Non-safety features matched by frequency/layer may not be appropriate controls.
- **Scope attack**: Limited to Gemma Scope - generalizability to other models unclear.
- **Verdict**: STRONG NOVELTY, WEAK ENGINEERING - needs careful annotation protocol.

### Against Candidate C (Steering Intervention)

- **Implementation reality check**: DSPA paper (arXiv:2603.21461) published March 2026 provides direct methodology. Can adapt their steering implementation.
- **Reproducibility attack**: Steering is sensitive to alpha values and feature selection. Need to document sensitivity analysis.
- **Baseline sanity check**: Need non-absorbed control features for comparison. The synthetic hierarchy provides ground truth.
- **Scope attack**: Steering on synthetic features may not transfer to real Gemma Scope features.
- **Verdict**: STRONG - provides the causal validation that pilot is missing.

---

## Phase 4: Refinement

### Dropped Ideas
- **Asymmetry index (H2)**: Pilot data shows trained SAE (0.487) approx random baseline (0.471). This metric is fundamentally broken for this task. Do not pursue.
- **Ablation-based single-child absorption (H1)**: Already falsified in pilot - both conditions saturate at 1.0.

### Strengthened Ideas

**Candidate C (Steering)**: Now highest priority because:
1. Provides causal evidence (pilot lacks this entirely)
2. DSPA methodology is published and reproducible
3. Can use existing synthetic data without new training
4. Addresses reviewer concern: "so what if absorption exists?"

**Candidate A (Measurement fix)**: Secondary priority because:
1. Quick to implement (20 min)
2. Validates whether the proportional approach works
3. Provides publication-ready methodology

**Candidate B (H_Safe)**: Tertiary but valuable because:
1. Highest novelty (9/10 per novelty report)
2. No Korznikov collision
3. No new training required (Gemma Scope pretrained)

### Selected Front-Runner Sequence

1. **Candidate C first**: Steering intervention (30 min) - causal evidence
2. **Candidate A second**: Multi-child measurement (20 min) - validates methodology
3. **Candidate B third**: H_Safe on Gemma Scope (45 min) - high-value backup

---

## Phase 5: Final Proposal

### Title
**Causal Validation of Feature Absorption in Sparse Autoencoders via Steering Intervention**

### Hypothesis
Feature absorption in SAEs is causally responsible for reduced feature sensitivity: steering absorbed features toward parent feature directions will restore sensitivity comparable to non-absorbed features.

### Motivation
The pilot experiment established that absorption is a learned phenomenon (trained SAE: 0.50 vs random baseline: 0.25) but did not establish causation. Critics could argue absorption is merely correlational - features that fail to fire may do so for reasons unrelated to child feature absorption. This proposal provides causal validation through steering intervention, following DSPA's published methodology.

### Method

**Step 1: Identify Absorbed Features in Trained SAE**
- Use overlap method from pilot: absorption_rate = overlap(parent_topk, child_topk) / k
- Features with absorption_rate > 0.4 are "absorbed"
- Match with random baseline features of similar frequency

**Step 2: Compute Steering Vector (DSPA-style)**
- For absorbed feature f_abs: reconstruct parent direction from children's decoder subspace
- parent_dir = sum_i (proj(decoder[child_i], parent_direction))
- Steering vector = alpha * parent_dir_normalized

**Step 3: Steering Intervention**
- Add steering vector to activations: x_steered = x + alpha * parent_dir
- Test alpha in {0.05, 0.1, 0.15, 0.2, 0.25}
- Measure sensitivity on held-out text (same text where parent should fire)

**Step 4: Evaluate**
- Compare sensitivity_before vs sensitivity_after for absorbed features
- Compare with non-absorbed control features (should show no improvement)
- Statistical test: paired t-test, significance p < 0.01

### Simplest Version
Use existing pilot data (100 samples, 3 seeds, L0=32). Implement steering on synthetic hierarchy features. No Gemma Scope training needed. Estimated time: 30 minutes.

### Baselines
1. **Absorbed features without steering**: Expected low sensitivity (~0.2)
2. **Non-absorbed features without steering**: Expected baseline sensitivity (~0.6)
3. **Non-absorbed features with steering**: Expected no change (~0.6)
4. **Absorbed features with steering**: Expected improvement to ~0.5-0.6 (causal evidence)

### Experimental Plan
| Phase | Task | Duration | Dependency |
|-------|------|----------|------------|
| 1 | Implement steering on pilot data | 15 min | None |
| 2 | Run 3 seeds (42, 43, 44) | 10 min | Phase 1 |
| 3 | Statistical analysis | 5 min | Phase 2 |
| 4 | H_Safe pilot on Gemma Scope | 45 min | Phase 3 |
| 5 | Full experiment (5 seeds, 3 L0) | 60 min | Phase 4 |

### Resource Estimate
- **GPU-hours**: ~2 hours total (mostly existing pilot data)
- **Wall-clock time**: ~2.5 hours
- **Model sizes**: GPT-2 scale synthetic (d_model=128)
- **Gemma Scope**: Pretrained weights, no training

### Risk Assessment
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Steering destabilizes reconstruction | Medium | Monitor explained variance; use small alpha |
| Steering equally helps absorbed/non-absorbed | Medium | Would falsify H3; document as negative result |
| Gemma Scope features don't match annotation | Low | Use Neuronpedia community annotations |

### Novelty Claim
This is the **first causal validation of absorption's impact on feature sensitivity**. Prior work documents absorption; this work tests whether correcting absorption causally improves downstream reliability. Combined with the random baseline comparison (Korznikov framework), this provides both sanity check and causal evidence - a two-pronged contribution that survives the novelty review.

### Differentiation from Korznikov et al.
- Korznikov tests whether SAEs beat baselines on general interpretability metrics
- This work tests absorption-specific metrics AND provides causal validation via steering
- The steering intervention is novel and directly actionable for practitioners

---

## Immediate Action Items (Next 45 minutes)

1. **H3 Steering Pilot** (15 min): Implement steering using DSPA methodology on existing pilot data
2. **Measurement Fix** (10 min): Implement multi-child proportional absorption as backup metric
3. **H_Safe Setup** (20 min): List Gemma Scope features from Neuronpedia for safety-critical annotation

## Confidence: 0.75

The steering intervention is the highest-value next step because it directly addresses the gap in the current proposal (no causal evidence) and can reuse existing pilot infrastructure.

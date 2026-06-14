# Alternative Research Directions

These are backup ideas for potential pivot if the current direction proves unfruitful.

---

## Alternative 1: Metric-Centric Paper (Primary Backup)

**Core thesis**: The absorption metric itself (Af with random dictionary validation) is the primary contribution, not the empirical characterization. Establishes Af as a standard metric for SAE quality assessment.

**Novelty**: Establishes Af as a standard metric for SAE quality assessment, with random dictionary controls as best practice. Differs from prior work (polysemy, feature correlation) by using RVE-based absorption.

**Strengths**:
- Avoids the negative result problem (H1-H4 partially falsified/untested)
- Highly reproducible and actionable for the SAE community
- Validation framework can be applied to any SAE
- No new experiments needed -- CPU analysis only

**Weaknesses**:
- Lower ceiling than empirical characterization
- Reviewers may ask "why not just use existing superposition metrics?"
- Scope is more appropriate for workshop than main conference

**Experiments needed**: H2 and H6 analyses to validate/invalidate metric sensitivity.

---

## Alternative 2: SAE Training Curriculum

**Core thesis**: Train SAEs with controlled sparsity to find the optimal lambda that balances sparsity against absorption. Directly addresses H3's L0/L1 conflation.

**Novelty**: First systematic study of absorption as a function of training hyperparameters, not just post-hoc analysis.

**Strengths**:
- Directly addresses H3 (the L0/L1 conflation is resolved)
- Generates new data, not just analysis of existing data
- Practical guidance for SAE training
- Could yield novel engineering recommendations

**Weaknesses**:
- Requires GPU training (~6 hours total)
- Still may not address H4 (circuit faithfulness)

**Experiments needed**:
- Train 3 mini-SAEs with L1=[4e-5, 8e-5, 1.6e-4] on 10M tokens
- Evaluate absorption at each lambda
- Analyze L0 vs lambda relationship

---

## Alternative 3: GemmaScope Multi-Layer SAE Generalization

**Core thesis**: Test whether the layer-dependent absorption pattern generalizes across different SAE architectures (GemmaScope) and model sizes. Addresses GPT2-small specificity concern.

**Novelty**: First cross-architecture validation of layer-dependent absorption patterns.

**Strengths**:
- Directly addresses "may be GPT2-small specific" concern
- GemmaScope SAEs are publicly available with consistent methodology
- Could strengthen paper's claims significantly if pattern replicates

**Weaknesses**:
- Requires running experiments on different SAEs (~6 GPU hours)
- Still does not address H4 circuit faithfulness question

**Experiments needed**:
- Run H1 analysis on GemmaScope SAE at layers 0/4/8
- Confirm or falsify inverted-U pattern on different architecture

---

## Alternative 4: Circuit Faithfulness Redesign (Low Priority)

**Core thesis**: The correctly designed H4 experiment (comparing layer 4 vs layer 8 SAE on circuit patching) is the key to a strong paper if it yields positive results.

**Novelty**: First study connecting layer-dependent absorption to circuit-level computations.

**Strengths**:
- Could yield the strongest paper if positive: layer-dependence + circuit impact
- Addresses the most impactful question: does absorption affect interpretability reliability?

**Weaknesses**:
- H4 was uninformative in pilot -- redesign risk is high
- May still yield negative results even with correct experiment
- Requires GPU time (~3 hours)

**Experiments needed**:
- Run circuit patching with FULL SAE (not subset) at layer 4 vs layer 8
- Compare faithfulness between high-absorption (layer 4) vs low-absorption (layer 8) SAEs

---

## Pivot Decision Tree

```
H2 confirmed (negative correlation) + H6 artifact confirmed -> Continue front_runner (layer-dependence story)
H2 falsified + H6 artifact -> Pivot to Alternative 1 (metric-centric)
H2 confirmed + H6 shows genuine absorption -> Continue with additional analysis
H4 properly tested + positive result -> Strong paper with layer-dependence + circuit story
H4 properly tested + negative result -> Paper is metric + layer-dependence only
```

**Current status**: Critical path is H2+H6 on layer 4 data (NOT layer 8 as incorrectly attempted before). Both are CPU-only, ~4 hours total. **STAGNATION RISK: These analyses have been pending for 11 iterations without execution.**

---

## Dropped Alternatives

### "Activation Patching via Absorbed Features" -- DO NOT PURSUE

This direction is blocked because:
1. H4 is uninformative (not falsified -- the comparison was uninterpretable)
2. The correctly designed experiment (layer comparison) was never conducted
3. Without H4, there is no evidence that absorption affects circuit-level computations

**Recommendation**: Pursue only after completing H2+H6 and only if resources allow for a proper H4 redesign.
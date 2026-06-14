# Methodology: Activation Patching Validation for Feature Absorption

## Research Context

This ablation study (iteration 6) has established:
- **H1-H4**: Null results - absorption does not degrade steering/probing downstream
- **H5**: Robust finding - precision=1.0 universally, recall varies
- **H6**: Falsified - decoder correlation graph predicts zero absorption pairs
- **H7**: Trained SAE (0.034) < Random SAE (0.278) - metric sensitive to structure
- **H9/H10**: Co-occurrence tautological; random baseline confirms H7

**Critical gap (from evolution lessons)**: The 9 persistent core words (e.g., 'eight', 'lower', 'liked', 'offer', 'often') are claimed as genuine hierarchy-driven absorption, but **validation by activation patching was never performed**. This is the ONLY experiment that can distinguish 'metric miscalibrated' from 'JumpReLU genuinely has minimal absorption.'

## Objectives

1. **Primary**: Validate whether the 9 persistent core words represent genuine hierarchy-driven absorption via activation patching
2. **Secondary**: Compare trained SAE vs random SAE activation patching behavior
3. **Exploratory**: Investigate high dead feature ratio (89-99%) impact on results

## Hypothesis

**H-new (Activation Patching Validation)**:
- If absorption is genuine hierarchy-driven: suppressing child feature will recover parent feature activity
- If absorption is metric artifact: suppressing child feature will NOT recover parent (parent never fired)
- H7 suggests metric artifact: trained SAE has 8x less absorption than random

## Experimental Design

### Experiment 1: Activation Patching on Core Words (Pilot)

**Method**: For each of 9 persistent core words at layer 8:
1. Identify the child feature (absorbing) and parent feature (being absorbed)
2. Run baseline: prompt with core word, measure parent and child activations
3. Run patching: suppress child feature activation to zero, measure parent recovery
4. Calculate recovery rate = (parent_with_patch - parent_baseline) / (parent_expected - parent_baseline)

**Prompts** (from prior analysis):
- 'eight': e.g., "The sequence is eight, seven, six"
- 'lower': e.g., "The score is lower than expected"
- 'liked': e.g., "She liked the movie because"
- 'offer': e.g., "They offer free shipping on"
- 'often': e.g., "Things that happen often include"

**Pass criteria**:
- Recovery rate > 0.5: genuine absorption (parent recovers when child suppressed)
- Recovery rate < 0.2: metric artifact (parent doesn't recover)
- Pilot: n=9 words, seed=42, timeout=15min

### Experiment 2: Random SAE Baseline Activation Patching

**Method**: Same as Experiment 1, but using random SAE (frozen orthonormal decoder, random encoder)
- Compare recovery rates: trained SAE vs random SAE
- Random SAE should show LOWER recovery if H7 interpretation is correct

**Pass criteria**:
- Trained SAE recovery > Random SAE recovery: training optimizes away spurious absorption
- Pilot: n=9 words, seed=42, timeout=15min

### Experiment 3: Architecture Comparison (JumpReLU vs TopK)

**Critical issue from lessons**: E1 comparison was confounded by:
- JumpReLU: pretrained on Gemma data, d_SAE=24,576
- TopK: trained from scratch on OpenWebText, d_SAE=16,384

**Remediation plan**:
- Use SAME training data and comparable dictionary sizes
- Or: use published results with matched configurations
- Focus on: activation patching recovery rate as primary metric

**This is exploratory** - may not fit within time budget

## Metrics

| Metric | Description | Pass Threshold |
|--------|-------------|----------------|
| Recovery Rate | % parent activation recovered when child suppressed | > 0.5 = genuine |
| False Positive Rate | % of absorptions that don't recover | < 0.3 |
| Comparison | Trained vs Random recovery difference | Significant if p < 0.05 |

## Evaluation Benchmarks

- GPT-2 Small SAE (gpt2-small-res-jb, 24K latents, layer 8)
- Random SAE baseline (frozen orthonormal decoder, random encoder)
- Compare JumpReLU (if available) and TopK architectures

## Expected Visualizations

- **Figure 1**: Activation patching recovery rates (bar chart, 9 words)
- **Figure 2**: Trained SAE vs Random SAE recovery comparison
- **Figure 3**: L0 distribution vs recovery rate (does dead feature ratio matter?)
- **Table 1**: Summary of activation patching results

## Technical Considerations

### From SAELens Skill
- Use `steering_hook` for activation patching
- Monitor dead feature ratio impact on results
- Use TransformerLens for model loading

### From lm-evaluation-harness (if needed)
- Not directly applicable - this is causal intervention, not benchmark evaluation

### Dead Feature Analysis
- 89-99% dead feature ratio is concerning
- Plan: stratify analysis by feature L0 (active vs dead features)
- If results differ: high dead ratio may invalidate some conclusions

## Shared Resources

- GPT-2 Small model: transformer-lens pretrained
- gpt2-small-res-jb SAE: from SAELens release
- Random SAE: constructed from orthonormal decoder + random encoder
- Existing absorption results: `/exp/results/full/`

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| Recovery rates inconclusive | Medium | Medium | Report as "inconclusive but informative" |
| Dead features invalidate results | Medium | High | Stratify by L0; acknowledge limitation |
| Time overrun | Low | Medium | Focus on pilot (9 words), skip full sweep |
| GPU memory | Low | Low | Small model (GPT-2 Small), 1 GPU sufficient |

## Timeline

- Week 1: Activation patching pilot (9 core words) - 2 hours
- Week 1: Random SAE baseline comparison - 2 hours
- Week 2: Architecture comparison (if time permits) - 4 hours
- Week 2: Analysis and figure generation - 2 hours

**Total estimated: ~10 hours (within typical paper-writing timeline)**

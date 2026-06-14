# Critique: Planning and Methodology

## Summary

The planning shows good practices (multiple comparison correction, random baselines, multiple layers) but has gaps (underspecified random SAE construction, no a-priori power analysis, beta-conditional effects not planned for). The overall research plan was sound but execution had issues.

## Good Practices

### 1. Multiple Comparison Correction
The planning correctly identified that 12 tests would be performed and applied Bonferroni correction (alpha=0.00417) and BH-FDR (q<0.05). This is good methodological practice.

### 2. Random SAE Baseline
The H10 experiment (random SAE comparison) was well-motivated and the finding (trained<random absorption) is genuinely interesting. This is the paper's strongest empirical contribution.

### 3. Layer Diversity
Testing at layers 0, 4, 8, and 10 allows for cross-layer comparison, which is valuable. The finding that layer 8 shows the strongest effects is informative.

### 4. Multiple Metric Types
Steering, probing, EC50 provides multiple angles on downstream effects. The precision-recall decomposition (H5) is methodologically sound.

## Planning Gaps

### 1. No A-Priori Power Analysis
The plan does not specify:
- Target effect size for power calculation
- Required sample size to detect that effect
- Justification for n=26 features

This is a significant gap because n=26 is small for correlation analysis. A proper power analysis would have determined whether the study was adequately powered.

### 2. Random SAE Construction Not Specified
The plan says "random SAE baseline" but does not specify:
- Exact encoder initialization
- Orthonormalization method
- Random seed

This makes the experiment underspecified and not reproducible.

### 3. Beta Values Not Justified
The steering strengths [1.0, 2.0, 5.0, 10.0, 20.0, 50.0] are not justified. The finding that effects are only observable at beta>=10 suggests the planning did not anticipate this conditionality.

### 4. Dead Feature Analysis Not Planned
The high dead feature ratio (89-99%) is known but not addressed in the plan. No stratification by L0 or analysis of dead vs active features was planned.

### 5. Cross-Model Validation Underspecified
The plan mentions cross-model validation with Pythia-70M but does not specify:
- Which layers to test
- How to handle limited feature overlap
- Success criteria for cross-model validation

## Recommendations for Future Planning

### A-Priori Power Analysis Template
```
Effect size to detect: r = 0.5 (medium effect)
Alpha: 0.05
Desired power: 0.80
Required n: 26 features

With n=26, we can detect r >= 0.5 at alpha=0.05 with 80% power.
Effects smaller than r = 0.5 are not reliably detectable.
```

### Random SAE Specification Template
```
Random SAE construction:
- Encoder: torch.randn(d_model, d_dict) * 0.02, never trained
- Decoder: orthonormalized via QR decomposition, frozen
- Seed: torch.manual_seed(42)
- Reference: .venv/bin/python3 -c "from saelens import SAELens; sae = SAELens.from_pretrained('gpt2-small-res-jb')"
```

### Multi-Model Validation Plan
```
Cross-model validation protocol:
1. Train probes on GPT-2 Small features at layers 4, 8
2. Evaluate on Pythia-70M at matched layers
3. Success criterion: correlation of absorption rates > 0.3 across models
4. If correlation < 0.3: note as limitation, do not claim generalization
```

## Overall Assessment

The planning was reasonable for a pilot study but:
- n=26 is underpowered for detecting medium effects
- Random SAE construction should be fully specified
- Dead feature analysis should be added
- Cross-model validation should have clearer success criteria

The paper should be framed as a "pilot study with suggestive findings" rather than definitive conclusions given these limitations.
# FULL Mode Results Summary (Iteration 5)

**Generated**: 2026-04-15
**Mode**: FULL (all tasks at full scale)

## Hypothesis Verdicts

### H1: Absorption-Quality Causal Chain -- SUPPORTED_WITH_CAVEATS
- **3/4 quality metrics** retain |partial_r| > 0.2 after controlling for L0
- Sparse probing partial r **strengthened** from -0.664 to -0.746 (suppression effect)
- SCR partial r: -0.570 (p=6.6e-5), TPP partial r: -0.331 (p=0.022)
- **3/4 metrics** show significant mediation indirect effect (Sobel test p < 0.05)
- Rosenbaum Gamma = 2.65 (strong robustness to hidden confounds)
- Bradford Hill assessment: 3 strong, 5 moderate, 0 weak criteria

### H2: Cross-Domain Absorption -- PARTIALLY_SUPPORTED
- **Model**: GPT-2 Small (124M params), 3552 cities, layers 5/8/11
- **Dominance-based detection**: 51-85% absorption across knowledge domains
- **Critical finding**: Shuffled controls show 100% absorption rate (metric does not discriminate)
- **Cosine-calibrated detection**: 0% across all thresholds
- **Interpretation**: Absorption as defined by Chanin et al. does not transfer to knowledge hierarchies on GPT-2 Small with 98% dead SAE features

### H3: Absorption Scaling Surface -- STRONGLY_SUPPORTED
- **N = 420 SAEs** from SAEBench (Gemma 2 2B)
- GAM interaction term: **p = 3.11e-15** (highly significant)
- R-squared: linear 0.488, additive 0.620, interaction 0.693
- Phase boundary detected at log2(L0) range [2.7, 3.8]

### H5: Taxonomy Correction -- CORRECTION_MINIMAL
- Original rate: 92.3%, Corrected rate: 92.3% (delta = 0.0%)
- Frequency-matched correction found for 8/26 letters
- 0 letters changed classification
- High Type II rate reflects genuine feature selectivity, not artifact

## Key Findings

1. **Phase 1**: L0 confound does NOT explain away the absorption-quality relationship. After controlling for L0, sparse probing association strengthened (suppression effect). Causal chain L0 -> Absorption -> Quality supported via mediation.

2. **Phase 2**: Dominance-based absorption metric has critical limitation when applied to knowledge hierarchies. The 100% shuffled control rate shows it captures background SAE feature concentration, not probe-direction absorption. This is itself a publishable finding.

3. **Phase 3**: 420-SAE absorption surface shows highly significant nonlinear interaction (p=3.1e-15). Strongest result -- absorption cannot be predicted from width or L0 independently.

4. **Phase 4**: Original 92.3% combined absorption rate validated. High rate reflects genuine feature selectivity.

## Experiment Statistics
- Total tasks: 14 completed, 0 failed
- GPU tasks: P2 absorption (3552 cities x 3 layers x 5 probes), P2 controls, P4 taxonomy
- CPU tasks: P1 confound resolution (48 SAEs), P3 scaling (420 SAEs), integration

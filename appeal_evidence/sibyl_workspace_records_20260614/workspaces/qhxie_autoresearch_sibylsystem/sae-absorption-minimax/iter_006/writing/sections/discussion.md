# 5. Discussion

## 5.1 Absorption Does Not Degrade Steering

Our central finding is that absorption level does not predict steering effectiveness. Across all beta values tested (1, 3, 5, 10, 20), high-absorption and low-absorption features show statistically equivalent steering sensitivity. The aggregated $p$-value of 0.299 indicates no systematic relationship.

This finding has important implications:
- **For steering research**: Practitioners should not assume that high-absorption features are better or worse steering targets. Other factors (activation frequency, task relevance, feature clarity) may be more important for steering target selection.
- **For interpretability evaluation**: The Unsupervised Absorption Score (UAS) is useful for identifying absorbed features but not for predicting steering effectiveness.

## 5.2 Positive Correlation Between Absorption and Sensitivity

The pilot finding of $r = 0.59$ between absorption and sensitivity suggests these failure modes are **not independent** as hypothesized. Rather, features that are absorbed tend to also have low sensitivity.

Possible explanations for this positive correlation:
1. **Common cause**: Both failure modes may result from features being rare or geometrically unfavorable in ways that make them both harder to detect reliably and less useful for steering.
2. **Developmental coupling**: Features may become absorbed during training as a consequence of being low-sensitivity (rarely activating).
3. **Detection artifact**: The Chanin and Tian protocols may both be biased toward similar feature types, creating an artifactual correlation.

## 5.3 The Empty Q4 Quadrant

The complete absence of Q4 (low absorption + high sensitivity) features has profound implications:

1. **Trade-off hypothesis**: High-sensitivity features may be inherently more susceptible to absorption. Features that activate reliably may naturally subsume other features, causing absorption.

2. **Compound failure is common**: If Q4 is empty, then most features experience at least one failure mode. This could explain the Sanity Check finding: if most features are doubly-compromised (or at least singly-compromised), average causal validity approaches random.

3. **Theoretical revision needed**: The compound failure hypothesis must be revised. The four-quadrant model with a "best-case" Q4 does not match feature distribution.

## 5.4 Failed Protective Effect

The failure to replicate the protective effect of mutual coherence ($r = -0.786$ in pilot, $r = +0.36$ in replication) suggests:
1. The coherence-absorption relationship may be unstable across layers or feature subsets
2. The earlier pilot result may have been a false positive due to small sample size (28 valid features)
3. The relationship may depend on training configuration or SAE architecture

## 5.5 Implications for the Sanity Check

The Sanity Check finding (random baselines match SAEs) remains partially unexplained. Our findings suggest:
- Absorption alone does not explain Sanity Check (absorption does not degrade steering)
- Sensitivity may be more relevant (low-sensitivity features are more prevalent)
- The empty Q4 suggests compound failure is common, pulling average causal validity toward random

## 5.6 Limitations

1. **Sample size**: 43 features in the pilot study is modest. Larger samples are needed to confirm the absorption-sensitivity correlation.

2. **Single model and layer**: GPT-2 Small layer 8 may not generalize to larger models (Gemma-2B, Llama) or different layers.

3. **Pilot-scale experiments**: Full-scale experiments were skipped due to pilot failures. The steering-by-quadrant analysis remains untested at scale.

4. **Q2 and Q4 empty**: The absence of Q2 and Q4 features limits our ability to compare best-case vs worst-case scenarios directly.

## 5.7 Future Directions

1. **Investigate common cause**: Understanding why absorption and sensitivity are positively correlated may reveal fundamental properties of SAE feature learning.

2. **Scale up**: Larger experiments on more models and layers are needed to confirm whether Q4 is truly empty or just rare.

3. **Theoretical development**: New theoretical frameworks are needed to explain why high-sensitivity features appear to be absorbed.

4. **Alternative metrics**: New unsupervised metrics for predicting both absorption and sensitivity may enable better feature selection.

<!-- FIGURES
- None
-->
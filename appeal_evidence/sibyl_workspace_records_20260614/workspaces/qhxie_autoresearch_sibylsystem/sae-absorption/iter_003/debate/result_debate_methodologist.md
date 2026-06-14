# Result Debate: Methodologist Perspective

## Statistical Rigor Assessment

### AUROC Estimation (A1, A3)
- **Positive**: Bootstrap CI (10,000 resamples, seed=42) correctly computed. DeLong test used for pairwise comparison — appropriate for paired AUROC comparison.
- **Concern**: n_pos=18 at L6 makes bootstrap CI wide (~0.13-0.15 width). Standard error of AUROC ≈ √(AUROC(1-AUROC)/n_pos) ≈ 0.10 for n=18 — any CI below ±0.10 should be viewed cautiously.
- **Improvement needed**: Report exact bootstrap CI bounds explicitly; consider stratified bootstrap if positive class prevalence is very low.

### Cross-Architecture Comparison (A3)
**Critical confound**: Standard SAE uses `blocks.6.hook_resid_pre`; TopK-32k uses `blocks.6.hook_resid_post`. These are different activation spaces (before vs. after attention at layer 6). The AUROC difference (0.080) may reflect this hook difference rather than architectural differences. 

**Required fix**: Add a comparison using the same hook point for both architectures, or explicitly note this as a confound that prevents attribution of the AUROC difference to architecture.

### OMP Oracle Test (C1/C2)
- **Design**: OMP k=53, compared to feedforward SAE reconstruction for 18 absorbed features. Metric: false-negative rate reduction.
- **Concern**: OMP k=53 was tuned on reconstruction quality, not absorption detection. If OMP uses k=53 << true sparsity needed for absorbed features, the oracle is not truly "optimal" — it may just be suboptimally sparse.
- **Positive**: Testing all 18 absorbed features (not a subset) increases power. 0/18 FN reduction is convincing under this design.
- **Limitation to state**: The OMP oracle is optimal given a fixed k. A truly unrestricted oracle (or LASSO with very small λ) might show different results.

### Co-occurrence Analysis (B1/B2)
- **Positive**: Jaccard overlap O_jaccard correctly normalized. AUROC computed on same label set.
- **Concern**: ARS_v2 = A_cooccur × O_jaccard product form — the mathematical ceiling on A_cooccur (bounded at 0.33 from corpus statistics) was noted but not formally characterized. Why is it bounded at 0.33? Is this a property of the corpus or the SAE dictionary?
- **Missing**: What is the precision-recall curve for O_jaccard alone? AUROC can be deceptive with severe class imbalance.

### D2 Entity-Type Absorption (H3)
- **Fatal design issue**: Probes were trained on Qwen2.5-0.5B and projected to GPT-2 space via random QR decomposition. This is not a validated transfer approach.
- **Evidence that this is a probe failure**: AR = 0.000 for all three hierarchies suggests systematic failure, not genuine absence of signal.
- **Required**: Either (a) directly train probes on the target model activations, or (b) use probes from the same model family with matching dimensions.

### F1 Successive Refinement (F1)
- **Design**: Cosine similarity between absorbed 24k decoder directions and all 32k decoder directions (different hook: resid_post vs resid_pre). Same hook confound as A3.
- **Threshold (0.80)**: Arbitrary threshold. Cosine similarity > 0.80 might just mean the feature is close in activation space, not that it's "recovered."
- **Improvement**: Define "recovery" functionally — does the recovered 32k feature actually activate on absorbed-feature inputs?

## Overall Methodological Verdict

The core findings (encoder_norm AUROC, H2 OMP null, O_jaccard) are methodologically sound with acknowledged limitations. The cross-architecture comparison needs a hook-confound correction. H3 needs a completely different probe methodology. **Advance to writing with explicit limitation section addressing the hook confound and H3 methodology.**

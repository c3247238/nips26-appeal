## 5. Discussion

### 5.1 The L0 Confound

Our central finding is that L0—the average number of active features per token—is the dominant driver of absorption rate, overshadowing architectural differences. This has profound implications for the field:

1. **Existing mitigation claims are uninterpretable** without L0-matched controls. Matryoshka's reported 90% reduction and OrtSAE's 65% reduction may primarily reflect their lower natural L0, not their architectural innovations.
2. **Architecture comparisons must control for L0 where possible.** Future work should attempt L0-matching, but must also report when matching is impossible due to architectural constraints.
3. **TopK's low absorption is sparsity-driven, not architecture-driven.** The explicit k-selection mechanism enforces a fixed, low L0 that L1 regularization cannot replicate. The "advantage" is therefore a comparison artifact, not a genuine architectural improvement.

### 5.2 The Null Causal Result

Our dose-response study falsifies the hypothesis that absorption rate causally predicts downstream interpretability (as measured by feature recovery MCC). Two interpretations are possible:

1. **Genuine null effect.** Absorption may not harm interpretability because the "absorbed" parent information is still accessible through child features.
2. **Metric insensitivity.** MCC may be too coarse to capture subtle interpretability differences.

Either way, the community's assumption that "lower absorption = better interpretability" lacks causal support.

### 5.3 Contrarian Perspective: Absorption as Feature

The contrarian view—that absorption may be a feature, not a bug—deserves serious consideration. Hierarchical representation through child features mirrors human cognition: we recognize "animal" through "dog" or "cat" features, not through an explicit "animal" feature. If absorption reflects this natural hierarchical structure, efforts to eliminate it may be counterproductive.

### 5.4 Limitations

1. **Synthetic data.** Our evaluation uses controlled synthetic hierarchies, not real semantic features. Validation on GPT-2 small pretrained SAEs is ongoing.
2. **Scale.** All experiments use 1024 features, not the planned 16k. This limits generalizability to larger dictionaries.
3. **Metric sensitivity.** The flat MCC may indicate metric insensitivity rather than a genuine null effect. Alternative downstream metrics (steering efficacy, circuit-tracing) should be tested.
4. **Convergence.** Training completes in 2-3 seconds, which is unusually fast. Convergence diagnostics (loss curves) should be verified.

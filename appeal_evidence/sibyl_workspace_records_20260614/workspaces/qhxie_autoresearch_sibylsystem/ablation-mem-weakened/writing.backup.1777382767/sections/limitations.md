# 7. Limitations and Future Work

## 7.1 Limitations

1. **Single model family.** Only GPT-2 Small res-jb SAEs were tested. Our planned Gemma-2-2B experiments were blocked by gated HuggingFace access.
2. **Narrow feature set.** First-letter features (A--Z) have a shallow, uniform hierarchy. Semantic features (e.g., WordNet hierarchies) may exhibit stronger absorption and clearer task degradation.
3. **Small model.** GPT-2 Small (124M parameters) may not exhibit absorption as strongly as larger models with deeper hierarchies.
4. **Single absorption metric.** Only the Chanin differential correlation metric was used. SAEBench's ablation-based metric or alternative measures may yield different results.
5. **Two downstream tasks.** Only steering and probing were tested. Circuit finding and model editing, which require precise feature isolation, may be more sensitive to absorption.
6. **Single significant result.** Only H1b at layer 8 achieves significance. With multiple comparisons across four hypotheses and two layers, this result could arise by chance (family-wise error rate). Replication on independent data is needed.
7. **Low absorption variance.** Most features show near-zero absorption, limiting correlation power and the generalizability of our findings to feature sets with stronger absorption.

## 7.2 Future Work

1. Test with authenticated Gemma/Pythia access for cross-model validation.
2. Use semantic hierarchy features (WordNet) for richer structure.
3. Try alternative absorption metrics (ablation-based, SAEBench).
4. Test with JumpReLU SAEs, which reportedly show stronger absorption under alternative metrics.
5. Evaluate circuit finding and model editing tasks.
6. Test on larger models (Llama-3.1-8B, Gemma-2-9B).
7. Investigate why the delta steering effect is layer-dependent (significant at layer 8 but not layer 4).

<!-- FIGURES
- None
-->

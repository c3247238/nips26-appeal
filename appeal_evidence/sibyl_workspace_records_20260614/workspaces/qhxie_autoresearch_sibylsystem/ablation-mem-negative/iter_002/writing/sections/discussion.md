# 5. Discussion

## 5.1 Why Co-occurrence Clustering Cannot Detect Absorption

Our central finding is that co-occurrence clustering performs no better than random for absorption detection. The reason is fundamental: **absorption is a causal phenomenon (suppression), while clustering detects a correlational phenomenon (co-occurrence).**

Consider two features $A$ and $B$:
- **Correlation**: $A$ and $B$ frequently activate together because they are semantically related (e.g., "cat" and "dog" often appear in similar contexts). Clustering detects this.
- **Suppression**: $A$ activates and prevents $B$ from activating, even though $B$ would be semantically appropriate. Clustering does **not** detect this---in fact, suppression reduces co-occurrence, making absorbed features *less* likely to cluster together.

This explains why UAD's precision is near-zero: it detects thousands of correlated feature pairs, none of which are absorbed.

## 5.2 Theoretical Implications

Our results suggest a theoretical constraint: **unsupervised absorption detection may be impossible without modeling the causal structure of feature interactions.** Absorption is defined by what *does not* happen (the child feature is suppressed), which cannot be inferred from observations of what *does* happen (co-occurrence patterns).

This aligns with the causal inference literature: detecting "no effect" requires either (1) randomized interventions, or (2) strong structural assumptions. SAE feature dictionaries lack both.

## 5.3 Comparison with Prior Work

Chanin et al. [2024] achieved high-precision absorption detection using **supervised** hierarchy labels. Our work shows that removing this supervision eliminates detection capability. Matryoshka SAEs [Bussmann et al., 2025] take the alternative approach: **prevent** absorption through architecture design, obviating the need for detection. Our negative result strengthens the case for preventive approaches.

## 5.4 DFDA: Mitigation Without Detection

DFDA's 21.2\% improvement demonstrates that inference-time mitigation is feasible---but only when the absorbed pairs are already known. This creates an asymmetry: **mitigation is easier than detection.** Future work should explore whether mitigation can be applied blindly (to all features) without knowing which pairs are absorbed.

## 5.5 Limitations

1. **Single model**: All experiments on GPT-2 Small; larger models may have different absorption dynamics.
2. **Single concept set**: First-letter features may not represent all absorption types.
3. **Simplified UAD**: We test the most natural co-occurrence clustering formulation; more sophisticated variants might perform better.
4. **Single seed**: No multi-seed replication, though the near-random F1 suggests high variance is unlikely to change the conclusion.

## 5.6 Future Work

1. **Causal absorption detection**: Use interventions (e.g., ablating parent features and measuring child activation) to directly measure suppression signals.
2. **Preventive architectures**: Extend Matryoshka-style approaches to eliminate absorption at training time.
3. **Blind mitigation**: Test whether DFDA can be applied to all features without detection, achieving blanket compensation.
4. **Cross-model validation**: Test whether our findings generalize to Gemma-2B, Pythia-2.8B, and larger models.

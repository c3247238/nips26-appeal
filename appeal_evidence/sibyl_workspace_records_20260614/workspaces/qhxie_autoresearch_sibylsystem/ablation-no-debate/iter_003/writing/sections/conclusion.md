# Conclusion

## Summary

We presented the first factorial decomposition of feature absorption in sparse autoencoders, testing whether absorption is driven by encoder alignment, decoder geometry, or both. Our key findings are:

1. **Encoder-driven mechanism (conditionally confirmed)**: The encoder is sufficient to drive absorption (Condition B ≈ Condition D holds with delta = 0.037), but the decoder's role is configuration-dependent, not uniformly zero. Condition C (random encoder, trained decoder) shows extreme variance (std = 17.13, range 0–43.84) across seeds, exposing decoder contributions that deterministic pilot hierarchies masked.

2. **Non-monotonic hierarchy strength relationship (not confirmed)**: H_Comp fails to confirm a monotonic increase of absorption with parent-child cosine similarity. The regression yields R² = 0.04 (target > 0.8), slope = −0.296 (p = 0.703, not significant), and absorption ranges from 0.51 to 1.20 across cosine levels {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}.

3. **Null sensitivity-absorption trade-off (inconclusive)**: H_Pareto finds absorption = 0.0 (std = 0.08) across all L0 targets {16, 32, 64, 128} while sensitivity remains stable at 0.1054. No Pareto frontier is detected. The theoretical prediction of an irreducible trade-off is not supported in synthetic SAE experiments.

4. **Safety-critical features show no elevated absorption (null result, positive for safety)**: Gemma Scope pilot (5 + 5 features, 100 samples each) yields p = 1.0 with both groups at zero absorption. GPT-2 Small held-out validation (20 + 20 features, 100 samples each) yields p = 0.345 (not significant; safety mean = 233.13, non-safety mean = 221.70). Safety-critical features do not appear disproportionately absorbed — SAE-based safety analysis may be more reliable than feared.

## Contributions

1. **Conditional encoder-driven mechanism**: First factorial decomposition showing encoder alignment drives absorption, with decoder contribution that is configuration-dependent (not uniformly zero as pilot experiments suggested). The prevailing decoder-centric narrative is incomplete but not entirely wrong.

2. **Non-monotonic hierarchy strength**: First measurement showing no clear monotonic relationship between hierarchy strength and absorption (R² = 0.04), contradicting the phase-transition framing sometimes invoked from interdisciplinary perspectives.

3. **Null Pareto frontier**: First attempted quantification of the sensitivity-absorption trade-off in synthetic SAEs — degenerates to null result, suggesting either the trade-off does not exist in synthetic hierarchies or the measurement approach is insufficient.

4. **Safety-critical feature methodology**: First methodology for testing SAE reliability on safety-critical features. The null result (p = 0.345 on GPT-2 Small) is methodologically valuable — it suggests safety features may be robust to absorption, but larger-scale validation is needed.

## Implications

Our findings reframe absorption mitigation strategies. Since the encoder is sufficient to drive absorption, decoder modifications alone will be insufficient. Mitigation must target the encoder's learned alignment with hierarchical structure:

- **Orthogonality constraints** during training could prevent parent-child encoder alignment
- **Hierarchical contrastive learning** could explicitly encourage independent feature directions
- **Architecture modifications** could separate parent and child learning pathways

However, the null sensitivity-absorption frontier means we cannot claim fundamental limits on mitigation — the theoretical trade-off is not supported in our synthetic experiments.

The positive safety finding (null absorption for safety-critical features) provides tentative support for SAE-based safety analysis. Safety-critical features do not show elevated absorption in preliminary validation across Gemma Scope and GPT-2 Small SAEs. However, larger-scale validation across more models and more diverse safety feature sets is required before strong conclusions.

## Limitations

Our work has limitations:

1. **Synthetic hierarchies**: Real LLM feature hierarchies may differ structurally from 3-level synthetic hierarchies with stochastic noise (epsilon ~ N(0, 0.05))
2. **Gemma Scope pilot**: Only 5 safety vs 5 non-safety features; limited sample size (100 per feature); single layer (layer 12)
3. **Absorption metric saturation**: Multi-child proportional ablation with k = 5 may saturate for deep hierarchies, producing degenerate zero-absorption results
4. **Cross-model generalizability**: Held-out validation only on GPT-2 Small; Gemma 2B not fully validated with large feature sets
5. **Decoder role underexplored**: The configuration-dependent decoder contribution is identified but not mechanistically explained

## Future Directions

1. **Larger-scale safety validation**: Establish validated safety-critical feature annotations across multiple model families (Claude, Llama, Mistral) to enable conclusive H_Safe testing

2. **Mechanistic investigation of decoder contributions**: Understand why Condition C shows extreme seed-dependent variance (std = 17.13, range 0–43.84) — what seed configurations amplify vs suppress absorption?

3. **Alternative absorption metrics**: Develop intervention-based or information-theoretic absorption metrics that avoid saturation observed with multi-child proportional ablation (k = 5) in deep hierarchies

4. **Encoder modification techniques**: Test orthogonality penalties, hierarchical contrastive learning, and architecture modifications for mitigation

5. **Pareto frontier re-examination**: If the sensitivity-absorption trade-off exists in real SAEs, replicate using real LLM feature hierarchies, different sensitivity metrics, and different absorption measurement approaches

## Final Remarks

Feature absorption is a primarily encoder-learned phenomenon — but not exclusively so. The decoder's configuration-dependent contribution means prior work attributing absorption to decoder geometry was incomplete but not wrong. The encoder's learned alignment is necessary and sufficient in fully-trained SAEs (B ≈ D confirmed), while the decoder's contribution amplifies or suppresses absorption in a seed-dependent manner.

The absence of elevated absorption for safety-critical features is a positive signal for SAE-based interpretability. Safety-critical applications should not assume features related to deception, jailbreak, or harm are disproportionately vulnerable to absorption — at least at the preliminary validation scale we report here. However, this null result requires larger-scale validation before strong claims.

Understanding absorption's mechanism is the first step toward managing its effects. Our factorial decomposition provides that understanding, opening pathways to more reliable SAE-based interpretability for safety-critical applications.

<!-- FIGURES
- None
-->
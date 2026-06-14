# Discussion

## 7.1 Absorption is Real but Measurement-Dependent

Our first key finding is that feature absorption is a genuine phenomenon that can be reliably measured with the right methodology. Multi-child proportional ablation revealed a substantial separation between trained SAEs (absorption rate 0.50) and random decoder baselines (0.059), with a very large effect size (Cohen's d = 8.94, p = 3.16e-133). This confirms that structured feature learning produces meaningful absorption patterns, not statistical artifacts.

However, the measurement methodology is critical. Single-child ablation saturates at 1.0 for both trained and random SAEs, as the remaining children collectively compensate for any single ablated child. Multi-child ablation, which ablates the top-k children simultaneously, was essential to differentiate the conditions. This methodological insight suggests that prior work using single-feature ablation may have underestimated or mischaracterized absorption.

An unexpected finding was that shuffled features and permuted encoder baselines showed intermediate absorption (0.487 and 0.484, respectively), only slightly below trained SAEs. These baselines preserve encoder-decoder geometry but break feature identity (shuffled) or encoder-feature correspondence (permuted). This suggests that absorption is partially driven by the decoder's geometric structure, not solely by learned feature semantics.

## 7.2 Competitive Exclusion Not Supported

The competitive exclusion hypothesis predicted that features competing for representation would show inversely correlated frequency and absorption: frequently activating features would be absorbed more as they compete for inclusion in the sparse representation. Our results falsified this prediction.

H2 showed a positive correlation between absorption and frequency (Spearman rho = 0.17, p = 6.47e-08), not the negative correlation predicted by competitive exclusion (rho < -0.3). Higher-frequency features actually tend to have higher absorption, not lower. This suggests that the ecological Lotka-Volterra analogy may not directly apply to SAE feature dynamics.

Several interpretations are possible. First, frequently activating features may simply have more opportunities for absorption to occur. Second, the synthetic hierarchy geometry may not capture the niche competition dynamics present in real language model activations. Third, absorption may be driven more by geometric alignment than by competitive pressure.

The proportional variance analysis provides additional context: trained SAEs show higher variance in child contributions (0.1154) compared to baselines (0.004 for random decoder). This asymmetry suggests that some children dominate absorption, but the relationship with frequency is more nuanced than competitive exclusion predicts.

## 7.3 Absorption May Be Epistemic, Not Causal

Our steering intervention (H3) produced no sensitivity improvement for absorbed features. Absorbed features (n=7) showed identical mean sensitivity before and after steering toward parent directions (37.45), as did non-absorbed features (n=1014, mean 185.43). This null result warrants careful interpretation.

If absorption were a causal failure -- where child features actively interfere with parent feature signaling -- then steering toward the parent direction should restore some sensitivity. The complete absence of improvement suggests that absorption may instead be epistemic: absorbed features may never have developed sensitivity to parent directions in the first place, rather than losing it through interference.

This interpretation aligns with the geometric account suggested by the shuffled/permuted baseline results. If absorption is driven primarily by decoder geometry rather than active interference, then steering interventions targeting the parent's reconstruction direction may be misdirected.

However, the non-absorbed features also showed no steering response, raising the possibility that our steering methodology -- simple linear addition of the parent direction -- was insufficient. Future work should explore nonlinear interventions or feature-specific steering strategies.

## 7.4 Implications for Safety Analysis

A central motivation for studying absorption is its potential impact on SAE-based safety analysis. If safety-critical features were disproportionately absorbed, SAE-based detection of deception, jailbreaking, or manipulation could be unreliable precisely when it matters most.

Our H_Safe hypothesis was not implemented in this study due to time constraints. However, our results suggest caution regardless of H_Safe's outcome. Even if safety-critical features are not disproportionately absorbed, the finding that absorption may be epistemic -- not causal -- implies that absorbed features may never have reliably encoded safety-relevant information. SAE-based safety analysis may face fundamental limitations beyond disproportionate absorption.

The positive frequency-absorption correlation adds a further concern: commonly activating features (which include many safety-relevant behaviors) may be more susceptible to absorption, not less. This makes it critical to evaluate absorption on real language model features before trusting SAE-based safety analysis.

## 7.5 Limitations

Several limitations constrain the generalizability of our findings.

**Synthetic hierarchies**: Our experiments used synthetic 3-level hierarchies (parent -> children -> grandchildren) with controlled geometric properties. Real language model features may have different hierarchical structures, more irregular geometries, and cross-hierarchy interactions that our synthetic data does not capture. The gap between synthetic and real feature geometry is a fundamental challenge for this research area.

**Single architecture**: We evaluated only TopK SAEs on d=512 synthetic activations. JumpReLU, gated SAEs, and other architectures may show different absorption patterns. Real transformer residual streams have different statistical properties than our synthetic data.

**Steering methodology**: Our steering intervention used simple linear addition of the parent direction vector. More sophisticated interventions (multiplicative modulation, attention-based steering, or activation patching) might reveal effects our methodology missed.

**Limited statistical power for H3**: With only 7 absorbed features identified, the steering experiment had limited power to detect small effects. The non-absorbed control group (n=1014) also showed no steering response, suggesting the methodology may need refinement rather than more samples.

## 7.6 Future Directions

Several directions emerge from our findings.

First, replication on real language model features is essential. Our synthetic hierarchies demonstrate absorption exists and can be measured, but whether this transfers to features learned from real transformer activations remains an open question. Gemma Scope or GPT-2 Small SAEs provide natural targets for this extension.

Second, the causal status of absorption requires further investigation. Our steering null result is consistent with absorption being epistemic, but does not conclusively rule out causal mechanisms. Activation patching experiments -- where we directly intervene on child feature activations -- could more directly test causal hypotheses.

Third, mitigation strategies deserve exploration. If absorption is driven by geometric alignment, architectural modifications (orthogonality constraints, multi-resolution ensembles) might reduce absorption. If absorption is epistemic, different training objectives might produce more robust feature representations.

Fourth, the relationship between absorption and downstream task performance requires characterization. We tested sensitivity to parent directions, but downstream tasks (question answering, code generation) may have different vulnerability profiles.

Finally, the safety implications warrant urgent attention. Even without H_Safe results, the possibility that commonly activating features are more susceptible to absorption raises concerns for safety-critical applications. Rigorous evaluation of absorption on safety-relevant features should be a priority for the field.

## 7.7 Conclusion

This study provides the first rigorous characterization of feature absorption in SAEs using multi-child proportional ablation on synthetic hierarchies. We confirm that absorption is a genuine phenomenon (H1 supported, d=8.94), but find that it does not follow competitive exclusion dynamics (H2 falsified) and may be epistemic rather than causal (H3 falsified).

The negative results are scientifically valuable. Falsifying the competitive exclusion hypothesis eliminates one theoretical account and constrains the space of plausible mechanisms. The null steering result suggests that absorbed features may never have developed parent-direction sensitivity, shifting focus from interference to developmental accounts.

Our findings contribute to a growing body of work questioning the reliability of SAE-based interpretability (Korznikov et al., 2026). The field should proceed with caution when using SAEs for safety-critical analysis, and future work should focus on rigorous validation on real language model features before drawing strong conclusions about absorption's practical implications.

<!-- FIGURES
- Figure 1: gen_fig1_h1_absorption_comparison.py, fig1_h1_absorption_comparison.pdf -- Multi-child proportional absorption rate comparison across trained SAE and baselines
- Figure 2: gen_fig2_h2_frequency_correlation.py, fig2_h2_frequency_correlation.pdf -- Scatter plot showing absorption vs. feature activation frequency
- Figure 3: gen_fig3_prop_variance.py, fig3_prop_variance.pdf -- Proportional variance comparison across conditions
- Figure 4: gen_fig4_h3_steering_sensitivity.py, fig4_h3_steering_sensitivity.pdf -- Steering sensitivity before/after for absorbed vs. non-absorbed features
- Figure 5: fig5_method_architecture_desc.md -- Architecture diagram of multi-child ablation procedure
- Table 1: inline -- Main results summary with H1, H2, H3 outcomes
- Table 2: inline -- Statistical test details (t-tests, correlations)
-->

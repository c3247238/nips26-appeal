# Conclusion

This paper provides the first rigorous characterization of feature absorption in sparse autoencoders using multi-child proportional ablation on synthetic hierarchies with ground truth structure.

Our central result is that absorption is real: trained SAEs show absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133). This large separation confirms that structured feature learning produces genuine absorption patterns, not statistical artifacts of how we measure them.

However, two negative results constrain the practical implications. First, absorption does not follow competitive exclusion dynamics: contrary to ecological predictions, higher-frequency features show higher absorption, not lower (rho = +0.17). Second, steering absorbed features toward parent directions produces no sensitivity improvement, suggesting absorption may be epistemic rather than causal -- absorbed features may never have developed parent-direction sensitivity rather than actively losing it through interference.

These findings contribute to a growing body of work questioning the reliability of SAE-based interpretability. Korznikov et al. (2026) showed that SAEs recover only 9% of true features despite 71% explained variance. Our results add a specific mechanism: absorption, when measurable, appears to reflect learned geometry rather than active interference with downstream utility.

The safety implications warrant urgent investigation. H_Safe -- whether safety-critical features are disproportionately absorbed -- was not tested in this study due to resource constraints. However, the positive frequency-absorption correlation suggests that commonly activating features (which include many safety-relevant behaviors) may be more susceptible to absorption. SAE-based safety analysis may face limitations beyond disproportionate absorption of critical features.

Future work should pursue four directions. First, replication on real language model features (Gemma Scope, GPT-2 Small) is essential -- synthetic hierarchies demonstrate absorption exists but transferability remains unestablished. Second, activation patching experiments could more directly test causal hypotheses than steering interventions. Third, mitigation strategies (orthogonality constraints, multi-resolution ensembles) deserve systematic evaluation. Fourth, the relationship between absorption and downstream task performance requires characterization beyond feature sensitivity.

The methodology introduced here -- multi-child proportional ablation -- resolves a measurement crisis in prior work. Single-feature ablation saturates at 1.0 for both trained and random SAEs, preventing differentiation. Multi-child ablation, which ablates top-k children simultaneously, reveals separation that single-child ablation cannot detect. This methodological contribution extends the sanity-check framework of Korznikov et al. (2026) specifically to absorption metrics.

Ultimately, this work demonstrates that the path forward requires both methodological rigor and epistemic humility. Absorption can be measured, but what it measures -- and whether it matters for downstream applications -- remains an open question for the field.

<!-- FIGURES
- None
-->

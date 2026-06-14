# 8. Conclusion

## 8.1 Summary

We conducted the first systematic study linking feature absorption detection to downstream interpretability task performance. Using pre-trained GPT-2 Small SAEs (res-jb, 24,576 latents) across layers 0, 4, 8, and 10, we measured absorption rates for 26 first-letter features (A--Z) via the Chanin et al. differential correlation metric, then tested steering effectiveness and sparse probing accuracy.

The results are mixed. Raw steering success shows no significant correlation with absorption (Pearson $r = +0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8). Sparse probing shows no correlation at either layer ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). However, delta-corrected steering---subtracting random baseline steering success---reveals a significant negative correlation at layer 8 (Pearson $r = -0.431$, $p = 0.028$; Spearman $\rho = -0.502$, $p = 0.009$). The relationship is inconsistent across layers: H1b slopes have opposite signs ($\beta_4 = +1.441$, $\beta_8 = -2.491$), and H3 fails the consistency threshold. Only one of four hypotheses (H1b at layer 8) is supported.

## 8.2 Contributions

Our work makes five contributions:

1. **First quantitative bridge between absorption and task performance.** While absorption has been detected, standardized, and targeted by architectural innovations, no prior work measures whether it degrades the interpretability tasks that motivate SAE research. We provide that measurement for steering effectiveness and sparse probing accuracy, yielding a mixed result that is itself informative.

2. **Demonstration that random baseline control is essential for steering evaluation.** Raw steering metrics conflate feature-specific contribution with generic directional bias from arbitrary decoder directions. Our H1 (raw) vs. H1b (delta) contrast shows that the same data produce no correlation in raw form but a significant negative correlation after baseline subtraction. The field should adopt delta-corrected steering as standard practice.

3. **Training-free methodology accessible to any researcher.** Our approach requires no SAE training, only pre-trained models and open-source tools (SAELens, TransformerLens). The four-phase pipeline---absorption detection, steering with random baseline, probing, correlation analysis---can be replicated on any model with available SAEs.

4. **Evidence that absorption's impact is subtle, layer-dependent, and task-specific.** Absorption degrades delta-corrected steering at layer 8 but not at layer 4, and it does not degrade probing at either layer. This task-specificity is itself an important finding that should guide future SAE research and architectural design.

5. **Actionable guidance for the field.** The field should prioritize task-relevant evaluation over metric optimization. SAEBench and similar frameworks would benefit from downstream task benchmarks (steering fidelity, probing accuracy, circuit recovery) alongside structural metrics (absorption, sparsity, explained variance).

## 8.3 Closing Thought

The SAE credibility crisis demands rigorous, task-oriented evaluation---not just optimization of metrics that may be decoupled from real interpretability work. Our mixed result on absorption and downstream performance suggests that the relationship is more nuanced than previously assumed: absorption does matter for steering when properly measured (with delta correction), but not for probing, and the effect is layer-dependent at best.

Null results are valuable: they prevent the field from over-investing in solutions to non-problems. Carefully controlled positive findings are equally valuable: they reveal relationships that raw metrics obscure. Our work demonstrates both. Whether absorption matters for other models, other metrics, or other tasks remains an open question, and we hope our methodology enables the community to answer it.

<!-- FIGURES
- None
-->

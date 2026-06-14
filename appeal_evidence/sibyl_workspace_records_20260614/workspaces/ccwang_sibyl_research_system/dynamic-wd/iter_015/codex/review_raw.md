**Summary**

This paper argues that several strands of dynamic weight decay can be viewed through a unified PID-style control lens centered on the per-layer gradient-to-weight ratio, and proposes a proportional controller, UDWDC, as an explicit instantiation of that view. It also introduces three diagnostic metrics, reports fitting-based evidence that some existing methods map to the proposed control law, and presents experiments on CIFAR-10, CIFAR-100, and ImageNet showing that CPR performs best while UDWDC is unstable.

**Strengths**

- The paper tackles a real synthesis problem: dynamic weight decay methods are indeed scattered across scheduling, alignment-aware, and norm-constrained literatures, and the attempt to compare them under one framework is timely and potentially valuable.
- The central organizing variable, the per-layer gradient-to-weight ratio \(\rho_t^l\), is a plausible and interpretable diagnostic. Figure 1 is a useful qualitative illustration that several methods induce broadly similar early-training \(\rho\) trajectories.
- The paper is unusually candid about negative results. In particular, the fact that UDWDC underperforms FixedWD in Table 3 and Table 5, and that the authors foreground this rather than hiding it, is a positive aspect of the experimental reporting.
- The identification of confounds is sometimes strong. Section 6.3 explicitly acknowledges the “CWD magnitude confound,” and Table 3 makes clear that CWD’s effective WD budget is roughly half of FixedWD’s, which is exactly the kind of issue many papers would avoid discussing.
- The cross-method diagnostic ambition is good. Metrics such as BEM and CSI are motivated by the observation that accuracy alone is insufficient to characterize dynamic regularization methods.
- The paper is at its best when making modest, family-level claims rather than exact equivalence claims. Table 2 usefully shows that CWD and CPR fit the proposed law much better than SWD and DefazioCorrective, which helps delimit scope.

**Weaknesses**

- The core “unification” claim is substantially overstated relative to the evidence. By the paper’s own Table 2, the proposed control law fails badly for two of the most important nontrivial families discussed, SWD (45.81% error) and DefazioCorrective (37.56% error). That means the framework does not actually unify the space quantitatively; at best it provides a loose taxonomy for a subset of methods.
- The theoretical section is not at the standard of a top-tier ML theory contribution. Theorem 1 and Propositions 2–3 are presented largely as statements and intuition, but the text shown does not provide precise assumptions, proof sketches, or derivations sufficient to assess correctness. Several claims are qualitative (“strictly tighter,” “geometrically consistent,” “anti-correlated”) without a rigorous development.
- The derivation of the target trajectory \(\rho^*(t)=\eta_t/\tau\) in Section 3.1 is explicitly heuristic, relying on an adiabatic extension of a constant-learning-rate steady-state result to cosine decay. This is a critical bridge in the framework, yet it is not theoretically justified and may not hold in the regime where the controller is actually evaluated.
- The mapping from prior methods to PID terminology is often semantically loose. In particular, calling CWD “derivative” control is not convincing: the method uses an alignment mask, not a temporal derivative of the error signal. The paper itself effectively uses \(K_d\) to mean “alignment-modulated correction,” which is not standard PID.
- The proposed evaluation metric CSI is internally inconsistent. Section 4.2 defines CSI as \(1/\mathrm{Var}[\lambda_{\mathrm{eff}}]\), which cannot be negative, yet Table 6 reports negative CSI values for UDWDC, and the caption of Table 6 introduces a different formula, \(1/(1+\mathrm{Var}[\lambda_{\mathrm{eff}}/\mathrm{mean}(\lambda_{\mathrm{eff}})])\), which again cannot be negative. This is a major methodological problem because CSI is one of the paper’s headline contributions.
- BEM is also underdeveloped and potentially misleading. It normalizes accuracy gain by a cumulative WD budget \(\sum_{t,l}\lambda_t^l \|w_t^l\|^2\), but Table 3 shows an extreme outlier (UDWDC-v2 budget \(98599\) versus FixedWD \(0.48\)) caused by implementation/design choices involving BN layers. This suggests the metric may be dominated by artifact-laden scaling choices rather than meaningful “regularization efficiency.”
- The experimental comparison is incomplete in the most important setting. Table 5, the large-scale ImageNet result, is missing SWD, DefazioCorrective, and NoWD entirely, and CWD has only one seed. Given that the paper’s main claim is a unified benchmark across methods, incomplete large-scale coverage weakens the paper substantially.
- Several statistical claims are too strong for the available data. The ImageNet comparison in Table 5 reports highly significant Welch tests with only 3 seeds for CPR/UDWDC and 5 for FixedWD. That is not robust evidence for broad claims, especially when other baselines are absent. More generally, many conclusions are drawn from 3-seed experiments without effect-size framing or correction for multiple comparisons.
- The empirical scope is narrow relative to the paper’s framing. Most experiments are on SGD-trained CNNs with BatchNorm, yet the introduction and related work emphasize Adam-family optimizers, AdamW, ViTs, and scaling-law-style arguments. Proposition 2 is explicitly about Adam geometry, but the experiments do not test Adam-based settings at all.
- The strongest positive empirical result, CPR’s superior accuracy in Table 3 and Table 5, is not clearly attributable to the proposed framework rather than simply stronger effective regularization. The authors partially acknowledge this in Section 6.5, but this issue cuts directly against the claimed mechanistic insight.
- Key baselines are missing. The most obvious one, which the paper itself admits in Section 6.3, is a FixedWD baseline with halved \(\lambda\) to match CWD’s effective regularization level. Without that control, the CWD interpretation remains unresolved.
- The AIS analysis is not yet convincing. Pooling correlations across all \((t,\mathrm{config})\) pairs risks pseudoreplication and conflates temporal autocorrelation with cross-configuration differences. The reported AIS values therefore do not clearly establish that alignment is an informative control signal rather than a correlated byproduct of training phase.
- Figure 1 is based on a 10-epoch pilot, yet it is used in the introduction to support the strong claim that \(\rho_t^l\) is the shared control target for all methods. This is suggestive, not decisive.
- Writing quality is uneven. The manuscript is generally readable, but it repeatedly mixes careful caveats with broad headline claims, leading to a mismatch between evidence and rhetoric. The abstract is especially overpacked and makes conclusions sound firmer than the experiments support.

**Questions for Authors**

1. What is the exact final definition of CSI used to produce Table 6, and how can the metric become negative if the formulas in Section 4.2 are strictly nonnegative?
2. Can you provide the missing matched-magnitude baseline for CWD, i.e., FixedWD with \(\lambda = 0.5\lambda_{\text{base}}\), on at least CIFAR-10 and ideally ImageNet? Without this, how can one separate “alignment information” from simple WD reduction?
3. The framework is built around \(\rho^*(t)=\eta_t/\tau\), but Section 3.1 states this is a heuristic extension from constant-\(\eta\) theory. Do the main empirical conclusions still hold if one uses alternative target trajectories or no explicit target at all?
4. Why should CWD be interpreted as a derivative controller rather than an alignment-gated proportional or switching controller? Can you justify the PID terminology more rigorously?
5. Can you provide the full ImageNet benchmark with all methods and at least 3 seeds each? As it stands, Table 5 is too incomplete to support the paper’s “standardized benchmark” framing.
6. How sensitive are BEM and CSI to whether BN parameters are included in the layer set? Table 3 suggests that this design choice can dominate the metric values.
7. Are there actual proofs for Theorem 1 and Propositions 2–3 in the full submission, and if so, what are the precise assumptions? The current exposition is too informal to evaluate as a theory contribution.

**Missing Related Work**

The paper should discuss prior work on hyperparameter adaptation and bilevel optimization more directly, since the proposed controller is one way to adapt regularization online rather than the only principled one.

- Baydin et al., “Online Learning Rate Adaptation with Hypergradient Descent” (ICLR 2018 workshop): relevant as a classic hypergradient approach to adaptive optimization hyperparameters.
- Franceschi et al., “Bilevel Programming for Hyperparameter Optimization and Meta-Learning” (ICML 2018): relevant for framing adaptive WD as a bilevel control problem rather than a fixed handcrafted controller.
- Lorraine and Duvenaud, “Stochastic Hyperparameter Optimization through Hypernetworks” / related reversible-learning work on scalable hyperparameter optimization: relevant because adaptive regularization coefficients can be learned rather than hand-designed.
- Wu et al., “Understanding and Improving Information Transfer in Multi-Task Learning” is not directly relevant, but there are optimizer-control and control-theoretic tuning papers beyond PIDAO that should be situated more carefully if the paper wants to emphasize the control perspective.
- More broadly, the manuscript would benefit from citing work on hypergradient-based or meta-learned regularization schedules, not just hand-designed WD variants, because these are an obvious alternative to the proposed PID parameterization.

**Minor Issues**

- Section 4.2 and Table 6 give incompatible CSI definitions.
- The notation switches between \(\rho_t^l\) and \(r_l^*\) in Proposition 3 without explanation.
- The manuscript alternates between describing CWD as using \(\alpha_t^l<0\) to “disable” or “apply” decay and as “increasing \(\lambda\)” via the \(K_d\) term; this should be clarified.
- Table numbering is awkward: Table 7 is introduced before Table 5 in the narrative flow.
- The statement that UDWDC has “zero new hyperparameters” is somewhat misleading once clamp bounds, EMA smoothing, and floor clipping are introduced.
- The BEM discussion for ImageNet is inconsistent with the absence of a NoWD baseline in Table 5.
- “Derivative/alignment feedback” is nonstandard terminology and should either be justified or renamed.
- Some claims in the abstract are too strong relative to the evidence, especially “all four traditions implicitly manipulate a single quantity” and the conclusion that the proposed diagnostics “expose method differences invisible to accuracy-only evaluation” before the metrics are fully validated.

**Overall Assessment**

Score: 5/10

Confidence: 4/5

The paper has an interesting high-level idea and several good instincts: it tries to compare a fragmented literature under a common lens, it reports negative results honestly, and it identifies a real confound in CWD-style methods. However, the current submission falls short of top-venue standards because the central unification claim is only partially supported, the theory is too informal, one of the main proposed metrics (CSI) appears internally inconsistent, and the large-scale empirical benchmark is incomplete. I would be open to a substantially revised version with a corrected metric definition, stronger matched baselines, complete ImageNet experiments, and a more modest framing that presents the work as a partial taxonomy plus diagnostic benchmark rather than a full quantitative unification.
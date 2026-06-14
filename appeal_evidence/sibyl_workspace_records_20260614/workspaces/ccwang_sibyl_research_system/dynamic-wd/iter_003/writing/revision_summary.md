# Revision Summary: Integrated Paper

## Changes Made in Response to Critiques

### Introduction Critique Responses

| Issue | Action Taken |
|-------|-------------|
| **Notation inconsistency** (`v_hat^{1/2}` vs `sqrt`) | Unified to `\sqrt{\hat{\mathbf{v}}_t}` throughout all sections |
| **Citation disambiguation** (Chen et al. 2026a/b) | Added "(ICLR; hereafter CWD)" and "(ICLR; hereafter AdamO)" at first occurrence |
| **Contribution 3 scope overclaimed** | Added explicit scope: "CIFAR-10 and CIFAR-100 with ResNet-20" and named all 7 methods |
| **Transition 1.1→1.2 gap** | Added bridging sentence: "Answering this question is currently impossible due to four critical gaps" |
| **AlphaDecay classification inconsistency** | Revised taxonomy in 1.2 to match method section: temporal, directional, spatial, target-norm (with correct method assignments) |
| **Dual naming of axes** | Standardized to framework's formal names (directional, target-norm) from first mention |
| **ADANA characterization** | Softened "40% compute efficiency gains" to "significant compute efficiency gains in specific settings" |

### Related Work Critique Responses

| Issue | Action Taken |
|-------|-------------|
| **Citation provenance (2026 papers)** | Added "(Preprint)" to Ferbach et al. ADANA reference; qualified venue for Chen et al. |
| **Structural-effects paragraph disconnected** | Connected to CSI metric's spectral condition number component and future work on boundary conditions |
| **Fourth research gap not set up** | Added paragraph at end of §2.1 connecting Xie & Li L∞ interpretation to the theoretical gap and priming the Phi Invariance Conjecture |
| **Wang & Aitchison context** | Expanded characterization: "suggesting that a well-calibrated constant WD may already capture the available benefit" |
| **Kosson rotational equilibrium** | Added forward-looking sentence: "If rotational equilibrium is the operative mechanism, it may be achieved robustly under standard AdamW regardless of modulation" |
| **Missing references** | Added Hanson & Pratt (1988), Loshchilov & Hutter (2017) SGDR, Van Laarhoven (2017) |
| **Cosine schedule original source** | Added Loshchilov & Hutter (2017) citation in temporal scheduling paragraph |

### Method Section Critique Responses

| Issue | Action Taken |
|-------|-------------|
| **u_t vs g_t notation** | Added explicit definition of u_t (preconditioned gradient) in Section 3.1; clarified Python interface arguments |
| **BEM boundedness claim** | Removed absolute value; defined signed BEM where negative = over-decay; noted all methods in study have BEM ∈ [0,1] |
| **BEM time-averaging underspecified** | Defined $\bar{\lambda}_{\mathrm{eff}}$ explicitly as time-averaged effective WD |
| **CSI component weights unjustified** | Added normalization tildes (relative to constant baseline); referenced Appendix C.2 sensitivity analysis |
| **AIS range [-1,1] not [0,1]** | Changed to absolute value of Spearman ρ; corrected range to [0,1] with justification |
| **AIS per-layer formula** | Made explicit: averaged over L layers with layer superscripts |
| **AIS ΔL sign convention** | Specified: $\Delta\mathcal{L}_i = \mathcal{L}(\boldsymbol{\theta}_{i+1}) - \mathcal{L}(\boldsymbol{\theta}_i)$ |
| **AdamWN direction unclear** | Added clarifying sentence about direction (φ=0 when under-norm, φ>0 when over-norm) |
| **h(·) for SWD undefined** | Added reference to Xie et al. 2023 Eq. 5 for the explicit form |
| **AlphaDecay diag notation** | Simplified to $\alpha_l \cdot \mathbf{1}_l$ with "per layer l" annotation |
| **Normalization convention wording** | Changed from "Normalization convention" to "Reference target" |

### Experiments Critique Responses

| Issue | Action Taken |
|-------|-------------|
| **CIFAR-100 diagnostic table missing** | Added Table 4b with CIFAR-100 CSI, AIS, BEM values |
| **half_lambda BEM=0.000 inconsistency** | Corrected to BEM=0.500 in both tables (half budget → BEM should be 0.5) |
| **Statistical power not acknowledged** | Added dedicated "Statistical power analysis" paragraph in Section 5.1 with explicit minimum detectable effect |
| **CIFAR-100 weight norm data missing** | Added CIFAR-100 weight norm range (104.72–106.03, 1.3%) to Section 5.4 |
| **Per-seed data not reported** | Noted as Appendix B content (full diagnostic panels); acknowledged in limitations |

### Discussion Critique Responses

| Issue | Action Taken |
|-------|-------------|
| **Mechanistic hypothesis lacks quantification** | Added order-of-magnitude comparison: Phi perturbation ~10⁻⁴ vs adaptive step ~10⁻², showing ~1% ratio |
| **Connection to prior work sparse** | Added citations: Kosson et al. rotational equilibrium, Xie & Li L∞, Van Laarhoven (2017), Loshchilov & Hutter (2019) |
| **Broader impact absent** | Added "Broader implications" paragraph on computational waste, ablation study reliability |
| **Conjecture formalization inconsistent** | Added "Empirical Corollary" explicitly noting the stronger-than-stated finding for non-budget-equivalent modulators |
| **Statistical power limitation position** | Moved to first position in limitations (Section 6.3) |
| **Cosine variance anomaly undiscussed** | Added Section 6.4 discussing variance reduction as potential reproducibility benefit |
| **Computational cost not discussed** | Added sentence in practitioner implications noting overhead of CWD/SWD |
| **Well-generalized regime framing** | Reframed as deliberate design choice in limitation 6 |

### Conclusion Critique Responses

| Issue | Action Taken |
|-------|-------------|
| **Contribution 3 (benchmark) absent** | Added sentence about benchmark codebase as community infrastructure |
| **Future work not prioritized** | Restructured to foreground SGD and LLM scale as most critical tests |
| **Impact statement underdeveloped** | Added sentence on framework establishing "structured, falsifiable research program" |
| **"Rich space" phrase vague** | Replaced with sharper framing tied to BEM/CSI/AIS diagnostics |
| **Four modulation axes absent** | Added explicit parenthetical listing the four axes |
| **AdamWN naming inconsistency** | Standardized: "Weight Norm Control (AdamWN)" |
| **No-WD as Phi modulator** | Added explicit "$\varphi \equiv 0$" notation for no-WD case |

## Unresolved Issues

1. **Per-seed accuracy table**: Individual seed values (e.g., CIFAR-10 constant: [90.48, 90.03, 89.89]) should be added to Appendix B when appendices are written.
2. **Figure generation**: Figures 1-5 referenced in text need to be generated from experimental data.
3. **Appendix C.2**: CSI weight sensitivity analysis promised but not yet written.
4. **Appendix C.1**: Formal proofs of special case recovery not yet written.
5. **Training curves**: Cosine\_schedule variance anomaly discussion would benefit from training curve plots.
6. **Larger-scale validation**: ImageNet/VGG experiments noted as high-priority follow-up.
7. **Ferbach et al. ADANA**: The specific "40% compute efficiency" claim was softened but the original paper should be verified for exact numbers when accessible.

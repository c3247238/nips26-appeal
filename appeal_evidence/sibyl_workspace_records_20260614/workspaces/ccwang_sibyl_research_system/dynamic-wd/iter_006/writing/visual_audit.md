# Visual Audit Report

## Summary

- **Total figures in manuscript**: 6 (Figure 1 through Figure 6)
- **Total tables in manuscript**: 2 (Table 1, Table 2)
- **PDF figure files present**: 5 of 6 (Figure 1 is a conceptual diagram described in markdown)
- **Missing visuals**: 0 critical (see notes below)

## Figure Inventory

| Figure | File | Referenced Before Appearance | Present on Disk |
|--------|------|------------------------------|-----------------|
| Figure 1 | framework_diagram_desc.md | Yes (Section 3.3) | Yes (markdown description) |
| Figure 2 | main_results_bar.pdf | Yes (Section 6.2) | Yes |
| Figure 3 | weight_norm_trajectories.pdf | Yes (Section 6.3) | Yes |
| Figure 4 | lyapunov_curves.pdf | Yes (Section 6.3) | Yes |
| Figure 5 | pmpwd_switching.pdf | Yes (Section 6.3) | Yes |
| Figure 6 | bem_accuracy_scatter.pdf | Yes (Section 6.3) | Yes |

## Table Inventory

| Table | Location | Referenced Before Appearance |
|-------|----------|------------------------------|
| Table 1 | Section 3.1 (inline) | Yes (Section 1 and Section 3.1) |
| Table 2 | Section 6.2 (inline) | Yes (Section 6.2 opening) |

## Consistency Issues Found and Fixed

1. **Figure numbering consolidated**: The original sections used independent numbering (method section had "Figure 1", experiments had "Figure 2-6"). Renumbered sequentially across the full manuscript: framework diagram = Figure 1, main results = Figure 2, weight norms = Figure 3, Lyapunov = Figure 4, PMP-WD switching = Figure 5, BEM scatter = Figure 6.

2. **Table numbering consolidated**: Method section had "Table 1" (phi taxonomy), experiments had "Table 2" (main results). Kept as Table 1 and Table 2 in integrated paper. Both are referenced before appearance.

3. **Terminology standardization**: Verified consistent use of:
   - "alignment-aware" (not "directional") throughout
   - "cosine schedule" / "cosine-scheduled WD" (not "Cosine WD")
   - "gradient-weight alignment" (not "GWA" in running text)
   - CWD = Cautious Weight Decay (consistent)
   - SWD = Scheduled Weight Decay (consistent)

4. **SWD sensitivity function**: Table 1 references $h(\|g\|)$ for SWD. Added clarifying note in the table caption that $h$ is a sensitivity function mapping gradient norms to modulation weights, with $h(\|g\|_{\text{ref}})$ as normalization. This addresses a prior critique about $h(.)$ lacking a closed-form expression -- the framework intentionally leaves this as a general function class.

5. **Appendix references**: Changed "Appendix C" reference in Theorem 1 proof sketch to "Appendix A" for consistency with the outline's appendix ordering (A = proofs, B = configs, C = per-seed results, D = cost analysis, E = additional visualizations).

6. **CIFAR-100 spread**: Intro stated "0.67 percentage points" but experiments section data shows 0.58pp (63.42 - 62.84). Fixed intro to "0.58 percentage points" for consistency with the data.

## Figures Planned in Outline But Not in Current Manuscript

The outline planned 8 figures and 4 tables. The current manuscript includes 6 figures and 2 tables. Missing items:

| Planned Visual | Status | Reason |
|---------------|--------|--------|
| Figure 3 (outline): Certified band with method trajectories | Not generated | Requires instrumented epoch_metrics with $\lambda_{\min}$, $\lambda_{\max}$ fields; data partially available |
| Figure 7: Cumulative vs worst-case alignment scatter | Not generated | Requires rho_sweep experiment data (H2 validation) |
| Figure 8: BN vs Non-BN accuracy spread | Not generated | Non-BN experiments not yet run |
| Table 3: BN ablation and diagnostic metrics | Not generated | Requires non-BN experimental data |
| Table 4: Subsumption verification fractions | Not generated | Requires per-step band membership computation |

These missing visuals correspond to experiments that are planned but not yet completed (non-BN ablation, rho_sweep, certified band visualization). The current manuscript is self-consistent with the 6 figures and 2 tables present.

## Visual Verification Results (PDF Inspection)

All 5 PDF figures were visually inspected:

1. **main_results_bar.pdf (Figure 2)**: Bar chart with 6 methods, correct accuracy values (89.80--90.29), error bars present, "0.49pp spread" annotation visible, shaded band overlay. Axis labels clear. Colors distinct. Data matches Table 2. Pass.

2. **lyapunov_curves.pdf (Figure 4)**: Log-scale $V_t$ trajectories for all 6 methods. All curves monotonically increase (dominated by $\mu_t\|w_t\|^2$ norm term). PMP-WD (red) starts lower then converges with others after epoch 50. Lines mostly overlap in late training. Legend readable. Note: the text says "monotonically increasing $V_t$" which matches the figure -- $V_t$ increases because the $\mu_t\|w_t\|^2$ term dominates as norms grow. The description in Section 6.3 is accurate.

3. **bem_accuracy_scatter.pdf (Figure 6)**: Scatter with 6 labeled points, regression line, $r = 0.61$, $p = 0.19$ annotated. Constant at BEM=0, No WD at BEM=1.0, CWD/Cosine/PMP-WD clustered around BEM=0.5, SWD at BEM=0.9. Error bars present. Colors match other figures. Data matches Table 2. Pass.

4. **pmpwd_switching.pdf (Figure 5)**: Dual panel. Top: switching function $\sigma(t)$ oscillates around zero with pink/gray shading for ON/OFF regions. Bottom: switch rate curve starting ~0.45, stabilizing near 0.55 by epoch 50. Bang-bang pattern clearly visible. Amplitude decreases in late training as described. Pass.

5. **weight_norm_trajectories.pdf (Figure 3)**: All 6 methods plotted, trajectories nearly indistinguishable. Start at ~34, end at ~96-97. Legend shows all methods. Colors match other figures. Demonstrates BN-induced scale invariance as claimed. Pass.

**No rendering anomalies, truncation, or blank regions detected in any figure.**

## Suggestions for Additional Visuals

1. **Certified band overlay (high priority)**: A figure showing $[\lambda_{\min}(t), \lambda_{\max}(t)]$ as a shaded band with method trajectories overlaid would be the most impactful addition, directly visualizing the core theoretical claim.

2. **CIFAR-100 results table**: The CIFAR-100 results are reported in prose. A formal table (parallel to Table 2) would improve readability.

3. **Subsumption fraction table**: Quantitative subsumption data (% of steps in band per method) is mentioned in text but not tabulated.

# Visual Audit Report

## Summary (Updated: iter_014 writing_integrate -- editor integration pass, post-figure-generation audit)

- **Total figures in paper**: 4 (Figures 1--4), all with generated PDF and PNG files
- **Total tables**: 6 (Tables 1--6, all inline)
- **Missing visuals**: 0 critical, 4 optional (from outline's 8-figure plan)
- **Figure numbering**: Sequential 1--4 with no gaps
- **Table numbering**: Sequential 1--6 with no gaps

---

## Completeness Check

### Present figures -- visual inspection results

| Figure | File | Section | Visual Inspection |
|--------|------|---------|-------------------|
| Figure 1 (GW ratio trajectories) | `figures/rho_trajectories.pdf` | 1 (Introduction) | PASS -- two-panel layout: (a) mean rho_t^l trajectories over 10 epochs for 7 methods with distinct colors/markers matching style_config.py; (b) box plots showing per-layer distribution at epoch 10 for FixedWD, CWD, CPR, UDWDC. CPR starts highest, UDWDC shows wider whiskers consistent with negative CSI. Axis labels legible, legend complete. |
| Figure 2 (UDWDC control loop) | `figures/udwdc_control_loop.pdf` | 3.3 (UDWDC Algorithm) | PASS -- block diagram with Target Generator (rho*(t) = eta_t/tau), Summation, Proportional, Clamp, Weight Update (Plant), and Measurement blocks. Open-loop FixedWD bypass path shown. Three other methods (SWD, CWD, CPR) annotated at bottom with their respective feedback mechanisms. Color-coded regions (red = UDWDC, green = FixedWD path). Text small but legible at print resolution. |
| Figure 3 (Alignment SNR) | `figures/alignment_snr.pdf` | 5.4 (Batch-Size Sensitivity) | PASS -- three methods plotted (FixedWD gray, CWD blue, UDWDC red) across 5 batch sizes (64--1024). X-axis: Batch Size; Y-axis: Alignment SNR. Method differentiation clear, no truncation or rendering artifacts. |
| Figure 4 (CSI comparison) | `figures/csi_comparison.pdf` | 5.7 (CSI Stability) | PASS -- three panels (Temporal, Spatial, Combined CSI). All 6 methods present. UDWDC negative temporal CSI (-5.75) visually prominent as only negative bar. Values match Table 6 entries. |

### Absent figures (from outline's 8-figure plan, not referenced in paper body)

| Outline Figure | Planned Content | Priority | Notes |
|---------------|----------------|----------|-------|
| Figure 5 (outline) | Budget efficiency curves (accuracy vs. WD budget scatter) | LOW | BEM values already reported numerically in Table 3. Would be additive but not critical. |
| Figure 6 (outline) | Effective lambda_t trajectories showing CPR integral ramp-up | MEDIUM | Would visually demonstrate the integral control accumulation pattern. Strongest candidate for a 5th figure. |
| Figure 8 (outline) | ImageNet training curves (top-1 val accuracy vs. epoch) | LOW | ImageNet results are underpowered (4/7 methods, CWD at 1 seed). Adding curves for an incomplete comparison may draw attention to the gap. Consider deferring to appendix. |
| (Critique suggestion) | H7 temporal predictability R-squared histogram by layer type | LOW | Would replace 3 paragraphs of dense prose in Section 5.6. Appendix candidate. |

---

## Tables

| Table | Section | Content | Status |
|-------|---------|---------|--------|
| Table 1 | 3.2 | PID mapping (6 methods x 7 columns: Target, Kp, Ki, Kd, Feedback, Control Type) | Present, inline |
| Table 2 | 3.2 | Unification fitting results (5 methods: relative error, pass/fail) | Present, inline |
| Table 3 | 5.2 | CIFAR-10/ResNet-20 main results (8 methods: accuracy, WD budget, BEM, gen gap) | Present, inline |
| Table 4 | 5.3 | UDWDC gain ablation on CIFAR-100/VGG-16-BN (7 gain variants) | Present, inline |
| Table 5 | 5.5 | ImageNet/ResNet-50 results (4 methods: accuracy, WD budget) | Present, inline |
| Table 6 | 5.7 | CSI stability comparison (6 methods: temporal, spatial, combined CSI) | Present, inline |

---

## Consistency Check

### Figure references in body text
- Figure 1: Referenced in Section 1 (Introduction) paragraph 5 and caption immediately after. First figure in paper, serves as visual hook for unification claim.
- Figure 2: Referenced in Section 3.3 (UDWDC Algorithm). Placed directly after Algorithm 1 pseudocode.
- Figure 3: Referenced in Section 5.4 (Batch-Size Sensitivity). Placed after SNR discussion text, before the recommendation paragraph.
- Figure 4: Referenced in Section 5.7 (CSI Stability) and cross-referenced in Section 6.4 (UDWDC Instability discussion). Placed after Table 6 and the CSI interpretation text.
- No orphan figures (all 4 are referenced before appearance).
- No dangling references (no text references a figure that does not exist).

### Table references in body text
- All 6 tables are referenced by number in body text before or at their appearance.
- Table numbering is sequential with no gaps.

### Color consistency
- `figures/style_config.py` defines method colors: FixedWD (gray), CWD (blue), SWD (orange), CPR (green), Defazio (purple), NoWD (beige), UDWDC (red).
- Figures 1, 3, and 4 use these colors consistently.
- Figure 2 uses structural colors (red = UDWDC controller, green = open-loop path) rather than method colors, which is appropriate for a block diagram.

---

## Consistency Fixes Applied During Integration

1. **CWD alignment condition corrected** (related_work_critique Issue 3): Unified to alpha_t^l < 0 throughout (intro, related work, method, experiments).
2. **DefazioCorrective Kp fixed** (method_critique Issue 6): Table 1 updated to Kp=0, "Feedforward" control type (not proportional).
3. **SWD mapped to Kp,Ki > 0** (intro_critique): Proportional-integral classification used consistently.
4. **UDWDC rank clarified as rank-7** (intro_critique): No longer "rank-2"; positioned below NoWD on CIFAR-10.
5. **Arithmetic error fixed** (discussion_critique Issue 1): "1.26 pp" corrected to "2.01 pp" (70.53 - 68.52).
6. **H3 reported as FALSIFIED** (experiments_critique Issue 1, critical): Section 5.4 rewritten; CWD SNR non-monotonic, CWD accuracy collapse at bs>=512. Discussion Section 6.5 recommendation updated accordingly.
7. **BEM/CSI/AIS moved to Section 4** (method_critique Issue 2): Metric definitions removed from method section; standalone Section 4.
8. **CSI normalization documented**: Table 6 caption documents FixedWD=1.0 convention.
9. **Figure 4 cross-referenced in Discussion** (discussion_critique Issue 4): "See Figure 4" added to Section 6.4.
10. **UDWDC-v2 ablation note added** (experiments_critique Issue 8): Section 5.3 notes all variants use v2 floor-clipping.
11. **Conclusion reframed** (conclusion_critique): UDWDC instability as diagnostic discovery; BEM/CSI/AIS as unified diagnostic protocol.
12. **Figure numbering renumbered** to sequential 1--4 (was non-sequential 1,2,4,7 in raw sections).

---

## File Integrity

| File | Exists | Formats | Generator Script |
|------|--------|---------|-----------------|
| `figures/rho_trajectories` | Yes | PDF + PNG | `gen_rho_trajectories.py` |
| `figures/udwdc_control_loop` | Yes | PDF + PNG | `gen_udwdc_control_loop.py` |
| `figures/alignment_snr` | Yes | PDF + PNG | `gen_alignment_snr.py` |
| `figures/csi_comparison` | Yes | PDF + PNG | `gen_csi_comparison.py` |
| `figures/style_config.py` | Yes | Python | (shared style configuration) |

All 4 figures have both PDF (for LaTeX) and PNG (for preview) versions. All generator scripts are present for reproducibility.

---

## Suggestions for Additional Visuals

The paper has a figure-to-page ratio of approximately 4 figures across ~16 pages of content. For NeurIPS (8 pages + appendix), this is adequate but on the lower end.

1. **Priority 1: Effective lambda_t trajectories** (would become Figure 5). A 2-panel figure showing CPR's integral accumulation ramp-up vs. CWD's binary switching pattern would visually demonstrate the two mechanisms the PID framework unifies. This is the strongest candidate for a 5th figure.

2. **Priority 2: Budget efficiency scatter** (would become Figure 6). A BEM vs. accuracy scatter plot would make the efficiency argument in Section 6.2 immediately visual. Currently BEM values are reported only in Table 3 columns.

3. **Lower priority: ImageNet training curves**. Given that ImageNet results are underpowered (4/7 methods, CWD at 1 seed), adding training curves for an incomplete comparison may draw more attention to the gaps than strengthen the argument. Consider deferring to appendix.

4. **Appendix candidate: H7 R-squared histogram**. The bimodal R-squared distribution across layer-method combinations (Section 5.6) is currently prose-only. A histogram broken down by layer type would be more scannable.

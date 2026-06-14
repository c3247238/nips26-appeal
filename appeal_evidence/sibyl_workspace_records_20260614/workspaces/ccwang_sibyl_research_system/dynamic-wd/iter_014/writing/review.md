# Writing Quality Review

## Summary

This paper proposes a PID-style feedback control unification of four weight decay (WD) sub-traditions, using the per-layer gradient-to-weight ratio $\rho_t^l$ as the shared control variable. It maps CWD, SWD, CPR, and DefazioCorrective to gain configurations $(K_p, K_i, K_d)$, proposes UDWDC (a proportional controller), introduces three standardized metrics (BEM, CSI, AIS), and reports honest negative results across CIFAR-10, CIFAR-100, and ImageNet. The paper has been substantially revised: H3 is now correctly reported as falsified in Sections 6.4, 7.5, and 8 using full-experiment data; Section 7.6 consolidates five negative results. Three critical problems remain: (1) Figure 4 (alignment_snr.pdf) still uses pilot data showing CWD monotonically increasing, directly contradicting the updated text which reports CWD as non-monotonic and H3 falsified; (2) Figure 2 (UDWDC control loop) is a dead reference to a .md descriptor file; (3) four additional planned figures are absent from the paper body. The CSI normalization and AIS operational-definition problems are also unresolved. The score remains at 6.

---

## Detailed Assessment

### Structural Coherence: 8/10

The argument arc — fragmentation → shared variable → PID taxonomy → theory → metrics → evidence → failures — is coherent and consistently maintained. The Introduction's organization paragraph correctly lists all eight sections. Each section opens with an explicit transition.

This revision correctly propagates H3 falsification: Section 6.4 states "**H3 is falsified.**" with exact SNR values, Section 7.5 opens "the batch-size sweep (Section 6.4, Figure 4) falsifies H3," and Section 8 lists H3 among negative results. This cross-section consistency is a genuine improvement.

Two remaining issues:

1. Section 6.6 (H7 temporal predictability) is not mentioned in the Section 6 opening summary ("three dataset-architecture combinations"). Readers encounter H7 without context.

2. Section 4.1 introduction sentence duplicates Section 4 opener — minor redundancy only.

---

### Notation & Terminology Consistency: 8/10

**Resolved from prior reviews:** Table 1 DefazioCorrective has $K_p=0$, $K_i=0$, "Feedforward." UDWDC rank reads "rank-7-of-8." Section 7.6 arithmetic is 2.01 pp throughout.

**Persisting issue — CSI normalization undefined:** Section 5.2 states FixedWD achieves CSI$_\text{combined}$ = 1.0 "normalized relative to FixedWD's zero variance as the reference." This is inaccurate: FixedWD's $\rho_t^l$ is not zero-variance under mini-batch training. The normalization convention (divide all CSI values by FixedWD's CSI) must be stated explicitly. notation.md line 72 also needs updating.

**Persisting minor issue:** The abstract defines UDWDC via the additive PID form; Algorithm 1 uses the multiplicative clamp form. Section 3.3 bridges them, but the abstract should signal this: "(implemented as a multiplicative clamp; see Algorithm 1)."

**Glossary compliance verified:** "Closed-loop," "per-layer," "cosine annealing," "generalization gap," "batch size" are all used consistently throughout.

---

### Claim-Evidence Integrity: 6/10

**New critical discrepancy — Figure 4 contradicts Section 6.4 text (H3 falsification):**

Section 6.4 text (updated in this revision): "**H3 is falsified.** CWD SNR shows a non-monotonic pattern (0.0074 at bs=64 → 0.0075 at bs=128 → drops at bs=256 → partial recovery)." Figure 4 caption: "CWD and UDWDC show non-monotonic patterns."

Direct visual inspection of `writing/figures/alignment_snr.png`: CWD (blue diamond markers) is visually **monotonically increasing** from ~0.17 at bs=64 to ~0.29 at bs=1024, with no non-monotonic behavior. The figure uses pilot data (~10-epoch sweep); the text cites full 200-epoch data from `exp/results/full/phase2_batchsize/summary.md` (CWD: 0.0074, 0.0075, 0.0001, 0.0009, 0.0014). Any reviewer comparing Figure 4 to the text will conclude there is a data error. The fix is straightforward: the correct full-data figure exists at `exp/results/full/figures/fig04_batchsize_snr.pdf`.

Additionally, Section 6.4's final paragraph ("The monotonic SNR scaling for CWD and FixedWD supports recommending binary masking...") was not updated with H3 falsification — it still uses the old pilot-based framing, creating an internal contradiction within the same section.

**Remaining unresolved issues:**

- AIS operational definition: "optimal WD decision estimated retrospectively by correlating alignment with generalization-gap changes" is circular. metrics_results.json shows AIS = 0.566 (distinct from correlation_alpha_gengap = 0.698), but the paper provides no formula for the target variable. AIS cannot be reproduced.

- H7 data source: Section 6.6 reports 2640 traces and specific $R^2$ statistics (18.1% above 0.85, mean 0.279) with no source artifact cited in the workspace.

**Verified correct (unchanged):**
- CIFAR-10 Table 3: all values match `phase1_diagnostic/summary.md` exactly.
- CIFAR-100 Table 4: all values match `phase2_ablation/summary.md` exactly.
- ImageNet Table 5: CPR 74.74 ± 0.05%, FixedWD 71.72 ± 0.36% match `imagenet_main/pilot_summary.json`.
- UDWDC CSI$_\text{combined}$ = −2.41 matches metrics_results.json (−2.407632, rounded correctly).
- Table 6 CSI values match metrics_results.json refined values exactly.

---

### Visual Communication: 5/10

**Present figures (2 of 8 planned):**

- **Figure 7** (csi_comparison.pdf/png): Correct and strong. Three-panel bar chart, UDWDC −5.75 temporal bar visually prominent, values match Table 6. No changes needed.

- **Figure 4** (alignment_snr.pdf/png): Present but contains **wrong data** for the current text. Figure (pilot data) shows CWD monotonically increasing. Text reports CWD non-monotonic, H3 falsified (full data). Must be replaced with `exp/results/full/figures/fig04_batchsize_snr.pdf`.

**Missing figures (6 of 8):**

- **Figure 2** (UDWDC control loop, Section 3.3): Dead reference. `writing/figures/udwdc_control_loop_desc.md` is a text description, no image. LaTeX compilation blocker.

- **Figure 1** ($\rho_t^l$ trajectory, Introduction): Not referenced in body. Central unification claim is assertion-only. Pre-generated data exists at `exp/results/full/figures/fig01_rho_trajectories.png`.

- **Figures 3, 5, 6, 8**: Listed in end-of-paper list but not referenced in any body section. Pre-generated versions exist at `exp/results/full/figures/` (fig03, fig05, fig06, fig07) but are not embedded. Section 6.6 (H7 bimodal) is entirely text-only.

**Tables (6 of 6 present):** All correct. Table 5 (ImageNet) is missing BEM and Gen Gap columns present in CIFAR tables.

---

### Writing Quality: 8/10

Evidence-first, free of banned patterns. Key observations subsections lead with numbers and close with mechanisms.

**New clarity issue (this revision):** Section 6.4 internal contradiction. The section opens "**H3 is falsified.** CWD SNR shows a non-monotonic pattern" but closes "The monotonic SNR scaling for CWD and FixedWD supports recommending binary masking at $b \leq 256$." The recommendation contradicts the evidence found three paragraphs earlier in the same section.

**Persisting clarity issue:** Section 5.3 AIS — "optimal WD decision" is undefined operationally.

**No banned patterns found:** "In recent years," "Furthermore," "Moreover," "groundbreaking," "to the best of our knowledge" — none present.

**Positive writing (unchanged):**
- Section 3.3 ("These are engineering patches, not principled solutions...") — honest, mechanistic, specific.
- Abstract pre-registration of negative results — unusual and effective.
- Section 7.3 (CWD confound) — correct structure throughout.

---

## Issues for the Editor

1. **[Critical] Replace Figure 4 — wrong data, contradicts text.** `writing/figures/alignment_snr.pdf` shows CWD monotonically increasing (pilot data). Section 6.4 reports CWD non-monotonic, H3 falsified (full data). Replace with: `cp exp/results/full/figures/fig04_batchsize_snr.pdf writing/figures/alignment_snr.pdf`. Also rewrite Section 6.4's final paragraph to replace the old monotonic-CWD recommendation with a H3-consistent conclusion. **Location**: `writing/figures/alignment_snr.pdf`; Section 6.4 final paragraph.

2. **[Critical] Generate Figure 2 (UDWDC control loop) — LaTeX compilation blocker.** Section 3.3 has a dead reference to `figures/udwdc_control_loop_desc.md`. Generate a 5-block diagram as matplotlib or TikZ. Save as `writing/figures/udwdc_control_loop.pdf`. Replace the dead reference. **Location**: Section 3.3, line ~181.

3. **[Critical] Embed Figure 1 and resolve absent figures.** Pre-generated figures exist at `exp/results/full/figures/` but are not in the paper. At minimum: embed `fig01_rho_trajectories.pdf` in the Introduction. Embed `fig06_h7_temporal_gate.pdf` in Section 6.6. For Figures 3, 5, 6, 8: embed available pre-generated files or remove from the end-of-paper list. **Location**: Introduction, Section 6.6, "Figures and Tables" end section.

4. **[Major] Section 6.4 last paragraph contradicts H3 falsification.** Rewrite to: alignment-aware WD under $\lambda_\text{base} = 10^{-4}$ provides no reliable accuracy benefit at any tested batch size; practitioners need a dedicated ablation before relying on CWD. **Location**: Section 6.4, final paragraph.

5. **[Major] CSI normalization undefined.** Add: "All CSI values are normalized by FixedWD's CSI, establishing FixedWD = 1.0 as the reference." Update notation.md. **Location**: Section 5.2; notation.md.

6. **[Major] AIS "optimal WD decision" operationally undefined.** Add one sentence specifying the target variable in the Spearman correlation. Update glossary.md. **Location**: Section 5.3.

---

## What Works Well

1. **H3 falsification propagated correctly in all text sections.** Section 6.4, 7.5, and 8 are internally consistent. The full SNR values and accuracy deltas ($-$3.84 pp at bs=512, $-$5.63 pp at bs=1024) are correctly cited. The figure just needs to match.

2. **Figure 7 (CSI comparison) is clean and accurate.** Three panels, UDWDC −5.75 temporal bar prominent, values match Table 6 exactly. No changes needed.

3. **Negative result structure remains the paper's strongest differentiator.** Section 3.3 ("engineering patches, not principled solutions"), Section 7.6 five-item failure list, and the abstract's pre-registration of negative results are all precisely structured and unusual for a top-venue submission.

SCORE: 6

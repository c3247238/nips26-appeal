# Critique: Experiments (Revision Round 3)

## Summary Assessment

The Experiments section presents genuine multi-scale empirical evidence — CIFAR-10 diagnostic, CIFAR-100 gain ablation, ImageNet full comparison, batch-size sweep, temporal predictability, CSI stability — with negative results clearly labeled and specific numbers verified against source files. Prior Round 2 critical issues have been partially addressed: the H7 statistics (18.1%, mean R²=0.279, N=2640) now match the full 200-epoch H7 gate result in `phase3_alignment/h7_gate_result.json`, not the 10-epoch pilot. However, one Critical Issue remains unfixed and will cause rejection: Section 6.4 still claims H3 is confirmed ("monotonic SNR increase") when the full batch-size sweep data explicitly marks H3 as FALSIFIED with CWD collapsing catastrophically at large batch sizes. The section also contains one new major finding missed in prior rounds: Section 6.6 misstates which layer types have low R², attributing R² < 0.04 to "BN and bias layers" when shortcut connections (mean R² = 0.301), fc/weight layers (0.116), and fc/bias (0.062) are all above this threshold.

## Score: 6/10

**Justification**: Three-point improvement from Round 2 due to H7 statistics being correctly verified (Round 2 Issue 2 was a false alarm — the full-run H7 data was simply not consulted in that round). CIFAR and ImageNet numerical accuracy is confirmed against source files. The remaining score ceiling is the unfixed H3 claim (which flows into Discussion 7.5 and Figure 4 caption), the four still-absent figures, incomplete ImageNet Table 5, and the layer-type mischaracterization in 6.6. Reaching 8/10 requires fixing Issue 1 and adding training curves for ImageNet.

---

## Critical Issues

### Issue 1: Section 6.4 reports H3 as confirmed — the full batch-size sweep data marks H3 as FALSIFIED

- **Location**: Section 6.4, paragraphs 1–2 and Figure 4 caption
- **Quote**: "The alignment signal-to-noise ratio ... increases monotonically with batch size for both FixedWD and CWD, confirming that larger batches yield cleaner alignment estimates." and "The monotonic SNR scaling for CWD supports the recommendation that binary masking (CWD-style) is preferable at b ≤ 256."
- **Problem**: `exp/results/full/phase2_batchsize/summary.md` begins: "H3 result: FALSIFIED." The actual SNR numbers from that file directly contradict the text:

  | BS | FixedWD SNR | CWD SNR | FixedWD direction | CWD direction |
  |----|------------|---------|------------------|--------------|
  | 64 | 0.0281 | 0.0074 | — | — |
  | 128 | 0.0167 | 0.0075 | DOWN | UP |
  | 256 | 0.0011 | 0.0001 | DOWN | DOWN |
  | 512 | 0.0001 | 0.0009 | DOWN | UP |
  | 1024 | 0.0010 | 0.0014 | UP | UP |

  FixedWD SNR monotonically decreases from 64 to 512, with only a small uptick at 1024. CWD SNR collapses at bs=256 (0.0001), then partially recovers — this is non-monotonic. More importantly, CWD *accuracy* collapses catastrophically at large batch sizes: 66.16 ± 4.64% at bs=512 and 63.42 ± 7.16% at bs=1024, vs. FixedWD at 69.99 ± 0.10% and 69.05 ± 0.17%. This recommends the exact opposite of what the section states: CWD should be avoided at large batch sizes, and no batch size in the sweep showed CWD reliably outperforming FixedWD. The recommendation at b ≤ 256 is also wrong: at bs=64, CWD underperforms FixedWD by 1.25 pp; at bs=128, by 0.61 pp.
- **Cascading error**: The same false "monotonic" claim appears verbatim in Discussion Section 7.5 and in the Figure 4 caption ("SNR increases monotonically for FixedWD and CWD"). All three locations must be corrected together.
- **Fix**: Rewrite Section 6.4 around the falsified hypothesis. The genuine finding — that CWD's binary masking becomes catastrophically unstable at large batch sizes (std=7.16% at bs=1024, 6 pp below FixedWD) — is a stronger and more interesting result than the false confirmation. Present an inline SNR table with the actual data. Revise Figure 4 caption accordingly. Update Discussion 7.5 to reflect: "CWD's performance deteriorates severely at batch size ≥ 512, precluding its use in large-batch training regimes."

---

## Major Issues

### Issue 2: Section 6.6 misstates which layer types have R² < 0.04

- **Location**: Section 6.6, paragraph 2
- **Quote**: "Convolution weight layers show R² = 0.595 (median 0.825), while BN and bias layers show R² < 0.04."
- **Problem**: The full H7 gate result (`exp/results/full/phase3_alignment/h7_gate_result.json`) reports per-layer-type statistics showing this characterization is incomplete:

  | Layer type | Mean R² | Median R² | % > 0.85 |
  |-----------|---------|----------|---------|
  | conv/weight | 0.595 | 0.825 | 41.3% |
  | shortcut | 0.301 | 0.028 | 22.2% |
  | fc/weight | 0.116 | 0.109 | 0.0% |
  | fc/bias | 0.062 | 0.032 | 0.0% |
  | conv/bias | 0.034 | 0.028 | 0.0% |
  | bn/weight | 0.026 | 0.017 | 0.0% |
  | bn/bias | 0.024 | 0.018 | 0.0% |

  "BN and bias layers" with R² < 0.04 is accurate only for bn/weight, bn/bias, and conv/bias. Shortcut connections (mean 0.301, 22.2% above 0.85) and fc/weight (mean 0.116) are notably above the 0.04 threshold and carry the claim "not well-approximated by polynomial time" as strongly as convolution layers — but for different reasons. The text also implies the bimodal split is only between "conv/weight (predictable)" and "everything else (not predictable)," which the data refutes.
- **Fix**: Replace with: "Conv/weight layers show mean R² = 0.595 (median 0.825): alignment in large convolutional kernels is temporally predictable. Shortcut connections show intermediate predictability (mean R² = 0.301). BN parameters and biases show mean R² < 0.03, indicating that WD adjustments to these layers carry information that is not a simple function of training time. FC layers occupy an intermediate range (0.06–0.12)."

### Issue 3: ImageNet Table 5 still missing three baselines and all statistical significance tests

- **Location**: Section 6.5, Table 5 and surrounding text
- **Problem**: The 40-run ImageNet experiment (`imagenet_main_PROGRESS.json`) shows status = "partial_complete, 12/40 runs." Table 5 presents only CPR, FixedWD, CWD, UDWDC without disclosing that SWD, DefazioCorrective, NoWD, and UDWDC-v2 are absent entirely. This matters because: (a) without NoWD, BEM cannot be computed for ImageNet (no denominator); (b) UDWDC-v2 — the "stability-fixed" variant described in Section 3.3 — has no ImageNet result at all; (c) the headline claim "CPR leads by 3.02 pp over FixedWD" has no paired t-test. The 90-epoch FixedWD std (± 0.36%) and CPR std (± 0.05%) reflect 5 and 3 seeds respectively with different variances — a Welch's t-test is required.
- **Fix**: (1) Add "pending" rows for missing methods with footnote "(runs interrupted; partial results in Appendix)." (2) Add a NoWD row with the 2-epoch pilot result as a placeholder to enable BEM computation. (3) Report t-statistics for CPR vs. FixedWD and UDWDC vs. FixedWD: preliminary manual computation from the result files gives CPR 74.74 ± 0.05 vs. FixedWD 71.72 ± 0.36 — the CPR margin of 3.02 pp is clearly significant (t ≈ 8.3 at df ≈ 5), but this must be stated explicitly.

### Issue 4: CSI formula inconsistency across Section 3.5, notation.md, and Table 6

- **Location**: Section 6.7, Table 6 caption; method section 3.5; notation.md
- **Problem**: notation.md defines CSI = 1/Var_t[ρ_t^l] (variance of the GW ratio). Method section 3.5 defines CSI_temporal = 1/Var_t[λ_eff^l] (variance of the effective WD coefficient). For FixedWD, λ_eff is constant by definition, so 1/Var[λ_eff] = ∞, yet Table 6 reports CSI = 1.000. The `phase2_ablation/summary.md` full-run CSI column shows 0.2393 for all variants including FixedWD — a third inconsistent value. No reviewer can verify which formula was used to produce Table 6.
- **Fix**: Adopt a single normalized definition. The version in `metrics_results.json` key `cifar10_resnet20_per_method_refined` produces the Table 6 values and uses the effective-WD normalized variance with a clipping step that bounds FixedWD to 1.000. State this explicitly in Table 6 caption: "CSI = 1/(1 + Var_t[λ_eff^l / mean_t[λ_eff^l]]) averaged across last 25% of training." Update notation.md to match.

### Issue 5: Architecture generalization experiments absent with no disclosure

- **Location**: Outline Section 6.6 (ResNet-101, ViT-S/16), method section 3.4 Proposition 3
- **Problem**: Method section 3.4 states: "Our experiments verify the anti-correlation [from Proposition 3] on ResNet-50 (BN) but do not test ViT-S/16 (LN)." This implies ResNet-50 verification exists in the experiments section — it does not appear anywhere. The outline planned a Section 6.6 for architecture generalization (ResNet-101, ViT). The actual section numbering substitutes H7 temporal predictability in that slot with no disclosure. Pilot data for both architectures exists in `exp/results/pilots/imagenet_resnet101/` and `exp/results/pilots/imagenet_vit/` but is not reported.
- **Fix**: Add one paragraph to Section 6.5 or a new 6.6 using the available pilot data (single-seed, labeled as pilot). Alternatively, add "(unverified at full scale; see pilot data in Appendix)" to the Proposition 3 claim in Section 3.4 and add this gap to Section 7.7 Limitations.

### Issue 6: Ablation variants not disclosed as v2-stabilized

- **Location**: Section 6.3, opening of ablation description
- **Quote**: "To isolate the contribution of each PID component, we run 7 gain configurations on CIFAR-100/VGG-16-BN"
- **Problem**: Method Section 3.3 documents that UDWDC v1 Kp_only "produced zero WD budget" — the proportional-only controller without floor clipping disabled WD entirely. Table 4 (Section 6.3) shows Kp_only WD Budget = 0.300 (non-zero). This confirms the ablation uses v2 floor clipping, not v1. The section never discloses this. The conclusion "Adding proportional control (K_p > 0) consistently degrades accuracy" could be attributable to v2 floor clipping artifacts rather than to K_p alone.
- **Fix**: Add one sentence to the Section 6.3 setup: "All variants use the UDWDC-v2 floor-clipping fix (λ_min = 0.1λ_base); results reflect PID gain interactions in the stabilized setting, not v1 collapse behavior."

---

## Minor Issues

- **Section 6.6, H7 gate threshold not justified**: The 0.85 R² threshold is stated but not motivated. Either cite where it was pre-registered or add: "We chose R² > 0.85 as a conservative threshold for 'well-approximated by time alone'; the qualitative conclusion (gate passes) also holds at R² > 0.70, where 28.0% qualify."
- **Section 6.2 "restoring competitive performance"**: UDWDC-v2 at 90.36% ranks 5th of 8 methods. "Competitive" is imprecise. Replace with: "recovering to rank 5 of 8, 0.32 pp below FixedWD."
- **Section 6.3 Observation 1**: "alignment information provides marginal value when applied in isolation." The 0.11 pp difference (70.64 ± 0.30% vs. 70.53 ± 0.48%) is within one combined standard error. Rewrite as: "does not reliably distinguish from FixedWD at 3-seed resolution (Δ = 0.11 pp, combined std = 0.56%)."
- **Table 5 seeds column**: The column header "Seeds" but CWD row shows "1" without "(no CI)" annotation. Add the no-confidence-interval caveat in the CWD row.
- **Section 6.5 Observation 4 placement**: The "minimal augmentation" explanation for 71.72% vs. standard 76-77% should appear in the setup (Section 6.1) or before Table 5, not as observation 4. Reviewers see the number first and may flag it before reaching the explanation.
- **Section 6.5 Observation 4, missing citation**: "Standard ResNet-50 training recipes typically ... achieve 76–77% at 90 epochs" needs a citation (PyTorch vision model zoo or He et al., 2016 original paper).
- **Table 6 caption "pilot, 10 epochs"**: Updated in prior round to remove "pilot" — but per the current file, this still reads "CIFAR-10/ResNet-20 (pilot, 10 epochs)." The CSI values match the full 200-epoch refined analysis, not a 10-epoch pilot. Update caption to: "CIFAR-10/ResNet-20, 200 epochs, 3 seeds, computed over last 25% of training."
- **Section 6.4 UDWDC-v2 WD budget (98599) in Table 3**: Add footnote clarifying this is the sum $\sum_t \sum_l \lambda_t^l \|w_t^l\|^2$ over all 200 epochs × 65 layers, not a per-step value, to distinguish from FixedWD's 0.48.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan — 2 figures (4, 7) and 4 tables present; outline planned 8 figures and 6 tables. Absent: ρ_t trajectory comparison (Figure 3), ρ*(t) target-tracking (planned), budget-efficiency curves (Figure 5), effective-λ_t trajectories (Figure 6), ImageNet training curves (Figure 8), architecture generalization table.
- [x] All visuals referenced before appearance — Figures 4 and 7 both satisfy this requirement.
- [ ] Figure 4 caption factually wrong — "SNR increases monotonically for FixedWD and CWD" contradicts full data (Critical Issue 1). Must be corrected.
- [x] Figure 7 caption accurate — correctly describes UDWDC negative temporal CSI and all other methods > 0.9.
- [ ] Section 6.5 has no training curve — 90-epoch ImageNet results appear as a 4-row table with no convergence visualization. Data exists in `exp/results/full/phase4_imagenet_main/`.
- [ ] Section 6.2 discusses UDWDC control behavior with no ρ_t trajectory figure — the central visualization for the paper's feedback-control claim is absent.

---

## What Works Well

1. **H7 temporal predictability analysis (Section 6.6)**: The 2640 combination full-run analysis is correctly reported (18.1% > 0.85, mean R² = 0.279, verified against `phase3_alignment/h7_gate_result.json`). The bimodal interpretation — high predictability in conv/weight, near-zero in BN layers — is a genuine insight. The claim that CWD "derives value specifically from layers where time-polynomial fitting fails" is exactly the kind of mechanistic argument that elevates a benchmark paper to a theoretical contribution.

2. **CIFAR-10 observation chain (Section 6.2)**: Every observation leads with a number, then the mechanism. Observation 1 traces CPR's accuracy lead (91.74%) through integral control → budget accumulation (4.44, 9.3× FixedWD) → tighter gen gap (8.28%). All numbers verified against `phase1_diagnostic/summary.md`. This is the model format for result descriptions throughout the paper.

3. **CSI temporal/spatial decomposition (Section 6.7)**: UDWDC temporal CSI = −5.75 paired with spatial CSI = 0.935 identifies the failure mode precisely: temporal oscillation, not layer-to-layer inconsistency. No accuracy table can surface this distinction. The explanation in the final paragraph of 6.7 — "the instability is temporal ... not spatial" — is the section's strongest analytical statement and is directly verifiable from `metrics_results.json` key `cifar10_resnet20_per_method_refined`.

---

## Cross-Section Consistency Notes

- **Section 6.4 vs. Discussion 7.5**: Both describe H3 batch-size results. Both currently state the false "monotonic" claim. Both must be updated together (Critical Issue 1 fix).
- **Section 6.2 Observation 2 (CWD confound) vs. Discussion 7.3**: Both describe the 50% WD budget reduction. Consistent terminology. Good.
- **Section 6.5 vs. Discussion 7.6 Negative Result 4**: Discussion correctly notes "CWD has only 1 completed seed on ImageNet," consistent with Table 5. Good.
- **Section 6.7 vs. Method Section 3.5 (CSI formula)**: Inconsistency unresolved from Round 2 (Major Issue 4 above).
- **Figure numbering**: Section uses Figures 4 and 7. Outline assigns Figures 3–8 to experiments. With six figures absent, global figure numbering requires a reconciliation pass before final LaTeX compilation.
- **Section 6.3 observation 3** references "instability documented in Section 3.3." Method section 3.3 documents v1 instability. The ablation (Section 6.3) uses v2 variants. The forward reference is accurate only if the reader understands the mechanism is inherited in v2 through residual Kp-driven variance — this should be stated explicitly rather than relying on the cross-reference alone.

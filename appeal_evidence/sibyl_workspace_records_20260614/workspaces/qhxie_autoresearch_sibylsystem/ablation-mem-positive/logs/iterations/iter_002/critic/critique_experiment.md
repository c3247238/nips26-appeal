# Experiment Critique

## Overall Assessment

The experiments show significant progress from prior iterations, with successful validation of activation patching (67.3% mean recovery) and CV-based steering prediction (2x effect). The pilot experiments were well-designed with clear pass criteria, and the transition from pilots to full experiments followed a logical progression. However, critical methodological gaps remain: multiple comparisons are still uncorrected (Bonferroni would invalidate the layer 8 correlation), the full_cross_layer_critical results are not integrated into the paper, and the CV-steering connection remains correlational rather than causal. The data accuracy is verified—all reported numbers match source JSON files.

**Score: 5/10** (strong pilot validation, but full results integration and statistical rigor are lacking)

---

## Critical Issues

### 1. Multiple Comparisons Still Uncorrected

**Evidence**: At least 12 hypothesis tests were performed:
- 3 layer correlations (layers 5, 8, 10) in E6v2
- 3 Fisher z-tests (8 vs 5, 8 vs 10, 5 vs 10) in H2v2
- Mann-Whitney U for projection absorption (E7)
- Mann-Whitney U for ablation scores (E7)
- Cohen's d for effect size (E7)
- H3 test (E7)
- H1 implicit test (E3v2)

With Bonferroni-corrected alpha = 0.05/12 ≈ 0.0042:
- Layer 8 correlation: p = 0.023 → NOT significant (0.023 > 0.0042)
- Layer 8 vs 5: p = 0.0036 → significant (just barely)
- Layer 8 vs 10: p = 0.0011 → significant

**Verdict**: The layer 8 correlation itself does NOT survive Bonferroni correction. Only the pairwise comparisons survive. The paper presents the layer 8 finding as evidence of a real effect, which is incorrect after proper correction.

**Fix**: Apply Benjamini-Hochberg FDR or Bonferroni correction. Report corrected p-values. Explicitly acknowledge that layer 8's individual correlation is exploratory, not confirmatory.

### 2. Full_cross_layer_critical Results Not Integrated

**Evidence**: `exp/experiment_state.json` shows full_cross_layer_critical completed with status "completed". The proposal mentions measuring absorption at λ_c=5e-5 across layers 0, 3, 6, 9, 11, but the integrated_paper.md only covers layers 5, 8, 10 on GPT-2 and 5, 10, 15 on Gemma. The experiment results are not visible in any critique file.

**Analysis**: If the experiment found uniform saturation at λ_c, this would be a negative result contradicting the layer-dependence hypothesis. If variation was found, this would validate the H3-revised hypothesis. In either case, the results should be in the paper.

**Fix**: Either integrate the results or document them as a negative finding. Add a section to the paper reporting what was found.

### 3. Chi_ratio Below Sharp Transition Threshold

**Evidence**: chi_ratio = 1.88, but "sharp transition" threshold is 3.0. The proposal correctly frames this as "quasi-critical behavior," but the paper abstract and body still use "phase transition" language.

**Analysis**: The susceptibility peak (χ=11.19) at λ_c=5e-5 confirms a critical-like phenomenon, but the gradual transition (chi_ratio < 3.0) means it's not a sharp phase transition. The framing is technically accurate but could mislead readers expecting sharp transitions like in physics.

**Fix**: Use "quasi-critical threshold behavior" or "critical-like phenomenon" consistently. Do not claim "sharp phase transition" anywhere.

---

## Major Issues

### 4. CV-Steering Connection Is Correlational, Not Causal

**Evidence**: `exp/results/pilot_summary.json` shows pilot_steering_cv passed with high-CV mean effect 0.153 vs low-CV 0.075 (2x ratio). But this is n=30 per group (pilot stage), not full validation. The mechanism is not established.

**Analysis**: Why high-CV features are more steerable is unexplained. Possible mechanisms:
- High-CV features have more steering headroom (larger activation range)
- High-CV features are less redundant (less absorption)
- High-CV features have different decoder structures (more orthogonal)

**Fix**: Acknowledge explicitly that the CV-steering relationship is correlational, not causal. Run full validation with 1000 samples. Establish the mechanism.

### 5. Sample Size n=10 Is Inadequate

**Evidence**: Each layer correlation is computed over 10 categories. The 95% CI for rho=0.705 is [0.12, 0.93]—extremely wide.

**Analysis**: With n=10, the correlation estimate is essentially uninformative about the true effect size. The CI spans nearly the entire possible range of rho values.

**Fix**: Add explicit discussion of CI widths as a limitation. Consider whether n=10 provides sufficient power. Report exact CI values prominently.

### 6. Lambda_c Instability Not Prospectively Validated

**Evidence**: Proposal acknowledges "10x instability" in lambda_c but does not execute prospective validation. The value λ_c=5e-5 could be an artifact.

**Fix**: Test lambda_c=5e-5 on held-out data (different layers, different SAE configs). Report stability bounds.

---

## Minor Issues

### 7. Single Random Seed Only

Only seed=42 is used. No seed perturbation to verify robustness.

### 8. No Negative Controls

No random latent selection, no nonsense category probes, no shuffled labels. This limits confidence that high projection absorption is specific to genuine semantic features.

---

## Data Accuracy Verification

All reported numbers match source JSON files:
- Gemma mean AUROC = 0.980 → Verified (0.9804878...)
- Gemma ablation rate = 0.0% → Verified (0.0)
- GPT-2 layer 8 rho = 0.705 → Verified
- Layer 8 p = 0.023 → Verified (0.0227)
- Fisher z: 8 vs 5, p=0.0036 → Verified
- Fisher z: 8 vs 10, p=0.0011 → Verified

No data fabrication detected. The issue is statistical rigor, not data accuracy.

---

## What Works Well

1. **Activation patching validation is rigorous**: 67.3% mean recovery across 9 core words with clear pass criteria (>10% for 3/9 words). All 9 words passed. This is the paper's strongest validation.

2. **Pilot to full transition is logical**: Both pilots passed with clear pass criteria before proceeding to full experiments. The full experiments followed the same protocol at larger scale.

3. **Honest reporting of negative results**: H3 falsified (all layers saturate at λ=0.001), H6 falsified (graph topology), H2 falsified (mean rho = -0.194). All reported without spin.

4. **Clear pass criteria**: The pass/fail thresholds (>50% parent recovery for activation patching, high-CV > low-CV for steering) are specific and measurable.
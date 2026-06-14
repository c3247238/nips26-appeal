# Experiments Critique

## Score: 5/10

---

## Strengths

- The experimental narrative is logically structured and follows the paper's theoretical claims closely. Each subsection maps cleanly to a specific hypothesis or proposition (budget equivalence, LR coupling, alignment characterization), which aids readability.
- The budget equivalence experiment is the strongest piece of causal evidence in the paper. The design is clean: record the AADWD-Aggressive schedule, compute its time-average, run a constant WD with that average, and compare. The exact match result is a striking finding.
- The LR decoupling experiment (Section 5.4) demonstrates a mechanistically informative failure mode. The collapse trajectory description (best accuracy before milestone, weight norm decimation) is vivid and connects well to Proposition 1 in the method section.
- The alignment proxy table (Table 4) directly operationalizes the theoretical claim about the alignment signal being near-constant. Reporting phase-by-phase means is a good choice that illustrates the monotonic decrease.
- The hyperparameter sensitivity section covers a wide range of c and beta values, and the conclusion that AADWD is robust over a 5x range of c is well-supported.

---

## Issues (by priority)

### Critical

**C1. Norm-Matched WD is listed as a method but entirely absent from Table 1.**

The paper explicitly mentions "Norm-Matched WD (weight norm trajectory matched to AADWD)" in the Methods Compared section as one of the ablation controls. It is listed as a distinct experimental condition. However, Table 1 contains no row for this method. The actual experimental result (best_test_acc=90.44%, final=75.37%, weight_norm=5.51) shows a severe late-training collapse resembling the CWD failure mode. Omitting this result without explanation is a significant methodological omission. A reviewer will notice the methods description lists this control but the table does not include it. Either include the row with a note explaining the instability, or remove it from the methods description entirely. Selective omission of a collapsed result biases the presentation.

**C2. Single-seed experiments with no statistical acknowledgment.**

Every experiment in this paper uses seed=42. This is confirmed in all summary JSON files across all tiers (tier1, tier2, cross-arch, ablations, sensitivity). The paper never states this anywhere in the Experiments section. For a NeurIPS submission, this is a serious weakness: differences of 0.1-0.3% (e.g., AADWD Conservative at 92.37% vs. fixed WD at 92.54%) are within the typical variance of a single seed for CIFAR-10/ResNet-20. The paper uses phrases like "falls short of fixed WD by 0.36%" and "within noise" without providing any variance estimate to support those characterizations. At minimum, the paper should acknowledge the single-seed limitation and provide confidence intervals. Ideally, key comparisons (Table 1 main results) should be run with 3-5 seeds.

**C3. The "exact match" of Equiv. Cumulative WD to fixed WD at 5e-4 is suspicious and unexplained.**

The equiv_cumulative_wd config file shows `weight_decay: 0.0005`, which means the time-average of the AADWD-Aggressive schedule happened to equal exactly `5e-4` --- the same as the fixed WD baseline. The paper does not explain whether this exact equality is a coincidence, an artifact of the clipping bounds, or a design choice. This is the centerpiece causal experiment, and a reviewer will immediately ask: was the budget-equivalent constant independently computed from the AADWD trajectory, or was it set by inspection to `5e-4`? If the former, the paper must report the computed time-average and explain the remarkable coincidence. If the latter, the experiment is circular. The paper must provide the actual mean lambda value computed from the AADWD-Aggressive trajectory (e.g., "we computed lambda_bar = X.XXXX from the trajectory and set the constant WD to this value").

**C4. Inconsistency: paper claims 11 methods but Table 1 has 13 rows.**

The Methods Compared section states "We compare 11 weight decay methods." Table 1 contains 13 rows (No WD, 5 Fixed WD values, Stagewise WD, CWD, AADWD Conservative, AADWD Aggressive, AADWD Square, Random Dynamic WD, Equiv. Cumulative WD). If you count methods as in the text, the fixed WD grid search contributes 5 rows. Either correct the count or restructure the methods description.

### Major

**M1. The c-sweep in Table 5 uses different AADWD configurations than the main table.**

In Table 5 (c-sweep), the row for c=2.5 shows best accuracy 92.05% and weight norm 21.47, which matches the main table's AADWD-Aggressive. However, the c-sweep experiments must use `lambda_max=0.05` (as in the tier2_hyperparam_sensitivity/c_2.5 config) while the beta-sweep experiments use `c=2.5` with `lambda_max=0.05`. The paper does not state the lambda_max values used for the sensitivity sweep, making reproducibility incomplete. Also, the c-sweep result for c=1.0 reports weight_norm=41.27, but the raw data shows 41.2718. The data is 41.27 (rounded to 2 decimal places), while the main table uses 2 decimal places for weight norm. This rounding inconsistency is minor, but the lack of reported lambda_max for the sensitivity experiments is a genuine reproducibility gap.

**M2. Table 2 (cross-arch) is structurally asymmetric and missing important baselines.**

Table 2 compares methods across three settings but includes only: No WD, Fixed WD (5e-4), AADWD Conservative, AADWD Aggressive, and CWD. It omits Stagewise WD, which appears in Table 1 and achieves competitive performance (92.44%). It also omits Random Dynamic WD and Equiv. Cumulative WD from cross-arch validation, leaving the budget equivalence principle unverified across architectures and datasets. At minimum, Stagewise WD should be included for completeness, as it is a common practical baseline.

**M3. The alignment proxy r=0.849 is below the stated adequacy threshold.**

The paper states in the method section: "the Pearson correlation between the minibatch EMA and large-batch alignment measurements is r = 0.849." The all_results_table.json from the pilot phase records the diagnostic verdict as "NO-GO (adjust beta)" with `pearson_r_mini_ema_vs_large = 0.848947`, noting it fell short of the r >= 0.85 threshold. The final paper uses this same value (0.849) and presents it as confirmation of proxy fidelity. This requires explicit discussion: the proxy reliability marginally fails the threshold that the authors' own diagnostic established. The paper must either (a) re-run the diagnostic at the final beta=0.999 and report the updated r, or (b) acknowledge that 0.849 is the final measured value and discuss what this means for the fidelity claim.

**M4. The decoupled experiment uses substantially different c values than the main experiment.**

The aggressive_decoupled experiment uses c=0.25, while the main AADWD-Aggressive uses c=2.5. The conservative_decoupled uses c=0.0005, while the main conservative uses c=0.005. The paper does not mention this 10x difference in c values between the coupled and decoupled variants. The narrative compares them directly ("AADWD Conservative ($92.37\% \to 80.30\%$)") without clarifying that the decoupled variant uses a different scaling constant. A fair comparison would use the same c values and only remove the gamma_t multiplier. As presented, the decoupled experiment may be conflating the effect of decoupling with the effect of using a different c.

**M5. No figures are referenced in the experiments section.**

The paper describes training dynamics (weight norm collapse trajectories, alignment proxy evolution over training phases) that are inherently better communicated visually. The tier2_hyperparam_sensitivity directory contains multiple plot files (c_sweep_test_acc.png, beta_sweep_weight_norm.png, lambda_trajectories.png, etc.). None of these are referenced in the experiments section. A NeurIPS paper is expected to include learning curves and trajectory plots, especially for dramatic failure modes like the aggressive decoupled collapse (weight norm 21.47 → 0.0036). The lack of any figure reference in the experiments section is unusual and will be flagged by reviewers.

**M6. The hyperparameter sweep values in the paper (c in {0.5, 1.0, 2.5, 5.0, 10.0}) are inconsistent with the directory naming convention and the pilot sweep values.**

The pilot experiments used c values {0.001, 0.005, 0.01, 0.05, 0.1} while the full experiments use {0.5, 1.0, 2.5, 5.0, 10.0}. This is a legitimate experimental redesign, but the paper does not explain the change in scale. Additionally, the main AADWD-Aggressive experiment (c=2.5, beta=0.999) is a single configuration from this sweep but is presented in Table 1 as the representative AADWD-Aggressive result. It is not clear whether c=2.5 was chosen as the best-performing configuration from the sweep (c=1.0 achieves 92.18%, higher than c=2.5's 92.05%) or if it was fixed before the sweep. If c=1.0 is better, the main table should use c=1.0 for a fair comparison.

### Minor

**m1. Gen gap rounding is inconsistent across tables.**

Table 1 reports gen gap values to 2 decimal places (7.17, 7.64, etc.), but the underlying data has 4 decimal places (7.1651, 7.6351). The rounding is applied correctly, but the standard deviation of alignment in Table 4 is reported as the raw value 0.000753 without matching precision to the mean values reported to 6 decimal places. A consistent reporting standard is needed.

**m2. Table 4 (alignment proxy) has missing standard deviation entries for individual phases.**

For the early, mid, and late phase rows, the Std column shows "---". Only the overall standard deviation is reported (0.000753). This appears incomplete. If per-phase standard deviations were measured, they should be reported. If not, the table structure implies they were measured.

**m3. The training protocol does not state the random seed, data preprocessing random state, or CUDA determinism settings.**

Reproducibility requires stating that all experiments used seed=42, and whether `torch.backends.cudnn.deterministic = True` was used. The methods section describes the training protocol but omits this information entirely.

**m4. The Stagewise WD definition is ambiguous.**

The paper describes Stagewise WD as "LR-proportional at milestones" but does not give the exact formula. A reader needs to know the exact proportionality constant and how the WD changes at each milestone to reproduce the result.

**m5. Table 3 (decoupled variants) reports "Coupled (%)" as a percentage accuracy column but also includes "WN (Coupled)" and "WN (Decoupled)" as weight norm columns. The mixed units in a single table require a clearer header.**

---

## Data Accuracy Check

### Table 1 (Main Results, CIFAR-10/ResNet-20)

Cross-referencing against `final_analysis.json`:

| Row | Best Acc | Final Acc | Gen Gap | Weight Norm | Status |
|-----|----------|-----------|---------|-------------|--------|
| No WD | 90.49 | 90.31 | 9.17 | 129.42 | PASS (raw: gap=9.1732, wn=129.4235) |
| Fixed WD 1e-4 | 92.08 | 91.83 | 7.64 | 56.34 | PASS (raw: gap=7.6351, wn=56.3407) |
| Fixed WD 3e-4 | 92.39 | 91.97 | 7.52 | 30.70 | PASS (raw: gap=7.5232, wn=30.7043) |
| Fixed WD 5e-4 | 92.54 | 92.29 | 7.17 | 23.49 | PASS (raw: gap=7.1651, wn=23.4928) |
| Fixed WD 1e-3 | 92.32 | 91.92 | 6.95 | 17.13 | PASS (raw: gap=6.9542, wn=17.1297) |
| Fixed WD 3e-3 | 88.63 | 87.25 | 5.20 | 11.44 | PASS (raw: gap=5.2039, wn=11.4407) |
| Stagewise WD | 92.44 | 92.27 | 7.18 | 33.22 | PASS (raw: gap=7.1791, wn=33.2201) |
| CWD | 91.79 | 86.95 | 3.22 | 9.28 | PASS (raw: gap=3.2163, wn=9.2846) |
| AADWD Conservative | 92.37 | 92.22 | 7.15 | 23.60 | PASS (raw: gap=7.151, wn=23.6025) |
| AADWD Aggressive | 92.05 | 91.57 | 7.50 | 21.47 | PASS (raw: gap=7.4965, wn=21.4722) |
| AADWD Square | 92.13 | 91.78 | 7.47 | 38.75 | PASS (raw: gap=7.4728, wn=38.7468) |
| Random Dynamic WD | 92.06 | 91.95 | 7.54 | 34.19 | PASS (raw: gap=7.5372, wn=34.1895) |
| Equiv. Cumulative WD | 92.54 | 92.29 | 7.17 | 23.49 | PASS (raw: gap=7.1651, wn=23.4928) |
| **Norm-Matched WD** | **MISSING** | **MISSING** | **MISSING** | **MISSING** | **CRITICAL: present in methods, absent in table; raw best=90.44, final=75.37, wn=5.51** |

All reported numbers match the raw data. However, the omission of Norm-Matched WD from Table 1 is a critical gap.

### Table 2 (Cross-Architecture/Cross-Dataset)

Cross-referencing against `tier2_cross_arch/aggregate_summary.json`:

| Setting | Method | Paper Value | Raw Value | Status |
|---------|---------|-------------|-----------|--------|
| C10/ResNet-20 | No WD | 90.49 | 90.49 | PASS |
| C10/ResNet-20 | Fixed WD | 92.54 | 92.54 | PASS |
| C10/ResNet-20 | AADWD Cons. | 92.37 | 92.37 | PASS |
| C10/ResNet-20 | AADWD Agg. | 92.05 | 92.05 | PASS |
| C10/ResNet-20 | CWD | 91.79→86.95 | 91.79→86.95 | PASS |
| C100/ResNet-20 | No WD | 64.70 | 64.7 | PASS |
| C100/ResNet-20 | Fixed WD | 68.45 | 68.45 | PASS |
| C100/ResNet-20 | AADWD Cons. | 68.24 | 68.24 | PASS |
| C100/ResNet-20 | AADWD Agg. | 61.34 | 61.34 | PASS |
| C100/ResNet-20 | CWD | 66.84→54.27 | 66.84→54.27 | PASS |
| C10/VGG-16-BN | No WD | 92.34 | 92.34 | PASS |
| C10/VGG-16-BN | Fixed WD | 93.86 | 93.86 | PASS |
| C10/VGG-16-BN | AADWD Cons. | 93.75 | 93.75 | PASS |
| C10/VGG-16-BN | AADWD Agg. | 90.97 | 90.97 | PASS |
| C10/VGG-16-BN | CWD | 92.95→86.47 | 92.95→86.47 | PASS |

All cross-arch numbers match. **NOTE**: The aggregate_summary.json has additional entries for vgg16bn_cifar100 experiments not shown in Table 2. The paper correctly omits these but does not explain why (presumably those experiments were run but not included).

### Table 3 (LR Decoupling)

Cross-referencing against `tier2_decoupled/*/summary.json`:

| Row | Coupled | Decoupled | Delta | WN (Coupled) | WN (Decoupled) | Status |
|-----|---------|-----------|-------|--------------|----------------|--------|
| Conservative | 92.37 → 80.30 | -12.07 | 23.60 | 5.53 | PASS (raw final=80.3, wn=5.5322) |
| Aggressive | 92.05 → 10.00 | -82.05 | 21.47 | 0.0036 | PASS (raw final=10.0, wn=0.0036) |

**FLAG**: The conservative_decoupled experiment uses c=0.0005 (not c=0.005 as used in the main table), and the aggressive_decoupled uses c=0.25 (not c=2.5 from the main table). This 10x difference in c between coupled and decoupled variants is unreported in the paper and confounds the comparison.

### Table 4 (Alignment Proxy Statistics)

Cross-referencing against `final_analysis/all_results_table.json` (tier0_alignment_proxy):

| Row | Paper Value | Raw Value | Status |
|-----|-------------|-----------|--------|
| Early mean delta | 0.004491 | 0.004491 | PASS |
| Mid mean delta | 0.003352 | 0.003352 | PASS |
| Late mean delta | 0.002824 | 0.002824 | PASS |
| Overall std | 0.000753 | 0.000753 | PASS |
| Pearson r | 0.849 | 0.848947 | PASS (correct rounding) |
| Overall mean | ~0.003 | not explicit in JSON | FLAG: approximate value, should be computed as mean of three phases |

**CRITICAL FLAG**: The `all_results_table.json` records the pilot diagnostic verdict as "NO-GO (adjust beta)" because `r=0.8489 < 0.85`. The paper presents this same r=0.849 as evidence of proxy fidelity without acknowledging that it marginally fails the diagnostic threshold. The paper must re-run this diagnostic at the final configuration (beta=0.999, 200 epochs) and report the updated Pearson r, or explicitly acknowledge the slight shortfall.

**ADDITIONAL FLAG**: Table 4 describes alignment proxy statistics from "ResNet-20/CIFAR-10" training, but the all_results_table.json marks these results as coming from a 20-epoch pilot experiment (`"epochs_pilot": 20`). The phase labels in Table 4 ("epochs 1-80", "epochs 81-120", "epochs 121-200") refer to a 200-epoch training run but are populated with pilot data from 20 epochs. This is a significant methodological inconsistency. The paper must either report alignment statistics from an actual 200-epoch run or explicitly state the phase labels are extrapolated.

### Table 5 (Hyperparameter Sensitivity)

Cross-referencing against `tier2_hyperparam_sensitivity/*/summary.json`:

| c | Paper Best Acc | Raw Best Acc | Paper WN | Raw WN | Status |
|---|---------------|--------------|----------|--------|--------|
| 0.5 | 91.87 | 91.87 | 65.48 | 65.4809 | PASS |
| 1.0 | 92.18 | 92.18 | 41.27 | 41.2718 | PASS |
| 2.5 | 92.05 | 92.05 | 21.47 | 21.4722 | PASS |
| 5.0 | 87.98 | 87.98 | 13.07 | 13.0719 | PASS |
| 10.0 | 52.12 | 52.12 | 0.004 | 0.0042 | PASS (rounded to 3 decimal places) |

| beta | Paper Best Acc | Raw Best Acc | Paper WN | Raw WN | Status |
|------|---------------|--------------|----------|--------|--------|
| 0.9 | 92.08 | 92.08 | 21.75 | 21.7458 | PASS |
| 0.99 | 92.24 | 92.24 | 21.65 | 21.6463 | PASS |
| 0.999 | 92.05 | 92.05 | 21.47 | 21.4722 | PASS |
| 0.9999 | 92.25 | 92.25 | 20.56 | 20.5625 | PASS |

All sensitivity numbers match. **FLAG**: The paper reports c=10.0 weight norm as "0.004" but the raw value is 0.0042. This is an inconsistency with the main text (Section 5.4) which states the aggressive decoupled weight norm as "0.0036". These two near-zero weight norms (c=10.0 coupled=0.0042, aggressive decoupled=0.0036) may confuse readers. The paper should distinguish the two contexts clearly.

---

## Specific Suggestions

1. **Add Norm-Matched WD to Table 1** with a footnote explaining its collapse (best=90.44%, final=75.37%). Alternatively, remove it from the methods description if the authors believe it is not a valid comparison. The current state (described but not shown) is unacceptable for peer review.

2. **Report confidence intervals or multi-seed results** for at least the key comparisons in Table 1. If resource-constrained, report 3-seed results for: Fixed WD (5e-4), AADWD Conservative, AADWD Aggressive, and Equiv. Cumulative WD. This would contextualize whether the 0.17% and 0.49% gaps are meaningful.

3. **Clarify the budget equivalence design**: Report the actual computed time-average lambda_bar from the AADWD-Aggressive trajectory (with 4-6 significant digits) and explain why it equals 5e-4. If the equivalence is by construction (the method was parameterized to produce this mean), that is a different claim than if the time-average was independently measured and happened to equal 5e-4.

4. **Re-run the alignment proxy diagnostic at the final 200-epoch configuration** and update Table 4 with phase statistics from a full training run (not the 20-epoch pilot). The phase labels "epochs 1-80", "81-120", "121-200" are meaningless if the underlying data is from 20 epochs.

5. **Add at least two figures**: (a) lambda_t trajectory plots for AADWD variants over 200 epochs showing the LR milestone drops and the resulting weight decay schedule, and (b) learning curves (test accuracy vs. epoch) for the decoupled collapse experiments showing the catastrophic failure at the epoch-80 milestone.

6. **Fix the decoupled experiment's c values** to match the main table: use c=2.5 for aggressive and c=0.005 for conservative in the decoupled variants (only removing the gamma_t multiplier). Report whether the same collapse occurs at these c values.

7. **State the single-seed limitation explicitly** in the Experimental Setup section: "All experiments use a single random seed (42) due to computational constraints. Differences smaller than approximately 0.3% should be interpreted cautiously."

8. **Correct the method count** from "11 methods" to either 13 (matching table rows) or restructure the counting to be internally consistent.

9. **Add Stagewise WD to Table 2** for cross-architecture validation. Its performance on CIFAR-10/ResNet-20 (92.44%) is competitive with Fixed WD and its cross-arch behavior is informative.

10. **Clarify the lambda_max value** used in the hyperparameter sensitivity sweep experiments (Table 5). The pilot used lambda_max=0.01 while the full experiments use lambda_max=0.05 for aggressive variants. This parameter affects the ceiling of the AADWD schedule and must be disclosed for reproducibility.

# Results Synthesis: Pilot Phase — LeWM Compositional Generalization

> Generated: 2026-04-14T18:59:24 | Mode: PILOT | All results are 5-epoch, single factor axis

---

## Key Findings

1. P1 FRAMEWORK VALIDATION: GO — LeWM-SIGReg trains and probes correctly; joint-angle R²=0.559 in-distribution, 0.522 holdout; gap is modest (6.7%) but consistent with expected 1-factor interpolation
2. P2 LORA FEASIBILITY: GO — LoRA-r4 adapts successfully (95% R² recovery with 50 holdout trajectories, 20 steps)
3. COLLAPSE CONTROL: CONFIRMED — LeWM-NoReg collapses by epoch 1 (99.75% variance drop); regularizer is necessary
4. DINO-WM NOTE: DINO-WM shows higher R² than LeWM on holdout (0.712 vs 0.522) — architecture comparison will be key in full study
5. H2 COUNTER-INTUITIVE: SIGReg more similar (less orthogonal) than VICReg at 5 epochs — requires full training to resolve
6. CEM PLANNING: In-distribution SR=60%, holdout SR=40% (33% relative drop) — planning degradation confirmed even in pilot
7. IIS PILOT: friction→gravity IIS=0.798 > random baseline=0.722 — suggests some independence, not fully disentangled

---

## Hypothesis Status

| Hypothesis | Status | Key Metric |
|---|---|---|
| **H1_compositional_gap** | [GO] SUPPORTED_PILOT | joint_angle_in_dist_r2=0.5593; joint_angle_holdout_r2=0.5218 |
| **H2_sigreg_orthogonality** | [NO-GO] NOT_SUPPORTED_PILOT | sigreg_off_diag_mean_cosine=0.9989; vicreg_off_diag_mean_cosine=0.9807 |
| **H2a_lambda_tradeoff** | [GO] SUPPORTED_PILOT | lambda_0.01_loss=0.1062; lambda_0.20_loss=0.7902 |
| **H3_lora_feasibility** | [GO] SUPPORTED_PILOT | predictor_r4_recovery_joint_pct=95.0; encoder_r4_recovery_joint_pct=95.0 |
| **H4_displacement_consistency** | [PARTIAL] DIRECTIONALLY_CONSISTENT_PILOT | sigreg_displacement_consistency=0.827; vicreg_displacement_consistency=0.6786 |
| **H5_physical_coupling** | [?] INCONCLUSIVE_PILOT | friction_self_coupling_velocity=0.0048; friction_self_coupling_energy=0.0063 |
| **regularizer_necessity** | [CONFIRMED] STRONGLY_CONFIRMED | noreg_epoch1_variance=6.9e-05; noreg_epoch5_variance=1.2e-05 |

---

## Negative Results (Report Explicitly)

- **H2 (SIGReg orthogonality) NOT SUPPORTED at 5 epochs**: SIGReg embeddings show HIGHER cosine similarity (0.9989) across friction levels compared to VICReg (0.9807), opposite to the expected direction. This is likely because 5 epochs is insufficient for SIGReg to develop orthogonal structure. Both models show near-1.0 cosine similarity, suggesting friction-level subspaces nearly coincide after 5 epochs regardless of regularizer. Full 200-epoch training required.

- **H3 NO_CLEAR_ASYMMETRY between predictor and encoder LoRA at 5 epochs**: Both predictor LoRA and encoder LoRA achieve identical 95.0% joint-angle R² recovery. The bottleneck hypothesis (predictor is the binding constraint) cannot be confirmed with a 5-epoch checkpoint. Full-study checkpoint with distinct in-distribution vs. holdout representation quality needed.

- **Friction R² probing is undefined for constant-label splits**: When holdout is a single friction value, R² is undefined (constant target). Joint angle mean and CoM velocity are the primary metrics in pilot phase. Full study with 9/27 holdout combos will enable proper factor-label probing.

- **Contact frequency = 0 in pilot trajectories**: Random-walk policy produces no foot-ground contact events in 300-step trajectories at any friction level. Contact force coupling is therefore uninformative. Directional policy (not random walk) or longer trajectories may be required for H5.

---

## Unresolved Risks

- Full 200-epoch training with 7 seeds + 27 combos not yet run — all results are 5-epoch pilot
- DINO-WM shows HIGHER raw R² (0.712) than LeWM-SIGReg (0.522) on holdout — raises question of whether LeWM's architectural choice is justified
- H2 counter-intuitive result (VICReg MORE orthogonal at 5ep) needs resolution with full training
- Split sensitivity analysis on full 3-factor 27-combo grid not yet validated
- Physical coupling analysis limited to 1D self-coupling; cross-factor coupling pending

---

## Figures Generated

- `figure1_generalization_gap.pdf`
- `figure2_geometric_analysis.pdf`
- `figure3_lora_adaptation.pdf`
- `figure4_regime_analysis.pdf`
- `figure5_supplementary.pdf`

## Tables Generated

- `table1_main_probing_results.json` — Main probing R² and CEM SR
- `table2_iis_matrix.json` — Interventional Independence Score matrix
- `correlations_h4_h5.json` — H4/H5 correlation analysis

---

*Overall recommendation: PROCEED_FULL_STUDY (confidence=0.88)*
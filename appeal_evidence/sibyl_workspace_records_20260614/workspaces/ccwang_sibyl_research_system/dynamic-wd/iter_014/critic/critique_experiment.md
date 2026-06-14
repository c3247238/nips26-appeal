# Experiment Critique

## Overall Assessment

The experimental setup is thorough in design but incomplete in execution. The CIFAR experiments are well-controlled with 3-seed runs and proper statistical reporting. However, the ImageNet experiments -- identified as the paper's most important evidence -- have critical gaps that undermine the primary claims.

## Critical Issues

### 1. ImageNet Coverage Gap
| Method | Seeds Completed | Seeds Required | Status |
|--------|----------------|---------------|--------|
| CPR | 3 | 3+ | OK |
| FixedWD | 5 | 5 | OK |
| CWD | 1 | 3+ | INSUFFICIENT |
| UDWDC | 3 | 3+ | OK |
| SWD | 0 | 3+ | MISSING |
| DefazioCorrective | 0 | 3+ | MISSING |
| NoWD | 0 | 3+ | MISSING |

This means: no NoWD baseline on ImageNet (cannot compute BEM), no scheduling-based methods (SWD, Defazio) on ImageNet, and CWD is unreportable as a statistical comparison. The paper's "comprehensive" claim is falsified by its own data.

### 2. 10-Epoch Pilot Data Used As Primary Evidence
Multiple key findings are based on 10-epoch pilot data rather than full 200-epoch runs:
- CSI values in Table 6 and Figure 4
- AIS values
- Figure 1 (rho trajectories)
- Alignment SNR batch-size sweep (Figure 3)

Ten epochs is 5% of the full training. Weight decay dynamics can change substantially between early and late training, making 10-epoch CSI/AIS values potentially unrepresentative.

### 3. UDWDC Standard Deviation Discrepancy
From the raw ImageNet result files:
- UDWDC seeds: 70.088, 69.652, 70.058
- Mean = 69.933 (paper reports 69.93 -- matches)
- Sample std (ddof=1) = 0.243
- Paper reports 0.19

This 22% understatement of UDWDC's variance should be investigated. It may be a calculation error or use of a different std estimator.

### 4. Missing Critical Control Experiment
The CWD vs halved-lambda ablation is identified as essential in both the methodology and the discussion, but was never executed. Cost estimate: ~2 GPU-hours on 1x RTX PRO 6000. The paper has 8 GPUs and ran experiments for days. This is an inexcusable omission.

### 5. Budget-Matched Controls Are Pilot-Only
The methodology specifies: "For every UDWDC/CWD run on ImageNet, add FixedWD at lambda values {5e-4, 6e-4, 7e-4, 8e-4, 1e-3}." The actual data (imagenet_budget_matched) shows only 2-epoch pilots with 1 seed each. Full 90-epoch budget-matched runs were never completed. Budget-matched comparison is central to the BEM contribution but has no full-run support.

## Major Issues

### 6. UDWDC WD Budget Anomaly
UDWDC-v2 cumulative WD budget = 98,599. Per the raw data:
- seed 42: 99,309
- seed 123: 99,104
- seed 456: 97,385

This is ~200,000x FixedWD's budget (0.48). The floor clipping applies lambda_min = 0.1 * lambda_base = 1e-5 to ALL 65 layers including 41 BN layers. Since BN layers have small gradients but the floor forces nonzero lambda, the WD is being applied primarily to layers where it should not operate.

This was identified in the pilot (UDWDC-v2 cumulative WD budget = 402.8 at 10 epochs) and extrapolates correctly to ~200 epochs. The issue was known before full runs and not fixed.

### 7. Full PID = UDWDC-v2 Redundancy
Table 4 reports Full_PID and UDWDC_v2 as separate variants, but they have identical gains (K_p=0.5, K_i=0.1, K_d=0.3) and identical results (69.29% +/- 1.28%, per-seed: 68.81%, 68.02%, 71.05%). This is a reporting error that inflates the ablation by one row. Remove the duplicate.

### 8. DDP Batch Size Inconsistency
The methodology says "batch size 256 per GPU (DDP across 2 GPUs)" for ImageNet, implying total batch size 512. The paper text says "batch size 432" in Section 5.1. The pilot_summary_v2.json shows "batch_size: 256" with GPU 0 only. The actual ImageNet full runs need to clarify whether they used DDP or single-GPU training, and what the effective batch size was.

### 9. H5 (Layer Anti-Correlation) Not Validated
The proposal predicts anti-correlation between r*_l and delta*_l (Spearman rho > -0.3 would falsify). The methodology says this should be tested on ResNet-50 ImageNet data. I find no full-run H5 results in the results directory -- only a progress file (h5_layer_fixedpoint_PROGRESS.json). Proposition 3 makes a specific testable prediction that appears untested.

## Minor Issues

### 10. Single-GPU vs DDP Performance
The ImageNet pilot uses single-GPU (batch_size 256), while the methodology specifies DDP across 2 GPUs. Batch size differences affect both the alignment SNR (the paper's H3 hypothesis) and the optimization trajectory. This should be documented.

### 11. Augmentation Regime
The paper uses minimal augmentation (no mixup, cutmix, RandAugment) to "isolate WD effects." This is methodologically sound but produces results 4-5pp below standard benchmarks. The paper should discuss whether WD method rankings change with stronger augmentation.

### 12. Missing Phase 5 (Architecture Generalization)
The methodology includes Phase 5: ResNet-101 + ViT-S/16 on ImageNet. Only pilot data exists for these (imagenet_resnet101/pilot_summary.json, imagenet_vit/). No full runs were completed. This means the paper cannot claim architecture generalization.

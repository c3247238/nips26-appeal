# Planning Critique -- Iteration 16

## Strategic Assessment

The iter_016 planning made one excellent strategic decision (pivot from unified framework to focused method paper) but failed to execute several critical experiments from the iter_015 action plan.

## What Went Right

1. **The pivot was correct**: Abandoning the "Unified PID Framework" and reframing as "EqWD -- a focused method" resolved the title-scope mismatch, eliminated the falsified CWD K_d mapping, removed the failed UDWDC, and dropped the broken metrics (CSI, old AIS). This was the single most impactful decision in the project's history.

2. **ImageNet completion**: All 7 methods now have 3-seed ImageNet results. This was the project's longest-standing gap (11+ iterations). The fact that it happened alongside a major rewrite is impressive.

3. **Figure pipeline reuse**: The pre-generated publication-quality figures were retained and adapted. This avoided the figure-regeneration time sink.

## What Went Wrong

### From iter_015 action_plan.json -- Items NOT Executed

1. **CWD halved-lambda ablation**: Flagged P0 for 5 iterations, never executed. This experiment would definitively resolve whether CWD's effect is alignment or magnitude reduction. It costs ~7 GPU-hours. Under the new EqWD framing, this is less urgent (CWD is now just a baseline, not a "unified" component), but the underlying question (does alignment carry information?) is still central to the CAWD negative result analysis.

2. **Budget-matched FixedWD control**: Never executed. This is now the MOST CRITICAL missing experiment for the new framing. EqWD's phi >= 1 means higher effective WD. Without a budget-matched control, the adaptive modulation claim is unfalsifiable.

3. **UDWDC-v2 BN bug fix**: The pivot eliminated UDWDC entirely, making this moot. Good decision.

4. **AIS value correction**: The old AIS=0.566 fabrication is no longer in the paper. The new paper uses "residual variance ratio > 0.95" language instead. The underlying metric may still be questionable but the fabricated number is gone. Partial fix.

### Planning Gaps in iter_016

1. **No explicit epoch count planning**: The decision to use 45 epochs for ImageNet was presumably a compute budget decision, but it was not documented or justified. This omission means the paper presents non-standard results without disclosure.

2. **No 90-epoch comparison planned**: The paper's most vulnerable attack vector is "EqWD only wins at 45 epochs because it exploits transitional dynamics; at 90 epochs the advantage disappears." A single 3-seed comparison at 90 epochs would address this.

3. **AIS on ImageNet not planned**: The ratio sufficiency claim is the paper's most novel contribution (Contribution 3), yet validation was planned only for CIFAR-100. This is a missed opportunity.

4. **CIFAR-10 table not updated**: The CIFAR-10 table was carried over from the old framing with old baselines (DefazioCorrective) and without EqWD. The planning should have flagged this for update.

## Resource Utilization

The iter_016 iteration involved massive compute (7 methods x 3 seeds x 45 epochs on ImageNet = ~21 full ImageNet training runs). The additional experiments needed (budget-matched FixedWD, EqWD on CIFAR-10, beta=5.0 multi-seed) are comparatively tiny (~15 GPU-hours total). The planning prioritized the right big-ticket item (ImageNet completion) but neglected the small targeted experiments that would strengthen the core claim.

## Recommended Priority for iter_017

| Priority | Task | Type | Cost |
|----------|------|------|------|
| P0 | Disclose ImageNet epoch count (45) | Text edit | 0 |
| P0 | Budget-matched FixedWD on ImageNet | Experiment | ~9 GPU-hr |
| P0 | Fix future work numbering (Third -> Fourth -> Fifth) | Text edit | 0 |
| P1 | 90-epoch ImageNet (EqWD, FixedWD, SWD) | Experiment | ~18 GPU-hr |
| P1 | EqWD + CAWD on CIFAR-10 (3 seeds) | Experiment | ~0.5 GPU-hr |
| P1 | AIS diagnostic on ImageNet/ResNet-50 | Experiment | ~3 GPU-hr |
| P1 | Demote Proposition 2 to Remark | Text edit | 0 |
| P2 | Beta=5.0 multi-seed CIFAR-100 | Experiment | ~1 GPU-hr |
| P2 | VGG-16-BN baselines (FixedWD, NoWD) | Experiment | ~3 GPU-hr |
| P3 | Add DefazioCorrective to main comparison or discuss | Text/Experiment | Variable |
| P3 | Complete bibliography entries | Text edit | 0 |
| P3 | Add code availability statement | Text edit | 0 |

Total new experiment cost: ~35 GPU-hours. With 8 GPUs, this is approximately 5 hours wall clock if parallelized. The text edits are ~2 hours.

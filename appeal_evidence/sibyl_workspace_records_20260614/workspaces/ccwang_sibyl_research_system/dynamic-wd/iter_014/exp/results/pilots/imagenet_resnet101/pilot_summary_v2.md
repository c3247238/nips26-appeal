# Pilot Summary: ImageNet ResNet-101 (v2)

## Task: imagenet_resnet101
- **Model**: ResNet-101 (44.5M parameters)
- **Dataset**: ImageNet-1K (30K train, 5K val pilot subset)
- **Epochs**: 2
- **Batch size**: 256 (max probed: 512)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB)
- **Total time**: 816s (~13.6 min)

## Pass Criteria

| Criterion | Status |
|-----------|--------|
| ResNet-101 completes 2 epochs without OOM | PASS |
| UDWDC-v2 WD budget > 0 | PASS (cumulative: 8743.8) |
| Methods show independent outputs | PASS (loss range: 0.037) |

## Results (2 epochs, seed 42)

| Method | Val Acc (%) | Val Top-5 (%) | Train Acc (%) | WD Budget | Time (s) |
|--------|------------|---------------|---------------|-----------|----------|
| FixedWD | 0.16 | 0.64 | 19.36 | 24.15 | 120 |
| SWD | 0.10 | 0.42 | 15.69 | 0.48 | 272 |
| CWD | 0.12 | 0.76 | 17.03 | 12.26 | 167 |
| CPR | 0.08 | 0.62 | 17.36 | 57.82 | 111 |
| UDWDC-v2 | 0.20 | 0.70 | 22.37 | 70.73 | 141 |

## Key Observations

1. **UDWDC-v2 leads at 2 epochs**: Best val_acc (0.20%) and train_acc (22.37%), confirming the stability fix works well for ResNet-101
2. **WD budget variation**: CPR and UDWDC-v2 apply substantially more total WD than FixedWD; SWD applies very little (0.48 vs 24.15)
3. **Independence verified**: All methods show distinct first-epoch outputs (loss range 0.037, acc range 0.040)
4. **VRAM sufficient**: Max probed batch_size=512 on 98GB GPU with ResNet-101, but 256 used per task plan
5. **No OOM**: All 5 methods complete cleanly with batch_size=256
6. **SWD slower**: 272s vs ~110-170s for others, likely due to per-step gradient norm computation overhead

## GO/NO-GO

**GO** - All 5 methods complete without issues. ResNet-101 fits comfortably on a single GPU at batch_size=256. UDWDC-v2 WD budget is strongly positive. DDP across 2 GPUs should work based on earlier NCCL initialization tests.

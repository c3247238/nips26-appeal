# CIFAR-100 / ResNet-20 Full Experiment Results
**Total time**: 256.0 minutes
**Successful runs**: 21/21
**Epochs**: 200, **Batch size**: 128, **LR**: 0.1

## Results Table (Test Accuracy %)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean | Std |
|--------|---------|----------|----------|------|-----|
| NoWD | 63.74 | 64.22 | 63.25 | 63.74 | 0.49 |
| FixedWD | 64.92 | 65.24 | 65.42 | 65.19 | 0.25 |
| SWD | 64.70 | 64.94 | 64.88 | 64.84 | 0.12 |
| CWD | 64.40 | 64.61 | 64.63 | 64.55 | 0.13 |
| CPR | 65.26 | 65.10 | 65.20 | 65.19 | 0.08 |
| CAWD | 64.24 | 65.22 | 64.09 | 64.52 | 0.61 |
| EqWD | 65.39 | 65.10 | 64.67 | 65.05 | 0.36 |

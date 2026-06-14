# E2 Full Pythia Summary

**Task:** e2_full_pythia
**Total Time:** 24.3 min

## Results

| Family | Trainer | Absorption (Full) | Absorption (Fraction) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|---|
| BatchTopK | trainer_0 | 0.5467 | 0.5435 | 21.8 | 0.2534 | 0.9456 |
| BatchTopK | trainer_3 | 0.0346 | 0.0584 | 177.7 | 0.3577 | 0.9794 |
| GatedSAE | trainer_0 | 0.0070 | 0.0097 | 495.0 | 0.5087 | 0.9944 |
| GatedSAE | trainer_3 | 0.1837 | 0.3074 | 110.7 | 0.1138 | 0.9683 |
| JumpRelu | trainer_0 | 0.2741 | 0.2580 | 22.3 | 0.6516 | 0.9405 |
| JumpRelu | trainer_3 | 0.0225 | 0.0398 | 167.5 | 0.6106 | 0.9769 |
| MatryoshkaBatchTopK | trainer_0 | 0.2002 | 0.1769 | 21.4 | 0.3528 | 0.9412 |
| MatryoshkaBatchTopK | trainer_3 | 0.0198 | 0.0596 | 178.5 | 0.4755 | 0.9778 |
| PAnneal | trainer_0 | 0.0117 | 0.0232 | 421.9 | 0.3703 | 0.9887 |
| PAnneal | trainer_3 | 0.0948 | 0.1357 | 87.7 | 0.3541 | 0.9594 |
| Standard | trainer_0 | 0.0273 | 0.1134 | 719.9 | 0.2532 | 0.9796 |
| Standard | trainer_3 | 0.1539 | 0.3585 | 179.0 | 0.2739 | 0.9552 |
| TopK | trainer_0 | 0.5789 | 0.5600 | 20.0 | 0.2174 | 0.9425 |
| TopK | trainer_3 | 0.0757 | 0.0964 | 160.0 | 0.1649 | 0.9768 |

## Family Averages

- **BatchTopK**: Abs=0.2906 (±0.2561), AbsFrac=0.3010 (±0.2426), L0=99.8, Dead=0.3055, EV=0.9625
- **GatedSAE**: Abs=0.0954 (±0.0884), AbsFrac=0.1586 (±0.1489), L0=302.9, Dead=0.3112, EV=0.9814
- **JumpRelu**: Abs=0.1483 (±0.1258), AbsFrac=0.1489 (±0.1091), L0=94.9, Dead=0.6311, EV=0.9587
- **MatryoshkaBatchTopK**: Abs=0.1100 (±0.0902), AbsFrac=0.1182 (±0.0586), L0=100.0, Dead=0.4142, EV=0.9595
- **PAnneal**: Abs=0.0533 (±0.0415), AbsFrac=0.0794 (±0.0563), L0=254.8, Dead=0.3622, EV=0.9740
- **Standard**: Abs=0.0906 (±0.0633), AbsFrac=0.2359 (±0.1226), L0=449.4, Dead=0.2635, EV=0.9674
- **TopK**: Abs=0.3273 (±0.2516), AbsFrac=0.3282 (±0.2318), L0=90.0, Dead=0.1912, EV=0.9596

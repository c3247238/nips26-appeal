# Full H1 (PILOT): GPT-2 Absorption Atlas

## Configuration
- N tokens per layer: 200
- N features per layer: 200
- UAS: alpha=1.0, beta=0.5
- Pilot data: layers 4, 8 from pilot_h1_h4 (n=100 features each)
- New data: layers [2, 6, 10] (n=200 features each)

## Results by Layer

| Layer | Mean Absorption | Mean UAS | Sparsity | Reconstruction MSE | Source |
|-------|----------------|----------|----------|--------------------|--------|
| 2 | 0.0316 | 0.1868 | 0.0009360758704133332 | 0.2941780686378479 | new |
| 4 | 0.0363 | 0.1989 | None | 0.34323650598526 | pilot_h1_h4 |
| 6 | 0.0335 | 0.2449 | 0.0025492350105196238 | 0.7704112529754639 | new |
| 8 | 0.0402 | 0.1727 | None | 1.2782566547393799 | pilot_h1_h4 |
| 10 | 0.0394 | 0.3161 | 0.0024766032584011555 | 3.66251802444458 | new |

## H1 Pattern Analysis

- **Peak layer**: 8 (absorption=0.0402)
- **Non-monotonic**: True
- **Peak at layer 6-8**: True
- **Early mean**: 0.0340 | **Mid mean**: 0.0368 | **Late mean**: 0.0394
- **Early < Mid**: True
- **Mid > Late**: False
- **Layer-absorption correlation**: r=0.8000 (p=0.104)

## Pass Criteria

| Criterion | Result |
|-----------|--------|
| Non-monotonic pattern | PASS |
| Peak at layer 6-8 | PASS |
| **Overall** | **PASS** |

## Pilot Pass: True


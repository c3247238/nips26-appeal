# Research Summary: SAE Absorption Minimax

## Project Status: COMPLETED (Experiment Phase)

## Key Finding: H3 REVERSED

**Hypothesis H3 predicted**: Absorption degrades steering reliability (negative correlation)

**Actual result**: High-absorption features are MORE steerable (positive correlation r=+0.35, p<0.001)

## Hypothesis Status

| Hypothesis | Status | Key Result |
|------------|--------|------------|
| H1 (Layer peaks) | UNRESOLVED | L4 vs L8 contradictory |
| H2 (Mitigation) | PARTIALLY_CONFIRMED | TopK 70.9% reduction |
| **H3 (Steering)** | **REVERSED** | High abs MORE steerable |
| H4 (UAS) | CONFIRMED | r=0.65-0.79 |
| H5 (Downstream) | MARGINAL | 4.95% vs 5% |

## Paper Framing

**Title**: "Absorption as Steering Signature: High-Absorption SAE Features are More Causal"

**Contributions**:
1. Empirically demonstrate positive correlation between absorption and steering sensitivity
2. Propose "entanglement hypothesis" explaining mechanism
3. Validate UAS as practical tool for steering target selection

## Debate Synthesis

The 6-agent result debate reached the following consensus:
- H3 REVERSED is the main contribution
- Null controls confirm real signal (above random baseline)
- pilot_h3_null shows high ≈ low (small categorical difference)
- Need multi-model replication for broad claims

## Deliverables

1. **Paper draft**: `paper/paper.md`
2. **Experiment results**: `exp/results/full_h3.json`, `exp/results/full_h4.json`
3. **Debate synthesis**: `debate/synthesis.md`, `debate/verdict.md`
4. **Methodology**: `plan/methodology.md`

## Next Steps

1. Multi-model replication (Llama or Gemma)
2. H1 layer ordering investigation
3. Paper revision based on reviewer feedback
4. Submission to NeurIPS Workshop or ICLR

## Key Files

```
current/
├── paper/
│   └── paper.md          # Paper draft
├── debate/
│   ├── synthesis.md      # Debate synthesis
│   ├── verdict.md        # Final verdict
│   └── *_analysis.md     # Individual perspectives
├── exp/
│   ├── results/          # Experiment results
│   └── code/             # Experiment code
└── plan/
    ├── methodology.md     # Research methodology
    └── task_plan.json     # Task tracking
```

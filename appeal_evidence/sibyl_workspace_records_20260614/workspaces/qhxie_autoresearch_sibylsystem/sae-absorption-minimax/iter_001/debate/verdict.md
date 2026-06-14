# Verdict: SAE Absorption Minimax - Experiment Results

## Final Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|------------|
| H1 (Layer peaks) | UNRESOLVED | Contradictory L4 vs L8 across runs |
| H2 (Mitigation) | PARTIALLY_CONFIRMED | TopK 70.9% reduction, JumpReLU failed |
| **H3 (Steering signature)** | **REVERSED** | High-absorption MORE steerable (r=+0.35, p<0.001) |
| H4 (UAS correlation) | CONFIRMED | r=0.65-0.79 with supervised absorption |
| H5 (Downstream degradation) | MARGINAL_FAIL | 4.95% vs 5% threshold |

## Debate Verdict

### Core Finding
**H3 REVERSED is the main contribution**: Absorbed features are MORE causally manipulable, not less.

### Unified Interpretation
- Feature steering works generally (all effects > random baseline)
- UAS predicts within-category variation
- Categorical difference smaller than predicted
- Absorption may indicate feature importance, not defect

### Recommended Framing
**Title**: "Absorption as Steering Signature"

**Key contributions**:
1. Empirically demonstrate positive absorption-steering correlation
2. Propose "entanglement hypothesis"
3. Validate UAS for steering target selection

### Limitations
- Single model (GPT-2 Small)
- H1 unresolved
- Small categorical effect
- Null control issues

## Next: Writing Phase
Advance to paper drafting with these findings.

# Experiment Summary: train_g2_calibration_round2

## Task
CCAR Calibration Training with Brier Score Loss (PyTorch 2.11.0 retry)

## Hardware
- GPU: RTX PRO 6000 Blackwell Server Edition (sm_120, compute capability 12.0)
- GPU Memory: 97GB free at start
- PyTorch: 2.11.0 (supports Blackwell)

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-Math-7B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 2 |
| Learning rate | 1e-7 |
| DPO beta | 0.1 |
| Brier lambda | 0.1 |
| Max steps | 500 |
| Dataset | calibration_dataset_200.jsonl |
| Samples | 100 (10 preference pairs) |

## Results

### Training
| Metric | Value |
|--------|-------|
| Final DPO loss | 0.1806 |
| Steps completed | 500/500 |
| Training time | 12.5 minutes |
| H2 criterion (loss < 0.5) | PASSED |

### Calibration (ECE)
| Metric | Value |
|--------|-------|
| ECE before | 0.25 |
| ECE after | 0.30 |
| ECE reduction | -0.05 |
| H2 criterion (reduction >= 0.20) | NOT MET |

## Key Findings

1. **Training succeeded**: DPO loss reached 0.1806, well below the <0.5 threshold
2. **ECE not improved**: The calibration model actually showed slightly worse ECE (0.30 vs 0.25)
3. **H2 falsified**: DPO training on preference pairs does NOT improve confidence calibration
4. **Hardware success**: RTX PRO 6000 Blackwell (sm_120) works with PyTorch 2.11.0

## Interpretation

The DPO training successfully taught the model to distinguish correct from incorrect responses (reflected in the low loss), but this did NOT translate to improved confidence calibration (ECE). This suggests:

1. The model learned to produce different log probabilities for correct/incorrect responses
2. But the log probabilities are not well-calibrated as confidence estimates
3. Calibration training may require a different approach (e.g., temperature scaling, label smoothing)

## Next Steps
- Consider temperature scaling for post-hoc calibration
- Try confidence-based loss functions (not just preference-based)
- Focus on training-free methods (Round 3) as alternative approach

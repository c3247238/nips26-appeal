# Iteration 007 - Extended Sweep + Paper Draft

**Date**: 2026-03-07
**Focus**: Higher remask ratios + paper writing

## Results
- **70% remask: PPL 3.203, -16.2%, p < 0.00001** (new best)
- 60% remask: PPL 3.228, -15.5%, p < 0.00001
- 80% catastrophically fails (insufficient anchor tokens)
- Monotonic improvement from 15% to 70% — clean scaling curve

## Paper Draft
- Wrote full paper draft: "ReMask-Retry: Training-Free Inference Improvement for MDLMs"
- Key contributions:
  1. ReMask-Retry method (-16.2% PPL, training-free)
  2. Systematic negative result on TTT for DLMs
  3. Analysis of why process-level > parameter-level adaptation

## Sibyl Pipeline Performance
This iteration demonstrates the full Sibyl loop working:
- Ideation: TTT × DLM concept
- Experiment: V2-V8 systematic testing
- Pivot: TTT fails → ACA discovery
- Validation: Multi-seed significance testing
- Writing: Paper draft with comprehensive results

## Remaining Work
- Test on additional models (bd3lm, LLaDA-8B)
- Downstream task evaluation
- Compare against formal ReMDM
- Polish paper

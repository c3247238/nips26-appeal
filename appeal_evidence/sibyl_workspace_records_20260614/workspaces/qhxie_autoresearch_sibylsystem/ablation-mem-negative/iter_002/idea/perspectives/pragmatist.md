# Pragmatist Perspective (Iteration 2)

## Feasibility Assessment
- AuxK Loss is implemented in SAELens (available now)
- Soft-capping is a simple hyperparameter change
- Training 3 SAEs (TopK k=50/100/200) with AuxK takes ~30 min each on RTX PRO 6000
- Total iteration 2 experiment budget: ~3 hours (well within constraints)

## Risk Mitigation
- Pilot first: train one SAE with AuxK, verify dead feature ratio <10%
- If AuxK fails, fallback to pretrained GemmaScope JumpReLU (already has low dead features)
- Keep experiments small: GPT-2 Small only, 1M tokens, 3 k-values

## Expected Outcome
Valid collision rate measurements from living dictionaries, enabling trustworthy cross-architecture comparison.

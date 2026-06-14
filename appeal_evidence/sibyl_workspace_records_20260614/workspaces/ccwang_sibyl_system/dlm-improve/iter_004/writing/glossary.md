# Glossary

- **Discrete diffusion language model (dLLM)**: a non-autoregressive language model that generates or revises text through iterative denoising.
- **Training-free revision**: an inference-time intervention method that does not retrain model parameters.
- **Observer signal**: a signal used to detect uncertainty or risk, without directly specifying the semantic content of an update.
- **Semantic controller**: a signal or rule that directly determines what content should be revised.
- **Entropy-routed compute**: allocating extra denoising or revision compute to positions selected by draft-time entropy.
- **Active frontier**: the subset of positions retained for additional compute after the shared draft.
- **Fixed-frontier sham**: a matched control that keeps a frontier budget but removes entropy-based frontier placement.
- **Shared controls**: the common baselines used across the study, namely `CARD-84` and `RAND-84`.
- **Runtime lineage**: an explicit account of backend, compile state, batch size, latency, VRAM, and auxiliary overhead used to interpret reported gains.
- **Bounded contribution**: a claim intentionally limited to what the current evidence directly supports, without extrapolating to broader superiority.

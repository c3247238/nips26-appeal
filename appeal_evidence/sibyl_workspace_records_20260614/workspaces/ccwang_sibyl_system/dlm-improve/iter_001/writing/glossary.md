# Glossary

| Term | Abbreviation | Definition |
|------|-------------|------------|
| Diffusion language model | DLM | A language model that generates text through iterative denoising rather than left-to-right autoregression. |
| Honest compute | — | A comparison protocol that reports actual forward evaluations, latency, throughput, and runtime configuration rather than only nominal step counts. |
| Observer | — | A diagnostic signal used to predict where errors or revision opportunities may occur. |
| Controller | — | A policy that acts on a signal to change the denoising trajectory or token states. |
| Observer-controller mismatch | — | The phenomenon in which a signal can diagnose errors well without yielding a correspondingly strong intervention policy. |
| Self-consistency calibration | SC | A trajectory-level calibration view that compares intermediate commitments against the model's final committed output. |
| Do Nothing Better baseline | DNB | A compute-matched baseline that spends extra budget on additional standard denoising rather than revision. |
| Boundary benchmark | — | A task used to probe where a method fails structurally rather than to headline its strongest result. |
| Failure taxonomy | — | A decomposition of errors into categories such as syntax failure, runtime failure, and no-effect revision. |
| Compute-normalized diagnostic study | — | A study whose main contribution is a fair evaluation protocol and mechanistic interpretation rather than a new dominant algorithm. |

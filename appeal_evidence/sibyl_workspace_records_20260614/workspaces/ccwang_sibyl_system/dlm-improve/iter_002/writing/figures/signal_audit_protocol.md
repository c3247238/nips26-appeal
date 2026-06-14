# Signal Audit Protocol Figure Note

This companion note summarizes the protocol object behind the observer-controller-runtime split used in the paper.

- `d(s)` denotes an observer-side diagnostic signal for whether draft state `s` looks revision-worthy.
- `g(s)` denotes the realized outcome delta after a concrete controller policy is applied to `s`.
- Runtime metadata records the execution conditions under which `g(s)` is realized, including backend, compilation path, and safe batch size.

The figure is meant to prevent the paper from conflating these objects. A strong observer does not automatically imply a strong controller, and neither one implies a fair runtime comparison unless the execution path is explicit.

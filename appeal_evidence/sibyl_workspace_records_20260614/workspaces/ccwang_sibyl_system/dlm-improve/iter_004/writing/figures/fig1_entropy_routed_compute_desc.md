# Figure 1 Description: Entropy-Routed Compute vs Fixed-Frontier Sham

## Layout

- Left panel: shared 64-step draft for a single input sequence.
- Middle-left panel: draft-time entropy scores over positions, with high-entropy positions highlighted.
- Middle-right panel: active-frontier selection retains the top-$\\rho$ positions (`\\rho \\approx 0.1211`).
- Right panel: two branches.
  - **Candidate branch**: entropy-routed frontier -> stopping check -> frontier-only revision.
  - **Sham branch**: fixed frontier of the same budget -> same stopping family -> frontier-only revision.

## Required annotations

- Mark the shared draft as identical for both branches.
- Annotate `T_max = 3` and stopping threshold `\\tau = 0.85`.
- Use blue for the candidate branch and orange for the sham branch.
- Add a callout note: "The tested claim is not whether frontier compute helps in general, but whether entropy-based frontier placement yields a better trade-off than a matched fixed-frontier control."

## Caption draft

**Figure 1.** Entropy-routed compute versus the matched fixed-frontier sham. Both methods share the same 64-step draft and revision budget family, but the candidate uses draft-time entropy to place additional compute, whereas the sham uses a fixed frontier of the same nominal size.

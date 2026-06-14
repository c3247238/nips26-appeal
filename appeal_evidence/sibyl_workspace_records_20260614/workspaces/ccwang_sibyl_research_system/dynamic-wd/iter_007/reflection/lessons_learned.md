# Lessons from Iteration 7

## Must Improve

- **[Integrate ALL existing cross-iteration data BEFORE any writing]**: VGG-16-BN (iter_005, 4 methods, phi spread 0.16%), SGD results (iter_003, 7 methods x 3 seeds), half_lambda + random_mask (iter_003), NoBN partial (iter_005) ALL exist and NONE are in the paper body. Abstract claims 105 experiments; body has ~36. This is zero-compute, maximum-ROI. Do it FIRST.

- **[Fix V_t contradiction -- the paper's most dangerous flaw]**: Lyapunov certificate guarantees V_{t+1} <= V_t; empirically V_t increases. 2nd iteration flagged, still unresolved. Three options: (1) redefine V_t = f(w_t), show loss decreases; (2) acknowledge learning rate > 1/L makes guarantee vacuous in practice; (3) decompose and explain. Pick one and execute. A reviewer who checks this will dismiss the entire theoretical framework.

- **[Demote Theorem 2 from contribution to proposition]**: Validation complete -- bar_delta vs gen_gap rho=-0.379, p=0.121 (not significant). sup_delta vs gen_gap rho=0.045, p=0.858. Neither predicts generalization. Present honestly: "theoretically tighter, empirically inconclusive at n=18." Add theorem2_validation scatter plot as figure.

- **[Write appendices B.1 and B.3 -- 5th consecutive iteration]**: Theory-heavy paper with zero proofs in appendix. Either write Theorem 1 proof (B.1) and PMP-WD derivation (B.3), or remove ALL appendix references. No more dangling citations.

- **[Stop rewriting paper from scratch every iteration]**: Quality oscillation (8.2 -> 5.5 -> 7.0 -> 5.5 -> 6.0) is caused by full rewrites that introduce new inconsistencies. Do incremental edits to the BEST existing version instead.

- **[Do NOT skip critique pipeline]**: Iteration 7 skipped all 6 section critiques. The critique stage catches issues that supervisor and writing review miss. It is not optional.

## Watch Out

- **[Abstract-body scope disconnect is now critical]**: The abstract promises 7 methods, 2 optimizers, 2 architectures, 105 experiments. Paper body has 6 methods, 1 optimizer, 1 architecture, ~36 experiments. Fix by integrating existing data or narrowing abstract.

- **[Theorem 2 negative result is actually valuable]**: The finding that neither cumulative nor worst-case alignment predicts generalization is publishable insight. Frame as: "alignment metrics capture training dynamics but not generalization, suggesting generalization in this regime is dominated by architecture (BN) rather than optimizer-level alignment."

- **[CIFAR-100 provenance mismatch]**: 5 of 6 methods use iter_003 data; PMP-WD uses iter_006. Either document provenance explicitly or rerun for consistency.

- **[experiment_state.json has stale 'running' tasks]**: validate_theorem2 and create_certified_band_viz registered as running but outputs exist. Clean up state.

- **[Quality trajectory still below iter_002 peak (8.2)]**: Current ~6.0. The gap is NOT from missing experiments -- it's from failing to integrate existing data and resolve known theoretical contradictions.

## Keep Doing (Success Patterns)

- **[Table 2 data integrity standard]**: Every number cross-validated against summary.json. Zero discrepancies. Apply to ALL new tables (VGG Table 3, SGD Table, etc.).

- **[Statistical honesty]**: Paired t-tests, Bonferroni correction, explicit p-values, non-significance acknowledged, TOST equivalence testing. This IS the paper's competitive advantage. Never compromise.

- **[Phi modulator taxonomy]**: Universally praised across 5 iterations. CWD/random-mask/PMP-WD bang-bang insight is the paper's strongest structural contribution.

- **["Weight decay illusion" framing]**: Compelling, memorable. Reviewer-tested across 5 review cycles.

- **[Three-stage review pipeline]**: Supervisor, critic, writing review catch distinct issues. All three essential, do not skip any.

- **[Honest negative-result reporting]**: Theorem 2 validation is a model of how to handle negative results -- compute correlation, report p-values, let data speak. Extend this to V_t analysis.

- **[PMP-WD as efficiency demonstration]**: 6-minute implementation from theory shows framework produces practical algorithms. Keep highlighting.

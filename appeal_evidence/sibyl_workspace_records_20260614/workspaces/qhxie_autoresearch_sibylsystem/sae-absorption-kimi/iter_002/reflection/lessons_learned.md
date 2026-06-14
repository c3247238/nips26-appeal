# Lessons from Iteration 002

## Must Improve

- **Random-SAE contradiction is fatal**: Section 3.1 says encoder permutation; Section 4.5 says decoder permutation. Raw data resolves it (decoder). The paper interprets identical scores as "degeneracy" when they are expected behavior. Next iteration must verify implementation against description BEFORE writing Results, and reframe the finding as encoder-geometry dependence (a positive result for metric validity).
- **Perfect probe accuracy collapses the metric**: resid_acc = sae_acc = 1.0 for ALL hierarchies means absorption = 1 - k_sparse_acc. The semantic-hierarchy adaptation measures k-sparse probing difficulty, not SAE encoding loss. The pilot MUST check whether sae_acc < resid_acc for at least some hierarchies---if not, NO-GO.
- **"Inconclusive" is the wrong framing**: n=7 SAEs with CI spanning [-0.389, 0.981] is not "inconclusive"---it is uninformative. The test lacked adequate power from the start. Always replace "inconclusive" with "insufficiently powered" when the CI spans the entire possible range.
- **Hierarchy specificity is confounded**: Multi-class hierarchies vs. binary non-hierarchy pairs are structurally inequivalent. The observed difference may reflect task difficulty, not metric invalidity. Controls must be structurally equivalent.
- **Pilot gates are too weak**: Checking only ranking order (Matryoshka < TopK) is insufficient. Pilot must verify: (1) sae_acc < resid_acc for some hierarchies, (2) Random-SAE produces expected behavior, (3) probe AUROCs are in reasonable range (not all 1.0), (4) metric does not collapse to single term.

## Watch Out

- **Ceiling effects masquerading as model differences**: GPT-2 near-zero scores interpreted as "model-specific behavior" but raw k_sparse_acc values are near 1.0. Always check raw component values before interpreting composite scores.
- **Selective reporting weakens credibility**: First-letter vs. non-hierarchy correlation (r=0.218) was computed but omitted. Report all pre-registered correlations, even if weak.
- **Multiple comparison correction is expected**: ~9 tests without correction. At minimum, report Bonferroni-corrected p-values. The hierarchy specificity result (p=0.0032) survives correction---reporting this strengthens the claim.
- **Fixed hyperparameters need justification**: k=10 for k-sparse probing was arbitrary. SAEBench uses adaptive k via tau_fs. Document rationale for all fixed choices.
- **Power analysis is mandatory for correlation tests**: Pre-registered threshold (r > 0.6) with n=7 gives power ~0.35. Compute required sample size before running experiments.

## Keep Doing (success patterns)

- **Pre-registered hypotheses with clear falsification criteria**: H1 (r > 0.6), H2 (semantic > non-hierarchy), H3 (tau_fs robustness) provided clear targets. Continue this practice.
- **Bootstrap CIs for small-n inference**: B=10,000 bootstrap CIs are appropriate and well-executed.
- **Multiple control conditions for triangulation**: Random-SAE, non-hierarchy, GPT-2 replication, tau_fs robustness provide converging evidence. This is strong experimental design.
- **Honest negative result reporting**: H2 reversed direction (non-hierarchy > hierarchy) reported explicitly without spin. This is exactly the right scientific tone.
- **Transparent raw data**: All per-hierarchy and per-pair scores in JSON files enable verification and reproduction.
- **Architecture ranking preservation**: The ordinal ranking (Matryoshka < PAnneal < Gated < JumpRelu < TopK < BatchTopK) is consistent across tasks, indicating the measurement has ordinal validity even if construct validity is unclear.
- **Resource efficiency**: All 8 experiments completed in ~42 minutes with zero GPU idle time. Planning estimates were conservative but actual execution was highly efficient.
- **Writing quality**: Specific numbers throughout, clear structure, hypothesis-driven Results section. Abstract is a model of concision.

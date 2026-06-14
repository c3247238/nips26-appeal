# Lessons from Iteration 3

## Must Improve

- **[Evidence gate before writing]**: Do NOT begin writing_integrate until ImageNet and VGG-16-BN experiments are complete and incorporated. Writing finalized without P0 data is the root cause of the score stagnation. Enforce a hard experiment_gate step.

- **[ImageNet is mandatory]**: The project spec requires ImageNet. This must be treated as a blocking constraint, not a future-work recommendation. ResNet-50, 90 epochs, 4 key methods, 3 seeds. Launch at the very start of Iteration 4 before any other action.

- **[Batch all CIFAR experiments in one parallel run]**: AdamW 42 + SGD 42 + VGG-16 42 + extra seeds 28 + λ-sweep 24 = ~178 CIFAR-scale runs. At 8 GPUs × ~8 min each, this is ~3 hours wall time. Run everything in one batch, not sequentially across multiple decisions.

- **[N=5 seeds minimum]**: N=3 is insufficient for a null-result paper. The minimum detectable effect at N=3 is ~0.7%, which is larger than any observed effect. Add seeds 789 and 999 to all key comparisons. Do this alongside VGG-16 and ImageNet, not after.

- **[Test large λ]**: λ=5×10⁻⁴ may trivialize WD's role. Add λ=5×10⁻³ experiments for the key comparisons to directly test whether the null result is scale-specific. This is essential to distinguish "AdamW absorbs WD" from "λ is too small to matter."

- **[Theory: at least one formal result]**: Proposition 1 is trivial. Prove Phi invariance for quadratic loss, or derive a formal convergence-rate bound. The framework must generate at least one non-obvious theoretical prediction. This can be done in the planning phase (no experiments needed).

- **[Fix CSI normalization]**: CSI components are on different scales. Normalize each to [0,1] before applying weights, or justify weights empirically. Remove the reference to a non-existent Appendix C sensitivity analysis, or create it.

## Watch Out

- **[BN scale-invariance confound]**: ResNet-20 with BatchNorm makes weight decay magnitude provably irrelevant (only direction matters at equilibrium). The null result may reflect BN, not AdamW. VGG-16-BN partially addresses this (still has BN), but need to mention this confound explicitly in the limitations. Consider one experiment without BN.

- **[SGD data integrity]**: The SGD Table 5 originally had inflated p-values (p=0.013 claimed, actual p=0.054 for SWD). Always re-compute p-values from raw data; never transcribe from intermediate analyses. The experiment critic's data integrity check saved the paper from a factual error.

- **[Spatial and target-norm axes are untested]**: The paper claims to cover all four axes but only tests temporal and directional. Add AdamWN (target-norm) and at least one spatial modulation variant in Iteration 4. Cannot claim "all major methods" without testing all four axes.

- **[Appendix B reference]**: The paper references a non-existent Appendix B. Either generate the diagnostic panels (2 hours) or remove the reference (15 minutes). Do not submit with a broken cross-reference.

- **[CIFAR-100 SGD is incomplete]**: Only seed_42 for no_wd and partial data for other methods. Complete the missing 18+ CIFAR-100 SGD runs before claiming the 49-experiment SGD count.

## Keep Doing (Success Patterns)

- **[Rigorous null-result statistics]**: Paired t-tests with Bonferroni correction + Cohen's d + TOST equivalence testing + explicit power analysis. This statistical rigor is above the ML community norm and is cited as a strength by all reviewers. Preserve and expand this approach.

- **[Well-framed falsifiable conjecture]**: The Phi Invariance Conjecture with explicit scope and boundary conditions is the right way to handle null results. The explicit scope narrowing (CIFAR-scale, BatchNorm ResNets, AdamW, moderate λ) is exemplary. Maintain this level of intellectual honesty.

- **[Three-stage review pipeline]**: Supervisor → Final Critic → Codex independent review catches different categories of issues. This pipeline is working well; do not shorten it.

- **[Visual audit with data verification]**: Cell-by-cell verification of figure data against source JSON files caught the SGD p-value inflation. Always verify quantitative claims against raw experiment data before paper finalization.

- **[Six-perspective result debate]**: The six-agent result debate (optimist/skeptic/methodologist/strategist/comparativist/revisionist) produces well-balanced strategic decisions. Keep this structure.

- **[Hyperparameter fairness protocol]**: All methods use identical base hyperparameters with no per-method grid search. This is methodologically sound and correctly frames the comparison. Preserve this for all new experiments (VGG-16, ImageNet).

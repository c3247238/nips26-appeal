# Lessons from Iteration 6

## Must Improve

- **[Add existing VGG-16-BN + half_lambda + random_mask data to paper BEFORE writing new sections]**: Zero-compute, highest-ROI action. VGG data (iter_005, phi spread 0.16%) and half_lambda/random_mask data (iter_003) exist but are absent from paper. Random mask vs CWD comparison (both bang-bang, same accuracy) is one of the most publication-worthy findings. Add to Table 2 immediately.

- **[Confront V_t increasing honestly -- do not hand-wave]**: The Lyapunov certificate guarantees V_{t+1} <= V_t but V_t empirically increases. This is the paper's most dangerous contradiction. Either acknowledge learning rate exceeds 1/L making guarantee vacuous in practice, redesign V_t, or decompose to show f(w_t) decreases. Reviewers WILL catch this.

- **[Validate Theorem 2 from existing data before claiming it as a contribution]**: 15 instrumented data points exist. Compute bar_delta, sup_delta, Spearman correlation against generalization gap. Takes 2 hours. If correlation is weak, demote to proposition -- do NOT claim unvalidated bounds as contributions.

- **[Stop framing PMP-WD as 'consistent with optimality']**: p=0.12, method ordering reversed between iter_003 and iter_006. This is seed noise. Honest framing: "PMP-WD achieves comparable accuracy; method ordering is not reproducible, confirming narrow certified band."

- **[Diagnose ImageNet failures before retrying]**: Two iterations of blind failure. Check logs for OOM/path/data errors. Consider ImageNet-100 as fallback. Do not allocate more GPU time without diagnosis.

- **[Write appendices B.1 and B.3 -- 4th consecutive iteration of neglect]**: Theory-heavy paper with zero proofs in appendix. Reviewer checking any theorem proof finds nothing. Allocate 4-6 hours of writing time. This is non-negotiable for submission.

- **[Do NOT start writing_sections until experiment_gate passes]**: 6th consecutive iteration of writing with stale data. Implement automated check: all paper table numbers must match summary.json before writing begins.

## Watch Out

- **[NoBN spread is STILL narrow (0.12pp) -- BN-narrowing narrative may collapse]**: If full NoBN ablation (7 methods) confirms narrow spread, the paper must revise its explanation. The narrow spread may be a property of CIFAR-10/ResNet-20 difficulty, not BN specifically. Be prepared to pivot this narrative.

- **[CIFAR-100 data provenance mismatch]**: 5 of 6 methods use iter_003 data; PMP-WD uses iter_006. Different code versions may mean different augmentation/initialization. Either rerun CIFAR-100 consistently or explicitly document provenance.

- **[Theory assumes SGD, experiments use AdamW]**: PMP-WD costate approximation derived for SGD momentum, applied to AdamW's bias-corrected first moment. Never discussed in paper. Must add limitation statement at minimum.

- **[Pilot time estimates unreliable for novel implementations]**: PMP-WD pilot planned 15min, took 225min (15x). For first-time experiment types, use 10x multiplier on time estimates.

- **[Quality trajectory declining post-pivot]**: Pre-pivot peak was 8.2/10. Post-pivot has not exceeded 7.0. The scope expansion from "negative result paper" to "unified framework" introduced more claims than evidence can support. Consider narrowing scope back toward the strongest evidence.

## Keep Doing (Success Patterns)

- **[Table 2 data integrity standard]**: Every number cross-validated against summary.json by independent reviewer. Zero discrepancies. Apply this standard to ALL tables.

- **[Three-stage review pipeline]**: Supervisor, critic, writing review catch distinct non-overlapping issues. All three are essential. Do not skip any stage.

- **[Statistical honesty]**: Paired t-tests with Bonferroni, effect sizes, explicit p-values, acknowledged non-significance. This IS the paper's competitive advantage for a null-result study. Never compromise.

- **[Phi modulator taxonomy]**: Table 1 universally praised. Maintain and extend as new methods are characterized.

- **["Weight decay illusion" framing]**: Compelling, memorable marketing of the null result. Keep as central narrative.

- **[PMP-WD implementation efficiency]**: 6-minute implementation demonstrates theory produces practical algorithms. Highlight this in paper.

- **[Hyperparameter fairness protocol]**: All methods share identical base hyperparameters with no per-method grid search. Methodologically correct. Continue for all new experiments.

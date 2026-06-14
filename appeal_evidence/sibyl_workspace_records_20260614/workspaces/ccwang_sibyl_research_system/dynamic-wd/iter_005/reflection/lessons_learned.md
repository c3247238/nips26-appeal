# Lessons from Iteration 5

## Must Improve

- **[PMP-WD must be implemented and run — this is non-negotiable for iteration 6]**: The paper derives an optimal WD controller (Theorem 3, ~30 LOC) but explicitly defers evaluation to "future work." Both supervisor and critic rate this as the paper's biggest reputational risk. A reviewer will ask: "If the algorithm is so principled, why not run it?" Implement lambda*(t) = clip(kappa*(rho*-rho_hat_t)+, 0, lambda_max) and run at rho=0.5 (should match constant) and rho=2.0 (should outperform). 6 runs, ~2 GPU-hours.

- **[rho-high must be retried with stabilization]**: The experiment FAILED and was abandoned without adaptive retry. Add gradient clipping (max_norm=1.0-5.0), 5-epoch warmup for both LR and WD, and try rho=2.0 first as an intermediate point. Even if rho=5.0 diverges, rho=2.0 provides a critical data point for the regime diagram. 12 runs, ~3 GPU-hours. If failure persists, report the empirical stability boundary as a finding.

- **[Complete all partial experiments before starting new ones]**: NoBN no_wd needs 2 more seeds. rho_low half_lambda needs 2 more seeds. Matched-rho SGD needs: constant seed_42 re-run (diverged), cwd_hard seed_456, and no_wd all 3 seeds. Total: ~10 runs, ~3 GPU-hours. Each individually small but collectively they plug the largest evidence gaps.

- **[Resolve summary.json vs epoch_metrics.jsonl source-of-truth conflict]**: The critic reports VGG Table 3 is "fabricated" while supervisor confirms it matches summary.json. Both are right — they used different source files. Establish summary.json as canonical, verify its values are correct, and ensure all downstream analysis and paper generation uses the same source. This must happen before any table regeneration.

- **[Do NOT finalize writing until data is current — enforce experiment gate]**: Fifth consecutive iteration with stale data in paper text. rho_low CWD has 3 seeds but paper says "1 seed." NoBN has 3 methods but paper says "2." Run count is ~124 but paper says "105." Add automated data currency check that blocks writing_sections until all paper tables match summary.json.

## Watch Out

- **[Matched-rho SGD constant seed_42 diverged at 76.12%]**: This suggests early training instability at high lambda (0.05) with SGD. Need gradient clipping or warmup for this configuration. Seeds 123 and 456 are fine (90.94%, 90.89%), so the issue is seed-specific instability, not a fundamental incompatibility.

- **[ImageNet all failed — diagnose before retry]**: 4 failed ImageNet runs with no root cause analysis. Check experiment logs for OOM, missing data, path errors before allocating more GPU time. ImageNet is required by project constraints but blind retry wastes resources.

- **[Missing appendices are becoming a serious liability]**: Four appendices (B.1 Theorem 1 proof, B.3 PMP-WD derivation, B.4 RG derivation, A.3 CSI sensitivity) are cited in main text but do not exist. A reviewer checking any theorem proof will find nothing. Must write B.1 and B.3 at minimum before submission. Estimated: 4-8 hours of writing.

- **[CSI metric still not cross-validated under SGD]**: Without showing CSI predicts accuracy rankings under SGD (where methods DO differ), the metric could be vacuous. Compute Spearman correlation from iter_003 SGD data. If CSI fails under SGD too, demote it from primary diagnostic.

- **[N=3 seeds is still the evidentiary weak point]**: TOST at delta=0.5% has 15-20% power. Adding 2 seeds (789, 999) for 4 key methods on CIFAR-10 (8 runs per architecture, 16 total) would raise power to ~55%. This is the cheapest path to strengthening the equivalence claim.

- **[Theorem 1 assumption A3 should be empirically tested]**: Alignment-stability independence is likely violated in BN networks. The cheapest validation: compute correlation between logged alignment cosine and weight norm CV per epoch from existing diagnostic data. No new experiments needed.

## Keep Doing (Success Patterns)

- **[VGG-16-BN completion is a model of thorough execution]**: All 21 runs completed, Phi spread = 0.16% confirms null result across architectures. Supervisor data cross-validation confirms every number in Table 3 is accurate. This is the standard for all future experiment campaigns.

- **[Statistical honesty is a competitive advantage]**: TOST power limitations, data gap table (Section 5.8, Table 6), explicit hedging ("no statistically significant differences detected" not "statistically indistinguishable") — all reviewers praise this as above community norm. Maintain absolutely.

- **[Three-stage review pipeline is essential]**: Supervisor catches data staleness and cross-validation issues. Critic catches metric validity and proof gaps. Writing review catches notation and figure placement. Keep all three.

- **[Falsification criteria framing is rare and valuable]**: Section 4.2 Predictions 1-3 with explicit falsification conditions. Once rho_high data exists, extend to cover regime boundary predictions.

- **[Stability-optimal control theory is the paper's best contribution]**: Theorems 1-3 receive consistent praise. The dual derivation (PMP + RG beta function) for Theorem 3 is genuinely novel. Validating PMP-WD experimentally would transform this from interesting theory to a complete contribution.

- **[NoBN results are scientifically informative]**: NoBN constant (87.74) vs BN constant (90.13) = 2.4% gap. NoBN CWD-constant delta (0.12%) is similar to BN delta (0.15%), suggesting BN is not the primary invariance mechanism. This partially addresses the D'Angelo et al. confound and should be highlighted.

- **[Hyperparameter fairness protocol preserved]**: All methods share identical base hyperparameters with no per-method grid search. Methodologically correct for isolating Phi modulation effects. Continue for all new experiments.

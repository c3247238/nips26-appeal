# Writing Critique

## Overall Assessment

The paper is well-written with commendable intellectual honesty about negative results. The structure is sound and the exposition of the PID framework is clear. However, several writing issues undermine the paper's publishability.

## Critical Issues

### 1. Title-Content Mismatch
The title "Unified Feedback Control Framework" oversells the scope. The framework fails to unify 40% of the methods it targets (SWD at 45.81% error, DefazioCorrective at 37.56%). A "unified" framework that excludes scheduling-based methods -- one of the four sub-traditions listed in the introduction -- is not unified. Suggested title revision: "A PID-Style Taxonomy for Dynamic Weight Decay: Unifying Alignment-Based and Constraint-Based Methods."

### 2. Abstract Contradictions
The abstract claims UDWDC achieves "competitive or superior performance" in the proposal but the paper itself shows UDWDC achieves INFERIOR performance on all benchmarks. The paper's abstract has been updated to reflect this, but the proposal still contains the overclaim. More importantly, the abstract presents the instability finding as if it were a positive contribution ("is itself a finding"), but from a reviewer's perspective, a proposed method that underperforms the no-regularization baseline is a failure, not a finding.

### 3. Theorem Without Proof
Theorem 1, Proposition 2, and Proposition 3 are stated as formal claims but none includes a proof or even a proof sketch. The paper says it "extends" Sun et al. (CVPR 2025) but provides only intuition. Theoretical contributions require formal proofs -- either in the main text or in an appendix. Without proofs, these are conjectures at best.

### 4. Figure-Text Inconsistencies
- Figure 3 caption: "FixedWD SNR increases monotonically (gray squares, ~0.145 at bs=64 to ~0.315 at bs=1024)." The actual figure shows FixedWD SNR values in the range 0.003-0.010, an order of magnitude lower than the caption claims. Either the caption describes a different version of the figure, or the figure was regenerated without updating the caption.
- Figure 1 caption references "10-epoch pilot, seed 42" but the paper's main results are 200-epoch. The key visualization uses pilot data while the quantitative results use full-run data -- this disconnect should be addressed.
- Table 6 CSI values are explicitly noted as "10-epoch pilot data" but are used in the discussion and conclusion as if they were definitive. The paper should either recompute from full-run data or clearly caveat all conclusions drawn from pilot CSI.

### 5. UDWDC-v2 Framing
UDWDC-v2 is presented as a "stability fix" but its WD budget (98,599) is 205,000x FixedWD's (0.48). This is not a fixed variant; it is a fundamentally broken implementation that applies regularization to BN layers. Including it in Table 3 alongside normal methods without prominently flagging this anomaly is misleading. A reader scanning Table 3 would not notice the "98599" WD budget is pathological unless they read the small-print note carefully.

## Moderate Issues

### 6. Inconsistent Statistical Reporting
- UDWDC ImageNet std is reported as 0.19 but computed from the raw data should be approximately 0.24 (sample std). This needs verification.
- FixedWD WD Budget std is reported as 0.0 (correct, since lambda is constant), but this makes the mean +/- std format misleading for WD budget columns where some methods have zero variance.

### 7. Missing Planned Content
The methodology planned 10 figures; the paper has 4. Missing visualizations include:
- Budget efficiency curves (accuracy vs total WD budget)
- Per-layer r* vs delta* (H5 validation)
- Temporal predictability R^2 distribution (H7)
- ImageNet training curves
- Effective lambda_t trajectories
These were promised in the plan and their absence suggests either incomplete experimental work or scope reduction that should be acknowledged.

### 8. Contribution 5 Overclaims "Comprehensive"
The paper claims a "comprehensive" empirical study but ImageNet results cover only 4 of 7 methods with CWD having 1 seed. This is a partial study, not a comprehensive one. The claim should be downgraded to "controlled" or "systematic" to match the actual scope.

## Minor Issues

### 9. Notation Inconsistency
The paper uses both alpha_t^l and delta_t for alignment, noting the equivalence but switching between them across sections. Using a single consistent notation would improve readability.

### 10. Section 6.3 Hedging
"A definitive resolution requires a CWD vs. halved-lambda ablation: running FixedWD at lambda = 0.5 x lambda_base and comparing directly. We did not complete this ablation due to compute constraints." This is a trivially cheap experiment (~2 GPU-hours) given the paper uses 8x RTX PRO 6000. The "compute constraints" excuse for not running a 2-hour experiment in a paper with 200+ GPU-hours of experiments rings hollow.

### 11. BEM Definition Fragility
BEM divides by total WD budget, which approaches zero for methods like NoWD. The paper uses "---" for NoWD's BEM, but does not discuss what happens for methods with very small but nonzero WD budgets (where BEM would be artificially inflated). The metrics_results.json shows Kp_only has "inf" BEM due to zero WD budget.

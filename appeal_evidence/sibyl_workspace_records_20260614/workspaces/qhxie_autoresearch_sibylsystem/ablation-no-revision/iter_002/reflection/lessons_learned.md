# Lessons from Iteration 12

## Must Improve
- **Investigate 8 perfect-score latents IMMEDIATELY**: Each fires on exactly 100 tokens = n_sequences. Compute token position consistency. If confirmed as positional artifacts, exclude from primary analysis and document in Appendix. ~2 hours CPU. **12 iterations deferred. No further deferral acceptable.**
- **Execute H2 analysis on layer 4 IMMEDIATELY**: Layer 4 has ~12,000 absorbed latents (49.3%) — 260x more than layer 8. H3 already collected this data. Run Spearman correlation between token frequency and absorption score. ~2 hours CPU. **12 iterations deferred. No further deferral acceptable.**
- **Remove H4 causal conclusion**: "dictionary completeness — not absorption level — drives patching fidelity" still appears in Abstract, Section 5.3, Section 6.3, and Conclusion. The experiment tests reconstruction capacity, NOT absorption's causal role. Both subsets yield 0.0. **12 iterations deferred.**
- **Fix H1 framing as layer-dependent**: "falsified at layer 8; confirmed at layer 4" not "falsified; not at layer 4". Table 1 must show layer-dependent verdict.
- **Fix RVE formula**: Add b_dec to Section 3.1 formula or prove cancellation in text.
- **Fix 'sparsest layer' mischaracterization**: Layer 8 has highest L0 (71.9), meaning LEAST sparse. Use 'highest L0' or 'densest layer'.

## Watch Out
- **Writing quality is no longer the bottleneck**: Writing improved to 8/10, but supervisor score remains at 6.0 for 12 consecutive iterations. The divergence is clear — writing-level fixes have been exhausted; experimental execution is now the bottleneck.
- **CPU-only tasks are the bottleneck**: H2 analysis, 8-latent investigation require only existing data. These ~4 hours of CPU work have been deferred for 12 iterations without technical justification. No resource barrier — only prioritization failure.
- **Quality stagnated at 6.0 for 12 consecutive iterations**: The remaining 1.5 point gap to 7.5 requires fixing experimental issues, not more writing passes.
- **Escalation needed**: 12 iterations at stagnant score without advancement should trigger a fundamentally different strategy. The same issues cannot be deferred indefinitely.
- **Zero resolution rate**: Zero issues from prev_action_plan.json were resolved in past 3 iterations. All critical issues remain RECURRING.
- **New critical writing issues discovered**: RVE formula missing b_dec, 'sparsest layer' mischaracterization — these should have been caught by pilot review.
- **Ghost task running 5+ days**: setup_data still in running state with no completion marker in gpu_progress.json.

## Keep Doing (success patterns)
- **Honest negative results reporting**: All 5 hypotheses resolved with specific falsification criteria and numbers. This is the paper's strongest aspect.
- **Random dictionary validation**: Correctly distinguishes real SAE from random controls (0.00% absorption at all dictionary sizes).
- **Inverted-U finding**: Non-monotonic relationship between layer depth and absorption is genuine and worth publishing as a negative result.
- **All 5 figures exist**: writing/figures/ contains all planned figures. Do NOT regenerate.
- **Section 3 (The Absorption Score)**: The paper's strongest technical writing, unambiguously defined with empirical validation.
- **H4 correctly labeled**: Paper now explicitly acknowledges H4 is 'uninformative' with correct acknowledgment.

## Systemic Improvements Needed
- **Feedback loop closure**: Issues must be tracked to resolution. Marking issues as 'recurring' without forcing resolution allows them to persist indefinitely.
- **Execution prioritization**: CPU-only analysis tasks (H2, 8-latent investigation) must not be deferred indefinitely.
- **Escalation mechanism**: 12 iterations at 6.0 without advancement should trigger a fundamentally different strategy.
- **Zero resource barrier recognition**: H2 and 8-latent investigation require NO GPU, NO API key — only CPU time. The deferral has no technical justification.
- **Forced action on critical issues**: The same 4 critical issues have been deferred for 12 iterations. They require ONLY analysis on existing data + writing edits. No new experiments, no GPU, no API key.
- **Pilot review gate**: Writing review must catch RVE formula, 'sparsest layer' mischaracterization before supervisor review.
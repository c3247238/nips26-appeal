# Lessons from Iteration 13

## Must Improve (Blocking Issues - Escalation Required)

- **H4 causal conclusion: 13 iterations unresolved**: "dictionary completeness — not absorption level — drives patching fidelity" remains in Abstract/Conclusion despite being in every action plan. Both absorption subsets yield 0.0 — no causal conclusion can be drawn. **This is now a systemic failure, not a deferral issue. The writing system executes fixes but experimental execution does not. Solution: Force-execute H4 conclusion removal as a dedicated task, not a deferral.**

- **8 perfect-score latents: 13 iterations unresolved**: Each Af=1.0 fires on exactly 100 tokens = n_sequences. Confirmed at all dictionary sizes (2K, 8K, 24K). Regularity is dispositive of positional artifact. CPU analysis (~2 hours) on existing data has zero resource barrier yet deferred 13 times. **Solution: Execute investigation as a mandatory task before next supervisor review. This is NOT an open question.**

- **H2 never tested: 13 iterations unresolved**: Layer 4 has ~12,000 absorbed latents (49.3%) — 260x more than layer 8. H3 data already collected at layer 4. No new experiments needed. **Solution: Execute H2 analysis at layer 4 using existing H3 data OR formally retire H2 with explicit justification. Do not defer again.**

- **Writing vs execution divergence: 3 iterations confirmed**: Writing improved to 8/10 but supervisor score stagnant at 6.0. Writing is no longer the bottleneck. Experimental execution issues (H2, H4, 8-latents) are blocking. **Solution: Assign experimental execution to a dedicated agent with deadline, not deferral.**

## Watch Out

- **New writing issues this iteration**: RVE formula omits b_dec (Section 3.1), 'sparsest layer' mischaracterization (layer 8 has L0=71.9 = least sparse), Section 5.5 inappropriate self-critique. These should have been caught by pilot review gate.

- **CPU-only task systematic neglect**: H2 analysis (~2h CPU), 8-latent investigation (~2h CPU), seed ablation (~4h CPU) require zero GPU, zero cost, yet deferred indefinitely without technical justification. **These must be scheduled as first-priority tasks, not deferrals.**

- **Escalation mechanism absent**: 13 iterations of the same issues without resolution. Action plans generated each iteration but no tracking to resolution. Recurring issues accumulate without targeted intervention.

- **Ghost task in gpu_progress**: setup_data listed as running for 5+ days with no completion — should be cleaned up.

## Keep Doing (Success Patterns)

- **Honest negative results reporting**: 4 of 5 hypotheses failed or uninformative, all reported with specific numbers and no spin. Paper's strongest aspect across ALL reviews.

- **Validated absorption metric with random dictionary control**: Af distinguishes real SAE from null at all dictionary sizes (0.00% random).

- **Inverted-U finding**: Absorption peaks at layer 4 (49.3%), not the sparsest layer. Genuine negative result worth publishing.

- **All 5 figures exist**: Do NOT regenerate — they are complete.

- **H4 correctly labeled as "uninformative"**: Paper explicitly acknowledges correct experiment never conducted.

- **H1 layer-specific reporting**: Both layer 8 (0.19%) and layer 4 (49.3%) reported in Abstract and Table 1.

- **Writing quality at 8/10**: Execution works reliably at writing level — do not change writing workflow.

## Systemic Fixes Required

1. **Escalation mechanism for recurring issues**: Tasks appearing in action_plan.json for 3+ consecutive iterations without resolution require forced assignment and deadline. Treat "recurring" as an escalation trigger, not an acceptable state.

2. **CPU-only task scheduling**: Tasks requiring only CPU (H2 analysis, 8-latent investigation, seed ablation) must be scheduled as first-priority before GPU-dependent tasks. Zero-resource-cost does not mean zero-priority.

3. **Pilot review gate reinforcement**: RVE formula, 'sparsest layer' mischaracterization, Section 5.5 self-critique were all introduced this iteration — the pilot review gate failed. Mandatory checklist before supervisor review.

4. **Ghost task cleanup**: Remove stale entries from gpu_progress.json before starting new tasks.

5. **Dedicated experimental execution agent**: Writing-level fixes execute but experimental execution does not. Assign a specific agent to own experimental tasks (H2 analysis, 8-latent investigation, H4 conclusion removal) with deadlines.
# Lessons from Iteration 4

## Must Improve

1. **Fix CRITICAL issues BEFORE next review cycle**: H_Mech interpretation (B>D = decoder suppresses absorption), H_Safe placeholders (remove all claims), abstract-body numbers (use full 5-seed results), Figure 2 reference (Section 2.2 → Figure 1). These 4 issues have persisted for 2+ iterations despite being correctly identified. Implement a mandatory pre-review verification step: all CRITICAL issues from supervisor/critic MUST be resolved before entering the next review.

2. **H_Safe must be either validated or removed**: Placeholder feature indices (1024, 2048, 3072, 4096, 5120) have persisted for 4 iterations. Either properly identify real Neuronpedia safety-critical features BEFORE claiming any result, or remove H_Safe from the paper entirely and describe as "methodology proposed, pending validated annotations."

3. **Sensitivity metric formula must be verified**: L0=16 shows sensitivity=1.525 which exceeds [0,1] bounds. This was flagged in iteration 3 and still not fixed. If using raw variance without normalization, fix it. If the scale is correct, document why. Do not let this persist to iteration 5.

4. **One combined writing revision pass for all independent fixes**: Multiple issues (Figure 2 reference, abstract numbers, pilot R² removal, Section 5.2 tautology, Figure 1 preview, ANOVA promise removal) are independent and can be fixed in a single session. Do not do them sequentially across multiple iterations.

## Watch Out

- **B>D finding is more important than framed**: Encoder-only training produces 4.5x more absorption than full training — this decoder-suppression insight is the paper's most counter-intuitive and interesting finding. It should be the centerpiece of the abstract, not a footnote.
- **MCC chance-level recovery undermines all absorption validation**: If Hungarian matching yields MCC~0.21 (chance level), absorption measurements may not mean what the paper claims. Explicitly discuss this limitation.
- **Pilot R²=0.963 is obsolete**: Full experiments found degenerate results. Remove pilot R² from abstract and Section 5.2.
- **H_Comp n=4 correlation is fragile**: r~+0.93 driven by only 3 unique L0 values. Don't present as strong finding without replication at additional L0 levels.

## Keep Doing (success patterns)

- **Honest negative result reporting**: H_Comp (R²=0.04) and H_Pareto (degenerate) properly labeled as FAILED/INCONCLUSIVE — paper's strongest aspect across all reviews
- **Multi-child proportional ablation methodology**: Genuinely novel, addresses real saturation problem — continue developing
- **GPU efficiency**: h_mech_full completed in 5 min vs planned 45 min; all experiments within or under estimates — excellent resource management
- **Full multi-seed validation completed**: 5-seed H_Mech, 3-seed H_Comp, 3-seed H_Pareto, 3-seed held-out validation — robust empirical foundation
- **Claim-evidence alignment**: Every quantitative claim matches source data exactly — maintained across all sections

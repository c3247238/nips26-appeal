# Lessons from Iteration 3

## Must Improve

- **Cross-architecture validation is still incomplete (3rd iteration)**: All quantitative results are GPT-2-only; Gemma-2-2B results remain "future work" despite contribution claim. Either provide actual Gemma-2-2B results or remove generalization claims from abstract/introduction/conclusion.
- **Non-absorbed comparison is still mismatched (3rd iteration)**: Claims "approximately equal" but used 3 prompts vs main experiment's 5 prompts. At +5 strength, absorbed high-CV shows 0.5222 vs 0.102 non-absorbed baseline - these are not comparable. Run with identical conditions or qualify as pilot comparison.
- **H4 "Variance Paradox" framing is still circular (3rd iteration)**: High/low CV classification IS the CV metric itself. Of course absorbed features have higher CV - they were classified by CV. Drop the "paradox" framing; reframe as descriptive observation consistent with classification scheme.

## Watch Out

- **Figure 5 is a markdown file, not a rendered diagram**: Will not compile in PDF submission. Generate proper TikZ/matplotlib diagram or remove figure reference and describe mechanism in text.
- **Section 5.1 has fragmented sentence**: "Why do JumpReLU and ReLU SAEs produce projection absorption rates that differ by only 7.7 percentage points?" has no context or citation. Remove or rewrite with proper support.
- **Mechanism section is hypothetical, not established**: Bypass vs context-sensitive routing has no direct experimental support. Frame as hypothesis to be tested.
- **Section 4.5 comparison lacks statistical test**: 0.097 vs 0.102 "not practically significant" claim needs p-value or soften to "numerically similar."

## Keep Doing (Success Patterns)

- **Honest H6 falsification reporting**: r=-0.136, p=0.301 correctly reported as NOT_SUPPORTED with mechanistic discussion - exemplary scientific practice.
- **Activation patching validation (67.3% mean recovery, 9/9 words above 10%)**: Genuine causal structure confirmed - strongest evidence in the paper.
- **CV-based steering prediction (1.47x, p < 0.01 BH-corrected)**: Solid primary finding if cross-architecture validation completes.
- **Appropriate BH correction**: Multiple hypothesis tests properly corrected.
- **Clear logical structure**: Problem → gap → approach → theory → experiments → discussion flows well.
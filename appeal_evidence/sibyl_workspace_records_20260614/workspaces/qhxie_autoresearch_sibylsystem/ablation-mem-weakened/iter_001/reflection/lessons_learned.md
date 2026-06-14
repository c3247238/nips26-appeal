# Lessons from Iteration 0

## Must Improve

1. **Pre-register power analysis before any experiments**: The n=26 feature count was determined by the alphabet size, not statistical power requirements. Before the next iteration, compute minimum detectable effect sizes at 80% power and design experiments to meet those requirements. Do not run underpowered studies.

2. **Add random feature baseline for steering**: This is the highest-impact, lowest-effort addition. Without it, steering success rates are uninterpretable. Run 26 random latent steering experiments and compare success rates against feature-specific steering.

3. **Weaken claims to match evidence strength**: The abstract and conclusion must frame null results as "no detectable relationship given power constraints" rather than "no relationship." Every claim must be scoped to "in GPT-2 Small with first-letter features at layers 4 and 8."

4. **Distinguish encoder absorption from decoder degradation**: Steering bypasses the encoder, making it inherently robust to the type of absorption measured by differential correlation. This confound must be explicitly discussed in the paper, not treated as surprising evidence.

5. **Create formal plan document before experiments**: Pre-register hypotheses, power analysis, sample size justification, parameter selection rationale, and falsification criteria. Do not let experimental parameters emerge ad hoc from ideation output.

6. **Fix all writing review flags before submission**: Banned transitions ("Moreover", "Furthermore", "It is worth noting that") must be removed. Missing data claims (F1_full) must be either supported with numbers or removed. The editor-writer feedback loop must be strengthened.

## Watch Out

1. **Synthesizer tendency toward feasibility over ambition**: The gap between ideation output (6 rich perspectives with ambitious proposals) and executed study (narrow correlation on 26 features) is substantial. In future iterations, the synthesizer should scale down the most promising idea rather than defaulting to the simplest feasible study.

2. **"Limitation" framing as substitute for action**: Random baseline, alternative metric, semantic hierarchies, and cross-model validation are all flagged as "limitations" but none are implemented. Do not use "limitation" as an excuse to avoid addressing fundamental controls.

3. **Steering robustness confound**: This is the most damaging conceptual flaw. A knowledgeable reviewer will immediately recognize that steering bypassing the encoder makes it inherently robust to the type of absorption measured. Address this explicitly or the central test collapses.

4. **Pilot-full pattern**: The pilot (layer 8, 50 samples: r=-0.153, p=0.456) to full (layer 8, 100 samples: r=-0.301, p=0.136) pattern shows the full experiment strengthened the negative trend but did not achieve significance. Be cautious about interpreting trends from underpowered pilots.

5. **Missing controls may be fatal**: The missing random feature baseline is flagged by both the critic and supervisor as potentially fatal. If random directions produce comparable steering success, the entire steering analysis becomes uninterpretable.

## Keep Doing (Success Patterns)

1. **Honest negative result reporting**: This is the paper's strongest aspect across ALL reviews. Continue reporting null results without spin, with specific expected vs. observed values and clear explanations.

2. **Training-free methodology**: The four-phase pipeline is accessible and replicable. This is a genuine methodological contribution that should be emphasized.

3. **Section 6.4 "What Would Change Our Conclusion?"**: This is excellent scientific writing that anticipates reviewer objections. Continue using this framing for null-result papers.

4. **Clear abstract with exact numbers**: The abstract states the problem, gap, methodology, key results with exact numbers, and implication---all in one paragraph. Every sentence earns its place. Maintain this standard.

5. **Consistent notation and well-defined symbols**: Key symbols are introduced before first use. Maintain this discipline.

6. **Fixed random seed for reproducibility**: This ensures that results are replicable. Continue using fixed seeds and documenting them.

7. **Comprehensive literature survey**: The survey identified 7 research gaps with clear relevance. Continue thorough literature review before ideation.

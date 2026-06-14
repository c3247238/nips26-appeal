# Ideation Critique

## Executive Summary

The proposal was ambitious, well-structured, and addressed genuine gaps in the absorption literature. The three-tension framing (measurement vs. phenomenon, mechanism vs. intervention, metric vs. impact) is compelling. However, the core theoretical bet -- that Lotka-Volterra competitive exclusion provides a quantitative framework for absorption -- was fundamentally flawed, and this was predictable from first principles.

---

## Strengths of the Ideation

1. **Clear gap identification.** The three tensions are real and unresolved. No prior paper provided (a) probe-free absorption detection, (b) corpus-level prediction, or (c) systematic downstream correlation analysis. The proposal correctly identified these as the critical open questions.

2. **Falsifiable hypotheses.** All four hypotheses have explicit success criteria (F1 > 0.65, partial R^2 > 0.10, |r| < 0.2, 80% monotonicity). This is exemplary research design.

3. **Robust pivot structure.** The risk assessment and decision tree (alternatives A-D, minimal fallback) demonstrate mature planning. The paper survived all three core hypotheses failing or partially failing because the pivot structure was pre-designed.

4. **Training-free design.** Operating on pre-trained Gemma Scope SAEs with no retraining is a practical constraint that enables broad coverage (30+ SAE configurations).

---

## Weaknesses of the Ideation

### 1. The LV analogy was predictably flawed

The Lotka-Volterra competitive exclusion model assumes:
- **Shared resource competition**: species compete for the same finite resource
- **Equilibrium dynamics**: the system reaches a stable equilibrium where one species is excluded
- **Symmetric interaction**: competition coefficients capture bidirectional effects

None of these hold for SAE features:
- SAE latents do not compete for a shared activation budget at inference time. They interact through the sparsity penalty during training, which is not observable from activation statistics alone.
- SAE activations are not an equilibrium system -- they are a single forward pass through a fixed network. There is no dynamic process for one latent to "exclude" another at inference time.
- The parent-child relationship is asymmetric: the child suppresses the parent during training, not at inference.

Computing alpha_ij from inference-time activation statistics attempts to capture a training-time phenomenon from its residue. This is like measuring the footprint of a building to determine whether an earthquake caused structural damage -- the measurement captures something correlated with the phenomenon but not the phenomenon itself.

**The proposal should have included a theoretical sanity check**: under what conditions does alpha_ij at inference reflect competitive dynamics during training? This would have revealed the analogy's limitations before committing GPU-hours.

### 2. H2 (PMI as absorption predictor) conflated two causal pathways

The hypothesis was: corpus co-occurrence statistics predict which feature pairs are absorbed. The reasoning was that masked regularization works by disrupting co-occurrence, so co-occurrence must cause absorption.

This conflates:
- **Masked regularization disrupts co-occurrence during training** (interventional effect on the training loss landscape)
- **Corpus PMI predicts which pairs are absorbed across different SAE configurations** (observational association)

Even if co-occurrence is part of the causal mechanism, corpus PMI is a population-level statistic that does not vary across SAE configurations. The regression model regresses absorption_rate on {L0, width, layer, PMI} -- but PMI is constant across SAE configurations for a given letter. It only varies across letters (26 levels). With 806 observations that are 31 repeats of the same 26 PMI values, the test has very limited effective sample size for the PMI variable.

### 3. H3 success criterion was backwards

The pre-registered prediction was |r| < 0.2 (absorption is disconnected from downstream quality). The data showed |r| > 0.4 (strong connection). The paper frames this as "H3 falsified, which is the best outcome for the field."

But the pre-registration specifically predicted disconnection. Falsification of your own prediction is not a contribution -- it means the prediction was wrong. The finding that absorption correlates with downstream quality is not surprising: SAEBench was designed to include diverse metrics, and absorption is one of them. Features that fail to activate (absorbed features) will naturally degrade probe-based tasks that depend on feature activation.

The real question -- does REDUCING absorption CAUSE improved downstream performance? -- remains unanswered. The correlational evidence is a starting point, not a conclusion.

### 4. Width paradox (H4) was underspecified

The prediction that DAS(k=3) increases monotonically with width assumes that wider SAEs distribute absorption across more children. But wider SAEs also have more features competing for each parent, more feature splitting, and different L0/D ratios. The DAS metric via logistic regression on 3 features with 40 samples per letter is statistically fragile (n_word_act_samples = 40, not the 10,000 claimed in methodology). The hypothesis was not specific enough to survive the noise.

---

## Assessment of the Pivot

The actual paper ended up being built on the "Minimal Fallback" plan from alternatives.md: (1) taxonomy showing comprehensive rate > 15-35%, (2) LV coefficient as descriptive tool, and (3) SAEBench correlation analysis. This is an honest outcome -- the front-runner hypotheses failed and the fallback was activated. The decision tree worked as designed.

However, the paper's framing does not acknowledge this pivot. It presents the LV framework as a "formal test" (Section 3) rather than a failed hypothesis that was rescued by the fallback plan. A more honest framing would structure the paper around the question "Does absorption matter?" (H3, positive answer) rather than the question "Can we detect and predict absorption?" (H1 and H2, negative answers).

---

## Verdict

The ideation was strong in gap identification, hypothesis formulation, and pivot planning. It was weak in theoretical grounding of the LV analogy and in the conflation of training-time mechanisms with inference-time observables. The resulting paper is a competent empirical study of a failed theoretical framework, rescued by a pre-planned fallback that uncovered a genuinely useful correlational finding.

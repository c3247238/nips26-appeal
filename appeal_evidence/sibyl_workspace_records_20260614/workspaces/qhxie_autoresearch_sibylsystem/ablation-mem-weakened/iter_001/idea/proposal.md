# Final Research Proposal: Feature Absorption in Sparse Autoencoders

## Title

**"The Interpretability Illusion: Quantifying How Feature Absorption Degrades Downstream SAE Reliability"**

## Abstract

Feature absorption — where general SAE features fail to fire and are instead "absorbed" into more specific child features — has been identified as a fundamental failure mode of sparse autoencoders (Chanin et al., 2024). However, the field currently lacks answers to a critical practical question: does absorption actually matter for real interpretability work? This study provides the first systematic, quantitative bridge between absorption detection and downstream task performance. Using pre-trained SAEs from Gemma Scope and SAEBench, we measure how feature absorption degrades feature steering effectiveness and sparse probing accuracy across multiple model layers and dictionary sizes. Our results establish concrete, actionable relationships between absorption rates and task degradation, transforming a qualitative concern into a measurable phenomenon and providing practical guidance for SAE practitioners.

## Motivation

Sparse autoencoders have become the dominant paradigm for mechanistic interpretability, enabling circuit analysis, feature steering, model editing, and bias detection. Yet the community faces an escalating credibility crisis: Korznikov et al. (2026) show that SAEs recover only 9% of true features despite 71% explained variance; random baseline SAEs match trained SAEs on standard metrics; and DeepMind's mechanistic interpretability team has deprioritized SAE research after finding negative results on downstream tasks.

At the center of this crisis is feature absorption. Chanin et al. (2024) proved that hierarchical features cause absorption, and the phenomenon persists across all tested LLM SAEs. SAEBench (Karvonen et al., 2025) includes absorption as a standardized metric. Matryoshka SAEs, OrtSAE, HSAE, and other architectural innovations all aim to reduce absorption.

But a fundamental gap remains: **nobody has shown that absorption actually degrades the interpretability tasks that motivate SAE research.** Researchers use SAEs for steering, circuit finding, and editing. If absorbed features are systematically unreliable, these applications may produce false negatives or misleading results. Yet no existing work quantifies this degradation. This study bridges that gap.

## Research Questions

1. **RQ1 (Primary):** Does feature absorption cause measurable degradation in downstream interpretability tasks? Specifically, do features with higher absorption rates exhibit lower steering effectiveness and reduced sparse probing accuracy?

2. **RQ2 (Secondary):** Is the relationship between absorption rate and task degradation consistent across model layers, dictionary sizes, and model families?

3. **RQ3 (Exploratory):** Can we derive actionable rules of thumb (e.g., "avoid features with >X% absorption rate for steering tasks") from the empirical relationship?

## Hypotheses

**H1 (Primary):** Features with higher absorption rates exhibit proportionally lower steering effectiveness. For a feature with absorption rate A, steering success rate S follows: S = S_0 * (1 - k * A) where S_0 is the baseline success rate for non-absorbed features and k is a task-specific degradation coefficient.

**H2 (Secondary):** Features with higher absorption rates exhibit lower sparse probing F1 scores. The relationship is monotonic: as absorption rate increases, probing accuracy decreases.

**H3 (Exploratory):** The absorption-degradation relationship is consistent across layers and dictionary sizes within a model family, but the absolute degradation coefficients vary across model families.

## Expected Contributions

1. **First quantitative bridge between absorption detection and task performance.** While absorption has been detected and architectural solutions proposed, no prior work measures whether absorption actually degrades the interpretability tasks that motivate SAE research.

2. **Empirical characterization of the "interpretability illusion."** We provide concrete numbers for how much absorption costs in steering effectiveness and probing accuracy, transforming a qualitative concern into a measurable phenomenon.

3. **Practical guidance for SAE users.** Our correlation analysis yields actionable rules of thumb (e.g., "avoid features with >30% absorption rate for steering tasks").

4. **Training-free methodology.** Our approach requires no SAE training, making it accessible to any researcher with a GPU and open-source tools.

5. **Reproducible evaluation framework.** We release our code and evaluation protocol, enabling the community to extend our findings to new models, tasks, and metrics.

## Method

### Overview

Our approach is entirely training-free. We analyze existing pre-trained SAEs using SAELens, SAEBench, and sae-spelling. The methodology consists of four phases: absorption detection, feature steering experiments, sparse probing experiments, and correlation analysis.

### Phase 1: Absorption Detection

For each SAE configuration, we:
1. Load pre-trained SAE via SAELens
2. Run the Chanin et al. absorption metric (via SAEBench or sae-spelling)
3. On the first-letter task: for each letter A-Z, detect if "starts with X" feature is absorbed
4. Record: absorption rate per feature, absorbing latent IDs, ablation effect magnitudes
5. Classify features into: HIGH absorption (>50%), MEDIUM (10-50%), LOW (<10%)

**SAE Configurations:**
- Primary: Gemma-2-2B, Gemma Scope SAEs, layers 8, 12, 16, dictionary sizes 16K and 65K
- Validation: Pythia-160M, SAEBench SAEs, layer 8, dictionary size 32K
- Optional: Llama-3.1-8B, Llama Scope SAEs, layer 12, dictionary size 32K

### Phase 2: Feature Steering Experiment

For each "starts with X" feature identified in Phase 1:
1. Extract feature direction from SAE decoder: `direction = sae.W_dec[feature_id]`
2. Generate test prompts containing words starting with target letter (100 prompts per letter)
3. Run steering at strengths: [1.0, 2.0, 5.0, 10.0]
4. Measure: does steering increase probability of target-letter words?
5. Success metric: % increase in target-letter token probability vs. unsteered baseline

**Controls:**
- Match HIGH and LOW absorption features for base activation strength
- Include random feature baseline (randomly selected SAE latents)
- Include null steering baseline (steering strength = 0)

### Phase 3: Sparse Probing Experiment

For each SAE:
1. Train k-sparse linear probe (k=1, 5, 10) on first-letter classification
2. Measure probe F1 score using: (a) only main feature latents, (b) all latents
3. Compare F1 degradation for absorbed vs. non-absorbed features

### Phase 4: Correlation Analysis

1. Compute Pearson/Spearman correlation between absorption rate and steering effectiveness
2. Compute correlation between absorption rate and probing F1 degradation
3. Fit linear model: `task_degradation = beta * absorption_rate + epsilon`
4. Report R^2, confidence intervals, and p-values
5. Test for consistency across layers, dictionary sizes, and model families

### Experimental Plan

| Phase | What | Datasets/Models | Metrics | Time |
|-------|------|-----------------|---------|------|
| Pilot | 5 features, 1 SAE, 2 tasks | Gemma-2-2B layer 8, 16K | Steering success, probing F1 | ~3 hrs |
| Scale-up | 26 features (A-Z), 3 SAEs (layers 8, 12, 16) | Gemma-2-2B | Same + correlation | ~15 hrs |
| Cross-model | 26 features, 1 SAE | Pythia-160M layer 8 | Same | ~5 hrs |
| Ablation | Vary steering strength [1, 2, 5, 10, 20] | Gemma-2-2B layer 8 | Dose-response curve | ~3 hrs |

### Baselines

1. **No-absorption baseline:** Features with <5% absorption rate
   - Expected: High steering effectiveness (>80% success), high probing F1 (>0.9)

2. **Random feature baseline:** Randomly selected SAE latents
   - Expected: Near-zero steering effectiveness, random probing F1

3. **Full-activation baseline:** Using all SAE latents for probing
   - Expected: Higher F1 than k-sparse probing; measures recoverability

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| No absorbed features found in selected set | Low | High | Use Chanin et al. validated first-letter task; most SAEs show absorption |
| Steering effectiveness is noisy | Medium | Medium | Run multiple prompts per feature; use statistical testing |
| Absorption doesn't correlate with degradation | Medium | High | Core hypothesis; if falsified, paper becomes "absorption doesn't matter" — still publishable |
| SAEBench absorption metric too slow | Low | Low | Use sae-spelling directly; metric is the same |
| Gemma Scope SAEs unavailable | Low | High | Fallback to SAEBench collection or Pythia SAEs |
| First-letter task too narrow | Medium | Medium | Supplement with semantic hierarchy features (WordNet) |

## Resource Estimate

| Item | Estimate |
|------|----------|
| GPU | Single 24GB GPU (RTX 3090/4090 or A10) |
| Pilot | ~3 GPU-hours |
| Full study | ~25 GPU-hours |
| Wall-clock (with parallelization) | ~2-3 days |
| Model sizes | Gemma-2-2B (primary), Pythia-160M (validation) |
| Storage | <10GB for models + SAEs |

All experiments are training-free, using only pre-trained models and SAEs.

## Novelty Assessment

### What is New

1. **First quantitative bridge** between absorption detection and task performance
2. **Empirical characterization** of the "interpretability illusion" with concrete numbers
3. **Actionable rules of thumb** for SAE practitioners
4. **Training-free methodology** accessible to the entire community

### Prior Art Check

- Chanin et al. (2024): Identified and measured absorption, but did not connect to downstream tasks
- SAEBench (Karvonen et al., 2025): Includes absorption metric but does not study task degradation
- Li et al. (2025) "Interpretability Illusions": Discusses qualitatively but without quantitative task metrics
- Marks et al. (2024) "Sparse Feature Circuits": Does circuit finding but doesn't measure absorption impact
- Matryoshka SAEs, OrtSAE, HSAE: Propose solutions but don't quantify the problem they solve in task terms

**No existing paper systematically correlates absorption rates with steering effectiveness or probing accuracy.**

### Differentiation from Competing Directions

- **vs. Scaling law for absorption (Theoretical's Candidate C):** We focus on task impact rather than predictive scaling laws. Scaling laws are valuable but don't answer "so what?"
- **vs. Homeostatic repair (Innovator's Candidate B):** We diagnose rather than repair. Repair is valuable follow-up work but requires validating that the problem exists first
- **vs. Absorption as optimal compression (Contrarian's Candidate C):** We take the mainstream framing (absorption is a problem) and quantify its cost, which is the prerequisite for any reframing
- **vs. Metric validation (Empiricist's Candidate C):** We accept the Chanin metric as a working tool and study its consequences, rather than validating the metric itself

## Iteration 1 Improvements (Based on Reflection)

Based on the review and reflection from iteration 0, the following improvements are prioritized for iteration 1:

### Critical Fixes

1. **Add random feature steering baseline.** The iteration 0 paper omitted this control despite promising it in the proposal. Without it, steering success rates are uninterpretable. We will steer 26 randomly selected SAE latents and compare their success rates to the first-letter features.

2. **Fix the steering robustness confound discussion.** The iteration 0 paper presented Feature U (A=0.242, 100% steering success) as a surprising counterexample. Reviewers correctly noted that steering bypasses the encoder entirely, making it inherently robust to encoder-side absorption. We must explicitly discuss this encoder-vs-decoder distinction and reframe the finding accordingly.

3. **Add formal power analysis table.** The iteration 0 paper incorrectly claimed "80% power" for n=26, |r|>=0.50. The correct power is ~65%. We will include a power analysis table showing: (a) actual power for our sample size, (b) sample size needed for 80% power at various effect sizes, (c) interpretation of our observed correlations in this context.

### Major Improvements

4. **Weaken claims to match evidence.** The abstract and conclusion in iteration 0 made stronger claims than the evidence supported (e.g., "challenge the assumption"). We will qualify all claims with scope (GPT-2 Small, first-letter features, Chanin metric) and use "may not be" rather than "is not" language.

5. **Add alternative absorption metric.** The iteration 0 paper used only the Chanin differential correlation metric. We will add a second metric (SAEBench ablation-based) to test whether the null result is metric-specific.

6. **Expand to semantic hierarchy features.** First-letter features have shallow, uniform hierarchies. We will add a small set of semantic features (e.g., animal, mammal, dog) to test whether deeper hierarchies produce stronger absorption-degradation relationships.

7. **Create formal plan document with pre-registered hypotheses.** The iteration 0 pipeline lacked a formal plan. We will pre-register hypotheses, falsification criteria, and power analysis before running experiments.

### Success Criteria for Iteration 1

- Score >= 7.5 on supervisor review (up from 5.5)
- All critical issues from iteration 0 resolved
- At least 2 major improvements implemented
- Paper ready for external review

## Synthesis Reasoning

### How the Perspectives Were Weighted

**Highest weight: Pragmatist + Empiricist.** These perspectives converged on the same front-runner (Candidate B: task-oriented absorption impact) for the same reasons: it answers the most glaring "so what?" question, it's training-free, it's concrete and measurable, and it's reproducible. The pragmatist's engineering reality check and the empiricist's methodological rigor together provide the strongest foundation.

**Strong influence: Contrarian.** The contrarian's challenge — that absorption may not actually matter for real tasks — is precisely what makes this study necessary. If the contrarian is right and absorption has no task impact, that itself is a major finding that would reshape the field's priorities. The contrarian's steelman of the mainstream view also strengthened our experimental design (need for careful controls, frequency matching, etc.).

**Moderate influence: Theoretical.** The theoretical perspective's scaling law proposal (Candidate C) is elegant and would fill a real gap, but the pragmatist correctly identified that scaling laws alone don't answer the "so what?" question. We incorporate the theoretical insight by testing whether the absorption-degradation relationship is consistent across scales (H3), but we do not make scaling law prediction the central claim.

**Selective incorporation: Innovator + Interdisciplinary.** The innovator's "lateral inhibition lens" and the interdisciplinary's "explaining away" framework are intellectually rich but were judged as higher-risk for a first study. The neuroscience/Bayesian framings are valuable for follow-up work (especially if we find that absorption does matter and want to develop mitigation strategies), but they are not necessary for establishing the basic empirical fact. We acknowledge these framings in the Related Work and Future Directions sections.

### What Was Dropped

- **Innovator's homeostatic repair:** Too high-risk for a first study; requires proving the problem exists first
- **Theoretical's scaling law:** Valuable but secondary to the task-impact question
- **Theoretical's absorption-composition duality:** Relies on tied-weight assumption that doesn't hold in practice
- **Contrarian's optimal compression reframing:** Too provocative for a first study; better as a follow-up if we find weak or null effects
- **Empiricist's metric validation:** Important but orthogonal to our main question; can be pursued in parallel
- **Interdisciplinary's spin glass theory:** Fascinating but requires sophisticated mathematical tools; better as theoretical follow-up

### Why This Synthesis is Not a Compromise

The best proposal is not a compromise but a synthesis that takes the strongest elements from each perspective:
- From the **pragmatist**: engineering feasibility, clear metrics, reproducible design
- From the **empiricist**: rigorous controls, statistical testing, falsification criteria
- From the **contrarian**: the courage to question whether absorption matters, and the experimental design to answer that question definitively
- From the **theoretical**: the layer-depth and dictionary-size hypotheses (H3)
- From the **innovator**: the recognition that this is a foundational study that enables future repair/mitigation work
- From the **interdisciplinary**: the Bayesian/explaining-away insight that absorption is a structural property of sparse coding (acknowledged in Related Work)

The result is a focused, feasible, high-impact study that answers a critical unanswered question and lays the groundwork for the more ambitious directions proposed by the innovator, theoretical, and interdisciplinary perspectives.

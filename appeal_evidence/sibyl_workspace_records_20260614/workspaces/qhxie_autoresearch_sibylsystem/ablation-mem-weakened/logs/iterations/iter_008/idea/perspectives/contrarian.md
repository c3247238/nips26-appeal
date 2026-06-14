# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption**: Feature absorption is a fixable training artifact
   - Evidence challenging it: Cui et al. (ICLR 2026) prove standard SAEs generally cannot fully recover ground-truth features due to intrinsic representational interference. The theoretical limit suggests absorption may be mathematically inevitable, not merely a training problem.
   - Source: arXiv:2506.15963 (On the Limits of Sparse Autoencoders)

2. **Assumption**: Quantifying absorption leads to actionable improvements
   - Evidence challenging it: Basu et al. (2026) demonstrate 98.2% probe AUROC but only 45.1% output sensitivity; SAE steering produces zero effect due to residual stream compensation. This is a direct attack on the practical utility of absorption research.
   - Source: arXiv:2603.18353 (Interpretability without Actionability)

3. **Assumption**: SAEs reliably beat random baselines on interpretability tasks
   - Evidence challenging it: Adam et al. (2026) systematically show SAEs often fail to beat random baselines on standard metrics, raising concerns about current evaluation practices.
   - Source: arXiv:2602.14111 (Sanity Checks for Sparse Autoencoders)

4. **Assumption**: Absorption is the primary failure mode to address
   - Evidence challenging it: Chanin et al. (2025a) show absorption and hedging are two sides of the same coin — wide SAEs suffer absorption, narrow SAEs suffer hedging. Fixing one often exacerbates the other.
   - Source: arXiv:2505.11756 (Feature Hedging)

5. **Assumption**: Wider SAEs are better because they reduce absorption
   - Evidence challenging it: Chanin et al. reveal wide SAEs simply trade absorption for hedging at inner levels. The "optimal" configuration is architecture-dependent and context-dependent.
   - Source: arXiv:2505.11756

6. **Assumption**: Feature absorption is a distinct, quantifiable phenomenon
   - Evidence challenging it: The absorption metric (Chanin et al.) is computationally expensive (~26 min per SAE) and limited to early layers. The projection-based alternative may have its own limitations (probe quality dependence). There is no gold-standard measurement.
   - Source: arXiv:2409.14507, SAEBench (arXiv:2503.09532)

7. **Assumption**: Hierarchical features are the primary driver of absorption
   - Evidence challenging it: The phenomenon is characterized using first-letter spelling tasks, which may not generalize to semantic hierarchies, syntactic features, or factual knowledge hierarchies.
   - Source: Literature Gap #5 (generalization beyond spelling tasks)

8. **Assumption**: Pretrained SAEs are reliable for interpretability research
   - Evidence challenging it: The entire pretrained SAE ecosystem (GemmaScope, LlamaScope) exhibits absorption. Using these SAEs for circuit discovery or steering may produce misleading results.

### Landscape of Doubt

The feature absorption research program rests on several unexamined assumptions:

1. **That absorption is quantifiable with current metrics**: The ablation-based metric is limited to early layers; the projection alternative is probe-dependent. We lack a ground-truth validation approach (SynthSAEBench exists but uses synthetic data).

2. **That fixing absorption improves downstream utility**: No study has demonstrated that reducing absorption translates to better steering, circuit discovery, or concept erasure outcomes. Basu et al. suggest even perfect absorption detection may not yield actionable interventions.

3. **That the SAE representation of features is what we think it is**: If SAEs often fail to beat random baselines, perhaps the interpretability "features" we attribute to them are post-hoc rationalizations rather than causal mechanisms.

4. **That hierarchical structure in features is the problem**: Perhaps superposition itself (not hierarchy) is the fundamental issue, and absorption is just one manifestation of the impossibility of full disentanglement (Cui et al., ICLR 2026).

---

## Phase 2: Initial Candidates

### Candidate A: The "Absorption is Information-Theoretically Inevitable" Challenge

- **Challenged assumption**: Feature absorption is a fixable training artifact that we should invest in mitigating.
- **Evidence against**: Cui et al. (ICLR 2026) prove mathematically that standard SAEs cannot fully recover ground-truth features under realistic sparsity constraints. This is not a bug — it's a fundamental limit.
- **Contrarian hypothesis**: Absorption may be the SAE learning a lossy but functionally optimal representation. The "absorbed" features may not be missing — they may be integrated into the residual stream in ways that are invisible to ablation-based metrics but compensated for downstream.
- **Exploitation plan**: Design experiments that test whether absorbed features still influence model behavior indirectly through residual stream paths. If absorbed features leave detectable behavioral traces (via output changes) even when ablation shows no direct SAE contribution, this would challenge the necessity of "fixing" absorption.
- **Novelty estimate**: 7/10

### Candidate B: The "Zero Practical Impact" Challenge

- **Challenged assumption**: Quantifying and mitigating absorption will improve practical applications like model steering, circuit discovery, and concept erasure.
- **Evidence against**: Basu et al. (2026) show 98.2% AUROC for feature detection but only 45.1% output sensitivity. SAE steering produces zero effect. The residual stream compensates.
- **Contrarian hypothesis**: The entire absorption research program may be solving a problem that doesn't matter for downstream applications. The relevant question is not "can we detect absorption?" but "does reducing absorption change what we can do with SAEs?"
- **Exploitation plan**: Benchmark absorption quantification metrics (ablation rate, c_dec) against downstream task improvement (steering success rate, circuit completeness). If there's no correlation, the research program is misdirected.
- **Novelty estimate**: 8/10

### Candidate C: The "SAE Interpretability is an Illusion" Challenge

- **Challenged assumption**: SAEs extract human-interpretable features that correspond to real model computations.
- **Evidence against**: Adam et al. (2026) show SAEs often fail to beat random baselines. Feature interpretability scores (via LLM-as-judge) have no ground-truth validation. The entire Neuronpedia ecosystem of "interpretable features" may be confabulation.
- **Contrarian hypothesis**: The "interpretability" of SAE features is a post-hoc rationalization. Features that look interpretable (e.g., "the concept of dogs") may not have causal correspondence to model behavior, while truly important features may look uninterpretable.
- **Exploitation plan**: Use counterfactual tests: for features claimed to be interpretable, test whether directly manipulating them produces expected behavioral changes. Compare to matched random directions. This is a sanity check that the field largely hasn't done systematically.
- **Novelty estimate**: 6/10

---

## Phase 3: Self-Critique

### Against Candidate A: The "Absorption is Inevitable" Challenge

- **Steelman**: The fact that some architectures (OrtSAE, MP-SAE) reduce absorption suggests it's partially solvable. The theoretical limit from Cui et al. applies to "standard SAEs" under specific assumptions — modified architectures may push toward the limit. Moreover, even if absorption is inevitable, quantifying it still matters for knowing the reliability of interpretations.

- **Cherry-picking check**: I'm citing Cui et al. as evidence that absorption is inevitable, but they only prove this for standard SAEs. I'm ignoring the substantial body of work showing partial solutions (OrtSAE -65%, MP-SAE reduced absorption). This is selective citation.

- **Confounding check**: Even if absorption is partially inevitable, the question is whether the remaining reducible portion matters for downstream applications. The confounder is: does fixing the reducible absorption change anything practically?

- **Actionability check**: **This is the killer objection.** Even if I'm right that absorption is largely inevitable, this doesn't lead to a better method. It leads to "SAEs are flawed, don't use them" — which doesn't help the field advance.

- **Verdict**: WEAK as a standalone proposal. The steelman reveals the actionability problem is fatal. However, Candidate A may be valuable as a framing for interpreting results (manage expectations about absorption rates), not as a standalone method proposal.

### Against Candidate B: The "Zero Practical Impact" Challenge

- **Steelman**: Basu et al. (2026) only test on a clinical domain. Maybe clinical text has different structure than the tasks where SAEs have been most successful (e.g., spelling tasks, chess). Generalizing from one domain to all SAE applications is unwarranted.

- **Cherry-picking check**: I'm relying heavily on one paper (Basu et al.) that reports zero effect. There are many papers showing positive SAE steering results (e.g., Templeton et al., 2024 on Claude feature steering). I'm selectively citing negative results.

- **Confounding check**: The zero effect in Basu et al. may be due to specifics of the clinical domain and the SAE training data, not a general property of SAE steering. The confound is domain-specificity.

- **Actionability check**: **This is also fatal.** If this critique is correct, it means the entire research program is misdirected. But "don't work on this" is not a publishable positive result — it's a career-ender for everyone in the field.

- **Verdict**: STRONG as a critical sanity check, WEAK as a standalone proposal. The actionability problem is real, but the cherry-picking concern is also real. The value of this candidate is to force the field to justify why absorption quantification matters for downstream tasks.

### Against Candidate C: The "SAE Interpretability is an Illusion" Challenge

- **Steelman**: The sanity check paper (Adam et al.) is very recent (2026) and hasn't been widely adopted as a standard evaluation. The LLM-as-judge approach for interpretability is known to be imperfect but that's different from saying the entire enterprise is invalid. Also, many steering results (Templeton et al.) show behavioral changes from feature manipulation, which suggests some level of causal correspondence.

- **Cherry-picking check**: I'm citing "SAEs fail to beat random baselines" broadly, but the paper shows this for some metrics and some configurations, not universally. I'm overgeneralizing from specific findings.

- **Confounding check**: "Failing to beat random baselines" on standard metrics doesn't mean features are random. It means the metrics may not capture what makes SAE features useful. The confound is metric validity.

- **Actionability check**: If SAEs are fundamentally uninterpretable, what do we use instead? Transcoders? Mechanistic interpretability via circuit analysis? There's no ready alternative with the same coverage.

- **Verdict**: MODERATE. This is the most dangerous idea because it questions the fundamental premise. The steelman is strong (Templeton et al. steering results), but the sanity check critique is legitimate. The actionability problem is real but less fatal than for Candidate B — we can at least propose better sanity checks as a deliverable.

---

## Phase 4: Refinement

### Dropped Candidates

- **Candidate A (Absorption is inevitable)**: Survived steelman but failed actionability test. Not publishable as "don't work on this." May be valuable as a framing/context within another proposal.

- **Candidate B (Zero practical impact)**: Strong critique but too cherry-picked. Generalizing from one clinical domain study to all SAE applications is unwarranted. However, the underlying question (does absorption mitigation improve downstream utility?) is legitimate and worth investigating.

### Strengthened Candidates

- **Candidate C (Interpretability illusion)**: Refined to focus on **validation methodology** rather than wholesale dismissal. The contrarian insight is: the field lacks systematic sanity checks that connect interpretability claims to behavioral predictions. We don't know whether "interpretable" features are actually causal.

- **New front-runner from Gap #3**: Impact of absorption on downstream tasks. This is the most actionable contrarian angle — it doesn't challenge the field's premise but forces empirical validation of the assumed chain: absorption quantification → mitigation → improved downstream utility.

### Selected Front-Runner

**Focus: The Missing Validation Chain in Absorption Research**

The contrarian hypothesis: Research on feature absorption lacks systematic validation that mitigating absorption actually improves downstream applications (steering, circuit discovery, concept erasure). The field assumes this chain is true without testing it.

This is worth investigating because:
1. Basu et al. provide preliminary evidence the chain may break at the "downstream utility" step.
2. It leads to actionable research: design and execute validation experiments for the absorption → utility chain.
3. It doesn't require dismissing the entire research program — just adding the missing validation step.

---

## Phase 5: Final Proposal

### Title

**Rethinking Feature Absorption as an Intervention Target: Does Quantifying Absorption Help downstream Interpretability Tasks?**

Framed constructively as a validation study, not a dismissal.

### Challenged Assumption

The research community assumes that quantifying and mitigating feature absorption in SAEs will translate to improved downstream interpretability applications (model steering, circuit discovery, concept erasure). This assumption is untested.

### Evidence

**Supporting the assumption**:
- Chanin et al. (2024) demonstrate that absorption is real and quantifiable
- Templeton et al. (2024) show SAE steering works on Claude features
- SAEBench includes absorption as a standard metric, implicitly treating it as important

**Challenging the assumption**:
- Basu et al. (2026): 98.2% AUROC but 45.1% output sensitivity; zero steering effect in clinical domain
- Adam et al. (2026): SAEs often fail to beat random baselines on standard metrics
- Gap #3 in the literature: "impact of absorption on downstream interpretability tasks is largely unquantified"

### Hypothesis

The absorption metric (and related metrics like c_dec, decoder cosine similarity) may not correlate with downstream task improvement. The chain "absorption quantification → mitigation → better utility" may be broken at the last link.

### Method

Design a validation study that tests the entire chain:

1. **Baseline**: Measure absorption rates for pretrained SAEs (GemmaScope, GPT-2 SAEs) using standard metrics
2. **Correlate**: Compare absorption rates to downstream task performance (steering success rate, circuit completeness, concept erasure effectiveness)
3. **Test**: Use SAEs with varying absorption rates (from different architectures: TopK, JumpReLU, Matryoshka, OrtSAE) on identical downstream tasks
4. **Validate**: Compare against sanity-check baselines (random directions, scrambled features)

### Experimental Plan

**Task 1: Steering Success Rate Correlation** (30-45 min)
- Use GPT-2-small SAEs from SAELens (varying widths: 16k, 65k)
- Select 20 features with high interpretability scores (from Neuronpedia)
- For each feature, measure steering effect on text completion
- Compare effect size to absorption rate of the feature's latent
- Baseline: same steering using random directions

**Task 2: Circuit Discovery Completeness** (45-60 min)
- Use Gemma-2-2B + GemmaScope SAEs (layers 5, 10, 15)
- Run circuit discovery for a simple task (first-letter spelling)
- Compare circuit completeness (number of components found) to absorption rate of SAEs used
- Baseline: circuit discovery with scrambled SAE latents

**Task 3: Concept Erasure Effectiveness** (30-45 min)
- Use SAEBench concept erasure benchmark
- Compare erasure effectiveness across SAEs with varying absorption rates
- Test whether SAE width (which affects absorption) predicts erasure success

### Baselines

- Random direction steering (Sanity check)
- Scrambled feature names (Sanity check)
- High-absorption SAE vs low-absorption SAE on identical tasks
- Published steering results (Templeton et al.) as reference

### Risk Assessment

**Risk**: The hypothesis is wrong and absorption rate does predict downstream utility.
- **Mitigation**: Publish negative result (no correlation found) with careful analysis of why the field expected correlation. This is still a valuable contribution (it falsifies a common assumption).
- **Fallback**: If correlation exists in some domains but not others, document the boundary conditions.

**Risk**: Steering effects are too noisy to measure reliably.
- **Mitigation**: Use automated evaluation (loss change, log probability difference) rather than human judgment. Run multiple seeds.

**Risk**: Pretrained SAEs have confounding factors (L0 not optimal, different training data).
- **Mitigation**: Use c_dec metric to select SAEs with near-optimal L0 before comparing.

### Novelty Claim

This is the first systematic study to directly test whether absorption metrics predict downstream task improvement. All prior work assumes the chain is true; this study validates or falsifies it. This addresses **Gap #3** in the literature survey directly.

The contrarian insight: The field has been measuring absorption without asking whether it matters. If it doesn't, the entire research program needs recalibration.
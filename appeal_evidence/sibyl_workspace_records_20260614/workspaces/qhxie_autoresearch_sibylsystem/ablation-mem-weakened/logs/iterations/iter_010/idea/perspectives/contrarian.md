# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

**Assumption 1: Feature absorption is a failure mode that degrades SAE-based interpretability**
- Evidence challenging it: The project found zero significant correlations between absorption rates and downstream task performance (steering effectiveness, sparse probing accuracy, EC50) across 12 statistical tests after rigorous multiple comparison correction. Feature U (24.2% absorption) achieves 100% steering success, demonstrating that high absorption does not preclude functional utility.
- Source: idea_context.md (H1-H4 null results)

**Assumption 2: The Chanin absorption metric measures learned pathology rather than structural artifacts**
- Evidence challenging it: Random SAE baselines show 8x HIGHER absorption (mean=0.278) than trained SAEs (mean=0.034). If absorption were primarily a learned failure mode, random initialization should produce less organized structures, not more absorption. The metric appears sensitive to dictionary structure itself, not specifically to learned failure.
- Source: idea_context.md (H7: trained vs random comparison)

**Assumption 3: SAEs learn meaningful features that outperform random baselines**
- Evidence challenging it: "Sanity Checks for Sparse Autoencoders" (arXiv:2602.14111, 2026) shows frozen/random baselines match trained SAEs on standard metrics. The field must confront whether dictionary learning produces meaningful decomposition or merely provides a convenient basis.
- Source: literature.md (ref #16)

**Assumption 4: Higher interpretability implies better utility**
- Evidence challenging it: Wang et al. (ICLR 2026, arXiv:2510.03659) found weak correlation (tau_b ~ 0.3) between interpretability and steering utility. If absorption reduction improves interpretability scores but not utility, the entire mitigation agenda may be misdirected.
- Source: literature.md (ref #23)

**Assumption 5: Architectural solutions to absorption (Matryoshka, OrtSAE) meaningfully improve interpretability**
- Evidence challenging it: Chanin et al.'s Feature Hedging paper shows Matryoshka exacerbates hedging as a trade-off. Bussmann et al. reduce absorption from 0.49 to 0.05 but introduce reconstruction fidelity costs. Neither addresses whether absorption reduction actually improves downstream tasks.
- Source: literature.md (refs #5, #6, #7)

**Assumption 6: Absorption is specific to SAEs and represents a SAE-specific failure**
- Evidence challenging it: Transcoders (Paulo et al., arXiv:2501.18823) achieve Pareto dominance over SAEs on interpretability metrics with a different objective (cross-layer vs. self-reconstruction). This suggests the problem may be general to sparse dictionary learning, not SAE-specific.
- Source: literature.md (ref #10)

**Assumption 7: The field's focus on absorption mitigation is scientifically grounded**
- Evidence challenging it: No systematic study has demonstrated that reducing absorption improves any downstream interpretability task. The entire research direction rests on the assumption that absorption is harmful without empirical validation of that assumption.
- Source: Gap 2 in literature.md

### Landscape of Doubt

The literature reveals a field racing to address a problem (absorption) without first establishing whether the problem matters for downstream tasks. Key cracks:

1. **Metric validity crisis**: The Chanin absorption metric may measure dictionary structure artifacts rather than learned pathology. Trained SAEs show LESS absorption than random baselines, which inverts the expected pattern if absorption were a training artifact.

2. **Interpretability-utility disconnect**: Wang et al. show weak correlation between interpretability scores and steering utility. If absorption affects interpretability scores but not utility, the entire mitigation research program may be irrelevant.

3. **Sanity check challenge unanswered**: The Sanity Checks paper's observation that random baselines match trained SAEs on standard metrics remains inadequately addressed for absorption metrics specifically.

4. **Negative results not published**: The project found zero significant effects across 12 tests, but no such null results appear in the published literature. Publication bias may distort the field's perception of how problematic absorption actually is.

5. **Architectural solutions may create new problems**: Matryoshka introduces hedging; OrtSAE has lower explained variance. Solutions trade one problem for another without demonstrating utility improvement.

---

## Phase 2: Initial Candidates

### Candidate A: The Metric Artifact Hypothesis
- **Challenged assumption**: The Chanin absorption metric measures learned failure modes in trained SAEs
- **Evidence against**: Random SAEs show 8x higher absorption than trained SAEs (0.278 vs 0.034, p < 0.001). If absorption were a training artifact, random initialization should produce less organized structures with more "absorption-like" patterns, not less.
- **Contrarian hypothesis**: The Chanin absorption metric is not well-calibrated for distinguishing learned structure from random dictionary artifacts. Absorption in trained SAEs may be "residual structural artifact" that training partially suppresses, not a failure mode that training creates.
- **Exploitation plan**: Propose metric reform: absorption should be measured relative to random baseline controls, not as an absolute quantity. Publish the trained < random finding as evidence that training reduces structural artifacts.
- **Novelty estimate**: 8/10 (first systematic demonstration of random vs trained on absorption metrics specifically)

### Candidate B: The Benign Absorption Hypothesis
- **Challenged assumption**: Feature absorption degrades downstream interpretability tasks
- **Evidence against**: Zero significant correlations across 12 statistical tests (steering, probing, EC50). Feature U (24.2% absorption) achieves 100% steering success. The one robust finding (H5) shows precision = 1.0 universally—decoder directions preserve semantic content.
- **Contrarian hypothesis**: Absorption is benign for downstream tasks because it reflects encoder activation suppression while preserving decoder alignment. The SAE correctly routes information through child latents; the parent feature's semantic content is preserved, just redistributed.
- **Exploitation plan**: Frame absorption as "optimal information routing" rather than "failure". Demonstrate that high-absorption features remain functionally steerable. Propose that absorption mitigation research may be addressing a non-problem.
- **Novelty estimate**: 6/10 (builds on Chanin et al.'s Proposition 2 but extends with downstream task evidence)

### Candidate C: The Wrong Problem Hypothesis
- **Challenged assumption**: Absorption is the right target for SAE improvement research
- **Evidence against**: No published study demonstrates that reducing absorption improves downstream task performance. The field's focus on absorption may be misdirected attention from a metric that doesn't correlate with utility.
- **Contrarian hypothesis**: The field should stop optimizing absorption metrics until someone demonstrates absorption reduction actually improves any downstream task. Focus should shift to utility validation studies.
- **Exploitation plan**: Conduct a controlled experiment: compare high-absorption vs low-absorption SAE features on steering, circuit discovery, and concept erasure. If no difference, the entire absorption mitigation research program is invalidated.
- **Novelty estimate**: 7/10 (systematic utility validation of absorption reduction has not been done)

---

## Phase 3: Self-Critique

### Against Candidate A (Metric Artifact)

**Steelman**: Chanin et al. defined absorption as a phenomenon in trained SAEs and proved it is a logical consequence of sparsity loss (Proposition 2). Random SAEs are never evaluated in practice, so the metric's behavior on random dictionaries is irrelevant to its utility for measuring trained SAE quality. The metric correctly identifies a real structural property.

**Cherry-picking check**: We are selectively highlighting the random > trained result while ignoring that absorption exists in both. The absolute absorption level in trained SAEs (0.034 mean) is still non-zero and represents real structural mixing.

**Confounding check**: The random SAE has an orthonormal decoder, which is structurally different from trained decoders. Comparing absorption metrics across different decoder structures may not be meaningful.

**Actionability check**: If the metric is flawed, what replaces it? Without a validated alternative, dismissing absorption measurement leaves the field without tools.

**Verdict**: MODERATE. The metric artifact argument is valid but doesn't eliminate the need for absorption measurement—it suggests calibration is needed.

### Against Candidate B (Benign Absorption)

**Steelman**: Wang et al. show weak interpretability-utility correlation. If absorption affects interpretability but not utility, and our results show absorption doesn't affect utility, absorption may indeed be benign. Chanin et al.'s Proposition 2 proves absorption minimizes sparsity loss, which is exactly what an optimal encoder should do.

**Cherry-picking check**: We found zero significant effects, but the field has published many papers showing absorption correlates with interpretation quality (Neuronpedia annotations, AutoInterp scores). Are we cherry-picking our own negative results?

**Confounding check**: Our steering experiments used a limited set of first-letter features. The null result may not generalize to all feature types, layers, or models.

**Actionability check**: If absorption is benign, this is valuable information—researchers should stop spending resources on mitigation. But the claim needs stronger validation across more features and tasks.

**Verdict**: STRONG. The combination of zero downstream effects and training-reduces-absorption suggests absorption may be benign and optimal. However, limited to one model (GPT-2 Small).

### Against Candidate C (Wrong Problem)

**Steelman**: No published study demonstrates absorption reduction improves downstream tasks. The field may be solving an imaginary problem. Kantamneni et al. show SAEs don't consistently outperform baselines on probing; this may extend to absorption studies.

**Cherry-picking check**: We assume no study exists, but perhaps the architectural solutions (Matryoshka, OrtSAE) have been validated on downstream tasks we haven't reviewed thoroughly.

**Confounding check**: If absorption doesn't affect utility, but Matryoshka/OrtSAE improve absorption metrics, perhaps they improve other aspects (reconstruction, feature diversity) without addressing the original motivation.

**Actionability check**: This is highly actionable—design the controlled experiment comparing high vs low absorption features on utility tasks. This would be the first systematic utility validation study.

**Verdict**: STRONG. The absence of utility validation is a genuine gap. The experiment is straightforward and would provide definitive evidence for or against absorption mitigation research.

---

## Phase 4: Refinement

**Dropped**: Candidate A (metric artifact) survives the steelman but doesn't lead to a publishable result—metric reform without replacement is not actionable.

**Strengthened**: Candidate B (benign absorption) is strengthened by the precision=1.0 finding and the training-reduces-absorption result. The intellectual framing (absorption as optimal compression) is grounded in Chanin et al.'s Proposition 2 and our empirical findings.

**Strengthened**: Candidate C (wrong problem) is the most provocative and actionable. It directly challenges the research agenda of multiple active papers (Matryoshka, OrtSAE, ATM, H-SAE). The experiment is simple and would either invalidate or justify the entire absorption mitigation research direction.

**Additional search needed**: Search for "absorption mitigation downstream task improvement" to confirm no prior utility validation study exists.

**Selected front-runner**: Candidate C (The Wrong Problem Hypothesis) — it directly challenges the research agenda, is testable with existing resources, and would either justify or invalidate a significant portion of current SAE research.

---

## Phase 5: Final Proposal

### Title

**"Rethinking Feature Absorption: A Controlled Study on Whether Absorption Mitigation Improves Downstream Utility"**

Alternative: **"Is Feature Absorption the Right Target? A Utility-Centric Evaluation of SAE Absorption Mitigation"**

### Challenged Assumption

The field assumes that feature absorption in SAEs is a failure mode that degrades interpretability-based downstream tasks, justifying architectural modifications (Matryoshka, OrtSAE) and training objectives (ATM, H-SAE) aimed at reducing absorption. This assumption has never been directly tested.

### Evidence

**For the assumption**:
- Chanin et al. (2024) defined absorption as a logical consequence of sparsity loss, establishing it as a real phenomenon
- Neuronpedia community annotations suggest absorbed features are harder to interpret
- Architectural solutions consistently report absorption reduction as a positive outcome

**Against the assumption**:
- Zero significant correlation between absorption and steering/probing/EC50 across 12 tests (idea_context.md, H1-H4)
- Feature U achieves 100% steering success at 24.2% absorption
- Trained SAEs show LESS absorption than random SAEs (0.034 vs 0.278), suggesting absorption may be optimal compression behavior
- No published study demonstrates that reducing absorption improves any downstream task
- Wang et al. (ICLR 2026) show weak correlation between interpretability and utility generally

### Hypothesis

**Hypothesis**: High-absorption SAE features and low-absorption SAE features perform equivalently on downstream interpretability tasks (steering, circuit discovery, concept erasure).

**Prediction if correct**: Absorption reduction via Matryoshka, OrtSAE, or ATM provides no downstream utility improvement beyond non-absorption-reducing baselines.

**Prediction if incorrect**: Features with actively reduced absorption will show measurable improvement in steering fidelity, circuit precision, or concept erasure success.

### Method

1. **Select matched feature pairs**: Identify SAE features with high vs low absorption rates on the same semantic dimension (using Chanin et al. differential correlation metric)

2. **Apply absorption reduction**: Use Matryoshka SAE architecture or OrtSAE decoder orthogonality to reduce absorption in the high-absorption feature

3. **Measure downstream utility**:
   - Steering effectiveness: probability lift on target tokens
   - Circuit discovery: precision/recall on truthfulness circuit identification
   - Concept erasure: accuracy on counterfactually edited outputs

4. **Compare**: High-absorption original vs low-absorption modified vs matched low-absorption control

### Experimental Plan

| Experiment | Duration | Resource |
|---|---|---|
| Feature pair identification (26 first-letter features, GPT-2 Small L4/L8) | ~15 min | Local GPU |
| Matryoshka SAE loading and absorption measurement | ~15 min | SAELens + GemmaScope |
| Steering comparison (high vs low absorption) | ~20 min | SSH to server |
| Circuit discovery comparison | ~20 min | SSH to server |
| Concept erasure comparison | ~20 min | SSH to server |

**Total**: ~1.5 hours (within project constraint of ~1 hour per task with some parallelization)

### Baselines

1. **High-absorption features without reduction**: Same features measured in prior iterations (mean=0.034 absorption, 100% steering success)
2. **Random SAE baseline**: Confirmed absorption = 0.278, provides structural control
3. **Non-absorption-reducing SAE architectures**: Standard JumpReLU SAE as control for architectural confounders

### Risk Assessment

**What if the mainstream view is correct?**
- If absorption reduction does improve downstream utility, this study validates the entire research direction
- The field gains a confirmed target for optimization
- We publish a confirmatory result (less impactful but still valuable)

**What if we're right?**
- The study invalidates a significant portion of current SAE research (Matryoshka, OrtSAE, ATM, H-SAE)
- Research resources should redirect to utility validation for other SAE properties
- Publication is highly controversial but high-impact

**What if the result is ambiguous?**
- Partial support for either side weakens the paper
- Mitigated by clear prediction: either absorption affects utility or it doesn't, and the experiment is designed to distinguish

### Novelty Claim

This is the **first systematic utility validation study** for SAE absorption mitigation. All prior work reports absorption metrics without demonstrating downstream task improvement. The Sanity Checks paper (arXiv:2602.14111) addresses random baselines on general metrics but does not specifically test whether absorption reduction via architectural modification improves utility. This study directly addresses the gap that justified the entire absorption mitigation research agenda.

**Novelty score estimate**: 8/10 — directly addresses unvalidated core assumption of multiple published papers

### Relationship to Prior Art

- **vs. Chanin et al.**: We extend beyond describing absorption to testing whether it matters for utility
- **vs. Matryoshka/OrtSAE**: We test whether their core contribution (absorption reduction) actually helps downstream tasks
- **vs. Sanity Checks**: We focus on absorption-specific utility, not general metric comparison
- **vs. Wang et al.**: We test absorption specifically, not general interpretability-utility correlation

---

## Summary for Writing

This contrarian analysis identifies a critical gap: the entire absorption mitigation research agenda rests on an unvalidated assumption that absorption degrades downstream tasks. Our prior experiments found zero significant correlation between absorption and steering/probing/EC50 across 12 tests. The proposed study directly tests whether absorption reduction improves downstream utility—providing definitive evidence for or against the research agenda. The front-runner candidate (The Wrong Problem Hypothesis) is provocative, actionable within project constraints, and would produce a publishable result regardless of outcome.
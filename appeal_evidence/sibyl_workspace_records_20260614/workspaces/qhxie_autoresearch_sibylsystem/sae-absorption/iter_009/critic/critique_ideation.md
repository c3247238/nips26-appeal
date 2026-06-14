# Ideation Critique

## Overall Assessment: 7/10

The research idea -- extending absorption measurement from first-letter spelling to entity-attribute knowledge hierarchies -- is well-motivated and addresses a real gap. The gap identification is sound: every published absorption measurement uses a single proxy task, and the field needs cross-domain characterization. The execution plan is comprehensive, covering four hierarchies, causal mechanism testing, benign/pathological diagnostics, and five correlational predictors. However, three ideation-level issues limit the paper's contribution.

## Strengths

### 1. Gap Identification is Correct and Timely
The observation that every published absorption benchmark uses first-letter spelling is factually correct as of April 2026. SAEBench, Chanin et al. (2024), and all downstream evaluations restrict absorption to this one task. The gap is real and the field needs cross-domain data.

### 2. Five Well-Articulated Research Questions
The research questions progress logically: existence of cross-domain variation (RQ1), confound disentanglement (RQ2), architecture validity (RQ3), causal mechanism (RQ4), and theoretical predictors (RQ5). Each is falsifiable with clear metrics.

### 3. Honest Prior Integration
The proposal incorporates 8 prior iterations of pilot evidence, openly reporting failures (GAS rho=0.12, CMI rho=0.044, Absorption Tax rho=0.08) alongside successes (activation patching d=1.33). This prevents the common trap of designing experiments to confirm priors.

### 4. Backup Ideas Are Substantive
The four backup ideas (controlled dictionary experiment, phase transition analysis, post-hoc feature correction, revised ITAC) are genuinely distinct research directions, not minor variants of the main plan. Backup 1 (separating encoder from dictionary) is particularly compelling and may be stronger than some elements of the main paper.

## Weaknesses

### 1. The Cross-Domain Extension Is Incremental

The core contribution is applying an existing measurement protocol (Chanin et al., 2024) to new datasets (RAVEL instead of sae-spelling). The methodology is adapted, not invented. The RAVEL dataset already exists. The SAEs already exist. The integrated-gradients attribution pipeline already exists. The novel element is the combination, not any individual component.

This is not inherently disqualifying -- many important papers are careful measurement studies -- but it means the paper's value depends entirely on whether the cross-domain results are surprising and reliable. The finding that "absorption rates vary across hierarchies" is not surprising: different hierarchies have different properties (class count, balance, probe quality, semantic structure), so different absorption rates are expected. What would be surprising is a specific, quantitative pattern (e.g., "absorption scales with class imbalance" or "absorption is zero for hierarchies below a certain depth"). The paper does not find such a pattern.

### 2. Benign/Pathological Framing Is Misaligned

The proposal frames the benign/pathological diagnostic as testing "whether absorption faithfully represents computational redundancy." But the implemented diagnostic (ablating parent direction from child decoder) tests decoder geometry, not computational redundancy. The ideation should have recognized that a decoder-level intervention does not answer a computation-level question. A computation-level test would require ablating the parent feature's activation and measuring whether the model's downstream behavior changes -- which is closer to the activation patching already performed in Section 5.1.

This misalignment between the research question (causal) and the methodology (geometric) is an ideation failure, not just an execution issue.

### 3. Too Many Hypotheses, Too Little Depth

Nine hypotheses (H1-H9) are tested across four phases. Two are supported (H1, H3), two are falsified (H2', H8), and five fail or receive partial support (H4, H5, H6, H7-crossdomain, H9). While the breadth is impressive, the depth on any single finding is shallow. The cross-domain characterization (the primary contribution) gets one section (Section 4). The causal mechanism gets one section (Section 5). The architecture comparison gets one section (Section 6). Each section raises questions that the paper does not have space to address:

- Why does Europe absorb at 90% while Africa absorbs at 4%? (mentioned but unexplained)
- Why does the concentrated/distributed divide exist? (hypothesized but untested)
- Why do all correlational predictors fail? (catalogued but not analyzed mechanistically)

A tighter paper with 4-5 hypotheses explored in depth would be more convincing than a survey of 9 hypotheses explored superficially.

### 4. The "Absorption Tax" Concept Is Unvalidated

The theoretical concept of an "Absorption Tax" -- the minimum additional L0 cost for absorption-free representation -- appears in the title of the proposal ("The Absorption Tax") but is reported as NOT SUPPORTED in the experiments (ranking rho=-0.20, concordance 50%). Including a failed concept in the title is misleading. The paper sensibly drops this from the final title, but the theoretical framing still appears in the methodology and appendix without contributing to the paper's story.

## Missed Opportunities

1. **Controlled dictionary experiment** (Backup 1) would isolate encoder vs. dictionary causation -- a mechanistic contribution stronger than the cross-domain characterization. The proposal correctly identifies this as high-novelty (9/10) but relegates it to a backup.

2. **Within-hierarchy variation analysis**: the per-class data (Europe 90% vs. Africa 4%) is more informative about absorption mechanisms than the cross-hierarchy comparison. Why do some classes absorb and others do not? Entity count, training data frequency, and probe confidence per class could be analyzed. This is barely discussed.

3. **Connection to SAE training dynamics**: the paper treats SAEs as black boxes (load pre-trained, measure absorption). Connecting absorption to training loss curves, feature death rates, or encoder weight norms during training would deepen the mechanistic understanding.

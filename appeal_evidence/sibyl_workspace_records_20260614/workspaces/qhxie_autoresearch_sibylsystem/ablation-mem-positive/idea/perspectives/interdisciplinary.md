# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Efficient Coding Hypothesis (Barlow, 1961; Olshausen & Field, 1996)** — The brain minimizes metabolic cost by representing information with sparse, efficient codes. Natural images are encoded with sparse coding where neurons activate only when relevant features are present. This principle predicts that neural representations should exhibit high sparsity and that redundant features get compressed.

   **Key insight for SAE absorption**: The efficient coding framework predicts that during SAE training, features with high contextual specificity (high CV) would be preserved because they carry unique information that cannot be compressed without loss. This provides a principled explanation for the variance paradox — absorbed features have high CV because efficient coding selectively preserves high-variance specialized information.

2. **Neural Reuse Theory (العصري & Park, 2020; Anderson, 2010)** — Cortical areas are recruited for multiple cognitive functions beyond their apparent primary purpose. A single neural population participates in many different tasks depending on context. This reuse creates "cross-situational" neurons that activate across diverse contexts.

   **Key insight for SAE absorption**: If neural reuse is a fundamental principle, absorbed features may represent the SAE analog of cross-situational neurons — features that get subsumed into more specific child features but retain residual cross-context utility. This could explain why high-CV absorbed features remain steerable despite being absorbed — they represent genuinely useful cross-context information that survives compression.

3. **Sparse Coding in V1 (Olshausen & Field, 1996)** — The seminal sparse coding paper showed that natural image statistics drive the emergence of Gabor-like receptive fields. The sparse coding objective (reconstruction + sparsity) produces features that are spatially localized, oriented, and frequency-selective.

   **Key insight for SAE absorption**: The V1 sparse coding model is essentially the same objective as SAE training. The absorption phenomenon may be an inevitable consequence of this objective when features have hierarchical structure (as natural images do). This suggests absorption is not a bug but a feature of the architecture when operating in regimes with hierarchical feature co-occurrence.

4. **Predictive Coding Framework (Rao & Ballard, 1999; Clark, 2013)** — The brain uses hierarchical predictive models where higher layers generate predictions and lower layers signal prediction errors. This creates bidirectional information flow where feedback connections modulate feedforward processing.

   **Key insight for SAE absorption**: In predictive coding, "absorption" corresponds to prediction error signals being mediated through hierarchical levels. The CV-steering correlation might reflect whether parent feature prediction errors are directly accessible (mediated regime) or suppressed by child feature compensation (bypass regime).

#### Physics / Statistical Mechanics

5. **Phase Transitions in Sparse Systems (Engel & Van den Broeck, 2001)** — Statistical mechanics provides frameworks for understanding phase transitions in neural networks. At critical points, small parameter changes cause qualitative shifts in solution structure. Finite-size scaling describes how transitions smooth out in finite systems.

   **Key insight for SAE absorption**: The finite-size scaling with ν=3 discovered in pilot experiments suggests absorption exhibits critical behavior. This aligns with statistical physics predictions that sparse coding systems undergo phase transitions. The critical point λ_c may be universal across architectures, suggesting a deep connection between sparsity and feature absorption.

6. **Hierarchical Organization in Critical Systems (Bak, 1996; Jensen, 1998)** — Self-organized criticality (SOC) produces hierarchical structures across scales. The key property is that systems naturally evolve toward critical states where avalanches of all sizes occur. Hierarchical organization emerges as a consequence of optimization for both efficiency and robustness.

   **Key insight for SAE absorption**: Absorption may reflect the SAE's tendency to organize features hierarchically, similar to how SOC systems create hierarchical structures. The hierarchical co-occurrence that causes absorption might be the natural outcome of organizing features for maximum efficiency. The "absorption" of parent features could be analogous to the consolidation of information at higher levels in a hierarchical system.

7. **Renormalization Group Theory (Wilson, 1975; Goldenfeld, 1992)** — Renormalization describes how effective theories emerge at different scales. Coarse-graining at different scales reveals which features are essential vs. redundant. Fixed points determine universal behavior across scales.

   **Key insight for SAE absorption**: The parent-child absorption relationship may be understood through renormalization — child features represent the coarse-grained description of parent features at a more abstract level. High-CV absorbed features may correspond to features that are "relevant" at multiple scales (hence high variance across contexts), while low-CV features are "irrelevant" at the parent level (fully absorbed into coarse-grained description).

#### Biology / Evolutionary Biology

8. **Gene Duplication and Divergence (Ohno, 1970; Conrad, 1990)** — Gene duplication creates redundant copies that can subsequently diverge in function. One copy retains the original function while others can explore new functions. This process drives evolutionary innovation and functional specialization.

   **Key insight for SAE absorption**: The absorption phenomenon maps to a specific type of gene duplication outcome: when one gene's function fully subsumes another's, the absorbed gene may retain vestigial functionality (explaining why activation patching recovers parent function). High-CV absorbed features may represent cases where duplication divergence was recent — the child hasn't fully optimized away the parent's specialized context sensitivity.

9. **Functional Redundancy and Degeneracy (Edelman & Gally, 2001; Tononi et al., 1999)** — Biological systems exhibit degeneracy: different structures perform the same function. This provides robustness against damage and enables evolutionary exploration. However, redundancy can also lead to "hidden" functionality that's not apparent in healthy tissue.

   **Key insight for SAE absorption**: Absorbed features represent functional degeneracy in the SAE dictionary. The decoder weights encode the degenerate mapping where parent and child features both map to similar outputs. The steering heterogeneity (high-CV vs low-CV) may reflect different types of degeneracy: cases where degeneracy preserves unique information (high-CV, steerable) vs. cases where degeneracy is pure redundancy (low-CV, non-steerable).

10. **Vestigial Structures and Molecular fossils (Wagner, 2014)** — Vestigial features retain reduced functionality from ancestral states. In molecular biology, vestigial proteins sometimes retain partial activity that can be recovered under certain conditions. These represent incomplete absorption of ancestral functions.

    **Key insight for SAE absorption**: The activation patching recovery (67.3% mean) demonstrates that absorbed parent features retain vestigial functionality. This is analogous to atavisms in evolutionary biology — the residual function of absorbed features can be recovered under specific interventions (zeroing child features). The magnitude of recovery may correlate with how recently the absorption occurred.

#### Information Theory

11. **Rate-Distortion Theory (Shannon, 1948; Cover & Thomas, 1991)** — Lossy compression must trade off between bitrate and distortion. The rate-distortion function determines minimum achievable distortion at a given bitrate. High-variance signal components require more bits to preserve accurately.

    **Key insight for SAE absorption**: The variance paradox (absorbed features have 733x higher CV) reflects rate-distortion constraints: SAE training must allocate limited "bits" (active latents) across features. High-variance features consume more capacity and are more likely to get absorbed because they have more information to compress. The CV reflects how much unique information a feature carries — absorbed features have high CV because they carry information that cannot be discarded without significant distortion.

12. **Sparse Coding and Dictionary Learning (Olshausen & Field, 1996; Mairal et al., 2008)** — Sparse coding algorithms learn dictionaries that represent data efficiently. The learned atoms often correspond to meaningful features. Hierarchical dictionaries can capture multi-scale structure.

    **Key insight for SAE absorption**: Standard sparse coding doesn't explicitly model hierarchical relationships between features. When training data has hierarchical feature co-occurrence (as natural language does), the sparse objective will preferentially allocate capacity to more specific child features at the expense of general parent features. This is the underlying mechanism of absorption — not a bug but the mathematically optimal solution when hierarchical structure exists.

### Cross-Disciplinary Gaps

**Underexplored connections**:
1. **Neural reuse vs SAE feature absorption**: No prior work connects neural reuse theory to SAE absorption. The insight that absorbed features might retain cross-context utility (recovered via activation patching) is analogous to neural reuse in the brain.

2. **Gene duplication divergence angle**: The evolutionary biology perspective on gene duplication divergence hasn't been mapped to SAE feature hierarchies. The "recency of absorption" hypothesis (high-CV = recent absorption) is an untested analog of recent gene duplication.

3. **Rate-distortion theory for absorption**: While rate-distortion theory is mentioned in the literature, no prior work explicitly uses it to explain the variance paradox (733x CV difference between absorbed and non-absorbed features).

4. **Vestigial functionality recovery**: The activation patching recovery (67.3%) showing absorbed features retain vestigial parent functionality is analogous to atavism in evolutionary biology — not explored in SAE literature.

---

## Phase 2: Initial Candidates

### Candidate A: Neural Reuse and Cross-Context Utility (from Neuroscience)

- **Source principle**: Neural reuse theory (Anderson, 2010) — cortical areas are recruited for multiple functions beyond their primary purpose, creating "cross-situational" neurons that activate across diverse contexts.

- **Structural correspondence**: In SAEs, absorbed features may function like cross-situational neurons — the child feature subsumes the parent's function but retains vestigial cross-context utility. The CV measures how cross-context the feature is: high-CV features activate in diverse contexts (high variance), suggesting they may retain more cross-context utility even after absorption.

- **Hypothesis**: Absorbed features with high CV (high cross-context variance) retain measurable cross-context utility because they were recently "reused" across contexts before full absorption. Activation patching reveals this vestigial utility. Low-CV features are fully absorbed (no reuse residue) and show zero steering effect.

- **Why it's not just a metaphor**: Neural reuse predicts specific experimental observations — features with high cross-context activation should show larger activation patching recovery. This is testable by correlating CV with patching recovery magnitude, not just steering effectiveness.

- **Novelty estimate**: 8/10 — No prior work connects neural reuse theory to SAE absorption. The cross-context utility explanation is distinct from the bypass/mediated regime explanation proposed in other perspectives.

### Candidate B: Gene Duplication Divergence (from Evolutionary Biology)

- **Source principle**: Gene duplication and divergence (Ohno, 1970) — after duplication, one copy retains original function while others specialize. Incomplete divergence leaves "dual-function" genes that can act as both parent and child depending on context.

- **Structural correspondence**: Absorbed SAE features are like genes that have partially diverged — the child feature dominates (high activation frequency) but the parent feature retains vestigial function. The CV measures divergence stage: high-CV absorbed features are recently duplicated (more parent-like, more steerable), low-CV are fully diverged (more child-like, less steerable).

- **Hypothesis**: Absorption is an evolutionary process where parent features get "duplicated" into more specific child features. Features with high CV are early-stage duplication (more parent utility retained). Features with low CV are late-stage duplication (parent utility fully subsumed).

- **Why it's not just a metaphor**: Gene duplication predicts a specific correlation — features with high CV should show more parent-like behavior in ablation experiments (higher activation patching recovery). This is empirically testable.

- **Novelty estimate**: 7/10 — No prior work applies gene duplication theory to SAE feature hierarchies. The "recency of absorption" concept is novel.

### Candidate C: Rate-Distortion Optimal Compression (from Information Theory)

- **Source principle**: Rate-distortion theory (Shannon, 1948) — lossy compression preserves high-variance components essential for reconstruction fidelity. Low-variance components get compressed more aggressively because they contribute less to reconstruction error.

- **Structural correspondence**: SAE training performs rate-distortion optimization. Features with high CV carry more unique information (high variance = more discriminative) and are therefore preserved at the expense of low-CV features. This explains the variance paradox: absorbed features have high CV because they were too information-rich to compress away completely.

- **Hypothesis**: The CV-steering correlation reflects that high-CV absorbed features retained their steering potential because they carry unique reconstruction-critical information. Low-CV absorbed features lost steering potential because they were "safe to compress" — their information was recoverable from other features.

- **Why it's not just a metaphor**: Rate-distortion theory makes specific quantitative predictions: the compression ratio should correlate with the variance ratio. This could be tested by measuring how decoder weight magnitudes (a proxy for reconstruction importance) correlate with CV and steering effectiveness.

- **Novelty estimate**: 6/10 — Rate-distortion is mentioned in the literature survey but not specifically connected to the CV-steering correlation or variance paradox.

---

## Phase 3: Self-Critique

### Against Candidate A (Neural Reuse)

- **Shallow analogy attack**: Neural reuse refers to neural populations being recruited for multiple cognitive functions. SAE absorption refers to latent features being subsumed. The mapping may be superficially vocabulary-matched but structurally different — neurons are biological substrate, latents are mathematical objects.

- **Scale mismatch attack**: Neural reuse operates at the level of brain regions (macroscale) with billions of neurons. SAE absorption operates at the level of features in a 16k-dimensional latent space. The mechanisms may not transfer across these scales.

- **Prior transplant check**: No prior work connects neural reuse theory to SAE analysis. The analogy is genuinely unexplored, but this also means there's no validation.

- **Testability attack**: Can we design an experiment that distinguishes "this works because of neural reuse" from "this works because of generic sparsity"? The prediction (high-CV = larger patching recovery) is the same as the baseline prediction from the front-runner hypothesis.

- **Verdict**: MODERATE — The analogy is suggestive but not clearly more explanatory than the bypass/mediated regime mechanism. However, the "cross-context utility" framing may be more intuitive for practitioners.

### Against Candidate B (Gene Duplication)

- **Shallow analogy attack**: Gene duplication involves DNA, transcription, and protein synthesis. SAE absorption involves encoder-decoder weights and sparse activations. The biological substrate is entirely different — this is a vocabulary match, not a structural correspondence.

- **Scale mismatch attack**: Gene duplication occurs over evolutionary timescales with random mutations. SAE absorption occurs during a single training run with backpropagation. The dynamics are fundamentally different.

- **Prior transplant check**: No prior work applies evolutionary biology to SAE analysis. This is genuinely novel but may be too far-fetched.

- **Testability attack**: The "recency of absorption" hypothesis is testable (correlation between CV and patching recovery), but this is the same prediction as the baseline CV-steering hypothesis.

- **Verdict**: WEAK — The analogy is too superficial. Gene duplication and SAE absorption share only the word "duplication/hierarchy." The mechanism is entirely different.

### Against Candidate C (Rate-Distortion)

- **Shallow analogy attack**: Rate-distortion theory applies to communication channels and compression algorithms. SAEs are trained via gradient descent on a reconstruction + sparsity objective. The optimization dynamics differ from classic rate-distortion.

- **Scale mismatch attack**: Rate-distortion theory typically analyzes i.i.d. sources. Natural language has complex dependencies and hierarchical structure that violate i.i.d. assumptions. The theory may not directly apply.

- **Prior transplant check**: Rate-distortion is mentioned in the literature survey but not specifically connected to CV-steering correlation. However, the general connection between sparsity and variance preservation is well-known.

- **Testability attack**: The prediction (decoder magnitude correlates with CV) is testable and has been partially validated (Fano factor analysis is mentioned as control). But rate-distortion is a macroscopic theory — it may not explain the detailed steering heterogeneity.

- **Verdict**: MODERATE — Rate-distortion provides a principled explanation for the variance paradox (why absorbed features have high CV) but doesn't add new empirical predictions beyond what the front-runner already claims.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Gene Duplication) dropped because**: The analogy is too superficial. Gene duplication involves biological mechanisms (DNA, transcription, mutation) that have no analog in SAE training (gradient descent on reconstruction loss). The prediction is identical to the baseline CV-steering hypothesis, so it adds no value.

### Strengthened Ideas

- **Candidate A (Neural Reuse) strengthened by**:
  - Cross-context utility is a distinct concept from bypass/mediated regime — could provide alternative explanation
  - The "recovered vestigial functionality" framing is more intuitive for describing activation patching results
  - Neural reuse is established neuroscience theory with decades of empirical support

- **Candidate C (Rate-Distortion) strengthened by**:
  - Provides principled explanation for the variance paradox (733x CV difference)
  - Connects to information-theoretic bounds from Cui et al. (ICLR 2026)
  - The Fano factor control experiment directly tests whether CV is just a magnitude proxy

### Selected Front-Runner

**Candidate C: Rate-Distortion Optimal Compression**

The variance paradox (absorbed features have 733x higher CV than non-absorbed) is most directly explained by rate-distortion theory: the SAE training process preferentially preserves high-variance features because they carry unique reconstruction-critical information. The CV-steering correlation follows because steering effectiveness depends on how much unique information a feature carries — high-CV absorbed features retained their unique information (hence steerable), low-CV absorbed features were "safe to compress" (hence not steerable).

This connects to the information-theoretic bounds from Cui et al. and provides the cleanest explanation for the empirical findings. Neural reuse is retained as secondary framing.

---

## Phase 5: Final Proposal

### Title

**Rate-Distortion Theory Explains the Variance Paradox in SAE Feature Absorption**

### Source Principle

Rate-distortion theory (Shannon, 1948; Cover & Thomas, 1991) states that lossy compression must trade off reconstruction fidelity (distortion) against bitrate. High-variance signal components require more bits to preserve accurately and are therefore preserved at the expense of low-variance components. When compression is aggressive, low-variance components get fully absorbed into high-variance ones.

### Structural Correspondence

SAE training performs rate-distortion optimization:
- **Bitrate**: The sparsity constraint (k active latents) limits the bitrate
- **Distortion**: The reconstruction loss measures how well the SAE approximates the original activation
- **Feature variance**: The coefficient of variation (CV) measures how much unique information a feature carries

When features have hierarchical co-occurrence structure (as natural language does), the sparse optimization preferentially allocates the limited k latents to high-CV child features. Parent features (often lower CV) get absorbed because their information is "compressible" — recoverable from the child features.

The variance paradox (absorbed features have CV ≈ 7.33 vs non-absorbed CV ≈ 0.01) reflects this selective preservation: absorbed features have high CV precisely because they carry high-variance information that was too valuable to fully discard.

### Hypothesis

The coefficient of variation (CV = σ/μ) measures the unique information density of a feature. Absorbed features with high CV retained their steering potential because they carry information that cannot be fully recovered from other features — they are "rate-distortion optimal" survivors.

The CV-steering correlation follows: high-CV absorbed features remain steerable because they retained unique reconstruction-critical information (rate-distortion optimal preservation). Low-CV absorbed features lost steering potential because their information was safely compressible — recoverable from other features (not rate-distortion critical).

Formally: If S_f is steering effectiveness and CV_f is coefficient of variation, then E[S_f | CV_f > τ, f ∈ F_absorbed] > E[S_f | CV_f ≤ τ, f ∈ F_absorbed].

### Method

**Phase 1: Confirm Rate-Distortion Mechanism (20 min)**
- Test whether decoder weight magnitudes (proxy for reconstruction importance) correlate with CV
- Compute Fano factor (CV²/μ) to control for magnitude — if CV is purely magnitude proxy, correlation should disappear after Fano factor control
- If CV-steering correlation persists after Fano factor control, rate-distortion mechanism is supported

**Phase 2: Steering Effectiveness Test (30 min)**
- 30 high-CV vs 30 low-CV absorbed features on GPT-2 layer 6
- Steering strength +5, measure logit change
- One-sided Welch's t-test (α = 0.01) with Benjamini-Hochberg FDR correction

**Phase 3: Activation Patching Correlation (20 min)**
- Test whether CV correlates with activation patching recovery magnitude
- High-CV should show larger recovery (consistent with rate-distortion preservation of parent information)

**Phase 4: Cross-Architecture Validation (30 min)**
- Replicate on Gemma-2-2B layer 6 JumpReLU SAE
- Test whether CV threshold generalizes

### Diagnostic Experiment

The key diagnostic experiment is the Fano factor control:

If CV is purely a magnitude proxy (high-CV = high activation = more steering), then controlling for mean activation magnitude (Fano factor = CV²/μ) should eliminate the CV-steering correlation.

If CV-steering correlation persists after Fano factor control, then CV captures something beyond magnitude — specifically, the unique information density that rate-distortion theory predicts.

This experiment distinguishes "CV is just a magnitude proxy" from "CV captures rate-distortion optimality."

### Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: Rate-distortion validation | Decoder magnitude vs CV correlation | 15 min | Rate-distortion mechanism |
| E2: Fano factor control | CV-steering after magnitude control | 20 min | CV is not magnitude proxy |
| E3: Steering comparison | 30 high-CV vs 30 low-CV absorbed features | 30 min | Core hypothesis |
| E4: Patching-CV correlation | CV vs activation patching recovery | 20 min | Mechanism connection |
| E5: Gemma-2 validation | Gemma-2-2B layer 6 | 30 min | Cross-model generalization |

**Total**: ~115 min across 5 experiments (within project budget)

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Fano factor control eliminates CV-steering correlation | Medium | If CV is magnitude proxy, rate-distortion explanation fails but empirical correlation remains useful |
| Cross-architecture CV threshold differs | Medium | Gemma-2 may require different threshold — report as model-specific finding |
| Rate-distortion doesn't explain patching recovery | Low | May need additional mechanism (neural reuse framing as backup) |

### Novelty Claim

This is the **first application of rate-distortion theory to explain the variance paradox and CV-steering correlation in SAE feature absorption**.

Prior work establishes:
- Absorption exists and is caused by hierarchical co-occurrence (Chanin et al.)
- The actionability paradox — absorbed features appear non-steerable (Basu et al.)
- Theoretical limits on disentanglement (Cui et al.)

This work provides:
1. **Explanatory mechanism** for the variance paradox (733x CV difference) — rate-distortion optimal preservation of high-variance features
2. **Unified explanation** for both the CV-steering correlation and the activation patching recovery — absorbed features are rate-distortion survivors
3. **Diagnostic experiment** (Fano factor control) that distinguishes magnitude proxy from information density interpretation

The connection to rate-distortion theory is novel and provides principled theoretical grounding beyond the empirical correlation.

### Cross-Domain Contribution

The interdisciplinary contribution is demonstrating that SAE feature absorption is a rate-distortion phenomenon, not merely a training artifact. This reframes absorption as an inevitable consequence of optimizing reconstruction under sparsity constraints when features have hierarchical structure — similar to how lossy compression preserves high-variance signal components.

### References

- Shannon (1948): Mathematical Theory of Communication — rate-distortion foundation
- Cover & Thomas (1991): Elements of Information Theory — rate-distortion theory
- Olshausen & Field (1996): Sparse coding with overcomplete dictionaries — sparse representation theory
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning — phase transitions
- Cui et al. (2026): On the Limits of SAEs — information-theoretic impossibility
- Chanin et al. (2024): A is for Absorption — absorption detection
- Basu et al. (2026): Interpretability without Actionability — actionability paradox
- Anderson (2010): The Rui of Neural Reuse — neural reuse theory
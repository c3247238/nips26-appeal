# Backup Ideas for Pivot (Synthesis Round 5)

## Backup A: Controlled Dictionary Experiment -- Separating Encoder from Dictionary

**Status:** High-priority backup. Can be activated immediately with no new downloads.

**Idea:** Take the SAME pre-trained decoder dictionary D from a Gemma Scope SAE and measure absorption rates under three encoding strategies: (1) standard feedforward encoding, (2) Orthogonal Matching Pursuit (OMP) encoding, and (3) 2-pass encoding (feedforward init + OMP refinement). This definitively separates the Tang et al. explanation (sparsity landscape / dictionary) from the O'Neill et al. explanation (amortization gap / encoder).

**Why this matters:** The pilot's activation patching result (14.3% parent recovery when child is zeroed) confirms competitive exclusion at the encoder level. If OMP encoding with the same dictionary shows dramatically less absorption, it proves the feedforward encoder is the bottleneck -- changing mitigation prescriptions from "train wider dictionaries" to "fix the encoder."

**Hypotheses:**
- If amortization gap dominates: OMP absorption rate < 50% of feedforward at matched L0
- If sparsity landscape dominates: absorption rates similar across encoding methods
- 2-pass shows intermediate absorption, confirming both factors contribute

**Compute:** 1-2 GPU-hours. No new downloads. Run on Gemma Scope L12-16k.

**Novelty:** 9/10. First controlled dictionary experiment separating encoder from dictionary. MP-SAE (Costa et al.) uses iterative encoding but does NOT hold dictionary constant.

**Pivot trigger:** Activate if cross-domain probes fail on ALL hierarchies AND activation patching at scale shows no effect. This experiment works regardless of probe quality since it uses the existing first-letter metric.

---

## Backup B: Ecological Phase Transition Theory -- Limiting Similarity in SAE Features

**Status:** Theoretical backup. Strengthened by pilot data showing hierarchy-dependent absorption.

**Idea:** Formalize the pilot's key finding (absorption varies by hierarchy type) through the Lotka-Volterra competition framework. The cross-domain data from Phase 1 enables a direct test: plot absorption probability vs. decoder cosine similarity for all feature pairs across hierarchy types. The ecological model predicts a sharp phase transition at critical cosine similarity theta*(W, L0, freq_ratio), with DIFFERENT thresholds for different hierarchy types.

**Why this strengthened:** The pilot falsified H2 (first-letter is not worst case), suggesting hierarchy structure matters more than previously thought. The Lotka-Volterra framework provides a quantitative explanation: different hierarchies have different "niche overlap" distributions (alpha_pc distributions), leading to different absorption rates. This is precisely what the pilot observed.

**Connection to front-runner:** Can be run on the same data from Phase 1 of the main proposal. The ecological framework provides the theoretical explanation for WHY cross-domain absorption rates differ -- complementing the empirical characterization.

**Hypotheses:**
- Phase transition: absorption probability sigmoid with transition width < 0.1 in cosine similarity space
- Hierarchy-dependent theta*: first-letter theta* < city-language theta* < city-continent theta* (matching observed absorption ordering)
- Lotka-Volterra scaling: theta* ~ sqrt(1/W) * (L0)^(-1/2) * g(freq_ratio)

**Compute:** 2-3 GPU-hours using Phase 1 data.

**Novelty:** 9/10. Zero prior work on competitive exclusion in SAE context. The hierarchy-dependent phase transition prediction is entirely novel.

**Pivot trigger:** Activate as supplementary analysis whenever Phase 1 cross-domain data is available.

---

## Backup C: Absorption Scaling Laws from Cross-Domain Data

**Status:** Low-risk supplementary analysis.

**Idea:** Fit parametric scaling models to absorption rates across the cross-domain Phase 1 data: absorption ~ C * f(hierarchy_type, co_occurrence_density, probe_accuracy) * g(width, L0). The key advance over generic SAEBench curve fitting: the cross-domain axis provides a new dimension to fit against.

**Compute:** 30 min CPU using Phase 1 data.

**Novelty:** 6/10. Quantitative fit is new; qualitative trends partially known.

**Pivot trigger:** Always run as supplementary. Not activatable as primary.

---

## Backup D: Alternative Unsupervised Detection (Post-GAS)

**Status:** Deferred. Requires new methodological ideas.

**Idea:** Given that GAS failed (rho=0.12), explore alternative unsupervised detection approaches:
- **Activation patching-based detection:** Zero each feature systematically and measure recovery of other features. O(d_sae^2) but can be done on candidate pairs only.
- **Meta-SAE decomposition:** Use Leask et al.'s meta-SAE approach to detect whether "atomic" features contain absorbed parent information.
- **Encoder gradient analysis:** Measure the gradient of the encoder with respect to absorbed-parent probe directions. If the encoder has learned to suppress the parent, the gradient should show a characteristic pattern.

**Why deferred:** Each approach has significant computational or methodological challenges. The GAS failure suggests that static geometric signals (decoder similarity) are insufficient -- dynamic signals (how features respond to perturbations) may be required.

**Novelty:** Varies by approach (7-9/10).

**Pivot trigger:** Activate after GAS failure is documented and analyzed in the main paper. Best suited for a follow-up paper rather than the current one.

---

## Backup E: Absorption-Aware Post-Hoc Correction (from Contrarian)

**Status:** Medium-priority backup. Activated if the contrarian's question ("does fixing absorption actually help?") becomes central.

**Idea:** Given known absorption patterns (from Phase 1), construct "corrected" feature activations: when a child feature fires, automatically re-activate the absorbed parent feature. Evaluate whether this simple post-hoc correction recovers lost information without architectural changes.

**Why this matters:** If post-hoc correction is effective, it challenges the need for architectural mitigations (Matryoshka, OrtSAE, ATM). If it is ineffective, it confirms that absorption permanently destroys parent information.

**Hypotheses:**
- Post-hoc correction recovers >= 50% of the SAE-vs-probe gap on downstream tasks
- Or: post-hoc correction has negligible effect because absorption permanently rewires the decoder direction

**Compute:** 1-2 GPU-hours.

**Novelty:** 8/10. First evaluation of post-hoc absorption correction.

**Pivot trigger:** Activate if reviewers ask "so what should practitioners do about absorption?" and the main paper lacks prescriptive guidance.

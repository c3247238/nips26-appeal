# Backup Ideas for Pivot (Iteration 10)

## Backup 1: Controlled Dictionary Experiment -- Separating Encoder from Dictionary in Absorption Causation

**Candidate ID:** `cand_controlled_dictionary`
**Novelty:** 8/10
**Status:** Backup (highest-priority pivot option)

**Hypothesis:** Feature absorption is primarily an encoder failure (the feedforward encoder makes a greedy decision to suppress the parent), not a dictionary failure (the decoder lacks capacity to represent both parent and child). Holding the decoder dictionary constant while varying the encoder (feedforward vs. OMP vs. 2-pass) will isolate the causal locus.

**Method:**
1. Load a Gemma Scope SAE (L12, 16k width) -- extract the decoder dictionary W_dec
2. Compute 3 different encodings of the same input activations:
   - Standard feedforward: z = ReLU(W_enc * x + b)
   - Orthogonal Matching Pursuit (OMP): greedy sparse coding against W_dec
   - 2-pass: feedforward first pass, then re-encode residual
3. For each encoding: compute absorption rate on first-letter task using Chanin et al. metric
4. If OMP absorption << feedforward absorption at matched L0, the encoder is the causal locus
5. If absorption similar across encoders, the dictionary/sparsity landscape is the cause

**Why this is a strong backup:**
- First controlled dictionary experiment in the SAE literature (9/10 novelty)
- Pilot activation patching results (14.3% recovery) suggest encoder-level competitive exclusion, which this directly tests
- Quick to implement: ~1-2 GPU-hours
- Independent of probe quality concerns (uses the same first-letter task as Chanin et al.)
- Always valuable as a supplementary experiment even if the cross-domain paper succeeds

**Pivot trigger:** Activate if probe degradation ablation (H10) reveals cross-domain variation is entirely a probe artifact, undermining the primary contribution.

---

## Backup 2: Feature Absorption as Competitive Exclusion -- Phase Transitions and Limiting Similarity

**Candidate ID:** `cand_ecological_phase_transition`
**Novelty:** 9/10
**Status:** Backup (strongest theoretical contribution)

**Hypothesis:** Feature absorption exhibits sharp phase transitions at critical decoder cosine similarity thresholds, analogous to Lotka-Volterra competitive exclusion in ecology. The critical threshold theta* is hierarchy-dependent: first-letter < city-language < city-continent (matching the pilot absorption ordering).

**Method:**
1. From the cross-domain Phase 1 data, compute decoder cosine similarity for all identified parent-child feature pairs
2. Plot absorption probability as a function of decoder cosine similarity
3. Test for sigmoid-shaped phase transition (logistic fit R^2 > 0.7)
4. Estimate transition width -- sharp transition (width < 0.1) supports phase-transition model
5. Test whether theta* differs across hierarchy types
6. Fit Lotka-Volterra scaling: theta* ~ sqrt(1/W) * (L0)^(-1/2)

**Why this is a strong backup:**
- Zero prior work on competitive exclusion or phase transitions in SAE absorption
- Provides the theoretical "why" for cross-domain variation (different hierarchies have different theta*)
- Can be computed from Phase 1 data with zero additional GPU cost
- If the phase-transition model fits well, it subsumes both the cross-domain characterization and the rate-distortion predictor framework

**Pivot trigger:** Activate as supplementary analysis when Phase 1 data is available. Particularly strong if H9 (three-factor predictor) succeeds (cos_sim is one of the three factors).

---

## Backup 3: Absorption-Aware Post-Hoc Feature Correction

**Candidate ID:** `cand_absorption_aware_correction`
**Novelty:** 8/10
**Status:** Backup (prescriptive contribution)

**Hypothesis:** Given known absorption patterns (parent-child pairs with measured absorption), a post-hoc correction that re-activates absorbed parent features when child features fire can recover >= 50% of the SAE-vs-probe gap on downstream tasks.

**Method:**
1. For each confirmed absorption instance from Phase 1:
   - When child feature fires and parent feature does not, inject the parent's decoder direction into the SAE reconstruction
   - Scale injection by the average parent activation magnitude when it does fire
2. Measure downstream metrics: probe accuracy, CE loss recovered, spurious correlation removal
3. Compare: original SAE vs. corrected SAE vs. dense probe ceiling
4. Report recovery fraction: how much of the probe-SAE gap does correction close?

**Why this is a strong backup:**
- All existing absorption mitigations are training-time architectural changes (Matryoshka, OrtSAE, ATM). Post-hoc inference-time correction is completely unexplored.
- Practical value: researchers with existing trained SAEs cannot retrain them. Post-hoc correction is immediately deployable.
- If it works: first inference-time absorption mitigation
- If it fails: demonstrates absorption permanently alters decoder geometry, which is also informative

**Pivot trigger:** Activate if reviewers ask for prescriptive guidance (common reviewer critique: "this paper characterizes the problem but does not propose a solution"). Also activate if H8 reframing shows most absorbed parent information IS recoverable from child decoders.

---

## Backup 4: Within-Hierarchy Variation Analysis -- Why Europe Absorbs at 90% While Africa Absorbs at 4%

**Candidate ID:** `cand_within_hierarchy_variation`
**Novelty:** 7/10
**Status:** Backup (depth alternative to breadth)

**Hypothesis:** Within a single hierarchy (e.g., city-continent), absorption rates vary dramatically across parent classes (Europe: 90%, Africa: 4% in pilot data). This variation is predictable from: (a) number of child entities per parent (class size), (b) training data frequency of parent-child co-occurrences, and (c) decoder cosine similarity between parent and most-absorbed child.

**Method:**
1. For city-continent at L24 16k: compute per-class absorption rate for each continent
2. Compute per-class predictors: class size (number of cities), estimated training frequency (Wikipedia page view proxy), decoder cosine similarity
3. Regression: per_class_absorption ~ class_size + log(frequency) + cos_sim
4. Cross-validate on city-country (per-country absorption variation)
5. Report which predictors explain the most variance

**Why this is a strong backup:**
- The within-hierarchy variation (90% vs. 4%) is potentially more informative about absorption mechanisms than the cross-hierarchy comparison
- Addresses the iter-9 critique that "the paper has breadth but not depth"
- Quick to compute from existing data (~30 min CPU)
- Could explain WHY semantic hierarchies show higher absorption: they have more extreme class imbalance

**Pivot trigger:** Activate if cross-domain variation becomes less interesting after probe degradation ablation, or if reviewers request deeper mechanistic analysis within a single hierarchy.

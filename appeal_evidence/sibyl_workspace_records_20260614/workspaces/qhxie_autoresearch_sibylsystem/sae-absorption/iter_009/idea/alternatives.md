# Backup Ideas for Potential Pivot

## Backup 1: Controlled Dictionary Experiment -- Separating Encoder from Dictionary in Absorption Causation

### Motivation
All absorption research conflates two possible causes: (1) the SAE encoder (a learned linear map + nonlinearity) makes suboptimal routing decisions, absorbing parent features because the feedforward pass is too greedy, vs. (2) the dictionary itself is structured so that absorption is the only way to reconstruct well. No experiment has separated these. Pilot activation patching results (14.3% parent recovery when child zeroed) suggest the encoder-level competitive exclusion is real, but the controlled dictionary experiment would definitively isolate the mechanism.

### Method
Hold the SAE decoder dictionary constant (use a pre-trained Gemma Scope SAE's decoder weights). Replace the encoder with:
- **Feedforward encoder** (standard, the learned linear map + activation)
- **Orthogonal Matching Pursuit (OMP)** (optimal k-sparse encoding given the dictionary)
- **Two-pass encoder** (first pass identifies top-k features, second pass re-encodes accounting for parent-child relationships)

For each encoder, measure absorption rate on the first-letter task and at least one RAVEL hierarchy. If OMP shows dramatically less absorption than the feedforward encoder at matched L0, the encoder (amortization gap) is the bottleneck. If absorption is similar across all encoders, the dictionary structure itself causes absorption.

### Novelty
9/10. No prior work holds the dictionary constant and varies the encoder to isolate absorption causation. The closest is Matching Pursuit SAE (Loshchilov et al.), which changes both encoder and dictionary.

### Resource Estimate
1-2 GPU-hours. Uses existing pre-trained decoder weights.

### Pivot Trigger
Activate if cross-domain probe quality remains below 0.85 at all layers (making Phase 1 unreliable) AND activation patching effect disappears at larger n. Also valuable as supplementary experiment regardless.

---

## Backup 2: Feature Absorption as Competitive Exclusion -- Phase Transitions and Limiting Similarity

### Motivation
The Interdisciplinary perspective's rate-distortion phase transition framework and the Innovator's ecological competition theory both predict that absorption should exhibit a sharp onset as a function of sparsity level. If confirmed, this would be the first theoretical prediction about absorption that is both quantitative and validated on real SAEs. The cross-domain data from Phase 1 (which spans multiple hierarchy types with different co-occurrence statistics) provides exactly the data needed to test universality.

### Method
Using all absorption measurements from Phase 1 (across multiple SAE configurations and hierarchy types):
1. Plot absorption rate vs. effective sparsity parameter (L0 or lambda) for each hierarchy type
2. Test for sharp onset (phase transition) vs. gradual increase (no critical point)
3. If sharp: fit scaling law A ~ (lambda - lambda_c)^gamma near the critical point
4. Test universality: compare gamma across hierarchy types (predicted: same gamma, different lambda_c)
5. Test the Lotka-Volterra predictor: compute decoder cosine similarity and maximum activation for all feature pairs, predict which pairs exhibit absorption

### Novelty
9/10. No prior work models absorption as a phase transition or derives scaling laws for absorption onset. The ecological competition coefficient as a pre-training predictor of absorption is also novel.

### Resource Estimate
2-3 GPU-hours, mostly reusing Phase 1 data. Additional compute only for computing decoder cosine similarity matrices and fitting scaling laws.

### Pivot Trigger
Activate as supplementary analysis when Phase 1 data is available. Provides theoretical explanation for WHY cross-domain rates differ. Particularly valuable if cross-domain absorption variation is confirmed but the rate-distortion three-factor model (H9) fails.

---

## Backup 3: Absorption-Aware Post-Hoc Feature Correction

### Motivation
The Contrarian perspective raises an important practical question: given known absorption patterns, can we correct SAE feature activations at inference time without retraining? If a child feature fires and we know (from prior analysis) that it absorbs a specific parent feature, we can re-activate the parent feature post-hoc. This converts absorption knowledge into a practical tool.

### Method
1. From Phase 1, identify all absorbed parent-child pairs (using the Chanin et al. metric)
2. For each absorbed pair: when child fires and parent does not, inject the parent feature activation with magnitude estimated from the child's activation and the decoder alignment
3. Measure the effect on downstream task performance: sparse probing F1, circuit discovery accuracy, concept steering effectiveness
4. Compare: corrected SAE vs. uncorrected SAE vs. dense linear probe (ceiling)

### Novelty
8/10. First evaluation of post-hoc absorption correction. All existing mitigation approaches require SAE retraining (Matryoshka, OrtSAE, ATM, masked regularization).

### Resource Estimate
1-2 GPU-hours. Uses known absorption patterns from Phase 1.

### Pivot Trigger
Activate if reviewers ask for prescriptive guidance ("so what should practitioners DO about absorption?") or if the benign-vs-pathological analysis (H8) shows that most absorption is pathological. Also valuable as a practical contribution alongside the characterization paper.

---

## Backup 4: Information-Theoretic Absorption Completeness (ITAC) -- Revised Approach

### Motivation
The Innovator's ITAC proposal was ambitious but the pilot evidence refuted the purely geometric approach (GAS rho=0.12). However, the conditional independence component (Component 2 of ITAC) was never tested. A revised ITAC that focuses on activation-based conditional independence rather than decoder geometry alone may succeed where GAS failed.

### Method
1. For each SAE feature pair with high decoder cosine similarity (> 0.5):
   - Collect activation data on diverse text corpus (10k tokens)
   - Test whether parent direction's activation is conditionally independent of the input given the child's activation
   - If conditionally independent: absorption detected
   - If conditionally dependent: legitimate feature composition (not absorption)
2. Validate against known absorption (first-letter task + RAVEL hierarchies from Phase 1)
3. Compute Absorption Completeness Score (ACS) = fraction of directions "owned" by dedicated features

### Novelty
8/10. The conditional independence test for distinguishing absorption from composition is novel. GAS failed but the activation-based component was not tested.

### Resource Estimate
3-4 GPU-hours (requires activation collection on large corpus).

### Pivot Trigger
Activate only if the community demand for unsupervised absorption detection is strong (reviewer feedback) and the conditional independence component shows promise in a quick pilot (30 min). Lower priority than Backups 1-3 because GAS failure suggests geometric/statistical approaches face fundamental limitations.

# Backup Ideas for Pivot

## Alternative A: Temporal Dynamics of Absorption Emergence

**Core question**: When does absorption emerge during SAE training? Can we detect early warning signals and prevent absorption via curriculum learning?

**Why it's viable**:
- Directly extends the front-runner: if absorption *does* cause harm (RQ2 positive), preventing it is the natural next step.
- Innovator perspective's core contribution, partially inspired by crystal nucleation analogy.
- "Time-Aware Feature Selection" (Li & Ren, 2025) shows temporal masking reduces absorption, but does not track emergence dynamics.

**Approach**:
1. Log absorption metrics every N training steps during SAE training
2. Identify "absorption onset point"---the training step where absorption rate exceeds baseline noise
3. Test curriculum strategies: gradually increase sparsity (sparsity annealing), early stopping at pre-onset point
4. Connect to feature hedging dynamics (Bussmann et al. 2025): does hedging precede absorption?

**Pivot trigger**: If RQ2 shows strong causal link (absorption causes harm), pivot to prevention.

**Risk**: Requires training SAEs from scratch (expensive). Mitigation: small models, short runs, or use SAELens training pipeline with hooks.

---

## Alternative B: Cross-Layer Absorption Propagation

**Core question**: Does absorption in early-layer SAEs propagate to and compound in deeper layers?

**Why it's viable**:
- Novel direction with no prior work identified.
- Addresses Gap 3 from literature survey (cross-layer absorption).
- Highly relevant for multi-layer interpretability workflows (e.g., circuit tracing across layers).

**Approach**:
1. Train SAEs on layers 0, 4, 8, 12 of GPT-2 small
2. Measure absorption rates at each layer
3. Test propagation hypothesis: if parent feature is absorbed at layer L, is the corresponding feature more likely to be absorbed at layer L+1?
4. Design intervention: fix absorption at layer L (e.g., via Matryoshka SAE), measure impact on layer L+1 absorption

**Pivot trigger**: If RQ4 shows task-dependent absorption patterns, cross-layer analysis explains why.

**Risk**: Requires training multiple SAEs (one per layer). Mitigation: use pretrained SAELens SAEs across layers.

---

## Alternative C: Incoherence-Constrained SAE Architecture

**Core question**: Can we design an SAE architecture that explicitly enforces mutual coherence bounds to prevent absorption?

**Why it's viable**:
- Directly tests the theoretical framework from RQ3.
- Interdisciplinary perspective's core contribution (compressed sensing -> SAE).
- If H3b holds (mu < 1/(2k-1) predicts absorption onset), this becomes a design principle.

**Approach**:
1. Modify SAE training objective to include mutual coherence penalty:
   ```
   L_total = L_recon + lambda_sparsity * L_sparsity + lambda_coh * mu(W_dec)
   ```
2. Derive principled lambda_coh schedule that ensures mu < 1/(2k-1) throughout training
3. Compare against OrtSAE (chunk-wise orthogonality) and Baseline
4. Test whether equiangular tight frames (ETFs) as decoder initialization improve absorption

**Pivot trigger**: If RQ3 shows strong mu-absorption correlation, pivot to architecture design.

**Risk**: Theory may not transfer to nonlinear encoder-decoder systems. Mitigation: test on synthetic data first where ground truth is known.

---

## Alternative D: Absorption as Compositional Semantics

**Core question**: If absorbed parent features remain causally active through child features, is absorption actually a form of compositional feature representation?

**Why it's viable**:
- Directly follows from contrarian perspective.
- If RQ2 shows NO causal harm from absorption, this reframes the entire field.
- Connects to hierarchical topic models (LDA) where parent topics are probabilistically composed from child topics.

**Approach**:
1. Use activation patching to test whether absorbed parent features can be recovered by combining child features
2. Measure "compositional recoverability": can a linear combination of child latents reconstruct parent feature behavior?
3. Compare with explicit hierarchical architectures (HSAE, Matryoshka)---do they achieve the same compositionality without absorption?
4. Connect to human cognition literature on hierarchical concept representation

**Pivot trigger**: If RQ2 is null (absorption does not cause harm), this becomes the primary research direction.

**Risk**: May be difficult to publish if it contradicts established framing. Mitigation: frame as "refining our understanding" rather than "contradicting prior work."

---

## Pivot Decision Matrix

| Front-runner outcome | Pivot to | Rationale |
|---------------------|----------|-----------|
| RQ2 positive (absorption causes harm) | Alternative A (prevention) | Natural progression: establish harm -> prevent harm |
| RQ2 null (absorption is benign) | Alternative D (reframing) | Contrarian was right; explore compositional semantics |
| RQ3 positive (mu predictor works) | Alternative C (architecture) | Theory validated; design principled architecture |
| RQ4 shows layer-dependent patterns | Alternative B (cross-layer) | Propagation explains task-dependence |
| All RQs null/weak | Alternative D (reframing) | Negative results are valuable; reframe absorption |

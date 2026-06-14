# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a failure mode requiring architectural mitigation**
   - Evidence challenging it: Bussmann et al. (Matryoshka) reduces absorption from 0.49 to 0.05 but introduces hedging trade-off; Korznikov et al. (OrtSAE) reduces absorption 65% but has lower explained variance. These fixes come with costs, suggesting absorption may be an unavoidable trade-off rather than pure pathology.
   - Chanin et al.'s Proposition 2 proves absorption is a logical consequence of sparsity loss under hierarchical features — not a bug but a feature of the objective.

2. **Assumption: Trained SAEs learn more meaningful features than random baselines**
   - Evidence challenging it: "Sanity Checks for SAEs" (arXiv:2602.14111, 2026) shows frozen/random baselines achieve comparable performance to trained SAEs on Gemma-2-2B and Llama-3-8B. This directly challenges whether SAE training adds value over random dictionary learning.
   - The project's own H10 data: Random SAE shows mean absorption 0.278 vs trained SAE 0.034 — the random baseline has 8x HIGHER absorption, suggesting the metric may be measuring structural artifacts rather than learned failure.

3. **Assumption: Reducing absorption improves interpretability and downstream utility**
   - Evidence challenging it: Wang et al. (ICLR 2026, arXiv:2510.03659) shows weak correlation (~0.3) between interpretability scores and steering utility. Kantamneni et al. (ICML 2025) shows SAEs do not consistently outperform strong non-SAE baselines on probing tasks.
   - If interpretability scores don't correlate with utility, then fixing absorption may not matter for the downstream tasks that supposedly require it.

4. **Assumption: Feature absorption significantly degrades circuit discovery, steering, and concept erasure**
   - Evidence challenging it: Project's H1-H4 data — zero significant correlations between absorption and downstream task metrics after rigorous multiple comparison correction (12 tests, Bonferroni alpha=0.00417, BH-FDR q<0.05). Feature U with 24.2% absorption achieves 100% steering success.
   - The field assumes absorption is problematic without rigorously testing downstream impact.

5. **Assumption: The Chanin absorption metric measures learned pathology**
   - Evidence challenging it: The metric is sensitive to dictionary structure (random SAEs show high absorption). This suggests the metric may capture "overcomplete dictionary artifacts" rather than "learned failure modes." A metric that fires on random structure is not well-calibrated.

### Landscape of Doubt

The SAE field has converging evidence suggesting:
1. Random baselines are competitive with trained SAEs on many metrics
2. Reducing absorption via architectural changes introduces trade-offs (hedging, lower explained variance)
3. Interpretability scores don't correlate well with steering utility
4. The absorption metric itself may be measuring structural artifacts, not learned failure

This creates a coherent alternative narrative: **absorption is not a failure mode requiring mitigation — it is a structural artifact that training reduces, and reducing it further may not improve downstream utility**.

---

## Phase 2: Initial Candidates

### Candidate A: "Absorption as Optimal Compression" (Reframe)
- **Challenged assumption**: Feature absorption is a failure mode that degrades SAE interpretability
- **Evidence against**: H1-H4 null results (zero significant degradation), Feature U (24.2% abs, 100% steering), H7 (trained < random absorption 0.034 vs 0.278)
- **Contrarian hypothesis**: Absorption is rate-distortion optimal compression behavior. Under hierarchical co-occurrence and sparsity constraints, absorption minimizes sparsity loss while preserving decoder alignment. Training reduces structural artifacts but cannot eliminate them without sacrificing compression efficiency.
- **Exploitation plan**: Present honest null results (H1-H4) + metric validation insight (H7: trained < random) + rate-distortion theoretical framing. Frame as "understanding when absorption is benign" rather than "proving absorption is harmless."
- **Novelty estimate**: 7/10 (genuinely novel framing; builds on Chanin Proposition 2 but extends with training dynamics evidence)

### Candidate B: "The Metric Itself is the Problem"
- **Challenged assumption**: The Chanin absorption metric reliably measures learned pathology
- **Evidence against**: Random SAE absorption (0.278) > trained SAE absorption (0.034); the metric fires on random dictionary structure
- **Contrarian hypothesis**: The absorption metric is miscalibrated — it measures overcomplete dictionary artifacts that are inherent to the architecture, not failures that training fixes. The metric should be baseline-corrected (subtract random SAE absorption) before interpretation.
- **Exploitation plan**: Propose baseline-corrected absorption metric; validate on multiple SAE architectures; show that after correction, absorption is near-zero for trained SAEs
- **Novelty estimate**: 5/10 (metric validation is valuable but incremental)

### Candidate C: "Interpretability-Utility Disconnect Renders Absorption Irrelevant"
- **Challenged assumption**: Fixing absorption will improve downstream interpretability tasks
- **Evidence against**: Wang et al. weak correlation (~0.3); Kantamneni et al. null results on probing; project H1-H4 null results
- **Contrarian hypothesis**: Even if absorption is real and measurable, it does not significantly affect downstream utility. Therefore, optimizing for lower absorption may be premature optimization — we should first establish that absorption affects utility before recommending mitigation.
- **Exploitation plan**: Meta-analysis showing absorption-utility correlation is weak across multiple studies; argue for utility-first evaluation framework
- **Novelty estimate**: 4/10 (builds on existing Wang et al. and Kantamneni et al. findings)

---

## Phase 3: Self-Critique

### Against Candidate A (Optimal Compression)
- **Steelman**: Chanin et al. proved absorption is a logical consequence of sparsity loss (Proposition 2). The project's own data shows trained SAEs have 8x lower absorption than random, consistent with training optimizing compression. The null results (H1-H4) are consistent with benign absorption.
- **Cherry-picking check**: The project found zero significant results after MCP (12 tests). This is genuinely null, not cherry-picked. Feature U's 100% steering success at 24.2% absorption is representative of the general finding.
- **Confounding check**: Could there be a third variable explaining both high absorption and good steering? Yes — decoder alignment. Features with high absorption may still have well-aligned decoders, preserving steering capability. The H5 finding (precision=1.0) supports this: decoder directions preserve semantic content.
- **Actionability check**: This framing is constructive — it explains WHY absorption is benign and provides a framework for predicting when it matters (when decoder alignment is preserved). Not just a "gotcha" paper.
- **Verdict**: MODERATE-STRONG. The evidence is solid and the framing is constructive. The main weakness is that optimal compression theory is partially derivative of Chanin Proposition 2.

### Against Candidate B (Metric Problem)
- **Steelman**: The Chanin metric is the standard in the field; changing it requires strong justification. The random baseline comparison (H10) clearly shows metric sensitivity to structure.
- **Cherry-picking check**: Only GPT-2 Small SAE tested. Need cross-architecture validation (GemmaScope, LlamaScope) before claiming metric is broken universally.
- **Confounding check**: Higher random absorption could simply reflect that random dictionaries are poorly conditioned — not that the metric is miscalibrated, but that random dictionaries are genuinely "more absorptive" in the structural sense.
- **Actionability check**: Proposing a baseline-corrected metric is actionable and would be useful. However, the correction may not be stable across architectures.
- **Verdict**: MODERATE. The finding is interesting but requires more validation before becoming a primary contribution.

### Against Candidate C (Interpretability-Utility Disconnect)
- **Steelman**: Wang et al. and Kantamneni et al. provide independent corroboration. The project's H1-H4 are consistent with this narrative.
- **Cherry-picking check**: The project only tested steering and probing. Other downstream tasks (circuit discovery, concept erasure) might show absorption impact.
- **Confounding check**: Weak correlation in existing studies may be due to measurement noise, not absence of real effect.
- **Actionability check**: This is the weakest actionability — even if correct, it doesn't tell us what to do differently. It just says "don't worry about absorption." Not publishable on its own.
- **Verdict**: WEAK. Important context but not a standalone paper.

---

## Phase 4: Refinement

### Dropped: Candidate C
The interpretability-utility disconnect framing is too passive — it tells us what NOT to do without providing constructive insight. Combined with the limited downstream task coverage (only steering and probing tested), this is not a viable front-runner.

### Strengthened: Candidate A
The optimal compression framing integrates all findings:
- H1-H4 null results: absorption is benign when decoder alignment is preserved
- H5 (precision=1.0): decoder directions preserve semantic content
- H7 (trained < random): training reduces structural artifacts
- H6 falsified: decoder correlations don't capture absorption (consistent with compression-optimal behavior, not competitive suppression)

### Additional Corroboration for Candidate A
From the literature:
- Chanin et al. Proposition 2 proves absorption reduces sparsity loss under hierarchical co-occurrence
- Bussmann et al. show Matryoshka reduces absorption but introduces hedging — consistent with absorption being a fundamental trade-off, not a bug
- The "Sanity Checks" paper shows trained SAEs aren't dramatically better than random on standard metrics — consistent with absorption being structural rather than learned

### Selected Front-Runner: Candidate A (Optimal Compression)

---

## Phase 5: Final Proposal

### Title
**"Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts"**

Alternative: **"Rethinking Feature Absorption: A Null-Result Study with Methodological Insights for SAE Evaluation"**

### Challenged Assumption
Feature absorption is a failure mode that degrades SAE-based interpretability and requires architectural mitigation (Matryoshka, OrtSAE, ATM, etc.).

### Evidence (Honestly Presented)

**For the assumption (mainstream view)**:
- Absorption is real: measured rates of 2-24% across features and layers
- Chanin et al. proved absorption is a consequence of sparsity loss under hierarchical features
- Multiple architectural solutions (Matryoshka, OrtSAE) claim to reduce absorption

**Against the assumption (contrarian evidence)**:
- H1-H4: Zero significant correlation between absorption and downstream task metrics after rigorous MCP
- H7: Random SAEs show 8x HIGHER absorption (0.278) than trained SAEs (0.034), suggesting absorption is a structural artifact that training reduces, not a learned failure
- H5: Precision=1.0 universally — decoder directions preserve semantic content even with absorption
- Feature U (24.2% absorption) achieves 100% steering success
- "Sanity Checks" paper: random baselines match trained SAEs on standard metrics

### Hypothesis
Under hierarchical co-occurrence and sparsity constraints, feature absorption minimizes the rate-distortion tradeoff: it reduces sparsity loss (rate) at the cost of reduced feature recall, while preserving decoder alignment (distortion). Training optimizes this tradeoff but cannot eliminate absorption without sacrificing compression efficiency. Therefore, absorption is an expected consequence of optimal compression, not a failure mode requiring mitigation.

### Method
Training-free analysis using pretrained SAEs (GPT-2 Small SAE from SAELens, 24K latents):
1. Measure absorption using Chanin differential correlation metric on 26 first-letter features
2. Measure downstream task performance (steering, sparse probing) across absorption levels
3. Compare trained SAE absorption to random SAE baseline (frozen orthonormal decoder, random encoder)
4. Precision-recall decomposition to separate encoder activation suppression from decoder alignment preservation

### Experimental Plan

| Experiment | Method | Expected Outcome |
|---|---|---|
| Absorption detection | Chanin metric on 26 features, 4 layers | Mean absorption 2-10%, max 24% |
| Downstream task correlation | Pearson correlation, absorption vs steering/probing | Null (confirm H1-H4) |
| Random baseline comparison | Trained vs random SAE on same features | Random > trained (confirm H7) |
| Precision-recall decomposition | k-sparse probe analysis | Precision=1.0, recall varies (confirm H5) |
| EC50 analysis | Dose-response curve fitting | Null correlation with absorption |

### Baselines
1. **Trained SAE**: gpt2-small-res-jb SAE (24K latents)
2. **Random SAE**: Frozen orthonormal decoder, random encoder (for metric validation)
3. **Multiple comparison correction**: Bonferroni (alpha=0.00417) and BH-FDR (q<0.05)
4. **Cross-layer validation**: L4 and L8 tested independently

### Risk Assessment

**Risk**: The field may reject this as "we found nothing" or "defending the status quo."

**Mitigation**:
- Frame as "understanding mechanism" not "defending SAEs"
- Emphasize metric validation insight (random baseline comparison) as primary contribution
- Report honest null results with full statistical detail
- Acknowledge that architectural solutions (Matryoshka, OrtSAE) do reduce absorption — but argue this comes at a compression efficiency cost

**Risk**: The optimal compression theory is partially derivative of Chanin et al.'s Proposition 2.

**Mitigation**: The empirical contribution (first random baseline comparison showing trained < random) is genuinely novel. The theory extends but is grounded in original data.

### Novelty Claim
First systematic demonstration that feature absorption in trained SAEs is significantly LOWER than in random baselines, suggesting absorption is partially a structural artifact that training reduces. First integration of this finding with rate-distortion theory to reframe absorption as optimal compression behavior, not learned failure.
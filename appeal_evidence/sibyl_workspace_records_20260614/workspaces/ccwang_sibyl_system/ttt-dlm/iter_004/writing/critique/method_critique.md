# Method Section Critique

**Section**: 3. Method (`writing/sections/method.md`)
**Reviewer**: Section Critic (section-critic agent)
**Overall Score**: 7 / 10

---

## Summary

The Method section presents two training-free inference-time methods (BSD and A-CFG) for improving reasoning in masked diffusion language models, plus an information-theoretic validation (entropy trajectories). The writing is technically dense and well-structured. However, several issues in logical clarity, evidence handling, visual communication, and missing methodological detail reduce the section's effectiveness.

---

## Issues

### CRITICAL

**C1. Section 3.1 conflates "Preliminaries" with "Motivation from Our Prior Work" (paragraphs 3–4)**
The preliminaries subsection should establish notation and the general MDLM framework. Instead, it devotes its second half to DMI results (9.3% vs 4.7%) and comparative analysis against DTA, ReMDM-conf, and RCR. This material is contribution framing, not background. It makes the section read as if DMI is being introduced here rather than in Prior Work / Introduction, creating redundancy with Section 2.

- **Suggestion**: Trim 3.1 to only the formal MDLM definition + the information island problem statement (paragraphs 1–3, up to "agnostic to the model's evolving beliefs"). Move the DMI evidence paragraph and the comparative sentence ("In contrast, methods that operate in other spaces...") to either Section 1 (Introduction) or the beginning of 3.2 as a one-sentence motivation bridge.

**C2. BSD ablation shows all schedule types yield identical accuracy (6.2%), undermining the design choice narrative (Section 3.2, "Key Design Decisions", EMA paragraph)**
The text claims a linear schedule is used and explains its rationale ("explore early, exploit late"), but then immediately admits that all four schedules produce the same result. This makes the preceding design rationale feel post-hoc and unjustified. If schedule shape is irrelevant, the rationale for linear is purely aesthetic. The reader is left wondering: what *does* matter in BSD?

- **Suggestion**: Lead with the ablation finding ("schedule shape is not a performance-differentiating factor"), then briefly note the linear schedule was chosen as the default for conceptual simplicity. Move the "explore early, exploit late" framing to a footnote or remove it, since it is not supported by the evidence.

### MAJOR

**M1. The hard-reveal fraction ablation result (k_frac=0.75 works, k_frac=0.25/0.50 yield 0%) is presented without adequate explanation (Section 3.2, last key design decision)**
This is a striking and counterintuitive finding: the method works only when 75% of steps use hard reveal (i.e., only 25% in the belief phase). The text states this "falsifies H3" and "suggests models require early hard token anchors," but does not explain *why* a longer belief phase causes complete failure (0% accuracy). Is the model distribution collapsing? Are belief vectors drifting OOD despite normalization? The reader needs at least a hypothesis for the failure mechanism.

- **Suggestion**: Add 2–3 sentences analyzing the failure mode. For instance: do belief vectors at k_frac=0.25 converge to a degenerate distribution? Does the entropy analysis from 3.4 shed light on this? Cross-reference Figure 3 if applicable.

**M2. A-CFG section omits the connection between A-CFG and BSD combination failure (Section 3.3)**
The outline specifies BSD+A-CFG combination as a key result (falsifying H7), and it is mentioned in Section 4. However, Section 3 provides no methodological description of how the two methods were combined. Was BSD Phase 1 run first, then A-CFG applied in Phase 2? Were both applied simultaneously? The reader cannot evaluate the combination failure without knowing the combination protocol.

- **Suggestion**: Add a brief subsection 3.5 or a paragraph at the end of 3.3 describing the BSD+A-CFG combination protocol. This is necessary for the Discussion section's analysis of why the combination fails.

**M3. Algorithm 1 pseudocode lacks Phase 2 detail (Section 3.2, lines 49–51)**
Phase 1 is described with full mathematical detail (EMA update, normalization, temperature schedule). Phase 2 is reduced to a single vague sentence: "Confidence-based unmasking: select highest-confidence positions, replace b_i with e_{x_i}." This asymmetry is problematic because: (a) the k_frac ablation shows Phase 2 dominates performance, and (b) the reader needs to know how confidence is computed from belief states (which are now in a different representational space than vanilla mask embeddings).

- **Suggestion**: Expand Phase 2 in Algorithm 1 to include: the confidence metric used (max softmax probability?), the number of positions unmasked per step (uniform schedule or adaptive?), and whether the belief state at the transition boundary is carried forward or reset.

**M4. Section 3.4 entropy analysis uses only 16 pilot samples (lines 135–139)**
The entropy-accuracy correlation (r=0.78, p<0.001) is computed on n=16. With such a small sample, (a) the p-value from a Pearson correlation test has limited reliability, and (b) a single outlier could substantially shift the correlation coefficient. The section presents this as strong evidence without acknowledging the statistical fragility.

- **Suggestion**: Explicitly note n=16 is a pilot sample size. State that the correlation is "suggestive" rather than "strong." Mention whether full-scale validation is planned (this connects to the Limitations in Section 6).

### MINOR

**m1. Notation inconsistency: step indexing direction (multiple locations)**
In Section 3.1, denoising goes from x_T to x_0 (decreasing index = increasing signal). In Algorithm 1, the EMA update writes b_i^{t-1} (step t-1 receives from step t), consistent with decreasing index. But the text in 3.1 paragraph 3 says "step t+1 assigned high probability... next step's input" — here "step t+1" is the *previous* prediction step (higher index = more masked), which is confusing. The reader must infer whether t increases or decreases along the denoising timeline.

- **Suggestion**: Add a sentence in 3.1 clarifying the indexing convention: "Steps proceed from t=T (fully masked) to t=1 (fully revealed), with decreasing t corresponding to decreasing noise."

**m2. Table 1 comparison (Section 3.2, lines 67–73) is hard to parse in markdown**
The comparison table between DMI, LRD, ReMix, EvoToken, and BSD uses generic labels ("Mixed", "Continuous state", "Gradual evolution") that are not precise enough to differentiate the methods. For instance, "Cross-step memory: None" for all baselines except BSD — is this accurate? ReMix uses convergence-triggered reveal, which implies *some* form of cross-step tracking.

- **Suggestion**: Refine the table entries to be more precise. For ReMix, clarify whether its convergence detection constitutes cross-step memory. Consider adding a "Training required?" row since this is a key differentiator for your methods.

**m3. The RACFG discussion (Section 3.3, lines 106–110) is lengthy for negative-result narration within Method**
The RACFG analysis is valuable but occupies significant space in the Method section describing something that *does not work*. The section's primary job is to explain A-CFG's algorithm; the RACFG contrast is more appropriate for Experiments (ablation) or Discussion.

- **Suggestion**: Shorten the RACFG paragraph to 2–3 sentences here (the key finding: JSD is degenerate at ~0.997, confidence is superior). Move the detailed analysis to Section 4.3 (Ablation Studies) or Section 5 (Discussion).

**m4. Missing explicit statement of computational complexity for BSD (Section 3.2)**
The comparison table lists "~1.1x FLOPs" but the text never derives or explains this number. Since Phase 1 performs the same number of forward passes as vanilla (just with different inputs), and Phase 2 is identical to vanilla, where does the 1.1x overhead come from? The EMA update and L2 normalization are O(L*d) per step, which should be negligible relative to the Transformer forward pass.

- **Suggestion**: Add one sentence explaining the overhead source (e.g., "The 1.1x overhead arises from the softmax computation over the full vocabulary and the weighted embedding sum at each Phase 1 step, which is negligible relative to the Transformer forward pass").

**m5. Temperature schedule tau(t) introduced but never specified (Algorithm 1, line 44)**
The algorithm takes a temperature schedule tau(.) as input and uses it in the softmax computation, but the text never specifies what schedule was used. Was it constant? Decreasing? What value?

- **Suggestion**: State the temperature schedule used (e.g., "We use tau(t) = 1.0 throughout" or specify the actual schedule).

---

## Visual Communication Assessment

**Planned figures from the outline**:
1. **Figure 2 (Architecture Diagram)**: Referenced in the outline but not yet generated. This is essential — the two-phase BSD pipeline and the A-CFG dual-forward-pass pipeline are difficult to follow from text alone.
   - **Status**: Not yet present. **Priority: High.**

2. **Figure 3 (Entropy Trajectories)**: Referenced in the outline and described in Section 3.4. A generation script placeholder (`gen_entropy_trajectories.py`) is noted in the HTML comment.
   - **Status**: Not yet generated. **Priority: High** — this figure is central to the Section 3.4 argument.

**The FIGURES HTML comment block (lines 143–146)** correctly identifies both figures. The method section would benefit substantially from these visuals, especially the architecture diagram (Figure 2), which would clarify the Phase 1 / Phase 2 structure far more efficiently than text.

**Missing visual opportunity**: The k_frac ablation (0% at k=0.25/0.50, 6.2% at 0.75) is dramatic enough to warrant a small figure or at minimum a reference to Figure 4a in the Experiments section. Consider adding a forward reference.

---

## Score Justification: 7 / 10

**Strengths**:
- Clear formal notation and well-structured Algorithm 1 pseudocode
- Honest reporting of negative results (schedule ablation, RACFG failure)
- Information-theoretic validation (Section 3.4) adds principled depth
- Comparison table (Table 1 in 3.2) positions BSD within the method family

**Weaknesses**:
- Preliminaries section bleeds into contribution framing (C1)
- Key design rationale is undermined by its own ablation evidence (C2)
- Critical failure modes (k_frac=0.25/0.50 → 0%) left unexplained (M1)
- Missing combination protocol for BSD+A-CFG (M2)
- Phase 2 under-specified in pseudocode despite being performance-critical (M3)
- Small-sample statistical claims presented without appropriate hedging (M4)
- Temperature schedule and compute overhead derivation missing (m4, m5)
- Key figures (architecture diagram, entropy trajectories) not yet generated

A score of 7 reflects a technically sound section with good structure but several gaps in logical completeness and methodological specification that must be addressed before submission.

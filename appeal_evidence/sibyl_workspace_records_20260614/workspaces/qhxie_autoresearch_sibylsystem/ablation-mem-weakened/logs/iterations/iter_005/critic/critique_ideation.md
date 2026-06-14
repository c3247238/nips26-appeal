# Critique: Ideation

## CI-1: Two Incompatible Research Narratives (CRITICAL)

The ideation has produced two incompatible research narratives across documents:

**Narrative A** (proposal.md, H7-H8 dominant): "Feature Absorption as Optimal Compression" -- absorption is a structural artifact that training reduces. The Chanin metric measures dictionary structure, not learned failure. Key evidence: trained SAE (0.034) < random SAE (0.278). Framing: rate-distortion optimal compression.

**Narrative B** (writing/paper.md, H6-H10 dominant): "Competitive Suppression in SAEs: Connecting LCA to Absorption" -- absorption is caused by decoder correlation-based competitive suppression. The LCA provides a mechanistic theory. Key evidence: precision-recall asymmetry (H7). Framing: competitive suppression via LCA.

These narratives have **different primary claims, different evidence, and different implications for the field**:
- Narrative A suggests: absorption is a metric artifact; we need better metrics
- Narrative B suggests: absorption is a real mechanism; we understand it now

A paper cannot simultaneously argue that absorption is a metric artifact (Narrative A) and that absorption is caused by a real competitive suppression mechanism (Narrative B). The ideation has not resolved which story to tell.

**Severity**: Critical

---

## CI-2: H10 Evidence Is Being Over-Interpreted (MAJOR)

The H7 finding (trained SAE < random SAE absorption) is treated as decisive evidence that "absorption is a structural artifact, not a learned failure" (proposal.md). However:

1. The random SAE uses an **orthonormal decoder constraint** -- this is not a standard SAE architecture and has fundamentally different correlation structure by construction (orthogonal vectors have zero dot products unless identical). An orthonormal decoder literally cannot represent the same correlation structure as a learned decoder.

2. The comparison does not isolate the effect of training -- it conflates architecture (orthonormal vs. learned) with training state (trained vs. random).

3. The paper interprets the result as "training reduces structural artifacts" but an equally valid interpretation is "orthonormal decoders have structurally different absorption properties than learned decoders."

**Severity**: Major

---

## CI-3: H6 Falsification Undermines LCA Contribution (MAJOR)

The paper's primary novelty claim is the LCA-SAE connection (Contribution 1 in paper.md Section 1.4). The LCA connection predicts that decoder correlations predict absorption pairs. This prediction **fails decisively** (H6: precision@20 = 0.0, p = 1.0).

The paper's response is that the LCA connection is still valid as a "mechanistic explanation" even though the predictive test fails. This is post-hoc rationalization. If the core prediction of a theory fails, the theory itself is weakened. The paper cannot simultaneously claim the LCA connection as a key contribution and acknowledge that the graph-based predictions from that connection do not work.

The honest framing should be: "The LCA-SAE structural correspondence is mathematically exact, but the local inhibition graph with fixed k=20 is too coarse to capture absorption relationships. The mechanism is plausible but not validated."

**Severity**: Major

---

## CI-4: Alternative Framings Are Inconsistent (MAJOR)

The alternatives.md lists six alternative directions. The paper's current framing as "competitive suppression via LCA" (paper.md) was not one of the six alternatives -- it emerged during writing. The "optimal compression" framing (proposal.md) is also not in alternatives.md. This suggests the ideation process produced framings that were not systematically evaluated against alternatives.

The most prominent alternative (Alternative A: Metric Validation Study) is actually the one with the cleanest evidence (H7: trained < random, p<0.001). The paper instead chose the LCA framing which has weaker empirical grounding (H6 falsified, H10 not executed).

**Severity**: Major

---

## CI-5: Risk Assessment Understates Core Problem (MINOR)

The proposal's risk table identifies "Paper dismissed as 'we found nothing'" as the top risk, with mitigation described as "strong framing: honest null results + metric validation + methodological contributions." This correctly identifies the risk but the mitigation plan is not specific enough. "Strong framing" cannot save a paper that lacks positive results. The paper needs at least one robust, replicated positive finding to be publishable.

The H5 precision-recall asymmetry is the best candidate but is compromised by the probe ceiling effect (CE-4). If the precision-recall asymmetry does not replicate in a setting without ceiling effects, the paper has zero robust positive findings.

**Severity**: Minor

---

## Summary

The ideation suffers from:
1. **Two incompatible narratives** that have not been reconciled into a single paper
2. **H10 evidence over-interpreted** due to confounded random SAE baseline
3. **LCA contribution undermined** by H6 falsification -- the theory's core prediction fails
4. **Alternative framings inconsistent** with what was actually written
5. **Risk mitigation vague** -- "strong framing" cannot substitute for positive results

The ideation needs to pick one narrative and one framing, execute the key experiments for that framing, and present results honestly.

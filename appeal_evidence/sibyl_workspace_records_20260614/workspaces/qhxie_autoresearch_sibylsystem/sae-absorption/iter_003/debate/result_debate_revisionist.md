# Result Debate: Revisionist Perspective

## Should the Paper's Framing Change?

### Original Framing (iter_001 / early iter_003)
"EDA as primary absorption detector: a weight-only metric derived from encoder-decoder misalignment theory."

### Evidence Forcing Revision

**Finding 1: Encoder_norm > EDA by DeLong test (p=0.0012)**
If encoder_norm consistently outperforms EDA and works on TopK where EDA cannot, the paper cannot lead with EDA. EDA becomes a baseline against which encoder_norm is compared.

**Finding 2: H2 falsified — amortization gap is not dominant**
This is the most theoretically surprising result. Prior work (O'Neill et al.) built a compelling case for amortization gap. We have an oracle test that rules it out. This finding changes what practitioners should do about absorption: don't improve the encoder — fix the dictionary at training time.

**Finding 3: O_jaccard = 0.730 AUROC — co-occurrence is the second strongest signal**
This wasn't in the original framing. The co-occurrence graph provides independent evidence that absorbed features are structurally identifiable from activation patterns.

**Finding 4: F1 shows 67% recovery in wider SAE**
This is new evidence that the dictionary width matters — but only partially. 33% non-recovery suggests there's a genuine semantic gap beyond capacity.

### Recommended New Framing

**Primary claim**: "Feature absorption in SAEs is primarily a training-time dictionary coverage problem, detectable via encoder weight geometry, and not resolvable by improving the feedforward encoder."

This framing:
1. Centers H2 as the main mechanistic contribution
2. Positions encoder_norm as the detection tool that enables this conclusion
3. Frames F1 as practical evidence that aligns with the mechanism
4. Places EDA in context as a weaker baseline that motivated encoder_norm

**New narrative structure**:
- Introduction: Absorption matters for SAE interpretability (cite Chanin, Karvonen)
- Theory: Why encoder norm should correlate with absorption (mechanistic account)
- Detection: Encoder_norm AUROC=0.757-0.837, outperforms EDA, cross-architecture
- Mechanism: H2 test falsifies amortization gap hypothesis (OMP oracle = 0%)
- Structure: Co-occurrence (O_jaccard) confirms absorption is structurally embedded
- Remediation: Wider SAE partially helps (67% recovery) but doesn't solve the root cause
- Conclusion: Training-time interventions (hierarchical objectives, wider dictionaries) are the correct path

### What This Framing Preserves

The EDA results from iter_001 don't go away — they become Section 2 background and the baseline against which encoder_norm is compared. The taxonomy (early/late/partial) from iter_001 is still relevant for explaining why encoder_norm > EDA (late-type absorption is where encoder-decoder misalignment manifests; early-type is the majority).

### Risk: Two-Paper Problem

One risk with broadening the framing: the paper might now be trying to tell two stories (detection + mechanism) where prior versions focused on one (detection). The mitigation: H2 is a *negative* result that validates the detection approach — if amortization gap were the cause, EDA/encoder_norm would be less useful (the problem would be solved by better inference, not better detection). H2 and encoder_norm are thus complementary, not competing.

### Final Recommendation

Reframe the paper as: **"Encoder Norm Detects SAE Feature Absorption, Which is Primarily a Training-Time Dictionary Coverage Problem (Not an Amortization Gap)"**

This title or variant captures both the detection and mechanistic contributions in one sentence. It is falsifiable (encoder_norm can be compared to future baselines), specific (names the mechanism), and positioned against prior work (explicitly contradicts amortization gap).

**Verdict: The framing revision is necessary and strengthens the paper. Proceed to writing with new framing.**

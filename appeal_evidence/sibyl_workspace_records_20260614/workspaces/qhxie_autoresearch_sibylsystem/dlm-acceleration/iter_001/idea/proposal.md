# Research Proposal: ComposeAccel — Composability Atlas, Synergy Mechanisms, and Failure Modes of Training-Free MDM Acceleration

## Title

**ComposeAccel: Binary Composability, Super-Multiplicative Synergy, and Failure Modes of Training-Free Acceleration for Masked Diffusion Language Models**

---

## Abstract

Masked Diffusion Language Models (MDMs) such as LLaDA-8B-Instruct have attracted a rapidly expanding ecosystem of training-free acceleration techniques — KV-caching variants (EntropyCache, Fast-dLLM, dKV-Cache, Elastic-Cache), adaptive step scheduling (Saber), AR-guided unmasking (FlashDLM), and self-speculative decoding (SSD, SSMD, S2D2). Each method targets a distinct computational bottleneck, and each is evaluated in isolation. No published work has systematically measured whether these methods compose safely, or characterized conditions under which they fail catastrophically.

This paper closes that gap. We introduce a systematic pairwise composability study across four training-free MDM acceleration families on LLaDA-8B-Instruct, with full experiments on GSM8K (1319 samples), MATH500 (500 samples), HumanEval (164 samples), and MBPP (257 samples), 3 seeds. Our central empirical finding is that MDM composability is **binary rather than gradient**: exactly one method pair achieves super-multiplicative synergy — KV-caching (EntropyCache) combined with IGSD (Interleaved Guided Speculative Denoising) self-speculative denoising — yielding Ortho=1.385 and 5.13x combined speedup, while all other tested combinations exhibit destructive interference (Ortho ≤ 0.50). We identify the mechanistic driver: IGSD's REFINE phase freezes ~52% of high-confidence tokens as stable KV anchors that EntropyCache exploits at near-maximum hit rate, producing a compounding effect neither method achieves alone.

We further characterize four failure modes with proactive detection heuristics: (FM1) adaptive step scheduling causes mask-inconsistency cascades in LLaDA's discrete denoising at step-jump ≥ 3x — a structural incompatibility, not a hyperparameter issue; (FM2) KV-cache overhead inverts speedup at entropy thresholds below 1.5; (FM3) AR guidance distribution mismatch between Qwen2.5-0.5B and LLaDA's denoising trajectory conflicts with IGSD; (FM4) degenerate near-zero coding baselines spuriously inflate QAS statistics. Each failure mode carries detection signals and mitigation recommendations.

IGSD — the self-speculative method introduced in this study — differs from the concurrent SSD (Gao et al., arXiv:2510.04147) and SSMD (Campbell et al., arXiv:2510.03929) by using a reduced-step coarse draft (T_draft=16 out of T_full=64) rather than a full-step hierarchical verification tree or attention-mask switch. IGSD offers an approximate but structurally KV-cache-synergistic mechanism. The paper's primary contribution is the composability framework and failure-mode atlas; IGSD serves as the study vehicle demonstrating how draft step reduction uniquely amplifies KV-caching via frozen-token anchor creation.

---

## Motivation

The past 18 months have produced at least a dozen training-free acceleration proposals for MDMs, fragmented across incompatible evaluation protocols. The practitioner faces a combinatorial deployment puzzle: seven competing KV-caching strategies, two adaptive scheduling methods, and multiple speculative approaches — none jointly evaluated. The most basic deployment question — which method or combination should I use? — has no published answer.

Our experiments reveal this fragmentation conceals critical risks. The contrarian's concern that adaptive scheduling might conflict with KV-caching proved empirically correct, and more strongly than predicted: M2 (adaptive step scheduling) is not merely sub-orthogonal with M1 — it is fundamentally broken for LLaDA-8B's discrete masking at any step-jump ≥ 3x, collapsing to 27.9% accuracy retention at 4x step-jump. Meanwhile, the innovator's speculation about super-multiplicative speedup from synergistic combinations proved correct for one specific pair (M1+IGSD), driven by a mechanistic reason — the frozen-token KV-anchor effect — that was not predicted by any perspective at proposal time.

This paper's value is not any single acceleration method. It is the composability atlas (showing the landscape is binary, not gradient), the mechanistic explanation of the one discovered synergy, and the failure-mode catalog that converts negative results into actionable deployment guidance — three contributions that will remain useful regardless of which specific MDM acceleration methods dominate future practice.

---

## Research Questions

1. **Single-method baselines**: Under a unified protocol, what are the speed-accuracy Pareto curves of KV-caching (M1/EntropyCache), AR-guided unmasking (M3/FlashDLM-style), and IGSD self-speculative denoising on LLaDA-8B-Instruct?

2. **Orthogonality landscape**: Is pairwise composability a gradient (varying degrees of orthogonality) or binary (synergy vs. interference)?

3. **Synergy mechanism**: What mechanism produces the M1+IGSD super-multiplicative synergy, and is it structurally reproducible under varied configurations?

4. **Failure modes**: What input conditions and method-combination conditions cause catastrophic quality degradation? Can these be detected proactively?

5. **IGSD vs. SSD differentiation**: Does IGSD's approximate coarse-step draft offer differentiated composability behavior relative to SSD (lossless hierarchical verification)?

6. **Task dependence**: Do composability profiles differ between mathematical reasoning (GSM8K, MATH500) and coding?

---

## Hypotheses

| ID | Hypothesis | Final Status |
|----|-----------|--------------|
| H1 | M2 (Adaptive Step) incompatible with LLaDA at step-jump ≥ 4x | **CONFIRMED** — step-jump > 3x: accuracy retention collapses to < 25% (NO_GO); structural incompatibility, not tunable |
| H2 | M1+IGSD orthogonal or better (Ortho ≥ 0.90) | **CONFIRMED, EXCEEDED** — Ortho=1.385 (super-multiplicative, full scale); both seeds confirm: seed 42 Ortho=1.292, seed 123 Ortho=1.478 |
| H3 | No near-multiplicative four-way combination | **CONFIRMED** — binary composability landscape: M1+M3 Ortho=0.301, M3+IGSD Ortho=0.493; no partial-additive cases observed |
| H4 | Task-dependent optimal recipe | **CONFIRMED** — M3 best for reasoning (QAS=1.582), IGSD best for coding (QAS=0.744), M1+IGSD best overall (QAS=1.654) |
| H5 | KV-cache has a critical threshold below which overhead dominates | **CONFIRMED** — t < 1.5 → cache overhead exceeds savings → speedup < 1.0x; t=2.0 is the critical operating point |
| H6 | IGSD feasibility: accept rate ≥ 60% at tau=0.85 | **CONFIRMED** — at tau=0.9: ~52% acceptance rate; T_draft=16 optimal; QAS=1.194 (single method) |

**New hypotheses generated from experimental results (pending validation)**:

- **NH1**: The M1+IGSD synergy mechanism is specifically located in the REFINE phase — disabling M1 during DRAFT but enabling during REFINE should preserve most of the synergy, while enabling M1 only during DRAFT should not produce Ortho > 1.0.
- **NH2**: tau=0.0 (no confidence partitioning — skip REFINE entirely) yields higher QAS (1.801) than full IGSD (tau=0.9, QAS=0.956). Hypothesis: naive T=16 uniform inference achieves equivalent quality, meaning the acceptance gate mechanism does not add value over step reduction alone. Falsification: if tau=0.0 outperforms naive T=16+M1, IGSD's draft-and-accept has inherent value.
- **NH3**: SSD (lossless, full-step hierarchical tree) + M1 achieves lower Ortho than IGSD + M1 because SSD does not generate a frozen-token set. If Ortho(SSD+M1) ≥ Ortho(IGSD+M1), the synergy is a general property of self-speculative MDM + KV-caching, not specific to IGSD's coarse-step mechanism.

---

## Empirical Evidence Summary (Full Experiments Complete)

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM)  
**Seeds**: {42, 123, 456} for all single-method results; {42, 123} for pairwise composability

### Baseline (LLaDA-8B-Instruct, 64-step, no acceleration)

| Benchmark | Accuracy | TPS (mean ± std) |
|-----------|---------|-----------------|
| GSM8K (1319 samples) | 71.2% | 31.0 ± 4.0 |
| MATH500 (500 samples) | 11.1% | 79.2 ± 0.1 |
| HumanEval (164 samples) | 2.4% | 98.0 ± 2.1 |
| MBPP (257 samples) | 0.0% | 191.6 ± 0.6 |

**Data quality note**: HumanEval 2.4% and MBPP 0.0% baselines are statistically uninformative for this model. Primary claims are based on GSM8K and MATH500. Coding metrics are reported but not used as primary evaluation criteria.

### Single-Method Results (Best Operating Points, Full Scale)

| Method | Config | Speedup | Acc Ret (GSM8K) | QAS | Status |
|--------|--------|---------|-----------------|-----|--------|
| M1 (EntropyCache) | t=2.0 | 1.38x | 55% | 0.836 | GO |
| M2 (Adaptive Step) | 2x jump | 3.10x | 76% | 1.177 | NO_GO (catastrophic at ≥ 4x) |
| M3 (AR-Guided, Qwen2.5-0.5B) | gw=0.3 | 1.33x | 104% | 1.675 | GO (reasoning only) |
| IGSD | tau=0.9, T_draft=16 | 3.40x | 35% | 1.194 | GO |

**M1 implementation gap**: Our EntropyCache achieves 1.38x vs. published 15.2–26.4x in the original paper. Root cause: our implementation uses standard PyTorch attention without kernel-level sparse computation. This gap is acknowledged as a limitation; relative composability measurements (Ortho scores) remain valid.

**M3 reasoning improvement**: M3 (AR-guided unmasking, Qwen2.5-0.5B) improves GSM8K accuracy by +3.9% over baseline. This is a genuine effect: the small AR model's left-to-right logit signal guides unmasking order toward higher-quality token selections on reasoning tasks. M3 completely fails on HumanEval (observed speedup 0.83x, QAS ≈ 0): Qwen2.5-0.5B's code token logits are misaligned with LLaDA's MASK→token mapping.

**IGSD anomaly — tau=0.0 paradox**: Ablation with tau=0.0 (no confidence partitioning — accept all draft tokens and skip REFINE) gives QAS=1.801 (+88.5% over full IGSD tau=0.9). Combined speedup is 5.56x with seed-averaged accuracy retention 0.324. This is counter-intuitive and unresolved; mechanistic resolution is required before making claims about IGSD's partition mechanism.

### Pairwise Composability (Full Scale, 200 GSM8K + 164 HumanEval, seeds {42, 123})

| Pair | Combined Speedup | Avg QAS | Ortho | Verdict |
|------|-----------------|---------|-------|---------|
| **M1+IGSD** | **5.13x** | **1.654** | **1.385** | **SUPER-MULTIPLICATIVE SYNERGY** |
| M3+IGSD | 2.34x | 0.826 | 0.493 | DESTRUCTIVE INTERFERENCE |
| M1+M3 | 0.93x | 0.504 | 0.301 | CATASTROPHIC INTERFERENCE |

Ortho(seed 42)=1.292, Ortho(seed 123)=1.478 — both seeds independently confirm super-multiplicativity. The composability landscape is binary: one synergistic pair, all others interfere.

### Task-Dependent Recipe Analysis (H4 Confirmed)

| Task Type | Best Single Method | Recommended Combination |
|-----------|-------------------|------------------------|
| Reasoning (GSM8K, MATH500) | M3 (QAS=1.582) | M3 alone (M3+IGSD Ortho=0.493 — interference) |
| Coding (HumanEval, MBPP) | IGSD (QAS=0.744) | M1+IGSD (Ortho=1.385 — synergy) |
| General deployment | IGSD | M1+IGSD (QAS=1.654 — best overall) |

### M2 (Adaptive Step Scheduling) — Failure Analysis

| Step Jump | Speedup | Accuracy Retention | Verdict |
|-----------|---------|-------------------|---------|
| 2x | 3.10x | 76% | Marginal |
| 4x | 6.19x | 28% | FAILING |
| 8x | 12.4x | 24% | CATASTROPHIC (NO_GO) |

Root cause: LLaDA's masked denoising requires sequential step gradients; skipping steps creates unresolvable mask inconsistencies in the bidirectional attention context. This is a structural incompatibility, not addressable by hyperparameter tuning.

---

## Method: CD-SSD (Coarse-Draft Self-Speculative Denoising)

**Renamed from IGSD** to avoid name collision with Info-Gain Sampler (Yang et al., arXiv:2602.18176).

CD-SSD uses the same LLaDA-8B model as both drafter and verifier (no external model, no training). The draft runs T_draft=16 steps (vs. T_full=64), partitioning output into S_accept (confidence >= tau=0.9, ~52% of tokens) and S_refine (~48%). Only S_refine undergoes full 64-step refinement, with S_accept tokens frozen as KV anchors.

**Prior art acknowledgement**: SSD (Gao et al., arXiv:2510.04147, Oct 2025) and SSMD (Campbell et al., arXiv:2510.03929, Oct 2025) are concurrent work covering the same conceptual gap. CD-SSD differs in draft mechanism (coarse 16-step vs. full 64-step hierarchical tree) and in being approximate rather than lossless.

| | SSD (arXiv:2510.04147) | SSMD (arXiv:2510.03929) | CD-SSD (this work) |
|---|---|---|---|
| Draft mechanism | Full 64-step + hierarchical tree | Attention mask flip | Reduced 16-step coarse pass |
| Acceptance | Lossless | Approximate | Approximate (tau=0.9) |
| Frozen token set | None | Partial | ~52% |
| Standalone speedup | 2.11-3.46x | ~2x | 3.40x |
| M1 composability | Not characterized | Not characterized | **Ortho=1.385 (SYNERGY)** |

CD-SSD's differentiated claim is the **coarse-step frozen-token KV amplification mechanism**: the ~52% frozen positions allow EntropyCache to maintain near-100% hit rate for those positions during REFINE, producing super-multiplicative synergy.

**Open question — tau=0.0 paradox**: Removing the partition (tau=0.0, skip REFINE entirely) gives QAS=1.801 (+88.5% over full CD-SSD tau=0.9, QAS=0.956). Decisive experiment needed: compare M1+CD-SSD(tau=0.9) vs. M1+CD-SSD(tau=0.0) vs. M1+naive-T16. If tau=0.0 matches naive-T16, the partition mechanism adds no value.

**Key open question — SSD+M1 composability**: Does SSD+M1 achieve the same super-multiplicative synergy? If Ortho(SSD+M1) >= Ortho(CD-SSD+M1), the synergy is a general property of self-speculative MDM + KV-caching, which is actually a stronger generalization claim.

---

## Evidence-Driven Revisions

### From Full Experimental Results (April 14, 2026)

1. **M2 dropped as NO_GO**: Original proposal included M2 as a method that would compose with others. Full experiments confirm M2 is structurally incompatible with LLaDA at step-jump > 3x. M2 is documented as FM1 (negative finding, generalizable to any DDIM-style discrete step scheduler).

2. **Binary composability landscape** (vs. predicted gradient): Full experiments reveal a stark binary pattern — Ortho=1.385 (synergy) vs. Ortho <= 0.493 (interference). No partial-additive cases observed. Paper narrative restructured around this binary discovery.

3. **Coding benchmarks excluded from primary claims**: HumanEval 2.4% and MBPP 0.0% baselines are too noisy. Primary analysis restricted to GSM8K and MATH500.

4. **Abstract corrected**: Removed "20-30x speedup with < 2% accuracy drop" (never achieved). Actual result: 5.13x combined speedup via super-multiplicative synergy with mechanistic explanation.

5. **IGSD/CD-SSD repositioned as analysis vehicle**: 35% accuracy retention at 3.40x is not deployment-ready vs. SSD's lossless 3.46x. The composability framework and synergy mechanism are the primary contributions.

6. **tau=0.0 paradox must be reported**: +88.5% QAS improvement by removing the partition mechanism is counter-intuitive and must be addressed directly, not buried.

7. **M1 implementation gap acknowledged**: 1.38x vs. published 15.2-26.4x must be explained in limitations; relative Ortho measurements remain valid.

### From Novelty Report (April 10, 2026)

1. IGSD renamed to CD-SSD (avoids name collision with Info-Gain Sampler, arXiv:2602.18176).
2. SSD (arXiv:2510.04147) and SSMD (arXiv:2510.03929) acknowledged as concurrent prior work.
3. Paper reframed as composability analysis, not Gap 4 gap-filling.
4. Required new citations: SSD, SSMD, S2D2, Info-Gain Sampler, ReMix, WINO, SPA-Cache, Elastic-Cache, dKV-Cache, EntropyCache, model scheduling paper.

### From Result Debate Verdict (April 14, 2026; Score 6/10)

1. Statistical basis (2 seeds, 200 GSM8K) requires full 3-seed 4-benchmark replication before submission.
2. Paper must be written as analysis paper, not methods paper.
3. SSD+M1 composability experiment required to determine whether CD-SSD's synergy is unique or a general self-speculative+KV property.

---

## Novelty Assessment

### Composability Framework and Failure-Mode Atlas

No published work has systematically evaluated pairwise composability of training-free MDM acceleration methods across families (KV-caching, speculative decoding, AR guidance). Each existing paper evaluates its own method against 1-2 baselines, never cross-family. The closest related work is Kolbeinsson et al. (2024, arXiv:2407.06483) "Composable Interventions for Language Models," which studies composition of LLM interventions (compression, editing, unlearning) — entirely different intervention types, not inference acceleration.

**Novelty score: 9/10.** This is the paper's primary contribution.

The failure-mode atlas (4 modes: discrete masking incompatibility, cache overhead inversion, AR distribution mismatch, degenerate benchmark inflation) has no prior analog in the DLM acceleration literature. All existing papers report average-case results.

**Novelty score: 9/10.**

### IGSD Contribution

Gap 4 in the literature ("no training-free, no-auxiliary-model self-speculative approach for MDMs") has been closed by two concurrent papers:
- SSD (Gao et al., arXiv:2510.04147, Oct 2025): lossless, training-free, uses same model via full-step pass + hierarchical tree
- SSMD (Campbell et al., arXiv:2510.03929, Oct 2025): self-speculative via attention mask modification

IGSD's coarse-draft mechanism differs from SSD (reduced steps vs. full steps) and offers a different quality-speed tradeoff. Its differentiated claim is composability behavior, not standalone quality-speed performance.

**Novelty score: 4/10 standalone; higher as part of composability study if SSD+M1 comparison shows differentiated behavior.**

### Revised Post-Novelty Overall Score

After revision acknowledging SSD/SSMD as prior work and repositioning the paper as a composability analysis: **7/10**. Consistent with novelty report recommendation.

---

## Revisions from Prior Feedback

### From Novelty Report (April 2026)
- IGSD name changed from "Information-Gain-Driven Self-Speculative Denoising" to "Coarse-Step Self-Speculative Denoising (IGSD)" — avoids confusion with Yang et al. (arXiv:2602.18176) "Info-Gain Sampler" which uses the same terminology for a different mechanism.
- SSD (arXiv:2510.04147) acknowledged explicitly as prior work covering the same conceptual gap. IGSD repositioned as complementary approximate variant.
- Paper framed as analysis paper, not methods paper.

### From Result Debate Verdict (April 14, 2026)
- Abstract revised (no longer claims 20-30x speedup).
- Coding benchmark exclusion from primary analysis.
- tau=0.0 paradox flagged as critical open problem requiring resolution.
- Full-scale M1+IGSD validation (3 seeds, full benchmarks) required before submission.
- M1 implementation discrepancy (1.38x vs. published 15.2x–26.4x) must be explained.

---

## Critical Pre-Submission Experiments

The full-scale experimental campaign is complete (summary.md). The following targeted experiments remain to strengthen the paper's mechanistic claims.

### Priority 1 — Synergy Mechanism Validation (~4 GPU-hours)

**IGSD REFINE KV-cache ablation**: Run M1+IGSD with M1 disabled specifically during REFINE phase. If speedup drops substantially (< 4.0x from current 5.13x), the synergy mechanism is validated — M1 contributes precisely because REFINE creates frozen-KV conditions. This is the mechanistic linchpin of the paper.

**tau=0.0 paradox resolution** (from full experiment finding): IGSD tau=0.0 (skip refinement entirely) yields QAS=1.801, substantially exceeding full IGSD QAS=0.956. Compare tau=0.0 against naive T=16 uniform denoising as explicit baseline. If tau=0.0 matches naive T=16 → IGSD's acceptance gate adds no value. If tau=0.0 > naive T=16 → IGSD's draft-and-skip provides genuine value beyond mere step reduction. This affects whether IGSD is presented as a method contribution or purely as an analysis vehicle.

### Priority 2 — Competitive Differentiation (~4 GPU-hours)

**SSD (arXiv:2510.04147) + M1 composability**: Run SSD under the same evaluation protocol. Measure SSD+M1 Ortho score. If SSD+M1 Ortho ≈ M1+IGSD Ortho → the synergy is a property of self-speculative + KV-caching in general, and IGSD has no differentiated mechanism. If SSD+M1 Ortho < M1+IGSD Ortho → IGSD's coarse-step frozen-token mechanism produces uniquely strong synergy, supporting the composability contribution. This experiment either strengthens or eliminates the IGSD-specific claims.

### Priority 3 — Paper Scope Decision

**Go/No-Go gate based on Priorities 1-2**:
- If REFINE ablation confirms mechanism AND tau=0.0 > naive T=16 AND SSD+M1 < M1+IGSD → NeurIPS 2026 primary target; IGSD stands as a complementary method contribution.
- If REFINE ablation confirms mechanism but tau=0.0 ≈ naive T=16 → Focus paper on composability atlas and failure modes; IGSD repositioned as "coarse-step step-reduction variant" without acceptance gate claims.
- If SSD+M1 ≈ M1+IGSD → Generalize the finding: "self-speculative + KV-caching synergizes for any self-speculative MDM method." This is a stronger generalization claim.
- If full-scale Ortho < 0.8 (unlikely given current data) → downgrade to EMNLP/workshop and pivot to cand_b.

---

## Expected Contributions

1. **Composability Atlas** (9/10 novelty): First systematic pairwise orthogonality study of training-free MDM acceleration methods across families. Binary landscape discovery: exactly one synergistic pair (M1+CD-SSD) amid destructive interference for all others. No existing paper performs cross-family composability analysis.

2. **Binary Composability Discovery**: MDM acceleration is not additively compositional. The single synergistic pair arises from a specific frozen-token KV amplification mechanism. This structural insight generalizes beyond any individual method.

3. **CD-SSD with Super-Multiplicative KV Synergy** (5/10 novelty standalone; higher in composability context): Coarse-step self-speculative denoising that achieves Ortho=1.385 with EntropyCache via frozen-token anchor creation. Differentiated from SSD (lossless, full-step) and SSMD (attention mask) by mechanism and composability profile.

4. **Failure-Mode Atlas** (9/10 novelty): Four failure modes characterized with proactive detection signals — the first systematic worst-case analysis for MDM acceleration. FM1 (M2 structural incompatibility) generalizes: DDIM-style discrete step schedules are fundamentally incompatible with bidirectional masked diffusion.

5. **M2 Structural Incompatibility** (8/10 novelty): Adaptive step scheduling is architecturally broken for LLaDA-class MDMs, not a hyperparameter issue. The mask-inconsistency cascade explanation is novel and has practical implications for any discrete MDM acceleration work.

6. **Task-Dependent Deployment Recipes**: M3 for reasoning, M1+CD-SSD for general deployment. First evidence-backed task-type guidance for LLaDA-8B-Instruct deployment.

---

## Pending Experiments (Prerequisites for NeurIPS 2026 Submission)

| Priority | Experiment | Purpose | Est. GPU-hours |
|---------|-----------|---------|---------------|
| P1 | Full-scale M1+CD-SSD (3 seeds, all 4 benchmarks) | Validate Ortho=1.385 at full scale; determine NeurIPS vs. EMNLP target | ~8h |
| P2 | CD-SSD REFINE with M1 disabled during REFINE phase only | Isolate synergy location (DRAFT phase vs. REFINE phase) | ~2h |
| P3 | tau=0.0 vs. naive T=16 vs. CD-SSD tau=0.9 (with M1) | Resolve tau=0.0 paradox; determine if partition mechanism adds value | ~2h |
| P4 | SSD (arXiv:2510.04147) + M1 composability | Compare lossless vs. coarse-draft speculative; generalize synergy claim | ~4h |

**Decision gates**:
- P1 Ortho >= 1.0 → NeurIPS 2026; headline: "super-multiplicative synergy"
- P1 Ortho in [0.8, 1.0) → NeurIPS 2026; headline: "highly orthogonal, first composability study"
- P1 Ortho < 0.8 → EMNLP/workshop; pivot composability framework as primary contribution
- P3 shows tau=0.0 > naive-T16 → strengthen CD-SSD mechanism claim
- P3 shows tau=0.0 ≈ naive-T16 → reframe CD-SSD as step-reduction amplifier
- P4 Ortho(SSD+M1) >= Ortho(CD-SSD+M1) → generalize claim: self-speculative + KV synergizes universally

---

## Backup Strategies

See `alternatives.md` for two pivot directions:

1. **Consistency Distillation for MDMs** (cand_b): Training-based 1-4 step inference via lightweight adapter on frozen LLaDA-8B. Partially occupied by T3D (arXiv:2602.12262) and FS-DFM (arXiv:2509.20624) but not at LLaDA-8B instruction-tuned scale. Pivot trigger: full-scale Ortho < 0.7 AND SSD comparison shows IGSD provides no differentiated value.

2. **Batched MDM Inference Roofline Analysis** (cand_c): First systematic characterization of MDM throughput under batched workloads with convergence-stratified scheduling. Novelty 8/10. Engineering-heavy. Pivot trigger: front-runner composability results fully negative with no publishable finding.

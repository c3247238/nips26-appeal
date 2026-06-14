# Result Debate Synthesis: Anatomy of Feature Absorption in SAEs (Post-R4 Update)

**Synthesizer**: Senior Research Director
**Date**: 2026-04-13
**Perspectives integrated**: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist
**Round**: 4 (all blocking experiments completed; this supersedes the 2026-04-12 pre-R4 synthesis)

---

## 1. Consensus Map

The following conclusions are endorsed by all or nearly all six perspectives and constitute **high-confidence findings**.

### 1.1 EDA Lower Bound Theorem — Validated (Unanimous)

All perspectives accept that the biconvex lower bound theorem (EDA >= delta^2 * sin^2(theta) / (2 + delta^2)) is mathematically correct and empirically supported by SynthSAEBench (AUROC = 1.0, F1 = 0.974). Real-data group separation at Gemma L12-16k (Mann-Whitney p = 6.4e-5, Cohen's d = 1.02) is not contested. The theorem is the paper's strongest theoretical anchor.

### 1.2 EDA Is Mathematically Equivalent to 1 − Decoder Cosine Similarity (Unanimous)

R4 confirms across all three model families (Gemma 2B, GPT-2 Small, Llama-3.1-8B) that EDA = 1 − cos(w_enc, d_dec), with Pearson r = −1.000. This is an exact mathematical identity, not a correlation. Every perspective acknowledges this: the paper cannot present EDA as a novel metric. The novelty is confined to (a) the formal lower-bound theorem providing the first theoretical justification for an existing SAEBench metric, and (b) the systematic regime characterization of where that metric discriminates absorbed from non-absorbed latents.

### 1.3 EDA Passes in Only a Regime-Specific Subset of Configurations (Unanimous)

3 of 8 tested SAE configurations pass AUROC >= 0.65: Gemma L5-16k (0.698, proxy labels), Gemma L12-16k (0.776, proxy labels), GPT-2 L6 (0.650, direct labels). Five configurations fail, including one showing a directionally reversed signal (GPT-2 L10, AUROC = 0.344, Cohen's d = −0.37). All perspectives agree that the "regime-specific detector" framing is the only defensible one.

### 1.4 H3 Cross-Domain Absorption Is Falsified — Not Merely Pending (Unanimous)

This is the most important update relative to the 2026-04-12 synthesis. R4's shuffled hierarchy control shows that real absorption rates are statistically indistinguishable from shuffled null for all 9 hierarchy-SAE combinations (0/9 exceed the shuffled p95 threshold; real/shuffled ratios range 0.89–1.43). The intra-RAVEL coherence rho = 0.924 from R3 is also revealed as artifactual: shuffled labels produce rho = 1.0 trivially, demonstrating that coherence reflects SAE-config-level scaling of absorption rates rather than genuine hierarchical structure. H3 is definitively falsified, not downgraded. Bridge-model probes never measured genuine hierarchical absorption in Gemma SAE space.

### 1.5 Three-Subtype Taxonomy Exists; Early Dominance Is Threshold-Contingent (Largely Unanimous)

The late > early EDA ordering holds robustly across all 5 tested threshold values (tau = 0.2–0.4) on L12-65k, which is the only statistically powered configuration (n = 65, KW p = 0.0002). All perspectives agree on this ordering. However, the "75% early dominance" headline is not a stable constant: early proportion varies from ~32% (tau = 0.2) to ~95% (tau = 0.4) on L12-65k. The qualitative finding that many absorbed latents lack corresponding decoder directions is robust; the 75% figure is threshold-contingent. L12-16k (n = 16) is underpowered and cannot independently validate the taxonomy.

### 1.6 D-EDA Is Mathematically Redundant (Unanimous)

As a direct consequence of EDA = 1 − DecCos, D-EDA's norm is a monotone transformation of EDA. Any AUROC comparison between EDA and D-EDA as raw predictors yields delta = 0.0 identically — confirmed on GPT-2 direct labels. D-EDA must be dropped entirely as a separate contribution. The R3 GPT-2 L10 result (D-EDA = 0.762 vs. EDA = 0.336) referred to a complex derived "absorption indicator," not the raw D-EDA norm; this distinction requires clarification in the paper.

### 1.7 ITAC Fails H5 (Unanimous)

Mean FN reduction = 2.69–3.14% against a 20% target. H5 is falsified. The most important mechanistic insight, agreed upon by all perspectives, is that ITAC is structurally inapplicable to approximately 75% of absorbed latents (early-type, decoder-absent) — ITAC can only operate on the ~13% late-type minority. The single case with notable improvement (j_idx = 61217, ~22.7% FN reduction) confirms the mechanism works for late-type targets but does not rescue ITAC as a general contribution. ITAC was evaluated only on synthetic activations; R4D (real-activation validation) was not completed.

### 1.8 H6 Scaling Analysis Is Falsified (Unanimous)

Partial rho(width, absorption | L0) = +0.37 (positive, contradicting the predicted negative sign). The test was underpowered by design — L0 variation across canonical Gemma Scope SAEs at matched layers spans only 65–72, insufficient to isolate the width effect. H6 should be reported as a methodological negative result noting that canonical SAEs do not provide the L0 variation needed to test this hypothesis.

---

## 2. Conflict Resolution

### Conflict A: How Much Does EDA = DecCos Destroy the Contribution?

**Optimist/Strategist position**: The formula being equivalent to an existing SAEBench metric is a transparency advantage, not a flaw. Novelty resides in the formal theorem, the regime map, and the cross-model empirical characterization. Prior work used decoder cosine similarity as an ad-hoc diagnostic without theoretical grounding or regime understanding.

**Skeptic/Methodologist position**: A reviewer who knows SAEBench will immediately see that EDA = 1 − DecCos and conclude the paper proposes nothing new at the metric level. The theorem is genuinely novel, but "we proved a lower bound on an existing metric" is a weaker contribution than "we proposed a new metric."

**Judgment**: The skeptic is right on the severity of the reviewer risk — this is a HIGH-probability objection. The optimist is right that the theorem + regime characterization constitute a genuine, non-trivial contribution. The correct resolution is to **front-load the equivalence explicitly in the abstract or introduction**, immediately pivoting to the novelty of the theorem and the empirical regime map. Burying the equivalence or discovering it late in the paper is a fatal strategic error. The paper's working title and framing must not suggest "new metric" — it must frame the work as "theoretical grounding and regime characterization of an existing but atheoretical diagnostic."

### Conflict B: Is GPT-2 L6 AUROC = 0.650 Sufficient Evidence for Cross-Model Detection?

**Optimist position**: Direct FeatureAbsorptionCalculator labels, statistically significant (p = 0.027, Cohen's d = 0.53), AUROC CI [0.531, 0.761]. The GPT-2 result is the cleanest data point in the paper and confirms the Gemma proxy-label results are not inflated artifacts.

**Skeptic/Methodologist position**: n_pos = 18 is too small for stable AUROC estimation. The CI lower bound (0.531) is barely above chance. EDA and DecCos AUROC are identical at GPT-2 L6 (delta = 0.0), meaning EDA provides no advantage over the simpler metric for this model. Cohen's d = 0.53 is "medium," not "large."

**Judgment**: Both are partially correct. The GPT-2 L6 result is genuinely positive — the statistical tests are significant and the direct-label methodology is sound. But it is a single data point with a small positive pool. The honest framing: "We confirm that the decoder cosine metric detects absorption at AUROC = 0.650 on GPT-2 L6 with direct labels from the canonical measurement tool; this validates the theoretical regime prediction that 16k-width, mid-layer SAEs are a favorable operating envelope." The claim cannot be stronger than "regime-specific, cross-model consistency" — not "reliable universal detection."

### Conflict C: Is the GPT-2 L10 Reversal (AUROC = 0.344) Damaging or Informative?

**Optimist position**: EDA's reversal at L10 is mechanistically informative — absorbed latents at deep layers may involve different dynamics (encoder alignment toward competing features rather than generic misalignment). D-EDA rescues detection at L10 (AUROC = 0.762 per R3 results using the complex indicator).

**Skeptic/Revisionist position**: The reversal is highly significant (p = 0.0008, d = −0.37) and means EDA actively anti-predicts absorption at L10. This is not merely "regime failure" — it suggests the lower bound theorem's premise (that absorbed latents always have higher encoder-decoder misalignment) is incomplete when non-absorbed latents in the same config are more polysemantic than absorbed ones.

**Judgment**: The skeptic's concern is valid and must be prominently disclosed. The revisionist's mechanistic explanation (polysemanticity interacts with EDA signal-to-noise) is the most compelling theoretical account. The paper should present the L10 reversal as a finding that reveals a boundary condition of the theorem: EDA functions as an absorption signal only when the background polysemanticity level is lower than the absorption-driven EDA elevation. This is actually a scientifically interesting negative result, not merely a limitation.

### Conflict D: Does the Taxonomy Finding Survive the Threshold Sensitivity Problem?

**Optimist position**: The late > early ordering is robust at all 5 thresholds. The qualitative insight (many absorbed latents lack decoder coverage) survives regardless of whether "many" is 32% or 95%. KW p = 0.0002 on L12-65k.

**Skeptic/Revisionist position**: The "75% early dominance" headline is load-bearing for the paper's central narrative ("most absorption is dictionary coverage failure"). If early% drops to 32% at tau = 0.2, the narrative collapses to "absorption is one-third dictionary coverage failure, one-third encoder suppression, one-third partial" — which is a weaker story.

**Judgment**: The skeptic identifies the key vulnerability correctly. The **robust claim** is: "The late > early EDA ordering is consistent, indicating absorbed latents classified as late-type (decoder-present, encoder-suppressed) have systematically higher encoder-decoder misalignment than those classified as early-type." The **fragile claim** is any specific percentage (75%, 32%, etc.). The paper must report the full threshold sensitivity curve as a primary figure, frame the early-dominance observation as "at the standard threshold," and emphasize the ordering robustness as the scientific finding. The practitioner implication — that most absorbed latents are resistant to encoder-side inference-time correction — holds for any threshold where early > 50%, which is satisfied at tau >= 0.25 on L12-65k.

### Conflict E: PROCEED or PIVOT?

**Strategist/Optimist position**: PROCEED immediately to paper writing. Two genuine contributions (EDA regime characterization, three-subtype taxonomy) are validated. Run Backup A (amortization gap experiment, 1–2 GPU-hours) as a third contribution before writing begins. This is the highest information-gain-per-GPU-hour experiment available and adjudicates the Tang vs. O'Neill mechanistic debate.

**Skeptic/Comparativist position**: The comparativist gives the paper 6.5/10 for MI workshop and 4.5/10 for top-tier. A two-contribution paper with thin evidence (n_pos = 18 for cross-model detection, 2 SAE configs for taxonomy) may be marginally publishable at workshop level only. Expanding the taxonomy to GPT-2 configs (where direct labels already exist) would materially strengthen the claim at low cost.

**Judgment**: PROCEED is the correct verdict, but the strategist's Backup A priority is well-argued. Before entering the writing phase, two low-cost additions are recommended: (1) run Backup A (amortization gap, ~1–2 GPU-hours) to potentially add a third mechanistic contribution; (2) apply the taxonomy pipeline to GPT-2 L6 and L10 (direct labels already available, no new data needed, ~0.5 GPU-hours) to expand the taxonomy evidence base from 2 Gemma-only configs to 2 Gemma + 2 GPT-2 configs, strengthening the cross-model robustness of the early-dominance observation.

---

## 3. Result Quality Score: 5.0 / 10

**Rationale** (updated from 5.5/10 in the 2026-04-12 synthesis to reflect post-R4 data):

| Component | Score | Justification |
|-----------|-------|---------------|
| Theory: EDA lower bound theorem | +2.0 | Valid, synthetic-validated, genuinely novel as formal result |
| EDA regime characterization | +1.0 | 3/8 pass (reduced from pre-R4 framing of 2/6); GPT-2 direct-label anchors the claim |
| Three-subtype taxonomy | +1.0 | KW p = 0.0002, ordering robust at all thresholds; threshold-sensitivity concern is real |
| EDA = DecCos identity finding | +0.5 | The confirmed identity is a finding in itself; bridges theory to SAEBench practice |
| D-EDA | 0.0 | Mathematically redundant; no credit |
| ITAC | −0.5 | H5 falsified; creates unmet expectations, demoted to negative result |
| H3 cross-domain | −1.0 | Falsified by R4 shuffled control; previous synthesis gave +1.0 for this, now reversed |
| Methodology gaps | −0.5 | Gemma 2B still gated; proxy-label Gemma AUROCs unverifiable; Llama contributes no AUROC |
| H6 scaling | 0.0 | Cleanly falsified; honest negative result, no net contribution |
| Honest negative results | +0.5 | ITAC failure + H3 falsification + H6 falsification = informative characterization of what does not work |

**Net: 5.0/10.** This is a publishable characterization and negative-results paper at the workshop level, upgradeable to a focused EMNLP short paper or AISTATS submission if Backup A produces a clean result and the taxonomy is extended to additional configs.

---

## 4. Key Findings (High-Confidence)

1. **EDA = 1 − decoder cosine similarity: a formal theorem grounds an existing atheoretical metric.** The biconvex lower bound theorem is the paper's primary theoretical contribution. It explains why the existing SAEBench decoder cosine metric correlates with absorption in some regimes (mid-layers, narrow SAEs) and fails in others. The theorem is empirically anchored by SynthSAEBench (AUROC = 1.0) and real-data group separation (Gemma L12-16k, Cohen's d = 1.02).

2. **The decoder cosine metric detects absorption in a clearly delineated regime (16k-width, layers 5–12) and fails outside it.** 3/8 tested configurations pass AUROC >= 0.65 across two model families (Gemma 2B and GPT-2 Small). The regime-dependence is informative: EDA's failure at 65k-width SAEs is consistent with the early-absorption dominance finding (early-type latents have lower EDA regardless of absorption, diluting the signal). GPT-2 L10's reversed direction reveals an additional boundary condition: EDA anti-predicts absorption when background polysemanticity among non-absorbed latents exceeds absorption-driven EDA elevation in absorbed latents.

3. **Three-subtype taxonomy: the late > early EDA ordering is robust across all tested thresholds.** Absorbed latents classified as late-type (decoder-present, encoder-suppressed) consistently exhibit higher encoder-decoder misalignment than early-type (decoder-absent) absorbed latents. This ordering is statistically significant (KW p = 0.0002 on L12-65k) and stable across all 5 threshold variants (tau = 0.2–0.4). The proportion of early-type absorbed latents is threshold-dependent (32–75% at tau = 0.2–0.3 on L12-65k), but early-dominance holds at the standard threshold (tau = 0.3: ~72–75% on both configs).

4. **Most SAE absorption corresponds to dictionary coverage failure, not encoder suppression.** At the canonical threshold, most absorbed latents lack a corresponding decoder direction — the SAE never allocated dictionary capacity to the parent feature. This means training-time solutions (wider dictionaries, hierarchically-aware training objectives like Matryoshka SAE) are appropriate for the majority of absorbed latents, while inference-time encoder corrections (ITAC, MP-SAE-style) are structurally applicable only to the minority late-type category (~13%). This reframes the dominant focus of the absorption mitigation literature.

5. **H3 cross-domain absorption is falsified.** Absorption rates in RAVEL entity-attribute hierarchies measured with bridge-model probes are statistically indistinguishable from shuffled null for all 9 tested domain-SAE combinations. The R3 intra-RAVEL rho = 0.924 is artifactual. Cross-domain absorption characterization requires same-model probes trained on the target model (Gemma 2B or Llama-3.1-8B), which remain HF-gated. Honesty about this falsification is essential for the paper's credibility.

---

## 5. Methodology Gaps (Critical, from Methodologist + Skeptic)

### Critical Gap 1: Gemma 2B Remains HF-Gated — Proxy Labels Unverifiable

The two strongest Gemma AUROC results (L5-16k: 0.698, L12-16k: 0.776) rely on Neuronpedia proxy labels rather than direct FeatureAbsorptionCalculator outputs. These results cannot be independently verified. The GPT-2 L6 direct-label result (0.650) is the paper's single fully reproducible data point for detection validation. This asymmetry (best results proxy-based; only confirmed direct result is weaker) must be prominently disclosed. The paper must not imply equivalence between Gemma proxy-label and GPT-2 direct-label AUROCs.

### Critical Gap 2: Taxonomy Evidence Base Is Narrow (2 SAE Configs, Gemma Only)

The three-subtype taxonomy is validated on only two SAE configs (L12-16k: n = 16, L12-65k: n = 65), both from Gemma Scope. At L12-16k, n = 16 absorbed latents is severely underpowered (KW p = 0.237, no significance). The entire taxonomic finding rests on L12-65k. Applying the same analysis to GPT-2 configs (where direct absorption labels are already available) would materially strengthen the claim at minimal additional cost (~0.5 GPU-hours).

### Serious Gap 3: Threshold Sensitivity Must Be in Main Text, Not Supplementary

The threshold sensitivity curve (early/late/partial proportion vs. tau = 0.2–0.4) is the most important nuance in the paper's central finding. At tau = 0.2, early proportion drops to ~32% on L12-65k. This is not a supplementary ablation — it is the methodological basis for assessing the robustness of the early-dominance claim. It belongs in the main paper, presented with the ordering stability (late > early at all thresholds) as the robust result and the 75% figure as the threshold-conditional result.

### Serious Gap 4: EDA = DecCos Identity Creates a Spurious "Baseline Comparison"

The previously reported "+0.396 AUROC advantage of EDA over decoder cosine baseline" is an artifact of comparing EDA against a differently parameterized version of the same metric. With EDA ≡ 1 − DecCos confirmed, any comparison between the two as raw predictors yields delta = 0.0 identically. This spurious comparison must be removed from all figures and tables; the relevant comparison is EDA (= DecCos) vs. alternative baselines (e.g., random, decoder norm, dead latent fraction).

### Minor Gap 5: Llama AUROC Contribution Is Not Reproducible

Llama-3.1-8B's contribution to the paper is limited to weight-level statistics (mean EDA, layer trends). No absorption labels exist for Llama. The "3 model families" claim in the cross-model characterization refers to weight-level statistics, not detection validation. The paper must clearly distinguish weight-level EDA characterization from detection AUROC claims; these are not interchangeable evidence for generalizability.

---

## 6. Competitive Position (from Comparativist, synthesized)

**Unique contribution**: This paper provides the first formal theoretical justification for the decoder cosine similarity metric as an absorption proxy (the lower bound theorem), and the first mechanistic subtype taxonomy of SAE feature absorption. No concurrent work addresses either claim. The "Sanity Checks for SAEs" framing (Korznikov et al., 2026) that SAEs are fundamentally broken actually motivates this work — understanding failure modes is necessary precisely when SAE utility is contested.

**Where the paper competes well**:
- The theorem is novel (no prior work derives a formal lower bound connecting encoder-decoder geometry to absorption degree).
- The three-subtype taxonomy is novel (Chanin et al. 2024 treats all absorption uniformly; Feature Hedging characterizes a complementary failure mode but not subtypes).
- The connection between early-dominance and Matryoshka SAE's superior SAEBench performance is a new explanatory insight (the nested training inherently allocates dictionary capacity to parent features, directly attacking the dominant subtype).
- Honest reporting of 4 falsified hypotheses strengthens credibility with reviewers who value scientific rigor.

**Where the paper is weak relative to SOTA**:
- Detection quality: EDA (AUROC 0.65–0.78 in favorable regime, 3/8 pass) vs. Chanin et al. supervised metric (ground-truth, tested on hundreds of SAEs via SAEBench). The gap is large in scope, smaller in per-config performance.
- Mitigation: ITAC (2.69% FN reduction) is not competitive with Matryoshka SAE (#1 SAEBench absorption), OrtSAE (65% reduction), or masked regularization. It is a negative result only.
- Scale: 8 SAE configurations vs. SAEBench's 200+. This is a focused mechanistic study, not a benchmark.

**Venue**: NeurIPS 2026 MI Workshop (strong fit; honest negative results and mechanistic taxonomy are valued). EMNLP 2026 main (possible with expanded taxonomy evidence and strong framing). Not top-tier main (NeurIPS/ICML/ICLR) with current evidence base.

---

## 7. Hypothesis Update

| Hypothesis | Final Status | Revised Claim |
|-----------|-------------|---------------|
| H1: EDA lower bound | **Confirmed (regime-specific)** | The theorem is valid; the practical detector based on it (= decoder cosine similarity) achieves AUROC 0.65–0.78 in 16k-width, mid-layer SAEs across Gemma and GPT-2 architectures. |
| H2: D-EDA improvement | **Falsified (definitively)** | EDA = 1 − DecCos is exact; D-EDA norm is a monotone transform; AUROC delta = 0.0 identically on direct labels. D-EDA is not a separate metric. |
| H3: Cross-domain absorption | **Falsified (definitively)** | R4 shuffled control: 0/9 domain-SAE combinations exceed shuffled p95. Intra-RAVEL rho = 0.924 from R3 is artifactual (shuffled produces rho = 1.0). Cross-domain claim requires same-model probes. |
| H4: Three-subtype taxonomy | **Confirmed with caveats** | Late > early EDA ordering robust across all 5 thresholds on L12-65k (KW p = 0.0002). Early-dominance proportion is threshold-dependent (32–75%). Taxonomy validated on only one statistically powered configuration. |
| H5: ITAC efficacy | **Falsified** | 2.69–3.14% FN reduction vs. 20% target. ITAC is structurally inapplicable to ~75% of absorbed latents (early-type). Negative result confirms early-dominance finding from a complementary direction. |
| H6: Scaling sign reversal | **Falsified** | Partial rho = +0.37 (wrong sign). Test underpowered by design (canonical SAEs have near-zero L0 variation at matched layers). |

---

## 8. Action Plan

### VERDICT: PROCEED — Two Proven Contributions + Run Backup A for a Potential Third

The paper must not wait for Gemma 2B model access (external gating, uncontrolled timeline). The writing gate (r4_writing_gate.json: go_write = true) has been passed. The dominant strategy is:

### Step 1: Run Backup A — Amortization Gap Controlled Dictionary Experiment (1–2 GPU-hours, IMMEDIATE)

**Rationale**: This is the highest information-gain-per-GPU-hour experiment available. It adjudicates the core mechanistic question raised by the taxonomy: is early-type absorption caused by the encoder's inability to represent the parent feature (amortization gap, O'Neill et al.) or by the optimization dynamics of dictionary learning (loss landscape structure, Tang et al.)? Either outcome directly contributes to the paper and requires no new data downloads.

**Protocol**: Fix Gemma Scope L12-16k decoder weights. Run three encoding methods on 10,000 tokens: (1) standard feedforward encoder, (2) Orthogonal Matching Pursuit (OMP) on D at matched L0, (3) 2-pass encoder with one residual correction step. Compute absorption rates using the Chanin first-letter metric. Compare with paired Wilcoxon signed-rank test.

**Decision gate**: OMP absorption < 50% of feedforward → amortization gap dominates (practitioners should use iterative encoding). OMP absorption ~ feedforward → loss landscape dominates (dictionary and training objectives must change). Both outcomes are publishable and connect directly to the 75% early-dominance finding.

### Step 2: Extend Taxonomy to GPT-2 Configs (0.5 GPU-hours, OPTIONAL but Recommended)

Apply the three-subtype classification pipeline to GPT-2 L6 (n_pos = 18) and GPT-2 L10 (n_pos = 39), which already have direct FeatureAbsorptionCalculator labels. No new data collection required. This transforms the taxonomy from a Gemma-only observation to a cross-model finding, directly responding to the skeptic's concern that the evidence base is too thin.

**Expected outcome**: If early-type dominance at tau = 0.3 replicates on GPT-2 L6 and L10, the taxonomy claim is strengthened substantially with no additional compute cost beyond ~0.5 GPU-hours.

### Step 3: Finalize Paper as Two- or Three-Contribution Study (0 GPU-hours)

**Contribution 1 (Primary): EDA/DecCos Regime Characterization with Formal Lower Bound.**
Frame as: "We derive the first formal lower bound connecting encoder-decoder cosine similarity to absorption degree (Theorem 1), and systematically characterize the operating regime of this existing SAEBench metric: it discriminates absorbed from non-absorbed latents at AUROC 0.65–0.78 in 16k-width, mid-layer (5–12) SAEs across Gemma 2B and GPT-2 Small architectures. We explain EDA's regime failure at 65k-width and deep layers via the early-absorption dominance finding."

**Contribution 2 (Primary): Three-Subtype Taxonomy with Early-Absorption Dominance.**
Frame as: "We introduce the first mechanistic subtype taxonomy of SAE feature absorption, revealing that absorbed latents are geometrically heterogeneous. At the standard threshold (tau = 0.3), approximately 75% of absorbed latents are early-type (no corresponding decoder direction — dictionary coverage failure), while ~13% are late-type (encoder-suppressed despite decoder presence). The late > early EDA ordering is robust across all thresholds. This reframes absorption primarily as a training-time dictionary allocation problem rather than an inference-time encoder correction problem, explaining why Matryoshka SAEs outperform alternatives and why ITAC fails."

**Contribution 3 (Contingent on Backup A): Amortization Gap Controlled Dictionary Experiment.**
If Backup A produces a clean result, add as a mechanistic third contribution connecting the taxonomy to the causal question of why early absorption happens.

**Supplementary / Honest Negatives**:
- H3 falsification (cross-domain): the failed RAVEL experiment with the shuffled control result; frame as methodological progress (the measurement infrastructure is established; the finding awaits same-model probe access).
- ITAC failure: confirmatory evidence for early-dominance, not a standalone contribution.
- H6 failure: methodological negative; canonical SAEs are an underpowered test bed for scaling hypotheses.
- D-EDA: mathematically redundant; the complex derived indicator may capture directional information at deep layers but requires further investigation.

### Step 4 (Background, Non-Blocking): Pursue Gemma 2B and Llama-3.1-8B Access

If granted before camera-ready: re-run EDA validation with direct Chanin labels on Gemma configs (potentially upgrading from 3/8 to 5/8 passing, and verifying whether proxy-label AUROCs hold up); re-run taxonomy with Gemma probes for cross-domain validation; compute Llama AUROC. These are revision-stage improvements, not blocking dependencies.

---

## 9. Open Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Reviewer: "EDA = DecCos, so where is the novelty?" | HIGH (high probability) | Front-load the equivalence in abstract/intro; argue novelty is the theorem + regime characterization, not the formula. Pre-empt rather than discover. |
| Reviewer: "75% early is threshold-contingent" | HIGH (high probability) | Present full threshold sensitivity curve in main text. Emphasize the robust result: late > early ordering at all thresholds. |
| Reviewer: "GPT-2 L10 reversal invalidates the theorem" | MEDIUM | Explain polysemanticity boundary condition; present as a finding that illuminates when EDA works, not just a failure. |
| Taxonomy rests on L12-65k alone (n=65) | MEDIUM | Run GPT-2 taxonomy extension (Step 2); if early-dominance replicates cross-model, this concern is largely answered. |
| Sanity Checks for SAEs framing (Korznikov et al.) | MEDIUM | Cite "Use SAEs for Discovery" counter-paper; frame absorption characterization as enabling informed use, not defending SAE utility unconditionally. |
| Backup A produces ambiguous result (50–80% reduction) | LOW-MEDIUM | Report partial reduction with confidence intervals; "both mechanisms contribute" is still a publishable finding. |
| Gemma 2B access never resolved | MEDIUM (realized) | Accept GPT-2 direct-label result as primary validation; Gemma proxy-label results as supplementary with explicit caveat. |

# Experiment Result Analysis

## Key Results Summary

All four primary hypotheses were tested in iter_002. The quantitative outcomes are:

| Metric | Value | Verdict |
|---|---|---|
| H1: RD threshold AUROC (absorbed pairs, decoder cos²) | 0.318 (null = 0.500) | FALSIFIED — reversed direction |
| H2: Cross-domain absorption (semantic hierarchies) | ~0.0 absorption rate (10⁻⁸) | NOT SUPPORTED |
| H3: ASI AUROC | 0.422 (null mean = 0.497) | FALSIFIED — anti-correlated |
| H4a: Phase transition (LRT p) | p = 0.456, BIC diff = −3.22 | NOT SUPPORTED |
| H4b: Hysteresis | Absorption 0.959 → 0.960 (saturated) | NOT CONFIRMABLE |
| EDA AUROC (proxy labels, JB L6) | 0.681 | Moderate positive signal |
| EDA AUROC (exact Chanin labels, JB L6) | 0.650 | Moderate positive signal |
| EDA AUROC (JB L10) | 0.337 | REVERSED — layer-specific failure |
| Encoder norm AUROC (exact Chanin labels) | 0.757 | Strongest detector, post-hoc |
| cos(enc_p, dec_c) AUROC (pairwise pilot, proxy labels) | 0.730 | Strong new signal |
| Standard/ReLU vs TopK EDA gap | 0.200 (z = −10.4) | Significant architectural signal |
| Wilcoxon p (absorbed vs non-absorbed decoder geometry, L6) | p = 1.34×10⁻⁶ | Significant (direction reversed from H1) |
| C2 first-letter absorption rate (v9) | 0.008 (1/120 events) | Near-null |
| Saturation across all E1 SAE configurations | 85–99% absorption | Phase transition untestable |

---

## Debate Perspectives Summary

- **Optimist:** Found five publishable findings — EDA AUROC=0.681 on proxy/0.650 on exact labels, enc_norm AUROC=0.757 as novel highest detector, reversed H1 geometry as a publishable theoretical reframing ("encoder amplification, decoder avoidance"), L1 vs TopK architectural gap (z=−10.4), and near-perfect animacy probe (F1=1.000). Proposed an EDA-centric reframing paper and outlined six follow-up experiments. Assessed the story as publishable, though on different terms than originally proposed.

- **Skeptic:** Raised four FATAL-severity issues: proxy label contamination (Jaccard=0.115 between proxy and exact labels), EDA failure in 5 of 6 tested SAE configurations (works only for JB L6, reversed for L10 and AJT), C2 cross-domain rate = 0, and H1/H3 already confirmed falsified. Also flagged an unresolved theory-data tension: Proposition 2 predicts small EDA for absorbed features, but the data shows large EDA. Concluded that a publishable path exists but requires exact behavioral labels, honest scope limitations, and resolution of the Proposition 2 contradiction.

- **Strategist:** Assessed signal strength dimension by dimension. H1, H3: NOISE. H2: NOISE for semantic hierarchies. H4: WEAK/absent. EDA: MODERATE signal (works at L6, fails at L10, n_pos=18 exact labels). Rendered explicit PROCEED verdict with mandatory scope reduction. Justified by: (1) at least one hypothesis has moderate+ signal with a clear publication path; (2) GPU budget is tractable (~4–6 GPU-hours); (3) pivot to alternatives has lower expected return. Recommended three priority experiments: cross-directional cosine validation (~1h), EDA-norm layer profile (~1h), Gemma Scope replication (~2–3h).

- **Comparativist:** Found no prior published AUROC-based probe-free absorption detector, making EDA the first such result even at modest AUROC=0.681. Flagged Proposition 1 + Corollary 1 (frequency cancels) as genuinely novel relative to published work. Assessed venue as mid-tier (BlackboxNLP/EMNLP workshop, not NeurIPS/ICML) in current form; ICLR 2027 if validated on Gemma Scope ground-truth labels. Strengthening plan: run FeatureAbsorptionCalculator on all 26 letters (not just 8), measure actual theta for confirmed absorbed pairs, and report negative results systematically.

- **Methodologist:** Identified high-severity unresolved issues: encoder norm (AUROC=0.757) outperforms EDA (0.650) and was discovered post-hoc; EDA AUROC varies from 0.469 to 0.681 depending on threshold (not fixed a priori); precision@k = 0.0 for all k (no practical detection utility); C2 protocol recovers only 0.83% vs. Chanin's 15–35% (method is insensitive by 20×); Proposition 2 mechanism contradicted by B1 decomposition. Scored reproducibility at 3/5 due to non-fixed threshold and private iter_001 exact label source. Required D2 EDA-norm comparison, fixed threshold, and C2 calibration before writing.

- **Revisionist:** Required a fundamental mental model revision. Corrected assumption: "absorbed pairs have high decoder cosine similarity" → actual: "absorbed (letter) features are geometrically isolated." The `cos(enc_p, dec_c)` metric (AUROC=0.730) is theoretically more grounded than EDA. Proposed three new testable hypotheses (NH1: geometric isolation, NH2: cos(enc_p, dec_c) generalizability, NH3: Chanin measurement is task-specific). Endorsed research question reframing from causal theory to empirical geometric characterization.

---

## Analysis

### 1. Method Feasibility

The core detection methods — EDA and encoder norm — produce detectable signals above chance against exact absorption labels at GPT-2 Small layer 6. The EDA AUROC of 0.650 on exact Chanin labels (n_pos=18) is statistically significant (z=+2.49, Wilcoxon p=1.34×10⁻⁶). The encoder norm AUROC=0.757 is a post-hoc finding but coherent. The `cos(enc_p, dec_c)` pairwise metric achieves AUROC=0.730 on proxy labels and represents the most theoretically grounded absorption indicator.

However, the methods have critical scope limitations: EDA works only for the JB/Standard L1 SAE at layer 6. At layer 10 (same model) AUROC=0.337 (reversed). For all three AJT architecture variants, EDA is also inverted. This means the detection method currently generalizes across neither layers nor architectures within the same base model. This is a serious limitation but does not constitute a fatal flaw — it is a characterization result that defines the scope of the finding.

The practical precision is low: precision@k = 0.0 for all tested k, meaning the detector cannot be used to rank-select absorbed features reliably. This is an honest limitation that must be disclosed but does not prevent publication of the AUROC finding.

### 2. Performance

Against the only valid benchmark (exact Chanin ground-truth labels, n_pos=18), the best surviving metrics are:
- Encoder norm: AUROC=0.757, AUPRC=5.68×base rate
- cos(enc_p, dec_c): AUROC=0.730 (proxy labels, needs exact label validation)
- EDA: AUROC=0.650, AUPRC=2.09×base rate

These are the first published probe-free absorption detection results at any AUROC level. Since no prior method exists for comparison, the performance is evaluated relative to random (AUROC=0.500): all three metrics exceed random significantly. The original proposal required AUROC≥0.70 for H3 — which is not met by EDA (0.650), but is met by encoder norm (0.757) and plausibly by cos(enc_p, dec_c) (0.730 on proxy labels).

Three of four primary hypotheses are falsified or unsupported, but the surviving empirical signals are statistically robust and genuinely novel in the literature.

### 3. Improvement Headroom

There is a clear, tractable improvement path that requires modest additional compute (~4–6 GPU-hours):

1. **Validate cos(enc_p, dec_c) on exact Chanin labels** (~1h): This metric achieves AUROC=0.730 on proxy labels with z=6.4 above null. If it holds on exact labels, it becomes the primary contribution (stronger than EDA, better theoretical grounding). Risk: medium (AUROC could drop below 0.60 on the exact 18-label set).

2. **Compute EDA-norm layer profile across L2–L10** (~1h): Characterizes layer-specificity, provides honest scope, and tests the "post-absorption re-alignment" hypothesis. This is a required characterization for any paper claim about EDA.

3. **Gemma Scope replication** (~2–3h): Cross-model validation is the single highest-impact experiment for venue target. If EDA-norm and/or cos(enc_p, dec_c) generalize to Gemma 2 2B with AUROC≥0.65, the paper becomes publishable at ICLR/EMNLP. If not, the finding is GPT-2 Small-specific and targets a workshop.

These three experiments have clearly defined success/failure criteria and collectively determine the paper's venue ceiling. The improvement path is not speculative — it is a direct extension of confirmed positive signals.

### 4. Time-Cost Tradeoff

Continuing the current direction requires ~4–6 GPU-hours to complete the three priority experiments above. Starting fresh with any of the four backup alternatives (absorption-as-diagnostic, mitigation benchmark, deconfounding, feature anchoring) would require 3–8+ GPU-hours and would not have the benefit of already-validated positive signals to build on.

The current direction has:
- One confirmed moderate-to-strong geometric detection signal (EDA, encoder norm, cos(enc_p, dec_c))
- A formally novel theoretical result (Proposition 1 + Corollary 1) that is correct even though its geometric direction was empirically reversed for post-convergence weights
- A clean null result story (H1 direction reversal, H2 cross-domain null, H3 ASI failure) that enriches rather than destroys the paper narrative if framed as "what we learned about what absorption is not"

No backup alternative has comparable starting capital. The opportunity cost of pivoting is the abandonment of all accumulated validated signals.

### 5. Critical Objections Assessment

The skeptic's four FATAL issues are evaluated individually:

**Fatal 1 — Proxy label contamination:** Resolvable. The exact Chanin labels already exist (n_pos=18 at GPT-2 L6 from iter_001 R4). Running FeatureAbsorptionCalculator on all 26 letters (not just 8) would expand n_pos to ~50–80 and allow fully label-clean evaluation. This is a 30–45 minute experiment, not a new research direction.

**Fatal 2 — EDA works in only 1 of 5+ SAE configurations:** Real but not fatal. It is a scope limitation. The paper can honestly report: "EDA predicts absorption in L1-trained SAEs with unit-norm decoder columns (AUROC=0.650–0.757) and fails in TopK/AJT architectures." This is a publishable characterization, not a falsification of the method. The mechanistic explanation (L1 penalty + unit-norm decoder forces encoder-decoder dissociation in a specific way) is testable and plausible.

**Fatal 3 — C2 cross-domain absorption rate = 0:** H2 is dead. The cross-domain claim cannot be made. This removes one of the four original contributions but does not invalidate the geometric detection contribution (H3/EDA reframing). The null result for semantic hierarchies can be framed honestly as a scoping insight.

**Fatal 4 — Proposition 2 theory-data tension:** Addressable. The paper should demote Proposition 2 from "theoretical prediction" to "mechanistic conjecture (unverified)" and present EDA as an empirical finding. Alternatively, derive a revised theory explaining why large EDA (not small EDA) characterizes absorbed features via the "encoder amplification, decoder avoidance" model proposed by the optimist. Either resolution allows the paper to proceed without internal contradiction.

None of these four issues requires abandoning the current direction. All are resolvable with minimal additional compute or honest framing adjustments.

---

## Decision Rationale

**The core case for PROCEED:**

1. **A publishable empirical signal exists and is confirmed.** EDA AUROC=0.650 on exact ground-truth labels (Chanin et al.), encoder norm AUROC=0.757, and cos(enc_p, dec_c) AUROC=0.730 on proxy labels — these are the first probe-free absorption detection metrics ever validated against any ground-truth labels. The signal is modest but real, above null by a margin that survives permutation testing. No prior paper has this.

2. **The theoretical contribution (Proposition 1) is intact.** The formal proof that absorption is rate-distortion optimal under flat sparsity penalties with the counterintuitive corollary (frequency cancels) is a genuine novel formal result, not contradicted by the empirical results. The direction reversal applies to post-convergence geometry, not to the optimality argument at training time. These are two different claims that the paper can separate cleanly.

3. **The improvement path is short and well-defined.** ~4–6 GPU-hours to validate cos(enc_p, dec_c) on exact labels, profile EDA-norm across layers, and run Gemma replication. Each experiment has clear success/failure criteria and directly determines the paper's contribution scope. This is far more efficient than starting fresh.

4. **Pivot alternatives offer lower expected return.** No backup idea has a validated empirical signal. Starting with absorption-as-diagnostic, mitigation benchmark, or other alternatives means rebuilding from zero. The current direction already has partial empirical validation and a formal theory — abandoning it now wastes that investment.

5. **The "falsified H1/H3" story enriches the paper.** The skeptic and revisionist both identified that the pattern of what works (geometric isolation, encoder-decoder cross-alignment) and what doesn't (decoder proximity, frequency ratio) is itself a publishable characterization of absorption geometry. A paper honestly reporting H1/H3 as falsified alongside EDA/encoder-norm as validated is more valuable to the field than one that only reports successes.

**The one serious risk (Gemma replication failure):** If EDA-norm and cos(enc_p, dec_c) do not generalize to Gemma 2 2B (AUROC<0.55), the entire finding becomes GPT-2 Small-specific. In that case, the venue target drops to a workshop, not NeurIPS/ICLR. But this risk exists in the current direction and in any pivot alternative — any paper on SAE absorption needs cross-model validation eventually. Running Gemma replication as Priority 3 (after two shorter experiments) is the correct risk management strategy, not a reason to pivot now.

**Conclusion:** The current direction has a confirmed, novel empirical signal, a formally intact theoretical contribution, and a clear 4–6 GPU-hour path to determining the paper's venue ceiling. No pivot alternative offers comparable starting capital. The four "fatal" objections from the skeptic are all addressable through additional experiments (1 is already nearly done) or honest framing adjustments. The three-agent consensus (strategist: explicit PROCEED; optimist: PROCEED with reframing; revisionist: reframe research question but keep direction) supports continuing.

DECISION: PROCEED

# Backup Research Ideas (Alternatives for Pivot)

## Backup 1: Absorption-as-Representational-Diagnostic

**Title:** Rethinking Feature Absorption: Cross-Layer Absorption Profiles as a Representational Fingerprint

**Core Idea:** Instead of treating absorption as a failure to mitigate, use absorption patterns as a *diagnostic signal* about the model's internal representation geometry. The "absorption graph" (directed graph of absorber→absorbee relationships) encodes the model's latent feature hierarchy and can be used to characterize how different layers, domains, and model architectures organize information.

**Key Hypothesis:** Absorption rate in a given layer/domain is a quantitative proxy for the degree to which that layer's representations violate SAE assumptions (linearity and independence). Layers with high absorption have more hierarchically entangled representations; layers with low absorption have more context-independent, atomic representations. The cross-layer absorption profile reveals the model's computational structure.

**Pivot Trigger:** The rate-distortion theory fails empirically (AUROC < 0.65) AND cross-domain absorption rates are indistinguishable from null controls. In this case, the reframing question — "what does absorption tell us about the model?" — becomes the primary contribution.

**Experiments:**
1. Measure absorption rate on the first-letter task across all 26 layers of Gemma 2 2B for both 16k and 65k SAEs (20–30 min). Test whether cross-layer absorption profile has systematic structure (peaks at middle layers, lower at early/late layers).
2. Construct the absorption graph for the first-letter task: which latents absorb which letter features? Analyze clustering, consistency across SAE widths, and correlation with token co-occurrence statistics (30 min).
3. Test whether the absorption graph predicts model behavior: tokens whose first-letter features are absorbed — does the model also struggle with those tokens in first-letter identification tasks at the behavioral level? (30 min)
4. Compare Matryoshka vs. TopK SAE absorption graphs: does Matryoshka's reduced absorption reflect genuine recovery of parent features, or merely re-encoding hierarchy (making the probe-based test trivially pass)? (30 min)

**Novelty:** No prior work has constructed the absorption graph and analyzed its representational structure. The diagnostic framing (absorption as signal rather than noise) is implied by the unified SDL theory but not empirically operationalized.

**Risk:** If absorption graphs are inconsistent across SAE training runs (consistent with Song et al., 2025 showing feature inconsistency), the absorption graph has limited diagnostic value. Mitigation: test cross-run consistency as the first analysis; require absorption graph similarity > 0.5 before proceeding.

---

## Backup 2: Systematic Mitigation Benchmark With Unified Metric Suite

**Title:** A Comprehensive Head-to-Head Evaluation of SAE Absorption Mitigation Methods Under Controlled Conditions

**Core Idea:** No existing paper has compared Matryoshka SAE, OrtSAE, ATM SAE, and masked regularization on the same model, layer, and absorption metrics in a single controlled study. Their reported results span different models, layers, and metrics, making fair comparison impossible. This paper provides the controlled benchmark that the community needs to make principled architectural choices.

**Key Hypothesis:** No single mitigation method dominates across all absorption metrics and downstream tasks. The choice of mitigation involves trade-offs that are currently invisible because methods have not been compared head-to-head. Specifically: Matryoshka reduces absorption but partially at the cost of reconstruction fidelity; OrtSAE reduces absorption without significant reconstruction penalty but may sacrifice feature diversity; ATM reduces absorption at the cost of training overhead.

**Pivot Trigger:** The rate-distortion theory and cross-domain characterization both fail to find interesting results, but there is still a clear community need for a rigorous benchmark comparison.

**Experiments (using pre-trained SAEs where available):**
1. Evaluate Matryoshka (pre-trained, available via SAEBench) vs. TopK vs. JumpReLU on absorption + RAVEL + sparse probing metrics using SAEBench evaluation code. Models: Gemma Scope SAEs on Gemma 2 2B. (2–3 hours)
2. Evaluate OrtSAE on the same model/layer if weights are available. Compare all methods on: absorption rate (Chanin et al.), RAVEL disentanglement, sparse probing F1, reconstruction CE loss delta, feature sensitivity (Tian et al.).
3. Test whether absorption reductions generalize: if Method A reduces first-letter absorption, does it also reduce absorption on entity-type hierarchies?

**Novelty:** While SAEBench provides some architecture comparisons, it does not include OrtSAE or masked regularization, does not compare on cross-domain absorption, and does not test whether absorption improvements transfer across hierarchy types.

**Risk:** Incremental without conceptual insight. Acceptance at top venues would require theoretical framing (e.g., "what makes each method work?"), which brings back elements of the rate-distortion theory. Best used as a supporting study within a larger paper.

---

## Backup 3: Deconfounding Measurement — Seed Dependence and Ensemble Methods

**Title:** Rethinking Feature Absorption: Deconfounding Measurement, Quantifying Seed-Dependence, and Ensemble Methods for Robust SAE Features

**Core Idea:** Inspired by the Contrarian perspective, this backup tests whether measured absorption rates are inflated by confounds: (1) L0 misspecification (feature hedging masquerading as absorption), (2) seed-dependent decomposition artifacts, and (3) evaluation on artificial hierarchies that may not reflect the model's computational structure.

**Key Hypotheses:**
- H1: At least 30% of measured absorption is attributable to L0 misspecification rather than true sparsity-driven absorption. Measuring absorption at the correct L0 (per "Sparse but Wrong" proxy) will reduce the measured rate by ≥ 30%.
- H2: The seed-invariant absorption rate (features absorbed in ≥ 80% of seeds, controlling for configuration) is less than 50% of the absorption rate on any single seed.
- H3: An ensemble of K ≥ 3 SAEs (same architecture, different seeds), where a feature is "active" if it fires in ANY member, achieves lower absorption than any individual SAE.

**Pivot Trigger:** The main proposal's empirical results are weaker than expected but the methodological concerns raised by the Contrarian are confirmed. The paper becomes a rigorous measurement standards contribution.

**Both outcomes are publishable:** If H1 and H2 are falsified (L0 correction has negligible effect and absorption is highly seed-consistent), the paper confirms that absorption is real and robust — equally valuable for the community.

**Risk:** If absorption survives all deconfounding, this becomes a replication/validation paper rather than a challenge paper. Frame the analysis prospectively: "We examined whether absorption is an artifact of measurement, and found it is not" is still a strong contribution.

---

## Backup 4: Feature Anchoring for Real LLM Absorption Reduction

**Title:** Does Feature Anchoring Work? Evaluating the Unified SDL Theory's Proposed Remedy on Real LLMs

**Core Idea:** The unified SDL theory (arXiv:2512.05534) proposes "feature anchoring" as a principled solution to absorption and identifiability failure in SAEs. Feature anchoring has been validated on synthetic benchmarks and on real neural representations (as of the April 2026 literature check). This backup evaluates whether anchoring reduces absorption specifically on the Chanin et al. absorption metric on real LLMs, and whether it interacts with existing architectures (Matryoshka, OrtSAE).

**Key Hypothesis:** Feature anchoring reduces absorption rate by at least 30% relative to unanchored TopK baseline on Gemma 2 2B. The reduction is largest for feature pairs with high decoder cosine similarity (near the absorption threshold from the rate-distortion theory).

**Pivot Trigger:** The main proposal's theoretical work demonstrates that absorption is rate-distortion optimal, but the empirical validation of the ASI or cross-domain characterization is weaker than expected. Feature anchoring provides a natural applied extension.

**Note:** Tang et al. (2512.05534) has already validated anchoring broadly; our contribution would be specifically testing it on the absorption metric and interaction with cross-domain hierarchies.

**Experiments:**
1. Implement feature anchoring in SAELens on top of a standard TopK SAE. Train on Gemma 2 2B layer 12 for 3000–5000 steps on A100.
2. Compare anchored vs. unanchored SAE absorption rates at matched L0.
3. Use the rate-distortion threshold from the main proposal to predict which feature pairs benefit most from anchoring.
4. Test interaction with OrtSAE: does combining anchoring + orthogonality penalty provide compounded absorption reduction?

**Risk:** Requires SAE training (not training-free), increasing compute budget. Tang et al. already did broad anchoring validation; the residual novelty is narrow.

---

## Candidate Status Summary

| Candidate | Status | Priority | Trigger |
|---|---|---|---|
| Rate-distortion + cross-domain + ASI + phase transition | front_runner | Primary | — |
| Absorption-as-representational-diagnostic | backup | First fallback | Theory AUROC < 0.65 AND cross-domain null |
| Systematic mitigation benchmark | backup | Second fallback | Both theory and cross-domain fail |
| Deconfounding measurement / seed dependence | backup | Parallel option | If contrarian concerns deserve standalone paper |
| Feature anchoring evaluation | backup | Third fallback | Theory succeeds but needs applied angle |

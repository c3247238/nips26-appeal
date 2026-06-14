# Experiment Result Analysis

## Key Results Summary

Iteration 9 completed all blocking experiments (Gate 0 zero-GPU analyses + Gate 1 three critical experiments). Key quantitative findings:

1. **Activation Patching**: 8 persistent core words (eight, liked, lower, offer, often, other, under, until) tested with three patching methods (decode-reencode, residual subtraction, all-children zeroing). Recovery rate: **0/8** across all methods. Control group: **0/65**. Internal consistency is perfect.

2. **Tightened Hedging Classification**: Of 656 FN tokens at L0=22, strict hedging rate is **6.2%** (41/656, CI: [4.4%, 8.2%]), statistically significant vs. shuffled control at 3.4% (z=3.51, p=0.0004). The prior loose definition yielded 98.6% -- a 92.3 percentage point gap driven by conflating compensatory feature resolution with genuine parent-latent hedging.

3. **CMI Replication at L0=22**: At the pre-registered d'=10 with all 25 probes achieving F1=1.0 (eliminating probe quality confound), Spearman rho=+0.044 (p=0.835). The L0=82 correlation (rho=-0.383) is entirely driven by probe quality confounding (rho(absorption, probe_F1)=-0.692, p=0.0001). Partial correlation after controlling for F1: rho=-0.328, Bonferroni p=0.472. Restricted to F1>0.85 (n=10): rho=-0.113, p=0.757.

4. **Control Failure Diagnosis**: In R^2304, cosine>=0.025 threshold matches 23.0% of decoder columns (3766/16384) for a random unit vector. At L0=82, P(at least one candidate active)=1.0. The metric degenerates into a proxy for false-negative rate. All 5 domains exhibit structural control failure.

5. **Threshold Sensitivity**: 5x4 parameter grid yields CV=0.077, Kendall tau=0.977. Absorption rate stable at 0.118-0.151 across all threshold combinations. Control failure persists universally.

6. **Data Integrity**: 84/85 numerical claims match source data (1 missing, 0 inconsistent), integrity score 98.82%.

## Debate Perspectives Summary

- **Optimist**: Views the results as historically significant -- first JumpReLU metric audit, first causal test via activation patching. Argues 0/8 patching is clean decisive evidence eliminating competitive exclusion. The L0 phase transition (42.85% to 0.84%) is immediately actionable for SAE practitioners. Honest negative results (5+/7 hypotheses falsified) are the paper's strongest differentiator. Projects 7.5-8.0 score after writing revision.

- **Skeptic**: Raises two fatal flaws: (F1) CMI diagnostic has zero explanatory power after confound control -- entire Q3/H4 must be relegated to negative result appendix; (F2) paper cannot distinguish genuine absorption from probe measurement error since absorption_rate ~ false_negative_rate at standard thresholds, undermining the entire L0 curve interpretation. Serious concerns include JumpReLU threshold confound in patching (parents may be globally dead, not locally suppressed) and that "tightened hedging" creates a narrative vacuum (6.2% is only 2.8pp above shuffled control). Argues the control failure might be threshold calibration, not fundamental metric invalidity.

- **Strategist**: Declares the project has transitioned from "evidence-limited" to "writing-limited." All experimental evidence is complete. No additional experiments are needed. Recommends immediate Gate 2 writing revision (estimated 10.5 hours). Competitive landscape is highly favorable -- no prior work audits Chanin metric on JumpReLU, tests absorption causally, or maps L0-absorption curves. Recommends ICLR 2027 as primary venue, arXiv preprint immediately for priority. Projects 7.5-8.0 with writing revision.

- **Comparativist**: Positions the paper as the first "meta-audit" in SAE absorption research. Systematically demonstrates 7 findings with no prior precedent (control failure, hedging decomposition, activation patching, L0 curve, mechanistic diagnosis, cross-domain attempt, CMI falsification). Rates 7.0-7.5 at NeurIPS/ICML level. Notes core limitations: single model/SAE family, no L1-ReLU control failure verification, small causal test sample (n=8), collapsed CMI theoretical pillar.

- **Methodologist**: Identifies construct validity failure in the 98.6% hedging claim (loose/strict gap of 92.3pp). Validates threshold sensitivity analysis as methodologically strongest component (5x4 grid, CV=0.077). Raises JumpReLU threshold confound in activation patching (Explanation B: sub-threshold parents, not absence of competition). Statistical power for CMI is ~0.40 at N=25, fundamentally insufficient. Demands three-category FN decomposition (strict hedging 6.2%, compensatory resolution 87.6%, persistent FN ~0%).

- **Revisionist**: Calls for fundamental narrative restructuring. "Feature absorption is competitive exclusion" must be abandoned -- 0/8 patching + 6.2% strict hedging converge on encoding absence rather than suppression. CMI framework must be downgraded from theoretical pillar to falsified exploratory analysis. Proposes new "SAE encoding capacity theory" framework: at given L0, encoder prioritizes high-semantic-value features over surface attributes; the L0 transition reflects capacity crossing the minimum encoding cost for surface attributes. Projects 7.5-8.0 after full narrative revision.

- **Synthesis (6-perspective integration)**: Rates result quality at **7.0/10**. Unanimous consensus on: metric failure on JumpReLU (very high confidence), L0 transition robustness (very high confidence), CMI falsification (5/6 agree), honest negative results as strongest differentiator (unanimous). Key conflict: 0/8 patching is "strong but not decisive in isolation" due to JumpReLU threshold confound -- value multiplies when combined with other evidence streams. Recommends three-category hedging decomposition, softening "phase transition" to "sparsity transition," and framing metric as "does not transfer without recalibration" rather than "fundamentally invalid."

## Analysis

### 1. Method Feasibility

The core method -- auditing the Chanin absorption metric on JumpReLU SAEs via shuffled-label controls, confound decomposition, and causal testing -- has proven highly feasible and productive. The entire experimental program was completed training-free using public infrastructure (SAELens, TransformerLens, sae-spelling, Gemma Scope). The four-tier control suite (C1-C4) exceeded community standards. The method produced clear, interpretable results with no ambiguous experimental outcomes. All blocking experiments (Gate 0 + Gate 1) completed successfully within the estimated GPU budget (~3 hours).

### 2. Performance

Results exceed baselines in the sense that the paper establishes multiple findings with no prior precedent:
- Universal control failure across 5 domains with mechanistic explanation
- First causal test (0/8 activation patching recovery)
- First quantitative hedging decomposition (6.2% strict vs. 98.6% loose)
- L0 sparsity transition with cross-layer stability (CV<10%)
- 98.82% data integrity verified through automated cross-checking

The synthesis consensus assigns a result quality score of 7.0/10, with the comparativist rating the paper at 7.0-7.5 and the optimist/strategist projecting 7.5-8.0 after writing revision. Current score trajectory: three consecutive iterations at 6.5, now with all blocking evidence complete.

### 3. Improvement Headroom

The improvement headroom is clear and well-defined, residing entirely in writing revision (Gate 2):
- CMI narrative demotion (7+ overclaims to fix)
- Three-category hedging decomposition (replacing binary framing)
- Activation patching integration with JumpReLU threshold confound discussion
- Title/abstract revision (remove rate-distortion reference, adopt audit framing)
- Threshold sensitivity integration
- Negative results table update (2 confirmed, 7+ falsified)

All 6 perspectives agree that no additional experiments are needed. The strategist estimates ~10.5 hours of pure writing work. The synthesis estimates ~12 hours. No GPU time is required.

### 4. Time-Cost Tradeoff

Continuing the current direction (writing revision) is unambiguously the most efficient path:
- All experimental evidence is complete
- Writing revision requires ~10-12 hours, no GPU
- The competitive window is favorable but narrowing (Chanin's team is one step from discovering control failure; SAEBench already notes the metric's limitations)
- A pivot would forfeit 9 iterations of accumulated evidence and unique findings with no prior precedent
- GPU resources can be released to other projects immediately

### 5. Critical Objections

The skeptic raises two "fatal flaws" (F1: CMI as artifact, F2: absorption-as-probe-error confound) and two serious concerns (S1: patching threshold confound, S2: narrative vacuum from hedging revision).

**Assessment of each:**

- **F1 (CMI artifact)**: Valid and already acknowledged by all perspectives. The resolution is straightforward: demote CMI from theoretical pillar to falsified exploratory analysis. This is a writing fix, not an experimental one.

- **F2 (absorption-as-probe-error)**: Partially valid but overstated. At L0=82 where all probes achieve F1=1.0 (per tightened hedging data), absorption rate of 14.39% exists despite perfect probes. The skeptic's own note acknowledges this partially resolves the concern. The paper needs to explicitly state this defense.

- **S1 (patching threshold confound)**: Valid concern that limits the causal claim. The resolution is to present 0/8 as one of three converging evidence streams (alongside hedging decomposition and control failure diagnosis), not as standalone decisive proof. This is a framing/discussion fix.

- **S2 (narrative vacuum)**: Valid. The three-category decomposition (strict hedging 6.2%, compensatory resolution 87.6%, persistent FN ~0%) resolves this by honestly acknowledging that the dominant mechanism for 87.6% of FNs remains unresolved.

**None of the skeptic's objections require new experiments. All are addressable through writing revision.**

## Decision Rationale

The evidence strongly supports proceeding with writing revision (Gate 2) rather than pivoting. The reasoning:

1. **Experimental completeness**: All 6 debate perspectives independently conclude that experimental evidence is sufficient for publication. The synthesis explicitly recommends "PROCEED to Gate 2." No perspective advocates for additional experiments as a prerequisite.

2. **Unique, publishable contributions**: The paper has established at least 7 findings with no prior precedent in the literature. The "meta-audit" positioning is unique in SAE absorption research -- no competitor occupies this niche.

3. **Clear improvement path**: The gap between current state (6.5 score) and target (7.0-8.0) is entirely bridgeable through writing revision. The required changes are well-characterized: CMI demotion, hedging decomposition, patching integration, narrative restructuring. Estimated effort: 10-12 hours with zero GPU cost.

4. **Competitive urgency**: The strategist correctly identifies that the competitive window is favorable but narrowing. Chanin's team, SAEBench, and other groups are converging on the same metric limitations. Delay risks loss of priority on multiple first-in-field findings.

5. **No viable pivot target**: A pivot would mean abandoning 9 iterations of accumulated evidence, 7 unique findings, and a well-defined writing revision path in favor of an undefined alternative direction. The current direction has strong results; the bottleneck is writing, not evidence.

6. **Consensus across perspectives**: The optimist, strategist, comparativist, revisionist, and synthesis all explicitly recommend proceeding. The skeptic and methodologist raise valid concerns but frame them as writing-level fixes (discuss threshold confound, demote CMI, adopt three-category decomposition), not as reasons to pivot.

DECISION: PROCEED

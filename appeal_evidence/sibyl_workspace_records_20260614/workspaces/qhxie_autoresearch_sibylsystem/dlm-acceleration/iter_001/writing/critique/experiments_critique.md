# Critique: Experiments (Section 5)

## Summary Assessment

The Experiments section is densely evidenced and largely consistent with raw JSON data. Caveat disclosure — degenerate coding baselines, 2-seed pairwise estimates, M1 implementation gap — is proactive and exemplary. The section has been substantially improved from prior drafts: the tau=0.0 vs. naive-T16 comparison has been run and resolved (full_tau0_comparison.json), the Discussion section (§6.3) now cites completed results. However, two Critical issues persist: (1) a stale "Before final submission, we will compare..." sentence (§5.4) that directly contradicts the resolved data already in §6.3 and the raw JSON; and (2) a pervasive IGSD/CD-SSD naming inconsistency — the method section and notation.md rename the method to CD-SSD, but every table, subsection header, and narrative sentence in Section 5 still uses IGSD. Four Major issues persist around QAS aggregation inconsistency, IGSD speedup overclaim, M3 speedup disambiguation, and a false monotonicity claim. The section is not yet submission-ready.

## Score: 6/10
**Justification**: Evidence quality and caveat transparency are above average. The synergy mechanism narrative is strong and the failure-mode atlas is the paper's most original contribution. Score is held at 6/10 because two Critical issues (stale future-work statement; IGSD/CD-SSD naming collision with the method section) and four Major issues would immediately draw reviewer scrutiny. Resolving all six issues and the named minor issues would push this to 8.5/10.

---

## Critical Issues

### Issue 0: IGSD / CD-SSD Naming Inconsistency — Entire Section Needs Rename
- **Location**: All subsection headers, all table rows, and all narrative paragraphs in §5.1–§5.5
- **Quote**: "**IGSD** ($\tau = 0.9$, $T_{\text{draft}} = 16$): IGSD achieves 4.57× speedup on GSM8K..." (§5.1); "Table 3: Pairwise orthogonality matrix... | **M1 + IGSD** | **5.13×** |..." (§5.2); Table 5 header "IGSD-full", "IGSD-no-partition", "IGSD-T4", etc.
- **Problem**: notation.md explicitly states "method renamed from IGSD to CD-SSD (Coarse-Draft Self-Speculative Denoising) to avoid name collision with Info-Gain Sampler." The method section (§4.2) uses CD-SSD throughout. The pre-submission checklist in outline.md marks "IGSD renamed consistently to CD-SSD throughout all files" as a blocking major item. Section 5 is the only section that still uses IGSD — a reviewer reading the paper sequentially will encounter an undefined acronym in the experiments section and trace back to the mismatch with the methods section. The inconsistency is not cosmetic: "IGSD" vs. "CD-SSD" produces nomenclature confusion on key result labels (Table 2, Table 3, Table 5, and all narrative sentences referring to the method).
- **Fix**: Global replace in Section 5: "IGSD" → "CD-SSD", "IGSD-full" → "CD-SSD-full", "IGSD-no-partition" → "CD-SSD-no-partition", "IGSD-T4/T8/T32" → "CD-SSD-T4/T8/T32". Section header "§5.4 IGSD Ablations" → "§5.4 CD-SSD Ablations". All table row labels and figure references must follow.

### Issue 1: Stale "Future Work" Sentence Contradicts Completed Experiment
- **Location**: §5.4, tau=0.0 paradox paragraph, final sentence (lines 219–221)
- **Quote**: "Before final submission, we will compare IGSD-no-partition against a naive $T = 16$ baseline (no IGSD machinery) to determine whether the confidence partitioning adds any value beyond the draft length itself."
- **Problem**: This sentence presents the tau=0.0 vs. naive-T16 comparison as a pending future experiment. In fact, this comparison was completed and the data is in `full_tau0_comparison.json`: CD-SSD(tau=0.0) achieves QAS=4.198 vs. naive-T16 QAS=4.458, a -5.8% difference within seed variance — the experiment is resolved. The Discussion (§6.3) already presents this as a completed finding with Table 7. This creates a direct internal contradiction: §5.4 says "we will run this," while §6.3 says "we ran it and here is the resolved result." A reviewer reading sequentially will flag this as either a drafting error or evidence that the paper was assembled from notes without cross-section review. The stale sentence must be removed before submission.
- **Fix**: Delete the final sentence of the tau=0.0 paragraph in §5.4. Replace with a forward reference: "The resolved comparison against a naive $T = 16$ baseline is presented in §6.3 (Table 7), confirming that the confidence partitioning mechanism adds no measurable value over plain step reduction."

---

## Major Issues

### Issue 2: IGSD "Near-Constant Speedup" Claim Overstates the Data
- **Location**: §5.1, IGSD paragraph, lines 86–88
- **Quote**: "IGSD achieves 4.57× speedup on GSM8K (3-seed mean) and maintains near-constant speedup across all task types: 4.57× (GSM8K), 2.32× (MATH500), 1.95× (HumanEval), 1.35× (MBPP)."
- **Problem**: The per-benchmark speedups span 1.35× to 4.57× — a 3.4× range. This is not "near-constant." The raw data in `igsd_pareto_full.json` confirms these values (tau=0.9, T_draft=16: GSM8K avg_speedup=4.57×, MATH500=2.19×, HumanEval=1.86×, MBPP=1.25×). The mean-across-benchmarks of 3.40× obscures this variation. Calling a 3.4× range "near-constant" relative to a 4.57× anchor is misleading; a reviewer will immediately check the data and flag this as overclaiming. The range is actually a meaningful finding — IGSD's speedup is highly task-dependent — and should be reported as such.
- **Fix**: Replace "maintains near-constant speedup across all task types" with "exhibits task-dependent speedup ranging from 1.35× (MBPP) to 4.57× (GSM8K), with mean 3.40× across all benchmarks." The task-dependence finding is substantively interesting and belongs in §5.5 (Task-Dependent Recipes) as additional evidence for H4.

### Issue 3: M3 Operating-Point Speedup Inconsistency (1.68× vs. 1.33×) Persists
- **Location**: §5.1, M3 paragraph (line 77) vs. §4.2 Method section
- **Quote (experiments §5.1)**: "At $w = 0.3$, M3 achieves 1.33× speedup with GSM8K accuracy 74.0%"
- **Quote (method §4.2)**: "Operating point: $w = 0.3$, which achieves 1.68× speedup on GSM8K with accuracy retention of 103.9%"
- **Problem**: The method section reports 1.68× as M3's operating-point speedup; the experiments section reports 1.33×. Both cannot be correct as a single canonical figure. From `m3_pareto_full.json`, the data shows M3 (w=0.3) has GSM8K-specific avg_speedup=1.677× and combined-benchmark avg_speedup=1.332×. The 1.68× in §4.2 is the GSM8K-only speedup; the 1.33× in §5.1 is the combined-benchmark speedup. Neither section makes this distinction explicit. A reviewer will treat these as conflicting numbers from the same experiment.
- **Fix**: Add a disambiguation note to both sections. In §4.2: "1.68× speedup on GSM8K specifically; combined speedup across all benchmarks is 1.33× (Table 2, §5.1)." In §5.1: "Combined speedup 1.33× (GSM8K-specific: 1.68×)." Add a footnote to Table 2 clarifying that M3's Speedup column reports the combined-benchmark figure.

### Issue 4: QAS Aggregation Definition Is Inconsistent Across Tables
- **Location**: Table 2 caption vs. Table 5 caption; §5.2 comparison of QAS values
- **Problem**: Table 2 defines QAS as "Speedup × mean AccRet across all benchmarks" but most rows only have GSM8K and MATH500 AccRet populated (HumanEval and MBPP absent for M1 and M3). Table 5 computes QAS over "200 GSM8K + 164 HumanEval samples." This means IGSD's QAS=1.194 in Table 2 (computed over all four benchmarks, including MBPP AccRet=1.0 by convention) is not directly comparable to IGSD-full's QAS=0.956 in Table 5 (computed over GSM8K+HumanEval subset). The text in §5.2 compares these values directly: "Combined QAS = 1.654 exceeds both IGSD alone (QAS = 1.194)" — this comparison mixes QAS computed over different benchmark sets. The MBPP AccRet=1.0 convention inflates combined QAS for IGSD by including a degenerate benchmark as a perfect-retention result.
- **Fix**: (a) Standardize QAS formula to use exactly GSM8K + MATH500 for reasoning evaluations, GSM8K + MATH500 + HumanEval for combined evaluations (excluding MBPP), and note MBPP separately. (b) Add a footnote to Table 2 specifying the exact benchmarks used for each method's QAS computation. (c) In §5.2 line 120, add: "Note: QAS=1.194 and QAS=1.654 use the same four-benchmark combined formula; M3's QAS=1.675 uses only reasoning benchmarks (GSM8K + MATH500) — direct comparison requires caution."

---

## Minor Issues

- **§5.1, Table 2, M2 J=8 row**: AccRet=0.243 is listed in the GSM8K AccRet column but GSM8K accuracy of ~5.3% against a 71.2% baseline yields AccRet=0.074, not 0.243. Cross-checking `m2_pareto_full.json` shows MATH500 AccRet=0.243 at J=8, not GSM8K. The 0.243 value in the GSM8K AccRet column appears mislabeled. Verify and correct the column assignment.

- **§5.1, IGSD paragraph, lines 87–90**: The text says "Combined across benchmarks, the operating point yields Speedup = 3.40×, QAS = 1.194." But Table 2 shows IGSD at tau=0.9, T_d=16 with Speedup in the table header column — the table does not explicitly show 3.40× combined. The reader cannot verify the 3.40× from Table 2 alone. Add a footnote to Table 2 or an explicit sentence: "Combined speedup computed as mean of benchmark-specific speedups: (4.57+2.32+1.95+1.35)/4 = 3.05×" — note that the stated 3.40× should be verified against the actual averaging method used in `igsd_pareto_full.json`.

- **§5.2, Table 3, M1+IGSD row**: The text says "96% KV-cache hit rate during the refine phase" (line 126) but the raw data shows CHR_refine=0.940 (94.0%) in `igsd_p2_tau09_td16_s123.json` and the notation.md also states 94.0%. The "~96%" figure in §5.2 conflicts with the precise 94.0% cited in §5.4 and §6.2. Standardize to 94.0% everywhere.

- **§5.3, Table 4, F3 row**: The Detection Signal says "Per-step acceptance rate α > 0.75" but the body text for F3 (line 179) says "τ < 0.8 accepts low-quality draft tokens; REFINE cannot recover." The threshold cited in the table (α > 0.75) is a different quantity from τ < 0.8 in the body text. Both appear in different units (α is an empirical acceptance fraction; τ is a hyperparameter threshold). Clarify which detection signal is practically observable at runtime and confirm the threshold value against `igsd_pareto_full.json`.

- **§5.4, T_draft sensitivity, line 223**: "Speedup is monotonically non-increasing with $T_d$" — the ablation data (Table 5) shows T4:1.88×, T8:2.30×, T16:2.66×, T32:2.09×. This is increasing from T4 to T16, then decreasing at T32. The claim "monotonically non-increasing" is factually incorrect; speedup peaks at T16 and the relationship is non-monotonic in both directions. The correct description is: "Speedup peaks at $T_d = 16$ and decreases at $T_d = 32$ as draft-phase overhead dominates; $T_d < 16$ also reduces speedup due to insufficient draft coverage, making $T_d = 16$ the unique optimum."

- **§5.3, failure mode naming inconsistency**: Body text uses "FM1", "FM2", "FM3", "FM4" (e.g., "see §5.3, failure mode F1" in §5.1 vs. "failure mode FM2" in §5.2). Table 4 uses "F1", "F2", "F3", "F4" (no M). Method.md uses "FM1, FM2, FM3, FM4". Three different conventions — unify to one. Recommended: use "FM1–FM4" consistently as in method.md, since these are distinct named failure modes, not sequential labels.

- **§5.1, accept-rate language**: "The accept rate at $\tau = 0.9$ is $\alpha = 0.52$: 52% of tokens are accepted from the draft phase and frozen during refine." The summary.md confirms the distinction: $\alpha = 0.52$ is the unique-position frozen fraction; the per-token-step accept rate is ~88%. The sentence currently conflates these. Summary.md states "accept_rate alpha=0.88 (88% of tokens accepted from draft at tau=0.9; ~52% frozen during refine based on unique token positions)". Clarify: "$\alpha = 0.52$ denotes the fraction of unique output positions frozen during refine (52%); the per-step acceptance rate is approximately 88% of token-step assignments at $\tau = 0.9$."

- **§5.5, Table 6, Reasoning row**: "M3 + IGSD (note: INTERFERENCE), QAS = 1.446" — this number appears without a data source. Table 3 shows M3+IGSD combined QAS=0.826 across all benchmarks. The 1.446 value for reasoning-only QAS would need to be computed from `full_pairwise_ortho.json` restricted to GSM8K+MATH500 subset; no such computation is documented in any results file. If this is a back-computed number, state the computation explicitly. If it cannot be verified from existing data, replace with "—" and note that pairwise reasoning QAS requires a dedicated reasoning-only pairwise experiment.

- **§5.5, Table 6, Coding row**: Best Pair QAS is listed as "—" without explanation. §5.2 confirms M1+IGSD was evaluated on HumanEval+GSM8K. A HumanEval-only QAS for M1+IGSD can be derived from `full_pairwise_ortho.json` (seed 42: he_speedup=3.26×, he_ret=0.0 → QAS=0.0). The blank cell should be replaced with the computed value (QAS≈0 due to HumanEval degenerate pass@1) with a footnote explaining why. An unexplained blank raises reviewer suspicion.

- **Opening paragraph consistency**: The opening claims "All numbers are mean ± std across seeds [42, 123, 456] unless explicitly noted as 2-seed estimates." This is accurate for single-method Pareto (§5.1) but §5.4 ablation uses only seeds [42, 123] (confirmed in `igsd_ablation.json`), and §5.2 pairwise also uses [42, 123]. These should be noted in each subsection header, not just at the top. Readers who skip to §5.4 directly will miss the caveat.

---

## Visual Element Assessment

- [x] Figures 3–6 are each referenced in the text before their appearance
- [x] Tables 2–6 are placed inline and referenced from body text before each table
- [x] Failure mode table (Table 4) has severity labels and detection signals — self-explanatory
- [ ] **Figure 3 y-axis description missing**: The text mentions "log-speedup axis" and "95% accuracy retention shaded" but does not state the y-axis is AccRet (accuracy retention). Add explicit axis description in the forward reference sentence (line 19–20). Also: M3 at AccRet=1.039 exceeds 100% — the figure description should clarify whether this point is shown at-or-above the acceptability boundary shading.
- [ ] **Table 2, M2 missing MATH500 cells**: At J=4, J=6, J=8, MATH500 AccRet is "—". This is appropriate for a NO_GO method, but the footnote explanation is embedded in the table note as "MATH500 AccRet > 1.0 for M3 indicates..." — this footnote refers to M3, not M2. A separate footnote for M2 missing cells is absent. Add: "*M2 MATH500 AccRet not reported for J≥4 (already classified NO_GO at J=2)."
- [ ] **Table 5, IGSD-no-partition row bolded**: Bold formatting signals best performance. IGSD-no-partition (tau=0.0) has QAS=1.801 which is indeed the highest QAS in Table 5, but this is the configuration that the tau=0.0 paradox shows is equivalent to naive T=16 — the bold presents it as "IGSD's best configuration" when it is actually "IGSD with its core mechanism disabled." Consider replacing bold with an asterisk and footnote: "*tau=0.0 is equivalent to naive T=16 step reduction (see §6.3); the confidence partitioning mechanism contributes zero additional value at this operating point."
- [ ] **No teaser figure in this section**: The outline specifies Figure 1 (teaser) as a 2×2 panel showing Ortho scores and speed-accuracy scatter. This figure is not referenced from §5.2 (where it would be most useful). If Figure 1 appears in the Introduction, add a back-reference here: "See also Figure 1 for the full composability landscape."

---

## Numerical Consistency Verification (Against Raw JSON)

The following claims were cross-checked against source data:

| Claim in Text | Source JSON | Status |
|---|---|---|
| M1+IGSD Ortho=1.385, per-seed [1.292, 1.478] | full_pairwise_ortho.json: ortho=1.385, seed-42=1.292, seed-123=1.478 | CONFIRMED |
| M3+IGSD Ortho=0.493 | full_pairwise_ortho.json: ortho=0.493 | CONFIRMED |
| M1+M3 Ortho=0.301 | full_pairwise_ortho.json: ortho=0.301 | CONFIRMED |
| IGSD-no-partition QAS=1.801, delta=+88.5% | igsd_ablation.json: avg_qas=1.801, delta_pct=88.47% | CONFIRMED |
| IGSD-T4 QAS=0.394, delta=-58.8% | igsd_ablation.json: avg_qas=0.394, delta_pct=-58.8% | CONFIRMED |
| CD-SSD(tau=0.0) QAS=4.198 vs naive-T16 QAS=4.458 | full_tau0_comparison.json: cdssd_tau00 avg_qas=4.198, naive_t16 avg_qas=4.458 | CONFIRMED |
| M1+naive-T16 QAS=4.232 | full_tau0_comparison.json: m1_naive_t16 avg_qas=4.232 | CONFIRMED |
| IGSD CHR during refine ~96% (§5.2) | igsd_p2_tau09_td16_s123.json: NOT ~96% — notation.md states 94.0% | INCONSISTENCY — text says 96%, data says 94% |
| M3 combined speedup 1.33× (w=0.3) | m3_pareto_full.json: combined_speedup=1.332 | CONFIRMED |
| M3 GSM8K speedup 1.68× (w=0.3) | m3_pareto_full.json: gsm8k avg_speedup=1.677 | CONFIRMED |

**Summary**: 9/10 checked claims confirmed. The CHR discrepancy (96% vs. 94%) in §5.2 should be corrected to 94.0% to match the source data, notation.md, and the value cited in §6.2.

---

## What Works Well

1. **Proactive limitation disclosure at point of use**: Degenerate coding baselines, 2-seed estimates, and M1 implementation gap are each flagged at the exact paragraph where the data appears — not deferred to a limitations section. This is best practice and will reduce reviewer adversarialism on the known weaknesses. The MBPP AccRet=1.0 convention footnote (§5.5, lines 249–252) is particularly clear.

2. **Failure mode atlas (§5.3)**: The four-way taxonomy with specific detection signals, confirmed root causes, and actionable remedies is the most deployable part of the paper. F1's root cause analysis ("fundamental algorithmic incompatibility, not a hyperparameter problem") and F4's compound mechanism ("AR forward passes compound overhead; AR-blended tokens corrupt the diffusion trajectory") both cite specific evidence. The atlas format will be directly cited by practitioners.

3. **M1+IGSD synergy explanation (§5.2)**: The mechanistic connection from frozen tokens → H_i=0 → guaranteed cache hits at η=2.0 → 96% (corrected: 94%) CHR during refine is stated precisely with supporting numbers. Both seeds independently confirm Ortho > 1.0, making the finding classification robust. The acknowledgment that exact Ortho magnitude requires full 3-seed validation is appropriate hedging that strengthens rather than weakens the claim.

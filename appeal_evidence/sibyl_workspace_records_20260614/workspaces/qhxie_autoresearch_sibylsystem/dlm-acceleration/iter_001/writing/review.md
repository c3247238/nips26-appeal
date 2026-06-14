# Writing Quality Review

## Summary

ComposeAccel presents the first systematic pairwise composability study of training-free MDM acceleration methods. The paper defines a formal Orthogonality metric and Quality-Adjusted Speedup (QAS), finds that exactly one of three feasible method pairs (M1 + CD-SSD, Ortho = 1.385, 5.13× combined speedup, QAS = 1.654) achieves super-multiplicative synergy, provides a mechanistic explanation grounded in frozen-token entropy collapse, and delivers a failure mode atlas with actionable detection signals. Relative to the previous review (SCORE: 6), this revision has resolved all three critical numerical issues: CHR is consistently ~94% throughout, Table 2's M1 Speedup column shows GSM8K-specific values (0.61×, 0.62×, 1.50×, 1.91×) with a clarifying caption, and CD-SSD QAS is now reported as 3.40 × 0.703 = 2.39 with no hidden penalty. Appendix B pseudocode is populated. The IGSD → CD-SSD rename is complete throughout paper.md. The paper is substantially publication-ready in its prose, structure, and quantitative integrity. Two blocking pre-submission items remain: Figure 2 is still missing, and 37 citation placeholders are unresolved.

---

## Detailed Assessment

### Structural Coherence: 8/10

The introduction's three-problem framing (deployment confusion, hidden conflicts, missing composability vehicle) maps cleanly onto the paper's three contributions. The roadmap paragraph at the end of the introduction accurately previews Sections 2–6. The section order — Methods → Experiments → Analysis → Related Work → Conclusion — is logical and transitions are well-motivated.

Section 4 is well-organized into four sub-problems (why binary, frozen-token mechanism, τ=0.0 implication, limitations), and the deployment guidance in §4.5 is now explicitly anchored to §4.1–4.3 via the bridge sentence added in revision.

One structural concern persists: the abstract mentions "four training-free acceleration families," and the revision correctly added "(of which one — adaptive step scheduling — receives an immediate NO_GO verdict and is excluded from pairwise experiments)." This is now clear. However, the Introduction's contribution list (item 1) still says "four families of training-free MDM acceleration methods" without the NO_GO clarification that was added to the abstract. A reader who skips the abstract and starts at the contributions list will encounter the same confusion. Add "(M2 excluded as NO_GO)" parenthetically in contribution item 1.

The scope and honest positioning paragraph (after the contribution list) is well-placed and preempts foreseeable reviewer objections. The paper correctly labels itself an analysis paper.

### Notation and Terminology Consistency: 8/10

The CD-SSD rename is complete in paper.md (verified by text search: zero instances of "IGSD" in paper.md). KV-cache is consistently hyphenated throughout. All banned terms from glossary.md are absent from paper.md. The "DLM" that appears in Table 1 ("Saber | DLM (code)") is quoting Saber's reported base model from the external paper — this is acceptable per glossary.md, which permits "DLM" when quoting external literature.

CHR notation is now consistent: $\text{CHR}_{\text{refine}}$ is used when referring to the refine-phase measurement in §2.2, §3.2, §4.2, §6.1, and the Figure 8 caption. The abstract uses the informal "~94%" rather than the formal symbol, which is appropriate for the abstract. No issues found.

One minor remaining deviation: notation.md defines $H_i = -\sum_{v} p_\theta(v \mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t)$ without a position subscript on the probability argument. The paper consistently uses the subscript form $p_\theta(v \mid \tilde{x}_t)_i$, which is clearer. The deviation is in notation.md, not in the paper — the paper's usage is correct and consistent.

### Claim-Evidence Integrity: 8/10

The three critical numerical issues from the previous review have been resolved and verified against raw data:

1. **CHR ~94%**: All instances in the paper now correctly state "~94%" or "0.940." The raw data (`igsd_p2_tau09_td16_s123.json`, `igsd_p2_tau09_td16_s456.json`) confirms `avg_kv_hit_rate_refine = 0.940` for GSM8K in both seeds. The quantitative consistency check in §4.2 has been corrected from "10× reduction" to "~6.7× reduction" (from 40% to 6% of positions needing recompute at CHR=94%). This is numerically correct: (1 − 0.60) / (1 − 0.94) = 0.40/0.06 = 6.67×.

2. **Table 2 M1 Speedup column**: All M1 rows now show GSM8K-specific speedup (η=0.5: 0.61×, η=1.0: 0.62×, η=2.0: 1.50×, η=3.0: 1.91×). Raw data confirms: η=2.0 GSM8K speedup_mean = 1.4959 ≈ 1.50× — correct. The table caption clarifies that QAS (combined) uses the combined speedup (1.38×) while the Speedup column shows GSM8K-specific values.

3. **CD-SSD QAS**: Table 2 now reports QAS = 3.40 × 0.703 = 2.39 using the standard formula. The footnote explains the sample-weighted combined AccRet = 70.3%. This is internally consistent.

One new potential consistency issue to flag: Table 2 lists M1 at η=2.0 as GSM8K AccRet = 0.550, and the body text (§3.1) says "GSM8K accuracy collapses to 14.5% (η = 3.0)." The raw data shows η=3.0 GSM8K accuracy_mean = 0.10337 (10.3%), giving AccRet = 0.145. But the text says "14.5%" which is AccRet ×100 — this is just a format inconsistency (percent notation vs. decimal notation) rather than a factual error, but the sentence should read "10.3%" to match Table 2's accuracy column ("10.3%"). **Minor issue.**

The M2 step_starvation numbers in the failure mode atlas (Table 4, §3.3) report GSM8K AccRet = 0.544 at J=2 and 0.130 at J=4 — these match the Table 2 values. The F1 description in Table 4 states "AccRet: 0.544 at J≥2, 0.130 at J=4" — consistent.

One note on scope accuracy: §3.4 Table 5 footnote correctly states the τ=0.0 vs. naive-T16 comparison yields −5.8% QAS within noise. The tau=0.0 comparison is presented as confirming CD-SSD's confidence-scoring mechanism adds no standalone value. This is a clean, honest ablation finding.

### Visual Communication: 7/10

Six of eight figures exist and are placed correctly before their first discussion reference. All six tables are inline with clear headers, footnotes, and bolded operating points.

**Blocking issue:**

1. **Figure 2 (CD-SSD architecture diagram) is still missing.** The paper references `figures/fig2_igsd_architecture.pdf` but the file does not exist (confirmed). The spec at `writing/figures/fig2_igsd_architecture_desc.md` is complete and detailed. Figure 2 is the primary visual for the paper's novel method and is required before any submission. The in-text reference in §2.2 ("As illustrated in Figure 2, CD-SSD decomposes...") creates a broken reference in the compiled PDF. This is the single most important remaining action item.

**Items no longer blocking (resolved since previous review):**

2. The CHR value in Figure 8 caption now correctly reads "~94%" and "measured: 0.940."
3. Appendix B contains full CD-SSD pseudocode (Algorithm 1) with four implementation notes covering the acceptance criterion, KV-cache interaction, bidirectional attention scope, and the τ=0.0 degenerate case.

**Remaining minor issues:**

4. The Figures and Tables section at the end of paper.md still contains the marker `*[TODO: Generate from fig2_igsd_architecture_desc.md spec via TikZ or diagram tool]*` for Figure 2. Once Figure 2 is generated, this TODO comment should be removed from the section listing.

5. Appendices A, C, and D remain as placeholders ("*[Placeholder: ...]*"). Appendix A (full per-seed results) is needed for the submitted paper's supplemental. Appendix C (M2 negative results) and D (qualitative examples) are supportive. These are not blocking for review-stage assessment but must be populated before camera-ready.

6. The Table 6 (task-dependent recipes) footnote correctly flags that M3 QAS = 1.582 differs from Table 2's 1.675 because Table 6 is domain-specific. This is clear and well-handled.

### Writing Quality: 8/10

All major banned patterns from previous reviews have been eliminated and do not reappear:
- No "growing ecosystem," "novel" (unquantified), "state-of-the-art," "significantly," "Moreover," "Furthermore," "It is worth noting," "groundbreaking," or "In recent years."
- No "consistent 3.40×" — all CD-SSD speedup claims use "mean 3.40× (range: 1.35× on MBPP to 4.57× on GSM8K)."
- No "IGSD" in the paper body.
- No Wilcoxon or "p < 0.05" claims.

The introduction's self-contained three-problem structure is tight and well-paced. The methods section leads with the formal metric definitions before method descriptions — correct ordering. The experiments section's roadmap sentence ("This section presents results in four parts...") effectively orients the reader.

**Two remaining writing issues:**

1. **Section 3.1 contains a textual error**: "GSM8K accuracy collapses to 14.5% (η = 3.0)" but Table 2 shows M1 at η=3.0 has GSM8K accuracy = 10.3% and GSM8K AccRet = 0.145. The text appears to be reporting AccRet × 100 (14.5%) rather than the absolute accuracy (10.3%). This will confuse readers since the sentence contextualizes "accuracy" (which should mean 10.3%) not "AccRet percentage" (14.5%). **Fix**: change "14.5%" to "10.3% (AccRet = 0.145)."

2. **Section 4.1 last paragraph** claims "exactly one pair (M1 + CD-SSD) satisfies this condition" using the framing "trajectory-preserving vs. trajectory-modifying." The banned term list flags `"binary composability" (as universal law)` — and indeed the paper uses this framing carefully with "binary pattern observed across the three method pairs" in the conclusion (§6.1). However, §4.1 ends with: "Only trajectory-preserving combinations can achieve Ortho ≥ 1.0. Among the three active methods in this study, exactly one pair (M1 + CD-SSD) satisfies this condition." This is appropriately scoped to the three active methods. No correction needed — the scoping is correct.

**What works well in writing:**

- The fraud-prevention framing around M3's MATH500 AccRet of 243.9% ("a statistical artifact of the 11.1% baseline and should not be interpreted as a real improvement") is exactly right — flagging the artifact preempts a potential reviewer accusation of cherry-picking.
- The "scope and honest positioning" paragraph in the introduction correctly positions CD-SSD as a composability vehicle, not a standalone claim, and pre-empts the "why is 5.13× below published 15–26×?" objection.
- The deployment guidance (§4.5) with three named rules and runtime detection signals is the most practically actionable part of the paper and is clearly written.

### 37 unresolved citation placeholders

All 37 `[CITE:xxx]` placeholders remain. These span the abstract, introduction, methods, experiments, related work, and conclusion. This is blocking for submission and will cause compilation failures in LaTeX. The BibTeX file at `writing/latex/references.bib` must be checked for whether entries exist and keys need to be matched, or whether entries must be sourced.

---

## Issues for the Editor

1. **[Critical] Generate Figure 2 (CD-SSD architecture).** `fig2_igsd_architecture.pdf` does not exist. The spec is at `writing/figures/fig2_igsd_architecture_desc.md`. Key elements: three-phase horizontal flow (Draft 16-step → confidence histogram with τ=0.9 split → Refine 64-step on S_refine with frozen S_accept in context), lock icons on frozen tokens in refine phase, annotation "$H_i = 0 \Rightarrow$ guaranteed KV-cache hit" pointing from frozen token icons to M1 cache representation. Once generated, remove the `[TODO]` comment from the Figures and Tables section listing.

2. **[Critical] Replace all 37 [CITE:xxx] placeholders** with real BibTeX citation keys. Check `writing/latex/references.bib` for existing entries. Missing entries must be sourced from arxiv/DBLP before submission. This is the second blocking item.

3. **[Minor] Fix Section 3.1 accuracy notation error.** "GSM8K accuracy collapses to 14.5% (η = 3.0)" should read "10.3% (AccRet = 0.145, η = 3.0)." The 14.5% value is AccRet × 100, not the absolute accuracy, which Table 2 shows as 10.3%. **Fix**: in §3.1 M1 paragraph, change "cache hits increase but GSM8K accuracy collapses to 14.5% (η = 3.0)" to "cache hits increase but GSM8K accuracy collapses to 10.3% (AccRet = 0.145, η = 3.0)."

4. **[Minor] Introduction contribution item 1 needs NO_GO clarification.** Add "(one family, M2, receives a NO_GO verdict and is excluded from pairwise experiments)" to contribution item 1 to match the abstract's phrasing and prevent reader confusion.

5. **[Non-blocking] Populate Appendices A, C, D.** Appendix A (full per-seed results), Appendix C (M2 detailed sweep), and Appendix D (qualitative examples) remain as structural placeholders. These are required before camera-ready submission. Appendix A data is available in the per-seed JSON files.

---

## What Works Well

1. **Numerical integrity across the paper is now high.** The previous review's three critical discrepancies (CHR ~96% vs. ~94%, Table 2 M1 speedup column inconsistency, CD-SSD QAS contradiction) are all resolved and verified against raw data. The quantitative chain — frozen tokens → H_i = 0 → CHR_refine = 94% → 9.4% super-multiplicative premium — is now internally consistent end to end, matching notation.md, the raw JSON files, and the paper body simultaneously.

2. **The failure mode atlas (Table 4, §3.3)** is the paper's most practically differentiated contribution. Each of the four failure modes has a severity label (CRITICAL vs. MODERATE), a mechanistic root cause, a runtime-observable detection signal (e.g., "AccRet < 0.55 at J ≥ 2 → auto-reject"), and a proactive remedy. The F1 root cause explanation — "LLaDA's masked denoising requires sequential cumulative conditioning; aggressively unmasking J× more tokens per step commits positions before sufficient diffusion context has accumulated" — is specific, mechanistic, and informative.

3. **The τ=0.0 ablation resolution (§4.3)** demonstrates scientific rigor. Rather than obscuring the finding that CD-SSD's confidence partitioning adds no standalone value over naive T=16 step reduction, the paper presents this directly and uses it to sharpen the argument: CD-SSD's value is structural (enabling frozen-token KV synergy), not as a standalone quality-speed improvement. This honest framing strengthens the paper's credibility.

SCORE: 7

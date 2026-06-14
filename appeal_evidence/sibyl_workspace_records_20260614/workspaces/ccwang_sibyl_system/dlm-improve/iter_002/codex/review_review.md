# Codex Independent Review - review

**Review Time**: 2026-03-12 18:04:02 AEDT  
**Model**: Codex / GPT-5 style local fallback review (`mcp__codex__codex` unavailable in this runtime)  
**Mode**: Independent third-party paper review

## Review Comments

### Overall Verdict

This is a disciplined and much better scoped paper than a typical late-stage research draft. The authors have clearly resisted the temptation to turn a small gain into a hero-method story, and the shift toward a compute-normalized diagnostic framing is intellectually honest. That said, the draft is not yet submission-ready. The central paper claim is about auditable runtime fairness and interpretable bucket-level evidence, but the current artifact bundle still leaves a reviewer with too much room to doubt whether the comparison is internally consistent and whether the observed gain reflects reasoning repair rather than output-completion effects.

### Strengths

1. The paper has a clear identity. It no longer oversells a new controller family and instead presents a narrower protocol/diagnostic contribution.
2. The bucket decomposition is the right kind of evidence for a small headline delta. Reporting `fixed / harmed / no_effect` is much more informative than only reporting `+3pp`.
3. The draft already downscopes the seed result appropriately. It does not claim full robustness and mostly treats the three-seed check as a sign-consistency gate.

### Major Concerns

#### 1. The honest-compute argument is undermined by inconsistent runtime evidence.

The paper repeatedly centers the audited execution envelope `eager|compile=True` with safe batch size `57` as the basis for fair interpretation. That is supported by `exp/results/runtime_probe_iter2.json`, and the headline bucket audit in `exp/results/benefit_bucket_audit_pilot.json` is also recorded under `compile_enabled=true` and `batch_size=57` for both methods.

However, `exp/results/runtime_fairness_matrix.json` still reports the headline methods from a different source artifact with materially different realized settings: `Standard-64` at batch size `115`, and `Entropy-Revise-64+3` at batch size `32` with `compile_enabled=false`. For a paper whose credibility rests on runtime lineage, this is not a minor bookkeeping issue. A skeptical reviewer will ask whether the manuscript is mixing an old exploratory fairness table with a new audited pairwise comparison. Unless this is explicitly separated in the text, the runtime-fairness contribution becomes self-undermining.

What to fix:

- Regenerate the fairness matrix under the same audited runtime contract as the paper's headline comparison, or
- explicitly distinguish "historical exploratory matrix" from "final audited comparison" and stop using the former as direct support for the latter.

#### 2. "Full coverage" is currently ambiguous and can be read as overstating the empirical scope.

The paper says the bucket audit is at "full coverage," but `exp/results/benefit_bucket_audit_pilot.json` has `num_samples = 100`, while `exp/results/runtime_probe_iter2.json` reports GSM8K with `num_samples = 1319`. So "coverage = 100%" means full coverage of the reviewed slice, not full coverage of the benchmark.

That distinction matters. Right now a reader could reasonably misread the paper as claiming a benchmark-level audited decomposition, when the actual artifact is a 100-example slice. The draft should make the audited scope explicit everywhere the result is mentioned: abstract, setup, results, and conclusion.

What to fix:

- Replace "full coverage" with "100% of the audited 100-example slice" unless the full benchmark is actually audited.
- State the slice construction protocol clearly: random sample, fixed subset, or convenience slice.

#### 3. The bucket evidence currently looks at least partly like completion/truncation repair, not yet clearly reasoning repair.

The per-sample evidence in `exp/results/benefit_bucket_audit_pilot.json` and `exp/results/benefit_bucket_examples_pilot.json` suggests that many `fixed` and `harmed` cases are driven by short or damaged outputs:

- Fixed examples such as indices `24`, `28`, `33`, `59`, and `88` move from empty/incomplete baseline outputs to short finalized numeric answers.
- Harmed examples such as `55` and `62` look like output damage or truncation (`"can"`, `"$25,00"`) rather than deep reasoning reversals.

This does not invalidate the result, but it changes the interpretation. A reviewer could conclude that the current evidence mainly shows that revision sometimes rescues incomplete generations and sometimes corrupts otherwise correct short answers. That is a weaker and more specific story than "bucket-level recoverability explains revision behavior" in a mechanistic sense.

What to fix:

- Add a second-level annotation for fixed/harmed cases: truncation/completion repair, arithmetic correction, reasoning-path correction, formatting-only, etc.
- Quantify how much of the `+3pp` comes from completion rescue versus substantive reasoning correction.
- If most gains are shallow-completion gains, say so directly and frame that as the real contribution.

#### 4. The robustness section is acceptable only if it stays visibly modest.

The paper mostly handles this correctly, but the evidence remains thin: the three-seed deltas are `+0.03`, `+0.01`, and `+0.01` in `exp/results/seed_sensitivity_spotcheck.json`. That is enough for sign consistency, but it is very weak evidence for stability, especially on a 100-example audited slice. Reviewers will still wonder about uncertainty.

What to fix:

- Keep the current modest language and avoid any stronger phrasing.
- If possible, report paired win/loss counts or confidence intervals for the headline delta, even on the slice.

#### 5. The paper is missing the scholarly scaffolding expected of a paper, especially in Related Work.

`writing/paper.md` currently reads like a well-structured internal manuscript, not yet like a submission-ready paper. The Related Work section is conceptually organized, but it has no citations and no concrete engagement with prior DLM revision/control papers, calibration-as-observer work, compute-accounting literature, or error-decomposition precedents.

For a protocol-first paper, missing citations are especially costly: the reader needs help understanding whether this is a new paper type, a synthesis of known concerns, or a response to a concrete failure mode in existing practice.

What to fix:

- Add citations throughout Section 2 and anchor each thread to specific representative papers.
- Make clear what is genuinely new here: the particular audited bundle and paper framing, not the general idea that compute details matter.

### Minor Comments

1. The paper would benefit from one compact table that aligns the three evidence objects side by side: bucket audit, runtime contract, and seed closure.
2. The paper should define the audited slice once and then reuse exactly the same phrasing everywhere.
3. If boundary evidence is not a main contribution in this draft, avoid even light mentions unless they directly help the scoped claim.

### Recommendation

Promising draft with a credible core idea, but the current manuscript still has one major internal-consistency problem and one major interpretability problem. I would not treat it as publication-ready until the runtime evidence is unified and the bucket gains are broken down into shallow-completion versus genuine reasoning repairs.

## Score

6/10

# Writing Critique

## Verdict

This draft is strategically much better than a method-forward version, but it is still written as if the framing itself were stronger evidence than it actually is. The prose is disciplined, yet it spends too many lines defending what the paper is not and too few lines confronting the most reviewer-visible weaknesses in the evidence.

## Major Problems

### 1. The abstract and introduction oversell the strength of the headline

The paper leads with `+3pp`, `full coverage`, and an auditable runtime story (`writing/paper.md:5`, `writing/paper.md:15`, `writing/paper.md:75`), but it does not foreground that the bucket audit is only `n=100` while GSM8K on disk contains 1319 examples (`exp/results/benefit_bucket_audit_pilot.json:6`, `exp/results/runtime_probe_iter2.json:58-66`). That is the kind of omission that turns a fair number into an unfair impression.

The same issue appears in the seed narrative. The paper cites deltas `+0.03`, `+0.01`, and `+0.01` (`writing/paper.md:81`), but the conclusion still sounds anchored on the best-seed outcome instead of the multi-seed center of mass. The honest headline is not "we got +3pp"; it is "on a 100-example audited slice, the direction stayed positive across three seeds and the effect was small."

### 2. The paper states a stronger scientific thesis than the workspace evidence supports

The manuscript repeatedly presents the observer-controller split as a central conceptual result (`writing/paper.md:25`, `writing/paper.md:35-39`, `writing/paper.md:103`). But the workspace does not contain a direct matched experiment that validates this split. The optional minimal controller probe was skipped on purpose (`exp/results/min_controller_decoupling_probe.json`), and the protocol asset only defines the split rather than demonstrating it (`exp/results/observer_controller_protocol.json:4-37`).

That means the current text should treat the split as framing or motivation, not as a demonstrated empirical takeaway.

### 3. Related Work is too generic and too ungrounded

`writing/paper.md:21-29` reads like a placeholder synthesis rather than a submission-ready related-work section. There are no citation anchors in the Markdown draft, and the claims are broad enough that a reviewer could easily say, "this is all true, but it is not yet scholarship." The paper needs named prior threads, concrete contrasts, and a sharper explanation of what exactly is new here.

### 4. The draft repeats its self-positioning instead of using that space for hard caveats

The manuscript says several times that it is "not a hero-controller paper" (`writing/paper.md:11-19`, `writing/paper.md:89-103`). That message only needs to land once. The extra repetitions should be spent on the hard caveats that matter more:

- the audit is a slice, not the full benchmark;
- the multi-seed effect is smaller than the headline seed;
- the runtime-lineage bundle still mixes current audited runtime facts with historical headline rows.

## Section-Level Notes

### Abstract

The abstract is clean, but it is too comfortable with `full coverage` and `+3pp`. Add `n=100 audited slice` and say explicitly that the seed evidence constrains the effect to a small, sign-consistent gain rather than a robust benchmark-level win.

### Introduction

The introduction has the right strategic move, but it currently makes a stronger promise than the evidence can cash out. It should promise an audit of one headline pair under honest-compute conditions, not a general account of when observer quality fails to become controller gain.

### Related Work

This section needs real literature, not only categories. Add specific DLM revision baselines, prior uncertainty/calibration work, and prior compute-accounting/reporting discussions. Without that, the novelty claim sounds self-declared.

### Experiments

The experiment section is readable, but it hides the most dangerous ambiguity: the paper discusses the recommended runtime path as if it cleanly governs the headline comparison, while the runtime matrix still contains historical rows from different execution settings.

### Discussion and Conclusion

These sections are honest in tone, but they still speak too confidently about what the protocol bundle has established. The protocol bundle is strongest as a credibility aid; it is weakest as evidence for a new general law.

## Recommended Rewrite Direction

The most defensible version of this paper is:

- one sharply bounded GSM8K slice audit;
- one explicit honest-compute lineage bundle;
- one minimal seed sign-consistency check;
- one direct paragraph admitting that observer-controller separation remains a motivating framing, not a proven general result.

If the draft adopts that voice, it will feel smaller, but much harder to attack.

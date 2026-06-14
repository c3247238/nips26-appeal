# Ideation Critique

## Verdict

The ideation pivot away from a hero-controller paper was correct. The problem is that the proposal still preserves one layer of ambition too many: it wants the observer-controller split to function as the paper's defining intellectual frame even though the delivered evidence mostly supports a narrower "bucket audit + honest compute" paper.

## What the Proposal Gets Right

- It correctly abandons generic controller/scheduler storytelling (`idea/proposal.md:5`, `idea/proposal.md:123-129`).
- It correctly identifies `cand_bucket` as the main evidence lane and runtime lineage as support (`idea/proposal.md:83-105`, `idea/proposal.md:111-121`).
- It correctly forces a minimal seed check into the quality gate (`idea/proposal.md:146-154`).

These are strong strategic calls.

## Where the Ideation Still Overreaches

### 1. The top-level thesis is still too ambitious for the current evidence

The proposal's main thesis says that aggregate gain is insufficient and that stronger observer quality does not automatically turn into stronger controller gain (`idea/proposal.md:20-28`). The first half is supported by the bucket audit. The second half is not yet supported by the current workspace unless you add a matched test. Right now it is a plausible interpretation, not a delivered result.

This matters because the entire proposal hierarchy still treats `Observer-Controller Split` as the framing shell (`idea/proposal.md:32-44`, `idea/proposal.md:188-195`, `idea/proposal.md:243-246`). That shell is elegant, but it remains under-evidenced.

### 2. The proposal promised a richer recoverability story than the manuscript now delivers

The ideation document explicitly asks for reasoning/code boundary linkage and a boundary-sensitive failure audit (`idea/proposal.md:45-58`, `idea/proposal.md:89-91`). The current paper does not deliver that. It gives bucket counts and a few examples, but it does not yet show which repairs are shallow completion fixes, which are true reasoning repairs, and how that interacts with any code/reasoning boundary.

### 3. The backup path is safer than the main framing

`Backup 1: cand_minimal_scope` says the paper can shrink to a minimal `honest compute + bucket audit` diagnostic and drop the stronger observer/controller expression (`idea/proposal.md:227-231`; `idea/alternatives.md:3-14`). Based on the current evidence, that backup is not just a fallback. It is arguably the cleanest mainline paper.

By contrast, `Backup 2: cand_boundary_audit` is not ready because the current manuscript barely surfaces the boundary artifacts it would need (`idea/alternatives.md:15-26`).

## Recommendation

The ideation layer should make an explicit choice:

1. Either keep the observer-controller split only as motivation and promote `cand_minimal_scope` to the main paper identity.
2. Or run one additional matched experiment that actually tests the split.

The current middle state is risky. It keeps the ambitious intellectual framing while only partially delivering the empirical support.

## Concrete Ideation Fixes

- Rewrite the main thesis so that the fully supported part comes first: bucket decomposition plus realized-runtime lineage changes how one training-free revision gain should be interpreted.
- Recast `observer/controller split` as a hypothesis or framing sentence unless a direct test is added.
- Move `boundary-sensitive failure audit` back to future work unless the paper actually analyzes boundary artifacts in the main body.
- Treat `cand_minimal_scope` as the preferred plan, not as a defensive fallback.

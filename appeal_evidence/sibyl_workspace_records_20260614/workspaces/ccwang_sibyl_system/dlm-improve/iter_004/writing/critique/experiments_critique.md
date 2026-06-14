# Experiments Critique

**Score: 7.1/10**

## Strengths

- The section reports the key numbers clearly and uses the right contrast as the center of gravity.
- The figure/table plan has been realized with actual scripts and PDFs, which materially improves the draft.
- The runtime-lineage discussion is appropriately surfaced instead of hidden.

## Main Issues

1. The section lacks confidence intervals or statistical tests, even though the margins are small.
2. Table 1 currently mixes quality and speed metrics without explicitly reminding the reader that the candidate is not the absolute fastest method.
3. Table 2 is useful, but the text should explain more directly why net repair is informative yet insufficient on its own.

## Suggested Revisions

- Add a paragraph or appendix pointer on uncertainty estimation (paired bootstrap, Wilson intervals, or McNemar tests).
- Add one sentence after Table 1 explicitly stating that the paper's strongest empirical claim is candidate-versus-sham, not raw speed dominance.
- If possible, add a small runtime-lineage table or appendix reference with batch, VRAM, and auxiliary overhead values.

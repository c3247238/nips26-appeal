# Figure 1 Description: Minimal Audit Template for Small-Gain DLM Revision Claims

## Intended caption

**Figure 1.** Minimal audit template used in this paper. A localized signal becomes publishable only after it is filtered through a compute-matched active control, a budget-matched sham control, a sample-level audit, and a current-only artifact closure. The template narrows the paper to an audited negative case rather than a new controller claim.

## Layout

- Use a left-to-right flow diagram with five blocks.
- Block 1: `Observed small gain`.
- Block 2: `Compute-matched active control (DNB-84)`.
- Block 3: `Budget-matched sham control (RAND-84)`.
- Block 4: `Sample-level audit + current-only artifact closure`.
- Block 5: `Claim ceiling`.

## Arrows and logic

- Draw an arrow from `Observed small gain` to `Compute-matched active control`.
- Draw an arrow from `Compute-matched active control` to `Budget-matched sham control`.
- Draw an arrow from `Budget-matched sham control` to `Sample-level audit + current-only artifact closure`.
- Draw an arrow from `Sample-level audit + current-only artifact closure` to `Claim ceiling`.

## Text inside blocks

- `Observed small gain`
  - `CARD-84 > DNB-84 on GSM8K`
  - `net repaired = +7`
- `Compute-matched active control (DNB-84)`
  - `Rules out "more budget only"`
  - `Does not rule out generic targeting`
- `Budget-matched sham control (RAND-84)`
  - `CARD-84 vs RAND-84`
  - `net repaired = +1`
  - `Positive controller claim fails`
- `Sample-level audit + current-only artifact closure`
  - `per_sample_audit.csv`
  - `transition_matrix.csv`
  - `claim_to_asset_map.json`
  - `runtime_contract.json`
- `Claim ceiling`
  - `Entropy is a risk marker`
  - `Audited-slice scope only`
  - `No winning-method claim`

## Visual style

- Use neutral gray for the first and fourth blocks.
- Use blue for the compute-matched active control block.
- Use orange for the sham-control block.
- Use red outline or accent on the claim-ceiling block to emphasize restriction, not triumph.
- Keep the figure printable in grayscale.

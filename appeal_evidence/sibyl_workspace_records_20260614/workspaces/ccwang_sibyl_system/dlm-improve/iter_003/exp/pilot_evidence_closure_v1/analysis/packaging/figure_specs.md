# Figure Specs For The Audited Negative Case

## Figure 1: Main results table
Source: `main_results_table.csv`
Focus: show that `CARD-84` exceeds `DNB-84` on GSM8K but does not separate from `RAND-84` on the same audited slice.

## Figure 2: Repair / harm stacked bars
Source: `repair_harm_table.csv`
Focus: compare `fixed`, `harmed`, `unchanged_correct`, and `unchanged_wrong` counts across `dnb84_vs_dnb64`, `card84_vs_dnb64`, `rand84_vs_dnb64`, `card84_vs_dnb84`, and `card84_vs_rand84`.

## Figure 3: MBPP harm profile
Source: `harm_profile_table.csv`
Focus: show that code-side behavior should be written as harm profile / failure localization, not as a mechanism story.

## Figure 4: Claim ceiling table
Source: `claim_scope_map.json`
Focus: list allowed wording, forbidden wording, and required disclosures for the paper.

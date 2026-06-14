# Critique: Related Work

## Score: 7.5/10

## Strengths
- Clear definition of feature absorption with formal notation
- Good coverage of supervised detection methods and their limitations
- Accurate description of UAD pipeline

## Issues
1. **Line 9**: Formal definition uses expectation notation but no grounding in citation. The equation is intuitive but not directly from Chanin et al.
2. **Section 2.4**: Collision rate section is good but could be expanded slightly to explain *why* collision rate is mentioned as a potential indicator (currently just states it is).
3. **Missing**: No mention of other unsupervised methods beyond UAD (e.g., decoder weight similarity approaches).

## Suggestions
- Add a brief forward reference to Section 5.4 where decoder weight similarity is proposed
- Consider mentioning that UAD's core assumption has not been validated before this work

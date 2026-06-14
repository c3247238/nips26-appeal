# C2A PMI Extraction — PILOT Summary

**Task:** C2A_pmi_extraction  
**Mode:** PILOT  
**Date:** 2026-04-14  
**Go/No-Go: GO**

## Configuration
- Token corpus: 100k tokens from Skylion007/openwebtext (first 87 docs, ~1.1k tokens/doc average)
- Tokenizer: Gemma 2 (unsloth/gemma-2-2b, vocab_size=256000)
- Letters: A, B, C, D, E
- Sliding window size: 5

## Results

| Letter | Token types matching | PMI entries |
|--------|---------------------|-------------|
| A | 978 | 2346 |
| B | 659 | 2259 |
| C | 1181 | 2195 |
| D | 714 | 2095 |
| E | 616 | 2004 |

**Total PMI entries:** 10899  
**PMI range:** [-2.38, 1.96] (range = 4.33)  
**NaN/Inf values:** None

## Pass Criteria

| Criterion | Required | Actual | Pass? |
|-----------|----------|--------|-------|
| >= 5 distinct child tokens per letter | 5 | 2004 (min) | YES |
| PMI values span [-3, 5] range (not degenerate) | range >= 1.0 | 4.33 | YES |
| No NaN/inf in output | 0 | 0 | YES |

## Top PMI tokens per letter

- **A:** Jet (1.09), Care (1.06), professor (1.05), Port (1.00), Canada (0.99)
- **B:** reme (1.54), concert (1.52), Prey (1.47), ATI (1.41), submitted (1.41)
- **C:** aters (1.91), size (1.57), Medical (1.47), Vehicle (1.47), Blackjack (1.37)
- **D:** ites (1.96), mployment (1.63), Vs (1.63), Evil (1.55), count (1.52)
- **E:** code (1.79), ] (1.79), Sus (1.79), pect (1.79), Planning (1.75)

## Notes

- The tokenizer used is the Gemma 2 tokenizer (256k vocabulary, SentencePiece BPE), which produces subword tokens.
  Many top PMI tokens are subword fragments (e.g., "reme", "aters") — this is expected and correct for the PMI computation.
- PMI values are bounded in [-2.4, 1.96], which is well within the [-3, 5] acceptable range.
- Full experiment (1M tokens, all 26 letters) will use the same tokenizer and pipeline.
- Runtime: ~2 minutes (pilot), estimated 20 minutes for full 1M-token run.

## Output Files

- `pilots/C2A_pmi_extraction_pilot.json` — pilot summary
- `full/C2A_pmi_features.json` — PMI features (pilot data, top-20 per letter)

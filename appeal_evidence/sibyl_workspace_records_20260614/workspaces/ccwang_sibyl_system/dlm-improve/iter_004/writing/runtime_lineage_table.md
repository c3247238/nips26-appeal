# Runtime Lineage Summary

| Method | Batch | Peak VRAM (MB) | Wall-clock (s) | Equal-quality speed (tok/s) | Avg. NFE | Extra forwards | Auxiliary overhead (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cand_espd | 54 | 15289 | 2713.81 | 124.42 | 67.93 | 3868 | 2569.81 |
| ESPD-FixedFrontier | 52 | 41973 | 3193.59 | 105.73 | 68.00 | 3961 | 3049.62 |
| CARD-84 | 57 | 40910 | 2678.23 | 126.08 | 68.00 | n/a | n/a |
| RAND-84 | 57 | 40910 | 2638.06 | 128.00 | 67.00 | n/a | n/a |

## Interpretation

- The candidate does not dominate the shared controls in absolute speed.
- The strongest runtime claim is candidate-versus-sham, not candidate-versus-all-controls.
- The sham remains only partially matched because batch, VRAM, and auxiliary overhead differ materially.

# E2 Full GPT-2 Summary

**Task:** e2_full_gpt2
**Total Time:** 13.7 min
**Successful:** 15/15

## Results

| Checkpoint | Family | Absorption (Full) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|
| gpt2-small-res-jb/blocks.0.hook_resid_pre | Standard | 0.412941620878041 | 12.998122215270996 | 0.2187092900276184 | 0.9762586429715157 |
| gpt2-small-res-jb/blocks.4.hook_resid_pre | Standard | 0.6731560266340741 | 34.53474426269531 | 0.03804522752761841 | 0.9957637810148299 |
| gpt2-small-res-jb/blocks.8.hook_resid_pre | Standard | 0.4781737264780423 | 66.57709503173828 | 0.001708984375 | 0.9886560905724764 |
| gpt2-small-res-jb/blocks.11.hook_resid_pre | Standard | 0.5888515049272721 | 59.37789535522461 | 0.002888977527618408 | 0.9598175697028637 |
| gpt2-small-resid-post-v5-32k/blocks.0.hook_resid_post | TopK | 0.3768049581289262 | 32.0 | 0.1231689453125 | 0.8252654820680618 |
| gpt2-small-resid-post-v5-32k/blocks.4.hook_resid_post | TopK | 0.32001282855543195 | 32.0 | 0.065643310546875 | 0.9921718733385205 |
| gpt2-small-resid-post-v5-32k/blocks.8.hook_resid_post | TopK | 0.19432507292120066 | 32.0 | 0.0950927734375 | 0.9804341737180948 |
| gpt2-small-resid-post-v5-32k/blocks.11.hook_resid_post | TopK | 0.036438089897235106 | 32.0 | 0.204376220703125 | 0.8764684125781059 |
| gpt2-small-resid-post-v5-128k/blocks.4.hook_resid_post | TopK | 0.4112976314104633 | 32.0 | 0.34597015380859375 | 0.9933201236992049 |
| gpt2-small-resid-post-v5-128k/blocks.8.hook_resid_post | TopK | 0.2489843979067389 | 32.0 | 0.3477020263671875 | 0.9833335293192352 |
| gpt2-small-mlp-out-v5-32k/blocks.4.hook_mlp_out | TopK_MLP | 0.20859587306361804 | 31.968751907348633 | 0.01544189453125 | 0.8327264934778214 |
| gpt2-small-mlp-out-v5-32k/blocks.8.hook_mlp_out | TopK_MLP | 0.06359293682619072 | 32.0 | 0.01629638671875 | 0.7007135450839996 |
| gpt2-small-mlp-out-v5-128k/blocks.8.hook_mlp_out | TopK_MLP | 0.007687751982272746 | 32.0 | 0.183746337890625 | 0.7448147690712367 |
| gpt2-small-attn-out-v5-32k/blocks.4.hook_attn_out | TopK_Attn | 0.17191332749481888 | 32.0 | 0.07928466796875 | 0.7096702754497528 |
| gpt2-small-attn-out-v5-32k/blocks.8.hook_attn_out | TopK_Attn | 0.04996221538579848 | 32.0 | 0.11834716796875 | 0.7501286417245865 |

## Family-Level Statistics


# Notation

This document fixes the notation used across the iteration-3 paper draft.

| Symbol | Meaning |
| --- | --- |
| \(s\) | One audited sample from the current evidence bundle. |
| \(\mathcal{S}\) | The audited slice of 100 samples used in the paper-facing analysis. |
| \(d(s)\) | Dataset membership of sample \(s\), with \(d(s) \in \{\text{GSM8K}, \text{MBPP}\}\). |
| \(a\) | One inference arm from \(\{\text{DNB-64}, \text{DNB-84}, \text{CARD-84}, \text{RAND-84}\}\). |
| \(y_a(s)\) | Output produced by arm \(a\) on sample \(s\). |
| \(c_a(s)\) | Binary correctness indicator for arm \(a\) on sample \(s\). |
| \(u(s)\) | Precomputed observer-side entropy score used for sample stratification. |
| \(\tau(s)\) | Selection stratum for sample \(s\), such as `high_entropy` or `balanced_mid_low`. |
| \(\mathrm{NFE}(a)\) | Average number of denoising function evaluations used by arm \(a\). |
| \(\Delta_{\text{repair}}(a,b;D)\) | Net repaired count of arm \(a\) relative to arm \(b\) on dataset \(D\), defined as fixed minus harmed samples. |
| \(\mathrm{Fix}(a,b;D)\) | Count of samples fixed by \(a\) relative to \(b\) on dataset \(D\). |
| \(\mathrm{Harm}(a,b;D)\) | Count of samples harmed by \(a\) relative to \(b\) on dataset \(D\). |
| \(m\) | Failure mode label on MBPP, such as `NameError` or `SyntaxError`. |

Notation policy for this paper:

- `CARD-84` is the audited intervention arm, not a validated winning method.
- \(u(s)\) is treated as a risk marker that informed audit slice construction, not as a validated targeting rule.
- \(\Delta_{\text{repair}}(\text{CARD-84}, \text{RAND-84}; \text{GSM8K})\) is the key quantity that enforces the paper's claim ceiling.

# Conclusion

This paper reports an audited negative case for training-free DLM revision. Within a current-only 100-sample slice, `CARD-84` shows a localized GSM8K advantage over the compute-matched `DNB-84` control, but that advantage contracts to a minimal margin once the budget-matched sham control `RAND-84` is introduced. The resulting interpretation is therefore narrower than the original method-forward story: the evidence supports a localized signal and a useful risk marker, not a validated entropy-guided controller.

The supporting contribution is a minimal audit template for small-gain settings. By combining a compute-matched active control, a sham control, sample-level repair/harm accounting, and explicit claim-to-asset lineage, the template turns an appealing but overstated success story into a more credible scientific object. In this iteration, that object is not a new decoding method. It is a bounded, reviewer-auditable negative case.

The durable lesson is simple. In training-free DLM revision, compute-matched comparisons alone are not enough when a budget-matched sham control can still rewrite the conclusion. Future work should treat that stronger control logic as a default requirement before small gains are promoted into controller claims.

<!-- FIGURES
- None
-->

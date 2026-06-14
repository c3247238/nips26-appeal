# Writing Critique

## Executive view

The manuscript is now disciplined enough to read as a real paper rather than an internal note, but its rhetoric still needs one more turn toward reviewer convenience. The strongest version is not "we found a small but real controller signal." It is "a sham control materially rewrote what would otherwise have been a plausible small-gain story."

## Main writing risks

1. The paper still introduces a tempting positive contrast too vividly before fully locking the reader into the sham-control interpretation. The +7 repair margin over `DNB-84` is memorable, so the paper must make the +1 margin over `RAND-84` feel like the true decision variable from the outset.
2. The "minimal audit template" language is good, but it can sound bigger than the evidence. The paper should keep reminding the reader that the template is demonstrated through one audited negative case, not validated as a universal framework.
3. The runtime disclosure is honest but slightly over-detailed relative to its role in the argument. It should read as a transparency clause, not as a second scientific storyline.
4. The bibliography and related-work positioning are adequate, but still a bit thin for a paper that depends so heavily on interpretive framing rather than raw empirical breadth.

## Suggested fixes

- Move the sham-control reinterpretation to the earliest possible point in the abstract and introduction.
- Add one sentence explicitly stating that the paper's generality claim is only about audit logic, not about benchmark-wide empirical prevalence.
- Shorten the runtime paragraph so the reader immediately sees what was actually measured and what was only probed later for disclosure.
- Strengthen the related-work bridge to uncertainty-guided control and sampler-evaluation papers so the manuscript feels positioned, not merely honest.

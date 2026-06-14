# Codex Review Of Idea Debate

## 最严重的三个风险

1. 论文对象仍然大于证据对象。当前资产最多支持一个经审计的负案例，不支持领域级 protocol paper。
2. “审计成功阻止过度叙事”本身还不足以自动构成强新意，必须被压成一句更窄、更锋利的外部 lesson。
3. runtime / fairness 资产还不够干净，不能再把它们写成护城河式贡献，只能写成被收缩并显式记录的条件。

## 必须砍掉的句子级主张

- skeptical audit paper / protocol-first contribution / necessary contribution
- new interface / new paradigm / reviewer-friendly framework
- almost works / near-win / reasoning signal still alive
- runtime fairness has been fully solved

## 足以撑住论文对象的资产

- `summary.md` 与 `idea_validation_decision.json` 足以撑住一个 audited negative case study 的核心结论。
- `per_sample_audit.csv`、`transition_matrix.csv`、`claim_to_asset_map.json`、`code_failure_modes.md`、`runtime_contract.json` 足以撑住 sample-level 可复核性。
- perspectives 与 debate 全套文本足以说明内部综合后，唯一诚实路线是 negative pivot。

## 最低可接受 framing

本文是一篇关于一个 entropy-guided training-free DLM revision 近阳性案例的 **audited negative case study**。在 matched-compute baseline、sham control、sample-level audit 和 runtime contract 约束下，该方法未能与 budget-matched random targeting 清晰分离。本文的价值不在于提出新 controller，也不在于宣布新 protocol 范式，而在于给出一个最小可复核审计模板，并证明在 DLM small-gain regime 中，`compute-matched baseline` 仍不足以排除伪改进，`sham control` 会实质改变解释。

## 总评

有条件通过，5/10。方向对，但 framing 还要再砍一刀，彻底避免把负案例的元价值写大。

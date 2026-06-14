# Iteration 3 Reflection

## 总判断

这一轮最重要的进展不是“又找到一个更强方法”，而是把论文稳定收缩成一个可信的 negative-case submission package。`CARD-84` 没有再被包装成 winning controller；核心结论已经稳定成：

- `CARD-84 > DNB-84` 只能支持 active-control gain
- `CARD-84 ≈ RAND-84` 说明 attribution 不能落到 entropy-guided targeting
- 因而论文贡献应定位为 audited negative case + audit template，而不是新控制器

同时，LaTeX 链路、图、参考文献和 PDF 编译都已经打通，三路 review 也给出了高度一致的判断：论文现在是诚实、连贯、可推进的，但还没有达到强 submission-ready 状态。

## 本轮真正修复了什么

1. 写作链闭环了：`paper.md`、`review.md`、LaTeX 源和 `main.pdf` 已经对齐，不再停留在“内容大致完成但产物不完整”的状态。
2. 负面结果 framing 稳定了：没有再尝试把 `CARD-84` 救回正面方法故事，sham-control 结论被完整保留。
3. 审查机制可消费了：critic、supervisor、codex 三路审查已经沉淀为稳定产物，能够直接供 reflection 和 quality gate 使用。
4. 参考文献和模板不再是硬阻塞：NeurIPS 模板、BibTeX、图表复制与 PDF 编译已经跑通。

## 为什么还没过线

当前 supervisor score 是 `7.1/10`，说明问题已经从“方向不可信”降成“submission package 仍有几个 reviewer-facing 缺口”。这些缺口集中在四件事上：

1. **claim scope 仍需更硬地收口**：headline 仍容易被读成 broader DLM revision statement，而当前证据只足以支撑 `n=100 audited slice` 的 bounded case study。
2. **runtime lineage 仍不够 reviewer-friendly**：正文披露了 batch probe、executed batch 和 backend 差异，但 reviewer 还需要自己拼接。
3. **reproducibility 还少最后一句承诺**：artifact checklist 很强，但还没有显式 release / replication statement。
4. **最终排版只差 submission polish**：PDF 已成功编译，但 bibliography 仍偏 minimal，appendix-style artifact inventory 仍有轻微 overfull/underfull warning。

## 下一轮不该做什么

- 不要重新打开“hero method”叙事。
- 不要为了追求更高 novelty 再引入新 controller probe。
- 不要把 scoped negative case 再写成 benchmark-level effectiveness claim。
- 不要碰不安全的 Feishu pending queue。

## 下一轮应该只做什么

把焦点锁死在 submission polish，而不是方向再探索：

1. 在 abstract / intro / contribution language 中重复写死 audited-slice scope。
2. 增补一个 reviewer-friendly runtime-lineage note 或小表，明确 executed backend / batch / compile 状态。
3. 加入明确的 artifact-release / replication statement。
4. 做一次 bibliography enrichment + appendix inventory typesetting cleanup。

如果这四件事完成，下一轮最有希望把当前 `7.1` 的 reviewer judgment 推到 `8+`，而不需要再发散到新的实验方向。

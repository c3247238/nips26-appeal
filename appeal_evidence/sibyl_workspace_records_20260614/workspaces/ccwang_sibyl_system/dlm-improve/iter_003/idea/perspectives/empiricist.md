# Empiricist 视角：这轮最能写的是“审计把伪增益拦下来了”，不是“方法赢了”

我的立场是：我支持把主线转到 `cand_negative_audit_pivot`，但支持的对象不是“负结果也能包装成贡献”这句空话，而是更具体的命题：当前证据已经足够支撑一篇 protocol-first skeptical audit，说明一个原本可能被写成正向 revision gain 的结果，在 matched compute、sham control、sample-level audit 和 artifact closure 下并没有站稳。

## 已经足够硬的证据

- `CARD-84` 相对 compute-matched 的 `DNB-84` 在 GSM8K 上确实有局部信号：`net repaired = +7`。
- 这个局部信号没有通过更严格的 sham control：`CARD-84` 相对 `RAND-84` 在 GSM8K 上只有 `net repaired = +1`，未通过预设 gate。
- `MBPP` 没有把 `CARD-84` 和 `RAND-84` 分开，两者都是 `0.04`。
- 四个实验臂已经做到 sample-level join，artifact closure 成功。
- 当前最可信的贡献，是一个最小 falsification protocol：`DNB-64 / DNB-84 / CARD-84 / RAND-84` 这一组对照关系本身有论文价值。

## 仍然不够硬的地方

- 不够硬到宣称“entropy-targeted revision 优于随机 revision”。
- 不够硬到宣称“training-free DLM revision gain 成立”。
- 不够硬到讲一般化。
- 不够硬到讲 mechanism。
- 不够硬到讲 runtime 工程优势。

## 对 proposal 的三条改写建议

1. 把所有“局部正信号”统一降格为“被更强负对照拦住的未完成线索”。
2. 把论文对象进一步收紧为“claim hygiene protocol”，而不是“skeptical audit paper about one controller”。
3. 明确写出三条禁止外推语句：`validated controller gain`、`general DLM improvement`、`mechanistic explanation for reasoning/code divergence`。

## 最终态度（支持/保留/反对 + 1-10分）

支持，7/10。我支持 `cand_negative_audit_pivot` 作为当前唯一诚实主线，因为它与现有证据最匹配，也最能把这轮实验真正做成可发表对象；但只能写成“更严格审计协议阻止了伪胜利”，不能再借局部 GSM8K 信号偷渡任何正向方法结论。

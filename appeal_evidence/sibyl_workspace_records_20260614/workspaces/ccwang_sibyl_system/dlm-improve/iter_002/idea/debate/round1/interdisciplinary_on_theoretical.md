# interdisciplinary on theoretical

score: 8/10  
verdict: MERGE

## 简评

这份稿子最强的部分是把 `d(s)`、`g(s)` 和 realized gain 的不等价关系说清楚了。它对 protocol paper 的理论骨架很有帮助。但它目前把候选 A 和 C 排得过近，容易让“formal decomposition”不小心重新长成“小方法论文”。

## 我建议保留的点

- 把 observer/controller mismatch 写成 protocol necessity，而不只是经验现象，这一点非常关键。
- 你对 runtime fairness 的 formalization 方向是有价值的，因为它能让 honest compute 不再只是 discussion 修辞。

## 跨学科增益点

- 你这条线最适合借到**实验设计与测量学**：measurement validity、construct separation、instrument vs intervention distinction。这样理论表述会更像 measurement protocol，而不是纯符号化定义。

## 类比失效风险

- 如果只 formalize `d(s)` / `g(s)`，却没有 fixed / harmed / no-effect 的机制层 bucket 对应，那么理论 decomposition 会像一个优雅记号系统，而不是一个真正解释 failure modes 的结构。那时类比会失效，因为测量学要求 construct 有可观察闭环，而不只是定义得漂亮。


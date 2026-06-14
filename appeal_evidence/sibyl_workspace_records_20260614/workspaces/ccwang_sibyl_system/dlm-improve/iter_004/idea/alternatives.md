# Iteration 4 Alternatives

## 当前 serious pool 外的备选方向

### DCD-lite / Late-Commit Revision

- 定位：保守但 reviewer-friendly 的第三梯队替代
- 什么时候考虑：
  - 如果 `cand_bsr` 的核心问题确实是“过早 commit”
  - 但 `cand_ugr` 的 uplift 层被证明太复杂
- 为什么暂不升格：
  - 与 `cand_bsr` 的 object-level 假设有较强重叠
  - 当前更需要先验证局部对象线本身，而不是再扩展一个相近 family

### Cache-First Speed Lane

- 定位：工程速度线替代
- 什么时候考虑：
  - 如果 `cand_espd` 的 stopping / routing 信号不稳定
  - 但整体速度目标仍值得继续
- 为什么暂不升格：
  - 更像系统优化而不是核心 research mainline
  - 容易把方法贡献稀释成纯实现贡献

### AR-Guided Diffusion Lite

- 定位：高风险高回报替代
- 什么时候考虑：
  - 只有 `cand_bsr` 和 `cand_espd` 都显示“training-free 信号存在，但纯 diffusion 信号仍不够”时
- 为什么暂不升格：
  - hidden compute 风险高
  - 过早引入会破坏这一轮的归因清晰度

## 已归档而非 serious pool 的方向

### cand_mgcd

- 状态：archived negative control
- 保留原因：
  - 作为 richer-controller 失败先例
  - 用于提醒后续 proposal 不要再犯 front-runner drift

### cand_dsg

- 状态：archived lighter negative control
- 保留原因：
  - 作为 signal-only gate 的失败参考

## 当前明确拒绝的路线

### 继续救 MGCD

- 拒绝原因：
  - 与真实 pilot 证据冲突
  - 与 `idea_validation_decision = PIVOT` 冲突

### 把速度优化直接写成核心 research contribution

- 拒绝原因：
  - 用户要的是研究贡献，不是单纯系统调优
  - 速度线必须绑定一个清晰的 mechanism claim，例如 entropy-routed compute

### 在没有新 screening signal 前直接进入 full benchmark

- 拒绝原因：
  - 违反 pilot-vs-full 约束
  - 会再次放大 narrative drift

# 备选方案（Pivot 候选）

如果 DTA 核心方案失败，以下备选方案按优先级排序：

---

## 备选方案 1：DMI + SCP 组合（轻量信息增强路线）

### 触发条件
DTA 的 LoRA 更新在 Dream-7B 上导致数值不稳定或计算开销不可接受（>4x 标准去噪）。

### 核心思路
放弃参数级适应，转向更轻量的跨步信息传递：
- **DMI（Diffusion Memory Injection）**：将上一步的 top-k logits 分布作为软标签注入当前步的 mask embedding，使 remasked token 保留前步的"软记忆"
- **SCP（Self-Contradiction Probing）**：利用 DLM 的 MLM 能力对已揭示 token 做 leave-one-out probing，检测自矛盾 token 并优先 remask

### 与 DTA 的区别
- 无参数更新，无反向传播，纯前向传播操作
- DMI 仅增加 ~5% 计算开销，SCP 增加 ~50%
- 理论深度较浅（无 EM/ELBO 理论），但工程风险极低

### 投稿定位
"Beyond Remasking: Information-Augmented Inference-Time Scaling for Discrete Diffusion"
- 目标：NeurIPS/ICLR 主会（如果效果显著）或 EMNLP（中等效果）

### 成功概率
DMI: 55%, SCP: 45%, 组合: ~70%（至少一个有效）

### 关键参考
- Soft-Masked Diffusion (arXiv 2510.17206): 软 embedding 混合的先驱
- CORE (arXiv 2602.04096): 上下文鲁棒性探测
- MetaState (arXiv 2603.01331): 跨步记忆动机

---

## 备选方案 2：DLM 去噪 = 隐式 TTT 的理论分析论文

### 触发条件
所有方法实验均无显著提升（DTA/DMI/SCP 均失败），但理论分析有价值发现。

### 核心思路
将 18 轮负面结果 + DTA 的（可能的）失败统一在一个理论框架下解释：

1. **DLM 去噪与 TTT 的数学对偶性**
   - 证明 DLM 的 T 步去噪等价于 TTT 的 T 步在线学习（目标函数为 masked token 预测损失）
   - 证明 DLM 相比 AR 做 TTT 的结构性优势与劣势

2. **Remasking 的 Lipschitz 收敛率界**
   - 证明 remasking 的有效性受限于去噪网络的 Lipschitz 常数
   - 经验估计不同模型（0.6B, 7B, 8B）的 Lipschitz 常数，解释为什么小模型退化而大模型饱和

3. **推理时计算扩展的信息论上界**
   - 推导无外部信号时 DLM remasking 的质量改善上界
   - 该上界由模型校准误差和 mask 对上下文的干扰共同决定

### 投稿定位
"Denoising as Test-Time Learning: Why Inference-Time Scaling Is Hard for Masked Diffusion"
- 目标：NeurIPS/ICML（如果理论足够强）或 ICLR 研讨会

### 成功概率
理论框架成立: 80%, 论文可发表: 60%

### 关键参考
- Zheng et al. (arXiv 2409.02908): MDM 时间无关性
- Jiang et al. (arXiv 2602.00286): Remasking 无法保证分布正确性
- Piskorz et al. (arXiv 2511.21338): Masks 干扰上下文
- Chen, Cong & Li (arXiv 2511.04647): MDM 采样误差刻画

---

## 备选方案 3：TTT 层作为 DLM 可插拔记忆模块（需轻量训练）

### 触发条件
DTA 的 training-free 方案效果不足（<2pp 提升），但 LoRA 在线更新本身稳定，暗示参数级适应有潜力但需要更好的初始化。

### 核心思路
不在推理时从零开始做 LoRA 更新，而是在预训练 DLM 中插入 TTT 层，通过少量数据微调使其适应 DLM 的去噪过程：

1. 取 Dream-7B 预训练权重
2. 在 [4, 8, 12, 16] 层后插入 TTT-Linear 模块（参考 TPTT）
3. LoRA (r=16) 微调：GSM8K train set + Countdown 合成数据，~2K 步
4. 推理时 TTT 层自动在去噪过程中积累上下文记忆

### 与 DTA 的区别
- 需要训练（~1 天 2xA100），但训练后推理时无梯度计算
- 推理开销更低（仅额外的 TTT-Linear 前向传播，~10%）
- 新颖度稍低（TPTT 已验证 AR 模型的可行性）

### 投稿定位
"Memory-Augmented Diffusion Language Models via Test-Time Training Layers"
- 目标：NeurIPS/ICLR 主会

### 成功概率
微调成功: 70%, 显著提升: 45%, 论文可发表: 50%

### 关键参考
- TPTT (arXiv 2506.17671): 预训练 Transformer -> Titans 转化
- lucidrains/titans-pytorch: 非官方 PyTorch 实现
- AllMem (arXiv 2602.13680): SWA + TTT 记忆网络
- Locas (arXiv 2602.05085): 可插拔参数化记忆

---

## Pivot 决策流程

```
第 1 周末：DTA 概念验证结果
   |
   +-- DTA 在小模型上有效 (+3pp Countdown) --> 继续 DTA 主线
   |
   +-- DTA 数值不稳定 --> Pivot 到备选 1 (DMI+SCP)
   |
   +-- DTA 稳定但无效果 --> 两条路并行：
       |
       +-- 备选 3（TTT 层微调）：需要训练的参数适应
       +-- 备选 2（理论分析）：DLM=TTT 的理论框架

第 3 周末：主实验结果
   |
   +-- 任一方法在 Dream-7B 上 +3pp --> 全力推进该方案
   |
   +-- 无方法显著有效 --> 备选 2 理论论文 + 18 轮数据整合
```

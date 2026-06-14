# 乐观者分析：AADWD 实验结果

## 核心立场

尽管 AADWD 未能在准确率上超越 Fixed WD 基线，这一系列实验的真正价值远远超出了「打败基线」这一单一维度。实验揭示了深度学习正则化中多个深刻的结构性洞察，这些发现本身就构成了一篇高影响力论文的坚实基础。

## 积极发现一：对齐信号确实存在且可测量

实验数据确认了理论框架的核心前提：梯度-参数对齐度 delta_t 是一个可计算、可追踪的量。Conservative AADWD 在所有设定上都接近 Fixed WD 的表现（CIFAR-10/ResNet20: 92.37% vs 92.54%，差距仅 0.17%；CIFAR-100/ResNet20: 68.24% vs 68.45%，差距 0.21%；CIFAR-10/VGG16: 93.75% vs 93.86%，差距 0.11%），这恰恰证明了对齐信号虽小（delta 约 10^{-3}），但不会破坏训练稳定性。Conservative 变体的权重范数（23.60）与 Fixed WD（23.49）几乎一致，generalization gap（7.15 vs 7.17）也高度吻合，说明 AADWD 的理论框架在实践中是「无害的」——这是将其推向更复杂场景的必要前提。

## 积极发现二：等价累积 WD 实验是黄金级因果证据

等价累积 WD（Equivalent Cumulative WD）实验产出了极为干净的因果结论：当动态 WD 的时间平均值与固定 WD 完全匹配时，性能完全一致（均为 92.54%）。这不是 0.1% 的近似匹配，而是精确到小数点后两位的完美匹配。这一发现的理论意义重大：**正则化的总量（mean magnitude）决定性能，而非正则化的时间分配模式（temporal dynamics）**。这一结论将 weight decay 的理论理解从「调度问题」简化为「预算问题」，这是对 Sun et al. (CVPR 2025) 框架的实质性推进。

## 积极发现三：LR 耦合机制的科学发现

解耦实验揭示了一个此前未被明确阐述的机制：gamma_t（学习率乘数）对 weight decay 的稳定性至关重要。去耦后 aggressive 变体完全崩溃（final acc 降至 10.00%，weight norm 仅 0.0036），conservative 变体也严重退化（best acc 从 92.37% 降至 90.32%，final acc 从 92.22% 骤降至 80.30%）。这不仅解释了 AADWD 的失败模式，更揭示了 L2 正则化相对于解耦 weight decay 的一个内在优势：L2 天然提供 lr * wd 的缩放关系。这一发现对 AdamW 等解耦优化器的 WD 调度有直接指导意义。

## 积极发现四：CWD 系统性崩溃的跨架构验证

CWD 在所有三个设定上都表现出晚期崩溃：CIFAR-10/ResNet20（91.79% -> 86.95%, delta=4.84%）、CIFAR-100/ResNet20（66.84% -> 54.27%, delta=12.57%）、CIFAR-10/VGG16（92.95% -> 86.47%, delta=6.48%）。这是对 ICLR 2026 工作的重要实证反驳。coordinate-wise decay 导致的系统性权重收缩（WN 从正常的 23-41 降至 9.28-25.40）在多架构上具有一致性，这一发现本身就值得报告。

## 积极发现五：超参数灵敏度实验揭示了稳定区域

Aggressive 变体的 beta 扫描显示了出人意料的稳定性：beta 从 0.9 到 0.9999 时，best acc 仅在 92.05%-92.25% 之间波动（极差仅 0.20%）。这说明 EMA 平滑参数对性能影响很小，降低了实际部署的调参负担。c 值扫描则清晰地划定了安全边界：c <= 2.5 时性能稳定（91.87%-92.18%），c=5.0 开始退化（87.98%），c=10.0 完全崩溃（52.12% -> 10.00%）。

## 论文转型建议

这些结果最适合写成一篇「principled negative result + mechanistic insight」论文。标题方向：「Why Dynamic Weight Decay Degenerates to Static: The Dominance of Learning Rate Coupling in Nonconvex SGD」。核心贡献：(1) 证明 alignment signal 在标准训练设定中太弱以至于不可行动，(2) 揭示 LR-WD 耦合是稳定性的必要条件而非权宜之计，(3) 等价累积实验证明 WD 时间分配不影响最终性能。这种「回答为什么不行」的论文在 ICML/NeurIPS 上有良好的接受历史。

## 明确判断

**应该继续此研究方向，但需要转型为负面结果论文。** 现有数据量已足够支撑一篇高质量论文，无需额外实验。

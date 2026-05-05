# RL调度器局部奖励与综合奖励设计方案

## 1. 局部奖励设计 (Local Reward)

**核心逻辑：**
基于一阶泰勒展开，将单步全局 Loss 的下降比例 ($\Delta L / L_{t-1}$) 线性拆解为每个独立张量的物理贡献。

**计算公式：**
对于第 $i$ 个张量，单步的原始局部奖励定义为：
$$r_{local, i} = \frac{- (\nabla_{\theta_i} L)^T \Delta \theta_i}{L_{t-1}}$$

* $\nabla_{\theta_i} L$：当前 step 张量 $i$ 的梯度。
* $\Delta \theta_i$：张量 $i$ 在当前 step 的实际更新步长。该步长受调度器网络输出的学习率 $\eta_i$ 强控（例如在 Adam 中，$\Delta \theta_i = \eta_i \cdot \frac{m_i}{\sqrt{v_i} + \epsilon}$）。
* $L_{t-1}$：上一步的全局 Loss，用作分母消除前期大 Loss 和后期小 Loss 的量级差异，强制将 $r_{local, i}$ 约束在“Loss 相对变化率”的量纲下。

**工程抗噪策略（EMA 平滑）：**
为防止 DataLoader 随机采样带来的 batch 级高频梯度噪声击穿 Critic 网络，必须在环境层面对计算出的原始奖励进行指数滑动平均处理：
$$r_{local, i}^{(ema)} = \gamma \cdot r_{local, i}^{(ema)} + (1 - \gamma) \cdot r_{local, i}$$
*注：$\gamma$ 为平滑系数，通常取 0.9 左右。后续参与计算的局部奖励均指平滑后的 $r_{local, i}^{(ema)}$。*

---

## 2. 综合奖励设计 (Final Reward)

**核心逻辑：**
将微观的物理贡献（Local）与宏观的训练状态（Global）直接线性加权，作为 PPO 算法中各张量的真实反馈信号。

**计算公式：**
针对第 $i$ 个张量，其最终环境奖励为：
$$r_{final, i} = \alpha \cdot r_{local, i}^{(ema)} + \beta \cdot r_{global}$$

* $\alpha, \beta$：权重超参数。得益于 $L_{t-1}$ 的引入，此时两项奖励在数值量级上已经天然对齐，避免了“大数吃小数”的问题。
* **训练策略建议：** 可采用动态权重。在探索初期调大 $\alpha$（依赖强局部信号快速启动策略），在训练中后期逐渐调大 $\beta$（依赖全局信号保证长期收敛稳定性）。

**信用分配收口机制：**
为了让 $r_{final, i}$ 准确指导对应的动作 $\eta_i$，Critic 网络必须输出长度与张量数量一致的价值预测向量 $[V_1, V_2, ..., V_n]$。
对于第 $i$ 个张量，其优势函数独立计算：
$$A_i = r_{final, i} - V_i$$
在计算 Actor 损失时，必须确保动作对数概率 $\log \pi(a_i)$ 与 $A_i$ 严格进行逐元素相乘（Element-wise multiply），避免不同张量间的信用污染。
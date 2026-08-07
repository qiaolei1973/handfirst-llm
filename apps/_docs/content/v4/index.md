# v4：优化曲线

> 你将学到：标准化（Z-score）、Momentum、RMSProp、Adam、训练集/验证集、过拟合、Early Stopping。

v3 让模型学会了画曲线——16 个 ReLU 神经元拼出 sin 的形状。但仔细看训练结果，还有两个问题：

1. **两端收敛慢**——sin 在靠近 0 和 2π 的地方，模型迟迟对不齐
2. **SGD 震荡**——每次 mini-batch 抽到的数据不一样，梯度东一下西一下

这一章要给 v3"收尾"——两条路，一个目标：**让训练更快、更稳**。

```
         管数据：标准化 — 让所有输入在同一尺度上
优化曲线
         管更新：更好的优化器 — 不止是"往梯度反方向走"

再加一个保险：泛化 — 怎么知道模型真的学会了，不是在背答案？
```

---

## 标准化：让所有输入在同一尺度

### 回顾 v2：中心化

v2 做了一件事——把 x 全部减去它们自己的平均值：

```
x_centered = x - mean(x)
```

效果：一半 x 为正，一半为负。W 的梯度不再"所有人朝一个方向推"——推力自己平衡了。

### v3 的新问题：参数多了，尺度不一致

v2 只有一个参数 W（加一个 bias）。中心化只改变了 x 的位置（均值变 0），没改变 x 的**尺度**（散布范围）。

这在 v3 成了问题。v3 的隐藏层有 N 个神经元，每个都有自己的 `w_j` 和 `b_j`。所有神经元都接收同一个 x。x 的尺度直接决定了所有 `w_j` 梯度的尺度：

```
∂L/∂w_j = ∂L/∂z_j · x

x 在 [0, 6.28] → 靠右的点的梯度是左边的 6+ 倍
```

神经元之间、数据点之间，梯度的"力度"天生不均——有些参数被推得很猛，有些几乎不动。收敛速度参差不齐，表现在最终结果上就是**两端收敛慢**。

### Z-score 标准化：统一位置 + 统一尺度

在中心化的基础上再除以标准差：

```
μ = (1/n) · Σ x_i
σ = √[(1/n) · Σ (x_i − μ)²]

z_i = (x_i − μ) / σ
```

两步合在一起：**先减均值，再除标准差**。结果：mean = 0, std = 1。

```typescript
const mean = features.reduce((s, v) => s + v, 0) / n;
const std = Math.sqrt(
  features.reduce((s, v) => s + (v - mean) ** 2, 0) / n
);
const standardized = features.map(x => (x - mean) / std);
```

![原始数据 → 中心化 → 标准化，三个阶段的分布对比](/v4/standardization-comparison.png)
<!-- gen_v4_images.py: #1 standardization-comparison — 三列直方图：raw(mean=3.14,std=1.8)→centered(mean=0,std=1.8)→standardized(mean=0,std=1)，标注"中心化只挪位置"vs"标准化统一尺度" -->

标准化之后，x 的取值范围从 `[0, 6.28]` 收束到大约 `[−2, 2]`。所有神经元的 `w_j` 梯度现在在同一个量级上竞争——**公平的梯度竞赛**。

> **中心化 vs 标准化**：中心化只平移，不缩放——解决了 v2 单参数场景的"推力同号"问题。标准化平移 + 缩放——额外解决了 v3 多参数场景的"梯度尺度不均"问题。单参数时中心化就够了，多参数时必须标准化。

---

## 更好的优化器：不只是"往梯度反方向走"

标准化管了数据端。但梯度算出来之后，**怎么用它更新参数**——这件事 v1 到 v3 完全没变过：

```typescript
W -= lr * gradW;   // 永远是这一行
```

这个最简单的更新规则有三个问题：

1. **没记忆**：每一步只看当前 batch 的梯度，之前的方向完全不记得。mini-batch 带来的随机抖动直接体现在参数更新上。
2. **一刀切 lr**：所有参数——不管梯度大还是小——用同一个学习率。梯度大的参数震荡，梯度小的参数爬不动。
3. **只靠 lr 控速度**：lr 是唯一能调的旋钮。大了震荡，小了太慢，没有中间地带的自动调节。

优化器的演进就是逐一解决这三个问题。

### Momentum：给梯度加惯性

**要解决的问题**：SGD 没有记忆，每一步被当前 batch 的 noisy gradient 带偏。

**做法**：不直接用当前梯度更新，而是维护一个"速度" `v`——梯度的指数移动平均。每一步，新梯度只贡献一部分，大部分来自历史积累：

```
v_t = β · v_{t-1} + (1 − β) · grad_t       ← 速度 = 历史惯性 + 新梯度
W   = W − lr · v_t                          ← 用速度更新，不是用原始梯度
```

`β`（通常 = 0.9）控制"记忆有多长"。β = 0.9 意味着当前梯度只占 10%，历史占 90%——大约最近 10 步的加权平均。

直觉：这就像一个球从山上滚下来——它不会东一下西一下，因为它有**惯性**。梯度方向一致时速度越来越大（加速下山），方向反复横跳时互相抵消（噪声被平滑掉）。

```typescript
// Momentum
this.vW = 0.9 * this.vW + 0.1 * gradW;
W -= lr * this.vW;
```

### RMSProp：每个参数自己的学习率

**Momentum 没解决的问题**：所有参数还是同一个 lr。有些参数（比如输出层的 w_out，对应的 h 值可能很大）天生梯度大，容易震荡；有些参数（比如深层神经元的 w，梯度经过 ReLU 截断后很小）几乎不更新。

**做法**：维护每个参数梯度的"近期波动量" `s`（二阶矩估计），用 `√s` 来**缩放每个参数的步长**：

```
s_t = β · s_{t-1} + (1 − β) · grad_t²     ← 梯度平方的指数移动平均
W   = W − lr · grad_t / √(s_t + ε)         ← 除以后，陡的慢走、缓的快走
```

`ε`（通常 = 1e-8）很小，只防止除以零。

直觉：
- 某个参数梯度一直很大 → `s` 大 → `√s` 大 → 步长被**压小**（防止震荡）
- 某个参数梯度一直很小 → `s` 小 → `√s` 小 → 步长被**放大**（加速学习）

每个参数有了**自己的等效学习率**，不再一刀切。

```typescript
// RMSProp
this.sW = 0.999 * this.sW + 0.001 * gradW * gradW;
W -= lr * gradW / (Math.sqrt(this.sW) + 1e-8);
```

### Adam：集大成者

**Adam** = **Ada**ptive **M**oment Estimation。把 Momentum（一阶矩）和 RMSProp（二阶矩）合在一起，再加上偏差修正：

```
m_t = β₁ · m_{t-1} + (1 − β₁) · grad_t         ← 一阶矩（Momentum）
v_t = β₂ · v_{t-1} + (1 − β₂) · grad_t²        ← 二阶矩（RMSProp）

m̂_t = m_t / (1 − β₁^t)                          ← 偏差修正
v̂_t = v_t / (1 − β₂^t)                          ← （解决早期步偏小）

W   = W − lr · m̂_t / √(v̂_t + ε)                 ← 力矩方向 / 尺度缩放
```

**偏差修正是干嘛的？** 训练刚开始时 `m_0 = 0, v_0 = 0`。前几步的 `m_t` 和 `v_t` 偏向 0——t=1 时，`m_1 = 0.1 × grad_1`（只有 10%，其余 90% 是 0）。除以 `(1 − β₁^t)` 把它放大回正确量级。随着 t 增大，修正项趋近于 1——自动淡出。

```typescript
// Adam
const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
this.t++;

this.mW = beta1 * this.mW + (1 - beta1) * gradW;       // 一阶矩
this.vW = beta2 * this.vW + (1 - beta2) * gradW ** 2;  // 二阶矩

const mHat = this.mW / (1 - beta1 ** this.t);           // 偏差修正
const vHat = this.vW / (1 - beta2 ** this.t);

W -= lr * mHat / (Math.sqrt(vHat) + eps);
```

**默认超参**：`lr = 0.001`，`β₁ = 0.9`，`β₂ = 0.999`，`ε = 1e-8`。这些值几乎不需要调——Adam 是业界默认首选。

### 对比：SGD vs Momentum vs Adam

![SGD、Momentum、Adam 在 sin 拟合任务上的 loss 下降对比](/v4/optimizer-comparison.png)
<!-- gen_v4_images.py: #2 optimizer-comparison — 8 神经元网络，三种优化器 2000 epoch 的 loss 曲线，Momentum 比 SGD 平滑，Adam 收敛最快 -->

- **SGD**：震荡明显，loss 下降最慢
- **Momentum**：惯性平滑了震荡，比纯 SGD 更快更稳
- **Adam**：收敛最快最稳——自适应的学习率 + 惯性，前几百步就降到了 SGD 跑几千步才到的位置

> Adam 不是"更好的 SGD 可选项"——是默认选项。几乎所有现代 ML 训练都用 Adam（或它的变体 AdamW）。从下一章开始，优化器默认用 Adam。

---

## 泛化：怎么知道模型真的学会了？

标准化 + Adam 让训练更快更稳。但还有一个根本问题没有回答：**模型是在"学习规律"还是在"背答案"？**

### v3 的隐患：参数比数据点多

v3 用 16 个 ReLU 神经元拟合 60 个点——模型有 16×2 + 1 = 33 个参数。参数越多，"拟合能力"越强——但也能用来"记住"每一个数据点的噪声，而不是学到背后的 sin 曲线。

### 训练集 & 验证集

要知道模型是在学还是在背，最简单的办法：**留一部分数据不训练**。

```
全部 60 个点 → 训练集 48 个（80%） + 验证集 12 个（20%）
```

- **训练集**：用来算梯度、更新参数——模型看得到
- **验证集**：不参与训练——模型看不到，只用来检验

```typescript
const splitIdx = Math.floor(n * 0.8);
const trainX = features.slice(0, splitIdx);
const trainY = labels.slice(0, splitIdx);
const valX = features.slice(splitIdx);
const valY = labels.slice(splitIdx);
```

两个 loss：`trainLoss`（在训练集上算）和 `valLoss`（在验证集上算）。两个都下降 → 模型在学。一个降一个升 → 模型在背。

### 过拟合（Overfitting）

训练 epoch 多了之后，会出现一种典型情况：

![训练 loss 持续下降但验证 loss 开始上升：过拟合的信号](/v4/overfitting.png)
<!-- gen_v4_images.py: #3 overfitting — 上：train loss + val loss 双曲线，标注过拟合区域和 early stopping 点；下：early-stopped 的预测 vs 过拟合的预测 -->

- **训练 loss**：一直下降——模型在努力拟合训练集里的每个点
- **验证 loss**：先降、**然后开始升**——模型开始"背诵"训练集的噪声，但噪声在验证集里不一样，所以验证 loss 反弹

这就叫**过拟合（Overfitting）**——模型把训练集里的随机噪声也当规律学了。参数太多 + 训练太久 = 过拟合。

### Early Stopping

最简单的防过拟合策略：**验证 loss 不再下降就停。**

```typescript
let bestValLoss = Infinity;
let patience = 0;
const PATIENCE_LIMIT = 200;  // 连续 200 epoch 没改善就停

for (let epoch = 0; epoch < maxEpochs; epoch++) {
  const { trainLoss, valLoss } = step();

  if (valLoss < bestValLoss) {
    bestValLoss = valLoss;
    patience = 0;
    saveCheckpoint();          // 保存当前最佳参数
  } else {
    patience++;
    if (patience >= PATIENCE_LIMIT) break;  // 停了
  }
}
```

`patience` 是"容忍度"——允许验证 loss 连续不改善的 epoch 数。太小可能误停（验证 loss 偶尔波动），太大则浪费时间。

---

## 合在一起：v4 的训练循环

v1 → v4，`step()` 一步步进化：

```typescript
step(): EpochEvent {
  // 1. 采样
  const batch = sampleBatch(trainSet, this._batchSize);

  // 2. 前向传播
  const h = this.hidden.forward(batch.features);    // Linear + ReLU
  const yPred = this.output.forward(h);             // Linear

  // 3. 算 loss（训练集 + 验证集）
  const trainLoss = mse(yPred, batch.labels);
  const valLoss = this.evaluate(valSet);             // NEW!

  // 4. 反向传播（v3）
  this.output.backward(/* ∂L/∂yPred */);
  this.hidden.backward(/* 上游梯度 */);

  // 5. 更新参数（用 Adam 替代 SGD）               // NEW!
  this.optimizer.step();   // Adam internally
}
```

v1 → v4 的变化：

| | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 损失 | MAE | MSE | MSE | MSE |
| 采样 | 全量 | SGD | SGD（全量也行） | SGD |
| 数据预处理 | 无 | 中心化 | 中心化 | **标准化** |
| 优化器 | `w -= lr*grad` | `w -= lr*grad` | `w -= lr*grad` | **Adam** |
| 模型 | `y=Wx+b` | `y=Wx+b` | 1→N→1 MLP | 1→N→1 MLP |
| 泛化保障 | 无 | 无 | 无 | **验证集 + Early Stopping** |

---

## 试试看

```bash
# 终端裸跑
pnpm exec tsx apps/v4/train.ts

# 浏览器实时可视化
pnpm dev:v4     # → http://localhost:3004
```

> 📸 **待截图**：打开 `http://localhost:3004`，训练完成，截 dashboard 全貌——包含训练/验证 loss 双曲线。

---

## 总结

v4 是"优化"的一章——不回头的优化（标准化管数据、Adam 管更新）、有保险的优化（验证集 + Early Stopping 防过拟合）：

| 概念 | 是什么 | 解决的问题 |
|---|---|---|
| 标准化（Z-score） | `z = (x−μ)/σ`，统一位置 + 尺度 | 多参数下梯度尺度不均 → 两端收敛慢 |
| Momentum | 梯度的指数移动平均，加惯性 | SGD 震荡、没方向记忆 |
| RMSProp | 每个参数自己的缩放因子 | 一刀切 lr，陡的震荡缓的爬不动 |
| Adam | Momentum + RMSProp + 偏差修正 | 以上所有问题，默认首选 |
| 训练/验证集 | 留一部分数据不训练 | 怎么知道模型是在学不是背？ |
| 过拟合 | 训练 loss↓，验证 loss↑ | 参数太多 → 背噪声 |
| Early Stopping | 验证 loss 不降就停 | 过拟合的简单解法 |

v1 到 v4，所有操作都是基于 `number` 的——每个参数是标量，`∂L/∂w` 是偏导数。这四章覆盖了机器学习最核心的概念基础。

在进入矩阵之前，先停下来，把所有概念串成一张地图。

> 继续阅读：[复习：机器学习是什么](/recap)

# v2：进入机器学习的世界

v1 里你完成了一次完整的训练——从瞎猜到拟合。那些朴素的操作，在机器学习里都有自己的名字。这一章先**给 v1 的操作起名**，再用这些名字来**优化 v1**。

---

## v1 的操作在机器学习里叫什么

### 模型（Model）

`y = W·x + bias` 这一行公式，在 ML 里叫**模型**。W 和 bias 叫**参数**（Parameters）。

模型 = 公式，参数 = 公式里可以调的数。

```typescript
// 这就是你的第一个模型——线性模型
const linear = (x: number, W: number, bias: number) => x * W + bias;
```

### 损失函数（Loss Function）

把每个点的差距加起来取平均——这个**总差距**就叫**损失（Loss）**。算损失的公式叫**损失函数**。

v1 用的是绝对值取平均，官方名字叫 **MAE**（Mean Absolute Error，平均绝对误差）：

```
预测值 yPred  = W·x + bias
差距    diff  = yPred - y
MAE:    L     = (1/n) · Σ |diff|
```

### 梯度（Gradient）

计算机不是靠试错来猜方向的。它用微积分算：把损失函数对参数求导，就知道"参数变大一丢丢，损失是变大还是变小"。这个导数叫**梯度**。

v1 对 MAE 求导：

```
∂L/∂W    = (1/n) · Σ sign(diff) · x
∂L/∂bias = (1/n) · Σ sign(diff)
```

求导后绝对值被消掉了，只剩下正负号。所以 MAE 的梯度**只看方向，不看距离**——差 100 和差 0.01，推得一样狠。

![MAE 的梯度：sign(diff)——只看方向不看距离](/v1/mae-gradient.png)
<!-- gen_v1_images.py: #8 mae-gradient — 右半边 grad=sign(diff)，红 diff>0→+1，蓝 diff<0→−1，标注差100/差0.01推力都是±1 -->

### 参数更新 & 学习率（Learning Rate）

算出梯度后，沿着梯度的**反方向**走一步——损失就会变小。这叫**参数更新**。

走多大步由**学习率**控制。v1 的学习率是 0.1——只取梯度的 10%：

```typescript
W    -= 0.1 * gradW;
bias -= 0.1 * gradBias;
```

### 训练 & 梯度下降（Gradient Descent）

猜 → 算损失 → 算梯度 → 更新参数 → 再猜 → … 这个循环叫**训练（Training）**。每一轮叫一个 **epoch**。

其中"更新参数"这一步叫**梯度下降（Gradient Descent）**：

```
拿到梯度 → w = w - 学习率 × 梯度
```

梯度下降只描述"怎么更新参数"，不关心梯度是怎么来的——全量算的还是随机抽的，它不关心。

> 训练 = 外层循环。梯度下降 = 内层那一行 `w -= lr * grad`。

![Dashboard 全景](/v1/training-done.png)
<!-- dashboard 截图：v1 训练完成全貌 -->

---

---

## 优化 v1

### MSE（Mean Squared Error）—— 让梯度带上距离

在 v1 中，梯度永远在 ±1 之间跳动——差 100 和差 0.01，对参数的推动力完全一样。回头看上一节的梯度图：

![MAE 损失函数和符号梯度](/v1/mae-gradient.png)
<!-- gen_v1_images.py: #8 mae-gradient — 复用 v1 的图说明问题 -->

原因出在 MAE 本身：`L = (1/n) · Σ |diff|`，求导之后绝对值被消掉，只剩 `sign(diff)`。**MAE 的梯度和距离脱钩了。**

要解决这个问题，需要一种"差得越远、罚得越重"的损失函数——**MSE**（Mean Squared Error，均方误差）：

```
MSE:  L = (1/n) · Σ (yPred - y)²
```

用平方替代绝对值。对 MSE 求导：

```
∂L/∂W    = (2/n) · Σ (yPred - y) · x
∂L/∂bias = (2/n) · Σ (yPred - y)
```

`diff` 直接出现在梯度里。和 MAE 的 `sign(diff)` 不一样——**误差大梯度就大，误差小梯度自然减小**。

```typescript
// v2: MSE——diff 自带距离信息
gradW += 2 * diff * x;
gradBias += 2 * diff;
```

![MAE vs MSE loss curves and gradients](/v2/loss-comparison.png)
<!-- gen_v2_images.py: #1 loss-comparison — 2x2: loss 曲线 + 梯度对比 + 放大 + 公式总结 -->

左图是两种损失函数的曲线——MAE 在零点有一道尖锐的折角，MSE 是光滑的抛物线。右图是它们的梯度：MAE 的梯度是平的（永远 ±1），MSE 的梯度随 diff 线性增长。底行放大看接近零点——MSE 的梯度在那里自己小下来，自然减速。

![bias 在 MSE 下收敛更快](/v1/loss-comparison.png)
<!-- gen_v2_images.py: #4 — L(bias) 曲面，MAE 等步长 vs MSE 自动减速 -->

这张图更直观：同一个起点 (bias=0)，MAE 每一步都一样大，到了谷底附近还在猛冲；MSE 越靠近谷底步子越小，稳稳收敛。

> ⚠️ MSE 梯度量级比 MAE 大很多（`2×diff` vs `±1`），**学习率从 0.1 降到 0.01**，不然参数会震荡。

---

### SGD（Stochastic Gradient Descent）—— 用随机性探索

v1 里每个 epoch 扫完所有 12 个点，在全部数据上算梯度。这条路是确定的——每次训练走的路径一模一样，没有随机性帮你发现更好的方向。

梯度下降只管 `w -= lr * grad`，不关心梯度怎么来的。**采样和更新是两回事：**

```
采样方式              更新方式
────────              ────────
全量                 梯度下降（GD）                      ← v1
随机 Mini-batch      动量（Momentum）                    （后续讲）
单样本               Adam                               （后续讲）
```

把左边换成**随机 Mini-batch**——每次只抽一小撮（batch=8）：

```typescript
const batch = sampleBatch({ features: X, labels: Y }, 8);
```

只在这 8 个点上算梯度、更新。下一轮再随机抽 8 个——每次看到的子集都不一样。参数的更新方式还是 `w -= lr * grad` 没变，但梯度的**来源**带上了随机性。

`Stochastic = 随机采样`，`Gradient Descent = 梯度下降`。两者独立，合起来就是 **SGD**（随机梯度下降）。

![SGD loss 曲线有抖动但整体下降](/v2/sgd-fluctuation.png)
<!-- gen_v2_images.py: #2 sgd-fluctuation — BGD 平滑 + SGD 抖动但趋势相同 -->

SGD 的 loss 每一步都在抖——抽到的 8 个点不一样，梯度也不一样。但趋势整体向下。抖动是代价，探索是收获。

---

### 中心化（Centering）—— 让梯度推力自己平衡

在 v1 中，x 全部 ≥ 0（从 0 到 20）。W 的梯度里有一项 `sign(diff) * x`，`x` 永远正——**所有数据点朝同一个方向推 W**。推过头了也没有反方向的力量把参数拉回来。

把 x 全部减去它们自己的平均值：

```
x_centered = x - mean(x)
```

变换后一半 x 是正，一半是负。W 的梯度现在有正的推力，也有负的推力——**自己就平衡了**。

```typescript
const mean = features.reduce((s, v) => s + v, 0) / n;
features = features.map(x => x - mean);
```

![中心化前后数据分布对比](/v2/centering.png)
<!-- gen_v2_images.py: #3 centering — 上图=原始 x 全在右侧, 下图=中心化后以 0 对称 -->

上图是原始数据——所有点挤在 0 右侧，推力永远同号。下图是中心化之后——以 0 为对称轴分布，推力和反推力各占一半。

> 这种预处理只减了均值，没除标准差，所以叫"中心化"就够了。标准化（再除以标准差）和归一化（缩放到 [0,1]）是另外两种预处理，v2 不需要。

---

## 合在一起：v2 的完整 step()

```typescript
step(): EpochEvent {
  const batch = sampleBatch({ features: X, labels: Y }, this._batchSize);

  let gradW = 0, gradBias = 0, batchLoss = 0;
  for (const { feature: x, label: y } of batch) {
    const yPred = x * this.params.W + this.params.bias;
    const diff = yPred - y;
    gradW += 2 * diff * x;         // MSE 梯度：带距离
    gradBias += 2 * diff;          // MSE 梯度：带距离
    batchLoss += diff * diff;
  }
  gradW /= this._batchSize;
  gradBias /= this._batchSize;

  this.params.W    -= this._lr * gradW;
  this.params.bias -= this._lr * gradBias;
}
```

v1 → v2：

| | v1 | v2 | 为什么改 |
|---|---|---|---|
| 损失 | MAE | **MSE** | 梯度带距离，自动加/减速 |
| 采样 | 全量 | **SGD** | 随机性带来的探索能力 |
| 数据 | 原始 x | **中心化** | 梯度推力自己平衡 |

---

## 试试看

```bash
# 终端裸跑
pnpm exec tsx apps/v2/train.ts

# 浏览器实时可视化
pnpm dev:v2     # → http://localhost:3002
```

> 📸 **待截图**：打开 `http://localhost:3002`，跑完 600 epoch，截 dashboard 全貌。

---

## 总结

给了名字之后，用名字去理解并解决问题。v2 的模型仍然是一条直线 `y = W·x + bias`——能处理的只有直线数据。下一个问题：**如果数据不是直线，模型该怎么画曲线？** 那就是 v3 要聊的了。

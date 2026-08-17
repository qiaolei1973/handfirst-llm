# v5：矩阵 — 同时支持多特征和批处理

> 你将学到：多维特征、权重矩阵、批处理、矩阵乘法、3D 损失曲面。

v1 到 v4 的模型都在处理**一个特征**——输入是单个 x，输出是 ŷ。但真实世界的数据从来不止一个特征：房价取决于面积、楼层、朝向、周边学校……光知道面积不够。

这一章引入机器学习中最重要的数据结构：**矩阵**。它带来两个好处：

1. **多特征**：输入从 1 个数变成 d 个数的向量，权重自然从向量变成矩阵
2. **批处理**：一次处理 B 个样本，输入从向量变成矩阵，一次矩阵乘法算完

两者是同一个道理——矩阵就是"把多个向量整齐排在一起"，区别只是排的是"权重"还是"数据"。

---

## 真实数据不止一个特征

v4 的房价回归只是一个简化版——数据集只有一个特征 `x`（归一化的位置），标签 `y` 是房价。但真正的房价数据是这样的：

```
| 面积(m²) | 卧室数 | 楼层 | 到地铁距离(min) | ... | 房价(万) |
|----------|-------|------|---------------|-----|---------|
| 85       | 2     | 5    | 8             | ... | 420     |
| 120      | 3     | 12   | 15            | ... | 680     |
| 65       | 1     | 3    | 3             | ... | 310     |
| ...      | ...   | ...  | ...           | ... | ...     |
```

每一行是一个样本，每一列是一个**特征（Feature）**。如果有 d 个特征，每个样本就是一个 d 维向量：

```typescript
// v4: 一个样本 = 一个数
const x = 0.73;

// v5: 一个样本 = 一个向量
const x = [85, 2, 5, 8, ...]; // d 个特征
```

v4 的模型结构 `1 → N → 1`（1 个输入 → N 个隐藏神经元 → 1 个输出），到了 v5 就变成 `d → N → 1`。

---

## 矩阵是什么

在看到 `d → N → 1` 的数学之前，先停下来理解**矩阵到底是什么**。这不是数学课，但 30 秒的直觉比 30 分钟的公式有用得多。

### 矩阵 = 数据表

矩阵就是你已经在用的 Excel 表格——行 × 列，放进一个方括号里：

![数据矩阵：行=样本，列=特征，每行是一个房子的完整数据](/v5/matrix-data-table.svg)

日常直觉：**矩阵 = 组织好的数据**。你想查"第二个样本的卧室数"→ 行 2 列 2。这就是 `X[2][2] = 3`。

### 权重矩阵 = 连接网络

神经网络里，权重矩阵是最核心的概念。v4 的隐藏层有 N 个神经元，每个神经元有一个 `w_i`（权重）和一个 `b_i`（偏置）。**当输入从 1 个变成 d 个时，每个神经元需要 d 个权重**——每个特征配一个权重：

![权重矩阵 W_h[N x d]：每个格子是一个权重值，横向读=一个神经元，纵向读=一个特征](/v5/matrix-weight-grid.svg)

两种读法：
- **横向（一行）**：一个神经元看向**所有特征**的方式
- **纵向（一列）**：一个特征连接到**所有神经元**的方式

`w₁₂ = 0.8`（深色）= "卧室数"这个特征对 1 号神经元很重要。`w₂₃ = -0.3`（浅色）= "楼层"对 2 号神经元是负影响。

### 向量 × 矩阵 = 加权求和

一个样本 `x = [面积, 卧室数, 楼层]` 输入隐藏层时发生了什么？

![矩阵乘法：x（输入向量）乘 W（权重矩阵）得到 z（每个神经元的激活值）](/v5/matrix-multiply.svg)

每个特征乘上自己那一列的权重，全加起来，加偏置，过 ReLU。N 个神经元 = N 次这样的加权求和。

**这就是矩阵乘法 `z = W_h @ x + b_h` 的物理含义**：不是"先乘后加"这个操作，而是"每个神经元独立地对每个特征打分，然后汇总"。

### 矩阵 × 矩阵 = 批处理

上面是一个样本 `x`（d 维向量）怎么算。但训练时我们不是一次只喂一个样本——v4 已经用了 mini-batch，每次抽 B 个样本。与其写一个 `for` 循环逐个算，不如把 B 个样本**并排堆成一个矩阵**：

```
单个样本 x [d × 1]:      批处理 X [d × B]:
  [ 面积 ]                  [ 面积₁  面积₂  ...  面积_B ]
  [ 卧室数 ]                [ 卧室₁  卧室₂  ...  卧室_B ]
  [ 楼层 ]                  [ 楼层₁  楼层₂  ...  楼层_B ]
```

每个样本是一列。B 个样本就是 B 列。然后一次矩阵乘法：

```
Y = W_h @ X + b_h        [N × d] @ [d × B] = [N × B]
```

**完全同一个公式**——`Y = W @ X`，只是 `X` 从 1 列变成了 B 列。矩阵乘法天然同时算 B 列，每一列的输出就是对应样本的隐藏层激活值。

反向传播也一样。梯度在 batch 维自动累加：

```
∂L/∂W = ∂L/∂Y @ Xᵀ       B 个样本的梯度贡献自动加进矩阵乘法里
```

这就是**批处理（Batching）**的价值：不是新算法，只是把"逐个样本的循环"变成"一次矩阵运算"。代码里 `for (sample of batch)` 这个循环消失了，取而代之的是堆矩阵 + 一次 `forward`。

---

## 从 1→N→1 到 d→N→1

v4 里，模型是这样：

```
x (标量) → 隐藏层 (N 个神经元) → ŷ (标量)
  w_i: 标量（每个神经元一个权重）
```

v5 里，x 从标量变成向量，权重从标量变成向量：

```
x (d维) → 隐藏层 (N 个神经元) → ŷ (标量)
  W_h: 矩阵 [N × d]（每个神经元 d 个权重）
```

数学公式看起来变了，但其实没变：

| | v4 | v5 |
|---|---|---|
| 隐藏层前向 | `h_i = ReLU(w_i · x + b_i)` | `h_i = ReLU(w_i₁·x₁ + ... + w_i_d · x_d + b_i)` |
| 权重 | `w_i`（一个数） | `w_i₁, ..., w_i_d`（d 个数，一个向量） |
| 全部神经元权重 | `w`（N 个数的向量） | `W`（N×d 个数的矩阵） |

代码层面，`Linear` 层的 `inDim` 从 v3/v4 的 `1` 改成 `d` 就行。

---

## 代码：矩阵在 v5 里长什么样

概念讲完了，落到代码。v5 相对 v4 多了两样东西：`Mat`（矩阵数据结构），以及 `DataLoader.generate()` 返回矩阵而不是样本数组。`Linear` 的 forward/backward 也随之全部改成矩阵运算。

### Mat —— 「行 × 列」加一段连续数据

```typescript
class Mat {
  data: Float64Array;   // 所有元素拍平成一段
  rows: number;
  cols: number;
}
```

矩阵不引入新的数学对象，只是把一堆数按「行 × 列」排好。`data[i * cols + j]` 就是第 i 行第 j 列。v5 训练只用这几个操作：

| 方法 | 含义 | 在 v5 里用在哪 |
|---|---|---|
| `a.matmul(b)` | 矩阵乘 `A @ B` | `W @ X`（前向） |
| `a.transpose()` | 转置 `Aᵀ` | `Xᵀ`、`Wᵀ`（反向） |
| `a.dotmul(b)` | 逐元素乘（哈达玛积） | `(ŷ-y)²` |
| `a.scale(s)` | 逐元素乘标量 | `2(ŷ-y)/B` |
| `a.sum(dim)` | 求和（无参=总和，`1`=按行） | loss（`sum()`）、bias 梯度（`sum(1)`） |
| `a.add(b)` / `a.sub(b)` | 加 / 减（带广播） | `W@X + b`、`ŷ - Y` |

两个容易混的：

- **`dotmul` 不是矩阵乘。** `A @ B` 是「行 × 列」的信息融合；`A.dotmul(B)` 是「对应位置直接乘」，算 `diff²` 用的就是它。
- **`add` / `sub` 带广播。** `weight.matmul(x)` 是 `[outDim × B]`，`bias` 只有 `[outDim × 1]`——`add` 自动把 bias 复制到每一列，省掉 `for` 循环。

### DataLoader.generate() —— 样本排成矩阵

v3/v4 的 `generate()` 返回样本数组，要 `for` 循环逐个算；v5 改成返回矩阵：

```typescript
generate(): { X: Mat; Y: Mat } {
  const indices = [...Array(n).keys()].sort(() => Math.random() - 0.5).slice(0, m);

  const X = new Mat(this._dim, m);   // 输入 [d × B]
  const Y = new Mat(1, m);           // 标签 [1 × B]
  for (let b = 0; b < m; b++) {
    const f = this.dataset.features[indices[b]];
    for (let i = 0; i < this._dim; i++) X.data[i * m + b] = f[i];   // 第 b 列 = 第 b 个样本
    Y.data[b] = this.dataset.labels[indices[b]];
  }
  return { X, Y };
}
```

关键在 `X.data[i * m + b] = f[i]`：**每一列是一个样本**。d 个特征、B 个样本，排成一个 `[d × B]` 矩阵；标签 Y 是一行 `[1 × B]`。

从这里开始，「逐个样本的 `for` 循环」就消失了——样本被堆进矩阵，后面全是矩阵运算。

### Linear —— 矩阵版 forward / backward

forward 就是一行 `W @ X + b`：

```typescript
forward(x: Mat): Mat {
  this._x = x;                                          // 记住输入，backward 要用
  const weight = new Mat(this.outDim, this.inDim, this.weight.data);
  const bias   = new Mat(this.outDim, 1, this.bias.data);
  return weight.matmul(x).add(bias);                    // W @ X + b
}
```

backward 是三条链式法则，全用矩阵运算：

```typescript
backward(gradOut: Mat): Mat {                           // 上游梯度 ∂L/∂Y，[outDim × B]
  const x = this._x;

  // ① ∂L/∂W = ∂L/∂Y @ Xᵀ —— B 个样本的梯度贡献，自动累加进矩阵乘法
  const gradWMat = gradOut.matmul(x.transpose());
  for (let i = 0; i < gradWMat.data.length; i++) this.weight.grad[i] += gradWMat.data[i];

  // ② ∂L/∂b = ∂L/∂Y 沿 batch 维求和
  const gradBMat = gradOut.sum(1);
  for (let j = 0; j < this.outDim; j++) this.bias.grad[j] += gradBMat.data[j];

  // ③ ∂L/∂X = Wᵀ @ ∂L/∂Y —— 传给上一层
  const weight = new Mat(this.outDim, this.inDim, this.weight.data);
  return weight.transpose().matmul(gradOut);
}
```

对比 v4 逐个样本 `for` 累加梯度，这里 ①②③ 三行矩阵乘法就完成。「**B 个样本的梯度累加**」这个动作，被矩阵乘法里的 `Σ_k A[i][k]·B[k][j]` 内置了——这才是批处理的真正收益，也是这一章引入矩阵的原因。

### 合在一起：v5 的完整 _step()

> `_step()` 是基类 `Trainer` 的钩子：`step()` 调它并记一次 epoch，所以训练代码写的是 `_step()`。

```typescript
protected _step() {
  // 1. 采样：随机一批样本，堆成 [d×B] 输入 + [1×B] 标签
  const { X, Y } = this._train.generate();
  const B = X.cols;

  // 2. 前向：W_h @ X → ReLU → W_o @ H，一次算完 B 个样本
  this.model.zeroGrad();
  const yPred = this.model.forward(X);                  // [1 × B]

  // 3. 训练 loss：MSE = Σ(ŷ-y)² / B
  const diff = yPred.sub(Y);
  const trainLoss = diff.dotmul(diff).sum().data[0] / B;

  // 4. 反向：∂L/∂ŷ = 2(ŷ-y)/B，梯度在 batch 维自动累加
  this.model.backward(diff.scale(2 / B));

  // 5. 更新参数（Adam）
  this._opt.step();

  // 6. 验证集：全量评估一次
  const { X: vX, Y: vY } = this._val.generate();
  const vPred = this.model.forward(vX);
  const vDiff = vPred.sub(vY);
  const valLoss = vDiff.dotmul(vDiff).sum().data[0] / vX.cols;

  return {
    trainLoss: Number(trainLoss.toFixed(6)),
    valLoss: Number(valLoss.toFixed(6)),
  };
}
```

和 v4 的 `_step()` 并排看，唯一的结构差异：v4 里 `for (sample of batch)` 逐个样本循环，v5 里变成 `generate()` 返回矩阵 + 三次矩阵运算（`forward` / `sub` / `backward`）。循环没了，每一步做的事完全一样。

---

## d=2：可视化"曲面"

v3 的可视化依赖于 x 只有一维——我们可以在 2D 平面上画"折线拟合曲线"。到了 v5，输入是二维的，可视化变成 3D 曲面。

### 造一个曲面

给定 `f(x₁, x₂) = sin(√(x₁² + x₂²))`，加噪声造 200 个样本。

![真实曲面：f(x₁,x₂) = sin(√(x₁²+x₂²)) 的 3D 图 + 带噪声的采样点](/v5/true-surface.png)

这是一个从中心向外辐射的"涟漪"。

### ReLU 的 3D 形态

v3 里，一个 ReLU 是"一段折线"。到 v5，一个神经元的输出变成**一个折面**：

```
h_j = ReLU(w_j₁ · x₁ + w_j₂ · x₂ + b_j)
```

`w_j₁·x₁ + w_j₂·x₂ + b_j = 0` 是一条分割线——一边是 0，另一边向上或向下倾斜。

![2×2：四种 w₁, w₂, b 组合下的 ReLU 曲面](/v5/relu-surfaces.png)

不同的权重组合产生不同方向、不同坡度的折面。N 个神经元 = N 个不同折面 → 加起来逼近任何曲面。

---

## 训练

损失函数和优化器沿用 v4：MSE + Adam + Early Stopping + 80/20 训练/验证分离。

![训练后模型预测曲面 vs 真实曲面](/v5/trained-surface.png)

![Loss 下降曲线](/v5/loss-curve.png)

### Dashboard

和 v3/v4 类似的实时训练监控，但这次展示的是 3D 曲面。

---

## 总结

| 概念 | v4 | v5 |
|---|---|---|
| 输入 | x（标量，1 维） | x（向量，d 维）→ X（矩阵，d×B） |
| 隐藏层权重 | w（N 个数的向量） | W（N×d 矩阵） |
| 前向 | `ReLU(w_i·x + b_i)` 逐样本 | `ReLU(W @ X + b)` 一次算 B 个 |
| 梯度累加 | 逐个样本累加 | 矩阵乘法自动累加 |
| 可视化 | 2D 折线 | 3D 曲面 |

**矩阵不是新的数学对象——它就是"把多个向量整齐排列在一起"的结果。** 多特征让权重排成矩阵（W 的每一行是一个神经元的 d 个权重），批处理让数据排成矩阵（X 的每一列是一个样本）。同一个 `Y = W @ X`，两种好处。

---

> 继续阅读：[v6：实践](/v6)

# v3：画曲线 — 让模型学会弯曲

> 你将学到：激活函数、ReLU、神经元、隐藏层、输出层、前向传播、反向传播、MLP、Layer 抽象。

v1 和 v2 都在画直线——模型是 `y = Wx + b`，一条直线走到底。但真实世界的数据不总是直的。这一章要解决一个新问题：**让模型学会画曲线。**

---

## 直线不够用

60 个点，背后的规律是 `y = sin(x)`，加了一点噪声：

![60 个散点 + sin 虚线 + MSE 最优直线](/v3/line-vs-sin.png)
<!-- gen_v3_images.py: #1 line-vs-sin — 蓝色散点，绿色 sin 虚线，红色最优直线，标注"直线永远画不出曲线" -->

不管怎么调 W 和 bias，`y = Wx + b` 画出来的永远是一条直线。sin 有上有下——直线天生做不到。

问题的根源不是参数不够，而是**模型里只有线性变换**。要给模型加入"弯折"的能力。

---

## 激活函数：打破线性

v1 和 v2 里写了无数遍的 `W·x + b`——输入乘上权重加偏置——这个操作在 ML 里叫**线性变换（Linear）**。它只做三件事：缩放、旋转、平移。不管做多少次，合在一起还是线性：

```
一层 Linear:   y = W₁x + b₁
两层 Linear:   y = W₂(W₁x + b₁) + b₂
              = (W₂W₁)x + (W₂b₁ + b₂)    ← 还是一条直线！
```

**两层 Linear 叠在一起，等价于一层 Linear。** 堆多少层都没用——永远画不出曲线。

必须在线性变换之间，插入一个**非线性函数**——这东西叫**激活函数（Activation Function）**。

常见的激活函数有 Sigmoid、Tanh、ReLU……我们选最简单的那个——**ReLU**（Rectified Linear Unit，修正线性单元）：

```
        ⎧ 0    z ≤ 0
ReLU(z) ⎨
        ⎩ z    z > 0
```

z ≤ 0 时输出 0，z > 0 时输出 z 本身。翻译成代码只需要一行：

```typescript
Math.max(0, w * x + b); // ReLU(w·x + b)：线性变换 → 过 ReLU
```

`w` 和 `b` 控制折点的位置和方向：

- 折点 `x = −b/w`——ReLU 在这里从 0 开始激活
- `w > 0`：折后向上；`w < 0`：折后向下
- `|w|` 越大：折后越陡

![四种 w,b 组合下 ReLU 的折线形态](/v3/relu-shapes.png)
<!-- gen_v3_images.py: #2 relu-shapes — 2×2：四种 w,b 配出不同折点位置和折后斜率，虚线标记折点 -->

看这四种——同一个 `ReLU(w·x + b)`，w,b 不同，折线的形态完全不同。折点可以在任何位置，可以向上折也可以向下折，可以陡也可以缓。

---

## 神经元：一个计算单元

有了激活函数，下一步是把它包装成一个可复用的计算单元。

机器学习里的"**神经元（Neuron）**"这个词，是从生物神经元借来的比喻：

```
生物神经元:
  树突（接收信号）→ 细胞体（加总）→ 超过阈值？→ 轴突（输出脉冲）

人工神经元:
  输入 xᵢ           → Σ wᵢ·xᵢ + b   → 激活函数   → 输出
```

生物神经元接收到足够强的信号才会"发放"（fire）；人工神经元也一样——加权求和的结果，过激活函数，决定"放不放、放多少"。对 ReLU 来说：z > 0 才放行，z ≤ 0 不给过。

所以**一个神经元就是两样东西的组合**：

```
神经元 = 加权求和 (Linear) + 激活函数 (ReLU)

一个 ReLU 神经元:  output = ReLU(w·x + b)
                    ↑        ↑
                激活函数    加权求和
```

一个神经元 = 一段折线。换不同的 w,b，就能控制这段折线在哪里折、怎么折。

但一段折线远远不够——sin 曲线上上下下，需要很多个神经元，各自负责一段。

---

## 隐藏层：把神经元组织起来

把 N 个神经元并排放置，都接收同一个输入 x，各自算出自己的输出。这一层夹在输入和最终输出之间，外界看不见它的值——所以叫**隐藏层（Hidden Layer）**。

每个神经元独立做 Linear + ReLU。隐藏层的输出是一个长度为 N 的向量——每个元素是一个神经元的激活值。

最后用一层**不带激活函数**的 Linear 把 N 个值加权求和，得到最终预测：

> **为什么权重必须随机初始化？** 如果所有神经元的 `w` 和 `b` 初始值相同，它们在前向时输出一样、反向时梯度一样、更新后还是一样——N 个神经元退化成一个。用小随机数初始化，每个神经元的"出发点"不同，它们才能各自学到不同的折线。

![神经元、隐藏层、Linear、ReLU 之间的结构关系](/v3/architecture.svg)
<!-- gen: excalidraw v3/architecture.excalidraw → v3/architecture.svg -->

注意区分：
- **一个神经元** = 一次 Linear + 一个 ReLU
- **隐藏层** = N 个神经元 = N 次 Linear + N 个 ReLU
- **输出层** = 一次 Linear，不加激活

![2/4/8/16 个 ReLU 神经元分别拟合 sin 的效果](/v3/neurons-approximate.png)
<!-- gen_v3_images.py: #3 neurons-approximate — 2×2：不同神经元数训练后的预测曲线 vs sin，灰色虚线是每个神经元的贡献 -->

- **2 个神经元**：两段折线，大致勾出 sin 轮廓
- **4 个神经元**：四段折线，开始有 sin 的样子
- **8 个神经元**：八段折线，细节越来越准
- **16 个神经元**：几乎贴合 sin——Loss 从 0.16 降到 0.02

神经元越多 → 折线越多 → 逼近能力越强。

这个结构——输入 → 隐藏层 → 输出层——在机器学习里叫**多层感知机（Multilayer Perceptron，MLP）**。"多层"是指有一层隐藏 + 一层输出，v3 是两层的 MLP。虽然简单，已经能逼近任意曲线。

---

## Layer：把重复的模式打包

回头看隐藏层和输出层：

| | 隐藏层 | 输出层 |
|---|---|---|
| 前向 | 输入 → **Linear**(Wx+b) → **ReLU** → 输出 | 输入 → **Linear**(Wx+b) → 输出 |
| 反向 | 接收上游梯度 → 算自己的梯度 → 传给下游 | 接收上游梯度 → 算自己的梯度 → 传给下游 |

结构完全一样——都是"Wx+b → (激活) → 输出"。唯一不同：隐藏层的激活是 ReLU，输出层没有激活（或者说激活就是"直通"）。

既然模式一样，先抽成 **Layer 类**：

```typescript
class Layer {
  constructor(inputDim: number, outputDim: number, activation: 'relu' | 'linear')

  forward(x: number[]): number[]       // 前向传播
  backward(gradOut: number[]): number[] // 反向传播
}
```

隐藏层 = `new Layer(1, N, 'relu')`，输出层 = `new Layer(N, 1, 'linear')`。网络 = 两个 Layer 串起来。

**前向传播（Forward Pass）** 就是一层一层往前算——v1、v2 里已经做了无数次了：

```typescript
// 前向传播：x → hidden → h → output → yPred
const h = hidden.forward([x]);       // [1] → [N]   Linear + ReLU
const yPred = output.forward(Array.from(h))[0];  // [N] → [1]   Linear
```

下面重点讲反向传播——梯度是怎么一层一层传回去的。

---

### 反向传播：链式法则，一步一步往回传

v1 和 v2 里算梯度很简单——模型只有一层 Linear，对 loss 求导直接得到 `∂L/∂W` 和 `∂L/∂bias`。

但 v3 有两层。输出层的参数（w_out、b_out）好办——它们紧挨着 loss。**隐藏层的参数（w_i、b_i）和 loss 中间隔着 ReLU 和输出层**——不能直接求导。

答案是一条两百年前的微积分定律：**链式法则（Chain Rule）**。

```
如果 L 通过 yPred 依赖 h，又通过 h 依赖 w_i：
    ∂L     ∂L     ∂yPred    ∂h
    ───  =  ──── · ────── · ───
    ∂w_i    ∂yPred   ∂h      ∂w_i
```

误差从输出端开始，沿网络结构**反方向**一层一层流回去。下面是计算图视角——圆是值（x、z、h、ŷ、L），边上是局部偏导数，红色箭头是梯度流向。

---

**① 前向：算出每个节点的值**

![前向传播：x → z → h → ŷ → L](/v3/backprop-step-1.svg)
<!-- gen: excalidraw v3/backprop-step-1.excalidraw → v3/backprop-step-1.svg -->

这是 forward pass——从输入 x 一路算到 Loss。边上标注的是每步的局部偏导数：`∂z/∂x = W`、`∂h/∂z = ReLU'(z)`… 它们等下在链式法则里都要用到。

---

**② 反向第一步：从 Loss 求 ∂L/∂ŷ**

![反向 Step 2：∂L/∂ŷ = 2(ŷ−y)](/v3/backprop-step-2.svg)
<!-- gen: excalidraw v3/backprop-step-2.excalidraw → v3/backprop-step-2.svg -->

梯度从 L 出发，沿红色箭头回传。用 MSE：`L = (ŷ − y)²`，所以 `∂L/∂ŷ = 2(ŷ − y)`。

这是链式法则的起点——最粗的箭头，因为离 Loss 最近、信息最完整。

---

**③ 反向第二步：穿过输出层，∂L/∂h = ∂L/∂ŷ · w_out**

![反向 Step 3：链式法则穿过输出层](/v3/backprop-step-3.svg)
<!-- gen: excalidraw v3/backprop-step-3.excalidraw → v3/backprop-step-3.svg -->

输出层没有激活函数，`ŷ = Σ w_out_j · h_j + b_out`。链式法则：

```
∂L/∂w_out_j = ∂L/∂ŷ · h_j
∂L/∂b_out   = ∂L/∂ŷ
∂L/∂h_j     = ∂L/∂ŷ · w_out_j    ← 传给下游
```

> 直觉：`∂L/∂h_j` 是"改变 h_j 会对 loss 产生多大影响"。h_j 通过权重 w_out_j 影响 ŷ，所以梯度要乘上 w_out_j 往回传。

---

**④ 反向第三步：穿过 ReLU**

![反向 Step 4：穿过 ReLU 激活函数](/v3/backprop-step-4.svg)
<!-- gen: excalidraw v3/backprop-step-4.excalidraw → v3/backprop-step-4.svg -->

隐藏层：`h_j = ReLU(z_j)`，`z_j = w_j · x + b_j`。

ReLU 的导数很简单：

```
        ⎧ 1    z_j > 0    → 直通
ReLU'(z_j) ⎨
        ⎩ 0    z_j ≤ 0    → 截断
```

用链式法则：`∂L/∂z_j = ∂L/∂h_j · ReLU'(z_j)`。

> 如果该神经元没激活（z ≤ 0），`∂L/∂z_j = 0`——这个神经元的参数本次更新量为零。这就是"**死神经元**"的数学原因。

---

**⑤ 反向最后一步：隐藏层参数**

![反向 Step 5：∂L/∂W = ∂L/∂z · x](/v3/backprop-step-5.svg)
<!-- gen: excalidraw v3/backprop-step-5.excalidraw → v3/backprop-step-5.svg -->

最后一步，从 z_j 到 w_j 和 b_j：

```
∂L/∂w_j = ∂L/∂z_j · x
∂L/∂b_j = ∂L/∂z_j
```

反向传播到这里结束——所有参数的梯度都算出来了。注意红色箭头越来越细：梯度从输出端传到输入端，每过一层都会"衰减"一些信息。

---

这个过程就叫**反向传播（Backpropagation）**——误差从输出层开始，沿网络反方向流动，链式法则在每一步乘上当前层的局部导数。

每一层的 `backward()` 只做三件事：

1. **收**：接收上游传来的梯度 `∂L/∂output`
2. **算**：用链式法则算出自己参数（W, b）的梯度，累加
3. **传**：算出传给下游的梯度 `∂L/∂input`，return 出去

```typescript
class Layer {
  backward(gradOut: Float64Array): Float64Array {
    // 1. 穿过激活函数: gradPreAct = gradOut ⊙ activation'(z)
    // 2. 累加自己的梯度: gradW += gradPreAct ⊗ input, gradB += gradPreAct
    // 3. 传给下游: gradInput = W^T · gradPreAct
    return gradInput;
  }
}
```

训练 = forward 一趟算出 loss → backward 一趟算出所有梯度 → 梯度下降更新参数。

---

## 训练 & 试试看

损失函数和优化方式沿用 v2 的：MSE + Mini-batch SGD。模型结构变了，训练框架没变。

Dashboard：

- **📊 数据空间**：散点 + sin 真实曲线 + 模型预测的折线——看折线怎么慢慢贴合 sin
- **🧩 ReLU 神经元**：每个 ReLU 神经元各自的输出 vs x——看每个神经元"负责"哪一段
- **📉 Loss 曲线**：MSE loss 下降趋势

```bash
# 终端裸跑
pnpm exec tsx apps/v3/train.ts

# 浏览器实时可视化
pnpm dev:v3     # → http://localhost:3003
```

> 📸 **待截图**：打开 `http://localhost:3003`，训练完成（3000 epoch，16 个 ReLU 神经元），截 dashboard 全貌。

---

## 总结

直线 → 曲线的关键一跃：

| 概念 | 是什么 | v3 怎么用的 |
|---|---|---|
| 激活函数 | 打破线性的函数 | ReLU(z) = max(0, z) |
| 神经元 | 加权求和 + 激活函数 | ReLU(w·x + b)，一段折线 |
| 隐藏层 | 多个神经元的集合，外界看不见 | N 个 ReLU 神经元，输出 N 个值 |
| 输出层 | 最后一层，不加激活 | 加权求和，输出预测值 |
| Layer | 前向 + 反向的封装 | hidden = Layer(1,N,'relu'), output = Layer(N,1,'linear') |

> 继续阅读：[v4：优化曲线](/v4)

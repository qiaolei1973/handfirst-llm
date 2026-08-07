# HandFirst LLM

从零开始，逐步深入学习机器学习与 LLM。


**HandFirst = 先动手，再看学习。** 每个版本从一个具体问题出发。

**从物理世界的感受到数学世界的推导** 概念不是词典条目——它先在 demo 里出现，你亲眼看见了，然后才被命名。命名之后跟上代码实现，最后用数学推导解释。每一步都踩在前一步的感受上。

**TypeScript 贯穿全程。** 算法、server、gui 一种语言。

**可交互的可视化。** 

## 快速开始

```bash
pnpm dev          # 教程 → localhost:3000
pnpm dev:v1       # v1 · 猜一条直线 → localhost:3001
pnpm dev:v2       # v2 · 进入机器学习的世界 → localhost:3002
pnpm dev:v3       # v3 · 画曲线 → localhost:3003

# 终端裸跑
pnpm exec tsx apps/v1/train_simple.ts
pnpm exec tsx apps/v2/train.ts
pnpm exec tsx apps/v3/train.ts
```

## 结构

```
apps/v1/          猜一条直线——零术语，纯直觉
apps/v2/          进入机器学习的世界——术语命名 + MSE/SGD/中心化
apps/v3/          画曲线——单隐藏层 ReLU 神经网络
apps/_docs/       教程站点（Markdown，含 matplotlib 生成的概念图）
packages/         utils（Trainer, Layer, WS）| datasets | charts | viz
```

MIT

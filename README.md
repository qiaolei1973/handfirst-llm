# HandFirst LLM

从零开始，逐步深入学习机器学习与 LLM。

## 为什么叫 HandFirst？

"先动手，再看书。"

不是先讲理论再写代码，而是**从代码中学习**。每个版本从一个具体问题出发——"这条直线怎么画？""这组数据怎么拟合？"——用最自然的直觉写 TypeScript 代码解决问题，然后观察效果，问"能不能更好？"。

答案永远不是"因为书上这么说"，而是"因为你刚看到的效果告诉你了"。v1 的 MAE 变成 v2 的 MSE，不是因为 MSE 更高级，而是因为读者亲眼看见 MAE 的梯度在接近最优时不知道怎么减速。

## 教学路径：Demo → 概念 → 算法 → 数学

```
1. Demo      你看到什么现象？（可视化，交互式）
   ↓
2. 概念      这个现象在 ML 里叫什么？解决了什么问题？
   ↓
3. 算法      这个概念对应什么具体实现？TypeScript 代码写给你看
   ↓
4. 数学      背后的数学推导是什么？matplotlib 图辅助理解
```

概念在 demo 里**先出现**，然后才被命名。读者每学一个新概念，都有上下文可以依附。

## 为什么用 TypeScript？

 - **一种语言贯全程。** 算法、WebSocket 服务、图表渲染、浏览器交互——全是 TypeScript。不用在 Python（训练）和 JS（界面）之间切来切去。
 - **类型即文档。** `V3Params`、`Layer`、`AxesConfig`——类型让读者一眼看见数据结构长什么样。
 - **浏览器原生可交互。** Canvas2D 图表 + React dashboard，不需要装任何东西，开浏览器就能玩。

## 可交互

每个版本都有两层验证：

```bash
# 终端裸跑：看训练结果
pnpm exec tsx apps/vN/train.ts

# 浏览器实时可视化：看训练过程
pnpm dev:v3
```

浏览器里看到的不只是静态图——**播放、暂停、单步、调速、重置**，每一步参数怎么变、loss 怎么降、神经元怎么"折"，全部实时可见。训练不是黑盒，是一个你能参与进去的过程。

## 快速开始

```bash
pnpm dev          # 教程站点 → http://localhost:3000

pnpm dev:v1       # v1 · 猜一条直线 → http://localhost:3001
pnpm dev:v2       # v2 · 进入机器学习的世界 → http://localhost:3002
pnpm dev:v3       # v3 · 画曲线 → http://localhost:3003

# 终端裸跑
pnpm exec tsx apps/v1/train_simple.ts
pnpm exec tsx apps/v2/train.ts
pnpm exec tsx apps/v3/train.ts
```

## 结构

```
apps/
  v1/              猜一条直线——从直觉出发（入门）
  v2/              进入机器学习的世界——术语命名 + MSE/SGD/中心化
  v3/              画曲线——单隐藏层 ReLU 神经网络
  _docs/           教程站点
packages/
  utils/           Trainer 基类、WS server、useWsTrainer hook、Layer
  datasets/        linearData()、sinData()、sampleBatch()
  charts/          Canvas2D 图表库（Axes + CanvasManager + biz charts）
  viz/             React dashboard（v1/v2 共享）
```

## 许可证

MIT

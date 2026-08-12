# AGENTS.md

## 项目定位

HandFirst LLM — 从代码学 ML。每个版本从一个具体问题出发，用直觉写代码，观察效果，再引入概念和数学。Demo → 概念 → 算法 → 数学，概念必须在 demo 里先出现再被命名。

## 项目结构

```
apps/
  v1-v5/         每个版本自包含
    nn/             Linear, ReLU, Sequential, SGD/Adam（逐版本复制演进）
    data.ts         mock 数据 + DataLoader + 预处理
    train.ts        Trainer 类（只含训练逻辑）
    test.ts         CLI 验证入口
    server.ts       WS 启动 + dashboard 序列化
    app/            Next.js dashboard
  _docs/          教程站点
    content/vN/     Markdown 教程
    scripts/vN/     图生成脚本（matplotlib + excalidraw）
    public/         生成产物（gitignore）
packages/          纯基础设施（算法无关）
  utils/            BaseTrainer, WSServer, useWsTrainer, Mat, serial helpers
  charts/           Canvas2D 图表原语
  components/       ImageViewer
  viz/              旧 React 图表组件
docs/               项目文档
  diagram/          图设计规范
  excalidraw-style.md
```

## 代码原则

**训练代码只做训练。** `train.ts` = model + optimizer + DataLoader + step()。仪表盘展示逻辑（params 序列化、isBest、类型定义）放在 `server.ts` 或 dashboard 自己处理。

**算法概念随版本演进，各版本自包含。** `nn/` 和 `data.ts` 在每个版本目录下独立存在。不共享——读者打开一个版本就看到全部。`packages/` 只放算法无关的基础设施。

**一个函数只做一件事，签名随版本自然演化。** 不要用两个版本兼容一个函数。v5 的 `sampleBatch` 接受 `number[][]` 而不是同时保留 `number[]` 版本。

**PyTorch 是设计参考。** `model.parameters()`、`opt.reset()`、`DataLoader`、早停放调用方——这些都是 PyTorch 的标准做法。

**早停是调用方的控制逻辑。** `test.ts` 自己 `if (patience >= N) break`，WS server 用 `maxEpochs`。不在 Trainer 里维护 `_stopped/_patience/_bestVal` 状态机。

**可重置性由组件自己提供。** `Linear.resetParameters()` 存初始值复本。`Adam.reset()` 清零 m/v。不需要 `new Optimizer()` 重建。

**变量名让学习者能读懂。** 用 `gradOutMat` 不用 `GO`。数学公式写在注释里。

**不缓存可从 model 读取的信息。** `_N`、`_dim` 这些冗余字段删掉——`model.layers[0].inDim` 直接读。

**预处理和训练分离。** 数据提前标准化，DataLoader 只迭代。trainer 接收训练就绪的数组。

## 关键之前

**问**：这个字段/方法属于训练逻辑吗？不属于 → 移走或删除。
**问**：这个类型/函数只被仪表盘用吗？是 → 放 dashboard 或 server.ts，不放 train.ts。
**问**：这个信息已经存在在 model 对象里了吗？是 → 不要另存一份。
**问**：PyTorch 怎么做这件事？→ 参考它。

## 技术栈

- **pnpm** monorepo（`packages/*`, `apps/*`）
- **Next.js 15** App Router（v1-v5 :3001-3005, _docs :3000）
- **WebSocket**（ws）：训练跑 Node，浏览器 viz
- **Canvas2D**（`@handfirst/charts`）：dashboard 图表
- **matplotlib**：数据图，CJK 用 `~/.fonts/NotoSansSC-Regular.ttf`
- **Excalidraw + excalirender**：架构图、计算图、渐进讲解图
- **marked + highlight.js**：_docs MD 渲染

## 运行

```bash
pnpm exec tsx apps/vN/test.ts      # 终端裸跑
pnpm dev:vN                         # 浏览器 dashboard
pnpm dev                            # _docs
```

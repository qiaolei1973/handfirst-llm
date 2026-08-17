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

**GUI 不侵蚀训练。** 分清三种东西：产出值（`step()` 的 return loss，等于 PyTorch `criterion` 的 return，合法保留）、模型状态（参数/梯度，读即可，不另存快照）、GUI 副本（`history.push` 或存 params 快照 → 移走）。epoch 是训练概念，由 base 统一记，`_step()` 不感知。

**算法概念随版本演进，各版本自包含。** `nn/` 和 `data.ts` 在每个版本目录下独立存在。不共享——读者打开一个版本就看到全部。`packages/` 只放算法无关的基础设施。

**一个函数只做一件事，签名随版本自然演化。** 不要用两个版本兼容一个函数。`DataLoader.generate()` 在 v3/v4 返回样本数组、v5 返回 `{X, Y}` 矩阵——各版本写各版本的，不保留旧签名。

**PyTorch 是设计参考。** `model.parameters()`、`param.grad`、`DataLoader`、早停放调用方——这些都是 PyTorch 的标准做法。

**早停是调用方的控制逻辑。** `test.ts` 自己 `if (patience >= N) break`，WS server 用 `maxEpochs`。不在 Trainer 里维护 `_stopped/_patience/_bestVal` 状态机。

**reset = 重新 factory()。** 组件不存初始值复本、优化器不维护内部状态重置。需要重置时 server 用同一个 factory 重新 new 一个 trainer。

**变量名让学习者能读懂。** 用 `gradOutMat` 不用 `GO`。数学公式写在注释里。

**不缓存可从 model 读取的信息。** `_N`、`_dim` 这些冗余字段删掉——`model.layers[0].inDim` 直接读。

**预处理和训练分离。** 数据提前标准化，DataLoader 只迭代。trainer 接收训练就绪的数组。

## 关键之前

**问**：这个字段/方法属于训练逻辑吗？不属于 → 移走或删除。
**问**：这个类型/函数只被仪表盘用吗？是 → 放 dashboard 或 server.ts，不放 train.ts。
**问**：这个信息已经存在在 model 对象里了吗？是 → 不要另存一份。
**问**：这是产出值、模型状态，还是专为 GUI 存的副本？→ 前两者合法，第三种移走。
**问**：PyTorch 怎么做这件事？→ 参考它。
**问**：这个改动动了 v1-v4 的概念（新增/改名/删除）吗？→ 同步检查 `_docs/content/recap` 复习章（它把 v1-v4 串起来，最容易漏）。

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

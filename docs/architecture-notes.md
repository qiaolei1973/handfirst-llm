# Architecture Notes

> 2026-08-06 — 图表架构重构后的讨论与后续规划

## 当前状态

图表架构已完成三层分层：

```
viz/                        React 组件（薄 wrappers + dashboard）
  LineChart.tsx             <LineChart series={...} />
  LossLandscape.tsx         <LossLandscape data={...} />
  ModelFit.tsx              <ModelFit data={...} />
  surgery-dashboard.tsx     状态管理 + 派生数据 + 布局（259 行）

charts/biz/                 业务图表 = Axes + canvas + 特殊绘制逻辑
  line-chart.ts
  loss-landscape.ts
  model-fit.ts

charts/canvas.ts            DOM 管理层：ResizeObserver + DPR + onResize
charts/axes.ts              坐标管理 + 6 种 Layer + autoRange + draw 编排
charts/primitives.ts        纯绘制函数：drawAxes / drawGrid / drawLegend
```

## 讨论中的后续优化方向

### 1. 是否去掉 `packages/viz`？

**结论：暂不去掉。**

理由：
- 当前两版 app 只有 15 行，`SurgeryDashboard` 是主要的复用代码
- 砍掉 viz 意味着 v1/v2 各自复制 259 行，后续会分化

但 viz 目前的三个 React wrapper（LineChart / LossLandscape / ModelFit）应该搬入 `packages/charts`，因为它们就是 imperative API 的 React binding，属于图表库的一部分。

### 2. 是否建 `packages/components`？

**结论：是的，后续方向。**

Dashboard 可以拆为独立组件，在 app 中组装（接受一定重复）：

```
packages/components/src/
  TrainingControls.tsx      标题 + 播放/暂停/单步/重置 + 速度 slider
  StatusBar.tsx             epoch | loss | W | bias | gradW | gradBias
  LossLandscapeCard.tsx     标题 + 公式 div + LossLandscape 图
  useTrainerState.ts        WS 事件 → React state（hook 复用）
  styles/                   CSS
```

组件库 + hook 的好处：
- app 中自由搭配（v3 可以少放 LossLandscape、多加别的图）
- 数据管线由 hook 统一（WS 四个事件回调不重复）
- 布局不约束，但交互逻辑不重复

### 3. React wrappers 搬家后的结构

```
packages/charts/src/
  react/
    LineChart.tsx            从 viz 搬来
    LossLandscape.tsx        从 viz 搬来
    ModelFit.tsx             从 viz 搬来

packages/components/src/     ← 新建
  TrainingControls.tsx
  StatusBar.tsx
  LossLandscapeCard.tsx
  useTrainerState.ts
```

### 4. 待决问题

| 问题 | 状态 |
|------|------|
| LossLandscapeCard 自包含（内部算 lossCoeffs/mse）还是传 props？ | 未定 |
| CSS 拆成 CSS module 还是保持全局？ | 未定 |
| `useDashboardData` 要不要抽成独立 hook？ | 倾向于做，等 content 再多些 |

### 5. 不做的事

- 不引入 d3（工作量/收益比低；niceTicks + toX/toY 已够用）
- 不再进一步抽象 Axes（3 种图表 + 6 种 Layer 已覆盖需求）
- 不变外部 API 签名（backward compatible）

---

## 附：文档图的生成（与 charts 代码无关）

`packages/charts` 是**代码里跑的运行时图表**（Canvas2D）。文档里的**教学图**（架构图、计算图、loss 曲线）走的是另一套离线管线：

- Excalidraw 图 / matplotlib 脚本放 `apps/_docs/scripts/<vn>/`。
- `apps/_docs/scripts/generate-all.ts` 统一渲染到 `apps/_docs/public/<vn>/`（gitignore）。
- 风格见 `docs/excalidraw-style.md`；画图用 `excalidraw-diagram-generator` 技能。

别把这两套混起来——charts 是 dashboard 的交互图表，Excalidraw 是教程的静态插图。

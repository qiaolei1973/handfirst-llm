# AGENTS.md

## 项目定位

**HandFirst LLM** — 从零开始，逐步深入机器学习与 LLM。

"HandFirst" = 编码优先。不是先讲理论再写代码，而是**从代码中学习**。每个版本只有两个文件读者需要看：一个 `train.ts`（算法全部在这里），一个浏览器 dashboard 看可视化结果。理论在 `_docs/` 里配合代码讲，不是反过来。

## 核心理念

- **每一步只引入一个新概念。**
- **代码是教学材料，不是生产代码。** `train.ts` 要求一个文件读完，显式循环、不抽象、注释比代码多。
- **终端可以独立跑，浏览器可以看，互不依赖。** 基础 run：`pnpm exec tsx train.ts` 打印结果。UI run：`pnpm dev:vN` 起 WS + Next.js dev，浏览器实时可视化。UI run 时不需要先跑终端。

## 项目结构

```
apps/
  v1/              猜一条直线——从直觉出发（入门）
  v2/              进入机器学习的世界——术语命名 + MSE/SGD/中心化
  _docs/           教程站点（默认 pnpm dev）
packages/
  utils/           Trainer 基类、WS server、useWsTrainer hook、Mat、Optimizer
  datasets/        linearData()、sampleBatch()
  charts/          Canvas2D 图表
  viz/             React 手术台 dashboard
```

### 每个 version 的文件约定

```
apps/vN/
  train.ts         ← 唯一算法文件（类 + 终端入口）
  server.ts        ← WS 启动（~10 行，数据管线 + 端口）
  app/page.tsx     ← 浏览器 viz 入口（useWsTrainer + SurgeryDashboard）
```

## 工作原则

1. **改动 train.ts 之前，想清楚这一版要引入什么新概念。** 每个版本的新概念应该在 `_docs/content/vN/index.md` 里明确交代。
2. **教程和代码同步改。** 改了算法就要改对应的 md，反之亦然。
3. **数据管线统一。** `train.ts` 终端块和 `server.ts` 用同一套数据生成逻辑（`linearData()` + 预处理），不存在两份不同的创建方式。
4. **概念图用 matplotlib 生成。** 需要 CJK 字体（`~/.fonts/NotoSansSC-Regular.ttf`），输出到 `apps/_docs/public/vN/`。UI 截图人工补。

### 辅助图规范

文档中有两类图，AI 助手理解它们的方式不同：

**matplotlib 生成图**（脚本在 `scripts/` 下，输出到 `apps/_docs/public/vN/`）：

生成脚本是图的**权威定义**——数据点、线条、标注全在代码里。AI 读脚本就能 100% 还原图的内容，比任何文字描述都精确。Markdown 中的 HTML 注释只作为快速索引，写清楚「这张图由哪个脚本的哪个 figure 生成的」即可：

```markdown
![图片标题](/v1/some-chart.png)
<!-- gen_v1_images.py: #1 scatter-only.png — 纯散点 12 个蓝点，无直线 -->
```

**dashboard 截图**（人工截取）：

没有生成脚本，只能靠注释描述。截图时确保状态栏数字清晰可见，控制栏不要遮挡图表。注释写清楚 Epoch 数、参数值、loss 值等关键数字：

```markdown
![v1 训练完成 dashboard 全貌](/v1/training-done.png)
<!-- dashboard 全景截图：Epoch=600, W≈2.0, bias≈10.0, Loss 曲线全平（=0），六图展开 -->
```

**生成脚本规范**：
- 放在 `scripts/gen_vN_images.py`，一个版本一个文件
- 使用 `FontProperties(fname=FONT_PATH)` 处理 CJK 字体（`~/.fonts/NotoSansSC-Regular.ttf`）
- 用 `np.random.seed()` 固定随机种子，确保数据可复现
- 每个 `fig.savefig()` 前写清楚这是文档中的哪张图
5. **学习率、batch size 等超参数是有教学意图的。** 改动它们之前，想想这会让读者学到什么。

## 技术栈

- **pnpm** monorepo，`pnpm-workspace.yaml` 包含 `packages/*` 和 `apps/*`
- **Next.js 15** App Router（v1 :3001, v2 :3002, _docs :3000）
- **WebSocket**（ws）：训练跑在 Node 端，浏览器只做 viz
- **Canvas2D**（`@handfirst/charts`）：所有图表
- **marked + highlight.js**：_docs 的 MD 渲染
- **matplotlib**：概念图生成

## 两种运行方式

```bash
# 基础 run：终端裸跑，打印结果
pnpm exec tsx apps/vN/train.ts

# UI run：WS 训练服务 + Next.js dev，浏览器实时可视化
pnpm dev:v1     # v1 → localhost:3001
pnpm dev:v2     # v2 → localhost:3002
pnpm dev        # _docs → localhost:3000
```

两者互不依赖——UI run 自动启动 WS server，不需要先跑终端。

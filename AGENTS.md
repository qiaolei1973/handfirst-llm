# AGENTS.md

## 项目定位

**HandFirst LLM** — 从零开始，逐步深入机器学习与 LLM。

"HandFirst" = 编码优先。不是先讲理论再写代码，而是**从代码中学习**。每个版本只有两个文件读者需要看：一个 `train.ts`（算法全部在这里），一个浏览器 dashboard 看可视化结果。理论在 `_docs/` 里配合代码讲，不是反过来。

## 核心理念

### HandFirst：编码优先

不是先讲理论再写代码，而是**从代码中学习**。每个版本从一个具体问题出发——"这条直线怎么画？""这组数据怎么拟合？"——用最自然的直觉写代码解决问题，然后观察效果，问"能不能更好？"。

答案永远不是"因为书上这么说"，而是"因为你刚看到的效果告诉你了"。v1 的 MAE 变成 v2 的 MSE，不是因为 MSE 高级，而是因为读者亲眼看见 MAE 的梯度在接近最优时不减速。

### Demo → 概念 → 算法 → 数学

每个版本按这个顺序层层递进：

```
1. Demo      你看到什么现象？（怎么画出来的？为什么画不好？）
   ↓
2. 概念      这个现象在 ML 里叫什么名字？这个"名字"解决的是什么问题？
   ↓
3. 算法      这个概念对应什么具体实现？代码写给你看
   ↓
4. 数学      背后的数学推导是什么？为什么这个公式能 work？
```

概念必须在 demo 里**先出现**，然后才被命名。"先引出概念，概念的解释，然后再讲具体实现"——这个顺序保证了读者每学一个新概念，都有上下文可以依附。

SGD、Adam、backprop 这些概念，应该是"你刚才遇到的问题 → 你需要一个什么东西来解 → 这个东西的名字叫 X → X 本质上是在做什么"的节奏，而不是词典条目。

### 概念和实现严格对齐

不要空讲概念。**每个概念在对应版本的 `train.ts` 里一定有对应的代码片段**。读者读文档看到概念时，同时看到代码长什么样。反过来，train.ts 里每段核心逻辑，文档里都有对应的概念解释。

### 数学解释要扎实

"这个概念为什么有效？"不是一句话带过。数学推导要完整——从损失函数求导到梯度公式，每一步都写出来。同时佐以"深入浅出"的例子：对 MAE 求导后导数是 ±1，这件事用"差 100 推力是 1，差 0.01 推力也是 1"这句直觉就能记住。

**数学推导必须辅以清晰的图（matplotlib 生成）来表达。** 抽象的公式配具体的几何图——损失函数画成曲线，梯度画成切线或箭头，参数空间画成等高线——让读者一眼看见"数学在做什么"。图的生成脚本放在 `scripts/gen_vN_images.py`，是图的权威定义。

- **每一步只引入一个新概念。** 每个版本只有两个文件读者需要看：一个 `train.ts`（算法全部在这里），一个浏览器 dashboard 看可视化结果。理论在 `_docs/` 里配合代码讲，不是反过来。
- **代码是教学材料，不是生产代码。** `train.ts` 要求一个文件读完，显式循环、不抽象、注释比代码多。
- **终端可以独立跑，浏览器可以看，互不依赖。** 基础 run：`pnpm exec tsx train.ts` 打印结果。UI run：`pnpm dev:vN` 起 WS + Next.js dev，浏览器实时可视化。UI run 时不需要先跑终端。

## 项目结构

```
apps/
  v1/              猜一条直线——从直觉出发（入门）
  v2/              进入机器学习的世界——术语命名 + MSE/SGD/中心化
  v3/              画曲线——单隐藏层 ReLU 神经网络
  _docs/           教程站点（默认 pnpm dev）
packages/
  utils/           Trainer 基类、WS server、useWsTrainer hook、Layer、Mat
  datasets/        linearData()、sinData()、sampleBatch()
  charts/          Canvas2D 图表（Axes + CanvasManager + biz charts）
  viz/             React 手术台 dashboard（v1/v2 共享）
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
6. **教学顺序 = Demo → 概念 → 算法 → 数学。** 每个概念先从可视化或现象中引出，再给名字，再给代码实现，最后给数学推导。不先抛理论再找例子——反方向。

## 技术栈

- **pnpm** monorepo，`pnpm-workspace.yaml` 包含 `packages/*` 和 `apps/*`
- **Next.js 15** App Router（v1 :3001, v2 :3002, v3 :3003, _docs :3000）
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
pnpm dev:v3     # v3 → localhost:3003
pnpm dev        # _docs → localhost:3000
```

两者互不依赖——UI run 自动启动 WS server，不需要先跑终端。

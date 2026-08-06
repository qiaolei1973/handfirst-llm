# 🤖 HandFirst ML

> 从零开始，手写每一行 ML 代码。先动手，再动名词。

这是一个**动手学机器学习**的教程。每个版本只解决一个实际问题，你只需要读一个 `train.ts`（或 `train_simple.ts`），剩下的全是可视化。

## 怎么开始？

```bash
git clone <repo>
pnpm install

# 打开教程文档
pnpm dev

# 启动某个版本的训练台
pnpm dev:v1     # 或者 pnpm dev:v2
```

## 版本地图

| 版本 | 主题 | 适合谁 |
|------|------|--------|
| [v1](/v1) | 猜一条直线 | 没接触过 ML，从直觉出发 |
| [v2](/v2) | 进入机器学习的世界 | 学完 v1 后，给操作起名 + 改进方法 |

## 文件清单（每个版本都一样）

- **`train_simple.ts`** / **`train.ts`** — 算法全部在这里，读完就能理解一切
- `server.ts` — WebSocket 服务（~10 行）
- `app/page.tsx` — 浏览器可视化（不包含算法）

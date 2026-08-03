# 🤖 手摸手机器学习

> 从零开始，手写每一行 ML 代码。

这是一个**动手学机器学习**的教程。每个版本解决一个具体问题，你只需要读 `train_simple.ts` 和 `train_class.ts` 两个文件，剩下的全是可视化。

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

| 版本 | 主题 | 核心问题 |
|------|------|----------|
| [v1](/v1) | 认识机器学习 | loss、梯度、学习率是什么？MAE 有什么不足？ |
| [v2](/v2) | MSE + SGD | 换损失函数 + 随机采样，W/bias 能平衡收敛吗？ |

## 文件清单（每个版本都一样）

- **`train_simple.ts`** — 最简过程式实现，60 行看到底
- **`train_class.ts`** — 同样的算法套了 class，给浏览器可视化用
- `server.ts` — WebSocket 服务
- `app/page.tsx` — 浏览器 UI（不包含任何算法）

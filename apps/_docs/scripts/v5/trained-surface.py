"""
#3 trained-surface — 训练后的模型预测曲面 vs 真实曲面
"""
import matplotlib
matplotlib.use("Agg")
import _cjk_font
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 复用 surfaceData 的 trueFn
scale = 2 * np.pi
def true_fn(x1, x2):
    return np.sin(np.sqrt(x1**2 + x2**2) * scale)

# 加载训练好的参数（从 train.ts 写入的 JSON）
# 这里先用占位生成，真实数据由 train.ts 写入后替换
grid = np.linspace(-1, 1, 40)
X1, X2 = np.meshgrid(grid, grid)

fig = plt.figure(figsize=(16, 7))

# 左：真实曲面
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
Z_true = true_fn(X1, X2)
ax1.plot_surface(X1, X2, Z_true, cmap="coolwarm", alpha=0.7, edgecolor="none")
ax1.set_title("真实曲面 f(x1, x2)", fontsize=14, fontproperties=_cjk_font.cjk_fp)
ax1.set_xlabel("x1", fontproperties=_cjk_font.cjk_fp)
ax1.set_ylabel("x2", fontproperties=_cjk_font.cjk_fp)
ax1.set_zlabel("y", fontproperties=_cjk_font.cjk_fp)
ax1.view_init(elev=25, azim=-60)

# 右：模型预测曲面（占位，由 train.ts 输出替换）
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.set_title("模型预测曲面 (N 个 ReLU)", fontsize=14, fontproperties=_cjk_font.cjk_fp)
ax2.set_xlabel("x1", fontproperties=_cjk_font.cjk_fp)
ax2.set_ylabel("x2", fontproperties=_cjk_font.cjk_fp)
ax2.set_zlabel("y_pred", fontproperties=_cjk_font.cjk_fp)
ax2.view_init(elev=25, azim=-60)

out = Path(__file__).resolve().parent.parent.parent / "public" / "v5" / "trained-surface.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  → {out.relative_to(out.parent.parent.parent)}")
plt.close()

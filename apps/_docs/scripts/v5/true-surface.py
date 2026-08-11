"""
#1 true-surface — f(x1,x2)=sin(sqrt(x1^2+x2^2)*pi*2) 的真实 3D 曲面 + 噪声采样点
"""
import matplotlib
matplotlib.use("Agg")
import _cjk_font
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

scale = 2 * np.pi

def true_fn(x1, x2):
    return np.sin(np.sqrt(x1**2 + x2**2) * scale)

# 曲面网格
grid = np.linspace(-1, 1, 80)
X1, X2 = np.meshgrid(grid, grid)
Z = true_fn(X1, X2)

# 噪声采样点
np.random.seed(42)
n = 200
x1s = 2 * np.random.rand(n) - 1
x2s = 2 * np.random.rand(n) - 1
ys = true_fn(x1s, x2s) + (np.random.rand(n) - 0.5) * 0.3

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection="3d")

# 半透明曲面
surf = ax.plot_surface(X1, X2, Z, cmap="coolwarm", alpha=0.5, edgecolor="none")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="y")

# 噪声采样点（不带 label，legend 通过 proxy 创建）
ax.scatter(x1s, x2s, ys, c="#1e293b", s=15, alpha=0.7)

ax.set_xlabel("x1 (特征 1)", fontsize=12, fontproperties=_cjk_font.cjk_fp)
ax.set_ylabel("x2 (特征 2)", fontsize=12, fontproperties=_cjk_font.cjk_fp)
ax.set_zlabel("y (目标值)", fontsize=12, fontproperties=_cjk_font.cjk_fp)
ax.set_title(r"$f(x_1, x_2) = \sin(\sqrt{x_1^2 + x_2^2} \cdot 2\pi)$  + noise", fontsize=15, fontproperties=_cjk_font.cjk_fp)

# Proxy legend — 3D scatter legend shows too many dots by default
from matplotlib.lines import Line2D
proxy = Line2D([0], [0], marker="o", color="w", markerfacecolor="#1e293b",
               markersize=8, alpha=0.7, label="训练样本 (200 points)")
ax.legend(handles=[proxy], fontsize=12, prop=_cjk_font.cjk_fp)

ax.view_init(elev=25, azim=-60)

out = Path(__file__).resolve().parent.parent.parent / "public" / "v5" / "true-surface.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  -> {out.relative_to(out.parent.parent.parent)}")
plt.close()

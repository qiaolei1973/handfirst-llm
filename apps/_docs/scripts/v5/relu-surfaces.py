"""
#2 relu-surfaces — 2×2 不同 (w1,w2,b) 组合下的 ReLU 曲面形态
"""
import matplotlib
matplotlib.use("Agg")
import _cjk_font
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

grid = np.linspace(-1, 1, 60)
X1, X2 = np.meshgrid(grid, grid)

cases = [
    (1.5, 0, 0,    "w1=1.5, w2=0\n沿 x1 折"),
    (-1.5, 0, 0,   "w1=-1.5, w2=0\n沿 x1 反折"),
    (0.8, 0.8, 0,  "w1=0.8, w2=0.8\n斜向折"),
    (1.5, -1, 0.3, "w1=1.5, w2=-1, b=0.3\n折线+平移"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 13), subplot_kw={"projection": "3d"})

for ax, (w1, w2, b, title) in zip(axes.flat, cases):
    Z = np.maximum(w1 * X1 + w2 * X2 + b, 0)
    ax.plot_surface(X1, X2, Z, cmap="coolwarm", alpha=0.8, edgecolor="none")
    ax.set_xlabel("x1", fontsize=10, fontproperties=_cjk_font.cjk_fp)
    ax.set_ylabel("x2", fontsize=10, fontproperties=_cjk_font.cjk_fp)
    ax.set_title(title, fontsize=13, fontproperties=_cjk_font.cjk_fp)
    ax.view_init(elev=25, azim=-55)

fig.suptitle("ReLU(w1*x1 + w2*x2 + b) 的四种曲面形态", fontsize=16, y=0.98, fontproperties=_cjk_font.cjk_fp)

out = Path(__file__).resolve().parent.parent.parent / "public" / "v5" / "relu-surfaces.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  → {out.relative_to(out.parent.parent.parent)}")
plt.close()

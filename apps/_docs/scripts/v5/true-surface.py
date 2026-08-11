"""
#1 true-surface — f(x1,x2)=sin(sqrt(x1^2+x2^2)*pi*2) 的真实 3D 曲面 + 噪声采样点
"""
import matplotlib
matplotlib.use("Agg")
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
ax.plot_surface(X1, X2, Z, cmap="coolwarm", alpha=0.5, edgecolor="none")
# 噪声采样点
ax.scatter(x1s, x2s, ys, c="#1e293b", s=15, alpha=0.7, label="训练样本 (200 points)")

ax.set_xlabel("x₁ (特征1)", fontsize=12)
ax.set_ylabel("x₂ (特征2)", fontsize=12)
ax.set_zlabel("y (目标值)", fontsize=12)
ax.set_title(r"$f(x_1, x_2) = \sin(\sqrt{x_1^2 + x_2^2} \cdot 2\pi)$  + noise", fontsize=15)
ax.legend(fontsize=12)
ax.view_init(elev=25, azim=-60)

out = Path(__file__).resolve().parent.parent.parent / "public" / "v5" / "true-surface.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  → {out.relative_to(out.parent.parent.parent)}")
plt.close()

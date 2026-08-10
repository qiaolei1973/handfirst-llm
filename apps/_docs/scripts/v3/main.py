"""
v3 教学图片 — 全部使用设计数据，不需要训练。
Output: apps/_docs/public/v3/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / ".." / "public" / "v3"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_sm = fm.FontProperties(fname=FONT_PATH, size=9)
fp_md = fm.FontProperties(fname=FONT_PATH, size=10)
fp_lg = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl = fm.FontProperties(fname=FONT_PATH, size=13)
fp_legend = fm.FontProperties(fname=FONT_PATH, size=9)

COLORS = {
    "red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e",
    "amber": "#f59e0b", "purple": "#8b5cf6", "gray": "#94a3b8",
    "gray_dark": "#64748b",
}

np.random.seed(42)
X_SCALE = 2 * np.pi
N_DATA = 60
x_norm = np.linspace(0, 1, N_DATA)
x_orig = x_norm * X_SCALE
true_y = np.sin(x_orig)
noise = (np.random.rand(N_DATA) - 0.5) * 0.3
labels = true_y + noise

# ================================================================
#  1. line-vs-sin.png — 直线 vs 曲线
# ================================================================
def best_fit_line(x, y):
    n = len(x)
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - a * np.sum(x)) / n
    return a, b

w_best, b_best = best_fit_line(x_orig, labels)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(x_orig, labels, c="#3b82f6", s=35, zorder=5, alpha=0.7, label="数据点 (sin + noise)")
x_smooth = np.linspace(0, X_SCALE, 300)
ax.plot(x_smooth, np.sin(x_smooth), COLORS["green"], lw=2.5, ls="--", label="y = sin(x)")
ax.plot(x_smooth, w_best * x_smooth + b_best, COLORS["red"], lw=2.5, label="最佳直线（MSE）")
ax.annotate("直线永远画不出曲线\n不管怎么调 W 和 bias",
            (x_orig[35], labels[35]), textcoords="offset points", xytext=(30, -30),
            fontsize=9, color=COLORS["red"], fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["red"], alpha=0.8))
ax.set_xlim(-0.3, X_SCALE + 0.3); ax.set_ylim(-1.6, 1.6)
ax.grid(True, alpha=0.2); ax.legend(prop=fp_legend, loc="lower right")
ax.set_xlabel("x", fontproperties=fp_md); ax.set_ylabel("y", fontproperties=fp_md)
ax.set_title("y = Wx + b 只能画直线——数据是弯的，模型也得弯", fontproperties=fp_lg, pad=10)
ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(OUT / "line-vs-sin.png", dpi=144)
plt.close(fig)
print("  v3/line-vs-sin.png")

# ================================================================
#  2. relu-shapes.png — ReLU 在不同 w,b 下的形态
# ================================================================
configs = [
    {"w": 1.5, "b": -1.5, "color": "#3b82f6", "desc": "w=1.5, b=-1.5\n折点 x=-b/w=1.0\n折后向上"},
    {"w": -2.0, "b": 2.0, "color": "#ef4444", "desc": "w=-2, b=2\n折点 x=1.0\n折后向下"},
    {"w": 3.0, "b": -1.5, "color": "#f59e0b", "desc": "w=3, b=-1.5\n折点 x=0.5\n折后很陡"},
    {"w": 0.8, "b": -1.6, "color": "#8b5cf6", "desc": "w=0.8, b=-1.6\n折点 x=2.0\n折后平缓"},
]
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
axes = axes.flatten()
x_range = np.linspace(0, X_SCALE, 300)
for i, cfg in enumerate(configs):
    ax = axes[i]; w, b = cfg["w"], cfg["b"]
    z = w * x_range + b
    ax.plot(x_range, z, cfg["color"], lw=0.8, ls=":", alpha=0.5)
    ax.plot(x_range, np.maximum(0, z), cfg["color"], lw=2.5)
    kink_x = -b / w
    if 0 <= kink_x <= X_SCALE:
        ax.axvline(x=kink_x, color=cfg["color"], ls="--", lw=1, alpha=0.5)
        ax.scatter([kink_x], [0], c=cfg["color"], s=60, zorder=10, edgecolors="white", linewidth=1.5)
        ax.annotate(f"折点\nx={kink_x:.1f}", (kink_x, 0), textcoords="offset points", xytext=(15, 15),
                    fontsize=8, color=cfg["color"], fontproperties=fp_sm,
                    arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=0.8))
    ax.axhline(y=0, color="#94a3b8", lw=0.5, alpha=0.5)
    ax.set_xlim(0, X_SCALE); ax.set_ylim(-2.5, 4.5); ax.grid(True, alpha=0.15)
    ax.set_title(cfg["desc"], fontproperties=fp_md, color=cfg["color"])
    ax.set_xlabel("x", fontproperties=fp_sm); ax.set_ylabel("ReLU(w·x + b)", fontproperties=fp_sm)
    ax.tick_params(labelsize=8)
fig.suptitle("ReLU(z) = max(0, z)：w, b 控制「在哪折」和「怎么折」", fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "relu-shapes.png", dpi=144)
plt.close(fig)
print("  v3/relu-shapes.png")

# ================================================================
#  3. neurons-approximate.png — 设计数据：等距 ReLU 神经元逼近 sin
#
#  教学要点：不需要训练。用等距分段线性插值构造 ReLU 分解。
#  N 个神经元 = N 段折线。越多越逼近。
#  数学：PWL(x) = y0 + s0*x + sum(Δs_k * ReLU(x - break_k))
# ================================================================
def pwl_neurons(n_segments):
    """等距分段线性逼近 sin。返回神经元参数（标准化域 [0,1]）"""
    breaks = np.linspace(0, 1, n_segments + 1)
    y_breaks = np.sin(breaks * X_SCALE)
    slopes = np.diff(y_breaks) / np.diff(breaks)
    hw = np.ones(n_segments - 1)              # 每个 ReLU 的 w = 1
    hb = -breaks[1:-1]                         # 折点在 break 位置
    ow = np.diff(slopes)                       # 斜率变化量
    ob = y_breaks[0]                           # 截距 = 第一个点
    return hw, hb, ow, ob

neuron_counts = [2, 4, 8, 16]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()
x_plot = np.linspace(0, 1, 300)

for i, n in enumerate(neuron_counts):
    ax = axes[i]
    hw, hb, ow, ob = pwl_neurons(n)
    h = np.maximum(0, np.outer(x_plot, hw) + hb)
    y_pred = float(ob) + h @ ow
    mse = float(np.mean((y_pred - np.sin(x_plot * X_SCALE))**2))

    ax.scatter(x_orig, labels, c="#94a3b8", s=18, zorder=3, alpha=0.5)
    ax.plot(x_orig, true_y, COLORS["green"], lw=2, ls="--", alpha=0.7, label="y = sin(x)")
    ax.plot(x_plot * X_SCALE, y_pred, COLORS["purple"], lw=2.5,
            label=f"模型 ({len(hw)+1} 段折线)")

    for j in range(len(hw)):
        ax.plot(x_plot * X_SCALE, ow[j] * np.maximum(0, hw[j] * x_plot + hb[j]),
                COLORS["gray"], lw=0.5, alpha=0.3)

    ax.set_xlim(-0.3, X_SCALE + 0.3); ax.set_ylim(-1.6, 1.6); ax.grid(True, alpha=0.15)
    ax.legend(prop=fp_legend, loc="lower right", fontsize=8)
    ax.set_title(f"{n} 段折线（{n} 个 ReLU）  MSE={mse:.4f}", fontproperties=fp_lg)
    ax.set_xlabel("x", fontproperties=fp_sm); ax.set_ylabel("y", fontproperties=fp_sm)
    ax.tick_params(labelsize=8)

fig.suptitle("神经元越多，折线越多 → 越逼近曲线（等距分段线性逼近，非训练结果）",
             fontproperties=fp_xl, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "neurons-approximate.png", dpi=144)
plt.close(fig)
print("  v3/neurons-approximate.png")

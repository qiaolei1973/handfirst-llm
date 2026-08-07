"""
Generate v3 tutorial images with CJK support.
Output: apps/_docs/public/v3/
Run: python3 scripts/gen_v3_images.py

Images generated:
  1. line-vs-sin.png         — scatter + sin curve + best-fit line (line can't bend)
  2. relu-shapes.png         — 4 ReLU neurons with different w,b showing kink control
  3. neurons-approximate.png — 2×2: 2, 4, 8, 16 neuron networks fitting sin
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "apps/_docs/public/v3"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_sm  = fm.FontProperties(fname=FONT_PATH, size=9)
fp_md  = fm.FontProperties(fname=FONT_PATH, size=10)
fp_lg  = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl  = fm.FontProperties(fname=FONT_PATH, size=13)
fp_legend = fm.FontProperties(fname=FONT_PATH, size=9)
fp_title = fm.FontProperties(fname=FONT_PATH, size=12)

COLORS = {
    "red":    "#ef4444",
    "blue":   "#3b82f6",
    "green":  "#22c55e",
    "amber":  "#f59e0b",
    "purple": "#8b5cf6",
    "gray":   "#94a3b8",
    "gray_dark": "#64748b",
    "true_fn":  "#22c55e",
    "pink":     "#ec4899",
    "cyan":     "#06b6d4",
    "orange":   "#f97316",
}

np.random.seed(42)

# ================================================================
#  Shared: sin dataset (matching sinData(60) in packages/datasets)
# ================================================================
X_SCALE = 2 * np.pi
N_DATA = 60
x_norm = np.linspace(0, 1, N_DATA)
x_orig = x_norm * X_SCALE
true_y = np.sin(x_orig)
noise = (np.random.rand(N_DATA) - 0.5) * 0.3
labels = true_y + noise


# ================================================================
#  1. line-vs-sin.png — 直线 vs 曲线：y=Wx+b 的极限
# ================================================================
def best_fit_line(x, y):
    """Ordinary least squares: y = ax + b"""
    n = len(x)
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - a * np.sum(x)) / n
    return a, b

w_best, b_best = best_fit_line(x_orig, labels)

fig, ax = plt.subplots(figsize=(8, 4.8))

# Scatter
ax.scatter(x_orig, labels, c="#3b82f6", s=35, zorder=5, alpha=0.7, label="数据点 (sin + noise)")

# True sin curve
x_smooth = np.linspace(0, X_SCALE, 300)
ax.plot(x_smooth, np.sin(x_smooth), COLORS["green"], lw=2.5, ls="--",
        label="y = sin(x)")

# Best-fit line
y_line = w_best * x_smooth + b_best
ax.plot(x_smooth, y_line, COLORS["red"], lw=2.5, label="最佳直线（MSE）")

# Annotate the failure
mid = N_DATA // 2
ax.annotate("直线永远画不出曲线\n不管怎么调 W 和 bias",
            (x_orig[35], labels[35]), textcoords="offset points",
            xytext=(30, -30), fontsize=9, color=COLORS["red"],
            fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=COLORS["red"], alpha=0.8))

ax.set_xlim(-0.3, X_SCALE + 0.3)
ax.set_ylim(-1.6, 1.6)
ax.grid(True, alpha=0.2)
ax.legend(prop=fp_legend, loc="lower right")
ax.set_xlabel("x", fontproperties=fp_md)
ax.set_ylabel("y", fontproperties=fp_md)
ax.set_title("y = Wx + b 只能画直线——数据是弯的，模型也得弯",
             fontproperties=fp_lg, pad=10)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig(OUT / "line-vs-sin.png", dpi=144)
plt.close(fig)
print("  ✓ v3/line-vs-sin.png")


# ================================================================
#  2. relu-shapes.png — ReLU 在不同 w,b 下的形态
# ================================================================
def relu(z):
    return np.maximum(0, z)

# 4 configurations showing how w,b control the kink
configs = [
    {"w": 1.5, "b": -1.5,  "color": "#3b82f6",
     "desc": "w=1.5, b=−1.5\n折点 x=−b/w=1.0\n折后向上"},
    {"w": -2.0, "b": 2.0,  "color": "#ef4444",
     "desc": "w=−2, b=2\n折点 x=1.0\n折后向下"},
    {"w": 3.0, "b": -1.5,  "color": "#f59e0b",
     "desc": "w=3, b=−1.5\n折点 x=0.5\n折后很陡"},
    {"w": 0.8, "b": -1.6,  "color": "#8b5cf6",
     "desc": "w=0.8, b=−1.6\n折点 x=2.0\n折后平缓"},
]

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
axes = axes.flatten()

x_range = np.linspace(0, X_SCALE, 300)

for i, cfg in enumerate(configs):
    ax = axes[i]
    w, b = cfg["w"], cfg["b"]

    # ReLU output
    z = w * x_range + b
    y_relu = relu(z)

    # Pre-activation line (dotted)
    ax.plot(x_range, z, cfg["color"], lw=0.8, ls=":", alpha=0.5)

    # ReLU output (solid)
    ax.plot(x_range, y_relu, cfg["color"], lw=2.5)

    # Mark kink point
    kink_x = -b / w
    if 0 <= kink_x <= X_SCALE:
        ax.axvline(x=kink_x, color=cfg["color"], ls="--", lw=1, alpha=0.5)
        ax.scatter([kink_x], [0], c=cfg["color"], s=60, zorder=10,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(f"折点\nx={kink_x:.1f}", (kink_x, 0),
                    textcoords="offset points", xytext=(15, 15),
                    fontsize=8, color=cfg["color"], fontproperties=fp_sm,
                    arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=0.8))

    # y=0 reference
    ax.axhline(y=0, color="#94a3b8", lw=0.5, alpha=0.5)

    ax.set_xlim(0, X_SCALE)
    ax.set_ylim(-2.5, 4.5)
    ax.grid(True, alpha=0.15)
    ax.set_title(cfg["desc"], fontproperties=fp_md, color=cfg["color"])
    ax.set_xlabel("x", fontproperties=fp_sm)
    ax.set_ylabel("ReLU(w·x + b)", fontproperties=fp_sm)
    ax.tick_params(labelsize=8)

fig.suptitle("ReLU(z) = max(0, z)：w, b 控制「在哪折」和「怎么折」",
             fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "relu-shapes.png", dpi=144)
plt.close(fig)
print("  ✓ v3/relu-shapes.png")


# ================================================================
#  3. neurons-approximate.png — 训练 2/4/8/16 个神经元逼近 sin
# ================================================================
def train_nn(x, y, n_neurons, epochs=4000, lr=0.02, batch_size=40, seed=42):
    """Train a single-hidden-layer ReLU network (pure numpy, scalar-style).
    Returns: (hidden_w, hidden_b, output_w, output_b, final_loss)"""
    rng = np.random.RandomState(seed)
    n = len(x)

    # Init weights (matching Layer constructor)
    hidden_w = rng.rand(n_neurons) * 1.2 - 0.6
    hidden_b = np.array([-hidden_w[j] * rng.rand() for j in range(n_neurons)])
    output_w = rng.rand(n_neurons) * 1.2 - 0.6
    output_b = 0.0

    indices = np.arange(n)

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_idx = indices[:batch_size]
        xb, yb = x[batch_idx], y[batch_idx]

        # ---- Forward ----
        # hidden: [batch, neurons]
        z = np.outer(xb, hidden_w) + hidden_b  # pre-activation
        h = np.maximum(0, z)                     # ReLU
        y_pred = h @ output_w + output_b

        diff = y_pred - yb
        loss = np.mean(diff ** 2)

        # ---- Backward (MSE) ----
        grad_out = (2 * diff) / batch_size  # [batch]

        grad_output_w = h.T @ grad_out       # [neurons]
        grad_output_b = np.sum(grad_out)

        grad_h = np.outer(grad_out, output_w)        # [batch, neurons]
        grad_preact = grad_h * (z > 0)                # ReLU derivative

        grad_hidden_w = xb @ grad_preact              # [neurons]
        grad_hidden_b = np.sum(grad_preact, axis=0)   # [neurons]

        # ---- Update ----
        output_w -= lr * grad_output_w
        output_b -= lr * grad_output_b
        hidden_w -= lr * grad_hidden_w
        hidden_b -= lr * grad_hidden_b

    # Final loss on all data
    z = np.outer(x, hidden_w) + hidden_b
    h = np.maximum(0, z)
    y_pred = h @ output_w + output_b
    final_loss = np.mean((y_pred - y) ** 2)

    return hidden_w, hidden_b, output_w, output_b, final_loss


def predict_nn(x, hidden_w, hidden_b, output_w, output_b):
    h = np.maximum(0, np.outer(x, hidden_w) + hidden_b)
    return h @ output_w + output_b


neuron_counts = [2, 4, 8, 16]
all_results = []

for n_neurons in neuron_counts:
    hw, hb, ow, ob, loss = train_nn(x_norm, labels, n_neurons,
                                     epochs=4000, lr=0.02, seed=42)
    all_results.append((hw, hb, ow, ob, loss))
    print(f"  trained {n_neurons}-neuron network: loss={loss:.6f}")

# ---- Plot: 2×2 grid ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

x_plot = np.linspace(0, 1, 300)

for i, (n_neurons, (hw, hb, ow, ob, loss)) in enumerate(
    zip(neuron_counts, all_results)
):
    ax = axes[i]
    y_pred = predict_nn(x_plot, hw, hb, ow, ob)

    # Scatter
    ax.scatter(x_orig, labels, c="#94a3b8", s=18, zorder=3, alpha=0.5)

    # True sin
    ax.plot(x_orig, true_y, COLORS["green"], lw=2, ls="--", alpha=0.7,
            label="y = sin(x)")

    # Model prediction
    ax.plot(x_plot * X_SCALE, y_pred, COLORS["purple"], lw=2.5,
            label=f"模型 ({n_neurons} 个 ReLU)")

    # Individual neuron contributions (dotted, faint)
    for j in range(n_neurons):
        neuron_out = ow[j] * np.maximum(0, hw[j] * x_plot + hb[j])
        ax.plot(x_plot * X_SCALE, neuron_out,
                COLORS["gray"], lw=0.5, alpha=0.3)

    ax.set_xlim(-0.3, X_SCALE + 0.3)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, alpha=0.15)
    ax.legend(prop=fp_legend, loc="lower right", fontsize=8)
    ax.set_title(f"{n_neurons} 个 ReLU 神经元  (Loss={loss:.4f})",
                 fontproperties=fp_lg)
    ax.set_xlabel("x", fontproperties=fp_sm)
    ax.set_ylabel("y", fontproperties=fp_sm)
    ax.tick_params(labelsize=8)

fig.suptitle("神经元越多，折线越多 → 越逼近曲线",
             fontproperties=fp_xl, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "neurons-approximate.png", dpi=144)
plt.close(fig)
print("  ✓ v3/neurons-approximate.png")

# ================================================================
print(f"\n✓ All images generated:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

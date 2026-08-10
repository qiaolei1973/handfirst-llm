"""
Generate v2 tutorial images with CJK support.
Output: apps/_docs/public/v2/ + apps/_docs/public/v1/loss-comparison.png
Run: python3 scripts/gen_v2_images.py

Images generated:
  1. v2/gradient-intuition.png — 下山比喻：抛物线上不同位置的坡度箭头
  2. v2/loss-comparison.png    — MAE vs MSE: loss curves + gradients side-by-side
  3. v2/sgd-fluctuation.png    — BGD (smooth) vs SGD (jittery) loss over epochs
  4. v2/centering.png          — feature distribution before/after mean-centering
  5. v1/loss-comparison.png    — L(bias) landscape: MAE vs MSE trajectory to valley
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT_V2 = Path(__file__).resolve().parent.parent.parent / "apps/_docs/public/v2"
OUT_V1 = Path(__file__).resolve().parent.parent.parent / "apps/_docs/public/v1"
OUT_V2.mkdir(parents=True, exist_ok=True)
OUT_V1.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_sm  = fm.FontProperties(fname=FONT_PATH, size=9)
fp_md  = fm.FontProperties(fname=FONT_PATH, size=10)
fp_lg  = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl  = fm.FontProperties(fname=FONT_PATH, size=13)
fp_bold = fm.FontProperties(fname=FONT_PATH, size=10, weight="bold")
fp_legend = fm.FontProperties(fname=FONT_PATH, size=9)
fp_tick = fm.FontProperties(fname=FONT_PATH, size=9)

np.random.seed(42)

COLORS = {
    "red":    "#ef4444",
    "blue":   "#3b82f6",
    "green":  "#22c55e",
    "amber":  "#f59e0b",
    "purple": "#8b5cf6",
    "gray":   "#94a3b8",
    "gray_dark": "#64748b",
}

# ================================================================
#  1. v2/gradient-intuition.png — 下坡比喻：梯度就是脚下的坡度
# ================================================================
w_range = np.linspace(-1.5, 3.5, 500)
# Parabola: L(W) = (W-1)^2 + 0.3  (valley at W=1)
L_vals = (w_range - 1)**2 + 0.3

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Plot the loss landscape
ax.plot(w_range, L_vals, "#64748b", lw=2, label="Loss L(W)")
ax.set_xlim(-1.5, 3.5); ax.set_ylim(-0.5, 7.5)

# Pick 5 points along the curve to show gradient arrows
points = [
    {"W": -0.5, "label": "坡度陡\n→ 大步", "color": "#ef4444", "xytext": (-10, -30)},
    {"W": 0.3,  "label": "坡度缓\n→ 小步", "color": "#f59e0b", "xytext": (-10, -30)},
    {"W": 1.0,  "label": "谷底\n梯度=0",  "color": "#22c55e", "xytext": (10, -30)},
    {"W": 1.8,  "label": "坡度缓\n→ 小步", "color": "#f59e0b", "xytext": (20, -30)},
    {"W": 2.5,  "label": "坡度陡\n→ 大步", "color": "#ef4444", "xytext": (20, -30)},
]

for pt in points:
    W = pt["W"]
    L = (W - 1)**2 + 0.3
    grad = 2 * (W - 1)  # derivative of (W-1)²+0.3

    # Point marker
    ax.scatter([W], [L], c=pt["color"], s=60, zorder=10, edgecolors="white", linewidth=1)

    # Tangent arrow: points downhill along the actual tangent line
    # For L=(W-1)²: grad = 2(W-1). dy/dx = grad.
    if abs(grad) > 0.01:  # skip valley (grad=0, no slope)
        arrow_dx = -np.sign(grad) * 0.5
        arrow_dy = grad * arrow_dx
        ax.arrow(W, L, arrow_dx, arrow_dy, head_width=0.25, head_length=0.15,
                 fc=pt["color"], ec=pt["color"], alpha=0.85, lw=1.5, zorder=5)

    # Label
    ax.annotate(pt["label"], (W, L), textcoords="offset points",
                xytext=pt["xytext"], fontsize=8, color=pt["color"],
                ha="center", fontproperties=fp_sm,
                arrowprops=dict(arrowstyle="->", color=pt["color"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=pt["color"], alpha=0.8))

# Trajectory dots (simulated gradient descent steps)
w_cur = 2.8
traj_w = [w_cur]
lr = 0.15
for _ in range(12):
    grad = 2 * (w_cur - 1)
    w_cur -= lr * grad
    traj_w.append(w_cur)
traj_w = np.array(traj_w)
ax.scatter(traj_w, (traj_w - 1)**2 + 0.3, c="#6366f1", s=25, zorder=8)
ax.plot(traj_w, (traj_w - 1)**2 + 0.3, "#6366f1", lw=1.2, alpha=0.6,
        label="梯度下降路径")

ax.legend(prop=fp_legend, loc="upper left", fontsize=8)
ax.set_xlabel("参数 W", fontproperties=fp_md)
ax.set_ylabel("Loss L(W)", fontproperties=fp_md)
ax.set_title("蒙眼下山：梯度 = 脚下的坡度，告诉你往哪走、走多远",
            fontproperties=fp_lg)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig(OUT_V2 / "gradient-intuition.png", dpi=144)
plt.close(fig)
print("  ✓ v2/gradient-intuition.png")

# ================================================================
#  2. v2/loss-comparison.png — MAE vs MSE: loss curves + gradients
# ================================================================
diff_range = np.linspace(-8, 8, 600)

# ---- Loss functions ----
mae_loss  = np.abs(diff_range)
mse_loss  = diff_range ** 2

# ---- Gradients ----
mae_grad = np.sign(diff_range)
mse_grad = 2 * diff_range

fig = plt.figure(figsize=(11, 5))

# -- Left: Loss curves --
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(diff_range, mae_loss, COLORS["red"], lw=2.5, label="MAE: L = |diff|")
ax1.plot(diff_range, mse_loss, "#8b5cf6", lw=2.5, label="MSE: L = diff²")
# highlight key differences
ax1.axhline(y=25, color="#94a3b8", ls="--", lw=0.8, alpha=0.5)
ax1.annotate("diff=5 → MAE=5, MSE=25\nMSE 罚得更重",
             (5, 25), textcoords="offset points", xytext=(30, 15),
             fontsize=8, color="#64748b", fontproperties=fp_sm,
             arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))
ax1.annotate("diff=1 → MAE=MSE=1\n小误差时差不多",
             (1.5, 1.5), textcoords="offset points", xytext=(10, -25),
             fontsize=8, color="#64748b", fontproperties=fp_sm,
             arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))
ax1.set_xlim(-8, 8); ax1.set_ylim(-2, 40)
ax1.grid(True, alpha=0.2)
ax1.legend(prop=fp_legend, loc="upper center")
ax1.set_title("损失函数：怎么给「差距」评分", fontproperties=fp_lg)
ax1.set_xlabel("diff = yPred − y", fontproperties=fp_md)
ax1.set_ylabel("Loss", fontproperties=fp_md)
ax1.tick_params(labelsize=9)

# -- Right: Gradients --
ax2 = fig.add_subplot(2, 2, 2)

# MAE gradient — step function with separate colors for positive/negative
for i in range(len(diff_range) - 1):
    color = "#3b82f6" if diff_range[i] < 0 else ("#ef4444" if diff_range[i] > 0 else "#94a3b8")
    ax2.plot(diff_range[i:i+2], mae_grad[i:i+2], color=color, lw=2.5, alpha=0.7)
ax2.plot([], [], COLORS["red"], lw=2.5, label="MAE: grad = sign(diff)")

# MSE gradient — linear
ax2.plot(diff_range, mse_grad, "#8b5cf6", lw=2.5, label="MSE: grad = 2·diff")

ax2.axhline(y=0, color="#94a3b8", lw=0.5)
ax2.axvline(x=0, color="#94a3b8", lw=0.5)
ax2.set_xlim(-8, 8); ax2.set_ylim(-5, 5)
ax2.grid(True, alpha=0.2)
ax2.legend(prop=fp_legend, loc="upper left")
ax2.set_title("梯度：怎么决定「怎么改」", fontproperties=fp_lg)
ax2.set_xlabel("diff = yPred − y", fontproperties=fp_md)
ax2.set_ylabel("Gradient", fontproperties=fp_md)
ax2.tick_params(labelsize=9)

# -- Bottom row: zoomed comparison at a specific diff --
ax3 = fig.add_subplot(2, 2, 3)

# Show MSE parabola is quadratic: near minimum, gradient → 0
zoom_x = np.linspace(-3, 3, 300)
ax3.plot(zoom_x, np.abs(zoom_x), COLORS["red"], lw=2, label="MAE")
ax3.plot(zoom_x, zoom_x**2, "#8b5cf6", lw=2, label="MSE")
ax3.axvline(x=0, color="#94a3b8", lw=0.5)

# Show arrows: MAE has constant slope even near 0
pts = np.array([[-1, 1, -0.3, 0.09], [-0.5, 0.5, -0.15, 0.0225]])
for diff_val, mae_l, mae_dl, mse_l in pts:
    ax3.annotate("", xy=(diff_val+0.4, mae_l + 0.4*1), xytext=(diff_val-0.2, mae_l - 0.2*1),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))
    ax3.annotate("", xy=(diff_val+0.4, mse_l + 0.4*2*abs(diff_val)),
                xytext=(diff_val-0.2, mse_l - 0.2*2*abs(diff_val)),
                arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=1.5))

ax3.set_xlim(-3, 3); ax3.set_ylim(-1, 10)
ax3.grid(True, alpha=0.2)
ax3.legend(prop=fp_legend, loc="upper center")
ax3.set_title("放大看：接近最优时发生了什么", fontproperties=fp_lg)
ax3.set_xlabel("diff", fontproperties=fp_md)
ax3.set_ylabel("Loss", fontproperties=fp_md)
ax3.tick_params(labelsize=9)

# Bottom-right: annotated formula comparison
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis("off")
formula_text = (
    "MAE：梯度永远是 ±1\n"
    "  差 100  → 推力 1\n"
    "  差 0.01 → 推力 1\n"
    "  x 该刹车时刹不住\n"
    "\n"
    "MSE：梯度和误差成比例\n"
    "  差 10   → 推力 20\n"
    "  差 0.01 → 推力 0.02\n"
    "  ✓ 自动加/减速，自然收敛"
)
ax4.text(0.05, 0.95, formula_text, transform=ax4.transAxes,
         fontsize=10, fontproperties=fp_md, va="top",
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc",
                   edgecolor="#e2e8f0"))
ax4.set_title("本质区别", fontproperties=fp_lg)

fig.suptitle("MAE vs MSE：同样的 diff，梯度一个看方向一个看距离",
             fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_V2 / "loss-comparison.png", dpi=144)
plt.close(fig)
print("  ✓ v2/loss-comparison.png")

# ================================================================
#  2. v2/sgd-fluctuation.png — BGD (smooth) vs SGD (jittery)
# ================================================================
np.random.seed(42)
EPOCHS = 600
epochs = np.arange(1, EPOCHS + 1)

# Simulate BGD: smooth exponential decay
bgd_loss = 30 * np.exp(-epochs / 150) + 0.8 * np.exp(-epochs / 30) + 0.3

# Simulate SGD: same trend but with noise (std decreases as training progresses)
sgd_noise = np.random.randn(EPOCHS) * (2.5 * np.exp(-epochs / 200) + 0.3)
sgd_loss = bgd_loss + sgd_noise
sgd_loss = np.maximum(sgd_loss, 0.02)  # loss can't be negative

# Smooth trend line for SGD (moving average)
window = 30
sgd_trend = np.convolve(sgd_loss, np.ones(window)/window, mode='same')

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(epochs, bgd_loss, COLORS["blue"], lw=2, label="BGD（全量梯度下降）",
        alpha=0.85)
ax.plot(epochs, sgd_loss, COLORS["red"], lw=0.6, alpha=0.5,
        label="SGD 每步 loss（Mini-batch，batch=8）")
ax.plot(epochs, sgd_trend, COLORS["red"], lw=2.0, alpha=0.9,
        label="SGD 趋势（30-epoch 移动平均）")

# Annotate the key observation
ax.annotate("SGD 每步都在抖\n但整体方向是对的",
            (150, sgd_loss[149]), textcoords="offset points",
            xytext=(40, 35), fontsize=9, color="#64748b",
            fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1))
ax.annotate("BGD 平滑但缺乏随机性\n每次走的路完全一样",
            (100, bgd_loss[99]), textcoords="offset points",
            xytext=(-60, -35), fontsize=9, color="#64748b",
            fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1))

ax.set_xlim(0, EPOCHS); ax.set_ylim(-0.5, 15)
ax.grid(True, alpha=0.2)
ax.legend(prop=fp_legend, loc="upper right")
ax.set_title("SGD（Mini-batch 随机采样）vs BGD（全量）— Loss 下降对比",
            fontproperties=fp_xl, pad=12)
ax.set_xlabel("Epoch", fontproperties=fp_md)
ax.set_ylabel("MSE Loss", fontproperties=fp_md)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig(OUT_V2 / "sgd-fluctuation.png", dpi=144)
plt.close(fig)
print("  ✓ v2/sgd-fluctuation.png")

# ================================================================
#  3. v2/centering.png — Before/after distribution + training effect
# ================================================================
np.random.seed(42)
N, MAX_X = 12, 20
features = np.linspace(0, MAX_X, N)
noise = (np.random.rand(N) - 0.5) * 6
labels = 2 * features + 10 + noise

mean_x = np.mean(features)
features_centered = features - mean_x

fig = plt.figure(figsize=(11, 5.8))

# ---- Top-left: Before centering (data distribution) ----
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(features, np.ones_like(features) * 0.5,
            c=COLORS["red"], s=60, zorder=5, edgecolors="white", linewidth=0.5)
ax1.axvline(x=mean_x, color="#64748b", ls="--", lw=1.2, alpha=0.7,
            label=f"均值 = {mean_x:.1f}")

# Show gradient push direction for each point (all right)
for xp in features:
    ax1.arrow(xp, 0.5, 2.5, 0, head_width=0.06, head_length=0.6,
              fc=COLORS["red"], ec=COLORS["red"], alpha=0.35, lw=0.8)

ax1.set_ylim(0, 1.2); ax1.set_xlim(-3, 23)
ax1.set_yticks([])
ax1.set_title("中心化前：所有推力朝同一个方向", fontproperties=fp_lg)
ax1.set_xlabel("x（全正）", fontproperties=fp_md)
ax1.legend(prop=fp_legend, loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.15, axis="x")
ax1.tick_params(labelsize=9)

# ---- Top-right: After centering (data distribution) ----
ax2 = fig.add_subplot(2, 2, 2)
scatter_colors = [COLORS["blue"] if xc < 0 else COLORS["red"] for xc in features_centered]
ax2.scatter(features_centered, np.ones_like(features_centered) * 0.5,
            c=scatter_colors, s=60, zorder=5, edgecolors="white", linewidth=0.5)
ax2.axvline(x=0, color="#64748b", ls="--", lw=1.2, alpha=0.7,
            label="新均值 = 0")

# Show gradient push directions (left for positive x, right for negative x)
for xp, xc in zip(features, features_centered):
    direction = 2.5 if xc > 0 else -2.5
    color = COLORS["red"] if xc > 0 else COLORS["blue"]
    ax2.arrow(xc, 0.5, direction, 0, head_width=0.06, head_length=0.6,
              fc=color, ec=color, alpha=0.35, lw=0.8)

ax2.set_ylim(0, 1.2); ax2.set_xlim(-13, 13)
ax2.set_yticks([])
x_ticks = [-10, -5, 0, 5, 10]
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f"{v:.0f}" for v in x_ticks])
ax2.set_title("中心化后：正/负推力各一半，自然平衡", fontproperties=fp_lg)
ax2.set_xlabel("x_centered = x − mean(x)", fontproperties=fp_md)
ax2.legend(prop=fp_legend, loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.15, axis="x")
ax2.tick_params(labelsize=9)

# ---- Bottom: Training effect — W trajectory with vs without centering ----
# Simulate W convergence with same data, same lr, same MSE, but with/without centering
np.random.seed(42)
TRUE_W, TRUE_BIAS = 2.0, 10.0
INIT_W, INIT_BIAS = 1.0, 0.0
LR = 0.01
EPOCHS = 100

# Without centering: raw features
W_raw, bias_raw = INIT_W, INIT_BIAS
traj_raw_w = [W_raw]
for _ in range(EPOCHS):
    for xp, yp in zip(features, labels):
        pred = W_raw * xp + bias_raw
        diff = pred - yp
        W_raw -= LR * 2 * diff * xp / N
        bias_raw -= LR * 2 * diff / N
    traj_raw_w.append(W_raw)

# With centering
W_cen, bias_cen = INIT_W, INIT_BIAS
traj_cen_w = [W_cen]
for _ in range(EPOCHS):
    for xp, yp in zip(features_centered, labels):
        pred = W_cen * xp + bias_cen
        diff = pred - yp
        W_cen -= LR * 2 * diff * xp / N
        bias_cen -= LR * 2 * diff / N
    traj_cen_w.append(W_cen)

ax3 = fig.add_subplot(2, 1, 2)
epochs_arr = np.arange(EPOCHS + 1)
ax3.plot(epochs_arr, traj_raw_w, COLORS["red"], lw=2,
         label="W（无中心化）— 锯齿震荡")
ax3.plot(epochs_arr, traj_cen_w, COLORS["blue"], lw=2,
         label="W（中心化）— 平滑收敛")
ax3.axhline(y=TRUE_W, color="#22c55e", ls="--", lw=1.2, alpha=0.6,
            label=f"目标 W = {TRUE_W}")

# Annotate zigzag vs smooth
ax3.annotate("左右来回甩\n过冲了没人拉",
             (35, traj_raw_w[35]), textcoords="offset points",
             xytext=(30, 25), fontsize=8, color=COLORS["red"],
             fontproperties=fp_sm,
             arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1))
ax3.annotate("正负推力平衡\n每一步都干净",
             (60, traj_cen_w[60]), textcoords="offset points",
             xytext=(30, -25), fontsize=8, color=COLORS["blue"],
             fontproperties=fp_sm,
             arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1))

ax3.set_xlim(0, EPOCHS); ax3.set_ylim(0.8, 2.8)
ax3.grid(True, alpha=0.2)
ax3.legend(prop=fp_legend, loc="center right", fontsize=8)
ax3.set_title("训练效果对比：W 参数收敛轨迹（相同学习率、相同 MSE）",
            fontproperties=fp_lg)
ax3.set_xlabel("Epoch", fontproperties=fp_md)
ax3.set_ylabel("W", fontproperties=fp_md)
ax3.tick_params(labelsize=9)

fig.suptitle("中心化：把数据摆对称，让梯度自己平衡",
            fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_V2 / "centering.png", dpi=144)
plt.close(fig)
print("  ✓ v2/centering.png")

# ================================================================
#  4. v1/loss-comparison.png — L(bias) landscape: MAE vs MSE
#     Shows that MSE gradient carries distance → converges faster
# ================================================================
# Simulate bias convergence on L(bias) for a fixed W
# Using real data from linearData(12, 20) with seed 42
W_FIXED = 2.0   # assume W is near optimal
TRUE_BIAS = 10
LR_MAE = 0.1
LR_MSE = 0.01

# MSE: L(bias) = bias² + b·bias + c  (a=1 for bias)
# Actual: L(bias) = (1/n)·Σ(W·x + bias - y)² = bias² + 2(W·x̄ - ȳ)·bias + const
# MAE: L(bias) = (1/n)·Σ|W·x + bias - y|  (piecewise linear, not quadratic)

bias_range = np.linspace(5, 15, 500)

# Compute actual MAE and MSE for each bias value
def compute_mae(bias):
    preds = W_FIXED * features + bias
    return np.mean(np.abs(preds - labels))

def compute_mse(bias):
    preds = W_FIXED * features + bias
    return np.mean((preds - labels) ** 2)

mae_losses = np.array([compute_mae(b) for b in bias_range])
mse_losses = np.array([compute_mse(b) for b in bias_range])

# Simulate trajectories
np.random.seed(42)
INIT_BIAS = 0.0
STEPS = 80

# MAE trajectory
b_mae = INIT_BIAS
traj_mae = [b_mae]
for _ in range(STEPS):
    preds = features * W_FIXED + b_mae
    diffs = preds - labels
    grad = np.mean(np.sign(diffs))
    b_mae -= LR_MAE * grad
    traj_mae.append(b_mae)

# MSE trajectory
b_mse = INIT_BIAS
traj_mse = [b_mse]
for _ in range(STEPS):
    preds = features * W_FIXED + b_mse
    diffs = preds - labels
    grad = 2 * np.mean(diffs)
    b_mse -= LR_MSE * grad
    traj_mse.append(b_mse)

traj_mae = np.array(traj_mae)
traj_mse = np.array(traj_mse)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

# -- Left: MAE --
ax1.plot(bias_range, mae_losses, COLORS["red"], lw=2.5, label="MAE Loss L(bias)")
valley = bias_range[np.argmin(mae_losses)]
ax1.axvline(x=valley, color="#22c55e", ls="--", lw=1, alpha=0.5, label=f"谷底 bias≈{valley:.1f}")

# Trajectory dots
step_interval = 8
ax1.scatter(traj_mae[::step_interval], [compute_mae(b) for b in traj_mae[::step_interval]],
            c=COLORS["red"], s=20, zorder=10, alpha=0.8)
ax1.plot(traj_mae, [compute_mae(b) for b in traj_mae],
         COLORS["red"], lw=0.8, alpha=0.3)

# Show step sizes are constant
ax1.annotate("每一步大小\n都一样（±0.1）",
             (traj_mae[10], compute_mae(traj_mae[10])),
             textcoords="offset points", xytext=(-60, 20),
             fontsize=8, color="#64748b", fontproperties=fp_sm,
             arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))

ax1.set_xlim(5, 15)
y_max = max(np.max(mae_losses), np.max(mse_losses))
yl = max(0, min(np.min(mae_losses)*0.8, np.min(mse_losses)*0.8))
ax1.set_ylim(yl, y_max * 1.1)
ax1.grid(True, alpha=0.2)
ax1.legend(prop=fp_legend, loc="upper left")
ax1.set_title("MAE：grad = ±1，永不减速", fontproperties=fp_lg)
ax1.set_xlabel("bias", fontproperties=fp_md)
ax1.set_ylabel("Loss", fontproperties=fp_md)
ax1.tick_params(labelsize=9)

# -- Right: MSE --
ax2.plot(bias_range, mse_losses, "#8b5cf6", lw=2.5, label="MSE Loss L(bias)")
valley_mse = bias_range[np.argmin(mse_losses)]
ax2.axvline(x=valley_mse, color="#22c55e", ls="--", lw=1, alpha=0.5, label=f"谷底 bias≈{valley_mse:.1f}")

ax2.scatter(traj_mse[::step_interval], [compute_mse(b) for b in traj_mse[::step_interval]],
            c="#8b5cf6", s=20, zorder=10, alpha=0.8)
ax2.plot(traj_mse, [compute_mse(b) for b in traj_mse],
         "#8b5cf6", lw=0.8, alpha=0.3)

# Show step sizes decreasing near minimum
ax2.annotate("越近谷底\n步子越小",
             (traj_mse[-30], compute_mse(traj_mse[-30])),
             textcoords="offset points", xytext=(30, 20),
             fontsize=8, color="#64748b", fontproperties=fp_sm,
             arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))

ax2.set_xlim(5, 15)
ax2.set_ylim(yl, y_max * 1.1)
ax2.grid(True, alpha=0.2)
ax2.legend(prop=fp_legend, loc="upper right")
ax2.set_title("MSE：grad = 2·diff，自动减速", fontproperties=fp_lg)
ax2.set_xlabel("bias", fontproperties=fp_md)
ax2.tick_params(labelsize=9)

fig.suptitle("同一个起点 (bias=0)，MSE 比 MAE 更快收敛到谷底",
            fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT_V1 / "loss-comparison.png", dpi=144)
plt.close(fig)
print("  ✓ v1/loss-comparison.png")

# ================================================================
print(f"\n✓ All {4} images generated:")
for f in sorted(list(OUT_V2.glob("*.png")) + [OUT_V1 / "loss-comparison.png"]):
    if f.exists():
        print(f"  {f.parent.name}/{f.name}  ({f.stat().st_size:,} bytes)")

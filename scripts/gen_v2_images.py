"""
Generate v2 tutorial images with CJK support.
Output: apps/_docs/public/v2/ + apps/_docs/public/v1/loss-comparison.png
Run: python3 scripts/gen_v2_images.py

Images generated:
  1. v2/loss-comparison.png  — MAE vs MSE: loss curves + gradients side-by-side
  2. v2/sgd-fluctuation.png  — BGD (smooth) vs SGD (jittery) loss over epochs
  3. v2/centering.png        — feature distribution before/after mean-centering
  4. v1/loss-comparison.png  — L(bias) landscape: MAE vs MSE trajectory to valley
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT_V2 = Path(__file__).resolve().parent.parent / "apps/_docs/public/v2"
OUT_V1 = Path(__file__).resolve().parent.parent / "apps/_docs/public/v1"
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
#  1. v2/loss-comparison.png — MAE vs MSE: loss curves + gradients
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
#  3. v2/centering.png — Feature distribution before/after centering
# ================================================================
np.random.seed(42)
# Reproduce v1 data exactly
N, MAX_X = 12, 20
features = np.linspace(0, MAX_X, N)
noise = (np.random.rand(N) - 0.5) * 6
labels = 2 * features + 10 + noise

mean_x = np.mean(features)
features_centered = features - mean_x

fig = plt.figure(figsize=(10, 5.5))

# ---- Top: Before centering ----
ax1 = fig.add_subplot(2, 1, 1)
# Strip plot + histogram
ax1.scatter(features, np.ones_like(features) * 0.5,
            c=COLORS["red"], s=80, zorder=5, edgecolors="white", linewidth=0.5)
# vertical mean line
ax1.axvline(x=mean_x, color="#64748b", ls="--", lw=1.5, alpha=0.7,
            label=f"均值 = {mean_x:.1f}")
# Annotate key insight
for x_val, label in [(0, "x 全是正数"), (MAX_X, "范围 [0, 20]")]:
    ax1.annotate(label, (x_val, 0.5), textcoords="offset points",
                xytext=(0, -25 if x_val == MAX_X else -25),
                fontsize=9, color="#64748b", ha="center",
                fontproperties=fp_sm)

# Gradient direction annotation
ax1.annotate("W 的梯度 ∝ x\n所有 x>0 → 推力永远同号 → 刹不住车",
             xy=(mean_x, 0.5), textcoords="offset points",
             xytext=(0, 30), fontsize=9, color="#dc2626",
             ha="center", fontproperties=fp_sm,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2",
                       edgecolor="#ef4444", alpha=0.9))

ax1.set_ylim(0, 1.2)
ax1.set_xlim(-3, 23)
ax1.set_yticks([])
ax1.set_title("原始数据：x 全部在 0 右侧，梯度推力永远同号", fontproperties=fp_lg)
ax1.set_xlabel("x", fontproperties=fp_md)
ax1.legend(prop=fp_legend, loc="upper right")
ax1.grid(True, alpha=0.15, axis="x")
ax1.tick_params(labelsize=9)

# ---- Bottom: After centering ----
ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(features_centered, np.ones_like(features_centered) * 0.5,
            c=COLORS["blue"], s=80, zorder=5, edgecolors="white", linewidth=0.5)
ax2.axvline(x=0, color="#64748b", ls="--", lw=1.5, alpha=0.7,
            label="新均值 = 0")

# Color-code positive and negative x
ax2.axvspan(-0.5, 0, alpha=0.05, color="#3b82f6", label="负 x：反向推 W")
ax2.axvspan(0, 0.5, alpha=0.05, color="#ef4444", label="正 x：正向推 W")

ax2.annotate("梯度天然平衡：\n一半正推力，一半负推力",
             xy=(0, 0.5), textcoords="offset points",
             xytext=(0, 30), fontsize=9, color="#2563eb",
             ha="center", fontproperties=fp_sm,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff",
                       edgecolor="#3b82f6", alpha=0.9))

ax2.set_ylim(0, 1.2)
ax2.set_xlim(-13, 13)
ax2.set_yticks([])
x_ticks = [-10, -5, 0, 5, 10]
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f"{v:.0f}" for v in x_ticks])
ax2.set_title("中心化后：以 0 对称分布，梯度推力自然平衡", fontproperties=fp_lg)
ax2.set_xlabel("x_centered = x − mean(x)", fontproperties=fp_md)
ax2.legend(prop=fp_legend, loc="upper right")
ax2.grid(True, alpha=0.15, axis="x")
ax2.tick_params(labelsize=9)

fig.suptitle("均值中心化：把数据摆对称，让梯度自己平衡",
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

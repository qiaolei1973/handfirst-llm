"""
v4 教学图片 — 全部使用设计数据，不需要训练。
Output: apps/_docs/public/v4/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "public" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_xs   = fm.FontProperties(fname=FONT_PATH, size=7.5)
fp_sm   = fm.FontProperties(fname=FONT_PATH, size=8.5)
fp_md   = fm.FontProperties(fname=FONT_PATH, size=9.5)
fp_lg   = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl   = fm.FontProperties(fname=FONT_PATH, size=12.5)
fp_bold = fm.FontProperties(fname=FONT_PATH, size=9, weight="bold")
fp_legend = fm.FontProperties(fname=FONT_PATH, size=8.5)

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
labels = true_y + (np.random.rand(N_DATA) - 0.5) * 0.3

# ================================================================
#  1. standardization-comparison.png — 三列直方图
# ================================================================
x_centered = x_orig - np.mean(x_orig)
x_standardized = x_centered / np.std(x_orig)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
datasets = [
    (x_orig, "原始数据", COLORS["gray_dark"], "mean=3.14  std=1.81"),
    (x_centered, "中心化（v2）", COLORS["blue"], "mean=0  std=1.81"),
    (x_standardized, "标准化（v4）", COLORS["purple"], "mean=0  std=1.00"),
]
for i, (data, title, color, stat_text) in enumerate(datasets):
    ax = axes[i]
    ax.hist(data, bins=14, color=color, alpha=0.6, edgecolor=color, lw=1.2)
    ax.axvline(x=np.mean(data), color=color, lw=2.5, ls="--", alpha=0.7)
    ax.set_title(title, fontproperties=fp_lg, color=color, pad=8)
    ax.set_xlabel("x", fontproperties=fp_sm)
    if i == 0: ax.set_ylabel("频次", fontproperties=fp_sm)
    ax.text(0.97, 0.95, stat_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontproperties=fp_bold, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.85, lw=1.2))
    ax.grid(True, alpha=0.15); ax.tick_params(labelsize=8)
fig.text(0.77, 0.15, "÷ std 统一尺度", ha="center", fontsize=8.5,
         color=COLORS["purple"], fontproperties=fp_sm)
fig.suptitle("中心化 vs 标准化", fontproperties=fp_xl, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "standardization-comparison.png", dpi=144, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  v4/standardization-comparison.png")

# ================================================================
#  2. optimizer-comparison.png — 合成 loss 曲线
#
#  教学要点：SGD 震荡、Momentum 平滑、Adam 收敛最快。
#  使用指数衰减 + 不同噪声强度伪随机生成——每个优化器的"性格"一目了然。
# ================================================================
EPOCHS = 2000
t = np.arange(0, EPOCHS, 20)
np.random.seed(1)

# Base: same exponential decay for all, different noise
base = 0.01 + 0.5 * np.exp(-t / 400) + 0.05 * np.exp(-t / 40)

# SGD: large noise, long tail
sgd = base + np.random.randn(len(t)) * 0.03 + np.sin(t / 50) * 0.008 * np.exp(-t / 600)
# Momentum: medium noise, smoothed
mom_noise = np.random.randn(len(t)) * 0.015
mom = base + mom_noise * 0.7
# Adam: tiny noise, fastest decay
adam_base = 0.01 + 0.5 * np.exp(-t / 200) + 0.03 * np.exp(-t / 25)
adam = adam_base + np.random.randn(len(t)) * 0.005

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(t, sgd, COLORS["red"], lw=1.2, alpha=0.7)
ax.plot(t, mom, COLORS["amber"], lw=1.5, alpha=0.8)
ax.plot(t, adam, COLORS["purple"], lw=2.0)

# Smoothed trend lines
def smooth(y, w=5):
    return np.convolve(y, np.ones(w)/w, mode='same')

ax.plot(t, smooth(sgd, 7), COLORS["red"], lw=2.2, label="SGD\nw -= lr*grad")
ax.plot(t, smooth(mom, 7), COLORS["amber"], lw=2.2, label="Momentum\nv = 0.9·v+0.1·grad")
ax.plot(t, smooth(adam, 7), COLORS["purple"], lw=2.2, label="Adam\nm+v+偏差修正")

ax.text(1600, 0.045, "SGD: 震荡大", fontsize=8.5, color=COLORS["red"], fontproperties=fp_sm)
ax.text(1600, 0.025, "Momentum: 惯性平滑", fontsize=8.5, color=COLORS["amber"], fontproperties=fp_sm)
ax.text(200, 0.015, "Adam: 最快最稳", fontsize=8.5, color=COLORS["purple"], fontproperties=fp_bold)
ax.set_xlabel("Epoch", fontproperties=fp_md)
ax.set_ylabel("MSE Loss (train)", fontproperties=fp_md)
ax.set_title("SGD → Momentum → Adam：Loss 下降对比（合成数据）", fontproperties=fp_lg, pad=10)
ax.legend(prop=fp_legend, loc="upper right", framealpha=0.85)
ax.grid(True, alpha=0.15); ax.tick_params(labelsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "optimizer-comparison.png", dpi=144, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  v4/optimizer-comparison.png")

# ================================================================
#  3. overfitting.png — 合成 train/val loss + 模型预测
#
#  教学要点：训练 loss 持续下降、验证 loss 先降后升 = 过拟合。
#  Early stop 模型是平滑 sin，过拟合模型在训练点之间乱抖。
# ================================================================
OF_EPOCHS = 3000
te = np.arange(0, OF_EPOCHS, 20)
np.random.seed(5)

# Train loss: keeps decreasing
train_loss = 0.005 + 0.5 * np.exp(-te / 300) + 0.02 * np.exp(-te / 60) + np.random.randn(len(te)) * 0.003
# Val loss: decreases then rises
val_decay = 0.008 + 0.5 * np.exp(-te / 350) + 0.02 * np.exp(-te / 80)
val_rise = np.maximum(0, 0.00025 * (te - 1000))
val_loss = val_decay + val_rise + np.random.randn(len(te)) * 0.004

# Best epoch
best_idx = np.argmin(val_loss)
best_epoch = int(te[best_idx])
best_val = float(val_loss[best_idx])

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7.5))

ax_top.plot(te, train_loss, COLORS["blue"], lw=2, label="训练 Loss（模型看到的）", alpha=0.85)
ax_top.plot(te, val_loss, COLORS["red"], lw=2, label="验证 Loss（模型看不到的）", alpha=0.85)
ax_top.axvline(x=best_epoch, color=COLORS["green"], lw=2.5, ls="--", alpha=0.8)
ax_top.annotate(f"Early Stop\nepoch={best_epoch}\nval loss 最低",
                (best_epoch, best_val), textcoords="offset points", xytext=(35, 20),
                fontsize=8.5, color=COLORS["green"], fontproperties=fp_bold,
                arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["green"], alpha=0.85))
overfit_start = best_epoch + 200
ax_top.axvspan(overfit_start, OF_EPOCHS, alpha=0.06, color=COLORS["red"])
ax_top.text((overfit_start + OF_EPOCHS) / 2, ax_top.get_ylim()[1] * 0.5,
            "过拟合区域\n(train↓ val↑)", ha="center", fontsize=8.5,
            color=COLORS["red"], fontproperties=fp_bold, alpha=0.7)
ax_top.set_xlabel("Epoch", fontproperties=fp_sm)
ax_top.set_ylabel("MSE Loss", fontproperties=fp_sm)
ax_top.set_title("训练 Loss 继续下降，但验证 Loss 开始上升 → 过拟合", fontproperties=fp_lg, pad=8)
ax_top.legend(prop=fp_legend, loc="upper right")
ax_top.grid(True, alpha=0.15); ax_top.tick_params(labelsize=8)

# Bottom: model predictions — designed "good" vs "overfit"
x_plot = np.linspace(0, 1, 300)
x_plot_orig = x_plot * X_SCALE
y_true = np.sin(x_plot_orig)

# "Good" model: smooth sin approximation (early stop)
y_good = y_true + np.random.randn(len(x_plot)) * 0.08

# "Overfit" model: wiggly, hits training points too exactly
# Add high-frequency oscillation between training points
wiggle = 0.3 * np.sin(x_plot * 25) * np.exp(-((x_plot - 0.5)**2) / 0.05) + \
         0.15 * np.sin(x_plot * 40)
y_overfit = y_true + wiggle * 0.4

# Train/val split
rng = np.random.RandomState(42)
n_train = 40
idx = rng.permutation(N_DATA)
x_train_raw = x_norm[idx[:n_train]]; y_train = labels[idx[:n_train]]
x_val_raw = x_norm[idx[n_train:]];   y_val = labels[idx[n_train:]]

ax_bot.scatter(x_train_raw * X_SCALE, y_train, c=COLORS["blue"], s=30, zorder=5, alpha=0.6,
               label=f"训练集 ({n_train} 个点)")
ax_bot.scatter(x_val_raw * X_SCALE, y_val, c=COLORS["red"], s=30, zorder=5, alpha=0.6,
               label=f"验证集 ({N_DATA - n_train} 个点)")
ax_bot.plot(x_plot_orig, y_true, COLORS["green"], lw=1.8, ls="--", alpha=0.6, label="y = sin(x)")
ax_bot.plot(x_plot_orig, y_good, COLORS["green"], lw=2.5, label=f"Early Stop")
ax_bot.plot(x_plot_orig, y_overfit, COLORS["red"], lw=1.8, alpha=0.7, label="过拟合")

ax_bot.annotate("过拟合：训练点之间乱抖\n把噪声也学进去了",
                (x_plot_orig[180], y_overfit[180]),
                textcoords="offset points", xytext=(-30, -35), fontsize=8,
                color=COLORS["red"], fontproperties=fp_sm,
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["red"], alpha=0.8))
ax_bot.set_xlim(-0.3, X_SCALE + 0.3); ax_bot.set_ylim(-1.8, 1.8)
ax_bot.set_xlabel("x", fontproperties=fp_sm); ax_bot.set_ylabel("y", fontproperties=fp_sm)
ax_bot.set_title("Early Stop 学的是 sin，过拟合背的是噪声（合成教学数据）", fontproperties=fp_lg, pad=8)
ax_bot.legend(prop=fp_legend, loc="lower right", ncol=2, fontsize=7.5)
ax_bot.grid(True, alpha=0.15); ax_bot.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "overfitting.png", dpi=144, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  v4/overfitting.png")

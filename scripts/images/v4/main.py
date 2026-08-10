"""
Generate v4 tutorial images.
Output: apps/_docs/public/v4/

训练结果缓存在 precompute.npz，图片脚本只负责画图。
重新生成训练数据: python3 precompute.py
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent.parent / "apps/_docs/public/v4"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_xs  = fm.FontProperties(fname=FONT_PATH, size=7.5)
fp_sm  = fm.FontProperties(fname=FONT_PATH, size=8.5)
fp_md  = fm.FontProperties(fname=FONT_PATH, size=9.5)
fp_lg  = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl  = fm.FontProperties(fname=FONT_PATH, size=12.5)
fp_bold = fm.FontProperties(fname=FONT_PATH, size=9, weight="bold")
fp_title = fm.FontProperties(fname=FONT_PATH, size=11, weight="bold")
fp_legend = fm.FontProperties(fname=FONT_PATH, size=8.5)

COLORS = {
    "red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e",
    "amber": "#f59e0b", "purple": "#8b5cf6", "gray": "#94a3b8",
    "gray_dark": "#64748b", "pink": "#ec4899", "cyan": "#06b6d4",
    "orange": "#f97316", "indigo": "#6366f1",
}

np.random.seed(42)
X_SCALE = 2 * np.pi
N_DATA = 60
x_norm = np.linspace(0, 1, N_DATA)
x_orig = x_norm * X_SCALE
true_y = np.sin(x_orig)
noise = (np.random.rand(N_DATA) - 0.5) * 0.3
labels = true_y + noise
x_centered = x_orig - np.mean(x_orig)
x_standardized = (x_orig - np.mean(x_orig)) / np.std(x_orig)

# ---- Load precomputed training results ----
DATA = np.load(HERE / "precompute.npz")

# ================================================================
#  1. standardization-comparison.png
# ================================================================
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
    ax.text(0.97, 0.95, stat_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, fontproperties=fp_bold,
            color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor=color, alpha=0.85, lw=1.2))
    ax.grid(True, alpha=0.15); ax.tick_params(labelsize=8)
fig.text(0.77, 0.15, "← 除以 std\n统一了尺度", ha="center", fontsize=8.5,
         color=COLORS["purple"], fontproperties=fp_sm)
fig.suptitle("中心化 vs 标准化", fontproperties=fp_xl, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "standardization-comparison.png", dpi=144, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  v4/standardization-comparison.png")

# ================================================================
#  2. optimizer-comparison.png
# ================================================================
configs = [
    ("sgd",      COLORS["red"],    "SGD\nw -= lr*grad"),
    ("momentum", COLORS["amber"],  "Momentum\nv = 0.9·v+0.1·grad"),
    ("adam",     COLORS["purple"], "Adam\nm+v+偏差修正"),
]

all_histories = {}
for opt_name, color, label in configs:
    arr = DATA[f"opt_{opt_name}"]
    all_histories[opt_name] = [(int(r[0]), float(r[1]), float(r[2])) for r in arr]

fig, ax = plt.subplots(figsize=(10, 5.5))
for opt_name, color, label in configs:
    hist = all_histories[opt_name]
    epochs_arr  = np.array([h[0] for h in hist])
    train_losses = np.array([h[1] for h in hist])
    ax.plot(epochs_arr, train_losses, color=color, lw=2.2, label=label, alpha=0.9)
ax.text(1600, 0.048, "SGD: 震荡大，收敛慢", fontsize=8.5, color=COLORS["red"], fontproperties=fp_sm)
ax.text(1600, 0.028, "Momentum: 惯性平滑", fontsize=8.5, color=COLORS["amber"], fontproperties=fp_sm)
ax.text(200, 0.018, "Adam: 最快最稳", fontsize=8.5, color=COLORS["purple"], fontproperties=fp_bold)
ax.set_xlabel("Epoch", fontproperties=fp_md)
ax.set_ylabel("MSE Loss (train)", fontproperties=fp_md)
ax.set_title("SGD → Momentum → Adam：Loss 下降对比", fontproperties=fp_lg, pad=10)
ax.legend(prop=fp_legend, loc="upper right", framealpha=0.85)
ax.grid(True, alpha=0.15); ax.tick_params(labelsize=8.5)
ax.text(0.5, -0.18, "从 v1 到 v4：w -= lr*grad → Adam", transform=ax.transAxes,
        ha="center", fontsize=8.5, color=COLORS["gray_dark"], fontproperties=fp_sm)
fig.tight_layout()
fig.savefig(OUT / "optimizer-comparison.png", dpi=144, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  v4/optimizer-comparison.png")

# ================================================================
#  3. overfitting.png
# ================================================================
of = DATA
history_of = [(int(r[0]), float(r[1]), float(r[2])) for r in of["of_history"]]
best_epoch = int(of["of_best_epoch"])
best_val_loss = float(of["of_best_val"])
checkpoint_params = (of["of_best_hw"], of["of_best_hb"], of["of_best_ow"], float(of["of_best_ob"]))
final_params = (of["of_final_hw"], of["of_final_hb"], of["of_final_ow"], float(of["of_final_ob"]))
x_mean_of = float(of["of_x_mean"]); x_std_of = float(of["of_x_std"])

def predict_at(x_raw, params):
    hw_p, hb_p, ow_p, ob_p = params
    xs = (x_raw - x_mean_of) / x_std_of
    z = np.outer(xs, hw_p) + hb_p
    return np.maximum(0, z) @ ow_p + ob_p

# Reconstruct the overfitting data split (deterministic from seed)
rng = np.random.RandomState(42)
n_train_overfit = 40
idx_of = rng.permutation(N_DATA)
train_idx_of = idx_of[:n_train_overfit]
val_idx_of = idx_of[n_train_overfit:]
x_train_raw = x_norm[train_idx_of]; y_train = labels[train_idx_of]
x_val_raw = x_norm[val_idx_of];     y_val = labels[val_idx_of]
epochs_of = 3000

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7.5))
epochs_arr_of = np.array([h[0] for h in history_of])
train_losses_of = np.array([h[1] for h in history_of])
val_losses_of   = np.array([h[2] for h in history_of])

ax_top.plot(epochs_arr_of, train_losses_of, COLORS["blue"], lw=2, label="训练 Loss（模型看到的）", alpha=0.85)
ax_top.plot(epochs_arr_of, val_losses_of, COLORS["red"], lw=2, label="验证 Loss（模型看不到的）", alpha=0.85)
ax_top.axvline(x=best_epoch, color=COLORS["green"], lw=2.5, ls="--", alpha=0.8)
ax_top.annotate(f"Early Stop\nepoch={best_epoch}\nval loss 最低",
                (best_epoch, best_val_loss), textcoords="offset points", xytext=(35, 20),
                fontsize=8.5, color=COLORS["green"], fontproperties=fp_bold,
                arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["green"], alpha=0.85))
overfit_start = best_epoch + 100
ax_top.axvspan(overfit_start, epochs_of, alpha=0.06, color=COLORS["red"])
ax_top.text((overfit_start + epochs_of) / 2, ax_top.get_ylim()[1] * 0.5,
            "过拟合区域\n(train↓ val↑)", ha="center", fontsize=8.5,
            color=COLORS["red"], fontproperties=fp_bold, alpha=0.7)
ax_top.set_xlabel("Epoch", fontproperties=fp_sm)
ax_top.set_ylabel("MSE Loss", fontproperties=fp_sm)
ax_top.set_title('32 个神经元、只给 40 个训练点：模型开始「背诵」', fontproperties=fp_lg, pad=8)
ax_top.legend(prop=fp_legend, loc="upper right")
ax_top.grid(True, alpha=0.15); ax_top.tick_params(labelsize=8)

x_plot = np.linspace(0, 1, 300)
x_plot_orig = x_plot * X_SCALE
y_early = predict_at(x_plot, checkpoint_params)
y_overfit = predict_at(x_plot, final_params)
ax_bot.scatter(x_train_raw * X_SCALE, y_train, c=COLORS["blue"], s=30, zorder=5, alpha=0.6,
               label=f"训练集 ({n_train_overfit} 个点)")
ax_bot.scatter(x_val_raw * X_SCALE, y_val, c=COLORS["red"], s=30, zorder=5, alpha=0.6,
               label=f"验证集 ({N_DATA - n_train_overfit} 个点)")
ax_bot.plot(x_plot_orig, np.sin(x_plot_orig), COLORS["green"], lw=1.8, ls="--", alpha=0.6, label="y = sin(x)")
ax_bot.plot(x_plot_orig, y_early, COLORS["green"], lw=2.5, label=f"Early Stop (epoch={best_epoch})")
ax_bot.plot(x_plot_orig, y_overfit, COLORS["red"], lw=1.8, alpha=0.7, label=f"过拟合 (epoch={epochs_of})")
ax_bot.annotate("过拟合：把噪声也背下来了", (x_plot_orig[180], y_overfit[180]),
                textcoords="offset points", xytext=(-30, -35), fontsize=8, color=COLORS["red"],
                fontproperties=fp_sm, arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["red"], alpha=0.8))
ax_bot.set_xlim(-0.3, X_SCALE + 0.3); ax_bot.set_ylim(-1.8, 1.8)
ax_bot.set_xlabel("x", fontproperties=fp_sm)
ax_bot.set_ylabel("y", fontproperties=fp_sm)
ax_bot.set_title("Early Stop 学的是 sin，过拟合背的是噪声", fontproperties=fp_lg, pad=8)
ax_bot.legend(prop=fp_legend, loc="lower right", ncol=2, fontsize=7.5)
ax_bot.grid(True, alpha=0.15); ax_bot.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "overfitting.png", dpi=144, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  v4/overfitting.png")

"""
Generate v4 tutorial images with CJK support.
Output: apps/_docs/public/v4/
Run: python3 scripts/gen_v4_images.py

Images generated:
  1. standardization-comparison.png — raw vs centered vs standardized histograms
  2. optimizer-comparison.png    — SGD vs Momentum vs Adam loss curves
  3. overfitting.png             — train/val loss + early stopping + model fits
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "apps/_docs/public/v4"
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
    "red":       "#ef4444",
    "blue":      "#3b82f6",
    "green":     "#22c55e",
    "amber":     "#f59e0b",
    "purple":    "#8b5cf6",
    "gray":      "#94a3b8",
    "gray_dark": "#64748b",
    "pink":      "#ec4899",
    "cyan":      "#06b6d4",
    "orange":    "#f97316",
    "indigo":    "#6366f1",
}

np.random.seed(42)

# ================================================================
#  Shared: sin dataset
# ================================================================
X_SCALE = 2 * np.pi
N_DATA = 60
x_norm = np.linspace(0, 1, N_DATA)
x_orig = x_norm * X_SCALE
true_y = np.sin(x_orig)
noise = (np.random.rand(N_DATA) - 0.5) * 0.3
labels = true_y + noise

x_centered = x_orig - np.mean(x_orig)
x_standardized = (x_orig - np.mean(x_orig)) / np.std(x_orig)


# ================================================================
#  1. standardization-comparison.png
#     Three histograms: raw → centered → standardized
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
    ax.axvline(x=np.mean(data), color=color, lw=2.5, ls="--", alpha=0.7,
               label=f"mean={np.mean(data):.2f}")

    ax.set_title(title, fontproperties=fp_lg, color=color, pad=8)
    ax.set_xlabel("x", fontproperties=fp_sm)
    if i == 0:
        ax.set_ylabel("频次", fontproperties=fp_sm)

    # Stats box
    ax.text(0.97, 0.95, stat_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, fontproperties=fp_bold,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.85, lw=1.2))

    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)

# Annotation between middle and right panel
fig.text(0.77, 0.15, "← 除以 std\n统一了尺度", ha="center", fontsize=8.5,
         color=COLORS["purple"], fontproperties=fp_sm,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#faf5ff",
                   edgecolor=COLORS["purple"], alpha=0.7, lw=1))

fig.suptitle("中心化 vs 标准化：中心化只挪位置，标准化统一位置 + 尺度",
             fontproperties=fp_xl, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "standardization-comparison.png", dpi=144,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ v4/standardization-comparison.png")


# ================================================================
#  2. optimizer-comparison.png
#     Train 8-neuron MLP with SGD, Momentum, Adam; compare loss
# ================================================================
def train_mlp_with_optimizer(x, y, n_neurons=8, optimizer="sgd",
                              epochs=2000, lr=None, batch_size=40, seed=42):
    """
    Train a 1→N→1 ReLU MLP on standardized data.
    Returns list of (epoch, train_loss, val_loss).
    """
    rng = np.random.RandomState(seed)
    n = len(x)

    # ---- Train / val split (80/20) ----
    n_train = int(n * 0.8)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # ---- Standardize ----
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    x_train_s = (x_train - x_mean) / x_std
    x_val_s = (x_val - x_mean) / x_std

    # ---- Init weights ----
    hidden_w = rng.randn(n_neurons) * 0.5
    hidden_b = rng.randn(n_neurons) * 0.1
    output_w = rng.randn(n_neurons) * 0.5
    output_b = 0.0

    # ---- Optimizer defaults ----
    if lr is None:
        lr = {"sgd": 0.02, "momentum": 0.02, "adam": 0.01}[optimizer]

    if optimizer == "momentum":
        beta = 0.9
        v_hw = np.zeros_like(hidden_w); v_hb = np.zeros_like(hidden_b)
        v_ow = np.zeros_like(output_w); v_ob = 0.0

    if optimizer == "adam":
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8; t_step = 0
        m_hw = np.zeros_like(hidden_w); v_hw = np.zeros_like(hidden_w)
        m_hb = np.zeros_like(hidden_b); v_hb = np.zeros_like(hidden_b)
        m_ow = np.zeros_like(output_w); v_ow = np.zeros_like(output_w)
        m_ob = 0.0; v_ob = 0.0

    n_train_batch = len(x_train_s)
    indices = np.arange(n_train_batch)
    history = []

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_idx = indices[:batch_size]
        xb, yb = x_train_s[batch_idx], y_train[batch_idx]

        # ---- Forward ----
        z = np.outer(xb, hidden_w) + hidden_b
        h = np.maximum(0, z)
        y_pred = h @ output_w + output_b

        # ---- Backward ----
        diff = y_pred - yb
        grad_out = (2 * diff) / batch_size
        grad_output_w = h.T @ grad_out
        grad_output_b = np.sum(grad_out)
        grad_h = np.outer(grad_out, output_w)
        grad_preact = grad_h * (z > 0)
        grad_hidden_w = xb @ grad_preact
        grad_hidden_b = np.sum(grad_preact, axis=0)

        # ---- Update ----
        if optimizer == "sgd":
            hidden_w  -= lr * grad_hidden_w
            hidden_b  -= lr * grad_hidden_b
            output_w  -= lr * grad_output_w
            output_b  -= lr * grad_output_b

        elif optimizer == "momentum":
            v_hw = beta * v_hw + (1 - beta) * grad_hidden_w
            v_hb = beta * v_hb + (1 - beta) * grad_hidden_b
            v_ow = beta * v_ow + (1 - beta) * grad_output_w
            v_ob = beta * v_ob + (1 - beta) * grad_output_b
            hidden_w -= lr * v_hw; hidden_b -= lr * v_hb
            output_w -= lr * v_ow; output_b -= lr * v_ob

        elif optimizer == "adam":
            t_step += 1
            # Hidden W
            m_hw = beta1 * m_hw + (1 - beta1) * grad_hidden_w
            v_hw = beta2 * v_hw + (1 - beta2) * grad_hidden_w**2
            mh = m_hw / (1 - beta1**t_step)
            vh = v_hw / (1 - beta2**t_step)
            hidden_w -= lr * mh / (np.sqrt(vh) + eps)
            # Hidden B
            m_hb = beta1 * m_hb + (1 - beta1) * grad_hidden_b
            v_hb = beta2 * v_hb + (1 - beta2) * grad_hidden_b**2
            mhb = m_hb / (1 - beta1**t_step)
            vhb = v_hb / (1 - beta2**t_step)
            hidden_b -= lr * mhb / (np.sqrt(vhb) + eps)
            # Output W
            m_ow = beta1 * m_ow + (1 - beta1) * grad_output_w
            v_ow = beta2 * v_ow + (1 - beta2) * grad_output_w**2
            mow = m_ow / (1 - beta1**t_step)
            vow = v_ow / (1 - beta2**t_step)
            output_w -= lr * mow / (np.sqrt(vow) + eps)
            # Output B
            m_ob = beta1 * m_ob + (1 - beta1) * grad_output_b
            v_ob = beta2 * v_ob + (1 - beta2) * grad_output_b**2
            mob = m_ob / (1 - beta1**t_step)
            vob = v_ob / (1 - beta2**t_step)
            output_b -= lr * mob / (np.sqrt(vob) + eps)

        # ---- Record loss every 20 epochs ----
        if epoch % 20 == 0:
            # Train loss (full train set)
            zt = np.outer(x_train_s, hidden_w) + hidden_b
            ht = np.maximum(0, zt)
            yt = ht @ output_w + output_b
            train_loss = np.mean((yt - y_train)**2)
            # Val loss
            zv = np.outer(x_val_s, hidden_w) + hidden_b
            hv = np.maximum(0, zv)
            yv = hv @ output_w + output_b
            val_loss = np.mean((yv - y_val)**2)
            history.append((epoch, train_loss, val_loss))

    # ---- Final prediction function ----
    def predict(x_raw):
        xs = (x_raw - x_mean) / x_std
        z = np.outer(xs, hidden_w) + hidden_b
        return np.maximum(0, z) @ output_w + output_b

    return history, predict


# Train with all 3 optimizers
configs = [
    ("sgd",      COLORS["red"],    "SGD\nw -= lr*grad"),
    ("momentum", COLORS["amber"],  "Momentum\nv = 0.9·v+0.1·grad"),
    ("adam",     COLORS["purple"], "Adam\nm+v+偏差修正"),
]

# Use x_norm (scaled to [0,1]) as raw input so standardization is meaningful
all_histories = {}
all_predicts = {}
for opt_name, color, label in configs:
    hist, pred_fn = train_mlp_with_optimizer(
        x_norm, labels, n_neurons=8, optimizer=opt_name, epochs=2000, seed=42
    )
    all_histories[opt_name] = hist
    all_predicts[opt_name] = pred_fn
    final_train = hist[-1][1]
    final_val = hist[-1][2]
    print(f"  trained {opt_name:8s}: train_loss={final_train:.6f}  val_loss={final_val:.6f}")


# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 5.5))

for opt_name, color, label in configs:
    hist = all_histories[opt_name]
    epochs_arr  = np.array([h[0] for h in hist])
    train_losses = np.array([h[1] for h in hist])
    ax.plot(epochs_arr, train_losses, color=color, lw=2.2, label=label, alpha=0.9)

# Annotations
ax.text(1600, 0.048, "SGD: 震荡大，收敛慢", fontsize=8.5, color=COLORS["red"],
        fontproperties=fp_sm)
ax.text(1600, 0.028, "Momentum: 惯性平滑", fontsize=8.5, color=COLORS["amber"],
        fontproperties=fp_sm)
ax.text(200, 0.018, "Adam: 最快最稳", fontsize=8.5, color=COLORS["purple"],
        fontproperties=fp_bold)

ax.set_xlabel("Epoch", fontproperties=fp_md)
ax.set_ylabel("MSE Loss (train)", fontproperties=fp_md)
ax.set_title("SGD → Momentum → Adam：Loss 下降对比（8 ReLU 神经元，sin 拟合）",
             fontproperties=fp_lg, pad=10)
ax.legend(prop=fp_legend, loc="upper right", framealpha=0.85)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=8.5)

# Insight box
ax.text(0.5, -0.18,
        "从 v1 到 v4：w -= lr*grad → Adam。优化器解决了三个问题：震荡（Momentum 惯性）、一刀切（RMSProp 自适应）、冷启动（偏差修正）",
        transform=ax.transAxes, ha="center", fontsize=8.5,
        color=COLORS["gray_dark"], fontproperties=fp_sm)

fig.tight_layout()
fig.savefig(OUT / "optimizer-comparison.png", dpi=144,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  ✓ v4/optimizer-comparison.png")


# ================================================================
#  3. overfitting.png
#     Train with 32 neurons / 40 train points → overfit
#     Show train+val loss curves + model predictions
# ================================================================
n_neurons_overfit = 32
n_train_overfit = 40  # leave 20 for val

rng = np.random.RandomState(42)
idx = rng.permutation(N_DATA)
train_idx = idx[:n_train_overfit]
val_idx = idx[n_train_overfit:]

x_train_raw, y_train = x_norm[train_idx], labels[train_idx]
x_val_raw,   y_val   = x_norm[val_idx],   labels[val_idx]

# Standardize
x_mean_of = np.mean(x_train_raw)
x_std_of  = np.std(x_train_raw)
x_train_s = (x_train_raw - x_mean_of) / x_std_of
x_val_s   = (x_val_raw   - x_mean_of) / x_std_of

# Init bigger network
hw = rng.randn(n_neurons_overfit) * 0.5
hb = rng.randn(n_neurons_overfit) * 0.1
ow = rng.randn(n_neurons_overfit) * 0.5
ob = 0.0

# Adam optimizer state
beta1, beta2 = 0.9, 0.999
eps = 1e-8; t_step = 0
m_hw = np.zeros_like(hw); v_hw = np.zeros_like(hw)
m_hb = np.zeros_like(hb); v_hb = np.zeros_like(hb)
m_ow = np.zeros_like(ow); v_ow = np.zeros_like(ow)
m_ob = 0.0; v_ob = 0.0

epochs_of = 3000
lr_of = 0.01
batch_size_of = 20  # smaller batch to make overfitting more visible
n_train_batch_of = len(x_train_s)
indices_of = np.arange(n_train_batch_of)

history_of = []                         # (epoch, train_loss, val_loss)
checkpoint_params = None                # best params (lowest val loss)
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(epochs_of):
    rng.shuffle(indices_of)
    batch_idx = indices_of[:batch_size_of]
    xb, yb = x_train_s[batch_idx], y_train[batch_idx]

    # Forward
    z = np.outer(xb, hw) + hb
    h = np.maximum(0, z)
    y_pred = h @ ow + ob

    # Backward
    diff = y_pred - yb
    grad_out = (2 * diff) / batch_size_of
    grad_ow = h.T @ grad_out
    grad_ob = np.sum(grad_out)
    grad_h = np.outer(grad_out, ow)
    grad_preact = grad_h * (z > 0)
    grad_hw = xb @ grad_preact
    grad_hb = np.sum(grad_preact, axis=0)

    # Adam update
    t_step += 1
    m_hw = beta1 * m_hw + (1 - beta1) * grad_hw
    v_hw = beta2 * v_hw + (1 - beta2) * grad_hw**2
    hw -= lr_of * (m_hw / (1 - beta1**t_step)) / (np.sqrt(v_hw / (1 - beta2**t_step)) + eps)
    m_hb = beta1 * m_hb + (1 - beta1) * grad_hb
    v_hb = beta2 * v_hb + (1 - beta2) * grad_hb**2
    hb -= lr_of * (m_hb / (1 - beta1**t_step)) / (np.sqrt(v_hb / (1 - beta2**t_step)) + eps)
    m_ow = beta1 * m_ow + (1 - beta1) * grad_ow
    v_ow = beta2 * v_ow + (1 - beta2) * grad_ow**2
    ow -= lr_of * (m_ow / (1 - beta1**t_step)) / (np.sqrt(v_ow / (1 - beta2**t_step)) + eps)
    m_ob = beta1 * m_ob + (1 - beta1) * grad_ob
    v_ob = beta2 * v_ob + (1 - beta2) * grad_ob**2
    ob -= lr_of * (m_ob / (1 - beta1**t_step)) / (np.sqrt(v_ob / (1 - beta2**t_step)) + eps)

    # Record & checkpoint every 20 epochs
    if epoch % 20 == 0:
        zt = np.outer(x_train_s, hw) + hb
        ht = np.maximum(0, zt)
        train_loss = np.mean((ht @ ow + ob - y_train)**2)
        zv = np.outer(x_val_s, hw) + hb
        hv = np.maximum(0, zv)
        val_loss = np.mean((hv @ ow + ob - y_val)**2)
        history_of.append((epoch, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_params = (hw.copy(), hb.copy(), ow.copy(), ob)


def predict_at(x_raw, params):
    hw_p, hb_p, ow_p, ob_p = params
    xs = (x_raw - x_mean_of) / x_std_of
    z = np.outer(xs, hw_p) + hb_p
    return np.maximum(0, z) @ ow_p + ob_p

# Get final params (last epoch for overfit comparison)
final_params = (hw.copy(), hb.copy(), ow.copy(), ob)

print(f"  overfitting experiment: best_epoch={best_epoch}  "
      f"best_val_loss={best_val_loss:.6f}  "
      f"final_train_loss={history_of[-1][1]:.6f}")

# ---- Plot: 2 rows ----
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7.5))

# ---- Top: loss curves ----
epochs_arr_of = np.array([h[0] for h in history_of])
train_losses_of = np.array([h[1] for h in history_of])
val_losses_of   = np.array([h[2] for h in history_of])

ax_top.plot(epochs_arr_of, train_losses_of, COLORS["blue"], lw=2,
            label="训练 Loss（模型看到的）", alpha=0.85)
ax_top.plot(epochs_arr_of, val_losses_of, COLORS["red"], lw=2,
            label="验证 Loss（模型看不到的）", alpha=0.85)

# Mark best epoch (early stopping point)
ax_top.axvline(x=best_epoch, color=COLORS["green"], lw=2.5, ls="--", alpha=0.8)
ax_top.annotate(f"Early Stop\nepoch={best_epoch}\nval loss 最低",
                (best_epoch, best_val_loss),
                textcoords="offset points", xytext=(35, 20),
                fontsize=8.5, color=COLORS["green"], fontproperties=fp_bold,
                arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COLORS["green"], alpha=0.85))

# Shade overfitting region
overfit_start = best_epoch + 100
ax_top.axvspan(overfit_start, epochs_of, alpha=0.06, color=COLORS["red"])
ax_top.text((overfit_start + epochs_of) / 2, ax_top.get_ylim()[1] * 0.5,
            "过拟合区域\n(train↓ val↑)", ha="center", fontsize=8.5,
            color=COLORS["red"], fontproperties=fp_bold, alpha=0.7)

ax_top.set_xlabel("Epoch", fontproperties=fp_sm)
ax_top.set_ylabel("MSE Loss", fontproperties=fp_sm)
ax_top.set_title('32 个神经元、只给 40 个训练点：模型开始「背诵」', fontproperties=fp_lg, pad=8)
ax_top.legend(prop=fp_legend, loc="upper right")
ax_top.grid(True, alpha=0.15)
ax_top.tick_params(labelsize=8)

# ---- Bottom: model predictions ----
x_plot = np.linspace(0, 1, 300)
x_plot_orig = x_plot * X_SCALE

# Early-stopped prediction
y_early = predict_at(x_plot, checkpoint_params)
# Overfit (final) prediction
y_overfit = predict_at(x_plot, final_params)

# Training & validation scatter
ax_bot.scatter(x_train_raw * X_SCALE, y_train, c=COLORS["blue"], s=30,
               zorder=5, alpha=0.6, label=f"训练集 ({n_train_overfit} 个点)")
ax_bot.scatter(x_val_raw * X_SCALE, y_val, c=COLORS["red"], s=30,
               zorder=5, alpha=0.6, label=f"验证集 ({N_DATA - n_train_overfit} 个点)")

# True sin
ax_bot.plot(x_plot_orig, np.sin(x_plot_orig), COLORS["green"], lw=1.8, ls="--",
            alpha=0.6, label="y = sin(x)")

# Early-stopped
ax_bot.plot(x_plot_orig, y_early, COLORS["green"], lw=2.5,
            label=f"Early Stop (epoch={best_epoch})")

# Overfit
ax_bot.plot(x_plot_orig, y_overfit, COLORS["red"], lw=1.8, alpha=0.7,
            label=f"过拟合 (epoch={epochs_of})")

# Annotate overfit wiggliness
ax_bot.annotate("过拟合：把噪声也背下来了\n训练点之间乱抖",
                (x_plot_orig[180], y_overfit[180]),
                textcoords="offset points", xytext=(-30, -35),
                fontsize=8, color=COLORS["red"], fontproperties=fp_sm,
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=COLORS["red"], alpha=0.8))

ax_bot.set_xlim(-0.3, X_SCALE + 0.3)
ax_bot.set_ylim(-1.8, 1.8)
ax_bot.set_xlabel("x", fontproperties=fp_sm)
ax_bot.set_ylabel("y", fontproperties=fp_sm)
ax_bot.set_title("Early Stop 学的是 sin，过拟合背的是噪声",
                 fontproperties=fp_lg, pad=8)
ax_bot.legend(prop=fp_legend, loc="lower right", ncol=2, fontsize=7.5)
ax_bot.grid(True, alpha=0.15)
ax_bot.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT / "overfitting.png", dpi=144, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  ✓ v4/overfitting.png")

# ================================================================
print(f"\n✓ All v4 images generated:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

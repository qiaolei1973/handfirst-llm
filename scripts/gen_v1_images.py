"""
Generate v1 tutorial images with CJK support.
Output: apps/_docs/public/v1/
Run: python3 scripts/gen_v1_images.py

Uses FontProperties(fname=...) for reliable CJK rendering.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "apps/_docs/public/v1"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp = fm.FontProperties(fname=FONT_PATH)
fp_sm = fm.FontProperties(fname=FONT_PATH, size=9)
fp_md = fm.FontProperties(fname=FONT_PATH, size=10)
fp_lg = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl = fm.FontProperties(fname=FONT_PATH, size=13)
fp_bold = fm.FontProperties(fname=FONT_PATH, size=10, weight="bold")
fp_legend = fm.FontProperties(fname=FONT_PATH, size=10)

# ---- reproducible data (linearData(12, 20) with seed 42) ----
np.random.seed(42)
N, MAX_X = 12, 20
features = np.linspace(0, MAX_X, N)
noise = (np.random.rand(N) - 0.5) * 6
labels = 2 * features + 10 + noise

TRUE_W, TRUE_BIAS = 2, 10
INIT_W, INIT_BIAS = 1.0, 0.0
LEARN_RATE = 0.1

def scatter_pts(ax, alpha=1.0):
    ax.scatter(features, labels, c="#3b82f6", s=48, zorder=5, alpha=alpha)

def draw_line(ax, W, bias, color, label, lw=2, ls="-"):
    x0, x1 = -1, MAX_X + 1
    ax.plot([x0, x1], [W*x0 + bias, W*x1 + bias],
            color=color, lw=lw, ls=ls, label=label)

def label_ax(ax, xlab, ylab, title):
    ax.set_xlabel(xlab, fontproperties=fp_md)
    ax.set_ylabel(ylab, fontproperties=fp_md)
    ax.set_title(title, fontproperties=fp_xl, pad=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

# ===== 1. scatter-only.png =====
fig, ax = plt.subplots(figsize=(7, 4.5))
scatter_pts(ax)
ax.set_xlim(-1, MAX_X + 1); ax.set_ylim(0, 55)
label_ax(ax, "x", "y", "计算机看到的只有这些点")
fig.tight_layout()
fig.savefig(OUT / "scatter-only.png", dpi=144)
plt.close(fig)

# ===== 2. initial-guess.png =====
fig, ax = plt.subplots(figsize=(7, 4.5))
scatter_pts(ax)
draw_line(ax, TRUE_W, TRUE_BIAS, "#10b981", "真实: y=2x+10", lw=2.5)
draw_line(ax, INIT_W, INIT_BIAS, "#a855f7",
          f"模型: y={INIT_W}x+{INIT_BIAS}", lw=2.5)
ax.set_xlim(-1, MAX_X + 1); ax.set_ylim(0, 55)
label_ax(ax, "x", "y", "第一步：随便猜一条线（W=1, bias=0）")
ax.legend(loc="upper left", prop=fp_legend)
fig.tight_layout()
fig.savefig(OUT / "initial-guess.png", dpi=144)
plt.close(fig)

# ===== 3. one-point-diff.png =====
fig, ax = plt.subplots(figsize=(7, 4.5))
scatter_pts(ax)
draw_line(ax, INIT_W, INIT_BIAS, "#a855f7", f"y={INIT_W}x+{INIT_BIAS}", lw=2.5)

idx = 3
xp, yp = features[idx], labels[idx]
ypred = INIT_W * xp + INIT_BIAS
diff_val = ypred - yp

ax.plot([xp, xp], [ypred, yp], "#ef4444", lw=3, zorder=6)

def annot(text, xy, xytext, color, **kw):
    ax.annotate(text, xy, textcoords="offset points", xytext=xytext,
                fontsize=9, color=color, fontproperties=fp_sm, **kw)

annot(f"预测 yPred={ypred:.1f}", (xp, ypred), (12, 8), "#a855f7",
      arrowprops=dict(arrowstyle="->", color="#a855f7"))
annot(f"真实 y={yp:.1f}", (xp, yp), (12, -12), "#3b82f6",
      arrowprops=dict(arrowstyle="->", color="#3b82f6"))
ax.annotate(f"diff = {diff_val:.1f}", (xp, (ypred+yp)/2),
            textcoords="offset points", xytext=(-40, 0),
            fontsize=10, fontproperties=fp_bold, color="#ef4444",
            arrowprops=dict(arrowstyle="->", color="#ef4444"))

ax.set_xlim(-1, MAX_X + 1); ax.set_ylim(-5, 55)
label_ax(ax, "x", "y", "第二步：看猜得有多差（只看一个点）")
fig.tight_layout()
fig.savefig(OUT / "one-point-diff.png", dpi=144)
plt.close(fig)

# ===== 4. all-points-error.png =====
fig, ax = plt.subplots(figsize=(7, 4.5))
scatter_pts(ax)
draw_line(ax, INIT_W, INIT_BIAS, "#a855f7", f"y={INIT_W}x+{INIT_BIAS}", lw=2.5)

for xp, yp in zip(features, labels):
    ypred = INIT_W * xp + INIT_BIAS
    ax.plot([xp, xp], [ypred, yp], "#ef4444", lw=1.5, alpha=0.7, zorder=3)

total = np.sum(np.abs(INIT_W * features + INIT_BIAS - labels))
ax.text(0.98, 0.92,
        f"总差距 = |diff1| + ... + |diff12| = {total:.0f}",
        transform=ax.transAxes, fontsize=10, ha="right",
        fontproperties=fp_md,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2",
                  edgecolor="#ef4444", alpha=0.9))

ax.set_xlim(-1, MAX_X + 1); ax.set_ylim(-5, 55)
label_ax(ax, "x", "y", "12 个点全算一遍——差距一目了然")
fig.tight_layout()
fig.savefig(OUT / "all-points-error.png", dpi=144)
plt.close(fig)

# ===== 5. direction-sign.png =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

scenarios = [
    (ax1, "点 A：预测值太大\ndiff > 0", 10, 30, 40, "W ↓ 往回拉", +2, -12),
    (ax2, "点 B：预测值太小\ndiff < 0", 10, 30, 20, "W ↑ 往前推", -2, 12),
]
for ax, title_text, xp, true_y, pred_y, direction, arrow_y_off, arrow_dy in scenarios:
    ax.scatter([xp], [true_y], c="#3b82f6", s=80, zorder=5, label="真实值")
    ax.scatter([xp], [pred_y], c="#a855f7", s=80, marker="D", zorder=5, label="预测值")
    ax.plot([xp, xp], [pred_y, true_y], "#ef4444", lw=2.5, zorder=3)
    ax.annotate(direction, (xp, pred_y + arrow_y_off),
                textcoords="offset points", xytext=(30, arrow_dy),
                fontsize=11, fontproperties=fp_lg, color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
    ax.set_xlim(4, 16); ax.set_ylim(10, 50)
    ax.set_title(title_text, fontproperties=fp_lg)
    ax.legend(fontsize=9, prop=fp_sm)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)

fig.suptitle("第三步：只看方向，不看大小", fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "direction-sign.png", dpi=144)
plt.close(fig)

# ===== 6. x-leverage.png =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

for ax, xp, desc in [(ax1, 2.0, "x=2：对 W 推力小"),
                       (ax2, 18.0, "x=18：对 W 推力大")]:
    true_y = 2 * xp + 10 + np.interp(xp, features, noise)
    pred_y = INIT_W * xp + INIT_BIAS
    ax.scatter([xp], [true_y], c="#3b82f6", s=100, zorder=5)
    ax.scatter([xp], [pred_y], c="#a855f7", s=80, marker="D", zorder=5)
    ax.plot([xp, xp], [pred_y, true_y], "#ef4444", lw=2, zorder=3)
    arrow_len = xp * 0.3
    ax.arrow(xp, pred_y, arrow_len * 0.5, arrow_len * 1.5,
             head_width=1.2, head_length=1.5,
             fc="#dc2626", ec="#dc2626", lw=2, zorder=10)
    ax.set_xlim(-1, MAX_X + 1); ax.set_ylim(-5, 55)
    ax.set_title(desc, fontproperties=fp_lg)
    ax.set_xlabel("x", fontproperties=fp_md)
    ax.set_ylabel("y", fontproperties=fp_md)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)

fig.suptitle("x 越大的点，对斜率的「话语权」越大", fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "x-leverage.png", dpi=144)
plt.close(fig)

# ===== 7. one-step.png =====
preds = INIT_W * features + INIT_BIAS
diffs = preds - labels
signs = np.sign(diffs)
gradW = np.mean(signs * features)
gradBias = np.mean(signs)
new_W = INIT_W - LEARN_RATE * gradW
new_B = INIT_BIAS - LEARN_RATE * gradBias

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, W, bias, label in [
    (ax1, INIT_W, INIT_BIAS,
     f"更新前: W={INIT_W:.2f}, bias={INIT_BIAS:.2f}"),
    (ax2, new_W, new_B,
     f"更新后: W={new_W:.2f}, bias={new_B:.2f}")
]:
    scatter_pts(ax)
    draw_line(ax, W, bias, "#a855f7", label, lw=2.5)
    for xp, yp in zip(features, labels):
        ypred = W * xp + bias
        ax.plot([xp, xp], [ypred, yp], "#ef4444", lw=1, alpha=0.5, zorder=3)
    ax.set_xlim(-1, MAX_X + 1); ax.set_ylim(-5, 55)
    ax.legend(fontsize=10, prop=fp_legend, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x", fontproperties=fp_md)
    ax.set_ylabel("y", fontproperties=fp_md)
    ax.tick_params(labelsize=9)

fig.suptitle(f"第四步：走一小步（学习率={LEARN_RATE}）——误差线全缩短了！",
             fontproperties=fp_xl)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "one-step.png", dpi=144)
plt.close(fig)

# ===== 8. mae-loss.png — MAE 损失函数 L = |diff|（V 形） =====
diff_range = np.linspace(-10, 10, 500)
mae_loss = np.abs(diff_range)

fig, ax = plt.subplots(figsize=(6, 4.3))

ax.plot(diff_range, mae_loss, "#ef4444", lw=2.5, label="MAE: L = |diff|")
ax.axvline(x=-5, color="#94a3b8", ls="--", lw=1, alpha=0.6)
ax.axvline(x=5, color="#94a3b8", ls="--", lw=1, alpha=0.6)
ax.axhline(y=5, color="#94a3b8", ls="--", lw=1, alpha=0.6)
ax.annotate("差 -5 → L=5", (-5, 0), textcoords="offset points", xytext=(0, -20),
            fontsize=8, color="#64748b", ha="center", fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))
ax.annotate("差 +5 → L=5", (5, 0), textcoords="offset points", xytext=(0, -20),
            fontsize=8, color="#64748b", ha="center", fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))
ax.set_xlabel("diff = yPred − 真实值", fontproperties=fp_md)
ax.set_ylabel("L（损失）", fontproperties=fp_md)
ax.set_title("损失函数 MAE：L = |diff|——差 5 就是 5，差多少就是多少",
            fontproperties=fp_lg)
ax.legend(prop=fp_sm, loc="upper right")
ax.grid(True, alpha=0.2)
ax.tick_params(labelsize=9)
ax.set_xlim(-10, 10); ax.set_ylim(-1, 11)

fig.tight_layout()
fig.savefig(OUT / "mae-loss.png", dpi=144)
plt.close(fig)

# ===== 9. mae-gradient.png — MAE 梯度 grad = sign(diff)（只看方向） =====
mae_grad = np.sign(diff_range)
mae_grad[diff_range == 0] = 0

fig, ax = plt.subplots(figsize=(6, 4.3))

colors_grad = ["#3b82f6" if d < 0 else "#ef4444" if d > 0 else "#94a3b8"
               for d in diff_range]
for i in range(len(diff_range)-1):
    ax.plot(diff_range[i:i+2], mae_grad[i:i+2], color=colors_grad[i], lw=2.5)
ax.plot([], [], "#ef4444", lw=2.5, label="diff>0 → grad=+1")
ax.plot([], [], "#3b82f6", lw=2.5, label="diff<0 → grad=−1")

ax.annotate("差 100\n推力也是 1", (8, 1), textcoords="offset points",
            xytext=(30, 15), fontsize=8, color="#64748b", ha="center",
            fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))
ax.annotate("差 0.01\n推力也是 1", (1, 1), textcoords="offset points",
            xytext=(30, -15), fontsize=8, color="#64748b", ha="center",
            fontproperties=fp_sm,
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8))
ax.set_xlabel("diff = yPred − 真实值", fontproperties=fp_md)
ax.set_ylabel("grad（梯度）", fontproperties=fp_md)
ax.set_title("MAE 的梯度：grad = sign(diff)——只看方向，不看距离",
            fontproperties=fp_lg)
ax.legend(prop=fp_sm, loc="upper left")
ax.grid(True, alpha=0.2)
ax.tick_params(labelsize=9)
ax.set_xlim(-10, 10); ax.set_ylim(-1.5, 1.5)

fig.tight_layout()
fig.savefig(OUT / "mae-gradient.png", dpi=144)
plt.close(fig)

print(f"✓ Generated images:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

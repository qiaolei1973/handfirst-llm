"""
Generate v3 backpropagation flow diagram.
Output: apps/_docs/public/v3/backprop.png

Shows forward pass (left) and backward pass (right) side by side,
with gradient formulas at each step.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "apps/_docs/public/v3"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_sm   = fm.FontProperties(fname=FONT_PATH, size=7.5)
fp_md   = fm.FontProperties(fname=FONT_PATH, size=8.5)
fp_lg   = fm.FontProperties(fname=FONT_PATH, size=9.5)
fp_xl   = fm.FontProperties(fname=FONT_PATH, size=11)
fp_bold = fm.FontProperties(fname=FONT_PATH, size=8.5, weight="bold")
fp_title = fm.FontProperties(fname=FONT_PATH, size=10, weight="bold")
fp_arrow = fm.FontProperties(fname=FONT_PATH, size=7)

COLORS = {
    "loss":    "#6366f1",   # indigo
    "out_wb":  "#10b981",   # emerald (output layer params)
    "out_grad":"#059669",
    "hidden_h":"#f59e0b",   # amber (hidden layer output)
    "relu":    "#ef4444",   # red
    "hidden_wb":"#3b82f6",  # blue (hidden layer params)
    "arrow_fwd":"#64748b",
    "arrow_bwd":"#ef4444",
    "bg":      "#fafbfc",
    "text":    "#1e293b",
}

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")

# ---- Layout constants ----
LEFT_CX   = 3.5    # center x of forward pass column
RIGHT_CX  = 10.5   # center x of backward pass column
BOX_W     = 3.0
BOX_H     = 0.65
GAP_Y     = 0.35
START_Y   = 7.0

# ---- Helper ----
def draw_box(x, y, w, h, text, color, text_color="auto", alpha=0.12, lw=2.5, fontsize=8.5):
    tc = text_color if text_color != "auto" else color
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=color,
                          alpha=alpha, lw=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tc, fontproperties=fp_bold, zorder=4)

def draw_arrow(x1, y1, x2, y2, color=COLORS["arrow_fwd"], lw=2, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                zorder=zorder)

def draw_formula(x, y, text, color, fontsize=7):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontproperties=fp_sm, zorder=4)

def draw_brace_arrow(x1, y1, x2, y2, color=COLORS["arrow_fwd"], lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle="arc3,rad=0"),
                zorder=2)

# ================================================================
#  LEFT: Forward Pass (top → bottom)
# ================================================================

ax.text(LEFT_CX, 7.85, "前向传播 (Forward)", ha="center", fontsize=11,
        color=COLORS["arrow_fwd"], fontproperties=fp_title)

# Input x
draw_box(LEFT_CX, 6.8, 0.8, 0.5, "x", COLORS["arrow_fwd"], alpha=0.08, lw=1.8, fontsize=9)
draw_arrow(LEFT_CX, 6.55, LEFT_CX, 5.9)

# Hidden layer: z = w·x + b
draw_box(LEFT_CX, 5.55, BOX_W, BOX_H, "z_j = w_j·x + b_j", COLORS["hidden_wb"])
draw_formula(LEFT_CX + 1.8, 5.55, "← Linear", COLORS["hidden_wb"], 7)
draw_arrow(LEFT_CX, 5.2, LEFT_CX, 4.65)

# ReLU
draw_box(LEFT_CX, 4.33, BOX_W, BOX_H, "h_j = ReLU(z_j)", COLORS["relu"])
draw_formula(LEFT_CX + 1.8, 4.33, "← 激活函数", COLORS["relu"], 7)
draw_arrow(LEFT_CX, 3.98, LEFT_CX, 3.43)

# Output layer
draw_box(LEFT_CX, 3.1, BOX_W, BOX_H, "yPred = sum  w_out_j·h_j + b_out", COLORS["out_wb"])
draw_formula(LEFT_CX + 1.8, 3.1, "← Linear", COLORS["out_wb"], 7)
draw_arrow(LEFT_CX, 2.75, LEFT_CX, 2.2)

# Loss
draw_box(LEFT_CX, 1.88, BOX_W, 0.6, "L = (yPred − y)²", COLORS["loss"])
draw_formula(LEFT_CX + 1.8, 1.88, "← MSE", COLORS["loss"], 7)

# Labels for forward
ax.text(LEFT_CX - 2.2, 5.55, "隐藏层", ha="right", va="center", fontsize=8,
        color=COLORS["hidden_wb"], fontproperties=fp_bold)
ax.text(LEFT_CX - 2.2, 3.1, "输出层", ha="right", va="center", fontsize=8,
        color=COLORS["out_wb"], fontproperties=fp_bold)

# Bracket enclosing hidden layer (z + ReLU)
b_hidden_left  = LEFT_CX - BOX_W/2 - 0.15
b_hidden_right = LEFT_CX + BOX_W/2 + 0.15
b_hidden_y     = 5.95
b_hidden_h     = 4.23
rect_hidden = FancyBboxPatch((b_hidden_left, b_hidden_h), b_hidden_right - b_hidden_left, b_hidden_y - b_hidden_h,
                              boxstyle="round,pad=0.1", facecolor="none",
                              edgecolor=COLORS["hidden_wb"], lw=1.5, ls="--", alpha=0.5, zorder=1)
ax.add_patch(rect_hidden)

# ================================================================
#  RIGHT: Backward Pass (bottom → top)
# ================================================================

ax.text(RIGHT_CX, 7.85, "反向传播 (Backward)", ha="center", fontsize=11,
        color=COLORS["arrow_bwd"], fontproperties=fp_title)

# Step 0: loss gradient
draw_box(RIGHT_CX, 1.88, BOX_W + 0.4, 0.6,
         "∂L/∂yPred = 2(yPred − y)", COLORS["loss"], lw=2.5)
# upward arrow
draw_arrow(RIGHT_CX, 2.2, RIGHT_CX, 2.75, color=COLORS["arrow_bwd"])

# Step 1: output layer gradients
draw_box(RIGHT_CX, 3.1, BOX_W + 0.4, BOX_H,
         "∂L/∂w_out = ∂L/∂yPred · h\n∂L/∂b_out = ∂L/∂yPred · 1",
         COLORS["out_wb"])

# Side formulas for step 1
draw_formula(RIGHT_CX + 1.9, 3.25, "∂L/∂h = ∂L/∂yPred · w_out", COLORS["out_wb"], 7)
draw_arrow(RIGHT_CX + 1.55, 3.25, RIGHT_CX + 0.45, 3.25, color=COLORS["out_wb"], lw=1)
draw_arrow(RIGHT_CX, 3.45, RIGHT_CX, 3.83, color=COLORS["arrow_bwd"])

# Step 2: ReLU gate
draw_box(RIGHT_CX, 4.33, BOX_W + 0.4, BOX_H,
         "∂L/∂z = ∂L/∂h · ReLU'(z)\n    (z>0→1, z≤0→0)",
         COLORS["relu"])
draw_arrow(RIGHT_CX, 4.68, RIGHT_CX, 5.05, color=COLORS["arrow_bwd"])

# Step 3: hidden layer gradients
draw_box(RIGHT_CX, 5.55, BOX_W + 0.4, BOX_H,
         "∂L/∂w = ∂L/∂z · x\n∂L/∂b = ∂L/∂z · 1",
         COLORS["hidden_wb"], lw=2.5)

# Step 4: receiver boxes (收/算/传) - annotation on the side
draw_box(RIGHT_CX + 2.2, 4.33, 1.7, 2.4, "", "#94a3b8", alpha=0.05, lw=1.5)
ax.text(RIGHT_CX + 2.2, 5.25, "backward() 三步", ha="center", fontsize=7.5,
        color="#64748b", fontproperties=fp_bold)
ax.text(RIGHT_CX + 2.2, 4.7,
        "① 收: ∂L/∂output\n② 算: W,b 的梯度\n③ 传: ∂L/∂input",
        ha="center", va="center", fontsize=7, color="#64748b",
        fontproperties=fp_sm)

# ================================================================
#  Center: "chain rule" vertical label & gradient flow
# ================================================================

# Thin red gradient path on the backward side (decorative dots along arrows)
for y in [2.5, 3.3, 4.1, 4.9]:
    ax.plot(RIGHT_CX, y, "o", color=COLORS["arrow_bwd"], markersize=4, alpha=0.4, zorder=5)

# ================================================================
#  Labels for right side
# ================================================================
ax.text(RIGHT_CX - 2.3, 3.1, "输出层\nbackward()", ha="right", va="center", fontsize=7.5,
        color=COLORS["out_wb"], fontproperties=fp_bold)
ax.text(RIGHT_CX - 2.3, 5.55, "隐藏层\nbackward()", ha="right", va="center", fontsize=7.5,
        color=COLORS["hidden_wb"], fontproperties=fp_bold)

# ================================================================
#  Section dividers
# ================================================================
ax.plot([7.0, 7.0], [0.5, 7.9], color="#cbd5e1", lw=1, ls=":", zorder=0)

# Title at top
ax.text(7.0, 8.1, "前向算输出，反向算梯度——链式法则把误差从输出端传回输入端",
        ha="center", fontsize=12, color=COLORS["text"], fontproperties=fp_xl)

# ================================================================
#  Bottom: key insight box
# ================================================================
insight_y = 0.4
insight_w = 11
insight_h = 0.55
rect_insight = FancyBboxPatch((7.0 - insight_w/2, insight_y - insight_h/2),
                               insight_w, insight_h,
                               boxstyle="round,pad=0.15",
                               facecolor="#f8fafc", edgecolor="#cbd5e1",
                               alpha=0.8, lw=1.2, zorder=2)
ax.add_patch(rect_insight)
ax.text(7.0, insight_y,
        "链式法则的核心：高层（输出层）的梯度是低层（隐藏层）梯度的「上游」——下游的 w_out 越大，上游的 h 受到的梯度越大",
        ha="center", va="center", fontsize=8.5,
        color=COLORS["text"], fontproperties=fp_md, zorder=3)

fig.tight_layout(pad=0.5)
fig.savefig(OUT / "backprop.png", dpi=144, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)

print("  ✓ v3/backprop.png")

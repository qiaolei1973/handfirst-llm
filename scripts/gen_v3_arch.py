"""
Generate v3 architecture diagram: neuron, hidden layer, Linear, ReLU relationships.
Output: apps/_docs/public/v3/architecture.png
Run: python3 scripts/gen_v3_arch.py
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "apps/_docs/public/v3"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_sm  = fm.FontProperties(fname=FONT_PATH, size=8)
fp_md  = fm.FontProperties(fname=FONT_PATH, size=9)
fp_lg  = fm.FontProperties(fname=FONT_PATH, size=10)
fp_xl  = fm.FontProperties(fname=FONT_PATH, size=12)
fp_bold = fm.FontProperties(fname=FONT_PATH, size=9, weight="bold")
fp_title = fm.FontProperties(fname=FONT_PATH, size=11, weight="bold")

FIG_W, FIG_H = 14, 8

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# ---- Color palette ----
C_BG      = "#fafbfc"
C_LINEAR  = "#3b82f6"   # blue
C_RELU    = "#ef4444"   # red
C_NEURON  = "#8b5cf6"   # purple
C_HIDDEN  = "#f59e0b"   # amber
C_OUTPUT  = "#10b981"   # emerald
C_INPUT   = "#94a3b8"   # gray
C_OUTVAL  = "#06b6d4"   # cyan
C_BORDER  = "#475569"
C_ARROW   = "#64748b"
C_TEXT    = "#1e293b"
C_WHITE   = "#ffffff"

# ---- Helper functions ----
def draw_box(x, y, w, h, color, label, sublabel="", fontsize=9, lw=2, fill_alpha=0.15):
    """Draw a rounded box with label."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=color,
                          alpha=fill_alpha, lw=lw, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=fontsize,
            color=color, fontproperties=fp_bold, zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - h*0.3, sublabel,
                ha="center", va="center", fontsize=fontsize-2,
                color=color, alpha=0.7, fontproperties=fp_sm, zorder=3)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)

def draw_braced_group(x1, x2, y, label, color):
    """Draw a horizontal brace annotation under a group of items."""
    mid_x = (x1 + x2) / 2
    ax.annotate(label, xy=(mid_x, y), xytext=(mid_x, y - 0.7),
                ha="center", va="top", fontsize=8, color=color,
                fontproperties=fp_bold,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                zorder=4)

def section_title(x, y, text):
    ax.text(x, y, text, fontsize=11, color=C_TEXT,
            fontproperties=fp_xl, ha="center", va="center", zorder=5)

# ================================================================
#  SECTION A (left, top): What is a Neuron?
# ================================================================

# Title
ax.text(3.0, 7.5, "一个神经元", fontsize=12, color=C_NEURON,
        fontproperties=fp_title, ha="center", va="center", weight="bold")

# Input node
ax.plot(0.6, 6.3, "o", color=C_INPUT, markersize=18, zorder=3)
ax.text(0.6, 6.3, "x", ha="center", va="center", fontsize=10, color="white",
        fontproperties=fp_bold, zorder=4)

# Arrow x -> Linear box
draw_arrow(0.9, 6.3, 1.9, 6.3)

# Linear box
draw_box(1.9, 5.8, 2.2, 1.0, C_LINEAR, "Linear", "w·x + b", lw=2.5)

# Arrow Linear -> ReLU
draw_arrow(4.1, 6.3, 5.1, 6.3)

# ReLU box
draw_box(5.1, 5.8, 1.6, 1.0, C_RELU, "ReLU", "max(0, z)", lw=2.5)

# Arrow ReLU -> output
draw_arrow(6.7, 6.3, 7.8, 6.3)

# Output node
ax.plot(8.0, 6.3, "o", color=C_OUTVAL, markersize=18, zorder=3)
ax.text(8.0, 6.3, "h", ha="center", va="center", fontsize=10, color="white",
        fontproperties=fp_bold, zorder=4)

# Bracket: "一个神经元 = Linear + ReLU"
x_left = 1.6
x_right = 7.0
# draw a bracket line
ax.plot([x_left, x_left, x_right, x_right], [5.45, 5.25, 5.25, 5.45],
        color=C_NEURON, lw=2, zorder=3)
ax.text((x_left + x_right)/2, 5.05, "一个神经元",
        ha="center", va="top", fontsize=10, color=C_NEURON,
        fontproperties=fp_bold)

# ================================================================
#  SECTION B (left, bottom): The Hidden Layer = N neurons
# ================================================================

ax.text(3.0, 4.3, "隐藏层 = N 个神经元", fontsize=12, color=C_HIDDEN,
        fontproperties=fp_title, ha="center", va="center")

# N stacked neuron strips (compact representation)
N_STRIPS = 5
strip_y_base = 3.9
strip_h = 0.35
strip_gap = 0.08

for i in range(N_STRIPS):
    y = strip_y_base - i * (strip_h + strip_gap)
    # Linear half
    rect1 = FancyBboxPatch((1.9, y+0.02), 2.2, strip_h-0.04,
                           boxstyle="round,pad=0.05",
                           facecolor=C_LINEAR, edgecolor="none",
                           alpha=0.2 - i*0.02, zorder=2)
    ax.add_patch(rect1)
    # ReLU half
    rect2 = FancyBboxPatch((4.1, y+0.02), 1.6, strip_h-0.04,
                           boxstyle="round,pad=0.05",
                           facecolor=C_RELU, edgecolor="none",
                           alpha=0.2 - i*0.02, zorder=2)
    ax.add_patch(rect2)
    # Neuron label on left of each strip
    ax.text(1.2, y + strip_h/2, f"神经元 {i+1}",
            ha="right", va="center", fontsize=7, color=C_TEXT,
            fontproperties=fp_sm, alpha=0.7)

# Border around all strips
border_y = strip_y_base - (N_STRIPS-1) * (strip_h + strip_gap) - 0.15
border_h = strip_y_base + 0.3 - border_y
rect_border = FancyBboxPatch((1.6, border_y), 5.4, border_h,
                              boxstyle="round,pad=0.15",
                              facecolor="none", edgecolor=C_HIDDEN,
                              lw=2.5, zorder=1)
ax.add_patch(rect_border)

# Label: "N 个 (Linear + ReLU)"
ax.text(1.9 + 5.4/2, border_y - 0.15, "N 个 Neuron = N 个 (Linear + ReLU)",
        ha="center", va="top", fontsize=9, color=C_HIDDEN,
        fontproperties=fp_bold)

# ================================================================
#  SECTION C (right): Full Architecture
# ================================================================

cx = 10.5  # center x of right section

ax.text(cx, 7.5, "完整结构：1 → N → 1 网络", fontsize=12, color=C_TEXT,
        fontproperties=fp_title, ha="center", va="center")

# ---- Input layer ----
ax.plot(cx, 6.9, "o", color=C_INPUT, markersize=22, zorder=3)
ax.text(cx, 6.9, "x", ha="center", va="center", fontsize=11, color="white",
        fontproperties=fp_bold, zorder=4)
ax.text(cx, 6.4, "输入\n(1维)", ha="center", va="top", fontsize=8,
        color=C_INPUT, fontproperties=fp_sm)

# Arrow input -> hidden
draw_arrow(cx, 6.6, cx, 5.7)

# ---- Hidden Layer (big box) ----
hx, hy, hw, hh = cx - 1.6, 3.9, 3.2, 1.8
# Outer box
rect_hidden = FancyBboxPatch((hx, hy), hw, hh,
                              boxstyle="round,pad=0.2",
                              facecolor=C_HIDDEN, edgecolor=C_HIDDEN,
                              alpha=0.1, lw=3, zorder=2)
ax.add_patch(rect_hidden)

# Inner neurons (vertical strips)
NN = 5
for i in range(NN):
    ny = hy + 0.3 + i * 0.28
    # Linear bar
    rect_l = FancyBboxPatch((hx + 0.3, ny), 1.1, 0.2,
                            boxstyle="round,pad=0.03",
                            facecolor=C_LINEAR, edgecolor=C_LINEAR,
                            alpha=0.4, lw=0.8, zorder=3)
    ax.add_patch(rect_l)
    # ReLU bar
    rect_r = FancyBboxPatch((hx + 1.5, ny), 0.9, 0.2,
                            boxstyle="round,pad=0.03",
                            facecolor=C_RELU, edgecolor=C_RELU,
                            alpha=0.4, lw=0.8, zorder=3)
    ax.add_patch(rect_r)

ax.text(cx, hy + hh/2, "隐藏层", ha="center", va="center", fontsize=10,
        color=C_HIDDEN, fontproperties=fp_bold, alpha=0.5, zorder=4)
ax.text(cx, hy - 0.08, "N 个 ReLU 神经元", ha="center", va="top",
        fontsize=8, color=C_HIDDEN, fontproperties=fp_sm)

# Arrow hidden -> output
draw_arrow(cx, 3.9, cx, 3.1)

# ---- Output Layer ----
draw_box(cx - 1.1, 2.5, 2.2, 0.6, C_OUTPUT, "Linear", "sum(h) + bias", lw=2.5)

# Arrow output -> prediction
draw_arrow(cx, 2.5, cx, 1.7)

# Prediction node
ax.plot(cx, 1.35, "o", color=C_OUTVAL, markersize=22, zorder=3)
ax.text(cx, 1.35, "f(x)", ha="center", va="center", fontsize=8, color="white",
        fontproperties=fp_bold, zorder=4)
ax.text(cx, 0.75, "预测\n(1维)", ha="center", va="top", fontsize=8,
        color=C_OUTVAL, fontproperties=fp_sm)

# ================================================================
#  Legend / Key at bottom
# ================================================================

legend_y = 0.2
items = [
    (1.5,  C_LINEAR, "Linear = Wx + b（线性变换）"),
    (5.0,  C_RELU,   "ReLU = max(0, z)（激活函数）"),
    (8.5,  C_NEURON, "神经元 = Linear + ReLU"),
    (11.5, C_HIDDEN, "隐藏层 = N 个神经元"),
]
for x, color, label in items:
    rect = FancyBboxPatch((x, legend_y), 2.8, 0.35,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=color,
                          alpha=0.12, lw=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + 1.4, legend_y + 0.175, label,
            ha="center", va="center", fontsize=8, color=color,
            fontproperties=fp_bold, zorder=3)

# ================================================================
#  Section dividers (dotted lines)
# ================================================================
ax.plot([7.5, 7.5], [7.9, 0.05], color="#cbd5e1", lw=1, ls=":", zorder=0)
ax.plot([0.05, 13.95], [4.9, 4.9], color="#cbd5e1", lw=1, ls=":", zorder=0)

# Section labels
ax.text(4.0, 7.85, "概念拆解", ha="center", fontsize=8, color="#94a3b8",
        fontproperties=fp_sm, alpha=0.6)
ax.text(4.0, 4.75, "组织方式", ha="center", fontsize=8, color="#94a3b8",
        fontproperties=fp_sm, alpha=0.6)
ax.text(10.5, 7.85, "完整结构", ha="center", fontsize=8, color="#94a3b8",
        fontproperties=fp_sm, alpha=0.6)

fig.tight_layout(pad=0.5)
fig.savefig(OUT / "architecture.png", dpi=144, bbox_inches="tight",
            facecolor=C_WHITE, edgecolor="none")
plt.close(fig)

print("  ✓ v3/architecture.png")

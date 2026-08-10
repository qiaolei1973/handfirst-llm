"""
Generate recap chapter images.
Output: apps/_docs/public/recap/
Run: python3 scripts/gen_recap_images.py

Images:
  1. training-loop.png — 训练循环全景：预测→损失→梯度→更新，四步循环
  2. concept-map.png  — 概念地图：机器学习所有概念的层级关系
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "apps/_docs/public/recap"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/home/qingquan/.fonts/NotoSansSC-Regular.ttf"
fp_xs   = fm.FontProperties(fname=FONT_PATH, size=7.5)
fp_sm   = fm.FontProperties(fname=FONT_PATH, size=8.5)
fp_md   = fm.FontProperties(fname=FONT_PATH, size=9.5)
fp_lg   = fm.FontProperties(fname=FONT_PATH, size=11)
fp_xl   = fm.FontProperties(fname=FONT_PATH, size=13)
fp_xxl  = fm.FontProperties(fname=FONT_PATH, size=16, weight="bold")
fp_bold = fm.FontProperties(fname=FONT_PATH, size=9, weight="bold")
fp_title = fm.FontProperties(fname=FONT_PATH, size=12, weight="bold")

COLORS = {
    "red":       "#ef4444",
    "blue":      "#3b82f6",
    "green":     "#22c55e",
    "amber":     "#f59e0b",
    "purple":    "#8b5cf6",
    "gray":      "#94a3b8",
    "gray_dark": "#64748b",
    "indigo":    "#6366f1",
    "cyan":      "#06b6d4",
    "pink":      "#ec4899",
    "bg":        "#fafbfc",
    "text":      "#1e293b",
}

# ================================================================
#  1. training-loop.png — 训练循环全景
# ================================================================
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.set_aspect("equal")
ax.axis("off")

# Four nodes in a circle
cx, cy = 6.5, 3.75
radius = 2.8

nodes = [
    {
        "angle": 90,
        "title": "模型预测",
        "subtitle": "Forward",
        "color": COLORS["blue"],
        "items": ["Linear (Wx+b)", "激活函数 (ReLU)", "神经元 = Linear+ReLU", "隐藏层 = N个神经元", "Layer.forward()"],
    },
    {
        "angle": 0,
        "title": "算损失",
        "subtitle": "Loss",
        "color": COLORS["red"],
        "items": ["MSE = sum(yPred-y)²", "训练损失", "验证损失"],
    },
    {
        "angle": -90,
        "title": "算梯度",
        "subtitle": "Gradient",
        "color": COLORS["amber"],
        "items": ["梯度 = dL/d参数", "链式法则", "反向传播", "Layer.backward()", "收→算→传"],
    },
    {
        "angle": 180,
        "title": "更新参数",
        "subtitle": "Optimize",
        "color": COLORS["green"],
        "items": ["梯度下降: W-=lr*grad", "Momentum (惯性)", "RMSProp (自适应lr)", "Adam (默认首选)"],
    },
]

box_w, box_h = 3.8, 2.8

for node in nodes:
    angle_rad = np.radians(node["angle"])
    nx = cx + radius * np.cos(angle_rad)
    ny = cy + radius * np.sin(angle_rad)

    # Main box
    rect = FancyBboxPatch((nx - box_w/2, ny - box_h/2), box_w, box_h,
                          boxstyle="round,pad=0.15",
                          facecolor=node["color"], edgecolor=node["color"],
                          alpha=0.10, lw=2.2, zorder=3)
    ax.add_patch(rect)

    # Title
    ax.text(nx, ny + box_h/2 - 0.4, node["title"], ha="center", va="top",
            fontsize=11.5, color=node["color"], fontproperties=fp_title, zorder=4)
    # Subtitle
    ax.text(nx, ny + box_h/2 - 0.85, node["subtitle"], ha="center", va="top",
            fontsize=8, color=node["color"], fontproperties=fp_sm, alpha=0.7, zorder=4)

    # Items
    for j, item in enumerate(node["items"]):
        ax.text(nx, ny + box_h/2 - 1.25 - j * 0.33, item, ha="center", va="top",
                fontsize=7.3, color=COLORS["text"], fontproperties=fp_xs, zorder=4)

# Arrows between nodes (clockwise)
arrow_color = COLORS["gray_dark"]
for i in range(4):
    a1 = np.radians(nodes[i]["angle"])
    a2 = np.radians(nodes[(i + 1) % 4]["angle"])
    # Start from edge of box in direction of next node
    dx_start = (box_w/2 + 0.15) * np.cos(a1 - np.radians(25))
    dy_start = (box_h/2 + 0.15) * np.sin(a1 - np.radians(25))
    # Adjust: compute tangent point on circle
    mid_angle = (nodes[i]["angle"] + nodes[(i+1)%4]["angle"]) / 2
    if nodes[i]["angle"] - nodes[(i+1)%4]["angle"] > 180:
        mid_angle += 180
    if nodes[(i+1)%4]["angle"] - nodes[i]["angle"] > 180:
        mid_angle += 180

    # Simple arrow from near one box edge toward the next - use arc
    s_ang = np.radians(nodes[i]["angle"] - 40)
    e_ang = np.radians(nodes[(i+1)%4]["angle"] + 40)
    # Adjust for the 90→0→-90→180 wrap
    sx = cx + (radius - 0.3) * np.cos(s_ang)
    sy = cy + (radius - 0.3) * np.sin(s_ang)
    ex = cx + (radius - 0.3) * np.cos(e_ang)
    ey = cy + (radius - 0.3) * np.sin(e_ang)

    # Use a curved arrow
    rad = radius - 0.3
    # Arc path
    theta1 = nodes[i]["angle"] - 40
    theta2 = nodes[(i+1)%4]["angle"] + 40
    # Handle wrap-around
    if theta2 < theta1:
        theta2 += 360

    arc_theta = np.linspace(np.radians(theta1), np.radians(theta2), 60)
    arc_x = cx + rad * np.cos(arc_theta)
    arc_y = cy + rad * np.sin(arc_theta)

    ax.plot(arc_x, arc_y, color=arrow_color, lw=2.2, alpha=0.5, zorder=2)

    # Arrowhead at end
    end_idx = -1
    dx_head = arc_x[end_idx] - arc_x[end_idx-3]
    dy_head = arc_y[end_idx] - arc_y[end_idx-3]
    ax.annotate("", xy=(arc_x[end_idx], arc_y[end_idx]),
                xytext=(arc_x[end_idx-3], arc_y[end_idx-3]),
                arrowprops=dict(arrowstyle="->", color=arrow_color,
                               lw=2, alpha=0.6), zorder=5)

# Center label
ax.text(cx, cy, "训练\n循环", ha="center", va="center", fontsize=16,
        color=COLORS["text"], fontproperties=fp_xxl, zorder=5)
ax.text(cx, cy - 0.5, "epoch = 一圈", ha="center", va="center", fontsize=8.5,
        color=COLORS["gray_dark"], fontproperties=fp_sm, zorder=5)

# Surrounding annotations
annotations = [
    (cx + 5.3, cy + 1.0, "标准化\n(z-score)", COLORS["purple"]),
    (cx + 4.8, cy - 2.2, "Early\nStopping", COLORS["green"]),
    (cx - 5.3, cy - 1.0, "Mini-batch\nSGD", COLORS["amber"]),
    (cx - 4.8, cy + 2.2, "中心化", COLORS["blue"]),
]
for ax_x, ax_y, label, color in annotations:
    ax.text(ax_x, ax_y, label, ha="center", va="center", fontsize=8,
            color=color, fontproperties=fp_bold,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.85, lw=1.5), zorder=4)

# Title
ax.text(cx, 7.2, "机器学习 = 一个循环，转起来", ha="center", fontsize=14,
        color=COLORS["text"], fontproperties=fp_xxl)

fig.tight_layout(pad=0.5)
fig.savefig(OUT / "training-loop.png", dpi=144, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  ✓ recap/training-loop.png")


# ================================================================
#  2. concept-map.png — 概念层级地图
# ================================================================
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")

# ---- Tree layout ----
# Root
root_x, root_y = 7.0, 8.2

# Level 1: five main branches
branches = [
    {"label": "模型\nModel", "x": 1.4, "y": 6.2, "color": COLORS["blue"]},
    {"label": "损失\nLoss",   "x": 4.2, "y": 6.2, "color": COLORS["red"]},
    {"label": "梯度\nGradient","x": 7.0, "y": 6.2, "color": COLORS["amber"]},
    {"label": "优化器\nOptimizer","x": 9.8, "y": 6.2, "color": COLORS["green"]},
    {"label": "防线\nGuardrails","x": 12.6, "y": 6.2, "color": COLORS["purple"]},
]

# Level 2+3 data
children = {
    "模型": [
        ("Linear\nWx+b", COLORS["blue"], []),
        ("激活函数\nReLU=max(0,z)", COLORS["blue"], []),
        ("神经元\nLinear+ReLU", COLORS["blue"], [
            ("隐藏层\nN个神经元并行", "#2563eb"),
            ("输出层\nLinear 不加激活", "#2563eb"),
        ]),
        ("Layer\nforward+backward", COLORS["blue"], []),
    ],
    "损失": [
        ("MAE\nsum|diff|", COLORS["red"], []),
        ("MSE\nsum diff² 默认", COLORS["red"], []),
        ("训练损失\n模型看得到的", COLORS["red"], []),
        ("验证损失\n模型看不到的", COLORS["red"], []),
    ],
    "梯度": [
        ("dL/d参数\n损失对参数的斜率", COLORS["amber"], []),
        ("链式法则\ndL/dw = dL/dy * dy/dh * dh/dw", COLORS["amber"], []),
        ("反向传播\n链式法则 按层组织", COLORS["amber"], []),
        ("收→算→传\nbackward()三步", COLORS["amber"], []),
    ],
    "优化器": [
        ("梯度下降\nW-=lr*grad", COLORS["green"], []),
        ("SGD\n随机采样+梯度下降", COLORS["green"], []),
        ("Momentum\n指数移动平均 惯性", COLORS["green"], []),
        ("RMSProp\n每个参数自己的lr", COLORS["green"], []),
        ("Adam\nMomentum+RMSProp+bias修正", COLORS["green"], []),
    ],
    "防线": [
        ("中心化\nx-mean(x)", COLORS["purple"], []),
        ("标准化\n(x-mean)/std", COLORS["purple"], []),
        ("训练/验证分离\n留一部分不训练", COLORS["purple"], []),
        ("过拟合\ntrain↓ val↑", COLORS["purple"], []),
        ("Early Stopping\nval不降就停", COLORS["purple"], []),
    ],
}


def draw_node(x, y, label, color, fontsize=7.8, bold=False):
    fp = fp_bold if bold else fp_sm
    lines = label.split("\n")
    for i, line in enumerate(lines):
        ax.text(x, y - i * 0.32, line, ha="center", va="center",
                fontsize=fontsize, color=color, fontproperties=fp, zorder=6)

def draw_branch(x1, y1, x2, y2, color, lw=1.2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.5, zorder=1)
    # small dot at child
    ax.plot(x2, y2, "o", color=color, markersize=2.5, alpha=0.5, zorder=1)


# Root
ax.text(root_x, root_y, "机器学习", ha="center", va="center",
        fontsize=15, color=COLORS["text"], fontproperties=fp_xxl, zorder=5)
rect_root = FancyBboxPatch((root_x - 1.4, root_y - 0.4), 2.8, 0.8,
                            boxstyle="round,pad=0.15",
                            facecolor="#f1f5f9", edgecolor=COLORS["gray"],
                            alpha=0.6, lw=2, zorder=2)
ax.add_patch(rect_root)

# Draw branches from root
for b in branches:
    draw_branch(root_x, root_y - 0.6, b["x"], b["y"] + 0.5, COLORS["gray"], 1.5)
    # Branch label box
    rect = FancyBboxPatch((b["x"] - 1.1, b["y"] - 0.35), 2.2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=b["color"], edgecolor=b["color"],
                           alpha=0.12, lw=2, zorder=3)
    ax.add_patch(rect)
    ax.text(b["x"], b["y"] + 0.05, b["label"], ha="center", va="center",
            fontsize=9, color=b["color"], fontproperties=fp_bold, zorder=4)

# Draw Level 2 (children)
for branch_key, branch_data in children.items():
    parent = [b for b in branches if b["label"].startswith(branch_key)][0]
    n = len(branch_data)
    start_x = parent["x"] - 0.4 * (n - 1)
    for i, (label, color, grandchildren) in enumerate(branch_data):
        child_x = start_x + i * 0.85
        child_y = parent["y"] - 1.3
        lines = label.split("\n")

        # Draw connecting line
        draw_branch(parent["x"], parent["y"] - 0.5, child_x, child_y + 0.25, color)

        # Node background for items with multiple lines
        if len(lines) > 1:
            rect_h = 0.22 + 0.3 * (len(lines) - 1)
            rect = FancyBboxPatch((child_x - 1.25, child_y - rect_h/2), 2.5, rect_h,
                                   boxstyle="round,pad=0.08",
                                   facecolor="white", edgecolor=color,
                                   alpha=0.8, lw=1.2, zorder=3)
            ax.add_patch(rect)

        draw_node(child_x, child_y + 0.08, label, color, fontsize=7.2)

        # Level 3 (grandchildren)
        if grandchildren:
            for j, (g_label, g_color) in enumerate(grandchildren):
                gx = child_x - 0.42 + j * 0.9
                gy = child_y - 0.75
                glines = g_label.split("\n")
                draw_branch(child_x, child_y - 0.15, gx, gy + 0.15, g_color, 0.8)

                grect_h = 0.18 + 0.28 * (len(glines) - 1)
                grect = FancyBboxPatch((gx - 0.95, gy - grect_h/2), 1.9, grect_h,
                                        boxstyle="round,pad=0.06",
                                        facecolor="white", edgecolor=g_color,
                                        alpha=0.7, lw=1, ls="--", zorder=3)
                ax.add_patch(grect)
                draw_node(gx, gy + 0.05, g_label, g_color, fontsize=6.5)

# Sup title
ax.text(7.0, 8.85, "当我们在谈机器学习时，我们在谈些什么", ha="center",
        fontsize=13, color=COLORS["text"], fontproperties=fp_xl)

fig.tight_layout(pad=0.3)
fig.savefig(OUT / "concept-map.png", dpi=144, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  ✓ recap/concept-map.png")

print(f"\n✓ All recap images generated:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

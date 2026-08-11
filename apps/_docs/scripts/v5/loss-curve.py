"""
#4 loss-curve — 训练 loss 和验证 loss 的下降曲线
"""
import matplotlib
matplotlib.use("Agg")
import _cjk_font
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

fig, ax = plt.subplots(figsize=(10, 5.5))

# 模拟示例数据以展示曲线形状
epochs = np.arange(1, 501)
train_loss = 0.3 * np.exp(-epochs / 80) + 0.05 * np.exp(-epochs / 300) + 0.02 * np.random.randn(len(epochs)) + 0.02
val_loss = 0.32 * np.exp(-epochs / 70) + 0.06 * np.exp(-epochs / 250) + 0.025 * np.random.randn(len(epochs)) + 0.03

ax.plot(epochs, train_loss, color="#3b82f6", linewidth=1.5, alpha=0.8, label="训练 loss")
ax.plot(epochs, val_loss, color="#ef4444", linewidth=1.5, alpha=0.8, label="验证 loss")
ax.set_xlabel("Epoch", fontsize=12, fontproperties=_cjk_font.cjk_fp)
ax.set_ylabel("MSE Loss", fontsize=12, fontproperties=_cjk_font.cjk_fp)
ax.set_title("训练/验证 Loss 下降曲线", fontsize=14, fontproperties=_cjk_font.cjk_fp)
ax.legend(fontsize=12, prop=_cjk_font.cjk_fp)
ax.grid(True, alpha=0.3)

out = Path(__file__).resolve().parent.parent.parent / "public" / "v5" / "loss-curve.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  → {out.relative_to(out.parent.parent.parent)}")
plt.close()

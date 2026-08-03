"""Generate images for v2 tutorial with Noto Sans SC CJK support."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / 'apps' / 'docs' / 'public' / 'v2'
OUT.mkdir(parents=True, exist_ok=True)

# ---- Register CJK font ----
FONT_PATH = '/home/qingquan/.fonts/NotoSansSC-Regular.ttf'
fm.fontManager.addfont(FONT_PATH)
font_name = fm.FontProperties(fname=FONT_PATH).get_name()
print(f'Using font: {font_name}')

# ---- Color palette ----
BLUE = '#2563eb'
RED = '#dc2626'
GREEN = '#16a34a'
GRAY = '#64748b'
DARK = '#1e293b'

plt.rcParams.update({
    'font.family': font_name,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.edgecolor': DARK,
    'axes.labelcolor': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
    'text.color': DARK,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.alpha': 0.25,
})


def loss_and_gradient():
    """MAE vs MSE — loss curves and their gradients."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    diff = np.linspace(-8, 8, 300)

    # --- Left: Loss functions ---
    mae_loss = np.abs(diff)
    mse_loss = diff ** 2

    ax1.plot(diff, mae_loss, color=RED, linewidth=2.2, label='MAE = |d|')
    ax1.plot(diff, mse_loss, color=BLUE, linewidth=2.2, label='MSE = d²')

    for x_val in [6]:
        ax1.annotate(f'MSE = {x_val**2}\nMAE = {x_val}',
                     xy=(x_val, x_val**2), xytext=(x_val + 1.2, x_val**2 - 4),
                     fontsize=10, color=BLUE,
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
        ax1.plot(x_val, x_val**2, 'o', color=BLUE, markersize=6)
        ax1.plot(x_val, x_val, 'o', color=RED, markersize=6)

    ax1.set_xlabel('diff = y_pred − y')
    ax1.set_title('损失函数 Loss Function')
    ax1.legend(loc='upper center', fontsize=10)
    ax1.grid(True)
    ax1.axhline(0, color=GRAY, linewidth=0.5)
    ax1.axvline(0, color=GRAY, linewidth=0.5)
    ax1.text(6.5, 28, '同样差 6，\nMAE = 6，MSE = 36\nMSE 惩罚更重',
             fontsize=11, color=BLUE, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=GRAY, alpha=0.7))

    # --- Right: Gradients ---
    mae_grad = np.sign(diff)
    mae_grad[diff == 0] = 0
    mse_grad = 2 * diff

    ax2.plot(diff, mae_grad, color=RED, linewidth=2.2,
             label='MAE: gradient = sign(d)')
    ax2.plot(diff, mse_grad, color=BLUE, linewidth=2.2,
             label='MSE: gradient = 2d')

    ax2.axhline(1, color=RED, linewidth=1, linestyle='--', alpha=0.35)
    ax2.axhline(-1, color=RED, linewidth=1, linestyle='--', alpha=0.35)

    ax2.annotate('无论差多远\n推动力都是 ±1',
                 xy=(7, 1), xytext=(3, 3.5),
                 fontsize=10, color=RED,
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    ax2.annotate('差得远 → 推得猛\n差得近 → 自然减速',
                 xy=(6, 12), xytext=(1.5, 14.5),
                 fontsize=10, color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

    ax2.set_xlabel('diff = y_pred − y')
    ax2.set_title('梯度（导数）Gradient')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True)
    ax2.axhline(0, color=GRAY, linewidth=0.5)
    ax2.axvline(0, color=GRAY, linewidth=0.5)
    ax2.set_ylim(-16, 16)

    fig.tight_layout()
    fig.savefig(OUT / 'loss-comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Done: loss-comparison.png')


def centering():
    """Before/after mean centering visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))

    np.random.seed(3)
    x_raw = np.linspace(0, 20, 12) + np.random.uniform(-0.3, 0.3, 12)
    mean_x = x_raw.mean()
    x_centered = x_raw - mean_x

    y_jitter = np.zeros_like(x_raw)

    # --- Before ---
    ax1.scatter(x_raw, y_jitter, c=RED, s=80, zorder=3,
                edgecolors='white', linewidth=0.5)
    ax1.axvline(0, color=GRAY, linewidth=1)
    ax1.axvline(mean_x, color=BLUE, linewidth=1.5, linestyle='--',
                label=f'均值 mean = {mean_x:.1f}')

    for x in x_raw:
        ax1.annotate('', xy=(x + 1.5, 0), xytext=(x, 0),
                     arrowprops=dict(arrowstyle='->', color=RED,
                                     alpha=0.5, lw=1.5))

    ax1.annotate('所有 x ≥ 0\n梯度推力全同号',
                 xy=(10, 0.27), fontsize=13, color=RED, ha='center',
                 fontweight='bold')

    ax1.set_xlim(-12, 22)
    ax1.set_ylim(-0.38, 0.50)
    ax1.set_yticks([])
    ax1.set_xlabel('x')
    ax1.set_title('中心化前')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='x')

    # --- After ---
    ax2.scatter(x_centered, y_jitter, c=GREEN, s=80, zorder=3,
                edgecolors='white', linewidth=0.5)
    ax2.axvline(0, color=BLUE, linewidth=2, linestyle='-',
                label='new center = 0', alpha=0.6)

    for x in x_centered:
        direction = 1.5 if x < 0 else -1.5
        ax2.annotate('', xy=(x + direction, 0), xytext=(x, 0),
                     arrowprops=dict(arrowstyle='->', color=GREEN,
                                     alpha=0.5, lw=1.5))

    ax2.annotate('对称分布在 0 两侧\n正负推力互相平衡',
                 xy=(0, 0.27), fontsize=13, color=GREEN, ha='center',
                 fontweight='bold')

    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-0.38, 0.50)
    ax2.set_yticks([])
    ax2.set_xlabel('x − mean')
    ax2.set_title('中心化后')
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='x')

    fig.tight_layout()
    fig.savefig(OUT / 'centering.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Done: centering.png')


def sgd_fluctuation():
    """SGD noisy loss vs full-batch smooth."""
    fig, ax = plt.subplots(figsize=(10, 4.2))

    np.random.seed(42)
    epochs = np.arange(200)

    full_loss = 30 * np.exp(-epochs / 60) + 2 * np.exp(-epochs / 15) + 0.3
    full_loss += np.random.normal(0, 0.02, len(epochs))

    sgd_loss = full_loss + np.random.normal(0, 0.8, len(epochs))
    spike_idx = np.random.choice(epochs, 8, replace=False)
    sgd_loss[spike_idx] += np.random.uniform(1, 3, 8)

    ax.plot(epochs, full_loss, color=BLUE, linewidth=2, alpha=0.7,
            label='全量梯度 Full-batch（12/12）')
    ax.plot(epochs, sgd_loss, color=RED, linewidth=1, alpha=0.85,
            label='随机梯度 SGD（8/12）')

    ax.fill_between(epochs, sgd_loss - 0.5, sgd_loss + 0.5,
                     alpha=0.08, color=RED)

    ax.annotate('SGD：有波动，但总体下降',
                xy=(100, sgd_loss[100]), xytext=(130, sgd_loss[100] + 4),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    ax.annotate('Full-batch：平滑单调下降',
                xy=(80, full_loss[80]), xytext=(120, full_loss[80] + 2.5),
                fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('SGD vs 全量梯度：Loss 曲线对比')
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.set_xlim(0, 200)

    fig.tight_layout()
    fig.savefig(OUT / 'sgd-fluctuation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Done: sgd-fluctuation.png')


if __name__ == '__main__':
    loss_and_gradient()
    centering()
    sgd_fluctuation()
    print(f'\nAll saved → {OUT}')

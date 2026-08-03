"""Generate conceptual images for v1 tutorial."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / 'apps' / 'docs' / 'public' / 'v1'
OUT.mkdir(parents=True, exist_ok=True)

# ---- Register CJK font ----
FONT_PATH = '/home/qingquan/.fonts/NotoSansSC-Regular.ttf'
fm.fontManager.addfont(FONT_PATH)
font_name = fm.FontProperties(fname=FONT_PATH).get_name()

BLUE = '#2563eb'
RED = '#dc2626'
GREEN = '#16a34a'
ORANGE = '#ea580c'
PURPLE = '#7c3aed'
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


def mae_gradient():
    """MAE loss function and its gradient (sign function)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    d = np.linspace(-6, 6, 400)

    # --- Left: MAE loss = |d| ---
    loss = np.abs(d)
    ax1.plot(d, loss, color=BLUE, linewidth=2.5)

    # Highlight key points
    for x_val, label in [(-5, '差 -5 → loss=5'), (5, '差 +5 → loss=5')]:
        ax1.plot(x_val, abs(x_val), 'o', color=BLUE, markersize=8)
        ax1.annotate(label,
                     xy=(x_val, abs(x_val)), xytext=(x_val + 1.5, abs(x_val) - 1.5),
                     fontsize=10, color=BLUE,
                     arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))

    ax1.set_xlabel('diff = y_pred − y')
    ax1.set_ylabel('Loss')
    ax1.set_title('MAE 损失函数 L = |d|')
    ax1.grid(True)
    ax1.axhline(0, color=GRAY, linewidth=0.5)
    ax1.axvline(0, color=GRAY, linewidth=0.5)

    # --- Right: gradient = sign(d) ---
    sign = np.sign(d)
    sign[d == 0] = 0
    ax2.step(d, sign, color=RED, linewidth=2.5, where='mid')

    ax2.axhline(1, color=RED, linewidth=0.8, linestyle='--', alpha=0.3)
    ax2.axhline(-1, color=RED, linewidth=0.8, linestyle='--', alpha=0.3)

    ax2.annotate('梯度 = +1\n(预测偏大 → 减小 W)',
                 xy=(3, 1), xytext=(3.5, 2.5),
                 fontsize=10, color=RED,
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax2.annotate('梯度 = −1\n(预测偏小 → 增大 W)',
                 xy=(-3, -1), xytext=(-3.5, -2.5),
                 fontsize=10, color=RED,
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    # Highlight: flat regions
    ax2.fill_between(d, 0.9, 1.1, alpha=0.06, color=RED)
    ax2.fill_between(d, -1.1, -0.9, alpha=0.06, color=RED)

    ax2.set_xlabel('diff = y_pred − y')
    ax2.set_ylabel('梯度')
    ax2.set_title('MAE 梯度 = sign(d)\n（只看方向，不关心大小）')
    ax2.grid(True)
    ax2.axhline(0, color=GRAY, linewidth=0.5)
    ax2.axvline(0, color=GRAY, linewidth=0.5)
    ax2.set_ylim(-4, 4)

    fig.tight_layout()
    fig.savefig(OUT / 'mae-gradient.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Done: mae-gradient.png')


def gradient_asymmetry():
    """Why W converges faster than bias — gradient magnitude comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    x_vals = np.linspace(0, 20, 12)

    # --- Left: W gradient bar chart ---
    # gradient_W = sign * x, with x up to 20
    grad_w_positive = x_vals  # when sign=1, grad = x
    grad_w_negative = -x_vals  # when sign=-1, grad = -x

    bars1 = ax1.bar(np.arange(12), grad_w_positive, color=BLUE, alpha=0.85,
                    label='sign=+1 时（预测偏大）')
    bars2 = ax1.bar(np.arange(12), grad_w_negative, color=RED, alpha=0.85,
                    label='sign=−1 时（预测偏小）')

    ax1.axhline(0, color=GRAY, linewidth=1)
    ax1.axhline(1, color=BLUE, linewidth=0.8, linestyle=':', alpha=0.4)
    ax1.axhline(-1, color=RED, linewidth=0.8, linestyle=':', alpha=0.4)

    ax1.annotate('x 最大 20\n→ 推力可达 ±20',
                 xy=(11, 20), xytext=(7, 23),
                 fontsize=11, color=BLUE, ha='center', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))

    ax1.set_xlabel('数据点（按 x 从小到大）')
    ax1.set_ylabel('W 梯度 = sign(diff) × x')
    ax1.set_title('W 的梯度：被 x 放大')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_xticks([])
    ax1.grid(True, axis='y')
    ax1.set_ylim(-22, 28)

    # --- Right: bias gradient bar chart ---
    # gradient_bias = sign, always ±1
    grad_b_positive = np.ones(12)  # always 1
    grad_b_negative = -np.ones(12)  # always -1

    ax2.bar(np.arange(12), grad_b_positive, color=BLUE, alpha=0.85)
    ax2.bar(np.arange(12), grad_b_negative, color=RED, alpha=0.85)

    ax2.axhline(0, color=GRAY, linewidth=1)
    ax2.axhline(1, color=BLUE, linewidth=0.8, linestyle=':', alpha=0.4)
    ax2.axhline(-1, color=RED, linewidth=0.8, linestyle=':', alpha=0.4)

    ax2.annotate('始终 ±1\n无论 x 多大',
                 xy=(6, 1), xytext=(7.5, 2),
                 fontsize=11, color=BLUE, ha='center', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))

    ax2.set_xlabel('数据点')
    ax2.set_ylabel('bias 梯度 = sign(diff)')
    ax2.set_title('bias 的梯度：始终 ≤ 1')
    ax2.set_xticks([])
    ax2.grid(True, axis='y')
    ax2.set_ylim(-2.5, 3.5)

    fig.suptitle('为什么 W 收敛快、bias 收敛慢？', fontsize=15, fontweight='bold',
                 y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / 'gradient-asymmetry.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Done: gradient-asymmetry.png')


def loss_comparison():
    """W vs bias convergence on their respective 1D loss surfaces."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # True values: W=2, bias=10
    W_true = 2
    bias_true = 10

    # ---- Left: L(W) surface ----
    W_range = np.linspace(0.5, 4.5, 300)
    # Loss in W direction: for a fixed bias ~= 0, predict y for all x, compute MAE
    # This is a conceptual loss surface — not exact, just illustrative
    # L(W) ≈ mean(|(W - W_true) * x_mean|) + some offset
    x_mean = 10  # average x ~ 10
    L_W = np.abs(W_range - W_true) * x_mean + 3  # x_mean amplifies the loss

    ax1.plot(W_range, L_W, color=BLUE, linewidth=2.5)
    ax1.axvline(W_true, color=GREEN, linewidth=2, linestyle='--',
                label=f'真值 W = {W_true}')

    # Simulate gradient descent steps
    W_path = [1.0, 1.6, 1.85, 1.93, 1.97, 1.99, 2.0, 2.0]
    for i, (w_from, w_to) in enumerate(zip(W_path[:-1], W_path[1:])):
        loss_from = abs(w_from - W_true) * x_mean + 3
        loss_to = abs(w_to - W_true) * x_mean + 3
        ax1.annotate('', xy=(w_to, loss_to), xytext=(w_from, loss_from),
                     arrowprops=dict(arrowstyle='->', color=RED, lw=2,
                                     connectionstyle='arc3,rad=0.2'))
        if i == 0:
            ax1.plot(w_from, loss_from, 'o', color=RED, markersize=10,
                     zorder=5, label=f'起点 W={w_from}')
        elif i == 3:
            ax1.annotate(f'大步跳！\n({i+1} 步接近真值)',
                         xy=(w_from, loss_from),
                         xytext=(w_from + 0.4, loss_from + 15),
                         fontsize=10, color=RED,
                         arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    ax1.axvline(1, color=RED, linewidth=0.8, linestyle=':', alpha=0.4)
    ax1.set_xlabel('W')
    ax1.set_ylabel('Loss')
    ax1.set_title('L(W)：大步流星，几步到位')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True)

    # ---- Right: L(bias) surface ----
    bias_range = np.linspace(-5, 25, 300)
    # L(bias) ≈ mean(|bias - bias_true|) — no amplification
    L_bias = np.abs(bias_range - bias_true) + 3

    ax2.plot(bias_range, L_bias, color=BLUE, linewidth=2.5)
    ax2.axvline(bias_true, color=GREEN, linewidth=2, linestyle='--',
                label=f'真值 bias = {bias_true}')

    # Simulate bias gradient descent — much slower
    bias_path = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    for i, (b_from, b_to) in enumerate(zip(bias_path[:-1], bias_path[1:])):
        loss_from = abs(b_from - bias_true) + 3
        loss_to = abs(b_to - bias_true) + 3
        ax2.annotate('', xy=(b_to, loss_to), xytext=(b_from, loss_from),
                     arrowprops=dict(arrowstyle='->', color=RED, lw=1,
                                     connectionstyle='arc3,rad=0.1'))
        if i == 0:
            ax2.plot(b_from, loss_from, 'o', color=RED, markersize=10,
                     zorder=5, label=f'起点 bias={b_from}')
        elif i == 10:
            ax2.annotate(f'蜗牛爬行…\n({i+1} 步还差一半)',
                         xy=(b_from, loss_from),
                         xytext=(b_from + 3, loss_from + 4),
                         fontsize=10, color=RED,
                         arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    ax2.plot(bias_path[-1], abs(bias_path[-1] - bias_true) + 3, '*',
             color=RED, markersize=14, zorder=5, label=f'{len(bias_path)-1} 步到达')

    ax2.axvline(0, color=RED, linewidth=0.8, linestyle=':', alpha=0.4)
    ax2.set_xlabel('bias')
    ax2.set_ylabel('Loss')
    ax2.set_title('L(bias)：蜗牛爬行，步步艰难')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True)

    fig.suptitle('W vs bias 收敛对比（MAE 下相同学习率）', fontsize=15,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / 'loss-comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Done: loss-comparison.png')


if __name__ == '__main__':
    mae_gradient()
    gradient_asymmetry()
    loss_comparison()
    print(f'\nAll saved → {OUT}')

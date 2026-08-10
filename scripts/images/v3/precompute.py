"""预计算 v3 图片所需的训练结果，存入 precompute.npz"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "precompute.npz"
X_SCALE = 2 * np.pi
N_DATA = 60

np.random.seed(42)
x_norm = np.linspace(0, 1, N_DATA)
true_y = np.sin(x_norm * X_SCALE)
noise = (np.random.rand(N_DATA) - 0.5) * 0.3
labels = true_y + noise


def train_nn(x, y, n_neurons, epochs=4000, lr=0.02, batch_size=40, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x)
    hidden_w = rng.rand(n_neurons) * 1.2 - 0.6
    hidden_b = np.array([-hidden_w[j] * rng.rand() for j in range(n_neurons)])
    output_w = rng.rand(n_neurons) * 1.2 - 0.6
    output_b = 0.0
    indices = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_idx = indices[:batch_size]
        xb, yb = x[batch_idx], y[batch_idx]
        z = np.outer(xb, hidden_w) + hidden_b
        h = np.maximum(0, z)
        y_pred = h @ output_w + output_b
        diff = y_pred - yb
        grad_out = (2 * diff) / batch_size
        grad_output_w = h.T @ grad_out
        grad_output_b = np.sum(grad_out)
        grad_h = np.outer(grad_out, output_w)
        grad_preact = grad_h * (z > 0)
        grad_hidden_w = xb @ grad_preact
        grad_hidden_b = np.sum(grad_preact, axis=0)
        output_w -= lr * grad_output_w
        output_b -= lr * grad_output_b
        hidden_w -= lr * grad_hidden_w
        hidden_b -= lr * grad_hidden_b
    z = np.outer(x, hidden_w) + hidden_b
    h = np.maximum(0, z)
    y_pred = h @ output_w + output_b
    final_loss = np.mean((y_pred - y) ** 2)
    return hidden_w, hidden_b, output_w, output_b, final_loss


def run():
    results = {}
    for n in [2, 4, 8, 16]:
        hw, hb, ow, ob, loss = train_nn(x_norm, labels, n, epochs=4000, lr=0.02, seed=42)
        results[f"n{n}"] = dict(hw=hw, hb=hb, ow=ow, ob=ob, loss=loss)
    np.savez_compressed(OUT, **results)
    print(f"  saved {OUT}")

if __name__ == "__main__":
    run()

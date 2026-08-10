"""预计算 v4 图片所需的训练结果，存入 precompute.npz"""
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


def train_optimizer(x_raw, y, n_neurons=8, optimizer="sgd", epochs=2000,
                    lr=None, batch_size=40, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x_raw)
    n_train = int(n * 0.8)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]; val_idx = idx[n_train:]
    x_train_raw, y_train = x_raw[train_idx], y[train_idx]
    x_val_raw, y_val = x_raw[val_idx], y[val_idx]
    x_mean = np.mean(x_train_raw); x_std = np.std(x_train_raw)
    x_train = (x_train_raw - x_mean) / x_std
    x_val = (x_val_raw - x_mean) / x_std

    hw = rng.randn(n_neurons) * 0.5; hb = rng.randn(n_neurons) * 0.1
    ow = rng.randn(n_neurons) * 0.5; ob = 0.0

    if lr is None:
        lr = {"sgd": 0.02, "momentum": 0.02, "adam": 0.01}[optimizer]

    if optimizer == "momentum":
        beta = 0.9
        v_hw = np.zeros_like(hw); v_hb = np.zeros_like(hb)
        v_ow = np.zeros_like(ow); v_ob = 0.0
    if optimizer == "adam":
        beta1, beta2 = 0.9, 0.999; eps = 1e-8; t_step = 0
        m_hw = np.zeros_like(hw); v_hw = np.zeros_like(hw)
        m_hb = np.zeros_like(hb); v_hb = np.zeros_like(hb)
        m_ow = np.zeros_like(ow); v_ow = np.zeros_like(ow)
        m_ob = 0.0; v_ob = 0.0

    indices = np.arange(len(x_train))
    history = []

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_idx = indices[:batch_size]
        xb, yb = x_train[batch_idx], y_train[batch_idx]

        z = np.outer(xb, hw) + hb; h = np.maximum(0, z)
        y_pred = h @ ow + ob
        diff = y_pred - yb
        grad_out = (2 * diff) / batch_size
        g_ow = h.T @ grad_out; g_ob = np.sum(grad_out)
        g_h = np.outer(grad_out, ow)
        g_pre = g_h * (z > 0)
        g_hw = xb @ g_pre; g_hb = np.sum(g_pre, axis=0)

        if optimizer == "sgd":
            hw -= lr * g_hw; hb -= lr * g_hb; ow -= lr * g_ow; ob -= lr * g_ob
        elif optimizer == "momentum":
            v_hw = beta * v_hw + (1 - beta) * g_hw
            v_hb = beta * v_hb + (1 - beta) * g_hb
            v_ow = beta * v_ow + (1 - beta) * g_ow
            v_ob = beta * v_ob + (1 - beta) * g_ob
            hw -= lr * v_hw; hb -= lr * v_hb; ow -= lr * v_ow; ob -= lr * v_ob
        else:  # adam
            t_step += 1
            m_hw = beta1 * m_hw + (1 - beta1) * g_hw
            v_hw = beta2 * v_hw + (1 - beta2) * g_hw**2
            hw -= lr * (m_hw/(1-beta1**t_step)) / (np.sqrt(v_hw/(1-beta2**t_step)) + eps)
            m_hb = beta1 * m_hb + (1 - beta1) * g_hb
            v_hb = beta2 * v_hb + (1 - beta2) * g_hb**2
            hb -= lr * (m_hb/(1-beta1**t_step)) / (np.sqrt(v_hb/(1-beta2**t_step)) + eps)
            m_ow = beta1 * m_ow + (1 - beta1) * g_ow
            v_ow = beta2 * v_ow + (1 - beta2) * g_ow**2
            ow -= lr * (m_ow/(1-beta1**t_step)) / (np.sqrt(v_ow/(1-beta2**t_step)) + eps)
            m_ob = beta1 * m_ob + (1 - beta1) * g_ob
            v_ob = beta2 * v_ob + (1 - beta2) * g_ob**2
            ob -= lr * (m_ob/(1-beta1**t_step)) / (np.sqrt(v_ob/(1-beta2**t_step)) + eps)

        if epoch % 20 == 0:
            zt = np.outer(x_train, hw) + hb
            train_loss = np.mean((np.maximum(0, zt) @ ow + ob - y_train)**2)
            zv = np.outer(x_val, hw) + hb
            val_loss = np.mean((np.maximum(0, zv) @ ow + ob - y_val)**2)
            history.append((epoch, float(train_loss), float(val_loss)))

    return history


def run_overfit(x_raw, y, n_neurons=32, n_train=40, epochs=3000,
                lr=0.01, batch_size=20, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N_DATA)
    train_idx = idx[:n_train]; val_idx = idx[n_train:]
    x_train_raw, y_train = x_raw[train_idx], y[train_idx]
    x_val_raw, y_val = x_raw[val_idx], y[val_idx]
    x_mean = np.mean(x_train_raw); x_std = np.std(x_train_raw)
    x_train = (x_train_raw - x_mean) / x_std
    x_val = (x_val_raw - x_mean) / x_std

    hw = rng.randn(n_neurons) * 0.5; hb = rng.randn(n_neurons) * 0.1
    ow = rng.randn(n_neurons) * 0.5; ob = 0.0
    beta1, beta2 = 0.9, 0.999; eps = 1e-8; t_step = 0
    m_hw = np.zeros_like(hw); v_hw = np.zeros_like(hw)
    m_hb = np.zeros_like(hb); v_hb = np.zeros_like(hb)
    m_ow = np.zeros_like(ow); v_ow = np.zeros_like(ow)
    m_ob = 0.0; v_ob = 0.0

    indices = np.arange(n_train)
    history = []
    best_val = float("inf")
    best_params = None; best_epoch = 0

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_idx = indices[:batch_size]
        xb, yb = x_train[batch_idx], y_train[batch_idx]
        z = np.outer(xb, hw) + hb; h = np.maximum(0, z)
        diff = (h @ ow + ob) - yb
        g_out = (2 * diff) / batch_size
        g_ow = h.T @ g_out; g_ob = np.sum(g_out)
        g_h = np.outer(g_out, ow)
        g_pre = g_h * (z > 0)
        g_hw = xb @ g_pre; g_hb = np.sum(g_pre, axis=0)

        t_step += 1
        m_hw = beta1 * m_hw + (1 - beta1) * g_hw; v_hw = beta2 * v_hw + (1 - beta2) * g_hw**2
        hw -= lr * (m_hw/(1-beta1**t_step)) / (np.sqrt(v_hw/(1-beta2**t_step)) + eps)
        m_hb = beta1 * m_hb + (1 - beta1) * g_hb; v_hb = beta2 * v_hb + (1 - beta2) * g_hb**2
        hb -= lr * (m_hb/(1-beta1**t_step)) / (np.sqrt(v_hb/(1-beta2**t_step)) + eps)
        m_ow = beta1 * m_ow + (1 - beta1) * g_ow; v_ow = beta2 * v_ow + (1 - beta2) * g_ow**2
        ow -= lr * (m_ow/(1-beta1**t_step)) / (np.sqrt(v_ow/(1-beta2**t_step)) + eps)
        m_ob = beta1 * m_ob + (1 - beta1) * g_ob; v_ob = beta2 * v_ob + (1 - beta2) * g_ob**2
        ob -= lr * (m_ob/(1-beta1**t_step)) / (np.sqrt(v_ob/(1-beta2**t_step)) + eps)

        if epoch % 20 == 0:
            zt = np.outer(x_train, hw) + hb
            train_loss = float(np.mean((np.maximum(0, zt) @ ow + ob - y_train)**2))
            zv = np.outer(x_val, hw) + hb
            val_loss = float(np.mean((np.maximum(0, zv) @ ow + ob - y_val)**2))
            history.append((epoch, train_loss, val_loss))
            if val_loss < best_val:
                best_val = val_loss; best_epoch = epoch
                best_params = (hw.copy(), hb.copy(), ow.copy(), ob)

    return {
        "history": history,
        "best_epoch": best_epoch, "best_val": best_val,
        "best_hw": best_params[0], "best_hb": best_params[1],
        "best_ow": best_params[2], "best_ob": float(best_params[3]),
        "final_hw": hw, "final_hb": hb, "final_ow": ow, "final_ob": float(ob),
        "x_mean": float(x_mean), "x_std": float(x_std),
    }


def run():
    r = {}
    for opt in ["sgd", "momentum", "adam"]:
        hist = train_optimizer(x_norm, labels, n_neurons=8, optimizer=opt,
                               epochs=2000, seed=42)
        r[f"opt_{opt}"] = np.array(hist, dtype=np.float32)

    of = run_overfit(x_norm, labels, n_neurons=32, n_train=40, epochs=3000, seed=42)
    r["of_history"] = np.array(of["history"], dtype=np.float32)
    r["of_best_epoch"] = of["best_epoch"]
    r["of_best_val"] = of["best_val"]
    r["of_best_hw"] = of["best_hw"]; r["of_best_hb"] = of["best_hb"]
    r["of_best_ow"] = of["best_ow"]
    # Scalar stored separately from arrays
    np.savez_compressed(OUT,
        opt_sgd=r["opt_sgd"], opt_momentum=r["opt_momentum"],
        opt_adam=r["opt_adam"],
        of_history=r["of_history"],
        of_best_epoch=r["of_best_epoch"], of_best_val=of["best_val"],
        of_best_hw=of["best_hw"], of_best_hb=of["best_hb"],
        of_best_ow=of["best_ow"], of_best_ob=of["best_ob"],
        of_final_hw=of["final_hw"], of_final_hb=of["final_hb"],
        of_final_ow=of["final_ow"], of_final_ob=of["final_ob"],
        of_x_mean=of["x_mean"], of_x_std=of["x_std"],
    )
    print(f"  saved {OUT}")

if __name__ == "__main__":
    run()

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythonML.AutoEncoderClassNew import Conv1Dminimal


# =========================
# User config
# =========================
CSV_PATH = "./data/triangular_anisotropy_uniform_Js4_B60_Nk250_NJ30.csv"
CKPT_PATH = "./data/ConvChannel_anisotropy_LSWT_Js4_Nk250_sigma0.1.pth"

OUT_DIR = "./plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model config for this checkpoint family
LATENT_DIM = 4
HIDDEN_DIM = 64
HAS_VARIANCE = True
IF_DECODE = True

# Match your previous split style
TRAIN_RATIO = 0.9
SPLIT_SEED = 1234

# Evaluate on: "test", "train", or "all"
EVAL_SPLIT = "test"

# Optional: inject bandwidth noise before inference
APPLY_INPUT_NOISE = True
NOISE_SIGMA = 0.1
NOISE_SEED = 1234

# Plot style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["data", "disp", "Js"]:
        df[col] = df[col].apply(json.loads)

    disp = np.array(df["disp"].tolist(), dtype=np.float32)   # (N, L)
    inten = np.array(df["data"].tolist(), dtype=np.float32)  # (N, L)
    X_raw = np.stack([disp, inten], axis=1)                  # (N, 2, L)

    y = np.array(df["Js"].tolist(), dtype=np.float32)        # (N, 4)
    y[:, 1] *= y[:, 0]
    return X_raw, y


def add_bandwidth_noise(X, sigma, seed=1234, intensity_channel=1):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)

    max_vals = X.max(axis=2, keepdims=True)
    min_vals = X.min(axis=2, keepdims=True)

    bandwidth = max_vals - min_vals
    bandwidth[:, intensity_channel:intensity_channel + 1, :] = max_vals[:, intensity_channel:intensity_channel + 1, :]
    return X + sigma * bandwidth * eps


def split_indices(n_total, train_ratio=0.9, split_seed=1234):
    n_train = int(train_ratio * n_total)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=g).numpy()
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return train_idx, test_idx


def build_model(input_length, in_channels=2):
    model = Conv1Dminimal(
        in_channels=in_channels,
        input_length=input_length,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        has_variance=HAS_VARIANCE,
        ifdecode=IF_DECODE,
    )
    return model


def load_model_and_stats(ckpt_path, input_length):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = build_model(input_length=input_length, in_channels=2).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    X_mean = to_numpy(ckpt["X_mean"])
    X_std = to_numpy(ckpt["X_std"])
    return model, X_mean, X_std


@torch.no_grad()
def predict_mu_var(model, X_raw, X_mean, X_std, batch_size=512):
    Xn = (X_raw - X_mean) / X_std
    n = Xn.shape[0]

    mu_list = []
    std_list = []

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.from_numpy(Xn[s:e].astype(np.float32)).to(DEVICE)
        out = model(xb)

        mu = out["mu"].detach().cpu().numpy()
        var = out["var"].detach().cpu().numpy() if out["var"] is not None else None

        mu_list.append(mu)
        if var is None:
            std_list.append(np.full_like(mu, np.nan, dtype=np.float32))
        else:
            std_list.append(np.sqrt(np.maximum(var, 0.0)))

    y_pred = np.concatenate(mu_list, axis=0)
    y_std = np.concatenate(std_list, axis=0)
    return y_pred, y_std


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def add_error_inset(ax, errors, bins=35):
    axins = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=1.0)
    axins.hist(errors, bins=bins, color="#4c72b0", alpha=0.85, edgecolor="white")
    axins.axvline(0.0, color="r", linestyle="--", linewidth=1.0)
    axins.set_xlabel("Error (pred - true)", fontsize=6)
    axins.set_ylabel("Density", fontsize=6)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(alpha=0.15)

    mean_e = np.mean(errors)
    std_e = np.std(errors)
    mae = np.mean(np.abs(errors))
    axins.text(
        0.05,
        0.95,
        f"mean={mean_e:.3f}\nstd={std_e:.3f}\nmae={mae:.3f}",
        transform=axins.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    return axins


def plot_parity_with_error_insets(y_true, y_pred, labels, out_path):
    n_params = y_true.shape[1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

    for i in range(n_params):
        ax = axes[i]
        t = y_true[:, i]
        p = y_pred[:, i]
        err = p - t

        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        pad = 0.05 * (hi - lo + 1e-8)
        lo, hi = lo - pad, hi + pad

        ax.scatter(t, p, s=12, alpha=0.45, color="#1f77b4", linewidths=0)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(labels[i])
        ax.grid(alpha=0.25)

        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        r2 = r2_score_np(t, p)
        ax.text(
            0.04,
            0.96,
            f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        add_error_inset(ax, err)

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_raw, y = load_data(CSV_PATH)
    n, c, L = X_raw.shape
    assert c == 2, "Expected channels [dispersion, intensity]."

    train_idx, test_idx = split_indices(n, train_ratio=TRAIN_RATIO, split_seed=SPLIT_SEED)

    if EVAL_SPLIT == "train":
        idx = train_idx
    elif EVAL_SPLIT == "test":
        idx = test_idx
    else:
        idx = np.arange(n)

    X_eval = X_raw[idx].copy()
    y_eval = y[idx].copy()

    if APPLY_INPUT_NOISE:
        X_eval = add_bandwidth_noise(X_eval, sigma=NOISE_SIGMA, seed=NOISE_SEED)

    model, X_mean, X_std = load_model_and_stats(CKPT_PATH, input_length=L)
    y_pred, y_pred_std = predict_mu_var(model, X_eval, X_mean, X_std, batch_size=512)

    labels = [r"$J_1$", r"$J_1\Delta$", r"$J_{\pm\pm}$", r"$J_2$"]

    parity_path = os.path.join(OUT_DIR, "Js4_sigma0p1_pred_vs_true_inset_error.pdf")
    plot_parity_with_error_insets(y_eval, y_pred, labels, parity_path)

    print(f"Saved figure: {parity_path}")
    print(f"Eval split: {EVAL_SPLIT}, N={len(idx)}, noise_applied={APPLY_INPUT_NOISE}, sigma={NOISE_SIGMA}")


if __name__ == "__main__":
    main()